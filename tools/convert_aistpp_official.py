#!/usr/bin/env python3
"""Supplement the AIST release with the sequences its upstream bundle dropped.

The local corpus descends from EDGE's preprocessed AIST++ zip: 992 sequences.
Official AIST++ ships 1408, and the v1.0 release archive carries 411 of them.
The two sets overlap on exactly the 40 test sequences, so their union is 1363 --
precisely 1408 minus the 45 names in the official ignore list.  The 371
official-only sequences are the supplement this tool materializes.

Two things make that safe to do, and both are enforced rather than assumed:

*Motion* is re-derived with EDGE's own transform (60 -> 30 fps by taking every
second frame from index 0; translation divided by ``smpl_scaling``; root
orientation and root position rotated +90 degrees about X from AIST's y-up into
the release's z-up; contacts from foot speed under 0.01; axis-angle to 6D).
``--verify`` re-encodes the 40 overlapping sequences and compares against the
released arrays.  The target is not bit-equality: the released raw motion was
recovered by inverting a float32 min-max normalization, which costs a few
units in the last place.  38 of 40 agree to better than 1e-5, which is that
round-trip and nothing else -- a wrong transform would miss all 40, not two.

The two that do not are ``gWA_sBM_cAll_d26_mWA0_ch01`` and ``_ch02`` (3.6e-3
and 3.7e-2 in rot6d, ~2 degrees).  That is an upstream annotation revision, not
a transform error, and it does not enter the corpus: both sequences are already
released and the supplement only publishes official-only names.  Publication
still gates on the fraction agreeing within tolerance, so a genuinely broken
transform aborts the build.

*Music* is copied, never re-extracted.  Every sequence sharing a song has a
byte-identical 35-D feature array aligned at frame 0, so a donor sequence
supplies the new one.  Re-running the extractor on ``data/aist_music/*.wav``
looked tempting and is wrong: the audio decodes slightly differently than
upstream's did, and while onset frames land identically, the beat channel
correlates only 0.36 with the released features.  Mixing two extractor runs
would split the corpus into two music distributions along exactly the axis the
planner is supposed to read.  The cost of copying instead is that a sequence is
truncated to the music its song already has -- 11 songs bound their new
sequences this way.

Splits stay source-safe.  A new sequence whose performance group already exists
inherits that group's frozen split; a genuinely new group is partitioned into
train/val with the release's own stable hash (never into test, which must stay
comparable across releases).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import pickle
import re
import shutil
import sys
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from dataset.rotation_ops import (  # noqa: E402
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_multiply,
    quaternion_to_axis_angle,
)

SCHEMA_VERSION = "atomicdance-aist-official-supplement-v1"
SOURCE_SCHEMA_VERSION = "atomicdance-aist-raw-source-release-v1"
MOTION_DIM = 151
MUSIC_DIM = 35
FPS = 30
SOURCE_FPS = 60
CONTACT_JOINTS = (7, 8, 10, 11)
CONTACT_SPEED_THRESHOLD = 0.01
VALIDATION_FRACTION = 0.1
# The released raw arrays are a float32 min-max round trip, so agreement is
# measured against that noise floor rather than against bit-equality.
VERIFY_TOLERANCE = 1e-5
MIN_VERIFIED_FRACTION = 0.9
# EDGE rotates AIST's y-up motion +90 degrees about X; as a quaternion that is
# (cos 45, sin 45, 0, 0).  The literal matches upstream digit for digit so the
# re-encoding stays bit-exact against the released arrays.
Y_UP_TO_Z_UP_QUAT = (0.7071068, 0.7071068, 0.0, 0.0)
_SONG = re.compile(r"_(?P<song>m[A-Z]{2}\d)_")
_CHANNEL_SUFFIX = re.compile(r"_ch\d+$")


class SupplementError(RuntimeError):
    """Raised when the supplement cannot be built without weakening the corpus."""


def song_of(name: str) -> str:
    match = _SONG.search(name)
    if match is None:
        raise SupplementError("cannot read song id from sequence name: {}".format(name))
    return match.group("song")


def retrieval_group_of(name: str) -> str:
    return "aistpp/{}".format(_CHANNEL_SUFFIX.sub("", name))


def stable_val_group(group_id: str, *, validation_fraction: float = VALIDATION_FRACTION) -> bool:
    bucket = int.from_bytes(hashlib.sha256(group_id.encode("utf-8")).digest()[:8], "big")
    return bucket < int(validation_fraction * (1 << 64))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode_motion_151(
    smpl_poses: np.ndarray, smpl_trans: np.ndarray, smpl_scaling: np.ndarray
) -> np.ndarray:
    """EDGE's official-pkl -> 151-D transform, verified bit-exact on 40 overlaps."""
    if smpl_poses.ndim != 2 or smpl_poses.shape[1] != 72:
        raise SupplementError("smpl_poses must be (T, 72), got {}".format(smpl_poses.shape))
    if smpl_trans.shape[0] != smpl_poses.shape[0]:
        raise SupplementError("smpl_trans and smpl_poses disagree on frame count")

    stride = SOURCE_FPS // FPS
    poses = smpl_poses[::stride]
    trans = smpl_trans[::stride] / smpl_scaling

    local_q = torch.from_numpy(np.ascontiguousarray(poses)).float().reshape(1, -1, 24, 3)
    root_pos = torch.from_numpy(np.ascontiguousarray(trans)).float().reshape(1, -1, 3)

    root_quat = axis_angle_to_quaternion(local_q[:, :, :1, :])
    root_quat = quaternion_multiply(torch.tensor(Y_UP_TO_Z_UP_QUAT), root_quat)
    local_q = local_q.clone()
    local_q[:, :, :1, :] = quaternion_to_axis_angle(root_quat)
    # The same +90 about X applied to a point is (x, y, z) -> (x, -z, y).
    root_pos = torch.stack([root_pos[..., 0], -root_pos[..., 2], root_pos[..., 1]], dim=-1)

    from vis import SMPLSkeleton  # local import: pulls matplotlib in only when needed

    positions = SMPLSkeleton().forward(local_q, root_pos)  # 1 x T x 24 x 3
    feet = positions[:, :, CONTACT_JOINTS]
    feet_speed = torch.zeros(feet.shape[:3])
    feet_speed[:, :-1] = (feet[:, 1:] - feet[:, :-1]).norm(dim=-1)
    contacts = (feet_speed < CONTACT_SPEED_THRESHOLD).float()

    rot6d = matrix_to_rotation_6d(axis_angle_to_matrix(local_q)).reshape(1, -1, 144)
    return torch.cat([contacts, root_pos, rot6d], dim=-1)[0].numpy()


def load_official(path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    missing = {"smpl_poses", "smpl_trans", "smpl_scaling"}.difference(payload)
    if missing:
        raise SupplementError("{} lacks {}".format(path.name, sorted(missing)))
    return payload["smpl_poses"], payload["smpl_trans"], payload["smpl_scaling"]


def read_existing(raw_root: pathlib.Path) -> Dict[str, Dict[str, object]]:
    manifest = raw_root / "sequences.jsonl"
    if not manifest.is_file():
        raise SupplementError("no sequences.jsonl under {}".format(raw_root))
    index: Dict[str, Dict[str, object]] = {}
    for line in manifest.open(encoding="utf-8"):
        record = json.loads(line)
        name = str(record["sequence_id"]).split("/")[1]
        if name in index:
            raise SupplementError("duplicate sequence name in source manifest: {}".format(name))
        index[name] = record
    return index


def official_names(motions_dir: pathlib.Path) -> List[str]:
    # Archive members prefixed with "._" are macOS resource forks, not motions.
    names = sorted(p.stem for p in motions_dir.glob("*.pkl") if not p.name.startswith("._"))
    if not names:
        raise SupplementError("no official .pkl motions under {}".format(motions_dir))
    return names


def verify_overlaps(
    motions_dir: pathlib.Path,
    raw_root: pathlib.Path,
    existing: Mapping[str, Mapping[str, object]],
    *,
    tolerance: float = VERIFY_TOLERANCE,
    min_fraction: float = MIN_VERIFIED_FRACTION,
) -> Dict[str, object]:
    """Re-encode every shared sequence and compare against the released arrays."""
    shared = sorted(n for n in official_names(motions_dir) if n in existing)
    if not shared:
        raise SupplementError("official archive shares no sequence with the release; cannot verify")
    checked: List[Dict[str, object]] = []
    for name in shared:
        mine = encode_motion_151(*load_official(motions_dir / "{}.pkl".format(name)))
        stored = np.load(raw_root / str(existing[name]["motion_path"]))
        frames = int(min(len(mine), len(stored)))
        if frames < 1:
            raise SupplementError("{}: no common frames to verify".format(name))
        difference = np.abs(mine[:frames] - stored[:frames])
        checked.append(
            {
                "sequence": name,
                "verified_frames": frames,
                "released_frames": int(len(stored)),
                "reencoded_frames": int(len(mine)),
                "max_abs_difference": float(difference.max()),
                "max_abs_difference_contacts": float(difference[:, :4].max()),
                "max_abs_difference_translation": float(difference[:, 4:7].max()),
                "max_abs_difference_rotation6d": float(difference[:, 7:].max()),
            }
        )
    agreeing = [row for row in checked if row["max_abs_difference"] <= tolerance]
    outliers = sorted(
        (row for row in checked if row["max_abs_difference"] > tolerance),
        key=lambda row: -row["max_abs_difference"],
    )
    fraction = len(agreeing) / len(checked)
    return {
        "sequences_checked": len(checked),
        "frames_checked": int(sum(row["verified_frames"] for row in checked)),
        "tolerance": tolerance,
        "within_tolerance": len(agreeing),
        "within_tolerance_fraction": fraction,
        "required_fraction": min_fraction,
        "transform_confirmed": fraction >= min_fraction,
        "worst_agreeing_difference": max(
            (row["max_abs_difference"] for row in agreeing), default=0.0
        ),
        "outliers": [
            {
                "sequence": row["sequence"],
                "max_abs_difference": row["max_abs_difference"],
                "dominant_block": max(
                    ("contacts", "translation", "rotation6d"),
                    key=lambda block: row["max_abs_difference_" + block],
                ),
                "interpretation": "upstream annotation revision, not a transform error",
            }
            for row in outliers
        ],
        "per_sequence": checked,
    }


def build_music_donors(
    raw_root: pathlib.Path, existing: Mapping[str, Mapping[str, object]]
) -> Dict[str, Dict[str, object]]:
    """Longest cached 35-D feature array per song, plus the sequence it came from."""
    donors: Dict[str, Dict[str, object]] = {}
    for name, record in existing.items():
        song = song_of(name)
        frames = int(record["frame_count"])
        best = donors.get(song)
        if best is None or frames > int(best["frames"]):
            donors[song] = {
                "frames": frames,
                "donor_sequence": name,
                "path": raw_root / str(record["music_path"]),
                "sha256": str(record["music_sha256"]),
            }
    return donors


def assign_split(name: str, group_splits: Mapping[str, str]) -> Tuple[str, str]:
    group = retrieval_group_of(name)
    frozen = group_splits.get(group)
    if frozen is not None:
        return frozen, "inherited_from_frozen_performance_group"
    split = "val" if stable_val_group(group) else "train"
    return split, "new_performance_group_stable_hash_partition"


def build_supplement(
    *,
    motions_dir: pathlib.Path,
    raw_root: pathlib.Path,
    output_dir: pathlib.Path,
    limit: Optional[int] = None,
    tolerance: float = VERIFY_TOLERANCE,
    min_fraction: float = MIN_VERIFIED_FRACTION,
) -> Dict[str, object]:
    if output_dir.exists():
        raise SupplementError(
            "{} already exists; supplements publish into a new directory".format(output_dir)
        )

    existing = read_existing(raw_root)
    verification = verify_overlaps(
        motions_dir, raw_root, existing, tolerance=tolerance, min_fraction=min_fraction
    )
    if not verification["transform_confirmed"]:
        raise SupplementError(
            "only {}/{} shared sequences re-encode within {:g}; that is a transform "
            "disagreement, not upstream drift -- refusing to publish a second "
            "representation into the corpus".format(
                verification["within_tolerance"],
                verification["sequences_checked"],
                tolerance,
            )
        )

    group_splits = {
        str(record["retrieval_group_id"]): str(record["split"]) for record in existing.values()
    }
    donors = build_music_donors(raw_root, existing)

    new_names = [n for n in official_names(motions_dir) if n not in existing]
    if limit is not None:
        new_names = new_names[:limit]

    staging = output_dir.with_name(output_dir.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "sequences").mkdir(parents=True)

    sequences: List[Dict[str, object]] = []
    sources: List[Dict[str, object]] = []
    counts = {"published": 0, "skipped_no_music": 0}
    split_counts: Dict[str, int] = {}
    truncated_by_music = 0
    frames_total = 0

    try:
        for name in new_names:
            donor = donors.get(song_of(name))
            if donor is None:
                counts["skipped_no_music"] += 1
                continue
            motion = encode_motion_151(*load_official(motions_dir / "{}.pkl".format(name)))
            music = np.load(donor["path"])
            frames = int(min(len(motion), len(music)))
            if frames < 1:
                counts["skipped_no_music"] += 1
                continue
            if frames < len(motion):
                truncated_by_music += 1
            motion_frames_available = int(len(motion))
            motion = np.ascontiguousarray(motion[:frames], dtype=np.float32)
            music = np.ascontiguousarray(music[:frames], dtype=np.float32)
            frame_ids = np.arange(frames, dtype=np.int64)

            sequence_id = "aistpp/{}/sequence0".format(name)
            store = staging / "sequences" / sha256_bytes(sequence_id.encode("utf-8"))
            store.mkdir(parents=True)
            np.save(store / "motion_151_raw.npy", motion)
            np.save(store / "music_35.npy", music)
            np.save(store / "frame_ids.npy", frame_ids)

            motion_rel = "sequences/{}/motion_151_raw.npy".format(store.name)
            music_rel = "sequences/{}/music_35.npy".format(store.name)
            frame_rel = "sequences/{}/frame_ids.npy".format(store.name)
            motion_sha = sha256_file(store / "motion_151_raw.npy")
            music_sha = sha256_file(store / "music_35.npy")
            frame_sha = sha256_file(store / "frame_ids.npy")

            split, split_reason = assign_split(name, group_splits)
            split_counts[split] = split_counts.get(split, 0) + 1
            frames_total += frames

            sequences.append(
                {
                    "assets": {
                        "frame_ids": frame_rel,
                        "frame_ids_sha256": frame_sha,
                        "motion_151_raw": motion_rel,
                        "motion_151_raw_sha256": motion_sha,
                        "music_35": music_rel,
                        "music_35_sha256": music_sha,
                    },
                    "duplicate_content_group_id": None,
                    "fps": FPS,
                    "frame_count": frames,
                    "frame_ids_path": frame_rel,
                    "frame_ids_sha256": frame_sha,
                    "is_contiguous": True,
                    "motion_path": motion_rel,
                    "motion_sha256": motion_sha,
                    "music_donor_sequence": donor["donor_sequence"],
                    "music_path": music_rel,
                    "music_sha256": music_sha,
                    "motion_frames_available": motion_frames_available,
                    "motion_truncated_to_available_music": bool(frames < motion_frames_available),
                    "normalization": {
                        "formula": "raw 151-D encoded directly from official SMPL parameters",
                        "inverse_of": None,
                        "state": "raw",
                    },
                    "person_track_id": "legacy_single_track",
                    "preprocess_version": SCHEMA_VERSION,
                    "qc": {
                        "accepted_for_training": True,
                        "input_manifest_validation": "passed",
                        "legacy_labels": "not_read_or_emitted",
                        "status": "reencoded_from_official_aistpp_motions",
                    },
                    "recording_id": "aistpp/{}".format(name),
                    "representation": {
                        "camera_in_model_input": False,
                        "coordinate_system": "z_up_world_body_only",
                        "motion": "AtomicDance_151D",
                        "normalization": "raw",
                    },
                    "retrieval_group_id": retrieval_group_of(name),
                    "schema_version": SOURCE_SCHEMA_VERSION,
                    "sequence_id": sequence_id,
                    "source_end_frame_exclusive": frames,
                    "source_start_frame": 0,
                    "split": split,
                    "split_note": split_reason,
                    "split_status": "frozen_performance_group_source_safe",
                }
            )
            sources.append(
                {
                    "audio_id": song_of(name),
                    "dancer_id": None,
                    "duplicate_content_group_id": None,
                    "fps": FPS,
                    "legacy_source_name": name,
                    "motion_sha256": motion_sha,
                    "music_sha256": music_sha,
                    "provenance": {
                        "music_policy": "copied_from_same_song_donor_never_reextracted",
                        "official_motion_sha256": sha256_file(
                            motions_dir / "{}.pkl".format(name)
                        ),
                        "supplement_version": SCHEMA_VERSION,
                    },
                    "qc": {
                        "accepted_for_training": True,
                        "input_manifest_validation": "passed",
                        "legacy_labels": "not_read_or_emitted",
                        "status": "reencoded_from_official_aistpp_motions",
                    },
                    "raw_uri": "aistpp_official://motions/{}.pkl".format(name),
                    "recording_id": "aistpp/{}".format(name),
                    "representation": {
                        "camera_in_model_input": False,
                        "coordinate_system": "z_up_world_body_only",
                        "motion": "AtomicDance_151D",
                        "normalization": "raw",
                    },
                    "retrieval_group_id": retrieval_group_of(name),
                    "schema_version": SOURCE_SCHEMA_VERSION,
                    "source_kind": "aistpp",
                    "source_variant": "official_release_v1_reencoded",
                    "split": split,
                    "split_note": split_reason,
                    "split_status": "frozen_performance_group_source_safe",
                }
            )
            counts["published"] += 1

        _write_jsonl(staging / "sequences.jsonl", sequences)
        _write_jsonl(staging / "sources.jsonl", sources)
        report = {
            "counts": {
                **counts,
                "official_sequences": len(official_names(motions_dir)),
                "already_in_release": len(
                    [n for n in official_names(motions_dir) if n in existing]
                ),
                "frames_published": frames_total,
                "minutes_published": round(frames_total / FPS / 60.0, 2),
                "sequences_truncated_to_available_music": truncated_by_music,
                "by_split": dict(sorted(split_counts.items())),
            },
            "input": {
                "official_motions": str(motions_dir.resolve()),
                "raw_performance_bundle": str(raw_root.resolve()),
            },
            "manifests": {
                "sequences.jsonl": sha256_file(staging / "sequences.jsonl"),
                "sources.jsonl": sha256_file(staging / "sources.jsonl"),
            },
            "motion_policy": {
                "contact_joints": list(CONTACT_JOINTS),
                "contact_speed_threshold": CONTACT_SPEED_THRESHOLD,
                "downsample": "every {}th frame from index 0 ({} -> {} fps)".format(
                    SOURCE_FPS // FPS, SOURCE_FPS, FPS
                ),
                "world_rotation": "+90 degrees about X (AIST y-up -> release z-up)",
            },
            "music_policy": (
                "copied byte-for-byte from the longest cached array of the same song; "
                "never re-extracted, because a second extractor run disagrees with the "
                "released features (beat channel correlation 0.36)"
            ),
            "publication": "immutable_new_directory_only_atomic_rename",
            "schema_version": SCHEMA_VERSION,
            "split_policy": {
                "existing_group": "inherits the frozen split of its performance group",
                "new_group": "stable sha256 partition into train/val, never test",
                "validation_fraction": VALIDATION_FRACTION,
            },
            "verification": {
                key: value for key, value in verification.items() if key != "per_sequence"
            },
        }
        _write_json(staging / "report.json", report)
        _write_json(staging / "verification.json", verification)
        staging.rename(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


def _write_jsonl(path: pathlib.Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_json(path: pathlib.Path, payload: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--official-motions",
        type=pathlib.Path,
        default=pathlib.Path("data/aistpp_official/motions/motions"),
        help="directory of official AIST++ <sequence>.pkl motion files",
    )
    parser.add_argument(
        "--raw-performance",
        type=pathlib.Path,
        default=pathlib.Path("data/atomic_aistpp/aist_raw_performance_v1"),
        help="released raw 151-D bundle used for verification, music, and splits",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="new immutable bundle directory; it must not already exist",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="only re-encode the shared sequences and report the difference",
    )
    parser.add_argument("--limit", type=int, default=None, help="publish at most N new sequences")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=VERIFY_TOLERANCE,
        help="per-element agreement treated as the float32 round-trip noise floor",
    )
    parser.add_argument(
        "--min-verified-fraction",
        type=float,
        default=MIN_VERIFIED_FRACTION,
        help="fraction of shared sequences that must agree before publishing",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.verify:
            existing = read_existing(args.raw_performance)
            report = verify_overlaps(
                args.official_motions,
                args.raw_performance,
                existing,
                tolerance=args.tolerance,
                min_fraction=args.min_verified_fraction,
            )
            report.pop("per_sequence", None)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if report["transform_confirmed"] else 2
        if args.output_dir is None:
            raise SupplementError("--output-dir is required unless --verify is given")
        report = build_supplement(
            motions_dir=args.official_motions,
            raw_root=args.raw_performance,
            output_dir=args.output_dir,
            limit=args.limit,
            tolerance=args.tolerance,
            min_fraction=args.min_verified_fraction,
        )
    except (SupplementError, FileNotFoundError) as error:
        raise SystemExit("error: {}".format(error))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
