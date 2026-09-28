#!/usr/bin/env python3
"""Bridge the wild corpus into the bundle shape the atomic chain consumes.

``fit_motion_normalizer``, ``discover_kinematic_atomics`` and
``materialize_atomic_windows`` all read the ``aist_raw_performance_v1`` shape:
a ``sources.jsonl`` plus a ``sequences.jsonl`` whose ``motion_path`` and
``music_path`` are relative to the bundle root.  The wild side instead carries
absolute asset paths in a post-HMR manifest with an audio bundle beside it.
This tool materializes the former from the latter, and nothing else -- no
re-derivation, no resampling, no relabelling.

Only rows whose *audio* feature status is ``candidate`` come across.  That
status already implies a validated 151-D conversion, since the audio bundle is
built downstream of ``reconcile-wild-hmr``, and it additionally implies the
music was long enough to index frame-exactly.  Anything less is not trainable,
so it is left behind rather than carried as a hole.

Two identifiers, and they are not interchangeable.  In the schema downstream
consumes, ``recording_id`` is the sequence's own footage and must be unique per
row, while ``retrieval_group_id`` is the *exclusion* unit that may span many
rows -- on AIST, one performance's several choreographies.  The wild manifest
names things the other way around: its ``recording_id`` is the upload and its
``sequence_id`` is the clip.  So a clip becomes the recording, and the upload
becomes the retrieval group.

That mapping is what keeps the corpus honest: TikTok clips named
``<video>__clipNNN`` are cuts of one upload, so two clips of the same source
landing either side of a train/val boundary would leak a rehearsal of the same
choreography.  This tool re-derives group ownership and refuses to publish if
any upload straddles splits, rather than trusting that upstream got it right.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

SCHEMA_VERSION = "atomicdance-wild-raw-performance-v1"
MOTION_DIM = 151
MUSIC_DIM = 35
ALLOWED_SPLITS = ("train", "val", "test")


class BundleError(RuntimeError):
    pass


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_sha256(motion_sha256: str, music_sha256: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"AtomicDance wild raw performance release v1\n")
    digest.update(b"motion:")
    digest.update(motion_sha256.encode("ascii"))
    digest.update(b"\nmusic:")
    digest.update(music_sha256.encode("ascii"))
    digest.update(b"\n")
    return digest.hexdigest()


def _candidate_rows(manifest: pathlib.Path) -> List[Dict[str, object]]:
    rows = []
    for line in manifest.open(encoding="utf-8"):
        record = json.loads(line)
        if record.get("qc", {}).get("audio_feature_status") == "candidate":
            rows.append(record)
    if not rows:
        raise BundleError("no audio candidates in {}".format(manifest))
    return rows


def _check_group_ownership(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    """An upload must live in exactly one split, or the corpus leaks."""
    by_upload: Dict[str, set] = {}
    seen_clips = set()
    for record in rows:
        upload = str(record["recording_id"])          # wild naming: the upload
        clip = str(record["sequence_id"])             # wild naming: the clip
        if clip in seen_clips:
            raise BundleError("duplicate sequence_id {!r}".format(clip))
        seen_clips.add(clip)
        split = str(record.get("split"))
        if split not in ALLOWED_SPLITS:
            raise BundleError(
                "{} has unsupported split {!r}".format(clip, split))
        by_upload.setdefault(upload, set()).add(split)
    straddling = {k: sorted(v) for k, v in by_upload.items() if len(v) > 1}
    if straddling:
        sample = list(straddling.items())[:5]
        raise BundleError(
            "{} uploads straddle splits, e.g. {}".format(len(straddling), sample))
    return {"uploads": len(by_upload), "clips": len(seen_clips)}


def build(
    *,
    audio_manifest: pathlib.Path,
    output_dir: pathlib.Path,
    limit: Optional[int] = None,
) -> Dict[str, object]:
    if output_dir.exists():
        raise BundleError("{} exists; bundles publish into a new directory".format(output_dir))

    rows = _candidate_rows(audio_manifest)
    if limit is not None:
        rows = rows[:limit]
    group_report = _check_group_ownership(rows)

    staging = output_dir.with_name(output_dir.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "sequences").mkdir(parents=True)

    sequences: List[Dict[str, object]] = []
    sources: List[Dict[str, object]] = []
    split_counts: Dict[str, int] = {split: 0 for split in ALLOWED_SPLITS}
    frames_total = 0

    try:
        for record in rows:
            sequence_id = str(record["sequence_id"])
            upload_id = str(record["recording_id"])
            split = str(record["split"])
            assets = record["assets"]
            motion_source = pathlib.Path(str(assets["motion_151_raw"]))
            music_source = pathlib.Path(str(assets["music_35"]))
            frame_ids_source = pathlib.Path(str(record["timeline"]["frame_ids_path"]))

            motion = np.load(motion_source)
            music = np.load(music_source)
            frame_ids = np.asarray(np.load(frame_ids_source), dtype=np.int64).reshape(-1)
            if motion.ndim != 2 or motion.shape[1] != MOTION_DIM:
                raise BundleError("{}: motion is {}, expected [T,{}]".format(
                    sequence_id, motion.shape, MOTION_DIM))
            # A DECLARED tail trim is the one length difference allowed here.
            # extract_wild_music_features, run with --max-tail-trim-frames, cuts
            # a clip's motion back to where its audio actually ends rather than
            # quarantining the clip over one or two rounding frames; it records
            # how many frames in audio_feature.tail_trim_frames, and this is
            # where that is paid.  Anything undeclared, or declared but not
            # matching the observed difference, stays fatal -- the point of the
            # equality gate is that a silent off-by-k never reaches training.
            tail_trim = 0
            audio_feature = record.get("audio_feature")
            if isinstance(audio_feature, Mapping):
                tail_trim = int(audio_feature.get("tail_trim_frames") or 0)
            if tail_trim:
                if tail_trim < 0 or tail_trim >= len(motion):
                    raise BundleError("{}: declared tail_trim_frames {} is not a usable trim of {} motion frames".format(
                        sequence_id, tail_trim, len(motion)))
                if len(music) != len(motion) - tail_trim:
                    raise BundleError("{}: declared tail_trim_frames {} but music {} against motion {}".format(
                        sequence_id, tail_trim, music.shape, motion.shape))
                motion = motion[:-tail_trim]
                frame_ids = frame_ids[:-tail_trim]
            if music.shape != (len(motion), MUSIC_DIM):
                raise BundleError("{}: music {} does not match motion {}".format(
                    sequence_id, music.shape, motion.shape))
            if len(frame_ids) != len(motion):
                raise BundleError("{}: frame_ids {} does not match motion {}".format(
                    sequence_id, len(frame_ids), len(motion)))
            if not (np.isfinite(motion).all() and np.isfinite(music).all()):
                raise BundleError("{}: non-finite arrays".format(sequence_id))

            store = staging / "sequences" / hashlib.sha256(sequence_id.encode()).hexdigest()
            store.mkdir(parents=True)
            np.save(store / "motion_151_raw.npy", np.ascontiguousarray(motion, dtype=np.float32))
            np.save(store / "music_35.npy", np.ascontiguousarray(music, dtype=np.float32))
            np.save(store / "frame_ids.npy", frame_ids)

            relative = {name: "sequences/{}/{}".format(store.name, name) for name in
                        ("motion_151_raw.npy", "music_35.npy", "frame_ids.npy")}
            hashes = {name: sha256_file(store / name) for name in relative}

            frames = int(len(motion))
            frames_total += frames
            split_counts[split] += 1

            conversion_metadata = _read_conversion_metadata(record)
            common = {
                "duplicate_content_group_id": record.get("duplicate_content_group_id"),
                "fps": 30,
                # Downstream requires recording_id unique per row: the clip is
                # the footage, the upload is the exclusion unit.
                "recording_id": sequence_id,
                "representation": {
                    "camera_in_model_input": False,
                    "coordinate_system": "z_up_world_body_only",
                    "motion": "AtomicDance_151D",
                    "normalization": "raw",
                },
                # Every __clipNNN of one video is the same performance.
                "retrieval_group_id": upload_id,
                "schema_version": SCHEMA_VERSION,
                "split": split,
                "split_note": "inherited from the wild staging manifest; upload-level",
                "split_status": "frozen_recording_source_safe",
                "qc": {
                    "accepted_for_training": True,
                    "input_manifest_validation": "passed",
                    "status": "wild_post_hmr_audio_candidate",
                },
            }
            # What the music was computed from, carried through rather than
            # left behind in the audio bundle.  Until 2026-08-25 the bundle
            # recorded only ``music_sha256`` -- the hash of the derived array --
            # so "is this row's music the music of this row's clip?" had no
            # answer inside the bundle at all: the audio bytes appeared nowhere.
            # That is why the corpus could carry 234 evaluation rows (14.9%)
            # whose music was another generation's while every hash in the
            # bundle agreed with every other, and why ``audit_clip_freshness``
            # could gate 3D and S3D but not music.  The 3D and S3D stages each
            # write the hash of what they read beside their own output; this is
            # the same discipline for the one stage that could not.
            audio_feature = record.get("audio_feature") or {}
            source_audio_sha256 = audio_feature.get("source_audio_sha256")
            sequences.append({
                **common,
                "assets": {
                    "frame_ids": relative["frame_ids.npy"],
                    "frame_ids_sha256": hashes["frame_ids.npy"],
                    "motion_151_raw": relative["motion_151_raw.npy"],
                    "motion_151_raw_sha256": hashes["motion_151_raw.npy"],
                    "music_35": relative["music_35.npy"],
                    "music_35_sha256": hashes["music_35.npy"],
                },
                "source_audio_sha256": source_audio_sha256,
                "camera_tracking": conversion_metadata,
                "frame_count": frames,
                "frame_ids_path": relative["frame_ids.npy"],
                "frame_ids_sha256": hashes["frame_ids.npy"],
                "is_contiguous": bool(np.array_equal(frame_ids, np.arange(frame_ids[0], frame_ids[0] + frames))),
                "motion_path": relative["motion_151_raw.npy"],
                "motion_sha256": hashes["motion_151_raw.npy"],
                "music_path": relative["music_35.npy"],
                "music_sha256": hashes["music_35.npy"],
                "person_track_id": record.get("person_track_id"),
                "preprocess_version": SCHEMA_VERSION,
                "sequence_id": sequence_id,
                "source_end_frame_exclusive": int(frame_ids[-1]) + 1,
                "source_start_frame": int(frame_ids[0]),
            })
            sources.append({
                **common,
                "audio_id": None,
                # Identity of the (motion, music) pair itself, so two rows that
                # carry the same content are recognisable as such regardless of
                # how they were named.  Domain-separated per corpus so a wild
                # row can never collide with an AIST one.
                "content_sha256": _content_sha256(
                    hashes["motion_151_raw.npy"], hashes["music_35.npy"]),
                "dancer_id": None,
                "legacy_source_name": record.get("legacy_clip_id"),
                "motion_sha256": hashes["motion_151_raw.npy"],
                "music_sha256": hashes["music_35.npy"],
                "provenance": {
                    "audio_bundle_manifest": str(audio_manifest.resolve()),
                    "camera_tracking": conversion_metadata,
                    "source_video": str(record["assets"].get("source_video")),
                    # The clip audio these 35-D features were extracted from --
                    # the join a music staleness check needs and the bundle did
                    # not carry.  ``source_audio`` names the file, the hash says
                    # which generation of it.
                    "source_audio": audio_feature.get("source_audio"),
                    "source_audio_sha256": source_audio_sha256,
                },
                "raw_uri": "wild3d://{}".format(sequence_id),
                "source_kind": "tiktok_wild",
                "source_variant": "gvhmr_world_smplx_to_151d",
            })

        _write_jsonl(staging / "sequences.jsonl", sequences)
        _write_jsonl(staging / "sources.jsonl", sources)
        report = {
            "counts": {
                "sequences": len(sequences),
                "frames": frames_total,
                "minutes": round(frames_total / 30.0 / 60.0, 2),
                "by_split": split_counts,
                **group_report,
            },
            "input": {
                "audio_manifest": str(audio_manifest.resolve()),
                "audio_manifest_sha256": sha256_file(audio_manifest),
            },
            "manifests": {
                "sequences.jsonl": sha256_file(staging / "sequences.jsonl"),
                "sources.jsonl": sha256_file(staging / "sources.jsonl"),
            },
            "publication": "immutable_new_directory_only_atomic_rename",
            "retrieval_exclusion_unit": "retrieval_group_id = the source upload; recording_id = one clip",
            "schema_version": SCHEMA_VERSION,
            "selection": "audio_feature_status == candidate (implies validated 151-D and frame-exact music)",
        }
        _write_json(staging / "report.json", report)
        os.rename(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


def _read_conversion_metadata(record: Mapping[str, object]) -> Dict[str, object]:
    """Carry the tracker's identity forward; a corpus mixing trackers must say so."""
    path = record.get("assets", {}).get("conversion_metadata")
    if not path:
        return {}
    try:
        metadata = json.loads(pathlib.Path(str(path)).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    extract = metadata.get("extract_meta", {})
    carried = {
        "visual_odometry": extract.get("visual_odometry"),
        "static_cam": extract.get("static_cam"),
        "visual_odometry_attempts_allowed": extract.get("visual_odometry_attempts_allowed"),
    }
    if "static_cam_justification" in extract:
        carried["static_cam_justification"] = extract["static_cam_justification"]
    return {key: value for key, value in carried.items() if value is not None}


def _write_jsonl(path: pathlib.Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_json(path: pathlib.Path, payload: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--audio-manifest", type=pathlib.Path, required=True,
        help="sequences_audio.jsonl from extract_wild_music_features.py")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build(
            audio_manifest=args.audio_manifest,
            output_dir=args.output_dir,
            limit=args.limit,
        )
    except (BundleError, FileNotFoundError) as error:
        raise SystemExit("error: {}".format(error))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
