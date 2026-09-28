#!/usr/bin/env python3
"""Export a wild split into the two directories ``extract_aist_features`` reads.

M6's FID and Div compare two *distributions*: ``eval/evaluate.py`` takes
``--prediction-features`` and ``--ground-truth-features`` and cannot run with
one of them.  On AIST that ground-truth side is ``runs/m6_gt_features`` -- 411
sequences, four feature families, extracted once and reused by every checkpoint.
The wild corpus has no equivalent, so a wild completion checkpoint has nothing
to be scored against, and the AIST set is not a substitute: the pipeline plan's
first standing caveat is that wild FID may only be read against a *wild* ground
truth and never mixed with AIST figures.

This tool builds the input for that set.  Two properties are the whole point:

**The ground truth goes through the same forward kinematics as the generated
motion.**  ``infer_atomic.decode_motion`` turns a 151-D tensor into
``full_pose`` with ``SMPLSkeleton().forward(rotations, root_positions)``, and
``motion_151_to_joints`` is that same call.  So both sides of the FID reach
``load_keypoints`` as a ``full_pose`` pickle and take its identical z-up -> y-up
branch.  Had the ground truth been exported through any other joint convention,
FID would have measured the convention.

**Music is copied, never re-extracted.**  The 35-D array in the bundle is what
the planner was trained on and what it is conditioned by at inference.  BAS
needs music beats, and re-deriving them from the ingest tree's audio would put
the beat channel on a second extractor run: measured on the one corpus where
both exist, onset frames land identically and the beat channel correlates
**0.36** (``tools/convert_aistpp_official.py``).  So the audio directory here
holds ``.npy``, which ``eval/extract_aist_features`` reads through
``beat_channel_from_features``.

Three things are checked rather than assumed, because each has already cost
this repo something:

* every array's sha256 is verified against the manifest that names it.  A tree
  whose files exist and are empty is this repo's canonical silent failure
  (CLAUDE.md 1) and a 0-byte ``.npy`` would surface as a shape error hundreds of
  sequences later, if at all;
* motion and music must agree on frame count.  They are frame-aligned by
  construction -- the audio was extracted against the converted 3D's
  ``frame_ids`` -- so a disagreement means the two came from different runs;
* a split naming no rows is refused instead of producing an empty directory
  that the feature extractor would happily report as zero sequences.

Usage::

    python3 tools/export_wild_eval_motion.py \\
        --bundle /dev/shm/atomicdance-acct/performance --split test \\
        --output-dir runs/wild_v4_acct_gt_eval

    python3 eval/extract_aist_features.py \\
        --motion-dir runs/wild_v4_acct_gt_eval/motion \\
        --audio-dir  runs/wild_v4_acct_gt_eval/audio \\
        --output     runs/wild_v4_acct_gt_features --workers 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import pickle
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.convert_motion_to_guofeats import motion_151_to_joints  # noqa: E402

MOTION_DIM = 151
MUSIC_DIM = 35


class ExportError(RuntimeError):
    pass


def read_jsonl(path: pathlib.Path) -> List[Dict[str, object]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_checked(bundle: pathlib.Path, relative: str, digest: Optional[str],
                 name: str, kind: str) -> np.ndarray:
    """Load an array named by the manifest and confirm it is the named bytes."""
    path = bundle / relative
    if not path.is_file():
        raise ExportError("{}: {} array missing at {}".format(name, kind, path))
    payload = path.read_bytes()
    if not payload:
        raise ExportError("{}: {} array is 0 bytes at {}".format(name, kind, path))
    if digest:
        seen = hashlib.sha256(payload).hexdigest()
        if seen != digest:
            raise ExportError("{}: {} array is not the bytes the manifest names "
                              "({} != {})".format(name, kind, seen, digest))
    import io

    return np.load(io.BytesIO(payload))


def cross_check_training_bundle(rows, training_bundle, policy="refuse"):
    """Do these ground-truth sequences hold the same bytes the models trained on?

    Everything already checked in this file is checked against the eval bundle's
    *own* manifest, and that is the trap: a bundle built from a stale generation
    of the corpus is perfectly self-consistent.  Its hashes verify, its motion
    and music agree on frame count, and nothing in a run says the models were
    trained on different bytes for the same ``recording_id``.

    Measured 2026-08-24 by joining the two bundles on ``recording_id``:
    13,450 shared sequences, of which 13,133 (97.6%) hold identical motion
    bytes, **116 (0.9%) hold different bytes at the same frame count** and 201
    (1.5%) differ in length as well; 933 (6.9%) differ in music.  333 sequences
    exist only in the eval line.  On the clean5b5 corpus the two lines differ
    for 317 of 1,999 recordings (15.9%), and on the 65 clips M6 actually scores,
    for 24 (36.9%) -- of which a frame-count comparison finds only 16.

    That last number is why this compares hashes and not lengths.  The repo has
    paid for it once already: the R0 re-ingest round called 3,929 clips stale by
    frame count and 3,991 by content hash, and the 62 in between were the ones
    whose old cut happened to come out the same length.
    """
    if policy not in ("refuse", "warn", "off"):
        raise ValueError("training cross-check policy must be refuse/warn/off, "
                         "got {!r}".format(policy))
    report = {"policy": policy, "training_bundle": str(training_bundle or ""),
              "checked": False, "match": 0, "motion_differs": 0,
              "music_differs": 0, "absent_in_training": 0,
              "motion_differs_at_equal_length": 0, "differing": []}
    if policy == "off" or training_bundle is None:
        return report
    training = {str(row.get("recording_id")): row
                for row in read_jsonl(pathlib.Path(training_bundle) / "sequences.jsonl")}
    report["checked"] = True
    report["training_sequences"] = len(training)
    for row in rows:
        name = str(row.get("recording_id"))
        other = training.get(name)
        if other is None:
            report["absent_in_training"] += 1
            report["differing"].append({"name": name, "reason": "absent_in_training"})
            continue
        motion_differs = row.get("motion_sha256") != other.get("motion_sha256")
        music_differs = row.get("music_sha256") != other.get("music_sha256")
        if not motion_differs and not music_differs:
            report["match"] += 1
            continue
        if motion_differs:
            report["motion_differs"] += 1
            if row.get("frame_count") == other.get("frame_count"):
                report["motion_differs_at_equal_length"] += 1
        if music_differs:
            report["music_differs"] += 1
        report["differing"].append({
            "name": name,
            "reason": "motion" if motion_differs and not music_differs else
                      "music" if music_differs and not motion_differs else "motion+music",
            "eval_frames": row.get("frame_count"),
            "training_frames": other.get("frame_count"),
        })
    bad = len(report["differing"])
    if bad:
        head = ", ".join("{} ({}, {} vs {} frames)".format(
            d["name"], d["reason"], d.get("eval_frames"), d.get("training_frames"))
            for d in report["differing"][:5])
        message = ("{} of {} sequence(s) do not hold the bytes the training "
                   "bundle names for the same recording_id -- {} differ in "
                   "motion ({} of them at the SAME frame count, which a length "
                   "comparison cannot see), {} in music, {} absent there "
                   "entirely: {}{}. Scoring against these measures the "
                   "difference between two generations of the corpus."
                   .format(bad, len(rows), report["motion_differs"],
                           report["motion_differs_at_equal_length"],
                           report["music_differs"], report["absent_in_training"],
                           head, " ..." if bad > 5 else ""))
        if policy == "refuse":
            raise ExportError(message)
        print("WARNING: " + message)
    return report


def export(bundle: pathlib.Path, split: str, output_dir: pathlib.Path,
           limit: Optional[int] = None, training_bundle: Optional[pathlib.Path] = None,
           training_check: str = "refuse") -> Dict[str, object]:
    sequences = read_jsonl(bundle / "sequences.jsonl")
    rows = [row for row in sequences if str(row.get("split")) == split]
    if not rows:
        raise ExportError("{} names no sequence in split {!r}; an empty export would "
                          "be reported downstream as zero ground-truth sequences "
                          "rather than as a missing input".format(bundle, split))
    rows.sort(key=lambda row: str(row.get("recording_id")))
    if limit is not None:
        rows = rows[:limit]

    # Before a single array is written: an export that has already produced its
    # directories is one somebody will use.
    training_report = cross_check_training_bundle(rows, training_bundle, training_check)

    motion_dir = output_dir / "motion"
    audio_dir = output_dir / "audio"
    motion_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    names, frames = [], []
    for index, row in enumerate(rows):
        name = str(row["recording_id"])
        motion = load_checked(bundle, str(row["motion_path"]),
                              row.get("motion_sha256"), name, "motion")
        music = load_checked(bundle, str(row["music_path"]),
                             row.get("music_sha256"), name, "music")
        if motion.ndim != 2 or motion.shape[1] != MOTION_DIM:
            raise ExportError("{}: motion is {}, expected [T,{}]".format(
                name, motion.shape, MOTION_DIM))
        if music.ndim != 2 or music.shape[1] != MUSIC_DIM:
            raise ExportError("{}: music is {}, expected [T,{}]".format(
                name, music.shape, MUSIC_DIM))
        # Frame-aligned by construction: the 35-D features were extracted
        # against the converted 3D's own frame_ids.  A mismatch therefore means
        # the two arrays came from different runs of the pipeline, which is
        # exactly the split-the-corpus failure this file exists to avoid.
        if len(motion) != len(music):
            raise ExportError("{}: {} motion frames against {} music frames; these "
                              "are frame-aligned by construction, so a disagreement "
                              "means two different runs".format(
                                  name, len(motion), len(music)))

        joints = motion_151_to_joints(motion)
        with open(str(motion_dir / (name + ".pkl")), "wb") as handle:
            # ``full_pose`` and nothing else: it is the key ``load_keypoints``
            # takes first, and it is the key ``infer_atomic`` writes, so ground
            # truth and generated motion take one branch rather than two.
            pickle.dump({"full_pose": joints.astype(np.float32)}, handle)
        np.save(str(audio_dir / (name + ".npy")), music.astype(np.float32))
        names.append(name)
        frames.append(int(len(motion)))
        if (index + 1) % 200 == 0:
            print("  exported {}/{}".format(index + 1, len(rows)), flush=True)

    manifest = {
        "schema_version": "atomicdance-wild-eval-motion-v1",
        "bundle": str(bundle),
        "split": split,
        "sequences": len(names),
        "names": names,
        # Whether these are the bytes the models trained on, per recording_id.
        # In the manifest rather than only on stdout, so an export made with the
        # cross-check relaxed is identifiable from its own record.
        "training_bundle_cross_check": training_report,
        "frames": {"total": int(sum(frames)),
                   "median": float(np.median(frames)),
                   "min": int(min(frames)), "max": int(max(frames))},
        "motion_representation": (
            "full_pose [T,24,3], z-up, from SMPLSkeleton forward kinematics -- the "
            "same call infer_atomic.decode_motion makes, so generated and ground "
            "truth reach the feature extractor by one path"),
        "music_representation": (
            "the bundle's own 35-D array, copied not re-extracted; re-deriving beats "
            "from audio would put BAS on a second extractor run (beat channel "
            "correlates 0.36, see tools/convert_aistpp_official.py)"),
        "not_comparable_to": (
            "AIST FID/Div figures -- a wild distribution may only be read against a "
            "wild ground truth (WILD_ATOMIC_PIPELINE_PLAN.md 5.1)"),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=pathlib.Path, required=True,
                        help="wild performance bundle (sequences.jsonl + sequences/)")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--training-bundle", type=pathlib.Path, default=None,
                        help="the bundle the models were TRAINED from. Every "
                             "other check in this tool compares the eval bundle "
                             "against its own manifest, which a stale generation "
                             "of the corpus passes perfectly; this one compares "
                             "motion_sha256 and music_sha256 per recording_id "
                             "across the two lines. Measured 2026-08-24: 24 of "
                             "the 65 clips M6 scores (36.9%%) differ, and only "
                             "16 of those differ in frame count")
    parser.add_argument("--training-check", choices=("refuse", "warn", "off"),
                        default="refuse",
                        help="what a disagreement does. Without --training-bundle "
                             "nothing is compared and the manifest says so")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="export only the first N sequences, for a smoke run")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = export(args.bundle, args.split, args.output_dir, args.limit,
                          args.training_bundle, args.training_check)
    except ExportError as error:
        print("export refused: {}".format(error), file=sys.stderr)
        return 2
    print("{} sequence(s) -> {}".format(manifest["sequences"], args.output_dir))
    cross = manifest["training_bundle_cross_check"]
    if not cross["checked"]:
        # "Not compared" and "compared and agreed" must not print the same line.
        print("training cross-check: NOT RUN ({})".format(
            "no --training-bundle" if not cross["training_bundle"] else "policy off"))
    else:
        print("training cross-check: {} match, {} motion differ ({} at equal "
              "length), {} music differ, {} absent in training".format(
                  cross["match"], cross["motion_differs"],
                  cross["motion_differs_at_equal_length"], cross["music_differs"],
                  cross["absent_in_training"]))
    print("frames: {} total, median {:.0f}, range {}-{}".format(
        manifest["frames"]["total"], manifest["frames"]["median"],
        manifest["frames"]["min"], manifest["frames"]["max"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
