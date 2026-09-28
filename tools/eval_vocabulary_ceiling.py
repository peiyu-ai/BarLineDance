#!/usr/bin/env python3
"""Score a vocabulary by the R-precision it still permits, without training anything.

Tab. 2 prices in-group re-clustering in FID and R, and those are end-to-end
numbers: they need a planner and a completion model trained on the vocabulary
before the column can be filled.  That is the right experiment and it is also
days of GPU per vocabulary, which is a slow way to discover that a vocabulary
was never going to work.

This is the cheap upper bound on the same question.  The paper itself reports
the corresponding row -- Tab. 3's "Using GT" line, where the planner is replaced
by ground-truth labels -- so the construction is the paper's own: keep the real
label sequence, and rebuild the dance out of the vocabulary by retrieving, for
every labelled run, a real segment carrying the same label from the training
split, scaled to the run's duration.  No model predicts anything, so what comes
out is the best a perfect planner could do on this vocabulary.  If two
vocabularies differ here, they differ before any training; if they do not, the
training will be deciding something else.

Four rows are always reported together, because three of them exist to stop the
fourth from being over-read:

* ``ground_truth`` -- the real motion.  The reference R, not a result.
* ``retrieved`` -- rebuilt through the vocabulary under true labels.
* ``random_label`` -- rebuilt by retrieving a *random* label's segment for each
  run.  This is the control that matters: dance motion resembles dance motion,
  so a reconstruction scores well above chance even when the labels carry
  nothing.  The gap between ``retrieved`` and ``random_label`` is the part of R
  the vocabulary is actually responsible for.
* ``fixed_exemplar`` -- every run of a class replaced by the *same* representative
  of that class, so the exemplar choice stops being a source of noise.  This is
  the cleanest reading of what a label by itself encodes, and the one to compare
  vocabularies on.
* ``duration_only`` -- retrieves ignoring the label but matching the duration,
  which separates "the vocabulary knows the move" from "the vocabulary knows how
  long the move is".  Duration alone is a real signal and it is not a semantic
  one.

Transitions (label 0) are filled by linear interpolation between the segments on
either side, identically in every row, so the comparison is unaffected by the
choice; only ``ground_truth`` keeps real motion there.

Usage:
    python3 tools/eval_vocabulary_ceiling.py \\
        --labels data/wild3d/wild_v2_ingroup_nollm \\
        --bundle data/wild3d/wild_performance_v1 \\
        --split test --output runs/ceiling_nollm.json
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.eval_r_precision import (  # noqa: E402
    RPrecisionError,
    motion_clip_feature,
    music_clip_feature,
    r_precision,
    split_into_clips,
)
from tools.recluster_atomics_ingroup import segments_of  # noqa: E402

FPS = 30.0
TRANSITION = 0


class CeilingError(RuntimeError):
    pass


def load_rows(labels_dir: pathlib.Path, bundle: pathlib.Path
              ) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    sequences = {json.loads(line)["recording_id"]: json.loads(line)
                 for line in (bundle / "sequences.jsonl").open(encoding="utf-8")}
    labels = {json.loads(line)["recording_id"]: json.loads(line)
              for line in (labels_dir / "labels.jsonl").open(encoding="utf-8")}
    missing = set(labels) - set(sequences)
    if missing:
        raise CeilingError("{} labelled recordings are absent from {}".format(
            len(missing), bundle))
    return sequences, labels


def build_library(labels_dir: pathlib.Path, sequences: Dict[str, Dict],
                  label_rows: Dict[str, Dict], split: str
                  ) -> Tuple[Dict[int, List[Dict]], Dict[str, object]]:
    """Every labelled run in ``split``, indexed by label.

    The library is drawn from the training split only, which is what makes the
    reconstruction honest: a test sequence rebuilt from segments of itself would
    measure nothing but the copy.
    """
    library: Dict[int, List[Dict]] = collections.defaultdict(list)
    runs = 0
    for recording, row in label_rows.items():
        if sequences[recording].get("split") != split:
            continue
        labels = np.load(labels_dir / row["labels_path"])
        for start, end, label in segments_of(labels):
            if label <= TRANSITION or end - start < 4:
                continue
            library[int(label)].append({
                "recording_id": recording, "start": int(start), "end": int(end),
                "length": int(end - start),
                "retrieval_group_id": sequences[recording].get("retrieval_group_id"),
            })
            runs += 1
    if not library:
        raise CeilingError("the {} split holds no labelled runs".format(split))
    for entries in library.values():
        entries.sort(key=lambda entry: entry["length"])
    return library, {"library_split": split, "library_runs": runs,
                     "library_labels": len(library)}


def resample(joints: np.ndarray, length: int) -> np.ndarray:
    """Scale a segment onto ``length`` frames by linear interpolation in joint space.

    The paper's completion stage scales a retrieved primitive to the target
    duration, so the reconstruction does the same.  Interpolating positions
    rather than rotations is a simplification, and it is safe here because the
    only consumers are the kinetic and manual features, which read positions.
    """
    if len(joints) == length:
        return joints
    source = np.linspace(0.0, 1.0, len(joints))
    target = np.linspace(0.0, 1.0, length)
    flat = joints.reshape(len(joints), -1)
    scaled = np.stack([np.interp(target, source, flat[:, column])
                       for column in range(flat.shape[1])], axis=1)
    return scaled.reshape(length, *joints.shape[1:])


def pick(entries: Sequence[Dict], length: int, group: Optional[str],
         rng: np.random.Generator, *, duration_aware: bool) -> Optional[Dict]:
    """Choose one library entry, excluding anything from the query's own group."""
    usable = [entry for entry in entries if entry["retrieval_group_id"] != group]
    if not usable:
        return None
    if not duration_aware:
        return usable[int(rng.integers(len(usable)))]
    lengths = np.asarray([entry["length"] for entry in usable])
    return usable[int(np.argmin(np.abs(lengths - length)))]


def reconstruct(joints: np.ndarray, labels: np.ndarray, library: Dict[int, List[Dict]],
                joints_for, group: Optional[str], mode: str,
                rng: np.random.Generator,
                flat_pool: Optional[Sequence[Dict]] = None
                ) -> Tuple[np.ndarray, Dict[str, int]]:
    """Rebuild one sequence out of the vocabulary under the requested policy."""
    if mode == "ground_truth":
        return joints.copy(), {"runs": 0, "filled": 0, "unfillable": 0}

    output = np.full_like(joints, np.nan)
    all_labels = sorted(library)
    if flat_pool is None:
        flat_pool = [entry for entries in library.values() for entry in entries]
    stats = {"runs": 0, "filled": 0, "unfillable": 0}
    for start, end, label in segments_of(labels):
        if label <= TRANSITION or end - start < 4:
            continue
        stats["runs"] += 1
        if mode == "retrieved":
            entries, duration_aware = library.get(int(label), []), True
        elif mode == "fixed_exemplar":
            # One fixed representative per class -- the median-duration member,
            # not a medoid in any feature space -- so two clips carrying the same
            # label receive the *same* motion.  Duration-nearest retrieval does
            # not: it hands near-identical clips different members of the class,
            # and that choice is noise with respect to the music, enough to bury
            # whatever signal the class carries.  This mode is the clean
            # measurement of what a label alone encodes.
            entries = library.get(int(label), [])
            entries = entries[len(entries) // 2:len(entries) // 2 + 1] if entries else []
            duration_aware = False
        elif mode == "random_label":
            entries = library[all_labels[int(rng.integers(len(all_labels)))]]
            duration_aware = True
        elif mode == "duration_only":
            # Ignore the label entirely: every run in the corpus is a candidate
            # and only the duration decides.  A bounded random draw keeps this
            # affordable without biasing which lengths are reachable.
            entries = [flat_pool[index] for index in
                       rng.integers(len(flat_pool), size=min(400, len(flat_pool)))]
            duration_aware = True
        else:
            raise CeilingError("unknown reconstruction mode {}".format(mode))
        chosen = pick(entries, end - start, group, rng, duration_aware=duration_aware)
        if chosen is None:
            stats["unfillable"] += 1
            continue
        donor = joints_for(chosen["recording_id"])[chosen["start"]:chosen["end"]]
        output[start:end] = resample(donor, end - start)
        stats["filled"] += 1

    return fill_gaps(output), stats


def fill_gaps(output: np.ndarray) -> np.ndarray:
    """Bridge transition frames by interpolating between the segments around them.

    Every row of the comparison passes through this identically, so whatever the
    policy costs, it costs each vocabulary the same.
    """
    filled = np.isfinite(output).all(axis=(1, 2))
    if not filled.any():
        raise CeilingError("nothing was reconstructed for this sequence")
    known = np.flatnonzero(filled)
    frames = np.arange(len(output))
    flat = output.reshape(len(output), -1)
    for column in range(flat.shape[1]):
        flat[:, column] = np.interp(frames, known, flat[known, column])
    return flat.reshape(output.shape)


def evaluate(labels_dir: pathlib.Path, bundle: pathlib.Path, *, split: str,
             library_split: str, clip_frames: int, feature: str,
             music_aggregate: str, modes: Sequence[str], seed: int,
             limit: Optional[int] = None) -> Dict[str, object]:
    from tools.convert_motion_to_guofeats import motion_151_to_joints

    sequences, label_rows = load_rows(labels_dir, bundle)
    library, library_info = build_library(labels_dir, sequences, label_rows, library_split)

    cache: Dict[str, np.ndarray] = {}

    def joints_for(recording: str) -> np.ndarray:
        if recording not in cache:
            cache[recording] = motion_151_to_joints(
                np.load(bundle / sequences[recording]["motion_path"]))
        return cache[recording]

    targets = [recording for recording, row in label_rows.items()
               if sequences[recording].get("split") == split]
    targets.sort()
    if limit is not None:
        targets = targets[:limit]
    if not targets:
        raise CeilingError("no {} sequences carry labels".format(split))

    groups = {sequences[recording].get("retrieval_group_id") for recording in targets}
    overlap = groups & {entry["retrieval_group_id"]
                        for entries in library.values() for entry in entries}
    if overlap:
        raise CeilingError(
            "{} retrieval groups appear in both the {} split and the {} library; "
            "a sequence rebuilt from itself measures the copy, not the vocabulary"
            .format(len(overlap), split, library_split))

    flat_pool = [entry for entries in library.values() for entry in entries]
    rows: Dict[str, Dict[str, List]] = {
        mode: {"music": [], "motion": [], "groups": []} for mode in modes}
    run_stats: Dict[str, collections.Counter] = {
        mode: collections.Counter() for mode in modes}

    for index, recording in enumerate(targets):
        row = label_rows[recording]
        labels = np.load(labels_dir / row["labels_path"])
        joints = joints_for(recording)
        music = np.load(bundle / sequences[recording]["music_path"])
        group = sequences[recording].get("retrieval_group_id")
        spans = split_into_clips(min(len(joints), len(music)), clip_frames)
        if not spans:
            continue
        for mode in modes:
            rng = np.random.default_rng(seed + index)
            try:
                rebuilt, stats = reconstruct(joints, labels, library, joints_for,
                                             group, mode, rng, flat_pool)
            except CeilingError:
                continue
            run_stats[mode].update(stats)
            for start, end in spans:
                rows[mode]["motion"].append(motion_clip_feature(rebuilt[start:end], feature))
                rows[mode]["music"].append(music_clip_feature(music[start:end],
                                                              music_aggregate))
                rows[mode]["groups"].append(str(group))
        if (index + 1) % 25 == 0:
            print("  rebuilt {}/{} sequences".format(index + 1, len(targets)), flush=True)

    report = {
        "labels": str(labels_dir), "bundle": str(bundle), "split": split,
        "clip_frames": int(clip_frames), "clip_seconds": round(clip_frames / FPS, 3),
        "motion_feature": feature,
        "music_feature": "35-D baseline, aggregated by {}".format(music_aggregate),
        "sequences": len(targets), "seed": seed, **library_info,
        "construction": ("true labels kept, every labelled run replaced by a "
                         "duration-nearest run of the same label from the library "
                         "split; this is the paper's Tab. 3 'Using GT' setting "
                         "without a trained planner"),
        "rows": {},
    }
    for mode in modes:
        if not rows[mode]["motion"]:
            report["rows"][mode] = {"error": "nothing reconstructed"}
            continue
        # Per-clip ranks kept so two vocabularies can be compared *paired*.
        # Their means each carry a standard error near 0.017 on this split,
        # which is the size of the differences being argued about; the clips
        # and the music are identical across runs, so the paired difference is
        # the comparison that is actually available.
        measured = r_precision(np.stack(rows[mode]["music"]),
                               np.stack(rows[mode]["motion"]),
                               groups=rows[mode]["groups"], keep_ranks=True)
        measured["runs_filled"] = int(run_stats[mode]["filled"])
        measured["runs_unfillable"] = int(run_stats[mode]["unfillable"])
        report["rows"][mode] = measured
        print("  {:>14}: R {:.2f}  (chance {:.2f}, {} clips)".format(
            mode, measured["R"], measured["chance_R"], measured["clips"]), flush=True)

    if "retrieved" in report["rows"] and "random_label" in report["rows"]:
        retrieved = report["rows"]["retrieved"].get("R")
        control = report["rows"]["random_label"].get("R")
        if retrieved is not None and control is not None:
            report["label_information"] = round(retrieved - control, 2)
            report["label_information_note"] = (
                "R(retrieved) - R(random_label): the share of R that the labels "
                "are responsible for, rather than the share that any dance motion "
                "would have produced")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=pathlib.Path, required=True)
    parser.add_argument("--bundle", type=pathlib.Path, required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--library-split", default="train",
                        choices=("train", "val", "test"))
    parser.add_argument("--clip-frames", type=int, default=150)
    parser.add_argument("--feature", default="kinetic", choices=("kinetic", "manual"))
    parser.add_argument("--music-aggregate", default="mean_std", choices=("mean", "mean_std"))
    parser.add_argument("--modes",
                        default="ground_truth,retrieved,fixed_exemplar,"
                                "random_label,duration_only",
                        help="comma-separated; the controls are not optional decoration, "
                             "an R without them cannot be attributed to the vocabulary")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    try:
        report = evaluate(args.labels, args.bundle, split=args.split,
                          library_split=args.library_split, clip_frames=args.clip_frames,
                          feature=args.feature, music_aggregate=args.music_aggregate,
                          modes=modes, seed=args.seed, limit=args.limit)
    except (CeilingError, RPrecisionError) as error:
        print("ceiling refused: {}".format(error), file=sys.stderr)
        return 2
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print("wrote {}".format(args.output))
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
