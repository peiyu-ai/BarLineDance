#!/usr/bin/env python3
"""When an atomic label recurs in a clip, is it the same motion twice?

THE DEFECT.  ``IndexedAtomicMotionLibrary.retrieve`` caches on
``(label, target_length, excluded, rule, period)`` and the default ``duration``
rule is a deterministic ``min`` over a fixed candidate list.  So the second
occurrence of a label in a clip returns the identical tensor -- and the bar grid
forces every segment to the same number of beats, which makes ``target_length``
near-constant and the key collide on essentially every repeat.

THE PAPER asks for the opposite in as many words (atomicDance 3.4): "as
choreographic theory indicates, structured movements should exhibit variation
when they recur."  So this is not an invented target: the criterion is the
paper's sentence, and the reading it produces on the shipped code is that the
variation is exactly zero.

TWO READOUTS, because the first one is too easy to satisfy.

``identical``  share of same-label span pairs that are bit-identical after the
               shared time-stretch.  Gate: 0.  A change that only perturbs the
               bytes passes this and nothing else.

               **Read it on the DRAFT** (``--data-root``), not on the generated
               output: the reverse diffusion perturbs every frame, so the output
               is never bit-identical even when the draft it was built from was
               two copies of one tensor.  Measuring identity on the output was
               the first version of this file and it read 0.000 on the very code
               that has the defect.
``contrast``   mean pose distance between same-label spans, divided by the mean
               between different-label spans.  Meaningful on the output as well
               as the draft, and on the output it asks a sharper question: does
               the label survive generation at all?  Measured 2026-08-30, the
               shipped arm reads 0.801 and the ground truth 0.776, while the
               current bar-2 arm reads **0.951** -- its same-label spans are
               nearly as far apart as its different-label ones.  This is the one that says the
               vocabulary means anything: if repeating a label gives you as
               different a movement as changing it, the label is decoration.
               **The target is measured from the ground truth in the same run,
               not assumed** -- reported side by side, never hard-coded.

MEASURED 2026-08-30, 100 wild clips, 145 same-label span pairs, drafts rebuilt
from each clip's own recorded plan:

    rule                       identical   same m   diff m   contrast
    shipped (duration + cache)     0.207    2.578    3.659      0.704
    --draft-recurrence-variety     0.000    3.237    3.650      0.887

**Both columns must be read.**  The fix removes literal repetition, which is
the paper's ask; it also pushes ``contrast`` from 0.704 toward 1.0, i.e. two
spans of the SAME label become nearly as far apart as two spans of different
labels.  That is the direction in which the vocabulary stops meaning anything,
and the ground truth's own output-level contrast is 0.776.  Two criteria
pointing opposite ways is the situation CLAUDE.md 2.1 gate 4 names: do not pick
the flattering one.  The tie is broken downstream, on energy and on what the
generated dance looks like, not here.

WHAT IT CANNOT SAY.  Nothing here is about quality.  A draft can score perfectly
by picking wildly unsuitable prototypes; that is the paper's own ``Random
Choice`` ablation and it is worse on FID.  Read this beside
``tools/score_arm_table.py``'s energy and FID columns or not at all.
"""
import argparse
import itertools
import json
import pathlib
import pickle
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

MIN_SPAN = 20
COMPARE = 30


def runs_of(labels):
    labels = np.asarray(labels)
    if len(labels) == 0:
        return []
    edges = np.flatnonzero(np.diff(labels)) + 1
    bounds = [0, *edges.tolist(), len(labels)]
    return [(bounds[i], bounds[i + 1], int(labels[bounds[i]]))
            for i in range(len(bounds) - 1)]


def clip_readings(features, labels):
    """``features`` is [T, D]: root-relative joints flattened, or raw 151-D."""
    spans = [r for r in runs_of(labels) if r[1] - r[0] >= MIN_SPAN and r[2] != 0]
    same, different, identical, pairs = [], [], 0, 0
    for (a0, a1, la), (b0, b1, lb) in itertools.combinations(spans, 2):
        n = min(a1 - a0, b1 - b0, COMPARE)
        left, right = features[a0:a0 + n], features[b0:b0 + n]
        distance = float(np.linalg.norm(left - right, axis=-1).mean())
        if la == lb:
            same.append(distance)
            pairs += 1
            identical += int(distance < 1e-6)
        else:
            different.append(distance)
    return same, different, identical, pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", required=True)
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--ground-truth-dir", default=None)
    parser.add_argument("--labels-root", default=None)
    parser.add_argument("--json", default=None)
    parser.add_argument("--gate-identical", type=float, default=None)
    parser.add_argument("--data-root", default=None,
                        help="release root; when given, each arm's DRAFT is rebuilt from "
                             "its own recorded plan and measured instead of its output. "
                             "Retrieval carries no RNG for the default rule, so the "
                             "rebuild is exact rather than approximate")
    parser.add_argument("--recurrence-variety", action="store_true",
                        help="rebuild the drafts with the variety rule, to read what it "
                             "would have produced without rerunning inference")
    args = parser.parse_args()

    clips = [line.strip() for line in open(args.clips) if line.strip()]
    labels_by_clip = {}
    if args.labels_root:
        root = pathlib.Path(args.labels_root)
        for line in open(root / "labels.jsonl"):
            record = json.loads(line)
            if record["sequence_id"] in clips:
                labels_by_clip[record["sequence_id"]] = np.load(root / record["labels_path"])

    library = None
    if args.data_root:
        from infer_atomic import IndexedAtomicMotionLibrary, _source_safe_draft, unnormalize_motion
        library = IndexedAtomicMotionLibrary(args.data_root)

    arms = [spec.partition("=")[::2] for spec in args.arm]
    if args.ground_truth_dir:
        arms.insert(0, ("ground truth", args.ground_truth_dir))

    print("{:<34} {:>6} {:>10} {:>10} {:>10} {:>10}".format(
        "arm", "pairs", "identical", "same m", "diff m", "contrast"))
    reports, failed = [], []
    for name, directory in arms:
        directory = pathlib.Path(directory)
        same, different, identical, pairs = [], [], 0, 0
        for clip in clips:
            path = directory / (clip + ".pkl")
            if not path.is_file():
                continue
            payload = pickle.load(open(path, "rb"))
            joints = np.asarray(payload["full_pose"], float)
            features = None
            labels = payload.get("atomic_labels")
            if labels is None:
                labels = labels_by_clip.get(clip)
            if labels is None:
                continue
            if library is not None and name != "ground truth":
                import torch
                group = library.query_retrieval_group_id(clip)
                draft, _mask = _source_safe_draft(
                    library, torch.from_numpy(np.asarray(labels)[:len(joints)].astype(np.int64)),
                    library.motion.shape[-1], group,
                    recurrence_variety=args.recurrence_variety,
                    variety_rng=np.random.default_rng(20260830))
                raw = unnormalize_motion(draft, str(pathlib.Path(args.data_root) / "normalizer.pt"))
                # Compared in the 151-D feature space rather than as joints: the
                # question is whether the retrieved bytes repeat, and decoding
                # would add a forward-kinematics step that can only blur that.
                # The raw 151-D vector, NOT made root-relative: the question is
                # whether the retrieved bytes repeat.  Subtracting joint 0 from a
                # feature vector -- which the joint path does and the first
                # version of this file did here too -- zeroes the whole array and
                # every distance with it, which reads as "no difference anywhere"
                # for both arms.
                features = np.asarray(raw, float)
            else:
                relative = joints - joints[:, :1, :]
                features = relative.reshape(len(relative), -1)
            s, d, i, p = clip_readings(features, np.asarray(labels)[:len(features)])
            same += s
            different += d
            identical += i
            pairs += p
        if not pairs:
            continue
        contrast = float(np.mean(same) / np.mean(different)) if different else float("nan")
        report = {"arm": name, "pairs": pairs, "identical": identical / pairs,
                  "same_m": float(np.mean(same)), "diff_m": float(np.mean(different)),
                  "contrast": contrast}
        reports.append(report)
        print("{arm:<34} {pairs:>6} {identical:>10.3f} {same_m:>10.3f} "
              "{diff_m:>10.3f} {contrast:>10.3f}".format(**report))
        if args.gate_identical is not None and report["identical"] > args.gate_identical:
            failed.append("{}: {:.3f} of same-label pairs are bit-identical".format(
                name, report["identical"]))
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(reports, indent=2), encoding="utf-8")
    for line in failed:
        print("FAIL " + line)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
