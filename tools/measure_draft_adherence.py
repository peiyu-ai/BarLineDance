"""Does the completion model follow the retrieval draft's TIMING?

This is the mechanism behind the defect the operator reports as "no move is
ever executed to completion".  Measured on the shipped completion over 20
held-out clips:

  * per-frame joint-speed correlation between draft and output: **-0.023**;
  * on the frames where the DRAFT is holding still, the output moves at
    **1.02x its own median speed** -- it does not slow down at all;
  * output median speed is **1.78x** the draft's.

The draft is not a suggestion the model is free to ignore: on conditioned
frames it is the retrieved prototype, and it carries ground-truth-level
landings (hold share 2.4% on retrieved frames against ground truth's 2.1%).
The output has 0.6%.  So the landings exist in the material and are lost in
completion, and any improvement made on the retrieval side is lost with them --
which is why a better tie-break bought +0.0413 of across-clip novelty on the
draft and -0.0025 on the output.

``speed_on_draft_holds`` is the column that matters and it is deliberately
normalised by the ARM'S OWN median rather than the draft's: an arm that simply
moves less would otherwise look like it was following.  A value near 1.0 means
"ignores the hold"; a value near ground truth's own behaviour means "lands
where the material lands".
"""

import argparse
import json
import pathlib
import pickle
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

FPS = 30.0


def joint_speed(joints):
    joints = np.asarray(joints, float)
    relative = joints - joints[:, :1, :]
    return np.linalg.norm(np.diff(relative, axis=0), axis=2).mean(axis=1) * FPS


def adherence(draft_speed, output_speed, *, hold_fraction=0.25):
    """Correlation, and what the output does where the draft holds."""
    n = min(len(draft_speed), len(output_speed))
    draft_speed, output_speed = draft_speed[:n], output_speed[:n]
    if n < 8 or draft_speed.std() == 0 or output_speed.std() == 0:
        return None
    draft_median = float(np.median(draft_speed))
    output_median = float(np.median(output_speed))
    if draft_median <= 0 or output_median <= 0:
        return None
    held = draft_speed < hold_fraction * draft_median
    if held.sum() < 3:
        return None
    return {
        "speed_correlation": float(np.corrcoef(draft_speed, output_speed)[0, 1]),
        "speed_on_draft_holds": float(np.median(output_speed[held]) / output_median),
        "output_over_draft_speed": float(output_median / draft_median),
    }


def load_speed(path):
    return joint_speed(np.asarray(pickle.load(open(path, "rb"))["full_pose"], float))


def run(arms, draft_dir, clips, ground_truth_dir=None, **options):
    report = {"draft_dir": str(draft_dir), "arms": {}}
    for name, directory in arms:
        rows, missing = [], 0
        for clip in clips:
            draft = pathlib.Path(draft_dir) / (clip + ".pkl")
            out = pathlib.Path(directory) / (clip + ".pkl")
            if not (draft.exists() and out.exists()):
                missing += 1
                continue
            measured = adherence(load_speed(draft), load_speed(out), **options)
            if measured is not None:
                rows.append(measured)
        if not rows:
            raise SystemExit(
                "error: arm {!r} scored 0 of {} clips against {} ({} pairs "
                "missing).  Nothing was measured.".format(
                    name, len(clips), draft_dir, missing))
        report["arms"][name] = {
            key: float(np.median([r[key] for r in rows])) for key in rows[0]}
        report["arms"][name]["clips"] = len(rows)
    if ground_truth_dir is not None:
        # Ground truth against the same draft: the number an arm should be
        # compared to, since a real dancer does not follow this draft either.
        rows = []
        for clip in clips:
            draft = pathlib.Path(draft_dir) / (clip + ".pkl")
            truth = pathlib.Path(ground_truth_dir) / (clip + ".pkl")
            if not (draft.exists() and truth.exists()):
                continue
            measured = adherence(load_speed(draft), load_speed(truth), **options)
            if measured is not None:
                rows.append(measured)
        if rows:
            report["ground_truth_vs_same_draft"] = {
                key: float(np.median([r[key] for r in rows])) for key in rows[0]}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--draft", required=True)
    parser.add_argument("--clips", required=True, type=pathlib.Path)
    parser.add_argument("--ground-truth")
    parser.add_argument("--hold-fraction", type=float, default=0.25)
    parser.add_argument("--out", type=pathlib.Path)
    arguments = parser.parse_args()

    arms = []
    for entry in arguments.arm:
        name, _, directory = entry.rpartition("=")
        if not pathlib.Path(directory).is_dir():
            raise SystemExit("error: --arm {!r} points at {!r}, which does not "
                             "exist.".format(name, directory))
        arms.append((name, directory))
    clips = [line.strip() for line in arguments.clips.read_text().splitlines()
             if line.strip()]
    report = run(arms, arguments.draft, clips, arguments.ground_truth,
                 hold_fraction=arguments.hold_fraction)
    text = json.dumps(report, indent=2, sort_keys=True)
    if arguments.out:
        arguments.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
