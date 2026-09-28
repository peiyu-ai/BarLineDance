#!/usr/bin/env python3
"""Does the body turn on its own axis, and if so was that in the library or in the splice?

THE OBSERVATION this exists to settle (operator, 2026-09-01, watching
``output/samples_20260901c``):

    "动作协调性也不高,有整个人位轴转的动作,motion 库里不该有吧,
     毫无美感也不遵循 physics"
    -- there are whole-body turns about the vertical axis; the library should
       not contain those.

That is two claims and they need different answers, so this tool separates
them rather than reporting one number:

  1. HOW MUCH the body turns          -> ``deg/s``, against ground truth.
  2. WHERE the turning happens        -> ``@bnd``, the share of the VIOLENT
     turning that falls on plan-segment boundaries, divided by the share of
     FRAMES those boundaries are.  Around 1 means the turning is spread evenly
     and the content itself turns; >> 1 means it is manufactured where two
     retrieved prototypes are pasted together.
  3. HOW OFTEN a clip has one at all  -> ``clips`` with at least one frame over
     the threshold.  This is the column that matches what a reviewer sees, and
     it is the one the first version of this file missed.

FIRST VERSION CORRECTED, because the correction is the point.  This tool at
first reported the MEAN turn rate per clip and the median per-clip ``@bnd``, and
on that reading the arms looked nearly fine: 74.1 deg/s against ground truth's
63.0, ``@bnd`` 0.94.  Both numbers were true and both hid the defect.  A spin is
a BRIEF violent event in an otherwise ordinary trace, so a mean dilutes it
(median peak rate: ground truth 100 deg/s, the same arm 523 deg/s -- 5x, on the
very clips the operator was watching), and a per-clip ``@bnd`` median is
degenerate at 0 because most clips' extreme frames miss a boundary.  Everything
below is POOLED and thresholded.

------------------------------------------------------------------ THE MEASURE

Facing is read off the HIPS -- the ground-plane direction of the vector from
the left hip to the right hip (SMPL joints 1 and 2, z is up), unwrapped over
time.  Read from joint POSITIONS, not from the SMPL global-orient parameter,
because positions are what the renderer draws and what the operator watched; a
parameterisation can carry a rotation that the mesh does not show, and vice
versa.  Smoothed with the same 3-tap Hann window the speed instruments use, so
per-frame reconstruction jitter is not counted as turning.

------------------------------------------------------------------- THE NULL

``@bnd`` needs a control or it is unreadable: some concentration is expected
simply because plan boundaries are chosen where the music changes and dancers
do turn at musical boundaries.  So the SAME boundary positions -- the generated
plan's own -- are applied to GROUND TRUTH.  Ground truth's concentration is the
baseline that a splice artefact has to beat.  Without this the number would be
compared against an assumed 1.0, and 1.0 is not what a real dance reads.

WHY THE MECHANISM IS PLAUSIBLE BEFORE MEASURING, stated so the measurement can
refute it: ``build_draft`` re-aligns the root POSITION across segments (and in
the shipped configuration ``draft_root_continuity`` is ``off``, so not even
that), and it never re-aligns root ORIENTATION.  Consecutive prototypes come
from different recordings, each carrying that recording's facing, so the pasted
draft can demand an arbitrary yaw change -- up to 180 degrees -- inside one
frame, and the completion model turns that into a fast spin.  If that is what
is happening, ``@bnd`` is large for the draft and stays large for the arms that
follow the draft most closely.
"""
import argparse
import json
import pathlib
import pickle
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools import arm_sourcing  # noqa: E402

FPS = 30.0
LEFT_HIP, RIGHT_HIP = 1, 2
SMOOTH = 3


def facing_yaw(joints):
    """Unwrapped ground-plane facing angle per frame, radians."""
    across = joints[:, RIGHT_HIP, :2] - joints[:, LEFT_HIP, :2]
    return np.unwrap(np.arctan2(across[:, 1], across[:, 0]))


def turn_rate(yaw, smooth=SMOOTH):
    """|d yaw / dt| in degrees per second, jitter-smoothed."""
    rate = np.abs(np.diff(yaw)) * FPS * 180.0 / np.pi
    if smooth > 1:
        kernel = np.hanning(smooth + 2)[1:-1]
        rate = np.convolve(rate, kernel / kernel.sum(), mode="same")
    return rate


def boundary_mask(labels, frames, radius=1):
    """Frames at a plan-segment boundary, on the rate axis (length frames-1)."""
    mask = np.zeros(frames - 1, bool)
    changes = np.flatnonzero(np.diff(np.asarray(labels)[:frames]) != 0)
    for offset in range(-radius, radius + 1):
        index = np.clip(changes + offset, 0, frames - 2)
        mask[index] = True
    return mask


def joints_of(path):
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    return np.asarray(payload["full_pose"], np.float64), payload.get("atomic_labels")


def score(clips, motion_dir, plan_dir, threshold, radius=1):
    """Pooled: over-threshold frames, where they sit, and how many clips have one."""
    over_total = 0
    over_on_boundary = 0
    boundary_frames = 0
    frames = 0
    clips_with_one = 0
    clips_seen = 0
    peaks = []
    off_boundary_rate = []
    directedness = []
    for clip in clips:
        motion = pathlib.Path(motion_dir) / (clip + ".pkl")
        plan = pathlib.Path(plan_dir) / (clip + ".pkl")
        if not (motion.is_file() and plan.is_file()):
            continue
        joints, _ = joints_of(motion)
        _, labels = joints_of(plan)
        if labels is None or len(joints) < 60:
            continue
        length = min(len(joints), len(labels))
        yaw = facing_yaw(joints[:length])
        rate = turn_rate(yaw)
        mask = boundary_mask(labels, length, radius)
        over = rate > threshold
        clips_seen += 1
        clips_with_one += bool(over.any())
        over_total += int(over.sum())
        over_on_boundary += int((over & mask).sum())
        boundary_frames += int(mask.sum())
        frames += len(mask)
        peaks.append(float(np.percentile(rate, 99)))
        if (~mask).any():
            off_boundary_rate.append(float(rate[~mask].mean()))
        # Net turn over the clip divided by total turning: ground truth's facing
        # OSCILLATES (it comes back), a spin ACCUMULATES.  Reported because a
        # rate alone cannot tell a lively dancer from a slowly rotating statue.
        total = float(rate.sum()) / FPS
        if total > 0:
            directedness.append(abs(yaw[-1] - yaw[0]) * 180.0 / np.pi / total)
    if not clips_seen:
        return {"clips": 0}
    expected = boundary_frames / frames if frames else float("nan")
    share = over_on_boundary / over_total if over_total else float("nan")
    return {
        "clips": clips_seen,
        "over_frames": over_total,
        "clips_with_a_spin": clips_with_one,
        "at_boundary": share / expected if expected else float("nan"),
        "p99": float(np.median(peaks)),
        "off_boundary": float(np.median(off_boundary_rate)) if off_boundary_rate else float("nan"),
        "directedness": float(np.median(directedness)) if directedness else float("nan"),
    }


def ground_truth_threshold(clips, motion_dir, quantile=99.5):
    """The bar a turn has to clear to count as violent: ground truth's own tail.

    Taken from ground truth rather than chosen, so the threshold cannot be tuned
    until an arm passes.  It is pooled over frames, not over clips, because the
    question is 'how fast does a real dancer ever turn', not 'per clip'.
    """
    rates = []
    for clip in clips:
        path = pathlib.Path(motion_dir) / (clip + ".pkl")
        if path.is_file():
            rates.append(turn_rate(facing_yaw(joints_of(path)[0])))
    if not rates:
        raise ValueError("no ground-truth motion found for the threshold")
    return float(np.percentile(np.concatenate(rates), quantile))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", required=True)
    parser.add_argument("--ground-truth-dir", required=True)
    parser.add_argument("--plan-dir", required=True,
                        help="run whose atomic_labels define the segment boundaries; "
                             "the SAME boundaries are applied to every arm including "
                             "ground truth, which is what makes @bnd readable")
    parser.add_argument("--arm", action="append", default=[], metavar="NAME=DIR")
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--quantile", type=float, default=99.5)
    parser.add_argument("--json")
    arm_sourcing.add_arguments(parser)
    args = parser.parse_args()

    requested = [line.strip() for line in open(args.clips) if line.strip()]
    # See tools/arm_sourcing.py: a clip with no retrieval group got an all-zero
    # draft, so its facing trace is the completion on music alone.  The same
    # surviving clip list is used for ground truth, or the null moves too.
    arm_specs = [(spec.partition("=")[0], spec.partition("=")[2]) for spec in args.arm]
    clips, sourcing = arm_sourcing.select_clips(
        requested, arm_specs, threshold=args.sourcing_threshold,
        include_unsourced=args.include_unsourced)
    for line in arm_sourcing.format_header(sourcing):
        print(line)
    threshold = ground_truth_threshold(clips, args.ground_truth_dir, args.quantile)
    jobs = [("ground truth", args.ground_truth_dir)] + arm_specs

    print("threshold = ground truth's pooled p{} = {:.0f} deg/s".format(
        args.quantile, threshold))
    print("{:<28}{:>5}{:>9}{:>9}{:>13}{:>9}{:>11}{:>9}".format(
        "arm", "n", "p99", "spins", "clips w/ spin", "@bnd", "off-bnd", "net/tot"))
    out = []
    for name, directory in jobs:
        row = score(clips, directory, args.plan_dir, threshold, args.radius)
        row["arm"] = name
        out.append(row)
        if not row["clips"]:
            print("{:<28}{:>5}".format(name, 0))
            continue
        print("{:<28}{:>5}{:>9.0f}{:>9}{:>9}/{:<3}{:>9.2f}{:>11.1f}{:>9.3f}".format(
            name, row["clips"], row["p99"], row["over_frames"],
            row["clips_with_a_spin"], row["clips"], row["at_boundary"],
            row["off_boundary"], row["directedness"]))
    print("\n@bnd  = (share of over-threshold frames on plan boundaries) / (share of frames they are).")
    print("off-bnd = mean turn rate away from boundaries -- the LIBRARY's own content, for the draft.")
    print("net/tot = |net facing change| / total turning: ground truth oscillates, a spin accumulates.")
    if args.json:
        # Dict, not the bare list this used to write: the sourcing header rides
        # with the numbers.  Old readers take payload["arms"].
        pathlib.Path(args.json).write_text(
            json.dumps({"sourcing": sourcing, "arms": out}, indent=2))


if __name__ == "__main__":
    main()
