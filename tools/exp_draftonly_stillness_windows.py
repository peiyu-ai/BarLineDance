"""Share of random 150-frame windows that contain a SUSTAINED stillness.

WHY THIS FILE EXISTS.  docs/DANCE_QUALITY_DEFECTS.md section 22 replaced the
deciding column for "does the body ever land" with *the share of random
150-frame windows that contain a sustained stillness*, because the older column
(hold share, section 18.3) has no power at the 150-frame scale: ground truth's
2.1% of held frames is ~3 frames in five seconds and the per-clip median is
0.00% (section 21.1).  Section 22 reports ground truth 29%, the training windows
27%, the shipped arm 6% -- but the computation lived in a scratch script that
no longer exists, so this is a RE-IMPLEMENTATION from the section's wording,
and it says so.  Its calibration against the section's numbers is a runtime
check (``--expect-ground-truth``), not an assumption.

DEFINITION (each choice is the one section 20/22 already made):

  * window       150 frames, ``--windows-per-clip`` starts drawn uniformly per
                 clip from a seeded generator -- the same sampling section 22
                 used to put the training windows and the generations on one
                 scale ("对照必须对齐尺度").
  * speed        root-relative mean joint speed after a 9-frame moving average
                 (``measure_motion_dynamics.low_pass`` / ``SMOOTH_WIDTH``): the
                 averaging is what separates a sustained still state from the
                 zero-crossing dip of a body that never stops (section 20.2:
                 ground truth keeps 1.76% of frames held after it, every
                 generated arm reads 0.00%).
  * held frame   smoothed speed below ``--hold-fraction`` (0.25) of the
                 window's OWN median smoothed speed.  Relative, so a slow song
                 is not one long hold; the window's own median, so the window
                 stands alone the way a 150-frame training window does.  A
                 window whose own median is zero is REFUSED, not scored: see
                 tools/stillness_criterion.py, which holds this definition and
                 is shared with measure_motion_dynamics and
                 exp_guidance_stillness so the three cannot disagree again.
  * contains     the window has at least ``--min-run`` consecutive held frames
                 (default 1 -- after a 9-frame average a single held frame
                 already means the body was slow for most of a third of a
                 second; raise it to demand longer holds).

The statistic is the pooled share of windows, and beside it the share of CLIPS
whose whole-clip smoothed speed has any held frame (section 22: truth 80%,
shipped 25%).  ``p90_over_p10`` inside the window is carried too, since section
22 prints them side by side.

CONTROLS (tests/test_exp_draftonly_stillness_windows.py): a constant-velocity
body must read 0%; a body that travels then holds must read near 100%; a hold
that is a single-frame dip must NOT count after the average (the section 20.2
distinction); and, with the release present, ground truth on the 20 held-out
clips must land in the 27-29% band section 22 reports, otherwise the
re-implementation is not the criterion it claims to be and the run refuses.
"""

import argparse
import json
import pathlib
import pickle
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.stillness_criterion import (FPS, HOLD_FRACTION,  # noqa: E402
                                       MEASURABLE_MEDIAN_SPEED, SMOOTH_WIDTH,
                                       DegenerateMotion, DropLog, dynamic_range,
                                       held_frames, joint_speed, longest_run,
                                       low_pass)

WINDOW = 150

# The definition and its refusal live in tools/stillness_criterion.py, shared
# with measure_motion_dynamics and exp_guidance_stillness, because the three
# copies agreed on the definition and DISAGREED on the degenerate input.
# OLD BEHAVIOUR here: ``if speed.size < 3 or median <= 0: return False, nan, 0``
# -- a motionless window was reported as a window WITHOUT a hold and counted in
# the denominator of ``window_share_with_sustained_hold``; a motionless clip was
# reported as ``clip_has_sustained_hold: False`` and counted in the denominator
# of ``clip_share_with_sustained_hold``.  NEW BEHAVIOUR: it raises, the window
# or clip is dropped, and the drop is named in the JSON header.


def window_starts(frames, count, rng, window=WINDOW):
    """Uniform random starts; a clip shorter than the window yields none."""
    if frames < window:
        return np.zeros(0, int)
    return rng.integers(0, frames - window + 1, size=count)


def window_contains_hold(joints_window, *, hold_fraction=HOLD_FRACTION, min_run=1,
                         smooth_width=SMOOTH_WIDTH):
    """(contains, p90_over_p10, longest_run_frames) for ONE window of joints.

    RAISES ``DegenerateMotion`` when the window has no speed scale of its own.
    ``p90_over_p10`` is None (not a clamp-fabricated 1e11) when only the raw p10
    has vanished, which is a different and weaker degeneracy than the hold
    threshold's."""
    speed = joint_speed(low_pass(joints_window, smooth_width))
    held = held_frames(speed, hold_fraction, what="window")
    run = longest_run(held)
    try:
        ratio = dynamic_range(speed, what="window")
    except DegenerateMotion:
        ratio = None
    return bool(run >= min_run), ratio, int(run)


def clip_has_sustained_hold(joints, *, hold_fraction=HOLD_FRACTION,
                            smooth_width=SMOOTH_WIDTH):
    """Whole-clip scale: any held frame after the moving average.  RAISES on a
    motionless body rather than answering False, which is the answer that made
    "completely still" and "never still" the same reading."""
    speed = joint_speed(low_pass(joints, smooth_width))
    return bool(held_frames(speed, hold_fraction, what="clip").any())


def score_clip(joints, *, windows_per_clip, seed, hold_fraction=HOLD_FRACTION,
               min_run=1, smooth_width=SMOOTH_WIDTH, window=WINDOW):
    rng = np.random.default_rng(seed)
    starts = window_starts(len(joints), windows_per_clip, rng, window)
    contains, ratios, runs = [], [], []
    dropped_windows, dropped_ratio = 0, 0
    for s in starts:
        try:
            c, r, run = window_contains_hold(
                joints[s:s + window], hold_fraction=hold_fraction,
                min_run=min_run, smooth_width=smooth_width)
        except DegenerateMotion:
            dropped_windows += 1
            dropped_ratio += 1
            continue
        contains.append(c)
        if r is None:
            # p90/p10 needs a nonzero raw p10, the hold threshold needs a
            # nonzero median: a window can lose the first and keep the second,
            # so the ratio column has its own denominator.
            dropped_ratio += 1
        else:
            ratios.append(r)
        runs.append(run)
    try:
        whole = clip_has_sustained_hold(joints, hold_fraction=hold_fraction,
                                        smooth_width=smooth_width)
    except DegenerateMotion:
        whole = None
    return {
        "windows_drawn": int(len(starts)),
        "windows_measured": int(len(contains)),
        "windows_dropped": int(dropped_windows),
        "windows_dropped_p90_over_p10": int(dropped_ratio),
        "windows_with_hold": int(sum(contains)),
        "p90_over_p10_median": (float(np.median(ratios)) if ratios else None),
        "longest_run_frames_max": int(max(runs)) if runs else 0,
        # None, not False: the criterion has no reading on a motionless body.
        "clip_has_sustained_hold": whole,
        "clip_measurable": whole is not None,
        "frames": int(len(joints)),
    }


def load(path):
    return np.asarray(pickle.load(open(path, "rb"))["full_pose"], float)


def score_arm(directory, clips, **options):
    """Pooled over the windows and clips the criterion can measure.

    OLD BEHAVIOUR: ``windows`` counted every drawn window and ``clips`` every
    present clip, so a motionless draft put its windows in the denominator as
    windows without a hold.  NEW BEHAVIOUR: the denominators are the measured
    counts and the refusals are named in ``dropped_clips``."""
    log = DropLog("clips")
    per_clip, absent = {}, []
    for clip in clips:
        path = pathlib.Path(directory) / (clip + ".pkl")
        if not path.exists():
            absent.append(clip)
            continue
        # One generator per clip, seeded by the clip's position, so the arms
        # draw their windows from the same generator state; lengths differ
        # between ground truth and generation so the starts cannot be equal,
        # but the sampling is the same and reproducible.
        row = score_clip(load(path), **options)
        per_clip[clip] = row
        if row["clip_measurable"]:
            log.measured.append(clip)
        else:
            log.drop(clip, "whole-clip median smoothed speed <= {:.0e} m/s "
                           "(motionless body: the self-relative hold threshold "
                           "collapses to zero)".format(MEASURABLE_MEDIAN_SPEED))
    if not per_clip:
        raise SystemExit("error: 0 of {} clips scored from {}".format(len(clips), directory))
    if not log.measured:
        raise SystemExit(
            "error: every one of the {} clips found in {} is unmeasurable by "
            "the sustained-stillness criterion; refusing to report a number.\n{}"
            .format(len(per_clip), directory, "\n".join(log.lines())))
    drawn = sum(r["windows_drawn"] for r in per_clip.values())
    measured = sum(r["windows_measured"] for r in per_clip.values())
    with_hold = sum(r["windows_with_hold"] for r in per_clip.values())
    clips_with = sum(1 for r in per_clip.values() if r["clip_has_sustained_hold"])
    ratios = [r["p90_over_p10_median"] for r in per_clip.values()
              if r["p90_over_p10_median"] is not None]
    return {
        "clips": len(log.measured),
        "windows_drawn": drawn,
        "windows_measured": measured,
        "windows_dropped": drawn - measured,
        "windows_dropped_p90_over_p10": sum(r["windows_dropped_p90_over_p10"]
                                            for r in per_clip.values()),
        "window_share_with_sustained_hold": ((with_hold / measured) if measured
                                             else None),
        "clip_share_with_sustained_hold": clips_with / len(log.measured),
        "clips_with_sustained_hold": "{}/{}".format(clips_with, len(log.measured)),
        "p90_over_p10_window_median": (float(np.median(ratios)) if ratios
                                       else None),
        "absent_clips": absent,
        "per_clip": per_clip,
    } | log.header()


def run(arms, clips, ground_truth_dir, *, expect_ground_truth=None, **options):
    truth = score_arm(ground_truth_dir, clips, **options)
    report = {"definition": {
        "window_frames": WINDOW, "smooth_width": options.get("smooth_width", SMOOTH_WIDTH),
        "hold_fraction": options.get("hold_fraction", HOLD_FRACTION),
        "min_run": options.get("min_run", 1),
        "windows_per_clip": options["windows_per_clip"], "seed": options["seed"],
        "measurable_median_speed_floor": MEASURABLE_MEDIAN_SPEED,
        "note": "re-implementation of DANCE_QUALITY_DEFECTS.md section 22; "
                "calibrated against its ground-truth reading at run time"},
        "ground_truth": truth, "arms": {},
        # THE HEADER: what the criterion refused, per arm and by name.
        "measurability": {"clips_requested": len(clips), "ground truth": {
            key: truth[key] for key in
            ("clips_measured", "clips_dropped", "dropped_clips", "absent_clips",
             "windows_drawn", "windows_measured", "windows_dropped",
             "windows_dropped_p90_over_p10")}}}
    if expect_ground_truth is not None:
        lo, hi = expect_ground_truth
        share = truth["window_share_with_sustained_hold"]
        report["calibration"] = {"expected_band": [lo, hi], "ground_truth_read": share,
                                 "passed": bool(lo <= share <= hi)}
        if not (lo <= share <= hi):
            raise SystemExit(
                "error: ground truth reads {:.3f} of windows with a sustained hold, outside "
                "the expected band [{}, {}] from section 22.  This re-implementation is not "
                "the criterion it claims to be; not scoring any arm behind it.".format(
                    share, lo, hi))
    for name, directory in arms:
        arm = score_arm(directory, clips, **options)
        arm["window_share_vs_truth"] = round(
            arm["window_share_with_sustained_hold"]
            / max(truth["window_share_with_sustained_hold"], 1e-9), 3)
        report["arms"][name] = arm
        report["measurability"][name] = {
            key: arm[key] for key in
            ("clips_measured", "clips_dropped", "dropped_clips", "absent_clips",
             "windows_drawn", "windows_measured", "windows_dropped",
             "windows_dropped_p90_over_p10")}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--clips", required=True, type=pathlib.Path)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--windows-per-clip", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--hold-fraction", type=float, default=0.25)
    parser.add_argument("--min-run", type=int, default=1)
    parser.add_argument("--smooth-width", type=int, default=SMOOTH_WIDTH)
    parser.add_argument("--expect-ground-truth", type=float, nargs=2, metavar=("LO", "HI"),
                        default=None,
                        help="refuse unless ground truth's window share lands in [LO, HI]; "
                             "section 22 reports 0.27-0.29")
    parser.add_argument("--out", type=pathlib.Path)
    arguments = parser.parse_args()
    arms = []
    for entry in arguments.arm:
        name, _, directory = entry.rpartition("=")
        if not pathlib.Path(directory).is_dir():
            raise SystemExit("error: --arm {!r} points at {!r}, which does not exist."
                             .format(name, directory))
        arms.append((name, directory))
    clips = [line.strip() for line in arguments.clips.read_text().splitlines() if line.strip()]
    report = run(arms, clips, arguments.ground_truth,
                 expect_ground_truth=arguments.expect_ground_truth,
                 windows_per_clip=arguments.windows_per_clip, seed=arguments.seed,
                 hold_fraction=arguments.hold_fraction, min_run=arguments.min_run,
                 smooth_width=arguments.smooth_width)
    text = json.dumps(report, indent=2, sort_keys=True)
    if arguments.out:
        arguments.out.write_text(text)
    summary = {"ground truth": report["ground_truth"]}
    summary.update(report["arms"])
    print("%-28s %8s %8s %10s %14s" % ("arm", "win%", "clip%", "p90/p10",
                                       "measured/drawn"))
    for name, row in summary.items():
        print("%-28s %7.1f%% %8s %10.2f %7d/%-6d" % (
            name, 100 * row["window_share_with_sustained_hold"],
            row["clips_with_sustained_hold"], row["p90_over_p10_window_median"],
            row["windows_measured"], row["windows_drawn"]))
        if row["clips_dropped"] or row["windows_dropped"]:
            print("    REFUSED %d clip(s), %d window(s) as unmeasurable: %s" % (
                row["clips_dropped"], row["windows_dropped"],
                ", ".join(sorted(row["dropped_clips"])) or "-"))


if __name__ == "__main__":
    main()
