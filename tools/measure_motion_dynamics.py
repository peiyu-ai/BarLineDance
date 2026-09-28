"""Does the dancer ever LAND?  Holds, bursts, and speed dynamic range.

The operator's report is "no move is ever executed to completion, nothing lands,
and it does not sit on the music".  Those are three descriptions of one physical
property: real dancing alternates between travelling fast and being still, and
the stillness is what a beat can be landed on.  A body that moves continuously
at a middling speed has nothing to place on a beat, and reads as small and
fidgety however large its average excursion is.

MEASURED (20 held-out clips, 2026-09-03).  Ground truth spends 2.1% of frames
nearly still and its longest single stillness is 0.27 s.  The shipped generation
spends 0.5% and its longest stillness in a fourteen-second clip is **0.05 s** --
the body never stops at all.  The retrieval draft, before the completion model
touches it, spends 4.9% and holds for 0.58 s.  So the material has the landings
and the completion removes them.

DEFINITIONS, and why each is relative rather than absolute.  A "hold" is a frame
whose speed is below a fraction of THAT CLIP's median speed: an absolute
threshold would call a calm song's whole clip a hold and a fast song's none.
``p90_over_p10`` is the same idea for the whole distribution -- dance has bursts
and stillness, mush has neither -- and it is scale free for the same reason.

``longest_hold_seconds`` is reported beside ``hold_share`` because they fail
differently: a body that stutters to a stop for one frame thirty times has the
same share as one that lands once and holds, and only the second is dancing.

WHAT IT REFUSES.  Being relative is what makes the criterion scale free and it
is also where it breaks: at a median speed of zero the hold threshold is zero
and a MOTIONLESS body reads ``hold_share = 0.0``.  The definition and the
refusal now live in ``tools/stillness_criterion.py`` so that this tool and
``exp_guidance_stillness`` / ``exp_draftonly_stillness_windows`` cannot drift
apart again -- until 2026-09-04 this tool silently DROPPED such clips (16 of 20
on ``runs/txy_t_draft_ep12_index``) while ``exp_guidance_stillness`` silently
printed 0.0 for the same four.  Dropped clips are now named in the JSON header
(``clips_measured`` / ``clips_dropped`` / ``dropped_clips``).
"""

import argparse
import json
import pathlib
import pickle
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools import arm_sourcing  # noqa: E402
from tools.stillness_criterion import (FPS, HOLD_FRACTION,  # noqa: E402
                                       MEASURABLE_MEDIAN_SPEED,
                                       SMOOTH_WIDTH, DegenerateMotion, DropLog,
                                       dynamic_range, held_frames, joint_speed,
                                       longest_run, low_pass, speed_scale)

# FPS / SMOOTH_WIDTH / joint_speed / low_pass / longest_run now live in
# tools/stillness_criterion.py, which is the single definition of this
# criterion and of what it refuses; they are re-exported here because callers
# and tests import them from this module.  SMOOTH_WIDTH is still nine: that is
# where the separation is cleanest in the measurement that motivated the
# sustained column (ground truth keeps 1.76% of frames held after it, every
# generated arm reads 0.00%).


def dynamics(speed, *, hold_fraction=HOLD_FRACTION, burst_fraction=2.0):
    """Hold/burst/dynamic-range for one speed track.

    REFUSES (raises ``DegenerateMotion``) rather than returning a number when
    the clip has no speed scale of its own -- see stillness_criterion.  OLD
    BEHAVIOUR: on a body whose median smoothed speed was 0 (the all-zero draft
    ``_source_safe_draft`` returns when a clip has no retrieval group, and any
    draft frozen for more than half its frames) this returned ``None`` and
    ``score`` skipped the clip WITHOUT SAYING SO, printing a smaller ``clips:``
    count with no names.  NEW BEHAVIOUR: it raises, ``score`` records the clip
    in a ``DropLog``, and the JSON header names it.
    """
    speed = np.asarray(speed, float)
    median = speed_scale(speed)
    held = held_frames(speed, hold_fraction)
    try:
        # None, not a number, when p10 has vanished.  OLD BEHAVIOUR: divided by
        # ``max(p10, 1e-9)``, so a draft frozen for a tenth of its frames
        # reported a dynamic range of ~1e11 fabricated out of the clamp -- that
        # is what pushed the published draft ``p90_over_p10`` median to 17.96
        # where the measurable clips read 8.37.
        #
        # This degeneracy is NOT the hold criterion's: p90/p10 needs a nonzero
        # p10, the hold threshold needs a nonzero MEDIAN, and a clip can lose
        # the first while keeping the second (five of the twenty draft clips
        # do).  Losing p90/p10 therefore drops that COLUMN, not the clip --
        # otherwise this tool would refuse nine clips where the two window
        # tools refuse four, which is the disagreement all of this is fixing.
        ratio = dynamic_range(speed)
    except DegenerateMotion:
        ratio = None
    return {
        "hold_share": float(held.mean()),
        "burst_share": float((speed > burst_fraction * median).mean()),
        "p90_over_p10": ratio,
        "longest_hold_seconds": longest_run(held) / FPS,
    }


def load(path):
    return np.asarray(pickle.load(open(path, "rb"))["full_pose"], float)


def score(directory, clips, **options):
    """Median of the columns over the clips this criterion can measure.

    Every clip that is not measured leaves a name behind: ``absent_clips`` for
    a missing pkl, ``dropped_clips`` for one the criterion refused.  OLD
    BEHAVIOUR: both were silent ``continue``s and only the surviving count was
    printed, so ``clips: 16`` out of 20 asked-for clips looked like a complete
    reading of a 16-clip set.
    """
    log = DropLog("clips")
    absent, rows = [], []
    for clip in clips:
        path = pathlib.Path(directory) / (clip + ".pkl")
        if not path.exists():
            absent.append(clip)
            continue
        joints = load(path)
        try:
            measured = dynamics(joint_speed(joints), **options)
            # THE DECIDING COLUMN.  Measured on smoothed motion, where ground
            # truth keeps 1.76% and every generated arm so far reads 0.00% --
            # the two do not overlap, while the unsmoothed 2.10% vs 0.49%
            # still could.  A clip whose SMOOTHED track is degenerate is
            # dropped whole rather than given 0.0 for these two columns, which
            # is what the old code did.
            smoothed = dynamics(joint_speed(low_pass(joints, SMOOTH_WIDTH)),
                                **options)
        except DegenerateMotion as refusal:
            log.drop(clip, refusal)
            continue
        measured["sustained_hold_share"] = smoothed["hold_share"]
        measured["sustained_longest_hold"] = smoothed["longest_hold_seconds"]
        log.measured.append(clip)
        rows.append(measured)
    if not rows:
        raise SystemExit(
            "error: 0 of {} clips scored from {} ({} absent, {} refused as "
            "unmeasurable).  Nothing was measured, so nothing is reported.\n{}"
            .format(len(clips), directory, len(absent), len(log.dropped),
                    "\n".join(log.lines())))
    def median(key):
        values = [r[key] for r in rows if r[key] is not None]
        return float(np.median(values)) if values else None
    return {key: median(key) for key in rows[0]} | {
        "clips": len(rows), "absent_clips": absent,
        "clips_measured_names": list(log.measured),
        # The p90/p10 column has its own denominator: see ``dynamics``.
        "clips_measured_p90_over_p10": sum(1 for r in rows
                                           if r["p90_over_p10"] is not None),
    } | log.header()


def run(arms, clips, ground_truth_dir, sourcing=None, **options):
    truth = score(ground_truth_dir, clips, **options)
    report = {"ground_truth": truth, "arms": {},
              "measurability": {"clips_requested": len(clips),
                                "floor_m_per_s": MEASURABLE_MEDIAN_SPEED,
                                "ground_truth": {
                                    k: truth[k] for k in
                                    ("clips_measured", "clips_dropped",
                                     "dropped_clips", "absent_clips")}}}
    if sourcing is not None:
        report["sourcing"] = sourcing
    for name, directory in arms:
        arm = score(directory, clips, **options)
        report["measurability"][name] = {
            k: arm[k] for k in ("clips_measured", "clips_dropped",
                                "dropped_clips", "absent_clips")}
        arm["hold_share_vs_truth"] = round(
            arm["hold_share"] / truth["hold_share"], 3) if truth["hold_share"] else None
        arm["dynamic_range_vs_truth"] = round(
            arm["p90_over_p10"] / truth["p90_over_p10"], 3) if (
                arm["p90_over_p10"] and truth["p90_over_p10"]) else None
        # THE MATCHED RATIO.  ``hold_share_vs_truth`` above divides this arm's
        # median over the clips IT could measure by ground truth's median over
        # the clips GROUND TRUTH could measure -- different sets whenever the
        # arm dropped anything, which is the mismatch this tool's header
        # comment already describes for the sourcing case (an 18-clip numerator
        # over a 20-clip denominator read 4.13 where the matched comparison was
        # 5.582).  This one re-scores ground truth on exactly the arm's
        # measured clips, so numerator and denominator are the same set.
        if arm["clips_dropped"]:
            matched = score(ground_truth_dir, arm["clips_measured_names"], **options)
            arm["ground_truth_on_matched_clips"] = {
                key: matched[key] for key in
                ("hold_share", "sustained_hold_share", "p90_over_p10",
                 "clips_measured")}
            arm["hold_share_vs_truth_matched"] = round(
                arm["hold_share"] / matched["hold_share"], 3
            ) if matched["hold_share"] else None
        else:
            arm["hold_share_vs_truth_matched"] = arm["hold_share_vs_truth"]
        report["arms"][name] = arm
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--clips", required=True, type=pathlib.Path)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--hold-fraction", type=float, default=HOLD_FRACTION)
    parser.add_argument("--burst-fraction", type=float, default=2.0)
    parser.add_argument("--out", type=pathlib.Path)
    arm_sourcing.add_arguments(parser)
    arguments = parser.parse_args()

    arms = []
    for entry in arguments.arm:
        name, _, directory = entry.rpartition("=")
        if not pathlib.Path(directory).is_dir():
            raise SystemExit("error: --arm {!r} points at {!r}, which does not "
                             "exist.".format(name, directory))
        arms.append((name, directory))
    requested = [line.strip() for line in arguments.clips.read_text().splitlines()
                 if line.strip()]
    # See tools/arm_sourcing.py.  MEASURED 2026-09-04 on runs/txy_t_m6_draft:
    # this tool ALREADY dropped the two unsourced clips from that arm, silently
    # and for an unrelated reason -- their draft is a frozen body, median speed
    # 0, so ``dynamics`` returned None -- and printed ``clips: 18`` while still
    # scoring GROUND TRUTH on all 20.  ``hold_share_vs_truth`` was therefore an
    # 18-clip numerator over a 20-clip denominator: 4.13 where the matched
    # comparison is 5.582.  Excluding by SOURCING drops the same clips from both
    # sides, and says which ones.
    clips, sourcing = arm_sourcing.select_clips(
        requested, arms, threshold=arguments.sourcing_threshold,
        include_unsourced=arguments.include_unsourced)
    for line in arm_sourcing.format_header(sourcing):
        print(line)
    report = run(arms, clips, arguments.ground_truth, sourcing=sourcing,
                 hold_fraction=arguments.hold_fraction,
                 burst_fraction=arguments.burst_fraction)
    # The drops are printed, not only written: a reader of the terminal has to
    # see that the medians below are over fewer clips than were asked for.
    for who, block in report["measurability"].items():
        if isinstance(block, dict) and block.get("clips_dropped"):
            print("{}: measured {}, REFUSED {} as unmeasurable".format(
                who, block["clips_measured"], block["clips_dropped"]))
            for clip, reason in block["dropped_clips"].items():
                print("    {}  ({})".format(clip, reason))
    text = json.dumps(report, indent=2, sort_keys=True)
    if arguments.out:
        arguments.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
