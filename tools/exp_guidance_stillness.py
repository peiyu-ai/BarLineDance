#!/usr/bin/env python3
"""Sustained-stillness WINDOW share and velocity dynamic range, per arm.

WHY THIS FILE EXISTS.  docs/DANCE_QUALITY_DEFECTS.md 21.1-22.2 replaced the
"hold-frame share" column with "share of random 150-frame windows that contain
sustained stillness", because on a 150-frame window the ground truth's own
hold-frame share reads 0.00% and a column that reads zero on the reference has
no power (CLAUDE.md 2.1 rule 3).  The 22.1 table (ground truth 29%, training
windows 27%, shipped generation 6%) was produced by a heredoc that did not
survive, so this file re-implements the criterion as a tool with the controls
the CLAUDE.md asks for, for the guidance-weight sweep (runs/opt_guidance*).

DEFINITIONS (all from tools/measure_motion_dynamics.py, kept identical so the
whole-clip columns here agree with that tool byte for byte):

  speed            root-relative mean joint speed per frame, m/s
                   (``joint_speed``)
  low_pass(w=9)    9-frame moving average over joint positions; the width at
                   which 20.2/20.3 found ground truth keeps 1.76% held and every
                   generated arm reads 0.00%.  A momentary dip of an oscillating
                   body does not survive it; a landing does.
  hold             smoothed speed < 0.25 x the median smoothed speed OF THE SAME
                   WINDOW.  The threshold is the window's own, not the clip's:
                   the 22.1 comparison included training windows, for which no
                   clip context exists, and the criterion has to read the same
                   thing on a window whatever it was cut from.
  window share     fraction of random 150-frame windows (``--draws-per-clip``
                   per clip, fixed seed) in which at least one frame is held.

The share is reported pooled over all windows AND per clip, because the
per-clip form is what a paired test against the shipping arm needs: the arms
here share the plan and the per-clip seed, so a sign test over 20 clips is the
honest statistic, not a difference of two pooled numbers.

CONTROLS (``--with-controls`` puts them in the JSON next to the arms; the unit
tests in tests/test_exp_guidance_stillness.py pin each one):

  * constant velocity   every joint drifts at a constant speed.  Must read
                        0.00% -- there is no stillness to find.
  * move-then-hold      a body that alternates 3 s of travel with 1 s held
                        still (+ small noise).  Must read well above zero, and
                        the smoothing must NOT remove it.  This is the
                        positive control WITH THE RIGHT DIRECTION (2.1 rule 2):
                        a real landing survives the low-pass.
  * drift + oscillation a body travelling at a constant drift with a coherent
                        5 Hz oscillation twice as fast as the drift.  The
                        UNSMOOTHED hold share is > 0 (every crossing is a slow
                        frame) and the smoothed share must be 0: this is the
                        20.2 distinction between a landing and a zero crossing,
                        and a criterion that cannot tell them apart would call
                        the shipped arm's jitter "stillness".  (A pure sinusoid
                        is NOT this control -- see the function's docstring.)
  * ground truth        must read in the 22.1 band (~27-29%).  Checked at run
                        time and printed; a reading far outside it means the
                        re-implementation is not the 22.1 criterion and its
                        numbers cannot be quoted against that table.
  * frozen body         a body that never moves.  Must be REFUSED, not scored.
                        This is the input the criterion has no scale on, and
                        the shape ``_source_safe_draft`` hands back when a clip
                        has no retrieval group (23.9).  Until 2026-09-04 it read
                        ``window_share 0.0 / p90_over_p10 0.0 / sustained_hold
                        0.0`` here -- "completely still" printed as "no
                        stillness".

WHAT THIS TOOL REFUSES, AND WHAT THAT CHANGED (2026-09-04).  The definition and
the refusal now live in ``tools/stillness_criterion.py``, shared with
``measure_motion_dynamics`` and ``exp_draftonly_stillness_windows`` -- the three
agreed on the definition and disagreed on the degenerate input, one dropping the
clip in silence and two reporting a number.  On the 20 T-line eval clips the
retrieval draft has FOUR clips and 199 of 1000 sampled windows the criterion
cannot measure, and every one of those windows used to sit in the denominator of
``window_share_pooled`` as a window WITHOUT stillness.  Re-read over the
measurable windows the draft's pooled share is 0.438, not the published 0.351;
ground truth and all four generated arms have nothing to refuse and read
byte-identically to the published values, which is the positive control on the
change itself.

Also reported, per arm: whole-clip p90/p10 of raw speed (the 21.2 power-proven
column), median over the same random windows of the window-level p90/p10, the
whole-clip sustained hold share and longest sustained hold (20.3), and paired
sign tests of each arm against ``--reference``.
"""

import argparse
import json
import math
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

# joint_speed / low_pass / longest_run / held_frames / dynamic_range are
# imported, not re-declared: they WERE copies of measure_motion_dynamics's, and
# the copies had drifted on exactly the question this tool got wrong.
#
# OLD BEHAVIOUR of ``held_frames`` here: ``if speed.size < 3 or median <= 0:
# return np.zeros(shape, bool)`` -- a body that never moves has median 0 and so
# was reported as having NO held frames.  On runs/txy_t_draft_ep12_index that
# is four of the twenty eval clips (two of them motionless end to end, the
# all-zero draft of DANCE_QUALITY_DEFECTS 23.9) and 199 of the 1000 sampled
# windows, every one of them counted in the denominator of the draft's
# ``window_share_pooled`` as a window WITHOUT stillness.
# OLD BEHAVIOUR of ``dynamic_range``: ``p90 / max(p10, 1e-9)``, which turned a
# vanished p10 into a ratio of ~1e11 rather than a refusal; 64 further draft
# windows hit that clamp.
# NEW BEHAVIOUR: both raise ``DegenerateMotion``, and the window or clip is
# recorded as dropped in the JSON header.


def window_starts(n_frames, rng, draws, window=WINDOW):
    """Random window starts; a clip shorter than the window yields none."""
    if n_frames < window:
        return np.zeros(0, int)
    return rng.integers(0, n_frames - window + 1, size=draws)


def score_windows(joints, starts, window=WINDOW, smooth=SMOOTH_WIDTH,
                  hold_fraction=HOLD_FRACTION):
    """Per window: does it contain sustained stillness; its raw p90/p10.

    A window the criterion cannot measure is NOT scored as "no stillness"; it
    is left out of both arrays and counted in the returned ``dropped`` counts,
    which reach the JSON.  The stillness column and the p90/p10 column
    degenerate on different statistics (the smoothed median and the raw p10),
    so they are dropped independently and counted separately.
    """
    joints = np.asarray(joints, float)
    has_hold, ranges = [], []
    dropped_hold, dropped_range = 0, 0
    for start in starts:
        piece = joints[int(start):int(start) + window]
        smoothed_speed = joint_speed(low_pass(piece, smooth))
        try:
            has_hold.append(bool(held_frames(smoothed_speed, hold_fraction).any()))
        except DegenerateMotion:
            dropped_hold += 1
        try:
            ranges.append(dynamic_range(joint_speed(piece)))
        except DegenerateMotion:
            dropped_range += 1
    return (np.asarray(has_hold, bool), np.asarray(ranges, float),
            dropped_hold, dropped_range)


def _or_none(function, *arguments, **keywords):
    """Value, or None when the criterion refuses -- never a fabricated 0.0."""
    try:
        return function(*arguments, **keywords)
    except DegenerateMotion:
        return None


def score_clip(joints, rng, draws, **options):
    joints = np.asarray(joints, float)
    starts = window_starts(len(joints), rng, draws)
    has_hold, ranges, dropped_hold, dropped_range = score_windows(
        joints, starts, **options)
    raw = joint_speed(joints)
    smoothed = joint_speed(low_pass(joints, options.get("smooth", SMOOTH_WIDTH)))
    held = _or_none(held_frames, smoothed,
                    options.get("hold_fraction", HOLD_FRACTION))
    raw_held = _or_none(held_frames, raw)
    return {
        "frames": int(len(joints)),
        "windows": int(len(starts)),
        "windows_measured": int(len(has_hold)),
        "windows_dropped": int(dropped_hold),
        "windows_dropped_p90_over_p10": int(dropped_range),
        "windows_with_stillness": int(has_hold.sum()),
        # Denominator is the windows the criterion COULD measure.  Old code put
        # every degenerate window in the denominator as a "no stillness" window.
        "window_share": (float(has_hold.mean()) if len(has_hold) else None),
        "window_p90_over_p10_median": (float(np.median(ranges))
                                       if len(ranges) else None),
        "clip_p90_over_p10": _or_none(dynamic_range, raw),
        "clip_sustained_hold_share": (None if held is None
                                      else float(held.mean())),
        "clip_sustained_longest_hold_s": (None if held is None
                                          else longest_run(held) / FPS),
        "clip_raw_hold_share": (None if raw_held is None
                                else float(raw_held.mean())),
        "clip_measurable": held is not None,
    }


def load(path):
    return np.asarray(pickle.load(open(path, "rb"))["full_pose"], float)


def score_arm(directory, clips, seed, draws, **options):
    """One arm.  The rng is re-seeded per CLIP from (seed, clip index) so every
    arm draws the SAME window starts for the same clip -- the window sample is
    part of what makes two arms comparable, exactly like the per-clip seed.

    A clip whose whole-clip track the criterion refuses is recorded in
    ``dropped_clips`` and left out of every clip-level median.  OLD BEHAVIOUR:
    it was scored 0.0 on all three stillness columns and pooled in."""
    log = DropLog("clips")
    per_clip, missing = {}, []
    for index, clip in enumerate(clips):
        path = pathlib.Path(directory) / (clip + ".pkl")
        if not path.exists():
            missing.append(clip)
            continue
        rng = np.random.default_rng([seed, index])
        row = score_clip(load(path), rng, draws, **options)
        per_clip[clip] = row
        if row["clip_measurable"]:
            log.measured.append(clip)
        else:
            log.drop(clip, "whole-clip median smoothed speed <= {:.0e} m/s "
                           "(motionless body: the self-relative hold threshold "
                           "collapses to zero)".format(MEASURABLE_MEDIAN_SPEED))
    if not per_clip:
        raise SystemExit("error: 0 of {} clips scored from {}; nothing is reported."
                         .format(len(clips), directory))
    if not log.measured:
        raise SystemExit(
            "error: every one of the {} clips found in {} is unmeasurable by "
            "the sustained-stillness criterion; refusing to report a number.\n{}"
            .format(len(per_clip), directory, "\n".join(log.lines())))
    return summarise(per_clip) | {"missing": missing,
                                  "per_clip": per_clip} | log.header()


def summarise(per_clip):
    """Pooled and per-clip medians over the MEASURABLE part only.

    ``windows`` was the denominator of ``window_share_pooled`` and counted every
    drawn window, degenerate ones included -- on the retrieval draft that put
    199 motionless windows in the denominator as windows without stillness.  It
    is now ``windows_measured``, and ``windows_dropped`` sits beside it."""
    rows = [r for r in per_clip.values() if r["windows"]]
    drawn = sum(r["windows"] for r in rows)
    measured = sum(r["windows_measured"] for r in rows)
    dropped = sum(r["windows_dropped"] for r in rows)
    with_still = sum(r["windows_with_stillness"] for r in rows)
    def med(key):
        values = [r[key] for r in rows if r[key] is not None]
        return float(np.median(values)) if values else None
    return {
        "clips": len(per_clip),
        "clips_with_windows": len(rows),
        "windows_drawn": drawn,
        "windows_measured": measured,
        "windows_dropped": dropped,
        "windows_dropped_p90_over_p10": sum(r["windows_dropped_p90_over_p10"]
                                            for r in rows),
        "window_share_pooled": with_still / measured if measured else None,
        "window_share_clip_median": med("window_share"),
        "clips_with_any_still_window": int(sum(r["windows_with_stillness"] > 0
                                             for r in rows)),
        "window_p90_over_p10_median": med("window_p90_over_p10_median"),
        "clip_p90_over_p10_median": med("clip_p90_over_p10"),
        "clip_sustained_hold_share_median": med("clip_sustained_hold_share"),
        "clip_sustained_longest_hold_s_median": med("clip_sustained_longest_hold_s"),
        "clip_raw_hold_share_median": med("clip_raw_hold_share"),
    }


def sign_test(a, b):
    """Two-sided exact sign test on paired values; ties are dropped."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a - b
    n = int((diff != 0).sum())
    wins = int((diff > 0).sum())
    if n == 0:
        return {"n": 0, "wins": 0, "p": 1.0}
    k = min(wins, n - wins)
    p = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"n": n, "wins": wins, "p": p}


def paired(arm, reference, key):
    common = [c for c in arm["per_clip"] if c in reference["per_clip"]
              and arm["per_clip"][c][key] is not None
              and reference["per_clip"][c][key] is not None]
    a = [arm["per_clip"][c][key] for c in common]
    b = [reference["per_clip"][c][key] for c in common]
    out = sign_test(a, b)
    out["clips"] = len(common)
    out["median_delta"] = float(np.median(np.asarray(a) - np.asarray(b))) if common else None
    return out


# ------------------------------------------------------------------ CONTROLS

def synthetic_constant_velocity(frames=600, joints=24, seed=0):
    """Every joint drifts at its own constant velocity: no stillness exists."""
    rng = np.random.default_rng(seed)
    velocity = rng.normal(size=(joints, 3)) * 0.01
    velocity[0] = 0.0  # root fixed so root-relative speed is the joints' own
    t = np.arange(frames)[:, None, None]
    return t * velocity[None]


def synthetic_move_then_hold(frames=600, joints=24, seed=0, move_s=3.0, hold_s=1.0,
                             noise=0.0005):
    """Travel for ``move_s`` seconds, hold still for ``hold_s``, repeat."""
    rng = np.random.default_rng(seed)
    period = int((move_s + hold_s) * FPS)
    move = int(move_s * FPS)
    velocity = rng.normal(size=(joints, 3)) * 0.01
    velocity[0] = 0.0
    out = np.zeros((frames, joints, 3))
    position = np.zeros((joints, 3))
    for f in range(frames):
        if (f % period) < move:
            position = position + velocity
        out[f] = position
    return out + rng.normal(size=out.shape) * noise


def synthetic_drift_plus_oscillation(frames=600, joints=24, seed=0, hz=4.7,
                                     drift=0.005, ratio=2.0):
    """A body travelling at a constant drift with a coherent oscillation
    whose velocity amplitude is ``ratio`` x the drift.  Raw speed is
    |1 + ratio*cos| x drift: it dips to zero at every crossing, so the RAW hold
    share is > 0 (about 10% at ratio 2).  A 9-frame box filter attenuates a
    ~5 Hz component at 30 fps by |sin(4.5w)/(9 sin(w/2))| = 0.20, so the smoothed
    oscillation is 0.4 x the drift and the speed never drops below 0.6 x its
    median: no held frames.  This is the 20.2 finding in synthetic form.

    ``hz`` IS 4.7 AND NOT 5.0, and that is the whole control rather than a
    detail.  5 Hz at 30 fps is exactly 6 frames per cycle, so the discrete
    signal only ever samples SIX phases, the same six in every cycle -- and
    none of them lands near the zero crossing.  Measured at hz=5.0 the raw
    speed bottoms out at 0.654 x its median and the raw hold share reads
    0.0000, i.e. the control asserted "this body has slow frames" while
    containing none, and could not have caught a criterion that mistook
    oscillation for stillness.  4.7 Hz is incommensurate with 30 fps
    (6.383 frames per cycle), so the sampled phase sweeps the crossing:
    raw hold share 0.1002, smoothed hold share 0.0000, which is the gap the
    control exists to demonstrate.  (5.3 and 3.7 Hz behave the same way;
    4.3 leaves a residual 0.0017 smoothed and is therefore weaker.)

    A PURE sinusoid is not this control: with no drift the relative threshold is
    scale-invariant and the smoothed body still reverses at every extreme, so
    the smoothed share stays at 100% -- which is correct for a pendulum (it
    does stop at its turning points) and useless as a jitter model."""
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=(joints, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    direction[0] = 0.0
    omega = 2 * np.pi * hz / FPS  # radians per frame
    t = np.arange(frames, dtype=float)
    profile = drift * (t + (ratio / omega) * np.sin(omega * t))
    return profile[:, None, None] * direction[None]


def synthetic_frozen(frames=600, joints=24):
    """A body that never moves.  THE REFUSAL CONTROL.

    This is the shape of ``_source_safe_draft``'s all-zero draft
    (DANCE_QUALITY_DEFECTS 23.9): every frame identical.  It must be REFUSED,
    never scored.  Before 2026-09-04 it read ``window_share 0.0 /
    p90_over_p10 0.0 / sustained_hold 0.0`` here -- "completely still" reported
    as "no stillness" -- while measure_motion_dynamics dropped it in silence."""
    return np.zeros((frames, joints, 3))


def control_readings(seed=0, draws=200):
    out = {}
    for name, maker in (("constant_velocity", synthetic_constant_velocity),
                        ("move_then_hold", synthetic_move_then_hold),
                        ("drift_plus_oscillation", synthetic_drift_plus_oscillation),
                        ("frozen_body", lambda seed=0: synthetic_frozen())):
        joints = maker(seed=seed)
        rng = np.random.default_rng([seed, 999])
        row = score_clip(joints, rng, draws)
        # The unsmoothed window share, so the oscillation control can show the
        # gap between "has slow frames" and "has sustained stillness".
        starts = window_starts(len(joints), np.random.default_rng([seed, 999]), draws)
        raw_has = [_or_none(lambda s=s: bool(held_frames(
            joint_speed(joints[int(s):int(s) + WINDOW])).any())) for s in starts]
        kept = [v for v in raw_has if v is not None]
        row["window_share_unsmoothed"] = float(np.mean(kept)) if kept else None
        row["windows_unmeasurable_unsmoothed"] = len(raw_has) - len(kept)
        out[name] = row
    return out


# ---------------------------------------------------------------------- MAIN

def run(arms, clips, ground_truth_dir, seed, draws, reference=None,
        with_controls=False):
    truth = score_arm(ground_truth_dir, clips, seed, draws)
    report = {"seed": seed, "draws_per_clip": draws, "window": WINDOW,
              "smooth_width": SMOOTH_WIDTH, "hold_fraction": HOLD_FRACTION,
              "measurable_median_speed_floor": MEASURABLE_MEDIAN_SPEED,
              "ground_truth": truth, "arms": {}, "paired_vs_reference": {},
              "paired_vs_ground_truth": {},
              # THE HEADER.  Which clips and windows the criterion refused, per
              # arm and by name.  Without it a pooled share cannot be read: the
              # published draft numbers 0.351 / 0.390 were shares whose
              # denominator held 199 windows the criterion could not measure.
              "measurability": {"clips_requested": len(clips)}}
    gt_share = truth["window_share_pooled"]
    report["ground_truth_in_22_1_band"] = (gt_share is not None
                                           and 0.22 <= gt_share <= 0.36)
    for name, directory in arms:
        report["arms"][name] = score_arm(directory, clips, seed, draws)
    for who, arm in [("ground truth", truth)] + list(report["arms"].items()):
        report["measurability"][who] = {
            key: arm[key] for key in
            ("clips_measured", "clips_dropped", "dropped_clips", "missing",
             "windows_drawn", "windows_measured", "windows_dropped",
             "windows_dropped_p90_over_p10")}
    for name, arm in report["arms"].items():
        report["paired_vs_ground_truth"][name] = {
            key: paired(arm, truth, key)
            for key in ("window_share", "clip_p90_over_p10")}
    if reference and reference in report["arms"]:
        ref = report["arms"][reference]
        for name, arm in report["arms"].items():
            if name == reference:
                continue
            report["paired_vs_reference"][name] = {
                key: paired(arm, ref, key)
                for key in ("window_share", "clip_p90_over_p10",
                            "window_p90_over_p10_median")}
        report["reference"] = reference
    if with_controls:
        report["controls"] = control_readings(seed=seed)
    return report


def print_table(report):
    def row(name, r):
        line = ("{:28s} windows {:5d}/{:<5d}  still-window share {:6.1%} "
                "(clip median {:6.1%}, {:2d}/{:2d} clips any)  "
                "p90/p10 clip {:5.2f} window {:5.2f}  "
                "sustained hold {:5.2%} / {:.2f}s").format(
            name, r["windows_measured"], r["windows_drawn"],
            r["window_share_pooled"], r["window_share_clip_median"],
            r["clips_with_any_still_window"], r["clips_with_windows"],
            r["clip_p90_over_p10_median"], r["window_p90_over_p10_median"],
            r["clip_sustained_hold_share_median"],
            r["clip_sustained_longest_hold_s_median"])
        if r["clips_dropped"] or r["windows_dropped"]:
            line += ("\n    REFUSED {} clip(s) and {} window(s) as unmeasurable"
                     " ({} windows have no p90/p10, a weaker degeneracy that"
                     " includes those): {}").format(
                r["clips_dropped"], r["windows_dropped"],
                r["windows_dropped_p90_over_p10"],
                ", ".join(sorted(r["dropped_clips"])) or "-")
        return line
    print(row("ground truth", report["ground_truth"]))
    print("  ground truth inside the 22.1 band (22-36%): {}".format(
        report["ground_truth_in_22_1_band"]))
    for name, arm in report["arms"].items():
        print(row(name, arm))
        vs = report["paired_vs_reference"].get(name)
        if vs:
            s = vs["window_share"]
            d = vs["clip_p90_over_p10"]
            print("    vs {}: still-window share up on {}/{} clips (p={:.3f}, median delta {:+.3f}); "
                  "p90/p10 up on {}/{} (p={:.3f}, median delta {:+.3f})".format(
                      report["reference"], s["wins"], s["n"], s["p"], s["median_delta"],
                      d["wins"], d["n"], d["p"], d["median_delta"]))
    if "controls" in report:
        for name, c in report["controls"].items():
            if c["window_share"] is None:
                print("  control {:20s} REFUSED: {} of {} windows unmeasurable "
                      "(this is the pass condition for the frozen body)".format(
                          name, c["windows_dropped"], c["windows"]))
                continue
            print("  control {:20s} still-window share smoothed {:6.1%}  unsmoothed {:6.1%}".format(
                name, c["window_share"], c["window_share_unsmoothed"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--clips", required=True, type=pathlib.Path)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--reference", default=None,
                        help="arm NAME every other arm is paired against")
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--draws-per-clip", type=int, default=50)
    parser.add_argument("--with-controls", action="store_true")
    parser.add_argument("--out", type=pathlib.Path)
    args = parser.parse_args()
    arms = [tuple(a.rsplit("=", 1)) for a in args.arm]
    clips = [l.strip() for l in args.clips.read_text().splitlines() if l.strip()]
    report = run(arms, clips, args.ground_truth, args.seed, args.draws_per_clip,
                 reference=args.reference, with_controls=args.with_controls)
    print_table(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
