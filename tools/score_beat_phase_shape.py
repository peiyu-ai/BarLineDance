#!/usr/bin/env python3
"""WHERE in the beat the body is slowest -- signed, so settling and lurching differ.

WHY A SECOND PHASE TOOL.  ``tools/score_beat_phase_profile.py`` reports
``modulation``, twice the magnitude of the profile's first Fourier coefficient.
A magnitude answers "is there once-per-beat structure" and cannot answer "is it
the RIGHT once-per-beat structure", and on 2026-09-01 that gap turned into a
disqualification: the time-reversal negative control PASSED that column
(P=0.021).  Every generated arm also carries strong once-per-beat structure --
stronger than ground truth's -- while looking wrong, because its structure is
inverted.  This file measures the sign.

--------------------------------------------------------------- THE MEASUREMENT

Per clip, over sliding windows of ``--window-beats`` beats (the locality the
operator required: "同舞者每一段都可能有区别"), z-score the body speed inside the
window and bin its frames by beat phase, phase 0 = the beat.  Average those
window tables BEAT-ALIGNED -- never aligned to each window's own extremum, which
is a mistake this file's first draft made and which manufactures a trough out of
noise: self-aligned, ground truth, a generated arm and a shuffled control all
read peak-to-trough 0.76.

Two numbers come out of the aligned profile:

    settle = -profile[beat bin]   positive => the body is SLOWEST on the beat
    depth  = max - min            how pronounced the cycle is, in sd units

``settle`` is the column that separates settling from lurching; ``depth`` alone
cannot, and is reported beside it precisely so an arm cannot buy a good reading
by being more violent.

------------------------------------------------------------------- PROVENANCE

MEASURED, not invented -- and that distinction is the point of §2.1, so it is
stated plainly.  It was not derived from the paper.  On 88 ground-truth clips
the aligned profile has its minimum AT the beat (-0.0472 sd) and its maximum at
phase 0.58-0.67 (+0.0434/+0.0438), peak-to-trough 0.0909 = 2.85x the foreign-grid
null's 0.0319.  The paper is consistent with it but does not state it: it asks
segmentation to cover "complete motion processes ... a kick, which comprises the
preparatory weight shift, leg extension, and recovery", i.e. movements that
RESOLVE, and a movement that resolves on the beat is slowest there.

-------------------------------------------------------------------- CONTROLS

The four this instrument is allowed to make judgements behind, and what each is
for.  Run them with ``--controls``.

  * foreign grid (the null, always on) -- the same motion against other clips'
    beat times.  Must read ``settle`` ~ 0.  This is what makes ``settle`` a
    statement about the MUSIC's beats rather than about motion in general.
  * half-beat rotation -- ground truth's own speed trace rolled by half a beat
    period.  Must FLIP THE SIGN, from about +0.047 to about -0.047.  This is the
    synthetic form of the exact defect being hunted, so it is the sharpest
    check that the column can read the defect at all: a statistic that cannot
    produce a negative number here cannot detect a lurching arm (§2.1 rule 3,
    prove the measurement has power before reporting a zero).
  * block shuffle -- must collapse ``settle`` and ``depth`` toward the null.
  * time reversal -- the control that disqualified ``modulation`` (it passed
    there at P=0.021).  Here it FAILS: settle -0.0299, 40/93, P=0.93.
    PREDICTION CORRECTED.  This docstring first said reversal was not expected
    to fail, reasoning that a reversed dance still lands on the beats and that a
    trough at phase 0 is nearly reversal-symmetric.  Measurement says otherwise
    and the mechanism is plainer than the reasoning: ``reversed_speed`` reverses
    the TRACE and leaves the beat grid where it was, so beat b_i no longer marks
    the moment the dancer settled -- it is a wrong-grid control wearing a
    time-reversal label, and the profile collapses to flat (largest bin 0.027
    against ground truth's 0.061).  It is reported as a control that fails, not
    as evidence that the column is direction-sensitive; the column's direction
    sensitivity is what the half-beat roll shows.

The clip is the unit and the win rate is the headline; the pooled profile is
printed for shape only.  Two dancers who put the emphasis in different places
must not be averaged into a flat line -- that is why ``settle`` is taken per
clip against that clip's OWN nulls before anything is pooled.
"""
import argparse
import json
import pathlib
import sys
from math import comb

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools import arm_sourcing  # noqa: E402

from tools.score_beat_phase_profile import (BINS, NULL_GRIDS, PARTS,  # noqa: E402
                                            WINDOW_BEATS, beat_period, beat_phase,
                                            load_clip, profile, rescale_grid,
                                            reversed_speed, shuffled_speed)


def aligned_profile(speed, beats, window_beats=WINDOW_BEATS, bins=BINS):
    """Mean of the per-window beat-phase tables, aligned to the BEAT.

    Aligned to the beat and not to each window's own minimum.  Self-alignment
    forces a V out of any trace at all: measured, ground truth, a generated arm,
    a block-shuffled control and pure noise all read peak-to-trough ~0.76 under
    it, i.e. the statistic reads its own construction.
    """
    beats = np.asarray(beats)
    tables = []
    for start in range(0, len(beats) - window_beats):
        first, last = int(beats[start]), int(beats[start + window_beats])
        if last > len(speed) or last - first < bins * 3:
            continue
        table, _, _ = profile(speed[first:last],
                              beat_phase(beats[start:start + window_beats + 1] - first,
                                         last - first),
                              bins)
        if table is not None:
            tables.append(table)
    if len(tables) < 2:
        return None
    return np.mean(np.asarray(tables), axis=0)


def shape_of(table):
    """``settle`` and ``depth`` from an aligned profile."""
    return {"settle": float(-table[0]),
            "depth": float(table.max() - table.min()),
            "trough_bin": int(np.argmin(table)),
            "peak_bin": int(np.argmax(table))}


def part_order_rho(arm_parts, truth_parts, parts):
    """How well an arm reproduces GROUND TRUTH'S ORDERING OF THE BODY PARTS.

    THE COLUMN THIS FILE WAS MISSING.  ``settle`` pooled over the whole body
    saturates: measured 2026-09-12 on the 20 eval clips, the shipped arm reads
    +0.1387 and the aligned arm +0.0731 against ground truth's +0.0594 -- both
    arms BEAT ground truth, so the column can no longer separate them and every
    experiment judged on it was judged by a ruler already pinned at the top.

    The separation survives one level down.  Ground truth lands the beat with
    the LOWER BODY -- feet 0.0943, hips_knees 0.0649, torso 0.0477, hands
    0.0233, shoulders 0.0207 -- while the aligned arm inverts it: shoulders
    0.1201, torso 0.1085, hands 0.0782, feet 0.0339, hips_knees 0.0093.  The
    beat is being carried by the arms, which is the operator's "举手太多" and
    "没有支撑脚" in numbers.

    Spearman rho over the parts, so it reads the ORDER and not the magnitudes:
    an arm that brakes three times as hard as a dancer but in the dancer's order
    scores 1.0, and an arm that matches magnitudes with the order inverted
    scores -1.0.  Magnitude is judged separately, by the parts themselves.

    Only parts that passed the half-beat flip control are used; see
    ``validated_parts``.
    """
    from scipy.stats import spearmanr

    pairs = [(arm_parts[k], truth_parts[k]) for k in parts
             if k in arm_parts and k in truth_parts
             and arm_parts[k] == arm_parts[k] and truth_parts[k] == truth_parts[k]]
    if len(pairs) < 3:
        return float("nan")
    rho = spearmanr([a for a, _ in pairs], [t for _, t in pairs]).statistic
    return float(rho)


def validated_parts(truth, rotated, parts):
    """The parts whose sign flips when ground truth is rolled half a beat.

    A part that does not flip is not measuring beat phase on this corpus, and
    including it in ``spread`` or in the ordering silently mixes a broken
    channel into a judgement.  Measured 2026-09-12: 5 of 6 flip, ELBOWS does
    not, so elbows is dropped rather than printed-and-used.
    """
    return [k for k in parts
            if (truth.get(k, 0.0) > 0 > rotated.get(k, 0.0)
                or truth.get(k, 0.0) < 0 < rotated.get(k, 0.0))]


def clip_reading(speed, beats, foreign_grids, window_beats=WINDOW_BEATS, bins=BINS):
    table = aligned_profile(speed, beats, window_beats, bins)
    if table is None:
        return None
    nulls = []
    for grid in foreign_grids:
        scaled = rescale_grid(grid, len(speed))
        if scaled is None:
            continue
        null_table = aligned_profile(speed, scaled, window_beats, bins)
        if null_table is not None:
            nulls.append(shape_of(null_table))
    if len(nulls) < 5:
        return None
    reading = shape_of(table)
    reading["profile"] = table.tolist()
    reading["null_settle"] = float(np.mean([n["settle"] for n in nulls]))
    reading["null_depth"] = float(np.mean([n["depth"] for n in nulls]))
    reading["settle_gain"] = reading["settle"] - reading["null_settle"]
    reading["depth_ratio"] = (reading["depth"] / reading["null_depth"]
                              if reading["null_depth"] > 0 else float("nan"))
    return reading


def half_beat_rotated(speed, index, beats):
    """Ground truth's own trace, moved half a beat -- the defect, synthesised."""
    period = beat_period(beats)
    if not np.isfinite(period) or period < 4:
        return speed
    return np.roll(speed, int(round(period / 2.0)))


def score_arm(clips, motion_dir, audio_dir, grids, window_beats=WINDOW_BEATS,
              bins=BINS, transform=None, seed=0, part=None):
    rng = np.random.default_rng(seed)
    rows = []
    for index, clip in enumerate(clips):
        speed, beats = load_clip(clip, motion_dir, audio_dir, part=part)
        if speed is None or len(beats) < window_beats + 3:
            continue
        if transform is not None:
            speed = transform(speed, index, beats)
        others = [g for name, g in grids.items() if name != clip and len(g) >= 6]
        if len(others) > NULL_GRIDS:
            others = [others[i] for i in rng.choice(len(others), NULL_GRIDS, replace=False)]
        reading = clip_reading(speed, beats, others, window_beats, bins)
        if reading:
            reading["clip"] = clip
            rows.append(reading)
    return rows


def summarise(name, rows):
    if not rows:
        return {"arm": name, "clips": 0}
    n = len(rows)
    wins = sum(1 for r in rows if r["settle_gain"] > 0)
    p = sum(comb(n, k) for k in range(wins, n + 1)) / (2.0 ** n)
    pooled = np.mean(np.asarray([r["profile"] for r in rows]), axis=0)
    return {"arm": name, "clips": n,
            "settle": float(np.median([r["settle"] for r in rows])),
            "null_settle": float(np.median([r["null_settle"] for r in rows])),
            "settle_gain": float(np.median([r["settle_gain"] for r in rows])),
            "wins": wins, "p": float(min(1.0, p)),
            "depth": float(np.median([r["depth"] for r in rows])),
            "null_depth": float(np.median([r["null_depth"] for r in rows])),
            "depth_ratio": float(np.median([r["depth_ratio"] for r in rows])),
            "pooled_profile": pooled.tolist(),
            # PER CLIP, because without it this file can only ask whether an arm
            # beats its OWN null and never whether one arm beats ANOTHER.  Two
            # arms that are each significant against their own nulls say nothing
            # about each other, and on 2026-09-01 that was exactly the question
            # (baseline settle +0.0585 57/93 P=0.019 against learned +0.0631
            # 64/93 P=0.00018 -- unpairable without these rows).
            "per_clip": {r["clip"]: {"settle": float(r["settle"]),
                                     "settle_gain": float(r["settle_gain"]),
                                     "depth": float(r["depth"])}
                         for r in rows}}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", required=True)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--ground-truth-dir", required=True)
    parser.add_argument("--arm", action="append", default=[], metavar="NAME=DIR")
    parser.add_argument("--window-beats", type=int, default=WINDOW_BEATS)
    parser.add_argument("--bins", type=int, default=BINS)
    parser.add_argument("--controls", action="store_true")
    parser.add_argument("--parts", action="store_true",
                        help="also read settle per body part (feet, hips+knees, "
                             "torso, shoulders, elbows, hands), the quantified "
                             "form of the operator's '用不同肢体部位换着卡点'. "
                             "The whole-body column cannot answer it: a body "
                             "whose every part lands together and one that "
                             "hands the accent from foot to hip to hand read "
                             "the same there. POSITIVE CONTROL: every part's "
                             "reading on ground truth must FLIP SIGN when "
                             "ground truth's own trace is rolled half a beat -- "
                             "the same direction-sensitivity check the "
                             "whole-body column uses, and the only one that is "
                             "valid on any corpus. An ordering copied from "
                             "another corpus is NOT a control: the v5 numbers "
                             "(feet +0.0707 > hips +0.0462 > hands +0.0320 > "
                             "torso +0.0223, 93 clips) do not reproduce on the "
                             "20 T-series clips, where torso outranks hands; "
                             "different dancers may hand the accent around "
                             "differently, so that ordering is an observation "
                             "about a corpus, not a property of the instrument. "
                             "SPREAD is the column for '换着部位卡点': sd/mean "
                             "across parts, high when the body passes the "
                             "accent from part to part and low when every part "
                             "stops on the same frame.")
    parser.add_argument("--json")
    arm_sourcing.add_arguments(parser)
    args = parser.parse_args()

    requested = [line.strip() for line in open(args.clips) if line.strip()]
    # See tools/arm_sourcing.py.  A clip that never went through retrieval is
    # the completion on music alone; its beat phase is not this arm's.  The same
    # surviving list feeds ground truth, the controls and the foreign-grid null,
    # because a null built on a different clip set is not this table's null.
    arm_specs = [(spec.partition("=")[0], spec.partition("=")[2]) for spec in args.arm]
    clips, sourcing = arm_sourcing.select_clips(
        requested, arm_specs, threshold=args.sourcing_threshold,
        include_unsourced=args.include_unsourced)
    for line in arm_sourcing.format_header(sourcing):
        print(line)
    grids = {}
    for clip in clips:
        speed, beats = load_clip(clip, args.ground_truth_dir, args.audio_dir)
        if speed is not None and len(beats) >= 6:
            grids[clip] = beats

    jobs = [("ground truth", args.ground_truth_dir, None)]
    if args.controls:
        jobs += [("  control: GT half-beat rotated", args.ground_truth_dir, half_beat_rotated),
                 ("  control: GT time-reversed", args.ground_truth_dir,
                  lambda s, i, b: reversed_speed(s, i)),
                 ("  control: GT 0.5s shuffled", args.ground_truth_dir,
                  lambda s, i, b: shuffled_speed(s, i))]
    jobs += [(name, directory, None) for name, directory in arm_specs]

    out = []
    header = "{:<34}{:>4}{:>10}{:>9}{:>9}{:>8}{:>11}{:>8}{:>8}".format(
        "arm", "n", "settle", "null", "gain", "wins", "P", "depth", "d/null")
    print(header)
    for name, directory, transform in jobs:
        rows = score_arm(clips, directory, args.audio_dir, grids,
                         args.window_beats, args.bins, transform)
        row = summarise(name, rows)
        out.append(row)
        if not row["clips"]:
            print("{:<34}{:>4}".format(name, 0))
            continue
        print("{:<34}{:>4}{:>10.4f}{:>9.4f}{:>9.4f}{:>5}/{:<3}{:>11.3g}{:>8.4f}{:>8.2f}".format(
            name, row["clips"], row["settle"], row["null_settle"], row["settle_gain"],
            row["wins"], row["clips"], row["p"], row["depth"], row["depth_ratio"]))

    print("\npooled beat-aligned profile (sd units, bin 0 = the beat)")
    print("{:<34}{}".format("phase", "".join("{:>8.2f}".format(k / args.bins)
                                             for k in range(args.bins))))
    for row in out:
        if row["clips"]:
            print("{:<34}{}".format(row["arm"], "".join(
                "{:>8.3f}".format(v) for v in row["pooled_profile"])))

    if args.parts:
        print("\npart settle (positive => that part is slowest ON the beat)")
        names = list(PARTS)
        part_jobs = [(n, d, t) for n, d, t in jobs if t is None]
        part_jobs.insert(1, ("  control: GT half-beat rotated",
                             args.ground_truth_dir, half_beat_rotated))
        part_rows, part_per_clip, arm_rho = {}, {}, {}
        for name, directory, transform in part_jobs:
            values, per_clip = {}, {}
            for key in names:
                rows = score_arm(clips, directory, args.audio_dir, grids,
                                 args.window_beats, args.bins, transform,
                                 part=PARTS[key])
                summary = summarise(name, rows)
                values[key] = summary.get("settle") if summary["clips"] else float("nan")
                # KEPT, not discarded.  summarise() has always computed these
                # and this loop used to drop them, so the part columns could
                # only be read pooled -- two arms could not be paired clip by
                # clip, and the operator's own labelled clips could not be used
                # to check the column at all.
                for clip, reading in summary.get("per_clip", {}).items():
                    per_clip.setdefault(clip, {})[key] = reading["settle"]
            part_rows[name] = values
            part_per_clip[name] = per_clip

        gt = part_rows.get("ground truth", {})
        rot = part_rows.get("  control: GT half-beat rotated", {})
        used = validated_parts(gt, rot, names)
        dropped = [k for k in names if k not in used]

        # WHICH CLIPS THIS COLUMN IS ALLOWED TO JUDGE.  The half-beat roll is a
        # synthesised defect with a known answer, so a clip where the rolled
        # ground truth still ranks its parts like the real one is a clip this
        # column cannot read.  Measured 2026-09-12 over the 20 eval clips: the
        # control inverts the ordering on 16 of 19, median rho -0.70 -- and the
        # one clip it CANNOT read is 7618203431723357818:clip000, which is the
        # very clip the operator keeps calling out.  Pooling it in would let the
        # hardest clip vote with noise, so it is named and excluded rather than
        # silently averaged.
        truth_clip_parts = part_per_clip.get("ground truth", {})
        rot_clip_parts = part_per_clip.get("  control: GT half-beat rotated", {})
        control_rho = {clip: part_order_rho(parts, truth_clip_parts.get(clip, {}), used)
                       for clip, parts in rot_clip_parts.items()
                       if clip in truth_clip_parts}
        judgeable = sorted(c for c, v in control_rho.items() if v == v and v < 0.0)
        blind = sorted(c for c, v in control_rho.items() if not (v == v and v < 0.0))

        print("{:<34}{}{:>10}{:>9}{:>9}{:>5}".format(
            "arm", "".join("{:>12}".format(k + ("*" if k in dropped else ""))
                           for k in names), "spread", "order", "med/clip", "n"))
        for name in (n for n, _, _ in part_jobs):
            values = part_rows[name]
            cells = "".join("{:>12.4f}".format(values[k]) if values[k] == values[k]
                            else "{:>12}".format("-") for k in names)
            # spread and order are computed over the VALIDATED parts only.
            finite = [values[k] for k in used if values[k] == values[k]]
            mean = np.mean(finite) if finite else float("nan")
            spread = (np.std(finite) / abs(mean)) if finite and mean else float("nan")
            rho = part_order_rho(values, gt, used)
            per = part_per_clip.get(name, {})
            arm_rho[name] = {c: part_order_rho(per[c], truth_clip_parts.get(c, {}), used)
                             for c in judgeable
                             if c in per and c in truth_clip_parts}
            judged = [part_order_rho(per[c], truth_clip_parts.get(c, {}), used)
                      for c in judgeable if c in per and c in truth_clip_parts]
            judged = [v for v in judged if v == v]
            median_rho = float(np.median(judged)) if judged else float("nan")
            print("{:<34}{}{:>10.3f}{:>9.2f}{:>9.2f}{:>5}".format(
                name, cells, spread, rho, median_rho, len(judged)))

        print("positive control -- every part flips sign under a half-beat roll: "
              "{} ({}/{} parts{})".format(
                  "PASS" if not dropped else "FAIL",
                  len(used), len(names),
                  "" if not dropped else "; * DROPPED from spread and order: "
                  + ", ".join(dropped)))
        # PAIRED AGAINST THE FIRST ARM, which is the only comparison this column
        # can actually make.  The pooled ``order`` medians each part over the
        # clips and THEN ranks, so moving one part's median across a rank
        # boundary flips it without any clip's own ordering changing: measured
        # 2026-09-12, --draft-feet-lead moved pooled order -0.80 -> +0.30 while
        # the paired per-clip difference was -0.081, 6 of 14, sign P = 0.79.
        # Reporting only the pooled number would have called that a win.
        arms_only = [n for n, _, _ in part_jobs
                     if not n.startswith(("ground", "  control"))]
        if len(arms_only) > 1:
            base = arms_only[0]
            print("\npaired against {!r}, per clip, on the clips this column can "
                  "read".format(base))
            print("{:<34}{:>6}{:>10}{:>10}{:>10}{:>9}".format(
                "arm", "n", "mean d", "median d", "wins", "sign P"))
            for name in arms_only[1:]:
                pairs = [arm_rho[name][c] - arm_rho[base][c]
                         for c in arm_rho.get(name, {})
                         if c in arm_rho.get(base, {})
                         and arm_rho[name][c] == arm_rho[name][c]
                         and arm_rho[base][c] == arm_rho[base][c]]
                if not pairs:
                    continue
                values = np.asarray(pairs)
                moved = values[values != 0]
                wins = int((moved > 0).sum())
                total = len(moved)
                p = (sum(comb(total, k) for k in range(wins, total + 1))
                     / 2.0 ** total) if total else 1.0
                print("{:<34}{:>6}{:>10.3f}{:>10.3f}{:>6}/{:<3}{:>9.3f}".format(
                    name, len(values), float(values.mean()),
                    float(np.median(values)), wins, total, min(1.0, p)))
            print("(clips where the two arms rank the parts identically are "
                  "excluded from the sign test and shown in n)")

        print("order    = Spearman rho of this arm's POOLED part ranking against "
              "ground truth's, over the validated parts (1.00 = the dancer's order)")
        print("med/clip = the same rho taken PER CLIP and then medianed, over the "
              "clips this column can read -- the pairable one")
        if blind:
            print("EXCLUDED from med/clip ({} clip{}, the half-beat control fails "
                  "there so the column is blind): {}".format(
                      len(blind), "" if len(blind) == 1 else "s",
                      ", ".join(c.split(":")[-2][-12:] + ":" + c.split(":")[-1]
                                for c in blind)))

        # Per clip, so arms can be paired and so the operator's own labelled
        # clips can be used as the check on this column.
        truth_per_clip = part_per_clip.get("ground truth", {})
        for row in out:
            name = row["arm"]
            if name not in part_per_clip:
                continue
            row["parts"] = part_rows[name]
            row["parts_used"] = used
            row["parts_dropped"] = dropped
            row["part_per_clip"] = part_per_clip[name]
            row["part_order_rho"] = part_order_rho(part_rows[name], gt, used)
            row["judgeable_clips"] = judgeable
            row["blind_clips"] = blind
            row["control_rho_per_clip"] = control_rho
            row["part_order_rho_per_clip"] = {
                clip: part_order_rho(parts, truth_per_clip.get(clip, {}), used)
                for clip, parts in part_per_clip[name].items()
                if clip in truth_per_clip}

    if args.json:
        # Dict, not the bare list this used to write, so the excluded clips
        # travel with the readings.  Old readers take payload["arms"].
        pathlib.Path(args.json).write_text(
            json.dumps({"sourcing": sourcing, "arms": out}, indent=2))


if __name__ == "__main__":
    main()
