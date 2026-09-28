#!/usr/bin/env python3
"""Within-clip beat articulation: hit depth in m/s, hit density per second, and
the alignment gain that survives the bar-grid splice.

WHY THIS EXISTS.  ``tools/score_arm_table.py`` gates beat hitting on ``e_corr``,
a CROSS-CLIP Pearson correlation between an arm's per-clip mean energy and the
ground truth's.  The operator's complaint is not cross-clip: within a clip, at
every strong beat, the ground truth accelerates and then stops, and the
generation does not ("gt 每次音乐歌词的强节拍顿挫,相应的动作都能在对应位置有
加速停顿 ... 卡点的击中密度比 gt 差远了").  A correlation over 90 clips can
neither pass nor fail for that reason.  This file is the within-clip
instrument, and everything below is what had to be measured before it was
allowed to judge anything (CLAUDE.md §2.1).

THE DEFINITION.  Body speed is the mean over the 24 SMPL joints of the
root-relative per-frame displacement magnitude x 30 fps, in m/s -- the
per-frame form of ``score_arm_table.energy``, so the two live in the same
units and the same space.  It is smoothed with a 3-tap Hann kernel (~10 Hz)
before extrema are read, because D7's 5-15 Hz residual floor and Wave 5's
jitter of up to 0.765 would otherwise be read as staccato.  At each music beat
frame b (channel 34, the beat one-hot):

    trough = argmin speed over [b-3, b+4]           the "stop"
    peak   = max    speed over [b-6, trough]        the acceleration before it
    depth  = speed[peak] - speed[trough]            METRES PER SECOND, absolute

The peak is forced to precede the trough, so the statistic reads
accelerate-then-stop, not the reverse.  The +-4 frame tolerance is what makes
it phase-forgiving: the median beat period here is 15 frames (p5 12, p95 19),
so the 10-frame search span stays inside one beat.

NO WITHIN-CLIP AMPLITUDE NORMALIZATION.  ``docs/DANCE_QUALITY_DEFECTS.md``
§12.1: four earlier beat instruments all read "generated >= GT" and all four
divided by the clip's own median speed, which divides out exactly the energy
collapse they had to detect.  Depth is in m/s, the hit threshold is in m/s.

THE THREE READOUTS, and what each is allowed to say.

``depth``   per-clip mean beat-hit depth, m/s.
``hit/s``   beats per second of clip whose depth clears an ABSOLUTE threshold
            (default 0.5 m/s = the ground truth's own mean depth, fixed before
            any arm was scored -- a calibration in the sense ``skate``'s 0.394
            is, not a level tuned until a favoured arm passed).
            **NEITHER OF THESE TWO MAY GATE ANYTHING.**  Measured on 7
            generated arms plus ground truth, 90 val clips: across arms
            ``depth`` correlates with ``energy`` at r=0.990 and ``hit/s`` at
            r=0.984, and regressing ``depth`` on ``energy`` over the generated
            arms puts the ground truth 7.66 residual sd BELOW its own line --
            i.e. per unit of energy the ground truth has LESS beat-hit depth
            than every generated arm.  These columns are the ``energy`` column
            restated in the operator's vocabulary.  They are printed because
            "hit density 0.51/s against 0.67/s" is the sentence the operator
            asked for, not because they carry new information.

``gain``    depth minus the same clip's ROTATED-BEAT null, m/s, plus the paired
            per-clip win rate.  The null circularly shifts the beat grid by a
            uniform random offset mod the clip length, 40 draws averaged: it
            preserves the beat count and every inter-beat gap exactly and
            destroys only the phase, so the difference is the ALIGNMENT part of
            the depth and nothing else.  This is the column that is new.

THE CONFOUND THAT MUST BE MASKED, or the column reads backwards.  With every
beat included, the no-aux baseline reads gain +0.0368 (72/90, P=8.1e-09)
against the ground truth's +0.0244 (62/90, P=4.4e-04) -- "the generation hits
the beat better than the ground truth", the §12.1 failure repeating.  It is
D4: ``plan_bar_grid`` snaps every plan segment boundary onto the beat grid, and
the prototype-to-prototype splice brakes to 0.15x for exactly one frame there.
Split the baseline's beats by distance to its own plan boundaries:

    beats within 3 frames of a plan boundary   gain +0.1442   84/90  P=1.1e-18
    beats farther than 3 frames                gain -0.0360   28/90  P=4.4e-04
    a RANDOM subset of the same size           gain +0.0439   69/90  P=3.9e-07

The random-subset row is the control on the mask itself: dropping 41% of the
beats at random does not move the reading, so the collapse belongs to the
boundary beats and not to the smaller sample.  ``--mask-arm`` therefore takes
ONE arm's plan boundaries and applies the SAME beat subset to every row,
ground truth included, so the rows are comparable frame for frame.

THE VALIDATED READING (90 val clips, mask arm = the no-aux baseline, 1766 of
3100 beats kept, ``runs/beat_articulation_masked.json``):

    arm                energy  depth  hit/s  hits/GT     gain   wins       P
    ground truth        0.797  0.455  0.384    1.000  +0.0265  61/90  9.7e-04
    no-aux baseline     0.624  0.338  0.234    0.519  -0.0344  32/90  8.0e-03
    phase s01           0.683  0.402  0.313    0.750  -0.0168  34/90  2.6e-02
    phase s02           0.582  0.321  0.235    0.500  -0.0315  34/90  2.6e-02
    tb s02              0.665  0.412  0.321    0.800  -0.0052  45/90  1.0
    ct tb s01           0.489  0.273  0.171    0.408  -0.0093  37/90  1.1e-01
    ct+fk.25 s01        0.433  0.238  0.125    0.286  -0.0148  37/90  1.1e-01
    em10 s01            0.225  0.125  0.002    0.000  -0.0013  42/90  6.0e-01
    em30 s01            0.102  0.048  0.000    0.000  -0.0019  28/90  4.4e-04
    dyn  (REJECTED)     0.723  0.389  0.304    0.700  -0.0372  31/90  4.2e-03
    snap (REJECTED)     0.615  0.358  0.263    0.640  -0.0096  42/90  6.0e-01

**The ground truth is the only row with a positive gain.**  The operator's
sentence, in absolute units: the shipped arm lands 0.234 hits per second of
0.5 m/s depth against the ground truth's 0.384, i.e. 52% of the hit density,
and the hits it does land sit where the beats are not.

THE FOUR CONTROLS, all run before this file printed a verdict.

  (i) POSITIVE.  Ground truth reads +0.0265, 61/90, P=9.7e-04, and the mask
      does not create it: unmasked it reads +0.0244, 62/90, P=4.4e-04.  Sign
      and significance survive the whole instrument sweep -- peak reach in
      {4,6,8} x trough reach in {3,4,5} x smoothing in {1,3,5}, 27 settings,
      ground truth positive in all 27 (+0.006 to +0.028) and the baseline
      negative in all 27 (-0.005 to -0.179).  This is the power at n=90 that
      ``beat_R`` does not have: its own positive control fails at this sample
      size (§11.3).
 (ii) NEGATIVE, alignment.  The rotated-beat null is the null itself.
(iii) NEGATIVE, known-bad.  ``em30 s01`` (Wave 5, energy 0.133 of GT) reads
      depth 0.048 m/s, hit rate 0.000/s, and the worst win rate in the table,
      28/90.
 (iv) AMPLITUDE-MATCHED.  Ground-truth motion progressively low-passed in time
      -- the F2 mean-regression failure mode, applied to real dance:

          Hann taps      1      3      5      9     15     25
          energy     0.797  0.752  0.715  0.635  0.519  0.375
          depth      0.455  0.392  0.353  0.283  0.202  0.118
          gain      +.0265 +.0191 +.0187 +.0151 +.0124 +.0077
          wins        61/90  60/90  61/90  56/90  58/90  55/90

      Damping costs gain but never flips its sign: at 25 taps the ground truth
      is at energy 0.375 -- BELOW ``ct tb s01``'s 0.489 and ``ct+fk.25``'s
      0.433 -- and still reads +0.0077, 55/90, while both of those arms read
      negative.  So the generated arms' negative gain is not "they are
      smoother than the ground truth".  They are actively flatter AT the beats
      than at other phases of the same music.

NULL FAIRNESS -- the 2026-08-31 null was biased against every generated arm,
and this is the correction (CLAUDE.md 2.1 gate 2, 4.5: both versions stated).

WHAT WAS WRONG.  ``--mask-arm`` keeps only beats farther than 3 frames from a
plan boundary, but the ROTATED beats were not filtered the same way, so the
null could land on the very splice transients the signal was masked away from.
That is not a small asymmetry here, because ``plan_bar_grid`` snaps boundaries
onto every second beat: measured on ``probe_final_s20260902``, 1,580 masked
beats sit at a mean segment phase of 0.501 with 60.3% of them inside the single
decile 0.5-0.6, while a rotated beat lands uniformly.  And the generated
motion's speed is organised by that grid rather than by the music -- mean body
speed by segment phase reads 0.725 / 0.692 / 0.613 / 0.623 / 0.632 / 0.641 /
0.629 / 0.632 / 0.710 / 0.542 (max/min 1.34, structure at the edges only)
against the ground truth's 0.825 ... 0.832 on the same frames (max/min 1.04,
i.e. flat -- the ground truth is not organised by the generated plan at all).
So the signal sampled the flat plateau and the null sampled the transients.

WHAT IT CHANGED.  Rotated beats are now filtered through the same boundary
guard.  Ground truth, which has no plan boundaries of its own, gets slightly
STRONGER; every generated arm moves from "negative" to "chance":

    arm                    gain (biased null)      gain (guarded null)
    ground truth           +0.0265  61/90          +0.0323  61/90  P=9.7e-4
    no-aux tb s01          -0.0034  38/90          +0.0084  45/90  P=1.0
    no-aux tb s02          -0.0052  45/90          +0.0082  51/90  P=0.25
    shipped final s02      -0.0344  32/90          -0.0075  43/90  P=0.75
    phct  (mn+phase) s01   -0.0086  40/90          -0.0018  45/90  P=1.0
    phct  (mn+phase) s02   -0.0203  35/90          -0.0065  41/90  P=0.46
    ctfk25 (music-deaf)s01 -0.0148  37/90          -0.0073  41/90  P=0.46
    ctfk25 (music-deaf)s02 -0.0186  32/90          -0.0116  34/90  P=0.026
    em30 s01  (known-bad)  -0.0019  28/90          -0.0008  35/90  P=0.045
    dyn (REJECTED postproc)-0.0372  31/90          -0.0086  42/90  P=0.6

CORRECTED READING.  The conclusion that survives is "the ground truth is the
only phase-locked row"; the conclusion that does NOT survive is "the generated
arms are actively flatter AT the beats".  They are at CHANCE (34-51 of 90).
The arms are beat-AGNOSTIC, not beat-avoiding, and roughly a third of the
apparent anti-alignment was this null.

A CONTROLLED PAIR THAT THE CORRECTED COLUMN SETTLES.  ``phct`` (music
normalization + phase features) and ``ctfk25`` (music-deaf) differ ONLY in
whether the model can numerically hear the beat -- same corpus
(release_v3_timebase), same contact 10 + fk 0.25, two seeds each.  Guarded
gain -0.0018 / -0.0065 against -0.0073 / -0.0116, win rates 45,41 against
41,34: no separation.  Making the beat channel legible (0.41% -> 15.1% of the
first layer's output variance) does not by itself buy beat articulation.

CONTROLS RE-RUN ON THE CORRECTED FORM, all four passing:
    GT + a hit imposed on every beat        gain +0.5332  90/90  P=1.6e-27
    the SAME hit shifted +7 frames          gain -0.5814   3/90  P=2.0e-22
    GT unmodified                           gain +0.0323  61/90  P=9.7e-04
    GT low-passed to energy 0.388           gain +0.0054  52/90  P=0.17
The shifted-hit row is the strongest of these and is new: the column separates
"hits, on the beat" from "identical hits, half a beat off", so it is reading
phase and not merely the presence of dynamics.  The low-pass row is the
amplitude-matched control -- at energy 0.388, below every generated arm except
the two Wave-5 ones, the sign still does not flip.

``--unfair-null`` reproduces the old column for anyone re-reading the old table.

THE CEILING.  A synthetic body that stops on every single beat reads
gain/depth = 0.31 (``tests/test_score_beat_articulation.py``); the +-4 frame
tolerance that makes the column phase-forgiving also lets a rotated beat fall
inside a dip.  Ground truth reads 0.027/0.455 = 0.06, about a fifth of what is
reachable.  A future arm reading much above 0.3 of its own depth should be
disbelieved before it is celebrated.

IS IT NEW, OR IS IT ``energy`` AGAIN?  Measured across the 8 generated arms:
``depth`` correlates with ``energy`` at r=+0.993 and ``hit/s`` at r=+0.974, and
the ground truth sits within 1 residual sd of the generated arms' own
energy-to-depth line -- those two columns ARE the energy column.  ``gain``
correlates with ``energy`` at r=-0.60 (P=0.11, not significant) and puts the
ground truth +4.8 residual sd off that line; against the scorecard's ``e_corr``
it reads r=-0.15 (P=0.73).  ``gain`` is the only part of this file that the
existing table does not already contain.

WHAT ``gain`` STILL CANNOT DO ALONE.  It is an absolute m/s difference, so an
arm with no motion has no gain to lose: ``em30`` reads -0.0019, arithmetically
closer to the ground truth than the baseline's -0.0344, while being the deadest
arm in the table.  Read ``gain`` beside ``depth``/``energy`` or not at all --
the same rule ``jitter`` carries.  The scale-free companion is the WIN RATE,
which does not collapse for a dead arm: ground truth 61/90, baseline 32/90,
em30 28/90.

PROVENANCE.  Written 2026-08-31 for this repository's own complaint; the beat
channel and the FPS come from the corpus contract, the joint set and the
root-relative convention from ``score_arm_table``, the rotated-beat null is
this file's own (the repo's ``shuffled_events`` shuffles motion events, not
beats), and the boundary confound is D4 restated.  Nothing here is inherited
from the paper; the accelerate-then-stop shape is the operator's sentence.
"""
import argparse
import json
import pathlib
import pickle

import numpy as np
from scipy.stats import binomtest, wilcoxon

FPS = 30.0
BEAT_CHANNEL = 34
PRE = 6           # peak search reaches this many frames before the beat
POST = 4          # trough search reaches this many frames after the beat
TROUGH_PRE = 3    # the trough may sit slightly before the beat (dancers hit early)
SMOOTH = 3        # Hann taps on the speed envelope
THRESHOLD = 0.5   # m/s, absolute; the ground truth's own mean depth
NULL_DRAWS = 40
BOUNDARY_GUARD = 3


def body_speed(joints, smooth=SMOOTH):
    """Root-relative mean joint speed per frame, m/s -- energy's per-frame form."""
    relative = joints - joints[:, :1, :]
    speed = np.linalg.norm(np.diff(relative, axis=0), axis=2).mean(1) * FPS
    if smooth > 1:
        kernel = np.hanning(smooth + 2)[1:-1]
        speed = np.convolve(speed, kernel / kernel.sum(), mode="same")
    return speed


def beat_depths(speed, beats, pre=PRE, post=POST, trough_pre=TROUGH_PRE):
    """Per-beat accelerate-then-stop depth, m/s.  The peak is forced before the trough."""
    out = []
    frames = len(speed)
    for beat in beats:
        low, high = beat - pre, beat + post
        if low < 0 or high >= frames:
            continue
        window = speed[low:high + 1]
        offset = pre - trough_pre
        trough = int(np.argmin(window[offset:])) + offset
        out.append(float(window[:trough + 1].max() - window[trough]))
    return np.asarray(out, float)


def rotated_beats(beats, frames, rng):
    """Circularly shift the beat grid: same count, same gaps, phase destroyed."""
    return np.sort((np.asarray(beats) + int(rng.integers(0, frames))) % frames)


def plan_boundaries(path):
    payload = pickle.load(open(path, "rb"))
    labels = payload.get("atomic_labels")
    if labels is None:
        return None
    return np.flatnonzero(np.diff(np.asarray(labels))) + 1


def clip_reading(joints, music, rng, keep_beats=None, threshold=THRESHOLD,
                 draws=NULL_DRAWS, smooth=SMOOTH, boundaries=None, guard=BOUNDARY_GUARD):
    """One clip's depth, hit rate and rotated-beat null.

    ``boundaries`` is the mask arm's plan-segment boundary array.  When it is
    given, the ROTATED beats are filtered through the same boundary guard as
    the real ones -- see NULL FAIRNESS in the module docstring.  Passing None
    reproduces the biased 2026-08-31 null.
    """
    frames = len(joints)
    beats = np.flatnonzero(np.asarray(music)[:frames, BEAT_CHANNEL] > 0.5)
    total_beats = len(beats)
    if keep_beats is not None:
        beats = np.asarray([b for b in beats if b in keep_beats], int)
    if len(beats) < 6:
        return None
    speed = body_speed(joints, smooth)
    seconds = len(speed) / FPS
    depth = beat_depths(speed, beats)
    if len(depth) < 4:
        return None
    nulls, null_rates = [], []
    for _ in range(draws):
        drawn = rotated_beats(beats, len(speed), rng)
        if boundaries is not None and len(boundaries):
            drawn = np.asarray(
                [b for b in drawn if np.min(np.abs(b - boundaries)) > guard], int)
            if len(drawn) < 4:
                continue
        shifted = beat_depths(speed, drawn)
        if len(shifted) >= 4:
            nulls.append(shifted.mean())
            null_rates.append((shifted >= threshold).sum() / seconds)
    if not nulls:
        return None
    return {
        "depth": float(depth.mean()),
        "depth_null": float(np.mean(nulls)),
        "hits_per_s": float((depth >= threshold).sum() / seconds),
        "hits_per_s_null": float(np.mean(null_rates)),
        "beats_used": int(len(depth)),
        "beats_total": int(total_beats),
        "energy": float(speed.mean()),
    }


def beat_mask(mask_dir, clips, audio_dir, guard):
    """Beats farther than ``guard`` frames from one arm's plan boundaries.

    ONE arm supplies the mask for every row, ground truth included, so all rows
    read the same beats.  The mask's own control is in the module docstring: a
    random subset of the same size leaves the reading where it was.
    """
    keep, bounds = {}, {}
    for clip in clips:
        path = mask_dir / (clip + ".pkl")
        audio = audio_dir / (clip + ".npy")
        if not path.is_file() or not audio.is_file():
            continue
        boundaries = plan_boundaries(path)
        beats = np.flatnonzero(np.load(audio)[:, BEAT_CHANNEL] > 0.5)
        if boundaries is None or len(boundaries) == 0:
            keep[clip] = set(int(b) for b in beats)
            continue
        keep[clip] = {int(b) for b in beats
                      if np.min(np.abs(b - boundaries)) > guard}
        bounds[clip] = boundaries
    return keep, bounds


def score_arm(arm_dir, clips, audio_dir, ground_truth, seed, threshold, smooth, keep,
              bounds=None, guard=BOUNDARY_GUARD):
    rng = np.random.default_rng(seed)
    rows = []
    for clip in clips:
        path = arm_dir / (clip + ".pkl")
        audio = audio_dir / (clip + ".npy")
        if not path.is_file() or not audio.is_file():
            continue
        joints = np.asarray(pickle.load(open(path, "rb"))["full_pose"], float)
        got = clip_reading(joints, np.load(audio), rng,
                           keep.get(clip) if keep else None, threshold, smooth=smooth,
                           boundaries=(bounds or {}).get(clip), guard=guard)
        if got is None:
            continue
        got["clip"] = clip
        reference = ground_truth.get(clip)
        if reference is not None:
            got["gt_depth"] = reference["depth"]
            got["gt_hits_per_s"] = reference["hits_per_s"]
        rows.append(got)
    return rows


def summarise(name, rows):
    depth = np.array([r["depth"] for r in rows])
    null = np.array([r["depth_null"] for r in rows])
    hits = np.array([r["hits_per_s"] for r in rows])
    wins = int((depth > null).sum())
    paired = [r["depth"] / max(r["gt_depth"], 1e-9) for r in rows if "gt_depth" in r]
    paired_hits = [r["hits_per_s"] / max(r["gt_hits_per_s"], 1e-9)
                   for r in rows if r.get("gt_hits_per_s", 0) > 0]
    return {
        "arm": name,
        "clips": len(rows),
        "beats_used": int(sum(r["beats_used"] for r in rows)),
        "beats_total": int(sum(r["beats_total"] for r in rows)),
        "energy": float(np.mean([r["energy"] for r in rows])),
        "depth": float(depth.mean()),
        "depth_vs_gt": float(np.median(paired)) if paired else float("nan"),
        "hits_per_s": float(hits.mean()),
        "hits_vs_gt": float(np.median(paired_hits)) if paired_hits else float("nan"),
        "depth_null": float(null.mean()),
        "gain": float((depth - null).mean()),
        "gain_wins": "{}/{}".format(wins, len(rows)),
        "gain_p": float(binomtest(wins, len(rows), 0.5).pvalue) if len(rows) else float("nan"),
        "gain_p_wilcoxon": float(wilcoxon(depth, null).pvalue) if len(rows) > 5 else float("nan"),
    }


HEADER = ("{:<28} {:>4} {:>7} {:>7} {:>7} {:>7} {:>7} {:>9} {:>8} {:>10}".format(
    "arm", "n", "energy", "depth", "vs GT", "hit/s", "hits/GT", "gain", "wins", "P"))
ROW = ("{arm:<28} {clips:>4} {energy:>7.3f} {depth:>7.3f} {depth_vs_gt:>7.3f} "
       "{hits_per_s:>7.3f} {hits_vs_gt:>7.3f} {gain:>+9.4f} {gain_wins:>8} {gain_p:>10.2g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", required=True, help="file of clip ids, or a manifest.json")
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--ground-truth-dir", required=True)
    parser.add_argument("--arm", action="append", default=[], metavar="NAME=DIR")
    parser.add_argument("--mask-arm", default=None, metavar="DIR",
                        help="one arm whose plan-segment boundaries define the beat "
                             "subset used for EVERY row.  Without it the bar-grid "
                             "splice dominates and the column reads backwards; see "
                             "the module docstring.")
    parser.add_argument("--boundary-guard", type=int, default=BOUNDARY_GUARD)
    parser.add_argument("--unfair-null", action="store_true",
                        help="reproduce the biased 2026-08-31 null: rotated beats are "
                             "NOT boundary-guarded, so they sample the segment-edge "
                             "splice transients the real beats are masked away from.")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--smooth", type=int, default=SMOOTH)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    if args.clips.endswith(".json"):
        clips = json.load(open(args.clips))["names"]
    else:
        clips = [line.strip() for line in open(args.clips) if line.strip()]
    audio_dir = pathlib.Path(args.audio_dir)
    gt_dir = pathlib.Path(args.ground_truth_dir)

    keep, bounds = None, None
    if args.mask_arm:
        keep, bounds = beat_mask(pathlib.Path(args.mask_arm), clips, audio_dir,
                                 args.boundary_guard)
    if args.unfair_null:
        bounds = None

    gt_rows = score_arm(gt_dir, clips, audio_dir, {}, args.seed,
                        args.threshold, args.smooth, keep, bounds, args.boundary_guard)
    ground_truth = {r["clip"]: r for r in gt_rows}
    for row in gt_rows:
        row["gt_depth"] = row["depth"]
        row["gt_hits_per_s"] = row["hits_per_s"]

    reports = [summarise("ground truth", gt_rows)]
    print(HEADER)
    print(ROW.format(**reports[0]))
    for spec in args.arm:
        name, _, directory = spec.partition("=")
        rows = score_arm(pathlib.Path(directory), clips, audio_dir, ground_truth,
                         args.seed, args.threshold, args.smooth, keep, bounds,
                         args.boundary_guard)
        if not rows:
            print("{:<28} no clips".format(name))
            continue
        reports.append(summarise(name, rows))
        print(ROW.format(**reports[-1]))
    used, total = reports[0]["beats_used"], reports[0]["beats_total"]
    print("\nbeats used {}/{} ({:.0%}); mask arm {}; threshold {:.2f} m/s; smooth {}".format(
        used, total, used / max(total, 1), args.mask_arm or "NONE (splice confound present)",
        args.threshold, args.smooth))
    print("null: rotated beats {} boundary-guarded".format(
        "are NOT" if (args.unfair_null or not args.mask_arm) else "ARE"))
    if not args.mask_arm:
        print("WARNING no --mask-arm: the D4 bar-grid splice inflates every generated "
              "row's gain; see the module docstring.")
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(reports, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
