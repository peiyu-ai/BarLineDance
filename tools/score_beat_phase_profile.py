#!/usr/bin/env python3
"""Is the motion's energy distributed over the beat cycle, LOCALLY, or spread flat?

THE TARGET, in the operator's words (2026-09-01):

    "我们关心的不是能量的大小,而是能量在时间上的分布,相位和峰谷值能不能和节奏同步"
    -- not how much energy, but its distribution in time: do its phase and its
       peaks and troughs synchronise with the rhythm.

and, one message later, the correction that fixes this file's first two drafts:

    "跨clip找mean是个回退操作,每个舞者压的正反拍和节奏密度都不一致,
     同舞者每一段都可能有区别"
    -- a cross-clip mean is a regression; every dancer's on/off-beat emphasis
       and rhythmic density differ, and even one dancer differs segment to segment.

Those two sentences separate three axes that every earlier instrument here
confounded, and fix the unit of measurement:

    AMPLITUDE   how fast the body moves          -- tools/score_arm_table.energy
    PHASE       when within the beat it moves    -- this file
    LOCALITY    over what span that is stable    -- a few beats, NOT a whole clip

------------------------------------------------------------------ THE MEASURE

``s(t)`` is the root-relative mean joint speed per frame in m/s, 3-tap Hann
smoothed -- the same function as ``score_beat_articulation.body_speed``, which is
``score_arm_table.energy`` before its time average.  One trace, so the amplitude
column and this file cannot drift apart.

Inside a window of ``--window-beats`` consecutive beats (default 4, one bar of
4/4):

1. z-score ``s`` over the window.  This is "size does not matter" made exact:
   every reading is invariant to scaling the whole dance's speed, asserted to
   1e-9 by test.  Amplitude is not thereby hidden -- it stays a separate,
   separately reported column, which is what makes removing it a decision here
   rather than the accident docs §12.1 records (four earlier beat instruments
   divided by the clip's own median speed and so divided out the very energy
   collapse they existed to detect).
2. bin the frames by BEAT PHASE, ``phi = (t - b_i) / (b_{i+1} - b_i)``, phase
   rather than frame offset so a tempo change does not smear the profile.
3. ``modulation`` = twice the magnitude of the profile's first Fourier
   coefficient: the depth of the once-per-beat structure, in units of that
   window's own speed standard deviation.  MAGNITUDE, so it does not care
   whether this dancer sits on the beat or behind it.

The clip's reading is the MEDIAN over its windows, and the arm's reading is the
median over clips.  Medians of magnitudes, never means of phases.

WHY LOCAL, MEASURED.  A synthetic dancer who locks perfectly to the beat for the
first half of a clip and perfectly to the OFF-beat for the second half reads, on
a whole-clip profile, ``0.068`` -- against pure noise at ``0.027`` and a steady
locked control at ``1.206``.  Whole-clip averaging turns a perfect dancer into
noise.  The same trace, windowed at 4 beats, reads ``1.220`` against the steady
control's ``1.225``.  That is the operator's point, and it is a unit test.

------------------------------------------------------------------- THE NULL

The null is OTHER CLIPS' BEAT GRIDS, time-scaled onto this clip.

Two nulls were built and discarded first; both failures are recorded because
neither is obvious.

  * Rotating the BEATS is what ``score_beat_articulation`` does.  Refuting
    review on 2026-08-31 found that with a plan-boundary mask the real beats
    concentrate at one segment phase (55.5% inside a single decile) while
    rotated beats land uniformly, so signal and null sample different frames.
  * Rotating the MOTION was this file's first attempt, and it is wrong in a way
    that is invisible until tested: ``modulation`` is the MAGNITUDE of a Fourier
    coefficient, and shifting a signal changes that coefficient's phase but not
    its magnitude.  For motion that oscillates at the beat period the null
    therefore equals the signal exactly.  Measured before replacement: three
    known-locked synthetics read gain <= 0, and 30 real ground-truth clips read
    18/30 wins at P=0.18.  A validated positive control failing disqualifies the
    instrument, not the control (§2.1).

The deeper reason both fail: "invariant to WHERE in the beat the trough sits"
and "tests whether the trough is locked" are in tension.  If the offset does not
matter then any motion oscillating at the beat period is locked by definition,
and the only remaining question is whether its PERIOD and GRID are the music's.
A foreign grid carries a real song's tempo and regularity and breaks only the
correspondence with THIS music, so motion locked to its own song beats it while
motion that merely oscillates does not.

``period_win`` is the second, phase-independent reading: the motion's own
autocorrelation at the music's beat period against its autocorrelation at
adjacent non-harmonic lags.  Autocorrelation is shift-invariant, so no rotation
can defeat it, and CLAUDE.md §2.1 records this exact form succeeding (511/800
clips, binomial p < 1e-14) on a day when the phase-based test read null.  It
assumes stationarity over the clip, which the windowed statistic above
deliberately does not -- they are reported together for that reason.

The headline is a WIN RATE against binomial(n, 0.5).  A rate, because per-clip
and per-segment phase offsets differ and pooling them is exactly the regression
the operator named.

--------------------------------------------------------------- WHAT IT CANNOT SAY

It reads one beat cycle.  A dance that hits every other beat, or lands on the
bar, carries structure at a different period; ``--harmonic`` reads the 2nd and
4th components for that, but the headline stays the fundamental.  It says
nothing about WHICH movement happens: a clip can be perfectly locked and still
be one pose repeated.  And it says nothing about amplitude -- by construction,
on purpose, and that column must be read beside it.
"""
import argparse
import json
import pathlib
import pickle
import sys
from math import comb

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

FPS = 30.0
BEAT_CHANNEL = 34
SMOOTH = 3
BINS = 12
WINDOW_BEATS = 4
NULL_GRIDS = 24


def body_speed(joints, smooth=SMOOTH, part=None):
    """Root-relative mean joint speed per frame, m/s -- energy's per-frame form.

    ``part`` restricts the mean to a joint subset (see ``PARTS``).  It is still
    ROOT-RELATIVE for every part, so a part's reading is what that part does
    relative to the body, not how far the body travelled -- otherwise every part
    would inherit the root's trace and they would all read the same.
    """
    relative = joints - joints[:, :1, :]
    if part is not None:
        relative = relative[:, list(part), :]
    speed = np.linalg.norm(np.diff(relative, axis=0), axis=2).mean(1) * FPS
    if smooth > 1:
        kernel = np.hanning(smooth + 2)[1:-1]
        speed = np.convolve(speed, kernel / kernel.sum(), mode="same")
    return speed


def joints_of(path):
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    pose = np.asarray(payload["full_pose"], np.float64)
    return pose.reshape(len(pose), -1, 3)


def beat_phase(beats, frames):
    """Phase in [0, 1) for every frame lying inside a beat interval; NaN outside."""
    phase = np.full(frames, np.nan)
    for start, end in zip(beats[:-1], beats[1:]):
        if end <= start or end > frames:
            continue
        phase[start:end] = np.arange(end - start) / float(end - start)
    return phase


def profile(speed, phase, bins=BINS):
    """Mean z-scored speed per beat-phase bin, and the once-per-beat depth."""
    valid = ~np.isnan(phase)
    if valid.sum() < bins * 3:
        return None, np.nan, np.nan
    values = speed[valid]
    spread = values.std()
    if spread <= 0:
        return None, np.nan, np.nan
    values = (values - values.mean()) / spread
    index = np.minimum((phase[valid] * bins).astype(int), bins - 1)
    table = np.array([values[index == k].mean() if np.any(index == k) else np.nan
                      for k in range(bins)])
    if np.isnan(table).any():
        return None, np.nan, np.nan
    coefficient = (table * np.exp(-2j * np.pi * np.arange(bins) / bins)).mean()
    return table, 2.0 * abs(coefficient), float(np.angle(coefficient) % (2 * np.pi))


def local_modulations(speed, beats, window_beats=WINDOW_BEATS, bins=BINS):
    """One modulation per sliding window of ``window_beats`` beats.

    The unit of measurement, and the whole reason this function exists rather
    than a whole-clip profile: a dancer who emphasises the downbeat in one
    section and the backbeat in the next is TWO locked segments, and averaging
    their profiles cancels them into a flat line.  Measured on a synthetic that
    switches phase halfway: whole-clip 0.068 (noise reads 0.027), windowed 1.220
    (a steady locked control reads 1.225).
    """
    beats = np.asarray(beats)
    out, phases = [], []
    for start in range(0, len(beats) - window_beats):
        first, last = int(beats[start]), int(beats[start + window_beats])
        if last > len(speed) or last - first < bins * 3:
            continue
        window = speed[first:last]
        local_beats = beats[start:start + window_beats + 1] - first
        table, modulation, preferred = profile(
            window, beat_phase(local_beats, len(window)), bins)
        if table is not None and np.isfinite(modulation):
            out.append(modulation)
            phases.append(preferred)
    return np.asarray(out), np.asarray(phases)


def beat_period(beats):
    if len(beats) < 3:
        return float("nan")
    return float(np.median(np.diff(beats)))


def period_score(speed, period, span=4, harmonics=(0.5, 1.0, 2.0)):
    """Autocorrelation at the beat period against adjacent non-harmonic lags.

    Phase-independent, so no shift can defeat it.  Adjacent lags rather than a
    global baseline because a motion trace has a rising low-frequency
    autocorrelation and lag 15 against lag 60 would measure that slope.
    """
    if not np.isfinite(period) or period < 4:
        return np.nan, np.nan
    values = speed - speed.mean()
    denominator = float((values * values).sum())
    if denominator <= 0:
        return np.nan, np.nan

    def ac(lag):
        lag = int(round(lag))
        if lag < 1 or lag >= len(values):
            return np.nan
        return float((values[:-lag] * values[lag:]).sum() / denominator)

    target = ac(period)
    excluded = {int(round(period * h)) for h in harmonics}
    neighbours = [ac(period + d) for d in range(-span, span + 1)
                  if d != 0 and int(round(period + d)) not in excluded]
    neighbours = [v for v in neighbours if np.isfinite(v)]
    if not np.isfinite(target) or not neighbours:
        return np.nan, np.nan
    return target, float(np.median(neighbours))


def rescale_grid(grid, frames):
    """A foreign clip's beat grid, stretched onto this clip's length."""
    grid = np.asarray(grid, float)
    if len(grid) < 3 or grid[-1] <= 0:
        return None
    scaled = np.unique(np.round(grid / grid[-1] * (frames - 1)).astype(int))
    return scaled if len(scaled) >= 4 else None


def clip_reading(speed, beats, foreign_grids=(), window_beats=WINDOW_BEATS,
                 bins=BINS, harmonic=False):
    values, phases = local_modulations(speed, beats, window_beats, bins)
    if len(values) < 2:
        return None
    nulls = []
    for grid in foreign_grids:
        scaled = rescale_grid(grid, len(speed))
        if scaled is None:
            continue
        null_values, _ = local_modulations(speed, scaled, window_beats, bins)
        if len(null_values) >= 2:
            nulls.append(float(np.median(null_values)))
    if len(nulls) < 5:
        return None
    nulls = np.asarray(nulls)
    modulation = float(np.median(values))
    period = beat_period(beats)
    target_ac, neighbour_ac = period_score(speed, period)
    reading = {
        "modulation": modulation,
        "windows": int(len(values)),
        "null_mean": float(nulls.mean()),
        "gain": float(modulation - nulls.mean()),
        "percentile": float((nulls < modulation).mean()),
        # Spread of the per-window preferred phase, reported and never pooled:
        # a dancer who moves the emphasis around is what the local unit exists
        # to keep, so it must be visible rather than averaged away.
        "phase_spread": float(1.0 - abs(np.exp(1j * phases).mean())) if len(phases) else None,
        "period_frames": float(period),
        "period_win": (bool(target_ac > neighbour_ac)
                       if np.isfinite(target_ac) and np.isfinite(neighbour_ac) else None),
    }
    if harmonic:
        for order in (2, 4):
            per_window = []
            for start in range(0, len(beats) - window_beats):
                first, last = int(beats[start]), int(beats[start + window_beats])
                if last > len(speed) or last - first < bins * 3:
                    continue
                table, _, _ = profile(speed[first:last],
                                      beat_phase(np.asarray(beats[start:start + window_beats + 1]) - first,
                                                 last - first), bins)
                if table is not None:
                    per_window.append(2.0 * abs((np.asarray(table) * np.exp(
                        -2j * np.pi * order * np.arange(bins) / bins)).mean()))
            reading["harmonic_{}".format(order)] = (float(np.median(per_window))
                                                    if per_window else None)
    return reading


# SMPL's 24 joints, grouped the way the operator names body parts: "表现动作可以
# 用四肢,腰跨,肩部,等等".  Feet/hips+knees/hands/torso are the four the ground
# truth's ordering was measured on (2026-09-01); shoulders and elbows are here
# because the operator named shoulders explicitly and a group nobody measures is
# a group that cannot fail.
PARTS = {
    "feet": (7, 8, 10, 11),            # ankles + toes
    "hips_knees": (1, 2, 4, 5),
    "torso": (3, 6, 9, 12),            # spine1-3 + neck
    "shoulders": (13, 14, 16, 17),     # collars + shoulders
    "elbows": (18, 19),
    "hands": (20, 21, 22, 23),         # wrists + hands
}


def load_clip(clip, motion_dir, audio_dir, part=None):
    motion = pathlib.Path(motion_dir) / (clip + ".pkl")
    audio = pathlib.Path(audio_dir) / (clip + ".npy")
    if not (motion.is_file() and audio.is_file()):
        return None, None
    music = np.load(str(audio))
    speed = body_speed(joints_of(str(motion)), part=part)
    length = min(len(music) - 1, len(speed))
    if length < 90:
        return None, None
    beats = np.flatnonzero(music[:length, BEAT_CHANNEL] > 0.5)
    return speed[:length], beats


def score_arm(clips, motion_dir, audio_dir, grids, window_beats=WINDOW_BEATS,
              bins=BINS, harmonic=False, transform=None, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for index, clip in enumerate(clips):
        speed, beats = load_clip(clip, motion_dir, audio_dir)
        if speed is None or len(beats) < window_beats + 3:
            continue
        if transform is not None:
            speed = transform(speed, index)
        others = [g for name, g in grids.items() if name != clip and len(g) >= 6]
        if len(others) > NULL_GRIDS:
            picks = rng.choice(len(others), NULL_GRIDS, replace=False)
            others = [others[i] for i in picks]
        reading = clip_reading(speed, beats, others, window_beats, bins, harmonic)
        if reading:
            reading["clip"] = clip
            rows.append(reading)
    return rows


def summarise(name, rows):
    if not rows:
        return {"arm": name, "clips": 0}
    n = len(rows)
    wins = sum(1 for r in rows if r["gain"] > 0)
    p = sum(comb(n, k) for k in range(wins, n + 1)) / (2.0 ** n)
    period_rows = [r for r in rows if r["period_win"] is not None]
    period_wins = sum(1 for r in period_rows if r["period_win"])
    pn = len(period_rows)
    pp = (sum(comb(pn, k) for k in range(period_wins, pn + 1)) / (2.0 ** pn)) if pn else 1.0
    out = {"arm": name, "clips": n,
           "modulation": float(np.median([r["modulation"] for r in rows])),
           "null": float(np.median([r["null_mean"] for r in rows])),
           "gain": float(np.median([r["gain"] for r in rows])),
           "wins": wins, "p": float(min(1.0, p)),
           "period_wins": period_wins, "period_n": pn, "period_p": float(min(1.0, pp)),
           "phase_spread": float(np.median([r["phase_spread"] for r in rows
                                            if r["phase_spread"] is not None] or [np.nan]))}
    if rows[0].get("harmonic_2") is not None:
        for order in (2, 4):
            key = "harmonic_{}".format(order)
            out[key] = float(np.median([r[key] for r in rows if r.get(key) is not None]))
    return out


def reversed_speed(speed, index):
    return speed[::-1].copy()


def shuffled_speed(speed, index, block=15):
    rng = np.random.default_rng(9000 + index)
    blocks = [speed[i:i + block] for i in range(0, len(speed) - block + 1, block)]
    if len(blocks) < 3:
        return speed
    out = np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])
    return np.pad(out, (0, len(speed) - len(out)), mode="edge")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", required=True)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--ground-truth-dir", required=True)
    parser.add_argument("--arm", action="append", default=[], metavar="NAME=DIR")
    parser.add_argument("--window-beats", type=int, default=WINDOW_BEATS)
    parser.add_argument("--bins", type=int, default=BINS)
    parser.add_argument("--harmonic", action="store_true")
    parser.add_argument("--controls", action="store_true")
    parser.add_argument("--json")
    args = parser.parse_args()

    clips = [line.strip() for line in open(args.clips) if line.strip()]
    grids = {}
    for clip in clips:
        audio = pathlib.Path(args.audio_dir) / (clip + ".npy")
        if audio.is_file():
            music = np.load(str(audio))
            grids[clip] = np.flatnonzero(music[:, BEAT_CHANNEL] > 0.5)

    table = [summarise("ground truth", score_arm(
        clips, args.ground_truth_dir, args.audio_dir, grids,
        args.window_beats, args.bins, args.harmonic))]
    if args.controls:
        for label, transform in (("  control: GT time-reversed", reversed_speed),
                                 ("  control: GT 0.5s shuffled", shuffled_speed)):
            table.append(summarise(label, score_arm(
                clips, args.ground_truth_dir, args.audio_dir, grids,
                args.window_beats, args.bins, args.harmonic, transform=transform)))
    for spec in args.arm:
        name, _, directory = spec.partition("=")
        table.append(summarise(name, score_arm(
            clips, directory, args.audio_dir, grids,
            args.window_beats, args.bins, args.harmonic)))

    header = "{:<32}{:>5}{:>11}{:>8}{:>8}{:>10}{:>10}{:>12}{:>8}".format(
        "arm", "n", "modulation", "null", "gain", "wins", "P", "period win", "phase sd")
    if args.harmonic:
        header += "{:>9}{:>9}".format("2/beat", "4/beat")
    print(header)
    for row in table:
        if not row.get("clips"):
            print("{:<32}{:>5}".format(row["arm"], 0))
            continue
        line = "{:<32}{:>5}{:>11.4f}{:>8.4f}{:>8.4f}{:>6}/{:<3}{:>10.3g}{:>8}/{:<3}{:>8.3f}".format(
            row["arm"], row["clips"], row["modulation"], row["null"], row["gain"],
            row["wins"], row["clips"], row["p"],
            row["period_wins"], row["period_n"], row["phase_spread"])
        if args.harmonic:
            line += "{:>9.4f}{:>9.4f}".format(row["harmonic_2"], row["harmonic_4"])
        print(line)
    print("\nwindow = {} beats; null = up to {} other clips' beat grids rescaled onto "
          "this clip; modulation and gain are MEDIANS over windows then over clips."
          .format(args.window_beats, NULL_GRIDS))

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(table, indent=2), encoding="utf-8")
        print(args.json)


if __name__ == "__main__":
    main()
