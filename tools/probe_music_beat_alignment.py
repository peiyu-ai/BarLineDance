#!/usr/bin/env python3
"""Does the MUSIC BEAT GRID predict where cuts / motion settle points fall?

WHAT THIS ASKS.  A dance clip carries, frame-aligned at 30 fps, a 35-D music
feature (``data/audio_extraction/baseline_features.extract_audio``).  Channel 34
is a one-hot beat grid from ``librosa.beat.beat_track``; channel 33 is one-hot
onset peaks from ``librosa.onset.onset_detect``; channel 0 is the onset
envelope both are derived from.  If cuts -- or the dancer's own settle points --
sat on that grid above chance, the grid would be a cut prior available with no
3D, no S3D, and defined wherever audio exists.

PROVENANCE (CLAUDE.md 2.1 gate 1).  The *question* is the operator's, not the
paper's: atomicDance.pdf uses music only in stage 2 (the D3PM planner is
conditioned on ``c_music``) and in the BAS metric.  Algorithm 1 takes
``Paired T-frame dance video V and T-frame 3D dance motion M`` and nothing else,
and the clustering step encodes segments with TMR.  So this file measures an
*invented* criterion and says so.  What is NOT invented is the event set it
scores: motion beats are the paper's own construction ("selected keyframes
identified as motion beats (local minima of segment-wise joint velocities)"),
reused unmodified from ``tools/motion_beats.find_motion_beats``.

THE NULLS.  CLAUDE.md 2.1 gate 3 records that a level test at beat frames,
averaged across clips, cancelled a real effect because each dancer has her own
phase.  So every statistic here is computed PER CLIP and combined by a paired
sign test over clips, and there are three separate nulls:

* ``shift``   -- the clip's own beat grid rotated circularly by s frames,
  averaged over every s in 1..T-1.  Preserves the beat period and the beat
  count exactly; destroys only the phase.  This is the null for "do events land
  ON beats".
* ``moved``   -- for a segmentation arm only, ``shuffled_spans``: the same
  segment lengths laid down somewhere else.  Same denominator the settle
  criterion uses, so the two are comparable.
* ``lag``     -- for the phase-locking statistic only.  Circular shift cannot
  be the null there: rotating the grid rotates every phase by the same constant
  and leaves the resultant length R unchanged, which is exactly why R survives
  a per-dancer lag.  The null for R is a grid with the WRONG PERIOD (the clip's
  own period +/- delta), i.e. the adjacent-lag control worklog.md already used.

CONTROLS IN BOTH DIRECTIONS (gate 2).  Three event sets with known answers are
scored by the identical code path:

* ``_onsetpeaks`` -- channel 33.  Known-GOOD: onset peaks and the beat grid are
  two different algorithms run on the same onset envelope, so they must agree
  above chance.  If this reads at chance the instrument has no power and no
  null result may be read off it.
* ``_atbeats``    -- events placed exactly on channel 34.  Must read 1.000.
* ``_antiphase``  -- events placed half a beat period off.  Known-BAD; must
  read below the shift null.

Usage::

    python3 tools/probe_music_beat_alignment.py \\
        --bundle /cache/atomicdance-assets/data/wild3d/wild_v4_raw_bundle \\
        --clips runs/clean5/clips.txt --limit 400 \\
        --arm r0visual=runs/wild_v4_seg_r0visual/segmentation.json \\
        --output runs/music_beats/clean5.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.convert_motion_to_guofeats import motion_151_to_joints  # noqa: E402
from tools.motion_beats import find_motion_beats  # noqa: E402
from tools.probe_segmentation_boundaries import (  # noqa: E402
    canonical_key, load_arm, paired_sign_test, shuffled_spans, spans_of,
)

FPS = 30.0
TOLERANCES = (1, 2, 3)
BEAT_CHANNEL = 34
PEAK_CHANNEL = 33


def bundle_index(bundle: pathlib.Path) -> dict:
    """{canonical key: (motion path, music path, frame count)}."""
    index = {}
    with open(pathlib.Path(bundle) / "sequences.jsonl", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = canonical_key(row.get("recording_id") or row["sequence_id"])
            index[key] = (pathlib.Path(bundle) / row["motion_path"],
                          pathlib.Path(bundle) / row["music_path"],
                          int(row["frame_count"]))
    return index


def near_mask(beats: np.ndarray, total: int, tol: int) -> np.ndarray:
    """[T] bool: frame t is within ``tol`` frames of a beat, wrapping circularly."""
    mask = np.zeros(total, dtype=bool)
    for offset in range(-tol, tol + 1):
        mask[(beats + offset) % total] = True
    return mask


def hit_rate(events: np.ndarray, mask: np.ndarray) -> float:
    return float(mask[events % len(mask)].mean())


def shift_null(events: np.ndarray, mask: np.ndarray) -> tuple:
    """(mean hit rate over every circular shift s=1..T-1, right-tail fraction).

    Shifting the *grid* by s is the same as shifting the *events* by -s, which
    is what is done here so the mask is built once.
    """
    total = len(mask)
    shifts = np.arange(1, total)
    grid = mask[(events[None, :] - shifts[:, None]) % total].mean(axis=1)
    return float(grid.mean()), float((grid >= hit_rate(events, mask)).mean())


def beat_phase(events: np.ndarray, beats: np.ndarray) -> np.ndarray:
    """Phase in [0,2pi) of each event within the beat interval containing it.

    Interpolated between the bracketing beats rather than taken modulo a single
    period, so the tracker's jitter (median inter-beat sd 0.54 frames on
    clean5) does not smear the phase.
    """
    if len(beats) < 2:
        return np.asarray([], dtype=np.float64)
    inside = (events >= beats[0]) & (events < beats[-1])
    events = events[inside]
    if len(events) == 0:
        return np.asarray([], dtype=np.float64)
    slot = np.searchsorted(beats, events, side="right") - 1
    low = beats[slot].astype(np.float64)
    high = beats[slot + 1].astype(np.float64)
    return 2.0 * np.pi * (events - low) / np.maximum(high - low, 1e-9)


def resultant(phases: np.ndarray) -> float:
    """Rayleigh R: 0 = phases uniform round the beat, 1 = all at one phase."""
    if len(phases) == 0:
        return float("nan")
    return float(abs(np.exp(1j * phases).mean()))


def shuffled_events(events: np.ndarray, low: int, high: int, rng) -> np.ndarray:
    """Same inter-event gaps, laid down from a random circular offset.

    The honest denominator for the phase-locking statistic.  A circular shift of
    the GRID cannot be the null there (it rotates every phase by the same
    constant and leaves R unchanged), and a wrong-PERIOD grid turned out to be
    confounded: the own-period grid is the tracker's real, jittered grid while
    the wrong-period grids are synthetic uniform ones, so that comparison mixes
    "different period" with "different construction".  Shuffling the events
    against the one real grid holds the construction fixed.
    """
    span = int(high) - int(low)
    if span <= 1 or len(events) < 2:
        return np.asarray(events, dtype=np.int64)
    gaps = np.diff(np.sort(np.asarray(events, dtype=np.int64)))
    rng.shuffle(gaps)
    laid = np.concatenate([[0], np.cumsum(gaps)])
    return np.sort(low + (laid + int(rng.integers(0, span))) % span).astype(np.int64)


def uniform_grid(period: float, total: int, phase0: int = 0) -> np.ndarray:
    count = max(int(total / period), 2)
    return np.clip(np.round(np.arange(count) * period + phase0).astype(np.int64),
                   0, total - 1)


def interior_cuts(boundaries) -> np.ndarray:
    if len(boundaries) < 3:
        return np.asarray([], dtype=np.int64)
    return np.asarray([int(b) for b in boundaries[1:-1]], dtype=np.int64)


def score_events(events: np.ndarray, beats: np.ndarray, total: int) -> dict:
    out = {}
    for tol in TOLERANCES:
        mask = near_mask(beats, total, tol)
        observed = hit_rate(events, mask)
        null, tail = shift_null(events, mask)
        out["hit{}".format(tol)] = observed
        out["null{}".format(tol)] = null
        out["tail{}".format(tol)] = tail
    distance = np.abs(events[:, None] - beats[None, :]).min(axis=1)
    out["mean_nearest"] = float(distance.mean())
    out["n_events"] = int(len(events))
    return out


def summarise(rows: list, rng) -> dict:
    """Per-clip medians plus the paired sign test the clips-are-the-unit rule wants."""
    if not rows:
        return {}
    out = {"clips": len(rows)}
    for tol in TOLERANCES:
        obs = [r["hit{}".format(tol)] for r in rows]
        null = [r["null{}".format(tol)] for r in rows]
        out["hit{}".format(tol)] = float(np.median(obs))
        out["null{}".format(tol)] = float(np.median(null))
        out["pooled_hit{}".format(tol)] = float(
            np.average(obs, weights=[r["n_events"] for r in rows]))
        test = paired_sign_test(null, obs, rng)
        out["sign{}".format(tol)] = test
    out["mean_nearest"] = float(np.median([r["mean_nearest"] for r in rows]))
    out["events_per_clip"] = float(np.median([r["n_events"] for r in rows]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--arm", action="append", default=[])
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--min-length", type=int, default=18)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    index = bundle_index(pathlib.Path(args.bundle))
    arms = {}
    for spec in args.arm:
        name, path = spec.split("=", 1)
        arms[name] = load_arm(pathlib.Path(path))

    keys = [canonical_key(line.strip()) for line in
            open(args.clips, encoding="utf-8") if line.strip()]
    keys = [k for k in keys if k in index]
    if len(keys) > args.limit:
        pick = np.random.default_rng(args.seed).choice(
            len(keys), args.limit, replace=False)
        keys = [keys[int(i)] for i in sorted(pick)]

    collected: dict = {}
    tempo = []
    settle_gaps = []
    lock_rows = []
    for key in keys:
        motion_path, music_path, frames = index[key]
        music = np.load(music_path)
        motion = np.load(motion_path)
        total = min(len(music), len(motion))
        if total < 60:
            continue
        beats = np.flatnonzero(music[:total, BEAT_CHANNEL] > 0.5)
        if len(beats) < 4:
            continue
        period = float(np.median(np.diff(beats)))
        tempo.append(period)

        joints = motion_151_to_joints(motion[:total])
        settle = np.asarray(find_motion_beats(joints, max_beats=10 ** 6),
                            dtype=np.int64)
        settle = settle[(settle >= 0) & (settle < total)]
        if len(settle) >= 2:
            settle_gaps.extend(np.diff(settle).tolist())

        events = {}
        if len(settle) >= 3:
            events["settle"] = settle
        peaks = np.flatnonzero(music[:total, PEAK_CHANNEL] > 0.5)
        if len(peaks) >= 3:
            events["_onsetpeaks"] = peaks
        events["_atbeats"] = beats
        events["_antiphase"] = np.clip(
            beats + int(round(period / 2.0)), 0, total - 1)

        for name, arm in arms.items():
            boundaries = arm.get(key)
            if not boundaries:
                continue
            cuts = interior_cuts(boundaries)
            if len(cuts) < 2:
                continue
            events[name] = cuts
            spans = shuffled_spans(spans_of(boundaries), total, rng)
            moved = interior_cuts([a for a, _ in spans] + [spans[-1][1]]) if spans else []
            if len(moved):
                events[name + "_moved"] = np.asarray(moved, dtype=np.int64)

        for name, ev in events.items():
            ev = np.asarray(ev, dtype=np.int64)
            collected.setdefault(name, []).append(score_events(ev, beats, total))

        # phase locking, invariant to a per-dancer lag: own period vs wrong periods
        row = {"clip": key, "period": period}
        for name in ("settle",) + tuple(arms):
            if name not in events:
                continue
            row[name] = {"own": resultant(beat_phase(events[name], beats)),
                         "n": int(len(beat_phase(events[name], beats)))}
            for delta in (-3, -2, 2, 3):
                grid = uniform_grid(period + delta, total, int(beats[0]))
                row[name]["lag{:+d}".format(delta)] = resultant(
                    beat_phase(events[name], grid))
            grid = uniform_grid(period, total, int(beats[0]))
            row[name]["own_uniform"] = resultant(beat_phase(events[name], grid))
            draws = [resultant(beat_phase(
                shuffled_events(events[name], 0, total, rng), beats))
                for _ in range(8)]
            row[name]["shuffled"] = float(np.nanmean(draws))
        lock_rows.append(row)

    report = {"clips_scored": len(lock_rows),
              "beat_period_frames_median": float(np.median(tempo)),
              "bpm_median": float(60.0 * FPS / np.median(tempo)),
              "bpm_quartiles": [float(60.0 * FPS / p) for p in
                                np.percentile(tempo, [75, 50, 25])],
              "settle_interval_frames_median": float(np.median(settle_gaps)),
              "arms": {}}
    for name, rows in sorted(collected.items()):
        report["arms"][name] = summarise(rows, rng)

    lock = {}
    for name in ("settle",) + tuple(arms):
        own = [r[name]["own"] for r in lock_rows if name in r]
        if len(own) < 10:
            continue
        entry = {"clips": len(own), "R_own": float(np.median(own)),
                 "R_own_uniform": float(np.median(
                     [r[name]["own_uniform"] for r in lock_rows if name in r]))}
        for delta in (-3, -2, 2, 3):
            key = "lag{:+d}".format(delta)
            entry["R_" + key] = float(np.median(
                [r[name][key] for r in lock_rows if name in r]))
        # mean, not max: the max of four wrong-period draws beats a single
        # own-period draw even under the null, which would be a rigged control.
        wrong = [float(np.mean([r[name]["lag-3"], r[name]["lag-2"],
                                r[name]["lag+2"], r[name]["lag+3"]]))
                 for r in lock_rows if name in r]
        entry["R_wrong_period_mean"] = float(np.median(wrong))
        entry["sign_vs_wrong_period"] = paired_sign_test(wrong, own, rng)
        own_uniform = [r[name]["own_uniform"] for r in lock_rows if name in r]
        entry["sign_uniform_own_vs_wrong_period"] = paired_sign_test(
            wrong, own_uniform, rng)
        shuffled = [r[name]["shuffled"] for r in lock_rows if name in r]
        entry["R_shuffled_events"] = float(np.median(shuffled))
        entry["sign_vs_shuffled_events"] = paired_sign_test(shuffled, own, rng)
        # what R a uniformly-distributed phase would give at this event count,
        # so a low R is readable as "no locking" rather than merely "small".
        # MEDIAN, not mean: R for n uniform phases is ~Rayleigh(1/sqrt(2n)),
        # whose median is sqrt(ln2/n).  The mean sqrt(pi/4n) is 6% higher, and
        # comparing that against a median of R would manufacture a deficit.
        entry["R_expected_if_uniform"] = float(np.median(
            [np.sqrt(np.log(2.0) / max(r[name].get("n", 1), 1))
             for r in lock_rows if name in r]))
        lock[name] = entry
    report["phase_locking"] = lock

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        path = pathlib.Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
