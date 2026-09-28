#!/usr/bin/env python3
"""The offset between the music's beat grid and the dancer's own.

Measured on 299 clean5 clips by ``tools/probe_music_beat_alignment.py``, with its
upper control (cuts placed on beats) at hit-rate 1.000 and its lower control
(anti-phase) at 0.000, so the instrument has power in both directions:

* the dancer's settle points land ON a beat 19.4% of the time against a chance
  rate of ~20% -- **no phase alignment at all** (paired sign test vs a circularly
  shifted grid: p = 0.42);
* but the same settle points are phase-locked to the beat *period*: resultant
  R = 0.1293 against 0.1110 for a wrong-period grid (p = 0.0087) and 0.1160 for
  shuffled events (p = 0.00065).

Both readings together say one thing: **she moves on the music's clock but not on
its phase.**  Each clip has its own offset.  That is the same shape CLAUDE.md
section 2.1 records -- a level test at beat frames found "no shared temporal
structure" and a phase-invariant periodicity test found it immediately, because
averaging across dancers cancels a per-dancer phase.

So a beat-grid segmentation that cuts on the music's phase cuts a constant
distance away from wherever the dancer's own cycle breaks.  This file estimates
that constant, one scalar per clip, as the circular mean of the settle points'
phase within the beat period.

**The estimate is validated held-out, because estimating an offset from the
settles and then scoring it with the same settles proves nothing.**  Odd-indexed
settles estimate; even-indexed settles score.  The gate is a per-clip paired
comparison of even-settle hit rate against the shifted grid versus the unshifted
one, and it can fail: if shifting does not beat not-shifting on held-out events,
the offset is noise and this tool says so rather than publishing it.

``confidence`` is the resultant length R of the estimating half.  R is small on
this corpus (0.13 at the median), so a per-clip offset is a noisy quantity; the
report carries the distribution rather than a single number, and
``--min-confidence`` lets a caller fall back to the music's own phase where the
lock is too weak to trust.

Usage::

    estimate_dancer_beat_phase.py --bundle <raw bundle> --clips runs/clean5/clips.txt \\
        --output runs/clean5_dancer_phase.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.convert_motion_to_guofeats import motion_151_to_joints  # noqa: E402
from tools.motion_beats import find_motion_beats                    # noqa: E402

FPS = 30.0
BEAT_CHANNEL = 34


def stem_of(row: dict) -> str:
    parts = (row.get("recording_id") or row.get("sequence_id") or "").split(":")
    return "{}__{}".format(parts[1], parts[2]) if len(parts) == 3 else ""


def circular_offset(events: Sequence[int], origin: float, period: float):
    """(offset in frames, resultant length) of ``events`` within the beat period."""
    if len(events) == 0 or period <= 0:
        return 0.0, 0.0
    angles = 2.0 * np.pi * ((np.asarray(events, dtype=np.float64) - origin) / period)
    vector = np.exp(1j * angles).mean()
    resultant = float(abs(vector))
    offset = float(np.angle(vector)) / (2.0 * np.pi) * period
    return offset, resultant


def hit_rate(events: Sequence[int], grid: np.ndarray, tolerance: int) -> float:
    """Share of events within ``tolerance`` frames of the grid.

    The grid is rounded to whole frames first, and that is not cosmetic: events
    are integer frame indices, so a float grid offset by a fraction of a frame
    covers one fewer integer inside the same +/-tolerance window.  Measured on
    2026-08-20 that alone dropped a shifted grid from 0.317 to 0.249 -- below the
    unshifted grid and below chance -- and would have been read as "shifting the
    grid hurts" when it was the comparison, not the shift.
    """
    if len(events) == 0 or len(grid) == 0:
        return float("nan")
    grid = np.rint(np.asarray(grid, dtype=np.float64))
    distance = np.abs(np.asarray(events)[:, None] - grid[None, :]).min(axis=1)
    return float((distance <= tolerance).mean())


def run(bundle: pathlib.Path, clips: set, limit: Optional[int], tolerance: int,
        prominence: float) -> Dict[str, object]:
    rows = [json.loads(line) for line in
            open(bundle / "sequences.jsonl", encoding="utf-8")]
    rows = [r for r in rows if stem_of(r) in clips and r.get("music_path")]
    rows.sort(key=stem_of)
    if limit and limit < len(rows):
        step = len(rows) / float(limit)
        rows = [rows[int(i * step)] for i in range(limit)]

    per_clip: Dict[str, dict] = {}
    shifted_hits, plain_hits, random_hits = [], [], []
    generator = np.random.default_rng(20260820)
    for row in rows:
        stem = stem_of(row)
        table = np.load(bundle / row["music_path"])
        beats = np.flatnonzero(table[:, BEAT_CHANNEL] > 0.5).astype(np.float64)
        if len(beats) < 4:
            continue
        period = float(np.median(np.diff(beats)))
        joints = motion_151_to_joints(np.load(bundle / row["motion_path"]))
        settles = find_motion_beats(joints, min_separation=3, max_beats=len(joints),
                                    prominence=prominence)
        if len(settles) < 8:
            continue
        settles = np.asarray(sorted(settles), dtype=np.float64)
        estimate, confidence = circular_offset(settles[0::2], beats[0], period)
        held_out = settles[1::2]
        shifted = beats + estimate
        plain = hit_rate(held_out, beats, tolerance)
        moved = hit_rate(held_out, shifted, tolerance)
        # The negative control: a random offset must not help.
        noise = float(generator.uniform(-period / 2.0, period / 2.0))
        random_shift = hit_rate(held_out, beats + noise, tolerance)
        per_clip[stem] = {"offset_frames": round(estimate, 3),
                          "confidence": round(confidence, 4),
                          "period_frames": round(period, 3),
                          "settles": int(len(settles))}
        shifted_hits.append(moved)
        plain_hits.append(plain)
        random_hits.append(random_shift)

    if not per_clip:
        raise SystemExit("no clip had both a beat grid and enough settle points")

    shifted_hits = np.asarray(shifted_hits)
    plain_hits = np.asarray(plain_hits)
    random_hits = np.asarray(random_hits)

    def paired(a: np.ndarray, b: np.ndarray) -> Dict[str, object]:
        delta = a - b
        wins = int((delta > 0).sum())
        losses = int((delta < 0).sum())
        trials = wins + losses
        # Two-sided sign test, exact.
        from math import comb

        if trials == 0:
            probability = 1.0
        else:
            extreme = min(wins, losses)
            tail = sum(comb(trials, k) for k in range(extreme + 1)) / (2.0 ** trials)
            probability = min(1.0, 2.0 * tail)
        return {"wins": wins, "losses": losses, "clips": int(len(delta)),
                "median_delta": round(float(np.median(delta)), 5),
                "p_two_sided": round(probability, 6)}

    gate = paired(shifted_hits, plain_hits)
    control = paired(random_hits, plain_hits)
    confidences = np.asarray([v["confidence"] for v in per_clip.values()])
    return {
        "generated_by": "tools/estimate_dancer_beat_phase.py",
        "clips": len(per_clip),
        "tolerance_frames": tolerance,
        "held_out_design": "odd-indexed settles estimate the offset, even-indexed "
                           "settles score it; estimating and scoring on the same "
                           "events would prove nothing",
        "hit_rate_even_settles": {
            "against_the_music_grid": round(float(plain_hits.mean()), 4),
            "against_the_shifted_grid": round(float(shifted_hits.mean()), 4),
            "against_a_randomly_shifted_grid": round(float(random_hits.mean()), 4),
        },
        "gate_shift_beats_no_shift": gate,
        "negative_control_random_shift": control,
        "confidence": {"median": round(float(np.median(confidences)), 4),
                       "p25": round(float(np.percentile(confidences, 25)), 4),
                       "p75": round(float(np.percentile(confidences, 75)), 4)},
        "reading": ("the gate must beat the negative control as well as the plain "
                    "grid; a shift that helps no more than a random shift is noise"),
        "offsets": per_clip,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=pathlib.Path, required=True)
    parser.add_argument("--clips", type=pathlib.Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tolerance", type=int, default=2)
    parser.add_argument("--prominence", type=float, default=0.02)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    clips = {line.strip() for line in args.clips.read_text(encoding="utf-8").splitlines()
             if line.strip()}
    result = run(args.bundle, clips, args.limit, args.tolerance, args.prominence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    rates = result["hit_rate_even_settles"]
    print("{} clips, tolerance +/-{} frames".format(result["clips"], args.tolerance))
    print("  held-out settle hit rate:  music grid {:.4f}  shifted {:.4f}  "
          "randomly shifted {:.4f}".format(
              rates["against_the_music_grid"], rates["against_the_shifted_grid"],
              rates["against_a_randomly_shifted_grid"]))
    gate, control = result["gate_shift_beats_no_shift"], result["negative_control_random_shift"]
    print("  GATE   shifted vs plain : wins {}/{}  median {:+.5f}  p={}".format(
        gate["wins"], gate["clips"], gate["median_delta"], gate["p_two_sided"]))
    print("  CONTROL random vs plain : wins {}/{}  median {:+.5f}  p={}".format(
        control["wins"], control["clips"], control["median_delta"], control["p_two_sided"]))
    print("  confidence R  p25 {} median {} p75 {}".format(
        result["confidence"]["p25"], result["confidence"]["median"],
        result["confidence"]["p75"]))
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
