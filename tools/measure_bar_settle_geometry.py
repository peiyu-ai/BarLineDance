#!/usr/bin/env python3
"""How far is a bar line from the nearest settled pose, on this corpus?

WHAT DECISION THIS SERVES.  The operator's proposal for the T series is: keep
the k-beat bar grid as the skeleton, and slide the boundary onto the point
where the dancer lands.  Whether that is a small correction to the grid or a
demolition of it is a property of the FOOTAGE, not of the idea, and it is the
number below: the distance from a bar line to the nearest settle.  If that
distance is routinely more than half a beat, then "snap to the nearest settle"
has stopped being a bar grid and the two objectives are in conflict; if it is
a few frames, the snap is free.

THE NULL IS NOT OPTIONAL.  "The nearest settle is close" is unreadable on its
own, because a dense point set is close to everything -- at fourteen settles
over a twenty-second clip, a +-6 frame window already covers a third of the
timeline.  So the same question is asked of a random point set with the SAME
COUNT as this clip's settles, and both numbers are printed.  The real minus the
null is the only part that says settles know anything about bar lines.

WHAT THIS DOES NOT MEASURE.  Not which segmentation is better.  A settle-snapped
arm wins a settle criterion by construction; ``tools/render_cut_comparison.py``
carries that argument and answers it with the operator's eye on a video.  This
tool only says how much room the snap has to work in.

Usage::

    python3 tools/measure_bar_settle_geometry.py \\
        --clips runs/eval_clips_txy29.txt --output runs/t_bar_settle_geometry.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.render_cut_comparison import settle_frames  # noqa: E402
from tools.segment_on_music_beats import choose_phase, grid_bounds  # noqa: E402

FPS = 30.0
BEAT_CHANNEL = 34
ONSET_CHANNEL = 0


class GeometryError(RuntimeError):
    pass


def null_distances(boundaries: np.ndarray, count: int, frames: int,
                   generator: np.random.Generator, draws: int) -> np.ndarray:
    """Distances to a random point set of the same size as the real one."""
    if count <= 0 or frames <= 3:
        return np.zeros(0)
    out = []
    population = np.arange(1, frames - 1)
    size = min(count, len(population))
    for _ in range(draws):
        fake = generator.choice(population, size=size, replace=False)
        out.append(np.abs(boundaries[:, None] - np.sort(fake)[None, :]).min(axis=1))
    return np.concatenate(out)


def phase_hit_shares(beats: np.ndarray, targets: np.ndarray, frames: int,
                     beats_per_segment: int, min_length: int,
                     tolerance: int) -> List[float]:
    """Settle hit share of each of the ``k`` places the bar line could sit."""
    from tools.render_cut_comparison import hit_share

    shares = []
    for phase in range(beats_per_segment):
        bounds = grid_bounds(beats, frames, beats_per_segment, phase,
                             min_length, edges="keep")
        share = hit_share(bounds, targets, tolerance)
        shares.append(0.0 if share is None else float(share))
    return shares


def measure_clip(joints: np.ndarray, music: np.ndarray, *, beats_per_segment: int,
                 min_length: int, prominence: float, separation: int,
                 draws: int, generator: np.random.Generator,
                 tolerance: int = 2) -> Optional[Dict[str, object]]:
    frames = int(min(len(joints), len(music)))
    joints, music = joints[:frames], music[:frames]
    beats = np.flatnonzero(music[:, BEAT_CHANNEL] > 0.5)
    if len(beats) < beats_per_segment + 2:
        return None
    settles = settle_frames(joints, prominence=prominence, min_separation=separation)
    if not len(settles):
        return None
    period = float(np.median(np.diff(beats)))
    phase = int(choose_phase(beats, music[:, ONSET_CHANNEL], beats_per_segment)["phase"])
    bounds = grid_bounds(beats, frames, beats_per_segment, phase, min_length, edges="keep")
    interior = np.asarray([int(b) for b in bounds][1:-1], dtype=int)
    if not len(interior):
        return None
    real = np.abs(interior[:, None] - settles[None, :]).min(axis=1)
    null = null_distances(interior, len(settles), frames, generator, draws)
    # "Move the whole rigid grid onto the landings" -- the conservative reading
    # of sliding the anchor.  The number it produces is a MAXIMUM over k
    # candidates, so it is biased upward even when the settles know nothing
    # about the grid, and reporting it without the matching maximum over a
    # random point set is exactly the shape of error CLAUDE.md 2.1 records.
    shares = phase_hit_shares(beats, settles, frames, beats_per_segment,
                              min_length, tolerance)
    energy_phase_share = shares[phase]
    null_best, null_energy = [], []
    for _ in range(draws):
        fake = np.sort(generator.choice(np.arange(1, frames - 1),
                                        size=min(len(settles), frames - 2),
                                        replace=False))
        fake_shares = phase_hit_shares(beats, fake, frames, beats_per_segment,
                                       min_length, tolerance)
        null_best.append(max(fake_shares))
        null_energy.append(fake_shares[phase])
    row: Dict[str, object] = {
        "frames": frames,
        "phase_best_share": float(max(shares)),
        "phase_best_share_null": float(np.mean(null_best)),
        "phase_energy_share": float(energy_phase_share),
        "phase_energy_share_null": float(np.mean(null_energy)),
        "phase_best_beats_its_null": bool(max(shares) > np.mean(null_best)),
        "beat_period_frames": period,
        "settles": int(len(settles)),
        "bar_lines": int(len(interior)),
        "distance_median_frames": float(np.median(real)),
        "distance_median_frames_null": float(np.median(null)) if len(null) else None,
        "beyond_half_beat": float((real > period / 2.0).mean()),
    }
    if len(settles) >= 2:
        row["settle_gap_seconds"] = float(np.median(np.diff(settles)) / FPS)
        row["settle_gap_beats"] = float(np.median(np.diff(settles)) / period)
    for window in (3, 6, 9, 12):
        row["within_{}".format(window)] = float((real <= window).mean())
        row["within_{}_null".format(window)] = (
            float((null <= window).mean()) if len(null) else None)
    return row


def run(clips: Sequence[str], motion_dir: pathlib.Path, audio_dir: pathlib.Path,
        *, beats_per_segment: int, min_length: int, prominence: float,
        separation: int, draws: int, seed: int, tolerance: int = 2) -> Dict[str, object]:
    generator = np.random.default_rng(seed)
    rows: List[Dict[str, object]] = []
    for record in clips:
        motion_path = motion_dir / (record + ".pkl")
        audio_path = audio_dir / (record + ".npy")
        if not motion_path.is_file() or not audio_path.is_file():
            continue
        payload = pickle.load(motion_path.open("rb"))
        joints = payload["full_pose"] if isinstance(payload, dict) else payload
        row = measure_clip(np.asarray(joints), np.load(audio_path),
                           beats_per_segment=beats_per_segment, min_length=min_length,
                           prominence=prominence, separation=separation,
                           draws=draws, generator=generator, tolerance=tolerance)
        if row is not None:
            row["clip"] = record
            rows.append(row)
    if not rows:
        raise GeometryError("no clip had both a beat grid and a settle")

    def mean(key: str) -> Optional[float]:
        values = [r[key] for r in rows if r.get(key) is not None]
        return round(float(np.mean(values)), 4) if values else None

    summary = {key: mean(key) for key in (
        "beat_period_frames", "settle_gap_seconds", "settle_gap_beats",
        "distance_median_frames", "distance_median_frames_null", "beyond_half_beat",
        "phase_best_share", "phase_best_share_null",
        "phase_energy_share", "phase_energy_share_null")}
    summary["clips"] = len(rows)
    summary["clips_where_best_phase_beats_its_null"] = int(
        sum(1 for r in rows if r.get("phase_best_beats_its_null")))
    summary["bar_lines_total"] = int(sum(r["bar_lines"] for r in rows))
    for window in (3, 6, 9, 12):
        summary["within_{}".format(window)] = mean("within_{}".format(window))
        summary["within_{}_null".format(window)] = mean("within_{}_null".format(window))
    return {
        "schema_version": "atomicdance-bar-settle-geometry-v1",
        "config": {"beats_per_segment": beats_per_segment, "min_length": min_length,
                   "settle_prominence": prominence, "settle_separation": separation,
                   "null_draws": draws, "seed": seed, "tolerance": tolerance},
        "summary": summary,
        "per_clip": rows,
        "how_to_read": (
            "distance_median_frames is the snap a boundary would have to make to "
            "reach the nearest settle; compare it against "
            "distance_median_frames_null, which is the same question asked of a "
            "random point set of the same size, and against half a beat "
            "(beat_period_frames / 2), past which the grid is no longer a grid. "
            "within_N minus within_N_null is the only part that says settles know "
            "where the bar lines are. phase_best_share is the best of k bar-line "
            "positions and is therefore a maximum: read it ONLY against "
            "phase_best_share_null, the same maximum taken over a random point "
            "set. If the null is as high, choosing the bar phase by settle "
            "alignment buys nothing."),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", type=pathlib.Path, required=True)
    parser.add_argument("--motion-dir", type=pathlib.Path,
                        default=pathlib.Path("runs/wild_v5_song_gt_eval/motion"))
    parser.add_argument("--audio-dir", type=pathlib.Path,
                        default=pathlib.Path("runs/wild_v5_song_gt_eval/audio"))
    parser.add_argument("--beats-per-segment", type=int, default=4)
    parser.add_argument("--min-length", type=int, default=18)
    parser.add_argument("--settle-prominence", type=float, default=0.30)
    parser.add_argument("--settle-separation", type=int, default=15)
    parser.add_argument("--null-draws", type=int, default=200)
    parser.add_argument("--tolerance", type=int, default=2,
                        help="frames within which a bar line counts as landing "
                             "on a settle, for the phase columns")
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    clips = [line.strip() for line in args.clips.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    report = run(clips, args.motion_dir, args.audio_dir,
                 beats_per_segment=args.beats_per_segment, min_length=args.min_length,
                 prominence=args.settle_prominence, separation=args.settle_separation,
                 draws=args.null_draws, seed=args.seed, tolerance=args.tolerance)
    summary = report["summary"]
    print("{} clips, {} bar lines at {} beats".format(
        summary["clips"], summary["bar_lines_total"], args.beats_per_segment))
    print("beat period {:.2f} frames ({:.3f} s); {} beats = {:.2f} s".format(
        summary["beat_period_frames"], summary["beat_period_frames"] / FPS,
        args.beats_per_segment,
        args.beats_per_segment * summary["beat_period_frames"] / FPS))
    print("settle every {:.3f} s = {:.2f} beats".format(
        summary["settle_gap_seconds"], summary["settle_gap_beats"]))
    print("bar line to nearest settle: median {:.2f} frames, null {:.2f}".format(
        summary["distance_median_frames"], summary["distance_median_frames_null"]))
    for window in (3, 6, 9, 12):
        print("  within +-{:2d} frames ({:.2f} s): {:.3f}   null {:.3f}".format(
            window, window / FPS, summary["within_{}".format(window)],
            summary["within_{}_null".format(window)]))
    print("beyond half a beat (grid would be destroyed): {:.3f}".format(
        summary["beyond_half_beat"]))
    print("choosing the bar phase by settles: best-of-{} {:.4f}, its null {:.4f} "
          "({} of {} clips beat their own null)".format(
              args.beats_per_segment, summary["phase_best_share"],
              summary["phase_best_share_null"],
              summary["clips_where_best_phase_beats_its_null"], summary["clips"]))
    print("  the onset-energy phase for comparison: {:.4f}, its null {:.4f}".format(
        summary["phase_energy_share"], summary["phase_energy_share_null"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print("wrote {}".format(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
