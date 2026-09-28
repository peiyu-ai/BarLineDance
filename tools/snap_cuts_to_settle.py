#!/usr/bin/env python3
"""Keep Alg.1's decision about WHERE ROUGHLY to cut; move the cut onto the settle.

The measurement this exists for: Alg.1 places cuts at transitions of a k-means
labelling of self-similarity rows, and frame similarity changes fastest when the
body moves fastest, so its cuts are drawn toward the velocity *peak*.  The paper
wants the opposite -- "a kick, which comprises the preparatory weight shift, leg
extension, and recovery, forms a complete motion segment" -- the boundary belongs
where the motion has settled.  Measured on 400 clean5 clips, only 22.3% of shipped
segments begin and end within 0.1 s of a settled pose, against 23.8% for the same
segment lengths dropped anywhere.

**Why snap instead of cutting at beats directly.**  Cutting at motion beats alone
throws away the only thing Alg.1 contributes -- the judgement that the material
either side is *different*.  That arm already exists as the settle probe's ``_beats``
control, and it is the ruler's ceiling by construction, so it proves nothing.
Snapping separates the two decisions: Alg.1 keeps deciding which frames separate
dissimilar material, and the beat grid fixes only the *phase* of that cut.

**``--max-snap`` is the whole experiment, and it is swept, not chosen.**  With it at
0 this is the input arm; unbounded, it is the beat grid wearing the arm's segment
count.  The interesting question is how small a move buys the alignment, and that
is a curve, not a constant.

**What scoring this with ``probe_settle_alignment.py`` does and does not show.**  It
snaps to the very beats that probe measures against, so a better settle_ratio is
arithmetic, not evidence -- read it only as a check that the snap did what it says.
The claim "this segmentation is better" needs either a human boundary set, which this
repository does not have, or the operator's eye on a contact sheet.  Reporting the
settle_ratio of a settle-snapped arm as a quality result would be the 2026-08-19
mistake in new clothes.

What *is* non-circular here and is therefore reported: how far the cuts had to move,
how many could not move without breaking the minimum length, and what the duration
distribution does.

Usage::

    snap_cuts_to_settle.py --arm runs/wild_v4_seg_r0visual/segmentation.json \\
        --bundle /cache/atomicdance-assets/data/wild3d/wild_v4_raw_bundle \\
        --clips runs/clean5/clips.txt --max-snap 8 \\
        --output runs/wild_v4_seg_r0visual_snap8/segmentation.json
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.convert_motion_to_guofeats import motion_151_to_joints  # noqa: E402
from tools.motion_beats import (canonical_pose, find_motion_beats,   # noqa: E402
                                joint_speed)

FPS = 30.0


class SnapError(RuntimeError):
    """A snap that would produce an invalid segmentation is refused."""


def deep_beats(joints: np.ndarray, beats: Sequence[int], quantile: float,
               smooth: int = 3) -> List[int]:
    """Keep only the beats whose speed is in the lowest ``quantile`` of the clip.

    ``find_motion_beats`` returns every dip that clears a relative prominence, and
    on this corpus that is ~42 per clip -- about one every 0.4 s, dense enough that
    "the nearest beat" is nearly always a few frames away and the snap stops being
    selective.  The operator's ask is the boundary at a *low speed* frame, not at
    any inflection, so this filters to the deepest settles.  ``quantile`` 1.0 keeps
    all of them and reproduces the unfiltered behaviour exactly.
    """
    if quantile >= 1.0:
        return list(beats)
    speed = np.asarray(joint_speed(joints), dtype=np.float64)
    if smooth > 1:
        speed = np.convolve(speed, np.ones(smooth) / smooth, mode="same")
    beats = [int(b) for b in beats if 0 <= b < len(speed)]
    if not beats:
        return []
    cutoff = float(np.quantile(speed[beats], quantile))
    kept = [b for b in beats if speed[b] <= cutoff]
    return kept or beats


def pose_recurrence(joints: np.ndarray, beats: Sequence[int],
                    exclude_frames: int = 15, quantile: float = 0.10) -> np.ndarray:
    """For each beat, the share of distant frames whose pose is close to it.

    The operator's reading of the contact sheet, 2026-08-20, and it is the flaw in
    snapping to the nearest speed minimum: "the dancer reaches a pose and pauses,
    but that pose is not a relaxed one -- it is at the *extended* limit, with the
    recovery still to come."  That is the paper's own kick, whose middle phase is
    the leg at full extension, and whose own sentence rules it out as a boundary:
    "any single frame of the extended leg ... would fail to convey the full event."
    A speed minimum cannot tell that apex from the end of the recovery; both are
    pauses.

    What separates them is not speed but whether the posture *recurs*.  An extreme
    extension is close to unique in a clip; the posture a phrase returns to is one
    the dancer keeps passing through.  Poses are compared with
    ``motion_beats.canonical_pose`` -- root-centred, hips facing +x, shoulder-width
    normalised -- so this is a shape question, not a position or camera one.

    Measured on 120 clean5 clips / 5,091 beats: p25 0.017, p75 0.116, a 7x spread,
    and within a single clip the least- and most-recurring beats read 0.000 and
    0.218.  The signal is large.  It is also **unimodal**, so it supports a
    preference order and not a threshold -- which is why this returns a score and
    the caller ranks with it rather than filtering on it.
    """
    frames = len(joints)
    beats = [int(b) for b in beats if 0 <= b < frames]
    if not beats:
        return np.zeros(0)
    poses = np.stack([canonical_pose(joints[t]) for t in range(frames)]).reshape(frames, -1)
    upper = np.triu_indices(frames, k=1)
    sample = np.linalg.norm(poses[upper[0]] - poses[upper[1]], axis=-1)
    epsilon = float(np.quantile(sample, quantile)) if len(sample) else 0.0
    scores = []
    index = np.arange(frames)
    for beat in beats:
        far = np.abs(index - beat) > exclude_frames
        if not far.any():
            scores.append(0.0)
            continue
        distance = np.linalg.norm(poses[far] - poses[beat], axis=-1)
        scores.append(float((distance < epsilon).mean()))
    return np.asarray(scores)


def stem_of(row: dict) -> str:
    name = row.get("recording_id") or row.get("sequence_id") or ""
    parts = name.split(":")
    return "{}__{}".format(parts[1], parts[2]) if len(parts) == 3 else name


def snap_one(bounds: Sequence[int], beats: Sequence[int], *, max_snap: int,
             min_length: int, weights: Optional[Sequence[float]] = None,
             prefer: str = "nearest") -> Dict[str, object]:
    """Move each interior cut to the nearest beat it may legally reach.

    Left to right, because a cut's legal window depends on where its left
    neighbour ended up.  A cut with no beat inside its window stays put, and is
    counted rather than dropped -- silently leaving cuts unmoved while reporting
    an improved alignment is exactly the kind of half-applied change this
    repository has been bitten by.
    """
    if len(bounds) < 3:
        return {"boundaries": list(bounds), "moved": 0, "unmoved": 0, "distances": []}
    beats = np.asarray(sorted(set(int(b) for b in beats)), dtype=np.int64)
    out = [int(bounds[0])]
    moved, unmoved, distances = 0, 0, []
    for index in range(1, len(bounds) - 1):
        cut = int(bounds[index])
        low = max(out[-1] + min_length, cut - max_snap)
        # The right bound has to leave room for every cut still to come.
        remaining = len(bounds) - 1 - index
        high = min(int(bounds[-1]) - remaining * min_length, cut + max_snap)
        if high < low:
            out.append(max(cut, out[-1] + min_length))
            unmoved += 1
            continue
        window = beats[(beats >= low) & (beats <= high)]
        if not len(window):
            out.append(min(max(cut, low), high))
            unmoved += 1
            continue
        if prefer == "recurrence" and weights is not None:
            mask = (beats >= low) & (beats <= high)
            candidate_weights = np.asarray(weights, dtype=np.float64)[mask]
            # Ties, and near-ties, go to the closer beat: the point is to fix the
            # phase of Alg.1's cut, not to relocate it to the best pose in reach.
            best = candidate_weights.max()
            good = window[candidate_weights >= best - 1e-12]
            target = int(good[np.argmin(np.abs(good - cut))])
        else:
            target = int(window[np.argmin(np.abs(window - cut))])
        out.append(target)
        distances.append(abs(target - cut))
        if target != cut:
            moved += 1
    out.append(int(bounds[-1]))
    for index in range(1, len(out)):
        if out[index] <= out[index - 1]:
            raise SnapError("snap produced a non-increasing boundary list")
    return {"boundaries": out, "moved": moved, "unmoved": unmoved,
            "distances": distances}


def run(arm: pathlib.Path, bundle: pathlib.Path, clips: Optional[set],
        max_snap: int, min_length: int, prominence: float,
        limit: Optional[int], beat_depth: float = 1.0,
        prefer: str = "nearest") -> Dict[str, object]:
    report = json.loads(arm.read_text(encoding="utf-8"))
    motion_of = {}
    with open(bundle / "sequences.jsonl", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            motion_of[stem_of(row)] = bundle / row["motion_path"]

    records = report["records"]
    if clips is not None:
        records = [r for r in records if r["sequence"] in clips]
    records = [r for r in records if r["sequence"] in motion_of]
    if not records:
        raise SnapError("no record of {} has motion in {}".format(arm, bundle))
    if limit:
        step = len(records) / float(limit)
        records = [records[int(i * step)] for i in range(min(limit, len(records)))]

    out_records, moved, unmoved, distances = [], 0, 0, []
    already = 0
    for record in records:
        joints = motion_151_to_joints(np.load(motion_of[record["sequence"]]))
        beats = find_motion_beats(joints, min_separation=3, max_beats=len(joints),
                                  prominence=prominence)
        beats = deep_beats(joints, beats, beat_depth)
        weights = pose_recurrence(joints, beats) if prefer == "recurrence" else None
        snapped = snap_one(record["boundaries"], beats, max_snap=max_snap,
                           min_length=min_length, weights=weights, prefer=prefer)
        bounds = snapped["boundaries"]
        moved += snapped["moved"]
        unmoved += snapped["unmoved"]
        distances.extend(snapped["distances"])
        already += sum(1 for d in snapped["distances"] if d == 0)
        new = dict(record)
        new["boundaries"] = bounds
        new["segments"] = [{"start": a, "end": b, "frames": b - a}
                           for a, b in zip(bounds[:-1], bounds[1:])]
        new["snapped_from"] = str(arm)
        new["snap_max_frames"] = max_snap
        out_records.append(new)

    durations = [(b - a) / FPS for r in out_records
                 for a, b in zip(r["boundaries"][:-1], r["boundaries"][1:])]
    distances_array = np.asarray(distances, dtype=np.float64)
    result = dict(report)
    result["records"] = out_records
    result["sequences"] = len(out_records)
    result["total_segments"] = len(durations)
    result["segments_per_sequence"] = len(durations) / max(1, len(out_records))
    result["duration_seconds"] = {"mean": float(np.mean(durations)),
                                  "median": float(np.median(durations))}
    result["snap"] = {
        "source_arm": str(arm),
        "max_snap_frames": max_snap,
        "min_length_frames": min_length,
        "beat_prominence": prominence,
        "beat_depth_quantile": beat_depth,
        "prefer": prefer,
        "interior_cuts": moved + unmoved,
        "cuts_moved": moved,
        "cuts_already_on_a_beat": already,
        "cuts_with_no_reachable_beat": unmoved,
        "move_frames": {
            "median": float(np.median(distances_array)) if len(distances_array) else None,
            "mean": float(distances_array.mean()) if len(distances_array) else None,
            "p90": float(np.percentile(distances_array, 90)) if len(distances_array) else None,
        },
        "reading": ("settle_ratio measured on this arm is circular -- it snaps to the "
                    "beats the probe scores against.  What is not circular: how far "
                    "cuts moved, how many could not, and the duration distribution."),
    }
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", type=pathlib.Path, required=True)
    parser.add_argument("--bundle", type=pathlib.Path, required=True)
    parser.add_argument("--clips", type=pathlib.Path, default=None)
    parser.add_argument("--max-snap", type=int, required=True,
                        help="frames a cut may move; sweep it, do not pick one")
    parser.add_argument("--min-length", type=int, default=18)
    parser.add_argument("--prominence", type=float, default=0.02)
    parser.add_argument("--beat-depth-quantile", type=float, default=1.0,
                        help="keep only beats in the lowest speed quantile; "
                             "1.0 keeps all and is the unfiltered behaviour")
    parser.add_argument("--prefer", choices=("nearest", "recurrence"),
                        default="nearest",
                        help="which beat inside the window to snap to: the closest, "
                             "or the one whose posture recurs most in the clip. The "
                             "second is the operator's reading -- a held extension "
                             "is a pause too, and only recurrence tells it from the "
                             "end of a recovery.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    clips = None
    if args.clips:
        clips = {line.strip() for line in args.clips.read_text(encoding="utf-8").splitlines()
                 if line.strip()}
    result = run(args.arm, args.bundle, clips, args.max_snap, args.min_length,
                 args.prominence, args.limit, beat_depth=args.beat_depth_quantile,
                 prefer=args.prefer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result), encoding="utf-8")
    snap = result["snap"]
    print("{} clips, {} segments, median {:.3f} s".format(
        result["sequences"], result["total_segments"],
        result["duration_seconds"]["median"]))
    print("  interior cuts {}: moved {} ({:.1%}), already on a beat {}, "
          "no reachable beat {}".format(
              snap["interior_cuts"], snap["cuts_moved"],
              snap["cuts_moved"] / max(1, snap["interior_cuts"]),
              snap["cuts_already_on_a_beat"], snap["cuts_with_no_reachable_beat"]))
    print("  move distance frames: median {} mean {} p90 {}".format(
        snap["move_frames"]["median"], snap["move_frames"]["mean"],
        snap["move_frames"]["p90"]))
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
