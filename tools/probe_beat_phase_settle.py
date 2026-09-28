#!/usr/bin/env python3
"""Does cutting ON the beat also cut where the dancer SETTLES?  Phase is the null.

WHAT THIS ASKS, AND WHY IT IS NOT THE SAME QUESTION AS ``probe_settle_alignment``.

Arm D (``tools/segment_on_music_beats.py --mode grid --beats-per-segment 4``)
places every cut on the music beat grid, so its cuts inherit two things at once:
a *period* (four beats, ~1.9 s here) and a *phase* (which beat of the bar, which
``--phase energy`` guesses because ``librosa.beat.beat_track`` returns beats and
not bars).  ``probe_settle_alignment`` controls for the period -- its
``shuffled_spans`` null keeps each arm's own segment lengths and drops them
anywhere -- but that null also destroys the *regularity* of the grid, so an arm
whose cuts are periodic is being compared against cuts that are not.

This file controls for the period a second way, by keeping the grid intact and
moving only its phase:

    control cuts = the arm's own cuts, all shifted by the same delta frames,
                   wrapped inside [1, frames-1]

Every inter-cut gap survives the shift (one gap moves through the wrap), so the
control is periodic exactly where the arm is, has the same number of cuts, and
differs from the arm ONLY in where the grid sits against the dancer.  A reading
above its control therefore says "this phase of this grid lands nearer the
settle points than another phase of the same grid would" -- which is the whole
of what "cut on the beat" could buy on the motion side, and nothing more.

THE SETTLE TARGET (provenance, CLAUDE.md 2.1 gate 1).  Not invented here: the
motion beat, ``tools/motion_beats.find_motion_beats``, unmodified, called once
per clip with ``max_beats`` raised -- the same call and the same scope change
that ``tools/probe_settle_alignment`` already documents.  The paper's sentence
behind it (atomicDance.pdf 3.2): keyframes are "identified as motion beats
(local minima of segment-wise joint velocities)", and a complete segment is "a
kick, which comprises the preparatory weight shift, leg extension, and
recovery" -- so the boundary belongs at the recovery, i.e. at a speed minimum.

THREE READOUTS, so a threshold artefact cannot carry the answer alone:

* ``distance``  mean frames from each interior cut to the nearest motion beat.
* ``hit@k``     share of cuts within k frames of one.  Chance is high here
                (beats fall about one per 14 frames), which is exactly why it
                is reported as a ratio to the shifted control and never alone.
* ``speed``     mean smoothed joint speed AT the cut frames, divided by the
                clip's median speed.  Threshold-free, and the only one of the
                three that does not depend on the beat-pruning rule.  Lower is
                better: the paper's boundary is where the motion has settled.

GATES 2 AND 3 (CLAUDE.md 2.1): two synthetic arms are built from the reference
arm's spans and scored identically, so a null reading can be told apart from a
blind ruler.

* ``_beats``  cuts placed AT motion beats, same count, same min-length.
              KNOWN-GOOD.  If it does not read high, stop.
* ``_peaks``  cuts placed at the maxima of the same speed trace.  KNOWN-BAD --
              this is the placement CLAUDE.md 2.1 names as the defect.  If it
              does not read low, stop.
* ``_null``   ONE circular shift of the reference arm's cuts, scored against
              eight fresh ones.  It is the null of this whole procedure and
              must read ~1.000 with a paired sign test near 50%.  Anything else
              is a bug in the readout, not a finding -- and one of the three
              readouts *is* biased: ``hit@k`` for an arm with ~9 cuts is a mean
              over ~9 Bernoulli draws, whose median sits below its own
              expectation, while the control averages eight times as many, so
              the arm loses a coin flip it should tie.  ``_null`` measures that
              bias instead of leaving it to be discovered as a finding.

CONFOUND ACCOUNTING, stated because the point of this file is to be the reading
that is NOT confounded with granularity:

* the control has the arm's own cut count and own gaps, so an arm that cuts
  twice as often is compared against a control that also cuts twice as often;
* it does NOT make arms comparable to each other in absolute terms -- a finer
  arm still has more cuts and a smaller raw ``distance``.  Compare each arm to
  its own control, never one arm's ``distance`` to another's.

ONE HONEST BIAS, measured and reported.  A shift by delta that happens to be a
whole number of grid periods reproduces the arm's own cuts, so those draws pull
the control towards the arm and the ratio towards 1.  ``control_equals_arm``
reports the share of draws whose shifted cut set matches the arm's within one
frame; it is a conservative bias (it can only hide an effect, never invent one).

Usage::

    python3 tools/probe_beat_phase_settle.py \\
        --bundle /cache/atomicdance-assets/data/wild3d/wild_v4_raw_bundle \\
        --clips runs/clean5/clips.txt --limit 2103 \\
        --arm D=runs/wild_v4_seg_beat4/segmentation.json \\
        --output runs/beat_phase/D.json
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
    canonical_key, load_arm, load_bundle_index, paired_sign_test, spans_of,
)
from tools.probe_settle_alignment import (  # noqa: E402
    beat_spans, interior_cuts, load_frame_counts, peak_spans_speed, smoothed_speed,
)

FPS = 30.0
BEAT_CHANNEL = 34


def music_index(bundle: pathlib.Path) -> dict:
    """{canonical key: music_35 path}; the beat grid lives in channel 34."""
    out = {}
    with open(bundle / "sequences.jsonl", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rel = row.get("music_path") or row["assets"].get("music_35")
            if rel:
                out[canonical_key(row.get("recording_id") or row["sequence_id"])] = bundle / rel
    return out


def shift_cuts(cuts: np.ndarray, frames: int, delta: int) -> np.ndarray:
    """The arm's cuts rotated inside [1, frames-1] -- every gap survives."""
    span = frames - 1
    return np.sort(1 + ((cuts - 1 + delta) % span))


def readouts(cuts: np.ndarray, beats: np.ndarray, speed: np.ndarray,
             median_speed: float) -> dict:
    if len(cuts) == 0 or len(beats) == 0 or median_speed <= 0:
        return {}
    distance = np.abs(cuts[:, None] - beats[None, :]).min(axis=1)
    inside = cuts[(cuts >= 0) & (cuts < len(speed))]
    return {
        "distance": float(distance.mean()),
        "hit1": float((distance <= 1).mean()),
        "hit2": float((distance <= 2).mean()),
        "hit3": float((distance <= 3).mean()),
        "speed": float(speed[inside].mean() / median_speed) if len(inside) else float("nan"),
    }


def score_clip(cuts, beats, speed, median_speed, frames, rng, draws):
    arm = readouts(cuts, beats, speed, median_speed)
    if not arm:
        return None
    controls, identical = [], 0
    for _ in range(draws):
        delta = int(rng.integers(1, max(2, frames - 1)))
        moved = shift_cuts(cuts, frames, delta)
        if len(moved) == len(cuts) and np.abs(moved - cuts).max() <= 1:
            identical += 1
        row = readouts(moved, beats, speed, median_speed)
        if row:
            controls.append(row)
    if not controls:
        return None
    control = {k: float(np.mean([c[k] for c in controls])) for k in arm}
    return {
        "arm": arm, "control": control, "cuts": int(len(cuts)),
        "control_equals_arm": identical / draws,
        # higher is better for all three
        "distance_ratio": (control["distance"] + 0.5) / (arm["distance"] + 0.5),
        "hit2_ratio": (arm["hit2"] + 1e-3) / (control["hit2"] + 1e-3),
        "speed_ratio": control["speed"] / arm["speed"] if arm["speed"] > 0 else float("nan"),
    }


def probe(bundle, clips, arms, limit, seed, min_length, draws):
    index = load_bundle_index(bundle)
    music = music_index(bundle)
    frame_count = load_frame_counts(bundle)
    wanted = [canonical_key(line.strip()) for line in
              pathlib.Path(clips).read_text(encoding="utf-8").splitlines() if line.strip()]
    shared = [k for k in wanted if k in index]
    missing_bundle = len(wanted) - len(shared)
    stale = {name: 0 for name in arms}
    keep = []
    for key in shared:
        ok = True
        for name, arm in arms.items():
            bounds = arm.get(key)
            if not bounds or len(bounds) < 3 or bounds[-1] != frame_count.get(key):
                stale[name] += 1
                ok = False
        if ok:
            keep.append(key)
    shared = keep
    if limit and limit < len(shared):
        stride = len(shared) / limit                 # even stride, never a prefix
        shared = [shared[int(i * stride)] for i in range(limit)]
    if not shared:
        raise SystemExit("no clip survives the arm/bundle intersection")

    names = list(arms) + ["_beats", "_peaks", "_null"]
    per_clip = {name: [] for name in names}
    reference_arm = next(iter(arms))
    rng = np.random.default_rng(seed)
    grid_periods = []

    for key in shared:
        motion = np.load(index[key]).astype(np.float32)
        joints = motion_151_to_joints(motion)
        speed = smoothed_speed(joints)
        median_speed = float(np.median(speed))
        beats = np.asarray(find_motion_beats(joints, max_beats=10 ** 6), dtype=np.int64)
        frames = len(motion)
        if key in music:
            table = np.load(music[key])
            grid = np.flatnonzero(table[:, BEAT_CHANNEL] > 0.5)
            if len(grid) > 2:
                grid_periods.append(float(np.median(np.diff(grid))))
        base = spans_of(arms[reference_arm][key])
        built = {
            "_beats": beat_spans(beats, speed, base, min_length),
            "_peaks": peak_spans_speed(speed, base, min_length),
        }
        null_cuts = shift_cuts(interior_cuts(base), frames,
                               int(rng.integers(1, max(2, frames - 1))))
        for name, arm in arms.items():
            per_clip[name].append(score_clip(
                interior_cuts(spans_of(arm[key])), beats, speed, median_speed,
                frames, rng, draws))
        for name, spans in built.items():
            per_clip[name].append(score_clip(
                interior_cuts(spans), beats, speed, median_speed, frames, rng, draws)
                if len(spans) >= 2 else None)
        per_clip["_null"].append(score_clip(null_cuts, beats, speed, median_speed,
                                            frames, rng, draws))

    out = {
        "clips_requested": len(wanted), "clips_scored": len(shared),
        "clips_not_in_bundle": missing_bundle, "clips_dropped_stale": stale,
        "reference_arm": reference_arm, "random_draws": draws, "seed": seed,
        "min_length_frames": min_length,
        "music_grid_period_frames_median": (float(np.median(grid_periods))
                                            if grid_periods else None),
        "arms": {},
    }
    for name in names:
        rows = [r for r in per_clip[name] if r]
        if not rows:
            continue
        pick = lambda f: np.asarray([r[f] for r in rows], dtype=np.float64)
        arm_of = lambda f: np.asarray([r["arm"][f] for r in rows], dtype=np.float64)
        ctl_of = lambda f: np.asarray([r["control"][f] for r in rows], dtype=np.float64)
        row = {
            "clips": len(rows),
            "interior_cuts_total": int(pick("cuts").sum()),
            "control_equals_arm": float(np.mean(pick("control_equals_arm"))),
            "distance_frames": float(np.median(arm_of("distance"))),
            "control_distance_frames": float(np.median(ctl_of("distance"))),
            "distance_ratio": float(np.median(pick("distance_ratio"))),
            "speed_at_cut": float(np.median(arm_of("speed"))),
            "control_speed_at_cut": float(np.median(ctl_of("speed"))),
            "speed_ratio": float(np.median(pick("speed_ratio"))),
        }
        for k in ("hit1", "hit2", "hit3"):
            row[k] = float(np.median(arm_of(k)))
            row["control_" + k] = float(np.median(ctl_of(k)))
        row["hit2_ratio"] = float(np.median(pick("hit2_ratio")))
        # per-clip paired sign tests, arm against its own shifted control
        for field, better_high in (("distance", False), ("hit2", True), ("speed", False)):
            a = [r["control"][field] if r else float("nan") for r in per_clip[name]]
            b = [r["arm"][field] if r else float("nan") for r in per_clip[name]]
            if not better_high:
                a, b = [-x for x in a], [-x for x in b]
            result = paired_sign_test(a, b, np.random.default_rng(seed))
            if result:
                row["paired_" + field] = result
        out["arms"][name] = row
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=pathlib.Path, required=True)
    parser.add_argument("--clips", type=pathlib.Path, required=True)
    parser.add_argument("--arm", action="append", default=[], required=True)
    parser.add_argument("--limit", type=int, default=2103)
    parser.add_argument("--min-length", type=int, default=18)
    parser.add_argument("--random-draws", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    arms = {}
    for spec in args.arm:
        name, path = spec.split("=", 1)
        arms[name] = load_arm(pathlib.Path(path))
    report = probe(args.bundle, args.clips, arms, args.limit, args.seed,
                   args.min_length, args.random_draws)

    print("clips scored {} of {} | control = the arm's own cuts rotated by a random "
          "delta ({} draws)".format(report["clips_scored"], report["clips_requested"],
                                    report["random_draws"]))
    print("dropped: not in bundle {}, stale-per-arm {} | music grid period median "
          "{:.1f} frames".format(report["clips_not_in_bundle"],
                                 report["clips_dropped_stale"],
                                 report["music_grid_period_frames_median"] or float("nan")))
    print("%-10s %6s %8s %8s %7s %7s %7s %7s %7s %7s %7s" % (
        "arm", "clips", "cut->mb", "ctl->mb", "distR", "hit2", "ctl", "hit2R",
        "spd@cut", "ctl", "spdR"))
    for name, row in report["arms"].items():
        print("%-10s %6d %8.2f %8.2f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f" % (
            name, row["clips"], row["distance_frames"], row["control_distance_frames"],
            row["distance_ratio"], row["hit2"], row["control_hit2"], row["hit2_ratio"],
            row["speed_at_cut"], row["control_speed_at_cut"], row["speed_ratio"]))
    print("\nper-clip paired sign tests, arm vs its OWN phase-shifted control:")
    for name, row in report["arms"].items():
        bits = []
        for field in ("distance", "hit2", "speed"):
            r = row.get("paired_" + field)
            if r:
                bits.append("%s %d/%d p=%.4f" % (field, r["wins"], r["clips"],
                                                 r["p_two_sided"]))
        print("  %-10s %s  [draws matching the arm: %.1f%%]" % (
            name, " | ".join(bits), 100 * row["control_equals_arm"]))
    print("\n_beats is KNOWN-GOOD (cuts at motion beats), _peaks is KNOWN-BAD "
          "(cuts at speed maxima), _null is one shift scored against eight and "
          "must read ~1.000 -- read every arm against _null, not against 1.000.  "
          "Compare each arm to ITS OWN control column, never one arm's raw "
          "number to another's.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("report -> {}".format(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
