#!/usr/bin/env python3
"""Do the cuts land where the movement SETTLES?  With controls in both directions.

WHY THIS EXISTS.  ``tools/probe_segmentation_boundaries.py`` scores a cut by
*boundary contrast* -- the frame-to-frame change at the cut over the median
change inside the segment, higher-is-better.  That criterion can fail (a
same-lengths-moved control reads 0.997) but it points the wrong way: it is
maximised by cutting at the velocity **peak**, and the paper's own example says
a complete motion segment is "a kick, which comprises the preparatory weight
shift, leg extension, and recovery" -- the boundary belongs at the *recovery*,
i.e. where the motion has settled, not where it is fastest.  CLAUDE.md section
2.1 records that miss.  This file is the criterion that points the right way.

PROVENANCE (gate 1).  Not invented here.  Two sentences of atomicDance.pdf,
both verbatim:

* section 3.2: "We aim to extract complete motion processes that are clearly
  distinct from their temporal context.  For example, a kick, which comprises
  the preparatory weight shift, leg extension, and recovery, forms a complete
  motion segment".
* section 3.2, the M3 caption schema: "we use PoseScript [4] to describe
  selected keyframes identified as *motion beats* (local minima of
  segment-wise joint velocities)", where a signature pose is "a relatively
  stable, sculptural posture" and the dynamics are "the transitions between
  static poses".

So the paper already owns a construction for "where the movement is settled"
-- the motion beat -- and this file reuses the repository's implementation of
it, ``tools/motion_beats.find_motion_beats``, unmodified.  The one thing that
changes is scope: M3 calls it *inside a segment* (its ``prominence`` reference
is the segment mean); a segmentation ruler has no segments yet, so it is called
once per clip and ``max_beats`` is raised out of the way.  That difference is
stated rather than hidden, and ``--beat-reference local`` re-runs the whole
probe with a rolling reference instead of the clip mean so the ranking can be
checked against it.

THE CRITERION.  For one clip, with interior cut points C (the arm's boundaries
minus the two clip ends) and beats B:

    d(c)   = min over b in B of |c - b|, in frames
    D_arm  = mean over c in C of d(c)
    D_rand = the same quantity for ``--random-draws`` same-lengths-moved
             controls built from *this arm's own* spans, averaged
    R      = (D_rand + 0.5) / (D_arm + 0.5)

and the arm's score is the median of R over clips.  Higher is unambiguously
better: R = 1 means the cuts sit no closer to a settled pose than the same
segment lengths dropped anywhere else in the clip; R < 1 means the cuts
systematically *avoid* settled poses, which is what cutting mid-movement looks
like.  The +0.5 frame is there because a perfectly placed cut gives D_arm = 0.

Each arm gets its own random denominator on purpose.  Raw nearest-beat distance
is bounded by segment length, so arms that cut more finely would win on the raw
number for no reason other than being finer; dividing by a control that has
that arm's own lengths and only moves the cuts takes the fineness out.

Note the asymmetry: this measures *cut to nearest beat*, not *beat to nearest
cut*.  The paper puts the signature pose **inside** a segment, so a segment is
allowed -- expected -- to contain beats that are not cuts.  What it does not
allow is a cut in the middle of the leg extension.

CONTROLS, BOTH DIRECTIONS (gates 2 and 3).  Three synthetic arms are built from
the reference arm's cut counts and scored exactly like a real one:

* ``_beats``  -- cuts placed AT beats (deepest first), same interior-cut count,
  honouring ``--min-length``.  Known-good.  If this does not read high, the
  criterion is not measuring what its name says.
* ``_peaks``  -- cuts placed at the largest values of the same joint-speed
  signal, same count, same min-length.  This is the placement CLAUDE.md section
  2.1 names as the defect.  Known-bad.  If this does not read low, stop.
* ``_random`` -- one same-lengths-moved draw, scored against eight fresh ones.
  It is the null and must read ~1.000; anything else is a bug in the
  normalisation, not a finding.

``adjacent_over_random`` from ``probe_segmentation_boundaries`` is recomputed
here on the same clips and the same sample, because gate 4 of CLAUDE.md section
2.1 says two rulers that disagree are a finding, and comparing against a
published number measured on a different sample would not be a comparison.

Usage::

    python3 tools/probe_settle_alignment.py \\
        --bundle /cache/atomicdance-assets/data/wild3d/wild_v4_raw_bundle \\
        --clips runs/clean5/clips.txt --limit 400 \\
        --arm r0visual=runs/wild_v4_seg_r0visual/segmentation.json \\
        --output runs/settle/clean5.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.convert_motion_to_guofeats import motion_151_to_joints  # noqa: E402
from tools.motion_beats import find_motion_beats, joint_speed  # noqa: E402
from tools.probe_segmentation_boundaries import (  # noqa: E402
    canonical_key, load_arm, load_bundle_index, neighbour_ratio, paired_sign_test,
    shuffled_spans, spans_of,
)

FPS = 30.0
HIT_FRAMES = 3          # 0.1 s: the secondary, threshold-shaped readout
# SMPL's 24 joints split cleanly at 12: 0-11 are pelvis, hips, spine, knees,
# ankles and feet; 12-23 are neck, collars, head, shoulders, elbows, wrists and
# hands (the index names in ``tools/motion_beats`` agree).  The split exists so
# a positive control can be built from a joint set the ruler never looks at --
# see ``--speed-joints``.
JOINT_SETS = {"all": slice(0, 24), "lower": slice(0, 12), "upper": slice(12, 24)}


def load_frame_counts(bundle: pathlib.Path) -> dict:
    """{canonical key: frame_count} straight from the bundle manifest."""
    out = {}
    with open(pathlib.Path(bundle) / "sequences.jsonl", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            out[canonical_key(row.get("recording_id") or row["sequence_id"])] = int(
                row["frame_count"])
    return out


def smoothed_speed(joints: np.ndarray, smooth: int = 3) -> np.ndarray:
    """The exact trace ``find_motion_beats`` takes its minima of.

    Reproduced here so ``_peaks`` cuts at the maxima of the *same* signal whose
    minima define a beat.  A known-bad control built on a different signal
    would not be the defect CLAUDE.md section 2.1 names.
    """
    speed = joint_speed(joints)
    if smooth > 1 and len(speed) >= smooth:
        speed = np.convolve(speed, np.ones(smooth) / smooth, mode="same")
    return speed


def clip_beats(joints: np.ndarray, reference: str = "clip",
               window: int = 61) -> np.ndarray:
    """Beat frames for a whole clip.

    ``reference="clip"`` is ``find_motion_beats`` called unmodified with
    ``max_beats`` raised, so the prominence reference is the clip mean.
    ``reference="local"`` is the sensitivity check: the same minimum/separation
    rule, but a dip only has to sit below a centred rolling mean, so a clip with
    one fast passage and one slow one cannot have its whole slow half declared
    settled.
    """
    if reference == "clip":
        return np.asarray(find_motion_beats(joints, max_beats=10 ** 6), dtype=np.int64)
    speed = smoothed_speed(joints)
    if len(speed) < 3:
        return np.asarray([0] if len(speed) else [], dtype=np.int64)
    kernel = np.ones(min(window, len(speed)))
    rolling = np.convolve(speed, kernel, "same") / np.convolve(
        np.ones_like(speed), kernel, "same")
    is_minimum = (speed[1:-1] < speed[:-2]) & (speed[1:-1] <= speed[2:])
    deep_enough = speed[1:-1] <= rolling[1:-1] * (1.0 - 0.02)
    minima = np.arange(1, len(speed) - 1)[is_minimum & deep_enough]
    if len(minima) == 0:
        return np.asarray([int(speed.argmin())], dtype=np.int64)
    chosen: list = []
    for index in minima[np.argsort(speed[minima])]:
        if all(abs(int(index) - taken) >= 5 for taken in chosen):
            chosen.append(int(index))
    return np.asarray(sorted(chosen), dtype=np.int64)


def interior_cuts(spans) -> np.ndarray:
    """The cuts the algorithm actually chose -- the clip's two ends are not choices."""
    if len(spans) < 2:
        return np.asarray([], dtype=np.int64)
    return np.asarray([int(b) for a, b in spans[:-1]], dtype=np.int64)


def mean_nearest(cuts: np.ndarray, beats: np.ndarray):
    """(mean nearest-beat distance in frames, hit rate within HIT_FRAMES)."""
    if len(cuts) == 0 or len(beats) == 0:
        return float("nan"), float("nan")
    distance = np.abs(cuts[:, None] - beats[None, :]).min(axis=1)
    return float(distance.mean()), float((distance <= HIT_FRAMES).mean())


def placed_spans(positions: np.ndarray, order: np.ndarray, wanted: int,
                 low: int, high: int, min_length: int):
    """Greedily place ``wanted`` cuts at ``positions`` taken in ``order``.

    Shared by ``_beats`` and ``_peaks`` so the two controls differ only in which
    frames they are offered and in what order -- if they differed in the placing
    rule as well, a gap between them would not be attributable to the placement.
    """
    cuts: list = []
    for index in order:
        if len(cuts) >= wanted:
            break
        candidate = int(positions[index])
        if candidate - low < min_length or high - candidate < min_length:
            continue
        if all(abs(candidate - taken) >= min_length for taken in cuts):
            cuts.append(candidate)
    edges = [low] + sorted(cuts) + [high]
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def beat_spans(beats, speed, spans, min_length):
    low, high = int(spans[0][0]), int(spans[-1][1])
    beats = np.asarray([b for b in beats if low < b < high], dtype=np.int64)
    if len(beats) == 0:
        return []
    return placed_spans(beats, np.argsort(speed[beats]), len(spans) - 1,
                        low, high, min_length)


def peak_spans_speed(speed, spans, min_length):
    low, high = int(spans[0][0]), min(int(spans[-1][1]), len(speed))
    window = np.arange(low + 1, high)
    if len(window) == 0:
        return []
    return placed_spans(window, np.argsort(speed[window])[::-1], len(spans) - 1,
                        low, high, min_length)


def score_spans(spans, beats, block, rng, draws):
    """R, its two parts, and the adjacent/random neighbour ratio, for one clip."""
    cuts = interior_cuts(spans)
    arm_distance, arm_hits = mean_nearest(cuts, beats)
    if arm_distance != arm_distance:
        return None
    total = len(block)
    control_distance, control_hits = [], []
    for _ in range(draws):
        control = shuffled_spans(spans, total, rng)
        if len(control) < 2:
            continue
        distance, hits = mean_nearest(interior_cuts(control), beats)
        if distance == distance:
            control_distance.append(distance)
            control_hits.append(hits)
    if not control_distance:
        return None
    rand_distance = float(np.mean(control_distance))
    rand_hits = float(np.mean(control_hits))
    return {
        "settle_ratio": (rand_distance + 0.5) / (arm_distance + 0.5),
        "arm_distance": arm_distance,
        "rand_distance": rand_distance,
        "hit_ratio": (arm_hits + 1e-3) / (rand_hits + 1e-3),
        "arm_hits": arm_hits,
        "cuts": int(len(cuts)),
        "median_length": float(np.median([b - a for a, b in spans])),
        "neighbour": neighbour_ratio(block, spans, rng),
    }


def probe(bundle, clips, arms, limit, seed, min_length, draws, reference,
          speed_joints="all"):
    index = load_bundle_index(bundle)
    wanted = [canonical_key(line.strip()) for line in
              pathlib.Path(clips).read_text(encoding="utf-8").splitlines() if line.strip()]
    shared = [k for k in wanted if k in index]
    missing_bundle = len(wanted) - len(shared)
    # Every arm must be able to score every clip, or the arms are compared on
    # different corpora.  A stale arm whose last boundary no longer matches the
    # bundle's frame count is *excluded per clip*, and the count is reported --
    # never silently clipped, which would move its last cut for free.
    frame_count = load_frame_counts(bundle)
    stale = {name: 0 for name in arms}
    keep = []
    for key in shared:
        ok = True
        for name, arm in arms.items():
            bounds = arm.get(key)
            # The bundle's own ``frame_count`` is the referee, not the first
            # arm: with the first arm as referee a single stale arm would blame
            # every fresh one for disagreeing with it.
            if not bounds or len(bounds) < 3 or bounds[-1] != frame_count.get(key):
                stale[name] += 1
                ok = False
        if ok:
            keep.append(key)
    shared = keep
    if limit and limit < len(shared):
        stride = len(shared) / limit                    # even stride, never a prefix
        shared = [shared[int(i * stride)] for i in range(limit)]
    if not shared:
        raise SystemExit("no clip survives the arm/bundle intersection")

    names = list(arms) + ["_beats", "_beats_upper", "_peaks", "_random"]
    per_clip = {name: [] for name in names}
    reference_arm = next(iter(arms))
    rng = np.random.default_rng(seed)
    beat_counts, frame_counts, beat_cuts_short = [], [], 0

    for key in shared:
        motion = np.load(index[key]).astype(np.float32)
        joints = motion_151_to_joints(motion)
        ruler_joints = joints[:, JOINT_SETS[speed_joints], :]
        speed = smoothed_speed(ruler_joints)
        beats = clip_beats(ruler_joints, reference=reference)
        upper = joints[:, JOINT_SETS["upper"], :]
        upper_beats = clip_beats(upper, reference=reference)
        block = motion[:, 7:]
        beat_counts.append(len(beats))
        frame_counts.append(len(motion))
        for name, arm in arms.items():
            per_clip[name].append(score_spans(spans_of(arm[key]), beats, block, rng, draws))
        base = spans_of(arms[reference_arm][key])
        built = {
            "_beats": beat_spans(beats, speed, base, min_length),
            "_beats_upper": beat_spans(upper_beats, smoothed_speed(upper), base, min_length),
            "_peaks": peak_spans_speed(speed, base, min_length),
            "_random": shuffled_spans(base, len(motion), rng),
        }
        if len(built["_beats"]) < len(base):
            beat_cuts_short += 1
        for name, spans in built.items():
            per_clip[name].append(
                score_spans(spans, beats, block, rng, draws) if len(spans) >= 2 else None)

    out = {
        "clips_requested": len(wanted), "clips_scored": len(shared),
        "clips_not_in_bundle": missing_bundle, "clips_dropped_stale": stale,
        "beat_reference": reference, "random_draws": draws, "seed": seed,
        "min_length_frames": min_length, "reference_arm": reference_arm,
        "speed_joints": speed_joints,
        "beats_per_clip_median": float(np.median(beat_counts)),
        "beat_spacing_frames_median": float(np.median(
            np.asarray(frame_counts) / np.maximum(np.asarray(beat_counts), 1))),
        "clips_where_beat_arm_ran_short": beat_cuts_short,
        "arms": {},
    }
    for name in names:
        rows = [r for r in per_clip[name] if r]
        if not rows:
            continue
        pick = lambda field: np.asarray([r[field] for r in rows], dtype=np.float64)
        neigh = pick("neighbour")
        out["arms"][name] = {
            "clips": len(rows),
            "settle_ratio": float(np.median(pick("settle_ratio"))),
            "cut_to_beat_frames": float(np.median(pick("arm_distance"))),
            "random_cut_to_beat_frames": float(np.median(pick("rand_distance"))),
            "hit_rate_0p1s": float(np.median(pick("arm_hits"))),
            "hit_ratio": float(np.median(pick("hit_ratio"))),
            "interior_cuts_total": int(pick("cuts").sum()),
            "median_segment_seconds": float(np.median(pick("median_length")) / FPS),
            "adjacent_over_random": float(np.median(neigh[neigh == neigh]))
            if (neigh == neigh).any() else None,
        }
    out["paired_settle_vs_random"] = {}
    null = [r["settle_ratio"] if r else float("nan") for r in per_clip["_random"]]
    for name in names:
        if name == "_random":
            continue
        series = [r["settle_ratio"] if r else float("nan") for r in per_clip[name]]
        result = paired_sign_test(null, series, np.random.default_rng(seed))
        if result:
            out["paired_settle_vs_random"][name] = result
    out["_per_clip"] = {name: [r["settle_ratio"] if r else None for r in per_clip[name]]
                        for name in names}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=pathlib.Path, required=True)
    parser.add_argument("--clips", type=pathlib.Path, required=True)
    parser.add_argument("--arm", action="append", default=[], required=True)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--min-length", type=int, default=18)
    parser.add_argument("--random-draws", type=int, default=8)
    parser.add_argument("--beat-reference", choices=("clip", "local"), default="clip")
    parser.add_argument("--speed-joints", choices=tuple(JOINT_SETS), default="all",
                        help="which joints the ruler's own beats come from; "
                             "'lower' makes the _beats_upper control disjoint from it")
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    arms = {}
    for spec in args.arm:
        name, path = spec.split("=", 1)
        arms[name] = load_arm(pathlib.Path(path))
    report = probe(args.bundle, args.clips, arms, args.limit, args.seed,
                   args.min_length, args.random_draws, args.beat_reference,
                   args.speed_joints)
    # The run echoes the two switches that change what is being measured.  An
    # earlier revision accepted --speed-joints and dropped it on the floor at
    # this call; two runs came back byte-identical and only the echo showed it.
    assert report["speed_joints"] == args.speed_joints
    assert report["beat_reference"] == args.beat_reference

    print("clips scored {} of {} requested | beat reference {!r} | "
          "median {:.0f} beats/clip, one per {:.1f} frames".format(
              report["clips_scored"], report["clips_requested"], report["beat_reference"],
              report["beats_per_clip_median"], report["beat_spacing_frames_median"]))
    print("dropped: not in bundle {}, stale-per-arm {}".format(
        report["clips_not_in_bundle"], report["clips_dropped_stale"]))
    print("ruler beats from the {!r} joint set".format(report["speed_joints"]))
    print("%-14s %6s %8s %9s %9s %9s %9s %9s %8s" % (
        "arm", "clips", "settleR", "cut->beat", "rand->b", "hit@0.1s", "hitR",
        "adj/rand", "median s"))
    for name, row in report["arms"].items():
        print("%-14s %6d %8.3f %9.2f %9.2f %9.3f %9.3f %9s %8.2f" % (
            name, row["clips"], row["settle_ratio"], row["cut_to_beat_frames"],
            row["random_cut_to_beat_frames"], row["hit_rate_0p1s"], row["hit_ratio"],
            "%.3f" % row["adjacent_over_random"] if row["adjacent_over_random"] else "--",
            row["median_segment_seconds"]))
    print("\nper-clip paired sign test on settle_ratio, vs the _random null:")
    for name, row in report["paired_settle_vs_random"].items():
        print("  %-12s %d/%d clips better, median delta %+.3f, p = %.5f" % (
            name, row["wins"], row["clips"], row["median_delta"], row["p_two_sided"]))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("report -> {}".format(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
