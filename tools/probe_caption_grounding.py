#!/usr/bin/env python3
"""Are the caption's fields true of the motion underneath them?

Why this exists
---------------
``probe_caption_quality`` answers whether a captioner is *usable* (on-schema),
*consistent* (two segments of one prototype get similar words) and *varied*
(not one sentence for everything).  All three can be satisfied by a captioner
that is confidently wrong: answer "pose, arms raised, middle level, in place,
smooth" for everything vaguely still, and consistency is high, discrimination
looks acceptable, and no gate fires.

What none of those ask is whether the words are *true*.  On clean5b5 that
question is not academic:

    travel      = in_place  for 93.7% of segments
    body_action = pose      for 57.2%
    level       = changing  for 0.14% (17 of 11,971)

Each of those is either the captioner defaulting, or the corpus being what it
is -- 1.9-second segments of a dancer in a fixed frame, who genuinely does not
cross the room and genuinely does hold shapes.  The two readings have opposite
consequences for a model substitution, and no amount of agreement between two
captioners separates them.  The 3D motion does.

The ruler, and why it is fair to any captioner
----------------------------------------------
Each test names a field value, a geometric quantity that value is *about*, and
**the direction the caption must move it, declared before the number is read**.
The statistic is the AUC of that quantity separating the segments carrying the
value from the segments that do not:

* 0.5 is chance, exactly, whatever the class balance -- which matters when one
  value holds 93.7% of the corpus and a raw accuracy would read 0.937 for a
  captioner that always says it.
* 1.0 is the analytic ceiling: it is what a label read straight off the
  geometry would score, so it is not a target a language model should reach.
* **Below 0.5 on a test that declares "higher" is a failure, not a hit.**
  Reporting ``|AUC - 0.5|`` would pay a captioner for being reliably backwards,
  which is the shape of the 2026-08-19 ``boundary contrast`` defect: a
  criterion that rewarded the very thing it was meant to remove.

The null shuffles captions across segments and re-scores.  It has to land on
0.5; if it does not, the pairing between caption rows and motion is wrong and
no other column may be read.

Contamination, per axis
-----------------------
``--posescript`` puts a rule-based pose sentence in the prompt, so the VLM is
*told* part of the answer for some axes.  ``motion_beats.describe_pose`` emits
arm heights/reaches, foot stance, one-foot-lifted and knees-bent-low.  So:

    travel, body_action, dynamics   clean -- the cue says nothing about them
    level                           partly -- "knees bent low" implies low
    arms, legs                      contaminated -- the cue states them

Only the clean axes are evidence about what the VLM *saw*.  The contaminated
ones are reported anyway, marked, because a captioner that cannot even copy a
cue it was handed is a different and worse failure.

What this does not answer
-------------------------
Whether the caption is the *best* description of the movement, or whether the
schema can express the movement at all.  A segment whose truth is "half a
turn into a freeze" scores well here as long as the fields it does have are
right.  Expressiveness is a property of the vocabulary, not of the captioner,
and it is measured separately.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
from typing import Callable, Dict, List, Sequence

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MIN_GROUP = 20          # a value needs enough segments for an AUC to mean anything
ROOT, L_HIP, R_HIP = 0, 1, 2
L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 4, 5, 7, 8


def auc(values: np.ndarray, member: np.ndarray) -> float:
    """P(a random member ranks above a random non-member), ties counted as half.

    Rank-based rather than a mean difference: these quantities are heavy-tailed
    (one jump dominates a mean displacement) and the question is ordering, not
    magnitude.
    """
    n_in, n_out = int(member.sum()), int((~member).sum())
    if n_in < MIN_GROUP or n_out < MIN_GROUP:
        return float("nan")
    from scipy.stats import rankdata
    # Mid-ranks for ties, so a constant column reads exactly 0.5 rather than
    # whatever order the sort happened to leave it in.
    ranks = rankdata(values, method="average")
    total = ranks[member].sum()
    return float((total - n_in * (n_in + 1) / 2.0) / (n_in * n_out))


def measures(joints: np.ndarray) -> Dict[str, float]:
    """Geometric quantities for one segment, in units of the dancer's own body.

    Every quantity is divided by that dancer's leg length rather than left in
    world units: GVHMR's scale is not comparable between clips, and a metric
    threshold would rank tall dancers as travelling further.
    """
    hips = 0.5 * (joints[:, L_HIP] + joints[:, R_HIP])
    # Sum of the thigh and shank, not the straight hip-to-ankle distance, which
    # is exactly what ``motion_beats.describe_pose`` uses and for the reason
    # this probe learned the hard way: the straight line SHORTENS when the knee
    # bends, so dividing by it inflates the normalised height of a crouch.  The
    # first run of this tool reported ``legs=crouched`` at 0.329 -- reliably
    # *backwards* -- because its normaliser was a function of the thing being
    # measured.
    thigh = np.linalg.norm(joints[:, L_HIP] - joints[:, L_KNEE], axis=-1)
    shank = np.linalg.norm(joints[:, L_KNEE] - joints[:, L_ANKLE], axis=-1)
    leg = float((thigh + shank).mean())
    leg = leg if leg > 1e-6 else 1.0
    root = joints[:, ROOT]

    horizontal = np.linalg.norm(root[-1, :2] - root[0, :2]) / leg
    path = np.linalg.norm(np.diff(root[:, :2], axis=0), axis=-1).sum() / leg

    # Facing from the hip axis; unwrapped so a turn past pi is not read as a
    # turn back the other way.
    axis = joints[:, L_HIP] - joints[:, R_HIP]
    yaw = np.unwrap(np.arctan2(axis[:, 1], axis[:, 0]))
    rotation = float(abs(yaw[-1] - yaw[0]))

    height = hips[:, 2] / leg
    speed = np.linalg.norm(np.diff(joints, axis=0), axis=-1).mean(axis=-1) / leg
    ankle = np.linalg.norm(np.diff(joints[:, [L_ANKLE, R_ANKLE]], axis=0),
                           axis=-1).max(axis=-1) / leg
    acceleration = np.abs(np.diff(speed)) if len(speed) > 1 else np.zeros(1)
    # Jerk, the third difference of joint position.  "Smooth" in a dance
    # vocabulary is about continuity of flow, not about holding one speed -- a
    # sustained arc accelerates steadily and is smooth -- so ``speed_variation``
    # is this probe's own invention and needs the alternative measured beside
    # it before either is read (CLAUDE.md 2.1 point 1).
    if len(joints) > 3:
        jerk = np.linalg.norm(np.diff(joints, n=3, axis=0), axis=-1).mean() / leg
    else:
        jerk = 0.0

    # --- schema v2 measures ---------------------------------------------
    # A rhythm axis is only worth adding if it can be checked, so each of its
    # values gets a quantity here before any caption is written with it.
    if len(speed) > 2:
        steps = np.arange(len(speed), dtype=np.float64)
        slope = float(np.polyfit(steps, speed, 1)[0])
        trend = slope * len(speed) / speed.mean() if speed.mean() else 0.0
    else:
        trend = 0.0
    # Vertical reversals of the pelvis per second of segment, at 30 fps: a
    # bounce is a repeated up-down, which no single-frame quantity can see.
    dz = np.diff(hips[:, 2])
    reversals = int(np.count_nonzero(np.diff(np.sign(dz)) != 0)) if len(dz) > 1 else 0
    oscillation = reversals / (len(joints) / 30.0) if len(joints) else 0.0
    # Share of the segment spent nearly still: separates "stop and go" from a
    # movement that is merely slow throughout.
    floor = 0.25 * float(np.median(speed)) if len(speed) else 0.0
    stopped = float(np.mean(speed < floor)) if len(speed) else 0.0

    return {
        "displacement": float(horizontal),
        "speed_trend": float(trend),
        "oscillation": float(oscillation),
        "stopped_fraction": stopped,
        "path_length": float(path),
        "rotation": rotation,
        "vertical_range": float((root[:, 2].max() - root[:, 2].min()) / leg),
        "height": float(height.mean()),
        "height_range": float(height.max() - height.min()),
        "speed": float(speed.mean()),
        "ankle_speed": float(ankle.max()) if len(ankle) else 0.0,
        "peak_acceleration": float(acceleration.max()),
        "speed_variation": float(speed.std() / speed.mean()) if speed.mean() else 0.0,
        "jerk": float(jerk),
    }


# field, value(s), measure, direction the caption must move it, cue contamination
TESTS: Sequence[tuple] = (
    ("travel", ("in_place",), "path_length", "lower", "clean"),
    ("travel", ("forward", "sideways", "backward"), "displacement", "higher", "clean"),
    ("travel", ("rotating",), "rotation", "higher", "clean"),
    ("body_action", ("pose",), "speed", "lower", "clean"),
    ("body_action", ("jump",), "vertical_range", "higher", "clean"),
    ("body_action", ("kick",), "ankle_speed", "higher", "clean"),
    ("body_action", ("turn", "spin"), "rotation", "higher", "clean"),
    ("dynamics", ("explosive",), "peak_acceleration", "higher", "clean"),
    ("dynamics", ("smooth",), "speed_variation", "lower", "clean"),
    ("dynamics", ("smooth",), "jerk", "lower", "clean"),
    ("dynamics", ("sharp",), "jerk", "higher", "clean"),
    ("dynamics", ("sustained",), "jerk", "lower", "clean"),
    ("dynamics", ("slow",), "speed", "lower", "clean"),
    ("level", ("low",), "height", "lower", "partly"),
    ("level", ("high",), "height", "higher", "partly"),
    ("level", ("changing",), "height_range", "higher", "partly"),
    ("legs", ("lifted",), "height_range", "higher", "cue-fed"),
    ("legs", ("crouched",), "height", "lower", "cue-fed"),
    # --- schema v2 axes.  Skipped automatically on a v1 caption file, because
    # the field is absent and the group falls under MIN_GROUP.
    ("intensity", ("explosive",), "peak_acceleration", "higher", "clean"),
    ("intensity", ("strong",), "peak_acceleration", "higher", "clean"),
    ("intensity", ("gentle",), "peak_acceleration", "lower", "clean"),
    ("fluidity", ("flowing",), "jerk", "lower", "clean"),
    ("fluidity", ("sustained",), "jerk", "lower", "clean"),
    ("fluidity", ("sharp", "staccato"), "jerk", "higher", "clean"),
    ("fluidity", ("staccato",), "stopped_fraction", "higher", "clean"),
    ("rhythm", ("held",), "speed", "lower", "clean"),
    ("rhythm", ("steady",), "speed_variation", "lower", "clean"),
    ("rhythm", ("pulsing",), "oscillation", "higher", "clean"),
    ("rhythm", ("accelerating",), "speed_trend", "higher", "clean"),
    ("rhythm", ("decelerating",), "speed_trend", "lower", "clean"),
    ("rhythm", ("stop_and_go",), "stopped_fraction", "higher", "clean"),
)


def run_tests(fields: List[Dict[str, str]], table: Dict[str, np.ndarray],
              rng, rounds: int = 200) -> List[dict]:
    rows = []
    for field, values, measure, direction, contamination in TESTS:
        member = np.array([row.get(field) in values for row in fields])
        score = auc(table[measure], member)
        if np.isnan(score):
            continue
        # Declared before reading: a "lower" test is passed by a LOW auc, so it
        # is reported as 1 - auc.  Both are then on one scale where 0.5 is
        # chance and below 0.5 means the caption is reliably backwards.
        oriented = score if direction == "higher" else 1.0 - score
        # Averaged over many permutations, not drawn once.  A single shuffle of
        # a 24-segment group has a standard error near 0.06 on this corpus, so
        # a one-draw null reads anywhere between 0.39 and 0.61 while nothing is
        # wrong -- and an instrument check built on it fires on its own noise,
        # which is what happened the first time this was run.
        draws = np.array([auc(table[measure], member[rng.permutation(len(member))])
                          for _ in range(rounds)])
        # Orient the null the same way as the reading before differencing them.
        # Taking ``score - null`` with one of the two flipped reports a
        # backwards caption as a strong hit and a good one as a miss.
        null = float(np.nanmean(draws))
        oriented_null = null if direction == "higher" else 1.0 - null
        spread = float(np.nanstd(draws))
        rows.append({
            "field": field, "values": list(values), "measure": measure,
            "direction": direction, "cue": contamination,
            "n": int(member.sum()), "share": float(member.mean()),
            "auc": oriented,
            "null": oriented_null,
            "null_sd": spread,
            "z": float((oriented - oriented_null) / spread) if spread > 0
                 else float("nan"),
        })
    return rows


def calibration(table: Dict[str, np.ndarray], measure: str, share: float,
                rng, agreements=(1.0, 0.9, 0.8, 0.7, 0.6)) -> Dict[str, float]:
    """What AUC a label that is right p% of the time actually scores.

    Without this, "0.62" is a number with no scale attached.  The oracle label
    is the measure's own top ``share`` of segments, so 1.0 is the ceiling by
    construction; each lower row corrupts that label at random.
    """
    values = table[measure]
    cut = np.quantile(values, 1.0 - share)
    oracle = values >= cut
    out = {}
    for level in agreements:
        noisy = oracle.copy()
        flip = rng.random(len(noisy)) > level
        noisy[flip] = ~noisy[flip]
        out["{:.0%}".format(level)] = auc(values, noisy)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--captions", type=pathlib.Path, required=True)
    parser.add_argument("--bundle", type=pathlib.Path, required=True,
                        help="stage-E performance bundle holding the 151-dim motion")
    parser.add_argument("--output", type=pathlib.Path, default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="score only the first N recordings; for a smoke run")
    parser.add_argument("--rounds", type=int, default=200,
                        help="permutations behind each null; a single draw on a "
                             "small group is noisier than the effect measured")
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()

    from tools.cluster_atomics_tmr import build_row_index, resolve_row
    from tools.convert_motion_to_guofeats import motion_151_to_joints

    rows = [json.loads(line) for line in
            args.captions.open(encoding="utf-8")]
    by_recording: Dict[str, List[dict]] = collections.defaultdict(list)
    for row in rows:
        by_recording[str(row["recording_id"])].append(row)

    bundle_rows = {row["recording_id"]: row for row in
                   (json.loads(line) for line
                    in (args.bundle / "sequences.jsonl").open(encoding="utf-8"))}
    index = build_row_index(bundle_rows)

    names = sorted(by_recording)
    if args.limit:
        names = names[: args.limit]

    fields: List[Dict[str, str]] = []
    collected: List[Dict[str, float]] = []
    skipped_recordings = skipped_segments = 0
    for name in names:
        row = resolve_row(index, name) or bundle_rows.get(name)
        if row is None:
            skipped_recordings += 1
            continue
        joints = motion_151_to_joints(np.load(args.bundle / row["motion_path"]))
        for entry in by_recording[name]:
            start, end = int(entry["start"]), min(int(entry["end"]), len(joints))
            if end - start < 4:
                skipped_segments += 1
                continue
            collected.append(measures(joints[start:end]))
            fields.append(entry["fields"])

    print("recordings scored        : {} ({} unresolved)".format(
        len(names) - skipped_recordings, skipped_recordings))
    print("segments scored          : {} ({} too short)".format(
        len(collected), skipped_segments))
    if not collected:
        raise SystemExit("no segment could be measured")

    table = {key: np.array([row[key] for row in collected])
             for key in collected[0]}
    rng = np.random.default_rng(args.seed)
    results = run_tests(fields, table, rng, rounds=args.rounds)

    print()
    print("{:<12} {:<24} {:<18} {:>6} {:>6} {:>7} {:>7} {:>7} {:>8}".format(
        "field", "value(s)", "measure", "n", "share", "auc", "null", "z", "cue"))
    for row in results:
        print("{:<12} {:<24} {:<18} {:>6} {:>5.1f}% {:>7.3f} {:>7.3f} {:>7.1f} {:>8}"
              .format(row["field"], ",".join(row["values"])[:24], row["measure"],
                      row["n"], 100 * row["share"], row["auc"], row["null"],
                      row["z"], row["cue"]))

    nulls = np.array([row["null"] for row in results])
    print()
    if np.abs(nulls - 0.5).max() > 0.02:
        print("INSTRUMENT_FAIL a shuffled caption reads {:.3f} at worst, not 0.5 -- "
              "the caption rows and the motion are not paired".format(
                  nulls[np.abs(nulls - 0.5).argmax()]), flush=True)
        return 1
    print("instrument ok: shuffled captions read {:.3f}-{:.3f} (chance is 0.500)"
          .format(nulls.min(), nulls.max()))

    scale = calibration(table, "path_length", 0.2, rng)
    print("scale, on this corpus: a label agreeing with the geometry "
          + ", ".join("{} of the time reads {:.3f}".format(k, v)
                      for k, v in scale.items()))

    report = {"segments": len(collected), "tests": results, "calibration": scale}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
        print("wrote {}".format(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
