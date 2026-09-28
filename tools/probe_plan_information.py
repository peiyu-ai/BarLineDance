#!/usr/bin/env python3
"""How much does the plan tell you about the motion, before any model sees it?

``score_generation_diagnostics.follow_rate`` asks how much of the retrieved
prototype reached the *generated* motion.  It reports a single-digit percentage
on the wild 340 arm, and the natural reading -- the completion model is
discarding its draft -- silently assumes the draft had something to say.  This
probe measures that assumption with the model removed entirely.

For held-out windows the release already holds both halves: ``labels.npy`` is a
ground-truth plan and ``motion.npy`` is what the dancer actually did on those
same frames.  So retrieve exactly as inference does -- the same
``IndexedAtomicMotionLibrary``, the same duration-nearest rule, the same
retrieval-group exclusion -- and ask how close the retrieved prototype sits to
the motion its own plan describes.

Three controls, because a distance without one says nothing:

* ``random_label`` -- identical segment boundaries, every label replaced by one
  drawn from the corpus label distribution.  This is the control that matters:
  it isolates *class identity* from everything else a draft carries anyway
  (segment count, durations, being real dance at all).  The published
  follow-rate control does not do this -- shuffling frames leaves the class in
  place -- so a vocabulary that carries nothing would still score above zero
  there and cannot score above zero here.
* ``shuffled`` -- the retrieved prototype with its frame order destroyed, the
  same control ``follow_rate`` uses, reported so the two tools' numbers can be
  read on one axis.
* ``oracle_in_class`` -- the closest candidate of the true class instead of the
  duration-nearest one.  The gap between it and the retrieved value is the
  headroom of the retrieval *rule*; the gap between it and ``random_label`` is
  the headroom of the *vocabulary*.  Those are different repairs, and a single
  number cannot tell them apart.

The ceiling this establishes is the one ``follow_rate`` must be read against.
If knowing the class buys X% here, no completion model can deliver more than X%
there, and quoting the model's number without X is quoting a fraction with no
denominator.

Everything is measured in **un-normalised** space.  The library hands out the
release's min-max scaled array, whose rot6d columns describe no rotation until
inverted -- the defect this repo hit in ``follow_rate`` itself on 2026-08-18,
where it capped a 0-to-1 metric at 0.456.

Usage::

    python3 tools/probe_plan_information.py \\
        --data-root /dev/shm/atomicdance-acct/release_w2_340 \\
        --split val --sequences 300 \\
        --output runs/t8_discovery/plan_information_val.json
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dataset.atomic import labels_to_segments  # noqa: E402
from infer_atomic import IndexedAtomicMotionLibrary, unnormalize_motion  # noqa: E402
from tools.score_generation_diagnostics import (  # noqa: E402
    _rotations_from_151, geodesic_rad,
)

MOTION_DIM = 151


def _first_slice_rows(names):
    """One window per recording, so overlapping slices cannot vote many times.

    Windows overlap ten deep in this release; sampling rows uniformly would draw
    the same five seconds of dance repeatedly and report its noise as a
    population spread.
    """
    chosen = {}
    for row, name in enumerate(names):
        recording = name.rsplit("_slice", 1)[0]
        if recording not in chosen:
            chosen[recording] = row
    return [chosen[key] for key in sorted(chosen)]


def _label_distribution(labels):
    """Empirical label frequency over the corpus, transitions excluded.

    The random-label control draws from this rather than uniformly over the
    4,159 classes: a uniform draw would over-sample the rare tail, whose
    prototypes are short and few, and the control would then be easy to beat for
    reasons that have nothing to do with class identity.
    """
    counts = collections.Counter()
    for row in labels:
        values, freq = np.unique(np.asarray(row), return_counts=True)
        for value, count in zip(values.tolist(), freq.tolist()):
            if value:
                counts[int(value)] += int(count)
    ids = np.array(sorted(counts), dtype=np.int64)
    weights = np.array([counts[int(i)] for i in ids], dtype=np.float64)
    return ids, weights / weights.sum()


def _relabel(labels, rng, ids, weights, available):
    """Same segmentation, different classes -- one draw per segment."""
    out = labels.clone()
    for segment in labels_to_segments(labels):
        if not segment.label:
            continue
        while True:
            pick = int(rng.choice(ids, p=weights))
            if pick in available:
                break
        out[segment.start:segment.end] = pick
    return out


def _candidates(library, label, exclude):
    return [c for c in library.index.get(int(label), ())
            if isinstance(c[3], str) and c[3] and c[3] not in exclude]


def _segment_values(library, candidate, length):
    sample, start, end, _ = candidate
    values = torch.from_numpy(np.array(library.motion[sample, start:end], copy=True))
    if len(values) != length:
        values = torch.nn.functional.interpolate(
            values.T.unsqueeze(0), size=length, mode="linear",
            align_corners=True).squeeze(0).T
    return values


def _rule_distance(library, normalizer, truth_rot, labels, exclude, cap, rng, rule):
    """Distance to the true motion under an alternative retrieval rule.

    The shipped rule is duration-nearest: ``min(candidates, key=|len - target|)``
    picks by length alone and never looks at the motion.  The in-class oracle
    says a rule that could see the answer would sit 20% closer, and that number
    is only interesting if some *implementable* rule collects part of it.  Two do
    not need the answer:

    * ``medoid`` -- the candidate closest to the other candidates of its own
      class, i.e. the class's most typical member.  Precomputed per class, so it
      is a property of the vocabulary rather than of the query.
    * ``continuity`` -- among the candidates, the one whose first frame is
      closest to the previous conditioned segment's last frame.  This is the
      only rule here that can also reduce the boundary jerk, which is 2.1-2.4x
      the rest of the clip.

    Both see at most ``cap`` candidates, the same budget the oracle gets, so the
    three are compared on equal information about the corpus and differ only in
    what they are allowed to use.
    """
    total, frames = 0.0, 0
    previous_last = None
    for segment in labels_to_segments(labels):
        if not segment.label:
            continue
        usable = _candidates(library, segment.label, exclude)
        if not usable:
            continue
        if len(usable) > cap:
            usable = [usable[i] for i in rng.choice(len(usable), cap, replace=False)]
        rotations = [_rotations_from_151(_segment_values(library, c, segment.length),
                                         normalizer) for c in usable]
        if rule == "medoid":
            if len(rotations) == 1:
                pick = 0
            else:
                sums = [sum(geodesic_rad(a, b) for j, b in enumerate(rotations) if j != i)
                        for i, a in enumerate(rotations)]
                pick = int(np.argmin(sums))
        elif rule == "continuity":
            if previous_last is None:
                pick = 0
            else:
                pick = int(np.argmin([geodesic_rad(previous_last[None], r[:1])
                                      for r in rotations]))
        else:
            raise ValueError("unknown rule {!r}".format(rule))
        previous_last = rotations[pick][-1]
        total += geodesic_rad(truth_rot[segment.start:segment.end],
                              rotations[pick]) * segment.length
        frames += segment.length
    return (total / frames) if frames else None


def _oracle_distance(library, normalizer, truth_rot, labels, exclude, cap, rng):
    """Best candidate of the true class, over at most ``cap`` of them.

    Capped rather than exhaustive because the largest classes hold thousands of
    segments and the answer converges long before that; the cap is reported so a
    reader can tell an oracle from a sample of one.
    """
    total, frames, capped = 0.0, 0, 0
    for segment in labels_to_segments(labels):
        if not segment.label:
            continue
        candidates = library.index.get(int(segment.label), ())
        usable = [c for c in candidates
                  if isinstance(c[3], str) and c[3] and c[3] not in exclude]
        if not usable:
            continue
        if len(usable) > cap:
            capped += 1
            picks = [usable[i] for i in rng.choice(len(usable), cap, replace=False)]
        else:
            picks = usable
        target = truth_rot[segment.start:segment.end]
        best = None
        for sample, start, end, _ in picks:
            values = torch.from_numpy(np.array(library.motion[sample, start:end], copy=True))
            if len(values) != segment.length:
                values = torch.nn.functional.interpolate(
                    values.T.unsqueeze(0), size=segment.length,
                    mode="linear", align_corners=True).squeeze(0).T
            rot = _rotations_from_151(values, normalizer)
            distance = geodesic_rad(target, rot)
            best = distance if best is None else min(best, distance)
        if best is None:
            continue
        total += best * segment.length
        frames += segment.length
    return (total / frames if frames else None), capped


def _predicted_plans(run):
    """Each clip's *predicted* plan and generated motion, keyed by recording.

    The release's ``labels.npy`` is a ground-truth plan, so a ceiling measured
    from it is the ceiling under a plan the planner did not have to produce.
    ``follow_rate`` scores against the predicted plan, so putting the two numbers
    on one axis needs this one.
    """
    import pickle

    out = {}
    for path in sorted(pathlib.Path(run).glob("*.pkl")):
        with open(path, "rb") as handle:
            clip = pickle.load(handle)
        out[path.stem] = (
            np.asarray(clip["atomic_labels"]).astype(np.int64),
            np.asarray(clip["smpl_poses"], dtype=np.float32).reshape(-1, 24, 3),
        )
    return out


def probe(data_root, split, sequences, seed, oracle_cap, recordings=None,
          predicted_plan_run=None, rule_names=()):
    root = pathlib.Path(data_root)
    normalizer = root / "normalizer.pt"
    if not normalizer.is_file():
        raise SystemExit("error: no normalizer at {}".format(normalizer))
    library = IndexedAtomicMotionLibrary(str(root))
    available = set(library.index)

    held = root / split
    motion = np.load(str(held / "motion.npy"), mmap_mode="r")
    labels_all = np.load(str(held / "labels.npy"), mmap_mode="r")
    names = json.load(open(str(held / "names.json")))
    groups = json.load(open(str(held / "retrieval_groups.json")))

    rows = _first_slice_rows(names)
    rng = np.random.default_rng(seed)
    if recordings:
        wanted = set(recordings)
        rows = [row for row in rows if names[row].rsplit("_slice", 1)[0] in wanted]
        found = {names[row].rsplit("_slice", 1)[0] for row in rows}
        absent = sorted(wanted - found)
        if absent:
            # Refused rather than scored on what happened to be present: the
            # point of naming recordings is to compare against a run built from
            # those same recordings, and a silently smaller set is a different
            # comparison wearing the same name.
            raise SystemExit(
                "error: {} of {} named recording(s) are not in the {} split, "
                "e.g. {}".format(len(absent), len(wanted), split, absent[:3]))
    elif sequences and sequences < len(rows):
        rows = [rows[i] for i in sorted(rng.choice(len(rows), sequences, replace=False))]

    ids, weights = _label_distribution(labels_all[:: max(1, len(labels_all) // 2000)])
    predicted = _predicted_plans(predicted_plan_run) if predicted_plan_run else {}

    per_row, unseen_labels, missing_segments = [], 0, 0
    for row in rows:
        labels = torch.from_numpy(np.array(labels_all[row], dtype=np.int64))
        exclude = (groups[row],) if isinstance(groups[row], str) and groups[row] else ()
        truth = torch.from_numpy(np.array(motion[row], dtype=np.float32))
        truth_rot = _rotations_from_151(truth, normalizer)

        for segment in labels_to_segments(labels):
            if segment.label and int(segment.label) not in available:
                unseen_labels += 1

        draft, mask = library.build_draft(
            labels, MOTION_DIM, exclude_retrieval_group_ids=exclude, allow_missing=True)
        keep = mask[:, 0] > 0
        if int(keep.sum()) < 2:
            missing_segments += 1
            continue
        proto = _rotations_from_151(draft, normalizer)

        control_labels = _relabel(labels, rng, ids, weights, available)
        control_draft, control_mask = library.build_draft(
            control_labels, MOTION_DIM,
            exclude_retrieval_group_ids=exclude, allow_missing=True)
        control_rot = _rotations_from_151(control_draft, normalizer)
        # Scored on the *true* plan's frames so the two drafts are compared on
        # one population; a control measured on its own mask would differ in
        # which frames it covers as well as in what it says there.
        control_keep = keep & (control_mask[:, 0] > 0)
        if int(control_keep.sum()) < 2:
            continue

        order = torch.from_numpy(rng.permutation(int(keep.sum())))
        honest = geodesic_rad(truth_rot[keep], proto[keep])
        shuffled = geodesic_rad(truth_rot[keep], proto[keep][order])
        random_label = geodesic_rad(truth_rot[control_keep], proto[control_keep])
        random_label_control = geodesic_rad(truth_rot[control_keep], control_rot[control_keep])
        oracle, capped = _oracle_distance(
            library, normalizer, truth_rot, labels, set(exclude), oracle_cap, rng)
        rules = {}
        for rule in rule_names:
            rules[rule] = _rule_distance(
                library, normalizer, truth_rot, labels, set(exclude),
                oracle_cap, rng, rule)

        predicted_row = None
        recording = names[row].rsplit("_slice", 1)[0]
        if predicted and recording in predicted:
            # Same frames, same window, same library -- only the plan changes,
            # and then only the side being compared against it.
            plan, generated = predicted[recording]
            width = min(len(plan), truth.shape[0], len(generated))
            plan_t = torch.from_numpy(np.array(plan[:width]))
            p_draft, p_mask = library.build_draft(
                plan_t, MOTION_DIM,
                exclude_retrieval_group_ids=exclude, allow_missing=True)
            p_keep = p_mask[:, 0] > 0
            if int(p_keep.sum()) >= 2:
                p_rot = _rotations_from_151(p_draft, normalizer)
                p_order = torch.from_numpy(rng.permutation(int(p_keep.sum())))
                truth_w = truth_rot[:width][p_keep]
                gen_w = torch.from_numpy(generated[:width])[p_keep]
                proto_w = p_rot[p_keep]
                truth_d = geodesic_rad(truth_w, proto_w)
                truth_c = geodesic_rad(truth_w, proto_w[p_order])
                gen_d = geodesic_rad(gen_w, proto_w)
                gen_c = geodesic_rad(gen_w, proto_w[p_order])
                predicted_row = {
                    "frames": int(p_keep.sum()),
                    "truth_follows_predicted_plan":
                        (truth_c - truth_d) / truth_c if truth_c > 1e-9 else None,
                    "generation_follows_predicted_plan":
                        (gen_c - gen_d) / gen_c if gen_c > 1e-9 else None,
                }

        per_row.append({
            "name": names[row],
            "predicted_plan": predicted_row,
            "conditioned_frames": int(keep.sum()),
            "transition_share": float((labels == 0).float().mean()),
            "distance_rad": honest,
            "shuffled_control_rad": shuffled,
            # honest and random_label are both measured on control_keep so the
            # ratio below is a within-frame comparison, not two populations.
            "paired_distance_rad": random_label,
            "random_label_control_rad": random_label_control,
            "oracle_in_class_rad": oracle,
            "oracle_candidates_capped_segments": capped,
            "rules_rad": rules,
        })

    def median(key):
        values = [r[key] for r in per_row if r.get(key) is not None]
        return float(np.median(values)) if values else None

    def median_predicted(key):
        values = [r["predicted_plan"][key] for r in per_row
                  if r.get("predicted_plan") and r["predicted_plan"].get(key) is not None]
        return float(np.median(values)) if values else None

    def median_rule(rule):
        values = [r["rules_rad"].get(rule) for r in per_row
                  if r.get("rules_rad") and r["rules_rad"].get(rule) is not None]
        return float(np.median(values)) if values else None

    honest = median("distance_rad")
    paired = median("paired_distance_rad")
    random_label = median("random_label_control_rad")
    shuffled = median("shuffled_control_rad")
    oracle = median("oracle_in_class_rad")

    def gain(control, value):
        if control is None or value is None or control <= 1e-9:
            return None
        return (control - value) / control

    return {
        "data_root": str(root),
        "split": split,
        "sequences_scored": len(per_row),
        "sequences_requested": len(rows),
        "sequences_without_any_prototype": missing_segments,
        "recording_list_applied": bool(recordings),
        "segments_naming_a_label_absent_from_train": unseen_labels,
        "oracle_candidate_cap": oracle_cap,
        "seed": seed,
        "median_rad": {
            "retrieved": honest,
            "retrieved_on_paired_frames": paired,
            "random_label_control": random_label,
            "shuffled_frames_control": shuffled,
            "oracle_in_class": oracle,
        },
        # Each implementable rule beside the two bounds it sits between: the
        # shipped duration-nearest rule below it, the in-class oracle above.
        "retrieval_rules": {
            rule: {
                "median_rad": median_rule(rule),
                "gain_over_shipped_rule": gain(honest, median_rule(rule)),
                "share_of_oracle_headroom": (
                    None if (median_rule(rule) is None or honest is None
                             or oracle is None or abs(honest - oracle) < 1e-9)
                    else (honest - median_rule(rule)) / (honest - oracle)
                ),
            }
            for rule in rule_names
        },
        "information": {
            # The headline: what knowing the class buys over not knowing it.
            "over_random_label": gain(random_label, paired),
            "over_shuffled_frames": gain(shuffled, honest),
            # Headroom, split by which repair would collect it.
            "retrieval_rule_headroom": gain(honest, oracle),
            "vocabulary_ceiling_over_random_label": gain(random_label, oracle),
        },
        # The comparison follow_rate needs to be readable: both sides scored on
        # the same frames, the same window and the *predicted* plan, so the only
        # difference is whether the true motion or the generated motion is being
        # asked how closely it tracks the draft.  The first is the ceiling for
        # the second.
        "predicted_plan": {
            "run": str(predicted_plan_run) if predicted_plan_run else None,
            "sequences": sum(1 for r in per_row if r.get("predicted_plan")),
            "truth_follows": median_predicted("truth_follows_predicted_plan"),
            "generation_follows": median_predicted("generation_follows_predicted_plan"),
        },
        "per_sequence": per_row,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=pathlib.Path, required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--sequences", type=int, default=200,
                        help="one window per recording; 0 means every recording")
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--oracle-cap", type=int, default=48,
                        help="candidates per segment the in-class oracle may see")
    parser.add_argument("--recording-list", type=pathlib.Path, default=None,
                        help="score exactly these recordings, one per line, so the "
                             "ceiling can be read against a generation run built "
                             "from the same clips; refuses if any is absent")
    parser.add_argument("--predicted-plan-run", type=pathlib.Path, default=None,
                        help="a generation run; additionally score the *predicted* "
                             "plan's draft against both the true and the generated "
                             "motion, which is the pair follow_rate must be read as")
    parser.add_argument("--retrieval-rules", default="",
                        help="comma-separated alternatives to score beside the "
                             "shipped duration-nearest rule: medoid, continuity")
    parser.add_argument("--output", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)

    recordings = None
    if args.recording_list:
        recordings = [line.strip() for line in
                      args.recording_list.read_text(encoding="utf-8").splitlines()
                      if line.strip()]
    rule_names = tuple(r for r in args.retrieval_rules.split(",") if r.strip())
    report = probe(args.data_root, args.split, args.sequences, args.seed,
                   args.oracle_cap, recordings, args.predicted_plan_run, rule_names)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    slim = {k: v for k, v in report.items() if k != "per_sequence"}
    print(json.dumps(slim, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
