#!/usr/bin/env python3
"""Does the evaluation set cross the training set, and did a re-split fix it.

The corpus is split by upload/account, and that key does not separate
choreographies: the same routine is posted by several accounts, so a test clip's
dance can sit in train under a different upload id.  Measured 2026-08-25 on the
split the released models were trained on: **38 of 184 test clips (20.7%) have a
fingerprint-confirmed same-track recording in train, all of them under a
different upload**, so neither the split nor the upload-level retrieval
exclusion removes them.

What that costs is not what it looks like.  Three separate measurements that
day, all on the 65-clip M6 set:

* **Retrieval is not the channel.**  Re-running all 21 leaking clips with the
  twin's upload excluded from the prototype library moved the generated motion
  by a median of 0.0018 m and moved its distance to the clip's own
  reconstruction by +0.0000; the plan came out bit-identical in 21 of 21.  The
  draft is not what carries the answer.
* **The planner's weights are.**  The plan it emits agrees with the *training*
  twin's label track at a median 0.257 and with the clip's own labels at 0.088
  (clips with no twin: 0.149 against their own).  Four clips reached 0.96-1.00
  agreement with the twin -- the planner replays the training instance when the
  same song conditions it.
* **It shows in the motion.**  A clip's generation ranks its own reconstruction
  at percentile 0.037 among the other clips' references when a twin is in train,
  against 0.398 when there is none (n=15 vs 40, permutation p=0.0010).

So this file measures leakage twice, because the two answers are not the same
question:

**Declared** -- which fingerprint groups straddle the split.  This is the
criterion a re-split would be built on, and a re-split that consumes it is
circular: it cannot discover a pair the fingerprint missed.

**Measured** -- nearest neighbour in *pose* space, which never touches the
audio.  This is the check on the criterion (CLAUDE.md 2.1: a criterion does not
get to render judgment until something else has verified it).  The scale is
computed from the data on every run rather than hardcoded, because the one
hardcoded scale this repo tried was wrong: a "same clip reconstructed twice"
ceiling of 0.1191 quoted on 2026-08-24 turned out to be seven pairs of a clip
against *its own half-speed mis-cut*, and the same pairs read 0.0182 once the
speed was undone.  A yardstick built out of a defect measures the defect.

The scales this prints, and what they were on the wild corpus:

    two reconstructions of one clip   ~0.02   (the floor: instrument noise)
    same choreography, two dancers    0.05-0.07
    two unrelated recordings          0.2058  (the null, measured 2026-08-25)
    fingerprint-declared pairs        0.1459  (bimodal: same track is not same dance)

Note the fourth line against the third: **a declared pair is not reliably closer
than an unrelated one**, because roughly half of "same track" is only the same
song.  So an absolute distance cannot be the criterion, and the reading is a
per-query ratio with its threshold set from a negative control -- see
``measured_leakage``, whose docstring records what the absolute version did
wrong and what it cost.

**Acceptance test after a re-split.**  Leakage is gone when, on the retrained
model, the rank statistic for clips that used to have a train twin becomes
indistinguishable from clips that never had one (0.4-ish, not 0.04), and the
plan agreement with any train recording falls to the random level (0.015 on this
corpus).  ``--generated`` runs exactly that, and it is the only reading here
that can tell "the split is clean" from "the split is clean and the model
already memorised the answer before we cleaned it".

**"used to have" is the whole point, and deriving it from the current split
destroys it.**  Until 2026-08-26 ``model_leakage`` built the twin group from
``records[t]["split"] == "train"`` -- the twins that are in train *now*.  On the
split this test exists to accept, that group is empty by construction: the
song-disjoint split puts every fingerprint-linked recording on one side, so
``with_train_twin`` reads 0 clips and the criterion cannot fire in either
direction.  A criterion that goes silent exactly when the fix works is worse
than no criterion (see CLAUDE.md 2.1), and it would have gone silent only after
the retrain it is meant to judge had already been paid for.

So the cohort is now an **input**, not a derivation.  ``--emit-twin-cohort``
writes, from the generation being disproved, the recordings that were each
clip's *training* twin at that time; ``--twin-cohort`` reads that file back and
measures against those same recordings **whatever split they now sit in**.  That
is the reading that answers the question: those recordings are no longer in
train, so if the plan still replays their label track, the re-split did not
close the channel.  Recording ids carry a generation prefix
(``wild_v4:<upload>:clip000`` vs ``wild_v5:...``), so the file is matched on the
id with that prefix stripped, and a cohort that resolves onto no clip is a hard
error rather than an empty group that reads like a pass.

Nothing is excluded silently: every run prints how many recordings were skipped
and why, and whether its own controls separated, because "we found none", "we
could not look" and "the criterion cannot tell" print the same otherwise.

Usage::

    # before: how bad is the split in use
    audit_split_leakage.py --bundle /dev/shm/.../wild_v4_bundle \\
        --pairs runs/wild_v4_music_groups_pairs.jsonl --output runs/split_leakage.json

    # after: did the re-split and retrain actually fix it
    audit_split_leakage.py --bundle .../wild_v4_bundle_v2 \\
        --pairs runs/wild_v4_music_groups_pairs.jsonl \\
        --windows /dev/shm/.../clean5b5_windows_v2 --generated runs/m6_v2/motion \\
        --policy refuse --output runs/split_leakage_v2.json
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import pickle
import random
import statistics
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.convert_motion_to_guofeats import motion_151_to_joints  # noqa: E402

# Compared frame-for-frame, so a pair with less overlap than this is not a
# reading.  Two seconds is below anything the corpus holds (its shortest clip is
# 340 frames); it exists to stop a truncated array from being scored at all.
MIN_OVERLAP_FRAMES = 60

# A ratio needs a neighbourhood to be a ratio: fewer comparisons than this and
# the 5th percentile it is divided by is one or two draws, so the query is
# reported as skipped rather than scored on a denominator that is noise.
MIN_POOL_FOR_RATIO = 20

# How many nearest candidates get the expensive lag scan.  Lag 0 is right for
# two clips cut from the same instant of the same track and wrong for everything
# else, so the sweep ranks at lag 0 and then re-scores the head.  Reported, so a
# reader can see how far down the tool looked.
LAG_RESCORE_TOP_K = 5
LAG_SCAN_FRAMES = 300
LAG_SCAN_STEP = 5


class AuditError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# motion


def rootless(joints):
    """Joint positions with the pelvis subtracted, so translation cannot score.

    Root translation is where a wild reconstruction is least reliable (DPVO
    drifts, and the corpus's cameras move), and two clips of one routine filmed
    from different places differ in it by metres while the pose matches.
    """
    joints = np.asarray(joints, dtype=np.float64)
    return joints - joints[:, 0:1, :]


def pose_distance(left, right, lag=0):
    """Median per-joint distance over the overlap, at a stated frame offset.

    **Convention, pinned by a test in both directions:** frame ``t`` of ``left``
    is compared with frame ``t - lag`` of ``right``, so a *positive* lag means
    ``left`` runs ahead of ``right``.  This is spelled out because a sign that
    flips silently still produces a plausible number -- the one measured defect
    of this kind in this repo (2026-08-19) cost a day of a wrong explanation
    before it turned out to be an index off by one.

    Returns ``(distance, frames)``; ``(None, 0)`` when the overlap is too short
    to be a reading rather than silently returning a number built from a handful
    of frames.
    """
    left = rootless(left)
    right = rootless(right)
    start = max(0, lag)
    stop = min(len(left), len(right) + lag)
    if stop - start < MIN_OVERLAP_FRAMES:
        return None, 0
    a = left[start:stop]
    b = right[start - lag:stop - lag]
    return float(np.median(np.linalg.norm(a - b, axis=-1))), int(stop - start)


def best_over_lag(left, right, span=LAG_SCAN_FRAMES, step=LAG_SCAN_STEP):
    """The closest reading over a lag window, and the lag that produced it."""
    best = (None, 0, 0)
    for lag in range(-span, span + 1, step):
        value, frames = pose_distance(left, right, lag)
        if value is not None and (best[0] is None or value < best[0]):
            best = (value, frames, lag)
    return best


# --------------------------------------------------------------------------- #
# inputs


def read_bundle(path):
    """``recording_id -> record`` for a performance bundle's sequences.jsonl."""
    root = pathlib.Path(path)
    manifest = root / "sequences.jsonl" if root.is_dir() else root
    if not manifest.is_file():
        raise AuditError("no sequences.jsonl at {}".format(path))
    records = {}
    for line in manifest.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        records[record["recording_id"]] = record
    if not records:
        raise AuditError("{} names no sequence".format(manifest))
    return manifest.parent, records


def read_pairs(path):
    """Fingerprint-confirmed same-track pairs as an adjacency map."""
    adjacency = collections.defaultdict(set)
    count = 0
    lags = {}
    for line in pathlib.Path(path).open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        left, right = row["left"], row["right"]
        adjacency[left].add(right)
        adjacency[right].add(left)
        lags[(left, right)] = int(row.get("lag_frames", 0))
        lags[(right, left)] = -int(row.get("lag_frames", 0))
        count += 1
    return adjacency, lags, count


def merge_pair_files(paths, universe):
    """Every measured same-track list, merged onto one bundle's recording ids.

    Two fingerprint runs over two cuts of the same videos do not miss the same
    pairs, and each run's recall is well under half, so the newest list is not
    the evidence -- it is *some* of it.  Measured 2026-08-26: v4's list held
    4,352 edges v5's did not, and 636 of them crossed the split built from v5's
    list alone (281 test<->train), touching 176 of 1,384 test clips.

    This has to consume at least as much evidence as the split did, or the
    audit is weaker than the thing it audits -- which is backwards, and would
    read as a clean bill of health.

    Ids are matched with the generation prefix stripped, because the list being
    carried in is by construction from an older corpus
    (``wild_v4:<upload>:clip000`` against ``wild_v5:...``).  Each file is
    reported on its own row: a list that lands nothing and a list that adds
    nothing are the same number after merging, and only one of them is a
    mistake.
    """
    by_stripped = {}
    for name in universe:
        by_stripped.setdefault(strip_generation(name), name)
    adjacency = collections.defaultdict(set)
    lags, per_file = {}, []
    for path in paths:
        here, here_lags, rows = read_pairs(path)
        landed, lost, added = set(), 0, 0
        for left, neighbours in here.items():
            mapped_left = by_stripped.get(strip_generation(left))
            for right in neighbours:
                mapped_right = by_stripped.get(strip_generation(right))
                if mapped_left is None or mapped_right is None or mapped_left == mapped_right:
                    lost += 1
                    continue
                edge = frozenset((mapped_left, mapped_right))
                if edge in landed:
                    continue
                landed.add(edge)
                if mapped_right not in adjacency[mapped_left]:
                    added += 1
                adjacency[mapped_left].add(mapped_right)
                adjacency[mapped_right].add(mapped_left)
                lag = here_lags.get((left, right))
                if lag is not None:
                    lags.setdefault((mapped_left, mapped_right), lag)
                    lags.setdefault((mapped_right, mapped_left), -lag)
        per_file.append({
            "path": str(path),
            "rows": rows,
            "edges_on_this_bundle": len(landed),
            "edge_ends_not_in_this_bundle": lost // 2,
            "edges_this_file_added": added,
        })
    return dict(adjacency), lags, per_file


def components(adjacency, universe):
    """Connected components over ``universe``, singletons included.

    A component is what a source-safe split must keep on one side.  Singletons
    are kept because the size distribution is the thing that decides whether a
    balanced split is still possible after merging.
    """
    seen = {}
    groups = []
    for node in sorted(universe):
        if node in seen:
            continue
        stack, group = [node], []
        seen[node] = len(groups)
        while stack:
            current = stack.pop()
            group.append(current)
            for neighbour in adjacency.get(current, ()):
                if neighbour in universe and neighbour not in seen:
                    seen[neighbour] = len(groups)
                    stack.append(neighbour)
        groups.append(sorted(group))
    return groups


def joints_for(bundle_root, record, cache):
    """Forward-kinematic joints for one bundle record, memoised."""
    name = record["recording_id"]
    if name not in cache:
        motion = np.load(str(bundle_root / record["motion_path"]))
        cache[name] = motion_151_to_joints(motion)
    return cache[name]


# --------------------------------------------------------------------------- #
# declared leakage


def declared_leakage(records, adjacency):
    """Which fingerprint groups straddle the split, and what it would cost to fix.

    ``moved`` is the smallest number of recordings that must change split to put
    every group on one side -- for each straddling group, everything outside its
    largest split.
    """
    universe = set(records)
    groups = components(adjacency, universe)
    straddling, moved, affected = [], 0, collections.Counter()
    for group in groups:
        splits = collections.Counter(records[name]["split"] for name in group)
        if len(splits) < 2:
            continue
        moved += len(group) - splits.most_common(1)[0][1]
        straddling.append({"size": len(group), "splits": dict(splits),
                           "members": group[:8]})
        for split in splits:
            affected[split] += splits[split]
    sizes = sorted((len(g) for g in groups), reverse=True)
    return {
        "recordings": len(universe),
        "groups": len(groups),
        "largest_groups": sizes[:10],
        "singletons": sum(1 for s in sizes if s == 1),
        "straddling_groups": len(straddling),
        "recordings_in_straddling_groups": sum(g["size"] for g in straddling),
        "recordings_that_must_move": moved,
        "by_split": dict(affected),
        "worst": sorted(straddling, key=lambda g: -g["size"])[:10],
    }


# --------------------------------------------------------------------------- #
# measured leakage


def calibrate(bundle_root, records, adjacency, sample, rng):
    """The two scales every threshold here is read against.

    ``null`` -- unrelated recordings, drawn as pairs the fingerprint does not
    connect.  ``declared`` -- pairs it does.  Printing both on every run is what
    makes the threshold auditable: if they overlap, the reading below is not
    separating anything and the run says so instead of ranking.
    """
    cache = {}
    names = sorted(records)
    null = []
    # Bounded, and the bound is reported.  An unbounded "draw until you have N"
    # loop never returns on a corpus where the draws cannot produce a reading --
    # a set of two recordings, one of them too short -- and a calibration step
    # that hangs is worse than one that says it found nothing.
    attempts = 0
    budget = max(20, sample * 20)
    while len(null) < sample and len(names) >= 2 and attempts < budget:
        attempts += 1
        left, right = rng.sample(names, 2)
        if right in adjacency.get(left, ()):
            continue
        value, _ = pose_distance(joints_for(bundle_root, records[left], cache),
                                 joints_for(bundle_root, records[right], cache))
        if value is not None:
            null.append(value)
    declared = []
    linked = [n for n in names if adjacency.get(n)]
    rng.shuffle(linked)
    for left in linked:
        for right in sorted(adjacency[left]):
            if right not in records:
                continue
            value, _, _ = best_over_lag(joints_for(bundle_root, records[left], cache),
                                        joints_for(bundle_root, records[right], cache))
            if value is not None:
                declared.append(value)
            break
        if len(declared) >= sample:
            break
    return {
        "null_pairs": len(null),
        "null_draws_attempted": attempts,
        "null_median": _median(null),
        "null_p01": _quantile(null, 0.01),
        "null_p05": _quantile(null, 0.05),
        "declared_pairs": len(declared),
        "declared_median": _median(declared),
        "declared_p95": _quantile(declared, 0.95),
    }, cache


def _median(values):
    return float(statistics.median(values)) if values else None


def _quantile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return float(ordered[index])


def measured_leakage(bundle_root, records, adjacency, threshold, *,
                     test_sample, train_sample, rng, cache):
    """Is a held-out clip's nearest training neighbour an outlier, or just the smallest?

    The reading is a *ratio*, not a distance, and the first version of this
    function got that wrong in a way worth recording.  It compared the nearest of
    400 candidates against the 1st percentile of single random pairs and called
    everything below it a leak: on the corpus that is 64 of 80 held-out
    recordings, which is not a finding, it is an order statistic.  A minimum over
    400 draws is *supposed* to sit far into the tail of the single-draw
    distribution.

    So each query is normalised against its own neighbourhood:

        ratio_query = (nearest training neighbour) / (5th percentile of the same
                       query's distances to the whole pool)

    A dance that is merely typical of the corpus has a smooth neighbour
    distribution and lands near 1.0 however close its nearest happens to be.  A
    dance the pool actually contains a copy of has one neighbour detached from
    the rest, and lands well below 1.

    That still is not enough, and the second version of this function is why.
    Run on the corpus it named ``wild_v4:7636320671338313856:clip000`` as the
    nearest neighbour of **53 of 200** held-out recordings.  That clip's
    per-frame joint speed is 0.00232 against the corpus median 0.01387 -- a
    near-static reconstruction, a failed track -- and a still figure sits near
    the mean pose, so its median distance to 150 random recordings is 0.1506
    against the corpus null of 0.2058.  It is close to everyone.  This is
    hubness, and a hub answers "is there a copy of this dance in train" with yes
    for every query.

    So the flag is the *worse* of two normalisations -- the query's own
    neighbourhood and the candidate's:

        ratio = max(d / p05(query's distances), d / p05(candidate's distances))

    A real duplicate is unusually close in both directions.  A hub is unusually
    close only in the query's direction, and its own column is tight, so the
    second term returns it to ~1.  The candidate column is taken over the same
    query set that was already computed, so this costs nothing.

    Threshold by control, not by hand.  Two classes are drawn from the data:

    * **positive** -- queries whose fingerprint-declared partner is in the pool;
    * **negative** -- queries the fingerprint connects to nothing anywhere.

    Both distributions are reported on every run, and the default threshold is
    the negative class's 5th percentile, so the false-positive rate the reading
    is run at is a stated number rather than a hope.  If the two classes do not
    separate, the run says so and the count below is not evidence -- printing a
    number from a criterion that cannot discriminate is how this repo spent
    2026-08-19.

    Candidates are ranked at lag 0 -- right for two cuts of one track, wrong for
    anything offset -- then the closest few are re-scored over a lag window, so a
    pair that is only close after a shift is still caught.
    """
    held = [n for n, r in records.items() if r["split"] in ("test", "val")]
    train = [n for n, r in records.items() if r["split"] == "train"]
    if not held or not train:
        raise AuditError("bundle has no held-out split or no train split")
    held = sorted(rng.sample(held, min(test_sample, len(held))))
    pool = sorted(rng.sample(train, min(train_sample, len(train))))
    pool_set = set(pool)

    # One pass builds the whole query x candidate matrix, because the hubness
    # correction needs each candidate's column and recomputing it would double
    # the cost.  NaN marks a pair with too little overlap to be a reading.
    #
    # The second matrix is the same search against every candidate played
    # BACKWARDS.  Time reversal keeps a recording's poses, its joint speeds and
    # its length, and destroys the one thing a leak is made of -- the order the
    # movements come in.  So whatever the forward search flags, the reversed
    # search flags too unless the flag is about choreography.  This is here
    # because an independent check on 2026-08-25 ran the forward version alone
    # and got 120 of 250 test recordings with a training neighbour under 0.10 --
    # then got 120 of 250 from the reversed null, paired difference centred on
    # zero.  Every one of those 120 was an extreme-value artifact of minimising
    # over 1,500 candidates x 121 lags.  Only the excess over this null is
    # evidence, and the excess is what gets printed.
    matrix = np.full((len(held), len(pool)), np.nan)
    reversed_matrix = np.full((len(held), len(pool)), np.nan)
    pool_joints = [joints_for(bundle_root, records[other], cache) for other in pool]
    for i, name in enumerate(held):
        left = joints_for(bundle_root, records[name], cache)
        for j, right in enumerate(pool_joints):
            value, _ = pose_distance(left, right)
            if value is not None:
                matrix[i, j] = value
            value, _ = pose_distance(left, right[::-1])
            if value is not None:
                reversed_matrix[i, j] = value

    def _p05(values):
        values = values[~np.isnan(values)]
        return float(np.quantile(values, 0.05)) if len(values) >= MIN_POOL_FOR_RATIO else None

    energies = {}
    for j, other in enumerate(pool):
        joints = rootless(joints_for(bundle_root, records[other], cache))
        energies[other] = float(np.median(np.linalg.norm(np.diff(joints, axis=0), axis=-1)))
    corpus_energy = float(np.quantile(list(energies.values()), 0.05))

    rows, skipped = [], 0
    for i, name in enumerate(held):
        row = matrix[i]
        query_reference = _p05(row)
        if query_reference is None:
            skipped += 1
            continue
        order = np.argsort(np.where(np.isnan(row), np.inf, row))
        left = joints_for(bundle_root, records[name], cache)
        rescored = []
        for j in order[:LAG_RESCORE_TOP_K]:
            if np.isnan(row[j]):
                continue
            best, _, lag = best_over_lag(left, joints_for(bundle_root, records[pool[j]], cache))
            rescored.append((best if best is not None else float(row[j]), int(j), lag))
        if not rescored:
            skipped += 1
            continue
        rescored.sort()
        best, j, lag = rescored[0]
        neighbour = pool[j]
        candidate_reference = _p05(matrix[:, j])
        ratios = [best / query_reference] if query_reference else []
        if candidate_reference:
            ratios.append(best / candidate_reference)
        linked = adjacency.get(name, ())
        rows.append({
            "recording_id": name, "split": records[name]["split"],
            "nearest_train": neighbour, "distance": round(best, 4), "lag": lag,
            "query_reference_p05": round(query_reference, 4) if query_reference else None,
            "candidate_reference_p05": round(candidate_reference, 4) if candidate_reference else None,
            # the worse of the two: a hub is close to the query but its own
            # column is tight, and that is what returns it to ~1
            "ratio": round(max(ratios), 4) if ratios else None,
            "candidate_is_near_static": bool(energies[neighbour] < corpus_energy),
            "declared": neighbour in linked,
            "declared_partner_in_pool": bool(set(linked) & pool_set),
            "declared_anywhere": bool(linked),
            "pool_median": round(float(np.nanmedian(row)), 4),
        })
    # The same statistic on the reversed matrix, query by query -- TWICE, and
    # the pair of readings is the point.
    #
    # ``null_ratios`` is the reading this tool has always produced: the lag-0
    # argmin, no rescore.  ``lag_matched_null_ratios`` gives the null the search
    # the forward reading actually performs -- LAG_RESCORE_TOP_K candidates each
    # minimised over the 121 lags of ``best_over_lag``.
    #
    # Why both, rather than replacing the first with the second.  The block
    # comment above says this null exists to cancel "an extreme-value artifact
    # of minimising over 1,500 candidates x 121 lags", and a null that is not
    # allowed to perform that minimisation cannot cancel it: the original is a
    # strictly weaker search than the statistic it is the null for, so its
    # ratios run large and ``excess_over_time_reversed_null`` is biased
    # positive.  That much is arithmetic.  But the matched null is not simply
    # the better one: on this file's own positive control -- a planted copy,
    # ``test_the_time_reversed_null_is_reported_and_can_veto_the_forward_
    # reading`` -- it cancels a leak that is really there, so adopting it alone
    # would trade a false-positive bias for a false-negative one.  Two criteria
    # pointing opposite ways is the case CLAUDE.md 2.1 says to stop on rather
    # than pick the one that reads better, so both travel in the report and a
    # verdict that depends on which one is used says so in
    # ``the_two_nulls_disagree``.
    null_ratios = []
    lag_matched_null_ratios = []
    lag_matched_null_by_name = {}
    for i in range(len(held)):
        row = reversed_matrix[i]
        query_reference = _p05(row)
        if query_reference is None:
            continue
        j = int(np.nanargmin(np.where(np.isnan(row), np.inf, row)))
        best = float(row[j])
        candidate_reference = _p05(reversed_matrix[:, j])
        ratios = [best / query_reference]
        if candidate_reference:
            ratios.append(best / candidate_reference)
        null_ratios.append(max(ratios))

        order = np.argsort(np.where(np.isnan(row), np.inf, row))
        left = joints_for(bundle_root, records[held[i]], cache)
        rescored = []
        for k in order[:LAG_RESCORE_TOP_K]:
            if np.isnan(row[k]):
                continue
            matched, _, _ = best_over_lag(left, pool_joints[k][::-1])
            rescored.append((matched if matched is not None else float(row[k]), int(k)))
        if not rescored:
            continue
        rescored.sort()
        matched, k = rescored[0]
        matched_candidate_reference = _p05(reversed_matrix[:, k])
        matched_ratios = [matched / query_reference]
        if matched_candidate_reference:
            matched_ratios.append(matched / matched_candidate_reference)
        lag_matched_null_ratios.append(max(matched_ratios))
        lag_matched_null_by_name[held[i]] = max(matched_ratios)

    positive = [r["ratio"] for r in rows if r["declared_partner_in_pool"] and r["ratio"]]
    negative = [r["ratio"] for r in rows if not r["declared_anywhere"] and r["ratio"]]
    if threshold is None:
        threshold = _quantile(negative, 0.05) if len(negative) >= 10 else None
    separates = (len(positive) >= 5 and len(negative) >= 10
                 and _median(positive) is not None
                 and _median(positive) < _quantile(negative, 0.25))
    flagged = [r for r in rows
               if threshold is not None and r["ratio"] is not None and r["ratio"] < threshold]
    undeclared = [r for r in flagged if not r["declared"]]
    null_flagged = [r for r in null_ratios if threshold is not None and r < threshold]
    lag_matched_null_flagged = [r for r in lag_matched_null_ratios
                                if threshold is not None and r < threshold]
    # Paired, which is what this file's own plan-level reading already does
    # (``clips_whose_plan_beats_its_own_reversed_null``).  Comparing two counts
    # against a shared threshold asks whether the forward search flags more
    # queries than the reversed one; comparing each query against ITS OWN
    # reversed reading asks whether this recording resembles a training
    # recording more than an equally hard search on choreography-destroyed
    # candidates does.  The second question is the one a leak answers, it
    # survives a threshold that lands badly, and it does not depend on the two
    # searches flagging the same number of things.
    paired = [(r["ratio"], lag_matched_null_by_name[r["recording_id"]]) for r in rows
              if r["ratio"] is not None and r["recording_id"] in lag_matched_null_by_name]
    beats_own_null = sum(1 for forward, reverse in paired if forward < reverse)
    paired_median = (statistics.median([reverse - forward for forward, reverse in paired])
                     if paired else None)
    return {
        "statistic": "nearest training neighbour / this query's own 5th-percentile distance",
        "threshold": round(threshold, 4) if threshold is not None else None,
        "threshold_basis": "5th percentile of queries the fingerprint links to nothing",
        "controls_separate": bool(separates),
        "positive_control": {"queries": len(positive), "ratio_median": _median(positive)},
        "negative_control": {"queries": len(negative), "ratio_median": _median(negative),
                             "ratio_p05": _quantile(negative, 0.05)},
        "held_out_checked": len(rows),
        "skipped_too_few_comparisons": skipped,
        "train_pool": len(pool),
        "flagged": len(flagged),
        "flagged_fraction": round(len(flagged) / len(rows), 4) if rows else None,
        # Time-reversed candidates: same poses, same speeds, no choreography.
        # Anything the forward search flags that this also flags is an
        # extreme-value artifact, not a leak.
        "time_reversed_null_flagged": len(null_flagged),
        "excess_over_time_reversed_null": len(flagged) - len(null_flagged),
        "reading_is_at_its_noise_floor": len(flagged) <= len(null_flagged),
        # The second null, and the only honest way to read the first one: the
        # forward statistic minimises over LAG_RESCORE_TOP_K candidates x 121
        # lags and the null above does not, so the excess above is inflated by
        # whatever that extra minimisation buys.  These say by how much.
        "lag_matched_null_flagged": len(lag_matched_null_flagged),
        "excess_over_lag_matched_null": len(flagged) - len(lag_matched_null_flagged),
        "reading_is_at_its_lag_matched_noise_floor": (
            len(flagged) <= len(lag_matched_null_flagged)),
        # Loud when the answer depends on which null is used.  A verdict that
        # flips here is not a verdict; it is a statement about the search.
        # The paired reading, on the same footing as the plan-level one.  Under
        # the null that this recording is not a copy, forward and reversed are
        # two draws of the same search and each query beats its own reversed
        # reading half the time; a corpus with leaks pushes this up.
        "queries_paired_against_their_own_lag_matched_null": len(paired),
        "queries_closer_to_train_than_to_their_own_reversed_null": beats_own_null,
        "paired_median_reversed_minus_forward": _round(paired_median),
        "the_two_nulls_disagree": (
            (len(flagged) <= len(null_flagged))
            != (len(flagged) <= len(lag_matched_null_flagged))),
        "undeclared_by_fingerprint": len(undeclared),
        "undeclared_examples": undeclared[:10],
        "nearest_distance_median": _median([r["distance"] for r in rows]),
        # Reported rather than quietly dropped: a near-static reconstruction is
        # a 3D failure that sits near the mean pose, so it is close to
        # everything.  The hubness correction stops it deciding the flag; this
        # number says how much of the pool is in that state at all.
        "near_static_in_pool": sum(1 for name in pool if energies[name] < corpus_energy),
        "pool_energy_p05": round(corpus_energy, 5),
        "flagged_whose_neighbour_is_near_static": sum(
            1 for r in flagged if r["candidate_is_near_static"]),
        "hubness_top": collections.Counter(
            r["nearest_train"] for r in rows).most_common(5),
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
# the model-level acceptance test


def stitch_labels(windows_root):
    """Per-frame label track per recording, from a window set's slices."""
    root = pathlib.Path(windows_root)
    arrays = {}
    for split in ("train", "val", "test"):
        path = root / split / "labels.npy"
        if path.is_file():
            arrays[split] = np.load(str(path), mmap_mode="r")
    index = collections.defaultdict(list)
    for line in (root / "windows.jsonl").open(encoding="utf-8"):
        line = line.strip()
        if line:
            row = json.loads(line)
            index[row["recording_id"]].append(row)
    tracks = {}
    for name, windows in index.items():
        length = max(w["end_frame_exclusive"] for w in windows)
        track = np.full(length, -1, dtype=np.int64)
        for window in sorted(windows, key=lambda w: w["start_frame"]):
            labels = arrays[window["split"]][window["array_index"]][:window["length"]]
            track[window["start_frame"]:window["end_frame_exclusive"]] = labels
        tracks[name] = track
    return tracks


def label_agreement(left, right):
    """Fraction of jointly-labelled frames that carry the same label."""
    length = min(len(left), len(right))
    valid = (left[:length] >= 0) & (right[:length] >= 0)
    if valid.sum() < MIN_OVERLAP_FRAMES:
        return None
    return float((left[:length][valid] == right[:length][valid]).mean())


def read_generated(directory):
    """``recording_id -> (joints, plan)`` for generated pickles, seed suffix stripped."""
    out = {}
    for path in sorted(pathlib.Path(directory).glob("*.pkl")):
        stem = path.stem
        marker = stem.rfind("_s")
        if marker > 0 and stem[marker + 2:].isdigit():
            stem = stem[:marker]
        if stem in out:
            continue
        with path.open("rb") as handle:
            blob = pickle.load(handle)
        out[stem] = (np.asarray(blob["full_pose"]),
                     np.asarray(blob["atomic_labels"]) if "atomic_labels" in blob else None)
    return out


def resolve_generated(generated, records):
    """Land a directory of generated pickles onto this bundle's ids, or refuse.

    ``read_generated`` keys on the pickle's own file name, and that name is the
    recording id of *the corpus that generated it*.  Read against another
    corpus -- which is the whole reason ``strip_generation`` exists, and what
    ``--twin-cohort`` and ``--pairs`` are already written to survive -- an exact
    key match lands nothing, and landing nothing is not visible downstream:
    ``_summarise([])`` reports ``clips: 0`` and
    ``closer_to_train_twin_than_own: 0``, which is exactly the shape of a clean
    reading.  So this does what the module docstring already says the cohort
    does: match on the stripped id, and make "resolved onto nothing" a hard
    error instead of an empty group that reads like a pass.

    Exact matches are taken first and never overridden, so a directory from the
    bundle's own generation resolves to precisely the pairing the exact-key
    filter used to produce.
    """
    by_stripped = {}
    for name in records:
        by_stripped.setdefault(strip_generation(name), name)
    resolved, unresolved, collisions = {}, [], 0
    for name in sorted(generated):
        target = name if name in records else by_stripped.get(strip_generation(name))
        if target is None:
            unresolved.append(name)
            continue
        if target in resolved:
            collisions += 1
            continue
        resolved[target] = generated[name]
    if not resolved:
        raise SystemExit(
            "--generated holds {} pickle(s) and none of them resolves onto a recording of "
            "this bundle ({} recording(s) read); ids are matched exactly and then with the "
            "generation prefix stripped, so check that the motion was generated from this "
            "corpus lineage. First unresolved: {}".format(
                len(generated), len(records), unresolved[:5]))
    stats = {
        "pickles_read": len(generated),
        "clips_resolved": len(resolved),
        "pickles_not_in_this_bundle": len(unresolved),
        "pickles_dropped_to_a_taken_id": collisions,
    }
    return resolved, stats


def strip_generation(name):
    """``wild_v4:7643034659055492840:clip000`` -> ``7643034659055492840:clip000``.

    The generation prefix is what changes between the corpus being disproved
    and the corpus being accepted, and it is the only part that changes: the
    upload id and clip index are the corpus's own identity.  Matching on the
    stripped id is what lets a v4 cohort be read against a v5 bundle.
    """
    return name.split(":", 1)[1] if ":" in name else name


def twin_cohort_from_split(records, adjacency):
    """Who has a twin *in train* right now -- the reading to freeze and carry.

    This is the rule ``model_leakage`` used to apply inline.  It is correct on
    the corpus whose leakage is being characterised and vacuous on the corpus
    that fixed it, which is why it is emitted there and read here.
    """
    cohort = {}
    for name, record in records.items():
        if record["split"] == "train":
            continue
        twins = sorted(t for t in adjacency.get(name, ())
                       if t in records and records[t]["split"] == "train")
        if twins:
            cohort[name] = twins
    return cohort


def read_twin_cohort(path, records):
    """Resolve an emitted cohort onto this bundle, or refuse.

    Both ends are resolved by stripped id, and both ends can fail to resolve:
    a clip that no longer exists (the re-cut dropped it) and a twin that no
    longer exists are counted and reported separately, because "the cohort
    shrank" and "the cohort did not match" have very different meanings and an
    unreported merge of the two is how a criterion quietly stops measuring.
    """
    payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    entries = payload.get("cohort", payload)
    by_stripped = {}
    for name in records:
        by_stripped.setdefault(strip_generation(name), name)
    resolved, dropped_clips, dropped_twins = {}, [], 0
    for name, twins in entries.items():
        here = by_stripped.get(strip_generation(name))
        if here is None:
            dropped_clips.append(name)
            continue
        mapped = []
        for twin in twins:
            target = by_stripped.get(strip_generation(twin))
            if target is None:
                dropped_twins += 1
            else:
                mapped.append(target)
        if mapped:
            resolved[here] = mapped
        else:
            dropped_clips.append(name)
    if not resolved:
        raise SystemExit(
            "--twin-cohort {} resolves onto no recording of this bundle ({} entr(ies) read). "
            "Ids are matched with the generation prefix stripped; check that the cohort was "
            "emitted from the same corpus lineage.".format(path, len(entries)))
    stats = {
        "source": str(path),
        "entries_read": len(entries),
        "clips_resolved": len(resolved),
        "clips_not_in_this_bundle": len(dropped_clips),
        "twin_recordings_not_in_this_bundle": dropped_twins,
    }
    return resolved, stats


def model_leakage(bundle_root, records, adjacency, generated, tracks, cache,
                  cohort=None, plan_train_sample=200, rng=None):
    """Does the generation reproduce its own clip, or the training copy of it.

    Two readings per clip, both normalised inside the clip so an easy dance
    cannot look like a leak:

    ``rank`` -- where the clip's own reconstruction sits among every other
    measured clip's, for this generation.  0.0 means its own is the closest of
    all; chance is 0.5.

    ``plan_own`` / ``plan_twin`` -- how far the emitted plan agrees with the
    clip's own label track versus its training twin's.

    ``plan_best_train`` -- the reading that survives a *correct* split, and the
    one the acceptance criterion needs on more than a handful of clips.  The
    twin group is thin by construction after a song-disjoint re-split (21 clips
    on wild_v5 against 65 generated), so "does the plan replay *this* training
    recording" has to be widened to "does it replay *any* of them".  It is the
    best label agreement over a sample of training recordings.

    A max over N samples is an extreme-value statistic, and comparing one
    against a median is precisely the mistake ``measured_leakage`` records in
    its own docstring -- it convicted 64 of 80 recordings before it was fixed.
    So the null here is the same one that fixed that: **the identical max, taken
    over the same training recordings with their label tracks reversed in
    time.**  A memorised sequence agrees forwards and not backwards; a search
    floor agrees equally either way, and only the excess over the reversed
    reading is evidence.  Both numbers are printed, never just the forward one.

    Clips whose generated length disagrees with their reference by more than 1%
    are reported separately rather than dropped: that disagreement is itself a
    defect (the generated length is the music length), and mixing the two
    populations is what made the first pass of this measurement ambiguous.
    """
    measurable = [n for n in generated if n in records]
    # Drawn once, so every clip is scored against the same training recordings
    # -- a per-clip redraw would let the sample itself explain a difference
    # between two clips.
    train_pool = sorted(name for name, record in records.items()
                        if record["split"] == "train" and name in tracks)
    if rng is not None and len(train_pool) > plan_train_sample:
        train_pool = sorted(rng.sample(train_pool, plan_train_sample))
    elif len(train_pool) > plan_train_sample:
        train_pool = train_pool[:plan_train_sample]
    reversed_tracks = {name: tracks[name][::-1].copy() for name in train_pool}
    lengths, rows = {}, []
    for name in measurable:
        reference = joints_for(bundle_root, records[name], cache)
        lengths[name] = len(generated[name][0]) / max(1, len(reference))
    matched = [n for n in measurable if abs(lengths[n] - 1) <= 0.01]
    for name in measurable:
        motion, plan = generated[name]
        own, _ = pose_distance(motion, joints_for(bundle_root, records[name], cache))
        if own is None:
            continue
        others = []
        for other in matched:
            if other == name:
                continue
            value, _ = pose_distance(motion, joints_for(bundle_root, records[other], cache))
            if value is not None:
                others.append(value)
        # Two readings, kept apart on purpose.  ``in_train_now`` is what the
        # current split says and is 0 on a corpus that fixed the split;
        # ``twins`` is what the criterion is read on -- the recordings that
        # were this clip's training twin in the generation being disproved,
        # wherever they sit today.  With no cohort supplied the two coincide,
        # which is the pre-2026-08-26 behaviour.
        in_train_now = [t for t in adjacency.get(name, ()) if t in records
                        and records[t]["split"] == "train"]
        twins = cohort.get(name, []) if cohort is not None else in_train_now
        twin_distance = None
        if twins:
            scored = [best_over_lag(motion, joints_for(bundle_root, records[t], cache))[0]
                      for t in twins]
            scored = [s for s in scored if s is not None]
            twin_distance = min(scored) if scored else None
        row = {
            "recording_id": name,
            "has_train_twin": bool(twins),
            "twin_in_train_now": bool(in_train_now),
            "length_ratio": round(lengths[name], 4),
            "distance_to_own": round(own, 4),
            # `is not None`, not truthiness: a distance of 0.0 is the strongest
            # possible reading of this measurement -- the generation *is* the
            # training copy -- and truthiness would report it as "not measured".
            "distance_to_train_twin": _round(twin_distance),
            "rank_of_own": round(sum(1 for v in others if v < own) / len(others), 4)
            if others else None,
        }
        if tracks and plan is not None and train_pool:
            forward = [label_agreement(plan, tracks[t]) for t in train_pool if t != name]
            backward = [label_agreement(plan, reversed_tracks[t])
                        for t in train_pool if t != name]
            forward = [v for v in forward if v is not None]
            backward = [v for v in backward if v is not None]
            if forward:
                row["plan_vs_best_train"] = _round(max(forward))
                row["plan_vs_median_train"] = _round(statistics.median(forward))
                row["train_recordings_compared"] = len(forward)
            if backward:
                row["plan_vs_best_train_reversed"] = _round(max(backward))
        if tracks and plan is not None and name in tracks:
            row["plan_vs_own"] = _round(label_agreement(plan, tracks[name]))
            twin_scores = [label_agreement(plan, tracks[t]) for t in twins if t in tracks]
            twin_scores = [s for s in twin_scores if s is not None]
            row["plan_vs_train_twin"] = _round(max(twin_scores)) if twin_scores else None
        rows.append(row)
    return {
        "generated": len(generated),
        "twin_source": "cohort" if cohort is not None else "current split",
        "plan_vs_train_pool": len(train_pool),
        "all_clips": _summarise(rows),
        "clips_with_a_twin_in_train_now": sum(1 for r in rows if r["twin_in_train_now"]),
        "in_bundle": len(measurable),
        "length_matched": len(matched),
        "length_mismatched": len(measurable) - len(matched),
        "with_train_twin": _summarise([r for r in rows if r["has_train_twin"]]),
        "without_train_twin": _summarise([r for r in rows if not r["has_train_twin"]]),
        "rows": rows,
    }


def _round(value):
    return round(value, 4) if value is not None else None


def _summarise(rows):
    def median_of(key):
        values = [r[key] for r in rows if r.get(key) is not None]
        return _round(statistics.median(values)) if values else None
    forward = [r["plan_vs_best_train"] for r in rows
               if r.get("plan_vs_best_train") is not None]
    backward = [r["plan_vs_best_train_reversed"] for r in rows
                if r.get("plan_vs_best_train_reversed") is not None]
    return {
        "clips": len(rows),
        "rank_of_own_median": median_of("rank_of_own"),
        "plan_vs_best_train_median": median_of("plan_vs_best_train"),
        "plan_vs_best_train_reversed_median": median_of("plan_vs_best_train_reversed"),
        "plan_vs_median_train_median": median_of("plan_vs_median_train"),
        # The only part of the forward reading that is evidence.  Paired, not a
        # difference of medians: the null is drawn on the same clip against the
        # same recordings, so the pairing is free and a difference of medians
        # would throw it away.
        "plan_excess_over_reversed_median": _round(statistics.median(
            [f - b for f, b in zip(forward, backward)])) if forward and backward else None,
        "clips_whose_plan_beats_its_own_reversed_null": sum(
            1 for r in rows
            if r.get("plan_vs_best_train") is not None
            and r.get("plan_vs_best_train_reversed") is not None
            and r["plan_vs_best_train"] > r["plan_vs_best_train_reversed"]),
        "distance_to_own_median": median_of("distance_to_own"),
        "distance_to_train_twin_median": median_of("distance_to_train_twin"),
        "plan_vs_own_median": median_of("plan_vs_own"),
        "plan_vs_train_twin_median": median_of("plan_vs_train_twin"),
        "closer_to_train_twin_than_own": sum(
            1 for r in rows
            if r.get("distance_to_train_twin") is not None
            and r["distance_to_train_twin"] < r["distance_to_own"]),
    }


# --------------------------------------------------------------------------- #


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Measure how far the evaluation set crosses the training set")
    parser.add_argument("--bundle", required=True,
                        help="performance bundle directory or its sequences.jsonl")
    parser.add_argument("--pairs", required=True, nargs="+",
                        help="fingerprint-confirmed same-track pairs (jsonl).  Takes "
                             "every list ever measured on this corpus, not the newest: "
                             "each run's recall is well under half and two runs do not "
                             "miss the same pairs, so auditing against one list is "
                             "auditing against less evidence than the split was built "
                             "from.  Ids are matched with the generation prefix stripped")
    parser.add_argument("--windows", help="window set root, for the plan-level reading")
    parser.add_argument("--generated", help="directory of generated *.pkl to accept or refuse")
    parser.add_argument("--twin-cohort",
                        help="cohort file from --emit-twin-cohort: the recordings that were "
                             "each clip's TRAINING twin in the generation being disproved. "
                             "The --generated reading is taken against those recordings "
                             "wherever they sit in this split -- which is the only way the "
                             "acceptance test survives the re-split that empties the group")
    parser.add_argument("--emit-twin-cohort", type=pathlib.Path,
                        help="write the cohort implied by THIS bundle's split and exit-code "
                             "unchanged; run it on the corpus whose leakage was measured")
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--test-sample", type=int, default=80,
                        help="held-out recordings to sweep for a training neighbour")
    parser.add_argument("--train-sample", type=int, default=400,
                        help="training recordings the sweep looks through")
    parser.add_argument("--calibration-sample", type=int, default=200)
    parser.add_argument("--threshold", type=float,
                        help="ratio (nearest neighbour / this query's own 5th-percentile "
                             "distance) below which a training neighbour is called a copy; "
                             "default is the 5th percentile of the negative control measured "
                             "on this corpus at run time, so the false-positive rate is stated")
    parser.add_argument("--policy", choices=("refuse", "warn"), default="warn",
                        help="refuse exits non-zero when any leakage is measured")
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rng = random.Random(args.seed)
    bundle_root, records = read_bundle(args.bundle)
    pairs = ([args.pairs] if isinstance(args.pairs, (str, pathlib.Path))
             else list(args.pairs))
    adjacency, _lags, per_pair_file = merge_pair_files(pairs, records)
    pair_count = sum(entry["rows"] for entry in per_pair_file)

    report = {
        "schema_version": "atomicdance-split-leakage-v1",
        "bundle": str(args.bundle),
        "pairs": [str(path) for path in pairs],
        "pairs_read": per_pair_file,
        "pair_rows": pair_count,
        "sequences": len(records),
        "by_split": dict(collections.Counter(r["split"] for r in records.values())),
    }
    report["declared"] = declared_leakage(records, adjacency)

    if args.emit_twin_cohort:
        # Split-level only -- no motion is read, so this mode runs from a bare
        # sequences.jsonl long after the generation it describes has been
        # evicted.  That matters: the cohort has to be captured from the corpus
        # being disproved, which is by definition the older one.
        cohort = twin_cohort_from_split(records, adjacency)
        args.emit_twin_cohort.parent.mkdir(parents=True, exist_ok=True)
        args.emit_twin_cohort.write_text(json.dumps({
            "schema_version": "atomicdance-twin-cohort-v1",
            "bundle": str(args.bundle),
            "pairs": [str(path) for path in pairs],
            "rule": "held-out recordings whose fingerprint twin was in this bundle's train split",
            "clips": len(cohort),
            "twin_recordings": len({t for twins in cohort.values() for t in twins}),
            "cohort": cohort,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report["twin_cohort_emitted"] = {"path": str(args.emit_twin_cohort),
                                         "clips": len(cohort)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("emitted twin cohort: {} held-out clip(s) with a twin in this split's train "
              "-> {}".format(len(cohort), args.emit_twin_cohort))
        if not cohort:
            print("REFUSED: this bundle's split leaves no held-out clip with a train twin, "
                  "so there is nothing to carry -- emit from the corpus whose leakage was "
                  "measured, not from the one that fixed it", file=sys.stderr)
            return 1
        return 0

    scales, cache = calibrate(bundle_root, records, adjacency, args.calibration_sample, rng)
    report["scales"] = scales
    report["measured"] = measured_leakage(
        bundle_root, records, adjacency, args.threshold,
        test_sample=args.test_sample, train_sample=args.train_sample, rng=rng, cache=cache)

    if args.generated:
        tracks = stitch_labels(args.windows) if args.windows else {}
        cohort = None
        if args.twin_cohort:
            cohort, cohort_stats = read_twin_cohort(args.twin_cohort, records)
            report["twin_cohort"] = cohort_stats
        resolved_generated, generated_stats = resolve_generated(
            read_generated(args.generated), records)
        report["generated_source"] = generated_stats
        report["model"] = model_leakage(
            bundle_root, records, adjacency, resolved_generated, tracks, cache,
            cohort=cohort, plan_train_sample=args.train_sample, rng=rng)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    declared = report["declared"]
    measured = report["measured"]
    print("{} sequence(s) {}".format(report["sequences"], report["by_split"]))
    for entry in per_pair_file:
        print("  {}: {} row(s) -> {} edge(s) here (+{} new, {} end outside this bundle)"
              .format(entry["path"], entry["rows"], entry["edges_on_this_bundle"],
                      entry["edges_this_file_added"],
                      entry["edge_ends_not_in_this_bundle"]))
    print("declared: {} same-track group(s), {} straddle the split, "
          "{} recording(s) would have to move".format(
              declared["groups"], declared["straddling_groups"],
              declared["recordings_that_must_move"]))
    print("scales: unrelated pairs median {}, fingerprint-declared pairs median {} "
          "(declared is bimodal by construction -- same track is not same choreography)".format(
              _fmt(scales["null_median"]), _fmt(scales["declared_median"])))
    print("controls: {} positive quer(ies) ratio median {}, {} negative ratio median {} -> {}".format(
        measured["positive_control"]["queries"], _fmt(measured["positive_control"]["ratio_median"]),
        measured["negative_control"]["queries"], _fmt(measured["negative_control"]["ratio_median"]),
        "separated" if measured["controls_separate"] else
        "NOT SEPARATED -- the count below is not evidence"))
    print("measured: {} of {} held-out recording(s) have a training neighbour detached from their "
          "own neighbourhood (ratio < {}) -- {} of them not declared by the fingerprint".format(
              measured["flagged"], measured["held_out_checked"],
              _fmt(measured["threshold"]), measured["undeclared_by_fingerprint"]))
    print("  time-reversed null flags {} of the same queries -> excess {}{}".format(
        measured["time_reversed_null_flagged"], measured["excess_over_time_reversed_null"],
        "; the forward reading is at its own noise floor and is not evidence"
        if measured["reading_is_at_its_noise_floor"] else ""))
    print("  a clean measured reading does NOT certify independence: this instrument's "
          "sensitivity to same-choreography was calibrated at ~13% (2026-08-25)")
    if measured["skipped_too_few_comparisons"]:
        print("  {} skipped for want of comparable pool members (not counted as clean)".format(
            measured["skipped_too_few_comparisons"]))
    for row in measured["undeclared_examples"][:5]:
        print("    {} -> {}  distance {} ratio {}".format(
            row["recording_id"], row["nearest_train"], _fmt(row["distance"]), _fmt(row["ratio"])))
    if "model" in report:
        model = report["model"]
        if "twin_cohort" in report:
            stats = report["twin_cohort"]
            print("twin cohort: {} of {} entr(ies) resolve onto this bundle "
                  "({} clip(s) absent, {} twin recording(s) absent); {} of them still have a "
                  "twin in train -- the group below is read against the carried recordings "
                  "wherever they now sit".format(
                      stats["clips_resolved"], stats["entries_read"],
                      stats["clips_not_in_this_bundle"],
                      stats["twin_recordings_not_in_this_bundle"],
                      model["clips_with_a_twin_in_train_now"]))
        else:
            print("twin cohort: none supplied -- the group below is whoever has a twin in "
                  "train NOW ({}), which a correct song-disjoint split empties by "
                  "construction; pass --twin-cohort to read the acceptance test".format(
                      model["clips_with_a_twin_in_train_now"]))
        broad = model["all_clips"]
        print("plan vs any of {} training recording(s): best {} forward, {} on the same "
              "recordings reversed; paired excess {}, {} of {} clip(s) beat their own "
              "reversed null (random-train level on wild_v4 was 0.0150)".format(
                  model["plan_vs_train_pool"],
                  _fmt(broad["plan_vs_best_train_median"]),
                  _fmt(broad["plan_vs_best_train_reversed_median"]),
                  _fmt(broad["plan_excess_over_reversed_median"]),
                  broad["clips_whose_plan_beats_its_own_reversed_null"], broad["clips"]))
        for tag in ("with_train_twin", "without_train_twin"):
            group = report["model"][tag]
            print("{}: {} clip(s), own-reference rank {}, plan vs own {}, plan vs train twin {}, "
                  "{} closer to the twin than to their own video".format(
                      tag.replace("_", " "), group["clips"], _fmt(group["rank_of_own_median"]),
                      _fmt(group["plan_vs_own_median"]), _fmt(group["plan_vs_train_twin_median"]),
                      group["closer_to_train_twin_than_own"]))
        if report["model"]["length_mismatched"]:
            print("  {} generated sequence(s) disagree with their reference on length -- "
                  "read separately, that is a music-provenance defect not a leak".format(
                      report["model"]["length_mismatched"]))
    print("wrote", args.output)

    # The declared reading is exact and can convict on its own.  The measured
    # one convicts only on its excess over the time-reversed null -- and it can
    # never acquit: an independent calibration put its sensitivity to
    # same-choreography at ~13%, so "measured nothing" is a statement about the
    # instrument, not about the corpus.  That is printed, not implied.
    leaking = declared["straddling_groups"] or max(0, measured["excess_over_time_reversed_null"])
    if leaking and args.policy == "refuse":
        print("REFUSED: the evaluation set crosses the training set", file=sys.stderr)
        return 1
    return 0


def _fmt(value):
    return "n/a" if value is None else "{:.4f}".format(value)


if __name__ == "__main__":
    raise SystemExit(main())
