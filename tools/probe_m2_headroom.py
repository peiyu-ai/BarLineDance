#!/usr/bin/env python3
"""How much tighter could M2's partition be, and would a different K or space get it?

The question this exists for
----------------------------
``tools/probe_prototype_coherence.py`` says M2's 53 prototypes have a median
within/between ratio of 0.947 against a shuffled null of 1.000, and that only a
handful sit below 0.85.  A reviewer reading that asked the obvious next thing:
is the effect small because the clustering is weak, or because wild dance data
does not support tighter grouping?  Those two have opposite consequences --
re-run M2 differently, or accept this and move on -- and the coherence table
alone cannot tell them apart.

The trap this avoids
--------------------
The tempting comparison is "K-Means fitted directly on the pose descriptors",
and ``probe_pose_space_tightness`` already reports it as a floor.  But that arm
**optimises the exact quantity being read**, so quoting it as the headroom
overstates it by an unknown amount.  It is a bound on the metric, not a bound on
what an honest method achieves.

So every fitted arm here is scored **out of sample**: K-Means is fitted on one
half of the uploads, the other half is assigned by nearest centroid, and the
coherence table is computed on the held-out half only.  A clustering that merely
memorised its training set now scores no better than one that found structure.
The split is by **upload**, not by segment: consecutive segments of one dancer
are near-duplicates, and a segment-level split would leak a member of almost
every group across the boundary.

The arms
--------
* ``m2`` -- M2's actual labels restricted to the held-out half. What we have.
* ``tmr_refit`` at each K -- K-Means on the **TMR embeddings** (what M2 clusters),
  fitted on the training half. Answers "is K the problem": same space, same
  method, different K, scored the same way.
* ``pose_refit`` -- K-Means on the **pose descriptors**, same protocol. Answers
  "is the TMR embedding the problem": if fitting in pose space transfers much
  better to held-out pose coherence, the embedding is losing what the descriptor
  keeps.
* ``shuffled`` -- size-matched permutation on the held-out half. The floor.

``pose_refit`` still has an edge -- it is fitted in the space it is scored in,
even if not on the same segments -- so read it as optimistic. ``tmr_refit`` at
K=53 versus ``m2`` is the clean comparison: identical space, identical K,
identical scoring, differing only in that one was fitted on half the data.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.probe_prototype_coherence import table  # noqa: E402


def summarise(rows: dict) -> dict:
    """Group-level summary, with unreadable groups counted rather than absorbed.

    A group whose members all come from one upload has no cross-upload pair and
    so no ratio.  ``np.median`` over an array containing one NaN returns NaN, and
    the first run of this probe duly printed ``nan`` for three of five K values --
    which is the good failure.  The bad one would have been ``nanmedian`` quietly
    reporting a median over whichever groups happened to be readable, because the
    unreadable ones are not a random subset: they are the smallest groups, and
    they get commoner as K rises, so the summary would improve with K for a
    reason that has nothing to do with clustering quality.

    So they are excluded *and counted*, and the count is part of the report.
    """
    raw = np.array([r["ratio"] for r in rows.values()], dtype=np.float64)
    sizes = np.array([r["n"] for r in rows.values()])
    readable = np.isfinite(raw)
    ratios = raw[readable]
    if not len(ratios):
        raise ValueError("no group had a cross-upload pair")
    return {
        "groups": int(len(raw)),
        "groups_readable": int(readable.sum()),
        "groups_single_upload": int((~readable).sum()),
        "median_ratio": float(np.median(ratios)),
        "mean_ratio": float(ratios.mean()),
        "q25_ratio": float(np.percentile(ratios, 25)),
        "n_at_or_above_1": int((ratios >= 1.0).sum()),
        "n_below_085": int((ratios < 0.85).sum()),
        "median_size": int(np.median(sizes)),
    }


def held_out_labels(fit_space: np.ndarray, score_index: np.ndarray,
                    train_index: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Fit K-Means on the training half, assign the held-out half to centroids."""
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(
        fit_space[train_index])
    return model.predict(fit_space[score_index])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=pathlib.Path,
                        default=pathlib.Path("/dev/shm/atomicdance-m2-review/"
                                             "segment_features.npz"))
    parser.add_argument("--embeddings", type=pathlib.Path,
                        default=pathlib.Path("/dev/shm/atomicdance-m2-review/"
                                             "clean5b5_tmr_embeddings.npz"))
    parser.add_argument("--k-sweep", type=int, nargs="+",
                        default=[20, 53, 100, 200, 375])
    parser.add_argument("--output", type=pathlib.Path,
                        default=REPO / "runs/clean5b5_m2_headroom.json")
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    data = np.load(args.features, allow_pickle=False)
    pose = data["vector"].astype(np.float64)
    uploads = data["upload"]
    m2 = data["label"]
    shuffled = data["shuffled"]

    rng = np.random.default_rng(args.seed)
    unique_uploads = np.unique(uploads)
    rng.shuffle(unique_uploads)
    train_uploads = set(unique_uploads[:len(unique_uploads) // 2].tolist())
    is_train = np.array([u in train_uploads for u in uploads])
    train_index = np.flatnonzero(is_train)
    score_index = np.flatnonzero(~is_train)
    print("{} segments: {} train / {} held out, split by upload "
          "({} / {} uploads)".format(
              len(pose), len(train_index), len(score_index),
              len(train_uploads), len(unique_uploads) - len(train_uploads)),
          flush=True)

    su, sp = uploads[score_index], pose[score_index]
    arms = {}

    arms["m2"] = summarise(table(sp, m2[score_index], su, args.seed))
    # A fresh size-preserving permutation, not the stored column: that column was
    # built size-matched for the *run* unit, and under --spans-from it is a
    # different partition of the same frames.
    perm = np.random.default_rng(args.seed).permutation(len(score_index))
    arms["shuffled"] = summarise(table(sp, m2[score_index][perm], su, args.seed))
    print("m2        {median_ratio:.4f}  ({n_below_085} of {groups_readable} below "
          ".85, {n_at_or_above_1} at/above 1.0)".format(**arms["m2"]), flush=True)
    print("shuffled  {median_ratio:.4f}".format(**arms["shuffled"]), flush=True)

    # pose_refit: optimistic (fitted in the space it is scored in), still out of sample
    for k in args.k_sweep:
        labels = held_out_labels(pose, score_index, train_index, k, args.seed)
        arms["pose_refit_k{}".format(k)] = summarise(table(sp, labels, su, args.seed))
        print("pose_refit K={:<4} {median_ratio:.4f}  ({n_below_085} of "
              "{groups_readable} below .85, {groups_single_upload} unreadable)".format(
                  k, **arms["pose_refit_k{}".format(k)]), flush=True)

    if args.embeddings.is_file():
        blob = np.load(args.embeddings, allow_pickle=True)
        # The cache holds every segment the beat grid produced (14,231), keyed by
        # (recording, start, end); the feature table holds only the 10,652 both
        # renderers draw. Joining on the span rather than trusting row order is
        # the point: the two files were built by different passes, and a silent
        # mis-join would score M2's labels against someone else's embeddings.
        index = {(str(r), int(s0), int(e0)): i for i, (r, s0, e0) in enumerate(
            zip(blob["recordings"], blob["starts"], blob["ends"]))}
        want = list(zip(data["recording"], data["start"], data["end"]))
        rows = [index.get((str(r), int(s0), int(e0))) for r, s0, e0 in want]
        missing = sum(1 for r in rows if r is None)
        if missing:
            print("tmr_refit SKIPPED: {} of {} segments have no embedding under "
                  "their (recording, start, end) key".format(missing, len(rows)),
                  flush=True)
        else:
            tmr = np.asarray(blob["embeddings"], dtype=np.float64)[np.array(rows)]
            print("tmr embeddings joined: {} on (recording, start, end), "
                  "0 missing".format(tmr.shape), flush=True)
            for k in args.k_sweep:
                labels = held_out_labels(tmr, score_index, train_index, k, args.seed)
                name = "tmr_refit_k{}".format(k)
                arms[name] = summarise(table(sp, labels, su, args.seed))
                print("tmr_refit  K={:<4} {median_ratio:.4f}  ({n_below_085} of "
                      "{groups_readable} below .85, {groups_single_upload} "
                      "unreadable)".format(k, **arms[name]), flush=True)
    else:
        print("no embeddings at {} -- skipping tmr_refit".format(args.embeddings),
              flush=True)

    report = {
        "protocol": "K-Means fitted on half the uploads, held-out half assigned by "
                    "nearest centroid, coherence scored on the held-out half only",
        "why_held_out": "a clustering fitted and scored on the same segments is graded "
                        "on what it optimised; the floor in probe_pose_space_tightness "
                        "is a bound on the metric, not on what an honest method reaches",
        "split": "by upload, not by segment: consecutive segments of one dancer are "
                 "near-duplicates and would leak across a segment-level split",
        "scored_in": "signature-pose space (canonical joints at up to 4 motion beats)",
        "pairs": "cross-upload only",
        "segments_scored": int(len(score_index)),
        "clean_comparison": "tmr_refit_k53 vs m2 -- same space, same K, same scoring",
        "caveat": "pose_refit is fitted in the space it is scored in, so it is "
                  "optimistic even out of sample",
        "arms": arms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nwrote {}".format(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
