#!/usr/bin/env python3
"""Per-prototype: are its members closer to each other than to the rest of the corpus?

RETRACTED READING (2026-08-21) -- read this before quoting any number below
--------------------------------------------------------------------------
This tool scores prototypes in **signature-pose space**, and M2 clusters **TMR
embeddings**.  The table it produced was used to tell an operator that 12 of 53
prototypes "buy nothing" and that prototype 8 is a junk drawer.  Scored in the
space M2 actually optimises, all 12 flip: median 0.7703, **0** prototypes at or
above 1.0, all 53 below 0.85 -- and prototype 8 reads 0.748, tighter than the
median.  The two rankings are uncorrelated (Spearman rho = -0.174, p = 0.21,
n = 53), so the pose ranking carried no information about the TMR one.

What was missing is CLAUDE.md 2.1 point 2: a criterion needs a positive control
**in the space it will judge in**.  Both nulls sit at 1.00, so the reading
looked calibrated; what was never measured is what a *good* clustering scores
in pose space.  See ``tools/probe_coherence_two_spaces.py``, which reports both
columns and refuses to emit a verdict, and the 2026-08-21 worklog entry.

**What this tool still measures, stated narrowly:** whether a grouping
corresponds to similar signature poses.  That is a fair question for a grouping
formed somewhere *other* than pose space -- M3's caption-formed sub-prototypes,
for instance -- and it is not a verdict on M2.

The paragraphs below are the original text, kept so the overturned claim and
the evidence against it are both readable.

Why a per-prototype table and not one corpus number
---------------------------------------------------
``tools/probe_pose_space_tightness.py`` answers "does this clustering carry pose
structure at all" and returns a single figure (M2 recovers ~36% of the distance
from shuffled labels to a clustering fitted on pose directly).  That number is
an average, and on 2026-08-21 a reviewer looking at contact sheets asked the
question the average hides: *some prototypes look coherent and some clearly do
not -- which is which?*

An average cannot answer that, and worse, it hides the case that matters most.
Prototype 8 has mean within-group distance 14.463 against a corpus mean of
6.936: it is twice as scattered as the typical prototype, and it contributes to
the corpus average as one of 53 while containing 42 of the corpus's most
peripheral segments.

The statistic
-------------
For each prototype: mean distance among its own members, divided by mean
distance from its members to a fixed random sample of non-members.  Both terms
count **cross-upload pairs only** -- consecutive segments of one dancer resemble
each other for reasons unrelated to the vocabulary, and these prototypes already
concentrate accounts (lift 1.274), so same-upload pairs would pay a prototype
for capturing identity.

*Ratio below 1* means members are closer to each other than to the corpus at
large: the thing a cluster is supposed to be.
*Ratio at or above 1* means membership buys nothing -- the group is no tighter
than an arbitrary set of the same size.

The ratio is used rather than the raw within-distance because the two are not
interchangeable: a prototype sitting in a sparse region of pose space has a
large within-distance *and* a large to-rest distance, and calling it incoherent
on the first number alone would confuse "unusual" with "incoherent".  Prototype
8 is incoherent on both readings, which is why it is worth naming; a prototype
that were merely unusual would separate the two.

Why this can fail
-----------------
K-Means minimises within-cluster distance in the **TMR embedding**.  These are
canonical joint positions at motion beats -- a different space, never optimised.
A clustering that found nothing transferable would read ~1.0 everywhere.  The
size-matched shuffled arm is carried through as the explicit null, so each
prototype's ratio is reported next to what a same-sized random group achieves.

That null is necessary and was never the problem.  It bounds the *floor* --
what an arbitrary group of the same size scores -- and says nothing about the
*ceiling*, which is what a clustering built elsewhere can reach when read from
here.  A reading of 0.95 is far from the 1.00 floor and may still be the best
any foreign clustering achieves in this space; without the ceiling, "0.95" is
not evidence of anything.  This is why the retraction above exists.

What it does not answer
-----------------------
Whether a coherent prototype is a *movement*.  A tight group of similar static
shapes reads exactly as well.  That still needs the video sheet and a person.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SAMPLE = 1500          # non-members drawn per prototype; fixed for comparability


def cross_upload_mean(a: np.ndarray, b: np.ndarray, ua: np.ndarray, ub: np.ndarray,
                      *, same_set: bool) -> float:
    gram = a @ b.T
    da, db = (a * a).sum(1), (b * b).sum(1)
    distance = np.sqrt(np.maximum(da[:, None] + db[None, :] - 2.0 * gram, 0.0))
    cross = ua[:, None] != ub[None, :]
    if same_set:
        cross &= ~np.eye(len(a), dtype=bool)
    return float(distance[cross].mean()) if cross.any() else float("nan")


def table(vectors: np.ndarray, labels: np.ndarray, uploads: np.ndarray,
          seed: int) -> dict:
    rng = np.random.default_rng(seed)
    rows = {}
    for prototype in np.unique(labels):
        inside = np.flatnonzero(labels == prototype)
        outside = np.flatnonzero(labels != prototype)
        drawn = rng.choice(outside, size=min(SAMPLE, len(outside)), replace=False)
        within = cross_upload_mean(vectors[inside], vectors[inside],
                                   uploads[inside], uploads[inside], same_set=True)
        to_rest = cross_upload_mean(vectors[inside], vectors[drawn],
                                    uploads[inside], uploads[drawn], same_set=False)
        rows[int(prototype)] = {
            "n": int(len(inside)),
            "uploads": int(len(set(uploads[inside].tolist()))),
            "within": within,
            "to_rest": to_rest,
            "ratio": within / to_rest if to_rest else float("nan"),
        }
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=pathlib.Path,
                        default=pathlib.Path("/dev/shm/atomicdance-m2-review/"
                                             "segment_features.npz"))
    parser.add_argument("--output", type=pathlib.Path,
                        default=REPO / "runs/clean5b5_prototype_coherence.json")
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--null", choices=("stored", "permute"), default="stored",
                        help="'stored' uses the shuffled label column, which is "
                             "size-matched only for the run unit it was built for. "
                             "'permute' draws a fresh size-preserving permutation of "
                             "the real labels -- required when the features were "
                             "dumped with --spans-from, where the stored column is a "
                             "different partition of the same frames.")
    args = parser.parse_args()

    data = np.load(args.features, allow_pickle=False)
    vectors = data["vector"].astype(np.float64)
    uploads = data["upload"]

    real = table(vectors, data["label"], uploads, args.seed)
    if args.null == "permute":
        rng = np.random.default_rng(args.seed)
        null_labels = data["label"][rng.permutation(len(data["label"]))]
    else:
        null_labels = data["shuffled"]
    null = table(vectors, null_labels, uploads, args.seed)

    # The null is size-matched label-for-label, so a prototype's own id indexes a
    # random group of the same size -- the comparison each row needs.
    for key, row in real.items():
        row["null_ratio"] = null[key]["ratio"]
        row["null_within"] = null[key]["within"]
        row["margin"] = row["null_ratio"] - row["ratio"]

    ordered = sorted(real.items(), key=lambda kv: kv[1]["ratio"])
    ratios = np.array([r["ratio"] for _, r in ordered])
    null_ratios = np.array([r["null_ratio"] for _, r in ordered])

    report = {
        "statistic": "mean cross-upload distance among members / same to a fixed "
                     "random sample of non-members, in signature-pose space",
        "reading": "below 1.0 = members closer to each other than to the corpus. "
                   "at or above 1.0 = membership buys nothing.",
        "space": "canonical joint positions at up to 4 motion beats, 264 dims -- "
                 "M2 clustered TMR embeddings, not these",
        "pairs": "cross-upload only",
        "non_member_sample": SAMPLE,
        "null_construction": args.null,
        "unit": "label runs" if len(data["label"]) == 10652 else "embedding spans",
        "prototypes": len(real),
        "real_ratio_median": float(np.median(ratios)),
        "null_ratio_median": float(np.median(null_ratios)),
        "n_ratio_at_or_above_1": int((ratios >= 1.0).sum()),
        "n_null_at_or_above_1": int((null_ratios >= 1.0).sum()),
        "incoherent": [int(k) for k, r in ordered if r["ratio"] >= 1.0],
        "tightest": [int(k) for k, _ in ordered[:8]],
        "per_prototype": {str(k): v for k, v in ordered},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("{:>6} {:>5} {:>8} {:>8} {:>7} {:>7}".format(
        "proto", "n", "within", "to-rest", "ratio", "null"))
    for key, row in ordered:
        flag = "  <-- buys nothing" if row["ratio"] >= 1.0 else ""
        print("{:>6} {:>5} {:>8.3f} {:>8.3f} {:>7.3f} {:>7.3f}{}".format(
            key, row["n"], row["within"], row["to_rest"], row["ratio"],
            row["null_ratio"], flag))
    print()
    print("real median ratio {:.3f} vs null {:.3f}; {} of {} prototypes at or above "
          "1.0 (null: {})".format(
              report["real_ratio_median"], report["null_ratio_median"],
              report["n_ratio_at_or_above_1"], report["prototypes"],
              report["n_null_at_or_above_1"]), flush=True)
    print("wrote {}".format(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
