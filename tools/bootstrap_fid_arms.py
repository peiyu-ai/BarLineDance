#!/usr/bin/env python3
"""Does M6 have the power to rank the arms it is used to rank?

M6 reports FID as a point estimate.  FID_k is a 72-dimensional statistic and
step 3b computes it from 92 generated sequences, so its covariance term is a
72x72 matrix estimated from 92 points.  Measured 2026-08-23 by resampling:

    leak-free (n=92)   fid_k 6.486   95% CI [6.44, 13.12]   width = 103% of the point
    all       (n=260)  fid_k 9.760   95% CI [8.95, 12.65]   width =  38%

An unpaired interval that wide cannot rank anything.  But the arms are *paired*
-- same clips, same seeds, same ground-truth set, only the rule differs -- so
the right statistic is the paired difference under a clip-level bootstrap:
resample clip identities, recompute both arms on that same resample, and read
the distribution of the difference.  Clips rather than sequences, because the
four seeds of one clip are not independent draws of a dance.

This is a gate on the *instrument*: if zero is inside the paired interval, M6
cannot separate the two arms and no ordering may be reported from it.

Usage::

    python3 tools/bootstrap_fid_arms.py --arms runs/m6_a runs/m6_b \\
        --subset leak-free --resamples 500
"""

from __future__ import annotations

import argparse
import collections
import glob
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from eval.metrics import calc_fid, normalize  # noqa: E402

SEED_SUFFIX = re.compile(r"_seed\d+$|_s\d+$")


def clip_of(stem: str) -> str:
    """Strip the seed suffix the feature extractor adds, leaving the clip id."""
    return SEED_SUFFIX.sub("", stem)


def load(directory):
    paths = sorted(glob.glob(str(pathlib.Path(directory) / "*.npy")))
    if not paths:
        raise SystemExit("error: no features in {}".format(directory))
    stems = [pathlib.Path(p).stem for p in paths]
    return np.stack([np.load(p) for p in paths]), stems


def by_clip(features, stems):
    groups = collections.OrderedDict()
    for row, stem in zip(features, stems):
        groups.setdefault(clip_of(stem), []).append(row)
    return groups


def fid_for(groups, order, ground_truth):
    rows = [row for clip in order for row in groups[clip]]
    generated = np.asarray(rows, dtype=np.float64)
    gt, gen = normalize(ground_truth, generated)
    return calc_fid(gen, gt)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", nargs="+", required=True,
                        help="M6 output directories, first one is the reference")
    parser.add_argument("--subset", choices=("all", "leak-free"), default="leak-free")
    parser.add_argument("--feature", default="kinetic_features")
    parser.add_argument("--ground-truth", default=None,
                        help="ground-truth feature dir; defaults to the arm's own "
                             "clean_gt for leak-free")
    parser.add_argument("--resamples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args(argv)

    sub = "clean_pred" if args.subset == "leak-free" else "features"
    per_arm, orders = {}, None
    for arm in args.arms:
        features, stems = load(pathlib.Path(arm) / sub / args.feature)
        groups = by_clip(features, stems)
        per_arm[arm] = groups
        clips = set(groups)
        orders = clips if orders is None else (orders & clips)
    order = sorted(orders)
    if not order:
        raise SystemExit("error: the arms share no clips")

    gt_dir = args.ground_truth or (
        str(pathlib.Path(args.arms[0]) / "clean_gt" / args.feature)
        if args.subset == "leak-free"
        else "runs/wild_v4_acct_gt_features/" + args.feature)
    ground_truth, _ = load(gt_dir)

    print("clips {}   sequences/arm {}   feature dim {}   subset {}".format(
        len(order), sum(len(per_arm[args.arms[0]][c]) for c in order),
        ground_truth.shape[1], args.subset))
    point = {arm: fid_for(per_arm[arm], order, ground_truth) for arm in args.arms}
    for arm in args.arms:
        print("  {:34s} fid = {:8.4f}".format(pathlib.Path(arm).name, point[arm]))

    rng = np.random.default_rng(args.seed)
    reference = args.arms[0]
    deltas = {arm: [] for arm in args.arms[1:]}
    for _ in range(args.resamples):
        draw = [order[i] for i in rng.integers(0, len(order), len(order))]
        try:
            base = fid_for(per_arm[reference], draw, ground_truth)
            for arm in args.arms[1:]:
                deltas[arm].append(fid_for(per_arm[arm], draw, ground_truth) - base)
        except Exception:
            continue

    print("\npaired difference against {} (clip-level bootstrap, {} resamples)".format(
        pathlib.Path(reference).name, args.resamples))
    for arm in args.arms[1:]:
        d = np.asarray(deltas[arm])
        lo, hi = np.percentile(d, [2.5, 97.5])
        separated = lo > 0 or hi < 0
        print("  {:34s} delta = {:+8.4f}   95% CI [{:+8.4f}, {:+8.4f}]   {}".format(
            pathlib.Path(arm).name, point[arm] - point[reference], lo, hi,
            "SEPARATED" if separated else "NOT separated (zero is inside)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
