#!/usr/bin/env python3
"""Across-clip novelty per arm: does every generated clip dance like the others?

NOT A NEW CRITERION.  The descriptor, the window length, the stride and the
normalisation are imported from ``tools/measure_motion_repetition`` (``windows``,
``load``) so this reads the same thing that tool reads; the ONLY change is where
the nearest neighbour is looked for -- in OTHER clips instead of far-away frames
of the same clip.  Low = the arm has a house style it replays regardless of the
song, which is the operator's "招单一".  This is the parameterised form of the
one-off scratchpad script that produced the 0.383-vs-0.458 pair quoted in
docs/DANCE_QUALITY_DEFECTS.md; it is a tool here so the guidance sweep's arms
are all read by the same code path.

CONTROL.  ``--shuffled-control`` re-runs the same computation with the clip
ownership labels permuted, which destroys the "which clip did this window come
from" structure while keeping the window bank identical.  The reading must move
toward the pooled scale (~1.0-ish) under that permutation; if it does not, the
number is not measuring across-clip identity and must not be quoted.
"""

import argparse
import json
import pathlib
import sys

import numpy as np
from scipy.spatial.distance import cdist

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.measure_motion_repetition import windows, load  # noqa: E402

LEN, STRIDE = 15, 5
SCALE_STRIDE = 7


def _bank(directory, clips):
    out = {}
    missing = []
    for clip in clips:
        p = pathlib.Path(directory) / (clip + ".pkl")
        if not p.exists():
            missing.append(clip)
            continue
        blocks, _ = windows(load(p), LEN, STRIDE)
        out[clip] = blocks
    return out, missing


def _cross(bank, owner_override=None):
    """Median nearest-neighbour distance to a DIFFERENT clip, over the pooled mean.

    THE SELF MASK IS NOT REDUNDANT, and leaving it out silently destroys the
    shuffled control.  In the real reading a window's own clip is excluded, so
    the window can never match itself.  Under ``owner_override`` the excluded
    group is a random one, the query's own row is usually still in the bank, and
    every query then finds itself at distance 0 -- the control read 0.0000 for
    ground truth and for every arm alike, which is not a control but a bug.
    Masking the query's own global row makes the override mean what it says:
    exclude a RANDOM group of the same size instead of the clip, keeping the
    window's own clip-mates available.  Those clip-mates are the nearest matches
    the real reading throws away, so the control must read LOWER than the real
    one; if it does not, clip identity is not what the column is measuring."""
    names = list(bank)
    allb = np.concatenate([bank[n] for n in names])
    owner = np.concatenate([[i] * len(bank[n]) for i, n in enumerate(names)])
    offsets = np.cumsum([0] + [len(bank[n]) for n in names])
    if owner_override is not None:
        owner = owner_override
    scale = float(cdist(allb[::SCALE_STRIDE], allb[::SCALE_STRIDE]).mean())
    per = {}
    for i, name in enumerate(names):
        d = cdist(bank[name], allb)
        d[:, owner == i] = np.inf
        rows = np.arange(len(bank[name]))
        d[rows, offsets[i] + rows] = np.inf
        per[name] = float(np.median(d.min(axis=1)) / scale)
    return per


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--clips", required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--shuffled-control", action="store_true")
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--out")
    args = ap.parse_args()

    clips = [l.strip() for l in pathlib.Path(args.clips).read_text().split() if l.strip()]
    arms = [("ground truth", args.ground_truth)]
    arms += [tuple(a.split("=", 1)) for a in args.arm]

    rows = {}
    for name, directory in arms:
        bank, missing = _bank(directory, clips)
        if not bank:
            rows[name] = {"error": "no clips found", "dir": directory}
            continue
        per = _cross(bank)
        row = {"clips": len(bank), "missing": missing,
               "novelty_median": float(np.median(list(per.values()))),
               "per_clip": {k: round(v, 4) for k, v in per.items()}}
        if args.shuffled_control:
            allb_n = sum(len(v) for v in bank.values())
            rng = np.random.default_rng(args.seed)
            owner = np.concatenate([[i] * len(bank[n]) for i, n in enumerate(bank)])
            shuffled = rng.permutation(owner)
            assert len(shuffled) == allb_n
            per_s = _cross(bank, owner_override=shuffled)
            row["novelty_median_shuffled_owner"] = float(np.median(list(per_s.values())))
        rows[name] = row

    gt = rows["ground truth"].get("per_clip", {})
    for name, row in rows.items():
        if name == "ground truth" or "per_clip" not in row:
            continue
        worse = sum(1 for c, v in row["per_clip"].items() if c in gt and v < gt[c])
        row["clips_more_house_style_than_truth"] = "%d/%d" % (worse, len(gt))

    out = {"options": {"window_frames": LEN, "stride": STRIDE, "seed": args.seed},
           "arms": rows}
    text = json.dumps(out, indent=1, sort_keys=True)
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(text)
    for name, row in rows.items():
        print("%-22s novelty %.4f  %s" % (
            name, row.get("novelty_median", float("nan")),
            row.get("clips_more_house_style_than_truth", "")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
