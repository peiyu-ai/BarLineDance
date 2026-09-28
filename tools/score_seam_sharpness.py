"""WHERE the body's sharpest moments are, and how big they are.

THE DEFECT THIS MEASURES, in the operator's words (2026-09-13):
"整体均匀变钝…卡旋律和节奏型的动作不够 sharp".  Measured, that is NOT a lack of
sharpness -- it is sharpness in the wrong place:

    top-10% |delta speed| mean   ground truth 0.01237, our output 0.01507 (122%),
                                 the retrieval draft 0.02833 (229%), and the
                                 draft before seam blending 0.05488 (444%)
    share of those at a bar line ground truth 13.7% (1.17x chance, i.e. none),
                                 our output 21.4% (1.82x), the draft 40.0% (3.41x)

So our biggest movements of the body are the JOINS, not the dancing.  On screen a
join reads as a glitch rather than as an accent, and the smooth stretch between
two joins reads as floaty -- which is the "uniformly duller" impression.

WHY NOT COUNT ACCENTS.  ``render_plan_strip.accents`` thresholds at each clip's
OWN 90th percentile, so the draft's seam spikes raise the bar and hide the
ordinary accents beneath it: that column said we produce 9% FEWER accents than
the dancer (P=0.0013) and the whole effect vanished at a common absolute
threshold (ours 3.039/s against ground truth's 2.928).  Any cross-arm accent
count has to fix the threshold first, or it measures the seams.

A THREE-SIDED TARGET, so no one-sided column can buy it:
  * peak    top-10% |delta speed| mean   -> DOWN toward 0.01237
  * count   changes over a FIXED absolute threshold -> must not fall (GT 2.928/s)
  * jitter  shake band share -> must not rise (T-line ground truth 0.0509)
An arm that smooths everything wins the first and loses the other two.
"""
import argparse
import json
import pathlib
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from infer_atomic import bar_grid_bounds  # noqa: E402
from tools.score_arm_table import jitter_share  # noqa: E402

FPS = 30.0
SEAM_WINDOW = 3          # frames either side of a bar line
TOP_FRACTION = 0.10


def speed_change(joints):
    joints = np.asarray(joints, float)
    speed = np.linalg.norm(np.diff(joints, axis=0), axis=2).mean(1)
    return np.abs(np.diff(speed))


def clip_reading(joints, seams, threshold):
    change = speed_change(joints)
    if len(change) < 10:
        return None
    top = change >= np.percentile(change, 100 * (1 - TOP_FRACTION))
    index = np.flatnonzero(top)
    at_seam = float("nan")
    if len(seams) and len(index):
        hits = sum(1 for i in index
                   if np.min(np.abs(np.asarray(seams) - i)) <= SEAM_WINDOW)
        at_seam = hits / len(index)
    return {"peak": float(change[top].mean()),
            "count": float((change >= threshold).sum() / (len(change) / FPS)),
            "at_seam": at_seam,
            "chance": float(min(1.0, len(seams) * (2 * SEAM_WINDOW + 1) / len(change))),
            "jitter": float(jitter_share(joints))}


def load(path):
    return np.asarray(pickle.load(open(path, "rb"))["full_pose"], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--audio-dir", default="runs/txy_t_gt_eval/audio")
    ap.add_argument("--ground-truth-dir", default="runs/txy_t_gt_eval/motion")
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=DIR")
    ap.add_argument("--json")
    args = ap.parse_args()

    clips = [c.strip() for c in open(args.clips) if c.strip()]
    seams, truth = {}, {}
    pooled = []
    for clip in clips:
        gt = pathlib.Path(args.ground_truth_dir) / (clip + ".pkl")
        audio = pathlib.Path(args.audio_dir) / (clip + ".npy")
        if not gt.is_file() or not audio.is_file():
            continue
        joints = load(gt)
        music = np.load(audio)
        bounds, _ = bar_grid_bounds(torch.as_tensor(music[:len(joints)]), 4,
                                    length=len(joints))
        truth[clip] = joints
        seams[clip] = ([int(b) for b in list(bounds)[1:-1]] if bounds is not None else [])
        pooled.append(speed_change(joints))
    if not truth:
        raise SystemExit("no clip had both ground-truth motion and audio")

    # THE THRESHOLD IS GROUND TRUTH'S OWN, pooled, and fixed for every arm --
    # the whole point of this file is that a per-clip threshold measures seams.
    threshold = float(np.percentile(np.concatenate(pooled), 100 * (1 - TOP_FRACTION)))

    rows = {"ground truth": {c: clip_reading(truth[c], seams[c], threshold)
                             for c in truth}}
    for spec in args.arm:
        name, _, directory = spec.rpartition("=")
        row = {}
        for clip in truth:
            path = pathlib.Path(directory) / (clip + ".pkl")
            if path.is_file():
                row[clip] = clip_reading(load(path), seams[clip], threshold)
        rows[name] = row

    gt_row = rows["ground truth"]
    target = {k: float(np.nanmean([v[k] for v in gt_row.values() if v]))
              for k in ("peak", "count", "at_seam", "jitter")}
    print("absolute threshold (ground truth's own pooled p90): {:.5f}".format(threshold))
    print("target: peak DOWN to {peak:.5f}, count NOT below {count:.3f}/s, "
          "jitter NOT above {jitter:.4f}".format(**target))
    print("\n{:<24}{:>5}{:>10}{:>11}{:>12}{:>10}{:>9}".format(
        "arm", "n", "peak", "vs GT", "count/s", "at seam", "jitter"))
    for name, row in rows.items():
        values = [v for v in row.values() if v]
        if not values:
            continue
        peak = float(np.mean([v["peak"] for v in values]))
        print("{:<24}{:>5}{:>10.5f}{:>10.0f}%{:>12.3f}{:>9.1%}{:>9.4f}".format(
            name, len(values), peak, 100 * peak / target["peak"],
            float(np.mean([v["count"] for v in values])),
            float(np.nanmean([v["at_seam"] for v in values])),
            float(np.nanmean([v["jitter"] for v in values]))))
    chance = float(np.nanmean([v["chance"] for v in gt_row.values() if v]))
    print("\n'at seam' is the share of the top-10% changes within {} frames of a "
          "bar line; {:.1%} is chance. Ground truth sits at chance because it "
          "has no seams.".format(SEAM_WINDOW, chance))

    print("\nPASS needs all three: peak <= {:.5f}, count >= {:.3f}, jitter <= {:.4f}"
          .format(target["peak"], target["count"], target["jitter"]))
    for name, row in rows.items():
        if name == "ground truth":
            continue
        values = [v for v in row.values() if v]
        if not values:
            continue
        peak = float(np.mean([v["peak"] for v in values]))
        count = float(np.mean([v["count"] for v in values]))
        jit = float(np.nanmean([v["jitter"] for v in values]))
        verdict = ("PASS" if peak <= target["peak"] and count >= target["count"]
                   and jit <= target["jitter"] else "fail")
        why = ", ".join(w for w, bad in (
            ("peak too big", peak > target["peak"]),
            ("count too low", count < target["count"]),
            ("too shaky", jit > target["jitter"])) if bad)
        print("  {:<22}{}{}".format(name, verdict, " -- " + why if why else ""))

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(
            {"threshold": threshold, "target": target, "rows": rows}, indent=2))
        print("wrote", args.json)


if __name__ == "__main__":
    main()
