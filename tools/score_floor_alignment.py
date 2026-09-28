"""Where the body stands, per clip, against the ground truth of the same clip.

WHAT THIS IS FOR.  2026-09-14 the operator reported "z too high and too low, the
feet are not on the floor" and asked whether extraction or completion was at
fault.  Neither pooled column answers it: measured on 20 clips, the generated
arm's median foot height (+0.085 m) and its fraction of frames more than 15 cm
up (28.4%) both MATCH ground truth (+0.088 m, 28.5%) -- while the per-clip
correlation between the two is -0.10 (P=0.67).  The arm produces the right
DISTRIBUTION of hovering on the wrong clips, and the pooled numbers are the
average of errors that cancel.  So every column here is a PER-CLIP PAIRED
difference against that clip's own ground truth, never a pooled mean.

THE COLUMNS, and what each one can and cannot say:

  floor_err   the arm's own floor (5th percentile of the lowest foot joint,
              the rule ``anchor_floor`` and ``render_avatar_video.floor_of``
              both use) minus ground truth's.  This is what ``--floor-anchor``
              controls.  It was already fine before this tool existed -- max
              6.1 cm, 0/20 over 10 cm -- so a change here is a REGRESSION
              signal, not a win condition.
  hover       fraction of frames whose lowest foot is more than 15 cm above the
              clip's OWN floor, minus ground truth's.  Ground truth hovers too
              (28.5% of its frames), so the target is ZERO, not a low number:
              driving this negative means a dancer who never leaves the ground.
  drift       range of the one-second rolling lowest foot, minus ground truth's.
              This is the accumulation that retrieval stitching produces --
              ground truth's own is 0.247 m median, so again the target is zero.
  pelvis      median pelvis height above the clip's own floor, minus ground
              truth's.  Reported because the hover is carried by the ROOT and
              not by the legs: corr(d_foot, d_pelvis) = +0.966 against
              corr(d_foot, -d_leg) = +0.364, so a fix that moves ``hover``
              without moving ``pelvis`` is doing something else.

Judgement is the sign test on the paired differences, because n = 20 is the
whole test split and cannot rank arms on means alone.
"""
import argparse
import pathlib
import pickle

import numpy as np
from scipy import stats

FOOT_JOINTS = [7, 8, 10, 11]       # SMPL ankles and feet
PELVIS = 0
FLOOR_PERCENTILE = 5
HOVER_METRES = 0.15
FPS = 30


def _rolling(values, window=FPS):
    if len(values) < window:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")[:len(values)]


def read(path):
    joints = np.asarray(pickle.load(open(path, "rb"))["full_pose"], dtype=np.float64)
    lowest = joints[:, FOOT_JOINTS, 2].min(axis=1)
    pelvis = joints[:, PELVIS, 2]
    floor = float(np.percentile(lowest, FLOOR_PERCENTILE))
    rolled = _rolling(lowest)
    return {
        "floor": floor,
        "hover": float(np.mean(lowest - floor > HOVER_METRES)),
        "drift": float(rolled.max() - rolled.min()),
        "pelvis": float(np.median(pelvis - floor)),
        "foot": float(np.median(lowest - floor)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", default="runs/txy_t_gt_eval/motion")
    ap.add_argument("--arm", action="append", required=True,
                    metavar="NAME=DIR", help="repeatable")
    ap.add_argument("--clips", default=None,
                    help="optional list; default is every pickle both sides have")
    args = ap.parse_args()

    truth = pathlib.Path(args.truth)
    arms = {}
    for entry in args.arm:
        name, _, directory = entry.partition("=")
        arms[name] = pathlib.Path(directory)

    names = sorted(p.name for p in truth.glob("wild_v5:*.pkl"))
    if args.clips:
        wanted = {line.strip() + ".pkl" for line in open(args.clips) if line.strip()}
        names = [n for n in names if n in wanted]
    for name, directory in arms.items():
        names = [n for n in names if (directory / n).is_file()]
    if not names:
        raise SystemExit("no clip is present in the truth and every arm")

    gt = {n: read(truth / n) for n in names}
    print("{} clips, ground truth from {}".format(len(names), truth))
    print("  ground truth itself: floor {:.3f}-{:.3f} m, hover {:.1%}, "
          "drift {:.3f} m (medians)".format(
              min(g["floor"] for g in gt.values()),
              max(g["floor"] for g in gt.values()),
              float(np.median([g["hover"] for g in gt.values()])),
              float(np.median([g["drift"] for g in gt.values()]))))

    for name, directory in arms.items():
        arm = {n: read(directory / n) for n in names}
        print("\n=== {} ({})".format(name, directory))
        for column in ("floor", "hover", "drift", "pelvis", "foot"):
            if column == "floor":
                diff = np.array([arm[n]["floor"] - gt[n]["floor"] for n in names])
                over = int((np.abs(diff) > 0.10).sum())
                print("  floor_err  median {:+.3f} m  max|d| {:.3f}  >10cm {}/{}"
                      .format(float(np.median(diff)), float(np.max(np.abs(diff))),
                              over, len(names)))
                continue
            diff = np.array([arm[n][column] - gt[n][column] for n in names])
            positive = int((diff > 0).sum())
            p = stats.binomtest(positive, len(diff), 0.5).pvalue
            unit = "" if column == "hover" else " m"
            fmt = "{:+.1%}" if column == "hover" else "{:+.3f}"
            print(("  {:9s} paired median " + fmt + "{}  mean|d| " +
                   ("{:.1%}" if column == "hover" else "{:.3f}") +
                   "  {}/{} positive  sign P={:.3f}").format(
                column, float(np.median(diff)), unit, float(np.mean(np.abs(diff))),
                positive, len(diff), p))
            if column in ("hover", "drift"):
                arm_values = [arm[n][column] for n in names]
                gt_values = [gt[n][column] for n in names]
                # The correlation is the column that separates "hovers as much
                # as ground truth" from "hovers on the SAME CLIPS", so it is
                # reported rather than computed silently -- and it needs three
                # clips and some spread, which a smoke test does not have.
                if len(names) < 3 or np.std(gt_values) == 0 or np.std(arm_values) == 0:
                    print("             per-clip corr: not computable on {} "
                          "clip(s) with this spread".format(len(names)))
                else:
                    r, pr = stats.pearsonr(gt_values, arm_values)
                    print("             per-clip corr with ground truth r={:+.3f} "
                          "(P={:.3f})".format(r, pr))
        worst = sorted(names, key=lambda n: -(arm[n]["drift"] - gt[n]["drift"]))[:3]
        print("  worst three by drift:")
        for n in worst:
            print("    {:22s} gt {:.3f}  arm {:.3f}  ({:+.3f})".format(
                n.split(":")[1][-6:] + ":" + n.split(":")[2].replace(".pkl", ""),
                gt[n]["drift"], arm[n]["drift"], arm[n]["drift"] - gt[n]["drift"]))


if __name__ == "__main__":
    main()
