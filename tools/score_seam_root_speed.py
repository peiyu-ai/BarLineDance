#!/usr/bin/env python3
"""Does the body brake at the bar lines?  Root speed against distance to the nearest seam.

THE COMPLAINT.  The operator, 2026-09-16, on output/sample_20260916_fix2: "蒙皮上偶发
的位置跳变 ... 视觉上偶尔就会有顿挫感,缺流畅连续" -- occasional jolts in the skinned
render.  The render only draws the root it is given, so the first question is
whether the ROOT itself stutters, and where.

THE MEASUREMENT.  Horizontal pelvis speed per frame, divided by that clip's own
median speed so fast and slow dancers pool, binned by distance to the nearest
bar seam (``prototype_retrieval.plan_postprocess.bar_bounds`` of a reference
arm).  Two summary columns:

    brake   = mean speed at distances 0-1 / mean speed at distances 13-14
    lurch   = max over distances 3-8 / mean speed at distances 13-14

A body that brakes into every bar line and lurches out of it reads brake << 1
and lurch > 1.

THE NULL IS GROUND TRUTH AT THE SAME FRAME INDICES.  Ground truth has no seams
there, so any structure it shows at those indices is the music's, not the
pipeline's; that is what makes a low ``brake`` a statement about the draft.
Measured 2026-09-16 on the ten vis clips: ground truth brake 1.19, fix2 0.13,
lower on 10/10 clips, P=0.002, and fix2 and its draft identical frame for frame
(the completion holds the root to the draft).

A fix is not allowed to buy a good ``brake`` by slowing the whole root down, so
the clip median speed and the root path are printed beside it.

HEIGHT, AND WHY IT IS THE COLUMN THE OPERATOR SEES.  The strip renders with
--lock-root by default, which replaces every generated row's HORIZONTAL root with
ground truth's and leaves height alone.  So ``brake``/``lurch`` describe a defect
that is real in the data and invisible in the default view, while a pelvis
HEIGHT step at a seam is exactly what shows up as a "位置跳变".  ``--height``
prints |dz| across the seam frame, split into seams at a label change and seams
inside a label run (build_draft chains bar units, and only label changes were
ever blended), with ground truth at the same indices as the null.
"""
import argparse
import math
import pathlib
import pickle
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.score_beat_phase_profile import joints_of  # noqa: E402

FPS = 30.0
MAX_DISTANCE = 15


def two_sided(differences):
    d = [x for x in differences if x == x and x != 0.0]
    n = len(d)
    wins = sum(1 for x in d if x > 0)
    k = min(wins, n - wins)
    p = min(1.0, 2 * sum(math.comb(n, j) for j in range(0, k + 1)) / 2 ** n) if n else 1.0
    return wins, n, p


def load(directory, clip):
    path = pathlib.Path(directory) / (clip + ".pkl")
    if not path.is_file():
        return None
    try:
        with open(path, "rb") as handle:
            record = pickle.load(handle)
        return np.asarray(record["full_pose"], np.float64)
    except (KeyError, pickle.UnpicklingError):
        return joints_of(str(path))


def profile(joints, seams):
    speed = np.linalg.norm(np.diff(joints[:, 0, :2], axis=0), axis=1) * FPS
    median = float(np.median(speed)) or 1e-9
    table = {}
    for distance in range(MAX_DISTANCE + 1):
        values = []
        for seam in seams:
            for sign in ((1, -1) if distance else (1,)):
                index = seam + sign * distance
                if 0 <= index < len(speed):
                    values.append(speed[index] / median)
        table[distance] = float(np.mean(values)) if values else float("nan")
    far = np.nanmean([table[13], table[14]])
    brake = np.nanmean([table[0], table[1]]) / far
    lurch = max(table[d] for d in range(3, 9)) / far
    path = float(np.linalg.norm(np.diff(joints[:, 0, :2], axis=0), axis=1).sum())
    return table, brake, lurch, median, path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", default="runs/vis_clips_t10.txt")
    ap.add_argument("--truth", default="runs/txy_t_gt_eval/motion")
    ap.add_argument("--seams-from", required=True,
                    help="arm whose bar_bounds define the seams (all arms must share the plan)")
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--height", action="store_true",
                    help="also print the pelvis height step across each seam, by seam type")
    args = ap.parse_args()

    clips = [c.strip() for c in pathlib.Path(args.clips).read_text().split() if c.strip()]
    arms = [("ground truth", args.truth)] + [tuple(s.split("=", 1)) for s in args.arm]
    rows = {name: [] for name, _ in arms}
    curves = {name: [] for name, _ in arms}
    for clip in clips:
        ref = pathlib.Path(args.seams_from) / (clip + ".pkl")
        if not ref.is_file():
            continue
        with open(ref, "rb") as handle:
            seams = pickle.load(handle)["prototype_retrieval"]["plan_postprocess"]["bar_bounds"][1:-1]
        loaded = {name: load(d, clip) for name, d in arms}
        if any(v is None for v in loaded.values()):
            continue
        for name, joints in loaded.items():
            table, brake, lurch, median, path = profile(joints, seams)
            rows[name].append((brake, lurch, median, path))
            curves[name].append([table[d] for d in range(MAX_DISTANCE + 1)])

    n = len(rows["ground truth"])
    print("{} clips; root speed / clip median, by distance to the nearest bar seam".format(n))
    print("{:<22}".format("distance") + "".join("{:>6}".format(d) for d in range(MAX_DISTANCE + 1)))
    for name in rows:
        mean = np.nanmean(np.array(curves[name]), axis=0)
        print("{:<22}".format(name[:22]) + "".join("{:6.2f}".format(v) for v in mean))
    print("\n{:<22}{:>8}{:>8}{:>14}{:>12}".format("arm", "brake", "lurch", "median m/s", "path m"))
    for name in rows:
        r = np.array(rows[name])
        print("{:<22}{:8.2f}{:8.2f}{:14.3f}{:12.2f}".format(name[:22], *r.mean(axis=0)))
    print("\npaired against ground truth, two-sided sign test")
    truth = np.array(rows["ground truth"])
    for name in list(rows)[1:]:
        r = np.array(rows[name])
        for column, label in ((0, "brake"), (1, "lurch"), (2, "median"), (3, "path")):
            w, m, p = two_sided(list(r[:, column] - truth[:, column]))
            print("  {:<20} {:<7} {:+8.3f}  higher on {}/{}  P={:.4f}".format(
                name[:20], label, float(np.mean(r[:, column] - truth[:, column])), w, m, p))
    if args.height:
        height_report(args, clips, arms)


def height_steps(joints, seams, label_starts):
    z = joints[:, 0, 2]
    in_run, at_change = [], []
    for seam in seams:
        if 1 <= seam < len(z):
            (at_change if seam in label_starts else in_run).append(abs(float(z[seam] - z[seam - 1])))
    return in_run, at_change


def height_report(args, clips, arms):
    import torch
    from dataset.atomic import labels_to_segments
    table = {name: ([], []) for name, _ in arms}
    for clip in clips:
        ref = pathlib.Path(args.seams_from) / (clip + ".pkl")
        if not ref.is_file():
            continue
        with open(ref, "rb") as handle:
            record = pickle.load(handle)
        seams = record["prototype_retrieval"]["plan_postprocess"]["bar_bounds"][1:-1]
        labels = torch.as_tensor(np.asarray(record["atomic_labels"]))
        label_starts = {segment.start for segment in labels_to_segments(labels)}
        for name, directory in arms:
            joints = load(directory, clip)
            if joints is None:
                continue
            a, b = height_steps(joints, seams, label_starts)
            table[name][0].extend(a)
            table[name][1].extend(b)
    print("\npelvis HEIGHT step across the seam frame, cm (what a root-locked strip shows)")
    print("{:<22}{:>30}{:>30}".format("arm", "inside a label run: med/p90/max", "at a label change: med/p90/max"))
    for name, (a, b) in table.items():
        fmt = lambda v: "n={:<3} {:5.1f} {:5.1f} {:5.1f}".format(
            len(v), *(np.array([np.median(v), np.percentile(v, 90), np.max(v)]) * 100)) if v else "n=0"
        print("{:<22}{:>30}{:>30}".format(name[:22], fmt(a), fmt(b)))


if __name__ == "__main__":
    main()
