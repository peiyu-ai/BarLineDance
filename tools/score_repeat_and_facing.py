"""Two defects the operator names from the video, made computable.

BOTH WERE FOUND BY WATCHING, NOT BY A COLUMN, and both are per-clip rather than
pooled -- which is why the arm tables missed them:

  "我看的这个视频有高度的重复动作,重复了三次"  (2026-09-13)
  "很多sample 有长期背对或侧对相机镜头的问题"   (2026-09-13, and again after
                                                 the same report on 2026-09-08)

REPLAY is measured on the MOTION, not the labels: two bars can carry the same
label and look different, and two different labels can be the same prototype.
Bars are compared in canonical pose (root-relative, shoulder-normalised) so
neither where the dancer stands nor which way they face can hide a replay.
Measured: on wild_v5:7664973324456613370:clip000 the selector arm replays 25%
of its bars against 0% for both the baseline and ground truth, while the
twenty-clip MEAN is 2.6% against ground truth's 2.5% -- the pooled number sees
nothing, which is the whole reason this reports per clip.

FACING is the share of frames whose chest is not pointing at the lens, and the
longest unbroken such stretch.  Ground truth turns away too -- its worst clip
holds 7.6 s -- so the target is not zero; it is ground truth's own distribution.

Neither column may be read pooled.  A single clip that replays a third of
itself is a defect the operator will name, and an average over twenty hides it.
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

FPS = 30.0
REPLAY_DISTANCE = 0.25      # canonical-pose units; below this two bars look alike
FACING_TOWARDS = 0.3        # chest-normal y component above which it faces the lens


def canonical(joints):
    joints = np.asarray(joints, float)
    joints = joints - joints[:, :1, :]
    width = np.linalg.norm(joints[:, 16, :] - joints[:, 17, :], axis=-1).mean()
    return joints / max(float(width), 1e-6)


def replay_share(joints, music):
    """Share of bars that repeat an earlier bar of the same clip."""
    canon = canonical(joints)
    bounds, _ = bar_grid_bounds(torch.as_tensor(music[:len(canon)]), 4, length=len(canon))
    if bounds is None:
        return float("nan")
    bars = [canon[int(lo):int(hi)] for lo, hi in zip(list(bounds)[:-1], list(bounds)[1:])
            if int(hi) - int(lo) > 8]
    if len(bars) < 3:
        return float("nan")
    repeats = 0
    for i in range(len(bars)):
        for earlier in range(i):
            span = min(len(bars[i]), len(bars[earlier]))
            gap = np.linalg.norm(bars[i][:span] - bars[earlier][:span], axis=-1).mean()
            if gap < REPLAY_DISTANCE:
                repeats += 1
                break
    return repeats / len(bars)


def facing(joints):
    """(share of frames not facing the lens, longest such stretch in seconds)."""
    joints = np.asarray(joints, float)
    across = joints[:, 17, :] - joints[:, 16, :]
    normal = np.stack([-across[:, 1], across[:, 0]], axis=1)
    normal = normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-6)
    away = normal[:, 1] < FACING_TOWARDS
    longest = current = 0
    for turned in away:
        current = current + 1 if turned else 0
        longest = max(longest, current)
    return float(away.mean()), longest / FPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--audio-dir", default="runs/txy_t_gt_eval/audio")
    ap.add_argument("--ground-truth-dir", default="runs/txy_t_gt_eval/motion")
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=DIR")
    ap.add_argument("--away-seconds", type=float, default=2.0,
                    help="an unbroken stretch this long is what reads as "
                         "'长期背对' on screen")
    ap.add_argument("--json")
    args = ap.parse_args()

    clips = [c.strip() for c in open(args.clips) if c.strip()]
    arms = [("ground truth", args.ground_truth_dir)]
    arms += [(s.rpartition("=")[0], s.rpartition("=")[2]) for s in args.arm]

    report = {}
    print("{:<22}{:>9}{:>12}{:>10}{:>12}{:>12}".format(
        "arm", "replay%", "worst clip", "away%", "longest s", "clips >= {:.0f}s".format(
            args.away_seconds)))
    for name, directory in arms:
        rows = {}
        for clip in clips:
            path = pathlib.Path(directory) / (clip + ".pkl")
            audio = pathlib.Path(args.audio_dir) / (clip + ".npy")
            if not path.is_file() or not audio.is_file():
                continue
            joints = np.asarray(pickle.load(open(path, "rb"))["full_pose"], float)
            share, longest = facing(joints)
            rows[clip] = {"replay": replay_share(joints, np.load(audio)),
                          "away": share, "longest_away_s": longest}
        if not rows:
            continue
        report[name] = rows
        replay = np.array([v["replay"] for v in rows.values()])
        away = np.array([v["away"] for v in rows.values()])
        longest = np.array([v["longest_away_s"] for v in rows.values()])
        print("{:<22}{:>8.1%}{:>12.0%}{:>10.1%}{:>12.1f}{:>12}".format(
            name, float(np.nanmean(replay)), float(np.nanmax(replay)),
            float(away.mean()), float(longest.mean()),
            int((longest >= args.away_seconds).sum())))

    print("\nPER CLIP -- the pooled numbers above hide exactly the clips the "
          "operator names:")
    truth = report.get("ground truth", {})
    for name, rows in report.items():
        if name == "ground truth":
            continue
        worst = sorted(rows.items(), key=lambda kv: -(kv[1]["replay"] or 0))[:3]
        print("  {} -- worst replays: {}".format(name, ", ".join(
            "{} {:.0%}".format(c.split(":")[1][-12:], v["replay"]) for c, v in worst)))
        turned = sorted(rows.items(), key=lambda kv: -kv[1]["longest_away_s"])[:3]
        print("  {} -- longest away : {}".format(name, ", ".join(
            "{} {:.1f}s".format(c.split(":")[1][-12:], v["longest_away_s"])
            for c, v in turned)))

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(report, indent=2))
        print("wrote", args.json)


if __name__ == "__main__":
    main()
