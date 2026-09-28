"""Measure how much a clip repeats itself, against ground truth on the same song.

The operator's report is "the moves are highly repetitive".  Choreography
repeats on purpose -- a chorus dances like a chorus -- so an absolute repetition
score decides nothing.  What decides it is the SAME reading on the ground truth
recording of the same clip: a generated clip is too repetitive only if it
repeats more than the dancer it is standing in for.

METHOD.  Slice the clip into half-second windows of root-relative joint
positions.  For each window, find its nearest neighbour among windows at least
``--min-gap`` seconds away in time -- far enough that adjacent frames of one
continuous movement cannot answer for a repeat.  Divide that distance by the
clip's own mean window-to-window distance, so the reading is scale free and a
clip that simply moves a lot cannot look varied for that reason alone.

  low  = every moment has a near twin elsewhere in the clip  -> repetitive
  high = moments are unlike anything else in the clip        -> varied

Two controls, both in ``tests/test_measure_motion_repetition.py``: a clip built
by tiling ONE block must read near zero, and a clip of independent blocks must
read high.  Without the first, a metric that is merely noisy would pass for one
that works.
"""

import argparse
import json
import pathlib
import pickle
import sys

import numpy as np
from scipy.spatial.distance import cdist

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

FPS = 30.0


def windows(joints, length, stride):
    relative = np.asarray(joints, float)
    relative = relative - relative[:, :1, :]
    flat = relative.reshape(len(relative), -1)
    starts = range(0, max(1, len(flat) - length + 1), stride)
    return np.stack([flat[s:s + length].ravel() for s in starts
                     if s + length <= len(flat)]), np.asarray(list(starts))


def novelty(joints, *, window_seconds=0.5, stride_frames=5, min_gap_seconds=1.5):
    """Median distance to the nearest window that is far away in time."""
    length = max(2, int(round(window_seconds * FPS)))
    if len(joints) < length * 2:
        return None
    block, starts = windows(joints, length, stride_frames)
    if len(block) < 4:
        return None
    # Pairwise distances between windows, then mask out everything that is
    # close in time -- including the window itself.  cdist rather than a
    # broadcast subtraction: the (n, n, d) intermediate is hundreds of MB on a
    # long clip and made the whole census take minutes.
    distance = cdist(block, block)
    gap = np.abs(starts[:, None] - starts[None, :])
    far = gap >= int(round(min_gap_seconds * FPS))
    if not far.any():
        return None
    masked = np.where(far, distance, np.inf)
    nearest = masked.min(axis=1)
    usable = np.isfinite(nearest)
    if usable.sum() < 2:
        return None
    scale = float(distance[far].mean())
    if scale <= 0:
        return None
    return float(np.median(nearest[usable]) / scale)


def load(path):
    payload = pickle.load(open(path, "rb"))
    return np.asarray(payload["full_pose"], float)


def run(arms, clips, ground_truth_dir, **options):
    truth_scores, report = {}, {"options": options, "arms": {}}
    for clip in clips:
        path = pathlib.Path(ground_truth_dir) / (clip + ".pkl")
        if path.exists():
            value = novelty(load(path), **options)
            if value is not None:
                truth_scores[clip] = value
    report["ground_truth"] = {
        "clips": len(truth_scores),
        "novelty": float(np.median(list(truth_scores.values()))) if truth_scores else None,
    }
    for name, directory in arms:
        scores, paired = {}, []
        for clip in clips:
            path = pathlib.Path(directory) / (clip + ".pkl")
            if not path.exists():
                continue
            value = novelty(load(path), **options)
            if value is None:
                continue
            scores[clip] = value
            if clip in truth_scores:
                paired.append(value < truth_scores[clip])
        if not scores:
            raise SystemExit(
                "error: arm {!r} scored 0 of {} clips from {}.  Nothing was "
                "measured.".format(name, len(clips), directory))
        report["arms"][name] = {
            "clips": len(scores),
            "novelty": float(np.median(list(scores.values()))),
            "clips_more_repetitive_than_truth": "{}/{}".format(len(
                [x for x in paired if x]), len(paired)),
            "per_clip": {k: round(v, 4) for k, v in sorted(scores.items())},
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--clips", required=True, type=pathlib.Path)
    parser.add_argument("--ground-truth", required=True, type=pathlib.Path)
    parser.add_argument("--window-seconds", type=float, default=0.5)
    parser.add_argument("--stride-frames", type=int, default=5)
    parser.add_argument("--min-gap-seconds", type=float, default=1.5)
    parser.add_argument("--out", type=pathlib.Path)
    arguments = parser.parse_args()

    arms = []
    for entry in arguments.arm:
        name, _, directory = entry.rpartition("=")
        if not pathlib.Path(directory).is_dir():
            raise SystemExit("error: --arm {!r} points at {!r}, which does not "
                             "exist.".format(name, directory))
        arms.append((name, directory))
    clips = [line.strip() for line in arguments.clips.read_text().splitlines()
             if line.strip()]
    report = run(arms, clips, arguments.ground_truth,
                 window_seconds=arguments.window_seconds,
                 stride_frames=arguments.stride_frames,
                 min_gap_seconds=arguments.min_gap_seconds)
    if arguments.out:
        arguments.out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"ground_truth": report["ground_truth"],
                      "arms": {k: {kk: vv for kk, vv in v.items() if kk != "per_clip"}
                               for k, v in report["arms"].items()}},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
