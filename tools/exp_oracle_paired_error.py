#!/usr/bin/env python3
"""Per-frame error to the clip's OWN ground truth, for the arms that have a target.

WHY THIS COLUMN EXISTS AND WHY ONLY HERE.  Every other scorer in this repository
compares DISTRIBUTIONS -- an arm's mean energy against ground truth's mean
energy -- because the shipping arm generates from music alone and has no paired
target to be near.  The completion ORACLE does: it was handed the clip's own
ground-truth motion as its draft, so "did it reproduce it" is answerable
directly, and a distributional column cannot answer it (an arm can match ground
truth's mean energy while dancing something else entirely).

THE TWO NUMBERS.

``mpjpe_rootrel``   mean over frames and joints of the distance between the
                    arm's root-relative joint position and ground truth's, in
                    metres.  Root-relative because the root carries the whole
                    clip's travel and would otherwise dominate; ``mpjpe_global``
                    is reported beside it so the travel is not hidden.

``speed_corr``      per-clip Pearson correlation between the arm's frame-by-frame
                    root-relative mean joint speed and ground truth's.  This is
                    the timing column: dataset/atomic.py measured the shipping
                    output correlating with its own DRAFT's speed profile at
                    -0.110, and the question here is whether a draft that IS the
                    target changes that.  Reported both raw and after the
                    9-frame moving average the stillness criterion uses, because
                    a correlation at 30 fps is dominated by frame-scale noise.

CONTROLS (CLAUDE.md 2.1 -- a criterion may not judge before it has been checked
in both directions).

* positive, exact:  ground truth scored against itself must read mpjpe 0.000 and
  speed_corr 1.000.  ``--controls`` runs it; a reading other than that means the
  loader or the alignment is wrong and no arm behind it may be quoted.
* scale:            the shipping OUTPUT and the retrieval DRAFT are scored on
  the same columns, so the oracle's reading has a "this is what no information
  looks like" level beside it rather than being read against 0.
* null for the correlation: each clip's ground-truth speed profile against a
  DIFFERENT clip's, truncated to the shorter -- the value the correlation takes
  when two real dances of the same dancer are compared with no shared timing.
  Without it, a small positive correlation cannot be told from the floor.

WHERE THE ERROR LIVES.  Beyond the pooled numbers the report splits the
per-frame error three ways, because "it reproduces some of the clip" and "it
reproduces the clip except at the joins" are different findings:

* ``by_label``    frames the ground-truth plan calls 0 (filler/transition)
                  against frames it names an atomic movement.
* ``by_seam``     frames within ``--seam-radius`` of a completion WINDOW
                  junction (the overlap-add joins at multiples of the stride)
                  against the rest.
* ``worst``       the named frame and second of each clip's largest departure.
"""

import argparse
import json
import pathlib
import pickle
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.measure_motion_dynamics import SMOOTH_WIDTH, joint_speed, low_pass  # noqa: E402

FPS = 30.0


def load_joints(path):
    with open(str(path), "rb") as handle:
        return np.asarray(pickle.load(handle)["full_pose"], float)


def root_relative(joints):
    return joints - joints[:, :1, :]


def per_frame_error(arm, truth):
    """(root-relative per-frame MPJPE, global per-frame MPJPE), both [T]."""
    frames = min(len(arm), len(truth))
    a, t = arm[:frames], truth[:frames]
    rootrel = np.linalg.norm(root_relative(a) - root_relative(t), axis=2).mean(1)
    glob = np.linalg.norm(a - t, axis=2).mean(1)
    return rootrel, glob


def correlate(a, b):
    frames = min(len(a), len(b))
    a, b = np.asarray(a[:frames], float), np.asarray(b[:frames], float)
    if frames < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def speed_profiles(joints):
    return joint_speed(joints), joint_speed(low_pass(joints, SMOOTH_WIDTH))


def label_track(labels_root, name, frames):
    """The ground-truth label track for one clip, or None when the release lacks it."""
    if labels_root is None:
        return None
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import infer_atomic as IA

    global _PLANS
    try:
        plans = _PLANS
    except NameError:
        plans = _PLANS = IA.GroundTruthPlanStore(labels_root)
    if not plans.has_sequence(name):
        return None
    track = plans._full_track(name)
    padded = np.zeros(frames, np.int64)
    covered = min(len(track), frames)
    padded[:covered] = np.maximum(track[:covered], 0)
    return padded


def seam_frames(frames, window, stride, radius):
    """Frames within ``radius`` of a completion window junction.

    The junction sits at the middle of the overlap (``_blend_weights``), i.e. at
    ``start + window - overlap/2`` for each window after the first.
    """
    flag = np.zeros(frames, bool)
    if frames <= window:
        return flag
    overlap = window - stride
    starts = list(range(0, frames - window + 1, stride))
    if starts[-1] != frames - window:
        starts.append(frames - window)
    for index, start in enumerate(starts):
        if index == 0:
            continue
        junction = start + overlap // 2
        flag[max(0, junction - radius): min(frames, junction + radius + 1)] = True
    return flag


def score_arm(directory, clips, truth_dir, *, labels_root, window, stride, seam_radius):
    per_clip = {}
    for name in clips:
        path = pathlib.Path(directory) / (name + ".pkl")
        if not path.exists():
            continue
        arm = load_joints(path)
        truth = load_joints(pathlib.Path(truth_dir) / (name + ".pkl"))
        frames = min(len(arm), len(truth))
        rootrel, glob = per_frame_error(arm, truth)
        arm_raw, arm_smooth = speed_profiles(arm[:frames])
        truth_raw, truth_smooth = speed_profiles(truth[:frames])
        row = {
            "frames": int(frames),
            "mpjpe_rootrel": float(rootrel.mean()),
            "mpjpe_global": float(glob.mean()),
            "speed_corr": correlate(arm_raw, truth_raw),
            "speed_corr_smoothed": correlate(arm_smooth, truth_smooth),
            "worst_frame": int(rootrel.argmax()),
            "worst_second": round(float(rootrel.argmax() / FPS), 2),
            "worst_mpjpe": float(rootrel.max()),
        }
        labels = label_track(labels_root, name, frames)
        if labels is not None:
            filler = labels == 0
            if filler.any() and (~filler).any():
                row["mpjpe_filler_frames"] = float(rootrel[filler].mean())
                row["mpjpe_atomic_frames"] = float(rootrel[~filler].mean())
                row["filler_frame_share"] = float(filler.mean())
        seam = seam_frames(frames, window, stride, seam_radius)
        if seam.any() and (~seam).any():
            row["mpjpe_seam_frames"] = float(rootrel[seam].mean())
            row["mpjpe_interior_frames"] = float(rootrel[~seam].mean())
            row["seam_frame_share"] = float(seam.mean())
        per_clip[name] = row
    if not per_clip:
        raise SystemExit("error: 0 of {} clips scored from {}".format(len(clips), directory))

    def pooled(key):
        values = [row[key] for row in per_clip.values() if key in row
                  and not np.isnan(row[key])]
        return float(np.mean(values)) if values else float("nan")

    summary = {"clips": len(per_clip)}
    for key in ("mpjpe_rootrel", "mpjpe_global", "speed_corr", "speed_corr_smoothed",
                "mpjpe_filler_frames", "mpjpe_atomic_frames",
                "mpjpe_seam_frames", "mpjpe_interior_frames"):
        summary[key] = pooled(key)
    summary["per_clip"] = per_clip
    return summary


def shuffled_speed_null(clips, truth_dir, seed=20260904):
    """Each clip's ground-truth speed against a DIFFERENT clip's ground truth."""
    profiles = {}
    for name in clips:
        path = pathlib.Path(truth_dir) / (name + ".pkl")
        if path.exists():
            joints = load_joints(path)
            profiles[name] = speed_profiles(joints)
    names = sorted(profiles)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(names))
    # A derangement, so no clip is paired with itself.
    for index in range(len(order)):
        if order[index] == index:
            swap = (index + 1) % len(order)
            order[index], order[swap] = order[swap], order[index]
    raw, smooth = [], []
    for index, name in enumerate(names):
        other = names[order[index]]
        raw.append(correlate(profiles[name][0], profiles[other][0]))
        smooth.append(correlate(profiles[name][1], profiles[other][1]))
    return {"pairs": len(names),
            "speed_corr": float(np.nanmean(raw)),
            "speed_corr_smoothed": float(np.nanmean(smooth))}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--clips", required=True, type=pathlib.Path)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--labels-root", default=None,
                        help="release root whose windows.jsonl carries the "
                             "ground-truth label track, for the filler/atomic split")
    parser.add_argument("--window", type=int, default=150)
    parser.add_argument("--stride", type=int, default=75)
    parser.add_argument("--seam-radius", type=int, default=5)
    parser.add_argument("--controls", action="store_true")
    parser.add_argument("--out", type=pathlib.Path)
    args = parser.parse_args()

    clips = [line.strip() for line in args.clips.read_text().splitlines() if line.strip()]
    options = dict(labels_root=args.labels_root, window=args.window,
                   stride=args.stride, seam_radius=args.seam_radius)
    report = {"definition": {"fps": FPS, "smooth_width": SMOOTH_WIDTH,
                             "window": args.window, "stride": args.stride,
                             "seam_radius": args.seam_radius},
              "arms": {}}
    if args.controls:
        identity = score_arm(args.ground_truth, clips, args.ground_truth, **options)
        report["control_ground_truth_against_itself"] = {
            "mpjpe_rootrel": identity["mpjpe_rootrel"],
            "speed_corr": identity["speed_corr"],
            "passed": bool(identity["mpjpe_rootrel"] < 1e-9
                           and abs(identity["speed_corr"] - 1.0) < 1e-9)}
        if not report["control_ground_truth_against_itself"]["passed"]:
            raise SystemExit(
                "error: ground truth does not score 0.000 / 1.000 against itself "
                "({}); the loader or the alignment is wrong, not scoring any arm."
                .format(report["control_ground_truth_against_itself"]))
        report["control_shuffled_speed_null"] = shuffled_speed_null(
            clips, args.ground_truth)
    for entry in args.arm:
        name, _, directory = entry.rpartition("=")
        if not pathlib.Path(directory).is_dir():
            raise SystemExit("error: --arm {!r} points at {!r}".format(name, directory))
        report["arms"][name] = score_arm(directory, clips, args.ground_truth, **options)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print("%-26s %6s %9s %9s %9s %9s" % ("arm", "clips", "mpjpe", "mpjpe_g",
                                         "spd_corr", "spd_sm"))
    if "control_shuffled_speed_null" in report:
        null = report["control_shuffled_speed_null"]
        print("%-26s %6d %9s %9s %9.3f %9.3f" % (
            "NULL gt vs other clip", null["pairs"], "-", "-",
            null["speed_corr"], null["speed_corr_smoothed"]))
    for name, row in report["arms"].items():
        print("%-26s %6d %9.4f %9.4f %9.3f %9.3f" % (
            name, row["clips"], row["mpjpe_rootrel"], row["mpjpe_global"],
            row["speed_corr"], row["speed_corr_smoothed"]))


if __name__ == "__main__":
    main()
