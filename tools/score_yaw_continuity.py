"""Does the body ever SNAP around, and does it still turn?

WHY BOTH COLUMNS.  2026-09-14 the operator reported "会偶尔出现一次人体旋转的跳变,
视觉上看着不连续,缺帧" on a render.  It was real -- ``face_camera`` wrapped its
error per sample while low-passing an unwrapped yaw, so the correction jumped by
``2*pi*strength`` whenever the smoothed yaw crossed an odd multiple of pi.  But
the obvious fix for a snap is to stop rotating at all, and that would replace a
visible defect with an invisible one: ``face_camera``'s own docstring says a
dancer who never turns is its own defect and ground truth spends 12.3% of its
frames off camera.  So this reports the snap AND the turning, and neither alone
is a verdict.

THE THRESHOLD IS GROUND TRUTH'S.  Over the 20 T-line eval clips the ground
truth's worst frame-to-frame yaw change is 25.2 degrees, so 30 is a line it
never crosses -- which makes "ground truth reads zero" a positive control this
tool passes rather than an assumption.  A 15-degree line does NOT work: ground
truth crosses it 39 times, so a gate built on it would fire on the reference.

Yaw is the hip axis, via ``infer_atomic._body_forward_yaw`` -- the same function
``face_camera`` corrects against, so the reading and the thing being judged are
in one space (CLAUDE.md 2.1.2).
"""
import argparse
import pathlib
import pickle
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from infer_atomic import _body_forward_yaw  # noqa: E402

SPIKE_DEGREES = 30.0
FPS = 30.0


def read(path):
    joints = np.asarray(pickle.load(open(path, "rb"))["full_pose"], dtype=np.float64)
    yaw = np.unwrap(_body_forward_yaw(joints))
    step = np.abs(np.degrees(np.diff(yaw)))
    return {
        "worst": float(step.max()) if len(step) else 0.0,
        "spikes": int((step > SPIKE_DEGREES).sum()),
        # Total turning is the SUM of the steps, which is what "the dance still
        # turns" means; the net angle would read zero for a dancer who spins one
        # way and back.
        "turning": float(np.degrees(np.abs(np.diff(yaw))).sum()),
        "off_camera": float((np.sin(_body_forward_yaw(joints)) > -0.5).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", default="runs/txy_t_gt_eval/motion")
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=DIR")
    args = ap.parse_args()

    truth = pathlib.Path(args.truth)
    arms = {}
    for entry in args.arm:
        name, _, directory = entry.partition("=")
        arms[name] = pathlib.Path(directory)
    names = sorted(p.name for p in truth.glob("wild_v5:*.pkl"))
    for directory in arms.values():
        names = [n for n in names if (directory / n).is_file()]
    if not names:
        raise SystemExit("no clip is present in the truth and every arm")

    print("{} clips; a spike is >{:g} deg in ONE frame at {:g} fps"
          .format(len(names), SPIKE_DEGREES, FPS))
    print("{:22s} {:>8s} {:>8s} {:>9s}  {:>9s}  {:>11s}".format(
        "arm", "clips", "spikes", "worst", "off-camera", "turning/clip"))
    rows = [("ground truth", truth)] + list(arms.items())
    for name, directory in rows:
        values = [read(directory / n) for n in names]
        affected = sum(1 for v in values if v["spikes"])
        print("{:22s} {:5d}/{:<2d} {:8d} {:8.1f}d  {:9.1%}  {:10.0f}d".format(
            name, affected, len(names), sum(v["spikes"] for v in values),
            max(v["worst"] for v in values),
            float(np.mean([v["off_camera"] for v in values])),
            float(np.median([v["turning"] for v in values]))))
    print("\nA fix that drives 'turning' far below ground truth's has replaced a "
          "visible defect with an invisible one; read both columns.")


if __name__ == "__main__":
    main()
