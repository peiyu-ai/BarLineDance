"""Per-window worst frame-to-frame body yaw, for every window in a release.

WHY IT HAS TO EXIST.  2026-09-14 the operator reported an occasional whole-body
rotation snap in a render.  Two sources were found; this one is the library
itself.  Decoded through the repository's own ``_decode_raw_151`` and read with
``infer_atomic._body_forward_yaw`` -- the same function the judgement uses --
14.5% of a 400-window sample of the T-line train split turns more than 30
degrees in ONE frame and 1.5% turns more than 90, worst 145.3.  At 30 fps that
is 4350 deg/s; the ten held-out ground-truth clips never exceed 25.2 deg/frame.
It is a reconstruction artefact, and retrieval pastes it into the dance.

VALIDATED BEFORE USE, which the first attempt at this was not.  The first
version read the yaw off the global orient's rot6d first column instead of
decoding -- 6 lines instead of an FK pass -- and it correlated only +0.34 with
the hip-based reading and called ground-truth clips 148 deg/frame that are
actually 22.  Decoding the raw 151-D and reading it the same way the judgement
does reproduces the pickle-based reading EXACTLY on all ten staged ground
truths (mean |difference| 0.00 deg, corr +1.000), which is the positive control
this tool needs before anything is filtered on it.

The output is a JSON keyed by window name, for ``--retrieval-max-yaw-step``.
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from infer_atomic import _body_forward_yaw               # noqa: E402
from tools.render_dance_video import _decode_raw_151     # noqa: E402


SPEED_JOINTS = [1, 2, 4, 5, 7, 8, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21]
SPEED_BASELINE = 15      # frames; the local median the spike is measured against


def speed_spikes(joints):
    """Per-frame speed divided by the LOCAL median speed.

    Relative, not absolute, because a dance is fast in places and that is not a
    defect: what reads as a jolt is a frame that moves far more than the frames
    around it.  Sliced into 150-frame windows, ground truth's worst such frame
    over the 20 eval clips is 3.12x and it never once reaches 4x (0 of 62
    windows), while the T-line library reaches 13.4x and 5.8% of its windows
    carry a spike above 4x.  That is the line, and it is ground truth's, not one
    I picked.
    """
    joints = np.asarray(joints, dtype=np.float64)
    if len(joints) < 3:
        return np.zeros(max(len(joints) - 1, 0), dtype=np.float32)
    speed = np.linalg.norm(np.diff(joints[:, SPEED_JOINTS], axis=0), axis=2).mean(1)
    pad = SPEED_BASELINE // 2
    base = np.array([np.median(speed[max(0, i - pad):i + pad + 1])
                     for i in range(len(speed))])
    return (speed / np.maximum(base, 1e-9)).astype(np.float32)


def worst_step(joints):
    joints = np.asarray(joints, dtype=np.float64)
    if len(joints) < 2:
        return 0.0
    yaw = np.unwrap(_body_forward_yaw(joints))
    return float(np.max(np.abs(np.degrees(np.diff(yaw)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--split", action="append", default=None,
                    help="default: train, val and test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps-out", default=None, metavar="NPZ",
                    help="also write the PER-FRAME steps, so a filter can ask "
                         "about the range it actually intends to use instead of "
                         "blacklisting a whole 150-frame window for one bad frame")
    args = ap.parse_args()

    release = pathlib.Path(args.release)
    splits = args.split or ["train", "val", "test"]
    normalizer = torch.load(release / "normalizer.pt", map_location="cpu",
                            weights_only=False)
    # THE RELEASE IS NORMALISED TO [-1, 1], NOT [0, 1].  ``_normalizer_affine``
    # in infer_atomic.py is the authority and returns ``(span/2, low + span/2)``;
    # this file used the [0, 1] inverse (``x*span + low``) until 2026-09-16, which
    # scales every channel 2x and shifts it, so the decoded skeleton is a
    # different dance.  Caught by a check that should have been the first one
    # run: the release's own music window for ``<clip>_slice0`` matches the eval
    # export's music at offset 0 EXACTLY (max|diff| 0.00000), so the frames do
    # correspond -- and once the affine was right the root-relative pose error
    # against the same frames of the eval pkl went 0.39959 m -> 0.00000 m and
    # peak ankle speed 7.54 -> 4.51 m/s, which is the eval pkl's own reading.
    #
    # It matters here and not only cosmetically: this file's output feeds
    # ``--retrieval-max-yaw-step`` and ``--retrieval-max-speed-spike``, so every
    # threshold derived from the old npz was derived from a doubled skeleton.
    low = normalizer["data_min"].numpy()
    high = normalizer["data_max"].numpy()
    span = (high - low).copy()
    span[span == 0] = 1.0
    scale = span / 2.0
    low = low + span / 2.0

    steps = {}
    speeds = {}
    per_frame = {}
    per_frame_speed = {}
    for split in splits:
        motion_path = release / split / "motion.npy"
        if not motion_path.is_file():
            continue
        motion = np.load(motion_path, mmap_mode="r")
        names = json.loads((release / split / "names.json").read_text())
        started = time.time()
        for index in range(len(motion)):
            raw = np.asarray(motion[index]) * scale[None, :] + low[None, :]
            joints, _ = _decode_raw_151(raw)
            series = np.abs(np.degrees(np.diff(np.unwrap(
                _body_forward_yaw(np.asarray(joints, dtype=np.float64))))))
            per_frame[names[index]] = series.astype(np.float32)
            steps[names[index]] = float(series.max()) if len(series) else 0.0
            spikes = speed_spikes(joints)
            per_frame_speed[names[index]] = spikes
            speeds[names[index]] = float(spikes.max()) if len(spikes) else 0.0
            if index and index % 500 == 0:
                done = index / len(motion)
                print("  {} {}/{} ({:.0%}), {:.0f}s elapsed".format(
                    split, index, len(motion), done, time.time() - started),
                    flush=True)
        print("{}: {} windows in {:.0f}s".format(split, len(motion),
                                                 time.time() - started))

    values = np.array(list(steps.values()))
    payload = {
        "rule": "max |d yaw| per frame, degrees; yaw from infer_atomic."
                "_body_forward_yaw on tools.render_dance_video._decode_raw_151",
        "windows": len(steps),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
        "over_30": int((values > 30).sum()),
        "over_90": int((values > 90).sum()),
        "steps": steps,
        "speed_rule": "per-frame joint speed / local median over {} frames".format(
            SPEED_BASELINE),
        "speed_over_4": int(sum(1 for v in speeds.values() if v > 4.0)),
        "speed_max": float(max(speeds.values())) if speeds else 0.0,
        "speed_spikes": speeds,
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    if args.steps_out:
        steps_out = pathlib.Path(args.steps_out)
        steps_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(steps_out, **per_frame)
        speed_out = steps_out.with_name(steps_out.stem + "_speed.npz")
        np.savez_compressed(speed_out, **per_frame_speed)
        print("per-frame speed spikes -> {}".format(speed_out))
        print("per-frame steps -> {}".format(steps_out))
    print("{} windows -> {}  (p50 {:.1f}, p99 {:.1f}, max {:.1f}; >30 deg: {}, "
          ">90 deg: {})".format(len(steps), out, payload["p50"], payload["p99"],
                                payload["max"], payload["over_30"],
                                payload["over_90"]))


if __name__ == "__main__":
    main()
