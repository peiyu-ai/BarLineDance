#!/usr/bin/env python3
"""Low-pass a generated clip, and refuse to call the result an improvement on its own.

The wild 340-frame arm is rough in a way that is visible before it is
measurable: 144 frames per ten seconds exceed the 99th percentile of ground
truth's own jerk, against 3.2 for ground truth itself, and the root carries
3.6x ground truth's share of power above 5 Hz.  Roughly a third of the rough
frames sit within +-2 frames of a plan boundary; the rest are spread through
the clip, which is why a draft-side fix (``--draft-gap-fill interpolate``,
which removes 41% of the boundary excess) leaves most of them standing.

This filters what is left, and the reason it is a separate tool rather than a
step inside inference is that **smoothing can always succeed at the wrong
thing**.  A strong enough filter drives every roughness statistic to ground
truth by deleting the dance: joint speed falls, amplitude falls, and the
numbers that were supposed to show the fix working all improve.  So the report
prints the cost columns beside the benefit columns and the gate fails when the
cost exceeds ``--max-speed-loss``.  There is no setting at which this tool
reports success without also reporting what it spent.

Filtering happens on the *rotations*, not the joint positions.  A Savitzky-Golay
pass over ``full_pose`` smooths each joint independently and therefore stretches
the skeleton -- bone lengths stop being constant, and the decoded clip is no
longer a body.  Rotations are filtered in the continuous 6-D representation
(axis-angle wraps at +-pi, and filtering across a wrap inverts the joint), then
re-orthonormalised by the same ``ax_from_6v`` the generator's own decode uses,
and the pose is rebuilt through the repo's forward kinematics.  Root translation
is filtered directly, being a plain position.

Usage::

    python3 tools/smooth_generated_motion.py --run runs/vis_wild_20260817b/arm340_seed20260817 \\
        --output runs/smoothed --ground-truth-dir runs/wild_v4_acct_gt_eval/motion \\
        --window 9 --polyorder 3
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import pickle
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/edge-matplotlib-cache")

import numpy as np
import torch
from scipy.signal import savgol_filter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dataset.quaternion import ax_from_6v, ax_to_6v  # noqa: E402

FPS = 30.0


def _high_frequency_share(values: np.ndarray, cutoff_hz: float = 5.0) -> float:
    """Share of motion power above ``cutoff_hz``; the shape of the roughness.

    **Windowed and detrended, and the first version was neither.**  A bare
    ``rfft`` with only the mean removed treats a ~520-frame track as one period
    of a periodic signal; a track with a large slow drift is not periodic over
    its own length, so the rectangular window's sidelobes -- which fall only as
    1/f^2 -- smear that drift across the entire spectrum.  Measured 2026-08-29
    against known answers:

    ==================================  ==============  ==============
    input (true power above 5 Hz)       old reading     this reading
    ==================================  ==============  ==============
    pure 0.5 Hz sine (exactly 0)              0.0686%          0.0000%
    + white noise, sd 0.5 mm                  0.0690%          0.0011%
    + white noise, sd 10 mm                   0.0840%          0.3400%
    ==================================  ==============  ==============

    The old version moved by 22% while the thing it exists to measure moved by
    20x, and it reported 0.0686% where the answer is zero.  On 400 matched clips
    it read generated 1.433% against truth 1.350% -- ratio 1.06x -- while this
    version reads 1.168% against 0.0166%, ratio **70.3x**.

    Every figure computed with the old version is leakage-dominated and must not
    be compared against one from this version.  That includes the ``root HF
    16.5% vs truth 2.2%`` column in ``tools/run_wild_acct_c_line.sh:184-192``:
    the truth's 2.2% there is almost entirely sidelobe, so the ratio of the two
    is not interpretable.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values.reshape(len(values), -1)
    frames = len(values)
    if frames < 8:
        return 0.0
    # Linear detrend before the window, so the drift is removed rather than
    # tapered; the taper alone leaves a residual ramp at the ends.
    time = np.arange(frames, dtype=np.float64)
    design = np.stack([np.ones(frames), time - time.mean()], axis=1)
    values = values - design @ np.linalg.lstsq(design, values, rcond=None)[0]
    values = values * np.hanning(frames)[:, None]
    freqs = np.fft.rfftfreq(frames, 1.0 / FPS)
    power = (np.abs(np.fft.rfft(values, axis=0)) ** 2).sum(axis=1)
    total = power[freqs > 0.2].sum()
    return float(power[freqs > cutoff_hz].sum() / max(total, 1e-12))


def _speed(joints: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(joints, axis=0), axis=-1).mean(axis=1) * FPS


def _jerk(joints: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(joints, n=3, axis=0), axis=-1).mean(axis=1) * FPS ** 3


def smooth_payload(payload: dict, window: int, polyorder: int,
                   channels: str = "all") -> dict:
    """Return a copy of one generated clip with the chosen channels filtered.

    ``channels="root"`` filters the three root-translation dimensions and leaves
    every rotation bit-identical.  That is not a weaker version of the default;
    it is the one the measurement asks for.  Split by channel on the wild 340
    arm (tools/probe_roughness_structure.py), the generated *articulation* --
    the pose with the root subtracted -- has jerk 404 against ground truth's
    398 and an across-joint speed CV of 0.743 against 0.737, i.e. it is already
    the recording.  The root translation has jerk 1695 against 91, and because
    every joint inherits it through forward kinematics it carries 90% of the
    whole-body figure.

    So filtering the rotations spends the dance to fix a channel the dance is
    not in, which is what the earlier whole-clip runs measured as a 7-30% loss
    of joint speed.  Re-seating the body rigidly on a smoothed root cannot
    change a joint angle at all: measured over 60 clips the articulation jerk is
    bit-identical at 403.562 for every window tried, while whole-body jerk falls
    from 4.47x ground truth to 1.15x and the floor-path speed from 3.21x to
    1.10x.

    This is a mitigation and not a fix, and the report says so: the model still
    emits the broken channel and this rewrites it afterwards.
    """
    from vis import SMPLSkeleton

    poses = np.asarray(payload["smpl_poses"], dtype=np.float32).reshape(-1, 24, 3)
    trans = np.asarray(payload["smpl_trans"], dtype=np.float32)
    length = len(poses)
    # savgol needs an odd window no longer than the signal; a clip too short to
    # filter is returned untouched rather than filtered with a silently
    # different window than the one that was asked for.
    if window > length:
        return dict(payload)
    if channels not in ("all", "root"):
        raise ValueError("unknown channel set {!r}".format(channels))
    six = ax_to_6v(torch.from_numpy(poses)).numpy().reshape(length, -1)
    if channels == "all":
        six = savgol_filter(six, window, polyorder, axis=0)
    trans = savgol_filter(trans, window, polyorder, axis=0)
    rotations = ax_from_6v(torch.from_numpy(six.reshape(length, 24, 6)))
    root = torch.from_numpy(trans)
    full = SMPLSkeleton().forward(rotations.unsqueeze(0), root.unsqueeze(0))[0]

    out = dict(payload)
    out["smpl_poses"] = rotations.reshape(length, 72).numpy()
    out["smpl_trans"] = trans
    out["full_pose"] = full.numpy()
    out["postprocess"] = {
        "filter": "savitzky_golay",
        "window": window,
        "polyorder": polyorder,
        "channels": channels,
        "applied_to": ("6d_rotations_and_root_translation" if channels == "all"
                       else "root_translation_only"),
        "note": "rotations refiltered in 6-D and re-orthonormalised; joint "
                "positions are re-derived by forward kinematics, never filtered "
                "directly, which would not preserve bone length",
    }
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="an infer_atomic.py output directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--ground-truth-dir", help="per-clip .pkl to read the target roughness from")
    parser.add_argument("--window", type=int, default=9)
    parser.add_argument("--polyorder", type=int, default=3)
    parser.add_argument("--channels", choices=("all", "root"), default="all",
                        help="root: filter only the 3 root-translation dimensions, "
                             "leaving every rotation bit-identical")
    parser.add_argument(
        "--max-speed-loss",
        type=float,
        default=0.10,
        help="fail if the median joint speed falls by more than this fraction; "
             "a filter that buys smoothness by deleting the dance must not pass; "
             "applies to --channels all, where speed loss and dance loss cannot "
             "be told apart",
    )
    parser.add_argument(
        "--max-articulation-change",
        type=float,
        default=0.01,
        help="--channels root only: fail if the root-relative jerk moves at all, "
             "which would mean the filter reached a joint angle",
    )
    options = parser.parse_args(argv)
    if options.window % 2 == 0 or options.window < 3:
        parser.error("--window must be an odd number >= 3")
    if options.polyorder >= options.window:
        parser.error("--polyorder must be smaller than --window")

    run = pathlib.Path(options.run)
    out_dir = pathlib.Path(options.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = sorted(run.glob("*.pkl"))
    if not clips:
        raise SystemExit("no generated clips under {}".format(run))

    rows = []
    for path in clips:
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        smoothed = smooth_payload(payload, options.window, options.polyorder,
                                  channels=options.channels)
        with open(out_dir / path.name, "wb") as handle:
            pickle.dump(smoothed, handle)
        before = np.asarray(payload["full_pose"], dtype=np.float64)
        after = np.asarray(smoothed["full_pose"], dtype=np.float64)
        row = {
            "clip": path.stem,
            "hf_root_before": _high_frequency_share(before[:, 0, :]),
            "hf_root_after": _high_frequency_share(after[:, 0, :]),
            "hf_ankle_before": _high_frequency_share(before[:, [7, 8]]),
            "hf_ankle_after": _high_frequency_share(after[:, [7, 8]]),
            "speed_before": float(np.median(_speed(before))),
            "speed_after": float(np.median(_speed(after))),
            "jerk_before": float(np.median(_jerk(before))),
            "jerk_after": float(np.median(_jerk(after))),
            # The dance itself: the pose with the root subtracted.  A filter
            # that touches only the root cannot change this, and that is a
            # check rather than a claim -- see the gate below.
            "articulation_jerk_before": float(np.median(_jerk(before[:, 1:] - before[:, 0:1]))),
            "articulation_jerk_after": float(np.median(_jerk(after[:, 1:] - after[:, 0:1]))),
        }
        if options.ground_truth_dir:
            gt_path = pathlib.Path(options.ground_truth_dir) / path.name
            if gt_path.is_file():
                with open(gt_path, "rb") as handle:
                    truth = np.asarray(pickle.load(handle)["full_pose"], dtype=np.float64)
                truth = truth[:len(before)]
                row["hf_root_truth"] = _high_frequency_share(truth[:, 0, :])
                row["speed_truth"] = float(np.median(_speed(truth)))
                row["jerk_truth"] = float(np.median(_jerk(truth)))
        rows.append(row)

    median = lambda key: float(np.median([r[key] for r in rows if key in r]))
    speed_loss = 1.0 - median("speed_after") / max(median("speed_before"), 1e-9)
    # Did the filter change the dance?  For channels="root" the answer is
    # arithmetically no -- the body is re-seated rigidly on a smoothed root and
    # no joint angle is touched -- so this is the check that turns that from a
    # claim into a measurement.  It is what lets the speed criterion below be
    # about distance to ground truth rather than about loss: "joint speed fell"
    # is only a cost when it cannot be told apart from "the dance was deleted",
    # and here it can.
    articulation_change = abs(
        median("articulation_jerk_after") / max(median("articulation_jerk_before"), 1e-9) - 1.0)
    has_truth = any("speed_truth" in r for r in rows)
    toward = None
    if has_truth:
        truth_speed = max(median("speed_truth"), 1e-9)
        before_gap = abs(median("speed_before") / truth_speed - 1.0)
        after_gap = abs(median("speed_after") / truth_speed - 1.0)
        toward = {
            "speed_ratio_to_truth": [median("speed_before") / truth_speed,
                                     median("speed_after") / truth_speed],
            "speed_gap_to_truth": [before_gap, after_gap],
            "speed_moved_toward_truth": bool(after_gap < before_gap),
        }
        if "jerk_truth" in rows[0]:
            truth_jerk = max(median("jerk_truth"), 1e-9)
            toward["jerk_ratio_to_truth"] = [median("jerk_before") / truth_jerk,
                                             median("jerk_after") / truth_jerk]
            toward["jerk_moved_toward_truth"] = bool(
                abs(median("jerk_after") / truth_jerk - 1.0)
                < abs(median("jerk_before") / truth_jerk - 1.0))
    report = {
        "run": str(run),
        "output": str(out_dir),
        "clips": len(rows),
        "filter": {"window": options.window, "polyorder": options.polyorder},
        "benefit": {
            "hf_root": [median("hf_root_before"), median("hf_root_after")],
            "hf_ankle": [median("hf_ankle_before"), median("hf_ankle_after")],
            "jerk": [median("jerk_before"), median("jerk_after")],
        },
        "cost": {
            "median_joint_speed": [median("speed_before"), median("speed_after")],
            "speed_loss_fraction": speed_loss,
            "max_speed_loss": options.max_speed_loss,
        },
    }
    report["cost"]["articulation_jerk"] = [median("articulation_jerk_before"),
                                           median("articulation_jerk_after")]
    report["cost"]["articulation_change_fraction"] = articulation_change
    report["cost"]["max_articulation_change"] = options.max_articulation_change
    if toward:
        report["toward_ground_truth"] = toward

    if options.channels == "root":
        # Root-only: the dance must be provably untouched, and the two axes the
        # filter is for must end closer to ground truth than they started.
        # Falling joint speed is not a cost here -- the unfiltered arm is 1.58x
        # ground truth *because* of the jitter this removes -- so gating on
        # "speed fell" would reject the filter for doing its job.
        reasons = []
        if articulation_change > options.max_articulation_change:
            reasons.append("articulation jerk changed by {:.2%}, more than the allowed {:.2%} -- "
                           "a root-only filter must not touch a joint angle".format(
                               articulation_change, options.max_articulation_change))
        if toward is None:
            reasons.append("no --ground-truth-dir, so 'closer to ground truth' cannot be tested")
        else:
            if not toward["speed_moved_toward_truth"]:
                reasons.append("joint speed moved away from ground truth: ratio {:.3f} -> {:.3f}"
                               .format(*toward["speed_ratio_to_truth"]))
            if not toward.get("jerk_moved_toward_truth", True):
                reasons.append("jerk moved away from ground truth: ratio {:.2f} -> {:.2f}"
                               .format(*toward["jerk_ratio_to_truth"]))
        report["passed"] = not reasons
        report["gate"] = "root_channel"
        report["reasons"] = reasons
    else:
        report["passed"] = bool(speed_loss <= options.max_speed_loss)
        report["gate"] = "speed_loss"
        report["reasons"] = ([] if report["passed"] else
                             ["joint speed fell {:.1%}, more than the allowed {:.1%}".format(
                                 speed_loss, options.max_speed_loss)])
    if any("hf_root_truth" in r for r in rows):
        report["ground_truth"] = {
            "hf_root": median("hf_root_truth"),
            "median_joint_speed": median("speed_truth"),
        }
    (out_dir / "smoothing_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        print("SMOOTHING REJECTED ({}):".format(report["gate"]), file=sys.stderr)
        for reason in report["reasons"]:
            print("  " + reason, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
