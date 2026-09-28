#!/usr/bin/env python3
"""Score generated clips on the four things the headline FID cannot see.

``eval/metrics.py`` standardises each distribution by its own per-dimension mean
and std before computing FID and diversity, so ``fid_k`` and ``div_k`` are exactly
invariant to any per-dimension affine change of the feature vector -- including a
uniform rescaling of motion amplitude.  Halving every clip's motion leaves
``fid_k`` at 0.0 and ``div_k`` bit-identical.  Three of the four symptoms this
repo is chasing (inter-segment roughness, amplitude collapse, motion-energy
redistribution) are therefore unfalsifiable against the headline, which is why
they need their own instrument.

Four families, each with the control that stops it being over-read:

* ``follow_rate`` -- on frames the plan filled with a prototype (mask=1), how much
  closer the generated joint rotations sit to *that prototype* than to the same
  prototype with its frame order shuffled.  The shuffle control is the point:
  dance resembles dance, so an uncontrolled distance says nothing.  The draft is
  rebuilt, not stored -- retrieval carries no RNG and each clip records its own
  ``query_retrieval_group_id``, so the rebuild is exact rather than approximate.
  The rebuilt draft is **un-normalised before its rotations are read**: the
  library is the release's min-max scaled ``motion.npy``, whose rot6d columns
  describe no rotation at all until inverted.  Before 2026-08-18 they were not,
  and the consequence was a metric with a ceiling of 0.456 rather than 1.0 --
  a generation that *was* the draft scored 0.456.  The published wild numbers
  (2.98% / 3.55% on the 24 held-out clips) become 5.33% / 5.86% once the two
  sides share a space; the conclusion they were quoted for -- that the plan
  reaches a single-digit percentage of the motion -- is unchanged, because the
  ceiling moved with them.

* ``jerk`` -- roughness at two kinds of position, each against the same clip's own
  quiet frames.  ``plan_boundary`` is where the draft steps between prototypes;
  ``blend_step`` is where ``infer_atomic._blend_weights`` produces a discontinuity.
  That second one exists only when ``2*(window - stride) > window``: the two
  linspace slices then overwrite each other and the weight jumps mid-window
  (window 340 / stride 75: 0.2820 -> 0.9962 between local frames 74 and 75,
  against a typical per-frame step of 0.0038).  At stride >= window/2 the
  positions list is empty and the family reports ``null`` rather than 1.0 -- an
  absent mechanism is not a ratio of one.

* ``amplitude`` -- world coordinates, deliberately NOT ``canonical_pose``.  That
  helper removes translation, facing and shoulder-width scale, which is exactly
  what the gallery camera shows; measuring amplitude through it once produced
  "amplitude is fine, 3% from ground truth" and the world-coordinate re-measure
  overturned it.  Two columns here are biased in ground truth's favour and are
  labelled as such: wild ground truth is a GVHMR reconstruction whose root drifts
  more than the generations do, which inflates its net displacement and root
  height span.  ``reach_span`` and ``turn_rate`` do not have that bias.

* ``speed_profile`` -- per-joint median speed, reported under BOTH of the repo's
  two conventions because they disagree by 1.4x on ground truth and 1.07x on
  generated motion, and a claim quoted under one of them ("generated is 41-46%
  faster") evaporates under the other.  The ratio between the two conventions is
  itself the finding: ground truth has a long tail (whipped extremities) and the
  generations do not.

The gate is a cost gate, in the same shape as ``smooth_generated_motion.py``: any
run compared against a ``--baseline-run`` must not buy its roughness improvement
by slowing the dance down.  There is no setting at which this tool reports an
improvement without also reporting what it spent.

**One criterion was removed on 2026-08-18 because it fired on its own control.**
"Median joint speed is below ground truth" failed the 340 arm scored against
*itself* at a different sampling seed: 1.0767 -> 1.0002 across seeds 20260817
and 20260818, on the same 24 clips at the same stride, which is 7.1% of movement
from re-drawing alone and lands just under ground truth's 1.0631.  It is now
reported as ``below_ground_truth`` rather than a failure.  The 7.1% also sizes
the surviving criterion: a 10% paired-baseline threshold sits barely above the
noise, so it separates a 23% loss (which is what ``--draft-noise-ratio 0.05``
cost) and says nothing about a 2% one.

Usage::

    python3 tools/score_generation_diagnostics.py \\
        --run runs/t_stride_sweep/s170_seed20260817 \\
        --baseline-run runs/vis_wild_20260817b/arm340_seed20260817 \\
        --ground-truth-dir runs/wild_v4_acct_gt_eval/motion \\
        --data-root /dev/shm/atomicdance-acct/release_w2_340 \\
        --output runs/t_stride_sweep/score_s170_seed20260817.json
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

FPS = 30.0
MOTION_DIM = 151
# SMPL joint indices used by name.  vis.SMPLSkeleton's forward returns [T, 24, 3].
PELVIS = 0
HEAD = 15
WRISTS = (20, 21)
ANKLES = (7, 8)
SHOULDERS = (16, 17)
BOUNDARY_HALF_WIDTH = 2


# ----------------------------------------------------------------- geometry --


def _joint_speeds(full_pose: np.ndarray) -> np.ndarray:
    """[T,24,3] world positions -> [T-1,24] per-joint speed in m/s."""
    return np.linalg.norm(np.diff(full_pose, axis=0), axis=-1) * FPS


def _jerk(full_pose: np.ndarray) -> np.ndarray:
    """Per-frame jerk magnitude, averaged over joints; length T-3."""
    accel = np.diff(full_pose, n=2, axis=0) * (FPS ** 2)
    jerk = np.diff(accel, axis=0) * FPS
    return np.linalg.norm(jerk, axis=-1).mean(axis=1)


def _shoulder_width(full_pose: np.ndarray) -> float:
    width = np.linalg.norm(
        full_pose[:, SHOULDERS[0]] - full_pose[:, SHOULDERS[1]], axis=-1
    )
    value = float(np.median(width))
    # A degenerate skeleton would silently turn every shoulder-width-relative
    # column into an enormous number; refuse instead.
    if not np.isfinite(value) or value <= 1e-3:
        raise ValueError("shoulder width is degenerate ({}); cannot normalise reach".format(value))
    return value


def _turn_rate_deg_s(full_pose: np.ndarray) -> float:
    """Median absolute yaw rate of the shoulder line, in degrees per second."""
    vec = full_pose[:, SHOULDERS[0], :2] - full_pose[:, SHOULDERS[1], :2]
    angle = np.unwrap(np.arctan2(vec[:, 1], vec[:, 0]))
    return float(np.median(np.abs(np.diff(angle))) * FPS * 180.0 / np.pi)


def amplitude_columns(full_pose: np.ndarray) -> dict:
    """World-coordinate amplitude.  No de-translation, no de-rotation, no rescale."""
    root = full_pose[:, PELVIS]
    floor_steps = np.linalg.norm(np.diff(root[:, :2], axis=0), axis=-1)
    reach = np.linalg.norm(full_pose - root[:, None, :], axis=-1)
    extremities = np.concatenate(
        [reach[:, list(WRISTS)], reach[:, list(ANKLES)]], axis=1
    ).max(axis=1)
    width = _shoulder_width(full_pose)
    seconds = len(full_pose) / FPS
    net = float(np.linalg.norm(root[-1, :2] - root[0, :2]))
    path = float(floor_steps.sum())
    return {
        "net_displacement_m": net,
        "floor_path_m": path,
        "floor_path_m_per_s": path / seconds,
        # Path efficiency separates "travels" from "jitters in place"; it needs no
        # normalisation choice and both of its parts are in the same units.
        "path_efficiency": net / path if path > 1e-9 else 0.0,
        "root_height_span_m": float(
            np.percentile(root[:, 2], 95) - np.percentile(root[:, 2], 5)
        ),
        "turn_rate_deg_s": _turn_rate_deg_s(full_pose),
        "reach_p95_shoulders": float(np.percentile(extremities, 95) / width),
        "reach_span_shoulders": float(
            (np.percentile(extremities, 95) - np.percentile(extremities, 5)) / width
        ),
        "seconds": seconds,
    }


def speed_profile(full_pose: np.ndarray) -> dict:
    """Both of the repo's speed conventions, plus the across-joint contrast."""
    speeds = _joint_speeds(full_pose)
    per_joint = np.median(speeds, axis=0)
    pelvis = float(per_joint[PELVIS])
    return {
        # tools/build_dance_gallery.py:333 -- median over the whole [T-1,24] array
        "median_of_all": float(np.median(speeds)),
        # tools/smooth_generated_motion.py:70 -- mean over joints, then median
        "median_of_frame_mean": float(np.median(speeds.mean(axis=1))),
        "wrist": float(per_joint[list(WRISTS)].mean()),
        "ankle": float(per_joint[list(ANKLES)].mean()),
        "pelvis": pelvis,
        "head": float(per_joint[HEAD]),
        "wrist_over_pelvis": float(per_joint[list(WRISTS)].mean() / pelvis)
        if pelvis > 1e-9 else float("nan"),
        # The single number that says "flat" rather than "small": ground truth
        # whips the extremities while the trunk holds, so its across-joint spread
        # is large.  A model that moves every joint at a middling speed has a
        # small one, at any overall amplitude.
        "across_joint_cv": float(per_joint.std() / per_joint.mean()),
    }


# ------------------------------------------------------------- jerk anchors --


def window_starts(length: int, window: int, stride: int) -> list:
    """Mirror of infer_atomic._window_starts, including the snapped final window."""
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    final = length - window
    if starts[-1] != final:
        starts.append(final)
    return starts


def blend_step_frames(length: int, window: int, stride: int) -> list:
    """Absolute frames carrying the _blend_weights discontinuity.

    ``_blend_weights`` writes ``weights[:overlap]`` as a 0->1 ramp and then
    ``weights[-overlap:]`` as a 1->0 ramp.  With ``2*overlap > window`` the second
    assignment overwrites the tail of the first and the profile jumps at local
    index ``window - overlap == stride``.  Only windows that are neither first nor
    last receive both ramps, so only they carry the step.
    """
    overlap = window - stride
    if overlap <= 0 or 2 * overlap <= window:
        return []
    starts = window_starts(length, window, stride)
    if len(starts) <= 2:
        return []
    return [start + stride for start in starts[1:-1] if start + stride < length]


def plan_boundary_frames(labels: np.ndarray) -> list:
    return (np.nonzero(np.diff(labels) != 0)[0] + 1).tolist()


def _ratio_at(jerk: np.ndarray, frames, half=BOUNDARY_HALF_WIDTH):
    """Median jerk within +-half of ``frames``, over median jerk elsewhere.

    Returns None when the anchor set is empty -- an absent mechanism has no
    ratio, and reporting 1.0 for it would read as "measured, and fine".
    """
    if len(jerk) == 0 or not len(frames):
        return None
    mask = np.zeros(len(jerk), dtype=bool)
    for frame in frames:
        lo = max(0, frame - half)
        hi = min(len(jerk), frame + half + 1)
        if lo < hi:
            mask[lo:hi] = True
    if not mask.any() or mask.all():
        return None
    near = float(np.median(jerk[mask]))
    far = float(np.median(jerk[~mask]))
    return near / far if far > 1e-12 else None


# ------------------------------------------------------------- follow rate --


def _rotations_from_151(motion: torch.Tensor, normalizer_path=None):
    """151-D -> [T,24,3] axis-angle, using the generator's own decode.

    ``normalizer_path`` is not optional in practice: the prototype library is
    the release's *normalised* ``motion.npy``, and min-max scaling is applied
    per dimension, so its rot6d columns no longer describe a rotation.  Running
    ``ax_from_6v`` on them yields a rotation field of some other body -- which is
    the conclusion ``eval_r_precision.unnormalize`` already reached for the same
    array, and ``infer_atomic.decode_motion`` inverts before decoding for the
    same reason.

    Measured cost of not inverting: hand this metric a generation that *is* the
    draft, and it scores 0.456 instead of 1.0 (24 clips, median).  So the
    published follow rate was a fraction of a ceiling it could never reach, and
    ``tests/test_score_generation_diagnostics.py`` pins that ceiling at 1.0.
    """
    if normalizer_path is not None:
        from infer_atomic import unnormalize_motion

        motion = unnormalize_motion(motion, normalizer_path)
    from dataset.quaternion import ax_from_6v

    values = motion[:, 4:]
    return ax_from_6v(values[:, 3:].reshape(-1, 24, 6))


def _geodesic_rad(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean geodesic angle between two [T,24,3] axis-angle fields.

    Axis-angle differences wrap at +-pi, so a plain L2 on the vectors reports a
    large distance for two nearly identical rotations.  Compare through rotation
    matrices instead.
    """
    from pytorch3d_compat import axis_angle_to_matrix  # noqa: F401  (optional)


def _axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """Rodrigues, batched over [..., 3]."""
    theta = aa.norm(dim=-1, keepdim=True)
    axis = aa / theta.clamp(min=1e-8)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = torch.zeros_like(x)
    K = torch.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero], dim=-1
    ).reshape(aa.shape[:-1] + (3, 3))
    eye = torch.eye(3, dtype=aa.dtype, device=aa.device).expand_as(K)
    s = torch.sin(theta).unsqueeze(-1)
    c = torch.cos(theta).unsqueeze(-1)
    return eye + s * K + (1.0 - c) * (K @ K)


def geodesic_rad(a: torch.Tensor, b: torch.Tensor) -> float:
    Ra = _axis_angle_to_matrix(a)
    Rb = _axis_angle_to_matrix(b)
    rel = Ra.transpose(-1, -2) @ Rb
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    return float(torch.arccos(cos).mean())


def follow_rate(clip: dict, library, feature_dim: int, seed: int,
                normalizer_path=None) -> dict:
    """How much of the prototype reached the motion, against a shuffle control."""
    labels = torch.from_numpy(np.asarray(clip["atomic_labels"]).astype(np.int64))
    group = (clip.get("prototype_retrieval") or {}).get("query_retrieval_group_id")
    exclude = (group,) if group else ()
    draft, mask = library.build_draft(
        labels, feature_dim, exclude_retrieval_group_ids=exclude, allow_missing=True
    )
    keep = mask[:, 0] > 0
    if int(keep.sum()) < 2:
        return {"frames": int(keep.sum()), "follow_rate": None,
                "reason": "fewer than two prototype frames"}
    proto = _rotations_from_151(draft, normalizer_path)
    generated = torch.from_numpy(
        np.asarray(clip["smpl_poses"], dtype=np.float32)
    ).reshape(-1, 24, 3)
    n = min(len(proto), len(generated), len(keep))
    keep = keep[:n]
    proto, generated = proto[:n][keep], generated[:n][keep]

    honest = geodesic_rad(generated, proto)
    # The control shuffles the prototype's frame order: same poses, same marginal
    # distribution, no temporal correspondence.  Anything the generation gets
    # from "this prototype at this moment" shows up as honest < control.
    rng = np.random.default_rng(seed)
    order = torch.from_numpy(rng.permutation(len(proto)))
    control = geodesic_rad(generated, proto[order])
    return {
        "frames": int(keep.sum()),
        "distance_rad": honest,
        "shuffled_control_rad": control,
        "follow_rate": (control - honest) / control if control > 1e-9 else None,
    }


# ------------------------------------------------------------------ driver --


def _load_clips(directory: pathlib.Path) -> dict:
    return {path.stem: path for path in sorted(directory.glob("*.pkl"))}


def _median(values):
    clean = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(clean)) if clean else None


def score_run(run: pathlib.Path, ground_truth_dir, data_root, window, stride,
              seed, skip_follow=False) -> dict:
    clips = _load_clips(run)
    if not clips:
        raise ValueError("no .pkl under {}".format(run))
    library = None
    normalizer_path = None
    feature_dim = MOTION_DIM
    if not skip_follow:
        from infer_atomic import IndexedAtomicMotionLibrary

        library = IndexedAtomicMotionLibrary(str(data_root))
        # The library hands out normalised motion; the generation's smpl_poses
        # are real axis-angle.  Comparing them needs the release's own inverse,
        # not a second copy of the arithmetic.
        normalizer_path = pathlib.Path(data_root) / "normalizer.pt"
        if not normalizer_path.is_file():
            raise ValueError(
                "follow rate needs the release normalizer to put the prototype "
                "and the generation in one space: {} is missing".format(normalizer_path)
            )

    gt = _load_clips(pathlib.Path(ground_truth_dir)) if ground_truth_dir else {}
    per_clip, rows_gt = [], []
    for name, path in clips.items():
        with open(path, "rb") as handle:
            clip = pickle.load(handle)
        pose = np.asarray(clip["full_pose"], dtype=np.float64)
        labels = np.asarray(clip["atomic_labels"])
        jerk = _jerk(pose)
        row = {
            "name": name,
            "frames": int(len(pose)),
            "amplitude": amplitude_columns(pose),
            "speed": speed_profile(pose),
            "jerk": {
                "median": float(np.median(jerk)),
                "plan_boundary_ratio": _ratio_at(jerk, plan_boundary_frames(labels)),
                "blend_step_ratio": _ratio_at(
                    jerk, blend_step_frames(len(pose), window, stride)
                ),
                "blend_step_count": len(blend_step_frames(len(pose), window, stride)),
                "plan_boundary_count": len(plan_boundary_frames(labels)),
            },
            "transition_frame_share": float((labels == 0).mean()),
        }
        if library is not None:
            row["follow"] = follow_rate(clip, library, feature_dim, seed,
                                        normalizer_path)
        per_clip.append(row)
        if name in gt:
            with open(gt[name], "rb") as handle:
                truth = pickle.load(handle)
            tp = np.asarray(truth["full_pose"], dtype=np.float64)
            tj = _jerk(tp)
            rows_gt.append({
                "amplitude": amplitude_columns(tp),
                "speed": speed_profile(tp),
                "jerk": {"median": float(np.median(tj))},
            })

    def agg(rows, group, key):
        return _median([r[group][key] for r in rows if r.get(group)])

    summary = {
        "clips": len(per_clip),
        "amplitude": {k: agg(per_clip, "amplitude", k)
                      for k in per_clip[0]["amplitude"]},
        "speed": {k: agg(per_clip, "speed", k) for k in per_clip[0]["speed"]},
        "jerk": {k: agg(per_clip, "jerk", k) for k in per_clip[0]["jerk"]},
        "transition_frame_share": _median(
            [r["transition_frame_share"] for r in per_clip]
        ),
    }
    if library is not None:
        summary["follow_rate"] = _median(
            [r["follow"].get("follow_rate") for r in per_clip if r.get("follow")]
        )
        summary["follow_distance_rad"] = _median(
            [r["follow"].get("distance_rad") for r in per_clip if r.get("follow")]
        )
        summary["follow_shuffled_control_rad"] = _median(
            [r["follow"].get("shuffled_control_rad") for r in per_clip if r.get("follow")]
        )
    ground_truth = None
    if rows_gt:
        ground_truth = {
            "clips": len(rows_gt),
            "amplitude": {k: agg(rows_gt, "amplitude", k)
                          for k in rows_gt[0]["amplitude"]},
            "speed": {k: agg(rows_gt, "speed", k) for k in rows_gt[0]["speed"]},
            "jerk": {"median": agg(rows_gt, "jerk", "median")},
            "bias_note": (
                "net_displacement_m and root_height_span_m are biased in ground "
                "truth's favour: wild ground truth is a GVHMR reconstruction whose "
                "root drifts more than the generations do.  reach_span_shoulders "
                "and turn_rate_deg_s do not carry that bias."
            ),
        }
    return {
        "run": str(run), "window": window, "stride": stride,
        "summary": summary, "ground_truth": ground_truth, "per_clip": per_clip,
    }


def apply_gate(report: dict, baseline: dict, max_speed_loss: float) -> dict:
    """Roughness gains must not be bought with the dance's speed."""
    a = report["summary"]["speed"]["median_of_frame_mean"]
    b = baseline["summary"]["speed"]["median_of_frame_mean"]
    loss = (b - a) / b if b else 0.0
    failures = []
    if loss > max_speed_loss:
        failures.append(
            "median joint speed fell {:.2%} against the baseline, above the "
            "permitted {:.2%}: a roughness improvement bought this way is the "
            "dance being slowed down".format(loss, max_speed_loss)
        )
    # "Slower than ground truth" was a second failure here until 2026-08-18 and
    # is now an observation, because it fires on the control: the *same*
    # checkpoint, same 24 clips, same stride, re-run at seed 20260818 instead of
    # 20260817, moves median joint speed from 1.0767 to 1.0002 -- 7.1% -- and
    # lands 0.5% under ground truth's 1.0631.  A criterion that fails an arm
    # against itself cannot separate two arms, and the repo's own rule is that a
    # gate which fires without a cause is worse than no gate.  The paired
    # comparison above survives, and the number it is compared against is
    # reported so the threshold can be read against the noise it sits in.
    gt = (report.get("ground_truth") or {}).get("speed", {}).get("median_of_frame_mean")
    return {
        "baseline": baseline["run"],
        "speed_before": b, "speed_after": a,
        "speed_loss_fraction": loss, "max_speed_loss": max_speed_loss,
        "ground_truth_speed": gt,
        "below_ground_truth": bool(gt and a < gt),
        "same_arm_reseed_speed_spread": 0.071,
        "passed": not failures, "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=pathlib.Path, required=True)
    parser.add_argument("--baseline-run", type=pathlib.Path, default=None)
    parser.add_argument("--ground-truth-dir", type=pathlib.Path, default=None)
    parser.add_argument("--data-root", type=pathlib.Path, default=None,
                        help="release root holding the prototype library; required "
                             "unless --skip-follow-rate")
    parser.add_argument("--window", type=int, default=340,
                        help="completion window the run was produced with")
    parser.add_argument("--stride", type=int, required=True,
                        help="completion stride the run was produced with; the "
                             "blend-step anchors are a function of it")
    parser.add_argument("--baseline-stride", type=int, default=None,
                        help="defaults to --stride")
    parser.add_argument("--seed", type=int, default=20260818,
                        help="seed for the follow-rate shuffle control only")
    parser.add_argument("--max-speed-loss", type=float, default=0.10)
    parser.add_argument("--skip-follow-rate", action="store_true")
    parser.add_argument("--output", type=pathlib.Path, default=None)
    args = parser.parse_args()

    if not args.skip_follow_rate and args.data_root is None:
        parser.error("--data-root is required unless --skip-follow-rate")

    report = score_run(args.run, args.ground_truth_dir, args.data_root,
                       args.window, args.stride, args.seed, args.skip_follow_rate)
    if args.baseline_run:
        baseline = score_run(args.baseline_run, args.ground_truth_dir, args.data_root,
                             args.window,
                             args.baseline_stride if args.baseline_stride is not None
                             else args.stride,
                             args.seed, args.skip_follow_rate)
        report["gate"] = apply_gate(report, baseline, args.max_speed_loss)
        report["baseline_summary"] = baseline["summary"]

    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    slim = {k: v for k, v in report.items() if k != "per_clip"}
    print(json.dumps(slim, indent=2, ensure_ascii=False))
    if report.get("gate") and not report["gate"]["passed"]:
        for failure in report["gate"]["failures"]:
            print("GATE FAILED: {}".format(failure), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
