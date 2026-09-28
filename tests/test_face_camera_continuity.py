"""infer_atomic.face_camera must not snap the body around.

The operator, 2026-09-14, on a floornorm render: "会偶尔出现一次人体旋转的跳变,
视觉上看着不连续,缺帧".  Measured on the 20 eval clips with a threshold ground
truth never crosses (its worst frame-to-frame yaw change is 25.2 degrees):
ground truth 0/20 clips and 0 spikes above 30 deg/frame, the shipped arm 8/20
and 12 spikes, worst 115 degrees -- in ONE frame, i.e. 1/30 s.

THE MECHANISM, and it is arithmetic.  ``face_camera`` low-passes the UNWRAPPED
yaw, so ``smoothed`` is continuous and may leave (-pi, pi].  It then takes the
error against the camera target and re-wraps it PER SAMPLE with
``arctan2(sin, cos)``.  Every time the smoothed yaw crosses an odd multiple of
pi the wrapped error jumps by 2*pi, so the correction it drives jumps by
``2*pi*strength`` -- 252 degrees at the shipped strength 0.7, which an unwrapped
reading folds to 360-252 = 108, exactly the size of the observed spikes.

These tests do not assert the correction is SMALL -- turning the dancer back
toward the lens is the whole point and the docstring is explicit that it must
not drive off-camera time to zero.  They assert it is CONTINUOUS.
"""
import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import infer_atomic as IA  # noqa: E402

FPS = 30.0
SPIKE_DEGREES = 30.0     # above anything ground truth does (its max is 25.2)


def _turning_clip(frames=600, turns=1.25, seed=0):
    """A body that turns steadily, so the smoothed yaw must cross +-pi.

    A dancer who slowly turns more than half a circle is not exotic -- the T
    line's ground truth turns a median 919 degrees over a clip -- and it is the
    only thing needed to put ``smoothed`` outside (-pi, pi].
    """
    rng = np.random.default_rng(seed)
    joints = np.zeros((frames, 24, 3))
    yaw = np.linspace(0.0, 2 * math.pi * turns, frames)
    half = 0.06
    # HIP_JOINTS[0] is the left hip, [1] the right; the pair defines the facing.
    left, right = IA.HIP_JOINTS
    joints[:, left, 0] = -half * np.cos(yaw)
    joints[:, left, 1] = -half * np.sin(yaw)
    joints[:, right, 0] = half * np.cos(yaw)
    joints[:, right, 1] = half * np.sin(yaw)
    joints[:, :, 2] = 0.9
    joints += 1e-4 * rng.standard_normal(joints.shape)
    return {
        "full_pose": joints,
        "smpl_trans": np.zeros((frames, 3)),
        "smpl_poses": np.zeros((frames, 72)),
    }


def _worst_step(joints):
    yaw = np.unwrap(IA._body_forward_yaw(np.asarray(joints, dtype=np.float64)))
    return float(np.max(np.abs(np.degrees(np.diff(yaw)))))


def test_the_input_itself_is_smooth():
    """Positive control: the synthetic clip has no snap before the correction."""
    clip = _turning_clip()
    assert _worst_step(clip["full_pose"]) < 2.0


@pytest.mark.parametrize("strength", [0.35, 0.7, 1.0])
def test_no_yaw_snap_after_face_camera(strength):
    clip = _turning_clip()
    out = IA.face_camera(clip, strength, window_seconds=3.0, fps=FPS)
    worst = _worst_step(out["full_pose"])
    assert worst < SPIKE_DEGREES, (
        "face_camera introduced a {:.1f} deg/frame jump at strength {}; the "
        "wrapped error makes the correction discontinuous".format(worst, strength))


def test_the_correction_still_turns_the_dancer_toward_the_camera():
    """The fix must not be 'stop correcting'.  Off-camera share has to fall."""
    clip = _turning_clip()
    before = float((np.sin(IA._body_forward_yaw(clip["full_pose"])) > -0.5).mean())
    out = IA.face_camera(clip, 0.7, window_seconds=3.0, fps=FPS)
    assert out["facing_camera"]["off_camera_share"] < before


def test_global_orient_turns_as_smoothly_as_the_joints():
    """The renderer gates on full_pose but DRAWS smpl_poses, so a snap that the
    joints no longer have must not survive in the global orient either.

    Compared as MATRICES, not as axis-angle: axis-angle flips sign at pi and a
    test that read its components would report a jump the rotation does not
    have -- which is how a representation gets blamed for a bug.
    """
    import torch
    from dataset.rotation_ops import axis_angle_to_matrix

    clip = _turning_clip(frames=300)
    out = IA.face_camera(clip, 0.7, window_seconds=3.0, fps=FPS)
    orient = torch.as_tensor(np.asarray(out["smpl_poses"])[:, :3],
                             dtype=torch.float64)
    matrices = axis_angle_to_matrix(orient).numpy()
    relative = matrices[1:] @ np.transpose(matrices[:-1], (0, 2, 1))
    trace = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    worst = float(np.max(np.degrees(np.arccos(trace))))
    assert worst < SPIKE_DEGREES, (
        "the global orient turns {:.1f} deg in one frame".format(worst))
