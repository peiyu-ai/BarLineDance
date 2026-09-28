"""``--face-camera``: turn the dance back toward the lens without flattening it.

WHY.  The operator, 2026-09-08, on the retrained rhythm planner: "画面的确有长时间
背对或者侧对的情况".  Share of frames NOT facing the camera -- profile counts,
which is half of what was named -- is 12.3% for ground truth and 29.9% for that
arm, and its MEDIAN longest continuous off-camera span is 1.5 s against ground
truth's 0.3 s.

WHERE IT IS FIXED, and why not elsewhere.  Measured before writing any of this:
the DRAFT is already 32.3% off camera and the completion brings it to 29.9%, so
the completion is not the source; and ``--draft-facing-anchor`` pulls toward the
clip's own OPENING yaw, which is itself off camera on 30% of drafts against
ground truth's 10%, so raising it locks those in.  Hence a correction on the
finished clip against a target that is known rather than estimated: forward
(0, -1) is where ``render_avatar_video --view front`` puts the lens.

THE POSITIVE CONTROL is ``test_a_clip_turned_ninety_degrees_is_brought_back``:
a clip built facing sideways must come back facing the lens.  Without it a
do-nothing implementation passes everything else here.

THE INVARIANT THAT MATTERS MOST is
``test_the_global_orient_carries_the_same_rotation_as_the_joints``: the renderer
draws ``smpl_poses``/``smpl_trans`` and uses ``full_pose`` only as a GATE, so a
correction that turned the joints and left the global orient behind would make
the mesh and the joints disagree and the render would be refused.
"""
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.rotation_ops import axis_angle_to_matrix
from infer_atomic import FACING_CAMERA_YAW, _body_forward_yaw, face_camera

FRAMES = 240


def _clip(yaw_series, root=None):
    """A minimal result whose hips encode ``yaw_series`` exactly.

    ``full_pose[:, 0]`` is bitwise ``smpl_trans`` in real artifacts, so the
    fixture keeps that true -- the correction's root-pivot half relies on it.
    """
    yaw_series = np.asarray(yaw_series, dtype=np.float64)
    joints = np.zeros((len(yaw_series), 24, 3))
    if root is None:
        root = np.zeros((len(yaw_series), 3))
    joints[:, 0] = root
    # hip axis is +90 deg from forward, so forward = (cos yaw, sin yaw)
    across = np.stack([np.sin(yaw_series), -np.cos(yaw_series)], axis=1)
    joints[:, 1, :2] = root[:, :2] - 0.1 * across
    joints[:, 2, :2] = root[:, :2] + 0.1 * across
    joints[:, 3, :2] = root[:, :2] + 0.5 * np.stack(
        [np.cos(yaw_series), np.sin(yaw_series)], axis=1)      # a marker out front
    return {"full_pose": joints, "smpl_trans": root.copy(),
            "smpl_poses": np.zeros((len(yaw_series), 72))}


def test_a_clip_turned_ninety_degrees_is_brought_back():
    """POSITIVE CONTROL: sideways in, facing the lens out."""
    sideways = np.full(FRAMES, FACING_CAMERA_YAW + math.pi / 2.0)
    out = face_camera(_clip(sideways), strength=0.0)
    got = _body_forward_yaw(np.asarray(out["full_pose"]))
    assert np.allclose(np.sin(got - FACING_CAMERA_YAW), 0.0, atol=1e-9)
    assert out["facing_camera"]["off_camera_share"] == 0.0


def test_the_rigid_half_alone_cannot_change_the_dance():
    """strength 0 is a world rotation: every root-relative distance survives."""
    rng = np.random.default_rng(0)
    yaw = FACING_CAMERA_YAW + 1.2 + 0.4 * np.sin(np.linspace(0, 8, FRAMES))
    root = np.cumsum(rng.normal(scale=0.01, size=(FRAMES, 3)), axis=0)
    before = _clip(yaw, root)
    after = face_camera(before, strength=0.0)
    def shape(joints):
        j = np.asarray(joints)
        return np.linalg.norm(j - j[:, :1, :], axis=2)
    assert np.allclose(shape(before["full_pose"]), shape(after["full_pose"]), atol=1e-9)
    # and the root PATH is carried along rather than left behind
    step_before = np.diff(np.asarray(before["smpl_trans"])[:, :2], axis=0)
    step_after = np.diff(np.asarray(after["smpl_trans"])[:, :2], axis=0)
    assert np.allclose(np.linalg.norm(step_before, axis=1),
                       np.linalg.norm(step_after, axis=1), atol=1e-9)


def test_the_slow_half_turns_the_body_without_moving_the_root():
    yaw = FACING_CAMERA_YAW + np.linspace(0.0, 2.0, FRAMES)
    root = np.stack([np.linspace(0, 1, FRAMES), np.zeros(FRAMES), np.zeros(FRAMES)], 1)
    out = face_camera(_clip(yaw, root), strength=0.6, window_seconds=1.0)
    joints = np.asarray(out["full_pose"])
    # rigid part rotates the path; the slow part must not move it again, so the
    # root is still bitwise full_pose[:, 0]
    assert np.allclose(joints[:, 0], np.asarray(out["smpl_trans"]), atol=1e-9)


def test_a_sustained_deviation_is_pulled_back_and_a_brief_one_is_not():
    """The window is the whole point: slow drift yields, fast turns survive."""
    base = np.full(FRAMES, FACING_CAMERA_YAW)
    sustained = base.copy(); sustained[60:180] += 1.5          # 4 s off camera
    brief = base.copy(); brief[100:110] += 1.5                 # 0.33 s off camera
    kw = dict(strength=0.8, window_seconds=2.0)
    slow_out = face_camera(_clip(sustained), **kw)
    fast_out = face_camera(_clip(brief), **kw)
    def excursion(res, lo, hi):
        y = _body_forward_yaw(np.asarray(res["full_pose"]))
        return float(np.abs(np.arctan2(np.sin(y - FACING_CAMERA_YAW),
                                       np.cos(y - FACING_CAMERA_YAW)))[lo:hi].mean())
    kept_slow = excursion(slow_out, 60, 180)
    kept_fast = excursion(fast_out, 100, 110)
    assert kept_slow < 0.9, kept_slow          # the 4 s stretch is pulled back
    assert kept_fast > 1.1, kept_fast          # the 0.33 s turn is left alone


def test_the_global_orient_carries_the_same_rotation_as_the_joints():
    """The renderer draws smpl_poses and gates on full_pose; they must agree."""
    rng = np.random.default_rng(1)
    yaw = FACING_CAMERA_YAW + 0.9 + 0.5 * np.sin(np.linspace(0, 6, FRAMES))
    clip = _clip(yaw)
    clip["smpl_poses"][:, :3] = rng.normal(scale=0.3, size=(FRAMES, 3))
    out = face_camera(clip, strength=0.4, window_seconds=1.5)

    before = axis_angle_to_matrix(torch.as_tensor(clip["smpl_poses"][:, :3]))
    after = axis_angle_to_matrix(torch.as_tensor(out["smpl_poses"][:, :3]))
    applied = (after @ before.transpose(1, 2)).numpy()
    # every applied rotation must be a pure rotation about z
    assert np.allclose(applied[:, 2, 2], 1.0, atol=1e-8)
    orient_angle = np.arctan2(applied[:, 1, 0], applied[:, 0, 0])

    joint_angle = (_body_forward_yaw(np.asarray(out["full_pose"]))
                   - _body_forward_yaw(np.asarray(clip["full_pose"])))
    wrapped = np.arctan2(np.sin(orient_angle - joint_angle),
                         np.cos(orient_angle - joint_angle))
    assert np.abs(wrapped).max() < 1e-7, float(np.abs(wrapped).max())


def test_it_does_not_mutate_its_input():
    yaw = FACING_CAMERA_YAW + np.linspace(0, 1.0, FRAMES)
    clip = _clip(yaw)
    keep = np.asarray(clip["full_pose"]).copy()
    face_camera(clip, strength=0.5)
    assert np.array_equal(keep, np.asarray(clip["full_pose"]))
