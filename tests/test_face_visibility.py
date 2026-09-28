"""render2d/aapose_video.face_confidence -- a face that fades, and only when it should.

Three operator reports converge here (2026-09-15): "有背身正身不连续的问题,正面
和背影切换", "头一直低着,脸不对着相机", "转身突然加速闪烁,看着画面变化不丝滑".
The face is SYNTHESISED, so every one of its cues is something this repository
chose, and the first two versions chose wrong:

  * drawn at full confidence on every frame -- a body that had turned away still
    carried a front-facing face, which is a cue that contradicts the limbs;
  * then gated on a 7-degree band, which on a turn measured at 24.6 deg/frame is
    less than one frame, so the face did not fade, it popped;
  * and with the ears sharing the nose's band, a FRONTAL ear scored exactly 0.5
    and the ears vanished on two thirds of frames, which no detector does.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from render2d.aapose_video import face_confidence  # noqa: E402
from render2d.project_pose_2d import SMPL  # noqa: E402


def turning(frames=300, turns=1.0, height=1.7):
    """A body that turns steadily through every facing, including straight away."""
    joints = np.zeros((frames, 24, 3))
    yaw = np.linspace(0.0, 2 * np.pi * turns, frames)
    half = 0.18
    across = np.stack([np.cos(yaw), np.sin(yaw), np.zeros(frames)], axis=1)
    centre = np.array([0.0, 0.0, height * 0.82])
    joints[:, SMPL["l_shoulder"]] = centre + across * half
    joints[:, SMPL["r_shoulder"]] = centre - across * half
    joints[:, SMPL["head"]] = [0, 0, height]
    joints[:, SMPL["l_hip"]] = [0, 0, height * 0.52]
    joints[:, SMPL["r_hip"]] = [0, 0, height * 0.52]
    return joints, yaw


def test_the_face_is_there_when_the_body_faces_the_lens():
    joints, yaw = turning()
    face = face_confidence(joints)
    # ``head_frame`` takes forward = cross(across, up), so with across built
    # from ``yaw`` the body faces the +y camera at yaw = pi.  Derived from the
    # code rather than assumed: the first version of this test guessed -pi/2 and
    # sampled the profile instead of the front.
    frontal = np.abs(np.angle(np.exp(1j * (yaw - np.pi)))) < np.radians(30)
    assert float(face[frontal, 0].min()) > 0.9, "the nose went out while frontal"
    # Both ears, but only while the head is SQUARE on: at the 30-degree edge of
    # the window above the far ear is already fading, which is what an ear on
    # the side of a turning head does and what a detector reports.
    square = np.abs(np.angle(np.exp(1j * (yaw - np.pi)))) < np.radians(10)
    assert float(face[square, 3:].min()) > 0.9, "an ear went out while square on"


def test_the_nose_goes_out_when_the_body_turns_away():
    joints, yaw = turning()
    face = face_confidence(joints)
    away = np.abs(np.angle(np.exp(1j * yaw))) < np.radians(25)
    assert float(face[away, 0].max()) < 0.5, "a back-facing body kept its nose"


def test_the_ears_survive_a_body_that_has_turned_away():
    """Behind a person you still see both ears; this is what separates the ear
    band from the nose band."""
    joints, yaw = turning()
    face = face_confidence(joints)
    away = np.abs(np.angle(np.exp(1j * yaw))) < np.radians(15)
    assert float(face[away, 3:].mean()) > 0.5


def test_the_fade_is_never_a_pop():
    """The defect the operator called 闪烁.  A full turn in 300 frames is slow;
    the guard is on the SHAPE of the transition, so it has to hold when the turn
    is fast too."""
    for frames in (300, 60, 30):
        joints, _ = turning(frames=frames)
        face = face_confidence(joints)
        step = float(np.max(np.abs(np.diff(face, axis=0))))
        assert step < 0.35, ("confidence jumped {:.2f} in one frame at {} frames "
                             "per turn".format(step, frames))
