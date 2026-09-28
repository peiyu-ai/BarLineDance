"""render2d/aapose_video -- the pose PICTURE has to look like what the detector draws.

SteadyDancer was trained on skeletons drawn from a real 2D detector's output, so
every place our synthesised picture departs from that distribution is a cue the
model reads in a way nobody chose.  The 2026-09-21 front/back census found four
such departures, each measured against the real detector (DWPose on 781k corpus
frames, ViTPose-H on the fixed ten):

  * the far EAR vanished from |yaw| 30 deg (detector keeps it >= 79% at every
    angle) -- a head the model has only seen on someone turning away;
  * the HANDS were rotated the same way, so the left hand was always palm-to-
    camera and the right always back-of-hand, on every frame;
  * the HIPS were SMPL's femoral heads, 0.32 of the shoulder width against the
    detector's 0.76 -- a pelvis that reads as turned 55-65 deg;
  * the turned FACE collapsed (ear spread to 0.03 L) where the detector's stays
    at >= 0.13 L and slides to the edge of the head.

These tests pin the calibrated modes to those measurements.  They say nothing
about whether the RENDER improves -- that is judged on the ten test clips.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from render2d.aapose_video import (  # noqa: E402
    face_confidence, face_confidence_calibrated, face_points_calibrated,
    retarget_body, synth_hands_v2, to_aapose20, shoulder_yaw)
from render2d.project_pose_2d import SMPL, project  # noqa: E402


def standing(frames=1, yaw_deg=0.0):
    """A T-shaped standing figure facing +y (the camera) rotated by ``yaw_deg``.

    Positive yaw turns the chest normal toward +x, which the projection puts on
    the image LEFT.
    """
    body = {
        # Hips at SMPL's height: 0.439 of neck-to-ankle below the neck.
        "pelvis": (0.0, 0.0, 0.85), "l_hip": (-0.09, 0.0, 0.832), "r_hip": (0.09, 0.0, 0.832),
        "spine1": (0.0, 0.0, 1.05), "l_knee": (-0.1, 0.0, 0.5), "r_knee": (0.1, 0.0, 0.5),
        "spine2": (0.0, 0.0, 1.18), "l_ankle": (-0.1, 0.0, 0.08), "r_ankle": (0.1, 0.0, 0.08),
        "spine3": (0.0, 0.0, 1.3), "l_foot": (-0.12, 0.12, 0.02), "r_foot": (0.12, 0.12, 0.02),
        "neck": (0.0, 0.0, 1.45), "l_collar": (-0.08, 0.0, 1.42), "r_collar": (0.08, 0.0, 1.42),
        "head": (0.0, 0.0, 1.58), "l_shoulder": (-0.19, 0.0, 1.42), "r_shoulder": (0.19, 0.0, 1.42),
        "l_elbow": (-0.25, 0.0, 1.15), "r_elbow": (0.25, 0.0, 1.15),
        "l_wrist": (-0.28, 0.0, 0.9), "r_wrist": (0.28, 0.0, 0.9),
        "l_hand": (-0.29, 0.0, 0.82), "r_hand": (0.29, 0.0, 0.82),
    }
    # Facing +y means the person's LEFT is at -x (their right hand is at +x):
    # forward = cross(left - right, up) = cross(-x, z) = +y.
    joints = np.zeros((frames, 24, 3))
    for name, point in body.items():
        joints[:, SMPL[name]] = point
    angle = np.radians(yaw_deg)
    # Rotating +y toward +x is a rotation by -angle about z.
    c, s = np.cos(-angle), np.sin(-angle)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return joints @ rotation.T


def test_the_test_figure_faces_the_way_it_says():
    """Guard the fixture itself: a wrong sign here would make every test below
    check the mirror image of what it claims."""
    assert abs(float(shoulder_yaw(standing())[0])) < 1e-6
    assert abs(float(shoulder_yaw(standing(yaw_deg=60))[0]) - 60.0) < 1e-6


def test_both_ears_are_drawn_where_the_geometric_band_dropped_one():
    for yaw in (45.0, 60.0, 75.0, 110.0, 150.0):
        joints = standing(frames=40, yaw_deg=yaw)
        calibrated = face_confidence_calibrated(joints)
        assert float(calibrated[:, 3:].min()) >= 0.99, yaw
    # ...and the old band really did drop it -- otherwise this test proves nothing.
    old = face_confidence(standing(frames=40, yaw_deg=60.0))
    assert float(old[:, 3:].min()) < 0.5


def test_the_nose_survives_until_the_detector_loses_it():
    """The detector's nose is drawn on 55% of FULL back views; its 50% point is
    about 172 deg.  Eyes never drop below 58%."""
    for yaw, drawn in ((90.0, True), (150.0, True), (165.0, True), (178.0, False)):
        face = face_confidence_calibrated(standing(frames=40, yaw_deg=yaw))
        assert (float(face[20, 0]) >= 0.5) is drawn, yaw
        assert float(face[:, 1:3].min()) >= 0.99, yaw


def test_the_turned_face_keeps_its_ear_spread():
    width, height = 480, 832
    for yaw in (0.0, 45.0, 90.0, 135.0):
        joints = standing(yaw_deg=yaw)
        world = to_aapose20(joints)
        uv = project(world, width, height)
        placed = face_points_calibrated(uv, joints, width, height, world) * [width, height]
        neck = placed[0, 1]
        hips = 0.5 * (placed[0, 8] + placed[0, 11])
        unit = np.linalg.norm(neck - hips)
        spread = abs(placed[0, 16, 0] - placed[0, 17, 0]) / unit
        # Real ear spread never falls below 0.13 L at any yaw.
        assert spread >= 0.129, (yaw, spread)


def test_the_face_turns_toward_the_side_the_chest_faces():
    """Positive yaw puts the chest normal on the image LEFT, so the nose must be
    left of the neck."""
    width, height = 480, 832
    joints = standing(yaw_deg=60.0)
    world = to_aapose20(joints)
    placed = face_points_calibrated(project(world, width, height), joints, width, height, world)
    assert placed[0, 0, 0] < placed[0, 1, 0]
    joints = standing(yaw_deg=-60.0)
    world = to_aapose20(joints)
    placed = face_points_calibrated(project(world, width, height), joints, width, height, world)
    assert placed[0, 0, 0] > placed[0, 1, 0]


def turning_clip(final_yaw, head_forward=0.0, frames_front=30, frames_turned=10):
    """A clip mostly facing the camera that ends turned: the medians the face
    fit uses come from the frontal part, as in a real dance."""
    front = standing(frames=frames_front)
    turned = standing(frames=frames_turned, yaw_deg=final_yaw)
    joints = np.concatenate([front, turned])
    if head_forward:
        # Forward = the chest normal; rotate the offset with each frame's body.
        for t, yaw in enumerate([0.0] * frames_front + [final_yaw] * frames_turned):
            a = np.radians(yaw)
            joints[t, SMPL["head"]] += head_forward * np.array([np.sin(a), np.cos(a), 0.0])
    return joints


def nose_toward_face_side(joints, yaw):
    width, height = 480, 832
    world = to_aapose20(joints)
    placed = face_points_calibrated(project(world, width, height), joints, width, height, world)
    placed = placed * [width, height]
    neck = placed[:, 1]
    unit = np.median(np.linalg.norm(neck - 0.5 * (placed[:, 8] + placed[:, 11]), axis=1))
    side = -1.0 if yaw > 0 else 1.0
    return (placed[-1, 0, 0] - neck[-1, 0]) * side / unit


def test_a_head_forward_of_the_shoulders_does_not_push_the_turned_face_out():
    """SMPL's head joint sits ~0.1 L ahead of the shoulder midpoint.  The first
    version's nod term carried that forward offset * sin(yaw), so a turned nose
    landed 0.30 L out against the template's 0.20 (review 2026-09-21)."""
    for yaw in (60.0, 90.0, -90.0):
        flat = nose_toward_face_side(turning_clip(yaw), yaw)
        forward = nose_toward_face_side(turning_clip(yaw, head_forward=0.06), yaw)
        assert abs(forward - flat) < 0.02, (yaw, flat, forward)


def test_the_ears_pass_through_each_other_continuously_near_profile():
    """The template's near/far ear gap flips sign near 102.6 deg; a hard floor
    swapped the ears 0.13 L in one frame there."""
    width, height = 480, 832
    yaws = np.arange(90.0, 115.0, 0.1)
    joints = np.concatenate([standing(frames=20)] + [standing(yaw_deg=y) for y in yaws])
    world = to_aapose20(joints)
    placed = face_points_calibrated(project(world, width, height), joints, width, height, world)
    placed = placed[20:] * [width, height]
    step = np.abs(np.diff(placed[:, 16:18, 0], axis=0)).max()
    unit = np.median(np.linalg.norm(placed[:, 1] - 0.5 * (placed[:, 8] + placed[:, 11]), axis=1))
    assert step / unit < 0.02, step / unit


def test_a_square_on_face_is_labelled_as_a_front_view():
    """At exactly yaw 0 the person's right eye and ear must be on the image
    LEFT, as for any frontal frame."""
    width, height = 480, 832
    joints = standing()
    world = to_aapose20(joints)
    placed = face_points_calibrated(project(world, width, height), joints, width, height, world)[0]
    assert placed[14, 0] < placed[15, 0]
    assert placed[16, 0] < placed[17, 0]


def test_the_retargeted_pelvis_is_the_detectors_width():
    width, height = 480, 832
    joints = standing()
    world, _shift = retarget_body(to_aapose20(joints))
    px = project(world, width, height)[0] * [width, height]
    shoulders = abs(px[2, 0] - px[5, 0])
    hips = abs(px[8, 0] - px[11, 0])
    neck_ankle = 0.5 * (px[10, 1] + px[13, 1]) - px[1, 1]
    assert 0.70 <= hips / shoulders <= 0.85
    hip_height = (0.5 * (px[8, 1] + px[11, 1]) - px[1, 1]) / neck_ankle
    assert 0.38 <= hip_height <= 0.42


def test_the_retarget_is_3d_so_a_turn_still_foreshortens():
    width, height = 480, 832

    def widths(yaw):
        world, _ = retarget_body(to_aapose20(standing(yaw_deg=yaw)))
        px = project(world, width, height)[0] * [width, height]
        return abs(px[2, 0] - px[5, 0]), abs(px[8, 0] - px[11, 0])

    front, turned = widths(0.0), widths(60.0)
    for a, b in zip(front, turned):
        assert abs(b / a - 0.5) < 0.05


def test_the_two_hands_are_mirror_images_and_turn_over_with_the_body():
    """Image-space handedness of a fan: the sign of cross(wrist->middle tip,
    wrist->thumb tip).  Real hands agree 64% of the time; the old fans 0%."""
    def handedness(joints):
        hands = synth_hands_v2(joints)
        signs = []
        for hand in hands:
            uv = project(np.concatenate([to_aapose20(joints), hand], axis=1), 480, 832,
                         extent=to_aapose20(joints))[0, 20:]
            axis, thumb = uv[12] - uv[0], uv[4] - uv[0]
            signs.append(np.sign(axis[0] * thumb[1] - axis[1] * thumb[0]))
        return signs

    left, right = handedness(standing())
    assert left == -right, "a frontal figure's hands must be mirror images"
    back_left, back_right = handedness(standing(yaw_deg=180.0))
    assert back_left == -left and back_right == -right, "the fans must turn over with the body"
