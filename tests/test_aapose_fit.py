"""render2d/aapose_video.fit_to_character -- the pose is fitted, not the still.

The operator's correction was directional: "pose scale and the cartoon figure do
not match, it is too big -- align to the cartoon figure's first frame".  The
first implementation cropped the STILL to the pose; the second matched the
detector's BOX to the keypoints' bounding box, which reads 10-24% too big
because the box runs to the top of the hair while AAPose's highest keypoint is
an eye.  This one matches LANDMARK TO LANDMARK on the one span both sides have
and neither a hairstyle nor a raised arm can move: neck to ankle.
"""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from render2d.aapose_video import fit_to_character  # noqa: E402

WIDTH, HEIGHT = 480, 832
NECK, R_ANKLE, L_ANKLE = 1, 10, 13


def _clip(frames=120, seed=0):
    """A figure that moves, including arms that go overhead on some frames --
    the case a bounding-box fit gets wrong, because the highest point becomes a
    WRIST."""
    rng = np.random.default_rng(seed)
    uv = np.zeros((frames, 20, 2))
    time = np.linspace(0, 1, frames)
    for joint in range(20):
        uv[:, joint, 0] = 0.5 + 0.10 * np.sin(2 * np.pi * (time + joint / 20))
        uv[:, joint, 1] = 0.15 + 0.75 * joint / 19 + 0.02 * np.cos(4 * np.pi * time)
    uv[:, 4, 1] = 0.05 + 0.02 * np.sin(6 * np.pi * time)      # a wrist overhead
    # A face laid out the way a detector reports one: the head fit scales on the
    # ear-to-ear distance, which on the bare ladder above is a VERTICAL gap and
    # means nothing.
    neck_v = uv[:, 1, 1]
    eye_v = neck_v - 0.077
    centre_u = uv[:, 1, 0]
    uv[:, 0] = np.stack([centre_u, eye_v + 0.012], axis=1)
    uv[:, 14] = np.stack([centre_u - 0.020, eye_v], axis=1)
    uv[:, 15] = np.stack([centre_u + 0.020, eye_v], axis=1)
    uv[:, 16] = np.stack([centre_u - 0.046, eye_v + 0.009], axis=1)
    uv[:, 17] = np.stack([centre_u + 0.046, eye_v + 0.009], axis=1)
    uv += 0.002 * rng.standard_normal(uv.shape)
    return uv


FACE = [0, 14, 15, 16, 17]


def _reference(neck_v=0.515, ankle_v=0.908, neck_u=0.501, face=None):
    """A detected reference.  A FACE is always present, because the fit gives the
    head its own similarity and a reference with no face would map ours to the
    origin -- which is what the first version of this helper did."""
    body = np.zeros((20, 3))
    body[NECK] = (neck_u * WIDTH, neck_v * HEIGHT, 1.0)
    body[R_ANKLE] = (neck_u * WIDTH - 20, ankle_v * HEIGHT, 1.0)
    body[L_ANKLE] = (neck_u * WIDTH + 20, ankle_v * HEIGHT, 1.0)
    if face is None:
        eye_v = neck_v - 0.077
        face = {0: (neck_u, eye_v + 0.012), 14: (neck_u - 0.020, eye_v),
                15: (neck_u + 0.020, eye_v), 16: (neck_u - 0.046, eye_v + 0.009),
                17: (neck_u + 0.046, eye_v + 0.009)}
    for slot, (u, v) in face.items():
        body[slot] = (u * WIDTH, v * HEIGHT, 1.0)
    return body


def test_neck_to_ankle_matches_the_reference():
    uv = _clip()
    reference = _reference()
    out, fit = fit_to_character(uv, reference, WIDTH, HEIGHT)
    neck = np.median(out[:, NECK, 1])
    ankle = np.median(0.5 * (out[:, R_ANKLE, 1] + out[:, L_ANKLE, 1]))
    assert abs((ankle - neck) - fit["character_neck_ankle"]) < 1e-6
    assert abs(neck - fit["character_neck_v"]) < 1e-6


def test_a_raised_wrist_does_not_change_the_scale():
    """The bounding-box fit's failure, locked out: putting a hand higher must
    not shrink the body, because the span being matched does not include it."""
    plain = _clip()
    raised = plain.copy()
    raised[:, 4, 1] -= 0.10
    reference = _reference()
    _, a = fit_to_character(plain, reference, WIDTH, HEIGHT)
    _, b = fit_to_character(raised, reference, WIDTH, HEIGHT)
    assert abs(a["scale"] - b["scale"]) < 1e-9


def test_the_scale_is_the_same_on_both_axes_in_pixels():
    uv = _clip()
    out, fit = fit_to_character(uv, _reference(), WIDTH, HEIGHT)
    du = np.median((out[:, 5, 0] - out[:, 2, 0]) / (uv[:, 5, 0] - uv[:, 2, 0]))
    dv = np.median((out[:, 5, 1] - out[:, 2, 1]) / (uv[:, 5, 1] - uv[:, 2, 1]))
    assert abs(du - dv) < 1e-9, (du, dv)
    assert abs(du - fit["scale"]) < 1e-9


def test_the_dancer_still_rises_and_falls():
    uv = _clip()
    out, fit = fit_to_character(uv, _reference(), WIDTH, HEIGHT)
    # The BODY's extent: the face is fitted separately, so including it would
    # measure the head transform rather than the dancer rising and falling.
    body = [1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13]
    before = uv[:, body, 1].max(axis=1) - uv[:, body, 1].min(axis=1)
    after = out[:, body, 1].max(axis=1) - out[:, body, 1].min(axis=1)
    assert np.std(after) > 0
    assert abs(np.std(after) / np.std(before) - fit["scale"]) < 1e-6


def test_asking_for_the_pose_s_own_landmarks_changes_nothing():
    """Positive control."""
    uv = _clip()
    neck_v = float(np.median(uv[:, NECK, 1]))
    ankle_v = float(np.median(0.5 * (uv[:, R_ANKLE, 1] + uv[:, L_ANKLE, 1])))
    neck_u = float(np.median(uv[:, NECK, 0]))
    own = {slot: tuple(np.median(uv[:, slot], axis=0)) for slot in FACE}
    out, fit = fit_to_character(uv, _reference(neck_v, ankle_v, neck_u, own),
                                WIDTH, HEIGHT)
    assert abs(fit["scale"] - 1.0) < 1e-9
    assert abs(fit["head_scale"] - 1.0) < 1e-6
    assert np.max(np.abs(out - uv)) < 1e-6


def test_the_head_lands_on_the_character_s_own_face():
    """A cartoon is not built like a body, so one similarity cannot serve both.

    Measured on townfair: the reference's eye-to-neck is 0.1965 of its
    neck-to-ankle and ours is 0.1569, so a body-sized fit left the face 13 px
    low on an 832-line frame -- and the animator, asked to put the eyes there,
    tilted the head DOWN to reach them.  The face therefore gets its own
    similarity onto the reference's own face points.
    """
    uv = _clip()
    # A face a body-scaled fit would NOT land on: higher and wider than ours.
    reference = _reference(face={0: (0.50, 0.400), 14: (0.48, 0.385),
                                 15: (0.52, 0.385), 16: (0.455, 0.395),
                                 17: (0.545, 0.395)})
    out, fit = fit_to_character(uv, reference, WIDTH, HEIGHT)
    for slot in (0, 14, 15, 16, 17):
        got = np.median(out[:, slot], axis=0)
        want = reference[slot][:2] / np.array([WIDTH, HEIGHT])
        assert np.max(np.abs(got - want)) < 0.01, (slot, got, want)
    assert "head_scale" in fit


def test_the_head_still_moves_after_it_is_fitted():
    """Landing the MEDIAN on the character must not freeze the head: a fit that
    pinned every frame would delete every nod and turn the dance contains."""
    uv = _clip()
    moving = uv.copy()
    # A nod moves the WHOLE head.  Perturbing the nose alone also moves the
    # centroid the head transform is built on, which damps it -- so the test
    # would be measuring the fixture, not the fit.
    nod = 0.03 * np.sin(np.linspace(0, 6 * np.pi, len(uv)))
    moving[:, FACE, 1] += nod[:, None]
    out, fit = fit_to_character(moving, _reference(), WIDTH, HEIGHT)
    # The precise property: the nod is carried through at the head's own scale.
    # Comparing against a still clip instead was measuring the fixture's own
    # vertical wobble, which is why an arbitrary "3x" threshold failed on a fit
    # that was in fact passing the motion through exactly.
    # The head passes through BOTH transforms: the body similarity first, then
    # its own.  Checking against head_scale alone reads as a failure on a fixture
    # where the two happen to be reciprocal.
    total = fit["scale"] * fit["head_scale"]
    before = float(np.std(moving[:, 0, 1]))
    after = float(np.std(out[:, 0, 1]))
    assert after == pytest.approx(before * total, rel=0.02), (
        "the head fit changed how much the head moves: {:.5f} -> {:.5f} at "
        "total scale {:.3f}".format(before, after, total))
    assert after > 0.5 * before, "the nod was flattened"
