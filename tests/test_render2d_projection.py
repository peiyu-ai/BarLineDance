"""The 3D->2D projection, whose constants were derived after two wrong guesses.

Both errors were ARITHMETIC and both were caught by measuring the rendered
figure, not by looking at it: multiplying v by width/height put the dancer at
43% of frame height, and dividing the span the other way put it at 240% with
29% of its joints outside the frame.  The requirement is stated once here, as
a test, so a future edit that flips a ratio fails immediately.
"""
import numpy as np
import pytest

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "render2d"))

from project_pose_2d import COCO_NAMES, DIRECT, SMPL, project, to_coco18


def standing(frames=30, height=1.7):
    """A figure standing upright, arms out, facing +y."""
    joints = np.zeros((frames, 24, 3))
    joints[:, SMPL["head"]] = [0, 0, height]
    joints[:, SMPL["neck"]] = [0, 0, height * 0.85]
    joints[:, SMPL["l_shoulder"]] = [0.18, 0, height * 0.82]
    joints[:, SMPL["r_shoulder"]] = [-0.18, 0, height * 0.82]
    joints[:, SMPL["l_elbow"]] = [0.40, 0, height * 0.82]
    joints[:, SMPL["r_elbow"]] = [-0.40, 0, height * 0.82]
    joints[:, SMPL["l_wrist"]] = [0.60, 0, height * 0.82]
    joints[:, SMPL["r_wrist"]] = [-0.60, 0, height * 0.82]
    joints[:, SMPL["l_hip"]] = [0.10, 0, height * 0.52]
    joints[:, SMPL["r_hip"]] = [-0.10, 0, height * 0.52]
    joints[:, SMPL["l_knee"]] = [0.10, 0, height * 0.28]
    joints[:, SMPL["r_knee"]] = [-0.10, 0, height * 0.28]
    joints[:, SMPL["l_ankle"]] = [0.10, 0, 0.02]
    joints[:, SMPL["r_ankle"]] = [-0.10, 0, 0.02]
    return joints


def test_the_figure_fills_the_frame_to_the_margin():
    """THE ONE THAT CAUGHT BOTH BUGS.  margin 0.12 means the body spans 76% of
    the frame height -- not 43% (v multiplied by width/height) and not 240%
    (the span scaled the wrong way)."""
    uv = project(to_coco18(standing()), 720, 1280, margin=0.12)
    span = float(np.median(uv[..., 1].max(axis=1) - uv[..., 1].min(axis=1)))
    assert span == pytest.approx(0.76, abs=0.06), span


def test_it_holds_for_a_landscape_frame_too():
    """A portrait-only fix would pass the test above and still be wrong."""
    uv = project(to_coco18(standing()), 1280, 720, margin=0.12)
    span = float(np.median(uv[..., 1].max(axis=1) - uv[..., 1].min(axis=1)))
    assert span == pytest.approx(0.76, abs=0.08), span


def test_one_metre_is_the_same_pixel_count_on_both_axes():
    """The invariant the derivation is built on; if it breaks the character is
    sheared."""
    joints = standing()
    uv = project(to_coco18(joints), 720, 1280)
    coco = to_coco18(joints)[0]
    wrist_gap_m = abs(coco[4, 0] - coco[7, 0])           # r_wrist to l_wrist, in x
    head_ankle_m = abs(coco[0, 2] - coco[10, 2])         # nose to r_ankle, in z
    px_x = abs(uv[0, 4, 0] - uv[0, 7, 0]) * 720
    px_y = abs(uv[0, 0, 1] - uv[0, 10, 1]) * 1280
    assert (px_x / wrist_gap_m) == pytest.approx(px_y / head_ankle_m, rel=0.02)


def test_the_frame_holds_the_body_vertically():
    """What the height-based fit actually guarantees, and only that.

    The figure is fitted to 76% of the frame HEIGHT, so nothing may leave the
    frame vertically -- that is the property the two arithmetic bugs broke.
    Horizontally there is no such guarantee on a portrait frame: a body with its
    arms straight out is as wide as it is tall, and the repository has already
    measured that this crops (docs: "cropping is framing, not hovering" --
    ground truth's own DWPose crops the same way).  Asserting a single
    everything-inside number conflated the two, and a 5% change in the face's
    height was enough to tip it.
    """
    uv = project(to_coco18(standing()), 720, 1280)
    assert float(np.min(uv[..., 1])) >= 0.0
    assert float(np.max(uv[..., 1])) <= 1.0
    # Arms down: then the horizontal fits too, and a regression that widens the
    # figure shows up here rather than being written off as "arms out".
    narrow = standing()
    narrow[:, SMPL["l_wrist"], 0] = 0.20
    narrow[:, SMPL["r_wrist"], 0] = -0.20
    narrow[:, SMPL["l_elbow"], 0] = 0.20
    narrow[:, SMPL["r_elbow"], 0] = -0.20
    tight = project(to_coco18(narrow), 720, 1280)
    assert float(np.mean((tight >= 0) & (tight <= 1))) == 1.0


def test_the_face_matches_the_real_detector_s_proportions():
    """The face is SYNTHESISED, so its proportions have to be borrowed.

    The first version put five points on a guessed 0.16 m sphere and this test
    only asked that they not collapse into a line -- a threshold calibrated on
    the wrong face, which is why it passed while the nose sat four times too
    low and the ears sat ABOVE the eyes.  The operator saw the consequence:
    "头一直低着,脸不对着相机".  The numbers here are the medians of the ingest's
    own DWPose over the fixed ten clips, and they are checked as RATIOS, never
    as fractions of the frame -- the real videos are framed differently from our
    camera, so an absolute comparison measures the framing, not the face.

    Vertical against vertical and horizontal against horizontal, too: the
    projection is isotropic in PIXELS, so normalised u and normalised v are not
    the same unit and a width divided by a height is wrong by height/width --
    which is exactly how the first correction came out 1.62x too wide.
    """
    uv = project(to_coco18(standing()), 720, 1280)
    nose, neck = uv[:, 0], uv[:, 1]
    eye = 0.5 * (uv[:, 14] + uv[:, 15])
    ear = 0.5 * (uv[:, 16] + uv[:, 17])
    down = np.maximum(neck[:, 1] - eye[:, 1], 1e-6)
    shoulders = np.maximum(np.abs(uv[:, 5, 0] - uv[:, 2, 0]), 1e-6)

    assert float(np.median((nose[:, 1] - eye[:, 1]) / down)) == pytest.approx(
        0.153, abs=0.02), "the nose is not where a detector puts it"
    assert float(np.median((ear[:, 1] - eye[:, 1]) / down)) == pytest.approx(
        0.119, abs=0.02), "the ears must sit BELOW the eyes, not above"
    assert float(np.median(np.abs(uv[:, 15, 0] - uv[:, 14, 0]) / shoulders)) \
        == pytest.approx(0.223, abs=0.03)
    assert float(np.median(np.abs(uv[:, 17, 0] - uv[:, 16, 0]) / shoulders)) \
        == pytest.approx(0.566, abs=0.06)


def test_the_coco_layout_is_the_ingest_s_own():
    """The ingest stores DWPose as COCO-18 in this order; emitting a different
    order would silently drive the animator with swapped limbs."""
    assert COCO_NAMES[:8] == ["nose", "neck", "r_shoulder", "r_elbow", "r_wrist",
                              "l_shoulder", "l_elbow", "l_wrist"]
    assert DIRECT[2] == "r_shoulder" and DIRECT[5] == "l_shoulder"
    assert DIRECT[10] == "r_ankle" and DIRECT[13] == "l_ankle"


def test_left_and_right_are_not_mirrored():
    """Validated once against the clip's own DWPose: the naive mapping gave
    -0.82..-0.93 correlation in x on every joint, a consistent sign flip."""
    uv = project(to_coco18(standing()), 720, 1280)
    # The figure's LEFT shoulder is at +x in the world, which is the image's LEFT
    # side (smaller u) once the camera looks down +y.
    assert uv[0, 5, 0] < uv[0, 2, 0]
