"""The facing anchor's correction is spread over the segment, not dropped on the seam.

WHY.  The operator, 2026-09-12, on output/sample_20260912_postfix10: "渲染存在1次
姿态的跳变。未有衔接motion 过度".  Located to 7030793823240424742:clip000 at
t=12.47 s, where the rendered clip turns **110.4 degrees in a single frame**
against that clip's own ground-truth maximum of **17.7**.  On screen the dancer's
back becomes her front with nothing in between.

THE CAUSE IS THE ANCHOR'S SHAPE, NOT ITS EXISTENCE.  ``--draft-facing-anchor``
stops continuity from accumulating the dancer away from the camera by pulling a
fraction of the drift out at every join, and it did that by subtracting from the
ALIGNMENT CONSTANT -- so the whole correction landed in the seam's one frame.
At that seam the accumulated drift was -206.4 degrees and the anchor's share
0.6 x 206.4 = 123.8, which matches the observed step.

Three things were ruled out before the anchor was blamed, and each is worth
keeping because each looked plausible: the prototype itself is clean (sample
115, its own yaw moves a median 0.6 degrees a frame, maximum 3.2, no frame over
40); both sides of the seam are conditioned (safe_draft_condition_fraction 1.0);
and the seam blend does not cause it -- it only HIDES the step for one frame,
which is why the first measurement of the blended draft read 12.5 degrees.

POSITIVE CONTROL is ``test_the_correction_is_actually_applied``: every smoothness
assertion here is satisfied by doing nothing at all.
"""
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import infer_atomic
from infer_atomic import CONTACT_CHANNELS, ROOT_POSITION_DIMS, ROOT_POSITION_START

ROT6D_START = CONTACT_CHANNELS + ROOT_POSITION_DIMS
FEATURE_DIM = ROT6D_START + 24 * 6


class _Library:
    """The smallest object ``_rotate_about_z_varying`` needs."""

    def __init__(self):
        self._affine = (torch.ones(FEATURE_DIM), torch.zeros(FEATURE_DIM))

    GLOBAL_ORIENT_START = infer_atomic.IndexedAtomicMotionLibrary.GLOBAL_ORIENT_START
    _normalizer_affine = infer_atomic.IndexedAtomicMotionLibrary._normalizer_affine
    _facing_yaw = infer_atomic.IndexedAtomicMotionLibrary._facing_yaw
    _rotate_about_z_varying = \
        infer_atomic.IndexedAtomicMotionLibrary._rotate_about_z_varying


def _segment(frames, yaw0=0.0, walk=0.05):
    """A body facing ``yaw0`` throughout, walking ``walk`` per frame along +x."""
    values = torch.zeros(frames, FEATURE_DIM)
    c, s = math.cos(yaw0), math.sin(yaw0)
    values[:, ROT6D_START + 0] = c
    values[:, ROT6D_START + 1] = s
    values[:, ROT6D_START + 3] = -s
    values[:, ROT6D_START + 4] = c
    values[:, ROOT_POSITION_START] = torch.arange(frames, dtype=torch.float32) * walk
    return values


def _yaw(library, values):
    yaw, _ = library._facing_yaw(values)
    return np.degrees(np.unwrap(yaw.numpy()))


def _ramp(frames):
    return 0.5 - 0.5 * torch.cos(math.pi * torch.arange(frames, dtype=torch.float32)
                                 / (frames - 1))


def test_the_correction_is_actually_applied():
    """POSITIVE CONTROL: without it, a no-op passes every other test here."""
    library = _Library()
    values = _segment(40)
    turned = library._rotate_about_z_varying(values, -math.radians(120.0) * _ramp(40))
    yaw = _yaw(library, turned)
    assert abs(yaw[-1] - yaw[0] + 120.0) < 1.0, (yaw[0], yaw[-1])


def test_the_first_frame_does_not_move():
    """The seam is where the jump was: the ramp must start at exactly zero, so
    the alignment that lands this frame on the previous segment's facing is not
    then undone by the correction."""
    library = _Library()
    values = _segment(40)
    turned = library._rotate_about_z_varying(values, -math.radians(120.0) * _ramp(40))
    assert torch.allclose(turned[0], values[0], atol=1e-6)


def test_no_frame_turns_more_than_ground_truth_ever_does():
    """The defect in one number.  Ground truth's largest single-frame yaw change
    on the clip that showed the jump is 17.7 degrees; the rendered arm read
    110.4.  Spread over a bar, 120 degrees is under 5 a frame."""
    library = _Library()
    values = _segment(47)
    turned = library._rotate_about_z_varying(values, -math.radians(123.8) * _ramp(47))
    step = np.abs(np.diff(_yaw(library, turned)))
    assert step.max() < 17.7, step.max()


def test_the_rate_is_flat_at_the_seam_too():
    """A linear ramp would be continuous in facing and STEP in turning rate,
    which is the same defect one derivative up -- the mistake this file has
    made at two other levels (see _blend_draft_seams)."""
    library = _Library()
    values = _segment(47)
    turned = library._rotate_about_z_varying(values, -math.radians(123.8) * _ramp(47))
    step = np.abs(np.diff(_yaw(library, turned)))
    assert step[0] < 0.25 * step.max(), (step[0], step.max())


def test_the_distance_walked_is_unchanged():
    """The root turns WITH the body: each frame's displacement is rotated, so
    the path bends but its length does not change. Rotating the body without
    its travel is what manufactures foot skate."""
    library = _Library()
    values = _segment(40, walk=0.05)
    turned = library._rotate_about_z_varying(values, -math.radians(120.0) * _ramp(40))
    root = slice(ROOT_POSITION_START, ROOT_POSITION_START + ROOT_POSITION_DIMS)
    before = torch.linalg.norm(torch.diff(values[:, root], dim=0), dim=1)
    after = torch.linalg.norm(torch.diff(turned[:, root], dim=0), dim=1)
    assert torch.allclose(before, after, atol=1e-6)


def test_a_zero_correction_changes_nothing():
    library = _Library()
    values = _segment(40)
    turned = library._rotate_about_z_varying(values, torch.zeros(40))
    assert torch.allclose(turned, values, atol=1e-6)


def test_a_one_frame_segment_is_left_alone():
    library = _Library()
    values = _segment(1)
    turned = library._rotate_about_z_varying(values, torch.zeros(1))
    assert torch.allclose(turned, values, atol=1e-6)
