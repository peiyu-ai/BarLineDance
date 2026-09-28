"""The time warp that makes a draft's accents land where they claim to.

Exists because of one number: the completion's output speed profile correlates
with its own conditioning draft's at -0.110 (50 clips, 2026-08-31) -- the model
transmits none of the draft's timing, having been trained on drafts whose
timing was uncorrelated with the target by construction.  The warp is the
mechanical half of the repair; these tests hold what it may and may not do.
"""
import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from dataset.atomic import ROT6D_START, motion_accent_frames, warp_to_anchors


def _impulse_motion(frames=60, accents=(15, 40), dim=151):
    """Rotation channels move constantly except at the accent frames, where the
    change rate dips -- a synthetic settle with a known answer."""
    motion = torch.zeros(frames, dim)
    position = torch.cumsum(torch.ones(frames), 0)
    for accent in accents:
        position[accent:] -= 0.9          # near-zero step INTO the accent frame
    motion[:, ROT6D_START:] = position.unsqueeze(-1)
    return motion


def test_accent_frames_find_the_planted_settles():
    found = motion_accent_frames(_impulse_motion(accents=(15, 40)))
    assert 15 in found or 14 in found
    assert 40 in found or 39 in found


def test_warp_moves_a_settle_onto_its_anchor():
    motion = _impulse_motion(accents=(20,))
    warped = warp_to_anchors(motion, np.array([20]), np.array([30]))
    after = motion_accent_frames(warped)
    assert any(abs(int(frame) - 30) <= 1 for frame in after)
    # endpoints stay put -- the warp re-times, it does not crop
    assert torch.allclose(warped[0], motion[0])
    assert torch.allclose(warped[-1], motion[-1])


def test_no_anchors_is_identity():
    motion = _impulse_motion()
    assert torch.equal(warp_to_anchors(motion, np.array([]), np.array([20])), motion)
    assert torch.equal(warp_to_anchors(motion, np.array([15]), np.array([])), motion)


def test_the_stretch_cap_drops_rather_than_clamps():
    """An anchor that would need a 4x local stretch is refused entirely; a
    clamped warp would put the accent NEAR the anchor, and near defeats the
    purpose -- either it lands or the movement is left alone."""
    motion = _impulse_motion(frames=60, accents=(50,))
    warped = warp_to_anchors(motion, np.array([50]), np.array([5]), max_stretch=1.6)
    assert torch.equal(warped, motion)


def test_warp_preserves_value_range():
    torch.manual_seed(0)
    motion = torch.randn(80, 151)
    warped = warp_to_anchors(motion, np.array([20, 55]), np.array([25, 50]))
    assert warped.shape == motion.shape
    assert warped.max() <= motion.max() + 1e-6
    assert warped.min() >= motion.min() - 1e-6


def test_monotonicity_survives_crossing_anchor_requests():
    """Two targets that would need the same source in reverse order: the second
    pairing is dropped, the map stays monotone, and the result is finite."""
    motion = _impulse_motion(frames=80, accents=(30, 32))
    warped = warp_to_anchors(motion, np.array([30, 32]), np.array([50, 20]))
    assert torch.isfinite(warped).all()
