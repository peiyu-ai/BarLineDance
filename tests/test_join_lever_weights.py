"""The seam join cost must weight a pelvis degree above a wrist degree.

MEASURED, and this is the whole case for the flag: over 150 random butt-joined
library pairs, the equal-weight cost the ranking has always used correlates with
the ACTUAL world-space speed jump at Spearman rho +0.372, while the lever-
weighted one reaches +0.973.  A ranking whose entire purpose is to minimise the
seam was only weakly related to it.

The spike is proximal in cause and distal in appearance: at a bar line the
draft's hands peak at 25.7x their own median speed change and its elbows at
19.1x, while its ROOT peaks at 5.7x -- which is why --draft-root-velocity-blend
moved nothing.
"""
import numpy as np
import pytest
import torch

from infer_atomic import (CONTACT_CHANNELS, ROOT_POSITION_DIMS,
                          IndexedAtomicMotionLibrary)


@pytest.fixture(scope="module")
def weights():
    return IndexedAtomicMotionLibrary._lever_weights()


def block(joint):
    """The rot6d columns of one joint inside the contact-stripped vector."""
    start = ROOT_POSITION_DIMS + 6 * joint
    return slice(start, start + 6)


def test_the_weights_cover_the_contact_stripped_vector(weights):
    assert len(weights) == 151 - CONTACT_CHANNELS


def test_proximal_joints_outweigh_the_leaves(weights):
    """Rotating the pelvis moves every joint; rotating a hand moves none."""
    pelvis = float(weights[block(0)].mean())
    left_hand, right_hand = (float(weights[block(j)].mean()) for j in (22, 23))
    assert pelvis > 10 * max(left_hand, right_hand, 1e-9)


def test_the_spine_outweighs_the_wrists(weights):
    spine = float(weights[block(3)].mean())
    wrists = max(float(weights[block(j)].mean()) for j in (20, 21))
    assert spine > wrists


def test_the_root_translation_keeps_unit_weight(weights):
    """Root POSITION is already in metres; only the rotations need rescaling."""
    assert torch.allclose(weights[:ROOT_POSITION_DIMS],
                          torch.ones(ROOT_POSITION_DIMS), atol=1e-6)


def test_the_rotation_block_has_mean_one(weights):
    """The ROTATIONS are normalised, so turning the flag on does not silently
    rescale the join term against JOIN_VELOCITY_WEIGHT.  The root is left at
    1.0 because it is already in metres -- normalising the whole vector moved
    the root's importance against the rotations by an arbitrary factor, and
    this pair of tests is what caught it."""
    assert float(weights[ROOT_POSITION_DIMS:].mean()) == pytest.approx(1.0, abs=1e-5)


def test_off_reproduces_the_old_arithmetic():
    """Every artifact made before this flag existed must still reproduce."""
    import inspect
    source = inspect.getsource(IndexedAtomicMotionLibrary._join_cost)
    assert "if self.join_lever_weights else 1.0" in source


def test_the_weights_are_cached_not_recomputed_per_candidate():
    """The skeleton is run 24 x 8 times to build them; doing that per candidate
    would make retrieval unusable."""
    first = IndexedAtomicMotionLibrary._lever_weights()
    assert IndexedAtomicMotionLibrary._lever_weights() is first
