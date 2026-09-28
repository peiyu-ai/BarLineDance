"""The rotation-space exaggeration, and the invariants that make it safe.

It exists because the completion's per-clip energy barely tracks the song
(correlation 0.210 with ground truth over 90 clips) while its own draft's does
(1.031 of ground truth) -- so one scalar per segment is moved from draft to
output.  Scaling joint POSITIONS would change bone lengths; scaling rotation
DEVIATIONS around the segment mean cannot.
"""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.transfer_draft_dynamics import (_axis_angle, _quaternions,
                                           exaggerate_rotations, gain_schedule)


def test_axis_angle_round_trip():
    """Compared AS ROTATIONS.  An axis-angle vector with angle > pi round-trips
    to the equivalent (2*pi - angle, flipped-axis) form -- same rotation,
    different vector -- so a raw vector comparison fails on a rotation that is
    perfectly fine.  SMPL pose channels stay well under pi, but the sampler
    here does not, and the first version of this test compared vectors."""
    rng = np.random.default_rng(0)
    aa = rng.normal(scale=0.8, size=(40, 23, 3))
    q0 = _quaternions(aa)
    q1 = _quaternions(_axis_angle(q0))
    assert np.abs((q0 * q1).sum(-1)).min() > 1 - 1e-9


def test_gain_one_is_identity():
    rng = np.random.default_rng(1)
    aa = rng.normal(scale=0.5, size=(30, 23, 3))
    out = _axis_angle(exaggerate_rotations(aa, np.ones(30)))
    q0, q1 = _quaternions(aa), _quaternions(out)
    # compare as rotations (q and -q are the same rotation)
    dot = np.abs((q0 * q1).sum(-1))
    assert dot.min() > 1 - 1e-6


def test_gain_scales_the_deviation_angle():
    frames = 24
    aa = np.zeros((frames, 1, 3))
    aa[:, 0, 0] = np.linspace(-0.2, 0.2, frames)     # swing about x, mean ~0
    out = _axis_angle(exaggerate_rotations(aa, np.full(frames, 1.5)))
    assert out[-1, 0, 0] == pytest.approx(0.3, abs=0.01)
    assert out[0, 0, 0] == pytest.approx(-0.3, abs=0.01)


def test_gain_schedule_ramps_not_steps():
    gain = gain_schedule(60, [(20, 40, 1.6)], ramp=5)
    assert gain[:15].max() == pytest.approx(1.0)
    assert gain[28] == pytest.approx(1.6)
    assert np.abs(np.diff(gain)).max() < 0.3        # no single-frame jump of 0.6
