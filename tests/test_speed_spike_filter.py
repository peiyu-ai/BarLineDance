"""The library's jolts, and the line that is ground truth's rather than mine.

The operator, 2026-09-16, correcting an earlier diagnosis of mine: the sudden
acceleration is "偶尔的出现", not spread evenly -- which ruled out the frame-rate
conversion (that one steps four times a SECOND) and pointed upstream.  Measured
in the 3D motion, before any projection or render:

    frames above 2.5x the local median speed   ground truth 0.07%   ours 0.22%
    worst single frame                         ground truth 2.8x    ours 9.8x

and in the library, sliced the way the release is:

    150-frame windows reaching 4x   ground truth 0 of 62   library 5.8%
    worst window                    ground truth 3.12x     library 13.43x
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.census_release_yaw_steps import speed_spikes, SPEED_JOINTS  # noqa: E402


def _dance(frames=150, seed=0):
    """Joints that move smoothly, at a speed that itself varies -- a dance is
    fast in places and that must not read as a spike."""
    rng = np.random.default_rng(seed)
    time = np.arange(frames) / 30.0
    joints = np.zeros((frames, 24, 3))
    tempo = 1.0 + 0.6 * np.sin(2 * np.pi * time / 3.0)
    phase = np.cumsum(tempo) * 0.15
    for index in SPEED_JOINTS:
        joints[:, index, 0] = 0.3 * np.sin(phase + index)
        joints[:, index, 2] = 0.9 + 0.2 * np.cos(phase + index * 0.7)
    joints += 1e-4 * rng.standard_normal(joints.shape)
    return joints


def test_a_smooth_dance_never_reaches_the_line():
    """Positive control: speeding up and slowing down is not a jolt."""
    assert float(speed_spikes(_dance()).max()) < 4.0


def test_one_bad_frame_is_found():
    joints = _dance()
    joints[75:] += np.array([0.25, 0.0, 0.0])      # a teleport, one frame wide
    spikes = speed_spikes(joints)
    assert float(spikes.max()) > 4.0
    assert int(np.argmax(spikes)) in (74, 75), int(np.argmax(spikes))


def test_the_reading_is_relative_not_absolute():
    """A dance twice as fast everywhere must read the same: the defect is a
    frame out of step with its NEIGHBOURS, not a fast passage."""
    slow = _dance()
    fast = _dance()[::2]
    assert abs(float(speed_spikes(slow).max())
               - float(speed_spikes(fast).max())) < 1.5


def test_a_short_window_does_not_crash():
    assert len(speed_spikes(_dance(frames=2))) == 1
    assert len(speed_spikes(_dance(frames=1))) == 0
