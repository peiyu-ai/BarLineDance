"""The time-base audit, and the two controls that decide what it may claim.

The defect it exists for is the 2026-08-25 one: a clip whose reconstruction was
the first half of the dance stretched over the whole song, with matching frame
counts so every length check passed.  The instrument had to be rebuilt twice
before it could say anything -- once because whole-frame pixel motion on real
footage tracks the background rather than the dancer, and once because dance is
periodic and a periodic trace correlates with a compressed copy of itself.  Both
failures are pinned here.
"""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.audit_footage_motion_sync import (SCALES, best_scale, joint_motion,
                                             keypoint_motion)


def _dance(frames=480, period=45, seed=0):
    """A periodic trace with noise -- periodic on purpose: that is the confound."""
    rng = np.random.default_rng(seed)
    t = np.arange(frames)
    base = (1.0 + np.sin(2 * np.pi * t / period)
            + 0.4 * np.sin(2 * np.pi * t / (period / 2) + 0.7))
    return base + rng.normal(scale=0.15, size=frames)


def _at(readings, scale):
    return float(readings[list(SCALES).index(scale)])


def test_a_matching_pair_reads_scale_one():
    trace = _dance()
    readings = best_scale(trace, trace)
    assert SCALES[int(np.nanargmax(readings))] == 1.0
    assert _at(readings, 1.0) == pytest.approx(1.0, abs=1e-6)


def test_a_reconstruction_covering_half_the_clip_reads_one_half():
    """The defect, synthesised: the 3D trace spans only the first half."""
    trace = _dance()
    readings = best_scale(trace, trace[:len(trace) // 2])
    assert SCALES[int(np.nanargmax(readings))] == 0.5
    assert _at(readings, 0.5) > 3 * abs(_at(readings, 1.0))


def test_periodicity_alone_can_produce_an_off_scale_peak():
    """Why every reading is scored against the clip's own baseline.

    Not any periodic signal: a PURE sinusoid compressed 2x is orthogonal to
    itself and reads 0.003, which is what the first version of this test
    asserted the opposite of.  The mechanism is the SECOND HARMONIC -- compress
    a trace carrying both f and 2f and its 2f component lands on the original's
    f -- and a dancer's speed trace carries exactly that (a step and its
    half-step).  Measured on a real clip: the 2D trace's correlation with its
    own compressed copy is 0.402, higher than any of the suspicious 3D readings,
    which is why an off-scale peak means nothing until it beats this baseline.
    """
    t = np.arange(480)
    harmonic = np.sin(2 * np.pi * t / 60) + 0.8 * np.sin(4 * np.pi * t / 60)
    baseline = best_scale(harmonic, harmonic)
    assert _at(baseline, 0.5) > 0.4      # large, and entirely an artefact
    assert _at(baseline, 1.0) == pytest.approx(1.0, abs=1e-6)

    pure = np.sin(2 * np.pi * t / 60)
    assert abs(_at(best_scale(pure, pure), 0.5)) < 0.05


def test_the_null_is_near_zero():
    rng = np.random.default_rng(3)
    trace = _dance()
    nulls = [np.nanmax(best_scale(trace, rng.permutation(trace))) for _ in range(20)]
    assert np.median(nulls) < 0.25


def test_keypoint_motion_removes_camera_translation():
    """The 2D trace is made relative to keypoint 0, so panning the camera must
    not register as the dancer moving."""
    rng = np.random.default_rng(1)
    keypoints = rng.normal(size=(200, 18, 2))
    still = keypoint_motion(keypoints)
    panned = keypoint_motion(keypoints + np.cumsum(
        rng.normal(scale=5.0, size=(200, 1, 2)), axis=0))
    assert np.allclose(still, panned, atol=1e-9)


def test_joint_motion_is_root_relative():
    rng = np.random.default_rng(2)
    joints = rng.normal(size=(120, 24, 3))
    walked = joints + np.cumsum(rng.normal(scale=0.3, size=(120, 1, 3)), axis=0)
    assert np.allclose(joint_motion(joints), joint_motion(walked), atol=1e-9)
