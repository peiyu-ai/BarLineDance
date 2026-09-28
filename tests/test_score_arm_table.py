"""The arm-comparison table, and the controls that make its columns admissible.

The table exists because the one it replaces had three columns and chose
``--completion-sample-steps 5``, an arm that buys arm span by cutting motion
energy to 0.536 of the ground truth's.  So the first thing tested is that the
energy column moves when the energy moves, and the last is that the gate the
column feeds actually fails.

Measured controls on real artifacts, run 2026-08-30 (not re-run here because
they need the corpora, but they are what licenses the gate):

    clean5b5 vel4, the generation the operator judged good   energy 1.070  PASS
    clean5b5 "before", the one the worklog records as worse  energy 0.678  FAIL
    shipped wild baseline                                    energy 0.773  FAIL
    the 2026-08-30 sample-steps pick                         energy 0.536  FAIL

A gate that a known-good artifact passes and known-bad ones fail is the point;
a gate calibrated only against the arms it was written for is not a gate.
"""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.score_arm_table import (arm_span, energy, lag_zero_share, root_range,
                                   segments_per_second, summarise)


def _walker(frames=300, amplitude=1.0, travel=0.0, phase=0.0):
    """A synthetic body whose limbs swing at a known amplitude and phase.

    Built relative to the root and then translated once -- the first version
    added the root twice and put the two wrists at different heights, so two of
    these tests failed against correct code.  A control that is wrong in a way
    that looks like a failure is worse than no control, which is the whole
    lesson of section 2.2.

    SMPL indices: 0 root, 1/2 hips, 16-23 the four limb chains, 20/21 wrists.
    """
    joints = np.zeros((frames, 24, 3))
    t = np.arange(frames) / 30.0
    left = amplitude * np.sin(2 * np.pi * 1.0 * t)
    right = amplitude * np.sin(2 * np.pi * 1.0 * t + phase)
    joints[:, 1, 0], joints[:, 2, 0] = -0.1, 0.1
    for index in (16, 18, 20, 22):          # left arm chain
        joints[:, index, 0] = -0.3 + 0.2 * left
        joints[:, index, 2] = 1.4
    for index in (17, 19, 21, 23):          # right arm chain
        joints[:, index, 0] = 0.3 + 0.2 * right
        joints[:, index, 2] = 1.4
    for index in (1, 4, 7, 10):             # left leg chain
        joints[:, index, 1] = 0.2 * left
    for index in (2, 5, 8, 11):             # right leg chain
        joints[:, index, 1] = 0.2 * right
    joints[:, 0, :] = 0.0
    ground = travel * np.sin(2 * np.pi * 0.2 * t)
    joints[:, :, 0] += ground[:, None]      # the whole body travels together
    return joints


def test_energy_tracks_amplitude():
    """Halving the motion must halve the reading.  FID cannot do this -- it
    standardises per dimension, so halving every clip leaves fid_k at 0.0."""
    full = energy(_walker(amplitude=1.0))
    half = energy(_walker(amplitude=0.5))
    assert half == pytest.approx(full / 2, rel=0.02)


def test_arm_span_reads_the_wrist_separation():
    joints = _walker(amplitude=0.0)
    assert arm_span(joints) == pytest.approx(0.6, abs=1e-6)
    wide = _walker(amplitude=0.0)
    wide[:, (17, 19, 21, 23), 0] += 0.4
    assert arm_span(wide) == pytest.approx(1.0, abs=1e-6)


def test_root_range_reads_travel_not_limb_motion():
    assert root_range(_walker(amplitude=2.0, travel=0.0)) == pytest.approx(0.0, abs=1e-9)
    assert root_range(_walker(amplitude=0.0, travel=1.0)) == pytest.approx(1.0, abs=0.02)


def test_lag0_separates_synchronous_from_offset_limbs():
    """The whole point of the column: in phase reads 1, in antiphase does not."""
    assert lag_zero_share(_walker(phase=0.0)) == 1.0
    assert lag_zero_share(_walker(phase=np.pi / 2)) < 1.0


def test_lag0_is_length_dependent_which_is_why_it_is_whole_clip_only():
    """Guards the caveat rather than the value.  A conclusion was published from
    ignoring this: 40-frame library prototypes read 0.000 against whole ground
    truth clips at 0.628, and the gap was attributed to splicing."""
    rng = np.random.default_rng(0)
    joints = _walker(frames=600, phase=0.7)
    joints[:, :, :] += rng.normal(scale=0.02, size=joints.shape)
    whole = lag_zero_share(joints)
    chunks = np.mean([lag_zero_share(joints[start:start + 40])
                      for start in range(0, 560, 40)])
    assert chunks < whole


def test_segments_per_second_counts_runs_not_labels():
    labels = np.array([3, 3, 3, 7, 7, 3, 3])
    assert segments_per_second(labels, 7) == pytest.approx(3 / (7 / 30.0))


def test_the_summary_reports_a_missing_beat_column_as_missing():
    """A column with nothing in it must read nan, never 0.0 -- an absent
    measurement and a measurement of zero are different claims."""
    rows = {"energy": [1.0], "span": [0.5], "root": [0.2], "lag0": [1.0], "seg": [0.5],
            "jitter": [0.1], "twist": [30.0], "wristsync": [0.5], "skate": [0.4],
            "energy_ratio": [1.0], "root_ratio": [1.0],
            "R": [], "R_null": []}
    report = summarise("x", rows)
    assert np.isnan(report["beat_R"])
    assert report["beat_wins"] == "-"
