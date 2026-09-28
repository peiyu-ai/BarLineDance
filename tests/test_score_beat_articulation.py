"""Controls for the beat-articulation instrument, as executable assertions.

Every one of these fails on a plausible wrong implementation, which is the
point: ``docs/DANCE_QUALITY_DEFECTS.md`` §6.2 records a test suite that passed
on a broken version because it asserted a downstream effect instead of the
invariant.  These assert the invariants.

The fixtures are built in JOINT POSITION space by integrating a designed speed
profile, so the instrument's own differencing, root-relative subtraction and
24-joint averaging are all exercised rather than bypassed.
"""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.score_beat_articulation import (BEAT_CHANNEL, FPS, beat_depths, body_speed,
                                           clip_reading, rotated_beats)

MOVING_JOINTS = 2.0
JOINTS = 24.0


def _from_profile(profile):
    """Two wrists travelling with the given per-frame speed, in metres."""
    track = np.concatenate([[0.0], np.cumsum(profile[:-1])]) / FPS
    joints = np.zeros((len(profile), 24, 3))
    joints[:, 20, 0] = track
    joints[:, 21, 0] = -track
    return joints                       # mean joint speed = profile * 2/24


def _stops_at(frames, period, offset, amplitude, width=1.5):
    """Speed that dips to zero within ``width`` frames of every ``period``-th frame."""
    t = np.arange(frames)
    distance = (t - offset) % period
    distance = np.minimum(distance, period - distance)
    return amplitude * (1.0 - np.exp(-(distance ** 2) / (2 * width ** 2)))


def _music(frames, period, offset=0):
    music = np.zeros((frames, 35), np.float32)
    music[np.arange(offset, frames, period), BEAT_CHANNEL] = 1.0
    return music


def test_constant_speed_reads_zero_depth_and_a_stop_reads_the_full_swing():
    frames, period, amplitude = 300, 15, 12.0
    beats = np.arange(period, frames - period, period)
    steady = _from_profile(np.full(frames, amplitude))
    assert beat_depths(body_speed(steady), beats).mean() == pytest.approx(0.0, abs=1e-6)
    stopping = _from_profile(_stops_at(frames, period, 0, amplitude))
    expected = amplitude * MOVING_JOINTS / JOINTS          # 1.0 m/s of mean joint speed
    assert beat_depths(body_speed(stopping), beats).mean() == pytest.approx(expected, rel=0.15)


def test_depth_is_absolute_metres_per_second(scale=0.5):
    """No within-clip normalization: halving the motion halves the reading (F4)."""
    frames, period = 300, 15
    beats = np.arange(period, frames - period, period)
    big = beat_depths(body_speed(_from_profile(_stops_at(frames, period, 0, 12.0))), beats).mean()
    small = beat_depths(body_speed(_from_profile(_stops_at(frames, period, 0, 12.0 * scale))),
                        beats).mean()
    assert small == pytest.approx(big * scale, rel=0.02)


def test_a_stop_off_the_beat_reads_lower_than_a_stop_on_it():
    frames, period = 600, 16
    beats = np.arange(period, frames - period, period)
    on_beat = beat_depths(body_speed(_from_profile(_stops_at(frames, period, 0, 12.0))), beats)
    off_beat = beat_depths(body_speed(_from_profile(_stops_at(frames, period, period // 2, 12.0))),
                           beats)
    assert off_beat.mean() < 0.35 * on_beat.mean()


def test_peak_must_precede_trough():
    """Accelerate-then-stop is a hit; stop-then-accelerate is not."""
    frames, period, amplitude = 600, 20, 12.0
    beats = np.arange(period, frames - period, period)
    ramp = np.tile(np.linspace(0.0, amplitude, period), frames // period)[:frames]
    hit = beat_depths(body_speed(_from_profile(ramp)), beats).mean()            # peak at b-1
    reverse = beat_depths(body_speed(_from_profile(ramp[::-1].copy())), beats).mean()
    assert hit > 0.5
    assert reverse < 0.5 * hit


def test_rotated_null_keeps_count_and_gaps_and_the_gain_survives_only_on_the_grid():
    frames, period = 896, 16          # a whole number of beats, so the wrap gap is a gap too
    joints = _from_profile(_stops_at(frames, period, 0, 12.0))
    music = _music(frames, period)
    beats = np.flatnonzero(music[:, BEAT_CHANNEL] > 0.5)
    shifted = rotated_beats(beats, frames, np.random.default_rng(3))
    assert len(shifted) == len(beats)
    assert sorted(np.diff(shifted).tolist()) == sorted(np.diff(beats).tolist())
    got = clip_reading(joints, music, np.random.default_rng(4), draws=40)
    # The CEILING, and it is not 100%: the +-4 frame tolerance that makes the
    # instrument phase-forgiving also lets a rotated beat land inside a dip
    # roughly 60% of the time at a 16-frame period.  A synthetic body that
    # stops on EVERY beat gives gain/depth = 0.28/0.90 = 31%; the ground truth
    # reads 0.027/0.455 = 6%, i.e. about a fifth of what is reachable.  Any
    # future reading above ~0.3 of its own depth should be disbelieved first.
    assert 0.25 < got["depth"] - got["depth_null"] < 0.45


def test_a_body_incommensurate_with_the_beat_grid_reads_no_gain():
    """The instrument must not manufacture alignment out of any periodic body."""
    frames = 1200
    music = _music(frames, 15)
    joints = _from_profile(_stops_at(frames, 23, 0, 12.0))   # 23 vs 15: incommensurate
    got = clip_reading(joints, music, np.random.default_rng(5), draws=40)
    assert abs(got["depth"] - got["depth_null"]) < 0.2 * got["depth"]


def test_hit_rate_threshold_is_absolute_not_relative():
    """A quiet clip cannot earn hits by being internally spiky -- the §12.1 defect."""
    frames, period = 600, 15
    music = _music(frames, period)
    loud = clip_reading(_from_profile(_stops_at(frames, period, 0, 12.0)), music,
                        np.random.default_rng(6), threshold=0.5, draws=5)
    quiet = clip_reading(_from_profile(_stops_at(frames, period, 0, 12.0 * 0.05)), music,
                         np.random.default_rng(6), threshold=0.5, draws=5)
    assert loud["hits_per_s"] > 1.0
    assert quiet["hits_per_s"] == 0.0
