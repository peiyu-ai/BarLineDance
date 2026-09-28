"""Controls for the repetition metric.

The metric must read LOW on a clip we tiled from one block by hand and HIGH on
a clip of independent blocks.  Only the pair makes it a measurement: a metric
that reads low on everything would satisfy the first test alone.
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.measure_motion_repetition import novelty, windows


def block(frames, seed, joints=24):
    rng = np.random.default_rng(seed)
    time = np.linspace(0.0, 2.0 * np.pi, frames)[:, None, None]
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(1, joints, 3))
    amplitude = rng.uniform(0.15, 0.45, size=(1, joints, 3))
    return amplitude * np.sin(time + phase)


def tiled(block_frames, repeats, seed=0):
    one = block(block_frames, seed)
    return np.concatenate([one] * repeats, axis=0)


def varied(block_frames, count, seed=0):
    return np.concatenate([block(block_frames, seed + i) for i in range(count)], axis=0)


def test_a_clip_tiled_from_one_block_reads_near_zero():
    # POSITIVE CONTROL: the answer is known -- every second of this clip has an
    # exact twin one block away.
    assert novelty(tiled(60, 8)) < 0.05


def test_a_clip_of_independent_blocks_reads_high():
    # NEGATIVE CONTROL: without this, a metric that reads low on everything
    # would pass the test above.
    assert novelty(varied(60, 8)) > 0.4


def test_the_two_controls_are_separated_by_a_wide_margin():
    assert novelty(varied(60, 8)) > 8 * novelty(tiled(60, 8))


def test_the_reading_is_scale_free():
    # A dancer who simply moves further must not read as more varied.
    small = varied(60, 8)
    assert novelty(small) == pytest.approx(novelty(small * 5.0), rel=1e-6)


def test_the_reading_is_root_relative():
    # Translating the whole body must not change the reading.
    motion = varied(60, 8)
    moved = motion + np.linspace(0.0, 3.0, len(motion))[:, None, None]
    assert novelty(motion) == pytest.approx(novelty(moved), rel=1e-6)


def test_neighbours_close_in_time_cannot_answer_for_a_repeat():
    # One continuous slow movement is not a repeat.  With the time gap enforced
    # it must not read as one; the gap is the whole reason the metric is not
    # just "how slowly does this clip move".
    slow = block(240, seed=7)
    assert novelty(slow, min_gap_seconds=1.5) > 0.4


def test_a_clip_too_short_to_hold_two_windows_is_unmeasured():
    assert novelty(block(10, seed=1)) is None
    assert novelty(block(120, seed=1), min_gap_seconds=100.0) is None


def test_windows_are_flattened_in_frame_order():
    motion = np.arange(24 * 3 * 10, dtype=float).reshape(10, 24, 3)
    block_, starts = windows(motion, length=4, stride=2)
    assert block_.shape == (4, 4 * 24 * 3)
    assert list(starts) == [0, 2, 4, 6]
