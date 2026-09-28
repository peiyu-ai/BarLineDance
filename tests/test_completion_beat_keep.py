"""--completion-beat-keep / --completion-beat-stride / --completion-beat-keep-holds (DEFECTS §97): which frames the
completion may re-draw.  Pinned on a synthetic beat channel (music channel 34), no model."""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import infer_atomic as ia  # noqa: E402


def _music(beats, frames=100):
    m = np.zeros((frames, 35))
    m[beats, 34] = 1
    return m


def test_beats_are_kept_and_the_middle_is_freed():
    free = ia._beat_free_profile(_music([10, 26, 42]), 100, 0.2)
    assert free[10] == 0 and free[26] == 0 and free[42] == 0          # every beat kept
    assert free[18] == 1.0 and free[34] == 1.0                        # interval middles freed
    assert np.all(free[10:13] == 0)                                   # 20% of a 16-frame interval kept after the beat
    assert np.all(free[:10] == 0) and np.all(free[42:] == 0)          # outside the grid untouched


def test_stride_anchors_every_second_beat_from_the_first_bar_line():
    free = ia._beat_free_profile(_music([10, 26, 42, 58, 74, 90]), 100, 0.2, stride=2, bar_start=26)
    assert free[26] == 0 and free[58] == 0 and free[90] == 0
    assert free[42] == 1.0                                            # beat 2 of the bar is re-drawn with its span


def test_held_intervals_are_left_to_the_draft():
    held = np.zeros(100, dtype=bool)
    held[26:42] = True
    free = ia._beat_free_profile(_music([10, 26, 42]), 100, 0.2, held=held)
    assert free[18] == 1.0 and np.all(free[26:42] == 0)
