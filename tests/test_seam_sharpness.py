"""The three-sided sharpness judge: peak DOWN, count NOT down, jitter NOT up.

Each side is here because a one-sided version of this column was already bought
once.  On 2026-09-13 ``--completion-inpaint-seam-width 16`` moved the peak to
99% of ground truth and the seam share to 11.2% (below chance) -- a clean win on
both of those -- while the accent count fell to 2.218 against ground truth's
2.939, because a width of 16 regenerates about 64% of a 50-frame bar and smooths
the dance along with the join.  A judge without the count term calls that a win.
"""
import numpy as np
import pytest

from tools.score_seam_sharpness import SEAM_WINDOW, clip_reading, speed_change


def moving(frames=400, joints=24, seed=0, hits=(), size=0.12):
    rng = np.random.default_rng(seed)
    t = np.arange(frames)[:, None, None]
    motion = 0.3 * np.sin(t / 9.0 + rng.normal(size=(1, joints, 3)))
    for hit in hits:
        motion[hit:hit + 2] += size
    return motion


def test_a_bigger_jump_raises_the_peak():
    small = clip_reading(moving(hits=(100, 200, 300), size=0.05), [], 0.001)
    big = clip_reading(moving(hits=(100, 200, 300), size=0.40), [], 0.001)
    assert big["peak"] > small["peak"]


def test_smoothing_lowers_the_peak_and_the_count_together():
    """The trade this judge exists to see: you cannot lower the peak by
    smoothing without also losing counts."""
    from scipy.ndimage import uniform_filter1d
    original = moving(hits=range(40, 360, 30))
    threshold = float(np.percentile(speed_change(original), 90))
    sharp = clip_reading(original, [], threshold)
    dull = clip_reading(uniform_filter1d(original, size=9, axis=0), [], threshold)
    assert dull["peak"] < sharp["peak"]
    assert dull["count"] < sharp["count"], (
        "if smoothing could lower the peak without costing counts, the count "
        "term would not be protecting anything")


def test_jumps_placed_at_the_seams_read_as_at_seam():
    seams = [100, 200, 300]
    at = clip_reading(moving(hits=seams, size=0.5), seams, 0.001)
    away = clip_reading(moving(hits=[s + 20 for s in seams], size=0.5), seams, 0.001)
    assert at["at_seam"] > away["at_seam"]


def test_the_seam_window_is_narrow_enough_to_localise():
    """At +-3 frames a bar of ~50 frames gives a chance share near 12%; a window
    so wide that everything is 'at a seam' would make the column unreadable."""
    assert SEAM_WINDOW <= 4


def test_the_threshold_is_shared_so_seam_spikes_cannot_raise_the_bar():
    """The defect that refuted the previous column: a per-clip threshold lets a
    few huge seam spikes hide the ordinary accents beneath them."""
    import pathlib
    source = pathlib.Path("tools/score_seam_sharpness.py").read_text()
    assert "np.concatenate(pooled)" in source
    assert "fixed for every arm" in source


def test_count_uses_the_passed_threshold_not_a_percentile():
    calm = moving(hits=())
    loose = clip_reading(calm, [], 0.0)["count"]
    strict = clip_reading(calm, [], 1.0)["count"]
    assert loose > strict == 0.0
