"""Controls for the plan-shape columns.

The load-bearing test is ``test_a_collapsed_clip_is_invisible_to_segment_count``:
it builds the exact situation the pooled table missed -- a clip that holds one
movement for most of its length while still carrying the same NUMBER of
segments -- and demands that the new column separates them while the old one
does not.  Without it this file would only prove the arithmetic runs.
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.score_plan_shape import segments, shape_of, summarise


def plan(*runs):
    """``plan((3, 60), (0, 30))`` -> 60 frames of class 3 then 30 of filler."""
    return np.concatenate([np.full(n, label, dtype=np.int64) for label, n in runs])


def test_segments_splits_on_every_label_change_including_filler():
    assert segments(plan((3, 5), (0, 2), (7, 4))) == [(3, 0, 5), (0, 5, 7), (7, 7, 11)]


def test_filler_is_excluded_from_segment_count_and_lengths():
    shape = shape_of(plan((3, 30), (0, 60), (7, 30)))
    assert shape["segments"] == 2
    assert shape["filler_share"] == pytest.approx(0.5)
    assert shape["longest_hold_s"] == pytest.approx(1.0)


def test_a_collapsed_clip_is_invisible_to_segment_count():
    # THE POINT OF THIS FILE.  Both clips have four named segments and nearly
    # the same duration; one is danced, the other holds a single pose for ten
    # seconds.  Segment count cannot tell them apart -- the longest hold can.
    danced = plan((1, 90), (0, 30), (2, 90), (0, 30), (3, 90), (0, 30), (4, 90))
    collapsed = plan((1, 300), (2, 20), (3, 20), (4, 20))
    assert shape_of(danced)["segments"] == shape_of(collapsed)["segments"] == 4
    assert shape_of(danced)["longest_hold_s"] == pytest.approx(3.0)
    assert shape_of(collapsed)["longest_hold_s"] == pytest.approx(10.0)


def test_filler_separates_wall_to_wall_movement_from_a_resting_dancer():
    # The shipped arm read 0.4% against ground truth's 30.4%; this is the
    # column that says so.
    resting = plan((1, 70), (0, 30), (2, 70), (0, 30))
    relentless = plan((1, 100), (2, 100))
    assert shape_of(resting)["filler_share"] == pytest.approx(0.3)
    assert shape_of(relentless)["filler_share"] == 0.0


def test_an_all_filler_plan_reports_no_segments_rather_than_crashing():
    shape = shape_of(plan((0, 100)))
    assert shape["segments"] == 0
    assert shape["longest_hold_s"] == 0.0
    assert shape["filler_share"] == 1.0


def test_an_empty_plan_is_unmeasured():
    assert shape_of(np.array([], dtype=np.int64)) is None


def test_summary_counts_clips_over_the_threshold_not_frames():
    rows = [{"filler_share": 0.3, "segments": 4, "median_segment_s": 2.0,
             "longest_hold_s": s} for s in (1.0, 3.0, 5.0, 9.0)]
    summary = summarise(rows, hold_threshold=4.0)
    assert summary["clips_over_threshold"] == "2/4"
    assert summary["longest_hold_max_s"] == pytest.approx(9.0)
    assert summary["longest_hold_median_s"] == pytest.approx(4.0)


def test_summary_of_nothing_is_none_not_a_row_of_zeros():
    # A row of zeros would read as "a perfect plan" in the printed table.
    assert summarise([], hold_threshold=4.0) is None


def test_the_reading_is_in_seconds_at_thirty_fps():
    assert shape_of(plan((1, 30)))["longest_hold_s"] == pytest.approx(1.0)
    assert shape_of(plan((1, 15)))["longest_hold_s"] == pytest.approx(0.5)
