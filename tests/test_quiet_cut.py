"""``--draft-quiet-cut``: move the retrieval seam off the beat, into a gap.

WHY, measured on the eval clips' own music and motion (2026-09-13):
* the strongest onset inside a bar sits within 3 frames of the bar line 44.8%
  of the time, and those frames are ~12% of a bar -- 3.7x chance;
* our retrieval seam is on that line 100% of the time;
* the output's top-10% speed changes are 21.4% within 3 frames of a bar line
  (the raw draft 40.0%) against ground truth's 13.7%, which IS chance because
  ground truth has no seams.

So the pipeline puts its largest artifact exactly where the music is loudest.
This slides the CUT; it does not warp the content -- warping onto the beat is
refuted four times over and one monotone time map settles a whole joint chain
at the same instants.
"""
import numpy as np
import pytest
import torch

from infer_atomic import quiet_bar_bounds


def music_with_envelope(values):
    music = torch.zeros(len(values), 35)
    music[:, 0] = torch.as_tensor(values, dtype=torch.float32)
    return music


def test_zero_slide_is_the_published_behaviour():
    music = music_with_envelope(np.random.default_rng(0).random(200))
    bounds = [0, 50, 100, 150, 200]
    assert quiet_bar_bounds(bounds, music, 0) is bounds


def test_the_cut_moves_to_the_quietest_frame_in_reach():
    loud = np.ones(200)
    loud[47] = 0.0                      # the quiet frame, 3 before the cut
    bounds = [0, 50, 100, 150, 200]
    moved = quiet_bar_bounds(bounds, music_with_envelope(loud), 4)
    assert moved[1] == 47


def test_it_cannot_reach_past_the_slide():
    loud = np.ones(200)
    loud[30] = 0.0                      # quiet, but 20 frames away
    bounds = [0, 50, 100, 150, 200]
    moved = quiet_bar_bounds(bounds, music_with_envelope(loud), 4)
    assert abs(moved[1] - 50) <= 4


def test_ties_go_to_the_frame_nearest_the_original_cut():
    """A flat stretch of silence must not drag every bar to one edge of its
    window -- that would turn a slide into a systematic shift."""
    flat = np.ones(200)
    flat[46:55] = 0.0
    bounds = [0, 50, 100, 150, 200]
    moved = quiet_bar_bounds(bounds, music_with_envelope(flat), 4)
    assert moved[1] == 50


def test_the_ends_never_move():
    music = music_with_envelope(np.zeros(200))
    bounds = [0, 50, 100, 150, 200]
    moved = quiet_bar_bounds(bounds, music, 6)
    assert moved[0] == 0 and moved[-1] == 200


def test_the_result_is_strictly_increasing():
    """A non-increasing cut list makes labels_to_segments emit an empty or
    reversed span, in silence."""
    rng = np.random.default_rng(3)
    music = music_with_envelope(rng.random(400))
    bounds = [0] + list(range(20, 400, 20))
    moved = quiet_bar_bounds(bounds, music, 9)
    assert all(b > a for a, b in zip(moved[:-1], moved[1:]))


def test_adjacent_cuts_cannot_cross():
    """With a slide wider than the gap between two cuts, the window is clamped
    by the neighbours rather than allowed to overtake them."""
    values = np.ones(60)
    values[5] = 0.0
    bounds = [0, 10, 20, 60]
    moved = quiet_bar_bounds(bounds, music_with_envelope(values), 30)
    assert 0 < moved[1] < moved[2] < 60


def test_too_few_bounds_is_left_alone():
    music = music_with_envelope(np.zeros(100))
    assert quiet_bar_bounds([0, 100], music, 5) == [0, 100]
    assert quiet_bar_bounds(None, music, 5) is None
