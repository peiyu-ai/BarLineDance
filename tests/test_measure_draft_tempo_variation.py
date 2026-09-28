"""Every criterion in the tempo tool, with the control that would fail it.

The zero controls here are the ones CLAUDE.md 2.1 asks for by name: ground
truth has no seams and ``s = 1`` everywhere, so a flat trace and a constant
stretch must read EXACTLY zero variation.  If they do not, the instrument
manufactures the effect it is used to attribute.
"""

import math
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dataset.atomic import labels_to_segments
from tools.measure_draft_tempo_variation import (LEVEL_FRAMES, SEAM_TRIM,
                                                 boundary_jumps, speed_levels,
                                                 summarise, within_clip_stretch)


def two_segments(length=30):
    labels = torch.tensor([1] * length + [2] * length)
    return labels_to_segments(labels)


def test_constant_stretch_is_the_zero_control():
    # Ground truth's own segments: native == slot, so s == 1 in every slot.
    reading = within_clip_stretch([1.0, 1.0, 1.0, 1.0])
    assert reading["sd_log"] == 0.0
    assert reading["max_over_min"] == 1.0
    assert reading["adjacent_mean_abs_log"] == 0.0
    assert reading["adjacent_max_abs_log"] == 0.0
    assert reading["segments"] == 4


def test_alternating_stretch_is_the_positive_control():
    # Half speed then double speed: the reading must be the exact ratio, so a
    # null on real data cannot be blamed on a flat instrument.
    reading = within_clip_stretch([0.5, 2.0])
    assert reading["sd_log"] == pytest_approx(math.log(2.0))
    assert reading["max_over_min"] == pytest_approx(4.0)
    assert reading["adjacent_mean_abs_log"] == pytest_approx(math.log(4.0))


def test_one_segment_refuses_rather_than_reading_zero():
    # "No spread to measure" and "measured, no spread" are different claims and
    # only the second is evidence that the draft is steady.
    assert within_clip_stretch([1.3]) is None
    assert within_clip_stretch([]) is None


def test_speed_level_never_reads_the_straddling_sample():
    # The 2026-08-19 off-by-one, as a test: speed[i] is the sample BETWEEN
    # frame i and i+1, so speed[end-1] belongs to neither side of the seam.
    segments = two_segments()
    speed = np.concatenate([np.ones(29), np.full(30, 2.0)])
    speed[29] = 100.0  # the seam pop
    levels = speed_levels(speed, segments)
    assert levels[0]["tail"] == pytest_approx(1.0)
    assert levels[1]["head"] == pytest_approx(2.0)
    jumps = boundary_jumps(speed, segments, levels)
    assert jumps[0]["frame"] == 30
    assert jumps[0]["level_log_jump"] == pytest_approx(math.log(2.0))
    # The pop is reported, in its own column, and not folded into the level.
    assert jumps[0]["step_over_median"] == pytest_approx(100.0 / np.median(speed))


def test_flat_trace_reads_no_jump_and_no_pop():
    segments = two_segments()
    speed = np.ones(59)
    jumps = boundary_jumps(speed, segments, speed_levels(speed, segments))
    assert jumps[0]["level_log_jump"] == 0.0
    assert jumps[0]["step_over_median"] == pytest_approx(1.0)


def test_short_segment_is_refused_not_averaged():
    segments = labels_to_segments(torch.tensor([1] * 4 + [2] * 30))
    speed = np.ones(33)
    levels = speed_levels(speed, segments)
    assert levels[0] is None
    assert boundary_jumps(speed, segments, levels)[0]["level_log_jump"] is None
    assert 2 * SEAM_TRIM + LEVEL_FRAMES > 4


def _clip(stretches, speeds):
    return {
        "clip": "c", "frames": 100, "transition_frame_share": 0.0,
        "segment_rows": [{"stretch": s, "draft_speed": v, "truth_speed": v,
                          "native_speed_ratio": 1.0}
                         for s, v in zip(stretches, speeds)],
        "draft_boundaries": [{"frame": 10, "left_label": 1, "right_label": 2,
                              "level_log_jump": 0.5, "step_over_median": 3.0}],
        "truth_boundaries": [{"frame": 10, "left_label": 1, "right_label": 2,
                              "level_log_jump": 0.2, "step_over_median": 1.0}],
        "draft_speed_median": 1.0, "truth_speed_median": 1.0,
    }


def test_summarise_reports_the_speed_columns_for_a_real_arm():
    report = summarise([_clip([1.0, 1.0], [1.0, 2.0])])
    assert report["speed_columns"] is True
    assert report["draft_level_log_jump_median"] == pytest_approx(0.5)
    assert report["share_exact"] == 1.0


def test_summarise_withholds_speed_columns_under_the_positive_control():
    # The control re-draws the candidate but the rendered draft is still the
    # shipped one; printing its speed under the control's name would be a flag
    # recorded but not applied.
    report = summarise([_clip([0.5, 2.0], [1.0, 2.0])], speed_columns=False)
    assert report["speed_columns"] is not True
    assert report["draft_level_log_jump_median"] is None
    assert report["per_clip"]["c"]["draft_speed_sd_log"] is None
    # The stretch columns, which the control does own, are still reported.
    assert report["within_clip_max_over_min_median"] == pytest_approx(4.0)


def pytest_approx(value, tolerance=1e-9):
    class _Approx(float):
        def __eq__(self, other):
            return abs(float(other) - value) <= tolerance
    return _Approx(value)
