"""Tests for the plan-collapse probe.

The probe's whole value is that it reproduces ``infer_plan`` rather than
approximating it, so the load-bearing test is the one that runs both on the
same input and demands the same answer.  Without it the probe could measure a
pipeline that no longer exists and its stage attribution would be fiction.
"""

import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.probe_plan_collapse import accumulate, shape_of


class ShapeTests:
    pass


def test_shape_ignores_transition_frames():
    # Label 0 is "no atomic movement here"; counting it would let a plan look
    # diverse by being mostly empty.
    assert shape_of([0, 0, 0, 5, 5, 7])["classes_used"] == 2
    assert shape_of([0, 0, 0, 5, 5, 7])["frames"] == 3


def test_a_flat_marginal_reads_low_and_a_collapsed_one_reads_high():
    flat = shape_of(list(range(1, 21)) * 10)
    collapsed = shape_of([3] * 190 + list(range(1, 11)))
    assert flat["top_share"] == pytest.approx(0.05)
    assert collapsed["top_share"] > 0.9
    assert flat["classes_used"] == 20
    assert collapsed["classes_used"] == 10  # 3 is already inside 1..10


def test_top3_share_is_the_sum_of_the_three_largest():
    values = [1] * 50 + [2] * 30 + [3] * 15 + [4] * 5
    assert shape_of(values)["top3_share"] == pytest.approx(0.95)


def test_an_all_transition_plan_is_reported_as_empty_not_as_diverse():
    empty = shape_of([0, 0, 0, 0])
    assert empty["frames"] == 0
    assert empty["top_share"] is None
    assert empty["classes_used"] == 0


def test_accumulate_pools_across_clips():
    store = {}
    accumulate(store, "1_raw", np.array([1, 1, 2, 0]))
    accumulate(store, "1_raw", np.array([2, 3, 0, 0]))
    assert dict(store["1_raw"]) == {1: 2, 2: 2, 3: 1}


class PipelineAgreementTests:
    pass


def test_the_probe_reproduces_infer_plan_stage_for_stage():
    """The probe's last stage must equal what ``infer_plan`` returns.

    Both are driven by the same fake planner so the draw is fixed; if
    ``infer_plan`` grows a step the probe does not mirror, this fails.
    """
    import infer_atomic
    from tools.probe_plan_collapse import stage_plans

    torch.manual_seed(0)
    frames, window = 300, 120
    music = torch.zeros(frames, 35)
    music[::15, 34] = 1.0  # a beat every half second, for the bar grid

    class FakePlanner(torch.nn.Module):
        """Returns a fixed, reproducible label field per window."""

        def sample(self, music_batch, padding_mask=None, temperature=1.0,
                   deterministic=False, guidance_weight=1.0,
                   transition_logit_bias=0.0, **kwargs):
            generator = torch.Generator().manual_seed(7)
            return torch.randint(0, 6, (len(music_batch), music_batch.shape[1]),
                                 generator=generator)

    planner = FakePlanner()
    stats = {}
    expected = infer_atomic.infer_plan(
        planner, music, window, torch.device("cpu"), plan_stride=15,
        plan_fusion="vote", plan_bar_grid=True, plan_bar_beats=4,
        vote_window=5, min_segment_length=6, stats=stats)
    stages = stage_plans(planner, music, window, torch.device("cpu"),
                         temperature=1.0, guidance_weight=1.0, plan_stride=15,
                         fusion="vote", tie_break="centre", bar_beats=4,
                         vote_window=5, min_segment_length=6,
                         transition_policy="protect", merge_order="shortest")
    assert torch.equal(stages["4_refined"], expected)


def test_the_stages_are_reported_in_pipeline_order():
    from tools.probe_plan_collapse import stage_plans
    import infer_atomic

    class FakePlanner(torch.nn.Module):
        def sample(self, music_batch, padding_mask=None, **kwargs):
            return torch.full((len(music_batch), music_batch.shape[1]), 3)

    music = torch.zeros(200, 35)
    music[::15, 34] = 1.0
    stages = stage_plans(FakePlanner(), music, 120, torch.device("cpu"),
                         temperature=1.0, guidance_weight=1.0, plan_stride=15,
                         fusion="vote", tie_break="centre", bar_beats=4,
                         vote_window=5, min_segment_length=6,
                         transition_policy="protect", merge_order="shortest")
    assert list(stages) == ["1_raw_draw", "2_fused", "3_bar_snapped", "4_refined"]
    # A planner that only ever says 3 must survive every stage as 3: if a stage
    # can turn a constant plan into something else, the probe is measuring that
    # stage's bug rather than the collapse.
    for labels in stages.values():
        assert set(np.unique(labels.numpy())) <= {3}
