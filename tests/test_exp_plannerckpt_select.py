"""Tests for the planner-checkpoint selection criterion.

The load-bearing ones are not the arithmetic checks.  They are:

  * the gate reproduces the operator's own verdict on the two checkpoints that
    have one (epoch 6 rejected, epoch 12 chosen by eye) -- CLAUDE.md section 2.1
    rule 2, a positive control with the right direction, not merely a criterion
    that is able to fail;
  * the rule REFUSES a checkpoint it could not measure instead of ranking it,
    because a silently dropped checkpoint is indistinguishable from a losing one;
  * column (c) separates two plans that every summary column reads as identical,
    which is the whole reason the column exists (score_plan_shape's docstring:
    a clip with one 20 s hold still has five segments).
"""

import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.exp_plannerckpt_select import (Refusal, class_histogram,  # noqa: E402
                                          cross_clip_novelty, default_gates,
                                          fit_length, frame_agreement,
                                          jensen_shannon, select,
                                          swap_partners)
from tools.score_plan_shape import shape_of, summarise  # noqa: E402

# Ground truth's own reading on the 20 held-out clips, re-derived from
# data/wild3d/txy_t_labels through tools.score_plan_shape and recorded in
# runs/t_plan_shape.json.  The gate is built from these five numbers.
GT_SHAPE = {
    "filler_share": 0.3035863764082386,
    "segments": 5.0,
    "median_segment_s": 1.9666666666666666,
    "longest_hold_median_s": 3.433333333333333,
    "longest_hold_p90_s": 4.420000000000001,
    "longest_hold_max_s": 9.2,
}
# The two arms whose verdict the operator already gave (runs/t_plan_shape.json).
EP6 = {"checkpoint": "planner_epoch6_step504.pt", "js_train": 0.05,
       "filler_share": 0.004374283165540971, "segments": 5.0,
       "longest_hold_median_s": 5.833333333333333,
       "longest_hold_p90_s": 15.41666666666667, "longest_hold_max_s": 20.4}
EP12 = {"checkpoint": "planner_epoch12_step1008.pt", "js_train": 0.20,
        "filler_share": 0.2970294864540537, "segments": 4.5,
        "longest_hold_median_s": 3.75,
        "longest_hold_p90_s": 7.513333333333335, "longest_hold_max_s": 10.4}
EP20 = {"checkpoint": "planner_epoch20_step1680.pt", "js_train": 0.10,
        "filler_share": 0.008967048493717515, "segments": 5.0,
        "longest_hold_median_s": 5.566666666666666,
        "longest_hold_p90_s": 9.253333333333334,
        "longest_hold_max_s": 12.266666666666667}


# --------------------------------------------------------------------------
# (1) the divergence column
# --------------------------------------------------------------------------
def test_js_of_identical_marginals_is_zero():
    counts = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
    assert jensen_shannon(counts, counts) == pytest.approx(0.0, abs=1e-12)
    # and unnormalised copies are the same distribution, so still zero
    assert jensen_shannon(counts, counts * 17.5) == pytest.approx(0.0, abs=1e-12)


def test_js_is_bounded_by_one_and_reaches_it_on_disjoint_support():
    left = np.array([1.0, 1.0, 0.0, 0.0])
    right = np.array([0.0, 0.0, 1.0, 1.0])
    assert jensen_shannon(left, right) == pytest.approx(1.0)


def test_a_collapsed_marginal_scores_worse_than_the_ground_truth_marginal():
    """One class at 40% must be further from train than ground truth is.

    This is the direction check.  ``js_train`` is only allowed to decide
    anything if a plan that puts 40% of its danced frames in one class -- the
    shape the operator sees as 招单一 -- reads WORSE than the real test
    distribution does against the same reference.
    """
    train = np.full(20, 1.0 / 20)          # the flat-ish training marginal
    gt = np.array([500, 159, 363, 83, 387, 329, 390, 56, 363, 243,
                   387, 423, 290, 450, 335, 473, 226, 1318, 365, 339],
                  dtype=float)             # runs/t_gt_class_usage.json
    collapsed = gt.copy()
    collapsed[2] = 0.4 / 0.6 * (gt.sum() - gt[2])   # class 3 -> 40% of the mass
    assert jensen_shannon(collapsed, train) > jensen_shannon(gt, train)
    # and the collapse is what moved it, not the renormalisation
    assert jensen_shannon(gt, train) < 0.15


def test_a_class_the_plan_never_uses_costs_more_than_one_it_underuses():
    train = np.full(20, 1.0)
    underused = np.full(20, 1.0)
    underused[7] = 0.2
    dropped = np.full(20, 1.0)
    dropped[7] = 0.0
    assert jensen_shannon(dropped, train) > jensen_shannon(underused, train)


def test_class_histogram_excludes_filler_and_out_of_range_labels():
    counts = class_histogram([0, 0, 1, 1, 20, 21, -1], num_classes=20)
    assert counts[0] == 2 and counts[19] == 1
    assert counts.sum() == 3          # the two zeros, the 21 and the -1 are gone


# --------------------------------------------------------------------------
# (2) the refusal
# --------------------------------------------------------------------------
def test_the_rule_refuses_a_checkpoint_with_a_missing_column():
    holed = dict(EP12)
    holed["checkpoint"] = "planner_epochX.pt"
    holed["longest_hold_p90_s"] = None
    with pytest.raises(Refusal) as caught:
        select([EP12, holed], default_gates(GT_SHAPE))
    assert "planner_epochX.pt" in str(caught.value)
    assert "longest_hold_p90_s" in str(caught.value)


def test_the_rule_refuses_when_the_objective_is_missing():
    holed = dict(EP12)
    holed["checkpoint"] = "planner_epochY.pt"
    holed["js_train"] = None
    with pytest.raises(Refusal):
        select([holed], default_gates(GT_SHAPE))


def test_a_refusal_is_not_a_silent_skip():
    """A missing column must not simply hand the win to the other checkpoint."""
    holed = dict(EP6)
    holed["filler_share"] = None
    with pytest.raises(Refusal):
        select([EP12, holed], default_gates(GT_SHAPE))


# --------------------------------------------------------------------------
# (3) the gate reproduces the operator's verdict
# --------------------------------------------------------------------------
def test_the_gate_rejects_the_arm_the_operator_rejected_and_keeps_the_one_chosen():
    decision = select([EP6, EP12, EP20], default_gates(GT_SHAPE))
    passes = {v["checkpoint"]: v["passes_gate"] for v in decision["verdicts"]}
    assert passes["planner_epoch6_step504.pt"] is False
    assert passes["planner_epoch12_step1008.pt"] is True
    assert passes["planner_epoch20_step1680.pt"] is False
    # and the reason ep6 fails is the defect the operator named, not an
    # incidental column: a 20.4 s hold and a plan with no rest in it.
    failed = {f["column"] for v in decision["verdicts"]
              for f in v["gate_failures"] if v["checkpoint"] == EP6["checkpoint"]}
    assert {"filler_share", "longest_hold_p90_s", "longest_hold_max_s"} <= failed


def test_the_gate_outranks_the_objective():
    """A checkpoint with the best js_train still loses if it fails the gate.

    EP6 carries js_train 0.05 here, better than EP12's 0.20; the gate must
    still hand the win to EP12.  Without this ordering the criterion is just
    the marginal again and the hold columns are decoration.
    """
    decision = select([EP6, EP12], default_gates(GT_SHAPE))
    assert decision["winner"] == "planner_epoch12_step1008.pt"


def test_no_survivor_is_reported_as_no_winner_not_as_the_least_bad():
    decision = select([EP6, EP20], default_gates(GT_SHAPE))
    assert decision["survivors"] == []
    assert decision["winner"] is None


def test_the_gate_also_rejects_a_plan_chopped_too_fine():
    """The opposite failure: half-second pieces would win every hold column."""
    chopped = dict(EP12)
    chopped["checkpoint"] = "chopped.pt"
    chopped["segments"] = 40.0
    chopped["longest_hold_median_s"] = 0.5
    decision = select([chopped], default_gates(GT_SHAPE))
    assert decision["survivors"] == []


# --------------------------------------------------------------------------
# (4) column (c) separates plans that every other column reads as identical
# --------------------------------------------------------------------------
def _plan(runs):
    return np.concatenate([np.full(length, label) for label, length in runs])


def test_equal_segment_count_different_longest_hold_is_separated_by_column_c():
    """Both plans: 600 frames, 4 named segments, same filler share.

    ``segments`` and ``filler_share`` cannot tell them apart -- that is exactly
    the blindness ``score_plan_shape`` was written for -- and
    ``longest_hold_max_s`` must.
    """
    even = _plan([(1, 160), (0, 60), (2, 160), (0, 60), (3, 160), (0, 60), (4, 160)])
    held = _plan([(1, 580), (0, 60), (2, 20), (0, 60), (3, 20), (0, 60), (4, 20)])
    a, b = shape_of(even), shape_of(held)
    assert a["frames"] == b["frames"] == 820
    assert a["segments"] == b["segments"] == 4
    assert a["filler_share"] == pytest.approx(b["filler_share"])
    assert a["longest_hold_s"] == pytest.approx(160 / 30.0)
    assert b["longest_hold_s"] == pytest.approx(580 / 30.0)

    summary_even = summarise([a], 4.0)
    summary_held = summarise([b], 4.0)
    gates = dict((c, (low, high)) for c, low, high, _ in default_gates(GT_SHAPE))
    low, high = gates["longest_hold_max_s"]
    assert summary_even["longest_hold_max_s"] <= high
    assert summary_held["longest_hold_max_s"] > high


def test_a_single_collapsed_clip_survives_the_median_but_not_the_max():
    """One clip in twenty holding a pose for 20 s: the median cannot see it."""
    normal = [shape_of(_plan([(1, 100), (0, 40), (2, 100), (0, 40), (3, 100)]))
              for _ in range(19)]
    collapsed = shape_of(_plan([(1, 600), (0, 40), (2, 60)]))
    summary = summarise(normal + [collapsed], 4.0)
    assert summary["longest_hold_median_s"] == pytest.approx(100 / 30.0)
    assert summary["longest_hold_max_s"] == pytest.approx(20.0)
    gates = dict((c, (low, high)) for c, low, high, _ in default_gates(GT_SHAPE))
    assert summary["longest_hold_max_s"] > gates["longest_hold_max_s"][1]


# --------------------------------------------------------------------------
# the control columns
# --------------------------------------------------------------------------
def test_a_planner_that_ignored_music_reads_exactly_zero_on_the_swap_control():
    """Identical plans under different music mean the disagreement is 0.000.

    The control is built so that this reading is only reachable by ignoring the
    conditioning: the noise seed and the window count are held fixed, so the
    only thing that changed between the two draws is the music.
    """
    plan = _plan([(1, 100), (0, 50), (2, 100)])
    assert 1.0 - frame_agreement(plan, plan.copy()) == pytest.approx(0.0)
    changed = plan.copy()
    changed[:50] = 7
    assert 1.0 - frame_agreement(plan, changed) == pytest.approx(50 / 250)


def test_novelty_is_zero_for_twenty_identical_plans_and_positive_when_they_differ():
    same = [class_histogram(_plan([(1, 100), (2, 100)])) for _ in range(20)]
    assert cross_clip_novelty(same) == pytest.approx(0.0, abs=1e-12)
    varied = [class_histogram(_plan([(index % 20 + 1, 200)])) for index in range(20)]
    assert cross_clip_novelty(varied) == pytest.approx(1.0)


def test_swap_donor_is_the_longest_other_clip_so_music_is_cut_not_repeated():
    clips = ["a", "b", "c"]
    lengths = {"a": 100, "b": 300, "c": 200}
    donors = swap_partners(clips, lengths)
    assert donors["a"] == "b" and donors["c"] == "b"
    assert donors["b"] == "c"          # the longest clip borrows from the second
    music = np.arange(300, dtype=np.float32)[:, None]
    cut, wrapped = fit_length(music, 100)
    assert len(cut) == 100 and wrapped == 0
    tiled, wrapped = fit_length(music[:80], 100)
    assert len(tiled) == 100 and wrapped == 20


# --------------------------------------------------------------------------
# the report is reproducible from what it saves
# --------------------------------------------------------------------------
def test_the_saved_gates_carry_their_reason(tmp_path):
    decision = select([EP12], default_gates(GT_SHAPE))
    assert all(gate["why"] for gate in decision["gates"])
    round_trip = json.loads(json.dumps(decision))
    assert round_trip["winner"] == "planner_epoch12_step1008.pt"
    (tmp_path / "decision.json").write_text(json.dumps(round_trip))
