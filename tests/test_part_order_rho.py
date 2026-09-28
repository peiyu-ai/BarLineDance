"""The part-ordering column: does an arm land the beat with the parts a dancer does?

The two tests that matter are ``test_inverted_ordering_reads_negative`` (the
defect this column was built to see, which pooled ``settle`` cannot: measured
2026-09-12, both generated arms BEAT ground truth on pooled settle while
inverting the part ordering) and ``test_a_part_that_fails_the_flip_control_is_dropped``
(a broken channel must not vote -- elbows fails the half-beat control on this
corpus and used to be averaged into the spread anyway).
"""
import numpy as np
import pytest

from tools.score_beat_phase_shape import part_order_rho, validated_parts

PARTS = ["feet", "hips_knees", "torso", "shoulders", "elbows", "hands"]
# The measured ground truth on the 20 eval clips: the lower body lands the beat.
TRUTH = {"feet": 0.0943, "hips_knees": 0.0649, "torso": 0.0477,
         "shoulders": 0.0207, "elbows": 0.0257, "hands": 0.0233}


def test_ground_truth_against_itself_is_one():
    assert part_order_rho(TRUTH, TRUTH, PARTS) == pytest.approx(1.0)


def test_inverted_ordering_reads_negative():
    """The shipped aligned arm's own numbers: the beat is carried by the
    shoulders and the hips barely move on it."""
    arm = {"feet": 0.0339, "hips_knees": 0.0093, "torso": 0.1085,
           "shoulders": 0.1201, "elbows": 0.0303, "hands": 0.0782}
    assert part_order_rho(arm, TRUTH, PARTS) < -0.5


def test_magnitude_does_not_count_only_order():
    """An arm that brakes three times as hard as the dancer, in the dancer's
    order, is a different defect and must not be punished by THIS column."""
    tripled = {k: v * 3.0 for k, v in TRUTH.items()}
    assert part_order_rho(tripled, TRUTH, PARTS) == pytest.approx(1.0)


def test_an_arm_that_brakes_everywhere_equally_is_not_rewarded():
    flat = {k: 0.1 for k in PARTS}
    rho = part_order_rho(flat, TRUTH, PARTS)
    assert not (rho > 0.5), rho


def test_a_part_that_fails_the_flip_control_is_dropped():
    """elbows is the measured case: ground truth +0.0257 and the half-beat roll
    +0.0136 -- same sign, so rolling the dancer half a beat does not change what
    that channel says, and it cannot be reading beat phase."""
    rotated = {"feet": -0.1280, "hips_knees": -0.0126, "torso": -0.0473,
               "shoulders": -0.0193, "elbows": 0.0136, "hands": -0.0648}
    used = validated_parts(TRUTH, rotated, PARTS)
    assert "elbows" not in used
    assert set(used) == {"feet", "hips_knees", "torso", "shoulders", "hands"}


def test_dropping_a_part_changes_the_reading():
    """If excluding the broken channel never moved the number, the exclusion
    would be decoration rather than a judgement."""
    arm = {"feet": 0.0339, "hips_knees": 0.0093, "torso": 0.1085,
           "shoulders": 0.1201, "elbows": 5.0, "hands": 0.0782}
    all_parts = part_order_rho(arm, TRUTH, PARTS)
    without = part_order_rho(arm, TRUTH,
                             [p for p in PARTS if p != "elbows"])
    assert all_parts != pytest.approx(without)


def test_too_few_parts_is_nan_not_a_number():
    assert np.isnan(part_order_rho({"feet": 1.0}, TRUTH, ["feet"]))


def test_missing_and_nan_parts_are_skipped_not_zeroed():
    arm = dict(TRUTH)
    arm["torso"] = float("nan")
    del arm["hands"]
    # The surviving four are still in the dancer's order, so it must read 1.0 --
    # a zero-fill would drag it down and look like a defect that is not there.
    assert part_order_rho(arm, TRUTH, PARTS) == pytest.approx(1.0)
