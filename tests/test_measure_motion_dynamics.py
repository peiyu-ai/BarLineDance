"""Controls for the hold/dynamic-range criterion.

This criterion is about to decide whether a change to the completion model is an
improvement, so per CLAUDE.md 2.1 it has to be validated first, and validated in
the direction it will be USED: it must read high on motion that lands and holds,
low on motion that never stops, and -- the one that matters -- it must DROP when
motion is smoothed, because "the completion smooths the landings away" is the
claim it is being used to support.  A criterion that could not detect smoothing
would have no business testing a smoothing hypothesis.
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.measure_motion_dynamics import (dynamics, joint_speed, longest_run,
                                           score)

FPS = 30.0


def move_and_hold(cycles=6, move_frames=24, hold_frames=16, joints=24, seed=0):
    """Travel, then be still: what a danced move looks like.

    The holds carry a little jitter rather than freezing exactly.  A perfectly
    frozen hold is not just unrealistic for captured motion, it makes the
    clip's MEDIAN speed zero once holds outnumber moving frames, and the
    criterion then correctly refuses to score it -- the first version of this
    fixture did exactly that and read ``None``.
    """
    rng = np.random.default_rng(seed)
    pieces = [np.zeros((1, joints, 3))]
    for _ in range(cycles):
        target = rng.uniform(-0.4, 0.4, size=(1, joints, 3))
        start = pieces[-1][-1:]
        ramp = np.linspace(0.0, 1.0, move_frames)[:, None, None]
        pieces.append(start + (target - start) * ramp)
        rest = np.repeat(pieces[-1][-1:], hold_frames, axis=0)
        pieces.append(rest + rng.normal(0.0, 3e-4, size=rest.shape))
    return np.concatenate(pieces, axis=0)


def never_stops(frames=300, joints=24, seed=1):
    """Continuous motion at an even speed: nothing ever lands."""
    rng = np.random.default_rng(seed)
    time = np.linspace(0.0, 12.0 * np.pi, frames)[:, None, None]
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(1, joints, 3))
    return 0.25 * np.sin(time + phase)


def smooth(motion, width=9):
    """A moving average -- a stand-in for what a low-pass does to landings."""
    kernel = np.ones(width) / width
    flat = motion.reshape(len(motion), -1)
    padded = np.pad(flat, ((width, width), (0, 0)), mode="edge")
    out = np.stack([np.convolve(padded[:, i], kernel, mode="same")
                    for i in range(flat.shape[1])], axis=1)
    return out[width:-width].reshape(motion.shape)


def test_motion_that_lands_and_holds_reads_high():
    measured = dynamics(joint_speed(move_and_hold()))
    assert measured["hold_share"] > 0.4
    assert measured["longest_hold_seconds"] > 0.5


def test_motion_that_never_stops_reads_near_zero():
    # NEGATIVE CONTROL: without it, a criterion that reads "holds" everywhere
    # would pass the test above.
    measured = dynamics(joint_speed(never_stops()))
    assert measured["hold_share"] < 0.05
    assert measured["longest_hold_seconds"] < 0.2


def test_the_two_controls_are_widely_separated():
    landing = dynamics(joint_speed(move_and_hold()))
    mush = dynamics(joint_speed(never_stops()))
    assert landing["p90_over_p10"] > 5 * mush["p90_over_p10"]


def test_smoothing_destroys_the_holds_the_criterion_counts():
    """POWER PROOF for the actual hypothesis under test.

    The claim is "the completion model smooths the landings away".  This shows
    the criterion responds to smoothing specifically: the same motion, low-pass
    filtered, must lose hold time and dynamic range.
    """
    motion = move_and_hold()
    before = dynamics(joint_speed(motion))
    after = dynamics(joint_speed(smooth(motion, width=15)))
    assert after["longest_hold_seconds"] < before["longest_hold_seconds"]
    assert after["p90_over_p10"] < before["p90_over_p10"]


def test_the_reading_is_scale_free():
    # A dancer who moves further must not read as landing more or less.
    motion = move_and_hold()
    assert dynamics(joint_speed(motion))["hold_share"] == pytest.approx(
        dynamics(joint_speed(motion * 7.0))["hold_share"])


def test_the_reading_is_root_relative():
    motion = move_and_hold()
    travelling = motion + np.linspace(0.0, 4.0, len(motion))[:, None, None]
    assert dynamics(joint_speed(motion))["hold_share"] == pytest.approx(
        dynamics(joint_speed(travelling))["hold_share"])


def test_longest_hold_separates_one_long_stop_from_many_stutters():
    """The two columns must fail differently, or one of them is redundant.

    Thirty one-frame stops and one thirty-frame stop have the SAME hold share;
    only the second is a dancer landing.  This is why both are reported.
    """
    stutter = np.zeros(300, bool)
    stutter[::10] = True
    landed = np.zeros(300, bool)
    landed[100:130] = True
    assert stutter.mean() == pytest.approx(landed.mean())
    assert longest_run(stutter) == 1
    assert longest_run(landed) == 30


def test_a_still_clip_is_refused_rather_than_scored():
    """OLD BEHAVIOUR: ``dynamics`` returned None here and ``score`` skipped the
    clip silently -- on runs/txy_t_draft_ep12_index that was 4 of 20 clips and
    the tool printed ``clips: 16`` with no names, while exp_guidance_stillness
    printed 0.0 for the same four.  NEW BEHAVIOUR: it raises, so a caller has to
    decide what to do and cannot pool a fabricated number."""
    from tools.stillness_criterion import DegenerateMotion

    with pytest.raises(DegenerateMotion):
        dynamics(np.zeros(100))
    with pytest.raises(DegenerateMotion):
        dynamics(np.array([1.0, 2.0]))


def test_an_arm_with_no_clips_is_an_error_not_an_empty_score(tmp_path):
    with pytest.raises(SystemExit) as failure:
        score(tmp_path, ["absent"])
    assert "0 of 1 clips" in str(failure.value)


def test_sustained_holds_survive_smoothing_but_momentary_dips_do_not():
    """The column that separates a landing from a zero-crossing.

    Ground truth keeps 1.76% of frames held after a 9-frame average; every
    generated arm measured so far drops to 0.00%.  This pins the mechanism on
    hand-built samples: a body that stops stays stopped under averaging, a body
    that oscillates through zero does not.
    """
    from tools.measure_motion_dynamics import low_pass, SMOOTH_WIDTH

    landing = move_and_hold()
    oscillating = never_stops()
    kept = dynamics(joint_speed(low_pass(landing, SMOOTH_WIDTH)))
    lost = dynamics(joint_speed(low_pass(oscillating, SMOOTH_WIDTH)))
    assert kept["hold_share"] > 0.2, "a real hold must survive smoothing"
    assert lost["hold_share"] < 0.01, "a zero-crossing must not survive it"


def test_low_pass_preserves_shape_and_length():
    from tools.measure_motion_dynamics import low_pass

    motion = move_and_hold()
    assert low_pass(motion, 9).shape == motion.shape
    assert np.allclose(low_pass(motion, 1), motion)
