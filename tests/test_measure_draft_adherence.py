"""Controls for the draft-adherence criterion.

This criterion is about to judge whether a retrain worked, so it has to be
validated in the direction it will be used: it must read HIGH adherence on an
output that copies the draft, LOW on one that ignores it, and it must not be
fooled by an arm that simply moves less -- which is the obvious way a
smoothed-out arm could fake "following".
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.measure_draft_adherence import adherence, joint_speed, run


def draft_speed(frames=300, seed=0):
    """A draft that moves in bursts and holds between them."""
    rng = np.random.default_rng(seed)
    speed = np.full(frames, 1.0)
    for start in range(20, frames - 30, 60):
        speed[start:start + 25] = 0.05          # a hold
        speed[start + 25:start + 35] = 3.0      # a burst out of it
    return speed * rng.uniform(0.9, 1.1, frames)


def test_an_output_that_copies_the_draft_reads_as_following():
    d = draft_speed()
    got = adherence(d, d.copy())
    assert got["speed_correlation"] > 0.95
    # Where the draft holds, a follower is far below its own median.
    assert got["speed_on_draft_holds"] < 0.2


def test_an_output_that_ignores_the_draft_reads_as_not_following():
    # NEGATIVE CONTROL: the shipped model's shape -- constant motion.
    d = draft_speed()
    rng = np.random.default_rng(1)
    got = adherence(d, rng.uniform(0.9, 1.1, len(d)))
    assert abs(got["speed_correlation"]) < 0.2
    assert got["speed_on_draft_holds"] > 0.8


def test_the_two_controls_are_widely_separated():
    d = draft_speed()
    rng = np.random.default_rng(2)
    following = adherence(d, d.copy())["speed_on_draft_holds"]
    ignoring = adherence(d, rng.uniform(0.9, 1.1, len(d)))["speed_on_draft_holds"]
    assert ignoring > 4 * following


def test_an_arm_that_merely_moves_less_does_not_look_like_a_follower():
    """The obvious way to fake this, and why the normaliser is the arm's own.

    Scaling every speed down changes nothing about WHERE the arm is slow, so a
    uniformly slower arm must read exactly as unfollowing as before.
    """
    d = draft_speed()
    rng = np.random.default_rng(3)
    ignoring = rng.uniform(0.9, 1.1, len(d))
    assert (adherence(d, ignoring)["speed_on_draft_holds"]
            == pytest.approx(adherence(d, ignoring * 0.1)["speed_on_draft_holds"]))


def test_the_speed_ratio_reports_the_scale_difference_separately():
    d = draft_speed()
    got = adherence(d, d * 2.0)
    assert got["output_over_draft_speed"] == pytest.approx(2.0)
    # ...while adherence itself is unaffected by that scale.
    assert got["speed_on_draft_holds"] < 0.2


def test_a_draft_with_no_holds_is_unmeasured_rather_than_perfect():
    flat = np.full(300, 1.0)
    assert adherence(flat, flat) is None


def test_too_few_frames_are_unmeasured():
    assert adherence(np.ones(4), np.ones(4)) is None


def test_speeds_are_root_relative():
    motion = np.zeros((60, 24, 3))
    motion += np.linspace(0.0, 5.0, 60)[:, None, None]
    assert joint_speed(motion).max() < 1e-9


def test_an_arm_with_no_pairs_is_an_error_not_an_empty_score(tmp_path):
    with pytest.raises(SystemExit) as failure:
        run([("A", str(tmp_path))], tmp_path, ["absent"])
    assert "0 of 1 clips" in str(failure.value)
