"""Controls for the sustained-stillness WINDOW criterion (tools/exp_guidance_stillness.py).

Each test is one of the four checks CLAUDE.md 2.1 asks of a criterion before it
may judge: a negative that must read zero, a positive whose direction is known,
the power proof that the smoothing distinguishes a landing from a zero
crossing, and (data present) the calibration against the 22.1 table.
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools import exp_guidance_stillness as m  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]


def _score(joints, draws=100, seed=3):
    return m.score_clip(joints, np.random.default_rng([seed, 1]), draws)


def test_constant_velocity_reads_no_stillness():
    row = _score(m.synthetic_constant_velocity())
    assert row["windows"] == 100
    assert row["window_share"] == 0.0
    assert row["clip_sustained_hold_share"] == 0.0


def test_move_then_hold_reads_stillness_and_it_survives_smoothing():
    # 3 s move + 1 s hold, 150-frame (5 s) windows: every window overlaps a hold.
    row = _score(m.synthetic_move_then_hold())
    assert row["window_share"] >= 0.95
    # The whole-clip sustained share sits near the 25% duty cycle of the hold.
    assert 0.15 <= row["clip_sustained_hold_share"] <= 0.35
    assert row["clip_sustained_longest_hold_s"] >= 0.5


def test_drift_plus_oscillation_has_slow_frames_but_no_sustained_stillness():
    joints = m.synthetic_drift_plus_oscillation()
    raw = m.joint_speed(joints)
    raw_share = m.held_frames(raw).mean()
    assert 0.03 <= raw_share <= 0.20, raw_share  # ~9% of frames are crossings
    row = _score(joints)
    assert row["window_share"] == 0.0, "smoothing must remove crossings, else jitter reads as stillness"
    assert row["clip_sustained_hold_share"] == 0.0


def test_the_oscillation_control_is_not_sampled_at_an_exact_frame_divisor():
    """Regression: hz=5.0 at 30 fps is 6.000 frames/cycle, so the discrete
    signal samples the same six phases forever and never lands near the zero
    crossing -- the control then reads 0.0000 raw hold share while asserting it
    has slow frames, i.e. it silently stops being a control.  Pin BOTH the
    aliased reading and the working one so the default cannot drift back."""
    aliased = m.joint_speed(m.synthetic_drift_plus_oscillation(hz=5.0))
    assert m.held_frames(aliased).mean() == 0.0, "5 Hz should alias -- if this "\
        "fires the sampling changed and the comment below needs rechecking"
    assert 30.0 % 5.0 == 0.0

    working = m.joint_speed(m.synthetic_drift_plus_oscillation())
    assert m.held_frames(working).mean() > 0.03
    assert 30.0 % 4.7 != 0.0


def test_the_frozen_body_control_is_refused_by_control_readings():
    """The fourth control (2026-09-04): a motionless body must come back
    REFUSED, not scored.  Before, this row read window_share 0.0 /
    p90_over_p10 0.0 / sustained_hold 0.0 -- "completely still" printed as "no
    stillness" -- and it is what two of the twenty T-line eval clips got as
    their retrieval draft (DANCE_QUALITY_DEFECTS.md 23.9/23.10)."""
    controls = m.control_readings(draws=20)
    frozen = controls["frozen_body"]
    assert frozen["window_share"] is None
    assert frozen["clip_sustained_hold_share"] is None
    assert frozen["clip_measurable"] is False
    assert frozen["windows_dropped"] == frozen["windows"] == 20
    # And the three older controls are still MEASURED, so the refusal has not
    # simply swallowed everything.
    for name in ("constant_velocity", "move_then_hold", "drift_plus_oscillation"):
        assert controls[name]["clip_measurable"] is True, name
        assert controls[name]["windows_dropped"] == 0, name


def test_pure_sinusoid_is_not_the_jitter_control():
    # Documented limit: a relative threshold is scale-invariant under a linear
    # filter, so a pendulum keeps its turning-point stops after smoothing.
    rng = np.random.default_rng(0)
    direction = rng.normal(size=(24, 3)); direction[0] = 0.0
    t = np.arange(600) / m.FPS
    joints = np.sin(2 * np.pi * 5.0 * t)[:, None, None] * 0.05 * direction[None]
    assert _score(joints)["window_share"] == 1.0


def test_window_threshold_is_the_windows_own_median():
    # A clip that is fast for its first half and slow-but-moving for its second:
    # with a CLIP median threshold the whole second half would read held.  With
    # the window's own median (the 22.1 form) a window entirely inside the slow
    # half has a uniform speed and no held frames.
    fast = m.synthetic_constant_velocity(frames=300, seed=1) * 10.0
    slow = m.synthetic_constant_velocity(frames=300, seed=2)
    slow = slow + fast[-1:]  # continue from where the fast half ended
    joints = np.concatenate([fast, slow[1:]], axis=0)
    starts = np.array([320, 340, 360, 400])  # all inside the slow half
    has_hold, _, dropped, _ = m.score_windows(joints, starts)
    assert dropped == 0, "these windows are measurable; nothing should be refused"
    assert not has_hold.any()


def test_clip_shorter_than_window_yields_no_windows_not_a_crash():
    row = _score(m.synthetic_constant_velocity(frames=100))
    assert row["windows"] == 0 and row["window_share"] is None


def test_same_seed_same_windows_across_arms():
    a = m.window_starts(600, np.random.default_rng([7, 0]), 20)
    b = m.window_starts(600, np.random.default_rng([7, 0]), 20)
    assert np.array_equal(a, b)


def test_sign_test_is_exact_and_two_sided():
    assert m.sign_test([1, 2, 3], [1, 2, 3]) == {"n": 0, "wins": 0, "p": 1.0}
    r = m.sign_test([2] * 10, [1] * 10)
    assert r["n"] == 10 and r["wins"] == 10
    assert abs(r["p"] - 2 / 1024) < 1e-12


def test_whole_clip_columns_match_measure_motion_dynamics():
    from tools import measure_motion_dynamics as ref
    joints = m.synthetic_move_then_hold(seed=5)
    mine = _score(joints)
    theirs = ref.dynamics(ref.joint_speed(joints))
    smoothed = ref.dynamics(ref.joint_speed(ref.low_pass(joints, ref.SMOOTH_WIDTH)))
    assert mine["clip_p90_over_p10"] == pytest.approx(theirs["p90_over_p10"])
    assert mine["clip_raw_hold_share"] == pytest.approx(theirs["hold_share"])
    assert mine["clip_sustained_hold_share"] == pytest.approx(smoothed["hold_share"])
    assert mine["clip_sustained_longest_hold_s"] == pytest.approx(smoothed["longest_hold_seconds"])


@pytest.mark.skipif(not (REPO / "runs/txy_t_gt_eval/motion").exists()
                    or not (REPO / "runs/eval_clips_txy_t20.txt").exists(),
                    reason="T-line ground truth not present")
def test_ground_truth_reads_in_the_22_1_band():
    clips = [l.strip() for l in (REPO / "runs/eval_clips_txy_t20.txt").read_text().splitlines()
             if l.strip()]
    truth = m.score_arm(REPO / "runs/txy_t_gt_eval/motion", clips, 20260904, 50)
    # 22.1: ground truth 29%, training windows 27%.  A re-implementation that
    # reads far outside this band is not that criterion.
    assert 0.20 <= truth["window_share_pooled"] <= 0.38, truth["window_share_pooled"]
