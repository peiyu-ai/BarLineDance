"""The phase instrument: blind to amplitude, sensitive to timing, LOCAL.

Three properties the operator separated by hand on 2026-09-01, in order:
  1. "不是能量的大小" -- amplitude must not move it.
  2. "能量在时间上的分布,相位和峰谷值能不能和节奏同步" -- timing must.
  3. "跨clip找mean是个回退操作 ... 同舞者每一段都可能有区别" -- the unit is a
     few beats, not a clip; a dancer who moves the emphasis must not cancel.

Property 3 is the one that killed two earlier drafts of this file, so it gets
the sharpest test: a synthetic that locks to the beat for half a clip and to the
off-beat for the other half is TWO perfectly locked segments, and a whole-clip
profile scores it as noise.
"""
import numpy as np
import pytest

from tools.score_beat_phase_profile import (beat_phase, clip_reading,      # noqa: E402
                                            body_speed, local_modulations,
                                            profile, period_score,
                                            rescale_grid, reversed_speed,
                                            shuffled_speed)

PERIOD, FRAMES = 15, 900
BEATS = np.arange(0, FRAMES, PERIOD)


def locked(strength, offset=0.0, noise=0.25, seed=0, frames=FRAMES):
    rng = np.random.default_rng(seed)
    phase = 2 * np.pi * np.arange(frames) / PERIOD + offset
    return 1.0 + strength * np.sin(phase) + noise * rng.standard_normal(frames)


def switching(strength=0.6, noise=0.25, seed=0):
    """Locked on-beat for the first half, off-beat for the second."""
    rng = np.random.default_rng(seed)
    phase = 2 * np.pi * np.arange(FRAMES) / PERIOD
    half = FRAMES // 2
    swing = np.concatenate([np.sin(phase[:half]), np.sin(phase[half:] + np.pi)])
    return 1.0 + strength * swing + noise * rng.standard_normal(FRAMES)


def foreign_grids(count=12, seed=3):
    """Grids at other tempi, the shape of the real null."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        step = float(rng.uniform(9, 26))
        out.append(np.arange(0, FRAMES, step).astype(int))
    return out


# ------------------------------------------------------ 1. amplitude blindness

@pytest.mark.parametrize("scale", [0.1, 0.5, 2.0, 17.0])
def test_scaling_the_whole_dance_changes_nothing(scale):
    grids = foreign_grids()
    base = clip_reading(locked(0.4), BEATS, grids)
    scaled = clip_reading(locked(0.4) * scale, BEATS, grids)
    assert base["modulation"] == pytest.approx(scaled["modulation"], abs=1e-9)
    assert base["gain"] == pytest.approx(scaled["gain"], abs=1e-9)


def test_adding_a_constant_changes_nothing():
    grids = foreign_grids()
    base = clip_reading(locked(0.4), BEATS, grids)
    lifted = clip_reading(locked(0.4) + 5.0, BEATS, grids)
    assert base["modulation"] == pytest.approx(lifted["modulation"], abs=1e-9)


def test_it_has_not_become_the_energy_column():
    """The failure being guarded: three times the energy, same timing, must read
    the same.  If this fails the column is measuring amplitude again."""
    grids = foreign_grids()
    quiet = clip_reading(locked(0.4), BEATS, grids)["modulation"]
    loud = clip_reading(locked(0.4) * 3.0, BEATS, grids)["modulation"]
    assert quiet == pytest.approx(loud, abs=1e-9)


# --------------------------------------------------------- 2. timing sensitivity

def test_a_locked_trace_beats_a_foreign_grid():
    reading = clip_reading(locked(0.7), BEATS, foreign_grids())
    assert reading["gain"] > 0
    assert reading["percentile"] > 0.8


def test_pure_noise_does_not():
    rng = np.random.default_rng(5)
    reading = clip_reading(1.0 + rng.standard_normal(FRAMES), BEATS, foreign_grids())
    assert reading["percentile"] < 0.9


def test_modulation_rises_with_lock_strength():
    grids = foreign_grids()
    values = [clip_reading(locked(s), BEATS, grids)["modulation"]
              for s in (0.0, 0.2, 0.5, 1.0)]
    assert values == sorted(values), values


def test_block_shuffling_destroys_it_without_touching_amplitude():
    grids = foreign_grids()
    speed = locked(0.7, noise=0.2)
    broken = shuffled_speed(speed, 0, block=PERIOD + 4)
    assert broken.mean() == pytest.approx(speed.mean(), rel=0.05)
    assert broken.std() == pytest.approx(speed.std(), rel=0.08)
    assert (clip_reading(speed, BEATS, grids)["modulation"]
            > clip_reading(broken, BEATS, grids)["modulation"])


# ------------------------------------------------------------- 3. LOCALITY

def test_a_dancer_who_switches_phase_is_not_cancelled():
    """THE test.  Whole-clip averaging reads this as noise; the windowed unit
    must not.  Both halves are perfectly locked -- only the emphasis moved."""
    speed = switching()
    whole, whole_mod, _ = profile(speed, beat_phase(BEATS, len(speed)))
    windowed, _ = local_modulations(speed, BEATS)
    steady, _ = local_modulations(locked(0.6), BEATS)
    assert whole_mod < 0.25, whole_mod                    # cancelled, reads as noise
    assert np.median(windowed) > 0.8, np.median(windowed)  # kept
    assert np.median(windowed) == pytest.approx(np.median(steady), rel=0.35)


def test_the_whole_clip_statistic_really_would_have_cancelled():
    """Positive control for the test above: without it, the windowed assertion
    passes on any statistic at all, including one that never cancelled."""
    _, steady_whole, _ = profile(locked(0.6), beat_phase(BEATS, FRAMES))
    _, switch_whole, _ = profile(switching(), beat_phase(BEATS, FRAMES))
    assert steady_whole > 4 * switch_whole, (steady_whole, switch_whole)


def test_phase_spread_reports_the_switch_rather_than_hiding_it():
    grids = foreign_grids()
    steady = clip_reading(locked(0.6), BEATS, grids)
    switch = clip_reading(switching(), BEATS, grids)
    assert switch["phase_spread"] > steady["phase_spread"]


# ------------------------------------------- the phase-independent second reading

def test_period_score_finds_the_right_lag_and_is_shift_invariant():
    speed = locked(0.7, noise=0.3)
    target, neighbours = period_score(speed, PERIOD)
    assert target > neighbours
    rolled_target, rolled_neighbours = period_score(np.roll(speed, 37), PERIOD)
    assert rolled_target == pytest.approx(target, rel=0.2)


def test_period_score_is_null_on_noise():
    rng = np.random.default_rng(6)
    target, neighbours = period_score(1.0 + rng.standard_normal(FRAMES), PERIOD)
    assert abs(target - neighbours) < 0.2


# ------------------------------------------------------------------ plumbing

def test_the_discarded_rotation_null_would_have_read_zero():
    """Kept as a regression: rotating the MOTION cannot be the null, because
    modulation is a Fourier MAGNITUDE and a shift only moves its phase."""
    speed = locked(0.8, noise=0.0)
    phase = beat_phase(BEATS, FRAMES)
    _, before, _ = profile(speed, phase)
    _, after, _ = profile(np.roll(speed, 7), phase)
    assert after == pytest.approx(before, rel=0.05)


def test_body_speed_is_energy_before_the_time_average():
    rng = np.random.default_rng(10)
    joints = np.cumsum(rng.standard_normal((120, 24, 3)) * 0.01, axis=0)
    speed = body_speed(joints, smooth=1)
    relative = joints - joints[:, :1, :]
    expected = np.linalg.norm(np.diff(relative, axis=0), axis=2).mean(1) * 30.0
    assert np.allclose(speed, expected)


def test_rescale_grid_keeps_a_grid_a_grid():
    grid = np.arange(0, 600, 20)
    scaled = rescale_grid(grid, 300)
    assert scaled is not None and len(scaled) >= 4
    assert scaled[0] >= 0 and scaled[-1] <= 299
    assert np.all(np.diff(scaled) > 0)


def test_too_few_nulls_returns_nothing_rather_than_a_number():
    assert clip_reading(locked(0.5), BEATS, foreign_grids(count=2)) is None
