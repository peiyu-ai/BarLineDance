"""The signed phase column: settling on the beat must read the opposite of lurching.

The disqualification this file exists to avoid repeating (2026-09-01): the
``modulation`` column of ``score_beat_phase_profile`` is a Fourier MAGNITUDE, so
it reads "there is once-per-beat structure" without reading WHERE, and its
time-reversal negative control passed at P=0.021.  Every test below is therefore
about sign and position, and the sharpest one -- ``test_a_half_beat_shift_flips
_the_sign`` -- is the power demonstration required before a zero reading from
this instrument may be reported as an absence (CLAUDE.md §2.1 rule 3).
"""
import numpy as np
import pytest

from tools.score_beat_phase_shape import (aligned_profile, clip_reading,  # noqa: E402
                                          half_beat_rotated, shape_of)

PERIOD, FRAMES = 15, 900
BEATS = np.arange(0, FRAMES, PERIOD)


def trace(amplitude, sign=+1.0, noise=0.0, seed=0, frames=FRAMES):
    """``sign=+1`` is slowest ON the beat (ground truth's shape); ``-1`` lurches."""
    rng = np.random.default_rng(seed)
    phase = 2 * np.pi * np.arange(frames) / PERIOD
    return (1.0 - sign * amplitude * np.cos(phase)
            + noise * rng.standard_normal(frames))


def foreign_grids(count=12, seed=3):
    rng = np.random.default_rng(seed)
    return [np.arange(0, FRAMES, float(rng.uniform(9, 26))).astype(int)
            for _ in range(count)]


# --------------------------------------------------------------- 1. the SIGN

def test_settling_on_the_beat_reads_positive():
    table = aligned_profile(trace(0.5), BEATS)
    assert shape_of(table)["settle"] > 0.3
    assert shape_of(table)["trough_bin"] == 0


def test_lurching_on_the_beat_reads_negative():
    """The generated arms' shape.  A magnitude column cannot tell these apart;
    this one must, and by sign, not by size."""
    table = aligned_profile(trace(0.5, sign=-1.0), BEATS)
    assert shape_of(table)["settle"] < -0.3
    assert shape_of(table)["peak_bin"] == 0


def test_a_half_beat_shift_flips_the_sign():
    """POWER.  Before a near-zero reading may be called 'no phase structure',
    the column has to be shown capable of producing the negative number.  A
    half-beat roll of a settling trace is the defect, synthesised."""
    speed = trace(0.5)
    settled = shape_of(aligned_profile(speed, BEATS))["settle"]
    lurched = shape_of(aligned_profile(half_beat_rotated(speed, 0, BEATS), BEATS))["settle"]
    assert settled > 0 > lurched
    assert lurched == pytest.approx(-settled, rel=0.25)


# ---------------------------------------------------------- 2. amplitude blind

@pytest.mark.parametrize("scale", [0.1, 3.0, 17.0])
def test_scaling_the_whole_dance_changes_nothing(scale):
    """'不是能量的大小' -- an arm must not buy a good reading by moving harder."""
    base = shape_of(aligned_profile(trace(0.5), BEATS))
    scaled = shape_of(aligned_profile(trace(0.5) * scale, BEATS))
    assert base["settle"] == pytest.approx(scaled["settle"], abs=1e-9)
    assert base["depth"] == pytest.approx(scaled["depth"], abs=1e-9)


def test_a_constant_offset_changes_nothing():
    base = shape_of(aligned_profile(trace(0.5), BEATS))
    lifted = shape_of(aligned_profile(trace(0.5) + 9.0, BEATS))
    assert base["settle"] == pytest.approx(lifted["settle"], abs=1e-9)


# ------------------------------------------------- 3. it must not invent a shape

def test_noise_does_not_manufacture_a_trough():
    """THE regression.  This file's first draft aligned every window to its own
    minimum, which forces a V out of anything: ground truth, a generated arm and
    a shuffled control all read peak-to-trough 0.76 under it.  Beat-aligned, a
    trace with no beat structure must read a small depth and a settle near 0."""
    rng = np.random.default_rng(5)
    table = aligned_profile(1.0 + rng.standard_normal(FRAMES), BEATS)
    shape = shape_of(table)
    assert abs(shape["settle"]) < 0.15, shape
    assert shape["depth"] < 0.5, shape


def test_a_foreign_beat_grid_reads_near_zero():
    """What makes ``settle`` a claim about THIS music's beats: score the same
    locked motion against other clips' grids and the sign must vanish."""
    speed = trace(0.5)
    reading = clip_reading(speed, BEATS, foreign_grids())
    assert reading["settle"] > 0.3
    assert abs(reading["null_settle"]) < 0.15, reading["null_settle"]
    assert reading["settle_gain"] > 0.2


def test_depth_rises_with_the_lock_and_the_null_does_not():
    grids = foreign_grids()
    readings = [clip_reading(trace(a, noise=0.2), BEATS, grids) for a in (0.0, 0.2, 0.6)]
    assert [r["depth"] for r in readings] == sorted(r["depth"] for r in readings)
    nulls = [r["null_depth"] for r in readings]
    assert max(nulls) - min(nulls) < 0.5 * (readings[-1]["depth"] - readings[0]["depth"])


# ----------------------------------------------------------------- 4. plumbing

def test_too_few_windows_returns_nothing_rather_than_a_number():
    assert aligned_profile(trace(0.5, frames=60), np.arange(0, 60, PERIOD)) is None


def test_too_few_nulls_returns_nothing_rather_than_a_number():
    assert clip_reading(trace(0.5), BEATS, foreign_grids(count=2)) is None


def test_the_profile_is_averaged_beat_aligned_not_self_aligned():
    """Stated as an invariant on the code path, because the difference is
    invisible in the output: two windows whose troughs sit at DIFFERENT phases
    must partially cancel.  Self-alignment would stack them into one deep V."""
    frames = FRAMES
    half = frames // 2
    phase = 2 * np.pi * np.arange(frames) / PERIOD
    swing = np.concatenate([-np.cos(phase[:half]), np.cos(phase[half:])])
    both = shape_of(aligned_profile(1.0 + 0.5 * swing, BEATS))
    one = shape_of(aligned_profile(trace(0.5), BEATS))
    assert both["depth"] < 0.4 * one["depth"], (both, one)
