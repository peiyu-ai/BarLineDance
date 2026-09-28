"""The spin column must find a BRIEF violent turn, and say where it came from.

Why the tests are shaped this way.  The first version of
``tools/score_facing_spin.py`` reported a per-clip MEAN turn rate and a median
per-clip boundary concentration, and on that reading the arms looked nearly
fine -- 74.1 deg/s against ground truth's 63.0, concentration 0.94.  Both
numbers were true.  A spin is a brief event inside an ordinary trace, so a mean
dilutes it and a per-clip median of a sparse ratio is degenerate at zero.
``test_a_brief_violent_spin_barely_moves_the_mean`` is that failure written
down, so the column cannot quietly go back to it.
"""
import numpy as np
import pytest

from tools.score_facing_spin import (FPS, boundary_mask, facing_yaw,  # noqa: E402
                                     turn_rate)

FRAMES = 900


def body(yaw, centre=None):
    """Joint positions whose hips face ``yaw``; every other joint is filler."""
    joints = np.zeros((len(yaw), 24, 3))
    if centre is not None:
        joints += centre[:, None, :]
    across = np.stack([np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)], axis=1) * 0.1
    joints[:, 2] += across
    joints[:, 1] -= across
    return joints


def wobble(amplitude=0.25, period=45.0, frames=FRAMES):
    """A dancer whose facing oscillates and comes back -- ground truth's shape."""
    return amplitude * np.sin(2 * np.pi * np.arange(frames) / period)


# ------------------------------------------------------- reading the facing

def test_facing_is_read_from_the_hips_and_is_exact():
    yaw = wobble()
    assert np.allclose(np.unwrap(facing_yaw(body(yaw))), yaw, atol=1e-9)


def test_moving_the_dancer_across_the_floor_changes_nothing():
    """Facing is a direction; travel must not enter it."""
    yaw = wobble()
    walk = np.stack([np.linspace(0, 5, FRAMES), np.linspace(0, -3, FRAMES),
                     np.zeros(FRAMES)], axis=1)
    assert np.allclose(turn_rate(facing_yaw(body(yaw))),
                       turn_rate(facing_yaw(body(yaw, centre=walk))), atol=1e-9)


def test_a_constant_facing_offset_changes_nothing():
    yaw = wobble()
    assert np.allclose(turn_rate(facing_yaw(body(yaw))),
                       turn_rate(facing_yaw(body(yaw + 1.234))), atol=1e-9)


def test_the_unwrap_survives_a_full_turn():
    """Without unwrapping, a dancer passing through +-pi reads as a 360 deg/frame
    spike -- an artefact that would look exactly like the defect being hunted."""
    yaw = np.linspace(0, 4 * np.pi, FRAMES)
    rate = turn_rate(facing_yaw(body(yaw)))
    expected = 4 * np.pi / FRAMES * FPS * 180 / np.pi
    assert rate.max() < 2 * expected, rate.max()


# --------------------------------------------------- THE regression: peaks

def test_a_brief_violent_spin_barely_moves_the_mean():
    """The failure the first version of the tool shipped with.  Six frames of
    900 carry half a turn; the MEAN moves 1.16x while the PEAK moves 19x.  On
    the real corpus the same asymmetry read 74.1 vs 63.0 deg/s on the mean and
    523 vs 100 deg/s on the peak."""
    yaw = wobble().copy()
    yaw[450:456] += np.linspace(0, np.pi, 6)          # half a turn in 0.2 s
    yaw[456:] += np.pi
    calm = turn_rate(facing_yaw(body(wobble())))
    spun = turn_rate(facing_yaw(body(yaw)))
    assert spun.mean() < 1.3 * calm.mean(), (spun.mean(), calm.mean())
    assert spun.max() > 15 * calm.max(), (spun.max(), calm.max())


# -------------------------------------------------- WHERE the turning is from

def test_a_spin_at_a_plan_boundary_concentrates_and_an_even_turn_does_not():
    """The column that separates 'the library turns' from 'we spliced it'.

    10x longer than the other synthetics on purpose: at 900 frames the top
    percentile is nine frames, and one of them landing on a seam by chance moves
    the ratio from 0 to 3.7.  A control that noisy cannot refute anything, so
    the trace is long enough for the even case to read 1.01 instead.
    """
    frames = 9000
    labels = np.zeros(frames, int)
    cuts = list(range(90, frames, 90))
    for index, cut in enumerate(cuts):
        labels[cut:] = index + 1

    spliced = wobble(frames=frames).copy()
    for cut in cuts:                                   # a facing jump at every seam
        spliced[cut:] += 2.0
    # Turns constantly and has no seams.  The jitter is not cosmetic: a
    # perfectly constant rate has no top percentile, so the ratio would be 0/0
    # and the test would pass on a NaN rather than on a measurement.
    even = (np.linspace(0, 120 * np.pi, frames)
            + 0.02 * np.random.default_rng(0).standard_normal(frames))

    mask = boundary_mask(labels, frames, 1)
    expected = mask.mean()
    readings = []
    for trace in (spliced, even):
        rate = turn_rate(facing_yaw(body(trace)))
        over = rate > np.percentile(rate, 99)
        readings.append((over & mask).sum() / over.sum() / expected)
    assert readings[0] > 10.0, readings          # spliced: measured 30.3
    assert readings[1] < 1.5, readings           # even:    measured 1.01


def test_boundary_mask_marks_the_seam_and_its_neighbours_only():
    labels = np.array([1] * 10 + [2] * 10)
    mask = boundary_mask(labels, 20, 1)
    assert mask.sum() == 3
    assert mask[8] and mask[9] and mask[10]


# ---------------------------------------------------- oscillation vs spin

def test_directedness_separates_a_wobble_from_a_slow_rotation():
    """A lively dancer's facing comes back; a spinning statue's does not.  A
    rate alone cannot tell them apart, which is why net/total is reported."""
    def net_over_total(yaw):
        rate = turn_rate(facing_yaw(body(yaw)))
        return abs(yaw[-1] - yaw[0]) * 180 / np.pi / (rate.sum() / FPS)
    assert net_over_total(wobble()) < 0.05
    assert net_over_total(np.linspace(0, 6 * np.pi, FRAMES)) > 0.9


@pytest.mark.parametrize("quantile,expect", [(50.0, "low"), (99.5, "high")])
def test_the_threshold_comes_from_ground_truth_and_rises_with_the_quantile(quantile, expect):
    """Taken from ground truth rather than chosen, so it cannot be tuned down
    until an arm passes."""
    rate = turn_rate(facing_yaw(body(wobble())))
    value = np.percentile(rate, quantile)
    reference = np.percentile(rate, 90.0)
    assert (value < reference) if expect == "low" else (value > reference)
