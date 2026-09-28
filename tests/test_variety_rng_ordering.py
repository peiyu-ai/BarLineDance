"""A clip's retrieval draws must depend on its NAME, never on its neighbours.

The defect (measured 2026-08-31, 8 clips, everything else byte-identical):
``variety_rng`` was one ``np.random.default_rng(seed)`` for the whole run and
its draws were consumed in clip order, so reversing the clip list moved the
DRAFT on 6 of 8 clips by up to 2.19 m -- while appending 12 clips to the end
moved nothing.  Order, not length; the retrieval, not the noise.  Two arms
scored on differently ordered lists were comparing different dances.

This is the second appearance of one bug: ``sample_seed`` fixed exactly this
for the diffusion noise on the same day, and the retrieval kept it because the
fix went where the symptom was rather than to the class.  Hence the tests below
assert the PROPERTY (a clip depends only on ``(seed, name)``) rather than any
particular construction of the generator.
"""
import numpy as np
import pytest

from infer_atomic import _variety_rng, sample_seed


NAMES = ["wild_v5:1:clip000", "wild_v5:2:clip000", "wild_v5:3:clip001"]


def draws(seed, names, count=6):
    """What each named clip would draw, in the order ``names`` is processed."""
    return {name: _variety_rng(seed, name).integers(1000, size=count).tolist()
            for name in names}


def test_a_clip_draws_the_same_whatever_order_the_run_is_in():
    forward = draws(7, NAMES)
    backward = draws(7, NAMES[::-1])
    assert forward == backward


def test_a_clip_draws_the_same_when_other_clips_are_added():
    small = draws(7, NAMES)
    large = draws(7, NAMES + ["wild_v5:9:clip000", "wild_v5:8:clip000"])
    for name in NAMES:
        assert small[name] == large[name]


def test_the_shared_run_generator_really_would_have_failed_this():
    """Positive control for the two tests above: without it they would pass on
    any construction at all, including one that never had the bug to begin
    with.  This reproduces the old line and shows it changing with order."""
    def old_style(names, count=6):
        rng = np.random.default_rng(7)                      # ONE for the run
        return {name: rng.integers(1000, size=count).tolist() for name in names}
    forward = old_style(NAMES)
    backward = old_style(NAMES[::-1])
    assert forward != backward
    assert forward[NAMES[0]] != backward[NAMES[0]]


def test_different_clips_still_get_different_draws():
    """The feature must survive the fix: two clips must not become identical,
    which is what a constant seed would do."""
    made = draws(7, NAMES)
    assert len({tuple(v) for v in made.values()}) == len(NAMES)


def test_different_seeds_still_move_it():
    assert draws(7, NAMES) != draws(8, NAMES)


def test_it_is_a_separate_stream_from_the_noise():
    """Kept separate so that changing the variety draw cannot silently re-roll
    the diffusion noise for the same clip, and vice versa."""
    name = NAMES[0]
    assert _variety_rng(11, name).integers(2 ** 31 - 1) != sample_seed(11, name)


@pytest.mark.parametrize("name", NAMES)
def test_the_seed_is_a_legal_numpy_seed(name):
    assert 0 <= sample_seed(20260901, "variety|" + name) < 2 ** 31 - 1
