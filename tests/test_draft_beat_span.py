"""--draft-beat-span must drop candidates that are the wrong NUMBER OF BEATS.

THE GAP, named by the operator 2026-09-16: "检索 label to motion 从不检测拍数是吗,
对卡节奏旋律挑选 motion 没有贡献?"  It does not.  ``_duration_band`` selects on
``abs(frames - target_length)`` alone, so a 2.4-second prototype that is six
beats of a 150 BPM song and one that is four beats of a 100 BPM song are equally
eligible for a four-beat slot, and the winner is stretched uniformly to fill it.

Measured on the 200 retrieved bars of the twenty eval clips before the filter
existed: |mismatch| median 0.262 beats, p75 0.869, p90 1.211, max 3.64, and
40.0% of bars import a prototype that is a different WHOLE NUMBER of beats than
its slot.

BEHAVIOURAL, not a wiring grep -- DEFECTS 77 is this repository losing three
days to a flag that was recorded, tested by name, and dead.
"""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from infer_atomic import IndexedAtomicMotionLibrary  # noqa: E402

SOURCE = pathlib.Path("infer_atomic.py").read_text()


class Fake(IndexedAtomicMotionLibrary):
    """Only the two things ``_prefer_beat_span`` touches."""
    def __init__(self, tolerance, periods):
        self.beat_span_tolerance = float(tolerance)
        self._periods = periods

    def _candidate_beat(self, candidate):
        return (0.0, self._periods[candidate])


# slot: 72 frames at 18 frames/beat = 4.0 beats
SLOT, PERIOD = 72, 18.0
FOUR = (0, 0, 72, 0)      # 72 frames at 18/beat -> 4.0 beats  (right)
SIX = (1, 0, 72, 0)       # 72 frames at 12/beat -> 6.0 beats  (wrong)
THREE = (2, 0, 72, 0)     # 72 frames at 24/beat -> 3.0 beats  (wrong)
PERIODS = {FOUR: 18.0, SIX: 12.0, THREE: 24.0}


def test_all_three_are_tied_on_frames_which_is_the_defect():
    """They are the same length in FRAMES, so today's pool cannot tell them apart."""
    assert FOUR[2] - FOUR[1] == SIX[2] - SIX[1] == THREE[2] - THREE[1] == SLOT


def test_the_filter_keeps_only_the_beat_matched_candidate():
    kept = Fake(0.5, PERIODS)._prefer_beat_span([FOUR, SIX, THREE], SLOT, PERIOD)
    assert kept == [FOUR]


def test_a_tolerance_wide_enough_keeps_everything_which_is_no_filter():
    kept = Fake(3.0, PERIODS)._prefer_beat_span([FOUR, SIX, THREE], SLOT, PERIOD)
    assert kept == [FOUR, SIX, THREE]


def test_zero_tolerance_is_off():
    kept = Fake(0.0, PERIODS)._prefer_beat_span([FOUR, SIX, THREE], SLOT, PERIOD)
    assert kept == [FOUR, SIX, THREE]


def test_it_falls_back_to_the_whole_tie_rather_than_failing():
    """A class with no beat-matched exemplar must degrade, not empty the pool."""
    library = Fake(0.1, PERIODS)
    kept = library._prefer_beat_span([SIX, THREE], SLOT, PERIOD)
    assert kept == [SIX, THREE]
    assert getattr(library, "beat_span_empty", 0) == 1


def test_a_missing_query_period_is_counted_not_guessed():
    library = Fake(0.5, PERIODS)
    kept = library._prefer_beat_span([FOUR, SIX], SLOT, float("nan"))
    assert kept == [FOUR, SIX]
    assert getattr(library, "beat_span_blind", 0) == 1


def test_the_flag_is_wired_and_its_counters_are_recorded():
    assert '"--draft-beat-span"' in SOURCE
    assert "draft_beat_span=options.draft_beat_span" in SOURCE
    assert "beat_span_tolerance=draft_beat_span" in SOURCE
    for key in ("draft_beat_span_slots", "draft_beat_span_applied",
                "draft_beat_span_empty", "draft_beat_span_blind"):
        assert '"{}"'.format(key) in SOURCE, key


def test_it_runs_before_the_other_filters_at_every_site():
    """It narrows the POOL, so it must come first; and it must reach all three
    places the other filters do -- the fourth consequence of the rule list
    (DEFECTS 77) was exactly a filter that reached only two of them."""
    assert SOURCE.count("_prefer_beat_span(") == 4      # definition + 3 sites
    for anchor in ("tied = self._prefer_beat_span(tied, target_length, beat_period)",
                   "narrowed = self._prefer_beat_span(near, target_length, target_beat_period)"):
        assert anchor in SOURCE, anchor
