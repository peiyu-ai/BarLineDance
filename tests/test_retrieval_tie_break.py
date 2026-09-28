"""The duration rule's ties, and what breaking them differently must and must not change.

MEASURED (20 held-out clips, 91 retrieved segments, class pool median 690):
73.6% of segments have more than one candidate at the minimum |duration
difference|, the median tie holds 6 and the largest 38, and 79.1% have an exact
length match.  ``min`` returns the first at the minimum, so all 79 distinct
``(label, target_length)`` keys returned the same prototype every time -- in
every clip.  That is the "every clip dances the same" the operator reported.

The load-bearing tests here are the two that constrain the FIX rather than the
defect: the default must reproduce the old behaviour exactly, and the new mode
must only ever choose among candidates that are EXACTLY as good on duration --
otherwise it is the paper's Random Choice ablation, which loses on FID because
unconstrained draws need heavier time-stretching.
"""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from infer_atomic import IndexedAtomicMotionLibrary as Library


class FakeLibrary:
    """Only ``_duration_pick`` is under test, so nothing else is built.

    ``_duration_pick`` now runs the tied candidates through five selection
    filters before the tie-break.  They are bound here as the REAL methods with
    their switches OFF -- not stubbed -- so "the default reproduces ``min``
    exactly" is asserted through the filter chain the shipped code actually
    runs.  Until 2026-09-16 this double had only ``tie_break``, and every test
    below raised AttributeError from the first filter call; the file had been
    red since the filters were added and nobody had run it.
    """

    def __init__(self, tie_break):
        self.tie_break = tie_break
        self.beat_span_tolerance = 0.0
        self.hold_by_music = False
        self.feet_beat_lead = False
        self.beat_fit = False
        self.feet_lead = False
        self.hop_guard = False
        self.music_energy = False
        # counted at the point of use by _prefer_feet_beat_lead
        self.feet_lead_slots = 0
        self.feet_lead_skipped = 0

    pick = Library._duration_pick
    _prefer_beat_span = Library._prefer_beat_span
    _prefer_hop_guard = Library._prefer_hop_guard
    _prefer_music_energy = Library._prefer_music_energy
    _prefer_hold_by_music = Library._prefer_hold_by_music
    _prefer_feet_beat_lead = Library._prefer_feet_beat_lead
    _prefer_beat_fit = Library._prefer_beat_fit
    _prefer_feet_lead = Library._prefer_feet_lead


def candidates(lengths):
    # (sample, start, end, group) -- end - start is the candidate's length.
    return [(i, 0, length, "src{}".format(i)) for i, length in enumerate(lengths)]


def test_index_mode_reproduces_min_exactly():
    """The default must be byte-identical to the shipped rule."""
    pool = candidates([10, 20, 20, 20, 31])
    chosen = FakeLibrary("index").pick(pool, 20, frozenset({"q"}))
    assert chosen is min(pool, key=lambda c: abs((c[2] - c[1]) - 20))
    assert chosen[0] == 1, "must be the FIRST of the tied candidates"


def test_salted_mode_only_ever_picks_a_tied_candidate():
    """The whole safety argument: no duration quality is traded away.

    Every group id must land on a candidate whose length distance equals the
    minimum -- so the resampling factor is exactly what the index rule would
    have produced.
    """
    pool = candidates([10, 20, 20, 20, 31])
    library = FakeLibrary("salted")
    for i in range(200):
        chosen = library.pick(pool, 20, frozenset({"group{}".format(i)}))
        assert abs((chosen[2] - chosen[1]) - 20) == 0


def test_salted_mode_gives_different_recordings_different_exemplars():
    """The axis that was broken: across clips."""
    pool = candidates([20] * 12)
    library = FakeLibrary("salted")
    picked = {library.pick(pool, 20, frozenset({"wild_v5:{}".format(i)}))[0]
              for i in range(60)}
    assert len(picked) > 1, "every recording still got the same prototype"


def test_salted_mode_is_stable_for_one_recording():
    """Within-clip behaviour measures fine already and must not be disturbed.

    Repeated calls for the same (group, label, length) must return the same
    prototype, so this switch cannot be blamed for a within-clip change.
    """
    pool = candidates([20] * 12)
    library = FakeLibrary("salted")
    got = {library.pick(pool, 20, frozenset({"wild_v5:abc"}))[0] for _ in range(50)}
    assert len(got) == 1


def test_salted_mode_is_order_independent():
    # The pick must not depend on how many segments were retrieved before it --
    # the defect that made variety_rng per-clip in the first place.
    pool = candidates([20] * 9)
    library = FakeLibrary("salted")
    first = library.pick(pool, 20, frozenset({"g"}))
    for _ in range(30):
        library.pick(pool, 20, frozenset({"other"}))
    assert library.pick(pool, 20, frozenset({"g"})) is first


def test_a_lone_best_candidate_is_taken_by_both_modes():
    pool = candidates([10, 20, 35])
    assert FakeLibrary("index").pick(pool, 20, frozenset({"g"}))[0] == 1
    assert FakeLibrary("salted").pick(pool, 20, frozenset({"g"}))[0] == 1


def test_ties_at_a_nonzero_distance_are_still_ties():
    # No exact match exists; the two at distance 3 are equally good.
    pool = candidates([17, 23, 40])
    library = FakeLibrary("salted")
    for i in range(50):
        chosen = library.pick(pool, 20, frozenset({"g{}".format(i)}))
        assert abs((chosen[2] - chosen[1]) - 20) == 3


def test_an_unknown_tie_break_is_refused_at_construction(tmp_path):
    with pytest.raises(ValueError) as failure:
        Library(tmp_path, tie_break="whatever")
    assert "tie-break" in str(failure.value)


def test_the_cache_key_separates_the_two_modes():
    """Two modes must never share a cached prototype.

    Without this, running both in one process would serve the second whatever
    the first had already cached, and the comparison would silently be of one
    mode against itself.
    """
    source = pathlib.Path("infer_atomic.py").read_text()
    start = source.index("        key = (int(label), int(target_length)")
    assert "self.tie_break" in source[start:start + 400]
