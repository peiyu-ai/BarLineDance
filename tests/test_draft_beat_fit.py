"""``--draft-beat-fit``: pick, from the tie, a prototype whose feet brake on
this slot's beats and whose shoulders do not.

The tests that earn their keep are ``test_it_is_a_filter_not_a_reranking``
(``--retrieval-rule phase`` landed its criterion perfectly and made the dance
worse by squeezing out variety) and ``test_the_cache_key_carries_the_slot_beats``
(two slots of the same class and length whose beats sit differently must not
share a cached answer, or the second slot silently reuses the first's pick).
"""
import types

import numpy as np
import pytest
import torch

import infer_atomic


def test_slot_beats_are_offsets_inside_the_slot():
    """Offsets, not absolute frames: the candidate is judged after being
    resampled into the slot, so its frame 0 is the slot's frame 0."""
    music = torch.zeros(200, 35)
    music[[10, 60, 110, 160], infer_atomic.BEAT_CHANNEL] = 1.0
    assert infer_atomic._slot_beat_frames(music, 50, 150) == (10, 60)


def test_beats_outside_the_slot_are_dropped():
    music = torch.zeros(100, 35)
    music[[5, 95], infer_atomic.BEAT_CHANNEL] = 1.0
    assert infer_atomic._slot_beat_frames(music, 40, 60) == ()


def test_no_music_is_no_beats_rather_than_an_error():
    assert infer_atomic._slot_beat_frames(None, 0, 50) == ()


class Stub:
    """Only the selection logic, with the margin supplied rather than skinned."""

    BEAT_FIT_LEAD = infer_atomic.IndexedAtomicMotionLibrary.BEAT_FIT_LEAD
    BEAT_FIT_TRAIL = infer_atomic.IndexedAtomicMotionLibrary.BEAT_FIT_TRAIL

    def __init__(self, margins, beat_fit=True):
        self.margins = margins
        self.beat_fit = beat_fit

    def _beat_fit_margin(self, candidate, target_length, beats):
        return self.margins[candidate]

    _prefer_beat_fit = infer_atomic.IndexedAtomicMotionLibrary._prefer_beat_fit


def test_it_keeps_the_best_quarter():
    tied = list("abcdefgh")
    stub = Stub({c: float(i) for i, c in enumerate(tied)})
    assert stub._prefer_beat_fit(tied, 50, (0, 12)) == ["h", "g"]


def test_it_is_a_filter_not_a_reranking():
    """The survivors are handed back to the existing tie-break, so more than
    one candidate has to survive whenever the tie is large.  An argmax here
    would be the ``--retrieval-rule phase`` failure again: criterion landed,
    variety gone."""
    tied = list("abcdefghijkl")
    stub = Stub({c: float(i) for i, c in enumerate(tied)})
    kept = stub._prefer_beat_fit(tied, 50, (0, 12))
    assert 1 < len(kept) < len(tied)


def test_a_small_tie_is_left_alone():
    """Under four candidates a quartile is one candidate, which is an argmax;
    the tie is passed through instead."""
    tied = list("abc")
    stub = Stub({c: float(i) for i, c in enumerate(tied)})
    assert stub._prefer_beat_fit(tied, 50, (0, 12)) == tied


def test_a_slot_with_no_beats_is_left_alone():
    tied = list("abcdef")
    stub = Stub({c: float(i) for i, c in enumerate(tied)})
    assert stub._prefer_beat_fit(tied, 50, ()) == tied


def test_off_is_off():
    tied = list("abcdef")
    stub = Stub({c: float(i) for i, c in enumerate(tied)}, beat_fit=False)
    assert stub._prefer_beat_fit(tied, 50, (0, 12)) == tied


def test_ties_in_the_margin_break_on_position_not_dict_order():
    tied = list("abcdefgh")
    stub = Stub({c: 1.0 for c in tied})
    first = stub._prefer_beat_fit(tied, 50, (0, 12))
    assert first == stub._prefer_beat_fit(tied, 50, (0, 12))
    assert first == ["a", "b"], "equal margins must keep the incoming order"


def test_the_cache_key_carries_the_slot_beats():
    """Two slots of the same class and length whose beats fall differently must
    not share a cached retrieval, or the second silently replays the first."""
    import inspect
    source = inspect.getsource(infer_atomic.IndexedAtomicMotionLibrary.retrieve)
    head = source.split("if self.retrieval_rule in", 1)[0]
    assert "target_beats" in head, (
        "the slot's beats are part of what the answer depends on and must be "
        "in the cache key")


def test_the_lead_and_trail_groups_are_the_judged_ones():
    from tools.score_beat_phase_profile import PARTS
    library = infer_atomic.IndexedAtomicMotionLibrary
    assert tuple(library.BEAT_FIT_LEAD) == tuple(PARTS["feet"])
    assert tuple(library.BEAT_FIT_TRAIL) == tuple(PARTS["shoulders"])
