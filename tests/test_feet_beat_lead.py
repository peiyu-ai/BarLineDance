"""``--draft-feet-beat-lead``: pick, from the tie, a prototype whose FEET lead
the beat -- and only that, because only that part of the target is stable.

Two constraints are pinned, each for a measured reason:

* the contrast is feet minus the WHOLE BODY's mean, not feet minus shoulders.
  ``--draft-beat-fit`` used feet-minus-shoulders, which is won by braking hard,
  and it took jitter from 0.0531 to 0.0996 (ground truth 0.0509).  Against the
  body mean, a candidate that brakes everything equally scores exactly zero.
* it reads every part the judge reads.  The per-part ORDER below the feet is not
  stable -- recomputed with the judge's own aligned_profile, train and test agree
  at Spearman +0.94 that the feet lead by 0.09-0.10 against <=0.05 for the rest,
  while elbows move 4th -> 2nd and torso 3rd -> last with the measurement -- so
  the order must NOT be encoded, only the feet's lead over the average.
"""
import numpy as np
import pytest

from infer_atomic import IndexedAtomicMotionLibrary as Library
from tools.score_beat_phase_profile import PARTS


class Stub:
    BEAT_PARTS = Library.BEAT_PARTS

    def __init__(self, margins, feet_beat_lead=True):
        self.margins = margins
        self.feet_beat_lead = feet_beat_lead
        # The real filter counts at the point of use (``self.feet_lead_slots
        # += 1``), so a stub has to carry the counters or every call raises
        # AttributeError -- which is how this file went red without anyone
        # noticing, the same way tests/test_retrieval_tie_break.py did.
        self.feet_lead_slots = 0
        self.feet_lead_skipped = 0
        self.feet_lead_applied = 0
        self.feet_lead_changed = 0

    def _feet_lead_margin(self, candidate, target_length, beats):
        return self.margins[candidate]

    _prefer_feet_beat_lead = Library._prefer_feet_beat_lead


def test_it_reads_the_parts_the_judge_reads():
    assert set(Library.BEAT_PARTS) == set(PARTS)
    for name, joints in Library.BEAT_PARTS.items():
        assert tuple(joints) == tuple(PARTS[name]), name


def test_it_keeps_the_better_half():
    """A HALF, and the test says so because a report once said 'quarter': the
    edit that would have narrowed it failed its own assertion, the script
    carried on, and the arms were regenerated under the unchanged setting while
    being described as stronger.  The test now states what the code does."""
    tied = list("abcdefgh")
    stub = Stub({c: float(i) for i, c in enumerate(tied)})
    assert stub._prefer_feet_beat_lead(tied, 50, (0, 12)) == ["h", "g", "f", "e"]


def test_a_tie_of_two_still_chooses():
    """Requiring four tied candidates made the filter fire on 9 clips of 20 and
    the judge could not tell whether it did anything."""
    tied = ["a", "b"]
    stub = Stub({"a": 0.0, "b": 1.0})
    assert stub._prefer_feet_beat_lead(tied, 50, (0, 12)) == ["b"]


def test_a_single_candidate_and_a_beatless_slot_are_left_alone():
    stub = Stub({"a": 1.0, "b": 0.0})
    assert stub._prefer_feet_beat_lead(["a"], 50, (0, 12)) == ["a"]
    assert stub._prefer_feet_beat_lead(["a", "b"], 50, ()) == ["a", "b"]


def test_off_is_off():
    tied = list("abcd")
    stub = Stub({c: float(i) for i, c in enumerate(tied)}, feet_beat_lead=False)
    assert stub._prefer_feet_beat_lead(tied, 50, (0, 12)) == tied


def test_equal_margins_keep_the_incoming_order():
    tied = list("abcd")
    stub = Stub({c: 1.0 for c in tied})
    assert stub._prefer_feet_beat_lead(tied, 50, (0, 12)) == ["a", "b"]


def test_the_contrast_is_against_the_body_mean_not_the_shoulders():
    """THE ONE THAT MATTERS.  Feet-minus-shoulders is won by braking hard;
    feet-minus-mean gives a body that brakes everywhere exactly zero."""
    import inspect
    source = inspect.getsource(Library._feet_lead_margin)
    assert 'scores["feet"] - float(np.mean(list(scores.values())))' in source
    assert "shoulders" not in source.split("margin =")[1]


def test_the_unstable_part_order_is_not_encoded():
    """Only the feet's lead is used; if the routine ever starts comparing the
    order below the feet, it is encoding something that moves with the
    measurement (elbows 4th -> 2nd, torso 3rd -> last)."""
    import inspect
    source = inspect.getsource(Library._feet_lead_margin)
    body = source.split('"""', 2)[-1]        # the code, not the docstring
    for banned in ("spearman", "rank", "argsort"):
        assert banned not in body.lower()
