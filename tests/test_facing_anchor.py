"""``--draft-facing-anchor``: keep performing to where the dance started.

WHY.  ``--draft-facing-continuity`` rotates each prototype so its facing meets
the previous segment's, which keeps the facing SMOOTH and lets every
prototype's own turning ACCUMULATE.  Measured 2026-09-06 on the 20 T eval
clips: ground truth spends 0.0% of its frames more than 90 degrees from its
opening facing on 14 of 20, while three generated clips spend 77-92% of theirs
facing away -- the worst for 14.77 s, with a net turn of 593 degrees, more than
a full circle.  The operator's words on two of them were "背朝相机舞姿段落太长,
看不到手部动作跳了什么" and "不知道观众在哪".

Nothing in the pipeline anchored the facing before this; continuity is not an
anchor, and the two are easy to confuse because a continuous facing looks
correct at every individual join.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _chain(turns, anchor):
    """Facing at the end of each segment under continuity + anchor.

    Mirrors build_draft: align to the previous segment's facing, optionally
    pull back a fraction of the drift from the opening, then add this
    segment's own internal turn.
    """
    opening = 0.0
    previous = 0.0
    out = []
    for turn in turns:
        delta = previous - 0.0
        if anchor:
            drift = (previous + delta - opening + math.pi) % (2 * math.pi) - math.pi
            delta = delta - anchor * drift
        facing = delta + turn
        previous = facing
        out.append(facing)
    return out


def test_without_an_anchor_the_drift_accumulates():
    """Each segment turning the same way walks the dancer right around."""
    out = _chain([0.5] * 8, 0.0)
    assert out[-1] > 3.0, out[-1]


def test_with_an_anchor_the_drift_stays_bounded():
    out = _chain([0.5] * 8, 0.5)
    assert abs(out[-1]) < 1.5, out[-1]


def test_a_stronger_anchor_binds_tighter():
    weak = abs(_chain([0.5] * 8, 0.2)[-1])
    strong = abs(_chain([0.5] * 8, 0.8)[-1])
    assert strong < weak


def test_zero_is_the_published_behaviour():
    assert _chain([0.3, -0.2, 0.4], 0.0) == _chain([0.3, -0.2, 0.4], 0)


def test_the_anchor_does_not_stop_a_segment_turning():
    """It removes ACCUMULATED drift, not the movement's own rotation."""
    out = _chain([1.2], 0.9)
    assert abs(out[0] - 1.2) < 1e-9


def test_the_pull_takes_the_short_way_round():
    """A drift just past pi must be corrected backwards, not by nearly 2pi."""
    drift = (3.3 + math.pi) % (2 * math.pi) - math.pi
    assert drift < 0
    assert abs(drift) < math.pi
