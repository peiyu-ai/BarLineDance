"""``labels_to_segments(min_fragment=...)`` -- dropping a bar line that would
leave a sliver.

WHY THIS EXISTS.  ``--draft-bar-prototypes`` adds every bar line as a split so
each bar gets its own prototype.  A bar line and a label change rarely coincide
to the frame -- the plan's boundaries come from the vote and
``--plan-min-segment``, the bar lines from the beat channel -- so a line landing
just past a label change carves off a fragment.  Measured on BAR ep160
(2026-09-05, 18 eval clips) the fold removes 2 of 174 retrieval units.  It does
NOT remove that arm's worst stretch (1.400, a 7-frame slot filled with a 5-frame
prototype): re-measuring found all 6 of its sub-beat units start at frame 0, so
they are the partial bar at the clip head, bounded by a genuine label change
from the bar-token expansion rather than by a line the fold could decline.  That
is the caller's problem, and the fold deliberately leaves it alone.  The
threshold is one beat because the T vocabulary's shortest movement is a whole
4-beat bar (1.23 s), so nothing in the corpus is a quarter of that.

WHAT MUST NOT REGRESS.  The fold may only decline to add a split of its own.  If
it could merge two different labels, or move a label change, it would silently
edit the plan -- and a short run the PLAN contains is the planner's output,
governed by ``--plan-min-segment`` upstream.  The label-preservation test below
is the one that would catch that, so it is written as an exact equality on the
per-frame label track rather than on the segment list.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.atomic import labels_to_segments


def _track(segments, length):
    """Repaint the segments back into a per-frame label track."""
    track = torch.zeros(length, dtype=torch.long)
    for segment in segments:
        track[segment.start:segment.end] = segment.label
    return track


def test_a_bar_line_next_to_a_label_change_is_dropped():
    """The measured case: the line would leave a 3-frame sliver."""
    labels = torch.tensor([1] * 30 + [2] * 30)
    # 33 sits 3 frames past the change at 30; 60-frame bars would put a line there.
    kept = labels_to_segments(labels, split_at=[33], min_fragment=8)
    assert [(s.label, s.start, s.end) for s in kept] == [(1, 0, 30), (2, 30, 60)]


def test_the_same_line_IS_kept_when_it_leaves_real_bars():
    """Positive control: without it, the fold could be a constant "drop"."""
    labels = torch.tensor([1] * 120)
    kept = labels_to_segments(labels, split_at=[60], min_fragment=8)
    assert [(s.label, s.start, s.end) for s in kept] == [(1, 0, 60), (1, 60, 120)]


def test_off_by_default_so_every_earlier_artifact_reproduces():
    """min_fragment=None must behave exactly as before it existed."""
    labels = torch.tensor([1] * 30 + [2] * 30)
    assert (labels_to_segments(labels, split_at=[33])
            == labels_to_segments(labels, split_at=[33], min_fragment=0)
            == labels_to_segments(labels, split_at=[33], min_fragment=None))
    assert len(labels_to_segments(labels, split_at=[33])) == 3


def test_labels_are_never_changed_by_the_fold():
    """The rigid one: the per-frame label track is identical with and without.

    A fold that merged two different labels, or moved a boundary, would pass
    every count-based assertion above and fail here.
    """
    torch.manual_seed(20260905)
    for _ in range(200):
        labels = torch.randint(0, 4, (97,))
        lines = list(range(7, 97, 11))
        plain = _track(labels_to_segments(labels, split_at=lines), 97)
        folded = _track(labels_to_segments(labels, split_at=lines, min_fragment=9), 97)
        assert torch.equal(plain, folded)
        assert torch.equal(plain, labels)


def test_segments_still_tile_the_track_with_no_hole_or_overlap():
    labels = torch.tensor([1] * 25 + [2] * 40 + [1] * 35)
    segments = labels_to_segments(labels, split_at=[26, 30, 64, 70], min_fragment=9)
    assert segments[0].start == 0 and segments[-1].end == 100
    for before, after in zip(segments, segments[1:]):
        assert before.end == after.start


def test_a_cluster_of_close_lines_cannot_admit_each_other():
    """Greedy acceptance, not "far from the ORIGINAL bounds".

    Lines at 40, 45 and 50 are each >=8 from the label bounds at 0 and 90, so a
    naive test against the original bounds alone would accept all three and
    rebuild exactly the 5-frame slivers this exists to prevent.
    """
    labels = torch.tensor([1] * 90)
    segments = labels_to_segments(labels, split_at=[40, 45, 50], min_fragment=8)
    assert [(s.start, s.end) for s in segments] == [(0, 40), (40, 50), (50, 90)]
    assert min(s.end - s.start for s in segments) >= 8
