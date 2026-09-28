"""The settle criterion's direction is pinned here, not left to a run.

CLAUDE.md section 2.1 exists because ``boundary contrast`` could fail and still
pointed the wrong way.  So the thing this file asserts is not "the statistic
computes"; it is **which placement wins**: on a fixture whose settle points are
known by construction, cutting at those points must beat random placement, and
cutting at the velocity peaks must lose to it.  Break that ordering and the
criterion is measuring something else.
"""

import unittest

import numpy as np

from tools.probe_settle_alignment import (
    beat_spans, clip_beats, interior_cuts, mean_nearest, peak_spans_speed,
    smoothed_speed,
)
from tools.probe_segmentation_boundaries import shuffled_spans


def _swinging(frames=360, joints=22, period=40):
    """A body whose left hand swings, settling every ``period/2`` frames.

    The settle points are the turning points of the sine, which are exactly the
    frames where speed is minimal -- so the fixture knows its own answer.
    """
    pose = np.zeros((joints, 3))
    pose[:, 2] = np.linspace(0.0, 1.7, joints)
    pose[1] = [0.1, 0.1, 0.9]
    pose[2] = [0.1, -0.1, 0.9]
    pose[16] = [0.0, 0.2, 1.4]
    pose[17] = [0.0, -0.2, 1.4]
    body = np.repeat(pose[None], frames, axis=0)
    phase = np.sin(np.arange(frames) * 2 * np.pi / period)
    body[:, 20, 2] = 1.0 + 0.5 * phase
    body[:, 21, 2] = 1.0 - 0.5 * phase
    return body


def _settle_ratio(spans, beats, joints, seed=0):
    """The criterion itself: (random cut->beat + 0.5) / (arm cut->beat + 0.5)."""
    rng = np.random.default_rng(seed)
    arm, _ = mean_nearest(interior_cuts(spans), beats)
    control = []
    for _ in range(16):
        drawn = shuffled_spans(spans, len(joints), rng)
        if len(drawn) >= 2:
            distance, _ = mean_nearest(interior_cuts(drawn), beats)
            if distance == distance:
                control.append(distance)
    return (float(np.mean(control)) + 0.5) / (arm + 0.5)


class DirectionTests(unittest.TestCase):
    def setUp(self):
        self.joints = _swinging()
        self.speed = smoothed_speed(self.joints)
        self.beats = clip_beats(self.joints, "clip")
        edges = list(range(0, len(self.joints) + 1, 24))
        self.base = list(zip(edges[:-1], edges[1:]))

    def test_the_fixture_settles_where_the_sine_turns(self):
        # A 40-frame period turns twice per period; nothing else settles.
        self.assertGreaterEqual(len(self.beats), 10)
        gaps = np.diff(np.sort(self.beats))
        self.assertLess(abs(float(np.median(gaps)) - 20.0), 3.0)

    def test_cutting_at_beats_scores_far_above_random(self):
        spans = beat_spans(self.beats, self.speed, self.base, 18)
        self.assertGreater(_settle_ratio(spans, self.beats, self.joints), 3.0)

    def test_cutting_at_peaks_scores_below_random(self):
        spans = peak_spans_speed(self.speed, self.base, 18)
        self.assertLess(_settle_ratio(spans, self.beats, self.joints), 1.0)

    def test_random_placement_sits_at_one(self):
        rng = np.random.default_rng(7)
        spans = shuffled_spans(self.base, len(self.joints), rng)
        ratio = _settle_ratio(spans, self.beats, self.joints, seed=3)
        self.assertGreater(ratio, 0.7)
        self.assertLess(ratio, 1.4)

    def test_the_two_controls_are_placed_by_the_same_rule(self):
        # Same cut count and the same min-length, so a gap between them is the
        # placement and not the fineness.
        beats = beat_spans(self.beats, self.speed, self.base, 18)
        peaks = peak_spans_speed(self.speed, self.base, 18)
        self.assertEqual(len(peaks), len(self.base))
        self.assertLessEqual(len(beats), len(self.base))
        for spans in (beats, peaks):
            self.assertTrue(all(b - a >= 18 for a, b in spans))


if __name__ == "__main__":
    unittest.main()
