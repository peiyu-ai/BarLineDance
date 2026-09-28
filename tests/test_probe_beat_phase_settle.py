"""The phase ruler must separate a settled cut from a peak cut, and read ~1 on its null.

Why these tests and not others.  ``probe_beat_phase_settle`` exists to answer
one question -- does cutting on the music beat also cut where the dancer settles
-- and its claim to being readable rests on the control being the arm's OWN cuts
rotated, so period and cut count are held fixed and only phase moves.  Three
things can go wrong and each is tested:

* the ruler is blind (a cut on a settle point reads like a cut on a speed peak),
  in which case a null reading on a real arm means nothing;
* the rotation does not preserve the arm's gaps, in which case the control is
  not the same segmentation at another phase;
* a readout is biased, so the null itself does not read 1.0 -- CLAUDE.md 2.1
  gate 3 in the form this file needs it.

THE FIXTURE'S GEOMETRY IS NOT ARBITRARY, and an earlier version of it made this
file fail for a reason the corpus does not have.  With settle points exactly
every P frames and cuts exactly every kP, a rotation by delta gives EVERY cut in
the clip the same offset ``delta mod P``, so the clip's hit rate is 1 or 0 and
nothing in between; the median of a ratio of such things is not 1 and the null
test fails on the fixture's commensurability rather than on the code.  On
clean5 the two periods are close but not commensurate -- motion beats one per
13.7 frames (median, ``runs/d_on_the_record/settle_D.json``) against a music
grid of 15.0 (``music_grid_period_frames_median``, ``beat_phase_D.json``) -- so
the offsets within one clip are spread.  The fixture reproduces that.
"""

import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools import probe_beat_phase_settle as probe

BEAT_PERIOD = 13.7      # motion beats, clean5 median
CUT_PERIOD = 60         # four music beats at the corpus's 15.0-frame grid
FRAMES = 1200


def _fixture():
    t = np.arange(FRAMES)
    speed = 1.0 - 0.9 * np.cos(2 * np.pi * t / BEAT_PERIOD)
    minima = np.flatnonzero((speed[1:-1] < speed[:-2]) & (speed[1:-1] <= speed[2:])) + 1
    grid = np.arange(CUT_PERIOD, FRAMES - 1, CUT_PERIOD, dtype=np.int64)
    return speed, minima.astype(np.int64), grid


class TheRulerSeparates(unittest.TestCase):
    def setUp(self):
        self.speed, self.beats, self.grid = _fixture()
        self.median = float(np.median(self.speed))
        maxima = self.beats + int(round(BEAT_PERIOD / 2))
        self.settled = np.array(
            [self.beats[np.abs(self.beats - c).argmin()] for c in self.grid], dtype=np.int64)
        self.peaked = np.array(
            [maxima[np.abs(maxima - c).argmin()] for c in self.grid], dtype=np.int64)

    def _score(self, cuts, seed=5, draws=64):
        return probe.score_clip(np.asarray(cuts, dtype=np.int64), self.beats,
                                self.speed, self.median, FRAMES,
                                np.random.default_rng(seed), draws=draws)

    def test_cuts_on_the_settle_points_read_high_on_all_three_readouts(self):
        row = self._score(self.settled)
        self.assertGreater(row["distance_ratio"], 3.0)
        self.assertGreater(row["hit2_ratio"], 1.5)
        self.assertGreater(row["speed_ratio"], 1.5)

    def test_cuts_on_the_speed_peaks_read_low_on_all_three_readouts(self):
        row = self._score(self.peaked)
        self.assertLess(row["distance_ratio"], 1.0)
        self.assertLess(row["hit2_ratio"], 1.0)
        self.assertLess(row["speed_ratio"], 1.0)

    def test_the_null_reads_about_one(self):
        """A rotated copy of the arm scored against fresh rotations.

        If this drifts from 1.0 the readouts are biased and every arm's reading
        is being compared against the wrong number.  The real corpus check is
        the ``_null`` row the tool builds; this is the same statement on a
        fixture, so a regression fails in CI instead of in a report.
        """
        ratios = {"distance_ratio": [], "hit2_ratio": [], "speed_ratio": []}
        for seed in range(60):
            rng = np.random.default_rng(100 + seed)
            moved = probe.shift_cuts(self.grid, FRAMES,
                                     int(rng.integers(1, FRAMES - 1)))
            row = probe.score_clip(moved, self.beats, self.speed, self.median,
                                   FRAMES, rng, draws=16)
            for key in ratios:
                ratios[key].append(row[key])
        for key, values in ratios.items():
            self.assertAlmostEqual(float(np.median(values)), 1.0, delta=0.35,
                                   msg="{} null drifted to {:.3f}".format(
                                       key, float(np.median(values))))


class TheRotationKeepsTheSegmentation(unittest.TestCase):
    """The control must be the same cut pattern at another phase, nothing else."""

    def test_every_gap_survives_except_the_one_that_wraps(self):
        cuts = np.array([10, 40, 70, 100, 130], dtype=np.int64)
        frames = 200
        gaps = sorted(np.diff(cuts).tolist())
        for delta in (1, 7, 63, 150, 198):
            moved = probe.shift_cuts(cuts, frames, delta)
            self.assertEqual(len(moved), len(cuts))
            self.assertTrue((moved >= 1).all() and (moved <= frames - 1).all())
            kept = sorted(np.diff(moved).tolist())
            common = [g for g in kept if g in gaps]
            self.assertGreaterEqual(len(common), len(gaps) - 1)

    def test_a_zero_shift_reproduces_the_arm_exactly(self):
        cuts = np.array([10, 40, 70], dtype=np.int64)
        np.testing.assert_array_equal(probe.shift_cuts(cuts, 200, 0), cuts)


if __name__ == "__main__":
    unittest.main()
