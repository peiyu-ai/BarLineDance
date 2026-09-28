"""The bar-anchored arms, and the control that makes them readable.

Round one of the cut comparison ranked nothing on purpose: a settle-snapped arm
wins a settle criterion by construction.  Round two adds arms that all start
from the same 4-beat grid, so the only free variable is what happens at a bar
line -- and a control, ``shift_like``, that moves boundaries by the same
distances to nowhere in particular.  These tests are about that control being a
real one: if it accidentally landed on settles too, or if it silently returned
the input, the whole comparison would read as a win either way.
"""

import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.render_cut_comparison import (  # noqa: E402
    METHOD_SETS, gate_to, settle_phase, shift_like, snap_with_offsets)
from tools.segment_on_music_beats import grid_bounds  # noqa: E402

PERIOD = 16


class SnapWithOffsetsTests(unittest.TestCase):
    def test_a_boundary_with_a_settle_in_reach_moves_and_reports_how_far(self):
        cuts = [0, 64, 128, 200]
        snapped, offsets = snap_with_offsets(cuts, np.array([60, 131]), 6, 200, 8)
        self.assertEqual(snapped, [0, 60, 131, 200])
        self.assertEqual(offsets, [-4, 3])

    def test_a_boundary_with_nothing_in_reach_stays_on_the_beat(self):
        """The fallback is the whole reason this is still a bar grid.  Over half
        of real bar lines have no settle within half a beat, so a rule without
        this branch would leave the grid rather than preserve it."""
        snapped, offsets = snap_with_offsets([0, 64, 128, 200], np.array([20]),
                                             6, 200, 8)
        self.assertEqual(snapped, [0, 64, 128, 200])
        self.assertEqual(offsets, [0, 0])

    def test_offsets_never_exceed_max_snap(self):
        targets = np.array([10, 40, 70, 100, 130, 160])
        _, offsets = snap_with_offsets([0, 64, 128, 200], targets, 6, 200, 8)
        self.assertTrue(all(abs(o) <= 6 for o in offsets))

    def test_the_first_and_last_boundary_are_never_moved(self):
        snapped, _ = snap_with_offsets([0, 64, 200], np.array([3, 197]), 6, 200, 8)
        self.assertEqual(snapped[0], 0)
        self.assertEqual(snapped[-1], 200)


class ShiftLikeControlTests(unittest.TestCase):
    def test_the_control_keeps_the_displacement_magnitudes(self):
        cuts = [0, 64, 128, 192, 256]
        shifted = shift_like(cuts, [-5, 3, 6], 256, 8, seed=1)
        moved = sorted(abs(a - b) for a, b in zip(shifted[1:-1], cuts[1:-1]))
        self.assertEqual(moved, [3, 5, 6])

    def test_the_control_is_not_the_snapped_arm(self):
        """If these ever agreed, the comparison could not tell 'landed on a
        settle' from 'moved off the beat at all'."""
        cuts = [0, 64, 128, 192, 256]
        settles = np.array([59, 131, 187])
        snapped, offsets = snap_with_offsets(cuts, settles, 6, 256, 8)
        shifted = shift_like(cuts, offsets, 256, 8, seed=20260902)
        self.assertNotEqual(snapped, shifted)

    def test_zero_displacements_leave_the_grid_alone(self):
        cuts = [0, 64, 128, 256]
        self.assertEqual(shift_like(cuts, [0, 0], 256, 8, seed=3), cuts)

    def test_no_interior_boundary_is_returned_unchanged(self):
        self.assertEqual(shift_like([0, 100], [4], 100, 8, seed=3), [0, 100])

    def test_the_control_stays_inside_the_clip(self):
        shifted = shift_like([0, 8, 250, 256], [-40, 40], 256, 4, seed=5)
        self.assertTrue(all(0 <= c <= 256 for c in shifted))


class GateTests(unittest.TestCase):
    def test_bar_lines_without_a_settle_are_dropped_not_kept(self):
        """This is the arm's whole claim: rather than cut mid-movement, do not
        cut, and let the segment run on to the next bar."""
        gated = gate_to([0, 64, 128, 192, 256], np.array([130]), 6, 256, 8)
        self.assertEqual(gated, [0, 130, 256])

    def test_every_interior_boundary_it_returns_is_a_settle(self):
        settles = np.array([61, 130, 190])
        gated = gate_to([0, 64, 128, 192, 256], settles, 6, 256, 8)
        self.assertTrue(set(gated[1:-1]).issubset(set(int(s) for s in settles)))

    def test_no_settles_at_all_leaves_one_segment(self):
        self.assertEqual(gate_to([0, 64, 128, 256], np.zeros(0), 6, 256, 8),
                         [0, 256])


class SettlePhaseTests(unittest.TestCase):
    def setUp(self):
        self.frames = 16 * PERIOD
        self.beats = np.arange(0, self.frames, PERIOD)

    def test_positive_control_recovers_the_phase_the_settles_were_built_on(self):
        for planted in range(4):
            bounds = grid_bounds(self.beats, self.frames, 4, planted, 8, edges="keep")
            settles = np.asarray(bounds[1:-1], dtype=int)
            chosen, shares = settle_phase(self.beats, settles, self.frames, 4, 8, 2)
            self.assertEqual(chosen, planted)
            self.assertGreater(max(shares) - min(shares), 0.5)

    def test_negative_control_settles_between_the_bar_lines_do_not_pick_a_phase(self):
        """Settles placed a whole beat off every bar line: no phase can claim
        them, so the spread must stay at zero.  Without this the tool would
        report a winning phase on any footage at all."""
        settles = np.arange(PERIOD // 2, self.frames, PERIOD)
        _, shares = settle_phase(self.beats, settles, self.frames, 4, 8, 2)
        self.assertEqual(max(shares) - min(shares), 0.0)

    def test_a_phase_is_returned_for_every_candidate(self):
        _, shares = settle_phase(self.beats, np.array([30]), self.frames, 4, 8, 2)
        self.assertEqual(len(shares), 4)


class MethodSetTests(unittest.TestCase):
    def test_the_bar_set_carries_its_control(self):
        self.assertIn("beat4_shift", METHOD_SETS["bar"])

    def test_the_survey_set_carries_its_control(self):
        self.assertIn("random", METHOD_SETS["survey"])

    def test_both_sets_fit_the_six_panel_video(self):
        for name, methods in METHOD_SETS.items():
            self.assertLessEqual(len(methods), 6, name)


if __name__ == "__main__":
    unittest.main()
