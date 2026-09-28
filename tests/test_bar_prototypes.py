"""One prototype per bar, not per label run.

The defect: with --plan-bar-grid the plan carries one label per bar, so two
adjacent bars drawing the SAME label merge into one run and retrieval stretches
a single prototype across both. Measured on the T corpus, that pushed plan runs
to 2.40 s median / 6.19 s p90 against a 2.00 s vocabulary, while the ground
truth's own runs are 1.90 s / 3.86 s.
"""

import pathlib
import sys
import unittest

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dataset.atomic import labels_to_segments  # noqa: E402
from infer_atomic import bar_bounds_of  # noqa: E402


def labels(*runs):
    out = []
    for value, length in runs:
        out.extend([value] * length)
    return torch.tensor(out, dtype=torch.long)


class SplitTests(unittest.TestCase):
    def test_without_split_two_same_labelled_bars_are_one_segment(self):
        """The behaviour being fixed, pinned so a regression is visible."""
        segments = labels_to_segments(labels((3, 128)))
        self.assertEqual(len(segments), 1)
        self.assertEqual((segments[0].start, segments[0].end), (0, 128))

    def test_splitting_at_the_bar_line_yields_one_segment_per_bar(self):
        segments = labels_to_segments(labels((3, 128)), split_at=[64])
        self.assertEqual([(s.start, s.end) for s in segments], [(0, 64), (64, 128)])
        self.assertEqual([s.label for s in segments], [3, 3])

    def test_label_boundaries_are_still_honoured(self):
        segments = labels_to_segments(labels((3, 64), (5, 64)), split_at=[32, 96])
        self.assertEqual([(s.start, s.end) for s in segments],
                         [(0, 32), (32, 64), (64, 96), (96, 128)])
        self.assertEqual([s.label for s in segments], [3, 3, 5, 5])

    def test_a_split_point_on_an_existing_boundary_does_not_duplicate_it(self):
        segments = labels_to_segments(labels((3, 64), (5, 64)), split_at=[64])
        self.assertEqual([(s.start, s.end) for s in segments], [(0, 64), (64, 128)])

    def test_split_points_outside_the_clip_are_ignored(self):
        segments = labels_to_segments(labels((3, 64)), split_at=[0, 64, 999, -5])
        self.assertEqual([(s.start, s.end) for s in segments], [(0, 64)])

    def test_no_empty_segments_are_produced(self):
        segments = labels_to_segments(labels((3, 64)), split_at=[10, 10, 20])
        self.assertTrue(all(s.end > s.start for s in segments))

    def test_none_reproduces_the_published_behaviour(self):
        seq = labels((3, 30), (0, 10), (7, 40))
        self.assertEqual([(s.label, s.start, s.end) for s in labels_to_segments(seq)],
                         [(s.label, s.start, s.end)
                          for s in labels_to_segments(seq, split_at=None)])


class BarBoundsTests(unittest.TestCase):
    def _music(self, frames, period=16, offset=0):
        music = torch.zeros(frames, 35)
        music[offset::period, 34] = 1.0
        return music

    def test_bars_are_every_k_beats_from_the_phase(self):
        got = bar_bounds_of(self._music(256), beats_per_segment=4, phase=0)
        self.assertEqual(got, [0, 64, 128, 192])

    def test_phase_shifts_the_grid(self):
        got = bar_bounds_of(self._music(256), beats_per_segment=4, phase=1)
        self.assertEqual(got, [16, 80, 144, 208])

    def test_no_phase_means_no_split(self):
        self.assertIsNone(bar_bounds_of(self._music(256), 4, None))

    def test_music_without_beats_means_no_split(self):
        self.assertIsNone(bar_bounds_of(torch.zeros(256, 35), 4, 0))

    def test_the_stretch_this_removes(self):
        """The whole point, as arithmetic: a 2-bar run over a 2.0 s vocabulary
        stretches 2.17x; split per bar it is 1.09x."""
        music = self._music(256)
        bounds = bar_bounds_of(music, 4, 0)
        merged = labels_to_segments(labels((3, 128)))
        split = labels_to_segments(labels((3, 128)), split_at=bounds)
        self.assertEqual(len(merged), 1)
        self.assertEqual(len(split), 2)
        vocabulary_frames = 60.0                      # 2.0 s at 30 fps
        self.assertAlmostEqual((merged[0].end - merged[0].start) / vocabulary_frames,
                               2.133, places=2)
        self.assertAlmostEqual((split[0].end - split[0].start) / vocabulary_frames,
                               1.067, places=2)


if __name__ == "__main__":
    unittest.main()
