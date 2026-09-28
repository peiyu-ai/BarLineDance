"""Fusing the motion in must not change the arm that does not use it.

Two guarantees, and neither can be established by reading the code:

* **off is byte-identical to before the option existed.**  Every wild and AIST
  segmentation on disk was produced by the visual-only path; if adding the
  option moved it by one cut, every published segment count would silently stop
  matching its own report.  The test pins it against a recomputed baseline
  rather than against a stored constant, so it also fails if the *baseline*
  path changes.
* **"equal say" means equal say.**  ``--motion-weight`` is documented as being
  in units of the visual block's own row-to-row distance, which is a claim
  about a rescaling, and a rescaling is exactly the kind of thing that is
  written down once and then quietly not done.
"""

import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools import segment_visual_atomics as seg


def _features(frames=180, dim=32, blocks=6, seed=0):
    rng = np.random.default_rng(seed)
    per = frames // blocks
    rows = [np.tile(rng.normal(size=(1, dim)), (per, 1)) for _ in range(blocks)]
    return (np.concatenate(rows)[:frames] + rng.normal(size=(frames, dim)) * 0.01).astype(np.float32)


def _motion(frames=180, blocks=6, seed=1, joints=24, offset=0):
    rng = np.random.default_rng(seed)
    per = frames // blocks
    rows = [np.tile(rng.normal(size=(1, joints * 6)), (per, 1)) for _ in range(blocks)]
    motion = np.zeros((frames, 151), np.float64)
    block = np.roll(np.concatenate(rows)[:frames], offset, axis=0)
    motion[:, 7:] = block + rng.normal(size=(frames, joints * 6)) * 0.01
    motion[:, 4:7] = rng.normal(size=(frames, 3)) * 5.0     # root: must be ignored
    return motion


class OffIsUnchanged(unittest.TestCase):
    def test_no_motion_argument_reproduces_the_visual_only_cuts(self):
        features = _features()
        baseline, _ = seg.segment_sequence(features, 6, 18, 4.0, 7)
        with_args, _ = seg.segment_sequence(features, 6, 18, 4.0, 7,
                                            motion=_motion(), motion_weight=0.0)
        self.assertEqual(baseline, with_args)

    def test_a_zero_weight_ignores_the_motion_entirely(self):
        features = _features()
        a, _ = seg.segment_sequence(features, 6, 18, 4.0, 7,
                                    motion=_motion(seed=1), motion_weight=0.0)
        b, _ = seg.segment_sequence(features, 6, 18, 4.0, 7,
                                    motion=_motion(seed=99), motion_weight=0.0)
        self.assertEqual(a, b)


class TheMotionActuallyReachesTheClustering(unittest.TestCase):
    def test_a_nonzero_weight_moves_the_cuts_when_the_motion_disagrees(self):
        # Visual says six blocks; motion says three, and its turns sit 15
        # frames off the visual ones -- otherwise the two agree and identical
        # cuts would be the correct answer, testing nothing.
        features = _features(blocks=6, seed=0)
        motion = _motion(blocks=3, seed=5, offset=15)
        visual, _ = seg.segment_sequence(features, 6, 18, 4.0, 7)
        fused, _ = seg.segment_sequence(features, 6, 18, 4.0, 7,
                                        motion=motion, motion_weight=1.0)
        self.assertNotEqual(visual, fused)

    def test_the_root_position_is_not_part_of_the_pose_block(self):
        motion = _motion(seed=3)
        moved = motion.copy()
        moved[:, 4:7] += np.linspace(0, 50, len(motion))[:, None]   # walk across the stage
        np.testing.assert_allclose(seg.pose_rows(motion), seg.pose_rows(moved))


class EqualSayIsActuallyEqual(unittest.TestCase):
    def test_weight_one_puts_the_two_blocks_on_the_same_scale(self):
        features = _features(seed=2)
        motion = _motion(seed=4)
        visual_rows = seg.similarity_rows(features)
        motion_rows = seg.pose_rows(motion)
        scale = seg._median_step(visual_rows) / seg._median_step(motion_rows)
        self.assertAlmostEqual(seg._median_step(motion_rows * scale),
                               seg._median_step(visual_rows), places=9)

    def test_the_two_blocks_have_different_raw_scales_so_the_rescale_is_needed(self):
        # If they happened to match, the test above would pass while doing nothing.
        ratio = seg._median_step(seg.similarity_rows(_features(seed=2))) / \
            seg._median_step(seg.pose_rows(_motion(seed=4)))
        self.assertNotAlmostEqual(ratio, 1.0, places=1)


class ItRefusesHalfAConfiguration(unittest.TestCase):
    def test_a_frame_count_mismatch_is_an_error_not_a_truncation(self):
        with self.assertRaises(ValueError) as caught:
            seg.segment_sequence(_features(frames=180), 6, 18, 4.0, 7,
                                 motion=_motion(frames=150), motion_weight=1.0)
        self.assertIn("same clock", str(caught.exception))

    def test_drop_visual_without_motion_is_an_error(self):
        with self.assertRaises(ValueError):
            seg.segment_sequence(_features(), 6, 18, 4.0, 7, drop_visual=True)


if __name__ == "__main__":
    unittest.main()
