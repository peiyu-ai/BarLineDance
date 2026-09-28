import unittest

import numpy as np

from tools.postprocess_atomic_plan import (
    TRANSITION,
    majority_vote,
    merge_short_segments,
    postprocess,
    segments_of,
)


class SegmentsTests(unittest.TestCase):
    def test_runs_are_reported_as_half_open_intervals(self):
        self.assertEqual(segments_of(np.array([1, 1, 2, 2, 2])),
                         [(0, 2, 1), (2, 5, 2)])

    def test_empty_input_has_no_segments(self):
        self.assertEqual(segments_of(np.array([], dtype=np.int64)), [])


class MajorityVoteTests(unittest.TestCase):
    def test_an_isolated_frame_error_inside_a_segment_is_removed(self):
        labels = np.array([3] * 10 + [7] + [3] * 10)
        voted = majority_vote(labels, window=5)
        self.assertTrue(np.all(voted == 3))

    def test_a_genuine_boundary_survives(self):
        labels = np.array([3] * 15 + [7] * 15)
        voted = majority_vote(labels, window=5)
        self.assertEqual(len(segments_of(voted)), 2)
        # The boundary may shift by at most half a window, never vanish.
        self.assertEqual(set(np.unique(voted)), {3, 7})

    def test_a_tie_keeps_the_incumbent_rather_than_inventing_a_label(self):
        labels = np.array([1, 1, 2, 2])
        voted = majority_vote(labels, window=4)
        self.assertEqual(set(np.unique(voted)), {1, 2})

    def test_window_of_one_is_a_passthrough(self):
        labels = np.array([1, 5, 1])
        np.testing.assert_array_equal(majority_vote(labels, window=1), labels)


class MergeTests(unittest.TestCase):
    def test_a_short_segment_joins_its_longer_neighbour(self):
        labels = np.array([1] * 20 + [2] * 3 + [3] * 8)
        merged = merge_short_segments(labels, minimum_frames=6)
        self.assertNotIn(2, set(np.unique(merged)))
        self.assertEqual(merged[20], 1)  # 20-frame neighbour beats the 8-frame one

    def test_transitions_are_never_merged_away(self):
        labels = np.array([1] * 20 + [TRANSITION] * 2 + [1] * 20)
        merged = merge_short_segments(labels, minimum_frames=8)
        self.assertEqual(int((merged == TRANSITION).sum()), 2)

    def test_a_fragment_between_two_transitions_becomes_transition(self):
        labels = np.array([TRANSITION] * 10 + [5] * 2 + [TRANSITION] * 10)
        merged = merge_short_segments(labels, minimum_frames=6)
        self.assertTrue(np.all(merged == TRANSITION))

    def test_label_embeddings_override_the_length_heuristic(self):
        # Short segment of label 2 sits between a long 1 and a short 3.  Length
        # alone would pick 1; embeddings make 3 the compatible neighbour.
        labels = np.array([1] * 20 + [2] * 3 + [3] * 8)
        embeddings = np.zeros((4, 2), dtype=np.float32)
        embeddings[1] = [1.0, 0.0]
        embeddings[2] = [0.0, 1.0]
        embeddings[3] = [0.0, 1.0]
        merged = merge_short_segments(labels, minimum_frames=6, embeddings=embeddings)
        self.assertEqual(merged[20], 3)

    def test_merging_terminates_on_a_sequence_of_only_short_segments(self):
        labels = np.array([1, 1, 2, 2, 3, 3, 4, 4])
        merged = merge_short_segments(labels, minimum_frames=6)
        self.assertEqual(len(merged), 8)
        self.assertEqual(len(segments_of(merged)), 1)


class PostprocessTests(unittest.TestCase):
    def test_over_segmentation_is_reduced_and_accounted(self):
        rng = np.random.default_rng(0)
        labels = np.repeat([1, 2, 3, 4], 30).astype(np.int64)
        noisy = labels.copy()
        noisy[rng.choice(len(noisy), size=12, replace=False)] = 9  # speckle
        report = postprocess(noisy, window=9, minimum_frames=12)
        self.assertLess(report["segments_after"], report["segments_before"])
        self.assertEqual(report["segments_after"], 4)
        self.assertEqual(report["frames"], len(noisy))
        self.assertGreater(report["frames_changed"], 0)

    def test_a_clean_plan_is_left_alone(self):
        labels = np.repeat([1, 2, 3], 40).astype(np.int64)
        report = postprocess(labels, window=9, minimum_frames=12)
        np.testing.assert_array_equal(report["labels"], labels)
        self.assertEqual(report["frames_changed"], 0)
        self.assertEqual(report["segments_after"], 3)

    def test_the_compatibility_rule_used_is_recorded(self):
        report = postprocess(np.repeat([1, 2], 40), window=5, minimum_frames=8)
        self.assertEqual(report["compatibility"], "longer_neighbour_then_earlier")
        embedded = postprocess(np.repeat([1, 2], 40), window=5, minimum_frames=8,
                               embeddings=np.eye(3, dtype=np.float32))
        self.assertEqual(embedded["compatibility"], "label_embedding_cosine")


if __name__ == "__main__":
    unittest.main()
