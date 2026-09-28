"""Pin the segmentation invariants of ``tools.segment_visual_atomics``.

The merge loop and the adaptive cluster count are the parts a refactor could
quietly break: a wrong merge direction yields segments below ``L_min`` (which
downstream clustering would happily average into mush), and a broken scaling
returns to ~6 s segments on long sequences.
"""

import unittest

import pytest

import numpy as np

from tools.segment_visual_atomics import cluster_count, segment_sequence


def synthetic_features(block_lengths, dim=32, seed=0):
    """Piecewise-constant features with distinct blocks plus small noise."""
    rng = np.random.RandomState(seed)
    blocks = []
    for index, length in enumerate(block_lengths):
        center = rng.randn(dim)
        center /= np.linalg.norm(center)
        blocks.append(center[None, :] + 0.01 * rng.randn(length, dim))
    return np.concatenate(blocks).astype(np.float32)


class SegmentSequenceTests(unittest.TestCase):
    def test_boundaries_are_ordered_and_cover_sequence(self):
        features = synthetic_features([40, 50, 45, 60])
        boundaries, _ = segment_sequence(features, 4, 24, 4.0, 0)
        self.assertEqual(boundaries[0], 0)
        self.assertEqual(boundaries[-1], len(features))
        self.assertEqual(boundaries, sorted(boundaries))

    def test_min_length_is_enforced(self):
        features = synthetic_features([40, 50, 45, 60], seed=3)
        for min_length in (10, 24, 30):
            boundaries, _ = segment_sequence(features, 6, min_length, 4.0, 0)
            lengths = np.diff(boundaries)
            self.assertTrue(
                (lengths >= min_length).all(),
                "min_length {} violated: {}".format(min_length, lengths.tolist()),
            )

    def test_recovers_planted_block_structure(self):
        """Clearly distinct blocks should produce cuts near the true joints."""
        block_lengths = [60, 60, 60]
        features = synthetic_features(block_lengths, seed=7)
        boundaries, _ = segment_sequence(features, 3, 24, 4.0, 0)
        interior = boundaries[1:-1]
        self.assertEqual(len(interior), 2)
        for true_cut in (60, 120):
            nearest = min(abs(b - true_cut) for b in interior)
            self.assertLessEqual(
                nearest, 8, "no cut within 8 frames of {}".format(true_cut)
            )

    def test_short_sequence_returns_single_segment(self):
        features = synthetic_features([20], seed=1)
        boundaries, _ = segment_sequence(features, 4, 24, 4.0, 0)
        self.assertEqual(boundaries, [0, 20])


class ClusterCountTests(unittest.TestCase):
    def test_explicit_n_wins(self):
        self.assertEqual(cluster_count(1425, 8, 40), 8)

    def test_scales_with_length_and_clamps(self):
        self.assertEqual(cluster_count(319, 0, 40), 8)     # ~10 s sequence
        self.assertEqual(cluster_count(1425, 0, 40), 32)   # ~47 s clamps at 32
        self.assertEqual(cluster_count(50, 0, 40), 2)      # floor at 2


if __name__ == "__main__":
    unittest.main()


def test_merge_refuses_shards_segmented_with_different_parameters(tmp_path):
    """Half a corpus at one L_min and half at another is a silent corruption:
    every consumer reads `config` as describing every record in the file."""
    import json

    from tools.segment_visual_atomics import merge

    def write(name, min_length):
        (tmp_path / name).write_text(json.dumps({
            "features_dir": "f",
            "config": {"clusters": 0, "frames_per_cluster": 34,
                       "min_length_frames": min_length, "index_weight": 4.0,
                       "seed": 1, "fps": 30.0},
            "records": [{"sequence": name, "segments": [{"start": 0, "end": 30,
                                                         "frames": 30}]}],
        }), encoding="utf-8")

    write("a.json", 18)
    write("b.json", 24)
    with pytest.raises(SystemExit):
        merge(str(tmp_path / "*.json"), tmp_path / "merged.json")


def test_merge_refuses_a_sequence_that_appears_in_two_shards(tmp_path):
    import json

    from tools.segment_visual_atomics import merge

    config = {"clusters": 0, "frames_per_cluster": 34, "min_length_frames": 18,
              "index_weight": 4.0, "seed": 1, "fps": 30.0}
    for name in ("a.json", "b.json"):
        (tmp_path / name).write_text(json.dumps({
            "features_dir": "f", "config": config,
            "records": [{"sequence": "same", "segments": [{"start": 0, "end": 30,
                                                           "frames": 30}]}],
        }), encoding="utf-8")
    with pytest.raises(SystemExit):
        merge(str(tmp_path / "*.json"), tmp_path / "merged.json")
