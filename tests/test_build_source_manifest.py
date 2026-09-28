import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.build_source_manifest import ManifestBuildError, build_source_manifest


MOTION_DIM = 151
MUSIC_DIM = 35
WINDOW_LENGTH = 4
WINDOW_STRIDE = 2


def _write_split(root, split, names, motion_rows, music_rows):
    split_root = root / split
    split_root.mkdir(parents=True)
    np.save(split_root / "motion.npy", np.asarray(motion_rows, dtype=np.float32))
    np.save(split_root / "music.npy", np.asarray(music_rows, dtype=np.float32))
    (split_root / "names.json").write_text(json.dumps(names), encoding="utf-8")


def _timeline(frames, dims, offset):
    return (np.arange(frames * dims, dtype=np.float32).reshape(frames, dims) + offset)


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class BuildSourceManifestTests(unittest.TestCase):
    def _build(self, root, output, splits=("train", "test")):
        return build_source_manifest(
            root,
            output,
            splits=splits,
            window_length=WINDOW_LENGTH,
            window_stride=WINDOW_STRIDE,
            motion_dim=MOTION_DIM,
            music_dim=MUSIC_DIM,
            fps=30,
        )

    def test_reconstructs_source_timelines_without_loading_or_emitting_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "legacy"
            root.mkdir()
            (root / "manifest.json").write_text('{"release":"fixture"}\n', encoding="utf-8")
            dance_a_motion = _timeline(6, MOTION_DIM, 10.0)
            dance_a_music = _timeline(6, MUSIC_DIM, 100.0)
            dance_b_motion = _timeline(4, MOTION_DIM, 1000.0)
            dance_b_music = _timeline(4, MUSIC_DIM, 2000.0)
            # Deliberately place slice1 before slice0 in indexed rows.  The
            # manifest must use the slice identity, never dataloader order.
            _write_split(
                root,
                "train",
                ["dance_a_slice1", "dance_a_slice0", "dance_b_slice0"],
                [dance_a_motion[2:6], dance_a_motion[:4], dance_b_motion],
                [dance_a_music[2:6], dance_a_music[:4], dance_b_music],
            )
            test_motion = _timeline(4, MOTION_DIM, 3000.0)
            test_music = _timeline(4, MUSIC_DIM, 4000.0)
            _write_split(
                root,
                "test",
                ["dance_c_slice0"],
                [test_motion],
                [test_music],
            )

            output = Path(directory) / "source_manifest"
            report = self._build(root, output)
            self.assertEqual(report["counts"], {"sources": 3, "sequences": 3, "windows": 4, "labels": 0})
            self.assertFalse((output / "labels.jsonl").exists())

            sources = _read_jsonl(output / "sources.jsonl")
            sequences = _read_jsonl(output / "sequences.jsonl")
            windows = _read_jsonl(output / "windows.jsonl")
            self.assertEqual([row["recording_id"] for row in sources], [
                "aistpp/dance_a", "aistpp/dance_b", "aistpp/dance_c"
            ])
            self.assertEqual({row["split"] for row in sources}, {"train", "test"})
            self.assertTrue(all(row["source_kind"] == "aistpp" for row in sources))
            for row in [*sources, *sequences, *windows]:
                self.assertEqual(row["retrieval_group_id"], row["recording_id"])
                self.assertIsNone(row["duplicate_content_group_id"])
                self.assertEqual(row["split_status"], "provisional_legacy_source_disjoint")
                self.assertIn("not an approved held-out benchmark", row["split_note"])
                self.assertIn(row["split"], {"train", "test"})
            sequence = next(row for row in sequences if row["recording_id"] == "aistpp/dance_a")
            self.assertTrue(sequence["is_contiguous"])
            self.assertEqual(sequence["frame_count"], 6)
            self.assertTrue(
                np.array_equal(np.load(output / sequence["motion_path"]), dance_a_motion)
            )
            self.assertTrue(
                np.array_equal(np.load(output / sequence["music_path"]), dance_a_music)
            )
            self.assertTrue(
                np.array_equal(np.load(output / sequence["frame_ids_path"]), np.arange(6))
            )
            dance_a_windows = [row for row in windows if row["recording_id"] == "aistpp/dance_a"]
            self.assertEqual([row["start_frame"] for row in dance_a_windows], [0, 2])
            self.assertEqual([row["legacy_window"]["input_row_index"] for row in dance_a_windows], [1, 0])
            self.assertTrue(all(row["label_state"] == "unavailable_not_canonical" for row in windows))
            self.assertTrue(all(row["label_space_id"] is None for row in windows))

            # A published bundle is immutable from this builder's perspective.
            with self.assertRaises(ManifestBuildError):
                self._build(root, output)

    def test_refuses_recording_level_cross_split_leakage_before_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "legacy"
            root.mkdir()
            motion = _timeline(4, MOTION_DIM, 0.0)
            music = _timeline(4, MUSIC_DIM, 0.0)
            _write_split(root, "train", ["shared_recording_slice0"], [motion], [music])
            _write_split(root, "test", ["shared_recording_slice0"], [motion], [music])
            output = Path(directory) / "source_manifest"
            with self.assertRaisesRegex(ManifestBuildError, "recording-level split leakage"):
                self._build(root, output)
            self.assertFalse(output.exists())

    def test_refuses_nonidentical_motion_or_music_overlap_before_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "legacy"
            root.mkdir()
            motion = _timeline(6, MOTION_DIM, 0.0)
            music = _timeline(6, MUSIC_DIM, 0.0)
            corrupt_motion = motion[2:6].copy()
            corrupt_motion[0, 0] += 1.0
            _write_split(
                root,
                "train",
                ["broken_slice0", "broken_slice1"],
                [motion[:4], corrupt_motion],
                [music[:4], music[2:6]],
            )
            output = Path(directory) / "source_manifest"
            with self.assertRaisesRegex(ManifestBuildError, "inconsistent motion values"):
                self._build(root, output, splits=("train",))
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
