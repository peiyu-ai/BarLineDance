import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tools.build_source_manifest import build_source_manifest
from tools.fit_motion_normalizer import fit_motion_normalizer
from tools.rebase_atomic_aist_source import (
    AISTRebaseError,
    MOTION_DIM,
    _stable_val_group,
    rebase_atomic_aist_source,
)


WINDOW = 4
STRIDE = 2
MUSIC_DIM = 35


def _write_split(root: Path, split: str, names, motions, music):
    directory = root / split
    directory.mkdir(parents=True)
    np.save(directory / "motion.npy", np.asarray(motions, dtype=np.float32))
    np.save(directory / "music.npy", np.asarray(music, dtype=np.float32))
    (directory / "names.json").write_text(json.dumps(names), encoding="utf-8")


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class RebaseAtomicAISTSourceTests(unittest.TestCase):
    def _choose_names(self, validation_fraction: float):
        train = val = None
        for index in range(1000):
            name = "gFI_sFM_cAll_d{:02d}_mFI0".format(index)
            group = "aistpp/" + name
            if _stable_val_group(group, validation_fraction=validation_fraction):
                val = val or name
            else:
                train = train or name
            if train and val:
                return train, val
        self.fail("fixture could not obtain stable train and val groups")

    def _source_bundle(self, root: Path, *, validation_fraction: float):
        data_root = root / "legacy"
        data_root.mkdir()
        train_name, val_name = self._choose_names(validation_fraction)
        # Two camera channels of one recording group must stay together.
        names = [
            train_name + "_ch01_slice0",
            train_name + "_ch02_slice0",
            val_name + "_ch01_slice0",
        ]
        motion = np.zeros((WINDOW, MOTION_DIM), dtype=np.float32)
        motion[:, 0] = np.linspace(-1.0, 1.0, WINDOW, dtype=np.float32)
        motion[:, 1] = np.float32(0.25)
        music = np.arange(WINDOW * MUSIC_DIM, dtype=np.float32).reshape(WINDOW, MUSIC_DIM)
        _write_split(data_root, "train", names, [motion, motion * 0.5, motion * -0.5], [music, music + 1, music + 2])
        test_name = "gTE_sTM_cAll_d99_mTE0"
        _write_split(data_root, "test", [test_name + "_ch01_slice0"], [motion * 0.75], [music + 3])
        normalizer = data_root / "normalizer.pt"
        data_min = torch.full((MOTION_DIM,), -4.0, dtype=torch.float32)
        data_max = torch.full((MOTION_DIM,), 6.0, dtype=torch.float32)
        torch.save(
            {
                "data_min": data_min,
                "data_max": data_max,
                # Match the upstream normalizer contract: its name roster is
                # the complete pre-rebase legacy train-source roster, not an
                # arbitrary token that happens to accompany a valid tensor.
                "training_names": [
                    train_name + "_ch01",
                    train_name + "_ch02",
                    val_name + "_ch01",
                ],
            },
            normalizer,
        )
        bundle = root / "source_manifest_v3"
        build_source_manifest(
            data_root, bundle, splits=("train", "test"), window_length=WINDOW,
            window_stride=STRIDE, motion_dim=MOTION_DIM, music_dim=MUSIC_DIM,
        )
        # A poisoned label file demonstrates that rebasing does not depend on
        # it.  The source builder similarly never opened it.
        (data_root / "train" / "labels.npy").write_bytes(b"not-a-npy-label-file")
        return bundle, normalizer, train_name, val_name, test_name, motion

    def test_publishes_raw_camera_group_safe_release_and_feeds_train_only_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fraction = 0.5
            bundle, normalizer, train_name, val_name, test_name, normalized = self._source_bundle(
                root, validation_fraction=fraction
            )
            output = root / "aist_raw_release"
            report = rebase_atomic_aist_source(bundle, normalizer, output, validation_fraction=fraction)

            self.assertEqual(report["counts"]["sources"], 4)
            self.assertEqual(report["counts"]["groups"], 3)
            self.assertTrue((output / "report.json").is_file())
            self.assertFalse((output / "labels.jsonl").exists())
            self.assertEqual(
                report["input"]["upstream_normalizer_sha256"], hashlib.sha256(normalizer.read_bytes()).hexdigest()
            )
            sources = _read_jsonl(output / "sources.jsonl")
            sequences = _read_jsonl(output / "sequences.jsonl")
            windows = _read_jsonl(output / "windows.jsonl")
            self.assertEqual(len(windows), 4)
            source_by_legacy = {row["legacy_source_name"]: row for row in sources}
            first = source_by_legacy[train_name + "_ch01"]
            second = source_by_legacy[train_name + "_ch02"]
            self.assertEqual(first["retrieval_group_id"], second["retrieval_group_id"])
            self.assertEqual(first["split"], second["split"])
            self.assertEqual(first["split"], "train")
            self.assertEqual(source_by_legacy[val_name + "_ch01"]["split"], "val")
            self.assertEqual(source_by_legacy[test_name + "_ch01"]["split"], "test")
            self.assertTrue(all(row["qc"]["accepted_for_training"] is True for row in sources))
            self.assertTrue(all(row["qc"]["accepted_for_training"] is True for row in sequences))
            self.assertTrue(all(row["representation"]["normalization"] == "raw" for row in sequences))
            self.assertTrue(all(row["representation"]["camera_in_model_input"] is False for row in sequences))
            sequence = next(row for row in sequences if row["recording_id"] == first["recording_id"])
            raw = np.load(output / sequence["motion_path"], allow_pickle=False)
            expected = (normalized + 1.0) * 5.0 - 4.0
            self.assertTrue(np.array_equal(raw, expected.astype(np.float32)))
            self.assertEqual(sequence["assets"]["motion_151_raw"], sequence["motion_path"])
            self.assertEqual(sequence["assets"]["music_35"], sequence["music_path"])
            self.assertEqual(sequence["assets"]["frame_ids"], sequence["frame_ids_path"])
            self.assertTrue(all(row["label_state"] == "unavailable_not_canonical" for row in windows))
            self.assertTrue(all(row["label_space_id"] is None for row in windows))

            # The existing fitter consumes the freshly rebased raw contract,
            # resolving its bundle-relative assets and train-only source group.
            fit_dir = root / "fit"
            fit = fit_motion_normalizer(output / "sequences.jsonl", output / "sources.jsonl", fit_dir)
            self.assertGreater(fit["counts"]["selected_sequences"], 0)
            self.assertTrue((fit_dir / "normalizer.pt").is_file())

    def test_refuses_wrong_normalizer_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, normalizer, *_ = self._source_bundle(root, validation_fraction=0.5)
            wrong = root / "wrong.pt"
            torch.save(
                {
                    "data_min": torch.zeros(MOTION_DIM),
                    "data_max": torch.ones(MOTION_DIM),
                    "training_names": ["not-the-upstream-artifact"],
                },
                wrong,
            )
            output = root / "must_not_exist"
            with self.assertRaisesRegex(AISTRebaseError, "normalizer hash"):
                rebase_atomic_aist_source(bundle, wrong, output, validation_fraction=0.5)
            self.assertFalse(output.exists())

    def test_refuses_normalizer_with_wrong_train_roster_even_when_manifest_hash_is_rebound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, normalizer, *_ = self._source_bundle(root, validation_fraction=0.5)
            payload = torch.load(normalizer, map_location="cpu", weights_only=True)
            payload["training_names"] = ["not-a-legacy-source"]
            poisoned = root / "poisoned_roster.pt"
            torch.save(payload, poisoned)
            # This simulates a malicious/stale provenance edit.  The rebase
            # tool must bind the tensor roster to source identities rather
            # than accepting a matching build.json hash alone.
            build_path = bundle / "build.json"
            build = json.loads(build_path.read_text(encoding="utf-8"))
            build["input"]["normalizer_sha256"] = hashlib.sha256(poisoned.read_bytes()).hexdigest()
            build_path.write_text(json.dumps(build, sort_keys=True) + "\n", encoding="utf-8")
            output = root / "must_not_exist"
            with self.assertRaisesRegex(AISTRebaseError, "training_names do not exactly match"):
                rebase_atomic_aist_source(bundle, poisoned, output, validation_fraction=0.5)
            self.assertFalse(output.exists())

    def test_output_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, normalizer, *_ = self._source_bundle(root, validation_fraction=0.5)
            output = root / "release"
            rebase_atomic_aist_source(bundle, normalizer, output, validation_fraction=0.5)
            with self.assertRaisesRegex(AISTRebaseError, "refusing to overwrite"):
                rebase_atomic_aist_source(bundle, normalizer, output, validation_fraction=0.5)


if __name__ == "__main__":
    unittest.main()
