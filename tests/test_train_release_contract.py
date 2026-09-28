import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from train_atomic import (
    DatasetReleaseContractError,
    train,
    validate_training_data_root,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class TrainReleaseContractTests(unittest.TestCase):
    def _write_verified_root(self, root: Path) -> Path:
        """Make the compact current materializer layout without invoking training."""
        normalizer_path = root / "normalizer.pt"
        torch.save(
            {
                "data_min": torch.zeros(151, dtype=torch.float32),
                "data_max": torch.ones(151, dtype=torch.float32),
            },
            normalizer_path,
        )
        normalizer_hash = _sha256(normalizer_path)

        split_hashes = {}
        for split in ("train", "val", "test"):
            split_root = root / split
            split_root.mkdir()
            np.save(split_root / "motion.npy", np.zeros((1, 2, 151), dtype=np.float32), allow_pickle=False)
            np.save(split_root / "music.npy", np.zeros((1, 2, 35), dtype=np.float32), allow_pickle=False)
            np.save(split_root / "labels.npy", np.zeros((1, 2), dtype=np.int64), allow_pickle=False)
            np.save(split_root / "label_valid_mask.npy", np.ones((1, 2), dtype=np.bool_), allow_pickle=False)
            (split_root / "names.json").write_text(
                json.dumps(["{}_recording_slice0".format(split)]) + "\n", encoding="utf-8"
            )
            (split_root / "retrieval_groups.json").write_text(
                json.dumps(["performance/{}".format(split)]) + "\n", encoding="utf-8"
            )
            split_hashes[split] = {
                name: _sha256(split_root / name)
                for name in (
                    "motion.npy",
                    "music.npy",
                    "labels.npy",
                    "label_valid_mask.npy",
                    "names.json",
                    "retrieval_groups.json",
                )
            }

        (root / "windows.jsonl").write_text("{}\n", encoding="utf-8")
        (root / "quarantine.jsonl").write_text("", encoding="utf-8")
        source_hash = "a" * 64
        sequence_hash = "b" * 64
        label_hash = "c" * 64
        build = {
            "schema_version": "atomic-window-materialization-v1",
            "materializer_version": "source-manifest-atomic-window-materializer-v1",
            "input_manifests": {
                "sources.jsonl": {"path": "/immutable/source/sources.jsonl", "sha256": source_hash},
                "sequences.jsonl": {"path": "/immutable/source/sequences.jsonl", "sha256": sequence_hash},
                "labels.jsonl": {"path": "/immutable/source/labels.jsonl", "sha256": label_hash},
            },
            "window_policy": {"motion_dim": 151, "music_dim": 35, "fps": 30},
            "representation_contract": {
                "motion_representation_id": "AtomicDance_151D",
                "coordinate_system": "z_up_world_body_only",
                "normalization_state": "normalized",
                "normalization_artifact_sha256": normalizer_hash,
                "normalization_fit_split": "train",
                "camera_in_model_input": False,
            },
            "normalizer": {
                "source_artifact_sha256": normalizer_hash,
                "published_artifact": "normalizer.pt",
                "published_artifact_sha256": normalizer_hash,
                "fit_split": "train",
                "fit_source_manifest_sha256": source_hash,
                "fit_report": "/immutable/normalizer/report.json",
                "fit_report_sha256": "d" * 64,
            },
            "counts": {"materialized_windows": {"train": 1, "val": 1, "test": 1}},
            "artifacts": {
                "normalizer.pt": normalizer_hash,
                "splits": split_hashes,
                "windows.jsonl": _sha256(root / "windows.jsonl"),
                "quarantine.jsonl": _sha256(root / "quarantine.jsonl"),
            },
        }
        (root / "build.json").write_text(json.dumps(build, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return root

    def test_valid_materialized_root_is_pinned_and_checkpoint_persists_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._write_verified_root(Path(directory))
            provenance = validate_training_data_root(str(root))

            self.assertTrue(provenance["release_contract_validated"])
            self.assertTrue(provenance["headline_eligible"])
            self.assertEqual(provenance["validation_split"], "val")
            self.assertEqual(provenance["validation_protocol"], "SOURCE_DISJOINT_VAL_ONLY_TEST_HELD_OUT")
            self.assertEqual(provenance["normalizer"]["sha256"], _sha256(root / "normalizer.pt"))
            self.assertEqual(provenance["source_manifest"]["sha256"], "a" * 64)
            self.assertEqual(provenance["label_manifest"]["sha256"], "c" * 64)
            self.assertEqual(provenance["build"]["sha256"], _sha256(root / "build.json"))

            # A bounded real train call proves the verified release chooses
            # val rather than loading test for its training-time diagnostic.
            result = train(
                SimpleNamespace(
                    stage="planner",
                    data_root=str(root),
                    output_dir=str(root / "run"),
                    device="cpu",
                    resume="",
                    seed=1,
                    epochs=1,
                    log_every_epochs=1,
                    save_every_epochs=1,
                    max_steps=0,
                    limit=1,
                    validation_limit=1,
                    batch_size=1,
                    workers=0,
                    learning_rate=1e-3,
                    weight_decay=0.0,
                    grad_clip=1.0,
                    num_classes=100,
                    motion_dim=151,
                    music_dim=35,
                    seq_len=2,
                    latent_dim=8,
                    layers=1,
                    heads=1,
                    ff_size=16,
                    dropout=0.0,
                    diffusion_steps=1,
                    transition_weight=1.0,
                    cond_drop_prob=0.0,
                    guidance_weight=1.0,
                    draft_noise_ratio=0.25,
                    min_safe_draft_fraction=0.99,
                )
            )
            self.assertEqual(result["metrics"]["validation_split"], "val")
            self.assertEqual(result["metrics"]["validation_protocol"], "SOURCE_DISJOINT_VAL_ONLY_TEST_HELD_OUT")
            self.assertEqual(result["dataset_provenance"], provenance)
            checkpoint_path = Path(result["checkpoint"])
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["dataset_provenance"], provenance)
            self.assertEqual(checkpoint["metrics"]["validation_split"], "val")

    def test_tampered_listed_artifact_is_rejected_before_loader_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._write_verified_root(Path(directory))
            # The build still has the original digest, so a release gate must
            # stop before AtomicSequenceDataset has a chance to mmap it.
            with (root / "train" / "motion.npy").open("ab") as handle:
                handle.write(b"tampered")
            with self.assertRaisesRegex(DatasetReleaseContractError, "artifacts.splits.train.motion.npy SHA-256 mismatch"):
                validate_training_data_root(str(root))

    def test_tampered_explicit_retrieval_groups_are_rejected_before_loader_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._write_verified_root(Path(directory))
            (root / "train" / "retrieval_groups.json").write_text(
                json.dumps(["performance/tampered"]) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                DatasetReleaseContractError,
                "artifacts.splits.train.retrieval_groups.json SHA-256 mismatch",
            ):
                validate_training_data_root(str(root))

    def test_verified_release_requires_the_fixed_30_fps_window_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._write_verified_root(Path(directory))
            build_path = root / "build.json"
            build = json.loads(build_path.read_text(encoding="utf-8"))
            build["window_policy"]["fps"] = 25
            build_path.write_text(json.dumps(build, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(DatasetReleaseContractError, "window_policy.fps must be exactly 30"):
                validate_training_data_root(str(root))

    def test_verified_release_requires_pinned_disjoint_retrieval_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._write_verified_root(Path(directory))
            build_path = root / "build.json"
            build = json.loads(build_path.read_text(encoding="utf-8"))
            # A v1 source-safe release cannot omit the sidecar merely because
            # the arrays/names themselves are present.
            del build["artifacts"]["splits"]["train"]["retrieval_groups.json"]
            build_path.write_text(json.dumps(build, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                DatasetReleaseContractError,
                "artifacts.splits.train is missing retrieval_groups.json",
            ):
                validate_training_data_root(str(root))

        with tempfile.TemporaryDirectory() as directory:
            root = self._write_verified_root(Path(directory))
            # Simulate a self-consistent hand-built bundle: update both the
            # sidecar and its build hash, but make val reuse train's
            # performance group.  Hash pinning alone is insufficient; the
            # release validator must reject the source split overlap.
            groups_path = root / "val" / "retrieval_groups.json"
            groups_path.write_text(json.dumps(["performance/train"]) + "\n", encoding="utf-8")
            build_path = root / "build.json"
            build = json.loads(build_path.read_text(encoding="utf-8"))
            build["artifacts"]["splits"]["val"]["retrieval_groups.json"] = _sha256(groups_path)
            build_path.write_text(json.dumps(build, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                DatasetReleaseContractError,
                "cross-split retrieval-group leakage between train and val",
            ):
                validate_training_data_root(str(root))

    def test_legacy_root_is_allowed_only_as_unverified_test_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            normalizer = root / "normalizer.pt"
            torch.save(
                {
                    "data_min": torch.zeros(151, dtype=torch.float32),
                    "data_max": torch.ones(151, dtype=torch.float32),
                },
                normalizer,
            )

            provenance = validate_training_data_root(str(root))
            self.assertFalse(provenance["release_contract_validated"])
            self.assertFalse(provenance["headline_eligible"])
            self.assertEqual(provenance["release_contract"], "absent_legacy_layout")
            self.assertEqual(provenance["validation_split"], "test")
            self.assertEqual(provenance["validation_protocol"], "LEGACY_TEST_FALLBACK_CODE_SMOKE_ONLY")
            self.assertEqual(provenance["normalizer"]["sha256"], _sha256(normalizer))
            self.assertIsNone(provenance["source_manifest"]["sha256"])
            self.assertIsNone(provenance["label_manifest"]["sha256"])


if __name__ == "__main__":
    unittest.main()
