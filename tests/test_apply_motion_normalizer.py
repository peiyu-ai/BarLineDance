import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.apply_motion_normalizer import (
    MOTION_DIM,
    NormalizerApplicationError,
    apply_motion_normalizer,
)
from tools.fit_motion_normalizer import RAW_REPRESENTATION_CONTRACT, fit_motion_normalizer
from tools.materialize_atomic_windows import materialize_atomic_windows


def _write_jsonl(path: Path, records) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8"
    )


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(
    recording_id: str,
    group: str,
    split: str,
    *,
    duplicate_group=None,
    accepted: bool = True,
):
    return {
        "schema_version": "atomic-source-v1",
        "recording_id": recording_id,
        "retrieval_group_id": group,
        "duplicate_content_group_id": duplicate_group,
        "fps": 30,
        "split": split,
        "content_sha256": hashlib.sha256(("content:" + recording_id).encode("utf-8")).hexdigest(),
        "qc": {"accepted_for_training": accepted},
    }


def _sequence(
    sequence_id: str,
    recording_id: str,
    group: str,
    split: str,
    motion_path: Path,
    *,
    music_path: Path,
    frame_ids_path: Path,
    manifest_parent: Path,
    accepted: bool,
    duplicate_group=None,
    representation=None,
    camera="camera/provenance.json",
):
    values = np.load(motion_path, mmap_mode="r")
    def relative(path: Path) -> str:
        return str(path.resolve().relative_to(manifest_parent.resolve()))

    raw_reference = relative(motion_path)
    music_reference = relative(music_path)
    frame_ids_reference = relative(frame_ids_path)
    return {
        "schema_version": "atomic-sequence-v1",
        "sequence_id": sequence_id,
        "recording_id": recording_id,
        "retrieval_group_id": group,
        "duplicate_content_group_id": duplicate_group,
        "fps": 30,
        "split": split,
        "assets": {
            "motion_151_raw": raw_reference,
            "camera": camera,
            "music_35": music_reference,
            "frame_ids": frame_ids_reference,
            "source_cache": "cache/{}".format(sequence_id.replace("/", "_")),
        },
        "representation": dict(RAW_REPRESENTATION_CONTRACT if representation is None else representation),
        "frame_count": int(values.shape[0]),
        "source_start_frame": 0,
        "source_end_frame_exclusive": int(values.shape[0]),
        "is_contiguous": True,
        "motion_path": raw_reference,
        "motion_sha256": _sha256(motion_path),
        "music_path": music_reference,
        "music_sha256": _sha256(music_path),
        "frame_ids_path": frame_ids_reference,
        "frame_ids_sha256": _sha256(frame_ids_path),
        "timeline": {"fps": 30, "frame_count": int(values.shape[0]), "frame_ids_path": frame_ids_reference},
        "qc": {"accepted_for_training": accepted},
    }


class ApplyMotionNormalizerTests(unittest.TestCase):
    def _prepare_valid_bundle(self, root: Path):
        train = np.zeros((2, MOTION_DIM), dtype=np.float32)
        train[:, 0] = 3.0  # exact constant dimension: safe range must be one.
        train[:, 1] = np.asarray((0.0, 10.0), dtype=np.float32)
        train[:, 2] = 5.0
        val = np.zeros((2, MOTION_DIM), dtype=np.float32)
        val[:, 0] = 3.0
        val[:, 1] = np.asarray((20.0, 5.0), dtype=np.float32)  # deliberately outside train range.
        val[:, 2] = 5.0
        test = np.zeros((1, MOTION_DIM), dtype=np.float32)
        test[:, 0] = 3.0
        test[:, 1] = -10.0  # deliberately outside train range in the other direction.
        test[:, 2] = 5.0
        assets_root = root / "source_release" / "assets"
        assets_root.mkdir(parents=True)
        source_root = assets_root.parent
        asset_paths = {}
        for name, values in (("train", train), ("val", val), ("test", test)):
            motion_path = assets_root / "{}.npy".format(name)
            music_path = assets_root / "{}-music.npy".format(name)
            frame_ids_path = assets_root / "{}-frame-ids.npy".format(name)
            np.save(motion_path, values)
            np.save(music_path, np.zeros((len(values), 35), dtype=np.float32))
            np.save(frame_ids_path, np.arange(len(values), dtype=np.int64))
            asset_paths[name] = (motion_path, music_path, frame_ids_path)

        sources = [
            _source("recording/train", "retrieval/train", "train"),
            _source("recording/val", "retrieval/val", "val"),
            _source("recording/test", "retrieval/test", "test"),
        ]
        sequences = [
            _sequence(
                "sequence/train",
                "recording/train",
                "retrieval/train",
                "train",
                asset_paths["train"][0],
                music_path=asset_paths["train"][1],
                frame_ids_path=asset_paths["train"][2],
                manifest_parent=source_root,
                accepted=True,
                camera="camera/train.json",
            ),
            _sequence(
                "sequence/val",
                "recording/val",
                "retrieval/val",
                "val",
                asset_paths["val"][0],
                music_path=asset_paths["val"][1],
                frame_ids_path=asset_paths["val"][2],
                manifest_parent=source_root,
                accepted=True,
                camera="camera/val.json",
            ),
            _sequence(
                "sequence/test",
                "recording/test",
                "retrieval/test",
                "test",
                asset_paths["test"][0],
                music_path=asset_paths["test"][1],
                frame_ids_path=asset_paths["test"][2],
                manifest_parent=source_root,
                accepted=True,
                camera="camera/test.json",
            ),
        ]
        source_path = source_root / "sources.jsonl"
        sequence_path = source_root / "sequences_raw.jsonl"
        _write_jsonl(source_path, sources)
        _write_jsonl(sequence_path, sequences)
        normalizer_bundle = root / "normalizer_train_only"
        fit_motion_normalizer(sequence_path, source_path, normalizer_bundle, block_frames=1)
        return {
            "sources": sources,
            "sequences": sequences,
            "source_path": source_path,
            "sequence_path": sequence_path,
            "normalizer_bundle": normalizer_bundle,
            "raw": {"train": train, "val": val, "test": test},
            "asset_paths": asset_paths,
            "source_root": source_root,
        }

    def test_exact_frozen_transform_round_trip_and_canonicalizes_retained_local_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = self._prepare_valid_bundle(root)
            input_bytes = prepared["sequence_path"].read_bytes()
            output = root / "normalized_v1"
            report = apply_motion_normalizer(
                prepared["sequence_path"],
                prepared["source_path"],
                prepared["normalizer_bundle"],
                output,
                block_frames=1,
            )

            self.assertEqual(prepared["sequence_path"].read_bytes(), input_bytes)
            self.assertEqual(report["normalizer"]["fit_split"], "train")
            self.assertTrue(report["provenance"]["val_test_used_frozen_train_normalizer"])
            self.assertEqual(report["counts"]["sequences_by_split"], {"test": 1, "train": 1, "val": 1})
            rows = {row["sequence_id"]: row for row in _read_jsonl(output / "sequences_normalized.jsonl")}
            import torch

            parameters = torch.load(
                prepared["normalizer_bundle"] / "normalizer.pt", map_location="cpu", weights_only=True
            )
            data_min = parameters["data_min"].numpy()
            data_max = parameters["data_max"].numpy()
            safe_range = np.where(data_max == data_min, np.float32(1.0), data_max - data_min)

            for split, sequence_id in (("train", "sequence/train"), ("val", "sequence/val"), ("test", "sequence/test")):
                row = rows[sequence_id]
                raw_path = Path(row["assets"]["motion_151_raw"])
                normalized_path = Path(row["assets"]["motion_151_normalized"])
                expected_motion, expected_music, expected_frame_ids = prepared["asset_paths"][split]
                self.assertTrue(normalized_path.is_file())
                self.assertEqual(row["assets"]["motion_151_model_input"], str(normalized_path))
                self.assertEqual(raw_path, expected_motion.resolve())
                self.assertEqual(row["assets"]["motion_151_raw_sha256"], _sha256(raw_path))
                self.assertEqual(row["assets"]["motion_151_normalized_sha256"], _sha256(normalized_path))
                self.assertEqual(row["motion_path"], str(normalized_path))
                self.assertEqual(row["motion_sha256"], _sha256(normalized_path))
                self.assertEqual(row["assets"]["music_35"], str(expected_music.resolve()))
                self.assertEqual(row["music_path"], str(expected_music.resolve()))
                self.assertEqual(row["assets"]["frame_ids"], str(expected_frame_ids.resolve()))
                self.assertEqual(row["frame_ids_path"], str(expected_frame_ids.resolve()))
                self.assertEqual(
                    row["timeline"]["frame_ids_path"], str(expected_frame_ids.resolve())
                )
                self.assertEqual(
                    row["assets"]["camera"], str((prepared["source_root"] / "camera" / "{}.json".format(split)).resolve())
                )
                self.assertEqual(
                    row["assets"]["source_cache"],
                    str((prepared["source_root"] / "cache" / sequence_id.replace("/", "_")).resolve()),
                )
                self.assertEqual(row["representation"]["normalization_state"], "normalized")
                self.assertEqual(row["representation"]["normalization"], "normalized")
                self.assertFalse(row["representation"]["camera_in_model_input"])
                self.assertEqual(row["normalization_state"], "normalized")
                self.assertEqual(row["normalization_fit_split"], "train")
                self.assertEqual(
                    row["normalization_artifact_sha256"], report["normalizer"]["artifact_sha256"]
                )
                self.assertEqual(
                    row["normalization"]["normalizer_fit_report"],
                    str((prepared["normalizer_bundle"] / "report.json").resolve()),
                )
                normalized_values = np.load(normalized_path)
                expected = np.float32(2.0) * (prepared["raw"][split] - data_min) / safe_range - np.float32(1.0)
                np.testing.assert_array_equal(normalized_values, expected)
                # The inverse is deliberately unclipped too.  It recovers
                # val/test values outside the train range using frozen params.
                recovered = (normalized_values + np.float32(1.0)) * safe_range / np.float32(2.0) + data_min
                np.testing.assert_allclose(recovered, prepared["raw"][split], rtol=0, atol=1e-6)

            self.assertTrue(np.all(np.load(Path(rows["sequence/train"]["assets"]["motion_151_normalized"]))[:, 0] == -1.0))
            self.assertEqual(float(np.load(Path(rows["sequence/val"]["assets"]["motion_151_normalized"]))[0, 1]), 3.0)
            self.assertEqual(float(np.load(Path(rows["sequence/test"]["assets"]["motion_151_normalized"]))[0, 1]), -3.0)

    def test_relative_source_assets_remain_materializable_from_the_new_manifest_root(self):
        """Regression: applying a normalizer must not rebase raw-bundle paths.

        The raw manifest is deliberately below ``source_release/`` while the
        normalized manifest is a sibling.  Materialization exercises every
        relevant alias (normalized motion, music, and frame ids), rather than
        merely checking a string rewrite in the output JSONL.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = self._prepare_valid_bundle(root)
            normalized_root = root / "normalized_release"
            apply_motion_normalizer(
                prepared["sequence_path"],
                prepared["source_path"],
                prepared["normalizer_bundle"],
                normalized_root,
                block_frames=1,
            )
            normalized_manifest = normalized_root / "sequences_normalized.jsonl"
            rows = _read_jsonl(normalized_manifest)
            labels_root = root / "kinematic_labels"
            labels_root.mkdir()
            source_hash = _sha256(prepared["source_path"])
            labels = []
            for index, row in enumerate(rows):
                frame_count = row["frame_count"]
                labels_path = labels_root / "labels_{}.npy".format(index)
                mask_path = labels_root / "mask_{}.npy".format(index)
                np.save(labels_path, np.arange(frame_count, dtype=np.int64) % 3, allow_pickle=False)
                np.save(mask_path, np.ones(frame_count, dtype=np.bool_), allow_pickle=False)
                labels.append(
                    {
                        "sequence_id": row["sequence_id"],
                        "recording_id": row["recording_id"],
                        "retrieval_group_id": row["retrieval_group_id"],
                        "duplicate_content_group_id": row["duplicate_content_group_id"],
                        "split": row["split"],
                        "status": "accepted",
                        "labels_path": str(labels_path),
                        "label_valid_mask_path": str(mask_path),
                        "labels_sha256": _sha256(labels_path),
                        "label_valid_mask_sha256": _sha256(mask_path),
                        "label_space_id": "atomic-101-v1",
                        "producer_version": "relative-path-regression-v1",
                        "fit_split": "train",
                        "fit_source_manifest_sha256": source_hash,
                        "producer_artifact_sha256": hashlib.sha256(
                            ("producer:" + row["sequence_id"]).encode("utf-8")
                        ).hexdigest(),
                        "input_motion_sha256": row["motion_sha256"],
                        "input_motion_representation_id": row["motion_representation_id"],
                        "input_normalization_state": row["normalization_state"],
                        "input_coordinate_system": row["coordinate_system"],
                        "input_normalization_artifact_sha256": row[
                            "normalization_artifact_sha256"
                        ],
                    }
                )
            labels_manifest = labels_root / "labels.jsonl"
            _write_jsonl(labels_manifest, labels)

            indexed_root = root / "indexed_release"
            report = materialize_atomic_windows(
                prepared["source_path"],
                normalized_manifest,
                labels_manifest,
                indexed_root,
                window_length=1,
                window_stride=1,
                motion_dim=MOTION_DIM,
                music_dim=35,
                num_classes=101,
            )
            self.assertEqual(report["counts"]["materialized_windows"], {"train": 2, "val": 2, "test": 1})
            self.assertTrue((indexed_root / "train" / "motion.npy").is_file())
            self.assertTrue((indexed_root / "val" / "music.npy").is_file())
            self.assertTrue((indexed_root / "test" / "labels.npy").is_file())

    def test_rejects_stale_source_manifest_before_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = self._prepare_valid_bundle(root)
            # Semantically equivalent whitespace still makes the source split
            # artifact stale; a re-fit is required rather than a silent reuse.
            prepared["source_path"].write_text(
                prepared["source_path"].read_text(encoding="utf-8") + "\n", encoding="utf-8"
            )
            output = root / "must_not_publish"
            with self.assertRaisesRegex(NormalizerApplicationError, "stale normalizer"):
                apply_motion_normalizer(
                    prepared["sequence_path"],
                    prepared["source_path"],
                    prepared["normalizer_bundle"],
                    output,
                )
            self.assertFalse(output.exists())

    def test_rejects_stale_sequence_manifest_before_parsing_or_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = self._prepare_valid_bundle(root)
            # Equivalent JSON with one extra blank line has a different frozen
            # input hash.  It must not be treated as the corpus fit by the
            # normalizer or opened to inspect its raw arrays.
            prepared["sequence_path"].write_text(
                prepared["sequence_path"].read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            output = root / "must_not_publish"
            with self.assertRaisesRegex(
                NormalizerApplicationError,
                "stale normalizer: report sequence_manifest_sha256",
            ):
                apply_motion_normalizer(
                    prepared["sequence_path"],
                    prepared["source_path"],
                    prepared["normalizer_bundle"],
                    output,
                )
            self.assertFalse(output.exists())

    def test_rejects_mixed_raw_and_normalized_input_representations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = self._prepare_valid_bundle(root)
            mixed = list(prepared["sequences"])
            mixed[1] = dict(mixed[1])
            mixed[1]["representation"] = dict(mixed[1]["representation"])
            mixed[1]["representation"]["normalization"] = "normalized"
            mixed_path = root / "sequences_mixed.jsonl"
            _write_jsonl(mixed_path, mixed)
            # This test exercises the later raw-contract gate.  The separate
            # stale-manifest test above proves arbitrary mutation is rejected
            # before parsing; a deliberately re-bound report is required here
            # to reach the representation validator.
            report_path = prepared["normalizer_bundle"] / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["input"]["sequence_manifest_sha256"] = _sha256(mixed_path)
            report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
            output = root / "must_not_publish"
            with self.assertRaisesRegex(NormalizerApplicationError, "unsafe raw representation.normalization"):
                apply_motion_normalizer(
                    mixed_path,
                    prepared["source_path"],
                    prepared["normalizer_bundle"],
                    output,
                )
            self.assertFalse(output.exists())

    def test_rejects_normalizer_report_with_nonraw_contract_and_cross_split_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = self._prepare_valid_bundle(root)
            report_path = prepared["normalizer_bundle"] / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["representation_contract"] = dict(report["representation_contract"])
            report["representation_contract"]["coordinate_system"] = "camera_relative_y_up"
            report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
            output = root / "bad_representation"
            with self.assertRaisesRegex(NormalizerApplicationError, "exact raw AtomicDance-151D contract"):
                apply_motion_normalizer(
                    prepared["sequence_path"],
                    prepared["source_path"],
                    prepared["normalizer_bundle"],
                    output,
                )
            self.assertFalse(output.exists())

            # Restore the raw-contract report, then bind it to a deliberately
            # unsafe source manifest.  The source-level group audit must still
            # reject it even though a malicious/stale report hash was changed.
            report["representation_contract"] = dict(RAW_REPRESENTATION_CONTRACT)
            unsafe_sources = list(prepared["sources"])
            unsafe_sources[1] = dict(unsafe_sources[1])
            unsafe_sources[1]["retrieval_group_id"] = "retrieval/train"
            _write_jsonl(prepared["source_path"], unsafe_sources)
            report["input"]["source_manifest_sha256"] = _sha256(prepared["source_path"])
            report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
            output = root / "unsafe_groups"
            with self.assertRaisesRegex(NormalizerApplicationError, "source-safety violation: retrieval_group_id"):
                apply_motion_normalizer(
                    prepared["sequence_path"],
                    prepared["source_path"],
                    prepared["normalizer_bundle"],
                    output,
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
