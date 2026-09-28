import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tools.fit_motion_normalizer import (
    MOTION_DIM,
    RAW_REPRESENTATION_CONTRACT,
    NormalizerFitError,
    fit_motion_normalizer,
)


def _write_jsonl(path: Path, records):
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _source(
    recording_id: str,
    group: str,
    split: str,
    *,
    duplicate_group=None,
    accepted: bool = True,
    qc_extra=None,
):
    qc = {"accepted_for_training": accepted}
    if qc_extra:
        qc.update(qc_extra)
    return {
        "schema_version": "atomic-source-v1",
        "recording_id": recording_id,
        "retrieval_group_id": group,
        "duplicate_content_group_id": duplicate_group,
        "split": split,
        "qc": qc,
    }


def _sequence(
    sequence_id: str,
    recording_id: str,
    group: str,
    split: str,
    motion_path: Path | str,
    *,
    accepted: bool,
    representation=None,
    qc_extra=None,
    duplicate_group=None,
):
    qc = {"accepted_for_training": accepted}
    if qc_extra:
        qc.update(qc_extra)
    return {
        "schema_version": "atomic-sequence-v1",
        "sequence_id": sequence_id,
        "recording_id": recording_id,
        "retrieval_group_id": group,
        "duplicate_content_group_id": duplicate_group,
        "split": split,
        "assets": {"motion_151_raw": str(motion_path)},
        "representation": dict(RAW_REPRESENTATION_CONTRACT if representation is None else representation),
        "qc": qc,
    }


class FitMotionNormalizerTests(unittest.TestCase):
    def _fit(self, root: Path, sources, sequences, *, output_name="normalizer_v1", block_frames=2):
        source_path = root / "sources.jsonl"
        sequence_path = root / "sequences.jsonl"
        _write_jsonl(source_path, sources)
        _write_jsonl(sequence_path, sequences)
        output = root / output_name
        report = fit_motion_normalizer(
            sequence_path,
            source_path,
            output,
            block_frames=block_frames,
        )
        return report, output, source_path, sequence_path

    def test_fits_exact_minmax_from_accepted_train_only_and_does_not_open_val(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_a = np.linspace(-3.0, 2.0, num=3 * MOTION_DIM, dtype=np.float32).reshape(3, MOTION_DIM)
            train_b = np.linspace(-7.0, 9.0, num=4 * MOTION_DIM, dtype=np.float32).reshape(4, MOTION_DIM)
            np.save(root / "train_a.npy", train_a)
            np.save(root / "train_b.npy", train_b)
            # A validly declared val record has no 3D asset or representation
            # at all.  Success proves non-train identity rows are audited but
            # not read or held to the selected-train raw-array contract.
            sources = [
                _source("recording/train-a", "group/train-a", "train"),
                _source("recording/train-b", "group/train-b", "train"),
                _source("recording/train-pending", "group/train-pending", "train"),
                _source("recording/val", "group/val", "val"),
                _source("recording/test", "group/test", "test"),
            ]
            sequences = [
                _sequence(
                    "sequence/train-a",
                    "recording/train-a",
                    "group/train-a",
                    "train",
                    root / "train_a.npy",
                    accepted=True,
                ),
                _sequence(
                    "sequence/train-b",
                    "recording/train-b",
                    "group/train-b",
                    "train",
                    root / "train_b.npy",
                    accepted=True,
                ),
                _sequence(
                    "sequence/train-pending",
                    "recording/train-pending",
                    "group/train-pending",
                    "train",
                    root / "pending_does_not_exist.npy",
                    accepted=False,
                ),
                {
                    "schema_version": "atomic-sequence-v1",
                    "sequence_id": "sequence/val",
                    "recording_id": "recording/val",
                    "retrieval_group_id": "group/val",
                    "duplicate_content_group_id": None,
                    "split": "val",
                },
                {
                    "schema_version": "atomic-sequence-v1",
                    "sequence_id": "sequence/test",
                    "recording_id": "recording/test",
                    "retrieval_group_id": "group/test",
                    "duplicate_content_group_id": None,
                    "split": "test",
                },
            ]
            report, output, source_path, sequence_path = self._fit(root, sources, sequences)

            saved = torch.load(output / "normalizer.pt", map_location="cpu", weights_only=True)
            self.assertEqual(set(saved), {"data_min", "data_max"})
            self.assertEqual(saved["data_min"].dtype, torch.float32)
            self.assertEqual(tuple(saved["data_min"].shape), (MOTION_DIM,))
            expected = np.concatenate((train_a, train_b), axis=0)
            self.assertTrue(np.array_equal(saved["data_min"].numpy(), expected.min(axis=0)))
            self.assertTrue(np.array_equal(saved["data_max"].numpy(), expected.max(axis=0)))
            self.assertEqual(report["fit_split"], "train")
            self.assertEqual(report["counts"]["selected_sequences"], 2)
            self.assertEqual(report["counts"]["selected_frames"], 7)
            self.assertEqual(report["counts"]["train_not_explicitly_accepted"], 1)
            self.assertEqual(report["counts"]["excluded_val_sequences"], 1)
            self.assertEqual(report["counts"]["excluded_test_sequences"], 1)
            self.assertEqual(
                report["input"]["source_manifest_sha256"],
                hashlib.sha256(source_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                report["input"]["sequence_manifest_sha256"],
                hashlib.sha256(sequence_path.read_bytes()).hexdigest(),
            )
            fitted_rows = [
                json.loads(line)
                for line in (output / "fit_sequences.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["sequence_id"] for row in fitted_rows], ["sequence/train-a", "sequence/train-b"])
            self.assertEqual(
                report["selected_sequences"]["ids_sha256"],
                hashlib.sha256(b"sequence/train-a\nsequence/train-b\n").hexdigest(),
            )

    def test_source_rejected_cannot_be_fit_even_when_its_sequence_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # A missing file makes this also prove the source gate is checked
            # before any selected-train array is opened.
            sources = [
                _source(
                    "recording/rejected",
                    "group/rejected",
                    "train",
                    accepted=False,
                )
            ]
            sequences = [
                _sequence(
                    "sequence/rejected",
                    "recording/rejected",
                    "group/rejected",
                    "train",
                    root / "must_not_be_read.npy",
                    accepted=True,
                )
            ]
            with self.assertRaisesRegex(NormalizerFitError, "source.qc.accepted_for_training"):
                self._fit(root, sources, sequences)
            self.assertFalse((root / "normalizer_v1").exists())

    def test_requires_explicit_nullable_duplicate_group_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            motion = np.zeros((2, MOTION_DIM), dtype=np.float32)
            np.save(root / "motion.npy", motion)

            source_without_group = _source("recording/train", "group/train", "train")
            del source_without_group["duplicate_content_group_id"]
            sequence = _sequence(
                "sequence/train",
                "recording/train",
                "group/train",
                "train",
                root / "motion.npy",
                accepted=True,
            )
            with self.assertRaisesRegex(NormalizerFitError, "must explicitly include duplicate_content_group_id"):
                self._fit(root, [source_without_group], [sequence])
            self.assertFalse((root / "normalizer_v1").exists())

            source = _source("recording/train", "group/train", "train")
            sequence_without_group = dict(sequence)
            del sequence_without_group["duplicate_content_group_id"]
            with self.assertRaisesRegex(NormalizerFitError, "must explicitly include duplicate_content_group_id"):
                self._fit(
                    root,
                    [source],
                    [sequence_without_group],
                    output_name="normalizer_missing_sequence_group",
                )
            self.assertFalse((root / "normalizer_missing_sequence_group").exists())

    def test_rejects_selected_normalized_or_unknown_representation_before_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            motion = np.zeros((2, MOTION_DIM), dtype=np.float32)
            np.save(root / "motion.npy", motion)
            representation = dict(RAW_REPRESENTATION_CONTRACT)
            representation["normalization"] = "normalized"
            sources = [_source("recording/train", "group/train", "train")]
            sequences = [
                _sequence(
                    "sequence/train",
                    "recording/train",
                    "group/train",
                    "train",
                    root / "motion.npy",
                    accepted=True,
                    representation=representation,
                )
            ]
            with self.assertRaisesRegex(NormalizerFitError, "unsafe representation.normalization"):
                self._fit(root, sources, sequences)
            self.assertFalse((root / "normalizer_v1").exists())

    def test_rejects_nonfinite_selected_train_motion_without_repair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            motion = np.zeros((3, MOTION_DIM), dtype=np.float32)
            motion[1, 12] = np.nan
            np.save(root / "nonfinite.npy", motion)
            sources = [_source("recording/train", "group/train", "train")]
            sequences = [
                _sequence(
                    "sequence/train",
                    "recording/train",
                    "group/train",
                    "train",
                    root / "nonfinite.npy",
                    accepted=True,
                )
            ]
            with self.assertRaisesRegex(NormalizerFitError, "contains non-finite values"):
                self._fit(root, sources, sequences)
            self.assertFalse((root / "normalizer_v1").exists())

    def test_rejects_retrieval_group_cross_split_even_when_val_is_not_fitted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            motion = np.zeros((2, MOTION_DIM), dtype=np.float32)
            np.save(root / "train.npy", motion)
            sources = [
                _source("recording/train", "duplicate-group", "train"),
                _source("recording/val", "duplicate-group", "val"),
            ]
            sequences = [
                _sequence(
                    "sequence/train",
                    "recording/train",
                    "duplicate-group",
                    "train",
                    root / "train.npy",
                    accepted=True,
                ),
                _sequence(
                    "sequence/val",
                    "recording/val",
                    "duplicate-group",
                    "val",
                    root / "missing_val.npy",
                    accepted=False,
                ),
            ]
            with self.assertRaisesRegex(NormalizerFitError, "splits retrieval_group_id across splits"):
                self._fit(root, sources, sequences)
            self.assertFalse((root / "normalizer_v1").exists())

    def test_rejects_duplicate_content_group_cross_split_even_when_val_is_not_fitted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            motion = np.zeros((2, MOTION_DIM), dtype=np.float32)
            np.save(root / "train.npy", motion)
            sources = [
                _source(
                    "recording/train",
                    "retrieval/train",
                    "train",
                    duplicate_group="same-underlying-content",
                ),
                _source(
                    "recording/val",
                    "retrieval/val",
                    "val",
                    duplicate_group="same-underlying-content",
                ),
            ]
            sequences = [
                _sequence(
                    "sequence/train",
                    "recording/train",
                    "retrieval/train",
                    "train",
                    root / "train.npy",
                    accepted=True,
                    duplicate_group="same-underlying-content",
                ),
                {
                    "schema_version": "atomic-sequence-v1",
                    "sequence_id": "sequence/val",
                    "recording_id": "recording/val",
                    "retrieval_group_id": "retrieval/val",
                    "duplicate_content_group_id": "same-underlying-content",
                    "split": "val",
                },
            ]
            with self.assertRaisesRegex(NormalizerFitError, "duplicate_content_group_id across splits"):
                self._fit(root, sources, sequences)
            self.assertFalse((root / "normalizer_v1").exists())

    def test_output_bundle_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            motion = np.ones((2, MOTION_DIM), dtype=np.float32)
            np.save(root / "motion.npy", motion)
            sources = [_source("recording/train", "group/train", "train")]
            sequences = [
                _sequence(
                    "sequence/train",
                    "recording/train",
                    "group/train",
                    "train",
                    root / "motion.npy",
                    accepted=True,
                )
            ]
            self._fit(root, sources, sequences)
            with self.assertRaises(FileExistsError):
                self._fit(root, sources, sequences)


if __name__ == "__main__":
    unittest.main()
