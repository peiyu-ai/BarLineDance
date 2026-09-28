import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tools.discover_kinematic_atomics import (
    KinematicDiscoveryError,
    discover_kinematic_atomics,
    kinematic_frame_descriptor,
    rotation6d_to_matrix,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _motion(frames: int, offset: float) -> np.ndarray:
    values = np.zeros((frames, 151), dtype=np.float32)
    values[:, :4] = (np.arange(frames)[:, None] + np.arange(4)[None, :] + int(offset)) % 2
    values[:, 4] = np.linspace(0.0, 1.0 + offset, frames, dtype=np.float32)
    values[:, 5] = np.sin(np.linspace(0.0, 3.0 + offset, frames, dtype=np.float32))
    values[:, 6] = np.cos(np.linspace(0.0, 2.0 + offset, frames, dtype=np.float32))
    rotations = values[:, 7:].reshape(frames, 24, 6)
    rotations[..., 0] = 1.0
    rotations[..., 4] = 1.0
    # Give all joints a smooth, non-degenerate variation so the temporal
    # segmenter has 3-D kinematic changes to partition.
    rotations[..., 1] = np.sin(np.linspace(0.0, 4.0 + offset, frames, dtype=np.float32))[:, None] * 0.2
    rotations[..., 3] = np.cos(np.linspace(0.0, 5.0 + offset, frames, dtype=np.float32))[:, None] * 0.2
    # AtomicDance's z-up root-facing axis is the third rotation column, while
    # its local up direction is the second.  Give the fixture a valid root
    # orientation (and a gentle genuine turn) rather than an identity matrix
    # whose third column points vertically.
    base_root = np.asarray(((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)), dtype=np.float32)
    angles = np.linspace(0.0, 0.5 + offset * 0.1, frames, dtype=np.float32)
    yaw = np.zeros((frames, 3, 3), dtype=np.float32)
    yaw[:, 0, 0] = np.cos(angles)
    yaw[:, 0, 1] = -np.sin(angles)
    yaw[:, 1, 0] = np.sin(angles)
    yaw[:, 1, 1] = np.cos(angles)
    yaw[:, 2, 2] = 1.0
    root_matrices = np.matmul(yaw, base_root[None, :, :])
    rotations[:, 0] = root_matrices[:, :2, :].reshape(frames, 6)
    return values


def _apply_global_z_yaw(motion: np.ndarray, angle: float) -> np.ndarray:
    """Express the same body motion in a globally yaw-rotated z-up frame."""
    values = np.asarray(motion, dtype=np.float32).copy()
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    yaw = np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)), dtype=np.float32
    )
    root = values[:, 4:7].copy()
    values[:, 4:7] = np.matmul(yaw[None, :, :], root[..., None]).squeeze(-1)
    rotations = rotation6d_to_matrix(values[:, 7:].reshape(len(values), 24, 6))
    rotations[:, 0] = np.matmul(yaw[None, :, :], rotations[:, 0])
    # AtomicDance/PyTorch3D rotation-6D stores the first two *rows*.
    values[:, 7:] = rotations[..., :2, :].reshape(len(values), -1)
    return values


def _fixture(
    root: Path,
    *,
    target_shift: float = 0.0,
    leak_group: bool = False,
    leak_content: bool = False,
    target_name_prefix: str = "",
    source_fps: object = 30,
    sequence_fps: object = 30,
):
    root.mkdir(parents=True, exist_ok=True)
    source_rows = []
    sequence_rows = []
    groups = [
        ("train_a", "train", 0.0),
        ("train_b", "train", 1.0),
        (target_name_prefix + "val_a", "val", 2.0 + target_shift),
        (target_name_prefix + "test_a", "test", 3.0 + target_shift),
    ]
    for number, (name, split, offset) in enumerate(groups):
        recording = "recording/{}".format(name)
        group = "performance/{}".format(name)
        if leak_group and split == "val":
            group = "performance/train_a"
        content_sha256 = "{:064x}".format(1 if leak_content and split == "val" else number + 1)
        raw = _motion(48, offset)
        normalized = raw.copy()
        raw_path = root / (name + "_raw.npy")
        normalized_path = root / (name + "_normalized.npy")
        np.save(raw_path, raw, allow_pickle=False)
        np.save(normalized_path, normalized, allow_pickle=False)
        qc = {"accepted_for_training": True, "status": "passed"}
        source_row = {
            "recording_id": recording,
            "retrieval_group_id": group,
            "duplicate_content_group_id": None,
            "content_sha256": content_sha256,
            "split": split,
            "qc": qc,
        }
        if source_fps is not None:
            source_row["fps"] = source_fps
        source_rows.append(source_row)
        sequence_row = {
            "sequence_id": recording + "/sequence0",
            "recording_id": recording,
            "retrieval_group_id": group,
            "duplicate_content_group_id": None,
            "split": split,
            "frame_count": len(raw),
            "qc": qc,
            "assets": {
                "motion_151_raw": str(raw_path),
                "motion_151_raw_sha256": _sha256(raw_path),
                "motion_151_model_input": str(normalized_path),
                "motion_151_model_input_sha256": _sha256(normalized_path),
            },
            "representation": {
                "motion": "AtomicDance_151D",
                "coordinate_system": "z_up_world_body_only",
                "normalization": "normalized",
                "camera_in_model_input": False,
                "normalizer_fit_split": "train",
                "normalizer_artifact_sha256": "a" * 64,
            },
            "normalization": {
                "state": "normalized",
                "input_motion_151_raw": str(raw_path),
                "input_motion_151_raw_sha256": _sha256(raw_path),
            },
        }
        if sequence_fps is not None:
            sequence_row["fps"] = sequence_fps
        sequence_rows.append(sequence_row)
    source_path = root / "sources.jsonl"
    sequence_path = root / "sequences.jsonl"
    _write_jsonl(source_path, source_rows)
    _write_jsonl(sequence_path, sequence_rows)
    return source_path, sequence_path


class KinematicAtomicDiscoveryTests(unittest.TestCase):
    def _run(self, root: Path, *, target_shift: float = 0.0, target_name_prefix: str = ""):
        sources, sequences = _fixture(
            root, target_shift=target_shift, target_name_prefix=target_name_prefix
        )
        return discover_kinematic_atomics(
            sources,
            sequences,
            root / "labels",
            num_classes=2,
            keep_quantile=1.0,
            minimum_cluster_support=1,
            segment_stride=1,
            target_segment_frames=8,
            minimum_segment_frames=4,
            transition_gutter=2,
            segment_iterations=3,
            kmeans_iterations=8,
            embedding_frames=12,
            dct_coefficients=4,
            seed=23,
            device="cpu",
        )

    def test_rotation_converter_matches_6d_contract_and_descriptor_is_finite(self):
        motion = _motion(12, 0.0)
        matrices = rotation6d_to_matrix(motion[:, 7:].reshape(12, 24, 6))
        self.assertEqual(matrices.shape, (12, 24, 3, 3))
        identity = np.matmul(np.swapaxes(matrices, -1, -2), matrices)
        self.assertTrue(np.allclose(identity, np.eye(3), atol=1e-5))
        descriptor = kinematic_frame_descriptor(motion)
        self.assertEqual(descriptor.shape, (12, 4 + 3 + 3 + 24 * 6 + 24))
        self.assertTrue(np.isfinite(descriptor).all())

    def test_descriptor_is_invariant_to_constant_global_yaw_without_erasing_turns(self):
        motion = _motion(24, 0.0)
        descriptor = kinematic_frame_descriptor(motion)
        rotated_descriptor = kinematic_frame_descriptor(_apply_global_z_yaw(motion, 1.37))
        # A camera/world yaw gauge changes both root translation and global
        # root orientation.  The descriptor must be unchanged as a whole,
        # including its root pose and angular-velocity channels, so actual
        # turning dynamics are retained rather than normalized away per frame.
        self.assertTrue(np.allclose(descriptor, rotated_descriptor, atol=2e-5, rtol=2e-5))
        self.assertGreater(float(np.linalg.norm(descriptor[-1] - descriptor[0])), 0.0)

    def test_full_sequence_labels_have_explicit_unknown_and_transition_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = self._run(root)
            output = root / "labels"
            self.assertTrue((output / "producer.npz").is_file())
            self.assertTrue((output / "labels.jsonl").is_file())
            self.assertEqual(report["counts"]["total_clusters"], 2)
            rows = [json.loads(line) for line in (output / "labels.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 4)
            for row in rows:
                labels = np.load(row["labels_path"], allow_pickle=False)
                mask = np.load(row["label_valid_mask_path"], allow_pickle=False)
                probabilities = np.load(row["probabilities_path"], allow_pickle=False)
                self.assertEqual(labels.shape, (48,))
                self.assertEqual(mask.dtype, np.bool_)
                self.assertTrue(np.all(labels[~mask] == -1))
                self.assertTrue(np.all((labels[mask] >= 0) & (labels[mask] <= 2)))
                self.assertEqual(probabilities.shape, (48, 3))
                self.assertEqual(row["fit_split"], "train")
                self.assertEqual(row["input_normalization_state"], "normalized")
                self.assertEqual(row["producer_artifact_sha256"], _sha256(output / "producer.npz"))

    def test_val_and_test_motion_do_not_change_train_fit_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._run(root / "first", target_shift=0.0)
            self._run(root / "second", target_shift=100.0)
            with np.load(root / "first" / "labels" / "producer.npz", allow_pickle=False) as first, np.load(
                root / "second" / "labels" / "producer.npz", allow_pickle=False
            ) as second:
                for key in (
                    "frame_mean",
                    "frame_std",
                    "embedding_mean",
                    "embedding_std",
                    "centers",
                    "cluster_distance_thresholds",
                    "cluster_support",
                    "softmax_temperature",
                ):
                    self.assertTrue(np.array_equal(first[key], second[key]), key)

    def test_held_out_identity_order_does_not_change_train_fit_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # The same target values receive names that sort before the train
            # rows.  A positional seed (seed + all-row index) used to change
            # train segmentation here; a stable sequence-id seed must not.
            self._run(root / "first")
            self._run(root / "second", target_name_prefix="aaa_")
            with np.load(root / "first" / "labels" / "producer.npz", allow_pickle=False) as first, np.load(
                root / "second" / "labels" / "producer.npz", allow_pickle=False
            ) as second:
                for key in (
                    "frame_mean",
                    "frame_std",
                    "embedding_mean",
                    "embedding_std",
                    "centers",
                    "cluster_distance_thresholds",
                    "cluster_support",
                    "softmax_temperature",
                ):
                    self.assertTrue(np.array_equal(first[key], second[key]), key)

    def test_retrieval_group_cross_split_is_rejected_before_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, sequences = _fixture(root, leak_group=True)
            output = root / "must_not_exist"
            with self.assertRaisesRegex(KinematicDiscoveryError, "crosses"):
                discover_kinematic_atomics(
                    sources,
                    sequences,
                    output,
                    num_classes=2,
                    keep_quantile=1.0,
                    minimum_cluster_support=1,
                    segment_stride=1,
                    target_segment_frames=8,
                    minimum_segment_frames=4,
                    segment_iterations=2,
                    kmeans_iterations=2,
                    embedding_frames=12,
                    dct_coefficients=4,
                )
            self.assertFalse(output.exists())

    def test_exact_content_cross_split_is_rejected_before_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, sequences = _fixture(root, leak_content=True)
            output = root / "must_not_exist"
            with self.assertRaisesRegex(KinematicDiscoveryError, "content_sha256.*crosses"):
                discover_kinematic_atomics(
                    sources,
                    sequences,
                    output,
                    num_classes=2,
                    keep_quantile=1.0,
                    minimum_cluster_support=1,
                    segment_stride=1,
                    target_segment_frames=8,
                    minimum_segment_frames=4,
                    segment_iterations=2,
                    kmeans_iterations=2,
                    embedding_frames=12,
                    dct_coefficients=4,
                )
            self.assertFalse(output.exists())

    def test_requires_explicit_30fps_source_and_sequence_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, sequences = _fixture(root, source_fps=None)
            with self.assertRaisesRegex(KinematicDiscoveryError, "explicit numeric fps"):
                discover_kinematic_atomics(
                    sources,
                    sequences,
                    root / "missing-source-fps",
                    num_classes=2,
                    keep_quantile=1.0,
                    minimum_cluster_support=1,
                    segment_stride=1,
                    target_segment_frames=8,
                    minimum_segment_frames=4,
                    segment_iterations=2,
                    kmeans_iterations=2,
                    embedding_frames=12,
                    dct_coefficients=4,
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, sequences = _fixture(root, sequence_fps=24)
            with self.assertRaisesRegex(KinematicDiscoveryError, "fps must be exactly 30"):
                discover_kinematic_atomics(
                    sources,
                    sequences,
                    root / "wrong-sequence-fps",
                    num_classes=2,
                    keep_quantile=1.0,
                    minimum_cluster_support=1,
                    segment_stride=1,
                    target_segment_frames=8,
                    minimum_segment_frames=4,
                    segment_iterations=2,
                    kmeans_iterations=2,
                    embedding_frames=12,
                    dct_coefficients=4,
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is optional for discovery")
    def test_cuda_fit_and_apply_keep_cluster_tensors_on_one_device(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, sequences = _fixture(root)
            report = discover_kinematic_atomics(
                sources,
                sequences,
                root / "cuda_labels",
                num_classes=2,
                keep_quantile=1.0,
                minimum_cluster_support=1,
                segment_stride=1,
                target_segment_frames=8,
                minimum_segment_frames=4,
                transition_gutter=2,
                segment_iterations=2,
                kmeans_iterations=3,
                embedding_frames=12,
                dct_coefficients=4,
                seed=23,
                device="cuda:0",
            )
            self.assertEqual(report["config"]["clustering"]["device"], "cuda:0")


if __name__ == "__main__":
    unittest.main()
