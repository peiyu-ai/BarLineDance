import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dataset.atomic import source_id_from_name
from dataset.atomic_dataset import AtomicSequenceDataset
from tools.materialize_atomic_windows import MaterializationError, materialize_atomic_windows


MOTION_DIM = 151
MUSIC_DIM = 35
WINDOW_LENGTH = 4


def _sha_bytes(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


class MaterializeAtomicWindowsTests(unittest.TestCase):
    def _fixture(self, root, specs):
        """Write a small explicit source/sequence/label manifest trio.

        ``specs`` has source and sequence identity plus optional labels/masks,
        status, and representation fields.  Every accepted fixture binds its
        label provenance to the exact bytes of sources.jsonl.
        """
        manifest_root = root / "manifest"
        label_root = root / "labels"
        assets_root = manifest_root / "assets"
        label_assets_root = label_root / "assets"
        assets_root.mkdir(parents=True)
        label_assets_root.mkdir(parents=True)
        sources = {}
        sequences = []
        pending_labels = []
        originals = {}
        for index, spec in enumerate(specs):
            recording_id = spec["recording_id"]
            split = spec.get("split", "train")
            retrieval_group_id = spec.get("retrieval_group_id", recording_id)
            duplicate_group = spec.get("duplicate_content_group_id")
            nested_wild_schema = spec.get("nested_wild_schema", False)
            source_row = {
                "recording_id": recording_id,
                "retrieval_group_id": retrieval_group_id,
                "duplicate_content_group_id": duplicate_group,
                "fps": spec.get("source_fps", 30),
                "split": split,
                "qc": {"accepted_for_training": spec.get("source_accepted", True)},
            }
            source_content_sha256 = spec.get("content_sha256", _sha_bytes("content:" + recording_id))
            # Which field carries the identity.  AIST++ arrives through two
            # ingestion paths and only one of them writes content_sha256.
            identity_field = spec.get("identity_field", "content_sha256")
            if nested_wild_schema:
                source_row["assets"] = {"content_sha256": source_content_sha256}
            else:
                source_row[identity_field] = source_content_sha256
            if spec.get("music_sha256") is not None:
                source_row["music_sha256"] = spec["music_sha256"]
            sources.setdefault(recording_id, source_row)
            frame_count = spec.get("frame_count", 6)
            base = float(spec.get("base", index * 10000))
            motion = (
                np.arange(frame_count * MOTION_DIM, dtype=np.float32).reshape(frame_count, MOTION_DIM) + base
            )
            music = (
                np.arange(frame_count * MUSIC_DIM, dtype=np.float32).reshape(frame_count, MUSIC_DIM) + base + 0.25
            )
            frame_ids = np.arange(spec.get("source_start", 0), spec.get("source_start", 0) + frame_count, dtype=np.int64)
            motion_path = assets_root / "motion_{}.npy".format(index)
            music_path = assets_root / "music_{}.npy".format(index)
            frame_ids_path = assets_root / "frame_ids_{}.npy".format(index)
            np.save(motion_path, motion, allow_pickle=False)
            np.save(music_path, music, allow_pickle=False)
            np.save(frame_ids_path, frame_ids, allow_pickle=False)
            representation_id = spec.get("motion_representation_id", "AtomicDance_151D")
            normalization_state = spec.get("normalization_state", "normalized")
            normalization_hash = spec.get("normalization_artifact_sha256")
            normalization_fit_split = spec.get("normalization_fit_split")
            sequence_id = spec["sequence_id"]
            motion_reference = str(motion_path.resolve()) if spec.get("absolute_sequence_assets") else "assets/{}".format(motion_path.name)
            music_reference = str(music_path.resolve()) if spec.get("absolute_sequence_assets") else "assets/{}".format(music_path.name)
            frame_ids_reference = str(frame_ids_path.resolve()) if spec.get("absolute_sequence_assets") else "assets/{}".format(frame_ids_path.name)
            coordinate_system = spec.get(
                "coordinate_system",
                "z_up_world_body_only",
            )
            common_sequence = {
                "sequence_id": sequence_id,
                "recording_id": recording_id,
                "retrieval_group_id": retrieval_group_id,
                "duplicate_content_group_id": duplicate_group,
                "fps": spec.get("sequence_fps", 30),
                "split": split,
                "qc": {"accepted_for_training": spec.get("sequence_accepted", True)},
            }
            if nested_wild_schema:
                # Match reconcile-wild-hmr / extract_wild_music_features exactly:
                # timeline + assets + representation, with absolute artifacts.
                common_sequence.update(
                    {
                        "timeline": {
                            "fps": spec.get("timeline_fps", spec.get("sequence_fps", 30)),
                            "frame_count": frame_count,
                            "source_start_frame": int(frame_ids[0]),
                            "source_end_frame_exclusive": int(frame_ids[-1] + 1),
                            "frame_ids_path": str(frame_ids_path.resolve()),
                            "motion_frames_are_contiguous": spec.get("is_contiguous", True),
                        },
                        "assets": {
                            "motion_151_raw": str(motion_path.resolve()),
                            "motion_151_raw_sha256": _sha_file(motion_path),
                            "music_35": str(music_path.resolve()),
                            "music_35_sha256": _sha_file(music_path),
                        },
                        "representation": {
                            "motion": representation_id,
                            "coordinate_system": coordinate_system,
                            "normalization": normalization_state,
                            **(
                                {
                                    "normalization_artifact_sha256": normalization_hash,
                                    "normalization_fit_split": normalization_fit_split,
                                }
                                if normalization_state != "raw"
                                else {}
                            ),
                        },
                    }
                )
            else:
                common_sequence.update(
                    {
                        "frame_count": frame_count,
                        "source_start_frame": int(frame_ids[0]),
                        "source_end_frame_exclusive": int(frame_ids[-1] + 1),
                        "is_contiguous": spec.get("is_contiguous", True),
                        "motion_path": motion_reference,
                        "music_path": music_reference,
                        "frame_ids_path": frame_ids_reference,
                        "motion_sha256": _sha_file(motion_path),
                        "music_sha256": _sha_file(music_path),
                        "motion_representation_id": representation_id,
                        "coordinate_system": coordinate_system,
                        "normalization_state": normalization_state,
                        "normalization_artifact_sha256": normalization_hash,
                        "normalization_fit_split": normalization_fit_split,
                        "camera_in_model_input": False,
                    }
                )
            sequences.append(common_sequence)
            originals[sequence_id] = {"motion": motion, "music": music}
            pending_labels.append((index, spec, motion, motion_path, representation_id, normalization_state, normalization_hash))
        sources_path = manifest_root / "sources.jsonl"
        sequences_path = manifest_root / "sequences.jsonl"
        labels_path = label_root / "labels.jsonl"
        _write_jsonl(sources_path, [sources[key] for key in sorted(sources)])
        source_manifest_hash = _sha_file(sources_path)
        # Build the exact immutable fit-artifact shape consumed by inference
        # and verified by the materializer.  The fixture motion arrays are
        # declared as already-applied model input; values themselves are only
        # used to assert range-exact copying below.
        normalizer_dir = root / "frozen_normalizer"
        normalizer_dir.mkdir()
        normalizer_path = normalizer_dir / "normalizer.pt"
        torch.save(
            {
                "data_min": torch.zeros(MOTION_DIM, dtype=torch.float32),
                "data_max": torch.ones(MOTION_DIM, dtype=torch.float32),
            },
            normalizer_path,
        )
        normalizer_hash = _sha_file(normalizer_path)
        normalizer_report_path = normalizer_dir / "report.json"
        normalizer_report_path.write_text(
            json.dumps(
                {
                    "schema_version": "atomicdance-motion-normalizer-fit-v1",
                    "fit_split": "train",
                    "representation_contract": {
                        "motion": "AtomicDance_151D",
                        "coordinate_system": "z_up_world_body_only",
                        "normalization": "raw",
                        "camera_in_model_input": False,
                    },
                    "input": {"source_manifest_sha256": source_manifest_hash},
                    "normalizer": {"sha256": normalizer_hash},
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        for index, (spec, sequence) in enumerate(zip(specs, sequences)):
            # A raw fixture is intentionally left as raw to exercise the hard
            # rejection gate.  Every ordinary fixture emulates apply-normalizer
            # output and binds to the same frozen artifact/report.
            state = (
                sequence["representation"]["normalization"]
                if spec.get("nested_wild_schema", False)
                else sequence["normalization_state"]
            )
            if state == "raw":
                continue
            if spec.get("nested_wild_schema", False):
                assets = sequence["assets"]
                motion_path = Path(assets["motion_151_raw"])
                motion_hash = _sha_file(motion_path)
                # Deliberately distinguish raw and model-input arrays.  This
                # proves the materializer follows apply-normalizer's
                # ``motion_151_model_input`` rather than accidentally using
                # the retained raw HMR tensor.
                model_motion = np.asarray(np.load(motion_path), dtype=np.float32) + np.float32(0.5)
                model_motion_path = assets_root / "motion_model_input_{}.npy".format(index)
                np.save(model_motion_path, model_motion, allow_pickle=False)
                model_motion_hash = _sha_file(model_motion_path)
                assets.update(
                    {
                        "motion_151_model_input": str(model_motion_path),
                        "motion_151_model_input_sha256": model_motion_hash,
                        "motion_151_normalized": str(model_motion_path),
                        "motion_151_normalized_sha256": model_motion_hash,
                    }
                )
                originals[sequence["sequence_id"]]["model_motion"] = model_motion
                sequence["representation"].update(
                    {
                        "normalization": "normalized",
                        "normalization_state": "normalized",
                        "camera_in_model_input": False,
                        "normalizer_artifact_sha256": normalizer_hash,
                        "normalizer_fit_split": "train",
                        "normalization_artifact_sha256": normalizer_hash,
                        "normalization_fit_split": "train",
                    }
                )
                sequence.update(
                    {
                        "normalization_state": "normalized",
                        "normalization_artifact_sha256": normalizer_hash,
                        "normalization_fit_split": "train",
                        "camera_in_model_input": False,
                    }
                )
            else:
                sequence.update(
                    {
                        "normalization_state": "normalized",
                        "normalization_artifact_sha256": normalizer_hash,
                        "normalization_fit_split": "train",
                        "camera_in_model_input": False,
                    }
                )
            sequence["normalizer_artifact_path"] = str(normalizer_path)
            sequence["normalizer_fit_report_path"] = str(normalizer_report_path)
            sequence["normalizer_fit_report_sha256"] = _sha_file(normalizer_report_path)
            sequence["normalization"] = {
                "schema_version": "fixture-normalization-v1",
                "state": "normalized",
                "normalizer_artifact": str(normalizer_path),
                "normalizer_artifact_sha256": normalizer_hash,
                "normalizer_fit_report": str(normalizer_report_path),
                "normalizer_fit_report_sha256": _sha_file(normalizer_report_path),
                "fit_split": "train",
            }
        _write_jsonl(sequences_path, sequences)
        labels = []
        for index, spec, motion, motion_path, representation_id, normalization_state, normalization_hash in pending_labels:
            if spec.get("omit_label", False):
                continue
            sequence = sequences[index]
            if spec.get("nested_wild_schema", False):
                representation_id = sequence["representation"]["motion"]
                normalization_state = sequence["representation"]["normalization"]
                normalization_hash = sequence["representation"].get("normalization_artifact_sha256")
                label_motion_path = Path(
                    sequence["assets"].get("motion_151_model_input", sequence["assets"]["motion_151_raw"])
                )
            else:
                representation_id = sequence["motion_representation_id"]
                normalization_state = sequence["normalization_state"]
                normalization_hash = sequence.get("normalization_artifact_sha256")
                label_motion_path = motion_path
            frame_count = len(motion)
            values = np.asarray(spec.get("labels", np.arange(frame_count) % 5), dtype=np.int64)
            mask = np.asarray(spec.get("mask", np.ones(frame_count, dtype=bool)), dtype=bool)
            labels_array_path = label_assets_root / "labels_{}.npy".format(index)
            mask_path = label_assets_root / "mask_{}.npy".format(index)
            np.save(labels_array_path, values, allow_pickle=False)
            np.save(mask_path, mask, allow_pickle=False)
            labels_reference = str(labels_array_path.resolve()) if spec.get("absolute_label_assets") else "assets/{}".format(labels_array_path.name)
            mask_reference = str(mask_path.resolve()) if spec.get("absolute_label_assets") else "assets/{}".format(mask_path.name)
            labels.append(
                {
                    "sequence_id": spec["sequence_id"],
                    "recording_id": spec["recording_id"],
                    "retrieval_group_id": spec.get("retrieval_group_id", spec["recording_id"]),
                    "duplicate_content_group_id": spec.get("duplicate_content_group_id"),
                    "split": spec.get("split", "train"),
                    "status": spec.get("status", "accepted"),
                    "labels_path": labels_reference,
                    "label_valid_mask_path": mask_reference,
                    "labels_sha256": _sha_file(labels_array_path),
                    "label_valid_mask_sha256": _sha_file(mask_path),
                    "label_space_id": "atomic-101-v1",
                    "producer_version": "fixture-labeler-v1",
                    "fit_split": spec.get("fit_split", "train"),
                    "fit_source_manifest_sha256": spec.get("fit_source_manifest_sha256", source_manifest_hash),
                    "producer_artifact_sha256": _sha_bytes("producer:" + spec["sequence_id"]),
                    "input_motion_sha256": _sha_file(label_motion_path),
                    "input_motion_representation_id": representation_id,
                    "input_normalization_state": normalization_state,
                    "input_coordinate_system": (
                        "z_up_world_body_only"
                        if spec.get("nested_wild_schema", False) and "coordinate_system" not in spec
                        else spec.get("coordinate_system", "z_up_world_body_only")
                    ),
                    "input_normalization_artifact_sha256": normalization_hash,
                }
            )
        _write_jsonl(labels_path, labels)
        return sources_path, sequences_path, labels_path, originals

    def _materialize(self, sources, sequences, labels, output, **kwargs):
        return materialize_atomic_windows(
            sources,
            sequences,
            labels,
            output,
            window_length=WINDOW_LENGTH,
            window_stride=WINDOW_LENGTH,
            motion_dim=MOTION_DIM,
            music_dim=MUSIC_DIM,
            **kwargs,
        )

    def test_materializes_exact_cuts_and_recording_safe_legacy_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, sequences, labels, originals = self._fixture(
                root,
                [
                    {
                        "recording_id": "rec/a",
                        "sequence_id": "rec/a/track0",
                        "base": 1,
                        "absolute_sequence_assets": True,
                        "absolute_label_assets": True,
                    },
                    {"recording_id": "rec/a", "sequence_id": "rec/a/track1", "base": 1001},
                    {"recording_id": "rec/b", "sequence_id": "rec/b/track0", "split": "val", "base": 2001},
                    {"recording_id": "rec/c", "sequence_id": "rec/c/track0", "split": "test", "base": 3001},
                ],
            )
            output = root / "indexed"
            report = self._materialize(sources, sequences, labels, output)
            self.assertEqual(report["counts"]["materialized_windows"], {"train": 2, "val": 1, "test": 1})
            train_motion = np.load(output / "train/motion.npy")
            train_music = np.load(output / "train/music.npy")
            train_labels = np.load(output / "train/labels.npy")
            train_mask = np.load(output / "train/label_valid_mask.npy")
            self.assertEqual(train_motion.shape, (2, WINDOW_LENGTH, MOTION_DIM))
            self.assertEqual(train_music.shape, (2, WINDOW_LENGTH, MUSIC_DIM))
            self.assertTrue(np.array_equal(train_motion[0], originals["rec/a/track0"]["motion"][:WINDOW_LENGTH]))
            self.assertTrue(np.array_equal(train_motion[1], originals["rec/a/track1"]["motion"][:WINDOW_LENGTH]))
            self.assertTrue(np.array_equal(train_music[1], originals["rec/a/track1"]["music"][:WINDOW_LENGTH]))
            self.assertTrue(np.array_equal(train_labels[0], np.array([0, 1, 2, 3], dtype=np.int64)))
            self.assertTrue(bool(np.all(train_mask)))
            names = json.loads((output / "train/names.json").read_text(encoding="utf-8"))
            self.assertEqual(names, ["rec/a_slice0", "rec/a_slice1"])
            retrieval_groups = json.loads(
                (output / "train/retrieval_groups.json").read_text(encoding="utf-8")
            )
            self.assertEqual(retrieval_groups, ["rec/a", "rec/a"])
            self.assertEqual(
                report["artifacts"]["splits"]["train"]["retrieval_groups.json"],
                _sha_file(output / "train/retrieval_groups.json"),
            )
            self.assertTrue(all(source_id_from_name(name) == "rec/a" for name in names))
            windows = [json.loads(line) for line in (output / "windows.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["sequence_id"] for row in windows if row["split"] == "train"], ["rec/a/track0", "rec/a/track1"])
            self.assertTrue(all(row["label_valid_fraction"] == 1.0 for row in windows))
            train_dataset = AtomicSequenceDataset(str(output), split="train")
            self.assertEqual(len(train_dataset), 2)
            self.assertEqual(train_dataset[1]["name"], "rec/a_slice1")
            self.assertEqual(train_dataset[1]["retrieval_group_id"], "rec/a")
            self.assertEqual(len(AtomicSequenceDataset(str(output), split="val")), 1)
            self.assertEqual(len(AtomicSequenceDataset(str(output), split="test")), 1)
            self.assertEqual(report["representation_contract"]["normalization_state"], "normalized")
            self.assertTrue((output / "normalizer.pt").is_file())
            self.assertEqual(_sha_file(output / "normalizer.pt"), report["normalizer"]["published_artifact_sha256"])
            self.assertEqual(
                report["normalizer"]["source_artifact_sha256"], report["normalizer"]["published_artifact_sha256"]
            )
            self.assertEqual(report["normalizer"]["fit_split"], "train")
            with self.assertRaisesRegex(MaterializationError, "refusing to overwrite"):
                self._materialize(sources, sequences, labels, output)

    def test_partial_or_ambiguous_unknown_labels_are_quarantined_not_transition_filled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, sequences, labels, _ = self._fixture(
                root,
                [
                    {"recording_id": "rec/good", "sequence_id": "rec/good/track0"},
                    {"recording_id": "rec/val", "sequence_id": "rec/val/track0", "split": "val"},
                    {"recording_id": "rec/test", "sequence_id": "rec/test/track0", "split": "test"},
                    {
                        "recording_id": "rec/partial",
                        "sequence_id": "rec/partial/track0",
                        "labels": np.array([0, 1, -1, 3, 4, 0]),
                        "mask": np.array([True, True, False, True, True, True]),
                    },
                    {
                        "recording_id": "rec/ambiguous",
                        "sequence_id": "rec/ambiguous/track0",
                        "labels": np.array([0, 1, 0, 3, 4, 0]),
                        "mask": np.array([True, True, False, True, True, True]),
                    },
                ],
            )
            output = root / "indexed"
            self._materialize(sources, sequences, labels, output, min_label_valid_fraction=0.5)
            self.assertEqual(np.load(output / "train/labels.npy").shape[0], 1)
            self.assertTrue(np.all(np.load(output / "train/label_valid_mask.npy")))
            quarantine = [json.loads(line) for line in (output / "quarantine.jsonl").read_text(encoding="utf-8").splitlines()]
            partial = [row for row in quarantine if row.get("recording_id") == "rec/partial"]
            self.assertEqual(partial[0]["reason_code"], "partial_label_mask_not_legacy_compatible")
            ambiguous = [row for row in quarantine if row.get("recording_id") == "rec/ambiguous"]
            self.assertEqual(ambiguous[0]["reason_code"], "invalid_label_not_sentinel")
            self.assertFalse(any(row.get("recording_id") == "rec/partial" for row in [
                json.loads(line) for line in (output / "windows.jsonl").read_text(encoding="utf-8").splitlines()
            ]))

    def test_a_source_identified_only_by_motion_sha256_is_accepted(self):
        """The contract asks for one of several hashes; this tool asked for one.

        371 of AIST++'s 1,363 recordings come from the official-supplement
        path, which writes motion and music hashes but no content_sha256, so
        demanding that spelling blocked a corpus the contract accepts.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, sequences, labels, _ = self._fixture(
                root,
                [
                    {"recording_id": "rec/a", "sequence_id": "rec/a/track0"},
                    {"recording_id": "rec/b", "sequence_id": "rec/b/track0",
                     "identity_field": "motion_sha256"},
                    {"recording_id": "rec/v", "sequence_id": "rec/v/track0", "split": "val",
                     "identity_field": "motion_sha256"},
                    {"recording_id": "rec/t", "sequence_id": "rec/t/track0", "split": "test"},
                ],
            )
            output = root / "mixed"
            self._materialize(sources, sequences, labels, output)
            build = json.loads((output / "build.json").read_text(encoding="utf-8"))
            # Both fields are named, because hashes from different fields are
            # not comparable and a later audit must not average over that.
            self.assertEqual(build["source_identity_hash_fields"],
                             {"content_sha256": 2, "motion_sha256": 2})

    def test_music_sha256_alone_does_not_identify_a_recording(self):
        # It identifies the *song*: 1,363 AIST++ rows carry 158 distinct music
        # hashes, one of them shared by 23 recordings, because the supplement
        # copied each music array from a same-song donor.  Accepting it would
        # turn every second dance to one song into duplicated content.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, sequences, labels, _ = self._fixture(
                root,
                [{"recording_id": "rec/a", "sequence_id": "rec/a/track0",
                  "identity_field": "music_sha256"}],
            )
            output = root / "song-identity"
            with self.assertRaisesRegex(MaterializationError, "music_sha256 is not accepted"):
                self._materialize(sources, sequences, labels, output)
            self.assertFalse(output.exists())

    def test_two_splits_may_share_a_song_without_being_a_leak(self):
        # The regression this pair exists for: dances to the same song land in
        # different splits by design, and a gate keyed on the music would call
        # that cross-split leakage and refuse the corpus.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shared_song = _sha_bytes("music:mBR0")
            sources, sequences, labels, _ = self._fixture(
                root,
                [
                    {"recording_id": "rec/train", "sequence_id": "rec/train/track0",
                     "music_sha256": shared_song},
                    {"recording_id": "rec/test", "sequence_id": "rec/test/track0",
                     "split": "test", "identity_field": "motion_sha256",
                     "music_sha256": shared_song},
                    {"recording_id": "rec/val", "sequence_id": "rec/val/track0",
                     "split": "val", "music_sha256": shared_song},
                ],
            )
            output = root / "same-song"
            self._materialize(sources, sequences, labels, output)
            self.assertTrue((output / "build.json").is_file())

    def test_a_source_with_no_usable_identity_hash_is_still_refused(self):
        # Widening the accepted fields must not become accepting none of them.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, sequences, labels, _ = self._fixture(
                root,
                [{"recording_id": "rec/a", "sequence_id": "rec/a/track0",
                  "identity_field": "unrelated_sha256"}],
            )
            output = root / "no-identity"
            with self.assertRaisesRegex(MaterializationError, "content_sha256 or motion_sha256"):
                self._materialize(sources, sequences, labels, output)
            self.assertFalse(output.exists())

    def test_rejects_cross_split_groups_and_label_provenance_not_bound_to_sources_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, sequences, labels, _ = self._fixture(
                root,
                [
                    {"recording_id": "rec/train", "sequence_id": "rec/train/track0", "retrieval_group_id": "shared"},
                    {"recording_id": "rec/test", "sequence_id": "rec/test/track0", "split": "test", "retrieval_group_id": "shared"},
                ],
            )
            output = root / "cross-split"
            with self.assertRaisesRegex(MaterializationError, "cross-split leakage"):
                self._materialize(sources, sequences, labels, output)
            self.assertFalse(output.exists())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, sequences, labels, _ = self._fixture(
                root,
                [
                    {"recording_id": "rec/good", "sequence_id": "rec/good/track0"},
                    {"recording_id": "rec/val", "sequence_id": "rec/val/track0", "split": "val"},
                    {"recording_id": "rec/test", "sequence_id": "rec/test/track0", "split": "test"},
                    {
                        "recording_id": "rec/wrong-provenance",
                        "sequence_id": "rec/wrong-provenance/track0",
                        "fit_source_manifest_sha256": _sha_bytes("wrong-frozen-split"),
                    }
                ],
            )
            output = root / "wrong-provenance"
            self._materialize(sources, sequences, labels, output)
            self.assertEqual(np.load(output / "train/labels.npy").shape[0], 1)
            quarantine = [json.loads(line) for line in (output / "quarantine.jsonl").read_text(encoding="utf-8").splitlines()]
            wrong = [row for row in quarantine if row.get("recording_id") == "rec/wrong-provenance"]
            self.assertEqual(wrong[0]["reason_code"], "label_fit_source_manifest_mismatch")

    def test_requires_explicit_consistent_30fps_source_and_sequence_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, sequences, labels, _ = self._fixture(
                root,
                [{"recording_id": "rec/missing-fps", "sequence_id": "rec/missing-fps/track0", "source_fps": None}],
            )
            output = root / "missing-fps"
            with self.assertRaisesRegex(MaterializationError, "requires explicit numeric fps"):
                self._materialize(sources, sequences, labels, output)
            self.assertFalse(output.exists())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, sequences, labels, _ = self._fixture(
                root,
                [{"recording_id": "rec/24fps", "sequence_id": "rec/24fps/track0", "sequence_fps": 24}],
            )
            output = root / "24fps"
            with self.assertRaisesRegex(MaterializationError, "fps must be exactly 30"):
                self._materialize(sources, sequences, labels, output)
            self.assertFalse(output.exists())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, sequences, labels, _ = self._fixture(
                root,
                [
                    {
                        "recording_id": "rec/contradictory-fps",
                        "sequence_id": "rec/contradictory-fps/track0",
                        "nested_wild_schema": True,
                        "sequence_fps": 30,
                        "timeline_fps": 24,
                    }
                ],
            )
            output = root / "contradictory-fps"
            with self.assertRaisesRegex(MaterializationError, "conflicting fps"):
                self._materialize(sources, sequences, labels, output)
            self.assertFalse(output.exists())

    def test_requires_qc_acceptance_and_one_explicit_representation_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, sequences, labels, _ = self._fixture(
                root,
                [
                    {"recording_id": "rec/good", "sequence_id": "rec/good/track0"},
                    {"recording_id": "rec/not-accepted", "sequence_id": "rec/not-accepted/track0", "sequence_accepted": False},
                    {
                        "recording_id": "rec/other-rep",
                        "sequence_id": "rec/other-rep/track0",
                        "motion_representation_id": "atomic-151-yup-normalized-v1",
                        "normalization_state": "normalized",
                    },
                ],
            )
            output = root / "mixed"
            with self.assertRaisesRegex(MaterializationError, "motion_representation_id must be"):
                self._materialize(sources, sequences, labels, output)
            self.assertFalse(output.exists())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # The label is generated against the same mutated coordinate
            # string; acceptance must still fail rather than trusting a label
            # to bless camera-relative model input.
            sources, sequences, labels, _ = self._fixture(
                root,
                [
                    {
                        "recording_id": "rec/camera-relative",
                        "sequence_id": "rec/camera-relative/track0",
                        "coordinate_system": "camera_relative_y_up",
                    }
                ],
            )
            output = root / "camera-relative"
            with self.assertRaisesRegex(MaterializationError, "coordinate_system must be"):
                self._materialize(sources, sequences, labels, output)
            self.assertFalse(output.exists())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, sequences, labels, _ = self._fixture(
                root,
                [
                    {"recording_id": "rec/good", "sequence_id": "rec/good/track0"},
                    {"recording_id": "rec/val", "sequence_id": "rec/val/track0", "split": "val"},
                    {"recording_id": "rec/test", "sequence_id": "rec/test/track0", "split": "test"},
                    {
                        "recording_id": "rec/not-source-accepted",
                        "sequence_id": "rec/not-source-accepted/track0",
                        "source_accepted": False,
                    },
                    {
                        "recording_id": "rec/not-sequence-accepted",
                        "sequence_id": "rec/not-sequence-accepted/track0",
                        "sequence_accepted": False,
                    },
                ],
            )
            output = root / "not-accepted"
            self._materialize(sources, sequences, labels, output)
            self.assertEqual(np.load(output / "train/labels.npy").shape[0], 1)
            quarantine = [json.loads(line) for line in (output / "quarantine.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertIn("source_not_accepted_for_training", {row["reason_code"] for row in quarantine})
            self.assertIn("sequence_not_accepted_for_training", {row["reason_code"] for row in quarantine})

    def test_refuses_empty_train_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, sequences, labels, _ = self._fixture(
                root,
                [
                    {
                        "recording_id": "rec/pending",
                        "sequence_id": "rec/pending/track0",
                        "sequence_accepted": False,
                    }
                ],
            )
            output = root / "empty-train"
            with self.assertRaisesRegex(MaterializationError, "zero materialized train windows"):
                self._materialize(sources, sequences, labels, output)
            self.assertFalse(output.exists())

    def test_refuses_empty_test_bundle_required_by_current_trainer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, sequences, labels, _ = self._fixture(
                root,
                [
                    {"recording_id": "rec/train-only", "sequence_id": "rec/train-only/track0"},
                    {"recording_id": "rec/val", "sequence_id": "rec/val/track0", "split": "val"},
                ],
            )
            output = root / "empty-test"
            with self.assertRaisesRegex(MaterializationError, "zero materialized test windows"):
                self._materialize(sources, sequences, labels, output)
            self.assertFalse(output.exists())

    def test_refuses_empty_val_bundle_to_keep_model_selection_off_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, sequences, labels, _ = self._fixture(
                root,
                [
                    {"recording_id": "rec/train", "sequence_id": "rec/train/track0"},
                    {"recording_id": "rec/test", "sequence_id": "rec/test/track0", "split": "test"},
                ],
            )
            output = root / "empty-val"
            with self.assertRaisesRegex(MaterializationError, "zero materialized val windows"):
                self._materialize(sources, sequences, labels, output)
            self.assertFalse(output.exists())

    def test_rejects_raw_or_unverifiable_normalizer_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, sequences, labels, _ = self._fixture(
                root,
                [{"recording_id": "rec/raw", "sequence_id": "rec/raw/track0", "normalization_state": "raw"}],
            )
            output = root / "raw-rejected"
            with self.assertRaisesRegex(MaterializationError, "is raw"):
                self._materialize(sources, sequences, labels, output)
            self.assertFalse(output.exists())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, sequences, labels, _ = self._fixture(
                root,
                [{"recording_id": "rec/tampered", "sequence_id": "rec/tampered/track0"}],
            )
            # The manifest's SHA remains frozen while the source artifact is
            # tampered with: copying it into an inference root must fail.
            (root / "frozen_normalizer/normalizer.pt").write_bytes(b"not a normalizer")
            output = root / "tampered"
            with self.assertRaisesRegex(MaterializationError, "artifact hash does not match"):
                self._materialize(sources, sequences, labels, output)
            self.assertFalse(output.exists())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, sequences, labels, _ = self._fixture(
                root,
                [{"recording_id": "rec/missing", "sequence_id": "rec/missing/track0"}],
            )
            row = json.loads(sequences.read_text(encoding="utf-8"))
            row.pop("normalizer_artifact_path")
            row["normalization"].pop("normalizer_artifact")
            _write_jsonl(sequences, [row])
            output = root / "missing"
            with self.assertRaisesRegex(MaterializationError, "normalizer_artifact_path"):
                self._materialize(sources, sequences, labels, output)
            self.assertFalse(output.exists())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, sequences, labels, _ = self._fixture(
                root,
                [{"recording_id": "rec/ambiguous", "sequence_id": "rec/ambiguous/track0"}],
            )
            alternate_report = root / "alternate-report.json"
            alternate_report.write_text((root / "frozen_normalizer/report.json").read_text(encoding="utf-8"), encoding="utf-8")
            row = json.loads(sequences.read_text(encoding="utf-8"))
            row.pop("normalizer_fit_report_path")
            row.pop("normalizer_fit_report_sha256")
            row["normalization"].pop("normalizer_fit_report")
            row["normalization"].pop("normalizer_fit_report_sha256")
            row["normalizer_fit_report_path"] = str(alternate_report)
            _write_jsonl(sequences, [row])
            output = root / "ambiguous"
            with self.assertRaisesRegex(MaterializationError, "is ambiguous"):
                self._materialize(sources, sequences, labels, output)
            self.assertFalse(output.exists())

    def test_materializes_nested_reconciled_wild_schema_without_rewriting_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, sequences, labels, originals = self._fixture(
                root,
                [
                    {
                        "recording_id": "tiktok:example-recording",
                        "sequence_id": "tiktok:example-recording:clip000",
                        "nested_wild_schema": True,
                        "base": 123,
                    },
                    {
                        "recording_id": "tiktok:example-val-recording",
                        "sequence_id": "tiktok:example-val-recording:clip000",
                        "split": "val",
                        "base": 666,
                    },
                    {
                        "recording_id": "tiktok:example-test-recording",
                        "sequence_id": "tiktok:example-test-recording:clip000",
                        "split": "test",
                        "base": 999,
                    },
                ],
            )
            source_before = sources.read_text(encoding="utf-8")
            sequence_before = sequences.read_text(encoding="utf-8")
            source_row = json.loads(source_before.splitlines()[0])
            sequence_row = json.loads(sequence_before.splitlines()[0])
            self.assertNotIn("content_sha256", source_row)
            self.assertEqual(source_row["assets"]["content_sha256"], _sha_bytes("content:tiktok:example-recording"))
            self.assertNotIn("motion_path", sequence_row)
            self.assertEqual(sequence_row["assets"]["motion_151_raw"], str((root / "manifest/assets/motion_0.npy").resolve()))
            self.assertEqual(sequence_row["assets"]["music_35"], str((root / "manifest/assets/music_0.npy").resolve()))
            self.assertTrue(sequence_row["timeline"]["motion_frames_are_contiguous"])
            output = root / "indexed"
            report = self._materialize(sources, sequences, labels, output)
            # Materialization creates only internal canonical views; the raw
            # reconciled manifests remain byte-for-byte unchanged.
            self.assertEqual(sources.read_text(encoding="utf-8"), source_before)
            self.assertEqual(sequences.read_text(encoding="utf-8"), sequence_before)
            self.assertTrue(
                np.array_equal(
                    np.load(output / "train/motion.npy")[0],
                    originals["tiktok:example-recording:clip000"]["model_motion"][:WINDOW_LENGTH],
                )
            )
            self.assertTrue(
                np.array_equal(
                    np.load(output / "train/music.npy")[0],
                    originals["tiktok:example-recording:clip000"]["music"][:WINDOW_LENGTH],
                )
            )
            window = next(
                json.loads(line)
                for line in (output / "windows.jsonl").read_text(encoding="utf-8").splitlines()
                if '"sequence_id":"tiktok:example-recording:clip000"' in line
            )
            self.assertEqual(window["sample_name"], "tiktok:example-recording_slice0")
            self.assertEqual(source_id_from_name(window["sample_name"]), "tiktok:example-recording")
            self.assertEqual(window["motion_representation_id"], "AtomicDance_151D")
            self.assertEqual(window["coordinate_system"], "z_up_world_body_only")
            self.assertEqual(window["normalization_state"], "normalized")
            self.assertEqual(window["normalization_artifact_sha256"], report["normalizer"]["source_artifact_sha256"])
            self.assertEqual(report["representation_contract"]["motion_representation_id"], "AtomicDance_151D")


if __name__ == "__main__":
    unittest.main()
