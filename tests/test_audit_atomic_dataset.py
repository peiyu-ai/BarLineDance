import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.audit_atomic_dataset import (LEGACY_NUM_CLASSES, audit_dataset,
                                        audit_source_safe_retrieval, audit_split,
                                        declared_num_classes, song_id)


class SourceSafeRetrievalAuditTests(unittest.TestCase):
    def _write_split(self, root, split_name, groups=None):
        split = Path(root) / split_name
        split.mkdir()
        np.save(
            split / "motion.npy",
            np.zeros((3, 4, 151), dtype=np.float32),
            allow_pickle=False,
        )
        np.save(
            split / "music.npy",
            np.zeros((3, 4, 35), dtype=np.float32),
            allow_pickle=False,
        )
        np.save(
            split / "labels.npy",
            np.ones((3, 4), dtype=np.int64),
            allow_pickle=False,
        )
        (split / "names.json").write_text(
            json.dumps(
                [
                    "dance42_front_slice0",
                    "dance42_side_slice0",
                    "dance43_front_slice0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        if groups is not None:
            (split / "retrieval_groups.json").write_text(
                json.dumps(groups) + "\n", encoding="utf-8"
            )

    def test_uses_explicit_performance_groups_not_camera_window_names(self):
        with tempfile.TemporaryDirectory() as directory:
            # Three distinct recording names are insufficient evidence of an
            # external prototype when they all belong to one performance.
            self._write_split(directory, "train", ["performance/dance42"] * 3)
            report = audit_source_safe_retrieval(Path(directory))
            self.assertTrue(report["checked"])
            self.assertEqual(report["source_safe_atomic_frame_fraction"], 0.0)
            self.assertEqual(report["label_retrieval_group_counts"], {"1": 1})
            self.assertEqual(report["labels_with_only_one_retrieval_group"], ["1"])

        with tempfile.TemporaryDirectory() as directory:
            self._write_split(
                directory,
                "train",
                ["performance/dance42", "performance/dance42", "performance/dance43"],
            )
            report = audit_source_safe_retrieval(Path(directory))
            self.assertTrue(report["checked"])
            self.assertEqual(report["source_safe_atomic_frame_fraction"], 1.0)
            self.assertEqual(report["label_retrieval_group_counts"], {"1": 2})

    def test_missing_group_sidecar_is_unverifiable_not_name_derived(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_split(directory, "train")
            report = audit_source_safe_retrieval(Path(directory))
            self.assertFalse(report["checked"])
            self.assertIn("missing explicit retrieval_groups.json", report["reason"])

    def test_dataset_audit_fails_when_group_coverage_cannot_be_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_split(directory, "train")
            self._write_split(directory, "test", ["performance/test"] * 3)
            report = audit_dataset(
                Path(directory), eval_source_list=None, window_stride=None
            )
            self.assertFalse(report["valid"])
            self.assertTrue(
                any("source-safe retrieval audit is unverifiable" in error for error in report["errors"])
            )


class SourceSafeRetrievalNullTests(unittest.TestCase):
    """The ceiling the class-size distribution allows, in both regimes.

    Without it the 0.99 bound cannot be read: coverage falls as a vocabulary
    grows finer, so a fixed bound quietly becomes a statement about the class
    count.  The null has to be able to say *both* "this shortfall is only
    granularity" and "this shortfall is real concentration", or it is decoration.
    """

    def _split(self, root, labels, groups):
        split = Path(root) / "train"
        split.mkdir()
        rows, width = labels.shape
        np.save(split / "motion.npy", np.zeros((rows, width, 151), dtype=np.float32),
                allow_pickle=False)
        np.save(split / "music.npy", np.zeros((rows, width, 35), dtype=np.float32),
                allow_pickle=False)
        np.save(split / "labels.npy", labels.astype(np.int64), allow_pickle=False)
        (split / "names.json").write_text(
            json.dumps(["clip{}_slice0".format(index) for index in range(rows)]) + "\n",
            encoding="utf-8")
        (split / "retrieval_groups.json").write_text(
            json.dumps(groups) + "\n", encoding="utf-8")

    def test_a_class_that_cannot_recur_puts_the_ceiling_at_the_observed_value(self):
        # One window holds a class that occurs nowhere else at all.  No
        # permutation can move it into a second performance, so the null equals
        # the observation and the shortfall is granularity, not concentration.
        labels = np.array([[1, 1], [1, 1], [2, 2]], dtype=np.int64)
        groups = ["p/a", "p/b", "p/c"]
        with tempfile.TemporaryDirectory() as directory:
            self._split(directory, labels, groups)
            report = audit_source_safe_retrieval(Path(directory), null_permutations=64)
            self.assertAlmostEqual(report["source_safe_atomic_frame_fraction"], 4 / 6)
            self.assertAlmostEqual(report["null"]["null_mean"], 4 / 6)
            self.assertAlmostEqual(report["null"]["observed_over_null_mean"], 1.0)

    def test_a_class_confined_to_one_performance_leaves_the_ceiling_above_it(self):
        # Class 2 occurs in two windows that happen to share a performance.
        # Permuting performance membership separates them most of the time, so
        # the null sits well above the observation: real concentration.
        labels = np.array([[1, 1], [1, 1], [2, 2], [2, 2]], dtype=np.int64)
        groups = ["p/a", "p/b", "p/c", "p/c"]
        with tempfile.TemporaryDirectory() as directory:
            self._split(directory, labels, groups)
            report = audit_source_safe_retrieval(Path(directory), null_permutations=200)
            self.assertAlmostEqual(report["source_safe_atomic_frame_fraction"], 0.5)
            self.assertGreater(report["null"]["null_mean"], 0.8)
            self.assertLess(report["null"]["observed_over_null_mean"], 0.7)

    def test_the_null_is_skipped_rather_than_faked_when_not_asked_for(self):
        labels = np.array([[1, 1], [1, 1]], dtype=np.int64)
        with tempfile.TemporaryDirectory() as directory:
            self._split(directory, labels, ["p/a", "p/b"])
            report = audit_source_safe_retrieval(Path(directory))
            self.assertNotIn("null", report)


class SongDisjointnessAuditTests(unittest.TestCase):
    """A split can be performance-disjoint and still share every backing track.

    That is exactly what aist_kinematic_release_v1 does: all 16 val songs are
    also train songs.  A model can then recognise the track instead of
    responding to it, so val cannot support a music-conditioned generalization
    claim even though its performances are held out.
    """

    def _write_split(self, root, split_name, names):
        split = Path(root) / split_name
        split.mkdir()
        count = len(names)
        np.save(split / "motion.npy", np.zeros((count, 4, 151), dtype=np.float32), allow_pickle=False)
        np.save(split / "music.npy", np.zeros((count, 4, 35), dtype=np.float32), allow_pickle=False)
        np.save(split / "labels.npy", np.ones((count, 4), dtype=np.int64), allow_pickle=False)
        (split / "names.json").write_text(json.dumps(names) + "\n", encoding="utf-8")
        (split / "retrieval_groups.json").write_text(
            json.dumps(["group/{}".format(name) for name in names]) + "\n", encoding="utf-8"
        )
        (Path(root) / "normalizer.pt").touch()

    def _audit(self, directory, **kwargs):
        return audit_dataset(Path(directory), window_stride=None, **kwargs)

    def test_song_id_extraction(self):
        self.assertEqual(song_id("gBR_sBM_cAll_d04_mBR1_ch03_slice0"), "mBR1")
        self.assertEqual(song_id("aistpp/gHO_sBM_cAll_d20_mHO5_ch02_slice2"), "mHO5")
        # An unrecognised scheme must resolve to None rather than to a guess.
        self.assertIsNone(song_id("tiktok_6933985164326489359__clip001"))

    def test_shared_song_between_train_and_val_is_a_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_split(directory, "train", ["gBR_sBM_cAll_d04_mBR1_ch01_slice0"])
            self._write_split(directory, "val", ["gBR_sBM_cAll_d06_mBR1_ch09_slice0"])
            self._write_split(directory, "test", ["gPO_sBM_cAll_d10_mPO2_ch03_slice0"])
            report = self._audit(directory)

            pairs = report["song_disjointness_audit"]["pairs"]
            self.assertEqual(pairs["train_val"]["shared_songs"], 1)
            self.assertEqual(pairs["train_test"]["shared_songs"], 0)
            # A shared val song must not fail the release outright...
            self.assertTrue(any("train/val share" in item for item in report["warnings"]))
            self.assertFalse(any("train/val share" in item for item in report["errors"]))

    def test_shared_song_with_val_can_be_promoted_to_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_split(directory, "train", ["gBR_sBM_cAll_d04_mBR1_ch01_slice0"])
            self._write_split(directory, "val", ["gBR_sBM_cAll_d06_mBR1_ch09_slice0"])
            self._write_split(directory, "test", ["gPO_sBM_cAll_d10_mPO2_ch03_slice0"])
            report = self._audit(directory, require_song_disjoint_splits=True)
            self.assertTrue(any("train/val share" in item for item in report["errors"]))
            self.assertFalse(report["valid"])

    def test_shared_song_with_test_is_always_an_error(self):
        """test backs the benchmark, so a shared track invalidates it outright."""
        with tempfile.TemporaryDirectory() as directory:
            self._write_split(directory, "train", ["gBR_sBM_cAll_d04_mBR1_ch01_slice0"])
            self._write_split(directory, "test", ["gBR_sBM_cAll_d10_mBR1_ch03_slice0"])
            report = self._audit(directory)
            self.assertTrue(any("train/test share" in item for item in report["errors"]))
            self.assertFalse(report["valid"])

    def test_val_split_is_audited(self):
        """val was previously skipped entirely, yet it selects the model."""
        with tempfile.TemporaryDirectory() as directory:
            self._write_split(directory, "train", ["gBR_sBM_cAll_d04_mBR1_ch01_slice0"])
            self._write_split(directory, "val", ["gJS_sBM_cAll_d06_mJS3_ch09_slice0"])
            self._write_split(directory, "test", ["gPO_sBM_cAll_d10_mPO2_ch03_slice0"])
            report = self._audit(directory)
            self.assertEqual(report["splits"]["val"]["split"], "val")
            self.assertIn("train_val", report["cross_split_name_overlap_by_pair"])
            self.assertIn("val_test", report["cross_split_source_overlap_by_pair"])

    def test_absent_val_is_a_warning_not_an_error(self):
        """The upstream package ships train/test only; that is not corruption."""
        with tempfile.TemporaryDirectory() as directory:
            self._write_split(directory, "train", ["gBR_sBM_cAll_d04_mBR1_ch01_slice0"])
            self._write_split(directory, "test", ["gPO_sBM_cAll_d10_mPO2_ch03_slice0"])
            report = self._audit(directory)
            self.assertFalse(report["splits"]["val"]["present"])
            self.assertTrue(any("val split is absent" in item for item in report["warnings"]))
            # Match the split prefix, not a bare "val" -- that also appears
            # inside words like "retrieval".
            self.assertFalse(any(item.startswith("val:") for item in report["errors"]))

    def test_legacy_train_test_keys_are_preserved(self):
        """Downstream reports read these by name; adding val must not drop them."""
        with tempfile.TemporaryDirectory() as directory:
            self._write_split(directory, "train", ["gBR_sBM_cAll_d04_mBR1_ch01_slice0"])
            self._write_split(directory, "val", ["gJS_sBM_cAll_d06_mJS3_ch09_slice0"])
            self._write_split(directory, "test", ["gPO_sBM_cAll_d10_mPO2_ch03_slice0"])
            report = self._audit(directory)
            self.assertEqual(report["cross_split_name_overlap"], 0)
            self.assertEqual(report["cross_split_source_overlap"], 0)

    def test_unresolvable_names_show_up_as_low_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_split(directory, "train", ["tiktok_123__clip001_slice0"])
            self._write_split(directory, "test", ["tiktok_456__clip002_slice0"])
            coverage = self._audit(directory)["song_disjointness_audit"]["coverage"]
            self.assertEqual(coverage["train"]["fraction"], 0.0)
            self.assertEqual(coverage["train"]["unique_songs"], 0)


class DeclaredVocabularyTests(unittest.TestCase):
    """The label bound comes from the release, not from a constant.

    It used to be a literal 100 -- the paper's prototype count -- so the first
    release built on a re-clustered vocabulary failed on its labels alone: the
    AIST++ M3 bundle holds 846 sub-prototypes and was reported as corrupt.  A
    bound that rejects a correct artifact is a wrong check, not a strict one.
    """

    def _root(self, tmp, policy=None):
        root = Path(tmp)
        if policy is not None:
            (root / "build.json").write_text(
                json.dumps({"window_policy": policy}) + "\n", encoding="utf-8")
        return root

    def test_the_bound_is_read_from_build_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp, {"num_classes": 847})
            self.assertEqual(declared_num_classes(root), 847)

    def test_a_legacy_root_keeps_the_upstream_bound(self):
        # Nothing in it declares a vocabulary, and 100 prototypes plus the
        # transition token is what those windows were built with.
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(declared_num_classes(self._root(tmp)), LEGACY_NUM_CLASSES)

    def test_an_explicit_override_wins_over_the_declaration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp, {"num_classes": 847})
            self.assertEqual(declared_num_classes(root, 12), 12)

    def test_an_unreadable_declaration_falls_back_rather_than_raising(self):
        # A corrupt build.json is a separate error with its own report line;
        # the label check must not be the thing that crashes on it.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "build.json").write_text("{not json", encoding="utf-8")
            self.assertEqual(declared_num_classes(root), LEGACY_NUM_CLASSES)

    def test_a_label_at_the_bound_is_out_of_range(self):
        # Labels are 0..num_classes-1: the release declares 847 for a 846-class
        # vocabulary because the transition token is class 0.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split = root / "train"
            split.mkdir()
            np.save(split / "motion.npy", np.zeros((1, 4, 151), dtype=np.float32))
            np.save(split / "music.npy", np.zeros((1, 4, 35), dtype=np.float32))
            np.save(split / "labels.npy", np.full((1, 4), 847, dtype=np.int64))
            (split / "names.json").write_text(json.dumps(["a_slice0"]), encoding="utf-8")
            result = audit_split(root, "train", num_classes=847)
            self.assertTrue(any("labels must be in [0,847)" in e for e in result["errors"]))
