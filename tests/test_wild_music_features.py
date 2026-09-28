import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.extract_wild_music_features import (
    FEATURE_DIM,
    materialize_wild_music_features,
    publish_wild_music_bundle,
)


class WildMusicFeatureTests(unittest.TestCase):
    def _candidate(self, root: Path, *, frame_ids, frames=None, fps=30.0):
        cache = root / "cache"
        cache.mkdir()
        (cache / "audio.wav").write_bytes(b"synthetic-audio-for-injected-extractor")
        frame_ids = np.asarray(frame_ids, dtype=np.int64)
        frame_ids_path = root / "frame_ids.npy"
        np.save(frame_ids_path, frame_ids)
        motion_path = root / "atomic_motion_151.npy"
        np.save(motion_path, np.zeros((len(frame_ids), 151), dtype=np.float32))
        return {
            "schema_version": "atomic-sequence-v1",
            "sequence_id": "tiktok:example:clip001",
            "recording_id": "tiktok:example",
            "stage": "post_hmr_reconciled",
            "timeline": {
                "fps": fps,
                "frame_count": len(frame_ids) if frames is None else frames,
                "frame_ids_path": str(frame_ids_path),
                "motion_frames_are_contiguous": True,
            },
            "assets": {"source_cache": str(cache), "motion_151_raw": str(motion_path)},
            "qc": {"hmr_status": "candidate", "accepted_for_training": True, "reason_codes": []},
        }

    def test_candidate_selects_exact_source_frame_rows_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = self._candidate(root, frame_ids=[2, 3, 4])
            full = np.arange(10 * FEATURE_DIM, dtype=np.float32).reshape(10, FEATURE_DIM)
            seen = []

            def extractor(path):
                seen.append(path)
                return full

            artifact_root = root / "staging" / "music_35"
            public_root = root / "published" / "music_35"
            records, summary = materialize_wild_music_features(
                [record],
                artifact_root=artifact_root,
                public_artifact_root=public_root,
                input_manifest_sha256="a" * 64,
                feature_extractor=extractor,
            )
            self.assertEqual(summary["status_counts"], {"not_attempted": 0, "candidate": 1, "quarantine": 0})
            self.assertEqual(seen, [root / "cache" / "audio.wav"])
            result = records[0]
            self.assertEqual(result["qc"]["hmr_status"], "candidate")
            self.assertEqual(result["qc"]["audio_feature_status"], "candidate")
            self.assertFalse(result["qc"]["accepted_for_training"])
            self.assertTrue(record["qc"]["accepted_for_training"])
            self.assertNotIn("audio_feature_status", record["qc"])
            self.assertEqual(result["audio_feature"]["frame_selection"]["source_start_frame"], 2)
            self.assertEqual(result["audio_feature"]["frame_selection"]["source_end_frame_exclusive"], 5)
            written = next(artifact_root.glob("*.npy"))
            self.assertTrue(np.array_equal(np.load(written), full[2:5]))
            self.assertEqual(
                result["assets"]["music_35_sha256"],
                hashlib.sha256(written.read_bytes()).hexdigest(),
            )
            self.assertTrue(result["assets"]["music_35"].startswith(str(public_root)))

    def test_insufficient_audio_quarantines_without_padding_or_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = self._candidate(root, frame_ids=[8, 9, 10])
            records, summary = materialize_wild_music_features(
                [record],
                artifact_root=root / "staging" / "music_35",
                public_artifact_root=root / "published" / "music_35",
                input_manifest_sha256="b" * 64,
                feature_extractor=lambda _: np.zeros((10, FEATURE_DIM), dtype=np.float32),
            )
            self.assertEqual(summary["status_counts"], {"not_attempted": 0, "candidate": 0, "quarantine": 1})
            result = records[0]
            self.assertEqual(result["qc"]["audio_feature_status"], "quarantine")
            self.assertIn("audio_feature_insufficient_audio_frames", result["qc"]["reason_codes"])
            self.assertEqual(result["assets"]["music_35"], None)
            self.assertFalse(list((root / "staging" / "music_35").glob("*.npy")))

    def test_non_candidate_never_invokes_extractor_and_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = self._candidate(root, frame_ids=[0, 1, 2])
            record["qc"]["hmr_status"] = "pending"
            records, summary = materialize_wild_music_features(
                [record],
                artifact_root=root / "staging" / "music_35",
                public_artifact_root=root / "published" / "music_35",
                input_manifest_sha256="c" * 64,
                feature_extractor=lambda _: self.fail("pending record tried to extract audio"),
            )
            self.assertEqual(summary["status_counts"], {"not_attempted": 1, "candidate": 0, "quarantine": 0})
            self.assertEqual(records[0], record)

    def test_publish_is_immutable_bundle_with_full_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = self._candidate(root, frame_ids=[0, 1, 2])
            input_path = root / "sequences_hmr.jsonl"
            input_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            output_dir = root / "audio_35_v1"
            summary = publish_wild_music_bundle(
                input_path,
                output_dir,
                feature_extractor=lambda _: np.ones((3, FEATURE_DIM), dtype=np.float32),
            )
            manifest = output_dir / "sequences_audio.jsonl"
            self.assertTrue(manifest.is_file())
            self.assertTrue((output_dir / "summary.json").is_file())
            self.assertEqual(summary["output_manifest_sha256"], hashlib.sha256(manifest.read_bytes()).hexdigest())
            published = json.loads(manifest.read_text(encoding="utf-8"))
            music_path = Path(published["assets"]["music_35"])
            self.assertTrue(music_path.is_file())
            with self.assertRaises(FileExistsError):
                publish_wild_music_bundle(input_path, output_dir, feature_extractor=lambda _: np.zeros((3, FEATURE_DIM)))



class TailTrimTests(unittest.TestCase):
    """A clip whose audio ends a frame or two early is rounding, not silence.

    The positive case has to buy back an otherwise-lost clip; the negative
    control has to prove the budget is a cap and not a policy, because a budget
    that quietly grows to fit is the same instrument as no budget at all.
    """

    # Borrowed rather than inherited: subclassing WildMusicFeatureTests would
    # re-run its four cases under this name as well.
    _candidate = WildMusicFeatureTests._candidate

    def _run(self, root, frame_ids, audio_rows, *, budget):
        record = self._candidate(root, frame_ids=frame_ids)
        full = np.arange(audio_rows * FEATURE_DIM, dtype=np.float32).reshape(audio_rows, FEATURE_DIM)
        return materialize_wild_music_features(
            [record],
            artifact_root=root / "staging" / "music_35",
            public_artifact_root=root / "published" / "music_35",
            input_manifest_sha256="c" * 64,
            feature_extractor=lambda _: full,
            max_tail_trim_frames=budget,
        ), full

    def test_shortfall_within_budget_is_trimmed_not_quarantined(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # frame_ids [8,9,10] against 10 rows of audio: row 10 does not exist,
            # so the shortfall is exactly 1 -- the corpus's most common case.
            (records, summary), full = self._run(root, [8, 9, 10], 10, budget=1)
            self.assertEqual(summary["status_counts"]["candidate"], 1)
            self.assertEqual(summary["status_counts"]["quarantine"], 0)
            self.assertEqual(summary["status_counts"]["tail_trimmed"], 1)
            result = records[0]
            self.assertEqual(result["qc"]["audio_feature_status"], "candidate")
            self.assertIn("audio_feature_tail_trimmed", result["qc"]["reason_codes"])
            feature = result["audio_feature"]
            self.assertEqual(feature["tail_trim_frames"], 1)
            self.assertEqual(feature["frame_selection"]["frame_count"], 2)
            self.assertEqual(feature["frame_selection"]["manifest_frame_count"], 3)
            self.assertEqual(feature["frame_selection"]["source_end_frame_exclusive"], 10)
            written = next((root / "staging" / "music_35").glob("*.npy"))
            # The audio is the audio: rows 8 and 9, never an invented row 10.
            self.assertTrue(np.array_equal(np.load(written), full[8:10]))

    def test_shortfall_beyond_budget_is_still_quarantined(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Shortfall 3 against a budget of 1.  This is the negative control:
            # the wild corpus has rows short by 100-355 frames, and those must
            # not become trainable just because a trim exists.
            (records, summary), _ = self._run(root, [8, 9, 10, 11, 12], 10, budget=1)
            self.assertEqual(summary["status_counts"]["quarantine"], 1)
            self.assertEqual(summary["status_counts"]["candidate"], 0)
            result = records[0]
            self.assertEqual(result["audio_feature"]["failure_code"], "insufficient_audio_frames")
            self.assertFalse(list((root / "staging" / "music_35").glob("*.npy")))

    def test_default_budget_is_zero_so_behaviour_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (records, summary), _ = self._run(root, [8, 9, 10], 10, budget=0)
            self.assertEqual(summary["status_counts"]["quarantine"], 1)
            self.assertEqual(summary["feature_contract"]["max_tail_trim_frames"], 0)

    def test_trim_may_not_consume_the_whole_clip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # One frame of motion, no audio at all: trimming it away would
            # publish an empty sequence, so this stays a quarantine.
            (records, summary), _ = self._run(root, [0], 0, budget=4)
            self.assertEqual(summary["status_counts"]["quarantine"], 1)


if __name__ == "__main__":
    unittest.main()


class PreflightTests(unittest.TestCase):
    """An unavailable extractor must stop the run, not be written into records.

    The failure this guards: invoked as ``python3 tools/...``, sys.path[0] is
    ``tools/`` and the repo root is absent, so ``data.audio_extraction`` does
    not import.  Caught per sequence, that published 290 identical
    ``extraction_failed`` records and a summary reading ``candidate: 0`` --
    an environment fault dressed as a verdict about the corpus.
    """

    def test_preflight_passes_when_the_extractor_imports(self):
        from tools.extract_wild_music_features import preflight_extractor
        preflight_extractor()          # must not raise in a correct environment

    def test_preflight_raises_rather_than_returning_when_the_import_fails(self):
        import builtins
        from tools.extract_wild_music_features import preflight_extractor
        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "data.audio_extraction" or name.startswith("data.audio_extraction"):
                raise ImportError("blocked for the test")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = blocked
        try:
            with self.assertRaises(SystemExit) as caught:
                preflight_extractor()
        finally:
            builtins.__import__ = real_import
        self.assertIn("environment fault", str(caught.exception))

    def test_the_module_puts_the_repo_root_on_sys_path(self):
        """The one-line fix itself: without it the preflight above would be the
        thing that fires, on every invocation from outside the repo root."""
        import pathlib as _pathlib
        import sys as _sys
        import tools.extract_wild_music_features as module
        root = str(_pathlib.Path(module.__file__).resolve().parents[1])
        self.assertIn(root, _sys.path)
