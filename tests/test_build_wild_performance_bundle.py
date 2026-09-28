import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.build_wild_performance_bundle import (
    MOTION_DIM,
    MUSIC_DIM,
    BundleError,
    build,
)


def _clip(root: Path, name, *, recording, split, frames=12, status="candidate",
          music_frames=None, metadata=None):
    clip = root / name
    clip.mkdir(parents=True, exist_ok=True)
    motion = clip / "atomic_motion_151.npy"
    music = clip / "music_35.npy"
    frame_ids = clip / "frame_ids.npy"
    np.save(motion, np.zeros((frames, MOTION_DIM), dtype=np.float32))
    np.save(music, np.zeros((music_frames or frames, MUSIC_DIM), dtype=np.float32))
    np.save(frame_ids, np.arange(frames, dtype=np.int64))
    assets = {
        "motion_151_raw": str(motion),
        "music_35": str(music),
        "source_video": "/videos/{}.mp4".format(name),
    }
    if metadata is not None:
        meta_path = clip / "metadata.json"
        meta_path.write_text(json.dumps(metadata), encoding="utf-8")
        assets["conversion_metadata"] = str(meta_path)
    return {
        "sequence_id": "wild:{}".format(name),
        "recording_id": recording,
        "legacy_clip_id": name,
        "split": split,
        "duplicate_content_group_id": None,
        "person_track_id": "gvhmr:get_one_track",
        "assets": assets,
        "timeline": {"frame_ids_path": str(frame_ids), "frame_count": frames},
        "qc": {"audio_feature_status": status},
    }


def _manifest(root: Path, records):
    path = root / "sequences_audio.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


class SelectionTests(unittest.TestCase):
    def test_only_audio_candidates_cross_over(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                _clip(root, "a__clip000", recording="rec:a", split="train"),
                _clip(root, "b__clip000", recording="rec:b", split="train",
                      status="quarantine"),
                _clip(root, "c__clip000", recording="rec:c", split="val",
                      status="not_attempted"),
            ]
            report = build(audio_manifest=_manifest(root, records),
                           output_dir=root / "bundle")
            self.assertEqual(report["counts"]["sequences"], 1)
            rows = [json.loads(l) for l in (root / "bundle" / "sequences.jsonl").open()]
            self.assertEqual(rows[0]["sequence_id"], "wild:a__clip000")

    def test_refuses_when_no_candidate_survives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [_clip(root, "a__clip000", recording="rec:a", split="train",
                             status="quarantine")]
            with self.assertRaises(BundleError):
                build(audio_manifest=_manifest(root, records), output_dir=root / "bundle")


class LeakageTests(unittest.TestCase):
    def test_a_recording_split_across_train_and_val_blocks_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                _clip(root, "a__clip000", recording="rec:a", split="train"),
                _clip(root, "a__clip001", recording="rec:a", split="val"),
            ]
            with self.assertRaises(BundleError) as caught:
                build(audio_manifest=_manifest(root, records), output_dir=root / "bundle")
            self.assertIn("straddle", str(caught.exception))
            self.assertFalse((root / "bundle").exists())

    def test_many_clips_of_one_recording_in_one_split_are_fine(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                _clip(root, "a__clip000", recording="rec:a", split="train"),
                _clip(root, "a__clip001", recording="rec:a", split="train"),
                _clip(root, "b__clip000", recording="rec:b", split="val"),
            ]
            report = build(audio_manifest=_manifest(root, records),
                           output_dir=root / "bundle")
            self.assertEqual(report["counts"]["sequences"], 3)
            self.assertEqual(report["counts"]["uploads"], 2)
            self.assertEqual(report["counts"]["clips"], 3)
            self.assertEqual(report["counts"]["by_split"], {"train": 2, "val": 1, "test": 0})
            rows = [json.loads(l) for l in (root / "bundle" / "sequences.jsonl").open()]
            # The exclusion unit must be the upload, and the recording the clip.
            self.assertEqual({r["retrieval_group_id"] for r in rows}, {"rec:a", "rec:b"})
            self.assertEqual(len({r["recording_id"] for r in rows}), 3)


class IntegrityTests(unittest.TestCase):
    def test_music_shorter_than_motion_is_refused_not_padded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [_clip(root, "a__clip000", recording="rec:a", split="train",
                             frames=12, music_frames=8)]
            with self.assertRaises(BundleError):
                build(audio_manifest=_manifest(root, records), output_dir=root / "bundle")

    def test_published_arrays_match_their_recorded_hashes(self):
        import hashlib

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [_clip(root, "a__clip000", recording="rec:a", split="train")]
            build(audio_manifest=_manifest(root, records), output_dir=root / "bundle")
            row = json.loads((root / "bundle" / "sequences.jsonl").read_text().splitlines()[0])
            for path_key, hash_key in (("motion_path", "motion_sha256"),
                                       ("music_path", "music_sha256")):
                blob = (root / "bundle" / row[path_key]).read_bytes()
                self.assertEqual(hashlib.sha256(blob).hexdigest(), row[hash_key])

    def test_tracker_identity_is_carried_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = {"extract_meta": {"visual_odometry": "DPVO", "static_cam": False,
                                         "visual_odometry_attempts_allowed": 3}}
            records = [_clip(root, "a__clip000", recording="rec:a", split="train",
                             metadata=metadata)]
            build(audio_manifest=_manifest(root, records), output_dir=root / "bundle")
            row = json.loads((root / "bundle" / "sequences.jsonl").read_text().splitlines()[0])
            self.assertEqual(row["camera_tracking"]["visual_odometry"], "DPVO")
            self.assertEqual(row["camera_tracking"]["visual_odometry_attempts_allowed"], 3)

    def test_refuses_to_overwrite_and_leaves_no_staging_behind(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [_clip(root, "a__clip000", recording="rec:a", split="train")]
            manifest = _manifest(root, records)
            out = root / "bundle"
            build(audio_manifest=manifest, output_dir=out)
            with self.assertRaises(BundleError):
                build(audio_manifest=manifest, output_dir=out)
            self.assertFalse(out.with_name(out.name + ".staging").exists())



class ContentIdentityTests(unittest.TestCase):
    def test_source_rows_carry_a_content_hash_over_the_motion_music_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [_clip(root, "a__clip000", recording="rec:a", split="train")]
            build(audio_manifest=_manifest(root, records), output_dir=root / "bundle")
            source = json.loads(
                (root / "bundle" / "sources.jsonl").read_text().splitlines()[0])
            self.assertRegex(source["content_sha256"], r"^[0-9a-f]{64}$")
            # It must be a function of both assets, not just one of them.
            from tools.build_wild_performance_bundle import _content_sha256

            self.assertEqual(
                source["content_sha256"],
                _content_sha256(source["motion_sha256"], source["music_sha256"]))
            self.assertNotEqual(
                source["content_sha256"],
                _content_sha256(source["music_sha256"], source["motion_sha256"]))


class DeclaredTailTrimTests(unittest.TestCase):
    """The bundle's length gate is the last place an off-by-k can be caught.

    ``extract_wild_music_features --max-tail-trim-frames`` cuts a clip's motion
    back to where its audio ends rather than dropping the clip over one or two
    rounding frames, and declares how many.  Honouring the declaration must not
    weaken the gate: an undeclared mismatch, or one that does not match what is
    declared, still has to be fatal.
    """

    def _trimmed(self, root, name, *, trim, motion_frames=12):
        record = _clip(root, name, recording="rec:" + name, split="train",
                       frames=motion_frames, music_frames=motion_frames - trim)
        record["audio_feature"] = {"status": "candidate", "tail_trim_frames": trim}
        return record

    def test_declared_trim_shortens_motion_and_frame_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = self._trimmed(root, "a__clip000", trim=2, motion_frames=12)
            report = build(audio_manifest=_manifest(root, [record]),
                           output_dir=root / "bundle")
            self.assertEqual(report["counts"]["sequences"], 1)
            row = json.loads((root / "bundle" / "sequences.jsonl").read_text().strip())
            self.assertEqual(row["frame_count"], 10)
            motion = np.load(root / "bundle" / row["motion_path"])
            music = np.load(root / "bundle" / row["music_path"])
            self.assertEqual(len(motion), 10)
            self.assertEqual(len(music), 10)

    def test_undeclared_mismatch_is_still_fatal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = _clip(root, "a__clip000", recording="rec:a", split="train",
                           frames=12, music_frames=10)
            with self.assertRaises(BundleError):
                build(audio_manifest=_manifest(root, [record]),
                      output_dir=root / "bundle")

    def test_declaration_that_does_not_match_the_arrays_is_fatal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = self._trimmed(root, "a__clip000", trim=2, motion_frames=12)
            record["audio_feature"]["tail_trim_frames"] = 1   # arrays say 2
            with self.assertRaises(BundleError):
                build(audio_manifest=_manifest(root, [record]),
                      output_dir=root / "bundle")


if __name__ == "__main__":
    unittest.main()
