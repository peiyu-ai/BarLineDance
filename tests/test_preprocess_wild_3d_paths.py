"""A manifest must not name the cache mount it happened to be read through.

Large trees live in OSS; ``tools/oss_assets.py pull --cache`` drops them under
``/cache/atomicdance-assets/<repo path>`` and leaves a symlink at the repo path,
so ``data/wild3d/ingest_v1_converted`` is a link and ``evict`` is allowed to
empty what it points at.  ``Path.resolve()`` follows that link, so a manifest
built with it records ``/cache/...`` for every clip -- a location that stops
existing the moment the cache is reclaimed, while the repo path keeps working
because the next ``pull`` re-creates the link in place.  That is how the v4
staging manifest died, and ``tools/run_wild_rebuild.sh`` stage C now refuses a
manifest holding ``/cache`` paths outright.

These tests hold the writing side to the same rule: normalise the path, do not
resolve it, and do not make it absolute either -- the repo-relative spelling is
also the OSS key, and ``run_wild_stage_c_oss.publish_rows`` refuses to publish a
row containing an absolute path at all.
"""

import json
import os
import pathlib
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools import preprocess_wild_3d


def _converted(directory, frames=8):
    """The minimum a reconcile touches once validation has said yes.

    The strict validator wants a full GVHMR audit trail (camera gauge, run
    provenance, DPVO trajectories); fabricating one here would test the
    validator, not the paths, so these tests stub it out and lay down only the
    files ``reconcile_wild_hmr_sequences`` itself opens afterwards.
    """
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(str(directory / "atomic_motion_151.npy"),
            np.zeros((frames, preprocess_wild_3d.ATOMIC_MOTION_DIM), dtype=np.float32))
    np.save(str(directory / "frame_ids.npy"), np.arange(frames, dtype=np.int64))
    (directory / "metadata.json").write_text(
        json.dumps({"fps": 30.0, "backend": "GVHMR"}), encoding="utf-8")
    (directory / "quality.json").write_text(json.dumps({}), encoding="utf-8")
    np.savez(str(directory / "camera.npz"))
    return directory


class StablePath(unittest.TestCase):
    def test_a_symlink_is_not_followed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            (root / "store").mkdir()
            (root / "link").symlink_to(root / "store")
            stable = preprocess_wild_3d._stable_path(root / "link" / "x.npy")
            self.assertEqual(stable, root / "link" / "x.npy")
            self.assertNotIn("store", str(stable))

    def test_a_relative_path_stays_relative(self):
        stable = preprocess_wild_3d._stable_path(pathlib.Path("data/wild3d/x"))
        self.assertFalse(stable.is_absolute())
        self.assertEqual(str(stable), "data/wild3d/x")

    def test_dot_dot_is_still_normalised(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            stable = preprocess_wild_3d._stable_path(root / "a" / ".." / "b")
            self.assertEqual(stable, root / "b")


class ReconcilePaths(unittest.TestCase):
    def setUp(self):
        self._real_validate = preprocess_wild_3d.validate_converted_output
        preprocess_wild_3d.validate_converted_output = lambda root: {
            "output_dir": str(root), "frames": 8, "valid": True, "errors": []}

    def tearDown(self):
        preprocess_wild_3d.validate_converted_output = self._real_validate

    def _reconcile(self, converted_root):
        staging = [{
            "sequence_id": "seq0",
            "legacy_clip_id": "u__clip000",
            "assets": {},
            "timeline": {},
            "representation": {},
            "qc": {},
        }]
        records, _ = preprocess_wild_3d.reconcile_wild_hmr_sequences(
            staging, converted_root=pathlib.Path(converted_root))
        return records[0]

    def test_manifest_keeps_the_root_it_was_given(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            _converted(root / "store" / "u__clip000")
            (root / "link").symlink_to(root / "store")
            record = self._reconcile(root / "link")
            self.assertEqual(record["qc"]["hmr_status"], "candidate")
            paths = [record["conversion_output_dir"],
                     record["timeline"]["frame_ids_path"],
                     record["assets"]["motion_151_raw"],
                     record["assets"]["camera"],
                     record["assets"]["conversion_metadata"],
                     record["assets"]["conversion_quality"]]
            for path in paths:
                # The link, not what it points at: the payload only has to be
                # reachable, and the repo path survives an evict-then-pull.
                self.assertTrue(path.startswith(str(root / "link")), path)
            self.assertTrue(os.path.isfile(record["assets"]["motion_151_raw"]))

    def test_a_relative_root_is_recorded_as_given(self):
        # The driver passes ``data/wild3d/ingest_v1_converted``, and that
        # spelling is also the OSS key the bytes live under, so it is what a
        # publishable manifest has to carry.
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            _converted(root / "converted" / "u__clip000")
            previous = os.getcwd()
            os.chdir(raw)
            try:
                record = self._reconcile("converted")
            finally:
                os.chdir(previous)
            self.assertEqual(record["conversion_output_dir"], "converted/u__clip000")
            self.assertEqual(record["assets"]["motion_151_raw"],
                             "converted/u__clip000/atomic_motion_151.npy")


class InventoryPaths(unittest.TestCase):
    def test_cache_dir_and_pose_paths_keep_the_link(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            clip = root / "store" / "u__clip000"
            clip.mkdir(parents=True)
            (clip / "meta.json").write_text(json.dumps({"fps": 30.0, "source": "u.mp4"}),
                                            encoding="utf-8")
            np.save(str(clip / "keypoints.npy"), np.zeros((8, 17, 2), dtype=np.float32))
            np.save(str(clip / "scores.npy"), np.ones((8, 17), dtype=np.float32))
            (root / "link").symlink_to(root / "store")
            records = preprocess_wild_3d.inventory_wild_cache(
                root / "link", recursive=False, min_frames=1, min_visible_fraction=0.0,
                min_score=0.0, max_frozen_fraction=1.0)
            self.assertEqual(len(records), 1)
            for key in ("cache_dir", "pose2d_path", "scores_path"):
                self.assertTrue(records[0][key].startswith(str(root / "link")),
                                (key, records[0][key]))


class CorpusSourceVideo(unittest.TestCase):
    """A re-cut clip records the scratch file it was cut from, which will go away.

    Two families exist on this corpus (2026-08-25): 2,185 clips name
    ``scratch/c1/refix/uploads/<id>.mp4`` and 278 name
    ``scratch/c1/cfr_uploads/<id>.mp4`` -- the CFR re-encodes made so ``cut_clip``
    would name the same span in picture and in sound.  Both are working space.
    The upload they came from is still in the corpus under its own id, and the
    rewrite was checked rather than assumed: all 278 CFR clips' recorded
    ``cfr_normalized_from`` de-caches to exactly the name-derived upload, and on
    40 sampled clips of the older family the upload's ``r_frame_rate`` equals
    the ``source_fps`` their meta recorded.
    """

    def _clip(self, root, name, meta):
        clip = pathlib.Path(root) / name
        clip.mkdir(parents=True)
        (clip / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        np.save(str(clip / "keypoints.npy"), np.zeros((8, 17, 2), dtype=np.float32))
        np.save(str(clip / "scores.npy"), np.ones((8, 17), dtype=np.float32))
        return clip

    def _inventory(self, cache_root, upload_root=None):
        return preprocess_wild_3d.inventory_wild_cache(
            pathlib.Path(cache_root), recursive=False, min_frames=1,
            min_visible_fraction=0.0, min_score=0.0, max_frozen_fraction=1.0,
            upload_root=pathlib.Path(upload_root) if upload_root else None)

    def test_an_untouched_clip_keeps_the_source_it_recorded(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            cache = root / "cache"
            self._clip(cache, "u__clip000", {"fps": 30.0, "source": "videos/u.mp4"})
            record = self._inventory(cache, upload_root=root / "videos")[0]
            self.assertEqual(record["source_video"], "videos/u.mp4")
            self.assertNotIn("source_video_working_copy", record)

    def test_a_cfr_reencode_is_rewritten_to_what_it_was_made_from(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            previous = os.getcwd()
            os.chdir(raw)
            try:
                (root / "data" / "uploads").mkdir(parents=True)
                (root / "data" / "uploads" / "u.mp4").write_bytes(b"\x00")
                cache = root / "cache"
                self._clip(cache, "u__clip000", {
                    "fps": 30.0,
                    "source": "/cache/atomicdance-assets/scratch/c1/cfr_uploads/u.mp4",
                    "cfr_normalized_from": "/cache/atomicdance-assets/data/uploads/u.mp4",
                })
                record = self._inventory(cache, upload_root="data/uploads")[0]
            finally:
                os.chdir(previous)
            self.assertEqual(record["source_video"], "data/uploads/u.mp4")
            # The cut a clip came out of is provenance, so it is kept, not dropped.
            self.assertEqual(record["source_video_working_copy"],
                             "/cache/atomicdance-assets/scratch/c1/cfr_uploads/u.mp4")

    def test_the_older_family_falls_back_to_the_upload_root(self):
        # These metas predate ``cfr_normalized_from``, so the only link back to
        # the upload is the clip name's prefix.
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            previous = os.getcwd()
            os.chdir(raw)
            try:
                (root / "videos").mkdir()
                (root / "videos" / "u.mp4").write_bytes(b"\x00")
                cache = root / "cache"
                self._clip(cache, "u__clip002", {
                    "fps": 30.0,
                    "source": "/cache/atomicdance-assets/scratch/c1/refix/uploads/u.mp4",
                })
                record = self._inventory(cache, upload_root="videos")[0]
            finally:
                os.chdir(previous)
            self.assertEqual(record["source_video"], "videos/u.mp4")

    def test_the_spelling_matches_the_untouched_rows(self):
        # One manifest, one spelling: a rewritten row that named the same file
        # absolutely would read as a different upload to anything grouping by
        # the string.
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            previous = os.getcwd()
            os.chdir(raw)
            try:
                (root / "videos").mkdir()
                (root / "videos" / "u.mp4").write_bytes(b"\x00")
                cache = root / "cache"
                self._clip(cache, "u__clip000", {"fps": 30.0, "source": "videos/u.mp4"})
                self._clip(cache, "u__clip001", {
                    "fps": 30.0,
                    "source": "/cache/atomicdance-assets/scratch/c1/refix/uploads/u.mp4",
                })
                records = self._inventory(cache, upload_root="videos")
            finally:
                os.chdir(previous)
            self.assertEqual({r["source_video"] for r in records}, {"videos/u.mp4"})

    def test_no_upload_root_leaves_the_path_alone_rather_than_guessing(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            cache = root / "cache"
            self._clip(cache, "u__clip000", {
                "fps": 30.0,
                "source": "/cache/atomicdance-assets/scratch/c1/refix/uploads/u.mp4",
            })
            record = self._inventory(cache)[0]
            self.assertEqual(record["source_video"],
                             "scratch/c1/refix/uploads/u.mp4")
            self.assertNotIn("source_video_working_copy", record)


class OssPublishRefusesNonKeys(unittest.TestCase):
    """The OSS-resident stage publishes keys, so it has to refuse non-keys.

    ``to_repo_keys`` strips the checkout root and the cache mount, which turns
    ``/cache/atomicdance-assets/scratch/c1/refix/uploads/u.mp4`` into
    ``scratch/c1/refix/uploads/u.mp4`` -- relative, so the old absolute-path
    check passed it, and nothing is ever published under that prefix.  2,463
    clips of this corpus record exactly such a path.
    """

    def test_a_scratch_path_is_not_mistaken_for_a_key(self):
        from tools import run_wild_stage_c_oss as oss
        hits = oss.absolute_strings({"assets": {"source_video": "scratch/c1/refix/u.mp4"}})
        self.assertEqual(hits, [".assets.source_video=scratch/c1/refix/u.mp4"])

    def test_an_absolute_path_still_fails(self):
        from tools import run_wild_stage_c_oss as oss
        self.assertEqual(oss.absolute_strings({"cache_dir": "/tmp/x"}), [".cache_dir=/tmp/x"])

    def test_a_repo_key_passes(self):
        from tools import run_wild_stage_c_oss as oss
        self.assertEqual(oss.absolute_strings(
            {"cache_dir": "data/wild_ingest_v1/u__clip000",
             "source_video": "data/wild_videos_20260811/u.mp4"}), [])


if __name__ == "__main__":
    unittest.main()
