"""A derived artifact is fresh only if it was built from the bytes that are there now.

The published corpus once carried 3,991 clips whose S3D features existed, were
the right length, and joined to everything downstream -- and had been extracted
from an older cut of the video.  The gate in front of that stage asked whether a
feature file was present, and present it was.  These tests hold the replacement
to the harder question and, more importantly, hold it to *failing* when the
answer is no: a gate that cannot be shown to fail is worse than no gate, because
it reads like a check that happened.

Three properties, one per way the old shape went wrong:

* a clip nothing touched reads fresh, and reads it from a real hash comparison
  rather than from an absent file;
* a clip whose bytes changed reads stale, on every stage that recorded a hash;
* a clip whose video cannot be read is neither -- the comparison did not
  happen, and saying "fresh" there is the failure this repo keeps meeting.
"""

import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools import audit_clip_freshness as audit


def _clip(root, stem, payload):
    directory = pathlib.Path(root) / stem
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "clip.mp4").write_bytes(payload)
    return hashlib.sha256(payload[: audit.HASH_LIMIT]).hexdigest()


class FreshnessJudgment(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.workspace.name) / "ingest"
        self.root.mkdir()
        self.addCleanup(self.workspace.cleanup)
        self._previous = audit._ingest_root
        audit._ingest_root = str(self.root)
        self.addCleanup(lambda: setattr(audit, "_ingest_root", self._previous))

    def test_matching_hash_reads_fresh_on_both_stages(self):
        digest = _clip(self.root, "a__clip000", b"video bytes")
        with mock.patch.object(audit, "recorded_3d", return_value=digest), \
             mock.patch.object(audit, "recorded_s3d", return_value=digest):
            row = audit.judge("a__clip000", ["3d", "s3d"])
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["verdicts"], {"3d": "fresh", "s3d": "fresh"})
        # The verdict came from a comparison, not from a default: the hash the
        # tool computed is present and is the one it compared against.
        self.assertEqual(row["clip_sha256_1mb"], digest)

    def test_changed_bytes_read_stale(self):
        _clip(self.root, "a__clip000", b"the new cut of this clip")
        stale = hashlib.sha256(b"the old cut").hexdigest()
        with mock.patch.object(audit, "recorded_3d", return_value=stale), \
             mock.patch.object(audit, "recorded_s3d", return_value=stale):
            row = audit.judge("a__clip000", ["3d", "s3d"])
        self.assertEqual(row["verdicts"], {"3d": "stale", "s3d": "stale"})

    def test_missing_derivative_is_absent_not_fresh(self):
        _clip(self.root, "a__clip000", b"video bytes")
        with mock.patch.object(audit, "recorded_3d", return_value=None), \
             mock.patch.object(audit, "recorded_s3d", return_value=None):
            row = audit.judge("a__clip000", ["3d", "s3d"])
        self.assertEqual(row["verdicts"], {"3d": "absent", "s3d": "absent"})

    def test_unreadable_video_is_not_a_pass(self):
        row = audit.judge("never__clip000", ["3d", "s3d"])
        self.assertEqual(row["status"], "video_missing")
        self.assertNotIn("verdicts", row)

    def test_overridden_ingest_root_does_not_fall_back_to_the_store(self):
        """The scratch tree is the whole point of the override.

        Falling back to OSS for a clip the caller deliberately staged elsewhere
        would compare against the released generation -- the exact confusion
        this audit exists to catch, reintroduced by the audit itself.
        """
        with mock.patch.object(audit.asset_io, "remote_path") as remote:
            self.assertIsNone(audit.clip_head("absent__clip000"))
        remote.assert_not_called()


class GateExitCode(unittest.TestCase):
    """The gate's verdict has to reach the shell, or a driver cannot stop on it."""

    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.base = pathlib.Path(self.workspace.name)
        self.root = self.base / "ingest"
        self.root.mkdir()
        self.digest = _clip(self.root, "a__clip000", b"video bytes")
        (self.base / "clips.txt").write_text("a__clip000\n", encoding="utf-8")

    def _run(self, recorded):
        script = (
            "import sys, json;"
            "sys.path.insert(0, {repo!r});"
            "from unittest import mock;"
            "from tools import audit_clip_freshness as audit;"
            "p1 = mock.patch.object(audit, 'recorded_3d', return_value={rec!r});"
            "p2 = mock.patch.object(audit, 'recorded_s3d', return_value={rec!r});"
            "p1.start(); p2.start();"
            "sys.exit(audit.main(['--clips', {clips!r}, '--output', {out!r},"
            " '--ingest-root', {root!r}]))"
        ).format(repo=str(pathlib.Path(__file__).resolve().parents[1]),
                 rec=recorded, clips=str(self.base / "clips.txt"),
                 out=str(self.base / "audit.json"), root=str(self.root))
        return subprocess.run([sys.executable, "-c", script],
                              capture_output=True, text=True)

    def test_agreement_exits_zero_and_disagreement_exits_one(self):
        agreeing = self._run(self.digest)
        self.assertEqual(agreeing.returncode, 0, agreeing.stdout + agreeing.stderr)
        self.assertIn("still disagreeing 0", agreeing.stdout)

        disagreeing = self._run(hashlib.sha256(b"an older cut").hexdigest())
        self.assertEqual(disagreeing.returncode, 1,
                         disagreeing.stdout + disagreeing.stderr)
        self.assertIn("still disagreeing 2", disagreeing.stdout)

        written = json.loads((self.base / "audit.json").read_text(encoding="utf-8"))
        self.assertEqual(written["stale"]["s3d"], ["a__clip000"])


if __name__ == "__main__":
    unittest.main()


class MusicStage(unittest.TestCase):
    """The stage that had no gate until 2026-08-25.

    3D and S3D write the hash of the video they read into their own output.
    The 35-D music features are a bare ``.npy``, so the record has to come from
    a manifest -- and the performance bundle did not carry one, which is how 234
    of 1,575 evaluation rows could hold another generation's music while every
    hash inside the bundle agreed with every other.  Two things therefore have
    to hold: the comparison is against ``audio.wav`` whole (what the extractor
    hashed), and a manifest with no provenance is refused rather than reported
    as a clean corpus.
    """

    def _corpus(self, root, stem, audio):
        directory = pathlib.Path(root) / stem
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "audio.wav").write_bytes(audio)
        (directory / "clip.mp4").write_bytes(b"video")
        return hashlib.sha256(audio).hexdigest()

    def test_the_clip_side_is_the_whole_audio_not_its_first_megabyte(self):
        # The extractor hashes the whole file.  A gate comparing a window
        # against that number disagrees with everything and reads as a
        # measurement while measuring nothing.
        with tempfile.TemporaryDirectory() as raw:
            payload = b"x" * (audit.HASH_LIMIT + 4096)
            digest = self._corpus(raw, "u__clip000", payload)
            with mock.patch.object(audit, "_ingest_root", raw):
                self.assertEqual(audit.clip_audio_hash("u__clip000"), digest)
            self.assertNotEqual(digest,
                                hashlib.sha256(payload[:audit.HASH_LIMIT]).hexdigest())

    def test_music_built_from_these_bytes_is_fresh_and_from_others_is_stale(self):
        with tempfile.TemporaryDirectory() as raw:
            digest = self._corpus(raw, "u__clip000", b"the audio as it is now")
            with mock.patch.object(audit, "_ingest_root", raw), \
                 mock.patch.dict(audit._music_records,
                                 {"u__clip000": digest}, clear=True):
                row = audit.judge("u__clip000", ["music"])
            self.assertEqual(row["verdicts"], {"music": "fresh"})
            self.assertEqual(row["clip_audio_sha256"], digest)

            with mock.patch.object(audit, "_ingest_root", raw), \
                 mock.patch.dict(audit._music_records,
                                 {"u__clip000": "0" * 64}, clear=True):
                row = audit.judge("u__clip000", ["music"])
            self.assertEqual(row["verdicts"], {"music": "stale"})

    def test_a_clip_the_manifest_never_mentions_is_absent_not_fresh(self):
        with tempfile.TemporaryDirectory() as raw:
            self._corpus(raw, "u__clip000", b"audio")
            with mock.patch.object(audit, "_ingest_root", raw), \
                 mock.patch.dict(audit._music_records, {}, clear=True):
                row = audit.judge("u__clip000", ["music"])
            self.assertEqual(row["verdicts"], {"music": "absent"})

    def test_a_missing_audio_beside_a_readable_video_is_its_own_verdict(self):
        # Not "stale" and not "fresh": the comparison did not happen.  Folding
        # it into either is the failure this whole tool exists against.
        with tempfile.TemporaryDirectory() as raw:
            directory = pathlib.Path(raw) / "u__clip000"
            directory.mkdir(parents=True)
            (directory / "clip.mp4").write_bytes(b"video")   # no audio.wav
            with mock.patch.object(audit, "_ingest_root", raw), \
                 mock.patch.dict(audit._music_records,
                                 {"u__clip000": "0" * 64}, clear=True), \
                 mock.patch.object(audit, "recorded_3d", lambda s: None):
                row = audit.judge("u__clip000", ["3d", "music"])
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["verdicts"]["music"], "clip_bytes_missing")

    def test_a_bundle_that_predates_the_field_is_refused_not_read_as_clean(self):
        with tempfile.TemporaryDirectory() as raw:
            manifest = pathlib.Path(raw) / "sequences.jsonl"
            manifest.write_text("".join(json.dumps(r) + "\n" for r in [
                {"sequence_id": "wild_v4:111:clip000", "music_sha256": "a" * 64},
                {"sequence_id": "wild_v4:111:clip001", "music_sha256": "b" * 64},
            ]), encoding="utf-8")
            with self.assertRaises(SystemExit) as caught:
                audit.load_music_manifest(manifest)
            self.assertIn("source_audio_sha256", str(caught.exception))

    def test_both_manifest_shapes_are_read_and_named_onto_ingest_stems(self):
        with tempfile.TemporaryDirectory() as raw:
            manifest = pathlib.Path(raw) / "m.jsonl"
            manifest.write_text("".join(json.dumps(r) + "\n" for r in [
                # a performance bundle row, from 2026-08-25
                {"sequence_id": "wild_v4:111:clip000", "source_audio_sha256": "a" * 64},
                # an audio bundle row, where the field has always lived
                {"sequence_id": "wild_v4:222:clip003",
                 "audio_feature": {"source_audio_sha256": "b" * 64}},
                # and legacy_source_name wins where it exists
                {"sequence_id": "wild_v4:333:clip000", "legacy_source_name": "333__clip007",
                 "provenance": {"source_audio_sha256": "c" * 64}},
            ]), encoding="utf-8")
            records = audit.load_music_manifest(manifest)
            self.assertEqual(records, {"111__clip000": "a" * 64,
                                       "222__clip003": "b" * 64,
                                       "333__clip007": "c" * 64})
