"""Resuming must not swallow a deliberate re-cut, and a matching span is not proof.

On 2026-08-19 the fps re-cut was launched over 1,335 uploads and did nothing.
Seven shards printed ``1335 of 1335 already recorded`` and exited 0, because the
resume gate skips any upload the append-only manifests mention and the original
ingest had mentioned all of them.  The census that ran afterwards then reported
no orphans, no changed spans and a fully fresh corpus -- four clean readings
from a corpus nobody had touched.  Every one of those readings was correct; the
run was the lie.

Two gates were wrong and only one of them is obvious:

* the **upload** gate skipped everything, which is the loud failure above;
* the **clip** gate skipped on ``source_frame_span`` equality alone.  A span
  can be unchanged while the clip is wrong -- on a 25 fps upload whose single
  span is the whole file, old and new spans are both ``[0, N]`` while the old
  clip.mp4 holds N frames of 25 fps action restamped to 30.  That one would
  have survived the fix to the first, silently, on exactly the clips whose
  names and spans matched.
"""

import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools import ingest_wild_uploads as ingest


class ManifestReading(unittest.TestCase):
    def test_the_later_row_supersedes_the_earlier_one(self):
        """Otherwise every accounting reader doubles a re-ingested upload."""
        with tempfile.TemporaryDirectory() as workspace:
            root = pathlib.Path(workspace)
            (root / "ingest_shard0.jsonl").write_text(
                json.dumps({"upload": "u", "status": "ok", "seconds": 10.0}) + "\n"
                + json.dumps({"upload": "v", "status": "ok", "seconds": 5.0}) + "\n",
                encoding="utf-8")
            (root / "ingest_shard1.jsonl").write_text(
                json.dumps({"upload": "u", "status": "ok", "seconds": 20.0}) + "\n",
                encoding="utf-8")
            rows = ingest.manifest_rows(root)
        self.assertEqual(sorted(rows), ["u", "v"])
        self.assertEqual(rows["u"]["seconds"], 20.0)
        self.assertEqual(sum(r["seconds"] for r in rows.values()), 25.0)

    def test_a_corrupt_line_does_not_take_the_reader_down(self):
        with tempfile.TemporaryDirectory() as workspace:
            root = pathlib.Path(workspace)
            (root / "ingest_shard0.jsonl").write_text(
                "not json\n" + json.dumps({"upload": "u"}) + "\n", encoding="utf-8")
            self.assertEqual(sorted(ingest.manifest_rows(root)), ["u"])


class _CutReached(Exception):
    """Raised by the stubbed cutter so the test stops at the gate it is about."""


class ClipSkipGate(unittest.TestCase):
    """``ingest`` decides per clip whether an existing output is the same clip.

    The stubs replace the dancer-tracking and pose machinery, which this test is
    not about; the cutter raises as soon as it is reached, so "was the clip
    re-cut" is observed directly rather than inferred from what the run left
    behind.
    """

    def _run(self, previous_meta, span=(0, 500)):
        import numpy as np

        from tools import dwpose_video

        with tempfile.TemporaryDirectory() as workspace:
            root = pathlib.Path(workspace)
            out = root / "out"
            clip_dir = out / "up__clip000"
            clip_dir.mkdir(parents=True)
            (clip_dir / "meta.json").write_text(json.dumps(previous_meta),
                                                encoding="utf-8")
            upload = root / "up.mp4"
            upload.write_bytes(b"x")

            frames = 600
            detections = [np.zeros((1, 4), np.float32)] * frames
            cut_scores = np.zeros(frames, np.float32)
            track = {"frames": np.arange(frames),
                     "boxes": np.tile(np.array([10, 10, 50, 50], np.float32),
                                      (frames, 1))}
            calls = []

            def fake_cut(upload_path, begin, finish, destination, source_fps=30.0):
                calls.append((int(begin), int(finish)))
                raise _CutReached()

            # ``probe_rates``, not ``probe_fps``: since 2026-08-24 ``ingest``
            # reads BOTH of the container's rate claims, because the picture is
            # cut by frame number and the sound by seconds and those name the
            # same span only while the two agree.  A constant-rate upload is
            # (25.0, 25.0).
            with mock.patch.object(ingest, "probe_rates", return_value=(25.0, 25.0)), \
                 mock.patch.object(ingest, "scan_upload",
                                   return_value=(detections, cut_scores)), \
                 mock.patch.object(dwpose_video, "link_tracks", return_value=[track]), \
                 mock.patch.object(dwpose_video, "score_track", return_value=1.0), \
                 mock.patch.object(ingest, "find_spans",
                                   return_value=[(span[0], span[1], "upload_end")]), \
                 mock.patch.object(ingest, "find_cuts",
                                   return_value=np.zeros(frames, bool)), \
                 mock.patch.object(ingest, "cut_clip", side_effect=fake_cut):
                try:
                    record = ingest.ingest(upload, out, extractor=None,
                                           max_seconds=24.0, write_bbx=False)
                except _CutReached:
                    record = None
            return record, calls

    def test_an_old_generation_clip_is_recut_even_when_its_span_is_unchanged(self):
        """The span matches, so the oldest gate skipped it -- and it was wrong."""
        _, calls = self._run({"source_frame_span": [0, 500], "fps": 30.0})
        self.assertEqual(calls, [(0, 500)], "the clip must actually be re-cut")

    def test_a_generation_marker_made_of_a_field_going_stale_is_the_defect(self):
        """The 2026-08-25 failure, held in place.

        The gate used to read "does this meta carry ``source_fps``" as "was it
        cut by the current ingest".  The 2026-08-19 generation carries that
        field and was *not* current, so on 7564725723304119595__clip000 -- span
        [0, 399] against a constant-rate re-encode's span [0, 399], the same two
        integers naming two different intervals -- the clip was skipped and kept
        its 2x-fast picture.  Carrying ``source_fps`` must not on its own buy a
        skip.
        """
        record, calls = self._run(
            {"source_frame_span": [0, 500], "fps": 30.0, "source_fps": 25.0})
        self.assertEqual(calls, [(0, 500)], "the clip must actually be re-cut")
        self.assertEqual(record, None)      # the stub cutter stops the run

    def test_the_same_bytes_and_the_same_span_are_the_only_skip(self):
        import hashlib
        # The identity ``ingest`` computes for an upload of b"x".
        key = "{}:{}".format(1, hashlib.sha256(b"x").hexdigest())
        record, calls = self._run({"source_frame_span": [0, 500], "fps": 30.0,
                                   "source_fps": 25.0, "source_key": key})
        self.assertEqual(calls, [], "an already-correct clip must not be redone")
        self.assertEqual(record["clips"][0]["status"], "exists")

    def test_a_re_encode_under_the_same_name_is_not_the_same_clip(self):
        # What the constant-rate normalisation does: same upload name, same
        # span integers, different bytes -- and therefore a different span of
        # time.  This is the case the field-presence marker could not see.
        _, calls = self._run({"source_frame_span": [0, 500], "fps": 30.0,
                              "source_fps": 25.0, "source_key": "9:" + "0" * 64})
        self.assertEqual(calls, [(0, 500)])

    def test_a_moved_span_is_recut_whatever_generation_wrote_it(self):
        import hashlib
        key = "{}:{}".format(1, hashlib.sha256(b"x").hexdigest())
        _, calls = self._run(
            {"source_frame_span": [0, 400], "fps": 30.0, "source_fps": 25.0,
             "source_key": key})
        self.assertEqual(calls, [(0, 500)])


if __name__ == "__main__":
    unittest.main()


class VariableRateGate(unittest.TestCase):
    """An upload whose two rate claims disagree must not be cut at all.

    ``cut_clip`` selects the picture by frame number and the sound by seconds
    computed from the rate.  When ``avg_frame_rate`` and ``r_frame_rate``
    disagree those are two different spans: measured 2026-08-24 on one upload,
    the picture came from 27.17-54.37 s and the sound from 13.63 s for 13.65 s.
    Nothing downstream can see it -- the clip's own container is consistent --
    so the refusal has to happen here.
    """

    def _run(self, rates, cfr_cache=None):
        with tempfile.TemporaryDirectory() as workspace:
            root = pathlib.Path(workspace)
            upload = root / "up.mp4"
            upload.write_bytes(b"x")
            with mock.patch.object(ingest, "probe_rates", return_value=rates):
                return ingest.ingest(upload, root / "out", extractor=None,
                                     max_seconds=24.0, write_bbx=False,
                                     cfr_cache=cfr_cache)

    def test_a_variable_rate_upload_is_refused_and_names_both_rates(self):
        record = self._run((31.617, 60.0))
        self.assertEqual(record["status"], "variable_frame_rate")
        # Both, so the record says by how much rather than only that it failed.
        self.assertAlmostEqual(record["avg_frame_rate"], 31.617, places=3)
        self.assertAlmostEqual(record["r_frame_rate"], 60.0, places=3)
        self.assertNotIn("clips", record)

    def test_a_failed_normalisation_is_refused_rather_than_cut_anyway(self):
        with tempfile.TemporaryDirectory() as cache:
            with mock.patch.object(ingest, "normalize_to_cfr", return_value=None):
                record = self._run((31.617, 60.0), cfr_cache=pathlib.Path(cache))
        self.assertEqual(record["status"], "cfr_normalize_failed")

    def test_an_unreadable_rate_is_still_refused(self):
        self.assertEqual(self._run((0.0, 0.0))["status"], "unreadable_frame_rate")

