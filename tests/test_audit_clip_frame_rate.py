"""The frame-rate audit must be able to say "no", not only "ok".

``tools/audit_clip_frame_rate.py`` answers one question -- does this clip's
picture run at real time, or is it permanently fast or slow -- and it answers it
about a corpus that is currently clean.  A checker that has only ever seen a
clean corpus has not been shown to work, so the centre of this file is a
*positive control built from the real code*: the pre-2026-08-19 ``cut_clip``,
loaded out of git history rather than retyped, run over a 60 fps upload.  That
is the exact defect the operator hit on 2026-08-19 (451 frames of 7.5 s of
action restamped to 15.03 s), and the audit must report ``playback_ratio`` 2.0
and status ``speed_defect`` for it.

Three arms, because one control is not enough to fix the direction:

* **A** -- 30 fps upload, pre-fix cut.  On a 30 fps source the pre-fix rule is
  the identity, so this must read 1.0.  Without it, a checker that flagged
  everything would pass the positive control.
* **B** -- 60 fps upload, pre-fix cut.  Must read 2.0.
* **C** -- the same 60 fps upload through the *current* ``cut_clip``.  Must read
  1.0.  This is what makes the audit a statement about the corpus rather than
  about the number 60.

And one more, which is the reason the tool probes ffprobe instead of reading
``meta.json``: run with ``--trust-meta-fps``, arm B reads ``ok``.  A pre-fix
meta has no ``source_fps`` field at all and its ``fps: 30.0`` is a constant, so
a checker that believes the meta cannot see the defect that the meta caused.
The flag exists only to keep that failure demonstrated here.
"""

import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools import audit_clip_frame_rate as audit
from tools.ingest_wild_uploads import cut_clip as current_cut_clip

REPO = pathlib.Path(__file__).resolve().parents[1]
# The commit that replaced the frame-renumbering cut with a resampling one; its
# parent is the last tree that still holds the defect.
PRE_FIX_COMMIT = "49e95a3^"


def _load_pre_fix_cutter(workdir):
    """The real pre-2026-08-19 cut_clip, out of git rather than retyped."""
    source = subprocess.run(
        ["git", "-C", str(REPO), "show",
         "{}:tools/ingest_wild_uploads.py".format(PRE_FIX_COMMIT)],
        capture_output=True, text=True)
    if source.returncode != 0:
        raise unittest.SkipTest("pre-fix ingest revision not reachable in this checkout")
    path = workdir / "ingest_pre_fix.py"
    path.write_text(source.stdout, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("ingest_pre_fix", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.cut_clip


def _upload(path, fps, seconds=4):
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=size=160x120:rate={}:duration={}".format(fps, seconds),
         "-f", "lavfi", "-i", "sine=frequency=440:duration={}".format(seconds),
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(path)], check=True)
    return path


def _write_meta(clip_dir, upload, span, frames, source_fps=None):
    clip_dir.mkdir(parents=True, exist_ok=True)
    meta = {"num_frames": int(frames), "video_w": 160, "video_h": 120,
            "fps": 30.0, "source": str(upload),
            "source_frame_span": [int(span[0]), int(span[1])],
            "cache_version": "atomicdance-wild-ingest-v1",
            "span_end_reason": "upload_end"}
    if source_fps is not None:
        meta["source_fps"] = float(source_fps)
    (clip_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


class FrameRateAuditControls(unittest.TestCase):
    """Built once: three real cuts, so every assertion reads the same evidence."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = pathlib.Path(cls._tmp.name)
        cls.root = root
        pre_fix_cut = _load_pre_fix_cutter(root)
        uploads = root / "uploads"
        uploads.mkdir()
        u30 = _upload(uploads / "u30.mp4", 30)
        u60 = _upload(uploads / "u60.mp4", 60)
        ingest = root / "ingest"
        # Spans name the *upload's* frames, so the same 4 seconds of action is
        # [0,120) at 30 fps and [0,240) at 60 fps.
        cls.written = {
            "A_pre_fix_30": pre_fix_cut(u30, 0, 120, ingest / "A_pre_fix_30" / "clip.mp4"),
            "B_pre_fix_60": pre_fix_cut(u60, 0, 240, ingest / "B_pre_fix_60" / "clip.mp4"),
            "C_current_60": current_cut_clip(u60, 0, 240,
                                             ingest / "C_current_60" / "clip.mp4",
                                             source_fps=60.0),
        }
        for name, frames in cls.written.items():
            if frames is None:
                raise unittest.SkipTest("ffmpeg could not produce control clip " + name)
        _write_meta(ingest / "A_pre_fix_30", u30, (0, 120), cls.written["A_pre_fix_30"])
        _write_meta(ingest / "B_pre_fix_60", u60, (0, 240), cls.written["B_pre_fix_60"])
        _write_meta(ingest / "C_current_60", u60, (0, 240), cls.written["C_current_60"],
                    source_fps=60.0)
        cls.ingest = ingest

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _audit(self, name, **kwargs):
        return audit.audit_clip(name, self.ingest, self.root, **kwargs)

    def test_pre_fix_cut_on_a_30_fps_upload_is_the_identity(self):
        """Negative control: 89.7% of the corpus was cut this way and is fine."""
        row = self._audit("A_pre_fix_30")
        self.assertEqual(row["status"], "ok", row)
        self.assertAlmostEqual(row["playback_ratio"], 1.0, delta=0.02)
        self.assertEqual(row["ingest_generation"], "pre_fps_fix")

    def test_pre_fix_cut_on_a_60_fps_upload_reads_two_times_slow(self):
        """Positive control: the defect of 2026-08-19, reproduced and caught."""
        row = self._audit("B_pre_fix_60")
        self.assertEqual(row["status"], "speed_defect", row)
        self.assertAlmostEqual(row["playback_ratio"], 2.0, delta=0.02)
        # The direction matters as much as the magnitude: the clip holds *more*
        # frames than the span's real time can fill, so it plays slow.
        self.assertGreater(row["num_frames"], row["expected_frames"])

    def test_current_cut_on_the_same_60_fps_upload_reads_real_time(self):
        row = self._audit("C_current_60")
        self.assertEqual(row["status"], "ok", row)
        self.assertAlmostEqual(row["playback_ratio"], 1.0, delta=0.02)
        self.assertEqual(row["ingest_generation"], "fps_aware")

    def test_trusting_the_meta_cannot_see_the_defect_the_meta_caused(self):
        """Why the rate is probed, not read: the circular check answers ok."""
        blind = self._audit("B_pre_fix_60", trust_meta_fps=True)
        self.assertEqual(blind["status"], "ok", blind)
        self.assertAlmostEqual(blind["playback_ratio"], 1.0, delta=0.02)
        seeing = self._audit("B_pre_fix_60")
        self.assertEqual(seeing["status"], "speed_defect")

    def test_recorded_source_fps_that_disagrees_with_the_file_is_reported(self):
        """A meta that names a rate the file does not have is its own defect."""
        _write_meta(self.ingest / "D_wrong_record", self.root / "uploads" / "u60.mp4",
                    (0, 240), self.written["C_current_60"], source_fps=30.0)
        row = self._audit("D_wrong_record")
        self.assertEqual(row["status"], "recorded_fps_disagrees", row)

    def test_variable_rate_is_reported_separately_from_speed(self):
        """avg != r is the 2026-08-25 defect and has a different repair."""
        original = audit.probe_rates
        audit.probe_rates = lambda path: (25.0, 30.0)
        try:
            row = self._audit("A_pre_fix_30")
        finally:
            audit.probe_rates = original
        # The picture is at the right speed for the rate the cut used, so the
        # ratio stays 1.0 and the finding is the rate pair, not the speed.
        self.assertEqual(row["status"], "vfr", row)
        self.assertAlmostEqual(row["playback_ratio"], 1.0, delta=0.02)

    def test_a_converted_3d_that_lost_half_its_frames_is_caught(self):
        """convert_gvhmr_result keeps every second frame when it believes 60 fps."""
        converted = self.root / "converted"
        (converted / "A_pre_fix_30").mkdir(parents=True)
        (converted / "A_pre_fix_30" / "metadata.json").write_text(json.dumps({
            "frames_30fps": self.written["A_pre_fix_30"] // 2, "fps": 30.0,
            "extract_meta": {"video_frames": self.written["A_pre_fix_30"] // 2}}),
            encoding="utf-8")
        row = self._audit("A_pre_fix_30", converted=converted)
        self.assertEqual(row["status"], "converted_frame_mismatch", row)

    def test_missing_upload_is_refused_rather_than_assumed_thirty(self):
        _write_meta(self.ingest / "E_no_upload", self.root / "uploads" / "gone.mp4",
                    (0, 120), 120)
        row = self._audit("E_no_upload")
        self.assertEqual(row["status"], "upload_missing", row)


if __name__ == "__main__":
    unittest.main()
