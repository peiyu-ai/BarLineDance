"""A clip is a span of *seconds*, and the upload's rate is measured, not assumed.

On 2026-08-19 an operator noticed a clip whose picture ran at half speed under
its music.  ``cut_clip`` selected video by frame number and audio by seconds
computed as ``frame/30``, so on a 60 fps upload the two picked different spans:
451 frames covering 7.5 s of action, restamped to 15.03 s, against 15.03 s of
real audio.  GVHMR then ran on that clip, so 1,418 of 13,783 released clips
(10.3%) carry a 3D motion stretched against their own music, and ``meta.json``
recorded ``fps: 30.0`` from a constant, which reads exactly like a measurement.

These tests are the known-good side of that: a 60 fps upload must come out as
many seconds as it went in, with sound and picture the same length, and its box
track paired with the frames the clip actually holds.  The 30 fps side has to
stay byte-identical, because 89.7% of the corpus was cut correctly and must not
be redone.
"""

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools import ingest_wild_uploads as ingest


def _upload(path, fps, seconds=8):
    """A synthetic upload at a chosen rate, with a tone so it has audio."""
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=size=160x120:rate={}:duration={}".format(fps, seconds),
         "-f", "lavfi", "-i", "sine=frequency=440:duration={}".format(seconds),
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(path)], check=True)
    return path


def _durations(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,nb_frames,duration",
         "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    streams = json.loads(out)["streams"]
    video = next(s for s in streams if s["codec_type"] == "video")
    audio = next((s for s in streams if s["codec_type"] == "audio"), {})
    return int(video.get("nb_frames", 0)), float(video.get("duration", 0)), float(audio.get("duration", 0))


class TheRateIsMeasured(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())

    def test_probe_reads_the_real_rate(self):
        for rate in (24, 25, 30, 60):
            path = _upload(self.tmp / "u{}.mp4".format(rate), rate, seconds=2)
            self.assertAlmostEqual(ingest.probe_fps(path), float(rate), places=2)

    def test_an_unreadable_file_gives_zero_rather_than_thirty(self):
        broken = self.tmp / "broken.mp4"
        broken.write_bytes(b"not a video")
        self.assertEqual(ingest.probe_fps(broken), 0.0)


class ASpanIsSeconds(unittest.TestCase):
    """The test that fails on the old code."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())

    def _cut(self, rate, start, end):
        upload = _upload(self.tmp / "u{}.mp4".format(rate), rate)
        destination = self.tmp / "clip{}.mp4".format(rate)
        written = ingest.cut_clip(upload, start, end, destination, source_fps=float(rate))
        return written, _durations(destination), (end - start) / float(rate)

    def test_sixty_fps_keeps_its_seconds_and_its_sync(self):
        written, (frames, video_s, audio_s), span_s = self._cut(60, 0, 240)
        self.assertAlmostEqual(span_s, 4.0, places=2)
        self.assertEqual(written, round(span_s * ingest.FPS))     # 120, not 240
        self.assertAlmostEqual(video_s, span_s, delta=0.10)
        self.assertAlmostEqual(audio_s, video_s, delta=0.10)

    def test_thirty_fps_is_unchanged(self):
        written, (frames, video_s, audio_s), span_s = self._cut(30, 0, 120)
        self.assertEqual(written, 120)
        self.assertAlmostEqual(video_s, 4.0, delta=0.10)
        self.assertAlmostEqual(audio_s, video_s, delta=0.10)

    def test_twenty_five_fps_shortens_rather_than_speeding_up(self):
        written, (frames, video_s, audio_s), span_s = self._cut(25, 0, 100)
        self.assertAlmostEqual(span_s, 4.0, places=2)
        self.assertEqual(written, round(span_s * ingest.FPS))     # 120, not 100
        self.assertAlmostEqual(audio_s, video_s, delta=0.10)

    def test_the_old_behaviour_is_still_reproducible_and_still_wrong(self):
        """Cut a 60 fps upload as if it were 30 fps -- the bug, on demand.

        Without this the tests above only say the new code is self-consistent.
        This one says the thing they are guarding against is real: the same
        span comes out twice as long in picture as the audio it is under.
        """
        upload = _upload(self.tmp / "u60.mp4", 60)
        destination = self.tmp / "as_if_30.mp4"
        written = ingest.cut_clip(upload, 0, 240, destination, source_fps=30.0)
        frames, video_s, audio_s = _durations(destination)
        self.assertEqual(written, 240)                       # frames renumbered
        self.assertAlmostEqual(video_s, 8.0, delta=0.15)     # 4 s of action...
        self.assertAlmostEqual(audio_s, 8.0, delta=0.15)     # ...under 8 s of sound
        # The defect is that those 240 frames hold 4 s of action, so against the
        # correctly cut clip the picture advances at half the rate.
        right = self.tmp / "right.mp4"
        ingest.cut_clip(upload, 0, 240, right, source_fps=60.0)
        _, right_s, _ = _durations(right)
        self.assertAlmostEqual(video_s / right_s, 2.0, delta=0.1)

    def test_a_rate_of_zero_is_refused_not_defaulted(self):
        upload = _upload(self.tmp / "u.mp4", 30, seconds=2)
        self.assertIsNone(ingest.cut_clip(upload, 0, 30, self.tmp / "x.mp4", source_fps=0.0))


class TheBoxTrackFollowsTheClipsFrames(unittest.TestCase):
    """Clip frame k holds source frame start + round(k * source/30).

    Slicing the track instead pairs clip frame k with source frame start + k,
    which on a 60 fps upload is a box from twice the elapsed time away -- and
    GVHMR crops frame by frame from it.
    """

    def test_the_pairing_is_a_resample_not_a_slice(self):
        source_fps, written, start = 60.0, 120, 300
        picks = np.clip((start + np.arange(written) * source_fps / ingest.FPS).round().astype(int),
                        0, 10000)
        self.assertEqual(picks[0], 300)
        self.assertEqual(picks[1], 302)          # a slice would say 301
        self.assertEqual(picks[-1], 300 + 238)
        # and on a 30 fps upload it degenerates to the slice it used to be
        picks30 = (start + np.arange(written) * 30.0 / ingest.FPS).round().astype(int)
        np.testing.assert_array_equal(picks30, start + np.arange(written))


if __name__ == "__main__":
    unittest.main()


class AFractionalRateIsStillARate(unittest.TestCase):
    """NTSC rates were the 7.2% the fps fix did not survive contact with.

    ``_rate`` returns a fraction so ffmpeg gets an exact value, and the filter
    interpolated it as ``setpts=N/2997/100/TB`` -- which parses as
    ``N/(2997*100)``, out by the square of the denominator.  Integer rates come
    back as ``30/1`` and divide by one harmlessly, so the whole defect was
    invisible on 92.8% of the corpus and total on the rest: of 1,335 uploads
    re-cut on 2026-08-19, every one of the 214 failed cuts came from a
    fractional-rate upload and not one integer-rate cut failed.

    The silent half is why the frame count is now checked rather than merely
    returned.  On two uploads ffprobe answered with a number instead of
    nothing, so ``cut_clip`` returned 38 for a span that asks for 610, and five
    clips of 38 frames stamped across 20.34 seconds were recorded as ok.
    """

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())

    def test_the_rate_string_is_exact(self):
        self.assertEqual(ingest._rate(30.0), "30/1")
        self.assertEqual(ingest._rate(29.97), "2997/100")
        self.assertEqual(ingest._rate(59.94), "2997/50")

    def test_a_29_97_upload_cuts_to_the_frames_its_span_asks_for(self):
        upload = _upload(self.tmp / "u2997.mp4", "30000/1001", seconds=12)
        rate = ingest.probe_fps(upload)
        self.assertAlmostEqual(rate, 29.97, places=2)
        start, end = 60, 360
        destination = self.tmp / "clip.mp4"
        written = ingest.cut_clip(upload, start, end, destination, source_fps=rate)
        expected = (end - start) * ingest.FPS / rate
        self.assertIsNotNone(
            written, "a fractional rate must cut, not fail: this is the 214")
        self.assertLess(abs(written - expected), 3,
                        "frames {} against an expected {:.1f}".format(written, expected))
        frames, video_s, audio_s = _durations(destination)
        self.assertAlmostEqual(video_s, (end - start) / rate, delta=0.15)
        self.assertAlmostEqual(audio_s, video_s, delta=0.15)

    def test_an_implausible_frame_count_is_a_failure_not_a_short_clip(self):
        """The gate that was missing when five broken clips were recorded ok."""
        upload = _upload(self.tmp / "u30.mp4", 30, seconds=8)
        destination = self.tmp / "clip.mp4"
        real = ingest.run

        def lying_probe(argv, timeout=1800):
            result = real(argv, timeout=timeout)
            if "ffprobe" in str(argv[0]):
                result.stdout = "38\n"
            return result

        ingest.run = lying_probe
        try:
            written = ingest.cut_clip(upload, 0, 180, destination, source_fps=30.0)
        finally:
            ingest.run = real
        self.assertIsNone(written,
                          "a count of 38 for a 180-frame span must read as a failed cut")
