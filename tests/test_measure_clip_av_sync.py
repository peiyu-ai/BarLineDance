"""Does a clip's picture and its sound come from the same span of the upload?

This is the only instrument in the repo that measures the defect itself rather
than the container property that predicts it, so it is the one that decides
whether a re-cut worked.  It reached that job after three self-corrections in a
single afternoon, and each of them has a test here, because all three had the
same shape: the instrument could not express the right answer, and the wrong
answer looked like a measurement.

1. the rival peak was taken from the winner's own shoulder, so the margin did
   not separate four wrong locations from 42 right ones;
2. the "is the picture's location also a peak" probe read a single sample, and
   the peak is one or two samples wide -- it scored a *correct* clip's own
   location at -2.3 sd against its peak's 22.4;
3. ``mode="valid"`` cannot place a clip that runs to the end of the upload, and
   all four displaced readings were last clips.

The end-to-end case is the positive control: a clip cut at a known offset from
a synthetic upload has to read 1.0 and 0.0.
"""

import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools import measure_clip_av_sync as sync


def _envelope(length, spikes, seed=0):
    """A sparse onset envelope: what real music gives this tool."""
    rng = np.random.default_rng(seed)
    signal = np.zeros(length, dtype=np.float32)
    signal[rng.choice(length, size=spikes, replace=False)] = 1.0
    return signal


def test_a_clip_at_the_very_end_of_the_upload_can_be_placed_at_all():
    # mode="valid" offers only lags where the clip fits entirely inside the
    # source, so a last clip's true lag is past the end of the range and argmax
    # lands on whatever earlier peak is tallest.  Four of 46 real clips read
    # -8.7 to -15.5 s that way and every one of them was a last clip.
    source = _envelope(4000, 400, seed=1)
    start = 4000 - 1200                      # the clip runs to the last sample
    clip = source[start:]
    found = sync.locate_audio(clip, source, expected_seconds=start / sync.ENVELOPE_HZ)
    assert abs(found["start_seconds"] - start / sync.ENVELOPE_HZ) < 0.05
    assert found["expected_is_a_peak"] is True


def test_the_expected_probe_reads_a_window_because_the_peak_is_one_sample_wide():
    source = _envelope(4000, 400, seed=2)
    start = 1500
    clip = source[start:start + 1200]
    truth = start / sync.ENVELOPE_HZ
    found = sync.locate_audio(clip, source, expected_seconds=truth)
    assert found["expected_is_a_peak"] is True
    # ...and the single-sample reading it replaced does not survive a 20 ms
    # error, which an encoder's rounding produces routinely.
    nudged = sync.locate_audio(clip, source, expected_seconds=truth + 0.02)
    assert nudged["expected_is_a_peak"] is True
    assert abs(nudged["expected_best_seconds"] - truth) < 0.05


def test_the_rival_is_taken_outside_the_winners_own_peak():
    source = _envelope(4000, 400, seed=3)
    clip = source[1500:2700]
    found = sync.locate_audio(clip, source)
    # The rival must be somewhere else entirely, not a sample beside the winner.
    assert abs(found["rival_seconds"] - found["start_seconds"]) >= sync.RIVAL_GUARD_S
    assert found["rival_z"] < found["peak_z"]


def test_repeating_music_is_reported_as_a_period_not_as_a_finding():
    # A source built out of one phrase repeated four times.  Any of the four is
    # an equally good answer, and the displacement between them is the period.
    phrase = _envelope(600, 60, seed=4)
    source = np.tile(phrase, 7)          # long enough to look out to 30 s
    periods = sync.repeat_periods(source, low=2.0, high=30.0)
    assert periods, "a repeating source must expose its period"
    assert any(abs(p["seconds"] - 6.0) < 0.1 for p in periods), periods


def test_a_source_with_no_repeat_reports_none_at_a_meaningful_correlation():
    source = _envelope(4000, 400, seed=5)
    periods = sync.repeat_periods(source, low=2.0, high=30.0)
    # Peaks are always returned -- argsort has to return something -- so the
    # criterion is their strength, which is what the caller compares against.
    assert all(p["r"] < 0.1 for p in periods), periods


def test_a_static_shot_is_ambiguous_rather_than_confidently_located():
    # Every distance is zero, so the winner is not distinguishable from the 199
    # frames tied with it.  The first version divided by the winner only "if
    # distance[best] > 0" and called this case infinitely distinguishable --
    # the one input that is entirely ambiguous, reported as certainty.
    frozen = np.tile(np.full(576, 0.5, dtype=np.float32), (200, 1))
    verdict = sync.locate_frame(frozen[0], frozen)
    assert verdict["ambiguous"] is True

    # An exact match with no rival nearby is still certain: the floor must not
    # make every zero-distance match ambiguous.
    rng = np.random.default_rng(60)
    distinct = rng.random((200, 576), dtype=np.float32)
    assert sync.locate_frame(distinct[42], distinct)["ambiguous"] is False

    rng = np.random.default_rng(6)
    moving = rng.random((200, 576), dtype=np.float32)
    verdict = sync.locate_frame(moving[137], moving)
    assert verdict["index"] == 137
    assert verdict["ambiguous"] is False


def _has_ffmpeg():
    return subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode == 0


@pytest.mark.skipif(not _has_ffmpeg(), reason="needs ffmpeg")
def test_end_to_end_a_clip_cut_at_a_known_offset_reads_one_and_zero():
    """The positive control.  Without it, 'it reads 1.0' means nothing."""
    with tempfile.TemporaryDirectory() as raw:
        tmp = pathlib.Path(raw)
        source = tmp / "up.mp4"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-loglevel", "error",
             "-f", "lavfi", "-i", "testsrc2=size=128x72:rate=30:duration=20",
             "-f", "lavfi", "-i", "anoisesrc=r=44100:d=20:c=pink",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
             "-shortest", str(source)], check=True)
        clip_dir = tmp / "up__clip000"
        clip_dir.mkdir()
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(source),
             "-ss", "6.0", "-t", "8.0", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-c:a", "aac", str(clip_dir / "clip.mp4")], check=True)
        (clip_dir / "meta.json").write_text(json.dumps({"source": str(source)}))

        row = sync.measure(clip_dir, source)
        assert row["status"] == "ok", row
        assert abs(row["picture_over_sound"] - 1.0) < 0.02, row
        assert abs(row["start_offset"]) < 0.25, row
        assert abs(row["picture_start"] - 6.0) < 0.25, row
        assert not row.get("sound_from_elsewhere")
