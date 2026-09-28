"""The strip mux must be BOUNDED in length.

The defect this pins, measured 2026-09-03: the plan-strip image is fed with
``-loop 1``, which is an infinite video input, and ``vstack`` pads its shorter
input rather than ending with it, so the filter graph never reaches EOF.  The
output-level ``-shortest`` does not save it, because the endless stream IS the
video output stream.  Four 14-second strips reached 373-444 MB and 1,105,642
frames -- ten hours of video for fourteen seconds of dance -- and never wrote a
moov atom, so every one was unplayable while looking like a render in progress.
Seven hours of GPU and 1.8 GB went into files that could not be opened.

Two independent bounds, because either alone has failed here:

  * ``shortest=1`` on the vstack that mixes the looped image with real video;
  * an explicit ``-t <seconds>`` on the output.

These read the ffmpeg argv rather than running ffmpeg, so they fail on the
COMMAND being wrong rather than only on a video coming out long -- which is the
difference between a test that catches this in a second and one that catches it
in six hours.
"""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.render_sample_strip import mux_command


def command(strip="/tmp/strip.png", seconds=14.334):
    return mux_command(["/tmp/footage.mp4", "/tmp/stick.mp4"], "/tmp/avatar.mp4",
                       "/tmp/audio.wav", "/tmp/out.mp4",
                       panel_height=520, width=1560, strip=strip, seconds=seconds)


def test_the_looped_image_input_is_present_so_the_risk_is_real():
    # If this ever stops being true the two bounds below become vacuous and
    # would keep passing while guarding nothing.
    assert ["-loop", "1"] == command()[command().index("-loop"):][:2]


def test_the_vstack_that_mixes_in_the_looped_image_ends_with_the_finite_side():
    joined = " ".join(command())
    assert "vstack=inputs=2:shortest=1[v]" in joined


def test_the_output_carries_an_explicit_duration():
    argv = command()
    assert "-t" in argv
    assert float(argv[argv.index("-t") + 1]) == pytest.approx(14.334)


def test_the_duration_is_the_one_it_was_given_not_a_default():
    # A bound that is present but wrong is the failure this catches: -t 0 would
    # satisfy the test above and write an empty file that looks like a render.
    argv = command(seconds=3.5)
    assert float(argv[argv.index("-t") + 1]) == pytest.approx(3.5)


def test_without_a_strip_there_is_no_endless_input_and_still_a_duration():
    argv = command(strip=None)
    assert "-loop" not in argv
    assert "shortest=1" not in " ".join(argv)
    assert float(argv[argv.index("-t") + 1]) == pytest.approx(14.334)


def test_the_audio_stream_index_accounts_for_the_looped_input():
    # Off by one here maps the wrong stream as audio, which renders silent --
    # a defect this repository has already shipped once.
    with_strip = command()
    without = command(strip=None)
    assert with_strip[with_strip.index("-map") + 3] == "4:a"
    assert without[without.index("-map") + 3] == "3:a"


@pytest.mark.parametrize("seconds", [0.0, -1.0, float("nan")])
def test_a_zero_length_avatar_is_refused_rather_than_bounded_to_nothing(seconds):
    # A zero bound satisfies "the command has a -t" and writes an empty mp4
    # that looks exactly like a successful render, so it must raise instead.
    with pytest.raises(SystemExit):
        command(seconds=seconds)


def test_the_playhead_sweep_is_scaled_to_the_clip_length():
    # The red playhead crosses the strip over t/seconds; if that denominator
    # were the endless input's length the head would never move.
    assert "t/14.3340" in " ".join(command())
    assert "t/3.5000" in " ".join(command(seconds=3.5))
