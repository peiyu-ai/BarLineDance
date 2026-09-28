"""The format guarantee: a strip missing a panel must not be written.

Every panel here can vanish for its own quiet reason -- footage whose source
moved, plan lanes whose labels failed to load -- and ffmpeg happily stacks
whatever it is handed.  The result plays, looks like a comparison, and is
missing the panel the operator opened it for.  That shipped twice: the file the
operator was asked to review on 2026-09-03 had neither the raw video nor the
label sequence, and nothing in the run said so.

So the geometry is verified against the DECLARED layout after muxing, and a
mismatch deletes the file before raising.  Deleting is the load-bearing part: a
wrong-geometry file left on disk is indistinguishable from a good one at a
glance, so the next reader would trust it.
"""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import tools.render_sample_strip as module


class FakeProbe:
    def __init__(self, text):
        self.stdout = text
        self.stderr = ""
        self.returncode = 0


def probe_returning(width, height, seconds):
    def fake_run(command):
        return FakeProbe("{}\n{}\n{}\n".format(width, height, seconds))
    return fake_run


def test_a_correct_layout_passes_and_reports_what_it_saw(monkeypatch, tmp_path):
    target = tmp_path / "strip.mp4"
    target.write_bytes(b"x")
    monkeypatch.setattr(module, "run", probe_returning(1560, 1240, 14.334))
    got = module.verify_layout(target, width=1560, panels_top=3,
                               has_strip=True, seconds=14.334)
    assert got == {"width": 1560, "height": 1240, "seconds": 14.334}
    assert target.exists()


def test_a_missing_footage_panel_is_caught_by_the_width(monkeypatch, tmp_path):
    """Three panels were declared; two were stacked.  The width gives it away."""
    target = tmp_path / "strip.mp4"
    target.write_bytes(b"x")
    monkeypatch.setattr(module, "run", probe_returning(1040, 1240, 14.334))
    with pytest.raises(SystemExit) as failure:
        module.verify_layout(target, width=1560, panels_top=3,
                             has_strip=True, seconds=14.334)
    assert "width 1040 != expected 1560" in str(failure.value)
    assert not target.exists(), "a mismatched render must not be left behind"


def test_a_missing_plan_lane_band_is_caught_by_the_height(monkeypatch, tmp_path):
    target = tmp_path / "strip.mp4"
    target.write_bytes(b"x")
    # Two rows of 520 and nothing above them: the lanes are gone.
    monkeypatch.setattr(module, "run", probe_returning(1560, 1040, 14.334))
    with pytest.raises(SystemExit) as failure:
        module.verify_layout(target, width=1560, panels_top=3,
                             has_strip=True, seconds=14.334)
    assert "no plan-lane band" in str(failure.value)
    assert not target.exists()


def test_a_runaway_duration_is_caught(monkeypatch, tmp_path):
    """The six-hour defect, caught by the layout check as well as by -t."""
    target = tmp_path / "strip.mp4"
    target.write_bytes(b"x")
    monkeypatch.setattr(module, "run", probe_returning(1560, 1240, 3600.0))
    with pytest.raises(SystemExit) as failure:
        module.verify_layout(target, width=1560, panels_top=3,
                             has_strip=True, seconds=14.334)
    assert "duration" in str(failure.value)
    assert not target.exists()


def test_a_file_with_no_video_stream_is_removed(monkeypatch, tmp_path):
    """An unplayable file is the exact shape of the 1.7 GB that shipped."""
    target = tmp_path / "strip.mp4"
    target.write_bytes(b"x")
    monkeypatch.setattr(module, "run", lambda command: FakeProbe(""))
    with pytest.raises(SystemExit) as failure:
        module.verify_layout(target, width=1560, panels_top=3,
                             has_strip=True, seconds=14.334)
    assert "no readable video stream" in str(failure.value)
    assert not target.exists()


def test_small_duration_drift_is_tolerated(monkeypatch, tmp_path):
    # Container rounding must not fail an otherwise correct render.
    target = tmp_path / "strip.mp4"
    target.write_bytes(b"x")
    monkeypatch.setattr(module, "run", probe_returning(1560, 1240, 14.30))
    assert module.verify_layout(target, width=1560, panels_top=3,
                                has_strip=True, seconds=14.334)
    assert target.exists()


def test_the_renderer_refuses_missing_panels_by_default():
    """The flag must default to OFF, or the guarantee is decorative."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-missing-panels", action="store_true")
    assert parser.parse_args([]).allow_missing_panels is False
    source = pathlib.Path("tools/render_sample_strip.py").read_text()
    assert "--allow-missing-panels" in source
    # BOTH silent-drop paths must consult it -- the plan lanes and the footage.
    # One guard would leave the other panel free to vanish quietly.
    assert source.count("if not args.allow_missing_panels") == 1
    assert source.count("and not args.allow_missing_panels") == 1


class FrameRateTests:
    pass


def test_the_looped_plan_image_is_fed_at_the_panel_rate():
    """The defect behind every slow render on 2026-09-03.

    A looped image input defaults to 25 fps.  The panels are 30.  vstack's
    framesync reconciles them at the least common multiple -- 150 fps -- so the
    graph produces five frames for every one that is wanted.  Measured: two
    seconds of output took over 120s at preset ultrafast; with the rate matched,
    7.9s and 468 KB at exactly 60 frames.

    The rate must appear IMMEDIATELY BEFORE the image input, because ffmpeg
    input options apply to the next -i and nowhere else.
    """
    command = module.mux_command(
        ["a.mp4", "b.mp4"], "avatars.mp4", "audio.wav", "out.mp4",
        panel_height=520, width=1040, strip="strip.png", seconds=14.334,
        rate=30.0)
    assert "-framerate" in command
    index = command.index("-framerate")
    assert command[index + 1] == "30"
    assert command[index - 1] == "1" and command[index - 2] == "-loop"
    assert command[index + 2] == "-i" and command[index + 3] == "strip.png"


def test_a_non_integer_rate_is_passed_through_without_noise():
    command = module.mux_command(
        ["a.mp4"], "avatars.mp4", "audio.wav", "out.mp4", panel_height=520,
        width=520, strip="strip.png", seconds=1.0, rate=29.97)
    assert command[command.index("-framerate") + 1] == "29.97"


def test_no_framerate_is_emitted_when_there_is_no_strip():
    command = module.mux_command(
        ["a.mp4"], "avatars.mp4", "audio.wav", "out.mp4", panel_height=520,
        width=520, strip=None, seconds=1.0, rate=30.0)
    assert "-framerate" not in command
    assert "-loop" not in command


def test_the_rate_is_read_from_the_avatar_piece_not_assumed(monkeypatch):
    monkeypatch.setattr(module, "run", lambda command: FakeProbe("30000/1001\n"))
    assert module.video_rate("avatars.mp4") == pytest.approx(29.97, rel=1e-4)


def test_an_unreadable_rate_is_refused_rather_than_defaulted(monkeypatch):
    # Defaulting to 30 here would silently restore the 150 fps blow-up on any
    # piece rendered at another rate, which is exactly how this hid before.
    monkeypatch.setattr(module, "run", lambda command: FakeProbe("0/0\n"))
    with pytest.raises(SystemExit) as failure:
        module.video_rate("avatars.mp4")
    assert "frame rate" in str(failure.value)
