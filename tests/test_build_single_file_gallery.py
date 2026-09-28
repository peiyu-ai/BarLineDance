"""The single-file gallery's one job: travel intact, or refuse to be built."""

from __future__ import annotations

import base64
import json
import pathlib
import subprocess
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.build_single_file_gallery import (SILENCE_FLOOR_DB, GalleryError,
                                             audio_mean_volume, build)


def _render(path: pathlib.Path, *, seconds: float = 1.0, silent: bool = False,
            audio: bool = True) -> None:
    """A tiny real mp4, optionally with a real tone, digital silence, or no track."""
    argv = ["ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc=size=64x64:rate=10:duration={}".format(seconds)]
    if audio:
        source = ("anullsrc=r=22050:cl=mono" if silent
                  else "sine=frequency=440:sample_rate=22050")
        argv += ["-f", "lavfi", "-i", "{}:duration={}".format(source, seconds)]
    argv += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-t", str(seconds)]
    argv += (["-c:a", "aac", "-shortest"] if audio else ["-an"])
    argv += [str(path)]
    subprocess.run(argv, check=True, capture_output=True)


def _gallery(tmp_path: pathlib.Path, names, **kwargs) -> pathlib.Path:
    root = tmp_path / "gallery"
    (root / "compare").mkdir(parents=True)
    entries = []
    for name in names:
        _render(root / "compare" / (name + ".mp4"), **kwargs)
        entries.append({
            "compare": "compare/{}.mp4".format(name),
            "rows": [
                {"label": "ground truth", "detail": "wild_v5:{}:clip000".format(name),
                 "seconds": 1.0, "diagnostics": {"frames": 30,
                                                 "median_joint_speed_m_per_s": 0.7}},
                {"label": "arm", "detail": "runs/arm", "seconds": 1.0,
                 "diagnostics": {"frames": 30, "median_joint_speed_m_per_s": 0.9}},
            ],
        })
    (root / "gallery.json").write_text(json.dumps({"entries": entries}), encoding="utf-8")
    return root


def test_the_videos_travel_inside_the_page(tmp_path):
    """The defect this exists for: a page that leaves its media behind.

    The rendered gallery links ``compare/<clip>.mp4`` by relative path, so the
    HTML alone shows neither picture nor sound and the reader cannot tell which
    half went missing.  Here the bytes are IN the file, so moving the file moves
    the dance and the music with it.
    """
    root = _gallery(tmp_path, ["a", "b"])
    out = tmp_path / "out" / "page.html"
    stats = build(root, out, "T", crf=32, limit=None, note="")
    page = out.read_text(encoding="utf-8")
    assert stats["clips"] == 2
    assert page.count("data:video/mp4;base64,") == 2
    assert "compare/a.mp4" not in page          # no relative reference survives
    # and the embedded payload is a real mp4, not a placeholder
    blob = page.split("data:video/mp4;base64,")[1].split('"')[0]
    assert base64.b64decode(blob)[4:8] == b"ftyp"


def test_a_silent_clip_stops_the_build_rather_than_shipping_quietly(tmp_path):
    """A well-formed AAC track carrying silence looks fine in every listing.

    The page exists to let someone judge dance against its music; silent, it is
    worth nothing, and nothing else in the pipeline can see the difference --
    ffprobe reports the stream and the byte count is plausible.
    """
    root = _gallery(tmp_path, ["a"], silent=True)
    with pytest.raises(GalleryError) as error:
        build(root, tmp_path / "page.html", "T", crf=32, limit=None, note="")
    assert "which is silence" in str(error.value)
    assert "compare/a.mp4" in str(error.value)


def test_a_clip_with_no_audio_track_is_reported_apart_from_a_silent_one(tmp_path):
    """"No music was muxed" and "the music is silent" are different defects."""
    root = _gallery(tmp_path, ["a"], audio=False)
    with pytest.raises(GalleryError) as error:
        build(root, tmp_path / "page.html", "T", crf=32, limit=None, note="")
    message = str(error.value)
    assert "1 clip(s) carry no audio stream" in message
    assert "0 carry a stream at or under" in message


def test_the_silence_floor_sits_between_two_measurements(tmp_path):
    """The gate this test already caught once, pinned so it cannot regress.

    The first version compared ``mean_volume == -inf`` and could never fire:
    AAC does not encode digital silence as silence, it encodes it at about
    -91 dB.  A gate that cannot fire reads exactly like a gate that passed, so
    the floor has to be a level, and it has to sit between a measured real clip
    and a measured silent one rather than being picked.
    """
    loud, quiet, mute = (tmp_path / n for n in ("loud.mp4", "quiet.mp4", "mute.mp4"))
    _render(loud)
    _render(quiet, silent=True)
    _render(mute, audio=False)

    audible = audio_mean_volume(loud)
    encoded_silence = audio_mean_volume(quiet)
    assert audio_mean_volume(mute) is None                   # absent, not silent

    # The failure the old gate had: encoded silence is a number, not -inf.
    assert encoded_silence > float("-inf")
    assert encoded_silence < SILENCE_FLOOR_DB < audible      # the floor separates them
    assert audible > SILENCE_FLOOR_DB + 30                   # and not by a hair


def test_a_sampler_difference_between_the_stacked_arms_is_printed(tmp_path):
    """Carried through from the rendered gallery, which refuses to hide it."""
    root = _gallery(tmp_path, ["a"])
    manifest = json.loads((root / "gallery.json").read_text(encoding="utf-8"))
    manifest["sampling_check"] = {"differing": {"plan_bar_grid": {"on": True, "off": False}}}
    (root / "gallery.json").write_text(json.dumps(manifest), encoding="utf-8")
    out = tmp_path / "page.html"
    build(root, out, "T", crf=32, limit=None, note="")
    page = out.read_text(encoding="utf-8")
    assert "sampled differently" in page
    assert "plan_bar_grid" in page


def test_the_plan_strip_names_filler_rather_than_leaving_it_blank(tmp_path):
    """Filler is the transition class, and it has to read as filler.

    A plan can carry many boundaries and still look under-danced, because the
    boundaries are between filler spans.  Measured over 120 held-out clips on
    2026-08-29 the two readings order the arms oppositely -- the ungridded arm
    has 60% MORE boundaries per second than the ground truth and 19% FEWER
    visible accents -- so the strip has to show both, and grey has to be
    labelled rather than merely uncoloured.
    """
    import numpy as np

    from tools.render_plan_strip import accents, filler_fraction, runs_of

    labels = np.array([0] * 30 + [7] * 30 + [0] * 30)
    assert filler_fraction(labels) == pytest.approx(2 / 3)
    assert [r[2] for r in runs_of(labels)] == [0, 7, 0]

    # The defect the first version of ``accents`` had: a percentile threshold
    # with no floor promotes a static body's numerical noise to the 90th
    # percentile, and every frame comes back an accent.
    still = np.zeros((60, 24, 3))
    assert len(accents(still)) == 0

    # Positive control: a body that stops sharply once must still register.
    moving = np.zeros((60, 24, 3))
    moving[:30, :, 0] = np.arange(30)[:, None] * 0.05    # travels, then halts
    moving[30:, :, 0] = moving[29, :, 0]
    assert len(accents(moving)) >= 1


def test_a_row_whose_plan_is_missing_is_dropped_not_drawn_as_filler(tmp_path):
    """An all-grey lane and an absent lane must not look the same on the page."""
    from tools.build_single_file_gallery import plan_rows

    entry = {"rows": [{"label": "ground truth", "detail": "wild_v5:1:clip000"},
                      {"label": "arm", "detail": str(tmp_path / "nowhere")}]}
    clip, rows = plan_rows(entry, None, None)
    assert clip == "wild_v5:1:clip000"
    assert rows == []                       # nothing invented for either row


def test_the_uv_unwrap_matches_the_body_it_textures(tmp_path):
    """A texture is only appearance if it lands on the same geometry.

    SMPL's unwrap splits seams, so 6,890 vertices carry 7,576 UV coordinates and
    the mesh has to be un-welded to be textured.  That rebuild is where a
    textured render could quietly stop being the motion the model produced, so
    the counts are pinned against the body model rather than trusted: 13,776
    faces, and every corner index inside the vertex count.
    """
    import pathlib as _pathlib

    from tools.render_avatar_video import load_uv

    if not _pathlib.Path("third_party/smpl_models/uv/smpl_uv.obj").is_file():
        pytest.skip("UV unwrap not vendored here")
    coords, corner_v, corner_t = load_uv()
    assert corner_v.shape == (13776, 3)          # SMPL's face count
    assert corner_t.shape == corner_v.shape
    assert coords.shape[1] == 2
    assert int(corner_v.max()) == 6889           # SMPL's vertex count - 1
    assert int(corner_t.max()) == len(coords) - 1
    assert len(coords) > 6890                    # seams really are split
