"""Tests for the VLM segment captioner (paper M3, caption step)."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.caption_segments_vlm import (  # noqa: E402
    FIELDS,
    VOCABULARY,
    caption_sentence,
    parse_caption,
    sample_frame_positions,
    shard_of,
    video_path_for,
)


def test_sample_positions_include_both_endpoints():
    positions = sample_frame_positions(10, 40, 6)
    assert positions[0] == 10
    assert positions[-1] == 39
    assert len(positions) == 6
    assert positions == sorted(positions)


def test_short_segment_returns_every_frame_not_repeats():
    assert sample_frame_positions(4, 8, 6) == [4, 5, 6, 7]


def test_single_frame_request_does_not_divide_by_zero():
    assert sample_frame_positions(0, 100, 1) == [0]


def test_sharding_is_stable_across_processes():
    """The property that matters: a subprocess must agree with this one.

    ``hash()`` would pass an in-process check and fail this one, which is the
    whole reason the function exists.
    """
    import subprocess

    key = ("tiktok:123:clip000", 40, 90)
    mine = shard_of(key, 8)
    code = (
        "import pathlib,sys;"
        "sys.path.insert(0, {!r});"
        "from tools.caption_segments_vlm import shard_of;"
        "print(shard_of(('tiktok:123:clip000', 40, 90), 8))"
    ).format(str(pathlib.Path(__file__).resolve().parents[1]))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={"PYTHONHASHSEED": "1", "PATH": "/usr/bin:/bin"})
    assert out.returncode == 0, out.stderr
    assert int(out.stdout.strip()) == mine


def test_shards_partition_without_overlap_or_loss():
    keys = [("tiktok:{}:clip000".format(i), i, i + 30) for i in range(500)]
    owned = [[k for k in keys if shard_of(k, 8) == s] for s in range(8)]
    flat = [k for group in owned for k in group]
    assert sorted(flat) == sorted(keys)
    assert len(set(flat)) == len(keys)


def test_parse_keeps_only_vocabulary_values():
    reply = ('{"body_action": "spin", "arms": "raised", "legs": "apart", '
             '"level": "high", "travel": "rotating", "dynamics": "sharp", '
             '"summary": "fast spin"}')
    parsed = parse_caption(reply)
    assert parsed["body_action"] == "spin"
    assert parsed["summary"] == "fast spin"


def test_off_vocabulary_value_becomes_unspecified_not_kept():
    """An invented value would be a unique token that isolates its segment."""
    reply = ('{"body_action": "moonwalk_backslide", "arms": "raised", '
             '"legs": "apart", "level": "high", "travel": "rotating", '
             '"dynamics": "sharp", "summary": "x"}')
    parsed = parse_caption(reply)
    assert parsed["body_action"] == "unspecified"
    assert parsed["arms"] == "raised"


def test_parse_tolerates_prose_and_fences_around_the_json():
    reply = ('Sure! Here is the description:\n```json\n'
             '{"body_action": "jump", "arms": "down", "legs": "together", '
             '"level": "low", "travel": "in_place", "dynamics": "explosive", '
             '"summary": "small hop"}\n```\nHope that helps.')
    parsed = parse_caption(reply)
    assert parsed is not None
    assert parsed["body_action"] == "jump"


def test_parse_returns_none_on_unusable_reply():
    assert parse_caption("I cannot see the images.") is None
    assert parse_caption("{not json at all}") is None


def test_missing_fields_do_not_raise():
    parsed = parse_caption('{"body_action": "step", "summary": "a step"}')
    assert parsed["arms"] == "unspecified"
    assert all(field in parsed for field in FIELDS)


def test_identical_fields_render_identical_sentences():
    """Two samples of the same movement must embed to the same point."""
    a = parse_caption('{"body_action": "step", "arms": "down", "legs": "apart", '
                      '"level": "middle", "travel": "sideways", "dynamics": "smooth", '
                      '"summary": "side step"}')
    b = parse_caption('{"body_action": "step", "arms": "down", "legs": "apart", '
                      '"level": "middle", "travel": "sideways", "dynamics": "smooth", '
                      '"summary": "stepping to the side, casually"}')
    assert caption_sentence(a) == caption_sentence(b)


def test_different_movements_render_different_sentences():
    a = parse_caption('{"body_action": "jump", "arms": "raised", "legs": "apart", '
                      '"level": "high", "travel": "in_place", "dynamics": "explosive", '
                      '"summary": "x"}')
    b = parse_caption('{"body_action": "slide", "arms": "down", "legs": "together", '
                      '"level": "low", "travel": "sideways", "dynamics": "smooth", '
                      '"summary": "x"}')
    assert caption_sentence(a) != caption_sentence(b)


def test_sentence_has_no_underscores_left_in_it():
    parsed = parse_caption('{"body_action": "step", "arms": "framing_face", '
                           '"legs": "apart", "level": "high", "travel": "in_place", '
                           '"dynamics": "sharp", "summary": "x"}')
    assert "_" not in caption_sentence(parsed)


def test_vocabulary_values_are_lowercase_and_unique():
    for field, values in VOCABULARY.items():
        assert len(values) == len(set(values)), field
        assert all(value == value.lower() for value in values), field


def test_video_path_maps_recording_id_to_file(tmp_path):
    (tmp_path / "6796814277069114624__clip000.mp4").write_bytes(b"x")
    found = video_path_for("tiktok:6796814277069114624:clip000", tmp_path)
    assert found is not None and found.name == "6796814277069114624__clip000.mp4"


def test_video_path_returns_none_when_absent(tmp_path):
    assert video_path_for("tiktok:999:clip000", tmp_path) is None
    assert video_path_for("malformed", tmp_path) is None


def test_aist_cAll_resolves_to_a_physical_camera(tmp_path):
    """``cAll`` is AIST++'s annotation-space name and no file ever carries it.

    Matching it literally returned None for all 12,946 AIST segments, and the
    run reported that as ``no_video`` rather than failing -- a whole corpus
    silently skipped.
    """
    (tmp_path / "gBR_sBM_c01_d04_mBR0_ch01.mp4").write_bytes(b"x")
    found = video_path_for("aistpp/gBR_sBM_cAll_d04_mBR0_ch01", tmp_path)
    assert found is not None and found.name == "gBR_sBM_c01_d04_mBR0_ch01.mp4"


def test_the_camera_choice_is_deterministic_when_several_exist(tmp_path):
    # Nine cameras see the same performance, so any is a correct answer -- but
    # two shards must not pick different ones, or the same segment gets two
    # captions and load_captions keeps whichever landed first.
    for camera in ("c05", "c01", "c09"):
        (tmp_path / "gBR_sBM_{}_d04_mBR0_ch01.mp4".format(camera)).write_bytes(b"x")
    found = video_path_for("aistpp/gBR_sBM_cAll_d04_mBR0_ch01", tmp_path)
    assert found is not None and "_c01_" in found.name


def test_an_exact_name_still_wins_over_the_camera_glob(tmp_path):
    (tmp_path / "gBR_sBM_cAll_d04_mBR0_ch01.mp4").write_bytes(b"x")
    (tmp_path / "gBR_sBM_c01_d04_mBR0_ch01.mp4").write_bytes(b"x")
    found = video_path_for("aistpp/gBR_sBM_cAll_d04_mBR0_ch01", tmp_path)
    assert found is not None and "_cAll_" in found.name


def test_person_boxes_and_a_frame_ratio_are_refused_together(tmp_path):
    """The boxes are indexed on the source grid; scaling frames moves off it.

    Nothing downstream could catch the pair: the crop would land on a real
    dancer at the wrong moment, and the caption would read as valid.
    """
    from tools.caption_segments_vlm import CaptionError, run

    with pytest.raises(CaptionError):
        run(labels_dir=tmp_path, bundle=tmp_path, video_dir=tmp_path,
            model_dir=tmp_path, output=tmp_path / "out.jsonl", shard=0, num_shards=1,
            frames_per_segment=8, max_side=448, min_frames=4, limit=None,
            device="cpu", max_new_tokens=64, dry_run=True,
            person_boxes=tmp_path, video_per_motion=2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_model_name_looks_past_a_revision_directory(tmp_path):
    """The OSS mirror stores weights under 'main/', and that name would
    otherwise be what every label row records as standing in for Gemini."""
    from tools.caption_segments_vlm import model_name

    revision = tmp_path / "Qwen3-VL-30B-A3B-Instruct" / "main"
    revision.mkdir(parents=True)
    assert model_name(revision) == "Qwen3-VL-30B-A3B-Instruct"


def test_model_name_keeps_a_plain_directory_name(tmp_path):
    from tools.caption_segments_vlm import model_name

    plain = tmp_path / "Qwen2.5-VL-7B-Instruct"
    plain.mkdir()
    assert model_name(plain) == "Qwen2.5-VL-7B-Instruct"


def test_segment_box_is_one_box_for_the_whole_segment_not_per_frame():
    """A per-frame crop re-centres the dancer and deletes the travel the
    caption is supposed to describe."""
    from tools.caption_segments_vlm import segment_box

    boxes = np.array([[100.0, 100.0, 200.0, 400.0],
                      [150.0, 100.0, 250.0, 400.0],
                      [200.0, 100.0, 300.0, 400.0]])
    box = segment_box(boxes, [0, 1, 2], margin=0.0, width=1000, height=1000)
    assert box == (100, 100, 300, 400)


def test_segment_box_adds_margin_and_clamps_to_the_frame():
    from tools.caption_segments_vlm import segment_box

    boxes = np.array([[10.0, 10.0, 110.0, 210.0]])
    box = segment_box(boxes, [0], margin=0.5, width=120, height=250)
    assert box == (0, 0, 120, 250)


def test_segment_box_ignores_frame_indices_beyond_the_track():
    from tools.caption_segments_vlm import segment_box

    boxes = np.array([[10.0, 20.0, 60.0, 120.0]])
    assert segment_box(boxes, [0, 99], margin=0.0, width=200, height=200) == (10, 20, 60, 120)
    assert segment_box(boxes, [99], margin=0.0, width=200, height=200) is None


def test_segment_box_rejects_a_degenerate_track():
    from tools.caption_segments_vlm import segment_box

    boxes = np.array([[50.0, 50.0, 50.0, 50.0]])
    assert segment_box(boxes, [0], margin=0.25, width=100, height=100) is None


def test_person_boxes_returns_none_when_the_track_is_absent(tmp_path):
    from tools.caption_segments_vlm import person_boxes_for

    assert person_boxes_for("tiktok:999:clip000", tmp_path) is None
    assert person_boxes_for("malformed", tmp_path) is None
