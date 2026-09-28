"""The video path that fed two frames, and the schema that had nowhere to put rhythm.

Both defects here shipped and neither was visible in any artifact.

The video path silently resampled whatever it was handed down to Qwen3-VL's
default two frames per second, so 6 frames and 32 frames produced byte-identical
input.  A caption written from two frames reads exactly like a caption written
from thirty-two, and the one artifact that could have settled which had been
used -- ``runs/genre_gate_video.json`` -- did not record its own input mode.

The schema had one single-valued ``dynamics`` axis carrying both intensity and
fluidity, so a movement could not be recorded as explosive *and* smooth, and no
axis at all for rhythm.  v1 stays the default because the published corpus was
captioned with it and has to stay reproducible.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools import caption_segments_vlm as caption  # noqa: E402
from tools.run_wild_stage_g_m3a_oss import m3a_keys  # noqa: E402
from tools.run_wild_stage_g_m3b_oss import m3b_keys  # noqa: E402


class _Grid:
    def __init__(self, rows):
        self._rows = rows

    def tolist(self):
        return self._rows


def test_a_video_batch_the_processor_shrank_is_refused():
    """t must be ceil(frames/2); two temporal positions for 32 frames is the bug."""
    with pytest.raises(caption.CaptionError) as raised:
        caption.check_video_frames_kept({"video_grid_thw": _Grid([[2, 28, 28]])}, [32])
    assert "resampled" in str(raised.value)


def test_the_measured_packing_rule_passes():
    for frames, temporal in ((4, 2), (6, 3), (8, 4), (16, 8), (31, 16), (32, 16)):
        caption.check_video_frames_kept(
            {"video_grid_thw": _Grid([[temporal, 28, 28]])}, [frames])


def test_no_video_grid_at_all_is_refused_rather_than_passed():
    with pytest.raises(caption.CaptionError):
        caption.check_video_frames_kept({}, [32])


def test_v1_is_still_the_default_and_still_reads_the_same():
    assert caption.active_schema()["fields"] == (
        "body_action", "arms", "legs", "level", "travel", "dynamics")
    assert caption.active_schema()["version"].endswith("v1")


def test_v2_splits_the_axis_v1_could_not_and_leaves_out_the_one_it_cannot_fill(
        monkeypatch):
    monkeypatch.setattr(caption, "_schema", caption.SCHEMAS["v2"])
    fields = caption.active_schema()["fields"]

    # The paper names path, rhythm, intensity and fluidity.  v1 had one axis for
    # intensity and fluidity together; v2 splits them.  Rhythm is deliberately
    # absent: measured on 255 segments it read at chance from stills and from
    # video alike, and its sharpest ruler read it backwards, so shipping the
    # axis would put a value in the vocabulary that nothing can fill.
    assert "rhythm" not in fields
    assert "rhythm" in caption.SCHEMAS["v2draft"]["fields"]
    assert "intensity" in fields and "fluidity" in fields
    assert "dynamics" not in fields
    assert "travel" in fields                     # path, unchanged

    sentence = caption.caption_sentence(
        {"body_action": "kick", "arms": "raised", "legs": "lifted",
         "level": "middle", "travel": "in_place", "intensity": "explosive",
         "fluidity": "sharp"})
    assert "explosive and sharp" in sentence
    assert "rhythm" not in sentence


def test_v2_can_record_a_movement_that_is_both_forceful_and_smooth(monkeypatch):
    """The combination v1 cannot express, which is why the axis was split."""
    monkeypatch.setattr(caption, "_schema", caption.SCHEMAS["v2"])
    vocabulary = caption.active_schema()["vocabulary"]
    assert "explosive" in vocabulary["intensity"]
    assert "flowing" in vocabulary["fluidity"]
    # In v1 both words live on one axis, so one excludes the other.
    assert "explosive" in caption.VOCABULARY["dynamics"]
    assert "smooth" in caption.VOCABULARY["dynamics"]


def test_the_rejected_rhythm_draft_is_kept_and_is_motion_only(monkeypatch):
    """This captioner is given frames and no audio.

    A value that can only be judged against a beat -- syncopation -- would be
    unanswerable and unscoreable, so every value has to be readable from the
    movement alone.
    """
    monkeypatch.setattr(caption, "_schema", caption.SCHEMAS["v2draft"])
    values = caption.active_schema()["vocabulary"]["rhythm"]
    assert "syncopated" not in values
    assert set(values) == {"steady", "pulsing", "accelerating", "decelerating",
                           "stop_and_go", "held"}


def test_a_second_caption_generation_does_not_overwrite_the_first():
    first = m3a_keys("clean5b5", "")
    second = m3a_keys("clean5b5", "v2video")

    assert first["captions"] != second["captions"]
    assert first["parts"] != second["parts"]
    # And the original names are untouched, so the published corpus still
    # resolves for every reader written before the suffix existed.
    assert first["captions"].endswith("captions.jsonl")


def test_a_caption_generation_keys_every_m3b_output_not_only_the_captions():
    """Two generations are two vocabularies from the same segments.

    Sharing an output name in a store that overwrites and cannot delete means
    the second run replaces the first, under a name that says only which arm it
    was.
    """
    plain = m3b_keys("clean5b5", "wild_v4")
    stamped = m3b_keys("clean5b5", "wild_v4", "v2video")

    for name in ("subprototypes", "cells", "bundle", "bundle_nollm", "report",
                 "captions"):
        assert plain[name] != stamped[name], name
    # The borrowed input is a property of the ingest, not of the generation.
    assert plain["group_keys"] == stamped["group_keys"]


def test_field_agreement_follows_the_captions_own_schema():
    from tools.summarize_subprototypes_llm import agreement

    a = {"body_action": "kick", "intensity": "explosive", "fluidity": "sharp",
         "rhythm": "stop_and_go"}
    b = {"body_action": "kick", "intensity": "gentle", "fluidity": "flowing",
         "rhythm": "steady"}

    # One of four, not one of six with three v1 axes silently counted absent.
    assert agreement(a, b) == pytest.approx(0.25)
