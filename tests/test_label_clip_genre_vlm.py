"""Tests for the per-clip genre labeller and its AIST reference gate.

The parsing and scoring are pure, so they are testable without a GPU or a
model.  What they pin is the distinction the tool exists to protect: a genre the
model actually asserted, versus one nobody said.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.label_clip_genre_vlm import (GENRES, OTHER, aist_truth,  # noqa: E402
                                        build_prompt, confusion,
                                        frame_positions, parse_genre)


def test_parses_a_clean_reply():
    assert parse_genre('{"genre": "break", "confidence": "high"}') == ("break", "high")


def test_parses_a_reply_wrapped_in_prose():
    text = 'Sure! {"genre":"LA_Hip_Hop","confidence":"low"} hope that helps'
    assert parse_genre(text) == ("la_hip_hop", "low")


def test_refuses_a_genre_outside_the_vocabulary():
    # "tango" is a real answer to a different question.  Accepting it would put
    # a class in the corpus that --genre-split has no bucket for.
    assert parse_genre('{"genre": "tango"}') == (None, None)


def test_refuses_a_reply_with_no_json():
    assert parse_genre("I think it is breaking") == (None, None)


def test_unparsed_is_not_other():
    """The two mean opposite things and must not collapse.

    ``other`` is the model saying "this is not one of the ten"; ``None`` is the
    model not answering.  Folding the second into the first would report a
    confident negative where there was silence.
    """
    silent, _ = parse_genre("...")
    stated, _ = parse_genre('{"genre": "other"}')
    assert silent is None
    assert stated == OTHER


def test_aist_truth_reads_the_sequence_id():
    assert aist_truth("gBR_sBM_c01_d04_mBR0_ch01") == "break"
    assert aist_truth("gJS_sFM_c01_d01_mJS0_ch01") == "street_jazz"


def test_aist_truth_is_absent_for_wild_clips():
    # Wild ids carry no genre; returning something here would invent ground
    # truth for the corpus that has none.
    assert aist_truth("7203621409615088908__clip000") is None


def test_frame_positions_span_the_clip_without_touching_the_ends():
    positions = frame_positions(500, 8)
    assert len(positions) == 8
    assert positions == sorted(positions)
    assert positions[0] > 0 and positions[-1] < 499


def test_frame_positions_handles_a_clip_shorter_than_the_sample():
    assert frame_positions(5, 8) == [0, 1, 2, 3, 4]


def test_confusion_scores_only_what_the_model_answered():
    records = [
        {"truth": "break", "genre": "break"},
        {"truth": "break", "genre": "pop"},
        {"truth": "pop", "genre": "pop"},
        {"truth": "house", "genre": None},          # no answer: not a wrong answer
    ]
    report = confusion(records)
    assert report["judged"] == 3
    assert report["unparsed"] == 1
    assert report["accuracy"] == round(2 / 3, 4)
    assert report["per_genre"]["break"]["recall"] == 0.5


def test_confusion_reports_chance_for_the_class_count():
    assert confusion([])["chance"] == round(1.0 / len(GENRES), 4)


def test_prompt_offers_every_genre_and_the_escape():
    prompt = build_prompt()
    for genre in GENRES:
        assert genre in prompt
    assert OTHER in prompt
