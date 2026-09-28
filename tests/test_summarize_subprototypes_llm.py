"""Tests for the paper's summarizing LLM step (M3, sub-prototype formation).

The model is stubbed throughout.  What is under test is the loop the paper
specifies -- iterate, remove what was grouped, stop on a small remainder -- and
its behaviour when the model misbehaves, which on a 100-group corpus it will.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.summarize_subprototypes_llm import (  # noqa: E402
    agreement,
    group_key,
    parse_reply,
    render_items,
    run,
    summarize_group,
)


def entry(caption, count, **fields):
    base = {"body_action": "step", "arms": "down", "legs": "apart",
            "level": "middle", "travel": "in_place", "dynamics": "smooth"}
    base.update(fields)
    return (caption, count, base)


class ScriptedLLM:
    """Replies from a list; records the prompts it was given."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else "no more"


def test_parse_reply_keeps_only_offered_line_numbers():
    """A hallucinated id would drag an unrelated caption into the group."""
    parsed = parse_reply('{"members": [0, 2, 99], "tag": "side steps"}', [0, 1, 2])
    assert parsed == ([0, 2], "side steps")


def test_parse_reply_rejects_a_subset_of_one():
    """Otherwise the loop 'progresses' one caption at a time and never ends."""
    assert parse_reply('{"members": [1], "tag": "x"}', [0, 1, 2]) is None


def test_parse_reply_rejects_unusable_text():
    assert parse_reply("I am not sure how to group these.", [0, 1]) is None
    assert parse_reply('{"members": []}', [0, 1]) is None


def test_parse_reply_survives_prose_and_fences():
    parsed = parse_reply('Sure:\n```json\n{"members": [0,1], "tag": "spins"}\n```', [0, 1])
    assert parsed == ([0, 1], "spins")


def test_parse_reply_dedupes_repeated_members():
    parsed = parse_reply('{"members": [1, 1, 2], "tag": "t"}', [1, 2])
    assert parsed == ([1, 2], "t")


def test_missing_tag_becomes_untagged_not_empty():
    parsed = parse_reply('{"members": [0, 1]}', [0, 1])
    assert parsed == ([0, 1], "untagged")


def test_loop_stops_once_the_remainder_is_below_threshold():
    entries = [entry("a", 40), entry("b", 40), entry("c", 10), entry("d", 10)]
    llm = ScriptedLLM(['{"members": [0, 1], "tag": "the big one"}'])
    result = summarize_group(entries, llm, residual_threshold=0.25,
                             max_rounds=20, subset_cap=12)
    # 20 of 100 segments remain, below the 25% threshold, so the LLM is asked
    # exactly once and the rest is attached without further calls.
    assert len(llm.prompts) == 1
    assert result["stopped_because"] == "residual_below_threshold"
    assert result["residual_segments"] == 20


def test_every_segment_survives_the_loop():
    """No segment may be dropped: each one is a frame span that needs a label."""
    entries = [entry("a", 5), entry("b", 7), entry("c", 3), entry("d", 11)]
    llm = ScriptedLLM(['{"members": [0, 1], "tag": "one"}',
                       '{"members": [2, 3], "tag": "two"}'])
    result = summarize_group(entries, llm, residual_threshold=0.0,
                             max_rounds=20, subset_cap=12)
    assert sum(sub["segments"] for sub in result["subprototypes"]) == 26


def test_residual_joins_the_subprototype_it_agrees_with():
    entries = [entry("jump high", 10, body_action="jump", level="high"),
               entry("jump higher", 10, body_action="jump", level="high"),
               entry("slide low", 10, body_action="slide", level="low"),
               entry("slide lower", 10, body_action="slide", level="low"),
               entry("slide sideways", 1, body_action="slide", level="low")]
    llm = ScriptedLLM(['{"members": [0, 1], "tag": "jumps"}',
                       '{"members": [2, 3], "tag": "slides"}'])
    result = summarize_group(entries, llm, residual_threshold=0.05,
                             max_rounds=20, subset_cap=12)
    slides = [sub for sub in result["subprototypes"] if sub["tag"] == "slides"][0]
    assert "slide sideways" in slides["captions"]
    assert slides["residual_segments"] == 1


def test_a_refusing_model_leaves_the_group_whole_not_scattered():
    entries = [entry("a", 10), entry("b", 10), entry("c", 10)]
    llm = ScriptedLLM(["I cannot help with that."])
    result = summarize_group(entries, llm, residual_threshold=0.0,
                             max_rounds=20, subset_cap=12)
    assert result["stopped_because"] == "llm_returned_no_usable_subset"
    assert len(result["subprototypes"]) == 1
    assert result["subprototypes"][0]["source"] == "residual"
    assert result["subprototypes"][0]["segments"] == 30


def test_max_rounds_bounds_a_model_that_grabs_two_lines_at_a_time():
    entries = [entry(str(i), 1) for i in range(40)]
    llm = ScriptedLLM(['{{"members": [{}, {}], "tag": "t{}"}}'.format(2 * i, 2 * i + 1, i)
                       for i in range(20)])
    result = summarize_group(entries, llm, residual_threshold=0.0,
                             max_rounds=3, subset_cap=12)
    assert result["rounds"] == 3
    assert result["stopped_because"] == "max_rounds"
    assert sum(sub["segments"] for sub in result["subprototypes"]) == 40


def test_line_numbers_in_the_prompt_match_the_ids_the_reply_uses():
    """The ids are positions; a mismatch would silently regroup the captions."""
    entries = [entry("alpha", 3), entry("beta", 2)]
    llm = ScriptedLLM(['{"members": [0, 1], "tag": "t"}'])
    summarize_group(entries, llm, residual_threshold=0.0, max_rounds=5, subset_cap=12)
    assert "0. alpha [3 segments]" in llm.prompts[0]
    assert "1. beta [2 segments]" in llm.prompts[0]


def test_rendered_items_are_stable_between_rounds_for_survivors():
    pool = [(0, "alpha", 3), (2, "gamma", 1)]
    assert render_items(pool) == "0. alpha [3 segments]\n2. gamma [1 segments]"


def test_agreement_is_the_share_of_matching_fields():
    a = {"body_action": "step", "arms": "down", "legs": "apart",
         "level": "middle", "travel": "in_place", "dynamics": "smooth"}
    b = dict(a, body_action="jump")
    assert agreement(a, a) == 1.0
    assert agreement(a, b) == pytest.approx(5 / 6)
    assert agreement(a, {}) == 0.0


def test_group_key_uses_the_genre_map_and_falls_back_to_unknown():
    row = {"prototype": 7, "recording_id": "tiktok:1:clip000"}
    assert group_key(row, {"tiktok:1:clip000": "gLH"}) == (7, "gLH")
    assert group_key(row, {}) == (7, "?")


def test_run_writes_a_report_and_groups_by_prototype(tmp_path):
    rows = []
    for prototype, caption in ((1, "a"), (1, "b"), (2, "c"), (2, "d")):
        for index in range(5):
            rows.append({"recording_id": "r{}".format(index), "start": index,
                         "end": index + 10, "prototype": prototype,
                         "caption": caption, "fields": {"body_action": caption}})
    path = tmp_path / "captions.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    llm = ScriptedLLM(['{"members": [0, 1], "tag": "one"}',
                       '{"members": [0, 1], "tag": "two"}'])
    report = run(captions=path, output=tmp_path / "subs.json", ask=llm,
                 model_name="stub", residual_threshold=0.0, max_rounds=5)
    assert report["groups"] == 2
    assert report["total_subprototypes"] == 2
    assert report["segments_total"] == 20
    assert json.loads((tmp_path / "subs.json").read_text())["model"] == "stub"


def test_llm_share_excludes_the_residue_it_did_not_group(tmp_path):
    rows = [{"recording_id": "r", "start": 0, "end": 10, "prototype": 1,
             "caption": caption, "fields": {"body_action": caption}}
            for caption in ("a", "a", "b", "b", "c")]
    path = tmp_path / "captions.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    llm = ScriptedLLM(['{"members": [0, 1], "tag": "one"}'])
    report = run(captions=path, output=tmp_path / "subs.json", ask=llm,
                 model_name="stub", residual_threshold=0.25, max_rounds=5)
    assert report["segments_total"] == 5
    assert report["segments_in_llm_formed_subprototypes"] == 4


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_size_floor_folds_tiny_subprototypes_into_a_sibling():
    """Fig. 4c puts only 15% of sub-prototypes under 20 samples; a run that
    produces far more has split past the point of trainability."""
    from tools.summarize_subprototypes_llm import merge_small

    entries = [entry("jump high", 30, body_action="jump"),
               entry("slide low", 30, body_action="slide"),
               entry("jump higher", 2, body_action="jump")]
    subs = [{"tag": "jumps", "captions": ["jump high"], "segments": 30, "source": "llm"},
            {"tag": "slides", "captions": ["slide low"], "segments": 30, "source": "llm"},
            {"tag": "tiny", "captions": ["jump higher"], "segments": 2, "source": "llm"}]
    merged = merge_small(subs, entries, minimum=20)
    assert merged == 1
    assert len(subs) == 2
    jumps = [s for s in subs if s["tag"] == "jumps"][0]
    assert jumps["segments"] == 32
    assert jumps["merged_segments"] == 2


def test_size_floor_off_by_default_leaves_the_split_alone():
    from tools.summarize_subprototypes_llm import merge_small

    subs = [{"tag": "a", "captions": ["x"], "segments": 1, "source": "llm"},
            {"tag": "b", "captions": ["y"], "segments": 1, "source": "llm"}]
    assert merge_small(subs, [entry("x", 1), entry("y", 1)], minimum=0) == 0
    assert len(subs) == 2


def test_size_floor_never_empties_a_group():
    """Every segment still needs a label, however small the group is."""
    from tools.summarize_subprototypes_llm import merge_small

    subs = [{"tag": "a", "captions": ["x"], "segments": 1, "source": "llm"},
            {"tag": "b", "captions": ["y"], "segments": 1, "source": "llm"}]
    merge_small(subs, [entry("x", 1), entry("y", 1)], minimum=100)
    assert len(subs) == 1
    assert subs[0]["segments"] == 2


def test_size_floor_conserves_segments():
    from tools.summarize_subprototypes_llm import merge_small

    entries = [entry(str(i), 5 * (i + 1)) for i in range(6)]
    subs = [{"tag": str(i), "captions": [str(i)], "segments": 5 * (i + 1), "source": "llm"}
            for i in range(6)]
    before = sum(s["segments"] for s in subs)
    merge_small(subs, entries, minimum=20)
    assert sum(s["segments"] for s in subs) == before


def four_prototype_captions(tmp_path):
    """Four cells, five segments each, so shards and resumes are countable."""
    rows = []
    for prototype in (1, 2, 3, 4):
        for caption in ("a", "b"):
            for index in range(5):
                rows.append({"recording_id": "r{}".format(index), "start": index,
                             "end": index + 10, "prototype": prototype,
                             "caption": "{}{}".format(caption, prototype),
                             "fields": {"body_action": caption}})
    path = tmp_path / "captions.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_shards_partition_the_cells_without_overlap(tmp_path):
    """Cells are independent -- the paper's loop never looks outside one -- so a
    corpus that needs hours can be split.  What must hold is that the shards
    together cover every cell exactly once."""
    path = four_prototype_captions(tmp_path)
    seen = []
    for shard in range(2):
        llm = ScriptedLLM(['{"members": [0, 1], "tag": "t"}'] * 4)
        report = run(captions=path, output=tmp_path / "s{}.json".format(shard), ask=llm,
                     model_name="stub", residual_threshold=0.0, max_rounds=5,
                     shard=shard, num_shards=2)
        seen.extend(group["prototype"] for group in report["groups_detail"])
    assert sorted(seen) == [1, 2, 3, 4]


def test_a_checkpointed_cell_is_not_bought_twice(tmp_path):
    """The report is written only after the last cell, so without this an
    interruption loses every call the run made.  ``run_aist_m3_llm.sh`` claimed
    the summarizer was resumable and only the captioner was."""
    path = four_prototype_captions(tmp_path)
    checkpoint = tmp_path / "cells.jsonl"

    first = ScriptedLLM(['{"members": [0, 1], "tag": "t"}'] * 2)
    run(captions=path, output=tmp_path / "a.json", ask=first, model_name="stub",
        residual_threshold=0.0, max_rounds=5, limit_groups=2, checkpoint=checkpoint)
    assert len(checkpoint.read_text(encoding="utf-8").strip().splitlines()) == 2

    # Same two cells again: every reply must come from the checkpoint.
    second = ScriptedLLM([])
    report = run(captions=path, output=tmp_path / "b.json", ask=second,
                 model_name="stub", residual_threshold=0.0, max_rounds=5,
                 limit_groups=2, checkpoint=checkpoint)
    assert second.prompts == []
    assert report["groups"] == 2


def test_a_truncated_checkpoint_row_is_dropped_not_trusted(tmp_path):
    """A row half-written when the process died is not a grouping."""
    path = four_prototype_captions(tmp_path)
    checkpoint = tmp_path / "cells.jsonl"
    first = ScriptedLLM(['{"members": [0, 1], "tag": "t"}'])
    run(captions=path, output=tmp_path / "a.json", ask=first, model_name="stub",
        residual_threshold=0.0, max_rounds=5, limit_groups=1, checkpoint=checkpoint)
    with checkpoint.open("a", encoding="utf-8") as handle:
        handle.write('{"prototype": 2, "genre": "?", "subproto')

    second = ScriptedLLM(['{"members": [0, 1], "tag": "t"}'])
    report = run(captions=path, output=tmp_path / "b.json", ask=second,
                 model_name="stub", residual_threshold=0.0, max_rounds=5,
                 limit_groups=2, checkpoint=checkpoint)
    # Cell 1 came from the checkpoint, cell 2 was re-bought rather than trusted.
    assert len(second.prompts) >= 1
    assert report["groups"] == 2
