"""Tests for the M6 clip selection and its leak flags.

The selection has two jobs and both are checkable from names alone: draw a
reproducible sample that *extends* rather than reshuffles when the count grows,
and mark every clip that provably shares a backing track across the split.  The
second is the one with a failure mode that looks like success -- a pair list
that flags nothing produces a "leak-free subset" identical to the whole set --
so that case is asserted to raise.

Two different things make a list flag nothing and only one is a defect: a list
that does not *resolve* against the bundle describes nothing and still raises,
while a song-disjoint split has every pair resolving and none crossing, which
is the state ``assign_wild_song_split`` exists to produce.  The second is
allowed through and said out loud in the report, because "leak-free" figures
that equal the unfiltered ones are a comparison a reader would otherwise assume
had happened.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.select_wild_eval_clips import (  # noqa: E402
    SelectionError,
    build_parser,
    main,
    select,
)


def _bundle(tmp_path, rows):
    bundle = tmp_path / "bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "sequences.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return bundle


def _corpus(tmp_path, test_clips=12, train_clips=8):
    rows = [{"recording_id": "wild_v4:{}:clip000".format(700 + index), "split": "test"}
            for index in range(test_clips)]
    rows += [{"recording_id": "wild_v4:{}:clip000".format(900 + index), "split": "train"}
             for index in range(train_clips)]
    return _bundle(tmp_path, rows)


def _pairs(tmp_path, pairs, name="pairs.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(
        json.dumps({"left": left, "right": right, "score": 0.9}) for left, right in pairs
    ) + "\n", encoding="utf-8")
    return path


def test_a_larger_count_extends_the_sample_rather_than_reshuffling_it(tmp_path):
    """Two checkpoints scored on different clip sets are not compared at all.

    Growing the sample has to keep every clip the smaller one had, or the two
    FIDs differ partly because the corpora under them differ.
    """
    bundle = _corpus(tmp_path)
    small = select(bundle, "test", 4, None)["clips"]
    large = select(bundle, "test", 9, None)["clips"]
    assert large[:4] == small
    # And it is a function of the names, not of the file order.
    again = select(_bundle(tmp_path / "b", [
        {"recording_id": "wild_v4:{}:clip000".format(700 + index), "split": "test"}
        for index in reversed(range(12))]), "test", 4, None)["clips"]
    assert again == small


def test_asking_for_more_clips_than_the_split_holds_is_refused(tmp_path):
    with pytest.raises(SelectionError, match="split test holds"):
        select(_corpus(tmp_path, test_clips=5), "test", 9, None)


def test_an_empty_split_is_refused(tmp_path):
    with pytest.raises(SelectionError, match="names no clip"):
        select(_corpus(tmp_path), "val", 2, None)


def test_a_pair_list_that_does_not_resolve_here_is_refused(tmp_path):
    """CLAUDE.md 2.  A leak-free subset that equals the whole set is not one.

    The failure is silent by nature: every downstream number comes out, and the
    only symptom is that the "clean" figure equals the contaminated one.  These
    ids belong to no recording in the bundle, so the list is not describing this
    corpus at all.
    """
    bundle = _corpus(tmp_path)
    pairs = _pairs(tmp_path, [("wild_v4:5555:clip000", "wild_v4:6666:clip000")])
    with pytest.raises(SelectionError, match="resolves against no recording"):
        select(bundle, "test", 4, pairs)


def test_a_split_no_pair_crosses_is_reported_not_refused(tmp_path):
    """The song-disjoint split's own goal state.

    Every pair resolves against the bundle and none crosses the line, which is
    what ``assign_wild_song_split`` refuses to publish without.  Refusing here
    would make the tool unusable exactly when the corpus is correct.  Measured
    on wild_v5_song, 2026-08-25: 10,392 verified pairs, all resolving, none
    crossing.
    """
    bundle = _corpus(tmp_path)
    pairs = _pairs(tmp_path, [
        ("wild_v4:700:clip000", "wild_v4:701:clip000"),   # test <-> test
        ("wild_v4:900:clip000", "wild_v4:901:clip000"),   # train <-> train
    ])
    report = select(bundle, "test", 4, pairs)
    assert report["leak"]["flagged_in_split"] == 0
    assert report["leak"]["leak_free_subset_equals_the_split"] is True
    assert report["leak"]["pair_list"]["pairs_resolved_against_bundle"] == 2
    assert report["leak"]["pair_list"]["pairs_crossing_the_split"] == 0


def test_a_split_with_a_real_leak_does_not_claim_to_be_clean(tmp_path):
    # The same field on the other side, so the flag is not just always true.
    bundle = _corpus(tmp_path)
    pairs = _pairs(tmp_path, [("wild_v4:700:clip000", "wild_v4:900:clip000")])
    report = select(bundle, "test", 4, pairs)
    assert report["leak"]["leak_free_subset_equals_the_split"] is False
    assert report["leak"]["pair_list"]["pairs_crossing_the_split"] == 1


def test_only_pairs_that_cross_the_split_are_flagged(tmp_path):
    bundle = _corpus(tmp_path)
    pairs = _pairs(tmp_path, [
        ("wild_v4:700:clip000", "wild_v4:900:clip000"),   # test <-> train: a leak
        ("wild_v4:701:clip000", "wild_v4:702:clip000"),   # test <-> test: not a leak
    ])
    report = select(bundle, "test", 12, pairs)
    assert report["leak"]["flagged_in_split"] == 1
    assert report["leak"]["flagged_clips_in_split"] == ["wild_v4:700:clip000"]
    assert report["leak"]["clips"]["wild_v4:700:clip000"] == ["train"]


def test_the_flagged_list_covers_the_split_not_only_the_selection(tmp_path):
    """The ground-truth side of a leak-free FID is the whole split.

    Filtering it by the flags of the *selected* clips leaves every other flagged
    clip in the "clean" reference -- on wild_v4 that was 618 of 825.  So the
    report has to carry the split's flagged set, not the selection's.
    """
    bundle = _corpus(tmp_path)
    pairs = _pairs(tmp_path, [
        ("wild_v4:{}:clip000".format(700 + index), "wild_v4:900:clip000")
        for index in range(12)])
    report = select(bundle, "test", 3, pairs)
    assert report["selected"] == 3
    assert len(report["leak"]["clips"]) == 3               # flags within the sample
    assert len(report["leak"]["flagged_clips_in_split"]) == 12   # flags across the split


def test_without_a_pair_list_the_report_says_so_rather_than_reporting_zero(tmp_path):
    report = select(_corpus(tmp_path), "test", 4, None)
    assert report["leak"]["pair_list"]["supplied"] is False
    assert report["leak"]["flagged_in_selection"] == 0
    assert report["leak"]["flagged_clips_in_split"] == []


def test_the_cli_writes_both_the_list_and_the_report(tmp_path):
    bundle = _corpus(tmp_path)
    pairs = _pairs(tmp_path, [("wild_v4:700:clip000", "wild_v4:900:clip000")])
    output = tmp_path / "sel"
    assert main(["--bundle", str(bundle), "--split", "test", "--count", "5",
                 "--music-pairs", str(pairs), "--output", str(output)]) == 0
    names = output.with_suffix(".txt").read_text(encoding="utf-8").split()
    report = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert names == report["clips"]
    assert len(names) == 5


def test_the_selection_salt_is_not_the_split_salt():
    """Reusing the split's salt would make the sample a systematic slice of it."""
    from tools.assign_account_disjoint_split import SALT as SPLIT_SALT
    from tools.select_wild_eval_clips import SALT as SELECT_SALT

    assert SELECT_SALT != SPLIT_SALT


def test_the_pair_flag_is_optional_on_the_command_line():
    assert build_parser().parse_args(
        ["--bundle", "b", "--output", "o"]).music_pairs is None
