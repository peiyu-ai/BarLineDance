"""The song-disjoint split's four refusals, each shown refusing and permitting.

The split exists to make one claim true -- that a test song was never heard in
train -- so every guard around it has to be able to fail on a corpus built to
break it.  The one that matters most is the *reported* one: the choreography
overlap a song-disjoint split necessarily creates on AIST++.  If that number
could not move, the caveat in the report would be decoration.
"""

import json
import pathlib

import pytest

from tools.assign_song_disjoint_split import (
    SPLIT_STATUS,
    SplitError,
    assign_songs,
    build,
    choreography_overlap,
    parse_name,
    summarise,
)


def _name(genre="BR", situation="BM", camera="All", dancer="04", song="mBR0", choreography="01"):
    return "g{}_s{}_c{}_d{}_{}_ch{}".format(genre, situation, camera, dancer, song, choreography)


def _corpus(names, tmp_path, *, groups=None, duplicate_groups=None):
    """A minimal two-manifest bundle: one source and one sequence per name."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    sources, sequences = [], []
    for name in names:
        group = (groups or {}).get(name, "aistpp/" + name.rsplit("_ch", 1)[0])
        row = {
            "legacy_source_name": name,
            "recording_id": "aistpp/" + name,
            "retrieval_group_id": group,
            "duplicate_content_group_id": (duplicate_groups or {}).get(name),
            "split": "train",
            "split_note": "before",
            "split_status": "frozen_performance_group_source_safe",
        }
        sources.append(row)
        sequences.append(dict(row, sequence_id="aistpp/{}/sequence0".format(name)))
    for filename, rows in (("sources.jsonl", sources), ("sequences.jsonl", sequences)):
        (bundle / filename).write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
        )
    (bundle / "sequences").mkdir()
    return bundle


def _full_corpus(genres=("BR", "HO"), songs_per_genre=3):
    """Every genre danced to several songs, one basic routine each."""
    names = []
    for genre in genres:
        for index in range(songs_per_genre):
            names.append(_name(genre=genre, song="m{}{}".format(genre, index)))
    return names


class TestSongAssignment:
    def test_every_genre_fills_test_and_val_and_keeps_the_rest(self):
        songs = {"BR": ["mBR0", "mBR1", "mBR2"], "HO": ["mHO0", "mHO1", "mHO2"]}
        assignment = assign_songs(songs, test_per_genre=1, val_per_genre=1)
        for genre, members in songs.items():
            drawn = [assignment[song] for song in members]
            assert sorted(drawn) == ["test", "train", "val"], genre

    def test_the_order_is_the_salted_hash_and_not_the_input_order(self):
        songs = {"BR": ["mBR0", "mBR1", "mBR2"]}
        forward = assign_songs(songs, test_per_genre=1, val_per_genre=1)
        backward = assign_songs(
            {"BR": list(reversed(songs["BR"]))}, test_per_genre=1, val_per_genre=1
        )
        assert forward == backward

    def test_a_genre_too_small_to_fill_three_splits_is_refused(self):
        # Two songs cannot fill test, val and train, and quietly emptying train
        # for that genre is exactly the silent hole this raises to prevent.
        with pytest.raises(SplitError, match="song"):
            assign_songs({"BR": ["mBR0", "mBR1"]}, test_per_genre=1, val_per_genre=1)


class TestNamesAreParsedOrRefused:
    def test_the_six_aist_fields_are_recovered(self):
        fields = parse_name(_name(genre="LO", situation="FM", dancer="17", song="mLO4",
                                  choreography="12"))
        assert fields["genre"] == "LO"
        assert fields["situation"] == "FM"
        assert fields["song"] == "mLO4"
        assert fields["choreography"] == "12"

    def test_an_unrecognised_name_raises_instead_of_becoming_a_bucket(self):
        with pytest.raises(SplitError, match="cannot parse"):
            parse_name("some_wild_clip_000")


class TestChoreographyOverlapCanMoveInBothDirections:
    def test_a_routine_danced_to_every_song_is_reported_as_fully_overlapping(self):
        # AIST's basic half: one routine, three songs.  Whichever song is held
        # out, its routine is still in train under another song.
        names = [_name(song="mBR{}".format(index)) for index in range(3)]
        fields = {name: parse_name(name) for name in names}
        split_of = {names[0]: "test", names[1]: "val", names[2]: "train"}
        overlap = choreography_overlap(fields, split_of)
        assert overlap["test"]["fraction_also_in_train"] == 1.0
        assert overlap["test"]["by_situation"]["BM"]["also_in_train"] == 1

    def test_song_specific_routines_leave_nothing_in_train(self):
        # AIST's advanced half: each routine belongs to one song, so holding the
        # song out holds the routine out too.
        names = [
            _name(situation="FM", song="mBR{}".format(index), choreography="0{}".format(index))
            for index in range(3)
        ]
        fields = {name: parse_name(name) for name in names}
        split_of = {names[0]: "test", names[1]: "val", names[2]: "train"}
        overlap = choreography_overlap(fields, split_of)
        assert overlap["test"]["fraction_also_in_train"] == 0.0
        assert overlap["test"]["by_situation"]["FM"]["also_in_train"] == 0


class TestBuildRefusesWhatItCannotPublish:
    def test_a_published_split_has_disjoint_songs_and_every_genre_everywhere(self, tmp_path):
        bundle = _corpus(_full_corpus(), tmp_path)
        report = build(bundle, tmp_path / "out", test_per_genre=1, val_per_genre=1)
        assert all(not shared for shared in report["summary"]["shared_songs"].values())
        for value in report["summary"]["splits"].values():
            assert sorted(value["genres"]) == ["BR", "HO"]
        rows = [
            json.loads(line)
            for line in (tmp_path / "out" / "sources.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert {row["split_status"] for row in rows} == {SPLIT_STATUS}
        assert {row["split"] for row in rows} == {"train", "val", "test"}

    def test_a_retrieval_group_spanning_two_songs_is_refused(self, tmp_path):
        # A performance group that holds two songs cannot land on one side of a
        # song boundary; publishing it would certify a split that leaks.
        names = _full_corpus()
        groups = {name: "aistpp/shared-group" for name in names[:2]}
        bundle = _corpus(names, tmp_path, groups=groups)
        with pytest.raises(SplitError, match="retrieval_group_id spans splits"):
            build(bundle, tmp_path / "out", test_per_genre=1, val_per_genre=1)

    def test_a_duplicate_content_group_spanning_two_songs_is_refused(self, tmp_path):
        names = _full_corpus()
        duplicates = {name: "dup-0" for name in names[:2]}
        bundle = _corpus(names, tmp_path, duplicate_groups=duplicates)
        with pytest.raises(SplitError, match="duplicate_content_group_id spans splits"):
            build(bundle, tmp_path / "out", test_per_genre=1, val_per_genre=1)

    def test_an_existing_output_bundle_is_never_overwritten(self, tmp_path):
        bundle = _corpus(_full_corpus(), tmp_path)
        (tmp_path / "out").mkdir()
        with pytest.raises(FileExistsError):
            build(bundle, tmp_path / "out", test_per_genre=1, val_per_genre=1)

    def test_a_sequence_without_a_source_row_is_refused(self, tmp_path):
        bundle = _corpus(_full_corpus(), tmp_path)
        orphan = {
            "legacy_source_name": _name(genre="JB", song="mJB0"),
            "recording_id": "aistpp/" + _name(genre="JB", song="mJB0"),
            "retrieval_group_id": "aistpp/orphan",
            "duplicate_content_group_id": None,
            "split": "train",
        }
        with (bundle / "sequences.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(orphan, sort_keys=True) + "\n")
        with pytest.raises(SplitError, match="absent from sources.jsonl"):
            build(bundle, tmp_path / "out", test_per_genre=1, val_per_genre=1)


class TestSummaryCounts:
    def test_recordings_are_counted_per_split(self, tmp_path):
        names = _full_corpus(genres=("BR",), songs_per_genre=4)
        fields = {name: parse_name(name) for name in names}
        assignment = assign_songs({"BR": [f["song"] for f in fields.values()]},
                                  test_per_genre=1, val_per_genre=1)
        split_of = {name: assignment[f["song"]] for name, f in fields.items()}
        summary = summarise(fields, split_of)
        assert summary["splits"]["train"]["recordings"] == 2
        assert summary["splits"]["val"]["recordings"] == 1
        assert summary["splits"]["test"]["recordings"] == 1
