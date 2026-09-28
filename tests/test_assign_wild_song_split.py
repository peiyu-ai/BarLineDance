"""A split whose unit is the same-track component.

The leak this removes was measured, not feared: on wild_v4 the planner's labels
agreed with a *training* recording of the same song at 0.257 against 0.088 for
the clip's own labels, four clips at 0.96-1.00.  So the property under test is
narrow and absolute -- no component may end up on two sides -- plus the two
things the tool must keep saying it does *not* fix, because a split that reads
as "leak removed" is worse than one that reads as "song leak removed".
"""

import json
import pathlib
import sys
import tempfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools import assign_wild_song_split as splitter
from tools.assign_account_disjoint_split import SplitError


def _bundle(root, recordings, pairs):
    root = pathlib.Path(root)
    (root / "sequences").mkdir(parents=True)
    rows = [{"recording_id": name, "sequence_id": name, "split": "train",
             "retrieval_group_id": name.rsplit(":", 1)[0], "frame_count": 300}
            for name in recordings]
    for filename in ("sequences.jsonl", "sources.jsonl"):
        (root / filename).write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    pairs_path = root / "pairs.jsonl"
    pairs_path.write_text("".join(
        json.dumps({"left": a, "right": b, "lag_frames": 0}) + "\n"
        for a, b in pairs), encoding="utf-8")
    return root, pairs_path


def _names(count, prefix="wild_v4:u{}"):
    return ["wild_v4:u{}:clip000".format(i) for i in range(count)]


def test_no_component_is_split_and_that_is_checked_not_assumed():
    names = _names(60)
    # One large component, one pair, and 57 singletons.
    pairs = [(names[0], names[i]) for i in range(1, 12)] + [(names[20], names[21])]
    with tempfile.TemporaryDirectory() as raw:
        root, pairs_path = _bundle(pathlib.Path(raw) / "in", names, pairs)
        report = splitter.build(root, pairs_path, pathlib.Path(raw) / "out",
                                test_share=0.1, val_share=0.1)
        assert report["components_straddling"] == 0
        assert report["largest_component"] == 12
        rows = [json.loads(l) for l in
                (pathlib.Path(raw) / "out" / "sequences.jsonl").read_text().splitlines()]
        side = {r["recording_id"]: r["split"] for r in rows}
        assert len({side[n] for n in names[:12]}) == 1
        assert side[names[20]] == side[names[21]]
        assert sum(report["by_split"].values()) == 60


def test_the_largest_component_does_not_land_in_the_eval_splits():
    # Largest-first placement exists for this: the biggest real component holds
    # 16.9% of the corpus, which is larger than the whole test share.
    names = _names(100)
    pairs = [(names[0], names[i]) for i in range(1, 25)]
    with tempfile.TemporaryDirectory() as raw:
        root, pairs_path = _bundle(pathlib.Path(raw) / "in", names, pairs)
        splitter.build(root, pairs_path, pathlib.Path(raw) / "out",
                       test_share=0.1, val_share=0.1)
        rows = [json.loads(l) for l in
                (pathlib.Path(raw) / "out" / "sequences.jsonl").read_text().splitlines()]
        side = {r["recording_id"]: r["split"] for r in rows}
        assert side[names[0]] == "train"


def test_the_assignment_does_not_depend_on_input_line_order():
    names = _names(40)
    pairs = [(names[2], names[3]), (names[10], names[11])]
    sides = []
    for ordering in (names, list(reversed(names))):
        with tempfile.TemporaryDirectory() as raw:
            root, pairs_path = _bundle(pathlib.Path(raw) / "in", ordering, pairs)
            splitter.build(root, pairs_path, pathlib.Path(raw) / "out",
                           test_share=0.1, val_share=0.1)
            rows = [json.loads(l) for l in
                    (pathlib.Path(raw) / "out" / "sequences.jsonl").read_text().splitlines()]
            sides.append({r["recording_id"]: r["split"] for r in rows})
    assert sides[0] == sides[1]


def test_a_recording_the_bundle_names_and_the_pairs_do_not_is_still_assigned():
    # Singletons are components too.  Dropping them would silently shrink the
    # corpus and the shares would still read as though nothing were missing.
    names = _names(30)
    with tempfile.TemporaryDirectory() as raw:
        root, pairs_path = _bundle(pathlib.Path(raw) / "in", names, [])
        report = splitter.build(root, pairs_path, pathlib.Path(raw) / "out",
                                test_share=0.1, val_share=0.1)
        assert report["components"] == 30 and report["singletons"] == 30
        assert sum(report["by_split"].values()) == 30


def test_the_report_states_what_the_split_does_not_fix():
    names = _names(30)
    with tempfile.TemporaryDirectory() as raw:
        root, pairs_path = _bundle(pathlib.Path(raw) / "in", names, [])
        report = splitter.build(root, pairs_path, pathlib.Path(raw) / "out",
                                test_share=0.1, val_share=0.1)
    text = " ".join(report["does_not_fix"])
    assert "dancer" in text            # the account channel stays open
    # No positive control sits beside this fixture's pairs file, and the report
    # says so rather than quoting a recall measured on another corpus: the
    # figure moved from 32.5% (wild_v4) to 43.1% (wild_v5) the first time the
    # corpus was re-cut, so a hardcoded one would have been wrong by a third.
    assert "unmeasured here" in text
    assert "audit_split_leakage" in report["next_check"]


def test_the_report_quotes_the_recall_its_own_pairs_file_measured():
    names = _names(30)
    with tempfile.TemporaryDirectory() as raw:
        root, written = _bundle(pathlib.Path(raw) / "in", names, [])
        # The grouping run names its outputs <name>.json and <name>_pairs.jsonl,
        # which is the link this reads back; a renamed pairs file is reported as
        # unmeasured rather than filled in from somewhere else.
        pairs_path = written.with_name("wild_test_music_groups_pairs.jsonl")
        written.rename(pairs_path)
        pairs_path.with_name("wild_test_music_groups.json").write_text(
            json.dumps({"positive_control": {
                "same_upload_verified": 1167,
                "same_upload_candidate_pairs": 2708,
                "recall_on_same_upload_pairs": 0.4309}}), encoding="utf-8")
        report = splitter.build(root, pairs_path, pathlib.Path(raw) / "out",
                                test_share=0.1, val_share=0.1)
    text = " ".join(report["does_not_fix"])
    assert "1167 of 2708" in text
    assert "43.1%" in text


def test_an_account_channel_that_was_not_measured_is_not_reported_as_closed():
    names = _names(20)
    with tempfile.TemporaryDirectory() as raw:
        root, pairs_path = _bundle(pathlib.Path(raw) / "in", names, [])
        report = splitter.build(root, pairs_path, pathlib.Path(raw) / "out",
                                test_share=0.1, val_share=0.1, group_keys=None)
    assert report["accounts"]["available"] is False
    assert "not the same as closed" in report["accounts"]["why"]


def test_shares_that_leave_no_training_split_are_refused():
    names = _names(10)
    with tempfile.TemporaryDirectory() as raw:
        root, pairs_path = _bundle(pathlib.Path(raw) / "in", names, [])
        with pytest.raises(SplitError):
            splitter.build(root, pairs_path, pathlib.Path(raw) / "out",
                           test_share=0.6, val_share=0.5)


def _upload_names(uploads, clips_each):
    return ["wild_v5:u{}:clip{:03d}".format(u, c)
            for u in range(uploads) for c in range(clips_each)]


def test_clips_of_one_upload_stay_together_without_any_fingerprint_pair():
    """The fingerprint is not what holds an upload together, and cannot be.

    Same-upload pairs are its own positive control and it recovers 43.1% of the
    ones it considers (wild_v5, 2026-08-25).  On the first wild_v5 song split,
    built from fingerprint edges alone, 1,147 of 9,186 uploads ended up with
    clips in more than one split -- 2,604 clips, the same video in train and in
    test.  The upload id is in the recording id, so this edge is free and exact.
    """
    names = _upload_names(40, 3)
    with tempfile.TemporaryDirectory() as raw:
        root, pairs_path = _bundle(pathlib.Path(raw) / "in", names, [])
        report = splitter.build(root, pairs_path, pathlib.Path(raw) / "out",
                                test_share=0.1, val_share=0.1)
        out = pathlib.Path(raw) / "out"
        split_of = {json.loads(line)["recording_id"]: json.loads(line)["split"]
                    for line in (out / "sequences.jsonl").read_text().splitlines()}
    by_upload = {}
    for name, split in split_of.items():
        by_upload.setdefault(name.rsplit(":", 1)[0], set()).add(split)
    assert all(len(sides) == 1 for sides in by_upload.values())
    # 40 uploads of 3 clips, no fingerprint pair at all -> 40 components, not 120.
    assert report["components"] == 40
    assert report["components_from_fingerprint_edges_alone"] == 120
    assert report["uploads_straddling"] == 0 and report["uploads"] == 40
    assert report["edges"]["fingerprint_pairs"] == 0
    assert report["edges"]["same_upload_recordings"] == 40 * 2


def test_the_upload_check_can_actually_fail():
    # A gate that cannot fire reads like a check and is worse than none, so the
    # edge set is removed and the refusal is observed rather than trusted.
    names = _upload_names(40, 3)
    with tempfile.TemporaryDirectory() as raw:
        root, pairs_path = _bundle(pathlib.Path(raw) / "in", names, [])
        real = splitter.upload_edges
        splitter.upload_edges = lambda sequences: {}
        try:
            with pytest.raises(SplitError) as error:
                splitter.build(root, pairs_path, pathlib.Path(raw) / "out",
                               test_share=0.1, val_share=0.1)
        finally:
            splitter.upload_edges = real
    assert "more than one split" in str(error.value)


def test_a_fingerprint_pair_across_uploads_still_merges_them():
    # The two edge kinds have to union, not replace: holding uploads together
    # must not stop the same track linking two different uploads.
    names = _upload_names(30, 2)
    pairs = [(names[0], names[2])]          # u0:clip000 -- u1:clip000
    with tempfile.TemporaryDirectory() as raw:
        root, pairs_path = _bundle(pathlib.Path(raw) / "in", names, pairs)
        report = splitter.build(root, pairs_path, pathlib.Path(raw) / "out",
                                test_share=0.1, val_share=0.1)
        out = pathlib.Path(raw) / "out"
        split_of = {json.loads(line)["recording_id"]: json.loads(line)["split"]
                    for line in (out / "sequences.jsonl").read_text().splitlines()}
    assert len({split_of[n] for n in names[:4]}) == 1     # both uploads, one side
    assert report["components"] == 29                     # 30 uploads, two merged
    assert report["largest_component"] == 4


# --------------------------------------------------------------------------- #
# Several pair lists.  Two fingerprint runs over two cuts of the same videos do
# not miss the same pairs: v4's list held 4,352 edges v5's did not, and 281 of
# them crossed the split built from v5's list alone (2026-08-26).
# --------------------------------------------------------------------------- #


def _extra_pairs(root, name, pairs):
    path = pathlib.Path(root) / name
    path.write_text("".join(
        json.dumps({"left": a, "right": b, "lag_frames": 0}) + "\n"
        for a, b in pairs), encoding="utf-8")
    return path


def test_an_edge_only_the_older_list_measured_still_holds_the_split():
    """The defect this reproduces: split on the new list, check with the old."""
    names = _names(60)
    new_list = [(names[0], names[1])]
    # Edges only the older run measured, spread across the corpus so the
    # assertion does not depend on which singleton the greedy pass happens to
    # place where.
    extra = [(names[i], names[i + 30]) for i in range(2, 20)]
    old_list = new_list + extra
    with tempfile.TemporaryDirectory() as raw:
        root, new_path = _bundle(pathlib.Path(raw) / "in", names, new_list)
        old_path = _extra_pairs(pathlib.Path(raw) / "in", "old.jsonl", old_list)

        only_new = splitter.build(root, [new_path], pathlib.Path(raw) / "out-new",
                                  test_share=0.1, val_share=0.1)
        side = {r["recording_id"]: r["split"] for r in
                (json.loads(l) for l in (pathlib.Path(raw) / "out-new" /
                                         "sequences.jsonl").read_text().splitlines())}
        assert only_new["components_straddling"] == 0     # clean against its own list
        crossing = sum(1 for a, b in extra if side[a] != side[b])
        assert crossing > 0                               # and not against the other

        both = splitter.build(root, [new_path, old_path], pathlib.Path(raw) / "out-both",
                              test_share=0.1, val_share=0.1)
        side = {r["recording_id"]: r["split"] for r in
                (json.loads(l) for l in (pathlib.Path(raw) / "out-both" /
                                         "sequences.jsonl").read_text().splitlines())}
        assert all(side[a] == side[b] for a, b in extra)
        assert [e["edges_crossing_the_split"] for e in both["pairs_read"]] == [0, 0]


def test_each_list_is_reported_on_its_own_line_so_a_no_op_list_is_visible():
    names = _names(40)
    shared = [(names[0], names[1])]
    with tempfile.TemporaryDirectory() as raw:
        root, first = _bundle(pathlib.Path(raw) / "in", names, shared)
        second = _extra_pairs(pathlib.Path(raw) / "in", "same.jsonl", shared)
        report = splitter.build(root, [first, second], pathlib.Path(raw) / "out",
                                test_share=0.1, val_share=0.1)
        rows = report["pairs_read"]
        assert [r["edges_on_this_bundle"] for r in rows] == [1, 1]
        # The second list repeats the first: it lands, and it adds nothing.
        assert [r["edges_this_file_added"] for r in rows] == [1, 0]
        assert report["pair_rows"] == 2


def test_a_list_from_another_generation_is_matched_with_its_prefix_stripped():
    names = _names(40)
    carried = [(n.replace("wild_v4:", "wild_v5:"), m.replace("wild_v4:", "wild_v5:"))
               for n, m in [(names[10], names[11])]]
    with tempfile.TemporaryDirectory() as raw:
        root, empty = _bundle(pathlib.Path(raw) / "in", names, [])
        other = _extra_pairs(pathlib.Path(raw) / "in", "v5.jsonl", carried)
        report = splitter.build(root, [empty, other], pathlib.Path(raw) / "out",
                                test_share=0.1, val_share=0.1)
        side = {r["recording_id"]: r["split"] for r in
                (json.loads(l) for l in (pathlib.Path(raw) / "out" /
                                         "sequences.jsonl").read_text().splitlines())}
        assert side[names[10]] == side[names[11]]
        assert report["pairs_read"][1]["edges_on_this_bundle"] == 1


def test_a_list_whose_rows_land_on_nothing_is_refused_not_counted_as_zero():
    """A wrong path and a list with nothing to add read identically otherwise."""
    names = _names(40)
    with tempfile.TemporaryDirectory() as raw:
        root, empty = _bundle(pathlib.Path(raw) / "in", names, [])
        stray = _extra_pairs(pathlib.Path(raw) / "in", "stray.jsonl",
                             [("other_corpus:9:clip000", "other_corpus:8:clip000")])
        with pytest.raises(SplitError) as error:
            splitter.build(root, [empty, stray], pathlib.Path(raw) / "out",
                           test_share=0.1, val_share=0.1)
        assert "names no pair of this bundle" in str(error.value)


def test_a_list_with_no_rows_at_all_is_accepted():
    """No fingerprint evidence is a real state; the same-upload edges still hold."""
    names = _names(40)
    with tempfile.TemporaryDirectory() as raw:
        root, empty = _bundle(pathlib.Path(raw) / "in", names, [])
        report = splitter.build(root, [empty], pathlib.Path(raw) / "out",
                                test_share=0.1, val_share=0.1)
        assert report["pairs_read"][0]["edges_on_this_bundle"] == 0
        assert report["components_straddling"] == 0


# --- pinned mode -------------------------------------------------------------
# Appending recordings to a published split must not move any of them: the eval
# list is the ruler and every checkpoint was trained on the old train split.
# Unpinned re-assignment moved 45-82 old T-line clips in simulation (2026-09-22).

def _pin_file(root, sides):
    path = pathlib.Path(root) / "pins.jsonl"
    path.write_text("".join(json.dumps({"recording_id": n, "split": s}) + "\n"
                            for n, s in sides.items()), encoding="utf-8")
    return path


def _sides(out):
    rows = [json.loads(l) for l in (out / "sequences.jsonl").read_text().splitlines()]
    return {r["recording_id"]: r["split"] for r in rows}


def test_pins_hold_and_new_clips_follow_their_song():
    old = ["wild_v4:o{}:clip000".format(i) for i in range(6)]
    new = ["wild_v4:n{}:clip000".format(i) for i in range(3)]
    pins = dict(zip(old, ["test", "val", "train", "train", "test", "val"]))
    # n0 shares a song with test o0, n1 with val o1, n2 with nothing.
    pairs = [(new[0], old[0]), (new[1], old[1])]
    with tempfile.TemporaryDirectory() as raw:
        root, pairs_path = _bundle(pathlib.Path(raw) / "in", old + new, pairs)
        out = pathlib.Path(raw) / "out"
        report = splitter.build(root, pairs_path, out, test_share=0.5, val_share=0.4,
                                pin=_pin_file(raw, pins), unpinned_to="train")
        side = _sides(out)
        assert all(side[n] == s for n, s in pins.items())
        assert (side[new[0]], side[new[1]], side[new[2]]) == ("test", "val", "train")
        p = report["pinning"]
        assert p["new_following_a_pinned_component"] == {"test": 1, "val": 1}
        assert p["new_in_all_new_components"] == {"train": 1}


def test_a_new_clip_joining_two_pinned_splits_is_refused_or_dropped():
    old = ["wild_v4:o0:clip000", "wild_v4:o1:clip000"]
    bridge = "wild_v4:n0:clip000"
    pins = {old[0]: "test", old[1]: "train"}
    pairs = [(bridge, old[0]), (bridge, old[1])]
    with tempfile.TemporaryDirectory() as raw:
        root, pairs_path = _bundle(pathlib.Path(raw) / "in", old + [bridge], pairs)
        pin_path = _pin_file(raw, pins)
        with pytest.raises(SplitError, match="more than one pinned split"):
            splitter.build(root, pairs_path, pathlib.Path(raw) / "a", test_share=0.1,
                           val_share=0.1, pin=pin_path, unpinned_to="train")
        # With the bridge gone its two ends are no longer linked; the pairs file
        # still names the crossing edge, which lands on a dropped recording only.
        report = splitter.build(root, pairs_path, pathlib.Path(raw) / "b", test_share=0.1,
                                val_share=0.1, pin=pin_path, unpinned_to="train",
                                drop_bridging=True)
        side = _sides(pathlib.Path(raw) / "b")
        assert bridge not in side and side == pins
        assert [d["recording_id"] for d in report["pinning"]["dropped_bridging"]] == [bridge]


def test_a_straddle_the_pinned_split_already_had_is_kept_and_reported():
    old = ["wild_v4:o0:clip000", "wild_v4:o1:clip000"]
    pins = {old[0]: "val", old[1]: "train"}
    with tempfile.TemporaryDirectory() as raw:
        root, pairs_path = _bundle(pathlib.Path(raw) / "in", old, [(old[0], old[1])])
        report = splitter.build(root, pairs_path, pathlib.Path(raw) / "out", test_share=0.1,
                                val_share=0.1, pin=_pin_file(raw, pins), unpinned_to="train")
        assert _sides(pathlib.Path(raw) / "out") == pins
        assert report["components_straddling_preexisting_between_pins"] == 1
        assert report["pairs_read"][0]["edges_crossing_between_pinned_recordings"] == 1
        assert report["pairs_read"][0]["edges_crossing_the_split"] == 0


def test_a_pinned_clip_missing_from_the_bundle_is_refused():
    names = ["wild_v4:o0:clip000"]
    pins = {names[0]: "train", "wild_v4:gone:clip000": "test"}
    with tempfile.TemporaryDirectory() as raw:
        root, pairs_path = _bundle(pathlib.Path(raw) / "in", names, [])
        pin_path = _pin_file(raw, pins)
        with pytest.raises(SplitError, match="1 of them test"):
            splitter.build(root, pairs_path, pathlib.Path(raw) / "a", test_share=0.1,
                           val_share=0.1, pin=pin_path, unpinned_to="train")
        report = splitter.build(root, pairs_path, pathlib.Path(raw) / "b", test_share=0.1,
                                val_share=0.1, pin=pin_path, unpinned_to="train",
                                allow_missing_pins=True)
        assert report["pinning"]["pins_absent_from_bundle"] == ["wild_v4:gone:clip000"]


def test_pin_requires_an_explicit_home_for_all_new_songs():
    names = ["wild_v4:o0:clip000", "wild_v4:n0:clip000"]
    with tempfile.TemporaryDirectory() as raw:
        root, pairs_path = _bundle(pathlib.Path(raw) / "in", names, [])
        with pytest.raises(SplitError, match="no default"):
            splitter.build(root, pairs_path, pathlib.Path(raw) / "a", test_share=0.1,
                           val_share=0.1, pin=_pin_file(raw, {names[0]: "train"}))
