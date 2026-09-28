"""The account split's invariants, each exercised in the direction it can fail."""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.assign_account_disjoint_split import (  # noqa: E402
    SplitError,
    assign_accounts,
    build,
    account_name_prefixes,
    music_collisions,
)


def _corpus(tmp_path, accounts, *, frames=300, clips_per_upload=1):
    """A bundle whose account sizes are exactly what the caller asked for.

    ``accounts`` maps an account name to how many uploads it holds, so a test
    can build the shape it wants to break rather than the shape wild_v4 has.
    """
    bundle = tmp_path / "performance"
    (bundle / "sequences").mkdir(parents=True)
    sources, sequences, groups = [], [], {}
    upload_id = 7000000000000000000
    for account, uploads in accounts.items():
        for _ in range(uploads):
            upload_id += 1
            groups[str(upload_id)] = account
            for cut in range(clips_per_upload):
                recording = "wild_t:{}:clip{:03d}".format(upload_id, cut)
                row = {
                    "recording_id": recording,
                    "retrieval_group_id": "wild_t:{}".format(upload_id),
                    "duplicate_content_group_id": None,
                    "split": "train",
                    "music_sha256": "{:064x}".format(upload_id * 10 + cut),
                }
                sources.append(dict(row))
                sequences.append(dict(row, frame_count=frames))
    for name, rows in (("sources.jsonl", sources), ("sequences.jsonl", sequences)):
        (bundle / name).write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    keys = tmp_path / "group_keys.json"
    keys.write_text(json.dumps(groups), encoding="utf-8")
    return bundle, keys


def _split_of(bundle):
    return {json.loads(line)["recording_id"]: json.loads(line)["split"]
            for line in (bundle / "sources.jsonl").read_text(encoding="utf-8").splitlines()}


def test_an_account_never_lands_on_both_sides(tmp_path):
    bundle, keys = _corpus(tmp_path, {"a{}".format(i): 10 for i in range(12)},
                           clips_per_upload=3)
    report = build(bundle, keys, tmp_path / "out", test_share=0.1, val_share=0.1,
                   max_eval_account_fraction=0.15, min_eval_accounts=1,
                   share_tolerance=0.1)
    for pair, shared in report["summary"]["shared_accounts"].items():
        assert shared == [], pair
    # And the upload check is not implied by it: every cut of one upload moved
    # together, which is what stops a rehearsal of one take spanning the line.
    split_of = _split_of(tmp_path / "out")
    by_upload = {}
    for recording, split in split_of.items():
        by_upload.setdefault(recording.rsplit(":", 1)[0], set()).add(split)
    assert all(len(splits) == 1 for splits in by_upload.values())


def test_an_oversized_account_is_train_only_and_says_so(tmp_path):
    """The rule is a fraction, not a name: whichever account trips it is train."""
    bundle, keys = _corpus(tmp_path, dict({"whale": 60}, **{"a{}".format(i): 10
                                                            for i in range(12)}))
    report = build(bundle, keys, tmp_path / "out", test_share=0.1, val_share=0.1,
                   max_eval_account_fraction=0.15, min_eval_accounts=1,
                   share_tolerance=0.1)
    forced = report["policy"]["accounts_forced_to_train_as_oversized"]
    assert [entry["account"] for entry in forced] == ["whale"]
    assert report["summary"]["splits"]["train"]["accounts"].count("whale") == 1
    for split in ("val", "test"):
        assert "whale" not in report["summary"]["splits"][split]["accounts"]


def test_it_refuses_when_the_eligible_accounts_cannot_fill_eval(tmp_path):
    """One account holding almost everything cannot be split around."""
    mass = {"whale": 1_000_000, "small": 1_000}
    with pytest.raises(SplitError, match="eligible for eval"):
        assign_accounts(mass, test_share=0.1, val_share=0.1,
                        max_eval_account_fraction=0.15)


def test_an_unresolved_upload_is_an_error_not_a_bucket(tmp_path):
    bundle, keys = _corpus(tmp_path, {"a{}".format(i): 10 for i in range(12)})
    groups = json.loads(keys.read_text(encoding="utf-8"))
    dropped = sorted(groups)[0]
    del groups[dropped]
    keys.write_text(json.dumps(groups), encoding="utf-8")
    with pytest.raises(SplitError, match="no recorded account"):
        build(bundle, keys, tmp_path / "out", test_share=0.1, val_share=0.1,
              max_eval_account_fraction=0.15, min_eval_accounts=1, share_tolerance=0.1)


def test_a_single_choreographer_eval_split_is_refused(tmp_path):
    """Two accounts can fill 10% each; asking for three per side cannot be met,
    and the refusal is what keeps 'test' from naming one person's style."""
    bundle, keys = _corpus(tmp_path, {"big1": 40, "big2": 40, "rest": 120})
    with pytest.raises(SplitError, match="at least 3"):
        build(bundle, keys, tmp_path / "out", test_share=0.1, val_share=0.1,
              max_eval_account_fraction=0.5, min_eval_accounts=3, share_tolerance=0.2)


def test_a_missed_share_target_fails_rather_than_being_reported(tmp_path):
    bundle, keys = _corpus(tmp_path, {"a{}".format(i): 10 for i in range(5)})
    with pytest.raises(SplitError, match="outside"):
        build(bundle, keys, tmp_path / "out", test_share=0.02, val_share=0.02,
              max_eval_account_fraction=0.5, min_eval_accounts=1, share_tolerance=0.01)


def test_the_assignment_is_reproducible_from_the_names_alone(tmp_path):
    mass = {"a{}".format(index): 100 + index for index in range(20)}
    first, _ = assign_accounts(mass, test_share=0.1, val_share=0.1,
                              max_eval_account_fraction=0.15)
    second, _ = assign_accounts(dict(reversed(list(mass.items()))), test_share=0.1,
                                val_share=0.1, max_eval_account_fraction=0.15)
    assert first == second


def test_music_collisions_are_counted_not_assumed_absent():
    """The only cross-split audio identity this corpus can prove is reported as
    a number, including when it is zero, so a reader never has to guess whether
    it was checked."""
    sequences = [{"recording_id": "a", "music_sha256": "ff"},
                 {"recording_id": "b", "music_sha256": "ff"},
                 {"recording_id": "c", "music_sha256": "ee"}]
    report = music_collisions(sequences, {"a": "train", "b": "test", "c": "train"})
    assert report["hashes_spanning_splits"] == 1
    assert report["distinct_values"] == 2
    # Wording changed 2026-08-16 with its reason (CLAUDE.md 2).  It used to say
    # the corpus had "no track id and no fingerprint yet", which was already
    # false when it was read: fingerprint_wild_music.py had run and found 3,925
    # crossing pairs.  A report cannot assert the absence of a measurement it
    # does not take, so it now names the tool that takes it.
    assert "fingerprint_wild_music.py" in report["what_this_does_not_cover"]


def test_without_a_pair_list_the_fingerprint_row_says_unmeasured():
    """Absence of a number must not read as a zero.

    The hash row alone looks like a clean result -- one crossing group on
    wild_v4 -- while the fingerprint's answer on the same split was 3,925 pairs.
    So the section states outright that it was not asked.
    """
    sequences = [{"recording_id": "a", "music_sha256": "ff"},
                 {"recording_id": "b", "music_sha256": "ee"}]
    report = music_collisions(sequences, {"a": "train", "b": "test"})
    assert report["fingerprint"]["supplied"] is False
    assert "not a zero" in report["fingerprint"]["reading"]
    assert "pairs_crossing_a_split" not in report["fingerprint"]


def test_verified_pairs_are_counted_only_where_they_cross_a_split():
    """A pair inside one split is not a leak, and must not be counted as one."""
    sequences = [{"recording_id": name, "music_sha256": digest}
                 for name, digest in (("a", "1"), ("b", "2"), ("c", "3"), ("d", "4"))]
    split_of = {"a": "train", "b": "test", "c": "train", "d": "train"}
    report = music_collisions(
        sequences, split_of,
        [frozenset(("a", "b")),           # crosses train/test
         frozenset(("c", "d")),           # both in train -- not a leak
         frozenset(("a", "zzz"))],        # names a clip this corpus does not hold
        {"path": "somewhere.jsonl"})
    fingerprint = report["fingerprint"]
    assert fingerprint["supplied"] is True
    assert fingerprint["pairs_examined"] == 3
    assert fingerprint["pairs_crossing_a_split"] == 1
    assert fingerprint["clips_touched_by_a_crossing_pair"] == 2
    assert fingerprint["by_boundary"] == {"test_train": 1}
    assert fingerprint["clips_touched_by_split"] == {"test": 1, "train": 1}
    assert fingerprint["provenance"]["path"] == "somewhere.jsonl"


def test_a_name_prefix_spanning_every_split_is_named():
    """Account disjointness does not buy independence, and the names say so.

    Three ``O-DOG`` choreographers in three different splits are still three
    choreographers of one studio; the report has to surface that rather than let
    "account-disjoint" be read as "independent".  A prefix only one account
    carries is not reported, because a prefix of one is just a name.
    """
    account_split = {"O-DOG-A": "train", "O-DOG-B": "val", "O-DOG-C": "test",
                     "solo": "train"}
    mass = {"O-DOG-A": 400, "O-DOG-B": 100, "O-DOG-C": 100, "solo": 400}
    report = account_name_prefixes(account_split, mass)
    assert report["prefixes_spanning_all_three_splits"] == ["O-DOG"]
    span = report["spans"][0]
    assert span["prefix"] == "O-DOG" and span["accounts"] == 3
    assert span["by_split"]["train"]["frame_fraction_of_corpus"] == 0.4
    assert [row["prefix"] for row in report["spans"]] == ["O-DOG"]


def test_a_personal_name_containing_a_separator_still_joins_its_prefix():
    """The real corpus has ``O-DOG编舞师-QZIKA_琴子💙``.

    Cutting each name at its last separator put that account in a group of one
    and dropped its 4.6% of frames out of the span -- the report then said 18
    accounts where the corpus has 19.  Every boundary is enumerated instead, so
    a separator inside the personal part cannot hide the affiliation.
    """
    account_split = {"O-DOG-LEO": "train", "O-DOG-QZIKA_Q": "test", "solo": "val"}
    mass = {"O-DOG-LEO": 100, "O-DOG-QZIKA_Q": 100, "solo": 100}
    report = account_name_prefixes(account_split, mass)
    assert [row["prefix"] for row in report["spans"]] == ["O-DOG"]
    assert report["spans"][0]["accounts"] == 2


def test_nested_prefixes_over_one_account_set_are_reported_once():
    """``O`` and ``O-DOG`` cover the same accounts; the longer one is the finding."""
    account_split = {"O-DOG-A": "train", "O-DOG-B": "test"}
    report = account_name_prefixes(account_split, {"O-DOG-A": 1, "O-DOG-B": 1})
    assert [row["prefix"] for row in report["spans"]] == ["O-DOG"]


def test_a_split_that_leaves_nothing_to_train_on_is_refused(tmp_path):
    """None of the five older invariants looks at train.

    Measured on the wild corpus 2026-08-21: ``--test-share 0.44 --val-share 0.45
    --max-eval-account-fraction 0.27 --share-tolerance 0.11`` exited 0 and
    published train 0 / val 1,167 / test 936 -- stamped
    ``split_status: account_disjoint_source_safe``.  2,812 of the 127,952
    parameter combinations that "succeeded" had an empty train.  The shares
    bound eval; nothing bounded what was left.
    """
    # Two equal accounts and both shares at a half: the eval targets are met
    # exactly, every earlier invariant passes, and train is left with nothing.
    bundle, keys = _corpus(tmp_path, {"a": 3, "b": 3}, clips_per_upload=2)

    with pytest.raises(SplitError) as raised:
        build(bundle, keys, tmp_path / "out",
              test_share=0.5, val_share=0.5, max_eval_account_fraction=0.6,
              min_eval_accounts=1, share_tolerance=0.05, min_train_share=0.0)

    assert "no clip at all" in str(raised.value)


def test_a_train_share_under_the_floor_is_refused_and_can_be_lowered(tmp_path):
    """The floor is a criterion, not a constant: a deliberately small train
    passes by saying so, which is the difference between a decision and an
    accident."""
    bundle, keys = _corpus(tmp_path, {"a": 4, "b": 3, "c": 3}, clips_per_upload=2)

    with pytest.raises(SplitError) as raised:
        build(bundle, keys, tmp_path / "strict",
              test_share=0.3, val_share=0.3, max_eval_account_fraction=0.35,
              min_eval_accounts=1, share_tolerance=0.10, min_train_share=0.60)
    assert "under the" in str(raised.value)

    report = build(bundle, keys, tmp_path / "loose",
                   test_share=0.3, val_share=0.3, max_eval_account_fraction=0.35,
                   min_eval_accounts=1, share_tolerance=0.10, min_train_share=0.30)
    assert report["summary"]["splits"]["train"]["clips"] > 0
    assert report["policy"]["min_train_share"] == 0.30
