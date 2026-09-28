#!/usr/bin/env python3
"""Re-split the wild corpus by uploader account, and publish what that cannot fix.

The wild split shipped by ``preprocess_wild_3d`` is **upload-level**: whole
TikTok uploads are ranked by ``sha256(seed:recording_id)`` and cut by count, so
two clips of one upload never straddle a boundary.  That stops the crudest leak
(a rehearsal of the same take on both sides) and nothing beyond it.

It does not stop the one that matters here.  One account is one choreographer,
and a choreographer posts the *same routine* across many uploads -- re-shot,
re-cut, re-uploaded, sometimes mirrored.  Measured on ``wild_v4``: 13,783 clips
come from 9,180 uploads but only **25 accounts**, and the largest holds 26% of
the corpus.  An upload-level split therefore puts near-duplicates of held-out
motion into train by construction, and every downstream metric reads better for
it with nothing reporting why.

So the unit here is the account.  What the account is **not** is a label: it
carries no movement information the way AIST's genre field does, and this tool
never treats it as one.  It is a *source* identity, which is exactly what a
train/eval boundary needs.

What this tool deliberately does not claim
------------------------------------------
An account-disjoint split of this corpus **cannot** be music-disjoint, and
saying so is the point of this docstring.  TikTok clips carry no track id: the
only audio identity in the bundle is ``music_sha256``, the content hash of the
clip's own 35-D feature array, and that is unique per clip by construction
(13,781 distinct values over 13,783 clips) because two clips of one song at
different offsets hash differently.  A shared backing track across two accounts
is therefore invisible to us today, not absent.  This tool measures what *is*
measurable -- exact music-feature collisions across the boundary -- and records
the rest as an open gap rather than a satisfied check.  Deriving a real track
identity (audio fingerprint) is the separate piece of work that would close it.

The oversized account, and why it is train
------------------------------------------
``--max-eval-account-fraction`` refuses to put an account larger than that
fraction of the corpus into val or test.  On ``wild_v4`` exactly one account
trips it (26%, next largest 5.6%), so the rule is not a name in a list: an
account that big *is* the eval split if it lands there, and "held-out" would
then mean "one choreographer".  The cost is real and is written into the report
rather than left to be discovered: the corpus's single largest style is only
ever trained on, so no eval number here says anything about generalising to it,
and train is roughly one-third that one account.

Assignment is deterministic and not tuned: accounts are ordered by
``sha256(SALT || account)``, then greedily filled into test until it reaches
its target share of *frames*, then val, and the rest is train.  ``SALT`` is a
frozen constant, so the split is reproducible from the account names alone and
no choice in it was made after looking at a score.  Greedy-on-a-hash-order
cannot hit a target share exactly with 25 unequal accounts, so the achieved
share is reported and a tolerance is enforced rather than assumed.

Five invariants are checked before anything is written, and each one can fail:

* every clip's upload resolves to a recorded account -- an unresolved upload is
  an error, not a ``"?"`` bucket, because a silent catch-all is how the genre
  pre-split quietly disappeared once already (see ``recluster_atomics_ingroup``
  and ``run_wild_stage_g_m3b_oss``);
* the account sets are pairwise disjoint;
* no ``retrieval_group_id`` (the upload) and no ``duplicate_content_group_id``
  spans splits;
* each eval split holds at least ``--min-eval-accounts`` accounts, so neither
  is one choreographer wearing the word "test";
* each eval split's frame share lands inside ``--share-tolerance`` of target.

Usage::

    python3 tools/assign_account_disjoint_split.py \\
        --bundle data/wild3d/wild_v4_performance \\
        --group-keys runs/wild_v4_group_keys.json \\
        --output-bundle data/wild3d/wild_v4_acct_performance \\
        --report runs/wild_v4_acct_assignment.json
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import pathlib
import re
import shutil
import sys
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# One reader for the fingerprint's pair list, imported rather than re-written.
# Two tools each parsing the same artifact is how this repo ended up with two
# music-id parsers that disagreed on which dances share a song (2026-08-15).
from tools.eval_r_precision import load_verified_music_pairs  # noqa: E402

SPLITS: Tuple[str, ...] = ("train", "val", "test")

# Frozen once, on 2026-08-16, so the ordering is reproducible from account names
# alone.  Changing it re-draws the benchmark; that is a new split, not a re-run.
SALT = "atomicdance-account-disjoint-v1"

SPLIT_STATUS = "account_disjoint_source_safe"


class SplitError(RuntimeError):
    """A split that would be wrong is refused rather than published."""


def upload_key(row: Mapping[str, Any]) -> str:
    """The upload id a manifest row belongs to, as the group-key file names it.

    ``retrieval_group_id`` is the upload (``wild_v4:7316544444038237474``) and
    ``recording_id`` is one clip of it (``...:clip003``); the group-key file is
    keyed by the bare upload id.  Reading the group from ``recording_id`` would
    work today and break the moment a corpus names clips differently, so the
    exclusion unit the bundle already declares is what is used.
    """
    group = row.get("retrieval_group_id")
    if not isinstance(group, str) or not group:
        raise SplitError("row lacks retrieval_group_id: {!r}".format(row.get("recording_id")))
    return group.rsplit(":", 1)[-1]


def recording_id(row: Mapping[str, Any]) -> str:
    value = row.get("recording_id")
    if not isinstance(value, str) or not value:
        raise SplitError("row lacks recording_id: {!r}".format(row))
    return value


def account_order_key(account: str) -> str:
    return hashlib.sha256((SALT + "\x00" + account).encode("utf-8")).hexdigest()


def assign_accounts(
    mass: Mapping[str, int],
    *,
    test_share: float,
    val_share: float,
    max_eval_account_fraction: float,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Whole accounts to splits, test filled first so it is never starved.

    ``mass`` is frames per account, because a window budget is frames and clip
    counts are only a proxy for it.  The two agree here to within 0.1 pp, which
    the report states so that the choice is visible rather than load-bearing.
    """
    total = sum(mass.values())
    if total <= 0:
        raise SplitError("corpus has no frames")

    oversized = sorted(
        account for account, frames in mass.items()
        if frames / total > max_eval_account_fraction
    )
    eligible = sorted(
        (account for account in mass if account not in set(oversized)),
        key=account_order_key,
    )
    eligible_mass = sum(mass[account] for account in eligible)
    wanted = (test_share + val_share) * total
    if eligible_mass < wanted:
        raise SplitError(
            "accounts eligible for eval hold {:.1%} of the corpus but val+test want {:.1%}; "
            "raise --max-eval-account-fraction or lower the shares".format(
                eligible_mass / total, test_share + val_share
            )
        )

    assignment: Dict[str, str] = {account: "train" for account in oversized}
    order: List[Tuple[str, str]] = []
    filled = 0
    target = test_share * total
    split = "test"
    for account in eligible:
        if split is None:
            assignment[account] = "train"
            order.append((account, "train"))
            continue
        assignment[account] = split
        order.append((account, split))
        filled += mass[account]
        if filled >= target:
            if split == "test":
                split, filled, target = "val", 0, val_share * total
            else:
                split = None
    if split is not None:
        raise SplitError(
            "ran out of eligible accounts while filling {}: the hash order reached "
            "only {:.1%} of its {:.1%} target".format(split, filled / total, target / total)
        )

    policy = {
        "unit": "uploader account (one choreographer), whole",
        "salt": SALT,
        "mass": "frames",
        "order": "sha256(SALT || account), greedy fill: test, then val, rest train",
        "max_eval_account_fraction": max_eval_account_fraction,
        "accounts_forced_to_train_as_oversized": [
            {"account": account, "frame_fraction": round(mass[account] / total, 4)}
            for account in oversized
        ],
        "oversized_rule_reason": (
            "an account larger than the eval target would *be* the eval split; "
            "'held out' would then name one choreographer, not the corpus"
        ),
        "hash_order": [{"account": account, "split": split} for account, split in order],
    }
    return assignment, policy


def _group_spans(rows: Iterable[Mapping[str, Any]], key: str,
                 split_of: Mapping[str, str]) -> Dict[str, List[str]]:
    spans: Dict[str, set] = collections.defaultdict(set)
    for row in rows:
        group = row.get(key)
        if group is None:
            continue
        spans[str(group)].add(split_of[recording_id(row)])
    return {group: sorted(values) for group, values in spans.items() if len(values) > 1}


def music_collisions(sequences: Sequence[Mapping[str, Any]],
                     split_of: Mapping[str, str],
                     verified_pairs: Optional[Sequence[FrozenSet[str]]] = None,
                     pairs_provenance: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """What this split can prove about audio shared across its boundary.

    Two identities, and the weaker one used to be the only one:

    ``music_sha256`` hashes the clip's own 35-D music array, so it collides only
    when two clips carry byte-identical audio features -- the same track at the
    same offset and length.  It is a *floor* on shared music, not a measure of
    it: the same song cut at a different second does not collide.

    ``tools/fingerprint_wild_music.py`` aligns chroma-CENS sequences by lag and
    finds the pairs the hash cannot: the same track at a different offset, in a
    different upload.  **Passing its pair list is what turns this section from a
    disclaimer into a measurement**, and the difference is not marginal -- on
    wild_v4 the hash found 1 crossing group while the fingerprint found 3,925
    crossing pairs over 3,316 clips.  This section said "no fingerprint yet"
    while that file already existed on disk; a claim a report cannot lose is
    exactly the shape CLAUDE.md 2 forbids, so the wording now depends on whether
    the measurement was supplied rather than asserting it cannot be.

    Still a floor even with the pairs: the fingerprint's own positive control
    recalls 0.564 of known-true pairs, so zero crossings here would mean no
    *provable* leak, never no leak.
    """
    by_hash: Dict[str, set] = collections.defaultdict(set)
    clips: Dict[str, List[str]] = collections.defaultdict(list)
    for row in sequences:
        digest = row.get("music_sha256")
        if not isinstance(digest, str):
            continue
        by_hash[digest].add(split_of[recording_id(row)])
        clips[digest].append(recording_id(row))
    crossing = {digest: sorted(splits) for digest, splits in by_hash.items() if len(splits) > 1}
    report: Dict[str, Any] = {
        "identity": "music_sha256 (content hash of the clip's own 35-D music array)",
        "distinct_values": len(by_hash),
        "sequences": len(sequences),
        "hashes_spanning_splits": len(crossing),
        "examples": {
            digest: {"splits": crossing[digest], "clips": sorted(clips[digest])}
            for digest in sorted(crossing)[:5]
        },
        "what_this_does_not_cover": (
            "the same backing track at a different offset hashes differently, so a "
            "shared song across the boundary is invisible to the hash; "
            "tools/fingerprint_wild_music.py measures it, and --music-pairs is how "
            "its answer reaches this report"
        ),
    }
    if verified_pairs is None:
        report["fingerprint"] = {
            "supplied": False,
            "reading": ("no pair list was passed, so the hash row above is the only "
                        "evidence here and it is a weak floor -- absence of a number "
                        "is not a zero"),
        }
        return report

    crossing_pairs, touched = [], set()
    per_boundary: Dict[str, int] = collections.defaultdict(int)
    for pair in verified_pairs:
        left, right = sorted(pair)
        left_split, right_split = split_of.get(left), split_of.get(right)
        if left_split is None or right_split is None or left_split == right_split:
            continue
        crossing_pairs.append((left, right))
        touched.update((left, right))
        per_boundary["{}_{}".format(*sorted((left_split, right_split)))] += 1
    per_split = collections.Counter(split_of[clip] for clip in touched)
    split_sizes = collections.Counter(split_of.values())
    # The corpus-wide fraction and the per-split fraction are different numbers
    # and the second is the one that matters: every headline figure is computed
    # on test, and on wild_v4 test is 52.4% flagged against a corpus-wide 24.1%.
    # Publishing only the corpus number invites quoting it for the split, which
    # under-states the leak on the split by a factor of two.
    per_split_fraction = {
        split: round(per_split.get(split, 0) / size, 4)
        for split, size in sorted(split_sizes.items()) if size
    }
    report["fingerprint"] = {
        "supplied": True,
        "identity": "chroma-CENS alignment by lag, above a calibrated operating point",
        "pairs_examined": len(verified_pairs),
        "pairs_crossing_a_split": len(crossing_pairs),
        "clips_touched_by_a_crossing_pair": len(touched),
        "clips_touched_fraction": round(len(touched) / max(len(sequences), 1), 4),
        "by_boundary": dict(sorted(per_boundary.items())),
        "clips_touched_by_split": dict(sorted(per_split.items())),
        "clips_touched_fraction_by_split": per_split_fraction,
        "examples": [{"clips": pair, "splits": sorted((split_of[pair[0]], split_of[pair[1]]))}
                     for pair in sorted(crossing_pairs)[:5]],
        "provenance": dict(pairs_provenance or {}),
        "reading": ("a lower bound on shared music across the boundary: the fingerprint "
                    "cannot link two cuts that share no audio (its own positive control "
                    "recalls 0.564), so zero would mean no *provable* leak, not no leak"),
    }
    return report


def account_name_prefixes(account_split: Mapping[str, str],
                          mass: Mapping[str, int]) -> Dict[str, Any]:
    """Accounts whose names share a prefix, and how that prefix straddles the split.

    Account disjointness is enforced and provable.  What it does *not* buy is
    independence between the sides, and the corpus says so in the one place it
    can be read for free: 19 of 25 accounts are ``O-DOG编舞师-*``, so 13 of them
    are in train while 3 sit in each eval split.  Choreographers of one studio
    share a room, a camera, and a pool of trending tracks -- which is also the
    most likely source of the cross-split same-track pairs the fingerprint
    finds.  Reported so "account-disjoint" is not read as "independent".

    This is an observation about *names*, not an org chart: the corpus carries
    no studio field, and this function invents none.  A prefix is only reported
    when at least two accounts share it, and its being meaningful is a judgement
    for whoever reads the names.
    """
    total = max(sum(mass.values()), 1)
    # Every separator-boundary prefix, not one cut per name.  A single cut is
    # wrong in both directions here: cutting at the first separator turns
    # ``O-DOG编舞师-LEO`` into ``O``, and cutting at the last one turns
    # ``O-DOG编舞师-QZIKA_琴子💙`` into ``O-DOG编舞师-QZIKA`` -- a group of one,
    # which silently drops that account from the affiliation it obviously shares
    # and under-reports the span by its 4.6% of the corpus.  Enumerating every
    # boundary has no such blind spot.
    grouped: Dict[str, List[str]] = collections.defaultdict(list)
    for account in account_split:
        for cut in (match.start() for match in re.finditer(r"[-_/|]", account)):
            head = account[:cut].strip()
            if head:
                grouped[head].append(account)
    # ``O`` and ``O-DOG编舞师`` cover the same 19 accounts, so only the longest
    # is reported: nested prefixes over one account set are one finding, and
    # printing both invites reading them as two.
    by_members: Dict[FrozenSet[str], str] = {}
    for prefix, accounts in grouped.items():
        if len(accounts) < 2:
            continue
        members = frozenset(accounts)
        if len(prefix) > len(by_members.get(members, "")):
            by_members[members] = prefix
    spans = []
    for members, prefix in sorted(by_members.items(),
                                  key=lambda item: (-len(item[0]), item[1])):
        accounts = sorted(members)
        per_split = {split: {"accounts": 0, "frames": 0} for split in SPLITS}
        for account in accounts:
            entry = per_split[account_split[account]]
            entry["accounts"] += 1
            entry["frames"] += mass[account]
        spans.append({
            "prefix": prefix,
            "accounts": len(accounts),
            "splits_spanned": sorted(split for split in SPLITS
                                     if per_split[split]["accounts"]),
            "by_split": {split: {"accounts": entry["accounts"],
                                 "frame_fraction_of_corpus": round(entry["frames"] / total, 4)}
                         for split, entry in per_split.items()},
        })
    return {
        "spans": spans,
        "prefixes_spanning_all_three_splits": [row["prefix"] for row in spans
                                               if len(row["splits_spanned"]) == 3],
        "reading": ("shared name prefix is not a verified affiliation, but a prefix "
                    "spanning all three splits means account disjointness did not buy "
                    "independence between them; report it beside the split, not instead "
                    "of it"),
    }


def summarise(sources: Sequence[Mapping[str, Any]],
              frames_of: Mapping[str, int],
              account_of: Mapping[str, str],
              split_of: Mapping[str, str]) -> Dict[str, Any]:
    clips: Dict[str, int] = {split: 0 for split in SPLITS}
    frames: Dict[str, int] = {split: 0 for split in SPLITS}
    accounts: Dict[str, set] = {split: set() for split in SPLITS}
    uploads: Dict[str, set] = {split: set() for split in SPLITS}
    for row in sources:
        name = recording_id(row)
        split = split_of[name]
        clips[split] += 1
        frames[split] += frames_of[name]
        accounts[split].add(account_of[name])
        uploads[split].add(upload_key(row))
    total_clips = sum(clips.values())
    total_frames = sum(frames.values())
    summary: Dict[str, Any] = {
        "splits": {
            split: {
                "clips": clips[split],
                "clip_fraction": round(clips[split] / total_clips, 4) if total_clips else 0.0,
                "frames": frames[split],
                "frame_fraction": round(frames[split] / total_frames, 4) if total_frames else 0.0,
                "hours": round(frames[split] / 30.0 / 3600.0, 2),
                "uploads": len(uploads[split]),
                "accounts": sorted(accounts[split]),
            }
            for split in SPLITS
        },
        "shared_accounts": {},
    }
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1:]:
            summary["shared_accounts"]["{}_{}".format(left, right)] = sorted(
                accounts[left] & accounts[right]
            )
    return summary


def rewrite_rows(rows: Sequence[Mapping[str, Any]], split_of: Mapping[str, str],
                 note: str) -> List[Dict[str, Any]]:
    rewritten = []
    for row in rows:
        updated = dict(row)
        updated["split"] = split_of[recording_id(row)]
        updated["split_note"] = note
        updated["split_status"] = SPLIT_STATUS
        rewritten.append(updated)
    return rewritten


def read_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise SplitError("missing manifest: {}".format(path))
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise SplitError("{}:{}: {}".format(path, number, error)) from error
    return rows


def write_jsonl(path: pathlib.Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def build(bundle: pathlib.Path, group_keys: pathlib.Path, output_bundle: pathlib.Path, *,
          test_share: float, val_share: float, max_eval_account_fraction: float,
          min_eval_accounts: int, share_tolerance: float,
          min_train_share: float = 0.40,
          music_pairs: Optional[pathlib.Path] = None) -> Dict[str, Any]:
    sources = read_jsonl(bundle / "sources.jsonl")
    sequences = read_jsonl(bundle / "sequences.jsonl")
    groups = json.loads(group_keys.read_text(encoding="utf-8"))
    if not isinstance(groups, dict) or not groups:
        raise SplitError("group-key file is not a non-empty object: {}".format(group_keys))

    account_of: Dict[str, str] = {}
    unresolved: List[str] = []
    for row in sources:
        name = recording_id(row)
        account = groups.get(upload_key(row))
        if not isinstance(account, str) or not account or account == "?":
            unresolved.append(name)
            continue
        account_of[name] = account
    if unresolved:
        raise SplitError(
            "{} clip(s) have no recorded account, e.g. {}; a '?' bucket here would "
            "make the split look account-disjoint while pooling every unknown "
            "uploader into one group".format(len(unresolved), ", ".join(unresolved[:5]))
        )

    frames_of: Dict[str, int] = {}
    for row in sequences:
        name = recording_id(row)
        count = row.get("frame_count")
        if not isinstance(count, int) or count < 1:
            raise SplitError("{} has no positive frame_count".format(name))
        frames_of[name] = count
    missing = sorted(set(account_of) - set(frames_of))
    if missing:
        raise SplitError(
            "{} source(s) have no sequence row, e.g. {}".format(len(missing), ", ".join(missing[:5]))
        )
    unknown = sorted(set(frames_of) - set(account_of))
    if unknown:
        raise SplitError(
            "{} sequence recording(s) are absent from sources.jsonl, e.g. {}".format(
                len(unknown), ", ".join(unknown[:5])
            )
        )

    mass: Dict[str, int] = collections.Counter()
    clip_mass: Dict[str, int] = collections.Counter()
    for name, account in account_of.items():
        mass[account] += frames_of[name]
        clip_mass[account] += 1

    account_split, policy = assign_accounts(
        mass, test_share=test_share, val_share=val_share,
        max_eval_account_fraction=max_eval_account_fraction,
    )
    split_of = {name: account_split[account] for name, account in account_of.items()}

    summary = summarise(sources, frames_of, account_of, split_of)
    for pair, shared in summary["shared_accounts"].items():
        if shared:
            raise SplitError("{} still share account(s): {}".format(pair, ", ".join(shared)))
    for split, target in (("test", test_share), ("val", val_share)):
        observed = summary["splits"][split]["frame_fraction"]
        if abs(observed - target) > share_tolerance:
            raise SplitError(
                "{} holds {:.1%} of frames, outside {:.1%} +/- {:.1%}".format(
                    split, observed, target, share_tolerance
                )
            )
        if len(summary["splits"][split]["accounts"]) < min_eval_accounts:
            raise SplitError(
                "{} holds {} account(s); at least {} are required or the split is one "
                "choreographer wearing the word {!r}".format(
                    split, len(summary["splits"][split]["accounts"]), min_eval_accounts, split
                )
            )

    # Train is checked last and separately, because none of the five invariants
    # above look at it and the shares only bound *eval*.  Measured 2026-08-21 on
    # this corpus: ``--test-share 0.44 --val-share 0.45 --max-eval-account-fraction
    # 0.27 --share-tolerance 0.11`` exits 0 and publishes train 0 / val 1,167 /
    # test 936, stamped ``split_status: account_disjoint_source_safe``.  2,812 of
    # the 127,952 parameter combinations that "succeed" have an empty train.  A
    # split with nothing to train on is not a split, and it is exactly the shape
    # CLAUDE.md 2 opens with: five gates that read like a check and cannot fail
    # on the thing that matters.
    train = summary["splits"].get("train", {})
    train_clips = int(train.get("clips", 0) or 0)
    train_fraction = float(train.get("frame_fraction", 0.0) or 0.0)
    if train_clips <= 0:
        raise SplitError(
            "train holds no clip at all; test={} val={} clip(s).  The eval shares "
            "asked for {:.0%} of the corpus".format(
                summary["splits"]["test"]["clips"], summary["splits"]["val"]["clips"],
                test_share + val_share))
    if train_fraction < min_train_share:
        raise SplitError(
            "train holds {:.1%} of frames, under the {:.1%} floor; raise it with "
            "--min-train-share if a split this small is intended".format(
                train_fraction, min_train_share))

    for key in ("retrieval_group_id", "duplicate_content_group_id"):
        for table, rows in (("sources", sources), ("sequences", sequences)):
            spans = _group_spans(rows, key, split_of)
            if spans:
                example = sorted(spans)[0]
                raise SplitError(
                    "{} {} spans splits, e.g. {} -> {}".format(
                        table, key, example, "/".join(spans[example])
                    )
                )

    pairs, pairs_provenance = (None, None)
    if music_pairs is not None:
        pairs, pairs_provenance = load_verified_music_pairs(music_pairs)
    music = music_collisions(sequences, split_of, pairs, pairs_provenance)
    forced = [entry["account"] for entry in policy["accounts_forced_to_train_as_oversized"]]
    # This note is stamped into every rewritten manifest row, so it is an input
    # to the sha256 that ``materialize_atomic_windows`` binds the normalizer and
    # every label row to.  **Do not put a measurement in it.**  Adding the
    # fingerprint's crossing count here would change sources.jsonl, change its
    # digest, and unbind a release and two trained checkpoints -- a data-format
    # change wearing the clothes of a documentation fix.  Measurements go in the
    # report, which nothing hashes.  (The claim itself is still accurate: this
    # corpus has no track id.  What was stale lived in the report's
    # ``what_this_does_not_cover``, and that is where it was fixed.)
    note = (
        "whole uploader accounts assigned by sha256({}) over frame mass; account sets "
        "are disjoint by construction, {} account(s) exceeding {:.0%} of the corpus are "
        "train-only by policy, and the split is NOT music-disjoint -- this corpus has "
        "no track id".format(SALT, len(forced), max_eval_account_fraction)
    )

    # Published by atomic rename, which is the convention every other bundle in
    # this repo records as ``immutable_new_directory_only_atomic_rename``.
    if output_bundle.exists():
        raise FileExistsError(output_bundle)
    staging = output_bundle.with_name(output_bundle.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=False)
    write_jsonl(staging / "sources.jsonl", rewrite_rows(sources, split_of, note))
    write_jsonl(staging / "sequences.jsonl", rewrite_rows(sequences, split_of, note))

    # The arrays stay where they are: a copy would duplicate 5.2 GB for two
    # rewritten manifests.  Recorded in the report because a tree whose payload
    # lives behind a symlink is censused as 0 bytes -- see CLAUDE.md 1.3.
    os.symlink(os.path.relpath(bundle / "sequences", output_bundle), staging / "sequences")
    staging.rename(output_bundle)

    total_frames = sum(mass.values())
    total_clips = sum(clip_mass.values())
    report = {
        "schema_version": "atomicdance-account-disjoint-split-v1",
        "input_bundle": str(bundle.resolve()),
        "output_bundle": str(output_bundle.resolve()),
        "group_keys": str(group_keys.resolve()),
        "payload_is_a_symlink_to": str((bundle / "sequences").resolve()),
        "policy": dict(policy, split_status=SPLIT_STATUS,
                       test_share=test_share, val_share=val_share,
                       min_eval_accounts=min_eval_accounts,
                       min_train_share=min_train_share,
                       share_tolerance=share_tolerance),
        "accounts": [
            {
                "account": account,
                "split": account_split[account],
                "clips": clip_mass[account],
                "clip_fraction": round(clip_mass[account] / total_clips, 4),
                "frames": mass[account],
                "frame_fraction": round(mass[account] / total_frames, 4),
            }
            for account in sorted(mass, key=lambda name: -mass[name])
        ],
        "summary": summary,
        "music_disjointness": music,
        "account_name_prefix_spans": account_name_prefixes(account_split, mass),
        "split_note": note,
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=pathlib.Path, required=True,
                        help="wild raw performance bundle to re-split")
    parser.add_argument("--group-keys", type=pathlib.Path, required=True,
                        help="upload id -> uploader account, as recorded at ingest")
    parser.add_argument("--output-bundle", type=pathlib.Path, required=True,
                        help="new bundle; refuses to overwrite an existing one")
    parser.add_argument("--test-share", type=float, default=0.10,
                        help="target fraction of frames held out for test")
    parser.add_argument("--val-share", type=float, default=0.10)
    parser.add_argument("--max-eval-account-fraction", type=float, default=0.15,
                        help="an account holding more than this is train-only")
    parser.add_argument("--min-eval-accounts", type=int, default=2,
                        help="fewest accounts an eval split may be built from")
    parser.add_argument("--min-train-share", type=float, default=0.40,
                        help="floor on train's share of frames.  None of the "
                             "other invariants looks at train at all -- the "
                             "shares bound eval -- so a configuration that hands "
                             "eval everything exits 0 and publishes a bundle "
                             "with nothing to train on, still stamped "
                             "source-safe")
    parser.add_argument("--share-tolerance", type=float, default=0.04,
                        help="how far the achieved eval share may sit from target")
    parser.add_argument("--music-pairs", type=pathlib.Path, default=None,
                        help="fingerprint_wild_music.py's verified pair list; without it "
                             "the music-disjointness section reports UNMEASURED rather "
                             "than a floor that reads like a clean result")
    parser.add_argument("--report", type=pathlib.Path, default=None)
    return parser


def main(argv: Sequence[str]) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build(
            args.bundle, args.group_keys, args.output_bundle,
            test_share=args.test_share, val_share=args.val_share,
            max_eval_account_fraction=args.max_eval_account_fraction,
            min_eval_accounts=args.min_eval_accounts,
            min_train_share=args.min_train_share,
            share_tolerance=args.share_tolerance,
            music_pairs=args.music_pairs,
        )
    except (SplitError, FileExistsError) as error:
        print("error: {}".format(error), file=sys.stderr)
        return 1
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
                               encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    for entry in report["policy"]["accounts_forced_to_train_as_oversized"]:
        print("train-only (oversized): {} at {:.1%} of frames".format(
            entry["account"], entry["frame_fraction"]))
    print("music hashes spanning splits: {} (floor, not a measure -- see report)".format(
        report["music_disjointness"]["hashes_spanning_splits"]))
    fingerprint = report["music_disjointness"]["fingerprint"]
    if fingerprint["supplied"]:
        print("verified same-track pairs crossing a split: {} over {} clip(s) ({:.1%})".format(
            fingerprint["pairs_crossing_a_split"],
            fingerprint["clips_touched_by_a_crossing_pair"],
            fingerprint["clips_touched_fraction"]))
        # Printed per split because the corpus number is the one that gets
        # quoted and the split number is the one that bounds a headline.
        print("  by split: {}".format(", ".join(
            "{} {:.1%}".format(split, fraction)
            for split, fraction in fingerprint["clips_touched_fraction_by_split"].items())))
    else:
        print("verified same-track pairs: UNMEASURED -- pass --music-pairs")
    for row in report["account_name_prefix_spans"]["prefixes_spanning_all_three_splits"]:
        print("account name prefix in all three splits: {}".format(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
