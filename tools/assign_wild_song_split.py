#!/usr/bin/env python3
"""Re-split the wild corpus so no backing track crosses the boundary.

The evidence this exists for, measured 2026-08-25 on ``wild_v4``: the planner's
per-frame labels agree with a *training* recording of the same song at 0.257,
with the clip's own labels at 0.088, and with a random training clip at 0.015 --
four clips reach 0.96-1.00, i.e. the plan is the training instance's plan.  The
retrieval library is ``train/motion.npy`` itself, so this is not a metric
artifact: the model was fitted on the thing it is being scored against.

The unit is therefore the **connected component of the same-track graph**, not
the recording and not the upload.  Two recordings the fingerprint links must
land together, and so must anything transitively linked to them; a split that
separates any pair inside a component has the leak it was drawn to remove.

That graph carries two kinds of edge, and it needs both.  A fingerprint pair is
*inferred*, and its recall is well under half, so it cannot be relied on to hold
even an upload together: the first wild_v5 run used fingerprint edges alone and
put **1,147 of 9,186 uploads on both sides of the boundary** -- 2,604 clips,
18.8% of the corpus, 45 uploads across all three splits -- which is the same
video in train and in test.  A same-upload edge is *exact*, because the upload
id is in the recording id, so those are added outright and the result is checked
per upload rather than assumed from the component check.

What this does not fix, stated because the number that matters is not the one
this tool improves
------------------------------------------------------------------------------
* **The same dancer is not the same song.**  One account is one choreographer
  and repeats their own movement vocabulary across different tracks.  Holding
  out songs leaves that channel completely open.  A doubly-disjoint split is not
  available on this corpus: taking song edges and account edges together
  collapses all 13,467 recordings into **one** component (measured), because
  295 pairs of the 25 accounts are joined by a shared track.  So this is a
  choice between two partial splits, not a step toward a complete one.
* **The fingerprint recovers well under half of what it should.**  Its own
  positive control -- two cuts of one upload are the same track by construction
  -- reads 1,758 of 5,408 same-upload candidate pairs on ``wild_v4`` (32.5%)
  and 1,167 of 2,708 on ``wild_v5`` (43.1%, 2026-08-25; higher because the
  re-cut clips align to each other where the mis-cut ones could not).  Neither
  number is quoted from here at run time: the report carries whichever one the
  ``--pairs`` file's own grouping run measured, because a criterion that states
  another corpus's number about this one is not stating a measurement.
  Components built from it are a lower bound on what shares a track, so a clean
  reading here is a *floor* on the leak, never a proof of independence.  ``tools/audit_split_leakage.py``
  measures the residual on poses, which is the check that can still fail after
  this tool has run, and it must be run afterwards.

  **And a floor built from one run is lower than it needs to be.**  ``--pairs``
  takes every list ever measured on this corpus, not the newest one, because two
  fingerprint runs over two cuts of the same videos do not miss the same pairs.
  Measured 2026-08-26, v4's list against v5's:

      v4 edges landing on the v5 bundle    14,437   (132 lost to the re-cut)
        also found by the v5 run           10,085
        found only by v4                    4,352
      v5 edges                             10,392
        found only by v5                      307

  The first v5 song split consumed the v5 list alone and was clean against it --
  0 straddling components, 0 straddling uploads.  Against v4's list **636 of its
  edges crossed that split, 281 of them test<->train**, touching 176 of the 1,384
  test clips (12.7%).  Neither run is wrong; each is partial, and taking the
  newest is discarding measurements.  Ids are matched with the generation prefix
  stripped (``wild_v4:<upload>:clip000`` vs ``wild_v5:...``), and a list that
  lands on no recording of this bundle is refused rather than counted as zero
  edges -- a mistyped path and a list with nothing to add read identically
  otherwise.

Assignment is deterministic: components are ordered by size (largest first,
ties broken by a hash of the smallest member so the order does not depend on
dictionary iteration), and each is placed in whichever split is furthest below
its target share.  Largest-first matters -- the biggest component holds 16.9%
of the corpus, and placing it last would force it into whichever split had room
rather than the one that can absorb it.

Usage::

    assign_wild_song_split.py --bundle data/wild3d/wild_v5_performance \\
        --pairs runs/wild_v4_music_groups_pairs.jsonl \\
        --output-bundle data/wild3d/wild_v5_song_performance \\
        --group-keys runs/wild_v4_group_keys.json \\
        --report runs/wild_v5_song_split.json
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import pathlib
import shutil
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# One reader per artifact, imported rather than re-written: this repo has
# already paid for two parsers disagreeing about which dances share a song.
from tools.assign_account_disjoint_split import (  # noqa: E402
    SplitError, read_jsonl, recording_id, rewrite_rows, upload_key, write_jsonl)
from tools.audit_split_leakage import (  # noqa: E402
    components, merge_pair_files, read_pairs, strip_generation)

SPLITS = ("train", "val", "test")
SPLIT_STATUS = "frozen_recording_song_disjoint"
SPLIT_NOTE = ("component of the fingerprint same-track graph; song-disjoint, "
              "NOT dancer-disjoint")
# Frozen so the order is reproducible from the component members alone.
SALT = "atomicdance-song-disjoint-v1"


def order_key(group: Sequence[str]) -> tuple:
    """Largest first, then by a salted hash of the smallest member.

    The hash is only a tie-break.  Ordering by size alone leaves components of
    equal size in whatever order the component walk produced, which depends on
    dictionary iteration and therefore on the input file's line order -- so the
    same corpus could be split two ways and neither run would say which.
    """
    digest = hashlib.sha256((SALT + "|" + min(group)).encode("utf-8")).hexdigest()
    return (-len(group), digest)


def assign(groups: Sequence[Sequence[str]], shares: Mapping[str, float]
           ) -> Dict[str, str]:
    """Each component to the split furthest below its share of the recordings."""
    total = sum(len(group) for group in groups)
    targets = {name: shares[name] * total for name in SPLITS}
    held = {name: 0 for name in SPLITS}
    split_of: Dict[str, str] = {}
    for group in sorted(groups, key=order_key):
        # "Furthest below target" in *recordings*, not in fraction: a fractional
        # deficit makes the smallest split win every early round and swallow the
        # large components, which is how the biggest one would end up in test.
        pick = max(SPLITS, key=lambda name: (targets[name] - held[name], name))
        held[pick] += len(group)
        for member in group:
            split_of[member] = pick
    return split_of


def read_pins(path: pathlib.Path) -> Dict[str, str]:
    """recording_id -> split from an already-published split bundle's manifest."""
    pins: Dict[str, str] = {}
    for row in read_jsonl(pathlib.Path(path)):
        name, side = recording_id(row), row.get("split")
        if side not in SPLITS:
            raise SplitError("pin file {} gives {!r} split {!r}".format(path, name, side))
        if pins.setdefault(name, side) != side:
            raise SplitError("pin file {} gives {!r} two splits".format(path, name))
    if not pins:
        raise SplitError("pin file {} names no recording".format(path))
    return pins


def assign_pinned(adjacency: Mapping[str, Any], universe: set, pins: Mapping[str, str],
                  unpinned_to: str, shares: Mapping[str, float], drop_bridging: bool
                  ) -> tuple:
    """Keep every pinned recording where it is and place only the new ones.

    Why this exists: re-running ``assign`` with new recordings re-places *every*
    component, because each new one changes the deficits seen by everything
    after it.  On the T line (2026-09-02) going from 282 to 295 recordings kept
    8 of the 20 test clips; simulated on 2026-09-22, adding 17-34 clips moved
    45-82 old ones and kept 0-9 of ``runs/eval_clips_txy_t20.txt`` in test.
    The eval list is the ruler (CLAUDE.md 1.5.7) and every checkpoint was
    trained on the old train split, so appending must not move either.

    Per same-track component, by the set of pinned splits among its members:

    * one split -- every member takes it.  A new clip sharing a song with a
      test clip goes to test: song-disjointness is the property this tool
      exists for, and it outranks a round test count.
    * none (all new) -- ``unpinned_to``: a named split, or ``assign`` for the
      usual largest-deficit placement counted on top of the pinned mass.
    * two or more, with a new member -- the new clip would join two sides.
      Refused, or with ``drop_bridging`` the new members are left out of the
      output and listed; either way no pin moves.
    * two or more, pinned members only -- a straddle the pinned split already
      had (a fresh fingerprint can see pairs the old one missed).  Kept as
      pinned and reported, because moving an old clip is exactly what pinning
      forbids; it is the operator's call, not this tool's.
    """
    live = set(universe)
    dropped: List[Dict[str, Any]] = []
    while True:
        groups = components(adjacency, live)
        bridging = []
        for group in groups:
            sides = {pins[m] for m in group if m in pins}
            new = [m for m in group if m not in pins]
            if len(sides) > 1 and new:
                bridging.append((group, sides, new))
        if not bridging:
            break
        if not drop_bridging:
            group, sides, new = bridging[0]
            raise SplitError(
                "{} component(s) join new recording(s) to more than one pinned split, e.g. "
                "{} links {} to pinned {}.  Placing them would put one song on two sides; "
                "pass --drop-bridging to leave them out instead.".format(
                    len(bridging), new[:3], sorted(m for m in group if m in pins)[:4],
                    sorted(sides)))
        for group, sides, new in bridging:
            for member in new:
                dropped.append({"recording_id": member, "component_size": len(group),
                                "pinned_splits_in_component": sorted(sides)})
                live.discard(member)

    split_of: Dict[str, str] = {}
    preexisting: List[Dict[str, Any]] = []
    unpinned_groups: List[List[str]] = []
    followed = collections.Counter()
    for group in groups:
        sides = {pins[m] for m in group if m in pins}
        if len(sides) == 1:
            side = next(iter(sides))
            for member in group:
                split_of[member] = side
                if member not in pins:
                    followed[side] += 1
        elif not sides:
            unpinned_groups.append(group)
        else:
            for member in group:
                split_of[member] = pins[member]
            preexisting.append({"members": {m: pins[m] for m in group}})
    placed = collections.Counter()
    if unpinned_to == "assign":
        total = len(live)
        targets = {name: shares[name] * total for name in SPLITS}
        held = collections.Counter(split_of.values())
        for group in sorted(unpinned_groups, key=order_key):
            pick = max(SPLITS, key=lambda name: (targets[name] - held[name], name))
            held[pick] += len(group)
            for member in group:
                split_of[member] = pick
                placed[pick] += 1
    else:
        for group in unpinned_groups:
            for member in group:
                split_of[member] = unpinned_to
                placed[unpinned_to] += 1
    summary = {"pinned": sum(1 for m in live if m in pins),
               "new": sum(1 for m in live if m not in pins),
               "new_following_a_pinned_component": dict(followed),
               "new_in_all_new_components": dict(placed),
               "unpinned_to": unpinned_to,
               "dropped_bridging": dropped,
               "preexisting_pinned_conflicts": preexisting}
    return split_of, groups, live, summary


def upload_edges(sequences: Sequence[Mapping[str, Any]]) -> Dict[str, set]:
    """Join every pair of clips cut from one upload.

    Two clips of one upload are one recording of one performance to one track;
    nothing about them is independent.  The fingerprint is supposed to catch
    that -- same-upload pairs are its own positive control -- but it recovers
    only 43.1% of the ones it even considers (wild_v5, 2026-08-25), so relying
    on it to hold an upload together leaves the rest free to be split.

    Measured on the first wild_v5 song split, which used fingerprint edges
    alone: **1,147 of 9,186 uploads had their clips land in more than one
    split** -- 2,604 clips, 18.8% of the corpus, and 45 uploads spread across
    all three.  The pose audit surfaced one directly
    (``wild_v5:7361306660742155557`` clip001 in test, clip000 in train, ratio
    0.5035), which is how it was found.

    These edges are exact rather than inferred, so they cost nothing to trust:
    the upload id is in the recording id.  The account split never had this
    problem because it assigns whole accounts, and an upload has one account.
    """
    by_upload: Dict[str, List[str]] = collections.defaultdict(list)
    for row in sequences:
        by_upload[upload_key(row)].append(recording_id(row))
    adjacency: Dict[str, set] = collections.defaultdict(set)
    for members in by_upload.values():
        if len(members) < 2:
            continue
        first = members[0]
        for other in members[1:]:      # a star is enough; components does the rest
            adjacency[first].add(other)
            adjacency[other].add(first)
    return adjacency


def merge_adjacency(*graphs: Mapping[str, Any]) -> Dict[str, set]:
    """Union of undirected edge sets, keyed the way ``components`` expects."""
    merged: Dict[str, set] = collections.defaultdict(set)
    for graph in graphs:
        for node, neighbours in graph.items():
            merged[node].update(neighbours)
    return merged


def fingerprint_recall(pairs: pathlib.Path) -> Optional[Mapping[str, Any]]:
    """The positive control measured by the run that produced ``pairs``.

    ``fingerprint_wild_music.py group`` writes ``<name>.json`` beside
    ``<name>_pairs.jsonl`` and puts its same-upload recall in it.  Reading it
    back is the difference between reporting what this corpus measured and
    reporting a number that was true of a different one: on wild_v4 the recall
    is 32.5% and on wild_v5 it is 43.1%, so a hardcoded figure would have been
    wrong by a third the first time the corpus was re-cut.
    """
    name = pairs.name
    if not name.endswith("_pairs.jsonl"):
        return None
    sibling = pairs.with_name(name[: -len("_pairs.jsonl")] + ".json")
    if not sibling.is_file():
        return None
    try:
        control = json.loads(sibling.read_text(encoding="utf-8")).get("positive_control")
    except (ValueError, OSError):
        return None
    return control if isinstance(control, Mapping) else None


def read_pair_files(paths: Sequence[pathlib.Path], universe: Sequence[str]
                    ) -> tuple:
    """Every measured same-track list, mapped onto this bundle's recording ids.

    The merge itself lives in ``audit_split_leakage.merge_pair_files`` and is
    imported rather than repeated: this repo has already paid for two parsers
    disagreeing about which dances share a song, and the auditor and the
    splitter disagreeing would be the same bill again -- the audit is supposed
    to be able to fail on the split's own evidence.

    What is added here is the refusal.  A list that *has* rows and lands none of
    them is a wrong file; a list with no rows at all is a corpus with no
    fingerprint evidence, which is a real state the same-upload edges still
    cover.  Raising the same way for both would merge "we were handed less
    evidence than we thought" into "there is no evidence", and nothing
    downstream would say so.
    """
    merged, _lags, per_file = merge_pair_files(paths, universe)
    for entry in per_file:
        if entry["rows"] and not entry["edges_on_this_bundle"]:
            raise SplitError(
                "{} names no pair of this bundle ({} row(s) read, 0 landing).  Ids are "
                "matched with the generation prefix stripped, so this is a wrong file "
                "rather than an empty one.".format(entry["path"], entry["rows"]))
    return merged, per_file


def straddling(groups: Sequence[Sequence[str]], split_of: Mapping[str, str]) -> List[list]:
    """Components whose members did not all land on the same side.

    True by construction of ``assign`` -- which is exactly why it is checked.
    The construction is the claim, and a claim nothing tests is how this repo
    keeps acquiring gates that cannot fail.
    """
    bad = []
    for group in groups:
        sides = {split_of[member] for member in group}
        if len(sides) > 1:
            bad.append(sorted(group))
    return bad


def account_report(sources: Sequence[Mapping], split_of: Mapping[str, str],
                   group_keys: Optional[pathlib.Path]) -> Dict:
    """Who is in each split, which this split does not control.

    Reported rather than enforced.  The account channel is open by construction
    here, so a number that describes how open is worth more than a threshold
    that pretends to close it.
    """
    if group_keys is None or not pathlib.Path(group_keys).is_file():
        return {"available": False,
                "why": "no --group-keys, so the account channel is unmeasured "
                       "here.  That is not the same as closed."}
    keys = json.loads(pathlib.Path(group_keys).read_text(encoding="utf-8"))
    per_split = {name: collections.Counter() for name in SPLITS}
    unresolved = 0
    for row in sources:
        name = recording_id(row)
        if name not in split_of:
            continue
        account = keys.get(upload_key(row))
        if not isinstance(account, str) or not account or account == "?":
            unresolved += 1
            continue
        per_split[split_of[name]][account] += 1
    out = {"available": True, "unresolved_uploads": unresolved,
           "accounts_total": len({a for c in per_split.values() for a in c})}
    for name in SPLITS:
        counter = per_split[name]
        total = sum(counter.values()) or 1
        top = counter.most_common(1)
        out[name] = {"accounts": len(counter), "recordings": sum(counter.values()),
                     "largest_account_share": round(top[0][1] / total, 4) if top else None}
    out["accounts_in_every_split"] = len(
        set(per_split["train"]) & set(per_split["val"]) & set(per_split["test"]))
    return out


def build(bundle: pathlib.Path, pairs: Sequence[pathlib.Path],
          output_bundle: pathlib.Path, *,
          test_share: float, val_share: float,
          group_keys: Optional[pathlib.Path] = None,
          pin: Optional[pathlib.Path] = None, unpinned_to: Optional[str] = None,
          drop_bridging: bool = False, allow_missing_pins: bool = False) -> Dict:
    sources = read_jsonl(bundle / "sources.jsonl")
    sequences = read_jsonl(bundle / "sequences.jsonl")
    universe = {recording_id(row) for row in sequences}
    pins: Dict[str, str] = {}
    pinning: Optional[Dict[str, Any]] = None
    if pin is not None:
        if unpinned_to not in SPLITS + ("assign",):
            raise SplitError("--pin needs --unpinned-to train|val|test|assign; there is no "
                             "default, because where all-new songs go is a decision")
        pins = read_pins(pin)
        absent = sorted(name for name in pins if name not in universe)
        if absent and not allow_missing_pins:
            raise SplitError(
                "{} pinned recording(s) are not in {}, e.g. {} ({} of them test).  A pinned "
                "eval clip that vanishes changes the ruler; pass --allow-missing-pins only "
                "if that is intended.".format(len(absent), bundle, absent[:4],
                                              sum(1 for n in absent if pins[n] == "test")))
        pins = {name: side for name, side in pins.items() if name in universe}
        pinning = {"pin_file": str(pin), "pins_absent_from_bundle": absent}
    elif unpinned_to is not None or drop_bridging:
        raise SplitError("--unpinned-to / --drop-bridging only mean something with --pin")
    if not universe:
        raise SplitError("{} names no sequence".format(bundle))
    by_stripped: Dict[str, str] = {}
    for name in universe:
        by_stripped.setdefault(strip_generation(name), name)

    pairs = [pathlib.Path(path) for path in
             ([pairs] if isinstance(pairs, (str, pathlib.Path)) else pairs)]
    fingerprint_adjacency, per_file = read_pair_files(pairs, universe)
    pair_rows = sum(entry["rows"] for entry in per_file)
    same_upload = upload_edges(sequences)
    adjacency = merge_adjacency(fingerprint_adjacency, same_upload)
    groups = components(adjacency, universe)
    fingerprint_only = components(fingerprint_adjacency, universe)
    shares = {"test": test_share, "val": val_share,
              "train": 1.0 - test_share - val_share}
    if shares["train"] <= 0:
        raise SplitError("test + val shares leave no training split")
    preexisting_members: set = set()
    if pins:
        split_of, groups, universe, pin_summary = assign_pinned(
            adjacency, universe, pins, unpinned_to, shares, drop_bridging)
        pinning.update(pin_summary)
        for conflict in pin_summary["preexisting_pinned_conflicts"]:
            preexisting_members.update(conflict["members"])
        kept = {row_name for row_name in universe}
        sequences = [row for row in sequences if recording_id(row) in kept]
        sources = [row for row in sources if recording_id(row) in kept]
        moved = sorted(name for name, side in pins.items() if split_of.get(name) != side)
        if moved:     # true by construction; checked because the construction is the claim
            raise SplitError("{} pinned recording(s) moved, e.g. {}".format(len(moved), moved[:4]))
    else:
        split_of = assign(groups, shares)

    missing = sorted(universe - set(split_of))
    if missing:
        raise SplitError("{} recording(s) were never assigned, e.g. {}".format(
            len(missing), ", ".join(missing[:5])))
    # A pre-existing straddle among pinned recordings is reported, not refused:
    # refusing would force moving an old clip, which is what pinning forbids.
    crossed = [group for group in straddling(groups, split_of)
               if not set(group) <= preexisting_members]
    if crossed:
        raise SplitError(
            "{} component(s) straddle the split after assignment, e.g. {}.  That "
            "is the defect this tool exists to remove, so it refuses rather than "
            "publishing.".format(len(crossed), crossed[0][:4]))

    # Checked separately from the component check above, because the two can
    # disagree only if the edge set is wrong -- and it was, on the first run.
    # An upload straddling the split is the same video on both sides, so this
    # refuses rather than reporting it.
    per_upload: Dict[str, set] = collections.defaultdict(set)
    for row in sequences:
        name = recording_id(row)
        if name in split_of:
            per_upload[upload_key(row)].add(split_of[name])
    torn = sorted(upload for upload, sides in per_upload.items() if len(sides) > 1)
    if torn:
        raise SplitError(
            "{} upload(s) have clips in more than one split, e.g. {}.  Two clips "
            "of one upload are one recording of one performance; the edge set "
            "that let them apart is the defect, not the assignment."
            .format(len(torn), torn[:4]))

    # Restated per file rather than inferred from the component check.  The
    # components were built from the merged graph, so this cannot fail unless
    # the merge dropped an edge -- which is exactly the failure worth having a
    # separate reading for, and it is the failure that put 281 test<->train
    # edges through the first v5 song split.
    for entry in per_file:
        adjacency_here, _lags, _rows = read_pairs(pathlib.Path(entry["path"]))
        crossing = 0
        between_pins = 0
        seen = set()
        for left, neighbours in adjacency_here.items():
            for right in neighbours:
                edge = frozenset((strip_generation(left), strip_generation(right)))
                if edge in seen:
                    continue
                seen.add(edge)
                ends = [name for name in (by_stripped.get(end) for end in edge)
                        if name in split_of]
                sides = {split_of[name] for name in ends}
                if len(sides) > 1:
                    if len(ends) == 2 and all(name in pins for name in ends):
                        between_pins += 1      # the pinned split's own, reported above
                    else:
                        crossing += 1
        entry["edges_crossing_the_split"] = crossing
        if pins:
            entry["edges_crossing_between_pinned_recordings"] = between_pins
        if crossing:
            raise SplitError(
                "{} edge(s) of {} cross the split after assignment.  Every list handed to "
                "--pairs is evidence the split has to respect; an edge that crosses means "
                "it was read but not merged.".format(crossing, entry["path"]))

    sizes = collections.Counter(len(group) for group in groups)
    control = next((fingerprint_recall(pathlib.Path(entry["path"]))
                    for entry in per_file
                    if fingerprint_recall(pathlib.Path(entry["path"]))), None)
    if control:
        missed_note = (
            "anything the fingerprint missed: on this corpus it recovered {} of "
            "{} same-upload candidate pairs ({:.1%}), so these components are a "
            "lower bound on what shares a track".format(
                control.get("same_upload_verified"),
                control.get("same_upload_candidate_pairs"),
                float(control.get("recall_on_same_upload_pairs", 0.0))))
    else:
        # Saying "unknown" beats quoting another corpus's 32.5%: the reader can
        # go and measure it, which a wrong number does not prompt anyone to do.
        missed_note = (
            "anything the fingerprint missed: no positive control was found "
            "beside any of {}, so the floor this split rests on is unmeasured "
            "here".format(", ".join(str(path) for path in pairs)))
    report = {
        "generated_by": "tools/assign_wild_song_split.py",
        "bundle": str(bundle),
        "pairs": [str(path) for path in pairs],
        "pairs_read": per_file,
        "pair_rows": pair_rows,
        "recordings": len(universe),
        "components": len(groups),
        "components_from_fingerprint_edges_alone": len(fingerprint_only),
        "edges": {
            "fingerprint_pairs": pair_rows,
            "same_upload_recordings": sum(len(v) for v in same_upload.values()) // 2,
            "why_both": "a fingerprint pair is inferred and its recall is well "
                        "under half; a same-upload edge is exact.  Fingerprint "
                        "edges alone let 1,147 uploads straddle the split on the "
                        "first wild_v5 run (2,604 clips, 18.8% of the corpus)",
        },
        "largest_component": max((len(g) for g in groups), default=0),
        "singletons": sizes.get(1, 0),
        "requested_shares": shares,
        "by_split": {name: sum(1 for s in split_of.values() if s == name)
                     for name in SPLITS},
        "components_straddling": 0,
        "components_straddling_preexisting_between_pins": len(
            (pinning or {}).get("preexisting_pinned_conflicts", [])),
        "pinning": pinning,
        "uploads_straddling": 0,          # checked above, not assumed
        "uploads": len(per_upload),
        "accounts": account_report(sources, split_of, group_keys),
        "does_not_fix": [
            "the same dancer across different backing tracks -- one account is "
            "one choreographer, and account+song edges together collapse this "
            "corpus into a single component, so a doubly-disjoint split does "
            "not exist here",
            missed_note,
        ],
        "next_check": "tools/audit_split_leakage.py --bundle <output> --pairs "
                      "<pairs>: the pose criterion is what can still fail here",
    }
    report["achieved_shares"] = {
        name: round(report["by_split"][name] / len(universe), 4) for name in SPLITS}

    output_bundle = pathlib.Path(output_bundle)
    staging = output_bundle.with_name(output_bundle.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    note = SPLIT_NOTE if not pins else (
        SPLIT_NOTE + "; pinned: recordings of {} keep their split, new ones follow their "
        "component's pinned split, else {}".format(pin, unpinned_to))
    write_jsonl(staging / "sequences.jsonl", rewrite_rows(sequences, split_of, note))
    write_jsonl(staging / "sources.jsonl", rewrite_rows(sources, split_of, note))
    # The payload is shared with the input bundle, not copied: only the split
    # labels differ.  A symlink, and recorded as one -- a tree that lives behind
    # a symlink is censused as 0 bytes and ``evict`` would call it safe to
    # delete (CLAUDE.md 1.3).
    os.symlink(os.path.relpath(bundle / "sequences", output_bundle),
               staging / "sequences")
    report["payload_is_a_symlink_to"] = str((bundle / "sequences").resolve())
    if output_bundle.exists():
        shutil.rmtree(output_bundle)
    staging.rename(output_bundle)
    report["output_bundle"] = str(output_bundle)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=pathlib.Path, required=True)
    parser.add_argument("--pairs", type=pathlib.Path, required=True, nargs="+",
                        help="fingerprint-confirmed same-track pairs, jsonl")
    parser.add_argument("--output-bundle", type=pathlib.Path, required=True)
    parser.add_argument("--group-keys", type=pathlib.Path, default=None,
                        help="upload -> account, used only to report how open "
                             "the account channel is left")
    parser.add_argument("--test-share", type=float, default=0.10)
    parser.add_argument("--val-share", type=float, default=0.10)
    parser.add_argument("--report", type=pathlib.Path, default=None)
    parser.add_argument("--pin", type=pathlib.Path, default=None,
                        help="sequences.jsonl of a published split: its recordings keep "
                             "their split; only recordings not in it are placed")
    parser.add_argument("--unpinned-to", choices=SPLITS + ("assign",), default=None,
                        help="with --pin: where components with no pinned member go")
    parser.add_argument("--drop-bridging", action="store_true",
                        help="with --pin: leave out new recordings that join two pinned "
                             "splits, instead of refusing")
    parser.add_argument("--allow-missing-pins", action="store_true",
                        help="with --pin: accept pinned recordings absent from --bundle")
    args = parser.parse_args(argv)

    report = build(args.bundle, args.pairs, args.output_bundle,
                   test_share=args.test_share, val_share=args.val_share,
                   group_keys=args.group_keys, pin=args.pin,
                   unpinned_to=args.unpinned_to, drop_bridging=args.drop_bridging,
                   allow_missing_pins=args.allow_missing_pins)
    if report.get("pinning"):
        p = report["pinning"]
        print("pinned {} kept in place; {} new: {} followed a pinned component, {} all-new -> "
              "{}; {} dropped as bridging; {} pre-existing straddle(s) between pins kept".format(
                  p["pinned"], p["new"], p["new_following_a_pinned_component"],
                  p["new_in_all_new_components"], p["unpinned_to"],
                  len(p["dropped_bridging"]), len(p["preexisting_pinned_conflicts"])))

    print("{} recording(s) in {} same-track component(s); largest {}".format(
        report["recordings"], report["components"], report["largest_component"]))
    for name in SPLITS:
        print("  {:5s} {:6d}  ({:.1%} against a target of {:.0%})".format(
            name, report["by_split"][name], report["achieved_shares"][name],
            report["requested_shares"][name]))
    accounts = report["accounts"]
    if accounts.get("available"):
        print("  accounts: {} total, {} present in all three splits".format(
            accounts["accounts_total"], accounts["accounts_in_every_split"]))
        for name in SPLITS:
            print("    {:5s} {} account(s), largest holds {:.1%}".format(
                name, accounts[name]["accounts"],
                accounts[name]["largest_account_share"] or 0.0))
    print("components straddling the split: 0 (checked, not assumed)")
    print("NOT fixed by this split:")
    for line in report["does_not_fix"]:
        print("  - {}".format(line))
    print("next: {}".format(report["next_check"]))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        print("wrote", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
