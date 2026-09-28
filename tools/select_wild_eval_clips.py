#!/usr/bin/env python3
"""Choose which wild test clips M6 generates for, and mark which of them leak.

Two decisions live here rather than in a driver script, because both have to be
auditable after the fact.

**How many, and which.**  Generating all 1,575 test clips at four seeds is 6,300
sequences; FID does not need that and the cards do.  The sample is drawn by
``sha256(SALT || recording_id)`` order -- the same idiom the account split uses
-- so it is reproducible from the names alone, and re-running with a larger
``--count`` *extends* the previous sample rather than replacing it.  That
property matters: two checkpoints compared on different clip sets are not
compared at all, and FID is biased by sample size, so the count is pinned in the
report and must be held fixed across any within-corpus comparison.

The draw is uniform over the split, deliberately.  A stratified or leak-free
draw would make the eval set stop resembling the split it is drawn from, and the
question M6 answers is about the corpus, not about a curated part of it.

**Which clips leak.**  ``tools/fingerprint_wild_music.py`` proves 3,925 pairs
that share a backing track across a split boundary, touching 825 test clips --
24.1% of the corpus.  The account split is account-disjoint and was never
music-disjoint, and that is measured, not feared.  Flagging the clips *at
selection time* costs nothing and buys the one thing that is expensive later:
FID and R can be recomputed on the leak-free subset **without generating
anything again**.  Discovering afterwards that the number needs a clean subset,
with no flags recorded, means paying for the whole generation twice.

The flag is a floor, not a verdict.  The fingerprint's own positive control
recalls 0.564 of known-true pairs, so a clip marked clean is one with no
*provable* shared track, never one proven independent.  Reports say so.

Usage::

    python3 tools/select_wild_eval_clips.py \\
        --bundle /dev/shm/atomicdance-acct/performance --split test \\
        --music-pairs runs/wild_v4_music_groups_pairs.jsonl \\
        --count 400 --output runs/wild_v4_acct_m6_clips
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import sys
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.eval_r_precision import load_verified_music_pairs  # noqa: E402

# Distinct from the split's salt: reusing it would make the eval draw correlated
# with the train/val/test assignment, so the sample would not be a sample of the
# split but a systematic slice of it.
SALT = "atomicdance-wild-m6-selection-v1"


class SelectionError(RuntimeError):
    pass


def order_key(name: str) -> str:
    return hashlib.sha256((SALT + "\0" + name).encode("utf-8")).hexdigest()


def leaked_clips(pairs: Sequence[frozenset], split_of: Dict[str, str]) -> Dict[str, List[str]]:
    """Clip -> the splits it provably shares a backing track with, across the line."""
    touched: Dict[str, set] = collections.defaultdict(set)
    for pair in pairs:
        left, right = sorted(pair)
        left_split, right_split = split_of.get(left), split_of.get(right)
        if left_split is None or right_split is None or left_split == right_split:
            continue
        touched[left].add(right_split)
        touched[right].add(left_split)
    return {clip: sorted(splits) for clip, splits in touched.items()}


def select(bundle: pathlib.Path, split: str, count: Optional[int],
           music_pairs: Optional[pathlib.Path],
           restrict_to: Optional[pathlib.Path] = None) -> Dict[str, object]:
    rows = []
    for line in (bundle / "sequences.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    split_of = {str(row["recording_id"]): str(row["split"]) for row in rows}
    candidates = sorted((str(row["recording_id"]) for row in rows
                         if str(row.get("split")) == split), key=order_key)
    if not candidates:
        raise SelectionError("{} names no clip in split {!r}".format(bundle, split))
    if count is not None and count > len(candidates):
        raise SelectionError("asked for {} clips, split {} holds {}".format(
            count, split, len(candidates)))
    # ``--restrict-to`` narrows only the *draw*.  Every split-level statistic
    # below -- above all ``flagged_clips_in_split``, which is what the leak-free
    # FID filters the ground truth by -- keeps using the whole split, because
    # filtering the reference distribution by a restricted pool would leave the
    # rest of the split's flagged clips sitting in the "clean" reference.  That
    # exact mistake was priced at 618 of 825 clips on wild_v4.
    restriction: Dict[str, object] = {"applied": False}
    pool = candidates
    if restrict_to is not None:
        allowed = {line.strip() for line in
                   restrict_to.read_text(encoding="utf-8").splitlines() if line.strip()}
        pool = [clip for clip in candidates if clip in allowed]
        if not pool:
            raise SelectionError(
                "{} names no clip of the {} split; a restriction that selects "
                "nothing is not a smaller eval set, it is an empty one".format(
                    restrict_to, split))
        restriction = {
            "applied": True,
            "path": str(restrict_to),
            "named": len(allowed),
            "in_split": len(pool),
            "why_this_is_not_the_split": (
                "the eval set no longer resembles the split it is drawn from, so "
                "any figure computed on it describes the restricted pool only; "
                "the restriction and its size must be quoted with the number"),
        }
    if count is not None and count > len(pool):
        raise SelectionError("asked for {} clips, the pool holds {}".format(
            count, len(pool)))
    chosen = pool if count is None else pool[:count]

    leaks: Dict[str, List[str]] = {}
    provenance: Dict[str, object] = {"supplied": False}
    if music_pairs is not None:
        pairs, pair_provenance = load_verified_music_pairs(music_pairs)
        leaks = leaked_clips(pairs, split_of)
        provenance = dict(pair_provenance, supplied=True)
        # Two different things can make this list flag nothing, and only one of
        # them is a defect.
        #
        # A pair list that does not *resolve* against this bundle -- ids from
        # another corpus generation, a wrong file -- flags nothing because it
        # describes nothing, and a "leak-free subset" defined by it is the whole
        # set wearing a label (CLAUDE.md 2).  That still refuses.
        #
        # A song-disjoint split flags nothing because no pair crosses the line,
        # which is the state ``assign_wild_song_split`` exists to produce and
        # separately refuses to publish without.  Measured on wild_v5_song,
        # 2026-08-25: 10,392 verified pairs, all resolving, none crossing.
        # Refusing that would make the tool unusable exactly when the corpus is
        # correct -- but it is still not free: the leak-free subset equals the
        # split, so anything computed "on the unflagged subset" is computed on
        # everything, and the report has to say so rather than let a reader
        # infer a filter that did nothing.
        resolved = sum(1 for pair in pairs
                       if all(end in split_of for end in pair))
        if not resolved:
            raise SelectionError(
                "{} resolves against no recording of this bundle ({} pair(s) read, "
                "0 with both ends in the manifest); it is not describing this "
                "corpus, so the leak-free subset it defines is the whole set "
                "wearing a label".format(music_pairs, len(pairs)))
        provenance = dict(provenance,
                          pairs_resolved_against_bundle=resolved,
                          pairs_crossing_the_split=sum(
                              1 for pair in pairs
                              if all(end in split_of for end in pair)
                              and len({split_of[end] for end in pair}) > 1))

    flagged = [clip for clip in chosen if clip in leaks]
    uploads = {clip.split(":")[1] for clip in chosen if clip.count(":") >= 2}
    return {
        "schema_version": "atomicdance-wild-m6-selection-v1",
        "bundle": str(bundle),
        "split": split,
        "salt": SALT,
        "order": "sha256(SALT || recording_id), ascending; a larger --count extends "
                 "this sample rather than replacing it",
        "available": len(candidates),
        "restricted_pool": restriction,
        "selected": len(chosen),
        # Cuts of one upload share a backing track by construction, so a
        # selection holding several cuts of one upload carries fewer distinct
        # tracks than it does clips.  Reported because R-precision's same-music
        # exclusion is built on exactly this key, and its pool shrinks with it.
        "distinct_uploads": len(uploads),
        "clips": chosen,
        "leak": {
            "pair_list": provenance,
            "flagged_in_selection": len(flagged),
            "flagged_fraction": round(len(flagged) / max(len(chosen), 1), 4),
            "flagged_in_split": sum(1 for clip in candidates if clip in leaks),
            # Stated, not inferred.  When this is true the "leak-free" figures
            # are the same numbers as the unfiltered ones, and a reader who
            # assumes a filter ran is reading a comparison that never happened.
            "leak_free_subset_equals_the_split": (
                provenance.get("supplied", False)
                and not any(clip in leaks for clip in candidates)),
            "clips": {clip: leaks[clip] for clip in flagged},
            # The whole split's flagged set, not just the selection's.  A
            # leak-free FID filters *both* distributions, and the ground-truth
            # side is the entire split -- filtering it by only the clips that
            # happened to be selected leaves the rest of the split's flagged
            # clips in the "clean" reference and the comparison is no longer
            # leak-free on either side.  Measured cost of getting this wrong on
            # wild_v4: 618 of 825 flagged ground-truth clips would have survived.
            "flagged_clips_in_split": sorted(clip for clip in candidates if clip in leaks),
            "reading": ("a flagged clip provably shares a backing track with a clip on "
                        "the other side of the split; an unflagged one has no *provable* "
                        "shared track, which is not the same as independent -- the "
                        "fingerprint's positive control recalls 0.564"),
            "what_it_is_for": ("FID and R can be recomputed on the unflagged subset "
                               "without generating anything a second time"),
        },
        "fid_note": ("FID is biased by sample size, so this count must be held fixed "
                     "across any two conditions compared within this corpus; wild "
                     "figures are never comparable to AIST ones at any count"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=pathlib.Path, required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--count", type=int, default=None,
                        help="how many clips to generate for; omit for the whole split")
    parser.add_argument("--music-pairs", type=pathlib.Path, default=None,
                        help="fingerprint pair list; without it no leak flags are "
                             "written and the report says so rather than reporting zero")
    parser.add_argument("--restrict-to", type=pathlib.Path, default=None,
                        help="file of recording ids the draw may use, one per line.  "
                             "Needed when part of the split is disqualified for a "
                             "reason the split itself does not know about -- e.g. a "
                             "borrowed completion checkpoint trained on 119 of this "
                             "split's 184 clips.  Split-level leak bookkeeping is "
                             "unaffected and the restriction is recorded in the report")
    parser.add_argument("--output", type=pathlib.Path, required=True,
                        help="writes <output>.txt (names) and <output>.json (report)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = select(args.bundle, args.split, args.count, args.music_pairs,
                        args.restrict_to)
    except SelectionError as error:
        print("selection refused: {}".format(error), file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".txt").write_text(
        "\n".join(report["clips"]) + "\n", encoding="utf-8")
    args.output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("{} of {} clip(s) in {} -> {}.txt".format(
        report["selected"], report["available"], report["split"], args.output))
    leak = report["leak"]
    if leak["pair_list"]["supplied"]:
        print("provable cross-split music: {} of the selection ({:.1%}), "
              "{} of the split".format(leak["flagged_in_selection"],
                                       leak["flagged_fraction"], leak["flagged_in_split"]))
    else:
        print("leak flags: NOT WRITTEN -- pass --music-pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
