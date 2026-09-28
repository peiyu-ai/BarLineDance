#!/usr/bin/env python3
"""Re-split AIST++ by backing track, and publish what that cannot fix.

``tools/audit_atomic_dataset.py`` raises an *error*, not a warning, when a song
appears on both sides of the test boundary::

    train/test share 10 backing track(s): mBR0, mHO5, mJB5, mJS3, mKR2
    val/test  share  2 backing track(s): mKR2, mWA0

and ``docs/FINETUNE_PLAN.md`` makes ``--require-song-disjoint-splits``
a precondition for training on any new vocabulary.  The frozen performance-group
split cannot satisfy either: measured on ``aist_full_performance_v1``, train
holds all 60 songs, so val's 23 songs and test's 10 songs are *subsets* of
train's.  A music-conditioned planner evaluated there can recognise the track
instead of responding to it, and no downstream metric can tell the difference.

This tool assigns whole **songs** to splits, stratified by genre, so that the
song sets are disjoint by construction rather than by luck.

What it deliberately does not claim
-----------------------------------
A song-disjoint split of AIST++ **cannot** be choreography-disjoint, and saying
so is the point of this docstring.  The corpus is built the other way round:

* each genre has **10 basic (``sBM``) choreographies, each danced to all 6 of
  that genre's songs** -- 1,166 of 1,363 recordings;
* the advanced (``sFM``) choreographies are song-specific -- the other 197.

So holding out song ``mBR0`` puts ``gBR_sBM_*_ch01..ch10`` into test while the
same ten choreographies stay in train under ``mBR1..mBR5``.  The escape hatch --
an ``sFM``-only test, which *is* both song- and choreography-disjoint -- leaves
about 33 recordings at one song per genre, which is the "test split is only 128
windows and distribution-shifted" defect the plan already recorded.

Neither split is free of leakage; they leak different things.  This tool
therefore **measures the choreography overlap it creates and writes it into the
report**, so the number is cited alongside the split rather than discovered
later.  A song-disjoint split buys a music-conditioning claim and pays for it in
motion novelty; that trade is a fact about AIST++, not a defect of this run.

Assignment is deterministic and not tuned: songs are ordered within their genre
by ``sha256(SALT || song_id)``, the first ``--test-songs-per-genre`` go to test,
the next ``--val-songs-per-genre`` to val, the rest to train.  ``SALT`` is a
frozen constant, so the split is reproducible from the song names alone and no
choice in it was made after looking at a score.

Four invariants are checked before anything is written, and each one can fail:

* every recording parses into (genre, song) -- an unparsed row is an error, not
  a ``"?"`` bucket, because a silent catch-all is how the genre pre-split
  quietly disappeared once already;
* every genre is present in all three splits;
* the song sets are pairwise disjoint;
* no ``retrieval_group_id`` and no ``duplicate_content_group_id`` spans splits.

Usage::

    python3 tools/assign_song_disjoint_split.py \\
        --bundle data/atomic_aistpp/aist_full_performance_v1 \\
        --output-bundle data/atomic_aistpp/aist_full_performance_songsplit_v1 \\
        --report runs/aist_songsplit_assignment.json
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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SPLITS: Tuple[str, ...] = ("train", "val", "test")

# Frozen once, on 2026-08-13, so the ordering is reproducible from song names
# alone.  Changing it re-draws the benchmark; that is a new split, not a re-run.
SALT = "atomicdance-song-disjoint-v1"

SPLIT_STATUS = "song_disjoint_genre_stratified"

# ``gBR_sBM_cAll_d04_mBR0_ch01``: genre, situation, camera, dancer, song, choreo.
AIST_NAME = re.compile(
    r"^g(?P<genre>[A-Z]{2})_s(?P<situation>[A-Z]{2})_c(?P<camera>[A-Za-z0-9]+)"
    r"_d(?P<dancer>\d+)_m(?P<song>[A-Za-z]{2}\d+)_ch(?P<choreography>\d+)$"
)


class SplitError(RuntimeError):
    """A split that would be wrong is refused rather than published."""


def recording_name(row: Mapping[str, Any]) -> str:
    """The AIST sequence name behind a manifest row."""
    legacy = row.get("legacy_source_name")
    if isinstance(legacy, str) and legacy:
        return legacy
    recording = row.get("recording_id")
    if not isinstance(recording, str) or not recording:
        raise SplitError("row has neither legacy_source_name nor recording_id: {!r}".format(row))
    return recording.rsplit("/", 1)[-1]


def parse_name(name: str) -> Dict[str, str]:
    """Split an AIST name into its six fields, or refuse it.

    Returning a ``"?"`` bucket here would let an unrecognised naming scheme
    collapse every recording into one genre, and the split would still look
    balanced.  The repo has that failure on record from the M3 genre pre-split,
    so this raises instead.
    """
    match = AIST_NAME.match(name)
    if match is None:
        raise SplitError("cannot parse AIST fields from {!r}".format(name))
    fields = match.groupdict()
    fields["song"] = "m" + fields["song"]
    return fields


def song_order_key(song: str) -> str:
    return hashlib.sha256((SALT + "\x00" + song).encode("utf-8")).hexdigest()


def assign_songs(
    songs_by_genre: Mapping[str, Sequence[str]],
    *,
    test_per_genre: int,
    val_per_genre: int,
) -> Dict[str, str]:
    """Whole songs to splits, ``test`` first so the benchmark is never starved."""
    assignment: Dict[str, str] = {}
    for genre in sorted(songs_by_genre):
        songs = sorted(set(songs_by_genre[genre]), key=song_order_key)
        if len(songs) < test_per_genre + val_per_genre + 1:
            raise SplitError(
                "genre {} has {} song(s); {} are needed to fill test, val and train".format(
                    genre, len(songs), test_per_genre + val_per_genre + 1
                )
            )
        for index, song in enumerate(songs):
            if index < test_per_genre:
                assignment[song] = "test"
            elif index < test_per_genre + val_per_genre:
                assignment[song] = "val"
            else:
                assignment[song] = "train"
    return assignment


def _group_spans(rows: Iterable[Mapping[str, Any]], key: str, split_of: Mapping[str, str]) -> Dict[str, List[str]]:
    spans: Dict[str, set] = collections.defaultdict(set)
    for row in rows:
        group = row.get(key)
        if group is None:
            continue
        spans[str(group)].add(split_of[recording_name(row)])
    return {group: sorted(values) for group, values in spans.items() if len(values) > 1}


def choreography_overlap(fields_by_name: Mapping[str, Mapping[str, str]],
                         split_of: Mapping[str, str]) -> Dict[str, Any]:
    """How much of the held-out motion is danced in train to a different song.

    A choreography is ``(genre, situation, choreography)``: the routine itself,
    independent of who performs it or which camera saw it.  Reported per
    situation because that is where the mechanism lives -- ``sBM`` routines are
    danced to every song in their genre, ``sFM`` routines to exactly one.
    """
    by_split: Dict[str, set] = {split: set() for split in SPLITS}
    by_split_situation: Dict[Tuple[str, str], set] = collections.defaultdict(set)
    recordings: Dict[Tuple[str, str], int] = collections.Counter()
    for name, fields in fields_by_name.items():
        split = split_of[name]
        key = (fields["genre"], fields["situation"], fields["choreography"])
        by_split[split].add(key)
        by_split_situation[(split, fields["situation"])].add(key)
        recordings[(split, fields["situation"])] += 1

    report: Dict[str, Any] = {"choreography_key": "(genre, situation, choreography)"}
    for split in ("val", "test"):
        shared = by_split[split] & by_split["train"]
        report[split] = {
            "choreographies": len(by_split[split]),
            "also_in_train": len(shared),
            "fraction_also_in_train": (len(shared) / len(by_split[split])) if by_split[split] else 0.0,
            "by_situation": {
                situation: {
                    "choreographies": len(by_split_situation[(split, situation)]),
                    "also_in_train": len(
                        by_split_situation[(split, situation)]
                        & by_split_situation[("train", situation)]
                    ),
                    "recordings": recordings[(split, situation)],
                }
                for situation in sorted(
                    {sit for (sp, sit) in by_split_situation if sp == split}
                )
            },
        }
    return report


def summarise(fields_by_name: Mapping[str, Mapping[str, str]],
              split_of: Mapping[str, str]) -> Dict[str, Any]:
    songs: Dict[str, set] = {split: set() for split in SPLITS}
    genres: Dict[str, set] = {split: set() for split in SPLITS}
    counts: Dict[str, int] = {split: 0 for split in SPLITS}
    for name, fields in fields_by_name.items():
        split = split_of[name]
        songs[split].add(fields["song"])
        genres[split].add(fields["genre"])
        counts[split] += 1
    summary: Dict[str, Any] = {
        "splits": {
            split: {
                "recordings": counts[split],
                "songs": len(songs[split]),
                "song_ids": sorted(songs[split]),
                "genres": sorted(genres[split]),
            }
            for split in SPLITS
        },
        "shared_songs": {},
    }
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            summary["shared_songs"]["{}_{}".format(left, right)] = sorted(
                songs[left] & songs[right]
            )
    return summary


def rewrite_rows(rows: Sequence[Mapping[str, Any]], split_of: Mapping[str, str],
                 note: str) -> List[Dict[str, Any]]:
    rewritten = []
    for row in rows:
        updated = dict(row)
        updated["split"] = split_of[recording_name(row)]
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


def build(bundle: pathlib.Path, output_bundle: pathlib.Path, *,
          test_per_genre: int, val_per_genre: int) -> Dict[str, Any]:
    sources = read_jsonl(bundle / "sources.jsonl")
    sequences = read_jsonl(bundle / "sequences.jsonl")

    fields_by_name: Dict[str, Dict[str, str]] = {}
    for row in sources:
        name = recording_name(row)
        fields_by_name[name] = parse_name(name)

    songs_by_genre: Dict[str, List[str]] = collections.defaultdict(list)
    for fields in fields_by_name.values():
        songs_by_genre[fields["genre"]].append(fields["song"])

    song_split = assign_songs(
        songs_by_genre, test_per_genre=test_per_genre, val_per_genre=val_per_genre
    )
    split_of = {name: song_split[fields["song"]] for name, fields in fields_by_name.items()}

    # Every sequence must belong to a source we just assigned; a sequence whose
    # recording is absent would otherwise silently keep its old split.
    unknown = sorted({recording_name(row) for row in sequences} - set(split_of))
    if unknown:
        raise SplitError(
            "{} sequence recording(s) are absent from sources.jsonl, e.g. {}".format(
                len(unknown), ", ".join(unknown[:5])
            )
        )

    summary = summarise(fields_by_name, split_of)
    for pair, shared in summary["shared_songs"].items():
        if shared:
            raise SplitError("{} still share song(s): {}".format(pair, ", ".join(shared)))
    all_genres = set().union(*(set(value["genres"]) for value in summary["splits"].values()))
    missing = {
        split: sorted(all_genres - set(value["genres"]))
        for split, value in summary["splits"].items()
    }
    for split, absent in missing.items():
        if absent:
            raise SplitError("split {} is missing genre(s): {}".format(split, ", ".join(absent)))

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

    overlap = choreography_overlap(fields_by_name, split_of)
    note = (
        "whole backing tracks assigned by genre-stratified sha256({}); song sets are "
        "disjoint by construction, and {:.0%} of test choreographies are still danced "
        "in train to a different song because AIST's basic routines span every song "
        "of their genre".format(SALT, overlap["test"]["fraction_also_in_train"])
    )

    # Published by atomic rename, which is the convention every other bundle in
    # this repo records as ``immutable_new_directory_only_atomic_rename``.
    # Writing in place would create the directory first and fill it after, so an
    # interrupted run leaves a half-written bundle that the driver's
    # ``[ ! -e "$BUNDLE" ]`` guard then skips -- a resume that reads absence of
    # completion as completion.
    if output_bundle.exists():
        raise FileExistsError(output_bundle)
    staging = output_bundle.with_name(output_bundle.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=False)
    write_jsonl(staging / "sources.jsonl", rewrite_rows(sources, split_of, note))
    write_jsonl(staging / "sequences.jsonl", rewrite_rows(sequences, split_of, note))

    # The arrays stay where they are: a copy would duplicate 5.6 GB for two
    # rewritten manifests.  Recorded in the report because a tree whose payload
    # lives behind a symlink is censused as 0 bytes -- see CLAUDE.md 1.3.
    os.symlink(os.path.relpath(bundle / "sequences", output_bundle), staging / "sequences")
    staging.rename(output_bundle)

    report = {
        "schema_version": "atomicdance-song-disjoint-split-v1",
        "input_bundle": str(bundle.resolve()),
        "output_bundle": str(output_bundle.resolve()),
        "payload_is_a_symlink_to": str((bundle / "sequences").resolve()),
        "policy": {
            "salt": SALT,
            "split_status": SPLIT_STATUS,
            "test_songs_per_genre": test_per_genre,
            "val_songs_per_genre": val_per_genre,
            "unit": "backing track (whole song), stratified by dance genre",
        },
        "song_assignment": {song: split for song, split in sorted(song_split.items())},
        "summary": summary,
        "choreography_overlap": overlap,
        "split_note": note,
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=pathlib.Path, required=True,
                        help="frozen performance bundle holding sources.jsonl and sequences.jsonl")
    parser.add_argument("--output-bundle", type=pathlib.Path, required=True,
                        help="new bundle directory; it must not already exist")
    parser.add_argument("--test-songs-per-genre", type=int, default=1)
    parser.add_argument("--val-songs-per-genre", type=int, default=1)
    parser.add_argument("--report", type=pathlib.Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build(
            args.bundle,
            args.output_bundle,
            test_per_genre=args.test_songs_per_genre,
            val_per_genre=args.val_songs_per_genre,
        )
    except (SplitError, FileExistsError) as error:
        print("error: {}".format(error), file=sys.stderr)
        return 1
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
    print(json.dumps(
        {
            "summary": report["summary"]["splits"],
            "shared_songs": report["summary"]["shared_songs"],
            "choreography_overlap": report["choreography_overlap"],
        },
        indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
