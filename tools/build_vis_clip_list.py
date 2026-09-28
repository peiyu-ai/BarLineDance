#!/usr/bin/env python3
"""The clip list a review render is allowed to use -- filtered, and pinned.

WHY THIS EXISTS.  ``runs/timebase_exclude_v1.jsonl`` (2026-08-31) names 2,032 of
15,201 clips whose 3D reconstruction runs at a different time base than its own
footage -- the 2026-08-25 defect family, whose whole signature is that every
frame-count check passes.  That manifest was applied to the *training* release
(``tools/filter_release_windows.py`` -> release_v3_timebase) and to nothing else.
The review render kept sampling from the unfiltered corpus, so the reviewer went
on being shown clips whose ground-truth panel cannot line up with its own footage
panel, and read that as a model defect.

That is not hypothetical.  On 2026-08-31 the reviewer picked
``7615198982276904931__clip000`` out of a ten-clip strip by eye -- "this one's GT
and raw video are at different frame rates" -- and it is in the manifest at
scale 0.5 with ``source_fps`` 60.0: a 60 fps upload reconstructed at 30.  The eye
and the census agree, and the render had no gate between them.  One clip in ten,
which is what a 13.4% corpus rate looks like when nothing filters it.

WHAT IT PRODUCES.  A newline-delimited ``wild_v5:<id>:<clipNNN>`` list, which is
what ``tools/render_sample_strip.py`` and the inference driver already consume.

  * every clip is present in the named split of the named release, so the list
    cannot outlive a rebuild silently;
  * every clip named by the exclusion manifest is dropped, by name -- CLAUDE.md
    §1.1: downstream consumes manifests, never a directory or a prefix listing;
  * ``--pin`` clips are placed first and are subject to the SAME filter.  A pin
    that is excluded is refused loudly rather than dropped quietly, because a
    pin is a reviewer asking for a specific clip and silently substituting
    another one is the failure this file exists to stop;
  * the remainder is filled deterministically from a seeded shuffle, so a
    re-render of "the same ten clips" really is the same ten.

WHAT IT DOES NOT CLAIM.  Passing this filter means "this clip's reconstruction
shares a time base with its footage".  It does not mean the clip is good, in
frame, or solo.  ``--rank-by-cleanliness`` orders candidates by
``runs/wild_clip_population.json`` (subject area, no rival dancer, steady crop)
when a human is going to be judging dance quality by eye and a two-dancer clip
would confound that -- it is an ordering, not a gate.

Usage::

    python3 tools/build_vis_clip_list.py \\
        --release /dev/shm/atomicdance-song-v5rekey/release_v3_timebase \\
        --split val --count 10 \\
        --pin 7646787035172290481__clip000 --pin 7653840177207863665__clip000 \\
        --output runs/vis_clips_val10.txt
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys


def sequence_key(clip: str) -> str:
    """``<id>__clipNNN`` -> the release's own ``wild_v5:<id>:clipNNN``."""
    if clip.startswith("wild_v5:"):
        return clip
    video, _, part = clip.partition("__")
    if not part:
        raise SystemExit("cannot parse clip id {!r}".format(clip))
    return "wild_v5:{}:{}".format(video, part)


def flat_id(sequence: str) -> str:
    parts = sequence.split(":")
    return parts[1] + "__" + parts[2] if len(parts) == 3 else sequence


def split_sequences(release: pathlib.Path, split: str) -> set:
    names = json.loads((release / split / "names.json").read_text())
    return {name.rsplit("_slice", 1)[0] for name in names}


def excluded_sequences(manifest: pathlib.Path) -> dict:
    rows = {}
    for line in manifest.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["sequence"]] = row
    return rows


def cleanliness(population: pathlib.Path) -> dict:
    """Per-clip visual-confounder readings, for ORDERING only (never a gate)."""
    if not population.is_file():
        return {}
    payload = json.loads(population.read_text())
    scored = {}
    for row in payload.get("rows", []):
        scored[sequence_key(row["clip"])] = (
            -float(row.get("subject_area_median", 0.0)),
            float(row.get("rival_ratio", 1.0)),
            float(row.get("crop_travel_p95", 1.0)),
        )
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--exclude", default="runs/timebase_exclude_v1.jsonl")
    parser.add_argument("--population", default="runs/wild_clip_population.json")
    parser.add_argument("--pin", action="append", default=[],
                        help="clip id to place first; repeatable; refused if excluded")
    parser.add_argument("--rank-by-cleanliness", action="store_true",
                        help="order the filler by subject area / no rival / steady crop")
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    release = pathlib.Path(args.release)
    present = split_sequences(release, args.split)
    excluded = excluded_sequences(pathlib.Path(args.exclude))

    pinned = []
    for clip in args.pin:
        key = sequence_key(clip)
        if key in excluded:
            row = excluded[key]
            raise SystemExit(
                "pinned clip {} is time-base defective (scale {}, source_fps {}); "
                "it is in {} and must not be rendered as ground truth".format(
                    clip, row.get("scale"), row.get("source_fps"), args.exclude))
        if key not in present:
            raise SystemExit("pinned clip {} is not in {}/{}".format(clip, release, args.split))
        if key not in pinned:
            pinned.append(key)

    candidates = sorted(present - set(pinned) - set(excluded))
    if args.rank_by_cleanliness:
        scores = cleanliness(pathlib.Path(args.population))
        # Unscored clips sort last rather than first: an absent reading is not
        # evidence of a clean clip.
        candidates.sort(key=lambda s: scores.get(s, (0.0, 1.0, 1.0)))
    else:
        random.Random(args.seed).shuffle(candidates)

    chosen = pinned + candidates[:max(0, args.count - len(pinned))]
    if len(chosen) < args.count:
        raise SystemExit("only {} clips survive the filter, {} requested".format(
            len(chosen), args.count))

    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(chosen) + "\n", encoding="utf-8")

    dropped = len(present & set(excluded))
    print("{}/{} {} sequences are time-base defective and were dropped ({:.1f}%)".format(
        dropped, len(present), args.split, 100.0 * dropped / max(len(present), 1)),
        file=sys.stderr)
    print("{} pinned, {} filled, -> {}".format(len(pinned), len(chosen) - len(pinned), output),
          file=sys.stderr)
    for sequence in chosen:
        print("  {}{}".format(flat_id(sequence), "  [pinned]" if sequence in pinned else ""),
              file=sys.stderr)


if __name__ == "__main__":
    main()
