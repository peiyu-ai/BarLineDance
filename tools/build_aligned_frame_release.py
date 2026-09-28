#!/usr/bin/env python3
"""A FRAME-level release whose labels are the music-aligned CCA classes.

WHY THIS STEP WAS MISSING.  ``release_bar_aligned_k8`` holds a label space that
is predictable from music -- clustering the same bars in the music-aligned CCA
subspace reads 0.2765 against a 0.2346 majority floor at K=4 and 0.1790 against
0.1383 at K=8, while five random projections of the SAME motion features read
-0.027 and +0.001 -- against the shipped 21-class TMR vocabulary's 0.1136
against a 0.1580 floor, i.e. worse than always guessing the most common class.
Its motion side includes ``slowest sub-bar phase`` and ``fastest sub-bar
phase``, so a class in it says something about WHEN inside the bar, which is the
thing [[timing-is-song-independent]] says nothing in the pipeline decides.

A planner was trained on it (``planner_aligned_k8``) and **no dance was ever
generated**, because inference needs a FRAME-level release: ``--data-root``
supplies the retrieval library, which is indexed by label, and the planner now
emits aligned classes.  That is what this builds.

NO REFITTING, SO NO NEW LEAKAGE.  The per-bar classes are read from the bar
release's own ``labels.npy`` and painted onto the frame release's timeline
through ``bars.jsonl``'s ``bar_frame_spans``, which are in sequence
coordinates.  The CCA, the scaler and the k-means were fit on TRAIN bars only
when that release was built; nothing here re-estimates them.

NOTE the trap: ``bars.jsonl`` still carries a ``bar_labels`` field, and it is
the OLD 21-class one -- the aligned build copied that manifest byte for byte and
changed only ``labels.npy``.  Reading the jsonl field would silently produce the
vocabulary this release exists to replace.

Only ``labels.npy`` and ``label_valid_mask.npy`` change; motion, music, names
and the window geometry are copied, so an arm from this release and an arm from
the source differ in exactly one array.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import sys

import numpy as np

SPLITS = ("train", "val", "test")
COPY = ("motion.npy", "music.npy", "names.json", "retrieval_groups.json")


def bar_label_map(bar_release: pathlib.Path):
    """``{(sequence_id, start, end): class}`` from the bar release's own arrays."""
    mapping = {}
    conflicts = 0
    for split in SPLITS:
        labels_path = bar_release / split / "labels.npy"
        if not labels_path.exists():
            continue
        labels = np.load(labels_path)
        rows = [json.loads(line) for line in open(bar_release / "bars.jsonl")
                if json.loads(line)["split"] == split]
        rows.sort(key=lambda r: r["array_index"])
        for row in rows:
            index = int(row["array_index"])
            if index >= len(labels):
                continue
            for bar, span in enumerate(row["bar_frame_spans"]):
                if bar >= labels.shape[1]:
                    break
                key = (row["sequence_id"], int(span[0]), int(span[1]))
                value = int(labels[index, bar])
                if key in mapping and mapping[key] != value:
                    conflicts += 1
                mapping[key] = value
    return mapping, conflicts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release", required=True, help="frame-level source release")
    parser.add_argument("--bar-release", required=True,
                        help="bar release holding the aligned labels.npy")
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-classes", type=int, default=8)
    args = parser.parse_args()

    source = pathlib.Path(args.release)
    bars = pathlib.Path(args.bar_release)
    out = pathlib.Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite {}: immutable-new-directory-only".format(out))

    mapping, conflicts = bar_label_map(bars)
    print("bar classes read: {:,} spans, {} disagreements between overlapping "
          "bar windows".format(len(mapping), conflicts))
    if conflicts:
        raise SystemExit("overlapping bar windows disagree about a bar's class; "
                         "the transfer would be ambiguous")

    per_sequence = {}
    for (sequence, start, end), value in mapping.items():
        per_sequence.setdefault(sequence, []).append((start, end, value))

    staging = out.with_name(out.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    windows = [json.loads(line) for line in open(source / "windows.jsonl")]
    by_split = {}
    for row in windows:
        by_split.setdefault(row["split"], {})[int(row["array_index"])] = row

    covered_total = frames_total = 0
    for split in SPLITS:
        directory = source / split
        if not directory.exists():
            continue
        labels = np.load(directory / "labels.npy")
        new = np.zeros_like(labels)
        rows = by_split.get(split, {})
        for index in range(len(labels)):
            row = rows.get(index)
            if row is None:
                continue
            offset = int(row["start_frame"])
            for start, end, value in per_sequence.get(row["sequence_id"], ()):
                lo = max(0, start - offset)
                hi = min(labels.shape[1], end - offset)
                if hi > lo:
                    new[index, lo:hi] = value
        covered_total += int((new > 0).sum())
        frames_total += int(new.size)
        target = staging / split
        target.mkdir(parents=True)
        np.save(target / "labels.npy", new)
        # A frame with no bar over it is class 0, the transition class, and its
        # mask must say so or the materializer's "all valid" contract refuses it.
        np.save(target / "label_valid_mask.npy",
                np.ones_like(new, dtype=np.load(directory / "label_valid_mask.npy").dtype))
        for item in COPY:
            if (directory / item).exists():
                shutil.copy2(directory / item, target / item)
        print("  {}: {} windows, {:.1%} of frames carry a bar class".format(
            split, len(labels), float((new > 0).mean())))
    print("overall labelled frame share: {:.1%}".format(covered_total / max(frames_total, 1)))

    shutil.copy2(source / "normalizer.pt", staging / "normalizer.pt")
    for item in ("windows.jsonl", "quarantine.jsonl"):
        if (source / item).exists():
            shutil.copy2(source / item, staging / item)

    build = json.load(open(source / "build.json"))
    build["derived_from"] = {
        "release": str(source),
        "bar_release": str(bars),
        "tool": "tools/build_aligned_frame_release.py",
    }
    build["label_space_id"] = "music_aligned_cca_k{}_v1".format(args.num_classes)
    build["num_classes"] = args.num_classes
    build["labels"] = ("music-aligned CCA k-means classes, transferred per bar "
                       "from the bar release's labels.npy; no refitting here")
    artifacts = build.get("artifacts")
    if isinstance(artifacts, dict) and isinstance(artifacts.get("splits"), dict):
        for split, entries in artifacts["splits"].items():
            for item in list(entries):
                path = staging / split / item
                if path.exists():
                    entries[item] = hashlib.sha256(path.read_bytes()).hexdigest()
    (staging / "build.json").write_text(json.dumps(build, indent=1))
    staging.rename(out)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
