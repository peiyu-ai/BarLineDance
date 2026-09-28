#!/usr/bin/env python3
"""Republish a release with the windows of manifest-excluded clips removed.

WHY.  ``runs/timebase_census_v1.jsonl`` (2026-08-31) judged 2,032 of 15,201
converted clips (13.4%) to have a 3D reconstruction running at a different time
base than its own footage -- the 2026-08-25 defect family, whose signature is
that every frame-count check passes.  Those clips contribute ~13% of the
windows in every split, and on the training side they are precisely "motion
that does not line up with its music", which is the relationship the completion
stage is being asked to learn.

WHAT IT CONSUMES.  ``runs/timebase_exclude_v1.jsonl`` -- the manifest, never the
census re-derived (CLAUDE.md: downstream reads manifests).  Each row carries the
sequence key ``wild_v5:<id>:clipNNN``; a window belongs to a clip when its name
is ``<sequence>_sliceK``.

WHAT IT CHANGES.  Row-subsets of the per-split arrays (motion/music/labels/
label_valid_mask + names + retrieval_groups) and a filtered ``windows.jsonl``.
The normalizer is HARDLINKED unchanged: it was fit on the unfiltered train
split, and refitting it here would silently change what every trained model's
normalized units mean.  That choice is recorded in ``derived_from`` so the
asymmetry is visible rather than discovered.

VERIFICATION, not trust: after writing, a random sample of kept rows is
compared byte-for-byte against the source rows they came from, and the flagged
sequence set is re-scanned against the surviving names (must be empty).
"""
import argparse
import hashlib
import json
import os
import pathlib
import shutil

import numpy as np


def sha256_file(path, block=1 << 22):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_of(window_name):
    return window_name.rsplit("_slice", 1)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exclude", required=True,
                        help="jsonl manifest with a 'sequence' field per excluded clip")
    parser.add_argument("--check-rows", type=int, default=2048)
    args = parser.parse_args()

    source = pathlib.Path(args.source)
    output = pathlib.Path(args.output)
    if output.exists():
        raise SystemExit("{} exists; releases are immutable".format(output))
    staging = output.with_name(output.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    excluded = {json.loads(line)["sequence"] for line in open(args.exclude)}
    print("{} excluded sequences from {}".format(len(excluded), args.exclude))

    build = json.loads((source / "build.json").read_text())
    kept_by_split = {}
    for split in ("train", "val", "test"):
        source_split = source / split
        if not source_split.is_dir():
            continue
        names = json.load(open(source_split / "names.json"))
        keep = np.array([sequence_of(name) not in excluded for name in names])
        kept = np.flatnonzero(keep)
        kept_by_split[split] = kept
        target_split = staging / split
        target_split.mkdir()

        for array_name in ("motion.npy", "music.npy", "labels.npy", "label_valid_mask.npy"):
            path = source_split / array_name
            if not path.is_file():
                continue
            array = np.load(path, mmap_mode="r")
            out = np.lib.format.open_memmap(target_split / array_name, mode="w+",
                                            dtype=array.dtype,
                                            shape=(len(kept),) + array.shape[1:])
            for position in range(0, len(kept), 2048):
                index = kept[position:position + 2048]
                out[position:position + len(index)] = array[index]
            out.flush()
            del out

        json.dump([names[i] for i in kept], open(target_split / "names.json", "w"))
        groups_path = source_split / "retrieval_groups.json"
        if groups_path.is_file():
            groups = json.load(open(groups_path))
            if isinstance(groups, list) and len(groups) == len(names):
                json.dump([groups[i] for i in kept], open(target_split / "retrieval_groups.json", "w"))
            else:
                shutil.copyfile(groups_path, target_split / "retrieval_groups.json")

        # Byte-level verification on a sample of kept rows.
        rng = np.random.default_rng(0)
        sample = rng.choice(len(kept), size=min(args.check_rows, len(kept)), replace=False)
        source_motion = np.load(source_split / "motion.npy", mmap_mode="r")
        target_motion = np.load(target_split / "motion.npy", mmap_mode="r")
        for position in np.sort(sample):
            if not np.array_equal(np.asarray(target_motion[position]),
                                  np.asarray(source_motion[kept[position]])):
                raise SystemExit("row copy mismatch at {} {}".format(split, position))
        survivors = {sequence_of(n) for n in json.load(open(target_split / "names.json"))}
        if survivors & excluded:
            raise SystemExit("excluded sequence survived in {}".format(split))
        print("  {}: {} -> {} windows ({} removed), {} rows byte-verified".format(
            split, len(names), len(kept), len(names) - len(kept), len(sample)))

    os.link(source / "normalizer.pt", staging / "normalizer.pt")
    kept_indices = {split: set(v.tolist()) for split, v in kept_by_split.items()}
    remap = {split: {old: new for new, old in enumerate(sorted(kept))}
             for split, kept in kept_indices.items()}
    with open(staging / "windows.jsonl", "w") as out_manifest:
        for line in open(source / "windows.jsonl"):
            row = json.loads(line)
            split = row.get("split")
            index = row.get("array_index")
            if split in kept_indices and index in kept_indices[split]:
                row["array_index"] = remap[split][index]
                out_manifest.write(json.dumps(row) + "\n")
    for name in ("quarantine.jsonl",):
        if (source / name).is_file():
            os.link(source / name, staging / name)

    for split in kept_by_split:
        for artifact in list(build["artifacts"]["splits"].get(split, {})):
            path = staging / split / artifact
            if path.is_file():
                build["artifacts"]["splits"][split][artifact] = sha256_file(path)
    build["artifacts"]["windows.jsonl"] = sha256_file(staging / "windows.jsonl") \
        if "windows.jsonl" in build["artifacts"] else build["artifacts"].get("windows.jsonl")
    # The contract validator cross-checks counts against names.json -- rightly:
    # a count that survives a filter unchanged is exactly the "manifest says one
    # thing, bytes say another" defect this pipeline keeps catching.
    counts = build.get("counts", {})
    windows_block = counts.get("materialized_windows", {})
    for split, kept in kept_by_split.items():
        if split in windows_block:
            windows_block[split] = int(len(kept))
    build.setdefault("derived_from", {})
    build["derived_from"] = {
        "release": str(source.resolve()),
        "tool": "tools/filter_release_windows.py",
        "exclusion_manifest": str(pathlib.Path(args.exclude).resolve()),
        "exclusion_manifest_sha256": sha256_file(args.exclude),
        "excluded_sequences": len(excluded),
        "normalizer": "hardlinked from the source release UNFILTERED -- refitting "
                      "would change what a normalized unit means for every model; "
                      "the filtered corpus is therefore normalized by statistics "
                      "that include the excluded clips",
    }
    (staging / "build.json").write_text(json.dumps(build, indent=2), encoding="utf-8")
    staging.rename(output)
    print(output)


if __name__ == "__main__":
    main()
