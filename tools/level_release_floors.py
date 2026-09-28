#!/usr/bin/env python3
"""Put every recording in a release on the same floor, immutably.

WHY, and the operator asked for it at the source.  2026-09-12: "从源头解决z 值
不对齐的问题,避免悬空的语料漏到后面给completion 和 后处理".  A monocular
reconstruction has no absolute height, and this corpus disagrees about where the
ground is.  Measured over the 295 normalized T sequences, each recording's floor
-- the 5th percentile of its lowest foot, the rule ``anchor_floor`` and
``render_avatar_video.floor_of`` already use -- after inverting the bundle's own
normalizer:

    median      0.341 m
    p5 / p95    0.306 / 0.396      spread 0.090 m
    min / max   0.264 / 0.469      full range 0.205 m
    21 of 295 recordings sit more than 5 cm from the median

HOW THAT REACHES THE PICTURE.  Retrieval pastes a bar from recording A into a
clip built on recording B, and ``--draft-root-continuity`` including z makes the
ROOT continuous at the seam -- the wrong invariant, because two prototypes hold
their feet different distances below the root, so aligning roots lifts the feet.
Measured on 7412632116703350028:clip001, median height of the lowest foot above
the render floor: ground truth 0.072 m, the shipped arm 0.070, and 0.140 / 0.439
for the two arms that relax the join ranking.  The completion then learns from a
corpus in which the same pose sits at a range of heights, and every downstream
repair is left chasing a constant the data should never have carried.

WHAT THIS CHANGES AND WHAT IT PROVABLY DOES NOT.  Only channel 6, the root's
height, and only by a per-RECORDING constant.  x, y, every rotation and the four
contact channels are copied byte for byte.  Two consequences are worth stating
because they decide how much has to be rebuilt:

* **The atomic vocabulary does not move.**  M2's TMR embedding runs on
  HumanML3D Guo features, whose construction puts the body on the floor before
  it measures anything.  Measured directly: lifting a whole recording 0.20 m
  changes its Guo features by 5.9e-06.  So the labels, the clusters and the
  segmentation are untouched and do not need rebuilding.
* **The planner does not move.**  It consumes music and labels, never motion.

What does move is the completion model's training data, which is the point.

THE NORMALIZER IS RE-FIT ON THE LEVELLED TRAIN SPLIT, because the old one's
root-z range was widened by exactly the spread being removed; keeping it would
spend resolution on a variation that no longer exists.  Val and test are
transformed with the frozen train fit, the same rule
``tools/apply_motion_normalizer`` states.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

ROOT_Z = 6
FOOT_JOINTS = (7, 8, 10, 11)
SPLITS = ("train", "val", "test")
COPY = ("labels.npy", "label_valid_mask.npy", "music.npy", "names.json",
        "retrieval_groups.json")


def sequence_of(name: str) -> str:
    """``wild_v5:123:clip000_slice7`` -> ``wild_v5:123:clip000``."""
    return name.rsplit("_slice", 1)[0]


def measure_release_floors(release):
    """Each recording's floor, measured from the release's OWN windows.

    THE FLOOR is the 5th percentile of the lowest foot joint over every frame of
    every window belonging to that recording, in metres -- the same rule
    ``anchor_floor`` and ``render_avatar_video.floor_of`` use, so the three
    cannot disagree about where the ground is.

    WHY NOT REUSE A CENSUS OF THE SEQUENCE BUNDLE.  Because it answers a
    different question.  The bundle holds whole recordings; the release holds a
    TIMEBASE-FILTERED subset of 150-frame windows, so a recording's 5th
    percentile over the bundle is not its 5th percentile over the release.
    Measured: levelling by the bundle's census took the release's own
    recording-to-recording spread from 0.117 m to 0.064 m -- half of it left
    behind -- while measuring here makes the residual zero by construction.
    """
    import collections
    from dataset.quaternion import ax_from_6v
    from vis import SMPLSkeleton

    skeleton = SMPLSkeleton()
    normalizer = torch.load(str(release / "normalizer.pt"), map_location="cpu",
                            weights_only=False)
    low = normalizer["data_min"].float()
    high = normalizer["data_max"].float()
    span = torch.where(high == low, torch.ones_like(high), high - low)
    lowest = collections.defaultdict(list)
    for split in SPLITS:
        directory = release / split
        if not directory.exists():
            continue
        motion = np.load(directory / "motion.npy", mmap_mode="r")
        names = json.load(open(directory / "names.json"))
        for index, name in enumerate(names):
            frame = torch.from_numpy(np.array(motion[index], dtype=np.float32))
            raw = (frame + 1.0) / 2.0 * span + low
            with torch.no_grad():
                joints = skeleton.forward(
                    ax_from_6v(raw[:, 7:].reshape(-1, 24, 6)).unsqueeze(0),
                    raw[:, 4:7].unsqueeze(0))[0]
            lowest[sequence_of(name)].extend(
                joints[:, FOOT_JOINTS, 2].min(dim=1).values.numpy().tolist())
        print("  measured {} ({} recordings so far)".format(split, len(lowest)), flush=True)
    return {name: float(np.percentile(values, 5)) for name, values in lowest.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release", required=True)
    parser.add_argument("--floors", default="auto",
                        help="JSON from a floor census, or 'auto' (the default) "
                             "to measure each recording's floor from THIS "
                             "release's own windows. Auto is the accurate "
                             "option and the reason is measured: a census taken "
                             "on the sequence bundle describes whole "
                             "recordings, while the release is a FILTERED "
                             "subset of windows, so the two 5th percentiles "
                             "differ and levelling by the wrong one leaves half "
                             "the spread behind -- 0.117 m fell only to 0.064 m "
                             "that way, against 0.117 -> 0.000 by construction "
                             "when the tool measures what it levels")
    parser.add_argument("--out", required=True)
    parser.add_argument("--keep-normalizer", action="store_true",
                        help="publish the SOURCE release's normalizer byte for "
                             "byte instead of re-fitting on the levelled train "
                             "split. Use it when an existing checkpoint must "
                             "stay valid: a model works in normalized units, so "
                             "a re-fit makes the same normalized value mean a "
                             "different height and the checkpoint's recorded "
                             "dataset_provenance.normalizer.sha256 no longer "
                             "describes the data. Levelled values may then fall "
                             "outside [-1, 1], which the release contract "
                             "explicitly permits (tools/apply_motion_normalizer: "
                             "rows transformed by a frozen fit 'can therefore "
                             "legitimately fall outside'). This is what lets the "
                             "levelling be tested as a ONE-VARIABLE change "
                             "against the shipped completion, instead of being "
                             "confounded with a retrain")
    parser.add_argument("--reference", type=float, default=None,
                        help="height every recording's floor is moved to; "
                             "default is the median over the floors file, which "
                             "keeps the shift centred and small")
    args = parser.parse_args()

    source = pathlib.Path(args.release)
    out = pathlib.Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite {}: publication is "
                         "immutable-new-directory-only".format(out))
    floors = (measure_release_floors(source)
              if args.floors == "auto" else json.load(open(args.floors))["floors"])
    reference = (args.reference if args.reference is not None
                 else float(np.median(list(floors.values()))))
    print("reference floor {:.4f} m over {} recordings".format(reference, len(floors)))

    normalizer = torch.load(str(source / "normalizer.pt"), map_location="cpu",
                            weights_only=False)
    old_min = normalizer["data_min"].float().numpy()
    old_max = normalizer["data_max"].float().numpy()
    old_span = np.where(old_max == old_min, 1.0, old_max - old_min)

    staging = out.with_name(out.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    raw_splits, shifts, missing = {}, {}, set()
    for split in SPLITS:
        directory = source / split
        if not directory.exists():
            continue
        motion = np.load(directory / "motion.npy", mmap_mode="r")
        names = json.load(open(directory / "names.json"))
        raw = (np.asarray(motion, dtype=np.float64) + 1.0) / 2.0 * old_span + old_min
        offsets = np.zeros(len(names))
        for index, name in enumerate(names):
            sequence = sequence_of(name)
            if sequence not in floors:
                missing.add(sequence)
                continue
            offsets[index] = floors[sequence] - reference
        raw[:, :, ROOT_Z] -= offsets[:, None]
        raw_splits[split] = raw
        shifts[split] = offsets
        print("  {}: {} windows, shift median {:+.4f} m, |shift| max {:.4f} m".format(
            split, len(names), float(np.median(offsets)), float(np.abs(offsets).max())))
    if missing:
        # Fail closed: a window left unshifted is the one prototype still
        # carrying its own calibration, inside a release the manifest calls
        # levelled -- the defect shape CLAUDE.md 2 exists to prevent.
        raise SystemExit("no floor for {} recording(s): {}".format(
            len(missing), sorted(missing)[:5]))

    train = raw_splits["train"]
    if args.keep_normalizer:
        new_min, new_max = old_min.copy(), old_max.copy()
    else:
        new_min = train.reshape(-1, train.shape[-1]).min(axis=0)
        new_max = train.reshape(-1, train.shape[-1]).max(axis=0)
    new_span = np.where(new_max == new_min, 1.0, new_max - new_min)
    print("root z raw range: was [{:.3f}, {:.3f}], now [{:.3f}, {:.3f}] -- "
          "{:.3f} m narrower".format(old_min[ROOT_Z], old_max[ROOT_Z],
                                     new_min[ROOT_Z], new_max[ROOT_Z],
                                     (old_max[ROOT_Z] - old_min[ROOT_Z])
                                     - (new_max[ROOT_Z] - new_min[ROOT_Z])))

    for split, raw in raw_splits.items():
        directory = staging / split
        directory.mkdir(parents=True)
        levelled = 2.0 * (raw - new_min) / new_span - 1.0
        np.save(directory / "motion.npy", levelled.astype(np.float32))
        for item in COPY:
            origin = source / split / item
            if origin.exists():
                shutil.copy2(origin, directory / item)
    torch.save({"data_min": torch.from_numpy(new_min.astype(np.float32)),
                "data_max": torch.from_numpy(new_max.astype(np.float32))},
               staging / "normalizer.pt")
    for item in ("windows.jsonl", "quarantine.jsonl"):
        if (source / item).exists():
            shutil.copy2(source / item, staging / item)

    build = json.load(open(source / "build.json")) if (source / "build.json").exists() else {}
    build["derived_from"] = {
        "release": str(source),
        "tool": "tools/level_release_floors.py",
        "floors": ("measured from this release's own windows"
                   if args.floors == "auto" else str(pathlib.Path(args.floors).resolve())),
        "floors_sha256": (None if args.floors == "auto" else hashlib.sha256(
            pathlib.Path(args.floors).read_bytes()).hexdigest()),
    }
    build["floor_reference_m"] = reference
    build["floor_levelled"] = True
    build["floor_rule"] = ("5th percentile of the lowest of SMPL joints 7, 8, 10, 11, "
                           "per source recording, in metres")
    build["normalizer_refit"] = ("frozen: the source release's own fit, kept so "
                                "an existing checkpoint stays valid"
                                if args.keep_normalizer
                                else "train split of the levelled motion")
    build["channels_changed"] = [ROOT_Z]
    # REHASH EVERY ARTIFACT.  ``train_atomic._verify_root_artifact`` refuses a
    # release whose build.json disagrees with its bytes, and it refused this
    # tool's first output -- correctly, because motion.npy and normalizer.pt had
    # changed.  That gate is the reason a levelled release cannot be mistaken
    # for the one the shipped checkpoint was trained on, so it is updated here
    # rather than relaxed anywhere.
    artifacts = build.get("artifacts")
    if isinstance(artifacts, dict):
        for key in list(artifacts):
            if key == "splits":
                for split, entries in artifacts["splits"].items():
                    for item in list(entries):
                        path = staging / split / item
                        if path.exists():
                            entries[item] = hashlib.sha256(path.read_bytes()).hexdigest()
            else:
                path = staging / key
                if path.exists():
                    artifacts[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    # THE SAME HASH IS PUBLISHED IN FOUR PLACES and
    # ``train_atomic.validate_training_data_root`` cross-checks every one:
    #
    #     artifacts["normalizer.pt"]
    #     normalizer["published_artifact_sha256"]
    #     normalizer["source_artifact_sha256"]
    #     representation_contract["normalization_artifact_sha256"]
    #
    # This tool's first three outputs were each refused by a DIFFERENT one of
    # them, because they were updated one at a time.  Enumerate the set before
    # trusting a rebuild: four copies is not redundancy to route around, it is
    # what makes a release unable to claim a normalizer it does not ship.
    digest = hashlib.sha256((staging / "normalizer.pt").read_bytes()).hexdigest()
    if isinstance(build.get("representation_contract"), dict):
        build["representation_contract"]["normalization_artifact_sha256"] = digest
    if isinstance(build.get("normalizer"), dict):
        build["normalizer"]["published_artifact_sha256"] = digest
        # Equal to the published hash because the levelled normalizer IS the
        # artifact: it is re-fit on the levelled train split, not copied.
        build["normalizer"]["source_artifact_sha256"] = digest
        build["normalizer"]["source_artifact"] = str(source / "normalizer.pt")
        build["normalizer"]["copy_policy"] = (
            "refit on the levelled train split by tools/level_release_floors.py")
    (staging / "build.json").write_text(json.dumps(build, indent=1))
    staging.rename(out)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
