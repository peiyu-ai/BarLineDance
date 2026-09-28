#!/usr/bin/env python3
"""Verify the released AtomicDance dataset before training or pseudo-labelling.

The official archive is a binary artifact and is ignored by git.  This checker
turns its implicit contract into a small JSON report: aligned frame counts,
expected 151-D/35-D dimensions, label range, duplicate names, and normalizer
metadata.  It intentionally does not normalize or rewrite any sample.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


SPLIT_ORDER = ("train", "val", "test")

# The upstream package's vocabulary: 100 prototypes plus the transition token.
# Only a fallback for roots that declare nothing; see declared_num_classes.
LEGACY_NUM_CLASSES = 101

_SLICE_NAME = re.compile(r"^(?P<source>.+)_slice(?P<index>\d+)$")


def source_id(name: str) -> str:
    """Return the source-video identifier behind an AtomicDance window name."""
    match = _SLICE_NAME.match(name)
    return match.group("source") if match else name


def _slice_reference(name: str) -> Optional[tuple[str, int]]:
    match = _SLICE_NAME.match(name)
    if match is None:
        return None
    return match.group("source"), int(match.group("index"))


# AIST++ encodes the backing track in the sequence name, e.g. the ``mBR1`` in
# ``gBR_sBM_cAll_d04_mBR1_ch03``.  Splitting by performance leaves songs free to
# appear on both sides, which silently invalidates any music-conditioned
# generalization claim: a model can recognise the track rather than respond to
# it.  Measured on aist_kinematic_release_v1, all 16 val songs are also train
# songs, and a planner's held-out atomic-class accuracy fell to the chance rate
# once evaluated on the song-disjoint test split instead.
_SONG_NAME = re.compile(r"_(?P<song>m[A-Za-z]{2}\d+)_")


def song_id(name: str) -> Optional[str]:
    """Return the AIST backing-track identifier, or None if absent.

    Returning None rather than guessing keeps non-AIST corpora honest: the
    audit reports how many names it could resolve, so an unrecognised naming
    scheme shows up as low coverage instead of a vacuous pass.
    """
    match = _SONG_NAME.search(name)
    return match.group("song") if match else None


def audit_song_disjointness(name_sets: Dict[str, Any]) -> Dict[str, Any]:
    """Cross-split backing-track overlap, per split pair."""
    songs = {}
    coverage = {}
    for split, names in name_sets.items():
        resolved = [song_id(name) for name in names]
        songs[split] = {value for value in resolved if value is not None}
        known = sum(1 for value in resolved if value is not None)
        coverage[split] = {
            "names": len(names),
            "with_song_id": known,
            "fraction": known / len(names) if names else 0.0,
            "unique_songs": len(songs[split]),
        }

    pairs = {}
    ordered = [split for split in SPLIT_ORDER if split in name_sets]
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            shared = sorted(songs[left] & songs[right])
            pairs["{}_{}".format(left, right)] = {
                "shared_songs": len(shared),
                "shared_fraction_of_{}".format(right): (
                    len(shared) / len(songs[right]) if songs[right] else 0.0
                ),
                "examples": shared[:20],
            }

    return {
        "checked": bool(name_sets),
        "coverage": coverage,
        "pairs": pairs,
    }


def _load_names(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
        raise ValueError("{} must contain a JSON list of strings".format(path))
    return values


def _load_retrieval_groups(path: Path, expected_length: int) -> List[str]:
    """Load explicit prototype-exclusion groups for an indexed split.

    A window name identifies one recording, but a dance performance can have
    several camera recordings.  Reconstructing a group from that name would
    therefore let the audit certify the exact cross-camera near-GT retrieval
    that the runtime rejects.  New materialized releases must publish this
    aligned sidecar; older layouts remain inspectable elsewhere but cannot
    pass a source-safe retrieval audit.
    """
    if not path.is_file():
        raise FileNotFoundError(
            "missing explicit retrieval_groups.json; cannot verify performance-safe retrieval: {}".format(
                path
            )
        )
    try:
        with path.open("r", encoding="utf-8") as handle:
            values = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot read {}: {}".format(path, error)) from error
    if (
        not isinstance(values, list)
        or len(values) != expected_length
        or any(not isinstance(value, str) or not value.strip() for value in values)
    ):
        raise ValueError(
            "{} must be a non-empty-string list aligned with {} samples".format(
                path, expected_length
            )
        )
    return values


def declared_num_classes(root: Path, override: Optional[int] = None) -> int:
    """The vocabulary this release says it holds.

    Read from ``build.json`` rather than pinned to a constant.  The constant
    used to be 100 -- the paper's prototype count -- so every release built on
    a re-clustered vocabulary failed the audit on its labels alone: the AIST++
    M3 bundle has 846 sub-prototypes and was reported as corrupt.  A bound that
    rejects a correct artifact is not a stricter check, it is a wrong one.

    Legacy roots have no build.json; they keep the old bound, because that is
    the vocabulary they were built with and nothing in them declares otherwise.
    """
    if override is not None:
        return override
    build_path = root / "build.json"
    if build_path.is_file():
        try:
            policy = json.loads(build_path.read_text(encoding="utf-8")).get("window_policy") or {}
        except (OSError, json.JSONDecodeError):
            policy = {}
        declared = policy.get("num_classes")
        if isinstance(declared, int) and declared > 0:
            return declared
    return LEGACY_NUM_CLASSES


def audit_split(root: Path, split: str, num_classes: int = LEGACY_NUM_CLASSES) -> Dict[str, Any]:
    split_root = root / split
    paths = {name: split_root / "{}.npy".format(name) for name in ("motion", "music", "labels")}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    names_path = split_root / "names.json"
    if not names_path.is_file():
        missing.append(str(names_path))
    if missing:
        raise FileNotFoundError("{} is missing {}".format(split_root, ", ".join(missing)))
    motion = np.load(str(paths["motion"]), mmap_mode="r")
    music = np.load(str(paths["music"]), mmap_mode="r")
    labels = np.load(str(paths["labels"]), mmap_mode="r")
    names = _load_names(names_path)
    errors: List[str] = []
    if motion.ndim != 3 or motion.shape[-1] != 151:
        errors.append("motion shape {} is not [N,T,151]".format(tuple(motion.shape)))
    if music.ndim != 3 or music.shape[-1] != 35:
        errors.append("music shape {} is not [N,T,35]".format(tuple(music.shape)))
    if labels.ndim != 2:
        errors.append("labels shape {} is not [N,T]".format(tuple(labels.shape)))
    if motion.ndim == music.ndim == 3 and labels.ndim == 2:
        if not (motion.shape[:2] == music.shape[:2] == labels.shape):
            errors.append(
                "N/T not aligned: motion={}, music={}, labels={}".format(
                    tuple(motion.shape), tuple(music.shape), tuple(labels.shape)
                )
            )
    if len(names) != len(motion):
        errors.append("names length {} != samples {}".format(len(names), len(motion)))
    if len(names) != len(set(names)):
        errors.append("duplicate names within {}".format(split))
    label_min = int(np.min(labels)) if labels.size else None
    label_max = int(np.max(labels)) if labels.size else None
    if label_min is not None and (label_min < 0 or label_max >= num_classes):
        errors.append("labels must be in [0,{}), got [{},{}]".format(
            num_classes, label_min, label_max))
    source_ids = [source_id(name) for name in names]
    source_counts: Dict[str, int] = {}
    for item in source_ids:
        source_counts[item] = source_counts.get(item, 0) + 1
    counts = sorted(source_counts.values())
    return {
        "split": split,
        "motion_shape": list(motion.shape),
        "music_shape": list(music.shape),
        "labels_shape": list(labels.shape),
        "motion_dtype": str(motion.dtype),
        "music_dtype": str(music.dtype),
        "labels_dtype": str(labels.dtype),
        "label_min": label_min,
        "label_max": label_max,
        "transition_frame_fraction": float(np.mean(labels == 0)) if labels.size else None,
        "samples": int(len(motion)),
        "names": int(len(names)),
        "unique_names": int(len(set(names))),
        "source_videos": int(len(source_counts)),
        "slices_per_source": {
            "min": int(counts[0]) if counts else 0,
            "median": float(np.median(counts)) if counts else 0.0,
            "max": int(counts[-1]) if counts else 0,
        },
        "valid": not errors,
        "errors": errors,
        "name_set": set(names),
        "source_set": set(source_ids),
    }


def audit_adjacent_window_labels(root: Path, split: str, stride: int) -> Dict[str, Any]:
    """Check whether labels agree in the shared frames of adjacent windows.

    AtomicDance release windows use names ending in ``_sliceN``.  A reliable
    full-source labeling pipeline must yield identical labels in identical
    overlapping frames before the source is cut into fixed-size training
    windows.  This check is deliberately label-only: it is cheap enough to run
    on every data version and catches clip-local label drift without rewriting
    the package.
    """
    if stride <= 0:
        raise ValueError("window stride must be positive")
    split_root = root / split
    labels = np.load(str(split_root / "labels.npy"), mmap_mode="r")
    names = _load_names(split_root / "names.json")
    if labels.ndim != 2 or len(names) != len(labels):
        return {
            "checked": False,
            "reason": "labels/names are not aligned; see split audit errors",
        }
    overlap = int(labels.shape[1]) - stride
    if overlap <= 0:
        return {
            "checked": False,
            "reason": "window length {} is not larger than stride {}".format(labels.shape[1], stride),
        }
    windows: Dict[tuple[str, int], int] = {}
    for index, name in enumerate(names):
        reference = _slice_reference(name)
        if reference is not None:
            windows[reference] = index
    pairs = 0
    equal = 0
    total = 0
    one_transition_other_atomic = 0
    both_atomic_disagree = 0
    examples: List[Dict[str, Any]] = []
    for (source, slice_index), left_index in sorted(windows.items()):
        right_index = windows.get((source, slice_index + 1))
        if right_index is None:
            continue
        left = np.asarray(labels[left_index, stride:])
        right = np.asarray(labels[right_index, :overlap])
        matches = left == right
        pair_equal = int(np.sum(matches))
        pair_total = int(matches.size)
        pairs += 1
        equal += pair_equal
        total += pair_total
        one_transition_other_atomic += int(np.sum((left == 0) != (right == 0)))
        both_atomic_disagree += int(np.sum((left != 0) & (right != 0) & ~matches))
        if len(examples) < 20 and pair_equal != pair_total:
            examples.append(
                {
                    "left": names[left_index],
                    "right": names[right_index],
                    "agreement": float(pair_equal / pair_total),
                }
            )
    if not pairs:
        return {
            "checked": False,
            "reason": "no adjacent _sliceN / _sliceN+1 pairs found",
            "stride": stride,
            "overlap_frames": overlap,
        }
    return {
        "checked": True,
        "stride": stride,
        "overlap_frames": overlap,
        "adjacent_pairs": pairs,
        "overlap_frames_compared": total,
        "label_agreement": float(equal / total),
        "one_transition_other_atomic_fraction": float(one_transition_other_atomic / total),
        "both_atomic_disagree_fraction": float(both_atomic_disagree / total),
        "disagreement_examples": examples,
    }


def _safe_frame_fraction(pairs: np.ndarray, counts: np.ndarray, groups: np.ndarray,
                         atomic_frames: int) -> float:
    """Share of atomic frames whose label also occurs in another retrieval group.

    ``pairs`` is one row per (window, label) with ``counts`` frames of that
    label, and ``groups[row]`` the window's retrieval group.  Written against
    those flat columns because the null below re-evaluates it a thousand times.
    """
    labels = pairs[:, 1]
    owners = groups[pairs[:, 0]]
    distinct = np.unique(np.stack([labels, owners], axis=1), axis=0)
    label_values, group_counts = np.unique(distinct[:, 0], return_counts=True)
    safe = label_values[group_counts > 1]
    return float(counts[np.isin(labels, safe)].sum() / atomic_frames)


def source_safe_retrieval_null(pairs: np.ndarray, counts: np.ndarray, groups: np.ndarray,
                               atomic_frames: int, *, permutations: int,
                               seed: int) -> Dict[str, Any]:
    """What this coverage would be if no label preferred a particular performance.

    The absolute bound alone cannot be read, so this reports the ceiling the
    corpus allows beside it.  The guess it was built to test was that coverage
    falls mechanically as a vocabulary grows finer -- that hundreds of
    sub-prototypes over a couple of hundred train performances must strand some
    class in a single one, making a fixed 0.99 a statement about the class count
    rather than about the data.

    **This null rejected that guess, and the rejection is the reason to keep
    it.**  Nothing forces stranding: a class may occur in every performance.  On
    aist_v1_norm (846 classes, 243 train performances) the ceiling is 0.999999
    against an observed 0.981138, so the 1.9% shortfall was 135 of 830 classes
    genuinely confined to one performance.  And a later release carried 599
    classes over 195 performances with *zero* stranded classes and coverage 1.0
    -- three times as many classes as performances, no mechanical floor at all.
    The bound was right and the vocabulary was what needed changing.

    The null permutes which performance each window belongs to, holding both
    the window contents and the performance sizes fixed.  It therefore keeps the
    class-size distribution exactly and destroys only the association between a
    class and a performance, which is the thing the gate is trying to see.  A
    ceiling near the observed value means the shortfall is the vocabulary's
    granularity; a ceiling far above it means classes really are memorising
    performances, and that is worth failing on.
    """
    if permutations <= 0:
        return {"checked": False, "reason": "permutations <= 0"}
    rng = np.random.default_rng(seed)
    shuffled = groups.copy()
    samples = np.empty(permutations, dtype=np.float64)
    for index in range(permutations):
        rng.shuffle(shuffled)
        samples[index] = _safe_frame_fraction(pairs, counts, shuffled, atomic_frames)
    return {
        "checked": True,
        "permutations": permutations,
        "seed": seed,
        "null_mean": float(samples.mean()),
        "null_sd": float(samples.std(ddof=1)) if permutations > 1 else 0.0,
        "null_min": float(samples.min()),
        "null_max": float(samples.max()),
        "permuted": "retrieval group of each window, holding window contents and group sizes fixed",
    }


def audit_source_safe_retrieval(root: Path, split: str = "train", *,
                                null_permutations: int = 0,
                                null_seed: int = 20260813) -> Dict[str, Any]:
    """Measure whether each atomic frame has a prototype in another group.

    Completion excludes every prototype from the query retrieval group, not
    merely its recording/window.  This audit is therefore a data-quality gate:
    it quantifies how much conditioning remains after performance-safe
    exclusion without weakening the leakage rule.

    With ``null_permutations`` it also reports the ceiling that this corpus's
    class-size distribution allows, so the bound can be read against something.
    See ``source_safe_retrieval_null``.
    """
    split_root = root / split
    labels = np.load(str(split_root / "labels.npy"), mmap_mode="r")
    names = _load_names(split_root / "names.json")
    if labels.ndim != 2 or len(labels) != len(names):
        return {
            "checked": False,
            "reason": "labels/names are not aligned; see split audit errors",
        }
    try:
        retrieval_groups = _load_retrieval_groups(
            split_root / "retrieval_groups.json", len(labels)
        )
    except (FileNotFoundError, ValueError) as error:
        return {"checked": False, "reason": str(error)}
    label_groups: Dict[int, set[str]] = {}
    for row, retrieval_group_id in zip(labels, retrieval_groups):
        for label in np.unique(row):
            label_value = int(label)
            if label_value > 0:
                label_groups.setdefault(label_value, set()).add(retrieval_group_id)
    group_counts = {str(label): len(groups) for label, groups in sorted(label_groups.items())}
    safe_labels = {label for label, groups in label_groups.items() if len(groups) > 1}
    atomic_frames = 0
    safe_frames = 0
    for row in labels:
        atomic = row[row > 0]
        atomic_frames += int(atomic.size)
        if atomic.size:
            safe_frames += int(np.isin(atomic, list(safe_labels)).sum())
    if not atomic_frames:
        return {
            "checked": False,
            "reason": "no non-transition atomic frames in split",
            "label_retrieval_group_counts": group_counts,
        }
    singleton_labels = [label for label, count in group_counts.items() if count == 1]
    result = {
        "checked": True,
        "atomic_frames": atomic_frames,
        "source_safe_atomic_frame_fraction": float(safe_frames / atomic_frames),
        "labels_with_any_retrieval_group": len(label_groups),
        "labels_with_only_one_retrieval_group": singleton_labels,
        "label_retrieval_group_counts": group_counts,
        "retrieval_groups": len(set(retrieval_groups)),
    }
    if null_permutations > 0:
        # One row per (window, label): the permutation re-labels which window
        # sits in which performance, so the per-window contents never move.
        rows, label_column, count_column = [], [], []
        for index, row in enumerate(labels):
            values, occurrences = np.unique(row[row > 0], return_counts=True)
            rows.extend([index] * len(values))
            label_column.extend(int(value) for value in values)
            count_column.extend(int(value) for value in occurrences)
        pairs = np.stack([np.asarray(rows, dtype=np.int64),
                          np.asarray(label_column, dtype=np.int64)], axis=1)
        counts = np.asarray(count_column, dtype=np.int64)
        codes = np.unique(np.asarray(retrieval_groups), return_inverse=True)[1].astype(np.int64)
        null = source_safe_retrieval_null(
            pairs, counts, codes, atomic_frames,
            permutations=null_permutations, seed=null_seed,
        )
        if null.get("checked") and null["null_mean"] > 0:
            null["observed_over_null_mean"] = (
                result["source_safe_atomic_frame_fraction"] / null["null_mean"]
            )
        result["null"] = null
    return result


def audit_dataset(
    root: Path,
    *,
    eval_source_list: Optional[Path] = None,
    window_stride: Optional[int] = 15,
    min_overlap_label_agreement: float = 0.99,
    min_safe_retrieval_fraction: float = 0.99,
    safe_retrieval_null_permutations: int = 0,
    require_song_disjoint_splits: bool = False,
    num_classes: Optional[int] = None,
) -> Dict[str, Any]:
    root = root.resolve()
    num_classes = declared_num_classes(root, num_classes)
    report: Dict[str, Any] = {
        "data_root": str(root),
        "num_classes": num_classes,
        "splits": {},
        "errors": [],
        "warnings": [],
    }
    name_sets = {}
    source_sets = {}
    # val must be audited too: it is the split used to select models, so a
    # defect there is at least as damaging as one in test.
    for split in SPLIT_ORDER:
        try:
            value = audit_split(root, split, num_classes)
            name_sets[split] = value.pop("name_set")
            source_sets[split] = value.pop("source_set")
            report["splits"][split] = value
            report["errors"].extend("{}: {}".format(split, error) for error in value["errors"])
        except FileNotFoundError as error:
            # The upstream package ships train/test only.  A genuinely absent
            # split is a fact about the release, not a defect; only train and
            # test are mandatory.
            report["splits"][split] = {"present": False, "reason": str(error)}
            if split != "val":
                report["errors"].append("{}: {}".format(split, error))
            else:
                report["warnings"].append(
                    "val split is absent; model selection has no held-out split"
                )
        except ValueError as error:
            report["splits"][split] = {"valid": False, "errors": [str(error)]}
            report["errors"].append("{}: {}".format(split, error))

    # Pairwise over whatever splits are present, so adding val cannot silently
    # disable the train/test checks the way a hardcoded pair would.  Order by
    # the pipeline's own split order rather than alphabetically, so the keys
    # read "train_val"/"train_test" instead of "test_train".
    present = [split for split in SPLIT_ORDER if split in name_sets]
    for left_index, left in enumerate(present):
        for right in present[left_index + 1 :]:
            pair = "{}_{}".format(left, right)
            overlap = sorted(name_sets[left] & name_sets[right])
            report.setdefault("cross_split_name_overlap_by_pair", {})[pair] = len(overlap)
            if overlap:
                report["errors"].append(
                    "{}/{} name overlap: {} examples".format(left, right, len(overlap))
                )
                report.setdefault("cross_split_name_overlap_examples_by_pair", {})[pair] = overlap[:20]

            source_overlap = sorted(source_sets[left] & source_sets[right])
            report.setdefault("cross_split_source_overlap_by_pair", {})[pair] = len(source_overlap)
            if source_overlap:
                report["errors"].append(
                    "{}/{} source-video overlap: {} sources".format(
                        left, right, len(source_overlap)
                    )
                )
                report.setdefault("cross_split_source_overlap_examples_by_pair", {})[pair] = (
                    source_overlap[:20]
                )

    # Preserve the historical top-level train/test keys; downstream reports and
    # tests read them by name.
    if {"train", "test"} <= set(name_sets):
        report["cross_split_name_overlap"] = report["cross_split_name_overlap_by_pair"]["train_test"]
        report["cross_split_source_overlap"] = report["cross_split_source_overlap_by_pair"][
            "train_test"
        ]

    song_audit = audit_song_disjointness(name_sets)
    report["song_disjointness_audit"] = song_audit
    for pair, value in song_audit.get("pairs", {}).items():
        if not value["shared_songs"]:
            continue
        message = "{} share {} backing track(s): {}".format(
            pair.replace("_", "/"), value["shared_songs"], ", ".join(value["examples"][:5])
        )
        # A song shared with test invalidates the benchmark outright.  A song
        # shared with val only invalidates music-conditioned *generalization*
        # claims, which is why it is a warning unless explicitly required.
        if pair.endswith("_test") or require_song_disjoint_splits:
            report["errors"].append(message)
        else:
            report["warnings"].append(
                message + " -- val cannot support music-conditioned generalization claims"
            )
    if window_stride is not None:
        if window_stride <= 0:
            raise ValueError("window_stride must be positive or None")
        report["adjacent_window_label_audit"] = {}
        for split in ("train", "test"):
            try:
                consistency = audit_adjacent_window_labels(root, split, window_stride)
                report["adjacent_window_label_audit"][split] = consistency
                if (
                    consistency.get("checked")
                    and consistency["label_agreement"] < min_overlap_label_agreement
                ):
                    report["errors"].append(
                        "{}: adjacent-window label agreement {:.6f} < required {:.6f}".format(
                            split,
                            consistency["label_agreement"],
                            min_overlap_label_agreement,
                        )
                    )
            except (FileNotFoundError, ValueError) as error:
                report["adjacent_window_label_audit"][split] = {
                    "checked": False,
                    "reason": str(error),
                }
                report["errors"].append("{}: overlap audit failed: {}".format(split, error))
    try:
        retrieval = audit_source_safe_retrieval(
            root, null_permutations=safe_retrieval_null_permutations
        )
        report["source_safe_retrieval_audit"] = retrieval
        if retrieval.get("checked"):
            if retrieval["source_safe_atomic_frame_fraction"] < min_safe_retrieval_fraction:
                report["errors"].append(
                    "train: source-safe retrieval coverage {:.6f} < required {:.6f}".format(
                        retrieval["source_safe_atomic_frame_fraction"],
                        min_safe_retrieval_fraction,
                    )
                )
        else:
            report["errors"].append(
                "train: source-safe retrieval audit is unverifiable: {}".format(
                    retrieval.get("reason", "unknown reason")
                )
            )
    except (FileNotFoundError, ValueError) as error:
        report["source_safe_retrieval_audit"] = {"checked": False, "reason": str(error)}
        report["errors"].append("train: source-safe retrieval audit failed: {}".format(error))
    if eval_source_list is not None:
        eval_source_list = eval_source_list.resolve()
        report["evaluation_source_list_path"] = str(eval_source_list)
        if not eval_source_list.is_file():
            report["errors"].append("missing evaluation source list: {}".format(eval_source_list))
        else:
            evaluation_sources = {
                source_id(line.strip())
                for line in eval_source_list.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            report["evaluation_sources"] = len(evaluation_sources)
            for split, sources in source_sets.items():
                overlap = sorted(sources & evaluation_sources)
                report["{}_evaluation_source_overlap".format(split)] = len(overlap)
                if overlap:
                    report["{}_evaluation_source_overlap_examples".format(split)] = overlap[:20]
                    if split == "train":
                        report["errors"].append(
                            "train/evaluation source-video overlap: {} sources".format(len(overlap))
                        )
    normalizer = root / "normalizer.pt"
    report["normalizer_path"] = str(normalizer)
    report["normalizer_present"] = normalizer.is_file()
    if not normalizer.is_file():
        report["errors"].append("missing normalizer.pt")
    report["valid"] = not report["errors"]
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/atomic_aistpp")
    parser.add_argument("--output", default=None, help="optional JSON report path")
    parser.add_argument(
        "--require-song-disjoint-splits",
        action="store_true",
        help="promote a shared backing track between train and val from a "
             "warning to an error; test is always required to be song-disjoint",
    )
    parser.add_argument(
        "--eval-source-list",
        default="data/splits/crossmodal_test.txt",
        help="source-level evaluation list that must be disjoint from training; pass an empty string to skip",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=15,
        help="frame stride used by _sliceN windows; pass 0 to skip overlap-label auditing",
    )
    parser.add_argument(
        "--min-overlap-label-agreement",
        type=float,
        default=0.99,
        help="hard lower bound for labels in shared adjacent-window frames",
    )
    parser.add_argument(
        "--min-safe-retrieval-fraction",
        type=float,
        default=0.99,
        help="hard lower bound for atomic frames with a prototype on another source",
    )
    parser.add_argument(
        "--safe-retrieval-null-permutations",
        type=int,
        default=0,
        help="permute which performance each train window belongs to, this many "
             "times, to report the ceiling the class-size distribution allows; "
             "0 skips it and the bound above is then reported without a scale",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="vocabulary bound for the label check; read from the release's "
             "build.json when omitted, and only falls back to the upstream 101 "
             "for legacy roots that declare nothing",
    )
    args = parser.parse_args(argv)
    report = audit_dataset(
        Path(args.data_root),
        eval_source_list=Path(args.eval_source_list) if args.eval_source_list else None,
        window_stride=args.window_stride if args.window_stride > 0 else None,
        min_overlap_label_agreement=args.min_overlap_label_agreement,
        min_safe_retrieval_fraction=args.min_safe_retrieval_fraction,
        safe_retrieval_null_permutations=args.safe_retrieval_null_permutations,
        require_song_disjoint_splits=args.require_song_disjoint_splits,
        num_classes=args.num_classes,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
