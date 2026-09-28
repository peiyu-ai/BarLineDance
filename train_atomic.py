"""Train and debug the atomic planner and motion completion stages."""

import argparse
import hashlib
import contextlib
import json
import random
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch
import torch.multiprocessing
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from dataset.atomic import (AtomicMotionLibrary, blank_seam_evidence,
                            plan_boundaries)
from dataset.atomic_dataset import AtomicSequenceDataset, collate_atomic_sequences
from model.atomic_completion import AtomicCompletionDecoder, AtomicCompletionDiffusion
from model.atomic_planner import AtomicPlannerTransformer, UniformD3PM


# A materialized wild-data release is deliberately more constrained than the
# historical ``data/atomic_aistpp`` download.  Keep this gate here, at the
# training boundary, so a valid preprocessing run cannot accidentally become
# an unpinned training run just because a caller points ``--data-root`` at it.
MATERIALIZED_BUILD_SCHEMA = "atomic-window-materialization-v1"
MODEL_MOTION_REPRESENTATION = "AtomicDance_151D"
MODEL_COORDINATE_SYSTEM = "z_up_world_body_only"
MODEL_MOTION_DIM = 151
MODEL_MUSIC_DIM = 35
_REQUIRED_SPLITS = ("train", "val", "test")
_REQUIRED_SPLIT_ARTIFACTS = (
    "motion.npy",
    "music.npy",
    "labels.npy",
    "label_valid_mask.npy",
    "names.json",
    "retrieval_groups.json",
)


class DatasetReleaseContractError(ValueError):
    """A materialized data-root is incomplete, tampered, or unsafe to train on."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _require_mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DatasetReleaseContractError("{} must be a JSON object".format(context))
    return value


def _require_sha256(value: object, context: str) -> str:
    if not _is_sha256(value):
        raise DatasetReleaseContractError("{} must be a SHA-256 hex digest".format(context))
    return str(value).lower()


def _root_artifact_path(root: Path, relative_path: str, context: str) -> Path:
    """Resolve a declared bundle asset without allowing an escape from its root."""
    if not isinstance(relative_path, str) or not relative_path:
        raise DatasetReleaseContractError("{} has an invalid relative artifact path".format(context))
    candidate = root / relative_path
    try:
        resolved = candidate.resolve()
        resolved.relative_to(root)
    except ValueError as error:
        raise DatasetReleaseContractError("{} escapes data root".format(context)) from error
    if not resolved.is_file():
        raise DatasetReleaseContractError("{} is missing: {}".format(context, resolved))
    return resolved


def _verify_root_artifact(root: Path, relative_path: str, expected_hash: object, context: str) -> str:
    expected = _require_sha256(expected_hash, "{} hash".format(context))
    actual = _sha256_file(_root_artifact_path(root, relative_path, context))
    if actual != expected:
        raise DatasetReleaseContractError(
            "{} SHA-256 mismatch: expected {}, got {}".format(context, expected, actual)
        )
    return actual


def _validate_normalizer_payload(path: Path) -> None:
    """Check the copied inference normalizer after its byte hash is pinned."""
    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError as error:  # pragma: no cover - old Torch cannot safely load release artifacts
        raise DatasetReleaseContractError(
            "validated materialized releases require torch.load(weights_only=True) support"
        ) from error
    except Exception as error:
        raise DatasetReleaseContractError("cannot safely load normalizer {}: {}".format(path, error)) from error
    if not isinstance(payload, Mapping) or set(payload) != {"data_min", "data_max"}:
        raise DatasetReleaseContractError("normalizer.pt must contain exactly data_min and data_max")
    arrays = {}
    for key in ("data_min", "data_max"):
        tensor = payload[key]
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32 or tuple(tensor.shape) != (MODEL_MOTION_DIM,):
            raise DatasetReleaseContractError(
                "normalizer.pt {} must be a torch.float32 [{}] tensor".format(key, MODEL_MOTION_DIM)
            )
        values = np.asarray(tensor.detach().cpu().numpy(), dtype=np.float32)
        if not np.isfinite(values).all():
            raise DatasetReleaseContractError("normalizer.pt {} contains non-finite values".format(key))
        arrays[key] = values
    if np.any(arrays["data_max"] < arrays["data_min"]):
        raise DatasetReleaseContractError("normalizer.pt data_max is below data_min")


def _provenance_entry(manifests: Mapping[str, Any], name: str) -> Dict[str, str]:
    entry = _require_mapping(manifests.get(name), "input_manifests.{}".format(name))
    path = entry.get("path")
    if not isinstance(path, str) or not path:
        raise DatasetReleaseContractError("input_manifests.{} path must be non-empty".format(name))
    return {"path": path, "sha256": _require_sha256(entry.get("sha256"), "input_manifests.{}".format(name))}


def _legacy_dataset_provenance(root: Path) -> Dict[str, Any]:
    """Compatibility metadata for the released upstream layout without a build contract."""
    normalizer = root / "normalizer.pt"
    return {
        "data_root": str(root),
        "release_contract": "absent_legacy_layout",
        "release_contract_validated": False,
        # This flag scopes eligibility to the data release.  It never changes
        # an evaluator's separate protocol/headline decision.
        "headline_eligible": False,
        "headline_eligibility_scope": "dataset_release_only",
        "validation_split": "test",
        "validation_protocol": "LEGACY_TEST_FALLBACK_CODE_SMOKE_ONLY",
        "build": {"path": None, "sha256": None},
        "normalizer": {
            "path": str(normalizer.resolve()) if normalizer.is_file() else None,
            "sha256": _sha256_file(normalizer) if normalizer.is_file() else None,
            "verified_against_build": False,
        },
        "source_manifest": {"path": None, "sha256": None},
        "sequence_manifest": {"path": None, "sha256": None},
        "label_manifest": {"path": None, "sha256": None},
        "reason": "build.json is absent; allowed only for compatibility/code smoke, never a headline data release",
    }


PLANNER_TOKEN_RESOLUTIONS = ("frame", "bar")

# A bar release's adjacent TOKENS differ this often at least.  The two
# distributions this separates are measured, not guessed
# (docs/DANCE_QUALITY_DEFECTS.md 27.4 and the bar release's own report):
#
#     release_v3 (frame tokens)      labels.npy (5329, 150)     1.24% of
#                                    adjacent token pairs differ
#     release_bar_v1 (bar tokens)    labels.npy (810, 4)       ~77% differ
#                                    (ground-truth bar corpus: 78.4%)
#
# 0.10 sits 8x above the frame reading and 7.7x below the bar reading, so the
# gate fires on the mistake it exists for -- pointing --planner-token-resolution
# bar at a frame release -- and cannot fire on the release it is meant to pass.
BAR_TOKEN_MIN_CHANGE_RATE = 0.10


def declare_planner_token_resolution(args, train_dataset, sample_limit=None):
    """What ONE TOKEN of this run is, checked against the data, then stamped.

    WHY A TRAINING FLAG AT ALL.  ``infer_atomic.planner_token_resolution`` reads
    ``planner_token_resolution`` off the checkpoint's own saved ``args`` and
    refuses ``--plan-bar-tokens`` unless it says ``"bar"``.  Nothing else could
    tell the two apart: ``MusicNormalization`` carries its z-scoring statistics
    as buffers whose shape is ``(music_dim,)`` whether they were fit on frames
    or on bar-pooled music, so a frame checkpoint fed pooled music loads
    cleanly and plans nonsense.  The declaration has to be made where the data
    is, which is here.

    WHY IT IS CHECKED AND NOT JUST RECORDED.  A declaration nobody verifies is
    the "gate that cannot fail" CLAUDE.md section 2 forbids.  Three things are
    checked against the dataset the run will actually train on:

    * the rows are ``args.seq_len`` tokens long -- in bar mode ``seq_len`` is
      what ``infer_atomic`` uses as the planner's window size, and there it
      counts BARS;
    * adjacent tokens change label at least ``BAR_TOKEN_MIN_CHANGE_RATE`` of
      the time, which a frame release cannot do (1.24%);
    * the music width matches ``args.music_dim``, so a ``mean_std`` pooling
      declared over a ``mean``-pooled release is refused rather than trained.

    Returns the evidence dict, which is also stamped onto ``args`` so it travels
    inside every checkpoint.  In frame mode it returns ``None`` and only stamps
    the declaration, so every existing run reproduces byte-identically.
    """
    resolution = str(getattr(args, "planner_token_resolution", "frame") or "frame")
    if resolution not in PLANNER_TOKEN_RESOLUTIONS:
        raise ValueError("--planner-token-resolution must be one of {}, got {!r}".format(
            list(PLANNER_TOKEN_RESOLUTIONS), resolution))
    args.planner_token_resolution = resolution
    if resolution == "frame":
        return None
    from dataset.bar_tokens import BAR_POOLINGS, DEFAULT_BAR_POOLING

    if args.stage != "planner":
        raise ValueError(
            "--planner-token-resolution bar describes the PLANNER's token axis; "
            "stage {!r} was asked for.  A bar release stores a placeholder motion "
            "array, so a completion run against it would train on it silently."
            .format(args.stage))
    pooling = str(getattr(args, "planner_bar_pooling", None) or DEFAULT_BAR_POOLING)
    if pooling not in BAR_POOLINGS:
        raise ValueError("--planner-bar-pooling must be one of {}, got {!r}".format(
            list(BAR_POOLINGS), pooling))
    args.planner_bar_pooling = pooling
    args.planner_bar_beats = int(getattr(args, "planner_bar_beats", 4) or 4)
    if getattr(args, "music_phase_features", False):
        raise ValueError(
            "--planner-token-resolution bar cannot be combined with "
            "--planner-music-phase-features: MusicPhaseFeatures reads the beat "
            "one-hot on channel 34 with `> 0.5`, and channel 34 of POOLED music "
            "is a beat density around 0.067 that never crosses it.")
    total = len(train_dataset)
    if total < 1:
        raise ValueError("the training split is empty; there is nothing to declare")
    limit = total if sample_limit is None else min(total, int(sample_limit))
    pairs = changes = 0
    lengths = set()
    music_widths = set()
    for index in range(limit):
        item = train_dataset[index]
        labels = [int(x) for x in item["labels"].reshape(-1).tolist()]
        lengths.add(len(labels))
        music_widths.add(int(item["music"].shape[-1]))
        for previous, current in zip(labels, labels[1:]):
            pairs += 1
            changes += int(previous != current)
    if lengths != {int(args.seq_len)}:
        raise ValueError(
            "--planner-token-resolution bar declares that one token is one bar and "
            "that --seq-len {} counts BARS, but the training rows carry {} tokens. "
            "infer_atomic uses the checkpoint's seq_len as the planner's window "
            "size, so a mismatch here plans over a window the model never saw."
            .format(int(args.seq_len), sorted(lengths)))
    if music_widths != {int(args.music_dim)}:
        raise ValueError(
            "--planner-token-resolution bar with --planner-bar-pooling {!r} declares "
            "{}-D music, but the release's rows are {}-D."
            .format(pooling, int(args.music_dim), sorted(music_widths)))
    if not pairs:
        raise ValueError(
            "the training rows carry no adjacent token pair, so nothing in this "
            "release encodes a bar transition")
    rate = changes / pairs
    if rate < BAR_TOKEN_MIN_CHANGE_RATE:
        raise ValueError(
            "--planner-token-resolution bar was asked for, but adjacent tokens in "
            "this release change label on only {:.2%} of {} pairs, under the {:.0%} "
            "floor.  A per-FRAME release reads 1.24% here; a bar release reads about "
            "77%.  This root is not tokenised by bars."
            .format(rate, pairs, BAR_TOKEN_MIN_CHANGE_RATE))
    evidence = {
        "planner_token_resolution": "bar",
        "planner_bar_pooling": pooling,
        "planner_bar_beats": int(args.planner_bar_beats),
        "tokens_per_row": int(next(iter(lengths))),
        "rows_inspected": int(limit),
        "adjacent_token_pairs": int(pairs),
        "adjacent_token_changes": int(changes),
        "adjacent_token_change_rate": float(rate),
        "min_change_rate": BAR_TOKEN_MIN_CHANGE_RATE,
    }
    args.planner_token_resolution_evidence = evidence
    return evidence


def validate_training_data_root(data_root: str) -> Dict[str, Any]:
    """Validate a materialized release before creating any dataset loader.

    Historical AtomicDance downloads have no immutable build manifest.  They
    remain loadable for implementation smoke tests, but the returned metadata
    records their unverified provenance and forces the old test-only fallback.
    A root with ``build.json`` must satisfy the complete current materialized
    release contract; a partial or malformed build never falls back to legacy.
    """
    root = Path(data_root).expanduser().resolve()
    build_path = root / "build.json"
    if not build_path.exists():
        return _legacy_dataset_provenance(root)
    if not build_path.is_file():
        raise DatasetReleaseContractError("build.json is not a regular file: {}".format(build_path))
    try:
        build = json.loads(build_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetReleaseContractError("cannot parse build.json {}: {}".format(build_path, error)) from error
    build = _require_mapping(build, "build.json")
    if build.get("schema_version") != MATERIALIZED_BUILD_SCHEMA:
        raise DatasetReleaseContractError(
            "unsupported build.json schema {!r}; expected {!r}".format(
                build.get("schema_version"), MATERIALIZED_BUILD_SCHEMA
            )
        )

    contract = _require_mapping(build.get("representation_contract"), "representation_contract")
    expected_contract = {
        "motion_representation_id": MODEL_MOTION_REPRESENTATION,
        "coordinate_system": MODEL_COORDINATE_SYSTEM,
        "normalization_state": "normalized",
        "normalization_fit_split": "train",
        "camera_in_model_input": False,
    }
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            raise DatasetReleaseContractError(
                "representation_contract.{} must be {!r}, got {!r}".format(key, expected, contract.get(key))
            )
    contract_normalizer_hash = _require_sha256(
        contract.get("normalization_artifact_sha256"), "representation_contract.normalization_artifact_sha256"
    )

    policy = _require_mapping(build.get("window_policy"), "window_policy")
    if policy.get("motion_dim") != MODEL_MOTION_DIM or policy.get("music_dim") != MODEL_MUSIC_DIM:
        raise DatasetReleaseContractError(
            "window_policy must declare {}-D motion and {}-D music".format(
                MODEL_MOTION_DIM, MODEL_MUSIC_DIM
            )
        )
    if policy.get("fps") != 30:
        raise DatasetReleaseContractError(
            "window_policy.fps must be exactly 30 for AtomicDance 150-frame timing"
        )

    artifacts = _require_mapping(build.get("artifacts"), "artifacts")
    expected_artifact_entries = {"normalizer.pt", "splits", "windows.jsonl", "quarantine.jsonl"}
    if set(artifacts) != expected_artifact_entries:
        raise DatasetReleaseContractError(
            "artifacts must contain exactly {}".format(", ".join(sorted(expected_artifact_entries)))
        )
    normalizer_hash = _verify_root_artifact(
        root, "normalizer.pt", artifacts.get("normalizer.pt"), "artifacts.normalizer.pt"
    )
    if normalizer_hash != contract_normalizer_hash:
        raise DatasetReleaseContractError(
            "representation normalizer hash does not match published root normalizer.pt"
        )
    normalizer_path = _root_artifact_path(root, "normalizer.pt", "artifacts.normalizer.pt")
    _validate_normalizer_payload(normalizer_path)

    normalizer = _require_mapping(build.get("normalizer"), "normalizer")
    if normalizer.get("published_artifact") != "normalizer.pt":
        raise DatasetReleaseContractError("normalizer.published_artifact must be normalizer.pt")
    if _require_sha256(normalizer.get("published_artifact_sha256"), "normalizer.published_artifact_sha256") != normalizer_hash:
        raise DatasetReleaseContractError("normalizer.published_artifact_sha256 does not match root normalizer.pt")
    if _require_sha256(normalizer.get("source_artifact_sha256"), "normalizer.source_artifact_sha256") != normalizer_hash:
        raise DatasetReleaseContractError("normalizer.source_artifact_sha256 does not match root normalizer.pt")
    if normalizer.get("fit_split") != "train":
        raise DatasetReleaseContractError("normalizer.fit_split must be 'train'")
    fit_report = normalizer.get("fit_report")
    if not isinstance(fit_report, str) or not fit_report:
        raise DatasetReleaseContractError("normalizer.fit_report must be non-empty provenance")
    _require_sha256(normalizer.get("fit_report_sha256"), "normalizer.fit_report_sha256")

    manifests = _require_mapping(build.get("input_manifests"), "input_manifests")
    source_manifest = _provenance_entry(manifests, "sources.jsonl")
    sequence_manifest = _provenance_entry(manifests, "sequences.jsonl")
    label_manifest = _provenance_entry(manifests, "labels.jsonl")
    if _require_sha256(normalizer.get("fit_source_manifest_sha256"), "normalizer.fit_source_manifest_sha256") != source_manifest["sha256"]:
        raise DatasetReleaseContractError("normalizer fit source-manifest hash does not match build input provenance")

    for artifact_name in ("windows.jsonl", "quarantine.jsonl"):
        _verify_root_artifact(root, artifact_name, artifacts.get(artifact_name), "artifacts.{}".format(artifact_name))
    split_artifacts = _require_mapping(artifacts.get("splits"), "artifacts.splits")
    if set(split_artifacts) != set(_REQUIRED_SPLITS):
        raise DatasetReleaseContractError(
            "artifacts.splits must contain exactly {}".format(", ".join(_REQUIRED_SPLITS))
        )
    retrieval_groups_by_split = {}
    for split in _REQUIRED_SPLITS:
        hashes = _require_mapping(split_artifacts.get(split), "artifacts.splits.{}".format(split))
        missing = [name for name in _REQUIRED_SPLIT_ARTIFACTS if name not in hashes]
        if missing:
            raise DatasetReleaseContractError(
                "artifacts.splits.{} is missing {}".format(split, ", ".join(missing))
            )
        for artifact_name, expected_hash in hashes.items():
            if not isinstance(artifact_name, str) or artifact_name not in _REQUIRED_SPLIT_ARTIFACTS:
                raise DatasetReleaseContractError(
                    "artifacts.splits.{} has unsupported artifact {!r}".format(split, artifact_name)
                )
            _verify_root_artifact(
                root,
                "{}/{}".format(split, artifact_name),
                expected_hash,
                "artifacts.splits.{}.{}".format(split, artifact_name),
            )
        groups_path = root / split / "retrieval_groups.json"
        try:
            groups = json.loads(groups_path.read_text(encoding="utf-8"))
            names = json.loads((root / split / "names.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise DatasetReleaseContractError(
                "cannot read explicit retrieval groups for split {}: {}".format(split, error)
            ) from error
        if (
            not isinstance(groups, list)
            or not isinstance(names, list)
            or len(groups) != len(names)
            or any(not isinstance(group, str) or not group.strip() for group in groups)
        ):
            raise DatasetReleaseContractError(
                "artifacts.splits.{}.retrieval_groups.json must be a non-empty-string list aligned with names.json".format(split)
            )
        retrieval_groups_by_split[split] = set(groups)

    # A verified source-level release must prove that a performance never
    # appears in more than one split.  The materializer already enforces this
    # before publication; repeat it here so a manually assembled but
    # hash-self-consistent build.json cannot claim a source-disjoint val/test
    # protocol while leaking another camera view of the same performance.
    for index, left_split in enumerate(_REQUIRED_SPLITS):
        for right_split in _REQUIRED_SPLITS[index + 1 :]:
            overlap = retrieval_groups_by_split[left_split] & retrieval_groups_by_split[right_split]
            if overlap:
                raise DatasetReleaseContractError(
                    "cross-split retrieval-group leakage between {} and {}: {}".format(
                        left_split, right_split, ", ".join(sorted(overlap)[:5])
                    )
                )

    counts = _require_mapping(build.get("counts"), "counts")
    materialized_windows = _require_mapping(counts.get("materialized_windows"), "counts.materialized_windows")
    val_count = materialized_windows.get("val")
    if isinstance(val_count, bool) or not isinstance(val_count, int) or val_count < 1:
        raise DatasetReleaseContractError(
            "verified release must contain at least one val window; test is held out from training-time selection"
        )
    try:
        val_names = json.loads((root / "val" / "names.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetReleaseContractError("cannot read validated val/names.json: {}".format(error)) from error
    if not isinstance(val_names, list) or len(val_names) != val_count or not val_names:
        raise DatasetReleaseContractError("counts.materialized_windows.val does not match non-empty val/names.json")

    return {
        "data_root": str(root),
        "release_contract": MATERIALIZED_BUILD_SCHEMA,
        "release_contract_validated": True,
        "headline_eligible": True,
        "headline_eligibility_scope": "dataset_release_only",
        "validation_split": "val",
        "validation_protocol": "SOURCE_DISJOINT_VAL_ONLY_TEST_HELD_OUT",
        "build": {"path": str(build_path), "sha256": _sha256_file(build_path)},
        "normalizer": {
            "path": str(normalizer_path),
            "sha256": normalizer_hash,
            "fit_split": normalizer["fit_split"],
            "fit_source_manifest_sha256": normalizer["fit_source_manifest_sha256"],
            "fit_report": normalizer.get("fit_report"),
            "fit_report_sha256": normalizer.get("fit_report_sha256"),
            "verified_against_build": True,
        },
        "source_manifest": source_manifest,
        "sequence_manifest": sequence_manifest,
        "label_manifest": label_manifest,
    }


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def make_dataset(root, split, limit=None, global_music=False,
                 global_music_shuffle_seed=None):
    dataset = AtomicSequenceDataset(
        root, split=split, global_music=global_music,
        global_music_shuffle_seed=global_music_shuffle_seed)
    if limit is not None:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    return dataset


def make_loader(dataset, batch_size, workers, shuffle, collate=None, sampler=None):
    if sampler is not None and shuffle:
        raise ValueError("a sampler owns the shuffling; do not ask the loader for it as well")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate or collate_atomic_sequences,
        drop_last=False,
        # Rebuilding workers every epoch re-forks a process that, on the wild
        # release, holds a 104 GiB prototype library.  Keep them alive.
        persistent_workers=bool(workers),
        prefetch_factor=4 if workers else None,
    )


class CompletionConditionCollate:
    """Build the completion stage's draft conditions inside the loader workers.

    ``completion_conditions`` used to run on the training process between the
    forward passes.  Measured on the wild 340-frame release with the full
    1,338,864-prototype library, it cost **0.589 s per batch of 64** while the
    model step on this card costs **0.184 s** -- so 62% of every step was the GPU
    waiting for a Python loop, which is exactly the 2.03 step/s the live W2 run
    advances at against the 5.42 step/s the same model reaches on synthetic
    conditions.

    Moving it here does not change a single value: the draft is a deterministic
    function of ``(labels, retrieval_group_id)`` with no RNG anywhere in the
    path, and ``tests/test_train_conditions.py`` asserts the two paths agree
    element-wise on the same batch.  What changes is *where* it runs -- in the
    ``num_workers`` loader processes, overlapped with the GPU, and parallel
    across them.

    The safe-coverage numerator and denominator are carried out per batch rather
    than reduced here, because the ``--min-safe-draft-fraction`` gate must keep
    firing on exactly the quantity it fired on before: safe atomic frames over
    all atomic frames in the batch.
    """

    def __init__(self, library, motion_dim, noise_ratio, align_timing=False,
                 seam_mask_half_width=0):
        self.library = library
        self.motion_dim = int(motion_dim)
        self.noise_ratio = float(noise_ratio)
        self.seam_mask_half_width = int(seam_mask_half_width)
        # Warp each prototype's settle points onto the TARGET window's settle
        # points, so the draft's timing is worth transmitting.  The measured
        # reason (dataset/atomic.warp_to_anchors): output-vs-own-draft speed
        # correlation -0.110 -- the model ignores draft timing because training
        # draft timing was uncorrelated with the target by construction.
        self.align_timing = bool(align_timing)

    def __call__(self, samples):
        batch = collate_atomic_sequences(samples)
        labels = batch["labels"]
        draft = torch.zeros(labels.shape[0], labels.shape[1], self.motion_dim, dtype=torch.float32)
        mask = torch.zeros(labels.shape[0], labels.shape[1], 1, dtype=torch.float32)
        safe_atomic_frames = 0
        atomic_frames = 0
        for index, (plan, retrieval_group_id) in enumerate(zip(labels, batch["retrieval_group_ids"])):
            if isinstance(retrieval_group_id, str) and retrieval_group_id.strip():
                self.library.fill_draft(
                    plan,
                    draft[index],
                    mask[index],
                    exclude_retrieval_group_ids=(retrieval_group_id,),
                    allow_missing=True,
                    align_to=batch["motion"][index] if self.align_timing else None,
                )
            # No explicit group means no proof that any prototype is external.
            # That sample stays zero-conditioned, as it was before this class.
            safe_atomic_frames += int(mask[index].sum().item())
            atomic_frames += int((plan != 0).sum().item())
        batch["draft"] = draft
        boundaries = plan_boundaries(labels)
        # Blank the draft's evidence around each seam BEFORE the noise ratio
        # scales it, so seam_mask_half_width=0 leaves the published behaviour
        # bit-identical.  See blank_seam_evidence for why.
        batch["draft_noise_mask"] = blank_seam_evidence(
            mask, boundaries, self.seam_mask_half_width) * self.noise_ratio
        batch["plan_boundaries"] = boundaries
        batch["safe_atomic_frames"] = safe_atomic_frames
        batch["atomic_frames"] = atomic_frames
        return batch


def batch_safe_draft_fraction(batch):
    atomic_frames = batch["atomic_frames"]
    return float(batch["safe_atomic_frames"] / atomic_frames) if atomic_frames else 1.0


def move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _loss_normalizer(args):
    """The release normalizer, for the geometric losses -- only when asked.

    Read from the release the run trains on, so the loss and the data cannot
    disagree about what a normalized unit means.  Returned as a plain dict; the
    model registers it as buffers (see MotionGeometry for why not a path).
    """
    wanted = (getattr(args, "fk_weight", 0.0) or getattr(args, "fk_velocity_weight", 0.0)
              or getattr(args, "contact_weight", 0.0))
    if not wanted:
        return None
    path = Path(args.data_root) / "normalizer.pt"
    return torch.load(str(path), map_location="cpu")


def _music_stats(args):
    """Load frozen music statistics when the run asked for them.

    Returned as a plain dict so the model can register them as buffers; the
    path is recorded in ``args`` for provenance, but the numbers themselves
    travel in the checkpoint, which is what makes them impossible to forget at
    inference (``load_state_dict`` is strict both ways).
    """
    path = getattr(args, "music_stats", None)
    if not path:
        return None
    payload = torch.load(str(path), map_location="cpu")
    if not {"mean", "std"} <= set(payload):
        raise ValueError("{} must hold 'mean' and 'std'".format(path))
    return payload


def planner_model(args):
    model = AtomicPlannerTransformer(
        num_atomic_classes=args.num_classes,
        music_dim=args.music_dim,
        latent_dim=args.latent_dim,
        num_layers=args.layers,
        num_heads=args.heads,
        ff_size=args.ff_size,
        dropout=args.dropout,
        max_seq_len=args.seq_len,
        global_music=getattr(args, "global_music", False),
        # Read with a default so a checkpoint saved before 2026-08-23 rebuilds
        # into exactly the module it was trained as.
        cond_drop_prob=getattr(args, "planner_cond_drop_prob", 0.0),
        head=getattr(args, "planner_head", "joint"),
        music_stats=_music_stats(args),
        # Read with a default so a checkpoint saved before this flag existed
        # rebuilds into exactly the module it was trained as.
        music_phase_features=getattr(args, "music_phase_features", False),
    )
    return UniformD3PM(
        model,
        num_steps=args.diffusion_steps,
        parameterization=getattr(args, "planner_parameterization", "x0"),
        # Read with a default so a checkpoint saved before this flag existed
        # rebuilds into exactly the module it was trained as.
        transition_weight=getattr(args, "planner_transition_weight", 1.0),
        high_noise_prob=getattr(args, "planner_high_noise_prob", 0.0),
    )


def completion_model(args):
    model = AtomicCompletionDecoder(
        motion_dim=args.motion_dim,
        seq_len=args.seq_len,
        music_dim=args.music_dim,
        latent_dim=args.latent_dim,
        ff_size=args.ff_size,
        num_layers=args.layers,
        num_heads=args.heads,
        dropout=args.dropout,
        # Opt-in second, discrete channel for the plan.  Measured on the wild
        # 340-frame arm: the generated motion follows the prototype it was
        # conditioned on by only 4-5% over a shuffled control, and a clean draft
        # does not move that, so the retrieved draft alone is too thin a channel
        # to carry "which atomic movement happens here".  Off unless asked for,
        # because every completion checkpoint before 2026-08-17 lacks it.
        num_classes=(args.num_classes if getattr(args, "completion_label_channel", False) else None),
        label_dim=getattr(args, "completion_label_dim", 64),
        music_stats=_music_stats(args),
        # Read with a default so a checkpoint saved before this flag existed
        # rebuilds into exactly the module it was trained as.
        music_phase_features=getattr(args, "music_phase_features", False),
        # The learned "no draft" token.  Built only when the run intends to drop
        # the draft, so it lands in the state_dict exactly for those runs and a
        # checkpoint cannot be rebuilt in the other mode without failing loudly.
        draft_guidance=bool(getattr(args, "draft_drop_prob", 0.0)),
    )
    return AtomicCompletionDiffusion(
        model,
        num_steps=args.diffusion_steps,
        transition_weight=args.transition_weight,
        # Read with a default so a checkpoint saved before 2026-08-23 rebuilds
        # into exactly the model it was trained as.
        velocity_weight=getattr(args, "velocity_weight", 0.0),
        velocity_skip_contact=getattr(args, "velocity_skip_contact", False),
        cond_drop_prob=args.cond_drop_prob,
        guidance_weight=args.guidance_weight,
        fk_weight=getattr(args, "fk_weight", 0.0),
        fk_velocity_weight=getattr(args, "fk_velocity_weight", 0.0),
        contact_weight=getattr(args, "contact_weight", 0.0),
        energy_match_weight=getattr(args, "energy_match_weight", 0.0),
        draft_drop_prob=getattr(args, "draft_drop_prob", 0.0),
        normalizer=_loss_normalizer(args),
    )


def build_library(dataset):
    motions = []
    labels = []
    names = []
    retrieval_group_ids = []
    for index in range(len(dataset)):
        sample = dataset[index]
        motions.append(sample["motion"])
        labels.append(sample["labels"])
        names.append(sample["name"])
        retrieval_group_ids.append(sample.get("retrieval_group_id"))
    library = AtomicMotionLibrary.from_sequences(
        motions,
        labels,
        names=names,
        retrieval_group_ids=retrieval_group_ids,
    )
    missing = sorted(set(range(1, 101)) - set(library.motions))
    if missing:
        print("Library does not contain classes: {}".format(missing))
    return library


def train_split_safe_draft_fraction(dataset, library):
    """The documented source-safe criterion, computed on the unit the plan states.

    ``docs/TRAINING_DATA_RELEASE.md`` records the release command's
    ``--min-safe-retrieval-fraction 0.99``, and ``audit_atomic_dataset.py``
    applies it to ``safe_frames / atomic_frames`` over the **whole train split**
    (``audit_source_safe_retrieval``).  That is the quantity the 2026-08-13
    failure was read on -- ``train: source-safe retrieval coverage 0.981138 <
    required 0.990000`` -- and the answer then was to change the vocabulary, not
    the floor.

    The per-batch check in the training loop uses the same 0.99 on a *different*
    unit, which no document states, and with batch 64 the two disagree on a
    corpus that passes: measured on wild_v5_song, the split is 0.999759 while the
    worst of 4,384 batches is 0.9901 and one real batch reached 0.989944.  That
    is batch-size variance, not a data property -- the shortfall is 38 of 4,522
    classes confined to a single train performance, and the audit's own
    permutation null puts the ceiling at 1.0, so it is genuine confinement
    rather than granularity.  So this function exists to check the stated
    quantity, and to check it *before* any card time rather than an hour in.

    Reads the memory-mapped label array directly where the release provides one,
    because the motion array is 38 GB here and none of it is needed.
    """
    import numpy as np

    groups_of = {}
    for label, prototypes in library.motions.items():
        groups_of[int(label)] = {
            prototype.retrieval_group_id for prototype in prototypes
            if isinstance(prototype.retrieval_group_id, str) and prototype.retrieval_group_id
        }

    arrays = getattr(dataset, "arrays", None)
    group_ids = getattr(dataset, "retrieval_group_ids", None)
    if arrays is not None and group_ids is not None:
        labels_array = arrays[2]
        rows = ((np.asarray(labels_array[index]), group_ids[index])
                for index in range(len(group_ids)))
    else:
        rows = ((np.asarray(dataset[index]["labels"]),
                 dataset[index].get("retrieval_group_id"))
                for index in range(len(dataset)))

    safe_frames = atomic_frames = 0
    stranded = set()
    for plan, own in rows:
        values, occurrences = np.unique(plan[plan > 0], return_counts=True)
        for value, occurrence in zip(values.tolist(), occurrences.tolist()):
            atomic_frames += occurrence
            available = groups_of.get(int(value), ())
            if isinstance(own, str) and own and (available - {own}):
                safe_frames += occurrence
            else:
                stranded.add(int(value))
    fraction = float(safe_frames / atomic_frames) if atomic_frames else 1.0
    return {"source_safe_atomic_frame_fraction": fraction,
            "atomic_frames": int(atomic_frames),
            "safe_atomic_frames": int(safe_frames),
            "classes_with_no_external_prototype_for_some_window": len(stranded)}


def completion_conditions(labels, retrieval_group_ids, library, motion_dim, noise_ratio, device):
    """Build completion conditions from ORACLE target atomic labels.

    This is appropriate for completion-stage training and a diagnostic, but it
    is not a self-driven planner-to-completion evaluation protocol.
    """
    if len(retrieval_group_ids) != len(labels):
        raise ValueError("completion condition retrieval groups and labels must have matching batch size")
    drafts = []
    masks = []
    safe_atomic_frames = 0
    atomic_frames = 0
    for plan, retrieval_group_id in zip(labels.cpu(), retrieval_group_ids):
        if isinstance(retrieval_group_id, str) and retrieval_group_id.strip():
            draft, mask = library.build_draft(
                plan,
                motion_dim,
                exclude_retrieval_group_ids=(retrieval_group_id,),
                allow_missing=True,
            )
        else:
            # No explicit group means no proof that any prototype is external.
            # This is intentionally zero-conditioned for legacy layouts.
            draft = torch.zeros(plan.shape[0], motion_dim, dtype=torch.float32)
            mask = torch.zeros(plan.shape[0], 1, dtype=torch.float32)
        drafts.append(draft)
        masks.append(mask * noise_ratio)
        safe_atomic_frames += int(mask.sum().item())
        atomic_frames += int((plan != 0).sum().item())
    return (
        torch.stack(drafts).to(device, non_blocking=True),
        torch.stack(masks).to(device, non_blocking=True),
        plan_boundaries(labels),
        float(safe_atomic_frames / atomic_frames) if atomic_frames else 1.0,
    )


@torch.no_grad()
def evaluate_planner(model, loader, device):
    model.eval()
    module = model.module if isinstance(model, DistributedDataParallel) else model
    batch = move_batch(next(iter(loader)), device)
    summary = batch.get("global_music")
    output = module.training_step(batch["labels"], batch["music"], batch["padding_mask"],
                                  global_music=summary)
    valid = ~batch["padding_mask"]
    denoising_accuracy = (
        (output.logits.argmax(dim=-1) == output.target_labels) & valid
    ).sum().float() / valid.sum().clamp_min(1)
    sample = module.sample(batch["music"], batch["padding_mask"], deterministic=False,
                           global_music=summary)
    sample_correct = (sample == batch["labels"]) & valid
    nonzero = valid & (batch["labels"] != 0)
    sample_accuracy = sample_correct.sum().float() / valid.sum().clamp_min(1)
    nonzero_accuracy = sample_correct[nonzero].float().mean() if nonzero.any() else torch.tensor(0.0, device=device)
    segment_counts = ((sample[:, 1:] != sample[:, :-1]) & valid[:, 1:]).sum(dim=1) + 1
    return {
        "loss": float(output.loss),
        "denoising_accuracy": float(denoising_accuracy),
        "sample_accuracy": float(sample_accuracy),
        "sample_nonzero_accuracy": float(nonzero_accuracy),
        "sample_transition_fraction": float(((sample == 0) & valid).sum().float() / valid.sum().clamp_min(1)),
        "sample_mean_segments": float(segment_counts.float().mean()),
        "sample_shape": list(sample.shape),
    }


@torch.no_grad()
def evaluate_completion(model, loader, library, args, device):
    """Run an ORACLE-plan completion diagnostic on one validation batch."""
    model.eval()
    batch = move_batch(next(iter(loader)), device)
    if "draft" in batch:
        draft = batch["draft"]
        mask = batch["draft_noise_mask"]
        boundaries = batch["plan_boundaries"]
        safe_draft_fraction = batch_safe_draft_fraction(batch)
    else:
        draft, mask, boundaries, safe_draft_fraction = completion_conditions(
            batch["labels"], batch["retrieval_group_ids"], library, args.motion_dim, args.draft_noise_ratio, device
        )
    module = model.module if isinstance(model, DistributedDataParallel) else model
    labels = batch["labels"] if getattr(args, "completion_label_channel", False) else None
    losses = module.training_step(batch["motion"], batch["music"], draft, mask, boundaries,
                                  labels=labels)
    sample = module.sample(batch["music"][:1], draft[:1], mask[:1], guidance_weight=1.0,
                           labels=None if labels is None else labels[:1])
    return {
        "evaluation_protocol": "ORACLE_GROUND_TRUTH_PLAN",
        "evaluation_kind": "ORACLE_COMPLETION_DIAGNOSTIC",
        "headline_eligible": False,
        "loss": float(losses.total),
        "denoising": float(losses.denoising),
        "transition": float(losses.transition),
        "sample_shape": list(sample.shape),
        "sample_finite": bool(torch.isfinite(sample).all()),
        "oracle_plan_safe_draft_condition_fraction": safe_draft_fraction,
    }


def save_checkpoint(path, model, optimizer, args, step, epoch, metrics=None, dataset_provenance=None):
    # Unwrap DDP before serializing.  A wrapped state dict prefixes every key
    # with ``module.``, and every consumer of these files -- infer_atomic.py, the
    # gate tools, ``--resume`` -- loads them with the bare model, so the prefix
    # would turn a finished multi-GPU run into a checkpoint nothing can read.
    module = model.module if isinstance(model, DistributedDataParallel) else model
    torch.save(
        {
            "stage": args.stage,
            "step": step,
            "epoch": epoch,
            "model": module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "metrics": metrics or {},
            "dataset_provenance": dataset_provenance,
        },
        str(path),
    )


def resolve_gpus(args):
    """Return the CUDA ordinals a run should occupy, or ``[]`` for the old path."""
    if not getattr(args, "gpus", ""):
        return []
    ordinals = []
    for piece in str(args.gpus).split(","):
        piece = piece.strip()
        if not piece:
            continue
        ordinals.append(int(piece))
    if len(set(ordinals)) != len(ordinals):
        raise ValueError("--gpus lists the same device twice: {}".format(args.gpus))
    available = torch.cuda.device_count()
    for ordinal in ordinals:
        if ordinal < 0 or ordinal >= available:
            raise ValueError("--gpus names cuda:{} but this host has {} devices".format(ordinal, available))
    return ordinals


DEFAULT_BATCH_SIZE = 16


def resolve_batch_sizes(args, world_size):
    """Split the declared batch across ranks, and say which quantity was declared.

    Data-parallel training changes the recipe unless someone says which number is
    held fixed, and the two choices are not interchangeable:

    * ``--batch-size`` is **per rank**.  Eight ranks of 64 is a global batch of
      512: the same number of epochs then costs 1/8 the optimizer steps at 8x the
      batch, which is a different optimization problem, not a faster version of
      the same one.  Learning rate and schedule have to be revisited before such
      a checkpoint is compared with a single-GPU arm.
    * ``--global-batch-size`` is the batch the optimizer sees.  Each rank takes
      ``global // world_size``, DDP averages the rank gradients, and with equal
      per-rank sizes that average *is* the mean over the global batch -- so the
      recipe is the single-GPU one, only spread over more cards.  This is what to
      use when a run has to stay comparable with an existing arm.

    Passing both is refused rather than resolved, because whichever one this
    function silently ignored would be the one the reader assumed was in force.

    "Was it typed" comes from argparse leaving the default in place, not from
    reading ``sys.argv``.  The first version scanned argv for the literal token
    ``--batch-size``, and argparse accepts any unambiguous prefix: ``--batch 64
    --global-batch-size 96`` then set ``batch_size`` to 64, saw no literal match,
    and resolved to the global 96 -- discarding the operator's number in exactly
    the way this refusal exists to prevent.
    """
    declared_global = getattr(args, "global_batch_size", None)
    declared_per_rank = getattr(args, "batch_size", None)
    if declared_global is not None and declared_per_rank is not None:
        raise ValueError("--batch-size and --global-batch-size both given; declare one")
    if declared_global is None:
        per_rank = DEFAULT_BATCH_SIZE if declared_per_rank is None else int(declared_per_rank)
        return per_rank, per_rank * world_size, "per-rank"
    if declared_global % world_size:
        raise ValueError(
            "--global-batch-size {} is not divisible by the {} ranks; an uneven split would "
            "make the gradient average something other than the mean over that batch".format(
                declared_global, world_size
            )
        )
    return int(declared_global // world_size), int(declared_global), "global"


def train(args):
    """Entry point: launch one rank per requested GPU, or run the single-GPU path."""
    gpus = resolve_gpus(args)
    if len(gpus) > 1:
        return launch_distributed(args, gpus)
    if gpus:
        args.device = "cuda:{}".format(gpus[0])
    # A single-process run goes through the same resolver, so that
    # ``--global-batch-size`` means the same thing at world size 1 as it does at
    # world size 6 rather than being quietly dropped on the one-card path.
    args.batch_size, _, _ = resolve_batch_sizes(args, world_size=1)
    return run_training(args, rank=0, world_size=1, shared=None)


def launch_distributed(args, gpus):
    """Build the corpus once in this process, then fork one rank per GPU.

    The fork is the point.  On the wild 340-frame release the prototype library
    costs 33 s to build and 104 GiB resident; ``torchrun`` would pay both once per
    rank, and six ranks would need 624 GiB of it.  Forking after the library is
    built leaves every rank reading the *same* physical pages -- and the vectorised
    retrieval index added alongside this keeps that true in practice, because a
    query now touches one prototype object instead of walking the label's whole
    candidate list and dirtying every page it lands on.

    Two things this process must not do before forking, both of which produce a
    crash with nothing in it that names the cause:

    * **Touch CUDA.**  A CUDA context created here leaves every child failing
      with ``initialization error``.  Devices are selected inside the children.
    * **Enter an OpenMP parallel region.**  libgomp's thread pool does not
      survive ``fork``: the child inherits a descriptor for threads that do not
      exist in it, and the next parallel region segfaults.  Measured here as
      SIGSEGV inside ``at::native::randperm_out_cpu`` -> ``GOMP_parallel``, from
      ``DistributedSampler.__iter__`` permuting 134,756 indices -- a crash whose
      stack names the sampler and says nothing about the library build that
      actually caused it.  ``build_library`` reaches OpenMP through the
      ``.float()`` cast of each 340x151 window (51,340 elements, above ATen's
      32,768 grain size), so the pre-fork phase runs single-threaded and each
      child restores its own thread count.  ``at::parallel_for`` skips the
      ``#pragma`` entirely at one thread, so no pool is ever created.

    Autograd is the third such trap, and it is why nothing here runs a backward
    pass -- see tests/test_train_distributed.py.
    """
    if args.log_every_epochs < 1 or args.save_every_epochs < 1:
        raise ValueError("epoch logging and saving intervals must be positive")
    parent_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    shared = prepare_corpus(args)
    per_rank, global_batch, declared = resolve_batch_sizes(args, len(gpus))
    if per_rank < 1:
        raise ValueError(
            "--global-batch-size {} over {} ranks leaves {} samples per rank".format(
                args.global_batch_size, len(gpus), per_rank
            )
        )
    windows = len(shared["train_dataset"])
    per_rank_windows = windows // len(gpus)
    print(
        "distributed: ranks={} gpus={} batch_declared={} per_rank_batch={} global_batch={}\n"
        "             train_windows={} per_rank_windows={} dropped_tail_per_epoch={}\n"
        "             steps_per_epoch={} (single-GPU equivalent at global batch: {})".format(
            len(gpus), ",".join(str(gpu) for gpu in gpus), declared, per_rank, global_batch,
            windows, per_rank_windows, windows - per_rank_windows * len(gpus),
            -(-per_rank_windows // per_rank), -(-windows // global_batch),
        ),
        flush=True,
    )
    port = _free_port()
    torch.multiprocessing.start_processes(
        _distributed_entry,
        args=(args, gpus, port, shared, per_rank, parent_threads),
        nprocs=len(gpus),
        start_method="fork",
        join=True,
    )
    return None


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _distributed_entry(rank, args, gpus, port, shared, per_rank_batch, parent_threads):
    # This child has no inherited OpenMP pool, so it may build its own.  Split
    # the parent's budget rather than giving every rank the whole machine, which
    # would oversubscribe the cores the loader workers also need.
    torch.set_num_threads(max(1, parent_threads // len(gpus)))
    torch.cuda.set_device(gpus[rank])
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:{}".format(port),
        rank=rank,
        world_size=len(gpus),
    )
    try:
        args.device = "cuda:{}".format(gpus[rank])
        args.batch_size = per_rank_batch
        # ``vars(args)`` is written into every checkpoint.  Without these two, a
        # later reader would find ``batch_size: 16`` and have no way to tell a
        # four-rank run of global batch 64 from a single-card run of batch 16.
        args.data_parallel_world_size = len(gpus)
        args.data_parallel_global_batch_size = per_rank_batch * len(gpus)
        run_training(args, rank=rank, world_size=len(gpus), shared=shared)
    finally:
        torch.distributed.destroy_process_group()


def prepare_corpus(args):
    """Validate the release and materialize the datasets and prototype library."""
    dataset_provenance = validate_training_data_root(args.data_root)
    validation_split = dataset_provenance["validation_split"]
    global_music = bool(getattr(args, "global_music", False))
    if global_music and args.stage != "planner":
        raise ValueError("--global-music conditions the planner; the paper's "
                         "completion stage takes periodic music input")
    shuffle_seed = getattr(args, "global_music_shuffle_seed", None)
    train_dataset = make_dataset(args.data_root, "train", args.limit,
                                 global_music=global_music,
                                 global_music_shuffle_seed=shuffle_seed)
    validation_dataset = make_dataset(args.data_root, validation_split,
                                      args.validation_limit,
                                      global_music=global_music,
                                      global_music_shuffle_seed=shuffle_seed)
    library = build_library(train_dataset) if args.stage == "completion" else None
    # Declared and CHECKED against the data before a single optimizer step: the
    # claim "one token is one bar" is what inference reads off the checkpoint.
    declare_planner_token_resolution(args, train_dataset)
    return {
        "dataset_provenance": dataset_provenance,
        "validation_split": validation_split,
        "train_dataset": train_dataset,
        "validation_dataset": validation_dataset,
        "library": library,
    }


def run_training(args, rank=0, world_size=1, shared=None):
    if args.log_every_epochs < 1 or args.save_every_epochs < 1:
        raise ValueError("epoch logging and saving intervals must be positive")
    # Ranks must not draw the same diffusion timesteps and dropout masks for
    # different data -- that would make the extra ranks less than a full sample.
    # DDP broadcasts rank 0's parameters at construction, so the weights are
    # identical regardless of what this seeds.
    seed_everything(args.seed + rank)
    device = resolve_device(args.device)
    is_leader = rank == 0
    # This precedes every AtomicSequenceDataset construction.  A verified
    # release supplies a source-disjoint val split; historical package data is
    # tolerated only as an explicitly ineligible code-smoke fallback.
    if shared is None:
        shared = prepare_corpus(args)
    dataset_provenance = shared["dataset_provenance"]
    validation_split = shared["validation_split"]
    train_dataset = shared["train_dataset"]
    validation_dataset = shared["validation_dataset"]
    library = shared["library"]

    collate = None
    if args.stage == "completion":
        # The documented floor, on the documented unit, before a single step.
        # This repository's own rule for gates (CLAUDE.md 2) is that they should
        # fail where failing is cheap: the previous form of this check ran inside
        # the training loop, so a corpus that could never pass it still cost an
        # hour of a card before saying so, and a corpus that passes the plan's
        # criterion could still be killed by one unlucky batch.
        coverage = train_split_safe_draft_fraction(train_dataset, library)
        if is_leader:
            print("source-safe draft coverage over the train split: {:.6f} "
                  "({}/{} atomic frames, {} class(es) with no external prototype "
                  "for some window; floor {:.6f})".format(
                      coverage["source_safe_atomic_frame_fraction"],
                      coverage["safe_atomic_frames"], coverage["atomic_frames"],
                      coverage["classes_with_no_external_prototype_for_some_window"],
                      args.min_safe_draft_fraction), flush=True)
        if coverage["source_safe_atomic_frame_fraction"] < args.min_safe_draft_fraction:
            raise RuntimeError(
                "source-safe draft coverage {:.6f} < required {:.6f} over the train "
                "split; the vocabulary strands classes in a single performance -- "
                "change the vocabulary or the split, do not disable source "
                "exclusion to fill prototype conditions".format(
                    coverage["source_safe_atomic_frame_fraction"],
                    args.min_safe_draft_fraction))
        args.train_split_safe_draft_fraction = coverage["source_safe_atomic_frame_fraction"]
        collate = CompletionConditionCollate(
            library, args.motion_dim, args.draft_noise_ratio,
            seam_mask_half_width=args.draft_seam_mask_width,
            align_timing=getattr(args, "draft_timing_align", False))
    sampler = None
    if world_size > 1:
        # drop_last, not padding.  DistributedSampler's default repeats samples to
        # square the split, which would put a handful of windows twice in one
        # epoch; the tail it drops instead is printed by launch_distributed.
        sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank,
                                     shuffle=True, seed=args.seed, drop_last=True)
    train_loader = make_loader(train_dataset, args.batch_size, args.workers,
                               shuffle=sampler is None, collate=collate, sampler=sampler)
    # One deterministic batch is read from this loader at the end of the run, so
    # it gets no workers: forking eight more copies of a process holding the
    # library would cost more than the batch does.
    validation_loader = make_loader(validation_dataset, args.batch_size, 0, False, collate=collate)

    if args.stage == "planner":
        model = planner_model(args).to(device)
    else:
        model = completion_model(args).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    start_step = 0
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        start_epoch = int(checkpoint.get("epoch", 0))
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[device.index],
            output_device=device.index,
            # The diffusion wrappers' buffers are the beta schedule and its
            # derived constants: computed identically on every rank from the same
            # arguments, never written to during training.  Broadcasting them
            # every step would move them across PCIe for nothing.
            broadcast_buffers=False,
        )

    amp_mode = getattr(args, "amp", "off")
    tf32 = bool(getattr(args, "tf32", False))
    amp_dtype = torch.bfloat16 if amp_mode == "bf16" else None
    step_context = (
        (lambda: torch.autocast("cuda", dtype=amp_dtype))
        if amp_dtype is not None
        else contextlib.nullcontext
    )
    if tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if is_leader:
        print(
            "device={} stage={} parameters={} samples={} validation_split={} validation_protocol={} release_contract_validated={} world_size={} amp={} tf32={}".format(
                device,
                args.stage,
                sum(parameter.numel() for parameter in model.parameters()),
                len(train_dataset),
                validation_split,
                dataset_provenance["validation_protocol"],
                dataset_provenance["release_contract_validated"],
                world_size,
                amp_mode,
                tf32,
            )
        )
    output_dir = Path(args.output_dir)
    if is_leader:
        output_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    step = start_step
    completed_epochs = start_epoch
    stop = args.max_steps is not None and step >= args.max_steps
    remaining_updates = max(0, (args.epochs - start_epoch) * len(train_loader))
    if args.max_steps is not None:
        remaining_updates = min(remaining_updates, max(0, args.max_steps - step))
    progress = tqdm(total=remaining_updates, desc="{} training".format(args.stage), unit="step",
                    disable=not is_leader)
    log_loss = 0.0
    log_epochs = 0
    # Wall-clock, reported at the end.  A run whose only symptom is that it is
    # slow has nothing that reports it -- this repository has paid for that twice
    # (the 26-100 window/min featuriser, and the draft builder this loop no
    # longer runs).  The clock restarts after the first step so the loader's
    # worker fork is not billed to the steady state, and both figures are kept.
    loop_start = time.perf_counter()
    steady_start = None
    steady_steps = 0
    for epoch in range(start_epoch, args.epochs):
        if stop:
            break
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch_loss = 0.0
        epoch_steps = 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with step_context():
                if args.stage == "planner":
                    output = model(
                        batch["labels"], batch["music"], batch["padding_mask"],
                        global_music=batch.get("global_music"),
                    )
                    loss = output.loss
                else:
                    # The floor itself is checked once, before any card time, on
                    # the split-level quantity the plan states -- see
                    # train_split_safe_draft_fraction.  What remains here is a
                    # runtime tripwire for the library or the provenance breaking
                    # mid-run: a batch that has atomic frames and not one
                    # source-safe prototype among them cannot be produced by
                    # class confinement (that costs 0.02% of frames on this
                    # corpus), only by the exclusion or the group ids going wrong.
                    if batch["atomic_frames"] and not batch["safe_atomic_frames"]:
                        raise RuntimeError(
                            "no source-safe prototype for any of this batch's {} atomic frame(s); "
                            "the retrieval library or the retrieval_group_id provenance broke at "
                            "runtime -- do not disable source exclusion to fill prototype "
                            "conditions".format(batch["atomic_frames"])
                        )
                    output = model(
                        batch["motion"], batch["music"], batch["draft"],
                        batch["draft_noise_mask"], batch["plan_boundaries"],
                        labels=(batch["labels"] if args.completion_label_channel else None),
                    )
                    loss = output.total
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            step += 1
            if steady_start is None:
                steady_start = time.perf_counter()
            else:
                steady_steps += 1
            loss_value = float(loss.detach())
            epoch_loss += loss_value
            epoch_steps += 1
            progress.update(1)
            progress.set_postfix(epoch=epoch + 1, loss="{:.4f}".format(loss_value))
            if args.max_steps is not None and step >= args.max_steps:
                stop = True
                break
        if world_size > 1 and epoch_steps:
            # The logged loss must describe the batch the optimizer saw, not the
            # slice this rank happened to hold.
            reduced = torch.tensor([epoch_loss, float(epoch_steps)], device=device)
            torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.SUM)
            epoch_loss = float(reduced[0]) / world_size
        if epoch_steps == len(train_loader):
            completed_epochs = epoch + 1
            epoch_average = epoch_loss / epoch_steps
            log_loss += epoch_average
            log_epochs += 1
            if completed_epochs % args.log_every_epochs == 0:
                progress.write(
                    "epoch={} mean_loss={:.6f}".format(
                        completed_epochs, log_loss / max(log_epochs, 1)
                    )
                )
                log_loss = 0.0
                log_epochs = 0
            if completed_epochs % args.save_every_epochs == 0 and is_leader:
                checkpoint_path = output_dir / "{}_epoch{}_step{}.pt".format(
                    args.stage, completed_epochs, step
                )
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    args,
                    step,
                    completed_epochs,
                    {"train_loss": epoch_average},
                    dataset_provenance=dataset_provenance,
                )
                progress.write("checkpoint={}".format(checkpoint_path))
        if stop:
            break
    progress.close()

    executed = step - start_step
    wall_seconds = time.perf_counter() - loop_start
    steady_seconds = (time.perf_counter() - steady_start) if steady_start is not None else 0.0
    throughput = {
        "steps": executed,
        "wall_seconds": round(wall_seconds, 2),
        "step_per_s": round(executed / wall_seconds, 4) if wall_seconds > 0 else None,
        "steady_step_per_s": round(steady_steps / steady_seconds, 4) if steady_steps and steady_seconds > 0 else None,
        "global_windows_per_s": (
            round(steady_steps * args.batch_size * world_size / steady_seconds, 1)
            if steady_steps and steady_seconds > 0 else None
        ),
    }
    if is_leader:
        print("throughput={}".format(json.dumps(throughput, sort_keys=True)), flush=True)

    if not is_leader:
        # Every rank holds the same weights, so only one writes them.  The
        # barrier keeps the followers alive until that write is done: tearing
        # down the process group underneath a rank that is still reducing would
        # abort the run after the training finished.
        torch.distributed.barrier()
        return None

    metrics = (
        evaluate_planner(model, validation_loader, device)
        if args.stage == "planner"
        else evaluate_completion(model, validation_loader, library, args, device)
    )
    metrics["validation_split"] = validation_split
    metrics["validation_protocol"] = dataset_provenance["validation_protocol"]
    if world_size > 1:
        metrics["data_parallel_world_size"] = world_size
        metrics["global_batch_size"] = args.batch_size * world_size
    metrics["amp"] = amp_mode
    metrics["throughput"] = throughput
    checkpoint_path = output_dir / "{}_step{}.pt".format(args.stage, step)
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        args,
        step,
        completed_epochs,
        metrics,
        dataset_provenance=dataset_provenance,
    )
    result = {
        "checkpoint": str(checkpoint_path),
        "epoch": completed_epochs,
        "step": step,
        "metrics": metrics,
        "dataset_provenance": dataset_provenance,
    }
    print(json.dumps(result, indent=2))
    if world_size > 1:
        torch.distributed.barrier()
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("planner", "completion"), required=True)
    parser.add_argument("--data-root", default="data/atomic_aistpp")
    parser.add_argument("--output-dir", default="runs/atomic_debug")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--log-every-epochs", type=int, default=5)
    parser.add_argument("--save-every-epochs", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=32)
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="per-rank batch. Left unset rather than defaulted to {} here so that "
             "resolve_batch_sizes can tell 'not given' from 'given the default "
             "value', which is what makes refusing both flags possible".format(
                 DEFAULT_BATCH_SIZE),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--motion-dim", type=int, default=151)
    parser.add_argument("--music-dim", type=int, default=35)
    parser.add_argument("--global-music", action="store_true",
                        help="planner only: additionally condition on a summary of the "
                             "whole track, broadcast to every frame.  The paper's "
                             "planner is full-music-awared; ours sees the window's 150 "
                             "frames and nothing else, and one generated segment "
                             "boundary in five sits on the window grid as a result")
    parser.add_argument("--global-music-shuffle-seed", type=int, default=None,
                        help="the null for --global-music: hand every sequence another "
                             "sequence's whole-track summary.  A whole-track vector is "
                             "constant within a song and can therefore degenerate into "
                             "a song id; a gain that survives this shuffle is not "
                             "measuring music")
    parser.add_argument("--seq-len", type=int, default=150)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ff-size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument(
        "--planner-parameterization",
        choices=["x0", "eq3"],
        default="x0",
        help="D3PM reverse parameterization; eq3 is the original literal "
             "reading of the paper and is kept only for comparison",
    )
    parser.add_argument("--transition-weight", type=float, default=1.0)
    parser.add_argument(
        "--velocity-weight", type=float, default=0.0,
        help="weight on the first-difference MSE, the term EDGE has and this "
             "completion did not.  Measured 2026-08-23 without it: asked to "
             "reproduce an essentially clean input the model returns a root "
             "path 397%% as long.  Default 0.0 so existing checkpoints rebuild "
             "unchanged")
    parser.add_argument(
        "--velocity-skip-contact", action="store_true",
        help="drop the four foot-contact channels from the velocity term.  "
             "Measured on the training set: they take 86.03%% of that term's "
             "budget while being strictly binary and flipping on 11.61%% of "
             "frame pairs, so --velocity-weight 4.0 spends most of its effort "
             "smoothing the one signal that must switch sharply, and leaves "
             "global orientation with 0.60%%.  Off by default so every earlier "
             "checkpoint reproduces from its own arguments.")
    parser.add_argument("--cond-drop-prob", type=float, default=0.25)
    parser.add_argument("--guidance-weight", type=float, default=2.0)
    parser.add_argument(
        "--planner-head", choices=("joint", "factorised"), default="joint",
        help="'joint' is one softmax over transition plus the K atomic classes, "
             "which couples them: uncertainty about which movement belongs here "
             "reads as confidence that there is none.  'factorised' gives "
             "transition its own sigmoid and normalises the classes among "
             "themselves.  Same K+1 distribution, same D3PM, different gradient. "
             "Default 'joint' so existing checkpoints rebuild unchanged")
    parser.add_argument(
        "--draft-timing-align", action="store_true",
        help="warp each training prototype's settle points onto the target "
             "segment's settle points before it becomes the draft. Measured "
             "2026-08-31: the completion transmits NONE of its draft's timing "
             "(output-vs-draft speed correlation -0.110 over 50 clips) because "
             "the training drafts' timing was uncorrelated with the target by "
             "construction, so ignoring it was the optimum. This makes the "
             "timing worth transmitting; the inference half puts the draft's "
             "settles on the music's beats")
    parser.add_argument(
        "--fk-weight", type=float, default=0.0,
        help="EDGE auxiliary loss: MSE between forward-kinematics joint positions "
             "of the prediction and the target, in metres. The plain MSE weighs a "
             "wrist swing and a thigh swing identically and gives the root 0.117%% "
             "of the budget; this is the term that makes geometry visible")
    parser.add_argument(
        "--fk-velocity-weight", type=float, default=0.0,
        help="EDGE auxiliary loss on FK joint velocities -- the term aimed at limb "
             "lag-0 synchrony (0.91-1.00 across nine checkpoints, ground truth "
             "0.642) and torso twist (29-34 deg/s against 49.3)")
    parser.add_argument(
        "--energy-match-weight", type=float, default=0.0,
        help="EDGE-family auxiliary that is NOT mean-seeking: relative L1 between "
             "the prediction's and the target's mean root-relative joint speed per "
             "window. Every FK-space MSE measured on 2026-08-31 damped energy "
             "(0.79 -> 0.49-0.65); this scalar gets WORSE under damping, and its "
             "supervision carries per-song intensity (the e_corr 0.210 defect)")
    parser.add_argument(
        "--contact-weight", type=float, default=0.0,
        help="EDGE consistency term: predicted contacts gate predicted foot "
             "velocities (gate detached)")
    parser.add_argument(
        "--music-stats", type=str, default="",
        help="frozen per-channel music mean/std from tools/fit_music_normalizer.py. "
             "The raw 35-D librosa stack reaches both stages through one bare Linear "
             "with per-channel std spanning 0.02 to 64, so the beat one-hot enters at "
             "about 0.05%% of the input variance. The statistics are stored as buffers "
             "in the checkpoint, not merely referenced, so a checkpoint cannot be run "
             "without them",
    )
    parser.add_argument(
        "--music-phase-features", action="store_true",
        help="append five rhythm channels DERIVED on the fly from the raw 35-D "
             "features (music_dim 35 -> 40 inside the model; release arrays and "
             "MODEL_MUSIC_DIM untouched): beat-phase sin/cos interpolated between "
             "the channel-34 beats, a validity flag, a frames-to-next-beat "
             "countdown, and window-z-scored onset energy.  Beat PHASE is "
             "otherwise represented nowhere -- the one-hot reaches cross-attention "
             "through a bare Linear and the FiLM path mean-pools it away -- and "
             "the measured cost is per-clip energy correlation 0.210, with "
             "normalization alone reaching only 0.17-0.26 "
             "(docs/DANCE_QUALITY_DEFECTS.md section 12.1).  Carried structurally "
             "in the checkpoint (widened projection plus a marker buffer), so a "
             "mismatched load refuses in both directions; see MusicPhaseFeatures "
             "in model/atomic_planner.py",
    )
    parser.add_argument(
        "--planner-token-resolution", choices=PLANNER_TOKEN_RESOLUTIONS, default="frame",
        help="what ONE TOKEN of the planner is.  'frame' is every planner this "
             "repository has trained: on release_v3 that means 98.76%% of the "
             "tokens the model emits are 'same as the previous frame' while a "
             "150-frame window spans only about 2.5 bars "
             "(docs/DANCE_QUALITY_DEFECTS.md 27.4).  'bar' declares that one "
             "token is one 4-beat bar, which is the unit M1 cut the corpus on; "
             "it is CHECKED against the training rows, not merely recorded, and "
             "it is what infer_atomic --plan-bar-tokens refuses to run without.  "
             "Default 'frame' so every existing checkpoint rebuilds unchanged")
    parser.add_argument(
        "--planner-bar-pooling", choices=("mean", "mean_std", "mean_rhythm"),
        default="mean",
        help="with --planner-token-resolution bar, how the release pooled a "
             "bar's frames of music into one token.  Recorded so inference "
             "pools the same way; 'mean_std' doubles --music-dim")
    parser.add_argument(
        "--planner-bar-beats", type=int, default=4,
        help="with --planner-token-resolution bar, how many beats one bar holds. "
             "Recorded so inference cuts the bar lines the checkpoint was "
             "trained on; the T-line corpus is 4")
    parser.add_argument(
        "--planner-high-noise-prob", type=float, default=0.0,
        help="planner only: probability of drawing the training timestep from "
             "the noisiest tenth of the schedule instead of uniformly. The "
             "reverse chain STARTS at t = T-1, where the noisy labels are "
             "nearly uniform and the only information is the music, so that "
             "step decides the plan -- and uniform sampling gives it 1%% of the "
             "loss budget while the easy low-noise steps, where the answer is "
             "already in the input, take the rest. Measured 2026-09-10 on the "
             "music-aligned K=8 labels under GroupKFold over 226 recordings "
             "(3,240 held-out bars): logistic regression on the same music "
             "reads +0.0386 over the majority floor, this planner reads "
             "-0.0012. Default 0.0 so every existing checkpoint rebuilds "
             "unchanged")
    parser.add_argument(
        "--planner-transition-weight", type=float, default=1.0,
        help="planner only: weight of the transition/filler class in the "
             "planner's cross-entropy. Ground truth spends 30.4%% of its bars "
             "in that class and the planner's plans spend 12.3%%, so the "
             "generated dance is named-moving almost all the time while the "
             "real one keeps stepping out of the vocabulary. THIS IS NOT "
             "--transition-weight: that flag reaches the completion decoder "
             "only, and a planner run given --transition-weight 2.5 recorded "
             "2.5 in its checkpoint while producing weights bit-identical "
             "(max difference 0.0) to the 1.0 baseline. Planner runs now refuse "
             "that flag; this is the one that acts. Default 1.0 so every "
             "existing planner checkpoint rebuilds unchanged")
    parser.add_argument(
        "--planner-cond-drop-prob", type=float, default=0.0,
        help="probability of replacing the planner's music condition with a "
             "learned null during training, which is what makes classifier-free "
             "guidance possible at inference.  The completion stage has used "
             "0.25 from the start; the planner has had none, and its measured "
             "failure on held-out music is a back-off toward the unconditional "
             "prior (71%% transition).  Default 0.0 so every existing planner "
             "checkpoint rebuilds unchanged")
    parser.add_argument(
        "--draft-seam-mask-width", type=int, default=0, metavar="FRAMES",
        help="blank the draft's conditioning mask within this many frames of "
             "every plan boundary during completion training. The draft is two "
             "prototypes butt-jointed, so those frames carry a velocity step no "
             "dancer produced -- and at mask 1.0 the model is told that step IS "
             "the evidence. Measured on the T line 2026-09-05 (17 eval clips, "
             "filler excluded): jerk within +-2 frames of a unit boundary is "
             "0.3515 against ground truth's 0.2553 at the same frame indices, "
             "while the interior matches (0.2135 vs 0.2045) -- the defect is "
             "entirely at the join. The label channel is NOT blanked, so the "
             "model still knows which class it leaves and which it enters. "
             "0 is the published behaviour and reproduces every earlier "
             "checkpoint")
    parser.add_argument("--draft-noise-ratio", type=float, default=0.25)
    parser.add_argument(
        "--draft-drop-prob",
        type=float,
        default=0.0,
        help="probability of replacing a sample's whole draft with the learned null "
             "draft during training -- the mirror of --cond-drop-prob for the plan. "
             "Without it the draft sits in BOTH terms of classifier-free guidance and "
             "cancels out, so guidance amplifies the music and never the plan "
             "(measured 2026-09-01: swapping the entire draft moves the output only "
             "0.57x as far as re-rolling the noise). Requires a fresh run: it adds a "
             "parameter to the state_dict.",
    )
    # Off by default so every completion checkpoint trained before 2026-08-17
    # keeps loading; a run that wants the plan to reach the motion asks for it.
    parser.add_argument(
        "--completion-label-channel",
        action="store_true",
        help="condition the completion stage on the per-frame atomic label as "
             "well as the retrieved draft",
    )
    parser.add_argument("--completion-label-dim", type=int, default=64)
    parser.add_argument(
        "--min-safe-draft-fraction",
        type=float,
        default=0.99,
        help="minimum source-safe prototype-conditioned atomic-frame fraction for completion training",
    )
    parser.add_argument(
        "--gpus",
        default="",
        help="comma-separated CUDA ordinals, e.g. '0,2,3'.  More than one forks that "
             "many DDP ranks from this process, after the prototype library is built, "
             "so every rank shares one copy of it",
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=None,
        help="batch the optimizer sees, split evenly across ranks.  Use this instead "
             "of --batch-size when a multi-GPU run has to stay comparable with a "
             "single-GPU arm; --batch-size is per rank and multiplies the global batch "
             "by the rank count",
    )
    parser.add_argument(
        "--amp",
        choices=("off", "bf16"),
        default="off",
        help="bf16 autocast.  Measured 1.8x on both stages' model step on this card, "
             "but it changes the arithmetic, so a bf16 checkpoint is a different arm "
             "than an fp32 one and must not be read as the same run made faster",
    )
    parser.add_argument(
        "--tf32",
        action="store_true",
        help="allow TF32 matmuls; also changes fp32 arithmetic, same caveat as --amp",
    )
    options = parser.parse_args()
    if options.stage == "planner" and options.transition_weight != 1.0:
        # Refused, not ignored.  --transition-weight reaches
        # AtomicCompletionDecoder only; a planner run given 2.5 recorded 2.5 in
        # its checkpoint args and trained weights BIT-IDENTICAL to the 1.0
        # baseline (max difference 0.0, measured 2026-09-09).  That is an arm
        # named by a flag that did nothing, which is the defect family this
        # repository keeps paying for -- so it fails here instead.
        parser.error(
            "--transition-weight is a COMPLETION flag and reaches nothing in a "
            "planner run; use --planner-transition-weight, which weights the "
            "transition class in the planner's own cross-entropy.")
    return options


if __name__ == "__main__":
    train(parse_args())
