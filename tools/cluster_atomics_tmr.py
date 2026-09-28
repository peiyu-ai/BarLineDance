#!/usr/bin/env python3
"""Paper M2: cluster segments in TMR's motion-text space into an atomic vocabulary.

Quoting the method: "each segment is encoded by a pretrained TMR motion encoder
and projected into the joint motion-text embedding space; we then run K-Means to
group segments into recurring atomic movement prototypes... Because not every
segment exhibits a clear pattern, we keep only segments near cluster centers and
discard ambiguous edge points."

This is that step with the real encoder.  The repo's earlier
``cluster_visual_atomics.py`` stood in an S3D segment mean because TMR was
believed unavailable; it is available, ``tools/tmr_runtime.py`` loads it, and
``tools/probe_tmr_alignment.py`` shows the bridge into its space is wired
correctly (text semantics track joint speed at p = 5e-37).  So the substitution
is retired here rather than carried forward.

Discipline that is not optional:

* **K-Means sees train-split segments only.**  A vocabulary fitted on held-out
  sequences is a leak that no later gate can detect, because every split would
  then look equally well-described.
* **The accept quantile is fitted on train too.**  Val/test segments are
  measured against the train thresholds; letting each split set its own would
  quietly equalise their acceptance rates.
* **The metric is a recorded choice, because the paper does not state one.**
  The paper says to cluster in TMR's space but not with what distance.  TMR was
  trained with InfoNCE on a *cosine* similarity matrix (its config carries
  ``temperature: 0.1`` and ``threshold_selfsim: 0.8``, both angular
  quantities), so mu's magnitude is a direction the contrastive objective never
  constrained.  ``--l2-normalize`` runs k-means on the unit sphere, which is
  that geometry; the default is plain Euclidean on raw mu, which is what this
  repo's published vocabulary used.  Both are labelled in the report and in
  every label row, because they are different vocabularies and nothing
  downstream could tell them apart otherwise.
* **Where discarded segments go is a recorded choice, defaulting to the paper.**
  The paper defines ``y_i = 0`` as "no atomic movement at frame i", and a
  discarded ambiguous segment is precisely a span with no assigned atomic
  movement -- so ``--discard-as transition`` (the default) sends them to 0.
  The alternative, ``rejected``, writes -1 with a false mask, which is this
  repo's older kinematic-baseline convention.  The difference is not cosmetic:
  on the wild corpus 15.8% of frames are discarded, and under ``rejected`` the
  planner sees *no* transition supervision at all while 15.8% of frames leave
  the loss entirely -- yet the completion stage exists to synthesise exactly
  those transitions.

Output matches ``atomicdance-kinematic-atomic-labels-v1`` so
``materialize_atomic_windows`` consumes it unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.convert_motion_to_guofeats import (  # noqa: E402
    GuofeatsError,
    motion_151_to_guofeats,
)

SCHEMA_VERSION = "atomicdance-kinematic-atomic-labels-v1"
PRODUCER_VERSION = "tmr-atomic-discovery-v1"
TRANSITION = 0
REJECTED = -1


class ClusterError(RuntimeError):
    pass


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def squared_distances(features: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """[N,K] squared distances via ||a||^2 - 2a.b + ||b||^2.

    The literal broadcast ``((features[:,None,:] - centers[None,:,:])**2).sum(2)``
    materialises an [N, K, D] intermediate -- 6 GB at 30k segments and 100
    prototypes, every Lloyd iteration.  The expansion is one matmul instead, so
    the memory is [N, K] and the arithmetic goes through BLAS.  Clipped at zero
    because the identity can produce small negatives in floating point.
    """
    return np.maximum(
        (features ** 2).sum(axis=1)[:, None]
        - 2.0 * features @ centers.T
        + (centers ** 2).sum(axis=1)[None, :],
        0.0,
    )


def kmeans(features: np.ndarray, clusters: int, seed: int, iterations: int = 100
           ) -> Tuple[np.ndarray, np.ndarray]:
    """k-means++ init then Lloyd; deterministic given the seed."""
    rng = np.random.default_rng(seed)
    count = len(features)
    if count < clusters:
        raise ClusterError("{} segments cannot form {} clusters".format(count, clusters))
    centers = np.empty((clusters, features.shape[1]), dtype=np.float64)
    centers[0] = features[rng.integers(count)]
    closest = ((features - centers[0]) ** 2).sum(axis=1)
    for index in range(1, clusters):
        total = closest.sum()
        probabilities = closest / total if total > 0 else np.full(count, 1.0 / count)
        centers[index] = features[rng.choice(count, p=probabilities)]
        closest = np.minimum(closest, ((features - centers[index]) ** 2).sum(axis=1))
    assignment = np.zeros(count, dtype=np.int64)
    for _ in range(iterations):
        distances = squared_distances(features, centers)
        updated = distances.argmin(axis=1)
        if np.array_equal(updated, assignment):
            break
        assignment = updated
        for index in range(clusters):
            members = features[assignment == index]
            if len(members):
                centers[index] = members.mean(axis=0)
            else:
                # An empty cluster re-seeds on the worst-explained point rather
                # than collapsing the vocabulary silently.
                centers[index] = features[distances.min(axis=1).argmax()]
    distances = np.sqrt(squared_distances(features, centers))
    return distances.argmin(axis=1), centers


def assign_to_centers(embeddings: np.ndarray, centers: np.ndarray, thresholds: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(assignment, nearest distance, accepted) against given centres and thresholds.

    The rule a fitting run already applies to val/test -- nearest centre, kept if
    within that prototype's train-fitted threshold -- lifted out so a *frozen*
    vocabulary can label new segments with exactly the same arithmetic.
    """
    distances = np.sqrt(squared_distances(embeddings, centers))
    assignment = distances.argmin(axis=1)
    nearest = distances[np.arange(len(embeddings)), assignment]
    return assignment, nearest, nearest <= thresholds[assignment]


def load_frozen_producer(producer: pathlib.Path, *, classes: int, accept_quantile: float,
                         metric: str) -> Dict[str, object]:
    """Centres and thresholds of a published vocabulary, checked against this run.

    Why a frozen vocabulary is a first-class path: re-clustering is a new
    vocabulary even on identical inputs.  Re-running the T line's own K=20 fit
    on CPU (2026-09-22) left 85 of 2,015 segments on a different prototype and
    moved centres by up to 0.943, while assigning the same embeddings to the
    published ``producer.npz`` reproduced 2,012 of 2,015 published labels (the 3
    within 0.006 of a threshold, none a change of prototype).  And every planner,
    completion and selector checkpoint is bound to the label space, so adding
    clips must not re-draw it.

    The producer's own sibling ``report.json`` says which metric it was fitted
    in; a mismatch refuses, because the same centres read in another geometry
    are a different vocabulary under the same ``label_space_id``.
    """
    with np.load(producer, allow_pickle=False) as frozen:
        payload = {key: frozen[key] for key in frozen.files}
    found = int(payload["classes"][0])
    if found != classes:
        raise ClusterError("{} has {} classes; this run asks for {}".format(producer, found, classes))
    if abs(float(payload["accept_quantile"][0]) - accept_quantile) > 1e-12:
        raise ClusterError("{} was fitted at accept quantile {}; this run asks for {}".format(
            producer, float(payload["accept_quantile"][0]), accept_quantile))
    report_path = producer.with_name("report.json")
    if report_path.is_file():
        fitted_metric = json.loads(report_path.read_text(encoding="utf-8")).get(
            "clustering", {}).get("embedding_metric")
        if fitted_metric and fitted_metric != metric:
            raise ClusterError("{} was fitted in {}; this run embeds in {}".format(
                producer, fitted_metric, metric))
    return {"centers": np.asarray(payload["centers"], dtype=np.float64),
            "thresholds": np.asarray(payload["thresholds"], dtype=np.float64),
            "seed": int(payload["seed"][0]), "sha256": sha256_file(producer),
            "path": str(producer.resolve())}


_CAMERA = __import__("re").compile(r"_c\w+?_")


def build_row_index(rows: Dict[str, Dict]) -> Dict[str, Dict]:
    """Key each bundle row by every name a segmentation might call it.

    Segmentation names come from video filenames; bundle rows are keyed by
    ``recording_id``.  The two corpora disagree differently: AIST videos are
    per-camera (``..._c01_...``) while its motion is camera-agnostic
    (``..._cAll_...``), and wild clips are ``<video>__clipNNN`` on disk but
    ``tiktok:<video>:clipNNN`` in the manifest.  Resolving by alias here beats
    a per-corpus branch at every call site.
    """
    index: Dict[str, Dict] = {}

    def add(key: Optional[str], row: Dict) -> None:
        if key and key not in index:
            index[key] = row

    for recording, row in rows.items():
        add(recording, row)
        tail = recording.rsplit("/", 1)[-1]
        add(tail, row)
        add(_CAMERA.sub("_cAll_", tail, count=1), row)
        add(row.get("legacy_source_name"), row)
        parts = recording.split(":")
        if len(parts) >= 3:
            add("{}__{}".format(parts[-2], parts[-1]), row)  # tiktok:<video>:clipNNN
    return index


def resolve_row(index: Dict[str, Dict], stem: str) -> Optional[Dict]:
    for key in (stem, _CAMERA.sub("_cAll_", stem, count=1), stem.replace("__", ":")):
        row = index.get(key)
        if row is not None:
            return row
    return None


def unnormalize(motion: np.ndarray, normalizer_bundle: pathlib.Path) -> np.ndarray:
    """Invert apply_motion_normalizer, exactly as infer_atomic does.

    Forward kinematics needs real rotations and metres.  A min-max scaled 151-D
    row has neither: its rot6d columns are not orthonormalisable and its root
    translation lives in [-1, 1] units, so encoding it would describe a body
    that does not exist.  Labels must still *bind* to the normalized sequence
    the release is built from, which is why provenance comes from that manifest
    while the arithmetic happens here on the inverted array.
    """
    import torch

    state = torch.load(normalizer_bundle / "normalizer.pt", map_location="cpu",
                       weights_only=False)
    data_min = state["data_min"].float().numpy().reshape(-1)
    data_max = state["data_max"].float().numpy().reshape(-1)
    safe = np.where(data_max == data_min, 1.0, data_max - data_min)
    return (np.asarray(motion, dtype=np.float64) + 1.0) * safe / 2.0 + data_min


def load_rows(bundle: pathlib.Path,
              normalized_sequences: Optional[pathlib.Path] = None) -> Dict[str, Dict]:
    manifest = normalized_sequences or (bundle / "sequences.jsonl")
    return {json.loads(l)["recording_id"]: json.loads(l)
            for l in manifest.open(encoding="utf-8")}


def cache_fingerprint(*, bundle: pathlib.Path, segmentation_path: pathlib.Path,
                      normalized_sequences: Optional[pathlib.Path],
                      normalizer_bundle: Optional[pathlib.Path],
                      min_segment_frames: int, limit: Optional[int]) -> str:
    """Identify everything upstream of the embeddings.

    A cache keyed only by path would happily serve embeddings computed from a
    different segmentation, and the resulting vocabulary would look perfectly
    healthy while describing spans that no longer exist.  So the key hashes the
    segmentation and the manifest by content, not by name.
    """
    parts = [
        str(bundle.resolve()),
        sha256_file(segmentation_path),
        sha256_file(normalized_sequences) if normalized_sequences else "-",
        str(normalizer_bundle.resolve()) if normalizer_bundle else "-",
        str(min_segment_frames),
        str(limit),
    ]
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()


def save_embedding_cache(path: pathlib.Path, encoded: Dict[str, object],
                         fingerprint: str) -> None:
    owners = encoded["owners"]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        embeddings=encoded["embeddings"],
        recordings=np.asarray([owner[0] for owner in owners]),
        starts=np.asarray([owner[1] for owner in owners], dtype=np.int64),
        ends=np.asarray([owner[2] for owner in owners], dtype=np.int64),
        skipped=np.asarray([json.dumps(encoded["skipped"], sort_keys=True)]),
        fingerprint=np.asarray([fingerprint]),
    )


def load_embedding_cache(path: pathlib.Path, fingerprint: str) -> Optional[Dict[str, object]]:
    with np.load(path, allow_pickle=False) as cached:
        if str(cached["fingerprint"][0]) != fingerprint:
            raise ClusterError(
                "{} was built from different inputs than this run; delete it or "
                "point --embedding-cache elsewhere".format(path))
        return {
            "embeddings": cached["embeddings"],
            "owners": [(str(r), int(s), int(e)) for r, s, e in
                       zip(cached["recordings"], cached["starts"], cached["ends"])],
            "skipped": json.loads(str(cached["skipped"][0])),
        }


def encode_segments(bundle: pathlib.Path, segmentation: Dict, encoder, *,
                    min_segment_frames: int, batch_size: int,
                    normalized_sequences: Optional[pathlib.Path] = None,
                    normalizer_bundle: Optional[pathlib.Path] = None,
                    limit: Optional[int] = None) -> Dict[str, object]:
    rows = load_rows(bundle, normalized_sequences)
    index = build_row_index(rows)

    records = segmentation["records"]
    if limit is not None:
        records = records[:limit]

    features: List[np.ndarray] = []
    owners: List[Tuple[str, int, int]] = []   # (recording_id, start, end) in 30 fps frames
    # ``non_finite_features`` is its own category, not folded into
    # ``convert_failed``: the conversion does not raise for these, it returns
    # NaN.  Measured 2026-08-25 on wild_v5_song, one guofeat row in 280 of one
    # clip carried NaN in the six columns of joint 19's rot6d, from a degenerate
    # cross product in TMR's quaternion path (``qbetween`` normalises the cross
    # of two vectors that were parallel at that frame).  That is one row of
    # 238,115, and it cost the entire vocabulary -- see the guard in ``build``.
    skipped = {"no_motion": 0, "convert_failed": 0, "too_short": 0,
               "non_finite_features": 0}
    for record in records:
        row = resolve_row(index, record["sequence"])
        if row is None:
            skipped["no_motion"] += 1
            continue
        try:
            motion = np.load(bundle / row["motion_path"])
            if normalizer_bundle is not None:
                motion = unnormalize(motion, normalizer_bundle)
            payload = motion_151_to_guofeats(motion)
        except (GuofeatsError, IndexError, ValueError):
            skipped["convert_failed"] += 1
            continue
        guofeats, source_index = payload["features"], payload["source_frame_index"]
        for segment in record["segments"]:
            if segment["frames"] < min_segment_frames:
                skipped["too_short"] += 1
                continue
            lo = int(np.searchsorted(source_index, segment["start"], side="left"))
            hi = int(np.searchsorted(source_index, segment["end"], side="right"))
            if hi - lo < 4:
                skipped["too_short"] += 1
                continue
            window = guofeats[lo:hi]
            if not np.isfinite(window).all():
                skipped["non_finite_features"] += 1
                continue
            features.append(window)
            owners.append((row["recording_id"], int(segment["start"]), int(segment["end"])))
    if not features:
        raise ClusterError("no encodable segments")
    embeddings = encoder.encode_motion(features, batch_size=batch_size)
    return {"embeddings": embeddings, "owners": owners, "skipped": skipped}


def drop_non_finite(encoded: Dict[str, object]) -> int:
    """Remove embeddings that are not finite, in place, and say how many.

    Applied to whatever ``build`` ends up holding -- freshly encoded or served
    from ``--embedding-cache`` -- because a cache written before this check
    existed still carries the bad rows, and the fingerprint has no reason to
    miss.  A finite input is not a promise of a finite embedding, and one row
    is enough: a NaN center makes an entire distance column NaN, ``argmin``
    then returns that column for every segment, and every threshold and
    acceptance test downstream silently evaluates False.  That is what happened
    on wild_v5_song, 2026-08-25: 1 row of 238,115, and the published bundle had
    all 7,234,580 frames labelled "transition".
    """
    embeddings = np.asarray(encoded["embeddings"])
    finite = np.isfinite(embeddings).all(axis=1)
    dropped = int((~finite).sum())
    if dropped:
        encoded["embeddings"] = embeddings[finite]
        encoded["owners"] = [owner for owner, keep
                             in zip(encoded["owners"], finite) if keep]
    skipped = encoded.get("skipped")
    if isinstance(skipped, dict):
        skipped["non_finite_embeddings"] = dropped
    if not len(encoded["owners"]):
        raise ClusterError("every segment encoded to a non-finite embedding")
    return dropped


def build(*, bundle: pathlib.Path, segmentation_path: pathlib.Path, output_dir: pathlib.Path,
          classes: int, accept_quantile: float, min_segment_frames: int, seed: int,
          device: str, batch_size: int, discard_as: str = "transition",
          normalized_sequences: Optional[pathlib.Path] = None,
          normalizer_bundle: Optional[pathlib.Path] = None,
          sources_manifest: Optional[pathlib.Path] = None,
          l2_normalize: bool = False,
          embedding_cache: Optional[pathlib.Path] = None,
          limit: Optional[int] = None,
          producer: Optional[pathlib.Path] = None) -> Dict[str, object]:
    if discard_as not in ("transition", "rejected"):
        raise ClusterError("discard_as must be 'transition' or 'rejected'")
    metric = "cosine_l2_normalized_mu" if l2_normalize else "euclidean_raw_mu"
    frozen = (load_frozen_producer(producer, classes=classes, accept_quantile=accept_quantile,
                                   metric=metric) if producer is not None else None)
    if output_dir.exists():
        raise ClusterError("{} exists; label bundles publish into a new directory".format(output_dir))
    segmentation = json.loads(segmentation_path.read_text(encoding="utf-8"))
    fingerprint = cache_fingerprint(
        bundle=bundle, segmentation_path=segmentation_path,
        normalized_sequences=normalized_sequences, normalizer_bundle=normalizer_bundle,
        min_segment_frames=min_segment_frames, limit=limit)
    # The encoding is the expensive half and it does not depend on the metric,
    # so the two metrics can be compared on *identical* embeddings rather than
    # on two encodings that might differ for unrelated reasons.
    encoded = None
    if embedding_cache is not None and embedding_cache.is_file():
        encoded = load_embedding_cache(embedding_cache, fingerprint)
    if encoded is None:
        from tools.tmr_runtime import TMREncoder

        encoder = TMREncoder(device=device, text=False)
        encoded = encode_segments(bundle, segmentation, encoder,
                                  min_segment_frames=min_segment_frames,
                                  batch_size=batch_size,
                                  normalized_sequences=normalized_sequences,
                                  normalizer_bundle=normalizer_bundle, limit=limit)
        if embedding_cache is not None:
            save_embedding_cache(embedding_cache, encoded, fingerprint)
    dropped_embeddings = drop_non_finite(encoded)
    if dropped_embeddings:
        print("   dropped {} non-finite embedding(s)".format(dropped_embeddings),
              flush=True)
    # materialize_atomic_windows refuses labels that do not declare which
    # manifests the vocabulary was fitted against -- that binding is what stops
    # a vocabulary fitted on one corpus being silently applied to another.
    fit_source_sha = sha256_file(sources_manifest) if sources_manifest else None
    embeddings = encoded["embeddings"].astype(np.float64)
    if l2_normalize:
        embeddings = embeddings / (
            np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    owners = encoded["owners"]
    rows = load_rows(bundle, normalized_sequences)

    splits = np.array([rows[recording]["split"] for recording, _, _ in owners])
    train = splits == "train"
    if train.sum() < classes:
        raise ClusterError("only {} train segments for {} clusters".format(int(train.sum()), classes))

    if frozen is not None:
        # Every split, train included, is judged against the frozen train fit.
        centers, thresholds = frozen["centers"], frozen["thresholds"]
        seed = frozen["seed"]
        assignment, nearest, accepted = assign_to_centers(embeddings, centers, thresholds)
    else:
        assignment_train, centers = kmeans(embeddings[train], classes, seed)
        distances = np.sqrt(squared_distances(embeddings, centers))
        assignment = distances.argmin(axis=1)
        nearest = distances[np.arange(len(embeddings)), assignment]

        # Thresholds are a train-only statistic; val/test are judged against them.
        thresholds = np.zeros(classes, dtype=np.float64)
        for index in range(classes):
            members = nearest[train & (assignment == index)]
            thresholds[index] = np.quantile(members, accept_quantile) if len(members) else 0.0
        accepted = nearest <= thresholds[assignment]

    # Two structural invariants, both violated by the 2026-08-25 wild_v5_song
    # run, which exited 0 and published a bundle in which every one of the
    # 7,234,580 frames was "transition":
    #
    #   * ``kmeans`` re-seeds an empty cluster on the worst-explained point, so
    #     after it returns every cluster has at least one train member; a
    #     cluster with one member has that member's distance as its quantile
    #     and therefore accepts it.  An empty prototype is thus impossible, not
    #     merely unlikely.
    #   * The thresholds are the ``accept_quantile`` of train distances, so
    #     train acceptance is that quantile by construction.  Zero acceptance
    #     cannot happen with finite inputs.
    #
    # These are exact consequences of the code above rather than tuned bounds,
    # which is why they are allowed to refuse.
    train_sizes = np.array([int((train & (assignment == index)).sum())
                            for index in range(classes)])
    if not accepted.any() or not train_sizes.all():
        raise ClusterError(
            "refusing to publish a degenerate vocabulary: {} of {} prototypes "
            "have no train member and {} of {} segments were accepted.  With "
            "finite embeddings neither is reachable, so the input is what to "
            "look at -- {} segment(s) were dropped for non-finite features and "
            "{} for non-finite embeddings."
            .format(int((train_sizes == 0).sum()), classes,
                    int(accepted.sum()), len(accepted),
                    (encoded.get("skipped") or {}).get("non_finite_features", 0),
                    dropped_embeddings))

    # The id space has to say which metric produced it.  Two vocabularies that
    # share a label_space_id are, to everything downstream, the same vocabulary:
    # a planner trained on one and evaluated against the other would report a
    # number with no meaning and nothing would flag it.
    label_space_id = "tmr_atomic_{}{}_v1".format(classes, "_l2" if l2_normalize else "")

    staging = output_dir.with_name(output_dir.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "labels").mkdir(parents=True)
    if frozen is not None:
        # The same bytes, so the same producer sha: to every consumer this is
        # the vocabulary it already knows, applied to more segments.
        shutil.copyfile(frozen["path"], staging / "producer.npz")
    else:
        np.savez_compressed(staging / "producer.npz", centers=centers, thresholds=thresholds,
                            classes=np.asarray([classes]), seed=np.asarray([seed]),
                            accept_quantile=np.asarray([accept_quantile]))
    producer_sha = sha256_file(staging / "producer.npz")
    if frozen is not None and producer_sha != frozen["sha256"]:
        raise ClusterError("copied producer does not hash to its source")
    # Paths are recorded relative to the manifest.  Absolute paths would name
    # the .staging directory that the atomic publish renames away, leaving
    # every row in a complete-looking bundle pointing at nothing -- and they
    # would also pin the bundle to one location.  The consumer resolves
    # relative paths against the manifest root and refuses ones that escape it.

    per_sequence: Dict[str, List[int]] = {}
    for position, (recording, start, end) in enumerate(owners):
        per_sequence.setdefault(recording, []).append(position)

    label_rows = []
    counts = {"accepted_segments": int(accepted.sum()), "segments": len(owners),
              "sequences": 0, "valid_frames": 0, "transition_frames": 0, "rejected_frames": 0}
    try:
        for recording, positions in sorted(per_sequence.items()):
            row = rows[recording]
            frames = int(row["frame_count"])
            labels = np.full(frames, TRANSITION, dtype=np.int64)
            mask = np.ones(frames, dtype=bool)
            for position in positions:
                _, start, end = owners[position]
                end = min(end, frames)
                if end <= start:
                    continue
                if accepted[position]:
                    labels[start:end] = int(assignment[position]) + 1  # 0 stays transition
                elif discard_as == "transition":
                    labels[start:end] = TRANSITION
                else:
                    labels[start:end] = REJECTED
                    mask[start:end] = False
            store = staging / "labels" / hashlib.sha256(recording.encode()).hexdigest()
            store.mkdir(parents=True, exist_ok=True)
            np.save(store / "labels.npy", labels)
            np.save(store / "label_valid_mask.npy", mask)
            counts["sequences"] += 1
            counts["valid_frames"] += int((labels > 0).sum())
            counts["transition_frames"] += int((labels == TRANSITION).sum())
            counts["rejected_frames"] += int((labels == REJECTED).sum())
            label_rows.append({
                "schema_version": SCHEMA_VERSION,
                "producer_version": PRODUCER_VERSION,
                "producer_artifact": "producer.npz",
                "producer_artifact_sha256": producer_sha,
                "sequence_id": row["sequence_id"],
                "recording_id": recording,
                "retrieval_group_id": row["retrieval_group_id"],
                "duplicate_content_group_id": row.get("duplicate_content_group_id"),
                "split": row["split"],
                "status": "accepted",
                "frame_count": frames,
                "valid_frames": int((labels > 0).sum()),
                "transition_frames": int((labels == TRANSITION).sum()),
                "label_space_id": label_space_id,
                "embedding_metric": metric,
                "discarded_segments_become": discard_as,
                "fit_split": "train",
                "fit_source_manifest_sha256": fit_source_sha,
                "input_motion_representation_id": row.get(
                    "motion_representation_id", "AtomicDance_151D"),
                "input_coordinate_system": row.get(
                    "coordinate_system", "z_up_world_body_only"),
                "input_normalization_state": row.get("normalization_state", "raw"),
                "input_normalization_artifact_sha256": row.get(
                    "normalization_artifact_sha256"),
                "input_motion_sha256": row["motion_sha256"],
                "labels_path": "labels/{}/labels.npy".format(store.name),
                "labels_sha256": sha256_file(store / "labels.npy"),
                "label_valid_mask_path": "labels/{}/label_valid_mask.npy".format(store.name),
                "label_valid_mask_sha256": sha256_file(store / "label_valid_mask.npy"),
                "segmentation_encoder": segmentation["records"][0]["encoder"],
                "embedding_encoder": "TMR tmr_humanml3d_guoh3dfeats motion_encoder (mu)",
            })

        with (staging / "labels.jsonl").open("w", encoding="utf-8") as handle:
            for entry in label_rows:
                handle.write(json.dumps(entry, sort_keys=True) + "\n")

        sizes = np.bincount(assignment[accepted] + 1, minlength=classes + 1)[1:]
        report = {
            "schema_version": SCHEMA_VERSION,
            "producer_version": PRODUCER_VERSION,
            "counts": counts,
            "skipped_segments": encoded["skipped"],
            "clustering": {
                "classes": classes,
                "accept_quantile": accept_quantile,
                "seed": seed,
                "fit_split": "train",
                "train_segments": int(train.sum()),
                "mean_samples_per_prototype": round(float(sizes.mean()), 2),
                "median_samples_per_prototype": float(np.median(sizes)),
                "empty_prototypes": int((sizes == 0).sum()),
                "largest_prototype_fraction": round(float(sizes.max() / max(sizes.sum(), 1)), 4),
                "acceptance_rate": round(float(accepted.mean()), 4),
                "discarded_segments_become": discard_as,
                "label_space_id": label_space_id,
                "embedding_metric": metric,
                "frozen_producer": ({"path": frozen["path"], "sha256": frozen["sha256"],
                                     "note": "no k-means and no threshold fit in this run: "
                                             "every segment, train included, was assigned to "
                                             "these centres with these thresholds"}
                                    if frozen is not None else None),
                "metric_note": "the paper names TMR's space but no distance; TMR itself "
                               "was trained with InfoNCE on cosine similarity, so the "
                               "l2-normalized run is the space's own geometry and the "
                               "raw run keeps an axis the objective never constrained",
            },
            "encoders": {
                "segmentation": segmentation["records"][0]["encoder"],
                "segmentation_note": "paper specifies I3D; S3D is a recorded substitution",
                "embedding": "TMR tmr_humanml3d_guoh3dfeats motion_encoder (mu), paper-aligned",
            },
            "input": {
                "bundle": str(bundle.resolve()),
                "segmentation": str(segmentation_path.resolve()),
            },
            "publication": "immutable_new_directory_only_atomic_rename",
        }
        (staging / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                             encoding="utf-8")
        os.rename(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=pathlib.Path, required=True)
    parser.add_argument("--segmentation", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--classes", type=int, default=100)
    parser.add_argument("--accept-quantile", type=float, default=0.85,
                        help="per-cluster train distance quantile kept as atomic")
    parser.add_argument("--min-segment-frames", type=int, default=12)
    parser.add_argument("--normalized-sequences", type=pathlib.Path, default=None,
                        help="sequences_normalized.jsonl; labels bind to it and the "
                             "motion is unnormalized before encoding")
    parser.add_argument("--normalizer-bundle", type=pathlib.Path, default=None)
    parser.add_argument("--sources", type=pathlib.Path, default=None,
                        help="sources.jsonl the release will be built from")
    parser.add_argument("--discard-as", choices=("transition", "rejected"),
                        default="transition",
                        help="where ambiguous segments go: the paper's transition "
                             "token 0 (default), or -1 with a false mask")
    parser.add_argument("--l2-normalize", action="store_true",
                        help="k-means on the unit sphere, TMR's own cosine geometry; "
                             "default is Euclidean on raw mu")
    parser.add_argument("--embedding-cache", type=pathlib.Path, default=None,
                        help="npz to reuse the TMR encoding across runs that differ "
                             "only in clustering; written if absent, verified if present")
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--producer", type=pathlib.Path, default=None,
                        help="producer.npz of a published vocabulary: assign every segment "
                             "to its frozen centres and thresholds instead of re-clustering "
                             "(--classes/--accept-quantile/metric must match it)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build(bundle=args.bundle, segmentation_path=args.segmentation,
                       output_dir=args.output_dir, classes=args.classes,
                       accept_quantile=args.accept_quantile,
                       min_segment_frames=args.min_segment_frames, seed=args.seed,
                       device=args.device, batch_size=args.batch_size,
                       discard_as=args.discard_as,
                       normalized_sequences=args.normalized_sequences,
                       normalizer_bundle=args.normalizer_bundle,
                       sources_manifest=args.sources,
                       l2_normalize=args.l2_normalize,
                       embedding_cache=args.embedding_cache, limit=args.limit,
                       producer=args.producer)
    except (ClusterError, FileNotFoundError) as error:
        raise SystemExit("error: {}".format(error))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
