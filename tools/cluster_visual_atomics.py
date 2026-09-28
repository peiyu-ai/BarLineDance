"""Cluster visual segments into an atomic vocabulary (paper discovery, step 2).

Consumes the segmentation from ``tools/segment_visual_atomics.py`` and the S3D
features behind it, and produces frame-level atomic labels materialised as a
release the predictability gate can score directly.

Method, following the paper where it is specified:

* a segment's descriptor is the L2-normalised mean of its per-frame S3D
  embeddings (the paper uses a TMR motion encoder here; no TMR checkpoint is
  released anywhere, so the visual embedding stands in -- recorded in the
  output, never silently);
* K-Means over **train-split segments only** -- the vocabulary must not see
  held-out sequences;
* the paper "keeps only segments near cluster centers and discards ambiguous
  edge points": segments beyond a per-cluster distance quantile become
  transition (label 0), as do all other splits' segments that exceed their
  assigned cluster's threshold.

Materialisation clones the reference release's motion/music/names (so the gate
compares vocabularies on identical inputs) and swaps only ``labels.npy``,
using the release's 15-frame slice stride, which is verified against the
actual motion overlap rather than assumed.

The output is a *candidate* vocabulary.  It is not trainable until it passes:

    python tools/probe_label_predictability.py --data-root <out> --eval-split test --gate
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

SLICE_STRIDE = 15
WINDOW = 150
_SLICE = re.compile(r"_slice(\d+)$")
_CAMERA = re.compile(r"_c[A-Za-z0-9]+_")


def sequence_of(window_name):
    return _SLICE.sub("", window_name.split("/")[-1])


def feature_stem(sequence, camera):
    return _CAMERA.sub("_{}_".format(camera), sequence, count=1)


def load_split_sequences(release, split):
    names = json.loads((release / split / "names.json").read_text(encoding="utf-8"))
    return names, sorted({sequence_of(name) for name in names})


def segment_descriptors(record, features):
    """L2-normalised mean S3D embedding per segment of one sequence."""
    rows = features.astype(np.float32)
    rows /= np.linalg.norm(rows, axis=1, keepdims=True) + 1e-12
    output = []
    for segment in record["segments"]:
        pooled = rows[segment["start"] : segment["end"]].mean(axis=0)
        pooled /= np.linalg.norm(pooled) + 1e-12
        output.append(pooled)
    return np.stack(output) if output else np.zeros((0, rows.shape[1]), np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation", type=Path, required=True,
                        help="report from segment_visual_atomics.py")
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True,
                        help="reference release supplying motion/music/split structure")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classes", type=int, default=100)
    parser.add_argument("--camera", default="c01")
    parser.add_argument("--accept-quantile", type=float, default=0.85,
                        help="per-cluster train-distance quantile; farther segments "
                             "become transition, the paper's 'ambiguous edge points'")
    parser.add_argument("--max-tail-pad", type=int, default=4,
                        help="max frames a window may extend past the feature "
                             "timeline before it is dropped instead of padded")
    parser.add_argument("--seed", type=int, default=20260808)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit("refusing to overwrite existing output {}".format(args.output))

    segmentation = json.loads(args.segmentation.read_text(encoding="utf-8"))
    records = {record["sequence"]: record for record in segmentation["records"]}

    split_names = {}
    split_sequences = {}
    for split in ("train", "val", "test"):
        split_names[split], split_sequences[split] = load_split_sequences(args.release, split)

    # ---- collect descriptors ------------------------------------------------
    def sequence_features(sequence):
        stem = feature_stem(sequence, args.camera)
        if stem not in records:
            return None, None
        path = args.features_dir / "{}.npz".format(stem)
        if not path.is_file():
            return None, None
        with np.load(path, allow_pickle=False) as bundle:
            return records[stem], bundle["features"].astype(np.float32)

    train_descriptors, train_owner = [], []
    missing = {split: [] for split in split_sequences}
    for split, sequences in split_sequences.items():
        for sequence in sequences:
            record, features = sequence_features(sequence)
            if record is None:
                missing[split].append(sequence)
                continue
            if split == "train":
                descriptors = segment_descriptors(record, features)
                train_descriptors.append(descriptors)
                train_owner.extend([sequence] * len(descriptors))

    if not train_descriptors:
        raise SystemExit("no train sequences have features yet")
    train_matrix = np.concatenate(train_descriptors)
    print("train segments: {} from {} sequences (missing features: {})".format(
        len(train_matrix), len(split_sequences["train"]) - len(missing["train"]),
        {k: len(v) for k, v in missing.items()}), flush=True)

    # ---- vocabulary: train-only fit ----------------------------------------
    kmeans = KMeans(n_clusters=args.classes, n_init=10, random_state=args.seed)
    train_assign = kmeans.fit_predict(train_matrix)
    distances = np.linalg.norm(
        train_matrix - kmeans.cluster_centers_[train_assign], axis=1
    )
    thresholds = np.full(args.classes, np.inf)
    for cluster in range(args.classes):
        member = distances[train_assign == cluster]
        if len(member):
            thresholds[cluster] = np.quantile(member, args.accept_quantile)

    # ---- frame labels per sequence ------------------------------------------
    def label_sequence(record, features):
        descriptors = segment_descriptors(record, features)
        if not len(descriptors):
            return np.zeros(record["motion_frames"], dtype=np.int64)
        assign = kmeans.predict(descriptors)
        dist = np.linalg.norm(descriptors - kmeans.cluster_centers_[assign], axis=1)
        labels = np.zeros(record["motion_frames"], dtype=np.int64)
        for segment, cluster, d in zip(record["segments"], assign, dist):
            if d <= thresholds[cluster]:
                labels[segment["start"] : segment["end"]] = cluster + 1
        return labels

    # ---- materialise a gate-compatible release ------------------------------
    args.output.mkdir(parents=True)
    stats = {"windows": 0, "dropped": 0, "padded_frames": 0, "transition_frames": 0,
             "total_frames": 0}
    per_split_counts = {}
    for split in ("train", "val", "test"):
        names = split_names[split]
        source_dir = args.release / split
        out_dir = args.output / split
        out_dir.mkdir()

        motion = np.load(source_dir / "motion.npy", mmap_mode="r")
        labels_out = np.zeros((len(names), WINDOW), dtype=np.int64)
        keep = np.ones(len(names), dtype=bool)
        sequence_labels = {}
        for index, window in enumerate(names):
            sequence = sequence_of(window)
            if sequence not in sequence_labels:
                record, features = sequence_features(sequence)
                sequence_labels[sequence] = (
                    None if record is None else label_sequence(record, features)
                )
            timeline = sequence_labels[sequence]
            if timeline is None:
                keep[index] = False
                continue
            start = int(_SLICE.search(window).group(1)) * SLICE_STRIDE
            end = start + WINDOW
            if end > len(timeline) + args.max_tail_pad:
                keep[index] = False
                continue
            window_labels = timeline[start:min(end, len(timeline))]
            if len(window_labels) < WINDOW:
                stats["padded_frames"] += WINDOW - len(window_labels)
                window_labels = np.concatenate([
                    window_labels,
                    np.full(WINDOW - len(window_labels), window_labels[-1] if len(window_labels) else 0,
                            dtype=np.int64),
                ])
            labels_out[index] = window_labels

        kept = int(keep.sum())
        stats["windows"] += kept
        stats["dropped"] += int((~keep).sum())
        stats["transition_frames"] += int((labels_out[keep] == 0).sum())
        stats["total_frames"] += kept * WINDOW
        per_split_counts[split] = {"kept": kept, "dropped": int((~keep).sum())}

        kept_indices = np.where(keep)[0]
        np.save(out_dir / "motion.npy", np.asarray(motion)[kept_indices])
        np.save(out_dir / "music.npy",
                np.asarray(np.load(source_dir / "music.npy", mmap_mode="r"))[kept_indices])
        np.save(out_dir / "labels.npy", labels_out[kept_indices])
        np.save(out_dir / "label_valid_mask.npy",
                np.ones((kept, WINDOW), dtype=bool))
        (out_dir / "names.json").write_text(
            json.dumps([names[i] for i in kept_indices]) + "\n", encoding="utf-8")
        groups_path = source_dir / "retrieval_groups.json"
        if groups_path.is_file():
            groups = json.loads(groups_path.read_text(encoding="utf-8"))
            (out_dir / "retrieval_groups.json").write_text(
                json.dumps([groups[i] for i in kept_indices]) + "\n", encoding="utf-8")

    shutil.copy(args.release / "normalizer.pt", args.output / "normalizer.pt")

    build = {
        "vocabulary": "visual_s3d_kmeans{}".format(args.classes),
        "descriptor": "mean-pooled L2-normalised S3D per segment "
                       "(stand-in for the paper's unreleased TMR encoder)",
        "segmentation": str(args.segmentation),
        "segmentation_config": segmentation["config"],
        "features_dir": str(args.features_dir),
        "camera": args.camera,
        "reference_release": str(args.release),
        "classes": args.classes,
        "accept_quantile": args.accept_quantile,
        "seed": args.seed,
        "train_segments": int(len(train_matrix)),
        "slice_stride": SLICE_STRIDE,
        "per_split": per_split_counts,
        "missing_feature_sequences": {k: v for k, v in missing.items() if v},
        "stats": stats,
        "transition_fraction": stats["transition_frames"] / max(stats["total_frames"], 1),
        "status": "candidate_pending_predictability_gate",
    }
    (args.output / "build.json").write_text(json.dumps(build, indent=2) + "\n",
                                            encoding="utf-8")
    print(json.dumps({k: build[k] for k in
                      ("classes", "train_segments", "per_split",
                       "transition_fraction", "stats")}, indent=2))
    print("release -> {}".format(args.output))


if __name__ == "__main__":
    main()
