"""Adaptive segmentation via visual self-similarity (paper Alg. 1, steps 2-7).

Input is the per-motion-frame S3D embeddings from
``tools/extract_visual_features.py``; output is, per sequence, a set of cut
points and segments -- the motion units that later clustering will turn into an
atomic vocabulary.

The algorithm follows the paper line by line:

    2. cosine self-similarity matrix ``A``, one row per frame;
    3. row ``t`` of ``A`` is frame ``t``'s feature: how it relates to the whole
       sequence, which is what makes frames of the same motion unit look alike;
    4. append the normalised index ``t/T`` to bias clusters toward temporal
       contiguity;
    5. K-means with ``N`` clusters over these rows;
    6. cut points wherever the cluster label changes;
    7. iteratively merge segments shorter than ``L_min`` into the shorter
       temporal neighbour.

**The 3D motion can be fused in (``--motion-bundle``).**  The shipped M1 cuts on
S3D alone, and on 2026-08-19 that was measured against the modality everything
downstream consumes: boundary contrast 1.014 where a control that keeps the
segment lengths and moves the cuts reads 0.993, and a cut-at-the-peaks upper
bound reads 1.406 (``docs/VOCABULARY_DIAGNOSIS.md`` section 2.A).  So a second
self-similarity block is available, built the same way from the motion's
rotation block, and concatenated to the visual one before step 5.  Two things
about it:

* **the pose block is rot6d only -- absolute root position is excluded.**
  Frames 4:7 are where the dancer stands in the world; including them would
  make "these two frames are alike" mean "the dancer is in the same corner".
* **``--motion-weight`` is in units of "equal say"**: the motion block is
  rescaled so its typical row-to-row distance matches the visual block's, then
  multiplied by the weight.  1.0 therefore means the two modalities carry the
  same weight, not that the raw numbers were added.

Default is off, and off is *byte-identical to before this existed* -- verified
by re-running, not by reading: see
``tests/test_segment_visual_atomics_fusion.py``.

The paper leaves ``N``, ``L_min`` and the index weighting unspecified, so they
are explicit parameters here, recorded in every output.  The index weight
matters more than it looks: similarity rows live in ``[~0.6, 1]^T`` while
``t/T`` spans ``[0, 1]`` once, so unweighted it is one coordinate against
hundreds and does nothing.  The default scales it to the typical row-to-row
distance so "temporally local" is a real preference rather than a comment.
"""

import argparse
import json
import pathlib
import sys
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.cluster import KMeans


def cluster_count(frames, clusters, frames_per_cluster):
    """N for one sequence: fixed if requested, else scaled to length.

    AIST sequences span roughly 10 s to 47 s.  A fixed N that yields ~1.3 s
    segments on a 10 s sequence yields ~6 s segments on a 47 s one, far outside
    the paper's 1-2.5 s atomic range -- so unless the caller pins N, scale it
    with length and clamp to a sane band.
    """
    if clusters > 0:
        return clusters
    return int(np.clip(round(frames / frames_per_cluster), 2, 32))


def similarity_rows(features):
    """Step 2: cosine self-similarity; row t is frame t's descriptor (step 3)."""
    x = np.asarray(features, dtype=np.float64)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    return x @ x.T


def _median_step(rows):
    return float(np.median(np.linalg.norm(np.diff(rows, axis=0), axis=1)))


def pose_rows(motion):
    """The motion side of the fusion: self-similarity of the rotation block.

    ``motion`` is raw 151-D = 4 contact + 3 root position + 144 rot6d.  Only the
    144 are used; see the module docstring for why the root is left out.
    """
    motion = np.asarray(motion, dtype=np.float64)
    if motion.shape[1] < 151:
        raise ValueError("expected 151-D motion, got {}".format(motion.shape[1]))
    return similarity_rows(motion[:, 7:])


def segment_sequence(features, clusters, min_length, index_weight, seed,
                     motion=None, motion_weight=0.0, drop_visual=False):
    """Return ordered cut points for one sequence's [T, D] features."""
    frames = len(features)
    if frames < 2 * min_length:
        return [0, frames], np.zeros(frames, dtype=np.int64)

    rows = similarity_rows(features)                       # steps 2-3

    if drop_visual:
        if motion is None:
            raise ValueError("--drop-visual needs a motion bundle to cluster instead")
        if len(motion) != frames:
            raise ValueError("motion has {} frames and the visual features {}".format(
                len(motion), frames))
        rows = pose_rows(motion)
    elif motion is not None and motion_weight > 0:
        if len(motion) != frames:
            raise ValueError(
                "motion has {} frames and the visual features {}; the two index the "
                "same clock, so a mismatch is a wiring error, not something to "
                "truncate quietly".format(len(motion), frames))
        block = pose_rows(motion)
        visual_step = _median_step(rows)
        motion_step = _median_step(block)
        if motion_step <= 0:
            raise ValueError("the motion block is constant; nothing to fuse")
        # "Equal say" first, then the caller's weight -- so 1.0 means the two
        # modalities weigh the same rather than whatever their raw units imply.
        rows = np.concatenate(
            [rows, block * (motion_weight * visual_step / motion_step)], axis=1)
    index = np.arange(frames, dtype=np.float64) / frames   # step 4
    # Scale so the appended coordinate competes with real row distances.  This
    # reads the rows *after* any fusion, so the index keeps its meaning relative
    # to the whole descriptor rather than to the visual half of it.
    scale = index_weight * _median_step(rows) * frames / max(frames - 1, 1)
    augmented = np.concatenate([rows, (index * scale)[:, None]], axis=1)

    effective = min(clusters, max(2, frames // min_length))
    labels = KMeans(n_clusters=effective, n_init=10, random_state=seed).fit_predict(
        augmented
    )                                                      # step 5

    boundaries = [0] + [t for t in range(1, frames) if labels[t] != labels[t - 1]] + [frames]

    # Step 7: repeatedly fold the shortest offending segment into the shorter
    # of its neighbours until everything satisfies min_length.
    while len(boundaries) > 2:
        lengths = np.diff(boundaries)
        short = int(np.argmin(lengths))
        if lengths[short] >= min_length:
            break
        left_length = lengths[short - 1] if short > 0 else np.inf
        right_length = lengths[short + 1] if short + 1 < len(lengths) else np.inf
        # Removing the boundary on the side of the shorter neighbour merges
        # the short segment into it.
        if left_length <= right_length:
            del boundaries[short]
        else:
            del boundaries[short + 1]

    return boundaries, labels


def summarise(records, features_dir, config):
    durations = np.array([item["frames"] for record in records
                          for item in record["segments"]])
    return {
        "features_dir": str(features_dir),
        "config": config,
        "sequences": len(records),
        "total_segments": int(len(durations)),
        "segments_per_sequence": float(len(durations) / max(len(records), 1)),
        "duration_frames": {
            "mean": float(durations.mean()),
            "median": float(np.median(durations)),
            "p10": float(np.percentile(durations, 10)),
            "p90": float(np.percentile(durations, 90)),
            "min": int(durations.min()),
            "max": int(durations.max()),
        },
        "duration_seconds": {
            "mean": float(durations.mean() / 30.0),
            "median": float(np.median(durations) / 30.0),
        },
        "records": records,
    }


def merge(pattern, output):
    """Combine shard outputs into one segmentation report.

    Shards that ran with different parameters are refused rather than merged:
    a corpus segmented half at one L_min and half at another is a silent
    corruption that nothing downstream could detect, and every consumer of this
    file reads ``config`` as if it described every record in it.
    """
    from glob import glob

    paths = sorted(glob(pattern))
    if not paths:
        raise SystemExit("no shard reports match {}".format(pattern))
    records, config, features_dir, seen = [], None, None, set()
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            shard = json.load(handle)
        if config is None:
            config, features_dir = shard["config"], shard["features_dir"]
        elif shard["config"] != config:
            raise SystemExit("{} was segmented with {} but the first shard used {}".format(
                path, shard["config"], config))
        for record in shard["records"]:
            if record["sequence"] in seen:
                raise SystemExit("sequence {} appears in two shards".format(
                    record["sequence"]))
            seen.add(record["sequence"])
            records.append(record)
    records.sort(key=lambda record: record["sequence"])
    report = summarise(records, features_dir, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("merged {} shards -> {} sequences, {} segments, median {:.2f}s".format(
        len(paths), report["sequences"], report["total_segments"],
        report["duration_seconds"]["median"]))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clusters", type=int, default=0,
                        help="N of the per-sequence N-means; 0 (default) scales N "
                             "with sequence length via --frames-per-cluster")
    parser.add_argument("--frames-per-cluster", type=int, default=40,
                        help="target frames per cluster when --clusters is 0")
    parser.add_argument("--min-length", type=int, default=24,
                        help="L_min in 30 fps motion frames; 24 = 0.8 s")
    parser.add_argument("--index-weight", type=float, default=4.0,
                        help="relative weight of the appended t/T coordinate")
    parser.add_argument("--motion-bundle", type=Path, default=None,
                        help="raw bundle root (sequences.jsonl + sequences/<sha>/"
                             "motion_151_raw.npy) to fuse the 3D motion in")
    parser.add_argument("--drop-visual", action="store_true",
                        help="the motion-only control arm: fuse nothing, cluster the "
                             "motion block alone.  Not a proposal -- it removes the "
                             "modality Alg. 1 is defined on; it exists so a fused "
                             "arm's gain can be split into 'used motion at all' and "
                             "'used both'")
    parser.add_argument("--motion-weight", type=float, default=0.0,
                        help="weight of the motion self-similarity block, in units "
                             "of 'equal say' with the visual block; 0 (default) is "
                             "the shipped visual-only behaviour")
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1,
                        help="split the corpus across processes; each writes its own "
                             "--output, then one --merge-glob run combines them")
    parser.add_argument("--merge-glob", default=None,
                        help="combine shard outputs matching this glob into --output "
                             "instead of segmenting")
    args = parser.parse_args()

    if args.merge_glob:
        return merge(args.merge_glob, args.output)

    if args.drop_visual and args.motion_bundle is None:
        raise SystemExit("--drop-visual needs --motion-bundle")
    if not args.drop_visual and (args.motion_weight > 0) != (args.motion_bundle is not None):
        # Either half alone is a configuration that reads like fusion and is not
        # one; the failure would be invisible in the output.
        raise SystemExit("--motion-bundle and a non-zero --motion-weight go together")
    motion_index = {}
    if args.motion_bundle is not None:
        from tools.probe_segmentation_boundaries import (
            canonical_key, load_bundle_index)
        motion_index = load_bundle_index(args.motion_bundle)

    files = sorted(args.features_dir.glob("*.npz"))
    if args.limit is not None:
        files = files[: args.limit]
    if args.num_shards > 1:
        files = files[args.shard:: args.num_shards]
    if not files:
        raise SystemExit("no feature files under {}".format(args.features_dir))

    records = []
    durations = []
    for index, path in enumerate(files, start=1):
        with np.load(path, allow_pickle=False) as bundle:
            features = bundle["features"].astype(np.float32)
            meta = json.loads(str(bundle["meta"]))
        motion = None
        if motion_index:
            key = canonical_key(path.stem)
            if key not in motion_index:
                raise SystemExit(
                    "{} has no motion in the bundle (key {!r}); refusing rather than "
                    "segmenting half the corpus with fusion and half without".format(
                        path.name, key))
            motion = np.load(motion_index[key]).astype(np.float64)
            if len(motion) != len(features):
                trim = min(len(motion), len(features))
                if abs(len(motion) - len(features)) > max(2, 0.01 * trim):
                    raise SystemExit(
                        "{}: {} motion frames vs {} visual frames".format(
                            path.name, len(motion), len(features)))
                motion, features = motion[:trim], features[:trim]
        n_clusters = cluster_count(len(features), args.clusters, args.frames_per_cluster)
        boundaries, _ = segment_sequence(
            features, n_clusters, args.min_length, args.index_weight, args.seed,
            motion=motion, motion_weight=args.motion_weight,
            drop_visual=args.drop_visual,
        )
        segments = [
            {"start": int(a), "end": int(b), "frames": int(b - a)}
            for a, b in zip(boundaries[:-1], boundaries[1:])
        ]
        durations.extend(item["frames"] for item in segments)
        records.append(
            {
                "sequence": path.stem,
                "encoder": meta.get("encoder"),
                "motion_frames": int(len(features)),
                "boundaries": [int(b) for b in boundaries],
                "segments": segments,
            }
        )
        if index % 50 == 0 or index == len(files):
            print("[{}/{}] segmented".format(index, len(files)), flush=True)

    durations = np.array(durations)
    report = summarise(records, args.features_dir, {
        "clusters": args.clusters,
        "frames_per_cluster": args.frames_per_cluster,
        "min_length_frames": args.min_length,
        "index_weight": args.index_weight,
        "seed": args.seed,
        "fps": 30.0,
        "motion_weight": args.motion_weight,
        "drop_visual": bool(args.drop_visual),
        "motion_bundle": str(args.motion_bundle) if args.motion_bundle else None,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(
        "\n{} sequences -> {} segments ({:.1f}/seq), duration median {:.2f}s "
        "mean {:.2f}s (p10 {:.2f}s, p90 {:.2f}s)".format(
            len(records), len(durations), report["segments_per_sequence"],
            report["duration_frames"]["median"] / 30.0,
            report["duration_frames"]["mean"] / 30.0,
            report["duration_frames"]["p10"] / 30.0,
            report["duration_frames"]["p90"] / 30.0,
        )
    )
    print("report -> {}".format(args.output))


if __name__ == "__main__":
    main()
