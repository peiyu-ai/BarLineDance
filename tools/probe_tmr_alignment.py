#!/usr/bin/env python3
"""T1 gate: is the bridge into TMR's motion-text space actually wired right?

Tensors flowing is not evidence.  A mirrored skeleton, a wrong axis convention,
a missed normalization or a frame-rate mistake all produce finite embeddings of
the correct shape while destroying the semantics.  TMR is a *joint* motion-text
space, so it admits a check nothing else in this repo can make: sentences must
retrieve the segments they describe.

Two gates, both label-free:

**G1 - text semantics track a physical quantity.**  "a person is standing
still" and "a person jumps energetically" are ranked against every segment.
The segments each retrieves are compared on median joint speed, computed
independently from the 151-D motion.  If the space is wired correctly the still
probe retrieves slower motion than the energetic one, with a Mann-Whitney
p-value to say so.  This needs no annotations and cannot be passed by accident:
a broken bridge gives two indistinguishable sets.

**G2 - neighbourhoods are genre-enriched.**  AIST encodes genre in the sequence
name (``gBR`` = break, ``gHO`` = house...).  A segment's nearest neighbours
should share its genre more often than the corpus base rate.  Dance genre is a
strong motion prior, so a space that has lost the motion would sit at chance.

Neither gate is about matching the paper's numbers; both are about refusing to
build M2 on a silently broken embedding.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.convert_motion_to_guofeats import (  # noqa: E402
    GuofeatsError,
    motion_151_to_guofeats,
    motion_151_to_joints,
)
from tools.tmr_runtime import TMREncoder, cosine_similarity  # noqa: E402

STILL_PROBES = [
    "a person is standing still",
    "a person barely moves",
    "a person stands in place without moving",
]
ENERGETIC_PROBES = [
    "a person jumps energetically",
    "a person leaps into the air",
    "a person performs a fast explosive jump",
]
_CAMERA = re.compile(r"_c\w+?_")
_GENRE = re.compile(r"^(g[A-Z]{2})_")


def to_cAll(name: str) -> str:
    """Segmentation names carry a camera id; the motion bundle is camera-agnostic."""
    return _CAMERA.sub("_cAll_", name, count=1)


def genre_of(name: str) -> Optional[str]:
    match = _GENRE.match(name)
    return match.group(1) if match else None


def median_joint_speed(motion: np.ndarray, fps: float = 30.0) -> float:
    joints = motion_151_to_joints(motion)
    if len(joints) < 2:
        return 0.0
    speed = np.linalg.norm(np.diff(joints, axis=0), axis=-1) * fps
    return float(np.median(speed))


def collect_segments(segmentation: Dict, bundle: pathlib.Path, limit_sequences: Optional[int],
                     min_frames: int = 12) -> Dict[str, List]:
    rows = {json.loads(l)["sequence_id"].split("/")[1]: json.loads(l)
            for l in (bundle / "sequences.jsonl").open(encoding="utf-8")}
    features: List[np.ndarray] = []
    speeds: List[float] = []
    genres: List[str] = []
    sequences: List[str] = []
    records = segmentation["records"]
    if limit_sequences is not None:
        records = records[:limit_sequences]
    skipped = 0
    for record in records:
        name = to_cAll(record["sequence"])
        row = rows.get(name)
        if row is None:
            skipped += 1
            continue
        motion = np.load(bundle / row["motion_path"])
        try:
            payload = motion_151_to_guofeats(motion)
        except (GuofeatsError, IndexError, ValueError):
            skipped += 1
            continue
        guofeats = payload["features"]
        source_index = payload["source_frame_index"]
        for segment in record["segments"]:
            if segment["frames"] < min_frames:
                continue
            # Segment bounds are 30 fps motion frames; map them onto the 20 fps
            # guofeats rows through the recorded source index.
            lo = int(np.searchsorted(source_index, segment["start"], side="left"))
            hi = int(np.searchsorted(source_index, segment["end"], side="right"))
            if hi - lo < 4:
                continue
            features.append(guofeats[lo:hi])
            speeds.append(median_joint_speed(motion[segment["start"]:segment["end"]]))
            genres.append(genre_of(record["sequence"]) or "??")
            sequences.append(name)
    return {"features": features, "speeds": np.asarray(speeds), "genres": np.asarray(genres),
            "sequences": np.asarray(sequences), "skipped": skipped}


def gate_text_semantics(embeddings: np.ndarray, encoder: TMREncoder, speeds: np.ndarray,
                        top_k: int) -> Dict:
    from scipy.stats import mannwhitneyu

    still = cosine_similarity(encoder.encode_text(STILL_PROBES), embeddings).mean(axis=0)
    fast = cosine_similarity(encoder.encode_text(ENERGETIC_PROBES), embeddings).mean(axis=0)
    still_top = np.argsort(-still)[:top_k]
    fast_top = np.argsort(-fast)[:top_k]
    statistic, p_value = mannwhitneyu(speeds[still_top], speeds[fast_top],
                                      alternative="less")
    return {
        "top_k": top_k,
        "still_median_speed_m_per_s": round(float(np.median(speeds[still_top])), 4),
        "energetic_median_speed_m_per_s": round(float(np.median(speeds[fast_top])), 4),
        "corpus_median_speed_m_per_s": round(float(np.median(speeds)), 4),
        "mannwhitney_u": float(statistic),
        "p_value_still_slower": float(p_value),
        "overlap_of_retrieved_sets": int(len(set(still_top) & set(fast_top))),
        "passes": bool(p_value < 0.01),
    }


def gate_genre_enrichment(embeddings: np.ndarray, genres: np.ndarray, neighbours: int) -> Dict:
    similarity = cosine_similarity(embeddings, embeddings)
    np.fill_diagonal(similarity, -np.inf)
    order = np.argsort(-similarity, axis=1)[:, :neighbours]
    same = (genres[order] == genres[:, None]).mean()
    _, counts = np.unique(genres, return_counts=True)
    fractions = counts / counts.sum()
    chance = float((fractions ** 2).sum() / fractions.sum())
    return {
        "neighbours": neighbours,
        "same_genre_fraction": round(float(same), 4),
        "chance_fraction": round(chance, 4),
        "lift": round(float(same) / chance, 3) if chance else None,
        "passes": bool(same > chance * 1.5),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--segmentation", type=pathlib.Path,
                        default=pathlib.Path("runs/visual_seg_v1/segmentation.json"))
    parser.add_argument("--bundle", type=pathlib.Path,
                        default=pathlib.Path("data/atomic_aistpp/aist_raw_performance_v1"))
    parser.add_argument("--limit-sequences", type=int, default=120)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--neighbours", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=pathlib.Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    segmentation = json.loads(args.segmentation.read_text(encoding="utf-8"))
    collected = collect_segments(segmentation, args.bundle, args.limit_sequences)
    features = collected["features"]
    if len(features) < 50:
        raise SystemExit("only {} usable segments; need more to test".format(len(features)))
    print("segments: {} (skipped {} sequences)".format(len(features), collected["skipped"]),
          flush=True)

    encoder = TMREncoder(device=args.device)
    embeddings = encoder.encode_motion(features)
    report = {
        "segments": len(features),
        "embedding_dim": int(embeddings.shape[1]),
        "all_finite": bool(np.isfinite(embeddings).all()),
        "segmentation_encoder": segmentation["records"][0]["encoder"],
        "G1_text_semantics": gate_text_semantics(embeddings, encoder, collected["speeds"],
                                                 args.top_k),
        "G2_genre_enrichment": gate_genre_enrichment(embeddings, collected["genres"],
                                                     args.neighbours),
    }
    report["T1_passes"] = bool(report["all_finite"]
                               and report["G1_text_semantics"]["passes"]
                               and report["G2_genre_enrichment"]["passes"])
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["T1_passes"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
