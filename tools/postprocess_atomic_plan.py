#!/usr/bin/env python3
"""Paper M4b: clean a planned atomic label sequence before completion reads it.

Quoting the method: "we first perform a sliding-window majority vote to remove
isolated frame-level errors and smooth local inconsistencies.  We then apply a
minimum-duration merging heuristic that detects abnormally short segments and
reassigns them to the most semantically compatible neighboring segment."

The paper's own Tab. 3 prices this at FID_k 25.26 -> 24.02 and R 26.6 -> 27.5,
so it is not cosmetic: a single mislabelled frame inside an atomic segment
splits it into three, and the completion stage then retrieves three primitives
where one belongs.

Two decisions the paper leaves open, made explicitly here:

*"most semantically compatible neighbour"* -- with no semantic embedding of the
label space available at this point in the pipeline, compatibility falls back to
the neighbour a short segment is most plausibly a fragment of: the longer
neighbour, and on a tie the earlier one, since a fragment at a boundary belongs
to the movement it interrupted.  Passing ``--label-embeddings`` swaps in real
cosine similarity between label prototypes when a vocabulary provides them, and
the choice actually used is recorded in the report.

*transition label 0* -- never absorbs and is never absorbed by the duration
rule.  Transitions are the plan's connective tissue, not undersized atomic
movements; merging them away would delete exactly the frames the completion
stage is meant to synthesise.

Since 2026-08-23 the arithmetic lives in ``dataset.atomic`` and this module is
the CLI over it.  It used to be a second, independent implementation, and the
two disagreed on exactly the decision above: ``infer_atomic`` -- the path that
actually feeds the completion stage -- called the ``dataset.atomic`` one, which
ranked neighbours by length alone and therefore absorbed short atomic movements
into long transitions.  So the rule documented here was never the rule that ran.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dataset.atomic import TRANSITION  # noqa: E402,F401  (re-exported)
from dataset.atomic import labels_to_segments  # noqa: E402
from dataset.atomic import majority_vote as _majority_vote  # noqa: E402
from dataset.atomic import merge_short_segments as _merge_short_segments  # noqa: E402


def majority_vote(labels: np.ndarray, window: int) -> np.ndarray:
    """Sliding-window majority vote; ties keep the incumbent label."""
    labels = np.asarray(labels, dtype=np.int64)
    if window <= 1:
        return labels.copy()
    if window % 2 == 0:
        window += 1  # a centred window needs an odd width
    voted = _majority_vote(torch.from_numpy(labels), window)
    return voted.numpy().astype(np.int64)


def segments_of(labels: np.ndarray) -> List[Tuple[int, int, int]]:
    """Contiguous runs as (start, end_exclusive, label)."""
    labels = np.asarray(labels, dtype=np.int64)
    if len(labels) == 0:
        return []
    return [
        (segment.start, segment.end, segment.label)
        for segment in labels_to_segments(torch.from_numpy(labels))
    ]


def compatibility_matrix(labels: np.ndarray,
                         embeddings: Optional[np.ndarray]) -> Optional[torch.Tensor]:
    """Cosine similarity between label prototypes, as a dense [K, K] table.

    ``dataset.atomic.merge_short_segments`` indexes compatibility by
    ``[source, target]``, so the per-pair cosine this module used to compute on
    demand becomes a matrix here.  Labels outside the embedding table score 0.0,
    which is what the pairwise version returned for them -- an unknown label is
    not evidence of compatibility, so it falls through to the length rule.
    """
    if embeddings is None:
        return None
    size = int(max(int(labels.max()) + 1 if len(labels) else 0, len(embeddings)))
    table = np.zeros((size, embeddings.shape[1]), dtype=np.float64)
    table[: len(embeddings)] = embeddings
    norms = np.linalg.norm(table, axis=1)
    safe = np.where(norms > 0, norms, 1.0)
    unit = table / safe[:, None]
    matrix = unit @ unit.T
    matrix[norms == 0, :] = 0.0
    matrix[:, norms == 0] = 0.0
    return torch.from_numpy(matrix)


def merge_short_segments(labels: np.ndarray, minimum_frames: int,
                         embeddings: Optional[np.ndarray] = None) -> np.ndarray:
    """Reassign sub-minimum atomic segments to their most compatible neighbour."""
    labels = np.asarray(labels, dtype=np.int64)
    merged = _merge_short_segments(
        torch.from_numpy(labels.copy()),
        minimum_frames,
        compatibility_matrix(labels, embeddings),
        transition_policy="protect",
    )
    return merged.numpy().astype(np.int64)


def postprocess(labels: Sequence[int], *, window: int = 9, minimum_frames: int = 12,
                embeddings: Optional[np.ndarray] = None) -> Dict[str, object]:
    original = np.asarray(labels, dtype=np.int64)
    voted = majority_vote(original, window)
    merged = merge_short_segments(voted, minimum_frames, embeddings)
    return {
        "labels": merged,
        "segments_before": len(segments_of(original)),
        "segments_after_vote": len(segments_of(voted)),
        "segments_after": len(segments_of(merged)),
        "frames_changed": int((merged != original).sum()),
        "frames": int(len(original)),
        "window": window,
        "minimum_frames": minimum_frames,
        "transition_policy": "protect",
        "compatibility": "label_embedding_cosine" if embeddings is not None
        else "longer_neighbour_then_earlier",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plans", type=pathlib.Path, required=True,
                        help="plans JSON from tools/sample_planner_plans.py")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--window", type=int, default=9,
                        help="sliding-window width in frames (odd; 9 = 0.3 s at 30 fps)")
    parser.add_argument("--minimum-frames", type=int, default=12,
                        help="atomic segments shorter than this are merged away")
    parser.add_argument("--label-embeddings", type=pathlib.Path, default=None,
                        help="optional [K,D] .npy of label prototypes for compatibility")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = json.loads(args.plans.read_text(encoding="utf-8"))
    embeddings = np.load(args.label_embeddings) if args.label_embeddings else None

    entries = payload["plans"] if isinstance(payload, dict) and "plans" in payload else payload
    reports = []
    for entry in entries:
        result = postprocess(entry["labels"], window=args.window,
                             minimum_frames=args.minimum_frames, embeddings=embeddings)
        entry["labels"] = result.pop("labels").tolist()
        entry["postprocess"] = result
        reports.append(result)

    summary = {
        "sequences": len(reports),
        "segments_before": int(sum(r["segments_before"] for r in reports)),
        "segments_after": int(sum(r["segments_after"] for r in reports)),
        "frames_changed": int(sum(r["frames_changed"] for r in reports)),
        "frames": int(sum(r["frames"] for r in reports)),
        "window": args.window,
        "minimum_frames": args.minimum_frames,
        "compatibility": reports[0]["compatibility"] if reports else None,
    }
    summary["segment_reduction"] = (
        round(1.0 - summary["segments_after"] / summary["segments_before"], 4)
        if summary["segments_before"] else 0.0)
    if isinstance(payload, dict):
        payload["postprocess_summary"] = summary
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
