#!/usr/bin/env python3
"""Did cutting on content change the segmentation, and in the direction claimed?

Two arms, same uploads, same Alg. 1 parameters, one difference: where the clips
were cut.  Three questions, in the order they can falsify the rebuild:

1. **Granularity.**  The paper's segments average 0.81 s.  The published corpus
   averages 1.022 s on full-length clips and 1.171 s on 6-7 s ones -- clip
   length pushes segment length monotonically, because Alg. 1 sets its cluster
   count to ``T / frames_per_cluster``.  If the content cut is worth anything it
   should shrink the spread across clips, not just the mean.
2. **False boundaries.**  Every clip contributes two segment boundaries that are
   really its own edges.  In the published corpus that is 10,564 of 78,227
   (13.5%).  Fewer, longer clips mean fewer of them; that is arithmetic, not a
   finding, so it is reported as a rate and next to the clip count that produced
   it.
3. **What the content cut threw away.**  Cutting on dancer presence discards
   footage the blind cut kept.  A method that improves every statistic by
   keeping only the easy 40% of the corpus has not improved anything, so the
   retained fraction is reported beside the wins rather than in a footnote.

The comparison deliberately stops before the vocabulary.  Clustering statistics
move with corpus size, and these two arms do not have the same number of
segments -- reading a coherence number across them would be reading the size.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List, Optional, Sequence

import numpy as np

FPS = 30.0


def segment_stats(path: pathlib.Path) -> Dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["records"]
    lengths: List[float] = []
    per_clip_mean: List[float] = []
    edge_boundaries = 0
    interior_boundaries = 0
    edge_segments = 0
    frames = 0
    for record in records:
        segment_frames = [s["frames"] for s in record["segments"]]
        if not segment_frames:
            continue
        lengths.extend(f / FPS for f in segment_frames)
        per_clip_mean.append(float(np.mean(segment_frames)) / FPS)
        frames += record["motion_frames"]
        # Every clip's first and last boundary are the clip's own edges.  A
        # segment can also *be* the whole clip, in which case both of its
        # boundaries are edges and it has no interior at all.
        edge_boundaries += 2
        interior_boundaries += max(len(record["boundaries"]) - 2, 0)
        edge_segments += min(len(segment_frames), 2)
    lengths_array = np.asarray(lengths)
    return {
        "clips": len(per_clip_mean),
        "segments": len(lengths),
        "video_seconds": round(frames / FPS, 1),
        "segment_seconds_mean": round(float(lengths_array.mean()), 4),
        "segment_seconds_median": round(float(np.median(lengths_array)), 4),
        "segments_per_second": round(len(lengths) / max(frames / FPS, 1e-9), 4),
        # The spread across clips is the granularity-consistency measure: the
        # published corpus's problem was not its mean, it was that the mean
        # depended on where ffmpeg's clock fell.
        "per_clip_mean_std": round(float(np.std(per_clip_mean)), 4),
        "per_clip_mean_p05_p95": [round(float(np.percentile(per_clip_mean, 5)), 4),
                                  round(float(np.percentile(per_clip_mean, 95)), 4)],
        "clip_seconds_mean": round(float(frames / FPS / max(len(per_clip_mean), 1)), 2),
        "boundaries_total": edge_boundaries + interior_boundaries,
        "boundaries_that_are_clip_edges": edge_boundaries,
        "edge_boundary_rate": round(
            edge_boundaries / max(edge_boundaries + interior_boundaries, 1), 4),
        # Also as a fraction of *segments*, which is the form the 13.5% figure
        # in the plan is quoted in -- two different denominators for the same
        # defect would read as two different measurements later.
        "segments_touching_a_clip_edge": edge_segments,
        "edge_segment_rate": round(edge_segments / max(len(lengths), 1), 4),
    }


def retained_seconds(ingest_root: Optional[pathlib.Path]) -> Dict:
    """How much of each upload the content cut kept, from the ingest manifests."""
    if ingest_root is None:
        return {}
    kept = 0.0
    total = 0.0
    uploads = 0
    reasons: Dict[str, int] = {}
    # One row per upload, the most recent.  The manifests are append-only, so
    # an upload re-ingested under --redo has two rows and summing both would
    # count its seconds and its clips twice -- reporting a corpus larger than
    # the one on disk, in the exact quantities this function exists to compare.
    from tools.ingest_wild_uploads import manifest_rows

    for record in manifest_rows(ingest_root).values():
        uploads += 1
        total += record.get("seconds", 0.0)
        for span in record.get("spans", []):
            kept += (span[1] - span[0]) / FPS
        for clip in record.get("clips", []):
            reason = clip.get("end_reason", clip.get("status", "?"))
            reasons[reason] = reasons.get(reason, 0) + 1
    return {"uploads": uploads, "upload_seconds": round(total, 1),
            "kept_seconds": round(kept, 1),
            "retained_fraction": round(kept / max(total, 1e-9), 4),
            "clip_end_reasons": reasons}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--blind", type=pathlib.Path, required=True)
    parser.add_argument("--content", type=pathlib.Path, required=True)
    parser.add_argument("--ingest-root", type=pathlib.Path, default=None)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--target-seconds", type=float, default=0.81,
                        help="the paper's mean segment length (Fig. 4a)")
    args = parser.parse_args(argv)

    blind = segment_stats(args.blind)
    content = segment_stats(args.content)
    report = {"blind": blind, "content": content,
              "retained": retained_seconds(args.ingest_root),
              "target_segment_seconds": args.target_seconds}

    rows = [
        ("clips", "clips", "{}"),
        ("segments", "segments", "{}"),
        ("video seconds", "video_seconds", "{}"),
        ("clip seconds (mean)", "clip_seconds_mean", "{}"),
        ("segment s (mean)", "segment_seconds_mean", "{}"),
        ("segment s (median)", "segment_seconds_median", "{}"),
        ("segments / s", "segments_per_second", "{}"),
        ("per-clip mean std", "per_clip_mean_std", "{}"),
        ("edge-boundary rate", "edge_boundary_rate", "{}"),
        ("edge-segment rate", "edge_segment_rate", "{}"),
    ]
    print("\n{:<22} {:>12} {:>12}".format("", "blind 16s", "content"))
    for label, key, fmt in rows:
        print("{:<22} {:>12} {:>12}".format(label, fmt.format(blind[key]),
                                            fmt.format(content[key])))
    print("\ntarget segment length (paper Fig. 4a): {} s".format(args.target_seconds))
    print("blind   |mean - target| = {:.4f}".format(
        abs(blind["segment_seconds_mean"] - args.target_seconds)))
    print("content |mean - target| = {:.4f}".format(
        abs(content["segment_seconds_mean"] - args.target_seconds)))
    if report["retained"]:
        print("\ncontent cut kept {} of {} upload seconds ({:.1%})".format(
            report["retained"]["kept_seconds"], report["retained"]["upload_seconds"],
            report["retained"]["retained_fraction"]))
        print("clip end reasons: {}".format(
            json.dumps(report["retained"]["clip_end_reasons"])))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("\nwrote {}".format(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
