#!/usr/bin/env python3
"""Score a discovery run against the paper's published statistics -- on AIST++.

Every earlier alignment check in this repo was run on the wild corpus, where
the paper's numbers are not a target: a different corpus has a different
segment-length distribution and a different cluster population, so a gap there
is uninformative.  AIST++ is the corpus Fig. 4 was measured on, so it is the
only place where "does our discovery reproduce the paper's" is a question with
an answer.

What the paper states, and where:

* text, §4.1  -- 1,408 sequences, 5.2 hours;
* Fig. 4a     -- segment durations in five buckets, 23,194 segments total;
* Fig. 4b     -- prototypes by sample count, 100 prototypes, 268.57 avg;
* Fig. 4c     -- sub-prototypes by sample count, 730 of them, 31.8 avg.

Three of those cannot all be true at once, and this tool prints the arithmetic
rather than quietly picking one:

* Fig. 4a's own buckets, costed at their **lower** edges, need 20,109 s of
  footage.  5.2 hours is 18,720 s.  The histogram does not fit the corpus it
  is drawn from, so 5.2 h / 23,194 = 0.807 s is not a segment-length target --
  Fig. 4a's own mean is 0.867 s at the bucket floors, ~1.01 s at the midpoints.
* Fig. 4b implies 100 x 268.57 = 26,857 clustered segments, which is *more*
  than Fig. 4a's 23,194 total -- yet clustering only ever discards segments
  ("we keep only segments near cluster centers").  Retention above 100% is not
  a parameter one can match.
* Fig. 4c implies 730 x 31.8 = 23,214 segments, which agrees with Fig. 4a to
  0.1% and disagrees with Fig. 4b by 14%.

So the comparisons here are **rate-based and shape-based**, never raw counts:
segments per hour rather than segment totals, and cluster sizes expressed in
units of each corpus's own mean rather than absolute sample counts.  A corpus
of a different size and a vocabulary of a different density can be compared
that way; totals cannot.

Usage::

    python3 tools/report_paper_alignment.py --segmentation runs/aist_m1_v1/segmentation.json
    python3 tools/report_paper_alignment.py --segmentation ... \\
        --labels data/atomic_aistpp/aist_atomic_v1 \\
        --embedding-cache runs/aist_tmr_embeddings.npz --output runs/aist_alignment.json
    python3 tools/report_paper_alignment.py --segmentation ... --labels ... \\
        --embedding-cache ... --sub-labels data/atomic_aistpp/aist_v1_ingroup

``--sub-labels`` adds the Fig. 4c row.  It is separate from ``--labels``
because M3 is a separate method with its own published row in Tab. 2, and a
run that has not done the re-clustering should report M1 and M2 rather than a
zero for a stage it never ran.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

FPS = 30.0

PAPER_SEQUENCES = 1408
PAPER_HOURS = 5.2
PAPER_FIG4A = {"<0.7": 1366, "0.7-0.9": 7712, "0.9-1.1": 8054, "1.1-1.3": 2095, ">1.3": 3967}
PAPER_FIG4B = {"<200": 10, "200-250": 31, "250-300": 32, "300-350": 18, ">350": 9}
PAPER_FIG4C = {"<20": 109, "20-35": 431, "35-50": 99, ">50": 91}
PAPER_SAMPLES_PER_PROTOTYPE = 268.57
PAPER_SAMPLES_PER_SUBPROTOTYPE = 31.8
PAPER_SUBPROTOTYPES_PER_PROTOTYPE = 7.3

DURATION_EDGES = [(0.0, 0.7, "<0.7"), (0.7, 0.9, "0.7-0.9"), (0.9, 1.1, "0.9-1.1"),
                  (1.1, 1.3, "1.1-1.3"), (1.3, np.inf, ">1.3")]
# Fig. 4b's buckets, carried into units of the paper's own mean so a corpus
# with a different cluster density can still be compared on shape.
SIZE_EDGES = [(0.0, 200.0, "<200"), (200.0, 250.0, "200-250"), (250.0, 300.0, "250-300"),
              (300.0, 350.0, "300-350"), (350.0, np.inf, ">350")]
# Fig. 4c's buckets, carried into units of the paper's own sub-prototype mean
# for the same reason SIZE_EDGES is: this corpus's sub-prototypes hold 12.25
# segments against the paper's 31.8, so an absolute bucket would score the
# density difference twice -- once as a rate and again as a shape.
SUB_SIZE_EDGES = [(0.0, 20.0, "<20"), (20.0, 35.0, "20-35"),
                  (35.0, 50.0, "35-50"), (50.0, np.inf, ">50")]


def shares(values: Sequence[float], edges) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    total = max(1, len(values))
    return {name: float(((values >= low) & (values < high)).sum()) / total
            for low, high, name in edges}


def total_variation(ours: Dict[str, float], theirs: Dict[str, float]) -> float:
    return 0.5 * sum(abs(ours[k] - theirs[k]) for k in theirs)


def normalise(counts: Dict[str, int]) -> Dict[str, float]:
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def paper_self_consistency() -> Dict[str, object]:
    """The three published totals, and the two ways they contradict each other."""
    floor_seconds = sum(low * count for low, _, name in DURATION_EDGES
                        for count in [PAPER_FIG4A[name]])
    midpoints = {"<0.7": 0.35, "0.7-0.9": 0.8, "0.9-1.1": 1.0, "1.1-1.3": 1.2, ">1.3": 1.5}
    fig4a_total = sum(PAPER_FIG4A.values())
    mid_seconds = sum(midpoints[name] * count for name, count in PAPER_FIG4A.items())
    fig4b_total = 100 * PAPER_SAMPLES_PER_PROTOTYPE
    fig4c_total = 100 * PAPER_SUBPROTOTYPES_PER_PROTOTYPE * PAPER_SAMPLES_PER_SUBPROTOTYPE
    return {
        "corpus_seconds": PAPER_HOURS * 3600,
        "fig4a_segments": fig4a_total,
        "fig4a_seconds_at_bucket_floors": round(floor_seconds, 1),
        "fig4a_exceeds_corpus_by": round(floor_seconds / (PAPER_HOURS * 3600) - 1, 4),
        "fig4a_mean_at_floors": round(floor_seconds / fig4a_total, 3),
        "fig4a_mean_at_midpoints": round(mid_seconds / fig4a_total, 3),
        "corpus_over_fig4a_segments": round(PAPER_HOURS * 3600 / fig4a_total, 3),
        "fig4b_implied_segments": round(fig4b_total, 0),
        "fig4b_over_fig4a": round(fig4b_total / fig4a_total, 4),
        "fig4c_implied_segments": round(fig4c_total, 0),
        "fig4c_over_fig4a": round(fig4c_total / fig4a_total, 4),
        "note": ("Fig. 4a cannot fit 5.2 h, and Fig. 4b claims more clustered segments "
                 "than Fig. 4a segmented at all; Fig. 4c agrees with Fig. 4a. "
                 "So counts are not comparable targets -- rates and shapes are."),
    }


def m1_report(segmentation: pathlib.Path) -> Dict[str, object]:
    data = json.loads(segmentation.read_text(encoding="utf-8"))
    durations = np.array([s["frames"] for r in data["records"] for s in r["segments"]],
                         dtype=np.float64) / FPS
    frames = sum(r["motion_frames"] for r in data["records"])
    hours = frames / FPS / 3600
    ours = shares(durations, DURATION_EDGES)
    theirs = normalise(PAPER_FIG4A)
    return {
        "config": data.get("config"),
        "sequences": data["sequences"],
        "hours": round(hours, 4),
        "segments": int(len(durations)),
        "segments_per_hour": round(len(durations) / hours, 1),
        "paper_segments_per_hour": round(sum(PAPER_FIG4A.values()) / PAPER_HOURS, 1),
        "mean_seconds": round(float(durations.mean()), 3),
        "median_seconds": round(float(np.median(durations)), 3),
        "paper_mean_seconds_at_midpoints": paper_self_consistency()["fig4a_mean_at_midpoints"],
        "histogram": {k: round(v, 4) for k, v in ours.items()},
        "paper_histogram": {k: round(v, 4) for k, v in theirs.items()},
        "total_variation_from_fig4a": round(total_variation(ours, theirs), 4),
    }


def prototype_counts(labels_dir: pathlib.Path, cache: pathlib.Path) -> np.ndarray:
    """Accepted segments per prototype, recomputed from the run's own artifacts.

    Read back from the frame labels instead, adjacent segments that landed in
    the same prototype would have merged into one run and the counts would come
    out low; the embeddings and the published centres are the segments as the
    clusterer actually saw them.
    """
    report = json.loads((labels_dir / "report.json").read_text(encoding="utf-8"))
    producer = np.load(labels_dir / "producer.npz", allow_pickle=True)
    embeddings = np.load(cache, allow_pickle=True)["embeddings"].astype(np.float64)
    if report["clustering"].get("embedding_metric", "").startswith("cosine"):
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    centers, thresholds = producer["centers"], producer["thresholds"]
    distance = np.linalg.norm(embeddings[:, None, :] - centers[None, :, :], axis=2)
    assigned, nearest = distance.argmin(1), distance.min(1)
    accepted = nearest <= thresholds[assigned]
    return np.bincount(assigned[accepted], minlength=len(centers))


def labelled_hours(labels_dir: pathlib.Path) -> Optional[float]:
    """Footage M2 actually saw, which is not always what M1 segmented.

    A video can run past the motion released for it -- AIST's supplement
    truncates a sequence to the music its song has -- so segments beyond the
    motion are dropped here.  Dividing M2's segment count by M1's hours would
    then understate the density by the size of that overhang.
    """
    path = labels_dir / "labels.jsonl"
    if not path.is_file():
        return None
    frames = sum(json.loads(line).get("frame_count", 0) for line in path.open(encoding="utf-8"))
    return frames / FPS / 3600 if frames else None


def m2_report(labels_dir: pathlib.Path, cache: pathlib.Path,
              hours: Optional[float]) -> Dict[str, object]:
    counts = prototype_counts(labels_dir, cache)
    hours = labelled_hours(labels_dir) or hours
    report = json.loads((labels_dir / "report.json").read_text(encoding="utf-8"))
    mean = float(counts.mean())
    relative = counts / mean
    # Both histograms in units of their own corpus mean, so a vocabulary half
    # as dense is still comparable on shape.
    rel_edges = [(low / PAPER_SAMPLES_PER_PROTOTYPE, high / PAPER_SAMPLES_PER_PROTOTYPE, name)
                 for low, high, name in SIZE_EDGES]
    ours = shares(relative, rel_edges)
    theirs = normalise(PAPER_FIG4B)
    out = {
        "metric": report["clustering"].get("embedding_metric"),
        "prototypes": int(len(counts)),
        "accepted_segments": int(counts.sum()),
        "acceptance_rate": report["clustering"].get("acceptance_rate"),
        "samples_per_prototype": round(mean, 2),
        "paper_samples_per_prototype": PAPER_SAMPLES_PER_PROTOTYPE,
        "dispersion_sd_over_mean": round(float(counts.std() / mean), 3),
        "paper_dispersion_sd_over_mean": 0.207,
        "paper_dispersion_note": "from Fig. 4b bucket midpoints; open buckets make it a floor",
        "smallest": int(counts.min()), "largest": int(counts.max()),
        "shape_histogram": {k: round(v, 4) for k, v in ours.items()},
        "paper_shape_histogram": {k: round(v, 4) for k, v in theirs.items()},
        "shape_total_variation_from_fig4b": round(total_variation(ours, theirs), 4),
    }
    if hours:
        out["labelled_hours"] = round(hours, 4)
        out["accepted_per_hour"] = round(counts.sum() / hours, 1)
        out["paper_accepted_per_hour"] = round(100 * PAPER_SAMPLES_PER_PROTOTYPE / PAPER_HOURS, 1)
    return out


def m3_report(sub_labels_dir: pathlib.Path, m2: Optional[Dict[str, object]],
              hours: Optional[float]) -> Dict[str, object]:
    """M3 sub-prototypes vs Fig. 4c, on the same rate-and-shape terms as M2.

    The sizes come from ``producer.npz`` rather than from the frame labels, for
    the reason ``prototype_counts`` gives one level up: adjacent segments that
    land in the same sub-prototype merge into a single run in the labels, and
    counting runs would undercount every sub-prototype by an amount that grows
    with how coherent the clustering is -- so the better the result, the worse
    the measurement.

    Two comparisons, and only one of them is a like-for-like:

    * **sub-prototypes per prototype** is a ratio, so it is directly comparable
      to the paper's 7.3 and is the headline number here;
    * **samples per sub-prototype** is not, because it inherits M2's density.
      This corpus accepts 13,686 segments where Fig. 4c implies 23,214, so a
      gap in this number is mostly M1/M2's gap arriving again.  It is reported
      with that ratio beside it rather than on its own.
    """
    report = json.loads((sub_labels_dir / "report.json").read_text(encoding="utf-8"))
    producer = np.load(sub_labels_dir / "producer.npz", allow_pickle=True)
    if "subprototype_sizes" not in producer:
        raise SystemExit(
            "error: {}/producer.npz predates subprototype_sizes; re-run "
            "tools/recluster_atomics_ingroup.py to publish it".format(sub_labels_dir))
    counts = np.asarray(producer["subprototype_sizes"], dtype=np.float64)
    counts = counts[counts > 0]
    mean = float(counts.mean())
    rel_edges = [(low / PAPER_SAMPLES_PER_SUBPROTOTYPE, high / PAPER_SAMPLES_PER_SUBPROTOTYPE, name)
                 for low, high, name in SUB_SIZE_EDGES]
    ours = shares(counts / mean, rel_edges)
    theirs = normalise(PAPER_FIG4C)
    recluster = report.get("recluster", {})
    prototypes = int(recluster.get("prototypes") or 0)
    out = {
        "method": recluster.get("method"),
        "paper_row": recluster.get("paper_row"),
        "genre_presplit": recluster.get("genre_presplit"),
        "subprototypes": int(len(counts)),
        "paper_subprototypes": int(sum(PAPER_FIG4C.values())),
        "subprototypes_per_prototype": round(len(counts) / prototypes, 2) if prototypes else None,
        "paper_subprototypes_per_prototype": PAPER_SUBPROTOTYPES_PER_PROTOTYPE,
        "samples_per_subprototype": round(mean, 2),
        "paper_samples_per_subprototype": PAPER_SAMPLES_PER_SUBPROTOTYPE,
        "clustered_segments": int(counts.sum()),
        "dispersion_sd_over_mean": round(float(counts.std() / mean), 3),
        "smallest": int(counts.min()), "largest": int(counts.max()),
        "shape_histogram": {k: round(v, 4) for k, v in ours.items()},
        "paper_shape_histogram": {k: round(v, 4) for k, v in theirs.items()},
        "shape_total_variation_from_fig4c": round(total_variation(ours, theirs), 4),
    }
    # Name where the density gap comes from, so it is not read as an M3 defect.
    if m2 and m2.get("accepted_segments"):
        out["m2_accepted_segments"] = int(m2["accepted_segments"])
        out["retained_from_m2"] = round(float(counts.sum()) / float(m2["accepted_segments"]), 4)
    if hours:
        out["subprototypes_per_hour"] = round(len(counts) / hours, 1)
        out["paper_subprototypes_per_hour"] = round(
            100 * PAPER_SUBPROTOTYPES_PER_PROTOTYPE / PAPER_HOURS, 1)
    return out


def render(report: Dict[str, object]) -> str:
    lines: List[str] = []
    m1 = report.get("m1")
    if m1:
        lines.append("M1 segmentation vs Fig. 4a")
        lines.append("  corpus      {} sequences, {} h   (paper: {} sequences, {} h)".format(
            m1["sequences"], m1["hours"], PAPER_SEQUENCES, PAPER_HOURS))
        lines.append("  segments    {}  =  {}/h   (paper: {}/h)".format(
            m1["segments"], m1["segments_per_hour"], m1["paper_segments_per_hour"]))
        lines.append("  duration    mean {}s  median {}s   (Fig. 4a midpoints: {}s)".format(
            m1["mean_seconds"], m1["median_seconds"], m1["paper_mean_seconds_at_midpoints"]))
        lines.append("  {:10s} {:>8s} {:>8s}".format("bucket", "ours", "paper"))
        for _, _, name in DURATION_EDGES:
            lines.append("  {:10s} {:7.2f}% {:7.2f}%".format(
                name, m1["histogram"][name] * 100, m1["paper_histogram"][name] * 100))
        lines.append("  total variation from Fig. 4a = {}".format(
            m1["total_variation_from_fig4a"]))
    m2 = report.get("m2")
    if m2:
        lines.append("")
        lines.append("M2 clustering vs Fig. 4b  [{}]".format(m2["metric"]))
        lines.append("  accepted    {} segments ({} of them)".format(
            m2["accepted_segments"], m2["acceptance_rate"]))
        lines.append("  per class   {}   (paper: {})".format(
            m2["samples_per_prototype"], m2["paper_samples_per_prototype"]))
        if "accepted_per_hour" in m2:
            lines.append("  per hour    {}   (paper: {})".format(
                m2["accepted_per_hour"], m2["paper_accepted_per_hour"]))
        lines.append("  spread      sd/mean {}   (paper: {}, a floor)".format(
            m2["dispersion_sd_over_mean"], m2["paper_dispersion_sd_over_mean"]))
        lines.append("  {:10s} {:>8s} {:>8s}   (buckets in units of each corpus's mean)".format(
            "bucket", "ours", "paper"))
        for _, _, name in SIZE_EDGES:
            lines.append("  {:10s} {:7.2f}% {:7.2f}%".format(
                name, m2["shape_histogram"][name] * 100, m2["paper_shape_histogram"][name] * 100))
        lines.append("  shape total variation from Fig. 4b = {}".format(
            m2["shape_total_variation_from_fig4b"]))
    m3 = report.get("m3")
    if m3:
        lines.append("")
        lines.append("M3 sub-prototypes vs Fig. 4c  [{}]".format(m3["method"]))
        lines.append("  row         {}".format(m3["paper_row"]))
        lines.append("  per class   {} sub-prototypes   (paper: {})".format(
            m3["subprototypes_per_prototype"], m3["paper_subprototypes_per_prototype"]))
        lines.append("  total       {}   (paper: {})".format(
            m3["subprototypes"], m3["paper_subprototypes"]))
        lines.append("  per sub     {} segments   (paper: {})".format(
            m3["samples_per_subprototype"], m3["paper_samples_per_subprototype"]))
        if "retained_from_m2" in m3:
            lines.append("  of which    {} clustered here from M2's {} ({:.1%}) --"
                         " the per-sub gap is mostly this".format(
                             m3["clustered_segments"], m3["m2_accepted_segments"],
                             m3["retained_from_m2"]))
        lines.append("  spread      sd/mean {}   (smallest {}, largest {})".format(
            m3["dispersion_sd_over_mean"], m3["smallest"], m3["largest"]))
        lines.append("  {:10s} {:>8s} {:>8s}   (buckets in units of each corpus's mean)".format(
            "bucket", "ours", "paper"))
        for _, _, name in SUB_SIZE_EDGES:
            lines.append("  {:10s} {:7.2f}% {:7.2f}%".format(
                name, m3["shape_histogram"][name] * 100, m3["paper_shape_histogram"][name] * 100))
        lines.append("  shape total variation from Fig. 4c = {}".format(
            m3["shape_total_variation_from_fig4c"]))
    consistency = report["paper_self_consistency"]
    lines.append("")
    lines.append("the paper's own ledger")
    lines.append("  Fig. 4a needs {} s of footage at its bucket floors; the corpus is {} s"
                 " ({:+.1%})".format(consistency["fig4a_seconds_at_bucket_floors"],
                                     int(consistency["corpus_seconds"]),
                                     consistency["fig4a_exceeds_corpus_by"]))
    lines.append("  Fig. 4b implies {:.0f} clustered segments from Fig. 4a's {} ({:.0%}) --"
                 " clustering only discards".format(consistency["fig4b_implied_segments"],
                                                    consistency["fig4a_segments"],
                                                    consistency["fig4b_over_fig4a"]))
    lines.append("  Fig. 4c implies {:.0f} ({:.0%}), which is Fig. 4a".format(
        consistency["fig4c_implied_segments"], consistency["fig4c_over_fig4a"]))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--segmentation", type=pathlib.Path, default=None,
                        help="M1 output; scored against Fig. 4a")
    parser.add_argument("--labels", type=pathlib.Path, default=None,
                        help="M2 output directory; scored against Fig. 4b")
    parser.add_argument("--embedding-cache", type=pathlib.Path, default=None,
                        help="the TMR embeddings that M2 clustered")
    parser.add_argument("--sub-labels", type=pathlib.Path, default=None,
                        help="M3 output directory from tools/recluster_atomics_ingroup.py; "
                             "scored against Fig. 4c")
    parser.add_argument("--output", type=pathlib.Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report: Dict[str, object] = {"paper_self_consistency": paper_self_consistency()}
    hours = None
    if args.segmentation:
        report["m1"] = m1_report(args.segmentation)
        report["m1"]["source"] = str(args.segmentation)
        hours = report["m1"]["hours"]
    if args.labels:
        if not args.embedding_cache:
            raise SystemExit("error: --labels needs --embedding-cache, the embeddings "
                             "M2 clustered; recomputing them here could silently use a "
                             "different segmentation")
        report["m2"] = m2_report(args.labels, args.embedding_cache, hours)
        report["m2"]["source"] = str(args.labels)
    if args.sub_labels:
        report["m3"] = m3_report(args.sub_labels, report.get("m2"), hours)
        report["m3"]["source"] = str(args.sub_labels)
    print(render(report))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
