#!/usr/bin/env python3
"""What the genre pre-split alone does to M3's vocabulary, before any clustering.

M3 begins by pre-splitting each prototype by dance genre, and only then groups
within each cell.  So a (prototype, genre) cell is the *floor* of the vocabulary
size: every non-empty cell yields at least one sub-prototype no matter what the
clusterer does.  On 2026-08-13 that floor turned out to be 700 of this repo's
849 sub-prototypes, which raises the question this tool answers:

    is the paper's "7.3 sub-prototypes per prototype" a clustering result, or
    is it the number of genres its prototypes happen to touch?

It matters because the two readings imply opposite actions.  If 7.3 is a
clustering result, it is a target and ``--target-size`` is the knob.  If it is
the genre span, then it is reproduced before M3 runs, tuning toward it is
tuning toward an artefact, and the shape gap has to be explained elsewhere.

The tool therefore reports three things and lets them disagree:

* **the count** -- non-empty cells per prototype, against the paper's 7.3;
* **a null for that count** -- genres permuted across segments, which preserves
  both prototype sizes and the genre marginals.  Without it the count is
  uninterpretable: with ~130 segments per prototype and ten genres, a
  genre-blind clustering would touch essentially all ten, so "7 out of 10" is
  only evidence of genre structure if the null says 10;
* **the shape** -- the cell-size histogram against Fig. 4c, in units of each
  side's own mean, so a corpus with fewer segments per cell is not penalised
  twice.

The falsifiable part is ``hypothesis.low_tail_ratio``.  If sub-prototypes were
cells, the share of them holding less than 20 samples would agree with Fig. 4c's
14.93%.  A ratio far from 1 refutes the identification no matter how well the
counts line up, and it is reported as a ratio rather than a verdict string so
that a later run on a different corpus can move it.

Usage::

    python3 tools/audit_genre_presplit.py \\
        --labels data/atomic_aistpp/aist_v1_labels \\
        --embedding-cache runs/aist_v1_tmr_embeddings.npz \\
        --output runs/aist_v1_presplit_audit.json
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.recluster_atomics_ingroup import genre_of, load_group_keys  # noqa: E402

# Fig. 4c, and the two summary numbers the text gives for it.
PAPER_FIG4C = {"<20": 109, "20-35": 431, "35-50": 99, ">50": 91}
PAPER_SUBPROTOTYPES_PER_PROTOTYPE = 7.3
PAPER_SAMPLES_PER_SUBPROTOTYPE = 31.8
SUB_SIZE_EDGES = [(0.0, 20.0, "<20"), (20.0, 35.0, "20-35"),
                  (35.0, 50.0, "35-50"), (50.0, np.inf, ">50")]
# Bucket midpoints, for the one summary Fig. 4c does not print.  The open top
# bucket makes every moment computed this way a *floor*, which is enough here:
# the question is whether the cells are more dispersed than Fig. 4c can be, and
# a floor answers that in the direction it is asked.
FIG4C_MIDPOINTS = {"<20": 10.0, "20-35": 27.5, "35-50": 42.5, ">50": 65.0}

MIN_SEGMENT_FRAMES = 4                       # the guard M3 applies before clustering


def load_cells(labels_dir: pathlib.Path, cache: pathlib.Path,
               group_keys: Optional[Dict[str, str]] = None
               ) -> Tuple[np.ndarray, np.ndarray]:
    """``(prototype, genre)`` for every segment M3 would re-cluster.

    Segment boundaries come from the embedding cache, not from runs of equal
    frame labels, for the reason recorded in ``recluster_atomics_ingroup``:
    two adjacent segments that M2 put in one prototype are a single run in the
    frame labels, so reading them back merges 32% of them away.  The filters
    here are M3's own, so the population this audits is the population it
    clusters -- an audit of a different set of segments would answer a question
    nobody asked.
    """
    cached = np.load(cache, allow_pickle=True)
    for key in ("recordings", "starts", "ends"):
        if key not in cached:
            raise SystemExit("error: {} has no '{}'; it is not a segment-level "
                             "embedding cache".format(cache, key))
    spans: Dict[str, List[Tuple[int, int]]] = collections.defaultdict(list)
    for name, start, end in zip(cached["recordings"], cached["starts"], cached["ends"]):
        spans[str(name)].append((int(start), int(end)))

    prototypes: List[int] = []
    genres: List[str] = []
    rows = (labels_dir / "labels.jsonl").read_text(encoding="utf-8").splitlines()
    for line in rows:
        if not line.strip():
            continue
        entry = json.loads(line)
        labels = np.load(labels_dir / entry["labels_path"])
        genre = genre_of(entry["recording_id"], group_keys) or "?"
        for start, end in spans.get(entry["recording_id"], ()):
            if end > len(labels) or end - start < MIN_SEGMENT_FRAMES:
                continue
            label = int(labels[start])
            if label > 0:
                prototypes.append(label)
                genres.append(genre)
    if not prototypes:
        raise SystemExit("error: no accepted segments found in {}".format(labels_dir))
    return np.asarray(prototypes), np.asarray(genres)


def shares(values: Sequence[float], mean: float) -> Dict[str, float]:
    """Cell sizes into Fig. 4c's buckets, each side in units of its own mean."""
    relative = np.asarray(values, dtype=np.float64) / mean
    out = {}
    for low, high, name in SUB_SIZE_EDGES:
        lo = low / PAPER_SAMPLES_PER_SUBPROTOTYPE
        hi = high / PAPER_SAMPLES_PER_SUBPROTOTYPE
        out[name] = float(((relative >= lo) & (relative < hi)).mean())
    return out


def permutation_null(prototypes: np.ndarray, genres: np.ndarray, observed: int,
                     repeats: int, seed: int) -> Dict[str, object]:
    """Cells under genre-blind assignment, prototype sizes and marginals kept.

    Permuting the genre column breaks any association between a prototype and a
    genre while leaving both distributions exactly as they are, so the null
    answers "how many cells would a clustering that ignores genre produce on
    this corpus" -- which is the only baseline against which 7 out of 10 means
    anything.
    """
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats, dtype=np.int64)
    pairs = prototypes.tolist()
    for index in range(repeats):
        shuffled = rng.permutation(genres).tolist()
        draws[index] = len(set(zip(pairs, shuffled)))
    n_prototypes = len(set(pairs))
    return {
        "repeats": repeats,
        "cells_mean": round(float(draws.mean()), 1),
        "cells_sd": round(float(draws.std()), 2),
        "cells_per_prototype": round(float(draws.mean()) / n_prototypes, 2),
        # One-sided: the question is whether genre concentrates prototypes, so
        # the interesting tail is fewer cells than chance, not more.
        "p_value_fewer_cells": round(float((draws <= observed).mean()), 4),
        "null": "genre column permuted across segments; prototype sizes and "
                "genre marginals preserved",
    }


def summarise(prototypes: np.ndarray, genres: np.ndarray,
              repeats: int = 200, seed: int = 20260813) -> Dict[str, object]:
    """The whole report from two aligned columns, so it can be tested on made-up
    corpora where the answer is known in advance."""
    cells = collections.Counter(zip(prototypes.tolist(), genres.tolist()))
    sizes = np.asarray(sorted(cells.values()), dtype=np.float64)
    n_prototypes = len(set(prototypes.tolist()))
    per_prototype = len(cells) / n_prototypes

    # How many genres a prototype *effectively* spans: exp(entropy) of its genre
    # distribution.  A prototype that touches ten genres with 90% of its mass in
    # one spans one, and the raw count would not say so -- which is the whole
    # difference between a cell that survives clustering and a cell that is a
    # singleton.
    touched, effective = [], []
    for prototype in sorted(set(prototypes.tolist())):
        counts = np.asarray([count for (proto, _), count in cells.items()
                             if proto == prototype], dtype=np.float64)
        share = counts / counts.sum()
        touched.append(len(counts))
        effective.append(float(np.exp(-(share * np.log(share)).sum())))

    ours = shares(sizes, float(sizes.mean()))
    total = float(sum(PAPER_FIG4C.values()))
    theirs = {name: value / total for name, value in PAPER_FIG4C.items()}
    variation = 0.5 * sum(abs(ours[name] - theirs[name]) for name in theirs)

    weights = np.asarray([PAPER_FIG4C[name] for name in FIG4C_MIDPOINTS], dtype=np.float64)
    points = np.asarray([FIG4C_MIDPOINTS[name] for name in FIG4C_MIDPOINTS], dtype=np.float64)
    paper_mean = float((weights * points).sum() / weights.sum())
    paper_sd = float(np.sqrt((weights * (points - paper_mean) ** 2).sum() / weights.sum()))

    report: Dict[str, object] = {
        "corpus": {
            "segments": int(len(prototypes)),
            "prototypes": n_prototypes,
            "genres": sorted(set(genres.tolist())),
        },
        "cells": {
            "count": len(cells),
            "per_prototype": round(per_prototype, 2),
            "paper_subprototypes_per_prototype": PAPER_SUBPROTOTYPES_PER_PROTOTYPE,
            "relative_gap": round(abs(per_prototype - PAPER_SUBPROTOTYPES_PER_PROTOTYPE)
                                  / PAPER_SUBPROTOTYPES_PER_PROTOTYPE, 4),
            "size_mean": round(float(sizes.mean()), 2),
            "size_median": float(np.median(sizes)),
            "sd_over_mean": round(float(sizes.std() / sizes.mean()), 3),
            "smallest": int(sizes.min()),
            "largest": int(sizes.max()),
            "singletons": int((sizes == 1).sum()),
            "below_paper_mean": int((sizes < PAPER_SAMPLES_PER_SUBPROTOTYPE).sum()),
        },
        "genre_span": {
            "touched_mean": round(float(np.mean(touched)), 2),
            "touched_median": float(np.median(touched)),
            "effective_mean": round(float(np.mean(effective)), 2),
            "effective_median": round(float(np.median(effective)), 2),
            "note": "effective = exp(entropy of the prototype's genre distribution); "
                    "the gap between touched and effective is the tail that "
                    "pre-splitting turns into small cells",
        },
        "null_genre_permutation": permutation_null(
            prototypes, genres, len(cells), repeats, seed),
        "fig4c": {
            "shape_histogram": {name: round(ours[name], 4) for name in ours},
            "paper_shape_histogram": {name: round(theirs[name], 4) for name in theirs},
            "total_variation": round(variation, 4),
            "paper_implied_sd_over_mean": round(paper_sd / paper_mean, 3),
            "paper_implied_sd_note": "from bucket midpoints; the open top bucket "
                                     "makes this a floor",
        },
    }

    # The identification the counts suggest, and the statistic that can refute it.
    low_tail = ours["<20"]
    report["hypothesis"] = {
        "statement": "the paper's sub-prototypes are its (prototype, genre) cells, "
                     "i.e. M3's grouping yields about one sub-prototype per cell",
        "count_ratio": round(per_prototype / PAPER_SUBPROTOTYPES_PER_PROTOTYPE, 3),
        "low_tail_share": round(low_tail, 4),
        "paper_low_tail_share": round(theirs["<20"], 4),
        "low_tail_ratio": round(low_tail / theirs["<20"], 2) if theirs["<20"] else None,
        "count_agrees_within_10pct": bool(
            abs(per_prototype - PAPER_SUBPROTOTYPES_PER_PROTOTYPE)
            <= 0.10 * PAPER_SUBPROTOTYPES_PER_PROTOTYPE),
        "shape_agrees_within_50pct": bool(abs(low_tail / theirs["<20"] - 1.0) <= 0.5)
        if theirs["<20"] else None,
    }

    # What the paper's own two numbers require of the corpus, given these cells.
    segments = float(len(prototypes))
    report["ledger"] = {
        "clusterable_segments": int(segments),
        "segments_per_cell": round(segments / len(cells), 2),
        "paper_samples_per_subprototype": PAPER_SAMPLES_PER_SUBPROTOTYPE,
        "segments_these_cells_would_need_for_31_8":
            int(round(len(cells) * PAPER_SAMPLES_PER_SUBPROTOTYPE)),
        "paper_fig4c_implied_segments":
            int(round(sum(PAPER_FIG4C.values()) * PAPER_SAMPLES_PER_SUBPROTOTYPE)),
        "shortfall_vs_fig4c": round(
            segments / (sum(PAPER_FIG4C.values()) * PAPER_SAMPLES_PER_SUBPROTOTYPE), 4),
        "note": "with about one sub-prototype per cell, samples-per-sub is fixed by "
                "the segment count -- it is an M1/M2 quantity arriving in M3, not a "
                "knob M3 owns",
    }
    return report


def audit(labels_dir: pathlib.Path, cache: pathlib.Path, repeats: int, seed: int,
          group_keys: Optional[Dict[str, str]] = None) -> Dict[str, object]:
    prototypes, genres = load_cells(labels_dir, cache, group_keys)
    return summarise(prototypes, genres, repeats, seed)


def render(report: Dict[str, object]) -> str:
    cells = report["cells"]
    null = report["null_genre_permutation"]
    fig4c = report["fig4c"]
    span = report["genre_span"]
    hypothesis = report["hypothesis"]
    ledger = report["ledger"]
    lines = [
        "genre pre-split, {} segments in {} prototypes over {} genres".format(
            report["corpus"]["segments"], report["corpus"]["prototypes"],
            len(report["corpus"]["genres"])),
        "",
        "the count",
        "  non-empty cells      {}  ->  {} per prototype   (paper: {})".format(
            cells["count"], cells["per_prototype"],
            cells["paper_subprototypes_per_prototype"]),
        "  genre-blind null     {} +/- {}  ->  {} per prototype  (p = {})".format(
            null["cells_mean"], null["cells_sd"], null["cells_per_prototype"],
            null["p_value_fewer_cells"]),
        "  genres per prototype {} touched, {} effective".format(
            span["touched_mean"], span["effective_mean"]),
        "",
        "the shape",
        "  cell size            mean {}, median {}, sd/mean {}  ({} singletons)".format(
            cells["size_mean"], cells["size_median"], cells["sd_over_mean"],
            cells["singletons"]),
        "  Fig. 4c implies      sd/mean >= {}".format(fig4c["paper_implied_sd_over_mean"]),
        "  {:10s} {:>8s} {:>8s}   (buckets in units of each side's own mean)".format(
            "bucket", "cells", "paper"),
    ]
    for _, _, name in SUB_SIZE_EDGES:
        lines.append("  {:10s} {:7.2f}% {:7.2f}%".format(
            name, fig4c["shape_histogram"][name] * 100,
            fig4c["paper_shape_histogram"][name] * 100))
    lines += [
        "  total variation      {}".format(fig4c["total_variation"]),
        "",
        "hypothesis: {}".format(hypothesis["statement"]),
        "  count      ratio {}   -> {}".format(
            hypothesis["count_ratio"],
            "agrees" if hypothesis["count_agrees_within_10pct"] else "disagrees"),
        "  low tail   ratio {}   -> {}".format(
            hypothesis["low_tail_ratio"],
            "agrees" if hypothesis["shape_agrees_within_50pct"] else "disagrees"),
        "",
        "the ledger",
        "  {} clusterable segments over {} cells = {} each (paper: {})".format(
            ledger["clusterable_segments"], cells["count"],
            ledger["segments_per_cell"], ledger["paper_samples_per_subprototype"]),
        "  these cells would need {} segments to average {}".format(
            ledger["segments_these_cells_would_need_for_31_8"],
            ledger["paper_samples_per_subprototype"]),
        "  Fig. 4c implies {} segments; this corpus has {:.1%} of them".format(
            ledger["paper_fig4c_implied_segments"], ledger["shortfall_vs_fig4c"]),
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=pathlib.Path, required=True,
                        help="M2 label bundle, the input M3 pre-splits")
    parser.add_argument("--embedding-cache", type=pathlib.Path, required=True,
                        help="the segment-level cache M2 clustered; supplies the "
                             "segment boundaries M3 uses")
    parser.add_argument("--group-keys", type=pathlib.Path, default=None,
                        help="JSON map to a pre-split key other than AIST's genre "
                             "field, e.g. the wild corpus's choreographer account")
    parser.add_argument("--repeats", type=int, default=200,
                        help="permutations for the genre-blind null")
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit(args.labels, args.embedding_cache, args.repeats, args.seed,
                   load_group_keys(args.group_keys))
    report["input"] = {"labels": str(args.labels), "embedding_cache": str(args.embedding_cache),
                       "group_keys": str(args.group_keys) if args.group_keys else None,
                       "presplit_key": "metadata_group_keys" if args.group_keys
                                       else "aist_sequence_id"}
    print(render(report))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
        print("\nwrote {}".format(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
