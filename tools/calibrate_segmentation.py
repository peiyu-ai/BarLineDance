#!/usr/bin/env python3
"""Set Alg. 1's two free parameters from the one distribution the paper prints.

The paper specifies the segmentation algorithm but not ``N`` (how many clusters
the per-sequence N-means uses) or ``L_min`` (the shortest segment it will keep).
Both were guessed here, and the guess is measurably wrong: with
``--frames-per-cluster 40 --min-length 24`` the wild corpus came out at a mean
of 1.33 s against the paper's 1.01 s, and **no** segment could land in the
paper's ``<0.7 s`` bucket -- 5.9% of Fig. 4a -- because ``L_min`` is 0.8 s and
forbids it.  That is not a property of the corpus, it is the floor we imposed.

Fig. 4a is the only observable the paper gives for this stage, so it is what
the two parameters get fitted to.  Fitting a *free* parameter to a published
distribution is calibration; it would only be curve-fitting if the paper had
specified the value and we moved it to make our numbers look better.

Distance is total variation over the five published buckets: half the sum of
absolute differences in share, which is bounded in [0, 1] and reads directly as
"this fraction of segments sits in the wrong bucket".
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.segment_visual_atomics import cluster_count, segment_sequence  # noqa: E402

FPS = 30.0
PAPER = {"<0.7": 1366, "0.7-0.9": 7712, "0.9-1.1": 8054, "1.1-1.3": 2095, ">1.3": 3967}
EDGES = [(0.0, 0.7, "<0.7"), (0.7, 0.9, "0.7-0.9"), (0.9, 1.1, "0.9-1.1"),
         (1.1, 1.3, "1.1-1.3"), (1.3, np.inf, ">1.3")]


def histogram(durations: Sequence[float]) -> Dict[str, float]:
    durations = np.asarray(durations)
    total = max(1, len(durations))
    return {name: float(((durations >= low) & (durations < high)).sum()) / total
            for low, high, name in EDGES}


def total_variation(ours: Dict[str, float], paper: Dict[str, float]) -> float:
    return 0.5 * sum(abs(ours[name] - paper[name]) for _, _, name in EDGES)


def paper_shares() -> Dict[str, float]:
    total = sum(PAPER.values())
    return {name: PAPER[name] / total for name in PAPER}


def boundary_contrast(array: np.ndarray, boundaries: Sequence[int], *, window: int,
                      rng: np.random.Generator) -> Optional[float]:
    """How much less alike the frames either side of a cut are, versus a random spot.

    This is the guard against fitting the histogram by cheating.  Pushing the
    ``t/T`` weight up far enough turns Alg. 1 into a uniform chopper, which can
    match any duration distribution while cutting in the middle of movements.
    A real cut sits where the visual content changes, so the similarity across
    it should be *lower* than across an arbitrary interior frame.  A setting
    whose contrast collapses toward zero is chopping, whatever its histogram
    says.
    """
    normalised = array / (np.linalg.norm(array, axis=1, keepdims=True) + 1e-12)
    interior = [int(b) for b in boundaries[1:-1]
                if window <= b <= len(array) - window]
    if not interior:
        return None

    def across(position: int) -> float:
        before = normalised[position - window:position].mean(axis=0)
        after = normalised[position:position + window].mean(axis=0)
        return float(before @ after / ((np.linalg.norm(before) * np.linalg.norm(after)) + 1e-12))

    cuts = [across(position) for position in interior]
    pool = [p for p in range(window, len(array) - window)
            if all(abs(p - b) > window for b in interior)]
    if not pool:
        return None
    sampled = rng.choice(pool, size=min(len(cuts), len(pool)), replace=False)
    return float(np.mean([across(int(p)) for p in sampled]) - np.mean(cuts))


def evaluate(features: Sequence[np.ndarray], *, frames_per_cluster: int,
             min_length: int, index_weight: float, seed: int,
             window: int = 5) -> Tuple[List[float], List[float]]:
    durations: List[float] = []
    contrasts: List[float] = []
    rng = np.random.default_rng(seed)
    for array in features:
        count = cluster_count(len(array), 0, frames_per_cluster)
        boundaries, _ = segment_sequence(array, count, min_length, index_weight, seed)
        durations.extend(np.diff(boundaries) / FPS)
        contrast = boundary_contrast(array, boundaries, window=window, rng=rng)
        if contrast is not None:
            contrasts.append(contrast)
    return durations, contrasts


def run(*, features_dir: pathlib.Path, sequences: int, grid: Sequence[Tuple[int, int]],
        index_weight: float, seed: int, output: Optional[pathlib.Path]) -> Dict[str, object]:
    files = sorted(features_dir.glob("*.npz"))
    if not files:
        raise SystemExit("error: no feature files under {}".format(features_dir))
    # An evenly spaced sample rather than the first N: the directory is sorted
    # by upload id, and the earliest uploads are not a random slice of the
    # corpus in either length or content.
    step = max(1, len(files) // sequences)
    chosen = files[::step][:sequences]

    loaded = []
    for path in chosen:
        with np.load(path, allow_pickle=False) as bundle:
            loaded.append(bundle["features"].astype(np.float32))
    print("   loaded {} sequences, {} frames".format(
        len(loaded), sum(len(a) for a in loaded)), flush=True)

    reference = paper_shares()
    results = []
    for frames_per_cluster, min_length in grid:
        durations, contrasts = evaluate(
            loaded, frames_per_cluster=frames_per_cluster, min_length=min_length,
            index_weight=index_weight, seed=seed)
        ours = histogram(durations)
        distance = total_variation(ours, reference)
        contrast = float(np.mean(contrasts)) if contrasts else float("nan")
        results.append({
            "frames_per_cluster": frames_per_cluster,
            "min_length_frames": min_length,
            "min_length_seconds": round(min_length / FPS, 3),
            "segments": len(durations),
            "mean_seconds": round(float(np.mean(durations)), 3),
            "median_seconds": round(float(np.median(durations)), 3),
            "histogram": {name: round(value, 4) for name, value in ours.items()},
            "total_variation_from_paper": round(distance, 4),
            "boundary_contrast": round(contrast, 5),
            "boundary_contrast_note": "similarity across a random interior frame minus "
                                      "similarity across a cut; > 0 means cuts sit on "
                                      "real visual change rather than on a clock",
        })
        print("   fpc={:3d} Lmin={:2d} -> mean {:.3f}s  TV {:.4f}  contrast {:.5f}".format(
            frames_per_cluster, min_length, results[-1]["mean_seconds"], distance,
            contrast), flush=True)

    results.sort(key=lambda row: row["total_variation_from_paper"])
    report = {
        "features_dir": str(features_dir),
        "sequences_sampled": len(loaded),
        "index_weight": index_weight,
        "seed": seed,
        "paper_reference": {"histogram": {k: round(v, 4) for k, v in reference.items()},
                            "mean_seconds": 1.014, "source": "Fig. 4a"},
        "best": results[0],
        "grid": results,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_grid(text: str) -> List[Tuple[int, int]]:
    grid = []
    for item in text.split(","):
        left, _, right = item.strip().partition(":")
        grid.append((int(left), int(right)))
    return grid


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-dir", type=pathlib.Path, required=True)
    parser.add_argument("--sequences", type=int, default=200)
    parser.add_argument("--grid", default="24:12,27:12,30:12,30:15,33:15,36:18,40:24",
                        help="comma separated frames_per_cluster:min_length pairs")
    parser.add_argument("--index-weight", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)
    report = run(features_dir=args.features_dir, sequences=args.sequences,
                 grid=parse_grid(args.grid), index_weight=args.index_weight,
                 seed=args.seed, output=args.output)
    print(json.dumps({"best": report["best"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
