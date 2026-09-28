"""Merge a discovered atomic vocabulary into coarser ones, for a granularity sweep.

The released ``kinematic_atomic_100_v2`` vocabulary is not predictable from
music on held-out songs (see worklog.md).  Two explanations survive that
measurement and they call for different fixes:

1. **Granularity.** 100 classes over 838 training sequences may simply be too
   fine to be learnable, even if the underlying descriptor space does carry
   music-related structure.  Coarser groupings would then be predictable.
2. **The descriptor space.** The clustering used heading-canonicalised
   kinematics with music never in the loop, so cluster identity may be
   music-orthogonal at *every* granularity.  Coarsening would change nothing.

This tool produces the label maps needed to tell those apart.  It reads the
frozen K-Means centres from a discovery ``producer.npz`` and merges them with
Ward agglomerative clustering, weighted by each cluster's support so that a
tiny cluster cannot drag a large one around.  Merging the published centres,
rather than re-running discovery, keeps the sweep an honest coarsening of the
*same* label space: every coarse label is a union of original labels.

Label 0 is the transition token and is never merged into an atomic group; it
stays 0 in every output map.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist


def load_centers(producer_path):
    """Return (centers, support) from a frozen discovery producer artifact."""
    with np.load(str(producer_path), allow_pickle=True) as bundle:
        if "centers" not in bundle.files:
            raise ValueError("{} has no 'centers' array".format(producer_path))
        centers = np.asarray(bundle["centers"], dtype=np.float64)
        support = (
            np.asarray(bundle["cluster_support"], dtype=np.float64)
            if "cluster_support" in bundle.files
            else np.ones(len(centers))
        )
    if centers.ndim != 2:
        raise ValueError("centers must be [K, D], got {}".format(centers.shape))
    if len(support) != len(centers):
        raise ValueError("cluster_support and centers disagree on K")
    return centers, support


def merge_to(centers, support, groups):
    """Ward-merge ``centers`` into ``groups`` clusters; return a 0-based mapping.

    Support weights enter as sample repetition in the linkage input, which is
    what makes a 3000-frame cluster count more than a 300-frame one.
    """
    if groups < 1 or groups > len(centers):
        raise ValueError("groups must be in [1, {}]".format(len(centers)))
    if groups == len(centers):
        return np.arange(len(centers))

    # Ward needs euclidean geometry; the discovery embedding is already
    # standardised, so raw distances are meaningful.
    weights = np.sqrt(support / support.mean()).reshape(-1, 1)
    weighted = centers * weights
    tree = linkage(pdist(weighted, metric="euclidean"), method="ward")
    assignment = fcluster(tree, t=groups, criterion="maxclust")
    # fcluster is 1-based and may return fewer than requested; compact it.
    unique = {value: index for index, value in enumerate(sorted(set(assignment)))}
    return np.array([unique[value] for value in assignment])


def build_label_map(assignment):
    """Original label -> coarse label, with 0 reserved for transition.

    Original labels are 1..K (0 is transition), so entry i of the returned list
    is the coarse label for original label i.
    """
    return [0] + [int(value) + 1 for value in assignment]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer", type=Path, required=True,
                        help="discovery producer.npz holding the frozen centres")
    parser.add_argument("--groups", type=int, nargs="+", required=True,
                        help="coarse class counts to emit, e.g. --groups 5 10 25 50")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    centers, support = load_centers(args.producer)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for groups in sorted(set(args.groups)):
        assignment = merge_to(centers, support, groups)
        label_map = build_label_map(assignment)
        realised = len(set(assignment))
        sizes = np.bincount(assignment, minlength=realised)
        merged_support = np.bincount(assignment, weights=support, minlength=realised)

        path = args.output_dir / "label_map_k{}.json".format(groups)
        payload = {
            "producer": str(args.producer),
            "source_classes": len(centers),
            "requested_groups": groups,
            "realised_groups": realised,
            # index i holds the coarse label for original label i; 0 -> 0.
            "label_map": label_map,
            "group_sizes": sizes.tolist(),
            "group_frame_support": merged_support.tolist(),
            "method": "ward agglomerative on support-weighted discovery centres",
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        summary.append(
            {
                "groups": groups,
                "realised": realised,
                "path": str(path),
                "largest_group_classes": int(sizes.max()),
                "smallest_group_classes": int(sizes.min()),
            }
        )
        print(
            "k={:3d} -> {:3d} groups | classes per group {}..{} | wrote {}".format(
                groups, realised, int(sizes.min()), int(sizes.max()), path.name
            )
        )

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
