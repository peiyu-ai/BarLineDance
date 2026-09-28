"""Brute-force ``knn_points``, the only ``pytorch3d.ops`` call GVHMR makes.

``hmr4d/utils/geo_transform.py:296`` uses it as::

    squared_dist, _, _ = knn.knn_points(points[None], query_verts[None],
                                        K=1, return_nn=False)

so the contract that matters is: **squared** euclidean distances, sorted
ascending, plus the corresponding indices.  Point clouds there are a single
frame's vertices, small enough that an exact ``cdist`` beats any tree.
"""

from typing import NamedTuple, Optional

import torch

__all__ = ["knn_points", "KNN"]


class KNN(NamedTuple):
    """Mirror of pytorch3d's ``_KNN`` return type."""

    dists: torch.Tensor
    idx: torch.Tensor
    knn: Optional[torch.Tensor]


def knn_points(
    p1: torch.Tensor,
    p2: torch.Tensor,
    lengths1: Optional[torch.Tensor] = None,
    lengths2: Optional[torch.Tensor] = None,
    norm: int = 2,
    K: int = 1,
    version: int = -1,
    return_nn: bool = False,
    return_sorted: bool = True,
) -> KNN:
    """K nearest neighbours of each point of ``p1`` within ``p2``.

    Args:
        p1: ``[B, P1, D]`` query points.
        p2: ``[B, P2, D]`` reference points.
        K: neighbours per query point.
        return_nn: also gather the neighbour coordinates.

    Returns:
        ``KNN(dists, idx, knn)`` where ``dists`` are **squared** distances for
        ``norm=2``, shaped ``[B, P1, K]``.
    """
    if p1.ndim != 3 or p2.ndim != 3:
        raise ValueError("knn_points expects [B, P, D] tensors")
    if p1.shape[0] != p2.shape[0] or p1.shape[2] != p2.shape[2]:
        raise ValueError("knn_points requires matching batch and feature dims")
    if lengths1 is not None or lengths2 is not None:
        # Ragged clouds would need masking before the top-k; GVHMR never uses
        # them, so refuse rather than silently ignoring the lengths.
        raise NotImplementedError("compat knn_points does not support ragged lengths")
    if norm not in (1, 2):
        raise ValueError("norm must be 1 or 2")

    k_effective = min(K, p2.shape[1])
    if norm == 2:
        # pytorch3d returns *squared* euclidean distances for norm=2.
        distance = torch.cdist(p1, p2, p=2.0) ** 2
    else:
        distance = torch.cdist(p1, p2, p=1.0)

    dists, idx = torch.topk(distance, k_effective, dim=-1, largest=False, sorted=return_sorted)

    neighbours = None
    if return_nn:
        batch = torch.arange(p1.shape[0], device=p1.device)[:, None, None]
        neighbours = p2[batch, idx]  # [B, P1, K, D]

    return KNN(dists=dists, idx=idx, knn=neighbours)
