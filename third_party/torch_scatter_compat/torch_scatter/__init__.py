"""Minimal ``torch_scatter`` stand-in covering only what DPVO's inference needs.

DPVO is the honest upgrade path for wild camera tracking: SimpleVO drops ~45%
of TikTok clips because one untextured frame pair among ~62 keyframes kills the
whole clip.  DPVO's own CUDA kernels build fine here, but it also imports
``torch_scatter``, which has no wheel for this machine's Python 3.12 / Torch 2.8
/ CUDA 12.9 (Blackwell ``sm_120``) environment and would mean a second
from-source CUDA build.

DPVO's inference path touches exactly two symbols, and both are thin
compositions over operations Torch has had natively since 1.x:

* ``scatter_sum``   -- ``Tensor.scatter_add_`` into a zeroed output;
* ``scatter_softmax`` -- a group-wise softmax, computed max-shifted for the
  same numerical stability upstream provides.

Both follow upstream's own reference implementations (``torch_scatter``'s
``scatter_sum`` and its composite ``scatter_softmax``), including the
``broadcast`` index-expansion rule, and are pinned against an explicit
per-group loop in ``tests/test_torch_scatter_compat.py``.

``scatter_max`` exists as a name and raises.  DPVO imports it only through
``loop_closure/long_term.py``, which loads lazily behind
``cfg.CLASSIC_LOOP_CLOSURE`` and is off on the trajectory path GVHMR uses.
Guessing its empty-group fill convention would be exactly the kind of partial
fake of a numerical library that is more dangerous than a missing one.

Put this directory on ``PYTHONPATH`` only when running DPVO.  It is
deliberately not installed, so nothing else can pick it up by accident.
"""

from typing import Optional

import torch

__all__ = ["broadcast", "scatter_sum", "scatter_add", "scatter_softmax", "scatter_max"]


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int) -> torch.Tensor:
    """Expand an index tensor to ``other``'s shape, upstream's rule verbatim."""
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    return src.expand(other.size())


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is not None:
        return out.scatter_add_(dim, index, src)
    size = list(src.size())
    if dim_size is not None:
        size[dim] = dim_size
    elif index.numel() == 0:
        size[dim] = 0
    else:
        size[dim] = int(index.max()) + 1
    out = torch.zeros(size, dtype=src.dtype, device=src.device)
    return out.scatter_add_(dim, index, src)


# Upstream exposes scatter_add as an alias of scatter_sum.
scatter_add = scatter_sum


def scatter_softmax(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """Softmax within each index group, shifted by the group max for stability."""
    if not torch.is_floating_point(src):
        raise ValueError(
            "scatter_softmax expects a floating-point src, got {}".format(src.dtype)
        )
    index_expanded = broadcast(index, src, dim)
    size = list(src.size())
    size[dim] = (
        dim_size
        if dim_size is not None
        else (int(index_expanded.max()) + 1 if index_expanded.numel() else 0)
    )

    # ``amax`` with include_self=False leaves untouched groups at the identity,
    # which for the shift below is harmless: those groups hold no source
    # elements, so nothing gathers from them.
    group_max = torch.zeros(size, dtype=src.dtype, device=src.device).scatter_reduce_(
        dim, index_expanded, src, reduce="amax", include_self=False
    )
    recentered = src - group_max.gather(dim, index_expanded)
    exponentiated = recentered.exp()
    group_sum = scatter_sum(exponentiated, index, dim=dim, dim_size=size[dim])
    return exponentiated / group_sum.gather(dim, index_expanded).clamp(min=1e-12)


def scatter_max(*args, **kwargs):
    raise NotImplementedError(
        "torch_scatter.scatter_max is not shimmed: it is reached only through "
        "DPVO's classic loop closure, which is off on the trajectory path, and "
        "its empty-group fill convention would have to be guessed. Install the "
        "real torch_scatter if you enable CLASSIC_LOOP_CLOSURE."
    )
