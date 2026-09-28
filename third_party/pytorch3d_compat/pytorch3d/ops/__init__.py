"""``pytorch3d.ops`` surface: ``knn`` only.

GVHMR imports this as ``import pytorch3d.ops.knn as knn``, so the submodule
must be importable as an attribute of the package.
"""

from . import knn  # noqa: F401
from .knn import KNN, knn_points  # noqa: F401

__all__ = ["knn", "knn_points", "KNN"]
