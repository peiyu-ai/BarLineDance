"""Minimal ``pytorch3d`` stand-in covering only what GVHMR's inference needs.

GVHMR is the chosen world-HMR backend for the wild-3D corpus (see worklog.md).
Its inference path imports ``pytorch3d``, which has no wheel for this machine's
Python 3.12 / Torch 2.8 / CUDA 12.9 (Blackwell ``sm_120``) environment and is
expensive to build from source.  GVHMR touches only thirteen pytorch3d symbols,
so shimming is far cheaper than compiling.

Put this directory **before** site-packages on ``PYTHONPATH`` only when running
GVHMR.  It is deliberately not installed, so nothing else in the repo can pick
it up by accident and mistake it for the real library.

Coverage:

* ``pytorch3d.transforms`` -- complete for GVHMR, backed by
  ``dataset.rotation_ops``, which is pinned against ``scipy`` to 1e-12.
* ``pytorch3d.ops.knn`` -- ``knn_points`` only, brute force.
* ``pytorch3d.structures`` / ``pytorch3d.renderer`` -- import-only stubs that
  raise when used.  GVHMR needs them importable because ``tools/demo/demo.py``
  imports its renderer at module scope, but rendering happens *after* the
  predictions are written, so extraction never calls them.  They raise instead
  of silently returning wrong geometry.

Anything outside that list is absent on purpose: a partial fake of a numerical
library is more dangerous than a missing one.
"""

import sys
from pathlib import Path

__version__ = "0.0.0+atomicdance-compat"

# This file lives at <repo>/third_party/pytorch3d_compat/pytorch3d/__init__.py,
# so the repo root -- which owns dataset/rotation_ops.py -- is three levels up.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from . import ops, renderer, structures, transforms  # noqa: E402,F401

__all__ = ["ops", "renderer", "structures", "transforms"]
