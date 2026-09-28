#!/usr/bin/env bash
#
# Prepare a GVHMR checkout so its *inference* path imports on this machine.
#
# GVHMR is the chosen world-HMR backend for the wild-3D corpus (worklog.md).
# Upstream targets Python 3.10 / Torch 2.3 / CUDA 12.1; this machine is Python
# 3.12 / Torch 2.8 / CUDA 12.9 on Blackwell (sm_120).  Rather than rebuild that
# stack, this script closes the four gaps that actually block importing.
#
# third_party/GVHMR is git-ignored (it is a runtime dependency, not project
# source), so these fixes have to live here to be reproducible.
#
# It does NOT download weights and does NOT touch licensed SMPL/SMPLX assets.
# Those still require registration at https://smpl-x.is.tue.mpg.de and remain
# the one hard gate on the wild-3D corpus.
#
# Usage:
#   bash tools/setup_gvhmr_env.sh [GVHMR_ROOT]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GVHMR_ROOT="${1:-${REPO_ROOT}/third_party/GVHMR}"
PYPI="${ATOMICDANCE_PYPI_INDEX:-https://pypi.org/simple}"

# The index is a *fallback*, not the default.  Upstream PyPI is reachable from
# this cluster but slow enough to look like a hang: on 2026-08-25 the pinned
# --index-url turned a four-package install into a fifteen-minute stall while a
# 14-shard fleet sat idle, and the local mirror served every one of them in
# seconds.  So: try the configured index first, and only reach for PyPI for the
# packages it does not carry.  ATOMICDANCE_PYPI_INDEX still overrides both.
pip_install() {
  python -m pip install --no-deps -q "$@" && return 0
  echo "   (not on the default index; retrying against ${PYPI})"
  python -m pip install --index-url "${PYPI}" --no-deps -q "$@"
}


if [[ ! -d "${GVHMR_ROOT}/hmr4d" ]]; then
  echo "error: no GVHMR checkout at ${GVHMR_ROOT}" >&2
  echo "  git clone https://github.com/zju3dv/GVHMR ${GVHMR_ROOT}" >&2
  exit 1
fi

echo "== 1/4 pip dependencies =="
# --no-deps keeps these from dragging in a different torch and breaking the
# CUDA 12.9 build that the rest of the repo (and any running training) uses.
# The local pip default index is an internal mirror that lacks several of
# these, so target PyPI explicitly.
for package in ultralytics smplx hydra_zen hydra_colorlog pycolmap; do
  pip_install "${package}"
done

echo "== 2/4 remove GVHMR's stray 'from turtle import forward' =="
# hmr4d/utils/body_model/body_model.py opens with `from turtle import forward`,
# an editor auto-import accident: `forward` is never used as a bare name.  It
# drags in the turtle graphics module, which imports tkinter, which is not part
# of a headless Python.  Deleting the line is strictly safer than installing a
# GUI toolkit to satisfy an unused import.
BODY_MODEL="${GVHMR_ROOT}/hmr4d/utils/body_model/body_model.py"
if head -1 "${BODY_MODEL}" | grep -q "^from turtle import forward"; then
  sed -i '1{/^from turtle import forward$/d}' "${BODY_MODEL}"
  echo "   patched ${BODY_MODEL}"
else
  echo "   already patched"
fi

echo "== 3/4 verify the pytorch3d compat shim =="
COMPAT="${REPO_ROOT}/third_party/pytorch3d_compat"
if [[ ! -f "${COMPAT}/pytorch3d/__init__.py" ]]; then
  echo "error: missing compat shim at ${COMPAT}" >&2
  exit 1
fi
echo "   ${COMPAT}"

echo "== 4/4 import check =="
cd "${GVHMR_ROOT}"
PYTHONPATH="${COMPAT}:${REPO_ROOT}:${GVHMR_ROOT}:${PYTHONPATH:-}" python - <<'PYTHON'
import importlib

modules = [
    "pytorch3d.transforms",
    "pytorch3d.ops.knn",
    "hmr4d.configs",
    "hmr4d.utils.geo_transform",
    "hmr4d.utils.smplx_utils",
    "hmr4d.utils.preproc",
    "hmr4d.utils.vis.renderer",
    "hmr4d.model.gvhmr.gvhmr_pl_demo",
]

failed = []
for name in modules:
    try:
        importlib.import_module(name)
        print("   OK   {}".format(name))
    except Exception as error:  # noqa: BLE001 - report every blocker at once
        failed.append(name)
        print("   FAIL {}: {}: {}".format(name, type(error).__name__, str(error)[:120]))

if failed:
    raise SystemExit("\nstill blocked: {}".format(", ".join(failed)))
print("\nGVHMR inference imports are satisfied.")
PYTHON

cat <<EOF

Run GVHMR with:
  cd ${GVHMR_ROOT}
  PYTHONPATH=${COMPAT}:${REPO_ROOT}:${GVHMR_ROOT} python tools/demo/demo.py ...

Still required before any real run (not handled here):
  * SMPLX body model  -> ${GVHMR_ROOT}/inputs/checkpoints/body_models/
    Registration at https://smpl-x.is.tue.mpg.de.  GVHMR's prediction math does
    not use it, but EnDecoder.__init__ constructs a body model unconditionally.
  * GVHMR network weights -> ${GVHMR_ROOT}/inputs/checkpoints/
  * Rendering is NOT available: the compat shim stubs pytorch3d's mesh types and
    raises if they are used.  Extraction writes predictions before any render
    step, so this does not block the corpus.
EOF
