#!/usr/bin/env bash
#
# Build DPVO's CUDA extensions so GVHMR can track the camera with it.
#
# DPVO replaces SimpleVO, which discarded 45% of the wild corpus by dying on
# the first frame pair without matchable texture (worklog.md).  Upstream DPVO
# targets Python 3.10 / Torch 2.0; this machine is Python 3.12 / Torch 2.8 /
# CUDA 12.9 on Blackwell (sm_120), so the extensions have to be rebuilt, and
# two upstream sources need edits before they compile.
#
# third_party/GVHMR is git-ignored (a runtime dependency, not project source),
# so these fixes have to live here to be reproducible.
#
# The runtime also needs `pypose` and a `torch_scatter`.  pypose installs from
# PyPI; torch_scatter has no wheel for this environment and would be a second
# from-source CUDA build, so this repo ships third_party/torch_scatter_compat
# instead -- see its docstring for exactly what is and is not covered.
#
# Usage:
#   bash tools/setup_dpvo_env.sh [DPVO_SOURCE]
#
# DPVO_SOURCE defaults to an existing checkout next to this repo; pass a path
# to a fresh `git clone https://github.com/princeton-vl/DPVO` otherwise.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DPVO_SOURCE="${1:-$(dirname "${REPO_ROOT}")/DPVO}"
DPVO_DEST="${REPO_ROOT}/third_party/GVHMR/third-party/DPVO"
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

# The card that will RUN this decides the architecture -- not a constant, and
# not the ambient environment.  Both of the other two have already been wrong
# here: the constant was 12.0 (Blackwell) and the fleet is now L20 (sm_89), and
# this container exports TORCH_CUDA_ARCH_LIST=5.2, which on 2026-09-02 built a
# cuda_corr the L20 could not load -- surfacing much later and far away as
# "undefined symbol: _ZN3c105ErrorC2E...", i.e. not as an architecture error at
# all.  Asking torch for the device's own capability is the only one of the
# three that cannot silently go stale.  ATOMICDANCE_DPVO_ARCH overrides it, for
# building on a host whose GPU is not the one that will run.
ARCH="${ATOMICDANCE_DPVO_ARCH:-}"
if [[ -z "${ARCH}" ]]; then
  ARCH="$(python3 - <<'PYARCH'
import torch
if torch.cuda.is_available():
    print("{}.{}".format(*torch.cuda.get_device_capability(0)))
PYARCH
)"
fi
if [[ -z "${ARCH}" ]]; then
  echo "error: no CUDA device visible, so the target architecture cannot be" >&2
  echo "       derived.  Set ATOMICDANCE_DPVO_ARCH=<major>.<minor> to build" >&2
  echo "       for a card this host cannot see." >&2
  exit 1
fi

if [[ ! -d "${DPVO_SOURCE}/dpvo" ]]; then
  echo "error: no DPVO checkout at ${DPVO_SOURCE}" >&2
  echo "  git clone https://github.com/princeton-vl/DPVO ${DPVO_SOURCE}" >&2
  exit 1
fi

echo "== vendor DPVO source into GVHMR's expected slot =="
mkdir -p "${DPVO_DEST}/thirdparty"
cp -r "${DPVO_SOURCE}/dpvo" "${DPVO_SOURCE}/setup.py" "${DPVO_SOURCE}/config" "${DPVO_DEST}/"
if [[ -d "${DPVO_SOURCE}/thirdparty/eigen-3.4.0" ]]; then
  cp -r "${DPVO_SOURCE}/thirdparty/eigen-3.4.0" "${DPVO_DEST}/thirdparty/"
elif [[ ! -d "${DPVO_DEST}/thirdparty/eigen-3.4.0" ]]; then
  echo "error: setup.py needs thirdparty/eigen-3.4.0; unzip DPVO's eigen-3.4.0.zip there" >&2
  exit 1
fi
find "${DPVO_DEST}" -name "__pycache__" -type d -prune -exec rm -rf {} +

echo "== patch the one ATen API drift that blocks compilation =="
# Every one of the 38 compile errors is the same thing: `Tensor.type()` returns
# a DeprecatedTypeProperties, and AT_DISPATCH_* has wanted a ScalarType for
# several releases.  `a.device().type()` is a different call and is left alone,
# which is why the substitution is anchored on DISPATCH lines.
sed -i 's/AT_DISPATCH_FLOATING_TYPES_AND_HALF(\([a-zA-Z_][a-zA-Z0-9_]*\)\.type()/AT_DISPATCH_FLOATING_TYPES_AND_HALF(\1.scalar_type()/g' \
  "${DPVO_DEST}/dpvo/altcorr/correlation_kernel.cu"
sed -i '/DISPATCH/ s/\.type()/.scalar_type()/g' \
  "${DPVO_DEST}/dpvo/lietorch/src/lietorch_gpu.cu" \
  "${DPVO_DEST}/dpvo/lietorch/src/lietorch_cpu.cpp" \
  "${DPVO_DEST}/dpvo/lietorch/src/lietorch.cpp"
# With the call sites passing a ScalarType, lietorch's own dispatch header must
# stop routing it through a helper that expected the deprecated type.
sed -i 's|at::ScalarType _st = ::detail::scalar_type(the_type);|at::ScalarType _st = the_type;|' \
  "${DPVO_DEST}/dpvo/lietorch/include/dispatch.h"

echo "== make MIXED_PRECISION actually govern the update path =="
# Two of the three autocast blocks read cfg.MIXED_PRECISION; the one wrapping
# the BA update is hardcoded True, so setting the flag to False silently left
# that half in fp16.  Behaviour is unchanged at the shipped default (True);
# this only makes the knob mean what it says, which matters the moment anyone
# tries to rule precision in or out as the cause of a divergence.
sed -i 's|^\( *\)with autocast(enabled=True):|\1with autocast(enabled=self.cfg.MIXED_PRECISION):|' \
  "${DPVO_DEST}/dpvo/dpvo.py"

echo "== python dependency: pypose (no-deps, so torch is not replaced) =="
pip_install pypose

echo "== build extensions for sm_${ARCH/./} =="
(cd "${DPVO_DEST}" && TORCH_CUDA_ARCH_LIST="${ARCH}" MAX_JOBS="${MAX_JOBS:-8}" \
  python3 setup.py build_ext --inplace)

echo "== verify =="
# Absolute, because third_party/GVHMR is a symlink into the asset cache on the
# machines that actually run this: `cd` follows it, so a relative
# "../torch_scatter_compat" resolves next to the *link target* and the shim is
# not there.  The import then fails on torch_scatter and the verification never
# reaches the CUDA call -- i.e. the one gate that would have caught the sm_52
# build could not fire.  Same reason run_gvhmr_ingest_shard.sh spells this path
# from $REPO.
(cd "${REPO_ROOT}/third_party/GVHMR" &&
 PYTHONPATH="${REPO_ROOT}/third_party/torch_scatter_compat:${REPO_ROOT}/third_party/GVHMR/third-party/DPVO:${REPO_ROOT}/third_party/GVHMR" python3 - <<'PY'
import torch

import cuda_ba  # noqa: F401
import cuda_corr  # noqa: F401
import lietorch_backends  # noqa: F401
from dpvo.dpvo import DPVO  # noqa: F401
from dpvo.lietorch import SE3

pose = SE3.exp(torch.randn(2, 6, device="cuda"))
assert pose.matrix().shape == (2, 4, 4)
print("DPVO extensions import and run on", torch.cuda.get_device_name(0))
PY
)

cat <<NOTE

Done.  Extraction with DPVO needs the shim and DPVO on PYTHONPATH, spelled
absolutely -- third_party/GVHMR is a symlink on these machines, so relative
entries resolve next to the link target and quietly find nothing:

  cd ${REPO_ROOT}/third_party/GVHMR
  PYTHONPATH=${REPO_ROOT}/third_party/torch_scatter_compat:${REPO_ROOT}/third_party/pytorch3d_compat:third-party/DPVO:. \\
    python ${REPO_ROOT}/tools/run_gvhmr_extract.py --video <clip.mp4> \\
      --output-root <dir> --use-dpvo

or, for the whole corpus, `bash tools/launch_parallel_pipeline.sh`, which uses
DPVO by default.
NOTE
