#!/usr/bin/env bash
#
# preinstall.sh — make a freshly created pod runnable again.
#
# Run this first on any new pod, before launching a stage:
#
#     bash preinstall.sh
#
# ---------------------------------------------------------------------------
# Why this script exists
#
# 2026-08-13: stage B was relaunched on a pod created that morning and failed
# on *every* clip with `exit=1` and no error text, because the shard captures
# the child's output and reports only its last `EXTRACT_FAIL` line -- and this
# failure produced none.  Run by hand, the child said:
#
#     ModuleNotFoundError: No module named 'hydra_zen'
#
# Twelve shards had already written ~480 `.extract_failed` markers to OSS by
# then, and those markers are not removable with the credentials this project
# has.  The cost of *not* having this script is therefore not "a stage errors
# out": it is a poisoned resume set that has to be worked around with
# `--retry-failed` forever after.
#
# ---------------------------------------------------------------------------
# What survives a pod replacement, and what does not
#
# Survives -- these are on network storage and need no action:
#   /workspace  (NAS)   the checkout, .venv_* under it, third_party/ shims
#   /cache      (CPFS)  third_party/GVHMR + its checkpoints, DPVO's compiled
#                       .so files, third_party/QwenVL, runs/, parked data/
#
# Does not survive -- the container filesystem is rebuilt:
#   * pip packages installed into the image's site-packages;
#   * everything under /root, including ~/.ossutilconfig.
#
# That second one is the subtle one.  `tools/asset_io.py` reads and writes OSS
# through the SDK using the pod's STS token, but it *lists* through the ossutil
# binary -- deliberately, because this is a CPFS-OSS bridge bucket that answers
# a deep LIST with a 502 while ossutil returns 113,921 objects in five seconds.
# The subprocess does not inherit the SDK's credentials; it reads
# ~/.ossutilconfig.  So without this file every listing fails, which means the
# frozen todo list cannot be computed, which means nothing runs at all.
#
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO="$(pwd)"
PYPI="${ATOMICDANCE_PYPI_INDEX:-https://pypi.org/simple}"
failures=0

note() { printf '\n== %s ==\n' "$*"; }
bad()  { printf 'FAIL %s\n' "$*"; failures=$((failures + 1)); }

# ---------------------------------------------------------------------------
note "1/4  OSS credentials for the ossutil subprocess"

# Not copied blindly: the file carries a long-lived key, so the source is
# explicit and the destination is 600.  Override with OSSUTIL_CONFIG_SOURCE.
OSSUTIL_CONFIG_SOURCE="${OSSUTIL_CONFIG_SOURCE:-$REPO/../Lodge/.secrets/ossutilconfig}"
if [ -s /root/.ossutilconfig ]; then
  echo "   /root/.ossutilconfig already present"
elif [ -s "$OSSUTIL_CONFIG_SOURCE" ]; then
  install -m 600 "$OSSUTIL_CONFIG_SOURCE" /root/.ossutilconfig
  echo "   restored /root/.ossutilconfig from $OSSUTIL_CONFIG_SOURCE (mode 600)"
else
  bad "no ossutil config: neither /root/.ossutilconfig nor $OSSUTIL_CONFIG_SOURCE"
  echo "     set OSSUTIL_CONFIG_SOURCE=<path to an ossutilconfig> and re-run."
fi

# ---------------------------------------------------------------------------
note "2/4  pip packages the image does not carry"

# --no-deps throughout: every one of these would otherwise pull its own torch
# and break the CUDA 12.9 / Blackwell build the rest of the repo runs on.
#
#   ultralytics smplx hydra_zen pycolmap -> GVHMR imports
#   pypose                               -> DPVO runtime
#
# pypose is easy to miss because it belongs to tools/setup_dpvo_env.sh rather
# than tools/setup_gvhmr_env.sh, while the shard runs with --use-dpvo by
# default -- so a pod with only GVHMR's packages still fails on every clip.
#
# The *default* index first, not PyPI.  tools/setup_gvhmr_env.sh hardcodes
# `--index-url https://pypi.org/simple` on the belief that the internal mirror
# lacks these; measured on 2026-08-13 that is no longer true, and it is an
# expensive belief -- pypi.org from this network did not finish in fifteen
# minutes, while the Aliyun mirror served the same wheels in seconds.  PyPI
# stays as the fallback for whatever the mirror really is missing.
PACKAGES="ultralytics smplx hydra_zen pycolmap pypose"

still_missing() {
  local out=""
  for pkg in $1; do
    python3 -c "import importlib,sys; sys.exit(0 if importlib.util.find_spec('${pkg//-/_}') else 1)" \
      2>/dev/null || out="$out $pkg"
  done
  echo "$out"
}

missing=$(still_missing "$PACKAGES")
if [ -n "$missing" ]; then
  echo "   from the default index:$missing"
  python3 -m pip install --no-deps -q $missing 2>&1 | grep -vE "WARNING: (Running pip|The repository)" || true
  missing=$(still_missing "$PACKAGES")
  if [ -n "$missing" ]; then
    echo "   falling back to PyPI for:$missing"
    python3 -m pip install --index-url "$PYPI" --no-deps -q $missing \
      2>&1 | grep -vE "WARNING: (Running pip|The repository)" || true
  fi
  missing=$(still_missing "$PACKAGES")
  [ -n "$missing" ] && bad "could not install:$missing"
else
  echo "   all present"
fi

# hydra_colorlog is deliberately not in PACKAGES.  tools/setup_gvhmr_env.sh
# installs it, but the import chain this repo actually exercises does not need
# it -- a full GVHMR extraction (DPVO + preprocess + hmr4d_results.pt) was run
# on 2026-08-13 without it.  It is on neither internal mirror, so requiring it
# would fail every pod for a dependency nothing imports.

# ---------------------------------------------------------------------------
note "3/4  GVHMR source patch (idempotent)"

# hmr4d/utils/body_model/body_model.py opens with `from turtle import forward`,
# an editor auto-import accident: the name is never used, and turtle imports
# tkinter, which a headless Python does not have.  The file lives on /cache so
# the patch normally survives, but a re-cloned checkout would reintroduce it.
BODY_MODEL="$REPO/third_party/GVHMR/hmr4d/utils/body_model/body_model.py"
if [ -f "$BODY_MODEL" ] && head -1 "$BODY_MODEL" | grep -q "^from turtle import forward"; then
  sed -i '1{/^from turtle import forward$/d}' "$BODY_MODEL"
  echo "   patched $BODY_MODEL"
else
  echo "   nothing to patch"
fi

# ---------------------------------------------------------------------------
note "4/4  verify — imports, OSS listing, assets, GPUs"

# GVHMR's root is on the path here because `hmr4d` is only importable from
# inside its checkout -- the shard gets it by running with cwd there, which a
# verification step run from the repo root does not inherit.
PYTHONPATH="$REPO/third_party/pytorch3d_compat:$REPO/third_party/torch_scatter_compat:$REPO/third_party/GVHMR:$REPO" \
python3 - <<'PYTHON' || failures=$((failures + 1))
import importlib, sys
bad = []
for name in ("torch", "hydra_zen", "smplx", "ultralytics", "pypose",
             "pytorch3d.transforms", "hmr4d.configs"):
    try:
        importlib.import_module(name)
        print("   OK   {}".format(name))
    except Exception as error:                              # noqa: BLE001
        bad.append(name)
        print("   FAIL {}: {}: {}".format(name, type(error).__name__, str(error)[:100]))
sys.exit(1 if bad else 0)
PYTHON

# The listing is the check that actually matters: it is the operation that
# needs ~/.ossutilconfig, and the one whose failure stops every stage.
python3 - <<'PYTHON' || failures=$((failures + 1))
import sys
sys.path.insert(0, ".")
try:
    from tools import asset_io
    found = asset_io.list_prefix("data/wild3d/ingest_v1_converted")
    print("   OK   OSS listing: {} objects under ingest_v1_converted".format(len(found)))
except Exception as error:                                  # noqa: BLE001
    print("   FAIL OSS listing: {}: {}".format(type(error).__name__, str(error)[:200]))
    sys.exit(1)
PYTHON

for path in third_party/GVHMR/inputs/checkpoints third_party/TMR/models runs; do
  [ -e "$path" ] && echo "   OK   $path" || bad "missing $path"
done
gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
echo "   OK   ${gpus} GPU(s) visible"

# ---------------------------------------------------------------------------
cat <<'EOF'

Not handled here, because each is expensive and only some stages need it:

  tools/setup_dwpose_env.sh   .venv_ortgpu + ONNX weights, for ingestion only
                              (tools/ingest_wild_uploads.py).  Lives under the
                              checkout, so it survives a pod; absent here
                              because ingestion is finished.
  tools/setup_tmr_env.sh      TMR weights + HumanML3D stats (md5-pinned).
  tools/setup_qwenvl_env.sh   Qwen-VL from the team OSS model store, 16.5 GB.
  tools/setup_dpvo_env.sh     rebuilds DPVO's CUDA extensions.  Only needed if
                              the compiled .so files under
                              third_party/GVHMR/third-party/DPVO/ are gone --
                              they are on /cache and normally are not.

EOF

if [ "$failures" -gt 0 ]; then
  echo "preinstall FAILED: $failures check(s) did not pass"
  exit 1
fi
echo "preinstall OK — this pod can run a stage"
