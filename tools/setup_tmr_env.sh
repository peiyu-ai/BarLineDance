#!/usr/bin/env bash
#
# Vendor the official TMR pretrained weights + HumanML3D feature machinery
# under third_party/TMR (gitignored: runtime dependency, not project source).
#
# The paper's M2 encodes segments with TMR's motion encoder.  An old repo
# comment claimed "no TMR checkpoint is released anywhere"; that was wrong --
# Mathux/TMR ships official weights behind a pinned gdown id.  This script
# makes that retrieval reproducible: weights archive (md5-pinned), the 263-D
# HumanML3D normalization stats, the Guo-features reference implementation
# (prepare/ + src/guofeats of the upstream repo, vendored as guofeats_ref/),
# and the DistilBERT text backbone TMR uses for its text tower.
#
# Everything is verified against the md5s pinned in tools/tmr_runtime.py; a
# rerun on an already-complete checkout is a no-op that just re-verifies.
#
# Usage:
#   bash tools/setup_tmr_env.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMR_ROOT="${REPO_ROOT}/third_party/TMR"
MODEL_DIR="${TMR_ROOT}/models/tmr_humanml3d_guoh3dfeats"
STATS_DIR="${TMR_ROOT}/stats/humanml3d/guoh3dfeats"
GUOFEATS_DIR="${TMR_ROOT}/guofeats_ref"
PYPI="${ATOMICDANCE_PYPI_INDEX:-https://pypi.org/simple}"

# Pinned identities (must match tools/tmr_runtime.py PROVENANCE).
GDOWN_ID="1n6kRb-d2gKsk8EXfFULFIpaUKYcnaYmm"
ARCHIVE_MD5="7b6d8814f9c1ca972f62852ebb6c7a6f"
UPSTREAM="https://raw.githubusercontent.com/Mathux/TMR/6d74688730d15d43b0a755ce2b0e1f2d76138fc1"
declare -A FILE_MD5=(
  ["${MODEL_DIR}/last_weights/motion_encoder.pt"]="14821969f7030f249be7096a281f28c6"
  ["${MODEL_DIR}/last_weights/text_encoder.pt"]="58f989c93843410a7968e0ad4ccc0040"
  ["${MODEL_DIR}/last_weights/motion_decoder.pt"]="99eb5ed9318183c5f0181e2ce1fa1628"
  ["${STATS_DIR}/mean.pt"]="3e33137ef6d80c8547077e7d99e44401"
  ["${STATS_DIR}/std.pt"]="41173c3057fb388e931f9e4ac0d0d2b9"
)

verify_all() {
  local ok=1
  for path in "${!FILE_MD5[@]}"; do
    if [[ ! -f "${path}" ]]; then
      echo "   missing ${path}"
      ok=0
    elif [[ "$(md5sum "${path}" | cut -d' ' -f1)" != "${FILE_MD5[${path}]}" ]]; then
      echo "   MD5 MISMATCH ${path}"
      ok=0
    fi
  done
  [[ ${ok} -eq 1 ]]
}

echo "== 1/4 weights + stats =="
if verify_all; then
  echo "   already vendored and md5-verified"
else
  workdir="$(mktemp -d "${TMR_ROOT}.download.XXXXXX")"
  trap 'rm -rf "${workdir}"' EXIT
  python3 -m gdown --version >/dev/null 2>&1 \
    || python3 -m pip install --index-url "${PYPI}" -q gdown
  ( cd "${workdir}" \
    && python3 -m gdown "https://drive.google.com/uc?id=${GDOWN_ID}" -O tmr_models.tgz )
  actual="$(md5sum "${workdir}/tmr_models.tgz" | cut -d' ' -f1)"
  if [[ "${actual}" != "${ARCHIVE_MD5}" ]]; then
    echo "error: archive md5 ${actual} != pinned ${ARCHIVE_MD5}" >&2
    exit 1
  fi
  mkdir -p "${TMR_ROOT}/models"
  tar -xzf "${workdir}/tmr_models.tgz" -C "${workdir}"
  # Archive layout: models/tmr_humanml3d_guoh3dfeats + models/tmr_kitml_guoh3dfeats
  rm -rf "${MODEL_DIR}"
  mv "${workdir}/models/tmr_humanml3d_guoh3dfeats" "${MODEL_DIR}"
  [[ -d "${workdir}/models/tmr_kitml_guoh3dfeats" ]] \
    && mv "${workdir}/models/tmr_kitml_guoh3dfeats" "${TMR_ROOT}/models/" || true
  # Stats are checked into the upstream repo, not the archive.
  mkdir -p "${STATS_DIR}"
  for name in mean.pt std.pt; do
    curl -sfL --retry 3 "${UPSTREAM}/stats/humanml3d/guoh3dfeats/${name}" \
      -o "${STATS_DIR}/${name}"
  done
  verify_all || { echo "error: verification failed after download" >&2; exit 1; }
  echo "   downloaded and verified"
fi

echo "== 2/4 guofeats reference implementation =="
# Vendored from upstream prepare/ + src/guofeats/, importable as the package
# guofeats_ref (sys.path must include third_party/TMR).  Only fetched when
# absent: the vendored copy may carry local fixes and is the source of truth.
if [[ -f "${GUOFEATS_DIR}/motion_representation.py" ]]; then
  echo "   already vendored"
else
  mkdir -p "${GUOFEATS_DIR}/common"
  declare -A FETCH=(
    ["${GUOFEATS_DIR}/motion_representation.py"]="src/guofeats/motion_representation.py"
    ["${GUOFEATS_DIR}/paramUtil.py"]="src/guofeats/paramUtil.py"
    ["${GUOFEATS_DIR}/common/quaternion.py"]="src/guofeats/common/quaternion.py"
    ["${GUOFEATS_DIR}/common/skeleton.py"]="src/guofeats/common/skeleton.py"
    ["${GUOFEATS_DIR}/compute_guoh3dfeats.py"]="prepare/compute_guoh3dfeats.py"
  )
  for dest in "${!FETCH[@]}"; do
    curl -sfL --retry 3 "${UPSTREAM}/${FETCH[${dest}]}" -o "${dest}"
  done
  printf 'from .motion_representation import joints_to_guofeats, guofeats_to_joints  # noqa\n' \
    > "${GUOFEATS_DIR}/__init__.py"
  : > "${GUOFEATS_DIR}/common/__init__.py"
  echo "   fetched from pinned upstream commit"
  echo "   NOTE: skeleton_example_h3d.npy (HumanML3D 000021 frame 0) is NOT"
  echo "   fetchable from the TMR repo; motion_representation.py loads it at"
  echo "   import.  Obtain via the HumanML3D pipeline if absent."
fi

echo "== 3/4 DistilBERT text backbone =="
python3 - <<'PYTHON'
from transformers import AutoModel, AutoTokenizer

# Cached download (~268 MB once); TMR's text tower consumes token-level
# last_hidden_state from exactly this backbone.
AutoTokenizer.from_pretrained("distilbert-base-uncased")
AutoModel.from_pretrained("distilbert-base-uncased")
print("   distilbert-base-uncased ready")
PYTHON

echo "== 4/4 strict-load + forward check =="
cd "${REPO_ROOT}"
env -u PYTHONPATH python3 - <<'PYTHON'
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import numpy as np

from tools.tmr_runtime import (
    TextEmbedder,
    encode_motions,
    encode_texts,
    load_motion_encoder,
    load_text_encoder,
)

motion_encoder, normalizer = load_motion_encoder()
text_encoder = load_text_encoder()
feats = [np.zeros((30, 263), dtype=np.float32)]
motion_latents = encode_motions(motion_encoder, normalizer, feats)
text_latents = encode_texts(text_encoder, TextEmbedder(), ["a person kicks"])
assert motion_latents.shape == (1, 256) and np.isfinite(motion_latents).all()
assert text_latents.shape == (1, 256) and np.isfinite(text_latents).all()
print("   both towers load strictly and emit finite 256-D latents")
PYTHON

echo "TMR environment ready."
