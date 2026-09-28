#!/usr/bin/env bash
#
# DWPose's runtime, which is not this repo's runtime.
#
# The interpreter everything else here runs under has CPU-only onnxruntime
# 1.17, and DWPose on CPU is not a real option at corpus scale: detection alone
# is ~10 ms/frame on a GPU, and the wild corpus is millions of frames.  So this
# builds a venv with onnxruntime-gpu for the ingestion step only, the same way
# Lodge isolates it, and fetches the two ONNX files.
#
# Usage:
#   bash tools/setup_dwpose_env.sh
#   .venv_ortgpu/bin/python tools/ingest_wild_uploads.py --help
#
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$(pwd)"

VENV="${VENV:-$REPO/.venv_ortgpu}"
WEIGHTS="$REPO/third_party/DWPose/weights"
mkdir -p "$WEIGHTS"

# Sizes are the check, not just presence: an interrupted download leaves a
# short file that loads as a corrupt ONNX graph with an unhelpful error.
declare -A EXPECTED=( ["yolox_l.onnx"]=216746733 ["dw-ll_ucoco_384.onnx"]=134399116 )
declare -A URLS=(
  ["yolox_l.onnx"]="https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx"
  ["dw-ll_ucoco_384.onnx"]="https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx"
)

for name in "${!URLS[@]}"; do
  dest="$WEIGHTS/$name"
  have="$(stat -c%s "$dest" 2>/dev/null || echo 0)"
  if [ "$have" = "${EXPECTED[$name]}" ]; then
    echo "have $name"
    continue
  fi
  echo "fetching $name (have ${have}, want ${EXPECTED[$name]})"
  curl -fL --retry 3 -o "$dest.partial" "${URLS[$name]}" && mv "$dest.partial" "$dest" || {
    rm -f "$dest.partial"
    echo "error: could not fetch $name; it is also in the team OSS store:" >&2
    echo "  python3 tools/oss_assets.py pull third_party/DWPose" >&2
    exit 1
  }
done

if [ ! -x "$VENV/bin/python" ]; then
  echo "creating $VENV"
  python3 -m venv --system-site-packages "$VENV" || exit 1
  "$VENV/bin/pip" install --upgrade pip >/dev/null
  # opencv-python-headless, not opencv-python: this runs headless and the GUI
  # build drags in X libraries that are not installed.
  # --system-site-packages above, and torch left out here, because this host
# pins torch to an NVIDIA build (2.8.0a0+...nv25.6) through a pip constraint
# file: a fresh venv cannot resolve it and the install dies on
# ResolutionImpossible.  torch is needed for exactly one call -- save_gvhmr_bbx
# writing bbx.pt -- so it is inherited rather than reinstalled.  The venv's own
# site-packages still take precedence, which is what keeps onnxruntime-gpu in
# front of the host's CPU-only onnxruntime; the check below is what proves it.
  # onnxruntime-gpu is *pinned*, not floated.  The unpinned install resolves to
  # the newest wheel, which on 2026-08-19 was 1.29.0 -- built against CUDA 13,
  # while this host is CUDA 12.9 (libcublasLt.so.12).  The provider then fails
  # to load at session creation and DWPose silently runs on CPU: one upload took
  # 35 s of wall clock and 59 minutes of user CPU, which over the 1,335-upload
  # re-cut is the difference between an hour and most of a day.  1.22.0 is the
  # newest wheel on the CUDA 12 line.
  "$VENV/bin/pip" install "onnxruntime-gpu==1.22.0" "opencv-python-headless" numpy || exit 1
fi

# Open a session on the device rather than asking which providers exist.
# ``get_available_providers()`` lists what the wheel was *compiled* with and
# says nothing about whether the shared library loads, so the previous check
# passed while every session silently fell back to CPU -- a gate that reads
# like a check and cannot fail for the failure it is there to catch.  Creating
# a session is what distinguishes the two, and the model is the real one this
# environment exists to run.
"$VENV/bin/python" - "$REPO" <<'PY'
import pathlib
import sys

import onnxruntime as ort

model = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".") / \
    "third_party/DWPose/weights/yolox_l.onnx"
print("onnxruntime", ort.__version__, ort.get_available_providers())
if not model.is_file():
    raise SystemExit("cannot verify the GPU path: no model at {}".format(model))
try:
    session = ort.InferenceSession(str(model), providers=["CUDAExecutionProvider"])
except Exception as error:                                    # noqa: BLE001
    raise SystemExit("onnxruntime could not open a CUDA session: {}: {}".format(
        type(error).__name__, error))
active = session.get_providers()
print("session providers", active)
if "CUDAExecutionProvider" not in active:
    raise SystemExit(
        "onnxruntime-gpu is installed and CUDAExecutionProvider is compiled in, "
        "but the session fell back to {} -- the wheel's CUDA major version does "
        "not match this host.  Host CUDA: check `python -c \"import torch; "
        "print(torch.version.cuda)\"`; this wheel wants what its name says."
        .format(active))
PY
echo
echo "ready: $VENV/bin/python tools/ingest_wild_uploads.py ..."
