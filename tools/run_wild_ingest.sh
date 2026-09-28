#!/usr/bin/env bash
#
# Re-cut the whole wild corpus on content, and hand GVHMR the dancer.
#
#   scan    detect every frame of every upload -> runs/ingest_scan_cache
#   cut     spans -> clip.mp4 + audio.wav + keypoints + bbx.pt
#
# The two stages are separate because only the second depends on --max-seconds,
# and the first is the expensive one (~19 GPU-hours over 10,793 uploads).  The
# scan can therefore run while the clip-length gate is still deciding, and a
# later change to the cap re-cuts from cache instead of re-detecting.
#
# Usage:
#   MAX_SECONDS=<from tools/scan_clip_length.py> GPUS="0 1 2 3 4 5" \
#     bash tools/run_wild_ingest.sh
#
#   STAGE=scan    fill the cache only (no --max-seconds needed)
#   STAGE=cut     cut from the cache
#   STAGE=both    (default)
#
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$(pwd)"
mkdir -p logs

VIDEOS="${VIDEOS:-data/wild_videos_20260811}"
OUT_ROOT="${OUT_ROOT:-data/wild_ingest_v1}"
CACHE="${CACHE:-runs/ingest_scan_cache}"
GPUS="${GPUS:-0 1 2 3 4 5}"
STAGE="${STAGE:-both}"
ORT_PY="${ORT_PY:-$REPO/.venv_ortgpu/bin/python}"
[ -x "$ORT_PY" ] || ORT_PY="${E2E_ROOT:-/workspace/e2e}/Lodge/.venv_ortgpu/bin/python"
[ -x "$ORT_PY" ] || { echo "no onnxruntime-gpu interpreter; run tools/setup_dwpose_env.sh" >&2; exit 1; }

gpu_list=($GPUS); n=${#gpu_list[@]}
step() { echo; echo "== $* =="; }

fan_out() {   # fan_out <label> <extra args...>
  local label="$1"; shift
  local pids=()
  for i in "${!gpu_list[@]}"; do
    CUDA_VISIBLE_DEVICES="${gpu_list[$i]}" setsid "$ORT_PY" tools/ingest_wild_uploads.py \
      --videos-dir "$VIDEOS" --out-root "$OUT_ROOT" --scan-cache "$CACHE" \
      --shard "$i" --num-shards "$n" "$@" \
      > "logs/ingest_${label}_${i}.log" 2>&1 &
    pids+=($!)
    echo "   shard $i -> gpu ${gpu_list[$i]} (pid ${pids[-1]})"
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
}

if [ "$STAGE" = scan ] || [ "$STAGE" = both ]; then
  step "scan $(ls "$VIDEOS"/*.mp4 | wc -l) uploads across ${n} GPUs"
  fan_out scan --scan-only
  echo "   cache: $(ls "$CACHE"/*.npz 2>/dev/null | wc -l) uploads"
fi

if [ "$STAGE" = cut ] || [ "$STAGE" = both ]; then
  : "${MAX_SECONDS:?set MAX_SECONDS from tools/scan_clip_length.py; there is no safe default}"
  step "cut at max ${MAX_SECONDS}s"
  fan_out cut --max-seconds "$MAX_SECONDS"
  clips="$(ls -d "$OUT_ROOT"/*__clip*/ 2>/dev/null | wc -l)"
  echo "   $clips clips in $OUT_ROOT"
  # The manifests are the accounting: every upload appears whether or not it
  # produced a clip, so "we lost some" can never hide as "we never tried".
  python3 - "$OUT_ROOT" <<'PY'
import collections, json, pathlib, sys
sys.path.insert(0, ".")
# One row per upload, the most recent: the manifests are append-only, so an
# upload re-ingested under --redo has two rows and summing both doubles every
# number this summary prints.
from tools.ingest_wild_uploads import manifest_rows
root = pathlib.Path(sys.argv[1])
status = collections.Counter()
clips = collections.Counter()
seconds = kept = 0.0
for record in manifest_rows(root).values():
    status[record["status"]] += 1
    seconds += record.get("seconds", 0.0)
    for span in record.get("spans", []):
        kept += (span[1] - span[0]) / 30.0
    for clip in record.get("clips", []):
        clips[clip.get("end_reason", clip.get("status", "?"))] += 1
print(json.dumps({"uploads": sum(status.values()), "upload_status": dict(status),
                  "clip_end_reasons": dict(clips),
                  "upload_seconds": round(seconds, 1), "kept_seconds": round(kept, 1),
                  "retained_fraction": round(kept / max(seconds, 1e-9), 4)}, indent=1))
PY
fi
