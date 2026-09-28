#!/usr/bin/env bash
#
# The clip-length gate, end to end.
#
#   cut      same uploads -> 8/16/24/32/48/64 s, production ffmpeg recipe
#   extract  GVHMR per GPU, model loaded once per shard (--video-list)
#   convert  raw result -> 151-D
#   report   physical measures as a function of length, vs the mocap p99 bar
#
# This exists to decide one constant -- how long a wild clip should be -- before
# spending 50-60 GPU-hours regenerating 3D for the whole corpus on a number
# inherited from another repo's training recipe.
#
# Usage:
#   GPUS="0 1 2 3 5" UPLOADS=24 bash tools/run_clip_length_scan.sh
#
# Every stage skips work whose output exists, so an interrupted scan resumes.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$(pwd)"
mkdir -p logs

WORK_DIR="${WORK_DIR:-runs/clip_length_scan}"
VIDEOS_DIR="${VIDEOS_DIR:-data/wild_videos_20260811}"
LENGTHS="${LENGTHS:-8 16 24 32 48 64}"
UPLOADS="${UPLOADS:-24}"
GPUS="${GPUS:-0 1 2 3 5}"
COMMON_WINDOW="${COMMON_WINDOW:-8}"
# The corpus was built with DPVO and three tracking attempts; a gate run under
# different tracking would be measuring the tracker, not the length.
USE_DPVO="${USE_DPVO:-1}"
VO_ATTEMPTS="${VO_ATTEMPTS:-3}"

RAW_ROOT="$REPO/$WORK_DIR/gvhmr_raw"
CONVERTED_ROOT="$REPO/$WORK_DIR/converted"
mkdir -p "$RAW_ROOT" "$CONVERTED_ROOT"

step() { echo; echo "== $* =="; }

step "cut ${UPLOADS} uploads to lengths: ${LENGTHS}"
python3 tools/scan_clip_length.py --stage cut --videos-dir "$VIDEOS_DIR" \
  --work-dir "$WORK_DIR" --lengths $LENGTHS --uploads "$UPLOADS" --gpus $GPUS \
  2>&1 | tee "logs/clip_length_cut.log" | tail -15 || exit 1

step "extract on GPUs: ${GPUS}"
if [ -n "$USE_DPVO" ] && [ "$USE_DPVO" != "0" ]; then
  VO_FLAG="--use-dpvo"
  VO_PYTHONPATH="$REPO/third_party/torch_scatter_compat:$REPO/third_party/pytorch3d_compat:third-party/DPVO:."
else
  VO_FLAG=""
  VO_PYTHONPATH="$REPO/third_party/pytorch3d_compat:."
fi

pids=()
for gpu in $GPUS; do
  list="$REPO/$WORK_DIR/video_lists/gpu${gpu}.txt"
  [ -s "$list" ] || { echo "   gpu $gpu: empty list, skipping"; continue; }
  # cd into GVHMR (its configs resolve relative to its root) but call the
  # extractor by absolute path: third_party/GVHMR is a symlink into the asset
  # cache, and ``../../tools`` from inside it lands in the cache's parent.
  (cd third_party/GVHMR &&
    PYTHONPATH="$VO_PYTHONPATH" CUDA_VISIBLE_DEVICES="$gpu" \
    python "$REPO/tools/run_gvhmr_extract.py" --video-list "$list" \
      --output-root "$RAW_ROOT" --vo-attempts "$VO_ATTEMPTS" $VO_FLAG) \
    > "$REPO/logs/clip_length_extract_${gpu}.log" 2>&1 &
  pids+=($!)
  echo "   gpu $gpu: pid ${pids[-1]}, $(wc -l < "$list") clips"
done
for pid in "${pids[@]:-}"; do [ -n "$pid" ] && wait "$pid"; done

step "convert to 151-D"
converted=0 skipped=0 failed=0
for result in "$RAW_ROOT"/*/hmr4d_results.pt; do
  [ -e "$result" ] || continue
  stem="$(basename "$(dirname "$result")")"
  if [ -f "$CONVERTED_ROOT/$stem/motion.npy" ]; then skipped=$((skipped + 1)); continue; fi
  if python3 tools/convert_gvhmr_result.py --result "$result" \
       --extract-meta "$RAW_ROOT/$stem/extract_meta.json" \
       --output-dir "$CONVERTED_ROOT/$stem" >/dev/null 2>&1; then
    converted=$((converted + 1))
  else
    failed=$((failed + 1)); echo "   FAIL convert $stem"
  fi
done
echo "   converted=$converted skipped=$skipped failed=$failed"

step "report"
python3 tools/scan_clip_length.py --stage report --work-dir "$WORK_DIR" \
  --lengths $LENGTHS --common-window "$COMMON_WINDOW" \
  2>&1 | tee "logs/clip_length_report.log" | tail -30
