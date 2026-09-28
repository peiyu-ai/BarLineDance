#!/usr/bin/env bash
#
# One shard of wild GVHMR extraction, batched: video -> world SMPL-X -> 151-D.
#
#   SHARD=<n> NUM_SHARDS=<k> GPU=<i> [VIDEO_ROOT=<dir>] [USE_DPVO=1] \
#     [STATIC_CAM_FALLBACK=1] [STATIC_CAM_THRESHOLD_DEG=3.0] \
#     bash tools/run_gvhmr_batch_shard.sh
#
# Replaces the per-clip loop in run_gvhmr_wild_shard.sh.  That loop spent 64 s
# of wall clock per clip against a median 17 s of real preprocessing: 74% went
# to interpreter start, hydra compose, model instantiation and checkpoint load,
# paid again for every clip.  Here extraction runs as one process over the
# whole shard, so that cost is paid once, and the cheap numpy-only convert and
# validate stages run afterwards in their own loop.
#
# Three passes, in order:
#
#   1. extract everything outstanding with the configured tracker;
#   2. for clips that produced no result, and only with STATIC_CAM_FALLBACK=1,
#      measure the camera's actual rotation and re-extract with --static-cam
#      *only* where that measurement says the assertion R_w2c = I is within
#      threshold.  The measurement is kept and recorded in the extraction's
#      provenance, so a static-cam clip always carries the evidence that
#      justified it.  Surveyed on 40 diverged clips, 6 pass at 3 degrees and
#      the median clip rotates 12 degrees -- this recovers a minority on
#      purpose, and refuses the rest;
#   3. convert + validate, marking clips that never produced a result.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$(pwd)"

: "${SHARD:?set SHARD}"; : "${NUM_SHARDS:?set NUM_SHARDS}"; : "${GPU:?set GPU}"

VIDEO_ROOT="${VIDEO_ROOT:-$REPO/data/wild_videos_all}"
USE_DPVO="${USE_DPVO:-}"
STATIC_CAM_FALLBACK="${STATIC_CAM_FALLBACK:-}"
STATIC_CAM_THRESHOLD_DEG="${STATIC_CAM_THRESHOLD_DEG:-3.0}"
RAW_ROOT="$REPO/data/wild3d/gvhmr_raw"
CONVERTED_ROOT="$REPO/data/wild3d/converted"
mkdir -p "$RAW_ROOT" "$CONVERTED_ROOT"

if [ -n "$USE_DPVO" ]; then
  VO_FLAG="--use-dpvo"
  VO_PYTHONPATH="../torch_scatter_compat:../pytorch3d_compat:third-party/DPVO:."
else
  VO_FLAG=""
  VO_PYTHONPATH="../pytorch3d_compat:."
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
TODO="$WORK/todo.txt"

# Membership is sha256(stem) mod NUM_SHARDS, matching the S3D shard scheme, so
# shards are disjoint and stable across relaunches.  Clips already converted,
# or marked failed by an earlier sweep, are dropped here rather than inside the
# loop, so the batch list is exactly the work.
python3 - "$VIDEO_ROOT" "$SHARD" "$NUM_SHARDS" "$CONVERTED_ROOT" "$RAW_ROOT" "$USE_DPVO" > "$TODO" <<'PY'
import hashlib
import pathlib
import sys

root, shard, num_shards = pathlib.Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
converted, raw, use_dpvo = pathlib.Path(sys.argv[4]), pathlib.Path(sys.argv[5]), bool(sys.argv[6])
for video in sorted(root.glob("*.mp4")):
    if int(hashlib.sha256(video.stem.encode()).hexdigest(), 16) % num_shards != shard:
        continue
    if (converted / video.stem / "metadata.json").is_file():
        continue
    # A marker means an earlier sweep failed this clip the same way it would
    # fail again -- unless the tracker itself has changed, which is exactly the
    # change the marker is waiting for.
    if (raw / video.stem / ".extract_failed").is_file() and not use_dpvo:
        continue
    print(video)
PY

total=$(wc -l < "$TODO")
echo "SHARD_START shard=$SHARD gpu=$GPU outstanding=$total dpvo=${USE_DPVO:-0} static_fallback=${STATIC_CAM_FALLBACK:-0}"
if [ "$total" -eq 0 ]; then
  echo "SHARD_DONE shard=$SHARD total=0 extracted=0 converted=0 failed=0"
  exit 0
fi

echo "== pass 1: batch extraction =="
(cd third_party/GVHMR &&
  PYTHONPATH="$VO_PYTHONPATH" CUDA_VISIBLE_DEVICES=$GPU \
  python ../../tools/run_gvhmr_extract.py \
    --video-list "$TODO" --output-root "$RAW_ROOT" $VO_FLAG)

if [ -n "$STATIC_CAM_FALLBACK" ]; then
  echo "== pass 2: evidence-gated static-camera retry =="
  while IFS= read -r video <&3; do
    stem="$(basename "$video" .mp4)"
    [ -f "$RAW_ROOT/$stem/hmr4d_results.pt" ] && continue
    evidence="$RAW_ROOT/$stem/camera_rotation.json"
    boxes="$RAW_ROOT/$stem/preprocess/bbx.pt"
    box_flag=""
    [ -f "$boxes" ] && box_flag="--person-boxes $boxes"
    if python tools/measure_camera_rotation.py --video "$video" $box_flag \
         --threshold-deg "$STATIC_CAM_THRESHOLD_DEG" > "$evidence" 2>/dev/null; then
      echo "STATIC_CAM_ELIGIBLE $stem"
      (cd third_party/GVHMR &&
        PYTHONPATH="$VO_PYTHONPATH" CUDA_VISIBLE_DEVICES=$GPU \
        python ../../tools/run_gvhmr_extract.py \
          --video "$video" --output-root "$RAW_ROOT" \
          --static-cam --static-cam-evidence "$evidence")
    else
      echo "STATIC_CAM_REFUSED $stem"
    fi
  done 3< "$TODO"
fi

echo "== pass 3: convert and validate =="
extracted=0 converted=0 failed=0
while IFS= read -r video <&3; do
  stem="$(basename "$video" .mp4)"
  result="$RAW_ROOT/$stem/hmr4d_results.pt"
  if [ ! -f "$result" ]; then
    mkdir -p "$RAW_ROOT/$stem"
    touch "$RAW_ROOT/$stem/.extract_failed"
    echo "FAIL $stem extract"; failed=$((failed + 1)); continue
  fi
  rm -f "$RAW_ROOT/$stem/.extract_failed"
  extracted=$((extracted + 1))
  if python tools/convert_gvhmr_result.py \
       --result "$result" \
       --extract-meta "$RAW_ROOT/$stem/extract_meta.json" \
       --output-dir "$CONVERTED_ROOT/$stem" >/dev/null &&
     python tools/preprocess_wild_3d.py validate \
       --output-dir "$CONVERTED_ROOT/$stem" >/dev/null; then
    converted=$((converted + 1))
    echo "OK $stem"
  else
    echo "FAIL $stem convert_or_validate"; failed=$((failed + 1))
  fi
done 3< "$TODO"

echo "SHARD_DONE shard=$SHARD total=$total extracted=$extracted converted=$converted failed=$failed"
