#!/usr/bin/env bash
#
# Fan wild-corpus GVHMR extraction across GPUs.  Re-running this script only
# fills gaps: a shard whose loop is already alive is skipped, output that
# already exists is skipped by the shard script itself, and logs are appended,
# never truncated.
#
#   GPU 1-6   wild GVHMR 3D extraction, 6 shards over the full 6041-clip pool
#
# GPU 0 is externally held.  GPU 7 stays free on purpose: inference, gate
# probes and gallery renders all need a card, and a sweep that eats every GPU
# means no one can look at a result for a day.
#
# S3D visual features are NOT launched here.  They contend for the same cards
# as GVHMR, and a clip whose 3D extraction fails is a clip whose visual
# features were wasted, so they belong *after* this sweep and restricted to
# converted clips:
#
#   CUDA_VISIBLE_DEVICES=$g python tools/extract_wild_s3d_shard.py \
#     --shard $s --num-shards $n --video-dir data/wild_videos_all
#
# Usage: bash tools/launch_parallel_pipeline.sh

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs

VIDEO_ROOT="$PWD/data/wild_videos_all"
NUM_SHARDS=6
# DPVO by default.  SimpleVO discarded 45% of this corpus by dying on the first
# frame pair without matchable texture; DPVO tracks those clips and, where both
# succeed, agrees to a median 0.47 degrees on root and body rotation alike.
# Set USE_DPVO= to fall back to SimpleVO.
USE_DPVO="${USE_DPVO-1}"
# When DPVO still diverges after its retries, measure the camera's real
# rotation and fall back to static-cam only where that measurement says the
# R_w2c = I assertion is within threshold.  Surveyed on 40 diverged clips, 6
# pass at 3 degrees and the median clip rotates 12 -- so this is a narrow,
# evidence-gated recovery, and every clip it accepts carries the measurement.
STATIC_CAM_FALLBACK="${STATIC_CAM_FALLBACK-1}"
STATIC_CAM_THRESHOLD_DEG="${STATIC_CAM_THRESHOLD_DEG:-3.0}"
MANIFEST="data/wild3d/source_manifests/tiktok_new_b2gpu/exact_content_merged_v1/sequences.jsonl"

echo "== link the QC-passed wild pool =="
python3 - "$MANIFEST" "$VIDEO_ROOT" <<'PYTHON'
import json
import pathlib
import sys

manifest, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
made = have = missing = 0
for line in manifest.open(encoding="utf-8"):
    record = json.loads(line)
    if record["qc"]["inventory_status"] != "ready_for_wham":
        continue
    source = pathlib.Path(record["assets"]["source_video"])
    link = out / (record["legacy_clip_id"] + ".mp4")
    if link.is_symlink() or link.exists():
        have += 1
    elif not source.exists():
        missing += 1
    else:
        link.symlink_to(source)
        made += 1
print("   linked {} new, {} already present, {} missing source".format(made, have, missing))
PYTHON

echo "== wild GVHMR extraction, ${NUM_SHARDS} shards on GPU 1-${NUM_SHARDS} =="
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu=$((shard + 1))
  # One pidfile per shard while its loop is alive.  pgrep on the script name is
  # unreliable here: monitors, editors and this very script mention it, and a
  # self-match reads as "already running" when nothing is.
  marker="logs/.gvhmr_all_shard${shard}.pid"
  if [ -f "$marker" ] && kill -0 "$(cat "$marker")" 2>/dev/null; then
    echo "   shard $shard already running (pid $(cat "$marker")), skipping"
    continue
  fi
  setsid nohup env SHARD=$shard NUM_SHARDS=$NUM_SHARDS GPU=$gpu VIDEO_ROOT="$VIDEO_ROOT" \
    USE_DPVO="$USE_DPVO" STATIC_CAM_FALLBACK="$STATIC_CAM_FALLBACK" \
    STATIC_CAM_THRESHOLD_DEG="$STATIC_CAM_THRESHOLD_DEG" \
    bash -c 'echo $$ > logs/.gvhmr_all_shard'"$shard"'.pid; exec bash tools/run_gvhmr_batch_shard.sh' \
    >> "logs/gvhmr_all_shard${shard}.log" 2>&1 < /dev/null &
  disown
  echo "   shard $shard launched on GPU $gpu"
done

echo
echo "all launched; tail logs/gvhmr_all_shard*.log for progress"
