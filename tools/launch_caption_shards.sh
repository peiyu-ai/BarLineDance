#!/usr/bin/env bash
#
# Fan the M3 caption step across GPUs, one shard per card.
#
# Segments are partitioned by a digest of (recording, start, end), so the shards
# are disjoint no matter how many workers run, in what order they start, or how
# often they are resumed.  Re-running this script only fills gaps: a shard whose
# worker is alive is skipped, and a worker that restarts re-reads its own output
# and skips the segments already captioned.
#
# Qwen3-VL needs transformers >= 4.56 for its architecture and the pod's global
# install is pinned at 4.51 by tensorrt-llm, so the newer library lives in a
# venv beside the weights.  Qwen2.5-VL loads under either; QWEN_PY selects.
#
# Usage:
#   bash tools/launch_caption_shards.sh                 # defaults below
#   MODEL=third_party/QwenVL/Qwen2.5-VL-7B-Instruct \
#     QWEN_PY=python3 OUT=runs/wild_captions_v1 bash tools/launch_caption_shards.sh

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs

LABELS="${LABELS:-data/wild3d/wild_tmr_labels_v1}"
BUNDLE="${BUNDLE:-data/wild3d/wild_performance_v1}"
VIDEO_DIR="${VIDEO_DIR:-data/wild_videos_converted}"
MODEL="${MODEL:-third_party/QwenVL/Qwen3-VL-30B-A3B-Instruct/main}"
QWEN_PY="${QWEN_PY:-third_party/QwenVL/.venv-qwen3vl/bin/python}"
OUT="${OUT:-runs/wild_captions_v1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
POSESCRIPT="${POSESCRIPT:---posescript}"
# Frame size drives vision-token count, which is what runs a 30B MoE out of
# memory on a 72 GB card long before the 7B notices: 60 GB of weights leaves
# little room, and 448px at batch 4 already OOMs.
MAX_SIDE="${MAX_SIDE:-448}"
# Crop each segment to the dancer GVHMR tracked.  On this corpus the dancer is
# a median 42% of frame height, so a full frame spends most of its vision
# tokens on the studio -- and in a wide class shot the caption ends up
# describing a crowd while the motion describes one person.
# ``:-`` would treat an explicit empty value as "unset" and hand the wild
# corpus's boxes to whatever is being captioned.  AIST++ is single-dancer and
# full-frame, so its correct value *is* empty -- and silently cropping it to
# boxes keyed by wild clip ids would simply find nothing, which looks like
# success.  Same trap NORMALIZED hit in run_atomic_discovery.sh.
PERSON_BOXES="${PERSON_BOXES-data/wild3d/gvhmr_raw}"
person_box_arg=""
[ -n "$PERSON_BOXES" ] && person_box_arg="--person-boxes $PERSON_BOXES"
FRAMES_PER_SEGMENT="${FRAMES_PER_SEGMENT:-6}"
# Which cards to use, in order.  A list rather than a first-index because the
# free cards are rarely contiguous -- one is usually holding a model for a
# probe -- and shard count follows from it, so the two can never disagree.
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
read -r -a GPU_LIST <<< "$GPUS"
NUM_SHARDS="${NUM_SHARDS:-${#GPU_LIST[@]}}"

mkdir -p "$OUT"
echo "== ${NUM_SHARDS} caption shards on GPU ${GPUS}, model $(basename "$(dirname "$MODEL")")/$(basename "$MODEL") =="

# One shard per card, never two, and nothing else on that card either.
# Measured 2026-08-13: a single Qwen3-VL-30B-A3B captioner sits at 71.4-72.5 GB
# on a 73.4 GB device -- it owns the card outright.  Anything co-scheduled with
# it (a second shard, a planner, an S3D fleet) will OOM one of the two, so the
# schedulers around this launcher have to serialise rather than pack.
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${GPU_LIST[$((shard % ${#GPU_LIST[@]}))]}"
  # A pidfile per shard rather than pgrep on the script name: this launcher,
  # any editor and any log tail all mention that name, and a self-match reads
  # as "already running" when nothing is.
  marker="logs/.caption_shard${shard}.pid"
  if [ -f "$marker" ] && kill -0 "$(cat "$marker")" 2>/dev/null; then
    echo "   shard $shard already running (pid $(cat "$marker")), skipping"
    continue
  fi
  setsid nohup env CUDA_VISIBLE_DEVICES=$gpu \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash -c 'echo $$ > '"$marker"'; exec '"$QWEN_PY"' tools/caption_segments_vlm.py \
      --labels '"$LABELS"' --bundle '"$BUNDLE"' --video-dir '"$VIDEO_DIR"' \
      --model '"$MODEL"' --output '"$OUT"'/captions_shard'"$shard"'.jsonl \
      --shard '"$shard"' --num-shards '"$NUM_SHARDS"' --batch-size '"$BATCH_SIZE"' \
      --max-side '"$MAX_SIDE"' --frames-per-segment '"$FRAMES_PER_SEGMENT"' \
      '"$person_box_arg"' \
      --device cuda:0 '"$POSESCRIPT" \
    >> "logs/caption_shard${shard}.log" 2>&1 < /dev/null &
  disown
  echo "   shard $shard launched on GPU $gpu"
done

echo
echo "all launched; tail logs/caption_shard*.log for progress"
echo "when every shard is done, concatenate:"
echo "  cat ${OUT}/captions_shard*.jsonl > ${OUT}/captions.jsonl"
