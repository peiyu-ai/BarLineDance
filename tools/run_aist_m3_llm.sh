#!/usr/bin/env bash
#
# Tab. 2's "w/ LLM" row on AIST++, as one resumable command.
#
#   1 caption    Qwen3-VL over every M2 segment, with PoseScript keyframe cues
#   2 merge      the per-shard caption files into one
#   3 summarise  the paper's summarizing LLM forms sub-prototypes and tags them
#   4 recluster  read that grouping back onto the segments -> the w/ LLM bundle
#   5 score      M1 + M2 + M3 against Fig. 4a/4b/4c
#
#   bash tools/run_aist_m3_llm.sh                 # all five, skipping what exists
#   GPUS="0 1 2 3" bash tools/run_aist_m3_llm.sh  # caption on four cards
#   FROM=3 bash tools/run_aist_m3_llm.sh          # start at the summarizer
#
# Three flags are pinned here rather than left to defaults, because each one
# silently produces a *plausible* wrong answer:
#
# --embedding-cache  Segment boundaries come from what M2 clustered.  Without
#   it both the captioner and the re-clusterer fall back to runs of equal frame
#   label, which merges 32.2% of the segments on this corpus -- and mixing the
#   two conventions is worse than either, because the caption rows are keyed
#   (recording, start, end) and simply stop matching.  Measured 2026-08-13:
#   9,273 runs against 12,946 segments.
#
# --video-per-motion 2  AIST++'s frame_ids.npy is the identity map over 30 fps
#   motion frames while the videos are 59.94 fps, so motion frame t is video
#   frame 2t.  At 1 every caption describes the right video at the wrong
#   moment, and nothing downstream can tell.  The wild corpus needs 1.
#
# --genre-map  The paper pre-splits each prototype by dance genre, and the
#   summarizer reads the key from this file.  Without it every AIST recording
#   lands in group "?", the pre-split quietly disappears from the LLM stage
#   while the re-clusterer still applies it, and the two stages disagree about
#   what a group is.
#
# The captioner and the summarizer are both resumable: the captioner skips
# segments already in its shard file, so a killed card is re-run, not restarted.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TAG="${TAG:-aist_v1}"
GPUS="${GPUS:-7}"
FROM="${FROM:-1}"
MODEL="${MODEL:-third_party/QwenVL/Qwen3-VL-30B-A3B-Instruct/main}"
POSESCRIPT="${POSESCRIPT:-third_party/PoseScript/capgen_CAtransfPSA2H2_dataPSA2ftPSH2/seed1/checkpoint_best.pth}"
BATCH="${BATCH:-4}"

LABELS="data/atomic_aistpp/${TAG}_labels"
BUNDLE="data/atomic_aistpp/aist_full_performance_v1"
VIDEOS="data/aist_videos"
CACHE="runs/${TAG}_tmr_embeddings.npz"
GENRES="runs/${TAG}_genre_map.json"
# Default to the directory the first shard already wrote into, so a chain
# started by hand is resumed rather than re-captioned.
CAPDIR="${CAPDIR:-runs/aist_captions_v1}"
CAPTIONS="${CAPDIR}/captions.jsonl"
SUBPROTO="runs/${TAG}_subprototypes_llm.json"
OUTDIR="data/atomic_aistpp/${TAG}_ingroup_llm"
SEG="runs/${TAG}_seg/segmentation.json"
REPORT="runs/${TAG}_paper_alignment_llm.json"

read -ra CARDS <<< "$GPUS"
NUM_SHARDS="${NUM_SHARDS:-8}"
mkdir -p logs "$CAPDIR"

if [ "$FROM" -le 1 ]; then
  echo "== 1 caption: ${NUM_SHARDS} shards over cards [${GPUS}] =="
  pids=()
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    gpu="${CARDS[$((shard % ${#CARDS[@]}))]}"
    CUDA_VISIBLE_DEVICES="$gpu" python3 tools/caption_segments_vlm.py \
        --labels "$LABELS" --bundle "$BUNDLE" --video-dir "$VIDEOS" \
        --model "$MODEL" --embedding-cache "$CACHE" --video-per-motion 2 \
        --posescript --posescript-model "$POSESCRIPT" \
        --batch-size "$BATCH" --shard "$shard" --num-shards "$NUM_SHARDS" \
        --output "${CAPDIR}/captions_shard${shard}.jsonl" \
        > "logs/${TAG}_caption_shard${shard}.log" 2>&1 &
    pids+=($!)
    echo "  shard ${shard} -> gpu ${gpu} pid ${!}"
    sleep 5                      # stagger the 58 GB model loads off one another
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=$((failed + 1)); done
  echo "  captioning finished, ${failed} shard(s) exited non-zero"
fi

if [ "$FROM" -le 2 ]; then
  echo "== 2 merge =="
  # Rebuilt from the shards every time rather than appended to: a merge that
  # accumulates would double every row on a re-run, and load_captions keeps the
  # first of a duplicate, so the damage would be invisible.
  cat "${CAPDIR}"/captions_shard*.jsonl > "$CAPTIONS"
  echo "  $(wc -l < "$CAPTIONS") caption rows -> ${CAPTIONS}"
fi

if [ "$FROM" -le 3 ]; then
  echo "== 3 summarise (the paper's second model) =="
  CUDA_VISIBLE_DEVICES="${CARDS[0]}" python3 tools/summarize_subprototypes_llm.py \
      --captions "$CAPTIONS" --model "$MODEL" --genre-map "$GENRES" \
      --output "$SUBPROTO" > "logs/${TAG}_summarize.log" 2>&1 || exit 1
  echo "  wrote ${SUBPROTO}"
fi

if [ "$FROM" -le 4 ]; then
  echo "== 4 recluster: Tab. 2 w/ LLM =="
  [ -e "$OUTDIR" ] && { echo "  ${OUTDIR} exists; bundles publish into a new directory"; exit 1; }
  python3 tools/recluster_atomics_ingroup.py \
      --labels "$LABELS" --bundle "$BUNDLE" --output-dir "$OUTDIR" \
      --embedding-cache "$CACHE" --genre-split \
      --captions "$CAPTIONS" --subprototypes "$SUBPROTO" \
      --target-size 32 --seed 20260810 \
      > "logs/${TAG}_recluster_llm.log" 2>&1 || exit 1
  echo "  wrote ${OUTDIR}"
fi

if [ "$FROM" -le 5 ]; then
  echo "== 5 score against Fig. 4a/4b/4c =="
  python3 tools/report_paper_alignment.py \
      --segmentation "$SEG" --labels "$LABELS" --embedding-cache "$CACHE" \
      --sub-labels "$OUTDIR" --output "$REPORT"
fi
