#!/usr/bin/env bash
#
# The paper's atomic movement discovery, end to end, as one command.
#
#   M1 segmentation (Alg. 1)      -> runs/<TAG>_seg/segmentation.json
#   M2 TMR clustering             -> <LABELS_ROOT>/<TAG>_labels
#   M3a caption every segment     -> runs/<TAG>_captions/captions.jsonl
#   M3b summarizing LLM           -> runs/<TAG>_captions/subprototypes.json
#   M3c in-group re-clustering    -> <LABELS_ROOT>/<TAG>_ingroup
#   validate: audit + gallery     -> runs/<TAG>_audit.json, runs/<TAG>_gallery
#
# Every stage is skipped when its output already exists, so a failed or
# rejected run is resumed rather than restarted -- which matters because the
# loop this script exists for is "validate, fix, re-run", and re-running the
# 22k-segment caption pass to change a re-clustering parameter would be an
# hour wasted each time.  Delete the output of the earliest stage you want
# redone and run again.
#
# Usage:
#   TAG=wild_v2 FRAMES_PER_CLUSTER=32 MIN_LENGTH=18 bash tools/run_atomic_discovery.sh
#
#   # AIST++, the corpus Fig. 4 was measured on: its raw bundle is already in
#   # the state TMR wants, so the normalizer binding is switched off, and the
#   # run stops at the vocabulary rather than spending 19 GPU-hours on captions.
#   TAG=aist_v1 FEATURES_DIR=data/aist_visual_s3d VIDEO_DIR=data/aist_videos \
#     BUNDLE=data/atomic_aistpp/aist_full_performance_v1 \
#     SOURCES=data/atomic_aistpp/aist_full_performance_v1/sources.jsonl \
#     NORMALIZED= NORMALIZER= LABELS_ROOT=data/atomic_aistpp \
#     FRAMES_PER_CLUSTER=36 MIN_LENGTH=20 STOP_AFTER=M2 \
#     bash tools/run_atomic_discovery.sh
#
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs runs

TAG="${TAG:-wild_v2}"
FEATURES_DIR="${FEATURES_DIR:-data/wild_visual_s3d}"
BUNDLE="${BUNDLE:-data/wild3d/wild_performance_v1}"
VIDEO_DIR="${VIDEO_DIR:-data/wild_videos_converted}"
# Empty means "this corpus has no normalized release yet", not "use the
# default": AIST's raw bundle is already in the state TMR wants, and passing
# an empty path would resolve to the repo root and encode whatever it found.
NORMALIZED="${NORMALIZED-data/wild3d/wild_normalized_v1/sequences_normalized.jsonl}"
NORMALIZER="${NORMALIZER-data/wild3d/wild_normalizer_v1}"
SOURCES="${SOURCES:-data/wild3d/wild_performance_v1/sources.jsonl}"
# Where the label bundles are published.  The wild corpus keeps them beside
# its own 3D data; AIST's live with AIST's.
LABELS_ROOT="${LABELS_ROOT:-data/wild3d}"
# Stop after a named stage: M1, M2, M3a, M3b, M3c.  M3a is a 19-hour caption
# pass on six GPUs, so "run discovery up to the vocabulary and look at it" has
# to be expressible without editing this file.
STOP_AFTER="${STOP_AFTER:-}"

# M1: the two parameters Alg. 1 leaves unspecified, fitted to Fig. 4a by
# tools/calibrate_segmentation.py rather than guessed.
FRAMES_PER_CLUSTER="${FRAMES_PER_CLUSTER:-32}"
MIN_LENGTH="${MIN_LENGTH:-18}"
INDEX_WEIGHT="${INDEX_WEIGHT:-4.0}"
SEG_SEED="${SEG_SEED:-20260808}"

# M2.  The embedding cache is keyed by the content of everything upstream of
# it, so it can be on by default: a changed segmentation misses the key rather
# than serving embeddings of spans that no longer exist.  It is also what
# tools/report_paper_alignment.py reads to recover per-prototype counts.
CLASSES="${CLASSES:-100}"
ACCEPT_QUANTILE="${ACCEPT_QUANTILE:-0.85}"
CLUSTER_SEED="${CLUSTER_SEED:-20260809}"
EMBEDDING_CACHE="${EMBEDDING_CACHE:-runs/${TAG}_tmr_embeddings.npz}"

# M3
CAPTION_MODEL="${CAPTION_MODEL:-third_party/QwenVL/Qwen2.5-VL-7B-Instruct}"
CAPTION_PY="${CAPTION_PY:-python3}"
SUMMARY_MODEL="${SUMMARY_MODEL:-$CAPTION_MODEL}"
SUMMARY_PY="${SUMMARY_PY:-$CAPTION_PY}"
GPUS="${GPUS:-0 1 2 3 4 5}"
# Empty means "this corpus needs no crop", not "use the default": AIST++ is
# single-dancer and full-frame.  See launch_caption_shards.sh.
PERSON_BOXES="${PERSON_BOXES-data/wild3d/gvhmr_raw}"
PROBE_GPU="${PROBE_GPU:-6}"
TARGET_SIZE="${TARGET_SIZE:-32}"

SEG="runs/${TAG}_seg/segmentation.json"
LABELS="${LABELS_ROOT}/${TAG}_labels"
CAPTIONS_DIR="runs/${TAG}_captions"
CAPTIONS="${CAPTIONS_DIR}/captions.jsonl"
SUBPROTOTYPES="${CAPTIONS_DIR}/subprototypes.json"
INGROUP="${LABELS_ROOT}/${TAG}_ingroup"

step() { echo; echo "== $* =="; }
halt_after() {
  [ "$STOP_AFTER" = "$1" ] || return 0
  echo; echo "== stopping after $1 as asked =="
  exit 0
}

step "M1 segmentation (fpc=${FRAMES_PER_CLUSTER}, L_min=${MIN_LENGTH}, iw=${INDEX_WEIGHT})"
if [ -s "$SEG" ]; then
  echo "   $SEG exists, skipping"
else
  mkdir -p "$(dirname "$SEG")"
  # Sharded, and with the thread count pinned.  Alg. 1 is CPU-bound and the
  # corpus is ~18k clips: one process is a 13-hour step.  Splitting it is what
  # --shard/--merge-glob exist for.  The thread cap is not optional -- numpy
  # takes every core it can see per process, so N unpinned shards oversubscribe
  # by N and run *slower* than one (measured: load average 643 on 256 cores).
  SEG_SHARDS="${SEG_SHARDS:-8}"
  seg_pids=()
  for i in $(seq 0 $((SEG_SHARDS - 1))); do
    OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6 \
    python3 tools/segment_visual_atomics.py --features-dir "$FEATURES_DIR" \
      --output "${SEG%.json}_shard${i}.json" --frames-per-cluster "$FRAMES_PER_CLUSTER" \
      --min-length "$MIN_LENGTH" --index-weight "$INDEX_WEIGHT" --seed "$SEG_SEED" \
      --shard "$i" --num-shards "$SEG_SHARDS" \
      > "logs/${TAG}_seg_${i}.log" 2>&1 &
    seg_pids+=($!)
  done
  seg_bad=0
  for pid in "${seg_pids[@]}"; do wait "$pid" || seg_bad=$((seg_bad + 1)); done
  [ "$seg_bad" -eq 0 ] || { echo "error: ${seg_bad}/${SEG_SHARDS} segmentation shards failed" >&2; exit 1; }
  python3 tools/segment_visual_atomics.py --merge-glob "${SEG%.json}_shard*.json" \
    --output "$SEG" --features-dir "$FEATURES_DIR" \
    2>&1 | tee "logs/${TAG}_seg.log" | tail -5 || exit 1
  rm -f "${SEG%.json}"_shard*.json
fi
halt_after M1

step "M2 TMR clustering into ${CLASSES} prototypes"
if [ -d "$LABELS" ]; then
  echo "   $LABELS exists, skipping"
else
  bind_args=()
  [ -n "$NORMALIZED" ] && bind_args+=(--normalized-sequences "$NORMALIZED")
  [ -n "$NORMALIZER" ] && bind_args+=(--normalizer-bundle "$NORMALIZER")
  CUDA_VISIBLE_DEVICES="$PROBE_GPU" python3 tools/cluster_atomics_tmr.py \
    --bundle "$BUNDLE" --segmentation "$SEG" --output-dir "$LABELS" \
    --classes "$CLASSES" --accept-quantile "$ACCEPT_QUANTILE" --seed "$CLUSTER_SEED" \
    "${bind_args[@]}" --embedding-cache "$EMBEDDING_CACHE" \
    --sources "$SOURCES" --device cuda:0 \
    2>&1 | tee "logs/${TAG}_cluster.log" | tail -20 || exit 1
fi
halt_after M2

step "M3a caption every segment with $(basename "$CAPTION_MODEL")"
if [ -s "$CAPTIONS" ]; then
  echo "   $CAPTIONS exists, skipping"
else
  LABELS="$LABELS" BUNDLE="$BUNDLE" VIDEO_DIR="$VIDEO_DIR" MODEL="$CAPTION_MODEL" \
    QWEN_PY="$CAPTION_PY" OUT="$CAPTIONS_DIR" GPUS="$GPUS" \
    PERSON_BOXES="$PERSON_BOXES" \
    bash tools/launch_caption_shards.sh || exit 1
  # The launcher returns as soon as the shards are detached; the pipeline
  # cannot continue until every one of them has finished writing.
  # The launcher detaches via setsid and the workers only match this pattern
  # after their `exec`, so polling immediately can see nothing and fall
  # straight through to a `cat` of files that do not exist yet.  Wait for them
  # to appear first, then wait for them to go.
  # Anchored at ^ so a watchdog or log tail mentioning this command cannot be
  # mistaken for a worker.  Unanchored, the second loop below never returns --
  # measured 2026-08-13 on the sibling GVHMR fleet, where a wait like this left
  # a handoff parked forever.
  for _ in $(seq 1 60); do
    pgrep -f "^python3 tools/caption_segments_vlm[.]py" >/dev/null && break
    sleep 5
  done
  while pgrep -f "^python3 tools/caption_segments_vlm[.]py" >/dev/null; do sleep 30; done
  cat "${CAPTIONS_DIR}"/captions_shard*.jsonl > "$CAPTIONS"
  written="$(wc -l < "$CAPTIONS")"
  echo "   ${written} captions"
  # Shards that die on load leave empty files, and pgrep then reports "done"
  # immediately.  Without this the run would carry on and fail three stages
  # later, where the cause is no longer visible.
  if [ "$written" -lt 1000 ]; then
    echo "error: only ${written} captions; check logs/caption_shard*.log" >&2
    rm -f "$CAPTIONS"
    exit 1
  fi
fi

halt_after M3a

step "M3b summarizing LLM forms sub-prototypes"
if [ -s "$SUBPROTOTYPES" ]; then
  echo "   $SUBPROTOTYPES exists, skipping"
else
  CUDA_VISIBLE_DEVICES="$PROBE_GPU" "$SUMMARY_PY" tools/summarize_subprototypes_llm.py \
    --captions "$CAPTIONS" --model "$SUMMARY_MODEL" --output "$SUBPROTOTYPES" \
    --device cuda:0 2>&1 | tee "logs/${TAG}_summarize.log" | tail -20 || exit 1
fi

halt_after M3b

step "M3c in-group re-clustering (paper's w/ LLM row)"
if [ -d "$INGROUP" ]; then
  echo "   $INGROUP exists, skipping"
else
  python3 tools/recluster_atomics_ingroup.py --labels "$LABELS" --bundle "$BUNDLE" \
    --output-dir "$INGROUP" --captions "$CAPTIONS" --subprototypes "$SUBPROTOTYPES" \
    --target-size "$TARGET_SIZE" ${GROUP_KEYS:+--group-keys "$GROUP_KEYS"} ${GENRE_SPLIT:-} 2>&1 | tee "logs/${TAG}_recluster.log" | tail -30 || exit 1
fi

halt_after M3c

step "validate against the paper"
python3 tools/audit_atomic_vocabulary.py --labels "$INGROUP" --bundle "$BUNDLE" \
  --captions "$CAPTIONS" --output "runs/${TAG}_audit.json" \
  2>&1 | tee "logs/${TAG}_audit.log" | tail -40
# The audit's exit code is the verdict, and it is at the head of the pipe:
# plain $? here would report tail's status, which is 0 whatever the audit said.
verdict="${PIPESTATUS[0]}"

python3 tools/build_atomic_gallery.py --labels "$INGROUP" --bundle "$BUNDLE" \
  --video-dir "$VIDEO_DIR" --output-dir "runs/${TAG}_gallery" \
  --person-boxes "$PERSON_BOXES" \
  2>&1 | tee "logs/${TAG}_gallery.log" | tail -10

echo
if [ "$verdict" -eq 0 ]; then
  echo "PASS: runs/${TAG}_audit.json"
else
  echo "FAIL: read runs/${TAG}_audit.json, fix the stage at fault, delete its output, re-run"
fi
echo "gallery: runs/${TAG}_gallery/index.html"
exit "$verdict"
