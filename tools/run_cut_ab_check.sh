#!/usr/bin/env bash
#
# The matched control for the content cut, on clips the rebuild already made.
#
# Arm A is Lodge's recipe verbatim (``ffmpeg -f segment -segment_time 16``) over
# a fixed 200-upload sample.  Arm B is the production content cut of *the same
# 200 uploads*, taken straight out of the rebuild's output rather than cut
# again.  Matching on uploads is the point: the published corpus is 2,434 clips
# and the rebuild is several times that, and this repo has already measured
# vocabulary statistics moving 7.7x with pool size -- an unmatched comparison
# would be reading corpus size.
#
# Both arms then get S3D features and Alg. 1 segmentation with identical
# parameters, so the cut is the only thing that differs.
#
# Usage:
#   GPUS="0 1 2 3 4 5" bash tools/run_cut_ab_check.sh
#
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$(pwd)"
mkdir -p logs

WORK="${WORK:-runs/cut_ab}"
INGEST="${INGEST:-data/wild_ingest_v1}"
GPUS="${GPUS:-0 1 2 3 4 5}"
FRAMES_PER_CLUSTER="${FRAMES_PER_CLUSTER:-34}"
MIN_LENGTH="${MIN_LENGTH:-18}"
INDEX_WEIGHT="${INDEX_WEIGHT:-4.0}"
gpu_list=($GPUS); n=${#gpu_list[@]}
step() { echo; echo "== $* =="; }

[ -s "$WORK/uploads.txt" ] || { echo "no $WORK/uploads.txt" >&2; exit 1; }

step "arm B: link the production content clips for the sampled uploads"
B_CLIPS="$WORK/content_clips"; mkdir -p "$B_CLIPS"
linked=0 missing=0
while IFS= read -r video <&3; do
  stem="$(basename "$video" .mp4)"
  found=0
  for d in "$INGEST/${stem}__clip"*/; do
    [ -f "$d/clip.mp4" ] || continue
    ln -sf "$REPO/$d/clip.mp4" "$B_CLIPS/$(basename "$d").mp4"
    linked=$((linked + 1)); found=1
  done
  [ "$found" -eq 1 ] || missing=$((missing + 1))
done 3< "$WORK/uploads.txt"
echo "   $linked content clips linked; $missing sampled uploads produced none"
# Uploads the content cut dropped are not a bookkeeping gap -- they are its
# cost, and the comparison has to carry them.
echo "$missing" > "$WORK/content_uploads_with_no_clip.txt"

step "S3D for both arms"
for arm in blind content; do
  src="$WORK/blind_clips"; [ "$arm" = content ] && src="$B_CLIPS"
  out="$WORK/${arm}_s3d"; mkdir -p "$out"
  pids=()
  for i in "${!gpu_list[@]}"; do
    CUDA_VISIBLE_DEVICES="${gpu_list[$i]}" python3 tools/extract_wild_s3d_shard.py \
      --shard "$i" --num-shards "$n" --video-dir "$src" --output-dir "$out" \
      > "logs/cut_ab_s3d_${arm}_${i}.log" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  echo "   $arm: $(ls "$out"/*.npz 2>/dev/null | wc -l) feature files"
done

step "Alg. 1 on both arms, identical parameters"
for arm in blind content; do
  seg="$WORK/${arm}_segmentation.json"
  # "Exists, skip" is not safe here.  A segmentation left by an earlier, smaller
  # run of this script looks identical to a finished one, and reusing it would
  # compare 165 blind sequences against 322 content ones and print a confident
  # table.  Skip only when the segmentation covers every feature file present.
  if [ -s "$seg" ]; then
    have="$(ls "$WORK/${arm}_s3d"/*.npz 2>/dev/null | wc -l)"
    covered="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['sequences'])" "$seg" 2>/dev/null || echo 0)"
    if [ "$covered" = "$have" ]; then
      echo "   $seg covers all $have sequences, skipping"; continue
    fi
    echo "   $seg covers $covered of $have sequences; redoing"
    rm -f "$seg"
  fi
  python3 tools/segment_visual_atomics.py --features-dir "$WORK/${arm}_s3d" \
    --output "$seg" --frames-per-cluster "$FRAMES_PER_CLUSTER" \
    --min-length "$MIN_LENGTH" --index-weight "$INDEX_WEIGHT" \
    > "logs/cut_ab_seg_${arm}.log" 2>&1 || exit 1
  echo "   $arm -> $seg"
done

step "compare"
python3 tools/compare_cut_ab.py --blind "$WORK/blind_segmentation.json" \
  --content "$WORK/content_segmentation.json" --ingest-root "$INGEST" \
  --output "$WORK/comparison.json" 2>&1 | tee logs/cut_ab_compare.log
