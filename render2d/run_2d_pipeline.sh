#!/usr/bin/env bash
# The whole 2D stage for the fixed ten clips, one arm.
#
#   3D motion -> 2D COCO-18 pose -> pose video -> (Wan Animate) -> mux audio
#
# Stages 1-3 need no weights and run anywhere; stage 4 needs a local Wan Animate
# checkpoint, which this machine cannot download (pypi answers, huggingface.co /
# cdn-lfs / modelscope / github's API do not).  With --dry-run the driver stops
# after stage 3 and reports the plan, so the parts that DO run here are still
# exercised end to end on all ten clips.
set -euo pipefail
cd "$(dirname "$0")/.."

ARM="${1:?usage: run_2d_pipeline.sh <arm-dir> <out-dir> [--dry-run|--model DIR]}"
OUT="${2:?}"
shift 2
MODE="${1:---dry-run}"
MODEL="${2:-/nonexistent}"

CLIPS=runs/vis_clips_t10.txt
# 2026-09-22: defaults moved to the T line's current version (configs/t_line/current.*): the T2
# shipped 3D arm and the re-decoded audio overlay (T1's audio.wav is up to 161 ms early).
# fix7 / T1 = ARM=/cache/atomicdance-assets/runs/t_beat/fix7 INGEST=/cache/atomicdance-assets/data/wild_ingest_v1
INGEST="${INGEST:-/cache/atomicdance-assets/data/wild_ingest_txy_t2_audiofix}"
mkdir -p "$OUT"

while read -r clip; do
  [ -n "$clip" ] || continue
  id="${clip#wild_v5:}"; id="${id%%:*}"; part="${clip##*:}"
  stem="${id}__${part}"
  motion="$ARM/${clip}.pkl"
  [ -f "$motion" ] || { echo "MISSING motion $motion"; exit 1; }
  work="$OUT/work/$stem"
  mkdir -p "$work"

  python3 render2d/project_pose_2d.py --motion "$motion" --out "$work/pose"
  python3 render2d/draw_pose_video.py --pose-dir "$work/pose" \
      --out "$work/pose.mp4" --audio "$INGEST/$stem/audio.wav"
  # ONE character for all ten clips, not one per clip: the demo is "this
  # character dancing these ten dances", and a different drawing per clip would
  # make the ten incomparable -- the same reason the fixed ten clips exist.
  if [ ! -f "$OUT/character.png" ]; then
    python3 render2d/make_character_image.py --motion "$motion" \
        --out "$OUT/character.png"
  fi

  if [ "$MODE" = "--dry-run" ]; then
    python3 render2d/animate_2d.py --character "$OUT/character.png" \
        --pose-video "$work/pose.mp4" --out "$OUT/$stem.mp4" \
        --model "$MODEL" --dry-run
  else
    python3 render2d/animate_2d.py --character "$OUT/character.png" \
        --pose-video "$work/pose.mp4" --audio "$INGEST/$stem/audio.wav" \
        --out "$OUT/$stem.mp4" --model "$MODEL"
  fi
  echo "  $stem done"
done < "$CLIPS"
echo "2D stage finished -> $OUT"
