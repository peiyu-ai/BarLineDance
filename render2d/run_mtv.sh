#!/usr/bin/env bash
# The ten test clips as 3D-pose-driven video: SMPL joints -> MTV-Crafter -> Wan2.1 I2V.
#
# The 2D path draws the dance as a flat skeleton and an orthographic projection
# cannot carry movement along the view axis -- measured, the wrists keep 80% of
# their motion and the body 77%.  This path conditions on the joint COORDINATES,
# so the depth survives.  One character for all ten, the same fixed ten clips,
# so every round is comparable frame for frame.
#
# SETTINGS, each one paid for on 818 (2026-09-17, output/sample_20260917_mtv/smoke):
#   * MTV-finetuned base, no adapter -- comfy_mtv.assert_motion_path;
#   * wrapper RoPE patch (render2d/patches/) -- without it the motion attention
#     is cos-only and the character barely moves;
#   * 49-frame context windows -- upstream trains on 49 frames and the key layout
#     depends on the token count; 81-frame windows followed at shape 0.083 vs
#     null 0.095, 49-frame at 0.049-0.053 vs 0.073-0.075 (a 3D render that
#     follows by construction reads 0.035 vs 0.110).  Overlap 16: 24/16/8 read
#     0.061/0.064/0.064, i.e. the overlap is not what costs the motion, and 16
#     leaves no seam on the contact sheet.  NOT chunk chaining -- it follows
#     better and loses the character (render2d/mtv_chain.py);
#   * motion guidance 2.0 (needs the wrapper patch) -- the one change that made
#     the dance arrive: on 818's first window, shape 0.048 against null 0.101
#     where strength alone read 0.053/0.073, and the amplitude matches the pose.
#     3.0 buys position but no shape and costs face confidence;
#   * strength 1.5 -- with guidance on it matches the pose's own amplitude (0.026 vs
#     0.027) where 1.2 read 0.021; 2.0 overshoots it by 50%,
#     which is what tore the faces and hands in the first batch;
#   * 10 steps -- 6 leaves melted hands on the fast clip, 16 adds nothing;
#   * CLIP embeds on, no noise on the reference -- as upstream.
# Each clip is scored by tools/score_mtv_follow.py against the joints it was
# conditioned on, and composited next to the sample strip of the same arm, so the
# four panels the strip carries sit beside the MTV panel.
set -uo pipefail
cd "$(dirname "$0")/.."

# 2026-09-22: defaults moved to the T line's current version (configs/t_line/current.*): the T2
# shipped 3D arm and the re-decoded audio overlay (T1's audio.wav is up to 161 ms early).
# fix7 / T1 = ARM=/cache/atomicdance-assets/runs/t_beat/fix7 INGEST=/cache/atomicdance-assets/data/wild_ingest_v1
ARM="${1:-/cache/atomicdance-assets/runs/t_beat/t2_ship_vis10}"
OUT="${2:-output/sample_20260918_mtv}"
STRIP="${3:-output/sample_20260922_t2_shipped}"
CHARACTER="${4:-townfair_fit.png}"
CLIPS=runs/vis_clips_t10.txt
INGEST="${INGEST:-/cache/atomicdance-assets/data/wild_ingest_txy_t2_audiofix}"
mkdir -p "$OUT/compare"

export MTV_CONTEXT="${MTV_CONTEXT:-49,16}" MTV_STRENGTH="${MTV_STRENGTH:-1.5}"
export MTV_MOTION_CFG="${MTV_MOTION_CFG:-2.0}" MTV_STEPS="${MTV_STEPS:-10}"
export MTV_ADAPTER="${MTV_ADAPTER:-none}" MTV_CLIP="${MTV_CLIP:-1}" MTV_NOISE_AUG="${MTV_NOISE_AUG:-0.0}"

while read -r clip; do
  [ -n "$clip" ] || continue
  id="${clip#wild_v5:}"; id="${id%%:*}"; part="${clip##*:}"; stem="${id}__${part}"
  work="$OUT/work/$stem"
  mkdir -p "$work"
  # </dev/null on every python call: without it the interpreter eats this loop's
  # stdin and the clip names arrive with their first characters missing.
  [ -f "$work/joints3d.npy" ] || python3 render2d/mtv_motion.py \
      --motion "$ARM/${clip}.pkl" --out "$work/joints3d.npy" </dev/null
  if [ -f "$OUT/$stem.mp4" ]; then
    echo "  $stem already rendered"
  else
    python3 render2d/comfy_mtv.py --character "$CHARACTER" \
        --joints "$work/joints3d.npy" --audio "$INGEST/$stem/audio.wav" \
        --out "$OUT/$stem.mp4" </dev/null || { echo "  $stem FAILED"; continue; }
  fi
  [ -f "$work/follow.json" ] || python3 tools/score_mtv_follow.py --video "$OUT/$stem.mp4" \
      --joints "$work/joints3d.npy" --stride 4 --device CUDAExecutionProvider \
      --out "$work/follow.json" </dev/null 2>&1 | grep -v -i warn | tail -1
  if [ -f "$STRIP/$stem.mp4" ] && [ ! -f "$OUT/compare/$stem.mp4" ]; then
    # The strip at its own height, the MTV panel scaled to match, the strip's audio.
    height=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$STRIP/$stem.mp4")
    ffmpeg -v error -y -i "$STRIP/$stem.mp4" -i "$OUT/$stem.mp4" -filter_complex \
        "[1:v]fps=30,scale=-2:${height}[m];[0:v][m]hstack=inputs=2:shortest=1[v]" \
        -map "[v]" -map 0:a? -c:v libx264 -crf 20 -pix_fmt yuv420p -c:a copy \
        "$OUT/compare/$stem.mp4" </dev/null
  fi
  echo "  $stem done"
done < "$CLIPS"
echo "MTV stage finished -> $OUT"
