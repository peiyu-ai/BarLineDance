#!/usr/bin/env bash
# The test clips as 2D cartoon dance videos, one character for all of them,
# each one delivered as FOUR PANELS SIDE BY SIDE:
#
#   [ raw video | 3D pose (this arm) | 2D pose (what the animator reads) | 2D skin ]
#
#   generated 3D motion -> AAPose 2D pose video -> Wan2.1 SteadyDancer -> + audio
#
# WHY FOUR PANELS AND NOT JUST THE CARTOON.  The four are the chain, in order,
# and each neighbouring pair isolates one stage: raw vs 3D pose says whether the
# dance itself is the defect, 3D pose vs 2D pose says whether the projection
# lost it (an orthographic projection cannot carry motion along the view axis),
# 2D pose vs 2D skin says whether the animator followed the pose it was given.
# A defect visible in only one panel names its own stage; that is the whole
# reason to pay for the strip (CLAUDE.md 1.5).  The 3D panel is the stick figure
# from `tools/render_dance_video.py` -- joint positions with no mesh that could
# silently fake geometry -- and the raw panel is cut by `render_sample_strip`'s
# own `footage_for`, so the footage is the one the ingest cut, not a re-derived
# one (frame-count arithmetic is what produced half-speed clips once).
#
# ONE character for all the clips, not one per clip: the demo is "this character
# dancing these dances", and a different drawing per clip would make them
# incomparable -- the same reason the fixed ten clips exist (CLAUDE.md 1.5.7).
#
# THE POSE IS FITTED TO THE CHARACTER, not the character cropped to the pose,
# and it is fitted LANDMARK TO LANDMARK.  Two earlier versions got this wrong in
# two different directions: the first cropped the still (throwing away the
# scene), the second matched the detector's BOX to the keypoints' bounding box
# -- which reads 10-24% too big, because the box runs to the top of the hair
# while AAPose's highest keypoint is an eye.  --match-character now runs the
# workflow's own ViTPose on the still and matches NECK TO ANKLE, the one span
# both sides have that neither a hairstyle nor a raised arm can move.
#
# WHAT IS PARALLEL AND WHAT CANNOT BE.  There is one L20 and ComfyUI runs one
# prompt at a time, so the sampling of N clips is N times one clip no matter how
# it is launched -- the GPU is already at 100% on a single clip, and two servers
# would split the same throughput while risking an OOM that costs more than it
# saves.  What IS parallel here is everything else: the pose video, the footage
# cut and the stick figure for every clip are built at once (PREP_JOBS workers)
# BEFORE the GPU starts, and each clip's four-panel composite runs in the
# background the moment its cartoon lands, so the GPU never waits for ffmpeg.
# Wall clock is therefore the sampling time plus the first clip's prep.
#
# Needs a ComfyUI serving ../ComfyUI_Wan on 127.0.0.1:8188.  The weights stay
# resident between clips, so only the first clip pays the load.
set -euo pipefail
cd "$(dirname "$0")/.."

# 2026-09-22: defaults moved to the T line's current version (configs/t_line/current.*): the T2
# shipped 3D arm and the re-decoded audio overlay (T1's audio.wav is up to 161 ms early).
# fix7 / T1 = ARM=/cache/atomicdance-assets/runs/t_beat/fix7 INGEST=/cache/atomicdance-assets/data/wild_ingest_v1
ARM="${ARM:-${1:-/cache/atomicdance-assets/runs/t_beat/t2_ship_vis10}}"
OUT="${OUT:-${2:-output/sample_20260920_2d}}"
CHARACTER="${CHARACTER:-${3:-townfair_fit.png}}"
CLIPS="${CLIPS:-${4:-runs/vis_clips_t10.txt}}"
CHARACTER_PATH="${E2E_ROOT:-/workspace/e2e}/ComfyUI_Wan/input/$CHARACTER"
INGEST="${INGEST:-/cache/atomicdance-assets/data/wild_ingest_txy_t2_audiofix}"
PREP_JOBS="${PREP_JOBS:-5}"
PANEL_H="${PANEL_H:-640}"          # height every panel is scaled to before hstack
# Extra flags for the two stages an experiment changes, word-split on purpose:
# AAPOSE_EXTRA goes to aapose_video.py (how the pose picture is drawn),
# SD_EXTRA to comfy_steadydancer.py (seed, context windows, prompt, strengths).
# Both are recorded -- the pose side in aapose.mp4.args, the sampler side in
# the sidecar .json the driver writes -- so an arm is known by what it ran.
AAPOSE_EXTRA="${AAPOSE_EXTRA:-}"
SD_EXTRA="${SD_EXTRA:-}"
# Free text cannot ride in SD_EXTRA (it is word-split); the negative-prompt
# addition gets its own variable and is passed as ONE argument.
SD_NEGATIVE_EXTRA="${SD_NEGATIVE_EXTRA:-}"
# The positive prompt likewise (free text, ONE argument); empty = the driver's default POSITIVE_PROMPT.
SD_PROMPT="${SD_PROMPT:-}"
FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
ARM_NAME="$(basename "$ARM")"
mkdir -p "$OUT/compare"

stem_of() { local c="${1#wild_v5:}"; echo "${c%%:*}__${c##*:}"; }

# Stage A, per clip, no GPU: the pose video the animator reads, the raw footage
# and the 3D stick figure.  Everything writes to .part and is renamed, so a
# half-written panel is never mistaken for a finished one by the loop below.
prep_one() {
  local clip="$1" stem work
  stem="$(stem_of "$clip")"
  work="$OUT/work/$stem"
  mkdir -p "$work"
  # </dev/null on every python call: without it the interpreter eats the
  # caller's stdin and the clip names arrive with their first characters missing.
  # Rebuilt whenever what it was built FROM differs, not only when it is
  # missing: reusing an OUT with a new AAPOSE_EXTRA, ARM or CHARACTER used to
  # render the old pose under the new arm's name (review 2026-09-21).
  local want="--motion $ARM/${clip}.pkl --match-character $CHARACTER_PATH --stick-width 5 --fps 16 $AAPOSE_EXTRA"
  if [ ! -f "$work/aapose.mp4" ] || [ "$(cat "$work/aapose.mp4.args" 2>/dev/null)" != "$want" ]; then
    python3 render2d/aapose_video.py \
        --motion "$ARM/${clip}.pkl" --out "$work/aapose.part.mp4" \
        --audio "$INGEST/$stem/audio.wav" \
        --match-character "$CHARACTER_PATH" --stick-width 5 --fps 16 $AAPOSE_EXTRA </dev/null
    mv "$work/aapose.part.mp4" "$work/aapose.mp4"
    [ -f "$work/aapose.part.mp4.yaw.npy" ] && mv "$work/aapose.part.mp4.yaw.npy" "$work/aapose.mp4.yaw.npy"
    echo "$want" >"$work/aapose.mp4.args"
  fi
  if [ ! -f "$work/footage.mp4" ]; then
    # footage_for returns None when the source upload is not on this machine.
    # Refused rather than dropped: the raw panel is the only one that is not a
    # reconstruction, and a strip missing it still plays and still looks like a
    # comparison (render_sample_strip's own header records that shipping twice).
    python3 -c "
import pathlib, sys
sys.path.insert(0, '.')
from tools.render_sample_strip import footage_for
out = footage_for('$stem', pathlib.Path('$work/footage.part.mp4'), 520, pad=False)
if out is None:
    raise SystemExit('no source footage for $stem')
" </dev/null
    mv "$work/footage.part.mp4" "$work/footage.mp4"
  fi
  if [ ! -f "$work/pose3d.mp4" ]; then
    # ONE azimuth, not the renderer's three: three views would cost three
    # quarters of this panel's width, and the panel is here to be compared with
    # the one beside it, not to be read alone.
    #
    # --follow-root, and it is not cosmetic.  With the cube spanning the whole
    # clip's travel the dancer came out 37% of the panel height, i.e. below the
    # size at which "did the arm reach" can be judged at all (CLAUDE.md 1.5.1);
    # following the root in the floor plane puts her at ~68%.  The two panels to
    # its right are camera-fixed on a character who does not leave the frame, so
    # following the root is also what makes the three COMPARABLE.  The vertical
    # axis stays anchored to the floor, so hover, jumps and crouches -- the
    # defects this line keeps finding -- still read.  Travel itself is therefore
    # not visible in this panel and is not judged here.
    #
    # --set-dpi 160: the figure is 4 inches wide, so the renderer's historical
    # dpi 80 is 320 px, and this panel is shown at 640.
    python3 -c "
import sys; sys.argv = sys.argv[1:]
from tools import render_dance_video as R
R.set_views((90,)); R.set_dpi(160); R.main()
" render_dance_video --result "$ARM/${clip}.pkl" --title "${ARM_NAME} · 3D pose" \
      --output "$work/pose3d.part.mp4" --stride 1 --follow-root </dev/null
    mv "$work/pose3d.part.mp4" "$work/pose3d.mp4"
  fi
}

# Stage C, per clip, no GPU: the four panels on one row at one height, the
# ingest's own audio muxed as mp3 (the reviewer's IDE will not play AAC).
compose_one() {
  local clip="$1" stem work seconds
  stem="$(stem_of "$clip")"
  work="$OUT/work/$stem"
  [ -f "$OUT/compare/$stem.mp4" ] && return 0
  # The bound comes from the cartoon, the shortest of the four by construction
  # (the sampler rounds the frame count), and it is passed EXPLICITLY as well as
  # via -shortest: a graph with no length bound is how a 14 s strip once became
  # ten hours of video that never wrote a moov atom.
  seconds=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT/$stem.mp4")
  ffmpeg -v error -y \
    -i "$work/footage.mp4" -i "$work/pose3d.mp4" -i "$work/aapose.mp4" -i "$OUT/$stem.mp4" \
    -i "$INGEST/$stem/audio.wav" -filter_complex "
      [0:v]fps=30,scale=-2:$PANEL_H,setsar=1,drawtext=fontfile=$FONT:text='raw video':
           x=10:y=8:fontsize=26:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=6[a];
      [1:v]fps=30,scale=-2:$PANEL_H,setsar=1[b];
      [2:v]fps=30,scale=-2:$PANEL_H,setsar=1,drawtext=fontfile=$FONT:text='2D pose':
           x=10:y=8:fontsize=26:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=6[c];
      [3:v]fps=30,scale=-2:$PANEL_H,setsar=1,drawtext=fontfile=$FONT:text='2D skin':
           x=10:y=8:fontsize=26:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=6[d];
      [a][b][c][d]hstack=inputs=4:shortest=1[v]" \
    -map "[v]" -map 4:a -c:v libx264 -preset veryfast -crf 21 -pix_fmt yuv420p \
    -c:a libmp3lame -b:a 128k -t "$seconds" -shortest "$OUT/compare/$stem.part.mp4" </dev/null
  mv "$OUT/compare/$stem.part.mp4" "$OUT/compare/$stem.mp4"
  echo "  $stem composed"
}

mapfile -t CLIP_LIST < <(grep -v '^[[:space:]]*$' "$CLIPS")
echo "arm=$ARM  clips=${#CLIP_LIST[@]}  out=$OUT"

echo "prep (pose video, footage, 3D stick) x${PREP_JOBS} ..."
for clip in "${CLIP_LIST[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$PREP_JOBS" ]; do wait -n; done
  prep_one "$clip" >"$OUT/work_prep_$(stem_of "$clip").log" 2>&1 &
done
wait
for clip in "${CLIP_LIST[@]}"; do
  stem="$(stem_of "$clip")"
  for piece in aapose footage pose3d; do
    [ -f "$OUT/work/$stem/$piece.mp4" ] || {
      echo "prep FAILED for $stem ($piece) -- see $OUT/work_prep_$stem.log"; exit 1; }
  done
done
echo "prep done"

for clip in "${CLIP_LIST[@]}"; do
  stem="$(stem_of "$clip")"
  if [ -f "$OUT/$stem.mp4" ]; then
    echo "  $stem already rendered"
  else
    python3 render2d/comfy_steadydancer.py --character "$CHARACTER" \
        --pose-video "$OUT/work/$stem/aapose.mp4" --audio "$INGEST/$stem/audio.wav" \
        --out "$OUT/$stem.mp4" $SD_EXTRA ${SD_NEGATIVE_EXTRA:+--negative-extra "$SD_NEGATIVE_EXTRA"} \
        ${SD_PROMPT:+--prompt "$SD_PROMPT"} ${SD_FACING:+--facing-yaw "$OUT/work/$stem/aapose.mp4.yaw.npy"} </dev/null || { echo "  $stem FAILED"; continue; }
    echo "  $stem sampled"
  fi
  compose_one "$clip" &
done
wait
echo "2D stage finished -> $OUT  (four-panel: $OUT/compare)"
