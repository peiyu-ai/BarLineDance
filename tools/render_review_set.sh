#!/usr/bin/env bash
# Render one review set: the same clips, the same stage, one row per arm.
#
# Why a script and not a loop in a shell history.  Every earlier review set was
# built by hand and none of them recorded WHICH arm each row came from, so a
# page that showed a fix working could not be told from one that showed the
# baseline twice.  The arm list is the argument here and it is echoed into the
# page's note, so the mapping travels with the picture.
#
# Rows share a stage on purpose -- see render_avatar_video.py's header: separate
# renders give each arm its own metres-per-pixel, and the arm that drifts most
# gets drawn smallest, i.e. the worse model looks calmer.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?usage: render_review_set.sh <output-dir> <clip-list-file> <arm=run-dir> ...}"
CLIPS="${2:?}"; shift 2
# Overridable, because the T series keeps its own converted tree: a review set
# rendered against the wrong ground-truth root silently stacks another corpus's
# dancer next to this one's generations.  Defaults are the v5 line's, unchanged.
INGEST="${INGEST:-/cache/atomicdance-assets/data/wild_ingest_v1}"
GT="${GT:-/cache/atomicdance-assets/data/wild3d/ingest_v1_converted}"
mkdir -p "$OUT"
while read -r clip; do
  [ -n "$clip" ] || continue
  id="${clip#wild_v5:}"; id="${id/:/__}"
  wav="$INGEST/$id/audio.wav"
  [ -f "$wav" ] || { echo "skip $clip (no audio)"; continue; }
  args=(--motion "ground truth:$GT/$id")
  for spec in "$@"; do
    args+=(--motion "${spec%%=*}:$REPO/runs/${spec#*=}/$clip.pkl")
  done
  python3 "$REPO/tools/render_avatar_video.py" "${args[@]}" \
    --audio "$wav" --output "$OUT/$id.mp4" \
    --vrm "$REPO/third_party/vrm/anime_female.vrm.glb" --view front </dev/null
  # </dev/null is load-bearing: without it the renderer reads the clip list this
  # loop is reading from, and the next `read` gets half a line -- the second
  # clip came back as "7631322618424308849:clip000", its "wild_v5:" prefix eaten.
done < "$CLIPS"
