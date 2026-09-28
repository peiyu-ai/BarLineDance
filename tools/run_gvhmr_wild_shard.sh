#!/usr/bin/env bash
#
# One shard of wild GVHMR extraction: video -> world SMPL-X -> 151-D -> validate.
#
#   SHARD=<n> NUM_SHARDS=<k> GPU=<i> [VIDEO_ROOT=<dir>] [USE_DPVO=1] \
#     bash tools/run_gvhmr_wild_shard.sh
#
# VIDEO_ROOT defaults to the 1000-clip pilot directory that batch #1 used.
# USE_DPVO=1 tracks the camera with DPVO instead of SimpleVO and, because a
# clip only failed for want of a camera track, also retries clips carrying an
# `.extract_failed` marker instead of skipping them.
# Membership is sha256(stem) mod NUM_SHARDS, matching the S3D shard scheme, so
# shards are disjoint and stable across relaunches.  Every stage is skipped
# when its output already exists, so the loop is resumable and shards may even
# overlap without corrupting anything -- only wasting time.  A clip that fails
# any stage is logged (FAIL <clip> <stage>) and the loop moves on; the wild
# corpus tolerates dropped clips, not silent gaps in accounting.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$(pwd)"

: "${SHARD:?set SHARD}"; : "${NUM_SHARDS:?set NUM_SHARDS}"; : "${GPU:?set GPU}"

VIDEO_ROOT="${VIDEO_ROOT:-$REPO/data/wild_videos_pilot}"
USE_DPVO="${USE_DPVO:-}"
RAW_ROOT="$REPO/data/wild3d/gvhmr_raw"
CONVERTED_ROOT="$REPO/data/wild3d/converted"
mkdir -p "$RAW_ROOT" "$CONVERTED_ROOT"

if [ -n "$USE_DPVO" ]; then
  VO_FLAG="--use-dpvo"
  # The compat shims are siblings of GVHMR *in this repo*; ``../`` from inside
  # a cached GVHMR would point at the cache's third_party, which has none.
  VO_PYTHONPATH="$REPO/third_party/torch_scatter_compat:$REPO/third_party/pytorch3d_compat:third-party/DPVO:."
else
  VO_FLAG=""
  VO_PYTHONPATH="$REPO/third_party/pytorch3d_compat:."
fi

# Membership is decided once, up front.  Spawning an interpreter per clip to
# hash a single string cost more than the hashing, and at pool scale that
# overhead is measured in hours across the fan-out.
SHARD_LIST="$(mktemp)"
trap 'rm -f "$SHARD_LIST"' EXIT
python3 - "$VIDEO_ROOT" "$SHARD" "$NUM_SHARDS" > "$SHARD_LIST" <<'PY'
import hashlib
import pathlib
import sys

root, shard, num_shards = pathlib.Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
for video in sorted(root.glob("*.mp4")):
    if int(hashlib.sha256(video.stem.encode()).hexdigest(), 16) % num_shards == shard:
        print(video)
PY

total=0 done_before=0 extracted=0 converted=0 failed=0
# Read the worklist on fd 3: the extractor and converter inherit stdin, and a
# child that consumes it would silently eat the rest of the shard.
while IFS= read -r video <&3; do
  stem="$(basename "$video" .mp4)"
  total=$((total + 1))

  if [ -f "$CONVERTED_ROOT/$stem/quality.json" ]; then
    done_before=$((done_before + 1))
    continue
  fi

  result="$RAW_ROOT/$stem/hmr4d_results.pt"
  if [ ! -f "$result" ]; then
    # A clip that already failed extraction fails again the same way under the
    # same tracker, so skip it instead of re-paying VO on every sweep.  Delete
    # the marker, or run with USE_DPVO=1, to force a retry: switching tracker
    # is exactly the extractor change the marker is waiting for.
    if [ -f "$RAW_ROOT/$stem/.extract_failed" ] && [ -z "$USE_DPVO" ]; then
      echo "SKIP_FAILED $stem"; failed=$((failed + 1)); continue
    fi
    # Absolute paths, not ``../..``: third_party/GVHMR can be a symlink into
    # the asset cache, and ``cd`` through it lands somewhere whose parent is
    # not this repo.  A relative hop upward then resolves to a directory that
    # simply has no tools/ in it, and every clip fails at the interpreter.
    (cd third_party/GVHMR &&
      PYTHONPATH="$VO_PYTHONPATH" CUDA_VISIBLE_DEVICES=$GPU \
      python "$REPO/tools/run_gvhmr_extract.py" \
        --video "$video" --output-root "$RAW_ROOT" $VO_FLAG)
    if [ ! -f "$result" ]; then
      mkdir -p "$RAW_ROOT/$stem"
      touch "$RAW_ROOT/$stem/.extract_failed"
      echo "FAIL $stem extract"; failed=$((failed + 1)); continue
    fi
    # Clear the marker once a clip actually extracts, or a recovered clip would
    # keep reading as a failure in every later count.
    rm -f "$RAW_ROOT/$stem/.extract_failed"
    extracted=$((extracted + 1))
  fi

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
done 3< "$SHARD_LIST"

echo "SHARD_DONE shard=$SHARD total=$total done_before=$done_before extracted=$extracted converted=$converted failed=$failed"
