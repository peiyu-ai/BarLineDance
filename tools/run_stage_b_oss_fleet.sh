#!/usr/bin/env bash
#
# Stage B over the OSS-resident corpus: 14 shards, two per card across GPUs 0-6.
#
#   bash tools/run_stage_b_oss_fleet.sh            # resume the frozen todo list
#   FREEZE=1 bash tools/run_stage_b_oss_fleet.sh   # recompute it first
#
# Nothing here writes to the NAS.  Each shard fetches one clip to /dev/shm,
# runs GVHMR, publishes the result to OSS and deletes the scratch, so the disk
# footprint is about 15 MB per shard no matter how large the corpus grows --
# which is the property the previous fleet did not have, and the reason a 438 GB
# ingest tree could fill a 1 TB quota while everything looked healthy.
#
# Two per card rather than three: the previous fleet ran 21 workers over seven
# cards and its recovery from a wedge was to leave a dead worker holding a CUDA
# context.  Fifty of those contexts had to be reaped by hand afterwards.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

NUM_SHARDS="${NUM_SHARDS:-14}"
GPUS="${GPUS:-0 1 2 3 4 5 6}"
SCRATCH="${ATOMICDANCE_SCRATCH:-/dev/shm/atomicdance-scratch}"
# RETRY_FAILED=1 makes the shards ignore .extract_failed markers.  It exists
# because those markers cannot be deleted: the credentials this project has are
# write-only against the bucket (403 AccessDenied on rm), so a marker written in
# error is permanent and the only way past it is to not consult it.
#
# That happened on 2026-08-13: a pod rebuild dropped GVHMR's pip dependencies,
# every clip failed with an ImportError the shard could not see, and twelve
# shards wrote ~450 markers before the run was stopped.  Pair this flag with the
# *existing* frozen todo list rather than re-freezing -- the frozen list already
# excludes the genuine failures from before that run, so the pair retries the
# bad markers and leaves the clips whose visual odometry truly diverged alone.
RETRY_FAILED="${RETRY_FAILED:-}"
retry_arg=()
[ -n "$RETRY_FAILED" ] && retry_arg=(--retry-failed)
mkdir -p logs "$SCRATCH"

if [ -n "${FREEZE:-}" ]; then
  # RETRY_FAILED must reach the freeze too.  Without it the freeze subtracts
  # every marker-carrying clip, so FREEZE=1 RETRY_FAILED=1 hands the shards a
  # list with nothing to retry -- and it overwrites the previous list, which
  # was the recovery input.
  python3 tools/run_gvhmr_ingest_shard_oss.py --freeze --num-shards "$NUM_SHARDS" \
      "${retry_arg[@]}" || exit 1
fi

read -ra CARDS <<< "$GPUS"
pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${CARDS[$((shard % ${#CARDS[@]}))]}"
  ATOMICDANCE_SCRATCH="$SCRATCH" \
  python3 tools/run_gvhmr_ingest_shard_oss.py \
      --shard "$shard" --num-shards "$NUM_SHARDS" --gpu "$gpu" "${retry_arg[@]}" \
      > "logs/stage_b_oss_${shard}.log" 2>&1 &
  pids+=($!)
  echo "shard ${shard} -> gpu ${gpu} pid ${!}"
  sleep 2                       # stagger the model loads off one another
done

echo "launched ${#pids[@]} shards; waiting"
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=$((failed + 1)); done
echo "fleet finished, ${failed} shard(s) exited non-zero"

# Report from the artifacts, not from the shards' own tallies: the previous
# run's SHARD_DONE lines were written by a bash loop whose script had been
# edited underneath it, and a summary that cannot be cross-checked is a summary
# that gets believed when it is wrong.
python3 - <<'PY'
import sys
sys.path.insert(0, ".")
import tools.run_gvhmr_ingest_shard_oss as shard
from tools import asset_io

todo = asset_io.read_json(shard.TODO_LIST)["clips"]
done = shard.stems_with(shard.CONVERTED_ROOT, "quality.json")
failed = shard.stems_with(shard.RAW_ROOT, ".extract_failed")
left = [clip for clip in todo if clip not in done and clip not in failed]
print("converted={} failed={} still_outstanding={}".format(len(done), len(failed), len(left)))
PY
