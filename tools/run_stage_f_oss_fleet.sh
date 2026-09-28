#!/usr/bin/env bash
#
# Stage F over the OSS-resident corpus: S3D features for every clip that has 3D.
#
#   bash tools/run_stage_f_oss_fleet.sh              # 7 shards, one per card
#   NUM_SHARDS=14 bash tools/run_stage_f_oss_fleet.sh
#
# The shape is stage B's: fetch a batch to /dev/shm, extract, publish to OSS,
# delete the scratch.  It differs in the batch size, because loading S3D costs
# seconds while one clip's features cost about one -- a per-clip process would
# spend most of its life loading the model.
#
# One shard per card rather than two: S3D is a single forward pass over a
# decoded clip, so the card saturates on one worker and a second only competes
# for the same memory.  Stage B runs two because GVHMR spends much of each clip
# on CPU-side SLAM.
#
# The work list is derived, not frozen: a clip is due iff it has 3D output and
# no .npz yet, which both shards and reruns can compute identically.  Stage B
# needs a freeze because its skip-set changes while it runs; this one only ever
# shrinks.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

NUM_SHARDS="${NUM_SHARDS:-7}"
GPUS="${GPUS:-0 1 2 3 4 5 6}"
BATCH="${BATCH:-50}"
SCRATCH="${ATOMICDANCE_SCRATCH:-/dev/shm/atomicdance-scratch}"
mkdir -p logs "$SCRATCH"

read -ra CARDS <<< "$GPUS"
pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${CARDS[$((shard % ${#CARDS[@]}))]}"
  ATOMICDANCE_SCRATCH="$SCRATCH" \
  python3 tools/run_wild_s3d_shard_oss.py \
      --shard "$shard" --num-shards "$NUM_SHARDS" --gpu "$gpu" --batch "$BATCH" \
      > "logs/stage_f_oss_${shard}.log" 2>&1 &
  pids+=($!)
  echo "shard ${shard} -> gpu ${gpu} pid ${!}"
  sleep 2
done

echo "launched ${#pids[@]} shards; waiting"
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=$((failed + 1)); done
echo "fleet finished, ${failed} shard(s) exited non-zero"

# Count from the store, not from the shards' tallies -- the same reason stage B
# does: a summary that cannot be cross-checked gets believed when it is wrong.
python3 - <<'PY'
import sys
sys.path.insert(0, ".")
import tools.run_gvhmr_ingest_shard_oss as stage_b
import tools.run_wild_s3d_shard_oss as stage_f
from tools import asset_io

have_3d = set(stage_b.stems_with(stage_b.CONVERTED_ROOT, "quality.json"))
features = {name.split("/")[-1][:-4] for name in asset_io.list_prefix(stage_f.FEATURES_ROOT)
            if name.endswith(".npz")}
print("with_3d={} with_features={} outstanding={}".format(
    len(have_3d), len(features), len(have_3d - features)))
PY
