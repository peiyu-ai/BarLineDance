#!/usr/bin/env bash
#
# One pass of the asset migration: push, then verify, and say plainly which
# trees are safe to evict.  Nothing is deleted here -- eviction is a separate
# command that re-runs the census first, so a tree can only be freed after its
# bytes have been counted in OSS.
#
# The two Qwen model directories are deliberately absent: they are already in
# the team store under models/, and tools/setup_qwenvl_env.sh is what restores
# them.  tools/oss_assets.py knows this and would skip them anyway; listing
# them here would only suggest they were forgotten.
#
#   bash tools/push_assets_to_oss.sh
#
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs

# Smallest first: a credential or endpoint fault shows up in seconds rather
# than after 72 GB of a single tree has moved.
TREES=(
  third_party/pytorch3d_compat
  third_party/torch_scatter_compat
  third_party/tram
  third_party/TMR
  third_party/WHAM
  third_party/GVHMR
  data/splits
  data/audio_extraction
  data/aist_visual_s3d
  data/aistpp_official
  data/atomic_aistpp.zip
  data/wild_visual_s3d
  data/atomic_aistpp
  data/aist_videos
  data/wild3d
)

for tree in "${TREES[@]}"; do
  echo
  echo "=== push ${tree} ==="
  python3 tools/oss_assets.py push "${tree}" || echo "push ${tree} returned $?"
done

echo
echo "=== verify all ==="
python3 tools/oss_assets.py verify "${TREES[@]}"
verdict=$?

echo
if [ "$verdict" -eq 0 ]; then
  echo "every tree is complete in OSS; free them with:"
  echo "  python3 tools/oss_assets.py evict ${TREES[*]} --yes"
else
  echo "at least one tree is incomplete -- re-run this script; sync is resumable"
fi
exit "$verdict"
