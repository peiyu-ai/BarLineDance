#!/usr/bin/env bash
# Rebuild the training release the shipped checkpoints were trained on.
#
# WHY IT HAD TO BE REBUILT.  It lived at /dev/shm/atomicdance-song-v5rekey and
# tmpfs was cleared, so 38 GB of it is gone.  Nothing was lost that cannot be
# recomputed -- every ingredient is a published store -- but the rebuild has to
# land on the SAME bytes or the checkpoints are being run on a corpus they were
# not fitted to.  The check for that is at the end: the normalizer's sha256 is
# recorded in every old manifest, so reproducing it proves the chain.
#
# It goes to /cache (CPFS), not to the working tree: the NAS is under a
# directory quota and this is ~100 GB across the three stages (CLAUDE.md 1.2).
set -euo pipefail

ROOT="${ROOT:-/cache/atomicdance-assets/scratch/v5rekey}"
V1="$ROOT/release_v1"
V2="$ROOT/release_v2_robust"
V3="$ROOT/release_v3_timebase"
# From runs/w6vis_seam04i/manifest.json -> dataset_provenance.normalizer.sha256
EXPECTED_NORMALIZER="b29aa832790522956d769748fdbb4f101060b75bd659cfa7c292cac812624975"

if [ ! -d "$V1" ]; then
  echo "=== 1/3 materialize windows ==="
  python3 tools/materialize_atomic_windows.py \
      --sources data/wild3d/wild_v5_song_performance/sources.jsonl \
      --sequences data/wild3d/wild_v5_song_normalized/sequences_normalized.jsonl \
      --labels data/wild3d/wild_v5_song_ingroup_llm_v5rekey/labels.jsonl \
      --output-dir "$V1" \
      --window-length 150 --window-stride 15 --min-label-valid-fraction 1.0 \
      --num-classes 4529
fi

if [ ! -d "$V2" ]; then
  echo "=== 2/3 robust re-normalisation (this is what moved root/GT 0.238 -> 1.01) ==="
  python3 tools/renormalize_release.py --source "$V1" --output "$V2"
fi

if [ ! -d "$V3" ]; then
  echo "=== 3/3 drop the timebase-damaged windows (13.4% of the corpus) ==="
  python3 tools/filter_release_windows.py --source "$V2" --output "$V3" \
      --exclude runs/timebase_exclude_v1.jsonl
fi

echo "=== provenance check ==="
python3 - "$V3" "$EXPECTED_NORMALIZER" <<'PY'
import hashlib, json, pathlib, sys

release, expected = pathlib.Path(sys.argv[1]), sys.argv[2]
digest = hashlib.sha256((release / "normalizer.pt").read_bytes()).hexdigest()
build = json.loads((release / "build.json").read_text())
counts = {split: build.get("counts", {}).get(split) for split in ("train", "val", "test")}
print("normalizer sha256 : {}".format(digest))
print("expected          : {}".format(expected))
print("windows           : {}".format(counts))
if digest != expected:
    # Not fatal on its own -- the A/B below runs both arms on whatever this is,
    # so the comparison stays valid -- but it means the historical numbers
    # (energy 1.179, settle -0.0769, skate 1.155) may not be quotable beside
    # these, and that has to be said rather than assumed.
    print("MISMATCH: this is not byte-identical to the release the shipped "
          "checkpoints were trained on; A/B remains valid, cross-run "
          "comparisons to older tables do not.")
else:
    print("MATCH: same normalizer as the shipped checkpoints were trained with.")
PY
echo "RELEASE=$V3"
