#!/usr/bin/env bash
#
# Line C on a song-disjoint AIST++: the whole cascade a re-split forces.
#
#   0 split       whole backing tracks to splits, stratified by genre
#   1 fit         the train-only min/max normalizer, on the *new* train
#   2 apply       normalize every sequence with that frozen artifact
#   3 M2          TMR k-means, fitted on the new train segments only
#   4 M3          in-group re-clustering with the genre pre-split
#   5 release     window the normalized motion and attach those labels
#   6 audit       the release contract, song disjointness included
#
#   bash tools/run_aist_song_disjoint_line.sh          # all seven, skipping what exists
#   FROM=3 bash tools/run_aist_song_disjoint_line.sh   # start at M2
#
# Why every step downstream of the split has to re-run, rather than the split
# being patched into the existing release:
#
# * the normalizer is fitted on `split == "train"` rows *only*, so a normalizer
#   fitted on the old train has already seen sequences the new split holds out.
#   Re-using it would leak the held-out distribution into the model input, and
#   nothing downstream reports it -- `fit_source_manifest_sha256` is the only
#   thing that would notice, which is why materialize checks it.
# * `cluster_atomics_tmr.py` fits K-Means *and* the accept quantile on train
#   segments only, for the same reason.  A vocabulary carried over from the old
#   split is a vocabulary fitted on the new test set.
# * the embedding cache is keyed by the content hash of the normalized sequence
#   manifest, so it correctly refuses to serve the old encoding here.  The
#   encoding is re-run rather than argued around; it is ~30 minutes.
#
# The card matters: stage B's fleet holds GPUs 0-6 and the captioner holds 7.
# TMR's motion encoder is ~100 MB, so M2 shares a stage-B card rather than
# waiting for one -- unlike the 58 GB captioner, which must not.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TAG="${TAG:-aist_songsplit}"
FROM="${FROM:-0}"
DEVICE="${DEVICE:-cuda}"
M2_GPU="${M2_GPU:-6}"
BATCH="${BATCH:-32}"
# A vocabulary variant reuses M2 and re-runs M3 onward into its own directories,
# because label bundles publish into a new directory and never in place.
VARIANT="${VARIANT:-}"
MIN_RETRIEVAL_GROUPS="${MIN_RETRIEVAL_GROUPS:-1}"
# Tab. 2's "w/ LLM" row on this split: the summarizing LLM's grouping, read back
# onto the segments.  CAPTIONS must already be re-keyed to *this* vocabulary --
# see tools/rekey_captions_to_labels.py, which exists because the grouping key
# on a caption row records whichever M2 run captioned it, and re-using it
# unchanged groups by a clustering the release was not built from.
CAPTIONS="${CAPTIONS:-}"
SUBPROTOTYPES="${SUBPROTOTYPES:-}"

RAW="data/atomic_aistpp/aist_full_performance_v1"
BUNDLE="data/atomic_aistpp/aist_full_performance_songsplit_v1"
NORMALIZER="data/atomic_aistpp/${TAG}_normalizer_v1"
NORMDIR="data/atomic_aistpp/${TAG}_normalized_v1"
NORMALIZED="${NORMDIR}/sequences_normalized.jsonl"
SEG="runs/aist_v1_seg/segmentation.json"
LABELS="data/atomic_aistpp/${TAG}_labels"
CACHE="runs/${TAG}_tmr_embeddings.npz"
GENRES="runs/aist_v1_genre_map.json"
INGROUP="data/atomic_aistpp/${TAG}${VARIANT}_ingroup_seg"
RELEASE="data/atomic_aistpp/${TAG}${VARIANT}_release_v1"
AUDIT="data/wild3d/reports/${TAG}${VARIANT}_release_audit.json"
mkdir -p logs "$(dirname "$AUDIT")"

step() { echo "== $* =="; }

if [ "$FROM" -le 0 ] && [ ! -e "$BUNDLE" ]; then
  step "0 song-disjoint split"
  python3 tools/assign_song_disjoint_split.py --bundle "$RAW" \
      --output-bundle "$BUNDLE" --report "runs/${TAG}_assignment.json" \
      > "logs/${TAG}_split.log" 2>&1 || exit 1
fi

if [ "$FROM" -le 1 ] && [ ! -e "$NORMALIZER" ]; then
  step "1 fit the train-only normalizer"
  python3 tools/fit_motion_normalizer.py \
      --sequence-manifest "${BUNDLE}/sequences.jsonl" \
      --source-manifest "${BUNDLE}/sources.jsonl" \
      --output-dir "$NORMALIZER" > "logs/${TAG}_normalizer_fit.log" 2>&1 || exit 1
fi

if [ "$FROM" -le 2 ] && [ ! -e "$NORMDIR" ]; then
  step "2 apply it"
  python3 tools/apply_motion_normalizer.py \
      --sequence-manifest "${BUNDLE}/sequences.jsonl" \
      --source-manifest "${BUNDLE}/sources.jsonl" \
      --normalizer-bundle "$NORMALIZER" --output-dir "$NORMDIR" \
      > "logs/${TAG}_normalizer_apply.log" 2>&1 || exit 1
fi
# The apply may already be in flight when this driver is launched; the bundle
# appears atomically, so waiting on the process is what makes FROM=3 safe.
# Anchored at ^ and with the dot bracketed, because `pgrep -f <name>` matches any
# process whose *command line contains* that text -- a watchdog, a log tail, an
# editor, or this script's own shell.  A self-match here does not skew a count,
# it makes the loop condition permanently true and the wait never returns.
# Measured 2026-08-13: 16 matches against 14 real workers, which left a handoff
# parked forever and two pipeline lines idle behind it.  tools/launch_caption_shards.sh
# and tools/launch_parallel_pipeline.sh avoid this with a pidfile plus `kill -0`,
# which is stronger and is what a new launcher should use; anchoring is the form
# available when attaching to a fleet that is already running.
while pgrep -f "^python3 tools/apply_motion_normalizer[.]py" > /dev/null; do sleep 20; done
[ -f "$NORMALIZED" ] || { echo "error: ${NORMALIZED} was never published"; exit 1; }

if [ "$FROM" -le 3 ] && [ ! -e "$LABELS" ]; then
  step "3 M2: TMR k-means on the new train segments"
  CUDA_VISIBLE_DEVICES="$M2_GPU" python3 tools/cluster_atomics_tmr.py \
      --bundle "$BUNDLE" --segmentation "$SEG" --output-dir "$LABELS" \
      --normalized-sequences "$NORMALIZED" --normalizer-bundle "$NORMALIZER" \
      --sources "${BUNDLE}/sources.jsonl" --embedding-cache "$CACHE" \
      --classes 100 --accept-quantile 0.85 --device "$DEVICE" \
      --batch-size "$BATCH" > "logs/${TAG}_m2.log" 2>&1 || {
        tail -5 "logs/${TAG}_m2.log"; exit 1; }
fi

if [ "$FROM" -le 4 ] && [ ! -e "$INGROUP" ]; then
  step "4 M3: in-group re-clustering"
  python3 tools/recluster_atomics_ingroup.py \
      --labels "$LABELS" --bundle "$BUNDLE" --output-dir "$INGROUP" \
      --embedding-cache "$CACHE" --genre-split --target-size 32 --seed 20260810 \
      --min-retrieval-groups "$MIN_RETRIEVAL_GROUPS" \
      ${CAPTIONS:+--captions "$CAPTIONS"} ${SUBPROTOTYPES:+--subprototypes "$SUBPROTOTYPES"} \
      > "logs/${TAG}${VARIANT}_m3.log" 2>&1 || {
        tail -5 "logs/${TAG}${VARIANT}_m3.log"; exit 1; }
fi
# Read outside the guards above, so it must not be allowed to fail quietly:
# the script runs under `set -uo pipefail` without -e, so an absent, truncated
# or renamed report would leave SUB empty and --num-classes would become the
# literal "1" from $((SUB + 1)) -- a release built with one class, published
# without complaint.
SUB=$(python3 -c "import json;print(json.load(open('${INGROUP}/report.json'))['recluster']['total_subprototypes'])") || {
  echo "error: cannot read total_subprototypes from ${INGROUP}/report.json"; exit 1; }
case "$SUB" in
  ''|*[!0-9]*) echo "error: total_subprototypes is not a positive integer: '${SUB}'"; exit 1 ;;
esac
echo "  ${SUB} sub-prototypes"

if [ "$FROM" -le 5 ] && [ ! -e "$RELEASE" ]; then
  step "5 materialize the training release"
  # num_classes is the value-space bound (sub-prototypes + transition 0), read
  # from M3's own report.  train_atomic.py is passed ${SUB}, because the model
  # adds the transition token itself; a constant here would silently add a class
  # that never occurs or push labels out of range.
  python3 tools/materialize_atomic_windows.py \
      --sources "${BUNDLE}/sources.jsonl" --sequences "$NORMALIZED" \
      --labels "${INGROUP}/labels.jsonl" --output-dir "$RELEASE" \
      --window-length 150 --window-stride 15 --min-label-valid-fraction 1.0 \
      --num-classes "$((SUB + 1))" > "logs/${TAG}${VARIANT}_release.log" 2>&1 || {
        tail -5 "logs/${TAG}${VARIANT}_release.log"; exit 1; }
fi

if [ "$FROM" -le 6 ]; then
  step "6 audit, with song disjointness required rather than merely reported"
  python3 tools/audit_atomic_dataset.py \
      --data-root "$RELEASE" --eval-source-list '' --window-stride 15 \
      --min-overlap-label-agreement 1.0 --require-song-disjoint-splits \
      --safe-retrieval-null-permutations 199 \
      --output "$AUDIT" > "logs/${TAG}${VARIANT}_audit.log" 2>&1
  status=$?
  python3 -c "
import json; d=json.load(open('${AUDIT}'))
print('  valid:', d['valid'])
for e in d['errors']: print('   error:', e)
for w in d['warnings']: print('   warn :', w)
"
  [ "$status" -eq 0 ] || exit 1
fi
echo "== song-disjoint line C ready to train: --num-classes ${SUB} =="
