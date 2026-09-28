#!/usr/bin/env bash
#
# Line C: from the M2 labels that bind to the normalized bundle, to a training
# release the planner can consume.
#
#   1 M3        in-group re-clustering, genre pre-split, segment-unit samples
#   2 release   window the normalized motion and attach those labels
#   3 audit     the release contract, before anything trains on it
#
#   bash tools/run_aist_c_line.sh              # waits for M2 if it is running
#   FROM=2 bash tools/run_aist_c_line.sh       # start at the release
#
# Why the labels had to be rebuilt at all: M1/M2/M3 first ran against the raw
# performance bundle, and every label row records the sha256 of the motion
# array it was computed from.  materialize_atomic_windows refuses to attach
# those labels to the normalized array -- correctly, since that check exists so
# that labels cannot be pinned onto a different array -- and the training
# contract requires normalized motion.  cluster_atomics_tmr solves it directly:
# --normalized-sequences binds the labels to the normalized manifest while
# --normalizer-bundle inverts the scaling before forward kinematics, because a
# min-max scaled 151-D row has no orthonormalisable rot6d and its root
# translation is in [-1,1] units, so encoding it would describe a body that
# does not exist.
#
# num_classes is read from M3's own report rather than passed in.  The vocabulary
# size is whatever the re-clustering produced, and a stale constant here would
# either waste a class or push labels out of range -- neither of which raises
# where it happened.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TAG="${TAG:-aist_v1_norm}"
FROM="${FROM:-1}"
LABELS="data/atomic_aistpp/${TAG}_labels"
BUNDLE="data/atomic_aistpp/aist_full_performance_v1"
NORMALIZED="data/atomic_aistpp/aist_full_performance_normalized_v1/sequences_normalized.jsonl"
CACHE="runs/${TAG}_tmr_embeddings.npz"
INGROUP="data/atomic_aistpp/${TAG}_ingroup_seg"
RELEASE="data/atomic_aistpp/${TAG}_release_v1"
AUDIT="data/wild3d/reports/${TAG}_release_audit.json"
mkdir -p logs "$(dirname "$AUDIT")"

# M2 may still be encoding when this is launched; the bundle appears atomically.
# Anchored at ^ and with the dot bracketed, because `pgrep -f <name>` matches any
# process whose *command line contains* that text -- a watchdog, a log tail, an
# editor, or this script's own shell.  A self-match here does not skew a count,
# it makes the loop condition permanently true and the wait never returns.
# Measured 2026-08-13: 16 matches against 14 real workers, which left a handoff
# parked forever and two pipeline lines idle behind it.  tools/launch_caption_shards.sh
# and tools/launch_parallel_pipeline.sh avoid this with a pidfile plus `kill -0`,
# which is stronger and is what a new launcher should use; anchoring is the form
# available when attaching to a fleet that is already running.
while pgrep -f "^python3 tools/cluster_atomics_tmr[.]py" > /dev/null; do sleep 20; done
[ -d "$LABELS" ] || { echo "error: ${LABELS} was never published; see logs/${TAG}_m2.log"; exit 1; }

if [ "$FROM" -le 1 ]; then
  echo "== 1 M3 in-group re-clustering =="
  [ -e "$INGROUP" ] || python3 tools/recluster_atomics_ingroup.py \
      --labels "$LABELS" --bundle "$BUNDLE" --output-dir "$INGROUP" \
      --embedding-cache "$CACHE" --genre-split --target-size 32 --seed 20260810 \
      > "logs/${TAG}_m3.log" 2>&1 || exit 1
  echo "  $(python3 -c "import json;print(json.load(open('${INGROUP}/report.json'))['recluster']['total_subprototypes'])") sub-prototypes"
fi

if [ "$FROM" -le 2 ]; then
  echo "== 2 materialize the training release =="
  CLASSES=$(python3 -c "import json;print(json.load(open('${INGROUP}/report.json'))['recluster']['total_subprototypes'] + 1)")
  echo "  num_classes = ${CLASSES} (sub-prototypes + the transition token)"
  [ -e "$RELEASE" ] || python3 tools/materialize_atomic_windows.py \
      --sources "${BUNDLE}/sources.jsonl" --sequences "$NORMALIZED" \
      --labels "${INGROUP}/labels.jsonl" --output-dir "$RELEASE" \
      --window-length 150 --window-stride 15 --min-label-valid-fraction 1.0 \
      --num-classes "$CLASSES" > "logs/${TAG}_release.log" 2>&1 || {
        tail -3 "logs/${TAG}_release.log"; exit 1; }
  echo "  wrote ${RELEASE}"
fi

if [ "$FROM" -le 3 ]; then
  echo "== 3 audit the release before training reads it =="
  python3 tools/audit_atomic_dataset.py \
      --data-root "$RELEASE" --eval-source-list '' --window-stride 15 \
      --min-overlap-label-agreement 1.0 --min-safe-retrieval-fraction 0.99 \
      --output "$AUDIT" > "logs/${TAG}_audit.log" 2>&1 || {
        tail -5 "logs/${TAG}_audit.log"; exit 1; }
  echo "  wrote ${AUDIT}"
fi
echo "== line C ready to train =="
