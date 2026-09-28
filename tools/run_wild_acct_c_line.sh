#!/usr/bin/env bash
#
# Line C on a re-split wild corpus: everything the re-split forces, from the
# summarizing LLM's shards through to a trained planner.
#
# ROOT and TAG are what pick the corpus, so the same steps serve the
# account-disjoint line (the default) and the song-disjoint one:
#
#   ROOT=/dev/shm/atomicdance-song TAG=wild_v5_song \
#     RELEASE=/dev/shm/atomicdance-song/release_v1 FROM=3 \
#     bash tools/run_wild_acct_c_line.sh
#
# ROOT holds labels/, performance/, normalized/, captions.jsonl,
# group_keys.json and tmr_embeddings.npz; steps 0-2 additionally write
# summarise_cells/ and subprototypes_shard*.json into it.
#
#   0 wait        the seven M3b shards, by their own output files
#   1 merge       one grouping from every shard's cells, no model in the room
#   2 M3c         re-cluster the segments onto that grouping
#   3 release     window the normalized motion and attach those labels
#   4 audit       the release contract
#   5 planner     compute-matched to the AIST arm, not epoch-matched
#   6 completion  the second stage, on the same release
#
#   bash tools/run_wild_acct_c_line.sh          # all seven, skipping what exists
#   FROM=3 bash tools/run_wild_acct_c_line.sh   # start at the release
#
# Why the split forced steps 1-3 to be re-run rather than inherited:
# `materialize_atomic_windows` binds the normalizer's fit report and every
# label row to the sha256 of the exact `sources.jsonl` handed to it, so a new
# split is a new source manifest and both the normalizer and the vocabulary
# stop binding.  That is a gate, not a convention -- see the 2026-08-16 worklog
# entry, and `run_aist_song_disjoint_line.sh`, which records the same cascade.
#
# The card matters: GPU 7 is held by another project's `hold_gpu.py`, so the
# fleet here is 0-6 and training takes card 0 once the LLM releases it.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ROOT="${ROOT:-/dev/shm/atomicdance-acct}"
TAG="${TAG:-wild_v4_acct}"
FROM="${FROM:-0}"
SHARDS="${SHARDS:-7}"
GPU="${GPU:-0}"
MODEL="third_party/QwenVL/Qwen3-VL-30B-A3B-Instruct/main"
QWEN_PY="third_party/QwenVL/.venv-qwen3vl/bin/python"

# 227 is inherited verbatim from the previous wild flat run, and deliberately:
# on the --subprototypes path it does not set the class count at all (that is
# the LLM's own grouping), it only governs the keyframe fallback.  Holding it
# fixed keeps the split as the single changed variable between the two flat
# vocabularies.  The 2026-08-15 argument that 32 is the better-justified value
# stands, and applies to the w/o-LLM arm, which this is not.
TARGET_SIZE="${TARGET_SIZE:-227}"

SUBPROTOTYPES="${ROOT}/subprototypes_llm.json"
INGROUP="${ROOT}/ingroup_llm"
RELEASE="${RELEASE:-/dev/shm/atomicdance-acct/release_v1}"
AUDIT="data/wild3d/reports/${TAG}_release_audit.json"
PLANNER_DIR="runs/planner_${TAG}_x0"
COMPLETION_DIR="runs/completion_${TAG}"
mkdir -p logs "$(dirname "$AUDIT")"

step() { echo "== $* == $(date +%H:%M:%S)"; }

if [ "$FROM" -le 0 ]; then
  step "0 wait for ${SHARDS} summarizer shard(s)"
  # Waited on by *output file*, not by pgrep: `pgrep -f <name>` matches any
  # command line containing the text -- a tail, an editor, this script's own
  # shell -- and a self-match makes the loop condition permanently true.  That
  # cost this repo a parked handoff on 2026-08-13.  A shard that dies without
  # writing its file also has to fail here rather than be waited on forever,
  # which is what the process check below is for.
  while :; do
    done_count=0
    for s in $(seq 0 $((SHARDS - 1))); do
      [ -s "${ROOT}/subprototypes_shard${s}.json" ] && done_count=$((done_count + 1))
    done
    [ "$done_count" -eq "$SHARDS" ] && break
    alive=$(pgrep -cf "summarize_subprototypes_llm[.]py --captions" || true)
    if [ "$alive" -eq 0 ]; then
      echo "error: ${done_count}/${SHARDS} shard file(s) written and no summarizer is running"
      for s in $(seq 0 $((SHARDS - 1))); do
        [ -s "${ROOT}/subprototypes_shard${s}.json" ] || echo "  missing shard ${s}: see logs/wild_acct_m3b_${s}.log"
      done
      exit 1
    fi
    sleep 60
  done
  echo "  all ${SHARDS} shards finished"
fi

if [ "$FROM" -le 1 ] && [ ! -f "$SUBPROTOTYPES" ]; then
  step "1 merge the shards' cells"
  MERGED="${ROOT}/summarise_cells/all.jsonl"
  cat "${ROOT}"/summarise_cells/shard*.jsonl > "$MERGED" || exit 1
  echo "  $(wc -l < "$MERGED") cell(s) merged"
  # --merge-only refuses rather than fills a gap: a cell missing because its
  # shard died would become a prototype with no sub-prototypes, and recluster
  # would quietly fall back to the keyframe criterion and report a clean run.
  "$QWEN_PY" tools/summarize_subprototypes_llm.py \
      --captions "${ROOT}/captions.jsonl" --model "$MODEL" \
      --output "$SUBPROTOTYPES" --checkpoint "$MERGED" \
      --max-prompt-captions 80 --max-rounds 80 --merge-only \
      > "logs/${TAG}_m3b_merge.log" 2>&1 || { tail -5 "logs/${TAG}_m3b_merge.log"; exit 1; }
  python3 -c "
import json; d=json.load(open('${SUBPROTOTYPES}'))
print('  {} sub-prototype(s) over {} cell(s); degenerate={}'.format(
    d['total_subprototypes'], d['groups'], d['grouping_is_degenerate']))"
fi

if [ "$FROM" -le 2 ] && [ ! -e "$INGROUP" ]; then
  step "2 M3c: re-cluster onto the LLM's grouping"
  python3 tools/recluster_atomics_ingroup.py \
      --labels "${ROOT}/labels" --bundle "${ROOT}/performance" \
      --output-dir "$INGROUP" --embedding-cache "${ROOT}/tmr_embeddings.npz" \
      --captions "${ROOT}/captions.jsonl" --group-keys "${ROOT}/group_keys.json" \
      --target-size "$TARGET_SIZE" --min-retrieval-groups 2 --seed 20260816 \
      --subprototypes "$SUBPROTOTYPES" \
      > "logs/${TAG}_m3c.log" 2>&1 || { tail -20 "logs/${TAG}_m3c.log"; exit 1; }
fi

# Read outside the guards, so an absent or renamed report cannot leave SUB empty
# and turn --num-classes into the literal 1 from $((SUB + 1)) -- a release built
# with one class, published without complaint.
SUB=$(python3 -c "import json;print(json.load(open('${INGROUP}/report.json'))['recluster']['total_subprototypes'])") || {
  echo "error: cannot read total_subprototypes from ${INGROUP}/report.json"; exit 1; }
case "$SUB" in
  ''|*[!0-9]*) echo "error: total_subprototypes is not a positive integer: '${SUB}'"; exit 1 ;;
esac
FALLBACK=$(python3 -c "import json;print(json.load(open('${INGROUP}/report.json'))['recluster'].get('segments_placed_by_keyframe_fallback','?'))")
echo "  ${SUB} sub-prototypes; ${FALLBACK} segment(s) placed by the keyframe fallback"

if [ "$FROM" -le 3 ] && [ ! -e "$RELEASE" ]; then
  step "3 materialize the training release"
  python3 tools/materialize_atomic_windows.py \
      --sources "${ROOT}/performance/sources.jsonl" \
      --sequences "${ROOT}/normalized/sequences_normalized.jsonl" \
      --labels "${INGROUP}/labels.jsonl" --output-dir "$RELEASE" \
      --window-length 150 --window-stride 15 --min-label-valid-fraction 1.0 \
      --num-classes "$((SUB + 1))" \
      > "logs/${TAG}_release.log" 2>&1 || { tail -20 "logs/${TAG}_release.log"; exit 1; }
fi

if [ "$FROM" -le 4 ]; then
  step "4 audit the release contract"
  # --require-song-disjoint-splits is deliberately NOT passed: the audit's song
  # parser is AIST's, so on this corpus it resolves nothing and the flag would
  # buy a vacuous pass.  Disjointness is proven upstream by whichever splitter
  # built ROOT, and both of them can fail -- assign_account_disjoint_split for
  # the account line, assign_wild_song_split for the song one, the latter also
  # refusing when any upload has clips on two sides.  A check that cannot fail
  # is worse than no check (CLAUDE.md 2).
  python3 tools/audit_atomic_dataset.py \
      --data-root "$RELEASE" --eval-source-list '' --window-stride 15 \
      --min-overlap-label-agreement 1.0 \
      --safe-retrieval-null-permutations 199 \
      --output "$AUDIT" > "logs/${TAG}_audit.log" 2>&1
  status=$?
  python3 -c "
import json; d=json.load(open('${AUDIT}'))
print('  valid:', d['valid'])
for e in d['errors']: print('   error:', e)
for w in d['warnings'][:10]: print('   warn :', w)
c=d.get('song_disjointness_audit',{}).get('coverage',{})
print('   song-id coverage (expected 0 on this corpus):',
      {k: v.get('fraction') for k, v in c.items()})"
  [ "$status" -eq 0 ] || { echo "audit failed; not training"; exit 1; }
fi

# Compute-matched, not epoch-matched.  The AIST 630 arm ran 14,409 windows at
# batch 64 for 600 epochs = 135,600 optimizer steps.  This corpus is ~19x
# larger, so 600 epochs here would be 19x the compute and a different
# experiment; EPOCHS is derived from the release's own train size so the number
# cannot go stale when the release does.
# The first-difference term EDGE has and this completion did not.  Without it,
# fed the ground truth noised to t=0 -- asked to reproduce an essentially clean
# input -- the model returns a root path 397% as long, and 41.8% of the root's
# motion energy in a finished clip is high-frequency vibration against the
# ground truth's 2.2%.  1.0 is the EDGE-faithful value; 4.0 is what this corpus
# measured better on every axis, over the 65 M6 clips through the whole
# pipeline:
#
#            root path   root HF   root spikes   pose spikes   pose step
#   truth      4.28 m      2.2%       0.084%        0.300%      0.02130
#   w = 1.0   20.93 m     20.7%       0.694%        0.171%      0.02685
#   w = 4.0   13.09 m     16.5%       0.231%        0.075%      0.02329
#
# Note the last column, because an earlier reading of it was wrong.  Probed with
# an *empty* draft the w=4.0 model returns 89% of the ground truth's pose step,
# which reads as the "smooth by dancing slower" failure this repository has a
# cost gate for.  With the retrieval draft attached -- which is what the
# pipeline actually does -- it is 109%, closer to the truth than w=1.0's 126%.
# Both numbers are real; only the second is about this pipeline.
VELOCITY_WEIGHT="${VELOCITY_WEIGHT:-4.0}"
STEPS="${STEPS:-135600}"
BATCH="${BATCH:-64}"
EPOCHS=$(python3 -c "
import json, math, pathlib
names = json.loads(pathlib.Path('${RELEASE}/train/names.json').read_text())
per_epoch = max(1, math.ceil(len(names) / ${BATCH}))
print(max(1, round(${STEPS} / per_epoch)))") || exit 1
echo "  train windows: $(python3 -c "
import json,pathlib;print(len(json.loads(pathlib.Path('${RELEASE}/train/names.json').read_text())))")  ->  ${EPOCHS} epoch(s) for ~${STEPS} steps"

if [ "$FROM" -le 5 ] && [ ! -e "$PLANNER_DIR" ]; then
  step "5 planner (${EPOCHS} epochs)"
  CUDA_VISIBLE_DEVICES="$GPU" python3 train_atomic.py --stage planner \
      --data-root "$RELEASE" --output-dir "$PLANNER_DIR" \
      --num-classes "$SUB" --epochs "$EPOCHS" --batch-size "$BATCH" \
      --planner-parameterization x0 --seed 20260816 \
      --save-every-epochs $(( EPOCHS / 6 > 0 ? EPOCHS / 6 : 1 )) \
      --log-every-epochs 1 --workers 8 \
      > "logs/${TAG}_planner.log" 2>&1 || { tail -20 "logs/${TAG}_planner.log"; exit 1; }
fi

if [ "$FROM" -le 6 ] && [ ! -e "$COMPLETION_DIR" ]; then
  step "6 completion (${EPOCHS} epochs)"
  CUDA_VISIBLE_DEVICES="$GPU" python3 train_atomic.py --stage completion \
      --data-root "$RELEASE" --output-dir "$COMPLETION_DIR" \
      --num-classes "$SUB" --epochs "$EPOCHS" --batch-size "$BATCH" \
      --seed 20260816 --velocity-weight "$VELOCITY_WEIGHT" \
      --save-every-epochs $(( EPOCHS / 6 > 0 ? EPOCHS / 6 : 1 )) \
      --log-every-epochs 1 --workers 8 \
      > "logs/${TAG}_completion.log" 2>&1 || { tail -20 "logs/${TAG}_completion.log"; exit 1; }
fi

step "line C on the account-disjoint wild corpus is done: --num-classes ${SUB}"
