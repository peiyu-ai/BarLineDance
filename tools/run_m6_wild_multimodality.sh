#!/usr/bin/env bash
#
# MultiModality for one wild arm: generate the clips the grouping needs that the
# headline selection missed, featurise the union, score it.
#
# This step existed before this script did, and that is why the script exists.
# It was run by hand on 2026-08-16; the only written record is one sentence in
# worklog.md.  The cost of it being ad hoc is on disk and measurable:
#
#     runs/m6_wild_v4_acct/seed_*/manifest.json        plan_stride 15,  vote
#     runs/m6_wild_v4_acct/extra_seed_*/manifest.json  plan_stride 150, none
#
# So 193 of the 320 clips behind that arm's published MM/Div came out of the
# non-overlapping default, which tools/run_m6_wild.sh:55-61 records as putting
# 18.92x of segment boundaries on the window seam against 2.24x, and fid_k
# 10.613 against 9.631.  The number was an average over two generators.  Nothing
# reported it, because nothing knew the two runs were meant to match.
#
# Hence: the sampler flags are arguments here with the same defaults the headline
# driver pins, they are echoed before the run, and the manifests are checked
# against them afterwards rather than trusted.
#
# Why an extra generation pass is needed at all: MultiModality needs several
# dances for one music.  The wild grouping is complete subgraphs of the verified
# fingerprint-pair graph spanning >= 2 uploads -- 144 groups over 320 clips --
# and the headline's 600 clips are drawn by a salted hash, so they cover only
# part of it (127 of 320 for the 600-clip selection).  Generating the remainder
# is what makes the metric definable; it is not a second sample of the headline.
#
#   ARM=wild_v4_acct_w2_340 \
#   RELEASE=/dev/shm/atomicdance-acct/release_w2_340 \
#   CKPT_P=runs/planner_wild_v4_acct_w2_340/planner_step58968.pt \
#   CKPT_C=runs/completion_wild_v4_acct_w2_340/completion_step58968.pt \
#   bash tools/run_m6_wild_multimodality.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ARM="${ARM:?set ARM to the M6 output tag, e.g. wild_v4_acct_w2_340}"
RELEASE="${RELEASE:?set RELEASE}"
CKPT_P="${CKPT_P:?set CKPT_P}"
CKPT_C="${CKPT_C:?set CKPT_C}"
BUNDLE="${BUNDLE:-/dev/shm/atomicdance-acct/performance}"
AUDIO="${AUDIO:-runs/wild_v4_acct_gt_eval/audio}"
PAIRS="${PAIRS:-runs/wild_v4_music_groups_pairs.jsonl}"
SMPL="${SMPL:-third_party/smpl_models/SMPL_MALE_clean.pkl}"
# The extra list is held rather than re-derived.  Its *set* is reproducible
# (grouping members minus the headline selection) but its *line order* is not,
# and order is load-bearing: infer_atomic seeds sample i with ``seed + i`` within
# its shard, so a re-derived list in another order is a different draw.
EXTRA="${EXTRA:-runs/wild_v4_acct_m6_trackgroup_extra.txt}"
SEEDS="${SEEDS:-20260816 20260817 20260818 20260819}"
# Same defaults tools/run_m6_wild.sh pins, for the same measured reason.  They
# are named here so that a run with different ones is visible in the log rather
# than only in a manifest nobody reads.
PLAN_STRIDE="${PLAN_STRIDE:-15}"
PLAN_FUSION="${PLAN_FUSION:-vote}"
GPUS="${GPUS:-1 2 3 4 5 6}"
OUT="runs/m6_${ARM}"
FROM="${FROM:-0}"

mkdir -p logs "$OUT"
step() { echo; echo "== [$(date '+%F %T')] $* =="; }
die()  { echo "MM FAILED: $*" >&2; exit 1; }

for path in "$CKPT_P" "$CKPT_C" "$SMPL" "$RELEASE" "$BUNDLE" "$EXTRA" "$PAIRS" "$AUDIO"; do
  [ -e "$path" ] || die "missing input: $path"
done
[ -d "$OUT/motion" ] || die "$OUT/motion absent -- run tools/run_m6_wild.sh for this arm first"

echo "arm=$ARM plan_stride=$PLAN_STRIDE plan_fusion=$PLAN_FUSION seeds=$(echo $SEEDS | wc -w) extra=$(wc -l < "$EXTRA")"

if [ "$FROM" -le 1 ]; then
  step "1 generate the $(wc -l < "$EXTRA") clip(s) the grouping needs and the headline missed"
  # The per-GPU directories go too, not just the flattened copy.  Leaving them
  # is how a rerun on fewer cards silently keeps the previous run's output: this
  # script's own first invocation on the 150-frame arm resharded 193 clips from
  # five cards onto three, and the two stale directories -- holding exactly the
  # non-overlapping-default samples this rerun exists to replace -- were still
  # sitting there with their old manifests. Caught by the check below, but a
  # check that fires on my own leftovers is a check doing someone else's job.
  rm -rf "$OUT/motion_extra" "$OUT/extra_shards"
  rm -rf "$OUT"/extra_seed_*
  mkdir -p "$OUT/motion_extra" "$OUT/extra_shards"
  ncards=$(echo $GPUS | wc -w)
  split -n "r/${ncards}" -d "$EXTRA" "$OUT/extra_shards/clips_"
  index=0; pids=()
  for gpu in $GPUS; do
    shard=$(printf "%s/extra_shards/clips_%02d" "$OUT" "$index")
    [ -s "$shard" ] || { index=$((index + 1)); continue; }
    (
      for seed in $SEEDS; do
        CUDA_VISIBLE_DEVICES="$gpu" python3 infer_atomic.py \
            --planner-checkpoint "$CKPT_P" --completion-checkpoint "$CKPT_C" \
            --data-root "$RELEASE" --audio-dir "$AUDIO" --sequence-list "$shard" \
            --output-dir "$OUT/extra_seed_${seed}_g${gpu}" --device cuda \
            --seed "$seed" --plan-stride "$PLAN_STRIDE" --plan-fusion "$PLAN_FUSION" \
            --overwrite \
            >> "logs/m6_${ARM}_extra_g${gpu}.log" 2>&1 || exit 1
      done
    ) &
    pids+=($!); index=$((index + 1))
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=$((failed + 1)); done
  [ "$failed" -eq 0 ] || die "$failed extra shard(s) failed; see logs/m6_${ARM}_extra_g*.log"

  for gpu in $GPUS; do
    for seed in $SEEDS; do
      dir="$OUT/extra_seed_${seed}_g${gpu}"
      [ -d "$dir" ] || continue
      for pkl in "$dir"/*.pkl; do
        [ -e "$pkl" ] || continue
        cp "$pkl" "$OUT/motion_extra/$(basename "${pkl%.pkl}")_s${seed}.pkl"
      done
    done
  done
fi

step "1b the sampler flags the manifests actually recorded"
# Checked, not trusted.  This is the exact defect the header describes: nothing
# compared the extra run's flags against the headline run's, so a default that
# measurably changes the generator went unnoticed for a day.
python3 - "$OUT" "$PLAN_STRIDE" "$PLAN_FUSION" <<'PY' || die "sampler flags disagree"
import glob, json, sys
out, want_stride, want_fusion = sys.argv[1], int(sys.argv[2]), sys.argv[3]
groups = {"headline": "{}/seed_*/manifest.json".format(out),
          "extra": "{}/extra_seed_*/manifest.json".format(out)}
bad = False
for label, pattern in groups.items():
    seen, files = set(), sorted(glob.glob(pattern))
    for path in files:
        sampling = json.load(open(path)).get("sampling", {})
        seen.add((sampling.get("plan_stride"), sampling.get("plan_fusion")))
    print("   {:8s} {:2d} manifest(s) -> {}".format(label, len(files), sorted(seen)))
    if not files:
        print("   {}: no manifest".format(label)); bad = True
    elif seen != {(want_stride, want_fusion)}:
        print("   {}: expected {}".format(label, (want_stride, want_fusion))); bad = True
raise SystemExit(1 if bad else 0)
PY

if [ "$FROM" -le 2 ]; then
  step "2 features over the union (headline clips + extra clips)"
  rm -rf "$OUT/features_trackgroup"; mkdir -p "$OUT/motion_union"
  rm -rf "$OUT/motion_union"; mkdir -p "$OUT/motion_union"
  # Symlinks, not copies: the union is 3,172 sequences and the two halves are
  # already on disk.  ln -s keeps this step from doubling the arm's footprint,
  # which CLAUDE.md 1.2 is explicit about.
  for pkl in "$OUT"/motion/*.pkl "$OUT"/motion_extra/*.pkl; do
    [ -e "$pkl" ] || continue
    ln -sf "$(realpath "$pkl")" "$OUT/motion_union/$(basename "$pkl")"
  done
  echo "   union: $(ls "$OUT"/motion_union/*.pkl 2>/dev/null | wc -l) sequence(s)"
  python3 eval/extract_aist_features.py --motion-dir "$OUT/motion_union" --audio-dir "$AUDIO" \
      --output "$OUT/features_trackgroup" --smpl-model "$SMPL" --workers 16 \
      > "logs/m6_${ARM}_trackgroup_features.log" 2>&1 || die "feature extraction"
  echo "   $(ls "$OUT"/features_trackgroup/kinetic_features 2>/dev/null | wc -l) feature set(s)"
fi

step "3 MultiModality over the verified-shared-track grouping"
# --all-pairs stays off: with the grouping supplied, a sample's identity is its
# clip with the seed stripped, so the four seeds of one clip are excluded from
# MM.  Counting them would turn the metric into "how unstable is re-sampling",
# which is not what the ground-truth calibration measured.
python3 tools/eval_multimodality.py \
    --feature-root "$OUT/features_trackgroup" --feature kinetic \
    --music-pairs "$PAIRS" --bundle "$BUNDLE" --split test \
    --permutations 200 --seed 20260812 \
    --output "runs/mm_model_${ARM}_trackgroup.json" \
    > "logs/m6_${ARM}_mm.log" 2>&1 || die "multimodality"
python3 - "runs/mm_model_${ARM}_trackgroup.json" <<'PY' | sed 's/^/   /'
import json, sys
d = json.load(open(sys.argv[1]))
print("MM/Div {:.4f}  (pooled {:.4f})  p {:.5f}  samples {}  groups {}".format(
    d["mm_over_div"], d["mm_pooled_over_div"], d["p_value"], d["samples"], d["musics_used"]))
print("MM {:.4f}  Div {:.4f}  cross_recording_pairs_only {}".format(
    d["multimodality"], d["diversity"], d.get("cross_recording_pairs_only")))
PY

step "done -> runs/mm_model_${ARM}_trackgroup.json"
echo "Read it against runs/mm_gt_wild_v4_acct_trackgroup_fullseq.json (0.5982), which is"
echo "the full-sequence ground truth on the same 144 groups.  NOT against"
echo "runs/mm_gt_wild_v4_acct_test.json (0.7194) -- that one is computed over the"
echo "release's 39,624 windows and is a different unit from one sequence per clip."
