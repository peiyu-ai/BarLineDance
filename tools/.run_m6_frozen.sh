#!/usr/bin/env bash
#
# M6 on the wild corpus.  Same metric suite as tools/run_m6_headline.sh, but
# three of that script's inputs do not exist here and the differences are the
# reason this is a separate file rather than a flag:
#
#   * **Audio.**  AIST has 62 WAVs in data/aist_music and infer_atomic rglobs
#     them.  A wild clip's music is the 35-D array the planner was trained on,
#     stored per clip; re-deriving beats from the ingest tree's audio would put
#     BAS on a second extractor run (beat channel correlates 0.36, measured in
#     tools/convert_aistpp_official.py).  So the audio directory here holds the
#     .npy that tools/export_wild_eval_motion.py copied out of the bundle.
#   * **The held-out unit.**  AIST holds out songs and generates for 10 of them.
#     This corpus has no track id, so the unit is the clip, chosen by
#     tools/select_wild_eval_clips.py -- which also records which clips provably
#     share a backing track across the split, so step 3 can score the leak-free
#     subset without generating anything twice.
#   * **The ground truth.**  FID is a distribution against a distribution, and a
#     wild distribution may only be read against a wild ground truth
#     (WILD_ATOMIC_PIPELINE_PLAN.md 5.1).  runs/wild_v4_acct_gt_features is that
#     set: 1,575 test sequences through the same forward kinematics the
#     generated motion takes.
#
#   bash tools/run_m6_wild.sh
#   GPUS="1 2 3 4 5 6" COUNT=400 bash tools/run_m6_wild.sh
#
# Steps:
#   0 select     which clips, and which of them leak
#   1 infer      frozen planner -> predicted plan -> completion, sharded over cards
#   2 features   generated motion -> kinetic/manual/dance/music
#   3 evaluate   FID_k, FID_m, Div_k, Div_m, BAS -- twice: all, and leak-free
#   4 model R    generated motion -> scoring bundle -> the release's own scorer,
#                with same-upload pairs AND fingerprint-verified pairs excluded
#   5 ceilings   release-level R and MM, which belong to the release not the
#                checkpoint, so they are pointers rather than recomputation
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TAG="${TAG:-wild_v4_acct}"
RELEASE="${RELEASE:-/dev/shm/atomicdance-acct/release_v1}"
BUNDLE="${BUNDLE:-/dev/shm/atomicdance-acct/performance}"
CKPT_P="${CKPT_P:-runs/planner_${TAG}_x0/planner_step134496.pt}"
CKPT_C="${CKPT_C:-runs/completion_${TAG}/completion_step134496.pt}"
GT_FEATURES="${GT_FEATURES:-runs/${TAG}_gt_features}"
AUDIO="${AUDIO:-runs/${TAG}_gt_eval/audio}"
PAIRS="${PAIRS:-runs/wild_v4_music_groups_pairs.jsonl}"
SMPL="${SMPL:-third_party/smpl_models/SMPL_MALE_clean.pkl}"
CLIPS="${CLIPS:-runs/${TAG}_m6_clips}"
COUNT="${COUNT:-400}"
# Four seeds per clip.  The planner is stochastic by default, so re-sampling is
# the generator's own variation rather than a trick to inflate n; MultiModality
# needs several draws of one clip and cannot be defined on one.
SEEDS="${SEEDS:-20260816 20260817 20260818 20260819}"
# Measured, not chosen.  The planner chunks a track into 150-frame windows, and
# with the default non-overlapping stride 18.92x of the generated segment
# boundaries land exactly on the seam (ground truth: 1.02x, i.e. chance).
# Overlapping the windows and fusing by majority vote drops that to 2.24x *and*
# improves fid_k from 10.613 to 9.631 on a paired 200-clip A/B -- the only
# configuration that moves both.  ``centre`` clears the artifact further (0.63x)
# and doubles fid_k, because it nearly doubles the boundary count and every
# extra boundary is another retrieved-prototype splice.
#
# Set here rather than as the library default so every artifact produced before
# 2026-08-16 still reproduces from its own command line.
PLAN_STRIDE="${PLAN_STRIDE:-15}"
PLAN_FUSION="${PLAN_FUSION:-vote}"
# How a tied vote is resolved, and whether the minimum-duration merge may
# delete a transition or absorb an atomic movement into one.  Both were
# unchosen before 2026-08-23: the fusion ended in ``counts.argmax``, which
# hands every tie to index 0 -- transition -- and the merge ranked neighbours
# by length with no regard for label.  On the 33 clean5b5 M6 clips the tie-break
# alone was 4.6 of the 22.0 points by which the plan's transition share exceeded
# the ground truth (tools/probe_plan_vote_ties.py).  Set to the pre-fix values
# to reproduce any artifact made before that date -- all three of
# PLAN_VOTE_TIE_BREAK=index PLAN_TRANSITION_POLICY=merge PLAN_MERGE_ORDER=first,
# because the order is a separate axis and two of the three is not a reproduction.
PLAN_VOTE_TIE_BREAK="${PLAN_VOTE_TIE_BREAK:-centre}"
PLAN_TRANSITION_POLICY="${PLAN_TRANSITION_POLICY:-protect}"
# The order the merge resolves offenders in, kept separate from the policy
# because binding them together costs the reproduction: an artifact from before
# 2026-08-23 needs index + merge + first, and merge alone is not it.
PLAN_MERGE_ORDER="${PLAN_MERGE_ORDER:-shortest}"
# Two calibrations of how much vocabulary the plan admits, both off by default.
# PLAN_BIAS is added to the transition class's x0 logit; it must be fitted on
# val, never on test -- runs/clean5b5_bias_fit_val.json holds the fit and the
# 80 val clips share no recording with this driver's test clips.  PLAN_BAR_GRID
# replaces the plan's segmentation with M1's own music bar grid, on which 97.3%
# of ground-truth boundaries sit against 30.7% of the planner's own.
# PLAN_BIAS stays off: fitted on val at -1.0 it overshot on test (0.2558 against
# a ground truth of 0.3214) while the grid alone landed at 0.3482, so the
# val->test transfer was negative on this corpus.  Set here rather than as the
# library default so every artifact produced before 2026-08-23 still reproduces
# from its own command line, which is the same rule PLAN_STRIDE follows.
PLAN_BIAS="${PLAN_BIAS:-0.0}"
# On since 2026-08-23.  Measured over 260 generations, holding the planner, the
# seeds and the clips fixed: transition 0.5194 -> 0.3482 (ground truth 0.3214),
# and the clips whose plan named no atomic movement at all fell from 12 to 11
# -- and to 0 once the bias is added.  fid_k cannot separate it from the
# ungridded arm (paired clip bootstrap, both subsets, zero inside the interval),
# so this is justified on the plan, not on the generation.
PLAN_BAR_GRID="${PLAN_BAR_GRID:-1}"
# Classifier-free guidance on the planner.  Needs a checkpoint trained with
# --planner-cond-drop-prob > 0 and is refused otherwise, so 1.0 stays the
# default and every earlier planner keeps running.  1.5 is what
# tools/fit_planner_guidance.py chose on val for runs/planner_c5_cfg -- fitted
# there and not swept here, because a knob with this much leverage read off a
# test table is a knob fitted on test.
# Default 1.0 rather than the fitted 1.5, because guidance is refused outright
# by a planner trained without --planner-cond-drop-prob and most checkpoints in
# this repository are.  The value fitted on val for runs/planner_c5_cfg is 1.5
# and the arm that uses it sets it; a driver default of 1.5 would turn every
# older planner's run into an immediate crash.
PLANNER_GUIDANCE="${PLANNER_GUIDANCE:-1.0}"
BAR_GRID_FLAG=""
[ "$PLAN_BAR_GRID" = "1" ] && BAR_GRID_FLAG="--plan-bar-grid"
# Sampling is a parameter, not a constant, since 2026-08-22.  The two settings
# are the same checkpoint and the same seed and differ by 60 points of the
# generated plan: measured on clean5b5's planner over six test clips,
# `deterministic` (argmax reverse steps) emits transition on 1% of frames while
# `stochastic` emits it on 58.5% -- against 34% in the ground truth.  Both are
# wrong, in opposite directions, so a driver that hard-codes one is reporting a
# sampling artifact as a model property.  Held at `stochastic` because that is
# what every figure before this date used.
PLAN_SAMPLING="${PLAN_SAMPLING:-stochastic}"
case "$PLAN_SAMPLING" in
  stochastic)    SAMPLING_FLAG="--stochastic-planner" ;;
  deterministic) SAMPLING_FLAG="--deterministic-planner" ;;
  *) echo "PLAN_SAMPLING must be stochastic or deterministic, got '$PLAN_SAMPLING'" >&2; exit 2 ;;
esac
# Which clips the selection may draw from.  Empty means the whole split; set it
# when part of the split is disqualified for a reason the split cannot see, such
# as a borrowed completion checkpoint having trained on it.
RESTRICT_TO="${RESTRICT_TO:-}"
GPUS="${GPUS:-1 2 3 4 5 6}"
OUT="${OUT:-runs/m6_${TAG}}"
# Every log this driver writes is keyed by the *run*, not by TAG.  Two arms of a
# sampling sweep share a TAG by construction -- they are the same corpus and the
# same checkpoint -- so TAG-keyed logs meant three concurrent runs wrote one
# `m6_clean5b5_evaluate.log` and each then grepped whatever had landed there
# last.  Measured 2026-08-22: three arms whose generated plans differ (4 / 12 /
# 30 segments, 93% / 83% / 0% transition) printed one identical fid_k to sixteen
# digits.  The FIDs had been computed and then overwritten.
RUN="$(basename "$OUT")"
FROM="${FROM:-0}"
# Upper bound as well as lower.  Step 4 re-features the generated release and on
# this corpus that is hours, so "produce the headline FID and stop" has to be
# expressible -- otherwise the only way to stop before it is to kill the driver,
# which leaves step 3b's clean_* trees half-copied and the next run reading them.
TO="${TO:-5}"
run_step() { [ "$FROM" -le "$1" ] && [ "$TO" -ge "$1" ]; }
mkdir -p logs "$OUT"

step() { echo; echo "== [$(date '+%F %T')] $* =="; }
die()  { echo "M6 FAILED: $*" >&2; exit 1; }

for path in "$CKPT_P" "$CKPT_C" "$SMPL" "$RELEASE" "$BUNDLE"; do
  [ -e "$path" ] || die "missing input: $path"
done
[ -d "$GT_FEATURES/kinetic_features" ] || die "wild ground-truth features absent: \
$GT_FEATURES -- build them with tools/export_wild_eval_motion.py then \
eval/extract_aist_features.py"
[ -d "$AUDIO" ] || die "per-clip music absent: $AUDIO"

if run_step 0; then
  step "0 select the clips, and flag the ones that provably leak"
  python3 tools/select_wild_eval_clips.py --bundle "$BUNDLE" --split test \
      --music-pairs "$PAIRS" --count "$COUNT" --output "$CLIPS" \
      ${RESTRICT_TO:+--restrict-to "$RESTRICT_TO"} \
      || die "clip selection"
fi
[ -s "${CLIPS}.txt" ] || die "clip list absent: ${CLIPS}.txt"

if run_step 1; then
  step "1 closed-loop inference over $(wc -l < "${CLIPS}.txt") clip(s) x $(echo $SEEDS | wc -w) seed(s) [$PLAN_SAMPLING, stride $PLAN_STRIDE, $PLAN_FUSION, tie=$PLAN_VOTE_TIE_BREAK, transition=$PLAN_TRANSITION_POLICY, order=$PLAN_MERGE_ORDER, bias=$PLAN_BIAS, bargrid=$PLAN_BAR_GRID, guidance=$PLANNER_GUIDANCE]"
  rm -rf "$OUT/motion" "$OUT/shards"; mkdir -p "$OUT/motion" "$OUT/shards"
  # Sharded by clip, not by seed: a shard that dies then costs some clips at
  # every seed rather than one whole seed at every clip, and MultiModality needs
  # all seeds of a clip or none of them.
  ncards=$(echo $GPUS | wc -w)
  split -n "r/${ncards}" -d "${CLIPS}.txt" "$OUT/shards/clips_"
  index=0
  pids=()
  for gpu in $GPUS; do
    shard=$(printf "%s/shards/clips_%02d" "$OUT" "$index")
    [ -s "$shard" ] || { index=$((index + 1)); continue; }
    (
      for seed in $SEEDS; do
        CUDA_VISIBLE_DEVICES="$gpu" python3 infer_atomic.py \
            --planner-checkpoint "$CKPT_P" --completion-checkpoint "$CKPT_C" \
            --data-root "$RELEASE" --audio-dir "$AUDIO" --sequence-list "$shard" \
            --output-dir "$OUT/seed_${seed}_g${gpu}" --device cuda \
            --seed "$seed" --plan-stride "$PLAN_STRIDE" --plan-fusion "$PLAN_FUSION" \
            --plan-vote-tie-break "$PLAN_VOTE_TIE_BREAK" \
            --plan-transition-policy "$PLAN_TRANSITION_POLICY" \
            --plan-merge-order "$PLAN_MERGE_ORDER" \
            --planner-guidance-weight "$PLANNER_GUIDANCE" \
            --planner-transition-logit-bias "$PLAN_BIAS" $BAR_GRID_FLAG \
            $SAMPLING_FLAG --overwrite \
            >> "logs/${RUN}_infer_g${gpu}.log" 2>&1 || exit 1
      done
    ) &
    pids+=($!)
    index=$((index + 1))
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=$((failed + 1)); done
  [ "$failed" -eq 0 ] || die "$failed inference shard(s) failed; see logs/${RUN}_infer_g*.log"

  # Flattened with a seed suffix because the feature extractor globs one
  # directory.  match_audio strips that suffix to find the clip's own music --
  # on AIST the music id made it unnecessary, here it is load-bearing.
  for gpu in $GPUS; do
    for seed in $SEEDS; do
      dir="$OUT/seed_${seed}_g${gpu}"
      [ -d "$dir" ] || continue
      for pkl in "$dir"/*.pkl; do
        [ -e "$pkl" ] || continue
        cp "$pkl" "$OUT/motion/$(basename "${pkl%.pkl}")_s${seed}.pkl"
      done
    done
  done
fi

produced=$(ls "$OUT"/motion/*.pkl 2>/dev/null | wc -l)
expected=$(( $(wc -l < "${CLIPS}.txt") * $(echo $SEEDS | wc -w) ))
echo "   generated $produced of $expected expected sequence(s)"
# Count the artifacts, not the exit code.  FID over a handful of clips is not a
# number, and a partially-completed shard set is exactly how a plausible-looking
# but under-sampled FID gets published.
[ "$produced" -ge 10 ] || die "only $produced generated sequences"
[ "$produced" -eq "$expected" ] || echo "   WARNING: $((expected - produced)) missing -- \
FID below is computed on what exists, and its n differs from any run that completed"

if run_step 2; then
  step "2 features from the generated motion"
  python3 eval/extract_aist_features.py --motion-dir "$OUT/motion" --audio-dir "$AUDIO" \
      --output "$OUT/features" --smpl-model "$SMPL" --workers 16 \
      > "logs/${RUN}_features.log" 2>&1 || die "feature extraction"
  echo "   $(ls "$OUT"/features/kinetic_features 2>/dev/null | wc -l) feature set(s)"
fi

if run_step 3; then
  step "3 FID / Div / BAS against the wild ground-truth set"
  python3 eval/evaluate.py --prediction-features "$OUT/features" \
      --ground-truth-features "$GT_FEATURES" \
      > "logs/${RUN}_evaluate.log" 2>&1 || die "evaluate"
  grep -E '"(fid|div|BAS|num)_?' "logs/${RUN}_evaluate.log" | sed 's/^/   /'

  # The leak-free twin.  52.4% of this corpus's test split provably shares a
  # backing track with a clip on the other side of the account split, so the
  # figure above is computed on a set that is half contaminated.  Both sides are
  # filtered -- keeping the full ground truth against a clean prediction set
  # would compare two different populations and call the difference FID -- and
  # both are filtered by the *split's* flagged set rather than the selection's.
  # Filtering the ground truth by only the selected clips' flags would leave 618
  # of this split's 825 flagged clips sitting in the "clean" reference.
  step "3b the same numbers on the leak-free subset"
  python3 - "$CLIPS.json" "$OUT" "$GT_FEATURES" "$OUT/features" <<'PY' || die "leak-free filter"
import json, pathlib, shutil, sys
report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
out, gt_src, pred_src = (pathlib.Path(p) for p in sys.argv[2:5])
flagged = set(report["leak"]["flagged_clips_in_split"])
def clip_of(stem):
    marker = stem.rfind("_s")
    return stem[:marker] if marker > 0 and stem[marker + 2:].isdigit() else stem
kept = {"gt": 0, "pred": 0}
for tag, src in (("gt", gt_src), ("pred", pred_src)):
    for family in sorted(p.name for p in src.iterdir() if p.is_dir()):
        target = out / ("clean_" + tag) / family
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        for path in sorted((src / family).glob("*.npy")):
            if clip_of(path.stem) not in flagged:
                shutil.copy2(path, target / path.name)
    first = sorted((out / ("clean_" + tag)).iterdir())[0]
    kept[tag] = len(list(first.glob("*.npy")))
print("   leak-free: {} ground-truth, {} generated (flagged {} of the selection)".format(
    kept["gt"], kept["pred"], report["leak"]["flagged_in_selection"]))
if kept["gt"] < 10 or kept["pred"] < 10:
    raise SystemExit("leak-free subset too small to support a covariance estimate")
PY
  python3 eval/evaluate.py --prediction-features "$OUT/clean_pred" \
      --ground-truth-features "$OUT/clean_gt" \
      > "logs/${RUN}_evaluate_clean.log" 2>&1 || die "evaluate (leak-free)"
  grep -E '"(fid|div|BAS|num)_?' "logs/${RUN}_evaluate_clean.log" | sed 's/^/   /'
fi

if run_step 4; then
  step "4 model-level R-precision (generated motion, through the release scorer)"
  # --exclude-same-music is not optional: several seeds of one clip share their
  # 35-D features exactly, so without it every clip's nearest music neighbour is
  # a sibling seed and R reports re-sampling stability as music-motion
  # structure.  --music-pairs is the second half of the same control: the upload
  # key removes cuts of one video, the pair list removes cross-upload pairs
  # proven to share a track, which the key cannot see.
  rm -rf "$OUT/scoring_bundle"
  python3 tools/materialize_generated_release.py --motion-dir "$OUT/motion" \
      --reference-release "$RELEASE" --output "$OUT/scoring_bundle" --split test \
      > "logs/${RUN}_bundle.log" 2>&1 || die "materialising the generated motion"
  python3 tools/eval_r_precision.py --release "$OUT/scoring_bundle" --split test \
      --exclude-same-music --music-pairs "$PAIRS" \
      --output "$OUT/r_precision_model.json" \
      > "logs/${RUN}_rprec_model.log" 2>&1 || die "model R-precision"
  python3 - "$OUT/r_precision_model.json" <<'PY' | sed 's/^/   /'
import json, sys
report = json.load(open(sys.argv[1]))
for key in ("pooled", "excluding_same_sequence", "excluding_same_music",
            "excluding_same_music_and_verified_pairs"):
    row = report.get(key)
    if row is None:
        continue
    print("{:42s} R {:6.2f}  chance {:5.2f}  pool {:7.1f}  rank {:.4f}".format(
        key, row["R"], row["chance_R"], row["mean_candidate_pool"],
        row["mean_normalised_rank"]))
PY
fi

if run_step 5; then
step "5 ceilings (from the release, not this checkpoint)"
ceiling_r="runs/rprec_gt_${TAG}_test.json"
ceiling_mm="runs/mm_gt_${TAG}_test.json"
[ -s "$ceiling_r" ] && echo "   $ceiling_r" || { echo "   MISSING $ceiling_r -- compute once for $RELEASE:";
  echo "     python3 tools/eval_r_precision.py --release $RELEASE --split test \\";
  echo "       --exclude-same-music --music-pairs $PAIRS --output $ceiling_r"; }
[ -s "$ceiling_mm" ] && echo "   $ceiling_mm" || { echo "   MISSING $ceiling_mm -- compute once for $RELEASE:";
  echo "     python3 tools/eval_multimodality.py --release $RELEASE --split test \\";
  echo "       --bundle $BUNDLE --music-pairs $PAIRS --output $ceiling_mm";
  echo "     (the grouping is complete subgraphs of the verified-pair graph spanning";
  echo "      two or more uploads: different performances of one proven-shared track.";
  echo "      Grouping by upload instead would measure how smooth one take is.)"; }

fi

step "M6 complete (steps $FROM..$TO) -> $OUT"
echo "Report alongside every number:"
echo " * 52.4% of this test split provably shares a backing track across the split;"
echo "   the leak-free twin in step 3b is the figure that is not contaminated, and"
echo "   it is still only a floor (the fingerprint's positive control recalls 0.564)."
echo " * Full-song conditioning is NOT implemented -- the planner sees 150 frames."
echo "   (A whole-track summary was tried and rejected on 2026-08-16: a *shuffled*"
echo "    summary bought the same structure-gate gain, so the gain was a song id.)"
echo " * Planner windows overlap (stride ${PLAN_STRIDE}, fusion ${PLAN_FUSION}); the"
echo "   non-overlapping default puts 18.9x of segment boundaries on the window seam."
echo " * The genre pre-split key is the uploader account, not a dance genre."
echo " * The captioner is a local Qwen, not the paper's Gemini-2.5-Pro."
echo " * Wild FID/Div are never comparable to AIST figures at any sample size."
