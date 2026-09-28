#!/usr/bin/env bash
#
# M6: the paper's metric suite over one planner+completion pair.
#
#   bash tools/run_m6_headline.sh                    # the 630-class song-disjoint pair
#   CKPT_C=<path> TAG=<name> bash tools/run_m6_headline.sh
#
# Order matters and each step feeds the next, so a failure stops the chain
# rather than letting a later step read a stale or partial input:
#
#   1 infer      frozen planner -> predicted plan -> completion   (SELF_DRIVEN)
#   2 features   generated motion -> kinetic/manual/dance
#   3 evaluate   FID_k, FID_m, Div_k, Div_m, BAS   (vs the 411-sequence GT set)
#   4 model R    generated motion -> scoring bundle -> the release's own
#                R-precision code, with same-song pairs excluded
#   5 ceilings   pointers to the release-level R-precision and MM/Div, which do
#                NOT depend on the checkpoint and are computed once per release
#
# Three things this pins, each because leaving it implicit has already cost
# something today:
#
# * **SMPL is the converted, chumpy-free model.**  The original v1.1.0 pkl holds
#   chumpy objects and chumpy 0.70 predates both Python 3.11 and numpy 1.24; the
#   feature extractor runs multiprocess, so a runtime shim would have to be
#   applied in every worker.
# * **Ground-truth features are computed once and reused.**  They cost 411 SMPL
#   forward passes and do not depend on the checkpoint under test.
# * **The numbers are only comparable at a fixed clip length and pool size.**
#   Ground-truth R is 0.2 here against the paper's 42.1 purely because the
#   candidate pool is the whole split rather than the paper's unstated one.  A
#   headline R without the calibration alongside it is a number with no ruler.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TAG="${TAG:-songsplit630}"
RELEASE="${RELEASE:-data/atomic_aistpp/aist_songsplit_rg2_v3_release_v1}"
CKPT_P="${CKPT_P:-runs/planner_songsplit_rg2v3_630_x0_e600/planner_step135600.pt}"
CKPT_C="${CKPT_C:-runs/completion_songsplit630_x0_e600/completion_step135600.pt}"
SMPL="${SMPL:-third_party/smpl_models/SMPL_MALE_clean.pkl}"
GT_FEATURES="${GT_FEATURES:-runs/m6_gt_features}"
AUDIO="${AUDIO:-data/aist_music}"
GPU="${GPU:-0}"
# Held-out songs only.  The audio directory holds all 62 AIST tracks and
# infer_atomic rglobs it, so the default would generate for the 40 training
# songs too and then score them against a ground-truth set that contains them.
# The split is song-disjoint by construction; throwing that away at the last
# step would be the only leak in the chain.
SONGS="${SONGS:-runs/m6_splits/test_songs.txt}"
# Four seeds over ten songs.  Ten sequences cannot support the covariance FID
# estimates -- and the planner is stochastic by default, so re-sampling is the
# generator's own variation rather than a trick to inflate n.
SEEDS="${SEEDS:-20260808 20260809 20260810 20260811}"
OUT="runs/m6_${TAG}"
mkdir -p logs "$OUT"

step() { echo; echo "== [$(date '+%F %T')] $* =="; }
die()  { echo "M6 FAILED: $*" >&2; exit 1; }

for path in "$CKPT_P" "$CKPT_C" "$SMPL" "$RELEASE"; do
  [ -e "$path" ] || die "missing input: $path"
done
[ -d "$GT_FEATURES/kinetic_features" ] || die "ground-truth features absent: $GT_FEATURES"

# --unsourced-retrieval, added 2026-08-16.  A query here is named for a held-out
# *song*, not a recording, so it resolves to no retrieval group and the
# fail-closed branch fired on every sample -- and the completion model takes
# (music, draft, mask) and never a label, so the plan reached the motion not at
# all.  Measured on the artifacts this script produced: 33 of 40 samples had a
# zero draft and the other 7 had plans that were 100% transition.  Retrieving
# without exclusion is sound precisely because the split is song-disjoint: no
# training window carries the held-out song, so there is nothing to exclude.
step "1 closed-loop inference (frozen planner -> predicted plan -> completion)"
[ -s "$SONGS" ] || die "song list absent: $SONGS"
rm -rf "$OUT/motion"; mkdir -p "$OUT/motion"
for seed in $SEEDS; do
  CUDA_VISIBLE_DEVICES="$GPU" python3 infer_atomic.py \
      --planner-checkpoint "$CKPT_P" --completion-checkpoint "$CKPT_C" \
      --data-root "$RELEASE" --audio-dir "$AUDIO" --sequence-list "$SONGS" \
      --output-dir "$OUT/seed_${seed}" --device cuda --seed "$seed" --overwrite \
      --unsourced-retrieval \
      >> "logs/m6_${TAG}_infer.log" 2>&1 || die "inference seed $seed"
  # Flattened with a seed suffix because extract_aist_features globs one
  # directory, and the suffix survives audio matching: mBR2_s20260808 still
  # yields the music id mBR2.
  for pkl in "$OUT/seed_${seed}"/*.pkl; do
    [ -e "$pkl" ] || continue
    cp "$pkl" "$OUT/motion/$(basename "${pkl%.pkl}")_s${seed}.pkl"
  done
  echo "   seed $seed -> $(ls "$OUT/seed_${seed}"/*.pkl 2>/dev/null | wc -l) sequence(s)"
done
produced=$(ls "$OUT"/motion/*.pkl 2>/dev/null | wc -l)
echo "   generated $produced sequence(s) over $(wc -l < "$SONGS") held-out song(s)"
# Count the artifacts, not the exit code, and refuse a sample count that cannot
# support a covariance estimate: FID over a handful of clips is not a number.
[ "$produced" -ge 10 ] || die "only $produced generated sequences; FID needs a real sample"

step "2 features from the generated motion"
python3 eval/extract_aist_features.py --motion-dir "$OUT/motion" --audio-dir "$AUDIO" \
    --output "$OUT/features" --smpl-model "$SMPL" --workers 8 \
    > "logs/m6_${TAG}_features.log" 2>&1 || die "feature extraction"
echo "   $(ls "$OUT"/features/kinetic_features 2>/dev/null | wc -l) feature set(s)"

step "3 FID / Div / BAS against the ground-truth set"
python3 eval/evaluate.py --prediction-features "$OUT/features" \
    --ground-truth-features "$GT_FEATURES" \
    > "logs/m6_${TAG}_evaluate.log" 2>&1 || die "evaluate"
grep -E '"(fid|div|BAS|num)_?' "logs/m6_${TAG}_evaluate.log" | sed 's/^/   /'

# Steps 4 and 5 are deliberately NOT here.
#
# ``eval_r_precision.py`` and ``eval_multimodality.py`` take ``--release`` and
# score the motion *inside that release* -- they never open a checkpoint and
# never see the generated pkls.  Run per checkpoint they cost 24 minutes to
# recompute a constant, and printing them under a model's heading would publish
# the vocabulary's ceiling as the model's score.  Measured 2026-08-14: the
# dry-run's "model" R was byte-identical to the ground-truth run from three
# hours earlier.
#
# They belong to the release, so they are computed once per release:
#
#   python3 tools/eval_r_precision.py   --release "$RELEASE" --split test --output ...
#   python3 tools/eval_multimodality.py --release "$RELEASE" --split test --group-by music --output ...
#
# Ceilings already on disk for the 630-class song-disjoint release:
#   R-precision val  1.8 excluding same sequence (chance 0.09), rank 0.2734
#   MM/Div      val  0.825 against a permutation null of 1.0046, p = 0.00498
#
# A model-level R-precision needs the generated motion materialised into those
# same arrays so the same code can score it.  That is what step 4 now does, via
# ``tools/materialize_generated_release.py``; the ceilings stay in step 5 as
# pointers, because they still belong to the release rather than to any
# checkpoint.

step "4 model-level R-precision (the generated motion, through the release scorer)"
# This is the step the comment above says was not done.  It is done by putting
# the generated motion into the arrays ``eval_r_precision`` reads, so the same
# code scores it -- the bundle carries no labels and is not a release.
#
# ``--exclude-same-music`` is not optional here: four seeds of one song share
# their 35-D music features exactly, so without it every clip's nearest music
# clip is one of its own siblings and R would report re-sampling stability as
# music-motion structure.
if [ -n "${SKIP_MODEL_R:-}" ]; then
  echo "   skipped (SKIP_MODEL_R set) -- the headline then has no model-level R"
else
  rm -rf "$OUT/scoring_bundle"
  python3 tools/materialize_generated_release.py --motion-dir "$OUT/motion" \
      --reference-release "$RELEASE" --output "$OUT/scoring_bundle" --split test \
      > "logs/m6_${TAG}_bundle.log" 2>&1 || die "materialising the generated motion"
  tail -3 "logs/m6_${TAG}_bundle.log" | sed 's/^/   /'
  python3 tools/eval_r_precision.py --release "$OUT/scoring_bundle" --split test \
      --exclude-same-music --output "$OUT/r_precision_model.json" \
      > "logs/m6_${TAG}_rprec_model.log" 2>&1 || die "model R-precision"
  python3 - "$OUT/r_precision_model.json" <<'PY' | sed 's/^/   /'
import json, sys
report = json.load(open(sys.argv[1]))
for key in ("pooled", "excluding_same_sequence", "excluding_same_music"):
    row = report[key]
    print("{:24s} R {:6.2f}  chance {:5.2f}  pool {:7.1f}  rank {:.4f}".format(
        key, row["R"], row["chance_R"], row["mean_candidate_pool"],
        row["mean_normalised_rank"]))
PY
fi

step "5 ceilings (from the release, not this checkpoint)"
# Named after the release, not after a hard-coded run.  A ceiling belongs to the
# vocabulary it was measured on, and printing another release's ceiling under
# this heading is the same mistake as printing the ceiling as the model's score.
# One command per file.  The first version printed *both* commands under *both*
# filenames, so following it wrote the MultiModality report to
# rprec_gt_<tag>_val.json and vice versa -- a file whose name says which ceiling
# it holds and whose contents say the other.  Nothing downstream reads the name
# and the contents together, so it would have been read as the wrong ceiling.
ceiling_r="runs/rprec_gt_${TAG}_val.json"
ceiling_mm="runs/mm_gt_${TAG}_val.json"
[ -s "$ceiling_r" ] && echo "   $ceiling_r" || { echo "   MISSING $ceiling_r -- compute it once for $RELEASE:";
  echo "     python3 tools/eval_r_precision.py --release $RELEASE --split val --output $ceiling_r"; }
[ -s "$ceiling_mm" ] && echo "   $ceiling_mm" || { echo "   MISSING $ceiling_mm -- compute it once for $RELEASE:";
  echo "     python3 tools/eval_multimodality.py --release $RELEASE --split val --group-by music --output $ceiling_mm"; }

step "M6 complete -> $OUT"
echo "Generated over held-out songs only ($(wc -l < "$SONGS")), $(echo $SEEDS | wc -w) seeds."
echo "Report alongside every number: full-song conditioning is NOT implemented"
echo "(seq_len 150 chunks), genre pre-split is off, and the captioner is a local"
echo "Qwen rather than the paper's Gemini-2.5-Pro."
