#!/usr/bin/env bash
# One arm of the 340-frame paired draft protocol: infer, featurise, evaluate,
# and evaluate again on the leak-free subset.
#
# The protocol is fixed so arms are comparable: same 300 clips, same seed, same
# completion checkpoint, same planner, and a FID reference built from the *same*
# 340 frames.  Only the draft-side flags move.  Every arm here is testing the
# same hypothesis -- that the plan's value is in the draft being real motion,
# not in the class being right (measured: no plan 5.771, planner's plan 3.551,
# perfect plan 3.277 on leak-free fid_k) -- so the flags that shape the draft as
# motion are the ones worth a 300-clip number.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
NAME="$1"; GPU="$2"; shift 2
SEED="${SEED:-20260816}"
# The completion checkpoint is a variable so a *training-side* arm can reuse this
# protocol unchanged.  It defaults to the arm every t14-t19 number was produced
# with, so every existing invocation still reproduces byte-for-byte.
CKPT_C="${CKPT_C:-runs/completion_wild_v4_acct_w2_340/completion_step58968.pt}"
OUT="runs/t18_draft/$NAME"
CUDA_VISIBLE_DEVICES="$GPU" python3 infer_atomic.py \
  --audio-dir runs/t14_oracle/audio340 --output-dir "$OUT/motion" \
  --planner-checkpoint runs/planner_wild_v4_acct_w2_340/planner_step58968.pt \
  --completion-checkpoint "$CKPT_C" \
  --data-root /dev/shm/atomicdance-acct/release_w2_340 \
  --sequence-list runs/t14_oracle/clips340.txt --seed "$SEED" \
  --plan-stride 15 --plan-fusion vote --completion-stride 75 \
  --inference-batch-size 4 --device cuda "$@" \
  > "logs/t18_${NAME}_infer.log" 2>&1 || { echo "$NAME infer FAILED"; exit 1; }
python3 eval/extract_aist_features.py --motion-dir "$OUT/motion" \
  --audio-dir runs/t14_oracle/audio340 --output "$OUT/features" \
  --smpl-model third_party/smpl_models/SMPL_MALE_clean.pkl --workers 8 \
  > "logs/t18_${NAME}_features.log" 2>&1 || { echo "$NAME features FAILED"; exit 1; }
python3 eval/evaluate.py --prediction-features "$OUT/features" \
  --ground-truth-features runs/t14_oracle/gt_features340 \
  > "logs/t18_${NAME}_evaluate.log" 2>&1 || { echo "$NAME evaluate FAILED"; exit 1; }
python3 - "$OUT" <<'PY' || exit 1
import json, pathlib, shutil, sys
out = pathlib.Path(sys.argv[1])
flagged = set(json.loads(pathlib.Path("runs/t9_ab/clips300.json").read_text())["leak"]["flagged_clips_in_split"])
def clip_of(stem):
    m = stem.rfind("_s")
    return stem[:m] if m > 0 and stem[m+2:].isdigit() else stem
src = out / "features"
for family in sorted(p.name for p in src.iterdir() if p.is_dir()):
    t = out / "clean" / family
    if t.exists(): shutil.rmtree(t)
    t.mkdir(parents=True)
    for path in sorted((src / family).glob("*.npy")):
        if clip_of(path.stem) not in flagged:
            shutil.copy2(path, t / path.name)
PY
python3 eval/evaluate.py --prediction-features "$OUT/clean" \
  --ground-truth-features runs/t14_oracle/clean_gt \
  > "logs/t18_${NAME}_evaluate_clean.log" 2>&1 || { echo "$NAME clean evaluate FAILED"; exit 1; }
echo "$NAME DONE"
