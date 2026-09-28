#!/usr/bin/env bash
# A/B the learned retrieval selector against the shipped duration rule.
#
# ONE VARIABLE.  Both arms use the same planner and completion checkpoints, the
# same seed, the same clip list in the same ORDER (order matters -- reversing an
# 8-clip list used to move 6 of the 8 drafts, docs/DANCE_QUALITY_DEFECTS.md
# section 13.4), and --inference-batch-size 1 so every clip gets its own noise.
# The only difference is --retrieval-rule.
#
# The rest of the configuration is copied from runs/w6vis_seam04i/manifest.json,
# which is the arm the operator watched as output/samples_20260901e -- so "the
# baseline" here is literally the footage the complaint was made about.
set -euo pipefail

RELEASE="${RELEASE:?set RELEASE to the materialized release directory}"
PLANNER="${PLANNER:-runs/planner_v5rekey_mn_cfg_s20260901/planner_step135935.pt}"
COMPLETION="${COMPLETION:-runs/completion_v5rekey_dd25_s01/completion_step118327.pt}"
SELECTOR="${SELECTOR:-runs/retrieval_selector_v1/selector.pt}"
CLIPS="${CLIPS:-runs/eval_clips_97.txt}"
AUDIO="${AUDIO:-runs/wild_v5_song_gt_eval/audio}"
GT="${GT:-runs/wild_v5_song_gt_eval/motion}"
SEED="${SEED:-20260816}"
TAG="${TAG:-m5sel}"
TOPK="${TOPK:-8}"
TEMPERATURE="${TEMPERATURE:-1.0}"

common=(
  --planner-checkpoint "$PLANNER"
  --completion-checkpoint "$COMPLETION"
  --data-root "$RELEASE"
  --audio-dir "$AUDIO"
  --sequence-list "$CLIPS"
  --seed "$SEED" --temperature 1.0
  --plan-stride 15 --plan-fusion vote --plan-vote-tie-break centre
  --plan-transition-policy protect --plan-merge-order shortest
  # --plan-bar-beats 2, NOT the CLI default 4.  The shipped arm used 2 and no
  # artifact recorded it: a first run of this script took the default and
  # produced 8.5 atomic segments per clip against the shipped arm's 14.1, i.e.
  # section 15.4's control instead of the arm under test.  infer_atomic now
  # writes this into the manifest so the next person does not have to infer it
  # from a sentence in the defect log.
  --plan-bar-grid --plan-bar-beats 2 --plan-vote-window 5 --plan-min-segment 6
  --planner-guidance-weight 1.0 --planner-transition-logit-bias 0.0
  --guidance-weight 2.0
  --completion-stride 75 --completion-blend-width 10
  --draft-recurrence-variety --draft-seam-blend 4 --draft-gap-fill interpolate
  --inference-batch-size 1
)

run_arm () {
  local name="$1"; shift
  local out="runs/${TAG}_${name}"
  if [ -d "$out" ] && [ -f "$out/manifest.json" ]; then
    echo "=== ${name}: already present, skipping ==="
    return
  fi
  echo "=== ${name} ==="
  python3 infer_atomic.py "${common[@]}" --output-dir "$out" "$@" \
      > "logs/${TAG}_${name}.log" 2>&1
  python3 - "$out" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1] + "/manifest.json"))
sampling = manifest["sampling"]
print("  rule={} selector_calls={} fallbacks={} clips={}".format(
    sampling["retrieval_rule"], sampling.get("retrieval_selector_calls"),
    sampling.get("retrieval_selector_fallbacks"), len(manifest["names"])))
PY
}

run_arm duration --retrieval-rule duration
run_arm learned  --retrieval-rule learned --retrieval-selector "$SELECTOR" \
                 --retrieval-selector-top-k "$TOPK" \
                 --retrieval-selector-temperature "$TEMPERATURE"

echo "=== scorecard ==="
python3 tools/score_arm_table.py \
    --clips "$CLIPS" --ground-truth-dir "$GT" --audio-dir "$AUDIO" \
    --arm "baseline duration=runs/${TAG}_duration" \
    --arm "learned selector=runs/${TAG}_learned" \
    --json "runs/arm_table_${TAG}.json"

echo "=== beat phase (settle), with the controls ==="
python3 tools/score_beat_phase_shape.py \
    --clips "$CLIPS" --audio-dir "$AUDIO" --ground-truth-dir "$GT" --controls \
    --arm "baseline duration=runs/${TAG}_duration" \
    --arm "learned selector=runs/${TAG}_learned" \
    --json "runs/beat_shape_${TAG}.json"

echo "=== facing spin ==="
python3 tools/score_facing_spin.py \
    --clips "$CLIPS" --ground-truth-dir "$GT" \
    --arm "baseline duration=runs/${TAG}_duration" \
    --arm "learned selector=runs/${TAG}_learned" \
    --json "runs/facing_spin_${TAG}.json" || true
