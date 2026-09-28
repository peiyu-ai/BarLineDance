#!/usr/bin/env bash
# Wait for the completion run to FINISH, then run S5 -- as TWO arms, and refuse
# to start at all if the run did not finish.
#
# Two gates on the checkpoint, both able to fail, because "the process is gone"
# and "the training finished" are different facts and a driver handed a
# half-written checkpoint would produce a complete, self-consistent, wrong M6
# table:
#   * the final checkpoint exists at the step the run was sized for, and
#   * the trainer printed its own end-of-run summary at that step.
#
# The wait is on the PID, never on a command-line pattern: `pgrep -f` /
# `pkill -f` match this script's own command line and have killed the
# controlling shell three times in this repository.
#
# WHY TWO ARMS.  run_m6_wild.sh defaults PLAN_BAR_GRID=1, and the justification
# written beside that default is a transition-share measurement on the *acct*
# corpus: the raw planner there emitted transition on 0.5194 of frames and the
# grid brought it to 0.3482 against a ground truth of 0.3216.  Its criterion is
# therefore "does the plan's transition share match the ground truth's".
#
# This corpus's planner does not over-emit transition -- its own end-of-training
# sample sits at 0.2675 -- and measured here on 100 test clips at seed
# 20260816, switching only the grid:
#
#                        GT      grid on     grid off
#   transition median   0.1296    0.0000       0.2730
#   clips at exactly 0  10/100    76/100        0/100
#   |err| median          --      0.1168       0.1518   <- see below
#   spearman vs GT        --      +0.148       +0.368
#                                 p=0.141      p=1.65e-4
#   segment ratio        1.00      0.53         1.62
#   within 2x             --      59/100       88/100
#
# The |err| row is the one that must NOT decide this: an all-zero predictor
# scores 0.1296 on it and so beats the ungridded arm, which disqualifies the
# criterion rather than the arm.  The three criteria that a floor predictor
# fails -- tracking the clip-to-clip variation, the segment ratio, and not
# collapsing 76% of clips to no transition at all -- all point the same way.
#
# But the headline is FID, and the driver records that fid_k could not separate
# these two arms on the acct corpus; "could not separate there" is not "cannot
# separate here".  Generation is cheap next to the wait, so both arms run at the
# full count, on the SAME clip list, and both get reported.  The choice is then
# made on 400 clips and on FID, not on a plan-level probe.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PID=543206
STEPS=135935
CKPT="runs/completion_wild_v5_song_v5rekey/completion_step${STEPS}.pt"
LOG="logs/wild_v5_song_v5rekey_completion.log"
NOTE="logs/m6_wild_v5_song_v5rekey_watcher.log"

say() { echo "[$(date -u '+%F %T')] $*" | tee -a "$NOTE"; }

say "waiting on completion pid ${PID} (expects ${CKPT})"
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
say "pid ${PID} is gone; checking that it finished rather than died"

[ -s "$CKPT" ] || { say "REFUSED: ${CKPT} absent -- completion did not reach step ${STEPS}. \
Last log line: $(tail -c 20000 "$LOG" | tr '\r' '\n' | grep -v '^$' | tail -1)"; exit 1; }
tail -c 20000 "$LOG" | tr '\r' '\n' | grep -q "\"step\": ${STEPS}" \
  || { say "REFUSED: ${LOG} carries no end-of-run summary at step ${STEPS}; the checkpoint \
exists but the run did not report finishing"; exit 1; }
say "completion finished: ${CKPT}"

# GPU 7 belongs to another project (66 GB resident, pid 3534456); 0-6 are ours
# once completion releases card 1.
export TAG=wild_v5_song \
       RELEASE=/dev/shm/atomicdance-song-v5rekey/release_v1 \
       BUNDLE=/dev/shm/atomicdance-song-v5rekey/performance \
       CKPT_P=runs/planner_wild_v5_song_v5rekey_x0/planner_step${STEPS}.pt \
       CKPT_C="$CKPT" \
       GT_FEATURES=runs/wild_v5_song_gt_features \
       AUDIO=runs/wild_v5_song_gt_eval/audio \
       CLIPS=runs/wild_v5_song_m6_clips \
       PAIRS=runs/wild_v5_music_groups_pairs.jsonl \
       GPUS="0 1 2 3 4 5 6" \
       FROM=1 TO=3
# FROM=1: the 400-clip selection is already built and audited (400 clips, 377
# distinct uploads, flagged_in_selection 0).  Re-deriving it here would put the
# run on a list nobody has checked, for no gain -- and both arms must be scored
# on the SAME clips or the FIDs are not comparable.
# TO=3: stop at the headline table.  Step 4 re-features the whole generated
# release and is hours; it is a separate decision once the FIDs are in.

status=0
for arm in grid nogrid; do
  case "$arm" in
    grid)   export PLAN_BAR_GRID=1 OUT=runs/m6_wild_v5_song_v5rekey ;;
    nogrid) export PLAN_BAR_GRID=0 OUT=runs/m6_wild_v5_song_v5rekey_nogrid ;;
  esac
  say "launching M6 arm '${arm}': $(basename "$OUT"), bargrid=${PLAN_BAR_GRID}, cards [$GPUS], steps ${FROM}..${TO}"
  bash tools/run_m6_wild.sh >> "logs/$(basename "$OUT")_driver.log" 2>&1
  rc=$?
  say "M6 arm '${arm}' exited ${rc}"
  [ "$rc" -eq 0 ] || status="$rc"
done
say "both arms done; overall ${status}"
exit "$status"
