#!/usr/bin/env bash
#
# The whole wild rebuild, from raw uploads to the atomic vocabulary, as one
# resumable command.
#
#   A  ingest      raw upload -> content-cut clips (2D, audio, dancer bbx),
#                  then the dancer-track continuity audit
#   B  3D          GVHMR per clip, cropping the bbx the ingest chose
#   C  staging     inventory -> staging manifest -> reconcile
#   D  audio       35-D features, frame-aligned to the converted motion
#   E  bundle      performance bundle -> normalizer -> normalized, then the
#                  3D quality audit against motion capture
#   F  S3D         visual features for Alg. 1
#   G  discovery   M1 segmentation -> M2 TMR -> M3 caption/summarise/recluster
#
# G pins Alg. 1's two free parameters rather than taking the discovery script's
# defaults.  Not pinning them silently changed the corpus between versions:
# wild_v2 was built at 34/18 and this driver would have built v4 at the 32/18
# default -- so the comparison the whole rebuild exists to make, same uploads
# cut two ways, would have carried two variables instead of one.
#
# 34/18 because that is what wild_v2 used, and for no stronger reason.  The
# vocabulary ceiling was tried as a tie-break and does not settle it: on the
# test clips 34/18 rebuilds at ground truth (0.3358 against GT 0.3380) and
# 32/18 manages 0.4121, but on val the order reverses and 36/20 wins.  The
# metric is deterministic across seeds -- retrieval is duration-nearest-same-
# label -- which is not the same as robust, and a ranking that flips with the
# evaluation set is not a ranking.
#
# It exists because the rebuild is ~70 GPU-hours of wall clock across six
# cards: no single sitting covers it, and a chain of hand-run commands loses
# its place.  Every stage skips work whose output already exists, so this can
# be re-launched after an interruption and will resume rather than restart.
#
# Usage:
#   TAG=wild_v4 MAX_SECONDS=24 GPUS="0 1 2 3 4 5" nohup bash tools/run_wild_rebuild.sh &
#
#   FROM=<stage letter>   start here, assuming everything before it is done
#   ONLY=<stage letter>   run just this stage and stop
#
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$(pwd)"
mkdir -p logs runs

TAG="${TAG:-wild_v4}"
VIDEOS="${VIDEOS:-data/wild_videos_20260811}"
GPUS="${GPUS:-0 1 2 3 4 5}"
MAX_SECONDS="${MAX_SECONDS:-24}"          # tools/scan_clip_length.py, 2026-08-11
FROM="${FROM:-A}"
ONLY="${ONLY:-}"
# Shards per GPU.  One worker leaves a card ~45% idle because each clip spends
# real time decoding video and writing files; two took the ingest from 44% to
# ~70% utilisation.  Raise only after checking nvidia-smi, not on principle.
PER_GPU="${PER_GPU:-2}"

INGEST="data/wild_ingest_v1"
RAW3D="data/wild3d/ingest_v1_gvhmr_raw"
CONV3D="data/wild3d/ingest_v1_converted"
STAGE_DIR="data/wild3d/${TAG}_staging"
INVENTORY="runs/${TAG}_inventory.jsonl"
HMR_SEQ="${STAGE_DIR}/sequences_hmr.jsonl"
AUDIO_DIR="data/wild3d/${TAG}_audio35"
BUNDLE="data/wild3d/${TAG}_performance"
NORMALIZER="data/wild3d/${TAG}_normalizer"
NORMALIZED="data/wild3d/${TAG}_normalized"
# Overridable, because S3D features are not per-corpus: they are keyed by
# clip stem, every tag that contains a clip wants the same file, and the
# OSS fleet (tools/run_stage_f_oss_fleet.sh) publishes them all to the one
# shared root.  A re-split corpus therefore reuses the features it already
# has rather than recomputing 17k of them under a new name.
S3D_DIR="${S3D_DIR:-data/wild_visual_s3d_${TAG}}"
S3D_VIDEOS="data/wild3d/ingest_v1_videos"

gpu_list=($GPUS); n=${#gpu_list[@]}
step() { echo; echo "== [$(date '+%F %T')] $* =="; }
want() {   # want <letter>
  [ -n "$ONLY" ] && { [ "$ONLY" = "$1" ]; return; }
  [[ "$1" > "$FROM" || "$1" == "$FROM" ]]
}
fan_out() {  # fan_out <log-prefix> <command template using {i}, {gpu}, {n}>
  local prefix="$1"; shift
  local per="${FAN_PER_GPU:-1}"
  local total=$((n * per))
  local pids=()
  for i in $(seq 0 $((total - 1))); do
    local gpu="${gpu_list[$((i % n))]}"
    local cmd="${*//\{i\}/$i}"; cmd="${cmd//\{gpu\}/$gpu}"
    cmd="${cmd//\{n\}/$total}"
    bash -c "$cmd" > "logs/${prefix}_${i}.log" 2>&1 &
    pids+=($!)
  done
  local bad=0
  for pid in "${pids[@]}"; do wait "$pid" || bad=$((bad+1)); done
  if [ "$bad" -gt 0 ]; then
    echo "   ERROR: $bad/$total shards exited non-zero (see logs/${prefix}_*.log)" >&2
    # Warning-and-continue was wrong here and it cost a whole run: a guard
    # blocked all twelve 3D shards, the driver carried on, and C-G completed on
    # the 11 clips that happened to exist -- producing a full set of plausible
    # artifacts over 0.06% of the corpus.  Each stage feeds the next, so a stage
    # that did not run must stop the chain rather than hand it an empty input.
    return 1
  fi
}

if want A; then
  step "A ingest: content cut at ${MAX_SECONDS}s"
  MAX_SECONDS="$MAX_SECONDS" GPUS="$GPUS" STAGE=cut OUT_ROOT="$INGEST" \
    bash tools/run_wild_ingest.sh 2>&1 | tail -25
  echo "   clips: $(ls -d "$INGEST"/*__clip*/ 2>/dev/null | wc -l)"
  # Track continuity, straight after the cut: the IoU bridge that keeps a take
  # alive through a turn also lets the track walk onto a second dancer, and the
  # clip then carries 2D, a crop and 3D that splice two people under one id.
  # Measured at 6.27% of clips.  It reads detections.npz only, so it costs
  # minutes on CPU and the flag exists before anything downstream consumes the
  # corpus.  A flag, not a filter -- who should drop these depends on the reader.
  [ -s "runs/${TAG}_dancer_tracks.json" ] || python3 tools/audit_dancer_tracks.py \
    --ingest-root "$INGEST" --output "runs/${TAG}_dancer_tracks.json" 2>&1 | tail -20
  [ -n "$ONLY" ] && exit 0
fi

if want B; then
  step "B 3D: GVHMR over the cut corpus, dancer bbx seeded"
  FAN_PER_GPU="$PER_GPU" fan_out gvhmr_ingest \
    "SHARD={i} NUM_SHARDS={n} GPU={gpu} INGEST_ROOT=$REPO/$INGEST RAW_ROOT=$REPO/$RAW3D CONVERTED_ROOT=$REPO/$CONV3D bash tools/run_gvhmr_ingest_shard.sh" \
    || { echo "stage B did not run; refusing to build C-G on a partial corpus" >&2; exit 1; }
  echo "   converted: $(ls -d "$CONV3D"/*/ 2>/dev/null | wc -l)"
  [ -n "$ONLY" ] && exit 0
fi

if want C; then
  step "C staging: inventory -> staging manifest -> reconcile"
  # ORPHANS is not optional on a corpus that has been re-cut.  ``inventory``
  # globs the ingest tree, and nothing ever deletes a clip directory: an upload
  # that used to yield three clips and now yields two leaves the third on disk,
  # indistinguishable from a current one.  Measured 2026-08-25: 17,985
  # directories carry a meta.json while the manifests produce 17,225, so the
  # glob hands back 760 orphans -- whose 3D, features and music were all built
  # from their own old bytes, so every hash agrees and no freshness gate
  # downstream can see them.  Regenerate the list with:
  #   python3 - <<'PY'  ... manifest_rows(INGEST) minus the dirs on disk ...
  # or take it from tools/refix_wild_fps_clips_census.py --orphan-list.
  ORPHANS="${ORPHANS:-runs/ingest_orphans_20260825.txt}"
  exclude_arg=()
  if [ -s "$ORPHANS" ]; then
    exclude_arg=(--exclude "$ORPHANS")
    echo "   excluding $(wc -l < "$ORPHANS") orphan clip name(s) from the inventory"
  else
    echo "   WARNING: no orphan list at $ORPHANS; the inventory will include every"
    echo "   directory on disk, which on a re-cut corpus re-admits clips no run"
    echo "   produces.  Set ORPHANS= explicitly to say you meant that."
  fi
  # --upload-root is what re-cut clips get rewritten to name.  Their meta.json
  # records the scratch working copy they were cut from -- 2,185 clips under
  # scratch/c1/refix/uploads (2026-08-19) and 278 under scratch/c1/cfr_uploads
  # (the CFR re-encodes) -- and that directory is working space, not corpus.
  [ -s "$INVENTORY" ] || python3 tools/preprocess_wild_3d.py inventory \
    --cache-root "$INGEST" "${exclude_arg[@]}" --upload-root "$VIDEOS" \
    --output "$INVENTORY" 2>&1 | tail -12
  [ -d "$STAGE_DIR" ] || python3 tools/preprocess_wild_3d.py build-wild-staging-manifest \
    --inventory "$INVENTORY" --output-dir "$STAGE_DIR" --corpus "$TAG" 2>&1 | tail -12
  [ -s "$HMR_SEQ" ] || python3 tools/preprocess_wild_3d.py reconcile-wild-hmr \
    --staging-sequences "$STAGE_DIR/sequences.jsonl" --converted-root "$CONV3D" \
    --output "$HMR_SEQ" 2>&1 | tail -12
  # What a manifest may name: a repo-relative path, which is also the OSS key
  # its bytes live under.  What it may not name is anywhere that stops existing
  # -- the evictable cache mount, the scratch directories the re-cuts worked in
  # -- or anywhere absolute, because an absolute path is not a key and
  # run_wild_stage_c_oss.publish_rows will not publish a row holding one.
  # Checking only "/cache" passed a manifest naming 2,463 scratch files on
  # 2026-08-25, so the check is on the shape of the path, not on one prefix.
  python3 - "$HMR_SEQ" <<'GATE' || exit 1
import json, sys
bad = {}
with open(sys.argv[1], encoding="utf-8") as handle:
    for number, line in enumerate(handle, 1):
        row = json.loads(line)
        stack = [("", row)]
        while stack:
            path, node = stack.pop()
            if isinstance(node, dict):
                stack.extend((path + "." + k, v) for k, v in node.items())
            elif isinstance(node, list):
                stack.extend((path + "[]", v) for v in node)
            elif isinstance(node, str) and (
                    node.startswith("/") or node.startswith("scratch/")):
                bad.setdefault(path, [0, node, number])
                bad[path][0] += 1
if bad:
    print("   ERROR: {} names locations that will not outlive the run:".format(sys.argv[1]))
    for field, (count, example, number) in sorted(bad.items()):
        print("     {} in {} row(s), e.g. line {}: {}".format(field, count, number, example))
    print("   Manifests carry repo-relative keys.  Fix the writer, not the file.")
    raise SystemExit(1)
GATE
  [ -n "$ONLY" ] && exit 0
fi

if want D; then
  step "D audio: 35-D features aligned to the converted frame_ids"
  [ -d "$AUDIO_DIR" ] || python3 tools/extract_wild_music_features.py \
    --input-sequences "$HMR_SEQ" --output-dir "$AUDIO_DIR" 2>&1 | tail -15
  [ -n "$ONLY" ] && exit 0
fi

if want E; then
  step "E bundle -> normalizer -> normalized"
  [ -d "$BUNDLE" ] || python3 tools/build_wild_performance_bundle.py \
    --audio-manifest "$AUDIO_DIR/sequences_audio.jsonl" --output-dir "$BUNDLE" 2>&1 | tail -12
  [ -d "$NORMALIZER" ] || python3 tools/fit_motion_normalizer.py \
    --sequence-manifest "$BUNDLE/sequences.jsonl" --source-manifest "$BUNDLE/sources.jsonl" \
    --output-dir "$NORMALIZER" 2>&1 | tail -12
  [ -d "$NORMALIZED" ] || python3 tools/apply_motion_normalizer.py \
    --sequence-manifest "$BUNDLE/sequences.jsonl" --source-manifest "$BUNDLE/sources.jsonl" \
    --normalizer-bundle "$NORMALIZER" --output-dir "$NORMALIZED" 2>&1 | tail -12

  # The cut traded clip length for content alignment: content clips average
  # 17.4 s against the blind cut's 13.2 s, and the length gate measured drift
  # rising with length.  Whether that trade cost 3D quality is answerable only
  # here, on the real corpus, against motion capture -- so it runs automatically
  # rather than waiting for somebody to think of it.  The published corpus's
  # numbers (16 s clips) are in runs/wild_3d_quality_audit.json for comparison.
  [ -s "runs/${TAG}_3d_quality.json" ] || python3 tools/audit_wild_3d_quality.py \
    --bundle "$BUNDLE" --output "runs/${TAG}_3d_quality.json" \
    --reference data/atomic_aistpp/aist_raw_performance_v1 \
    2>&1 | tail -25
  [ -n "$ONLY" ] && exit 0
fi

if want F; then
  step "F S3D features for Alg. 1"
  mkdir -p "$S3D_DIR"
  fan_out s3d_${TAG} \
    "CUDA_VISIBLE_DEVICES={gpu} python3 tools/extract_wild_s3d_shard.py --shard {i} --num-shards {n} --video-dir $S3D_VIDEOS --output-dir $S3D_DIR" \
    || { echo "stage F did not run; refusing to segment an incomplete feature set" >&2; exit 1; }
  echo "   features: $(ls "$S3D_DIR"/*.npz 2>/dev/null | wc -l)"
  [ -n "$ONLY" ] && exit 0
fi

if want G; then
  # Person boxes come from the ingest, not from the GVHMR run: the ingest is
  # where the dancer was chosen, it has a box for every clip whether or not the
  # 3D succeeded, and the copy under RAW3D is the same file seeded from it.
  step "G discovery M1-M3"
  # Vocabulary size is pinned to the paper's, not to corpus size.
  #
  # Sub-prototypes are "members / TARGET_SIZE" per prototype, so a fixed
  # TARGET_SIZE makes the vocabulary a linear function of how much footage there
  # is: at the inherited 32 this corpus would emit ~3,300 sub-classes against
  # wild_v2's 857, purely because it is bigger.  Nothing measured supports that,
  # and the paper's own sweep has 100 base clusters beating 125 (FID_k 32.68 vs
  # 34.57) -- more is not better, it over-fragments.
  #
  # The paper reports 100 prototypes x 7.3 sub-prototypes = 730.  So the size is
  # derived from the caption count that M3 will actually see, once it is known,
  # rather than guessed: TARGET_SIZE = captions / 730.  Override to compare.
  if [ -z "${TARGET_SIZE:-}" ] && [ -s "runs/${TAG}_captions/captions.jsonl" ]; then
    TARGET_SIZE="$(python3 -c "
import sys
n = sum(1 for _ in open(sys.argv[1], encoding='utf-8'))
print(max(1, round(n / 730.0)))" "runs/${TAG}_captions/captions.jsonl")"
    echo "   TARGET_SIZE=${TARGET_SIZE} (paper ratio: 100 x 7.3 = 730 sub-classes)"
  fi
  # The pre-split key: choreographer account, from metadata.  See
  # tools/recluster_atomics_ingroup.py::genre_of for why it is not a predicted
  # genre.  Built once here so a re-run cannot drift from the corpus.
  [ -s "runs/${TAG}_group_keys.json" ] || python3 - > "runs/${TAG}_group_keys.json" <<'PYK'
import json, pathlib
acct = {}
for line in pathlib.Path("logs/wild_source_20260811.state/oss_ids.txt").read_text(
        encoding="utf-8").splitlines():
    line = line.strip()
    if "/" in line:
        account, video = line.rsplit("/", 1)
        acct[video] = account
print(json.dumps(acct, ensure_ascii=False))
PYK
  GROUP_KEYS="$REPO/runs/${TAG}_group_keys.json" \
  GENRE_SPLIT="--genre-split" \
  TARGET_SIZE="${TARGET_SIZE:-124}" \
  FRAMES_PER_CLUSTER="${FRAMES_PER_CLUSTER:-34}" MIN_LENGTH="${MIN_LENGTH:-18}" \
  TAG="$TAG" FEATURES_DIR="$S3D_DIR" BUNDLE="$BUNDLE" VIDEO_DIR="$S3D_VIDEOS" \
    NORMALIZED="$NORMALIZED/sequences_normalized.jsonl" NORMALIZER="$NORMALIZER" \
    SOURCES="$BUNDLE/sources.jsonl" PERSON_BOXES="$REPO/$INGEST" \
    GPUS="$GPUS" bash tools/run_atomic_discovery.sh
fi

step "rebuild driver finished"
