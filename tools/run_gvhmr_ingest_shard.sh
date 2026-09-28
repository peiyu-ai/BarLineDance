#!/usr/bin/env bash
#
# One shard of GVHMR over the content-cut corpus, cropping the dancer the 2D
# already chose.
#
#   SHARD=<n> NUM_SHARDS=<k> GPU=<i> [INGEST_ROOT=<dir>] bash tools/run_gvhmr_ingest_shard.sh
#
# Two things differ from ``run_gvhmr_wild_shard.sh``, which drives the old
# flat-directory corpus:
#
# 1. **The dancer is given, not re-chosen.**  ``ingest_wild_uploads.py`` wrote
#    ``preprocess/bbx.pt`` per clip, and GVHMR's demo reads that file when it
#    exists instead of running its own YOLOv8 tracker (demo.py:108).  Seeding it
#    is what makes the 2D keypoints and the 3D describe the same person -- which
#    on this corpus is not automatic: 70% of clips have a rival dancer track,
#    and on half the clips checked, GVHMR's independent choice and ours were
#    different people, both of them squarely on a dancer.
# 2. **Clip videos are all named clip.mp4**, inside per-clip directories, and
#    GVHMR keys its output directory on the video's stem -- so every clip would
#    write to the same place.  A staging directory of symlinks named after the
#    clip id fixes that without copying 200 GB.
#
# Resumable: a clip whose quality.json exists is skipped, and a clip that failed
# extraction leaves a marker so a sweep does not re-pay visual odometry on it.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$(pwd)"

: "${SHARD:?set SHARD}"; : "${NUM_SHARDS:?set NUM_SHARDS}"; : "${GPU:?set GPU}"

INGEST_ROOT="${INGEST_ROOT:-$REPO/data/wild_ingest_v1}"
RAW_ROOT="${RAW_ROOT:-$REPO/data/wild3d/ingest_v1_gvhmr_raw}"
CONVERTED_ROOT="${CONVERTED_ROOT:-$REPO/data/wild3d/ingest_v1_converted}"
STAGING="${STAGING:-$REPO/data/wild3d/ingest_v1_videos}"
USE_DPVO="${USE_DPVO:-1}"
VO_ATTEMPTS="${VO_ATTEMPTS:-3}"
EXTRACT_TIMEOUT="${EXTRACT_TIMEOUT:-1800}"
mkdir -p "$RAW_ROOT" "$CONVERTED_ROOT" "$STAGING"

if [ -n "$USE_DPVO" ] && [ "$USE_DPVO" != "0" ]; then
  VO_FLAG="--use-dpvo"
  VO_PYTHONPATH="$REPO/third_party/torch_scatter_compat:$REPO/third_party/pytorch3d_compat:third-party/DPVO:."
else
  VO_FLAG=""
  VO_PYTHONPATH="$REPO/third_party/pytorch3d_compat:."
fi

# The worklist is globbed once, here, and never refreshed.  Starting while the
# cut is still producing therefore does not "pick up the rest later" -- it
# silently omits every clip written after this moment, and the omission is
# invisible afterwards because the shard reports success over the list it had.
# Measured during a resume test: a clip directory appeared 47 seconds after the
# list was built and was simply not in it.
# Match the worker, not the token.  "ingest_wild_uploads.py" alone also matches
# every shell whose command line mentions it -- a monitor polling for the
# workers, or the very command that launched this -- and that is not a
# hypothetical: it blocked all twelve 3D shards once while zero workers were
# actually running.  Same failure as a self-matching wait loop, one level up.
if [ -z "${ALLOW_PARTIAL_INGEST:-}" ] && \
   pgrep -f 'ingest_wild_uploads\.py --videos-dir' >/dev/null; then
  echo "error: ingest_wild_uploads.py is still running; this shard's worklist is" >&2
  echo "       fixed at startup and would omit every clip cut after now." >&2
  echo "       Wait for the cut to finish, or set ALLOW_PARTIAL_INGEST=1 if you" >&2
  echo "       mean to process only what exists (a later sweep fills the rest)." >&2
  exit 1
fi

# Clips whose tracked dancer changes identity mid-clip are excluded here rather
# than filtered downstream, because 3D on them is wasted GPU time: their pose
# splices two people, and what this corpus is for is stable single-dancer
# motion.  ``tools/audit_dancer_tracks.py`` produces the list; 6.27% of clips.
EXCLUDE_JSON="${EXCLUDE_JSON:-$REPO/runs/dancer_track_audit_full.json}"

SHARD_LIST="$(mktemp)"
trap 'rm -f "$SHARD_LIST"' EXIT
python3 - "$INGEST_ROOT" "$SHARD" "$NUM_SHARDS" "$EXCLUDE_JSON" > "$SHARD_LIST" <<'PY'
import hashlib
import json
import pathlib
import sys

root, shard, num_shards = pathlib.Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
excluded = set()
audit = pathlib.Path(sys.argv[4]) if len(sys.argv) > 4 else None
if audit is not None and audit.is_file():
    for record in json.loads(audit.read_text(encoding="utf-8"))["records"]:
        if record.get("switches"):
            excluded.add(record["clip"])
    print("# excluding {} clips whose dancer track switches".format(len(excluded)),
          file=sys.stderr)
for clip in sorted(root.glob("*__clip*")):
    if not (clip / "clip.mp4").exists() or clip.name in excluded:
        continue
    if int(hashlib.sha256(clip.name.encode()).hexdigest(), 16) % num_shards == shard:
        print(clip)
PY

total=0 done_before=0 extracted=0 converted=0 failed=0 seeded=0
while IFS= read -r clip_dir <&3; do
  stem="$(basename "$clip_dir")"
  total=$((total + 1))

  # Link before the skip, not after.  The staging directory is what stage F
  # globs for S3D features, so a clip converted by an earlier run would be
  # skipped here, never linked, and then silently absent from segmentation --
  # present in the 3D bundle and missing from the vocabulary, with nothing
  # reporting the gap.
  video="$STAGING/$stem.mp4"
  [ -L "$video" ] || ln -sf "$clip_dir/clip.mp4" "$video"

  if [ -f "$CONVERTED_ROOT/$stem/quality.json" ]; then
    done_before=$((done_before + 1)); continue
  fi

  # Seed the dancer box before the extractor runs; after it has run, GVHMR has
  # already written its own and seeding would be a lie about what produced it.
  if [ -f "$clip_dir/preprocess/bbx.pt" ] && [ ! -f "$RAW_ROOT/$stem/preprocess/bbx.pt" ]; then
    mkdir -p "$RAW_ROOT/$stem/preprocess"
    cp "$clip_dir/preprocess/bbx.pt" "$RAW_ROOT/$stem/preprocess/bbx.pt"
    seeded=$((seeded + 1))
  fi

  result="$RAW_ROOT/$stem/hmr4d_results.pt"
  if [ ! -f "$result" ]; then
    if [ -f "$RAW_ROOT/$stem/.extract_failed" ]; then
      echo "SKIP_FAILED $stem"; failed=$((failed + 1)); continue
    fi
    # A timeout, because a single clip must never be able to eat a worker.
    # One did: a multiprocessing deadlock in the SLAM reader parked 15 of 21
    # workers in atexit, asleep at 1-6% CPU, for up to five and a half hours,
    # while every liveness check reported a live process.  The deadlock itself
    # is fixed in run_gvhmr_extract.py; this is the backstop for the next one.
    # 1800 s against a ~90 s clip is loose on purpose -- it catches wedges, not
    # slow clips.
    extract_log="$(mktemp)"
    (cd third_party/GVHMR &&
      PYTHONPATH="$VO_PYTHONPATH" CUDA_VISIBLE_DEVICES=$GPU \
      timeout -k 60 "$EXTRACT_TIMEOUT" \
      python "$REPO/tools/run_gvhmr_extract.py" \
        --video "$video" --output-root "$RAW_ROOT" \
        --vo-attempts "$VO_ATTEMPTS" $VO_FLAG) 2>&1 | tee "$extract_log"
    rc="${PIPESTATUS[0]}"
    if [ ! -f "$result" ]; then
      mkdir -p "$RAW_ROOT/$stem"
      # Record *why*, not just *that*.  The marker used to be an empty touch,
      # so classifying 346 failures meant grepping 21 shard logs of 7 MB each
      # -- and the next driver run overwrites those logs.  With the reason in
      # the marker, a later sweep can retry the ones that failed on a defect
      # and leave the ones whose visual odometry genuinely diverged.
      { echo "exit=$rc"
        if [ "$rc" = "124" ] || [ "$rc" = "137" ]; then
          echo "reason=timeout after ${EXTRACT_TIMEOUT}s"
        else
          reason="$(grep -a "EXTRACT_FAIL" "$extract_log" | tail -1)"
          echo "reason=${reason:-no EXTRACT_FAIL line; see the shard log}"
        fi
      } > "$RAW_ROOT/$stem/.extract_failed"
      rm -f "$extract_log"
      echo "FAIL $stem extract rc=$rc"; failed=$((failed + 1)); continue
    fi
    rm -f "$extract_log" "$RAW_ROOT/$stem/.extract_failed"
    extracted=$((extracted + 1))
  fi

  if python3 tools/convert_gvhmr_result.py \
       --result "$result" \
       --extract-meta "$RAW_ROOT/$stem/extract_meta.json" \
       --output-dir "$CONVERTED_ROOT/$stem" >/dev/null &&
     python3 tools/preprocess_wild_3d.py validate \
       --output-dir "$CONVERTED_ROOT/$stem" >/dev/null; then
    converted=$((converted + 1)); echo "OK $stem"
  else
    echo "FAIL $stem convert_or_validate"; failed=$((failed + 1))
  fi
done 3< "$SHARD_LIST"

echo "SHARD_DONE shard=$SHARD total=$total done_before=$done_before seeded=$seeded extracted=$extracted converted=$converted failed=$failed"
