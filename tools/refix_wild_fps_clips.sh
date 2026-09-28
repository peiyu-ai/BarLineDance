#!/usr/bin/env bash
#
# Re-cut, re-pose and re-derive every clip whose upload is not 30 fps.
#
# What went wrong: cut_clip selected video by frame number and audio by seconds
# computed as frame/30, so an upload at any other rate produced a clip whose
# picture and sound covered different spans -- 2x apart on a 60 fps upload --
# and GVHMR ran on that clip, so the 3D motion inherited the stretch while the
# music did not.  1,418 of 13,783 released clips (10.3%) carry it.
#
# The 2026-08-19 run of this script fixed one family and created another.  It
# cut the sound by ``source_fps``, which it took from ``r_frame_rate`` alone --
# and on a container whose two rate claims disagree, that is the claim that is
# wrong.  Measured 2026-08-25 over the 230 clips this corpus currently produces
# from a variable-rate upload: 22 were re-cut that day, and of those **17 were
# made worse** -- 15 of them from a picture/sound ratio of ~0.998, i.e. already
# correct, to ~1.995 -- against 5 made better.  The other 208 were never re-cut
# at all, because that run selected uploads whose rate "is not 30" and 121 of
# these report r_frame_rate exactly 30.0 while holding 25 fps of content.
# The fix is --cfr-cache: normalise onto a constant rate once, then cut.
#
# Three things about this job that are not obvious:
#
# * **The spans change, not just the sampling.**  min_frames and max_seconds are
#   policies about seconds, so on a 60 fps upload they now cover twice as many
#   source frames.  An upload that used to split into two clips can come back as
#   one.  Clip *names* are upload + ordinal, so a surviving name holds new
#   content -- which is what we want on OSS, where writes overwrite -- but a
#   name that stops being produced becomes an orphan object that these
#   credentials cannot delete.  The orphan list is written out; nothing may
#   consume this corpus by globbing a directory.
# * **Stale outputs must be removed, not skipped.**  Every stage of
#   run_wild_rebuild.sh skips work whose output already exists, and the affected
#   clips' outputs all exist -- they are simply wrong.
# * **The detection scan is reusable.**  scan_upload's cache is per source frame
#   and the defect never touched it, so the expensive pass over the uploads is
#   not repeated; only the per-clip pose is.
#
# Usage:
#   GPUS="0 1 2 3 4 5 6" bash tools/refix_wild_fps_clips.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$(pwd)"

SCRATCH="${SCRATCH:-/cache/atomicdance-assets/scratch/c1/refix}"
UPLOADS="${UPLOADS:-$SCRATCH/uploads}"
INGEST="${INGEST:-/cache/atomicdance-assets/data/wild_ingest_v1}"
SCAN_CACHE="${SCAN_CACHE:-/cache/atomicdance-assets/runs/ingest_scan_cache}"
# Uploads whose avg_frame_rate and r_frame_rate disagree are re-encoded onto a
# constant rate here and cut from that.  Without this directory the ingest
# REFUSES them (status variable_frame_rate) and cuts nothing -- a correct
# refusal and a silently empty run, so the flag is not optional.  It lives on
# /cache, not the NAS: these are full re-encodes of whole uploads and
# /workspace is directory-quota limited (CLAUDE.md 1.2).
CFR_CACHE="${CFR_CACHE:-/cache/atomicdance-assets/scratch/c1/cfr_uploads}"
MAX_SECONDS="${MAX_SECONDS:-24}"
GPUS="${GPUS:-0 1 2 3}"
VENV="${VENV:-$REPO/.venv_ortgpu/bin/python}"

[ -x "$VENV" ] || { echo "no DWPose venv at $VENV; run tools/setup_dwpose_env.sh"; exit 1; }
[ -d "$UPLOADS" ] || { echo "no upload set at $UPLOADS"; exit 1; }
mkdir -p "$SCRATCH/logs" "$CFR_CACHE"

# The baseline is the one artifact that cannot be recomputed.  Once a single
# upload has been re-cut, "what this corpus produced before" is gone from disk:
# the new meta.json has overwritten the old span and the clip directories that
# stopped being produced look exactly like the ones that never existed.  A
# re-run of this script -- after an interruption, after a smoke test, after a
# shard died -- must therefore keep the first baseline rather than take a fresh
# one, or the orphan list comes back empty and reads as "nothing was orphaned".
if [ -s "$SCRATCH/clips_before.json" ]; then
  echo "=== before: keeping the existing baseline ==="
  echo "  $SCRATCH/clips_before.json ($(date -r "$SCRATCH/clips_before.json" '+%Y-%m-%d %H:%M'))"
  echo "  delete it deliberately if you mean to re-baseline a corpus that has"
  echo "  not been re-cut yet; taking a fresh one after a re-cut loses the orphans."
else
  echo "=== before: what these uploads currently produce ==="
  python3 tools/refix_wild_fps_clips_census.py --uploads "$UPLOADS" --ingest "$INGEST" \
    --output "$SCRATCH/clips_before.json" || exit 1
fi

# The uploads are already recorded in the ingest manifests -- that is what
# makes them a *re*-cut -- and the resume gate skips anything recorded.  On
# 2026-08-19 that gate ate the entire run: seven shards printed "1335 of 1335
# already recorded", did nothing, exited 0, and the census afterwards honestly
# reported no orphans and no changes, because nothing had changed.  Naming the
# uploads is what suspends the gate for exactly this set.
ls "$UPLOADS" | sed 's/\.[^.]*$//' | sort > "$SCRATCH/redo_uploads.txt"

# Stamped before the first shard starts, and read by the gate below.  Every
# manifest row carries ``ingested_at``, so this is what separates rows this run
# wrote from rows that were already there -- and that separation is the only
# thing that lets the gate fail.  A gate that reads the manifest without a
# cutoff passes on last generation's rows.
RUN_STARTED="$(date +%s)"
echo "=== re-ingest ($(wc -l < "$SCRATCH/redo_uploads.txt") uploads) ==="
n=0
for gpu in $GPUS; do
  CUDA_VISIBLE_DEVICES="$gpu" "$VENV" tools/ingest_wild_uploads.py \
    --videos-dir "$UPLOADS" --out-root "$INGEST" --max-seconds "$MAX_SECONDS" \
    --scan-cache "$SCAN_CACHE" --cfr-cache "$CFR_CACHE" \
    --shard "$n" --num-shards "$(echo $GPUS | wc -w)" \
    --redo "$SCRATCH/redo_uploads.txt" \
    --device cuda > "$SCRATCH/logs/ingest_$gpu.log" 2>&1 &
  n=$((n + 1))
done
wait
echo "=== after: what they produce now, and what became an orphan ==="
python3 tools/refix_wild_fps_clips_census.py --uploads "$UPLOADS" --ingest "$INGEST" \
  --output "$SCRATCH/clips_after.json" --compare "$SCRATCH/clips_before.json" \
  --orphan-list "$SCRATCH/orphans.txt" --audit-list "$SCRATCH/to_audit.txt"

# Did the re-ingest do anything, and did what it produced come out right?
# Nothing above answers either: a run that skipped every upload produces a
# census identical to the baseline, an empty orphan list, and a freshness audit
# in which everything agrees -- four clean readings from a corpus nobody
# touched.
#
# The gate this script carried until 2026-08-25 asked whether any clip records
# a ``source_fps``.  On this corpus that could not fail: the 2026-08-19 run had
# already written the field into 22 clips, so a run that refused every upload
# for want of --cfr-cache would have printed "22 of 230" and passed.  The check
# now reads the two rates the clip was cut from -- a field that did not exist
# before 2026-08-25, so no previous generation can satisfy it -- and counts
# only manifest rows written after RUN_STARTED.
echo "=== did the re-cut actually happen, and is the output fps-correct? ==="
python3 tools/check_recut_happened.py --census "$SCRATCH/clips_after.json" \
  --ingest-root "$INGEST" --redo "$SCRATCH/redo_uploads.txt" \
  --since "$RUN_STARTED" --output "$SCRATCH/recut_gate.json"
gate=$?
if [ "$gate" -ne 0 ]; then
  echo "stopping: the re-cut did not happen or did not come out clean, so"
  echo "nothing below would mean anything"
  exit "$gate"
fi

# The two manifests answer different questions and neither substitutes for the
# other.  The census says which clips *should exist*; the audit says which
# derived artifacts were built from bytes the clip no longer has.  An orphan is
# consistent with its own artifacts -- the old 3D was extracted from the old
# video and their hashes agree -- so the audit calls it fresh.  Freshness is not
# membership, and a consumer that applies only the audit keeps the orphans.
echo "=== which derived artifacts are now stale ==="
# Exits non-zero when anything disagrees, which after a re-cut is the expected
# state -- the run continues, because the disagreement list is the product.
python3 tools/audit_clip_freshness.py --clips "$SCRATCH/to_audit.txt" \
  --output "$SCRATCH/freshness.json" --workers 16 || true

echo
echo "=== what exists now, and what has to consume it ==="
echo "  $SCRATCH/clips_before.json  what these uploads produced as released"
echo "  $SCRATCH/clips_after.json   what they produce now, plus the comparison"
echo "  $SCRATCH/orphans.txt        names no consumer may read.  These objects"
echo "                              cannot be deleted -- PUT over an existing key"
echo "                              succeeds, DELETE answers 403 AccessDenied"
echo "                              because of bucket acl (both measured"
echo "                              2026-08-19) -- so exclusion by name is the"
echo "                              only mechanism there is."
echo "  $SCRATCH/freshness.json     stale.3d and stale.s3d: what to re-derive."
echo
echo "Nothing has been re-derived yet, and every stage of run_wild_rebuild.sh"
echo "skips work whose output already exists -- so a plain re-run would skip"
echo "exactly these clips, which are the ones that are wrong.  Stage F takes the"
echo "list directly:"
echo
echo "  REDO=$SCRATCH/freshness.json SHARD=n NUM_SHARDS=N GPU=g \\"
echo "    python3 tools/run_wild_s3d_shard_oss.py"
echo
echo "Stage B has no such flag yet; its stale list is stale.3d in the same file."
