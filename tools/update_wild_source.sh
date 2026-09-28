#!/usr/bin/env bash
# =============================================================================
# update_wild_source.sh — refresh the wild video supply for AtomicDance
#
# The curated-account crawler lives in the Lodge repo and stays there: its f2
# request signing is the hard part, it is maintained against a moving target,
# and a second copy here would rot silently while still appearing to work.  So
# stage 1 *calls* Lodge's fetcher.  Everything after it is ours, because the
# question "what is new" has a different answer for each repo -- Lodge asks it
# against its own merged corpus, and running Lodge's diff would mark videos as
# already-processed that this repo has never seen.
#
#   STAGE 1  fetch      Lodge's 20 curated accounts -> OSS      (their script)
#   STAGE 2  diff       OSS listing vs *this* repo's manifests  (ours)
#   STAGE 3  download   new mp4s into data/wild_videos_<TAG>    (ours)
#   STAGE 4  report     what is newly available, and what is
#                       already downloaded but never reconstructed
#
# Stage 4 exists because of what it found the first time it ran: the corpus was
# not short of video at all.  6,041 clips were segmented and 2D-tracked, 3,373
# of them had never been through GVHMR, and the 2,434-clip release was the
# result of an *audio* filter, not a shortage of footage.  Fetching more video
# before draining that backlog would add to the wrong end of the pipeline.
#
# Usage:
#   nohup bash tools/update_wild_source.sh > logs/wild_source_$(date +%Y%m%d).log 2>&1 &
#
#   TAG=20260811          batch label; same TAG resumes rather than restarts
#   MAX_PER_ACCOUNT=100   newest N per account (500 for a full backfill)
#   SKIP_FETCH=1          digest an existing OSS backlog without crawling
#   DRY_RUN=1             report the funnel and exit before touching anything
#
# Prerequisites all live in the Lodge checkout and are *not* duplicated here:
# .venv_f2, .secrets/ossutilconfig, www.douyin.com_cookies.txt.  A stale cookie
# is the usual failure and shows up as listed=0, which stage 1 treats as fatal
# rather than as "no new videos" -- the two look identical in the output and
# only one of them means the corpus is up to date.
# =============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
LODGE="${LODGE:-$(cd "$REPO/../Lodge" 2>/dev/null && pwd)}"
TAG="${TAG:-$(date +%Y%m%d)}"
MAX_PER_ACCOUNT="${MAX_PER_ACCOUNT:-100}"
SKIP_FETCH="${SKIP_FETCH:-0}"
DRY_RUN="${DRY_RUN:-0}"

OSS_PREFIX="${OSS_ASSET_PREFIX:-oss://example-bucket/example/prefix/}data/tiktok"
OSSUTIL="${OSSUTIL:-/opt/data-infra/ossutil64}"
OSSCONF="$LODGE/.secrets/ossutilconfig"
MANIFEST_ROOT="data/wild3d/source_manifests/tiktok_new_b2gpu"
DL_DIR="data/wild_videos_$TAG"
STATE="logs/wild_source_$TAG.state"; mkdir -p "$STATE" logs
PENDING="$STATE/pending_videos.txt"

say(){ echo "[$(date '+%F %T')] $*"; }
die(){ say "FATAL: $*"; exit 1; }
stage_done(){ [ -f "$STATE/$1.done" ]; }
mark_done(){ touch "$STATE/$1.done"; say "STAGE $1 DONE"; }

say "=== STAGE 0 preflight (TAG=$TAG, LODGE=$LODGE) ==="
[ -d "$LODGE" ] || die "Lodge checkout not found; set LODGE=<path>"
[ -x "$OSSUTIL" ] || die "ossutil missing at $OSSUTIL"
[ -f "$OSSCONF" ] || die "oss config missing at $OSSCONF"
[ -x "$LODGE/.venv_f2/bin/python" ] || die "$LODGE/.venv_f2 missing; see Lodge scripts/update_wild_data.sh"

# ---------- STAGE 4a: the backlog already on disk ----------
# Printed first, on purpose.  It is the number that decides whether fetching
# more video is the useful move at all.
say "=== STAGE 4a backlog: what is already here but not reconstructed ==="
python3 - <<'PY'
import json, pathlib
root = pathlib.Path("data/wild3d/source_manifests/tiktok_new_b2gpu")
path = root / "sequences_hmr_v3.jsonl"
if not path.is_file():
    print("  no hmr manifest yet"); raise SystemExit
rows = [json.loads(l) for l in path.open(encoding="utf-8")]
from collections import Counter
status = Counter((r.get("qc") or {}).get("hmr_status") for r in rows)
audio_path = root / "audio_35_v3" / "sequences_audio.jsonl"
audio = Counter()
if audio_path.is_file():
    audio = Counter((json.loads(l).get("audio_feature") or {}).get("status")
                    for l in audio_path.open(encoding="utf-8"))
print("  clips with 2D + segments : {}".format(len(rows)))
for name, count in status.most_common():
    print("    3D {:<12}: {}".format(str(name), count))
for name, count in audio.most_common():
    print("    audio {:<9}: {}".format(str(name), count))
pending = [r for r in rows if (r.get("qc") or {}).get("hmr_status") == "pending"]
uploads = {r["legacy_clip_id"].split("__")[0] for r in pending}
print("  -> {} clips from {} uploads are ready for GVHMR with no new download"
      .format(len(pending), len(uploads)))
PY

[ "$DRY_RUN" = "1" ] && { say "DRY_RUN=1, stopping before fetch"; exit 0; }

# ---------- STAGE 1: fetch (Lodge's crawler) ----------
if [ "$SKIP_FETCH" = "1" ] || stage_done fetch; then
  say "=== STAGE 1 fetch: skip ==="
else
  say "=== STAGE 1 fetch: curated accounts, max_per_account=$MAX_PER_ACCOUNT ==="
  ( cd "$LODGE" && .venv_f2/bin/python scripts/mining/phase1_fetch_videos_f2.py \
      --max_per_account "$MAX_PER_ACCOUNT" ) 2>&1 \
    | grep -v "HTTP Request" | tee "logs/wild_source_${TAG}_fetch.log"
  TOTAL_LINE="$(grep '=== TOTAL' "logs/wild_source_${TAG}_fetch.log" | tail -1 || true)"
  [ -n "$TOTAL_LINE" ] || die "no TOTAL line; cookie or network -- see logs/wild_source_${TAG}_fetch.log"
  LISTED="$(echo "$TOTAL_LINE" | grep -oP 'listed=\d+' | grep -oP '\d+' || echo 0)"
  # listed=0 and "nothing new" print the same way downstream, and only one of
  # them means the corpus is current, so this is fatal rather than a warning.
  [ "${LISTED:-0}" -gt 0 ] || die "listed=0 -- the douyin cookie has almost certainly expired; re-export it into $LODGE/www.douyin.com_cookies.txt"
  say "$TOTAL_LINE"
  mark_done fetch
fi

# ---------- STAGE 2: diff against THIS repo's ledger ----------
if stage_done diff; then
  say "=== STAGE 2 diff: skip (have $PENDING) ==="
else
  say "=== STAGE 2 diff: OSS listing vs this repo's manifests ==="
  "$OSSUTIL" ls "$OSS_PREFIX/" -c "$OSSCONF" \
    | grep -oP 'tiktok/[^/]+/\d+\.mp4$' | sed 's|tiktok/||; s|\.mp4$||' > "$STATE/oss_ids.txt" \
    || die "OSS listing failed"
  python3 - "$STATE/oss_ids.txt" "$PENDING" "$MANIFEST_ROOT" <<'PY' || exit 1
import json, pathlib, re, sys

oss_file, out_file, manifest_root = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
known = set()
# Every upload this repo has a manifest row for, in any generation of the
# manifest.  Reading the manifests rather than scanning directories is what
# makes this ledger ours: a directory that Lodge happens to hold says nothing
# about whether AtomicDance ever ingested it.
for path in manifest_root.rglob("*.jsonl"):
    for line in path.open(encoding="utf-8"):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        for field in ("source_recording_key", "recording_id", "legacy_clip_id"):
            value = row.get(field)
            if isinstance(value, str):
                known.update(re.findall(r"\d{15,19}", value))
lines = [l.strip() for l in open(oss_file, encoding="utf-8") if l.strip()]
seen, pending = set(), []
for line in lines:
    video = line.split("/", 1)[-1]
    if video not in known and video not in seen:
        seen.add(video)
        pending.append(line)
pathlib.Path(out_file).write_text("\n".join(pending) + ("\n" if pending else ""),
                                  encoding="utf-8")
print("[diff] oss={} known_here={} new={}".format(len(lines), len(known), len(pending)))
PY
  mark_done diff
fi

N_PEND="$(grep -c . "$PENDING" 2>/dev/null || echo 0)"
say "new videos not yet in this repo: $N_PEND"

# ---------- STAGE 3: download ----------
if [ "$N_PEND" -eq 0 ]; then
  say "=== STAGE 3 download: nothing new ==="
elif stage_done download; then
  say "=== STAGE 3 download: skip ==="
else
  say "=== STAGE 3 download: $N_PEND mp4 -> $DL_DIR ==="
  mkdir -p "$DL_DIR"
  fail=0
  while read -r item; do
    [ -n "$item" ] || continue
    video="${item##*/}"
    [ -s "$DL_DIR/$video.mp4" ] && continue
    "$OSSUTIL" cp "$OSS_PREFIX/$item.mp4" "$DL_DIR/$video.mp4" -c "$OSSCONF" >/dev/null 2>&1 \
      || { say "download failed: $item"; fail=$((fail+1)); }
  done < "$PENDING"
  say "downloaded $(ls "$DL_DIR" 2>/dev/null | wc -l) files, $fail failures"
  [ "$fail" -eq 0 ] && mark_done download
fi

say "=== STAGE 4b summary ==="
say "new video downloaded to : $DL_DIR"
say "next step is NOT another fetch: run GVHMR over the pending backlog printed"
say "above (tools/run_gvhmr_wild_shard.sh), then rebuild the performance bundle."
