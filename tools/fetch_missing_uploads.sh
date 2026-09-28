#!/usr/bin/env bash
#
# Bring down the raw uploads this repo has never held.
#
# `update_wild_source.sh` downloads what is *new to this repo*, which is the
# right question when the job is to extend the corpus.  Re-cutting it from
# scratch asks a different one: every upload OSS holds, including the ~2,800
# whose 16-second clips are already here but whose source video never was.
# Cutting on content needs the source, not the pieces somebody else cut.
#
# Downloads run in parallel because the serial loop in update_wild_source.sh
# spends its time in per-object round trips, not bandwidth.
#
#   LIST=<file of account/videoid lines> DEST=<dir> JOBS=16 bash tools/fetch_missing_uploads.sh
#
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$(pwd)"
LODGE="${LODGE:-$(cd "$REPO/../Lodge" && pwd)}"

LIST="${LIST:-logs/wild_source_20260811.state/complement_videos.txt}"
DEST="${DEST:-data/wild_videos_20260811}"
JOBS="${JOBS:-16}"
OSS_PREFIX="${OSS_ASSET_PREFIX:-oss://example-bucket/example/prefix/}data/tiktok"
OSSUTIL="${OSSUTIL:-/opt/data-infra/ossutil64}"
OSSCONF="${OSSCONF:-$LODGE/.secrets/ossutilconfig}"

[ -s "$LIST" ] || { echo "no list at $LIST" >&2; exit 1; }
[ -x "$OSSUTIL" ] || { echo "ossutil missing at $OSSUTIL" >&2; exit 1; }
[ -f "$OSSCONF" ] || { echo "oss config missing at $OSSCONF" >&2; exit 1; }
mkdir -p "$DEST"

export OSSUTIL OSSCONF OSS_PREFIX DEST
fetch_one() {
  item="$1"
  video="${item##*/}"
  dest="$DEST/$video.mp4"
  # Non-empty is the resume check; ossutil writes the whole object or nothing,
  # so a zero-byte file is a failed attempt and gets retried rather than kept.
  [ -s "$dest" ] && return 0
  "$OSSUTIL" cp "$OSS_PREFIX/$item.mp4" "$dest" -c "$OSSCONF" >/dev/null 2>&1 \
    || { echo "FAIL $item"; rm -f "$dest"; return 1; }
}
export -f fetch_one

total="$(wc -l < "$LIST")"
echo "fetching $total uploads into $DEST with $JOBS workers"
xargs -a "$LIST" -P "$JOBS" -I{} bash -c 'fetch_one "$@"' _ {} | tee /tmp/fetch_fail.$$
echo "have $(ls "$DEST"/*.mp4 2>/dev/null | wc -l) mp4 in $DEST"
