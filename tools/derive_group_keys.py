#!/usr/bin/env python3
"""Restrict the uploader-account map to the uploads one bundle actually holds.

``runs/<tag>_group_keys.json`` is ``{upload id: choreographer account}`` -- the
pre-split key M3b and M3c read (``recluster_atomics_ingroup.genre_of``).  It is
an *observed* field, read off the ingest path metadata, not something any stage
infers; see plan §7.10.  So a new corpus generation does not get a new map, it
gets the same field looked up for whatever uploads it kept.

Why a tool and not a one-liner: the interesting case is an upload the source
map has never seen.  ``load_group_keys`` drops falsy values and ``genre_of``
returns ``None`` on a miss, so an uncovered upload silently lands in a nameless
group -- the pre-split would still run and still report cells.  That is the
shape CLAUDE.md §2 warns about, a gate that cannot fail.  Here the miss is the
refusal: an upload with no recorded account is a real gap in the ingest
metadata and has to be answered there, not defaulted to here.

    python3 tools/derive_group_keys.py \\
        --bundle data/wild3d/wild_v5_song_performance \\
        --source runs/wild_v4_group_keys.json \\
        --output runs/wild_v5_group_keys.json
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools import asset_io  # noqa: E402


def upload_of(recording_id: str) -> str:
    """The upload a recording belongs to.

    Ids appear as ``<generation>:<upload>:<clip>`` in the bundles and as
    ``<upload>__clipNNN`` in the cache directories.  The account is a property
    of the upload in both shapes, and the generation prefix is exactly what
    must not reach the lookup -- the map is keyed by bare upload id.
    """
    stem = recording_id.rsplit("/", 1)[-1]
    if ":" in stem:
        parts = stem.split(":")
        return parts[-2] if len(parts) >= 3 else parts[0]
    return stem.split("__clip")[0]


def derive(bundle: str, source: pathlib.Path) -> dict:
    mapping = {str(k): str(v) for k, v in
               json.loads(source.read_text(encoding="utf-8")).items() if v}
    uploads = []
    for row in asset_io.read_jsonl("{}/sources.jsonl".format(bundle.rstrip("/"))):
        uploads.append(upload_of(str(row["recording_id"])))
    distinct = sorted(set(uploads))
    missing = [name for name in distinct if name not in mapping]
    if missing:
        raise SystemExit(
            "{} of {} uploads of {} carry no account in {}: {} ... -- the "
            "ingest metadata is the place to answer this, not a default here"
            .format(len(missing), len(distinct), bundle, source, missing[:5]))
    kept = {name: mapping[name] for name in distinct}
    counts = collections.Counter(kept.values())
    report = {
        "bundle": bundle,
        "source": str(source),
        "clips": len(uploads),
        "uploads": len(distinct),
        "accounts": len(counts),
        "largest_account": counts.most_common(1)[0][0],
        "largest_account_uploads": counts.most_common(1)[0][1],
        "source_uploads_not_in_this_bundle": len(mapping) - len(kept),
    }
    return {"mapping": kept, "report": report}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", required=True,
                        help="repo-relative bundle prefix holding sources.jsonl")
    parser.add_argument("--source", type=pathlib.Path, required=True,
                        help="the account map to restrict")
    parser.add_argument("--output", required=True,
                        help="repo-relative key for the restricted map")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = derive(args.bundle, args.source)
    print(json.dumps(result["report"], ensure_ascii=False, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    asset_io.write_json(args.output, result["mapping"], indent=None)
    print("wrote {} ({} uploads)".format(args.output, len(result["mapping"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
