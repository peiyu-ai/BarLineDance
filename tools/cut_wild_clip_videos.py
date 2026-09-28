#!/usr/bin/env python3
"""Re-cut the clip videos a v4 recording was ingested from, for re-extraction.

This is the **fallback** path, and it should stay one: the features are in the
object store and pulling them takes 53 seconds, so reach for
``tools/oss_assets.py pull --cache data/wild_visual_s3d`` first.  What this tool
is for is the case where the store cannot be reached or -- as on 2026-08-19 --
where the stored object under a clip's name turns out to belong to an older cut
of that clip (28.5% of the corpus; see ``docs/VOCABULARY_DIAGNOSIS.md``
section 2.F).  Nothing about those features is unrecoverable: every ingested
clip kept ``meta.json`` naming the local upload and the exact frame span it was
cut from, and all 10,793 uploads are on local CPFS.

The one contract that matters is the frame clock.  ``segment_visual_atomics``
fuses a visual self-similarity block with a motion one **frame by frame**; a
clip that came out one frame short would put the two modalities out of phase
for its whole length, and nothing downstream could see it.  So:

* the cut reuses ``ingest_wild_uploads.cut_clip`` -- the one that seeks on the
  decoded stream rather than on keyframes, and reads the written frame count
  back instead of assuming it;
* a clip whose written frame count differs from ``meta.json``'s ``num_frames``
  is **refused and deleted**, not kept with a warning;
* ``--limit`` samples at an even stride over the sorted recording list, never a
  prefix.  A prefix here is the first few accounts alphabetically, which is a
  handful of dancers rather than a sample of the corpus.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.ingest_wild_uploads import cut_clip                    # noqa: E402


def recordings_from_labels(labels_jsonl: pathlib.Path):
    out = []
    with open(labels_jsonl, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            out.append((row["recording_id"], row.get("split")))
    return sorted(out)


def clip_stem(recording_id: str) -> str:
    """``wild_v4:<upload>:clipNNN`` -> ``<upload>__clipNNN`` (the ingest dir)."""
    body = recording_id.split(":", 1)[1] if ":" in recording_id else recording_id
    return body.replace(":", "__")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=pathlib.Path,
                        default=REPO / "data/wild3d/wild_v4_acct/ingroup_llm/labels.jsonl")
    parser.add_argument("--ingest-root", type=pathlib.Path,
                        default=pathlib.Path("/cache/atomicdance-assets/data/wild_ingest_v1"))
    parser.add_argument("--upload-root", type=pathlib.Path,
                        default=pathlib.Path("/cache/atomicdance-assets/data"))
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--limit", type=int, default=800)
    parser.add_argument("--split", default=None, help="restrict to one split")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    rows = recordings_from_labels(args.labels)
    if args.split:
        rows = [r for r in rows if r[1] == args.split]
    if args.limit and args.limit < len(rows):
        stride = len(rows) / args.limit
        rows = [rows[int(i * stride)] for i in range(args.limit)]
    rows = rows[args.shard::args.num_shards]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    made = skipped = refused = 0
    for recording_id, _ in rows:
        stem = clip_stem(recording_id)
        destination = args.output_dir / (stem + ".mp4")
        if destination.exists():
            skipped += 1
            continue
        meta_path = args.ingest_root / stem / "meta.json"
        if not meta_path.exists():
            print("no meta.json for {}".format(stem), flush=True)
            refused += 1
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        upload = args.upload_root / pathlib.Path(meta["source"]).relative_to("data")
        if not upload.exists():
            print("upload missing for {}: {}".format(stem, upload), flush=True)
            refused += 1
            continue
        start, end = meta["source_frame_span"]
        written = cut_clip(upload, int(start), int(end), destination)
        if written is None or written != int(meta["num_frames"]):
            print("REFUSED {}: wrote {} frames, meta says {}".format(
                stem, written, meta["num_frames"]), flush=True)
            destination.unlink(missing_ok=True)
            refused += 1
            continue
        made += 1
        if (made + skipped) % 50 == 0:
            print("  {} made / {} skipped / {} refused".format(made, skipped, refused), flush=True)
    print("cut {} clips -> {} ({} already there, {} refused)".format(
        made, args.output_dir, skipped, refused))
    return 1 if refused and not made else 0


if __name__ == "__main__":
    sys.exit(main())
