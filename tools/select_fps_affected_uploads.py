#!/usr/bin/env python3
"""Which uploads the fps defect actually damaged, measured per clip.

``cut_clip`` selected video by frame number and audio by seconds computed as
``frame/30``.  On an upload that is not 30 fps those are different spans, and
the clip carries two errors of very different size:

* **a constant audio offset.**  The clip's first source frame ``start`` really
  happens at ``start/rate`` seconds, and the old cut took audio from
  ``start/30``.  The gap is ``|start/30 - start/rate|`` and it grows with how
  deep into the upload the clip begins -- which is why the rate alone does not
  say how bad a clip is.
* **a tempo error.**  The picture was restamped at 30 fps, so it plays at
  ``30/rate`` speed against its own music.  A 60 fps upload plays at half.

The released corpus was re-cut on the second quantity with an implicit
threshold of one frame per second (``|rate - 30| > 1``), which selected 1,054
uploads and 1,418 clips.  That rule was never written down anywhere -- only its
output, an ``uploads.txt`` -- so this tool exists to make the scope of a re-cut
a thing with a criterion instead of a thing with a file.

**It never excludes silently.**  Every run prints the offset distribution of
the clips it did *not* select and how many more it would take at two tighter
thresholds, because "we chose not to fix these" and "we did not notice these"
look identical in a list of what was fixed.  Measured 2026-08-19 over the 872
released clips from 29.9-30.1 fps uploads: median offset 13 ms (0.40 frames at
30 fps), p90 37 ms, max 158 ms; 46 clips above 50 ms, 4 above 125 ms.

Selection is per upload because re-cutting is per upload: the ingest re-splits
the whole file, so one damaged clip brings its siblings with it.

Usage::

    select_fps_affected_uploads.py --output selection.json --upload-list uploads.txt
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import pathlib
import statistics
import sys
from typing import Dict, List, Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

GRID_FPS = 30.0
DEFAULT_SOURCES = "data/wild3d/wild_v4_raw_bundle/sources.jsonl"
DEFAULT_INGEST = "data/wild_ingest_v1"
DEFAULT_VIDEOS = ["data/wild_videos_20260811"]
# The released decision, restated as the quantity it was really about: a rate
# more than 1 fps off the 30 fps grid is a tempo error above 1/30.
RELEASED_TEMPO_THRESHOLD = 1.0 / GRID_FPS


def measure_rates(uploads: Dict[str, pathlib.Path], workers: int) -> Dict[str, Optional[float]]:
    from tools.ingest_wild_uploads import probe_fps

    def one(item):
        stem, path = item
        rate = probe_fps(path)
        return stem, (round(rate, 6) if rate else None)

    rates: Dict[str, Optional[float]] = {}
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for index, (stem, rate) in enumerate(pool.map(one, uploads.items()), start=1):
            rates[stem] = rate
            if index % 1000 == 0 or index == len(uploads):
                print("[{}/{}] rates measured".format(index, len(uploads)), flush=True)
    return rates


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", type=pathlib.Path, default=pathlib.Path(DEFAULT_SOURCES),
                        help="bundle sources.jsonl; its legacy_source_name is "
                             "the released clip set, which is the population "
                             "that matters -- the ingest tree holds more")
    parser.add_argument("--ingest", type=pathlib.Path, default=pathlib.Path(DEFAULT_INGEST))
    parser.add_argument("--videos", type=pathlib.Path, nargs="+",
                        default=[pathlib.Path(v) for v in DEFAULT_VIDEOS])
    parser.add_argument("--max-tempo-error", type=float, default=RELEASED_TEMPO_THRESHOLD,
                        help="select an upload whose picture plays this far off "
                             "real time; default {:.4f} reproduces the released "
                             "scope (|rate - 30| > 1)".format(RELEASED_TEMPO_THRESHOLD))
    parser.add_argument("--max-offset-seconds", type=float, default=None,
                        help="also select an upload any of whose clips carries "
                             "an audio offset this large.  Off by default so "
                             "the default run reproduces the released scope; "
                             "what it would add is printed either way")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--upload-list", type=pathlib.Path, default=None)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)

    clips = [json.loads(line)["legacy_source_name"] for line in
             args.sources.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_upload: Dict[str, List[str]] = {}
    for name in clips:
        by_upload.setdefault(name.split("__clip")[0], []).append(name)
    print("{} released clips from {} uploads".format(len(clips), len(by_upload)))

    located: Dict[str, pathlib.Path] = {}
    for root in args.videos:
        if not root.is_dir():
            continue
        for path in root.iterdir():
            if path.stem in by_upload:
                located.setdefault(path.stem, path)
    missing = sorted(set(by_upload) - set(located))
    if missing:
        # Not skipped: an upload whose file is gone cannot be re-cut, and a
        # selection that quietly omits it reports a scope it cannot deliver.
        print("WARNING: {} upload(s) behind released clips have no file; "
              "they cannot be judged or re-cut".format(len(missing)))

    rates = measure_rates(located, args.workers)

    rows, unlocatable = [], []
    for upload, names in sorted(by_upload.items()):
        rate = rates.get(upload)
        if not rate:
            unlocatable.extend(names)
            continue
        tempo = abs(rate - GRID_FPS) / GRID_FPS
        for name in sorted(names):
            meta_path = args.ingest / name / "meta.json"
            start = None
            if meta_path.is_file():
                span = json.loads(meta_path.read_text(encoding="utf-8")).get("source_frame_span")
                if span:
                    start = int(span[0])
            offset = abs(start / GRID_FPS - start / rate) if start is not None else None
            rows.append({"clip": name, "upload": upload, "source_fps": rate,
                         "tempo_error": round(tempo, 6),
                         "start_frame": start,
                         "audio_offset_seconds": round(offset, 6) if offset is not None else None})

    def selected(row):
        if row["tempo_error"] > args.max_tempo_error:
            return True
        if args.max_offset_seconds is not None and row["audio_offset_seconds"] is not None:
            return row["audio_offset_seconds"] > args.max_offset_seconds
        return False

    chosen_uploads = sorted({r["upload"] for r in rows if selected(r)})
    chosen_clips = sorted(r["clip"] for r in rows if r["upload"] in set(chosen_uploads))
    rest = [r for r in rows if r["upload"] not in set(chosen_uploads)]

    offsets = sorted(r["audio_offset_seconds"] for r in rest
                     if r["audio_offset_seconds"] is not None)
    tail = {}
    if offsets:
        def share(limit):
            over = sum(1 for o in offsets if o > limit)
            return {"clips": over, "percent": round(100.0 * over / len(offsets), 2)}
        tail = {
            "clips_not_selected": len(rest),
            "offset_median": round(statistics.median(offsets), 6),
            "offset_p90": round(offsets[int(0.9 * (len(offsets) - 1))], 6),
            "offset_max": round(offsets[-1], 6),
            "above_50ms": share(0.05),
            "above_125ms": share(0.125),
        }

    result = {
        "generated_by": "tools/select_fps_affected_uploads.py",
        "criterion": {"max_tempo_error": args.max_tempo_error,
                      "max_offset_seconds": args.max_offset_seconds},
        "released_clips": len(clips),
        "selected_uploads": chosen_uploads,
        "selected_clips": chosen_clips,
        "not_selected": tail,
        "uploads_without_a_file": missing,
        "clips_without_a_measurable_upload": sorted(unlocatable),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = args.output.with_suffix(args.output.suffix + ".tmp")
    staging.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    staging.replace(args.output)
    if args.upload_list:
        args.upload_list.write_text("".join(u + "\n" for u in chosen_uploads), encoding="utf-8")
        print("wrote {} ({} uploads)".format(args.upload_list, len(chosen_uploads)))

    print("\nselected: {} uploads -> {} released clips".format(
        len(chosen_uploads), len(chosen_clips)))
    if tail:
        print("NOT selected: {} clips.  Their audio offset: median {:.0f} ms, "
              "p90 {:.0f} ms, max {:.0f} ms".format(
                  tail["clips_not_selected"], tail["offset_median"] * 1000,
                  tail["offset_p90"] * 1000, tail["offset_max"] * 1000))
        print("  tightening --max-offset-seconds would add: "
              "{} clips at 50 ms, {} clips at 125 ms".format(
                  tail["above_50ms"]["clips"], tail["above_125ms"]["clips"]))
    print("wrote {}".format(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
