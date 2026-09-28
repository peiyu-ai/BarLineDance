#!/usr/bin/env python3
"""Which uploads have a frame rate the ingest's frame-indexed cut cannot use.

This is a different defect from the one ``select_fps_affected_uploads.py``
selects for, and the two must not be merged.  That one is about uploads whose
rate is not 30: a *constant* rate the old cutter assumed away.  This one is
about uploads whose container reports two rates that disagree --
``avg_frame_rate`` (frames over duration, what the file holds) against
``r_frame_rate`` (what the timebase can express).  When they disagree, a frame
number does not name an instant, and ``cut_clip`` selects the picture by frame
number while selecting the sound by seconds.

Measured 2026-08-24 on ``7438547996335295781.mp4`` (avg 31.62, r 60.00), by
locating the clip's first and last frame inside the upload by pixel match and
its audio by onset cross-correlation:

    picture taken from 27.17-54.37 s   (span 27.20 s)
    sound   taken from 13.63 s, 13.65 s long
    ---> picture span / sound length = 1.992, start offset 13.54 s

Two constant-rate uploads measured the same way that day read 0.994 and 0.996
with a 0.00 s offset, so the reading separates rather than being a story about
one clip.  The clip's own container is internally consistent -- 409 frames,
13.63 s of video, 13.65 s of audio -- which is why nothing downstream can see
it: everything that could compare picture against sound was cut from the same
wrong span.

**It never excludes silently.**  Every run prints how many uploads were checked,
how many were unreadable, and the rate-disagreement distribution of the ones it
did *not* select, because "we found none" and "we could not look" are the same
line otherwise.

Usage::

    select_vfr_uploads.py --videos-dir data/wild_videos_20260811 \\
        --clips runs/wild_v4_acct_gt_eval/audio --output vfr.json
    select_vfr_uploads.py --uploads-from data/wild3d/wild_v4_raw_bundle/sources.jsonl \\
        --output vfr.json --workers 16
    select_vfr_uploads.py --ingest-root /cache/.../wild_ingest_v1 --output vfr.json

The third form is the one to use over a whole corpus, and it is not a
convenience.  ``--videos-dir`` globs a directory and ``--clips`` globs another;
both keep returning clips no run produces any more, which on this corpus is a
standing population -- a re-cut that merges two clips into one leaves the second
name on disk forever, because these credentials cannot delete it.  The ingest
manifest is each run's own statement of what it produced, so ``--ingest-root``
asks that instead, and the upload paths come from the same rows rather than
from a guess about which directory the uploads were read out of.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as futures
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.ingest_wild_uploads import VFR_TOLERANCE, is_variable_rate, probe_rates


def uploads_from_manifest(path: pathlib.Path, videos_dir: pathlib.Path) -> list:
    """Upload ids named by a sources/sequences manifest, in first-seen order."""
    seen = []
    known = set()
    for line in path.open(encoding="utf-8"):
        try:
            record = json.loads(line)
        except ValueError:
            continue
        key = record.get("recording_id") or record.get("sequence_id") or ""
        parts = str(key).split(":")
        if len(parts) != 3:
            continue
        if parts[1] not in known:
            known.add(parts[1])
            seen.append(parts[1])
    return [videos_dir / (upload + ".mp4") for upload in seen]


def uploads_from_ingest(ingest_root: pathlib.Path) -> tuple:
    """``([upload paths], {upload: [clip names]})`` from the ingest manifests.

    Both halves come from the manifest row rather than from the filesystem.
    The path matters because this corpus was ingested out of three different
    directories (the released set and two re-cut scratch trees), so there is no
    single ``--videos-dir`` that holds all of them; the row records which file
    was actually read.  The clip list matters because a clip directory outlives
    the run that produced it -- see the module docstring.
    """
    from tools.ingest_wild_uploads import manifest_rows

    uploads, clips = [], {}
    for upload, row in sorted(manifest_rows(ingest_root).items()):
        source = row.get("source")
        if not source:
            continue
        uploads.append(pathlib.Path(source))
        produced = [c.get("clip") for c in row.get("clips", [])
                    if c.get("status") in ("ok", "exists")]
        if produced:
            clips[upload] = produced
    return uploads, clips


def clips_by_upload(clips_dir: pathlib.Path) -> dict:
    """upload id -> the clip keys under it, from a directory of per-clip files.

    Keys are read from file *stems*, so this works for the music arrays, the
    motion pickles or anything else named ``wild_v4:<upload>:clipNNN``.
    """
    mapping = collections.defaultdict(list)
    for path in sorted(clips_dir.iterdir()):
        parts = path.stem.split(":")
        if len(parts) == 3:
            mapping[parts[1]].append(path.stem)
    return dict(mapping)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos-dir", type=pathlib.Path,
                        default=pathlib.Path("data/wild_videos_20260811"),
                        help="where the upload mp4s live")
    parser.add_argument("--uploads-from", type=pathlib.Path, default=None,
                        help="a sources/sequences .jsonl naming the uploads to "
                             "check; without it every mp4 in --videos-dir")
    parser.add_argument("--ingest-root", type=pathlib.Path, default=None,
                        help="an ingest root holding ingest_shard*.jsonl; the "
                             "uploads and their currently-produced clips both "
                             "come from those rows.  Overrides --videos-dir / "
                             "--uploads-from / --clips")
    parser.add_argument("--clips", type=pathlib.Path, default=None,
                        help="a directory of per-clip files named "
                             "wild_v4:<upload>:clipNNN.*, used to report which "
                             "already-cut clips sit on a selected upload")
    parser.add_argument("--tolerance", type=float, default=VFR_TOLERANCE,
                        help="relative disagreement that counts as variable")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    clips = {}
    if args.ingest_root:
        uploads, clips = uploads_from_ingest(args.ingest_root)
    elif args.uploads_from:
        uploads = uploads_from_manifest(args.uploads_from, args.videos_dir)
    else:
        uploads = sorted(args.videos_dir.glob("*.mp4"))
    if not uploads:
        raise SystemExit("no uploads to check")
    if args.clips and not args.ingest_root:
        clips = clips_by_upload(args.clips)

    def measure(path: pathlib.Path) -> dict:
        if not path.is_file():
            return {"upload": path.stem, "status": "missing"}
        avg, rate = probe_rates(path)
        if not avg or not rate:
            return {"upload": path.stem, "status": "unreadable"}
        return {"upload": path.stem, "status": "ok",
                "avg_frame_rate": round(avg, 4), "r_frame_rate": round(rate, 4),
                "disagreement": round(abs(avg - rate) / max(avg, rate), 5),
                "picture_over_sound": round(rate / avg, 4)}

    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(measure, uploads))

    readable = [r for r in rows if r["status"] == "ok"]
    selected = [r for r in readable
                if is_variable_rate(r["avg_frame_rate"], r["r_frame_rate"], args.tolerance)]
    rejected = [r for r in readable if r not in selected]
    for row in selected:
        row["clips"] = clips.get(row["upload"], [])

    report = {
        "tolerance": args.tolerance,
        "checked": len(rows),
        "readable": len(readable),
        "unreadable": sum(1 for r in rows if r["status"] == "unreadable"),
        "missing": sum(1 for r in rows if r["status"] == "missing"),
        "selected_uploads": len(selected),
        "selected_clips": sum(len(r.get("clips", [])) for r in selected),
        "clip_index_supplied": bool(clips),
        "clip_index_from": ("ingest manifest" if args.ingest_root else
                            "clip directory glob" if args.clips else None),
        "uploads": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("checked {} upload(s): {} readable, {} unreadable, {} missing".format(
        report["checked"], report["readable"], report["unreadable"], report["missing"]))
    print("selected {} upload(s) at tolerance {:.3f}{}".format(
        len(selected), args.tolerance,
        ", carrying {} already-cut clip(s)".format(report["selected_clips"])
        if clips else " (no --clips given, so no clip count)"))
    ratios = collections.Counter(round(r["picture_over_sound"], 2) for r in selected)
    if ratios:
        print("  picture-over-sound ratios: " + ", ".join(
            "{}x x{}".format(k, v) for k, v in ratios.most_common(8)))
    # What was NOT selected, so a threshold cannot quietly do the deciding.
    if rejected:
        near = sorted((r["disagreement"] for r in rejected), reverse=True)[:5]
        tighter = {t: sum(1 for r in rejected if r["disagreement"] > t)
                   for t in (args.tolerance / 2, args.tolerance / 4)}
        print("  not selected: {} upload(s); largest disagreements {}".format(
            len(rejected), ", ".join("{:.5f}".format(v) for v in near) or "none"))
        print("  a tolerance of {:.4f} would add {}, {:.4f} would add {}".format(
            args.tolerance / 2, tighter[args.tolerance / 2],
            args.tolerance / 4, tighter[args.tolerance / 4]))
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
