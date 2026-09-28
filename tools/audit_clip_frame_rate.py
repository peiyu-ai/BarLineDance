#!/usr/bin/env python3
"""Does a clip's motion play at real time, or at the wrong speed forever?

**What this asks, and why it is not the same question as retrieval stretch.**
``tools/measure_retrieval_stretch.py`` measures a stretch the *generator*
applies at inference: a prototype resampled to fill a plan slot.  That stretch
is per-slot and reversible -- change the rule and it goes away.  This tool asks
whether the *corpus itself* is at the speed it claims.  A clip is a resampling
of source frames ``[start, end)`` onto a 30 fps grid, so it covers
``(end-start)/source_fps`` seconds of real dancing and holds ``num_frames``
frames played back at 30 fps.  The ratio

    playback_ratio = (num_frames / 30) / ((end - start) / source_fps)

is 1.0 when the picture runs at real time.  Anything else is baked into the
bytes: GVHMR reconstructs whatever speed the picture shows, the 151-D motion
inherits it, and every atomic cut from that clip plays at that speed for the
rest of the corpus's life.  That is why this is reported per clip and not as a
corpus average -- one clip at 0.833 is one prototype family that is 20% fast.

**Provenance: this criterion was invented on 2026-09-04 and is not from the
paper.**  It is derived from ``tools/ingest_wild_uploads.py:cut_clip``, whose
docstring states the contract this checks ("Cut source frames [start, end) out
as a real-time span at 30 fps"), and from the defect that docstring records:
before 2026-08-19 the cut renumbered frames instead of resampling them, so a 60
fps upload gave ``num_frames = end - start`` -- a ``playback_ratio`` of 2.0 --
and 1,418 of 13,783 released clips carried it while ``meta.json`` said
``fps: 30.0``, which reads exactly like a measurement.

**source_fps is measured here, never taken from the meta.**  Taking it from the
meta would make the check circular in exactly the way that hid the original
defect: a pre-2026-08-19 meta has no ``source_fps`` field at all, and its
``fps: 30.0`` is the constant, not a reading.  So the upload named by
``meta["source"]`` is probed with ffprobe.  ``--trust-meta-fps`` exists only so
the test suite can exhibit the circular version failing to fire.

**Both of the container's rate claims are read.**  ``r_frame_rate`` is what the
cut used; ``avg_frame_rate`` disagreeing with it by more than ``--vfr-tolerance``
is the variable-frame-rate defect of 2026-08-25 (picture and sound cut from
different spans), reported separately as ``vfr`` rather than folded into the
ratio, because it is a different failure with a different repair.

**Positive control** (``tests/test_audit_clip_frame_rate.py``): a real upload is
re-encoded to 60 fps by frame duplication and cut with the pre-fix rule, and the
tool must report ``playback_ratio`` 2.0 and status ``speed_defect``; the same
span cut with the current rule from the same 60 fps file must report 1.0 and
``ok``.  A tool that only ever answers "ok" on a corpus that is fine has not
been shown to be able to answer anything else.

Usage::

    audit_clip_frame_rate.py --clips names.txt --ingest data/wild_ingest_v1 \
        --output audit.json [--converted <3d root>] [--workers 16]

``--clips`` is a manifest of clip directory names, one per line.  It is required
and there is no ``--all``: this corpus carries orphan clips that these
credentials cannot delete, so enumerating it by globbing a directory or listing
an OSS prefix returns clips that must never be read again (CLAUDE.md 1.1).
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# The grid every clip is cut onto and every downstream tool assumes.  It is the
# claim being audited, not an input to the audit.
CLIP_FPS = 30.0
# A clip may come back a frame or two off what the resampler was asked for --
# cut_clip says so and checks the same way.  Expressed in frames, not in ratio,
# because the tolerance is a property of the resampler and not of clip length.
FRAME_TOLERANCE = 2.0
# Below this the ratio is indistinguishable from resampler rounding on a short
# clip; above it, the picture is at the wrong speed.  0.02 is 2%, which is
# larger than the 29.97-vs-30 NTSC gap (0.1%) and far smaller than the smallest
# real defect this can produce (25 fps read as 30 is 20%).
RATIO_TOLERANCE = 0.02


def _rate(text: str) -> float:
    """ffprobe rational -> float.  '0/0' divides by zero rather than raising."""
    try:
        numerator, _, denominator = text.strip().partition("/")
        return float(numerator) / float(denominator or 1)
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe_rates(path: pathlib.Path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=r_frame_rate,avg_frame_rate", "-of", "json", str(path)],
        capture_output=True, text=True)
    try:
        stream = json.loads(result.stdout)["streams"][0]
    except (ValueError, KeyError, IndexError):
        return 0.0, 0.0
    return _rate(stream.get("avg_frame_rate", "0/0")), _rate(stream.get("r_frame_rate", "0/0"))


def audit_clip(name: str, ingest: pathlib.Path, repo: pathlib.Path,
               converted: pathlib.Path = None, trust_meta_fps: bool = False,
               vfr_tolerance: float = 0.02) -> dict:
    row = {"clip": name}
    meta_path = ingest / name / "meta.json"
    if not meta_path.is_file():
        row["status"] = "meta_missing"
        return row
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except ValueError:
        row["status"] = "meta_unreadable"
        return row

    span = meta.get("source_frame_span")
    frames = meta.get("num_frames")
    if not isinstance(span, list) or len(span) != 2 or not frames:
        row["status"] = "meta_incomplete"
        return row
    start, end = int(span[0]), int(span[1])
    row.update({"num_frames": int(frames), "source_frame_span": [start, end],
                "meta_source_fps": meta.get("source_fps"),
                # Which ingest generation cut this clip.  The field names are
                # the marker, and they are read rather than inferred: no
                # ``source_fps`` means the cut predates 2026-08-19 and did not
                # measure the upload at all.
                "ingest_generation": ("vfr_aware" if "source_key" in meta
                                      else "fps_aware" if "source_fps" in meta
                                      else "pre_fps_fix")})

    if trust_meta_fps:
        measured_r = float(meta.get("source_fps") or CLIP_FPS)
        measured_avg = measured_r
        row["source_fps_from"] = "meta"
    else:
        upload = pathlib.Path(meta.get("source", ""))
        if not upload.is_absolute():
            upload = repo / upload
        if not upload.is_file():
            row["status"] = "upload_missing"
            row["upload"] = str(upload)
            return row
        measured_avg, measured_r = probe_rates(upload)
        row["upload"] = str(upload)
        row["source_fps_from"] = "ffprobe"
    if not measured_r:
        row["status"] = "unreadable_frame_rate"
        return row

    row["measured_r_frame_rate"] = round(measured_r, 4)
    row["measured_avg_frame_rate"] = round(measured_avg, 4)
    # The cut used r_frame_rate (``container_fps or average_fps``), so the ratio
    # is computed against the same rate the cut used; the disagreement between
    # the two is a separate finding below.
    expected = (end - start) * CLIP_FPS / measured_r
    row["expected_frames"] = round(expected, 2)
    row["playback_ratio"] = round(float(frames) / expected, 4) if expected else None
    row["frame_error"] = round(float(frames) - expected, 2)

    if meta.get("source_fps") is not None and abs(float(meta["source_fps"]) - measured_r) > 0.05:
        row["status"] = "recorded_fps_disagrees"
        return row
    if row["playback_ratio"] is None:
        row["status"] = "unreadable_frame_rate"
        return row
    if (abs(row["frame_error"]) > FRAME_TOLERANCE
            and abs(row["playback_ratio"] - 1.0) > RATIO_TOLERANCE):
        row["status"] = "speed_defect"
        return row

    if measured_avg and abs(measured_r - measured_avg) / measured_r > vfr_tolerance:
        # Not folded into playback_ratio on purpose: the picture is at the right
        # speed for the rate the cut used, and what is wrong is that the file
        # has two rates.  Merging them would report one number for two repairs.
        row["status"] = "vfr"
        return row

    if converted is not None:
        metadata = converted / name / "metadata.json"
        if not metadata.is_file():
            row["status"] = "converted_missing"
            return row
        try:
            data = json.loads(metadata.read_text(encoding="utf-8"))
        except ValueError:
            row["status"] = "converted_unreadable"
            return row
        row["converted_frames"] = data.get("frames_30fps")
        row["converted_video_frames"] = (data.get("extract_meta") or {}).get("video_frames")
        row["converted_fps"] = data.get("fps")
        # The 3D is the artifact the library is cut from, so it has to agree
        # with the clip it claims to come from.  A converter that subsampled --
        # convert_gvhmr_result keeps every second frame when it believes the
        # source is 60 fps -- shows up here as half the frames.
        if row["converted_video_frames"] != int(frames):
            row["status"] = "converted_frame_mismatch"
            return row
        if row["converted_fps"] != CLIP_FPS:
            row["status"] = "converted_fps_mismatch"
            return row

    row["status"] = "ok"
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", type=pathlib.Path, required=True,
                        help="manifest of clip directory names, one per line")
    parser.add_argument("--ingest", type=pathlib.Path, required=True)
    parser.add_argument("--converted", type=pathlib.Path, default=None,
                        help="converted 3D root; enables the 3D frame-count check")
    parser.add_argument("--repo", type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parents[1],
                        help="root that relative meta['source'] paths resolve against")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--vfr-tolerance", type=float, default=0.02)
    parser.add_argument("--trust-meta-fps", action="store_true",
                        help="circular variant kept only so the tests can show "
                             "it failing to fire; never use it to judge a corpus")
    args = parser.parse_args()

    names = [line.strip() for line in args.clips.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    rows = []
    with futures.ThreadPoolExecutor(max(1, args.workers)) as pool:
        for row in pool.map(lambda n: audit_clip(
                n, args.ingest, args.repo, args.converted,
                args.trust_meta_fps, args.vfr_tolerance), names):
            rows.append(row)

    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    ratios = sorted(r["playback_ratio"] for r in rows if r.get("playback_ratio"))
    report = {
        "generated_by": "tools/audit_clip_frame_rate.py",
        "criterion": "playback_ratio = (num_frames/30) / ((end-start)/measured_source_fps)",
        "criterion_provenance": "invented 2026-09-04 from cut_clip's stated contract; not from the paper",
        "source_fps_read_from": "meta" if args.trust_meta_fps else "ffprobe on meta['source']",
        "clips_audited": len(rows),
        "status_counts": counts,
        "playback_ratio_min": ratios[0] if ratios else None,
        "playback_ratio_max": ratios[-1] if ratios else None,
        "defects": [r for r in rows if r["status"] not in ("ok",)],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("clips_audited", "status_counts", "playback_ratio_min",
                       "playback_ratio_max")}, indent=2))
    return 0 if set(counts) <= {"ok"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
