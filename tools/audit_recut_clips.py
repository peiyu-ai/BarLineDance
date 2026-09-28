#!/usr/bin/env python3
"""Spot-check a re-cut corpus: does each clip cover the span it claims?

The fps defect and its fix both live inside ``cut_clip``.  Video is selected by
source frame index and then restamped; audio is selected by time.  Before the
fix the restamp was ``setpts=N/30/TB`` and the audio span was ``[start/30,
end/30)``; after it, both are the real span ``[start/rate, end/rate)``.

**The obvious check is worthless here, and that is worth stating.**  "Does the
clip's video last as long as its audio" passes on a broken clip: before the fix
both came out ``(end-start)/30`` seconds long.  They agreed with each other and
disagreed with reality.  Two things do discriminate:

* **Duration.**  A correct clip lasts ``(end-start)/source_fps`` seconds -- the
  real time those source frames occupy.  A clip cut by the old code lasts
  ``(end-start)/30``, which on a 60 fps upload is twice as long.
* **Audio content.**  A correct clip's sound is the upload's sound from
  ``start/source_fps``.  The old code took it from ``start/30``, a different
  passage of the same song.  Cross-correlating the clip's audio against the
  upload's audio cut at the expected offset peaks at lag zero when they are the
  same passage and does not when they are not.

Both controls are run, because a check that only ever sees good input cannot be
distinguished from a check that always passes:

* a **control group** of clips from 30 fps uploads, which were never affected
  and must pass everything;
* a **negative control** per sampled clip -- the same correlation against the
  passage the *old* code would have taken.  On an upload that is not 30 fps
  that is a different span, so a working instrument must score it lower.  Where
  the two spans coincide (``start`` is 0, so both offsets are 0) the negative
  control is undefined and is reported as skipped rather than as a pass.

Usage::

    audit_recut_clips.py --sample 40 --output recut_audit.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

GRID_FPS = 30.0
INGEST = pathlib.Path("/cache/atomicdance-assets/data/wild_ingest_v1")
UPLOADS = pathlib.Path("/cache/atomicdance-assets/data/wild_videos_20260811")
SAMPLE_RATE = 22050


def run(argv) -> subprocess.CompletedProcess:
    return subprocess.run([str(a) for a in argv], capture_output=True, text=True)


def probe(path: pathlib.Path, stream: str, entries: str) -> str:
    result = run(["ffprobe", "-v", "error", "-select_streams", stream,
                  "-show_entries", entries, "-of", "default=nw=1:nk=1", path])
    return (result.stdout or "").strip().splitlines()[0] if result.stdout.strip() else ""


def media_seconds(path: pathlib.Path) -> Optional[float]:
    text = probe(path, "v:0", "format=duration")
    if not text:
        result = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                      "-of", "default=nw=1:nk=1", path])
        text = (result.stdout or "").strip()
    try:
        return float(text)
    except ValueError:
        return None


def load_audio(path: pathlib.Path, start: float = None, duration: float = None):
    """Mono float32 at SAMPLE_RATE, optionally a window, decoded through ffmpeg."""
    import numpy as np

    argv = ["ffmpeg", "-nostdin", "-v", "error"]
    if start is not None:
        argv += ["-ss", "{:.6f}".format(start)]
    argv += ["-i", str(path)]
    if duration is not None:
        argv += ["-t", "{:.6f}".format(duration)]
    argv += ["-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]
    result = subprocess.run(argv, capture_output=True)
    if result.returncode != 0 or not result.stdout:
        return None
    return np.frombuffer(result.stdout, dtype="<f4")


def agreement(a, b) -> Optional[float]:
    """Peak normalised cross-correlation of two mono signals, coarse and cheap.

    Envelope rather than waveform: the clip has been through AAC twice and the
    upload once, so sample-exact agreement is not on offer, but the loudness
    contour of the same passage survives both.  A different passage of the same
    song does not reproduce it.
    """
    import numpy as np

    if a is None or b is None:
        return None
    length = min(len(a), len(b))
    if length < SAMPLE_RATE:            # under a second: nothing to say
        return None
    a, b = a[:length].astype(np.float32), b[:length].astype(np.float32)
    hop = SAMPLE_RATE // 50             # 20 ms envelope
    frames = length // hop
    if frames < 20:
        return None
    ea = np.abs(a[: frames * hop]).reshape(frames, hop).mean(axis=1)
    eb = np.abs(b[: frames * hop]).reshape(frames, hop).mean(axis=1)
    ea = ea - ea.mean()
    eb = eb - eb.mean()
    denominator = float(np.linalg.norm(ea) * np.linalg.norm(eb))
    if denominator < 1e-9:
        return None
    return float(np.dot(ea, eb) / denominator)


def fetch_video(stem: str) -> Optional[pathlib.Path]:
    """One clip's video out of the store, to a temp file the caller deletes."""
    from tools import asset_io

    relative = "data/wild_ingest_v1/{}/clip.mp4".format(stem)
    handle, name = tempfile.mkstemp(suffix=".mp4")
    target = pathlib.Path(name)
    try:
        import os

        os.close(handle)
        target.write_bytes(asset_io.read_bytes(relative))
        return target
    except Exception:                                         # noqa: BLE001
        target.unlink(missing_ok=True)
        return None


def judge(stem: str, tolerance: float, want_audio: bool) -> Dict[str, object]:
    row: Dict[str, object] = {"clip": stem}
    directory = INGEST / stem
    meta_path = directory / "meta.json"
    if not meta_path.is_file():
        row["status"] = "no_meta"
        return row
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    span = meta.get("source_frame_span")
    rate = meta.get("source_fps")
    row["source_fps"] = rate
    row["source_frame_span"] = span
    row["num_frames"] = meta.get("num_frames")
    if not span:
        row["status"] = "no_span"
        return row
    video = directory / "clip.mp4"
    fetched = None
    if not video.is_file():
        # The 30 fps corpus was never re-cut, so its videos were evicted to the
        # store.  Those clips are the control group, and a control group that
        # cannot be read is not a control group -- so they are fetched rather
        # than dropped.
        fetched = fetch_video(stem)
        if fetched is None:
            row["status"] = "no_video"
            return row
        video = fetched

    upload = UPLOADS / "{}.mp4".format(stem.split("__clip")[0])
    if rate is None:
        # A clip from before the fps-aware ingest.  Its own record does not say
        # what rate it was cut at, so the expected duration has to come from the
        # upload -- and that is the point: it will not match.
        from tools.ingest_wild_uploads import probe_fps
        rate = probe_fps(upload) if upload.is_file() else None
        row["source_fps_measured_from_upload"] = rate
        row["fps_aware"] = False
    else:
        row["fps_aware"] = True
    if not rate:
        row["status"] = "no_rate"
        return row

    start, end = float(span[0]), float(span[1])
    expected = (end - start) / rate
    actual = media_seconds(video)
    row["expected_seconds"] = round(expected, 4)
    row["actual_seconds"] = round(actual, 4) if actual is not None else None
    if actual is None:
        row["status"] = "unreadable_video"
        return row
    row["duration_error"] = round(abs(actual - expected), 4)
    row["duration_ok"] = abs(actual - expected) <= tolerance
    # What the old code would have produced, so the reading has a scale.
    row["old_code_seconds"] = round((end - start) / GRID_FPS, 4)

    row["video_from_store"] = fetched is not None
    if want_audio and upload.is_file():
        clip_audio = load_audio(video)
        correct = load_audio(upload, start=start / rate, duration=expected)
        row["audio_match_correct_span"] = agreement(clip_audio, correct)
        wrong_offset = start / GRID_FPS
        if abs(wrong_offset - start / rate) < 0.05:
            # The two spans coincide (start is 0, or the rate is ~30).  A
            # negative control that is the same signal proves nothing, and
            # calling it a pass would be the instrument flattering itself.
            row["audio_match_old_span"] = None
            row["negative_control"] = "coincident_spans"
        else:
            wrong = load_audio(upload, start=wrong_offset, duration=expected)
            row["audio_match_old_span"] = agreement(clip_audio, wrong)
            row["negative_control"] = "distinct_spans"
    row["status"] = "ok"
    if fetched is not None:
        fetched.unlink(missing_ok=True)
    return row


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=int, default=40,
                        help="re-cut clips to check")
    parser.add_argument("--control", type=int, default=20,
                        help="clips from 30 fps uploads, which were never "
                             "affected and must pass everything")
    parser.add_argument("--tolerance", type=float, default=0.12,
                        help="seconds; the resampler lands within a frame or "
                             "two of the requested span and reads the count "
                             "back, so this is slack, not a threshold under test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--rates", type=pathlib.Path,
                        default=pathlib.Path("/cache/atomicdance-assets/scratch/"
                                             "c1/refix/upload_rates.json"),
                        help="measured upload rates, used to pick the control group")
    args = parser.parse_args(argv)

    recut, control = [], []
    for meta_path in INGEST.glob("*__clip*/meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if "source_fps" in meta:
            recut.append(meta_path.parent.name)
    rates = {}
    if args.rates.is_file():
        rates = json.loads(args.rates.read_text(encoding="utf-8"))
    for stem, rate in rates.items():
        if rate and abs(rate - GRID_FPS) < 1e-9:
            for directory in INGEST.glob(stem + "__clip*"):
                # Not filtered on a local clip.mp4.  These were never re-cut so
                # their videos live only in the store, and judge() fetches them
                # -- filtering here is what made the control group silently
                # empty, and a check with no control cannot be told apart from
                # a check that always passes.
                if (directory / "meta.json").is_file():
                    control.append(directory.name)

    random.seed(args.seed)
    random.shuffle(recut)
    random.shuffle(control)
    # Half the sample is drawn from clips that do not start at source frame 0.
    # Only those have a negative control worth anything: when start is 0 the
    # old and new audio offsets are both 0, the two spans coincide, and the
    # comparison cannot tell a working cut from a broken one.  Sampling
    # uniformly would fill the run with clip000s and quietly report a small
    # number of discriminating pairs as if it were the sample size.
    def starts_late(stem):
        try:
            meta = json.loads((INGEST / stem / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        span = meta.get("source_frame_span")
        return bool(span) and span[0] > 0

    late = [s for s in recut if starts_late(s)]
    early = [s for s in recut if s not in set(late)]
    half = args.sample // 2
    recut = (late[:half] + early[: args.sample - min(half, len(late))])[: args.sample]
    control = control[: args.control]
    print("re-cut clips sampled: {}   untouched 30 fps controls: {}".format(
        len(recut), len(control)), flush=True)
    if not control:
        print("NOTE: no control clips have a local clip.mp4 -- the 30 fps corpus "
              "was never re-cut, so its videos live only in OSS.  The control "
              "group is empty and the run says so rather than omitting it.")

    rows = [judge(stem, args.tolerance, not args.no_audio) for stem in recut]
    controls = [judge(stem, args.tolerance, not args.no_audio) for stem in control]

    def tally(group):
        ok = [r for r in group if r["status"] == "ok"]
        duration_pass = [r for r in ok if r.get("duration_ok")]
        with_audio = [r for r in ok if r.get("audio_match_correct_span") is not None]
        discriminating = [r for r in with_audio
                          if r.get("audio_match_old_span") is not None]
        return {
            "checked": len(group), "readable": len(ok),
            "duration_pass": len(duration_pass),
            "duration_fail": [r["clip"] for r in ok if not r.get("duration_ok")],
            "audio_checked": len(with_audio),
            "audio_correct_span_median": _median(
                [r["audio_match_correct_span"] for r in with_audio]),
            "audio_old_span_median": _median(
                [r["audio_match_old_span"] for r in discriminating]),
            "audio_beats_old_span": sum(
                1 for r in discriminating
                if r["audio_match_correct_span"] > r["audio_match_old_span"]),
            "audio_discriminating_pairs": len(discriminating),
        }

    result = {"generated_by": "tools/audit_recut_clips.py",
              "tolerance_seconds": args.tolerance,
              "recut": tally(rows), "control": tally(controls),
              "recut_rows": rows, "control_rows": controls}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")

    for name, group in (("re-cut", result["recut"]), ("control", result["control"])):
        if not group["checked"]:
            continue
        print("\n--- {} ---".format(name))
        print("  duration within {:.2f}s of (end-start)/source_fps : {}/{}".format(
            args.tolerance, group["duration_pass"], group["readable"]))
        if group["duration_fail"]:
            print("    FAILING: {}".format(", ".join(group["duration_fail"][:8])))
        if group["audio_checked"]:
            print("  audio envelope vs the span it should hold  : median {}".format(
                group["audio_correct_span_median"]))
            if group["audio_discriminating_pairs"]:
                print("  audio envelope vs the span the OLD code took: median {}".format(
                    group["audio_old_span_median"]))
                print("  correct span scores higher on {}/{} clips where the two "
                      "spans differ".format(group["audio_beats_old_span"],
                                            group["audio_discriminating_pairs"]))
    print("\nwrote {}".format(args.output))

    failed = result["recut"]["duration_fail"] or result["control"]["duration_fail"]
    return 1 if failed else 0


def _median(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    values = sorted(values)
    middle = len(values) // 2
    return round(values[middle] if len(values) % 2 else
                 0.5 * (values[middle - 1] + values[middle]), 4)


if __name__ == "__main__":
    raise SystemExit(main())
