#!/usr/bin/env python3
"""Do a clip's picture and its sound come from the same span of the upload?

This is the defect itself, not the container property that predicts it.
``tools/check_recut_happened.py`` asks whether the two rate claims agree, which
is cheap, decidable from metadata, and one inference away from the thing that
matters.  This asks the thing that matters, on the bytes:

* **Picture** -- the clip's first and last frames are located inside the upload
  by pixel match (32x18 grayscale, L1), and their *presentation timestamps* are
  read from the container rather than computed as index/rate.  That distinction
  is the whole subject: on a variable-rate file a frame index does not name an
  instant, so deriving the time from the index would build the assumption under
  test into the measurement.
* **Sound** -- the clip's audio is located inside the upload's audio by
  cross-correlating onset envelopes at 100 Hz.

Two numbers come out.  ``picture_over_sound`` is the picture's span divided by
the sound's length; ``start_offset`` is where the sound starts minus where the
picture starts.  A clip cut correctly reads 1.0 and 0.0.

Measured by hand 2026-08-24, which is what this tool automates:

    7438547996335295781__clip000   picture 27.17-54.37 s, sound 13.63 s for
                                   13.65 s  -> 1.992, offset 13.54 s
    two constant-rate controls                -> 0.994 and 0.996, offset 0.00 s

**It needs both controls to be read together.**  A ratio near 1.0 on its own
says the two spans have the same *length*, which a clip that took its picture
and its sound from two different places at the same scale also satisfies -- so
the offset is not a supplementary reading, it is half the criterion.  And the
pixel match can fail outright on a static shot, where hundreds of frames are
equally good matches; that is reported as ``picture_match_ambiguous`` rather
than as a span, because a confident wrong span is worse than no span.

Usage::

    measure_clip_av_sync.py --clip <ingest>/<upload>__clip000 --source u.mp4
    measure_clip_av_sync.py --clips a b c --ingest <ingest> --output sync.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

THUMB = (32, 18)          # the pixel-match resolution
AUDIO_RATE = 16000
ENVELOPE_HZ = 100         # onset envelope frames per second
# Below this the pixel match is not a match.  Two frames of the same shot at
# 32x18 differ by well under 0.02 in normalised L1; unrelated frames run 0.05
# and up.  The ratio between best and second-best over *non-adjacent* frames is
# what actually decides, so this is only a floor.
AMBIGUITY_RATIO = 1.5
# How far from the winning audio lag a rival has to sit to count as a rival
# rather than as the same peak's shoulder.  Clip envelopes here are 12-24 s
# long, so their autocorrelation is wide; 1 s is well inside one peak and well
# outside the loop periods that produce the real rivals (8-16 s, measured).
RIVAL_GUARD_S = 1.0
# Two peaks within this many sd of each other are not distinguishable by this
# surface.  Used only to ask whether the picture's location is *also* a peak.
PEAK_TIE_Z = 2.0
# Half-width of the window searched around the location the picture implies.
# Wide enough to cover the encoder's own rounding and the one-or-two-sample
# width of the peak, far narrower than the loop periods (8-16 s) that produce
# rival peaks.
EXPECTED_WINDOW_S = 0.2
# How close a displacement has to be to one of the music's own repeat periods
# to count as that repeat rather than as a coincidence.  A bar at 120 BPM is 2 s,
# so this is a twentieth of one.
REPEAT_TOLERANCE_S = 0.1


def run(command) -> subprocess.CompletedProcess:
    return subprocess.run([str(c) for c in command], capture_output=True, text=True)


def frame_times(video: pathlib.Path) -> np.ndarray:
    """Every video frame's presentation time, from the container.

    Read rather than computed.  ``index / rate`` is exactly the assumption this
    tool exists to test, and on the uploads in question it is false.
    """
    result = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                  "-show_entries", "frame=pts_time,best_effort_timestamp_time",
                  "-of", "json", video])
    try:
        frames = json.loads(result.stdout or "{}").get("frames", [])
    except ValueError:
        return np.zeros(0, dtype=np.float64)
    times = []
    for frame in frames:
        raw = frame.get("pts_time") or frame.get("best_effort_timestamp_time")
        try:
            times.append(float(raw))
        except (TypeError, ValueError):
            continue
    return np.asarray(times, dtype=np.float64)


def thumbnails(video: pathlib.Path, limit: Optional[int] = None) -> np.ndarray:
    """Every frame as a 32x18 grayscale row, in decode order."""
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        return np.zeros((0, THUMB[0] * THUMB[1]), dtype=np.float32)
    rows = []
    try:
        while limit is None or len(rows) < limit:
            ok, frame = capture.read()
            if not ok:
                break
            small = cv2.cvtColor(cv2.resize(frame, THUMB), cv2.COLOR_BGR2GRAY)
            rows.append(small.reshape(-1).astype(np.float32) / 255.0)
    finally:
        capture.release()
    return np.asarray(rows, dtype=np.float32) if rows else np.zeros(
        (0, THUMB[0] * THUMB[1]), dtype=np.float32)


def locate_frame(needle: np.ndarray, haystack: np.ndarray) -> Dict:
    """Where in ``haystack`` is ``needle``, and is the answer distinguishable?

    The second half is not a nicety.  A locked-off shot gives hundreds of frames
    that match a still pose equally well, and the argmin over them is noise
    wearing the shape of a measurement.
    """
    if len(haystack) == 0:
        return {"index": None, "reason": "source_unreadable"}
    distance = np.abs(haystack - needle[None, :]).mean(axis=1)
    best = int(np.argmin(distance))
    # Second best, ignoring the neighbourhood of the winner: adjacent frames of
    # the same shot are supposed to be similar, and counting them as rivals
    # would call every real match ambiguous.
    mask = np.ones(len(distance), dtype=bool)
    mask[max(0, best - 15):best + 16] = False
    rival = float(distance[mask].min()) if mask.any() else float("inf")
    # Divided by the winner or by a floor, never special-cased to "infinitely
    # distinguishable".  A frozen shot makes *every* distance zero, so the old
    # ``if distance[best] > 0 else inf`` reported the one case that is entirely
    # ambiguous as the one case that is perfectly certain -- and a locked-off
    # camera is common in this corpus.  With the floor, zero against zero reads
    # 0 (ambiguous) while an exact match against a distant rival still reads
    # large (certain), which is the distinction that was wanted.
    ratio = rival / max(distance[best], 1e-9)
    return {"index": best, "distance": float(distance[best]),
            "rival_distance": rival, "distinguishability": float(ratio),
            "ambiguous": bool(ratio < AMBIGUITY_RATIO)}


def onset_envelope(audio: pathlib.Path) -> np.ndarray:
    """A 100 Hz half-wave-rectified energy difference -- phase-robust enough to
    correlate two encodes of the same sound, cheap enough to run per clip."""
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(audio), "-vn",
         "-ac", "1", "-ar", str(AUDIO_RATE), "-f", "f32le", "-"],
        capture_output=True)
    if result.returncode != 0 or not result.stdout:
        return np.zeros(0, dtype=np.float32)
    samples = np.frombuffer(result.stdout, dtype=np.float32)
    hop = AUDIO_RATE // ENVELOPE_HZ
    usable = (len(samples) // hop) * hop
    if usable < hop * 4:
        return np.zeros(0, dtype=np.float32)
    energy = np.sqrt((samples[:usable].reshape(-1, hop) ** 2).mean(axis=1) + 1e-12)
    envelope = np.diff(np.log(energy), prepend=np.log(energy[0]))
    return np.maximum(envelope, 0.0).astype(np.float32)


def locate_audio(clip_env: np.ndarray, source_env: np.ndarray,
                 expected_seconds: Optional[float] = None) -> Dict:
    """Best lag of the clip's envelope inside the source's, by correlation.

    **Dance uploads loop.**  A four-bar phrase repeated through a 60 s video
    gives the correlation surface a rival peak at every repetition, and the
    tallest one is not reliably the right one -- measured 2026-08-25, four of
    46 re-cut clips located their sound 8.7 to 15.5 s before their picture, all
    at what turned out to be a whole number of loops.  So two things:

    * the rival is taken **outside the winner's own peak** (``RIVAL_GUARD_S``).
      The first version of this took the 100th-largest score, which for a peak
      more than a second wide is still on the same peak -- it measured the
      winner's shoulder and reported it as competition, and its margin
      therefore did not separate the four wrong locations from the 42 right
      ones (median 8.9 against 18.5, overlapping).
    * when ``expected_seconds`` is given -- the place the *picture* was found --
      the score there is reported too.  If the surface is as high at the
      picture's location as at its global maximum, the clip is not out of sync;
      the locator merely picked a different repetition of the same music, and
      saying "out of sync" there would be a confident wrong answer.
    """
    if len(clip_env) < ENVELOPE_HZ or len(source_env) < len(clip_env):
        return {"start_seconds": None, "reason": "envelope_too_short"}
    clip = clip_env - clip_env.mean()
    # Right-padded before correlating, and this is a correctness fix rather
    # than a convenience.  ``mode="valid"`` only offers lags at which the clip
    # fits entirely inside the source, so the **last clip of an upload cannot be
    # placed at all** -- its true lag is one or two samples past the end of the
    # valid range, and argmax then lands on whatever earlier peak is tallest.
    # Measured 2026-08-25: of 46 re-cut clips, four read a sound offset of -8.7
    # to -15.5 s and all four were last clips.  Padding moved three of them onto
    # their picture's own location (+0.01, +0.04, +0.01 s) and left a
    # known-correct control exactly where it was, at 17.83 s.  The fourth is a
    # genuine tie: its music repeats.
    source = np.concatenate([source_env, np.zeros(len(clip_env), dtype=np.float32)])
    source = source - source.mean()
    scores = np.correlate(source, clip, mode="valid")
    best = int(np.argmax(scores))
    spread = float(scores.std()) or 1e-9
    centre, scale = float(scores.mean()), spread

    def z(index: int) -> float:
        return float((scores[index] - centre) / scale)

    guard = int(RIVAL_GUARD_S * ENVELOPE_HZ)
    mask = np.ones(len(scores), dtype=bool)
    mask[max(0, best - guard):best + guard + 1] = False
    rival = int(np.argmax(np.where(mask, scores, -np.inf))) if mask.any() else best

    result = {"start_seconds": best / ENVELOPE_HZ,
              "peak_z": z(best),
              "rival_seconds": rival / ENVELOPE_HZ,
              "rival_z": z(rival),
              "length_seconds": len(clip_env) / ENVELOPE_HZ}
    if expected_seconds is not None:
        # A *window*, not a sample.  The envelope is a half-wave-rectified
        # energy difference at 100 Hz, which on music is close to an impulse
        # train, so the correlation peak is one or two samples wide and a
        # single-sample probe lands beside it.  Measured 2026-08-25 on a clip
        # whose sound and picture agree to 0.01 s: the peak read 22.4 sd at
        # 17.83 s and the sample at 17.82 s read **-2.3** -- so the first
        # version of this test called a correct clip's own location "not a
        # peak", and would have made every offset look like a real finding.
        centre_index = int(round(expected_seconds * ENVELOPE_HZ))
        half = int(EXPECTED_WINDOW_S * ENVELOPE_HZ)
        low, high = max(0, centre_index - half), min(len(scores), centre_index + half + 1)
        result["expected_seconds"] = expected_seconds
        if low < high:
            local = int(low + np.argmax(scores[low:high]))
            result["z_at_expected"] = z(local)
            result["expected_best_seconds"] = local / ENVELOPE_HZ
            # Both in sd units of the same surface, so "as high as" is a
            # comparison rather than a judgement call.
            result["expected_is_a_peak"] = bool(
                z(local) >= result["peak_z"] - PEAK_TIE_Z)
        else:
            # The lag the picture implies is outside the correlation's valid
            # range: the clip is longer than what remains of the upload's audio
            # from there.  Named, because "we could not look" is not "no peak".
            result["expected_out_of_range"] = True
            result["source_envelope_seconds"] = len(source_env) / ENVELOPE_HZ
    return result


def repeat_periods(source_env: np.ndarray, low: float = 2.0,
                   high: float = 30.0, top: int = 6) -> List[Dict]:
    """Lags at which the upload's own music repeats, from its autocorrelation.

    Independent of the clip and of the locator, which is the point: when a clip
    lands one repetition away from its picture, the two readings that disagree
    are both produced by the same correlation, and a third opinion has to come
    from somewhere else.  Measured 2026-08-25 on the one clip of 46 that stayed
    displaced after the padding fix: its two candidate sound locations were
    15.48 s apart, and this upload's envelope autocorrelates at 3.87, 7.74,
    **15.48** and 23.23 s -- a harmonic series on one bar -- against a baseline
    of 0.01 at unrelated lags.
    """
    if len(source_env) < int(high * ENVELOPE_HZ) + 2:
        return []
    centred = source_env - source_env.mean()
    auto = np.correlate(centred, centred, mode="full")[len(centred) - 1:]
    if not auto[0]:
        return []
    auto = auto / auto[0]
    lags = np.arange(len(auto)) / ENVELOPE_HZ
    window = (lags >= low) & (lags <= high)
    if not window.any():
        return []
    values, positions = auto[window], lags[window]
    order = np.argsort(values)[::-1][:top]
    return [{"seconds": round(float(positions[i]), 3), "r": round(float(values[i]), 4)}
            for i in sorted(order, key=lambda j: -values[j])]


def measure(clip_dir: pathlib.Path, source: pathlib.Path) -> Dict:
    video = clip_dir / "clip.mp4"
    row: Dict = {"clip": clip_dir.name, "source": str(source)}
    if not video.is_file():
        row["status"] = "clip_video_missing"
        return row
    if not source.is_file():
        row["status"] = "source_missing"
        return row

    clip_frames = thumbnails(video)
    if len(clip_frames) < 2:
        row["status"] = "clip_unreadable"
        return row
    source_frames = thumbnails(source)
    times = frame_times(source)
    if len(source_frames) == 0 or len(times) == 0:
        row["status"] = "source_unreadable"
        return row
    # Decode order and probe order are both presentation order here, but they
    # can differ in length by a frame or two on a truncated file; the shorter
    # governs, and the discrepancy is recorded rather than smoothed over.
    usable = min(len(source_frames), len(times))
    row["source_frames_decoded"] = int(len(source_frames))
    row["source_frames_probed"] = int(len(times))
    source_frames, times = source_frames[:usable], times[:usable]

    first = locate_frame(clip_frames[0], source_frames)
    last = locate_frame(clip_frames[-1], source_frames)
    row["first_frame"], row["last_frame"] = first, last
    if first.get("index") is None or last.get("index") is None:
        row["status"] = "picture_unlocated"
        return row
    if first.get("ambiguous") or last.get("ambiguous"):
        # A static shot.  Reported, not guessed at.
        row["status"] = "picture_match_ambiguous"
        return row
    start, end = float(times[first["index"]]), float(times[last["index"]])
    if end <= start:
        row["status"] = "picture_span_not_forward"
        row["picture_start"], row["picture_end"] = start, end
        return row
    row["picture_start"], row["picture_end"] = round(start, 4), round(end, 4)
    row["picture_span"] = round(end - start, 4)

    clip_env, source_env = onset_envelope(video), onset_envelope(source)
    sound = locate_audio(clip_env, source_env, expected_seconds=start)
    row["sound"] = sound
    if sound.get("start_seconds") is None:
        row["status"] = "sound_unlocated"
        return row
    row["sound_start"] = round(sound["start_seconds"], 4)
    row["sound_length"] = round(sound["length_seconds"], 4)
    row["picture_over_sound"] = round(row["picture_span"] / row["sound_length"], 4)
    row["start_offset"] = round(row["sound_start"] - row["picture_start"], 4)
    # A large offset means one of two different things and they must not be
    # merged.  Either the sound really came from elsewhere in the upload -- the
    # defect -- or the music repeats and the locator preferred another
    # repetition, in which case the surface is just as high where the picture
    # is.  Only the first is a finding.
    row["sound_location_ambiguous"] = bool(sound.get("expected_is_a_peak")
                                           and abs(row["start_offset"]) > 0.5)
    # A large offset where the picture's own location is *not* a peak, and the
    # lag could actually be examined.  This is the finding; the line above is
    # the thing that looks like one.
    row["sound_from_elsewhere"] = bool(
        abs(row["start_offset"]) > 0.5
        and not sound.get("expected_is_a_peak")
        and not sound.get("expected_out_of_range"))
    if abs(row["start_offset"]) > 0.5:
        # Third opinion, from the upload alone.  The two readings that disagree
        # both came out of the same correlation, so "which is right" cannot be
        # settled by looking harder at it.  If the displacement is one of the
        # music's own repeat periods, the locator preferred another repetition
        # and nothing is out of sync.
        periods = repeat_periods(source_env)
        row["source_repeat_periods"] = periods
        match = [p for p in periods
                 if abs(abs(row["start_offset"]) - p["seconds"]) <= REPEAT_TOLERANCE_S]
        row["explained_by_repeat"] = match[0] if match else None
        if match:
            row["sound_from_elsewhere"] = False
            row["sound_location_ambiguous"] = True
    row["status"] = "ok"
    return row


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clip", type=pathlib.Path, default=None,
                        help="one clip directory holding clip.mp4 and meta.json")
    parser.add_argument("--clips", nargs="*", default=None,
                        help="clip names, resolved under --ingest")
    parser.add_argument("--ingest", type=pathlib.Path, default=None)
    parser.add_argument("--source", type=pathlib.Path, default=None,
                        help="the upload to locate the clip in; by default the "
                             "``source`` recorded in the clip's meta.json, or "
                             "``cfr_normalized_from`` when the clip was cut "
                             "from a re-encode -- the question is about the "
                             "original, not the intermediate")
    parser.add_argument("--output", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)

    directories = []
    if args.clip:
        directories.append(args.clip)
    for name in (args.clips or []):
        if not args.ingest:
            raise SystemExit("--clips needs --ingest")
        directories.append(args.ingest / name)
    if not directories:
        raise SystemExit("nothing to measure: pass --clip or --clips")

    rows = []
    for directory in directories:
        source = args.source
        if source is None:
            meta_path = directory / "meta.json"
            meta = (json.loads(meta_path.read_text(encoding="utf-8"))
                    if meta_path.is_file() else {})
            source = pathlib.Path(meta.get("cfr_normalized_from")
                                  or meta.get("source") or "")
        row = measure(directory, source)
        rows.append(row)
        if row["status"] == "ok":
            print("{}  picture {:.2f}-{:.2f} s, sound {:.2f} s for {:.2f} s"
                  "  -> picture/sound {:.3f}, offset {:+.2f} s{}".format(
                      row["clip"], row["picture_start"], row["picture_end"],
                      row["sound_start"], row["sound_length"],
                      row["picture_over_sound"], row["start_offset"],
                      "  [the music repeats every {:.2f} s (r={:.3f}); the "
                      "displacement is one of those]".format(
                          row["explained_by_repeat"]["seconds"],
                          row["explained_by_repeat"]["r"])
                      if row.get("explained_by_repeat") else
                      "  [SOUND FROM ELSEWHERE: the picture's own location "
                      "scores {:.1f} against the peak's {:.1f}]".format(
                          row["sound"].get("z_at_expected", float("nan")),
                          row["sound"]["peak_z"])
                      if row.get("sound_from_elsewhere") else ""))
        else:
            print("{}  {}".format(row["clip"], row["status"]))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        staging = args.output.with_suffix(args.output.suffix + ".tmp")
        staging.write_text(json.dumps({"rows": rows}, indent=1) + "\n",
                           encoding="utf-8")
        staging.replace(args.output)
        print("wrote", args.output)
    return 0 if any(r["status"] == "ok" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
