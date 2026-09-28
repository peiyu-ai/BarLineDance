#!/usr/bin/env python3
"""Raw upload -> clip cache, cut on content instead of on a clock.

This is the front of the wild pipeline, brought into this repo from Lodge's
``scripts/preprocess_wild_videos.py``.  The move is not a copy: the two
constants that decided everything downstream were Lodge's, and neither was
chosen for AtomicDance.

    SEG = 16          # seconds per clip (match historical 16s cache)
    MIN_FRAMES = 180  # past60 + future120

``SEG`` is a historical cache size and the cut is ``ffmpeg -f segment``: blind,
fixed-length, aligned to nothing.  Measured on the published corpus that leaves
**10,564 of 78,227 segments (13.5%) with a boundary that is ffmpeg's clock**,
and because Alg. 1 sets its cluster count to ``T / frames_per_cluster``, it also
makes segmentation granularity a function of where the clock happened to fall:
mean segment length runs 1.171 s on 6-7 s clips down to 1.022 s on full-length
ones, monotonically, against the paper's 0.81 s target.  ``MIN_FRAMES = 180`` is
Lodge's motion-continuation window (60 past + 120 future frames) and means
nothing here.

What replaces them:

* **Where to cut** is a content question -- shot changes, and stretches with no
  dancer on screen.  Both are read from the same detection pass.
* **The lower bound** comes from Alg. 1 rather than from another model's
  training window: the segmenter needs enough clusters for a self-similarity
  structure to exist at all, and at ``T / 34`` per cluster, ten clusters is
  ``T >= 340`` frames.
* **The upper bound** is the one thing that cannot be reasoned out, because it
  trades segmentation granularity against monocular drift.  It is measured by
  ``tools/scan_clip_length.py`` and passed in as ``--max-seconds``; this tool
  does not carry a default for it, so a corpus cannot be built on a guess.

One decode, three consumers.  Detection runs once over the upload at full rate
(9.8 ms/frame) and feeds boundary detection, the dancer track, and the crop for
the pose pass -- which then costs 5 ms/frame because it runs on one box instead
of the ~7 this corpus averages.  See ``tools/dwpose_video.py`` for why the
dancer is chosen as a track, and for what ``rival_ratio`` records when the
footage is a group and the question has no answer.

The chosen dancer is written to ``preprocess/bbx.pt`` so GVHMR crops the same
person the 2D describes, rather than running its own tracker and picking
independently.

Output per clip, matching what ``preprocess_wild_3d.py`` inventories:

    <out>/<upload>__clip%03d/
        clip.mp4  audio.wav  keypoints.npy  scores.npy  detections.npz
        meta.json                       provenance, and why this cut is here
        preprocess/bbx.pt               the dancer, handed to GVHMR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import time
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.dwpose_video import (DWPoseExtractor, save, save_gvhmr_bbx)   # noqa: E402

FPS = 30

# The clip's own rate.  The *upload's* rate is measured per file by ``probe_fps``
# and is not this: 10.3% of the corpus is not 30 fps.


def _rate(fps: float) -> str:
    """ffmpeg wants an exact rate; 29.97 must not become 29.969999."""
    from fractions import Fraction

    ratio = Fraction(fps).limit_denominator(100000)
    return "{}/{}".format(ratio.numerator, ratio.denominator)
# Alg. 1 divides the clip into T/34 clusters; below ten of them there is not
# enough structure in the self-similarity matrix for a segmentation to mean
# anything.  This is the repo's own floor, derived from the algorithm it feeds.
FRAMES_PER_CLUSTER = 34
MIN_CLUSTERS = 10
MIN_FRAMES = FRAMES_PER_CLUSTER * MIN_CLUSTERS          # 340 frames = 11.3 s
# A dancer who leaves the frame for a beat has not ended the clip; one who is
# gone for a second has.
MAX_ABSENCE_FRAMES = 30
# Shot change, on two axes at once, because neither works alone.
#
# Measured over 40 uploads (~51k frames): 510 frames exceed 0.10 mean absolute
# greyscale difference, and exactly 8 of them are shot changes.  Cuts are a
# ~1e-4 event, which is why a quantile cannot place this threshold -- the whole
# calibration table sits inside the dancer-motion hump (p99 = 0.125).
#
# On the score axis the 8 real cuts run 0.237-0.754 and the next-largest
# non-cuts are 0.194 and below.  On the ratio-to-clip-median axis the cuts run
# 11.4-80.3 and the non-cuts 4.7 and below.  The gap is on *both* axes, so both
# are required: the floor alone fires on a camera sweeping past an obstruction,
# and the ratio alone explodes on a video whose median difference is literally
# zero (a static frame with an inset), where 2723x meant nothing.
#
# The constants sit inside the gap, placed toward the conservative end.  The two
# errors are not symmetric: a missed cut leaves a discontinuity that Alg. 1's
# self-similarity matrix sees anyway and usually segments at, while a spurious
# cut splits a good take and can drop both halves below the 340-frame floor,
# losing them entirely.
SHOT_CUT_FLOOR = 0.22
SHOT_CUT_RATIO = 8.0


class UnreadableUpload(RuntimeError):
    """The decoder refused this upload; the sweep records it and moves on."""


def run(command: Sequence[str], timeout: int = 1800) -> subprocess.CompletedProcess:
    return subprocess.run([str(c) for c in command], capture_output=True,
                          text=True, timeout=timeout)


def shot_cut_score(previous: np.ndarray, current: np.ndarray) -> float:
    """Mean absolute difference between consecutive frames, 0..1.

    Deliberately not a histogram or an edge measure.  What has to be separated
    here is "the camera cut to a different shot" from "the dancer moved fast",
    and on a fixed-camera dance video the second one changes a small part of the
    frame a lot while the first changes all of it -- which a whole-frame mean
    already separates.  Anything more elaborate is a second thing to calibrate.
    """
    return float(np.abs(current.astype(np.int16) - previous.astype(np.int16)).mean() / 255.0)


def find_spans(present: np.ndarray, cuts: np.ndarray, *, min_frames: int,
               max_frames: int, max_absence: int) -> List[Tuple[int, int, str]]:
    """Frame ranges that hold a continuous, single-shot, dancer-present take.

    Returns ``(start, end, reason)`` where the reason is why the span *ends*.
    It is recorded here rather than re-derived from the span afterwards, because
    re-deriving cannot tell a length split from a dancer walking out: the first
    version of this tool guessed, and reported 66% of boundaries as
    ``dancer_absent`` when most of them were its own even division.  A boundary's
    cause is known exactly once -- when it is created.

    Absence shorter than ``max_absence`` does not break a span: a dancer turning
    away or passing behind something is not a boundary, and cutting there would
    manufacture exactly the kind of false edge this tool exists to remove.

    A span longer than ``max_frames`` is divided into equal parts rather than
    truncated to the limit plus a short tail.  The tail is the problem: it is
    the short clip whose segmentation comes out coarse, and an equal division
    keeps every piece in the same granularity regime.

    But equal division alone destroys data, which a dry run over 278 cached
    uploads caught before any corpus was built: at a 16 s cap, a 500-frame take
    divides into two 250-frame pieces, *both* below the 340-frame floor, and the
    whole take is dropped -- 112 of 278 uploads produced nothing at all.  So the
    division is only used when it leaves every piece above the floor; otherwise
    the span is cut into whole ``max_frames`` pieces and the sub-floor remainder
    is dropped, which loses at most ``min_frames - 1`` frames instead of all of
    them.
    """
    spans: List[Tuple[int, int, str]] = []
    start: Optional[int] = None
    absent = 0
    for index in range(len(present)):
        if cuts[index] and start is not None:
            # The cut frame belongs to the next shot, so the span ends before it.
            spans.append((start, index - absent, "shot_cut"))
            start, absent = None, 0
        if present[index]:
            if start is None:
                start = index
            absent = 0
        elif start is not None:
            absent += 1
            if absent > max_absence:
                spans.append((start, index - absent + 1, "dancer_absent"))
                start, absent = None, 0
    if start is not None:
        spans.append((start, len(present) - absent,
                      "upload_end" if absent == 0 else "dancer_absent"))

    out: List[Tuple[int, int, str]] = []
    for begin, end, reason in spans:
        length = end - begin
        if length < min_frames:
            continue
        pieces = max(1, int(np.ceil(length / max_frames)))
        if length / pieces >= min_frames:
            # Even division, the preferred case: no short tail, and every piece
            # lands in [min_frames, max_frames].
            edges = np.linspace(begin, end, pieces + 1).astype(int)
        else:
            # Dividing evenly would put every piece under the floor and lose the
            # take entirely.  Take whole pieces from the start and drop the
            # sub-floor remainder: at most min_frames - 1 frames, not all of them.
            whole = int(length // max_frames)
            edges = begin + np.arange(whole + 1) * max_frames
        for position, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            if b - a < min_frames:
                continue
            # Only the last piece ends for the reason the take ended; every
            # earlier boundary is this function's own division.
            last = position == len(edges) - 2
            out.append((int(a), int(b), reason if last else "length_split"))
    return out


def manifest_rows(out_root: pathlib.Path) -> Dict[str, Dict]:
    """Each upload's *most recent* ingest record, keyed by upload stem.

    The manifests are append-only and are read by three places: this module's
    resume gate, ``compare_cut_ab.py``, and ``run_wild_ingest.sh``'s summary.
    Once ``--redo`` exists an upload can have two rows, and the three readers
    must agree that the later one describes the corpus.  Summing every row
    instead would count a re-ingested upload twice -- doubling uploads,
    upload_seconds, kept_seconds and every clip end reason -- and report a
    corpus larger than the one on disk.  Reading is shared so they cannot
    drift apart on that question.

    Shard membership changes with ``--num-shards``, so every manifest in the
    root is read rather than this shard's: which file an upload's row landed in
    is an artifact of how the work was divided, not information.

    "Most recent" is ``ingested_at``, not file order, and the difference is not
    academic.  The re-cut ran at 7 shards over a corpus originally ingested at
    14, so an upload's new row lands in ingest_shard0..6 while its old row can
    sit in ingest_shard10 -- which sorts *after* shard0 by name, so the old row
    won.  File mtime does not rescue it either: both files were appended to in
    the same run, and the file holding the older row can carry the later mtime.
    Rows written before this stamp existed sort as oldest, which is what they
    are.
    """
    candidates: Dict[str, tuple] = {}
    for manifest in sorted(out_root.glob("ingest_shard*.jsonl")):
        for index, line in enumerate(manifest.open(encoding="utf-8")):
            try:
                record = json.loads(line)
                upload = record["upload"]
            except (ValueError, KeyError):
                continue
            key = (float(record.get("ingested_at") or 0.0), manifest.name, index)
            if upload not in candidates or key > candidates[upload][0]:
                candidates[upload] = (key, record)
    return {upload: record for upload, (_, record) in candidates.items()}


VFR_TOLERANCE = 0.02


def probe_rates(upload: pathlib.Path) -> tuple:
    """``(avg_frame_rate, r_frame_rate)`` -- the two the container reports.

    They are not the same claim.  ``r_frame_rate`` is the rate the timebase can
    express; ``avg_frame_rate`` is frames divided by duration, i.e. what the
    file actually holds.  On a constant-rate file they agree and either will do.
    On a variable-rate one -- or one whose timebase is simply mis-tagged, which
    is common in phone uploads -- they do not, and *frame numbering has no fixed
    relationship to time*.  That matters here because ``cut_clip`` selects the
    picture by frame number and the sound by seconds; when the two rates
    disagree those are two different spans.

    Measured 2026-08-24 on ``7438547996335295781.mp4`` (avg 31.62, r 60.00):
    the clip's picture came from 27.17-54.37 s of the upload and its sound from
    13.63 s for 13.65 s -- **picture span over sound length 1.992, start
    offset 13.54 s**.  Two constant-rate controls cut the same day read 0.994
    and 0.996 with a 0.00 s offset, so the measurement separates.

    The old ``expected`` check in ``cut_clip`` could not catch this: it derives
    the frame count from the same rate the cut used, so it agrees with a wrong
    cut by construction.  A gate built out of the assumption it is meant to test
    cannot fail.
    """
    # JSON, not csv.  ffprobe emits the entries in the order the stream carries
    # them rather than the order they were asked for, so positional parsing gets
    # the two rates the wrong way round on some files -- which inverts the
    # verdict rather than breaking it, and reads as a measurement either way.
    result = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                  "-show_entries", "stream=avg_frame_rate,r_frame_rate",
                  "-of", "json", upload])
    try:
        stream = json.loads(result.stdout or "{}")["streams"][0]
    except (ValueError, KeyError, IndexError):
        return 0.0, 0.0

    def parse(field: str) -> float:
        # ZeroDivisionError as well as ValueError: ffprobe writes "0/0" for a
        # rate it cannot determine, and ``denominator or 1`` does not catch it
        # because the *string* "0" is truthy.  Uncaught, that crashes the ingest
        # on exactly the files whose rate is in question.
        try:
            numerator, _, denominator = str(field).partition("/")
            value = float(numerator) / float(denominator or 1)
        except (ValueError, ZeroDivisionError):
            return 0.0
        return value if 1.0 <= value <= 240.0 else 0.0

    return parse(stream.get("avg_frame_rate", "")), parse(stream.get("r_frame_rate", ""))


def is_variable_rate(avg: float, rate: float, tolerance: float = VFR_TOLERANCE) -> bool:
    """Do the container's two rate claims disagree enough to break a frame cut?"""
    if not avg or not rate:
        return False
    return abs(avg - rate) / max(avg, rate) > tolerance


def normalize_to_cfr(upload: pathlib.Path, destination: pathlib.Path,
                     rate: float) -> Optional[pathlib.Path]:
    """Re-encode an upload onto a constant frame rate, once, before cutting.

    This is the only fix that keeps ``cut_clip`` correct as written: it selects
    the picture by frame number, and a frame number only names an instant when
    the rate is constant.  Re-timing the cut instead would leave every other
    frame-indexed artifact of this clip -- the box track, the keypoints, the
    3D -- indexed against a timeline nothing else shares.

    ``-fps_mode cfr`` with an explicit ``-r`` duplicates or drops frames so that
    frame ``n`` is at ``n/rate`` seconds, which is what the rest of the ingest
    already assumes is true.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".partial.mp4")

    # ``-fps_mode`` is ffmpeg 5.0 and later.  This cluster ships 4.4
    # (libavcodec 58), where it is not merely ignored -- ffmpeg exits 1 with
    # "Unrecognized option 'fps_mode'", so the repair returned None for EVERY
    # upload and the variable-rate defect it exists to remove was never
    # removed.  ``-vsync cfr`` is the same behaviour under the older spelling.
    # Both are tried rather than sniffing a version string, because the version
    # is not the question -- whether this binary accepts the flag is.
    def encode(pacing):
        return run(["ffmpeg", "-nostdin", "-y", "-i", upload,
                    *pacing, "-r", _rate(rate),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                    "-c:a", "aac", partial])

    result = encode(["-fps_mode", "cfr"])
    if result.returncode != 0 and "fps_mode" in (result.stderr or ""):
        result = encode(["-vsync", "cfr"])
    if result.returncode != 0 or not partial.exists():
        partial.unlink(missing_ok=True)
        return None
    avg, r = probe_rates(partial)
    if is_variable_rate(avg, r):
        # It did not take.  Returning the file anyway would put the defect back
        # with a name that says it was fixed.
        partial.unlink(missing_ok=True)
        return None
    partial.rename(destination)
    return destination


def probe_fps(upload: pathlib.Path) -> float:
    """The upload's real frame rate, measured -- never assumed to be ``FPS``.

    2026-08-19: it had been assumed.  ``cut_clip`` selected video by *frame
    number* and audio by *seconds computed as frame/30*, so on an upload that
    is not 30 fps the two picked different spans: a 60 fps upload gave 451
    frames covering 7.5 s of action, restamped to 15.03 s, against 15.03 s of
    real audio -- the picture at half speed under the right music.  GVHMR then
    ran on that clip, so the 3D motion inherited the stretch while the music
    did not, and 1,418 of 13,783 released clips (10.3%) carried it.  Nothing
    reported it because ``meta.json`` recorded ``fps: 30.0`` from the constant,
    which reads exactly like a measurement.
    """
    result = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                  "-show_entries", "stream=r_frame_rate", "-of",
                  "default=nw=1:nk=1", upload])
    text = (result.stdout or "").strip()
    # See ``probe_rates``: "0/0" divides by zero rather than raising ValueError.
    try:
        numerator, _, denominator = text.partition("/")
        value = float(numerator) / float(denominator or 1)
    except (ValueError, ZeroDivisionError):
        return 0.0
    return value if 1.0 <= value <= 240.0 else 0.0


def cut_clip(upload: pathlib.Path, start: int, end: int,
             destination: pathlib.Path, source_fps: float = float(FPS)
             ) -> Optional[int]:
    """Cut source frames [start, end) out as a real-time span at 30 fps.

    ``start`` and ``end`` index the *upload's* frames, so the span they name is
    ``[start/source_fps, end/source_fps)`` seconds long.  The video is
    resampled onto a 30 fps grid over exactly that span and the audio is taken
    over exactly that span, which is the whole of the fix described in
    ``probe_fps``: before it, the video was renumbered rather than resampled
    and the audio span was computed as ``frame/30`` regardless of the source.

    On a 30 fps upload this is the identity -- ``setpts=(N/30)/TB`` then
    ``fps=30`` on an already-30 fps stream -- and that is checked by re-cutting
    rather than by reading, because 89.7% of the corpus depends on it.

    ``-ss`` before ``-i`` seeks on keyframes and would shift the span silently,
    so the seek is on the decoded stream.  The frame count is read back rather
    than assumed: the box track is indexed by frame, and a clip that came out a
    frame short would misalign every crop after it.
    """
    if not source_fps or source_fps <= 0:
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".partial.mp4")
    # The rate is parenthesised, and that is not cosmetic.  ``_rate`` returns a
    # fraction, so ``setpts=N/2997/100/TB`` parses as N/(2997*100) rather than
    # N/(2997/100) -- out by the square of the denominator.  Integer rates come
    # back as "30/1" and divide by 1 harmlessly, so the defect was invisible on
    # 92.8% of the corpus and hit every NTSC upload: on 2026-08-19 it failed 214
    # cuts loudly and, on two uploads where ffprobe returned a count instead of
    # nothing, produced five clips of 38 frames stamped across 20.34 seconds
    # that the run recorded as ok.
    result = run(["ffmpeg", "-nostdin", "-y", "-i", upload, "-vf",
                  "select=between(n\\,{}\\,{}),setpts=N/({})/TB,fps={}".format(
                      start, end - 1, _rate(source_fps), FPS),
                  "-af", "aselect=between(t\\,{}\\,{}),asetpts=N/SR/TB".format(
                      start / source_fps, end / source_fps),
                  "-r", FPS, "-c:v", "libx264", "-preset", "ultrafast",
                  "-c:a", "aac", partial])
    if result.returncode != 0 or not partial.exists():
        partial.unlink(missing_ok=True)
        return None
    partial.rename(destination)
    probe = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-count_frames", "-show_entries", "stream=nb_read_frames",
                 "-of", "default=nw=1:nk=1", destination])
    try:
        written = int(probe.stdout.strip())
    except ValueError:
        return None
    # The count is read back so it can be *checked*, not merely reported.  A
    # clip covering [start, end) source frames at source_fps must hold
    # (end-start) * 30 / source_fps frames; the resampler lands within a frame
    # or two of that.  Five clips came back holding 38 frames where the span
    # asked for 610 and were recorded as successes, because nothing compared
    # the number to anything.  A count this far off is a broken cut, and a
    # broken cut has to look like a failure rather than like a short clip.
    expected = (end - start) * FPS / source_fps
    if abs(written - expected) > max(3.0, 0.02 * expected):
        return None
    return written


def extract_audio(clip: pathlib.Path, destination: pathlib.Path) -> bool:
    result = run(["ffmpeg", "-nostdin", "-y", "-i", clip, "-ac", "1", "-ar", "22050",
                  "-vn", destination])
    return result.returncode == 0 and destination.exists()


class StaleScanCache(Exception):
    """A cached scan whose recorded source identity is not this file's."""


def source_key(upload: pathlib.Path) -> str:
    """``<bytes>:<sha256 of the first megabyte>`` -- this repo's video identity.

    The same key the S3D extractor and ``run_gvhmr_extract`` write beside their
    outputs, so a cache entry and a derived artifact are talking about the same
    thing when they agree.  The first megabyte is enough because a re-encode
    rewrites the container header; the size is carried too so a truncated copy
    that shares a header does not pass.
    """
    try:
        with open(upload, "rb") as handle:
            head = handle.read(1 << 20)
        return "{}:{}".format(upload.stat().st_size, hashlib.sha256(head).hexdigest())
    except OSError:
        return ""


def scan_upload(extractor: DWPoseExtractor, upload: pathlib.Path,
                cache_dir: Optional[pathlib.Path] = None
                ) -> Tuple[List[np.ndarray], np.ndarray]:
    """One decode: detections for the whole upload, plus the shot-cut trace.

    This is the expensive half of ingestion -- 9.8 ms/frame, ~22 GPU-hours over
    the corpus -- and none of it depends on the clip-length cap or the cut
    thresholds; only ``find_spans`` does, and that is arithmetic.  Caching it
    therefore lets the scan run before those constants are settled, and lets the
    corpus be re-cut under a different cap without paying for detection twice.

    Boxes are ragged (frames hold different numbers of people), so they are
    stored flat with per-frame counts rather than as an object array, which
    ``np.load`` would refuse without ``allow_pickle``.

    **The cache is per frame index, so it is only valid for the exact bytes it
    was built from.**  It used to be keyed by ``upload.stem`` and nothing else,
    which was safe only as long as no file was ever re-encoded -- and the
    variable-rate fix re-encodes uploads onto a constant rate under the *same
    file name*.  Loading the old entry there would hand a 25 fps scan to a 30
    fps video and index every detection against a grid it was not measured on:
    the same class of defect as the one being repaired, arriving through the
    repair.  Two things stop it.  The re-encodes get their own cache directory,
    so the names cannot meet; and every entry written from 2026-08-25 records
    ``source_key`` -- size plus the sha256 of the first megabyte, the identity
    the rest of this repo already joins on -- which is checked when present.
    Entries written before that carry no key and are still accepted, because
    the corpus scan cost ~22 GPU-hours and nothing has re-encoded in place.
    """
    import cv2

    cache_path = (cache_dir / "{}.npz".format(upload.stem)) if cache_dir else None
    key = source_key(upload) if cache_path is not None else None
    if cache_path is not None and cache_path.exists():
        try:
            stored = np.load(cache_path)
            stored_key = stored["source_key"].item() if "source_key" in stored else None
            if stored_key is not None and stored_key != key:
                raise StaleScanCache(
                    "{} was built from different bytes ({} != {})".format(
                        cache_path, stored_key, key))
            counts = stored["counts"]
            flat = stored["boxes"].reshape(-1, 4)
            edges = np.concatenate([[0], np.cumsum(counts)])
            return ([flat[a:b] for a, b in zip(edges[:-1], edges[1:])],
                    stored["cut_scores"])
        except StaleScanCache as error:
            # Not a corrupt file: a scan of a different video under this name.
            # Deleting and rescanning is right, and saying so is the point --
            # silently redoing it would hide an in-place re-encode.
            print("scan cache discarded: {}".format(error), flush=True)
            cache_path.unlink(missing_ok=True)
        except Exception:                      # noqa: BLE001 - a bad cache is redone
            cache_path.unlink(missing_ok=True)

    capture = cv2.VideoCapture(str(upload))
    if not capture.isOpened():
        # Distinguish the two causes: one is a missing file, the other is a
        # container OpenCV will not open even though ffprobe reads it happily
        # (one 201-second, 3 MB upload in this corpus does exactly that).  The
        # single old message sent the second case looking for the first.
        raise UnreadableUpload(
            "{} exists but cv2 could not open it".format(upload)
            if upload.exists() else "no such upload: {}".format(upload))
    scores: List[float] = []
    previous = None
    detections: List[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            detections.append(extractor.detect(frame))
            small = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
            scores.append(0.0 if previous is None else shot_cut_score(previous, small))
            previous = small
    finally:
        capture.release()
    cut_scores = np.asarray(scores, dtype=np.float32)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        counts = np.asarray([len(d) for d in detections], dtype=np.int32)
        flat = (np.concatenate(detections).astype(np.float32) if detections
                else np.zeros((0, 4), dtype=np.float32))
        partial = cache_path.with_suffix(".partial.npz")
        np.savez_compressed(partial, counts=counts, boxes=flat,
                            cut_scores=cut_scores,
                            source_key=np.asarray(key or ""))
        partial.rename(cache_path)
    return detections, cut_scores


def find_cuts(scores: np.ndarray, *, floor: float = SHOT_CUT_FLOOR,
              ratio: float = SHOT_CUT_RATIO) -> np.ndarray:
    """Frames where the shot changed: large in absolute terms *and* for this clip.

    The clip's own median is the scale.  A slow, static video and a fast, shaky
    one have frame-difference distributions an order of magnitude apart, and one
    absolute number cannot serve both -- but the ratio alone divides by zero on
    a static frame, so the floor is what keeps that finite.
    """
    if not len(scores):
        return np.zeros(0, dtype=bool)
    median = float(np.median(scores))
    return (scores > floor) & (scores > ratio * max(median, 1e-6))


def ingest(upload: pathlib.Path, out_root: pathlib.Path, *,
           extractor: DWPoseExtractor, max_seconds: float,
           min_frames: int = MIN_FRAMES, shot_floor: float = SHOT_CUT_FLOOR,
           shot_ratio: float = SHOT_CUT_RATIO,
           max_absence: int = MAX_ABSENCE_FRAMES, write_bbx: bool = True,
           cache_dir: Optional[pathlib.Path] = None,
           cfr_cache: Optional[pathlib.Path] = None) -> Dict:
    from tools.dwpose_video import VideoPose, link_tracks, score_track

    import cv2

    average_fps, container_fps = probe_rates(upload)
    source_fps = container_fps or average_fps
    if not source_fps:
        # Refuse rather than fall back to FPS.  Falling back is exactly what
        # produced the 2x-stretched clips: an unmeasured rate that reads like a
        # measured one, carried into meta.json and from there into the 3D.
        return {"upload": upload.name, "status": "unreadable_frame_rate"}
    normalized_from = None
    if is_variable_rate(average_fps, container_fps):
        # The picture is cut by frame number and the sound by seconds; on this
        # file those name different spans (measured 1.992x apart on one upload,
        # against 0.994/0.996 on constant-rate controls).  Either normalise the
        # upload once, or refuse -- producing clips here is producing the defect.
        if cfr_cache is None:
            return {"upload": upload.name, "status": "variable_frame_rate",
                    "avg_frame_rate": round(average_fps, 4),
                    "r_frame_rate": round(container_fps, 4)}
        target = pathlib.Path(cfr_cache) / upload.name
        normalized = (target if target.is_file()
                      else normalize_to_cfr(upload, target, average_fps))
        if normalized is None:
            return {"upload": upload.name, "status": "cfr_normalize_failed",
                    "avg_frame_rate": round(average_fps, 4),
                    "r_frame_rate": round(container_fps, 4)}
        # Resolved, not as given.  Re-cut runs are launched over a scratch
        # directory of symlinks (the uploads live in three different trees), and
        # that directory is deleted afterwards -- so recording the link would
        # leave every re-cut clip naming a path that no longer exists as the
        # only record of what it came from.
        normalized_from = str(upload.resolve())
        upload = normalized
        # The re-encode carries the original's file name, and the scan cache is
        # keyed by name.  Without this the 25 fps scan would be handed to the
        # 30 fps re-encode and every detection indexed against a grid it was
        # not measured on -- the defect being fixed, arriving through the fix.
        if cache_dir is not None:
            cache_dir = pathlib.Path(cache_dir) / "cfr"
        average_fps, container_fps = probe_rates(upload)
        source_fps = container_fps or average_fps
        if not source_fps:
            return {"upload": upload.name, "status": "unreadable_frame_rate"}

    # Identity of the file the clips are actually cut from -- after any
    # constant-rate normalisation, because that is the file whose frame numbers
    # the spans index.  Written into every clip's meta and read by the skip test
    # above.
    source_identity = source_key(upload)

    detections, cut_scores = scan_upload(extractor, upload, cache_dir)
    frames = len(detections)
    if not frames:
        return {"upload": upload.stem, "status": "unreadable", "clips": []}

    probe = cv2.VideoCapture(str(upload))
    width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    probe.release()

    tracks = link_tracks(detections)
    ranked = sorted(tracks, key=lambda t: -score_track(
        t, frames=frames, frame_area=float(width * height)))
    box_track = np.full((frames, 4), np.nan, dtype=np.float32)
    area = float(width * height)
    # Per-upload constants.  They were being recomputed six times per clip, and
    # score_track walks the whole track each call.
    scored = [score_track(t, frames=frames, frame_area=area) for t in ranked]
    best_score = scored[0] if scored else 0.0
    rival_ratio = (scored[1] / best_score) if len(scored) > 1 and best_score else 0.0
    rival_count = sum(1 for s in scored[1:] if best_score and s >= 0.5 * best_score)
    if ranked:
        box_track[ranked[0]["frames"]] = ranked[0]["boxes"]
    present = np.isfinite(box_track).all(axis=1)
    cuts = find_cuts(cut_scores, floor=shot_floor, ratio=shot_ratio)

    # Both bounds are policies about *seconds of dance*, and ``present``/``cuts``
    # are indexed by the upload's own frames, so both convert through the
    # upload's measured rate.  Reading them as 30 fps -- which is what happened
    # until 2026-08-19 -- makes a 60 fps upload's clips half the intended length
    # of real dance while still calling them 340 frames.
    spans = find_spans(present, cuts,
                       min_frames=int(round(min_frames * source_fps / FPS)),
                       max_frames=int(round(max_seconds * source_fps)),
                       max_absence=max_absence)

    record = {
        "upload": upload.stem,
        "source": str(upload),
        "frames": frames,
        "seconds": round(frames / source_fps, 2),
        "dancer_present_fraction": round(float(present.mean()), 4),
        "shot_cuts": int(cuts.sum()),
        "spans": [[int(a), int(b), why] for a, b, why in spans],
        "clips": [],
        "status": "ok" if spans else "no_usable_span",
    }

    for index, (start, end, end_reason) in enumerate(spans):
        name = "{}__clip{:03d}".format(upload.stem, index)
        out_dir = out_root / name
        if (out_dir / "meta.json").exists():
            # "Already there" is only a skip if it is the *same* clip.  The name
            # is upload + ordinal, so re-cutting the corpus under a different
            # --max-seconds would leave __clip000 holding a span from the old
            # constant under a name the new run believes it wrote -- two
            # generations of the cut mixed in one corpus with nothing to
            # separate them afterwards.
            try:
                previous = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
            except ValueError:
                previous = {}
            # Two conditions, and the second one used to be a generation
            # marker: "does this meta have a ``source_fps`` field", which the
            # fps-aware ingest of 2026-08-19 added.  **A marker made of a
            # field's presence goes stale the moment a third generation
            # exists**, and it did.  On 2026-08-25 the variable-rate re-cut hit
            # 7564725723304119595__clip000: the August clip records
            # ``source_fps: 33.0`` (so it read as current), its span is
            # [0, 399], and the span computed on the constant-rate re-encode is
            # also [0, 399] -- the same two integers naming two different
            # intervals of time, which is the entire defect being repaired.
            # It was skipped, and only the end gate's rate-pair check caught it.
            #
            # So the test is now what the clip was cut *from*, not which fields
            # its meta happens to carry: same source bytes and same span is a
            # skip, anything else is a re-cut.  A meta with no ``source_key`` --
            # every clip cut before today -- cannot be shown to match and is
            # therefore re-cut rather than trusted.  That is only ever reached
            # for uploads a run was deliberately launched over, since the outer
            # resume gate skips recorded uploads entirely unless --redo names
            # them.
            same_source = bool(source_identity) and previous.get("source_key") == source_identity
            same_span = previous.get("source_frame_span") == [int(start), int(end)]
            if same_source and same_span:
                record["clips"].append({"clip": name, "status": "exists"})
                continue
            record["clips"].append({
                "clip": name,
                "status": "recut" if same_span else "recut_span_moved",
                "reason": ("source bytes differ" if not same_source
                           else "span moved"),
                "was": previous.get("source_frame_span"), "now": [int(start), int(end)],
                "was_source_key": previous.get("source_key"),
                "now_source_key": source_identity})
        out_dir.mkdir(parents=True, exist_ok=True)
        written = cut_clip(upload, start, end, out_dir / "clip.mp4",
                           source_fps=source_fps)
        if written is None:
            record["clips"].append({"clip": name, "status": "cut_failed"})
            continue
        extract_audio(out_dir / "clip.mp4", out_dir / "audio.wav")

        # Pose only the frames this clip keeps, on the box already chosen for
        # them.  The clip may come back a frame or two off what was asked for;
        # the track is matched to what the file actually holds rather than
        # assumed to line up, because every crop after a misalignment is wrong.
        #
        # Both directions matter and only one is obvious.  Fewer frames than
        # asked: trim.  *More*: the arrays would be a row short of the video,
        # which is worse than it sounds -- GVHMR crops frame-by-frame from
        # bbx.pt and a short track misaligns or kills the whole clip.  Measured
        # over 1,200 produced clips this never happened, so the padding is a
        # guard rather than a fix; it holds the last box, which is what
        # save_gvhmr_bbx does for gaps anyway.
        # The clip is a *resampling* of [start, end) onto 30 fps, so clip frame
        # k holds source frame start + round(k * source_fps / 30).  Slicing the
        # track instead -- which is what this did while every upload was assumed
        # to be 30 fps -- pairs clip frame k with source frame start + k, and on
        # a 60 fps upload that is a box from half the elapsed time away.
        picks = np.clip(
            (start + np.arange(written) * source_fps / FPS).round().astype(int),
            0, len(box_track) - 1)
        span_boxes = box_track[picks]
        if len(span_boxes) < written:
            pad = np.repeat(span_boxes[-1:], written - len(span_boxes), axis=0)
            span_boxes = np.concatenate([span_boxes, pad])
            record.setdefault("frame_count_adjustments", []).append(
                {"clip": name, "asked": int(end - start), "written": int(written)})
        keypoints = np.full((len(span_boxes), 18, 2), np.nan, dtype=np.float32)
        scores = np.zeros((len(span_boxes), 18), dtype=np.float32)
        for position, frame in extractor._reread(out_dir / "clip.mp4",
                                                 range(len(span_boxes))):
            box = span_boxes[position]
            if not np.isfinite(box).all():
                continue
            body, body_scores = extractor.pose(frame, box)
            visible = body_scores > extractor.score_threshold
            body[~visible] = np.nan
            if visible.sum() < 3:
                body[:] = np.nan
            keypoints[position] = body
            scores[position] = body_scores

        # Every array is exactly as long as the video the clip actually holds.
        counts = [len(detections[i]) if i < len(detections) else 0 for i in picks]
        counts = counts[:len(span_boxes)]
        counts += [0] * (len(span_boxes) - len(counts))
        pose = VideoPose(
            keypoints=keypoints, scores=scores, dancer_box=span_boxes,
            person_count=np.asarray(counts, dtype=np.int32),
            frame_indices=np.arange(len(span_boxes), dtype=np.int32),
            width=width, height=height, fps=float(FPS),
            track_score=round(float(best_score), 6),
            rival_ratio=round(float(rival_ratio), 4),
            rival_count=int(rival_count))
        save(pose, out_dir)
        if write_bbx and np.isfinite(span_boxes).all(axis=1).any():
            save_gvhmr_bbx(pose, out_dir / "preprocess" / "bbx.pt")

        meta = {
            "num_frames": int(written),
            "video_w": width, "video_h": height, "fps": float(FPS),
            # The clip's rate is FPS by construction; the upload's is measured,
            # and the two were conflated until 2026-08-19.  Recording only the
            # first is what made a 2x stretch invisible for 1,418 clips.
            "source_fps": float(source_fps),
            # Both of the container's rate claims, not just the one the cut
            # used.  A clip cut from a file where these disagree is the
            # picture-and-sound-from-different-spans defect, and recording only
            # the rate that was used leaves nothing to detect it with later.
            "source_avg_frame_rate": float(average_fps),
            "source_r_frame_rate": float(container_fps),
            # Set when this clip came from a constant-rate re-encode rather than
            # the upload itself; names the original so the chain stays readable.
            "cfr_normalized_from": normalized_from,
            "source": str(upload),
            # ``<bytes>:<sha256 of the first megabyte>`` of that file.  The
            # skip test above compares this rather than asking which fields the
            # meta carries, so a re-encode of the same upload under the same
            # name cannot be mistaken for the same clip.
            "source_key": source_identity,
            "source_frame_span": [int(start), int(end)],
            "cache_version": "atomicdance-wild-ingest-v1",
            # Why this clip ends where it does, taken from the span builder
            # rather than reconstructed: a boundary made by the length division
            # is indistinguishable afterwards from a dancer walking out.
            "span_end_reason": end_reason,
            "dancer_present_fraction": round(
                float(np.isfinite(span_boxes).all(axis=1).mean()), 4),
            "rival_ratio": pose.rival_ratio,
            "rival_count": pose.rival_count,
            "detector": "DWPose yolox_l + dw-ll_ucoco_384 (third_party/DWPose)",
            "max_seconds": max_seconds,
            "min_frames": min_frames,
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        record["clips"].append({"clip": name, "status": "ok", "frames": int(written),
                                "end_reason": meta["span_end_reason"]})
    return record


def calibrate_cuts(uploads: Sequence[pathlib.Path], extractor: DWPoseExtractor,
                   *, samples: int = 12) -> Dict:
    """Report the frame-difference distribution, so the cut threshold has a source.

    A shot change is a large, isolated spike in whole-frame difference; dancer
    motion is a broad hump.  Printing the quantiles lets the threshold be placed
    above the hump rather than at a number somebody liked.
    """
    values: List[np.ndarray] = []
    for upload in uploads[:samples]:
        _, scores = scan_upload(extractor, upload)
        if len(scores) > 1:
            values.append(scores[1:])
    if not values:
        return {}
    pooled = np.concatenate(values)
    report = {"frames": int(len(pooled)),
              "quantiles": {str(q): round(float(np.percentile(pooled, q)), 5)
                            for q in (50, 90, 99, 99.5, 99.9)},
              "max": round(float(pooled.max()), 5)}
    print(json.dumps(report, indent=1))
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos-dir", type=pathlib.Path, required=True)
    parser.add_argument("--out-root", type=pathlib.Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="upper bound on clip length; no default on purpose -- "
                             "it is the output of tools/scan_clip_length.py")
    parser.add_argument("--min-frames", type=int, default=MIN_FRAMES,
                        help="Alg. 1 needs {} clusters of {} frames".format(
                            MIN_CLUSTERS, FRAMES_PER_CLUSTER))
    parser.add_argument("--shot-floor", type=float, default=SHOT_CUT_FLOOR)
    parser.add_argument("--shot-ratio", type=float, default=SHOT_CUT_RATIO)
    parser.add_argument("--max-absence", type=int, default=MAX_ABSENCE_FRAMES)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--calibrate-cuts", action="store_true",
                        help="report the frame-difference distribution and exit")
    parser.add_argument("--scan-cache", type=pathlib.Path, default=None,
                        help="cache per-upload detections here; the scan is the "
                             "expensive half and does not depend on --max-seconds, "
                             "so it can run before that constant is settled")
    parser.add_argument("--cfr-cache", type=pathlib.Path, default=None,
                        help="re-encode uploads whose avg_frame_rate and "
                             "r_frame_rate disagree onto a constant rate here, "
                             "then cut from that. Without it such uploads are "
                             "REFUSED (status variable_frame_rate) rather than "
                             "cut, because the picture is selected by frame "
                             "number and the sound by seconds, and on those "
                             "files the two name different spans")
    parser.add_argument("--scan-only", action="store_true",
                        help="fill the scan cache and stop, cutting nothing")
    parser.add_argument("--manifest", type=pathlib.Path, default=None)
    parser.add_argument("--exclude-uploads", type=pathlib.Path, default=None,
                        help="JSON with an 'excluded_upload_ids' list; those "
                             "uploads are dropped before --limit and --shard. "
                             "See runs/wild_upload_exclusions.json.")
    parser.add_argument("--redo", type=pathlib.Path, default=None,
                        help="file of upload stems, one per line, to ingest "
                             "again although the manifests already record "
                             "them; the later row supersedes the earlier one "
                             "for every reader that goes through manifest_rows")
    args = parser.parse_args(argv)

    import hashlib

    # Hash order, then limit, then shard -- in that order.  Limiting after
    # sharding would give each of N shards its own N uploads, so a 200-upload
    # pilot would quietly become 1200; and taking the first N by *name* would
    # sample one week of one account, because the filenames are TikTok ids.
    uploads = sorted(args.videos_dir.glob("*.mp4"),
                     key=lambda p: hashlib.sha256(p.stem.encode()).hexdigest())
    # Excluded uploads are dropped BEFORE the limit and the shard, so the list a
    # shard receives does not change when the exclusion list grows -- and so an
    # exclusion cannot be silently spent as one of --limit's N.  The exclusion
    # is a property of the footage, not of this run: on 2026-09-02 sixteen
    # AI-generated illustration videos sat under a choreographer's OSS prefix
    # and had already been cut, 2D-tracked and queued for 3D before anybody
    # looked at them.  Nothing measured them as wrong; an operator recognised
    # them in a contact sheet.  See runs/wild_upload_exclusions.json.
    if args.exclude_uploads is not None:
        payload = json.loads(args.exclude_uploads.read_text(encoding="utf-8"))
        excluded = {str(u) for u in payload.get("excluded_upload_ids", [])}
        if not excluded:
            parser.error("{} carries no excluded_upload_ids; refusing to run with an "
                         "exclusion list that excludes nothing".format(args.exclude_uploads))
        before = len(uploads)
        uploads = [u for u in uploads if u.stem not in excluded]
        print("exclusions: {} of {} uploads dropped by {} ({} ids on the list)".format(
            before - len(uploads), before, args.exclude_uploads, len(excluded)), flush=True)
    if args.limit:
        uploads = uploads[:args.limit]
    uploads = [u for u in uploads
               if int(hashlib.sha256(u.stem.encode()).hexdigest(), 16) % args.num_shards
               == args.shard]

    extractor = DWPoseExtractor(device=args.device)
    if args.calibrate_cuts:
        calibrate_cuts(uploads, extractor)
        return 0
    if args.scan_only:
        if args.scan_cache is None:
            parser.error("--scan-only needs --scan-cache to put the results in")
        for index, upload in enumerate(uploads, start=1):
            try:
                detections, _ = scan_upload(extractor, upload, args.scan_cache)
                print("[{}/{}] {} {} frames".format(index, len(uploads), upload.stem,
                                                    len(detections)), flush=True)
            except Exception as error:                    # noqa: BLE001
                print("[{}/{}] {} FAILED {!r}".format(index, len(uploads),
                                                      upload.stem, error), flush=True)
        return 0
    if args.max_seconds is None:
        parser.error("--max-seconds is required; run tools/scan_clip_length.py first")

    manifest = args.manifest or (args.out_root / "ingest_shard{}.jsonl".format(args.shard))
    manifest.parent.mkdir(parents=True, exist_ok=True)

    # Resume across *any* sharding.  The manifests are append-only and shard
    # membership changes with --num-shards, so a relaunch at a different width
    # would otherwise write a second row for uploads already done and the
    # accounting -- retained seconds, end reasons, upload counts -- would double
    # them.  Reading every manifest in the root makes the skip independent of
    # how the work was divided.
    already = set(manifest_rows(args.out_root))
    redo = set()
    if args.redo:
        text = args.redo.read_text(encoding="utf-8")
        redo = {line.strip() for line in text.splitlines() if line.strip()}
        # Named uploads are re-ingested although they are recorded.  Without
        # this the resume gate is absolute, and a re-cut of an already-ingested
        # corpus skips every upload it was launched for while printing the
        # lines a correct resume prints -- which is what happened on
        # 2026-08-19: seven shards, "1335 of 1335 already recorded", zero work,
        # exit 0, and a census afterwards that found no orphans because nothing
        # had changed.
        missing = redo - already
        print("redo: {} upload(s) named, {} of them not previously recorded"
              .format(len(redo), len(missing)), flush=True)
        already -= redo
    if already:
        before = len(uploads)
        uploads = [u for u in uploads if u.stem not in already]
        print("resuming: {} of this shard's {} uploads already recorded".format(
            before - len(uploads), before), flush=True)

    with manifest.open("a", encoding="utf-8") as handle:
        for index, upload in enumerate(uploads, start=1):
            # One upload the decoder chokes on must not take the shard with it.
            # It did once: a 201-second file that ffprobe reads and cv2 will not
            # open raised out of the loop and ended a shard of 850 uploads at
            # 158.  Every other sweep in this repo logs the failure and carries
            # on; this one now does too, and the manifest carries the reason so
            # the loss is counted rather than inferred from a short log.
            try:
                record = ingest(upload, args.out_root, extractor=extractor,
                                max_seconds=args.max_seconds, min_frames=args.min_frames,
                                shot_floor=args.shot_floor, shot_ratio=args.shot_ratio,
                                max_absence=args.max_absence, cache_dir=args.scan_cache,
                                cfr_cache=args.cfr_cache)
            except Exception as error:                       # noqa: BLE001
                record = {"upload": upload.stem, "source": str(upload),
                          "status": "failed", "error": "{}: {}".format(
                              type(error).__name__, error), "spans": [], "clips": []}
            record["ingested_at"] = time.time()
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print("[{}/{}] {} -> {} clips ({})".format(
                index, len(uploads), upload.stem, len(record["clips"]),
                record["status"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
