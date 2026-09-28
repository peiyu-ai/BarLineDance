#!/usr/bin/env python3
"""Retrieve the full song behind a short clip, and locate the clip inside it.

The wild clips are 12-24 s cuts of Douyin videos, and the video itself is usually
no longer than the clip (16 of the 20 eval uploads end within 4 s of the clip; the
other 4 are two-clip uploads, 25-44 s in total), so the rest of the song does not
exist anywhere in this repo.  A dance
*continuation* needs it.  This tool goes and gets it, using public services only:

    clip audio --Shazam, 10 s windows--> track identity (+ offset in its reference)
               --YouTube Music search--> top-K candidate uploads
               --yt-dlp bestaudio-----> candidate full-length audio
               --chroma alignment-----> which candidate IS this recording, and where
                                        the clip sits in it (offset, speed ratio)

It reads audio only.  No motion of the target clip is touched, so the output is a
legal inference-time input under CLAUDE.md 1.6 ("does it only look at the music").

Why every stage exists
----------------------
* **Shazam, per 10 s window, voted.**  ``shazamio`` fingerprints only the middle
  10 s of whatever it is given, so the clip is cut into windows (hop 4 s, plus one
  ending at the clip's end, see ``shazam_starts``) and the track keys are voted.  One Douyin upload switched songs mid-video
  (7676475940934935409: window 0 is a different track than windows 8-12), which is
  why the input is the *clip's* audio, not the upload's.
* **Top-K candidates, not the first hit.**  Artist names come back in two
  scripts (Shazam "LAY" is YouTube Music "张艺兴", "Qi Wang" is "王琪") and Douyin
  variants ("R&B版", "伴奏", "DJ版") are re-uploaded under other names, so name
  matching cannot pick the file.  The audio decides: 3 of the 13 verified eval
  clips were matched by the rank-1 result, not rank 0.
* **Alignment is the verdict, the name is not.**  Model: ``clip(t) = song(offset
  + rate * t)``; Douyin speed-ups resample (tempo and pitch move together), so the
  clip is resampled by ``rate`` before comparing.  Feature: chroma, z-scored per
  frame, so the score is the mean per-frame Pearson correlation across the 12
  pitch classes -- an unrelated position reads ~0, the same recording ~1.

The threshold is calibrated, and the calibration can fail
---------------------------------------------------------
Measured on the 20 T-line eval clips (2026-09-22), every clip searched against
every *other* clip's candidate songs with the same rate grid:

    wrong song      n=672   median 0.202   p99 0.354   max 0.429
    cover version   same composition, other singer: 0.446, 0.522  (月半弯 clip)
    same recording  n=13    min 0.752      median 0.961

``--min-score 0.60`` sits between the covers and the verified matches, so it
rejects both a wrong song and a cover of the right one.  Two independent checks
ride in the output instead of being folded into the score:

* **Shazam offset agreement.**  Each window's Shazam offset predicts where the
  clip sits in Shazam's reference.  On 9 of 13 verified clips the aligner agreed
  within 0.02 s, and on one more within 0.08 s (single window).  Two differ by a
  *constant* (0.43 s: a different master with a longer lead-in; 188.27 s: Shazam's
  reference is a Douyin cut that starts at the hook).  One drifts, -0.13 -> -0.17 s:
  the download there is a 1.8% slower version (rate 0.982), so its time axis is not
  Shazam's.  Constant = same alignment; a *spread* across windows = disagreement.
* **Second peak.**  A chorus repeats, so the clip can match two places.  When the
  runner-up is within 0.10 of the best (Justin Bieber "Intentions": 0.988 vs
  0.951), the offset is reported as ambiguous.

When only part of the clip is the song
--------------------------------------
A clip can hold a song switch or a voice-over, so the whole-clip score is diluted
while the part that IS the song still aligns.  When no candidate passes the
whole-clip gate, 5 s windows (hop 2.5 s) are aligned on their own, each keeping
its best 3 places; a candidate passes as ``verified_by_windows`` when at least
``--min-windows-agree`` (2) windows put the clip at one place (+-0.1 s) and one
speed, each over ``--min-window-score``.  Calibrated on the same 13 verified clips:

    window vs          n      median   max
    wrong song         3600   0.339    0.610   -> >=2 agreeing: 0/585 pairs at any gate >= 0.6
    cover (月半弯)      14     0.510    0.689   -> >=2 agreeing: 1/2 at 0.6, 0/2 at >= 0.7
    right recording    80     0.959    (min 0.678; 87.5% >= 0.8) -> 13/13 at 0.75, 12/13 at 0.8

The gate is 0.80, not the whole-clip 0.60: at 0.60 a cover passes, and that is
exactly what a refused download produces -- the right recording missing from
the candidates, its covers present.  The one clip 0.80 loses on windows passes
the whole-clip gate anyway.  If two songs qualify, the one whose agreeing windows
include the clip's LAST window wins, because a continuation follows the music
that is playing when the clip ends; a match that does not reach the end is
printed as such and carries ``covers_clip_end: false``.  Real-audio check
(2026-09-22): 8 s of 月半弯 + 10 s of "How Long" (and the reverse), whole-clip
0.555 / 0.421, each resolved to the song at the end at the right second.

Known blind spot: chroma sees harmony.  A percussive track with little harmonic
content can be identified by Shazam and still score at noise level here (2NE1
"I Am the Best": 0.21 at Shazam's own offset; a 64-band log-mel variant did not
see it either).  Such a clip comes out ``unverified`` -- the tool refuses rather
than guessing.

Storage (CLAUDE.md 1.1/1.2)
---------------------------
Songs land under ``/cache`` (local CPFS, outside the NAS quota) and are never
pushed to OSS by this tool: they are third-party recordings, and an OSS upload
cannot be deleted with the current credentials.

Dependencies are not in requirements.txt.  They live in their own directory,
``$FULL_SONG_PYLIB`` (default ``/cache/atomicdance-pylib/retrieve_full_song``),
which this tool puts on ``sys.path`` -- and on the ``PYTHONPATH`` of the yt-dlp
subprocess, which would not inherit a ``sys.path`` edit.  Everything is installed
``--no-deps``: shazamio pins numpy>=2.2 and would otherwise replace the numpy 1.26
that librosa here is built against (it runs fine on 1.26), and the resolver hangs
on the NGC extra index in the machine's pip.conf unless ``--isolated``::

    pip install --isolated --no-deps --target /cache/atomicdance-pylib/retrieve_full_song \\
        -i https://pypi.org/simple \\
        yt-dlp ytmusicapi shazamio shazamio-core pydub "dataclass-factory<3" aiohttp-retry

Outputs, in ``--out-dir`` (default ``/cache/atomicdance-assets/scratch/full_songs/<key>``)::

    result.json                 identity, candidates, offset/rate, every check
    songs/<videoId>.{webm,wav}  each downloaded candidate (original + 22.05 kHz mono)
    full_song_clip_speed.wav    the verified song re-timed to the clip's speed (rate
                                applied), 44.1 kHz stereo; the clip is [clip_start_s,
                                clip_end_s] of it, see result.json "timeline"
    continuation.wav            the same file from clip_end_s to the end
    music_35_full.npy           (--music35) 35-D features of full_song_clip_speed

and all of it -- every file under songs/ flat, plus the outputs above -- is copied
to ``--export-dir`` (default ``output/sample_<today>_fullsong/<key>``) after a
real write probe of that folder's quota.

Usage::

    python3 tools/retrieve_full_song.py --clip wild_v5:7618203431723357818:clip000 --music35 \\
        --listen output/sample_20260922_fullsong/7618203431723357818__clip000.mp3
    python3 tools/retrieve_full_song.py --audio path/to/any_video_or_audio.mp4

Exit code 0 = verified, 2 = unidentified / unverified (result.json says which).
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
# T2 (2026-09-22) is a superset of T1: it also resolves the 38 sequences added in T2.
DEFAULT_SOURCES = REPO / "data/wild3d/txy_t2_performance/sources.jsonl"
DEFAULT_OUT_ROOT = Path("/cache/atomicdance-assets/scratch/full_songs")
PYLIB = os.environ.get("FULL_SONG_PYLIB", "/cache/atomicdance-pylib/retrieve_full_song")
if os.path.isdir(PYLIB) and PYLIB not in sys.path:
    sys.path.append(PYLIB)  # appended: the system numpy/librosa must win over anything in it

SR, HOP, NFFT = 22050, 512, 4096
FPS = SR / HOP
WIN, WIN_HOP = 10.0, 4.0          # Shazam windows
WIN_S, WIN_S_HOP = 5.0, 2.5         # alignment windows (fallback when the whole clip fails)
MIN_SCORE, MIN_WINDOW_SCORE = 0.60, 0.80   # calibration in the module docstring
RATES = np.round(np.arange(0.80, 1.2501, 0.01), 3)


# --------------------------------------------------------------------------- identify
def shazam_starts(dur: float) -> list[float]:
    """Window starts: every WIN_HOP, plus one window that ends at the clip's end.

    The hop grid alone stops up to WIN_HOP short of the end, and a 12.9 s clip got
    ONE window, [0, 10] -- its last 2.9 s were never identified.  That is where the
    song switch of 7676475940934935409 sits (clip000 = 0-12.9 s of the upload: Shazam
    heard only its first song, while the fingerprint graph links clip000's audio to
    clip001's 2NE1), and it is the music a continuation has to follow.
    """
    if dur <= WIN:
        return [0.0]
    starts = [round(i * WIN_HOP, 1) for i in range(int((dur - WIN) // WIN_HOP) + 1)]
    if dur - WIN - starts[-1] > 0.5:  # more than half a second left uncovered
        starts.append(round(dur - WIN, 2))
    return starts


async def _shazam_windows(audio: Path) -> list[dict]:
    from shazamio import Shazam

    dur = _duration(audio)
    starts = shazam_starts(dur)
    shz, rows = Shazam(), []
    with tempfile.TemporaryDirectory() as td:
        for t0 in starts:
            w = os.path.join(td, "w.wav")
            _ffmpeg(["-ss", str(t0), "-t", str(WIN), "-i", str(audio), "-ac", "1", "-ar", "16000", w])
            out, err = {}, None
            for _ in range(3):
                try:
                    out = await shz.recognize(w)
                    err = None
                    break
                except Exception as exc:  # network hiccups: retry, then record
                    err = repr(exc)
                    await asyncio.sleep(3)
            track = out.get("track") or {}
            match = (out.get("matches") or [{}])[0]
            rows.append({"t0": t0, "key": track.get("key"), "title": track.get("title"),
                         "artist": track.get("subtitle"), "isrc": track.get("isrc"),
                         "offset": match.get("offset"), "timeskew": match.get("timeskew"),
                         "frequencyskew": match.get("frequencyskew"), "error": err})
            await asyncio.sleep(1.0)
    return rows


def identify(audio: Path) -> tuple[list[dict], list[dict]]:
    windows = asyncio.run(_shazam_windows(audio))
    votes = collections.Counter(w["key"] for w in windows if w["key"])
    tracks = []
    for key, n in votes.most_common():
        w = next(x for x in windows if x["key"] == key)
        tracks.append({"shazam_key": key, "votes": n, "n_windows": len(windows),
                       "artist": w["artist"], "title": w["title"], "isrc": w["isrc"]})
    return tracks, windows


# --------------------------------------------------------------------------- fetch
def search_candidates(tracks: list[dict], top_k: int, max_song_s: float) -> list[dict]:
    from ytmusicapi import YTMusic

    yt, out, seen = YTMusic(), [], set()
    for t in tracks:
        query = "{} {}".format(t["artist"] or "", t["title"] or "").strip()
        for rank, r in enumerate(yt.search(query, filter="songs", limit=top_k)[:top_k]):
            vid = r.get("videoId")
            if not vid or vid in seen:
                continue
            # a "DJ" query once returned a 32-minute non-stop mix; that is not a song
            if (r.get("duration_seconds") or 0) > max_song_s:
                continue
            seen.add(vid)
            out.append({"shazam_key": t["shazam_key"], "query": query, "rank": rank, "videoId": vid,
                        "url": "https://music.youtube.com/watch?v=" + vid, "yt_title": r.get("title"),
                        "yt_artists": ",".join(a["name"] for a in r.get("artists") or []),
                        "listed_duration_s": r.get("duration_seconds")})
    return out


def download(video_id: str, song_dir: Path, attempts: int = 4) -> tuple[Path | None, int]:
    """bestaudio via yt-dlp; returns (22.05 kHz mono wav or None, attempts used).

    Retried because YouTube intermittently answers the media request with 403:
    without a JS runtime yt-dlp can only use one player client here (node 16 is
    too old for its challenge solver), and on 2026-09-22 roughly 1 download in 10
    was refused -- once it was the only candidate that was the right recording,
    and the run came out "unverified" on the covers.  An immediate retry of the
    same video succeeded, so failures are retried with a pause, alternating host.

    ``song_dir`` may be shared by parallel runs (``--song-dir``): a per-video lock
    keeps one run from deleting another's partial download as "stale", and the wav
    is renamed into place so a crash cannot leave a truncated one that later runs
    would take as finished.  The locks live in a subfolder so the ``<id>.*`` globs
    below (and main's pick of the original file) never see them.
    """
    import fcntl

    song_dir.mkdir(parents=True, exist_ok=True)
    (song_dir / ".locks").mkdir(exist_ok=True)
    with open(song_dir / ".locks" / video_id, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _download_locked(video_id, song_dir, attempts)


def _download_locked(video_id: str, song_dir: Path, attempts: int) -> tuple[Path | None, int]:
    import time

    wav = song_dir / (video_id + ".wav")
    if wav.is_file():
        return wav, 0
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (env.get("PYTHONPATH"), PYLIB) if p)
    err = ""
    for attempt in range(1, attempts + 1):
        for stale in song_dir.glob(video_id + ".*"):  # partials from a refused attempt
            if stale.suffix != ".wav":
                stale.unlink()
        host = "music.youtube.com" if attempt % 2 else "www.youtube.com"
        proc = subprocess.run([sys.executable, "-m", "yt_dlp", "-q", "--no-warnings", "--no-playlist",
                               "-f", "bestaudio", "-o", str(song_dir / (video_id + ".%(ext)s")),
                               "https://{}/watch?v={}".format(host, video_id)],
                              capture_output=True, text=True, timeout=600, env=env)
        src = [p for p in song_dir.glob(video_id + ".*") if p.suffix not in (".wav", ".part")]
        if proc.returncode == 0 and src:
            tmp = song_dir / (video_id + ".wav.partial")
            _ffmpeg(["-i", str(src[0]), "-ac", "1", "-ar", str(SR), "-f", "wav", str(tmp)])
            os.replace(tmp, wav)
            return wav, attempt
        err = (proc.stderr.strip().splitlines() or ["?"])[-1]
        print("      download {} attempt {}/{} failed: {}".format(video_id, attempt, attempts, err[-160:]))
        time.sleep(5 * attempt)
    return None, attempts


def title_queries(title: str) -> list[str]:
    """A title as Shazam gives it, and without its bracketed version tag: "如果的事 (rnb版)" is also
    searched as "如果的事", because the Douyin edit is often filed under another tag or none.
    Only the title, not "artist title": an artist query answers with the artist's OTHER songs
    when the store lacks this one (不逢海 -> 零落摇, 真心错付, ...)."""
    title = (title or "").strip()
    base = re.sub(r"\s*[\(（\[【].*?[\)）\]】]\s*", " ", title).strip()
    return [q for q in dict.fromkeys([title, base]) if q]


def store_candidates(queries: list[str], stores: list[str], top_k: int, max_song_s: float) -> list[dict]:
    """Candidates named by something other than Shazam: a track name, or a Douyin post.

    Shazam only knows released masters.  A choreographer's own edit is named nowhere but
    the post that carries it (7650126416710192357: Shazam 0/4 windows on the clip AND on
    the clean 24 s sound; the description says 夏天的风), and it is published on the
    Chinese stores rather than YouTube Music.  These candidates go through exactly the
    same alignment gate as the Shazam ones -- the name is a hint, the audio decides.
    """
    sys.path.insert(0, str(REPO / "tools"))
    import song_stores

    out = []
    for rank, hit in enumerate(song_stores.search(queries, stores, limit=top_k, max_song_s=max_song_s)):
        out.append({"shazam_key": None, "query": hit["query"], "rank": rank, "store": hit["store"],
                    "store_id": hit["id"], "videoId": "{}:{}".format(hit["store"], hit["id"]),
                    "url": hit["url"], "yt_title": hit["name"], "yt_artists": hit["artist"],
                    "listed_duration_s": hit["duration_s"]})
    return out


def fetch_candidate(candidate: dict, song_dir: Path) -> tuple[Path | None, int]:
    """Download one candidate from whichever store named it, as 22.05 kHz mono wav."""
    if candidate.get("store") in ("netease", "kuwo", "qishui"):
        import fcntl

        sys.path.insert(0, str(REPO / "tools"))
        import song_stores

        stem = candidate["videoId"].replace(":", "_")
        (song_dir / ".locks").mkdir(parents=True, exist_ok=True)
        with open(song_dir / ".locks" / stem, "w") as lock:  # parallel runs share --song-dir
            fcntl.flock(lock, fcntl.LOCK_EX)
            src = song_stores.download({"store": candidate["store"], "id": candidate["store_id"]}, song_dir)
            if src is None:
                return None, 1
            wav = song_dir / (stem + ".wav")
            if not wav.is_file():
                tmp = song_dir / (stem + ".wav.partial")
                _ffmpeg(["-i", str(src), "-ac", "1", "-ar", str(SR), "-f", "wav", str(tmp)])
                os.replace(tmp, wav)
        return wav, 1
    return download(candidate["videoId"], song_dir)


# --------------------------------------------------------------------------- align
def _chroma_z(y, tuning=None):
    import librosa

    c = librosa.feature.chroma_stft(y=y, sr=SR, n_fft=NFFT, hop_length=HOP, tuning=tuning)
    c = c - c.mean(0, keepdims=True)
    return c / (np.linalg.norm(c, axis=0, keepdims=True) + 1e-8)


class _Clip:
    def __init__(self, y):
        self.y, self._feats = y, {}

    def feats(self, rate):
        import librosa

        rate = round(float(rate), 4)
        if rate not in self._feats:
            y = self.y if rate == 1.0 else librosa.resample(self.y, orig_sr=SR, target_sr=SR * rate,
                                                           res_type="soxr_hq")
            self._feats[rate] = _chroma_z(y)
        return self._feats[rate]


def _score_curve(cz, song_f, n_song, nfft):
    """Mean per-frame correlation at every valid offset; one irfft per call."""
    n = cz.shape[1]
    if n_song < n:
        return np.full(1, -1.0)
    spec = np.fft.rfft(cz[:, ::-1], n=nfft, axis=1)
    return np.fft.irfft((song_f * spec).sum(0), n=nfft)[n - 1:n_song] / n


def align(clip: _Clip, song_y, window_floor: float | None = None) -> dict:
    sz = _chroma_z(song_y)
    n_song = sz.shape[1]
    # linear (not circular) correlation needs nfft >= n_song + n_clip - 1 for every
    # rate tried, and the fine search can step 0.01 past the coarse grid's top
    nfft = 1 << int(np.ceil(np.log2(n_song + int(clip.feats(1.0).shape[1] * (RATES.max() + 0.05)) + 1)))
    song_f = np.fft.rfft(sz, n=nfft, axis=1)

    def best_over(rates):
        best = (-9.0, None, None)
        for r in rates:
            c = _score_curve(clip.feats(r), song_f, n_song, nfft)
            i = int(c.argmax())
            if c[i] > best[0]:
                best = (float(c[i]), float(r), i)
        return best

    score, rate, _ = best_over(RATES)
    score, rate, idx = best_over(np.round(np.arange(rate - 0.01, rate + 0.0101, 0.001), 4))
    curve = _score_curve(clip.feats(rate), song_f, n_song, nfft)
    guard = int(2.0 * FPS)  # a second peak must be more than 2 s away
    mask = np.ones_like(curve, bool)
    mask[max(0, idx - guard):idx + guard + 1] = False
    j = int(np.where(mask, curve, -9).argmax()) if mask.any() else -1
    return {"score": score, "rate": rate, "offset_s": idx / FPS,
            "second_score": float(curve[j]) if j >= 0 else float("nan"),
            "second_offset_s": j / FPS if j >= 0 else float("nan"),
            "song_duration_s": len(song_y) / SR,
            # the window scan is most of the cost; below ``window_floor`` the whole-clip score says
            # "wrong song" (wrong songs: median 0.20, p99 0.35), while a song holding even a fifth of
            # a longer input still reads ~0.4 -- so the partial-match fallback loses nothing it needs
            "windows": [] if window_floor is not None and score < window_floor
            else window_scan(clip, song_f, n_song, nfft)}


def _top_peaks(curve, k=3, guard=int(2.0 * FPS)):
    c, out = curve.copy(), []
    for _ in range(k):
        i = int(c.argmax())
        if c[i] <= -1.0:
            break
        out.append((float(c[i]), i))
        c[max(0, i - guard):i + guard + 1] = -9.0
    return out


def window_scan(clip: _Clip, song_f, n_song, nfft, keep=3) -> list[dict]:
    """The best few places for each WIN_S window of the clip on its own.

    Each peak implies where the *clip* starts in the song (``peak - rate * t0``).
    A window keeps its top ``keep`` places, more than 2 s apart: a hook that the
    song also plays in its intro gives one window two equally good places (818's
    window at 2.5 s scores 0.997 on the song's first seconds), and only one of
    them is where the rest of the clip is.  Reuses the clip features the
    whole-clip search cached, so it costs one irfft per (rate, window).
    """
    dur = len(clip.y) / SR
    starts = np.arange(0.0, max(dur - WIN_S, 0.0) + 1e-6, WIN_S_HOP)
    found = [[] for _ in starts]
    for r in RATES:
        feats = clip.feats(r)
        for k, t0 in enumerate(starts):
            a, b = int(t0 * r * FPS), int(min(t0 + WIN_S, dur) * r * FPS)
            for score, i in _top_peaks(_score_curve(feats[:, a:b], song_f, n_song, nfft)):
                found[k].append((score, float(r), i / FPS - t0 * float(r)))
    out = []
    for t0, peaks in zip(starts, found):
        kept = []
        for score, r, st in sorted(peaks, reverse=True):  # best first; drop near-duplicates
            if all(abs(st - q["implied_clip_start_s"]) > 2.0 for q in kept):
                kept.append({"score": score, "rate": r, "implied_clip_start_s": st})
            if len(kept) == keep:
                break
        out.append(dict(kept[0], t0=float(t0), peaks=kept))
    return out


def window_agreement(windows: list[dict], min_score: float, tol_s: float = 0.10) -> dict:
    """Largest set of windows with a passing peak that puts the clip at one place, one speed."""
    votes = [(w["t0"], p) for w in windows for p in w.get("peaks", [w]) if p["score"] >= min_score]
    best, best_key = None, (0, 0.0)
    for _, anchor in votes:
        group = {}
        for t0, p in votes:
            if abs(p["implied_clip_start_s"] - anchor["implied_clip_start_s"]) <= tol_s \
                    and abs(p["rate"] - anchor["rate"]) <= 0.011:
                if t0 not in group or p["score"] > group[t0]["score"]:
                    group[t0] = p
        key = (len(group), sum(p["score"] for p in group.values()))
        if key > best_key:
            best, best_key = group, key
    if not best:
        return {"n_agree": 0, "covers_clip_end": False}
    last_t0 = max(w["t0"] for w in windows)
    return {"n_agree": len(best), "t0s": sorted(best),
            "clip_start_s": float(np.median([p["implied_clip_start_s"] for p in best.values()])),
            "rate": float(np.median([p["rate"] for p in best.values()])),
            "mean_score": float(np.mean([p["score"] for p in best.values()])),
            "covers_clip_end": last_t0 in best}


# --------------------------------------------------------------------------- music_35
def music35_check(song_path: Path, clip_start_s: float, clip_id, sources):
    """Rebuild the clip from the song in the model's own 35-D feature space.

    ``song_path`` is already re-timed to the clip's speed, so it runs through the
    same extractor the bundle used as-is, and the rows at the clip's position are
    compared with the clip's *released* music_35.  Control: the same song 7.3 s
    later (same production, wrong place).  Returns the features for the whole song
    so a continuation can read past the clip's last frame.
    """
    import librosa

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "tools"))
    # importing it restores scipy.signal.hann, which librosa 0.10.1's beat_track
    # still calls; the released music_35 was extracted through the same shim
    import extract_wild_music_features  # noqa: F401
    from data.audio_extraction import baseline_features as bf

    y, _ = librosa.load(str(song_path), sr=bf.SR, mono=True)
    full = bf.extract_audio(y, "x", max_frames=None).astype(np.float32)
    start = int(round(clip_start_s * bf.FPS))
    report = {"clip_start_frame": start, "frames": int(len(full)), "fps": bf.FPS}
    if not clip_id:
        return full, report
    seq_path = Path(sources).with_name("sequences.jsonl")
    seq = next((json.loads(l) for l in open(seq_path) if json.loads(l)["sequence_id"] == clip_id), None)
    if seq is None:
        return full, report
    base = seq_path.parent
    released = np.load(base / seq["music_path"])
    fids = np.load(base / seq["frame_ids_path"])

    def compare(s0):
        idx = s0 + fids
        ok = idx < len(full)
        a, b = released[ok], full[idx[ok]]
        ca, cb = a[:, 21:33], b[:, 21:33]
        cos = (ca * cb).sum(1) / (np.linalg.norm(ca, axis=1) * np.linalg.norm(cb, axis=1) + 1e-8)
        ba, bb = np.flatnonzero(a[:, 34]), np.flatnonzero(b[:, 34])
        hit = float(np.mean([np.abs(bb - i).min() <= 2 for i in ba])) if len(ba) and len(bb) else float("nan")
        return {"chroma_cens_cos_median": float(np.median(cos)),
                "envelope_pearson": float(np.corrcoef(a[:, 0], b[:, 0])[0, 1]),
                "beat_within_2_frames": hit}

    report["aligned"] = compare(start)
    report["control_shift_7.3s"] = compare(start + int(round(7.3 * bf.FPS)))
    return full, report


# --------------------------------------------------------------------------- helpers
def _ffmpeg(args):
    subprocess.run(["ffmpeg", "-v", "error", "-y"] + args, check=True)


def _duration(p: Path) -> float:
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                          "-of", "csv=p=0", str(p)]).strip())


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_clip_speed_song(original: Path, rate: float, out_dir: Path, timeline: dict) -> dict:
    """Re-time the downloaded song to the clip's speed and cut the continuation.

    ``clip(t) = song(offset + rate*t)``, so the song is played ``rate`` x faster by
    resampling (asetrate): tempo and pitch move together, as in a Douyin speed-up.
    """
    full = out_dir / "full_song_clip_speed.wav"
    chain = "aresample=44100" if rate == 1.0 else \
        "aresample=44100,asetrate={},aresample=44100".format(int(round(44100 * rate)))
    _ffmpeg(["-i", str(original), "-af", chain, "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(full)])
    cont = out_dir / "continuation.wav"
    _ffmpeg(["-ss", "{:.4f}".format(timeline["clip_end_s"]), "-i", str(full), "-c:a", "pcm_s16le", str(cont)])
    return {"full_song_clip_speed": str(full), "full_song_clip_speed_sha256": _sha256(full),
            "continuation": str(cont), "continuation_s": _duration(cont)}


def check_retimed(clip: "_Clip", path: Path, expect_start_s: float, t0: float | None = None) -> dict:
    """The re-timed file must contain the clip at speed 1.0, at the promised second.

    With ``t0`` only the window [t0, t0 + WIN_S] of the clip is looked for (at
    ``expect_start_s + t0``): a clip verified by windows is not all this song.

    Both sides use ONE tuning, the clip's.  Left to chroma_stft, the tuning is
    estimated per signal, and a whole song's estimate can sit a fraction of a bin
    away from its 20 s excerpt's: 7456795199654694202__clip000 aligned at 0.965
    (rate 1.002), and its correct re-timed file re-checked at 0.590 -- under the
    gate -- with clip -0.05 / file +0.25 bins; one shared tuning reads 0.94.  The
    re-timed file plays at the clip's pitch, so the clip's tuning is the right one.
    (align() keeps per-signal tuning: its 0.60/0.80 gates were calibrated on it.)
    """
    import librosa

    y, _ = librosa.load(str(path), sr=SR, mono=True)
    tuning = float(librosa.estimate_tuning(y=clip.y, sr=SR, n_fft=NFFT, hop_length=HOP))
    sz = _chroma_z(y, tuning)
    feats = _chroma_z(clip.y, tuning)
    if t0 is not None:
        feats = feats[:, int(t0 * FPS):int((t0 + WIN_S) * FPS)]
        expect_start_s = expect_start_s + t0
    nfft = 1 << int(np.ceil(np.log2(sz.shape[1] + feats.shape[1] + 1)))
    curve = _score_curve(feats, np.fft.rfft(sz, n=nfft, axis=1), sz.shape[1], nfft)
    if t0 is None:
        i = int(curve.argmax())  # whole clip: the promised place must be THE best place
    else:
        # one window may also match a repeat elsewhere (a hook in the intro); it only
        # has to be right where the agreeing windows put it
        e = int(round(expect_start_s * FPS))
        lo, hi = max(0, e - int(FPS)), min(len(curve), e + int(FPS) + 1)
        i = lo + int(curve[lo:hi].argmax())
    return {"score_at_rate_1": float(curve[i]), "found_start_s": i / FPS,
            "error_s": i / FPS - expect_start_s, "window_t0": t0}


def write_listen_check(clip_y, song_y, res, path: Path, tail_s=10.0):
    """Stereo file: L = clip, R = song at the aligned place, running tail_s past the clip."""
    import librosa
    import soundfile as sf

    n = len(clip_y) + int(tail_s * SR)
    a = int(res["offset_s"] * SR)
    seg = song_y[a:a + int(np.ceil(n * res["rate"])) + SR]
    if res["rate"] != 1.0:
        seg = librosa.resample(seg, orig_sr=SR, target_sr=SR / res["rate"], res_type="soxr_hq")
    seg = np.pad(seg[:n], (0, max(0, n - len(seg))))
    left = np.pad(clip_y, (0, n - len(clip_y)))
    norm = lambda x: 0.8 * x / (np.abs(x).max() + 1e-8)
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = path.with_suffix(".wav")
    sf.write(str(wav), np.stack([norm(left), norm(seg)], 1), SR)
    if path.suffix != ".wav":
        _ffmpeg(["-i", str(wav), "-b:a", "128k", str(path)])
        wav.unlink()


def export_outputs(out_dir: Path, export_dir: Path, song_files: list[Path]) -> dict:
    """Copy this run's candidate songs plus the finished outputs into one folder.

    ``song_files`` is named by the caller rather than globbed from songs/, because
    with ``--song-dir`` that folder is shared and holds other clips' songs too.
    The songs are copied flat (the verified one is ``best.videoId`` in result.json),
    so the folder people browse holds everything the run fetched.  The export
    folder is usually on the NAS, whose *quota* -- not ``df`` -- is what fills, so
    the copy is preceded by a real write probe (tools/check_disk_headroom.py).
    """
    import shutil

    sys.path.insert(0, str(REPO / "tools"))
    from check_disk_headroom import probe

    files = list(song_files)
    files += [out_dir / n for n in ("full_song_clip_speed.wav", "continuation.wav", "music_35_full.npy",
                                    "music_from_clip_start.wav", "music_35_from_clip_start.npy",
                                    "clip_audio.wav") if (out_dir / n).is_file()]
    need = sum(p.stat().st_size for p in files)
    export_dir.mkdir(parents=True, exist_ok=True)
    ok, _, msg = probe(export_dir, max(0.1, 2.0 * need / 2**30))
    if not ok:
        print("      EXPORT SKIPPED: {} cannot take {:.2f} GB ({})".format(export_dir, need / 2**30, msg))
        return {"exported": False, "export_dir": str(export_dir), "reason": msg}
    for p in files:
        shutil.copy2(p, export_dir / p.name)
    return {"exported": True, "export_dir": str(export_dir), "bytes": need,
            "files": [p.name for p in files] + ["result.json"]}


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--clip", help="recording id, e.g. wild_v5:7618203431723357818:clip000")
    src.add_argument("--audio", type=Path, help="any audio/video file holding the clip")
    ap.add_argument("--sources", default=str(DEFAULT_SOURCES), help="sources.jsonl that resolves --clip")
    ap.add_argument("--out-dir", type=Path, help="default: {}/<clip key>".format(DEFAULT_OUT_ROOT))
    ap.add_argument("--song-dir", type=Path,
                    help="where candidate songs are downloaded (default <out-dir>/songs); share one across "
                         "runs so a song many clips use is fetched once -- downloads are locked per video")
    ap.add_argument("--top-k", type=int, default=3, help="YouTube Music results downloaded per track")
    ap.add_argument("--track", action="append", default=[], metavar="NAME",
                    help="search this name in --stores as well as (or instead of) what Shazam says")
    ap.add_argument("--douyin-id", metavar="AWEME_ID",
                    help="name the song from that Douyin post (its music entry and description); "
                         "needs cookie/www_douyin_com_cookies.json and f2 on PYTHONPATH")
    ap.add_argument("--douyin-post-json", type=Path,
                    help="with --douyin-id: read the post from this file (song_stores.douyin_post output) "
                         "instead of asking Douyin")
    ap.add_argument("--stores", default="netease",
                    help="stores for --track/--douyin-id names: netease, ytmusic (comma separated)")
    ap.add_argument("--store-top-k", type=int, default=30, help="candidates per name per store")
    ap.add_argument("--shazam-to-stores", action="store_true",
                    help="also search --stores for every title Shazam names (not only YouTube Music): "
                         "the Douyin remix of a released song is on the Chinese stores")
    ap.add_argument("--lyrics", action="append", default=[], metavar="LINE",
                    help="a lyric line seen in the video; the songs whose lyrics hold it become queries")
    ap.add_argument("--download-workers", type=int, default=1,
                    help="candidates fetched ahead in parallel (1 = one after another, as before)")
    ap.add_argument("--stop-at", type=float, default=None,
                    help="stop aligning candidates once one scores this high on the whole clip (e.g. 0.90: "
                         "the calibration's wrong songs top out at 0.43); off by default")
    ap.add_argument("--window-scan-floor", type=float, default=None,
                    help="skip the per-window scan for candidates whose whole-clip score is below this "
                         "(a speed-up for many candidates; off by default)")
    ap.add_argument("--end-after-last-beat-frames", type=int, default=89, metavar="N",
                    help="with --from-clip-start --music35: cut the beatless outro to N frames after the "
                         "last beat (89 = 2.97 s = the T vocabulary's longest move; 0 keeps everything)")
    ap.add_argument("--from-clip-start", action="store_true",
                    help="also write the music from the clip's own start to the end of the song -- "
                         "what a continuation is danced to")
    ap.add_argument("--max-song-s", type=float, default=900.0, help="skip search results longer than this")
    ap.add_argument("--min-score", type=float, default=MIN_SCORE, help="see the calibration in the docstring")
    ap.add_argument("--ambiguity-margin", type=float, default=0.10)
    ap.add_argument("--min-window-score", type=float, default=MIN_WINDOW_SCORE,
                    help="gate for one {:.0f} s window; see the calibration in the docstring".format(WIN_S))
    ap.add_argument("--min-windows-agree", type=int, default=2,
                    help="windows that must put the clip at the same place when the whole clip fails")
    ap.add_argument("--music35", action="store_true", help="also write the whole song's 35-D features")
    ap.add_argument("--listen", type=Path, help="write an L=clip / R=song stereo check here (.mp3/.wav)")
    ap.add_argument("--export-dir", type=Path,
                    help="where songs/* and the outputs are copied; default output/sample_<today>_fullsong/<key>")
    ap.add_argument("--no-export", action="store_true", help="leave everything under --out-dir only")
    args = ap.parse_args()

    import librosa

    if args.clip:
        row = next((json.loads(l) for l in open(args.sources) if json.loads(l)["recording_id"] == args.clip), None)
        if row is None:
            raise SystemExit("{} not in {}".format(args.clip, args.sources))
        source = Path(row["provenance"]["source_audio"])
        key = args.clip.split(":", 1)[1].replace(":", "__")
    else:
        source = args.audio
        # "clip.mp4" / "audio.wav" say nothing; the folder they sit in usually does
        key = source.stem if source.stem not in ("clip", "audio", "video") else \
            "{}__{}".format(source.parent.name, source.stem)
    if not source.is_file():
        raise SystemExit("input does not exist: {}".format(source))
    out_dir = args.out_dir or DEFAULT_OUT_ROOT / key
    out_dir.mkdir(parents=True, exist_ok=True)
    song_dir = args.song_dir or out_dir / "songs"
    # one decoded copy of whatever came in (wav, mp3, an mp4 with a video track...)
    audio = out_dir / "clip_audio.wav"
    _ffmpeg(["-i", str(source), "-vn", "-ac", "1", "-ar", str(SR), str(audio)])

    print("[1/4] identify  {}".format(audio))
    tracks, windows = identify(audio)
    for t in tracks:
        print("      {}/{} windows  {} - {}  (shazam {}, isrc {})".format(
            t["votes"], t["n_windows"], t["artist"], t["title"], t["shazam_key"], t["isrc"]))
    queries = list(args.track)
    post = None
    if args.douyin_id:
        sys.path.insert(0, str(REPO / "tools"))
        import song_stores

        if args.douyin_post_json:  # fetched earlier: Douyin throttles, so posts are fetched apart from matching
            post = json.loads(Path(args.douyin_post_json).read_text(encoding="utf-8"))
        else:
            post = song_stores.douyin_post(args.douyin_id)
        # Douyin's own names go FIRST: with --stop-at, a hit there ends the search early
        queries = [n for n in song_stores.song_names(post) if n] + [q for q in queries
                                                                   if q not in song_stores.song_names(post)]
        print("      douyin post {}: music '{}' by '{}' ({}s) -> queries {}".format(
            args.douyin_id, post.get("music_title"), post.get("music_author"),
            post.get("music_duration"), queries))
    lyric_names = {}
    if args.lyrics:
        sys.path.insert(0, str(REPO / "tools"))
        import song_stores

        for line in args.lyrics:
            lyric_names[line] = song_stores.names_from_lyrics(line)
            for name in lyric_names[line]:
                title = name.split(" ", 1)[-1]  # "artist title": search the title, the stores hold its versions
                if title not in queries:
                    queries.append(title)
        print("      lyrics -> {}".format(lyric_names))
    if args.shazam_to_stores:
        # a track only one window named is often noise (a 10 s window matched elsewhere): the top
        # track always, the others only when two windows agree
        for t in [x for i, x in enumerate(tracks) if i == 0 or x["votes"] >= 2]:
            for q in title_queries(t["title"]):
                if q not in queries:
                    queries.append(q)
    result = {"clip": args.clip, "clip_input": str(source), "clip_audio": str(audio), "shazam_tracks": tracks,
              "shazam_windows": windows, "douyin_post": post, "name_queries": queries, "lyric_names": lyric_names,
              "candidates": [], "status": "unidentified"}
    if tracks or queries or (post and post.get("qishui_track_id")):
        print("[2/4] search + download (top {} per track)".format(args.top_k))
        cands = []
        if post and post.get("qishui_track_id"):
            # the "汽水音乐" button under the video: Douyin's own link to the whole song, tried first
            qs = song_stores.qishui_track(post["qishui_track_id"])
            if qs:
                cands.append({"shazam_key": None, "query": "douyin:qishui_anchor", "rank": 0, "store": "qishui",
                              "store_id": qs["id"], "videoId": "qishui:" + qs["id"], "url": qs["url"],
                              "yt_title": qs["name"], "yt_artists": qs["artist"],
                              "listed_duration_s": qs["duration_s"], "preview_end_s": qs["preview_end_s"]})
        cands += search_candidates(tracks, args.top_k, args.max_song_s) if tracks else []
        if queries:
            cands += store_candidates(queries, [s for s in args.stores.split(",") if s],
                                      args.store_top_k, args.max_song_s)
        clip_y, _ = librosa.load(str(audio), sr=SR, mono=True)
        clip = _Clip(clip_y)
        print("[3/4] align {} candidates".format(len(cands)))
        songs = {}
        # downloads, not alignment, are the cost (measured 2026-09-23: ~1.5 s to align a candidate
        # against a 15 s input, ~30 s to fetch it from NetEase/Kuwo), so they are fetched ahead in
        # order by --download-workers threads while alignment consumes them in order
        import concurrent.futures

        pool = concurrent.futures.ThreadPoolExecutor(max(1, args.download_workers))
        fetches = [pool.submit(fetch_candidate, c, song_dir) for c in cands]
        for c, fetched in zip(list(cands), fetches):
            wav, c["download_attempts"] = fetched.result()
            if wav is None:
                c["status"] = "download_failed"
                continue
            songs[c["videoId"]], _ = librosa.load(str(wav), sr=SR, mono=True)
            c.update(align(clip, songs[c["videoId"]], args.window_scan_floor))
            c.update({"wav": str(wav), "wav_sha256": _sha256(wav)})
            print("      rank {} {:.3f} r={:.3f} @ {:7.2f}s  {} - {}".format(
                c["rank"], c["score"], c["rate"], c["offset_s"], c["yt_artists"], c["yt_title"]))
            if args.stop_at is not None and c["score"] >= args.stop_at:
                print("      {:.3f} >= --stop-at {}: the remaining {} candidates are not aligned".format(
                    c["score"], args.stop_at, len(cands) - cands.index(c) - 1))
                cands = cands[:cands.index(c) + 1]
                for f in fetches:
                    f.cancel()  # downloads not started yet are dropped; running ones finish into the cache
                break
        pool.shutdown(wait=False)
        result["candidates"] = cands
        scored = [c for c in cands if "score" in c]
        if scored:
            for c in scored:
                c["window_agreement"] = window_agreement(c["windows"], args.min_window_score)
            best = max(scored, key=lambda c: c["score"])
            clip_dur = len(clip_y) / SR
            status = "unverified"
            if best["score"] >= args.min_score:
                ambiguous = best["second_score"] > best["score"] - args.ambiguity_margin
                status = "verified_ambiguous_offset" if ambiguous else "verified"
            else:
                # only part of the clip may be this song (a mid-video switch, a voice-over):
                # accept when enough windows, each over the window gate, agree on ONE place.
                # If two songs qualify, the one still playing at the clip's END wins: that
                # is the music a continuation has to follow.
                eligible = [c for c in scored if c["window_agreement"]["n_agree"] >= args.min_windows_agree]
                if eligible:
                    wb = max(eligible, key=lambda c: (c["window_agreement"]["covers_clip_end"],
                                                      c["window_agreement"]["n_agree"],
                                                      c["window_agreement"]["mean_score"]))
                    agree = wb["window_agreement"]
                    best, status = wb, "verified_by_windows"
                    best["offset_s"], best["rate"] = agree["clip_start_s"], agree["rate"]
                    print("      whole clip under the gate; {} windows (t0 {}) agree on {:.2f}s, mean {:.3f}{}".format(
                        agree["n_agree"], agree["t0s"], agree["clip_start_s"], agree["mean_score"],
                        "" if agree["covers_clip_end"] else
                        "  -- NOT at the clip's end: the music there is something else"))
            shz = [round(best["offset_s"] + best["rate"] * w["t0"] - w["offset"], 3)
                   for w in windows if w["key"] == best["shazam_key"] and w["offset"] is not None]
            end = best["offset_s"] + best["rate"] * clip_dur
            result.update({"status": status, "best": {
                "videoId": best["videoId"], "url": best["url"], "song": "{} - {}".format(
                    best["yt_artists"], best["yt_title"]), "wav": best["wav"], "wav_sha256": best["wav_sha256"],
                "score": best["score"], "second_score": best["second_score"],
                "second_offset_s": best["second_offset_s"], "rate": best["rate"],
                "offset_s": best["offset_s"], "clip_duration_s": clip_dur,
                "song_duration_s": best["song_duration_s"], "song_before_clip_s": best["offset_s"],
                "song_after_clip_s": best["song_duration_s"] - end,
                "window_agreement": best["window_agreement"],
                "shazam_offset_minus_aligned_s": shz,
                "shazam_offset_spread_s": float(np.ptp(shz)) if shz else None}})
            print("[4/4] {}  score {:.3f} (min {:.2f}), 2nd peak {:.3f} @ {:.1f}s".format(
                status, best["score"], args.min_score, best["second_score"], best["second_offset_s"]))
            print("      clip = song[{:.2f}s .. {:.2f}s] at rate {:.3f}; {:.1f}s of song before, {:.1f}s after".format(
                best["offset_s"], end, best["rate"], best["offset_s"], best["song_duration_s"] - end))
            print("      shazam offset - aligned offset per window: {}".format(shz))
            if status.startswith("verified"):
                timeline = {"clip_start_s": best["offset_s"] / best["rate"]}
                timeline["clip_end_s"] = timeline["clip_start_s"] + clip_dur
                timeline["song_end_s"] = best["song_duration_s"] / best["rate"]
                # by the wav's OWN stem, not the videoId: a netease candidate is stored as
                # "netease_<id>.mp3" while its videoId reads "netease:<id>", so globbing the
                # videoId matched nothing and the run died right after picking the winner
                original = next(p for p in Path(best["wav"]).parent.glob(Path(best["wav"]).stem + ".*")
                                if p.suffix != ".wav")
                files = write_clip_speed_song(original, best["rate"], out_dir, timeline)
                by_windows = status == "verified_by_windows"
                check = check_retimed(clip, Path(files["full_song_clip_speed"]), timeline["clip_start_s"],
                                      t0=best["window_agreement"]["t0s"][0] if by_windows else None)
                gate = args.min_window_score if by_windows else args.min_score
                result.update({"timeline_clip_speed": timeline, "files": files, "retimed_check": check})
                print("      full_song_clip_speed.wav: clip at {:.2f}-{:.2f}s of {:.2f}s; continuation {:.1f}s".format(
                    timeline["clip_start_s"], timeline["clip_end_s"], timeline["song_end_s"], files["continuation_s"]))
                print("      re-check at rate 1.0: score {:.3f}, start found {:.3f}s (error {:+.3f}s)".format(
                    check["score_at_rate_1"], check["found_start_s"], check["error_s"]))
                if abs(check["error_s"]) > 2.0 / 30 or check["score_at_rate_1"] < gate:
                    result["status"] = status = "retime_failed"
                    print("      RETIME FAILED: the written file does not hold the clip where timeline says")
            if status.startswith("verified"):
                if args.music35:
                    full, rep = music35_check(Path(result["files"]["full_song_clip_speed"]),
                                              result["timeline_clip_speed"]["clip_start_s"], args.clip, args.sources)
                    np.save(out_dir / "music_35_full.npy", full)
                    result["music_35"] = dict(rep, path=str(out_dir / "music_35_full.npy"))
                    print("      music_35: {}".format(json.dumps(rep)))
                if args.from_clip_start:
                    # the music a CONTINUATION is danced to: the clip's own stretch first, then
                    # everything after it, on one timeline whose frame 0 is the clip's frame 0
                    seg = out_dir / "music_from_clip_start.wav"
                    _ffmpeg(["-ss", "{:.4f}".format(timeline["clip_start_s"]),
                             "-i", result["files"]["full_song_clip_speed"], "-c:a", "pcm_s16le", str(seg)])
                    result["files"]["music_from_clip_start"] = str(seg)
                    result["files"]["music_from_clip_start_s"] = _duration(seg)
                    print("      from clip start: {:.1f}s ({:.1f}s of it is the clip) -> {}".format(
                        _duration(seg), clip_dur, seg))
                    if args.music35:
                        full, rep = music35_check(seg, 0.0, args.clip, args.sources)
                        # A song's fade-out carries no beat, and the bar grid then merges it into
                        # ONE slot: 7650126416710192357's outro is 7.0 s after the last beat, and
                        # infer_atomic refused the run ("1 retrieval unit asks for more frames than
                        # its class contains ... 225 frames, longest candidate 150").  The library's
                        # longest move is 2.97 s = 89 frames, so the tail is cut to that.
                        beats = np.flatnonzero(full[:, 34] > 0.5)
                        keep = int(beats[-1]) + args.end_after_last_beat_frames if len(beats) else len(full)
                        if keep < len(full):
                            dropped = (len(full) - keep) / 30.0
                            full = full[:keep]
                            _ffmpeg(["-i", str(seg), "-t", "{:.4f}".format(keep / 30.0), "-c:a", "pcm_s16le",
                                     str(seg.with_name("music_from_clip_start_trimmed.wav"))])
                            seg.with_name("music_from_clip_start_trimmed.wav").replace(seg)
                            result["files"]["music_from_clip_start_s"] = _duration(seg)
                            rep = dict(rep, frames=int(keep), beatless_outro_dropped_s=round(dropped, 2))
                            print("      dropped {:.2f}s of beatless outro (one bar slot the vocabulary "
                                  "cannot fill); music now {:.1f}s".format(dropped, keep / 30.0))
                        np.save(out_dir / "music_35_from_clip_start.npy", full)
                        result["music_35_from_clip_start"] = dict(
                            rep, path=str(out_dir / "music_35_from_clip_start.npy"))
                        print("      music_35 from clip start: {}".format(json.dumps(rep)))
                if args.listen:
                    write_listen_check(clip_y, songs[best["videoId"]], best, args.listen)
                    result["listen_check"] = str(args.listen)
                    print("      listen check -> {}".format(args.listen))
    if not args.no_export:
        import datetime

        export_dir = args.export_dir or REPO / "output" / "sample_{}_fullsong".format(
            datetime.date.today().strftime("%Y%m%d")) / key
        # only candidates this run actually has: their wav plus the original next to it.
        # A glob over a shared --song-dir would also list another run's in-flight partials.
        song_files = []
        for c in result["candidates"]:
            if c.get("wav"):
                wav = Path(c["wav"])
                song_files += [wav] + [p for p in wav.parent.glob(wav.stem + ".*")
                                       if p.suffix not in (".wav", ".part", ".partial", ".ytdl")]
        result["export"] = export_outputs(out_dir, export_dir, sorted(set(song_files)))
    text = json.dumps(result, ensure_ascii=False, indent=1)
    # atomic: a batch that reads result.json (or a kill mid-write) never sees half a file
    tmp = out_dir / "result.json.partial"
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, out_dir / "result.json")
    print("wrote {}".format(out_dir / "result.json"))
    if result.get("export", {}).get("exported"):
        (Path(result["export"]["export_dir"]) / "result.json").write_text(text, encoding="utf-8")
        print("exported {} files ({:.1f} MB) -> {}".format(
            len(result["export"]["files"]), result["export"]["bytes"] / 2**20, result["export"]["export_dir"]))
    return 0 if result["status"].startswith("verified") else 2


if __name__ == "__main__":
    sys.exit(main())
