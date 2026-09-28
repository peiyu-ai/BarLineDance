#!/usr/bin/env python3
"""Write ``output/sample_menu.md`` (+ ``.tsv``): every raw T-line video, by split, with its song.

A sample code (``7030793823240424742__clip000``) says nothing to the person choosing
which clips to render.  The menu lists each RAW upload (the Douyin video id; its
12-24 s training clips are not listed separately) with the song it dances to, for the
split the CURRENT T-line model was trained and evaluated on (``configs/t_line/
current.env`` -> ``T_RELEASE``), so when the config moves to T3 the menu follows.
The operator asked for exactly two columns (2026-09-23: "我只要raw clip 名和对应的歌名");
everything behind a name -- per-clip status, scores, song positions, marks, renders --
goes to ``runs/txy_t2_song_menu/sample_menu_detail.tsv`` (and ``--detail-md``).

A raw upload's song is the strongest of: round 2, the whole upload's audio searched on
YouTube Music + NetEase + Kuwo with Shazam's names, on-screen titles and lyric lookups
(``runs/txy_t2_song_menu/run_identify_raw.py``); round 1, each clip on its own
(``run_identify.sh``).  The per-clip statuses below are what the detail table shows.

Where the name comes from
-------------------------
The Douyin caption is not stored with the corpus: Lodge's fetcher keeps only the mp4,
and the public share page answers with an anti-bot challenge instead of the post
(checked 2026-09-23).  Reading it takes the account cookie (``retrieve_full_song
--douyin-id``), which this menu's batch does not use.  So the name is the **song**,
recovered from the clip's own audio by ``tools/retrieve_full_song.py`` (Shazam ->
YouTube Music -> chroma alignment), run over every clip by
``runs/txy_t2_song_menu/run_identify.sh``.  A result reached another way -- a store
search on a name taken from the Douyin post (7650126416710192357, written by another
session) -- is shown with where its name came from.  What a name is worth
depends on how it was reached, and every row says which:

    verified            the downloaded recording aligns with the clip (score >= 0.60;
                        wrong songs max 0.43, covers 0.45-0.52 in the tool's calibration)
    verified_ambiguous  same, but the clip fits two places in the song (a repeated chorus)
    by_windows          only part of the clip aligns (song switch / voice-over); the
                        tool prefers the song playing at the clip's end, and says so
                        (``partial``) when the one it found does not reach the end
    retime_failed       the recording aligned (the name holds) but the re-timed song
                        file failed its re-check, so its continuation is not usable
    shazam_only         Shazam named it, no download aligned (cover, percussive track,
                        the right recording not among the downloads): Shazam's word,
                        qualified by its votes, whether those windows agree on one
                        place in the song, and the named song's own best candidate
    shazam_split        Shazam's windows named different songs with tied votes: no name
    inferred            not verified itself; a verified, whole, single-song clip that the
                        audio fingerprint directly pairs with it names the song, and this
                        clip's own Shazam answers (if any) include that song
    unidentified        Shazam answered every window with "no match"
    failed              a Shazam request errored (network) or result.json is
                        unreadable -- rerun, do not read it as "no match"
    not_run             no result.json yet -- counted in the header, never silent

Plus the **publish date**, which is the top 32 bits of the aweme id (checked: the 17
uploads fetched 2026-09-22 decode to 09-02..09-21, between the two fetches, and the
oldest T2 upload to 2021-11-11, the account's end cursor).

The cross-check that can fail
-----------------------------
``txy_t2_music_groups.json`` lists pairs of clips whose audio provably overlaps
(chroma alignment at a 0.1% false-positive operating point), independently of
Shazam.  Two such clips must share a song: the check is per verified PAIR, on every
track Shazam voted for (a clip that switches songs holds two), not per component --
a component is single-linkage, so its two ends can share no audio at all.  The
header prints how many pairs had both clips named and how many agree; a disagreeing
pair is listed with how much audio it shares.  First instance: 7676475940934935409's
two back-to-back clips share 10.8 s of the same music (lag -64 frames), and Shazam
answered that music with three different songs, one window each -- the listing says
that, and nothing about why.  (A first draft explained it as a mid-video song switch;
the pair's own lag says the music is the same, so that story was dropped.)

What "release windows" means
----------------------------
The 150-frame windows (stride 15) the release cut from the clip.  Train windows are
the training samples and the retrieval library (``infer_atomic`` indexes
``<data_root>/train`` only); val/test windows are for evaluation.  A clip with 0 was
removed by an exclusion manifest somewhere up the release's ``derived_from`` chain
(for T2: the timebase census ``runs/timebase_exclude_v1.jsonl``, applied at
release_v3) -- the reason is read from that manifest, not assumed.

Usage::

    python3 tools/build_sample_menu.py            # current T-line config -> output/sample_menu.{md,tsv}
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SONGS = Path("/cache/atomicdance-assets/scratch/full_songs/txy_t2_menu")  # round 1: per clip
DEFAULT_RAW = Path("/cache/atomicdance-assets/scratch/full_songs/txy_t2_raw")  # round 2: per whole upload
DEFAULT_LOCAL = Path("/cache/atomicdance-assets/scratch/full_songs/txy_t2_local")  # round 3: local catalog
DEFAULT_DETAIL = REPO / "runs/txy_t2_song_menu/sample_menu_detail.tsv"
DEFAULT_GROUPS = REPO / "runs/txy_t2_20260922/txy_t2_music_groups.json"
DEFAULT_SONG_SPLIT = REPO / "runs/txy_t2_20260922/txy_t2_song_split.json"
DEFAULT_TIMEBASE_DECISION = REPO / "runs/txy_t2_20260922/timebase_decision.json"
DEFAULT_PREVIOUS_ENV = REPO / "configs/t_line/t1_fix7.env"
DEFAULT_NOTES = REPO / "runs/txy_t2_song_menu/notes.json"  # hand-read on-screen titles and looked-at conclusions
MARK_LISTS = (  # tag, file, what it is
    ("vis10", "runs/vis_clips_t10.txt", "每轮固定渲的 10 条"),
    ("eval20", "runs/eval_clips_txy_t20.txt", "固定评测 20 条"),
    ("val30", "runs/eval_clips_txy_t_val30.txt", "排臂/选 planner seed 用的 val 30 条"),
)
SPLITS = ("test", "val", "train")
VERIFIED = ("verified", "verified_ambiguous", "by_windows", "retime_failed")  # the name was checked
UNNAMED = ("unidentified", "failed", "not_run", "shazam_split")
# Shazam windows voting for one track put the clip at one place in its reference when
# offset - t0 is constant.  Measured 2026-09-23 on the first 98 verified clips with >= 2 such
# windows: 90 spread <= 1.5 s, 8 spread 6-175 s (a repeated chorus matched elsewhere) -- so a
# spread is a weak warning, not a verdict; the menu prints the live base rate next to it.
OFFSET_SPREAD_S = 1.5
SHAZAM_SETTLES_S = 0.10  # Shazam-minus-aligned spread on one placement: 0.003-0.016 s on the settled repeats
LOW_SCORE = 0.70  # other versions over the same backing reached 0.61-0.67 in the test/val results
STRONG = 0.75  # same recording: min 0.752 in the tool's calibration; other versions reached 0.69
VERSION_FLOOR = 0.45  # over every wrong song (max 0.43): the same composition, maybe another version
# output/ folders whose mp4s show only the source footage, at a size the frame-size check
# cannot catch: render_cut_comparison.py montages (900x648 crops of the original video)
OVERLAP_MIN_S = 3.0  # song-time overlap that counts as "the same stretch of music"
FOOTAGE_ONLY_DIRS = ("sample_20260902_cut_methods", "sample_20260902_cut_bar4", "sample_20260902_dpvo_failures")
STATUS_SHOW = {  # status -> (mark in the md table, one-line meaning)
    "verified": ("✅", "下载到的录音与整条 clip 对齐(分数 ≥ 0.60;低于 0.70 时位置栏写出分数 —— 同伴奏的别的版本见过 0.61–0.67)"),
    "verified_ambiguous": ("✅?", "录音已对齐,但副歌重复,clip 在曲中的位置有两处候选,Shazam 也没能定下"),
    "by_windows": ("◐", "只有一部分对得上(中途换歌/旁白);优先取 clip 结尾在放的那首,取不到时标\"仅部分\""),
    "retime_failed": ("✅*", "录音已对齐(名字可信),但按 clip 速度重写的整曲文件复核没过 —— 续跳别用它"),
    "shazam_only": ("⚠", "Shazam 给了名字,但没有下载到能对齐的录音 —— 名字未核实"),
    "shazam_split": ("⚠?", "Shazam 各窗口给了不同的歌、票数打平 —— 没有可信的名字,列出的是各窗口的答案"),
    "inferred": ("⇢", "自己没核实;音频指纹直接配对的一条已核实 clip(单曲、整段)是这首歌,且与本条自己的 Shazam 不矛盾"),
    "unidentified": ("✗", "Shazam 每个窗口都答\"没匹配\""),
    "failed": ("!", "识别请求出错(网络)或结果文件读不了 —— 该重跑,不代表认不出"),
    "not_run": ("…", "还没跑识别"),
}


# --------------------------------------------------------------------------- inputs
def read_env(path: Path, seen=None) -> dict:
    """KEY=VALUE lines of a configs/t_line/*.env, following its ``. other.env`` includes."""
    seen = seen or set()
    if path in seen:
        raise SystemExit("include loop at {}".format(path))
    seen.add(path)
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(". "):
            out.update(read_env(REPO / line[2:].strip(), seen))
        elif "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def release_sources(release: Path) -> tuple[Path, list[dict]]:
    """The split manifest a release was built from, refusing one that changed since."""
    rec = json.loads((release / "build.json").read_text(encoding="utf-8"))["input_manifests"]["sources.jsonl"]
    path = Path(rec["path"])
    if sha256(path) != rec["sha256"]:
        raise SystemExit("{} changed since {} was built from it; the split shown would not be the "
                         "model's".format(path, release))
    return path, read_jsonl(path)


def exclusions(release: Path) -> dict:
    """recording id -> (short reason, manifest) for every exclusion manifest up the derived_from chain."""
    out, seen = {}, set()
    while release and release not in seen:
        seen.add(release)
        b = json.loads((release / "build.json").read_text(encoding="utf-8"))
        d = b.get("derived_from") or {}
        if d.get("exclusion_manifest"):
            for r in read_jsonl(Path(d["exclusion_manifest"])):
                m = re.search(r"3D spans ([\d.]+) of the clip", r.get("reason", ""))
                short = "时基剔除 {}×".format(m.group(1)) if m else "剔除"
                out.setdefault(r["sequence"], (short, d["exclusion_manifest"], r.get("reason", "")))
        release = Path(d["release"]) if d.get("release") else None
    return out


def publish_date(upload_id: str) -> str:
    """Douyin aweme ids carry the post time (unix seconds) in their top 32 bits; shown in UTC+8."""
    t = datetime.datetime.fromtimestamp(int(upload_id) >> 32, datetime.timezone(datetime.timedelta(hours=8)))
    return t.strftime("%Y-%m-%d")


def clip_key(recording_id: str) -> str:
    return recording_id.split(":", 1)[1].replace(":", "__")


# --------------------------------------------------------------------------- songs
def track_name(artist, title) -> str:
    return "{} - {}".format(artist, title) if artist else str(title)


def read_song(result_path: Path) -> dict:
    """One clip's result.json from retrieve_full_song, reduced to what the menu shows."""
    if not result_path.is_file():
        return {"status": "not_run", "shazam": []}
    try:
        r = json.loads(result_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {"status": "failed", "shazam": [], "why": "unreadable result.json"}
    tracks = r.get("shazam_tracks") or []
    status = r.get("status")
    windows = r.get("shazam_windows") or []
    out = {"n_windows": len(windows),
           "shazam": [{"key": t["shazam_key"], "name": track_name(t["artist"], t["title"]),
                       "votes": t["votes"]} for t in tracks]}
    best = r.get("best")
    mapped = {"verified": "verified", "verified_ambiguous_offset": "verified_ambiguous",
              "verified_by_windows": "by_windows", "retime_failed": "retime_failed"}.get(status)
    def coherent(key):
        starts = [w["offset"] - w["t0"] for w in windows if w.get("key") == key and w.get("offset") is not None]
        return (max(starts) - min(starts) <= OFFSET_SPREAD_S) if len(starts) >= 2 else None

    if mapped and best:
        # the name is the ALIGNED recording's, which can come from any Shazam track -- or from
        # no Shazam track at all: a --track / --douyin-id store search (7650126416710192357)
        cand = next((c for c in r.get("candidates") or [] if c.get("videoId") == best["videoId"]), {})
        wa = best.get("window_agreement") or {}
        via = "shazam" if cand.get("shazam_key") else ("douyin" if r.get("douyin_post") else "name")
        # Shazam's per-window offsets minus the aligned place: constant = every window agrees
        # with THIS place (a different master only shifts it), so a repeated chorus is settled
        sh = best.get("shazam_offset_minus_aligned_s") or []
        settled = len(sh) >= 2 and max(sh) - min(sh) <= SHAZAM_SETTLES_S
        credit = next((t for t in tracks if t["shazam_key"] == cand.get("shazam_key")), None)
        display = best["song"]
        if credit is None and cand.get("query"):
            # a store copy found by searching a title Shazam gave: the audio confirmed Shazam's
            # track, and a store listing can be a user upload ("罗大方 - 29 - NCT 127 - Fact Check")
            credit = next((t for t in tracks if cand["query"] in (t["title"], re.sub(
                r"\s*[\(（\[【].*?[\)）\]】]\s*", " ", t["title"] or "").strip())), None)
            if credit:
                display = track_name(credit["artist"], credit["title"])
        if mapped == "verified_ambiguous" and settled:
            mapped = "verified"  # the chroma runner-up is a repeat; Shazam's windows all sit at the chosen place
        out.update(status=mapped, song_key=cand.get("shazam_key") or "store:" + best["videoId"],
                   name=display, store_listing=best["song"], score=best["score"], url=best.get("url"),
                   offset_s=best["offset_s"],
                   rate=best["rate"], clip_s=best["clip_duration_s"], song_s=best["song_duration_s"],
                   partial=status == "verified_by_windows" and not wa.get("covers_clip_end", True),
                   ambiguous=status == "verified_ambiguous_offset" and not settled, settled_by_shazam=settled,
                   shazam_name=track_name(credit["artist"], credit["title"]) if credit else None, via=via,
                   queries=r.get("name_queries") or [], coherent=coherent(cand.get("shazam_key")))
        return out
    # evidence below the gate, kept for naming a video nothing verified: Douyin's own match of the
    # sound / its Qishui link, and the best candidate over the wrong-song ceiling (0.43) -- another
    # version of the same composition in the tool's calibration (covers 0.45-0.69)
    post = r.get("douyin_post") or {}
    out["douyin_names"] = [n for n in (post.get("matched_song_title"), post.get("qishui_title"))
                           if n and "原声" not in n]
    # last resort: what the description's hashtags call the piece ("#王一博s11特别舞台", "#蔚蓝海岸舞蹈"),
    # with the words that name the dance rather than the music taken out
    noise = r"舞蹈|编舞|翻跳|原创|hiphop|jazz|kpop|dance|挑战|随拍|日常|vlog|抖音|卡点舞|卡点|热门|潮流"
    hints = [re.sub(noise, "", t, flags=re.I).strip() for t in re.findall(r"#([^#\s_]+)", post.get("desc") or "")]
    out["desc_hint"] = next((h for h in hints if len(h) >= 2), None)
    near = [c for c in r.get("candidates") or [] if c.get("score", 0) >= VERSION_FLOOR]
    if near:
        c = max(near, key=lambda c: c["score"])
        name = c.get("song") or track_name(c.get("yt_artists"), c.get("yt_title"))  # round 3 lists "song"
        out["near_version"] = {"name": name, "score": c["score"]}
    if not tracks:
        errors = sum(bool(w.get("error")) for w in windows)
        if errors:
            return dict(out, status="failed", why="{} Shazam request(s) errored".format(errors))
        return dict(out, status="unidentified")
    top = tracks[0]
    if len(tracks) > 1 and tracks[1]["votes"] == top["votes"]:
        # a tie: the "top" track is only whichever window came first (Counter order)
        tied = [t for t in tracks if t["votes"] == top["votes"]]
        return dict(out, status="shazam_split", why=status, votes=top["votes"],
                    name=" / ".join(track_name(t["artist"], t["title"]) for t in tied))
    # recognised, not checked: Shazam's top vote is the name.  How strong it is: the votes,
    # whether those windows put the clip at ONE place in the reference (offset - t0 constant;
    # the tool's docstring: a spread = disagreement), and the named track's own best download
    own = [c["score"] for c in r.get("candidates") or [] if "score" in c and c.get("shazam_key") == top["shazam_key"]]
    return dict(out, status="shazam_only", song_key=top["shazam_key"], name=track_name(top["artist"], top["title"]),
                why=status, votes=top["votes"], best_score=max(own) if own else None,
                coherent=coherent(top["shazam_key"]))


def switch_tracks(song: dict) -> list[dict]:
    """Other tracks two or more windows agree on (one window alone can be noise)."""
    return [t for t in song.get("shazam", []) if t["key"] != song.get("song_key") and t["votes"] >= 2]


def fingerprint_check(clips: dict, pairs: list[dict]) -> dict:
    """Clips the fingerprint pairs directly must share a Shazam track; spread names where safe.

    A pair is checked when both clips carry ONE Shazam identity (verified, or shazam_only;
    a tie is no identity).  A name is spread from a donor -- a direct pair partner that is
    verified, whole (not a partial by_windows) and single-song (no second track two
    windows agree on) -- to a partner that is unidentified, or whose own Shazam tracks
    include the donor's song (a split vote, or an unchecked name the pair corroborates).
    """
    checked, agree, conflicts = 0, 0, []
    neighbours = collections.defaultdict(set)
    one_identity = VERIFIED + ("shazam_only",)
    for p in pairs:
        a, b = p["left"], p["right"]
        if a not in clips or b not in clips:
            continue
        neighbours[a].add(b)
        neighbours[b].add(a)
        sa, sb = clips[a]["song"], clips[b]["song"]
        if sa["status"] not in one_identity or sb["status"] not in one_identity:
            continue
        checked += 1
        if ({sa["song_key"]} | {t["key"] for t in sa["shazam"]}) & ({sb["song_key"]} | {t["key"] for t in sb["shazam"]}):
            agree += 1
        else:
            conflicts.append({"pair": (a, b), "score": p.get("score"), "shared_s": shared_seconds(p, clips)})
    inferred = 0
    for rid, c in clips.items():
        st = c["song"]["status"]
        if st not in ("unidentified", "shazam_split", "shazam_only"):
            continue
        donors = [clips[n]["song"] for n in neighbours[rid]
                  if clips[n]["song"]["status"] in VERIFIED and not clips[n]["song"].get("partial")
                  and not switch_tracks(clips[n]["song"])]
        keys = {d["song_key"] for d in donors}
        if len(keys) != 1:
            continue
        key = next(iter(keys))
        if st != "unidentified" and key not in {t["key"] for t in c["song"]["shazam"]}:
            continue  # Shazam says another song: that is a conflict to show, not a name to overwrite
        # carry the donor's score: an inference is only as strong as the alignment it borrows
        c["song"] = dict(c["song"], status="inferred", song_key=key, name=donors[0]["name"], was=st,
                         score=max(d.get("score", 0.0) for d in donors))
        inferred += 1
    return {"pairs": sum(1 for p in pairs if p["left"] in clips and p["right"] in clips),
            "checked": checked, "agree": agree, "conflicts": conflicts, "inferred": inferred}


def shared_seconds(pair: dict, clips: dict) -> float:
    """How much audio a fingerprint pair shares: lag in 30 fps music frames, as fingerprint_wild_music aligns it."""
    lag = int(pair.get("lag_frames") or 0)
    la, lb = (clips[pair[k]]["frames"] for k in ("left", "right"))
    return max(0, min(la - max(0, lag), lb - max(0, -lag))) / 30.0


# --------------------------------------------------------------------------- renders
def _frame_size(path: Path):
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "stream=width,height", "-of", "csv=p=0", str(path)],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        w, h = out.split(",")[:2]
        return int(w), int(h)
    except (ValueError, subprocess.SubprocessError):
        return None


def scan_renders(output_dir: Path, source_size: dict) -> dict:
    """key -> [(top-level dir, path relative to output/)], one per top-level dir, newest first.

    Newest = the date in ``sample_YYYYMMDD_*`` (the naming convention), and on the same
    date the most recently written file.  A file with its clip's own source frame size
    is a copy of the footage (sample_20260902_dpvo_failures keeps those), not a render.
    One file per folder is probed and stands for the folder: probing every candidate
    took minutes on this 4-core box, and a folder holds one kind of video.
    """
    found = collections.defaultdict(list)
    first_in = {}
    pat = re.compile(r"(\d{19}__clip\d{3})")
    for dp, dirs, files in os.walk(output_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")  # .staging-* = a render still being written
                   and not (Path(dp) == output_dir and d.startswith(FOOTAGE_ONLY_DIRS))]
        for f in files:
            m = pat.search(f)
            if f.endswith(".mp4") and m and m.group(1) in source_size:
                full = Path(dp, f)
                rel = full.relative_to(output_dir)
                m_date = re.match(r"samples?_(\d{8})", rel.parts[0])
                found[m.group(1)].append(((m_date.group(1) if m_date else "0", full.stat().st_mtime),
                                          rel.parts[0], str(rel)))
                first_in.setdefault(dp, (full, m.group(1)))
    footage = {dp for dp, (full, key) in first_in.items() if _frame_size(full) == source_size[key]}
    out = {}
    for key, items in found.items():
        per_dir = {}
        for age, top, rel in sorted(items, key=lambda x: x[0], reverse=True):
            if top not in per_dir and str(Path(output_dir, rel).parent) not in footage:
                per_dir[top] = (age, rel)
        out[key] = [(top, rel) for top, (age, rel) in sorted(per_dir.items(), key=lambda kv: kv[1][0], reverse=True)]
    return out


# --------------------------------------------------------------------------- render md
def mmss(s: float) -> str:
    s = int(round(max(0.0, s)))
    return "{}:{:02d}".format(s // 60, s % 60)


def position(song: dict) -> str:
    if song["status"] == "shazam_only":  # how strong the unchecked name is
        txt = "Shazam {}/{} 窗".format(song["votes"], song["n_windows"])
        if song.get("coherent") is not None:
            txt += "·偏移{}".format("一致" if song["coherent"] else "分散")
        txt += " · 本曲候选最好 {:.2f}".format(song["best_score"]) if song.get("best_score") is not None \
            else " · 本曲没有下载到候选"
        return txt
    if song["status"] == "shazam_split":
        return "Shazam {} 窗 {} 首,各 {} 票".format(song["n_windows"], len(song["name"].split(" / ")), song["votes"])
    if song["status"] == "unidentified":
        return "Shazam 0/{} 窗".format(song.get("n_windows", 0))
    if "offset_s" not in song:
        return ""
    end = song["offset_s"] + song["rate"] * song["clip_s"]
    txt = "{}–{} / {}".format(mmss(song["offset_s"]), mmss(end), mmss(song["song_s"]))
    if song.get("score", 1.0) < LOW_SCORE:
        txt += " · 对齐 {:.2f}{}".format(song["score"], "(Shazam 定位一致)" if song.get("settled_by_shazam") else "")
    if abs(song["rate"] - 1.0) >= 0.005:
        # clip(t) = song(offset + rate * t): the clip plays the song ``rate`` times as fast
        txt += " ×{:.2f}".format(song["rate"])
    if song.get("ambiguous"):
        txt += " (?)"
    return txt


def norm(text: str) -> str:
    return re.sub(r"[\s()\[\]（）【】·,.&'-]+", "", str(text)).lower()


def md_cell(text) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def notes_cell(c: dict) -> str:
    n = c.get("notes") or {}
    parts = []
    if n.get("onscreen"):
        parts.append("画面字幕「{}」".format(md_cell(n["onscreen"])))
    if n.get("note"):
        parts.append(md_cell(n["note"]) + ("([图]({}))".format(n["evidence"]) if n.get("evidence") else ""))
    return " <sub>{}</sub>".format(";".join(parts)) if parts else ""


def song_cell(song: dict) -> str:
    st = song["status"]
    if st == "shazam_split":
        return "(Shazam 分歧) " + md_cell(song["name"])
    if st in UNNAMED:
        return {"not_run": "(未跑)", "unidentified": "(未识别)", "failed": "(识别出错)"}[st]
    name = md_cell(song["name"])
    if song.get("partial"):
        name += " (仅部分)"
    if song.get("shazam_name") and norm(song["shazam_name"]) != norm(song["name"]):
        name += " <sub>Shazam: {}</sub>".format(md_cell(song["shazam_name"]))
    if song.get("via") in ("douyin", "name"):
        name += " <sub>(Shazam 没认出;{}搜曲库核实)</sub>".format(
            "按抖音作品描述里的名字" if song["via"] == "douyin" else "按给定名字")
    if st == "inferred":
        name += " <sub>(借指纹配对的已核实 clip)</sub>"
    # another track two windows agree on: a song switch, or another version of the same song
    # (Owl City "Good Time" + Loreen Harris' sound-alike, 7116515578478611716) -- not decided here
    extra = switch_tracks(song)
    if extra and st != "inferred":
        name += " <sub>+另有 Shazam 结果: {}</sub>".format(md_cell(extra[0]["name"]))
    return name


def status_label(song: dict) -> str:
    return song_cell(song) if song["status"] in UNNAMED else md_cell(song["name"])


def windows_cell(c: dict) -> str:
    if c["windows"]:
        return str(c["windows"])
    return "**0** " + c["excluded"][0] if c["excluded"] else "**0**"


def write_md(path: Path, clips: dict, meta: dict) -> None:
    by_split = {s: sorted((c for c in clips.values() if c["split"] == s), key=lambda c: c["rid"])
                for s in SPLITS}
    tagged = {t: [c for c in clips.values() if t in c["marks"]] for t in [m[0] for m in MARK_LISTS] + ["新增"]}
    L = []
    L.append("# 汤汤汤小圆 T 线 sample 菜单")
    L.append("")
    L.append("由 `tools/build_sample_menu.py` 生成于 {};切分 = 当前 T 线配置(`configs/t_line/current.env` → "
             "`{}`)。机器可读版:[`{}`]({})。".format(meta["generated"], meta["release"], meta["tsv"], meta["tsv"]))
    L.append("")
    L.append("## 一眼看")
    L.append("")
    L.append("| split | clip | 上传(视频) | release 窗口 | 0 窗口(被剔除) | 歌名已核实 | 仅 Shazam(含分歧) | 指纹推断 | 认不出 | 出错/未跑 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    tot = collections.Counter()
    for s in SPLITS:
        cs = by_split[s]
        st = collections.Counter(c["song"]["status"] for c in cs)
        row = {"clip": len(cs), "up": len({c["upload"] for c in cs}), "win": sum(c["windows"] for c in cs),
               "zero": sum(c["windows"] == 0 for c in cs), "ver": sum(st[k] for k in VERIFIED),
               "sh": st["shazam_only"] + st["shazam_split"], "inf": st["inferred"], "un": st["unidentified"],
               "nr": st["failed"] + st["not_run"]}
        tot.update(row)
        L.append("| **{}** | {clip} | {up} | {win} | {zero} | {ver} | {sh} | {inf} | {un} | {nr} |".format(s, **row))
    L.append("| 合计 | {clip} | {up} | {win} | {zero} | {ver} | {sh} | {inf} | {un} | {nr} |".format(**tot))
    L.append("")
    test_up = {c["upload"] for c in by_split["test"]}
    not_eval = [c for c in by_split["test"] if "eval20" not in c["marks"]]
    L.append("* **test 就这么大**:{} 条 clip 来自 {} 个上传。`eval20` 是其中 {} 条、覆盖 {}/{} 个 test 上传;"
             "`vis10` 的 {} 条{}全在 eval20 里。test 里不在 eval20 的 {} 条是 {}。".format(
                 len(by_split["test"]), len(test_up), len(tagged["eval20"]),
                 len({c["upload"] for c in tagged["eval20"] if c["split"] == "test"}), len(test_up),
                 len(tagged["vis10"]), "" if all("eval20" in c["marks"] for c in tagged["vis10"]) else "**不**",
                 len(not_eval), "、".join("`{}`{}".format(c["key"], "(相对 {} 新增)".format(meta["previous_name"])
                                                         if c["added"] else "") for c in not_eval) or "无"))
    L.append("* val 的 {} 条没参与训练,但**用过**:`val30`({} 条)是排臂、选 planner seed 的集合;"
             "把 val 当额外测试集时要记得它不是完全没被看过。".format(
                 len(by_split["val"]), sum(c["split"] == "val" for c in tagged["val30"])))
    added = [c for c in clips.values() if c["added"]]
    later = [c for c in added if int(c["upload"]) > int(meta["previous_newest"])]
    L.append("* `新增` = 相对上一版({})新加进来的 {} 条(按 split:{})。其中 {} 条来自 {} 个在 {} 最新一个视频({})之后发布的新视频,"
             "另 {} 条来自 {} 个更早的老视频,是这一版才切进来的。".format(
                 meta["previous_name"], len(added), ", ".join("{} {}".format(s, sum(c["split"] == s for c in added))
                                                               for s in SPLITS),
                 len(later), len({c["upload"] for c in later}), meta["previous_name"],
                 publish_date(meta["previous_newest"]), len(added) - len(later),
                 len({c["upload"] for c in added if c not in later})))
    crossing_groups = meta["fingerprint_groups_crossing"]
    L.append("* 切分的目标是**按歌不相交**(音频指纹同录音组整体进一个 split),**不是按舞者不相交** —— 全部是同一个账号。"
             "指纹靠对齐音频,只收对齐分 ≥ {} 的配对:同一首歌里不重叠的两段(一条用主歌、一条用副歌)连不上,"
             "重叠但音质差/混过音、对齐分不到门槛的也连不上,所以\"不同 split 没有同一首歌\"只是下界,"
             "下一条和文末\"按歌索引\"用 Shazam 的身份补上这一块。".format(meta["fingerprint_threshold"]) + (
                 "**并且有 {} 组指纹已经连上的同录音跨了 split**:{}。".format(len(crossing_groups), ";".join(
                     "{}{}".format("、".join("`{}`·{}".format(clip_key(m), clips[m]["split"]) for m in g["members"]),
                                   "(切分时钉住了上一版的旧划分)" if g["pinned"] else "")
                     for g in crossing_groups)) if crossing_groups else "指纹连上的同录音组没有跨 split 的。"))
    leaks = meta["position_overlaps"]
    pending = sum(c["song"]["status"] == "not_run" for c in clips.values())
    L.append("* **test/val 与 train 用了同一录音、且在原曲里位置重叠**(两边都已核实,重叠 ≥ {:.0f} s):{}{}".format(
        OVERLAP_MIN_S, ";".join("`{}`·{} 与 train `{}` 在《{}》里重叠 {:.0f} s{}{}".format(
            x["key"], x["split"], y["key"], md_cell(x["song"]["name"]), ov,
            "" if not x["marks"] else "(" + "、".join(x["marks"]) + ")",
            " —— **{}**([图]({}))".format(x["notes"]["note"], x["notes"].get("evidence", ""))
            if (x.get("notes") or {}).get("note") else "") for x, y, ov in leaks) or "无",
        "。同一段音乐上的同一个舞者,这条 test/val 的动作很可能在训练里见过(是不是同一段编舞要看画面)"
        if leaks else "", ) + ("。**train 还有 {} 条没跑完,此条不完整。**".format(pending) if pending else "。"))
    zero = [c for c in clips.values() if not c["windows"]]
    why = collections.Counter(c["excluded"][1] if c["excluded"] else "(找不到剔除记录)" for c in zero)
    rechecked = [c for c in zero if c["recheck"]]
    L.append("* **release 窗口** = 这条 clip 在当前 release 里切出的 150 帧窗口数(步长 15)。train 的窗口既是训练样本也是"
             "检索库(推理只检索 train);val/test 的窗口只用于评测。**0** 表示整条被 release 剔除,单元格里写了原因。"
             "{} 条 0 窗口的剔除记录来自:{}。".format(len(zero), "; ".join("`{}` {} 条".format(k, n) for k, n in why.items()))
             + ("其中 {} 条事后用更强的 2D-2D 检验复查、峰值在 1.0×,**可能是误杀**:{}(见 `{}`)。".format(
                 len(rechecked), "、".join("`{}`·{}".format(c["key"], c["split"]) for c in rechecked),
                 meta["timebase_decision"]) if rechecked else ""))
    unnamed = [c for c in clips.values() if c["song"]["status"] in UNNAMED]
    read = [c for c in unnamed if (c.get("notes") or {}).get("onscreen")]
    L.append("* **没有歌名的 {} 行**(认不出 / Shazam 分歧 / 出错)里,{} 行从画面上读到了标题或歌词字幕(人工读图,写在歌名栏的小字里;"
             "来源与读法见 `{}`);另有 {} 个上传的画面查过、没有字。".format(
                 len(unnamed), len(read), meta["notes_path"], meta["checked_no_text"]))
    fc = meta["fingerprint"]
    L.append("* **歌名交叉核对**:音频指纹直接配对的 {} 对 clip 里,{} 对两边都认出了名字,其中 {} 对共享 Shazam 曲目{};"
             "据此给 {} 条自己没认出的 clip 推断了歌名(⇢)。".format(
                 fc["pairs"], fc["checked"], fc["agree"],
                 "" if not fc["conflicts"] else ",**{} 对不共享(见文末)**".format(len(fc["conflicts"])), fc["inferred"]))
    L.append("")
    L.append("认定标记:" + " · ".join("{} {}".format(v[0], v[1]) for v in STATUS_SHOW.values()))
    L.append("")
    L.append("其他标记:" + " · ".join("`{}` {}".format(t, d) for t, _, d in MARK_LISTS)
             + " · `新增` 相对 {} 新加进来的 clip".format(meta["previous_name"]))
    L.append("")
    L.append("\"原曲位置\" = clip 在整首歌里的起止 / 全曲长;`×1.05` = clip 比**下载到的那一版**快 5%(可能是抖音加速,"
             "也可能是那一版本身慢了,两者这里分不开);`(?)` 表示副歌重复、位置有两处候选。歌名旁的 `Shazam: …` 是 Shazam 对"
             "同一录音的署名,与显示的(下载版本的)署名不同时列出 —— 两个曲库对同一录音的署名常不一样。"
             "⚠ 行这一格改写名字有多可信:`Shazam 4/4 窗` = 几个识别窗口投给了这首;`偏移一致/分散` = 这些窗口是否把 clip "
             "放在参考录音的同一处(已核实的 clip 里 {} 是一致的,分散多半是副歌重复,只是弱警告);"
             "`本曲候选最好 0.58` = 这首歌的下载候选里与 clip 最像的一版(门槛 0.60,错歌最高 0.43,同曲翻唱 0.45–0.52)。"
             "票数多、偏移一致而分数低,通常是对的录音没被下载到,不是名字错。"
             "发布日期由 aweme id 解出(北京时间)。\"最近渲染\"只算模型产出的视频,不算原片副本或切法对比拼图。".format(
                 meta["coherent_base_rate"]))
    L.append("")
    for s in SPLITS:
        cs = by_split[s]
        L.append("## {}({} 条)".format(s, len(cs)))
        L.append("")
        L.append("| # | 歌名 | 认定 | 原曲位置 | 发布 | 时长 | release 窗口 | 标记 | clip 编码 | 最近渲染 |")
        L.append("|---:|---|:-:|---|---|---:|---|---|---|---|")
        for i, c in enumerate(cs, 1):
            song = c["song"]
            rcell = ""
            if c["renders"]:
                top, rel = c["renders"][0]
                rcell = "[{}]({})".format(top, rel.replace(" ", "%20"))
                if len(c["renders"]) > 1:
                    rcell += " (另 {} 个目录)".format(len(c["renders"]) - 1)
            L.append("| {} | {} | {} | {} | {} | {:.1f}s | {} | {} | `{}` | {} |".format(
                i, song_cell(song) + notes_cell(c), STATUS_SHOW[song["status"]][0], position(song), c["date"], c["dur"],
                windows_cell(c), " ".join("`{}`".format(t) for t in c["marks"]), c["key"], rcell))
        L.append("")
    # song index
    songs = collections.defaultdict(list)
    for c in clips.values():
        if c["song"].get("song_key"):
            songs[c["song"]["song_key"]].append(c)
    L.append("## 按歌索引({} 首,按 Shazam 录音身份分组)".format(len(songs)))
    L.append("")
    L.append("同一首歌的不同版本(原版 / DJ 版 / 伴奏)在 Shazam 是不同的录音,这里分开列。"
             "`跨 split` = 同一录音出现在不止一个 split —— 多数是切分用的音频指纹没把它们连上(两段不重叠),"
             "Shazam 按身份认了出来;指纹连上了却仍跨 split 的见上面\"切分\"一条。只算认出了名字的 clip(含未核实的 ⚠)。")
    L.append("")
    L.append("| 歌名 | test | val | train | 跨 split | clip |")
    L.append("|---|---:|---:|---:|---|---|")
    crossing = collections.Counter()

    def display_name(cs):
        names = collections.Counter(c["song"]["name"] for c in cs if c["song"]["status"] in VERIFIED)
        names = names or collections.Counter(c["song"]["name"] for c in cs)
        return names.most_common(1)[0][0]

    for key, cs in sorted(songs.items(), key=lambda kv: (-len(kv[1]), display_name(kv[1]))):
        n = collections.Counter(c["split"] for c in cs)
        present = [s for s in SPLITS if n[s]]
        if len(present) > 1:
            crossing["+".join(present)] += 1
        L.append("| {} | {} | {} | {} | {} | {} |".format(
            md_cell(display_name(cs)), n["test"] or "", n["val"] or "", n["train"] or "",
            "⚠ " + "+".join(present) if len(present) > 1 else "",
            " ".join("`{}`{}".format(c["key"], "" if c["split"] == "train" else "·" + c["split"])
                     for c in sorted(cs, key=lambda c: (SPLITS.index(c["split"]), c["rid"])))))
    L.append("")
    L.append("跨 split 的录音:" + ("、".join("{} {} 首".format(k, v) for k, v in sorted(crossing.items())) or "无") + (
        "(还有 {} 条 clip 没跑识别,不完整)。".format(pending) if pending else "。"))
    L.append("")
    if fc["conflicts"]:
        L.append("## 指纹配对里不共享 Shazam 曲目的 clip 对")
        L.append("")
        for g in fc["conflicts"]:
            a, b = g["pair"]
            L.append("* `{}` → {};`{}` → {}。指纹对齐分 {},两边共享约 {:.1f} s 同一段音频,Shazam 却给了不同的歌 —— "
                     "至少一边的 Shazam 名字不可信(看两行的票数)。".format(
                         clip_key(a), "; ".join("{} ({} 票)".format(md_cell(t["name"]), t["votes"])
                                                for t in clips[a]["song"]["shazam"]),
                         clip_key(b), "; ".join("{} ({} 票)".format(md_cell(t["name"]), t["votes"])
                                                for t in clips[b]["song"]["shazam"]), g["score"], g["shared_s"]))
        L.append("")
    L.append("## 怎么重算")
    L.append("")
    L.append("```")
    L.append("bash runs/txy_t2_song_menu/run_identify.sh 4     # 识别歌名;已有定论的 clip 跳过,网络出错的会重跑(结果在 {})".format(
        meta["songs_root"]))
    L.append("python3 tools/build_sample_menu.py                # 重写本文件与 {}".format(meta["tsv"]))
    L.append("```")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


TSV_COLS = ("split", "recording_id", "key", "upload_id", "publish_date", "duration_s", "release_windows",
            "excluded_because", "marks", "song_status", "song", "song_key", "align_score", "song_offset_s",
            "song_rate", "song_duration_s", "shazam_tracks", "fingerprint_track", "latest_render", "n_render_dirs")


def write_tsv(path: Path, clips: dict) -> None:
    rows = ["\t".join(TSV_COLS)]
    for c in sorted(clips.values(), key=lambda c: (SPLITS.index(c["split"]), c["rid"])):
        s = c["song"]
        vals = (c["split"], c["rid"], c["key"], c["upload"], c["date"], "{:.2f}".format(c["dur"]), c["windows"],
                c["excluded"][2] if c["excluded"] else "", ",".join(c["marks"]), s["status"], s.get("name", ""),
                s.get("song_key", ""), "{:.3f}".format(s["score"]) if "score" in s else "",
                "{:.2f}".format(s["offset_s"]) if "offset_s" in s else "",
                "{:.3f}".format(s["rate"]) if "rate" in s else "",
                "{:.1f}".format(s["song_s"]) if "song_s" in s else "",
                "; ".join("{} ({} votes)".format(t["name"], t["votes"]) for t in s.get("shazam", [])),
                c["track"] or "", c["renders"][0][1] if c["renders"] else "", len(c["renders"]))
        rows.append("\t".join(" ".join(str(v).split()) for v in vals))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- per raw upload
def raw_menu(clips: dict, raw_root) -> list[dict]:
    """One row per raw upload (the whole Douyin video): its split and the best name any evidence gives.

    Evidence, strongest first: a recording aligned to the whole upload (round 2,
    ``raw_root/<upload>/result.json``) or to any of its clips (round 1); a name the audio
    fingerprint lends from a verified clip; Shazam's word alone; a 《title》 burned into the
    frames.  Two different aligned songs in one upload are both shown (a song switch).
    """
    by_upload = collections.defaultdict(list)
    for c in clips.values():
        by_upload[c["upload"]].append(c)
    rows = []
    for upload, cs in sorted(by_upload.items()):
        splits = {c["split"] for c in cs}
        if len(splits) != 1:
            raise SystemExit("upload {} spans splits {}: the split is upload-level".format(upload, splits))
        cs.sort(key=lambda c: c["rid"])
        roots = raw_root if isinstance(raw_root, (list, tuple)) else [raw_root]
        raws = [read_song(Path(root) / upload / "result.json") for root in roots]
        raw = raws[0]
        songs = raws + [c["song"] for c in cs]
        # a name borrowed through the fingerprint is indirect: it counts only when nothing aligned directly
        # (7629340559064710577: inferred 当你孤单你会想起谁, while its whole audio aligned 春天花会开 0.775
        # and the Douyin description says 春天花会开)
        verified = [x for x in songs if x["status"] in VERIFIED] or [x for x in songs if x["status"] == "inferred"]
        if verified:
            # the best-aligned recording names the video; a weaker "verified" of the SAME title is
            # another version of the composition (0.60-0.75: 7609672652890196474 read 0.604 on one
            # 孤单北半球 and 0.982 on its own), so it only adds a name when its title differs (a switch)
            ranked = sorted(verified, key=lambda x: -x.get("score", 1.0))
            names = [ranked[0]["name"]]
            for x in ranked[1:]:
                if x.get("score", 1.0) >= STRONG and not any(same_song(x["name"], n) for n in names):
                    names.append(x["name"])
            weak = ranked[0].get("score", 1.0) < STRONG
            song, status = " / ".join(names) + ("(版本待定)" if weak else ""), "verified"
        else:
            only = [x for x in songs if x["status"] == "shazam_only"]
            titles = [t for c in cs for t in re.findall(r"《(.+?)》", (c.get("notes") or {}).get("onscreen", ""))]
            near = [x["near_version"] for x in songs if x.get("near_version")]
            douyin = [n for x in songs for n in x.get("douyin_names", [])]
            if near:
                best = max(near, key=lambda v: v["score"])
                song, status = "{}(同曲,版本未找到)".format(best["name"]), "version"
            elif douyin:
                song, status = "{}(抖音识曲)".format(douyin[0]), "douyin"
            elif only:
                best = max(only, key=lambda x: (x["votes"], x["n_windows"]))
                song, status = "{}(未核实)".format(best["name"]), "unverified"
            elif titles:
                song, status = "{}(画面字幕,未核实)".format(titles[0]), "onscreen"
            elif any(x.get("desc_hint") for x in songs):
                hint = next(x["desc_hint"] for x in songs if x.get("desc_hint"))
                song, status = "未识别(抖音描述:#{})".format(hint), "unidentified"
            else:
                song, status = "未识别", "unidentified"
        rows.append({"raw_clip": upload, "split": splits.pop(), "song": " ".join(song.split()), "status": status,
                     "round2": raw["status"], "clips": [c["key"] for c in cs]})
    return rows


def base_title(name: str) -> str:
    """"artist - title (版本) - more" -> the title's core, for telling versions from different songs."""
    title = str(name).split(" - ", 1)[-1].split(" - ")[0]
    return norm(re.sub(r"\s*[\(（\[【].*?[\)）\]】]", "", title))


def same_song(a: str, b: str) -> bool:
    """Two listings of one composition: the cores contain each other or mostly agree
    (孤单北半球 / 孤單北半球, 半分真心 / 何止半分真心 -- scripts and store titles differ)."""
    import difflib

    x, y = base_title(a), base_title(b)
    return bool(x and y) and (x in y or y in x or difflib.SequenceMatcher(None, x, y).ratio() >= 0.6)


def write_raw_tsv(path: Path, rows: list[dict]) -> None:
    lines = ["raw_clip\tsong"] + ["{}\t{}".format(r["raw_clip"], r["song"]) for r in
                                  sorted(rows, key=lambda r: (SPLITS.index(r["split"]), r["raw_clip"]))]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_raw_md(path: Path, rows: list[dict], meta: dict) -> None:
    n = collections.Counter(r["status"] for r in rows)
    L = ["# 汤汤汤小圆 T 线 sample 菜单", "",
         "当前 T 线切分(`configs/t_line/current.env`)的 {} 个原视频(raw clip,即抖音视频 id),每个一行。"
         "歌名已核实 {},同曲但版本未找到 {},只有名字 {},未识别 {}。生成:`python3 tools/build_sample_menu.py`;"
         "同表:[`{}`]({})。".format(len(rows), n["verified"], n["version"], n["douyin"] + n["unverified"] + n["onscreen"],
                                  n["unidentified"], meta["tsv"], meta["tsv"]), "",
         "没有标注的歌名 = 下载到的整首歌与视频音频对齐(同一录音)。\"(版本待定)\" = 对齐 0.60–0.75,歌对、版本可能不同;"
         "\"(同曲,版本未找到)\" = 找到同一首歌的别的版本(对齐 0.45–0.60,错歌最高 0.43),视频用的那版没下载到;"
         "\"(抖音识曲)\" = 抖音自己给这段配乐标的歌名;\"(未核实)\" = 只有 Shazam 的名字(见过认错);"
         "\"(画面字幕,未核实)\" = 视频画面上写的歌名;\"未识别(抖音描述:…)\" = 没有歌名,只列出作品描述里的话题作线索。", ""]
    for s in SPLITS:
        rs = [r for r in rows if r["split"] == s]
        L += ["## {}({} 个)".format(s, len(rs)), ""]
        if s == "test" and meta.get("leak_note"):
            L += [meta["leak_note"], ""]
        L += ["| raw clip | 歌名 |", "|---|---|"]
        L += ["| `{}` | {} |".format(r["raw_clip"], md_cell(r["song"])) for r in rs]
        L.append("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- main
def build(env_file: Path, songs_root: Path, groups_path: Path, output_dir: Path,
          previous_env: Path = DEFAULT_PREVIOUS_ENV, song_split: Path = DEFAULT_SONG_SPLIT,
          timebase_decision: Path = DEFAULT_TIMEBASE_DECISION, notes: Path = DEFAULT_NOTES) -> tuple[dict, dict]:
    env = read_env(env_file)
    release = Path(env["T_RELEASE"])
    sources_path, sources = release_sources(release)
    seqs = {r["sequence_id"]: r for r in read_jsonl(sources_path.with_name("sequences.jsonl"))}
    ingest = Path(env["STRIP_INGEST"])

    windows = collections.Counter()
    for s in SPLITS:
        for name in json.loads((release / s / "names.json").read_text(encoding="utf-8")):
            rid, _, _ = name.rpartition("_slice")
            windows[(rid, s)] += 1
    marks = {}
    for tag, f, _ in MARK_LISTS:
        for line in (REPO / f).read_text(encoding="utf-8").split():
            rid = line if line.startswith("wild_v5:") else "wild_v5:" + line.replace("__", ":")
            marks.setdefault(rid, []).append(tag)
    previous_release = Path(read_env(previous_env)["T_RELEASE"])
    previous = {r["recording_id"] for r in release_sources(previous_release)[1]}
    previous_newest = max(int(rid.split(":")[1]) for rid in previous)  # aweme ids grow with post time
    excluded = exclusions(release)
    recheck = set()
    if timebase_decision.is_file():  # prose side finding: "<key> corr <x>" pairs that read 1.0 on re-check
        side = json.loads(timebase_decision.read_text(encoding="utf-8")).get("side_finding", "")
        if "read 1.0" in side:
            recheck = set(re.findall(r"(\d{19}__clip\d{3}) corr [\d.]+", side))
    groups = json.loads(groups_path.read_text(encoding="utf-8"))
    track_of = groups["track_of"]
    pairs = read_jsonl(REPO / groups["verified_pairs_path"])
    pinned = []
    if song_split.is_file():
        pinned = [set(g["members"]) for g in
                  json.loads(song_split.read_text(encoding="utf-8"))["pinning"].get("preexisting_pinned_conflicts", [])]

    clips = {}
    for r in sources:
        rid, split = r["recording_id"], r["split"]
        key = clip_key(rid)
        seq = seqs[rid]
        wrong = [s for s in SPLITS if s != split and windows[(rid, s)]]
        if wrong:
            raise SystemExit("{} is {} in the manifest but has release windows in {}".format(rid, split, wrong))
        upload = rid.split(":")[1]
        added = rid not in previous
        clip_marks = list(marks.get(rid, [])) + (["新增"] if added else [])
        meta_json = json.loads((ingest / key / "meta.json").read_text(encoding="utf-8"))
        clips[rid] = {"rid": rid, "key": key, "split": split, "upload": upload, "date": publish_date(upload),
                      "source_video": str(REPO / r["provenance"]["source_video"]),
                      "dur": seq["frame_count"] / float(seq["fps"]), "frames": seq["frame_count"],
                      "windows": windows[(rid, split)],
                      "marks": clip_marks, "added": added, "track": track_of.get(rid),
                      "excluded": excluded.get(rid) if not windows[(rid, split)] else None,
                      "recheck": key in recheck and not windows[(rid, split)],
                      "source_size": (meta_json["video_w"], meta_json["video_h"]),
                      "song": read_song(songs_root / key / "result.json")}
    notes_doc = json.loads(notes.read_text(encoding="utf-8")) if notes.is_file() else {}
    hand = notes_doc.get("clips", {})
    by_key = {c["key"]: c for c in clips.values()}
    stale = [k for k in hand if k not in by_key]
    if stale:
        raise SystemExit("{} names clips that are not in this split manifest: {}".format(notes, stale[:5]))
    for k, n in hand.items():
        by_key[k]["notes"] = n
    unknown = [m for m in marks if m not in clips]
    if unknown:
        raise SystemExit("marked clips not in the split manifest: {}".format(unknown[:5]))
    stray = [k for k in track_of if k not in clips]
    if stray:
        raise SystemExit("{} fingerprint-group members are not in this split manifest ({}...): wrong "
                         "--music-groups for this release".format(len(stray), stray[0]))
    by_track = collections.defaultdict(list)
    for rid, t in track_of.items():
        by_track[t].append(rid)
    crossing_groups = [{"track": t, "members": sorted(ms), "pinned": any(set(ms) & p for p in pinned)}
                       for t, ms in sorted(by_track.items()) if len({clips[m]["split"] for m in ms}) > 1]
    fc = fingerprint_check(clips, pairs)
    overlaps = []
    by_song = collections.defaultdict(list)
    for c in clips.values():
        if c["song"]["status"] in VERIFIED and not c["song"].get("partial"):
            by_song[c["song"]["song_key"]].append(c)
    for members in by_song.values():
        span = {c["rid"]: (c["song"]["offset_s"], c["song"]["offset_s"] + c["song"]["rate"] * c["song"]["clip_s"])
                for c in members}
        for x in members:
            for y in members:
                if x["split"] != "train" and y["split"] == "train":
                    ov = min(span[x["rid"]][1], span[y["rid"]][1]) - max(span[x["rid"]][0], span[y["rid"]][0])
                    if ov >= OVERLAP_MIN_S:
                        overlaps.append((x, y, ov))
    overlaps.sort(key=lambda t: (SPLITS.index(t[0]["split"]), t[0]["rid"], t[1]["rid"]))
    judged = [c["song"]["coherent"] for c in clips.values()
              if c["song"]["status"] in VERIFIED and c["song"].get("coherent") is not None]
    renders = scan_renders(output_dir, {c["key"]: c["source_size"] for c in clips.values()})
    for c in clips.values():
        c["renders"] = renders.get(c["key"], [])
    meta = {"generated": datetime.date.today().isoformat(), "release": str(release), "sources": str(sources_path),
            "songs_root": str(songs_root), "fingerprint": fc, "fingerprint_groups_crossing": crossing_groups,
            "fingerprint_threshold": groups["operating_point"]["alignment_threshold"], "position_overlaps": overlaps,
            "coherent_base_rate": "{}/{}".format(sum(judged), len(judged)),
            "notes_path": str(notes.relative_to(REPO)) if notes.is_relative_to(REPO) else str(notes),
            "checked_no_text": len(notes_doc.get("_checked_no_text", [])),
            "previous_name": previous_env.stem.split("_")[0].upper(), "previous_newest": str(previous_newest),
            "timebase_decision":
            str(timebase_decision.relative_to(REPO)) if timebase_decision.is_relative_to(REPO) else str(timebase_decision),
            "tsv": "sample_menu.tsv"}
    return clips, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--env", type=Path, default=REPO / "configs/t_line/current.env")
    ap.add_argument("--previous-env", type=Path, default=DEFAULT_PREVIOUS_ENV,
                    help="the version before; clips it lacks are marked 新增")
    ap.add_argument("--songs-root", type=Path, default=DEFAULT_SONGS,
                    help="per-clip retrieve_full_song outputs, <root>/<key>/result.json")
    ap.add_argument("--music-groups", type=Path, default=DEFAULT_GROUPS,
                    help="audio-fingerprint same-recording groups of this release's bundle")
    ap.add_argument("--song-split", type=Path, default=DEFAULT_SONG_SPLIT,
                    help="assign_wild_song_split report (which cross-split groups were pinned)")
    ap.add_argument("--timebase-decision", type=Path, default=DEFAULT_TIMEBASE_DECISION)
    ap.add_argument("--notes", type=Path, default=DEFAULT_NOTES,
                    help="hand-read facts per clip key: onscreen (burned-in title), note (+ evidence image)")
    ap.add_argument("--raw-root", type=Path, nargs="+", default=[DEFAULT_RAW, DEFAULT_LOCAL],
                    help="results on whole uploads, <root>/<upload id>/result.json (round 2, round 3)")
    ap.add_argument("--detail-tsv", type=Path, default=DEFAULT_DETAIL,
                    help="per-clip evidence table (scores, positions, marks, renders)")
    ap.add_argument("--detail-md", type=Path, default=None,
                    help="also write the long per-clip menu with every explanation (links relative to output/)")
    ap.add_argument("--output-dir", type=Path, default=REPO / "output")
    ap.add_argument("--name", default="sample_menu")
    args = ap.parse_args()
    clips, meta = build(args.env, args.songs_root, args.music_groups, args.output_dir,
                        args.previous_env, args.song_split, args.timebase_decision, args.notes)
    meta["tsv"] = args.name + ".tsv"
    rows = raw_menu(clips, args.raw_root)
    leaks = meta["position_overlaps"]
    if leaks:
        pairs = {}
        for x, y, _ in leaks:
            if x["split"] == "test":
                pairs.setdefault((x["upload"], y["upload"]), False)
                pairs[(x["upload"], y["upload"])] |= bool((x.get("notes") or {}).get("note"))
        if pairs:
            meta["leak_note"] = "注意:" + ";".join("`{}` 与 train 的 `{}` 用了同一段音乐{}".format(
                a, b, "(看过画面:同一段编舞)" if looked else "") for (a, b), looked in sorted(pairs.items())) + "。"
    md = args.output_dir / (args.name + ".md")
    write_raw_md(md, rows, meta)
    write_raw_tsv(args.output_dir / meta["tsv"], rows)
    write_tsv(args.detail_tsv, clips)
    if args.detail_md:
        write_md(args.detail_md, clips, meta)
    st = collections.Counter(r["status"] for r in rows)
    print("wrote {} and {}: {} raw clips; {}; per-clip detail -> {}".format(
        md, meta["tsv"], len(rows), dict(st), args.detail_tsv))
    fc = meta["fingerprint"]
    print("fingerprint cross-check: {}/{} directly paired, both-named pairs share a track; {} inferred".format(
        fc["agree"], fc["checked"], fc["inferred"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
