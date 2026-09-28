#!/usr/bin/env python3
"""Where a clip's song is NAMED and where its full length is FETCHED, beyond Shazam+YouTube.

``retrieve_full_song.py`` identifies with Shazam and fetches from YouTube Music.  That
pair failed completely on one T2 clip and the failure is structural, not incidental:

  7650126416710192357 (夏天的风, 猴哥 choreography).  Shazam returned no match on 4/4
  windows of the clip's audio AND on 4/4 windows of the clean 24 s sound the post
  carries -- the recording is a choreographer's edit, which no fingerprint database
  holds.  The Douyin post names it anyway, in its DESCRIPTION
  ("夏天的风_#舞蹈#hiphop_#夏天的风#猴哥编舞"): the music ENTRY is only "猴哥_ 的原声".
  NetEase then has 87 versions of that name, most of them the Douyin remixes.

So the two stages come apart, and each gets its own sources:

* **naming** -- Shazam (a released master), or the Douyin post: its music entry
  (title/author) plus the hashtags and free text of the description, which is where a
  choreographer's own edit is named.  Naming is a *hint*: it never decides.
* **fetching** -- YouTube Music (``retrieve_full_song``) or NetEase, whose catalogue
  carries the Chinese covers and the Douyin remixes that YouTube Music does not.
* **deciding** -- unchanged: ``retrieve_full_song.align`` over (rate, offset) with the
  0.60 whole-clip gate, calibrated so a cover of the right song (0.45-0.69) is refused.
  With 87 same-name candidates that gate is the only thing standing between a
  continuation on the dancer's own recording and one on somebody else's cover.

Douyin is an identification source ONLY.  The sound attached to a post is a cut --
24.3 s here against a 22.1 s clip (they align at 0.998, rate 1.000, 7/7 windows), so it
holds no continuation music at all.  The full length has to come from a music store.

The Douyin call needs the repo's browser cookie (``cookie/www_douyin_com_cookies.json``)
and ``f2`` for the request signing; f2 is not in requirements.txt, install it per the
memory note into a scratch dir (``PIP_CONFIG_FILE=/dev/null``, the nexus mirror).
The cookie is read here and goes no further than the request.

Usage::

    python3 tools/song_stores.py douyin 7650126416710192357          # name it
    python3 tools/song_stores.py netease "夏天的风" --limit 30 --out DIR   # fetch candidates
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import html
import json
import os
import pathlib
import re
import subprocess
import sys
import urllib.parse

REPO = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_COOKIE = REPO / "cookie/www_douyin_com_cookies.json"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/126.0.0.0 Safari/537.36")
# NetEase answers a VIP/unavailable track with a 107 KB placeholder instead of an error
NETEASE_MIN_BYTES = 200_000
# a store file shorter than this is a preview or a stub, not the song (search() already
# keeps only listings of 60 s or more)
MIN_SONG_S = 60.0


# --------------------------------------------------------------------------- naming
def _cookie_header(path: pathlib.Path) -> str:
    jar = json.load(open(path, encoding="utf-8"))
    return "; ".join("{}={}".format(c["name"], c["value"]) for c in jar if "douyin" in (c.get("domain") or ""))


def douyin_post(aweme_id: str, cookie: pathlib.Path = DEFAULT_COOKIE) -> dict:
    """The post's music entry, description, Douyin's own song match and its Qishui link.

    Needs f2 on sys.path and the cookie.  Read from the RAW aweme detail, because the
    fields that name a song behind "@xxx创作的原声" are not in f2's flattened dict:

    * ``music.matched_pgc_sound`` -- Douyin's own recognition of the sound (7663487196943700657:
      "拖拉机来咯（力量压迫）", where Shazam had said 不逢海 - 无人车站 and no store held it);
    * ``related_music_anchor`` (type luna_anchor_v2) -- the "汽水音乐" button under the video:
      ``luna_song_id`` is the Qishui (汽水音乐) track that plays the whole song.
    """
    from f2.apps.douyin.handler import DouyinHandler

    kwargs = {"headers": {"User-Agent": UA, "Referer": "https://www.douyin.com/"},
              "proxies": {"http://": os.environ.get("HTTP_PROXY"), "https://": os.environ.get("HTTPS_PROXY")},
              "cookie": _cookie_header(cookie), "timeout": 30}
    raw = asyncio.run(DouyinHandler(kwargs).fetch_one_video(aweme_id=aweme_id))._to_raw()
    d = raw.get("aweme_detail") or {}
    m = d.get("music") or {}
    pgc = m.get("matched_pgc_sound") or {}
    anchor = d.get("related_music_anchor") or {}
    try:
        extra = json.loads(anchor.get("extra") or "{}")
    except ValueError:
        extra = {}
    luna = extra.get("luna_sid") or (re.search(r"luna_song_id=(\d+)", anchor.get("schema_url") or "") or [None, None])[1]
    return {"aweme_id": aweme_id, "desc": d.get("desc"), "create_time": d.get("create_time"),
            "duration": d.get("duration"), "music_title": m.get("title"), "music_author": m.get("author"),
            "music_id": m.get("id_str"), "music_mid": m.get("mid"), "music_duration": m.get("duration"),
            "music_play_url": ((m.get("play_url") or {}).get("url_list") or [None])[0],
            "is_commerce_music": m.get("is_commerce_music"),
            "matched_song_title": pgc.get("title"), "matched_song_author": pgc.get("author"),
            "qishui_track_id": str(luna) if luna else None,
            "qishui_title": extra.get("title") if luna else None, "qishui_author": extra.get("author") if luna else None}


def song_names(post: dict) -> list[str]:
    """Search queries the post suggests, best first.

    The hashtags are where a choreographer's edit is named; the music entry is often only
    "<user> 的原声".  Tags that name the dance rather than the song (舞蹈/编舞/翻跳/hiphop
    and friends) are dropped, and the leading free text is kept because the description
    of 7650126416710192357 starts with the song's name before any tag.
    """
    out = []
    for name in (post.get("qishui_title"), post.get("matched_song_title")):  # Douyin's own song match
        if name and "原声" not in name and name not in out:
            out.append(name)
    desc = post.get("desc") or ""
    noise = re.compile(r"舞蹈|编舞|翻跳|原创|hiphop|jazz|kpop|dance|挑战|随拍|日常|vlog", re.I)
    # strip the dance words out of a tag instead of dropping it: "#蔚蓝海岸舞蹈" names the song
    # 蔚蓝海岸 (7102423611985726732: 微音 - 蔚蓝海岸 aligned 0.983 once it was searched)
    tags = [noise.sub("", t).strip() for t in re.findall(r"#([^#\s_]+)", desc)]
    tags = [t for t in tags if len(t) >= 2]
    head = re.split(r"[#_\s]", desc.strip())[0] if desc else ""
    for name in [head] + tags[:1 if out or head else 2]:
        # every tag x store x top-30 was up to 496 candidates per video (35 s each); the song is
        # named by Douyin's own match or the description's head in the cases seen (7650, 7609)
        if name and len(name) >= 2 and not noise.search(name) and name not in out:
            out.append(name)
    title, author = post.get("music_title"), post.get("music_author")
    if title and "原声" not in title:
        out.append("{} {}".format(author or "", title).strip())
    return out


# --------------------------------------------------------------------------- fetching
def _curl_json(url: str, referer: str = "") -> dict:
    cmd = ["curl", "-s", "-m", "30", "-A", UA]
    if referer:
        cmd += ["-e", referer]
    return json.loads(subprocess.check_output(cmd + [url]))


def netease_search(query: str, limit: int = 30) -> list[dict]:
    url = ("https://music.163.com/api/search/get/web?csrf_token=&s={}&type=1&offset=0&total=true&limit={}"
           .format(urllib.parse.quote(query), limit))
    songs = (_curl_json(url, "https://music.163.com").get("result") or {}).get("songs") or []
    return [{"store": "netease", "id": str(s["id"]), "name": s["name"],
             "artist": ",".join(a["name"] for a in s["artists"]),
             "duration_s": round(s["duration"] / 1000),
             "url": "https://music.163.com/song/{}".format(s["id"])} for s in songs]


def netease_download(song_id: str, dest_dir: pathlib.Path) -> pathlib.Path | None:
    """The public outer URL.  A VIP or removed track answers 200 with a stub, not an error."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "netease_{}.mp3".format(song_id)
    if dest.is_file() and dest.stat().st_size >= NETEASE_MIN_BYTES:
        return dest
    subprocess.run(["curl", "-s", "-m", "120", "-L", "-A", UA, "-e", "https://music.163.com", "-o", str(dest),
                    "https://music.163.com/song/media/outer/url?id={}.mp3".format(song_id)],
                   check=False, stdin=subprocess.DEVNULL)
    if not dest.is_file() or dest.stat().st_size < NETEASE_MIN_BYTES:
        if dest.is_file():
            dest.unlink()
        return None
    return dest


def kuwo_search(query: str, limit: int = 30) -> list[dict]:
    """Kuwo's public search.  It answers in a Python-literal dialect (single quotes), not JSON."""
    url = ("http://search.kuwo.cn/r.s?all={}&ft=music&itemset=web_2013&client=kt&pn=0&rn={}"
           "&rformat=json&encoding=utf8".format(urllib.parse.quote(query), limit))
    text = subprocess.check_output(["curl", "-s", "-m", "30", "-A", UA, url]).decode("utf-8", "replace")
    data = ast.literal_eval(text)

    def clean(x):  # its escapes survive literal_eval: "R\\u0026B", "&nbsp;"
        x = re.sub(r"\\+u0026", "&", str(x))
        return html.unescape(re.sub(r"\\+&", "&", x)).replace("\xa0", " ")

    return [{"store": "kuwo", "id": s["MUSICRID"].replace("MUSIC_", ""), "name": clean(s["SONGNAME"]),
             "artist": clean(s["ARTIST"]), "duration_s": int(s.get("DURATION") or 0),
             "url": "https://www.kuwo.cn/play_detail/" + s["MUSICRID"].replace("MUSIC_", "")}
            for s in data.get("abslist") or [] if s.get("MUSICRID")]


def kuwo_download(rid: str, dest_dir: pathlib.Path) -> pathlib.Path | None:
    """Full-length 128 kbps mp3 via the convert_url endpoint (probed 2026-09-23: a 195 s track came
    back whole).  A track Kuwo will not serve answers with no URL; a short file is a preview."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "kuwo_{}.mp3".format(rid)
    if dest.is_file() and _duration_s(dest) >= MIN_SONG_S:
        return dest
    url = subprocess.run(["curl", "-s", "-m", "30", "-A", UA,
                          "http://antiserver.kuwo.cn/anti.s?type=convert_url&rid=MUSIC_{}&format=mp3&response=url"
                          .format(rid)], capture_output=True, text=True, stdin=subprocess.DEVNULL).stdout.strip()
    if not url.startswith("http"):
        return None
    subprocess.run(["curl", "-s", "-m", "120", "-L", "-A", UA, "-o", str(dest), url],
                   check=False, stdin=subprocess.DEVNULL)
    if not dest.is_file() or _duration_s(dest) < MIN_SONG_S:
        if dest.is_file():
            dest.unlink()
        return None
    return dest


def _duration_s(path: pathlib.Path) -> float:
    try:
        return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                              "-of", "csv=p=0", str(path)], stdin=subprocess.DEVNULL).strip())
    except (subprocess.CalledProcessError, ValueError):
        return 0.0


def names_from_lyrics(line: str, limit: int = 5) -> list[str]:
    """Song names whose lyrics contain ``line``: NetEase lyric search (type 1006), then QQ Music's
    (t=7).  For a video that carries only lyric subtitles; like every name, only a hint."""
    names = []
    try:
        url = ("https://music.163.com/api/search/get/web?csrf_token=&s={}&type=1006&offset=0&limit={}"
               .format(urllib.parse.quote(line), limit))
        for s in (_curl_json(url, "https://music.163.com").get("result") or {}).get("songs") or []:
            names.append("{} {}".format(",".join(a["name"] for a in s.get("artists") or []), s["name"]).strip())
    except Exception as error:
        print("  netease lyric search failed for {!r}: {}".format(line, error), file=sys.stderr)
    try:
        url = ("https://c.y.qq.com/soso/fcgi-bin/client_search_cp?p=1&n={}&t=7&w={}&format=json"
               .format(limit, urllib.parse.quote(line)))
        for s in ((_curl_json(url, "https://y.qq.com").get("data") or {}).get("lyric") or {}).get("list") or []:
            names.append("{} {}".format(",".join(a["name"] for a in s.get("singer") or []), s["songname"]).strip())
    except Exception as error:
        print("  qq lyric search failed for {!r}: {}".format(line, error), file=sys.stderr)
    out = []
    for n in names:
        if n and n not in out:
            out.append(n)
    return out[:2 * limit]


def qishui_track(track_id: str) -> dict | None:
    """A Qishui (汽水音乐) track from its public share page: name, artist, and the playable audio.

    The page's ``_ROUTER_DATA`` carries ``loaderData.track_page.audioWithLyricsOption``; for
    a free track ``previewEnd`` equals the duration and the URL is unencrypted (checked
    2026-09-23 on 7670384942190954497: 120.68 s, encrypt false).  The URL expires, so it is
    read again right before a download."""
    ua = "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6"
    page = subprocess.run(["curl", "-s", "-m", "30", "-A", ua,
                           "https://music.douyin.com/qishui/share/track?track_id={}".format(track_id)],
                          capture_output=True, text=True, stdin=subprocess.DEVNULL).stdout
    i = page.find("_ROUTER_DATA")
    if i < 0:
        return None
    try:
        data, _ = json.JSONDecoder().raw_decode(page[page.index("{", i):])
        a = data["loaderData"]["track_page"]["audioWithLyricsOption"]
    except (ValueError, KeyError, TypeError):
        return None
    if a.get("encrypt") or not a.get("url"):
        return None
    return {"store": "qishui", "id": str(track_id), "name": a.get("trackName"), "artist": a.get("artistName"),
            "duration_s": float(a.get("duration") or 0), "preview_end_s": float(a.get("previewEnd") or 0),
            "audio_url": a["url"], "url": "https://music.douyin.com/qishui/share/track?track_id={}".format(track_id)}


def qishui_download(track_id: str, dest_dir: pathlib.Path) -> pathlib.Path | None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "qishui_{}.m4a".format(track_id)
    if dest.is_file() and _duration_s(dest) >= 20.0:
        return dest
    t = qishui_track(track_id)
    if not t:
        return None
    subprocess.run(["curl", "-s", "-m", "120", "-L", "-A", UA, "-o", str(dest), t["audio_url"]],
                   check=False, stdin=subprocess.DEVNULL)
    if not dest.is_file() or _duration_s(dest) < 20.0:  # a Qishui preview can be short; the gate decides
        if dest.is_file():
            dest.unlink()
        return None
    return dest


SEARCHERS = {"netease": netease_search, "kuwo": kuwo_search}


def search(queries: list[str], stores: list[str], limit: int = 30, max_song_s: float = 900.0) -> list[dict]:
    """Candidates from every store, de-duplicated, songs only (60 s .. max_song_s)."""
    out, seen = [], set()
    for query in queries:
        for store in stores:
            try:
                hits = SEARCHERS.get(store, _ytmusic_search)(query, limit)
            except Exception as error:                      # one store being down is not fatal
                print("  {} search failed for {!r}: {}".format(store, query, error), file=sys.stderr)
                continue
            for hit in hits:
                key = (hit["store"], hit["id"])
                if key in seen or not (60 <= (hit.get("duration_s") or 0) <= max_song_s):
                    continue
                seen.add(key)
                out.append(dict(hit, query=query))
    return out


def _ytmusic_search(query: str, limit: int) -> list[dict]:
    from ytmusicapi import YTMusic

    return [{"store": "ytmusic", "id": r["videoId"], "name": r.get("title"),
             "artist": ",".join(a["name"] for a in r.get("artists") or []),
             "duration_s": r.get("duration_seconds") or 0,
             "url": "https://music.youtube.com/watch?v=" + r["videoId"]}
            for r in YTMusic().search(query, filter="songs", limit=limit)[:limit] if r.get("videoId")]


def download(candidate: dict, dest_dir: pathlib.Path) -> pathlib.Path | None:
    if candidate["store"] == "netease":
        return netease_download(candidate["id"], dest_dir)
    if candidate["store"] == "kuwo":
        return kuwo_download(candidate["id"], dest_dir)
    if candidate["store"] == "qishui":
        return qishui_download(candidate["id"], dest_dir)
    sys.path.insert(0, str(REPO / "tools"))
    import retrieve_full_song as rfs

    wav, _ = rfs.download(candidate["id"], dest_dir)
    return wav


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("douyin", help="name a post's song from its music entry and description")
    d.add_argument("aweme_id")
    d.add_argument("--cookie", type=pathlib.Path, default=DEFAULT_COOKIE)
    n = sub.add_parser("netease", help="search and download candidates")
    n.add_argument("query")
    n.add_argument("--limit", type=int, default=30)
    n.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()

    if args.cmd == "douyin":
        post = douyin_post(args.aweme_id, args.cookie)
        print(json.dumps({"post": post, "search_queries": song_names(post)}, ensure_ascii=False, indent=1))
        return 0
    hits = netease_search(args.query, args.limit)
    for h in hits:
        line = "{id:>10}  {name:<30.30}  {artist:<20.20} {duration_s:>4}s".format(**h)
        if args.out:
            got = netease_download(h["id"], args.out)
            line += "  -> {}".format(got.name if got else "unavailable (VIP/removed)")
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
