"""``tools/build_sample_menu.py``: the names next to the sample codes, and how much each is worth.

WHY (2026-09-23):
* a name that was never checked must not look like one that was.  The menu shows
  Shazam's word (unverified) apart from an aligned recording, and the fingerprint
  graph may only *spread* a name from a verified member;
* the fingerprint cross-check has to survive a song switch: a clip that switches
  holds two Shazam tracks, and its neighbour across the switch shares only one of
  them.  The check is per directly-aligned pair, because a fingerprint group is
  single-linkage and its two ends may share no audio at all;
* a name is spread only from a checked, whole, single-song neighbour: a partial or
  switching donor would hand over the song the two clips do NOT share;
* retime_failed means the recording aligned and only the re-timed file failed
  (7456795199654694202 aligned at 0.965): its name is the aligned recording's, not
  Shazam's top vote, and it is not "unverified";
* ``rate`` in retrieve_full_song is song seconds per clip second, so a clip sped up
  5% is ``rate 1.05`` and must print as x1.05, not x0.95.
"""
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools import build_sample_menu as bsm  # noqa: E402


def shazam(*tracks):
    return [{"key": k, "name": "A - " + k, "votes": v} for k, v in tracks]


def clip(status, tracks=(), song_key=None, name=None):
    song = {"status": status, "shazam": shazam(*tracks)}
    if song_key:
        song.update(song_key=song_key, name=name or "A - " + song_key)
    return {"song": song, "frames": 388}


def test_the_publish_date_is_the_top_32_bits_of_the_aweme_id():
    # the newest upload fetched 2026-09-22, and the oldest T2 upload, which is the post the
    # account's end cursor (2021-11-11) stops at
    assert bsm.publish_date("7687981558738172081") == "2026-09-21"
    assert bsm.publish_date("7029295350347287812") == "2021-11-11"


def pair(a, b):
    return {"left": a, "right": b, "score": 0.9}


def test_a_song_switch_is_not_a_conflict():
    clips = {"a": clip("shazam_only", [("x", 2), ("y", 2)], "x"),
             "b": clip("verified", [("y", 3)], "y")}
    fc = bsm.fingerprint_check(clips, [pair("a", "b")])
    assert (fc["checked"], fc["agree"], fc["conflicts"]) == (1, 1, [])


def test_a_component_bridged_by_a_switch_is_not_a_conflict():
    # single-linkage: b and c are in one group only through a; they share no audio
    clips = {"a": clip("verified", [("x", 2), ("y", 2)], "x"), "b": clip("verified", [("x", 3)], "x"),
             "c": clip("verified", [("y", 3)], "y")}
    fc = bsm.fingerprint_check(clips, [pair("a", "b"), pair("a", "c")])
    assert (fc["checked"], fc["agree"], fc["conflicts"]) == (2, 2, [])


def test_disjoint_tracks_in_one_pair_are_reported():
    clips = {"a": clip("verified", [("x", 3)], "x"), "b": clip("verified", [("y", 3)], "y")}
    fc = bsm.fingerprint_check(clips, [pair("a", "b")])
    assert fc["agree"] == 0 and len(fc["conflicts"]) == 1


def test_a_name_is_inferred_only_from_a_verified_single_song_neighbour():
    clips = {"v": clip("verified", [("x", 3)], "x", "张三 - 歌"), "u": clip("unidentified")}
    bsm.fingerprint_check(clips, [pair("v", "u")])
    assert clips["u"]["song"]["status"] == "inferred" and clips["u"]["song"]["name"] == "张三 - 歌"

    # an unchecked Shazam name the pair corroborates is upgraded; one that names another song is not
    clips = {"v": clip("verified", [("x", 3)], "x", "姜云升 - 浪漫主义"), "s": clip("shazam_only", [("x", 3)], "x")}
    bsm.fingerprint_check(clips, [pair("v", "s")])
    assert clips["s"]["song"]["status"] == "inferred" and clips["s"]["song"]["name"] == "姜云升 - 浪漫主义"
    clips = {"v": clip("verified", [("x", 3)], "x"), "s": clip("shazam_only", [("z", 3)], "z")}
    fc = bsm.fingerprint_check(clips, [pair("v", "s")])
    assert clips["s"]["song"]["status"] == "shazam_only" and len(fc["conflicts"]) == 1

    for donor in (clip("shazam_only", [("x", 1)], "x"),              # nobody checked the name
                  clip("verified", [("x", 2), ("y", 2)], "x"),        # a switch: which song is shared?
                  dict(clip("by_windows", [("x", 3)], "x"))):         # partial, set below
        if donor["song"]["status"] == "by_windows":
            donor["song"]["partial"] = True
        clips = {"d": donor, "u": clip("unidentified")}
        fc = bsm.fingerprint_check(clips, [pair("d", "u")])
        assert clips["u"]["song"]["status"] == "unidentified" and fc["inferred"] == 0


def test_the_switch_note_never_repeats_the_shown_song():
    song = {"status": "by_windows", "name": "Y", "song_key": "y", "shazam": shazam(("x", 3), ("y", 2))}
    cell = bsm.song_cell(song)
    assert "A - x" in cell and "A - y" not in cell


def test_the_speed_factor_prints_in_the_direction_the_clip_plays():
    song = {"status": "verified", "offset_s": 60.0, "rate": 1.05, "clip_s": 20.0, "song_s": 200.0}
    assert bsm.position(song) == "1:00–1:21 / 3:20 ×1.05"
    assert bsm.position(dict(song, rate=1.0, ambiguous=True)) == "1:00–1:20 / 3:20 (?)"


def write_result(tmp_path, **r):
    path = tmp_path / "result.json"
    path.write_text(json.dumps(r), encoding="utf-8")
    return path


def test_each_tool_status_maps_to_what_the_menu_claims(tmp_path):
    assert bsm.read_song(tmp_path / "missing.json")["status"] == "not_run"
    assert bsm.read_song(write_result(tmp_path, status="unidentified", shazam_tracks=[],
                                      shazam_windows=[{}]))["status"] == "unidentified"
    track = {"shazam_key": "k", "artist": "LAY", "title": "Veil", "votes": 4}
    unverified = bsm.read_song(write_result(tmp_path, status="unverified", shazam_tracks=[track],
                                            shazam_windows=[{}] * 4, candidates=[]))
    assert unverified["status"] == "shazam_only" and unverified["name"] == "LAY - Veil"
    best = {"videoId": "v", "song": "张艺兴 - 面纱", "score": 0.7, "url": "u", "offset_s": 1.0, "rate": 1.0,
            "clip_duration_s": 20.0, "song_duration_s": 180.0, "window_agreement": {"covers_clip_end": False}}
    by_windows = bsm.read_song(write_result(
        tmp_path, status="verified_by_windows", shazam_tracks=[track], shazam_windows=[{}],
        candidates=[{"videoId": "v", "shazam_key": "k"}], best=best))
    assert by_windows["status"] == "by_windows" and by_windows["partial"] and by_windows["name"] == "张艺兴 - 面纱"
    # the recording aligned; only the re-timed file failed: the name is the aligned one, not Shazam's
    other = {"shazam_key": "o", "artist": None, "title": "Other", "votes": 5}
    retime = bsm.read_song(write_result(
        tmp_path, status="retime_failed", shazam_tracks=[other, track], shazam_windows=[{}] * 7,
        candidates=[{"videoId": "v", "shazam_key": "k"}], best=dict(best, window_agreement={})))
    assert retime["status"] == "retime_failed" and retime["name"] == "张艺兴 - 面纱" and retime["song_key"] == "k"
    assert retime["shazam"][0]["name"] == "Other"  # no "None - " when Shazam has no artist
    # a network error is not "no match"
    errored = bsm.read_song(write_result(tmp_path, status="unidentified", shazam_tracks=[],
                                         shazam_windows=[{"error": "ClientConnectorError"}, {}]))
    assert errored["status"] == "failed"
    (tmp_path / "result.json").write_text('{"status": "verif', encoding="utf-8")
    assert bsm.read_song(tmp_path / "result.json")["status"] == "failed"


CURRENT_ENV = pathlib.Path(bsm.REPO, "configs/t_line/current.env")


@pytest.mark.skipif(not pathlib.Path("/cache/atomicdance-assets").is_dir(), reason="needs the T-line assets")
def test_the_menu_lists_every_clip_of_the_current_release_once(tmp_path):
    clips, meta = bsm.build(CURRENT_ENV, tmp_path / "no_songs_yet", bsm.DEFAULT_GROUPS, tmp_path)
    release = pathlib.Path(meta["release"])
    with_windows = set()
    for split in bsm.SPLITS:
        for name in json.loads((release / split / "names.json").read_text()):
            rid = name.rpartition("_slice")[0]
            assert clips[rid]["split"] == split
            with_windows.add(rid)
    assert {r for r, c in clips.items() if c["windows"]} == with_windows
    assert all(c["song"]["status"] == "not_run" for c in clips.values())  # absent results are shown, not hidden
    eval20 = [c for c in clips.values() if "eval20" in c["marks"]]
    assert len(eval20) == 20 and all(c["split"] == "test" for c in eval20)
    bsm.write_md(tmp_path / "m.md", clips, meta)
    text = (tmp_path / "m.md").read_text(encoding="utf-8")
    assert all("`{}`".format(c["key"]) in text for c in clips.values())


def test_a_name_verified_through_a_store_search_is_not_lost_when_shazam_heard_nothing(tmp_path):
    # 7650126416710192357: Shazam 0/4, named by its Douyin post, NetEase recording aligned at 0.991
    best = {"videoId": "netease:3373426293", "song": "音乐的入门到改行 - 夏天的风 (R&B版)", "score": 0.991,
            "offset_s": 65.1, "rate": 0.999, "clip_duration_s": 22.1, "song_duration_s": 246.0,
            "window_agreement": {"covers_clip_end": True}}
    song = bsm.read_song(write_result(
        tmp_path, status="verified", shazam_tracks=[], shazam_windows=[{"key": None}] * 4,
        candidates=[{"videoId": "netease:3373426293", "shazam_key": None, "score": 0.991}], best=best,
        douyin_post={"desc": "夏天的风_#舞蹈"}, name_queries=["夏天的风"]))
    assert song["status"] == "verified" and song["via"] == "douyin" and song["song_key"] == "store:netease:3373426293"
    assert "抖音" in bsm.song_cell(song)


def test_a_tied_shazam_vote_names_no_song(tmp_path):
    tracks = [{"shazam_key": k, "artist": "A", "title": k, "votes": 1} for k in ("x", "y", "z")]
    song = bsm.read_song(write_result(tmp_path, status="unverified", shazam_tracks=tracks,
                                      shazam_windows=[{}] * 3, candidates=[]))
    assert song["status"] == "shazam_split" and song["name"] == "A - x / A - y / A - z"
    assert bsm.position(song) == "Shazam 3 窗 3 首,各 1 票"


def test_an_unchecked_name_is_qualified_by_its_own_candidates_and_offsets(tmp_path):
    tracks = [{"shazam_key": "x", "artist": "A", "title": "X", "votes": 2},
              {"shazam_key": "y", "artist": "B", "title": "Y", "votes": 1}]
    windows = [{"t0": 0.0, "key": "x", "offset": 50.0}, {"t0": 4.0, "key": "x", "offset": 54.02},
               {"t0": 8.0, "key": "y", "offset": 10.0}]
    cands = [{"shazam_key": "x", "score": 0.28}, {"shazam_key": "y", "score": 0.33}]  # 0.33 is ANOTHER song's
    song = bsm.read_song(write_result(tmp_path, status="unverified", shazam_tracks=tracks,
                                      shazam_windows=windows, candidates=cands))
    assert song["best_score"] == 0.28 and song["coherent"] is True
    assert bsm.position(song) == "Shazam 2/3 窗·偏移一致 · 本曲候选最好 0.28"


def test_cut_comparison_montages_are_not_renders(tmp_path):
    key = "7029295350347287812__clip000"
    for d in ("sample_20260902_cut_bar4", "sample_20260910_real"):
        (tmp_path / d).mkdir()
        (tmp_path / d / (key + ".mp4")).write_bytes(b"not a video")  # ffprobe fails -> size None
    assert [top for top, _ in bsm.scan_renders(tmp_path, {key: (1080, 1920)})[key]] == ["sample_20260910_real"]



def test_a_repeat_that_shazam_places_is_not_shown_as_ambiguous(tmp_path):
    # 面纱 clip001: chroma runner-up within 0.10 elsewhere, but all 4 Shazam windows sit at the chosen place
    track = {"shazam_key": "k", "artist": "LAY", "title": "Veil (Chinese Version)", "votes": 4}
    best = {"videoId": "v", "song": "张艺兴 - 面纱 (中文版)", "score": 0.653, "offset_s": 101.7, "rate": 1.0,
            "clip_duration_s": 22.2, "song_duration_s": 186.6, "window_agreement": {},
            "shazam_offset_minus_aligned_s": [-0.003, -0.016, -0.010, -0.008]}
    kw = dict(shazam_tracks=[track], shazam_windows=[{}] * 4, candidates=[{"videoId": "v", "shazam_key": "k"}])
    song = bsm.read_song(write_result(tmp_path, status="verified_ambiguous_offset", best=best, **kw))
    assert song["status"] == "verified" and not song["ambiguous"]
    assert bsm.position(song) == "1:42–2:04 / 3:07 · 对齐 0.65(Shazam 定位一致)"
    assert "Shazam: LAY - Veil (Chinese Version)" in bsm.song_cell(song)  # two catalogues, two credits

    split = dict(best, shazam_offset_minus_aligned_s=[0.0, 0.0, 35.6])  # In the Club: one window elsewhere
    song = bsm.read_song(write_result(tmp_path, status="verified_ambiguous_offset", best=split, **kw))
    assert song["status"] == "verified_ambiguous" and bsm.position(song).endswith("(?)")


# --------------------------------------------------------------------------- per raw upload
# WHY (2026-09-23): the operator wants one row per raw video with its song, and round 2
# searched whole uploads -- a whole-upload alignment must win over a clip's unchecked name,
# a genuine song switch must show both songs, and a split that is not upload-level is a bug.

def raw_clip(upload, n, split, song, notes=None):
    return {"rid": "wild_v5:{}:clip{:03d}".format(upload, n), "key": "{}__clip{:03d}".format(upload, n),
            "upload": upload, "split": split, "song": song, "notes": notes}


def test_the_whole_upload_alignment_names_the_raw_clip(tmp_path):
    best = {"videoId": "kuwo:1", "song": "徐良,小凌 - 坏女孩", "score": 0.9, "offset_s": 10.0, "rate": 1.0,
            "clip_duration_s": 17.5, "song_duration_s": 200.0, "window_agreement": {}}
    (tmp_path / "7664973324456613370").mkdir()
    write_result(tmp_path / "7664973324456613370", status="verified", shazam_tracks=[], shazam_windows=[{}] * 3,
                 candidates=[{"videoId": "kuwo:1", "shazam_key": None, "store": "kuwo"}], best=best)
    clips = {"a": raw_clip("7664973324456613370", 0, "test", {"status": "unidentified", "shazam": []})}
    rows = bsm.raw_menu(clips, tmp_path)
    assert [(r["raw_clip"], r["song"], r["status"]) for r in rows] == [("7664973324456613370", "徐良,小凌 - 坏女孩", "verified")]


def test_raw_clip_rows_combine_their_clips(tmp_path):
    v = lambda name: {"status": "verified", "name": name, "shazam": []}
    clips = {"a": raw_clip("1", 0, "train", v("A - X")), "b": raw_clip("1", 1, "train", v("A - Y")),
             "c": raw_clip("2", 0, "val", {"status": "shazam_only", "name": "B - Z", "votes": 3, "n_windows": 4,
                                           "shazam": []}),
             "d": raw_clip("3", 0, "val", {"status": "unidentified", "shazam": []}, {"onscreen": "《睫毛弯弯》 编舞:x"}),
             "e": raw_clip("4", 0, "test", {"status": "unidentified", "shazam": []})}
    rows = {r["raw_clip"]: r["song"] for r in bsm.raw_menu(clips, tmp_path)}
    assert rows == {"1": "A - X / A - Y", "2": "B - Z(未核实)", "3": "睫毛弯弯(画面字幕,未核实)", "4": "未识别"}
    bsm.write_raw_tsv(tmp_path / "m.tsv", bsm.raw_menu(clips, tmp_path))
    lines = (tmp_path / "m.tsv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "raw_clip\tsong" and all(len(l.split("\t")) == 2 for l in lines) and len(lines) == 5

    clips["f"] = raw_clip("1", 2, "test", v("A - X"))
    with pytest.raises(SystemExit):
        bsm.raw_menu(clips, tmp_path)


def test_a_weak_version_does_not_join_the_recording_that_aligned(tmp_path):
    # 7609672652890196474: h3b3's 孤单北半球 (R&B版) 0.604 on a clip, 欧得洋's AI cover 0.982 on the whole video
    v = lambda name, score: {"status": "verified", "name": name, "score": score, "shazam": []}
    clips = {"a": raw_clip("1", 0, "train", v("h3b3 - 孤单北半球 (R&B版)", 0.604)),
             "b": raw_clip("2", 0, "train", v("h3b3 - 孤单北半球 (R&B版)", 0.604)),
             "c": raw_clip("2", 1, "train", v("欧得洋 - 孤单北半球 (AI Cover)", 0.982)),
             "d": raw_clip("3", 0, "train", v("A - X", 0.95)), "e": raw_clip("3", 1, "train", v("B - Y", 0.90))}
    rows = {r["raw_clip"]: r["song"] for r in bsm.raw_menu(clips, tmp_path)}
    assert rows == {"1": "h3b3 - 孤单北半球 (R&B版)(版本待定)", "2": "欧得洋 - 孤单北半球 (AI Cover)",
                    "3": "A - X / B - Y"}


def test_an_unverified_video_is_named_by_the_strongest_evidence_below_the_gate(tmp_path):
    # 7646787035172290481: nothing aligned >= 0.60, but a 疑心病 version aligned 0.51 and Douyin
    # named the sound 疑心病（陆挚连 Remix）; 7648...: only Douyin's name; 7078...: only Shazam's
    base = {"status": "unidentified", "shazam": [], "douyin_names": []}
    clips = {"a": raw_clip("1", 0, "train", dict(base, douyin_names=["疑心病（陆挚连 Remix）"],
                                                  near_version={"name": "x - 疑心病 (DJ版)", "score": 0.51})),
             "b": raw_clip("2", 0, "train", dict(base, douyin_names=["爱的初体验(r&b)"])),
             "c": raw_clip("3", 0, "train", {"status": "shazam_only", "name": "G-DRAGON - A Boy", "votes": 3,
                                             "n_windows": 6, "shazam": [], "douyin_names": []})}
    rows = {r["raw_clip"]: (r["song"], r["status"]) for r in bsm.raw_menu(clips, tmp_path)}
    assert rows == {"1": ("x - 疑心病 (DJ版)(同曲,版本未找到)", "version"),
                    "2": ("爱的初体验(r&b)(抖音识曲)", "douyin"),
                    "3": ("G-DRAGON - A Boy(未核实)", "unverified")}

