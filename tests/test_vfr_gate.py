"""The frame-rate gate: does a container's two rate claims agree?

The defect being gated is that ``cut_clip`` selects the picture by frame number
and the sound by seconds.  Those name the same span only when the container's
``avg_frame_rate`` and ``r_frame_rate`` agree; when they do not, one upload
measured 2026-08-24 put its picture at 27.17-54.37 s of the source and its
sound at 13.63 s for 13.65 s.
"""

import json
import pathlib
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.ingest_wild_uploads import is_variable_rate, probe_rates


def _ffprobe_json(payload):
    return SimpleNamespace(stdout=json.dumps(payload), stderr="", returncode=0)


def test_the_rates_are_read_by_name_not_by_position():
    # ffprobe emits stream entries in the order the stream carries them, not the
    # order they were requested.  Positional parsing therefore returns the two
    # rates the wrong way round on some files -- which inverts every downstream
    # verdict while still reading like a measurement.  Both orders must give the
    # same answer.
    forward = {"streams": [{"avg_frame_rate": "3636/115", "r_frame_rate": "60/1"}]}
    reversed_ = {"streams": [{"r_frame_rate": "60/1", "avg_frame_rate": "3636/115"}]}
    for payload in (forward, reversed_):
        with mock.patch("tools.ingest_wild_uploads.run",
                        return_value=_ffprobe_json(payload)):
            avg, rate = probe_rates(pathlib.Path("x.mp4"))
        assert round(avg, 3) == 31.617
        assert rate == 60.0


def test_an_unreadable_or_absurd_rate_is_zero_rather_than_a_guess():
    for payload in ({"streams": [{"avg_frame_rate": "0/0", "r_frame_rate": "0/0"}]},
                    {"streams": []},
                    {}):
        with mock.patch("tools.ingest_wild_uploads.run",
                        return_value=_ffprobe_json(payload)):
            assert probe_rates(pathlib.Path("x.mp4")) == (0.0, 0.0)


def test_variable_rate_is_relative_and_needs_both_rates():
    assert is_variable_rate(31.617, 60.0) is True
    assert is_variable_rate(25.0, 30.0) is True        # the 1.2x family
    assert is_variable_rate(30.0, 30.0) is False
    assert is_variable_rate(29.97, 30.0) is False      # NTSC is not this defect
    # A missing rate is not evidence of disagreement; it is evidence of nothing,
    # and must not be reported as a clean file OR as a broken one by this call.
    assert is_variable_rate(0.0, 60.0) is False
    assert is_variable_rate(60.0, 0.0) is False


def _make_clip(path, rate, seconds=1):
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error",
                    "-f", "lavfi", "-i",
                    "testsrc=size=64x64:rate={}:duration={}".format(rate, seconds),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
                   check=True)


def test_a_constant_rate_upload_is_not_selected_and_the_rejects_are_reported(capsys):
    # The census must say what it did NOT select: "we found none" and "we could
    # not look" are the same line otherwise.
    from tools.select_vfr_uploads import main
    with tempfile.TemporaryDirectory() as raw:
        tmp = pathlib.Path(raw)
        videos = tmp / "videos"
        videos.mkdir()
        _make_clip(videos / "111.mp4", 30)
        out = tmp / "vfr.json"
        assert main(["--videos-dir", str(videos), "--output", str(out)]) == 0
        report = json.loads(out.read_text())
        assert report["checked"] == 1 and report["readable"] == 1
        assert report["selected_uploads"] == 0
        printed = capsys.readouterr().out
        assert "not selected: 1 upload(s)" in printed


def test_a_missing_upload_is_counted_rather_than_passing_as_clean():
    from tools.select_vfr_uploads import main
    with tempfile.TemporaryDirectory() as raw:
        tmp = pathlib.Path(raw)
        videos = tmp / "videos"
        videos.mkdir()
        _make_clip(videos / "111.mp4", 30)
        manifest = tmp / "sources.jsonl"
        manifest.write_text(
            "\n".join(json.dumps({"recording_id": "wild_v4:%s:clip000" % u})
                      for u in ("111", "222")) + "\n", encoding="utf-8")
        out = tmp / "vfr.json"
        main(["--videos-dir", str(videos), "--uploads-from", str(manifest),
              "--output", str(out)])
        report = json.loads(out.read_text())
        assert report["checked"] == 2
        assert report["readable"] == 1
        assert report["missing"] == 1


def test_the_clip_index_maps_already_cut_clips_onto_their_upload():
    from tools.select_vfr_uploads import clips_by_upload
    with tempfile.TemporaryDirectory() as raw:
        tmp = pathlib.Path(raw)
        for name in ("wild_v4:111:clip000.npy", "wild_v4:111:clip001.npy",
                     "wild_v4:222:clip000.npy", "not-a-clip.npy"):
            (tmp / name).write_bytes(b"")
        mapping = clips_by_upload(tmp)
        assert mapping["111"] == ["wild_v4:111:clip000", "wild_v4:111:clip001"]
        assert mapping["222"] == ["wild_v4:222:clip000"]
        assert "not-a-clip" not in mapping


@pytest.mark.parametrize("rate", (25, 30, 50))
def test_cfr_normalisation_produces_a_file_whose_rates_agree(rate):
    # The fix has to be checked on its output, not on its command line: an
    # ffmpeg invocation that returns 0 and leaves the defect in place would
    # otherwise be recorded as a repair.
    from tools.ingest_wild_uploads import normalize_to_cfr
    with tempfile.TemporaryDirectory() as raw:
        tmp = pathlib.Path(raw)
        source = tmp / "src.mp4"
        _make_clip(source, rate)
        out = normalize_to_cfr(source, tmp / "cfr" / "src.mp4", float(rate))
        assert out is not None and out.is_file()
        avg, r = probe_rates(out)
        assert not is_variable_rate(avg, r)
        assert abs(avg - rate) < 0.5


# --------------------------------------------------------------------------
# The end gate of tools/refix_wild_fps_clips.sh.
#
# The gate it replaced asked "does any clip record a source_fps?".  That is the
# shape CLAUDE.md §2 names: it reads like a check and could not return no,
# because the 2026-08-19 run had already written the field into 22 clips.  The
# tests below therefore include the exact corpus that fooled it.
# --------------------------------------------------------------------------

def _manifest(root, rows):
    """Write ingest_shard0.jsonl the way tools/ingest_wild_uploads.py does."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / "ingest_shard0.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _census(clips):
    return {"clips": {name: dict(row, clip=name) for name, row in clips.items()}}


def _row(upload, at, status="ok", clips=()):
    return {"upload": upload, "ingested_at": at, "status": status,
            "clips": [{"clip": c, "status": "ok"} for c in clips]}


def test_a_run_that_ingested_nothing_is_refused_however_old_rows_read():
    # 2026-08-19: seven shards printed "1335 of 1335 already recorded", did no
    # work and exited 0.  Every row in the manifest predates the run.
    from tools.check_recut_happened import check
    with tempfile.TemporaryDirectory() as raw:
        root = pathlib.Path(raw) / "ingest"
        _manifest(root, [_row("111", 1000.0, clips=["111__clip000"])])
        census = _census({"111__clip000": {"produced_now": True, "rates_agree": True}})
        report = check(census, root, ["111"], since=2000.0)
        assert report["passed"] is False
        assert report["redo_uploads_touched"] == 0
        assert any("no upload named by --redo" in r for r in report["reasons"])


def test_the_gate_the_old_one_could_not_fail_a_run_refused_for_want_of_cfr_cache():
    # Without --cfr-cache every variable-rate upload is refused and nothing is
    # cut, while the 22 clips re-cut on 2026-08-19 still carry source_fps.  The
    # old gate printed "22 of 230" and passed; this one has to refuse.
    from tools.check_recut_happened import check
    with tempfile.TemporaryDirectory() as raw:
        root = pathlib.Path(raw) / "ingest"
        _manifest(root, [_row("111", 3000.0, status="variable_frame_rate")])
        # ...and the census still shows last generation's clips, which record a
        # source_fps and no rate pair at all.
        census = _census({"111__clip000": {"produced_now": True,
                                           "records_source_fps": True,
                                           "rates_agree": None}})
        report = check(census, root, ["111"], since=2000.0)
        assert report["passed"] is False
        assert report["refused"] == 1
        assert any("--cfr-cache" in r for r in report["reasons"])
        # and the old gate's own reading, for the record: it would have passed.
        assert census["clips"]["111__clip000"]["records_source_fps"] is True


def test_a_clip_still_cut_from_a_disagreeing_container_is_refused():
    from tools.check_recut_happened import check
    with tempfile.TemporaryDirectory() as raw:
        root = pathlib.Path(raw) / "ingest"
        _manifest(root, [_row("111", 3000.0, clips=["111__clip000"])])
        census = _census({"111__clip000": {"produced_now": True, "rates_agree": False}})
        report = check(census, root, ["111"], since=2000.0)
        assert report["passed"] is False
        assert report["rates_disagree"] == 1
        assert "111__clip000" in report["disagreeing_examples"]


def test_never_measured_is_not_clean():
    # A clip with no rate pair was cut before the pair existed.  "Nobody looked"
    # must not read as "it is fine" -- that conflation is what let 208 clips sit
    # in the corpus uncut because the 2026-08-19 selection could not see them.
    from tools.check_recut_happened import check
    with tempfile.TemporaryDirectory() as raw:
        root = pathlib.Path(raw) / "ingest"
        _manifest(root, [_row("111", 3000.0, clips=["111__clip000"])])
        census = _census({"111__clip000": {"produced_now": True, "rates_agree": None}})
        report = check(census, root, ["111"], since=2000.0)
        assert report["passed"] is False
        assert report["rates_never_measured"] == 1


def test_an_orphan_does_not_hold_the_gate_open():
    # A clip nobody produces any more keeps its old meta forever and can never
    # gain a rate pair.  Judging it would make the gate unpassable rather than
    # strict, so produced_now False is excluded -- and the orphan list, not this
    # gate, is what keeps it out of the corpus.
    from tools.check_recut_happened import check
    with tempfile.TemporaryDirectory() as raw:
        root = pathlib.Path(raw) / "ingest"
        _manifest(root, [_row("111", 3000.0, clips=["111__clip000"])])
        census = _census({"111__clip000": {"produced_now": True, "rates_agree": True},
                          "111__clip001": {"produced_now": False, "rates_agree": None}})
        report = check(census, root, ["111"], since=2000.0)
        assert report["passed"] is True
        assert report["produced_clips"] == 1


def test_a_clean_run_passes_and_says_what_it_did_not_check(capsys):
    from tools.check_recut_happened import main
    with tempfile.TemporaryDirectory() as raw:
        tmp = pathlib.Path(raw)
        root = tmp / "ingest"
        _manifest(root, [_row("111", 3000.0, clips=["111__clip000"])])
        census = tmp / "after.json"
        census.write_text(json.dumps(_census({
            "111__clip000": {"produced_now": True, "rates_agree": True,
                             "cfr_normalized_from": "/x/111.mp4"}})))
        redo = tmp / "redo.txt"
        redo.write_text("111\n")
        assert main(["--census", str(census), "--ingest-root", str(root),
                     "--redo", str(redo), "--since", "2000"]) == 0
        printed = capsys.readouterr().out
        assert "cut from a constant-rate re-encode: 1" in printed
        # The limit is stated on every pass, because the container property is
        # not the defect: the defect is picture and sound covering different
        # spans, and only a measurement on the bytes sees that.
        assert "NOT checked" in printed


# --------------------------------------------------------------------------
# The scan cache is per frame index, and the variable-rate fix re-encodes an
# upload under the same file name.
# --------------------------------------------------------------------------

def test_a_scan_cache_built_from_other_bytes_is_discarded_not_reused():
    import numpy as np
    from tools.ingest_wild_uploads import scan_upload, source_key

    class OneBoxPerFrame:
        def detect(self, frame):
            return np.zeros((1, 4), dtype=np.float32)

    with tempfile.TemporaryDirectory() as raw:
        tmp = pathlib.Path(raw)
        cache = tmp / "cache"
        video = tmp / "111.mp4"
        _make_clip(video, 25, seconds=1)
        first, _ = scan_upload(OneBoxPerFrame(), video, cache)
        assert (cache / "111.npz").is_file()
        assert np.load(cache / "111.npz")["source_key"].item() == source_key(video)

        # The same name, different bytes -- what normalize_to_cfr produces.
        video.unlink()
        _make_clip(video, 30, seconds=2)
        second, _ = scan_upload(OneBoxPerFrame(), video, cache)
        assert len(second) != len(first)
        assert np.load(cache / "111.npz")["source_key"].item() == source_key(video)


def test_a_cache_written_before_the_key_existed_is_still_used():
    # ~22 GPU-hours of scans carry no source_key.  Refusing them would rescan
    # the corpus to guard against something that has not happened.
    import numpy as np
    from tools.ingest_wild_uploads import scan_upload

    class Never:
        def detect(self, frame):
            raise AssertionError("the cache should have answered")

    with tempfile.TemporaryDirectory() as raw:
        tmp = pathlib.Path(raw)
        cache = tmp / "cache"
        cache.mkdir()
        np.savez_compressed(cache / "111.npz",
                            counts=np.asarray([1, 1], dtype=np.int32),
                            boxes=np.zeros((2, 4), dtype=np.float32),
                            cut_scores=np.zeros(2, dtype=np.float32))
        detections, scores = scan_upload(Never(), tmp / "111.mp4", cache)
        assert len(detections) == 2 and len(scores) == 2
