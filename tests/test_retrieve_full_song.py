"""``tools/retrieve_full_song.py``: the alignment that decides which download IS the
clip's recording, and where the clip sits in it.

WHY these three, measured 2026-09-22 on the 20 T-line eval clips:
* every verified clip had a speed ratio within 2% of 1.0, so real data cannot tell
  a correct rate model from one with the direction flipped -- and Douyin's most
  common variant is a speed-up.  The synthetic clip here is 12% fast, and the
  re-timed file ffmpeg writes must hold it at speed 1.0, where the timeline says;
* the 0.60 gate was calibrated on wrong songs (max 0.429) and covers (0.45-0.52);
  an unrelated song must stay under it here too, or the gate is not a gate.

No network: the songs are random chord sequences, which is what chroma sees.
"""
import pathlib
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools import retrieve_full_song as rfs  # noqa: E402

SR = rfs.SR


def chord_song(seed, seconds=60.0, chord_s=0.5):
    """Random triads with a few harmonics: harmonic content that never repeats."""
    rng = np.random.default_rng(seed)
    n = int(chord_s * SR)
    t = np.arange(n) / SR
    env = np.minimum(1.0, np.minimum(t, t[::-1]) / 0.02)  # 20 ms fades, no clicks
    out = []
    for _ in range(int(seconds / chord_s)):
        chord = np.zeros(n)
        for pc in rng.choice(12, 3, replace=False):
            f0 = 220.0 * 2 ** (pc / 12)
            for h in (1, 2, 3):
                chord += np.sin(2 * np.pi * f0 * h * t) / h
        out.append(chord * env)
    y = np.concatenate(out)
    return (0.3 * y / np.abs(y).max()).astype(np.float32)


def speed_up(y, rate):
    """Play ``y`` ``rate`` x faster the way a Douyin speed-up does: resample."""
    import librosa

    return librosa.resample(y, orig_sr=SR, target_sr=SR / rate, res_type="soxr_hq")


OFFSET, RATE, CLIP_S = 31.3, 1.12, 8.0


@pytest.fixture(scope="module")
def song():
    return chord_song(0)


@pytest.fixture(scope="module")
def clip(song):
    a = int(OFFSET * SR)
    return speed_up(song[a:a + int(CLIP_S * RATE * SR)], RATE)


@pytest.fixture(autouse=True)
def narrow_rate_grid(monkeypatch):
    # the shipped grid is 0.80-1.25; a narrower one around the answer keeps this
    # test fast without letting the answer sit on the grid's edge
    monkeypatch.setattr(rfs, "RATES", np.round(np.arange(1.02, 1.2201, 0.01), 3))


def test_a_sped_up_clip_is_found_with_its_speed_and_place(song, clip):
    res = rfs.align(rfs._Clip(clip), song)
    assert abs(res["rate"] - RATE) <= 0.005
    assert abs(res["offset_s"] - OFFSET) <= 2 / rfs.FPS
    assert res["score"] > 0.9


def test_an_unrelated_song_stays_under_the_gate(clip):
    res = rfs.align(rfs._Clip(clip), chord_song(1))
    assert res["score"] < 0.60


def test_the_retimed_file_holds_the_clip_at_speed_one(song, clip, tmp_path):
    original = tmp_path / "song.wav"
    sf.write(str(original), song, SR)
    timeline = {"clip_start_s": OFFSET / RATE}
    timeline["clip_end_s"] = timeline["clip_start_s"] + CLIP_S
    files = rfs.write_clip_speed_song(original, RATE, tmp_path, timeline)
    check = rfs.check_retimed(rfs._Clip(clip), pathlib.Path(files["full_song_clip_speed"]),
                              timeline["clip_start_s"])
    assert abs(check["error_s"]) <= 2 / rfs.FPS
    assert check["score_at_rate_1"] > 0.9
    # the continuation is exactly the rest of the re-timed song
    assert files["continuation_s"] == pytest.approx(len(song) / SR / RATE - timeline["clip_end_s"], abs=0.05)


def test_a_clip_that_switches_songs_is_verified_by_its_agreeing_windows(song):
    """One Douyin upload switched songs mid-video.  The whole-clip template then
    scores under the gate, but the windows that ARE the song agree on one place."""
    a = int(OFFSET * SR)
    head = speed_up(song[a:a + int(6.5 * RATE * SR)], RATE)
    tail = chord_song(2, seconds=6.0)[:int(5.5 * SR)]
    res = rfs.align(rfs._Clip(np.concatenate([head, tail])), song)
    assert res["score"] < rfs.MIN_SCORE
    agree = rfs.window_agreement(res["windows"], rfs.MIN_WINDOW_SCORE)
    assert agree["n_agree"] >= 2
    assert abs(agree["clip_start_s"] - OFFSET) <= 0.10
    assert abs(agree["rate"] - RATE) <= 0.011
    assert max(agree["t0s"]) < 6.5          # only windows inside the song's part


def test_windows_do_not_agree_on_an_unrelated_song(clip):
    res = rfs.align(rfs._Clip(clip), chord_song(1))
    assert rfs.window_agreement(res["windows"], rfs.MIN_WINDOW_SCORE)["n_agree"] < 2


def test_a_window_that_also_matches_a_repeated_hook_still_votes_for_the_right_place():
    """Measured on the 818 clip: its song plays the clip's hook in its intro too, so
    one window's best peak is the intro (0.997) and its second is the real place.
    Counting only best peaks, the two windows over that song disagreed and a clip
    ending on it came out unverified."""
    def window(t0, *peaks):
        ps = [{"score": s, "rate": 1.0, "implied_clip_start_s": st} for s, st in peaks]
        return dict(ps[0], t0=t0, peaks=ps)

    windows = [window(0.0, (0.3, 50.0)), window(2.5, (0.3, 90.0)),
               window(5.0, (0.974, 178.01)), window(7.5, (0.997, -10.15), (0.97, 178.0))]
    agree = rfs.window_agreement(windows, rfs.MIN_WINDOW_SCORE)
    assert agree["n_agree"] == 2 and agree["t0s"] == [5.0, 7.5]
    assert abs(agree["clip_start_s"] - 178.0) <= 0.01
    assert agree["covers_clip_end"]


# --------------------------------------------------------------------------- batch plumbing
# WHY (2026-09-23): the tool was run over all 333 T2 clips for output/sample_menu.  A 12.9 s
# clip got ONE Shazam window, [0, 10], so a song switch in its last 2.9 s was never heard;
# and the batch shares one download folder across parallel runs, where a per-clip glob or an
# unlocked "stale partial" cleanup would copy or delete another clip's song.

@pytest.mark.parametrize("dur", [10.0, 11.3, 12.9, 14.0, 20.6, 22.0, 24.0])
def test_the_shazam_windows_reach_the_clip_end(dur):
    starts = rfs.shazam_starts(dur)
    assert starts[0] == 0.0
    assert all(0.0 <= s <= max(dur - rfs.WIN, 0.0) + 1e-9 for s in starts)
    assert min(dur, starts[-1] + rfs.WIN) >= dur - 0.5  # the last half second at most is unheard
    assert starts == sorted(set(starts))


def test_a_shared_song_dir_downloads_once_and_hides_its_locks(tmp_path, monkeypatch):
    calls = []

    def fake_yt_dlp(cmd, **kw):  # writes what yt-dlp would: <id>.<ext> in the -o folder
        calls.append(cmd)
        out = pathlib.Path(cmd[cmd.index("-o") + 1].replace("%(ext)s", "webm"))
        sf.write(str(out), np.zeros(SR // 10, np.float32), SR, format="WAV")  # ffmpeg probes content
        return type("P", (), {"returncode": 0, "stderr": ""})()

    real_run = rfs.subprocess.run  # ffmpeg still runs for real
    monkeypatch.setattr(rfs.subprocess, "run", lambda cmd, **kw: fake_yt_dlp(cmd, **kw)
                        if "yt_dlp" in cmd else real_run(cmd, **kw))
    wav, attempts = rfs.download("vidA", tmp_path)
    assert attempts == 1 and wav.is_file()
    again, attempts = rfs.download("vidA", tmp_path)
    assert again == wav and attempts == 0 and len(calls) == 1
    # main() picks the original as "vidA.* that is not the wav": nothing else may match
    assert [p.name for p in tmp_path.glob("vidA.*") if p.suffix != ".wav"] == ["vidA.webm"]


def test_export_copies_only_this_clips_songs(tmp_path, monkeypatch):
    shared = tmp_path / "songs"
    shared.mkdir()
    for vid in ("mine", "other"):
        for ext in (".webm", ".wav"):
            (shared / (vid + ext)).write_bytes(b"x")
    out_dir, export_dir = tmp_path / "clip", tmp_path / "export"
    out_dir.mkdir()
    fake = type(sys)("check_disk_headroom")
    fake.probe = lambda path, gb: (True, None, "ok")
    monkeypatch.setitem(sys.modules, "check_disk_headroom", fake)
    res = rfs.export_outputs(out_dir, export_dir, sorted(shared.glob("mine.*")))
    assert res["exported"]
    assert sorted(p.name for p in export_dir.iterdir()) == ["mine.wav", "mine.webm"]


def detuned_chords(seed, seconds, factor, chord_s=0.5):
    """chord_song with every pitch scaled by ``factor`` (a tuning offset, not a new key)."""
    rng = np.random.default_rng(seed)
    n = int(chord_s * SR)
    t = np.arange(n) / SR
    env = np.minimum(1.0, np.minimum(t, t[::-1]) / 0.02)
    out = []
    for _ in range(int(seconds / chord_s)):
        chord = np.zeros(n)
        for pc in rng.choice(12, 3, replace=False):
            for h in (1, 2, 3):
                chord += np.sin(2 * np.pi * 220.0 * 2 ** (pc / 12) * factor * h * t) / h
        out.append(chord * env)
    y = np.concatenate(out)
    return (0.3 * y / np.abs(y).max()).astype(np.float32)


def test_the_retime_recheck_compares_both_sides_at_one_tuning(tmp_path):
    # WHY (2026-09-23): 7456795199654694202__clip000 aligned at 0.965 and its correct
    # re-timed file re-checked at 0.590 -> "retime_failed", because chroma_stft
    # estimated the tuning of the whole song (+0.25 bin) apart from the clip's (-0.05).
    # Here the song around the clip sits 0.45 bin sharp: its tuning estimate moves,
    # the clip's does not, and per-signal chroma loses the match it should confirm.
    sharp = 2 ** (0.45 / 12)
    clip_y = detuned_chords(2, 20.0, 1.0)
    song = np.concatenate([detuned_chords(1, 40.0, sharp), clip_y, detuned_chords(3, 120.0, sharp)])
    sf.write(str(tmp_path / "song.wav"), song, SR)

    sz, cz = rfs._chroma_z(song), rfs._chroma_z(clip_y)  # the old, per-signal tuning
    nfft = 1 << int(np.ceil(np.log2(sz.shape[1] + cz.shape[1] + 1)))
    old = rfs._score_curve(cz, np.fft.rfft(sz, n=nfft, axis=1), sz.shape[1], nfft).max()
    assert old < 0.85  # the defect is present in this input, so the assertion below has teeth

    res = rfs.check_retimed(rfs._Clip(clip_y), tmp_path / "song.wav", 40.0)
    assert res["score_at_rate_1"] > 0.99 and abs(res["error_s"]) < 1.0 / 30
