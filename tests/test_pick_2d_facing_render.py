"""tools/pick_2d_facing_render -- which frames count as wrong, and which render wins.

The rule is the operator's two complaints plus the two ways a render can
disagree with the pose it was given; a frame counts once however many of them
it breaks, and a profile frame never counts (the scorer's profile calls are
its noisiest).
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.pick_2d_facing_render import wrong_frames, main  # noqa: E402


def row(head, body, side):
    return {"head": head, "body": body, "input_side": side}


def test_each_kind_of_wrong_is_counted_once():
    frames = [
        row("front", "front", 1),     # right
        row("back", "back", 1),       # phantom (a back view of a front input)
        row("front", "back", 1),      # phantom AND face on a back body -> one wrong frame
        row("back", "front", 1),      # back of the head on a front body
        row("front", "front", -1),    # missed a real turn
        row("back", "back", -1),      # right: a real back view
        row("profile", "profile", 1),  # profile never counts
        row("back", "profile", 1),    # back of the head on a profile body: counted
    ]
    counts = wrong_frames(frames)
    assert counts["phantom"] == 2
    assert counts["face_on_back"] == 1
    assert counts["hair_on_front"] == 2
    assert counts["missed"] == 1
    assert counts["wrong"] == 5


def scored(tmp, arm, clip, frames, flicker=0):
    d = tmp / arm
    (d / "facing").mkdir(parents=True, exist_ok=True)
    (d / (clip + ".mp4")).write_bytes(b"")
    summary = {"frames": len(frames), "body_flips": {"flips": 0, "flicker": flicker},
               "head_flips": {"flips": 0, "flicker": 0}}
    (d / "facing" / (clip + ".json")).write_text(json.dumps({"summary": summary, "frames": frames}))
    return str(d)


def test_the_cleanest_render_wins_and_ties_go_to_less_flicker(tmp_path):
    clean = [row("front", "front", 1)] * 10
    dirty = [row("back", "back", 1)] * 3 + [row("front", "front", 1)] * 7
    a = scored(tmp_path, "a", "c1__clip000", dirty)
    b = scored(tmp_path, "b", "c1__clip000", clean, flicker=2)
    c = scored(tmp_path, "c", "c1__clip000", clean, flicker=0)
    out = tmp_path / "pick.json"
    main([a, b, c, "--out", str(out)])
    pick = json.loads(out.read_text())["c1__clip000"]
    assert pathlib.Path(pick["arm"]).name == "c"


def test_a_clip_missing_from_an_arm_is_not_scored_clean(tmp_path):
    a = scored(tmp_path, "a", "c1__clip000", [row("back", "back", 1)] * 4)
    b = str(tmp_path / "b")
    (tmp_path / "b" / "facing").mkdir(parents=True)
    out = tmp_path / "pick.json"
    main([a, b, "--out", str(out)])
    pick = json.loads(out.read_text())["c1__clip000"]
    assert pathlib.Path(pick["arm"]).name == "a"


def test_a_render_of_the_wrong_length_is_refused_not_ranked(tmp_path):
    """The staging race once produced a 329-frame render for a 349-frame pose:
    a clean score for the wrong clip must not win."""
    import numpy as np
    short = scored(tmp_path, "short", "c1__clip000", [row("front", "front", 1)] * 10)
    (tmp_path / "short" / "work" / "c1__clip000").mkdir(parents=True)
    np.save(tmp_path / "short" / "work" / "c1__clip000" / "driven.npy", np.zeros((30, 20, 2)))
    right = scored(tmp_path, "right", "c1__clip000", [row("back", "back", 1)] * 2 + [row("front", "front", 1)] * 28)
    out = tmp_path / "pick.json"
    main([short, right, "--out", str(out)])
    assert pathlib.Path(json.loads(out.read_text())["c1__clip000"]["arm"]).name == "right"


def test_a_score_older_than_its_render_is_refused(tmp_path):
    import os
    stale = scored(tmp_path, "stale", "c1__clip000", [row("front", "front", 1)] * 10)
    video = tmp_path / "stale" / "c1__clip000.mp4"
    later = os.stat(tmp_path / "stale" / "facing" / "c1__clip000.json").st_mtime + 10
    os.utime(video, (later, later))
    fresh = scored(tmp_path, "fresh", "c1__clip000", [row("back", "back", 1)] * 3 + [row("front", "front", 1)] * 7)
    out = tmp_path / "pick.json"
    main([stale, fresh, "--out", str(out)])
    assert pathlib.Path(json.loads(out.read_text())["c1__clip000"]["arm"]).name == "fresh"


def test_the_profile_band_is_not_judged_and_unknown_bodies_are(tmp_path):
    frames = [row("back", "back", 1), row("front", "unknown", 1), row("front", "front", 1)]
    counts = wrong_frames(frames, sides=[0, 1, 1])     # frame 0 is inside the band
    assert counts["phantom"] == 0
    assert counts["unknown"] == 1 and counts["wrong"] == 1


def test_arms_scored_from_different_input_sources_are_refused(tmp_path):
    import pytest
    a = scored(tmp_path, "a", "c1__clip000", [row("front", "front", 1)] * 4)
    b = scored(tmp_path, "b", "c1__clip000", [row("front", "front", 1)] * 4)
    for arm, source in (("a", "cues"), ("b", "driven")):
        f = tmp_path / arm / "facing" / "c1__clip000.json"
        d = json.loads(f.read_text()); d["summary"]["input_side_source"] = source
        f.write_text(json.dumps(d))
    with pytest.raises(SystemExit):
        main([a, b])
