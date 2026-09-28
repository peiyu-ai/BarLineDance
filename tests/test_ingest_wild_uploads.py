"""Tests for the ingestion front-end: tracking, dancer choice, and where to cut.

These pin the decisions that replaced Lodge's two constants.  The tracking and
span logic are pure functions of detection boxes, so they are testable without
a GPU or a video, which is the point of having them as functions.
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.dwpose_video import link_tracks, score_track                # noqa: E402
from tools.ingest_wild_uploads import (FRAMES_PER_CLUSTER, MIN_CLUSTERS,  # noqa: E402
                                       MIN_FRAMES, find_spans, shot_cut_score)


def box(x, y, w=60, h=160):
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def test_link_tracks_follows_one_person_through_frames():
    detections = [np.stack([box(100 + i, 50)]) for i in range(40)]
    tracks = link_tracks(detections)
    assert len(tracks) == 1
    assert len(tracks[0]["frames"]) == 40


def test_link_tracks_keeps_two_people_apart():
    detections = [np.stack([box(100 + i, 50), box(600 - i, 50)]) for i in range(30)]
    tracks = link_tracks(detections)
    assert len(tracks) == 2
    assert all(len(t["frames"]) == 30 for t in tracks)


def test_link_tracks_survives_a_short_occlusion():
    detections = []
    for i in range(40):
        detections.append(np.zeros((0, 4), dtype=np.float32) if 10 <= i < 18
                          else np.stack([box(100 + i, 50)]))
    tracks = link_tracks(detections)
    # One track with a hole, not two tracks: a dancer turning away for a quarter
    # second is the case the gap tolerance exists for.
    assert len(tracks) == 1
    assert len(tracks[0]["frames"]) == 32


def test_link_tracks_breaks_after_a_long_absence():
    detections = []
    for i in range(60):
        detections.append(np.zeros((0, 4), dtype=np.float32) if 20 <= i < 45
                          else np.stack([box(100, 50)]))
    assert len(link_tracks(detections)) == 2


def test_score_track_prefers_the_moving_dancer_over_a_still_bystander():
    frames = 60
    dancer = {"frames": np.arange(frames),
              "boxes": np.stack([box(300 + 12 * np.sin(i / 4.0), 100) for i in range(frames)])}
    bystander = {"frames": np.arange(frames),
                 "boxes": np.stack([box(50, 100) for _ in range(frames)])}
    area = 1280.0 * 720.0
    assert (score_track(dancer, frames=frames, frame_area=area) >
            score_track(bystander, frames=frames, frame_area=area))


def test_score_track_prefers_persistence_over_a_brief_close_up():
    frames = 100
    steady = {"frames": np.arange(frames),
              "boxes": np.stack([box(300 + 10 * np.sin(i / 4.0), 100) for i in range(frames)])}
    passerby = {"frames": np.arange(8),
                "boxes": np.stack([box(300 + 10 * i, 100, w=300, h=600) for i in range(8)])}
    area = 1280.0 * 720.0
    assert (score_track(steady, frames=frames, frame_area=area) >
            score_track(passerby, frames=frames, frame_area=area))


def test_min_frames_comes_from_alg1_not_from_lodge():
    # Lodge's floor was 180 frames, annotated "past60 + future120" -- its own
    # motion-continuation window.  This repo's floor is the number of frames
    # Alg. 1 needs to have ten clusters to find structure among.
    assert MIN_FRAMES == FRAMES_PER_CLUSTER * MIN_CLUSTERS
    assert MIN_FRAMES != 180


def test_find_spans_drops_a_take_that_is_too_short_for_the_segmenter():
    present = np.zeros(300, dtype=bool)
    present[:200] = True
    spans = find_spans(present, np.zeros(300, dtype=bool),
                       min_frames=340, max_frames=900, max_absence=30)
    assert spans == []


def test_find_spans_ignores_a_blink_but_cuts_on_a_real_exit():
    present = np.ones(1400, dtype=bool)
    present[400:410] = False        # a blink: the dancer turns away
    present[900:1000] = False       # a real exit
    spans = find_spans(present, np.zeros(1400, dtype=bool),
                       min_frames=340, max_frames=1800, max_absence=30)
    assert len(spans) == 2
    assert spans[0][:2] == (0, 900)
    assert spans[0][2] == "dancer_absent"
    assert spans[1][0] == 1000


def test_find_spans_cuts_at_a_shot_change():
    present = np.ones(1400, dtype=bool)
    cuts = np.zeros(1400, dtype=bool)
    cuts[700] = True
    spans = find_spans(present, cuts, min_frames=340, max_frames=1800, max_absence=30)
    assert len(spans) == 2
    assert spans[0][1] <= 700 <= spans[1][0]
    assert spans[0][2] == "shot_cut"


def test_find_spans_divides_a_long_take_evenly_rather_than_leaving_a_tail():
    # A 50 s take under a 20 s cap is 2.5 caps.  Truncating would leave a 10 s
    # tail, and short clips are segmented more coarsely -- which is the defect
    # the content cut exists to remove, reintroduced by the length cap.
    present = np.ones(1500, dtype=bool)
    spans = find_spans(present, np.zeros(1500, dtype=bool),
                       min_frames=340, max_frames=600, max_absence=30)
    lengths = [b - a for a, b, _ in spans]
    assert len(spans) == 3
    assert max(lengths) - min(lengths) <= 1
    assert all(length <= 600 for length in lengths)


def test_shot_cut_score_separates_a_cut_from_dancer_motion():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, size=(90, 160), dtype=np.uint8)
    moved = frame.copy()
    moved[30:60, 60:100] = rng.integers(0, 255, size=(30, 40), dtype=np.uint8)
    cut = rng.integers(0, 255, size=(90, 160), dtype=np.uint8)
    assert shot_cut_score(frame, moved) < shot_cut_score(frame, cut)


def _counter_video(path, frames=60):
    """A video whose frame number is readable from its pixels.

    Each frame is a solid grey level ``4 * n``, so "which source frame is this"
    is answerable from the clip alone -- which is the only way to test the cut
    without trusting the thing under test to report its own alignment.
    """
    import subprocess

    import cv2

    raw = path.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 64))
    for n in range(frames):
        writer.write(np.full((64, 64, 3), 4 * n, dtype=np.uint8))
    writer.release()
    # Re-encode to h264 so the fixture matches what the corpus actually holds.
    subprocess.run(["ffmpeg", "-y", "-i", str(raw), "-c:v", "libx264",
                    "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)],
                   capture_output=True, check=True)
    return path


def _frame_level(path, index):
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        for position in range(index + 1):
            ok, frame = capture.read()
            if not ok:
                return None
        return float(frame.mean())
    finally:
        capture.release()


def test_cut_clip_lands_on_the_frame_it_was_asked_for(tmp_path):
    """An off-by-one here misaligns every crop in the clip, silently."""
    pytest.importorskip("cv2")
    from tools.ingest_wild_uploads import cut_clip

    source = _counter_video(tmp_path / "source.mp4", frames=60)
    out = tmp_path / "cut.mp4"
    written = cut_clip(source, 20, 40, out)
    assert written == 20
    # Compare against the source's *measured* levels, not the nominal 4n: h264
    # at yuv420p shifts a solid grey by a couple of levels, and a test that
    # asserted the ideal value would fail on the codec rather than on alignment.
    level = _frame_level(out, 0)
    neighbours = {index: _frame_level(source, index) for index in (19, 20, 21)}
    assert min(neighbours, key=lambda i: abs(level - neighbours[i])) == 20


@pytest.mark.parametrize("missing", [slice(0, 5), slice(20, 30), slice(55, 60)])
def test_gvhmr_bbx_is_written_for_every_frame_even_where_the_dancer_is_absent(missing, tmp_path):
    torch = pytest.importorskip("torch")
    # hmr4d lives in the GVHMR checkout, which is on the path only once
    # ``save_gvhmr_bbx`` puts it there; importorskip alone would skip a test
    # that can perfectly well run, and a skipped test pins nothing.
    root = pathlib.Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root / "third_party" / "pytorch3d_compat"),
                    str(root / "third_party" / "GVHMR")]
    pytest.importorskip("hmr4d", reason="GVHMR checkout not present")
    from tools.dwpose_video import VideoPose, save_gvhmr_bbx

    boxes = np.stack([box(100, 50) for _ in range(60)])
    boxes[missing] = np.nan
    pose = VideoPose(keypoints=np.zeros((60, 18, 2)), scores=np.zeros((60, 18)),
                     dancer_box=boxes, person_count=np.ones(60, dtype=np.int32),
                     frame_indices=np.arange(60), width=1280, height=720, fps=30.0,
                     track_score=1.0, rival_ratio=0.0, rival_count=0)
    out = tmp_path / "bbx.pt"
    save_gvhmr_bbx(pose, out)
    saved = torch.load(out, map_location="cpu")
    # GVHMR crops every frame; a NaN row would crop nothing and take the whole
    # clip down.  The absence itself stays recorded in detections.npz.
    assert saved["bbx_xyxy"].shape == (60, 4)
    assert bool(torch.isfinite(saved["bbx_xyxy"]).all())
    assert bool(torch.isfinite(saved["bbx_xys"]).all())


def test_find_spans_labels_its_own_division_as_such():
    """A boundary the length cap made is not a dancer walking out.

    The first version derived the reason from the span afterwards and could not
    tell the two apart, so it reported 66% of boundaries as ``dancer_absent``
    when most were its own even division -- a number that would have gone into
    the corpus accounting as a finding about the footage.
    """
    present = np.ones(1500, dtype=bool)
    spans = find_spans(present, np.zeros(1500, dtype=bool),
                       min_frames=340, max_frames=600, max_absence=30)
    reasons = [reason for _, _, reason in spans]
    assert reasons[:-1] == ["length_split"] * (len(spans) - 1)
    assert reasons[-1] == "upload_end"


def test_find_spans_marks_a_take_that_runs_to_the_end_of_the_upload():
    present = np.ones(500, dtype=bool)
    spans = find_spans(present, np.zeros(500, dtype=bool),
                       min_frames=340, max_frames=900, max_absence=30)
    assert spans == [(0, 500, "upload_end")]


def test_one_unreadable_upload_does_not_end_the_sweep(tmp_path, monkeypatch):
    """A decoder refusing one file must not take the shard with it.

    It did: a 201-second, 3 MB upload that ffprobe reads and cv2 will not open
    raised out of the loop and ended a shard of 949 uploads at 158.  Every other
    sweep in this repo logs the failure and carries on.
    """
    import json as jsonlib

    from tools import ingest_wild_uploads as tool

    videos = tmp_path / "videos"
    videos.mkdir()
    for name in ("aaa", "bbb", "ccc"):
        (videos / (name + ".mp4")).write_bytes(b"not really a video")
    out = tmp_path / "out"

    seen = []

    def fake_ingest(upload, out_root, **kwargs):
        seen.append(upload.stem)
        if upload.stem == "bbb":
            raise tool.UnreadableUpload("cv2 refused it")
        return {"upload": upload.stem, "status": "ok", "spans": [], "clips": []}

    monkeypatch.setattr(tool, "ingest", fake_ingest)
    monkeypatch.setattr(tool, "DWPoseExtractor", lambda **kwargs: object())

    tool.main(["--videos-dir", str(videos), "--out-root", str(out),
               "--max-seconds", "24"])

    assert len(seen) == 3, "the sweep stopped at the bad upload"
    rows = [jsonlib.loads(line)
            for line in (out / "ingest_shard0.jsonl").open(encoding="utf-8")]
    assert len(rows) == 3
    failed = [r for r in rows if r["status"] == "failed"]
    # The loss is recorded with its cause, not inferred later from a short log.
    assert len(failed) == 1
    assert "UnreadableUpload" in failed[0]["error"]
