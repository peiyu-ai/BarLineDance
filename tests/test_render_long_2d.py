"""``render2d/render_long_2d.py``: cutting a song-length pose video into
SteadyDancer-sized jobs and joining the renders back to the exact frame count.

WHY: a song is 3910 frames at 16 fps and one job holding them would OOM the
shared ComfyUI server; the longest job run here is 378 frames.  The join has two
constraints that no picture would reveal until hours of GPU were spent:
* every job is floored to 4k+1 frames by the sampler, so a chunk of any other
  size silently loses frames and the video drifts against the song;
* the first ~8 frames of every render are pulled toward the character still
  (measured on bones3 818), so a later chunk's head must never be shown.
"""
import importlib.util
import json
import pathlib

import pytest

spec = importlib.util.spec_from_file_location(
    "render_long_2d", pathlib.Path(__file__).resolve().parents[1] / "render2d/render_long_2d.py")
rl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rl)


def test_a_song_is_cut_into_4k_plus_1_chunks_that_cover_it_with_overlap():
    plan = rl.chunk_plan(3910, 353, 24)
    assert plan[0][0] == 0 and plan[-1][1] == 3910
    assert all(e - s == 353 for s, e in plan)
    overlaps = [plan[i][1] - plan[i + 1][0] for i in range(len(plan) - 1)]
    assert min(overlaps) >= 24


def test_a_chunk_size_the_sampler_would_floor_is_refused():
    with pytest.raises(SystemExit):
        rl.chunk_plan(3910, 354, 24)


def test_the_join_gives_every_frame_once_and_never_a_later_chunks_head():
    n, plan = 3910, rl.chunk_plan(3910, 353, 24)
    src = rl.source_plan(plan, n, 8)
    assert len(src) == n
    last = {}
    for g, parts in enumerate(src):
        assert sum(w for _, _, w in parts) == pytest.approx(1.0)
        for i, j, w in parts:
            assert plan[i][0] + j == g                      # the frame is the song's frame g
            if i > 0:
                assert j >= rl.STILL_FRAMES                 # past the still-contaminated head
            assert j >= last.get(i, -1)                     # readable sequentially
            last[i] = j
    assert sum(len(p) == 2 for p in src) == 8 * (len(plan) - 1)


def test_a_motion_that_fits_one_job_is_one_chunk():
    assert rl.chunk_plan(300, 353, 24) == [(0, 300)]


def test_chunks_are_recut_when_the_pose_video_changes(tmp_path):
    """Two arms of one query write the SAME pkl file name, so the second arm reused the
    first one's chunks: right frame counts, wrong dance, every skin "already rendered",
    and the stitched file was the first arm's dance under the second one's name (exit 0,
    measured 2026-09-23 on t2_7650_fix8full).  plan.json now carries the pose video's
    SHA-1 and a mismatch re-cuts every chunk."""
    import subprocess
    import types

    def pose_video(path, colour):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                        "color=c={}:s=64x64:r=16:d=4".format(colour), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", str(path)], check=True)

    work = tmp_path / "work"
    (work / "chunks").mkdir(parents=True)
    pose = work / "aapose.mp4"
    args = types.SimpleNamespace(chunk=33, min_overlap=12, xfade=2)

    pose_video(pose, "red")
    n = rl.count_frames(pose)
    rl.stage_chunks(args, work, pose, n)
    first = rl.sha1(work / "chunks/pose_00.mp4")
    assert json.loads((work / "chunks/plan.json").read_text())["pose_sha1"] == rl.sha1(pose)

    pose_video(pose, "blue")                      # the other arm's dance, same frame count
    assert rl.count_frames(pose) == n
    rl.stage_chunks(args, work, pose, n)
    assert rl.sha1(work / "chunks/pose_00.mp4") != first
    assert json.loads((work / "chunks/plan.json").read_text())["pose_sha1"] == rl.sha1(pose)
