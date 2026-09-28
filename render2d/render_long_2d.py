#!/usr/bin/env python3
"""Render a motion LONGER than one SteadyDancer job -- a full song -- as a 2D skin.

``run_2d_steadydancer.sh`` renders one eval clip per ComfyUI prompt, and that is
the right unit for 12-24 s.  A full song is not: 244 s at the pipeline's 16 fps is
3910 frames, and one prompt holding them would load the pose video as float32
(4.79 MB/frame, 18.7 GB), clone it, and decode 18.7 GB more -- inside a 48 GB
cgroup that the SHARED ComfyUI server lives in, which an OOM would take down for
every session.  The longest SteadyDancer job this repo has run is 378 frames.

So this driver does NOT re-implement anything; it runs the same two components
with the same settings and only adds the cutting and the joining:

  A  pose     render2d/aapose_video.py ONCE on the whole motion.  Heading
              (median facing), the fit to the character (medians) and the bone
              fit (90th percentile) are computed over whatever input it gets, so a
              pose video drawn per chunk would give each chunk its own framing,
              scale and rotation.  Drawn once, then cut frame-exactly.
  B  chunks   N chunks of --chunk frames (4k+1: the sampler floors every job to
              4k+1, so any other size silently loses up to 3 frames per chunk and
              12 chunks drift ~2 s against the audio), starts spread evenly so
              neighbours overlap by at least --min-overlap.
  C  sample   render2d/comfy_steadydancer.py per chunk, one at a time, each
              put at the BACK of the server's FIFO the moment the previous one
              ends -- what every other batch does -- so batches take turns and no
              prompt already in line is ever jumped.  (Two earlier versions waited
              first, for an empty queue and then for no prompt pending; with two
              other batches each resubmitting the moment its prompt ended, both
              conditions starved this driver -- 2026-09-22.)  Only one driver may
              run per work dir (driver.pid).  --wait-pid additionally waits for a named
              batch to exit first.  Same character still for every chunk:
              chaining the previous chunk's last frame as the next reference
              changed the garment and face by chunk 3 on MTV (mtv_chain.py).
              Resumable: a chunk whose sidecar names this pose chunk's SHA-1 is
              not sampled again.
  D  stitch   The first frame of every render IS the still (measured on bones3
              818: |f0 - still| 4.96/255, 22-25 from f8 on), so each later chunk's
              head is dropped inside the overlap, with a --xfade-frame crossfade
              ending exactly where the earlier chunk ends.  Exactly the pose
              video's frame count comes out, and the song is muxed ONCE (muxing it
              per chunk would put the song's first 20 s under every chunk).
  E  panels   The four panels of run_2d_steadydancer.sh -- raw video | 3D pose |
              2D pose | 2D skin -- over the whole song.  Source footage exists only
              for the stretch the clip was cut from, so the raw panel is black
              elsewhere and says so; the other three span the song.

What this is NOT: render_sample_strip (CLAUDE.md 1.5.6).  That tool cannot draw a
song: its ground-truth row is mandatory and its avatar length is the min over
rows, so a 244 s arm comes out 21.8 s long, under the wrong audio, with exit 0.
For the stretch that has ground truth, render the strip on a slice of this motion.

Usage::

    python3 render2d/render_long_2d.py --motion ARM/<name>.pkl \\
        --audio full_song_clip_speed.wav --out output/sample_YYYYMMDD_fullsong \\
        --footage source.mp4 --footage-start-s 188.1745 --wait-pid 2930019
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import shlex
import subprocess
import sys
import time
import urllib.request

import numpy as np

# Root of the sibling checkouts (ComfyUI_Wan, Lodge, this repo); set E2E_ROOT.
E2E_ROOT = os.environ.get("E2E_ROOT", "/workspace/e2e")

REPO = pathlib.Path(__file__).resolve().parents[1]
COMFY_INPUT = pathlib.Path(E2E_ROOT + "/ComfyUI_Wan/input")
SERVER = "http://127.0.0.1:8188"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
# the settings of the bones3 / fix8 2D arms (queue_fix8_2d.sh, 2026-09-22), so a
# full-song render is comparable with the clips rendered the same day
AAPOSE_EXTRA = "--face-model calibrated --torso detector_v2 --hands-model v2 --fit bones"
SD_EXTRA = "--seed 42 --cfg-step0 2.0"
SD_NEGATIVE_EXTRA = "背影，背面，后脑勺，back view, from behind, back of head"
STILL_FRAMES = 8   # frames at a render's head still pulled toward the character still


def run(cmd, **kw):
    print("  $", " ".join(shlex.quote(str(c)) for c in cmd)[:400], flush=True)
    subprocess.run([str(c) for c in cmd], check=True, stdin=subprocess.DEVNULL, **kw)


def count_frames(path) -> int:
    out = subprocess.check_output(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                                   "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)])
    return int(out.strip())


def probe_size(path) -> tuple[int, int]:
    out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "stream=width,height", "-of", "csv=p=0", str(path)]).decode()
    w, h = out.strip().split(",")
    return int(w), int(h)


def duration(path) -> float:
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                          "-of", "csv=p=0", str(path)]).strip())


def sha1(path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- A
def stage_pose(args, work):
    out = work / "aapose.mp4"
    want = "--motion {} --match-character {} --stick-width 5 --fps 16 {}".format(
        args.motion, COMFY_INPUT / args.character, args.aapose_extra)
    stamp = work / "aapose.mp4.args"
    if out.is_file() and stamp.is_file() and stamp.read_text() == want:
        print("A pose: reuse", out)
    else:
        part = work / "aapose.part.mp4"
        run([sys.executable, REPO / "render2d/aapose_video.py", "--motion", args.motion, "--out", part,
             "--match-character", COMFY_INPUT / args.character, "--stick-width", "5", "--fps", "16",
             *shlex.split(args.aapose_extra)], cwd=REPO)
        part.replace(out)
        for side in (".fit.json", ".yaw.npy"):  # aapose names its sidecars after --out
            if pathlib.Path(str(part) + side).is_file():
                pathlib.Path(str(part) + side).replace(pathlib.Path(str(out) + side))
        stamp.write_text(want)
    n = count_frames(out)
    print("A pose: {} frames at 16 fps = {:.3f} s".format(n, n / 16))
    return out, n


# --------------------------------------------------------------------------- B
def chunk_plan(n: int, chunk: int, min_overlap: int) -> list[tuple[int, int]]:
    if (chunk - 1) % 4:
        raise SystemExit("--chunk must be 4k+1 (the sampler floors to 4k+1), got {}".format(chunk))
    if n <= chunk:
        return [(0, n)]
    k = math.ceil((n - chunk) / (chunk - min_overlap)) + 1
    starts = [round(i * (n - chunk) / (k - 1)) for i in range(k)]
    return [(s, s + chunk) for s in starts]


def stage_chunks(args, work, pose, n):
    plan = chunk_plan(n, args.chunk, args.min_overlap)
    overlaps = [plan[i][1] - plan[i + 1][0] for i in range(len(plan) - 1)]
    print("B chunks: {} x {} frames, overlaps {}..{}".format(len(plan), args.chunk, min(overlaps or [0]),
                                                            max(overlaps or [0])))
    if overlaps and min(overlaps) < STILL_FRAMES + args.xfade + 1:
        raise SystemExit("overlap {} cannot hold the still ({}) plus the crossfade ({})".format(
            min(overlaps), STILL_FRAMES, args.xfade))
    cdir = work / "chunks"
    cdir.mkdir(exist_ok=True)
    # Which pose video these chunks were cut from.  Frame count alone is not identity:
    # two arms of the same query write the SAME pkl file name, so a second arm reused the
    # first one's chunks -- right length, wrong dance -- and every skin chunk then matched
    # its sidecar and was "already rendered".  The stitched file was the FIRST arm's dance
    # under the second one's name, with exit 0 (measured 2026-09-23, t2_7650_fix8full).
    pose_sha = sha1(pose)
    stamp = cdir / "plan.json"
    stale = (not stamp.is_file()) or json.loads(stamp.read_text()).get("pose_sha1") != pose_sha
    if stale and stamp.is_file():
        print("B chunks: the pose video changed -- re-cutting every chunk")
    stamp.write_text(json.dumps({"frames": n, "chunk": args.chunk, "plan": plan,
                                 "pose": str(pose), "pose_sha1": pose_sha}, indent=1))
    for i, (s, e) in enumerate(plan):
        out = cdir / "pose_{:02d}.mp4".format(i)
        if not stale and out.is_file() and count_frames(out) == e - s:
            continue
        # near-lossless: this is re-encoded once more than the clip path's pose video
        run(["ffmpeg", "-v", "error", "-y", "-i", pose, "-an", "-vf",
             "trim=start_frame={}:end_frame={},setpts=PTS-STARTPTS".format(s, e), "-r", "16",
             "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", out])
        got = count_frames(out)
        if got != e - s:
            raise SystemExit("pose chunk {} has {} frames, want {}".format(i, got, e - s))
    return plan


# --------------------------------------------------------------------------- C
def queue_waiting() -> int:
    """Prompts waiting behind the running one: ours goes in only when this is 0."""
    try:
        with urllib.request.urlopen(SERVER + "/queue", timeout=30) as r:
            q = json.load(r)
        return len(q.get("queue_pending", []))
    except OSError as error:
        raise SystemExit("ComfyUI server not reachable at {}: {}".format(SERVER, error))


def stage_sample(args, work, plan):
    cdir = work / "chunks"
    if args.wait_pid:
        while pathlib.Path("/proc/{}".format(args.wait_pid)).exists():
            print("C sample: waiting for PID {} to exit ({})".format(args.wait_pid, time.strftime("%H:%M")),
                  flush=True)
            time.sleep(120)
    for i, (s, e) in enumerate(plan):
        pose, skin = cdir / "pose_{:02d}.mp4".format(i), cdir / "skin_{:02d}.mp4".format(i)
        side = pathlib.Path(str(skin) + ".json")
        if skin.is_file() and side.is_file() and json.loads(side.read_text()).get("pose_video_sha1") == sha1(pose):
            print("C sample: chunk {:02d} already rendered".format(i))
            continue
        # Straight to the BACK of the FIFO, exactly like the other batches do: one
        # of ours in line at a time, never ahead of anyone.  Waiting for "nothing
        # pending" instead starved this driver: with two batches that each resubmit
        # the moment their prompt ends, the queue had a waiting prompt at every
        # 30 s poll from 14:15 to 14:29 on 2026-09-22 and chunk 03 never went in.
        print("C sample: joining the queue behind {} waiting prompt(s) ({})".format(
            queue_waiting(), time.strftime("%H:%M")), flush=True)
        t0 = time.time()
        cmd = [sys.executable, REPO / "render2d/comfy_steadydancer.py", "--character", args.character,
               "--pose-video", pose, "--out", skin, *shlex.split(args.sd_extra)]
        if args.negative_extra:
            cmd += ["--negative-extra", args.negative_extra]
        run(cmd, cwd=REPO)
        got = count_frames(skin)
        if got != e - s:
            raise SystemExit("skin chunk {} has {} frames, want {}".format(i, got, e - s))
        print("C sample: chunk {:02d}/{} done in {:.1f} min".format(i, len(plan) - 1, (time.time() - t0) / 60),
              flush=True)


# --------------------------------------------------------------------------- D
class Reader:
    """Sequential frame access to one video; frames must be asked for in order."""

    def __init__(self, path, w, h):
        self.p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo",
                                   "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, stdin=subprocess.DEVNULL)
        self.size, self.shape, self.at, self.frame = w * h * 3, (h, w, 3), -1, None

    def get(self, j):
        while self.at < j:
            buf = self.p.stdout.read(self.size)
            if len(buf) != self.size:
                raise SystemExit("ran out of frames at {}".format(self.at + 1))
            self.frame, self.at = np.frombuffer(buf, np.uint8).reshape(self.shape), self.at + 1
        return self.frame

    def close(self):
        self.p.stdout.close()
        self.p.wait()


def source_plan(plan, n, xfade):
    """Global frame -> [(chunk, local frame, weight)].  Chunk i owns up to its end;
    the last ``xfade`` frames of its overlap blend into chunk i+1."""
    src = []
    for g in range(n):
        own = max(i for i, (s, _) in enumerate(plan) if s <= g)          # latest chunk that has started
        i = own - 1 if own > 0 and g < plan[own - 1][1] else own         # still inside the previous one?
        if i == own:
            src.append([(i, g - plan[i][0], 1.0)])
            continue
        end = plan[i][1]
        if g < end - xfade:
            src.append([(i, g - plan[i][0], 1.0)])
        else:
            a = (g - (end - xfade) + 1) / (xfade + 1)
            src.append([(i, g - plan[i][0], 1.0 - a), (i + 1, g - plan[i + 1][0], a)])
    return src


def stage_stitch(args, work, plan, n):
    cdir = work / "chunks"
    w, h = probe_size(cdir / "skin_00.mp4")
    silent = work / "skin_stitched.mp4"
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                            "{}x{}".format(w, h), "-r", "16", "-i", "-", "-c:v", "libx264", "-crf", "17",
                            "-pix_fmt", "yuv420p", str(silent)], stdin=subprocess.PIPE)
    readers = {}
    for g, parts in enumerate(source_plan(plan, n, args.xfade)):
        for i in [k for k in readers if k < parts[0][0]]:
            readers.pop(i).close()
        acc = np.zeros((h, w, 3), np.float32)
        for i, j, wt in parts:
            if i not in readers:
                readers[i] = Reader(cdir / "skin_{:02d}.mp4".format(i), w, h)
            acc += wt * readers[i].get(j)
        enc.stdin.write(np.clip(acc + 0.5, 0, 255).astype(np.uint8).tobytes())
    for r in readers.values():
        r.close()
    enc.stdin.close()
    if enc.wait():
        raise SystemExit("stitch encode failed")
    out = args.out / "fullsong_2d_skin.mp4"
    run(["ffmpeg", "-v", "error", "-y", "-i", silent, "-i", args.audio, "-map", "0:v", "-map", "1:a",
         "-c:v", "copy", "-c:a", "libmp3lame", "-b:a", "128k", "-ac", "1", "-ar", "44100", "-movflags", "+faststart", "-t", "{:.4f}".format(n / 16), out])
    got = count_frames(out)
    if got != n:
        raise SystemExit("stitched skin has {} frames, want {}".format(got, n))
    print("D stitch: {} frames = {:.3f} s -> {}".format(got, got / 16, out))
    return out


# --------------------------------------------------------------------------- E
# Long outputs are muxed like the short compares that DO play in the reviewer's IDE: mono 44.1 kHz MP3 (the IDE plays
# no AAC) with the MP4 index at the front.  The first full songs were stereo with the index at the end of a 60-150 MB
# file, and the operator got no sound (2026-09-24).


def stage_panels(args, work, pose, skin):
    stick = work / "pose3d.mp4"
    if not stick.is_file():
        part = work / "pose3d.part.mp4"
        # the batch's own 3D panel: one azimuth, dpi 160, following the root
        run([sys.executable, "-c",
             "import sys; sys.argv = sys.argv[1:]\n"
             "from tools import render_dance_video as R\n"
             "R.set_views((90,)); R.set_dpi(160); R.main()",
             "render_dance_video", "--result", args.motion, "--title", "{} · 3D pose".format(args.title),
             "--output", part, "--stride", str(args.stick_stride), "--follow-root"], cwd=REPO)
        part.replace(stick)
    seconds = duration(skin)
    H = args.panel_h
    label = ("drawtext=fontfile={}:text='{}':x=10:y=8:fontsize=26:fontcolor=white:"
             "box=1:boxcolor=black@0.6:boxborderw=6")
    inputs = []
    if args.footage:
        fdur = duration(args.footage)
        start = args.footage_start_s
        # ONE tpad for both ends: chaining a second tpad for stop_duration silently
        # does nothing (measured 2026-09-22: 6299 frames instead of 7331), and the
        # short raw panel then cut the whole hstack to 210 s via shortest=1
        raw = ("[0:v]fps=30,scale=-2:{H},setsar=1,tpad=start_duration={a:.4f}:start_mode=add:"
               "stop_duration={b:.4f}:stop_mode=add:color=black,trim=duration={d:.4f},setpts=PTS-STARTPTS,"
               + label.format(FONT, "raw video (only {:.0f}-{:.0f} s exists)".format(start, start + fdur))
               + "[a]").format(H=H, a=start, b=max(0.0, seconds - start - fdur) + 1, d=seconds)
        inputs += ["-i", str(args.footage)]
    else:
        raw = ("color=c=black:s={W}x{H}:r=30:d={d:.4f}," + label.format(FONT, "raw video: none")
               + "[a]").format(W=H * 9 // 16, H=H, d=seconds)
    off = 1 if args.footage else 0
    inputs += ["-i", str(stick), "-i", str(pose), "-i", str(skin)]
    graph = ";".join([
        raw,
        "[{}:v]fps=30,scale=-2:{},setsar=1[b]".format(off, H),
        "[{}:v]fps=30,scale=-2:{},setsar=1,".format(off + 1, H) + label.format(FONT, "2D pose") + "[c]",
        "[{}:v]fps=30,scale=-2:{},setsar=1,".format(off + 2, H) + label.format(FONT, "2D skin") + "[d]",
        "[a][b][c][d]hstack=inputs=4:shortest=1[v]"])
    out = args.out / "fullsong_compare.mp4"
    part = args.out / "fullsong_compare.part.mp4"
    run(["ffmpeg", "-v", "error", "-y", *inputs, "-i", args.audio, "-filter_complex", graph,
         "-map", "[v]", "-map", "{}:a".format(off + 3), "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
         "-pix_fmt", "yuv420p", "-c:a", "libmp3lame", "-b:a", "128k", "-ac", "1", "-ar", "44100", "-movflags", "+faststart", "-t", "{:.4f}".format(seconds),
         "-shortest", part])
    part.replace(out)
    got = duration(out)
    print("E panels: {} ({:.1f} s)".format(out, got))
    if abs(got - seconds) > 0.5:
        raise SystemExit("the four panels are {:.1f} s but the skin is {:.1f} s: a panel ran short and "
                         "hstack's shortest=1 cut the strip".format(got, seconds))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--motion", type=pathlib.Path, required=True, help="generated pkl with full_pose [T,24,3], 30 fps")
    ap.add_argument("--audio", type=pathlib.Path, required=True, help="the song, same timeline as the motion")
    ap.add_argument("--out", type=pathlib.Path, required=True, help="deliverables land here")
    ap.add_argument("--work", type=pathlib.Path, help="intermediates (default /cache/.../scratch/long2d/<motion stem>)")
    ap.add_argument("--title", default="generated")
    ap.add_argument("--character", default="townfair_fit.png", help="a file under ComfyUI input/")
    ap.add_argument("--aapose-extra", default=AAPOSE_EXTRA)
    ap.add_argument("--sd-extra", default=SD_EXTRA)
    ap.add_argument("--negative-extra", default=SD_NEGATIVE_EXTRA)
    ap.add_argument("--chunk", type=int, default=353, help="frames per SteadyDancer job, 4k+1 (353 = 5 windows)")
    ap.add_argument("--min-overlap", type=int, default=24)
    ap.add_argument("--xfade", type=int, default=8, help="crossfade frames inside each overlap")
    ap.add_argument("--footage", type=pathlib.Path, help="source video for the raw panel")
    ap.add_argument("--footage-start-s", type=float, default=0.0, help="song time of the footage's first frame")
    ap.add_argument("--stick-stride", type=int, default=2, help="3D panel stride (must divide 30)")
    ap.add_argument("--panel-h", type=int, default=640)
    ap.add_argument("--wait-pid", type=int, help="do not sample until this PID has exited")
    ap.add_argument("--stages", default="pose,chunks,sample,stitch,panels")
    args = ap.parse_args()

    stages = set(args.stages.split(","))
    # keyed by ARM as well as query: every arm writes the same pkl file name, and a work
    # dir keyed on the stem alone silently served one arm's chunks to another
    work = args.work or (pathlib.Path("/cache/atomicdance-assets/scratch/long2d") /
                         "{}__{}".format(args.motion.parent.name, args.motion.stem.replace(":", "__")))
    work.mkdir(parents=True, exist_ok=True)
    # one driver per work dir: two would each submit the same next chunk
    lock = work / "driver.pid"
    if lock.is_file():
        other = int(lock.read_text().split()[0])
        if other != os.getpid() and pathlib.Path("/proc/{}".format(other)).exists():
            raise SystemExit("another driver (PID {}) is already rendering {}".format(other, work))
    lock.write_text("{} {}\n".format(os.getpid(), time.strftime("%F %T")))
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(REPO / "tools"))
    from check_disk_headroom import probe
    ok, _, msg = probe(args.out, 1.0)
    if not ok:
        raise SystemExit("output folder cannot take 1 GB: {}".format(msg))

    pose, n = stage_pose(args, work) if "pose" in stages else (work / "aapose.mp4", count_frames(work / "aapose.mp4"))
    plan = stage_chunks(args, work, pose, n) if "chunks" in stages else json.loads(
        (work / "chunks/plan.json").read_text())["plan"]
    if "sample" in stages:
        stage_sample(args, work, plan)
    skin = stage_stitch(args, work, plan, n) if "stitch" in stages else args.out / "fullsong_2d_skin.mp4"
    if "panels" in stages:
        stage_panels(args, work, pose, skin)
    (args.out / "fullsong_2d.json").write_text(json.dumps({
        "motion": str(args.motion), "audio": str(args.audio), "work": str(work), "character": args.character,
        "aapose_extra": args.aapose_extra, "sd_extra": args.sd_extra, "negative_extra": args.negative_extra,
        "chunk": args.chunk, "plan": plan, "xfade": args.xfade, "frames_16fps": n,
        "footage": str(args.footage) if args.footage else None, "footage_start_s": args.footage_start_s,
    }, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
