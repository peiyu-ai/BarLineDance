#!/usr/bin/env python3
"""Four panels per clip, one file: source footage, ground truth and generated.

    row 1:  [ source video     | ground-truth pose | generated pose ]
    row 2:  [ ground-truth skin | baseline skin     | generated skin ]
    row 3:  the plan as a time strip, one lane per arm, with a moving playhead

Two rows of three rather than one strip of five: at five panels wide each one is
a sixth of the screen and the reviewer is comparing postage stamps.  The split
is also the honest one -- the top row is what the joints do (footage and two
stick figures, nothing that can hide geometry) and the bottom row is what a
person looks like doing it.

Why four and not one.  The two renderers this composes answer different
questions and each hides what the other shows, which is stated in their own
headers: the stick figure is "exactly the joint positions the model produced --
no mesh, nothing that can silently fake geometry", and the avatar answers the
one thing a stick figure cannot, "does this read as a person dancing".  Beside
them the source footage is the only panel that is not a reconstruction at all,
so it is the reference for whether the *ground truth* itself is right -- twice
now a defect blamed on the model turned out to be in the ingest (a 60 fps
upload cut at half speed, worklog 2026-08-25).

Panel provenance, so nothing here is guessed:
* footage -- ``<ingest>/<clip>/clip.mp4`` when the cut copy is cached, else cut
  from ``meta.json``'s own ``source`` and ``source_frame_span`` at the source's
  own fps.  Never re-derived from frame counts: that is exactly the arithmetic
  that produced the half-speed clips.
* stick figure -- ``tools/render_dance_video.py``.
* avatars -- ``tools/render_avatar_video.py``, both rows on ONE stage so the
  panels stay comparable, with the camera on the centroid of every row.

The third row answers a question the first two cannot: whether a thin-looking
generation is thin because the plan is *dense with splices* (every block short,
the dancer never finishing a movement) or *sparse with filler* (long grey
stretches where the planner named no atomic movement and the completion is
improvising).  Those two produce the same impression on a skeleton and need
opposite fixes.  Filler is named rather than inferred -- label 0 is the
transition class -- and the lanes share the video's time axis, so the playhead
ties a block to the movement on screen.

Audio comes from the ingest's own ``audio.wav`` and is muxed as mp3, because
the reviewer's IDE plays mp3 in an mp4 container and will not play AAC.
"""
import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

import numpy as np

# Overridable, because a second corpus now exists: the T series keeps its own
# converted tree, its own eval export and its own label space (20 classes, not
# 4,528).  Rendered against the v5 defaults, a T clip either fails to resolve or
# -- worse -- draws another corpus's ground truth beside this one's generation.
# Defaults are unchanged, so every existing invocation behaves as before.
INGEST = pathlib.Path(os.environ.get(
    "STRIP_INGEST", "/cache/atomicdance-assets/data/wild_ingest_v1"))
AUDIO = pathlib.Path(os.environ.get(
    "STRIP_AUDIO",
    str(pathlib.Path(__file__).resolve().parents[1] / "runs/wild_v5_song_gt_eval/audio")))
GT_MOTION = pathlib.Path(os.environ.get(
    "STRIP_GT_MOTION",
    str(pathlib.Path(__file__).resolve().parents[1] / "runs/wild_v5_song_gt_eval/motion")))
LABELS = pathlib.Path(os.environ.get(
    "STRIP_LABELS", "/dev/shm/atomicdance-m3a-segmentation/ingroup_llm_v5rekey"))
CONVERTED = pathlib.Path(os.environ.get(
    "STRIP_CONVERTED", "/cache/atomicdance-assets/data/wild3d/ingest_v1_converted"))
DATA = pathlib.Path("/cache/atomicdance-assets/data")
REPO = pathlib.Path(__file__).resolve().parents[1]
# Invoked as ``python3 tools/render_sample_strip.py``, sys.path[0] is
# ``tools/`` and the repo root is absent, so ``import tools.render_plan_strip``
# raises ModuleNotFoundError -- which the except below turns into one
# printed line while the video still renders, silently missing the plan
# lanes.  Those lanes are the only panel that says whether a thin
# generation is dense-with-splices or sparse-with-filler, i.e. the reason
# to build this strip at all.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def slug(title):
    """A filesystem-safe, readable stem for a panel title.

    Runs of separators collapse to a single dash: a title like
    "epoch12 fix \u00b7 pose" has a space, a middle dot and another space in a
    row, and a single ``replace("--", "-")`` leaves "fix--pose" behind.
    """
    kept = "".join(c if (c.isalnum() or c in "-_") else "-" for c in title)
    return re.sub(r"-+", "-", kept).strip("-")[:40] or "panel"


def frames_of(video):
    """Duration in seconds, from ffprobe rather than from a frame count divided
    by an assumed fps -- that division is what produced half-speed clips once."""
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "csv=p=0", str(video)]).stdout.strip()
    return float(out) if out else 0.0


def run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit("failed: {}\n{}".format(" ".join(map(str, command)), result.stderr[-2000:]))
    return result


def video_rate(video):
    """Frames per second of a rendered piece, from ffprobe.

    Read rather than assumed: the looped plan-strip image must be fed at the
    SAME rate as the panels, and hardcoding 30 would silently reintroduce the
    framesync blow-up the moment a piece is rendered at another rate.
    """
    out = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0",
               str(video)]).stdout.strip()
    num, _, den = out.partition("/")
    try:
        value = float(num) / float(den or 1)
    except (TypeError, ValueError, ZeroDivisionError):
        value = 0.0
    if not value > 0:
        raise SystemExit(
            "error: could not read a frame rate from {} (ffprobe said {!r}), so "
            "the plan-strip image cannot be fed at a matching rate."
            .format(video, out))
    return value


def mux_command(top, avatars, audio, target, *, panel_height, width,
                strip, seconds, rate=30.0):
    """The single ffmpeg call that stacks the panels into one file.

    Pulled out of ``main`` so the LENGTH BOUND can be tested without rendering
    anything -- see ``tests/test_render_sample_strip_bounds.py`` for what went
    wrong without it.

    ``-loop 1`` on the plan-strip image is an INFINITE video input, and vstack
    pads its shorter input rather than ending with it, so the graph never
    reaches EOF.  Output-level ``-shortest`` does not rescue that, because the
    endless stream is itself the video output stream.  Measured before the fix:
    four 14-second strips reached 373-444 MB and 1,105,642 frames -- ten hours
    of video for fourteen seconds of dance -- and never wrote a moov atom, so
    each was unplayable while looking like a render still in progress.  Two
    independent bounds are used because either alone has failed here:
    ``shortest=1`` on that vstack, and an explicit ``-t``.
    """
    if not seconds > 0:
        # Refused here rather than passed through as ``-t 0``: a zero bound
        # satisfies "the command has a duration" and writes an empty mp4 that
        # looks exactly like a successful render.
        raise SystemExit(
            "error: {} reports a duration of {!r}, so there is no length to "
            "bound the strip render with.".format(avatars, seconds))
    inputs = []
    for piece in list(top) + [avatars]:
        inputs += ["-i", piece]
    # The avatar strip is already N panels wide; the top row is scaled to match
    # its width so vstack accepts them.  vstack refusing on a width mismatch is
    # the check we want -- a silent rescale would quietly change what "same
    # size" means between the two rows.
    chain = "".join("[{}:v]scale=-2:{}[t{}];".format(i, panel_height, i)
                    for i in range(len(top)))
    chain += "".join("[t{}]".format(i) for i in range(len(top)))
    chain += "hstack=inputs={}[top];".format(len(top))
    chain += "[{}:v]null[bot];".format(len(top))
    chain += "[top]scale={}:-2[topw];[topw][bot]vstack=inputs=2[stack]".format(width)
    audio_index = len(top) + 1
    if strip is not None:
        # -framerate MATCHING THE PANELS, and this is the whole difference
        # between a render that takes under a minute and one that takes hours.
        # A looped image input defaults to 25 fps.  The panels are 30.  When
        # vstack's framesync reconciles 25 with 30 it runs the graph at their
        # least common multiple -- 150 fps -- so every second of output costs
        # five times the frames, the encoder is handed five times the work, and
        # the file is five times the size.  Measured on one clip: two seconds of
        # output took over 120s at preset ultrafast and never finished at the
        # default; with the rate matched the same two seconds take 7.9s and
        # 468 KB, at exactly 60 frames.  This was behind every slow and
        # oversized strip render on 2026-09-03.
        inputs += ["-loop", "1", "-framerate", "{:g}".format(rate), "-i", strip]
        audio_index += 1
        chain += (";[{}:v]scale={}:-2,setsar=1[strip];"
                  "[strip]drawbox=x='(t/{:.4f})*iw':y=0:w=3:h=ih:"
                  "color=red@0.85:t=fill[striph];"
                  "[stack][striph]vstack=inputs=2:shortest=1[v]").format(
                      len(top) + 1, width, max(seconds, 1e-3))
    else:
        chain += ";[stack]null[v]"
    return ["ffmpeg", "-v", "error", "-y", *inputs, "-i", audio,
            "-filter_complex", chain, "-map", "[v]",
            "-map", "{}:a".format(audio_index),
            # veryfast, because this pass only COMPOSITES pieces that are
            # already encoded: at the default preset one 14-second strip took
            # over an hour of wall clock while the GPU box was also training,
            # and a render nobody waits for is a render nobody sees.  The
            # quality that costs is not visible in a side-by-side panel.
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "libmp3lame", "-b:a", "128k",
            "-t", "{:.4f}".format(seconds), "-shortest", target]


def verify_layout(target, *, width, panels_top, has_strip, seconds,
                  tolerance=0.5):
    """Refuse a rendered strip whose geometry is not the one that was asked for.

    THE GUARANTEE THIS PROVIDES.  Every panel in this pipeline can go missing
    for its own quiet reason -- footage whose source moved, plan lanes whose
    labels did not load -- and the ffmpeg graph is happy to stack whatever it
    was handed.  The result is a file that plays, looks like a comparison, and
    is missing the panel the operator opened it for.  Twice now that is exactly
    what shipped.

    So the geometry is checked against the declared layout AFTER the mux, and a
    file that does not match is DELETED before the error is raised: a
    wrong-geometry file left on disk is indistinguishable from a good one at a
    glance, and the next reader would trust it.
    """
    probe = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height",
                 "-show_entries", "format=duration",
                 "-of", "default=nw=1:nk=1", str(target)])
    fields = [line for line in probe.stdout.split() if line]
    if len(fields) < 3:
        pathlib.Path(target).unlink(missing_ok=True)
        raise SystemExit(
            "error: {} carries no readable video stream after muxing; it has "
            "been removed rather than left to look like a render."
            .format(target))
    got_width, got_height, got_seconds = int(fields[0]), int(fields[1]), float(fields[2])
    problems = []
    if got_width != width:
        problems.append("width {} != expected {} (top row holds {} panels)"
                        .format(got_width, width, panels_top))
    if abs(got_seconds - seconds) > tolerance:
        problems.append("duration {:.2f}s != expected {:.2f}s"
                        .format(got_seconds, seconds))
    # The two stacked rows are square panels; anything above them is the plan
    # strip.  Its exact height depends on the figure, so this asserts only that
    # it is THERE -- which is the thing that kept silently not being there.
    rows_height = got_height
    if has_strip and rows_height <= width // max(panels_top, 1) * 2:
        problems.append("no plan-lane band: height {} leaves nothing above the "
                        "two panel rows".format(got_height))
    if problems:
        pathlib.Path(target).unlink(missing_ok=True)
        raise SystemExit(
            "error: {} did not match the layout it was rendered for, and has "
            "been removed:\n  - {}".format(target, "\n  - ".join(problems)))
    return {"width": got_width, "height": got_height, "seconds": got_seconds}


def footage_for(stem, out, size, pad=True):
    """The clip's own pixels, cut the way the ingest cut them.

    ``pad`` squares the panel with white so a row of square panels lines up,
    which is what this strip's two rows need.  A caller that stacks the footage
    beside PORTRAIT panels (the 2D line: a 480x832 pose picture and the cartoon
    made from it) passes ``pad=False`` and gets the cut at its own aspect --
    otherwise a 9:16 phone clip arrives as 292 px of dancer inside 520 px of
    white, i.e. the reviewer is handed a dancer 44% narrower than the panel
    beside her for no reason but the padding.
    """
    cached = INGEST / stem / "clip.mp4"
    meta = json.loads((INGEST / stem / "meta.json").read_text())
    if cached.is_file():
        source, start, frames = cached, 0, meta["num_frames"]
        fps = meta["fps"]
    else:
        source = DATA / meta["source"].replace("data/", "", 1)
        if not source.is_file():
            # Returning None here used to drop the footage panel silently, and
            # the strip still rendered -- so the one panel that is not a
            # reconstruction, the thing the operator checks the reconstruction
            # AGAINST, could vanish without a word.  The caller decides whether
            # that is acceptable; this reports it.
            return None
        span = meta["source_frame_span"]
        probe = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                     "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(source)])
        num, _, den = probe.stdout.strip().partition("/")
        fps = float(num) / float(den or 1)
        start, frames = span[0], span[1] - span[0]
    run(["ffmpeg", "-v", "error", "-y", "-ss", "{:.4f}".format(start / fps), "-i", str(source),
         "-frames:v", str(frames), "-an",
         # force_divisible_by=2 on the unpadded branch: a 9:16 phone clip
         # scales to 293x520 and libx264 REFUSES an odd width, so the cut fails
         # outright.  Loud, not silent -- but the fix belongs here rather than
         # in each caller.  The padded branch cannot hit it, its output is
         # size x size.
         "-vf", ("fps={},scale={}:{}:force_original_aspect_ratio=decrease"
                 + (",pad={}:{}:(ow-iw)/2:(oh-ih)/2:white" if pad
                    else ":force_divisible_by=2")).format(
                     meta["fps"], size, size, size, size),
         "-pix_fmt", "yuv420p", str(out)])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", required=True, help="file of wild_v5:<id>:clipNNN lines")
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=DIR",
                        help="a generated run directory to show; repeat for a comparison. "
                             "The stick-figure panel is drawn for the LAST arm, the one "
                             "being proposed; every arm gets an avatar panel beside the "
                             "ground truth's, all on one stage")
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-missing-clips", action="store_true",
                        help="render whatever resolves instead of refusing. Only for a "
                             "deliberate subset; the fixed ten are a ruler and a short "
                             "batch is a different ruler")
    parser.add_argument("--allow-missing-panels", action="store_true",
                        help="write the strip even when the source footage or "
                             "the plan lanes could not be produced. Off by "
                             "default: a strip missing those two panels still "
                             "plays and still looks like a comparison, which is "
                             "how it shipped twice without either")
    parser.add_argument("--size", type=int, default=520)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--vrm", default=str(REPO / "third_party/vrm/anime_female.vrm.glb"))
    parser.add_argument("--fill", type=float, default=1.0,
                        help="share of the frame height the avatar fills; the renderer's "
                             "own default is 0.72, which leaves headroom for rows that "
                             "drift apart -- unnecessary here because the rows share a "
                             "ground track")
    parser.add_argument("--free-root", action="store_true",
                        help="let each avatar row stand where its own motion puts it. Off by "
                             "default, i.e. the rows share the reference's ground track and "
                             "compare as POSES.\n\n"
                             "The default is a judgement call and it is worth stating. The "
                             "travel difference is real and measured -- the generated root "
                             "moves 0.33 m against the ground truth's 1.21 m over 100 clips -- "
                             "but on screen it does not read as 'the model does not travel', "
                             "it reads as two dancers drifting apart until the camera has to "
                             "back off and both become small. This strip exists to judge the "
                             "DANCE, and the travel defect is reported as a number "
                             "(root_range_vs_gt) where it cannot be mistaken for anything "
                             "else. Pass --free-root to see it on screen instead.")
    args = parser.parse_args()

    arms = [spec.partition("=")[::2] for spec in args.arm]
    name, run_dir = arms[-1]
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Unique per process.  Four workers sharing one staging directory is a
    # race that fails late and confusingly: the first to finish removes the
    # tree, and the others then die on "No such file" naming a file they
    # themselves had just written.
    staging = out / ".staging-{}".format(os.getpid())
    staging.mkdir(exist_ok=True)

    clips = [line.strip() for line in open(args.clips) if line.strip()]
    # The ground truth's own plan, for the strip's reference lane.  Its pickles
    # carry joints only, so the labels come from the M3 tree that produced them.
    # A GATE THAT CAN FAIL, because the failure it guards is silent.  These
    # three paths default to the v5 line; pointed at a T-line run they simply do
    # not resolve, and the strip still renders -- without the music envelope,
    # without the ground-truth lane, and with "0.00 on beat" printed for every
    # arm as though it had been measured.  That is exactly what happened on
    # 2026-09-13, and it is the second time this default has cost a round (see
    # the note in worklog about a strip that dropped five ground truths and
    # exited 0).  STRIP_AUDIO / STRIP_GT_MOTION / STRIP_LABELS override them.
    missing = [name for name, path in (("STRIP_AUDIO", AUDIO),
                                       ("STRIP_GT_MOTION", GT_MOTION),
                                       ("STRIP_LABELS", LABELS))
               if not path.is_dir()]
    if missing:
        raise SystemExit(
            "the strip would silently lose a panel: {} do not resolve ({}). "
            "The music envelope, the ground-truth lane and the on-beat readout "
            "all come from these, and without them the strip still renders and "
            "prints 0.00 on beat. Set them for the line you are on -- the T "
            "line is runs/txy_t_gt_eval/audio, runs/txy_t_gt_eval/motion and "
            "data/wild3d/txy_t_labels.".format(
                ", ".join(missing),
                "; ".join("{}={}".format(n, p) for n, p in
                          (("STRIP_AUDIO", AUDIO), ("STRIP_GT_MOTION", GT_MOTION),
                           ("STRIP_LABELS", LABELS)))))

    labels_by_clip = {}
    if LABELS.is_dir():
        for line in open(LABELS / "labels.jsonl"):
            record = json.loads(line)
            if record["sequence_id"] in set(clips):
                labels_by_clip[record["sequence_id"]] = np.load(LABELS / record["labels_path"])
    skipped = []
    for clip in clips:
        stem = clip.replace("wild_v5:", "").replace(":", "__")
        generated = REPO / run_dir / (clip + ".pkl")
        available = [(title, REPO / directory / (clip + ".pkl")) for title, directory in arms]
        audio = INGEST / stem / "audio.wav"
        gaps = [str(path) for _t, path in available if not path.is_file()]
        if not audio.is_file():
            gaps.append(str(audio))
        if not (CONVERTED / stem).is_dir():
            gaps.append(str(CONVERTED / stem) + " (STRIP_CONVERTED)")
        if gaps:
            # NAMED AND COUNTED, not skipped silently.  This branch used to
            # print "skip <clip> (missing asset)" and carry on to exit 0, so a
            # round could deliver six clips of the fixed ten and look finished.
            # CLAUDE.md 1.5 rule 6 fixes those ten precisely so that a round
            # cannot be compared against a different set -- 换片子等于换尺子 --
            # and a silent skip is exactly how the set changes.  On 2026-09-13
            # four of the ten were dropped this way (their ingest_v1_converted
            # directories are not on this machine); the fix was to point
            # STRIP_CONVERTED at a staged tree, which nothing in the old
            # message suggested.
            skipped.append((stem, gaps))
            print("MISSING {} -> {}".format(stem, ", ".join(gaps)))
            continue
        target = out / (stem + ".mp4")

        shot = footage_for(stem, staging / (stem + "_footage.mp4"), args.size)
        sticks = []
        panels = ([("ground truth · pose", CONVERTED / stem / "atomic_motion_151.npy")]
                  + [("{} · pose".format(t), p) for t, p in available[-1:]])
        for index, (title, path) in enumerate(panels):
            # The staging name is derived from the title, so it must be unique
            # per panel.  It used to be ``title.split(" ")[0][:6]``, which turned
            # "epoch12 fix" into "epoch1" -- unreadable, and worse, two arms
            # whose first words share six characters ("epoch12 fix" and
            # "epoch16 fix") would have written to the SAME file and one panel
            # would have silently shown the other arm's motion.  The index makes
            # collision impossible regardless of what the titles are.
            piece = staging / "{}_stick_{}_{}.mp4".format(stem, index, slug(title))
            run([sys.executable, "-c",
                 "import sys; sys.argv = sys.argv[1:];"
                 "from tools import render_dance_video as R; R.set_views((90,));"
                 "R.main()",
                 "render_dance_video", "--result", str(path), "--title", title,
                 "--output", str(piece), "--stride", str(args.stride)])
            sticks.append(piece)
        avatars = staging / (stem + "_avatar.mp4")
        command = [sys.executable, str(REPO / "tools/render_avatar_video.py"),
                   "--motion", "ground truth:{}".format(CONVERTED / stem)]
        for title, path in available:
            command += ["--motion", "{}:{}".format(title, path)]
        command += ["--output", str(avatars), "--vrm", args.vrm, "--view", "front",
                    "--size", str(args.size), "--stride", str(args.stride),
                    "--fill", str(args.fill)]
        if not args.free_root:
            command.append("--lock-root")
        run(command)

        # The plan strip, drawn once as a still and given a playhead by ffmpeg
        # rather than re-rendered per frame: 500 matplotlib figures per clip
        # costs more than the rest of this pipeline put together, and the only
        # thing that changes between them is one vertical line.
        strip = None
        try:
            from tools import render_plan_strip as PS
            import pickle as _pickle
            lanes = []
            gt_labels = labels_by_clip.get(clip)
            # Joints, not None: ``render_plan_strip`` derives its accents/s
            # readout from them and reports 0.00 without them -- which is what
            # the first version of this call produced, for the ground truth too,
            # i.e. a readout whose positive control was failing in plain sight.
            gt_pose = GT_MOTION / (clip + ".pkl")
            gt_joints = (np.asarray(_pickle.load(open(gt_pose, "rb"))["full_pose"], float)
                         if gt_pose.is_file() else None)
            if gt_labels is not None and gt_joints is not None:
                lanes.append(("ground truth",
                              np.asarray(gt_labels)[:len(gt_joints)], gt_joints))
            for title, path in available:
                payload = _pickle.load(open(path, "rb"))
                lanes.append((title, np.asarray(payload["atomic_labels"]),
                              np.asarray(payload["full_pose"], float)))
            music = None
            music_path = AUDIO / (clip + ".npy")
            if music_path.is_file():
                music = np.load(music_path)
            png = PS.render(clip, lanes, music=music, width=13.0)
            strip = staging / (stem + "_strip.png")
            strip.write_bytes(png)
        except ModuleNotFoundError as error:
            # Distinguished from the catch-all below on purpose: this is not
            # "this clip has no plan", it is "the plan lanes cannot be drawn for
            # ANY clip", and it used to read as a per-clip note.
            raise SystemExit(
                "error: the plan-strip renderer is not importable ({}).  Every "
                "clip would render without its plan lanes, which is the panel "
                "this strip exists for.  Run from the repository root.".format(error))
        except Exception as error:                                   # noqa: BLE001
            if not args.allow_missing_panels:
                raise SystemExit(
                    "error: the plan lanes could not be drawn for {} ({}: {}).  "
                    "That panel is the label sequence the strip exists to show, "
                    "so the render is refused rather than written without it.  "
                    "Pass --allow-missing-panels to accept a partial strip."
                    .format(stem, type(error).__name__, error))
            print("plan strip unavailable for {}: {}".format(stem, error))

        if shot is None and not args.allow_missing_panels:
            raise SystemExit(
                "error: no source footage for {}; the raw video panel would be "
                "missing.  Pass --allow-missing-panels to accept a partial "
                "strip.".format(stem))

        top = ([str(shot)] if shot else []) + [str(x) for x in sticks]
        seconds = frames_of(avatars)
        width = args.size * (len(available) + 1)
        run(mux_command(top, str(avatars), str(audio), str(target),
                        panel_height=args.size, width=width,
                        strip=(str(strip) if strip is not None else None),
                        seconds=seconds, rate=video_rate(avatars)))
        verify_layout(target, width=width, panels_top=len(top),
                      has_strip=strip is not None, seconds=seconds)
        print(target)
    shutil.rmtree(staging, ignore_errors=True)
    if skipped and not args.allow_missing_clips:
        raise SystemExit(
            "{} of {} clips did not render, so this batch is NOT the fixed ten "
            "and must not be compared against a round that was:\n{}\n"
            "Point STRIP_CONVERTED at a tree that has them (a staged one built "
            "by tools/stage_ground_truth_for_render.py counts, it is gated "
            "joint-for-joint against the eval pickle), or pass "
            "--allow-missing-clips if a subset is what you meant.".format(
                len(skipped), len(clips),
                "\n".join("  {} -> {}".format(stem, ", ".join(gaps))
                           for stem, gaps in skipped)))


if __name__ == "__main__":
    main()
