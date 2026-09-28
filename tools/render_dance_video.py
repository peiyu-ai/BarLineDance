"""Render decoded dance to a watchable video, with the music muxed in.

The deliverable that actually convinces anyone is a full-speed video with
audio -- Lodge's post-mortem is explicit that stitched, real-time, with-music
video is the only trustworthy human check.  This renders the SMPL skeleton
(24 joints, the same FK used everywhere else in the repo) from three fixed
views, encodes with ffmpeg, and muxes the conditioning audio.

Input is either a generated result ``.pkl`` from ``infer_atomic.py`` (keys
``smpl_poses [T,72]``, ``smpl_trans [T,3]``, ``full_pose [T,24,3]``) or a raw
151-D window from a release (decoded through the verified path first).

Rendering draws with matplotlib line collections at ~30 fps of wall time per
~10 s clip -- no mesh, no pytorch3d, nothing that can silently fake geometry:
what is on screen is exactly the joint positions the model produced.
"""

import argparse
import json
import pickle
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib
import matplotlib.font_manager

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# The panel labels are the caller's arm names, which in this repository are
# Chinese.  Without this every one of them was burned into the frame as tofu
# boxes -- silently, because a missing glyph is not an error.  The sibling tool
# tools/render_plan_grid_sheet.py has had this gate since it was written; this
# module did not, and shipped boxes.
# FONTS.  The panel labels must render, and "render" has two directions that
# have to be checked separately.
#
# WHAT WENT WRONG WHEN ONLY ONE WAS CHECKED.  This block used to pick the first
# available CJK font and put it FIRST in the family list, with a comment saying
# its cmap "was CHECKED, not assumed -- every glyph of 真值生成基线... is
# present".  That check was real and it was half of the job: on this host the
# winner is ``Droid Sans Fallback``, whose cmap contains **no Latin at all** --
# not even lowercase "g".  Once the operator asked for English labels, every
# title in every rendered panel came out as boxes, and the gate that existed to
# prevent exactly that reported success, because it only ever asked about CJK.
#
# So the order is Latin-first with the CJK font behind it (matplotlib >= 3.6
# falls back per GLYPH across this list, so both alphabets resolve), and the
# gate below is a coverage test over BOTH samples rather than an allowlist of
# names.  A name allowlist cannot fail for the reason that actually bit.
_LATIN_SAMPLE = ("abcdefghijklmnopqrstuvwxyz"
                 "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,:%-()/·")
_CJK_SAMPLE = "真值生成基线选择器接缝卡点舞蹈"


def _covered(family, sample):
    """Which characters of ``sample`` this family can actually draw."""
    try:
        from fontTools.ttLib import TTFont

        path = matplotlib.font_manager.findfont(
            matplotlib.font_manager.FontProperties(family=family),
            fallback_to_default=False)
        font = TTFont(path, fontNumber=0)
        points = set()
        for table in font["cmap"].tables:
            points |= set(table.cmap)
    except Exception:                                            # pragma: no cover
        return set()
    return {c for c in sample if ord(c) in points}


def _font_stack():
    """A family list whose COMBINED coverage draws both alphabets."""
    installed = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    latin = [n for n in ("DejaVu Sans", "Liberation Sans", "Arial")
             if n in installed]
    cjk = [n for n in ("Noto Sans CJK JP", "Noto Sans CJK SC",
                       "Noto Serif CJK JP", "Droid Sans Fallback")
           if n in installed]
    stack = latin + cjk
    if not stack:
        raise SystemExit("no usable font installed for panel labels")
    missing_latin = set(_LATIN_SAMPLE) - set().union(
        *[_covered(n, _LATIN_SAMPLE) for n in stack])
    if missing_latin:
        raise SystemExit(
            "no installed font covers the Latin panel labels; missing {} from "
            "{}.  Labels would render as boxes -- which is exactly what shipped "
            "when this gate only checked CJK."
            .format(sorted(missing_latin)[:8], stack))
    return stack


plt.rcParams["font.sans-serif"] = _font_stack()

# SMPL kinematic tree, matching vis.py / tools/preprocess_wild_3d.py.
SMPL_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
    20, 21,
]
FPS = 30

# Fixed camera azimuths (degrees); three views catch sideways motion a single
# front view hides.
VIEWS = (0, 90, 30)


def set_views(views):
    """Narrow the azimuth set for a caller that composes panels itself.

    Three views exist because "a single view hides sideways motion" (below),
    and that reason is unchanged when this renderer is the whole picture.  It
    stops applying when the stick figure is one panel of four beside an avatar
    pair -- there the three views cost half the strip's width and the sideways
    motion is already readable on the avatars.  Module-level rather than an
    argument because the value is read from six places inside the frame loop.
    """
    global VIEWS
    VIEWS = tuple(int(v) for v in views)


DPI = 80


def set_dpi(dpi):
    """Render the same figure at more pixels, for a caller that enlarges it.

    The figure is 4 inches per view, so the historical dpi 80 is 320 px -- fine
    beside two other views, soft when this panel is blown up to 640 px next to a
    480x832 cartoon, and a soft stick figure is exactly the panel a reviewer
    cannot judge "did the arm reach" on (CLAUDE.md 1.5.1: the body has to be big
    enough to read).  Default unchanged, so every existing caller and every
    cached video still means what it meant.
    """
    global DPI
    DPI = int(dpi)


ZOOM = 1.7


def _grid_shape(count):
    """(blocks across, rows of blocks) for ``count`` motions drawn together.

    One motion keeps the historical single-row frame byte-for-byte (960x320 at
    dpi 80), so every existing caller and every cached video still means what it
    meant.  Two or more go into two columns: four arms in a row would be 3840 px
    wide, which on a page capped at 1800 px leaves the dancer about two
    centimetres tall, and four stacked would be 960x1280 -- a column the reader
    has to scroll while the whole point is comparing them at a glance.
    """
    if count <= 1:
        return 1, 1
    return 2, (count + 1) // 2


def load_motion(path):
    """Return [T, 24, 3] joint positions plus optional contacts."""
    path = Path(path)
    if path.suffix == ".pkl":
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        return np.asarray(payload["full_pose"], dtype=np.float32), payload
    if path.suffix == ".npy":
        return _decode_raw_151(np.load(path))
    raise SystemExit("expected a generated .pkl or a raw 151-D .npy, got {}".format(path))


def _decode_raw_151(motion):
    """Decode an already-unnormalized 151-D array through the repo's own FK.

    Ground truth lives in the release as raw 151-D, and the only honest way to
    put it beside a generated clip is to run it through the same decode the
    generator's output took -- ``infer_atomic.decode_motion`` minus the
    unnormalization step that raw arrays have already had applied.
    """
    import torch

    from dataset.quaternion import ax_from_6v
    from vis import SMPLSkeleton

    motion = torch.as_tensor(np.asarray(motion, dtype=np.float32))
    if motion.ndim != 2 or motion.shape[1] != 151:
        raise SystemExit("expected raw motion of shape [T, 151], got {}".format(tuple(motion.shape)))
    contacts, values = torch.split(motion, (4, 147), dim=-1)
    root_positions = values[:, :3]
    rotations = ax_from_6v(values[:, 3:].reshape(-1, 24, 6))
    full_pose = SMPLSkeleton().forward(rotations.unsqueeze(0), root_positions.unsqueeze(0))[0]
    payload = {
        "smpl_poses": rotations.reshape(-1, 72).numpy(),
        "smpl_trans": root_positions.numpy(),
        "full_pose": full_pose.numpy(),
        "contacts": contacts.numpy(),
        "source": "raw_151d",
    }
    return payload["full_pose"].astype(np.float32), payload


def draw_frame(axes, pose, ranges):
    for view_index, ax in enumerate(axes):
        ax.cla()
        ax.set_axis_off()
        azim = VIEWS[view_index]
        ax.view_init(elev=12, azim=azim)
        # Matplotlib's default 3D framing leaves most of the panel empty; the
        # dancer is the subject, so zoom in on the cube we actually set.
        ax.set_box_aspect((1, 1, 1), zoom=ZOOM)
        x, y, z = pose[:, 0], pose[:, 1], pose[:, 2]
        for joint, parent in enumerate(SMPL_PARENTS):
            if parent < 0:
                continue
            ax.plot(
                [x[joint], x[parent]],
                [y[joint], y[parent]],
                [z[joint], z[parent]],
                color="#1f77b4", linewidth=2.0,
            )
        ax.scatter(x, y, z, s=8, color="#d62728")
        (xr, yr, zr) = ranges
        ax.set_xlim(*xr)
        ax.set_ylim(*yr)
        ax.set_zlim(*zr)


def axis_ranges(motion, follow_root=False):
    """Return a per-frame ranges callable for the 3D axes.

    Fixed ranges keep the camera still, but a dancer who travels across the
    floor then occupies a few percent of the frame and nothing is legible.
    ``follow_root`` tracks the root in the floor plane only and leaves the
    vertical axis anchored to the floor, so travel stops shrinking the figure
    while jumps and crouches still read as vertical motion.
    """
    mins = motion.reshape(-1, 3).min(axis=0)
    maxs = motion.reshape(-1, 3).max(axis=0)
    if not follow_root:
        center = (mins + maxs) / 2
        radius = float((maxs - mins).max()) / 2 + 0.2
        fixed = tuple((center[i] - radius, center[i] + radius) for i in range(3))
        return lambda frame: fixed

    root = motion[:, 0, :]
    # A percentile, not the max: one flailing frame in a generated clip would
    # otherwise set the zoom for the whole video.
    radius = float(np.percentile(np.abs(motion - root[:, None, :]), 99.5)) + 0.15
    floor = float(mins[2])
    vertical = (floor, floor + 2.0 * radius)

    def ranges(frame):
        cx, cy = float(root[frame, 0]), float(root[frame, 1])
        return ((cx - radius, cx + radius), (cy - radius, cy + radius), vertical)

    return ranges


def _draw_block_labels(figure, blocks, titles, across, down):
    """Name each block inside its own corner, and rule the gutters between them.

    ``figure.text`` rather than ``ax.set_title``: ``draw_frame`` calls
    ``ax.cla()`` on every axes every frame, so an axes-level title survives
    exactly one frame.  The rules matter because the three views inside a block
    are packed edge to edge (``wspace=0``) -- without them a four-arm frame
    reads as twelve undifferentiated panels rather than as four dancers.
    """
    for block, title in zip(blocks, titles):
        if not title:
            continue
        boxes = [ax.get_position() for ax in block]
        figure.text(min(b.x0 for b in boxes) + 0.004,
                    max(b.y1 for b in boxes) - 0.008,
                    title, ha="left", va="top", fontsize=11, color="#111",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.75,
                              boxstyle="round,pad=0.25"))
    for index in range(1, across):
        figure.add_artist(plt.Line2D([index / across] * 2, [0, 1],
                                     color="#c8c8c8", linewidth=1.2))
    for index in range(1, down):
        figure.add_artist(plt.Line2D([0, 1], [index / down] * 2,
                                     color="#c8c8c8", linewidth=1.2))


def _extent(motion):
    """(centre, half-extent) of one motion, in the units the cube is built from."""
    flat = np.asarray(motion, dtype=np.float32).reshape(-1, 3)
    mins, maxs = flat.min(axis=0), flat.max(axis=0)
    return (mins + maxs) / 2, float((maxs - mins).max()) / 2


def _fixed_ranges(centre, radius):
    fixed = tuple((float(centre[i] - radius), float(centre[i] + radius))
                  for i in range(3))
    return lambda frame: fixed  # noqa: E731 - matches axis_ranges' callable shape


def cube_from(motion, margin=0.2):
    """The fixed cube ``axis_ranges`` derives from ``motion``, as six floats."""
    centre, half = _extent(motion)
    radius = half + margin
    return [v for i in range(3) for v in (centre[i] - radius, centre[i] + radius)]


FOOTAGE_HEIGHT = 480


def decode_footage(path, out_fps, staging):
    """The clip's own frames, resampled to ``out_fps`` BY TIMESTAMP.

    ``fps=`` maps the footage's seconds onto the video's seconds, which is the
    whole point: the skeleton panels advance at a fixed 30 fps by construction,
    so if a sequence's declared frame rate disagrees with the footage it was
    reconstructed from, the two dancers visibly drift apart instead of the
    error being resampled away.  Index-for-index copying would hide exactly the
    defect this panel was added to show.

    Returns (frame paths, frames, seconds).  The frames are files rather than
    an array because a 60 s clip at 480 px is ~200 MB in memory and the loop
    needs one frame at a time.
    """
    out_dir = staging / "footage"
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
         "-vf", "fps={},scale=-2:{}".format(out_fps, FOOTAGE_HEIGHT),
         "-start_number", "0", str(out_dir / "g%06d.png")],
        check=True, stdin=subprocess.DEVNULL)
    paths = sorted(out_dir.glob("g*.png"))
    if not paths:
        raise SystemExit("ffmpeg decoded no frames from {}".format(path))
    return paths, len(paths), len(paths) / float(out_fps)


def draw_footage(ax, paths, index, ended_note):
    """One footage frame, holding the last one once the clip has run out.

    Held rather than blanked, and *labelled* while held: a frozen dancer with
    no label reads as a frozen generation, which is a defect of a completely
    different kind.

    The length reading is NOT drawn here: ``imshow`` shrinks the axes to the
    image's aspect, so an axes-relative placement lands on top of a portrait
    clip instead of in the margin beside it.  It goes on the figure, once,
    where the margin actually is.
    """
    ax.cla()
    ax.set_axis_off()
    ax.imshow(plt.imread(str(paths[min(index, len(paths) - 1)])))
    if index >= len(paths):
        ax.text(0.5, 0.03, ended_note, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=10, color="#b02a2a",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8,
                          boxstyle="round,pad=0.25"))


def render(motions, output, audio=None, titles=None, stride=1, follow_root=False,
           cube=None, footage=None, footage_title="原视频"):
    """Draw one motion, or several side by side, into a single video.

    Several motions go into one figure rather than into several files stacked
    afterwards, and that is the whole point rather than a convenience.
    ``axis_ranges`` derives its cube from the motion it is handed, so N
    separately-rendered videos are N different metre-per-pixel scales -- and
    since ``draw_frame`` calls ``ax.set_axis_off()`` there is no tick, grid,
    floor or scale bar anywhere in the frame to reveal it.  Placed side by side
    they look comparable and are not: the arm that travels furthest gets the
    largest cube, is therefore drawn smallest, and its drift reads as the
    calmest of the four -- a model that got worse producing a picture that looks
    better.  Drawn together they share one cube by construction, and they are
    literally the same frames, so the panels cannot fall out of sync either.
    """
    if not isinstance(motions, (list, tuple)):
        motions = [motions]
    motions = [np.asarray(m, dtype=np.float32) for m in motions]
    if titles is None or isinstance(titles, str):
        titles = [titles] * len(motions)
    if len(titles) != len(motions):
        raise SystemExit("got {} motion(s) but {} title(s)".format(
            len(motions), len(titles)))
    if follow_root and len(motions) > 1:
        raise SystemExit(
            "--follow-root re-centres the camera on the root every frame, which "
            "removes root translation from the picture by construction; with "
            "several motions in one frame that is the thing worth comparing, so "
            "the combination is refused rather than silently ranked")
    if follow_root and footage is not None:
        raise SystemExit(
            "--follow-root moves the skeleton camera every frame while the "
            "footage panel cannot move with it, so travel would read as the "
            "dancer standing still beside a dancer who travels. Pick one.")
    if FPS % stride:
        # `-framerate FPS // stride` is integer division: stride 4 asks for 7.5
        # fps and gets 7 (playback 7% slow), stride 7 asks for 4.29 and gets 4,
        # and stride > FPS gives `-framerate 0`, which fails inside ffmpeg with
        # an opaque error.  Meanwhile the sidecar asserted "real-time" whatever
        # happened -- a claim that could not fail.  Refuse at the argument.
        raise SystemExit("stride {} does not divide {} fps; use one of {}".format(
            stride, FPS, [s for s in range(1, FPS + 1) if FPS % s == 0]))

    frames = max(len(m) for m in motions)
    if cube is not None:
        if follow_root:
            raise SystemExit("--axis-cube fixes the camera; --follow-root moves "
                             "it every frame. Pick one.")
        if len(cube) != 6:
            raise SystemExit("--axis-cube wants 6 floats (x0 x1 y0 y1 z0 z1), "
                             "got {}".format(len(cube)))
        fixed = tuple((float(cube[i]), float(cube[i + 1])) for i in (0, 2, 4))
        radius = (fixed[0][1] - fixed[0][0]) / 2
        centres = [np.array([sum(a) / 2 for a in fixed])] * len(motions)
        ranges_for = [(lambda frame, f=fixed: f)] * len(motions)
    elif follow_root:
        radius, centres = None, [None] * len(motions)
        ranges_for = [axis_ranges(m, True) for m in motions]
    else:
        # Shared SCALE, own CENTRE -- not one cube over the concatenation.
        # A single cube over everything is the wrong shared quantity here: the
        # arms sit in different parts of the floor, so their union is set by the
        # offset BETWEEN them rather than by any dancer's size.  Measured on
        # 7438547996335295781:clip001, the four motions' own cubes are 2.31 m to
        # 3.17 m across and their union is 5.38 m, so two thirds of every panel
        # would be the empty floor between arms and every skeleton would be
        # drawn at 43% of the size it needs to be legible.
        #
        # What has to match for panels to be comparable is metres per pixel; the
        # origin does not, because where on the floor a generation happens to
        # start is arbitrary and carries no claim.  So one radius -- the largest
        # any motion needs, which is a real bound and not a percentile that
        # could hide an excursion -- and each panel centred on its own motion.
        centres, halves = zip(*(_extent(m) for m in motions))
        radius = max(halves) + 0.2
        ranges_for = [_fixed_ranges(c, radius) for c in centres]

    # The footage, when given, is a block like any other -- first, so it reads
    # as the thing the others are claims about.
    panels = len(motions) + (1 if footage is not None else 0)
    panel_titles = ([footage_title] if footage is not None else []) + list(titles)
    across, down = _grid_shape(panels)
    figure, axes = plt.subplots(
        down, across * len(VIEWS),
        figsize=(4 * across * len(VIEWS), 4 * down),
        subplot_kw={"projection": "3d"},
    )
    axes = np.atleast_2d(axes)
    all_blocks = [list(axes[row, column * len(VIEWS):(column + 1) * len(VIEWS)])
                  for row, column in (divmod(i, across) for i in range(panels))]
    # An odd count leaves a hole; blank it rather than leaving a default 3D box.
    for row, column in (divmod(i, across) for i in range(panels, across * down)):
        for ax in axes[row, column * len(VIEWS):(column + 1) * len(VIEWS)]:
            ax.set_axis_off()

    if panels == 1:
        # Unchanged from before multi-motion support, down to the 0.95 band, so
        # a single-result render still produces the frame its cached videos and
        # its other caller were made with.
        if panel_titles[0]:
            figure.suptitle(panel_titles[0], fontsize=10)
        figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.95, wspace=0.0)
    else:
        figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0,
                               wspace=0.0, hspace=0.0)
        _draw_block_labels(figure, all_blocks, panel_titles, across, down)

    # Collapse the footage block's three 3D axes into one 2D axes over the same
    # rectangle.  After ``subplots_adjust``, because a manually added axes is
    # not laid out by it and would otherwise sit where the axes used to be.
    footage_ax, footage_rect, blocks = None, None, all_blocks
    if footage is not None:
        boxes = [ax.get_position() for ax in all_blocks[0]]
        rect = [min(b.x0 for b in boxes), min(b.y0 for b in boxes),
                max(b.x1 for b in boxes) - min(b.x0 for b in boxes),
                max(b.y1 for b in boxes) - min(b.y0 for b in boxes)]
        for ax in all_blocks[0]:
            ax.remove()
        footage_ax = figure.add_axes(rect)
        footage_ax.set_axis_off()
        footage_rect = rect
        blocks = all_blocks[1:]

    with tempfile.TemporaryDirectory() as staging:
        staging = Path(staging)
        footage_paths = footage_frames = footage_seconds = None
        if footage is not None:
            footage_paths, footage_frames, footage_seconds = decode_footage(
                footage, FPS // stride, staging)
        info_note = ""
        if footage_seconds is not None:
            ratio = footage_seconds / (frames / float(FPS))
            info_note = "原视频 {} 帧 / {:.2f}s\n序列 {} 帧 / {:.2f}s\n比值 {:.3f}".format(
                footage_frames, footage_seconds, frames, frames / float(FPS), ratio)
            if abs(ratio - 1.0) <= 1.0 / FPS:
                info_note += "\n（一致）"
            else:
                info_note += "\n（不一致：序列的帧率与它的素材对不上）"
        ended_note = ("原视频到此结束（{:.2f}s）".format(footage_seconds)
                      if footage_seconds is not None
                      and footage_seconds < frames / float(FPS) - 1.0 / FPS else "")
        if info_note:
            figure.text(footage_rect[0] + 0.006,
                        footage_rect[1] + footage_rect[3] - 0.055, info_note,
                        ha="left", va="top", fontsize=9.5, linespacing=1.6,
                        color="#b02a2a" if ended_note else "#444")
        written = 0
        for frame in range(0, frames, stride):
            if footage_ax is not None:
                draw_footage(footage_ax, footage_paths, written, ended_note)
            for block, motion, ranges in zip(blocks, motions, ranges_for):
                # Hold the last pose rather than clamping every block to the
                # shortest motion: clamping hides which arm ran long, and cuts
                # the ground truth whenever an arm is the longer one.
                draw_frame(block, motion[min(frame, len(motion) - 1)],
                           ranges(frame))
            figure.savefig(staging / "f{:06d}.png".format(written), dpi=DPI)
            written += 1

        video_only = staging / "video.mp4"
        encode = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(FPS // stride),
            "-i", str(staging / "f%06d.png"),
            "-pix_fmt", "yuv420p", "-crf", "23",
            str(video_only),
        ]
        # stdin=DEVNULL: ffmpeg reads stdin when it inherits one, so a driver
        # that renders inside a `while read` loop loses the rest of its
        # clip list to the encoder -- silently, as truncated file names.
        subprocess.run(encode, check=True, stdin=subprocess.DEVNULL)

        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if audio is not None:
            # Pad the audio; never truncate the video.  This was `-shortest`,
            # whose comment named only one direction ("the audio track may
            # outlast the rendered window").  In the other direction ffmpeg cuts
            # the *video* at the wav's end, and on 2026-08-24 that was found to
            # have silently deleted 25.1% of two clips on this corpus --
            # 7438547996335295781_c001 rendered 545 frames and the file held
            # 408 -- while the sidecar went on asserting 545.  `apad` plus an
            # explicit -t keeps every rendered frame and lets a short wav run
            # out into silence, which is a visible condition a page can label.
            seconds = written / float(FPS // stride)
            mux = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(video_only), "-i", str(audio),
                "-map", "0:v", "-map", "1:a",
                # MP3 rather than AAC: an editor's built-in preview plays this
                # and is silent on AAC (measured 2026-08-29 with the video stream
                # held byte-identical), and every probe calls the AAC track fine.
                "-c:v", "copy", "-c:a", "libmp3lame", "-ar", "44100", "-ac", "2",
                "-b:a", "160k", "-af", "apad",
                "-t", "{:.6f}".format(seconds),
                "-movflags", "+faststart",
                str(output),
            ]
            subprocess.run(mux, check=True, stdin=subprocess.DEVNULL)
        else:
            # shutil.move, not Path.replace: the staging dir is under
            # TMPDIR (local disk) while an output/ path is on the NAS, and
            # os.replace across filesystems raises EXDEV -- after the whole
            # render has already been paid for.
            shutil.move(str(video_only), str(output))
    plt.close(figure)
    return written, {"frames": footage_frames, "seconds": footage_seconds}


def _probe(path):
    """(frames actually in the file, video seconds, audio seconds or None).

    ``-count_frames`` decodes rather than trusting the container's header,
    because the header is exactly what was right while the file was wrong.
    """
    def ask(args):
        done = subprocess.run(["ffprobe", "-v", "error"] + args + [str(path)],
                              capture_output=True, text=True)
        try:
            return float(done.stdout.strip().split("\n")[0])
        except (ValueError, IndexError):
            return None

    return (ask(["-select_streams", "v:0", "-count_frames",
                 "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0"]),
            ask(["-select_streams", "v:0",
                 "-show_entries", "stream=duration", "-of", "csv=p=0"]),
            ask(["-select_streams", "a:0",
                 "-show_entries", "stream=duration", "-of", "csv=p=0"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True, action="append",
                        help="generated .pkl from infer_atomic.py; repeat it to "
                             "draw several motions into ONE video sharing one "
                             "axis cube -- which is the only way panels shown "
                             "side by side are on the same scale")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audio", type=Path, default=None,
                        help="conditioning .wav; muxed in when given, padded "
                             "with silence if it is shorter than the motion")
    parser.add_argument("--title", default=None, action="append",
                        help="panel label; repeat once per --result. An empty "
                             "string suppresses that panel's label -- which is "
                             "distinct from omitting the flag, and used to mean "
                             "the same thing")
    parser.add_argument("--stride", type=int, default=1,
                        help="render every Nth frame (2 halves wall time; "
                             "output fps is reduced to match, so playback "
                             "speed stays real-time). Must divide {}".format(FPS))
    parser.add_argument("--axis-cube", type=float, nargs=6, default=None,
                        metavar=("X0", "X1", "Y0", "Y1", "Z0", "Z1"),
                        help="impose this cube on every panel instead of "
                             "deriving one from the motions handed in. Use it "
                             "when the group's scale should be set by one "
                             "member (say the ground truth) rather than by "
                             "whichever arm travelled furthest")
    parser.add_argument("--footage", type=Path, default=None,
                        help="the clip's own video (data/wild_ingest_v1/<clip>/"
                             "clip.mp4), drawn as the first panel. Resampled to "
                             "the output rate BY TIMESTAMP, so a sequence whose "
                             "declared frame rate disagrees with the footage "
                             "drifts visibly instead of being resampled into "
                             "agreement")
    parser.add_argument("--footage-title", default="原视频",
                        help="label for the footage panel")
    parser.add_argument("--follow-root", action="store_true",
                        help="track the root in the floor plane so a travelling "
                             "dancer stays legible; height stays floor-anchored. "
                             "Refused with several --result, because it deletes "
                             "root translation from the picture")
    args = parser.parse_args()

    motions = []
    for result in args.result:
        motion, _payload = load_motion(result)
        motions.append(motion)
    titles = args.title if args.title is not None else [r.stem for r in args.result]
    if len(titles) != len(args.result):
        raise SystemExit("got {} --result but {} --title".format(
            len(args.result), len(titles)))

    frames, footage_meta = render(motions, args.output, args.audio, titles,
                                  args.stride, args.follow_root, args.axis_cube,
                                  args.footage, args.footage_title)
    shared_radius = (None if args.follow_root or args.axis_cube is not None else
                     max(_extent(m)[1] for m in motions) + 0.2)

    in_file, video_seconds, muxed_audio_seconds = _probe(args.output)
    # The SOURCE wav, not the muxed track.  `apad` pads the muxed one out to the
    # video's length by construction, so a coverage test against it reads `true`
    # for every clip -- including the ones whose tail is silence.  That is a gate
    # that cannot fail, which is the shape this repository keeps paying for.
    source_audio_seconds = None if args.audio is None else _probe(args.audio)[2]
    motion_seconds = max(len(m) for m in motions) / float(FPS)
    meta = {
        "result": [str(r) for r in args.result],
        "output": str(args.output),
        "audio": str(args.audio) if args.audio else None,
        "titles": ([args.footage_title] if args.footage else []) + titles,
        "panels": len(motions) + (1 if args.footage else 0),
        "footage": str(args.footage) if args.footage else None,
        "footage_frames": footage_meta["frames"],
        "footage_seconds": footage_meta["seconds"],
        "frames_rendered": frames,
        "frames_in_file": None if in_file is None else int(in_file),
        "motion_frames": [int(len(m)) for m in motions],
        "views_azimuth_deg": list(VIEWS),
        "camera": "root-following (floor plane)" if args.follow_root else "fixed",
        "axis_cube": list(args.axis_cube) if args.axis_cube is not None else None,
        "axis_half_extent_m": shared_radius,
        "panel_centres": (None if shared_radius is None else
                          [[round(float(v), 4) for v in _extent(m)[0]] for m in motions]),
        "scale_policy": ("root-following, radius from this motion" if args.follow_root
                         else "cube imposed by the caller" if args.axis_cube is not None
                         else "shared half-extent {:.4f} m across {} panel(s), each "
                              "centred on its own motion".format(
                                  shared_radius, len(motions))),
        "output_fps": FPS // args.stride,
        "video_seconds": video_seconds,
        "muxed_audio_seconds": muxed_audio_seconds,
        "source_audio_seconds": source_audio_seconds,
        "motion_seconds": motion_seconds,
        "audio_covers_motion": (source_audio_seconds is not None
                                and source_audio_seconds >= motion_seconds - 1.0 / FPS),
        # Not "did it decode" -- whether the footage spans the sequence the
        # panels claim to be reconstructions of.  A sequence declared at 30 fps
        # whose footage is a third shorter is not a short clip, it is a frame
        # rate that does not match its own source, and this is the field that
        # says so per artifact rather than per corpus.
        "footage_covers_motion": (footage_meta["seconds"] is not None
                                  and footage_meta["seconds"] >= motion_seconds - 1.0 / FPS),
        "footage_vs_motion_ratio": (None if footage_meta["seconds"] is None
                                    else round(footage_meta["seconds"] / motion_seconds, 4)),
        "silent_tail_seconds": (None if source_audio_seconds is None else
                                max(0.0, motion_seconds - source_audio_seconds)),
    }
    # This used to be written from the PNG counter, before the mux -- so when
    # `-shortest` dropped a quarter of the frames the sidecar still asserted the
    # full count, and nothing could contradict it.  Compare it to the artifact.
    if in_file is None:
        raise SystemExit("ffprobe could not count the frames in {}".format(args.output))
    if abs(in_file - frames) > 1:
        raise SystemExit("rendered {} frames but {} holds {}: the mux dropped "
                         "video".format(frames, args.output, int(in_file)))

    sidecar = args.output.with_suffix(".render.json")
    sidecar.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                       encoding="utf-8")
    print("rendered {} frames x {} panel(s) -> {}".format(
        frames, meta["panels"], args.output))


if __name__ == "__main__":
    main()
