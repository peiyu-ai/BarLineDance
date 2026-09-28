#!/usr/bin/env python3
"""Play a sub-prototype: its members dancing side by side, not posed side by side.

The card renderer draws motion beats, which answers "do these settle into the
same shapes".  It cannot answer "is this the same *move*", because a move is a
process -- the paper's own first criterion for an atomic movement is that it
"involves clear processes", and a still frame is exactly what that criterion
rules out.  So this renders each sub-prototype as a short looping video with one
member per cell.

One thing here is deliberately *not* what the descriptor does.  ``canonical_pose``
re-canonicalises every frame independently, turning the hips back onto +x each
time; that is correct for a descriptor, and catastrophic for a video, because a
dancer spinning through 360 degrees would be re-aligned every frame and appear
to stand still while the world turns.  So a segment is canonicalised **once**,
from its first frame, and that single transform is applied to the whole segment.
Rotation and travel then survive to the screen, which is the entire point of
showing motion instead of poses.  The consequence is worth stating plainly: what
you watch here is *more* than what the w/o-LLM clustering saw, so a group that
looks inconsistent on turns may still be consistent in the space it was built in.

Members are drawn from distinct uploads, and cells loop independently at their
own natural speed rather than being time-warped onto a common length, because
duration is part of what the vocabulary is supposed to have grouped.

Usage:
    python3 tools/render_prototype_videos.py \\
        --labels data/wild3d/wild_v2_ingroup_nollm \\
        --bundle data/wild3d/wild_performance_v1 \\
        --captions runs/wild_v2_captions_30b/captions.jsonl \\
        --output runs/wild_v2_videos_nollm
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.build_atomic_gallery import collect_members, spread_over_uploads  # noqa: E402
from tools.motion_beats import L_HIP, L_SHOULDER, R_HIP, R_SHOULDER, ROOT  # noqa: E402
from tools.render_prototype_cards import (  # noqa: E402
    BODY_JOINTS,
    CardError,
    caption_text,
    load_caption_index,
    project,
    select_subprototypes,
)

FPS = 30.0


class VideoError(RuntimeError):
    pass


def segment_transform(segment: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """The single canonicalising transform for a whole segment, from its first frame.

    Returns the origin to subtract, the rotation to apply and the scale to divide
    by.  Fixing all three at frame 0 is what lets rotation and travel reach the
    screen; recomputing them per frame -- which is right for the descriptor --
    would subtract exactly the motion a viewer is trying to see.
    """
    first = np.asarray(segment[0], dtype=np.float64)
    origin = first[ROOT].copy()
    hips = first[L_HIP] - first[R_HIP]
    angle = np.arctan2(hips[1], hips[0])
    cos, sin = np.cos(-angle), np.sin(-angle)
    rotation = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    scale = float(np.linalg.norm(first[L_SHOULDER] - first[R_SHOULDER]))
    return origin, rotation, scale if scale > 1e-6 else 1.0


def canonical_segment(segment: np.ndarray) -> np.ndarray:
    """Apply one fixed transform to every frame of a segment."""
    origin, rotation, scale = segment_transform(segment)
    moved = np.asarray(segment, dtype=np.float64) - origin
    return (moved @ rotation.T) / scale


def frame_extent(tracks: Sequence[np.ndarray], azimuth: float) -> Tuple[float, float]:
    """A shared view box, so cells are not each at their own silent zoom level.

    Per-cell autoscaling would make a small tight move and a large travelling one
    fill the frame identically, which is precisely the comparison this page
    exists to support.
    """
    horizontal, vertical = [], []
    for track in tracks:
        flat = project(track.reshape(-1, 3), azimuth)
        horizontal.append(np.abs(flat[:, 0]).max())
        vertical.append(flat[:, 1])
    span = max(2.2, float(np.max(horizontal)) * 1.05)
    low = min(-2.6, float(np.min([part.min() for part in vertical])) * 1.05)
    high = max(2.2, float(np.max([part.max() for part in vertical])) * 1.05)
    return span, (low, high)


def render_video(tracks: Sequence[np.ndarray], entries: Sequence[Dict],
                 captions: Dict[Tuple[str, int, int], Dict], *, output: pathlib.Path,
                 label: int, tag: str, size: int, uploads: int, azimuth: float,
                 seconds: float, fps: float,
                 hide_uploads: bool = False,
                 title_note: str = "") -> Optional[pathlib.Path]:
    """One MP4: a cell per member, each looping its own segment."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from vis import smpl_parents

    if not tracks:
        return None
    total = int(round(seconds * fps))
    span, (low, high) = frame_extent(tracks, azimuth)
    palette = ["#7fd1ff", "#ffd479", "#9ae6a0", "#ff9ab0", "#c6a8ff", "#8ce0d4"]

    figure, axes = plt.subplots(1, len(tracks), figsize=(2.4 * len(tracks), 3.6),
                                squeeze=False)
    figure.patch.set_facecolor("#16161a")
    lines, scatters = [], []
    for cell, (axis, track, entry) in enumerate(zip(axes[0], tracks, entries)):
        axis.set_facecolor("#16161a")
        axis.set_xlim(-span, span)
        axis.set_ylim(low, high)
        axis.set_aspect("equal")
        axis.axis("off")
        colour = palette[cell % len(palette)]
        bones = [axis.plot([], [], color=colour, linewidth=1.8,
                           solid_capstyle="round", zorder=2)[0]
                 for parent in smpl_parents[:BODY_JOINTS] if parent >= 0]
        lines.append(bones)
        scatters.append(axis.scatter([], [], s=4.0, color=colour, zorder=3))
        duration = (entry["end"] - entry["start"]) / FPS
        axis.set_title("{:.2f}s\n{}".format(duration, entry["upload"][:26]),
                       color="#c9c9d4", fontsize=6.5, pad=4)

    # See render_prototype_cards.render_card for why this can be suppressed.
    # See render_prototype_cards.render_card for why both of these are here.
    figure.suptitle("{}sub-prototype {}   “{}”   {} segments{}".format(
        title_note + "   |   " if title_note else "",
        label, tag, size, "" if hide_uploads else ", {} uploads".format(uploads)),
        color="#f2f2f5", fontsize=9, y=0.985)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    def draw(step: int):
        artists = []
        for cell, track in enumerate(tracks):
            pose = project(track[step % len(track)][:BODY_JOINTS], azimuth)
            bone = 0
            for joint, parent in enumerate(smpl_parents[:BODY_JOINTS]):
                if parent < 0:
                    continue
                lines[cell][bone].set_data([pose[joint, 0], pose[parent, 0]],
                                           [pose[joint, 1], pose[parent, 1]])
                artists.append(lines[cell][bone])
                bone += 1
            scatters[cell].set_offsets(pose)
            artists.append(scatters[cell])
        return artists

    anim = animation.FuncAnimation(figure, draw, frames=total, interval=1000.0 / fps,
                                   blit=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(fps=fps, codec="libx264",
                                    extra_args=["-pix_fmt", "yuv420p", "-crf", "28"])
    anim.save(str(output), writer=writer, savefig_kwargs={"facecolor": "#16161a"})
    plt.close(figure)
    return output


def build(*, labels_dir: pathlib.Path, bundle: pathlib.Path, output_dir: pathlib.Path,
          captions_path: Optional[pathlib.Path], subprototypes: int, members: int,
          seconds: float, fps: float, azimuth: float, selection: str,
          min_uploads: int, seed: int,
          hide_uploads: bool = False, title_note: str = "") -> Dict[str, object]:
    from tools.cluster_atomics_tmr import build_row_index, resolve_row
    from tools.convert_motion_to_guofeats import motion_151_to_joints

    if shutil.which("ffmpeg") is None:
        raise VideoError("ffmpeg is not on PATH; this tool encodes MP4 with it")

    grouped = collect_members(labels_dir, bundle)
    if not grouped:
        raise VideoError("{} yielded no labelled segments".format(labels_dir))
    tags_path = labels_dir / "subprototype_tags.json"
    tags = json.loads(tags_path.read_text(encoding="utf-8")) if tags_path.exists() else {}
    captions = load_caption_index(captions_path)

    chosen, _ = select_subprototypes(grouped, count=subprototypes, members=members,
                                     strategy=selection, min_uploads=min_uploads)
    if not chosen:
        raise VideoError("no sub-prototype had at least {} distinct uploads".format(
            min_uploads))
    picked = {label: spread_over_uploads(entries, members) for label, entries, _ in chosen}

    rows = {json.loads(line)["recording_id"]: json.loads(line)
            for line in (bundle / "sequences.jsonl").open(encoding="utf-8")}
    index = build_row_index(rows)
    cache: Dict[str, np.ndarray] = {}

    def joints_for(recording: str) -> Optional[np.ndarray]:
        if recording not in cache:
            row = resolve_row(index, recording) or rows.get(recording)
            if row is None:
                return None
            cache[recording] = motion_151_to_joints(np.load(bundle / row["motion_path"]))
        return cache[recording]

    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = output_dir / "clips"
    manifest: List[Dict[str, object]] = []
    for label, entries, uploads in chosen:
        tracks, kept = [], []
        for entry in picked[label]:
            joints = joints_for(entry["recording_id"])
            if joints is None:
                continue
            segment = joints[entry["start"]:entry["end"]]
            if len(segment) < 4:
                continue
            tracks.append(canonical_segment(segment)[:, :BODY_JOINTS])
            kept.append(entry)
        if not tracks:
            continue
        tag = str(tags.get(str(label), "(no LLM tag: this bundle is the w/o-LLM row)"))
        path = clips_dir / "sub_{:04d}.mp4".format(label)
        try:
            render_video(tracks, kept, captions, output=path, label=label, tag=tag,
                         size=len(entries), uploads=uploads, azimuth=azimuth,
                         seconds=seconds, fps=fps, hide_uploads=hide_uploads,
                         title_note=title_note)
        except Exception as error:            # ffmpeg or codec trouble, per clip
            print("  sub-prototype {} failed to encode: {}".format(label, error),
                  flush=True)
            continue
        manifest.append({
            "label": int(label), "tag": tag, "size": int(len(entries)),
            "uploads": int(uploads), "path": str(path.relative_to(output_dir)),
            "members": [{
                "recording_id": entry["recording_id"],
                "seconds": round((entry["end"] - entry["start"]) / FPS, 2),
                "caption": caption_text(captions.get(
                    (entry["recording_id"], entry["start"], entry["end"]))),
            } for entry in kept],
        })
        print("  encoded sub-prototype {} ({} members, {} segments)".format(
            label, len(tracks), len(entries)), flush=True)

    if not manifest:
        raise VideoError("no sub-prototype produced a clip")
    report = {
        "labels": str(labels_dir), "clips": len(manifest),
        "subprototypes_total": len(grouped), "members_per_clip": members,
        "seconds": seconds, "fps": fps, "view_azimuth_degrees": azimuth,
        "selection": selection, "seed": seed,
        "canonicalisation": ("one transform per segment taken from its first frame, "
                             "so rotation and travel survive; the descriptor "
                             "re-canonicalises every frame and does not see them"),
    }
    write_page(output_dir, manifest, report)
    (output_dir / "videos.json").write_text(
        json.dumps({**report, "manifest": manifest}, indent=2) + "\n", encoding="utf-8")
    return report


def write_page(output_dir: pathlib.Path, manifest: Sequence[Dict[str, object]],
               report: Dict[str, object]) -> None:
    parts = [
        "<title>Atomic sub-prototypes, playing</title>",
        "<style>",
        ":root{--bg:#fbfbfd;--fg:#1a1a1f;--muted:#5b5b6b;--card:#fff;--line:#e3e3ea}",
        "@media (prefers-color-scheme:dark){:root:not([data-theme='light'])",
        "{--bg:#111114;--fg:#f0f0f4;--muted:#9a9aab;--card:#1a1a20;--line:#2c2c36}}",
        ":root[data-theme='dark']{--bg:#111114;--fg:#f0f0f4;--muted:#9a9aab;",
        "--card:#1a1a20;--line:#2c2c36}",
        "body{background:var(--bg);color:var(--fg);margin:0;padding:2rem 1.25rem;",
        "font:15px/1.6 system-ui,-apple-system,'Segoe UI',sans-serif}",
        ".wrap{max-width:1100px;margin:0 auto}",
        "h1{font-size:1.45rem;margin:0 0 .4rem}.sub{color:var(--muted);margin:0 0 1.5rem}",
        ".card{background:var(--card);border:1px solid var(--line);border-radius:10px;",
        "padding:.9rem;margin-bottom:1.1rem}",
        ".hd{display:flex;gap:.8rem;align-items:baseline;flex-wrap:wrap;margin-bottom:.5rem}",
        ".tag{font-weight:650}.meta{color:var(--muted);font-size:.85rem}",
        "video{width:100%;height:auto;border-radius:6px;display:block;background:#16161a}",
        "ul{margin:.7rem 0 0;padding-left:1.1rem;color:var(--muted);font-size:.82rem}",
        "</style>",
        "<div class='wrap'>",
        "<h1>Atomic sub-prototypes, playing</h1>",
        "<p class='sub'>One cell per member segment, each from a different upload and "
        "each looping at its own natural speed. Every segment is canonicalised once "
        "from its first frame, so turns and travel are visible &mdash; unlike the "
        "descriptor, which re-canonicalises every frame and therefore never sees them.</p>",
    ]
    for row in manifest:
        parts.append("<div class='card'><div class='hd'>")
        parts.append("<span class='tag'>#{} &ldquo;{}&rdquo;</span>".format(
            row["label"], html.escape(str(row["tag"]))))
        parts.append("<span class='meta'>{} segments &middot; {} uploads</span>".format(
            row["size"], row["uploads"]))
        parts.append("</div>")
        parts.append("<video controls loop muted playsinline preload='metadata' "
                     "src='{}'></video>".format(row["path"]))
        parts.append("<ul>")
        for member in row["members"]:
            parts.append("<li>{:.2f}s &mdash; {}</li>".format(
                member["seconds"], html.escape(str(member["caption"]))))
        parts.append("</ul></div>")
    parts.append("</div>")
    (output_dir / "index.html").write_text("\n".join(parts) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=pathlib.Path, required=True)
    parser.add_argument("--bundle", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--captions", type=pathlib.Path, default=None)
    parser.add_argument("--subprototypes", type=int, default=12)
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=4.0,
                        help="clip length; shorter segments loop to fill it")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--view-azimuth", type=float, default=30.0)
    parser.add_argument("--select", default="spread", choices=("spread", "largest"))
    parser.add_argument("--min-uploads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--title-note", default="",
                        help="text prepended to each clip's title bar, e.g. which arm "
                             "of a paired sheet this is. Burned into the MP4.")
    parser.add_argument("--hide-uploads", action="store_true",
                        help="omit the upload count from the rendered clip. For "
                             "paired blind sheets: the control has systematically "
                             "more uploads, so a printed count gives the answer away.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build(labels_dir=args.labels, bundle=args.bundle, output_dir=args.output,
                       captions_path=args.captions, subprototypes=args.subprototypes,
                       members=args.members, seconds=args.seconds, fps=args.fps,
                       azimuth=args.view_azimuth, selection=args.select,
                       min_uploads=args.min_uploads, seed=args.seed,
                       hide_uploads=args.hide_uploads, title_note=args.title_note)
    except (VideoError, CardError) as error:
        print("videos refused: {}".format(error), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    print("open {}".format(args.output / "index.html"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
