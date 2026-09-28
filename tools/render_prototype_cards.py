#!/usr/bin/env python3
"""Show what a sub-prototype *is*, next to the words attached to it.

The paper's Fig. 3 puts a handful of rendered dancers beside a short phrase --
"Prototype: Stretch both arms" -- and that single picture carries a claim no
statistic in this repo checks: that members of one category are the same move,
and that the phrase names it.  ``tools/build_atomic_gallery.py`` shows source
video frames, which answers "what footage went in"; this tool renders the
*motion* the pipeline actually clustered, which answers "what did it group".

The two differ more than it sounds.  What the discovery stage sees is a
canonical pose -- root-centred, hips turned onto +x, divided by shoulder width
-- because otherwise the vocabulary would sort dancers by camera angle and
body size.  Drawing raw joint positions here would show a viewer something the
algorithm never had; drawing the canonical pose shows exactly what it did.

Frames are taken at motion beats, the local minima of joint speed, which is the
paper's own definition of a keyframe and what its "signature pose" means.  A
segment therefore appears as its settle points rather than as evenly spaced
samples, which is what makes two members of one sub-prototype comparable at a
glance even when they are danced at different speeds.

Rows are drawn from as many distinct uploads as exist, for the same reason the
gallery does it: several consecutive segments of one clip resemble each other
for reasons that have nothing to do with the vocabulary.

Each row carries its own VLM caption, so the correspondence the tool exists to
show -- text against motion -- is on the same line and can be judged rather than
assumed.

Usage:
    python3 tools/render_prototype_cards.py \\
        --labels data/wild3d/wild_v2_ingroup_30b_v2 \\
        --bundle data/wild3d/wild_performance_v1 \\
        --captions runs/wild_v2_captions_30b/captions.jsonl \\
        --output runs/wild_v2_cards_llm
"""

from __future__ import annotations

import argparse
import base64
import collections
import html
import io
import json
import pathlib
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.build_atomic_gallery import collect_members, spread_over_uploads  # noqa: E402
from tools.motion_beats import canonical_pose, find_motion_beats  # noqa: E402

FPS = 30.0
# The 22 body joints; the SMPL hand joints are identity here and only add two
# stubs that read as noise at thumbnail size.
BODY_JOINTS = 22


class CardError(RuntimeError):
    pass


def load_caption_index(path: Optional[pathlib.Path]) -> Dict[Tuple[str, int, int], Dict]:
    """Captions keyed by the exact segment they describe.

    Keyed by (recording, start, end) rather than by recording alone: a clip
    contributes many segments and attaching the wrong one would make the whole
    point of this page -- reading text against motion -- quietly false.
    """
    if path is None:
        return {}
    index: Dict[Tuple[str, int, int], Dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (row["recording_id"], int(row["start"]), int(row["end"]))
            index.setdefault(key, row)
    return index


def caption_text(row: Optional[Dict]) -> str:
    """The closed-vocabulary fields as one line, falling back to the summary.

    The structured fields are what the grouping consumed, so they are what a
    reviewer needs to see; the free summary is shown only when the fields are
    missing, and never blended with them, so it is always clear which of the two
    a judgement is being made on.
    """
    if not row:
        return "(no caption)"
    fields = row.get("fields") or row.get("caption_fields")
    if isinstance(fields, dict) and fields:
        return ", ".join("{}: {}".format(key, value) for key, value in sorted(fields.items()))
    caption = row.get("caption")
    if isinstance(caption, str) and caption.strip():
        return caption.strip()
    summary = row.get("summary")
    return str(summary).strip() if summary else "(no caption)"


def keyframe_poses(joints: np.ndarray, count: int) -> List[np.ndarray]:
    """Canonical poses at the segment's motion beats, padded by repeating the last.

    Padding repeats the final real beat rather than inserting a zero pose: a
    zero pose is a pose -- a skeleton collapsed at the origin -- and it would
    read as though the dancer had done something.
    """
    beats = find_motion_beats(joints, max_beats=count)
    if not beats:
        beats = [0]
    while len(beats) < count:
        beats.append(beats[-1])
    return [canonical_pose(joints[beat])[:BODY_JOINTS] for beat in beats[:count]]


def project(pose: np.ndarray, azimuth: float) -> np.ndarray:
    """[J,3] canonical pose -> [J,2] screen coordinates, turned off the front axis.

    A straight x-z view throws the depth axis away entirely, and that is not a
    harmless simplification: a dancer folding forward moves almost purely along
    y, so the figure loses two thirds of its height on screen and reads as a
    collapsed skeleton -- a data failure that did not happen.  Measured on one
    such segment, the body's y-span doubles (0.81 -> 1.73) exactly as its z-span
    halves (5.09 -> 2.48), which is a fold, not a break.

    Turning the camera off-axis puts depth back on screen at a cost in
    left-right foreshortening, and 30 degrees keeps sideways poses legible while
    making folds unmistakable.  The clustering itself never had this problem --
    ``descriptor_vector`` uses all three axes -- so this is a property of the
    picture only.
    """
    radians = np.deg2rad(azimuth)
    horizontal = pose[:, 0] * np.cos(radians) - pose[:, 1] * np.sin(radians)
    return np.stack([horizontal, pose[:, 2]], axis=1)


def draw_pose(axis, pose: np.ndarray, colour: str, azimuth: float = 30.0) -> None:
    """One canonical pose as a stick figure, seen from slightly off the front."""
    from vis import smpl_parents

    screen = project(pose, azimuth)
    for joint, parent in enumerate(smpl_parents[:BODY_JOINTS]):
        if parent < 0 or parent >= BODY_JOINTS or joint >= len(screen):
            continue
        axis.plot([screen[joint, 0], screen[parent, 0]],
                  [screen[joint, 1], screen[parent, 1]],
                  color=colour, linewidth=1.6, solid_capstyle="round", zorder=2)
    axis.scatter(screen[:, 0], screen[:, 1], s=3.0, color=colour, zorder=3)
    axis.set_xlim(-2.4, 2.4)
    axis.set_ylim(-2.8, 2.4)
    axis.set_aspect("equal")
    axis.axis("off")


def select_subprototypes(grouped: Dict[int, List[Dict]], *, count: int, members: int,
                         strategy: str, min_uploads: int
                         ) -> Tuple[List[Tuple[int, List[Dict], int]], Dict[int, List[Dict]]]:
    """Choose which sub-prototypes the page shows, and say how.

    ``largest`` reads well and is the wrong default: sizes here run from single
    digits to several hundred, so the biggest categories are systematically the
    vaguest ones, and a page built from them would answer a question about the
    tail rather than about the vocabulary.  ``spread`` walks evenly down the
    size-ordered list instead, so the page contains the large, the median and
    the small in the proportion they occur.
    """
    eligible = [(label, entries, len({entry["upload"] for entry in entries}))
                for label, entries in grouped.items()]
    eligible = [row for row in eligible if row[2] >= min_uploads]
    eligible.sort(key=lambda row: (-len(row[1]), row[0]))
    if not eligible:
        return [], {}
    if strategy == "largest":
        return eligible[:count], {}
    if strategy != "spread":
        raise CardError("unknown selection strategy {}".format(strategy))
    if count >= len(eligible):
        return eligible, {}
    steps = np.linspace(0, len(eligible) - 1, count)
    picked, seen = [], set()
    for position in steps:
        index = int(round(float(position)))
        while index in seen and index + 1 < len(eligible):
            index += 1
        if index in seen:
            continue
        seen.add(index)
        picked.append(eligible[index])
    return picked, {}


def render_card(entries: Sequence[Dict], joints_by_recording: Dict[str, np.ndarray], *,
                label: int, tag: str, size: int, uploads: int, frames: int,
                captions: Dict[Tuple[str, int, int], Dict], azimuth: float = 30.0,
                hide_uploads: bool = False, title_note: str = ""):
    """A figure for one sub-prototype: a row per member, a column per beat."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for entry in entries:
        joints = joints_by_recording.get(entry["recording_id"])
        if joints is None:
            continue
        segment = joints[entry["start"]:entry["end"]]
        if len(segment) < 4:
            continue
        rows.append((entry, keyframe_poses(segment, frames)))
    if not rows:
        return None

    figure, axes = plt.subplots(len(rows), frames,
                                figsize=(frames * 1.25 + 4.6, len(rows) * 1.55),
                                squeeze=False)
    figure.patch.set_facecolor("#16161a")
    palette = ["#7fd1ff", "#ffd479", "#9ae6a0", "#ff9ab0", "#c6a8ff", "#8ce0d4"]
    for row_index, (entry, poses) in enumerate(rows):
        colour = palette[row_index % len(palette)]
        for column, pose in enumerate(poses):
            axis = axes[row_index][column]
            axis.set_facecolor("#16161a")
            draw_pose(axis, pose, colour, azimuth)
        text = caption_text(captions.get(
            (entry["recording_id"], entry["start"], entry["end"])))
        duration = (entry["end"] - entry["start"]) / FPS
        axes[row_index][frames - 1].text(
            2.6, 0.4, "{:.2f}s  {}\n{}".format(duration, entry["upload"], text),
            transform=axes[row_index][frames - 1].transData,
            fontsize=6.0, color="#c9c9d4", va="center", ha="left", wrap=True)

    # ``hide_uploads`` exists for the paired review sheet, where the same
    # prototype id is drawn twice -- once real, once from shuffled labels -- and
    # which is which is meant to be hidden.  Scattering membership necessarily
    # raises upload diversity, so a printed upload count decodes the answer:
    # on 2026-08-21 the control read 41 against the real arm's 36 on the very
    # first pair a reader looked at.  Suppressed in the image, not just in the
    # page around it, because the count is baked into the PNG.
    # ``title_note`` is burned into the image, not written around it, because a
    # reviewer reads the picture and not the page: on 2026-08-21 two panels of a
    # paired sheet were read as ten members of one prototype, and the caption
    # that would have prevented it was in the surrounding HTML.
    figure.suptitle(
        "{}sub-prototype {}   |   “{}”   |   {} segments{}".format(
            title_note + "   |   " if title_note else "",
            label, tag, size, "" if hide_uploads else
            ", {} uploads".format(uploads)),
        color="#f2f2f5", fontsize=10, y=0.995)
    figure.tight_layout(rect=(0.0, 0.0, 0.66, 0.97))
    return figure


def figure_to_data_uri(figure) -> str:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=110, facecolor=figure.get_facecolor())
    buffer.seek(0)
    return "data:image/png;base64," + base64.b64encode(buffer.read()).decode("ascii")


def build(*, labels_dir: pathlib.Path, bundle: pathlib.Path, output_dir: pathlib.Path,
          captions_path: Optional[pathlib.Path], subprototypes: int, members: int,
          frames: int, seed: int, min_uploads: int = 2,
          selection: str = "spread", azimuth: float = 30.0,
          hide_uploads: bool = False, title_note: str = "") -> Dict[str, object]:
    from tools.cluster_atomics_tmr import build_row_index, resolve_row
    from tools.convert_motion_to_guofeats import motion_151_to_joints

    grouped = collect_members(labels_dir, bundle)
    if not grouped:
        raise CardError("{} yielded no labelled segments".format(labels_dir))
    tags_path = labels_dir / "subprototype_tags.json"
    tags = json.loads(tags_path.read_text(encoding="utf-8")) if tags_path.exists() else {}
    captions = load_caption_index(captions_path)

    chosen, picked_entries = select_subprototypes(
        grouped, count=subprototypes, members=members, strategy=selection,
        min_uploads=min_uploads)
    if not chosen:
        raise CardError("no sub-prototype had at least {} distinct uploads".format(min_uploads))
    for label, entries, _ in chosen:
        picked_entries[label] = spread_over_uploads(entries, members)

    needed = {entry["recording_id"] for label in picked_entries
              for entry in picked_entries[label]}
    rows = {json.loads(line)["recording_id"]: json.loads(line)
            for line in (bundle / "sequences.jsonl").open(encoding="utf-8")}
    index = build_row_index(rows)
    joints_by_recording: Dict[str, np.ndarray] = {}
    for recording in sorted(needed):
        row = resolve_row(index, recording) or rows.get(recording)
        if row is None:
            continue
        joints_by_recording[recording] = motion_151_to_joints(
            np.load(bundle / row["motion_path"]))
    print("  loaded motion for {}/{} recordings".format(
        len(joints_by_recording), len(needed)), flush=True)

    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    sheets = output_dir / "cards"
    sheets.mkdir(exist_ok=True)
    manifest: List[Dict[str, object]] = []
    for label, entries, uploads in chosen:
        tag = str(tags.get(str(label), "(no LLM tag: this bundle is the w/o-LLM row)"))
        figure = render_card(picked_entries[label], joints_by_recording, label=label,
                             tag=tag, size=len(entries), uploads=uploads, frames=frames,
                             captions=captions, azimuth=azimuth,
                             hide_uploads=hide_uploads, title_note=title_note)
        if figure is None:
            continue
        uri = figure_to_data_uri(figure)
        path = sheets / "sub_{:04d}.png".format(label)
        figure.savefig(path, dpi=110, facecolor=figure.get_facecolor())
        plt.close(figure)
        manifest.append({
            "label": int(label), "tag": tag, "size": int(len(entries)),
            "uploads": int(uploads), "path": str(path.relative_to(output_dir)),
            "data_uri": uri,
        })
        print("  rendered sub-prototype {} ({} segments)".format(label, len(entries)),
              flush=True)

    tag_counts = collections.Counter(tags.values()) if tags else collections.Counter()
    report = {
        "labels": str(labels_dir),
        "cards": len(manifest),
        "subprototypes_total": len(grouped),
        "distinct_tags": len(tag_counts) if tag_counts else None,
        "most_reused_tag": tag_counts.most_common(1)[0] if tag_counts else None,
        "selection": selection,
        "members_per_card": members,
        "keyframes_per_member": frames,
        "keyframe_rule": "motion beats (local minima of joint speed), the paper's keyframes",
        "pose_frame": "canonical: root-centred, hips on +x, shoulder-width normalised",
        "view_azimuth_degrees": azimuth,
        "view_note": ("turned off the front axis so a forward fold reads as a fold "
                      "rather than as a collapsed skeleton; clustering uses all three axes"),
        "seed": seed,
    }
    write_page(output_dir, manifest, report)
    (output_dir / "cards.json").write_text(
        json.dumps({**report, "manifest": [
            {key: value for key, value in row.items() if key != "data_uri"}
            for row in manifest]}, indent=2) + "\n", encoding="utf-8")
    return report


def write_page(output_dir: pathlib.Path, manifest: Sequence[Dict[str, object]],
               report: Dict[str, object]) -> None:
    """A self-contained page: images are inlined so it survives being moved."""
    parts = [
        "<title>Atomic sub-prototypes: motion against text</title>",
        "<style>",
        ":root{--bg:#fbfbfd;--fg:#1a1a1f;--muted:#5b5b6b;--card:#ffffff;--line:#e3e3ea}",
        "@media (prefers-color-scheme:dark){:root:not([data-theme='light'])",
        "{--bg:#111114;--fg:#f0f0f4;--muted:#9a9aab;--card:#1a1a20;--line:#2c2c36}}",
        ":root[data-theme='dark']{--bg:#111114;--fg:#f0f0f4;--muted:#9a9aab;",
        "--card:#1a1a20;--line:#2c2c36}",
        "body{background:var(--bg);color:var(--fg);margin:0;padding:2rem 1.25rem;",
        "font:15px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}",
        ".wrap{max-width:1180px;margin:0 auto}",
        "h1{font-size:1.5rem;margin:0 0 .4rem}",
        ".sub{color:var(--muted);margin:0 0 1.6rem}",
        ".card{background:var(--card);border:1px solid var(--line);border-radius:10px;",
        "padding:.9rem;margin-bottom:1.1rem;overflow-x:auto}",
        ".hd{display:flex;gap:.8rem;align-items:baseline;flex-wrap:wrap;margin-bottom:.5rem}",
        ".tag{font-weight:650}.meta{color:var(--muted);font-size:.85rem}",
        "img{max-width:100%;height:auto;display:block;border-radius:6px}",
        "</style>",
        "<div class='wrap'>",
        "<h1>Atomic sub-prototypes: the motion, and the text attached to it</h1>",
        "<p class='sub'>Each row is one member segment from a different upload; each "
        "column is a motion beat, drawn in the canonical frame the clustering "
        "actually uses. The phrase in the header is the summarizing LLM's tag; the "
        "line beside each row is that segment's own VLM caption.</p>",
    ]
    parts.append("<p class='sub'>{} cards from {} sub-prototypes".format(
        report["cards"], report["subprototypes_total"]))
    if report.get("distinct_tags"):
        parts.append(" &middot; {} distinct tags across the bundle".format(
            report["distinct_tags"]))
    parts.append("</p>")
    for row in manifest:
        parts.append("<div class='card'><div class='hd'>")
        parts.append("<span class='tag'>#{} &ldquo;{}&rdquo;</span>".format(
            row["label"], html.escape(str(row["tag"]))))
        parts.append("<span class='meta'>{} segments &middot; {} uploads</span>".format(
            row["size"], row["uploads"]))
        parts.append("</div><img alt='sub-prototype {}' src='{}'></div>".format(
            row["label"], row["data_uri"]))
    parts.append("</div>")
    (output_dir / "index.html").write_text("\n".join(parts) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=pathlib.Path, required=True)
    parser.add_argument("--bundle", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--captions", type=pathlib.Path, default=None)
    parser.add_argument("--subprototypes", type=int, default=24)
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--frames", type=int, default=4,
                        help="motion beats drawn per member; the paper's keyframes")
    parser.add_argument("--min-uploads", type=int, default=2,
                        help="skip sub-prototypes drawn from fewer uploads than this, "
                             "since one clip cannot demonstrate a category")
    parser.add_argument("--select", default="spread", choices=("spread", "largest"),
                        help="spread walks the size-ordered list evenly so the page is "
                             "representative; largest shows the head of the tail")
    parser.add_argument("--view-azimuth", type=float, default=30.0,
                        help="degrees off the straight-on view; 0 discards the depth "
                             "axis and makes forward folds look like broken data")
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--title-note", default="",
                        help="text prepended to each image's title bar, e.g. which arm "
                             "of a paired sheet this is. Burned into the PNG so it "
                             "cannot be separated from the picture.")
    parser.add_argument("--hide-uploads", action="store_true",
                        help="omit the upload count from the rendered image. For "
                             "paired blind sheets: the control has systematically "
                             "more uploads, so a printed count gives the answer away.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build(labels_dir=args.labels, bundle=args.bundle, output_dir=args.output,
                       captions_path=args.captions, subprototypes=args.subprototypes,
                       members=args.members, frames=args.frames, seed=args.seed,
                       min_uploads=args.min_uploads, selection=args.select,
                       azimuth=args.view_azimuth, hide_uploads=args.hide_uploads,
                       title_note=args.title_note)
    except CardError as error:
        print("cards refused: {}".format(error), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    print("open {}".format(args.output / "index.html"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
