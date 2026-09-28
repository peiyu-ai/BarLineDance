#!/usr/bin/env python3
"""Look at the discovered atomic movements instead of only measuring them.

Statistics can be satisfied by a vocabulary that is nonsense: split each
prototype seven ways at random and the histograms still match the paper.  The
check that cannot be gamed is watching several members of one sub-prototype
side by side and asking whether they are the same move.

Each sheet is one sub-prototype.  Rows are member segments, drawn from as many
distinct source clips as possible; columns are frames sampled across the
segment, taken from the source video through ``frame_ids.npy`` -- the same
mapping the captioner uses, so what is on screen is the footage the VLM saw.
The header carries the sub-prototype's semantic tag, its size, and **how many
distinct uploads it draws from**, which is the number that separates a genuine
prototype from one clip's mannerism memorised seven times.

An HTML index groups the sheets by prototype, so the question "did in-group
re-clustering split this prototype into things that differ?" is answered by
scrolling one page.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.audit_atomic_vocabulary import decode_ids  # noqa: E402
from tools.caption_segments_vlm import (  # noqa: E402
    load_frames,
    person_boxes_for,
    probe_frame_size,
    sample_frame_positions,
    segment_box,
    video_path_for,
)
from tools.cluster_atomics_tmr import build_row_index, resolve_row  # noqa: E402
from tools.recluster_atomics_ingroup import segments_of  # noqa: E402

FPS = 30.0


class GalleryError(RuntimeError):
    pass


def collect_members(labels_dir: pathlib.Path, bundle: pathlib.Path) -> Dict[int, List[Dict]]:
    """Every accepted segment, grouped by the label it carries."""
    rows = {json.loads(line)["recording_id"]: json.loads(line)
            for line in (bundle / "sequences.jsonl").open(encoding="utf-8")}
    index = build_row_index(rows)
    members: Dict[int, List[Dict]] = collections.defaultdict(list)
    for line in (labels_dir / "labels.jsonl").open(encoding="utf-8"):
        entry = json.loads(line)
        row = resolve_row(index, entry["recording_id"]) or rows.get(entry["recording_id"])
        if row is None:
            continue
        labels = np.load(labels_dir / entry["labels_path"])
        for start, end, label in segments_of(labels):
            if label <= 0 or end - start < 4:
                continue
            members[int(label)].append({
                "recording_id": entry["recording_id"],
                "start": start, "end": end,
                "frame_ids_path": row["frame_ids_path"],
                "upload": str(entry["recording_id"]).rsplit(":", 1)[0],
            })
    return members


def spread_over_uploads(entries: Sequence[Dict], count: int) -> List[Dict]:
    """Pick members from as many distinct uploads as possible.

    Taking the first N would routinely show N segments of one clip, which
    proves nothing: consecutive segments of one dancer resemble each other for
    reasons that have nothing to do with the vocabulary.
    """
    by_upload: Dict[str, List[Dict]] = collections.defaultdict(list)
    for entry in entries:
        by_upload[entry["upload"]].append(entry)
    order = sorted(by_upload, key=lambda key: (-len(by_upload[key]), key))
    picked: List[Dict] = []
    round_index = 0
    while len(picked) < count and any(len(by_upload[key]) > round_index for key in order):
        for key in order:
            if len(by_upload[key]) > round_index:
                picked.append(by_upload[key][round_index])
                if len(picked) == count:
                    return picked
        round_index += 1
    return picked


def sheet_for(entries: Sequence[Dict], bundle: pathlib.Path, video_dir: pathlib.Path, *,
              frames: int, thumb: int, header: str, subheader: str,
              person_boxes: Optional[pathlib.Path] = None, crop_margin: float = 0.25):
    """One contact sheet: a row per member segment, a column per sampled frame.

    Cropping matters as much here as it does for the captioner: a sheet of
    dancers forty pixels tall cannot be judged by eye either, and showing the
    reviewer a different framing from the one the VLM was given would make the
    gallery answer a question nobody asked.
    """
    from PIL import Image, ImageDraw

    rows = []
    for entry in entries:
        frame_ids = np.load(bundle / entry["frame_ids_path"])
        video = video_path_for(entry["recording_id"], video_dir)
        if video is None:
            continue
        positions = sample_frame_positions(entry["start"], entry["end"], frames)
        source = [int(frame_ids[p]) for p in positions if p < len(frame_ids)]
        box = None
        if person_boxes is not None:
            boxes = person_boxes_for(entry["recording_id"], person_boxes)
            if boxes is not None and len(frame_ids) and int(frame_ids.max()) < len(boxes):
                width, height = probe_frame_size(video)
                box = segment_box(boxes, source, margin=crop_margin,
                                  width=width, height=height)
        images = load_frames(video, source, thumb * 2, box)
        if images:
            rows.append((entry, images[:frames]))
    if not rows:
        return None

    pad, head = 4, 40
    width = pad + frames * (thumb + pad)
    height = head + len(rows) * (thumb + pad)
    sheet = Image.new("RGB", (width, height), (18, 18, 20))
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 6), header, fill=(245, 245, 245))
    draw.text((pad, 22), subheader, fill=(150, 150, 160))
    for row_index, (entry, images) in enumerate(rows):
        top = head + row_index * (thumb + pad)
        for column, image in enumerate(images):
            scaled = image.copy()
            scaled.thumbnail((thumb, thumb))
            sheet.paste(scaled, (pad + column * (thumb + pad), top))
        # Clamped: a narrow sheet (few frames, small thumbnails) would otherwise
        # place the duration off the left edge, where it is simply lost.
        draw.text((max(pad, width - 150), top + 2),
                  "{:.2f}s".format((entry["end"] - entry["start"]) / FPS),
                  fill=(140, 140, 150))
    return sheet


def build(*, labels_dir: pathlib.Path, bundle: pathlib.Path, video_dir: pathlib.Path,
          output_dir: pathlib.Path, prototypes: int, subprototypes: int,
          members: int, frames: int, thumb: int, seed: int,
          person_boxes: Optional[pathlib.Path] = None,
          crop_margin: float = 0.25) -> Dict[str, object]:
    grouped = collect_members(labels_dir, bundle)
    if not grouped:
        raise GalleryError("no labelled segments under {}".format(labels_dir))
    mapping = decode_ids(labels_dir / "producer.npz")
    tags_path = labels_dir / "subprototype_tags.json"
    tags = json.loads(tags_path.read_text()) if tags_path.is_file() else {}

    undecodable = sorted(set(grouped) - set(mapping))
    if undecodable:
        raise GalleryError(
            "labels carry ids {} that producer.npz does not decode; the bundle and "
            "its producer artefact are out of step".format(undecodable[:5]))
    by_prototype: Dict[int, List[int]] = collections.defaultdict(list)
    for label in grouped:
        by_prototype[mapping[label][0]].append(label)

    rng = np.random.default_rng(seed)
    # Sample prototypes rather than taking the largest: the biggest groups are
    # the easiest to make look coherent, and a gallery that only shows those is
    # a demo, not a check.
    chosen = sorted(rng.choice(sorted(by_prototype), size=min(prototypes, len(by_prototype)),
                               replace=False).tolist())

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sheets").mkdir(exist_ok=True)
    manifest: List[Dict[str, object]] = []
    for prototype in chosen:
        labels = sorted(by_prototype[prototype], key=lambda label: -len(grouped[label]))
        for label in labels[:subprototypes]:
            entries = grouped[label]
            uploads = len({entry["upload"] for entry in entries})
            tag = tags.get(str(label), "")
            picked = spread_over_uploads(entries, members)
            sheet = sheet_for(
                picked, bundle, video_dir, frames=frames, thumb=thumb,
                header="prototype {} / sub-prototype {}  ({} segments, {} uploads)".format(
                    prototype, mapping[label][1], len(entries), uploads),
                subheader=(tag or "(no semantic tag in this bundle)")[:160],
                person_boxes=person_boxes, crop_margin=crop_margin)
            if sheet is None:
                continue
            name = "p{:03d}_s{:02d}_label{:04d}.jpg".format(prototype, mapping[label][1], label)
            sheet.save(output_dir / "sheets" / name, quality=88)
            manifest.append({"prototype": prototype, "subprototype": mapping[label][1],
                             "label": label, "segments": len(entries), "uploads": uploads,
                             "upload_share_of_largest": round(max(
                                 collections.Counter(
                                     entry["upload"] for entry in entries).values())
                                 / len(entries), 3),
                             "tag": tag, "sheet": "sheets/" + name})
            print("  {} -> {} segments from {} uploads".format(name, len(entries), uploads),
                  flush=True)

    (output_dir / "gallery.json").write_text(
        json.dumps({"labels": str(labels_dir.resolve()), "sheets": manifest},
                   indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_index(output_dir, manifest, labels_dir)
    concentrated = [row for row in manifest if row["upload_share_of_largest"] > 0.5]
    return {
        "sheets": len(manifest),
        "prototypes_shown": len(chosen),
        "median_uploads_per_subprototype": float(np.median(
            [row["uploads"] for row in manifest])) if manifest else 0.0,
        "subprototypes_dominated_by_one_upload": len(concentrated),
        "index": str((output_dir / "index.html").resolve()),
    }


def write_index(output_dir: pathlib.Path, manifest: Sequence[Dict[str, object]],
                labels_dir: pathlib.Path) -> None:
    parts = ["<!doctype html><meta charset='utf-8'>",
             "<title>atomic vocabulary gallery</title>",
             "<style>body{background:#111;color:#eee;font:14px/1.5 system-ui;margin:24px}",
             "h2{margin:32px 0 8px;font-size:16px;color:#9cf}",
             "figure{margin:0 0 18px}img{max-width:100%;border-radius:6px}",
             "figcaption{color:#aaa;padding:4px 0}.warn{color:#f88}</style>",
             "<h1>Atomic vocabulary: {}</h1>".format(labels_dir.name),
             "<p>One sheet per sub-prototype. Rows are member segments from different "
             "uploads; columns are frames across the segment. If the rows do not look "
             "like the same move, in-group re-clustering did not work here.</p>"]
    for prototype in sorted({row["prototype"] for row in manifest}):
        parts.append("<h2>prototype {}</h2>".format(prototype))
        for row in [entry for entry in manifest if entry["prototype"] == prototype]:
            warn = (" <span class='warn'>&mdash; {:.0%} from one upload</span>".format(
                row["upload_share_of_largest"]) if row["upload_share_of_largest"] > 0.5 else "")
            parts.append(
                "<figure><img loading='lazy' src='{}'>"
                "<figcaption>sub-prototype {} &middot; {} segments &middot; {} uploads{}"
                "<br><b>{}</b></figcaption></figure>".format(
                    row["sheet"], row["subprototype"], row["segments"], row["uploads"],
                    warn, row["tag"] or "(no tag)"))
    (output_dir / "index.html").write_text("\n".join(parts) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=pathlib.Path, required=True)
    parser.add_argument("--bundle", type=pathlib.Path, required=True)
    parser.add_argument("--video-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--prototypes", type=int, default=8,
                        help="how many prototypes to sample")
    parser.add_argument("--subprototypes", type=int, default=4,
                        help="sub-prototypes per prototype")
    parser.add_argument("--members", type=int, default=6, help="member segments per sheet")
    parser.add_argument("--frames", type=int, default=6, help="frames per member")
    parser.add_argument("--thumb", type=int, default=140)
    parser.add_argument("--person-boxes", type=pathlib.Path, default=None,
                        help="GVHMR raw output root; crop each row to the tracked "
                             "dancer, matching what the captioner saw")
    parser.add_argument("--crop-margin", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260810)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = build(labels_dir=args.labels, bundle=args.bundle, video_dir=args.video_dir,
                        output_dir=args.output_dir, prototypes=args.prototypes,
                        subprototypes=args.subprototypes, members=args.members,
                        frames=args.frames, thumb=args.thumb, seed=args.seed)
    except (GalleryError, FileNotFoundError) as error:
        raise SystemExit("error: {}".format(error))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
