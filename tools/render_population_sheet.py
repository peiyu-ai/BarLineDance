#!/usr/bin/env python3
"""What the corpus actually looks like, sorted by how many people are in frame.

``tools/audit_clip_population.py`` says 30% of released clips are solo and half
have three or more people in most frames.  Those are numbers about a corpus
nobody has looked at as a whole.  This draws the same partition: a band of clips
per class, each with the measurements that put it there, so the classification
can be checked against the footage rather than trusted.

The last band is the one worth building the page for.  It is not a class of
population at all -- it is the clips whose *crop* travels furthest per frame,
which is where the footage that is not a dance take collects: transitions,
graphics, montages, a camera swinging across a room.  Nothing in this repo
measures "is this usable dance footage", and the population signals cannot see
it: the fireworks clip on the fps contact sheet reads solo, rival_count 0, a
person detected in 88% of frames.  Crop travel is the nearest thing to a handle
on it that already exists, and this band exists to show how well that handle
works -- including where it does not.

Usage::

    render_population_sheet.py --per-class 4 --output output/population_sheet.png
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from typing import Dict, List, Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

INGEST = pathlib.Path("/cache/atomicdance-assets/data/wild_ingest_v1")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def classify(row) -> Optional[str]:
    if row.get("subject_area_median") is None:
        return None
    if row["solo"]:
        return "solo"
    if row["share_three_or_more"] >= 0.5:
        return "three or more"
    if row["share_two_or_more"] >= 0.5:
        return "two"
    return "mixed"


def main(argv: Optional[List[str]] = None) -> int:
    from PIL import Image, ImageDraw, ImageFont

    from tools.render_recut_contact_sheet import frames_at, probe_duration

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--population", type=pathlib.Path,
                        default=pathlib.Path("runs/wild_clip_population.json"))
    parser.add_argument("--per-class", type=int, default=4)
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--thumb-height", type=int, default=124)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    data = json.loads(args.population.read_text(encoding="utf-8"))
    rows = [r for r in data["rows"] if r.get("released")]
    buckets: Dict[str, List[dict]] = {}
    for row in rows:
        name = classify(row)
        if name and (INGEST / row["clip"] / "clip.mp4").is_file():
            buckets.setdefault(name, []).append(row)
    # The travel band is drawn from every class, because it is a different axis.
    travelling = sorted((r for r in rows
                         if r.get("crop_travel_median") is not None
                         and (INGEST / r["clip"] / "clip.mp4").is_file()),
                        key=lambda r: -r["crop_travel_median"])
    order = ["solo", "two", "three or more", "mixed"]
    share = {k: len([r for r in rows if classify(r) == k]) / max(len(rows), 1)
             for k in order}

    random.seed(args.seed)
    plan = []
    for name in order:
        group = buckets.get(name, [])
        random.shuffle(group)
        for row in group[: args.per_class]:
            plan.append((name, row))
    for row in travelling[: args.per_class]:
        plan.append(("widest crop travel", row))
    print("{} clips to draw, over {} bands".format(len(plan), len(order) + 1), flush=True)

    drawn, skipped = [], []
    for index, (name, row) in enumerate(plan, start=1):
        video = INGEST / row["clip"] / "clip.mp4"
        duration = probe_duration(video)
        if not duration:
            skipped.append(row["clip"])
            continue
        marks = [duration * (i + 0.5) / args.columns for i in range(args.columns)]
        strip = frames_at(video, marks, args.thumb_height)
        if not strip:
            skipped.append(row["clip"])
            print("[{}/{}] {} SKIPPED".format(index, len(plan), row["clip"]), flush=True)
            continue
        drawn.append({"band": name, "row": row, "strip": strip})
        print("[{}/{}] {} {}".format(index, len(plan), name, row["clip"]), flush=True)
    if not drawn:
        raise SystemExit("nothing decoded")

    pad, label_w, gap = 18, 156, 6
    strip_w = max(sum(f.width for f in d["strip"]) + gap * (args.columns - 1) for d in drawn)
    band_header = 34
    panel_h = args.thumb_height + 34
    header_h = 92
    width = pad * 2 + label_w + strip_w
    bands = []
    for d in drawn:
        if not bands or bands[-1][0] != d["band"]:
            bands.append((d["band"], []))
        bands[-1][1].append(d)
    height = header_h + sum(band_header + len(items) * panel_h + pad
                            for _, items in bands) + pad

    canvas = Image.new("RGB", (width, height), (17, 18, 21))
    draw = ImageDraw.Draw(canvas)
    title = ImageFont.truetype(FONT_BOLD, 24)
    head = ImageFont.truetype(FONT, 14)
    band_font = ImageFont.truetype(FONT_BOLD, 17)
    small = ImageFont.truetype(FONT, 12)

    draw.text((pad, pad), "wild corpus by how many people are in frame",
              font=title, fill=(240, 240, 245))
    draw.text((pad, pad + 33),
              "Counts are yolox_l person detections per frame -- people, not dancers: an "
              "audience, a queue behind the shot and a studio's mirror are all people to it.",
              font=head, fill=(150, 154, 164))
    draw.text((pad, pad + 53),
              "Solo means at most one person in at least 90% of frames.  Shares are of the "
              "13,783 released clips.  Subject area is the tracked box over the frame.",
              font=head, fill=(150, 154, 164))

    colours = {"solo": (110, 176, 132), "two": (196, 168, 96),
               "three or more": (196, 108, 96), "mixed": (140, 140, 158),
               "widest crop travel": (150, 128, 200)}
    y = header_h
    for band, items in bands:
        pct = share.get(band)
        caption = band if pct is None else "{}  --  {:.1%} of released clips".format(band, pct)
        if band == "widest crop travel":
            caption = "widest crop travel  --  a different axis: where footage that is not a dance take collects"
        draw.text((pad, y + 6), caption, font=band_font, fill=colours.get(band, (220, 220, 230)))
        y += band_header
        for item in items:
            row, strip = item["row"], item["strip"]
            x = pad + label_w
            for frame in strip:
                canvas.paste(frame, (x, y))
                x += frame.width + gap
            draw.rectangle([pad + label_w - 3, y - 3, x - gap + 2, y + args.thumb_height + 2],
                           outline=colours.get(band, (90, 90, 100)), width=2)
            lines = [
                row["clip"][:26],
                "people med {:.0f} max {}".format(row["people_median"], row["people_max"]),
                "<=1 person {:.0%}".format(row["share_at_most_one"]),
                "area {:.3f}".format(row.get("subject_area_median") or 0),
                "travel {:.3f}".format(row.get("crop_travel_median") or 0),
            ]
            for offset, line in enumerate(lines):
                draw.text((pad, y + 2 + offset * 15), line, font=small,
                          fill=(226, 228, 234) if offset == 0 else (150, 154, 164))
            y += panel_h
        y += pad

    if skipped:
        draw.text((pad, height - 20), "{} skipped: {}".format(
            len(skipped), ", ".join(skipped[:4])), font=small, fill=(196, 108, 96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print("\nwrote {}  ({} clips, {}x{})".format(args.output, len(drawn), width, height))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
