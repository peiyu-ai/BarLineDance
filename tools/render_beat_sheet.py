"""The frames that land ON THE BEAT, cropped to the body, all arms side by side.

WHY A SHEET AND NOT THE VIDEO.  The video answers "is this any good"; it cannot
answer "is arm A better than arm B", because two videos cannot be watched at
once and the reader ends up comparing memory with sight.  The operator's defect
is about a single instant -- "真值在每个拍点都定在一个形状里，我们经常动到一半" --
so the instant is what has to be put side by side.

WHY CROPPED.  CLAUDE.md 1.5 rule 1: at 640x640 the body is a quarter of the
frame and "动作到不到位" cannot be judged at all.  Each panel is cropped to its
own body and rescaled to a common width.

AND THE WARNING THAT BELONGS ON EVERY SHEET THIS TOOL MAKES: a handful of beats
from one clip is not evidence.  On 2026-09-13 four beats of one clip read as
"ground truth reaches full extension, every arm keeps the elbows bent"; measured
over the 20 eval clips, arm straightness at the beat is 0.7792 for ground truth
against 0.7891 for the shipped arm -- ours is the straighter one.  The same trap
was paid for once before, with knee bend and wrist-above-shoulder.  Use the
sheet to FIND a candidate defect, then measure it on all twenty clips.
"""
import argparse
import pathlib
import subprocess

import numpy as np

PANEL_DEFAULT = 640


def beat_frames(audio_dir, clip, frames, first, count, channel=34):
    music = np.load(pathlib.Path(audio_dir) / (clip + ".npy"))
    beats = np.flatnonzero(music[:frames, channel] > 0.5)
    return [int(b) for b in beats[first:first + count]]


def extract(video, frame, out):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
                    "-vf", "select=eq(n\\,{})".format(frame), "-vframes", "1",
                    str(out)], check=True)


def body_box(panel, horizon=260, margin=40):
    """Tight box around the dancer, so the crop follows the body and not the
    panel.  Above the horizon the background is plain white, so anything
    coloured or dark there is the body."""
    pixels = np.asarray(panel.convert("RGB"), dtype=np.int16)
    top = pixels[:horizon]
    mask = ((np.abs(top[:, :, 0] - top[:, :, 1])
             + np.abs(top[:, :, 1] - top[:, :, 2]) > 8) | (top.sum(2) < 600))
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return None
    return (max(0, int(xs.min()) - margin), max(0, int(ys.min()) - 30),
            int(xs.max()) + margin, panel.height)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--clip", required=True, help="release clip id, for the audio")
    ap.add_argument("--audio-dir", default="runs/txy_t_gt_eval/audio")
    ap.add_argument("--panels", type=int, required=True)
    ap.add_argument("--titles", default="")
    ap.add_argument("--first-beat", type=int, default=8)
    ap.add_argument("--beats", type=int, default=4)
    ap.add_argument("--cell-width", type=int, default=360)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from PIL import Image, ImageDraw

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames,width", "-of", "csv=p=0",
         args.video], capture_output=True, text=True, check=True).stdout.split(",")
    width, frames = int(probe[0]), int(probe[1])
    panel = width // args.panels
    picked = beat_frames(args.audio_dir, args.clip, frames, args.first_beat, args.beats)
    if not picked:
        raise SystemExit("no beats in the requested range")

    scratch = pathlib.Path(args.out).with_suffix("")
    scratch.mkdir(parents=True, exist_ok=True)
    titles = [t for t in args.titles.split(",") if t] or \
        ["panel {}".format(i) for i in range(args.panels)]

    rows = []
    for index, frame in enumerate(picked):
        shot = scratch / "f{:04d}.png".format(frame)
        extract(args.video, frame, shot)
        image = Image.open(shot).convert("RGB")
        cells = []
        for i in range(args.panels):
            column = image.crop((i * panel, 0, (i + 1) * panel, image.height))
            box = body_box(column) or (200, 100, panel - 200, column.height)
            x0, y0, x1, _ = box
            span = max(x1 - x0, 180)
            centre = (x0 + x1) // 2
            crop = column.crop((max(0, centre - span // 2), y0,
                                min(panel, centre + span // 2), column.height))
            scale = args.cell_width / crop.width
            cells.append(crop.resize((args.cell_width, int(crop.height * scale))))
        height = max(c.height for c in cells)
        row = Image.new("RGB", (args.cell_width * args.panels, height), "white")
        for i, cell in enumerate(cells):
            row.paste(cell, (args.cell_width * i, 0))
        rows.append(row)

    sheet = Image.new("RGB", (args.cell_width * args.panels,
                              sum(r.height for r in rows) + 26), "white")
    draw = ImageDraw.Draw(sheet)
    for i, title in enumerate(titles[:args.panels]):
        draw.text((args.cell_width * i + 8, 7), title, fill="black")
    y = 26
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    sheet.save(args.out)
    print("{} beats {} -> {}".format(args.clip, picked, args.out))


if __name__ == "__main__":
    main()
