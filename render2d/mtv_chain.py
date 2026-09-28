"""A long clip as a CHAIN of 49-frame renders, each starting where the last ended.

WHY.  MTV-Crafter trains on 49 frames, and the motion attention's key layout
depends on the token count, so a 49-frame window is the one length the model has
seen.  Two ways to cover a 20-second clip with it:

  * context windows (``MTV_CONTEXT``): one denoise over the whole clip, sliding a
    49-frame window with overlap.  Every frame is the average of two windows,
    and measured on 818 that halves the motion -- the render moved 0.027 against
    the pose's 0.053, where a single 49-frame window matched it.
  * this chain: render 49 frames, take the last frame as the next render's
    reference image, render the next 49.  No averaging, every chunk gets the
    full-strength window, and the chunks are stitched end to end.

MEASURED ON 818, 2026-09-17, AND REJECTED FOR SHIPPING.  Against the context
windows at the same settings the chain does follow the dance better -- per-frame
limb shape 0.070 against 0.080, and it moves 0.077 where the pose moves 0.059,
against the windows' 0.041.  But the character does not survive it: by the third
chunk the top has become a different garment with red trim, and by the last the
face is a different person (``scratchpad/chain_bounds.png``; the whole-clip
``follow`` reads 0.149 against the windows' 0.086 because the body is no longer
in the same place either).  The detector's confidence does NOT see this -- it
reads 0.90/0.95 on both -- so the check that decides it is the contact sheet.
Kept because the measurement is worth repeating if the base is ever swapped for
one with proper long-video conditioning; not the shipping path.

The joints are sliced by MODEL FRAME (16 fps), the same units ``comfy_mtv.py``
takes for ``--start`` / ``--max-frames``.
"""
import argparse
import json
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from render2d.comfy_animate import COMFY_ROOT, probe  # noqa: E402

CHUNK = 49
MODEL_FPS = 16


def last_frame(video, target):
    """Write the video's last frame into ComfyUI's input directory."""
    frames = probe(video)["frames"]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video),
                    "-vf", "select='eq(n\\,{})'".format(frames - 1), "-vsync", "0",
                    "-frames:v", "1", str(target)], check=True)
    return target


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--character", required=True, help="a file under ComfyUI input/")
    ap.add_argument("--joints", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--audio", default=None)
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--keep-chunks", action="store_true")
    args = ap.parse_args()

    import numpy as np
    total = len(np.load(args.joints))
    out = pathlib.Path(args.out)
    work = out.parent / (out.stem + "_chunks")
    work.mkdir(parents=True, exist_ok=True)
    stem = out.stem

    character = args.character
    pieces, starts = [], []
    start = 0
    index = 0
    while start < total - 1:
        frames = min(args.chunk, total - start)
        if frames < 5:
            break
        piece = work / "{:02d}.mp4".format(index)
        if not piece.is_file():
            command = [sys.executable, "render2d/comfy_mtv.py", "--character", character,
                       "--joints", args.joints, "--out", str(piece),
                       "--start", str(start), "--max-frames", str(frames)]
            print(" ".join(command), flush=True)
            subprocess.run(command, check=True)
        pieces.append(piece)
        starts.append(start)
        # The next chunk starts ON this chunk's last frame, and that frame is its
        # reference image, so the joint slice must begin there too.
        produced = probe(piece)["frames"]
        start += produced - 1
        index += 1
        character = str(last_frame(piece, COMFY_ROOT / "input" / "{}_chain{:02d}.png".format(stem, index)))

    if not pieces:
        raise SystemExit("nothing to render")
    # Every chunk after the first repeats its reference frame, so drop frame 0 there.
    listing = work / "concat.txt"
    trimmed = []
    for position, piece in enumerate(pieces):
        if position == 0:
            trimmed.append(piece)
            continue
        cut = piece.with_name(piece.stem + "_cut.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(piece),
                        "-vf", "select='gt(n\\,0)'", "-vsync", "0", "-an", str(cut)], check=True)
        trimmed.append(cut)
    listing.write_text("".join("file '{}'\n".format(p.resolve()) for p in trimmed))
    joined = work / "joined.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
                    "-i", str(listing), "-c", "copy", str(joined)], check=True)
    if args.audio and pathlib.Path(args.audio).is_file():
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(joined), "-i", args.audio,
                        "-c:v", "copy", "-c:a", "libmp3lame", "-b:a", "128k", "-shortest",
                        str(out)], check=True)
    else:
        out.write_bytes(joined.read_bytes())
    info = probe(out)
    record = {"video": str(out), "joints": args.joints, "chunks": len(pieces),
              "chunk_frames": args.chunk, "chunk_starts": starts,
              "frames": info["frames"], "fps": info["fps"],
              "settings": json.loads(pieces[0].with_suffix(".json").read_text())}
    out.with_suffix(".json").write_text(json.dumps(record, indent=1))
    print("{} chunks -> {} ({} frames at {:g} fps)".format(
        len(pieces), out, info["frames"], info["fps"]))
    if not args.keep_chunks:
        for piece in trimmed:
            if piece.name.endswith("_cut.mp4"):
                piece.unlink()


if __name__ == "__main__":
    main()
