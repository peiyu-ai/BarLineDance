"""Draw COCO-18 keypoints as the OpenPose-style video the animator conditions on.

COLOURS AND WIDTHS ARE THE OPENPOSE CONVENTION, not a choice: the pose encoders
in these pipelines were trained on OpenPose renderings, so a skeleton drawn with
different limb colours is a different conditioning signal to them.  The 18-limb
list and its palette are the ones in the original openpose `body_25`/`coco`
renderer, restricted to COCO-18.
"""
import argparse
import json
import pathlib
import subprocess

import numpy as np

LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9),
         (9, 10), (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16),
         (0, 15), (15, 17)]
LIMB_COLOURS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
                (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
                (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
                (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
                (255, 0, 170)]
JOINT_COLOURS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
                 (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
                 (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
                 (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
                 (255, 0, 170), (255, 0, 85)]


def draw(keypoints, scores, width, height, threshold=0.3):
    from PIL import Image, ImageDraw

    frames = []
    for uv, sc in zip(keypoints, scores):
        image = Image.new("RGB", (width, height), (0, 0, 0))
        pen = ImageDraw.Draw(image)
        pixels = np.stack([uv[:, 0] * width, uv[:, 1] * height], axis=1)
        visible = (sc > threshold) & np.isfinite(pixels).all(axis=1)
        for (a, b), colour in zip(LIMBS, LIMB_COLOURS):
            if visible[a] and visible[b]:
                pen.line([tuple(pixels[a]), tuple(pixels[b])],
                         fill=colour, width=max(2, width // 90))
        radius = max(2, width // 110)
        for index, colour in enumerate(JOINT_COLOURS):
            if visible[index]:
                x, y = pixels[index]
                pen.ellipse([x - radius, y - radius, x + radius, y + radius], fill=colour)
        frames.append(image)
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose-dir", required=True, help="from project_pose_2d.py")
    ap.add_argument("--out", required=True, help="output .mp4")
    ap.add_argument("--audio", default=None, help="mux this wav in")
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    source = pathlib.Path(args.pose_dir)
    keypoints = np.load(source / "keypoints.npy")
    scores = np.load(source / "scores.npy")
    meta = json.loads((source / "meta.json").read_text())
    width, height = int(meta["video_w"]), int(meta["video_h"])

    frames = draw(keypoints, scores, width, height)
    staging = pathlib.Path(args.out).with_suffix("")
    staging.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        frame.save(staging / "{:05d}.png".format(index))

    # -framerate BEFORE -i, or ffmpeg reads the stills at its default 25 and the
    # pose runs slow against the music.  The repository has paid for this once:
    # a missing -framerate on a looped input ran at the LCM of the two rates.
    command = ["ffmpeg", "-y", "-loglevel", "error",
               "-framerate", str(args.fps), "-i", str(staging / "%05d.png")]
    if args.audio and pathlib.Path(args.audio).is_file():
        command += ["-i", args.audio, "-c:a", "aac", "-shortest"]
    command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", args.out]
    subprocess.run(command, check=True)
    for png in staging.glob("*.png"):
        png.unlink()
    staging.rmdir()
    print("{} frames at {} fps -> {}".format(len(frames), args.fps, args.out))


if __name__ == "__main__":
    main()
