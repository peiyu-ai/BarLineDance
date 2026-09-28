"""Does the rendered cartoon actually do the pose it was given?

WHY IT HAS TO EXIST.  The 2D stage has no ground truth of its own -- the output
is a drawing, and every judgement about it so far has been somebody looking at
frames.  But it does have a CONTRACT: the animator was handed a skeleton and the
character is supposed to take that pose.  So the check is to run the same
detector the workflow uses over the RENDERED video and compare its keypoints
with the ones we drew.  Where they disagree, the render is not following.

READ THE COLUMNS SEPARATELY, because they fail for different reasons:

  body      median distance between the driven and the detected joint, as a
            fraction of the detected body's neck-to-ankle length.  This is the
            one that answers "动作和 pose 对不上".
  head      the same for the five face points.  A cartoon head is bigger than a
            real one, so a body-sized fit leaves it offset; this column says by
            how much and in which direction.
  found     fraction of frames in which the detector found a person at all.  A
            frame where it cannot is a frame where the character has stopped
            being a character -- the melted-limb artefact, measured.
  jitter    median frame-to-frame motion of the DETECTED body, against the same
            for the driven pose.  A render that flickers moves more than its
            driver; a render that lags moves less.
"""
import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "render2d"))

BODY = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
FACE = [0, 14, 15, 16, 17]


def frames_of(path, stride):
    """Decode with ffmpeg and hand back (index, PIL image) pairs."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames,width,height", "-of", "csv=p=0",
         str(path)], capture_output=True, text=True, check=True).stdout.strip().split(",")
    width, height, count = int(probe[0]), int(probe[1]), int(probe[2])
    with tempfile.TemporaryDirectory() as work:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
             "-vf", "select='not(mod(n\\,{}))'".format(stride), "-vsync", "0",
             "{}/%05d.png".format(work)], check=True)
        for index, name in enumerate(sorted(pathlib.Path(work).iterdir())):
            yield index * stride, Image.open(name).convert("RGB"), width, height, count


def detect(video, stride):
    from frame_character import character_pose
    found, keypoints = [], {}
    for index, image, width, height, count in frames_of(video, stride):
        try:
            body = character_pose(image)
        except SystemExit:
            found.append(False)
            continue
        found.append(True)
        keypoints[index] = body
    return keypoints, float(np.mean(found)) if found else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", required=True)
    ap.add_argument("--pose", required=True, help="the driving aapose video's directory")
    ap.add_argument("--stride", type=int, default=12,
                    help="frames between samples; the detector is the cost here")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    # Resolved against the repository, not the caller's directory: a background
    # shell has its own cwd and the relative path died in ffprobe with a bare
    # CalledProcessError that said nothing about why.
    repo = pathlib.Path(__file__).resolve().parents[1]
    args.render = str((repo / args.render) if not pathlib.Path(args.render).is_absolute()
                      else pathlib.Path(args.render))
    args.pose = str((repo / args.pose) if not pathlib.Path(args.pose).is_absolute()
                    else pathlib.Path(args.pose))
    if not pathlib.Path(args.render).is_file():
        raise SystemExit("no such render: {}".format(args.render))

    driven = np.load(pathlib.Path(args.pose) / "driven.npy")
    detected, found = detect(args.render, args.stride)
    if not detected:
        raise SystemExit("the detector found no person in any sampled frame")

    # The render runs at 16 fps and the pose was authored at 30.
    scale = len(driven) / max(len(driven), 1)
    rows = []
    for index, body in sorted(detected.items()):
        source = int(round(index * 30.0 / 16.0))
        if source >= len(driven):
            continue
        ours = driven[source]
        neck = body[1, :2]
        ankle = 0.5 * (body[10, :2] + body[13, :2])
        unit = float(np.linalg.norm(ankle - neck))
        if unit < 1e-3:
            continue
        rows.append({
            "frame": index,
            "body": float(np.median(np.linalg.norm(body[BODY, :2] - ours[BODY], axis=1)) / unit),
            "head": float(np.median(np.linalg.norm(body[FACE, :2] - ours[FACE], axis=1)) / unit),
            "head_dv": float(np.median(body[FACE, 1] - ours[FACE, 1]) / unit),
            "unit": unit,
        })
    if not rows:
        raise SystemExit("no sampled frame could be compared")

    # HOW MUCH THE RENDER MOVES, against how much it was told to.  A character
    # that follows the pose in POSITION can still be doing a smaller dance, and
    # "舞蹈质量变差了" has no other measurable form: sampled frames are far
    # enough apart that this is amplitude, not jitter.
    order = [r["frame"] for r in rows]
    detected_seq = np.stack([detected[f][BODY, :2] / rows[i]["unit"]
                             for i, f in enumerate(order)])
    driven_seq = np.stack([driven[int(round(f * 30.0 / 16.0))][BODY]
                           / rows[i]["unit"] for i, f in enumerate(order)])
    moved_render = float(np.median(np.linalg.norm(np.diff(detected_seq, axis=0), axis=2)))
    moved_pose = float(np.median(np.linalg.norm(np.diff(driven_seq, axis=0), axis=2)))

    summary = {
        "render": str(args.render),
        "motion_render": moved_render,
        "motion_pose": moved_pose,
        "motion_ratio": moved_render / max(moved_pose, 1e-9),
        "sampled": len(rows),
        "person_found": found,
        "body_error": float(np.median([r["body"] for r in rows])),
        "head_error": float(np.median([r["head"] for r in rows])),
        "head_offset_down": float(np.median([r["head_dv"] for r in rows])),
    }
    print("{}".format(pathlib.Path(args.render).name))
    print("  person found in {:.1%} of {} sampled frames".format(found, len(rows)))
    print("  body  error {:.3f} of neck-to-ankle".format(summary["body_error"]))
    print("  head  error {:.3f}   (offset DOWN {:+.3f}; + means the render's face "
          "is lower than the pose asked)".format(summary["head_error"],
                                                 summary["head_offset_down"]))
    print("  moves {:.3f} against the pose's {:.3f} -> ratio {:.2f} "
          "(1.0 = the same dance; <1 means the render is doing less)".format(
              moved_render, moved_pose, summary["motion_ratio"]))
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()
