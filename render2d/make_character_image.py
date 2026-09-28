"""The single still the animator paints onto the pose -- rendered from our own VRM.

WHY NOT A DOWNLOADED PICTURE.  This machine cannot reach huggingface.co,
cdn-lfs, modelscope or github (checked host by host: pypi and
files.pythonhosted.org answer 200, the rest fail), so a pipeline that starts
with "fetch a character image" cannot run here at all.  The repository already
ships a VRM humanoid -- ``third_party/vrm/anime_female.vrm.glb``, the official
VRM 1.0 sample -- and it is exactly the 2D-cartoon look the demo wants.

WHY A REST POSE AND NOT A DANCE FRAME.  The animator reads the still for
IDENTITY (who this character is) and the pose video for MOTION.  A still taken
mid-dance hands it a second, contradictory motion cue, and the character then
fights its own reference.  The frame here is the first frame of the clip with
the body put in its own median pose, which keeps the character's proportions
and lighting identical to the 3D renders the demo sits beside.
"""
import argparse
import pathlib
import pickle
import subprocess
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]


def median_pose_clip(motion_path, out_pickle, frames=8):
    """A short clip of the dancer's own median pose, for the renderer to draw."""
    blob = pickle.load(open(motion_path, "rb"))
    poses = np.asarray(blob["smpl_poses"], dtype=np.float32)
    trans = np.asarray(blob["smpl_trans"], dtype=np.float32)
    full = np.asarray(blob["full_pose"], dtype=np.float32)
    # The MEDIAN pose, not frame 0: frame 0 is wherever the clip happens to
    # start, which on a third of these clips is mid-movement.
    still_pose = np.median(poses, axis=0)
    still_trans = np.median(trans, axis=0)
    index = int(np.argmin(np.abs(poses - still_pose).sum(axis=1)))
    payload = dict(blob)
    payload["smpl_poses"] = np.repeat(poses[index:index + 1], frames, axis=0)
    payload["smpl_trans"] = np.repeat(still_trans[None, :], frames, axis=0)
    payload["full_pose"] = np.repeat(full[index:index + 1], frames, axis=0)
    pathlib.Path(out_pickle).parent.mkdir(parents=True, exist_ok=True)
    with open(out_pickle, "wb") as handle:
        pickle.dump(payload, handle)
    return index


def crop_to_character(raw, out, size, margin=0.06):
    """Crop to the character on a flat ground, dropping the panel title.

    Two things in a ``render_avatar_video`` frame are wrong for an identity
    still and were visible on the first one: the panel's TITLE is drawn into
    the image (the animator would learn it as content), and the checkerboard
    floor fills the lower half, so the character is only about two thirds of
    the frame.  Both are removed here rather than by adding flags to the 3D
    renderer, whose framing serves the side-by-side comparison and should not
    be bent to serve this.
    """
    from PIL import Image

    image = Image.open(raw).convert("RGB")
    pixels = np.asarray(image, dtype=np.int16)
    # The character is the only strongly coloured thing: skin, hair and the
    # shoes are chromatic, while the title is grey text and the floor and sky
    # are near-neutral.
    chroma = (pixels.max(axis=2) - pixels.min(axis=2)) > 18
    chroma[:int(0.06 * image.height)] = False          # the title band
    ys, xs = np.nonzero(chroma)
    if len(xs) < 200:
        image.save(out)
        return {"box": [0, 0, image.width, image.height],
                "crop": [image.width, image.height], "paste": [0, 0],
                "side": max(image.width, image.height)}
    pad = int(margin * max(image.width, image.height))
    box = (max(0, xs.min() - pad), max(0, ys.min() - pad),
           min(image.width, xs.max() + pad), min(image.height, ys.max() + pad))
    crop = image.crop(box)
    geometry = {"box": [int(v) for v in box], "crop": [crop.width, crop.height]}
    # Square, so the animator is not handed a letterboxed identity.
    side = max(crop.width, crop.height)
    canvas = Image.new("RGB", (side, side), tuple(int(v) for v in pixels[2, 2]))
    canvas.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    geometry["paste"] = [(side - crop.width) // 2, (side - crop.height) // 2]
    geometry["side"] = side
    canvas.resize((size, size), Image.LANCZOS).save(out)
    return geometry


def write_still_joints(motion_pickle, image_path, geometry, size):
    """The still's COCO-18 joints, mapped through the same crop the image took."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from project_pose_2d import to_coco18, project

    blob = pickle.load(open(motion_pickle, "rb"))
    joints = np.asarray(blob["full_pose"], dtype=np.float64)
    coco = to_coco18(joints)[0]

    # RENDERED FROM THE SAME PROJECTION, not fitted to the drawing.
    #
    # Three geometric fits were tried and each was defeated by the character's
    # art rather than by arithmetic: matching bounding boxes squashed the figure
    # (4/18 joints on the body), anchoring the nose to the silhouette's top put
    # the shoulders at the chin because the hair rises above the nose (3/18),
    # and taking the widest upper row as the shoulder line found the HAIR, which
    # on this character is wider than the shoulders (5/18).
    #
    # The still is ours: it is rendered from a pose we hold.  So the joints are
    # produced by the SAME orthographic projection that makes the driving pose
    # video (``project_pose_2d.project``), and the still is re-rendered to match
    # that projection instead of the 3D comparison camera -- which is a
    # PERSPECTIVE camera with a distance solved from ``fill`` and ``stature``,
    # and therefore something no orthographic fit can ever reproduce.  One
    # projection for both sides, by construction.
    from PIL import Image

    drawing = Image.open(image_path).convert("RGB")
    uv = project(coco[None, ...], drawing.width, drawing.height)[0]
    mapped = uv * np.array([drawing.width, drawing.height], dtype=np.float64)
    np.save(pathlib.Path(image_path).with_suffix(".joints.npy"),
            mapped.astype(np.float32))


RENDER_SIZE = 768


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True)
    ap.add_argument("--out", required=True, help="output .png")
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--vrm", default=str(REPO / "third_party/vrm/anime_female.vrm.glb"))
    args = ap.parse_args()

    staging = pathlib.Path(args.out).with_suffix(".still.pkl")
    frame = median_pose_clip(args.motion, staging)
    staging_motion = staging
    video = pathlib.Path(args.out).with_suffix(".still.mp4")
    subprocess.run([sys.executable, str(REPO / "tools/render_avatar_video.py"),
                    "--motion", "character:{}".format(staging),
                    "--output", str(video), "--size", str(args.size),
                    "--vrm", args.vrm, "--view", "front"], check=True, cwd=REPO)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
                    "-vf", "select=eq(n\\,2)", "-vframes", "1", str(staging.with_suffix(".raw.png"))],
                   check=True)
    box = crop_to_character(staging.with_suffix(".raw.png"), args.out, args.size)
    # THE STILL'S OWN JOINTS, in the cropped image's coordinates.  The puppet
    # backend warps FROM these, so they have to come from the same pose and the
    # same camera the still was drawn with -- reading them off the still with a
    # detector would introduce a second source of truth for the same thing.
    write_still_joints(staging_motion, args.out, box, args.size)
    video.unlink(missing_ok=True)
    staging.unlink(missing_ok=True)
    staging.with_suffix(".raw.png").unlink(missing_ok=True)
    pathlib.Path(str(video) + ".source.json").unlink(missing_ok=True)
    print("median pose was frame {} -> {}".format(frame, args.out))


if __name__ == "__main__":
    main()
