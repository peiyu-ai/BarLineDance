"""Does the MTV video do OUR dance?  Detected 2D body against our projected 3D joints.

WHY IT HAS TO EXIST.  The first MTV renders loaded, sampled, decoded and muxed
without a single warning -- and the character stood still, because the base had
no motion path (``render2d/comfy_mtv.py:assert_motion_path``).  Every "it ran"
signal was green.  The only thing that can say whether the render FOLLOWS the
motion is to look for the motion in the pixels.

HOW.  The workflow's own detector (``render2d/frame_character.character_pose``,
AAPose-20, OpenPose-18 order) is run over the rendered video.  Our joints are the
camera-frame array the render was conditioned on (``joints3d.npy``: mm, y-down,
z = depth), projected with a pinhole (x/z, y/z).  MTV is never told a camera, so
the video is free to choose its own framing; the comparison therefore fits ONE
similarity (scale + offset) for the WHOLE clip and reads the residual:

  follow     median joint distance after the whole-clip fit, in units of the
             detected neck-to-ankle length.  Position AND pose: a character that
             takes the right shapes but stays put fails here.
  shape      the same after a PER-FRAME fit: limb configuration only.
  moves      frame-to-frame body motion of the render against the projected
             pose's, same units.  A static render reads ~0 whatever its shape.
  found      fraction of sampled frames whose detected body has a median keypoint
             confidence >= PERSON_CONFIDENCE.  NOT "the detector returned a box":
             on the first MTV-base render (a beige wall after frame 0) the detector
             returned a body in 11/11 frames, at median confidence 0.09-0.55,
             against 0.87-0.91 on a real rendered person.  Only these frames are scored.

THE CONTROL, printed next to every reading: the same fit with our pose shifted
by half the clip (``null_follow``).  A render that follows must beat it.  A
render whose ``follow`` is no better than its own shifted pose is not following,
and ``tests/test_score_mtv_follow.py`` pins that the statistic can tell.
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

# OpenPose-18 slot -> SMPL-24 joint.  Subject's right = OpenPose "R" = SMPL 2/5/8/17/19/21.
PAIRS = [(1, 12), (2, 17), (3, 19), (4, 21), (5, 16), (6, 18), (7, 20),
         (8, 2), (9, 5), (10, 8), (11, 1), (12, 4), (13, 7)]
# Nose, eyes, ears: the detector's confidence here falls when the face melts, which
# is the artefact the 6-step distill LoRA produces under strong motion.  Descriptive
# only -- an anime face is out of the detector's domain, so it ranks arms of the SAME
# character and nothing else.
FACE = [0, 14, 15, 16, 17]
OPENPOSE = [p for p, _ in PAIRS]
SMPL = [s for _, s in PAIRS]
MIN_CONFIDENCE = 0.3
PERSON_CONFIDENCE = 0.6


def sample_frames(path, stride):
    with tempfile.TemporaryDirectory() as work:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
                        "-vf", "select='not(mod(n\\,{}))'".format(stride), "-vsync", "0",
                        "{}/%05d.png".format(work)], check=True)
        for index, name in enumerate(sorted(pathlib.Path(work).iterdir())):
            yield index * stride, Image.open(name).convert("RGB")


def detect(video, stride, device):
    from frame_character import character_pose
    detected, faces, tried = {}, {}, 0
    for index, image in sample_frames(video, stride):
        tried += 1
        try:
            body = character_pose(image, device)
        except SystemExit:
            continue
        detected[index] = body[OPENPOSE]
        faces[index] = body[FACE]
    return detected, tried, faces


def people(detected):
    """Only the frames where the detector is confident a body is there."""
    return {f: d for f, d in detected.items() if np.median(d[:, 2]) >= PERSON_CONFIDENCE}


def project(joints):
    """[T, 24, 3] camera mm -> [T, 13, 2] pinhole coordinates of the matched joints."""
    picked = np.asarray(joints, dtype=np.float64)[:, SMPL]
    depth = np.maximum(picked[..., 2], 1.0)
    return picked[..., :2] / depth[..., None]


def fit_similarity(source, target, weight):
    """Scale + offset (no rotation: the camera is upright) minimising weighted L2."""
    w = weight[..., None]
    total = max(float(weight.sum()), 1e-9)
    mu_s = (source * w).sum(axis=tuple(range(source.ndim - 1))) / total
    mu_t = (target * w).sum(axis=tuple(range(target.ndim - 1))) / total
    cs, ct = source - mu_s, target - mu_t
    scale = float((cs * ct * w).sum() / max((cs * cs * w).sum(), 1e-12))
    return scale, mu_t - scale * mu_s


def body_unit(points):
    neck = points[:, 0]
    ankle = 0.5 * (points[:, 9] + points[:, 12])
    return np.linalg.norm(ankle - neck, axis=-1)


def score(detected, projected, fps_ratio=1.0, shift=0):
    """Readings for one pairing of detected frames with projected pose frames."""
    frames = sorted(detected)
    rows = []
    for f in frames:
        src = (int(round(f * fps_ratio)) + shift) % len(projected)
        rows.append((f, src))
    det = np.stack([detected[f][:, :2] for f, _ in rows])
    conf = np.stack([detected[f][:, 2] for f, _ in rows])
    ours = np.stack([projected[s] for _, s in rows])
    weight = (conf > MIN_CONFIDENCE).astype(np.float64)
    unit = np.median(body_unit(det))
    scale, offset = fit_similarity(ours, det, weight)
    placed = ours * scale + offset
    err = np.linalg.norm(placed - det, axis=-1)
    follow = float(np.median(err[weight > 0])) / unit
    per_frame = []
    for i in range(len(rows)):
        if weight[i].sum() < 6:
            continue
        s, o = fit_similarity(ours[i:i + 1], det[i:i + 1], weight[i:i + 1])
        e = np.linalg.norm(ours[i] * s + o - det[i], axis=-1)
        per_frame.append(np.median(e[weight[i] > 0]))
    shape = float(np.median(per_frame)) / unit if per_frame else float("nan")
    both = (weight[1:] * weight[:-1]) > 0
    moved_det = np.linalg.norm(np.diff(det, axis=0), axis=-1)[both]
    moved_ours = np.linalg.norm(np.diff(placed, axis=0), axis=-1)[both]
    return {"follow": follow, "shape": shape,
            "moves_render": float(np.median(moved_det)) / unit,
            "moves_pose": float(np.median(moved_ours)) / unit,
            "sampled": len(rows), "unit_px": float(unit), "fit_scale": scale}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--joints", required=True, help="the joints3d.npy the render was conditioned on")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--device", default="CPUExecutionProvider")
    ap.add_argument("--start", type=int, default=0,
                    help="model frame of the joints the video's first frame corresponds to")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    joints = np.load(args.joints)[args.start:]
    raw, tried, faces = detect(args.video, args.stride, args.device)
    detected = people(raw)
    if len(detected) < 4:
        summary = {"video": str(args.video), "joints": str(args.joints),
                   "found": len(detected) / max(tried, 1), "follow": None}
        print("{}  NO PERSON: confident body in only {} of {} sampled frames".format(
            pathlib.Path(args.video).name, len(detected), tried))
        if args.out:
            pathlib.Path(args.out).write_text(json.dumps(summary, indent=1))
        raise SystemExit(2)
    projected = project(joints)
    real = score(detected, projected)
    null = score(detected, projected, shift=len(projected) // 2)
    body_conf = float(np.median([np.median(d[:, 2]) for d in detected.values()]))
    face_conf = float(np.median([np.median(faces[f][:, 2]) for f in detected]))
    summary = {"video": str(args.video), "joints": str(args.joints),
               "found": len(detected) / max(tried, 1),
               "body_conf": body_conf, "face_conf": face_conf, **real,
               "null_follow": null["follow"], "null_shape": null["shape"],
               "follow_over_null": real["follow"] / max(null["follow"], 1e-9)}
    print("{}  found {:.0%} of {}  follow {:.3f} (null {:.3f}, ratio {:.2f})  "
          "shape {:.3f} (null {:.3f})  moves {:.3f} vs pose {:.3f}  conf {:.2f}/face {:.2f}".format(
              pathlib.Path(args.video).name, summary["found"], tried, real["follow"],
              null["follow"], summary["follow_over_null"], real["shape"], null["shape"],
              real["moves_render"], real["moves_pose"], body_conf, face_conf))
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()
