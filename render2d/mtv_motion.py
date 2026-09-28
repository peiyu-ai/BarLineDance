"""Our SMPL motion in the camera frame MTV-Crafter's tokenizer was trained on.

WHY THIS EXISTS AT ALL.  The 2D path draws the dance as a flat skeleton, and an
orthographic projection cannot represent movement along the view axis: measured
on the fixed ten clips, the wrists keep 80% of their motion and the whole body
77%.  MTV-Crafter is the one method we found that conditions on the joints
themselves -- its tokenizer's first layer is ``encoder.conv_in.weight``
[128, 3, 3, 3], three channels of raw XYZ, with no rendering step anywhere --
so the depth axis survives.  And in its own training statistics depth is not a
minor channel: reading ``MTV/data/std.npy``, the per-axis standard deviations
are x 243.2 mm, y 335.3 mm, **z 618.0 mm**.  The axis our projection never
reads is the one with the most variance in theirs.

THE FRAME.  ``MTV/data/mean.npy`` is [24, 3] in millimetres, camera space,
y-down, z = distance from the lens; its per-axis means are x +0.8, y +183.0,
z +1779.9, and it FACES the lens: the subject's left hip is at x +38, the
wrists are 100 mm nearer the lens than the pelvis.  Ours is metres, world, z-up;
``align_heading`` turns the chest to face +y -- toward a camera standing on the
+y side and looking back along -y.  So:

    x_cam = -x_world           (the subject's left, at world -x, lands at +x)
    y_cam = -z_world           (up becomes down)
    z_cam = -y_world           (nearer the +y camera = smaller depth)

a proper rotation (det +1).  **The first version had x_cam = x, z_cam = +y** --
also a proper rotation, but a camera on the -y side: MTV was shown the dancer's
BACK, against a front-facing reference image, and the render barely moved
(2026-09-17: arms at the sides through a dance with arms out and raised).
``assert_faces_camera`` now refuses a clip whose left shoulder is not on +x.

then millimetres, then a rigid offset per axis so the clip's own median lands on
the training mean.  The offset is RIGID and per clip: it moves where the dancer
stands relative to the lens, which is a framing choice, and leaves every
relative motion -- the thing being conditioned on -- untouched.

TWO MORE FRAMING CHOICES, forced by the first render on the MTV-finetuned base
(2026-09-17).  With the rigid offset alone, 818's first 81 frames put the pelvis
525-901 mm right of centre -- z = +4.65 against a training spread of 149 mm --
and the model did what it was told: the character left the frame on the right
and every frame after the first was the background wall.  (Motion strength 0 on
the same graph rendered a sane standing character, so the collapse was the motion
path, and the only blob left in frame was sliding off the right edge.)

  * FOLLOW-CAM.  The generator's root travels 2.4 m across a clip; MTV's training
    videos keep the pelvis within a 149 mm (1 sd) lateral band, i.e. a camera
    that frames the dancer.  The slow part of the horizontal root (Gaussian,
    ``FOLLOW_SECONDS``) is subtracted, which is what that camera does: steps and
    weight shifts faster than a second survive, the drift does not.  Height is
    never touched.  This is the same trade the strip makes with ``--lock-root``,
    done without reading anybody else's trajectory.
  * FRAMING BY DEPTH, NOT BY SIZE (corrected 2026-09-18).  The first version
    SCALED the clip to the training mean's neck-to-ankle (1350 -> 985 mm) so the
    dancer would fill the frame the way theirs does.  That is x0.73 on every
    displacement as well, and it is why the dance did not arrive: measured on
    7030, the conditioned wrists' temporal spread fell to 144/146/158 mm against
    a training per-joint std of 285/428/651, whole-clip |z| p99 0.81 against
    training data spanning +-2-3.  **The mechanism is the DiT's response, not
    information loss**: the VQ-VAE round-trips these clips to a 72 mm median
    per-joint error, so the tokens do carry the shape -- they are of small
    MAGNITUDE, and a small motion embedding moves the video little.  Reading the
    round trip again after this change separates the two: under the response
    reading it barely moves, under information loss it would fall.  A control
    from the peer session: ground-truth CAPTURED motion through the old path read
    z-score std 0.38 against our 0.24, so the damping is in the path, not in our
    generated dance.
    Apparent size is a function of depth as well as of height, so the framing is
    taken from DEPTH: their mean skeleton is 985 mm at 1822 mm, an angular size
    of 0.54, so a 1350 mm body goes at 1350 / 0.54 = 2500 mm.  Same size on
    screen, every displacement still in the millimetres the tokenizer was
    trained on.  ``--match-size`` keeps the old behaviour for comparison.
"""
import argparse
import pathlib
import pickle
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from render2d.project_pose_2d import align_heading  # noqa: E402
import os

# Root of the sibling checkouts (ComfyUI_Wan, Lodge, this repo); set E2E_ROOT.
E2E_ROOT = os.environ.get("E2E_ROOT", "/workspace/e2e")

STATS = pathlib.Path(E2E_ROOT + "/ComfyUI_Wan/custom_nodes/"
                     "ComfyUI-WanVideoWrapper/MTV/data")
MODEL_FPS = 16.0
SOURCE_FPS = 30.0
# At the training depth the frame is narrow, so the slow root drift has to go; at
# the depth this module now uses it is about 1040 mm wide, and 2 s keeps more of
# the dancer's own travel.
FOLLOW_SECONDS = 2.0
NECK, L_ANKLE, R_ANKLE, PELVIS = 12, 7, 8, 0


def load_stats():
    return np.load(STATS / "mean.npy"), np.load(STATS / "std.npy")


def follow_cam(joints, seconds=FOLLOW_SECONDS, fps=SOURCE_FPS):
    """World z-up joints with the slow horizontal root drift removed (x, y only)."""
    joints = np.asarray(joints, dtype=np.float64).copy()
    if seconds <= 0:
        return joints
    from scipy.ndimage import gaussian_filter1d
    slow = gaussian_filter1d(joints[:, PELVIS, :2], sigma=seconds * fps, axis=0, mode="nearest")
    joints[..., :2] -= slow[:, None, :]
    return joints


def match_body_size(joints, mean):
    """Uniformly scale world joints so the median neck-to-ankle height matches the
    training mean's (camera y is world -z, so the same vertical extent)."""
    ours = np.median(joints[:, NECK, 2] - 0.5 * (joints[:, L_ANKLE, 2] + joints[:, R_ANKLE, 2]))
    theirs = (0.5 * (mean[L_ANKLE, 1] + mean[R_ANKLE, 1]) - mean[NECK, 1]) / 1000.0
    return joints * (theirs / max(ours, 1e-6)), float(theirs / max(ours, 1e-6))


def training_angular_size(mean):
    """Their mean skeleton's neck-to-ankle height divided by its depth."""
    height = 0.5 * (mean[L_ANKLE, 1] + mean[R_ANKLE, 1]) - mean[NECK, 1]
    return float(height / mean[PELVIS, 2])


def depth_for_training_size(joints, mean):
    """The depth in mm that puts OUR body at the training mean's apparent size."""
    ours = np.median(joints[:, NECK, 2] - 0.5 * (joints[:, L_ANKLE, 2] + joints[:, R_ANKLE, 2]))
    return float(ours * 1000.0 / training_angular_size(mean))


def to_camera_frame(joints, mean, std, depth=None):
    """[T, 24, 3] world z-up metres -> [T, 24, 3] camera y-down millimetres.

    The rigid offset is SOLVED, not guessed.  The tokenizer normalises per
    JOINT, so matching the overall median leaves a residual -- it read +0.62 on
    the vertical axis, because our body's vertical distribution is shaped
    differently from theirs even once centred.  Since
    ``z = (c + o - m) / s``, the offset that puts the mean z-score at zero is

        o = -mean((c - m) / s) / mean(1 / s)

    per axis, which lands the clip in the middle of the distribution it will be
    tokenised against.  It is still one rigid translation of the whole clip:
    where the dancer stands relative to the lens is a framing choice, and every
    relative motion -- the thing actually being conditioned on -- is untouched.
    """
    joints = np.asarray(joints, dtype=np.float64)
    camera = np.stack([-joints[..., 0], -joints[..., 2], -joints[..., 1]], axis=-1)
    camera *= 1000.0
    inv = 1.0 / np.maximum(std, 1e-6)
    offset = np.empty(3)
    for axis in range(3):
        residual = ((camera[..., axis] - mean[None, :, axis]) * inv[None, :, axis]).mean()
        offset[axis] = -residual / inv[:, axis].mean()
    if depth is not None:
        # Depth is a framing choice (how far the lens is), so it is set outright
        # rather than solved: the solved value centres the clip in their DEPTH
        # distribution, which for an unscaled body means the wrong apparent size.
        offset[2] = depth - np.median(camera[..., 2])
    return camera + offset


L_SHOULDER, R_SHOULDER = 16, 17


def assert_faces_camera(camera, mean=None):
    """The training mean faces the lens (left shoulder at +x).  So must we."""
    across = np.median(camera[:, L_SHOULDER, 0] - camera[:, R_SHOULDER, 0])
    if mean is not None and (mean[L_SHOULDER, 0] - mean[R_SHOULDER, 0]) <= 0:
        raise SystemExit("the training mean itself does not face +x; the convention changed")
    if across <= 0:
        raise SystemExit("left shoulder is {:.0f} mm from the right along x: the clip shows "
                         "the camera its back".format(across))
    return float(across)


def resample(values, source_fps=SOURCE_FPS, target_fps=MODEL_FPS):
    """Linear in time, so a beat lands where it lands rather than on a dropped
    frame.  The generator runs at 30 and the model at 16."""
    count = int(round(len(values) * target_fps / source_fps))
    if count < 2 or count == len(values):
        return values
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, count)
    flat = values.reshape(len(values), -1)
    out = np.stack([np.interp(target, source, flat[:, c])
                    for c in range(flat.shape[1])], axis=1)
    return out.reshape(count, *values.shape[1:])


def prepare(motion_path, align=True, follow_seconds=FOLLOW_SECONDS, match_size=False,
            target_fps=MODEL_FPS):
    blob = pickle.load(open(motion_path, "rb"))
    joints = np.asarray(blob["full_pose"], dtype=np.float64)
    if joints.ndim != 3 or joints.shape[1] != 24:
        raise SystemExit("expected [T, 24, 3] joints, got {}".format(joints.shape))
    if align:
        joints, _before, _after = align_heading(joints)
    mean, std = load_stats()
    joints = follow_cam(joints, follow_seconds)
    depth = None
    if match_size:
        joints, _scale = match_body_size(joints, mean)
    else:
        depth = depth_for_training_size(joints, mean)
    camera = to_camera_frame(joints, mean, std, depth=depth)
    assert_faces_camera(camera, mean)
    return resample(camera, target_fps=target_fps), mean, std


def report(camera, mean, std):
    z = (camera - mean[None]) / np.maximum(std[None], 1e-6)
    return {
        "frames": int(len(camera)),
        "z_mean": float(z.mean()),
        "z_std": float(z.std()),
        "within_3": float(np.mean(np.abs(z) < 3.0)),
        "per_axis_z_mean": [float(v) for v in z.reshape(-1, 3).mean(axis=0)],
        "pelvis_z_mean": [float(v) for v in z[:, PELVIS].mean(axis=0)],
        "wrist_spread_mm": [float(v) for v in camera[:, [20, 21]].std(axis=0).mean(axis=0)],
        "body_mm": float(np.median(camera[:, L_ANKLE, 1] - camera[:, NECK, 1])),
        "depth_mm": float(np.median(camera[..., 2])),
        "pelvis_z_absmax": [float(v) for v in np.abs(z[:, PELVIS]).max(axis=0)],
        "depth_spread_mm": float(np.median(camera[..., 2].max(axis=1)
                                           - camera[..., 2].min(axis=1))),
        "lateral_spread_mm": float(np.median(camera[..., 0].max(axis=1)
                                             - camera[..., 0].min(axis=1))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True)
    ap.add_argument("--out", default=None, help="write the [T,24,3] array as .npy")
    ap.add_argument("--no-align-heading", dest="align", action="store_false")
    ap.add_argument("--follow-seconds", type=float, default=FOLLOW_SECONDS,
                    help="remove horizontal root drift slower than this (0 = keep the travel)")
    ap.add_argument("--match-size", action="store_true",
                    help="scale the body to the training mean instead of framing by depth "
                         "(the 2026-09-17 behaviour; it damps every displacement by x0.73)")
    ap.add_argument("--fps", type=float, default=MODEL_FPS,
                    help="resample to this rate (30 = the generator's own; upstream MTV-Crafter "
                         "tokenises consecutive source frames, stride 1)")
    args = ap.parse_args()

    camera, mean, std = prepare(args.motion, align=args.align,
                                follow_seconds=args.follow_seconds, match_size=args.match_size,
                                target_fps=args.fps)
    stats = report(camera, mean, std)
    print("{} -> {} frames at {:g} fps".format(
        pathlib.Path(args.motion).name, stats["frames"], args.fps))
    print("  in the tokenizer's normalised space: mean {:+.3f}, std {:.3f}, "
          "{:.1%} within |z|<3".format(stats["z_mean"], stats["z_std"],
                                       stats["within_3"]))
    print("  per-axis z: x {:+.3f}  y {:+.3f}  z {:+.3f}".format(*stats["per_axis_z_mean"]))
    print("  pelvis z: mean x {:+.2f} y {:+.2f} z {:+.2f}; max |z| x {:.2f} y {:.2f} z {:.2f}".format(
        *stats["pelvis_z_mean"], *stats["pelvis_z_absmax"]))
    print("  body {:.0f} mm at depth {:.0f} mm; wrist spread x {:.0f} y {:.0f} z {:.0f} mm "
          "(training per-joint std 285/428/651)".format(
              stats["body_mm"], stats["depth_mm"], *stats["wrist_spread_mm"]))
    print("  body spread: depth {:.0f} mm against lateral {:.0f} mm -- the depth "
          "the 2D projection discards".format(stats["depth_spread_mm"],
                                              stats["lateral_spread_mm"]))
    if args.out:
        np.save(args.out, camera.astype(np.float32))
        print("  wrote {}".format(args.out))


if __name__ == "__main__":
    main()
