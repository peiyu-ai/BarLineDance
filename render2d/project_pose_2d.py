"""Project a generated 3D dance onto the 2D pose format the animator reads.

OUTPUT IS THE INGEST'S OWN FORMAT, deliberately: every clip already carries
DWPose output as ``keypoints.npy`` [T, 18, 2] normalised to [0, 1] plus
``scores.npy`` [T, 18], in OpenPose COCO-18 order.  Emitting the same thing for
a GENERATED clip means the real recording and the generation drive the animator
through one identical path, so a 2D result that looks wrong can be checked
against the real clip's own pose run through the same nodes -- which is the only
way to tell "the 2D stage is broken" from "the dance is bad".

SMPL has 24 joints and COCO-18 has 18, and they are not a subset of each other:

  * COCO's ``neck`` is not an SMPL joint -- it is the midpoint of the two
    shoulders, which is what DWPose's neck effectively is;
  * COCO's four face points (eyes, ears) have no SMPL equivalent at all.  They
    are synthesised from the head joint and the head's own frame so the face
    turns with the head rather than being pasted on; without them the animator
    has nothing to anchor a face to and the character's head drifts.
  * SMPL's spine, collar and hand joints have no COCO slot and are dropped.

THE CAMERA is the one the 3D renders already use, so the 2D silhouette matches
the 3D video frame for frame: looking down +y at the clip's own centroid, with
the same vertical framing ``render_avatar_video`` computes from ``reach``.  A
different camera here would make the 2D and 3D demos disagree about where the
dancer is, and the pair is the point.

...AND SO IS THE HEADING, which the first version left out and which made that
claim false.  ``render_avatar_video`` does not only place a camera, it first
ROTATES each clip about the vertical so its median facing meets that camera
(``heading_align`` / ``body_facing``), and its own docstring says why: over 120
held-out clips the ground truth's facing sits at a median **121.9 degrees from
+y**, so "a camera parked on +y watches the back of the dancer more often than
the front".  Copying the camera without that rotation is exactly what produced
a 2D pose showing the dancer's BACK for a clip whose 3D panel shows the front --
on ``7030793823240424742:clip000`` the median |yaw| is 167 degrees, i.e. the
whole clip.  The operator saw it in the animation before any number did.

The rotation is rigid about z, so no joint angle and no relative motion changes,
and **every turn the dancer makes WITHIN the clip survives** -- that is the
property the renderer relies on too.  The target heading is the clip's OWN
median facing, never the ground truth's: the 3D strip can align the generated
rows to the reference row because the reference is on screen beside them, but
here there is no reference panel and reading the target clip's ground truth
would be reading its motion.
"""
import argparse
import json
import pathlib
import pickle

import numpy as np

# SMPL-24 indices.  Names from the skeleton this repository decodes with
# (``vis.SMPLSkeleton``); the ordering is the standard SMPL one.
SMPL = {"pelvis": 0, "l_hip": 1, "r_hip": 2, "spine1": 3, "l_knee": 4, "r_knee": 5,
        "spine2": 6, "l_ankle": 7, "r_ankle": 8, "spine3": 9, "l_foot": 10,
        "r_foot": 11, "neck": 12, "l_collar": 13, "r_collar": 14, "head": 15,
        "l_shoulder": 16, "r_shoulder": 17, "l_elbow": 18, "r_elbow": 19,
        "l_wrist": 20, "r_wrist": 21, "l_hand": 22, "r_hand": 23}

COCO_NAMES = ["nose", "neck", "r_shoulder", "r_elbow", "r_wrist",
              "l_shoulder", "l_elbow", "l_wrist", "r_hip", "r_knee", "r_ankle",
              "l_hip", "l_knee", "l_ankle", "r_eye", "l_eye", "r_ear", "l_ear"]

# COCO slot -> SMPL joint, for the ones that map directly.
DIRECT = {2: "r_shoulder", 3: "r_elbow", 4: "r_wrist",
          5: "l_shoulder", 6: "l_elbow", 7: "l_wrist",
          8: "r_hip", 9: "r_knee", 10: "r_ankle",
          11: "l_hip", 12: "l_knee", 13: "l_ankle"}

FPS = 30.0

# Face proportions, medians of the real DWPose of the fixed ten clips, as
# fractions of the eye-to-neck distance; and the two vertical anchors, as
# fractions of the neck-to-hip distance.  Re-derivable with the census in
# docs/DANCE_QUALITY_DEFECTS.md section 68.
NOSE_BELOW_EYE = 0.153
EAR_BELOW_EYE = 0.119
# HORIZONTAL against HORIZONTAL: the widths are given relative to the SHOULDER
# width, not to the eye-to-neck distance.  Mixing the two costs a factor of
# height/width: the projection is isotropic in PIXELS, so a normalised u and a
# normalised v are not the same unit, and a ratio measured as (width in u) /
# (height in v) came out 1.62x too wide when applied in world space.
EYE_SEPARATION_OF_SHOULDERS = 0.223
EAR_SEPARATION_OF_SHOULDERS = 0.566
EYE_ABOVE_NECK = 0.355
HEAD_JOINT_ABOVE_NECK = 0.178

# Which way the camera's viewer sits, as ``render_avatar_video.body_facing``
# measures facing.  DERIVED, not chosen: see ``tests/test_render2d_projection.py``
# -- projecting ground truth with each candidate and correlating against the
# clip's own DWPose ``keypoints.npy`` (the real 2D of the real video) picks it,
# the same way the x negation below was picked.
CAMERA_FACING = np.array([0.0, 1.0, 0.0])


def align_heading(joints, target=CAMERA_FACING):
    """Rotate a clip about the vertical so its median facing meets the camera.

    THE RENDERER'S OWN FUNCTIONS, imported rather than reimplemented: a second
    copy of "which way is this body pointing" is exactly the kind of thing that
    drifts, and the whole point here is that the 2D and the 3D panels agree.
    The import is ~7 s because that module pulls in pyrender and trimesh, which
    is paid once per clip and is worth it.
    """
    import sys as _sys
    _sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from tools.render_avatar_video import heading_align, body_facing  # noqa: E402

    joints = np.asarray(joints, dtype=np.float64)
    rotation = heading_align(joints, np.asarray(target, dtype=np.float64))
    # About the clip's own horizontal centre, so the rotation does not also
    # translate the dancer out of frame.
    centre = joints[:, 0, :2].mean(0)
    flat = joints.reshape(-1, 3).copy()
    flat[:, :2] -= centre
    flat[:] = flat @ rotation.T
    flat[:, :2] += centre
    turned = flat.reshape(joints.shape)
    before = body_facing(joints)
    after = body_facing(turned)
    return turned, before, after


def head_frame(joints):
    """Orthonormal (right, up, forward) at the head, per frame.

    ``forward`` is the chest normal rather than anything read off the head
    joint itself: SMPL's head has an orientation but the decoded joint
    POSITIONS do not carry it, and the shoulders do.
    """
    left, right = joints[:, SMPL["l_shoulder"]], joints[:, SMPL["r_shoulder"]]
    across = left - right
    across = across / np.maximum(np.linalg.norm(across, axis=1, keepdims=True), 1e-6)
    up = np.tile(np.array([0.0, 0.0, 1.0]), (len(joints), 1))
    forward = np.cross(across, up)
    forward = forward / np.maximum(np.linalg.norm(forward, axis=1, keepdims=True), 1e-6)
    return across, up, forward


def to_coco18(joints, head_radius=None):
    """[T, 24, 3] SMPL world joints -> [T, 18, 3] COCO-18 world points."""
    joints = np.asarray(joints, dtype=np.float64)
    out = np.zeros((len(joints), 18, 3), dtype=np.float64)
    for slot, name in DIRECT.items():
        out[:, slot] = joints[:, SMPL[name]]
    # neck: the shoulder midpoint, which is what DWPose's neck is
    out[:, 1] = 0.5 * (joints[:, SMPL["l_shoulder"]] + joints[:, SMPL["r_shoulder"]])
    across, up, forward = head_frame(joints)
    head = joints[:, SMPL["head"]]
    # THE FACE IS BUILT FROM MEASURED PROPORTIONS, not from a head radius.
    #
    # The first version placed five points on a fixed 0.16 m sphere and guessed
    # their offsets.  Compared against the REAL DWPose of the same ten clips
    # (the ingest's own ``keypoints.npy``), every one of them was wrong, and the
    # operator saw the result: "头一直低着,脸不对着相机,这个是在很多 clip 存在
    # 的问题".  As a fraction of the eye-to-neck distance:
    #
    #     nose below the eyes   real 0.153   guessed 0.621   (4.1x too low)
    #     ears below the eyes   real 0.119   guessed -0.138  (WRONG SIGN)
    #     eye separation        real 0.561   guessed 1.763   (3.1x too wide)
    #     ear separation        real 1.446   guessed 3.746   (2.6x too wide)
    #
    # A nose dropped four times too far with the ears above the eyes IS a face
    # looking at the floor, drawn three times life size; the animator rendered
    # what it was shown.
    #
    # The numbers below are those real medians.  The vertical anchor is measured
    # too: in real DWPose the eyes sit 0.355 of the neck-to-hip distance above
    # the neck, while SMPL's head JOINT sits only 0.178 above it -- the joint is
    # at about the jaw, not at eye level -- so the face rises a further 0.177 of
    # that distance.  It stays attached to the head joint rather than to the
    # neck, so a real bow of the head still moves it; only the SIZE is fixed,
    # from the clip's own median torso, so the face does not breathe.
    across, up, forward = head_frame(joints)
    head = joints[:, SMPL["head"]]
    up_unit = np.array([0.0, 0.0, 1.0])
    neck_z = out[:, 1, 2]
    hip_z = 0.5 * (joints[:, SMPL["l_hip"], 2] + joints[:, SMPL["r_hip"], 2])
    torso = float(np.median(neck_z - hip_z))
    eye_to_neck = EYE_ABOVE_NECK * torso
    # ANCHORED TO THE NECK FOR THE MEDIAN, TO THE HEAD FOR THE MOTION.  A fixed
    # rise above the head joint makes the eye-to-neck distance -- the unit every
    # face proportion is expressed in -- depend on that clip's own head-to-neck
    # distance, so a body built a little differently gets a differently
    # proportioned face.  Taking the head joint's DEVIATION from its own clip
    # median instead pins the median exactly while every real nod and bow within
    # the clip still moves the face.
    # eye_z - neck_z == eye_to_neck at the median, plus this clip's own nods:
    #   want   eye_z = neck_z + eye_to_neck + (h - median(h)),  h = head_z - neck_z
    #   so     eye_z = head_z + eye_to_neck - median(h)
    # i.e. a constant rise above the head joint, using THIS CLIP's median head
    # height rather than the corpus's HEAD_JOINT_ABOVE_NECK.
    head_rise = joints[:, SMPL["head"], 2] - neck_z
    rise = eye_to_neck - float(np.median(head_rise))
    shoulders = float(np.median(np.linalg.norm(
        joints[:, SMPL["l_shoulder"], :2] - joints[:, SMPL["r_shoulder"], :2],
        axis=1)))
    half_eye = 0.5 * EYE_SEPARATION_OF_SHOULDERS * shoulders
    half_ear = 0.5 * EAR_SEPARATION_OF_SHOULDERS * shoulders
    eye_level = head + up_unit * rise
    # A small forward offset on nose and eyes keeps the face geometrically in
    # front of the skull.  The view is orthographic down +y, so it does not
    # reach the image; it is there so the points mean what they are named.
    out[:, 0] = (eye_level - up_unit * NOSE_BELOW_EYE * eye_to_neck
                 + forward * 0.25 * eye_to_neck)
    out[:, 14] = eye_level - across * half_eye + forward * 0.15 * eye_to_neck
    out[:, 15] = eye_level + across * half_eye + forward * 0.15 * eye_to_neck
    out[:, 16] = (eye_level - up_unit * EAR_BELOW_EYE * eye_to_neck
                  - across * half_ear)
    out[:, 17] = (eye_level - up_unit * EAR_BELOW_EYE * eye_to_neck
                  + across * half_ear)
    return out


def project(points, width, height, margin=0.12, extent=None):
    """World points -> normalised [0, 1] image coordinates, y down.

    ``extent`` lets extra points (synthesised hands) ride through the SAME
    camera without changing it: the framing is derived from the body's height,
    and a fan of finger tips would move that. Pass the body-only array there and
    the full array as ``points``; default None keeps the old behaviour exactly.

    Orthographic down +y, the same axis the 3D renders look along, and scaled
    by the WHOLE CLIP's extent rather than per frame: a per-frame fit would
    make the character breathe in and out, which reads as a zoom the dance does
    not contain.
    """
    x = points[..., 0]
    z = points[..., 2]
    # THE BODY, NOT THE TRAVEL, sets the scale.  Fitting the clip's whole
    # extent shrinks the dancer by however far they walk: checked on the first
    # clip, the figure came out at about 45% of frame height with wide margins,
    # and a pose that small is a weak conditioning signal -- the encoder sees
    # fewer pixels of every limb.  The height of the BODY (ankles to head,
    # median over frames) is the invariant to frame by; the centre still
    # follows the whole clip so the dancer never leaves the frame.
    # ``span`` is metres per ONE unit of u (the horizontal axis).  The body's
    # height is a VERTICAL measurement, and v has (height/width) as many units
    # per metre as u does, so converting it to a u-span means multiplying by
    # width/height.  Getting this backwards put the figure at 135% of frame
    # height with 29% of its joints outside the frame; getting the division in
    # `v` backwards put it at 43%.  Both were arithmetic, and both were caught
    # by measuring the rendered figure rather than by looking at it.
    # DERIVED, after guessing the direction wrong twice (43% of frame height,
    # then 240%).  Requirement: one metre is the same number of PIXELS on both
    # axes, and the body fills (1 - 2*margin) of the frame HEIGHT.
    #
    #   u = x / span            -> pixels_x = u * W = x * W / span
    #   v = z / (span * k)      -> pixels_z = v * H = z * H / (span * k)
    #   equal pixels per metre  => W / span = H / (span * k)  => k = H / W
    #   body fills the height   => body * H / (span * k) = (1 - 2m) * H
    #                           => span = body / (1 - 2m) * W / H
    #
    # Checked numerically on the first clip: body 1.462 m gives span 1.0821 and
    # a body of exactly 0.760 v units, which is the margin.
    z_extent = z if extent is None else np.asarray(extent)[..., 2]
    body = float(np.nanmedian(np.nanmax(z_extent, axis=1) - np.nanmin(z_extent, axis=1)))
    span = max(body, 1e-6) / (1.0 - 2 * margin) * (width / height)
    aspect = height / width
    # THE FRAME FOLLOWS THE DANCER, low-passed.  Fitting the clip's whole extent
    # was the first attempt and it put the figure at 43% of frame height: on
    # this clip the travel extent is 1.80 m against a 1.46 m body, so the walk,
    # not the dancer, was setting the scale.  A per-frame centre would instead
    # lock the root to the middle and hide the travel entirely, which is a
    # different lie -- so the centre is the root smoothed over about a second,
    # which keeps the dancer large while the drift stays visible.
    root = 0.5 * (points[:, 8, :] + points[:, 11, :])        # hip midpoint
    window = max(3, int(round(FPS)) | 1)
    pad = window // 2
    padded = np.pad(root, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window) / window
    centre = np.stack([np.convolve(padded[:, axis], kernel, mode="valid")
                       for axis in range(3)], axis=1)
    cx = centre[:, 0:1]
    cz = centre[:, 2:3]
    # x is NEGATED.  Validated against the clip's own DWPose output: with the
    # naive mapping every one of the 18 joints correlated -0.82 to -0.93 in x
    # and +0.52 to +0.95 in y -- a consistent sign flip on one axis, i.e. a
    # left-right mirror, not a bad mapping (a bad mapping would scatter the
    # correlations, not flip them all together).  The world is z-up with the
    # camera looking down +y, so the image's right hand is -x.
    # ONE metre must be the same number of PIXELS on both axes, so the axis
    # that is longer in pixels spans FEWER normalised units per metre.  The
    # first version multiplied v by width/height instead of dividing, which on
    # a 720x1280 portrait frame squeezed the dancer to 43% of frame height --
    # arithmetic, not taste: 0.76 of the frame by the margin, times 0.5625, is
    # exactly the 43% that was measured.  A pose that small is a weaker
    # conditioning signal, since the encoder sees fewer pixels of every limb.
    u = 0.5 - (x - cx) / span
    v = 0.5 - (z - cz) / span / aspect
    return np.stack([u, v], axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True, help="a generated or eval .pkl")
    ap.add_argument("--out", required=True, help="directory for keypoints/scores")
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=1280)
    ap.add_argument("--no-align-heading", dest="align_heading",
                    action="store_false",
                    help="skip the rotation that turns the clip's median facing "
                         "toward the camera; the 2D then shows whatever the "
                         "arbitrary world heading of the reconstruction gives, "
                         "which is the dancer's back more often than the front")
    args = ap.parse_args()

    blob = pickle.load(open(args.motion, "rb"))
    if "full_pose" not in blob:
        raise SystemExit("{} has no full_pose".format(args.motion))
    joints = np.asarray(blob["full_pose"], dtype=np.float64)
    if joints.ndim != 3 or joints.shape[1] != 24:
        raise SystemExit("expected [T, 24, 3] joints, got {}".format(joints.shape))

    turn = None
    if args.align_heading:
        joints, before, after = align_heading(joints)
        turn = {"before": [float(v) for v in before],
                "after": [float(v) for v in after]}
        print("heading aligned: median facing {:+.3f},{:+.3f} -> {:+.3f},{:+.3f} "
              "(camera at {:+.0f},{:+.0f})".format(
                  before[0], before[1], after[0], after[1],
                  CAMERA_FACING[0], CAMERA_FACING[1]))
    coco = to_coco18(joints)
    uv = project(coco, args.width, args.height)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "keypoints.npy", uv.astype(np.float32))
    # Every joint is synthesised, so every score is 1.0 -- and that is recorded
    # rather than left implicit, because the real clips' scores are genuine
    # detector confidences and a reader comparing the two must not mistake one
    # for the other.
    np.save(out / "scores.npy", np.ones(uv.shape[:2], dtype=np.float32))
    (out / "meta.json").write_text(json.dumps({
        "source_motion": str(args.motion),
        "frames": int(len(uv)), "fps": FPS,
        "video_w": args.width, "video_h": args.height,
        "format": "openpose_coco18_normalised",
        "scores_are_synthetic": True,
        "joint_names": COCO_NAMES,
        "heading_alignment": turn,
    }, indent=2))
    print("{} frames -> {}".format(len(uv), out))


if __name__ == "__main__":
    main()
