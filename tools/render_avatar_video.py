#!/usr/bin/env python3
"""Render generated motion as a skinned SMPL body, several arms sharing one stage.

Why this exists beside ``tools/render_dance_video.py`` rather than replacing it
------------------------------------------------------------------------------
That renderer draws line collections and says why in its own header: *"no mesh,
no pytorch3d, nothing that can silently fake geometry -- what is on screen is
exactly the joint positions the model produced."*  That is a real property and
this file does not have it: a skin can hide a bad joint, and a light can hide a
bad skin.  So both stay, the stick figure remains the one to reach for when the
question is *what did the model output*, and this one is for the question a
stick figure genuinely cannot answer -- **does this read as a person dancing**.

Everything drawn here is derived from the pickle's own ``smpl_poses`` and
``smpl_trans``; nothing is retargeted, smoothed, or re-posed.  The check that
this is true is in the code and runs on every clip: the SMPL joints implied by
the mesh are compared against the pickle's ``full_pose``, and a disagreement
over a millimetre stops the render.  Measured 2026-08-29 on a shipped clip the
agreement is **5.4e-07 m**, so a millimetre is four orders of magnitude of slack
and the gate still cannot pass a substituted body.

Three things are load-bearing and each of them was a defect first
----------------------------------------------------------------

**The translation offset.**  ``smplx`` places the pelvis at ``J0_rest +
transl`` while this repository's ``SMPLSkeleton`` places the root *at*
``root_positions``.  Feeding ``smpl_trans`` straight in puts the body a constant
[-0.002, -0.241, 0.029] m from where the model put it -- constant, so it reads
as a slightly odd camera rather than as an error.  ``transl = smpl_trans -
J0_rest`` is what makes the joint gate above pass.

**One stage for every arm.**  ``render_dance_video.render`` records what
separate renders cost: each derives its own cube, so N arms are N
metres-per-pixel, "the arm that travels furthest gets the largest cube, is
therefore drawn smallest, and its drift reads as the calmest" -- a model that
got worse producing a picture that looks better.  Here the camera *and the
ground plane* are computed once from all arms together.

**The floor is the reference's, not each arm's.**  Ground truth on this corpus
does not sit at z = 0: measured over 31 clips the lowest joint averages
0.353 m, and on the probe clip the floor is 0.459 m.  Giving each arm its own
floor would slide each body down onto its own plane and **delete the very
defect this render was asked to show** -- generated bodies sit about 3 cm higher
than the ground truth while moving vertically far less (sustained airborne
0.03-0.05 of frames against the ground truth's 0.23), which on screen is a
dancer who never lands.  The floor comes from the first motion handed in, by the
rule ``tools/build_dance_gallery.py:358`` already uses: the **5th percentile** of
the per-frame lowest foot joint, not the minimum, because "one frame of a foot
punched through the ground would otherwise define the ground plane, and wild
reconstructions do that".

Usage::

    python3 tools/render_avatar_video.py --motion gt:runs/wild_v5_song_gt_eval/motion/<clip>.pkl \\
        --motion "bar grid:runs/m6_.../motion/<clip>_s20260816.pkl" \\
        --audio data/wild_ingest_v1/<upload>__<clip>/audio.wav \\
        --output out.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import pickle
import shutil
import subprocess
import sys
import tempfile

# Set before pyrender is imported: it reads the platform at import time, and on
# this host there is no display.  EGL here resolves to Mesa/llvmpipe (measured
# 2026-08-29 -- there is no NVIDIA EGL vendor ICD registered), which is CPU
# rasterisation and still renders a 13,776-face body at 0.011 s/frame.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import pyrender
import torch
import trimesh

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

FPS = 30
FOV = np.pi / 3.4
JOINT_TOLERANCE_M = 1e-3
# Ankles and toes, the same four this repository's contact channel uses
# (tools/build_dance_gallery.py:312).
FOOT_JOINTS = (7, 8, 10, 11)
SMPL_MODEL = "third_party/smpl_models/SMPL_MALE_clean.pkl"
SMPL_MODELS = "third_party/smpl_models/SMPL_{}_clean.pkl"
UV_OBJ = "third_party/smpl_models/uv/smpl_uv.obj"
DEFAULT_TEXTURE = "third_party/smpl_models/uv/dancer_female_01.png"
# SMPL-X, and going to it is a return rather than a change.  The converter's own
# metadata records what the source was: "SMPL local joint rotations (SMPL-X body
# joints 1..21; 22/23 identity)" -- our 24-joint vector is SMPL-X's 21 body
# joints with two identity rotations padded on (measured: |axis-angle| on joints
# 22/23 is at most 4.7e-03).  So poses[:, 3:66] feeds SMPL-X body_pose directly.
# It buys three things the SMPL male model cannot: a gender choice, real hands
# and feet (10,475 vertices against 6,890), and the shape parameters below.
SMPLX_MODELS = ("/cache/atomicdance-assets/third_party/GVHMR/inputs/checkpoints"
                "/body_models/smplx/SMPLX_{}.npz")
# Measured 2026-08-29: the female and male SMPL-X templates differ by 42.2 mm
# mean / 109.2 mm max per vertex, and 13.5 cm of stature (1.659 m vs 1.794 m).
# This is a real choice about who is on screen, not a colour swatch.
# Distinguishable at a glance and in greyscale, and the reference row is the one
# that reads as skin: an arm should not be able to look better by being prettier.
ROW_COLOURS = [(0.78, 0.72, 0.66), (0.42, 0.53, 0.70), (0.76, 0.55, 0.42),
               (0.50, 0.65, 0.52), (0.62, 0.52, 0.68)]


def load_smpl_motion(path):
    """``(smpl_poses [T,72], smpl_trans [T,3], joints [T,24,3] or None)``.

    Two layouts, because the two things being compared are stored differently
    and neither can be converted into the other after the fact:

    * **a generated pickle** -- ``smpl_poses`` / ``smpl_trans``, and
      ``full_pose`` for the gate.  ``full_pose`` is *joint positions*, not pose
      parameters, despite the name: ``full_pose[:, 0]`` is bitwise
      ``smpl_trans``.  Anything reading it as SMPL parameters gets coordinates,
      so the name is checked against the array rather than trusted.
    * **a converted ingest clip directory** -- the ground truth.  Its eval
      pickle (``tools/export_wild_eval_motion.py``) writes joints ONLY, so the
      ground truth cannot be skinned from the file the FID set was built from;
      the parameters live in ``data/wild3d/ingest_v1_converted/<clip>/`` as
      ``pose_axis_angle_z_up.npy`` [T,24,3] and ``root_translation_z_up.npy``.
      That directory also carries ``atomic_motion_151.npy``, which decodes to
      joints independently, so the gate stays armed on this path too rather
      than being skipped for the one row it matters most on.
    """
    path = pathlib.Path(path)
    if path.is_dir():
        raw = path / "atomic_motion_151.npy"
        if not raw.is_file():
            raise SystemExit("{} holds no atomic_motion_151.npy".format(path))
        # Through the repo's own decode, not through the sibling
        # pose_axis_angle_z_up.npy: 151-D is the encoding the generated rows are
        # decoded from, so taking the ground truth the same way means a
        # difference on screen is a difference in the motion rather than in the
        # path it arrived by -- the argument tools/build_dance_gallery.py makes
        # for reading ground truth through the generated clip's forward
        # kinematics.  It also arms the gate for free: the joints here come from
        # SMPLSkeleton and the joints below come from smplx, independently.
        from tools.render_dance_video import _decode_raw_151
        joints, payload = _decode_raw_151(np.load(raw))
        return (np.asarray(payload["smpl_poses"], np.float32),
                np.asarray(payload["smpl_trans"], np.float32),
                np.asarray(joints, np.float32))

    with path.open("rb") as handle:
        blob = pickle.load(handle)
    for key in ("smpl_poses", "smpl_trans"):
        if key not in blob:
            raise SystemExit(
                "{} carries no '{}'; a skinned render needs the SMPL parameters, "
                "and 'full_pose' is joint positions rather than parameters "
                "despite the name. For ground truth pass the converted ingest "
                "clip directory instead -- the eval pickle holds joints only."
                .format(path, key))
    poses = np.asarray(blob["smpl_poses"], np.float32)
    trans = np.asarray(blob["smpl_trans"], np.float32)
    if poses.ndim != 2 or poses.shape[1] != 72:
        raise SystemExit("{}: smpl_poses is {}, expected [T, 72] axis-angle"
                         .format(path, poses.shape))
    joints = np.asarray(blob["full_pose"], np.float32) if "full_pose" in blob else None
    return poses, trans, joints


def skin(poses, trans, model_path=SMPL_MODEL, joints_gate=None, gender=None,
         betas=None, family="smpl"):
    """Mesh vertices for a whole clip, with the joint gate applied.

    ``gender`` selects the SMPL-X path (male / female / neutral); leaving it
    ``None`` keeps the SMPL male model every earlier artifact was rendered with,
    so nothing already published changes silently.

    ``betas`` is this dancer's own shape.  Without it every arm is the model's
    mean body, which is not merely plainer -- it is a body nobody in the corpus
    has, and limb lengths change what a movement looks like.  The parameters are
    in the source GVHMR result (``smpl_params_global.betas``, [T,10]); the
    median over frames is used, because GVHMR re-estimates shape every frame and
    a per-frame body would breathe.
    """
    import smplx

    frames = len(poses)
    shape = (torch.zeros(frames, 10) if betas is None
             else torch.as_tensor(np.asarray(betas, np.float32))[None].repeat(frames, 1))
    if family == "smplx":
        # Better hands and feet, but SMPL-X has its own topology and the vendored
        # UV unwrap is SMPL's, so this path cannot be textured.
        body = smplx.SMPLX(model_path=SMPLX_MODELS.format((gender or "neutral").upper()),
                           gender=(gender or "neutral").upper(), batch_size=frames,
                           use_pca=False, flat_hand_mean=True)
        body_pose = torch.from_numpy(poses[:, 3:66])
    else:
        path = SMPL_MODELS.format(gender.upper()) if gender else model_path
        body = smplx.SMPL(model_path=path, gender=(gender or "male").upper(),
                          batch_size=frames)
        body_pose = torch.from_numpy(poses[:, 3:])
    rest_pelvis = (body.J_regressor[0] @ body.v_template).detach()
    with torch.no_grad():
        out = body(betas=shape,
                   global_orient=torch.from_numpy(poses[:, :3]),
                   body_pose=body_pose,
                   transl=torch.from_numpy(trans) - rest_pelvis)
    vertices = out.vertices.numpy()
    # 0..21 only.  SMPL's joints 22/23 are the hands; SMPL-X's are the jaw and an
    # eye, so a [:, :24] comparison lines a chin up against a wrist and reads
    # 0.76 m of drift that is not there.  The gate caught exactly that on the
    # first SMPL-X render, which is the argument for it existing.
    joints = out.joints.numpy()[:, :24]
    shared = 22
    if joints_gate is not None:
        # Root-relative, because a body model with different limb lengths puts
        # the joints somewhere else BY DESIGN and an absolute comparison would
        # only be measuring the shape change.  What must not change is the pose:
        # the angles the model produced.  On the SMPL path with zero betas this
        # is the same test as before to within the root, and it still refuses a
        # substituted motion.
        left = joints[:, :shared] - joints[:, :1]
        right = joints_gate[:, :shared] - joints_gate[:, :1]
        drift = float(np.abs(left - right).max())
        tolerance = (JOINT_TOLERANCE_M if (gender is None and betas is None)
                     else 0.30)
        if drift > tolerance:
            raise SystemExit(
                "the skinned body's joints are {:.4f} m (root-relative) from the "
                "joints in the pickle, over the {:.4f} m this gate allows -- the "
                "mesh is not hanging on the motion the model produced, so "
                "rendering it would show a body nobody generated"
                .format(drift, tolerance))
    return vertices, body.faces.astype(np.int64), joints


# Vertex -> body part, 27 parts covering all 10,475 SMPL-X vertices.  Downloaded
# 2026-08-29 from Meshcapade's public wiki
# (assets/SMPL_body_segmentation/smplx/smplx_vert_segmentation.json) and vendored
# beside the body models.  It is what lets the mesh wear clothes without a
# texture map, a UV layout, or a licence-gated scan: colour is assigned per
# vertex by which part it belongs to.
SEGMENTATION = {
    "smplx": "third_party/smpl_models/smplx_vert_segmentation.json",
    "smpl": "third_party/smpl_models/smpl_vert_segmentation.json",
}
# A dancer, not a mannequin.  The reviewer's objection to the first pass was that
# a naked grey body is hard to read AND hard to look at; the parts here are the
# ones a leotard-and-leggings outfit covers, so limb boundaries stay visible
# (which is what a reader needs) while the figure stops being a medical model.
OUTFIT = (
    (("spine", "spine1", "spine2", "leftShoulder", "rightShoulder",
      "leftArm", "rightArm"), (0.85, 0.29, 0.34)),
    (("hips", "leftUpLeg", "rightUpLeg", "leftLeg", "rightLeg"), (0.15, 0.17, 0.26)),
    (("leftFoot", "rightFoot", "leftToeBase", "rightToeBase"), (0.96, 0.96, 0.98)),
    (("head",), (0.23, 0.17, 0.16)),
)
SKIN = (0.80, 0.64, 0.52)


def outfit_colours(vertex_count, family):
    """Per-vertex RGBA for the dressed style, or ``None`` if the map is absent."""
    path = pathlib.Path(SEGMENTATION[family])
    if not path.is_file():
        return None
    parts = json.loads(path.read_text(encoding="utf-8"))
    colours = np.tile(np.array([*SKIN, 1.0], np.float32), (vertex_count, 1))
    for names, rgb in OUTFIT:
        for name in names:
            index = np.asarray(parts.get(name, []), int)
            if len(index):
                colours[index] = [*rgb, 1.0]
    return (colours * 255).astype(np.uint8)


STYLES = {
    # (base colour, ambient, key intensity, fill intensity, roughness)
    # Exposure is a measurement, not a taste: the first pass ran key 4.2 with
    # ambient 0.32 on a 0.78 base and every body clipped to near-white, which
    # removes the shading that carries the form -- a render that cannot show a
    # crouch is not a diagnostic.
    "realistic": ((0.72, 0.52, 0.45), 0.16, 2.6, 1.1, 0.68),
    "flat":      ((0.36, 0.62, 0.72), 0.30, 2.2, 1.0, 0.90),
    "toon":      ((0.36, 0.62, 0.72), 0.30, 2.2, 1.0, 0.90),
    "dressed":   ((0.80, 0.64, 0.52), 0.30, 2.9, 1.3, 0.72),
}


def source_betas(clip):
    """This dancer's own shape, from the GVHMR result the corpus was built from.

    ``data/wild3d/ingest_v1_converted`` kept the pose and dropped the shape, so
    this reaches past it to ``ingest_v1_gvhmr_raw/<clip>/hmr4d_results.pt``,
    which is remote.  Returns ``None`` when it cannot be fetched, and the caller
    says so rather than quietly rendering the mean body -- "we used the dancer's
    build" and "we used the average one" must not print the same.
    """
    import io as _io

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from tools import asset_io

    stem = clip.split(":", 1)[1].replace(":", "__") if ":" in clip else clip
    key = "data/wild3d/ingest_v1_gvhmr_raw/{}/hmr4d_results.pt".format(stem)
    try:
        blob = torch.load(_io.BytesIO(asset_io.read_bytes(key)), map_location="cpu",
                          weights_only=False)
    except Exception:                                            # noqa: BLE001
        return None
    betas = (blob.get("smpl_params_global") or {}).get("betas")
    if betas is None:
        return None
    return np.median(np.asarray(betas, np.float32), axis=0)


def toon(colour, depth, base):
    """Quantise the shading into bands and draw a silhouette line.

    Post-process rather than a shader because pyrender's pipeline is fixed; the
    depth buffer it already returns is enough for both halves.
    """
    lit = colour.astype(np.float32).mean(2) / 255.0
    bands = np.digitize(lit, [0.30, 0.52, 0.72]).astype(np.float32) / 3.0
    out = np.asarray(base, np.float32)[None, None, :] * (0.45 + 0.55 * bands[..., None])
    body = depth > 0
    edge = np.zeros_like(body)
    for axis in (0, 1):
        for shift in (1, -1):
            edge |= body ^ np.roll(body, shift, axis)
    inner = np.nan_to_num(np.hypot(*np.gradient(np.where(body, depth, np.nan)))) > 0.010
    out[~body] = (0.95, 0.95, 0.96)
    out[(edge | inner) & body] = (0.10, 0.13, 0.17)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def vrm_rows(rows, model_path, gender):
    """Skin a VRM avatar with each row's motion, in this tool's z-up frame.

    Returns ``[(frames_of_meshes, joints)]``.  The avatar's geometry is its own
    -- bone lengths, clothes, hair -- and only the joint angles and the scaled
    root come from the motion, so the three panels differ by the dance and by
    nothing else.
    """
    import smplx

    from tools import vrm_retarget as V

    avatar = V.VRMAvatar(model_path)
    primitives = avatar.skinned_primitives()
    images = {}
    for prim in primitives:
        if prim["material"] not in images:
            images[prim["material"]] = avatar.material_image(prim["material"])
    body = smplx.SMPL(model_path=SMPL_MODELS.format((gender or "female").upper()),
                      gender=(gender or "female").upper(), batch_size=1)
    rest = (body.J_regressor @ body.v_template).detach().numpy()
    align = avatar.alignment(rest)
    hips = avatar.global_rest[avatar.by_name["J_Bip_C_Hips"]][:3, 3]
    head = avatar.global_rest[avatar.by_name["J_Bip_C_Head"]][:3, 3]
    scale = float(np.linalg.norm(head - hips) / np.linalg.norm(rest[15] - rest[0]))
    out = []
    for _title, path in rows:
        poses, trans, joints_gate = load_smpl_motion(path)
        poses = poses.reshape(len(poses), 24, 3)
        rotations = V.axis_angle_to_matrix(poses)
        globals_ = np.zeros_like(rotations)
        for joint, parent in enumerate(V.SMPL_PARENTS):
            globals_[:, joint] = (rotations[:, joint] if parent < 0
                                  else globals_[:, parent] @ rotations[:, joint])
        out.append({"avatar": avatar, "primitives": primitives, "images": images,
                    "globals": globals_, "trans": trans, "align": align,
                    "scale": scale, "joints": joints_gate})
    return out


def vrm_frame(row, index):
    """``[(vertices_z_up, faces, image, uv)]`` for one row at one frame."""
    from tools import vrm_retarget as V

    avatar = row["avatar"]
    frame = min(index, len(row["globals"]) - 1)
    posed = np.stack([V.to_y_up_rotation(g) for g in row["globals"][frame]])
    trans = np.asarray(row["trans"][frame], float).copy()
    shift = row.get("root_shift")
    if shift is not None:
        trans[:2] += shift[min(frame, len(shift) - 1)]
    translation = V.Z_UP_TO_Y_UP @ trans
    transforms = avatar.pose(posed, translation, row["align"], row["scale"])
    out = []
    for prim in row["primitives"]:
        vertices = V.skin_vertices(prim, transforms)
        # back into the tool's z-up world, so the floor, the camera and the
        # stage arithmetic below are shared with the SMPL path unchanged
        vertices = vertices @ V.Z_UP_TO_Y_UP
        turn = row.get("heading")
        if turn is not None:
            rotation, centre = turn
            flat = np.asarray(vertices).reshape(-1, 3).copy()
            flat[:, :2] -= centre
            flat[:] = flat @ rotation.T
            flat[:, :2] += centre
            vertices = flat.reshape(np.asarray(vertices).shape)
        slide = row.get("position")
        if slide is not None:
            # Applied AFTER the heading rotation, matching the SMPL path's
            # order: the rotation is about the row's own centre, so rotating a
            # already-translated row would move it somewhere else.
            vertices = np.asarray(vertices).copy()
            vertices[..., :2] += slide
        out.append((vertices,
                    prim["indices"].reshape(-1, 3),
                    row["images"][prim["material"]], prim["uv"]))
    return out


def load_uv():
    """``(uv [C,2], faces [F,3])`` for the un-welded textured mesh.

    SMPL's unwrap splits the seams, so 6,890 vertices carry 7,576 UV
    coordinates and a corner cannot be addressed by vertex id.  The mesh is
    therefore rebuilt with one vertex per face corner.  This costs 3x the
    vertices at render time and changes nothing about the geometry: corner k of
    face f is still exactly ``vertices[faces[f, k]]``.
    """
    coords, corner_v, corner_t = [], [], []
    for line in pathlib.Path(UV_OBJ).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "vt":
            coords.append((float(parts[1]), float(parts[2])))
        elif parts[0] == "f":
            corner_v.append([int(x.split("/")[0]) - 1 for x in parts[1:4]])
            corner_t.append([int(x.split("/")[1]) - 1 for x in parts[1:4]])
    return (np.asarray(coords, np.float32), np.asarray(corner_v, np.int64),
            np.asarray(corner_t, np.int64))


def floor_of(joints):
    """This clip's ground plane: 5th percentile of the lowest foot joint.

    Not the minimum -- ``tools/build_dance_gallery.py:358`` records why: one
    frame of a foot punched through the ground would otherwise define the plane,
    and wild reconstructions do that.
    """
    lowest = np.asarray(joints)[:, FOOT_JOINTS, 2].min(axis=1)
    return float(np.percentile(lowest, 5))


def _z_up_to_y_up():
    return trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])


def stage(all_joints, floor, margin=0.45):
    """One camera and one ground plane for every arm, in y-up render space.

    Derived from every arm at once, for the reason
    ``render_dance_video.render`` states: a per-arm cube silently rescales each
    panel, and the arm that travels furthest is drawn smallest.
    """
    points = np.concatenate([np.asarray(j).reshape(-1, 3) for j in all_joints], 0)
    centre_x = float(np.median(points[:, 0]))
    centre_y = float(np.median(points[:, 1]))
    # Framed on the BODY, not on the trajectory.  Sizing the shot by how far the
    # arms travel makes a drifting arm draw itself small -- the failure
    # render_dance_video.render records -- and it also shrinks every arm to fit
    # the worst one.  A person is about 1.8 m; the shot is that plus margin, and
    # horizontal drift is left to read against the floor grid instead of being
    # absorbed into the zoom.
    stature = float(np.percentile(points[:, 2], 99) - floor)
    radius = max(stature, 1.2) / 2 + margin
    return {"centre": (centre_x, centre_y), "floor": floor, "radius": radius,
            "stature": stature, "facing": np.array([0.0, 1.0, 0.0])}


def _ground_mesh(stage_spec, extent=24.0, tile=0.5):
    """A checkerboard floor, because a plain plane hides the thing being judged.

    The camera below follows the dancer horizontally so the body stays large
    enough to read -- which on a featureless plane makes translation invisible,
    and root translation is one of the defects this render is meant to expose
    (root jerk measured at 5.6x ground truth with no draft at all).  A 0.5 m
    grid puts the drift back on screen: the dancer slides across the tiles.
    """
    squares = int(extent / tile)
    origin_x = stage_spec["centre"][0] - extent / 2
    origin_y = stage_spec["centre"][1] - extent / 2
    meshes = []
    for row in range(squares):
        for column in range(squares):
            if (row + column) % 2:
                continue
            quad = trimesh.creation.box(extents=[tile, tile, 0.006])
            quad.apply_translation([origin_x + (column + 0.5) * tile,
                                    origin_y + (row + 0.5) * tile,
                                    stage_spec["floor"] - 0.003])
            meshes.append(quad)
    dark = trimesh.util.concatenate(meshes)
    dark.visual.vertex_colors = np.tile([206, 211, 218, 255], (len(dark.vertices), 1))
    light = trimesh.creation.box(extents=[extent, extent, 0.004])
    light.apply_translation([stage_spec["centre"][0], stage_spec["centre"][1],
                             stage_spec["floor"] - 0.006])
    light.visual.vertex_colors = np.tile([233, 236, 240, 255], (len(light.vertices), 1))
    combined = trimesh.util.concatenate([light, dark])
    combined.apply_transform(_z_up_to_y_up())
    return combined


def heading_align(joints, target):
    """Rotate a motion about the vertical so its median heading is ``target``.

    **This changes no joint angle and no relative motion** -- it is a rigid
    rotation of the whole clip about z, which is the same freedom a camera
    placement has.  It is legitimate here because the world heading of a
    monocular reconstruction is arbitrary: ``convert_gvhmr_result``'s output is
    gravity-aligned, so up is real, but nothing fixes which way is "front".

    Why it is needed: the camera is placed on the reference row's facing, and
    measured 2026-08-30 the generated rows' median facing sits **71.9 and 74.9
    degrees** away from the ground truth's on one of three sample clips -- those
    panels were being watched from the side while the reference faced the
    viewer.  Aligning the headings removes that without giving each panel its
    own camera, which would break the shared stage the floor and the scale
    depend on.  What survives is every turn the dancer makes *within* the clip,
    which is what a reviewer is judging.
    """
    joints = np.asarray(joints, float)
    current = body_facing(joints)
    cosine = float(np.dot(current, target))
    sine = float(current[0] * target[1] - current[1] * target[0])
    angle = np.arctan2(sine, cosine)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                         [np.sin(angle), np.cos(angle), 0.0],
                         [0.0, 0.0, 1.0]])
    return rotation


def body_facing(joints):
    """The clip's median facing in the ground plane, as a unit vector.

    Measured 2026-08-29 over 120 held-out clips: the ground truth's own facing
    sits at a median 121.9 degrees from +y, so a camera parked on +y watches the
    back of the dancer more often than the front.  The generated arms are NOT
    flipped -- their facing differs from the ground truth's by a median 33 to 35
    degrees and exceeds 120 degrees on 3 clips of 120 -- but 33 degrees is enough
    to turn one panel away while another still faces you, which is what made a
    camera bug look like a pose bug.  Taking the median over the clip keeps the
    shot still while the dancer turns.
    """
    joints = np.asarray(joints)
    # up x (left->right) is forward, not (left->right) x up.  The first version
    # had the operands the other way round and parked the camera behind the
    # dancer -- which is indistinguishable from "the pose is flipped" until you
    # check it against a body whose facing you already know.
    hip = joints[:, 2, :] - joints[:, 1, :]
    forward = np.cross(np.array([0.0, 0.0, 1.0]), hip)
    forward[:, 2] = 0.0
    norm = np.linalg.norm(forward, axis=1, keepdims=True)
    forward = forward / np.maximum(norm, 1e-9)
    median = np.median(forward, axis=0)
    length = np.linalg.norm(median)
    return median / length if length > 1e-6 else np.array([0.0, 1.0, 0.0])


def _camera_pose(stage_spec, look_at=None, azimuth=1.0, view="front"):
    """``front`` is straight on, level, and never rotates.

    The first pass used a three-quarter angle that also drifted with the dancer,
    and a reviewer could not compare two panels because the shot itself was
    moving.  A fixed azimuth costs one thing -- when the dancer turns, you see
    their back -- and buys the thing a reviewer needs: the same viewpoint on
    every frame and every arm, so a difference on screen is a difference in the
    dance.  The camera still tracks the dancer horizontally, or the subject
    walks out of shot; the floor grid is what keeps that travel visible.
    """
    centre_x, centre_y = look_at if look_at is not None else stage_spec["centre"]
    stature = stage_spec.get("stature", 1.75)
    # Distance solved from the field of view rather than guessed as a multiple
    # of some radius: the body should fill about 72% of the frame height, and
    # h = 2 d tan(fov/2), so d = 0.72^-1 * stature / (2 tan(fov/2)).  Guessing a
    # multiplier is how the first pass put the dancer at a twelfth of the frame.
    distance = stature / (2 * stage_spec.get("fill", 0.72) * np.tan(FOV / 2))
    # ...then backed off until the widest row-to-row separation also fits.
    # Measured 2026-08-30 over 100 clips: the generated root travels 0.33 m
    # against the ground truth's 1.21 m, so with the camera on the reference
    # alone the peak separation is a median 1.198 m and a p90 of 2.005 m
    # against a half-frame of ~1.25 m -- half the clips put the generated
    # dancer on the frame edge and the p90 clip puts it outside altogether.
    # The reviewer then reads "the model does not move" as "the model is not on
    # screen".  Backing off keeps the honest picture (nothing is re-centred per
    # row, so the difference in travel is still what the eye sees) while
    # guaranteeing there is something to see.
    spread = float(stage_spec.get("spread", 0.0))
    if spread > 0.0:
        needed = (spread / 2 + 0.5 * stature) / (stage_spec.get("fill", 0.72) * np.tan(FOV / 2))
        distance = max(distance, needed)
    # ...and the same guarantee VERTICALLY, which was missing and is what the
    # operator saw.  2026-09-12, on a frame of 7608191311518369137:clip000:
    # "人体不在相机的fov 内,被截断了".  The framing above solves the distance
    # from ``fill`` alone, i.e. for a body STANDING ON THE FLOOR with its arms
    # down; it leaves no headroom.  Measured over the 20 T eval clips, the
    # lowest foot spends 28.5% of ground truth's frames and 28.2% of the
    # generated frames more than 0.15 m above the clip's own floor, p95 0.217 m
    # and 0.229 -- so hovering is NOT the defect (both sides do it equally,
    # docs 42) and cropping it is.  Add raised arms on top of that and the head
    # leaves the frame.
    #
    # ``reach`` is the highest joint above the floor over EVERY row, so the
    # shot is framed for the tallest moment any panel reaches rather than for
    # the reference's.  A percentile rather than the max: one frame of a
    # reconstruction spike would otherwise shrink every clip.  The camera
    # target sits at floor + 0.55 * stature, so the half-frame must cover
    # ``reach - 0.55 * stature`` above it and ``0.55 * stature`` below.
    reach = float(stage_spec.get("reach", 0.0))
    if reach > 0.0:
        half = max(0.55 * stature, reach - 0.55 * stature + 0.05)
        distance = max(distance, half / np.tan(FOV / 2))
    facing = stage_spec.get("facing", np.array([0.0, 1.0, 0.0]))
    side = np.array([-facing[1], facing[0], 0.0])
    lateral, depth, lift = ((0.0, 1.0, 0.10) if view == "front"
                            else (-0.42 * azimuth, 0.88, 0.24))
    offset = facing * depth * distance + side * lateral * distance
    eye = np.array([centre_x + offset[0], centre_y + offset[1],
                    stage_spec["floor"] + 0.55 * stature + lift * distance])
    target = np.array([centre_x, centre_y, stage_spec["floor"] + 0.55 * stature])
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    pose_z_up = np.eye(4)
    pose_z_up[:3, 0] = right
    pose_z_up[:3, 1] = up
    pose_z_up[:3, 2] = -forward
    pose_z_up[:3, 3] = eye
    return _z_up_to_y_up() @ pose_z_up


def align_row_positions(skinned):
    """Slide rows 1..N in the ground plane onto row 0's MEAN root position.

    Same standing as ``align_heading``, and for the same reason: the world
    POSITION of a monocular reconstruction is arbitrary -- ``convert_gvhmr_result``
    is gravity aligned, so up is real, and nothing fixes where in the room the
    dancer stands.  The operator, 2026-09-12, on a frame of
    7608191311518369137:clip000: "人体不在相机的fov 内,被截断了".

    WHAT IT REMOVES AND WHAT IT KEEPS.  Measured over the 20 T eval clips,
    generated against ground truth: the mean root sits 0.877 m away and the
    worst single frame 1.508 m; matching the means leaves 0.862 m.  So this
    takes out the CONSTANT and leaves every metre of the difference in TRAVEL,
    which is the thing being judged -- the same split ``align_heading`` makes
    between an arbitrary world heading and the turns made within the clip.

    Why it matters to the picture and not only to tidiness: the camera backs off
    by ``spread`` (the largest row-to-row root distance) to keep every row in
    frame, and ``spread`` enters as a LATERAL extent, so a row displaced toward
    the lens is drawn larger and CROPS instead of moving sideways.  Halving the
    spread fixes the crop and keeps the bodies large, which CLAUDE.md 1.5
    requires.

    Mutates ``skinned`` in place and returns ``{row: shift}``.
    """
    reference = np.asarray(skinned[0][2])[:, 0, :2].mean(0)
    shifts = {}
    for row in range(1, len(skinned)):
        vertices, faces, joints = skinned[row]
        shift = reference - np.asarray(joints)[:, 0, :2].mean(0)

        def slide(points, shift=shift):
            points = np.asarray(points).copy()
            points[..., :2] += shift
            return points

        skinned[row] = (None if vertices is None else slide(vertices),
                        faces, slide(joints))
        shifts[row] = shift
    return shifts


def render(rows, output, audio=None, size=640, stride=1, floor=None,
           model_path=SMPL_MODEL, gender=None, use_betas=False, style="realistic",
           texture=None, family="smpl", view="front", lock_root=False, vrm=None,
           align_heading=True, align_position=True, fill=0.72):
    """``rows`` = ``[(title, pkl_path)]``, drawn side by side on one stage.

    The first row is the reference: it supplies the floor every other row stands
    on, so a body that never comes down to it is visible rather than normalised
    away.
    """
    if style not in STYLES:
        raise SystemExit("--style must be one of {}".format(sorted(STYLES)))
    base, ambient, key_power, fill_power, roughness = STYLES[style]
    betas, betas_source = None, "mean body"
    if use_betas:
        clip = rows[0][0] if False else None
        for _, path in rows:
            name = pathlib.Path(path).name
            clip = name.split("_s")[0] if name.endswith(".pkl") else pathlib.Path(path).name
            betas = source_betas(clip)
            if betas is not None:
                betas_source = "GVHMR shape of {}".format(clip)
                break
        if betas is None:
            betas_source = "mean body (the dancer's shape could not be fetched)"
    dressing, uv, vrm_state = None, None, None
    if vrm is not None:
        vrm_state = vrm_rows(rows, vrm, gender)
    if texture is not None:
        if family != "smpl":
            raise SystemExit("a texture needs the SMPL body: the vendored UV "
                             "unwrap is SMPL's and SMPL-X has another topology")
        from PIL import Image
        uv = (*load_uv(), Image.open(texture).convert("RGB"))
    skinned, titles = [], []
    for index_row, (title, path) in enumerate(rows):
        poses, trans, joints_gate = load_smpl_motion(path)
        if vrm_state is not None:
            # The stage still comes from the MOTION's own joints, not from the
            # avatar: the floor, the camera and the shared cube must not move
            # when the character does, or two runs rendered with different
            # avatars would not be comparable.
            vertices, faces, joints = None, None, joints_gate
        else:
            vertices, faces, joints = skin(poses, trans, model_path, joints_gate,
                                           gender=gender, betas=betas, family=family)
        skinned.append((vertices, faces, joints))
        titles.append(title)
    if style == "dressed":
        family = "smplx" if gender is not None else "smpl"
        dressing = outfit_colours(len(skinned[0][0][0]), family)
        if dressing is None:
            raise SystemExit(
                "--style dressed needs {}, which is not vendored here"
                .format(SEGMENTATION[family]))
    if lock_root:
        # Every row's horizontal root is moved onto the reference's, so the
        # panels can be compared as POSES.  This deliberately hides the defect
        # the default view exposes -- on this checkpoint the generated root
        # travels 0.15x as far as the ground truth's, so with a shared camera
        # the reference walks away while the arms stay put and the arms render
        # larger.  That size difference is real and is the root pathology; it
        # also makes a pose hard to read, so both views exist and the video says
        # which one it is.  Height is never touched: it carries the floating.
        #
        # The guard used to be ``lock_root and vrm_state is None``, so the flag
        # was a silent no-op on exactly the path every review is rendered on
        # (``tools/render_review_set.sh`` always passes ``--vrm``).  A flag that
        # is accepted, recorded and then not applied is the defect shape this
        # repository has paid for repeatedly; the VRM path now carries the same
        # shift, applied to its own root translation before skinning.
        reference = np.asarray(skinned[0][2])[:, 0, :2]
        for row in range(1, len(skinned)):
            vertices, faces, joints = skinned[row]
            span = min(len(vertices) if vertices is not None else len(joints),
                       len(reference))
            shift = reference[:span] - joints[:span, 0, :2]
            joints = joints[:span].copy()
            if vertices is not None:
                vertices = vertices[:span].copy()
                vertices[..., :2] += shift[:, None, :]
            joints[..., :2] += shift[:, None, :]
            skinned[row] = (vertices, faces, joints)
            if vrm_state is not None:
                vrm_state[row]["root_shift"] = shift
    frames = min(len(j if v is None else v) for v, _, j in skinned)
    vrm_stature = None
    if floor is None:
        floor = floor_of(skinned[0][2])
    if vrm_state is not None:
        # The avatar is a different body, scaled to its own proportions, so the
        # floor implied by the MOTION's joints is not the floor its feet reach --
        # the first VRM render put the reference row's shoes through the ground.
        # Sample the reference row's own lowest rendered vertex, by the same 5th
        # percentile rule, and use that for every row so the arms stay comparable.
        probe = range(0, len(vrm_state[0]["globals"]),
                      max(1, len(vrm_state[0]["globals"]) // 40))
        lowest, highest = [], []
        for frame in probe:
            for verts, _f, _i, _u in vrm_frame(vrm_state[0], frame):
                column = np.asarray(verts)[:, 2]
                lowest.append(float(column.min()))
                highest.append(float(column.max()))
        if lowest:
            floor = float(np.percentile(lowest, 5))
            # ...and the avatar's own height, for the same reason.  ``stage``
            # derives ``stature`` from the SMPL joints, but on this path the
            # floor comes from the VRM's vertices, so the subtraction mixes two
            # bodies: the avatar's feet sit below the motion's lowest joint, the
            # difference is added to the height the camera solves its distance
            # from, and the dancer is drawn small in a frame sized for someone
            # taller.  Measured on the sample clips the body filled about 46% of
            # the panel where 88% was asked for.
            vrm_stature = float(np.percentile(highest, 95)) - floor
    spec = stage([j for _, _, j in skinned], floor)
    if vrm_stature is not None:
        spec["stature"] = vrm_stature
    # How much of the frame height the body should occupy.  0.72 was chosen
    # when the shot had to hold two rows that drift apart; with --lock-root the
    # rows share a track and the extra headroom just makes the dancer small,
    # which is what a reviewer notices first.
    spec["fill"] = float(fill)
    spec["facing"] = body_facing(skinned[0][2])
    if align_heading and len(skinned) > 1:
        # Rotate every row about the vertical so all share the reference's
        # median heading.  Measured 2026-08-30 on the sample clips: the
        # generated rows' median facing sits up to 74.9 degrees from the ground
        # truth's, so with one camera on the reference's facing those panels
        # were watched from the side.  This is a rigid rotation about z -- no
        # joint angle and no relative motion changes -- and the world heading of
        # a monocular reconstruction is arbitrary to begin with
        # (convert_gvhmr_result is gravity-aligned, so up is real; nothing fixes
        # front).  Every turn the dancer makes WITHIN the clip survives, which
        # is what is being judged.
        target = spec["facing"]
        for row in range(1, len(skinned)):
            vertices, faces, joints = skinned[row]
            rotation = heading_align(joints, target)
            centre = np.asarray(joints)[:, 0, :2].mean(0)
            def turn(points):
                points = np.asarray(points).copy()
                flat = points.reshape(-1, 3)
                flat[:, :2] -= centre
                flat[:] = flat @ rotation.T
                flat[:, :2] += centre
                return flat.reshape(points.shape)
            skinned[row] = (None if vertices is None else turn(vertices),
                            faces, turn(joints))
            if vrm_state is not None:
                vrm_state[row]["heading"] = (rotation, centre)

    if align_position and len(skinned) > 1:
        shifts = align_row_positions(skinned)
        if vrm_state is not None:
            for row, shift in shifts.items():
                vrm_state[row]["position"] = shift

    renderer = pyrender.OffscreenRenderer(size, size)
    ground = _ground_mesh(spec)
    # The camera looks at the CENTROID of every row's root, and every panel uses
    # that one camera.  Following each arm's own root would re-centre each panel
    # on its own drift and hide exactly the difference between them -- that part
    # of the original rule stands.  What did not stand was following row 0
    # alone: it is the honest picture only while the other rows are in frame,
    # and they measurably were not (see ``_camera_pose``).  The centroid keeps
    # every row's offset from every other visible and keeps them all on screen;
    # ``spread`` below makes the second half of that a guarantee rather than a
    # hope.
    roots = [np.asarray(j)[:, 0, :2] for _, _, j in skinned]
    span = min(len(r) for r in roots)
    roots = np.stack([r[:span] for r in roots])            # [rows, frames, 2]
    centroid = roots.mean(0)
    # Edge-padded, not zero-padded.  ``np.convolve(..., mode="same")`` pads with
    # zeros, i.e. with the world origin, so the shot was dragged toward (0, 0)
    # over the last and first 22 frames -- measured 0.737 m off the true target
    # at the final frame of one clip against a 0.069 m median in the interior,
    # 59% of the half-frame width, and it displaced every panel at once.
    window = 45
    half = window // 2
    padded = np.pad(centroid, ((half, window - 1 - half), (0, 0)), mode="edge")
    kernel = np.ones(window) / window
    smoothed = np.stack([np.convolve(padded[:, axis], kernel, mode="valid")
                         for axis in (0, 1)], 1)
    spec["spread"] = float(np.linalg.norm(
        roots[:, None, :, :] - roots[None, :, :, :], axis=-1).max()) if len(roots) > 1 else 0.0
    # The vertical counterpart, over every row and every frame: how far above
    # the shared floor any joint gets.  p99 rather than max, so a single
    # reconstruction spike cannot shrink the whole clip.
    spec["reach"] = float(np.percentile(
        np.concatenate([np.asarray(j)[:, :, 2].reshape(-1) for _, _, j in skinned])
        - spec["floor"], 99))
    staging = pathlib.Path(tempfile.mkdtemp(prefix="avatar-"))
    try:
        import imageio.v2 as imageio

        for index in range(0, frames, stride):
            camera_pose = _camera_pose(spec, look_at=smoothed[min(index, len(smoothed) - 1)],
                                       view=view)
            panels = []
            for row, (vertices, faces, _) in enumerate(skinned):
                scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0],
                                       ambient_light=([0.62, 0.62, 0.64] if vrm_state is not None
                                       else [ambient, ambient, ambient + 0.03]))
                scene.add(pyrender.Mesh.from_trimesh(ground, smooth=False))
                if vrm_state is not None:
                    for verts, faces_, image, uvs in vrm_frame(vrm_state[row], index):
                        piece = trimesh.Trimesh(verts, faces_, process=False)
                        piece.apply_transform(_z_up_to_y_up())
                        if image is not None and uvs is not None:
                            piece.visual = trimesh.visual.TextureVisuals(
                                uv=uvs, image=image,
                                material=trimesh.visual.texture.SimpleMaterial(image=image))
                        scene.add(pyrender.Mesh.from_trimesh(piece, smooth=False))
                    scene.add(pyrender.PerspectiveCamera(yfov=FOV), pose=camera_pose)
                    scene.add(pyrender.DirectionalLight(
                        color=np.array([1.0, 0.97, 0.94]), intensity=2.1),
                        pose=camera_pose)
                    lift = np.array(camera_pose, copy=True)
                    lift[:3, 3] += np.array([-2.5, 2.0, 0.5])
                    scene.add(pyrender.DirectionalLight(
                        color=np.array([0.80, 0.86, 1.0]), intensity=1.0), pose=lift)
                    colour_buffer, _ = renderer.render(
                        scene, flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
                    panels.append(colour_buffer)
                    continue
                if uv is not None:
                    coords, corner_v, corner_t, image = uv
                    corners = vertices[index][corner_v.reshape(-1)]
                    mesh = trimesh.Trimesh(
                        corners, np.arange(len(corners)).reshape(-1, 3), process=False,
                        visual=trimesh.visual.TextureVisuals(
                            uv=coords[corner_t.reshape(-1)], image=image,
                            material=trimesh.visual.texture.SimpleMaterial(image=image)))
                else:
                    mesh = trimesh.Trimesh(vertices[index], faces, process=False)
                mesh.apply_transform(_z_up_to_y_up())
                if uv is not None:
                    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
                elif dressing is not None:
                    # Per-vertex colour, no material: the outfit IS the colour, so
                    # a baseColorFactor here would multiply it flat.
                    mesh.visual.vertex_colors = dressing
                    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
                else:
                    tint = (ROW_COLOURS[row % len(ROW_COLOURS)]
                            if style == "realistic" else base)
                    material = pyrender.MetallicRoughnessMaterial(
                        baseColorFactor=[*tint, 1.0], metallicFactor=0.0,
                        roughnessFactor=roughness, alphaMode="OPAQUE")
                    scene.add(pyrender.Mesh.from_trimesh(mesh, material=material,
                                                         smooth=True))
                scene.add(pyrender.PerspectiveCamera(yfov=FOV),
                          pose=camera_pose)
                key = pyrender.DirectionalLight(color=np.array([1.0, 0.96, 0.90]),
                                                intensity=key_power)
                scene.add(key, pose=camera_pose)
                fill = np.array(camera_pose, copy=True)
                fill[:3, 3] += np.array([-2.5, 2.0, 0.5])
                scene.add(pyrender.DirectionalLight(color=np.array([0.72, 0.80, 1.0]),
                                                    intensity=fill_power), pose=fill)
                colour_buffer, depth_buffer = renderer.render(
                    scene, flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
                if style == "toon":
                    colour_buffer = toon(colour_buffer, depth_buffer, base)
                panels.append(colour_buffer)
            imageio.imwrite(staging / "{:06d}.png".format(index // stride),
                            np.concatenate(panels, axis=1))
        _encode(staging, output, audio, len(titles), titles, size, stride)
    finally:
        renderer.delete()
        shutil.rmtree(staging, ignore_errors=True)
    return {"frames": frames // stride, "rows": len(titles), "floor": floor,
            "radius": spec["radius"],
            "style": (pathlib.Path(vrm).stem if vrm else
                      ("textured" if texture else style))
                     + (" · root locked" if lock_root else ""),
            "body": "{} / {}".format(gender or "SMPL male", betas_source)}


def _encode(staging, output, audio, columns, titles, size, stride):
    output = pathlib.Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    label = ",".join(
        "drawtext=text='{}':x={}:y=12:fontsize=17:fontcolor=0x33404d".format(
            title.replace(":", "\\:").replace("'", ""), index * size + 14)
        for index, title in enumerate(titles))
    argv = ["ffmpeg", "-y", "-v", "error",
            "-framerate", str(FPS / stride), "-i", str(staging / "%06d.png")]
    if audio is not None:
        argv += ["-i", str(audio)]
    argv += ["-vf", label, "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart"]
    if audio is not None:
        # 44.1 kHz stereo, not the source's 22 kHz mono.  The clip audio in this
        # corpus is mono at 22050 Hz, and an editor's built-in preview can play
        # such a track silently while every command-line probe reports it
        # present and loud -- which is exactly how this was first reported.
        # The +3 dB restores what ffmpeg's mono-to-stereo downmix removes to
        # preserve total power.
        # MP3, not AAC.  Measured 2026-08-29 by holding the video stream
        # byte-identical (-c:v copy) and varying only the audio codec: an
        # editor's built-in preview played AAC at 22 kHz mono and at 48 kHz
        # stereo silently, refused WebM/VP9 outright, and played MP3.  Every
        # command-line probe reports the AAC track present and loud, so this
        # is invisible to anything but a person pressing play.
        argv += ["-af", "volume=3dB", "-c:a", "libmp3lame", "-ar", "44100",
                 "-ac", "2", "-b:a", "160k", "-shortest"]
    argv += [str(output)]
    completed = subprocess.run(argv, capture_output=True, text=True)
    if completed.returncode != 0:
        raise SystemExit("ffmpeg failed: {}".format(completed.stderr[-800:]))


def write_source_record(output, rows, stats):
    """Write ``<output>.source.json`` naming WHICH ARM each panel came from.

    WHY.  On 2026-09-12 the operator pointed at two rendered clips and said one
    was much worse than ground truth and the other slightly better -- the single
    most valuable labelling this project gets, because the video is the only
    valid judge.  It could not be used: the render directory recorded nothing
    about which arm produced it, four arms had plausible timestamps, and the two
    candidate arms disagree about which of those clips is better.  A labelled
    pair with no provenance is a measurement thrown away.

    It is the same defect already in the log one level up -- a manifest that
    omitted the flag the arm was named after -- so this records the arm's own
    identity and not a description of it: the pickle path, and from the arm
    directory's manifest the two checkpoints and the sampling dict, which is
    what a rerun would be reconstructed from.
    """
    import hashlib

    record = {"output": str(output), "frames": stats.get("frames"),
              "style": stats.get("style"), "rows": []}
    for title, path in rows:
        source = pathlib.Path(path)
        entry = {"title": title, "path": str(source)}
        # A panel's source is EITHER a pickle OR a directory holding
        # atomic_motion_151.npy -- ground truth is always the latter, because
        # its eval pickle carries joints and no SMPL parameters.  The first
        # version of this function assumed a file and crashed on the ground
        # truth row, which took the render driver down with it under `set -e`
        # after the video had already been written.
        payload_file = source
        if source.is_dir():
            payload_file = source / "atomic_motion_151.npy"
            entry["kind"] = "converted-151d-directory"
        if payload_file.is_file():
            digest = hashlib.sha256()
            with open(payload_file, "rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
            entry["sha256"] = digest.hexdigest()
            entry["hashed"] = str(payload_file)
        manifest = (source if source.is_dir() else source.parent) / "manifest.json"
        if manifest.exists():
            try:
                payload = json.loads(manifest.read_text())
            except (ValueError, OSError):
                payload = None
            if payload:
                entry["arm"] = {
                    "manifest": str(manifest),
                    "output_dir": payload.get("output_dir"),
                    "planner_checkpoint": payload.get("planner_checkpoint"),
                    "completion_checkpoint": payload.get("completion_checkpoint"),
                    "data_root": (payload.get("dataset_provenance") or {}).get("data_root"),
                    "sampling": payload.get("sampling"),
                }
        record["rows"].append(entry)
    sidecar = pathlib.Path(str(output) + ".source.json")
    sidecar.write_text(json.dumps(record, indent=2))
    return sidecar


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--motion", action="append", required=True, metavar="TITLE:PATH",
                        help="repeatable; the FIRST is the reference whose floor "
                             "every other row stands on")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--audio", type=pathlib.Path, default=None)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--floor", type=float, default=None,
                        help="override the reference's floor; only for a clip "
                             "rendered alone against a known plane")
    parser.add_argument("--smpl", default=SMPL_MODEL)
    parser.add_argument("--gender", choices=("male", "female", "neutral"), default=None,
                        help="use SMPL-X with this body model; the source poses are "
                             "SMPL-X's own 21 body joints, so this is a return to the "
                             "original parameterisation rather than a retarget")
    parser.add_argument("--betas", action="store_true",
                        help="render this dancer's own build, fetched from the GVHMR "
                             "result; without it every arm is the model's mean body")
    parser.add_argument("--style", choices=tuple(STYLES), default="realistic")
    parser.add_argument("--texture", nargs="?", const=DEFAULT_TEXTURE, default=None,
                        help="SMPL UV texture; bare flag uses the vendored dancer")
    parser.add_argument("--smplx", action="store_true",
                        help="SMPL-X body (better hands and feet, cannot be textured)")
    parser.add_argument("--view", choices=("front", "three-quarter"), default="front")
    parser.add_argument("--vrm", default=None,
                        help="drive a VRM humanoid avatar instead of the SMPL body; "
                             "the avatar brings its own proportions, clothes and hair")
    parser.add_argument("--no-align-position", dest="align_position",
                        action="store_false",
                        help="leave every row at the absolute world position "
                             "its reconstruction happened to have. On by "
                             "default: that position is arbitrary (nothing "
                             "fixes where in the room a monocular "
                             "reconstruction stands), it is 0.877 m from "
                             "ground truth's on the T eval clips, and the "
                             "camera's spread compensation treats it as a "
                             "lateral offset, so a row displaced toward the "
                             "lens is drawn larger and crops. Row-to-row "
                             "TRAVEL is untouched either way")
    parser.add_argument("--no-align-heading", dest="align_heading",
                        action="store_false",
                        help="do not rotate each row onto the reference's median "
                             "heading; the panels are then watched from whatever "
                             "azimuth each happens to face")
    parser.add_argument("--fill", type=float, default=0.72,
                        help="share of the frame height the body fills; higher is closer")
    parser.add_argument("--lock-root", action="store_true",
                        help="put every row at the reference's horizontal position so "
                             "the panels compare as poses; this HIDES the root "
                             "translation defect, which the default view shows")
    args = parser.parse_args(argv)

    rows = []
    for entry in args.motion:
        if ":" not in entry:
            raise SystemExit("--motion wants TITLE:PATH, got {!r}".format(entry))
        title, path = entry.split(":", 1)
        rows.append((title, path))
    stats = render(rows, args.output, audio=args.audio, size=args.size,
                   stride=args.stride, floor=args.floor, model_path=args.smpl,
                   gender=args.gender, use_betas=args.betas, style=args.style,
                   texture=args.texture, family="smplx" if args.smplx else "smpl",
                   view=args.view, lock_root=args.lock_root, vrm=args.vrm, fill=args.fill,
                   align_heading=args.align_heading,
                   align_position=args.align_position)
    sidecar = write_source_record(args.output, rows, stats)
    print("{} row(s) x {} frame(s), {} style, body: {}, floor {:.3f} m -> {}"
          .format(stats["rows"], stats["frames"], stats["style"], stats["body"],
                  stats["floor"], args.output))
    print("source record -> {}".format(sidecar))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
