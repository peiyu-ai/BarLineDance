#!/usr/bin/env python3
"""Drive a VRM humanoid avatar with this repository's SMPL motion.

Why a VRM and not a texture
---------------------------
Every appearance option tried before this one repainted the *same* SMPL mesh, so
the figure kept SMPL's proportions no matter what it wore -- which was the
reviewer's actual objection.  A VRM carries its own geometry, its own clothes
and hair as real surfaces, and its own proportions.

Why VRM specifically, out of the character formats
--------------------------------------------------
It is glTF 2.0 with a **specified** humanoid bone vocabulary (``VRMC_vrm``), so
the joint map below is written once against a standard rather than against one
artist's naming.  Measured 2026-08-29 against the official sample avatar: 22 of
SMPL's 24 joints map one-to-one, and the two that do not are SMPL's hand joints,
which this corpus holds at identity anyway (max |axis-angle| 4.7e-03).  A
Mixamo-style rig would have needed the same work per character.

What is retargeted, and what is deliberately not
------------------------------------------------
**Rotations only, plus the root translation.**  Nothing is smoothed, retimed, or
re-posed: bone ``b`` receives exactly the global rotation SMPL joint ``j``
produced, taken through the rest-pose alignment below.  The avatar's secondary
bones (``J_Sec_*`` for skirt and hair, ``J_Roll_``/``J_Aim_`` twist helpers) keep
their rest-local transforms and simply follow their parents; VRM's spring-bone
and constraint extensions are **not** evaluated, so hair and cloth are rigid.
That is a visible limitation and it is stated rather than hidden -- a swinging
skirt would be motion this pipeline did not generate.

**The rest-pose alignment is the whole difficulty.**  SMPL's template stands in
a shallow A-pose and a VRM stands in a T-pose, so handing SMPL's rotations
straight to VRM bones would leave the arms permanently 40 degrees out.  For each
mapped bone the alignment ``A_b`` is the minimal rotation carrying the VRM rest
bone direction onto the SMPL rest bone direction, computed from the two
skeletons' own rest geometry rather than tuned by eye:

    G_new(b) = R_smpl_global(j) @ A_b

so when SMPL sits in its own rest, every bone points where the VRM's rest points
it, and any SMPL rotation away from rest turns the VRM bone by the same amount
about the same world axis.

**Scale.**  The avatar and the corpus are both in metres but not the same
height, so the root translation is scaled by the ratio of the two skeletons'
hip-to-head span.  Without it a 1.4 m avatar would travel a 1.7 m dancer's
strides.
"""

from __future__ import annotations

import pathlib
import struct
from typing import Dict, Optional, Tuple

import numpy as np

# SMPL joint index -> VRM humanoid bone.  SMPL's 22/23 (the hands) have no VRM
# counterpart and are identity in this corpus.
SMPL_TO_VRM = {
    0: "J_Bip_C_Hips",
    1: "J_Bip_L_UpperLeg", 2: "J_Bip_R_UpperLeg",
    3: "J_Bip_C_Spine",
    4: "J_Bip_L_LowerLeg", 5: "J_Bip_R_LowerLeg",
    6: "J_Bip_C_Chest",
    7: "J_Bip_L_Foot", 8: "J_Bip_R_Foot",
    9: "J_Bip_C_UpperChest",
    10: "J_Bip_L_ToeBase", 11: "J_Bip_R_ToeBase",
    12: "J_Bip_C_Neck",
    13: "J_Bip_L_Shoulder", 14: "J_Bip_R_Shoulder",
    15: "J_Bip_C_Head",
    16: "J_Bip_L_UpperArm", 17: "J_Bip_R_UpperArm",
    18: "J_Bip_L_LowerArm", 19: "J_Bip_R_LowerArm",
    20: "J_Bip_L_Hand", 21: "J_Bip_R_Hand",
}
# SMPL's kinematic tree, the same list tools/render_dance_video.py carries.
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17,
                18, 19, 20, 21]
# This repository's motion is z-up (``convert_gvhmr_result``'s output contract:
# "EDGE/AtomicDance z-up") and glTF -- so every VRM -- is y-up.  Mixing the two
# is not a small error: the first retarget produced a dancer lying on her back,
# because the rest directions were compared across frames.  Everything from the
# motion side is brought into the avatar's frame by this basis change before it
# touches a bone.
Z_UP_TO_Y_UP = np.array([[1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0],
                         [0.0, -1.0, 0.0]])


def to_y_up_rotation(rotation):
    """A SMPL global rotation, with only its OUTPUT side brought to y-up.

    One-sided on purpose.  SMPL's template is y-up already -- ``v_template`` and
    therefore the rest joints this module aligns against -- and z-up appears only
    because this corpus's root orientation carries the conversion
    (``convert_gvhmr_result``: "EDGE/AtomicDance z-up; +90 degrees about X").  A
    global rotation here maps a y-up template direction to a z-up world
    direction, so only the world side needs turning.  The two-sided form
    ``C R C^T`` was the first version and it laid the avatar flat on her back:
    it converted the template side too, which was never in z-up.
    """
    return Z_UP_TO_Y_UP @ np.asarray(rotation, float)


def to_y_up_point(point):
    return np.asarray(point, float) @ Z_UP_TO_Y_UP.T


def _quat_to_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _align(source, target):
    """Minimal rotation carrying unit vector ``source`` onto ``target``."""
    source = np.asarray(source, float)
    target = np.asarray(target, float)
    ns, nt = np.linalg.norm(source), np.linalg.norm(target)
    if ns < 1e-9 or nt < 1e-9:
        return np.eye(3)
    source, target = source / ns, target / nt
    axis = np.cross(source, target)
    sin = np.linalg.norm(axis)
    cos = float(np.dot(source, target))
    if sin < 1e-9:
        return np.eye(3) if cos > 0 else -np.eye(3)
    axis = axis / sin
    cross = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
    angle = np.arctan2(sin, cos)
    return (np.eye(3) + np.sin(angle) * cross
            + (1 - np.cos(angle)) * (cross @ cross))


def _slerp_to_identity(rotation, weight):
    """Scale a rotation towards identity -- the constraint's ``weight``."""
    angle = np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-9:
        return np.eye(3)
    vector = np.array([rotation[2, 1] - rotation[1, 2],
                       rotation[0, 2] - rotation[2, 0],
                       rotation[1, 0] - rotation[0, 1]]) / (2 * np.sin(angle))
    return axis_angle_to_matrix(vector * angle * weight)


def axis_angle_to_matrix(vectors):
    """``[..., 3]`` axis-angle to ``[..., 3, 3]`` rotation matrices."""
    vectors = np.asarray(vectors, float)
    angle = np.linalg.norm(vectors, axis=-1, keepdims=True)
    axis = np.where(angle > 1e-12, vectors / np.maximum(angle, 1e-12), 0.0)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = np.zeros_like(x)
    cross = np.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1)
    cross = cross.reshape(vectors.shape[:-1] + (3, 3))
    eye = np.broadcast_to(np.eye(3), cross.shape)
    sin = np.sin(angle)[..., None]
    cos = np.cos(angle)[..., None]
    return eye + sin * cross + (1 - cos) * (cross @ cross)


class VRMAvatar:
    """A VRM read for skinning: hierarchy, rest pose, skins and mesh data."""

    def __init__(self, path):
        from pygltflib import GLTF2

        self.path = pathlib.Path(path)
        self.gltf = GLTF2().load(str(self.path))
        self._blob = self.gltf.binary_blob()
        self.names = {i: (n.name or "") for i, n in enumerate(self.gltf.nodes)}
        self.by_name = {name: i for i, name in self.names.items() if name}
        self.parent = {}
        for index, node in enumerate(self.gltf.nodes):
            for child in (node.children or []):
                self.parent[child] = index
        self.local_rest = {i: self._local(i) for i in range(len(self.gltf.nodes))}
        self.constraints = self._constraints()
        self.order = self._topological()
        self.global_rest = self._forward(self.local_rest)

    AIM_AXES = {"PositiveX": (1, 0, 0), "NegativeX": (-1, 0, 0),
                "PositiveY": (0, 1, 0), "NegativeY": (0, -1, 0),
                "PositiveZ": (0, 0, 1), "NegativeZ": (0, 0, -1)}
    ROLL_AXES = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}

    def _constraints(self):
        """``VRMC_node_constraint`` per node -- what drives the clothing.

        Without these the sleeves stay aimed where the artist left them while
        the arm swings away, which renders as two flat plates at the shoulders.
        They are not decoration: in this avatar the sleeve, the skirt panels and
        the upper-arm twist are all constraint-driven, and the bones that carry
        them hang off the *shoulder*, not off the arm, so inheriting the parent
        is not enough.
        """
        out = {}
        for index, node in enumerate(self.gltf.nodes):
            block = ((getattr(node, "extensions", None) or {})
                     .get("VRMC_node_constraint") or {}).get("constraint")
            if block:
                out[index] = block
        return out

    # ---- glTF plumbing -------------------------------------------------
    def _local(self, index):
        node = self.gltf.nodes[index]
        if node.matrix:
            return np.asarray(node.matrix, float).reshape(4, 4).T
        out = np.eye(4)
        if node.rotation:
            out[:3, :3] = _quat_to_matrix(node.rotation)
        if node.scale:
            out[:3, :3] = out[:3, :3] @ np.diag(node.scale)
        if node.translation:
            out[:3, 3] = node.translation
        return out

    def _topological(self):
        order, seen = [], set()

        def walk(index):
            if index in seen:
                return
            seen.add(index)
            order.append(index)
            for child in (self.gltf.nodes[index].children or []):
                walk(child)

        roots = [i for i in range(len(self.gltf.nodes)) if i not in self.parent]
        for root in roots:
            walk(root)
        return order

    def _forward(self, local):
        out = {}
        for index in self.order:
            parent = self.parent.get(index)
            out[index] = local[index] if parent is None else out[parent] @ local[index]
        return out

    def accessor(self, index):
        """Numpy view of one accessor, following its buffer view."""
        acc = self.gltf.accessors[index]
        view = self.gltf.bufferViews[acc.bufferView]
        dtype = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16,
                 5125: np.uint32, 5126: np.float32}[acc.componentType]
        count = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[acc.type]
        start = (view.byteOffset or 0) + (acc.byteOffset or 0)
        width = np.dtype(dtype).itemsize * count
        stride = view.byteStride or width
        raw = self._blob[start:start + stride * acc.count]
        if stride == width:
            data = np.frombuffer(raw, dtype=dtype, count=acc.count * count)
        else:
            rows = np.frombuffer(raw, dtype=np.uint8).reshape(acc.count, stride)
            data = np.frombuffer(rows[:, :width].tobytes(), dtype=dtype)
        return data.reshape(acc.count, count)

    def material_image(self, material_index):
        """The base-colour image of one glTF material, or ``None``.

        Without this the avatar renders as untextured grey geometry -- correct
        in shape and useless as the thing it was chosen for.  The pixels live in
        a buffer view like everything else, so they are decoded from the same
        blob rather than from a file beside the model.
        """
        import io as _io

        from PIL import Image

        if material_index is None or material_index >= len(self.gltf.materials):
            return None
        pbr = self.gltf.materials[material_index].pbrMetallicRoughness
        if pbr is None or pbr.baseColorTexture is None:
            return None
        texture = self.gltf.textures[pbr.baseColorTexture.index]
        if texture.source is None:
            return None
        image = self.gltf.images[texture.source]
        if image.bufferView is None:
            return None
        view = self.gltf.bufferViews[image.bufferView]
        start = view.byteOffset or 0
        raw = self._blob[start:start + view.byteLength]
        return Image.open(_io.BytesIO(raw)).convert("RGBA")

    def skinned_primitives(self):
        """``[(positions, indices, joints, weights, skin, material)]``."""
        out = []
        for node in self.gltf.nodes:
            if node.skin is None or node.mesh is None:
                continue
            skin = self.gltf.skins[node.skin]
            inverse = self.accessor(skin.inverseBindMatrices).reshape(-1, 4, 4)
            inverse = np.transpose(inverse, (0, 2, 1))
            for prim in self.gltf.meshes[node.mesh].primitives:
                attrs = prim.attributes
                out.append({
                    "positions": self.accessor(attrs.POSITION).astype(np.float64),
                    "indices": self.accessor(prim.indices).reshape(-1).astype(np.int64),
                    "joints": self.accessor(attrs.JOINTS_0).astype(np.int64),
                    "weights": self.accessor(attrs.WEIGHTS_0).astype(np.float64),
                    "uv": (self.accessor(attrs.TEXCOORD_0).astype(np.float64)
                           if attrs.TEXCOORD_0 is not None else None),
                    "skin_joints": np.asarray(skin.joints, np.int64),
                    "inverse_bind": inverse,
                    "material": prim.material,
                })
        return out

    # ---- retargeting ---------------------------------------------------
    def alignment(self, smpl_rest_joints, align_directions=False):
        """``{vrm node -> A_b}``: how a SMPL rotation is carried onto a VRM bone.

        **Default is no direction alignment, and that is a measurement rather
        than a simplification.**  The alignment existed to absorb a supposed
        A-pose/T-pose difference, but SMPL's template arms are already nearly
        horizontal: measured 2026-08-29 against this avatar, the upper arm
        differs by 6.8 degrees and the forearm by 2.2, while the *large*
        disagreements are pelvis 24.2, chest 37.4, neck 31.8 and clavicle 28.8 --
        every one of them a place where the two skeletons put their joints
        differently rather than a place where the two rest POSES differ.
        Aligning those bent the avatar's own shoulders up and out of its rest
        silhouette, which is exactly what the reviewer saw.  Keeping the rest
        orientation means the avatar stands the way its author drew it and only
        the motion's angles move it.

        ``align_directions=True`` restores the old behaviour for a skeleton whose
        rest pose really does differ.
        """
        smpl_children = {}
        for joint, parent in enumerate(SMPL_PARENTS):
            if parent >= 0:
                smpl_children.setdefault(parent, []).append(joint)
        out = {}
        for joint, bone in SMPL_TO_VRM.items():
            node = self.by_name.get(bone)
            if node is None:
                continue
            child_joint = None
            for candidate in smpl_children.get(joint, []):
                if candidate in SMPL_TO_VRM and SMPL_TO_VRM[candidate] in self.by_name:
                    child_joint = candidate
                    break
            rest_rotation = self.global_rest[node][:3, :3]
            if not align_directions:
                out[node] = rest_rotation
                continue
            if child_joint is None:
                # An end bone -- head, hand, toe -- has no direction to align, so
                # it keeps the avatar's own rest orientation.
                out[node] = rest_rotation
                continue
            child_node = self.by_name[SMPL_TO_VRM[child_joint]]
            smpl_dir = smpl_rest_joints[child_joint] - smpl_rest_joints[joint]
            vrm_dir = (self.global_rest[child_node][:3, 3]
                       - self.global_rest[node][:3, 3])
            # **The rest orientation has to survive the alignment.**  Two earlier
            # versions dropped it -- one used identity for end bones, one replaced
            # the whole orientation with the direction alignment -- and both left
            # the head pitched forward through every frame, because at SMPL's own
            # rest the bone then sits at the alignment instead of at the avatar's
            # rest.  Composing on the left is the version that satisfies both
            # requirements at once: with SMPL at rest the bone points where SMPL's
            # rest points it, and every SMPL rotation turns it by the same amount
            # about the same world axis.
            out[node] = _align(vrm_dir, smpl_dir) @ rest_rotation
        return out

    def pose(self, smpl_global_rotation, root_translation, align, scale):
        """New global transforms for every node, from one frame of SMPL motion.

        The hips are placed directly -- rotation from SMPL joint 0, translation
        from the scaled SMPL root -- and everything else is forward kinematics
        from there.  A mapped bone takes SMPL's global rotation and keeps its own
        rest offset from its parent, so the avatar's proportions are the
        avatar's and only the angles come from the motion.
        """
        target = {}
        for joint, bone in SMPL_TO_VRM.items():
            node = self.by_name.get(bone)
            if node is not None:
                target[node] = smpl_global_rotation[joint] @ align[node]
        hips = self.by_name["J_Bip_C_Hips"]
        # **Two passes, because 8 of this avatar's 14 constraints name a source
        # that comes LATER in the hierarchy** -- including both sleeve aims,
        # which point at the forearm.  A single pass finds the source
        # uncomputed and skips, and the first version skipped *silently*: the
        # sleeves stayed aimed where the artist left them and rendered as two
        # flat plates at the shoulders, with nothing anywhere saying a
        # constraint had been dropped.  Pass one resolves positions with the
        # constraints off; pass two applies them against those positions.
        reference = self._walk(target, root_translation, scale, hips, None)
        return self._walk(target, root_translation, scale, hips, reference)

    def _walk(self, target, root_translation, scale, hips, reference):
        globals_, local_delta = {}, {}
        self.constraints_skipped = 0
        for index in self.order:
            parent = self.parent.get(index)
            if index == hips:
                out = np.eye(4)
                out[:3, :3] = target[index]
                out[:3, 3] = np.asarray(root_translation, float) * scale
            elif parent is None or parent not in globals_:
                out = self.local_rest[index].copy()
            else:
                base = globals_[parent]
                rest_offset = self.local_rest[index][:3, 3]
                out = np.eye(4)
                out[:3, :3] = (target[index] if index in target
                               else base[:3, :3] @ self.local_rest[index][:3, :3])
                out[:3, 3] = base[:3, 3] + base[:3, :3] @ rest_offset
            globals_[index] = out
            constraint = self.constraints.get(index)
            if constraint and reference is not None:
                self._apply_constraint(index, constraint, globals_, local_delta,
                                       reference)
            parent_rotation = (np.eye(3) if self.parent.get(index) is None
                               else globals_[self.parent[index]][:3, :3])
            local_delta[index] = (self.local_rest[index][:3, :3].T
                                  @ parent_rotation.T @ globals_[index][:3, :3])
        return globals_

    def _apply_constraint(self, index, constraint, globals_, local_delta, reference):
        if "aim" in constraint:
            aim = constraint["aim"]
            source = aim.get("source")
            if source is None or source not in reference:
                self.constraints_skipped += 1
                return
            axis = np.asarray(self.AIM_AXES[aim.get("aimAxis", "PositiveX")], float)
            here = globals_[index][:3, 3]
            towards = reference[source][:3, 3] - here
            if np.linalg.norm(towards) < 1e-9:
                return
            current = globals_[index][:3, :3] @ axis
            turn = _align(current, towards)
            weight = float(aim.get("weight", 1.0))
            if weight < 1.0:
                turn = _slerp_to_identity(turn, weight)
            globals_[index] = globals_[index].copy()
            globals_[index][:3, :3] = turn @ globals_[index][:3, :3]
        elif "roll" in constraint:
            roll = constraint["roll"]
            source = roll.get("source")
            if source is None or source not in local_delta:
                self.constraints_skipped += 1
                return
            axis = np.asarray(self.ROLL_AXES[roll.get("rollAxis", "X")], float)
            delta = local_delta[source]
            angle = float(np.arctan2(
                np.dot(np.array([delta[2, 1] - delta[1, 2],
                                 delta[0, 2] - delta[2, 0],
                                 delta[1, 0] - delta[0, 1]]) / 2.0, axis),
                (np.trace(delta) - 1.0) / 2.0))
            angle *= float(roll.get("weight", 1.0))
            globals_[index] = globals_[index].copy()
            globals_[index][:3, :3] = (globals_[index][:3, :3]
                                       @ axis_angle_to_matrix(axis * angle))


def skin_vertices(primitive, globals_):
    """Linear blend skinning for one primitive, given new global transforms."""
    joints = primitive["skin_joints"]
    matrices = np.stack([globals_[int(j)] @ primitive["inverse_bind"][k]
                         for k, j in enumerate(joints)])
    positions = primitive["positions"]
    homogeneous = np.concatenate([positions, np.ones((len(positions), 1))], 1)
    out = np.zeros((len(positions), 3))
    for corner in range(primitive["joints"].shape[1]):
        index = primitive["joints"][:, corner]
        weight = primitive["weights"][:, corner]
        if not np.any(weight):
            continue
        transformed = np.einsum("nij,nj->ni", matrices[index], homogeneous)
        out += weight[:, None] * transformed[:, :3]
    return out
