#!/usr/bin/env python3
"""Motion beats and the pose descriptors the paper's M3 hangs off them.

The paper: "we use PoseScript to describe selected keyframes identified as
*motion beats* (local minima of segment-wise joint velocities)".  Motion beats
are where a movement momentarily settles -- the *signature pose* the VLM is
asked to name -- while the velocity between them is the *movement dynamics*.
Both halves of the paper's caption schema are therefore derived from this one
construction, so it lives on its own rather than inside either M3 variant.

A beat is a local minimum of mean joint speed, not merely a slow frame: a
segment that decelerates smoothly to a stop has one beat at the end, not a run
of them.  Minima are required to be separated (``--min-separation``) so a noisy
velocity trace cannot report a dozen beats inside a tenth of a second.

The pose descriptor at a beat is deliberately viewpoint- and scale-free:
joints are taken relative to the root, rotated so the hips face +x, and divided
by shoulder width.  Two dancers of different sizes filmed from different angles
performing the same shape must land in the same place, or in-group re-clustering
would be clustering camera angle and body size.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

FPS = 30.0
# SMPL joint indices, matching the rest of the repo.
ROOT, L_HIP, R_HIP, L_SHOULDER, R_SHOULDER = 0, 1, 2, 16, 17
L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 4, 5, 7, 8
L_FOOT, R_FOOT, HEAD, L_WRIST, R_WRIST = 10, 11, 15, 20, 21


def joint_speed(joints: np.ndarray, fps: float = FPS) -> np.ndarray:
    """[T,J,3] -> [T] mean joint speed, with the first frame carrying frame 1's."""
    joints = np.asarray(joints, dtype=np.float64)
    if len(joints) < 2:
        return np.zeros(len(joints))
    step = np.linalg.norm(np.diff(joints, axis=0), axis=-1).mean(axis=1) * fps
    return np.concatenate([step[:1], step])


def find_motion_beats(joints: np.ndarray, *, min_separation: int = 5,
                      max_beats: int = 4, smooth: int = 3,
                      prominence: float = 0.02) -> List[int]:
    """Frame indices of local minima in joint speed -- where movement settles.

    ``prominence`` is what separates a beat from arithmetic noise: a dip must
    sit at least this far below the segment's mean speed, relatively.  Without
    it, ``np.diff`` on an evenly-moving limb wobbles in the last ulp and a
    constant-speed pass-through reports a full quota of spurious beats.
    """
    speed = joint_speed(joints)
    if len(speed) < 3:
        return [0] if len(speed) else []
    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        speed = np.convolve(speed, kernel, mode="same")
    interior = np.arange(1, len(speed) - 1)
    # Strict on the entering side, loose on the leaving side: with ``<=`` both
    # ways a flat stretch reports *every* frame, which is the opposite of what
    # a beat means.  A genuine dip yields the first frame of its valley.
    is_minimum = (speed[1:-1] < speed[:-2]) & (speed[1:-1] <= speed[2:])
    reference = float(speed.mean())
    deep_enough = speed[1:-1] <= reference * (1.0 - prominence)
    minima = interior[is_minimum & deep_enough]
    if len(minima) == 0:
        # Nothing settles -- a constant or monotone segment.  Its slowest frame
        # is still the best available signature pose.
        return [int(speed.argmin())]
    # Prefer the slowest minima, then enforce separation so a jittery trace
    # cannot report a cluster of beats a few frames apart.
    chosen: List[int] = []
    for index in minima[np.argsort(speed[minima])]:
        if all(abs(int(index) - taken) >= min_separation for taken in chosen):
            chosen.append(int(index))
        if len(chosen) >= max_beats:
            break
    return sorted(chosen)


def canonical_pose(joints_at_frame: np.ndarray) -> np.ndarray:
    """[J,3] -> [J,3] root-centred, hip-facing-+x, shoulder-width-normalised.

    Without this the descriptor encodes where the camera was and how tall the
    dancer is, and in-group re-clustering would sort by those instead of by
    shape.
    """
    pose = np.asarray(joints_at_frame, dtype=np.float64) - joints_at_frame[ROOT]
    hips = pose[L_HIP] - pose[R_HIP]
    angle = np.arctan2(hips[1], hips[0])
    cos, sin = np.cos(-angle), np.sin(-angle)
    rotation = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    pose = pose @ rotation.T
    scale = np.linalg.norm(pose[L_SHOULDER] - pose[R_SHOULDER])
    return pose / scale if scale > 1e-6 else pose


def segment_descriptor(joints: np.ndarray, *, max_beats: int = 4) -> Dict[str, np.ndarray]:
    """Signature poses at the beats, plus the dynamics between them."""
    joints = np.asarray(joints, dtype=np.float64)
    beats = find_motion_beats(joints, max_beats=max_beats)
    poses = np.stack([canonical_pose(joints[b]) for b in beats])
    speed = joint_speed(joints)
    displacement = joints[-1, ROOT] - joints[0, ROOT]
    dynamics = np.array([
        speed.mean(), speed.max(), speed.std(),
        float(len(joints)) / FPS,
        np.linalg.norm(displacement[:2]),   # travel across the floor
        displacement[2],                    # net rise or fall
    ])
    return {"beats": np.asarray(beats, dtype=np.int64), "poses": poses, "dynamics": dynamics}


def descriptor_vector(joints: np.ndarray, *, max_beats: int = 4,
                      joints_used: int = 22) -> np.ndarray:
    """Flatten a segment into one fixed-width vector for sub-clustering.

    Beats are padded by repeating the last one rather than with zeros: a zero
    pose is a *pose* -- a collapsed skeleton at the origin -- and would pull
    short segments together into a spurious sub-prototype of their own.
    """
    parts = segment_descriptor(joints, max_beats=max_beats)
    poses = parts["poses"][:, :joints_used, :]
    if len(poses) < max_beats:
        poses = np.concatenate([poses, np.repeat(poses[-1:], max_beats - len(poses), axis=0)])
    return np.concatenate([poses.reshape(-1), parts["dynamics"]])


# Thresholds for the measures that have no landmark to compare against.  They
# were read off the wild corpus (40 sequences, 532 canonical poses) at the 85th
# percentile, so each clause fires on a clear minority of poses; a clause that
# fires on nearly every pose carries no information for the VLM to use.  Heights
# are not in this table on purpose -- head, shoulder and hip are right there in
# the same pose and are better references than any constant.
SIDEWAYS_REACH = 1.25    # |x| of the wrist, in shoulder widths
FORWARD_REACH = 0.95     # how far in front of the pelvis the wrist is
WIDE_STANCE = 2.10       # lateral distance between the feet
CLOSED_STANCE = 0.60
FOOT_LIFT = 0.35         # height difference between the two feet
CROUCH = 0.99            # pelvis height above the lower foot, in leg lengths


def describe_pose(pose: np.ndarray) -> str:
    """A compact PoseScript-style sentence for a canonical pose.

    PoseScript proper is a released model; this is the rule-based fallback that
    keeps the caption path runnable with no external dependency, and it is
    labelled as such wherever it is recorded so it is never mistaken for the
    real thing.

    Every height is stated against a landmark of the same body -- a wrist is
    "overhead" when it is above that dancer's head, not above a fixed number --
    which is the only formulation that survives different body proportions.
    Laterality reads the **x** axis: ``canonical_pose`` turns the hips onto +x,
    so x is the left-right axis and y is front-back.  Getting that pair the
    wrong way round produces sentences that are fluent and false, which is
    worse than none at all, so the axis is asserted by test.
    """
    parts = []
    head_z, hip_z = pose[HEAD][2], max(pose[L_HIP][2], pose[R_HIP][2])
    for name, wrist, shoulder, sign in (("left", pose[L_WRIST], pose[L_SHOULDER], 1.0),
                                        ("right", pose[R_WRIST], pose[R_SHOULDER], -1.0)):
        chest_z = 0.5 * (hip_z + shoulder[2])
        if wrist[2] > head_z:
            parts.append("{} arm raised overhead".format(name))
        elif sign * wrist[0] > SIDEWAYS_REACH and wrist[2] > hip_z:
            parts.append("{} arm extended sideways".format(name))
        elif wrist[2] > shoulder[2]:
            parts.append("{} arm lifted above the shoulder".format(name))
        elif -wrist[1] > FORWARD_REACH:
            parts.append("{} arm reaching forward".format(name))
        elif wrist[2] > chest_z:
            parts.append("{} arm at chest height".format(name))
        else:
            parts.append("{} arm low".format(name))

    left_foot, right_foot = pose[L_FOOT], pose[R_FOOT]
    stance = abs(left_foot[0] - right_foot[0])
    if stance > WIDE_STANCE:
        parts.append("feet wide apart")
    elif stance < CLOSED_STANCE:
        parts.append("feet together")
    if abs(left_foot[2] - right_foot[2]) > FOOT_LIFT:
        parts.append("one foot lifted")
    leg = (np.linalg.norm(pose[L_HIP] - pose[L_KNEE])
           + np.linalg.norm(pose[L_KNEE] - pose[L_ANKLE]))
    if leg > 1e-6 and -min(left_foot[2], right_foot[2]) / leg < CROUCH:
        parts.append("knees bent low")
    return ", ".join(parts)
