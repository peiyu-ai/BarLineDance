#!/usr/bin/env python3
"""Amplify a generated clip's motion, per segment, to the energy its own draft has.

THE MECHANISM THIS TARGETS, measured 2026-08-31 over 90 timebase-clean clips:
the completion outputs a near-generic energy level regardless of the song --
per-clip correlation between its energy and the ground truth's is **0.210**
(its coefficient of variation 0.22 against the ground truth's 0.32), so the
most energetic songs get the flattest dance relative to their target, and
"hits the beat" fails exactly there: with no stroke there is nothing to stop.
The operator's example clips sit at energy ratios 0.46-0.49.  Music-feature
normalization barely moves this (0.17-0.26), so the conditioning path does not
carry per-song intensity in any usable form.

The RETRIEVAL DRAFT, meanwhile, carries the right energy -- 1.031 of ground
truth on the same clips -- because it is real dance of the planned classes.
This tool moves that one scalar per segment from the draft to the output.

HOW, without breaking the body: amplitude is scaled in ROTATION space.  For
each joint, the segment's temporal mean rotation is the anchor and each frame's
deviation from it is scaled by the segment's gain in the tangent space
(quaternion log/exp), which preserves bone lengths by construction -- scaling
joint POSITIONS would not.  The root's horizontal deviations scale with the
same gain; height is left alone (it carries the floor).  Gains are clamped to
[1.0, --max-gain] (this tool only amplifies; where the output already moves
more than its draft, it is left alone) and cross-fade linearly over 5 frames at
segment boundaries so the gain schedule cannot itself add a splice.

WHAT MUST BE RE-MEASURED AFTERWARDS, every time: per-clip energy correlation
and the <0.6 tail (the point), but also jitter (amplification amplifies shake
too) and foot skate -- the standing columns exist so this trade is visible.
"""
import argparse
import json
import pathlib
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

FPS = 30.0


def _quaternions(axis_angle):
    """[T, J, 3] axis-angle -> [T, J, 4] unit quaternions (w, x, y, z)."""
    angle = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    half = angle / 2.0
    axis = np.where(angle > 1e-9, axis_angle / np.maximum(angle, 1e-9), 0.0)
    return np.concatenate([np.cos(half), axis * np.sin(half)], axis=-1)


def _axis_angle(quaternions):
    quaternions = quaternions / np.maximum(
        np.linalg.norm(quaternions, axis=-1, keepdims=True), 1e-12)
    w = np.clip(quaternions[..., :1], -1.0, 1.0)
    angle = 2.0 * np.arccos(np.abs(w))
    sign = np.where(w >= 0, 1.0, -1.0)
    sin_half = np.sqrt(np.maximum(1.0 - w * w, 1e-18))
    axis = sign * quaternions[..., 1:] / sin_half
    return axis * angle


def _multiply(a, b):
    aw, ax, ay, az = (a[..., i] for i in range(4))
    bw, bx, by, bz = (b[..., i] for i in range(4))
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def _conjugate(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def _mean_quaternion(quaternions):
    """Chordal mean with sign alignment -- adequate for within-segment spreads."""
    reference = quaternions[len(quaternions) // 2]
    aligned = quaternions * np.where(
        (quaternions * reference).sum(-1, keepdims=True) < 0, -1.0, 1.0)
    mean = aligned.mean(0)
    return mean / np.maximum(np.linalg.norm(mean, axis=-1, keepdims=True), 1e-12)


def exaggerate_rotations(axis_angle, gain):
    """Scale each joint's deviation from its temporal mean rotation by gain[t]."""
    quaternions = _quaternions(axis_angle)
    mean = _mean_quaternion(quaternions)
    deviation = _multiply(_conjugate(mean)[None], quaternions)
    log = _axis_angle(deviation)                    # tangent vectors
    scaled = log * gain[:, None, None]
    return _multiply(mean[None], _quaternions(scaled))


def joint_energy(full_pose):
    joints = np.asarray(full_pose, float)
    relative = joints - joints[:, :1, :]
    return float(np.linalg.norm(np.diff(relative, axis=0), axis=2).mean() * FPS)


def segments_of(labels):
    labels = np.asarray(labels)
    edges = np.flatnonzero(np.diff(labels)) + 1
    bounds = [0, *edges.tolist(), len(labels)]
    return [(bounds[i], bounds[i + 1], int(labels[bounds[i]]))
            for i in range(len(bounds) - 1)]


def gain_schedule(frames, spans, ramp=5):
    gain = np.ones(frames)
    for start, end, value in spans:
        gain[start:end] = value
    if ramp > 1:
        kernel = np.ones(ramp) / ramp
        padded = np.pad(gain, (ramp // 2, ramp - 1 - ramp // 2), mode="edge")
        gain = np.convolve(padded, kernel, mode="valid")
    return gain


def root_deviation(full_pose):
    ground = np.asarray(full_pose, float)[:, 0, :2]
    return float(np.linalg.norm(ground - ground.mean(0), axis=1).mean())


def transfer(payload, draft_pose, max_gain=1.6):
    labels = np.asarray(payload["atomic_labels"])
    output_pose = np.asarray(payload["full_pose"], float)
    frames = len(output_pose)
    spans = []
    for start, end, label in segments_of(labels[:frames]):
        if label == 0 or end - start < 10:
            continue
        target = joint_energy(draft_pose[start:end])
        actual = joint_energy(output_pose[start:end])
        if actual <= 1e-6 or target <= 1e-6:
            continue
        # Two gains per segment: the limbs' (root-relative joint energy) and the
        # root's own (horizontal deviation).  The first version applied the limb
        # gain to the root too, and pushed root travel -- already repaired to
        # 1.01 of ground truth by the robust normalizer -- to 1.27.  A fix that
        # overdrives an already-correct quantity is how the last round's
        # sample-steps arm was crowned; the columns exist so it is caught, and
        # it was.
        root_target = root_deviation(draft_pose[start:end])
        root_actual = root_deviation(output_pose[start:end])
        root_gain = (float(np.clip(root_target / root_actual, 1.0, max_gain))
                     if root_actual > 1e-6 and root_target > 1e-6 else 1.0)
        spans.append((start, end, float(np.clip(target / actual, 1.0, max_gain)),
                      root_gain))
    if not spans:
        return None
    gain = gain_schedule(frames, [(a, b, g) for a, b, g, _ in spans])
    gain_root = gain_schedule(frames, [(a, b, g) for a, b, _, g in spans])
    poses = np.asarray(payload["smpl_poses"], float).reshape(frames, -1, 3)
    # Root orientation (joint 0) carries the dancer's facing; scaling its
    # deviation swings the whole body and reads as staggering.  Body joints only.
    body = exaggerate_rotations(poses[:, 1:], gain)
    new_poses = poses.copy()
    new_poses[:, 1:] = _axis_angle(body) if body.shape[-1] == 4 else body
    trans = np.asarray(payload["smpl_trans"], float).copy()
    mean_xy = trans[:, :2].mean(0)
    trans[:, :2] = mean_xy + (trans[:, :2] - mean_xy) * gain_root[:, None]

    from vis import SMPLSkeleton
    full = SMPLSkeleton().forward(
        torch.tensor(new_poses, dtype=torch.float32).unsqueeze(0),
        torch.tensor(trans, dtype=torch.float32).unsqueeze(0))[0].numpy()
    out = dict(payload)
    out["smpl_poses"] = new_poses.reshape(frames, -1).astype(np.asarray(payload["smpl_poses"]).dtype)
    out["smpl_trans"] = trans.astype(np.asarray(payload["smpl_trans"]).dtype)
    out["full_pose"] = full.astype(np.asarray(payload["full_pose"]).dtype)
    out["dynamics_transfer"] = {"max_gain": max_gain,
                                "mean_gain": float(np.mean([s[2] for s in spans])),
                                "mean_root_gain": float(np.mean([s[3] for s in spans])),
                                "segments": len(spans)}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--clips", required=True)
    parser.add_argument("--data-root", default="/dev/shm/atomicdance-song-v5rekey/release_v2_robust")
    parser.add_argument("--max-gain", type=float, default=1.6)
    args = parser.parse_args()

    from infer_atomic import IndexedAtomicMotionLibrary, _source_safe_draft, decode_motion
    library = IndexedAtomicMotionLibrary(args.data_root)
    source = pathlib.Path(args.input)
    target = pathlib.Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    gains = []
    for line in open(args.clips):
        clip = line.strip()
        pkl = source / (clip + ".pkl")
        if not clip or not pkl.is_file():
            continue
        payload = pickle.load(open(pkl, "rb"))
        labels = torch.from_numpy(np.asarray(payload["atomic_labels"]).astype(np.int64))
        group = library.query_retrieval_group_id(clip)
        draft, _ = _source_safe_draft(library, labels, 151, group,
                                      recurrence_variety=True,
                                      variety_rng=np.random.default_rng(0))
        draft_pose = decode_motion(draft, library.normalizer_path)["full_pose"]
        result = transfer(payload, np.asarray(draft_pose, float), args.max_gain)
        out = result if result is not None else payload
        with open(target / (clip + ".pkl"), "wb") as handle:
            pickle.dump(out, handle)
        if result is not None:
            gains.append(result["dynamics_transfer"]["mean_gain"])
    print(json.dumps({"clips": len(gains), "mean_gain": float(np.mean(gains)) if gains else None}))


if __name__ == "__main__":
    main()
