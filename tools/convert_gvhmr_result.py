"""Convert one GVHMR extraction to the unnormalised 151-D representation.

Bridges ``tools/run_gvhmr_extract.py`` output to the AtomicDance corpus format,
reusing the numerically verified pieces of ``tools/preprocess_wild_3d.py``: the
same y-up -> z-up rotation, the same hardcoded neutral-SMPL forward kinematics
for contacts, and the same 6-D convention pinned against scipy elsewhere.

Two representational facts are handled explicitly rather than silently:

* GVHMR predicts **SMPL-X** body parameters: ``global_orient [T,3]`` plus
  ``body_pose [T,63]`` -- 21 body joints.  The 151-D contract wants SMPL's 24
  joints.  Joints 0..21 correspond one-to-one; SMPL's joints 22/23 are the
  hands, which SMPL-X moves into its separate hand articulation.  They are set
  to the identity rotation and ``hand_joints_identity: true`` is recorded --
  downstream consumers can weight or mask them, but they can never mistake
  zeroed wrists for observed ones.

* GVHMR's world frame is gravity-aligned y-up ('ay'), the same convention as
  WHAM's, so the established ``+90 deg about X`` conversion applies unchanged
  (their equality was verified against GVHMR's own ``tsf_axisangle`` table).

Outputs the same per-clip directory layout the WHAM converter produces, so
``preprocess_wild_3d.py validate`` and the downstream manifest tooling apply
as-is:

    atomic_motion_151.npy   [T,151]  4 contacts + 3 root + 24 x rot6d, z-up
    pose_axis_angle_z_up.npy, rotation6d_z_up.npy, root_translation_z_up.npy
    contacts.npy, frame_ids.npy, quality.json, metadata.json
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

# Invoked as `python tools/convert_gvhmr_result.py`, sys.path[0] is tools/,
# not the repo root; without this line the sibling import below only works
# when the ambient PYTHONPATH happens to include the CWD.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.preprocess_wild_3d import (
    Y_UP_TO_EDGE_Z_UP,
    contact_labels,
    matrices_to_rotation6d,
    motion_quality_summary,
)

FPS = 30.0
SMPL_JOINTS = 24
SMPLX_BODY_JOINTS = 21


def smpl24_from_smplx(global_orient, body_pose):
    """[T,3]+[T,63] SMPL-X body params -> [T,24,3] SMPL axis-angle."""
    frames = len(global_orient)
    body = np.asarray(body_pose, dtype=np.float64).reshape(frames, SMPLX_BODY_JOINTS, 3)
    pose = np.zeros((frames, SMPL_JOINTS, 3), dtype=np.float64)
    pose[:, 0] = np.asarray(global_orient, dtype=np.float64)
    pose[:, 1 : 1 + SMPLX_BODY_JOINTS] = body
    # Joints 22/23 (hands) stay identity; recorded in metadata, never implied.
    return pose


def _camera_representation_suffix(kind):
    if kind == "SimpleVO":
        return " + SimpleVO c2w [T,4,4]"
    if kind == "DPVO":
        return " + DPVO traj [T,7] and derived w2c [T,4,4]"
    return ""


def _dpvo_traj_to_w2c(traj):
    """[T,7] (tx,ty,tz,qx,qy,qz,qw) -> [T,4,4] world-to-camera.

    Exactly GVHMR's own reading of the same array (``load_data_dict`` takes
    ``traj[:, [6,3,4,5]]`` as (w,x,y,z), builds the rotation and transposes it),
    written out as full matrices so the audit asset needs no reader-side
    convention knowledge.  scipy's quaternion order is (x,y,z,w), which is
    ``traj[:, 3:7]`` unpermuted.  Translation stays up to scale, as monocular
    VO leaves it.
    """
    traj = np.asarray(traj, dtype=np.float64)
    rotation_c2w = Rotation.from_quat(traj[:, 3:7]).as_matrix()
    rotation_w2c = rotation_c2w.transpose(0, 2, 1)
    w2c = np.zeros((len(traj), 4, 4), dtype=np.float64)
    w2c[:, :3, :3] = rotation_w2c
    w2c[:, :3, 3] = -np.einsum("tij,tj->ti", rotation_w2c, traj[:, :3])
    w2c[:, 3, 3] = 1.0
    return w2c


def convert(result_path, output_dir, fps_in):
    payload = torch.load(result_path, map_location="cpu")
    params = payload["smpl_params_global"]
    global_orient = params["global_orient"].numpy()
    body_pose = params["body_pose"].numpy()
    transl = params["transl"].numpy()
    frames = len(transl)

    if fps_in == 30.0:
        keep = np.arange(frames)
    elif fps_in == 60.0:
        keep = np.arange(0, frames, 2)
    else:
        raise SystemExit(
            "unsupported source fps {}; expected 30 or 60".format(fps_in)
        )

    pose_y_up = smpl24_from_smplx(global_orient[keep], body_pose[keep])
    trans_y_up = np.asarray(transl[keep], dtype=np.float64)

    # Root orientation is global: the coordinate change applies on its left.
    rotations = Rotation.from_rotvec(pose_y_up.reshape(-1, 3)).as_matrix()
    rotations = rotations.reshape(len(keep), SMPL_JOINTS, 3, 3)
    rotations[:, 0] = np.einsum("ij,tjk->tik", Y_UP_TO_EDGE_Z_UP, rotations[:, 0])
    trans_z_up = np.einsum("ij,tj->ti", Y_UP_TO_EDGE_Z_UP, trans_y_up)
    pose_z_up = (
        Rotation.from_matrix(rotations.reshape(-1, 3, 3))
        .as_rotvec()
        .reshape(len(keep), SMPL_JOINTS, 3)
    )

    contacts = contact_labels(rotations, trans_z_up)
    # Same contract as the WHAM converter: a frame-id gap makes the one-frame
    # contact displacement meaningless, so zero it and mark it invalid rather
    # than let it look observed.  (GVHMR predicts every video frame, so for
    # 30 fps sources this mask is all-true; the logic still guards any future
    # source whose kept frames are not consecutive.)
    contact_valid_mask = np.ones_like(contacts, dtype=bool)
    if len(keep) > 1:
        gap_indices = np.flatnonzero(np.diff(keep) != 1)
        if len(gap_indices):
            contacts[gap_indices] = 0.0
            contact_valid_mask[gap_indices] = False
    rot6d = matrices_to_rotation6d(rotations)
    motion_151 = np.concatenate(
        [
            contacts.astype(np.float32),
            trans_z_up.astype(np.float32),
            rot6d.reshape(len(keep), -1).astype(np.float32),
        ],
        axis=1,
    )
    if motion_151.shape[1] != 151:
        raise AssertionError("assembled {} dims".format(motion_151.shape[1]))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "atomic_motion_151.npy", motion_151)
    np.save(output_dir / "pose_axis_angle_z_up.npy", pose_z_up.astype(np.float32))
    np.save(output_dir / "rotation6d_z_up.npy", rot6d.astype(np.float32))
    np.save(output_dir / "root_translation_z_up.npy", trans_z_up.astype(np.float32))
    np.save(output_dir / "contacts.npy", contacts.astype(np.float32))
    np.save(output_dir / "contact_valid_mask.npy", contact_valid_mask)
    np.save(output_dir / "frame_ids.npy", keep.astype(np.int64))

    # Camera audit asset, mirroring the WHAM converter's purpose: retain the
    # raw per-frame intrinsics estimate and, when visual odometry ran, the raw
    # SimpleVO camera-to-world trajectory in its own unregistered gauge.  It
    # is evidence for later audits, never an input to body motion.
    camera = {
        "intrinsics_K_fullimg": np.asarray(
            payload["K_fullimg"], dtype=np.float64)[keep],
        "camera_frame_ids": keep.astype(np.int64),
        "full_video_frame_count": np.asarray([frames], dtype=np.int64),
    }
    slam_path = (
        Path(result_path).parent / "preprocess" / "slam_results.pt"
    )
    visual_odometry_available = slam_path.is_file()
    visual_odometry_kind = None
    if visual_odometry_available:
        slam = np.asarray(
            torch.load(slam_path, map_location="cpu", weights_only=False),
            dtype=np.float64,
        )
        # The two trackers write different things.  SimpleVO writes 4x4
        # world-to-camera matrices; DPVO writes a 7-vector per frame,
        # (tx, ty, tz, qx, qy, qz, qw).  Store each in its own keys under its
        # own name rather than coercing one into the other's shape.
        if slam.ndim == 3 and slam.shape[1:] == (4, 4):
            visual_odometry_kind = "SimpleVO"
            if len(slam) != frames:
                raise SystemExit(
                    "SimpleVO trajectory {} does not cover the video ({} frames)".format(
                        slam.shape, frames))
            camera["simplevo_c2w_unregistered_full_video"] = slam
            camera["simplevo_c2w_unregistered_track"] = slam[keep]
        elif slam.ndim == 2 and slam.shape[1] == 7:
            visual_odometry_kind = "DPVO"
            if len(slam) != frames:
                raise SystemExit(
                    "DPVO trajectory {} does not cover the video ({} frames)".format(
                        slam.shape, frames))
            camera["dpvo_traj_unregistered_full_video"] = slam
            camera["dpvo_traj_unregistered_track"] = slam[keep]
            w2c = _dpvo_traj_to_w2c(slam)
            camera["dpvo_w2c_unregistered_full_video"] = w2c
            camera["dpvo_w2c_unregistered_track"] = w2c[keep]
        else:
            raise SystemExit(
                "unrecognized slam trajectory shape {}; expected [T,4,4] "
                "(SimpleVO) or [T,7] (DPVO)".format(slam.shape))
    np.savez_compressed(output_dir / "camera.npz", **camera)

    converted = {
        "atomic_motion_151": motion_151,
        "root_translation_z_up": trans_z_up,
        "rotation6d_z_up": rot6d,
        "contacts": contacts,
        "contact_valid_mask": contact_valid_mask,
        "frame_ids": keep,
    }
    quality = motion_quality_summary(converted, FPS)
    (output_dir / "quality.json").write_text(
        json.dumps(quality, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return len(keep), quality, visual_odometry_available, visual_odometry_kind


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True,
                        help="hmr4d_results.pt from run_gvhmr_extract.py")
    parser.add_argument("--extract-meta", type=Path, required=True,
                        help="extract_meta.json written next to the result")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    extract_meta = json.loads(args.extract_meta.read_text(encoding="utf-8"))
    # The video fps decides the 30 fps subsampling; take it from provenance
    # rather than guessing from frame counts.
    source_fps = float(extract_meta.get("video_fps", 30.0)) if "video_fps" in extract_meta else 30.0

    # Publish the whole clip directory atomically.  metadata.json is written
    # last, and the shard loop treats an existing quality.json as "already
    # converted", so an interruption between the two leaves a directory that
    # looks done, fails validation, and is skipped forever.  One clip in the
    # first 2667 landed in exactly that state.  A staging directory renamed
    # into place cannot be observed half-written.
    final_dir = args.output_dir
    staging_dir = final_dir.with_name(final_dir.name + ".staging")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)

    try:
        frames, quality, visual_odometry_available, visual_odometry_kind = convert(
            args.result, staging_dir, round(source_fps))
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    metadata = {
        "backend": "GVHMR",
        "converter": "convert_gvhmr_result.py",
        "source_result": str(args.result),
        "extract_meta": extract_meta,
        "frames_30fps": frames,
        "fps": FPS,
        "coordinate_convention": {
            "input": "GVHMR smpl_params_global, gravity-aligned y-up ('ay') world",
            "output": "EDGE/AtomicDance z-up; +90 degrees about X, (x,y,z)->(x,-z,y)",
            "root_orientation": "world/global root rotation after y-up -> z-up coordinate conversion",
            "body_orientation": "SMPL local joint rotations (SMPL-X body joints 1..21; 22/23 identity)",
            "camera_relation_to_body_world": "not registered; never coordinate-converted with body motion",
        },
        "camera": {
            "available": True,
            "asset": "camera.npz",
            "representation": "K_fullimg [T,3,3] per-frame intrinsics"
                              + _camera_representation_suffix(visual_odometry_kind),
            "interpretation": (
                "raw GVHMR camera evidence in its own unregistered gauge; "
                "monocular VO translation is up-to-scale and is not y-up/z-up "
                "body world. Note the legacy SimpleVO key says c2w but holds "
                "the world-to-camera matrices SimpleVO actually returns; the "
                "DPVO keys are named for what they hold."
            ),
            "registered_to_body_world": False,
            "visual_odometry_available": visual_odometry_available,
            "visual_odometry_kind": visual_odometry_kind,
        },
        "gvhmr_run_provenance": {
            "schema_version": "gvhmr-extract-v1",
            "checkpoint": extract_meta.get("checkpoint"),
            "checkpoint_sha256_1mb": extract_meta.get("checkpoint_sha256_1mb"),
            "world_convention": extract_meta.get("world_convention"),
            "visual_odometry": extract_meta.get("visual_odometry"),
            "static_cam": extract_meta.get("static_cam"),
            "video_sha256_1mb": extract_meta.get("video_sha256_1mb"),
        },
        # SMPL-X carries no SMPL hand joints; they are identity, not observed.
        "hand_joints_identity": True,
        "hand_joints": [22, 23],
    }
    (staging_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    os.rename(staging_dir, final_dir)
    print("converted {} frames -> {}".format(frames, final_dir))
    print(json.dumps(quality, indent=2)[:400])


if __name__ == "__main__":
    main()
