#!/usr/bin/env python3
"""Does a clip's 3D reconstruction span the whole clip, or only part of it?

WHY.  On 2026-08-25 a clip's "ground truth" turned out to be the first half of
the dance stretched over the whole song at half speed: ``cut_clip`` picked video
frames by frame number and audio by ``frame / 30`` seconds, so a 60 fps upload
got half the pictures and all of the sound.  **The frame counts matched by
construction** -- 480 poses for a 480-frame clip -- so every length check
passed, ``meta.json`` stayed self-consistent, and only a person watching the
panels noticed.  The cut was fixed in 49e95a3; this is the criterion, so the
next instance is caught by a gate.

WHAT IT COMPARES, AND WHY NOT PIXELS.  The first version correlated whole-frame
pixel motion against the reconstruction.  On real footage -- LED walls, other
dancers, a moving camera -- the pixel trace is dominated by background and
correlates with nothing: 0.040 at its best scale on the very clip that turned
out to be broken.  Both traces here follow the dancer instead:

* ``2D``  ``keypoints.npy``, made relative to keypoint 0 so camera motion drops
  out, then mean absolute inter-frame change.
* ``3D``  root-relative mean joint speed of the reconstruction.

The 3D trace is resampled to span a fraction ``s`` of the 2D trace and the two
are correlated; ``s`` is what the tool reports.  A correct clip peaks at 1.0.

THE CONFOUND, AND THE CONTROL THAT REMOVES IT.  Dance is periodic, so a trace
correlates with a 2x-compressed copy of itself for reasons that have nothing to
do with the cut.  Measured on five clips, the 2D trace's correlation with its
OWN compressed copy runs 0.026 to 0.402 -- as high as any suspicious reading.
So every clip is scored against its own periodicity baseline, and only a 3D
reading that **exceeds its own baseline** counts.  Without this control the tool
flags 11 of 99 clips; with it, 1.

CONTROLS (``tests/test_footage_motion_sync.py``, and measured here 2026-08-30):

* positive -- a stick-figure video rendered FROM a clip's own joints, scored
  against those joints: scale 1.000, correlation 0.775.
* negative -- the same, with only the first half of the joints: scale 0.500,
  correlation 0.773.
* null -- the 3D trace permuted: best correlation median 0.086, p95 0.165.
"""

import argparse
import pathlib
import subprocess

import numpy as np

SCALES = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4, 2.0])


def pixel_motion(video, size=64):
    """Mean absolute inter-frame difference, one number per frame.

    Forced to a square ``size x size`` rather than preserving the aspect ratio:
    the byte count then determines the frame count exactly.  The first version
    let the scaler pick the height and tried to recover it from the buffer
    length, which found the wrong factorisation and returned a handful of
    frames -- the correlation below then had nothing to correlate and every
    scale read NaN.  Aspect distortion costs nothing here because the trace is a
    whole-frame mean.
    """
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-vf",
         "scale={s}:{s},format=gray".format(s=size), "-f", "rawvideo", "-"],
        capture_output=True)
    raw = np.frombuffer(out.stdout, np.uint8)
    frames = raw.size // (size * size)
    if frames < 2:
        return np.array([])
    stack = raw[:frames * size * size].reshape(frames, size, size).astype(np.float32)
    return np.abs(np.diff(stack, axis=0)).mean(axis=(1, 2))


def keypoint_motion(keypoints):
    """2D trace, made relative to keypoint 0 so the camera drops out."""
    array = np.nan_to_num(np.asarray(keypoints, float))
    if array.ndim != 3 or len(array) < 3:
        return np.array([])
    relative = array - array[:, :1, :]
    return np.abs(np.diff(relative.reshape(len(relative), -1), axis=0)).mean(1)


def joint_motion(joints):
    relative = np.asarray(joints, float)
    relative = relative - relative[:, :1, :]
    return np.linalg.norm(np.diff(relative, axis=0), axis=2).mean(1)


def _z(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-12)


def best_scale(pixels, joints, scales=SCALES):
    """Correlation of the two traces at each time scale; returns every reading."""
    pixels = _z(pixels)
    readings = []
    for scale in scales:
        length = int(len(pixels) * scale)
        if length < 16:
            readings.append(np.nan)
            continue
        resampled = np.interp(np.linspace(0, len(joints) - 1, length),
                              np.arange(len(joints)), joints)
        n = min(len(pixels), len(resampled))
        readings.append(float((_z(pixels[:n]) * _z(resampled[:n])).mean()))
    return np.array(readings)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", required=True)
    parser.add_argument("--ingest", default="/cache/atomicdance-assets/data/wild_ingest_v1")
    parser.add_argument("--motion-dir", default="runs/wild_v5_song_gt_eval/motion")
    parser.add_argument("--tolerance", type=float, default=0.06,
                        help="how far the best scale may sit from 1.0")
    parser.add_argument("--margin", type=float, default=1.5,
                        help="how far above its own periodicity baseline an off-scale "
                             "reading must sit before it is called a defect")
    parser.add_argument("--quiet", action="store_true", help="print only the failures")
    args = parser.parse_args()

    import json
    import pickle

    ingest = pathlib.Path(args.ingest)
    motion_dir = pathlib.Path(args.motion_dir)
    one = list(SCALES).index(1.0)
    if not args.quiet:
        print("{:<32} {:>7} {:>7} {:>8} {:>9} {:>7}".format(
            "clip", "scale", "corr", "corr@1.0", "periodic", "fps"))
    failed = []
    for line in open(args.clips):
        clip = line.strip()
        if not clip:
            continue
        stem = clip.replace("wild_v5:", "").replace(":", "__")
        keypoints = ingest / stem / "keypoints.npy"
        pkl = motion_dir / (clip + ".pkl")
        if not keypoints.is_file() or not pkl.is_file():
            continue
        two_d = keypoint_motion(np.load(keypoints))
        joints = joint_motion(np.asarray(pickle.load(open(pkl, "rb"))["full_pose"], float))
        if len(two_d) < 32 or len(joints) != len(two_d):
            continue
        readings = best_scale(two_d, joints)
        baseline = best_scale(two_d, two_d)
        if np.all(np.isnan(readings)):
            continue
        index = int(np.nanargmax(readings))
        scale, value = float(SCALES[index]), float(readings[index])
        own = float(baseline[index])
        meta_path = ingest / stem / "meta.json"
        fps = json.load(open(meta_path)).get("source_fps") if meta_path.is_file() else None
        off = abs(scale - 1.0) > args.tolerance
        # A reading only counts as a defect if it also beats the clip's own
        # periodicity: an off-scale peak that a trace produces against ITSELF is
        # a property of dance, not of the cut.
        beats_periodicity = value > max(own, 0.0) * args.margin and value > 0.2
        if not args.quiet:
            print("{:<32} {:>7.2f} {:>7.3f} {:>8.3f} {:>9.3f} {:>7}".format(
                stem[:32], scale, value, float(readings[one]), own, str(fps)))
        if off and beats_periodicity:
            failed.append("{}: 3D spans {:.2f} of the clip (corr {:.3f} against a "
                          "periodicity baseline of {:.3f}, corr at 1.0 = {:.3f}, "
                          "source_fps {})".format(
                              stem, scale, value, own, float(readings[one]), fps))
    for line in failed:
        print("FAIL " + line)
    print("{} clip(s) flagged".format(len(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
