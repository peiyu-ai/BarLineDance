#!/usr/bin/env python3
"""Time-warp a generated clip so its settle points land on the music's beats.

WHERE THIS SITS IN THE CAUSAL CHAIN, measured 2026-08-31 (100/50-clip sets):

* the retrieval draft carries real dance and full accent structure
  (energy 1.031 of ground truth);
* the completion transmits NONE of the draft's timing -- output-vs-own-draft
  speed-profile correlation is **-0.110**, below even output-vs-unrelated-dance
  (+0.056) -- because the training drafts are другой performance of the same
  classes, timing-uncorrelated with the target by construction, so ignoring
  their timing was the optimum the model correctly learned;
* repairing that inside training was tried and measured before any GPU time:
  settle-anchor alignment moved draft-to-target timing correlation 0.036 to
  only 0.046, per-segment DTW to 0.071, while the whole-window DTW ceiling of
  0.456 turns out to live in the cross-segment macro structure (where the
  transition gaps sit), which is the label boundary structure and not movable.
  So the draft path cannot deliver "accent on the strong beat", and no sampler
  knob can either.

Hence a post-process, at the one place the timing still exists: the output's
own settle points (the paper's motion beats, "local minima of segment-wise
joint velocities") are pulled onto the music's beat frames by a monotone
piecewise-linear warp with a hard local-stretch cap.  This is unapologetically
post-hoc; what keeps it honest is that it is *bounded* (default cap 1.35x,
anchors farther than --tolerance frames from any beat are left alone), it warps
every channel of the artifact consistently (poses, translation, contacts,
joints), and it is measured: the hit-rate is reported against the ground
truth's own hit-rate on the same clips, with energy and jitter re-measured so a
warp that buys beats by shaking or slowing is caught by the standing columns.

The ground truth is the calibration, not perfection: dancers do not hit every
beat, so ``--tolerance`` and the anchor budget are chosen so the SNAPPED
hit-rate approaches the ground truth's, not 100%.
"""
import argparse
import json
import pathlib
import pickle
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

BEAT_CHANNEL = 34
FPS = 30.0


def joint_speed(full_pose):
    joints = np.asarray(full_pose, float)
    relative = joints - joints[:, :1, :]
    return np.linalg.norm(np.diff(relative, axis=0), axis=2).mean(1)


def settle_frames(speed, min_gap=5):
    minima = [i for i in range(1, len(speed) - 1)
              if speed[i] <= speed[i - 1] and speed[i] <= speed[i + 1]]
    kept = []
    for frame in minima:
        if kept and frame - kept[-1] < min_gap:
            if speed[frame] < speed[kept[-1]]:
                kept[-1] = frame
            continue
        kept.append(frame)
    return np.asarray(kept, int)


def beat_map(frames, settles, beats, tolerance=6, max_stretch=1.35):
    """Monotone map pulling each beat's nearest settle onto that beat."""
    pairs = []
    for beat in np.sort(np.asarray(beats, int)):
        if not (0 < beat < frames - 1) or len(settles) == 0:
            continue
        settle = int(settles[np.abs(settles - beat).argmin()])
        if abs(settle - beat) > tolerance or abs(settle - beat) == 0:
            if abs(settle - beat) != 0:
                continue
        if pairs and (settle <= pairs[-1][0] or beat <= pairs[-1][1]):
            continue
        previous = pairs[-1] if pairs else (0, 0)
        ratio = (settle - previous[0]) / max(beat - previous[1], 1)
        if ratio > max_stretch or ratio < 1.0 / max_stretch:
            continue
        pairs.append((settle, beat))
    if not pairs:
        return None
    source = np.array([0] + [p[0] for p in pairs] + [frames - 1], float)
    target = np.array([0] + [p[1] for p in pairs] + [frames - 1], float)
    tail = (source[-1] - source[-2]) / max(target[-1] - target[-2], 1)
    if tail > max_stretch or tail < 1.0 / max_stretch:
        source, target = source[:-1], target[:-1]
        if len(source) < 2:
            return None
        source = np.append(source, frames - 1)
        target = np.append(target, frames - 1)
    return np.interp(np.arange(frames, dtype=float), target, source)


def resample(array, positions):
    array = np.asarray(array)
    low = np.clip(np.floor(positions).astype(int), 0, len(array) - 1)
    high = np.clip(low + 1, 0, len(array) - 1)
    fraction = (positions - low).reshape((-1,) + (1,) * (array.ndim - 1))
    return array[low] * (1.0 - fraction) + array[high] * fraction


def snap_clip(payload, music, tolerance=6, max_stretch=1.35):
    full_pose = np.asarray(payload["full_pose"], float)
    frames = len(full_pose)
    beats = np.flatnonzero(np.asarray(music)[:frames, BEAT_CHANNEL] > 0.5)
    if len(beats) < 4:
        return None
    speed = joint_speed(full_pose)
    settles = settle_frames(speed)
    positions = beat_map(frames, settles, beats, tolerance, max_stretch)
    if positions is None:
        return None
    out = dict(payload)
    for key in ("smpl_poses", "smpl_trans", "full_pose", "contacts"):
        if key in out:
            out[key] = resample(out[key], positions).astype(np.asarray(payload[key]).dtype)
    # labels are categorical: nearest-frame gather, never blended
    if "atomic_labels" in out:
        nearest = np.clip(np.round(positions).astype(int), 0, frames - 1)
        out["atomic_labels"] = np.asarray(payload["atomic_labels"])[nearest]
    out["beat_snap"] = {"tolerance": tolerance, "max_stretch": max_stretch,
                        "anchors": int(len(settles)), "beats": int(len(beats))}
    return out


def hit_rate(full_pose, music, window=2):
    frames = len(full_pose)
    beats = np.flatnonzero(np.asarray(music)[:frames, BEAT_CHANNEL] > 0.5)
    settles = settle_frames(joint_speed(full_pose))
    if len(beats) < 4 or len(settles) < 2:
        return np.nan
    return float((np.abs(np.asarray(beats)[:, None] - settles[None, :]).min(1) <= window).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="run dir of generated pkls")
    parser.add_argument("--output", required=True)
    parser.add_argument("--clips", required=True)
    parser.add_argument("--audio-dir", default="runs/wild_v5_song_gt_eval/audio")
    parser.add_argument("--ground-truth-dir", default="runs/wild_v5_song_gt_eval/motion")
    parser.add_argument("--tolerance", type=int, default=6)
    parser.add_argument("--max-stretch", type=float, default=1.35)
    args = parser.parse_args()

    source = pathlib.Path(args.input)
    target = pathlib.Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    audio = pathlib.Path(args.audio_dir)
    gt_dir = pathlib.Path(args.ground_truth_dir)
    rows = {"gt": [], "before": [], "after": []}
    for line in open(args.clips):
        clip = line.strip()
        pkl = source / (clip + ".pkl")
        wav = audio / (clip + ".npy")
        if not clip or not pkl.is_file() or not wav.is_file():
            continue
        payload = pickle.load(open(pkl, "rb"))
        music = np.load(wav)
        snapped = snap_clip(payload, music, args.tolerance, args.max_stretch)
        out = snapped if snapped is not None else payload
        with open(target / (clip + ".pkl"), "wb") as handle:
            pickle.dump(out, handle)
        rows["before"].append(hit_rate(payload["full_pose"], music))
        rows["after"].append(hit_rate(out["full_pose"], music))
        gt_pkl = gt_dir / (clip + ".pkl")
        if gt_pkl.is_file():
            gt = pickle.load(open(gt_pkl, "rb"))["full_pose"]
            rows["gt"].append(hit_rate(np.asarray(gt, float)[:len(payload["full_pose"])], music))
    manifest = {"input": str(source), "tolerance": args.tolerance,
                "max_stretch": args.max_stretch,
                "beat_hit_rate": {k: float(np.nanmean(v)) for k, v in rows.items() if v},
                "clips": len(rows["before"])}
    (target / "beat_snap_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["beat_hit_rate"], indent=1), "n=", manifest["clips"])


if __name__ == "__main__":
    main()
