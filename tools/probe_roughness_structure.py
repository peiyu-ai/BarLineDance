#!/usr/bin/env python3
"""Where does the roughness live -- in time, in frequency, or in the body?

The wild 340-frame arm sits at 5.43x ground truth's jerk and the open question
is *why a clean conditioning signal makes the output rougher*: the draft the
model is handed has jerk 339 against ground truth's 360, and the model returns
1954.  Two candidates survive every control run so far -- the training objective
(``F.mse_loss`` over 151 equally-weighted dimensions plus a term that only fires
at plan boundaries) and a conflict between the draft and the music.  Separating
them end to end needs a GPU.  Three decompositions of the artifacts already on
disk do not, and each can falsify something.

**Temporal.**  Within one clip, is the roughness on the frames the draft filled,
or everywhere?  ``atomic_labels`` records the plan, so frames split into
draft-covered (label != 0) and transition (label 0, where the draft is a
constant pose under ``gap_fill=zero``).  Frames within ``--boundary-half`` of a
label change are dropped from both sides: boundary jerk is a separately measured
effect at 2.28x the rest of the clip and would land entirely in the covered
bucket.  If covered frames are much rougher than transition frames, the
roughness tracks the conditioning signal; if the two match, it is global.
Read against the no-plan arm, where every frame is uncovered and jerk is 170.

**Spectral.**  Welch PSD of root-relative joint position, generated against
ground truth, per band.  A per-frame independent prediction error is white --
its excess is flat across frequency.  A structural cause (window stitching, the
plan's segment rate, the 75-frame completion stride) is a peak.  Root
translation is excluded and reported separately because on this corpus ground
truth's root is a GVHMR reconstruction that drifts more than the generations do,
so any root-inclusive spectrum is biased in the generations' favour.

**Anatomical.**  This is the sharpest of the three, because the objective
hypothesis makes a quantitative prediction that the conflict hypothesis does
not.  ``mse_loss`` over rot6d weights every joint's rotation equally regardless
of its lever arm, so a model trained under it should make roughly *equal
rotational* error everywhere -- and the skeleton then amplifies that error into
position in proportion to how far the joint sits from the root.  So: angular
jerk should be flatter across joints in the generations than in ground truth
(a real dancer whips the extremities), while positional jerk excess should grow
with lever arm.  Both halves have to hold; either one alone has other
explanations.

Ground-truth *rotations* are not in the exported eval motions -- those carry
``full_pose`` only -- so they come from the release's own held-out
``motion.npy`` through the same inverse-normalisation the generator's output
takes.  That correspondence is not assumed: ``--check-alignment`` decodes the
release rows to joint positions and refuses if they disagree with the exported
ground truth by more than a tolerance, which is the check
``probe_plan_information`` already passed at 0.000003 m.

Usage::

    python3 tools/probe_roughness_structure.py \\
        --run runs/t14_oracle/arm_self \\
        --ground-truth-dir runs/wild_v4_acct_gt_eval/motion \\
        --data-root /dev/shm/atomicdance-acct/release_w2_340 \\
        --compare runs/t16_noplan/arm_noplan \\
        --output runs/t24_structure/self_duration_s1.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

FPS = 30.0
MOTION_DIM = 151
# Boundary jerk is 2.28x the rest of the clip and lives entirely on covered
# frames; leaving it in would answer the temporal question with a different
# effect's number.
DEFAULT_BOUNDARY_HALF = 2
# Bands in Hz at 30 fps.  The top band is where a per-frame independent error
# puts most of its power and where a dancer puts almost none.
BANDS = ((0.0, 1.0), (1.0, 3.0), (3.0, 6.0), (6.0, 10.0), (10.0, 15.0))

WRISTS = [20, 21]
# The pelvis is identically still in root-relative coordinates, so the hips
# are the proximal reference there and the same pair is used in both frames
# so the two columns are comparable.
HIPS = [1, 2]

SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]


class ProbeError(RuntimeError):
    """A decomposition could not be computed, which is not the same as flat."""


# --------------------------------------------------------------------------- #
# primitives


def jerk_per_frame(positions: np.ndarray) -> np.ndarray:
    """[T,J,3] -> [T-3] mean over joints of the third difference magnitude."""
    return np.linalg.norm(np.diff(positions, n=3, axis=0), axis=-1).mean(axis=-1) * (FPS ** 3)


def jerk_per_joint(positions: np.ndarray) -> np.ndarray:
    """[T,J,3] -> [J] median over frames, so one bad frame cannot set a joint."""
    return np.median(np.linalg.norm(np.diff(positions, n=3, axis=0), axis=-1), axis=0) * (FPS ** 3)


def angular_jerk_per_joint(axis_angle: np.ndarray) -> np.ndarray:
    """[T,J,3] axis-angle -> [J] median third difference of the rotation vector.

    Taken on the axis-angle vector rather than on geodesic increments because
    the question is only whether the *spread across joints* is flat, and a
    monotone reparametrisation of each joint's own magnitude cannot create or
    destroy that spread.

    The **median** is what makes that safe against the +-pi representation flip:
    a rotation of angle ``theta`` about ``u`` and one of ``theta - 2*pi`` about
    the same ``u`` are the same rotation and differ by ``2*pi`` here, so a flip
    puts one enormous third difference into the series -- and a median does not
    see it.  Measured on a synthetic field that flips, the median is unchanged
    to 1e-11.

    An unwrap was written for this and then deleted: it was wrong twice (once
    running ``np.unwrap`` on a magnitude, which is non-negative by construction
    and cannot wrap; once greedily, which cannot track a value that has drifted
    past one 2*pi), and the statistic never needed it.  Shipping a subtly wrong
    correction for a problem the estimator already resists is worse than
    shipping neither.  If this is ever changed to a mean, the unwrap comes back
    first and the test in tests/test_probe_roughness_structure.py says why.
    """
    return np.median(np.linalg.norm(np.diff(axis_angle, n=3, axis=0), axis=-1), axis=0) * (FPS ** 3)


def lever_arms(positions: np.ndarray) -> np.ndarray:
    """[T,J,3] -> [J] mean distance from the root joint."""
    return np.linalg.norm(positions - positions[:, :1], axis=-1).mean(axis=0)


def kinematic_depth() -> np.ndarray:
    depth = np.zeros(len(SMPL_PARENTS), dtype=np.int64)
    for joint, parent in enumerate(SMPL_PARENTS):
        depth[joint] = 0 if parent < 0 else depth[parent] + 1
    return depth


def welch_psd(signal: np.ndarray, segment: int) -> Tuple[np.ndarray, np.ndarray]:
    """Mean periodogram over half-overlapping Hann windows, averaged over columns.

    Written out rather than taken from scipy so the segment length -- which sets
    the frequency resolution and therefore whether a narrow peak is visible at
    all -- is a parameter of this file and lands in the report.
    """
    if len(signal) < segment:
        raise ProbeError("clip shorter than the PSD segment")
    window = np.hanning(segment)
    correction = (window ** 2).sum()
    step = segment // 2
    starts = range(0, len(signal) - segment + 1, step)
    total = None
    count = 0
    for start in starts:
        block = signal[start:start + segment]
        block = block - block.mean(axis=0)
        spectrum = np.abs(np.fft.rfft(block * window[:, None], axis=0)) ** 2
        total = spectrum if total is None else total + spectrum
        count += 1
    power = (total / count).mean(axis=1) / (correction * FPS)
    freqs = np.fft.rfftfreq(segment, d=1.0 / FPS)
    return freqs, power


def band_power(freqs: np.ndarray, power: np.ndarray) -> Dict[str, float]:
    out = {}
    for low, high in BANDS:
        mask = (freqs >= low) & (freqs < high)
        out["{:g}-{:g}Hz".format(low, high)] = float(power[mask].sum()) if mask.any() else 0.0
    return out


# --------------------------------------------------------------------------- #
# loading


def load_generated(directory: pathlib.Path) -> Dict[str, dict]:
    clips = {}
    for path in sorted(pathlib.Path(directory).glob("*.pkl")):
        clips[path.stem] = pickle.load(open(path, "rb"))
    if not clips:
        raise ProbeError("no generated clips in {}".format(directory))
    return clips


def load_ground_truth(directory: pathlib.Path) -> Dict[str, np.ndarray]:
    out = {}
    for path in sorted(pathlib.Path(directory).glob("*.pkl")):
        payload = pickle.load(open(path, "rb"))
        if "full_pose" in payload:
            out[path.stem] = np.asarray(payload["full_pose"], dtype=np.float64)
    if not out:
        raise ProbeError("no ground-truth motions in {}".format(directory))
    return out


def load_release_rotations(data_root: pathlib.Path, split: str,
                           wanted: Sequence[str]) -> Dict[str, np.ndarray]:
    """Ground-truth rotations, from the array the exported motions were decoded from.

    One row per recording (the first slice), which is the window the 340-frame
    protocol generates against.
    """
    from tools.score_generation_diagnostics import _rotations_from_151
    import torch

    root = pathlib.Path(data_root)
    held = root / split
    normalizer = root / "normalizer.pt"
    if not normalizer.is_file():
        raise ProbeError("no normalizer at {}".format(normalizer))
    motion = np.load(str(held / "motion.npy"), mmap_mode="r")
    names = json.load(open(str(held / "names.json")))

    first: Dict[str, int] = {}
    for row, name in enumerate(names):
        recording = name.rsplit("_slice", 1)[0]
        first.setdefault(recording, row)

    out = {}
    for clip in wanted:
        row = first.get(clip)
        if row is None:
            continue
        values = torch.from_numpy(np.array(motion[row], dtype=np.float32))
        out[clip] = _rotations_from_151(values, normalizer).numpy().astype(np.float64)
    if not out:
        raise ProbeError("no release rows matched the generated clips")
    return out


def check_alignment(data_root: pathlib.Path, split: str, clip: str,
                    ground_truth: np.ndarray, tolerance: float) -> float:
    """Decode one release row to joint positions and compare with the export.

    Not assumed: everything anatomical below reads rotations from one array and
    positions from another, and if they are not the same frames the whole
    section is comparing two dancers.
    """
    from infer_atomic import decode_motion
    import torch

    root = pathlib.Path(data_root)
    motion = np.load(str(root / split / "motion.npy"), mmap_mode="r")
    names = json.load(open(str(root / split / "names.json")))
    row = next((i for i, n in enumerate(names) if n.rsplit("_slice", 1)[0] == clip), None)
    if row is None:
        raise ProbeError("clip {} is not in the {} split".format(clip, split))
    decoded = decode_motion(torch.from_numpy(np.array(motion[row], dtype=np.float32)),
                            root / "normalizer.pt")["full_pose"]
    frames = min(len(decoded), len(ground_truth))
    error = float(np.abs(np.asarray(decoded[:frames]) - ground_truth[:frames]).max())
    if error > tolerance:
        raise ProbeError(
            "release row and exported ground truth disagree by {:.6f} m on {} "
            "(tolerance {}); the anatomical section would compare two bodies".format(
                error, clip, tolerance))
    return error


# --------------------------------------------------------------------------- #
# decompositions


def root_vs_articulation(clips: Dict[str, dict],
                         truth: Dict[str, np.ndarray]) -> Dict[str, object]:
    """Split the jerk into the root translation and everything else.

    This is the decomposition the other three point at.  ``full_pose[:, 0]`` is
    the root joint, which is exactly the 3 dimensions (4:7 of the 151-D vector)
    the model predicts as an *absolute position* per frame; every other joint
    inherits it through forward kinematics, so a jittering root is common-mode
    across the whole body.

    Two readings come out of it and both are load-bearing:

    * how much of the whole-body jerk those 3 dimensions carry, and
    * whether the articulation -- the same pose measured with the root
      subtracted -- is rough at all.  Measured on the wild 340 arm it is 0.78x
      ground truth, i.e. the dance itself is slightly *smoother* than the
      recording it was trained on, and the 5.4x whole-body figure is one
      channel.

    The joint-energy profile is reported the same way for the same reason: in
    world coordinates the generations look flat (across-joint speed CV 0.286
    against ground truth's 0.583), and with the root removed the two are
    indistinguishable (0.743 against 0.748).  A common-mode term added to every
    joint compresses the spread between them, so "the profile is flat" and "the
    root jitters" are the same observation twice.
    """
    def jerk_of(series: np.ndarray) -> float:
        step = np.linalg.norm(np.diff(series, n=3, axis=0), axis=-1)
        if step.ndim > 1:
            step = step.mean(axis=-1)
        return float(np.median(step)) * (FPS ** 3)

    def profile(pose: np.ndarray, relative: bool) -> Tuple[float, float]:
        values = pose - pose[:, 0:1] if relative else pose
        speed = np.linalg.norm(np.diff(values, axis=0), axis=-1) * FPS
        per_joint = np.median(speed, axis=0)
        # The root is identically still once the root is subtracted, so it can
        # neither enter the spread nor be the denominator of the ratio.
        body = np.delete(per_joint, 0) if relative else per_joint
        return (float(body.std() / body.mean()) if body.mean() else float("nan"),
                float(per_joint[WRISTS].mean() / per_joint[HIPS].mean()))

    rows = []
    for name, clip in sorted(clips.items()):
        pose = np.asarray(clip["full_pose"], dtype=np.float64)
        reference = truth.get(name)
        cv_world, wh_world = profile(pose, False)
        cv_local, wh_local = profile(pose, True)
        entry = {
            "clip": name,
            "root_jerk": jerk_of(pose[:, 0]),
            "articulation_jerk": jerk_of(pose[:, 1:] - pose[:, 0:1]),
            "whole_body_jerk": jerk_of(pose),
            "cv_world": cv_world, "cv_root_relative": cv_local,
            "wrist_over_hip_world": wh_world, "wrist_over_hip_root_relative": wh_local,
        }
        if reference is not None:
            ref = reference[:len(pose)]
            entry["truth_root_jerk"] = jerk_of(ref[:, 0])
            entry["truth_articulation_jerk"] = jerk_of(ref[:, 1:] - ref[:, 0:1])
            rcv_w, rwh_w = profile(ref, False)
            rcv_l, rwh_l = profile(ref, True)
            entry["truth_cv_world"] = rcv_w
            entry["truth_cv_root_relative"] = rcv_l
            entry["truth_wrist_over_hip_world"] = rwh_w
            entry["truth_wrist_over_hip_root_relative"] = rwh_l
        rows.append(entry)
    if not rows:
        raise ProbeError("no clips to decompose")

    def med(key):
        values = [r[key] for r in rows if key in r]
        return float(np.median(values)) if values else None

    summary = {
        "clips": len(rows),
        "root_jerk": med("root_jerk"),
        "articulation_jerk": med("articulation_jerk"),
        "whole_body_jerk": med("whole_body_jerk"),
        "root_share_of_whole_body": (med("root_jerk") / med("whole_body_jerk")
                                     if med("whole_body_jerk") else None),
        "cv_world": med("cv_world"),
        "cv_root_relative": med("cv_root_relative"),
        "wrist_over_hip_world": med("wrist_over_hip_world"),
        "wrist_over_hip_root_relative": med("wrist_over_hip_root_relative"),
    }
    if med("truth_root_jerk"):
        summary.update({
            "truth_root_jerk": med("truth_root_jerk"),
            "truth_articulation_jerk": med("truth_articulation_jerk"),
            "root_jerk_ratio": med("root_jerk") / med("truth_root_jerk"),
            "articulation_jerk_ratio": med("articulation_jerk") / med("truth_articulation_jerk"),
            "truth_cv_world": med("truth_cv_world"),
            "truth_cv_root_relative": med("truth_cv_root_relative"),
            "truth_wrist_over_hip_root_relative": med("truth_wrist_over_hip_root_relative"),
        })
    summary["per_clip"] = rows
    return summary


def temporal(clips: Dict[str, dict], boundary_half: int) -> Dict[str, object]:
    covered, transition, per_clip = [], [], []
    no_labels = 0
    for name, clip in sorted(clips.items()):
        labels = clip.get("atomic_labels")
        pose = np.asarray(clip["full_pose"], dtype=np.float64)
        if labels is None:
            no_labels += 1
            continue
        labels = np.asarray(labels)
        jerk = jerk_per_frame(pose)
        labels = labels[:len(jerk)]
        change = np.zeros(len(labels), dtype=bool)
        change[1:] = labels[1:] != labels[:-1]
        near = np.zeros(len(labels), dtype=bool)
        for offset in range(-boundary_half, boundary_half + 1):
            near |= np.roll(change, offset)
        is_covered = (labels != 0) & ~near
        is_transition = (labels == 0) & ~near
        if is_covered.sum() < 10 or is_transition.sum() < 10:
            continue
        c, t = float(np.median(jerk[is_covered])), float(np.median(jerk[is_transition]))
        covered.append(c)
        transition.append(t)
        per_clip.append({"clip": name, "covered": c, "transition": t,
                         "covered_frames": int(is_covered.sum()),
                         "transition_frames": int(is_transition.sum())})
    if not covered:
        raise ProbeError("no clip had both covered and transition frames away from a boundary")
    covered_a, transition_a = np.array(covered), np.array(transition)
    wins = int((covered_a > transition_a).sum())
    return {
        "clips": len(covered),
        "clips_without_labels": no_labels,
        "boundary_half_width": boundary_half,
        "covered_jerk_median": float(np.median(covered_a)),
        "transition_jerk_median": float(np.median(transition_a)),
        "ratio_covered_over_transition": float(np.median(covered_a) / np.median(transition_a)),
        "clips_where_covered_is_rougher": wins,
        "paired_median_difference": float(np.median(covered_a - transition_a)),
        "per_clip": per_clip,
    }


def spectral(clips: Dict[str, dict], truth: Dict[str, np.ndarray],
             segment: int) -> Dict[str, object]:
    def spectrum(pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        relative = (pose - pose[:, :1]).reshape(len(pose), -1)
        return welch_psd(relative, segment)

    gen_bands, truth_bands, ratios, root_ratio = [], [], [], []
    freqs = None
    for name, clip in sorted(clips.items()):
        if name not in truth:
            continue
        pose = np.asarray(clip["full_pose"], dtype=np.float64)
        reference = truth[name][:len(pose)]
        if len(reference) < segment or len(pose) < segment:
            continue
        freqs, gen_power = spectrum(pose)
        _, truth_power = spectrum(reference)
        gen_bands.append(band_power(freqs, gen_power))
        truth_bands.append(band_power(freqs, truth_power))
        ratios.append({k: gen_bands[-1][k] / truth_bands[-1][k]
                       for k in gen_bands[-1] if truth_bands[-1][k] > 0})
        _, gen_root = welch_psd(pose[:, 0], segment)
        _, truth_root = welch_psd(reference[:, 0], segment)
        gr, tr = band_power(freqs, gen_root), band_power(freqs, truth_root)
        root_ratio.append({k: gr[k] / tr[k] for k in gr if tr[k] > 0})
    if not ratios:
        raise ProbeError("no clip long enough for the PSD segment")
    keys = list(ratios[0])
    median_ratio = {k: float(np.median([r[k] for r in ratios if k in r])) for k in keys}
    values = np.array([median_ratio[k] for k in keys])
    return {
        "clips": len(ratios),
        "segment_frames": segment,
        "frequency_resolution_hz": float(FPS / segment),
        "band_ratio_generated_over_truth": median_ratio,
        "root_band_ratio_generated_over_truth": {
            k: float(np.median([r[k] for r in root_ratio if k in r])) for k in keys},
        # Flat means white means per-frame independent error; a peak means a
        # mechanism with a period.
        "ratio_spread_max_over_min": float(values.max() / values.min()),
        "highest_band_over_lowest": float(values[-1] / values[0]),
    }


def anatomical(clips: Dict[str, dict], truth: Dict[str, np.ndarray],
               truth_rotations: Dict[str, np.ndarray]) -> Dict[str, object]:
    gen_pos, truth_pos, gen_ang, truth_ang, arms = [], [], [], [], []
    for name, clip in sorted(clips.items()):
        if name not in truth or name not in truth_rotations:
            continue
        pose = np.asarray(clip["full_pose"], dtype=np.float64)
        reference = truth[name][:len(pose)]
        if len(reference) < len(pose):
            pose = pose[:len(reference)]
        gen_pos.append(jerk_per_joint(pose))
        truth_pos.append(jerk_per_joint(reference))
        arms.append(lever_arms(reference))
        gen_rot = np.asarray(clip["smpl_poses"], dtype=np.float64).reshape(len(clip["smpl_poses"]), 24, 3)
        ref_rot = truth_rotations[name][:len(gen_rot)]
        frames = min(len(gen_rot), len(ref_rot))
        gen_ang.append(angular_jerk_per_joint(gen_rot[:frames]))
        truth_ang.append(angular_jerk_per_joint(ref_rot[:frames]))
    if not gen_pos:
        raise ProbeError("no clip had both a ground-truth motion and a release row")

    gp, tp = np.median(np.stack(gen_pos), axis=0), np.median(np.stack(truth_pos), axis=0)
    ga, ta = np.median(np.stack(gen_ang), axis=0), np.median(np.stack(truth_ang), axis=0)
    arm = np.median(np.stack(arms), axis=0)
    depth = kinematic_depth()
    position_ratio = gp / np.maximum(tp, 1e-9)

    def cv(values):
        return float(values.std() / values.mean()) if values.mean() else float("nan")

    return {
        "clips": len(gen_pos),
        "per_joint": [
            {"joint": int(j), "depth": int(depth[j]), "lever_arm_m": float(arm[j]),
             "generated_position_jerk": float(gp[j]), "truth_position_jerk": float(tp[j]),
             "position_jerk_ratio": float(position_ratio[j]),
             "generated_angular_jerk": float(ga[j]), "truth_angular_jerk": float(ta[j])}
            for j in range(len(gp))
        ],
        # The objective hypothesis predicts both of these: equal rotational
        # error everywhere (generated angular spread below ground truth's), and
        # a positional excess that grows with lever arm.
        "angular_jerk_cv_generated": cv(ga),
        "angular_jerk_cv_truth": cv(ta),
        "angular_cv_ratio_generated_over_truth": cv(ga) / cv(ta) if cv(ta) else float("nan"),
        "corr_position_ratio_with_lever_arm": float(np.corrcoef(position_ratio, arm)[0, 1]),
        "corr_position_ratio_with_depth": float(np.corrcoef(position_ratio, depth)[0, 1]),
        "position_jerk_ratio_median": float(np.median(position_ratio)),
    }


# --------------------------------------------------------------------------- #
# driver


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", required=True)
    p.add_argument("--ground-truth-dir", required=True)
    p.add_argument("--data-root", help="release, for ground-truth rotations")
    p.add_argument("--split", default="test")
    p.add_argument("--compare", help="a second run scored the same way, e.g. the no-plan arm")
    p.add_argument("--boundary-half", type=int, default=DEFAULT_BOUNDARY_HALF)
    p.add_argument("--psd-segment", type=int, default=128)
    p.add_argument("--alignment-tolerance", type=float, default=1e-3)
    p.add_argument("--limit", type=int, default=None,
                   help="score the first N clips in name order; the count rides in the report")
    p.add_argument("--output")
    return p


def score_run(run_dir, truth, options) -> Dict[str, object]:
    clips = load_generated(pathlib.Path(run_dir))
    if options.limit:
        clips = {k: clips[k] for k in sorted(clips)[:options.limit]}
    out: Dict[str, object] = {"run": str(run_dir), "clips": len(clips)}
    out["root_vs_articulation"] = root_vs_articulation(clips, truth)
    out["temporal"] = temporal(clips, options.boundary_half)
    out["spectral"] = spectral(clips, truth, options.psd_segment)
    if options.data_root:
        names = sorted(set(clips) & set(truth))
        alignment = check_alignment(pathlib.Path(options.data_root), options.split,
                                    names[0], truth[names[0]], options.alignment_tolerance)
        rotations = load_release_rotations(pathlib.Path(options.data_root), options.split, names)
        out["alignment_max_error_m"] = alignment
        out["anatomical"] = anatomical(clips, truth, rotations)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    options = build_parser().parse_args(argv)
    try:
        truth = load_ground_truth(pathlib.Path(options.ground_truth_dir))
        report = {"primary": score_run(options.run, truth, options)}
        if options.compare:
            report["compare"] = score_run(options.compare, truth, options)
    except ProbeError as error:
        print("REFUSED: {}".format(error), file=sys.stderr)
        return 2
    text = json.dumps(report, indent=2, sort_keys=True)
    if options.output:
        path = pathlib.Path(options.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    summary = report["primary"]
    split = summary["root_vs_articulation"]
    if "root_jerk_ratio" in split:
        print("root/artic root jerk {:.0f} ({:.2f}x truth), articulation {:.0f} ({:.2f}x truth), "
              "root is {:.0%} of the whole body".format(
                  split["root_jerk"], split["root_jerk_ratio"],
                  split["articulation_jerk"], split["articulation_jerk_ratio"],
                  split["root_share_of_whole_body"]))
        print("           across-joint speed CV: world {:.3f} vs truth {:.3f}; "
              "root-relative {:.3f} vs truth {:.3f}".format(
                  split["cv_world"], split["truth_cv_world"],
                  split["cv_root_relative"], split["truth_cv_root_relative"]))
    print("temporal   covered {:.0f} vs transition {:.0f}  ratio {:.2f}  ({}/{} clips rougher when covered)".format(
        summary["temporal"]["covered_jerk_median"], summary["temporal"]["transition_jerk_median"],
        summary["temporal"]["ratio_covered_over_transition"],
        summary["temporal"]["clips_where_covered_is_rougher"], summary["temporal"]["clips"]))
    print("spectral   band ratios gen/truth: " + ", ".join(
        "{} {:.1f}x".format(k, v)
        for k, v in summary["spectral"]["band_ratio_generated_over_truth"].items()))
    print("           spread max/min {:.2f}  (flat => white => per-frame error)".format(
        summary["spectral"]["ratio_spread_max_over_min"]))
    if "anatomical" in summary:
        a = summary["anatomical"]
        print("anatomical angular-jerk CV: generated {:.3f} vs truth {:.3f}  (ratio {:.2f})".format(
            a["angular_jerk_cv_generated"], a["angular_jerk_cv_truth"],
            a["angular_cv_ratio_generated_over_truth"]))
        print("           corr(position jerk ratio, lever arm) = {:+.3f}".format(
            a["corr_position_ratio_with_lever_arm"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
