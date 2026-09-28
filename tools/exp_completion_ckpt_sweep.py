#!/usr/bin/env python3
"""Sweep every completion checkpoint on the columns the operator can SEE.

WHY THIS EXISTS.  The shipping completion is ``completion_step50400.pt`` -- 600
epochs over 5,329 overlapping 150-frame windows of ONE dancer.  It was picked
because at 3,360 steps the energy read 0.433x ground truth, i.e. on an
AMPLITUDE floor and nothing about diversity.  Nobody has ever swept the
completion.  The planner went through exactly this and it cost a week: its val
loss selected the most collapsed checkpoint (epoch 6, top class 42%, 20-second
holds) and epoch 12 was found only by an operator looking at renders.  This
sweep is the zero-GPU-training-cost version of that search for the completion:
if an earlier checkpoint is less collapsed, that is a free fix.

------------------------------------------------------ THE SHAPE, AND ITS BILL

Three defects paid for the shape of this file.  The first two killed an earlier
sweep (``scratchpad/opt/collapse/ckpt_sweep.py``, VOID); the third was found
while verifying this one and it reaches further than the sweep.

DEFECT 1 -- THE MASK WAS 4x OUT OF DISTRIBUTION.  ``draft_noise_mask`` is not a
gate, it is a NOISE SCALE.  Training multiplies the binary safe-draft mask by
``draft_noise_ratio`` (``train_atomic.py:463``, ``mask * self.noise_ratio``) and
inference does the same (``infer_atomic.py:2974``, ``noise_mask * ratio``), and
``AtomicCompletionDiffusion.perturb_draft`` then adds ``randn * mask``.  So the
model has only ever seen draft noise of sd 0.25.  The void sweep passed
``torch.ones``, i.e. sd 1.0, and every number it printed describes a regime the
model was never trained in.  THIS FILE reads ``draft_noise_ratio`` out of each
checkpoint's own saved ``args`` and asserts it is present; it never hardcodes a
value, and it prints the ratio per arm so a checkpoint trained at a different
ratio cannot be silently pooled with the rest.

DEFECT 2 -- THE WINDOW DRAW WAS THREE CLIPS.  The void sweep used
``range(0, 96, 8)``: consecutive release indices.  The release test split is 497
windows over 18 clips, laid out clip by clip (``names.json`` is
``<clip>_slice<N>``), so those 12 indices came from 3 clips, two of them
spinners, and its median was an artifact of that draw.  THIS FILE groups
``names.json`` by clip and draws an equal number of evenly spaced windows from
EVERY clip; ``--per-clip`` sets how many.  A draw covering fewer than
``--min-clips`` clips is refused rather than reported.

DEFECT 3 -- THE FACING ANGLE WAS MEASURED IN A VERTICAL PLANE.  In this
corpus's ``full_pose`` Z IS UP: root-relative head minus foot is
``[0.16, -0.26, 1.41]`` and the spine direction averages ``[0.06, 0.03, 0.99]``,
while the left-to-right hip vector lies in the ground plane (per-frame
|x| 0.073, |y| 0.083, |z| 0.010).  Pelvis azimuth is therefore
``atan2(v_y, v_x)`` -- which is what ``tools/score_facing_spin.facing_yaw``
uses.  The void sweep used ``atan2(v_x, v_z)``, and so, it turns out, did every
facing and twist number in ``docs/DANCE_QUALITY_DEFECTS.md`` sections 24 and 25
(this file reproduces them to the last digit under ``--legacy-plane-check``:
ground-truth range 110.4 deg, twist sd 10.32/19.19/1.62, shipping 17.8 deg/s,
18.9 deg, twist sd 4.59/12.47/0.83, draft 204.5 deg / 63.0 deg / 15.88).  That
angle is not azimuth: ``v_z`` is the near-zero VERTICAL component of the hip
line, so the reading is dominated by pelvic lateral tilt.

  THE CONTROL THAT SETTLES IT (2.1 rule 2, a positive control whose direction
  was checked).  Rotate an entire clip about the vertical axis by a constant
  angle.  Nothing about the dance changes; a turning measure must not move.
  On one ground-truth clip:

      plane                       constant world yaw   turn deg/s   range deg
      x-y  (azimuth, correct)     none                       58.7       456.0
      x-y  (azimuth, correct)     +45 deg                    58.7       456.0
      x-y  (azimuth, correct)     +90 deg                    58.7       456.0
      x-z  (sections 24/25)       none                       75.5       205.8
      x-z  (sections 24/25)       +45 deg                    42.9       183.4
      x-z  (sections 24/25)       +90 deg                    65.5       188.3

  The legacy plane's answer depends on where the camera was put.  It cannot
  be the instrument, so this file reports the CORRECT plane as its verdict
  column -- and carries the legacy column beside it, because every published
  number this sweep would otherwise contradict was measured in it, and 2.1
  rule 4 says two rulers that disagree get reported, not silently replaced.
  ``run_invariance_control`` re-runs the control above on the swept windows on
  every invocation, so the day the legacy column stops failing it, somebody
  finds out.

--------------------------------------------------------------------- COLUMNS

Everything is measured on the IDENTICAL windows for every arm, with a GROUND
TRUTH row and a FOREIGN-DRAFT row computed on those same windows, so each
column has a target rather than a direction.

  turn_deg_s    total |d azimuth| per second.  Ground truth is the target from
                both sides: a frozen body reads low and a spinner reads high.
  travel_deg    range of the unwrapped azimuth over the window: how much of a
                circle the dance actually visits.
  directed      travel / total turning.  A lively dancer oscillates and comes
                back (low); a slowly rotating statue accumulates (high).  This
                is ``score_facing_spin``'s ``net/tot`` with the range in place
                of the endpoint difference.
  spins         windows whose turn_deg_s exceeds ``--spin-threshold``.  The
                median hides this: the failure is bimodal, most windows frozen
                and a few spinning.
  spread        per-joint std over the window's frames, root-relative, mean
                over joints and axes: how much of the body's range is used.
  twist_sd      sd of the SIGNED shoulder-line-vs-hip-line angle per frame, no
                unwrap.  Section 25.1 already overturned the unwrapped form:
                unwrapping the two lines separately and subtracting inherits a
                2*pi jump at every seam, which is where "physically impossible
                115 deg torso twist" came from.
  still_share   share of windows containing a sustained hold, using
                ``tools/exp_guidance_stillness``'s verified ``low_pass`` +
                ``held_frames`` (its own synthetic controls pin the direction:
                constant velocity 0%, move-then-hold well above 0, and a
                coherent 5 Hz oscillation 0% after the low pass).
  noise_span    metres between two draws that differ ONLY in the sampler seed.
                The model's own stochasticity, which is what collapses first;
                a checkpoint whose noise_span is near zero is a deterministic
                function of its conditioning whatever else it scores.

Aggregation is the MEDIAN over windows for the continuous columns and a SHARE
for the two counting columns, and every per-window value is written to the JSON
so a median can be re-derived or challenged.
"""
import argparse
import glob
import json
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

FPS = 30.0
LEFT_HIP, RIGHT_HIP = 1, 2
LEFT_SHOULDER, RIGHT_SHOULDER = 16, 17
SMOOTH_WIDTH = 9
# The plane the correct azimuth lives in, and the one sections 24/25 used.
GROUND_PLANE = (0, 1)      # atan2(v_y, v_x): Z is up in this corpus
LEGACY_PLANE = (2, 0)      # atan2(v_x, v_z): contains the VERTICAL axis
SLICE_SUFFIX = re.compile(r"_slice\d+$")


# --------------------------------------------------------------- the columns

def low_pass(joints, width=SMOOTH_WIDTH):
    """Moving average over joint positions.

    Imported behaviour, not reinvented: ``tools/stillness_criterion.low_pass``,
    the one definition the stillness tools share, so the stillness column and
    the facing columns see the same body.
    """
    from tools.stillness_criterion import low_pass as reference
    return reference(joints, width)


def root_relative(joints):
    joints = np.asarray(joints, float)
    return joints - joints[:, :1, :]


def facing_yaw(joints, plane=GROUND_PLANE):
    """Unwrapped facing angle of the left-to-right hip vector, radians."""
    first, second = plane
    across = np.asarray(joints, float)[:, RIGHT_HIP, :] - np.asarray(joints, float)[:, LEFT_HIP, :]
    return np.unwrap(np.arctan2(across[:, second], across[:, first]))


def facing_columns(joints, plane=GROUND_PLANE):
    """(turn deg/s, travel deg, directedness) for one window."""
    yaw = facing_yaw(joints, plane)
    if len(yaw) < 2:
        return float("nan"), float("nan"), float("nan")
    total = float(np.degrees(np.abs(np.diff(yaw))).sum())
    seconds = (len(yaw) - 1) / FPS
    travel = float(np.degrees(yaw.max() - yaw.min()))
    return total / seconds, travel, (travel / total if total > 0 else float("nan"))


def spread(joints):
    """Per-joint positional std over the window, mean over joints and axes."""
    return float(root_relative(joints).std(axis=0).mean())


def twist_sd(joints, plane=GROUND_PLANE):
    """sd of the signed shoulder-vs-hip angle, degrees, NO unwrap.

    The signed angle between two 2-D lines is naturally in (-180, 180], so no
    unwrapping is needed -- and section 25.1 measured what unwrapping the two
    lines separately does at a seam: it manufactures a 2*pi step that the
    difference inherits, which is the whole of the retracted "115 deg torso
    twist" reading.
    """
    first, second = plane
    relative = root_relative(joints)
    hips = relative[:, RIGHT_HIP] - relative[:, LEFT_HIP]
    shoulders = relative[:, RIGHT_SHOULDER] - relative[:, LEFT_SHOULDER]
    cross = hips[:, first] * shoulders[:, second] - hips[:, second] * shoulders[:, first]
    dot = hips[:, first] * shoulders[:, first] + hips[:, second] * shoulders[:, second]
    return float(np.degrees(np.arctan2(cross, dot)).std())


def has_sustained_hold(joints, smooth=SMOOTH_WIDTH):
    """Does this window contain a held frame, or None if it cannot be asked?

    ``tools/stillness_criterion`` REFUSES a window whose median smoothed speed
    has collapsed, because the criterion's threshold is a fraction of that
    median: at zero the threshold is zero, ``speed < 0`` is unsatisfiable, and a
    motionless body would read "no stillness" -- inverted, not small.  This
    sweep is looking for collapse, so it is exactly the input that would arrive.
    A refusal is carried through as ``None`` and counted separately rather than
    folded into the share; a checkpoint whose windows are unmeasurable is
    reported as such, not scored 0.
    """
    from tools.stillness_criterion import DegenerateMotion, held_frames
    from tools.stillness_criterion import joint_speed
    from tools.stillness_criterion import low_pass as still_low_pass
    try:
        return bool(held_frames(joint_speed(still_low_pass(joints, smooth))).any())
    except DegenerateMotion:
        return None


def window_columns(joints, smooth=SMOOTH_WIDTH):
    """Every per-window column for one decoded window of joints [T, 24, 3]."""
    smoothed = low_pass(joints, smooth)
    turn, travel, directed = facing_columns(smoothed, GROUND_PLANE)
    legacy_turn, legacy_travel, legacy_directed = facing_columns(smoothed, LEGACY_PLANE)
    return {
        "turn_deg_s": turn,
        "travel_deg": travel,
        "directed": directed,
        "spread": spread(joints),
        "twist_sd": twist_sd(joints, GROUND_PLANE),
        "legacy_turn_deg_s": legacy_turn,
        "legacy_travel_deg": legacy_travel,
        "legacy_directed": legacy_directed,
        "legacy_twist_sd": twist_sd(joints, LEGACY_PLANE),
        "still": has_sustained_hold(joints, smooth),
    }


CONTINUOUS = ("turn_deg_s", "travel_deg", "directed", "spread", "twist_sd",
              "legacy_turn_deg_s", "legacy_travel_deg", "legacy_directed",
              "legacy_twist_sd")


def aggregate(rows, spin_threshold, noise_spans=None):
    """Median over windows for the continuous columns, share for the counts."""
    summary = {key: float(np.median([row[key] for row in rows])) for key in CONTINUOUS}
    summary["spins"] = int(sum(row["turn_deg_s"] > spin_threshold for row in rows))
    summary["windows"] = len(rows)
    measurable = [row["still"] for row in rows if row["still"] is not None]
    summary["still_share"] = float(np.mean(measurable)) if measurable else float("nan")
    summary["still_unmeasurable"] = len(rows) - len(measurable)
    summary["noise_span"] = (float(np.median(noise_spans))
                             if noise_spans is not None else None)
    return summary


# ------------------------------------------------------------ the window draw

def clip_of(name):
    return SLICE_SUFFIX.sub("", name)


def stratified_windows(names, per_clip):
    """``per_clip`` evenly spaced release indices from EVERY clip.

    Even spacing rather than a random draw so the sweep is reproducible without
    carrying a seed, and so no clip can contribute more windows than another --
    the release holds 14 to 38 windows per clip, and a flat random draw over
    497 indices would weight the long clips almost 3:1.
    """
    if per_clip < 1:
        raise ValueError("--per-clip must be at least 1")
    order = {}
    for index, name in enumerate(names):
        order.setdefault(clip_of(name), []).append(index)
    chosen = []
    for clip in sorted(order):
        indices = order[clip]
        if len(indices) < per_clip:
            picks = list(range(len(indices)))
        else:
            picks = np.unique(np.round(
                np.linspace(0, len(indices) - 1, per_clip)).astype(int)).tolist()
        chosen.extend(indices[position] for position in picks)
    return sorted(chosen)


# ------------------------------------------------------------------ decoding

def decode_batch(motion, normalizer_path, skeleton=None):
    """``infer_atomic.decode_motion``'s full_pose for a BATCH [B, T, 151].

    Same arithmetic, one forward-kinematics call instead of B of them; the unit
    test pins it against ``decode_motion`` window by window, because a decoder
    that quietly disagreed with the one every other tool uses would put this
    whole sweep in a different space from the numbers it is compared against.
    """
    import torch

    from dataset.quaternion import ax_from_6v
    from infer_atomic import CONTACT_CHANNELS, ROOT_POSITION_DIMS, unnormalize_motion
    from vis import SMPLSkeleton

    motion = torch.as_tensor(motion, dtype=torch.float32)
    if motion.ndim != 3 or motion.shape[-1] != 151:
        raise ValueError("expected [B, T, 151] normalized motion, got {}".format(
            tuple(motion.shape)))
    values = unnormalize_motion(motion, normalizer_path)[..., CONTACT_CHANNELS:]
    root = values[..., :ROOT_POSITION_DIMS]
    rotations = ax_from_6v(values[..., ROOT_POSITION_DIMS:].reshape(-1, 24, 6))
    rotations = rotations.reshape(motion.shape[0], motion.shape[1], 24, 3)
    skeleton = skeleton or SMPLSkeleton()
    return skeleton.forward(rotations, root).numpy()


# ---------------------------------------------------------------- the controls

def run_invariance_control(joints):
    """Rotate a window about the vertical axis; a turning measure must not move.

    Returns the two planes' readings at 0, 45 and 90 degrees of constant world
    yaw.  This is the control that disqualified the legacy plane (see the
    module docstring) and it runs on every invocation rather than being quoted,
    so the claim stays falsifiable on whatever windows are swept.
    """
    out = {}
    frames = len(joints)
    for label, degrees in (("0", 0.0), ("45", 45.0), ("90", 90.0)):
        angle = np.full(frames, np.radians(degrees))
        cos, sin = np.cos(angle), np.sin(angle)
        rotation = np.zeros((frames, 3, 3))
        rotation[:, 0, 0] = cos
        rotation[:, 0, 1] = -sin
        rotation[:, 1, 0] = sin
        rotation[:, 1, 1] = cos
        rotation[:, 2, 2] = 1.0
        turned = np.einsum("tij,tkj->tki", rotation, np.asarray(joints, float))
        out["ground_plane_" + label] = facing_columns(turned, GROUND_PLANE)[0]
        out["legacy_plane_" + label] = facing_columns(turned, LEGACY_PLANE)[0]
    out["ground_plane_invariant"] = bool(
        abs(out["ground_plane_0"] - out["ground_plane_90"]) < 1e-6
        and abs(out["ground_plane_0"] - out["ground_plane_45"]) < 1e-6)
    out["legacy_plane_invariant"] = bool(
        abs(out["legacy_plane_0"] - out["legacy_plane_90"]) < 1e-6
        and abs(out["legacy_plane_0"] - out["legacy_plane_45"]) < 1e-6)
    return out


def legacy_plane_check(ground_truth_dir, shipping_dir, clips):
    """Reproduce sections 24/25's whole-clip facing numbers, in both planes.

    The point is not the numbers, it is provenance: if this file's legacy
    column reproduces the published readings exactly, then the two planes'
    disagreement is a property of the PLANE and not of a re-implementation.
    """
    import pickle
    out = {}
    for label, directory in (("ground_truth", ground_truth_dir),
                             ("shipping", shipping_dir)):
        if not directory:
            continue
        rows = []
        for clip in clips:
            path = pathlib.Path(directory) / (clip + ".pkl")
            if not path.is_file():
                continue
            with open(path, "rb") as handle:
                joints = np.asarray(pickle.load(handle)["full_pose"], float)
            turn, travel, directed = facing_columns(joints, GROUND_PLANE)
            legacy = facing_columns(joints, LEGACY_PLANE)
            rows.append((turn, travel, directed, legacy[0], legacy[1], legacy[2],
                         twist_sd(joints, LEGACY_PLANE)))
        if not rows:
            continue
        median = np.median(np.asarray(rows), axis=0)
        out[label] = {
            "clips": len(rows),
            "turn_deg_s": float(median[0]), "travel_deg": float(median[1]),
            "directed": float(median[2]),
            "legacy_turn_deg_s": float(median[3]),
            "legacy_travel_deg": float(median[4]),
            "legacy_directed": float(median[5]),
            "legacy_twist_sd": float(median[6]),
        }
    return out


def stitch(windows, stride, blend_width):
    """Overlap-add the pipeline's way, in normalized motion space.

    Uses ``infer_atomic._blend_weights`` itself rather than a copy, so if the
    shipped blender changes this check changes with it.  ``windows`` are
    consecutive fixed-stride windows of one clip, already generated.
    """
    import torch
    from infer_atomic import _blend_weights
    windows = torch.as_tensor(windows, dtype=torch.float32)
    count, size, dim = windows.shape
    length = (count - 1) * stride + size
    output = torch.zeros(length, dim)
    total = torch.zeros(length, 1)
    overlap = size - stride
    for index in range(count):
        weights = _blend_weights(size, is_first=index == 0,
                                 is_last=index == count - 1,
                                 overlap=overlap, blend_width=blend_width)
        start = index * stride
        output[start:start + size] += windows[index] * weights
        total[start:start + size] += weights
    return output / total.clamp(min=1e-8)


def stitch_check(model, motion, music, names, ratio, args, device, decode):
    """Is the missing facing travel the MODEL's or the STITCHER's?

    THE QUESTION.  Section 25 reports the shipping arm's whole-CLIP facing
    travel at 18.9 deg against a ground truth of 110.4 (both in the legacy
    plane).  But a whole clip is never sampled as a clip: it is assembled from
    independently drawn 150-frame windows at stride 75.  If single windows
    already carry ground truth's share of the travel, then what is missing was
    removed when they were reassembled, and no checkpoint choice can bring it
    back.

    THE CONTROL THAT MAKES IT READABLE.  The identical stitch is applied to the
    GROUND TRUTH windows.  Consecutive release slices of one clip agree exactly
    where they overlap (release stride 15 frames, verified against
    ``motion.npy``), so stitching them must return the original frames to
    floating point -- ``control_max_abs_error`` is that residual and it is the
    proof the blender itself destroys nothing.  If ground truth survives the
    stitch and the model's windows do not, the loss is in reassembling
    INDEPENDENT draws, not in the blender's arithmetic.
    """
    stride_slices = args.stitch_stride // args.release_slice_stride
    results = []
    clips = sorted({clip_of(name) for name in names})
    for clip in clips[:args.stitch_check]:
        release = [index for index, name in enumerate(names) if clip_of(name) == clip]
        chosen = release[::stride_slices]
        if len(chosen) < 3:
            continue
        window_motion = motion[chosen]
        window_music = music[chosen]
        generated = sample_windows(model, window_music, window_motion, ratio,
                                   args.guidance_weight, args.seed, device,
                                   args.batch_size)
        stitched = stitch(generated, args.stitch_stride, args.stitch_blend_width)
        control = stitch(window_motion, args.stitch_stride, args.stitch_blend_width)
        reference = reference_span(window_motion, args.stitch_stride)
        joints_stitched = decode(stitched.unsqueeze(0))[0]
        joints_control = decode(control.unsqueeze(0))[0]
        per_window = [window_columns(frame) for frame in decode(generated)]
        truth_window = [window_columns(frame) for frame in decode(window_motion)]

        def facing(joints):
            return dict(zip(("turn_deg_s", "travel_deg", "directed"),
                            facing_columns(low_pass(joints))))

        def median_of(rows):
            return {key: float(np.median([row[key] for row in rows]))
                    for key in ("turn_deg_s", "travel_deg", "directed")}

        results.append({
            "clip": clip,
            "windows": len(chosen),
            "frames": int(len(stitched)),
            "generated_stitched": facing(joints_stitched),
            "generated_window_median": median_of(per_window),
            "ground_truth_stitched": facing(joints_control),
            "ground_truth_window_median": median_of(truth_window),
            "control_max_abs_error": float(
                (control - reference).abs().max().item()),
        })
    return results


def reference_span(window_motion, stride):
    """The original frames a stitch of overlapping GT windows must reproduce."""
    import torch
    pieces = [window_motion[0]]
    for index in range(1, len(window_motion)):
        pieces.append(window_motion[index][-stride:])
    return torch.cat(pieces, dim=0)


# -------------------------------------------------------------------- the sweep

def checkpoint_paths(arm_dirs, every):
    """Every ``completion_epoch*_step*.pt`` per arm, optionally subsampled.

    ``completion_step50400.pt`` is NOT listed separately: it is byte-different
    from ``completion_epoch600_step50400.pt`` (it carries different optimizer
    state) but its ``model`` tensors are equal element for element in all five
    arms, verified before this sweep ran.  Listing it would add a duplicate row
    that reads as a second measurement.  ``--alias-check`` sweeps it anyway for
    one arm and asserts the row is identical, which is the determinism control
    for the whole sweep.
    """
    jobs = []
    for directory in arm_dirs:
        found = sorted(
            glob.glob(str(pathlib.Path(directory) / "completion_epoch*_step*.pt")),
            key=lambda path: int(path.split("_step")[1].split(".")[0]))
        if every > 1:
            found = found[::every]
        if not found:
            raise FileNotFoundError("no completion checkpoints under " + str(directory))
        jobs.extend(found)
    return jobs


def sample_windows(model, music, draft, ratio, guidance, seed, device, batch_size):
    """One draw from the model for every window, mask = ones * ratio."""
    import torch
    outputs = []
    for start in range(0, len(music), batch_size):
        stop = min(start + batch_size, len(music))
        music_batch = music[start:stop].to(device)
        draft_batch = draft[start:stop].to(device)
        mask = torch.full((stop - start, draft.shape[1], 1), float(ratio),
                          device=device)
        torch.manual_seed(seed + start)
        torch.cuda.manual_seed_all(seed + start)
        with torch.no_grad():
            outputs.append(model.sample(music_batch, draft_batch, mask,
                                        guidance_weight=guidance).cpu())
    return torch.cat(outputs, dim=0)


def measure(joints_batch, spin_threshold, noise_spans=None):
    rows = [window_columns(joints_batch[index]) for index in range(len(joints_batch))]
    return rows, aggregate(rows, spin_threshold, noise_spans)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release", default="/cache/atomicdance-assets/scratch/txy_t/release_v3")
    parser.add_argument("--split", default="test")
    parser.add_argument("--arm", action="append", default=[], required=True,
                        help="directory holding completion_epoch*_step*.pt")
    parser.add_argument("--per-clip", type=int, default=3,
                        help="windows drawn from EVERY clip (evenly spaced)")
    parser.add_argument("--min-clips", type=int, default=15,
                        help="a draw covering fewer clips than this is refused")
    parser.add_argument("--every", type=int, default=1,
                        help="keep every Nth checkpoint per arm; >1 is recorded "
                             "in the output as a subsampled sweep")
    parser.add_argument("--guidance-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--noise-seed", type=int, default=90000,
                        help="second sampler seed, for noise_span")
    parser.add_argument("--spin-threshold", type=float, default=100.0)
    parser.add_argument("--batch-size", type=int, default=54)
    parser.add_argument("--foreign-draft-check", action="store_true",
                        help="also run the LAST checkpoint of the first arm with "
                             "a draft from a different clip, the condition the "
                             "shipping pipeline actually runs in")
    parser.add_argument("--alias-check", action="store_true",
                        help="re-sweep the first arm's completion_step50400.pt "
                             "and assert it reproduces the epoch600 row")
    parser.add_argument("--legacy-plane-check", nargs=3,
                        metavar=("CLIPS", "GROUND_TRUTH_DIR", "SHIPPING_DIR"),
                        help="reproduce sections 24/25's whole-clip numbers")
    parser.add_argument("--last-only", action="store_true",
                        help="sweep only the final checkpoint of each arm")
    parser.add_argument("--stitch-check", type=int, default=0, metavar="N",
                        help="after the sweep, re-assemble N clips from the "
                             "shipping checkpoint's own windows the way the "
                             "pipeline does, to split MODEL from STITCHER")
    parser.add_argument("--stitch-stride", type=int, default=75,
                        help="shipping --completion-stride")
    parser.add_argument("--stitch-blend-width", type=int, default=10,
                        help="shipping --completion-blend-width")
    parser.add_argument("--release-slice-stride", type=int, default=15,
                        help="frames between consecutive release slices")
    parser.add_argument("--out-dir", default="runs/opt_ckptsweep")
    args = parser.parse_args()

    import torch
    from infer_atomic import _load_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    release = pathlib.Path(args.release)
    split = release / args.split
    names = json.load(open(str(split / "names.json")))
    indices = stratified_windows(names, args.per_clip)
    clips = sorted({clip_of(names[index]) for index in indices})
    if len(clips) < args.min_clips:
        raise SystemExit("draw covers only {} clips (< --min-clips {}); refusing"
                         .format(len(clips), args.min_clips))

    motion_all = np.load(str(split / "motion.npy"), mmap_mode="r")
    music_all = np.load(str(split / "music.npy"), mmap_mode="r")
    normalizer = str(release / "normalizer.pt")
    motion = torch.tensor(np.asarray(motion_all[indices]), dtype=torch.float32)
    music = torch.tensor(np.asarray(music_all[indices]), dtype=torch.float32)

    from vis import SMPLSkeleton
    skeleton = SMPLSkeleton()

    def decode(batch):
        return decode_batch(batch, normalizer, skeleton)

    ground_truth_joints = decode(motion)
    report = {
        "release": str(release), "split": args.split,
        "window_indices": indices,
        "window_names": [names[index] for index in indices],
        "clips_covered": clips,
        "windows_used": len(indices),
        "per_clip": args.per_clip,
        "guidance_weight": args.guidance_weight,
        "seeds": [args.seed, args.noise_seed],
        "spin_threshold": args.spin_threshold,
        "checkpoint_subsampling_every": args.every,
        "subsampled": args.every > 1,
        "arms": [], "rows": [], "per_window": {},
        "argv": sys.argv,
    }

    # ---- controls first, so a broken instrument stops the sweep -------------
    report["invariance_control"] = run_invariance_control(ground_truth_joints[0])
    if not report["invariance_control"]["ground_plane_invariant"]:
        raise SystemExit("ground-plane facing is not invariant to world yaw; "
                         "the instrument is broken, not the checkpoints")
    if args.legacy_plane_check:
        clip_file, ground_truth_dir, shipping_dir = args.legacy_plane_check
        whole = [line.strip() for line in open(clip_file) if line.strip()]
        report["legacy_plane_check"] = legacy_plane_check(
            ground_truth_dir, shipping_dir, whole)

    # ---- reference rows on the identical windows ---------------------------
    def add_row(label, joints, extra=None, noise_spans=None):
        rows, summary = measure(joints, args.spin_threshold, noise_spans)
        summary["arm"] = label
        summary.update(extra or {})
        report["rows"].append(summary)
        report["per_window"][label] = rows
        return summary

    add_row("GROUND TRUTH (same windows)", ground_truth_joints,
            {"draft_noise_ratio": None, "kind": "reference"})

    # A draft from a DIFFERENT clip, built by rolling the window list past every
    # clip boundary: the shipping pipeline never retrieves from the query's own
    # retrieval group, so a same-clip draft would flatter every arm equally and
    # tell nothing about the condition the model ships in.
    clip_index = np.array([clips.index(clip_of(names[index])) for index in indices])
    foreign = np.zeros(len(indices), int)
    for position in range(len(indices)):
        step = 1
        while clip_index[(position + step) % len(indices)] == clip_index[position]:
            step += 1
        foreign[position] = (position + step) % len(indices)
    foreign_draft = motion[foreign]
    add_row("FOREIGN DRAFT (input itself)", decode(foreign_draft),
            {"draft_noise_ratio": None, "kind": "reference"})

    # ---- the sweep ---------------------------------------------------------
    jobs = checkpoint_paths(args.arm, args.every)
    if args.last_only:
        last = {}
        for job in jobs:
            last[pathlib.Path(job).parent.name] = job
        jobs = list(last.values())
    # The foreign-draft self-check runs on the SHIPPING checkpoint -- the
    # first arm's last one -- because that is the only row with published
    # whole-clip numbers to be checked against.
    foreign_draft_on = [job for job in jobs
                        if pathlib.Path(job).parent.name == pathlib.Path(args.arm[0]).name][-1]
    if args.alias_check:
        jobs.append(str(pathlib.Path(args.arm[0]) / "completion_step50400.pt"))
    for path in jobs:
        model, model_args = _load_checkpoint(path, "completion", device)
        ratio = getattr(model_args, "draft_noise_ratio", None)
        if ratio is None:
            raise SystemExit("{} has no draft_noise_ratio in its saved args; "
                             "refusing to guess the mask scale".format(path))
        first = sample_windows(model, music, motion, ratio, args.guidance_weight,
                               args.seed, device, args.batch_size)
        second = sample_windows(model, music, motion, ratio, args.guidance_weight,
                                args.noise_seed, device, args.batch_size)
        joints_first, joints_second = decode(first), decode(second)
        spans = [float(np.linalg.norm(
            root_relative(joints_first[index]) - root_relative(joints_second[index]),
            axis=-1).mean()) for index in range(len(indices))]
        label = "{}/{}".format(pathlib.Path(path).parent.name,
                               pathlib.Path(path).stem)
        summary = add_row(label, joints_first,
                          {"draft_noise_ratio": float(ratio),
                           "kind": "checkpoint",
                           "arm_dir": str(pathlib.Path(path).parent),
                           "checkpoint": str(path)}, spans)
        print("{:<62} ratio={:.3f} turn={:6.1f} travel={:7.1f} dir={:.3f} "
              "spins={:2d} spread={:.4f} twist={:5.2f} still={:.2f} "
              "span={:.4f}".format(
                  label, ratio, summary["turn_deg_s"], summary["travel_deg"],
                  summary["directed"], summary["spins"], summary["spread"],
                  summary["twist_sd"], summary["still_share"],
                  summary["noise_span"]), flush=True)
        if args.foreign_draft_check and path == foreign_draft_on:
            other = sample_windows(model, music, foreign_draft, ratio,
                                   args.guidance_weight, args.seed, device,
                                   args.batch_size)
            add_row(label + " [FOREIGN DRAFT]", decode(other),
                    {"draft_noise_ratio": float(ratio), "kind": "self-check",
                     "checkpoint": str(path)})
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    if args.alias_check:
        alias = report["rows"][-1]
        epoch600 = [row for row in report["rows"]
                    if row["arm"].endswith("completion_epoch600_step50400")
                    and row["arm"].startswith(pathlib.Path(args.arm[0]).name)]
        if epoch600:
            report["alias_check"] = {
                "epoch600": epoch600[0]["directed"], "final": alias["directed"],
                "identical": bool(abs(epoch600[0]["directed"] - alias["directed"]) < 1e-12),
            }

    if args.stitch_check:
        all_motion = torch.tensor(np.asarray(motion_all), dtype=torch.float32)
        all_music = torch.tensor(np.asarray(music_all), dtype=torch.float32)
        model, model_args = _load_checkpoint(foreign_draft_on, "completion", device)
        report["stitch_check"] = {
            "checkpoint": foreign_draft_on,
            "stride": args.stitch_stride,
            "blend_width": args.stitch_blend_width,
            "clips": stitch_check(model, all_motion, all_music, names,
                                  float(model_args.draft_noise_ratio), args,
                                  device, decode),
        }
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sweep.json").write_text(json.dumps(report, indent=2))
    (out_dir / "sweep.md").write_text(render_markdown(report))
    print("wrote {} and {}".format(out_dir / "sweep.json", out_dir / "sweep.md"))


def render_markdown(report):
    header = ("| arm | ratio | turn deg/s | travel deg | directed | spins | "
              "spread | twist sd | still | noise span | legacy turn | legacy dir |")
    rule = "|" + "---|" * 12
    lines = [
        "# Completion checkpoint sweep",
        "",
        "windows {} over {} clips ({} per clip, evenly spaced); guidance {}; "
        "seeds {}; spin threshold {} deg/s{}".format(
            report["windows_used"], len(report["clips_covered"]),
            report["per_clip"], report["guidance_weight"], report["seeds"],
            report["spin_threshold"],
            "; SUBSAMPLED every {}th checkpoint".format(
                report["checkpoint_subsampling_every"])
            if report.get("subsampled") else ""),
        "",
        header, rule,
    ]
    for row in report["rows"]:
        ratio = "-" if row.get("draft_noise_ratio") is None else "{:.2f}".format(
            row["draft_noise_ratio"])
        span = "-" if row.get("noise_span") is None else "{:.4f}".format(row["noise_span"])
        lines.append("| {} | {} | {:.1f} | {:.1f} | {:.3f} | {} | {:.4f} | {:.2f} | "
                     "{:.2f} | {} | {:.1f} | {:.3f} |".format(
                         row["arm"], ratio, row["turn_deg_s"], row["travel_deg"],
                         row["directed"], row["spins"], row["spread"],
                         row["twist_sd"], row["still_share"], span,
                         row["legacy_turn_deg_s"], row["legacy_directed"]))
    control = report.get("invariance_control", {})
    if control:
        lines += ["", "## World-yaw invariance control",
                  "", "| plane | 0 deg | 45 deg | 90 deg | invariant |",
                  "|---|---|---|---|---|"]
        for plane in ("ground_plane", "legacy_plane"):
            lines.append("| {} | {:.2f} | {:.2f} | {:.2f} | {} |".format(
                plane, control[plane + "_0"], control[plane + "_45"],
                control[plane + "_90"], control[plane + "_invariant"]))
    stitch = report.get("stitch_check")
    if stitch:
        lines += ["", "## Model or stitcher?  ({}, stride {}, blend width {})".format(
            pathlib.Path(stitch["checkpoint"]).stem, stitch["stride"],
            stitch["blend_width"]),
            "",
            "Windows are drawn independently and overlap-added.  The ground-truth "
            "rows are the control: consecutive release slices agree exactly where "
            "they overlap, so a stitch of them must return the original frames "
            "(`control err`).",
            "",
            "| clip | windows | gen stitched travel | gen window median travel | "
            "gen stitched dir | gen window median dir | GT stitched travel | "
            "GT window median travel | control err |",
            "|" + "---|" * 9]
        for row in stitch["clips"]:
            lines.append("| {} | {} | {:.1f} | {:.1f} | {:.3f} | {:.3f} | {:.1f} | "
                         "{:.1f} | {:.2e} |".format(
                             row["clip"], row["windows"],
                             row["generated_stitched"]["travel_deg"],
                             row["generated_window_median"]["travel_deg"],
                             row["generated_stitched"]["directed"],
                             row["generated_window_median"]["directed"],
                             row["ground_truth_stitched"]["travel_deg"],
                             row["ground_truth_window_median"]["travel_deg"],
                             row["control_max_abs_error"]))
    legacy = report.get("legacy_plane_check")
    if legacy:
        lines += ["", "## Whole-clip reproduction of sections 24/25",
                  "", "| arm | clips | turn (ground plane) | travel (ground plane) "
                  "| dir (ground plane) | turn (legacy) | travel (legacy) | "
                  "dir (legacy) | twist sd (legacy) |",
                  "|" + "---|" * 9]
        for label, row in legacy.items():
            lines.append("| {} | {} | {:.1f} | {:.1f} | {:.3f} | {:.1f} | {:.1f} | "
                         "{:.3f} | {:.2f} |".format(
                             label, row["clips"], row["turn_deg_s"],
                             row["travel_deg"], row["directed"],
                             row["legacy_turn_deg_s"], row["legacy_travel_deg"],
                             row["legacy_directed"], row["legacy_twist_sd"]))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
