"""Is the draft "sometimes fast and sometimes slow", and is the RESAMPLE why?

WHAT THE OPERATOR SAW (2026-09-04, ``output/LATEST/...__clip000_zoom.mp4``):
the retrieval draft's rhythm is inconsistent within one clip -- fast here, slow
there -- and the question asked was whether the library's moves are at
inconsistent frame rates.

WHAT THIS TOOL SEPARATES.  Three different claims live under that one sentence
and they need three different readings:

  1. ``s = slot_frames / native_frames`` -- the ONE global linear resample
     ``IndexedAtomicMotionLibrary._values_at`` applies to the chosen prototype.
     A prototype stretched by ``s`` plays back at ``1/s`` of its own speed
     (``tools/measure_retrieval_stretch.py`` derives it; this tool checks that
     derivation against the motion instead of assuming it).
  2. Whether ``s`` VARIES BETWEEN ADJACENT SEGMENTS of one clip.  A clip whose
     every segment is stretched by the same factor is uniformly slow, not
     erratic; "sometimes fast, sometimes slow" is a statement about neighbours
     disagreeing, so the pooled spread of ``s`` cannot answer it and the
     per-clip spread can.
  3. What the eye actually gets: the per-segment mean joint speed of the draft
     and how much it jumps at a segment boundary -- measured against GROUND
     TRUTH read at the same frame positions, which is the control, because
     ground truth is one continuous recording with no seams and ``s = 1``
     everywhere by construction.

PROVENANCE.  Reading 1 is not invented here: it is ``measure_retrieval_stretch``
(2026-09-01), and this tool imports ``chosen_candidate`` from it rather than
re-deriving the pick, so the two cannot drift.  Readings 2 and 3 are new on
2026-09-05 and are stated as such.

THE POSITIVE CONTROL, and why a null needs one (CLAUDE.md 2.1 rule 3).  If the
duration rule turns out to keep ``s`` near 1, "the resample is not the cause" is
a NULL, and a null is worthless until the measurement is shown to have power.
``--positive-control uniform`` re-runs every reading with the candidate drawn
UNIFORMLY from the very same pool -- the paper's ``Random Choice`` ablation --
which by construction stretches hard and unevenly.  Same clips, same slots, same
statistics: if the within-clip spread lights up there and not on the shipped
rule, the instrument can see the effect and the shipped rule does not have it.

THE MECHANISM CHECK (positive control for reading 1).  ``native_speed_ratio`` is
the chosen prototype's own mean joint speed divided by ``s`` times the speed the
draft actually plays it at.  It must read 1.0.  If it does not, ``s`` is not the
playback multiplier this tool claims it is, and nothing below can be believed.

WHAT IT REFUSES.  A clip with fewer than two atomic segments has no "adjacent
segments" and gets ``None`` for every within-clip column rather than a 0.0 --
zero spread and no spread to measure are different claims, and only one of them
is evidence that the draft is steady.
"""

import argparse
import json
import pathlib
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dataset.atomic import labels_to_segments  # noqa: E402
from infer_atomic import IndexedAtomicMotionLibrary, decode_motion  # noqa: E402
from tools.measure_retrieval_stretch import chosen_candidate, safe_candidates  # noqa: E402
from tools.stillness_criterion import FPS, joint_speed  # noqa: E402

# Frames trimmed from each end of a segment before its speed is averaged.  The
# shipped draft cross-fades +-4 frames at a prototype-to-prototype seam
# (``--draft-seam-blend 4``), so those frames are a blend of two prototypes and
# belong to neither segment's tempo.  5 = 4 + 1.
SEAM_TRIM = 5
# Frames used to read the sustained speed LEVEL on each side of a boundary.
# Short enough to be inside the shortest segment kept below, long enough that a
# single frame cannot set the level.
LEVEL_FRAMES = 5
# A segment shorter than this has no interior left after trimming both ends.
MIN_SEGMENT_FRAMES = 2 * SEAM_TRIM + LEVEL_FRAMES


def within_clip_stretch(stretches):
    """Spread of ``s`` inside ONE clip -- the operator's actual complaint.

    ``None`` when fewer than two segments: see the module docstring's refusal.
    ``sd_log`` is the population sd of ``ln s`` (log because the quantity is a
    ratio: 2x and 0.5x are the same distortion in opposite directions),
    ``max_over_min`` is the fastest-to-slowest playback ratio inside the clip,
    and ``adjacent_mean_abs_log`` is the mean of ``|ln s_next - ln s_prev|``
    over CONSECUTIVE segments, which is the one that says whether neighbours
    disagree rather than whether the clip as a whole is varied.
    """
    values = np.asarray(list(stretches), float)
    if values.size < 2 or not np.all(values > 0):
        return None
    logs = np.log(values)
    return {
        "segments": int(values.size),
        "sd_log": float(logs.std()),
        "max_over_min": float(values.max() / values.min()),
        "adjacent_mean_abs_log": float(np.abs(np.diff(logs)).mean()),
        "adjacent_max_abs_log": float(np.abs(np.diff(logs)).max()),
    }


def speed_levels(speed, segments):
    """Sustained speed at the start and end of each segment, and its interior.

    ``speed[i]`` is the sample BETWEEN frame ``i`` and ``i+1`` -- the off-by-one
    that cost 2026-08-19 a day (CLAUDE.md 2.2), so it is written down here: for
    a segment ``[start, end)`` the interior samples are ``speed[start:end-1]``
    and ``speed[end-1]`` is the sample that STRADDLES the seam into the next
    segment.  The straddling sample is never counted as either side's level.
    """
    rows = []
    for segment in segments:
        interior = speed[segment.start:max(segment.start, segment.end - 1)]
        if len(interior) < MIN_SEGMENT_FRAMES:
            rows.append(None)
            continue
        trimmed = interior[SEAM_TRIM:len(interior) - SEAM_TRIM]
        if len(trimmed) < 1:
            rows.append(None)
            continue
        rows.append({
            "mean": float(trimmed.mean()),
            "head": float(np.median(trimmed[:LEVEL_FRAMES])),
            "tail": float(np.median(trimmed[-LEVEL_FRAMES:])),
        })
    return rows


def boundary_jumps(speed, segments, levels):
    """Per-boundary readings, in the two units that must not be conflated.

    ``level_log_jump``  -- ``|ln(next head / previous tail)|``: the change in
    SUSTAINED speed across the boundary, i.e. "the next move is played faster
    or slower than this one".  This is what a varying ``s`` would produce.
    ``step_over_median`` -- the single straddling sample ``speed[end-1]``
    divided by the clip's median speed: the one-frame POP of pasting two
    prototypes together.  A big pop with no level change is a seam artefact,
    not a tempo change, and the two are reported apart on purpose.
    """
    median = float(np.median(speed)) if len(speed) else 0.0
    rows = []
    for index in range(len(segments) - 1):
        left, right = levels[index], levels[index + 1]
        frame = int(segments[index].end)
        step = (float(speed[frame - 1]) if 0 < frame <= len(speed) else None)
        rows.append({
            "frame": frame,
            "left_label": int(segments[index].label),
            "right_label": int(segments[index + 1].label),
            "level_log_jump": (None if not left or not right
                               or left["tail"] <= 0 or right["head"] <= 0
                               else float(abs(np.log(right["head"] / left["tail"])))),
            "step_over_median": (None if step is None or median <= 0
                                 else float(step / median)),
        })
    return rows


def native_mean_speed(library, candidate, normalizer_path):
    """The chosen prototype's OWN mean joint speed, before any resample."""
    sample, start, end = candidate[0], candidate[1], candidate[2]
    window = torch.from_numpy(
        np.array(library.motion[sample, start:end], copy=True)).float()
    if len(window) < 3:
        return None
    pose = decode_motion(window, normalizer_path)["full_pose"]
    return float(joint_speed(pose).mean())


def measure_clip(library, plan_dir, draft_dir, truth_dir, clip, rng=None,
                 normalizer_path=None, want_native=True):
    plan = pickle.load(open(str(pathlib.Path(plan_dir) / (clip + ".pkl")), "rb"))
    group = plan["prototype_retrieval"]["query_retrieval_group_id"]
    if group is None:
        return None
    labels = torch.as_tensor(np.asarray(plan["atomic_labels"]))
    segments = list(labels_to_segments(labels))
    draft = pickle.load(open(str(pathlib.Path(draft_dir) / (clip + ".pkl")), "rb"))
    truth = pickle.load(open(str(pathlib.Path(truth_dir) / (clip + ".pkl")), "rb"))
    draft_speed = joint_speed(draft["full_pose"])
    truth_speed = joint_speed(truth["full_pose"])
    draft_levels = speed_levels(draft_speed, segments)
    truth_levels = speed_levels(truth_speed, segments)
    rows = []
    for position, segment in enumerate(segments):
        if segment.label == 0:
            continue
        chosen, candidates = chosen_candidate(
            library, segment.label, segment.length, (group,))
        if chosen is None:
            continue
        if rng is not None:
            pool = safe_candidates(library, segment.label, frozenset((group,)))
            chosen = pool[int(rng.integers(len(pool)))]
        native = int(chosen[2] - chosen[1])
        if native <= 0:
            continue
        stretch = float(segment.length) / native
        row = {
            "clip": clip,
            "position": position,
            "label": int(segment.label),
            "start": int(segment.start),
            "end": int(segment.end),
            "slot_frames": int(segment.length),
            "native_frames": native,
            "stretch": stretch,
            "playback_speed_multiplier": 1.0 / stretch,
            "pool": len(candidates),
            "draft_speed": (draft_levels[position] or {}).get("mean"),
            "truth_speed": (truth_levels[position] or {}).get("mean"),
        }
        if want_native and normalizer_path is not None:
            own = native_mean_speed(library, chosen, normalizer_path)
            row["native_speed"] = own
            row["native_speed_ratio"] = (
                None if not own or not row["draft_speed"]
                else float(row["draft_speed"] * stretch / own))
        rows.append(row)
    return {
        "clip": clip,
        "frames": int(len(labels)),
        "transition_frame_share": float((labels == 0).float().mean()),
        "segment_rows": rows,
        "draft_boundaries": boundary_jumps(draft_speed, segments, draft_levels),
        "truth_boundaries": boundary_jumps(truth_speed, segments, truth_levels),
        "draft_speed_median": float(np.median(draft_speed)),
        "truth_speed_median": float(np.median(truth_speed)),
    }


def _log_sd(values):
    values = [v for v in values if v and v > 0]
    if len(values) < 2:
        return None
    return float(np.log(np.asarray(values, float)).std())


def summarise(clips, speed_columns=True):
    """``speed_columns=False`` under a positive control, and that is not tidiness.

    The control re-draws the candidate but there is no rendered draft for that
    draw -- the ``--draft-dir`` dump is still the SHIPPED one.  Reporting its
    speed columns beside control stretch numbers would print a reading that
    belongs to a different arm under the control's name, which is exactly the
    "a flag that is recorded but not applied" defect this repository keeps
    paying for.  They are emitted as ``None`` with a reason instead.
    """
    stretch = np.array([r["stretch"] for c in clips for r in c["segment_rows"]], float)
    per_clip = {}
    for clip in clips:
        stretches = [r["stretch"] for r in clip["segment_rows"]]
        per_clip[clip["clip"]] = {
            "stretch": within_clip_stretch(stretches),
            "draft_speed_sd_log": _log_sd([r["draft_speed"] for r in clip["segment_rows"]]),
            "truth_speed_sd_log": _log_sd([r["truth_speed"] for r in clip["segment_rows"]]),
            "transition_frame_share": round(clip["transition_frame_share"], 3),
        }
    draft_jumps = [b["level_log_jump"] for c in clips for b in c["draft_boundaries"]
                   if b["level_log_jump"] is not None]
    truth_jumps = [b["level_log_jump"] for c in clips for b in c["truth_boundaries"]
                   if b["level_log_jump"] is not None]
    draft_steps = [b["step_over_median"] for c in clips for b in c["draft_boundaries"]
                   if b["step_over_median"] is not None]
    truth_steps = [b["step_over_median"] for c in clips for b in c["truth_boundaries"]
                   if b["step_over_median"] is not None]
    ratios = [r["native_speed_ratio"] for c in clips for r in c["segment_rows"]
              if r.get("native_speed_ratio")]
    within = [v["stretch"] for v in per_clip.values() if v["stretch"]]
    if not speed_columns:
        draft_jumps = truth_jumps = draft_steps = truth_steps = ratios = []
        for entry in per_clip.values():
            entry["draft_speed_sd_log"] = None
            entry["truth_speed_sd_log"] = None
    return {
        "speed_columns": (True if speed_columns else
                          "withheld: the rendered draft belongs to the shipped "
                          "rule, not to this control's re-drawn candidates"),
        "clips": len(clips),
        "segments": int(stretch.size),
        "stretch_median": round(float(np.median(stretch)), 4),
        "stretch_p05": round(float(np.percentile(stretch, 5)), 4),
        "stretch_p25": round(float(np.percentile(stretch, 25)), 4),
        "stretch_p75": round(float(np.percentile(stretch, 75)), 4),
        "stretch_p95": round(float(np.percentile(stretch, 95)), 4),
        "share_beyond_1_4x": round(float((stretch > 1.4).mean()), 4),
        "share_below_inv_1_4x": round(float((stretch < 1.0 / 1.4).mean()), 4),
        "share_exact": round(float((stretch == 1.0).mean()), 4),
        "within_clip_sd_log_median": (round(float(np.median(
            [w["sd_log"] for w in within])), 4) if within else None),
        "within_clip_max_over_min_median": (round(float(np.median(
            [w["max_over_min"] for w in within])), 4) if within else None),
        "within_clip_adjacent_abs_log_median": (round(float(np.median(
            [w["adjacent_mean_abs_log"] for w in within])), 4) if within else None),
        "clips_with_two_or_more_segments": len(within),
        "draft_level_log_jump_median": (round(float(np.median(draft_jumps)), 4)
                                        if draft_jumps else None),
        "truth_level_log_jump_median": (round(float(np.median(truth_jumps)), 4)
                                        if truth_jumps else None),
        "draft_step_over_median_median": (round(float(np.median(draft_steps)), 4)
                                          if draft_steps else None),
        "truth_step_over_median_median": (round(float(np.median(truth_steps)), 4)
                                          if truth_steps else None),
        "boundaries": len(draft_jumps),
        "native_speed_ratio_median": (round(float(np.median(ratios)), 4)
                                      if ratios else None),
        "per_clip": per_clip,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--plan-dir", required=True,
                        help="run directory holding the plan (atomic_labels) pkls")
    parser.add_argument("--draft-dir", required=True,
                        help="--draft-only dump of the SAME plan")
    parser.add_argument("--ground-truth-dir", required=True)
    parser.add_argument("--clips", required=True, type=pathlib.Path)
    parser.add_argument("--positive-control", choices=("uniform",), default=None,
                        help="draw the candidate uniformly from the same pool")
    parser.add_argument("--control-seed", type=int, default=20260905)
    parser.add_argument("--no-native-speed", action="store_true",
                        help="skip the mechanism check (it runs SMPL FK per segment)")
    parser.add_argument("--out", type=pathlib.Path)
    arguments = parser.parse_args()

    clips = [line.strip() for line in arguments.clips.read_text().splitlines()
             if line.strip()]
    library = IndexedAtomicMotionLibrary(arguments.data_root, retrieval_rule="duration")
    rng = (np.random.default_rng(arguments.control_seed)
           if arguments.positive_control else None)
    measured = []
    for clip in clips:
        row = measure_clip(library, arguments.plan_dir, arguments.draft_dir,
                           arguments.ground_truth_dir, clip, rng=rng,
                           normalizer_path=library.normalizer_path,
                           want_native=not arguments.no_native_speed)
        if row is not None and row["segment_rows"]:
            measured.append(row)
    if not measured:
        raise SystemExit("error: no clip measured")
    report = {"plan_dir": arguments.plan_dir, "draft_dir": arguments.draft_dir,
              "positive_control": arguments.positive_control,
              **summarise(measured,
                            speed_columns=arguments.positive_control is None)}
    if arguments.out:
        arguments.out.write_text(json.dumps(
            {"summary": report, "clips": measured}, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
