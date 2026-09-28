"""Measure whether the roughness in a generated clip lives at the SEAMS.

A "seam" here is a specific, reconstructible thing, not a guess: ``build_draft``
retrieves one prototype per conditioned plan segment and pastes them end to end,
so wherever two conditioned segments TOUCH (segment i ends at frame f and
segment i+1 begins at frame f, neither of them the transition label 0) the draft
carries a join between two clips cut from two different recordings.  The segment
list is ``labels_to_segments(atomic_labels, split_at=bar_bounds)`` -- exactly the
call inference made -- so this tool recomputes the same boundaries rather than
trusting a bookkeeping note.

The reading is a ratio, ``jerk near seams / jerk away from seams``, and a ratio
above 1 on its own proves nothing: frames near a seam might simply be frames
where the music asks for a change.  So three controls decide it, and each one
can fail:

  * SHUFFLED SEAMS (negative).  Draw the same number of positions uniformly
    inside the same clip.  If a random frame reads the same ratio, the metric is
    measuring the clip's overall roughness, not the join.

  * GROUND TRUTH AT THE SAME FRAMES (negative, and the one that matters).  Score
    the *ground truth* motion of the same clip at the *generated* clip's seam
    frames.  Ground truth was danced by a person in one take and has no joins at
    all, so a ratio near 1 there is what says these frame positions are not
    intrinsically rough.  A ratio well above 1 would mean the seams landed on
    musically eventful frames and the whole reading is confounded.

  * DOSE RESPONSE (positive).  The per-bar arm cuts the same plan into more
    prototypes, so it has more seams.  If seams are what carries the roughness,
    the arm with more seams must carry more rough frames.  This is a prediction
    that can come out the other way.

The second statistic is the SHAPE of the defect rather than its size: within a
window around each seam, which offset holds the slowest frame.  A dancer that
brakes to a near stop one frame before the join and re-accelerates after it puts
a spike at one particular offset; a body that is merely rough scatters.
"""

import argparse
import json
import pathlib
import pickle
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dataset.atomic import labels_to_segments
import torch

FPS = 30.0
MUSIC_BEAT_CHANNEL = 34


def root_relative(joints):
    return joints - joints[:, :1, :]


def speed(joints):
    """Per-frame mean joint speed, m/s, defined on frames 1..T-1."""
    return np.linalg.norm(np.diff(root_relative(joints), axis=0), axis=2).mean(axis=1) * FPS


def jerk(joints):
    """Per-frame roughness: magnitude of the second difference, on 1..T-2."""
    relative = root_relative(joints)
    return np.linalg.norm(np.diff(relative, n=2, axis=0), axis=2).mean(axis=1) * FPS * FPS


def bar_bounds_from_music(music, beats_per_segment, phase):
    """The bar lines inference used: channel 34 is the beat one-hot."""
    grid = np.nonzero(np.asarray(music)[:, MUSIC_BEAT_CHANNEL] > 0.5)[0]
    if grid.size == 0:
        return []
    return [int(b) for b in grid[int(phase)::int(beats_per_segment)]]


def seam_frames(labels, bar_bounds=None):
    """Frames where one retrieved prototype butts against the next.

    Segments carrying the transition label 0 are never retrieved (``build_draft``
    skips them), so a boundary that touches one is a GAP, not a seam -- the draft
    is unconditioned there and the completion model was told so.  Only joins
    between two conditioned segments are counted.
    """
    segments = list(labels_to_segments(torch.as_tensor(np.asarray(labels)),
                                       split_at=bar_bounds))
    seams = []
    for previous, current in zip(segments[:-1], segments[1:]):
        if previous.label == 0 or current.label == 0:
            continue
        if previous.end != current.start:
            continue
        seams.append(int(current.start))
    return seams, len(segments)


def near_mask(frames, positions, radius):
    mask = np.zeros(frames, bool)
    for position in positions:
        low = max(0, int(position) - radius)
        high = min(frames, int(position) + radius + 1)
        mask[low:high] = True
    return mask


def ratio_at(values, positions, radius, guard):
    """Mean roughness within +-radius of the positions, over the mean away.

    MEAN, not median, and the choice is forced rather than stylistic: a paste
    between two prototypes puts its whole second-difference spike on one or two
    frames, so inside a five-frame window the median is by construction one of
    the frames WITHOUT the spike.  Scored on a hand-built splice whose answer is
    known, the median reads 0.89 -- it would have declared a motion we cut and
    glued ourselves to be smooth.  ``tests/test_measure_seam_judder.py`` pins
    both readings so the choice cannot be quietly reverted.

    ``guard`` widens the exclusion for the "away" pool so that the comparison
    pool is genuinely away from a seam rather than one frame outside the window.
    Returns None when either pool is empty -- an unmeasurable clip is reported
    as unmeasured, never folded in as 1.0.
    """
    values = np.asarray(values, float)
    if values.size == 0 or not len(positions):
        return None
    near = near_mask(values.size, positions, radius)
    away = ~near_mask(values.size, positions, guard)
    if not near.any() or not away.any():
        return None
    denominator = float(values[away].mean())
    if denominator <= 0:
        return None
    return float(values[near].mean() / denominator)


def brake_offsets(speeds, positions, radius):
    """Offset of the slowest frame in a window around each seam.

    ``speeds`` is defined on frames 1..T-1 and indexed here so that entry i is
    the speed entering frame i+1; offset 0 means the slowest frame is the seam
    itself, offset -1 the frame before it.
    """
    speeds = np.asarray(speeds, float)
    offsets = []
    for position in positions:
        low = max(0, int(position) - radius)
        high = min(speeds.size, int(position) + radius + 1)
        if high - low < 2 * radius + 1:
            continue
        window = speeds[low:high]
        offsets.append(int(np.argmin(window) + low - int(position)))
    return offsets


def load_generated(path):
    payload = pickle.load(open(path, "rb"))
    return (np.asarray(payload["full_pose"], float),
            np.asarray(payload["atomic_labels"]),
            payload["prototype_retrieval"]["plan_postprocess"])


def load_ground_truth(path):
    """Ground truth arrives in the same pickle shape as a generated arm."""
    payload = pickle.load(open(path, "rb"))
    return np.asarray(payload["full_pose"], float)


def measure_clip(joints, labels, bar_bounds, radius, guard, rng, draws):
    seams, segments = seam_frames(labels, bar_bounds)
    roughness = jerk(joints)
    speeds = speed(joints)
    real = ratio_at(roughness, seams, radius, guard)
    null = []
    for _ in range(draws):
        drawn = rng.integers(guard + 1, max(guard + 2, roughness.size - guard - 1),
                             size=len(seams)) if seams else []
        value = ratio_at(roughness, drawn, radius, guard)
        if value is not None:
            null.append(value)
    return {
        "frames": int(len(joints)),
        "segments": int(segments),
        "seams": seams,
        "seam_count": len(seams),
        "jerk_ratio": real,
        "jerk_ratio_null_median": float(np.median(null)) if null else None,
        "brake_offsets": brake_offsets(speeds, seams, radius),
    }


def uses_bar_prototypes(directory):
    """Ask the RUN whether it cut a prototype per bar, rather than guess.

    ``--draft-bar-prototypes`` is recorded in the manifest's ``sampling`` block,
    so the seam reconstruction reads the same switch inference was given.  A run
    without a manifest is refused rather than scored with the wrong boundaries:
    reconstructing the seams in the wrong places would quietly produce a
    plausible number for the wrong frames.
    """
    manifest = pathlib.Path(directory) / "manifest.json"
    if not manifest.exists():
        raise SystemExit(
            "error: {} has no manifest.json, so whether it used per-bar "
            "prototypes is unknown and its seams cannot be reconstructed."
            .format(directory))
    sampling = json.loads(manifest.read_text()).get("sampling", {})
    return bool(sampling.get("draft_bar_prototypes", False))


def run(arms, clips, ground_truth_dir, audio_dir, radius, guard, seed, draws):
    rng = np.random.default_rng(seed)
    report = {"radius": radius, "guard": guard, "seed": seed, "draws": draws,
              "arms": {}, "clips": list(clips)}
    for name, directory in arms:
        rows = {"jerk_ratio": [], "null": [], "seam_count": [], "segments": [],
                "gt_at_same_frames": [], "offsets": [], "clips": [],
                "gt_offsets": [], "paired": []}
        missing = []
        for clip in clips:
            path = pathlib.Path(directory) / (clip + ".pkl")
            if not path.exists():
                missing.append(clip)
                continue
            joints, labels, postprocess = load_generated(path)
            bounds = None
            if uses_bar_prototypes(directory):
                music = np.load(pathlib.Path(audio_dir) / (clip + ".npy"))
                bounds = bar_bounds_from_music(
                    music,
                    postprocess.get("plan_bar_beats", 4),
                    postprocess.get("bar_grid_phase", 0) or 0)
            measured = measure_clip(joints, labels, bounds, radius, guard, rng, draws)
            if measured["jerk_ratio"] is None:
                continue
            rows["clips"].append(clip)
            rows["jerk_ratio"].append(measured["jerk_ratio"])
            rows["null"].append(measured["jerk_ratio_null_median"])
            rows["seam_count"].append(measured["seam_count"])
            rows["segments"].append(measured["segments"])
            rows["offsets"].extend(measured["brake_offsets"])
            # The control that decides the whole reading: ground truth scored at
            # the frames THIS arm put its seams on.
            truth_path = pathlib.Path(ground_truth_dir) / (clip + ".pkl")
            if truth_path.exists():
                truth = load_ground_truth(truth_path)
                usable = [s for s in measured["seams"] if s < len(truth) - 2]
                value = ratio_at(jerk(truth), usable, radius, guard)
                if value is not None:
                    rows["gt_at_same_frames"].append(value)
                    # Paired, per clip: an arm whose seams carry excess
                    # roughness must beat ground truth AT ITS OWN SEAM FRAMES.
                    # Comparing two medians would let a few rough clips on one
                    # side answer for calm clips on the other.
                    rows["paired"].append(measured["jerk_ratio"] > value)
                # And the same question for the SHAPE: if ground truth also
                # puts its slowest frame one before these positions, then
                # offset -1 is a property of the frames, not of the join.
                rows["gt_offsets"].extend(brake_offsets(speed(truth), usable, radius))
        if not rows["clips"]:
            # A criterion that reports "None" when it could not read anything
            # is the failure mode this repository keeps paying for: it looks
            # like a measurement that came back empty rather than a run that
            # never happened.
            raise SystemExit(
                "error: arm {!r} scored 0 of {} clips from {} ({} clip files "
                "were missing).  Nothing was measured, so nothing is reported."
                .format(name, len(clips), directory, len(missing)))
        report["arms"][name] = summarise(rows)
        report["arms"][name]["clips_missing"] = len(missing)
    return report


def summarise(rows):
    def median(key):
        values = [v for v in rows[key] if v is not None]
        return float(np.median(values)) if values else None

    offsets = np.asarray(rows["offsets"], int)
    histogram = {}
    if offsets.size:
        for offset in range(int(offsets.min()), int(offsets.max()) + 1):
            histogram[str(offset)] = int((offsets == offset).sum())
    beats_null = [real > null for real, null in zip(rows["jerk_ratio"], rows["null"])
                  if null is not None]
    truth_offsets = np.asarray(rows.get("gt_offsets", []), int)
    truth_histogram = {}
    if truth_offsets.size:
        for offset in range(int(truth_offsets.min()), int(truth_offsets.max()) + 1):
            truth_histogram[str(offset)] = int((truth_offsets == offset).sum())
    paired = rows.get("paired", [])
    return {
        "clips_rougher_than_truth_at_own_seams": "{}/{}".format(
            sum(paired), len(paired)) if paired else None,
        "ground_truth_offset_histogram": truth_histogram,
        "ground_truth_offset_mode_share": (
            float(max(truth_histogram.values()) / truth_offsets.size)
            if truth_offsets.size else None),
        "clips": len(rows["clips"]),
        "seams_per_clip": median("seam_count"),
        "segments_per_clip": median("segments"),
        "jerk_ratio": median("jerk_ratio"),
        "jerk_ratio_shuffled_null": median("null"),
        "clips_above_own_null": "{}/{}".format(sum(beats_null), len(beats_null)),
        "ground_truth_at_same_frames": median("gt_at_same_frames"),
        "brake_offset_histogram": histogram,
        "brake_offset_total": int(offsets.size),
        "brake_offset_mode_share": (
            float(max(histogram.values()) / offsets.size) if offsets.size else None),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", action="append", required=True,
                        metavar="NAME=DIR", help="a generated arm to score")
    parser.add_argument("--clips", required=True, type=pathlib.Path)
    parser.add_argument("--ground-truth", required=True, type=pathlib.Path)
    parser.add_argument("--audio-dir", required=True, type=pathlib.Path)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--guard", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--out", type=pathlib.Path)
    arguments = parser.parse_args()

    arms = []
    for entry in arguments.arm:
        if "=" not in entry:
            raise SystemExit("--arm wants NAME=DIR, got {!r}".format(entry))
        # rpartition, not partition: an arm name may itself contain "=" (they
        # tend to, since the useful name of an arm is the switch it sets), and
        # splitting on the FIRST one silently hands the scorer a directory that
        # does not exist.
        name, _, directory = entry.rpartition("=")
        if not pathlib.Path(directory).is_dir():
            raise SystemExit(
                "error: --arm {!r} resolves to directory {!r}, which does not "
                "exist.".format(name, directory))
        arms.append((name, directory))
    clips = [line.strip() for line in arguments.clips.read_text().splitlines()
             if line.strip()]

    report = run(arms, clips, arguments.ground_truth, arguments.audio_dir,
                 arguments.radius, arguments.guard, arguments.seed, arguments.draws)
    text = json.dumps(report, indent=2, sort_keys=True)
    if arguments.out:
        arguments.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
