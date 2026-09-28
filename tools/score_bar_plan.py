#!/usr/bin/env python3
"""Score a generated plan on the BAR grid the corpus was cut on.

WHAT THIS MEASURES AND WHERE THE CRITERION COMES FROM
=====================================================
``docs/DANCE_QUALITY_DEFECTS.md`` section 27.2 reads the planner at the
resolution the vocabulary actually has.  M1 cut the T-line corpus on a 4-beat
music grid (``runs/txy_t_seg_beat4/segmentation.json``: ``mode grid /
beats_per_segment 4 / phase energy / edges drop``), so a "move" is one bar --
median 2.00 s, **maximum 2.97 s**.  Taking those bar lines and the majority
label inside each bar, ground truth changes label from one bar to the next
**78.4%** of the time, holds a class for 3 or more bars on 3.7% of its runs and
for 5 or more on 0.5%.

THE CRITERION IS TWO-SIDED.  78.4% is a target, not a ceiling: a planner that
changes on 95% of bar lines has not won, it has a different defect (a new move
every bar, no phrase structure).  ``change_rate_error`` is therefore reported as
the SIGNED distance to ground truth's own reading on the same clips, and the run
lengths are reported as a distribution rather than a mean.

WHY THIS FILE DOES NOT DEFINE THE STATISTIC
===========================================
It imports ``bar_shape_stats``, ``majority_label`` and
``recording_id_from_segmentation`` from ``tools/build_bar_planner_release.py``.
That is deliberate: the bar release the bar-resolution planner trains on and the
scorer that judges it must not be able to drift into two definitions of "a bar"
or of "changed label".  This file adds only the reading of a generated artifact.

WHAT COUNTS AS A BAR HERE
=========================
Exactly the segments in ``segmentation.json`` -- the frames the ``edges: drop``
policy discarded at the head and tail of each recording are NOT scored.  Two
consequences that must be stated wherever these numbers appear:

* the ground-truth **bar** filler share is 15.7% over 270 recordings and 21.5%
  over the 18 eval clips, while the ground-truth **frame** filler share is
  30.4%.  They differ because all 21,412 out-of-grid frames carry label 0.
  ``filler_bar_share`` and ``filler_frame_share`` are both reported and must
  never be put in one column.
* ``filler_frame_share`` is over the whole clip, so it does include those
  out-of-grid frames -- it is the column comparable with section 27.2's
  ``filler``, and ``filler_bar_share`` is the one comparable with the bar
  release's report.

THE GROUND-TRUTH ROW comes from the bar release's ``bars.jsonl``
(``bar_labels`` per window, overlapping windows asserted to agree), and the
scorer asserts that its ``bar_frame_spans`` are the same spans this scorer cuts
from ``segmentation.json``.  Without that assertion the reference row and the
scored row could be read off two different grids and still look reasonable.

USAGE
=====
    python3 tools/score_bar_plan.py \
        --segmentation /cache/atomicdance-assets/runs/txy_t_seg_beat4/segmentation.json \
        --clips /path/to/clips18_fullname.txt \
        --bar-release /cache/atomicdance-assets/scratch/txy_t/release_bar_v1 \
        --arm ep12=/cache/atomicdance-assets/runs/opt_planckpt_ep12 \
        --arm bar_ep100=/cache/.../plan_bar_ep100 \
        --json out.json
"""

import argparse
import json
import pathlib
import pickle
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.build_bar_planner_release import (  # noqa: E402
    FILLER_LABEL,
    bar_shape_stats,
    majority_label,
    recording_id_from_segmentation,
)

NUM_CLASSES = 21
# Ground truth's own reading over the 270 recordings that carry a bar grid,
# recorded in docs/DANCE_QUALITY_DEFECTS.md 27.2 and reproduced by
# tools/build_bar_planner_release.py's report (78.44%).  Used only as a printed
# reference; every comparison in this tool is against the ground-truth row
# recomputed on the SAME clips as the arm.
CORPUS_CHANGE_RATE = 0.784


def load_bar_grid(segmentation_path):
    """``{recording_id: [(start, end), ...]}`` -- the bars, in release ids."""
    payload = json.loads(pathlib.Path(segmentation_path).read_text())
    grid = {}
    for record in payload["records"]:
        spans = [(int(s["start"]), int(s["end"])) for s in record["segments"]]
        grid[recording_id_from_segmentation(record["sequence"])] = spans
    return grid


def bars_of_plan(labels, spans, num_classes=NUM_CLASSES):
    """One label per bar from a per-frame plan, by majority inside the bar.

    A bar whose span runs past the end of the plan is TRUNCATED to the frames
    that exist, and a bar with no frames at all is dropped -- with the count
    returned, because a silently shortened recording is exactly the failure
    ``covered_segments`` was written for in the release builder.
    """
    labels = np.asarray(labels)
    bars, truncated, dropped = [], 0, 0
    for start, end in spans:
        stop = min(int(end), labels.size)
        if stop <= int(start):
            dropped += 1
            continue
        if stop < int(end):
            truncated += 1
        bars.append(majority_label(labels[int(start):stop], num_classes))
    return bars, truncated, dropped


def ground_truth_bars(bar_release_root):
    """``{recording_id: [bar labels]}`` from the bar release's ``bars.jsonl``.

    Windows overlap at stride 1 bar, so each bar appears in up to
    ``window_bars`` rows.  Every repeat is COMPARED rather than skipped: two
    windows that disagreed about a bar's label would mean the release's own
    tokenisation is not a function of the recording, and that must raise here
    rather than be resolved by whichever row was read last.
    """
    root = pathlib.Path(bar_release_root)
    labels = defaultdict(dict)
    spans = defaultdict(dict)
    with open(str(root / "bars.jsonl")) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sequence = row["recording_id"]
            start = int(row["bar_index_start"])
            for offset, (label, span) in enumerate(
                    zip(row["bar_labels"], row["bar_frame_spans"])):
                index = start + offset
                span = (int(span[0]), int(span[1]))
                if index in labels[sequence]:
                    if labels[sequence][index] != int(label) or spans[sequence][index] != span:
                        raise ValueError(
                            "bars.jsonl disagrees with itself about bar {} of {}"
                            .format(index, sequence))
                labels[sequence][index] = int(label)
                spans[sequence][index] = span
    out_labels, out_spans = {}, {}
    for sequence, mapping in labels.items():
        order = sorted(mapping)
        if order != list(range(order[0], order[0] + len(order))):
            raise ValueError("bars.jsonl leaves a hole in {}".format(sequence))
        out_labels[sequence] = [mapping[i] for i in order]
        out_spans[sequence] = [spans[sequence][i] for i in order]
    return out_labels, out_spans


def assert_same_grid(release_spans, grid_spans, recording):
    """The reference row and the scored rows must be cut on the same bars.

    The release keeps a PREFIX of the grid's bars (its windows stop at the last
    whole 150-frame frame-release window, which can end inside the grid's last
    bar), so a prefix is accepted and anything else raises.
    """
    prefix = grid_spans[:len(release_spans)]
    if [tuple(s) for s in release_spans] != [tuple(s) for s in prefix]:
        raise ValueError(
            "the bar release and segmentation.json cut {} differently: release "
            "{} vs grid {}".format(recording, release_spans[:4], prefix[:4]))


def score_arm(artifact_dir, clips, grid, num_classes=NUM_CLASSES):
    """Bar-shape statistics for one generated arm, plus its retrieval gate."""
    artifact_dir = pathlib.Path(artifact_dir)
    sequences, missing = [], []
    filler_frames = total_frames = 0
    truncated = dropped = 0
    unfillable = 0
    worst_playback = None
    for clip in clips:
        path = artifact_dir / (clip + ".pkl")
        if not path.exists():
            missing.append(clip)
            continue
        with open(str(path), "rb") as handle:
            payload = pickle.load(handle)
        labels = np.asarray(payload["atomic_labels"])
        total_frames += int(labels.size)
        filler_frames += int((labels == FILLER_LABEL).sum())
        if clip not in grid:
            raise KeyError("{} has no bar grid; it cannot be scored on bars".format(clip))
        bars, clip_truncated, clip_dropped = bars_of_plan(labels, grid[clip], num_classes)
        truncated += clip_truncated
        dropped += clip_dropped
        sequences.append(bars)
        stretch = payload.get("prototype_retrieval", {}).get("retrieval_stretch") or {}
        unfillable += int(stretch.get("units_over_library_ceiling") or 0)
        playback = stretch.get("playback_min")
        if playback is not None:
            worst_playback = playback if worst_playback is None else min(worst_playback, playback)
    if missing:
        raise FileNotFoundError(
            "{} of {} clips have no artifact under {}: {}".format(
                len(missing), len(clips), artifact_dir, missing[:4]))
    stats = bar_shape_stats(sequences)
    stats.update({
        "artifact_dir": str(artifact_dir),
        "clips": len(sequences),
        "filler_frame_share": (filler_frames / total_frames) if total_frames else None,
        "frames": int(total_frames),
        "bars_truncated_by_plan_end": int(truncated),
        "bars_dropped_by_plan_end": int(dropped),
        "unfillable_retrieval_units": int(unfillable),
        "worst_playback_speed": worst_playback,
    })
    return stats


def score_ground_truth(bar_release_root, clips, grid):
    """The reference row, on the same clips and the same bars."""
    labels, spans = ground_truth_bars(bar_release_root)
    sequences = []
    for clip in clips:
        if clip not in labels:
            raise KeyError("{} is not in the bar release".format(clip))
        assert_same_grid(spans[clip], grid[clip], clip)
        sequences.append(labels[clip])
    stats = bar_shape_stats(sequences)
    stats.update({"artifact_dir": str(bar_release_root), "clips": len(sequences)})
    return stats


def compare(arm, reference):
    """The two-sided reading: signed distance to the reference on this set."""
    return {
        "change_rate_error": arm["change_rate"] - reference["change_rate"],
        "change_rate_abs_error": abs(arm["change_rate"] - reference["change_rate"]),
        "runs_ge_3_error": arm["runs_ge_3"] - reference["runs_ge_3"],
        "runs_ge_5_error": arm["runs_ge_5"] - reference["runs_ge_5"],
        "filler_bar_share_error": arm["filler_bar_share"] - reference["filler_bar_share"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--segmentation", required=True)
    parser.add_argument("--clips", required=True,
                        help="one release recording id per line")
    parser.add_argument("--bar-release", required=True,
                        help="the bar release, for the ground-truth reference row")
    parser.add_argument("--arm", action="append", default=[],
                        metavar="NAME=DIR", help="a generated artifact directory")
    parser.add_argument("--json", default=None)
    options = parser.parse_args()

    clips = [line.strip() for line in open(options.clips) if line.strip()]
    grid = load_bar_grid(options.segmentation)
    reference = score_ground_truth(options.bar_release, clips, grid)
    report = {"clips": clips, "ground_truth": reference,
              "corpus_change_rate": CORPUS_CHANGE_RATE, "arms": {}}
    rows = [("ground truth", reference, None)]
    for entry in options.arm:
        if "=" not in entry:
            raise SystemExit("--arm takes NAME=DIR, got {!r}".format(entry))
        name, directory = entry.split("=", 1)
        stats = score_arm(directory, clips, grid)
        stats["vs_ground_truth"] = compare(stats, reference)
        report["arms"][name] = stats
        rows.append((name, stats, stats["vs_ground_truth"]))

    header = ("{:<20} {:>8} {:>7} {:>7} {:>8} {:>8} {:>8} {:>4} {:>6}".format(
        "arm", "change%", "ge3%", "ge5%", "barfil%", "frmfil%", "top%", "cls", "unfill"))
    print(header)
    print("-" * len(header))
    for name, stats, delta in rows:
        print("{:<20} {:>8.1f} {:>7.1f} {:>7.1f} {:>8.1f} {:>8} {:>8.1f} {:>4} {:>6}".format(
            name,
            100 * stats["change_rate"],
            100 * stats["runs_ge_3"],
            100 * stats["runs_ge_5"],
            100 * stats["filler_bar_share"],
            "-" if stats.get("filler_frame_share") is None
            else "{:.1f}".format(100 * stats["filler_frame_share"]),
            100 * stats["top_class_share"],
            stats["distinct_classes"],
            "-" if delta is None else stats["unfillable_retrieval_units"]))
    if options.json:
        pathlib.Path(options.json).write_text(json.dumps(report, indent=1, sort_keys=True))
        print("wrote {}".format(options.json))


if __name__ == "__main__":
    main()
