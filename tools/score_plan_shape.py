"""The SHAPE of a plan in time, scored against ground truth's own plan.

WHY THIS EXISTS, and what it caught.  The arm table already carries
``seg_per_s``, and on 2026-09-03 it read 0.313 for the shipped arm against 0.309
for a candidate -- indistinguishable.  The rendered strip for one clip showed
the shipped arm holding a single atomic movement for 4.9 s and the candidate
holding one for 10.4 s, which is the "pose 招单一" the operator reports.  A
median over clips cannot see a clip that collapsed: a clip with one 20 s hold
still has five segments if the others are short, so segments-per-second stays
put while the dance becomes one long pose.

Measured over the 20 held-out clips once the right column existed:

    arm            longest hold (median / p90 / max)   clips with a hold > 4 s
    ground truth        3.43s  /  4.42s  /  9.20s               7/20
    shipped (ep6)       5.83s  / 15.42s  / 20.40s              17/20
    candidate (ep12)    3.75s  /  7.51s  / 10.40s               9/20

The second column that was missing is FILLER.  Ground truth spends 30.4% of its
frames on no named movement at all -- a dancer rests, and the beat-settle the
scorecard asks for happens in that rest.  The shipped arm spends 0.4%: it is
wall to wall movement with no breathing room.

Neither column is gated here.  Ground truth's own reading is printed beside
every arm and the decision is the caller's, because "more segments" is not
automatically better -- an arm that chopped the plan into half-second pieces
would win both columns and lose the dance.
"""

import argparse
import json
import pathlib
import pickle

import numpy as np

FPS = 30.0


def segments(labels):
    """(label, start, end) runs, transition included."""
    labels = np.asarray(labels)
    if labels.size == 0:
        return []
    change = np.nonzero(labels[1:] != labels[:-1])[0] + 1
    bounds = [0, *change.tolist(), len(labels)]
    return [(int(labels[a]), int(a), int(b))
            for a, b in zip(bounds[:-1], bounds[1:])]


def shape_of(labels):
    """Filler share, count, and the LONGEST named run, in seconds."""
    labels = np.asarray(labels)
    if labels.size == 0:
        return None
    named = [(b - a) / FPS for label, a, b in segments(labels) if label != 0]
    return {
        "frames": int(labels.size),
        "filler_share": float((labels == 0).mean()),
        "segments": len(named),
        "median_segment_s": float(np.median(named)) if named else 0.0,
        "longest_hold_s": float(max(named)) if named else 0.0,
    }


def summarise(rows, hold_threshold):
    if not rows:
        return None
    longest = np.array([r["longest_hold_s"] for r in rows])
    return {
        "clips": len(rows),
        "filler_share": float(np.median([r["filler_share"] for r in rows])),
        "segments": float(np.median([r["segments"] for r in rows])),
        "median_segment_s": float(np.median([r["median_segment_s"] for r in rows])),
        "longest_hold_median_s": float(np.median(longest)),
        "longest_hold_p90_s": float(np.percentile(longest, 90)),
        "longest_hold_max_s": float(longest.max()),
        "clips_over_threshold": "{}/{}".format(int((longest > hold_threshold).sum()),
                                               len(longest)),
    }


def ground_truth_labels(labels_root, clips):
    """Ground truth plans, addressed through the release's own manifest.

    The manifest is read rather than the directory globbed: the label files are
    named by content hash, so a glob would silently pick up any orphan left by
    an earlier build (CLAUDE.md section 1.1 -- downstream consumes manifests,
    never directory listings).
    """
    root = pathlib.Path(labels_root)
    wanted = set(clips)
    out = {}
    for line in (root / "labels.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["sequence_id"] in wanted:
            out[record["sequence_id"]] = np.load(root / record["labels_path"])
    return out


def arm_labels(directory, clips):
    out = {}
    for clip in clips:
        path = pathlib.Path(directory) / (clip + ".pkl")
        if path.exists():
            out[clip] = np.asarray(pickle.load(open(path, "rb"))["atomic_labels"])
    return out


def run(arms, clips, labels_root, hold_threshold):
    report = {"hold_threshold_s": hold_threshold, "arms": {}}
    truth = ground_truth_labels(labels_root, clips)
    report["ground_truth"] = summarise(
        [shape_of(v) for v in truth.values() if shape_of(v)], hold_threshold)
    for name, directory in arms:
        rows = [shape_of(v) for v in arm_labels(directory, clips).values()]
        rows = [r for r in rows if r]
        if not rows:
            raise SystemExit(
                "error: arm {!r} scored 0 of {} clips from {}.  Nothing was "
                "measured.".format(name, len(clips), directory))
        report["arms"][name] = summarise(rows, hold_threshold)
    return report


def render(report):
    header = ("%-24s %8s %8s %9s %9s %9s %9s %10s" %
              ("arm", "filler", "segs", "seg len", "hold med", "hold p90",
               "hold max", "over thr"))
    lines = [header, "-" * len(header)]

    def line(name, row):
        return ("%-24s %7.1f%% %8.1f %8.2fs %8.2fs %8.2fs %8.2fs %10s" %
                (name, 100 * row["filler_share"], row["segments"],
                 row["median_segment_s"], row["longest_hold_median_s"],
                 row["longest_hold_p90_s"], row["longest_hold_max_s"],
                 row["clips_over_threshold"]))

    if report.get("ground_truth"):
        lines.append(line("ground truth", report["ground_truth"]))
    for name, row in report["arms"].items():
        lines.append(line(name, row))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--clips", required=True, type=pathlib.Path)
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--hold-threshold", type=float, default=4.0,
                        help="seconds; the default is ground truth's own p90 "
                             "on the T corpus (4.42s), rounded down")
    parser.add_argument("--json", type=pathlib.Path)
    arguments = parser.parse_args()

    arms = []
    for entry in arguments.arm:
        name, _, directory = entry.rpartition("=")
        if not pathlib.Path(directory).is_dir():
            raise SystemExit("error: --arm {!r} points at {!r}, which does not "
                             "exist.".format(name, directory))
        arms.append((name, directory))
    clips = [line.strip() for line in arguments.clips.read_text().splitlines()
             if line.strip()]
    report = run(arms, clips, arguments.labels_root, arguments.hold_threshold)
    if arguments.json:
        arguments.json.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(render(report))


if __name__ == "__main__":
    main()
