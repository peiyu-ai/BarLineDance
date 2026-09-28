"""Torso twist rate measured ONLY on the frames the plan names a movement on.

WHY THIS FILE EXISTS.  ``tools/score_arm_table.py``'s ``twist`` column averages
the shoulder-line-vs-hip-line rotation rate over EVERY frame of a clip, and on
the draft-only arm that whole-clip average reads 25.7 deg/s against the shipped
completion arm's 31.8 -- the draft loses.  But 29.7% of the draft's frames carry
plan label 0 (filler / transition), and on those frames the draft is not dancing
at all: ``--draft-gap-fill interpolate`` fills them with a straight line between
the two neighbouring prototypes, and a straight line has almost no twist.  So
the whole-clip average is a mixture of two populations with different meanings,
and the mixing weight is a property of the PLAN, not of the motion.

This tool splits the mixture.  It reports, per arm:

  * ``twist_named``   the same measure over frame-steps whose BOTH endpoints
                      carry a named label (1..20);
  * ``twist_filler``  the same over steps whose both endpoints are label 0;
  * ``twist_all``     the whole-clip reading, which must reproduce
                      ``score_arm_table``'s column exactly (checked below);
  * the step counts, so the reader can verify that the three numbers combine
    into the whole-clip one rather than taking it on trust.

The angle is unwrapped ONCE over the whole clip and only then split, because
unwrapping each population separately would invent 2*pi jumps at every gap.
Steps that CROSS a boundary (one endpoint named, one filler) belong to neither
population and are counted separately as ``crossing``.

WHAT THIS CANNOT SAY.  A higher named-frame twist is not by itself "better
dance": ground truth is the reference, not the maximum, and the whole-clip
column stays in the decision table beside this one.  Section 2.1 rule 4 of
CLAUDE.md applies -- when the two readings order the arms differently, both get
reported and the difference gets explained by the filler share, which is why
that share is printed here rather than looked up.

CONTROLS (tests/test_exp_draftonly_named_twist.py):

  * a clip whose named frames twist at a known constant rate and whose filler
    frames are frozen must read that rate on ``twist_named``, ~0 on
    ``twist_filler``, and something strictly between on ``twist_all``
    (positive control, with the direction fixed in advance);
  * a clip that is entirely named must make ``twist_named`` equal
    ``score_arm_table.torso_twist_rate`` to floating-point tolerance -- the
    instrument check that this is the SAME measure and not a second one;
  * a clip that is entirely filler must yield ``twist_named`` = NaN rather than
    0.0, so an arm with no named frames cannot silently win the column.
"""

import argparse
import json
import pathlib
import pickle
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.score_arm_table import FPS, root_relative, torso_twist_rate  # noqa: E402,F401

FILLER_LABEL = 0


def twist_steps(joints):
    """Per-step |d(shoulder-hip angle)| in degrees per second, length T-1.

    Same construction as ``score_arm_table.torso_twist_rate``: root-relative
    joints, shoulders 16->17, hips 1->2, angle unwrapped over the whole clip.
    """
    relative = root_relative(np.asarray(joints, float))
    shoulders = relative[:, 17] - relative[:, 16]
    hips = relative[:, 2] - relative[:, 1]
    angle = np.unwrap(np.arctan2(shoulders[:, 1], shoulders[:, 0])
                      - np.arctan2(hips[:, 1], hips[:, 0]))
    return np.degrees(np.abs(np.diff(angle)) * FPS)


def split_by_label(joints, labels):
    """(named, filler, crossing, all) mean twist rates plus the step counts.

    A label track shorter than the motion truncates BOTH -- the annotation for a
    ground-truth recording can be a few frames shorter than the decoded motion,
    and scoring the tail against a label that does not exist would put filler
    frames into the named population by default.
    """
    joints = np.asarray(joints, float)
    labels = np.asarray(labels)
    usable = min(len(joints), len(labels))
    joints, labels = joints[:usable], labels[:usable]
    steps = twist_steps(joints)
    named_frame = labels != FILLER_LABEL
    left, right = named_frame[:-1], named_frame[1:]
    named = left & right
    filler = (~left) & (~right)
    crossing = ~(named | filler)

    def mean(mask):
        return float(steps[mask].mean()) if mask.any() else float("nan")

    return {
        "twist_all": float(steps.mean()) if steps.size else float("nan"),
        "twist_named": mean(named),
        "twist_filler": mean(filler),
        "twist_crossing": mean(crossing),
        "steps": int(steps.size),
        "steps_named": int(named.sum()),
        "steps_filler": int(filler.sum()),
        "steps_crossing": int(crossing.sum()),
        "named_frame_share": float(named_frame.mean()),
        "filler_frame_share": float((~named_frame).mean()),
    }


def read_labels_root(root):
    """clip -> per-frame label array, read the way score_arm_table reads it."""
    root = pathlib.Path(root)
    labels = {}
    for line in open(root / "labels.jsonl"):
        record = json.loads(line)
        if record.get("labels_path"):
            labels[record["sequence_id"]] = np.load(root / record["labels_path"])
    return labels


def load(path):
    payload = pickle.load(open(path, "rb"))
    labels = payload.get("atomic_labels")
    return np.asarray(payload["full_pose"], float), (None if labels is None else np.asarray(labels))


def score_arm(directory, clips, labels_by_clip=None):
    per_clip, missing_labels = {}, []
    for clip in clips:
        path = pathlib.Path(directory) / (clip + ".pkl")
        if not path.exists():
            continue
        joints, labels = load(path)
        if labels is None and labels_by_clip is not None:
            labels = labels_by_clip.get(clip)
        if labels is None:
            missing_labels.append(clip)
            continue
        per_clip[clip] = split_by_label(joints, labels)
    if not per_clip:
        raise SystemExit("error: 0 of {} clips scored from {}".format(len(clips), directory))
    rows = list(per_clip.values())

    def across(key):
        values = np.array([r[key] for r in rows], float)
        return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")

    summary = {key: across(key) for key in
               ("twist_all", "twist_named", "twist_filler", "twist_crossing",
                "named_frame_share", "filler_frame_share")}
    summary["clips"] = len(per_clip)
    summary["clips_without_labels"] = missing_labels
    # Pooled over steps as well as averaged over clips: the clip average is what
    # score_arm_table reports, the pooled one says whether a couple of long
    # clips are carrying the reading.
    for name, key in (("pooled_twist_named", "steps_named"), ("pooled_twist_filler", "steps_filler")):
        weight = np.array([r[key] for r in rows], float)
        value = np.array([r["twist_" + name.split("_")[-1]] for r in rows], float)
        good = np.isfinite(value) & (weight > 0)
        summary[name] = (float((value[good] * weight[good]).sum() / weight[good].sum())
                         if good.any() else float("nan"))
    summary["per_clip"] = per_clip
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--clips", required=True, type=pathlib.Path)
    parser.add_argument("--ground-truth", default=None,
                        help="ground-truth motion dir; needs --labels-root, since the "
                             "ground-truth pkls carry no atomic_labels")
    parser.add_argument("--labels-root", default=None,
                        help="the corpus label tree (labels.jsonl + labels/*.npy).  Ground "
                             "truth is split by its OWN annotation, not by a generated "
                             "arm's plan -- borrowing the plan would score the recording "
                             "against a label track it never had.")
    parser.add_argument("--out", type=pathlib.Path)
    arguments = parser.parse_args()

    clips = [line.strip() for line in arguments.clips.read_text().splitlines() if line.strip()]
    report = {"clips_requested": len(clips), "arms": {}}
    for entry in arguments.arm:
        name, _, directory = entry.rpartition("=")
        if not pathlib.Path(directory).is_dir():
            raise SystemExit("error: --arm {!r} points at {!r}, which does not exist."
                             .format(name, directory))
        report["arms"][name] = score_arm(directory, clips)
    if arguments.ground_truth:
        if not arguments.labels_root:
            raise SystemExit("error: --ground-truth needs --labels-root; ground-truth pkls "
                             "carry no atomic_labels and this tool will not invent them.")
        report["ground_truth"] = score_arm(arguments.ground_truth, clips,
                                           labels_by_clip=read_labels_root(arguments.labels_root))
        report["ground_truth_label_source"] = arguments.labels_root
    text = json.dumps(report, indent=2, sort_keys=True)
    if arguments.out:
        arguments.out.write_text(text)
    print("%-24s %8s %8s %8s %8s" % ("arm", "all", "named", "filler", "filler%"))
    rows = dict(report["arms"])
    if "ground_truth" in report:
        rows = {"ground truth": report["ground_truth"], **rows}
    for name, row in rows.items():
        print("%-24s %8.2f %8.2f %8.2f %7.1f%%" % (
            name, row["twist_all"], row["twist_named"], row["twist_filler"],
            100 * row["filler_frame_share"]))


if __name__ == "__main__":
    main()
