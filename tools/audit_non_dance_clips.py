#!/usr/bin/env python3
"""Does this clip's reconstruction contain a person DANCING, or a person standing still?

WHY THIS EXISTS.  On 2026-09-01 the operator watched a rendered sample and found
that ``wild_v5:7424041427996298534:clip000`` -- in the TEST split, and in the
ten-clip review set -- is not a dance video.  The footage is a close-up of a man
sitting at a table eating with chopsticks; its "ground truth" 3D motion is a
standing figure with both hands frozen near the face.  Nothing in the pipeline
had ever asked whether the tracked person was dancing.  ``ingest_wild_uploads``
selects a span where *a person is detected, continuously, in one shot, for at
least 340 frames*, and ``audit_clip_population`` counts how many people are on
screen and how big they are.  "Dance" entered this corpus only as a property of
the ACCOUNT the upload was crawled from, never as a property of the clip.

WHAT THIS MEASURES, AND ITS PROVENANCE.  ``pose_excursion_m``: decode the
released 151-D window to 24 world joints in METRES with
``model.atomic_completion.MotionGeometry`` (the same FK every other scorer in
this repo uses), subtract the root so the camera and the walk drop out, take
each joint's standard deviation over the window's 150 frames, reduce the three
axes with an L2 norm, and average over joints 1..23.  One number per window, in
metres; the clip's value is the MEDIAN over its windows.

  The statistic is mine -- CLAUDE.md section 2.1 gate 1 says to say so.  What it
  was checked against, before it was allowed to decide anything, is the paper's
  own definition of the thing this corpus is supposed to contain.
  ``atomicDance.pdf`` section 3.2, verbatim: "We aim to extract complete motion
  processes that are clearly distinct from their temporal context.  For example,
  a kick, which comprises the preparatory weight shift, leg extension, and
  recovery, forms a complete motion segment", and, for M3, a signature pose is
  "a relatively stable, sculptural posture" with the dynamics being "the
  transitions between static poses".  Both sentences describe a body that VISITS
  DISTINCT WHOLE-BODY CONFIGURATIONS.  A clip whose de-rooted skeleton never
  leaves a 3.8 cm ball has no distinct postures for M1 to cut between and
  nothing for M2/M3 to cluster; it is not a hard example of dance, it is not
  dance.  That is the whole content of the criterion.

NO WITHIN-CLIP AMPLITUDE NORMALIZATION, DELIBERATELY.  ``docs/DANCE_QUALITY_
DEFECTS.md`` section 12.1: four "beat" rulers all divided by the clip's own
median speed and all read backwards, because the defect they were chasing WAS
the per-clip amplitude.  Section 4.5 records the same trap one level up (D1/D2
divided by each row's own height).  Neither applies here by construction:
``vis.smpl_offsets`` is a single 24x3 constant shared by every clip in the
corpus, so the FK puts every reconstruction on the SAME skeleton and metres are
already comparable between clips.  Nothing here is divided by anything the clip
supplies.

INSTRUMENT CHECK (2026-09-01).  The release path (normalized array -> normalizer
-> ``MotionGeometry.joints``) was compared against the independent raw path
(``runs/wild_v5_song_gt_eval/motion/<key>.pkl``, which stores world joints
directly) on all 1,184 clips present in both: correlation 0.99999999993, median
relative difference 0.33%.  The 0.33% is the normalizer's own quantisation.

CONTROLS, BOTH DIRECTIONS.  ``runs/nondance_handlabels_v1.tsv`` holds 102 clips
labelled by looking at the TRACKED SUBJECT'S OWN CROP (``detections.npz``'s
``dancer_box``, i.e. the box GVHMR was handed) in the ingest footage.  Its
columns are ``sequence, tranche, label, why``, and the tranches are not
interchangeable -- what each one is allowed to support is different:

  A  40 clips drawn at random from the release, seed 20260901.  The only
     unbiased tranche; it is what the false-positive claim rests on, and its
     count of zero non-dance clips is also the only thing bounding the corpus
     base rate (0/40, so <=7.5% at 95%).
  B  the corpus's 24 LOWEST clips by this statistic, ENUMERATED not sampled.
  C,D  32 probes spread over ranks 25..780 -- the decision band.
  E  10 clips drawn at random from ranks 150..1500; all dance.
  F  the one flagged clip that ranks B..E had missed.
  OP the operator's clip.

  B..F were selected by this statistic's own rank, so the AUC quoted
  below is optimistic for it and pessimistic for anything uncorrelated with it.
  What is NOT biased is the enumeration: tranche B is every clip in that region,
  so "22 of the 24 lowest are not dance" is a census of the region, not a draw
  from it.

The readings:

* negative, and the criterion must FAIL them -- the operator's clip reads 0.0246
  m.  The corpus's 24 lowest clips were enumerated (not sampled) and eyeballed:
  22 are not dance (a still advertising poster; four talking heads; three
  spectators filming a battle; a child; tourists at a temple; a printed face on
  an LED wall; ...), 2 could not be judged, 0 are dance.
* positive, and the criterion must PASS them -- 40 clips drawn at random from
  the release, every one of them hand-confirmed dance.  Their lowest reading is
  0.0599 m, 1.59x the threshold; none is cut.
* the criterion CAN FAIL, and here it is failing.  8 of the 31 confirmed
  non-dance clips survive it: snowboarding (0.0795), two people on a sofa with
  their phones (0.0652), a clothing haul (0.0577), a printed face on an LED wall
  (0.0526), a clothing shop vlog (0.0489), a birthday cake (0.0477), a clapping
  spectator (0.0458), a child standing and talking (0.0411).  Recall on the
  hand-labelled negatives is 23/31 = 74%, precision 23/23 = 100%.  Those eight
  are re-listed as data in ``runs/nondance_census_v1_summary.json``'s
  ``hand_label_misses`` field, so a later reader need not trust this paragraph.

WHAT IT DOES NOT CLAIM.

* It is not a footage classifier.  It reads the RECONSTRUCTION, which is what
  training consumes.  Several of the clips it cuts are dance VIDEOS in which
  GVHMR tracked a bystander, a spectator, or a poster; the footage contains
  dancing and the motion does not.  That is the right side to measure from, but
  it means a "not dance" verdict here is a statement about the pkl, not about
  the upload.
* It does not separate the band 0.041..0.080 m, and no threshold on this
  statistic can.  That band holds torso-close-up hand dances (``手势舞``) and
  seated hand dances -- which ARE dance -- interleaved with talking heads,
  product videos and snowboarding, which are not.  The enumerated readings are
  in the docstring above and in ``runs/nondance_handlabels_v1.tsv``.
* It is not a duplicate of the already-computed visual statistics, and the
  cheapest possible detector does not exist.  Every column
  of ``runs/wild_clip_population.json`` was scored on the same hand-labelled set:
  the best of them, ``crop_travel_median``, reaches AUC 0.886;
  ``subject_area_median`` reaches 0.419 and ``people_median`` 0.441, i.e. WORSE
  than a coin in the obvious direction, because a non-dance clip is typically a
  close-up of ONE person and therefore has a LARGE subject and a LOW head count.
  ``pose_excursion_m`` reaches 0.990 and ``body_speed_mps`` 0.986 on the same
  100 judged labels.  (That comparison favours this statistic: five of the six
  tranches were drawn by its own rank, which concentrates the labels where it is
  most confident.  The unbiased parts are tranche A, on which it has no false
  positives, and the enumeration of tranche B.)
* It says nothing about time base (``runs/timebase_exclude_v1.jsonl``), about
  how many people are in frame, or about whether the reconstruction is good.

USAGE::

    # census the whole release
    python3 tools/audit_non_dance_clips.py \\
        --release /dev/shm/atomicdance-song-v5rekey/release_v3_timebase \\
        --output runs/nondance_census_v1.jsonl \\
        --summary runs/nondance_census_v1_summary.json

    # re-run the controls against the hand labels
    python3 tools/audit_non_dance_clips.py \\
        --release ... --labels runs/nondance_handlabels_v1.tsv --report-controls
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# The operating point.  It sits at the MIDPOINT of the tightest gap in the hand
# labels: the highest hand-confirmed NON-dance clip below it reads 0.0363 (five
# people standing for a group photo) and the lowest hand-confirmed DANCE clip
# above it reads 0.0394 (a torso-close-up hand dance).  Midpoint, not "just
# below the first dance", because the first version sat at 0.0350, i.e. exactly
# on a data point, and a 0.3% difference between two implementations of the same
# forward kinematics moved that clip across it.  A threshold a rounding error
# can flip is not a threshold.  The margin it now keeps -- 1.5 mm -- is 12x the
# 0.33% agreement measured between the two FK paths at this magnitude.
# Moving it further is a judgement about which error costs more, and the reading
# is written into every census row so the call can be re-made without re-running
# the FK.
DEFAULT_THRESHOLD_M = 0.0378

FPS = 30.0
LEG_JOINTS = (1, 2, 4, 5, 7, 8, 10, 11)


def pose_excursion(joints: np.ndarray) -> np.ndarray:
    """[B, T, 24, 3] world metres -> [B] metres.

    The de-rooted skeleton's RMS spread about its own time-mean, averaged over
    the 23 non-root joints.  Root-relative, so a moving camera and a travelling
    dancer both drop out -- what is left is how much of the body's own
    configuration space the window visits.
    """
    relative = joints - joints[:, :, :1, :]
    spread = relative.std(axis=1)                       # [B, 24, 3]
    spread = np.sqrt((spread ** 2).sum(axis=-1))        # [B, 24]
    return spread[:, 1:].mean(axis=1)


def body_speed(joints: np.ndarray) -> np.ndarray:
    """[B, T, 24, 3] -> [B] metres per second.

    Reported beside the decision, never used by it.  It is the repository's
    'energy' column (de-rooted mean joint speed) and it is here so a reader can
    see the two disagree -- CLAUDE.md section 2.1 gate 4 says that when two
    rulers point different ways you stop, and you cannot stop at a disagreement
    you never printed.
    """
    relative = joints - joints[:, :, :1, :]
    step = np.sqrt(((relative[:, 1:] - relative[:, :-1]) ** 2).sum(axis=-1)) * FPS
    return step[:, :, 1:].mean(axis=(1, 2))


def leg_excursion(joints: np.ndarray) -> np.ndarray:
    """Same statistic over hips/knees/ankles/toes only -- the paper's 'preparatory
    weight shift, leg extension'.  Reported, not used: a hand dance filmed as a
    torso close-up reads near zero here and IS dance."""
    relative = joints - joints[:, :, :1, :]
    spread = np.sqrt((relative.std(axis=1) ** 2).sum(axis=-1))
    return spread[:, list(LEG_JOINTS)].mean(axis=1)


def clip_features(release: pathlib.Path, splits=("train", "val", "test"),
                  batch=4096, device="cuda"):
    """Median-over-windows features, one row per clip in the release."""
    import torch
    from model.atomic_completion import MotionGeometry

    normalizer = torch.load(release / "normalizer.pt", map_location="cpu", weights_only=False)
    geometry = MotionGeometry(normalizer)
    if device == "cuda" and torch.cuda.is_available():
        geometry = geometry.cuda()
    else:
        device = "cpu"
    geometry.eval()

    rows = {}
    for split in splits:
        names = json.load(open(release / split / "names.json"))
        motion = np.load(release / split / "motion.npy", mmap_mode="r")
        if len(names) != len(motion):
            raise SystemExit("{}: names.json has {} rows, motion.npy has {}".format(
                split, len(names), len(motion)))
        keys = [n.rsplit("_slice", 1)[0] for n in names]
        gathered = {}
        for start in range(0, len(names), batch):
            chunk = torch.from_numpy(np.ascontiguousarray(motion[start:start + batch]))
            if device == "cuda":
                chunk = chunk.cuda()
            with torch.no_grad():
                joints = geometry.joints(chunk).float().cpu().numpy()
            triple = np.stack([pose_excursion(joints), body_speed(joints),
                               leg_excursion(joints)], axis=1)
            for offset in range(len(triple)):
                gathered.setdefault(keys[start + offset], []).append(triple[offset])
        for key, values in gathered.items():
            stack = np.asarray(values, np.float64)
            rows[key] = {
                "sequence": key,
                "split": split,
                "windows": len(values),
                "pose_excursion_m": float(np.median(stack[:, 0])),
                "body_speed_mps": float(np.median(stack[:, 1])),
                "leg_excursion_m": float(np.median(stack[:, 2])),
            }
    return rows


def decide(row, threshold=DEFAULT_THRESHOLD_M):
    """The gate.  One comparison, one direction, on one number in metres."""
    return "not_dance" if row["pose_excursion_m"] < threshold else "dance"


def read_labels(path):
    labels = {}
    for line in pathlib.Path(path).read_text().splitlines():
        if not line.strip():
            continue
        key, tranche, label, why = line.split("\t")
        labels[key] = {"tranche": tranche, "label": label, "why": why}
    return labels


def confusion(rows, labels, threshold):
    """Rows x hand labels -> counts, plus the list of clips it gets wrong."""
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    misses, false_cuts, unjudged = [], [], []
    for key, hand in labels.items():
        if key not in rows:
            continue
        verdict = decide(rows[key], threshold)
        if hand["label"] == "unsure":
            unjudged.append((key, verdict, rows[key]["pose_excursion_m"]))
            continue
        truth_not_dance = hand["label"] == "not_dance"
        cut = verdict == "not_dance"
        if truth_not_dance and cut:
            counts["tp"] += 1
        elif truth_not_dance and not cut:
            counts["fn"] += 1
            misses.append((key, rows[key]["pose_excursion_m"], hand["why"]))
        elif not truth_not_dance and cut:
            counts["fp"] += 1
            false_cuts.append((key, rows[key]["pose_excursion_m"], hand["why"]))
        else:
            counts["tn"] += 1
    return counts, misses, false_cuts, unjudged


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release", required=True,
                        help="release root holding normalizer.pt and {train,val,test}/")
    parser.add_argument("--output", help="one JSON object per clip")
    parser.add_argument("--summary", help="counts by split")
    parser.add_argument("--labels", default=str(REPO / "runs/nondance_handlabels_v1.tsv"))
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD_M,
                        help="pose_excursion_m below which a clip is called not-dance")
    parser.add_argument("--report-controls", action="store_true",
                        help="score the criterion against the hand labels and stop")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--features-cache",
                        help="reuse a previous run's per-clip features instead of re-running FK")
    args = parser.parse_args()

    release = pathlib.Path(args.release)
    if args.features_cache and pathlib.Path(args.features_cache).is_file():
        rows = json.load(open(args.features_cache))
    else:
        rows = clip_features(release, device=args.device)
        if args.features_cache:
            json.dump(rows, open(args.features_cache, "w"))

    if args.report_controls:
        labels = read_labels(args.labels)
        counts, misses, false_cuts, unjudged = confusion(rows, labels, args.threshold)
        covered = counts["tp"] + counts["fn"] + counts["fp"] + counts["tn"]
        print("threshold pose_excursion_m < {:.4f} m".format(args.threshold))
        print("hand-labelled clips scored: {} (+{} unsure)".format(covered, len(unjudged)))
        print("  cut and NOT dance (tp) {:3d}".format(counts["tp"]))
        print("  cut but IS dance  (fp) {:3d}".format(counts["fp"]))
        print("  kept and IS dance (tn) {:3d}".format(counts["tn"]))
        print("  kept but NOT dance(fn) {:3d}".format(counts["fn"]))
        denom = counts["tp"] + counts["fn"]
        if denom:
            print("  recall {:.3f}".format(counts["tp"] / denom))
        if counts["tp"] + counts["fp"]:
            print("  precision {:.3f}".format(counts["tp"] / (counts["tp"] + counts["fp"])))
        for name, group in (("MISSED (non-dance it keeps)", misses),
                            ("WRONGLY CUT (dance it cuts)", false_cuts)):
            print("\n{}:".format(name))
            for key, value, why in sorted(group, key=lambda g: g[1]):
                print("  {:.4f}  {}  {}".format(value, key, why))
        print("\nunsure, for the record:")
        for key, verdict, value in sorted(unjudged, key=lambda g: g[2]):
            print("  {:.4f}  {:9s} {}".format(value, verdict, key))
        return

    decisions = []
    for key in sorted(rows):
        row = dict(rows[key])
        row["decision"] = decide(row, args.threshold)
        row["threshold_m"] = args.threshold
        decisions.append(row)

    if args.output:
        with open(args.output, "w") as handle:
            for row in decisions:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    by_split = {}
    for row in decisions:
        bucket = by_split.setdefault(row["split"], {"clips": 0, "not_dance": 0})
        bucket["clips"] += 1
        bucket["not_dance"] += row["decision"] == "not_dance"
    for bucket in by_split.values():
        bucket["not_dance_share"] = round(bucket["not_dance"] / bucket["clips"], 5)

    labels = read_labels(args.labels) if pathlib.Path(args.labels).is_file() else {}
    counts, misses, false_cuts, unjudged = confusion(rows, labels, args.threshold)
    kept_negatives = sorted(value for _, value, _ in misses)
    summary = {
        "generated_by": "tools/audit_non_dance_clips.py",
        "threshold_m": args.threshold,
        "release": str(release),
        "criterion": "pose_excursion_m < {} (metres, de-rooted RMS joint spread, "
                     "median over 150-frame windows)".format(args.threshold),
        "clips": len(decisions),
        "not_dance": sum(r["decision"] == "not_dance" for r in decisions),
        "by_split": by_split,
        "hand_label_confusion": counts,
        "hand_label_unsure": len(unjudged),
        "hand_label_misses": [{"sequence": key, "pose_excursion_m": round(value, 5),
                               "what_it_is": why} for key, value, why in
                              sorted(misses, key=lambda m: m[1])],
        "hand_label_false_cuts": [{"sequence": key, "pose_excursion_m": round(value, 5),
                                   "what_it_is": why} for key, value, why in
                                  sorted(false_cuts, key=lambda m: m[1])],
        # The interval this criterion demonstrably CANNOT separate: the lowest
        # and the highest reading among hand-confirmed non-dance clips that the
        # gate keeps.  Hand-confirmed DANCE clips are interleaved through it.
        "unseparated_band_m": [round(kept_negatives[0], 5), round(kept_negatives[-1], 5)]
        if kept_negatives else None,
        "what_this_does_not_claim": (
            "a not_dance verdict is about the RECONSTRUCTION, not the upload: "
            "several cut clips are dance videos in which the tracker followed a "
            "spectator or a poster.  The band 0.041..0.080 m is not separated by "
            "this or any threshold on this statistic -- torso-close-up hand "
            "dances live there beside talking heads and product videos."),
    }
    if args.summary:
        pathlib.Path(args.summary).write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
