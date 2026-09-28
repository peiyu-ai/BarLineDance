#!/usr/bin/env python3
"""The high-solo account corpus, selected by a measured threshold rather than by name.

The wild corpus is 30.2% solo: ``audit_clip_population`` counts ``person_count``
per frame in every clip's ``detections.npz`` and calls a clip solo when at most
one person is on screen in at least 90% of its frames.  That share runs from
0.079 to 0.904 *by uploader account*, and the accounts at the top are the ones
whose 3D reconstruction is least contaminated by the two effects
``report_population_quality`` measured: foot skate degrades 1.07x-1.47x on
multi-person clips (growing with subject size), root speed 1.04x-1.15x, while
jitter shows no crowd effect at all.

**The selection is a threshold, not a list of names.**  A hand-typed account
list has no provenance: it cannot be re-derived when the population audit is
re-run, and nothing catches it drifting out of date.  Here the parameter is
``--min-solo-share`` and the account set is derived from it, so the corpus can
be rebuilt from the threshold alone and a changed measurement changes the
corpus rather than silently disagreeing with it.

Three exclusions, each reported rather than applied quietly:

* **orphans** -- clip names the fps re-cut no longer produces.  These
  credentials can PUT over a key but cannot delete one, so an orphan's objects
  live in OSS forever and every directory listing keeps returning them.  Ten of
  them are in the released bundle, which is why this cannot be skipped: reading
  the release is not sufficient protection.
* **clips outside the released bundle** -- the ingest tree holds 17,952 clip
  directories and the release holds 13,783.  Only the released ones have 3D,
  S3D and captions, and this corpus is defined to reuse them.
* **uploads that produced no released clip** -- counted per account and
  reported as the ingest backlog, because "this account has 628 uploads in OSS
  and 506 that yielded a clip" is a decision to make, not a number to discover
  after a training run.

**Footage and segmented material are two numbers, not one (2026-08-20).**
``totals.hours`` sums each record's ``motion_frames``, i.e. the footage the corpus
contains.  Every cut this repo had built until now ran frame 0 to the last frame,
so the two were the same and one number was honest.  A beat-grid cut is not like
that: ``segment_on_music_beats.py --edges drop`` starts at the first beat and stops
at the last, so on ``runs/wild_v4_seg_beat4h`` 1.69 h of the 10.10 h -- the lead-in
before the first beat and the trail-out after the last -- sits inside no segment.
Reporting only ``hours`` would let the corpus advertise 10.09 h to a consumer that
receives 8.41 h of segments, so ``totals.segmented_hours`` and
``segmented_fraction`` are carried beside it, taken from each record's
``covered_frames`` (falling back to the boundary span for cuts that predate it).

Usage::

    python3 tools/select_clean_accounts_corpus.py --min-solo-share 0.55 \\
        --output-dir runs/clean5
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
from typing import Dict, List, Mapping, Sequence

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import redo_manifest  # noqa: E402

SCHEMA = "atomicdance-clean-accounts-corpus-v1"


class SelectionError(RuntimeError):
    """A corpus that would be wrong is refused rather than written."""


def upload_of(stem: str) -> str:
    return stem.split("__")[0]


def load_orphans(path: pathlib.Path) -> set:
    if path is None or not path.is_file():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()}


def feature_frames(features_dir: pathlib.Path, stems: Sequence[str]) -> Dict[str, int]:
    """Row count of each clip's S3D feature array, as it is on disk right now.

    The segmentation's ``motion_frames`` is the feature length *at the time it
    ran* (``segment_visual_atomics.py``), so the two disagree exactly when the
    features were rebuilt afterwards -- which is what happened on 2026-08-19,
    when 3,991 features were re-extracted after the published M1 had already
    been computed.  Reading only ``meta`` keeps this cheap: the array itself is
    never decompressed.
    """
    import concurrent.futures as futures

    import numpy as np

    def one(stem: str):
        path = features_dir / (stem + ".npz")
        if not path.is_file():
            return stem, None
        try:
            with np.load(path, allow_pickle=False) as bundle:
                return stem, int(json.loads(str(bundle["meta"]))["motion_frames"])
        except Exception:                                     # noqa: BLE001
            return stem, None

    with futures.ThreadPoolExecutor(max_workers=32) as pool:
        return dict(pool.map(one, stems))


def select(population: pathlib.Path, group_keys: pathlib.Path,
           segmentation: pathlib.Path, orphans_path: pathlib.Path,
           min_solo_share: float, oss_listing: pathlib.Path = None,
           features_dir: pathlib.Path = None,
           allow_stale_segmentation: bool = False,
           stale_path: pathlib.Path = None) -> Dict[str, object]:
    pop = json.loads(population.read_text(encoding="utf-8"))
    accounts = pop["by_account"]
    chosen = sorted((name for name, row in accounts.items()
                     if row["solo_share"] > min_solo_share),
                    key=lambda name: -accounts[name]["solo_share"])
    if not chosen:
        raise SelectionError(
            "no account has solo_share > {}; the highest is {:.3f}".format(
                min_solo_share, max(row["solo_share"] for row in accounts.values())))

    account_of = json.loads(group_keys.read_text(encoding="utf-8"))
    report = json.loads(segmentation.read_text(encoding="utf-8"))
    orphans = load_orphans(orphans_path)
    # Clips whose derived 3D/S3D could not be rebuilt from the bytes the clip
    # now holds.  Kept apart from ``orphans`` rather than folded into it: an
    # orphan is a name nothing may ever read again, while these are live names
    # whose derivatives are provably from an older cut of the video -- on
    # 2026-08-20, 102 clips on which GVHMR's visual odometry diverged, so stage
    # B left the pre-re-cut 3D in place.  Reporting both under one label would
    # make the corpus claim a reason it did not measure.
    stale: set = set()
    if stale_path is not None:
        for stage in redo_manifest.STAGES:
            stale |= set(redo_manifest.load(stale_path, stage))

    keep: List[str] = []
    excluded = collections.Counter()
    excluded_examples: Dict[str, List[str]] = collections.defaultdict(list)
    unresolved: List[str] = []
    per = collections.defaultdict(lambda: {"clips": 0, "segments": 0, "frames": 0,
                                           "segmented_frames": 0, "uploads": set()})
    for record in report["records"]:
        stem = record["sequence"]
        account = account_of.get(upload_of(stem))
        if account is None:
            unresolved.append(stem)
            continue
        if account not in chosen:
            continue
        if stem in orphans:
            excluded["orphan"] += 1
            if len(excluded_examples["orphan"]) < 5:
                excluded_examples["orphan"].append(stem)
            continue
        # After the orphan test, so a name that is both is counted once and
        # under the stronger statement.
        if stem in stale:
            excluded["stale_derivatives"] += 1
            if len(excluded_examples["stale_derivatives"]) < 5:
                excluded_examples["stale_derivatives"].append(stem)
            continue
        keep.append(stem)
        entry = per[account]
        entry["clips"] += 1
        entry["segments"] += len(record["segments"])
        entry["frames"] += int(record["motion_frames"])
        # A beat-grid cut does not cover the whole clip: the lead-in before the
        # first beat and the trail-out after the last are outside every segment,
        # and on runs/wild_v4_seg_beat4h that is 1.69 h of the 10.10 h.  "hours"
        # below is footage; "segmented_hours" is what a downstream consumer of
        # the segments actually gets, and reporting only the first would let the
        # corpus claim 10.09 h of material that is 8.41 h of segments.
        entry["segmented_frames"] += int(record.get(
            "covered_frames", record["boundaries"][-1] - record["boundaries"][0]))
        entry["uploads"].add(upload_of(stem))

    # An unresolved upload is an error, not a "?" bucket: a silent catch-all is
    # how the genre pre-split quietly disappeared once already.
    if unresolved:
        raise SelectionError(
            "{} clip(s) have no recorded account, e.g. {}".format(
                len(unresolved), ", ".join(sorted(unresolved)[:5])))
    if not keep:
        raise SelectionError("threshold {} selected {} account(s) but no clips".format(
            min_solo_share, len(chosen)))

    # The gate that this tool's first version did not have, and paid for: it
    # reported hours and segment counts read out of a segmentation that indexes
    # feature arrays which no longer exist.  On 2026-08-20 that understated this
    # corpus by 20,141 frames (9.915 h reported against 10.101 h real), all of it
    # in one account, and nothing said so.  Presence of the segmentation file is
    # not the question; whether it describes the features on disk is.
    stale_segmentation: Dict[str, object] = {"checked": False}
    if features_dir is not None:
        current = feature_frames(features_dir, keep)
        frames_in_segmentation = {r["sequence"]: int(r["motion_frames"])
                                  for r in report["records"]}
        disagree = [stem for stem in keep
                    if current.get(stem) is not None
                    and current[stem] != frames_in_segmentation[stem]]
        unreadable = [stem for stem in keep if current.get(stem) is None]
        by_account = collections.Counter(
            account_of[upload_of(stem)] for stem in disagree)
        stale_segmentation = {
            "checked": True,
            "features_dir": str(features_dir),
            "clips_whose_segmentation_predates_their_features": len(disagree),
            "clips_with_unreadable_features": len(unreadable),
            "by_account": dict(by_account),
            "examples": sorted(disagree)[:5],
            "reading": "the segmentation's motion_frames is the feature length at "
                       "the time it ran; a disagreement means these boundaries "
                       "index an array that is no longer on disk, so every count "
                       "and duration derived from them is wrong",
        }
        if disagree and not allow_stale_segmentation:
            raise SelectionError(
                "{} of {} clips have a segmentation older than their S3D features "
                "(e.g. {}); every hour and segment count from it would be wrong. "
                "Point --segmentation at a segmentation built on the current "
                "features, or pass --allow-stale-segmentation to record the "
                "corpus anyway with this defect named in corpus.json".format(
                    len(disagree), len(keep), ", ".join(sorted(disagree)[:3])))

    # The ingest backlog: uploads OSS holds for these accounts that yielded no
    # released clip.  Only computable when a listing is supplied; absent, say so
    # rather than reporting zero.
    backlog: Dict[str, object]
    if oss_listing is not None and oss_listing.is_file():
        oss_uploads = collections.Counter()
        for line in oss_listing.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line.endswith(".mp4"):
                continue
            account, _, name = line.rpartition("/")
            if account in chosen:
                oss_uploads[account] += 1
        backlog = {
            "supplied": True,
            "by_account": {
                account: {
                    "uploads_in_oss": oss_uploads[account],
                    "uploads_with_a_released_clip": len(per[account]["uploads"]),
                    "uploads_never_yielding_a_clip":
                        oss_uploads[account] - len(per[account]["uploads"]),
                } for account in chosen},
        }
    else:
        backlog = {"supplied": False,
                   "reading": "no OSS listing supplied; the ingest backlog is "
                              "unmeasured here, not zero"}

    total_frames = sum(entry["frames"] for entry in per.values())
    total_segmented = sum(entry["segmented_frames"] for entry in per.values())
    total_segments = sum(entry["segments"] for entry in per.values())
    return {
        "schema_version": SCHEMA,
        "selection": {
            "rule": "account solo_share > min_solo_share, whole accounts",
            "min_solo_share": min_solo_share,
            "solo_definition": pop["solo_definition"],
            "measured_by": pop.get("generated_by", "tools/audit_clip_population.py"),
            "accounts": chosen,
        },
        "inputs": {
            "population": str(population),
            "group_keys": str(group_keys),
            "segmentation": str(segmentation),
            "orphans": str(orphans_path) if orphans_path else None,
            "stale": str(stale_path) if stale_path else None,
        },
        "totals": {
            "accounts": len(chosen),
            "clips": len(keep),
            "segments": total_segments,
            "frames": total_frames,
            "hours": round(total_frames / 30.0 / 3600.0, 4),
            "segmented_frames": total_segmented,
            "segmented_hours": round(total_segmented / 30.0 / 3600.0, 4),
            "segmented_fraction": (round(total_segmented / total_frames, 4)
                                   if total_frames else None),
            "segments_per_prototype_at_k100": round(total_segments / 100.0, 1),
            "paper_segments_per_prototype": 268.57,
        },
        "by_account": {
            account: {
                "solo_share": accounts[account]["solo_share"],
                "uploads": len(per[account]["uploads"]),
                "clips": per[account]["clips"],
                "segments": per[account]["segments"],
                "hours": round(per[account]["frames"] / 30.0 / 3600.0, 4),
                "segmented_hours": round(per[account]["segmented_frames"] / 30.0 / 3600.0, 4),
                "frame_fraction": round(per[account]["frames"] / total_frames, 4),
            } for account in chosen},
        "excluded": {
            "counts": dict(excluded),
            "examples": {key: value for key, value in excluded_examples.items()},
            "reading": "orphans are names no consumer may read; their objects "
                       "cannot be deleted from this bucket, so they are carried "
                       "by name rather than by absence.  stale_derivatives are "
                       "live names whose 3D or S3D was built from an older cut "
                       "of the video and could not be rebuilt; they are "
                       "excluded because no gate downstream can tell them from "
                       "clean ones",
        },
        "ingest_backlog": backlog,
        "segmentation_vs_features": stale_segmentation,
        "clips": sorted(keep),
    }


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--population", type=pathlib.Path,
                        default=REPO / "runs/wild_clip_population.json")
    parser.add_argument("--group-keys", type=pathlib.Path,
                        default=REPO / "runs/wild_v4_group_keys.json")
    parser.add_argument("--segmentation", type=pathlib.Path,
                        default=REPO / "runs/wild_v4_seg/segmentation.json")
    parser.add_argument("--orphans", type=pathlib.Path,
                        default=pathlib.Path("/cache/atomicdance-assets/scratch/"
                                             "c1/refix/orphans.txt"))
    parser.add_argument("--oss-listing", type=pathlib.Path, default=None,
                        help="one '<account>/<upload>.mp4' per line, to measure "
                             "the ingest backlog")
    parser.add_argument("--features-dir", type=pathlib.Path,
                        default=pathlib.Path("/cache/atomicdance-assets/data/wild_visual_s3d"),
                        help="S3D features the segmentation must still describe; "
                             "pass an empty string to skip the check")
    parser.add_argument("--allow-stale-segmentation", action="store_true",
                        help="record the corpus even though its segmentation "
                             "predates its features, naming the defect in corpus.json")
    parser.add_argument("--stale", type=pathlib.Path, default=None,
                        help="an audit_clip_freshness report; clips still stale "
                             "at any stage are excluded and counted separately "
                             "from orphans")
    parser.add_argument("--min-solo-share", type=float, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    features_dir = args.features_dir if str(args.features_dir) else None
    result = select(args.population, args.group_keys, args.segmentation,
                    args.orphans, args.min_solo_share, args.oss_listing,
                    features_dir=features_dir,
                    allow_stale_segmentation=args.allow_stale_segmentation,
                    stale_path=args.stale)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clips = result.pop("clips")
    (args.output_dir / "clips.txt").write_text("\n".join(clips) + "\n", encoding="utf-8")
    result["clips_manifest"] = str(args.output_dir / "clips.txt")
    (args.output_dir / "corpus.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    totals = result["totals"]
    print("accounts with solo_share > {}: {}".format(
        args.min_solo_share, ", ".join(result["selection"]["accounts"])))
    print("{} clips / {} segments / {:.2f} h of footage, {:.2f} h inside a segment "
          "({:.1%})".format(totals["clips"], totals["segments"], totals["hours"],
                            totals["segmented_hours"], totals["segmented_fraction"]))
    print("at K=100 that is {} segments per prototype (paper {})".format(
        totals["segments_per_prototype_at_k100"], totals["paper_segments_per_prototype"]))
    for account, row in result["by_account"].items():
        print("  {:<24} solo {:.3f}  {:>4} uploads  {:>4} clips  {:>6} seg  "
              "{:.2f} h  {:.1%} of frames".format(
                  account, row["solo_share"], row["uploads"], row["clips"],
                  row["segments"], row["hours"], row["frame_fraction"]))
    check = result["segmentation_vs_features"]
    if check.get("checked"):
        n = check["clips_whose_segmentation_predates_their_features"]
        print("segmentation vs features: {} clip(s) disagree{}".format(
            n, "" if not n else " -- " + ", ".join(
                "{} {}".format(v, k) for k, v in check["by_account"].items())))
    else:
        print("segmentation vs features: NOT CHECKED (no --features-dir)")
    for key, count in result["excluded"]["counts"].items():
        print("excluded {}: {}".format(key, count))
    backlog = result["ingest_backlog"]
    if backlog.get("supplied"):
        for account, row in backlog["by_account"].items():
            if row["uploads_never_yielding_a_clip"]:
                print("  ingest backlog {:<22} {} of {} uploads yielded no clip".format(
                    account, row["uploads_never_yielding_a_clip"], row["uploads_in_oss"]))
    else:
        print(backlog["reading"])
    print("wrote {} and {}".format(args.output_dir / "clips.txt",
                                   args.output_dir / "corpus.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
