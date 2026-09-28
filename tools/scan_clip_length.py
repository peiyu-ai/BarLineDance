#!/usr/bin/env python3
"""Measure what clip length costs, so the constant is this repo's and not Lodge's.

The wild corpus is cut into 16-second clips by ``ffmpeg -f segment``.  That
number came from Lodge's ``preprocess_wild_videos.py``, where the comment reads
"match historical 16s cache", and the lower bound of 180 frames is annotated
``past60 + future120`` -- Lodge's motion-continuation window.  Neither constant
was chosen for AtomicDance, and both change what this repo produces:

* Alg. 1 sets its cluster count to ``T / frames_per_cluster``, so a shorter clip
  is segmented **more coarsely**.  Measured on the published corpus, mean
  segment length runs 1.171 s at 6-7 s clips down to 1.022 s at full length,
  monotonically -- and the paper's target is 0.81 s.
* The other direction is not free either: the camera track is estimated over
  the whole sequence, and monocular drift accumulates.  The wild corpus already
  wanders 2.26x more than mocap at the supporting foot.

So there is a real trade-off and it should be measured rather than argued.  This
cuts the *same footage* to several lengths, runs the production 3D chain on each,
and reports the physical measures as a function of length, against the mocap p99
thresholds ``audit_wild_3d_quality.py`` already established.

Two metric families are reported per length, and the difference between them is
the point:

``full``
    over the whole cut -- what downstream actually consumes.  Drift has more
    frames to accumulate over, so this is expected to worsen with length; the
    question is where it crosses the mocap threshold.
``head``
    over the first ``--common-window`` seconds only -- *identical footage in
    every row*.  If this degrades with length too, then a longer clip damages
    even the part it shares with a short one, which is a statement about the
    estimator rather than about exposure.

Reported alongside is ``visual_odometry_attempts_used``: final success rate is a
censored measure, because a clip rescued on the third retry and one that tracked
first time both end up in the corpus looking identical.

Sampling caveat, stated here because it cannot be designed away: holding content
fixed across lengths requires uploads at least as long as the longest cut, so
the sample is drawn from the long tail of the corpus (p90 of upload duration is
49 s).  Long uploads are not a random sample of dance uploads.  This measures
how the estimator responds to length, not the corpus-average difficulty.

Stages, each resumable and skipped when its output exists:

  cut     select uploads, cut every length, write per-GPU video lists
  report  convert whatever extracted, measure, tabulate

The extraction between them is the existing chain, driven by
``tools/run_clip_length_scan.sh``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import subprocess
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.audit_wild_3d_quality import sequence_metrics          # noqa: E402

FPS = 30
# convert_gvhmr_result.py's name for the packed 151-D array.
MOTION_FILE = "atomic_motion_151.npy"

# The columns a length decision actually turns on: vertical stability (the one
# axis where wild already differs from mocap), tracking blowups, and skate.
# These are ``sequence_metrics``' own key names, and ``root_speed_max`` rather
# than ``root_speed_p99`` because that is the one the corpus audit's published
# threshold (mocap p99 = 4.5332) was computed from -- taking the other would
# quietly compare against a different bar.
GATED = ("lowest_toe_spread", "root_speed_max", "jitter_ratio",
         "frozen_fraction", "skate_p95")


def clip_name(stem: str, seconds: int) -> str:
    return "{}__L{:03d}.mp4".format(stem, seconds)


def probe_duration(video: pathlib.Path) -> Optional[float]:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(video)],
            capture_output=True, text=True, timeout=60)
        return float(out.stdout.strip())
    except (ValueError, subprocess.SubprocessError):
        return None


def select_uploads(videos_dir: pathlib.Path, *, count: int, min_seconds: float,
                   workers: int = 16) -> List[pathlib.Path]:
    """Uploads long enough that every length is the same footage, chosen stably.

    Order is by hash of the stem, not by name: filenames are TikTok ids, which
    sort by upload time, and taking the first N by name would sample one week.
    """
    candidates = sorted(videos_dir.glob("*.mp4"),
                        key=lambda p: hashlib.sha256(p.stem.encode()).hexdigest())
    chosen: List[pathlib.Path] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        # Probing all 7,947 to pick 24 is wasted IO; walk the hash order in
        # blocks and stop at the first block that fills the quota.
        for start in range(0, len(candidates), 256):
            block = candidates[start:start + 256]
            for video, duration in zip(block, pool.map(probe_duration, block)):
                if duration is not None and duration >= min_seconds:
                    chosen.append(video)
                    if len(chosen) == count:
                        return chosen
    return chosen


def cut(video: pathlib.Path, seconds: int, destination: pathlib.Path) -> bool:
    """One length of one upload, on the recipe the corpus is actually built with.

    Lodge's segmenter re-encodes to 30 fps with ultrafast x264 and aac audio;
    a scan that cut differently would measure the cut, not the length.  Every
    cut starts at t=0, which is what makes ``head`` the same footage everywhere.
    """
    if destination.exists():
        return True
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".partial.mp4")
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(video), "-t", str(seconds), "-r", str(FPS),
         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(partial)],
        capture_output=True, text=True, timeout=1800)
    if result.returncode != 0 or not partial.exists():
        partial.unlink(missing_ok=True)
        return False
    # Rename only on success: a half-written mp4 left at the final path would be
    # skipped as "already cut" on the next run and extracted as if it were whole.
    partial.rename(destination)
    return True


def stage_cut(*, videos_dir: pathlib.Path, work_dir: pathlib.Path,
              lengths: Sequence[int], uploads: int, gpus: Sequence[int]) -> Dict:
    clips_dir = work_dir / "clips"
    selection_path = work_dir / "selection.json"
    selected: List[pathlib.Path] = []
    if selection_path.exists():
        selected = [pathlib.Path(p) for p in json.loads(
            selection_path.read_text(encoding="utf-8"))["uploads"]]
    if len(selected) < uploads:
        # Selection walks a fixed hash order, so asking for more later extends
        # the sample rather than redrawing it -- the clips already extracted
        # stay valid, and a scan that came out underpowered can be widened
        # without paying for it twice.
        selected = select_uploads(videos_dir, count=uploads,
                                  min_seconds=max(lengths) + 1.0)
        work_dir.mkdir(parents=True, exist_ok=True)
        selection_path.write_text(json.dumps(
            {"uploads": [str(p) for p in selected],
             "min_seconds": max(lengths) + 1.0}, indent=1), encoding="utf-8")
    print("selected {} uploads >= {}s".format(len(selected), max(lengths) + 1))

    made, failed = [], []
    for video in selected:
        for seconds in lengths:
            # The length goes in the *filename*, not only the directory: the
            # extractor keys its output directory on the video stem alone, so
            # six same-named cuts would all write to one place and the scan
            # would silently compare a length against itself.
            destination = clips_dir / "L{:03d}".format(seconds) / clip_name(video.stem, seconds)
            (made if cut(video, seconds, destination) else failed).append(destination)
        print("  cut {}".format(video.stem), flush=True)

    # Shard by hash of the *whole* name, which carries the length: otherwise one
    # GPU would get every length of the same upload and the shards would be
    # balanced in count but not in seconds of video.
    lists_dir = work_dir / "video_lists"
    lists_dir.mkdir(parents=True, exist_ok=True)
    for index, gpu in enumerate(gpus):
        mine = [p for p in made
                if int(hashlib.sha256(str(p).encode()).hexdigest(), 16) % len(gpus) == index]
        (lists_dir / "gpu{}.txt".format(gpu)).write_text(
            "\n".join(str(p.resolve()) for p in mine) + "\n", encoding="utf-8")
        print("  gpu {}: {} clips, {} s of video".format(
            gpu, len(mine), sum(int(p.parent.name[1:]) for p in mine)))

    summary = {"uploads": len(selected), "lengths": list(lengths),
               "clips_cut": len(made), "clips_failed": len(failed)}
    print(json.dumps(summary, indent=1))
    return summary


def measure_one(payload):
    converted_dir, seconds, common_frames = payload
    stem = converted_dir.name
    record: Dict[str, object] = {"clip": stem, "length_s": seconds}
    try:
        from tools.convert_motion_to_guofeats import motion_151_to_joints

        motion = np.load(converted_dir / MOTION_FILE)
        joints = motion_151_to_joints(motion)
    except Exception as error:                      # noqa: BLE001 - recorded, not raised
        record["error"] = repr(error)
        return record
    record["full"] = sequence_metrics(joints)
    # The shared window is what separates "more frames to drift over" from
    # "a longer sequence estimates the shared part worse".
    if len(joints) >= common_frames:
        record["head"] = sequence_metrics(joints[:common_frames])
    return record


def mocap_thresholds(audit: pathlib.Path) -> Dict[str, float]:
    """p99 of each gated metric on mocap -- the same bar the corpus audit used.

    A threshold picked by hand would be a preference; the p99 of motion capture
    is "worse than the worst 1% of data that is right by construction".
    """
    records = json.loads(audit.read_text(encoding="utf-8"))["reference_records"]
    good = [r for r in records if "error" not in r]
    return {key: round(float(np.percentile([r[key] for r in good], 99)), 4)
            for key in GATED}


def stage_report(*, work_dir: pathlib.Path, converted_root: pathlib.Path,
                 raw_root: pathlib.Path, lengths: Sequence[int],
                 audit: Optional[pathlib.Path], common_window: int,
                 output: pathlib.Path, workers: int = 16) -> Dict:
    clips_dir = work_dir / "clips"
    common_frames = common_window * FPS

    payloads, attempted = [], []
    for seconds in lengths:
        for clip in sorted((clips_dir / "L{:03d}".format(seconds)).glob("*.mp4")):
            attempted.append((clip.stem, seconds))
            converted = converted_root / clip.stem
            if (converted / MOTION_FILE).exists():
                payloads.append((converted, seconds, common_frames))

    records: List[Dict] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for record in pool.map(measure_one, payloads, chunksize=4):
            records.append(record)

    # Extraction outcome per clip: the raw side knows about tracking, and a clip
    # that never produced a result is a data point about length, not a gap.
    outcomes: Dict[str, Dict] = {}
    for stem, seconds in attempted:
        meta_path = raw_root / stem / "extract_meta.json"
        entry = {"length_s": seconds, "extracted": meta_path.exists(), "attempts": None}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            entry["attempts"] = meta.get("visual_odometry_attempts_used")
            entry["preprocess_seconds"] = meta.get("preprocess_seconds")
        entry["failed_marker"] = (raw_root / stem / ".extract_failed").exists()
        outcomes[(stem, seconds)] = entry

    thresholds = mocap_thresholds(audit) if audit and audit.exists() else {}

    by_length: List[Dict] = []
    for seconds in lengths:
        mine = [r for r in records if r["length_s"] == seconds and "error" not in r]
        outs = [v for k, v in outcomes.items() if v["length_s"] == seconds]
        attempts = [o["attempts"] for o in outs if o["attempts"]]
        row: Dict[str, object] = {
            "length_s": seconds,
            "attempted": len(outs),
            "extracted": sum(1 for o in outs if o["extracted"]),
            "measured": len(mine),
            "extract_failure_rate": round(
                1.0 - sum(1 for o in outs if o["extracted"]) / max(len(outs), 1), 4),
            "vo_attempts_mean": round(float(np.mean(attempts)), 3) if attempts else None,
            "vo_retried_fraction": round(
                float(np.mean([a > 1 for a in attempts])), 4) if attempts else None,
            "preprocess_seconds_median": round(float(np.median(
                [o["preprocess_seconds"] for o in outs
                 if o.get("preprocess_seconds")] or [0.0])), 1),
        }
        for family in ("full", "head"):
            values = [r[family] for r in mine if family in r]
            if not values:
                continue
            row[family] = {key: round(float(np.median([v[key] for v in values])), 4)
                           for key in GATED}
            row[family + "_over_mocap_p99"] = {
                key: round(float(np.mean([v[key] > thresholds[key] for v in values])), 4)
                for key in GATED} if thresholds else {}
        by_length.append(row)

    report = {
        "work_dir": str(work_dir),
        "lengths": list(lengths),
        "common_window_s": common_window,
        "mocap_p99": thresholds,
        "by_length": by_length,
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1), encoding="utf-8")

    print("\nlength  n   fail   vo_att  retried | full: toe_spread root_speed jitter | head: toe_spread")
    for row in by_length:
        full, head = row.get("full", {}), row.get("head", {})
        print("{:>5}s {:>3} {:>6} {:>7} {:>8} | {:>15} {:>10} {:>6} | {:>16}".format(
            row["length_s"], row["measured"], row["extract_failure_rate"],
            row["vo_attempts_mean"] if row["vo_attempts_mean"] is not None else "-",
            row["vo_retried_fraction"] if row["vo_retried_fraction"] is not None else "-",
            full.get("lowest_toe_spread", "-"), full.get("root_speed_max", "-"),
            full.get("jitter_ratio", "-"), head.get("lowest_toe_spread", "-")))
    if thresholds:
        print("\nmocap p99: {}".format(json.dumps(thresholds)))
    print("\nwrote {}".format(output))
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("cut", "report"), required=True)
    parser.add_argument("--videos-dir", type=pathlib.Path,
                        default=pathlib.Path("data/wild_videos_20260811"))
    parser.add_argument("--work-dir", type=pathlib.Path,
                        default=pathlib.Path("runs/clip_length_scan"))
    parser.add_argument("--lengths", type=int, nargs="+",
                        default=[8, 16, 24, 32, 48, 64])
    parser.add_argument("--uploads", type=int, default=24)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 5])
    parser.add_argument("--converted-root", type=pathlib.Path, default=None,
                        help="default <work-dir>/converted")
    parser.add_argument("--raw-root", type=pathlib.Path, default=None,
                        help="default <work-dir>/gvhmr_raw")
    parser.add_argument("--common-window", type=int, default=8,
                        help="seconds of shared head footage the 'head' family measures")
    parser.add_argument("--audit", type=pathlib.Path,
                        default=pathlib.Path("runs/wild_3d_quality_audit.json"),
                        help="audit_wild_3d_quality.py output, for the mocap p99 bar")
    parser.add_argument("--output", type=pathlib.Path, default=None,
                        help="default <work-dir>/report.json")
    args = parser.parse_args(argv)

    work_dir = args.work_dir
    if args.stage == "cut":
        stage_cut(videos_dir=args.videos_dir, work_dir=work_dir,
                  lengths=args.lengths, uploads=args.uploads, gpus=args.gpus)
        return 0

    stage_report(work_dir=work_dir,
                 converted_root=args.converted_root or work_dir / "converted",
                 raw_root=args.raw_root or work_dir / "gvhmr_raw",
                 lengths=args.lengths, audit=args.audit,
                 common_window=args.common_window,
                 output=args.output or work_dir / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
