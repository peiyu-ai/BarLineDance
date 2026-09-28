#!/usr/bin/env python3
"""Recover DPVO-diverged clips as static-camera, but only where the footage says so.

WHY THIS EXISTS SEPARATELY.  ``run_gvhmr_batch_shard.sh`` already implements
this as its pass 2, but it drives the old flat video root.  The ingest-tree
runners -- ``run_gvhmr_ingest_shard.sh`` and ``run_gvhmr_ingest_shard_oss.py``
-- have no such pass, so on the content-cut corpus a diverged clip is simply
lost.  This is that pass for the ingest tree.  It is a new file rather than an
edit because the shard script is usually still running when this is needed, and
editing a live bash script misparses at the changed offset (a lesson this repo
has already paid for, 2026-08-09).

WHAT THE GATE ACTUALLY ASSERTS.  ``--static-cam`` asserts ``R_w2c = I`` for
every frame.  The error that assertion introduces is *exactly* the camera's
true rotation away from frame 0, which ``tools/measure_camera_rotation.py``
measures from a background-only homography with the dancer masked out.  So this
is not a fudge applied to whatever failed: it is applied where the measurement
says the model is right, and REFUSED with its number where it is not.

THE GATE CAN FAIL, AND MUST BE ALLOWED TO.  Measured on this repo's wild corpus
in 2026-08-09, only 6 of 40 diverged clips passed at 3 degrees (median rotation
12.12 deg) -- there static-cam was a narrow recovery and the tool mostly said
no.  Measured on the 汤汤汤小圆 account on 2026-09-02 it is the opposite: the
17 clips DPVO diverged on have median rotation 0.572 deg against 2.303 deg for
the 22 it solved (Mann-Whitney p=0.0046, permutation p=0.0025), and 15 of 17
pass at 3 degrees.  Those two numbers coming out opposite on two corpora is the
evidence that the gate is reading the footage rather than rubber-stamping.  A
run where it accepts everything is a run to distrust.

Every accepted clip carries its measurement into ``extract_meta.json`` as
``static_cam_justification`` (run_gvhmr_extract.py:295), so the evidence
travels with the artifact rather than living in a log.

Usage::

    python3 tools/run_static_cam_recovery.py \\
        --ingest-root /cache/atomicdance-assets/data/wild_ingest_txy_t0 \\
        --raw-root    /cache/atomicdance-assets/data/wild3d/txy_t0_gvhmr_raw \\
        --converted-root /cache/atomicdance-assets/data/wild3d/txy_t0_converted \\
        --report runs/t0_static_cam_recovery.json --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
from typing import Dict, List, Optional, Sequence

REPO = pathlib.Path(__file__).resolve().parents[1]

ACCEPT = "accepted_static"
REFUSE = "refused_rotating"
UNMEASURED = "refused_unmeasured"


class RecoveryError(RuntimeError):
    pass


def decide(measurement: Optional[Dict[str, object]], threshold_deg: float) -> Dict[str, object]:
    """Accept, refuse, or decline to judge -- from the measurement alone.

    Split out from the subprocess plumbing so the gate is testable without a
    GPU, a video, or GVHMR.  Three outcomes, not two: a clip whose background
    could not be fit is NOT the same as a clip measured to be still, and
    collapsing them would declare footage static on faith, which is the one
    thing measure_camera_rotation.py's docstring says not to do.
    """
    if not measurement:
        return {"decision": UNMEASURED, "why": "no measurement produced"}
    if not measurement.get("static_camera") and measurement.get("verdict_reason") in (
            "too few measurable pairs", "insufficient background"):
        return {"decision": UNMEASURED, "why": str(measurement.get("verdict_reason"))}
    rotation = measurement.get("max_rotation_from_first_frame_deg")
    if rotation is None:
        return {"decision": UNMEASURED, "why": "measurement carries no rotation"}
    if float(rotation) <= float(threshold_deg):
        return {"decision": ACCEPT,
                "why": "max rotation {:.3f} deg <= {:.3f}".format(float(rotation), threshold_deg),
                "max_rotation_deg": float(rotation)}
    return {"decision": REFUSE,
            "why": "max rotation {:.3f} deg > {:.3f}".format(float(rotation), threshold_deg),
            "max_rotation_deg": float(rotation)}


def outstanding(ingest_root: pathlib.Path, raw_root: pathlib.Path,
                clips: Optional[Sequence[str]]) -> List[str]:
    """Clips with a clip.mp4 and no 3D result. A failure marker is not required:
    a clip whose marker was cleaned still has no result, and skipping it would
    make this pass silently incomplete."""
    if clips is not None:
        names = [c for c in clips]
    else:
        names = sorted(p.name for p in ingest_root.iterdir()
                       if p.is_dir() or p.is_symlink())
    return [n for n in names
            if (ingest_root / n / "clip.mp4").is_file()
            and not (raw_root / n / "hmr4d_results.pt").is_file()]


def measure(clip: str, ingest_root: pathlib.Path, raw_root: pathlib.Path,
            threshold_deg: float, timeout: int) -> Optional[Dict[str, object]]:
    video = ingest_root / clip / "clip.mp4"
    boxes = ingest_root / clip / "preprocess" / "bbx.pt"
    command = [sys.executable, str(REPO / "tools/measure_camera_rotation.py"),
               "--video", str(video), "--threshold-deg", str(threshold_deg)]
    if boxes.is_file():
        command += ["--person-boxes", str(boxes)]
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        return json.loads(done.stdout)
    except Exception:                                   # noqa: BLE001
        return None


def extract_static(clip: str, ingest_root: pathlib.Path, raw_root: pathlib.Path,
                   evidence: pathlib.Path, gpu: str, timeout: int) -> bool:
    video = ingest_root / clip / "clip.mp4"
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # No DPVO on this path: run_gvhmr_extract.py refuses --static-cam together
    # with --use-dpvo, and the point here is that DPVO has already failed.
    environment["PYTHONPATH"] = os.pathsep.join([
        str(REPO / "third_party/pytorch3d_compat"),
        str(REPO / "third_party/GVHMR"),
    ])
    command = [sys.executable, str(REPO / "tools/run_gvhmr_extract.py"),
               "--video", str(video), "--output-root", str(raw_root),
               "--static-cam", "--static-cam-evidence", str(evidence)]
    done = subprocess.run(command, cwd=str(REPO / "third_party/GVHMR"),
                          env=environment, capture_output=True, text=True,
                          timeout=timeout)
    if done.returncode != 0:
        tail = (done.stdout or "")[-400:] + (done.stderr or "")[-400:]
        print("   extract failed rc={}: {}".format(done.returncode, tail.replace("\n", " ")[-300:]),
              flush=True)
    return (raw_root / clip / "hmr4d_results.pt").is_file()


def convert(clip: str, raw_root: pathlib.Path, converted_root: pathlib.Path,
            timeout: int) -> bool:
    ok = subprocess.run(
        [sys.executable, str(REPO / "tools/convert_gvhmr_result.py"),
         "--result", str(raw_root / clip / "hmr4d_results.pt"),
         "--extract-meta", str(raw_root / clip / "extract_meta.json"),
         "--output-dir", str(converted_root / clip)],
        capture_output=True, text=True, timeout=timeout).returncode == 0
    if not ok:
        return False
    return subprocess.run(
        [sys.executable, str(REPO / "tools/preprocess_wild_3d.py"), "validate",
         "--output-dir", str(converted_root / clip)],
        capture_output=True, text=True, timeout=timeout).returncode == 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ingest-root", type=pathlib.Path, required=True)
    parser.add_argument("--raw-root", type=pathlib.Path, required=True)
    parser.add_argument("--converted-root", type=pathlib.Path, required=True)
    parser.add_argument("--clips", type=pathlib.Path, default=None,
                        help="optional list of clip stems; default is every "
                             "clip in --ingest-root without a 3D result")
    parser.add_argument("--threshold-deg", type=float, default=3.0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="measure and decide, but run no GPU work; the "
                             "honest way to see what the gate would do")
    args = parser.parse_args(argv)

    clips = None
    if args.clips:
        clips = [l.strip() for l in args.clips.read_text(encoding="utf-8").splitlines()
                 if l.strip()]
    todo = outstanding(args.ingest_root, args.raw_root, clips)
    print("{} clip(s) without a 3D result".format(len(todo)), flush=True)

    rows = []
    for index, clip in enumerate(todo, 1):
        measurement = measure(clip, args.ingest_root, args.raw_root,
                              args.threshold_deg, args.timeout)
        verdict = decide(measurement, args.threshold_deg)
        row = {"clip": clip, "measurement": measurement, **verdict}
        print("[{}/{}] {} {}".format(index, len(todo), verdict["decision"], clip), flush=True)
        if verdict["decision"] == ACCEPT and not args.dry_run:
            (args.raw_root / clip).mkdir(parents=True, exist_ok=True)
            evidence = args.raw_root / clip / "camera_rotation.json"
            evidence.write_text(json.dumps(measurement, indent=1), encoding="utf-8")
            if extract_static(clip, args.ingest_root, args.raw_root, evidence,
                              args.gpu, args.timeout):
                row["extracted"] = True
                row["converted"] = convert(clip, args.raw_root, args.converted_root,
                                           args.timeout)
                if row["converted"]:
                    marker = args.raw_root / clip / ".extract_failed"
                    if marker.exists():
                        marker.unlink()
            else:
                row["extracted"] = False
                row["converted"] = False
        rows.append(row)

    counts: Dict[str, int] = {}
    for row in rows:
        counts[row["decision"]] = counts.get(row["decision"], 0) + 1
    recovered = sum(1 for r in rows if r.get("converted"))
    report = {
        "schema_version": "atomicdance-static-cam-recovery-v1",
        "config": {"threshold_deg": args.threshold_deg, "dry_run": bool(args.dry_run),
                   "ingest_root": str(args.ingest_root), "raw_root": str(args.raw_root)},
        "counts": counts,
        "recovered": recovered,
        "rows": rows,
        "how_to_read": (
            "accepted_static means the measured max rotation from frame 0 is "
            "within threshold, so asserting R_w2c = I introduces at most that "
            "much error; the measurement is written into the extraction's "
            "extract_meta.json as static_cam_justification. refused_rotating "
            "is a clip the gate declined WITH ITS NUMBER -- those are honest "
            "failures, not losses to be recovered another way. "
            "refused_unmeasured is neither: the background could not be fit, "
            "so nothing is asserted. A run in which nothing is refused is a "
            "run to distrust: on this repo's wider wild corpus the same gate "
            "accepted only 6 of 40."),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print()
    print("decisions: {}".format(counts))
    print("recovered (extracted+converted): {}".format(recovered))
    print("wrote {}".format(args.report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
