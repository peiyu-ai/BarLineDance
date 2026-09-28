#!/usr/bin/env python3
"""Convert the GVHMR results that were computed but never turned into 151-D.

Of the 3,373 clips carrying ``hmr_status: pending``, 3,111 already hold a
complete ``hmr4d_results.pt``.  GVHMR ran on them and succeeded; only the
conversion step never followed, so they sit outside the corpus for want of a
CPU pass rather than for want of a GPU.  221 carry an ``.extract_failed``
marker and 41 stopped after preprocessing -- those are genuine failures and
belong to the GPU re-run, not here.

The classification comes from reading what each clip directory actually
contains, not from the manifest's status field: the manifest says "pending",
which is one word for three different situations, and treating them alike
would either waste GPU time re-running work that succeeded or silently skip
clips that need it.

Conversion itself shells out to ``tools/convert_gvhmr_result.py`` rather than
importing its arithmetic.  That script publishes a clip atomically -- staging
directory, metadata last, rename into place -- because an interruption between
the motion array and ``metadata.json`` leaves a directory that looks converted,
fails validation, and is skipped forever.  One clip in the first 2,667 landed
in exactly that state.  Re-implementing the conversion here to save an
interpreter start would mean re-implementing that guarantee too.

Usage::

    python3 tools/backfill_gvhmr_conversion.py --states runs/gvhmr_raw_states.json \
        --manifest data/wild3d/source_manifests/tiktok_new_b2gpu/sequences_hmr_v3.jsonl \
        --output runs/gvhmr_conversion_backfill.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
import subprocess
import sys
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "data" / "wild3d" / "gvhmr_raw"
CONVERTED_ROOT = REPO_ROOT / "data" / "wild3d" / "converted"
# The converter reads three things, not two.  Omitting the SLAM trajectory does
# not fail the conversion and does not change one byte of the motion -- it is
# only ever written into ``camera.npz`` -- so the directory looks finished while
# carrying no camera track.  ``reconcile-wild-hmr`` is where that surfaces, as
# 3,069 clips quarantined for "camera.npz missing dpvo_traj_*", which is exactly
# the "looks converted" failure this tool's docstring warns about, one file over.
NEEDED = ("hmr4d_results.pt", "extract_meta.json", "preprocess/slam_results.pt")


def pending_hmr_done(states_path: pathlib.Path, manifest_path: pathlib.Path) -> List[str]:
    states = json.loads(states_path.read_text(encoding="utf-8"))
    status = {}
    for line in manifest_path.open(encoding="utf-8"):
        row = json.loads(line)
        status[row["legacy_clip_id"]] = (row.get("qc") or {}).get("hmr_status")
    return sorted(clip for clip, state in states.items()
                  if state == "hmr_done" and status.get(clip) == "pending")


def fetch_one(clip: str, bucket, key_prefix: str) -> Optional[str]:
    """Bring down only the two files the conversion reads.

    A clip directory is 7.3 MB and the conversion needs 1.8 MB of it; pulling
    whole directories would move 23 GB to use 5.6 GB of it.
    """
    destination = RAW_ROOT / clip
    destination.mkdir(parents=True, exist_ok=True)
    for name in NEEDED:
        local = destination / name
        # NEEDED now carries a nested key, so the clip directory alone is not
        # enough to write into.
        local.parent.mkdir(parents=True, exist_ok=True)
        if local.is_file() and local.stat().st_size > 0:
            continue
        try:
            bucket.get_object_to_file("{}{}/{}".format(key_prefix, clip, name), str(local))
        except Exception as error:                    # noqa: BLE001 - recorded per clip
            return "fetch {}: {!r}".format(name, error)
    return None


def _conversion_is_complete(output: pathlib.Path) -> bool:
    """Is this directory a finished conversion, or one that merely looks like it?

    ``metadata.json`` alone was the old test, and it is what let a whole batch
    land without a camera track: the converter writes metadata whether or not it
    found a SLAM trajectory.  Asking ``camera.npz`` whether it actually carries
    one makes the skip self-healing -- a directory converted by the earlier,
    two-file version is re-done instead of being skipped forever.
    """
    if not (output / "metadata.json").is_file():
        return False
    camera = output / "camera.npz"
    if not camera.is_file():
        return False
    import numpy as np

    with np.load(camera) as arrays:
        return any(name.startswith(("dpvo_", "simplevo_")) for name in arrays.files)


def convert_one(clip: str) -> Optional[str]:
    raw = RAW_ROOT / clip
    output = CONVERTED_ROOT / clip
    if _conversion_is_complete(output):
        return None
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "convert_gvhmr_result.py"),
         "--result", str(raw / "hmr4d_results.pt"),
         "--extract-meta", str(raw / "extract_meta.json"),
         "--output-dir", str(output)],
        capture_output=True, text=True, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        return (result.stderr.strip().splitlines() or ["exit {}".format(result.returncode)])[-1]
    return None


def build(*, states: pathlib.Path, manifest: pathlib.Path, output: pathlib.Path,
          workers: int, fetch_workers: int, limit: Optional[int]) -> Dict[str, object]:
    from tools.oss_assets import DEFAULT_PREFIX, make_bucket, parse_prefix

    clips = pending_hmr_done(states, manifest)
    if limit is not None:
        clips = clips[:limit]
    print("{} clips to convert".format(len(clips)), flush=True)

    _, root_key = parse_prefix(DEFAULT_PREFIX)
    key_prefix = "{}AtomicDance/data/wild3d/gvhmr_raw/".format(root_key)
    bucket = make_bucket(DEFAULT_PREFIX)

    fetch_errors: Dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=fetch_workers) as pool:
        futures = {pool.submit(fetch_one, clip, bucket, key_prefix): clip for clip in clips}
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            error = future.result()
            if error:
                fetch_errors[futures[future]] = error
            if index % 250 == 0 or index == len(clips):
                print("[fetch {}/{}] {} errors".format(index, len(clips), len(fetch_errors)),
                      flush=True)

    ready = [clip for clip in clips if clip not in fetch_errors]
    convert_errors: Dict[str, str] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for index, (clip, error) in enumerate(
                zip(ready, pool.map(convert_one, ready, chunksize=4)), start=1):
            if error:
                convert_errors[clip] = error
            if index % 250 == 0 or index == len(ready):
                print("[convert {}/{}] {} errors".format(index, len(ready), len(convert_errors)),
                      flush=True)

    converted = sum(1 for clip in ready
                    if (CONVERTED_ROOT / clip / "metadata.json").is_file())
    report = {
        "selected": len(clips),
        "fetch_failed": len(fetch_errors),
        "convert_failed": len(convert_errors),
        "converted": converted,
        "converted_root": str(CONVERTED_ROOT.resolve()),
        "raw_root": str(RAW_ROOT.resolve()),
        "errors": dict(list(fetch_errors.items())[:20] + list(convert_errors.items())[:20]),
        "note": "the manifest still says pending; rebuild it with preprocess_wild_3d "
                "reconcile so the new clips become candidates",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--states", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--fetch-workers", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    report = build(states=args.states, manifest=args.manifest, output=args.output,
                   workers=args.workers, fetch_workers=args.fetch_workers,
                   limit=args.limit)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["convert_failed"] == 0 and report["fetch_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
