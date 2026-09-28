#!/usr/bin/env python3
"""One shard of GVHMR over the content-cut corpus, with the corpus in OSS.

The bash shard this replaces needs the whole 438 GB ingest tree present on the
NAS: it globs it for a worklist, symlinks every clip into a staging directory,
and writes its 3D output back beside it.  That is what filled the quota, and it
is a shape that cannot be fixed by deleting things afterwards -- the peak is the
corpus, no matter how promptly anything is cleaned up.

Here the corpus lives in OSS and one clip at a time is on local disk:

    fetch clip.mp4 + preprocess/bbx.pt   ->  scratch (/dev/shm by default)
    GVHMR, convert, validate             ->  scratch
    publish raw/ and converted/          ->  OSS
    delete scratch                       ->  footprint returns to zero

Peak disk is therefore O(concurrent shards), about 15 MB each, instead of
O(corpus).  Nothing this writes can fill a disk, which is the whole point.

Two behaviours are carried over deliberately, because they were each bought
with a failure:

* **The dancer is given, not re-chosen.**  ``preprocess/bbx.pt`` is seeded
  before GVHMR runs, so the 2D keypoints and the 3D describe the same person.
  On this corpus that is not automatic: 70% of clips have a rival dancer track,
  and on half the clips checked GVHMR's own choice was a different person.
* **A failure records why.**  The marker is an object holding the exit code and
  the last EXTRACT_FAIL line, not an empty file, so a later sweep can retry a
  clip that hit a defect and leave one whose visual odometry truly diverged.

Resumable, and the skip set is read once from OSS rather than per clip: a
listing of two prefixes costs two round trips, where 16,715 HEAD requests would
cost minutes before any GPU work started.

Usage::

    SHARD=0 NUM_SHARDS=14 GPU=0 python3 tools/run_gvhmr_ingest_shard_oss.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import List, Sequence

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools import asset_io, redo_manifest  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
CORPUS = "data/wild_ingest_v1"
RAW_ROOT = "data/wild3d/ingest_v1_gvhmr_raw"
CONVERTED_ROOT = "data/wild3d/ingest_v1_converted"
WORKLIST = "runs/wild_ingest_v1_worklist.json"
TODO_LIST = "runs/stage_b_todo_oss.json"

# Published nowhere, because an upload here cannot be undone.  The credentials
# this project holds are write-only against the bucket -- `ossutil rm` returns
# 403 AccessDenied -- so every object this stage writes is permanent, and the
# earlier version of this stage has already left 255 GB that cannot be removed:
#
#   0_input_video.mp4         221.55 GB, GVHMR's copy of the input.  Byte-size
#                             identical to data/wild_ingest_v1/<stem>/clip.mp4
#                             for all 11,485 clips checked, 0 orphans -- it is
#                             the corpus object under a second name.
#   preprocess/vit_features.pt 33.52 GB, GVHMR's internal ViT cache.  Nothing in
#                             this repo reads it; the consumers are
#                             hmr4d_results.pt, extract_meta.json and
#                             preprocess/slam_results.pt.
#
# Skipping them also cuts the per-clip upload by about 85%, which is most of the
# non-GPU time in this loop.  See CLAUDE.md "OSS 上传是不可逆的".
RAW_NOT_WORTH_KEEPING = ("0_input_video.mp4", "vit_features.pt")


def shard_of(clip: str, num_shards: int) -> int:
    """The hash split the bash shard used.  Kept for comparing against it.

    Not used to assign work any more, and the reason is arithmetic: the first
    run split 21 ways, this one splits 14, and gcd(21, 14) = 7, so the two
    partitions are *correlated* through mod 7.  The clips left undone by the
    21-way run therefore concentrate into particular 14-way shards rather than
    spreading -- measured at 220 to 460 per shard, a 2x spread that would leave
    half the GPUs idle for the last hour of a run.
    """
    return int(hashlib.sha256(clip.encode()).hexdigest(), 16) % num_shards


def freeze_todo(num_shards: int, retry_failed: bool = False,
                redo: Sequence[str] = ()) -> List[str]:
    """The remaining clips, in one deterministic order, published for the run.

    Every shard slices the same frozen list, so the split is exactly even and
    independent of whatever partition produced the existing output.  Freezing
    is what makes the slice safe: if each shard computed the remainder itself
    at its own start time, shards launched seconds apart would see different
    skip sets, slice differently, and between them do some clips twice and
    others never -- with nothing reporting the gap.

    ``retry_failed`` has to reach *here*, not only the shards.  Failure markers
    cannot be deleted -- the credentials are write-only against the bucket --
    so the whole point of a retry is to re-attempt clips that carry one.  A
    freeze that subtracts them regardless produces a todo list holding none of
    them, and the shards' own ``--retry-failed`` then has nothing left to
    re-attempt: every shard prints ``already_failed=0`` exactly as a correct
    retry does and the run exits 0 having done nothing.  Worse, the freeze
    overwrites the previous list, which was the documented recovery input, and
    this function is the only thing that produces one.

    ``redo`` is the same argument in a different key.  A clip whose video was
    re-cut still has last generation's ``quality.json``, so ``done`` contains it
    and every skip test above passes it over; naming it here is what puts it
    back in the list.  It is also *added* when the worklist has never heard of
    it -- a re-cut re-splits an upload, so clip names appear that did not exist
    when the worklist was frozen, and a list built only by subtraction can never
    contain them.  Both counts go into the frozen record, because "we re-did 40
    clips" and "we re-did 40 and skipped 3 we could not place" look the same
    afterwards otherwise.

    The redo set is written into the frozen list rather than passed to each
    shard.  A shard launched with a different ``--redo`` than the freeze used
    would slice a list whose contents it disagrees with, and would skip exactly
    the clips it was launched to redo -- while printing the same lines a correct
    run prints.
    """
    work = [row["clip"] for row in asset_io.read_json(WORKLIST)["clips"]]
    done = stems_with(CONVERTED_ROOT, "quality.json")
    failed = stems_with(RAW_ROOT, ".extract_failed")
    forced = set(redo)
    skip = (done if retry_failed else (done | failed)) - forced
    todo = [clip for clip in work if clip not in skip]
    _, unknown = redo_manifest.partition(sorted(forced), work)
    todo += unknown
    asset_io.write_json(TODO_LIST, {
        "worklist": WORKLIST, "num_shards": num_shards,
        "converted_at_freeze": len(done), "failed_at_freeze": len(failed),
        "failed_are_retried": bool(retry_failed),
        "redo_named": sorted(forced),
        "redo_absent_from_worklist": unknown,
        "clips": todo})
    return todo


def consecutive_unattributed(previous: int, record: Dict[str, object]) -> int:
    """The circuit breaker's counter.

    Reset by a success, and also by any failure the extractor could *explain* --
    a visual-odometry divergence is a fact about the clip, and a corpus can
    legitimately hand a worker a long run of them.  It advances only on
    failures nobody could attribute, which is what a broken environment
    produces: an import that dies before the diagnostic is written, a convert
    step that raises on a library it has always had.

    The distinction is not cosmetic.  The first version of this breaker counted
    every failure and tripped on shard 1 within ten clips, on eight consecutive
    genuine divergences, in a recovery run where half the reopened clips were
    expected to fail again.  A breaker calibrated on the healthy failure rate
    cannot survive a run whose population is enriched for failure; one that
    only counts unexplained failures does not have to be recalibrated at all.
    """
    if record.get("ok") or record.get("attributed"):
        return 0
    return previous + 1


def stems_with(prefix: str, filename: str) -> set:
    """Clip stems under ``prefix`` that already own ``filename``, in one scan.

    A flat prefix listing, not a directory walk: the walk costs one request per
    clip directory, and a listing that dies half way through would be read as
    "these clips are not done" and re-run thousands of finished extractions.
    """
    found = set()
    for name in asset_io.list_prefix(prefix):
        parts = name.split("/")
        if len(parts) == 2 and parts[1] == filename:
            found.add(parts[0])
    return found


def run(argv, cwd=None, env=None, timeout=None):
    started = time.time()
    completed = subprocess.run(argv, cwd=cwd, env=env, timeout=timeout,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return completed.returncode, completed.stdout.decode(errors="replace"), time.time() - started


def extract_one(stem: str, scratch: pathlib.Path, gpu: str, args) -> dict:
    """Fetch, extract, convert, validate, publish.  Returns a record."""
    raw_root = scratch / "raw"
    converted = scratch / "converted" / stem
    (raw_root / stem / "preprocess").mkdir(parents=True, exist_ok=True)

    video = scratch / (stem + ".mp4")          # GVHMR keys its output dir on the stem
    video.write_bytes(asset_io.read_bytes("{}/{}/clip.mp4".format(CORPUS, stem)))
    seed = "{}/{}/preprocess/bbx.pt".format(CORPUS, stem)
    if asset_io.exists(seed):
        (raw_root / stem / "preprocess" / "bbx.pt").write_bytes(asset_io.read_bytes(seed))

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = ":".join([
        str(REPO / "third_party" / "torch_scatter_compat"),
        str(REPO / "third_party" / "pytorch3d_compat"),
        "third-party/DPVO", "."]) if args.use_dpvo else ":".join([
        str(REPO / "third_party" / "pytorch3d_compat"), "."])

    argv = [sys.executable, str(REPO / "tools" / "run_gvhmr_extract.py"),
            "--video", str(video), "--output-root", str(raw_root),
            "--vo-attempts", str(args.vo_attempts)]
    if args.use_dpvo:
        argv.append("--use-dpvo")
    try:
        code, log, seconds = run(argv, cwd=str(REPO / "third_party" / "GVHMR"),
                                 env=env, timeout=args.timeout)
    except subprocess.TimeoutExpired:
        # A single clip must never be able to eat a worker: one multiprocessing
        # deadlock in the SLAM reader parked 15 of 21 workers for up to five and
        # a half hours while every liveness check reported a live process.
        return {"stem": stem, "ok": False, "attributed": True,
                "reason": "timeout after {}s".format(args.timeout), "exit": 124}

    # ``attributed`` = the failure was diagnosed *about this clip*: the
    # extractor named it, or it hung.  Anything else -- no EXTRACT_FAIL line, a
    # convert or validate step that died -- is a failure the worker cannot
    # attribute, and both of 2026-08-13's environment breakages arrived through
    # exactly those paths.  The circuit breaker counts only these, because a
    # corpus where half the reopened clips genuinely diverge (this recovery
    # run) would otherwise trip a breaker calibrated on the healthy rate.
    result = raw_root / stem / "hmr4d_results.pt"
    if not result.is_file():
        tail = [line for line in log.splitlines() if "EXTRACT_FAIL" in line]
        return {"stem": stem, "ok": False, "exit": code, "attributed": bool(tail),
                "reason": tail[-1][:400] if tail else "no EXTRACT_FAIL line; exit={}".format(code)}

    code, log, _ = run([sys.executable, str(REPO / "tools" / "convert_gvhmr_result.py"),
                        "--result", str(result),
                        "--extract-meta", str(raw_root / stem / "extract_meta.json"),
                        "--output-dir", str(converted)])
    if code != 0:
        return {"stem": stem, "ok": False, "exit": code,
                "reason": "convert failed: " + log.strip().splitlines()[-1][:300] if log.strip() else "convert failed"}

    code, log, _ = run([sys.executable, str(REPO / "tools" / "preprocess_wild_3d.py"),
                        "validate", "--output-dir", str(converted)])
    if code != 0:
        return {"stem": stem, "ok": False, "exit": code,
                "reason": "validate failed: " + (log.strip().splitlines()[-1][:300] if log.strip() else "")}

    raw_keys = asset_io.publish_dir(raw_root / stem, "{}/{}".format(RAW_ROOT, stem),
                                    skip=("*.pyc",) + RAW_NOT_WORTH_KEEPING)
    conv_keys = asset_io.publish_dir(converted, "{}/{}".format(CONVERTED_ROOT, stem))
    return {"stem": stem, "ok": True, "seconds": round(seconds, 1),
            "objects": len(raw_keys) + len(conv_keys)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, default=int(os.environ.get("SHARD", 0)))
    parser.add_argument("--num-shards", type=int, default=int(os.environ.get("NUM_SHARDS", 14)))
    parser.add_argument("--gpu", default=os.environ.get("GPU", "0"))
    parser.add_argument("--vo-attempts", type=int, default=int(os.environ.get("VO_ATTEMPTS", 3)))
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("EXTRACT_TIMEOUT", 1800)))
    parser.add_argument("--use-dpvo", action="store_true",
                        default=os.environ.get("USE_DPVO", "1") not in ("", "0"))
    parser.add_argument("--retry-failed", action="store_true",
                        help="also re-attempt clips carrying a failure marker")
    parser.add_argument("--max-consecutive-failures", type=int,
                        default=int(os.environ.get("MAX_CONSECUTIVE_FAILURES", 8)),
                        help="stop the shard after this many failures in a row "
                             "(0 disables). The markers cannot be deleted, so a "
                             "broken environment must not be allowed to write one "
                             "per clip at the rate a failure returns.")
    parser.add_argument("--limit", type=int, default=None, help="smoke-test a few clips")
    parser.add_argument("--freeze", action="store_true",
                        help="(re)compute the frozen todo list and exit")
    parser.add_argument("--redo", type=pathlib.Path,
                        default=(pathlib.Path(os.environ["REDO"])
                                 if os.environ.get("REDO") else None),
                        help="clips to extract again although their 3D exists: "
                             "a plain list or the JSON from "
                             "tools/audit_clip_freshness.py (stale.3d).  Takes "
                             "effect at --freeze; the shards read it back out "
                             "of the frozen list")
    args = parser.parse_args()

    redo = []
    if args.redo:
        try:
            redo = redo_manifest.load(args.redo, "3d")
        except redo_manifest.ManifestError as error:
            raise SystemExit(str(error))
        if not args.freeze:
            # Silently ignoring it would be the worst outcome: the operator
            # believes a re-extraction was requested and the shard skips every
            # clip it names, printing the lines a correct run prints.
            raise SystemExit(
                "--redo only takes effect at --freeze; the shards read the set "
                "back out of {}.  Re-run with --freeze first.".format(TODO_LIST))
        print("redo named for {} clip(s) from {}".format(len(redo), args.redo))

    if args.freeze:
        todo = freeze_todo(args.num_shards, retry_failed=args.retry_failed,
                           redo=redo)
        frozen = asset_io.read_json(TODO_LIST)
        print("froze {} clips ({} failure markers {}; {} redo, {} of them new "
              "names the worklist never had) -> {}".format(
                  len(todo), len(stems_with(RAW_ROOT, ".extract_failed")),
                  "kept for retry" if args.retry_failed else "excluded",
                  len(frozen["redo_named"]),
                  len(frozen["redo_absent_from_worklist"]), TODO_LIST))
        return 0

    frozen = asset_io.read_json(TODO_LIST)
    mine = frozen["clips"][args.shard::args.num_shards]
    # Read from the frozen list, not from this process's flags: the freeze is
    # what decided the slice, so it is also what decides which of its members
    # are exempt from the re-check below.
    forced = set(frozen.get("redo_named") or ())

    # Re-check against the store even though the list was frozen: a shard that
    # died and was relaunched must not redo its whole slice, and the listing
    # costs five seconds against hours of GPU time.  A clip named for redo is
    # exempt -- its output exists and is precisely what has to be replaced.
    done = stems_with(CONVERTED_ROOT, "quality.json")
    failed = set() if args.retry_failed else stems_with(RAW_ROOT, ".extract_failed")
    todo = [stem for stem in mine
            if stem in forced or (stem not in done and stem not in failed)]
    print("shard {}/{} gpu={} | slice={} already_done={} already_failed={} "
          "redo={} todo={}{}".format(
              args.shard, args.num_shards, args.gpu, len(mine),
              sum(1 for s in mine if s in done and s not in forced),
              sum(1 for s in mine if s in failed and s not in forced),
              sum(1 for s in mine if s in forced),
              len(todo), " (limited to {})".format(args.limit) if args.limit else ""),
          flush=True)
    if args.limit:
        todo = todo[: args.limit]

    extracted = failures = 0
    consecutive = 0
    for index, stem in enumerate(todo, start=1):
        with asset_io.scratch_dir(prefix="shard{}-".format(args.shard)) as scratch:
            try:
                record = extract_one(stem, scratch, args.gpu, args)
            except Exception as error:                     # noqa: BLE001
                record = {"stem": stem, "ok": False, "exit": -1,
                          "reason": "{}: {}".format(type(error).__name__, error)}
        consecutive = consecutive_unattributed(consecutive, record)
        if record["ok"]:
            extracted += 1
            print("OK {} [{}/{}] {}s".format(stem, index, len(todo), record["seconds"]), flush=True)
            continue
        failures += 1
        asset_io.write_bytes(
            "{}/{}/.extract_failed".format(RAW_ROOT, stem),
            ("exit={}\nreason={}\n".format(record.get("exit"), record["reason"])).encode())
        print("FAIL {} [{}/{}] {}".format(stem, index, len(todo), record["reason"]), flush=True)
        if args.max_consecutive_failures and consecutive >= args.max_consecutive_failures:
            # Stop rather than keep marking.  A broken environment fails every
            # clip, fails it in seconds because no GPU work happens, and the
            # markers it writes cannot be deleted -- so an unattended shard
            # converts an environment fault into permanent corpus state faster
            # than a working one converts video into motion.
            #
            # Twice on 2026-08-13: a pod rebuild dropped GVHMR's pip
            # dependencies (666 markers), and a numpy upgrade pulled in by an
            # unrelated install broke torch's numpy bridge (1,669 markers in
            # 24 minutes).  Both were environmental, both marked every clip
            # they touched, and neither could be undone.
            #
            # Only *unattributed* failures count, so the threshold does not have
            # to be re-tuned for a run whose reopened clips genuinely diverge at
            # 50%.  Eight in a row with nothing able to say why is an
            # environment, not a corpus.
            print("SHARD_ABORT shard={} after {} consecutive unattributed failures; the "
                  "environment is the likely cause, not the clips. Fix it, then "
                  "re-run with RETRY_FAILED=1 to reopen these markers.".format(
                      args.shard, consecutive), flush=True)
            print("SHARD_DONE shard={} todo={} extracted={} failed={} aborted=1".format(
                args.shard, len(todo), extracted, failures), flush=True)
            return 3

    print("SHARD_DONE shard={} todo={} extracted={} failed={} aborted=0".format(
        args.shard, len(todo), extracted, failures), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
