#!/usr/bin/env python3
"""Publish the fps re-cut clip bytes, and prove the store now holds them.

The 2026-08-19 re-cut fixed two defects in clips from non-integer-frame-rate
uploads: ``setpts`` was handed ``N/2997/100/TB`` where it meant ``N/(2997/100)``
(the video ran at the wrong speed), and the audio segment was taken at
``start/30`` instead of ``start/source_fps`` (the clip held the wrong part of
the song -- correlation with the segment it should hold, measured over 60
clips, ``-0.0104`` against ``0.9852`` for the corrected cut).  The corrected
bytes were written into the local cache and **never pushed**, so
``data/wild_ingest_v1`` currently means two different things depending on who
asks:

* ``asset_io.read_bytes`` resolves local-first, so a stage running on this pod
  reads the corrected clip;
* anything reading the object store -- another pod, a later run after the cache
  is evicted, ``audit_clip_freshness``'s ranged fallback -- reads the released,
  broken one.

That is not a corpus in two states; it is a corpus whose state depends on the
reader, which is strictly worse because no gate can name it.  This tool closes
it in the only direction available: these credentials can PUT over an existing
key and cannot delete one, so the fix is an in-place overwrite, and the names
that survive are exactly the names that were already there.

**Why not ``oss_assets.py push``.**  That path is ``ossutil sync --update``,
which decides whether to upload by comparing *mtime*.  CLAUDE.md §2.3 records
what mtime is worth here -- a cache pull resets it, and one review already read
the opposite conclusion off it.  A re-cut whose local copy happened to look
older than the object would be skipped, and the skip prints like a success.
This tool overwrites unconditionally (``publish_dir`` is ``sync --force``) and
then goes and looks.

**The check, and its two controls.**  Verification re-reads the first megabyte
of the published object with a ranged GET and compares its sha256 with the same
window of the local file -- the same window every stage already hashes into
``video_sha256_1mb`` and the same instrument ``audit_clip_freshness`` uses, so
this compares against a number produced the same way rather than a
re-derivation of it.  Both directions are exercised by construction:

* *before* the push the two hashes must **differ** (these clips were re-cut);
  a stem that already matches was already pushed and is reported as
  ``already_current``, not as a failure and not as work;
* *after* the push they must **match**.  A run in which no file ever reads
  "same" would mean the comparison cannot detect equality at all, which is the
  failure mode a one-directional check hides.

``clip.mp4``, ``audio.wav`` and ``meta.json`` are the three verified: the first
carries the speed defect, the second carries the offset defect and is what
stage D turns into the beat grid the current M1 cuts on, and the third carries
``source_fps``, whose presence is what tells a later reader which cut this is.
The rest of the directory is published in the same sync and covered by the
sync's own exit status.

The clip list is a manifest, never a directory walk.  ``refix_wild_fps_clips_
census.py`` names 775 orphans -- clips the re-cut stopped producing, whose
objects cannot be deleted and which every enumeration therefore keeps handing
back.  Pushing bytes for one would be spending an irreversible write on a name
no consumer may read.

Usage::

    python3 tools/push_recut_clips.py --redo runs/clean5/freshness.json \\
        --stage 3d --orphans /cache/.../refix/orphans.txt \\
        --output runs/clean5/recut_push.json

    python3 tools/push_recut_clips.py --redo ... --limit 1     # pilot one clip
    python3 tools/push_recut_clips.py --redo ... --verify-only # re-read, no PUT
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools import asset_io, redo_manifest  # noqa: E402

INGEST = "data/wild_ingest_v1"
HASH_LIMIT = 1 << 20
OSSUTIL = "/opt/data-infra/ossutil64"
REPO = pathlib.Path(__file__).resolve().parents[1]
# The three files the two fps defects live in, plus the one that records which
# cut this is.  Verified individually; the rest of the clip directory rides the
# same sync and is covered by its exit status.
VERIFIED = ("clip.mp4", "audio.wav", "meta.json")


def local_head(path: pathlib.Path) -> Optional[str]:
    if not path.is_file():
        return None
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read(HASH_LIMIT)).hexdigest()


def remote_head(relative: str) -> Optional[str]:
    """First megabyte of the published object, or None if it is not there.

    Ranged ``ossutil cp`` rather than the SDK: this bridge bucket answers the
    SDK's LIST and ranged GET with a bare 502, and ``audit_clip_freshness``
    already settled on this call for the same reason.
    """
    url = "oss://" + asset_io.remote_path(relative)
    with tempfile.TemporaryDirectory() as workspace:
        target = pathlib.Path(workspace) / "head.bin"
        argv = [OSSUTIL, "cp", url, str(target),
                "--range=0-{}".format(HASH_LIMIT - 1), "-f"]
        config = REPO / "ossutilconfig"
        if config.is_file():
            argv += ["-c", str(config)]
        completed = subprocess.run(argv, capture_output=True, text=True)
        if completed.returncode != 0 or not target.is_file():
            return None
        return hashlib.sha256(target.read_bytes()).hexdigest()


def select_stems(manifest, stage: str, orphans: Optional[pathlib.Path]
                 ) -> Dict[str, object]:
    """Manifest stems minus orphan names, with the arithmetic reported.

    Separated from the transfer so the selection can be tested without a
    store: what this returns is what an irreversible write will be spent on.
    """
    named = redo_manifest.load(manifest, stage)
    orphan_set = set()
    if orphans is not None:
        orphan_set = {line.strip() for line in
                      pathlib.Path(orphans).read_text(encoding="utf-8").splitlines()
                      if line.strip()}
    dropped = sorted(set(named) & orphan_set)
    kept = [stem for stem in named if stem not in orphan_set]
    return {"named": len(named), "orphans_dropped": dropped, "stems": kept}


def clip_dir(stem: str) -> pathlib.Path:
    root = pathlib.Path(INGEST)
    return (root if root.is_absolute() else REPO / root) / stem


def inspect(stem: str) -> Dict[str, object]:
    """Local vs published hashes for one clip's verified files."""
    directory = clip_dir(stem)
    row: Dict[str, object] = {"clip": stem, "files": {}}
    for name in VERIFIED:
        relative = "{}/{}/{}".format(INGEST, stem, name)
        row["files"][name] = {"local": local_head(directory / name),
                              "remote": remote_head(relative)}
    return row


def verdict(row: Dict[str, object]) -> str:
    """One of: no_local / missing_remote / differs / already_current."""
    files = row["files"]
    if any(entry["local"] is None for entry in files.values()):
        return "no_local"
    if any(entry["remote"] is None for entry in files.values()):
        return "missing_remote"
    if all(entry["local"] == entry["remote"] for entry in files.values()):
        return "already_current"
    return "differs"


def push_one(stem: str) -> List[str]:
    return asset_io.publish_dir(clip_dir(stem), "{}/{}".format(INGEST, stem))


def run_pool(function, items: Sequence[str], workers: int):
    if workers <= 1:
        return [function(item) for item in items]
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(function, items))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--redo", required=True,
                        help="a freshness audit, a census, or a plain stem list")
    parser.add_argument("--stage", default="3d", choices=redo_manifest.STAGES)
    parser.add_argument("--orphans", default=None,
                        help="stems no consumer may read; excluded by name")
    parser.add_argument("--output", default=None, help="where to write the report")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None,
                        help="pilot the first N stems")
    parser.add_argument("--verify-only", action="store_true",
                        help="read both sides and report; upload nothing")
    args = parser.parse_args()

    try:
        selection = select_stems(args.redo, args.stage, args.orphans)
    except redo_manifest.ManifestError as error:
        raise SystemExit(str(error))
    stems: List[str] = list(selection["stems"])
    if selection["orphans_dropped"]:
        print("orphans excluded by name: {} (e.g. {})".format(
            len(selection["orphans_dropped"]), selection["orphans_dropped"][:3]))
    if args.limit:
        stems = stems[: args.limit]
    print("{} stem(s) named for {}, {} to push".format(
        selection["named"], args.stage, len(stems)), flush=True)

    before = run_pool(inspect, stems, args.workers)
    states: Dict[str, List[str]] = {}
    for row in before:
        states.setdefault(verdict(row), []).append(row["clip"])
    for state in sorted(states):
        print("  before: {:16s} {}".format(state, len(states[state])), flush=True)

    blocked = states.get("no_local", [])
    if blocked:
        print("REFUSING: {} stem(s) have no local bytes to publish, e.g. {}".format(
            len(blocked), blocked[:3]), flush=True)
        return 1

    todo = states.get("differs", []) + states.get("missing_remote", [])
    if args.verify_only:
        print("verify-only: {} stem(s) would be pushed".format(len(todo)))
        return 0

    published = 0
    for index, names in enumerate(run_pool(push_one, todo, args.workers), 1):
        published += len(names)
    print("pushed {} object(s) across {} clip(s)".format(published, len(todo)),
          flush=True)

    after = run_pool(inspect, todo, args.workers)
    still: List[str] = []
    matched = 0
    for row in after:
        state = verdict(row)
        if state == "already_current":
            matched += 1
        else:
            still.append(row["clip"])
    print("after: {} clip(s) now match the store, {} still disagree".format(
        matched, len(still)), flush=True)

    report = {
        "generated_by": "tools/push_recut_clips.py",
        "manifest": str(args.redo),
        "stage": args.stage,
        "named": selection["named"],
        "orphans_dropped": selection["orphans_dropped"],
        "attempted": len(todo),
        "before": {state: len(clips) for state, clips in states.items()},
        "verified_files": list(VERIFIED),
        "matched_after_push": matched,
        "still_disagree": still,
        "reading": "before/differs is the negative control (these clips were "
                   "re-cut, so the store must not already agree); "
                   "matched_after_push is the positive one (the same "
                   "comparison reading 'same' is what shows it can)",
    }
    if args.output:
        target = pathlib.Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                          encoding="utf-8")
        print("wrote {}".format(target), flush=True)

    if still:
        print("PUSH_FAIL {} clip(s) still differ after the overwrite: {}".format(
            len(still), still[:5]), flush=True)
        return 1
    if matched == 0 and todo:
        # Everything was uploaded and nothing reads "same": the comparison
        # cannot detect equality, so its "differs" verdicts mean nothing either.
        print("PUSH_FAIL nothing reads as current after the push -- the "
              "comparison never fires in the matching direction", flush=True)
        return 1
    print("PUSH_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
