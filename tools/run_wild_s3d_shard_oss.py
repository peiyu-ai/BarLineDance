#!/usr/bin/env python3
"""One shard of stage F -- S3D features -- with the videos in OSS.

The bash path this replaces globs a staging directory of symlinks that point
into ``data/wild_ingest_v1``.  Once that corpus is parked in OSS the symlinks
dangle, and stage F would report success over whatever subset still resolved:
the extractor skips a video it cannot open, and a shard that finds nothing
prints ``done=0`` and exits zero.  That is the failure this repo keeps meeting
-- absence read as completion -- so the fetch is explicit here.

Batched rather than one clip at a time.  Loading S3D costs seconds and
extracting one clip's features costs about one, so a per-clip process would
spend most of its life loading the model; a batch of 50 amortises it while
keeping the disk footprint at roughly 550 MB per shard.

Usage::

    SHARD=0 NUM_SHARDS=14 GPU=0 python3 tools/run_wild_s3d_shard_oss.py
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import time
from typing import List, Optional, Sequence

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools import asset_io, redo_manifest  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
CORPUS = "data/wild_ingest_v1"
CONVERTED_ROOT = "data/wild3d/ingest_v1_converted"
FEATURES_ROOT = "data/wild_visual_s3d"


def load_redo(path: pathlib.Path) -> List[str]:
    """Clip stems to re-extract whatever is already published for them.

    Delegated to ``tools/redo_manifest`` so stage B and stage F cannot disagree
    about what a redo list is.  Reading the audit directly is the point: the
    list of clips whose features were built from bytes the clip no longer has
    is a measurement, and retyping it by hand is where it stops being one.
    """
    try:
        return redo_manifest.load(path, "s3d")
    except redo_manifest.ManifestError as error:
        raise SystemExit(str(error))


def eligible(num_shards: int, shard: int,
             redo: Optional[Sequence[str]] = None) -> List[str]:
    """Clips that have 3D and need features, this shard's slice.

    Keyed off the converted set rather than the ingest worklist: a clip whose
    extraction failed has no motion for a feature to describe, and including it
    would spend GPU time producing a row nothing downstream can join to.

    "Needs features" used to mean "has no ``.npz`` in the store", and that is a
    presence test on a corpus where the failure mode is *contents*.  On
    2026-08-19 the published features included 3,991 clips whose objects
    existed, were the right length, and had been extracted from an older cut of
    the video; this function would have reported every one of them done.  It
    still cannot tell on its own -- deciding needs the hash the extractor
    recorded weighed against the clip's current bytes, which is a corpus-wide
    pass, not a listing.  So the judgment is made by
    ``tools/audit_clip_freshness.py`` and arrives here as ``redo``: those stems
    are extracted again even though their features are present.

    A stem in ``redo`` with no 3D is an error rather than a silent drop.  The
    caller believes it named work; returning quietly fewer clips than it asked
    for is how a re-extraction ends up covering part of the corpus and
    reporting success.
    """
    import tools.run_gvhmr_ingest_shard_oss as stage_b

    have_3d = sorted(stage_b.stems_with(CONVERTED_ROOT, "quality.json"))
    have_features = {name[: -len(".npz")]
                     for name in asset_io.list_prefix(FEATURES_ROOT)
                     if name.endswith(".npz") and "/" not in name}
    forced = set(redo or ())
    if forced:
        unknown = sorted(forced - set(have_3d))
        if unknown:
            raise SystemExit(
                "{} stem(s) named for re-extraction have no 3D under {}; "
                "first few: {}".format(len(unknown), CONVERTED_ROOT,
                                       ", ".join(unknown[:5])))
    todo = [stem for stem in have_3d
            if stem not in have_features or stem in forced]
    return todo[shard::num_shards]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, default=int(os.environ.get("SHARD", 0)))
    parser.add_argument("--num-shards", type=int, default=int(os.environ.get("NUM_SHARDS", 14)))
    parser.add_argument("--gpu", default=os.environ.get("GPU", "0"))
    parser.add_argument("--batch", type=int, default=int(os.environ.get("BATCH", 50)))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--redo", type=pathlib.Path,
                        default=(pathlib.Path(os.environ["REDO"])
                                 if os.environ.get("REDO") else None),
                        help="stems to re-extract even though features exist; "
                             "a plain list or the JSON from "
                             "tools/audit_clip_freshness.py")
    args = parser.parse_args()

    redo = load_redo(args.redo) if args.redo else []
    if redo:
        print("re-extraction named for {} stem(s) from {}".format(
            len(redo), args.redo), flush=True)
    todo = eligible(args.num_shards, args.shard, redo)
    if args.limit:
        todo = todo[: args.limit]
    print("shard {}/{} gpu={} todo={}".format(
        args.shard, args.num_shards, args.gpu, len(todo)), flush=True)

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = args.gpu

    produced = failed = 0
    for start in range(0, len(todo), args.batch):
        chunk = todo[start: start + args.batch]
        with asset_io.scratch_dir(prefix="s3d{}-".format(args.shard)) as scratch:
            videos, features = scratch / "videos", scratch / "features"
            videos.mkdir()
            features.mkdir()
            fetched = []
            for stem in chunk:
                try:
                    (videos / (stem + ".mp4")).write_bytes(
                        asset_io.read_bytes("{}/{}/clip.mp4".format(CORPUS, stem)))
                    fetched.append(stem)
                except Exception as error:                    # noqa: BLE001
                    failed += 1
                    print("FETCH_FAIL {} {}".format(stem, error), flush=True)
            if not fetched:
                continue

            started = time.time()
            completed = subprocess.run(
                [sys.executable, str(REPO / "tools" / "extract_visual_features.py"),
                 "--video-dir", str(videos), "--output-dir", str(features),
                 "--video-per-motion", "1"],
                cwd=str(REPO), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            log = completed.stdout.decode(errors="replace")
            if completed.returncode != 0:
                failed += len(fetched)
                print("BATCH_FAIL [{}..] rc={} {}".format(
                    start, completed.returncode, log.strip()[-300:]), flush=True)
                continue

            # Count the artifacts, not the exit code.  The extractor skips a
            # video it cannot decode and still exits zero, so "it returned 0"
            # and "it produced features" are different claims.
            made = sorted(features.glob("*.npz"))
            asset_io.publish_dir(features, FEATURES_ROOT)
            produced += len(made)
            missing = len(fetched) - len(made)
            failed += missing
            print("BATCH [{}/{}] fetched={} features={} missing={} {:.0f}s".format(
                min(start + args.batch, len(todo)), len(todo),
                len(fetched), len(made), missing, time.time() - started), flush=True)

    print("SHARD_DONE shard={} todo={} features={} failed={}".format(
        args.shard, len(todo), produced, failed), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
