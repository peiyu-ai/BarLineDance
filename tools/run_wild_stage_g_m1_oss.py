#!/usr/bin/env python3
"""Stage G / M1 over the OSS-resident corpus: Alg. 1 segmentation.

``segment_visual_atomics.py`` already shards -- ``--shard/--num-shards`` over a
features directory -- but it shards a *local* directory, and the S3D features
are 16,927 objects and 14.15 GiB on OSS.  Handing every shard the whole
directory would mean every shard fetching all of it.

So the sharding moves out one level: this driver splits the clip list, each
worker fetches only its own slice into scratch, and the segmenter runs over
that slice as if it were the whole corpus (``--num-shards 1``).  The result is
identical because Alg. 1 is per-clip; what changes is that a shard transfers
1.8 GiB instead of 14.15.

Two things it pins rather than inherits:

* **34/18, not the script's 32/18 default.**  ``run_wild_rebuild.sh`` pins
  these and says why: wild_v2 was built at 34/18, and taking the default would
  put two variables into the comparison the rebuild exists to make.  The
  vocabulary ceiling was tried as a tie-break and does not settle it -- 34/18
  wins on test, 32/18 on val, 36/20 on val by another reading -- so the reason
  is continuity with wild_v2 and nothing stronger.
* **The clip list comes from the published bundle, not from the feature
  store.**  The store holds 16,927 objects, 1,633 of which are stale features
  for clips of an older corpus that OSS cannot delete.  Segmenting those would
  quietly enlarge the corpus with clips no bundle contains.

``features_dir`` is recorded inside each shard's JSON and the merge step reads
it back from there, so a scratch path would survive into the merged
segmentation.  It is substituted for the key before merging, which also makes
the shards agree with each other.
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Dict, List

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import asset_io                                    # noqa: E402
from tools.run_wild_stage_c_oss import repo_key               # noqa: E402
from tools.run_wild_stage_e_oss import stage_e_keys           # noqa: E402

FEATURES_ROOT = "data/wild_visual_s3d"


def m1_keys(tag: str) -> Dict[str, str]:
    return {
        "parts": "runs/{}_seg_parts".format(tag),
        "segmentation": "runs/{}_seg/segmentation.json".format(tag),
    }


def feature_stem(row: Dict[str, object]) -> str:
    """The S3D feature file's stem for one bundle row.

    ``legacy_clip_id`` when the bundle kept it.  Otherwise derived from the
    sequence id, because the bundle deliberately re-points ``recording_id`` at
    the *clip* -- 'wild_v4:<upload>:clip000' names the same footage the feature
    store calls '<upload>__clip000', and using the sequence id verbatim would
    match nothing while looking like a corpus with no features.
    """
    clip = row.get("legacy_clip_id")
    if isinstance(clip, str) and clip:
        return clip
    sequence_id = str(row.get("sequence_id") or row.get("recording_id") or "")
    parts = sequence_id.split(":")
    return "{}__{}".format(parts[1], parts[2]) if len(parts) == 3 else sequence_id


def bundle_clips(tag: str) -> List[str]:
    """The clip ids the published performance bundle contains, sorted."""
    bundle = stage_e_keys(tag)["bundle"]
    stems = [feature_stem(row) for row
             in asset_io.read_jsonl(repo_key(bundle, "sequences.jsonl"))]
    return sorted({stem for stem in stems if stem})


def cmd_run(args: argparse.Namespace) -> int:
    keys = m1_keys(args.tag)
    stems = bundle_clips(args.tag)
    available = {name[:-4] for name in asset_io.list_prefix(FEATURES_ROOT)
                 if name.endswith(".npz")}
    usable = [stem for stem in stems if stem in available]
    missing = len(stems) - len(usable)
    mine = usable[args.shard::args.num_shards]
    if args.limit:
        mine = mine[: args.limit]
    print("M1 shard {}/{}: {} clips (bundle {}, features present {}, missing {})".format(
        args.shard, args.num_shards, len(mine), len(stems), len(usable), missing), flush=True)
    if missing:
        # Named, not swallowed: a clip in the bundle with no S3D feature cannot
        # be segmented, and the count belongs in the log rather than in a
        # silently shorter corpus.
        print("NOTE: {} bundle clips have no S3D feature and are not segmented".format(
            missing), flush=True)

    with asset_io.scratch_dir(prefix="m1-{}-".format(args.shard)) as scratch:
        features = scratch / "features"
        features.mkdir(parents=True, exist_ok=True)
        started = time.time()
        for index, stem in enumerate(mine):
            (features / (stem + ".npz")).write_bytes(
                asset_io.read_bytes("{}/{}.npz".format(FEATURES_ROOT, stem)))
            if (index + 1) % 500 == 0:
                print("   fetched {}/{} in {:.0f}s".format(
                    index + 1, len(mine), time.time() - started), flush=True)

        shard_json = scratch / "shard.json"
        env = dict(os.environ)
        env.update({name: str(args.threads) for name in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                     "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
        completed = subprocess.run(
            [sys.executable, "tools/segment_visual_atomics.py",
             "--features-dir", str(features), "--output", str(shard_json),
             "--frames-per-cluster", str(args.frames_per_cluster),
             "--min-length", str(args.min_length),
             "--index-weight", str(args.index_weight),
             "--seed", str(args.seed), "--shard", "0", "--num-shards", "1"],
            cwd=str(REPO), env=env)
        if completed.returncode != 0:
            raise SystemExit("segmentation shard {} failed rc={}".format(
                args.shard, completed.returncode))

        payload = json.loads(shard_json.read_text(encoding="utf-8"))
        # The merge reads features_dir back out of the shard files, so a scratch
        # path here would end up naming a deleted tmpfs directory in the merged
        # segmentation -- and the shards would disagree with each other besides.
        payload["features_dir"] = FEATURES_ROOT
        asset_io.write_json("{}/part-{:03d}.json".format(keys["parts"], args.shard),
                            payload)
    # Read the counts the segmenter computed rather than measuring the payload:
    # ``records`` is one row per sequence, so len(records) is a clip count that
    # reads as a segment count -- 20 where the answer was 312.
    print("SHARD_DONE shard={} sequences={} segments={} per_seq={}".format(
        args.shard, payload.get("sequences"), payload.get("total_segments"),
        payload.get("segments_per_sequence")), flush=True)
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    keys = m1_keys(args.tag)
    names = sorted(name for name in asset_io.list_prefix(keys["parts"])
                   if name.endswith(".json"))
    if not names:
        raise SystemExit("no parts under {}".format(keys["parts"]))
    with asset_io.scratch_dir(prefix="m1-merge-") as scratch:
        for name in names:
            (scratch / name).write_bytes(asset_io.read_bytes(repo_key(keys["parts"], name)))
        merged = scratch / "segmentation.json"
        completed = subprocess.run(
            [sys.executable, "tools/segment_visual_atomics.py",
             "--merge-glob", str(scratch / "part-*.json"),
             "--output", str(merged), "--features-dir", str(scratch)],
            cwd=str(REPO))
        if completed.returncode != 0:
            raise SystemExit("merge failed rc={}".format(completed.returncode))
        payload = json.loads(merged.read_text(encoding="utf-8"))
        payload["features_dir"] = FEATURES_ROOT
        text = json.dumps(payload, sort_keys=True)
        if "/dev/shm" in text:
            raise SystemExit("refusing to publish a segmentation that names scratch")
        asset_io.write_json(keys["segmentation"], payload)
    print("merged {} parts -> {}".format(len(names), keys["segmentation"]))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default=os.environ.get("TAG", "wild_v4"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="segment one shard of the bundle's clips")
    run.add_argument("--shard", type=int, default=int(os.environ.get("SHARD", 0)))
    run.add_argument("--num-shards", type=int, default=int(os.environ.get("NUM_SHARDS", 1)))
    run.add_argument("--frames-per-cluster", type=int, default=34)
    run.add_argument("--min-length", type=int, default=18)
    run.add_argument("--index-weight", type=float, default=4.0)
    run.add_argument("--seed", type=int, default=20260808)
    run.add_argument("--threads", type=int, default=6)
    run.add_argument("--limit", type=int, default=None)
    run.set_defaults(func=cmd_run)

    merge = subparsers.add_parser("merge", help="combine the shard segmentations")
    merge.set_defaults(func=cmd_merge)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    raise SystemExit(parsed.func(parsed))
