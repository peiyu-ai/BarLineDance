#!/usr/bin/env python3
"""Stage G / M2 over the OSS-resident corpus: TMR clustering into K prototypes.

Not sharded, for stage E's reason: the K-Means is fitted over every segment's
embedding at once, and a per-shard fit is a different statistic rather than a
faster one.  So the bundle and the normalized motion are materialised into
scratch, ``cluster_atomics_tmr.py`` runs against them unchanged, and only the
label bundle and the embedding cache are published.

The embedding cache *is* published, and it is the one intermediate that earns
it: encoding 230k segments through TMR is the expensive half of this stage, the
cache is keyed by the segmentation it was computed from, and M3c reads it back
rather than re-encoding.  Everything else scratch touches is deleted.

Bundle tag vs run tag (added 2026-08-20)
----------------------------------------
``--bundle-tag`` lets the *segmentation and its labels* live under a new tag
while the performance bundle, the normalized motion and the normalizer stay the
ones an earlier tag published.  It exists because clean5 replaced its M1 with a
music-beat grid (``tools/segment_on_music_beats.py``) and nothing else: the 3D,
the normalizer fit and the split are unchanged, so re-publishing a whole stage-E
tree under a new tag would spend hours to write bytes identical to wild_v4's --
into a store that cannot delete them.  The alternative, overwriting
``runs/wild_v4_seg/segmentation.json`` in place, would destroy a published
artifact to save a flag.

It is printed on every run rather than inferred, because a bundle silently
paired with someone else's segmentation is the exact shape of defect this
repository keeps paying for.

Defaults are the discovery script's, restated here rather than inherited:
K=100 and accept-quantile 0.85 are what wild_v2 was built at, and the whole
point of the rebuild is that the corpus is the only thing that changed.
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
from typing import Any, Dict, List

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import asset_io                                    # noqa: E402
from tools.run_wild_stage_c_oss import repo_key               # noqa: E402
from tools.run_wild_stage_e_oss import (                            # noqa: E402
    localise_tree, publish_tree, stage_e_keys)
from tools.run_wild_stage_g_m1_oss import m1_keys             # noqa: E402


def m2_keys(tag: str) -> Dict[str, str]:
    return {
        "labels": "data/wild3d/{}_labels".format(tag),
        "embeddings": "runs/{}_tmr_embeddings.npz".format(tag),
    }


def fetch_tree(key: str, dest: pathlib.Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    files = asset_io.fetch_dir(key, dest)
    print("   fetched {} objects for {}".format(len(files), key), flush=True)
    return len(files)


def cmd_run(args: argparse.Namespace) -> int:
    bundle_tag = args.bundle_tag or args.tag
    ekeys, mkeys, seg_keys = (stage_e_keys(bundle_tag), m2_keys(args.tag),
                              m1_keys(args.tag))
    print("tag {} | bundle tag {}{}".format(
        args.tag, bundle_tag,
        "  (segmentation and labels are this run's; bundle, normalized motion "
        "and normalizer are the other tag's)" if bundle_tag != args.tag else ""),
        flush=True)

    with asset_io.scratch_dir(prefix="m2-") as scratch:
        bundle = scratch / "performance"
        normalized = scratch / "normalized"
        normalizer = scratch / "normalizer"
        fetch_tree(ekeys["bundle"], bundle)
        fetch_tree(ekeys["normalized"], normalized)
        fetch_tree(ekeys["normalizer"], normalizer)

        segmentation = scratch / "segmentation.json"
        segmentation.write_bytes(asset_io.read_bytes(seg_keys["segmentation"]))

        # The published manifests carry repo-relative keys; the clustering tool
        # resolves them against the bundle root, so they have to be pointed at
        # the scratch copies before it reads them.
        fetched_subs = {str(bundle): ekeys["bundle"],
                        str(normalized): ekeys["normalized"],
                        str(normalizer): ekeys["normalizer"]}
        for tree in (bundle, normalized, normalizer):
            touched = localise_tree(tree, fetched_subs)
            print("   localised {} manifest(s) under {}".format(touched, tree.name), flush=True)

        labels = scratch / "labels"
        cache = scratch / "tmr_embeddings.npz"
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
        argv = [sys.executable, "tools/cluster_atomics_tmr.py",
                "--bundle", str(bundle),
                "--segmentation", str(segmentation),
                "--output-dir", str(labels),
                "--classes", str(args.classes),
                "--accept-quantile", str(args.accept_quantile),
                "--seed", str(args.seed),
                "--normalized-sequences", str(normalized / "sequences_normalized.jsonl"),
                "--normalizer-bundle", str(normalizer),
                "--sources", str(bundle / "sources.jsonl"),
                "--embedding-cache", str(cache),
                "--device", "cuda:0"]
        if args.limit:
            argv += ["--limit", str(args.limit)]
        print("$ {}".format(" ".join(argv)), flush=True)
        completed = subprocess.run(argv, cwd=str(REPO), env=env)
        if completed.returncode != 0:
            raise SystemExit("M2 failed rc={}".format(completed.returncode))

        subs = {str(bundle): ekeys["bundle"],
                str(normalized): ekeys["normalized"],
                str(normalizer): ekeys["normalizer"],
                str(segmentation): seg_keys["segmentation"],
                str(cache): mkeys["embeddings"],
                str(labels): mkeys["labels"]}
        publish_tree(labels, mkeys["labels"], subs, dry=args.dry_publish)
        if not args.dry_publish and cache.is_file():
            asset_io.write_bytes(mkeys["embeddings"], cache.read_bytes())
            print("   published {}".format(mkeys["embeddings"]), flush=True)

        report_path = labels / "report.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            # Read counts.* -- they are nested, and guessing at the top level
            # printed "None accepted of None segments", which is the number M3's
            # target-size is derived from.
            counts = report.get("counts") or {}
            accepted = counts.get("accepted_segments")
            total = counts.get("segments")
            print("\nM2: {} accepted of {} segments into {} prototypes".format(
                accepted, total, args.classes), flush=True)
            # The number M3 needs, computed here rather than left for somebody
            # to derive: run_atomic_discovery's --target-size 32 makes the
            # sub-class count a linear function of corpus size, and the paper's
            # own base-num scan says more is not better (100 -> FID_k 32.68,
            # 125 -> 34.57).  Scaling target-size by corpus size instead pins
            # the vocabulary at wild_v2's scale, which is the comparison the
            # rebuild exists to make.
            if isinstance(accepted, int) and accepted > 0:
                print("   target-size for a wild_v2-scale vocabulary "
                      "(857 sub-classes): {}".format(round(accepted / 857)), flush=True)
                print("   sub-classes at the script default --target-size 32: "
                      "{}".format(round(accepted / 32)), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default=os.environ.get("TAG", "wild_v4"))
    parser.add_argument("--bundle-tag", default=os.environ.get("BUNDLE_TAG"),
                        help="take the performance bundle, normalized motion and "
                             "normalizer from THIS tag while the segmentation and "
                             "the labels this run writes belong to --tag. Defaults "
                             "to --tag, i.e. the historical behaviour.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="encode, cluster, publish labels")
    run.add_argument("--classes", type=int, default=100)
    run.add_argument("--accept-quantile", type=float, default=0.85)
    run.add_argument("--seed", type=int, default=20260809)
    run.add_argument("--gpu", default=os.environ.get("GPU", "0"))
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--dry-publish", action="store_true")
    run.set_defaults(func=cmd_run)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    raise SystemExit(parsed.func(parsed))
