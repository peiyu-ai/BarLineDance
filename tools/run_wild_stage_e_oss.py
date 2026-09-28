#!/usr/bin/env python3
"""Stage E over the OSS-resident corpus: bundle -> normalizer -> normalized.

Unlike stages C and D this one does **not** shard, and the reason is a check
rather than a convenience.  ``build_wild_performance_bundle`` runs
``_check_group_ownership`` across the whole corpus -- one upload must not have
its clips split across groups -- and a sharded build would only ever check
within a shard, which is a weaker claim wearing the same name.  The normalizer
has the same shape: it is fitted on the train split, and a per-shard fit is a
different statistic, not a faster one.

So the working set is materialised into scratch once, the three tools run
against it unchanged, and only the contract products are published:

    <tag>_performance/   sequences.jsonl, sources.jsonl, per-sequence stores
    <tag>_normalizer/    the fitted bundle
    <tag>_normalized/    sequences_normalized.jsonl and the arrays
    runs/<tag>_3d_quality.json

Peak scratch is about 20 GB for 15k sequences on /dev/shm, and it is deleted
whether the run succeeds or not.  Nothing intermediate reaches the bucket,
because the bucket cannot delete.

The bundle's own manifest already stores paths relative to the bundle root, so
publishing it needs no rewriting; the audit and the input manifests do, and
they go through stage C's publish gate like everything else.
"""

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time
from typing import Any, Dict, List

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import asset_io                                    # noqa: E402
from tools.run_wild_stage_c_oss import (                      # noqa: E402
    absolute_strings, repo_key, tag_keys, to_repo_keys)
from tools.run_wild_stage_d_oss import audio_keys             # noqa: E402


def stage_e_keys(tag: str) -> Dict[str, str]:
    return {
        "bundle": "data/wild3d/{}_performance".format(tag),
        "normalizer": "data/wild3d/{}_normalizer".format(tag),
        "normalized": "data/wild3d/{}_normalized".format(tag),
        "quality": "runs/{}_3d_quality.json".format(tag),
    }


def run_tool(argv: List[str], label: str) -> None:
    started = time.time()
    print("\n== {} ==\n$ {}".format(label, " ".join(argv)), flush=True)
    completed = subprocess.run(argv, cwd=str(REPO))
    if completed.returncode != 0:
        raise SystemExit("{} failed with rc={}".format(label, completed.returncode))
    print("   {} ok in {:.0f}s".format(label, time.time() - started), flush=True)


def materialise_inputs(rows: List[Dict[str, Any]], scratch: pathlib.Path):
    """Fetch each candidate's motion and music, and point a manifest at them.

    Only ``audio_feature.status == candidate`` rows carry a music artifact, and
    the builder needs both arrays, so anything else is dropped here rather than
    failing one file at a time inside the builder.
    """
    store = scratch / "inputs"
    store.mkdir(parents=True, exist_ok=True)
    local_rows: List[Dict[str, Any]] = []
    # scratch path -> the key it stands in for.  Built here rather than
    # reconstructed at publish time because the mapping is not a prefix rebase:
    # a converted clip's ``atomic_motion_151.npy`` becomes ``motion_151_raw.npy``
    # under a directory named after the sequence, so only the code that made the
    # substitution knows how to undo it.
    subs: Dict[str, str] = {}
    skipped = 0
    for index, row in enumerate(rows):
        if (row.get("audio_feature") or {}).get("status") != "candidate":
            skipped += 1
            continue
        item = json.loads(json.dumps(row))
        assets = item["assets"]
        clip = store / item["sequence_id"].replace(":", "_")
        clip.mkdir(parents=True, exist_ok=True)
        for field, name in (("motion_151_raw", "motion_151_raw.npy"),
                            ("music_35", "music_35.npy")):
            target = clip / name
            target.write_bytes(asset_io.read_bytes(assets[field]))
            subs[str(target)] = assets[field]
            assets[field] = str(target)
        frame_ids = clip / "frame_ids.npy"
        frame_ids.write_bytes(asset_io.read_bytes(item["timeline"]["frame_ids_path"]))
        subs[str(frame_ids)] = item["timeline"]["frame_ids_path"]
        item["timeline"]["frame_ids_path"] = str(frame_ids)
        local_rows.append(item)
        if (index + 1) % 2000 == 0:
            print("   staged {}/{}".format(index + 1, len(rows)), flush=True)

    manifest = scratch / "sequences_audio_local.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for item in local_rows:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
    print("   {} candidates staged, {} rows skipped".format(len(local_rows), skipped),
          flush=True)
    return manifest, subs


def localise_tree(local_dir: pathlib.Path, subs: Dict[str, str]) -> int:
    """Point a fetched tree's manifests back at the bytes in scratch.

    The exact inverse of what publish_tree does, and needed for the same
    reason it exists.  Publishing rewrites scratch paths to repo-relative keys
    so a manifest survives the cache it was built on; but a consumer resolves
    those paths against the bundle root, so a key handed straight to
    ``cluster_atomics_tmr`` becomes ``<bundle>/data/wild3d/...`` and the file
    is not there.  Durable on the way out, local on the way in -- a driver that
    does only the first half publishes something nothing can read.

    Longest key first, so that one key which is a prefix of another cannot be
    rewritten twice.
    """
    inverse = sorted(((key, path) for path, key in subs.items()),
                     key=lambda pair: -len(pair[0]))
    touched = 0
    for manifest in sorted(local_dir.rglob("*.jsonl")) + sorted(local_dir.rglob("*.json")):
        text = original = manifest.read_text(encoding="utf-8")
        for key, path in inverse:
            if key in text:
                text = text.replace(key, path)
        if text != original:
            manifest.write_text(text, encoding="utf-8")
            touched += 1
    return touched


def publish_tree(local_dir: pathlib.Path, key: str, subs: Dict[str, str],
                 dry: bool = False) -> None:
    """Rewrite scratch paths to keys in every manifest, then publish.

    The refusal below is the same gate stage C and D carry, and it has now
    fired on all three: the builder copies its input paths into
    ``sources.jsonl`` as provenance, so a bundle published without this step
    would name a tmpfs directory that is deleted seconds later.  Substituting
    first and refusing second means the manifest is correct rather than merely
    rejected -- and a scratch path this map does not know about still stops the
    publish, because a provenance field pointing nowhere is worse than a run
    that has to be repeated.
    """
    for manifest in sorted(local_dir.rglob("*.jsonl")) + sorted(local_dir.rglob("*.json")):
        text = original = manifest.read_text(encoding="utf-8")
        for scratch_path, replacement in subs.items():
            if scratch_path in text:
                text = text.replace(scratch_path, replacement)
        if text != original:
            manifest.write_text(text, encoding="utf-8")
        if "/dev/shm" in text:
            # Report the offending token and the field that holds it, not the
            # line: a sources.jsonl row is 2 KB and the path is 60 characters
            # of it, so a truncated line says a leak exists without saying
            # which substitution is missing.
            tokens = sorted(set(re.findall(r'"([^"]*?/dev/shm[^"]*?)"', text)))
            fields = sorted(set(re.findall(r'"([A-Za-z0-9_]+)"\s*:\s*"[^"]*?/dev/shm', text)))
            raise SystemExit(
                "refusing to publish {}: {} still names scratch in field(s) {} -> {}".format(
                    key, manifest.relative_to(local_dir), fields[:5], tokens[:3]))
    if dry:
        print("   dry-publish ok: {} passes the scratch check".format(key), flush=True)
        return
    names = asset_io.publish_dir(local_dir, key)
    print("   published {} objects -> {}".format(len(names), key), flush=True)


def cmd_run(args: argparse.Namespace) -> int:
    keys, akeys, ekeys = tag_keys(args.tag), audio_keys(args.tag), stage_e_keys(args.tag)
    rows = list(asset_io.read_jsonl(akeys["manifest"]))
    if args.limit:
        rows = rows[: args.limit]
    print("stage E over {} audio rows".format(len(rows)), flush=True)

    with asset_io.scratch_dir(prefix="stagee-") as scratch:
        manifest, subs = materialise_inputs(rows, scratch)
        bundle = scratch / "performance"
        normalizer = scratch / "normalizer"
        normalized = scratch / "normalized"

        run_tool([sys.executable, "tools/build_wild_performance_bundle.py",
                  "--audio-manifest", str(manifest), "--output-dir", str(bundle)],
                 "build performance bundle")
        run_tool([sys.executable, "tools/fit_motion_normalizer.py",
                  "--sequence-manifest", str(bundle / "sequences.jsonl"),
                  "--source-manifest", str(bundle / "sources.jsonl"),
                  "--output-dir", str(normalizer)],
                 "fit normalizer on train")
        run_tool([sys.executable, "tools/apply_motion_normalizer.py",
                  "--sequence-manifest", str(bundle / "sequences.jsonl"),
                  "--source-manifest", str(bundle / "sources.jsonl"),
                  "--normalizer-bundle", str(normalizer),
                  "--output-dir", str(normalized)],
                 "apply normalizer")

        quality = scratch / "3d_quality.json"
        reference = REPO / "data/atomic_aistpp/aist_raw_performance_v1"
        if reference.exists():
            run_tool([sys.executable, "tools/audit_wild_3d_quality.py",
                      "--bundle", str(bundle), "--output", str(quality),
                      "--reference", str(reference)],
                     "3D quality audit against motion capture")
        else:
            # Say it rather than skip it quietly: the audit answers whether the
            # 24 s content cut cost 3D quality, and 'not run' must not read as
            # 'passed'.
            print("   NOTE: {} absent, 3D quality audit not run".format(reference), flush=True)

        # The three output roots stand in for their own keys too: the tools
        # record where they wrote as well as where they read.
        subs = dict(subs)
        # The local manifest handed to the builder stands in for the published
        # audio manifest: the builder records it as `audio_bundle_manifest`,
        # which is a provenance claim about which bundle produced these rows,
        # and the answer is the OSS one, not the tmpfs copy of it.
        subs[str(manifest)] = akeys["manifest"]
        subs[str(bundle)] = ekeys["bundle"]
        subs[str(normalizer)] = ekeys["normalizer"]
        subs[str(normalized)] = ekeys["normalized"]
        for local, key in ((bundle, ekeys["bundle"]),
                           (normalizer, ekeys["normalizer"]),
                           (normalized, ekeys["normalized"])):
            publish_tree(local, key, subs, dry=args.dry_publish)
        if quality.is_file():
            text = quality.read_text(encoding="utf-8")
            for scratch_path, replacement in subs.items():
                text = text.replace(scratch_path, replacement)
            report = to_repo_keys(json.loads(text))
            offenders = absolute_strings(report)
            if offenders:
                raise SystemExit("quality report still names a mount: {}".format(offenders[:3]))
            asset_io.write_json(ekeys["quality"], report)
            print("   published {}".format(ekeys["quality"]), flush=True)

    print("\nstage E complete; scratch removed")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default=os.environ.get("TAG", "wild_v4"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="bundle, fit, apply, audit, publish")
    run.add_argument("--limit", type=int, default=None,
                     help="first N audio rows only; for smoking the publish path")
    run.add_argument("--dry-publish", action="store_true",
                     help="run every check but do not upload")
    run.set_defaults(func=cmd_run)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    raise SystemExit(parsed.func(parsed))
