#!/usr/bin/env python3
"""Stage D over the OSS-resident corpus: 35-D audio aligned to converted frames.

Same boundary swap as stage C -- ``materialize_wild_music_features`` is called
unchanged, and only the I/O around it moves.  The feature extractor, the strict
frame-id alignment and the per-clip quarantine semantics are the parts that
must not drift, so none of them are reimplemented here.

Why this shards where the local driver does not.  ``publish_wild_music_bundle``
is all-or-nothing: one mkdtemp, one ``os.rename``, and it refuses a pre-existing
output directory.  That is the right contract for a directory of many files
written in one go, but 17,621 sequences of librosa is not one go -- a run
interrupted at sequence 14,000 loses all of it, and this session has already
had two multi-hour runs killed by something outside the job.

So the atomicity is kept, one level down: each ``music_35/<sha256>.npy`` is
still written once and never overwritten (the name is a hash of the sequence
id, so shards cannot collide), each shard publishes its own manifest part, and
the merged manifest is written *last*.  A reader that sees
``sequences_audio.jsonl`` sees a complete bundle; a reader that does not, sees
nothing to mistake for one.  What is given up is 'the directory appears at one
instant', which no OSS prefix offers anyway.
"""

import argparse
import hashlib
import json
import os
import pathlib
import sys
from typing import Any, Dict, List, Mapping, Sequence

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import asset_io                                    # noqa: E402
from tools import extract_wild_music_features as audio        # noqa: E402
from tools.run_wild_stage_c_oss import (                      # noqa: E402
    CONVERTED_ROOT, INGEST_ROOT, publish_rows, rebase_paths, repo_key,
    tag_keys, to_repo_keys)


def audio_keys(tag: str) -> Dict[str, str]:
    root = "data/wild3d/{}_audio35".format(tag)
    return {
        "root": root,
        "music": repo_key(root, "music_35"),
        "parts": repo_key(root, "sequences_audio_parts"),
        "manifest": repo_key(root, "sequences_audio.jsonl"),
        "summary": repo_key(root, "summary.json"),
    }


def input_manifest_hash(key: str) -> str:
    """sha256 of the whole reconciled manifest, not of this shard's slice.

    Provenance should answer 'which manifest produced this', and every shard
    was produced by the same one.  Hashing the slice would give each shard a
    different answer to a question that has one.
    """
    return hashlib.sha256(asset_io.read_bytes(key)).hexdigest()


def stage_candidate(row: Mapping[str, Any], scratch: pathlib.Path) -> Dict[str, Any]:
    """Copy one candidate's audio and frame ids into scratch, pointing at them."""
    item = json.loads(json.dumps(row))
    stem = item.get("legacy_clip_id")
    clip_dir = scratch / "ingest" / stem
    clip_dir.mkdir(parents=True, exist_ok=True)
    (clip_dir / "audio.wav").write_bytes(
        asset_io.read_bytes(repo_key(INGEST_ROOT, stem, "audio.wav")))
    frame_ids_key = item["timeline"]["frame_ids_path"]
    frame_ids_local = scratch / "converted" / stem / "frame_ids.npy"
    frame_ids_local.parent.mkdir(parents=True, exist_ok=True)
    frame_ids_local.write_bytes(asset_io.read_bytes(frame_ids_key))
    item["assets"]["source_cache"] = str(clip_dir)
    item["timeline"]["frame_ids_path"] = str(frame_ids_local)
    return item


def restore_keys(out: Dict[str, Any], scratch: str) -> Dict[str, Any]:
    """Point every injected scratch path back at the key it stood in for.

    Two rebases rather than a list of field names, for the reason stage C
    learned the same way: the extractor writes the paths it read into
    ``audio_feature.source_audio`` and
    ``audio_feature.frame_selection.frame_ids_path`` as well as the two fields
    this driver set, and a version of this function that named four fields
    published records pointing at a deleted temp directory -- caught only
    because publish_rows refuses them.  Rebasing the roots covers the fields
    nobody has thought of yet.
    """
    out = rebase_paths(out, "{}/ingest".format(scratch), INGEST_ROOT)
    return rebase_paths(out, "{}/converted".format(scratch), CONVERTED_ROOT)


def cmd_run(args: argparse.Namespace) -> int:
    keys, akeys = tag_keys(args.tag), audio_keys(args.tag)
    rows = list(asset_io.read_jsonl(keys["reconcile"]))
    manifest_sha = input_manifest_hash(keys["reconcile"])
    mine = rows[args.shard::args.num_shards]
    if args.limit:
        mine = mine[: args.limit]
    candidates = sum(1 for row in mine
                     if (row.get("qc") or {}).get("hmr_status") == "candidate")
    print("stage D shard {}/{}: {} sequences, {} candidates".format(
        args.shard, args.num_shards, len(mine), candidates), flush=True)

    produced: List[Dict[str, Any]] = []
    totals: Dict[str, int] = {}
    for start in range(0, len(mine), args.batch):
        chunk = mine[start: start + args.batch]
        with asset_io.scratch_dir(prefix="staged{}-".format(args.shard)) as scratch:
            staged: List[Dict[str, Any]] = []
            for row in chunk:
                if (row.get("qc") or {}).get("hmr_status") != "candidate":
                    staged.append(json.loads(json.dumps(row)))   # counted not_attempted
                    continue
                try:
                    staged.append(stage_candidate(row, scratch))
                except Exception as error:                     # noqa: BLE001
                    print("FETCH_FAIL {} {}".format(
                        row.get("sequence_id"), str(error)[:120]), flush=True)
                    staged.append(json.loads(json.dumps(row)))  # extractor quarantines it
            records, summary = audio.materialize_wild_music_features(
                staged,
                artifact_root=scratch / "music_35",
                public_artifact_root=REPO / akeys["music"],
                input_manifest_sha256=manifest_sha)
            # Count the artifacts, not the return value: the extractor
            # quarantines a clip and carries on, so 'it returned records' and
            # 'it wrote features' are different claims.
            made = sorted((scratch / "music_35").glob("*.npy"))
            if made:
                asset_io.publish_dir(scratch / "music_35", akeys["music"])
            scratch_root = str(scratch.resolve())
        produced.extend(restore_keys(record, scratch_root) for record in records)
        for name, value in (summary.get("status_counts") or {}).items():
            totals[name] = totals.get(name, 0) + int(value)
        print("  {}/{} published={} {}".format(
            min(start + args.batch, len(mine)), len(mine), len(made),
            json.dumps(totals, sort_keys=True)), flush=True)

    publish_rows("{}/part-{:03d}.jsonl".format(akeys["parts"], args.shard), produced)
    print("SHARD_DONE shard={} rows={} {}".format(
        args.shard, len(produced), json.dumps(totals, sort_keys=True)), flush=True)
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    akeys = audio_keys(args.tag)
    names = sorted(name for name in asset_io.list_prefix(akeys["parts"])
                   if name.endswith(".jsonl"))
    if not names:
        raise SystemExit("no parts under {}".format(akeys["parts"]))
    rows: List[Dict[str, Any]] = []
    seen = set()
    for name in names:
        for row in asset_io.read_jsonl(repo_key(akeys["parts"], name)):
            if row["sequence_id"] in seen:
                raise SystemExit("duplicate sequence_id across parts: {}".format(row["sequence_id"]))
            seen.add(row["sequence_id"])
            rows.append(row)

    counts: Dict[str, int] = {}
    for row in rows:
        status = (row.get("audio_feature") or {}).get("status", "not_attempted")
        counts[status] = counts.get(status, 0) + 1

    # Cross-check the manifest against the store rather than trusting it: a
    # published count that nobody verified is how this repo has been misled
    # before.
    published = sum(1 for name in asset_io.list_prefix(akeys["music"])
                    if name.endswith(".npy"))
    claimed = counts.get("candidate", 0)
    print("merged {} parts -> {} rows {}".format(len(names), len(rows),
                                                 json.dumps(counts, sort_keys=True)))
    print("music_35 objects in the store: {} (manifest claims {})".format(published, claimed))
    if published < claimed:
        raise SystemExit("refusing to publish: {} candidate rows but only {} artifacts".format(
            claimed, published))

    publish_rows(akeys["manifest"], rows)
    asset_io.write_json(akeys["summary"], to_repo_keys({
        "schema_version": audio.SCHEMA_VERSION,
        "corpus": args.tag,
        "input_manifest": tag_keys(args.tag)["reconcile"],
        "input_manifest_sha256": input_manifest_hash(tag_keys(args.tag)["reconcile"]),
        "status_counts": counts,
        "music_35_objects": published,
        "bundle_layout": {"music_features": "music_35/<sha256(sequence_id)>.npy"},
        "publication": "sharded; per-artifact immutable, merged manifest written last",
    }))
    print("wrote {}".format(akeys["manifest"]))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default=os.environ.get("TAG", "wild_v4"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="extract 35-D features for one shard")
    run.add_argument("--shard", type=int, default=int(os.environ.get("SHARD", 0)))
    run.add_argument("--num-shards", type=int, default=int(os.environ.get("NUM_SHARDS", 1)))
    run.add_argument("--batch", type=int, default=100)
    run.add_argument("--limit", type=int, default=None)
    run.set_defaults(func=cmd_run)

    merge = subparsers.add_parser("merge", help="assemble parts into the bundle manifest")
    merge.set_defaults(func=cmd_merge)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))
