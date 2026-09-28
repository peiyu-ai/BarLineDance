#!/usr/bin/env python3
"""Stage C over the OSS-resident corpus: inventory -> staging -> reconcile.

The corpus lives on OSS and the tools that read it do not: every path in
``preprocess_wild_3d`` is a local filesystem path, and the checkout cannot hold
the 408 GB ingest tree.  This module replaces the I/O boundary and calls the
same functions rather than reimplementing them -- the QC thresholds and the
strict conversion validation are exactly where a silent divergence would be
most expensive and least visible.  What was wrong with those tools was never
the logic.

Three properties it keeps, each because the alternative has already cost this
repo something:

* **A manifest records repo-relative keys, never absolute paths.**  A manifest
  holding ``/cache/...`` dies with the cache it names, and
  ``run_wild_rebuild.sh`` carries a guard against exactly that -- the guard
  exists because it happened on the v4 run.  Here the path a record carries is
  the key its bytes are under, so the manifest is readable from any machine
  that can reach the bucket.
* **Nothing intermediate is published.**  A clip's scratch copy is deleted
  once its record is written.  These credentials cannot delete (``rm`` returns
  403 AccessDenied), so an intermediate published once is permanent.
* **Every output key is derived from the corpus tag, not from the run.**
  Overwriting an existing key *is* permitted -- measured 2026-08-14, a
  byte-identical rewrite of an existing report returned the same sha256 -- so
  a stable key can be updated forever and never accumulates.  Minting a new
  key per run is what produces garbage nobody can remove.

Usage::

    python3 tools/run_wild_stage_c_oss.py inventory --shard 0 --num-shards 8
    python3 tools/run_wild_stage_c_oss.py merge --kind inventory
    python3 tools/run_wild_stage_c_oss.py staging
    python3 tools/run_wild_stage_c_oss.py reconcile --shard 0 --num-shards 8
    python3 tools/run_wild_stage_c_oss.py merge --kind reconcile
"""

import argparse
import json
import os
import pathlib
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import asset_io                                    # noqa: E402
from tools import preprocess_wild_3d as pw                    # noqa: E402
from tools import redo_manifest                               # noqa: E402

INGEST_ROOT = "data/wild_ingest_v1"
CONVERTED_ROOT = "data/wild3d/ingest_v1_converted"

# The files inventory_wild_cache actually opens.  Listed rather than fetching
# the whole clip directory because the clip also holds clip.mp4 (22 MB) and
# audio.wav (700 KB), and stage C reads neither.
INVENTORY_INPUTS = ("meta.json", "keypoints_clean2.npy", "keypoints_clean.npy",
                    "keypoints.npy", "scores.npy")


def tag_keys(tag: str) -> Dict[str, str]:
    """Every key this stage writes, derived from the tag and nothing else."""
    return {
        "inventory_parts": "runs/{}_inventory_parts".format(tag),
        "inventory": "runs/{}_inventory.jsonl".format(tag),
        "staging": "data/wild3d/{}_staging".format(tag),
        "reconcile_parts": "data/wild3d/{}_staging/sequences_hmr_parts".format(tag),
        "reconcile": "data/wild3d/{}_staging/sequences_hmr.jsonl".format(tag),
    }


def clip_contents(prefix: str) -> Dict[str, set]:
    """``{clip id: {file name}}`` for every clip under ``prefix``.

    One listing of the whole prefix, because that is the only form of listing
    this bucket answers: a per-directory walk is one LIST per clip and returns
    502.  It costs 7 seconds for 124,195 keys.

    It returns the file *names* and not just the ids because asking for a file
    that is not there is not free: ``read_bytes`` falls through to ossutil,
    which retries three times with a doubling backoff, so one absent key costs
    7.2 seconds against 0.00 for a present one.  Probing three optional
    keypoint variants per clip therefore cost ~14 s/clip -- 8.9 hours per shard
    -- and looked like a stall rather than a cost.  The listing already knows
    what exists, so nothing here guesses.
    """
    contents: Dict[str, set] = {}
    for name in asset_io.list_prefix(prefix):
        head, _, tail = name.partition("/")
        if tail:
            contents.setdefault(head, set()).add(tail)
    return contents


def repo_key(*parts: str) -> str:
    return "/".join(part.strip("/") for part in parts if part)


def _absolute_roots() -> List[str]:
    """Every absolute prefix that is really a repo-relative key in disguise.

    The checkout itself, and the asset cache that mirrors its layout.  A tree
    pulled with ``oss_assets.py pull --cache`` is a symlink into that cache, so
    anything that calls ``.resolve()`` on a path under it -- and
    ``build_wild_staging_manifests`` does -- comes back with ``/cache/...``.
    """
    roots = [str(REPO)]
    try:
        from tools import oss_assets                          # noqa: PLC0415
        roots.append(str(oss_assets.cache_root()))
    except Exception:                                         # noqa: BLE001
        roots.append("/cache/atomicdance-assets")
    return [root.rstrip("/") for root in roots if root]


def to_repo_keys(node: Any) -> Any:
    """Rewrite absolute paths under the checkout or the cache back to keys."""
    if isinstance(node, str):
        for root in _absolute_roots():
            if node == root:
                return ""
            if node.startswith(root + "/"):
                return node[len(root) + 1:]
        return node
    if isinstance(node, dict):
        return {key: to_repo_keys(value) for key, value in node.items()}
    if isinstance(node, list):
        return [to_repo_keys(value) for value in node]
    return node


# Directories the re-cuts worked in.  ``to_repo_keys`` strips the cache mount
# from ``/cache/atomicdance-assets/scratch/c1/...`` and leaves ``scratch/c1/...``,
# which looks like a key and is not one: nothing is ever published under it, and
# the bytes are deleted when the run ends.  2,463 clips of this corpus record
# such a path as their ``source_video`` (2,185 from the 2026-08-19 re-cut, 278
# from the 2026-08-25 CFR one), so this is a live case, not a hypothetical.
SCRATCH_PREFIXES = ("scratch/",)


def absolute_strings(node: Any, trail: str = "") -> List[str]:
    """Every remaining path that is not a key, with the field it sits in.

    Two shapes fail: an absolute path, which is not a key at all, and a
    ``scratch/`` path, which reads like one but names working space no reader
    can fetch.
    """
    if isinstance(node, str):
        unusable = node.startswith("/") or node.startswith(SCRATCH_PREFIXES)
        return ["{}={}".format(trail, node)] if unusable else []
    if isinstance(node, dict):
        return [hit for key, value in node.items()
                for hit in absolute_strings(value, "{}.{}".format(trail, key))]
    if isinstance(node, list):
        return [hit for value in node for hit in absolute_strings(value, trail + "[]")]
    return []


def publish_rows(key: str, rows: Sequence[Mapping[str, Any]]) -> str:
    """Normalise, then refuse to publish a manifest that still names a mount.

    Every write in this module goes through here, because the failure it
    guards is not hypothetical: ``run_wild_rebuild.sh`` carries a grep for
    ``"/cache/`` in the reconciled manifest, that guard exists because the v4
    run tripped it, and the first version of this module reproduced the bug
    anyway -- ``.resolve()`` inside the staging builder re-absolutised three
    fields per sequence after the inventory had carefully stored keys.

    A gate placed after the fact tells you afterwards.  This one runs before
    the bytes leave, and it can fail: a manifest naming ``/cache`` dies with
    the cache, and this stage is the last place that is cheap to fix.
    """
    cleaned = [to_repo_keys(dict(row)) for row in rows]
    offenders = [hit for row in cleaned for hit in absolute_strings(row)]
    if offenders:
        raise SystemExit(
            "refusing to publish {}: {} path(s) survived normalisation that are "
            "not keys -- absolute, or under scratch working space no reader can "
            "fetch.  First three: {}".format(key, len(offenders), offenders[:3]))
    return asset_io.write_jsonl(key, cleaned)


def rewrite_inventory_paths(record: Dict[str, Any], stem: str) -> Dict[str, Any]:
    """Replace scratch-absolute paths with the keys the bytes live under.

    ``inventory_wild_cache`` records ``cache_dir.resolve()`` and the resolved
    pose/score paths, which under this driver would name a temporary directory
    that is about to be deleted.  The identity a downstream reader needs is the
    key, so that is what is stored.
    """
    record["cache_dir"] = repo_key(INGEST_ROOT, stem)
    for field in ("pose2d_path", "scores_path"):
        value = record.get(field)
        if isinstance(value, str) and value:
            record[field] = repo_key(INGEST_ROOT, stem, pathlib.Path(value).name)
    return record


def inventory_stems(contents: Dict[str, set], clips, from_prefix: bool) -> List[str]:
    """The clips this stage will inventory, from a manifest rather than a listing.

    Until 2026-08-20 this was ``sorted(contents)`` -- the clip set derived by
    enumerating the OSS prefix, which CLAUDE.md §1.1 forbids for a reason this
    corpus makes concrete.  These credentials can PUT over a key and cannot
    delete one, so a clip that stops being produced leaves its object behind
    forever and every listing hands it back.  Measured on this prefix: 17,790
    stems carry a ``meta.json``, and **775 of them are fps re-cut orphans** --
    names no consumer may read, which the previous version inventoried, merged
    and published as part of the corpus.

    The listing is still what says which files a clip *has* (asking for an
    absent key costs 7.2 s of ossutil retries, so nothing here guesses), but it
    no longer decides *which clips exist*.

    A named stem the prefix cannot serve is an error, not a silent drop --
    stage F's ``eligible`` settled this shape already: the caller believes it
    named work, and quietly inventorying fewer clips than it asked for is how a
    re-run ends up covering part of the corpus and reporting success.

    ``from_prefix`` is the escape hatch, and it prints what it is including
    rather than looking like a manifest-driven run.  It exists because
    reproducing an older inventory requires the older behaviour, not because
    the behaviour is defensible.
    """
    present = {stem for stem, files in contents.items() if "meta.json" in files}
    if clips is None:
        if not from_prefix:
            raise SystemExit(
                "stage C needs --clips <manifest>: deriving the clip set from "
                "the OSS prefix returns every name the store ever held, "
                "including {} stem(s) this listing cannot distinguish from live "
                "ones.  Pass --from-prefix to do it anyway.".format(len(present)))
        print("clip set: OSS prefix enumeration, {} stem(s) with a meta.json -- "
              "this includes any orphan the store still holds".format(len(present)),
              flush=True)
        # Sorted so a shard's slice is the same on every machine and every
        # re-run; a shard assignment that moves would re-do finished work and
        # call it new.
        return sorted(present)

    try:
        named = redo_manifest.load(clips, "3d")
    except redo_manifest.ManifestError as error:
        raise SystemExit(str(error))
    unknown = sorted(set(named) - present)
    if unknown:
        raise SystemExit(
            "{} of {} stem(s) named by {} have no meta.json under {}; first few: "
            "{}".format(len(unknown), len(named), clips, INGEST_ROOT,
                        ", ".join(unknown[:5])))
    print("clip set: {} stem(s) from {} ({} stem(s) under the prefix not named)"
          .format(len(named), clips, len(present) - len(set(named))), flush=True)
    return sorted(set(named))


def cmd_inventory(args: argparse.Namespace) -> int:
    keys = tag_keys(args.tag)
    contents = clip_contents(INGEST_ROOT)
    stems = inventory_stems(contents, args.clips, args.from_prefix)
    mine = stems[args.shard::args.num_shards]
    if args.limit:
        mine = mine[: args.limit]
    print("inventory shard {}/{}: {} of {} clips".format(
        args.shard, args.num_shards, len(mine), len(stems)), flush=True)

    records: List[Dict[str, Any]] = []
    missing = 0
    for start in range(0, len(mine), args.batch):
        chunk = mine[start: start + args.batch]
        with asset_io.scratch_dir(prefix="stagec{}-".format(args.shard)) as scratch:
            staged: List[str] = []
            for stem in chunk:
                clip_dir = scratch / stem
                clip_dir.mkdir(parents=True, exist_ok=True)
                present = contents.get(stem, set())
                got_meta = False
                for name in INVENTORY_INPUTS:
                    if name not in present:
                        continue     # never ask for what the listing says is absent
                    payload = asset_io.read_bytes(repo_key(INGEST_ROOT, stem, name))
                    (clip_dir / name).write_bytes(payload)
                    got_meta = got_meta or name == "meta.json"
                if got_meta:
                    staged.append(stem)
                else:
                    missing += 1
                    print("NO_META {}".format(stem), flush=True)
            if not staged:
                continue
            # One call per batch over the staged tree: the same function the
            # local driver uses, with the same thresholds, so the two cannot
            # drift apart without a test noticing.
            batch = pw.inventory_wild_cache(
                scratch, recursive=False,
                min_frames=args.min_frames,
                min_visible_fraction=args.min_visible_fraction,
                min_score=args.min_score,
                max_frozen_fraction=args.max_frozen_fraction)
            by_clip = {record["clip_id"]: record for record in batch}
            for stem in staged:
                record = by_clip.get(stem)
                if record is None:
                    missing += 1
                    print("NO_RECORD {}".format(stem), flush=True)
                    continue
                records.append(rewrite_inventory_paths(record, stem))
        print("  {}/{} clips".format(min(start + args.batch, len(mine)), len(mine)),
              flush=True)

    part = "{}/part-{:03d}.jsonl".format(keys["inventory_parts"], args.shard)
    publish_rows(part, records)
    status: Dict[str, int] = {}
    for record in records:
        status[record["status"]] = status.get(record["status"], 0) + 1
    print("SHARD_DONE shard={} records={} missing={} status={}".format(
        args.shard, len(records), missing, json.dumps(status, sort_keys=True)), flush=True)
    return 0


def read_parts(prefix: str) -> Iterable[Dict[str, Any]]:
    """Every record from every published part, in shard order."""
    names = sorted(name for name in asset_io.list_prefix(prefix)
                   if name.endswith(".jsonl"))
    if not names:
        raise SystemExit("no parts under {}: run the sharded step first".format(prefix))
    print("merging {} part(s): {}".format(len(names), ", ".join(names)), flush=True)
    for name in names:
        yield from asset_io.read_jsonl(repo_key(prefix, name))


def cmd_merge(args: argparse.Namespace) -> int:
    keys = tag_keys(args.tag)
    source = keys["inventory_parts"] if args.kind == "inventory" else keys["reconcile_parts"]
    target = keys["inventory"] if args.kind == "inventory" else keys["reconcile"]
    rows = list(read_parts(source))

    # A duplicate id here means two shards claimed the same clip, which is the
    # failure a stale shard count produces and which every later stage would
    # read as a larger corpus rather than as a fault.
    field = "clip_id" if args.kind == "inventory" else "sequence_id"
    seen = set()
    for row in rows:
        value = row.get(field)
        if value in seen:
            raise SystemExit("duplicate {} in merged parts: {}".format(field, value))
        seen.add(value)

    publish_rows(target, rows)
    counts: Dict[str, int] = {}
    for row in rows:
        key = row.get("status") or row.get("qc", {}).get("hmr_status", "unknown")
        counts[key] = counts.get(key, 0) + 1
    print("wrote {} with {} rows: {}".format(target, len(rows), json.dumps(counts, sort_keys=True)))
    return 0


def cmd_staging(args: argparse.Namespace) -> int:
    keys = tag_keys(args.tag)
    records = list(asset_io.read_jsonl(keys["inventory"]))
    sources, sequences, summary = pw.build_wild_staging_manifests(
        records, corpus=args.tag, split_seed=args.split_seed,
        train_fraction=args.train_fraction, val_fraction=args.val_fraction,
        ready_only=not args.include_quarantine)
    publish_rows(repo_key(keys["staging"], "sources.jsonl"), sources)
    publish_rows(repo_key(keys["staging"], "sequences.jsonl"), sequences)
    asset_io.write_json(repo_key(keys["staging"], "summary.json"), to_repo_keys(summary))
    print("staging: {} sources, {} sequences -> {}".format(
        len(sources), len(sequences), keys["staging"]))
    return 0


def rebase_paths(node: Any, scratch_prefix: str, key_prefix: str) -> Any:
    """Rewrite every string under ``scratch_prefix`` to sit under ``key_prefix``.

    Recursive and blind rather than a list of field names, because the fields
    are not all in one place: ``reconcile_wild_hmr_sequences`` writes six paths
    across ``assets``, ``qc`` and the record root, and enumerating them is a
    list that goes stale the first time the function gains a seventh.  A record
    that still names the scratch directory names a directory that no longer
    exists, and the reader would only find out when it opened it.
    """
    if isinstance(node, str):
        if node == scratch_prefix:
            return key_prefix
        if node.startswith(scratch_prefix + "/"):
            return key_prefix + node[len(scratch_prefix):]
        return node
    if isinstance(node, dict):
        return {key: rebase_paths(value, scratch_prefix, key_prefix)
                for key, value in node.items()}
    if isinstance(node, list):
        return [rebase_paths(value, scratch_prefix, key_prefix) for value in node]
    return node


def cmd_reconcile(args: argparse.Namespace) -> int:
    keys = tag_keys(args.tag)
    staging = list(asset_io.read_jsonl(repo_key(keys["staging"], "sequences.jsonl")))
    mine = staging[args.shard::args.num_shards]
    if args.limit:
        mine = mine[: args.limit]
    print("reconcile shard {}/{}: {} of {} sequences".format(
        args.shard, args.num_shards, len(mine), len(staging)), flush=True)

    # Which clips have any converted output at all, from one listing.  This
    # decides 'pending' versus 'quarantine', and the two are different facts:
    # pending means 3D has not run and the clip is still re-runnable,
    # quarantine means it ran and produced something that failed validation.
    # Creating the directory unconditionally -- which the first version did --
    # makes is_dir() always true, so every un-converted clip is reported as a
    # broken conversion.  It read as 2,300 corrupt clips instead of 2,300
    # outstanding ones, and nothing downstream could tell the difference.
    converted_have = set(clip_contents(CONVERTED_ROOT))
    print("converted prefix holds {} clips".format(len(converted_have)), flush=True)

    rows: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    for index, staged in enumerate(mine):
        stem = staged.get("legacy_clip_id")
        with asset_io.scratch_dir(prefix="stagecr{}-".format(args.shard)) as scratch:
            converted = scratch / "converted"
            converted.mkdir(parents=True, exist_ok=True)
            if stem in converted_have:
                asset_io.fetch_dir(repo_key(CONVERTED_ROOT, stem), converted / stem)
            produced, tally = pw.reconcile_wild_hmr_sequences(
                [staged], converted_root=converted)
            # Rebase inside the scratch context: the prefix to replace is the
            # resolved scratch path, which stops existing the moment we leave.
            scratch_prefix = str((converted / stem).resolve())
        for row in produced:
            rows.append(rebase_paths(row, scratch_prefix,
                                     repo_key(CONVERTED_ROOT, stem)))
        # The second return value is a summary, and the tallies live one level
        # down in it; adding the summary's own string fields is a TypeError,
        # which is how this was caught before it ran over the whole corpus.
        for name, value in (tally.get("status_counts") or {}).items():
            counts[name] = counts.get(name, 0) + int(value)
        if (index + 1) % args.report_every == 0:
            print("  {}/{} {}".format(index + 1, len(mine), json.dumps(counts, sort_keys=True)),
                  flush=True)

    part = "{}/part-{:03d}.jsonl".format(keys["reconcile_parts"], args.shard)
    publish_rows(part, rows)
    print("SHARD_DONE shard={} rows={} {}".format(
        args.shard, len(rows), json.dumps(counts, sort_keys=True)), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default=os.environ.get("TAG", "wild_v4"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_shard_args(sub):
        sub.add_argument("--shard", type=int, default=int(os.environ.get("SHARD", 0)))
        sub.add_argument("--num-shards", type=int, default=int(os.environ.get("NUM_SHARDS", 1)))
        sub.add_argument("--limit", type=int, default=None)

    inventory = subparsers.add_parser("inventory", help="QC-gate the 2D caches")
    inventory.add_argument("--clips", default=os.environ.get("CLIPS"),
                           help="manifest naming the clips to inventory: a plain "
                                "stem list, a census, or a freshness audit. "
                                "Required, because the alternative is an OSS "
                                "prefix listing and this store cannot delete a "
                                "name it once held")
    inventory.add_argument("--from-prefix", action="store_true",
                           help="derive the clip set by enumerating the prefix, "
                                "orphans included; prints what it is including")
    add_shard_args(inventory)
    inventory.add_argument("--batch", type=int, default=200)
    inventory.add_argument("--min-frames", type=int, default=180)
    inventory.add_argument("--min-visible-fraction", type=float, default=0.60)
    inventory.add_argument("--min-score", type=float, default=0.30)
    inventory.add_argument("--max-frozen-fraction", type=float, default=0.30)
    inventory.set_defaults(func=cmd_inventory)

    merge = subparsers.add_parser("merge", help="assemble sharded parts")
    merge.add_argument("--kind", choices=("inventory", "reconcile"), required=True)
    merge.set_defaults(func=cmd_merge)

    staging = subparsers.add_parser("staging", help="recording/sequence staging manifests")
    staging.add_argument("--split-seed", type=int, default=20260805)
    staging.add_argument("--train-fraction", type=float, default=0.8)
    staging.add_argument("--val-fraction", type=float, default=0.1)
    staging.add_argument("--include-quarantine", action="store_true")
    staging.set_defaults(func=cmd_staging)

    reconcile = subparsers.add_parser("reconcile", help="join staging with validated 3D")
    add_shard_args(reconcile)
    reconcile.add_argument("--report-every", type=int, default=200)
    reconcile.set_defaults(func=cmd_reconcile)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
