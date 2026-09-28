#!/usr/bin/env python3
"""Park the bulk assets in OSS so the working tree holds code, not corpora.

``data/`` and ``third_party/`` are 191 GB between them, which is most of what
this checkout costs on the NAS.  Neither is source: ``data/`` is corpora and
derived bundles, ``third_party/`` is vendored code plus model weights.  This
tool moves them to the team OSS store and brings them back on demand, with the
one discipline that makes that safe: **nothing local is deleted until a byte
census says the remote copy is complete**.

Layout.  A repo-relative path maps to exactly one key::

    data/wild3d/...        -> <prefix>/AtomicDance/data/wild3d/...
    third_party/TMR/...    -> <prefix>/AtomicDance/third_party/TMR/...

The ``AtomicDance/`` component is not decoration.  The prefix already holds
``data/tiktok/`` and ``models/`` belonging to other work; pushing this repo's
``data/`` straight onto it would interleave two unrelated trees into one
namespace that no later reader could separate.

Two model directories are **not** mirrored, because they are already in the
store under keys of their own -- ``tools/setup_qwenvl_env.sh`` is what put them
there and is what brings them back.  Re-uploading them would spend 89 GB to
create a second copy that can drift from the first.  ``status`` shows them as
``upstream`` and ``push`` skips them; ``verify`` still checks them, because
"already in OSS" is a claim that has to hold before anything is deleted.

Credentials never live in this repo.  They come from ``OSS_ACCESS_KEY_ID`` /
``OSS_ACCESS_KEY_SECRET`` / ``OSS_ENDPOINT``, or from ``~/.ossutilconfig``
(mode 600) which ossutil already reads.

Usage::

    python3 tools/oss_assets.py status data third_party
    python3 tools/oss_assets.py push   data third_party
    python3 tools/oss_assets.py verify data third_party
    python3 tools/oss_assets.py evict  data/wild3d --yes
    python3 tools/oss_assets.py pull   data/wild3d/wild_performance_v1 --cache

Reading from OSS, without rewriting 170 call sites.  ``pull --cache`` puts the
tree under ``/cache/atomicdance-assets/<repo path>`` and leaves a symlink at
the repo path, so every ``np.load(bundle / row["motion_path"])`` in this
codebase keeps working untouched while the bytes sit off the NAS.  Streaming
each read straight out of OSS was the alternative and is the wrong trade here:
the training loop re-reads the same windows every epoch, so per-read fetches
would pay network latency thousands of times for data that fits on local disk.
Granularity is the bundle, not the corpus -- ``wild_performance_v1`` alone is
what most stages need, and pulling ``data/wild3d`` whole would undo the point.
"""

from __future__ import annotations

import argparse
import configparser
import fnmatch
import os
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_PREFIX = os.environ.get("OSS_ASSET_PREFIX", "oss://example-bucket/example/prefix/")
MIRROR_ROOT = "AtomicDance"
OSSUTIL = os.environ.get("OSSUTIL", "/opt/data-infra/ossutil64")

# Already in the team store under their own keys, with the script that restores
# them.  Mirroring these would duplicate 89 GB and create a second copy that
# can drift from the one every other project already pulls.
UPSTREAM: Dict[str, Tuple[str, str]] = {
    "third_party/QwenVL/Qwen2.5-VL-7B-Instruct":
        ("models/Qwen2.5-VL-7B-Instruct", "tools/setup_qwenvl_env.sh"),
    "third_party/QwenVL/Qwen3-VL-30B-A3B-Instruct":
        ("models/Qwen3-VL-30B-A3B-Instruct",
         "MODEL=Qwen3-VL-30B-A3B-Instruct tools/setup_qwenvl_env.sh"),
}

# Derived, machine-specific, or rebuilt by a setup script.  A virtualenv in
# particular hardcodes absolute interpreter paths, so a restored copy would be
# broken in a way that only shows up at import time.
# ``.venv*`` rather than ``.venv`` plus ``.venv-*``: the DWPose environment is
# ``.venv_ortgpu``, an underscore, which matched neither -- 298 MB of wheels one
# `oss_assets.py push` away from an object store these credentials cannot delete.
EXCLUDE = ("__pycache__", "*.pyc", ".git", ".pytest_cache", ".venv*",
           ".ipynb_checkpoints")

# ossutil's --exclude refuses any pattern containing directory information, so
# it can only screen *file names*.  Directory-level exclusion therefore cannot
# be delegated to it: ``excluded_dirs`` finds those trees locally and they are
# pushed as explicit sub-paths instead.  An extra object uploaded here cannot
# fail `verify`, which only asks whether every local file arrived.
OSSUTIL_EXCLUDE = ("*.pyc",)


class AssetError(RuntimeError):
    pass


def excluded(relative: pathlib.PurePath) -> bool:
    return any(fnmatch.fnmatch(part, pattern)
               for part in relative.parts for pattern in EXCLUDE)


def parse_prefix(prefix: str) -> Tuple[str, str]:
    """``oss://bucket/some/prefix/`` -> ``(bucket, 'some/prefix/')``."""
    if not prefix.startswith("oss://"):
        raise AssetError("prefix must start with oss://, got {}".format(prefix))
    bucket, _, key = prefix[len("oss://"):].partition("/")
    if not bucket:
        raise AssetError("no bucket in {}".format(prefix))
    return bucket, (key if key.endswith("/") or not key else key + "/")


def upstream_for(relative: str) -> Optional[Tuple[str, str, str]]:
    """(asset_root, remote_subkey, restore_hint) if this path is upstream-owned."""
    for root, (key, restore) in UPSTREAM.items():
        if relative == root or relative.startswith(root + "/"):
            return root, key, restore
    return None


def remote_key(relative: str, root_key: str) -> str:
    """Repo-relative path -> full OSS key, honouring the upstream table."""
    match = upstream_for(relative)
    if match is None:
        return "{}{}/{}".format(root_key, MIRROR_ROOT, relative)
    asset_root, key, _ = match
    tail = relative[len(asset_root):].lstrip("/")
    return "{}{}{}".format(root_key, key, "/" + tail if tail else "")


def cache_root() -> pathlib.Path:
    """Where parked assets are re-materialised, off the NAS by default.

    The point of the migration is to get bytes off ``/workspace``; downloading
    them back to ``/workspace`` would undo it.  ``/cache`` is the local CPFS
    mount, so a pulled tree is both cheap to read and outside the quota that
    prompted this.
    """
    return pathlib.Path(os.environ.get("ATOMICDANCE_ASSET_CACHE", "/cache/atomicdance-assets"))


def walk_local(path: pathlib.Path, relative_root: Optional[str] = None) -> Dict[str, int]:
    """Repo-relative path -> size, for every file under ``path``.

    ``relative_root`` is the repo path ``path`` stands for.  The two differ
    once a tree has been parked in the cache and symlinked back into the
    checkout: the bytes are at ``/cache/...`` but every key here, and so every
    key ``verify`` compares against OSS, must still be the repo-relative name.

    Symlinks *inside* the tree are skipped rather than followed.  Following
    them would upload the target under the link's name, and ``data/`` holds
    11k links into a sibling project whose videos are not ours to copy.
    """
    base = pathlib.PurePath(relative_root) if relative_root else path.relative_to(REPO_ROOT)
    found: Dict[str, int] = {}
    if path.is_file():
        if not excluded(base):
            found[str(base)] = path.stat().st_size
        return found
    for entry in path.rglob("*"):
        if entry.is_symlink() or not entry.is_file():
            continue
        relative = base / entry.relative_to(path)
        if excluded(relative):
            continue
        found[str(relative)] = entry.stat().st_size
    return found


def excluded_dirs(root: pathlib.Path, base: pathlib.PurePath) -> List[pathlib.Path]:
    """Every excluded directory under ``root``, without descending into them.

    ``base`` is the repo-relative name ``root`` stands for.  It is not always
    ``root.relative_to(REPO_ROOT)``: a parked tree lives in the cache and is
    only *named* by the checkout, and asking a cache path for its position
    inside the repo raises.
    """
    found: List[pathlib.Path] = []
    for dirpath, dirnames, _ in os.walk(root):
        keep = []
        for name in dirnames:
            child = pathlib.Path(dirpath) / name
            if excluded(base / child.relative_to(root)):
                found.append(child)
            else:
                keep.append(name)
        dirnames[:] = keep
    return found


def push_targets(path: pathlib.Path, forbidden: Sequence[pathlib.Path],
                 base: pathlib.PurePath) -> List[pathlib.Path]:
    """The largest sub-paths of ``path`` that contain nothing excluded.

    ossutil cannot be told to skip a directory, so a tree holding one is handed
    over as several smaller trees instead.  Splitting only where necessary
    keeps the common case -- a corpus directory with nothing to skip -- as a
    single sync, which is where ossutil's concurrency actually helps.
    """
    if not any(item == path or str(item).startswith(str(path) + os.sep)
               for item in forbidden):
        return [path]
    targets: List[pathlib.Path] = []
    for child in sorted(path.iterdir()):
        if child.is_symlink() or excluded(base / child.name):
            continue
        targets.extend([child] if child.is_file()
                       else push_targets(child, forbidden, base / child.name))
    return targets


REPO_OSSUTIL_CONFIG = pathlib.Path(__file__).resolve().parents[1] / "ossutilconfig"


def ossutil_config():
    """The first ossutil config that exists, or None.

    The repo-local one is searched *before* ``~/.ossutilconfig`` on purpose: the
    home copy in this container is stale and its key is refused outright
    (``InvalidAccessKeyId``, checked 2026-08-24), so preferring it would make
    every caller fail with a credentials error while a working config sat in the
    checkout.  Order is explicit rather than "whichever is found", because which
    key signed a push is not something a later reader can recover.
    """
    override = os.environ.get("OSSUTIL_CONFIG")
    if override:
        # Exclusive, not first-in-a-list.  A caller who names a config and gets
        # a different one authenticates as an identity it did not choose, and
        # nothing in the artifact would record which key signed the transfer --
        # the ambiguity this function exists to remove.  Absent means absent.
        candidate = pathlib.Path(override)
        return candidate if candidate.is_file() else None
    for candidate in (REPO_OSSUTIL_CONFIG, pathlib.Path.home() / ".ossutilconfig"):
        if candidate.is_file():
            return candidate
    return None


def make_bucket(prefix: str):
    import oss2

    bucket_name, _ = parse_prefix(prefix)
    key_id = os.environ.get("OSS_ACCESS_KEY_ID")
    secret = os.environ.get("OSS_ACCESS_KEY_SECRET")
    endpoint = os.environ.get("OSS_ENDPOINT")
    if not (key_id and secret):
        config_path = ossutil_config()
        if config_path is None:
            raise AssetError(
                "no OSS credentials: set OSS_ACCESS_KEY_ID / OSS_ACCESS_KEY_SECRET, "
                "or put an ossutil config at $OSSUTIL_CONFIG, {} or {}".format(
                    REPO_OSSUTIL_CONFIG, pathlib.Path.home() / ".ossutilconfig"))
        parser = configparser.ConfigParser()
        parser.read(config_path)
        section = parser["Credentials"]
        key_id = key_id or section.get("accessKeyID")
        secret = secret or section.get("accessKeySecret")
        endpoint = endpoint or section.get("endpoint")
    if not endpoint:
        raise AssetError("no OSS endpoint; set OSS_ENDPOINT")
    if not endpoint.startswith("http"):
        endpoint = "http://" + endpoint
    return oss2.Bucket(oss2.Auth(key_id, secret), endpoint, bucket_name)


def list_remote(bucket, key_prefix: str, attempts: int = 3) -> Dict[str, int]:
    """Every object under a key prefix -> size, via ossutil.

    Not ``oss2.ObjectIterator``.  This is a CPFS-OSS *bridge* bucket: it serves
    GET through the SDK but answers a deep LIST with a 502 carrying no request
    id, at both credential sets, every time -- while ``ossutil ls`` returns
    113,921 objects in five seconds.  The census is what stands between a push
    and an eviction, so a listing that cannot run means nothing can ever be
    freed; and a listing that *half* runs would be worse, because a short
    remote side reads as "not uploaded yet" and, in the other direction, would
    be the thing that let a delete through on a tree that was never complete.
    """
    bucket_name = bucket.bucket_name
    argv = [OSSUTIL, *_config_flag(), "ls",
            "oss://{}/{}".format(bucket_name, key_prefix)]
    last = None
    for attempt in range(attempts):
        completed = subprocess.run(argv, capture_output=True, text=True)
        if completed.returncode == 0:
            found = {}
            for line in completed.stdout.splitlines():
                fields = line.split()
                if len(fields) < 7 or not fields[-1].startswith("oss://"):
                    continue
                try:
                    size = int(fields[fields.index("UTC") + 1])
                except (ValueError, IndexError):
                    continue
                key = fields[-1][len("oss://"):].split("/", 1)[1]
                if not key.endswith("/"):
                    found[key] = size
            return found
        last = (completed.stderr or completed.stdout).strip()[:300]
        time.sleep(2 ** attempt)
    raise AssetError("listing {} failed after {} attempts: {}".format(
        key_prefix, attempts, last))


def census(paths: Sequence[str], prefix: str) -> Dict[str, object]:
    """Local files vs remote objects, per requested path."""
    _, root_key = parse_prefix(prefix)
    bucket = make_bucket(prefix)
    report: Dict[str, object] = {}
    for raw in paths:
        requested = REPO_ROOT / raw
        if not requested.exists():
            raise AssetError("{} does not exist".format(requested))
        # A parked tree is a symlink into the cache.  Resolving the *requested*
        # path keeps its repo-relative name -- which is what the OSS key is
        # built from -- while the census counts the bytes wherever they landed.
        linked = requested.is_symlink()
        relative = str(pathlib.Path(os.path.normpath(raw)))
        target = requested.resolve()
        local = walk_local(target, relative)
        remote = list_remote(bucket, remote_key(relative, root_key))
        missing, mismatched = [], []
        for name, size in sorted(local.items()):
            key = remote_key(name, root_key)
            if key not in remote:
                missing.append(name)
            elif remote[key] != size:
                mismatched.append((name, size, remote[key]))
        report[relative] = {
            "upstream": upstream_for(relative) is not None,
            "linked": linked,
            "materialised_at": str(target),
            "local_files": len(local),
            "local_bytes": sum(local.values()),
            "remote_objects": len(remote),
            "remote_bytes": sum(remote.values()),
            "missing": missing,
            "mismatched": mismatched,
        }
    return report


def human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return "{:.1f}{}".format(size, unit)
        size /= 1024
    return "{:.1f}TB".format(size)


def _config_flag():
    """``-c <config>`` for the config ``ossutil_config`` picked, or nothing.

    Passed explicitly rather than left to ossutil's own default, which is
    ``~/.ossutilconfig`` -- stale here, and its key is refused.  Without this the
    library half of this file (oss2, which reads the resolved config) and the
    subprocess half (ossutil, which did not) authenticate as two different
    identities, and only one of them works.
    """
    config = ossutil_config()
    return ["-c", str(config)] if config else []


def fetch_object(key: str, destination: pathlib.Path, prefix: str = DEFAULT_PREFIX) -> bool:
    """Bring one object down, on demand.  For a handful of files, not a tree.

    ``pull`` materialises whole trees, which is right for a corpus and wrong for
    "the six clips this figure draws": ``data/wild_ingest_v1`` is thousands of
    clips and a page needs a few.  The destination's parent is created; an
    existing file is left alone, so a second render costs nothing.
    """
    destination = pathlib.Path(destination)
    if destination.is_file():
        return True
    destination.parent.mkdir(parents=True, exist_ok=True)
    remote = "{}/{}".format(prefix.rstrip("/"), key.lstrip("/"))
    completed = subprocess.run(
        [OSSUTIL, *_config_flag(), "cp", remote, str(destination)],
        capture_output=True, text=True)
    return completed.returncode == 0 and destination.is_file()


def run_ossutil(argv: Sequence[str]) -> int:
    if not pathlib.Path(OSSUTIL).is_file() and shutil.which(OSSUTIL) is None:
        raise AssetError("ossutil not found at {}; set OSSUTIL".format(OSSUTIL))
    argv = [*_config_flag(), *argv]
    print("$ {} {}".format(OSSUTIL, " ".join(argv)), flush=True)
    return subprocess.call([OSSUTIL, *argv])


def cmd_status(args) -> int:
    report = census(args.paths, args.prefix)
    for name, entry in report.items():
        tag = " [upstream]" if entry["upstream"] else ""
        if entry["linked"]:
            tag += " [cached at {}]".format(entry["materialised_at"])
        complete = not entry["missing"] and not entry["mismatched"]
        print("{}{}\n  local  {:>7} in {} files\n  remote {:>7} in {} objects\n"
              "  {} missing, {} size-mismatched -> {}".format(
                  name, tag, human(entry["local_bytes"]), entry["local_files"],
                  human(entry["remote_bytes"]), entry["remote_objects"],
                  len(entry["missing"]), len(entry["mismatched"]),
                  "COMPLETE" if complete else "INCOMPLETE"))
    return 0


def cmd_push(args) -> int:
    _, root_key = parse_prefix(args.prefix)
    bucket_name, _ = parse_prefix(args.prefix)
    status = 0
    for raw in args.paths:
        # The repo-relative name and the bytes' actual location are two
        # different things once a tree has been parked in the cache and
        # symlinked back.  census() has always kept them apart; push did not,
        # and resolved first -- so `relative_to(REPO_ROOT)` raised on every
        # parked tree and runs/, parked on 2026-08-12, could never be uploaded
        # at all.  The key must come from the name, the bytes from the target.
        base = pathlib.PurePath(os.path.normpath(raw))
        target = (REPO_ROOT / raw).resolve()
        relative = str(base)
        match = upstream_for(relative)
        if match is not None:
            print("skip {}: already in the store as oss://{}/{} (restored by {})".format(
                relative, bucket_name, match[1], match[2]))
            continue
        forbidden = [] if target.is_file() else excluded_dirs(target, base)
        if forbidden:
            print("{}: splitting around {} excluded director{}".format(
                relative, len(forbidden), "y" if len(forbidden) == 1 else "ies"))
        for piece in push_targets(target, forbidden, base):
            name = base if piece == target else base / piece.relative_to(target)
            key = remote_key(str(name), root_key)
            if piece.is_file():
                # sync is directory-to-directory; a single file goes through
                # cp, and the key already carries the filename.
                status |= run_ossutil(
                    ["cp", str(piece), "oss://{}/{}".format(bucket_name, key),
                     "--update", "--parallel", str(args.parallel)])
                continue
            argv = ["sync", str(piece) + "/", "oss://{}/{}/".format(bucket_name, key),
                    "--update", "--jobs", str(args.jobs), "--parallel", str(args.parallel)]
            for pattern in OSSUTIL_EXCLUDE:
                argv += ["--exclude", pattern]
            status |= run_ossutil(argv)
    return status


def cmd_pull(args) -> int:
    """Bring a tree back: into the cache and symlinked, or into the checkout."""
    bucket_name, root_key = parse_prefix(args.prefix)
    status = 0
    for raw in args.paths:
        relative = str(pathlib.Path(os.path.normpath(raw)))
        source = "oss://{}/{}/".format(bucket_name, remote_key(relative, root_key))
        checkout = REPO_ROOT / relative
        if not args.cache:
            checkout.mkdir(parents=True, exist_ok=True)
            status |= run_ossutil(["sync", source, str(checkout) + "/", "--update",
                                   "--jobs", str(args.jobs), "--parallel", str(args.parallel)])
            continue
        # A real directory here would be shadowed by the symlink and silently
        # stranded -- bytes still on the NAS, invisible to every later census.
        if checkout.exists() and not checkout.is_symlink():
            raise AssetError(
                "{} is a real directory; evict it before pulling to cache".format(relative))
        destination = cache_root() / relative
        destination.mkdir(parents=True, exist_ok=True)
        status |= run_ossutil(["sync", source, str(destination) + "/", "--update",
                               "--jobs", str(args.jobs), "--parallel", str(args.parallel)])
        if checkout.is_symlink():
            checkout.unlink()
        checkout.parent.mkdir(parents=True, exist_ok=True)
        checkout.symlink_to(destination, target_is_directory=True)
        print("{} -> {}".format(relative, destination))
    return status


def cmd_verify(args) -> int:
    report = census(args.paths, args.prefix)
    bad = 0
    for name, entry in report.items():
        missing, mismatched = entry["missing"], entry["mismatched"]
        if not missing and not mismatched:
            print("OK   {}: {} files, {} all present remotely".format(
                name, entry["local_files"], human(entry["local_bytes"])))
            continue
        bad += 1
        print("FAIL {}: {} missing, {} size-mismatched".format(
            name, len(missing), len(mismatched)))
        for item in missing[:10]:
            print("       missing  {}".format(item))
        for item, local_size, remote_size in mismatched[:10]:
            print("       size     {} local {} remote {}".format(item, local_size, remote_size))
        if len(missing) > 10 or len(mismatched) > 10:
            print("       ... {} more".format(
                max(len(missing) - 10, 0) + max(len(mismatched) - 10, 0)))
    return 1 if bad else 0


def cmd_evict(args) -> int:
    """Delete local trees, but only the ones a byte census just cleared."""
    report = census(args.paths, args.prefix)
    clear, blocked = [], []
    for name, entry in report.items():
        if entry["linked"]:
            # Already parked; the checkout holds a link, not the bytes.
            # Deleting it would only cost the next reader a re-download.
            print("skipping {}: already cached at {}".format(
                name, entry["materialised_at"]))
            continue
        (clear if not entry["missing"] and not entry["mismatched"] else blocked).append(
            (name, entry))
    for name, entry in blocked:
        print("refusing {}: {} files are not in OSS yet -- push first".format(
            name, len(entry["missing"]) + len(entry["mismatched"])))
    if not args.yes:
        for name, entry in clear:
            print("would delete {} ({}, {} files); re-run with --yes".format(
                name, human(entry["local_bytes"]), entry["local_files"]))
        return 1 if blocked else 0
    failed = 0
    for name, entry in clear:
        target = REPO_ROOT / name
        try:
            # A single-file asset (data/atomic_aistpp.zip) is a legitimate
            # target, and rmtree raises NotADirectoryError on it.
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        except OSError as error:
            # One tree that will not go must not strand the rest of the queue
            # undeleted, which is what an uncaught raise here did.
            failed += 1
            print("failed {}: {}".format(name, error))
            continue
        print("deleted {} ({} freed)".format(name, human(entry["local_bytes"])))
    return 1 if (blocked or failed) else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prefix", default=os.environ.get("OSS_ASSET_PREFIX", DEFAULT_PREFIX))
    parser.add_argument("--jobs", type=int, default=8, help="ossutil files in flight")
    parser.add_argument("--parallel", type=int, default=8, help="ossutil parts per file")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, handler, extra in (
        ("status", cmd_status, False),
        ("push", cmd_push, False),
        ("pull", cmd_pull, False),
        ("verify", cmd_verify, False),
        ("evict", cmd_evict, True),
    ):
        child = sub.add_parser(name, help=handler.__doc__)
        child.add_argument("paths", nargs="+", help="repo-relative paths")
        if extra:
            child.add_argument("--yes", action="store_true",
                               help="actually delete; without it this is a dry run")
        if name == "pull":
            child.add_argument("--cache", action="store_true",
                               help="materialise under {} and symlink the repo path "
                                    "to it, so existing code paths keep working with "
                                    "the bytes off the NAS".format(cache_root()))
        child.set_defaults(handler=handler)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except AssetError as error:
        raise SystemExit("error: {}".format(error))


if __name__ == "__main__":
    raise SystemExit(main())
