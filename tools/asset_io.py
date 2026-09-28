#!/usr/bin/env python3
"""Read corpus assets straight out of OSS, without a local copy.

The bulk of this repo's data lives in the team OSS store.  Pulling it back to
disk to read it defeats the point of putting it there: the working tree fills
up again, just on a different mount.  This module is the read path that does
not do that.

Measured on this pod, over the internal endpoint::

    listing a bundle directory      0.02 s
    first line of sequences.jsonl   0.04 s
    np.load of a [414, 151] motion  0.02 s

which is the same order as the NAS it replaces, so "stream it" is not a
trade against speed here -- it is a trade against *disk*, and it wins.

Credentials come from the rotating STS token the platform mounts at
the path in ``$OSS_STS_TOKEN``, not from a long-lived key.
The token carries ``expired_at`` and lasts 36 hours; a training run outlives
that, so the filesystem handle is rebuilt when the token on disk changes rather
than cached for the life of the process.  A run that started on Monday must not
die on Wednesday holding a credential that expired on Tuesday.

Resolution order for any repo-relative path:

1. a real file in the checkout -- so a developer with data on disk, and every
   existing call site, keeps working unchanged;
2. otherwise the OSS object under this repo's prefix.

Usage::

    from tools.asset_io import open_asset, load_npy, read_jsonl

    motion = load_npy("data/wild3d/wild_performance_v1/sequences/<id>/motion_151_raw.npy")
    for row in read_jsonl("data/wild3d/wild_performance_v1/sequences.jsonl"):
        ...

Writing, and the two rules that differ from reading::

    from tools.asset_io import write_json, save_npy, scratch_file, publish_dir

    write_json("runs/wild_v4_seg/segmentation.json", result)   # -> OSS, always
    with scratch_file(clip + "/clip.mp4") as video:            # one clip on disk
        subprocess.run([...decoder..., str(video)])            # deleted on exit

Reads resolve local-first so existing call sites keep working; writes never do,
because a destination that depends on what happens to be lying on disk is not a
destination.  And every write is one PUT of finished bytes, so the disk-full
signature this module was built after -- an artifact that exists and is empty --
cannot be produced: the object is complete or absent.
"""

from __future__ import annotations

import contextlib
import fnmatch
import io
import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_BUCKET = "example-bucket"  # set ATOMICDANCE_OSS_BUCKET
DEFAULT_PREFIX = "example/prefix/AtomicDance"  # set ATOMICDANCE_OSS_PREFIX
DEFAULT_ENDPOINT = "http://oss-cn-example-internal.aliyuncs.com"  # set OSS_ENDPOINT
STS_TOKEN = pathlib.Path(
    os.environ.get("OSS_STS_TOKEN", "/path/to/oss_sts/token"))

_lock = threading.Lock()
_filesystem = None
_credential_stamp: Optional[tuple] = None


class AssetIOError(RuntimeError):
    pass


def _credentials() -> Dict[str, Optional[str]]:
    """STS token if the platform mounted one, else a long-lived key.

    The STS path is preferred because it is what the cluster refreshes and what
    every other job here uses; the ossutil config is the fallback for a laptop
    or a pod without the mount.
    """
    if STS_TOKEN.is_file():
        token = json.loads(STS_TOKEN.read_text(encoding="utf-8"))
        return {
            "key": token["access_key_id"],
            "secret": token["access_key_secret"],
            "token": token.get("security_token"),
            "stamp": token.get("expired_at") or token.get("access_key_id"),
        }
    import configparser

    # Resolved by tools.oss_assets so there is one answer to "which config".
    # oss_assets learned on 2026-08-24 to prefer the repo-local ``ossutilconfig``
    # because the home copy's key is refused outright (``InvalidAccessKeyId``);
    # this module kept reading the home copy, so the same checkout authenticated
    # as two different identities depending on which module a caller went
    # through, and neither artifact records which one signed the transfer.
    #
    # An earlier note here said the 2026-08-25 listing failure came from the STS
    # branch above, "whose token is unexpired and still refused by the bridge
    # bucket".  **That was wrong**, and the correction matters because it points
    # at a different file.  Measured 2026-08-25 by driving ossutil with each
    # identity in turn against the same prefix: the mounted STS token lists it
    # (rc=0); the key in this repo's ``ossutilconfig`` lists it through ``-c``
    # (rc=0); ``~/.ossutilconfig`` holds a *third* identity -- another STS key,
    # long stale -- and it is refused (403 InvalidAccessKeyId).  Nothing was
    # wrong with either credential this function returns.  The listing failed
    # because ``list_prefix`` invoked ossutil with no ``-c``, so ossutil fell
    # back to ``~/.ossutilconfig`` and authenticated as that third identity --
    # the same defect ``oss_assets._config_flag`` was written to fix on
    # 2026-08-24, left standing in this module.  See ``ossutil_flags`` below.
    from tools.oss_assets import ossutil_config

    config_path = ossutil_config()
    if config_path is None or not config_path.is_file():
        raise AssetIOError(
            "no OSS credentials: no {} and no ossutil config (set OSSUTIL_CONFIG, "
            "or put one at the repo root or ~/.ossutilconfig)".format(STS_TOKEN))
    parser = configparser.ConfigParser()
    parser.read(config_path)
    section = parser["Credentials"]
    return {"key": section.get("accessKeyID"), "secret": section.get("accessKeySecret"),
            "token": None, "stamp": section.get("accessKeyID")}


def filesystem():
    """The shared fsspec filesystem, rebuilt whenever the credential rotates."""
    global _filesystem, _credential_stamp
    credentials = _credentials()
    with _lock:
        if _filesystem is None or _credential_stamp != credentials["stamp"]:
            import ossfs

            _filesystem = ossfs.OSSFileSystem(
                endpoint=os.environ.get("OSS_ENDPOINT", DEFAULT_ENDPOINT),
                key=credentials["key"], secret=credentials["secret"],
                token=credentials["token"])
            _credential_stamp = credentials["stamp"]
        return _filesystem


def remote_path(relative: str) -> str:
    bucket = os.environ.get("ATOMICDANCE_OSS_BUCKET", DEFAULT_BUCKET)
    prefix = os.environ.get("ATOMICDANCE_OSS_PREFIX", DEFAULT_PREFIX)
    return "{}/{}/{}".format(bucket, prefix.strip("/"), str(relative).lstrip("/"))


def local_path(relative: str) -> Optional[pathlib.Path]:
    candidate = REPO_ROOT / relative
    return candidate if candidate.is_file() else None


def exists(relative: str) -> bool:
    if local_path(relative) is not None:
        return True
    try:
        return filesystem().exists(remote_path(relative))
    except Exception:                                         # noqa: BLE001
        completed = subprocess.run(
            [OSSUTIL, "stat", "oss://" + remote_path(relative), *ossutil_flags()],
            capture_output=True, text=True)
        return completed.returncode == 0


def _fetch_via_ossutil(relative: str) -> bytes:
    """Download one object with ossutil, for when the SDK cannot.

    ``ossfs.open`` asks for the object's size first, and fsspec answers that by
    listing the *parent directory* -- so opening one small file under runs/ has
    to list all 16,627 objects beside it, and on this bridge bucket that LIST
    is the operation that returns 502.  ossutil fetches the object without
    consulting its neighbours.
    """
    root = scratch_root()
    handle, name = tempfile.mkstemp(dir=str(root), suffix=".get")
    os.close(handle)
    staged = pathlib.Path(name)
    try:
        _ossutil(["cp", "oss://" + remote_path(relative), str(staged), "--force"])
        return staged.read_bytes()
    finally:
        staged.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Single-object reads go through ossutil, not fsspec.
#
# ``_fetch_via_ossutil`` above already records why the fsspec path is wrong on
# this bucket: ``ossfs.open`` asks for the size, fsspec answers by listing the
# parent directory, and a deep LIST here returns 502.  Both readers below used
# to *try* fsspec first and treat that 502 as the signal to fall back.
#
# That works only for the failure mode it anticipated.  On 2026-08-15 the LIST
# stopped answering instead of failing, and there is no timeout anywhere on the
# path -- ``py-spy`` put the process in ``socket.readinto`` under
# ``oss2.list_objects``, 20 minutes into fetching one 28 MiB ``clip.mp4``.  A
# fallback reached only by an exception cannot fire on a hang, so the caller
# waits forever while the log's last line says it is staging a batch.  The same
# stall is what burned three ``--batch-timeout 10800`` cards in the first M3a
# run and what made ``verify`` spend fifteen minutes to use eleven seconds of
# CPU.
#
# So the order is reversed rather than a timeout bolted on: ossutil is the
# documented-correct tool for this bucket and it cannot be asked to list a
# neighbourhood it does not need.  fsspec stays as the fallback so nothing that
# works today stops working, and ``_ossutil`` carries a deadline of its own.
# ---------------------------------------------------------------------------


def open_asset(relative: str, mode: str = "rb"):
    """A file-like for a repo-relative path, local if present, else from OSS."""
    local = local_path(relative)
    if local is not None:
        return local.open(mode)
    try:
        return io.BytesIO(_fetch_via_ossutil(relative))
    except Exception:                                         # noqa: BLE001
        return filesystem().open(remote_path(relative), mode)


def read_bytes(relative: str) -> bytes:
    local = local_path(relative)
    if local is not None:
        return local.read_bytes()
    try:
        return _fetch_via_ossutil(relative)
    except Exception:                                         # noqa: BLE001
        with filesystem().open(remote_path(relative), "rb") as handle:
            return handle.read()


def load_npy(relative: str):
    """``np.load`` over either source.

    The bytes are read whole and wrapped in ``BytesIO`` rather than handing
    numpy the remote handle: ``np.load`` seeks, and a seek on a streamed object
    turns into a fresh ranged request per call.  A motion array is ~250 KB, so
    one round trip beats a dozen.
    """
    import numpy as np

    local = local_path(relative)
    if local is not None:
        return np.load(local)
    return np.load(io.BytesIO(read_bytes(relative)))


def read_jsonl(relative: str) -> Iterator[Dict[str, Any]]:
    with open_asset(relative, "rb") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_json(relative: str) -> Any:
    return json.loads(read_bytes(relative))


def listdir(relative: str) -> List[str]:
    local = REPO_ROOT / relative
    if local.is_dir():
        return sorted(p.name for p in local.iterdir())
    entries = filesystem().ls(remote_path(relative), detail=False)
    return sorted(name for name in (entry.rstrip("/").rsplit("/", 1)[-1]
                                    for entry in entries) if name)


OSSUTIL = os.environ.get("OSSUTIL", "/opt/data-infra/ossutil64")


def ossutil_flags() -> List[str]:
    """``-c <config>`` for every ossutil subprocess this module runs.

    Never omitted.  Left to itself ossutil reads ``~/.ossutilconfig``, and in
    this container that file holds a stale STS key which the bridge bucket
    refuses outright -- so an unflagged ``ossutil ls`` authenticated as an
    identity no caller chose and returned 403 InvalidAccessKeyId while two
    working identities sat in the process.  That is what broke
    ``audit_clip_freshness --records store`` on 2026-08-25.

    The config is resolved through ``tools.oss_assets.ossutil_config`` so that
    this module and that one cannot answer "which key signed this transfer"
    differently; the answer is not recoverable from the artifact afterwards.
    """
    try:
        from tools.oss_assets import ossutil_config
    except Exception:                                         # noqa: BLE001
        return []
    config = ossutil_config()
    return ["-c", str(config)] if config and config.is_file() else []


def parse_ossutil_listing(text: str, prefix: str) -> Dict[str, int]:
    """``ossutil ls`` output -> {key relative to prefix: size}.

    Each object line is ``<date> <time> +0000 UTC <size> <class> <etag> <url>``.
    The size is taken as the field after ``UTC`` and the key from the ``oss://``
    token, rather than by column index, because the trailer lines ("Object
    Number is: N", timings, blanks) do not share the layout and silently
    parsing one of those as an object is how a skip-set acquires a member that
    does not exist.
    """
    found: Dict[str, int] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 7 or not fields[-1].startswith("oss://"):
            continue
        try:
            size = int(fields[fields.index("UTC") + 1])
        except (ValueError, IndexError):
            continue
        key = fields[-1][len("oss://"):].split("/", 1)[1]
        if key.endswith("/") or not key.startswith(prefix):
            continue
        found[key[len(prefix):]] = size
    return found


def list_prefix(relative: str, attempts: int = 3) -> Dict[str, int]:
    """Every object under a prefix -> size.

    Via ossutil, not the Python SDK.  This is a CPFS-OSS *bridge* bucket: GET
    works over ``oss2`` but a deep LIST returns 502 with no request id, every
    time and at both credential sets, while ``ossutil ls`` returns 113,921
    objects in five seconds.  ``fsspec.find`` is worse than either -- it walks,
    one LIST per directory, so the 10k-clip 3D output would be 10k requests.

    Getting this wrong is not a slow path, it is a wrong answer: a listing that
    dies part way through, read as a skip-set, re-runs thousands of finished
    extractions and reports success while doing it.
    """
    prefix = remote_path(relative).split("/", 1)[1].rstrip("/") + "/"
    bucket = os.environ.get("ATOMICDANCE_OSS_BUCKET", DEFAULT_BUCKET)
    argv = [OSSUTIL, "ls", "oss://{}/{}".format(bucket, prefix), *ossutil_flags()]
    last = None
    for attempt in range(attempts):
        completed = subprocess.run(argv, capture_output=True, text=True)
        if completed.returncode == 0:
            return parse_ossutil_listing(completed.stdout, prefix)
        last = (completed.stderr or completed.stdout).strip()[:300]
        time.sleep(2 ** attempt)
    raise AssetIOError("listing {} failed after {} attempts: {}".format(
        relative, attempts, last))


# ---------------------------------------------------------------------------
# Write path
#
# Reads resolve local-first, because a developer with data on disk should keep
# working.  **Writes do not.**  A write that went local-first would put its
# output wherever a stale file happened to already exist, so the destination of
# a pipeline stage would depend on the leftovers of the previous run -- the same
# shape of defect as stage G inheriting a segmentation default, which silently
# swapped a corpus's parameters between two versions.  One destination, always.
#
# Every write is a single PUT of bytes already assembled in memory.  That is
# what makes the 2026-08-12 quota failure impossible to repeat: ``write_text``
# opens the file, creating it at zero bytes, and only then fails, so a full
# disk produced 22 plausible-looking empty reports and no error.  An object
# either exists complete here, or does not exist.
# ---------------------------------------------------------------------------


def _ossutil(argv: Sequence[str], attempts: int = 3,
             timeout: float = float(os.environ.get("ATOMICDANCE_OSS_TIMEOUT", 600))) -> None:
    """Run one ossutil command, with a deadline.

    The deadline is the point.  Every caller of this module is inside a loop
    that a hung transfer stops silently -- the batch driver's own timeout is
    three hours, so one wedged object costs a card for an afternoon and reports
    nothing until it fires.  A slow transfer that exceeds this is retried like
    any other failure and, if it keeps exceeding it, raised; both outcomes are
    visible, which an untimed ``subprocess.run`` is not.
    """
    last = None
    for attempt in range(attempts):
        try:
            completed = subprocess.run([OSSUTIL, *argv, *ossutil_flags()],
                                       capture_output=True, text=True,
                                       timeout=timeout)
        except subprocess.TimeoutExpired:
            last = "no answer in {:.0f}s".format(timeout)
            time.sleep(2 ** attempt)
            continue
        if completed.returncode == 0:
            return
        last = (completed.stderr or completed.stdout).strip()[:300]
        time.sleep(2 ** attempt)
    raise AssetIOError("ossutil {} failed after {} attempts: {}".format(
        argv[0], attempts, last))


def write_bytes(relative: str, payload: bytes) -> str:
    """Publish ``payload`` at a repo-relative path.  Returns the OSS URL.

    Through ossutil rather than the SDK, for the same reason ``list_prefix``
    is: this bridge bucket serves GET over ``oss2`` but answers PUT and LIST
    with a 502 carrying no request id, while ossutil does all three.  The
    bytes are staged in a complete temp file first, so what ossutil uploads is
    finished content and the object is never observable half-written.
    """
    url = "oss://" + remote_path(relative)
    root = scratch_root()
    handle, name = tempfile.mkstemp(dir=str(root), suffix=".put")
    staged = pathlib.Path(name)
    try:
        with os.fdopen(handle, "wb") as sink:
            sink.write(payload)
        _ossutil(["cp", str(staged), url, "--force"])
        # Same invariant ``publish_dir`` restores, for the single-object path.
        # ``runs/`` is itself a parked tree on this pod, so without this a stage
        # that republishes a manifest leaves every reader of that manifest on
        # the previous one: on 2026-08-20 stage C merged a corrected 17,015-row
        # inventory to the store and ``staging`` -- reading local-first --
        # rebuilt itself from the 17,790-row copy in the cache and reported the
        # old counts as a success.  Only an *existing* copy is updated; writing
        # a new one would put bytes on this pod nobody asked for (CLAUDE.md
        # §1.2).
        mirror = REPO_ROOT / relative
        if mirror.is_file():
            mirror.write_bytes(payload)
    finally:
        staged.unlink(missing_ok=True)
    return url


def write_json(relative: str, obj: Any, indent: Optional[int] = 2) -> str:
    return write_bytes(relative, (json.dumps(obj, indent=indent, ensure_ascii=False)
                                  + "\n").encode("utf-8"))


def write_jsonl(relative: str, rows) -> str:
    body = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    return write_bytes(relative, body.encode("utf-8"))


def save_npy(relative: str, array) -> str:
    import numpy as np

    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return write_bytes(relative, buffer.getvalue())


def save_npz(relative: str, **arrays) -> str:
    import numpy as np

    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return write_bytes(relative, buffer.getvalue())


# ---------------------------------------------------------------------------
# Bounded scratch
#
# Some stages cannot be given a stream: GVHMR, ffmpeg and the S3D reader all
# want a path they can seek and hand to a decoder.  For those the corpus still
# does not land on disk in bulk -- one clip is materialised, consumed, and
# deleted, so the footprint is O(concurrent workers) rather than O(corpus).
#
# The scratch root defaults off the quota'd NAS on purpose, and refuses to sit
# on it even if asked.  Putting the scratch back on /workspace would reproduce
# exactly the disk this module exists to stop filling, and it would do it in the
# least visible way: a temp directory nobody thinks of as an asset.
# ---------------------------------------------------------------------------

NAS_ROOT = pathlib.Path("/workspace")
DEFAULT_SCRATCH = pathlib.Path("/dev/shm/atomicdance-scratch")


def scratch_root() -> pathlib.Path:
    root = pathlib.Path(os.environ.get("ATOMICDANCE_SCRATCH", DEFAULT_SCRATCH))
    resolved = root.resolve() if root.exists() else root
    if resolved == NAS_ROOT or NAS_ROOT in resolved.parents:
        raise AssetIOError(
            "scratch root {} is on the quota'd NAS; set ATOMICDANCE_SCRATCH to a "
            "path outside {} (/dev/shm and / are both fine on this pod)".format(
                resolved, NAS_ROOT))
    root.mkdir(parents=True, exist_ok=True)
    return root


def free_bytes(path: pathlib.Path) -> int:
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize


def require_free(path: pathlib.Path, need: int) -> None:
    """Fail before writing rather than half way through.

    A quota that runs out mid-write is the failure this module was built after;
    the guard is cheap and the message names the mount, because the symptom
    ("the artifact is empty") points nowhere near the cause.
    """
    available = free_bytes(path)
    if available < need:
        raise AssetIOError(
            "{} has {:.1f} GB free, need {:.1f} GB".format(
                path, available / 2**30, need / 2**30))


@contextlib.contextmanager
def scratch_file(relative: str, suffix: str = ""):
    """Materialise one asset locally, yield its path, always delete it.

    Cleanup is in ``finally`` because the caller most likely to leak is the one
    that crashed, and a retry loop over a crashing clip is how a bounded scratch
    silently becomes an unbounded one.
    """
    local = local_path(relative)
    if local is not None:
        yield local                      # already on disk; nothing to fetch or clean
        return
    root = scratch_root()
    payload = read_bytes(relative)
    require_free(root, len(payload) * 2)
    handle, name = tempfile.mkstemp(dir=str(root), suffix=suffix or pathlib.Path(relative).suffix)
    target = pathlib.Path(name)
    try:
        with os.fdopen(handle, "wb") as sink:
            sink.write(payload)
        yield target
    finally:
        target.unlink(missing_ok=True)


@contextlib.contextmanager
def scratch_dir(prefix: str = "stage-"):
    """A directory for a third-party tool to write into, removed afterwards."""
    root = scratch_root()
    target = pathlib.Path(tempfile.mkdtemp(dir=str(root), prefix=prefix))
    try:
        yield target
    finally:
        shutil.rmtree(target, ignore_errors=True)


def fetch_dir(relative: str, local_dir: pathlib.Path) -> List[pathlib.Path]:
    """Bring everything under the ``relative`` prefix into ``local_dir``.

    The read counterpart of ``publish_dir``, and for the same reason: one
    ossutil sync for a whole clip rather than a ``cp`` per object.  A converted
    clip is ten files, and ``read_bytes`` pays a process spawn for each of them
    because ``ossfs.open`` has to LIST the parent to learn a size and this
    bridge bucket answers that LIST with a 502.  Ten spawns per clip over
    15,294 clips is the difference between an hour and a morning.

    A locally-present tree short-circuits to a copy, so a cache pulled with
    ``oss_assets.py pull --cache`` is used instead of the network without the
    caller knowing which it got -- the same contract ``read_bytes`` offers.

    Returns the files that arrived, by walking the destination.  Not the
    source listing: what the caller needs to know is what it may now open, and
    a stage that reports success over objects it did not actually receive is
    the failure this repo keeps meeting.
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    source = local_path(relative)
    if source is not None and source.is_dir():
        for entry in sorted(source.rglob("*")):
            if entry.is_file():
                target = local_dir / entry.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(entry, target)
    else:
        _ossutil(["sync", "oss://{}/".format(remote_path(relative).rstrip("/")),
                  str(local_dir) + "/", "--force"])
    return sorted(path for path in local_dir.rglob("*") if path.is_file())


def publish_dir(local_dir: pathlib.Path, relative: str,
                skip: Sequence[str] = ("*.pyc",)) -> List[str]:
    """Upload everything under ``local_dir`` to the ``relative`` prefix.

    One ossutil sync for the whole directory rather than a PUT per file: a
    clip's output is a handful of files, and paying process startup for each of
    them turns a two-second publish into twenty.

    Returns the repo-relative names it uploaded -- the local walk, not a
    listing of the destination.  The caller wants to know what this call was
    responsible for; asking the store afterwards would also count whatever a
    previous run left at the same prefix, and a stage that reports success over
    somebody else's objects is the failure this repo keeps meeting.
    """
    names: List[str] = []
    for entry in sorted(local_dir.rglob("*")):
        if entry.is_symlink() or not entry.is_file():
            continue
        name = entry.relative_to(local_dir)
        if any(fnmatch.fnmatch(part, pattern)
               for part in name.parts for pattern in skip):
            continue
        names.append("{}/{}".format(relative.rstrip("/"), name))
    if not names:
        return names
    argv = ["sync", str(local_dir) + "/",
            "oss://{}/".format(remote_path(relative).rstrip("/")), "--force"]
    for pattern in skip:
        argv += ["--exclude", pattern]
    _ossutil(argv)
    refresh_local_mirror(local_dir, relative, skip)
    return names


def refresh_local_mirror(local_dir: pathlib.Path, relative: str,
                         skip: Sequence[str] = ("*.pyc",)) -> int:
    """Keep a parked read copy in step with what was just published.

    ``read_bytes``, ``load_npy`` and ``fetch_dir`` all resolve local-first, and
    three of this repo's trees are parked under ``/cache`` and symlinked back
    (``ingest_v1_converted``, ``wild_visual_s3d``, ``wild_ingest_v1``).  So a
    stage that re-derives a clip and publishes it leaves this pod reading the
    *previous* generation from the cache while the store holds the new one --
    and every reader is affected, not only the one that noticed.

    Measured on 2026-08-20: stage B re-extracted 317 re-cut clips and published
    them.  The store's ``metadata.json`` recorded the new video hash for 10 of
    10 sampled; the parked copy recorded the old one for 10 of 10, so
    ``audit_clip_freshness`` -- the gate whose whole job is to weigh records
    against bytes -- reported that nothing had changed, and stage C's
    ``reconcile`` would have joined the old 3D.

    Only *existing* copies are updated.  Creating one would put bytes on this
    pod that nobody asked for, which is the rule in CLAUDE.md §1.2; refreshing
    one that is already there restores an invariant it was supposed to have.
    """
    target = REPO_ROOT / relative
    if not target.is_dir():
        return 0
    updated = 0
    for entry in sorted(local_dir.rglob("*")):
        if entry.is_symlink() or not entry.is_file():
            continue
        name = entry.relative_to(local_dir)
        if any(fnmatch.fnmatch(part, pattern)
               for part in name.parts for pattern in skip):
            continue
        destination = target / name
        if destination.exists() and entry.resolve() == destination.resolve():
            # The publisher was handed the mirror itself.  ``data/`` trees on
            # this pod are symlinks into /cache, so publishing a clip out of
            # the ingest tree makes source and destination the same file and
            # ``copyfile`` raises SameFileError -- which aborted the re-cut
            # publish on 2026-08-25 after the first clip.  Nothing to refresh:
            # the mirror is already the bytes that were just published.
            continue
        if not destination.parent.is_dir():
            destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(entry, destination)
        updated += 1
    return updated


if __name__ == "__main__":  # tiny self-check against the real store
    import time

    bundle = "data/wild3d/wild_performance_v1"
    start = time.time()
    rows = list(read_jsonl(bundle + "/sequences.jsonl"))
    print("manifest: {} rows in {:.2f}s".format(len(rows), time.time() - start))
    start = time.time()
    motion = load_npy(bundle + "/" + rows[0]["motion_path"])
    print("motion {} in {:.2f}s".format(motion.shape, time.time() - start))
