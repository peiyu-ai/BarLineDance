#!/usr/bin/env python3
"""Refuse to start a stage that the disk cannot hold, and say so out loud.

On 2026-08-12 the NAS quota filled during a run.  Nothing reported it.  The
first visible symptom was twenty-two alignment reports that existed and were
zero bytes, because ``pathlib.write_text`` opens the file -- creating it -- and
only then fails; the error went to a stderr nobody was reading, and every
consumer downstream saw a file that was present.  It took re-running one report
in the foreground to see ``[Errno 122] Disk quota exceeded``.

Two things follow, and this tool is the first:

* **a stage checks before it writes.**  A stage that cannot finish should not
  start, and should say which mount and how short it is;
* **the check is by writing, not by ``df``.**  This is a quota, not a full
  filesystem: ``df`` on the NAS reports 10 PB free while a 1 KB write returns
  ``Errno 122``.  Anything reading ``df`` would have waved that day through.

Usage::

    python3 tools/check_disk_headroom.py --need-gb 40
    python3 tools/check_disk_headroom.py --path /dev/shm --need-gb 5 --quiet
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[1]


def writable(path: pathlib.Path) -> tuple:
    """(ok, detail) for an actual write of a megabyte at ``path``.

    A megabyte rather than a byte: a quota with a little slack left will accept
    a token write and then fail on the first real artifact, which is precisely
    the false reassurance that has to be avoided here.
    """
    try:
        handle, name = tempfile.mkstemp(dir=str(path), prefix=".headroom-")
        staged = pathlib.Path(name)
    except OSError as error:
        return False, "cannot create a file: {}".format(error)
    try:
        with os.fdopen(handle, "wb") as sink:
            sink.write(b"\0" * (1 << 20))
            sink.flush()
            os.fsync(sink.fileno())
        return True, "1 MB write succeeded"
    except OSError as error:
        return False, "{}".format(error)
    finally:
        staged.unlink(missing_ok=True)


def free_gb(path: pathlib.Path) -> float:
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize / 2**30


def probe(path: pathlib.Path, gigabytes: float) -> tuple:
    """Reserve ``gigabytes`` by actually writing it, then give it back.

    There is no cheap way to ask this mount how much room is left: ``statvfs``
    answers for the filesystem (10 PB) and the limit that actually bites is a
    directory quota it knows nothing about.  A gate built on that number can
    never fire, and a gate that can never fire is the thing this repo has twice
    written down as worse than no gate at all -- it reads as "checked".

    So the probe writes.  At 571 MB/s measured here a 2 GB probe costs about
    four seconds, which is nothing against a stage that runs for hours and
    everything against one that dies half way through with a corrupt artifact.
    """
    chunk = b"\0" * (1 << 20)
    written = 0
    target = int(gigabytes * (1 << 30))
    try:
        handle, name = tempfile.mkstemp(dir=str(path), prefix=".headroom-")
        staged = pathlib.Path(name)
    except OSError as error:
        return False, 0.0, "cannot create a file: {}".format(error)
    try:
        with os.fdopen(handle, "wb") as sink:
            while written < target:
                sink.write(chunk)
                written += len(chunk)
            sink.flush()
            os.fsync(sink.fileno())
        return True, written / 2**30, "wrote {:.1f} GB".format(written / 2**30)
    except OSError as error:
        return False, written / 2**30, "{} after {:.2f} GB".format(error, written / 2**30)
    finally:
        staged.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=pathlib.Path, default=REPO,
                        help="mount to check; defaults to the checkout")
    parser.add_argument("--probe-gb", type=float, default=2.0,
                        help="how much room to prove is there, by writing it")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    path = args.path
    path.mkdir(parents=True, exist_ok=True)
    ok, detail = writable(path)
    if not ok:
        print("DISK BLOCKED {}: {}".format(path, detail), file=sys.stderr)
        print("  statvfs claims {:.0f} GB free, which is why it is not the test: "
              "the limit here is a directory quota, not a full filesystem.".format(
                  free_gb(path)), file=sys.stderr)
        print("  Free space first -- `python3 tools/oss_assets.py status <tree>` "
              "shows what is already safe in OSS and therefore evictable.",
              file=sys.stderr)
        return 2

    if args.probe_gb > 0:
        ok, got, detail = probe(path, args.probe_gb)
        if not ok:
            print("DISK LOW {}: wanted {:.1f} GB, {}".format(
                path, args.probe_gb, detail), file=sys.stderr)
            print("  Free space first -- `python3 tools/oss_assets.py status <tree>` "
                  "shows what is already safe in OSS and therefore evictable.",
                  file=sys.stderr)
            return 1
        if not args.quiet:
            print("disk ok: {} took a {:.1f} GB write".format(path, got))
        return 0

    if not args.quiet:
        print("disk ok: {} writable".format(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
