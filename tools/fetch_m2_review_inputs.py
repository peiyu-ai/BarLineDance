#!/usr/bin/env python3
"""Pull only the objects the prototype renderers actually open.

Not ``fetch_dir``: the labels tree is 4,001 objects and the stage-E performance
bundle is 40,404, and this pod holds neither.  What the renderers read is a
strict subset -- every label array (they decide the grouping) but only the
motion of the recordings a card ends up drawing.  So the labels tree is mirrored
whole and the bundle is mirrored lazily, by name, from ``sequences.jsonl``.

Listing is avoided on purpose: ``asset_io.list_prefix`` shells out to ossutil,
which reads ~/.ossutilconfig, which a rebuilt pod does not have.  Every path
here is named by a manifest instead, so the SDK path (STS token) is enough.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import asset_io  # noqa: E402


def fetch(key: str, target: pathlib.Path) -> int:
    if target.is_file() and target.stat().st_size > 0:
        return target.stat().st_size
    payload = asset_io.read_bytes(key)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return len(payload)


def mirror(pairs, jobs: int = 32) -> int:
    total = 0
    with futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        pending = [pool.submit(fetch, key, target) for key, target in pairs]
        for done, future in enumerate(futures.as_completed(pending), 1):
            total += future.result()
            if done % 250 == 0:
                print("    {}/{}  {:.1f} MB".format(done, len(pending), total / 1e6),
                      flush=True)
    return total


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="clean5b5",
                        help="the M2 run whose labels are reviewed")
    parser.add_argument("--bundle-tag", default="wild_v4",
                        help="the stage-E tree the labels resolve against; separate "
                             "from --tag for the reason run_wild_stage_g_m2_oss.py "
                             "gives -- clean5 replaced its M1 and nothing else")
    parser.add_argument("--scratch", type=pathlib.Path,
                        default=pathlib.Path("/dev/shm/atomicdance-m2-review"),
                        help="staging root; must not be on the quota'd NAS")
    parser.add_argument("--jobs", type=int, default=32)
    args = parser.parse_args()

    global LABELS_KEY, BUNDLE_KEY
    LABELS_KEY = "data/wild3d/{}_labels".format(args.tag)
    BUNDLE_KEY = "data/wild3d/{}_performance".format(args.bundle_tag)
    labels_dir = args.scratch / "labels"
    bundle_dir = args.scratch / "bundle"
    print("labels {}\nbundle {}".format(LABELS_KEY, BUNDLE_KEY), flush=True)

    print("labels manifest", flush=True)
    fetch(LABELS_KEY + "/labels.jsonl", labels_dir / "labels.jsonl")
    rows = [json.loads(line) for line
            in (labels_dir / "labels.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    print("  {} label rows".format(len(rows)), flush=True)

    pairs = []
    for row in rows:
        for field in ("labels_path", "label_valid_mask_path"):
            rel = row.get(field)
            if rel:
                pairs.append((LABELS_KEY + "/" + rel, labels_dir / rel))
    seen, unique = set(), []
    for key, target in pairs:
        if key not in seen:
            seen.add(key)
            unique.append((key, target))
    print("  {} label arrays".format(len(unique)), flush=True)
    print("  {:.1f} MB".format(mirror(unique, args.jobs) / 1e6), flush=True)

    print("bundle manifest", flush=True)
    fetch(BUNDLE_KEY + "/sequences.jsonl", bundle_dir / "sequences.jsonl")
    seq = [json.loads(line) for line
           in (bundle_dir / "sequences.jsonl").read_text(encoding="utf-8").splitlines()
           if line.strip()]
    print("  {} sequence rows".format(len(seq)), flush=True)

    # Only the recordings that carry labels can ever be drawn.
    wanted = {row["recording_id"] for row in rows}
    from tools.cluster_atomics_tmr import build_row_index, resolve_row
    index = build_row_index({row["recording_id"]: row for row in seq})
    by_id = {row["recording_id"]: row for row in seq}
    motion = []
    missing = 0
    for recording in sorted(wanted):
        row = resolve_row(index, recording) or by_id.get(recording)
        if row is None:
            missing += 1
            continue
        motion.append((BUNDLE_KEY + "/" + row["motion_path"],
                       bundle_dir / row["motion_path"]))
    print("  {} motion arrays, {} labelled recordings unresolved".format(
        len(motion), missing), flush=True)
    print("  {:.1f} MB".format(mirror(motion, args.jobs) / 1e6), flush=True)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
