#!/usr/bin/env python3
"""Write down which clips the ingest corpus currently consists of.

Stage C used to answer this by enumerating ``data/wild_ingest_v1``.  That is
the enumeration CLAUDE.md §1.1 forbids, and this corpus shows why: these
credentials can PUT over a key and cannot delete one, so a clip the fps re-cut
stopped producing leaves its object behind forever and every listing keeps
handing it back.  ``runs/wild_v4_inventory.jsonl`` has 17,790 rows for exactly
that reason, and 775 of them are orphans.

What this writes is the list, once, with the arithmetic that produced it:

    corpus = {stems under the ingest prefix that have a meta.json} - {orphans}

Two counts are reported rather than folded in, because both are things a
reader will otherwise have to rediscover:

* **orphans excluded** -- names no consumer may read.  They are excluded *by
  name* because absence cannot do it: the objects are still there.
* **not in the frozen worklist** -- on 2026-08-20 this is 1,075 stems, and they
  are not a mystery: ``runs/wild_ingest_v1_worklist.json`` records
  ``excluded_dancer_switch: 1075``, clips whose dancer changes partway through,
  which stage B was deliberately never given.  They carry a ``meta.json``, so
  the prefix serves them and this manifest keeps them; they have no 3D, so
  ``reconcile`` drops them later.  Counting them here is what stops the next
  reader from taking 17,790 vs 16,715 for a corruption.

The manifest is a plain ``clips`` list, which ``tools/redo_manifest.py``
already reads, so stage C, stage F and stage B all consume it through the same
parser rather than three.

Usage::

    python3 tools/build_ingest_corpus_manifest.py \\
        --orphans /cache/atomicdance-assets/scratch/c1/refix/orphans.txt \\
        --output runs/wild_v4_corpus_clips.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, List, Set

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools import asset_io                                    # noqa: E402
from tools.run_wild_stage_c_oss import INGEST_ROOT, clip_contents  # noqa: E402

WORKLIST = "runs/wild_ingest_v1_worklist.json"


def build(present: Set[str], orphans: Set[str], worklist: Set[str]) -> Dict[str, object]:
    """The corpus and the two counts that explain its size.

    Kept separate from the listing so it can be tested without a store: what
    this returns is what every later stage will treat as "the corpus", and a
    silent difference between it and the prefix is the thing that has to be
    visible.
    """
    stems = sorted(present - orphans)
    orphans_seen = sorted(orphans & present)
    orphans_absent = sorted(orphans - present)
    outside = set(stems) - worklist
    # The worklist records how many stems it excluded for a mid-clip dancer
    # switch.  That count is over the prefix; this one is over the prefix minus
    # orphans, so the two differ by exactly the orphans that were *also*
    # dancer-switch exclusions.  Reported as a decomposition rather than as a
    # sentence: on 2026-08-20 the first version of this tool guessed the gap was
    # "names created after the worklist was frozen", which was wrong and read
    # like a measurement (CLAUDE.md §2.2).
    outside_orphans = sorted((present - worklist) & orphans)
    return {
        "schema_version": "atomicdance-ingest-corpus-v1",
        "generated_by": "tools/build_ingest_corpus_manifest.py",
        "ingest_root": INGEST_ROOT,
        "counts": {
            "prefix_with_meta_json": len(present),
            "orphans_named": len(orphans),
            "orphans_excluded": len(orphans_seen),
            "orphans_named_but_not_under_the_prefix": len(orphans_absent),
            "corpus": len(stems),
            "not_in_frozen_worklist": len(outside),
            "orphans_outside_the_worklist": len(outside_orphans),
            "worklist": len(worklist),
        },
        "reading": "corpus = prefix stems with a meta.json, minus orphans by "
                   "name.  not_in_frozen_worklist + orphans_outside_the_"
                   "worklist must equal the worklist's own "
                   "excluded_dancer_switch: those clips exist and are "
                   "inventoried, have no 3D, and are dropped at reconcile.  A "
                   "residue means names entered the prefix that neither the "
                   "worklist nor the orphan list accounts for.",
        "clips": stems,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--orphans", required=True,
                        help="one stem per line; names no consumer may read")
    parser.add_argument("--worklist", default=WORKLIST,
                        help="the frozen ingest worklist, read only to explain "
                             "the size difference")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    contents = clip_contents(INGEST_ROOT)
    present = {stem for stem, files in contents.items() if "meta.json" in files}
    orphans = {line.strip() for line
               in pathlib.Path(args.orphans).read_text(encoding="utf-8").splitlines()
               if line.strip()}
    worklist = {row["clip"] for row in asset_io.read_json(args.worklist)["clips"]}

    report = build(present, orphans, worklist)
    target = pathlib.Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    counts = report["counts"]
    for name in ("prefix_with_meta_json", "orphans_excluded", "corpus",
                 "not_in_frozen_worklist", "orphans_outside_the_worklist"):
        print("  {:34s} {}".format(name, counts[name]))

    # The gate: two independently produced numbers that must add up.  It can
    # fail, and a residue would name a real thing -- stems in the prefix that
    # neither the worklist's exclusions nor the orphan list explains.
    expected = asset_io.read_json(args.worklist).get("excluded_dancer_switch")
    residue = None
    if expected is not None:
        residue = (counts["not_in_frozen_worklist"]
                   + counts["orphans_outside_the_worklist"] - expected)
        report["counts"]["unexplained_outside_the_worklist"] = residue
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                          encoding="utf-8")
        print("  {:34s} {} + {} vs excluded_dancer_switch {} -> residue {}".format(
            "worklist reconciliation",
            counts["not_in_frozen_worklist"],
            counts["orphans_outside_the_worklist"], expected, residue))
    print("wrote {}".format(target))
    if residue:
        print("MANIFEST_WARN {} stem(s) under the prefix are explained by "
              "neither the worklist's dancer-switch exclusions nor the orphan "
              "list".format(residue), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
