#!/usr/bin/env python3
"""Did the fps re-cut actually run, and did it produce fps-correct clips?

This replaces the gate ``tools/refix_wild_fps_clips.sh`` carried until
2026-08-25, which asked whether *any* clip in the census records a
``source_fps``.  That question could not fail on this corpus.  The 2026-08-19
run had already written ``source_fps`` into nineteen clips, so the field was
present before the re-cut started; a run that ingested nothing -- which is
exactly what a missing ``--cfr-cache`` produces, every variable-rate upload
refused with status ``variable_frame_rate`` and zero clips cut -- would have
printed "clips carrying a measured source_fps: 19 of 249" and passed.  A gate
built out of a property the corpus already has is the shape §2 of CLAUDE.md
names: it reads like a check and cannot return no.

Three questions, none of which the corpus answers by construction:

1. **Did this run write manifest rows for the uploads it was launched for?**
   Every ingest row carries ``ingested_at``, so rows older than the run's start
   are last generation's and are counted separately rather than mixed in.  Zero
   fresh rows means the resume gate ate the run -- the 2026-08-19 failure, where
   seven shards printed "1335 of 1335 already recorded" and exited 0.

2. **Did it refuse any upload?**  ``variable_frame_rate`` is a refusal by
   design: the ingest will not cut a file whose ``avg_frame_rate`` and
   ``r_frame_rate`` disagree, because the picture is selected by frame number
   and the sound by seconds, and on such a file those name different spans.
   The refusal means ``--cfr-cache`` was not passed.  It is the whole reason
   this gate exists, so it is fatal rather than a warning.

3. **Do the produced clips come from a file whose two rates agree?**  This is
   the content question, and it is the one the old gate stood in for.  A clip
   cut on 2026-08-19 records ``source_fps`` and no rate pair at all; a clip cut
   after the variable-rate gate records both, either from a constant-rate
   upload or from the constant-rate re-encode it was normalised onto.  So
   "records a rate" is satisfied by the broken generation and "the two rates
   agree" is not.

What it does **not** check: whether the picture and the sound of a given clip
actually cover the same span.  That is a measurement on the bytes -- locate the
clip's frames in the upload by pixel match and its audio by onset correlation,
as was done by hand on 2026-08-24 -- and it is not automated here.  This gate
checks the container property that predicts the defect, not the defect.

Usage::

    check_recut_happened.py --census after.json --ingest-root I \\
        --redo redo_uploads.txt --since 1756000000
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Statuses the ingest returns instead of cutting.  Split by whether the fix is
# a flag or a file: a refusal is the operator's to correct and re-run, a
# failure is the upload's.
REFUSALS = ("variable_frame_rate",)
FAILURES = ("cfr_normalize_failed", "unreadable_frame_rate", "failed",
            "no_usable_span")


def fresh_rows(ingest_root: pathlib.Path, since: float) -> Dict[str, Dict]:
    """Each upload's latest manifest row, split by whether this run wrote it."""
    from tools.ingest_wild_uploads import manifest_rows

    rows = manifest_rows(ingest_root)
    return {upload: row for upload, row in rows.items()
            if float(row.get("ingested_at") or 0.0) >= since}


def check(census: Dict, ingest_root: pathlib.Path, redo: List[str],
          since: float) -> Dict:
    rows = fresh_rows(ingest_root, since)
    redo_set = set(redo)
    touched = sorted(redo_set & set(rows)) if redo_set else sorted(rows)
    statuses = collections.Counter(rows[u].get("status") for u in touched)

    clips = census.get("clips", {})
    produced = {name: row for name, row in clips.items()
                if row.get("produced_now") is not False}
    verdicts = collections.Counter()
    disagreeing, unmeasured = [], []
    for name, row in sorted(produced.items()):
        agree = row.get("rates_agree")
        if agree is True:
            verdicts["agree"] += 1
        elif agree is False:
            verdicts["disagree"] += 1
            disagreeing.append(name)
        else:
            verdicts["never_measured"] += 1
            unmeasured.append(name)

    normalized = sorted(name for name, row in produced.items()
                        if row.get("cfr_normalized_from"))

    reasons = []
    if not touched:
        reasons.append(
            "no upload named by --redo has a manifest row from this run.  The "
            "resume gate skipped everything (check the shard logs for 'already "
            "recorded'); every reading below describes the corpus as it was.")
    for status in REFUSALS:
        if statuses.get(status):
            reasons.append(
                "{} upload(s) were REFUSED with status {}.  The ingest will not "
                "cut a file whose two frame rates disagree; pass --cfr-cache so "
                "it normalises them onto a constant rate first.".format(
                    statuses[status], status))
    if verdicts["disagree"]:
        reasons.append(
            "{} produced clip(s) were cut from a container whose avg_frame_rate "
            "and r_frame_rate disagree.  Those are the picture-and-sound-from-"
            "different-spans clips this run exists to remove.".format(
                verdicts["disagree"]))
    if verdicts["never_measured"]:
        reasons.append(
            "{} produced clip(s) carry no rate pair at all, so they were cut "
            "before 2026-08-25 and this run did not replace them.  They are "
            "not clean; nobody looked.".format(verdicts["never_measured"]))

    return {
        "generated_by": "tools/check_recut_happened.py",
        "since": since,
        "since_readable": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(since)),
        "redo_named": len(redo_set),
        "rows_from_this_run": len(rows),
        "redo_uploads_touched": len(touched),
        "statuses": dict(statuses),
        "refused": sum(statuses.get(s, 0) for s in REFUSALS),
        "failed": sum(statuses.get(s, 0) for s in FAILURES),
        "produced_clips": len(produced),
        "rates_agree": verdicts["agree"],
        "rates_disagree": verdicts["disagree"],
        "rates_never_measured": verdicts["never_measured"],
        "cfr_normalized": len(normalized),
        "disagreeing_examples": disagreeing[:20],
        "unmeasured_examples": unmeasured[:20],
        "reasons": reasons,
        "passed": not reasons,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--census", type=pathlib.Path, required=True,
                        help="clips_after.json from refix_wild_fps_clips_census.py")
    parser.add_argument("--ingest-root", type=pathlib.Path, required=True)
    parser.add_argument("--redo", type=pathlib.Path, default=None,
                        help="the upload list the run was launched for; without "
                             "it every upload ingested since --since counts")
    parser.add_argument("--since", type=float, required=True,
                        help="epoch seconds the run started.  Rows older than "
                             "this are a previous generation's and are counted "
                             "apart -- that separation is the gate")
    parser.add_argument("--output", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)

    census = json.loads(args.census.read_text(encoding="utf-8"))
    redo = []
    if args.redo:
        redo = [line.strip() for line in
                args.redo.read_text(encoding="utf-8").splitlines() if line.strip()]

    report = check(census, args.ingest_root, redo, args.since)

    print("uploads ingested since {}: {} ({} of the {} named by --redo)".format(
        report["since_readable"], report["rows_from_this_run"],
        report["redo_uploads_touched"], report["redo_named"]))
    if report["statuses"]:
        print("  statuses: " + ", ".join(
            "{} x{}".format(k, v) for k, v in sorted(report["statuses"].items())))
    print("produced clips: {} -- rates agree {}, disagree {}, never measured {}".format(
        report["produced_clips"], report["rates_agree"],
        report["rates_disagree"], report["rates_never_measured"]))
    print("  cut from a constant-rate re-encode: {}".format(report["cfr_normalized"]))
    for name in report["disagreeing_examples"]:
        print("  still variable-rate: {}".format(name))
    for name in report["unmeasured_examples"]:
        print("  never measured: {}".format(name))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        staging = args.output.with_suffix(args.output.suffix + ".tmp")
        staging.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        staging.replace(args.output)
        print("wrote", args.output)

    if report["passed"]:
        print("PASS: this run cut clips, and every clip it produced came from a "
              "file whose two frame rates agree.")
        print("      NOT checked: whether any clip's picture and sound cover the "
              "same span.  That is a measurement on the bytes; this is the "
              "container property that predicts it.")
        return 0
    print("REFUSED:")
    for reason in report["reasons"]:
        print("  - {}".format(reason))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
