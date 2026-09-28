#!/usr/bin/env python3
"""Re-key VLM captions onto a different M2 vocabulary, without re-captioning.

A caption describes a *segment* -- the video frames and the pose at its
keyframes.  It says nothing about which prototype the segment landed in.  But
``summarize_subprototypes_llm.py`` groups captions by ``(prototype, genre)``,
reading ``prototype`` straight off each caption row, and that field records the
M2 run the captioner happened to be pointed at.

So the same captions cannot be summarised against a second vocabulary as they
stand, and the failure is silent: every row carries *some* integer, the
summarizer groups by it happily, and the resulting sub-prototypes describe cells
of a clustering that the release was not built from.  Nothing downstream
compares the two.

This re-keys instead of re-captioning, which is the whole point: the AIST++
caption pass is ~12,946 segments of Qwen3-VL at ~0.4 s each per card.  It is
sound only if the two runs segmented identically, and that is checked rather
than asserted: a caption is kept only when the target labels hold ONE value
across its whole ``[start, end)`` span.

An earlier version claimed this check and did not perform it -- it tested only
that ``start`` fell inside the array, and never read ``end`` at all.  Pointed at
a bundle built from a different segmentation, that let 982 of 4,426 kept rows
(22.2%) span a frame range over which the target label changed, 491 of them
landing on a prototype covering under half the caption's frames, with the report
showing zero problems.  A caption describes its whole span, so a span the target
splits is a caption re-keyed onto a segment it does not describe.  The count
rides in the report as ``span_crosses_a_target_boundary`` and its zero as
``segmentations_agree``, so the claim is now an observation.

Rows whose segment the target vocabulary did not accept are dropped rather than
re-labelled 0.  Acceptance is a train-fitted quantile, so it moves between runs
(0.8407 -> 0.8261 from aist_v1 to the song-disjoint fit), and a dropped segment
carries no prototype at all; writing 0 would file it under the transition token
and invent a cell the clustering never produced.

Usage::

    python3 tools/rekey_captions_to_labels.py \\
        --captions runs/aist_captions_v1/captions.jsonl \\
        --labels data/atomic_aistpp/aist_songsplit_labels \\
        --output runs/aist_captions_v1/captions_songsplit.jsonl \\
        --report runs/aist_captions_songsplit_rekey.json
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SCHEMA_VERSION = "atomicdance-caption-rekey-v1"


class RekeyError(RuntimeError):
    """A re-key that would mislabel a caption is refused rather than written."""


def load_label_rows(labels_dir: pathlib.Path) -> Dict[str, Dict[str, Any]]:
    path = labels_dir / "labels.jsonl"
    if not path.is_file():
        raise RekeyError("missing label manifest: {}".format(path))
    rows: Dict[str, Dict[str, Any]] = {}
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        rows[row["recording_id"]] = row
    if not rows:
        raise RekeyError("{} holds no rows".format(path))
    return rows


def read_captions(path: pathlib.Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise RekeyError("missing captions: {}".format(path))
    rows = []
    for number, line in enumerate(path.open(encoding="utf-8"), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise RekeyError("{}:{}: {}".format(path, number, error)) from error
    return rows


def rekey(captions: Sequence[Mapping[str, Any]], labels_dir: pathlib.Path,
          label_rows: Mapping[str, Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Rewrite ``prototype`` and ``split`` from the target vocabulary."""
    arrays: Dict[str, np.ndarray] = {}
    kept: List[Dict[str, Any]] = []
    counts = collections.Counter()
    unknown_recordings = set()
    changed = 0

    for row in captions:
        recording = str(row["recording_id"])
        start = int(row["start"])
        target = label_rows.get(recording)
        if target is None:
            unknown_recordings.add(recording)
            counts["recording_absent"] += 1
            continue
        if recording not in arrays:
            arrays[recording] = np.load(labels_dir / target["labels_path"])
        labels = arrays[recording]
        end = int(row["end"])
        if start >= len(labels) or end > len(labels):
            counts["segment_outside_labels"] += 1
            continue
        span = labels[start:end]
        prototype = int(labels[start])
        if prototype <= 0:
            counts["not_accepted_by_target"] += 1
            continue
        # The caption describes the whole span.  If the target vocabulary splits
        # that span, the caption belongs to no single one of its segments.
        if span.size == 0 or int(span.min()) != prototype or int(span.max()) != prototype:
            counts["span_crosses_a_target_boundary"] += 1
            continue
        updated = dict(row)
        if int(row.get("prototype", -1)) != prototype:
            changed += 1
        updated["prototype"] = prototype
        updated["split"] = target.get("split", row.get("split"))
        updated["rekeyed_from_label_space"] = row.get("label_space_id") or row.get("prototype_source")
        kept.append(updated)
        counts["kept"] += 1

    if unknown_recordings:
        raise RekeyError(
            "{} caption recording(s) are absent from the target labels, e.g. {}; the two "
            "runs did not segment the same corpus".format(
                len(unknown_recordings), ", ".join(sorted(unknown_recordings)[:3])))

    report = {
        "schema_version": SCHEMA_VERSION,
        "labels": str(labels_dir.resolve()),
        "captions_in": len(captions),
        "captions_out": len(kept),
        "prototype_changed": changed,
        "dropped": {
            "not_accepted_by_target": counts["not_accepted_by_target"],
            "segment_outside_labels": counts["segment_outside_labels"],
            "span_crosses_a_target_boundary": counts["span_crosses_a_target_boundary"],
        },
        "segmentations_agree": counts["span_crosses_a_target_boundary"] == 0,
        "note": ("captions describe segments, not clusters; only the grouping key is "
                 "rewritten. Segments the target vocabulary did not accept are dropped "
                 "rather than filed under the transition token."),
    }
    return kept, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--captions", type=pathlib.Path, required=True)
    parser.add_argument("--labels", type=pathlib.Path, required=True,
                        help="the M2 label bundle whose prototypes the captions should carry")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--report", type=pathlib.Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        captions = read_captions(args.captions)
        rows, report = rekey(captions, args.labels, load_label_rows(args.labels))
    except RekeyError as error:
        print("error: {}".format(error), file=sys.stderr)
        return 1
    if args.output.exists():
        print("error: {} exists; caption files publish into a new path".format(args.output),
              file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    report["output"] = str(args.output.resolve())
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
