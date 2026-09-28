#!/usr/bin/env python3
"""Stage G / M3b over the OSS-resident corpus: sub-prototypes from the captions.

M3a wrote one caption per M2 segment.  This is the other half of the paper's
M3 -- the summarizing LLM forms sub-prototypes out of those captions, and the
re-clusterer reads that grouping back onto the segments -- run against a corpus
that lives entirely in the object store.

Two things this pins, and both are arithmetic rather than taste.

**The pre-split key is the uploader account, and it is read, not predicted.**
The paper pre-splits each prototype by dance genre, which on AIST++ is a
recorded field.  TikTok has no such field, and a VLM asked to supply one
answered at chance (0.083 against 0.10), so predicting it would scatter one
movement across ten groups.  ``runs/<tag>_group_keys.json`` carries the
choreographer account instead: recorded metadata, same epistemic status as
AIST's field, and arguably tighter -- one choreographer's style is narrower
than "hip-hop".  Measured on wild_v4 it resolves for **13,783 of 13,783**
recordings into 25 groups, so the failure the AIST driver warns about (every
recording landing in group "?", the pre-split quietly vanishing from the LLM
stage while the re-clusterer still applies it) does not happen here.

**``--target-size`` and ``--genre-split`` are not independent, and the number
in the plan was measured without the second one.**  ``recluster_atomics_ingroup``
takes ``max(1, round(len(cell) / target_size))`` sub-prototypes per *cell*,
where a cell is a prototype when the pre-split is off and a (prototype, group)
pair when it is on:

    pre-split off   100 cells    mean 1,947 segments   -> round(1947/227) = 9
                                                       -> ~900 classes
    pre-split on    2,500 cells  mean    78 segments   -> round(78/227)  = 0
                                                       -> max(1, 0) = 1 each
                                                       -> ~2,500 classes

The ~857 in the plan (§7.9, and the 2026-08-14 worklog) is the first row.  Turn
the pre-split on and the vocabulary's size stops being set by ``--target-size``
at all and starts being set by how many accounts there are -- which is the
defect ``min_cell_size``'s comment already records on AIST++, where 700 of 849
classes came from the floor and 128 classes held a single sample.  So the two
flags are exposed together, the projected class count is printed *before* the
LLM is booked, and neither number is inherited from a run that used the other
setting.

On the ``--subprototypes`` path ``--target-size`` does not bind at all: the
count is ``max(len(slots), 1)`` from the LLM's own grouping.  It still governs
the keyframe fallback for cells the grouping does not cover, which is why it is
passed rather than dropped.

Bundle tag vs run tag (added 2026-08-21)
----------------------------------------
``--bundle-tag`` moves two inputs -- ``runs/<tag>_group_keys.json`` and the
stage-E performance bundle -- onto another tag, while the M2 labels, the
embedding cache, the captions and *every output* stay on ``--tag``.  M2 and M3a
grew the same flag on 2026-08-20 for the same reason: clean5 replaced M1 with a
music-beat grid and nothing else, so the 3D, the normalizer fit and the account
map are wild_v4's and re-publishing them under a new tag would write hours of
byte-identical objects into a store that cannot delete them.  Without the flag
this stage does not merely inherit a default, it 404s: measured 2026-08-20,
``runs/clean5b5_group_keys.json`` does not exist and
``data/wild3d/clean5b5_performance/`` holds 0 objects against wild_v4's 41,403.

The pairing it creates is checked, not printed and hoped for.
``recluster_atomics_ingroup.build`` *skips* a label row whose recording has no
bundle row rather than failing, so the wrong bundle re-clusters the
intersection and reports a clean run over it; ``check_bundle_covers_labels``
reads the bundle's ``sequences.jsonl`` (one object, 13,467 rows, 0.9 s) and
refuses before anything is staged or booked.

The two counts quoted above -- 13,783 recordings, 25 accounts -- are wild_v4's.
On clean5b5 the corpus is 1,999 clips over 5 accounts, so the cell arithmetic
in the ``--target-size`` table has to be re-read from ``project`` rather than
carried over.
"""

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
from typing import Any, Dict, List

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import asset_io                                    # noqa: E402
from tools.cluster_atomics_tmr import build_row_index, resolve_row  # noqa: E402
from tools.run_wild_stage_c_oss import repo_key               # noqa: E402
from tools.run_wild_stage_e_oss import stage_e_keys           # noqa: E402
from tools.run_wild_stage_g_m2_oss import m2_keys             # noqa: E402
from tools.run_wild_stage_g_m3a_oss import (                  # noqa: E402
    clustered_spans, fetch_tree_once, m3a_keys, published_keys, stage_segmentation)

MODEL = "third_party/QwenVL/Qwen3-VL-30B-A3B-Instruct/main"
QWEN_PY = "third_party/QwenVL/.venv-qwen3vl/bin/python"


def interpreter() -> str:
    """The environment this model was validated in, as M3a already uses."""
    return str(REPO / QWEN_PY) if (REPO / QWEN_PY).is_file() else sys.executable


def m3b_keys(tag: str, bundle_tag: str = None,
             generation: str = "") -> Dict[str, str]:
    """Where M3b reads and writes.  Only ``group_keys`` follows ``bundle_tag``:
    the account map is a property of the ingest, not of this run's segmentation,
    and every output below is this run's and must never be written under the
    tag it borrowed inputs from.

    ``generation`` names a caption generation -- M3a's ``--parts-suffix`` -- and
    keys **every output**, not only the captions it reads.  Two generations of
    captions produce two different vocabularies from the same segments, and a
    shared output name in a store that overwrites and cannot delete means the
    second run silently replaces the first while `` _ingroup_llm`` still says
    only which *arm* it was.  This is the same asymmetry that forced
    ``bundle_nollm``; it is worth stating twice because it is worth paying for
    only once.
    """
    def stamp(name: str) -> str:
        return name if not generation else "{}_{}".format(name, generation)

    return {
        "captions": m3a_keys(tag, generation)["captions"],
        "group_keys": "runs/{}_group_keys.json".format(bundle_tag or tag),
        # Every output is keyed by the pre-split, not just the checkpoints.  The
        # two configurations produce different vocabularies from the same
        # captions -- 20,222 classes at 9.38 samples with the pre-split against
        # the paper's 730 at 31.8 -- so one has to be comparable against the
        # other, and a shared key means running the second erases the first in a
        # store that cannot restore it.
        "subprototypes": stamp("runs/{}_subprototypes_llm".format(tag)) + ".json",
        "subprototypes_presplit": stamp("runs/{}_subprototypes_llm_presplit".format(tag)) + ".json",
        "cells": stamp("runs/{}_subprototypes_cells".format(tag)),
        "cells_presplit": stamp("runs/{}_subprototypes_cells_presplit".format(tag)),
        "bundle": stamp("data/wild3d/{}_ingroup_llm".format(tag)),
        "bundle_presplit": stamp("data/wild3d/{}_ingroup_llm_presplit".format(tag)),
        # ``--no-subprototypes`` is Tab. 2's *other* row: the same segments
        # re-clustered on keyframes with no LLM in the loop.  It used to land on
        # the two keys above, which in a store that can overwrite and cannot
        # delete means running it would replace the w/ LLM bundle in place --
        # under a name that still says ``_llm`` -- and nothing downstream could
        # tell which row it was reading.  Separate names make the two rows
        # comparable instead of mutually destructive.
        "bundle_nollm": stamp("data/wild3d/{}_ingroup_nollm".format(tag)),
        "bundle_nollm_presplit": stamp("data/wild3d/{}_ingroup_nollm_presplit".format(tag)),
        "report": stamp("runs/{}_paper_alignment_llm".format(tag)) + ".json",
    }


def check_bundle_covers_labels(labels_dir: pathlib.Path, bundle_tag: str) -> int:
    """Refuse a bundle that does not contain every recording M3b will cluster.

    ``recluster_atomics_ingroup.build`` resolves each label row against the
    bundle and *continues* when it cannot (``row is None``), so a bundle from
    the wrong tag does not raise: it re-clusters whatever the intersection
    happens to be, publishes it, and every count downstream is of that
    intersection.  ``--bundle-tag`` turns that pairing into something a person
    types, so it gets a criterion instead of a convention.

    One object, not the tree: the bundle's ``sequences.jsonl`` is 13,467 rows
    and 0.9 s, which is cheap enough to run before ``project`` -- the command
    that decides whether the LLM is worth booking -- and not only before
    ``recluster``, the one that needs the motion arrays.  Resolution goes
    through ``build_row_index`` / ``resolve_row``, the same aliasing ``build``
    applies (``tiktok:<video>:clipNNN`` against ``<video>__clipNNN``), so a name
    this gate accepts cannot be one that silently skips there.
    """
    key = repo_key(stage_e_keys(bundle_tag)["bundle"], "sequences.jsonl")
    try:
        rows = {row["recording_id"]: row for row in asset_io.read_jsonl(key)}
    except Exception as failure:                              # noqa: BLE001
        # A tag with no stage-E tree is the ordinary mistake this flag exists
        # for -- ``data/wild3d/clean5b5_performance/`` holds 0 objects -- and
        # the store answers it with a transfer error several frames deep.  Say
        # which tag was asked for instead.
        raise SystemExit(
            "cannot read {}: tag '{}' has no stage-E bundle ({}).  Pass "
            "--bundle-tag with the tag that published one, e.g. wild_v4"
            .format(key, bundle_tag, failure))
    index = build_row_index(rows)
    recordings = [json.loads(line)["recording_id"]
                  for line in (labels_dir / "labels.jsonl").open(encoding="utf-8")]
    missing = [name for name in recordings if resolve_row(index, name) is None]
    print("labels resolved in {}: {} of {}".format(
        key, len(recordings) - len(missing), len(recordings)), flush=True)
    if missing:
        raise SystemExit(
            "{} of {} label recording(s) have no row in {} (e.g. {}) -- the "
            "labels and the bundle are different generations".format(
                len(missing), len(recordings), key, ", ".join(missing[:3])))
    return len(recordings)


def stage_inputs(tag: str, root: pathlib.Path, *, want_bundle: bool,
                 want_captions: bool = True,
                 bundle_tag: str = None,
                 generation: str = "") -> Dict[str, pathlib.Path]:
    """Everything M3b reads, pulled to local disk once.

    ``want_captions`` is false for ``project``, which decides the vocabulary
    size from the label array and the pre-split alone.  That is the whole reason
    the projection is a separate subcommand: it answers the question that sets
    ``--target-size`` *before* M3a has finished writing the captions, so the
    constant is chosen against arithmetic rather than against whatever the
    merge happens to produce.
    """
    root.mkdir(parents=True, exist_ok=True)
    bundle_tag = bundle_tag or tag
    keys, staged = m3b_keys(tag, bundle_tag, generation), {}
    print("tag {} | bundle tag {}{}".format(
        tag, bundle_tag,
        "" if bundle_tag == tag else
        "  (labels, captions and every output are this run's; the account map "
        "and the performance bundle are the other tag's)"), flush=True)

    seg = stage_segmentation(tag, root)
    staged["labels"] = seg["labels"]
    staged["cache"] = seg["cache"]
    check_bundle_covers_labels(seg["labels"], bundle_tag)

    if want_captions:
        captions = root / ("captions.jsonl" if not generation
                           else "captions_{}.jsonl".format(generation))
        if not captions.is_file():
            partial = captions.with_name(
                captions.name + ".partial-{}".format(os.getpid()))
            partial.write_bytes(asset_io.read_bytes(keys["captions"]))
            try:
                partial.rename(captions)
            except OSError:
                partial.unlink(missing_ok=True)
        staged["captions"] = captions

    # Both staged names carry the tag they came from.  This root is shared with
    # M3a and across runs, and ``fetch_tree_once`` returns a populated directory
    # untouched, so an unqualified ``bundle/`` or ``group_keys.json`` is a file
    # from whichever tag ran here first -- the same trap the label tree was in
    # until 2026-08-21.
    groups = root / "{}_group_keys.json".format(bundle_tag)
    if not groups.is_file():
        groups.write_bytes(asset_io.read_bytes(keys["group_keys"]))
    staged["group_keys"] = groups

    if want_bundle:
        staged["bundle"] = fetch_tree_once(stage_e_keys(bundle_tag)["bundle"],
                                           root / "{}_bundle".format(bundle_tag))
    return staged


def project_class_count(tag: str, staged: Dict[str, pathlib.Path],
                        target_size: int, genre_split: bool) -> Dict[str, Any]:
    """What the vocabulary size will be, before anything expensive runs.

    Reproduces ``max(1, round(len(cell) / target_size))`` over the real cells
    rather than over an average, because the average is what made the plan's
    projection wrong by 1.8x once already.
    """
    sys.path.insert(0, str(REPO))
    import numpy as np
    from tools.recluster_atomics_ingroup import genre_of, load_group_keys

    groups = load_group_keys(staged["group_keys"]) if genre_split else {}
    spans = clustered_spans(staged["labels"], staged["cache"])

    cells: Dict[tuple, int] = {}
    labels_dir = staged["labels"]
    for line in (labels_dir / "labels.jsonl").open(encoding="utf-8"):
        entry = json.loads(line)
        recording = entry["recording_id"]
        owned = spans.get(recording)
        if not owned:
            continue
        array = np.load(labels_dir / entry["labels_path"])
        key = genre_of(recording, groups) if genre_split else None
        for start, _ in owned:
            cell = (int(array[int(start)]), key)
            cells[cell] = cells.get(cell, 0) + 1

    classes = sum(max(1, int(round(size / float(target_size)))) for size in cells.values())
    segments = sum(cells.values())
    singleton = sum(1 for size in cells.values() if size < target_size)
    return {"cells": len(cells), "segments": segments, "target_size": target_size,
            "genre_split": genre_split, "projected_classes": classes,
            "cells_below_target_size": singleton,
            "mean_samples_per_class": round(segments / max(classes, 1), 2)}


def cmd_cells(args: argparse.Namespace) -> int:
    """How big is one cell, in the unit the summarizing LLM actually consumes?

    The loop is shown *distinct captions* and removes at most ``--subset-cap``
    of them per round, stopping at ``--max-rounds``.  So the round budget has to
    be read against distinct captions per cell, not segments: on AIST++ a cell
    held ~20 segments and 20 rounds was slack, and the same 20 against a cell
    two orders of magnitude larger would group a sliver and hand the rest to the
    field-agreement residue -- a vocabulary that is the fallback rule wearing an
    LLM's name, with ``segments_in_llm_formed_subprototypes`` there to show it.

    Printed before the model is booked, for both pre-split settings, because
    this is the number that decides whether the pre-split is affordable rather
    than merely desirable.
    """
    import statistics

    staged = stage_inputs(args.tag, pathlib.Path(args.root), want_bundle=False,
                          bundle_tag=args.bundle_tag,
                          generation=args.generation)
    from tools.recluster_atomics_ingroup import genre_of, load_group_keys

    rows = []
    with staged["captions"].open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows.append((int(row["prototype"]), str(row["recording_id"]),
                         str(row["caption"])))
    groups = load_group_keys(staged["group_keys"])

    for genre_split in (False, True):
        cells: Dict[tuple, set] = {}
        sizes: Dict[tuple, int] = {}
        for prototype, recording, caption in rows:
            key = (prototype, genre_of(recording, groups) if genre_split else None)
            cells.setdefault(key, set()).add(caption)
            sizes[key] = sizes.get(key, 0) + 1
        distinct = sorted(len(value) for value in cells.values())
        needed = [max(1, -(-value // args.subset_cap)) for value in distinct]
        print(json.dumps({
            "genre_split": genre_split,
            "cells": len(cells),
            "segments_per_cell_median": int(statistics.median(sorted(sizes.values()))),
            "distinct_captions_per_cell_median": int(statistics.median(distinct)),
            "distinct_captions_per_cell_p90": distinct[int(0.9 * (len(distinct) - 1))],
            "rounds_to_exhaust_median_cell": int(statistics.median(needed)),
            "max_rounds_default": args.max_rounds,
            "cells_needing_more_than_max_rounds": sum(
                1 for value in needed if value > args.max_rounds),
        }, ensure_ascii=False, sort_keys=True))
    return 0


def cmd_project(args: argparse.Namespace) -> int:
    staged = stage_inputs(args.tag, pathlib.Path(args.root),
                          want_bundle=False, want_captions=False,
                          bundle_tag=args.bundle_tag,
                          generation=args.generation)
    for genre_split in (False, True):
        report = project_class_count(args.tag, staged, args.target_size, genre_split)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


def cmd_coverage(args: argparse.Namespace) -> int:
    """Does the caption set key onto the segments M3b will re-cluster?

    The same criterion ``run_wild_stage_g_m3a_oss.py verify`` applies, repeated
    here against the *merged* captions.jsonl rather than the parts, because that
    merged file is what M3b is actually handed and a merge can drop rows.
    """
    staged = stage_inputs(args.tag, pathlib.Path(args.root), want_bundle=False,
                          bundle_tag=args.bundle_tag,
                          generation=args.generation)
    spans = clustered_spans(staged["labels"], staged["cache"])
    total = sum(len(value) for value in spans.values())
    covered = orphan = 0
    with staged["captions"].open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            span = (int(row["start"]), int(row["end"]))
            if span in spans.get(row["recording_id"], set()):
                covered += 1
            else:
                orphan += 1
    coverage = covered / total if total else 0.0
    print("segments M3b re-clusters : {}".format(total))
    print("with a caption           : {}".format(covered))
    print("captions keying nowhere  : {}".format(orphan))
    print("coverage                 : {:.4f}  (floor {:.2f})".format(
        coverage, args.min_caption_coverage))
    if coverage < args.min_caption_coverage:
        print("COVERAGE_FAIL", flush=True)
        return 1
    print("COVERAGE_OK", flush=True)
    return 0


def stamped(stem: str, generation: str, suffix: str = "") -> str:
    """A local filename that carries the caption generation it belongs to.

    Written once and used everywhere this stage puts a file on disk, because
    doing it per call site is what produced four separate instances of the same
    defect on 2026-08-22 -- the summarise checkpoint, the recluster output tree,
    the per-shard report and the assembled grouping.  Each one looked like an
    isolated oversight; together they are one rule that was never written down:
    **the object keys are not the only names that have to be unique per
    generation.**  The fourth instance re-clustered v1's captions against v2's
    grouping and produced 53 sub-prototypes where there should have been 820,
    with ``ungrouped_segments: 0`` reported cheerfully beside it.
    """
    return "{}{}{}".format(stem, "_" + generation if generation else "", suffix)


def cells_dir(args: argparse.Namespace) -> str:
    """Cells from a pre-split run and cells from a flat run are not the same
    cells -- the key carries the group -- so they never share a directory or
    an object prefix.  Reusing one path for both is how a checkpoint from the
    configuration you abandoned gets read as progress on the one you kept.

    The **caption generation** belongs in this name for exactly the same
    reason, and did not carry it until 2026-08-22.  The object keys were
    stamped with the generation that day but this local path was not, so the
    v2 run resumed from v1's checkpoint, summarised nothing, and republished
    v1's cells under the v2 key -- byte-identical, while every shard printed
    ``published 7 cell(s)`` and exited 0.
    """
    stem = "summarise_cells_presplit" if args.genre_split else "summarise_cells"
    return stamped(stem, getattr(args, "generation", ""))


def guard_checkpoint_source(checkpoint: pathlib.Path,
                            captions: pathlib.Path) -> None:
    """Refuse a checkpoint that was not built from these captions.

    The path fix above is a *name*, and a name only protects against the
    mistakes someone thought of.  This is the criterion: the digest of the
    caption file each checkpoint was produced from is written beside it, and a
    resume whose captions have changed is refused rather than silently
    continued.  A summarising run is hours of cards; discovering afterwards
    that it re-emitted an older generation costs all of them.
    """
    import hashlib

    digest = hashlib.sha256(captions.read_bytes()).hexdigest()
    marker = checkpoint.with_name(checkpoint.name + ".source")
    if checkpoint.exists() and checkpoint.stat().st_size:
        recorded = marker.read_text(encoding="utf-8").strip() if marker.exists() else ""
        if recorded != digest:
            raise SystemExit(
                "{} holds cells built from {} captions, not from {} "
                "(sha256 {}...).  Point --generation at the generation that "
                "produced them, or remove the checkpoint to re-summarise."
                .format(checkpoint,
                        "unrecorded" if not recorded else recorded[:12] + "...",
                        captions, digest[:12]))
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(digest + "\n", encoding="utf-8")


def cells_key(args: argparse.Namespace, tag: str) -> str:
    keys = m3b_keys(tag, generation=getattr(args, "generation", ""))
    return keys["cells_presplit"] if args.genre_split else keys["cells"]


def variant_key(args: argparse.Namespace, tag: str, name: str) -> str:
    keys = m3b_keys(tag, generation=getattr(args, "generation", ""))
    return keys[name + "_presplit"] if args.genre_split else keys[name]


def summarise_argv(args: argparse.Namespace, staged: Dict[str, pathlib.Path],
                   out: pathlib.Path, checkpoint: pathlib.Path) -> List[str]:
    argv = [interpreter(), "tools/summarize_subprototypes_llm.py",
            "--captions", str(staged["captions"]),
            "--model", str(REPO / MODEL),
            "--output", str(out),
            "--checkpoint", str(checkpoint)]
    # The pre-split is off by default, which plan §7.10 already decided for this
    # corpus and today's cell census agrees with from the other side: with the
    # 25 uploader accounts on, 2,369 of 2,489 cells fall below --target-size and
    # each yields exactly one sub-prototype, so the vocabulary's size would be
    # set by how many accounts there are rather than by any clustering.
    if args.genre_split:
        argv += ["--genre-map", str(staged["group_keys"])]
    # Raised from the tool's default of 20, which was set against AIST++ cells
    # holding ~20 segments.  Measured on wild_v4 with the pre-split off, a cell
    # holds 1,954 segments and 304 *distinct captions*, and the loop removes at
    # most --subset-cap of those per round: 51 rounds to exhaust the median
    # cell, 65 at p90, and all 100 cells exceed 20.  Leaving it at 20 would end
    # every cell at the cap with ~39% of its captions grouped and the rest
    # assigned by field agreement -- a vocabulary that is the fallback rule
    # wearing an LLM's name, which is exactly what
    # segments_in_llm_formed_subprototypes exists to expose.  The residual
    # threshold still stops a healthy cell earlier, so this is a ceiling and
    # not a target.
    argv += ["--max-rounds", str(args.max_rounds),
             "--residual-threshold", str(args.residual_threshold),
             "--subset-cap", str(args.subset_cap),
             "--max-prompt-captions", str(args.max_prompt_captions)]
    return argv


def guard_subprototype_source(subprototypes: pathlib.Path,
                              captions: pathlib.Path) -> None:
    """Refuse a grouping that was not summarised from these captions.

    The stamped name above is a *name*, and a name only protects against the
    mistake someone already made -- the same sentence ``guard_checkpoint_source``
    opens with, and it is here because the name alone was what M3c had.  The
    criterion is the report's own ``captions`` field: the summariser records the
    file it read, so a grouping staged from another generation announces itself.
    On 2026-08-27 that field said ``captions_v2.jsonl`` while the run staged
    ``captions_v5union.jsonl`` -- one string comparison between a published
    vocabulary and a corpus it had never seen.

    Compared on basename, not on the full path: the staging root is shared and
    rebuilt, so the recorded absolute path is stale for reasons that are not
    defects.  What has to match is which caption file, i.e. which generation.
    """
    report = json.loads(subprototypes.read_text(encoding="utf-8"))
    recorded = pathlib.Path(str(report.get("captions", ""))).name
    if recorded != captions.name:
        raise SystemExit(
            "{} was summarised from {}, not from {}.  That is a different "
            "caption generation: re-clustering against it would publish a "
            "vocabulary built on captions this corpus never produced.  Point "
            "--generation at the generation that produced the grouping, or "
            "remove the staged file to re-fetch it."
            .format(subprototypes, recorded or "an unrecorded caption file",
                    captions.name))


def cmd_summarise(args: argparse.Namespace) -> int:
    staged = stage_inputs(args.tag, pathlib.Path(args.root), want_bundle=False,
                          bundle_tag=args.bundle_tag,
                          generation=args.generation)
    keys = m3b_keys(args.tag, generation=args.generation)
    root = pathlib.Path(args.root)
    out = root / stamped("subprototypes_shard{}".format(args.shard),
                         args.generation, ".json")
    checkpoint = (root / cells_dir(args)
                  / "shard-{:02d}.jsonl".format(args.shard))
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    guard_checkpoint_source(checkpoint, staged["captions"])

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    argv = summarise_argv(args, staged, out, checkpoint)
    argv += ["--shard", str(args.shard), "--num-shards", str(args.num_shards)]
    print("$ {}".format(" ".join(argv)), flush=True)
    if subprocess.run(argv, cwd=str(REPO), env=env).returncode != 0:
        return 1
    # Published per shard: the cells are the durable thing, and the assembled
    # report is derived from them.  A shard that finishes while another is still
    # running has its work in the store either way.
    asset_io.write_bytes(
        repo_key(cells_key(args, args.tag), checkpoint.name), checkpoint.read_bytes())
    print("published {} cell(s) for shard {}".format(
        sum(1 for _ in checkpoint.open(encoding="utf-8")), args.shard), flush=True)
    return 0


def cmd_assemble(args: argparse.Namespace) -> int:
    """One report from every shard's cells, with no model in the room.

    ``--merge-only`` refuses rather than fills a gap: a cell missing because its
    shard died would otherwise become a prototype with no sub-prototypes, and
    ``recluster`` would fall back to the keyframe criterion for it and report a
    clean run.
    """
    staged = stage_inputs(args.tag, pathlib.Path(args.root), want_bundle=False,
                          bundle_tag=args.bundle_tag,
                          generation=args.generation)
    keys = m3b_keys(args.tag, generation=args.generation)
    root = pathlib.Path(args.root)
    merged = root / cells_dir(args) / "all.jsonl"
    merged.parent.mkdir(parents=True, exist_ok=True)

    cells = []
    for name in sorted(asset_io.list_prefix(cells_key(args, args.tag))):
        if name.endswith(".jsonl"):
            cells.append(asset_io.read_bytes(repo_key(cells_key(args, args.tag), name)))
    if not cells:
        raise SystemExit("no shard cells under {}".format(cells_key(args, args.tag)))
    merged.write_bytes(b"".join(cells))
    print("merged {} shard file(s), {} cell(s)".format(
        len(cells), sum(1 for _ in merged.open(encoding="utf-8"))), flush=True)

    out = root / stamped("subprototypes_llm", args.generation, ".json")
    argv = summarise_argv(args, staged, out, merged) + ["--merge-only"]
    print("$ {}".format(" ".join(argv)), flush=True)
    if subprocess.run(argv, cwd=str(REPO)).returncode != 0:
        return 1
    asset_io.write_bytes(variant_key(args, args.tag, "subprototypes"), out.read_bytes())
    report = json.loads(out.read_text(encoding="utf-8"))
    print(json.dumps({k: v for k, v in report.items() if k != "groups_detail"},
                     ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def cmd_recluster(args: argparse.Namespace) -> int:
    staged = stage_inputs(args.tag, pathlib.Path(args.root), want_bundle=True,
                          bundle_tag=args.bundle_tag,
                          generation=args.generation)
    keys = m3b_keys(args.tag, generation=args.generation)

    report = project_class_count(args.tag, staged, args.target_size, args.genre_split)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)

    # ``stamped`` here for the same reason it is used three lines below, and
    # this is the fifth instance of that one defect -- the first on the *read*
    # side.  Until 2026-08-27 this staging name carried the pre-split but not
    # the generation, and the ``not is_file()`` reuse guard then trusted
    # whatever already sat there: a run of wild_v5_song's v5union generation
    # picked up clean5b5's v2 grouping from 2026-08-22 (53 groups, 1,004
    # sub-prototypes over 11,971 segments) and re-clustered 201,913 wild_v5
    # spans against it.  It does not crash; it publishes.
    subproto = (pathlib.Path(args.root)
                / stamped("subprototypes_llm_presplit" if args.genre_split
                          else "subprototypes_llm",
                          getattr(args, "generation", ""), ".json"))
    if args.subprototypes and not subproto.is_file():
        subproto.write_bytes(asset_io.read_bytes(
            variant_key(args, args.tag, "subprototypes")))
    if args.subprototypes:
        guard_subprototype_source(subproto, staged["captions"])

    # The local staging directory is keyed the same way as the object prefix,
    # and for the same reason: ``fetch_tree_once`` and this stage share a root
    # across runs, so one name for both rows would have a w/o-LLM tree sitting
    # where the next reader expects the w/ LLM one.
    stem = "ingroup_llm" if args.subprototypes else "ingroup_nollm"
    if args.genre_split:
        stem += "_presplit"
    # ...and the generation, for the third time in one day.  The object key was
    # stamped; this local directory was not, so re-clustering the v2 generation
    # deleted v1's staged labels and wrote its own in their place.  Scoring
    # "v1" after that read v2's labels against v1's captions and produced a
    # table that looked ordinary -- 833 groups where v1 has 709 was the only
    # thing that gave it away.  Any path this stage writes has to carry the
    # generation, not only the ones that leave the machine.
    out = pathlib.Path(args.root) / stamped(stem, getattr(args, "generation", ""))
    shutil.rmtree(out, ignore_errors=True)
    argv = [sys.executable, "tools/recluster_atomics_ingroup.py",
            "--labels", str(staged["labels"]),
            "--bundle", str(staged["bundle"]),
            "--output-dir", str(out),
            "--embedding-cache", str(staged["cache"]),
            "--captions", str(staged["captions"]),
            "--group-keys", str(staged["group_keys"]),
            "--target-size", str(args.target_size),
            "--min-retrieval-groups", str(args.min_retrieval_groups),
            "--seed", str(args.seed)]
    if args.genre_split:
        argv.append("--genre-split")
    if args.subprototypes:
        argv += ["--subprototypes", str(subproto)]
    print("$ {}".format(" ".join(argv)), flush=True)
    if subprocess.run(argv, cwd=str(REPO)).returncode != 0:
        return 1
    bundle = variant_key(args, args.tag,
                         "bundle" if args.subprototypes else "bundle_nollm")
    asset_io.publish_dir(out, bundle)
    print("published {}".format(bundle), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default=os.environ.get("TAG", "wild_v4"))
    parser.add_argument("--bundle-tag", default=os.environ.get("BUNDLE_TAG"),
                        help="take the account map and the stage-E performance "
                             "bundle from THIS tag; the labels, the captions "
                             "and every output belong to --tag.  Defaults to "
                             "--tag, i.e. the historical behaviour")
    parser.add_argument("--generation", default=os.environ.get("GENERATION", ""),
                        help="caption generation to consume, i.e. M3a's "
                             "--parts-suffix.  It keys every output as well as "
                             "the captions read: two generations are two "
                             "vocabularies, and this store cannot delete the "
                             "first one to make room for the second")
    parser.add_argument("--root", default="/dev/shm/atomicdance-m3a-segmentation",
                        help="shared with M3a, so the label tree and the M2 "
                             "embedding cache are staged once for both stages")
    subparsers = parser.add_subparsers(dest="command", required=True)

    project = subparsers.add_parser(
        "project", help="class count both ways, before booking the LLM")
    project.set_defaults(func=cmd_project)

    cells = subparsers.add_parser(
        "cells", help="distinct captions per cell -- the LLM's round budget")
    cells.add_argument("--subset-cap", type=int, default=6)
    cells.add_argument("--max-rounds", type=int, default=20)
    cells.set_defaults(func=cmd_cells)

    coverage = subparsers.add_parser(
        "coverage", help="M3b's caption-coverage criterion on the merged file")
    coverage.add_argument("--min-caption-coverage", type=float, default=0.9)
    coverage.set_defaults(func=cmd_coverage)

    summarise = subparsers.add_parser("summarise", help="the paper's second model")
    summarise.add_argument("--gpu", default=os.environ.get("GPU", "0"))
    summarise.add_argument("--shard", type=int, default=0)
    summarise.add_argument("--num-shards", type=int, default=1)
    summarise.set_defaults(func=cmd_summarise)

    assemble = subparsers.add_parser(
        "assemble", help="one subprototypes report from every shard's cells")
    assemble.set_defaults(func=cmd_assemble)

    for sub in (summarise, assemble):
        sub.add_argument("--max-rounds", type=int, default=80,
                         help="ceiling on the paper's loop per cell; 80 clears "
                              "the p90 cell (65 rounds) on wild_v4 with the "
                              "pre-split off.  The residual threshold still "
                              "stops a healthy cell first")
        sub.add_argument("--residual-threshold", type=float, default=0.03)
        sub.add_argument("--max-prompt-captions", type=int, default=80,
                         help="80 captions is ~2,000 tokens at this corpus's "
                              "98 chars each, against the ~1 GiB the MoE has "
                              "over its weights; the 225-caption cell that "
                              "OOMed asked the forward for 2.04 GiB")
        sub.add_argument("--subset-cap", type=int, default=6)
        sub.add_argument("--genre-split", action="store_true",
                         help="pre-split cells by uploader account.  Off by "
                              "default: plan 7.10 decided it for this corpus, "
                              "and with it on 95%% of cells yield exactly one "
                              "sub-prototype, so account count sets the "
                              "vocabulary size instead of the clustering")

    recluster = subparsers.add_parser("recluster", help="read the grouping back on")
    recluster.add_argument("--subprototypes", action="store_true", default=True,
                           help="the w/ LLM row; --no-subprototypes is w/o LLM")
    recluster.add_argument("--no-subprototypes", dest="subprototypes",
                           action="store_false")
    recluster.add_argument("--min-retrieval-groups", type=int, default=2,
                           help="2 enforces the paper's own premise that a "
                                "prototype is a recurring movement; it is what "
                                "made the AIST release pass its audit")
    recluster.add_argument("--seed", type=int, default=20260815)
    recluster.set_defaults(func=cmd_recluster)

    for sub in (project, recluster):
        sub.add_argument("--target-size", type=int, default=227,
                         help="target samples per sub-prototype; 227 is pinned "
                              "to the wild_v2 scale, and read the module "
                              "docstring before combining it with --genre-split")
    recluster.add_argument("--genre-split", action="store_true",
                           help="pre-split each prototype by uploader account; "
                                "this, not --target-size, then sets the "
                                "vocabulary size -- see the module docstring")
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    raise SystemExit(parsed.func(parsed))
