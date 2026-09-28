#!/usr/bin/env python3
"""Stage G / M3a over the OSS-resident corpus: caption every accepted segment.

This is the expensive and irreversible step -- 194,682 segments, about 26 card
hours -- and it is the only one that needs the video.  ``clip.mp4`` is 393 GiB
of the 408 GiB ingest tree and is deliberately never pulled whole: each batch
fetches the clips it is about to caption and deletes them, which is stage B's
shape and the reason a worker's disk footprint does not grow with the corpus.

Sharding is by **clip**, not by segment.  The captioner shards segments itself,
but a segment shard needs an unpredictable set of videos, so this driver would
have to replay the captioner's own partitioning to know what to fetch -- and
replaying a filter you do not own is how ``line_a_owed.py`` got three shard
counts wrong.  Partitioning the clip list instead is exact: each worker gets a
labels manifest containing only its clips and runs with ``--num-shards 1``.

What is pinned here rather than inherited from ``launch_caption_shards.sh``:

* ``--person-boxes data/wild_ingest_v1``, not the script's default
  ``data/wild3d/gvhmr_raw``.  On the rebuilt corpus the dancer box comes from
  the ingest, and ``person_boxes_for`` returns None rather than raising when it
  cannot find one -- so the old default would have captioned uncropped frames
  for every clip and said nothing.  The dancer is a median 42% of frame height
  here, so that is most of the vision budget spent on the room.
* ``genre_presplit: false``.  Measured on AIST, this VLM tags genre at 0.083
  against a 0.10 chance rate, and doubling the frames did not help; splitting
  prototypes on labels at chance is worse than not splitting.
* ``--embedding-cache``, which the first version of this driver omitted.  It is
  what tells the captioner to use the spans M2 clustered rather than runs of
  equal frame label, and the two are not the same segmentation: adjacent
  segments that M2 gave the same prototype merge into one.  Measured on the
  finished wild_v4 run, that is not a rounding error --

      M2's clustered spans              194,682
      runs of equal label (what ran)    171,876
      identical keys                    155,008
      spans no caption could key onto    39,674   (20.4% of M2's segments)

  -- and caption rows are keyed ``(recording_id, start, end)``, which is how
  ``recluster_atomics_ingroup`` looks them up.

  Measured against what was actually published, the hole is bigger than the
  segmentation alone, because 733 clips were lost to failed batches and never
  captioned under either rule:

      published captions                146,351
        keyed onto a clustered span     132,071
        keyed onto nothing               14,280
      coverage M3b would see             0.6784   (its floor is 0.90)
      owed                               62,611   = 39,674 mis-keyed
                                                  + 22,937 never captioned

  So the gate *would* have fired -- after 26 card hours were already spent.
  ``verify`` runs that same criterion against the store, before the cards are
  booked rather than after, and it is 30 seconds because it reads the parts as
  a tree.
"""

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import asset_io                                    # noqa: E402
from tools.run_wild_stage_c_oss import INGEST_ROOT, publish_rows, repo_key   # noqa: E402
from tools.run_wild_stage_e_oss import stage_e_keys           # noqa: E402
from tools.run_wild_stage_g_m2_oss import m2_keys             # noqa: E402

MODEL = "third_party/QwenVL/Qwen3-VL-30B-A3B-Instruct/main"
QWEN_PY = "third_party/QwenVL/.venv-qwen3vl/bin/python"


def m3a_keys(tag: str, parts_suffix: str = "") -> Dict[str, str]:
    """Object keys for one caption *generation*.

    ``parts_suffix`` names the generation because the store cannot delete: the
    first generation was captioned on the wrong segmentation and its objects
    are permanent.  Overwriting them in place is not available either, since a
    re-run partitions clips differently and so writes a different set of part
    names.  A suffixed prefix keeps each generation internally consistent and
    lets ``merge`` say exactly which ones it is reading -- the same answer the
    undeletable ``part-000.jsonl`` already forced on the canonical-name filter.
    """
    root = "runs/{}_captions".format(tag)
    parts = "parts" if not parts_suffix else "parts_{}".format(parts_suffix)
    # The merged file follows the suffix too.  It did not until 2026-08-22, so
    # merging a second generation overwrote the first one's captions.jsonl in
    # place while that generation's parts stayed where they were -- and every
    # downstream reader (M3b's coverage gate, recluster, the published bundle)
    # names only ``captions.jsonl``, so which generation it had just consumed
    # was not recorded anywhere.  Overwriting is the one thing this store *can*
    # do, which is exactly why it needs a name per generation rather than a
    # habit of care.
    merged = "captions.jsonl" if not parts_suffix else "captions_{}.jsonl".format(parts_suffix)
    return {"root": root,
            "parts": repo_key(root, parts),
            "embeddings": "runs/{}_tmr_embeddings.npz".format(tag),
            "captions": repo_key(root, merged)}


def clustered_spans(labels_dir: pathlib.Path, cache_path: pathlib.Path,
                    min_frames: int = 4) -> Dict[str, set]:
    """The span keys ``caption_segments_vlm`` emits, without their prototypes.

    A thin view over ``clustered_span_prototypes`` so the segment arithmetic
    below exists once.  See that function for why it is a replay rather than an
    approximation.
    """
    return {recording: set(spans) for recording, spans
            in clustered_span_prototypes(labels_dir, cache_path, min_frames).items()}


def clustered_span_prototypes(labels_dir: pathlib.Path, cache_path: pathlib.Path,
                              min_frames: int = 4) -> Dict[str, Dict[tuple, int]]:
    """``{recording: {(start, end): prototype}}`` for every clustered segment.

    The prototype is carried, not dropped, because a caption row records one
    too and the two can disagree without anything noticing: ``--reuse-parts``
    republishes an earlier generation's rows verbatim after checking only that
    the *span* still exists, and the M2 vocabulary those rows were captioned
    against may since have been refitted.  ``rekey_captions_to_labels.py``
    exists for exactly that and its docstring names the consequence -- the
    summarizer groups by whatever integer the row carries, and the resulting
    sub-prototypes describe cells of a clustering the release was not built
    from.  On 2026-08-27 that was 80,720 of 201,913 rows.

    Deliberately a replay of ``iter_segments``' own arithmetic rather than an
    approximation of it: clip the span to the label array, read the label at the
    start, drop transitions and anything under ``min_frames``.  A check that
    computes coverage some *other* way answers a question nobody asked -- the
    2026-08-13 near-retraction came from measuring an old bundle with the new
    rule, which erased exactly the defect it was meant to find.
    """
    import numpy as np

    spans: Dict[str, List[tuple]] = {}
    cached = np.load(cache_path, allow_pickle=True)
    for key in ("recordings", "starts", "ends"):
        if key not in cached:
            raise SystemExit("error: {} has no '{}'; it is not a segment-level "
                             "embedding cache".format(cache_path, key))
    for name, start, end in zip(cached["recordings"], cached["starts"], cached["ends"]):
        spans.setdefault(str(name), []).append((int(start), int(end)))

    keys: Dict[str, Dict[tuple, int]] = {}
    for line in (labels_dir / "labels.jsonl").open(encoding="utf-8"):
        entry = json.loads(line)
        recording = entry["recording_id"]
        owned = spans.get(recording)
        if not owned:
            continue
        labels = np.load(labels_dir / entry["labels_path"])
        found = {}
        for start, end in owned:
            end = min(int(end), len(labels))
            start = int(start)
            # Length first, then the label -- the original wrote this as one
            # short-circuited condition and a span starting past the end of the
            # array (clipped to nothing) reached the indexing otherwise.
            if end - start < min_frames:
                continue
            label = int(labels[start])
            if label <= 0:
                continue
            found[(start, end)] = label
        if found:
            keys[recording] = found
    return keys


def clip_stem(recording_id: str) -> str:
    parts = str(recording_id).split(":")
    return "{}__{}".format(parts[-2], parts[-1]) if len(parts) >= 3 else str(recording_id)


def stage_batch(scratch: pathlib.Path, label_rows: List[Dict[str, Any]],
                seq_rows: Dict[str, Dict[str, Any]], src_rows: List[Dict[str, Any]],
                tag: str, bundle_tag: str = None) -> Dict[str, pathlib.Path]:
    """Materialise one batch of clips: labels, bundle stores, videos, boxes.

    ``bundle_tag`` defaults to ``tag``; see the module docstring for why the
    bundle may come from a different tag than the labels.
    """
    ekeys, mkeys = stage_e_keys(bundle_tag or tag), m2_keys(tag)
    labels_dir = scratch / "labels"
    bundle_dir = scratch / "bundle"
    videos = scratch / "videos"
    ingest = scratch / "ingest"
    for path in (labels_dir, bundle_dir, videos, ingest):
        path.mkdir(parents=True, exist_ok=True)

    kept: List[Dict[str, Any]] = []
    for row in label_rows:
        recording = row["recording_id"]
        sequence = seq_rows.get(recording)
        if sequence is None:
            continue
        stem = clip_stem(recording)
        try:
            for field in ("labels_path", "label_valid_mask_path"):
                rel = row.get(field)
                if not rel:
                    continue
                target = labels_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(asset_io.read_bytes(repo_key(mkeys["labels"], rel)))
            for rel in sorted(set(v for k, v in (sequence.get("assets") or {}).items()
                                  if isinstance(v, str) and v.endswith(".npy"))):
                target = bundle_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(asset_io.read_bytes(repo_key(ekeys["bundle"], rel)))
            (videos / (stem + ".mp4")).write_bytes(
                asset_io.read_bytes(repo_key(INGEST_ROOT, stem, "clip.mp4")))
            box = ingest / stem / "preprocess" / "bbx.pt"
            box.parent.mkdir(parents=True, exist_ok=True)
            box.write_bytes(asset_io.read_bytes(
                repo_key(INGEST_ROOT, stem, "preprocess", "bbx.pt")))
        except Exception as error:                             # noqa: BLE001
            print("FETCH_FAIL {} {}".format(recording, str(error)[:120]), flush=True)
            continue
        kept.append(row)

    keep_ids = {row["recording_id"] for row in kept}
    with (labels_dir / "labels.jsonl").open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (bundle_dir / "sequences.jsonl").open("w", encoding="utf-8") as handle:
        for recording in keep_ids:
            handle.write(json.dumps(seq_rows[recording], sort_keys=True) + "\n")
    with (bundle_dir / "sources.jsonl").open("w", encoding="utf-8") as handle:
        for row in src_rows:
            if row.get("recording_id") in keep_ids or row.get("sequence_id") in keep_ids:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    return {"labels": labels_dir, "bundle": bundle_dir, "videos": videos,
            "ingest": ingest, "kept": len(kept)}


def fetch_tree_once(key: str, dest: pathlib.Path) -> pathlib.Path:
    """Fetch a prefix to ``dest``, or leave it alone if someone already did.

    Into a private directory and then renamed, because seven workers share this
    root and the obvious form -- ``if not dest.exists(): fetch(dest)`` -- has a
    window where the tree exists and is a third full.  A worker that walks in
    then reads a partial label tree as a complete one and computes a coverage
    number for a corpus that is not there.  ``rename`` onto a non-empty
    directory fails on Linux, which is exactly the arbitration wanted: the
    loser throws its copy away.  The repo already writes this down as
    ``immutable_new_directory_only_atomic_rename``; 2026-08-13 lost a bundle by
    publishing in place, and this is the same hazard on the read side.
    """
    if dest.is_dir() and any(dest.iterdir()):
        return dest
    staging = dest.parent / "{}.partial-{}".format(dest.name, os.getpid())
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    asset_io.fetch_dir(key, staging)
    try:
        staging.rename(dest)
    except OSError:
        shutil.rmtree(staging, ignore_errors=True)
    return dest


def m2_generation(tag: str) -> str:
    """A short id for *which run of M2* the store currently holds for ``tag``.

    ``<labels>/report.json`` is a few KB and changes on every refit: it carries
    the segment counts, the acceptance rate and the sha256 of the sources
    manifest the vocabulary was fitted against.  So one small GET distinguishes
    two generations of one tag, which is the thing the staged names could not
    do -- see ``stage_segmentation``.
    """
    payload = asset_io.read_bytes("{}/report.json".format(m2_keys(tag)["labels"]))
    return hashlib.sha256(payload).hexdigest()[:12]


def stage_segmentation(tag: str, root: pathlib.Path) -> Dict[str, pathlib.Path]:
    """The label tree and the M2 embedding cache, fetched once per worker.

    Once, not per batch: ``stage_batch`` pulls label arrays clip by clip, and
    the whole tree is 85 MiB against 27,569 individual GETs.  The cache is
    210 MiB and every batch needs the same one.

    **The staged names carry the tag AND the generation.**  The tag alone was
    not enough, twice, for the same reason: ``fetch_tree_once`` returns a
    non-empty directory untouched by design (seven workers share this root), so
    whatever is already staged wins.

    * *Across tags* (2026-08-21): the label tree used to land on
      ``<root>/labels`` for every tag, so a second tag staged into a root that
      already held the first read the first one's prototypes while reading its
      own embedding cache -- two generations paired, no error.
      ``/dev/shm/atomicdance-m3a-segmentation`` held ``clean5b4``'s tree under
      ``labels/`` beside ``clean5b4_tmr_embeddings.npz``.
    * *Within one tag* (2026-08-26): adding the tag fixed the first case and
      not this one.  The union re-split forced M2 to refit under the unchanged
      tag ``wild_v5_song``; the refit was published to the same keys (the store
      cannot delete, so in-place is the only shape that leaves no orphan), and
      the next ``verify`` read the **pre-refit** tree still sitting in
      ``/dev/shm`` -- 201,806 clustered spans against the store's 201,913, and
      no error either.  Had M3a been resumed instead of verified, seven cards
      would have captioned against the superseded vocabulary and the coverage
      gate would have passed, because coverage is computed against that same
      stale tree.

    So the generation is read from the store (one small GET) and put in both
    staged names.  A refit misses by construction rather than by whoever
    remembers to clear ``/dev/shm``.
    """
    root.mkdir(parents=True, exist_ok=True)
    generation = m2_generation(tag)
    labels_dir = fetch_tree_once(m2_keys(tag)["labels"],
                                 root / "{}-{}_labels".format(tag, generation))
    cache = root / "{}-{}_tmr_embeddings.npz".format(tag, generation)
    if not cache.is_file():
        # Same reasoning, one object: a half-written 210 MiB npz is a file that
        # exists and cannot be loaded, and the next worker would skip fetching it.
        partial = cache.with_suffix(".npz.partial-{}".format(os.getpid()))
        partial.write_bytes(asset_io.read_bytes(m3a_keys(tag)["embeddings"]))
        try:
            partial.rename(cache)
        except OSError:
            partial.unlink(missing_ok=True)
    return {"labels": labels_dir, "cache": cache, "generation": generation}


def published_keys(parts_key: str, staging: pathlib.Path
                   ) -> Dict[str, Dict[tuple, Dict[str, Any]]]:
    """Captions already in the store, by recording then by ``(start, end)``.

    Fetched as a tree with ``fetch_dir`` rather than object by object.  This
    prefix is 86 parts / 235 MiB and reading them through the fsspec path took
    over ten seconds each -- the same asymmetry ``list_prefix`` documents for
    listing, where ossutil returns in five seconds what fsspec walks for
    minutes.  Both callers here want every part, so there is no case for the
    per-object path.
    """
    fetch_tree_once(parts_key, staging)
    rows: Dict[str, Dict[tuple, Dict[str, Any]]] = {}
    for path in sorted(staging.rglob("*.jsonl")):
        if not re.match(r"^part-\d{3}-\d{4}\.jsonl$", path.name):
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                rows.setdefault(row["recording_id"], {})[
                    (int(row["start"]), int(row["end"]))] = row
    return rows


def cmd_run(args: argparse.Namespace) -> int:
    bundle_tag = args.bundle_tag or args.tag
    ekeys, mkeys = stage_e_keys(bundle_tag), m2_keys(args.tag)
    keys = m3a_keys(args.tag, args.parts_suffix)
    print("tag {} | bundle tag {}".format(args.tag, bundle_tag), flush=True)
    label_rows = sorted(asset_io.read_jsonl(repo_key(mkeys["labels"], "labels.jsonl")),
                        key=lambda row: row["recording_id"])
    seq_rows = {row["recording_id"]: row for row
                in asset_io.read_jsonl(repo_key(ekeys["bundle"], "sequences.jsonl"))}
    src_rows = list(asset_io.read_jsonl(repo_key(ekeys["bundle"], "sources.jsonl")))

    staged_seg = stage_segmentation(args.tag, pathlib.Path(args.segmentation_root))
    print("segmentation staged at {}".format(args.segmentation_root), flush=True)

    mine = label_rows[args.shard::args.num_shards]
    # Positions are fixed here, before every filter below, so a part name
    # means the same clip whatever this pass happens to skip.
    position = {row["recording_id"]: index for index, row in enumerate(mine)}

    # Clips that kill the captioner rather than failing it.  This is the third
    # state ``reconcile_wild_hmr_sequences`` keeps and the batch loop does not:
    # a clip that is *pending* comes back next pass, and one that is *poison*
    # comes back next pass and takes another 150 clips with it.
    #
    # wild_v4:7538444755769183539 earned the list.  It was ``first_lost`` of a
    # failed batch in the 6-shard run (shard 3, start 1350, 145 clips after it)
    # and again in the 7-shard run under a different partition (shard 4, start
    # 1050, 43 after it), and it holds **zero** captions in the first
    # generation -- while the other two ``first_lost`` uploads hold 17 and 13,
    # which is what a clip merely standing behind the crash looks like.
    #
    # **The zero-caption half of that signature has a precondition, and reading
    # it without checking the precondition convicts the innocent.**  Zero is
    # only evidence for a clip the earlier run actually *reached*.  Measured
    # 2026-08-26: ``wild_v5:7532132080416083260:clip000`` was ``first_lost`` of
    # a stalled batch and holds zero captions in the previous generation, which
    # reads exactly like the case above -- but it sits at position 1,374 of its
    # old shard's 1,977 clips and that shard only ever reached clip 1,200.  The
    # previous run never tried it, so its zero says nothing.  What convicted
    # 7538444755769183539 was **first_lost under two different partitions**;
    # the caption count is corroboration, not the test.  Check the earlier
    # run's progress line for that clip's shard before quoting its zero.  Both
    # deaths are the same cuDNN kernel:
    #
    #     RuntimeError: Expected mha_graph->execute(...).is_good() to be true
    #
    # Excluded by name, with the count of what that costs printed, because 28
    # spans of 194,682 is 0.014% and a batch is 150 clips.  Retrying it forever
    # is the expensive choice, and doing so silently is the dishonest one.
    poison = set(args.quarantine or [])
    if poison:
        before = len(mine)
        mine = [row for row in mine
                if clip_stem(row["recording_id"]).split("__clip")[0] not in poison]
        if before != len(mine):
            print("quarantine: dropped {} clip(s) from {} poisoned upload(s)".format(
                before - len(mine), len(poison)), flush=True)

    # Reuse: captions from an earlier generation whose key is still one of M2's
    # spans are paid for and correct, so only the difference costs a card.  The
    # seeding is per batch further down; here it only decides which clips still
    # need the GPU at all, because staging a clip means pulling its clip.mp4.
    reuse: Dict[str, Dict[tuple, Dict[str, Any]]] = {}
    wanted_spans: Dict[str, set] = {}
    if args.reuse_parts:
        wanted_spans = clustered_spans(staged_seg["labels"], staged_seg["cache"])
        earlier = published_keys(
            repo_key(m3a_keys(args.tag)["root"], args.reuse_parts),
            pathlib.Path(args.segmentation_root) / "published" / args.reuse_parts)
        owed_total = reused_total = 0
        still_owed = []
        for row in mine:
            recording = row["recording_id"]
            spans = wanted_spans.get(recording, set())
            have = {span: caption for span, caption
                    in earlier.get(recording, {}).items() if span in spans}
            reuse[recording] = have
            reused_total += len(have)
            owed = len(spans) - len(have)
            owed_total += owed
            if owed:
                still_owed.append(row)
        print("reuse: {} of this shard's {} clustered spans already captioned, "
              "{} owed across {} clips".format(
                  reused_total, reused_total + owed_total, owed_total,
                  len(still_owed)), flush=True)
        mine = still_owed
    if args.resume:
        # Skip clips a previous run already published.  A 15-hour job will be
        # interrupted -- this one already was, by a timeout I mis-sized -- and
        # without this every restart re-captions from clip zero and pays for
        # the same work twice.  Coverage is read from the store rather than
        # from a local marker, because the store is the thing a later reader
        # will consult.
        done = set()
        try:
            for name in sorted(asset_io.list_prefix(keys["parts"])):
                if not re.match(r"^part-\d{3}-\d{4}\.jsonl$", name):
                    continue
                for row in asset_io.read_jsonl(repo_key(keys["parts"], name)):
                    done.add(row.get("recording_id"))
        except Exception as error:                             # noqa: BLE001
            print("resume: could not read parts ({}), starting from the top".format(
                str(error)[:100]), flush=True)
        before = len(mine)
        mine = [row for row in mine if row["recording_id"] not in done]
        print("resume: {} of this shard's {} clips already published, {} left".format(
            before - len(mine), before, len(mine)), flush=True)
    if args.limit_clips:
        mine = mine[: args.limit_clips]
    print("M3a shard {}/{}: {} clips of {}, vocabulary generation {}".format(
        args.shard, args.num_shards, len(mine), len(label_rows),
        staged_seg["generation"]), flush=True)

    produced: List[Dict[str, Any]] = []
    started = time.time()
    for start in range(0, len(mine), args.batch):
        chunk = mine[start: start + args.batch]
        with asset_io.scratch_dir(prefix="m3a{}-".format(args.shard)) as scratch:
            staged = stage_batch(scratch, chunk, seq_rows, src_rows, args.tag,
                                 bundle_tag=args.bundle_tag)
            if not staged["kept"]:
                continue
            out = scratch / "captions.jsonl"
            # Seed the captioner's own resume set with what is already correct.
            # ``caption_segments_vlm`` keys ``done`` by (recording, start, end)
            # and reads it from this file, so writing the reusable captions here
            # buys the skip without a second mechanism that could disagree with
            # the first.  These rows are published from ``reuse`` below, never
            # from this file, so no key is emitted twice.
            seeded = 0
            if reuse:
                with out.open("w", encoding="utf-8") as handle:
                    for row in chunk:
                        for caption in reuse.get(row["recording_id"], {}).values():
                            handle.write(json.dumps(caption, sort_keys=True) + "\n")
                            seeded += 1
                print("   seeded {} reusable caption(s) into this batch".format(seeded),
                      flush=True)
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = args.gpu
            # The allocator's own advice from the OOM this driver hit: the
            # weights leave ~1 GiB of headroom, so fragmentation is the
            # difference between fitting and not.
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            argv = [str(REPO / QWEN_PY) if (REPO / QWEN_PY).is_file() else sys.executable,
                    "tools/caption_segments_vlm.py",
                    "--labels", str(staged["labels"]),
                    "--bundle", str(staged["bundle"]),
                    "--video-dir", str(staged["videos"]),
                    "--person-boxes", str(staged["ingest"]),
                    "--model", str(REPO / MODEL),
                    "--output", str(out),
                    "--shard", "0", "--num-shards", "1",
                    "--frames-per-segment", str(args.frames_per_segment),
                    "--max-side", str(args.max_side),
                    "--batch-size", str(args.batch_size),
                    "--schema", args.schema,
                    # The whole point of the second generation.  Without it the
                    # captioner segments by runs of equal frame label, which is
                    # a different segmentation from the one M2 clustered and the
                    # one M3b looks captions up by.
                    "--embedding-cache", str(staged_seg["cache"]),
                    "--device", "cuda:0"]
            if args.posescript:
                argv.append("--posescript")
            if args.as_video:
                argv.append("--as-video")
            if args.limit_segments:
                argv += ["--limit", str(args.limit_segments)]
            if args.dry_run:
                argv.append("--dry-run")
            print("$ {}".format(" ".join(argv)), flush=True)
            # Anchored, because an unanchored wait is how this repo has lost a
            # card before: on 2026-08-14 one shard froze inside libav on a clip
            # that decodes fine on its own, held GPU 0 at 0% for an hour, and
            # would have held it forever -- subprocess.run without a timeout
            # cannot notice.  The budget is generous (a healthy batch of 300
            # clips runs ~40 min at the measured 0.51 caption/s) and a batch
            # that exceeds it is dropped with its clip range named, not
            # silently: whoever re-runs needs to know which clips have no
            # captions.
            #
            # The wall-clock budget is the outer bound; the one that actually
            # fires is the stall check below.  On 2026-08-15 shard 2 froze
            # inside ``load_frames`` -- ``py-spy`` put it on
            # ``caption_segments_vlm.py:723``, the decode, with GPU 2 holding
            # 72 GB at 0% -- and it sat there for 1h44m with 70 more minutes of
            # budget left to burn.  A three-hour timeout does fire eventually,
            # but "eventually" is the wrong granularity for a hang whose cost is
            # a card: the captioner appends a row every ~2 seconds, so silence
            # is the signal, and it is available immediately.
            #
            # Signals cannot do this.  The decode blocks inside a native call,
            # where a Python SIGALRM handler does not run until the next
            # bytecode -- which is what makes this a parent's job.
            started_batch = time.time()
            stalled_at = out.stat().st_size if out.is_file() else 0
            quiet_since = time.time()
            process = subprocess.Popen(argv, cwd=str(REPO), env=env)
            failed, reason = False, ""
            while True:
                try:
                    process.wait(timeout=30)
                    failed = process.returncode != 0
                    reason = "rc={}".format(process.returncode)
                    break
                except subprocess.TimeoutExpired:
                    pass
                size = out.stat().st_size if out.is_file() else 0
                if size != stalled_at:
                    stalled_at, quiet_since = size, time.time()
                elif time.time() - quiet_since > args.stall_timeout:
                    process.kill()
                    process.wait()
                    failed = True
                    reason = "stalled: no caption written for {:.0f}s".format(
                        args.stall_timeout)
                    break
                if time.time() - started_batch > args.batch_timeout:
                    process.kill()
                    process.wait()
                    failed = True
                    reason = "timeout after {}s".format(args.batch_timeout)
                    break
            # Read whatever it wrote before it died: the captioner appends as it
            # goes, so a batch that timed out at clip 44 of 300 still holds 43
            # clips' worth of work, and throwing that away would be a second
            # loss on top of the first.
            written = list(asset_io.read_jsonl(str(out))) if out.is_file() else []
            # Drop the rows this batch was seeded with.  They are published from
            # ``reuse`` in one pass over the whole shard, so emitting them here
            # too would put the same key in two parts and ``merge`` would --
            # correctly -- refuse the whole generation.  ``written`` keeps the
            # seeded rows for the liveness check below, because a clip that this
            # batch never reached is not the same as one it had nothing to do.
            rows = [row for row in written
                    if (int(row["start"]), int(row["end"]))
                    not in reuse.get(row["recording_id"], {})] if seeded else written
            if failed:
                # Counted over the *new* captions, not over everything in the
                # file.  Seeding put this batch's reusable rows there before the
                # captioner started, so counting them made every seeded clip
                # look reached: shard 1's failed batch of 150 clips wrote 24 new
                # captions and reported ``lost=0``.  Every clip here is owed
                # something by construction -- that is what put it in ``mine``
                # -- so a clip with no new caption is a clip this batch did not
                # get to, seeded or not.
                done_ids = {row.get("recording_id") for row in rows}
                stuck = [row["recording_id"] for row in chunk
                         if row["recording_id"] not in done_ids]
                print("BATCH_FAIL shard={} start={} {} kept={} lost={} first_lost={}".format(
                    args.shard, start, reason, len(rows), len(stuck),
                    stuck[0] if stuck else None), flush=True)
            produced.extend(rows)
        # Published per batch, not once at the end.  The first version kept
        # every row in memory until the shard finished, so the shard that froze
        # was holding 504 finished captions that no reader would ever see.  A
        # batch-indexed key is still derived from the tag and the shard, so a
        # re-run overwrites rather than accumulates.
        if rows:
            publish_rows("{}/part-{:03d}-{:04d}.jsonl".format(
                keys["parts"], args.shard,
                position[chunk[0]["recording_id"]]), rows)
        print("   {}/{} clips, {} captions, {:.0f}s".format(
            min(start + args.batch, len(mine)), len(mine), len(produced),
            time.time() - started), flush=True)

    if args.dry_run:
        print("DRY_RUN shard={} clips={} (no captions written)".format(
            args.shard, len(mine)), flush=True)
        return 0

    # The carried-over half of the generation.  Written under names this shard
    # owns and in the same 150-clip unit, so a part is still a bounded object
    # and a re-run overwrites rather than accumulates.
    carried = 0
    if reuse:
        recordings = sorted(reuse)
        for index in range(0, len(recordings), args.batch):
            rows = [caption for recording in recordings[index: index + args.batch]
                    for caption in reuse[recording].values()]
            if not rows:
                continue
            publish_rows("{}/part-{:03d}-{:04d}.jsonl".format(
                keys["parts"], args.shard,
                5000 + position[recordings[index]]), rows)
            carried += len(rows)

    print("SHARD_DONE shard={} clips={} captions={} carried={}".format(
        args.shard, len(mine), len(produced), carried), flush=True)
    # A shard that captioned nothing still printed SHARD_DONE and exited 0.
    # Measured 2026-08-22: shard 6 lost both of its batches to CUDA OOM and
    # signed off as ``SHARD_DONE shard=6 clips=250 captions=0 carried=0`` --
    # which reads like success, and only the coverage gate two stages later
    # would have caught it, at 0.8815 against a 0.90 floor.  That is 0.019 of
    # margin: lose a smaller shard, or 100 clips instead of 250, and the run
    # sails through with a hole in it.  The shard knows its own emptiness
    # immediately and for free, so it says so here.
    if not produced and not carried and mine:
        print("SHARD_EMPTY shard={} produced no caption for {} clip(s); the "
              "batch log above says why".format(args.shard, len(mine)),
              flush=True)
        return 1
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """M3b's own acceptance criterion, run against the store before the cards.

    ``recluster_atomics_ingroup`` looks a caption up by ``(recording_id, start,
    end)`` and refuses the corpus when fewer than ``--min-caption-coverage`` of
    the segments it is re-clustering have one.  That check exists, and it fired
    on nothing here because it runs *after* captioning.  This is the same
    number, computed from the same two artifacts, at a point where failing it
    costs a listing instead of 26 card hours.
    """
    staged_seg = stage_segmentation(args.tag, pathlib.Path(args.segmentation_root))
    prototypes = clustered_span_prototypes(staged_seg["labels"], staged_seg["cache"])
    spans = {recording: set(value) for recording, value in prototypes.items()}
    total = sum(len(value) for value in spans.values())

    keys = m3a_keys(args.tag, args.parts_suffix)
    # A gate reads the store, never a cached copy of it.  ``fetch_tree_once``
    # skips a staging directory that already has content, which is right for a
    # captioning worker -- the parts it reuses were written before it started --
    # and wrong here: run ``verify`` twice while a pass is still publishing and
    # the second run answers from the first run's snapshot, reporting the
    # coverage the corpus had an hour ago as the coverage it has now.  That is
    # this repo's recurring shape, a check that cannot see the thing it checks.
    staging = pathlib.Path(args.segmentation_root) / "verify" / (args.parts_suffix or "parts")
    shutil.rmtree(staging, ignore_errors=True)
    have = published_keys(keys["parts"], staging)
    covered = orphan = stale = 0
    for recording, captions in have.items():
        wanted = spans.get(recording, set())
        owned = prototypes.get(recording, {})
        for span, row in captions.items():
            if span not in wanted:
                orphan += 1
                continue
            covered += 1
            # The second question this gate has to ask.  Coverage compares span
            # keys, and a caption row carries a ``prototype`` as well -- the M2
            # run the captioner was pointed at.  ``--reuse-parts`` republishes
            # an earlier generation's rows after checking only that the span
            # survived, so a vocabulary refitted in between leaves the caption
            # correct and its grouping key stale.  Nothing further down looks:
            # M3b builds its cells straight off this field, M3c joins on
            # ``(prototype, genre, caption text)`` and quietly sends the misses
            # to the keyframe fallback, and both report a clean run.  On
            # 2026-08-27 that was 80,720 rows of 201,913 with coverage at
            # 0.9997 -- the shape this repo keeps paying for, a gate that
            # cannot fail on the thing that went wrong.
            recorded = row.get("prototype")
            if recorded is not None and int(recorded) != owned.get(span):
                stale += 1

    coverage = covered / total if total else 0.0
    print("clustered spans (what M3b re-clusters) : {}".format(total))
    print("spans with a caption                   : {}".format(covered))
    print("captions keyed to no clustered span    : {}".format(orphan))
    print("coverage                               : {:.4f}  (floor {:.2f})".format(
        coverage, args.min_caption_coverage))
    print("captions whose prototype is not this vocabulary's : {}".format(stale))
    if coverage < args.min_caption_coverage:
        print("VERIFY_FAIL {} would refuse this corpus".format(
            "recluster_atomics_ingroup"), flush=True)
        return 1
    if stale:
        # Not a share of a floor: one row filed under a cell this clustering
        # never produced is a defect, and the repair is cheap and exact --
        # ``rekey_captions_to_labels.py`` rewrites the key without re-reading a
        # single frame, and refuses any row whose span the target segmentation
        # splits.  Re-captioning would be 26 card hours to fix an integer.
        print("VERIFY_FAIL {} of {} captions carry a prototype from another M2 "
              "run; re-key them onto these labels with "
              "tools/rekey_captions_to_labels.py and merge that generation "
              "instead".format(stale, covered), flush=True)
        return 1
    print("VERIFY_OK", flush=True)
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    keys = m3a_keys(args.tag, args.parts_suffix)
    # Only the canonical part-<shard>-<batch>.jsonl names.  The prefix also
    # holds part-000.jsonl from a four-clip smoke run, and the bucket cannot
    # delete it: merging everything that ends in .jsonl would fold 32 stale
    # captions into the release and duplicate the clips they cover.  A pattern
    # is the fix an undeletable store forces -- you cannot clean the prefix, so
    # the reader has to be specific about what it is reading.
    canonical = re.compile(r"^part-\d{3}-\d{4}\.jsonl$")
    all_names = sorted(name for name in asset_io.list_prefix(keys["parts"])
                       if name.endswith(".jsonl"))
    names = [name for name in all_names if canonical.match(name)]
    ignored = [name for name in all_names if not canonical.match(name)]
    if ignored:
        print("ignoring {} non-canonical part(s): {}".format(len(ignored), ignored[:4]))
    if not names:
        raise SystemExit("no canonical parts under {}".format(keys["parts"]))

    rows: List[Dict[str, Any]] = []
    seen = set()
    duplicates = 0
    for name in names:
        for row in asset_io.read_jsonl(repo_key(keys["parts"], name)):
            key = (row.get("recording_id"), row.get("start"), row.get("end"))
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            rows.append(row)
    if duplicates:
        # Loud, not silent: a duplicated segment is a re-run that overlapped,
        # and M3b would weight those segments twice when it groups captions.
        raise SystemExit(
            "refusing to merge: {} duplicate (recording_id, start, end) rows across "
            "{} parts -- a shard's slice was covered twice".format(duplicates, len(names)))

    publish_rows(keys["captions"], rows)
    print("merged {} parts -> {} ({} captions)".format(len(names), keys["captions"], len(rows)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default=os.environ.get("TAG", "wild_v4"))
    parser.add_argument("--bundle-tag", default=os.environ.get("BUNDLE_TAG"),
                        help="performance bundle / normalized / normalizer come "
                             "from THIS tag; labels and captions belong to --tag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="caption one shard of clips")
    run.add_argument("--shard", type=int, default=int(os.environ.get("SHARD", 0)))
    run.add_argument("--num-shards", type=int, default=int(os.environ.get("NUM_SHARDS", 1)))
    run.add_argument("--gpu", default=os.environ.get("GPU", "0"))
    run.add_argument("--batch", type=int, default=150,
                     help="clips per scratch batch; smaller means more model "
                          "reloads but a shorter unit of loss on failure")
    run.add_argument("--frames-per-segment", type=int, default=6)
    run.add_argument("--max-side", type=int, default=448)
    # 4, not launch_caption_shards.sh's 8.  That default predates this model:
    # Qwen3-VL-30B's weights alone hold 69.9 GiB of a 71.1 GiB card, and at
    # batch 8 the MoE forward asks for another 2.29 GiB it cannot have.  The
    # AIST M3 run that completed on this exact model used 4, so 4 is the
    # measured value rather than the inherited one.
    run.add_argument("--batch-size", type=int, default=4)
    run.add_argument("--schema", default="v1", choices=("v1", "v2", "v2draft"),
                     help="closed vocabulary the captioner fills.  v2 adds "
                          "rhythm and splits dynamics into intensity and "
                          "fluidity; pair it with a --parts-suffix, because a "
                          "second generation cannot share the first one's "
                          "object names in a store that will not delete them")
    run.add_argument("--as-video", action="store_true",
                     help="feed each segment through Qwen3-VL's video path, "
                          "which carries temporal positions the still path "
                          "does not.  Costs ~2.7x the visual tokens of the "
                          "6-frame still path at 32 frames")
    run.add_argument("--posescript", action="store_true", default=True)
    run.add_argument("--no-posescript", dest="posescript", action="store_false")
    run.add_argument("--limit-clips", type=int, default=None)
    run.add_argument("--limit-segments", type=int, default=None)
    # Sized from segments, not clips.  The first version budgeted "300 clips at
    # 0.51 caption/s = 40 min", which multiplied a clip count by a per-*segment*
    # rate; this corpus averages 14 segments per clip, so a 300-clip batch is
    # ~4,200 segments and needs ~8,200s.  The 5,400s that followed killed all
    # seven healthy first batches at 77% done and cost 490 clips.  A timeout
    # that fires on healthy work is worse than none: it converts a slow step
    # into permanent data loss, and it does it every single batch.
    run.add_argument("--stall-timeout", type=float, default=900.0,
                     help="seconds of no new caption before the batch is "
                          "declared hung.  A healthy captioner writes a row "
                          "every ~2s, so this is 450x the healthy gap; it is "
                          "what notices a decode that blocks forever, which the "
                          "wall-clock budget only notices hours later")
    run.add_argument("--batch-timeout", type=float, default=10800.0,
                     help="seconds before a batch is abandoned; a 150-clip "
                          "batch is ~2,100 segments and takes ~4,200s at the "
                          "measured 0.51 caption/s, so this is ~2.5x headroom")
    run.add_argument("--resume", action="store_true",
                     help="skip clips already covered by published parts")
    run.add_argument("--quarantine", nargs="*", default=["7538444755769183539"],
                     help="upload ids whose clips kill the captioner rather "
                          "than failing it; pass --quarantine with no values to "
                          "retry them.  The default earned its place across two "
                          "runs and two partitions, and the log says what it costs")
    run.add_argument("--reuse-parts", default=None,
                     help="an earlier generation's parts prefix (e.g. 'parts'); "
                          "its captions whose key is still one of M2's spans are "
                          "carried over instead of re-bought")
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(func=cmd_run)

    merge = subparsers.add_parser("merge", help="combine caption parts")
    merge.set_defaults(func=cmd_merge)

    verify = subparsers.add_parser(
        "verify", help="run M3b's caption-coverage criterion against the store")
    verify.add_argument("--min-caption-coverage", type=float, default=0.9,
                        help="the floor recluster_atomics_ingroup applies; "
                             "changing it here without changing it there means "
                             "the pre-flight check and the gate disagree")
    verify.set_defaults(func=cmd_verify)

    for sub in (run, verify, merge):
        sub.add_argument("--parts-suffix", default="",
                         help="caption generation to write or read; '' is the "
                              "original prefix, which was captioned on runs of "
                              "equal frame label rather than M2's spans")
        sub.add_argument("--segmentation-root",
                         default="/dev/shm/atomicdance-m3a-segmentation",
                         help="where the label tree and M2 embedding cache are "
                              "staged; fetched once per worker, not per batch")
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    raise SystemExit(parsed.func(parsed))
