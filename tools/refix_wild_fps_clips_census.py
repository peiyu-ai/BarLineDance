#!/usr/bin/env python3
"""What a set of uploads produces, before and after the fps re-ingest.

This is the manifest half of ``tools/refix_wild_fps_clips.sh``.  The re-ingest
changes *which* clips an upload yields, not only their contents: ``min_frames``
and ``max_seconds`` are policies about seconds, so on a 60 fps upload they now
cover twice as many source frames, and an upload that used to split into two
clips can come back as one.  Clip names are ``<upload>__clip%03d``, so:

* a name produced both times is **overwritten** in place -- measured, not
  assumed: a PUT over an existing key succeeds with these credentials
  (``runs/oss_capability/overwrite_probe.json``);
* a name that stops being produced becomes an **orphan** object that these
  credentials cannot delete -- ``ossutil rm`` answers ``StatusCode=403,
  AccessDenied ... because of bucket acl``, checked the same day.

So the orphan set is the thing that has to be carried forward by name.  Nothing
downstream may enumerate this corpus by globbing a directory or listing an OSS
prefix, because both will keep returning the orphans forever.  This tool writes
the list; excluding it is the consumer's job.

Two censuses, one tool::

    # before the re-ingest, over the corpus as released
    refix_wild_fps_clips_census.py --uploads U --ingest I --output before.json
    # after, and the comparison
    refix_wild_fps_clips_census.py --uploads U --ingest I --output after.json \
        --compare before.json

**The before-census cannot weigh bytes and does not pretend to.**  ``clip.mp4``
has been evicted from the local ingest tree to OSS, so the baseline knows each
clip's ``source_frame_span`` from ``meta.json`` and nothing about its content.
The comparison therefore reports three populations and refuses to merge them:
``changed`` (the span moved -- proof it differs), ``unchanged_span`` (the span
is identical), and, inside the latter, ``unverified_bytes`` -- clips whose
bytes were never compared because the baseline had none to compare against.
An unverified clip is not a clean clip.  The R0 round is why this distinction
is spelled out: judging staleness by frame count called 3,929 clips stale and
judging it by content hash called 3,991, and the 62 in between were the ones
whose old cut happened to be the same length.

The exact staleness judgment downstream needs is not made here.  Every stage's
own record carries the hash of the video it consumed -- the S3D extractor
writes ``video_sha256_1mb`` into each ``.npz``, ``run_gvhmr_extract`` into each
``extract_meta.json`` -- so each stage compares against its own record.  This
tool supplies the other side of that comparison: ``clip_sha256_1mb`` for every
clip that exists locally after the re-ingest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Dict, List, Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

HASH_LIMIT = 1 << 20        # first megabyte -- the key every stage already uses
# A clip counts as produced when the run that wrote the manifest row ended up
# with it.  ``cut_failed`` is deliberately not here: the cut did not happen, so
# whatever is in that directory is last generation's.
PRODUCED_STATUSES = ("ok", "exists")
# A cut that was attempted and failed is not an orphan and must not be put in
# the same list.  An orphan is finished with -- nothing may read it again.  A
# failed cut is work still owed: the directory holds last generation's clip
# under a name the run intended to produce.  Merging them means either
# retrying clips that should be abandoned or abandoning clips that should be
# retried, and neither is visible afterwards.
FAILED_STATUSES = ("cut_failed",)


def file_sha256(path: pathlib.Path, limit: int = HASH_LIMIT) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        digest.update(handle.read(limit))
    return digest.hexdigest()


def clip_record(clip_dir: pathlib.Path) -> Dict[str, object]:
    """One clip as the census sees it, with every failure named rather than skipped.

    A clip directory with no readable ``meta.json`` is not absent and is not
    fine; it gets a status of its own and makes the run exit non-zero.  The
    alternative -- dropping it -- would quietly shrink the baseline, and a
    clip missing from the baseline is indistinguishable from a clip that is new.
    """
    record: Dict[str, object] = {"clip": clip_dir.name}
    meta_path = clip_dir / "meta.json"
    if not meta_path.is_file():
        record["status"] = "no_meta"
        return record
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except ValueError as error:
        record["status"] = "unreadable_meta"
        record["error"] = str(error)
        return record

    record["status"] = "ok"
    for key in ("num_frames", "fps", "source_fps", "source_frame_span",
                "span_end_reason", "max_seconds", "min_frames",
                "source_avg_frame_rate", "source_r_frame_rate",
                "cfr_normalized_from"):
        if key in meta:
            record[key] = meta[key]
    # source_fps only exists on clips cut after the fix.  Its absence is the
    # signature of the old ingest, which wrote the 30.0 constant into "fps" and
    # recorded no measurement at all, so it is worth stating rather than
    # leaving as a missing key.
    record["records_source_fps"] = "source_fps" in meta
    # ...but recording *a* rate is not the same as having cut from a file where
    # frame numbers name instants.  The 2026-08-19 re-cut wrote source_fps from
    # r_frame_rate alone, and on a variable-rate container that is the rate that
    # is wrong: one upload measured 2026-08-24 reads r 60.00 against avg 30.07,
    # so the cut took 803 source frames (26.7 s of content) and called them
    # 13.40 s.  Nineteen clips were made about 2x worse that way while gaining
    # the source_fps field that says they were repaired.  The two rates are
    # therefore recorded separately and their agreement is the thing asked.
    average = meta.get("source_avg_frame_rate")
    container = meta.get("source_r_frame_rate")
    if average is None or container is None:
        # Not "they disagree" and not "they agree" -- nobody looked.  A clip cut
        # before 2026-08-25 has no reading, and the gate must not read that as a
        # pass.
        record["rates_agree"] = None
    else:
        from tools.ingest_wild_uploads import is_variable_rate
        record["rates_agree"] = not is_variable_rate(float(average), float(container))

    video = clip_dir / "clip.mp4"
    if video.is_file():
        record["clip_sha256_1mb"] = file_sha256(video)
        record["clip_bytes"] = video.stat().st_size
    else:
        # Expected for the released corpus: the tree is pulled without videos.
        # Recorded as a named absence so the comparison can refuse to call
        # these clips byte-identical.
        record["clip_sha256_1mb"] = None
        record["video_local"] = False
    return record


def produced_clips(ingest_root: pathlib.Path) -> Dict[str, set]:
    """Per upload, the clip names its most recent ingest run ended up with.

    Enumerating the directory cannot answer this and that is the whole point.
    Nothing deletes a clip directory, so an upload that used to yield four clips
    and now yields two leaves the other two on disk looking exactly like the
    ones that are still current -- same name, same meta.json, same everything
    except that no run produces them any more.  A census built on ``glob``
    therefore reports zero orphans on a corpus that just created hundreds, which
    is what this one did on 2026-08-19 before this function existed.

    The manifest row is the run's own statement of what it produced, so that is
    what is asked.  Uploads with no row at all are returned as ``None`` rather
    than as an empty set, because "produced nothing" and "never ran" are
    opposite facts that an empty set would merge.
    """
    from tools.ingest_wild_uploads import manifest_rows

    produced: Dict[str, set] = {}
    failed: Dict[str, set] = {}
    for upload, row in manifest_rows(ingest_root).items():
        clips = row.get("clips", [])
        produced[upload] = {c["clip"] for c in clips
                            if c.get("status") in PRODUCED_STATUSES}
        failed[upload] = {c["clip"] for c in clips
                          if c.get("status") in FAILED_STATUSES}
    return produced, failed


def census(uploads_dir: pathlib.Path, ingest_root: pathlib.Path,
           probe_rate: bool) -> Dict[str, object]:
    from tools.ingest_wild_uploads import probe_fps

    produced, failed_cuts = produced_clips(ingest_root)

    uploads = sorted(p for p in uploads_dir.iterdir()
                     if p.suffix.lower() in (".mp4", ".mov", ".webm", ".mkv"))
    if not uploads:
        raise SystemExit("no uploads under {}".format(uploads_dir))

    clips: Dict[str, object] = {}
    per_upload: Dict[str, object] = {}
    problems: List[Dict[str, object]] = []

    for index, upload in enumerate(uploads, start=1):
        stem = upload.stem
        entry: Dict[str, object] = {"upload": stem, "source": str(upload)}
        if probe_rate:
            rate = probe_fps(upload)
            entry["source_fps"] = round(rate, 6) if rate else None
            if not rate:
                problems.append({"upload": stem, "status": "unreadable_frame_rate"})
        # The name is the join key everywhere downstream, so the census
        # enumerates by name pattern rather than by anything inside the dirs.
        found = sorted(d for d in ingest_root.glob(stem + "__clip*") if d.is_dir())
        names = []
        for clip_dir in found:
            record = clip_record(clip_dir)
            record["upload"] = stem
            if stem in produced:
                record["produced_now"] = clip_dir.name in produced[stem]
                record["cut_failed"] = clip_dir.name in failed_cuts.get(stem, ())
            else:
                record["produced_now"] = None
                problems.append({"upload": stem, "clip": clip_dir.name,
                                 "status": "no_ingest_manifest_row"})
            clips[clip_dir.name] = record
            names.append(clip_dir.name)
            if record["status"] != "ok":
                problems.append({"clip": clip_dir.name, "status": record["status"]})
        entry["clips"] = names
        per_upload[stem] = entry
        if index % 200 == 0 or index == len(uploads):
            print("[{}/{}] uploads scanned, {} clips".format(
                index, len(uploads), len(clips)), flush=True)

    return {
        "generated_by": "tools/refix_wild_fps_clips_census.py",
        "uploads_dir": str(uploads_dir),
        "ingest_root": str(ingest_root),
        "uploads": len(uploads),
        "clip_count": len(clips),
        "videos_local": sum(1 for r in clips.values()
                            if r.get("clip_sha256_1mb") is not None),
        "uploads_index": per_upload,
        "clips": clips,
        "problems": problems,
    }


def compare(before: Dict[str, object], after: Dict[str, object]) -> Dict[str, object]:
    """Baseline against current, keeping "differs" and "not compared" apart.

    ``changed`` is a proof of difference; ``unchanged_span`` is not a proof of
    sameness.  Only clips where both censuses hold a hash land in
    ``bytes_identical`` or ``bytes_differ``; the rest are named in
    ``unverified_bytes`` and belong to whoever decides whether to re-derive.
    """
    old = before.get("clips", {})
    new = after.get("clips", {})
    old_names, new_names = set(old), set(new)

    # Two ways to stop being part of the corpus, and only the first is visible
    # in a directory listing.  A name can vanish from the tree (it never does
    # here -- nothing deletes), or it can still be on disk while no run produces
    # it.  The second is the one this corpus actually produces.
    vanished = old_names - new_names
    unproduced = {name for name, row in new.items() if row.get("produced_now") is False}
    failed = {name for name, row in new.items() if row.get("cut_failed")}
    orphaned = sorted((vanished | unproduced) - failed)
    created = sorted(n for n in (new_names - old_names)
                     if new[n].get("produced_now") is not False)
    common = sorted(old_names & new_names)

    changed, unchanged_span = [], []
    bytes_identical, bytes_differ, unverified = [], [], []
    orphan_set = set(orphaned) | failed
    for name in common:
        if name in orphan_set:
            # No longer produced.  Comparing its span to the baseline's would
            # read as "unchanged", which is true and completely misleading.
            continue
        was, now = old[name].get("source_frame_span"), new[name].get("source_frame_span")
        if was != now:
            changed.append({"clip": name, "was": was, "now": now})
        else:
            unchanged_span.append(name)
        old_hash, new_hash = old[name].get("clip_sha256_1mb"), new[name].get("clip_sha256_1mb")
        if old_hash and new_hash:
            (bytes_identical if old_hash == new_hash else bytes_differ).append(name)
        else:
            unverified.append(name)

    # Which uploads stopped producing a clip they used to produce.  Reported
    # separately because an orphan is cheap to explain at the upload level
    # ("this one re-split") and confusing at the clip level.
    orphan_uploads = sorted({(old.get(name) or new.get(name) or {}).get("upload")
                             for name in orphaned} - {None})

    return {
        "failed_to_produce": sorted(failed),
        "baseline_clip_count": len(old_names),
        "current_clip_count": len(new_names),
        "orphaned": orphaned,
        "orphan_uploads": orphan_uploads,
        "created": created,
        "changed": changed,
        "unchanged_span": unchanged_span,
        "bytes_identical": bytes_identical,
        "bytes_differ": bytes_differ,
        "unverified_bytes": unverified,
    }


def write_lines(path: pathlib.Path, names: List[str]) -> None:
    """One name per line, staged then renamed.

    A list read while it is being written is the "file exists and is empty"
    failure this repo was bitten by; an empty exclusion list in particular reads
    exactly like "nothing to exclude".
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(path.suffix + ".tmp")
    staging.write_text("".join(name + "\n" for name in names), encoding="utf-8")
    staging.replace(path)
    print("wrote {} ({} names)".format(path, len(names)))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--uploads", type=pathlib.Path, required=True,
                        help="directory of upload videos this census covers")
    parser.add_argument("--ingest", type=pathlib.Path, required=True,
                        help="ingest root holding <upload>__clipNNN directories")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--compare", type=pathlib.Path, default=None,
                        help="an earlier census to diff against; adds the "
                             "orphan list this corpus must be consumed through")
    parser.add_argument("--orphan-list", type=pathlib.Path, default=None,
                        help="write the orphaned clip names here, one per line: "
                             "the objects these credentials cannot delete and "
                             "no consumer may read.  Requires --compare")
    parser.add_argument("--audit-list", type=pathlib.Path, default=None,
                        help="write the clip names this corpus currently "
                             "produces here, one per line -- the input to "
                             "tools/audit_clip_freshness.py.  Orphans are not "
                             "in it: the question for them is not whether they "
                             "are fresh but who is still pointing at them")
    parser.add_argument("--no-probe-rate", action="store_true",
                        help="skip measuring each upload's frame rate (one "
                             "ffprobe per upload); the rate is the reason this "
                             "clip set exists, so it is measured by default")
    args = parser.parse_args(argv)

    if not args.uploads.is_dir():
        raise SystemExit("no upload directory at {}".format(args.uploads))
    if not args.ingest.is_dir():
        raise SystemExit("no ingest root at {}".format(args.ingest))

    result = census(args.uploads, args.ingest, probe_rate=not args.no_probe_rate)

    if args.compare is not None:
        baseline = json.loads(args.compare.read_text(encoding="utf-8"))
        result["compare"] = compare(baseline, result)
        result["compare"]["baseline"] = str(args.compare)
    elif args.orphan_list is not None:
        raise SystemExit("--orphan-list needs --compare: an orphan is defined "
                         "against a baseline, and without one this would write "
                         "an empty list that reads like 'none'")

    if args.orphan_list is not None:
        write_lines(args.orphan_list, result["compare"]["orphaned"])
    if args.audit_list is not None:
        # Orphans are left out.  The question for a clip nobody produces any
        # more is not whether its artifacts are fresh -- they are, they were
        # built from its own old bytes and agree with them -- but who is still
        # pointing at it.  Auditing them would return "fresh" for every one and
        # that reading is exactly the wrong thing to hand the next stage.
        excluded = set((result.get("compare") or {}).get("orphaned") or ())
        write_lines(args.audit_list,
                    [name for name in sorted(result["clips"]) if name not in excluded])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Staged then renamed: a census read while it is being written is exactly
    # the "file exists and is empty" failure this repo keeps meeting.
    staging = args.output.with_suffix(args.output.suffix + ".tmp")
    staging.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n",
                       encoding="utf-8")
    staging.replace(args.output)

    print("census: {} uploads -> {} clips ({} with local video)".format(
        result["uploads"], result["clip_count"], result["videos_local"]))
    rates = [e.get("source_fps") for e in result["uploads_index"].values()]
    measured = [r for r in rates if r]
    if measured:
        off_grid = sum(1 for r in measured if abs(r - 30.0) > 0.01)
        print("source frame rate: {} measured, {} not 30 fps".format(
            len(measured), off_grid))
    if result["problems"]:
        print("PROBLEMS: {}".format(len(result["problems"])))
        for problem in result["problems"][:20]:
            print("  {}".format(problem))

    diff = result.get("compare")
    if diff:
        print("--- against {} ---".format(diff["baseline"]))
        print("baseline {} clips -> now {}".format(
            diff["baseline_clip_count"], diff["current_clip_count"]))
        print("orphaned (undeletable on OSS, exclude by name) : {} from {} uploads".format(
            len(diff["orphaned"]), len(diff["orphan_uploads"])))
        print("cuts that failed (work still owed, NOT orphans): {}".format(
            len(diff["failed_to_produce"])))
        print("created                                        : {}".format(len(diff["created"])))
        print("span changed (proved different)                : {}".format(len(diff["changed"])))
        print("span unchanged                                 : {}".format(len(diff["unchanged_span"])))
        print("bytes identical / differ / never compared      : {} / {} / {}".format(
            len(diff["bytes_identical"]), len(diff["bytes_differ"]),
            len(diff["unverified_bytes"])))
        if diff["unverified_bytes"]:
            print("  NOTE: 'never compared' is not 'unchanged'.  The baseline held no "
                  "local video for these, so nothing weighed their bytes.")
        for item in diff["orphaned"][:20]:
            print("  orphan {}".format(item))

    print("wrote {}".format(args.output))
    return 1 if result["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
