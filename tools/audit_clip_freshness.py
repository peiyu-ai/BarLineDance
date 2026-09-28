#!/usr/bin/env python3
"""Which derived artifacts were built from bytes the clip no longer has.

Every stage that consumes a clip records the hash of the video it actually
read -- ``run_gvhmr_extract`` writes ``video_sha256_1mb`` into the converted
clip's ``metadata.json``, the S3D extractor writes the same field into every
``.npz``.  This tool weighs those records against the clip's current bytes and
names the disagreements.  It is the R0 gate of 2026-08-19 ("still disagreeing
0") turned into a tool instead of a command someone happened to type.

**Why presence is not the gate.**  The published corpus already carried a
generation of features whose objects existed, were the right length, and joined
to everything downstream -- and came from an older cut.  3,929 clips looked
stale by frame count; 3,991 were stale by content, and the 62 in between were
the ones whose old cut happened to be the same length.  A gate that asks "is
there a feature file for this clip" answers yes for all of them.  This gate asks
"was that file built from these bytes", which is the question, and it can fail.

**The instrument, and its positive control.**  A clip's current bytes are its
first megabyte -- the same window every stage already hashes, so the comparison
is against a number the stage itself wrote rather than a re-derivation of it.
Local ``clip.mp4`` is used when the ingest tree still has one; otherwise the
megabyte is fetched with ``ossutil cp --range=0-1048575``, which this bucket
serves in ~0.06 s even though it answers the SDK's LIST and ranged GET with a
bare 502.  Checked before being trusted: on a clip nothing has touched, the
ranged bytes, the 3D record and the S3D record are the same hash.

**Music arrived last, and could not have been added earlier.**  3D and S3D each
write the hash of the video they read beside their own output, so the record
side is in the artifact.  The 35-D music features are a bare ``.npy`` with
nowhere to put one, and the bundle that carries them recorded only
``music_sha256`` -- the hash of the derived array -- so nothing inside it said
which audio the array came from.  ``build_wild_performance_bundle`` carries
``source_audio_sha256`` through from 2026-08-25; before that the join did not
exist and this gate could not have been written, which is why 234 of 1,575
evaluation rows (14.9%) could hold another generation's music while every hash
in the bundle agreed with every other one.  The record side for ``music`` is
therefore a manifest (``--music-manifest``), not an artifact, and the clip side
is the whole of ``audio.wav`` rather than its first megabyte -- that is what the
extractor hashed, and a gate must compare the quantity the stage recorded.

Output is a manifest, because this corpus can no longer be consumed by
enumeration.  These credentials can PUT over an existing key but cannot delete
one, so a clip that stops being produced leaves an object behind forever and
any directory listing keeps returning it.  What this writes is the list a stage
should re-derive, and the list nothing should read.

Usage::

    audit_clip_freshness.py --clips stale_candidates.txt --output audit.json
    audit_clip_freshness.py --all --output audit.json --workers 16
    audit_clip_freshness.py --clips c.txt --stages music --output audit.json \
        --music-manifest <bundle>/sequences.jsonl
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools import asset_io  # noqa: E402

HASH_LIMIT = 1 << 20
INGEST = "data/wild_ingest_v1"
# Overridable because the fps refix cuts into a scratch tree before it replaces
# the released one, and because a gate that cannot be pointed at a known-bad
# corpus cannot be shown to fail.
_ingest_root = INGEST
# Where the *record* side is read from.  "store" is the question this gate
# exists to ask; see ``read_record_bytes``.
_records = "store"


CONVERTED = "data/wild3d/ingest_v1_converted"
FEATURES = "data/wild_visual_s3d"
OSSUTIL = "/opt/data-infra/ossutil64"
REPO = pathlib.Path(__file__).resolve().parents[1]


def read_record_bytes(relative: str) -> bytes:
    """The derived artifact's bytes, from the object store rather than a cache.

    ``asset_io.read_bytes`` resolves local-first, and three of this repo's
    trees are parked under ``/cache`` and symlinked back.  A stage publishes to
    the store; the parked copy does not follow.  So on 2026-08-20, after stage
    B re-extracted 317 re-cut clips, this audit read the *previous*
    generation's ``metadata.json`` out of the cache and reported that the
    re-extraction had changed nothing: ``stale=419`` before and after, while
    the store's record matched the clip's current bytes for 10 of 10 sampled
    and the cache matched for 0 of 10.

    A gate that cannot see the work it is meant to verify is worse than no
    gate, which is the shape CLAUDE.md §2 opens with.  So the record comes from
    the store, and the cache is available only when explicitly asked for --
    where it answers a different and also real question, "does this pod's read
    cache agree with the clip", i.e. whether a local mirror is safe to consume.
    """
    if _records == "local":
        return asset_io.read_bytes(relative)
    return asset_io._fetch_via_ossutil(relative)              # noqa: SLF001


def sha256_head(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def clip_head(stem: str) -> Optional[str]:
    """The first megabyte of this clip's video, local if there, else ranged.

    Returns ``None`` when the clip has no video in either place -- reported as
    its own status rather than folded into "fresh", because a missing video
    means the comparison did not happen, not that it passed.
    """
    root = pathlib.Path(_ingest_root)
    local = (root if root.is_absolute() else REPO / root) / stem / "clip.mp4"
    if local.is_file():
        with open(local, "rb") as handle:
            return sha256_head(handle.read(HASH_LIMIT))
    if _ingest_root != INGEST:
        # Pointed at a scratch tree on purpose.  Falling back to the object
        # store here would compare against the *released* generation of a clip
        # the caller is deliberately not looking at -- the exact confusion this
        # audit exists to catch, reintroduced by the audit itself.
        return None

    relative = "{}/{}/clip.mp4".format(INGEST, stem)
    url = "oss://" + asset_io.remote_path(relative)
    with tempfile.TemporaryDirectory() as workspace:
        target = pathlib.Path(workspace) / "head.bin"
        argv = [OSSUTIL, "cp", url, str(target),
                "--range=0-{}".format(HASH_LIMIT - 1), "-f"]
        config = REPO / "ossutilconfig"
        if config.is_file():
            argv += ["-c", str(config)]
        completed = subprocess.run(argv, capture_output=True, text=True)
        if completed.returncode != 0 or not target.is_file():
            return None
        return sha256_head(target.read_bytes())


AUDIO = "audio.wav"
# Which stage reads which half of the clip.  3D and S3D consume the picture and
# hash its first megabyte; the music extractor consumes audio.wav and hashes the
# whole file.  Comparing against the wrong quantity would make every music
# verdict "stale" while reading exactly like a measurement.
STAGE_INPUT = {"3d": "video", "s3d": "video", "music": "audio"}


def clip_audio_hash(stem: str) -> Optional[str]:
    """The whole of this clip's ``audio.wav``, local if there, else fetched.

    Whole, not the first megabyte.  ``extract_wild_music_features`` writes
    ``_sha256_file(audio_path)`` over the entire file, and a gate that compares
    a different window against that number disagrees with everything -- an
    always-failing gate is as uninformative as an always-passing one, and looks
    more convincing.  These are ~600 KB, so the whole file costs about what the
    video's megabyte does.
    """
    root = pathlib.Path(_ingest_root)
    local = (root if root.is_absolute() else REPO / root) / stem / AUDIO
    if local.is_file():
        return hashlib.sha256(local.read_bytes()).hexdigest()
    if _ingest_root != INGEST:
        return None
    relative = "{}/{}/{}".format(INGEST, stem, AUDIO)
    url = "oss://" + asset_io.remote_path(relative)
    with tempfile.TemporaryDirectory() as workspace:
        target = pathlib.Path(workspace) / AUDIO
        argv = [OSSUTIL, "cp", url, str(target), "-f"]
        config = REPO / "ossutilconfig"
        if config.is_file():
            argv += ["-c", str(config)]
        completed = subprocess.run(argv, capture_output=True, text=True)
        if completed.returncode != 0 or not target.is_file():
            return None
        return hashlib.sha256(target.read_bytes()).hexdigest()


# sequence_id -> the audio hash the music features were built from, loaded once
# from --music-manifest.  Empty until load_music_manifest runs, and ``music`` is
# refused as a stage without it rather than reporting every clip absent.
_music_records: Dict[str, str] = {}


def stem_of(record: Dict) -> Optional[str]:
    """The ingest clip stem a bundle row refers to.

    Bundles name a row ``wild_v4:<upload>:clipNNN`` and the ingest names the
    same clip ``<upload>__clipNNN``.  ``legacy_source_name`` already holds the
    second form where it exists; the split is the fallback, and a row that
    parses as neither is skipped by the caller rather than guessed at.
    """
    legacy = record.get("legacy_source_name")
    if legacy:
        return str(legacy)
    key = record.get("sequence_id") or record.get("recording_id") or ""
    parts = str(key).split(":")
    return "{}__{}".format(parts[1], parts[2]) if len(parts) == 3 else None


def load_music_manifest(path: pathlib.Path) -> Dict[str, str]:
    """``{clip stem: source_audio_sha256}`` from a bundle or audio manifest.

    Two shapes are accepted because the field lives in two places: at the top
    level of a performance bundle's ``sequences.jsonl`` (from 2026-08-25) and
    under ``audio_feature`` in the audio bundle's ``sequences_audio.jsonl``,
    which is where it has always been.  Rows with no hash are *counted*, not
    dropped -- a bundle built before the field existed produces an empty map,
    and "the manifest carries no provenance" must not read as "no clip is
    stale".
    """
    records, without = {}, 0
    for line in path.open(encoding="utf-8"):
        try:
            record = json.loads(line)
        except ValueError:
            continue
        stem = stem_of(record)
        if not stem:
            continue
        digest = (record.get("source_audio_sha256")
                  or (record.get("audio_feature") or {}).get("source_audio_sha256")
                  or ((record.get("provenance") or {}).get("source_audio_sha256")))
        if digest:
            records[stem] = str(digest)
        else:
            without += 1
    if not records:
        raise SystemExit(
            "{} carries no source_audio_sha256 on any of its rows ({} rows "
            "seen).  It predates the field, so a music audit against it would "
            "report every clip's music as absent -- which reads like 'nothing "
            "to re-derive'.  Rebuild the bundle, or point --music-manifest at "
            "the audio bundle's sequences_audio.jsonl.".format(path, without))
    if without:
        print("music manifest: {} row(s) carry a source audio hash, {} do not"
              .format(len(records), without), flush=True)
    return records


def recorded_music(stem: str) -> Optional[str]:
    """The audio hash the 35-D features were extracted from, per the manifest."""
    return _music_records.get(stem)


# Which stems have a record at all, listed once per prefix instead of probed
# per clip.  Stage B's docstring already spelled this out -- "a listing of two
# prefixes costs two round trips, where 16,715 HEAD requests would cost minutes
# before any GPU work started" -- and this tool ignored it: 909 of the 2,207
# clips in the first real run had no 3D or no features yet (they are new names
# the re-cut created), and each one paid an ossutil miss, three retries and an
# SDK call that this bucket answers with a 502.
_have: Dict[str, set] = {}


def have(stage: str) -> set:
    if stage in _have:
        return _have[stage]
    # Enumerated from the same side the record is read from.  A parked local
    # tree lists the *previous* generation's names, so a clip whose 3D was
    # published under a name the re-cut created reads as "absent" -- which the
    # verdict below reports as "no record", not as "stale".  Under --records
    # store this is one prefix listing (~7 s for 124,195 keys).
    if stage == "3d":
        local = REPO / CONVERTED
        if _records == "local" and local.is_dir():
            stems = {d.name for d in local.iterdir()
                     if (d / "metadata.json").is_file()}
        else:
            names = asset_io.list_prefix(CONVERTED)
            stems = {n.split("/")[0] for n in names if n.endswith("/metadata.json")}
    else:
        local = REPO / FEATURES
        if _records == "local" and local.is_dir():
            stems = {p.name[: -len(".npz")] for p in local.iterdir()
                     if p.name.endswith(".npz")}
        else:
            names = asset_io.list_prefix(FEATURES)
            stems = {n[: -len(".npz")] for n in names
                     if n.endswith(".npz") and "/" not in n}
    _have[stage] = stems
    return stems


def recorded_3d(stem: str) -> Optional[str]:
    """What GVHMR says it read.  Absent metadata means no 3D, not a match.

    The "is there one at all" half comes from a single listing, because trying
    to read an object that is not there costs an ossutil miss, three retries and
    an SDK call this bucket answers with a bare 502.
    """
    if stem not in have("3d"):
        return None
    relative = "{}/{}/metadata.json".format(CONVERTED, stem)
    try:
        meta = json.loads(read_record_bytes(relative).decode("utf-8"))
    except Exception:                                         # noqa: BLE001
        return None
    extract = meta.get("extract_meta") or {}
    provenance = meta.get("gvhmr_run_provenance") or {}
    return extract.get("video_sha256_1mb") or provenance.get("video_sha256_1mb")


def recorded_s3d(stem: str) -> Optional[str]:
    """What the S3D extractor says it read; absence answered from the listing."""
    if stem not in have("s3d"):
        return None

    import io

    import numpy as np

    relative = "{}/{}.npz".format(FEATURES, stem)
    local = REPO / relative
    try:
        source = (local if (_records == "local" and local.is_file())
                  else io.BytesIO(read_record_bytes(relative)))
        with np.load(source, allow_pickle=False) as bundle:
            meta = json.loads(str(bundle["meta"]))
    except Exception:                                         # noqa: BLE001
        return None
    return meta.get("video_sha256_1mb")


def judge(stem: str, stages: List[str]) -> Dict[str, object]:
    row: Dict[str, object] = {"clip": stem}
    # Each half of the clip is hashed at most once, and only if some requested
    # stage reads it -- a music-only audit must not pay for the video.
    needed = {STAGE_INPUT[stage] for stage in stages}
    current: Dict[str, Optional[str]] = {}
    if "video" in needed:
        current["video"] = clip_head(stem)
        row["clip_sha256_1mb"] = current["video"]
    if "audio" in needed:
        current["audio"] = clip_audio_hash(stem)
        row["clip_audio_sha256"] = current["audio"]
    if all(value is None for value in current.values()):
        # Nothing of this clip could be read, so no stage was compared.  The
        # status name is kept for the consumers that already read it.
        row["status"] = "video_missing"
        return row
    readers = {"3d": recorded_3d, "s3d": recorded_s3d, "music": recorded_music}
    verdicts = {}
    for stage in stages:
        mine = current[STAGE_INPUT[stage]]
        if mine is None:
            # The other half of the clip was readable and this one was not.
            # Its own verdict, because neither "fresh" nor "stale" happened.
            verdicts[stage] = "clip_bytes_missing"
            row["{}_recorded".format(stage)] = None
            continue
        recorded = readers[stage](stem)
        if recorded is None:
            verdicts[stage] = "absent"
        elif recorded == mine:
            verdicts[stage] = "fresh"
        else:
            verdicts[stage] = "stale"
        row["{}_recorded".format(stage)] = recorded
    row["verdicts"] = verdicts
    row["status"] = "ok"
    return row


def all_stems() -> List[str]:
    names = asset_io.list_prefix(CONVERTED)
    stems = sorted({name.split("/")[0] for name in names if "/" in name})
    return stems


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--clips", type=pathlib.Path,
                        help="file of clip stems, one per line ('-' for stdin)")
    source.add_argument("--all", action="store_true",
                        help="every clip that has 3D; enumerated from the "
                             "converted tree, which is the only place a clip "
                             "with derived artifacts is guaranteed to appear")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--stages", default="3d,s3d",
                        help="comma-separated: 3d, s3d, music")
    parser.add_argument("--music-manifest", type=pathlib.Path, default=None,
                        help="a performance bundle's sequences.jsonl or an "
                             "audio bundle's sequences_audio.jsonl, read for "
                             "source_audio_sha256.  Required by --stages music: "
                             "the music features are a bare .npy with nowhere "
                             "to record what they were built from")
    parser.add_argument("--workers", type=int, default=16,
                        help="parallel hash fetches; the megabyte is ~0.06 s "
                             "each, so this is the difference between one "
                             "minute and fifteen over the full corpus")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--records", choices=("store", "local"), default="store",
                        help="where the derived artifact's recorded hash is "
                             "read from.  'store' is the gate: a parked local "
                             "copy does not follow a publish, so it cannot see "
                             "the re-derivation it is meant to verify. 'local' "
                             "asks the other question -- whether this pod's "
                             "read cache agrees with the clip")
    parser.add_argument("--ingest-root", default=INGEST,
                        help="where <clip>/clip.mp4 lives; the released tree by "
                             "default, a scratch tree while a re-cut is staged")
    args = parser.parse_args(argv)

    global _ingest_root, _records
    _ingest_root = args.ingest_root
    _records = args.records

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in STAGE_INPUT]
    if unknown:
        raise SystemExit("unknown stage(s): {}".format(", ".join(unknown)))
    if "music" in stages:
        if args.music_manifest is None:
            raise SystemExit(
                "--stages music needs --music-manifest.  Without it every clip "
                "would be reported as having no music record, which reads like "
                "'nothing to re-derive' and is the opposite of what it means.")
        global _music_records
        _music_records = load_music_manifest(args.music_manifest)

    if args.all:
        stems = all_stems()
    elif str(args.clips) == "-":
        stems = [line.strip() for line in sys.stdin if line.strip()]
    else:
        text = args.clips.read_text(encoding="utf-8")
        if text.lstrip().startswith(("{", "[")):
            # Accept a census as input, so the two tools chain without a
            # hand-written intermediate list.
            parsed = json.loads(text)
            stems = sorted(parsed["clips"]) if isinstance(parsed, dict) else sorted(parsed)
        else:
            stems = [line.strip() for line in text.splitlines() if line.strip()]
    if args.limit:
        stems = stems[: args.limit]
    if not stems:
        raise SystemExit("no clips to audit")
    print("auditing {} clips over stage(s) {}".format(len(stems), ", ".join(stages)),
          flush=True)

    rows = []
    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, row in enumerate(pool.map(lambda s: judge(s, stages), stems), start=1):
            rows.append(row)
            if index % 500 == 0 or index == len(stems):
                print("[{}/{}]".format(index, len(stems)), flush=True)

    stale = {stage: sorted(r["clip"] for r in rows
                           if r.get("verdicts", {}).get(stage) == "stale")
             for stage in stages}
    absent = {stage: sorted(r["clip"] for r in rows
                            if r.get("verdicts", {}).get(stage) == "absent")
              for stage in stages}
    unreadable = sorted(r["clip"] for r in rows if r["status"] == "video_missing")

    result = {
        "generated_by": "tools/audit_clip_freshness.py",
        "stages": stages,
        "clips_audited": len(rows),
        "stale": stale,
        "missing_derivative": absent,
        "video_missing": unreadable,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = args.output.with_suffix(args.output.suffix + ".tmp")
    staging.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    staging.replace(args.output)

    for stage in stages:
        print("stage {:<4} stale={:<6} missing={:<6}".format(
            stage, len(stale[stage]), len(absent[stage])))
    if unreadable:
        print("video unreadable (comparison did NOT happen): {}".format(len(unreadable)))
    disagreeing = sum(len(v) for v in stale.values())
    print("still disagreeing {}".format(disagreeing))
    print("wrote {}".format(args.output))
    # A clip whose video could not be read is not a pass: it is an unmade
    # measurement, and this exits non-zero for it too.
    return 1 if (disagreeing or unreadable) else 0


if __name__ == "__main__":
    raise SystemExit(main())
