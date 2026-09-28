"""Fetch the AIST videos backing a released split, for visual atomic discovery.

The paper's segmentation (Alg. 1) runs an I3D encoder over the dance **video**
paired with each motion sequence, then cuts at self-similarity cluster
boundaries.  AIST++ ships motion annotations but not the footage, and this
machine has none: ``Lodge/aist_plusplus`` holds keypoints2d and wav only.  That
is why the local kinematic discovery had no visual input at all -- and it is
measurably music-orthogonal at every granularity (see worklog.md).

Release sequence names carry the ``cAll`` token, which is AIST++'s annotation-
level identifier covering every camera.  The video database stores one file per
physical camera, so a camera has to be chosen; ``--camera`` defaults to c01.
The choice is recorded in the manifest, because segmentation derived from one
viewpoint is not automatically valid for another.

Downloads are resumable and verified by length: a partial file from an
interrupted run is continued rather than silently accepted, and a short file is
reported instead of being handed to the encoder as if it were complete.

This fetches only what the release actually references, so the volume tracks the
split rather than the whole 13 TB database.
"""

import argparse
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_URL = "https://aistdancedb.ongaaccel.jp/v1.0.0/video/10M"
_SLICE = re.compile(r"_slice\d+$")
_CAMERA = re.compile(r"_c[A-Za-z0-9]+_")


def release_sequences(data_root, splits):
    """Unique sequence names referenced by the release, slices collapsed."""
    names = set()
    for split in splits:
        path = Path(data_root) / split / "names.json"
        if not path.is_file():
            continue
        for name in json.loads(path.read_text(encoding="utf-8")):
            names.add(_SLICE.sub("", name.split("/")[-1]))
    return sorted(names)


def interleave_by_genre(sequences):
    """Round-robin the sequence list across genres.

    Sequence ids sort into genre blocks (every ``gBR_*`` before every ``gHO_*``),
    so a partial download covers two or three of the ten genres.  That is fine
    for building a corpus and useless as a *reference*: scoring a ten-class
    labeller on three classes measures nothing, because the confusions it should
    be penalised for never had the chance to occur.  Interleaving makes any
    prefix of the download a balanced sample.
    """
    import collections

    buckets = collections.OrderedDict()
    for name in sequences:
        buckets.setdefault(name.split("_", 1)[0], []).append(name)
    out = []
    index = 0
    while any(len(v) > index for v in buckets.values()):
        for names in buckets.values():
            if len(names) > index:
                out.append(names[index])
        index += 1
    return out


def video_name(sequence, camera):
    """Swap the annotation-level camera token for a physical camera."""
    if not _CAMERA.search(sequence):
        raise ValueError("no camera token in {!r}".format(sequence))
    return _CAMERA.sub("_{}_".format(camera), sequence, count=1)


def remote_size(url, timeout):
    """Content-Length of the *final* response after redirects.

    The AIST database answers with a 301 to storage.repository.aist.go.jp, and
    ``curl -I -L`` prints the headers of every hop.  Taking the first
    Content-Length would read the redirect body's length -- 169 bytes -- and
    mark every fully downloaded video as truncated, so take the last one.
    """
    result = subprocess.run(
        ["curl", "-sSLI", "--max-time", str(timeout), url],
        capture_output=True, text=True,
    )
    size = None
    for line in result.stdout.splitlines():
        if line.lower().startswith("content-length:"):
            try:
                size = int(line.split(":", 1)[1].strip())
            except ValueError:
                continue
    return size


def fetch(sequence, camera, output_dir, timeout, verify):
    stem = video_name(sequence, camera)
    target = output_dir / "{}.mp4".format(stem)
    url = "{}/{}.mp4".format(BASE_URL, stem)

    expected = remote_size(url, timeout) if verify else None
    if target.is_file() and expected is not None and target.stat().st_size == expected:
        return {"sequence": sequence, "video": stem, "status": "cached",
                "bytes": target.stat().st_size}

    # -C - resumes a partial file; --retry covers transient upstream failures.
    result = subprocess.run(
        ["curl", "-sSL", "-C", "-", "--retry", "3", "--retry-delay", "3",
         "--max-time", str(timeout), url, "-o", str(target)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return {"sequence": sequence, "video": stem, "status": "error",
                "reason": (result.stderr or "curl exit {}".format(result.returncode)).strip()[:200]}

    size = target.stat().st_size if target.is_file() else 0
    if expected is not None and size != expected:
        # Report rather than delete: a truncated file is evidence for triage,
        # and the next run resumes it.
        return {"sequence": sequence, "video": stem, "status": "incomplete",
                "bytes": size, "expected": expected}
    return {"sequence": sequence, "video": stem, "status": "ok", "bytes": size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--camera", default="c01",
                        help="physical camera substituted for the cAll token")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--limit", type=int, default=None,
                        help="fetch only the first N sequences, for a smoke run")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip the length check (one fewer request per file)")
    parser.add_argument("--passes", type=int, default=3,
                        help="sweeps over still-failing sequences; upstream rate "
                             "limiting makes most failures transient")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--stratify-by-genre", action="store_true",
                        help="round-robin the download across the ten genres, so "
                             "an interrupted or --limit-ed fetch is still a usable "
                             "reference set")
    args = parser.parse_args()

    sequences = release_sequences(args.data_root, args.splits)
    if args.stratify_by_genre:
        sequences = interleave_by_genre(sequences)
    if args.limit is not None:
        sequences = sequences[: args.limit]
    if not sequences:
        raise SystemExit("no sequences found under {}".format(args.data_root))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("fetching {} videos (camera {}) into {}".format(
        len(sequences), args.camera, args.output_dir), flush=True)

    # Failures here are dominated by upstream rate limiting, not missing files:
    # a sequence that errors under load returns 206 on a later request.  Sweep
    # the stragglers in extra passes rather than leaving gaps for a human.
    outstanding = list(sequences)
    results = {}
    for attempt in range(1, args.passes + 1):
        if not outstanding:
            break
        if attempt > 1:
            print("\nretry pass {}: {} sequence(s)".format(attempt, len(outstanding)), flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(fetch, sequence, args.camera, args.output_dir,
                            args.timeout, not args.no_verify): sequence
                for sequence in outstanding
            }
            for done, future in enumerate(as_completed(futures), start=1):
                record = future.result()
                results[record["sequence"]] = record
                if done % 25 == 0 or record["status"] not in ("ok", "cached"):
                    print("  [{}/{}] {} {}".format(
                        done, len(outstanding), record["status"], record["video"]), flush=True)
        outstanding = [
            sequence for sequence in outstanding
            if results[sequence]["status"] not in ("ok", "cached")
        ]

    records = list(results.values())

    counts = {}
    for record in records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    total = sum(record.get("bytes", 0) for record in records)
    print("\n{}  total {:.1f} GB".format(counts, total / 1e9))

    manifest = args.manifest or (args.output_dir / "manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "data_root": str(args.data_root),
                "splits": args.splits,
                # Recorded because a segmentation is viewpoint-specific.
                "camera": args.camera,
                "base_url": BASE_URL,
                "sequences": len(sequences),
                "status_counts": counts,
                "total_bytes": total,
                "records": sorted(records, key=lambda item: item["sequence"]),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print("manifest -> {}".format(manifest))
    return 0 if counts.get("error", 0) == 0 and counts.get("incomplete", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
