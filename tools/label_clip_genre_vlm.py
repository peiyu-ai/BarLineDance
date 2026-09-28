#!/usr/bin/env python3
"""Dance genre per clip, and the reference system that says whether to believe it.

The paper's M3 pre-splits each atomic prototype by dance genre before the
summarizing LLM groups anything: "we first pre-split each atomic movement
prototype P_j by dance genre".  ``recluster_atomics_ingroup.py`` already
implements ``--genre-split``; what the wild corpus lacks is a genre *source*.
AIST names it in the sequence id, TikTok does not.

Two decisions here, and the second is the one that matters.

**Per clip, not per segment.**  A genre is a property of a dance, not of an
0.8-second fragment: a single step tells you very little, and asking the model
per segment would multiply the cost by ~17 while making each answer weaker.
The label attaches to the clip and every segment inherits it.

**Validated against AIST, not read and nodded at.**  AIST++ carries ground-truth
genre in the filename (``gBR_sBM_c01_...`` is break), so the same prompt run on
AIST clips produces a confusion matrix and an accuracy.  That is the gate: below
it, the wild corpus keeps ``genre_presplit: false`` and records the absence.
This is the same discipline as auditing the wild 3D against motion capture --
without a reference the number is readable but not judgeable, and a VLM's genre
would enter the corpus as a fact because nothing contradicted it.

The label is recorded as ``vlm_estimated_genre``.  It is never called ground
truth: on the wild corpus there is nothing to check it against, and a field
named for what produced it cannot later be mistaken for something measured.

Usage::

    # gate: does the prompt work where the answer is known?
    python3 tools/label_clip_genre_vlm.py --videos data/aist_videos \\
        --model third_party/QwenVL/Qwen3-VL-30B --output runs/genre_aist.json \\
        --reference aist --limit 300

    # apply: label the wild corpus
    python3 tools/label_clip_genre_vlm.py --videos data/wild3d/ingest_v1_videos \\
        --model third_party/QwenVL/Qwen3-VL-30B --output runs/genre_wild.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.caption_segments_vlm import (Captioner, load_frames,  # noqa: E402
                                        probe_frame_size, shard_of)

# AIST++'s ten genres, with the token its filenames use.  The wild corpus is
# labelled into the same ten so the pre-split means the same thing on both, plus
# an explicit escape: TikTok is not only these ten, and a model forced to choose
# among them would spread everything else across them silently.
GENRES = {
    "break": "gBR", "pop": "gPO", "lock": "gLO", "middle_hip_hop": "gMH",
    "la_hip_hop": "gLH", "house": "gHO", "waack": "gWA", "krump": "gKR",
    "street_jazz": "gJS", "ballet_jazz": "gJB",
}
OTHER = "other"
AIST_TOKEN_TO_GENRE = {token: name for name, token in GENRES.items()}

PROMPT = (
    "These frames are sampled across one dance video.\n"
    "Name the dance style. Choose exactly one value from this list:\n"
    "{choices}\n"
    'Use "other" if the dance is not any of them, or if there is no dance.\n'
    'Reply with JSON only: {{"genre": "<value>", "confidence": "high"|"low"}}'
)

FRAMES_PER_CLIP = 8


def build_prompt() -> str:
    return PROMPT.format(choices=", ".join(list(GENRES) + [OTHER]))


def parse_genre(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Pull the genre out of the model's reply, or report that it did not give one.

    Returning ``None`` rather than guessing is the point: an unparseable reply
    counted as ``other`` would be indistinguishable from a confident "this is
    not a dance", and the two mean opposite things about the corpus.
    """
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None, None
    try:
        payload = json.loads(match.group(0))
    except ValueError:
        return None, None
    genre = str(payload.get("genre", "")).strip().lower().replace(" ", "_")
    if genre not in GENRES and genre != OTHER:
        return None, None
    confidence = str(payload.get("confidence", "")).strip().lower() or None
    return genre, confidence


def aist_truth(stem: str) -> Optional[str]:
    """Ground-truth genre from an AIST++ sequence id, e.g. ``gBR_sBM_c01_...``."""
    token = stem.split("_", 1)[0]
    return AIST_TOKEN_TO_GENRE.get(token)


def frame_positions(total: int, count: int) -> List[int]:
    """Frames spread across the clip, avoiding the very first and last."""
    if total <= count:
        return list(range(total))
    step = total / (count + 1.0)
    return [int(round(step * (i + 1))) for i in range(count)]


def clip_frame_count(video: pathlib.Path) -> int:
    import cv2

    capture = cv2.VideoCapture(str(video))
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


def interleave_by_genre(videos: Sequence[pathlib.Path]) -> List[pathlib.Path]:
    """Round-robin the reference clips across genres before ``--limit`` cuts.

    Sequence ids sort into genre blocks, so the first 300 alphabetically are two
    classes -- and a ten-class accuracy computed on two classes measures
    nothing, because the confusions it should be penalised for never had the
    chance to occur.  The *download* is stratified for exactly this reason
    (``fetch_aist_videos.py --stratify-by-genre``); doing it there and not here
    moved the same mistake one level down, and it cost a 192-clip run to find.
    """
    buckets: Dict[str, List[pathlib.Path]] = {}
    for video in videos:
        buckets.setdefault(video.stem.split("_", 1)[0], []).append(video)
    out: List[pathlib.Path] = []
    index = 0
    while any(len(names) > index for names in buckets.values()):
        for names in buckets.values():
            if len(names) > index:
                out.append(names[index])
        index += 1
    return out


def confusion(records: Sequence[Dict]) -> Dict:
    """Confusion matrix and accuracy over records that carry a truth label."""
    judged = [r for r in records if r.get("truth") and r.get("genre")]
    matrix: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for record in judged:
        matrix[record["truth"]][record["genre"]] += 1
    correct = sum(1 for r in judged if r["genre"] == r["truth"])
    per_genre = {
        truth: {"n": sum(row.values()),
                "recall": round(row[truth] / max(sum(row.values()), 1), 4),
                "top_confusion": (row.most_common(1)[0][0] if row else None)}
        for truth, row in sorted(matrix.items())
    }
    return {
        "judged": len(judged),
        "unparsed": sum(1 for r in records if r.get("genre") is None),
        "accuracy": round(correct / max(len(judged), 1), 4),
        # Ten classes: chance is 0.10, and "better than chance" is not the bar --
        # a pre-split on a 30%-accurate label scatters the same movement across
        # groups and is worse than not splitting at all.
        "chance": round(1.0 / len(GENRES), 4),
        "per_genre": per_genre,
        "matrix": {truth: dict(row) for truth, row in sorted(matrix.items())},
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--reference", choices=("aist",), default=None,
                        help="score against ground truth in the filename")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--max-side", type=int, default=448)
    parser.add_argument("--frames", type=int, default=FRAMES_PER_CLIP)
    parser.add_argument("--as-video", action="store_true",
                        help="feed the frames through Qwen3-VL's video path "
                             "(temporal encoding) instead of as N stills; dance "
                             "style lives in movement, which stills discard")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args(argv)

    videos = sorted(args.videos.glob("*.mp4"))
    if args.reference == "aist":
        videos = interleave_by_genre(videos)
    if args.num_shards > 1:
        videos = [v for v in videos
                  if shard_of((v.stem, 0, 0), args.num_shards) == args.shard]
    if args.limit:
        videos = videos[:args.limit]
    if not videos:
        raise SystemExit("no .mp4 under {}".format(args.videos))

    # Resume: this is a long pass over a corpus and it will be interrupted.
    done = set()
    if args.output.exists() and args.output.suffix == ".jsonl":
        for line in args.output.open(encoding="utf-8"):
            try:
                done.add(json.loads(line)["clip"])
            except (ValueError, KeyError):
                continue

    captioner = Captioner(args.model, device=args.device,
                          max_new_tokens=args.max_new_tokens, prompt=build_prompt())
    records: List[Dict] = []
    pending: List[Tuple[pathlib.Path, List]] = []

    def flush() -> None:
        if not pending:
            return
        frames_only = [images for _, images in pending]
        replies = (captioner.caption_video_batch(frames_only) if args.as_video
                   else captioner.caption_batch(frames_only))
        for (video, _), reply in zip(pending, replies):
            genre, confidence = parse_genre(reply)
            record = {"clip": video.stem, "genre": genre, "confidence": confidence,
                      "raw": reply.strip()[:200]}
            if args.reference == "aist":
                record["truth"] = aist_truth(video.stem)
            records.append(record)
            print(json.dumps({k: v for k, v in record.items() if k != "raw"}), flush=True)
        pending.clear()

    for video in videos:
        if video.stem in done:
            continue
        total = clip_frame_count(video)
        if total <= 0:
            records.append({"clip": video.stem, "genre": None, "error": "unreadable"})
            continue
        width, height = probe_frame_size(video)
        images = load_frames(video, frame_positions(total, args.frames),
                             args.max_side, None)
        if not images:
            records.append({"clip": video.stem, "genre": None, "error": "no_frames"})
            continue
        pending.append((video, images))
        if len(pending) >= args.batch_size:
            flush()
    flush()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, object] = {
        "model": str(args.model), "videos": str(args.videos),
        "frames_per_clip": args.frames,
        # Recorded, not inferred from the filename.  ``runs/genre_gate_video.json``
        # did not say which path it took, and when the video path turned out to
        # have been feeding two frames (2026-08-22) there was no way to tell from
        # the artifact whether its 0.1125 was measured on video or on stills --
        # which mattered, because dropping the paper's genre pre-split rests on
        # that number.
        "input_mode": "video" if args.as_video else "images",
        "genres": list(GENRES) + [OTHER],
        "records": records,
    }
    if args.reference == "aist":
        payload["reference"] = confusion(records)
        print("\n" + json.dumps(payload["reference"], indent=1))
    args.output.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print("\nwrote {} ({} clips)".format(args.output, len(records)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
