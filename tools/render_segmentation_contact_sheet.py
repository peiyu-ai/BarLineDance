#!/usr/bin/env python3
"""One page per corpus: several clips, one row per segment, sampled across accounts.

``render_segmentation_example.py`` answers "is this one movement" for *one* clip.
This is the same question asked of a *corpus*, which needs three things that one
clip does not:

* **stratified sampling.**  Drawing at random from the whole corpus draws in
  proportion to clip count, so the largest account fills the page and the
  question "does this account cut differently" cannot be asked.  Here the draw
  is per account, and the pool size and seed are printed and written into the
  page -- picking the clip that looks best is the failure mode this guards.
* **the video comes from the object store, not from disk.**  The ingest tree's
  ``clip.mp4`` was evicted to OSS, and what *is* on disk for a re-cut clip is
  the new cut, which the published features do not describe.  Reading
  local-first would pair one generation's frames with the other generation's
  boundaries and the page would look fine -- that exact failure produced a
  contact sheet where both arms were identical on 2026-08-19.  Every frame here
  is fetched with ossutil.
* **the motion array comes from the converted tree.**  The released bundle's
  ``sequences/`` payload is a symlink into a staging directory that no longer
  exists, so ``load_bundle_index`` cannot resolve it.  ``ingest_v1_converted/
  <stem>/atomic_motion_151.npy`` is the same 151-D array before normalisation
  and is what the change signal is drawn from.

Arms are compared side by side because choosing one is a live decision: the
published M1 was computed before the S3D re-extraction of 2026-08-19 and indexes
features that no longer exist for 3,929 of 13,783 clips, so "which segmentation"
is not settled by reading the newest file.

Usage::

    render_segmentation_contact_sheet.py \\
        --clips runs/clean5/clips.txt \\
        --arm 'published=runs/wild_v4_seg/segmentation.json' \\
        --arm 'visual=/cache/.../c1/full/arms/visual_*.json' \\
        --per-account 3 --output output/clean5_segmentation.html
"""

from __future__ import annotations

import argparse
import collections
import glob
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.convert_motion_to_guofeats import motion_151_to_joints            # noqa: E402
from tools.probe_segmentation_boundaries import change_signal, visual_change  # noqa: E402
from tools.render_segmentation_example import as_data_uri, decode, strip, timeline  # noqa: E402
from tools import render_segmentation_example as example                       # noqa: E402

OSSUTIL = os.environ.get("OSSUTIL", "/opt/data-infra/ossutil64")


def load_arm(spec: str) -> Dict[str, List[int]]:
    """``name=path`` where path is a segmentation.json or a shard glob.

    The corrected arms were written as 64 shards and never merged, so a reader
    that only accepts a merged file would silently be reading the stale one.
    """
    paths = sorted(glob.glob(spec)) or [spec]
    out: Dict[str, List[int]] = {}
    for path in paths:
        report = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        for record in report["records"]:
            out[record["sequence"]] = [int(b) for b in record["boundaries"]]
    if not out:
        raise SystemExit("arm {} matched no records".format(spec))
    return out


def fetch_clip(stem: str) -> Optional[pathlib.Path]:
    """The clip as the object store holds it -- which is what the features describe."""
    from tools import asset_io

    handle, name = tempfile.mkstemp(suffix=".mp4")
    os.close(handle)
    target = pathlib.Path(name)
    url = "oss://" + asset_io.remote_path(
        "data/wild_ingest_v1/{}/clip.mp4".format(stem))
    argv = [OSSUTIL, "cp", url, str(target), "-f"]
    config = REPO / "ossutilconfig"
    if config.is_file():
        argv += ["-c", str(config)]
    done = subprocess.run(argv, capture_output=True, text=True)
    if done.returncode != 0 or not target.is_file() or target.stat().st_size == 0:
        target.unlink(missing_ok=True)
        return None
    return target


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", type=pathlib.Path, required=True)
    parser.add_argument("--group-keys", type=pathlib.Path,
                        default=REPO / "runs/wild_v4_group_keys.json")
    parser.add_argument("--arm", action="append", required=True,
                        help="name=path-or-glob; repeat for a side-by-side comparison")
    parser.add_argument("--motion-dir", type=pathlib.Path,
                        default=pathlib.Path("/cache/atomicdance-assets/data/wild3d/"
                                             "ingest_v1_converted"))
    parser.add_argument("--features-dir", type=pathlib.Path,
                        default=pathlib.Path("/cache/atomicdance-assets/data/wild_visual_s3d"))
    parser.add_argument("--mark-settles", type=float, default=None,
                        help="draw the clip's motion beats on the timeline, keeping "
                             "only the deepest QUANTILE of them (1.0 = all, ~42 per "
                             "clip and too dense to read; 0.4 is legible). They are "
                             "drawn as one more row of ticks so the reader can check "
                             "cut-against-settle directly instead of taking a "
                             "statistic's word for it.")
    parser.add_argument("--mark-beats", type=pathlib.Path, default=None,
                        help="raw bundle; draw the clip's MUSIC beat grid on the "
                             "timeline (music_35 channel 34). With a beat-grid arm "
                             "on the page this is what lets the reader check that a "
                             "cut really is on a whole beat rather than take the "
                             "arm's name for it.")
    parser.add_argument("--annotate", type=pathlib.Path, default=None,
                        help="freshness.json; clips it calls stale are labelled "
                             "as due for re-derivation rather than hidden")
    parser.add_argument("--per-account", type=int, default=3)
    parser.add_argument("--thumb-height", type=int, default=118)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    example.THUMB_H = args.thumb_height

    arms = {}
    for spec in args.arm:
        name, path = spec.split("=", 1)
        arms[name] = load_arm(path)

    account_of = json.loads(args.group_keys.read_text(encoding="utf-8"))
    clips = [line.strip() for line in args.clips.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    stale = set()
    if args.annotate and args.annotate.is_file():
        report = json.loads(args.annotate.read_text(encoding="utf-8"))
        for names in report.get("stale", {}).values():
            stale.update(names)

    by_account = collections.defaultdict(list)
    for stem in clips:
        if all(stem in arm for arm in arms.values()):
            by_account[account_of.get(stem.split("__")[0], "?")].append(stem)
    for names in by_account.values():
        names.sort()

    rng = np.random.default_rng(args.seed)
    drawn = []
    for account in sorted(by_account, key=lambda a: -len(by_account[a])):
        pool = by_account[account]
        take = min(args.per_account, len(pool))
        picks = rng.choice(len(pool), take, replace=False)
        for index in sorted(picks):
            drawn.append((account, pool[int(index)], len(pool)))

    print("{} clips drawn, {} per account, seed {}".format(
        len(drawn), args.per_account, args.seed), flush=True)

    sections = []
    skipped = []
    for account, stem, pool_size in drawn:
        video = fetch_clip(stem)
        if video is None:
            skipped.append(stem)
            print("  SKIP {} (no video in the store)".format(stem), flush=True)
            continue
        try:
            frames = decode(video, height=args.thumb_height)
        finally:
            video.unlink(missing_ok=True)

        motion_path = args.motion_dir / stem / "atomic_motion_151.npy"
        mchange = np.zeros(max(1, len(frames) - 1), dtype=np.float32)
        if motion_path.is_file():
            mchange = change_signal(np.load(motion_path).astype(np.float32))
        vchange = np.zeros_like(mchange)
        feature_path = args.features_dir / (stem + ".npz")
        if feature_path.is_file():
            with np.load(feature_path, allow_pickle=False) as bundle:
                feats = bundle["features"].astype(np.float32)
            feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
            vchange = visual_change(feats)

        # Arms that agree on this clip are drawn once under both names.  Drawing
        # the same cuts twice doubles the page for no information, and a reader
        # comparing two identical strips cannot tell agreement from a bug in the
        # page -- the label says which it is.
        panels = []
        by_bounds: Dict[tuple, List[str]] = collections.OrderedDict()
        for name, arm in arms.items():
            by_bounds.setdefault(tuple(arm[stem]), []).append(name)
        for bounds, names in by_bounds.items():
            spans = list(zip(bounds[:-1], bounds[1:]))
            panels.append((" = ".join(names), len(spans),
                           as_data_uri(strip(frames, spans))))
        cuts = {name: arms[name][stem] for name in arms}
        if args.mark_beats is not None:
            music_of = getattr(main, "_music_index", None)
            if music_of is None:
                music_of = {}
                with open(args.mark_beats / "sequences.jsonl", encoding="utf-8") as handle:
                    for line in handle:
                        row = json.loads(line)
                        parts = (row.get("recording_id") or "").split(":")
                        if len(parts) == 3 and row.get("music_path"):
                            music_of["{}__{}".format(parts[1], parts[2])] = (
                                args.mark_beats / row["music_path"])
                main._music_index = music_of
            music_path = music_of.get(stem)
            if music_path and music_path.is_file():
                table = np.load(music_path)
                beat_frames = np.flatnonzero(table[:, 34] > 0.5)
                cuts["music beat"] = ([0] + [int(b) for b in beat_frames]
                                      + [len(frames)])
        if args.mark_settles is not None and motion_path.is_file():
            from tools.motion_beats import find_motion_beats
            from tools.snap_cuts_to_settle import deep_beats

            joints = motion_151_to_joints(np.load(motion_path))
            beats = find_motion_beats(joints, min_separation=3,
                                      max_beats=len(joints), prominence=0.02)
            beats = deep_beats(joints, beats, args.mark_settles)
            # timeline() skips the first and last entry of every cut list (they are
            # the clip ends for an arm), so pad rather than lose two settles.
            cuts["settle (deepest {:.0%})".format(args.mark_settles)] = (
                [0] + sorted(int(b) for b in beats) + [len(frames)])
        span = min(len(mchange), len(vchange)) or 1
        svg = timeline(mchange[:span], vchange[:span], cuts)

        identical = len({tuple(arms[n][stem]) for n in arms}) == 1
        sections.append({
            "account": account, "stem": stem, "frames": len(frames),
            "pool": pool_size, "svg": svg, "panels": panels,
            "stale": stem in stale, "identical": identical,
        })
        print("  {} {} ({} frames, {} arms{})".format(
            account, stem, len(frames), len(arms),
            ", arms identical" if identical else ""), flush=True)

    head = [
        "<title>segmentation contact sheet</title>",
        "<style>body{font:14px/1.55 system-ui;margin:0;padding:28px;"
        "background:#faf9f7;color:#1a1a1a;max-width:1200px}"
        "@media(prefers-color-scheme:dark){body{background:#141416;color:#eaeaea}"
        "code{background:#222}}"
        "img{max-width:100%;height:auto;display:block;border-radius:3px}"
        "h1{font-size:20px}h2{font-size:15px;margin:26px 0 4px}"
        "h3{font-size:13px;margin:14px 0 6px;font-weight:600;opacity:.85}"
        ".clip{margin:34px 0;padding-top:18px;border-top:2px solid #8884}"
        ".scroll{overflow-x:auto;max-width:100%}"
        ".tag{font:11px ui-monospace,monospace;padding:1px 6px;border-radius:3px;"
        "background:#8883;margin-left:6px}"
        "code{font:12px ui-monospace,monospace;background:#8882;padding:1px 4px;"
        "border-radius:3px}p{margin:6px 0}</style>",
        "<h1>Segmentation contact sheet</h1>",
        "<p>{} clips, {} drawn per account with seed <code>{}</code>, from "
        "<code>{}</code> ({} clips). Each row is one segment; the frames in a row "
        "are sampled inside it. Read <b>across</b> a row for &ldquo;is this one "
        "movement&rdquo;, <b>down</b> the rows for &ldquo;did the cut fall between "
        "two movements&rdquo;. Video frames come from the object store, which is "
        "what the features describe.</p>".format(
            len(sections), args.per_account, args.seed, args.clips, len(clips)),
        "<p>Arms: {}</p>".format(", ".join(
            "<code>{}</code>".format(name) for name in arms)),
    ]
    if skipped:
        head.append("<p><b>{} clip(s) skipped</b> because the store had no video: "
                    "<code>{}</code></p>".format(len(skipped), ", ".join(skipped)))

    body = []
    current = None
    for section in sections:
        if section["account"] != current:
            current = section["account"]
            body.append("<h2>{}</h2>".format(current))
        tags = []
        if section["stale"]:
            tags.append("<span class='tag'>3D/S3D stale &mdash; this cut will be "
                        "re-derived</span>")
        if section["identical"] and len(arms) > 1:
            tags.append("<span class='tag'>arms identical</span>")
        body.append("<div class='clip'><h3>{}{}</h3>"
                    "<p>{} frames &middot; {:.1f} s &middot; drawn from {} clips "
                    "of this account</p>".format(
                        section["stem"], "".join(tags), section["frames"],
                        section["frames"] / 30.0, section["pool"]))
        body.append(section["svg"])
        for name, count, uri in section["panels"]:
            body.append("<h3>{} &mdash; {} segments</h3>"
                        "<div class='scroll'><img src='{}' alt='{}'></div>".format(
                            name, count, uri, name))
        body.append("</div>")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(head + body), encoding="utf-8")
    print("wrote {} ({:.1f} MB)".format(args.output,
                                        args.output.stat().st_size / 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
