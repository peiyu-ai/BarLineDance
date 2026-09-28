#!/usr/bin/env python3
"""Lay out the clips whose visual odometry diverged, beside ones that did not.

WHY THIS EXISTS.  ``run_gvhmr_ingest_shard.sh`` writes ``.extract_failed`` with
a *reason* rather than an empty touch, which is enough to count failures but not
enough to look at them.  When the failure rate moves -- 2026-09-02, a pending
subset ran at 25% against a corpus-wide 3.7% (``runs/gvhmr_raw_states.json``:
221 of 6041) -- the question is whether the footage is different or the
instrument is, and only one of those is answerable by counting.

So this writes the clips themselves out, with a CONTROL GROUP.  A folder of
failures alone invites the same mistake this repository has made before: every
failing clip will look like it has something wrong with it, because every clip
does.  The successes are copied beside them, with the identical statistics, so
"the failures are dark / short / crowded" can be checked against what a success
looks like instead of against an impression.

THE STATISTICS, and what each is for:

``first_second_*``  VO diverging at frame 4-11 of 600 is a claim about the
                    START of the clip, not the clip.  Luminance, Laplacian
                    variance (texture -- a blank wall or a motion-blurred pan
                    has little) and mean absolute inter-frame difference are
                    computed over the first 30 frames AND over the whole clip,
                    so a start that is unlike its own clip is visible as a
                    ratio rather than as an absolute anyone has to calibrate.

``rival_*``         from the ingest cache's meta.json: how much of the frame
                    another dancer occupied.  DPVO tracks the *camera*, and a
                    second person moving through frame is exactly the structure
                    it can mistake for camera motion.

``switches``        from runs/dancer_track_audit_full.json, when present.

Nothing here decides anything.  It is a contact sheet with numbers attached,
for the operator to look at.

Usage::

    python3 tools/report_vo_failures.py \\
        --raw-root /cache/atomicdance-assets/data/wild3d/ingest_txy_gvhmr_raw \\
        --ingest-root /cache/atomicdance-assets/data/wild_ingest_v1 \\
        --converted-root /cache/atomicdance-assets/data/wild3d/ingest_v1_converted \\
        --output-dir output/t_dpvo_failures --controls 6
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import sys
from typing import Dict, List, Optional

import numpy as np

REASON_FRAME = re.compile(r"from frame (\d+) of (\d+)")


def clip_stats(video: pathlib.Path, head_frames: int = 30) -> Dict[str, object]:
    """Luminance, texture and inter-frame motion, for the head and the whole."""
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        return {"readable": False}
    lum: List[float] = []
    tex: List[float] = []
    dif: List[float] = []
    previous = None
    width = height = 0
    index = 0
    # Every frame for the head (that is the half-second under investigation),
    # every 5th after it: the tail values are only a normalizer for the head,
    # and full-rate decoding of a 1080x1920 clip costs a minute each.
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        index += 1
        if index > head_frames and index % 5:
            previous = None
            continue
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Half resolution for the texture/luminance statistics; the Laplacian
        # variance changes scale but every clip is measured the same way and it
        # is read as a ratio to the same clip's own whole.
        grey = cv2.resize(grey, (grey.shape[1] // 2, grey.shape[0] // 2))
        height, width = grey.shape
        lum.append(float(grey.mean()))
        tex.append(float(cv2.Laplacian(grey, cv2.CV_64F).var()))
        if previous is not None:
            dif.append(float(np.abs(grey.astype(np.int16) - previous).mean()))
        previous = grey.astype(np.int16)
    capture.release()
    if not lum:
        return {"readable": False}

    def summarize(values: List[float], head: int) -> Dict[str, float]:
        array = np.asarray(values, dtype=float)
        head_slice = array[: max(1, min(head, len(array)))]
        whole = float(array.mean())
        return {
            "head": round(float(head_slice.mean()), 3),
            "whole": round(whole, 3),
            "head_over_whole": round(float(head_slice.mean() / whole), 3) if whole else None,
        }

    return {
        "readable": True,
        "frames": len(lum),
        "resolution": "{}x{} (statistics at half scale)".format(width * 2, height * 2),
        "luminance": summarize(lum, head_frames),
        "texture_laplacian_var": summarize(tex, head_frames),
        "interframe_abs_diff": summarize(dif, head_frames) if dif else None,
    }


def head_montage(video: pathlib.Path, destination: pathlib.Path, frames: int,
                 height: int = 320) -> bool:
    """A strip of the clip's FIRST frames, which is where the divergence is.

    "Non-finite camera track from frame 4 of 570" is a statement about half a
    second of footage, and a 16-second mp4 is the wrong object to look at to
    check it.  One row, one thumbnail per frame, frame number burned in.
    """
    import cv2
    from PIL import Image, ImageDraw

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        return False
    thumbs = []
    for index in range(frames):
        ok, frame = capture.read()
        if not ok:
            break
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        scale = height / image.height
        image = image.resize((max(1, int(image.width * scale)), height))
        draw = ImageDraw.Draw(image)
        label = str(index)
        draw.rectangle([0, 0, 8 + 7 * len(label), 16], fill=(0, 0, 0))
        draw.text((4, 3), label, fill=(255, 255, 0))
        thumbs.append(image)
    capture.release()
    if not thumbs:
        return False
    width = sum(t.width for t in thumbs)
    strip = Image.new("RGB", (width, height), (16, 16, 16))
    offset = 0
    for thumb in thumbs:
        strip.paste(thumb, (offset, 0))
        offset += thumb.width
    strip.save(destination, quality=88)
    return True


def load_meta(ingest_root: pathlib.Path, stem: str) -> Dict[str, object]:
    path = ingest_root / stem / "meta.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_track_audit(path: Optional[pathlib.Path]) -> Dict[str, Dict[str, object]]:
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["clip"]: row for row in payload.get("records", [])}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", type=pathlib.Path, required=True,
                        help="GVHMR raw output root; failures leave .extract_failed here")
    parser.add_argument("--ingest-root", type=pathlib.Path, required=True)
    parser.add_argument("--converted-root", type=pathlib.Path, required=True,
                        help="a clip with quality.json here is a success")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--track-audit", type=pathlib.Path,
                        default=pathlib.Path("runs/dancer_track_audit_full.json"))
    parser.add_argument("--controls", type=int, default=6,
                        help="how many successful clips to copy beside the failures")
    parser.add_argument("--no-copy-video", action="store_true")
    parser.add_argument("--montage-frames", type=int, default=15,
                        help="write a strip of the clip's first N frames; 0 disables")
    args = parser.parse_args(argv)

    failures = sorted(p.parent.name for p in args.raw_root.glob("*/.extract_failed"))
    attempted = sorted(p.name for p in args.raw_root.iterdir() if p.is_dir())
    successes = [s for s in attempted
                 if s not in set(failures)
                 and (args.converted_root / s / "quality.json").is_file()]
    if not failures:
        print("no .extract_failed under {}".format(args.raw_root), file=sys.stderr)

    audit = load_track_audit(args.track_audit)
    out = args.output_dir
    (out / "failed").mkdir(parents=True, exist_ok=True)
    (out / "succeeded").mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    chosen_success = successes[: max(0, args.controls)]
    for group, stems in (("failed", failures), ("succeeded", chosen_success)):
        for stem in stems:
            reason = ""
            frame = total = None
            marker = args.raw_root / stem / ".extract_failed"
            if marker.is_file():
                reason = marker.read_text(encoding="utf-8").strip()
                match = REASON_FRAME.search(reason)
                if match:
                    frame, total = int(match.group(1)), int(match.group(2))
            video = args.ingest_root / stem / "clip.mp4"
            record: Dict[str, object] = {
                "clip": stem,
                "group": group,
                "reason": reason,
                "diverged_at_frame": frame,
                "of_frames": total,
                "meta": load_meta(args.ingest_root, stem),
                "track_audit": audit.get(stem),
                "video_present": video.is_file(),
            }
            if video.is_file():
                record["stats"] = clip_stats(video)
                if not args.no_copy_video:
                    shutil.copy2(video, out / group / (stem + ".mp4"))
                if args.montage_frames > 0:
                    strip = out / group / (stem + ".head{}.jpg".format(args.montage_frames))
                    record["head_montage"] = head_montage(video, strip, args.montage_frames)
            rows.append(record)
            print("{:9s} {}".format(group, stem))

    (out / "failures.json").write_text(
        json.dumps({"raw_root": str(args.raw_root),
                    "attempted": len(attempted),
                    "failed": len(failures),
                    "succeeded_attempted": len(successes),
                    "records": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    lines = ["# 视觉里程计(DPVO)发散的 clip,以及同一批里没发散的对照",
             "",
             "`raw_root` = `{}`".format(args.raw_root),
             "",
             "本轮尝试 **{}** 条,失败 **{}** 条,成功 **{}** 条。".format(
                 len(attempted), len(failures), len(successes)),
             "",
             "视频在 `failed/` 与 `succeeded/` 两个子目录里,同名 `.mp4`;",
             "旁边的 `.head15.jpg` 是**该 clip 前 15 帧**的横条(帧号烧在左上角)——",
             "发散发生在第 4–11 帧,所以要看的是这半秒,不是那 16 秒。",
             "",
             "> **对照是刻意放的。** 只看失败样本,每一条都会显得\"有点特殊\" —— 因为每条 clip 都有点特殊。",
             "> 下面每个统计量,失败组和成功组都算了同一份,所以\"暗/糊/有别人入镜\"要跟成功组比,不是跟印象比。",
             "",
             "## 表",
             "",
             "| clip | 组 | 发散帧/总帧 | 分辨率 | 前 30 帧亮度(占全片) | 前 30 帧纹理(占全片) | 前 30 帧帧间差(占全片) | rival_ratio | 舞者在场 | span 结束原因 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        stats = row.get("stats") or {}
        meta = row.get("meta") or {}

        def cell(name: str) -> str:
            block = stats.get(name)
            if not isinstance(block, dict):
                return "—"
            return "{} ({})".format(block.get("head"), block.get("head_over_whole"))

        lines.append("| `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            row["clip"], row["group"],
            "{}/{}".format(row["diverged_at_frame"], row["of_frames"])
            if row["diverged_at_frame"] is not None else "—",
            stats.get("resolution", "—"),
            cell("luminance"), cell("texture_laplacian_var"), cell("interframe_abs_diff"),
            meta.get("rival_ratio", "—"), meta.get("dancer_present_fraction", "—"),
            meta.get("span_end_reason", "—")))
    lines += ["", "## 每条失败的原文", ""]
    for row in rows:
        if row["group"] != "failed":
            continue
        lines += ["### `{}`".format(row["clip"]), "", "```", str(row["reason"]), "```", ""]
        if row.get("track_audit"):
            lines += ["舞者轨迹审计:`{}`".format(json.dumps(row["track_audit"], ensure_ascii=False)), ""]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nwrote {} and {}".format(out / "README.md", out / "failures.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
