#!/usr/bin/env python3
"""Fold a rendered gallery into ONE self-contained HTML under ``output/``.

``tools/build_dance_gallery.py`` writes an ``index.html`` beside a tree of
``.mp4`` files and links them by relative path.  That is correct where it is
built and useless anywhere else: the page and the videos travel separately, so
whoever opens the copied HTML gets neither picture nor sound and has no way to
tell which of the two went missing.  Measured 2026-08-29 on
``runs/gallery_wild_v5_song_v5rekey``: the muxed audio was intact at
-7.7 dB mean (source wav -7.5 dB) and every video was h264/yuv420p, yet the
reviewer reported "no preview, no music" -- both symptoms of the same absent
directory, and neither of them a fault in the media.

Everything in ``output/`` is already a single file for this reason
(``beats_*.html`` embeds ``data:video/mp4;base64``), so this makes that the
tool rather than the habit.

**The gate is on the product, not the ingredient.**  ``build_dance_gallery``
refuses a clip whose ``audio.wav`` cannot be found, which is the right check on
the way in and says nothing about the way out: a mux can succeed, write a
well-formed AAC stream, and carry silence.  A page built to let someone judge
dance against its music is worth nothing silent, and silence is exactly the
failure that looks fine in every listing (`ffprobe` reports the stream, the
byte count is plausible).  So each video is measured with ``volumedetect``
before it is embedded, and no audio stream -- or a mean volume under the floor
below -- stops the build naming the clip.

**Why the floor is a level and not ``-inf``.**  The first version of this gate
tested ``mean_volume == -inf`` and could never have fired: AAC does not encode
digital silence as silence.  Measured 2026-08-29, both ends of the same
pipeline -- a real clip's muxed music reads **-7.7 dB** mean (its source wav
-7.5 dB), while ``anullsrc`` through the same AAC encoder reads **-91.0 dB**,
not ``-inf``.  ``SILENCE_FLOOR_DB`` sits between the two measurements rather
than being chosen: 50 dB below anything real on this corpus, 30 dB above what
the encoder produces from nothing.  It was the unit test that caught this, not
a review -- which is the argument for the test existing at all.

Usage::

    python3 tools/build_single_file_gallery.py \\
        --gallery runs/gallery_wild_v5_song_v5rekey \\
        --output output/wild_v5_song_v5rekey_gallery.html \\
        --title "Wild v5rekey — held-out clips"
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Imported where it is used, not here.  It pulls matplotlib in, and embedding
# videos does not need a plotting library -- keeping the import lazy also keeps
# matplotlib out of any process that only wants the rest of this module, which
# is what stopped the test suite segfaulting when torch and matplotlib were
# loaded into one pytest process on 2026-08-29.

MEAN_VOLUME = re.compile(r"mean_volume:\s*(-?[\d.]+|-inf)\s*dB")
# Between two measurements, not chosen: real music here is -7.7 dB mean and
# AAC-encoded digital silence is -91.0 dB.  See the module docstring.
SILENCE_FLOOR_DB = -60.0


class GalleryError(RuntimeError):
    pass


def audio_mean_volume(path: pathlib.Path) -> float | None:
    """Mean volume in dB, ``None`` when the file carries no audio stream.

    ``-inf`` comes back as ``float("-inf")`` rather than an exception, because
    "silent" and "absent" are different defects with the same symptom and the
    caller reports them apart.
    """
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True)
    if "audio" not in probe.stdout:
        return None
    completed = subprocess.run(
        ["ffmpeg", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True)
    found = MEAN_VOLUME.search(completed.stderr)
    if found is None:
        return None
    return float("-inf") if found.group(1) == "-inf" else float(found.group(1))


def transcode(source: pathlib.Path, target: pathlib.Path, crf: int) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(source),
         "-c:v", "libx264", "-crf", str(crf), "-preset", "slow",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         "-c:a", "libmp3lame", "-ar", "44100", "-ac", "2", "-b:a", "128k",
         str(target)],
        check=True, capture_output=True)


def embed(path: pathlib.Path) -> str:
    return "data:video/mp4;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def diagnostics_table(rows) -> str:
    keys = [("median_joint_speed_m_per_s", "speed m/s"),
            ("foot_skate_geometric_m_per_s", "skate m/s"),
            ("grounded_fraction_geometric", "grounded"),
            ("median_root_height_m", "root h m"),
            ("lowest_joint_z_m", "low z m"),
            ("frames", "frames")]
    head = "".join("<th>{}</th>".format(html.escape(label)) for _, label in keys)
    body = []
    for row in rows:
        diag = row.get("diagnostics", {})
        cells = []
        for key, _ in keys:
            value = diag.get(key)
            cells.append("<td>{}</td>".format(
                "—" if value is None else
                ("{:d}".format(int(value)) if key == "frames" else "{:.3f}".format(value))))
        body.append("<tr><td class=\"rl\">{}</td>{}</tr>".format(
            html.escape(row.get("label", "")), "".join(cells)))
    return ("<table><thead><tr><th class=\"rl\">row</th>{}</tr></thead>"
            "<tbody>{}</tbody></table>".format(head, "".join(body)))


def plan_rows(entry, ground_truth_dir, labels_root):
    """``[(label, labels, joints)]`` for one clip, ground truth first.

    The rows are read back out of the artefacts each video was rendered from,
    not re-derived: the ground truth's label track from the M3 label set, each
    arm's emitted plan from the pickle its own row names.  A row whose plan
    cannot be found is returned without one rather than silently drawn as
    filler -- an all-grey lane and a missing lane must not look the same.
    """
    import json as _json
    import pickle as _pickle

    clip = entry["rows"][0].get("detail", "")
    out = []
    labels, joints = None, None
    if labels_root is not None:
        index = getattr(plan_rows, "_index", None)
        if index is None or index[0] != labels_root:
            table = {}
            for line in (labels_root / "labels.jsonl").open(encoding="utf-8"):
                row = _json.loads(line)
                table[row["recording_id"]] = (row["labels_path"],
                                              row["label_valid_mask_path"])
            plan_rows._index = (labels_root, table)
            index = plan_rows._index
        found = index[1].get(clip)
        if found is not None:
            raw = np.load(labels_root / found[0])
            mask = np.load(labels_root / found[1]).astype(bool)
            labels = np.zeros(len(raw), int)
            labels[mask] = raw[mask]
    if ground_truth_dir is not None:
        path = ground_truth_dir / (clip + ".pkl")
        if path.is_file():
            with path.open("rb") as handle:
                joints = np.asarray(_pickle.load(handle)["full_pose"])
    if labels is not None:
        out.append((entry["rows"][0].get("label", "ground truth"), labels, joints))

    for row in entry["rows"][1:]:
        path = pathlib.Path(row.get("detail", "")) / (clip + ".pkl")
        if not path.is_file():
            continue
        with path.open("rb") as handle:
            blob = _pickle.load(handle)
        out.append((row.get("label", "arm"),
                    np.asarray(blob["atomic_labels"]),
                    np.asarray(blob["full_pose"])))
    return clip, out


def build(gallery: pathlib.Path, output: pathlib.Path, title: str, crf: int,
          limit: int | None, note: str, labels_root=None,
          ground_truth_dir=None, audio_features=None, stills=False) -> dict:
    manifest = json.loads((gallery / "gallery.json").read_text(encoding="utf-8"))
    entries = manifest["entries"]
    if limit is not None:
        entries = entries[:limit]
    if not entries:
        raise GalleryError("{} holds no entries".format(gallery / "gallery.json"))

    silent, no_audio, cards, embedded_bytes, strips = [], [], [], 0, 0
    tags = None
    if labels_root is not None:
        tags = {int(k): v for k, v in json.loads(
            (labels_root / "subprototype_tags.json").read_text(
                encoding="utf-8")).items()}
    staging = pathlib.Path(tempfile.mkdtemp(prefix="single-file-gallery-"))
    try:
        for entry in entries:
            source = gallery / entry["compare"]
            if not source.is_file():
                raise GalleryError("missing rendered video: {}".format(source))
            level = audio_mean_volume(source)
            if level is None:
                no_audio.append(entry["compare"])
                continue
            if level <= SILENCE_FLOOR_DB:
                silent.append("{} ({:.1f} dB)".format(entry["compare"], level))
                continue
            if stills:
                # The IDE preview will not play <video> -- recorded in
                # tools/render_m4_plan_sheet.py, and confirmed 2026-08-29 when a
                # 24 MB page of embedded mp4 rendered blank in it.  So the page
                # carries stills only and the playable copies go beside it, for
                # whichever viewer the reader actually has.
                videos = output.parent / (output.stem + "_videos")
                videos.mkdir(parents=True, exist_ok=True)
                target = videos / (pathlib.Path(entry["compare"]).stem + ".mp4")
                shutil.copyfile(source, target)
                embedded_bytes += target.stat().st_size
            else:
                target = staging / (pathlib.Path(entry["compare"]).stem + ".mp4")
                transcode(source, target, crf)
                embedded_bytes += target.stat().st_size
            clip = entry["rows"][0].get("detail", pathlib.Path(entry["compare"]).stem)
            strip = ""
            if labels_root is not None:
                name, rows = plan_rows(entry, ground_truth_dir, labels_root)
                if rows:
                    music = None
                    if audio_features is not None:
                        feature = audio_features / (name + ".npy")
                        if feature.is_file():
                            music = np.load(feature)
                    from tools import render_plan_strip
                    draw = (render_plan_strip.render_sheet if stills
                            else render_plan_strip.render)
                    png = draw(name, rows, music=music, tags=tags)
                    strip = ("<img class=\"strip\" alt=\"plan over time\" "
                             "src=\"data:image/png;base64,{}\">".format(
                                 base64.b64encode(png).decode("ascii")))
                    strips += 1
            cards.append(
                "<figure class=\"card\">"
                "<figcaption><span class=\"clip\">{clip}</span>"
                "<span class=\"meta\">{secs} s · music {level:+.1f} dB</span></figcaption>"
                "{player}"
                "{strip}"
                "<div class=\"tw\">{table}</div>"
                "</figure>".format(
                    clip=html.escape(clip),
                    secs=entry["rows"][0].get("seconds", "?"),
                    level=level,
                    player=("<p class=\"play\">plays with its music in any video "
                            "player: <code>{}</code></p>".format(html.escape(str(target)))
                            if stills else
                            "<video controls preload=\"metadata\" src=\"{}\">"
                            "</video>".format(embed(target))),
                    strip=strip,
                    table=diagnostics_table(entry["rows"])))
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    if no_audio or silent:
        raise GalleryError(
            "the page exists to let someone judge dance against its music, so a "
            "silent row is a failed build, not a quiet one: {} clip(s) carry no "
            "audio stream {} and {} carry a stream at or under {:.0f} dB mean, "
            "which is silence {}"
            .format(len(no_audio), no_audio[:3], len(silent),
                    SILENCE_FLOOR_DB, silent[:3]))

    labels = [row.get("label", "") for row in entries[0]["rows"]]
    differing = (manifest.get("sampling_check") or {}).get("differing") or {}
    warning = ""
    if differing:
        warning = ("<p class=\"warn\"><b>The runs were sampled differently</b> ({}). "
                   "A difference on screen is not attributable to the checkpoint "
                   "alone.</p>".format(html.escape(", ".join(sorted(differing)))))

    page = PAGE.format(
        title=html.escape(title),
        note=note,
        rows=" &nbsp;/&nbsp; ".join(html.escape(label) for label in labels),
        clips=len(cards),
        megabytes=embedded_bytes / 1e6,
        source=html.escape(str(gallery)),
        warning=warning,
        cards="".join(cards))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    return {"clips": len(cards), "embedded_mb": round(embedded_bytes / 1e6, 1),
            "plan_strips": strips,
            "page_mb": round(output.stat().st_size / 1e6, 1)}


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{
  --ground:#F6F7F9; --surface:#FFFFFF; --sunk:#EFF2F6;
  --ink:#12161C; --muted:#57626F; --faint:#7C8695;
  --rule:#DCE1E8; --strong:#C3CBD6; --accent:#1F5F8B; --warn:#7D6118; --warn-bg:#F4EDDA;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --ground:#0D1014; --surface:#151A21; --sunk:#10151B;
    --ink:#E7EBF1; --muted:#9AA5B3; --faint:#77828F;
    --rule:#242B35; --strong:#333C48; --accent:#79B4DA; --warn:#D6B14F; --warn-bg:#26200F;
  }}
}}
:root[data-theme="dark"] {{
  --ground:#0D1014; --surface:#151A21; --sunk:#10151B;
  --ink:#E7EBF1; --muted:#9AA5B3; --faint:#77828F;
  --rule:#242B35; --strong:#333C48; --accent:#79B4DA; --warn:#D6B14F; --warn-bg:#26200F;
}}
* {{ box-sizing:border-box }}
body {{
  margin:0; background:var(--ground); color:var(--ink);
  font:15px/1.6 "Helvetica Neue",Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased;
}}
.wrap {{ max-width:1080px; margin:0 auto; padding:48px 24px 96px }}
header {{ border-bottom:2px solid var(--ink); padding-bottom:20px; margin-bottom:8px }}
h1 {{ font-size:30px; line-height:1.15; margin:0 0 10px; letter-spacing:-.01em }}
.sub {{ color:var(--muted); margin:0 0 16px; max-width:70ch }}
.facts {{
  display:flex; flex-wrap:wrap; gap:6px 22px;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:11.5px; color:var(--faint);
}}
.facts b {{ color:var(--muted); font-weight:500 }}
.warn {{
  background:var(--warn-bg); color:var(--warn); border-left:3px solid var(--warn);
  padding:10px 14px; margin:20px 0 0; font-size:13.5px;
}}
.card {{ margin:36px 0 0; background:var(--surface); border:1px solid var(--rule) }}
figcaption {{
  display:flex; justify-content:space-between; align-items:baseline; gap:16px; flex-wrap:wrap;
  padding:12px 16px; border-bottom:1px solid var(--rule);
}}
.clip {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12.5px }}
.meta {{ font-size:12px; color:var(--faint); white-space:nowrap }}
video {{ display:block; width:100%; height:auto; background:#000 }}
.strip {{ display:block; width:100%; height:auto; background:#fff;
          border-top:1px solid var(--rule) }}
.play {{ margin:0; padding:11px 16px; font-size:12.5px; color:var(--muted);
         border-top:1px solid var(--rule) }}
.play code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
              font-size:11.5px; color:var(--ink) }}
.tw {{ overflow-x:auto; border-top:1px solid var(--rule) }}
table {{ border-collapse:collapse; width:100%; font-size:12.5px }}
th, td {{ padding:7px 14px; text-align:right; white-space:nowrap; border-bottom:1px solid var(--rule) }}
th.rl, td.rl {{ text-align:left }}
thead th {{
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:10px;
  letter-spacing:.08em; text-transform:uppercase; color:var(--faint); font-weight:500;
  border-bottom:1px solid var(--strong);
}}
tbody td {{ font-variant-numeric:tabular-nums }}
tbody tr:first-child td {{ background:var(--sunk); font-weight:600 }}
tbody tr:first-child td.rl {{ color:var(--accent) }}
tbody tr:last-child td {{ border-bottom:none }}
footer {{
  border-top:1px solid var(--strong); margin-top:56px; padding-top:16px;
  font-size:12.5px; color:var(--faint);
}}
</style></head><body><div class="wrap">
<header>
  <h1>{title}</h1>
  <p class="sub">{note}</p>
  <div class="facts">
    <span><b>rows, top to bottom</b> {rows}</span>
    <span><b>clips</b> {clips}</span>
    <span><b>video embedded</b> {megabytes:.1f} MB</span>
    <span><b>source</b> {source}</span>
  </div>
  {warning}
</header>
{cards}
<footer>Every video is embedded in this file — it plays with its music from any
folder, with nothing beside it. Each was measured before embedding; a clip whose
audio was missing or silent would have stopped the build rather than shipping
quietly.</footer>
</div></body></html>
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gallery", type=pathlib.Path, required=True,
                        help="a directory written by tools/build_dance_gallery.py")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--title", default="Held-out clips")
    parser.add_argument("--note", default="")
    parser.add_argument("--stills", action="store_true",
                        help="page carries PNG sheets only and the mp4 copies go "
                             "in a sibling folder; the IDE preview cannot play "
                             "<video> and a page of embedded mp4 renders blank in it")
    parser.add_argument("--crf", type=int, default=28,
                        help="x264 quality for the embedded copy; higher is smaller")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--labels-root", type=pathlib.Path, default=None,
                        help="M3 label set (labels.jsonl + subprototype_tags.json); "
                             "enables the plan strip under each video")
    parser.add_argument("--ground-truth-dir", type=pathlib.Path, default=None,
                        help="per-clip ground-truth .pkl, for the strip's accent marks")
    parser.add_argument("--audio-features", type=pathlib.Path, default=None,
                        help="per-clip 35-D .npy, for the strip's music lane")
    args = parser.parse_args(argv)
    try:
        stats = build(args.gallery, args.output, args.title, args.crf,
                      args.limit, args.note, labels_root=args.labels_root,
                      ground_truth_dir=args.ground_truth_dir,
                      audio_features=args.audio_features, stills=args.stills)
    except GalleryError as error:
        print("error: {}".format(error), file=sys.stderr)
        return 1
    print("{} clip(s), {} plan strip(s), {} MB of video -> {} ({} MB)".format(
        stats["clips"], stats["plan_strips"], stats["embedded_mb"],
        args.output, stats["page_mb"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
