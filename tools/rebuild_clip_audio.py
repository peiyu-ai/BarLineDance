#!/usr/bin/env python3
"""Re-decode each ingest clip's audio straight from its upload, on the picture's timeline.

``ingest_wild_uploads.cut_clip`` takes a clip's audio with
``aselect=between(t,start,end),asetpts=N/SR/TB``: it keeps whole *decoded
frames* whose timestamp falls in the span, then renumbers the survivors from
zero.  Both halves throw timing away, and on this box they do it in two ways
that were measured on 2026-09-22 (``runs/txy_t2_20260922/
audio_shift_census_txy_t295.json``, every one of the 295 T-corpus clips, audio
cross-correlated against the same span decoded here):

* **161.1 ms early, on every clip that starts at upload frame 0 and was cut
  after the 2026-09-02 environment regression** (19/19; 0/187 of the start-0
  clips cut on 08-14/08-19).  The HE-AAC uploads carry an mp4 edit list whose
  first ``media_time`` is 7106 samples (the encoder priming).  The system
  ffmpeg 4.4.2 applies that skip *twice*: its decode equals ffmpeg 7.0.2's
  decode with the first 7106 samples removed (mean |err| 3e-8), and its first
  decoded frame is stamped pts 0.161 -- correctly, for what it kept.
  ``asetpts=N/SR/TB`` then discards the stamp and plays that frame at t=0.
  At this dancer's 112 BPM that is 0.3 beat -- the scale of the timing defect
  CLAUDE.md §1.6 is about.
* **0-46.4 ms early on clips that start mid-upload** (37 of 80 HE-AAC ones
  above 23 ms; median 21.9, max 46.4 = one 2048-sample HE-AAC frame).
  ``aselect`` is frame-granular, so the clip's audio starts at the first frame
  boundary *after* the span start and is then renumbered to zero.  This one is
  older than the regression: 32 of the 37 were cut on 08-14.

What replaces it: one decode of ``[start, end)`` source frames' worth of time,
by an ffmpeg that honours the edit list once, trimmed on samples rather than on
frames (output ``-ss``/``-t`` are ``atrim``), written as the 22,050 Hz mono wav
``extract_wild_music_features.py`` reads.  It also skips the AAC re-encode that
``clip.mp4`` puts between the upload and ``audio.wav``.

**The fix is an overlay, not an edit.**  ``data/wild_ingest_v1`` is the /cache
read copy of a shared OSS tree, and ``clip.mp4`` is what every 3D result's
freshness hash names (``run_gvhmr_extract.py`` ``video_sha256_1mb``), so
neither is touched: each clip gets ``<overlay>/<clip>/`` holding a symlink to
every file of the ingest clip *except* ``audio.wav``, plus the re-decoded
``audio.wav`` and an ``audio_rebuild.json`` saying where it came from.  Point a
staging row's ``assets.source_cache`` at the overlay and every consumer reads
the same bytes as before except the audio.

**The decoder is checked by behaviour, not by version string.**  Before a clip
is written, its upload is decoded whole and the result must be at least as long
as the container's own audio duration minus one AAC frame: the double skip
shows up as exactly the 161 ms shortfall, so ffmpeg 4.4 fails this gate on the
first HE-AAC upload instead of producing a wrong corpus quietly.

Usage::

    python3 tools/rebuild_clip_audio.py \\
        --clips runs/txy_t2_20260922/t2_clips.txt \\
        --overlay-root /cache/atomicdance-assets/data/wild_ingest_txy_t2_audiofix \\
        --ffmpeg /cache/atomicdance-assets/third_party/ffmpeg-7.0.2-static/ffmpeg \\
        --report runs/txy_t2_20260922/audio_rebuild_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

INGEST_ROOT = pathlib.Path("/cache/atomicdance-assets/data/wild_ingest_v1")
SR = 22050
FPS = 30.0
# One HE-AAC frame at 44.1 kHz.  The double skip is 7106 samples, three times
# this, so the gate's tolerance cannot swallow the defect it exists to catch.
DECODE_TOLERANCE_S = 2048 / 44100.0


def resolve_span(clip: str, meta: dict, manifest_rows: Sequence[dict]
                 ) -> Tuple[str, int, int]:
    """(upload path, start frame, end frame) in the upload's own frames.

    ``meta.json`` records ``source`` and ``source_frame_span`` on every clip cut
    since the fps fix, and that per-clip record wins.  Older clips fall back to
    the ingest manifests, where a recut row lists a ``recut`` marker entry *and*
    an ``ok`` entry for the same clip -- pairing ``clips[i]`` with ``spans[i]``
    there misattributes every span after the first (it put clip000 at 17.5 s on
    19 clips of the 2026-09-22 census before this was fixed), so only ok/exists
    entries are paired, and the latest row for the upload wins.
    """
    span = meta.get("source_frame_span")
    if meta.get("source") and span:
        return meta["source"], int(span[0]), int(span[1])
    found: Optional[Tuple[str, int, int]] = None
    for row in sorted(manifest_rows, key=lambda r: r.get("ingested_at") or 0):
        good = [c for c in row.get("clips", []) if c.get("status") in ("ok", "exists")]
        for entry, sp in zip(good, row.get("spans", [])):
            if entry.get("clip") == clip:
                found = (row["source"], int(sp[0]), int(sp[1]))
    if found is None:
        raise KeyError("no span recorded for {}".format(clip))
    return found


def overlay_entries(clip_dir: pathlib.Path) -> List[str]:
    """Names in the ingest clip dir that the overlay links through unchanged."""
    return sorted(p.name for p in clip_dir.iterdir() if p.name != "audio.wav")


def build_overlay(clip_dir: pathlib.Path, overlay_dir: pathlib.Path) -> None:
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for name in overlay_entries(clip_dir):
        link = overlay_dir / name
        target = (clip_dir / name).resolve()
        if link.is_symlink() or link.exists():
            if link.is_symlink() and pathlib.Path(os.readlink(link)) == target:
                continue
            raise FileExistsError("overlay entry exists and is not our link: {}".format(link))
        link.symlink_to(target)


def container_audio_seconds(upload: str) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                          "stream=duration", "-of", "csv=p=0", upload],
                         capture_output=True, text=True, check=True).stdout.strip()
    return float(out)


def decoded_seconds(ffmpeg: str, upload: str) -> float:
    raw = subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-i", upload, "-vn", "-ac", "1",
                          "-ar", str(SR), "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    return len(raw) / 2.0 / SR


def check_decoder(ffmpeg: str, upload: str, cache: Dict[str, dict]) -> dict:
    """Refuse a decoder that loses the start of this upload's audio."""
    if upload in cache:
        return cache[upload]
    declared = container_audio_seconds(upload)
    got = decoded_seconds(ffmpeg, upload)
    verdict = {"declared_s": round(declared, 4), "decoded_s": round(got, 4),
               "ok": got >= declared - DECODE_TOLERANCE_S}
    cache[upload] = verdict
    return verdict


def decode_span(ffmpeg: str, upload: str, start_s: float, dur_s: float,
                destination: pathlib.Path) -> None:
    partial = destination.with_name(destination.stem + ".partial.wav")
    subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-y", "-i", upload,
                    "-ss", "{:.6f}".format(start_s), "-t", "{:.6f}".format(dur_s),
                    "-vn", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(partial)],
                   check=True)
    partial.replace(destination)


def wav_seconds(path: pathlib.Path) -> float:
    import soundfile as sf
    info = sf.info(str(path))
    return info.frames / float(info.samplerate)


def read_manifests(root: pathlib.Path) -> Dict[str, List[dict]]:
    by_upload: Dict[str, List[dict]] = {}
    for manifest in sorted(root.glob("ingest_*.jsonl")):
        for line in manifest.open(encoding="utf-8"):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            by_upload.setdefault(str(row.get("upload")), []).append(row)
    return by_upload


def rebuild(clips: Iterable[str], overlay_root: pathlib.Path, ffmpeg: str,
            ingest_root: pathlib.Path = INGEST_ROOT) -> List[dict]:
    manifests = read_manifests(ingest_root)
    decoder_cache: Dict[str, dict] = {}
    records = []
    for clip in clips:
        rec = {"clip": clip}
        try:
            clip_dir = ingest_root / clip
            meta = json.loads((clip_dir / "meta.json").read_text(encoding="utf-8"))
            upload_id = clip.split("__")[0]
            source, start, end = resolve_span(clip, meta, manifests.get(upload_id, []))
            source_fps = float(meta.get("source_fps") or FPS)
            gate = check_decoder(ffmpeg, source, decoder_cache)
            rec.update(source=source, source_frame_span=[start, end], source_fps=source_fps,
                       decoder_gate=gate)
            if not gate["ok"]:
                raise RuntimeError("decoder loses {:.3f}s of this upload's audio; refusing"
                                   .format(gate["declared_s"] - gate["decoded_s"]))
            out_dir = overlay_root / clip
            build_overlay(clip_dir, out_dir)
            decode_span(ffmpeg, source, start / source_fps, (end - start) / source_fps,
                        out_dir / "audio.wav")
            seconds = wav_seconds(out_dir / "audio.wav")
            frames = int(meta["num_frames"])
            rec.update(status="ok", audio_seconds=round(seconds, 4), clip_frames=frames,
                       audio_short_frames=round(frames - seconds * FPS, 2))
            (out_dir / "audio_rebuild.json").write_text(json.dumps({
                "tool": "tools/rebuild_clip_audio.py", "ffmpeg": ffmpeg,
                "ffmpeg_version": subprocess.run([ffmpeg, "-version"], capture_output=True,
                                                 text=True).stdout.splitlines()[0],
                "source": source, "source_frame_span": [start, end], "source_fps": source_fps,
                "sample_rate": SR, "replaces": str(clip_dir / "audio.wav"),
                "decoder_gate": gate}, indent=1) + "\n", encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 -- recorded per clip, counted below
            rec.update(status="failed", error=repr(exc)[:300])
        records.append(rec)
        print(json.dumps(rec), flush=True)
    return records


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clips", required=True, help="one clip id per line (<upload>__clipNNN)")
    ap.add_argument("--overlay-root", required=True, type=pathlib.Path)
    ap.add_argument("--ffmpeg", required=True,
                    help="an ffmpeg that honours the mp4 edit list once (7.0.2 verified)")
    ap.add_argument("--ingest-root", type=pathlib.Path, default=INGEST_ROOT)
    ap.add_argument("--report", required=True, type=pathlib.Path)
    args = ap.parse_args(argv)
    clips = [l.strip() for l in open(args.clips, encoding="utf-8") if l.strip()]
    if len(set(clips)) != len(clips):
        raise SystemExit("duplicate clip ids in --clips")
    records = rebuild(clips, args.overlay_root, args.ffmpeg, args.ingest_root)
    failed = [r for r in records if r["status"] != "ok"]
    summary = {"clips": len(records), "ok": len(records) - len(failed), "failed": len(failed),
               "overlay_root": str(args.overlay_root), "ffmpeg": args.ffmpeg,
               "argv": sys.argv, "records": records}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print("rebuilt {} / {} clips, {} failed -> {}".format(summary["ok"], len(records),
                                                         len(failed), args.report))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
