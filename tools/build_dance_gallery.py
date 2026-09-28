#!/usr/bin/env python3
"""Build a watchable side-by-side gallery of what the model actually produces.

Per-model videos already exist, but judging a dance model means judging it
against the ground truth it was trained on, at full speed, with the music
playing.  This stacks ground truth on top of each model's generation for the
same song into one clip, muxes the conditioning audio, and writes an index page
that plays them in a browser.

Ground truth is a real held-out performance of that song read from the release
as raw 151-D and decoded through the same forward kinematics the generated clip
went through, so a difference on screen is a difference in the motion and not
in the renderer.  It is deliberately *a* performance, not *the* answer: AIST++
gives every song many valid choreographies, which is the whole reason frame
accuracy is meaningless here and structure is what gets scored.

Two corpora, and the differences are arguments rather than options:

* **AIST (``--raw-performance``).**  The held-out unit is a song, ground truth is
  picked out of ``sequences.jsonl`` by song id, and the music is one WAV per song.
* **Wild (``--ground-truth-dir``).**  There is no song id, so the unit is the clip
  and ground truth is the per-clip ``.pkl`` ``tools/export_wild_eval_motion.py``
  already wrote -- the same array the FID ground-truth features were built from.
  The music is not a WAV per clip either: it lives inside the ingest tree as
  ``<upload>__<clipNNN>/audio.wav``, which is what ``--audio-layout ingest`` walks.
  Nothing falls back between the two layouts: a gallery that silently renders a
  dance with no music has removed the thing it was built to let a person judge,
  so an unresolved clip is counted and printed instead.

The rows are also checked against each other before anything is rendered.  Two
arms in one stack are only comparable if they were sampled the same way, and on
2026-08-16 this repo published a MultiModality number averaged over two different
plan strides precisely because nothing knew the two runs were meant to match.
So each run's ``manifest.json`` is read and the sampler flags compared, and a
mismatch stops the build unless ``--allow-mismatched-sampling`` says otherwise.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import re
import subprocess
import sys
import tempfile
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.render_dance_video import load_motion, render  # noqa: E402

_SONG = re.compile(r"_(?P<song>m[A-Z]{2}\d)_")
GT_LABEL = "ground truth"

# A wild clip id is ``<corpus>:<upload>:<clipNNN>``; the ingest tree names the
# same clip ``<upload>__<clipNNN>``.  Parsed rather than string-replaced so a id
# that does not have this shape fails here instead of resolving to some other
# clip's audio.
_WILD_CLIP = re.compile(r"^(?P<corpus>[^:]+):(?P<upload>[^:]+):(?P<clip>[^:]+)$")

# The flags that decide *what* was sampled rather than which draw came out.  Two
# runs stacked in one video must agree on all of them; ``seed`` is deliberately
# absent, because comparing two seeds of one arm is a legitimate use of this
# tool and is visible in the manifest either way.
SAMPLING_KEYS = (
    "temperature",
    "deterministic_planner",
    "completion_stride",
    "plan_stride",
    "plan_fusion",
    "guidance_weight",
    "draft_noise_ratio",
    "ground_truth_labels",
    # Added 2026-08-28.  Everything above dates from before 2026-08-23, when the
    # plan gained five more knobs that decide what is on screen -- and this list
    # is the only thing standing between two arms that differ on one of them and
    # a reader who will credit the difference to the checkpoint.  They are not
    # cosmetic: run_m6_wild.sh records that the tie-break ALONE was 4.6 of the
    # 22.0 points by which one arm's plan overshot the ground truth's transition
    # share, and that the bar grid moved it 0.5194 -> 0.3482.  A stack of a
    # gridded and an ungridded arm is a legitimate thing to build -- that
    # difference can be the subject -- but it has to be declared with
    # --allow-mismatched-sampling and printed, not passed over in silence.
    "plan_bar_grid",
    "plan_vote_tie_break",
    "plan_transition_policy",
    "plan_merge_order",
    "planner_transition_logit_bias",
    # Added 2026-09-01, and by the same route: ``plan_bar_beats`` was not in
    # ANY artifact, so a rerun reconstructed from a manifest silently used the
    # CLI default 4 where the shipped arm had used 2, and produced 8.5 atomic
    # segments per clip against 14.1 -- section 15.4's control rather than the
    # arm it was meant to reproduce.  The scorecard caught it (seg/s 0.493 vs
    # 0.842); nothing else would have.
    "plan_bar_beats",
    # The retrieval rule decides WHICH motion is pasted, so two rules stacked in
    # one video are two different dances.  That comparison is a legitimate
    # subject -- it is the point of the learned selector -- but like the bar
    # grid above it has to be declared with --allow-mismatched-sampling and
    # printed, not passed over.
    "retrieval_rule",
    "retrieval_selector",
)


class GalleryError(RuntimeError):
    pass


def song_of(name: str) -> Optional[str]:
    match = _SONG.search(name)
    return match.group("song") if match else None


def pick_ground_truth(
    raw_root: pathlib.Path, songs: Sequence[str], *, prefer_split: str = "val"
) -> Dict[str, Dict[str, object]]:
    """One held-out performance per song, preferring the split the models were scored on."""
    manifest = raw_root / "sequences.jsonl"
    if not manifest.is_file():
        raise GalleryError("no sequences.jsonl under {}".format(raw_root))
    wanted = set(songs)
    best: Dict[str, Dict[str, object]] = {}
    for line in manifest.open(encoding="utf-8"):
        record = json.loads(line)
        name = str(record["sequence_id"]).split("/")[1]
        song = song_of(name)
        if song not in wanted:
            continue
        current = best.get(song)
        rank = (record.get("split") == prefer_split, int(record.get("frame_count", 0)))
        if current is None or rank > current["_rank"]:
            best[song] = {
                "_rank": rank,
                "sequence": name,
                "split": record.get("split"),
                "frames": int(record.get("frame_count", 0)),
                "motion_path": raw_root / str(record["motion_path"]),
            }
    for entry in best.values():
        entry.pop("_rank")
    return best


def index_clip_ground_truth(
    directory: pathlib.Path, clips: Sequence[str]
) -> Dict[str, Dict[str, object]]:
    """One decoded ``.pkl`` per clip id -- the wild corpus's ground truth.

    No choosing happens here, and that is the difference from ``pick_ground_truth``:
    on AIST a song has many held-out performances and one is picked; a wild clip
    *is* the held-out unit, so the ground truth is that clip's own reconstruction.
    A clip present in every run but absent here is reported rather than skipped,
    because a stack silently missing its top row looks exactly like a stack whose
    top row is the model.
    """
    found: Dict[str, Dict[str, object]] = {}
    for clip in clips:
        path = directory / "{}.pkl".format(clip)
        if not path.is_file():
            continue
        found[clip] = {
            "sequence": clip,
            "split": None,
            "frames": None,
            "motion_path": path,
        }
    return found


def resolve_audio(
    audio_dir: Optional[pathlib.Path], clip: str, layout: str
) -> Optional[pathlib.Path]:
    """The clip's own music, under whichever layout the corpus stores it in.

    ``flat``   ``<audio_dir>/<clip>.wav``            -- AIST's one WAV per song.
    ``ingest`` ``<audio_dir>/<upload>__<clipNNN>/audio.wav`` -- the wild ingest tree.

    There is no fallback from one to the other on purpose.  The wild eval's
    ``audio`` directory holds the planner's 35-D ``.npy`` features under exactly
    the flat naming, so a fallback would find a file, fail to mux it, and leave a
    silent video that still looked configured.
    """
    if audio_dir is None:
        return None
    if layout == "flat":
        candidate = audio_dir / "{}.wav".format(clip)
    elif layout == "ingest":
        match = _WILD_CLIP.match(clip)
        if match is None:
            return None
        candidate = audio_dir / "{}__{}".format(
            match.group("upload"), match.group("clip")
        ) / "audio.wav"
    else:
        raise GalleryError("unknown audio layout {!r}".format(layout))
    return candidate if candidate.is_file() else None


def read_sampling(run: pathlib.Path) -> Optional[Mapping[str, object]]:
    """The ``sampling`` block infer_atomic.py wrote beside the generated clips."""
    manifest = run / "manifest.json"
    if not manifest.is_file():
        return None
    try:
        return json.loads(manifest.read_text(encoding="utf-8")).get("sampling")
    except (json.JSONDecodeError, OSError):
        return None


def check_sampling(
    runs: Sequence[Tuple[str, pathlib.Path]], *, strict: bool = True
) -> Dict[str, object]:
    """Compare the arms' sampler flags; refuse a stack of runs that disagree.

    Checked rather than assumed, for the reason recorded in
    ``tools/run_m6_wild_multimodality.sh``: two runs meant to match differed on
    ``plan_stride`` for a day and the only symptom was a metric that had quietly
    become an average over two generators.  A stacked video makes that worse, not
    better -- the reader attributes the difference to the checkpoint.
    """
    observed = {label: read_sampling(path) for label, path in runs}
    missing = sorted(label for label, block in observed.items() if block is None)
    differing: Dict[str, Dict[str, object]] = {}
    present = {label: block for label, block in observed.items() if block is not None}
    for key in SAMPLING_KEYS:
        values = {label: block.get(key) for label, block in present.items()}
        if len(set(map(repr, values.values()))) > 1:
            differing[key] = values
    report = {
        "checked_keys": list(SAMPLING_KEYS),
        "runs_without_manifest": missing,
        "differing": differing,
        "seeds": {label: (block or {}).get("seed") for label, block in observed.items()},
    }
    if strict and differing:
        raise GalleryError(
            "the runs were not sampled the same way, so stacking them would "
            "attribute a sampler difference to the checkpoint: {} "
            "(pass --allow-mismatched-sampling to override)".format(
                json.dumps(differing, sort_keys=True)
            )
        )
    return report


def stack_videos(
    clips: Sequence[pathlib.Path], audio: Optional[pathlib.Path], output: pathlib.Path
) -> None:
    """Vertically stack same-width clips; the shortest one bounds the result."""
    if not clips:
        raise GalleryError("nothing to stack")
    output.parent.mkdir(parents=True, exist_ok=True)
    command: List[str] = ["ffmpeg", "-y", "-loglevel", "error"]
    for clip in clips:
        command += ["-i", str(clip)]
    if audio is not None:
        command += ["-i", str(audio)]
    if len(clips) == 1:
        filter_graph = "[0:v]copy[v]"
    else:
        inputs = "".join("[{}:v]".format(i) for i in range(len(clips)))
        filter_graph = "{}vstack=inputs={}[v]".format(inputs, len(clips))
    command += ["-filter_complex", filter_graph, "-map", "[v]"]
    if audio is not None:
        # MP3: the page these clips are reviewed in may be an editor preview,
        # which plays MP3 and is silent on AAC while reporting nothing wrong.
        command += ["-map", "{}:a".format(len(clips)),
                    "-c:a", "libmp3lame", "-ar", "44100", "-ac", "2", "-b:a", "160k"]
    command += ["-pix_fmt", "yuv420p", "-crf", "23", "-shortest", str(output)]
    subprocess.run(command, check=True)


def probe_duration(path: pathlib.Path) -> Optional[float]:
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(path),
            ],
            check=True, capture_output=True, text=True,
        )
        return round(float(out.stdout.strip()), 1)
    except (subprocess.CalledProcessError, ValueError):
        return None


def decode_ground_truth(
    entry: Mapping[str, object], *, max_frames: Optional[int]
) -> Tuple[np.ndarray, Mapping[str, object]]:
    """Truth through the same decode a generated clip takes.

    Two storage forms, one decode.  AIST's raw 151-D goes through
    ``render_dance_video``'s own ``.npy`` branch -- the point of the round-trip
    through a temporary file is that the truth and the generation reach the
    screen by the same code path.  The wild corpus stores an already-decoded
    ``.pkl`` with ``full_pose``, written by the export that also produced the FID
    ground-truth features, so the same array that is being scored is the one
    being watched.
    """
    source = pathlib.Path(entry["motion_path"])
    if source.suffix == ".pkl":
        poses, payload = load_motion(source)
        if max_frames is not None:
            poses = poses[:max_frames]
            payload = dict(payload)
            for key in ("smpl_poses", "smpl_trans", "full_pose", "contacts"):
                value = payload.get(key)
                if isinstance(value, np.ndarray) and len(value) > max_frames:
                    payload[key] = value[:max_frames]
        return poses, payload
    motion = np.load(source)
    if max_frames is not None:
        motion = motion[:max_frames]
    with tempfile.TemporaryDirectory() as staging:
        path = pathlib.Path(staging) / "truth.npy"
        np.save(path, motion)
        return load_motion(path)


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "run"


CONTACT_JOINTS = (7, 8, 10, 11)  # L/R ankle, L/R toe -- the release's own order.
FPS = 30.0
# How close to a clip's own floor a foot has to be before its horizontal motion
# counts as sliding rather than stepping.  5 cm is the height of the ankle-joint
# centre above the ground in a normal stance to within roughly its own radius;
# the number is reported with the statistic so a reader can see it was chosen.
GROUND_BAND_M = 0.05
# The floor is the clip's own, taken as a low percentile rather than the minimum:
# one frame of a foot punched through the ground would otherwise define the
# ground plane, and wild reconstructions do that.
FLOOR_PERCENTILE = 5.0


def motion_diagnostics(poses: np.ndarray, payload: Mapping[str, object]) -> Dict[str, object]:
    """Cheap physical-plausibility numbers to put beside the video.

    Two foot-slide numbers, and the difference between them is the point:

    * ``foot_skate_m_per_s`` uses the clip's **own contact channel**, so it is
      self-consistency: a model can only fail it by contradicting itself, which
      no amount of choreographic freedom excuses.  It exists only where that
      channel does -- ``infer_atomic``'s output has one, and the wild ground
      truth does not, since ``export_wild_eval_motion`` writes ``full_pose``
      alone.
    * ``foot_skate_geometric_m_per_s`` asks a **geometric** question instead --
      how fast a foot moves horizontally while it is within ``GROUND_BAND_M`` of
      this clip's own floor -- and therefore has a value on every row including
      ground truth.  That is what makes it the column to compare *against* truth,
      and the contact-channel column the one to compare *between* arms.

    They are not interchangeable and are never averaged together: the first is
    scored against a channel the model emitted, the second against geometry.
    """
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (24, 3):
        return {}
    speed = np.linalg.norm(np.diff(poses, axis=0), axis=-1) * FPS  # [T-1, 24]
    feet = poses[:, CONTACT_JOINTS, :]
    slide = np.linalg.norm(np.diff(feet[..., :2], axis=0), axis=-1) * FPS  # [T-1, 4]
    stats: Dict[str, object] = {
        "frames": int(len(poses)),
        "median_root_height_m": round(float(np.median(poses[:, 0, 2])), 3),
        "median_joint_speed_m_per_s": round(float(np.median(speed)), 3),
        "lowest_joint_z_m": round(float(poses[..., 2].min()), 3),
    }

    floor = float(np.percentile(feet[..., 2].min(axis=1), FLOOR_PERCENTILE))
    near_ground = feet[:-1, :, 2] <= floor + GROUND_BAND_M
    stats["grounded_fraction_geometric"] = round(float(near_ground.mean()), 3)
    if near_ground.any():
        stats["foot_skate_geometric_m_per_s"] = round(float(slide[near_ground].mean()), 3)

    contacts = payload.get("contacts") if isinstance(payload, Mapping) else None
    if contacts is not None:
        contacts = np.asarray(contacts, dtype=np.float64)
        if contacts.shape[:1] == poses.shape[:1] and contacts.shape[-1] == len(CONTACT_JOINTS):
            grounded = contacts[:-1] > 0.5
            stats["contact_fraction"] = round(float(grounded.mean()), 3)
            if grounded.any():
                stats["foot_skate_m_per_s"] = round(float(slide[grounded].mean()), 3)
    return stats


def build(
    *,
    runs: Sequence[Tuple[str, pathlib.Path]],
    raw_root: Optional[pathlib.Path] = None,
    ground_truth_dir: Optional[pathlib.Path] = None,
    audio_dir: Optional[pathlib.Path],
    output_dir: pathlib.Path,
    include_ground_truth: bool = True,
    follow_root: bool = True,
    audio_layout: str = "flat",
    limit: Optional[int] = None,
    allow_mismatched_sampling: bool = False,
) -> Dict[str, object]:
    if include_ground_truth and (raw_root is None) == (ground_truth_dir is None):
        raise GalleryError(
            "give exactly one ground-truth source: --raw-performance (AIST, "
            "raw 151-D by song) or --ground-truth-dir (wild, one .pkl per clip)"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    sampling = check_sampling(runs, strict=not allow_mismatched_sampling)
    shared = sorted(
        set.intersection(*[{p.stem for p in path.glob("*.pkl")} for _, path in runs])
    )
    if not shared:
        raise GalleryError("the given runs share no generated song")
    # Truncation is printed rather than silent: a gallery of the first four of
    # six hundred clips is a sample, and which four is part of reading it.
    songs = shared[:limit] if limit else shared

    if not include_ground_truth:
        truth: Dict[str, Dict[str, object]] = {}
    elif ground_truth_dir is not None:
        truth = index_clip_ground_truth(ground_truth_dir, songs)
    else:
        truth = pick_ground_truth(raw_root, songs)
    missing_truth = [song for song in songs if song not in truth] if include_ground_truth else []
    entries: List[Dict[str, object]] = []
    missing_audio: List[str] = []

    for song in songs:
        audio = resolve_audio(audio_dir, song, audio_layout)
        if audio is None:
            missing_audio.append(song)
        rows: List[Dict[str, object]] = []
        clips: List[pathlib.Path] = []
        stem = _slug(song)

        generated_frames = []
        for _, run in runs:
            with (run / "{}.pkl".format(song)).open("rb") as handle:
                generated_frames.append(len(pickle.load(handle)["full_pose"]))

        gt = truth.get(song)
        if gt is not None:
            gt_video = output_dir / "ground_truth" / "{}.mp4".format(stem)
            poses, payload = decode_ground_truth(gt, max_frames=max(generated_frames))
            if not gt_video.is_file():
                title = "{} | {}".format(GT_LABEL, gt["sequence"])
                if gt["split"]:
                    title += " ({})".format(gt["split"])
                render(poses, gt_video, audio, title, 1, follow_root)
            clips.append(gt_video)
            rows.append(
                {
                    "label": GT_LABEL,
                    "detail": "{}{}".format(
                        gt["sequence"], " ({})".format(gt["split"]) if gt["split"] else ""
                    ),
                    "video": str(gt_video.relative_to(output_dir)),
                    "seconds": probe_duration(gt_video),
                    "diagnostics": motion_diagnostics(poses, payload),
                }
            )

        # The gallery renders every row itself rather than reusing a run's own
        # videos: rows must share one camera policy or the comparison is between
        # framings, not between dances.
        for label, run in runs:
            video = output_dir / _slug(label) / "{}.mp4".format(stem)
            poses, payload = load_motion(run / "{}.pkl".format(song))
            if not video.is_file():
                render(poses, video, audio, "{} | {}".format(label, song), 1, follow_root)
            clips.append(video)
            rows.append(
                {
                    "label": label,
                    "detail": str(run),
                    "video": str(video.relative_to(output_dir)),
                    "seconds": probe_duration(video),
                    "diagnostics": motion_diagnostics(poses, payload),
                }
            )

        compare = output_dir / "compare" / "{}.mp4".format(stem)
        if not compare.is_file():
            stack_videos(clips, audio, compare)
        entries.append(
            {
                "song": song,
                "compare": str(compare.relative_to(output_dir)),
                "seconds": probe_duration(compare),
                "rows": rows,
                "audio": str(audio) if audio else None,
            }
        )

    manifest = {
        "generated_songs": len(entries),
        "runs": [{"label": label, "path": str(path)} for label, path in runs],
        "ground_truth_source": str(ground_truth_dir or raw_root),
        "ground_truth_mode": "per-clip decoded .pkl" if ground_truth_dir else "raw 151-D by song",
        "layout": " / ".join([GT_LABEL] * bool(truth) + [label for label, _ in runs]),
        "camera": "root-following (floor plane)" if follow_root else "fixed",
        # A gallery that quietly dropped a row or a soundtrack still renders, so
        # the counts travel with it rather than being left to the eye.
        "selection": "first {} of {} clip(s) present in every run, sorted".format(
            len(songs), len(shared)
        ),
        "candidates_total": len(shared),
        "audio_layout": audio_layout,
        "clips_without_audio": missing_audio,
        "clips_without_ground_truth": missing_truth,
        "sampling_check": sampling,
        "entries": entries,
    }
    (output_dir / "gallery.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "index.html").write_text(_render_html(manifest), encoding="utf-8")
    return manifest


DIAGNOSTIC_COLUMNS = (
    ("median_root_height_m", "root height (m)"),
    ("median_joint_speed_m_per_s", "joint speed (m/s)"),
    ("grounded_fraction_geometric", "grounded (geom.)"),
    ("foot_skate_geometric_m_per_s", "skate, geom. (m/s)"),
    ("contact_fraction", "contact fraction"),
    ("foot_skate_m_per_s", "skate, own channel (m/s)"),
)


def _diagnostics_table(entry: Mapping[str, object]) -> str:
    if not any(row.get("diagnostics") for row in entry["rows"]):
        return ""
    header = "".join("<th>{}</th>".format(title) for _, title in DIAGNOSTIC_COLUMNS)
    body = []
    for row in entry["rows"]:
        stats = row.get("diagnostics") or {}
        cells = "".join(
            "<td>{}</td>".format(stats.get(key, "—")) for key, _ in DIAGNOSTIC_COLUMNS
        )
        body.append("<tr><th scope=\"row\">{}</th>{}</tr>".format(row["label"], cells))
    return (
        '<table><thead><tr><th scope="col"></th>{}</tr></thead><tbody>{}</tbody></table>'.format(
            header, "".join(body)
        )
    )


def _render_html(manifest: Mapping[str, object]) -> str:
    rows = []
    for entry in manifest["entries"]:
        legend = " · ".join(
            "<b>{}</b> {}".format(row["label"], row["detail"]) for row in entry["rows"]
        )
        rows.append(
            """
    <section>
      <h2>{song} <small>{seconds}s</small></h2>
      <video controls preload="metadata" src="{compare}"></video>
      <p class="legend">{legend}</p>
      {table}
    </section>""".format(
                song=entry["song"],
                seconds=entry["seconds"],
                compare=entry["compare"],
                legend=legend,
                table=_diagnostics_table(entry),
            )
        )
    wild = manifest.get("ground_truth_mode") == "per-clip decoded .pkl"
    truth_note = (
        "Ground truth is that clip's own 3D reconstruction — the same array the FID "
        "ground-truth features were built from, not a second performance of the song."
        if wild else
        "Ground truth is one held-out performance of the same song, not the only right "
        "answer — AIST++ gives every song many valid choreographies."
    )
    caveats = ["<b>Selection:</b> {}.".format(manifest.get("selection", "—"))]
    silent = manifest.get("clips_without_audio") or []
    if silent:
        caveats.append(
            "<b>{} clip(s) rendered without music</b> ({}) — those rows cannot be "
            "judged for musicality at all.".format(len(silent), ", ".join(silent[:3]))
        )
    differing = (manifest.get("sampling_check") or {}).get("differing") or {}
    if differing:
        caveats.append(
            "<b>The runs were sampled differently</b> ({}) — a difference on screen is "
            "not attributable to the checkpoint alone.".format(
                ", ".join(sorted(differing))
            )
        )
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AtomicDance generations</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 15px/1.5 system-ui, sans-serif; margin: 0 auto; max-width: 1040px; padding: 2rem 1rem 4rem; }}
  h1 {{ font-size: 1.4rem; margin-bottom: .25rem; }}
  .lede {{ opacity: .75; margin-top: 0; }}
  section {{ margin: 2.5rem 0; }}
  h2 {{ font-size: 1.1rem; margin-bottom: .5rem; }}
  h2 small {{ font-weight: 400; opacity: .6; }}
  video {{ width: 100%; background: #000; border-radius: 6px; }}
  .legend {{ font-size: 13px; opacity: .75; margin-top: .4rem; }}
  table {{ border-collapse: collapse; font-size: 13px; margin-top: .6rem; width: 100%; }}
  th, td {{ border-bottom: 1px solid rgba(128,128,128,.35); padding: .3rem .5rem; text-align: right; }}
  thead th {{ font-weight: 600; opacity: .7; }}
  tbody th {{ text-align: left; font-weight: 600; }}
  code {{ font-size: 13px; }}
</style>
</head>
<body>
<h1>AtomicDance — {heading}</h1>
<p class="lede">{count} clip(s). Each stacks, top to bottom: {layout}. Full speed, with the
conditioning music. {truth_note}</p>
<p class="lede">The tables are plausibility, not accuracy. <b>Skate, own channel</b> is how fast the
feet slide on the frames a clip <em>itself</em> marks as ground contact — choreographic freedom
excuses a different dance, never a clip contradicting its own contact channel — and it is blank
for ground truth, which carries no such channel. <b>Skate, geom.</b> asks the same question of
geometry alone (a foot within {band} m of that clip's own floor), so it is the column that has a
ground-truth row to be read against.</p>
<p class="lede">{caveats}</p>
{rows}
<p><code>gallery.json</code> carries the same manifest in machine-readable form.</p>
</body>
</html>
""".format(
        heading="held-out wild clips" if wild else "val song generations",
        band=GROUND_BAND_M,
        count=manifest["generated_songs"],
        layout=manifest["layout"],
        truth_note=truth_note,
        caveats=" ".join(caveats),
        rows="\n".join(rows),
    )


def _run_spec(value: str) -> Tuple[str, pathlib.Path]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("expected label:path, got {}".format(value))
    label, path = value.split(":", 1)
    return label, pathlib.Path(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        type=_run_spec,
        required=True,
        help="label:path of an infer_atomic.py output directory; repeatable",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--raw-performance",
        type=pathlib.Path,
        help="AIST: the raw-performance release whose sequences.jsonl holds the "
             "held-out performances (default when neither source is given)",
    )
    source.add_argument(
        "--ground-truth-dir",
        type=pathlib.Path,
        help="wild: a directory of <clip>.pkl written by tools/export_wild_eval_motion.py",
    )
    parser.add_argument("--audio-dir", type=pathlib.Path, default=pathlib.Path("data/aist_music_val"))
    parser.add_argument(
        "--audio-layout",
        choices=("flat", "ingest"),
        default="flat",
        help="flat: <audio-dir>/<clip>.wav (AIST).  ingest: "
             "<audio-dir>/<upload>__<clipNNN>/audio.wav (the wild ingest tree). "
             "No fallback between them -- see resolve_audio",
    )
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("runs/gallery"))
    parser.add_argument("--no-ground-truth", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        help="render only the first N shared clips, sorted; the manifest and the "
             "page both record N out of how many",
    )
    parser.add_argument(
        "--allow-mismatched-sampling",
        action="store_true",
        help="stack runs whose sampler flags disagree (recorded on the page either way)",
    )
    parser.add_argument(
        "--fixed-camera",
        action="store_true",
        help="frame each clip by its whole travel path instead of following the root",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    raw_root, ground_truth_dir = args.raw_performance, args.ground_truth_dir
    if raw_root is None and ground_truth_dir is None:
        raw_root = pathlib.Path("data/atomic_aistpp/aist_raw_performance_v1")
    try:
        manifest = build(
            runs=args.runs,
            raw_root=raw_root,
            ground_truth_dir=ground_truth_dir,
            audio_dir=args.audio_dir,
            output_dir=args.output_dir,
            include_ground_truth=not args.no_ground_truth,
            follow_root=not args.fixed_camera,
            audio_layout=args.audio_layout,
            limit=args.limit,
            allow_mismatched_sampling=args.allow_mismatched_sampling,
        )
    except (GalleryError, subprocess.CalledProcessError) as error:
        raise SystemExit("error: {}".format(error))
    silent = manifest["clips_without_audio"]
    orphan = manifest["clips_without_ground_truth"]
    print(
        "gallery: {} clip(s) of {} -> {}".format(
            manifest["generated_songs"], manifest["candidates_total"],
            args.output_dir / "index.html",
        )
    )
    if silent:
        print("   WARNING: {} clip(s) rendered silent: {}".format(len(silent), ", ".join(silent)))
    if orphan:
        print("   WARNING: {} clip(s) with no ground truth: {}".format(
            len(orphan), ", ".join(orphan)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
