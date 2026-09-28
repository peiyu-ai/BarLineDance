#!/usr/bin/env python3
"""Materialize frame-exact AtomicDance music features for wild 3D candidates.

This is deliberately a post-HMR step.  It accepts the reconciled sequence
manifest emitted by ``tools/preprocess_wild_3d.py`` and only reads audio for
records whose validated HMR status is ``candidate``.  The source cache's
``audio.wav`` is processed with AtomicDance's released 35-D baseline feature
extractor at 30 FPS, then indexed by the converted motion's original
``frame_ids.npy``.  The tool never pads, resamples, or substitutes the Lodge
39-D rhythm cache: any missing, short, or misaligned audio is quarantined.

The command publishes a new immutable bundle rather than modifying its input:

    <output-dir>/
      sequences_audio.jsonl
      summary.json
      music_35/<sha256(sequence_id)>.npy

One bounded exception to "never pads", and why it is not padding.  A clip
whose decoded audio ends a frame or three before its motion does is the common
case of this quarantine, not a clip with no music: of the 724 rows this tool
quarantined on the wild corpus, 539 were short by exactly one frame, and for
the 汤汤汤小圆 account every one of its 26 was short by 1-3.  Those are tail
rounding, and throwing the clip away costs 16 seconds of corpus to save 0.1.
``--max-tail-trim-frames N`` therefore allows the *motion* to be shortened to
where the audio actually ends -- never the audio to be extended to where the
motion ends.  The trim is recorded as ``audio_feature.tail_trim_frames`` so the
bundle builder can shorten motion and frame_ids by the same amount; an
undeclared length mismatch stays fatal there.  A shortfall larger than ``N``
is still quarantined, which is what keeps genuinely truncated audio out: the
same corpus has rows short by 100-355 frames, and no tail-trim budget anyone
would set reaches them.

``sequences_audio.jsonl`` retains every input record.  A successful row has
``qc.audio_feature_status == 'candidate'`` and an exact 35-D asset; a failed
candidate is explicitly ``quarantine``.  HMR-pending/non-candidate rows pass
through unchanged and are counted as ``not_attempted`` only in ``summary``.
No candidate is promoted to training eligibility: audio materialization is not
a duplicate-content, 3D/human-QC, label, or split-freeze approval.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

# Invoked as ``python3 tools/extract_wild_music_features.py`` -- which puts
# ``tools/`` on sys.path, NOT the repo root -- the extractor's own
# ``data.audio_extraction`` is unimportable, and every clip then fails
# individually.  Measured 2026-09-02: a run over 311 sequences recorded
# ``candidate: 0, quarantine: 290``, one identical "extraction_failed" per
# clip, which reads downstream as "this corpus has no usable audio".
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _restore_scipy_signal_hann() -> None:
    """librosa 0.10.1 calls ``scipy.signal.hann``, removed in scipy 1.13.

    This environment pairs librosa 0.10.1 with scipy 1.15.3, so beat tracking
    dies inside ``librosa.beat.beat_track`` -> ``__trim_beats``.  The alias is
    restored rather than upgrading librosa because ``scipy.signal.hann`` WAS
    ``scipy.signal.windows.hann`` -- the same object under an old name -- so
    this cannot move a feature value, whereas a librosa upgrade could.  The
    corpus this must stay comparable with was extracted 2026-08-25 under the
    pairing where the alias still existed.
    """
    import scipy.signal

    if not hasattr(scipy.signal, "hann"):
        scipy.signal.hann = scipy.signal.windows.hann


_restore_scipy_signal_hann()

import numpy as np


SCHEMA_VERSION = "atomic-wild-music-35d-v1"
FEATURE_DIM = 35
FEATURE_FPS = 30.0
HOP_LENGTH = 512
SAMPLE_RATE = int(FEATURE_FPS * HOP_LENGTH)
_OUTPUT_MANIFEST = "sequences_audio.jsonl"

FeatureExtractor = Callable[[Path], np.ndarray]


class AudioFeatureError(ValueError):
    """A per-sequence input error that must produce a quarantine row."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError("invalid JSONL at {}:{}: {}".format(path, line_number, error))
            if not isinstance(value, Mapping):
                raise ValueError("JSONL item at {}:{} is not an object".format(path, line_number))
            records.append(dict(value))
    return records


def atomicdance_baseline_feature_extractor(audio_path: Path) -> np.ndarray:
    """Run the released AtomicDance 35-D extractor over the full WAV timeline."""
    try:
        from data.audio_extraction import baseline_features
    except ImportError as error:  # pragma: no cover - environment diagnosis
        raise RuntimeError(
            "cannot import AtomicDance baseline audio extractor; install the project's audio dependencies"
        ) from error

    if baseline_features.FPS != int(FEATURE_FPS):
        raise RuntimeError(
            "AtomicDance baseline FPS changed from {} to {}; refusing frame-alignment ambiguity".format(
                FEATURE_FPS, baseline_features.FPS
            )
        )
    if baseline_features.HOP_LENGTH != HOP_LENGTH or baseline_features.SR != SAMPLE_RATE:
        raise RuntimeError("AtomicDance baseline audio sampling contract does not match 30 FPS / 512-hop")
    waveform, _ = baseline_features.librosa.load(str(audio_path), sr=baseline_features.SR)
    # ``max_frames=None`` is essential: the upstream helper otherwise defaults
    # to a five-second preview, which would silently truncate most wild clips.
    return np.asarray(
        baseline_features.extract_audio(waveform, audio_path.stem, max_frames=None), dtype=np.float32
    )


def _strict_frame_ids(path: Path, expected_frames: int) -> np.ndarray:
    if not path.is_file():
        raise AudioFeatureError("missing_frame_ids", "frame_ids asset does not exist: {}".format(path))
    try:
        raw = np.load(str(path), allow_pickle=False)
    except Exception as error:
        raise AudioFeatureError("invalid_frame_ids", "cannot load {}: {}".format(path, error)) from error
    values = np.asarray(raw, dtype=np.float64).reshape(-1)
    if len(values) != expected_frames:
        raise AudioFeatureError(
            "frame_count_mismatch",
            "frame_ids has {} rows but manifest frame_count is {}".format(len(values), expected_frames),
        )
    if expected_frames < 1:
        raise AudioFeatureError("invalid_frame_count", "manifest frame_count must be positive")
    if not np.isfinite(values).all() or not np.all(values == np.rint(values)):
        raise AudioFeatureError("invalid_frame_ids", "frame_ids must be finite exact integers")
    frame_ids = values.astype(np.int64)
    if int(frame_ids[0]) < 0 or np.any(np.diff(frame_ids) <= 0):
        raise AudioFeatureError("invalid_frame_ids", "frame_ids must be non-negative and strictly increasing")
    # Reconciled HMR candidates are required to be contiguous.  Retesting that
    # property here prevents a later stale/tampered manifest from pairing a
    # continuous motion tensor with discontinuous audio rows.
    if len(frame_ids) > 1 and np.any(np.diff(frame_ids) != 1):
        raise AudioFeatureError("noncontiguous_frame_ids", "frame_ids have gaps; no temporal padding is allowed")
    return frame_ids


def select_frame_exact_features(
    full_features: np.ndarray,
    frame_ids: np.ndarray,
    *,
    expected_frames: int,
    max_tail_trim: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and select raw 30-FPS audio rows without padding or interpolation.

    Returns ``(selected, frame_ids_used)``.  ``frame_ids_used`` is the input
    unless a tail trim was applied, in which case it is the input minus its
    last ``shortfall`` entries -- so the caller can record what the audio
    actually covers instead of what the motion claimed.
    """
    features = np.asarray(full_features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise AudioFeatureError(
            "invalid_audio_feature_shape",
            "baseline extractor must return [T, {}], got {}".format(FEATURE_DIM, tuple(features.shape)),
        )
    if not np.isfinite(features).all():
        raise AudioFeatureError("invalid_audio_features", "baseline audio features contain non-finite values")
    if len(frame_ids) != expected_frames:
        raise AudioFeatureError(
            "frame_count_mismatch",
            "frame_ids has {} rows but expected {}".format(len(frame_ids), expected_frames),
        )
    if len(frame_ids) == 0:
        raise AudioFeatureError("invalid_frame_count", "cannot select audio for an empty sequence")
    shortfall = int(frame_ids[-1]) - len(features) + 1
    if shortfall > 0:
        # Trim the motion back to where the audio ends; never the reverse.  The
        # budget is a cap, not a policy: a clip whose audio stops 100+ frames
        # early is a different defect and still fails here.
        if not 0 < shortfall <= int(max_tail_trim) or expected_frames - shortfall < 1:
            raise AudioFeatureError(
                "insufficient_audio_frames",
                "audio has {} feature rows but frame_ids require row {}".format(
                    len(features), int(frame_ids[-1])),
            )
        frame_ids = frame_ids[:-shortfall]
        expected_frames = expected_frames - shortfall
    selected = np.asarray(features[frame_ids], dtype=np.float32)
    if selected.shape != (expected_frames, FEATURE_DIM):  # defensive NumPy-indexing assertion
        raise AudioFeatureError(
            "invalid_audio_feature_shape",
            "selected audio has shape {}, expected [{}, {}]".format(
                tuple(selected.shape), expected_frames, FEATURE_DIM
            ),
        )
    return selected, frame_ids


def _candidate_inputs(record: Mapping[str, Any]) -> Tuple[Path, Path, int, float]:
    sequence_id = record.get("sequence_id")
    if not isinstance(sequence_id, str) or not sequence_id:
        raise AudioFeatureError("missing_sequence_id", "candidate lacks a non-empty sequence_id")
    timeline = record.get("timeline")
    assets = record.get("assets")
    if not isinstance(timeline, Mapping) or not isinstance(assets, Mapping):
        raise AudioFeatureError("invalid_manifest", "candidate lacks timeline or assets mapping")
    try:
        frames = int(timeline.get("frame_count"))
    except (TypeError, ValueError) as error:
        raise AudioFeatureError("invalid_frame_count", "candidate lacks an integer timeline.frame_count") from error
    if frames < 1:
        raise AudioFeatureError("invalid_frame_count", "candidate frame_count must be positive")
    try:
        fps = float(timeline.get("fps"))
    except (TypeError, ValueError) as error:
        raise AudioFeatureError("invalid_fps", "candidate lacks a numeric timeline.fps") from error
    if not math.isfinite(fps) or abs(fps - FEATURE_FPS) > 1e-6:
        raise AudioFeatureError(
            "unsupported_fps",
            "candidate fps {} is not AtomicDance's exact {} FPS audio timeline".format(fps, FEATURE_FPS),
        )
    if timeline.get("motion_frames_are_contiguous") is not True:
        raise AudioFeatureError(
            "noncontiguous_motion",
            "candidate does not attest contiguous motion frames; audio cannot repair a timeline gap",
        )
    frame_ids_value = timeline.get("frame_ids_path")
    source_cache_value = assets.get("source_cache")
    if not isinstance(frame_ids_value, str) or not frame_ids_value:
        raise AudioFeatureError("missing_frame_ids", "candidate lacks timeline.frame_ids_path")
    if not isinstance(source_cache_value, str) or not source_cache_value:
        raise AudioFeatureError("missing_source_cache", "candidate lacks assets.source_cache")
    frame_ids_path = Path(frame_ids_value).expanduser().resolve()
    audio_path = Path(source_cache_value).expanduser().resolve() / "audio.wav"
    if not audio_path.is_file():
        raise AudioFeatureError("missing_audio", "source cache audio.wav does not exist: {}".format(audio_path))
    return audio_path, frame_ids_path, frames, fps


def _reason_codes(qc: Dict[str, Any], code: str) -> None:
    current = qc.get("reason_codes", [])
    if not isinstance(current, list):
        current = [str(current)]
    if code not in current:
        current.append(code)
    qc["reason_codes"] = current


def _base_output_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    item = copy.deepcopy(dict(record))
    item["stage"] = "post_hmr_audio_features"
    assets = item.get("assets")
    qc = item.get("qc")
    if not isinstance(assets, Mapping):
        assets = {}
    if not isinstance(qc, Mapping):
        qc = {}
    item["assets"] = dict(assets)
    item["qc"] = dict(qc)
    item["assets"]["music_35"] = None
    item["assets"]["music_35_sha256"] = None
    # This pipeline is an eligibility *gate*, never a promotion path.
    item["qc"]["accepted_for_training"] = False
    return item


def materialize_wild_music_features(
    records: Sequence[Mapping[str, Any]],
    *,
    artifact_root: Path,
    public_artifact_root: Path,
    input_manifest_sha256: str,
    feature_extractor: Optional[FeatureExtractor] = None,
    max_tail_trim_frames: int = 0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Write candidate music arrays and return a full output manifest.

    ``artifact_root`` is where files are physically written (normally an
    unpublished staging directory); ``public_artifact_root`` is the final
    immutable bundle path written into the manifest.  Keeping them separate
    lets the CLI publish the whole bundle atomically.
    """
    if not isinstance(input_manifest_sha256, str) or len(input_manifest_sha256) != 64:
        raise ValueError("input_manifest_sha256 must be a SHA-256 hex digest")
    max_tail_trim_frames = int(max_tail_trim_frames)
    if max_tail_trim_frames < 0:
        raise ValueError("max_tail_trim_frames must be >= 0")
    seen_ids: set[str] = set()
    for record in records:
        sequence_id = record.get("sequence_id")
        if not isinstance(sequence_id, str) or not sequence_id:
            raise ValueError("input manifest has a record without a non-empty sequence_id")
        if sequence_id in seen_ids:
            raise ValueError("input manifest has duplicate sequence_id: {}".format(sequence_id))
        seen_ids.add(sequence_id)

    artifact_root.mkdir(parents=True, exist_ok=False)
    output: List[Dict[str, Any]] = []
    counts = {"not_attempted": 0, "candidate": 0, "quarantine": 0}
    extractor = feature_extractor
    for record in records:
        original_qc = record.get("qc")
        hmr_status = (
            str(original_qc.get("hmr_status", "unknown"))
            if isinstance(original_qc, Mapping)
            else "unknown"
        )
        if hmr_status != "candidate":
            # The audio stage is not entitled to reinterpret HMR pending or
            # quarantined rows.  Preserve them byte-for-byte at the JSON value
            # level so a later reconciliation can be compared directly.
            output.append(copy.deepcopy(dict(record)))
            counts["not_attempted"] += 1
            continue
        item = _base_output_record(record)
        qc = item["qc"]

        try:
            audio_path, frame_ids_path, frames, fps = _candidate_inputs(item)
            frame_ids = _strict_frame_ids(frame_ids_path, frames)
            if extractor is None:
                extractor = atomicdance_baseline_feature_extractor
            full_features = extractor(audio_path)
            selected, frame_ids = select_frame_exact_features(
                full_features, frame_ids, expected_frames=frames,
                max_tail_trim=max_tail_trim_frames,
            )
            tail_trim = frames - len(frame_ids)
            # Hash before publishing an artifact.  A source-read failure is a
            # per-clip data problem, while a failure writing the new immutable
            # bundle must abort the entire publication rather than masquerade
            # as a bad audio example.
            source_hash = _sha256_file(audio_path)
        except AudioFeatureError as error:
            qc["audio_feature_status"] = "quarantine"
            qc["audio_feature_validation"] = "failed"
            _reason_codes(qc, "audio_feature_{}".format(error.code))
            item["audio_feature"] = {
                "schema_version": SCHEMA_VERSION,
                "status": "quarantine",
                "failure_code": error.code,
                "failure": str(error),
                "input_manifest_sha256": input_manifest_sha256,
            }
            counts["quarantine"] += 1
        except Exception as error:  # per-clip decode/extractor failures are data quarantine, never fallback
            qc["audio_feature_status"] = "quarantine"
            qc["audio_feature_validation"] = "failed"
            _reason_codes(qc, "audio_feature_extraction_failed")
            item["audio_feature"] = {
                "schema_version": SCHEMA_VERSION,
                "status": "quarantine",
                "failure_code": "extraction_failed",
                "failure": "{}: {}".format(type(error).__name__, error),
                "input_manifest_sha256": input_manifest_sha256,
            }
            counts["quarantine"] += 1
        else:
            artifact_name = hashlib.sha256(item["sequence_id"].encode("utf-8")).hexdigest() + ".npy"
            write_path = artifact_root / artifact_name
            if write_path.exists():  # pragma: no cover - staging roots are fresh by construction
                raise RuntimeError("refusing to overwrite music artifact {}".format(write_path))
            np.save(str(write_path), selected)
            artifact_hash = _sha256_file(write_path)
            public_path = (public_artifact_root / artifact_name).resolve()
            item["assets"].update(
                {
                    "music_35": str(public_path),
                    "music_35_sha256": artifact_hash,
                }
            )
            item["audio_feature"] = {
                "schema_version": SCHEMA_VERSION,
                "status": "candidate",
                "extractor": "data.audio_extraction.baseline_features.extract_audio",
                "feature_dim": FEATURE_DIM,
                "fps": FEATURE_FPS,
                "sample_rate": SAMPLE_RATE,
                "hop_length": HOP_LENGTH,
                "source_audio": str(audio_path),
                "source_audio_sha256": source_hash,
                "frame_selection": {
                    # frame_count is what the AUDIO covers, which is the motion's
                    # frame_count minus tail_trim_frames.  Consumers that need
                    # the motion's own length must shorten it by that many.
                    "mode": "source_cache_frame_ids_exact",
                    "frame_ids_path": str(frame_ids_path),
                    "frame_count": int(len(frame_ids)),
                    "manifest_frame_count": frames,
                    "source_start_frame": int(frame_ids[0]),
                    "source_end_frame_exclusive": int(frame_ids[-1]) + 1,
                },
                "tail_trim_frames": int(tail_trim),
                "input_manifest_sha256": input_manifest_sha256,
                "artifact_sha256": artifact_hash,
            }
            qc["audio_feature_status"] = "candidate"
            qc["audio_feature_validation"] = "passed"
            if tail_trim:
                _reason_codes(qc, "audio_feature_tail_trimmed")
                counts["tail_trimmed"] = counts.get("tail_trimmed", 0) + 1
            counts["candidate"] += 1
        output.append(item)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "stage": "post_hmr_audio_features",
        "input_sequences": len(records),
        "input_manifest_sha256": input_manifest_sha256,
        "status_counts": counts,
        "feature_contract": {
            "feature_dim": FEATURE_DIM,
            "fps": FEATURE_FPS,
            "sample_rate": SAMPLE_RATE,
            "hop_length": HOP_LENGTH,
            "extractor": "data.audio_extraction.baseline_features.extract_audio",
            "frame_alignment": "select source-cache rows by exact converted frame_ids; no padding/resampling",
            "max_tail_trim_frames": max_tail_trim_frames,
        },
        "training_eligibility": (
            "none; audio candidate still requires duplicate-content QC, 3D/human QC, "
            "atomic labels, and a frozen source split"
        ),
    }
    return output, summary


def publish_wild_music_bundle(
    input_path: Path,
    output_dir: Path,
    *,
    feature_extractor: Optional[FeatureExtractor] = None,
    max_tail_trim_frames: int = 0,
) -> Dict[str, Any]:
    """Atomically publish an immutable audio-manifest bundle from JSONL input."""
    input_path = input_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError("input sequence manifest does not exist: {}".format(input_path))
    if output_dir.exists():
        raise FileExistsError(
            "refusing to overwrite existing audio bundle {}; choose a new versioned output directory".format(
                output_dir
            )
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    input_hash = _sha256_file(input_path)
    staging = Path(tempfile.mkdtemp(prefix=".{}-staging-".format(output_dir.name), dir=str(output_dir.parent)))
    try:
        records, summary = materialize_wild_music_features(
            _read_jsonl(input_path),
            artifact_root=staging / "music_35",
            public_artifact_root=output_dir / "music_35",
            input_manifest_sha256=input_hash,
            feature_extractor=feature_extractor,
            max_tail_trim_frames=max_tail_trim_frames,
        )
        _write_jsonl(staging / _OUTPUT_MANIFEST, records)
        summary.update(
            {
                "input_manifest": str(input_path),
                "output_manifest": str((output_dir / _OUTPUT_MANIFEST).resolve()),
                "output_manifest_sha256": _sha256_file(staging / _OUTPUT_MANIFEST),
                "bundle_layout": {"music_features": "music_35/<sha256(sequence_id)>.npy"},
            }
        )
        with (staging / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        # ``rename`` is an atomic same-filesystem publish and, unlike an
        # overwrite-oriented replace, preserves the immutable-output contract.
        os.rename(str(staging), str(output_dir))
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-sequences",
        required=True,
        help="post-HMR JSONL from preprocess_wild_3d.py reconcile-wild-hmr",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="new immutable bundle directory; it must not already exist",
    )
    parser.add_argument(
        "--max-tail-trim-frames",
        type=int,
        default=0,
        help=(
            "shorten a clip's motion by up to N frames when its audio ends that "
            "many frames early, instead of quarantining it.  0 (the default) is "
            "the historical behaviour.  Audio is never extended; a shortfall "
            "larger than N is still quarantined."
        ),
    )
    return parser


def preflight_extractor() -> None:
    """Fail the RUN, not every clip, when the extractor itself is unavailable.

    Without this the missing import is caught per sequence and written into
    each record as ``extraction_failed``, so an environment fault is published
    as a data verdict: the 2026-09-02 run over 311 sequences reported
    ``candidate: 0, quarantine: 290`` with 290 byte-identical failures, which
    downstream is indistinguishable from "this account's audio is unusable".
    One import, once, before any work: it costs nothing and it can only fail
    for the one reason it names.
    """
    try:
        from data.audio_extraction import baseline_features
    except ImportError as error:
        raise SystemExit(
            "error: the AtomicDance 35-D audio extractor is not importable "
            "({}).  Every clip would be quarantined as extraction_failed, "
            "which is an environment fault reported as a data verdict.  Run "
            "from the repository root, or set PYTHONPATH to it.".format(error))

    # Importing is not running.  The 2026-09-02 run got past the import and
    # still quarantined all 290 clips, first on a missing ``audioread`` and
    # then inside librosa's beat tracker -- both environment faults, both
    # published per clip.  So the preflight extracts from a synthetic tone:
    # it exercises load, onset, beat-track and the feature stack, which is
    # every dependency a real clip would touch.
    # ``extract_audio`` with max_frames=None is the production call
    # (see atomicdance_baseline_feature_extractor); ``extract`` is the
    # AIST-only wrapper that parses tempo out of the filename and caps at five
    # seconds, so preflighting through it would fail for reasons no wild clip
    # ever hits.
    rate = int(baseline_features.SR)
    samples = np.linspace(0.0, 6.0, rate * 6, endpoint=False)
    tone = 0.2 * np.sin(2.0 * np.pi * 220.0 * samples)
    tone[:: rate // 2] += 0.5          # a click train, so beat tracking has work
    try:
        features = np.asarray(
            baseline_features.extract_audio(tone.astype(np.float32), "preflight",
                                            max_frames=None))
    except Exception as error:                        # noqa: BLE001
        raise SystemExit(
            "error: the 35-D audio extractor imports but cannot run "
            "({}: {}).  Refusing to start, because this would be recorded "
            "as one extraction_failed per clip and read downstream as a "
            "verdict about the corpus.".format(type(error).__name__, error))
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise SystemExit(
            "error: the 35-D audio extractor returned shape {}, expected "
            "[frames, {}]".format(features.shape, FEATURE_DIM))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    preflight_extractor()
    try:
        summary = publish_wild_music_bundle(
            Path(args.input_sequences),
            Path(args.output_dir),
            max_tail_trim_frames=args.max_tail_trim_frames,
        )
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        raise SystemExit("error: {}".format(error))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
