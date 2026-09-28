"""Extract all AIST++ evaluation features from motion PKLs and WAV files."""

import argparse
import json
import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from eval.utils.kinetic import extract_kinetic_features
from eval.utils.manual import extract_manual_features
from eval.utils.motionbeat import extract_dance_beat_features
from eval.utils.musicbeat import beat_channel_from_features, extract_music_beat_features


_WORKER_SMPL = None
_WORKER_SMPL_PATH = None
DEFAULT_SMPL_MODEL = "smpl/SMPL_MALE.pkl"


def _import_smpl_class():
    from smplx import SMPL

    return SMPL


def _load_smpl_class():
    """Import SMPL only for PKLs that require SMPL forward kinematics.

    Most generated/evaluation PKLs already contain ``full_pose`` joints.  Those
    paths should remain usable in lightweight environments that intentionally do
    not install the optional SMPL runtime or model assets.
    """
    try:
        return _import_smpl_class()
    except ModuleNotFoundError as error:
        # Do not hide an unrelated missing transitive dependency from smplx.
        if error.name != "smplx":
            raise
        raise ModuleNotFoundError(
            "SMPL-dependent feature extraction requires the optional 'smplx' "
            "package. Install the evaluation dependencies before evaluating "
            "PKLs without full_pose."
        ) from error


def _smpl_model(model_path):
    global _WORKER_SMPL, _WORKER_SMPL_PATH
    if model_path is None:
        raise ValueError("--smpl-model is required for PKLs without full_pose")
    if _WORKER_SMPL is None or _WORKER_SMPL_PATH != model_path:
        smpl_class = _load_smpl_class()
        _WORKER_SMPL = smpl_class(model_path=model_path, gender="MALE", batch_size=1)
        _WORKER_SMPL_PATH = model_path
    return _WORKER_SMPL


def z_up_to_y_up(positions):
    positions = np.asarray(positions)
    return np.stack((positions[..., 0], positions[..., 2], -positions[..., 1]), axis=-1)


def _resample_integer(positions, source_fps, target_fps):
    if source_fps == target_fps:
        return positions
    if source_fps % target_fps != 0:
        raise ValueError("motion FPS must be an integer multiple of evaluation FPS")
    return positions[:: source_fps // target_fps]


def load_keypoints(path, smpl_model=None, evaluation_fps=30, generated_fps=30, raw_fps=60):
    with open(str(path), "rb") as handle:
        data = pickle.load(handle)
    if "full_pose" in data:
        positions = np.asarray(data["full_pose"], dtype=np.float32)
        positions = z_up_to_y_up(positions)
        return _resample_integer(positions, generated_fps, evaluation_fps)

    if "smpl_poses" in data and "smpl_trans" in data:
        poses = np.asarray(data["smpl_poses"], dtype=np.float32)
        translations = np.asarray(data["smpl_trans"], dtype=np.float32)
        scaling = np.asarray(data.get("smpl_scaling", 1.0), dtype=np.float32).reshape(-1)[0]
    elif "q" in data and "pos" in data:
        poses = np.asarray(data["q"], dtype=np.float32).reshape(-1, 72)
        translations = np.asarray(data["pos"], dtype=np.float32)
        scaling = np.asarray(data.get("scale", 1.0), dtype=np.float32).reshape(-1)[0]
    else:
        raise KeyError("{} has no supported motion representation".format(path))
    if scaling == 0:
        raise ValueError("SMPL scaling is zero in {}".format(path))
    model = _smpl_model(smpl_model)
    with torch.no_grad():
        positions = model(
            global_orient=torch.from_numpy(poses[:, :3]),
            body_pose=torch.from_numpy(poses[:, 3:]),
            transl=torch.from_numpy(translations / scaling),
        ).joints[:, :24].cpu().numpy()
    return _resample_integer(positions, raw_fps, evaluation_fps)


def _audio_map(audio_dir):
    """Name -> beat source: a WAV to track, or a stored 35-D array to read.

    The wild corpus has no local WAV, and re-extracting one from the ingest tree
    would put BAS's music beats on a different extractor run than the planner's
    conditioning -- 0.36 correlation, measured, see ``eval/utils/musicbeat``.  A
    stem carrying both forms is refused rather than resolved by precedence.
    """
    mapping = {}
    for suffix in ("*.wav", "*.npy"):
        for path in sorted(Path(audio_dir).rglob(suffix)):
            if path.stem in mapping:
                raise ValueError("duplicate audio basename: {} and {}".format(
                    mapping[path.stem], path))
            mapping[path.stem] = path
    return mapping


def _aist_basename(name):
    fields = name.split("_")
    for index, field in enumerate(fields):
        if field.startswith("g") and len(field) == 3:
            return "_".join(fields[index:])
    return name


def _without_seed(name):
    """``<something>_s20260816`` -> ``<something>``, or the name unchanged.

    Generated samples are named for what they were generated from plus the seed
    that produced them, because a feature directory is one flat namespace and
    four seeds of one clip would otherwise collide.  On AIST the seed suffix was
    harmless: ``mBR2_s20260808`` still yields the music id ``mBR2``, so the
    fallback below found the track anyway.  A wild clip has no such id -- its
    audio is stored per clip under the clip's own name -- so without this the
    seed suffix makes every generated wild sample unmatchable, and the failure
    lands at feature-extraction time on a directory that already cost an hour of
    inference to produce.
    """
    marker = name.rfind("_s")
    if marker > 0 and name[marker + 2:].isdigit():
        return name[:marker]
    return name


def _aist_music_id(name):
    """The music id embedded in an AIST++ motion name, e.g. mBR0.

    ``gBR_sBM_cAll_d04_mBR0_ch01`` is danced to ``mBR0``.  Ten dancers and ten
    choreographies share each track, so a music-id-named audio directory holds
    62 files where the motion directory holds 411.
    """
    for field in name.split("_"):
        if len(field) == 4 and field[0] == "m" and field[1:3].isalpha():
            return field
    return None


def match_audio(motion_name, audio):
    # The music id is tried last, after the two exact-name forms: an audio
    # directory that names files per motion is the stricter pairing and must
    # win when it exists, because falling back to the track would silently
    # evaluate every one of a song's ten performances against the same wav.
    candidates = [motion_name, _aist_basename(motion_name),
                  _without_seed(motion_name), _aist_basename(_without_seed(motion_name))]
    music_id = _aist_music_id(motion_name)
    if music_id:
        candidates.append(music_id)
    for candidate in candidates:
        if candidate in audio:
            return audio[candidate]
    raise FileNotFoundError("no matching audio for motion {}".format(motion_name))


def _extract_task(task):
    (
        motion_path,
        audio_path,
        output_root,
        smpl_model,
        evaluation_fps,
        generated_fps,
        raw_fps,
        max_frames,
    ) = task
    motion_path = Path(motion_path)
    output_root = Path(output_root)
    positions = load_keypoints(
        motion_path, smpl_model, evaluation_fps, generated_fps, raw_fps
    )
    if max_frames is not None:
        positions = positions[:max_frames]
    if len(positions) < 3:
        raise ValueError("motion is too short: {}".format(motion_path))
    local_positions = positions - positions[:1, :1]
    name = motion_path.stem
    np.save(
        str(output_root / "kinetic_features" / (name + ".npy")),
        extract_kinetic_features(local_positions),
    )
    np.save(
        str(output_root / "manual_features" / (name + ".npy")),
        extract_manual_features(local_positions),
    )
    np.save(
        str(output_root / "dance_features" / (name + ".npy")),
        extract_dance_beat_features(local_positions, fps=evaluation_fps),
    )
    audio_path = Path(audio_path)
    beats = (
        beat_channel_from_features(np.load(str(audio_path)), fps=evaluation_fps)
        if audio_path.suffix == ".npy"
        else extract_music_beat_features(audio_path, fps=evaluation_fps)
    )
    np.save(str(output_root / "music_features" / (name + ".npy")), beats)
    return name, len(positions)


def extract_directory(
    motion_dir,
    audio_dir,
    output_root,
    smpl_model=None,
    workers=1,
    evaluation_fps=30,
    generated_fps=30,
    raw_fps=60,
    include_names=None,
    max_frames=None,
):
    motion_paths = sorted(Path(motion_dir).glob("*.pkl"))
    if include_names is not None:
        requested = set(include_names)
        motion_paths = [path for path in motion_paths if path.stem in requested]
        missing = requested - {path.stem for path in motion_paths}
        if missing:
            raise FileNotFoundError(
                "missing requested motion PKLs: {}".format(sorted(missing)[:5])
            )
    if not motion_paths:
        raise FileNotFoundError("no motion PKLs under {}".format(motion_dir))
    audio = _audio_map(audio_dir)
    output_root = Path(output_root)
    feature_directories = (
        "kinetic_features",
        "manual_features",
        "dance_features",
        "music_features",
    )
    expected_names = {path.stem + ".npy" for path in motion_paths}
    for directory in feature_directories:
        feature_dir = output_root / directory
        feature_dir.mkdir(parents=True, exist_ok=True)
        for stale_path in feature_dir.glob("*.npy"):
            if stale_path.name not in expected_names:
                stale_path.unlink()
    tasks = [
        (
            str(path),
            str(match_audio(path.stem, audio)),
            str(output_root),
            smpl_model,
            evaluation_fps,
            generated_fps,
            raw_fps,
            max_frames,
        )
        for path in motion_paths
    ]
    if workers == 1:
        results = [_extract_task(task) for task in tqdm(tasks, desc="Extracting features")]
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as executor:
            results = list(
                tqdm(
                    executor.map(_extract_task, tasks),
                    total=len(tasks),
                    desc="Extracting features",
                )
            )
    manifest = {
        "samples": len(results),
        "names": [name for name, _ in results],
        "evaluation_fps": evaluation_fps,
        "generated_fps": generated_fps,
        "raw_fps": raw_fps,
        "motion_dir": str(motion_dir),
        "audio_dir": str(audio_dir),
        "smpl_model": smpl_model,
        "max_frames": max_frames,
        "motion_signature": [
            [path.stem, path.stat().st_size, path.stat().st_mtime_ns]
            for path in motion_paths
        ],
    }
    with open(str(output_root / "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion-dir", "--pkl_dir", dest="motion_dir", required=True)
    parser.add_argument("--audio-dir", "--audio_dir", dest="audio_dir", required=True)
    parser.add_argument("--output", "--save_dir", dest="output", required=True)
    parser.add_argument(
        "--smpl-model",
        "--smpl_dir",
        dest="smpl_model",
        default=DEFAULT_SMPL_MODEL,
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--evaluation-fps", type=int, default=30)
    parser.add_argument("--generated-fps", type=int, default=30)
    parser.add_argument("--raw-fps", type=int, default=60)
    parser.add_argument("--max-frames", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    options = parse_args()
    print(
        json.dumps(
            extract_directory(
                options.motion_dir,
                options.audio_dir,
                options.output,
                options.smpl_model,
                options.workers,
                options.evaluation_fps,
                options.generated_fps,
                options.raw_fps,
                max_frames=options.max_frames,
            ),
            indent=2,
        )
    )
