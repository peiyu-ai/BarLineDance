import os
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile

from eval.eval_bas import alignment_score, calculate_bas
from eval.evaluate import evaluate, resolve_features
from eval import extract_aist_features
from eval.extract_aist_features import load_keypoints
from eval.metrics import normalize_separately


class EvaluationTests(unittest.TestCase):
    def make_feature_root(self, root, offset=0.0):
        for directory in (
            "kinetic_features",
            "manual_features",
            "music_features",
            "dance_features",
        ):
            os.makedirs(os.path.join(root, directory))
        for index in range(4):
            name = "sample{}".format(index)
            kinetic = np.array([index, index ** 2, index + 1], dtype=np.float32) + offset
            manual = np.array([index % 2, index / 2], dtype=np.float32) + offset
            music = np.zeros(30, dtype=bool)
            dance = np.zeros(30, dtype=bool)
            music[[5, 15, 25]] = True
            dance[[5, 15, 25]] = True
            np.save(os.path.join(root, "kinetic_features", name + ".npy"), kinetic)
            np.save(os.path.join(root, "manual_features", name + ".npy"), manual)
            np.save(os.path.join(root, "music_features", name + ".npy"), music)
            np.save(os.path.join(root, "dance_features", name + ".npy"), dance)

    def test_bas_edge_cases(self):
        self.assertEqual(alignment_score(np.zeros(5), np.zeros(5)), 0.0)
        beats = np.array([False, True, False, True])
        self.assertEqual(alignment_score(beats, beats), 0.5)

    def test_unified_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            prediction = os.path.join(directory, "prediction")
            ground_truth = os.path.join(directory, "ground_truth")
            self.make_feature_root(prediction)
            self.make_feature_root(ground_truth)

            metrics = evaluate(prediction, ground_truth)
            self.assertAlmostEqual(metrics["fid_k"], 0.0, places=5)
            self.assertAlmostEqual(metrics["fid_m"], 0.0, places=5)
            self.assertAlmostEqual(metrics["BAS_pred"], 0.5)
            self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
            self.assertAlmostEqual(calculate_bas(prediction), 0.5)

    def test_starter_normalizes_prediction_and_gt_separately(self):
        features = np.array(
            [[0.0, 1.0], [1.0, 3.0], [3.0, 7.0], [6.0, 13.0]],
            dtype=np.float64,
        )
        transformed = features * np.array([4.0, 2.0]) + np.array([50.0, -20.0])
        np.testing.assert_allclose(
            normalize_separately(features),
            normalize_separately(transformed),
            atol=1e-12,
        )

        with tempfile.TemporaryDirectory() as directory:
            prediction = os.path.join(directory, "prediction")
            ground_truth = os.path.join(directory, "ground_truth")
            self.make_feature_root(prediction, offset=100.0)
            self.make_feature_root(ground_truth)

            metrics = evaluate(prediction, ground_truth)
            self.assertAlmostEqual(metrics["fid_k"], 0.0, places=5)
            self.assertAlmostEqual(metrics["fid_m"], 0.0, places=5)

    def test_generated_motion_feature_extraction_and_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            motions = os.path.join(directory, "motions")
            audio = os.path.join(directory, "audio")
            cache = os.path.join(directory, "cache")
            os.makedirs(motions)
            os.makedirs(audio)

            rng = np.random.RandomState(7)
            full_pose = rng.normal(size=(40, 24, 3)).astype(np.float32)
            motion_name = "sample_0_gBR_sBM_cAll_d04_mBR0_ch01"
            motion_path = os.path.join(motions, motion_name + ".pkl")
            with open(motion_path, "wb") as handle:
                pickle.dump({"full_pose": full_pose}, handle)
            with open(os.path.join(motions, "unrelated.pkl"), "wb") as handle:
                pickle.dump({"full_pose": full_pose}, handle)
            samples = np.arange(44100, dtype=np.float32)
            waveform = 0.1 * np.sin(2 * np.pi * 220 * samples / 44100)
            wavfile.write(
                os.path.join(audio, "gBR_sBM_cAll_d04_mBR0_ch01.wav"),
                44100,
                waveform,
            )

            root = resolve_features(
                "prediction",
                None,
                motions,
                audio,
                cache,
                workers=1,
                include_names=[motion_name],
            )
            for feature_dir in (
                "kinetic_features",
                "manual_features",
                "dance_features",
                "music_features",
            ):
                self.assertTrue((root / feature_dir / (motion_name + ".npy")).is_file())
            self.assertEqual(
                load_keypoints(motion_path)[0, 0].tolist(),
                [full_pose[0, 0, 0], full_pose[0, 0, 2], -full_pose[0, 0, 1]],
            )
            self.assertEqual(
                resolve_features(
                    "prediction",
                    None,
                    motions,
                    audio,
                    cache,
                    workers=1,
                    include_names=[motion_name],
                ),
                root,
            )

    def test_smpl_dependency_is_loaded_only_for_smpl_motion(self):
        with tempfile.TemporaryDirectory() as directory:
            motion_path = os.path.join(directory, "smpl_only.pkl")
            with open(motion_path, "wb") as handle:
                pickle.dump(
                    {
                        "smpl_poses": np.zeros((1, 72), dtype=np.float32),
                        "smpl_trans": np.zeros((1, 3), dtype=np.float32),
                    },
                    handle,
                )
            missing_smplx = ModuleNotFoundError(
                "No module named 'smplx'", name="smplx"
            )
            with patch.object(
                extract_aist_features,
                "_import_smpl_class",
                side_effect=missing_smplx,
            ):
                with self.assertRaisesRegex(
                    ModuleNotFoundError,
                    "SMPL-dependent feature extraction requires the optional 'smplx'",
                ):
                    load_keypoints(
                        motion_path,
                        smpl_model=os.path.join(directory, "SMPL_MALE.pkl"),
                    )


def test_stored_music_features_supply_the_beats_without_a_wav(tmp_path):
    """A corpus with no WAV can still be scored, on its own beat track.

    The wild corpus keeps no local audio: its music identity is the 35-D array
    the planner was conditioned on.  Re-tracking beats from a re-decoded WAV
    would give BAS a beat track correlating 0.36 with that one -- measured in
    ``tools/convert_aistpp_official.py`` -- so the stored channel is read, and
    the beats it yields are exactly the frames it marks.
    """
    import pickle as pkl

    import numpy as np
    import pytest

    from eval.extract_aist_features import _audio_map, extract_directory

    motions = tmp_path / "motions"
    audio = tmp_path / "audio"
    out = tmp_path / "features"
    motions.mkdir()
    audio.mkdir()

    name = "wild_v4:7195533766570282272:clip000"
    rng = np.random.RandomState(11)
    with open(str(motions / (name + ".pkl")), "wb") as handle:
        pkl.dump({"full_pose": rng.normal(size=(40, 24, 3)).astype(np.float32)}, handle)
    features = np.zeros((40, 35), dtype=np.float32)
    features[[3, 11, 19, 27], 34] = 1.0
    np.save(str(audio / (name + ".npy")), features)

    assert _audio_map(audio)[name].suffix == ".npy"
    extract_directory(str(motions), str(audio), str(out), workers=1)
    beats = np.load(str(out / "music_features" / (name + ".npy")))
    # Read, not re-derived: the marked frames come back and nothing else does.
    assert np.flatnonzero(beats).tolist() == [3, 11, 19, 27]


def test_a_stored_beat_track_is_not_silently_re_timed(tmp_path):
    """Resampling a beat track moves every beat, so a different fps is refused."""
    import numpy as np
    import pytest

    from eval.utils.musicbeat import beat_channel_from_features

    features = np.zeros((10, 35), dtype=np.float32)
    with pytest.raises(ValueError, match="cannot be re-timed"):
        beat_channel_from_features(features, fps=60)
    with pytest.raises(ValueError, match="baseline music features"):
        beat_channel_from_features(np.zeros((10, 39), dtype=np.float32))


def test_a_generated_sample_matches_the_audio_of_the_clip_it_came_from():
    """The seed suffix must not cost a generated wild sample its audio.

    Four seeds of one clip share one flat feature namespace, so each carries its
    seed in the name.  On AIST that was free -- ``mBR2_s20260808`` still yields
    the music id ``mBR2`` and the id fallback found the track.  A wild clip's
    audio is stored under the clip's own name and there is no id, so without
    stripping the seed the sample is unmatchable, and it fails at feature
    extraction on a directory that already cost the inference to build.
    """
    audio = {"wild_v4:7195533766570282272:clip000": "clip.npy",
             "mBR2": "mBR2.wav"}
    assert extract_aist_features.match_audio(
        "wild_v4:7195533766570282272:clip000_s20260816", audio) == "clip.npy"
    # AIST keeps working through the id fallback it always used.
    assert extract_aist_features.match_audio("mBR2_s20260808", audio) == "mBR2.wav"


def test_a_per_clip_audio_name_still_beats_the_music_id():
    """Stripping the seed must not promote the track over the exact pairing.

    A directory naming audio per motion is the stricter pairing; falling back to
    the track would score every one of a song's ten performances against one wav.
    """
    audio = {"gBR_sBM_cAll_d04_mBR0_ch01": "exact.wav", "mBR0": "track.wav"}
    assert extract_aist_features.match_audio(
        "gBR_sBM_cAll_d04_mBR0_ch01", audio) == "exact.wav"


def test_an_unmatchable_name_is_refused_rather_than_guessed():
    import pytest

    with pytest.raises(FileNotFoundError, match="no matching audio"):
        extract_aist_features.match_audio("nothing_like_it", {"other": "x.npy"})
