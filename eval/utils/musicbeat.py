"""Frame-aligned music beat extraction.

Two sources, and which one is correct depends on the corpus rather than on
convenience.  ``extract_music_beat_features`` re-tracks beats from a WAV, which
is what AIST has.  The wild corpus has no local WAV: its music identity is the
stored 35-D array whose last channel is the beat one-hot from the extractor run
the planner trained on, and ``beat_channel_from_features`` reads that channel.

Re-extraction where an array exists is the silent version of a real error.
``tools/convert_aistpp_official.py`` measured it on the one corpus that has
both: onset frames land identically across two runs of the same extractor and
the beat channel correlates **0.36**.  BAS is a beat-alignment score, so a
0.36-correlated beat track is not a rounding difference -- it is the metric's
independent variable.
"""

import os
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/edge-numba-cache")

import librosa
import numpy as np


def _get_tempo(audio_name):
    fields = audio_name.split("_")
    music = next((field for field in fields if field.startswith("m") and len(field) == 4), None)
    if music is None:
        raise ValueError("cannot infer AIST++ tempo from {}".format(audio_name))
    if music[:3] in ("mBR", "mPO", "mLO", "mMH", "mLH", "mWA", "mKR", "mJS", "mJB"):
        return int(music[3]) * 10 + 80
    if music[:3] == "mHO":
        return int(music[3]) * 5 + 110
    raise ValueError("unknown AIST++ music id: {}".format(music))


def extract_music_beat_features(audio_path, fps=30):
    sample_rate = 44100
    hop_length = int(round(sample_rate / float(fps)))
    audio, _ = librosa.load(str(audio_path), sr=sample_rate)
    envelope = librosa.onset.onset_strength(
        y=audio, sr=sample_rate, hop_length=hop_length
    )
    try:
        start_bpm = _get_tempo(Path(audio_path).stem)
    except ValueError:
        start_bpm = float(librosa.beat.tempo(y=audio, sr=sample_rate)[0])
    _, beat_indices = librosa.beat.beat_track(
        onset_envelope=envelope,
        sr=sample_rate,
        hop_length=hop_length,
        start_bpm=start_bpm,
        tightness=100,
    )
    beats = np.zeros_like(envelope, dtype=bool)
    beats[beat_indices] = True
    return beats


# ``[envelope(1), mfcc(20), chroma(12), peak_onehot(1), beat_onehot(1)]`` --
# the layout ``data/audio_extraction/baseline_features.extract_audio`` builds,
# named once here so a reader never has to count to 34.
BASELINE_FEATURE_DIM = 35
BASELINE_BEAT_CHANNEL = 34
BASELINE_FEATURE_FPS = 30


def beat_channel_from_features(features, fps=30):
    """The beat one-hot already stored in a 35-D baseline music array.

    Refuses a frame rate other than the one the array was built at instead of
    resampling.  A beat track is a set of frame indices, so resampling it moves
    every beat by up to half a frame and BAS would silently measure that shift
    alongside the model.  The caller that wants another rate has to say what it
    means by it.
    """
    features = np.asarray(features)
    if features.ndim != 2 or features.shape[1] != BASELINE_FEATURE_DIM:
        raise ValueError("expected [frames, {}] baseline music features, got {}".format(
            BASELINE_FEATURE_DIM, features.shape))
    if int(fps) != BASELINE_FEATURE_FPS:
        raise ValueError(
            "stored beats are at {} fps and cannot be re-timed to {} fps without "
            "moving every beat".format(BASELINE_FEATURE_FPS, fps))
    return features[:, BASELINE_BEAT_CHANNEL] > 0.5
