"""M6c's arithmetic, and the two ways the number lies if read alone."""

import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from eval.metrics import calc_multimodality  # noqa: E402
from tools.eval_multimodality import (  # noqa: E402
    MultiModalityError,
    music_key,
    score,
)


def test_music_key_reads_the_aist_naming():
    assert music_key("gBR_sBM_cAll_d04_mBR0_ch01") == "mBR0"
    assert music_key("gHO_sFM_c01_d19_mHO3_ch05_slice2") == "mHO3"
    # A wild clip carries no *track* id, and inventing one would group unrelated
    # dances under a shared label -- so the group is the upload, which is the
    # audio identity the corpus does record: cuts of one video share one backing
    # track by construction.  Two uploads on the same track stay unlinked, which
    # makes this a floor rather than a measurement.  See the r-precision test
    # for the full reasoning; both tools read the one function.
    assert music_key("7147719479768714530__clip001") == "wild:7147719479768714530"


def test_music_key_reads_a_generated_sample_name():
    """A generated sample is named for the song it was generated *for*, so the
    id leads the name instead of sitting between underscores.  The regex this
    module used to carry required both delimiters and returned None for all 40
    samples of the M6 run -- the tool then refused to score, which was the lucky
    outcome; grouping them into one bucket would have published a ratio for a
    grouping that never happened."""
    assert music_key("mBR2_s20260808") == "mBR2"
    assert music_key("mWA5_s20260811") == "mWA5"


def test_music_key_agrees_with_the_r_precision_parser():
    """Both tools decide which dances share a song, and a name that parses for
    one and not the other puts the same set into two different groupings."""
    from tools.eval_r_precision import music_key as r_precision_music_key

    for name in ("gBR_sBM_cAll_d04_mBR0_ch01", "mBR2_s20260808",
                 "gHO_sFM_c01_d19_mHO3_ch05_slice2", "aistpp/gLO_sBM_cAll_d13_mLO2_ch05",
                 "7147719479768714530__clip001", "mBR2", "no_music_here"):
        assert music_key(name) == r_precision_music_key(name), name


def test_multimodality_is_the_within_group_spread():
    # Two musics, two samples each, distances 2 and 4 inside them.
    features = np.array([[0.0], [2.0], [10.0], [14.0]])
    groups = ["mAA0", "mAA0", "mBB1", "mBB1"]
    assert calc_multimodality(features, groups) == pytest.approx(3.0)


def test_singletons_are_dropped_not_counted_as_zero():
    """A music with one dance has no spread; scoring it as 0 would drag the
    average down in proportion to how many singletons a split happens to hold."""
    features = np.array([[0.0], [2.0], [100.0]])
    with_singleton = calc_multimodality(features, ["m0", "m0", "m1"])
    without = calc_multimodality(features[:2], ["m0", "m0"])
    assert with_singleton == pytest.approx(without)


def test_undefined_when_every_music_has_one_dance():
    """The wild corpus's shape: one audio track per clip.  The tool has to say
    so rather than return a number assembled from nothing."""
    features = np.arange(12, dtype=np.float64).reshape(4, 3)
    with pytest.raises(MultiModalityError, match="undefined"):
        score(features, ["a", "b", "c", "d"])


def test_unconditioned_samples_do_not_beat_the_null():
    """The failure mode the ratio exists to catch: a model that ignores the
    music.  Its within-music spread is its across-music spread, so the ratio
    sits on the null and `conditioned` must be False -- even though its raw
    MultiModality is the highest of any model here."""
    for seed in range(6):  # one unlucky draw is not evidence either way
        rng = np.random.default_rng(seed)
        features = rng.normal(size=(400, 8))
        groups = ["m%d" % (i % 20) for i in range(400)]
        report = score(features, groups, repeats=200, seed=seed)
        assert report["conditioned"] is False, "seed %d" % seed
        assert report["p_value"] > 0.01
        assert report["mm_over_div"] == pytest.approx(
            report["permutation_null_mm_over_div"]["mean"], abs=0.05)


def test_conditioned_samples_fall_below_the_null():
    """Tight clusters per music: same distance, same standardisation, but the
    grouping now carries information and the ratio drops out of the null."""
    rng = np.random.default_rng(1)
    centres = rng.normal(size=(24, 8)) * 10.0
    features = np.concatenate([centres[i % 24] + rng.normal(size=(1, 8)) * 0.1
                               for i in range(120)])
    groups = ["m%d" % (i % 24) for i in range(120)]
    report = score(features, groups, repeats=100)
    assert report["conditioned"] is True
    assert report["p_value"] == pytest.approx(1.0 / 101, abs=1e-6)
    assert report["mm_over_div"] < report["permutation_null_mm_over_div"]["p05"]


def test_group_sizes_are_preserved_by_the_null():
    """The null shuffles labels, so it must report the same group census as the
    observed run -- a null that also resampled sizes would answer a different
    question, because small groups give noisier within-group means."""
    rng = np.random.default_rng(2)
    features = rng.normal(size=(30, 4))
    groups = ["m0"] * 12 + ["m1"] * 15 + ["m2"] * 3
    report = score(features, groups, repeats=50)
    assert report["samples_per_music"] == {"min": 3, "median": 12, "max": 15, "mean": 10.0}
    assert report["musics_used"] == 3


def test_genre_key_survives_the_corpus_prefix():
    """Release names carry a corpus prefix, so an anchored pattern matches
    nothing -- and the failure is a clean "no music id" error that reads like
    missing data rather than a wrong regex."""
    from tools.eval_multimodality import genre_key

    assert genre_key("aistpp/gHO_sBM_cAll_d20_mHO5_ch02_slice2") == "gHO"
    assert genre_key("gJS_sBM_cAll_d01_mJS3_ch01") == "gJS"
    assert genre_key("7147719479768714530__clip001") is None


def test_same_recording_slices_inflate_the_conditioning():
    """The ground-truth trap.  Two musics, each with two recordings; slices of
    one recording are near-identical, so counting them makes the music look far
    more constraining than it is.  Excluding same-recording pairs is what makes
    the number mean "several dances for this song"."""
    from eval.metrics import calc_multimodality

    # music m0: recording A at 0, recording B at 10.  Slices sit on top of each
    # other; the two recordings are 10 apart.
    features = np.array([[0.0], [0.1], [10.0], [10.1],
                         [0.0], [0.1], [10.0], [10.1]])
    groups = ["m0"] * 4 + ["m1"] * 4
    seqs = ["A", "A", "B", "B", "C", "C", "D", "D"]
    all_pairs = calc_multimodality(features, groups)
    cross = calc_multimodality(features, groups, seqs)
    assert all_pairs < cross
    # every cross-recording pair is ~10 apart; the all-pairs figure is dragged
    # down by the four within-recording pairs of ~0.1
    assert cross == pytest.approx(10.0, abs=0.15)
    assert all_pairs == pytest.approx(6.7, abs=0.2)


def test_cross_recording_needs_two_recordings():
    from eval.metrics import calc_multimodality

    features = np.array([[0.0], [1.0], [2.0]])
    with pytest.raises(ValueError, match="different recordings"):
        calc_multimodality(features, ["m0", "m0", "m0"], ["A", "A", "A"])


def _pairs_file(tmp_path, pairs, name="pairs.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(
        json.dumps({"left": left, "right": right, "score": 0.9}) for left, right in pairs
    ) + "\n", encoding="utf-8")
    return path


def test_a_chained_component_is_not_a_track_group(tmp_path):
    """Single linkage is what made the fingerprint's own grouping unusable.

    ``a--b--c`` with no ``a--c`` edge is one connected component and three clips
    that were never all shown to share a track.  On this corpus that chaining
    swallowed 2,335 clips into one "track".  Only complete subgraphs survive.
    """
    from tools.eval_multimodality import verified_track_groups

    split_of = {"wild_v4:1:clip000": "test", "wild_v4:2:clip000": "test",
                "wild_v4:3:clip000": "test",
                "wild_v4:8:clip000": "test", "wild_v4:9:clip000": "test"}
    pairs = _pairs_file(tmp_path, [
        ("wild_v4:1:clip000", "wild_v4:2:clip000"),   # chain a--b
        ("wild_v4:2:clip000", "wild_v4:3:clip000"),   # chain b--c, no a--c
        ("wild_v4:8:clip000", "wild_v4:9:clip000"),   # a complete pair
    ])
    grouping = verified_track_groups(pairs, split_of, "test")
    assert grouping["groups"] == 1
    assert set(grouping["group_of"]) == {"wild_v4:8:clip000", "wild_v4:9:clip000"}


def test_a_complete_triangle_is_kept(tmp_path):
    from tools.eval_multimodality import verified_track_groups

    clips = ["wild_v4:{}:clip000".format(index) for index in (1, 2, 3)]
    split_of = {clip: "test" for clip in clips}
    pairs = _pairs_file(tmp_path, [(clips[0], clips[1]), (clips[1], clips[2]),
                                   (clips[0], clips[2])])
    grouping = verified_track_groups(pairs, split_of, "test")
    assert grouping["groups"] == 1
    assert grouping["clips"] == 3
    assert len(set(grouping["group_of"].values())) == 1


def test_a_group_inside_one_upload_is_rejected(tmp_path):
    """Cuts of one upload share a track by construction and are one performance.

    Keeping them would put the tool back on the measurement it exists to avoid:
    how smooth a single take is, rather than how many dances a song admits.
    """
    from tools.eval_multimodality import verified_track_groups

    clips = ["wild_v4:7:clip{:03d}".format(cut) for cut in range(3)]
    split_of = {clip: "test" for clip in clips}
    split_of.update({"wild_v4:8:clip000": "test", "wild_v4:9:clip000": "test"})
    pairs = _pairs_file(tmp_path, [(clips[0], clips[1]), (clips[1], clips[2]),
                                   (clips[0], clips[2]),
                                   ("wild_v4:8:clip000", "wild_v4:9:clip000")])
    grouping = verified_track_groups(pairs, split_of, "test")
    assert grouping["groups"] == 1
    assert "wild_v4:7:clip000" not in grouping["group_of"]


def test_pairs_outside_the_split_are_not_grouped(tmp_path):
    from tools.eval_multimodality import verified_track_groups

    split_of = {"wild_v4:1:clip000": "test", "wild_v4:2:clip000": "train",
                "wild_v4:8:clip000": "test", "wild_v4:9:clip000": "test"}
    pairs = _pairs_file(tmp_path, [("wild_v4:1:clip000", "wild_v4:2:clip000"),
                                   ("wild_v4:8:clip000", "wild_v4:9:clip000")])
    grouping = verified_track_groups(pairs, split_of, "test")
    assert set(grouping["group_of"]) == {"wild_v4:8:clip000", "wild_v4:9:clip000"}


def test_a_split_with_no_usable_group_is_refused(tmp_path):
    """Refusing beats returning a number built from a grouping that is not one."""
    from tools.eval_multimodality import MultiModalityError, verified_track_groups

    split_of = {"wild_v4:1:clip000": "test", "wild_v4:2:clip000": "test",
                "wild_v4:3:clip000": "test"}
    pairs = _pairs_file(tmp_path, [("wild_v4:1:clip000", "wild_v4:2:clip000"),
                                   ("wild_v4:2:clip000", "wild_v4:3:clip000")])
    with pytest.raises(MultiModalityError, match="no complete multi-upload group"):
        verified_track_groups(pairs, split_of, "test")
