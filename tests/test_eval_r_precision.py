"""Tests for M6b R-precision.

The metric is a ranking rule over two distance matrices, so the tests are built
from geometry with a known answer rather than from recorded output: a corpus
where music and motion agree must score 100, one where they are independent must
sit at chance, and the same-sequence exclusion must actually change who is
eligible.
"""

import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.eval_r_precision import (  # noqa: E402
    RPrecisionError,
    build_parser,
    evaluate,
    load_verified_music_pairs,
    music_key,
    pairwise_distances,
    r_precision,
    r_precision_at_pool,
    sequence_key,
    split_into_clips,
    standardise,
)


def test_standardise_leaves_constant_dimensions_alone():
    features = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    scaled = standardise(features)
    assert np.allclose(scaled[:, 1], 0.0)
    assert np.isfinite(scaled).all()
    assert pytest.approx(scaled[:, 0].std(), abs=1e-9) == 1.0


def test_pairwise_distances_match_the_direct_computation():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(11, 4))
    expected = np.sqrt(((features[:, None, :] - features[None, :, :]) ** 2).sum(axis=2))
    # The expansion form trades a little precision for not materialising an
    # N x N x D tensor: cancellation in |a|^2 + |b|^2 - 2ab costs a few ulp, so
    # the tolerance is set where that noise lives and not tighter.
    assert np.allclose(pairwise_distances(features), expected, atol=1e-7)
    assert np.allclose(np.diagonal(pairwise_distances(features)), 0.0, atol=1e-7)


def test_perfectly_aligned_music_and_motion_score_100():
    rng = np.random.default_rng(1)
    shared = rng.normal(size=(40, 6))
    # Motion is a rotation and rescaling of the music layout, so the nearest
    # neighbour is the same clip in both spaces however distances are measured.
    rotation = np.linalg.qr(rng.normal(size=(6, 6)))[0]
    result = r_precision(shared, shared @ rotation * 3.0)
    assert result["R"] == 100.0
    assert result["scored_clips"] == 40
    assert result["chance_R"] < 10.0


def test_independent_spaces_land_near_chance():
    rng = np.random.default_rng(2)
    music = rng.normal(size=(300, 8))
    motion = rng.normal(size=(300, 8))
    result = r_precision(music, motion)
    # Chance is 3/299 ~ 1.0%; an unrelated pairing must not look structured.
    assert result["R"] < 5.0
    assert result["chance_R"] == pytest.approx(100.0 * 3 / 299, abs=0.05)


def test_adversarial_music_ordering_scores_zero():
    # Clips sit evenly on a line in music space, so clip i's nearest music
    # neighbour is i +/- 1.  Motion space applies the permutation 13i mod 31,
    # which sends those neighbours 13 places away while the motion-nearest
    # clips are i +/- 12 (13 * 12 = 1 mod 31).  Music adjacency therefore never
    # predicts motion adjacency and R must be exactly 0.
    count = 31
    index = np.arange(count)
    music = index.reshape(-1, 1).astype(float)
    motion = ((13 * index) % count).reshape(-1, 1).astype(float)
    result = r_precision(music, motion)
    assert result["R"] == 0.0


def test_ties_are_resolved_against_the_metric():
    # Every motion clip sits at the same distance from every other, so no
    # candidate is reliably inside a top-3 cut and R must be 0 rather than 100.
    music = np.eye(10)
    motion = np.zeros((10, 3))
    result = r_precision(music, motion)
    assert result["R"] == 0.0


def test_same_sequence_exclusion_changes_the_candidate_pool():
    rng = np.random.default_rng(3)
    # Two songs of ten clips each; within a song the clips are near-identical in
    # both spaces, so pooled R is perfect and excluding siblings destroys it.
    blocks, groups = [], []
    for song in range(2):
        centre = rng.normal(size=(1, 5)) * 50.0
        blocks.append(centre + rng.normal(size=(10, 5)) * 0.01)
        groups.extend(["song{}".format(song)] * 10)
    features = np.concatenate(blocks)
    pooled = r_precision(features, features, groups=groups)
    excluded = r_precision(features, features, groups=groups, exclude_same_group=True)
    assert pooled["R"] == 100.0
    assert pooled["mean_candidate_pool"] > excluded["mean_candidate_pool"]
    assert excluded["excluded_same_sequence"] is True


def test_exclusion_requires_groups():
    features = np.eye(8)
    with pytest.raises(RPrecisionError, match="needs a group"):
        r_precision(features, features, exclude_same_group=True)


def test_mismatched_pair_counts_are_refused():
    with pytest.raises(RPrecisionError, match="defined on pairs"):
        r_precision(np.eye(9), np.eye(8)[:, :9])


def test_too_few_clips_are_refused_rather_than_scored():
    with pytest.raises(RPrecisionError, match="cannot support"):
        r_precision(np.eye(4), np.eye(4))


def test_clips_whose_only_companions_are_siblings_are_dropped_not_missed():
    # One song of four clips plus a well-separated pair: with siblings excluded
    # the four have too few candidates left, so they are dropped from the
    # denominator instead of being counted as failures.
    rng = np.random.default_rng(4)
    features = np.concatenate([rng.normal(size=(4, 3)) * 0.01,
                               rng.normal(size=(6, 3)) * 0.01 + 100.0])
    groups = ["a"] * 4 + ["b", "c", "d", "e", "f", "g"]
    result = r_precision(features, features, groups=groups, exclude_same_group=True)
    assert result["dropped_for_small_pool"] == 0 or result["scored_clips"] < len(features)
    assert result["scored_clips"] + result["dropped_for_small_pool"] == len(features)


def test_split_into_clips_refuses_a_remainder():
    assert split_into_clips(150, 50) == [(0, 50), (50, 100), (100, 150)]
    # 150 does not divide by 40; the trailing 30 frames are dropped, not kept as
    # a shorter and noisier member of the same pool.
    assert split_into_clips(150, 40) == [(0, 40), (40, 80), (80, 120)]
    with pytest.raises(RPrecisionError, match="does not fit"):
        split_into_clips(150, 200)
    with pytest.raises(RPrecisionError, match="at least two frames"):
        split_into_clips(150, 1)


def test_sequence_key_strips_only_a_real_slice_suffix():
    assert sequence_key("tiktok:123:clip000_slice7") == "tiktok:123:clip000"
    assert sequence_key("tiktok:123:clip000") == "tiktok:123:clip000"
    # A name that merely ends in the word must survive intact, or two unrelated
    # recordings would be merged into one exclusion group.
    assert sequence_key("gBR_sBM_slice") == "gBR_sBM_slice"


def test_r_falls_as_the_candidate_pool_grows():
    # The property the fixed-pool option exists for: with the same clips, a
    # larger pool makes "among the three nearest" a harder target, so an R
    # quoted without its pool size is not comparable to anything.
    rng = np.random.default_rng(6)
    music = rng.normal(size=(400, 6))
    motion = music + rng.normal(size=(400, 6)) * 1.5
    small = r_precision_at_pool(music, motion, pool_size=40, repeats=12)
    large = r_precision_at_pool(music, motion, pool_size=400, repeats=3)
    assert small["R_mean"] > large["R_mean"]
    assert small["chance_R"] > large["chance_R"]


def test_fixed_pool_reports_its_spread_and_is_reproducible():
    rng = np.random.default_rng(7)
    music = rng.normal(size=(120, 5))
    motion = music + rng.normal(size=(120, 5))
    first = r_precision_at_pool(music, motion, pool_size=30, repeats=8, seed=11)
    second = r_precision_at_pool(music, motion, pool_size=30, repeats=8, seed=11)
    assert first == second
    assert first["draws_scored"] == 8
    assert first["R_min"] <= first["R_mean"] <= first["R_max"]
    assert first["R_std"] >= 0.0


def test_fixed_pool_refuses_a_pool_it_cannot_draw():
    features = np.eye(20)
    with pytest.raises(RPrecisionError, match="cannot draw"):
        r_precision_at_pool(features, features, pool_size=50)
    with pytest.raises(RPrecisionError, match="cannot support"):
        r_precision_at_pool(features, features, pool_size=4)


def test_lift_over_chance_is_reported():
    rng = np.random.default_rng(8)
    shared = rng.normal(size=(60, 5))
    aligned = r_precision(shared, shared * 2.0)
    assert aligned["lift_over_chance"] > 1.0
    assert aligned["R"] == 100.0


def test_parser_defaults_record_the_calibrated_choices():
    args = build_parser().parse_args(["--release", "somewhere"])
    assert args.split == "test"
    assert args.feature == "kinetic"
    assert args.music_aggregate == "mean_std"
    assert args.clip_frames == 150
    assert args.calibrate is None


def test_report_is_json_serialisable():
    rng = np.random.default_rng(5)
    result = r_precision(rng.normal(size=(50, 4)), rng.normal(size=(50, 4)))
    assert json.loads(json.dumps(result))["top_k"] == 3


def test_mean_rank_separates_conditions_that_top_three_cannot():
    # Two conditions whose top-3 hit counts are both ~chance, but whose overall
    # ordering differs: the rank statistic must see the difference that R misses.
    rng = np.random.default_rng(21)
    music = rng.normal(size=(400, 6))
    weak = r_precision(music, music + rng.normal(size=(400, 6)) * 12.0)
    none = r_precision(music, rng.normal(size=(400, 6)) * 12.0)
    assert weak["mean_normalised_rank"] < none["mean_normalised_rank"]
    assert weak["rank_chance"] == 0.5


def test_mean_rank_is_near_chance_for_unrelated_spaces():
    rng = np.random.default_rng(22)
    result = r_precision(rng.normal(size=(500, 6)), rng.normal(size=(500, 6)))
    assert abs(result["mean_normalised_rank"] - 0.5) < 0.06


def test_mean_rank_is_zero_when_the_spaces_agree():
    rng = np.random.default_rng(23)
    shared = rng.normal(size=(80, 5))
    result = r_precision(shared, shared * 3.0)
    assert result["mean_normalised_rank"] == 0.0
    assert result["R"] == 100.0


def _synthetic_release(tmp_path, names):
    """A minimal release directory: the scorer reads four files, so write four.

    Motion is random in the normalised range rather than recorded, because these
    tests are about which clips are *eligible* to be retrieved, not about what
    the features say.
    """
    import torch

    rng = np.random.default_rng(31)
    count = len(names)
    split = tmp_path / "test"
    split.mkdir(parents=True)
    np.save(split / "motion.npy",
            rng.uniform(-1.0, 1.0, size=(count, 150, 151)).astype(np.float32))
    np.save(split / "music.npy",
            rng.normal(size=(count, 150, 35)).astype(np.float32))
    (split / "names.json").write_text(json.dumps(names), encoding="utf-8")
    torch.save({"data_min": torch.zeros(151), "data_max": torch.ones(151)},
               tmp_path / "normalizer.pt")
    return tmp_path


def test_music_key_reads_the_aist_id_and_admits_when_it_cannot():
    assert music_key("aistpp/gBR_sBM_cAll_d04_mBR1_ch01_slice0") == "mBR1"
    assert music_key("mBR2_s20260808_slice12") == "mBR2"
    assert music_key("mBRX_ch01") is None
    # A name that says nothing about its audio still gets no bucket: one shared
    # fallback would exclude unrelated songs from each other's pools and quietly
    # shrink every candidate list.
    assert music_key("some_unrelated_name") is None


def test_music_key_groups_a_wild_clip_by_its_upload():
    """Changed 2026-08-16, and the reason travels with it.

    This used to assert None for a wild name, on the reasoning that the clip
    carries no music id.  True, and beside the point for what the key is used
    for: clips of one upload are cuts of one video, so they carry one backing
    track *by construction*, and leaving them in each other's candidate pools
    left the exact artifact ``--exclude-same-music`` exists to remove.

    The key is sound in the direction it is used and unsound in the other: two
    uploads dancing to the same track still land in different groups, because
    this corpus has no track id.  So a wild exclusion built on it is a floor,
    and the R it produces is an upper bound.  Both forms of the name resolve to
    the same group, because the release window and the source video spell the
    same clip differently.
    """
    assert music_key("wild_v4:7195533766570282272:clip000_slice3") == "wild_v4:7195533766570282272"
    assert music_key("wild_v4:7195533766570282272:clip004") == "wild_v4:7195533766570282272"
    assert music_key("7195533766570282272__clip001") == "wild:7195533766570282272"
    # A generated sample is named for the clip it was generated from plus its
    # seed; every seed of one clip must land in the group its ground truth is in.
    assert music_key("wild_v4:7195533766570282272:clip000_s20260816") == "wild_v4:7195533766570282272"


def test_same_music_exclusion_removes_hits_the_sequence_exclusion_leaves():
    # Four takes of each song, each take a near copy of its siblings: excluding
    # same-sequence pairs leaves the siblings eligible, and because they share
    # their music exactly they are the nearest music clip by construction.
    rng = np.random.default_rng(32)
    songs, music_rows, motion_rows, seq_groups, music_groups = 8, [], [], [], []
    for song in range(songs):
        base_music = rng.normal(size=6)
        base_motion = rng.normal(size=6)
        for take in range(4):
            music_rows.append(base_music + rng.normal(size=6) * 1e-6)
            motion_rows.append(base_motion + rng.normal(size=6) * 1e-6)
            seq_groups.append("m{:02d}_s{}".format(song, take))
            music_groups.append("m{:02d}".format(song))
    music = np.asarray(music_rows)
    motion = np.asarray(motion_rows)
    by_sequence = r_precision(music, motion, groups=seq_groups, exclude_same_group=True)
    by_music = r_precision(music, motion, groups=music_groups, exclude_same_group=True)
    assert by_sequence["R"] == 100.0
    assert by_music["R"] < by_sequence["R"]
    assert by_music["mean_candidate_pool"] < by_sequence["mean_candidate_pool"]


def test_evaluate_reports_the_music_control_and_the_unit_it_achieved(tmp_path):
    names = ["mBR{}_s2026080{}_slice{}".format(song, seed, index)
             for song in range(3) for seed in range(2) for index in range(2)]
    release = _synthetic_release(tmp_path, names)
    report = evaluate(release, "test", clip_frames=150, feature="kinetic",
                      music_aggregate="mean_std", exclude_same_music=True)
    assert report["music_group_unit"]["distinct_music_groups"] == 3
    assert report["music_group_unit"]["distinct_sequences"] == 6
    assert report["excluding_same_music"]["excluded_same_sequence"] is True
    # The stricter control can only shrink the pool it draws from.
    assert (report["excluding_same_music"]["mean_candidate_pool"]
            < report["excluding_same_sequence"]["mean_candidate_pool"])


def test_evaluate_refuses_the_music_control_on_names_without_an_id(tmp_path):
    # Wild names used to be the example here; they now carry an upload-level
    # audio group, so the case this guard covers needs a name that genuinely
    # says nothing -- the guard itself is unchanged and still fires.
    names = ["recording{}_slice{}".format(clip, index)
             for clip in range(4) for index in range(3)]
    release = _synthetic_release(tmp_path, names)
    with pytest.raises(RPrecisionError, match="no audio group"):
        evaluate(release, "test", clip_frames=150, feature="kinetic",
                 music_aggregate="mean_std", exclude_same_music=True)


def test_evaluate_runs_the_music_control_on_wild_upload_groups(tmp_path):
    """The wild corpus can now use the control that used to refuse it.

    Three cuts of each of four uploads: excluding the same upload must shrink
    the pool below what excluding the same sequence does, because a cut's
    siblings share its backing track and were previously sitting in its pool.
    """
    names = ["wild_v4:{}:clip{:03d}_slice0".format(upload, cut)
             for upload in range(4) for cut in range(3)]
    report = evaluate(_synthetic_release(tmp_path, names), "test", clip_frames=150,
                      feature="kinetic", music_aggregate="mean_std",
                      exclude_same_music=True)
    assert report["music_group_unit"]["distinct_music_groups"] == 4
    # And the report says out loud that this unit under-excludes, so the R
    # beside it is read as an upper bound.
    assert report["music_group_unit"]["is_a_floor"] is True
    assert (report["excluding_same_music"]["mean_candidate_pool"]
            < report["excluding_same_sequence"]["mean_candidate_pool"])


def test_music_control_is_off_unless_asked_for(tmp_path):
    assert build_parser().parse_args(["--release", "somewhere"]).exclude_same_music is False
    names = ["mBR{}_s2026080{}_slice{}".format(song, seed, index)
             for song in range(3) for seed in range(2) for index in range(2)]
    report = evaluate(_synthetic_release(tmp_path, names), "test", clip_frames=150,
                      feature="kinetic", music_aggregate="mean_std")
    assert "excluding_same_music" not in report
    assert "music_group_unit" not in report


def test_the_pool_sweep_gets_a_controlled_twin_when_the_music_control_is_on(tmp_path):
    names = ["mBR{}_s2026080{}_slice{}".format(song, seed, index)
             for song in range(4) for seed in range(2) for index in range(3)]
    release = _synthetic_release(tmp_path, names)
    report = evaluate(release, "test", clip_frames=150, feature="kinetic",
                      music_aggregate="mean_std", exclude_same_music=True,
                      pool_sizes=[10], pool_repeats=3)
    assert report["fixed_pool"][0]["excluded_same_sequence"] is False
    controlled = report["fixed_pool_excluding_same_music"][0]
    assert controlled["excluded_same_sequence"] is True
    assert controlled["pool_size"] == 10
    # Without the control there is no twin to mistake for it.
    plain = evaluate(release, "test", clip_frames=150, feature="kinetic",
                     music_aggregate="mean_std", pool_sizes=[10], pool_repeats=3)
    assert "fixed_pool_excluding_same_music" not in plain


def _pair_file(tmp_path, pairs, name="pairs.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(
        json.dumps({"left": left, "right": right, "score": 0.9, "lag_frames": 7})
        for left, right in pairs) + "\n", encoding="utf-8")
    return path


def test_verified_pairs_exclude_in_both_directions_and_are_counted():
    """The pair list is unordered, so naming (a, b) must also unseat (b, a).

    A one-directional exclusion is the failure that looks like it worked: half
    the artifact survives, R drops a little, and the number reads as controlled.
    """
    rng = np.random.default_rng(41)
    music = rng.normal(size=(8, 5))
    motion = rng.normal(size=(8, 5))
    keys = ["clip{}".format(index) for index in range(8)]
    plain = r_precision(music, motion)
    excluded = r_precision(music, motion, exclude_pairs=[frozenset(("clip0", "clip3"))],
                           pair_keys=keys)
    assert plain["excluded_verified_pairs"] == 0
    assert excluded["excluded_verified_pairs"] == 1
    # Both rows lose exactly one candidate, which is what symmetry means here.
    # The reported pool is rounded to one decimal, so the expectation is too --
    # comparing against the unrounded value would fail on the report's own format
    # rather than on the exclusion.
    assert excluded["mean_candidate_pool"] == round(
        plain["mean_candidate_pool"] - 2.0 / 8.0, 1)


def test_verified_pairs_remove_what_the_upload_key_cannot_see(tmp_path):
    """Two uploads dancing to one track: the key groups them apart, the pair joins them.

    This is the whole reason the pair list exists.  The upload key is sound in
    one direction only -- cuts of one video share a track -- so two uploads on
    the same track sit in each other's pools and the same-music control reports
    a number it did not control.  Fingerprint pairs close that direction, and
    the pool must shrink to prove they did.
    """
    names = ["wild_v4:{}:clip{:03d}_slice0".format(upload, cut)
             for upload in (10, 11, 12, 13) for cut in range(3)]
    release = _synthetic_release(tmp_path, names)
    pairs = _pair_file(tmp_path, [("wild_v4:10:clip000", "wild_v4:11:clip001"),
                                  ("wild_v4:12:clip002", "wild_v4:13:clip000")])
    report = evaluate(release, "test", clip_frames=150, feature="kinetic",
                      music_aggregate="mean_std", exclude_same_music=True,
                      music_pairs=pairs)
    tightened = report["excluding_same_music_and_verified_pairs"]
    assert tightened["excluded_verified_pairs"] == 2
    assert (tightened["mean_candidate_pool"]
            < report["excluding_same_music"]["mean_candidate_pool"])
    unit = report["music_group_unit"]["verified_pairs"]
    assert unit["applied_within_split"] == 2
    assert unit["pool_shrank_by"] > 0
    # The looser control stays in the report: dropping it would hide how much
    # of the exclusion the key already bought.
    assert "excluding_same_music" in report


def test_a_pair_list_naming_nothing_in_this_split_is_refused(tmp_path):
    """CLAUDE.md 2: a gate that cannot fire is worse than no gate.

    The fingerprint runs over the whole corpus, so most of its pairs live in
    train.  A file whose pairs touch *no* two clips of this split removed
    nothing, and an R computed that way is indistinguishable from one that never
    asked for the control -- so it is an error rather than a quiet pass.
    """
    names = ["wild_v4:{}:clip{:03d}_slice0".format(upload, cut)
             for upload in (10, 11, 12, 13) for cut in range(3)]
    release = _synthetic_release(tmp_path, names)
    pairs = _pair_file(tmp_path, [("wild_v4:90:clip000", "wild_v4:91:clip000")])
    with pytest.raises(RPrecisionError, match="removed nothing"):
        evaluate(release, "test", clip_frames=150, feature="kinetic",
                 music_aggregate="mean_std", exclude_same_music=True, music_pairs=pairs)


def test_pair_list_without_the_music_control_is_refused(tmp_path):
    names = ["wild_v4:{}:clip{:03d}_slice0".format(upload, cut)
             for upload in (10, 11, 12, 13) for cut in range(3)]
    release = _synthetic_release(tmp_path, names)
    pairs = _pair_file(tmp_path, [("wild_v4:10:clip000", "wild_v4:11:clip001")])
    with pytest.raises(RPrecisionError, match="needs --exclude-same-music"):
        evaluate(release, "test", clip_frames=150, feature="kinetic",
                 music_aggregate="mean_std", music_pairs=pairs)


def test_loading_pairs_dedupes_and_records_the_operating_point(tmp_path):
    path = _pair_file(tmp_path, [("a", "b"), ("b", "a"), ("c", "d"), ("e", "e")])
    pairs, provenance = load_verified_music_pairs(path)
    # (a, b) and (b, a) are one pair; a clip paired with itself is none.
    assert pairs == [frozenset(("a", "b")), frozenset(("c", "d"))]
    assert provenance["rows"] == 4
    assert provenance["distinct_pairs"] == 2
    assert provenance["min_score"] == 0.9


def test_the_pair_flag_is_off_by_default():
    assert build_parser().parse_args(["--release", "somewhere"]).music_pairs is None


# --- per-sequence featurisation -------------------------------------------
#
# The motion features need joint positions, and joint positions cost one SMPL
# forward pass.  Windows at stride 15 overlap ten deep, so doing that pass per
# window repeats it over every frame about nine times.  The fast path runs it
# once per sequence and slices the windows back out; these tests pin that this
# is an identity and not an approximation, and that it refuses to run on a
# release whose windows are not laid out the way it assumes.


def _overlapping_release(tmp_path, sequences, length=150, stride=15, policy=True):
    """A release whose windows really are strided cuts of a source sequence.

    ``_synthetic_release`` draws each window independently, which is right for
    the eligibility tests above and useless here: the whole question is whether
    two windows that share frames are handled consistently, and independent
    draws share none.
    """
    import torch

    rng = np.random.default_rng(7)
    windows, names = [], []
    for seq_name, frames in sequences:
        source = rng.uniform(-1.0, 1.0, size=(frames, 151)).astype(np.float32)
        starts = list(range(0, frames - length + 1, stride))
        for index, start in enumerate(starts):
            windows.append(source[start:start + length])
            names.append("{}_slice{}".format(seq_name, index))
    motion = np.stack(windows)
    split = tmp_path / "test"
    split.mkdir(parents=True)
    np.save(split / "motion.npy", motion)
    np.save(split / "music.npy",
            rng.normal(size=(len(motion), length, 35)).astype(np.float32))
    (split / "names.json").write_text(json.dumps(names), encoding="utf-8")
    torch.save({"data_min": torch.zeros(151), "data_max": torch.ones(151)},
               tmp_path / "normalizer.pt")
    if policy:
        (tmp_path / "build.json").write_text(json.dumps(
            {"window_policy": {"window_length": length, "window_stride": stride}}),
            encoding="utf-8")
    return tmp_path, names


def _features(release, **kwargs):
    from tools.eval_r_precision import build_clip_features

    return build_clip_features(release, "test", clip_frames=150, feature="kinetic",
                               music_aggregate="mean_std", **kwargs)


def test_forward_kinematics_is_frame_independent():
    """The property the fast path rests on, asserted rather than assumed.

    If SMPL FK had any temporal coupling -- a filter, a running root -- then
    slicing a sequence's joints would not equal running FK on the slice, and the
    per-sequence path would silently produce a different dance.
    """
    from tools.convert_motion_to_guofeats import motion_151_to_joints

    rng = np.random.default_rng(3)
    raw = rng.normal(size=(90, 151)).astype(np.float32)
    whole = motion_151_to_joints(raw)
    for start, stop in ((0, 30), (30, 60), (17, 77)):
        part = motion_151_to_joints(raw[start:stop])
        assert np.array_equal(whole[start:stop], part), (start, stop)


def test_per_sequence_featurisation_is_bit_identical_to_per_window(tmp_path):
    fast, _ = _overlapping_release(tmp_path / "fast", [("seqA", 300), ("seqB", 240)])
    slow, _ = _overlapping_release(tmp_path / "slow", [("seqA", 300), ("seqB", 240)],
                                   policy=False)

    quick = _features(fast)
    plain = _features(slow)

    assert quick["featurisation"]["mode"] == "per_sequence"
    assert quick["featurisation"]["sequences"] == 2
    assert plain["featurisation"]["mode"] == "per_window"
    # Bit-identical, not merely close: the point of the change is that already
    # published R figures stay comparable, and "almost the same features" is a
    # different claim from "the same features".
    assert np.array_equal(quick["motion"], plain["motion"])
    assert np.array_equal(quick["music"], plain["music"])
    assert quick["groups"] == plain["groups"]
    assert quick["clips"] == plain["clips"]
    assert quick["windows"] == plain["windows"]


def test_the_stitch_refuses_windows_that_do_not_overlap_as_stated(tmp_path):
    """The gate can fail, and here is the input that fires it.

    A release whose stated stride does not describe its array would otherwise be
    stitched into a sequence nobody generated, and every number downstream would
    be confidently wrong rather than absent.
    """
    release, _ = _overlapping_release(tmp_path, [("seqA", 300)])
    motion = np.load(release / "test" / "motion.npy")
    motion[3, 0, 0] = motion[3, 0, 0] + 1.0      # break window 3's overlap with window 2
    np.save(release / "test" / "motion.npy", motion)

    with pytest.raises(RPrecisionError, match="does not overlap window"):
        _features(release)


def test_a_release_without_a_stated_policy_takes_the_slow_path(tmp_path):
    release, _ = _overlapping_release(tmp_path, [("seqA", 300)], policy=False)
    assert _features(release)["featurisation"]["mode"] == "per_window"


def test_a_stride_that_does_not_overlap_takes_the_slow_path(tmp_path):
    """stride >= length leaves no redundant frame, so the fast path buys nothing."""
    release, _ = _overlapping_release(tmp_path, [("seqA", 600)], length=150, stride=150)
    assert _features(release)["featurisation"]["mode"] == "per_window"


def test_interleaved_windows_fall_back_instead_of_being_stitched(tmp_path):
    """Reordering breaks the layout the stitch assumes; it must notice, not guess."""
    release, names = _overlapping_release(tmp_path, [("seqA", 300), ("seqB", 240)])
    shuffled = list(names)
    shuffled[0], shuffled[-1] = shuffled[-1], shuffled[0]
    (release / "test" / "names.json").write_text(json.dumps(shuffled), encoding="utf-8")
    assert _features(release)["featurisation"]["mode"] == "per_window"


def test_the_limit_flag_still_takes_the_slow_path_when_it_cuts_a_sequence(tmp_path):
    """--limit truncates the array, so the last sequence loses its tail windows.

    Those remaining windows are still a contiguous 0..n-1 run, so stitching them
    is still exact -- it just rebuilds a shorter sequence.  Pinned here because
    the alternative reading (limit invalidates the fast path) would be a silent
    5x slowdown on every --limit run, and this repo has already paid for one
    defect whose only symptom was being slow.
    """
    release, _ = _overlapping_release(tmp_path, [("seqA", 300), ("seqB", 240)])
    cut = _features(release, limit=6)
    assert cut["featurisation"]["mode"] == "per_sequence"
    assert cut["windows"] == 6


def test_the_report_says_which_featurisation_path_it_took(tmp_path):
    """A run that is slow should say why, in the artifact and not only in a log.

    The two paths are bit-identical, so this field records cost rather than
    method -- but the defect it exists for had no symptom except being slow, and
    nothing reported it for a day.
    """
    release, _ = _overlapping_release(tmp_path, [("seqA", 300), ("seqB", 240)])
    report = evaluate(release, "test", clip_frames=150, feature="kinetic",
                      music_aggregate="mean_std")
    assert report["featurisation"]["mode"] == "per_sequence"
    assert report["featurisation"]["sequences"] == 2
    json.dumps(report)
