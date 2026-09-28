"""The learned retrieval rule, and the invariances that make it mean anything.

WHY THESE TESTS EXIST AND NOT OTHERS.  A scorer trained on "segment i+1 follows
segment i" is trivially winnable in two ways that produce excellent held-out
numbers and no ability, and both were observed rather than imagined:

  * the positive is the only candidate cut from the query's own upload, so
    anything upload-specific names it;
  * with uniform negatives the first smoke run read held-out recall@1 0.8467
    against a duration baseline of 0.2100, and STILL read 0.4067 with the seam
    block zeroed -- i.e. it could name the successor twice as often as chance
    with the join hidden, which is only possible if the features carry identity.

The first is closed by geometry (``align_to`` plus body-frame travel plus
joint-only summaries) and is asserted here as a hard invariance:
**rigidly spinning and translating a candidate must not move its feature row.**
The second is closed by the negative sampler and is asserted in
``test_negatives_include_a_same_recording_member``.

The rest of the file holds the small contracts that a later edit could break
silently: the feature width, the seam ablation actually zeroing the seam, the
sampler never leaving the top k, and the library refusing the rule when no
checkpoint was given -- the last because a run that asked for ``learned``, got
``duration``, and recorded ``learned`` in its manifest is the defect shape this
repository keeps paying for.
"""
import math
import pathlib
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dataset.rotation_ops import matrix_to_rotation_6d  # noqa: E402
import model.retrieval_selector as rs  # noqa: E402


def make_raw(frames, seed=0):
    """A [frames, 151] raw motion whose rot6d blocks are real rotations."""
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(frames, rs.MOTION_DIM, generator=generator)
    for joint in range(24):
        start = rs.ROT6D_START + 6 * joint
        matrices = torch.linalg.qr(
            torch.randn(frames, 3, 3, generator=generator))[0]
        values[:, start:start + 6] = matrix_to_rotation_6d(matrices)
    return values


def make_query(previous=None, target=28):
    return rs.QueryContext(
        previous_tail=previous,
        target_length=target,
        gap_frames=0,
        beat_phase=0.125,
        beat_period=15.0,
        onset_mean=0.3,
        onset_profile=torch.tensor([0.1, 0.2, -0.1, 0.4]),
        beats_in_span=2.0,
        next_contacts=torch.tensor([1.0, 0.0, 1.0, 0.0]),
        next_activity=torch.tensor([0.1, 0.2, 0.3, 0.4]))


def test_feature_row_has_the_declared_width():
    row = rs.candidate_features(make_query(make_raw(rs.EDGE_FRAMES, 1)),
                                rs.describe_segment(make_raw(30, 2)))
    assert row.numel() == rs.FEATURE_DIM
    assert torch.isfinite(row).all()


def test_rigidly_replacing_a_candidate_does_not_move_its_features():
    """THE leak check.  See the module docstring.

    A candidate spun about the vertical axis and moved across the floor is the
    same dance.  If the feature row moves, the scorer can read which recording
    a candidate came from, and the true successor -- the only candidate from the
    query's own upload -- is identifiable without the join being learned.
    Before the fix this read 2.4e-3, carried by the global-orient block leaking
    into the scalar summaries and by ground travel being expressed in the world.
    """
    query = make_query(make_raw(rs.EDGE_FRAMES, 1))
    candidate = make_raw(30, 2)
    moved = rs.rotate_about_z(candidate, 1.234)
    moved[:, rs.ROOT_POSITION_START:rs.ROOT_POSITION_START + 2] += torch.tensor([3.5, -2.0])

    before = rs.candidate_features(query, rs.describe_segment(candidate))
    after = rs.candidate_features(query, rs.describe_segment(moved))
    assert torch.allclose(before, after, atol=1e-5), \
        "max |delta| = {:.3e}".format(float((before - after).abs().max()))


def test_align_to_lands_the_candidate_on_the_previous_tail():
    previous = make_raw(rs.EDGE_FRAMES, 1)
    candidate = rs.describe_segment(make_raw(30, 2))
    aligned = rs.align_to(candidate.edges, previous)

    yaw_previous, _ = rs._yaw_and_matrices(previous[-1:])
    yaw_aligned, _ = rs._yaw_and_matrices(aligned[:1])
    assert abs(float(yaw_previous[0] - yaw_aligned[0])) < 1e-5
    ground = slice(rs.ROOT_POSITION_START, rs.ROOT_POSITION_START + 2)
    assert torch.allclose(aligned[0, ground], previous[-1, ground], atol=1e-5)


def test_align_to_never_takes_the_long_way_round():
    """A half-turn must be corrected by a half-turn, not by 3/2 of one."""
    previous = make_raw(rs.EDGE_FRAMES, 1)
    candidate = rs.rotate_about_z(make_raw(30, 2), 3.0)   # nearly pi away
    aligned = rs.align_to(rs.describe_segment(candidate).edges, previous)
    yaw_previous, _ = rs._yaw_and_matrices(previous[-1:])
    yaw_aligned, _ = rs._yaw_and_matrices(aligned[:1])
    gap = float(torch.remainder(yaw_previous[0] - yaw_aligned[0] + math.pi,
                                2 * math.pi) - math.pi)
    assert abs(gap) < 1e-5


def test_limb_grouping_matches_the_inference_path():
    """Duplicated to avoid importing the inference stack; kept honest here."""
    import infer_atomic

    assert rs.LIMB_JOINTS == infer_atomic._LIMB_JOINTS
    assert rs.LIMB_DIMS == {name: infer_atomic.LIMB_DIMS[name]
                            for name in rs.LIMB_NAMES}


def test_time_stretch_leaves_the_endpoints_exact():
    """Why the seam features may be read off the UNSTRETCHED segment.

    ``_values_at`` resamples with ``align_corners=True``; the first and last
    frames survive it byte for byte, so only the seam VELOCITY needs the
    1/stretch correction that ``candidate_features`` applies.
    """
    import torch.nn.functional as F

    values = make_raw(30, 3)
    stretched = F.interpolate(values.T.unsqueeze(0), size=45, mode="linear",
                              align_corners=True).squeeze(0).T
    assert torch.allclose(stretched[0], values[0], atol=1e-6)
    assert torch.allclose(stretched[-1], values[-1], atol=1e-6)


def test_feature_groups_tile_the_row_exactly():
    """A group table with a hole or an overlap makes every control a lie."""
    spans = sorted(rs.FEATURE_GROUPS.values())
    cursor = 0
    for low, high in spans:
        assert low == cursor, (low, cursor)
        cursor = high
    assert cursor == rs.FEATURE_DIM
    assert all(name in rs.FEATURE_GROUPS for name in rs.SEAM_GROUPS)


def test_seam_groups_are_the_ones_that_move_with_the_previous_tail():
    """The definition of the seam is "computed against the predecessor", and a
    control named seam-blind has to remove all of it.  The first version named
    only the two rot6d blocks and left contact and root continuity in, which is
    why its control read 0.9798 and looked like a leak."""
    import dataclasses

    candidate = rs.describe_segment(make_raw(30, 2))
    base = rs.candidate_features(make_query(make_raw(rs.EDGE_FRAMES, 1)), candidate)
    other = rs.candidate_features(make_query(make_raw(rs.EDGE_FRAMES, 9)), candidate)
    moved = {name for name, (low, high) in rs.FEATURE_GROUPS.items()
             if float((base - other)[low:high].abs().max()) > 1e-6}
    assert moved == set(rs.SEAM_GROUPS)


def test_dropping_the_seam_zeroes_exactly_the_seam_block():
    model = rs.RetrievalSelector().eval()   # dropout would make forward stochastic
    rows = torch.randn(4, rs.FEATURE_DIM)
    with torch.no_grad():
        plain = model(rows)
        ablated = model(rows, drop_seam=True)
        masked = rows.clone()
        for name in rs.SEAM_GROUPS:
            low, high = rs.FEATURE_GROUPS[name]
            masked[:, low:high] = 0.0
        expected = model(masked)
    assert torch.allclose(ablated, expected)
    assert not torch.allclose(ablated, plain)


def test_select_stays_inside_the_top_k_and_is_reproducible():
    model = rs.RetrievalSelector()
    rows = torch.randn(40, rs.FEATURE_DIM)
    with torch.no_grad():
        scores = model.eval()(rows)
    model.train()   # select must force eval itself, so leave it in train mode
    top = set(torch.topk(scores, 5).indices.tolist())
    for seed in range(20):
        pick = model.select(rows, top_k=5, temperature=1.0,
                            generator=np.random.default_rng(seed))
        assert pick in top
    a = model.select(rows, top_k=5, generator=np.random.default_rng(7))
    b = model.select(rows, top_k=5, generator=np.random.default_rng(7))
    assert a == b
    assert model.training, "select must restore the mode it was called in"


def test_zero_temperature_is_the_argmax_and_is_not_the_default():
    """The default must SAMPLE.  ``--retrieval-rule phase`` won its own
    criterion and lost on the output because an argmax collapsed the pool
    (docs/DANCE_QUALITY_DEFECTS.md section 15.8); a default that quietly became
    an argmax would repeat it."""
    model = rs.RetrievalSelector()
    rows = torch.randn(30, rs.FEATURE_DIM)
    with torch.no_grad():
        best = int(model.eval()(rows).argmax())
    model.train()
    assert model.select(rows, top_k=8, temperature=0.0) == best
    picks = {model.select(rows, top_k=8, generator=np.random.default_rng(s))
             for s in range(40)}
    assert len(picks) > 1


def test_onset_z_on_a_constant_track_is_exactly_zero():
    """Relative guard, same rule as MusicPhaseFeatures: for a constant channel
    float32 rounding leaves std at ~2e-7 and a literal ``std > 0`` test would
    amplify that rounding to full scale."""
    music = torch.zeros(90, 35)
    music[:, 0] = 0.37
    track = rs.onset_z(music)
    assert torch.allclose(track, torch.zeros_like(track))


def test_window_profile_bins_and_survives_an_empty_span():
    track = torch.arange(40, dtype=torch.float32)
    profile = rs.window_profile(track, 0, 40, bins=4)
    assert profile.numel() == 4
    assert profile[0] < profile[1] < profile[2] < profile[3]
    assert torch.allclose(rs.window_profile(track, 10, 10), torch.zeros(4))
    assert torch.allclose(rs.window_profile(None, 0, 10), torch.zeros(4))


def test_library_refuses_the_learned_rule_without_a_checkpoint():
    """Fail closed.  Degrading to ``duration`` while the manifest says
    ``learned`` would make two arms that are the same dance look like two."""
    import infer_atomic

    with pytest.raises(ValueError, match="retrieval-selector"):
        infer_atomic.IndexedAtomicMotionLibrary(
            "/nonexistent-release", retrieval_rule="learned")


def test_negatives_include_a_same_recording_member():
    """Without this the positive is the only candidate from its own upload."""
    import types

    import tools.train_retrieval_selector as trainer

    descriptors = [types.SimpleNamespace(length=30, beat_phase=0.1, beat_period=15.0)
                   for _ in range(12)]
    corpus = types.SimpleNamespace(
        descriptors=descriptors,
        meta=[(index, 1, 0, 30) for index in range(12)])
    groups = ["own" if index < 3 else "other-{}".format(index) for index in range(12)]
    picked = trainer.choose_negatives(
        corpus, list(range(12)), groups, "own", target=30, own_phase=0.1,
        own_period=15.0, count=8, rng=np.random.default_rng(0))
    assert any(groups[i] == "own" for i in picked)
    assert any(groups[i] != "own" for i in picked)
    assert len(picked) == len(set(picked))


def test_negatives_refuse_a_pool_with_no_outside_member():
    """A class whose only members are the query's own recording cannot supply a
    source-safe negative, and inventing one would train on a leak."""
    import types

    import tools.train_retrieval_selector as trainer

    descriptors = [types.SimpleNamespace(length=30, beat_phase=0.1, beat_period=15.0)
                   for _ in range(4)]
    corpus = types.SimpleNamespace(descriptors=descriptors,
                                   meta=[(i, 1, 0, 30) for i in range(4)])
    picked = trainer.choose_negatives(
        corpus, list(range(4)), ["own"] * 4, "own", target=30, own_phase=0.1,
        own_period=15.0, count=8, rng=np.random.default_rng(0))
    assert picked == []
