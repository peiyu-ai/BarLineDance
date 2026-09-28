"""The derived rhythm channels: beat phase, countdown, and windowed onset energy.

WHY THESE CHANNELS EXIST.  The completion's output energy barely tracks the
song -- per-clip energy correlation 0.210 against the ground truth, and the
music-normalization arm alone moved it only to 0.17-0.26
(docs/DANCE_QUALITY_DEFECTS.md section 12.1); post-processing was judged and
refused by the operator (section 12.4).  The input-side mechanism: beat PHASE
is represented nowhere.  Channel 34 is a one-hot that is zero on ~19 of every
20 frames, it reaches the model through a bare Linear into cross-attention,
and the FiLM path mean-pools the cond tokens, which destroys any periodic
signal.  ``MusicPhaseFeatures`` derives the phase explicitly from the raw
35-D input, so the release arrays never need rebuilding.

What is held here: the derivation itself against hand-computed references, the
opt-in (off by default, refused checkpoint loads in both directions -- the
pattern of tests/test_music_normalization.py), the derive-from-RAW ordering
(z-scoring rescales channel 34 out of its > 0.5 convention), and invariance to
the feature scales that vary across corpora.
"""
import argparse
import math
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from model.atomic_planner import AtomicPlannerTransformer, MusicPhaseFeatures
from model.atomic_completion import AtomicCompletionDecoder
import train_atomic

BEAT = MusicPhaseFeatures.BEAT_CHANNEL
SIN, COS, VALID, COUNTDOWN, ONSET_Z = range(5)


def _music(beat_frames, frames=150, batch=1):
    music = torch.zeros(batch, frames, 35)
    for frame in beat_frames:
        music[:, frame, BEAT] = 1.0
    # a non-constant onset so the z-score channel has something to read
    music[..., 0] = torch.linspace(0.0, 2.0, frames)
    return music


# ---------------------------------------------------------------------------
# (a) The phase itself, against a hand-computed reference.
# ---------------------------------------------------------------------------

def test_phase_advances_linearly_between_beats_and_is_continuous_at_them():
    beats = list(range(10, 150, 15))          # 10, 25, ..., 145: gaps of 15
    out = MusicPhaseFeatures()(_music(beats))[0]
    # inside every gap the phase is 2*pi*k/15, and at the closing beat it
    # wraps to exactly 0 rather than reaching 2*pi -- same point on the circle
    for start in beats[:-1]:
        for k in range(15):
            expected = 2.0 * math.pi * k / 15.0
            assert float(out[start + k, SIN]) == pytest.approx(math.sin(expected), abs=1e-5)
            assert float(out[start + k, COS]) == pytest.approx(math.cos(expected), abs=1e-5)
    # continuity through the wrap: within the valid span, no per-frame jump of
    # sin or cos may exceed the in-gap step (2*pi/15 bounds both derivatives),
    # and a raw-phase channel would fail this at every beat with a jump of ~2*pi
    step = 2.0 * math.pi / 15.0
    for channel in (SIN, COS):
        deltas = (out[11:146, channel] - out[10:145, channel]).abs()
        assert float(deltas.max()) <= step + 1e-5
    # a beat frame is the unique point (0, 1)
    assert torch.allclose(out[beats, SIN], torch.zeros(len(beats)), atol=1e-6)
    assert torch.allclose(out[beats, COS], torch.ones(len(beats)), atol=1e-6)


def test_validity_covers_first_beat_to_last_beat_and_zeroes_the_phase_outside():
    beats = list(range(10, 150, 15))          # first 10, last 145
    out = MusicPhaseFeatures()(_music(beats))[0]
    assert out[:10, VALID].eq(0).all() and out[146:, VALID].eq(0).all()
    assert out[10:146, VALID].eq(1).all()
    # outside: sin AND cos zeroed -- (0, 0) is off the unit circle, so an
    # invalid stretch is separable even by a model that never reads the flag
    for channel in (SIN, COS):
        assert out[:10, channel].eq(0).all()
        assert out[146:, channel].eq(0).all()


# ---------------------------------------------------------------------------
# (d) The countdown, against an independent per-frame reference.
# ---------------------------------------------------------------------------

def test_countdown_hits_zero_exactly_on_beats_and_saturates_where_none_is_coming():
    beats = set(range(10, 150, 15))
    out = MusicPhaseFeatures()(_music(sorted(beats)))[0]
    clamp = 30.0
    expected, upcoming = torch.ones(150), None
    for frame in range(149, -1, -1):          # deliberate slow reference
        if frame in beats:
            upcoming = frame
        if upcoming is not None:
            expected[frame] = min(float(upcoming - frame), clamp) / clamp
    assert torch.allclose(out[:, COUNTDOWN], expected, atol=1e-6)
    assert out[sorted(beats), COUNTDOWN].eq(0).all()
    # after the last beat there is no next beat: pinned at 1.0, not extrapolated
    assert out[146:, COUNTDOWN].eq(1).all()


# ---------------------------------------------------------------------------
# Windows with fewer than two beats: nothing is invented.
# ---------------------------------------------------------------------------

def test_a_beatless_window_is_flagged_invalid_not_faked():
    out = MusicPhaseFeatures()(_music([]))[0]
    assert torch.isfinite(out).all()
    assert out[:, VALID].eq(0).all()
    assert out[:, SIN].eq(0).all() and out[:, COS].eq(0).all()
    assert out[:, COUNTDOWN].eq(1).all()


def test_a_single_beat_is_valid_only_at_its_own_frame():
    out = MusicPhaseFeatures()(_music([70]))[0]
    assert out[:, VALID].sum() == 1 and out[70, VALID] == 1
    # phase at the one beat is defined: the point (0, 1)
    assert float(out[70, SIN]) == 0.0 and float(out[70, COS]) == 1.0
    # the countdown still counts down to it and saturates after it
    assert float(out[70, COUNTDOWN]) == 0.0
    assert float(out[69, COUNTDOWN]) == pytest.approx(1.0 / 30.0)
    assert float(out[40, COUNTDOWN]) == 1.0
    assert out[71:, COUNTDOWN].eq(1).all()


def test_batched_sequences_are_computed_independently():
    """The cummax scans run along dim=1; a leak across the batch would read as
    a plausible phase, not as an error, so it is pinned here."""
    music = torch.cat((_music(list(range(10, 150, 15))),
                       _music(list(range(5, 150, 20)))))
    module = MusicPhaseFeatures()
    together = module(music)
    each = torch.cat((module(music[0:1]), module(music[1:2])))
    assert torch.equal(together, each)


# ---------------------------------------------------------------------------
# (c) Scale invariance: the reason these channels need no corpus statistics,
# which is what lets them be derived on the fly with no release rebuild.
# ---------------------------------------------------------------------------

def test_derived_channels_ignore_the_feature_scales():
    torch.manual_seed(0)
    music = torch.rand(2, 150, 35)            # MFCC-slot noise included
    music[..., BEAT] = 0.0
    music[:, list(range(10, 150, 15)), BEAT] = 1.0
    module = MusicPhaseFeatures()
    base = module(music)
    for scale in (1e3, 1e6):
        scaled = module(music * scale)
        # phase, validity, countdown: bitwise identical -- they read only the
        # beat mask, which the threshold makes scale-free
        assert torch.equal(scaled[..., :ONSET_Z], base[..., :ONSET_Z])
        # onset-z: invariant up to float rounding, because a positive gain
        # scales the window mean and std identically
        assert torch.allclose(scaled[..., ONSET_Z], base[..., ONSET_Z], atol=1e-4)


def test_a_constant_onset_reads_zero_not_an_epsilon_spike():
    """The first guard was a literal ``std > 0``, and this test failed against
    it: for a constant window of 3.7, float32 rounding leaves the mean off by
    ~2e-7 and the std at ~2e-7 rather than 0, so the guard passed and pure
    rounding noise came out as a full-scale -1 on every frame.  The shipped
    guard is relative (std above 1e-6 of the channel's magnitude), which is
    what this asserts."""
    music = _music([10, 25])
    music[..., 0] = 3.7
    out = MusicPhaseFeatures()(music)
    assert out[..., ONSET_Z].eq(0).all()


def test_malformed_input_is_refused_loudly():
    module = MusicPhaseFeatures()
    with pytest.raises(ValueError):
        module(torch.zeros(150, 35))          # missing the batch dimension
    with pytest.raises(ValueError):
        module(torch.zeros(1, 150, 34))       # no beat channel to read


# ---------------------------------------------------------------------------
# (b) The opt-in and the checkpoint contract, in both stages -- the pattern of
# tests/test_music_normalization.py.
# ---------------------------------------------------------------------------

def _planner(**kw):
    return AtomicPlannerTransformer(num_atomic_classes=4, music_dim=35, latent_dim=16,
                                    num_layers=1, num_heads=2, ff_size=16,
                                    max_seq_len=8, **kw)


def _completion(**kw):
    return AtomicCompletionDecoder(motion_dim=12, seq_len=8, music_dim=35, latent_dim=16,
                                   ff_size=16, num_layers=1, num_heads=2, **kw)


@pytest.mark.parametrize("build", [_planner, _completion])
def test_it_is_off_unless_asked_for(build):
    assert build().music_phase_features is None
    assert build(music_phase_features=True).music_phase_features is not None


def test_the_music_projection_widens_by_exactly_the_derived_channels():
    assert _planner().music_projection.in_features == 35
    assert _planner(music_phase_features=True).music_projection.in_features == 40
    assert _completion().denoiser.cond_projection.in_features == 35
    assert _completion(music_phase_features=True).denoiser.cond_projection.in_features == 40


@pytest.mark.parametrize("build", [_planner, _completion])
def test_a_checkpoint_refuses_to_load_across_the_mismatch(build):
    """Same rationale as MusicNormalization: a flag recorded in args but not
    applied is the defect shape this repository paid for three times on
    2026-08-30.  Here the refusal is structural twice over -- the projection is
    5 wider AND the marker buffer exists -- and it must hold in both
    directions."""
    with_phase = build(music_phase_features=True).state_dict()
    without = build().state_dict()
    with pytest.raises(RuntimeError):
        build().load_state_dict(with_phase)
    with pytest.raises(RuntimeError):
        build(music_phase_features=True).load_state_dict(without)
    build(music_phase_features=True).load_state_dict(with_phase)


def test_the_clamp_horizon_travels_in_the_state_dict():
    """The countdown clamp is the derivation's one free constant; as a buffer
    it cannot silently differ between training and inference."""
    keys = _completion(music_phase_features=True).state_dict()
    assert "music_phase_features.countdown_clamp_frames" in keys
    keys = _planner(music_phase_features=True).state_dict()
    assert "music_phase_features.countdown_clamp_frames" in keys


def test_turning_it_on_changes_both_stages_outputs():
    music = _music(list(range(1, 8, 3)), frames=8)
    torch.manual_seed(0)
    plain_planner = _planner()
    torch.manual_seed(0)
    phased_planner = _planner(music_phase_features=True)
    labels = torch.zeros(1, 8, dtype=torch.long)
    steps = torch.zeros(1, dtype=torch.long)
    assert not torch.allclose(plain_planner(labels, music, steps),
                              phased_planner(labels, music, steps))
    torch.manual_seed(0)
    plain_completion = _completion().eval()
    torch.manual_seed(0)
    phased_completion = _completion(music_phase_features=True).eval()
    motion, draft = torch.randn(1, 8, 12), torch.randn(1, 8, 12)
    mask = torch.zeros(1, 8, 1)
    with torch.no_grad():
        assert not torch.allclose(plain_completion(motion, music, steps, draft, mask),
                                  phased_completion(motion, music, steps, draft, mask))


# ---------------------------------------------------------------------------
# The ordering that makes the derivation correct: RAW in, then normalize.
# ---------------------------------------------------------------------------

def _corrupting_stats():
    """Statistics under which normalize-then-derive would hallucinate a beat on
    EVERY frame: (0 - (-1)) / 1 = 1 > 0.5 on the beatless frames."""
    mean = torch.zeros(35)
    mean[BEAT] = -1.0
    return {"mean": mean, "std": torch.ones(35)}


@pytest.mark.parametrize("stage", ["planner", "completion"])
def test_derivation_reads_the_raw_music_not_the_normalized_music(stage):
    if stage == "planner":
        model = _planner(music_phase_features=True, music_stats=_corrupting_stats())
    else:
        model = _completion(music_phase_features=True, music_stats=_corrupting_stats())
    music = _music([2, 5], frames=8)
    seen = []
    model.music_phase_features.register_forward_pre_hook(
        lambda module, inputs: seen.append(inputs[0]))
    steps = torch.zeros(1, dtype=torch.long)
    if stage == "planner":
        model(torch.zeros(1, 8, dtype=torch.long), music, steps)
    else:
        model(torch.randn(1, 8, 12), music, steps,
              torch.randn(1, 8, 12), torch.zeros(1, 8, 1))
    assert len(seen) == 1
    assert torch.equal(seen[0], music), (
        "the phase module must see the RAW input; these stats would turn every "
        "normalized frame into a beat")


# ---------------------------------------------------------------------------
# The flag, and the factories every checkpoint rebuilds through
# (infer_atomic._load_checkpoint -> SimpleNamespace(**checkpoint['args'])).
# ---------------------------------------------------------------------------

def _factory_args(**extra):
    base = dict(num_classes=7, motion_dim=16, seq_len=8, music_dim=35, latent_dim=32,
                ff_size=32, layers=1, heads=2, dropout=0.0, diffusion_steps=4,
                transition_weight=1.0, cond_drop_prob=0.25, guidance_weight=2.0)
    base.update(extra)
    return argparse.Namespace(**base)


def test_the_flag_is_off_by_default_and_reaches_both_factories():
    source = pathlib.Path(train_atomic.__file__).read_text(encoding="utf-8")
    assert '"--music-phase-features"' in source
    # an args namespace WITHOUT the attribute is exactly what a checkpoint
    # saved before this flag existed hands to the factories; it must rebuild
    # the plain model, or no old checkpoint would load again
    old = _factory_args()
    assert train_atomic.planner_model(old).model.music_phase_features is None
    assert train_atomic.planner_model(old).model.music_projection.in_features == 35
    assert train_atomic.completion_model(old).model.music_phase_features is None
    new = _factory_args(music_phase_features=True)
    assert train_atomic.planner_model(new).model.music_phase_features is not None
    assert train_atomic.planner_model(new).model.music_projection.in_features == 40
    assert train_atomic.completion_model(new).model.music_phase_features is not None
    assert train_atomic.completion_model(new).model.denoiser.cond_projection.in_features == 40
