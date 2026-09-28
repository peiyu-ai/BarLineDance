"""Discrete diffusion planner for frame-wise atomic movement labels.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils import PositionalEncoding, SinusoidalPosEmb, make_beta_schedule


@dataclass
class PlannerOutput:
    loss: torch.Tensor
    logits: torch.Tensor
    noisy_labels: torch.Tensor
    target_labels: torch.Tensor
    timesteps: torch.Tensor


class MusicNormalization(nn.Module):
    """Per-channel z-scoring of the 35-D music vector, carried BY the checkpoint.

    The statistics are registered buffers, so they travel inside
    ``state_dict`` and a checkpoint physically cannot be run without the
    normalization it was trained with.  That is deliberate: the failure this
    repository keeps paying for is a flag that is recorded but not applied
    (three of them on 2026-08-30 alone), and a path in ``args`` that some call
    site forgets to read is exactly that shape again.  A buffer cannot be
    forgotten -- ``load_state_dict`` is strict, so a checkpoint trained with
    stats refuses to load into a module built without them, and vice versa.

    Why any of this is needed: both stages feed the raw librosa stack
    (``data/audio_extraction/baseline_features.py``: onset, 20 MFCC, 12 chroma,
    onset-peak, beat) through one bare ``nn.Linear``.  Measured per-channel
    std on a release array -- MFCC c1 63.9, chroma 0.02-0.05, **beat 0.23** --
    so the beat one-hot arrives at about 0.05% of the input variance and the
    chroma channels at less.  A linear layer's response at initialisation
    scales with the channel, so the beat is not in the input in any usable
    magnitude.  This makes it available; whether anything then uses it is a
    separate question with its own criterion (period lock at n>=400,
    R-precision against its 0.39 ground-truth ceiling).

    Constant channels keep std 1 rather than an epsilon-inflated one -- same
    rule as ``tools/fit_motion_normalizer.py``: scaling up a channel's noise is
    inventing information.
    """

    def __init__(self, music_stats):
        super().__init__()
        mean = torch.as_tensor(music_stats["mean"], dtype=torch.float32).reshape(-1)
        std = torch.as_tensor(music_stats["std"], dtype=torch.float32).reshape(-1)
        if mean.shape != std.shape:
            raise ValueError("music stats mean/std disagree: {} vs {}".format(
                mean.shape, std.shape))
        self.register_buffer("music_mean", mean)
        self.register_buffer("music_std", torch.where(std > 0, std, torch.ones_like(std)))

    def forward(self, music_features):
        if music_features.shape[-1] != self.music_mean.shape[0]:
            raise ValueError("music has {} channels, stats have {}".format(
                music_features.shape[-1], self.music_mean.shape[0]))
        return (music_features - self.music_mean) / self.music_std


class MusicPhaseFeatures(nn.Module):
    """Five rhythm channels DERIVED from the raw 35-D music and appended to it.

    WHY THIS EXISTS.  The completion's output energy barely tracks the song:
    per-clip energy correlation against the ground truth is 0.210, and the
    music-normalization arm alone moved it only to 0.17-0.26
    (docs/DANCE_QUALITY_DEFECTS.md section 12.1); post-processing is ruled out
    by the operator (section 12.4), so what is missing has to be learnable
    structure.  Look at what the model is given to learn rhythm from: the beat
    is channel 34, a one-hot that is zero on roughly 19 of every 20 frames,
    and it reaches the completion through a bare Linear into cross-attention --
    the only rhythm channel -- while the FiLM pathway conditions on a
    MEAN-POOLED cond token (``model/model.py``, ``mean_pooled_cond_tokens``),
    which reduces any periodic signal toward its DC component and carries no
    rhythm at all.  Beat PHASE -- where the current frame stands between one
    beat and the next, the quantity a dancer actually tracks -- is represented
    nowhere; the model would have to integrate a sparse impulse train through
    attention to recover it.  This module hands the phase over explicitly
    instead of asking the optimiser to invent an integrator.

    THE FIVE CHANNELS, in order.  All are scale-fixed by construction, so none
    needs corpus statistics -- which is what lets them be derived on the fly:
    the release arrays stay 35-D, ``MODEL_MUSIC_DIM`` stays 35, nothing is
    rebuilt.

    0  sin(phase)   phase runs 0..2pi piecewise-linearly between consecutive
    1  cos(phase)   beat frames (channel 34 > 0.5, the repo-wide convention:
                    ``infer_atomic.snap_plan_to_bar_grid``,
                    ``eval/utils/musicbeat.py``).  sin/cos rather than the raw
                    phase because the raw phase has a 2pi -> 0 cliff at every
                    beat; on the circle the wrap is continuous and a beat is
                    the unique point (0, 1).
    2  validity     1 from the first to the last beat frame inclusive, else 0.
                    Outside that span sin and cos are ZEROED, not extrapolated:
                    (0, 0) is off the unit circle, unreachable by any true
                    phase, so an invalid stretch stays separable even for a
                    model that never reads this flag.
    3  countdown    frames to the next beat at or after this frame, clamped at
                    ``countdown_clamp_frames`` (default 30 = 1 s at 30 fps) and
                    scaled to [0, 1]; exactly 0 on beat frames.  Tempi of
                    80-140 BPM give beat gaps of 13-22 frames, so between beats
                    it is unsaturated; it pins at 1.0 only where no beat is
                    coming (after the window's last beat, or a beatless
                    window).
    4  onset-z      channel 0 (onset strength) z-scored over THIS window's
                    frames, per sequence: a local loud/quiet signal invariant
                    to the corpus scale and to any per-channel affine
                    rescaling.  A constant onset yields exactly 0 rather than
                    an epsilon-inflated spike -- same rule as
                    ``MusicNormalization``, and the guard is relative (std
                    above 1e-6 of the channel's magnitude), because for a
                    constant window float32 rounding leaves std at ~2e-7
                    rather than 0 and a literal ``std > 0`` guard amplifies
                    that rounding to full scale.

    WINDOWS WITH FEWER THAN TWO BEATS get nothing invented for them.  Zero
    beats: validity 0 everywhere, sin/cos all zero, countdown saturated at 1.
    One beat: phase is defined (as 0) exactly at that beat frame, validity is 1
    only there; countdown still counts down to it and saturates after it.  A
    beat frame is its own predecessor and successor in the two scans below,
    which is why these cases fall out of the same arithmetic instead of being
    special-cased.

    ORDER OF OPERATIONS AT THE CALL SITES.  Derivation MUST read the RAW
    input: ``MusicNormalization`` rescales channel 34 (measured std 0.23, plus
    a fitted mean shift), after which ``> 0.5`` no longer means "a beat".
    Both consumers therefore derive first, z-score the original 35 second, and
    concatenate last; the derived channels are never themselves z-scored,
    being unit-scale by construction.  ``tests/test_music_phase_features.py``
    holds that order with a hook on this module's input.

    WHY A BUFFER.  ``countdown_clamp_frames`` is the derivation's one free
    constant, so it travels in the ``state_dict`` -- the same argument as
    ``MusicNormalization``'s statistics: a constant living only in code could
    change between training and inference with nothing refusing.  The module's
    presence also widens the music projection 35 -> 40, so a mismatched
    checkpoint is refused twice over, in both directions.
    """

    ONSET_CHANNEL = 0    # data/audio_extraction/baseline_features.py layout;
    BEAT_CHANNEL = 34    # same constants as tools/fit_music_normalizer.py
    EXTRA_CHANNELS = 5

    def __init__(self, countdown_clamp_frames: float = 30.0):
        super().__init__()
        self.register_buffer(
            "countdown_clamp_frames",
            torch.tensor(float(countdown_clamp_frames), dtype=torch.float32))

    def forward(self, raw_music: torch.Tensor) -> torch.Tensor:
        """[batch, frames, >=35] RAW music -> [batch, frames, 5] derived channels."""
        if raw_music.ndim != 3:
            raise ValueError("raw music must be [batch, frames, channels], got shape {}".format(
                tuple(raw_music.shape)))
        if raw_music.shape[-1] <= self.BEAT_CHANNEL:
            raise ValueError(
                "phase derivation needs the beat one-hot at channel {}, got {} "
                "channels; it must also be the RAW features -- normalization "
                "rescales the one-hot out of its > 0.5 convention".format(
                    self.BEAT_CHANNEL, raw_music.shape[-1]))
        batch, frames, _ = raw_music.shape
        dtype = raw_music.dtype
        beat = raw_music[..., self.BEAT_CHANNEL] > 0.5
        index = torch.arange(frames, device=raw_music.device)
        index = index.unsqueeze(0).expand(batch, frames)

        # Per frame, the nearest beat at-or-before (-1 while none yet) and
        # at-or-after (``frames`` once none is left), as two cummax scans:
        # batched, no python loop over frames or beats.
        previous = torch.cummax(
            torch.where(beat, index, torch.full_like(index, -1)), dim=1).values
        upcoming = -torch.cummax(
            torch.where(beat, -index, torch.full_like(index, -frames)).flip(1),
            dim=1).values.flip(1)

        position = index.to(dtype)
        valid = (previous >= 0) & (upcoming < frames)
        interval = (upcoming - previous).to(dtype)
        # On a beat frame previous == upcoming, the interval is 0 and the
        # phase is exactly 0 -- the wrap point, not a division.  The clamp
        # only guards the branch that ``where`` evaluates anyway.
        fraction = torch.where(
            interval > 0,
            (position - previous.to(dtype)) / interval.clamp(min=1.0),
            torch.zeros_like(position))
        phase = (2.0 * math.pi) * fraction
        validity = valid.to(dtype)
        phase_sin = torch.sin(phase) * validity
        phase_cos = torch.cos(phase) * validity

        clamp = self.countdown_clamp_frames.to(dtype)
        to_next = torch.where(
            upcoming < frames,
            upcoming.to(dtype) - position,
            torch.full_like(position, float("inf")))
        countdown = torch.minimum(to_next, clamp) / clamp

        onset = raw_music[..., self.ONSET_CHANNEL]
        onset_mean = onset.mean(dim=1, keepdim=True)
        onset_std = onset.std(dim=1, unbiased=False, keepdim=True)
        # The constant-channel guard must be RELATIVE, not ``std > 0``: for a
        # constant window of 3.7, float32 rounding leaves mean off by ~2e-7
        # and std at ~2e-7 rather than 0, and the literal guard then amplifies
        # pure rounding noise to a full-scale +-1 signal -- the epsilon
        # inflation MusicNormalization's rule forbids, arriving through the
        # numerator instead of the denominator (caught by this module's own
        # constant-onset test).  Variation below 1e-6 of the channel's
        # magnitude is beneath float32's resolution of the mean subtraction,
        # so it is rounding, not signal, and reads as exactly 0.
        floor = 1e-6 * onset.abs().amax(dim=1, keepdim=True).clamp(min=1.0)
        onset_z = torch.where(
            onset_std > floor,
            (onset - onset_mean) / onset_std.clamp(min=1e-20),
            torch.zeros_like(onset))

        return torch.stack((phase_sin, phase_cos, validity, countdown, onset_z), dim=-1)


class AtomicPlannerTransformer(nn.Module):
    """Music-conditioned Transformer used as the D3PM reverse model."""

    def __init__(
        self,
        num_atomic_classes: int,
        music_dim: int,
        latent_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        ff_size: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 18000,
        global_music: bool = False,
        cond_drop_prob: float = 0.0,
        head: str = "joint",
        music_stats=None,
        music_phase_features: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_atomic_classes + 1
        self.label_embedding = nn.Embedding(self.num_classes, latent_dim)
        # Derived rhythm channels (see MusicPhaseFeatures above).  Opt-in, off
        # by default: the widened projection refuses every earlier checkpoint's
        # state_dict, and that refusal travelling with the weights is the
        # point.  ``music_dim`` itself stays the 35-D release width -- the
        # widening happens here, on the way into the projection, so no release
        # array is rebuilt.
        self.music_phase_features = (MusicPhaseFeatures()
                                     if music_phase_features else None)
        self.music_projection = nn.Linear(
            music_dim + (MusicPhaseFeatures.EXTRA_CHANNELS
                         if music_phase_features else 0),
            latent_dim)
        self.music_normalization = (MusicNormalization(music_stats)
                                    if music_stats is not None else None)
        # Classifier-free guidance, which the completion stage in this
        # repository has had from the start (``cond_drop_prob`` 0.25,
        # ``guided_forward`` = ``uncond + w * (cond - uncond)``) and the planner
        # has never had.  That asymmetry was never a decision, and the planner's
        # measured failure mode is precisely the one guidance exists to fix.
        #
        # Measured 2026-08-23 on ``planner_v1A_s15/planner_step135648.pt``,
        # 512 windows per split, the x0 head at the noise level the reverse
        # chain starts from (t=99):
        #
        #     music      P(transition) on train    on test
        #     real              0.1781             0.4295
        #     shuffled          0.1782             0.4299
        #     zeroed            0.7142             0.7142
        #
        # Shuffling the music between songs of the same split does not move the
        # transition rate at all, while it changes 96.6% of the argmax labels.
        # So the model is not reading *which* song it is at that noise level --
        # it is reading whether the song is *familiar*, and interpolating
        # between a memorised answer (0.18, the training base rate) and an
        # unconditional prior that is 71% transition.  On held-out music it
        # lands halfway, which is the whole 2.2x excess.
        #
        # ``cond_drop_prob > 0`` trains a learned null condition so that
        # unconditional prior becomes addressable, and ``guided_forward`` can
        # then push away from it.  Off by default: a checkpoint trained without
        # it has no ``null_music`` parameter, so creating one unconditionally
        # would break every existing checkpoint's ``load_state_dict``.
        self.cond_drop_prob = float(cond_drop_prob)
        if self.cond_drop_prob > 0.0:
            self.null_music = nn.Parameter(torch.randn(latent_dim) * 0.02)
        else:
            self.null_music = None
        # The paper's planner is "Full-music-awared": c_music = Enc(M) over the
        # whole track, and it denoises the whole song's label sequence at once.
        # Ours is trained on 150-frame windows and the per-frame projection
        # above gives it exactly those 5 seconds of music, so nothing in the
        # model can see a chorus arrive.  This second path carries a summary of
        # the *entire* track, broadcast to every frame, which buys song-level
        # context without changing the window -- and therefore without turning
        # 14,409 training windows into 911 sequences, which is the sample
        # starvation that already cost the 1,286-class vocabulary its planner.
        #
        # Off by default: a checkpoint trained without it must keep loading.
        self.global_music = global_music
        self.global_projection = (
            nn.Sequential(nn.Linear(music_dim * 2, latent_dim), nn.SiLU(),
                          nn.Linear(latent_dim, latent_dim))
            if global_music else None
        )
        self.null_global = (
            nn.Parameter(torch.randn(latent_dim) * 0.02)
            if (global_music and self.cond_drop_prob > 0.0) else None
        )
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(latent_dim),
            nn.Linear(latent_dim, latent_dim * 4),
            nn.SiLU(),
            nn.Linear(latent_dim * 4, latent_dim),
        )
        self.position = PositionalEncoding(
            latent_dim, dropout=dropout, max_len=max_seq_len, batch_first=True
        )
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, self.num_classes))
        if head not in ("joint", "factorised"):
            raise ValueError("head must be 'joint' or 'factorised', got {!r}".format(head))
        # ``"joint"`` is one softmax over {transition} u {K atomic classes}, which
        # is what the paper writes and what every checkpoint before 2026-08-23
        # was trained as.  It couples two decisions that are not the same
        # decision: raising any class logit lowers P(transition), so when the
        # model is unsure *which* movement belongs here the mass it takes off
        # the classes lands on the one class that is 40x more frequent than the
        # average -- transition.
        #
        # Measured on the released planner, 65 test clips, raw per-window output
        # before any post-processing: 40% of clips come out at 0.188 transition
        # against a ground truth of 0.276 -- fine, if anything low -- while 35%
        # come out at 0.736 with 70% of their bars naming no class at all and
        # only 2.4 distinct classes in the whole clip.  That failing third
        # carries 113% of the corpus's total excess; the rest carries -13%.
        # The failure is per-clip collapse, not a global miscalibration, which
        # is also why a global logit bias fitted on val overshot on test.
        #
        # ``"factorised"`` predicts P(transition) with its own sigmoid and the
        # class identity with a softmax over the K atomic classes only, then
        # recombines them:
        #
        #     log p(0)  = log sigma(z_0)
        #     log p(k)  = log(1 - sigma(z_0)) + log_softmax(z_1..z_K)[k]
        #
        # That is still a distribution over the same K+1 labels, so every piece
        # of the D3PM -- ``q_sample``, ``posterior_logits``, the reverse chain --
        # is untouched; only the way the network parameterises it changes.  What
        # changes is the gradient: class competition no longer drains transition
        # mass, and uncertainty about *which* movement no longer reads as
        # confidence that there is none.
        self.head = head

    def forward(
        self,
        noisy_labels: torch.Tensor,
        music_features: torch.Tensor,
        timesteps: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        global_music: Optional[torch.Tensor] = None,
        cond_drop_prob: Optional[float] = None,
    ) -> torch.Tensor:
        if noisy_labels.ndim != 2:
            raise ValueError("noisy_labels must have shape [batch, frames]")
        if music_features.shape[:2] != noisy_labels.shape:
            raise ValueError("music_features and labels must share batch/frame dimensions")
        if self.global_music and global_music is None:
            # Refused rather than zero-filled.  A model built for the full-track
            # summary and fed none would train against a constant and report
            # nothing; the whole question this path exists to answer is whether
            # that summary carries anything.
            raise ValueError(
                "this planner was built with global_music, so the whole-track "
                "summary is required; passing none would silently make it a "
                "constant"
            )
        drop = self.cond_drop_prob if cond_drop_prob is None else float(cond_drop_prob)
        if drop > 0.0 and self.null_music is None:
            # A model built without the null condition has nothing to fall back
            # to, and silently conditioning anyway would make ``guided_forward``
            # return ``cond`` twice -- i.e. guidance that reports no effect for
            # the one reason no reading of the output could reveal.
            raise ValueError(
                "this planner was built with cond_drop_prob=0, so it has no "
                "learned null condition to drop to; rebuild it with "
                "cond_drop_prob > 0 to use classifier-free guidance")

        # Order matters: phase is derived from the RAW beat one-hot (channel
        # 34), which z-scoring rescales out of its > 0.5 convention -- so
        # derive first, normalize the original 35 second, concatenate last.
        derived = (self.music_phase_features(music_features)
                   if self.music_phase_features is not None else None)
        if self.music_normalization is not None:
            music_features = self.music_normalization(music_features)
        if derived is not None:
            music_features = torch.cat((music_features, derived), dim=-1)
        conditioned = self.music_projection(music_features)
        if drop > 0.0:
            keep = torch.rand(noisy_labels.shape[0], 1, 1, device=noisy_labels.device) >= drop
            conditioned = torch.where(keep, conditioned,
                                      self.null_music.to(conditioned.dtype))
        x = self.label_embedding(noisy_labels) + conditioned
        if self.global_projection is not None:
            summary = self.global_projection(global_music)
            if drop > 0.0:
                summary = torch.where(keep[:, 0], summary,
                                      self.null_global.to(summary.dtype))
            x = x + summary[:, None, :]
        x = x + self.time_embedding(timesteps)[:, None, :]
        x = self.position(x)
        logits = self.output(self.encoder(x, src_key_padding_mask=padding_mask))
        if self.head == "joint":
            return logits
        # Returned as log-probabilities rather than as raw logits.  Everything
        # downstream consumes these through ``softmax``/``log_softmax``, which
        # are idempotent on a normalised vector, so the two heads stay
        # interchangeable at every call site.
        transition = F.logsigmoid(logits[..., :1])
        remainder = F.logsigmoid(-logits[..., :1])
        classes = F.log_softmax(logits[..., 1:], dim=-1)
        return torch.cat((transition, remainder + classes), dim=-1)

    def guided_forward(
        self,
        noisy_labels: torch.Tensor,
        music_features: torch.Tensor,
        timesteps: torch.Tensor,
        guidance_weight: float,
        padding_mask: Optional[torch.Tensor] = None,
        global_music: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``uncond + w * (cond - uncond)`` on the x0 logits.

        Same shape as ``model.model.DanceDecoder.guided_forward``, which the
        completion stage has always used.  Applied to the x0 logits rather than
        to the reverse posterior because the posterior's likelihood term depends
        only on ``y_t`` and is identical under both branches, so guiding it
        would scale a term that carries no conditioning at all.
        """
        unconditional = self.forward(noisy_labels, music_features, timesteps,
                                     padding_mask, global_music, cond_drop_prob=1.0)
        conditional = self.forward(noisy_labels, music_features, timesteps,
                                   padding_mask, global_music, cond_drop_prob=0.0)
        return unconditional + guidance_weight * (conditional - unconditional)


class UniformD3PM(nn.Module):
    """Uniform categorical diffusion over frame-wise atomic labels.

    Two reverse parameterizations are available, because they behave very
    differently and the paper's Eq. (3) does not disambiguate them.

    ``"x0"`` (default) is standard D3PM (Austin et al. 2021): the network
    predicts the *clean* labels from ``y_t``, and a reverse step draws from the
    posterior ``q(y_{t-1} | y_t, y_0)`` marginalised over that prediction.

    ``"eq3"`` reproduces this repository's original literal reading of Eq. (3):
    the network is trained with cross-entropy against ``y_{t-1}``, itself a
    noisy sample.  It is kept only so the two can be compared, and it is
    measurably degenerate -- one forward step perturbs just a ``1 - alpha_t``
    fraction of tokens, so the optimal predictor of ``y_{t-1}`` given ``y_t`` is
    very nearly the identity.  A model trained this way learns to copy: on the
    released AIST kinematic split it reached 0.925 denoising accuracy while
    full-chain sampling stayed at 0.103 on its own training data, below the
    0.223 majority-class baseline, with the loss flat from epoch 175 onward.
    At large ``t`` the target is close to uniform noise, so the objective has an
    irreducible floor and more epochs cannot help.
    """

    def __init__(
        self,
        model: AtomicPlannerTransformer,
        num_steps: int = 100,
        transition_weight: float = 1.0,
        high_noise_prob: float = 0.0,
        schedule: str = "cosine",
        parameterization: str = "x0",
    ) -> None:
        super().__init__()
        if parameterization not in ("x0", "eq3"):
            raise ValueError(
                "parameterization must be 'x0' or 'eq3', got {!r}".format(parameterization)
            )
        self.model = model
        self.num_classes = model.num_classes
        self.num_steps = num_steps
        self.parameterization = parameterization
        if transition_weight <= 0.0:
            raise ValueError("transition weight must be positive, got {!r}"
                             .format(transition_weight))
        self.transition_weight = float(transition_weight)
        if not 0.0 <= high_noise_prob <= 1.0:
            raise ValueError("high noise probability must be in [0, 1], got {!r}"
                             .format(high_noise_prob))
        self.high_noise_prob = float(high_noise_prob)
        betas = torch.as_tensor(make_beta_schedule(schedule, num_steps), dtype=torch.float32)
        alphas = 1.0 - betas
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", torch.cumprod(alphas, dim=0))

    def _uniform_replace(self, labels: torch.Tensor, keep_prob: torch.Tensor) -> torch.Tensor:
        while keep_prob.ndim < labels.ndim:
            keep_prob = keep_prob.unsqueeze(-1)
        keep = torch.rand(labels.shape, device=labels.device) < keep_prob
        random_labels = torch.randint(self.num_classes, labels.shape, device=labels.device)
        return torch.where(keep, labels, random_labels)

    def _class_weights(self, device) -> Optional[torch.Tensor]:
        """Per-class cross-entropy weights, or None when the weight is 1.

        Class 0 is the transition/filler class.  Ground truth spends 30.4% of
        its bars there and the planner's own plans spend 12.3%, so the dance it
        writes is named-moving almost all the time while the real one keeps
        stepping out of the vocabulary.  This is the loss-side lever for that
        gap.

        NOTE: this is NOT ``--transition-weight``.  That flag reaches
        ``AtomicCompletionDecoder`` only; passing it to a planner run trained a
        checkpoint whose recorded ``transition_weight`` was 2.5 and whose
        weights were bit-identical (max difference 0.0) to the 1.0 baseline.
        Planner runs now refuse it, and this is the parameter that does the job.
        """
        if self.transition_weight == 1.0:
            return None
        weights = torch.ones(self.num_classes, device=device)
        weights[0] = self.transition_weight
        return weights

    def q_sample(self, clean_labels: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Sample q(y_t | y_0) using the cumulative token-retention rate."""
        keep_prob = self.alpha_bars.gather(0, timesteps)
        return self._uniform_replace(clean_labels, keep_prob)

    def q_step(self, labels: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Sample q(y_t | y_{t-1}); timesteps use zero-based diffusion indices."""
        keep_prob = self.alphas.gather(0, timesteps)
        return self._uniform_replace(labels, keep_prob)

    def forward(self, *args, **kwargs) -> PlannerOutput:
        """Alias for ``training_step``, so DistributedDataParallel can wrap this.

        DDP hangs its gradient-reduction hooks off ``forward``.  Calling
        ``training_step`` on a wrapped module instead runs identical arithmetic
        with no synchronisation at all: each rank would quietly train its own
        copy on its own shard, the loss curve would look ordinary, and the saved
        checkpoint would be one that had seen 1/N of the corpus.  Nothing about
        that failure is visible from the outside, which is why the training loop
        must call the module, not the method.
        """
        return self.training_step(*args, **kwargs)

    def training_step(
        self,
        clean_labels: torch.Tensor,
        music_features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        global_music: Optional[torch.Tensor] = None,
    ) -> PlannerOutput:
        batch = clean_labels.shape[0]
        if timesteps is None:
            timesteps = torch.randint(self.num_steps, (batch,), device=clean_labels.device)
            if self.high_noise_prob > 0.0:
                # OVERSAMPLE THE NOISE LEVEL THE PLAN IS DECIDED AT.
                #
                # Uniform t spends most of the loss on steps where the noisy
                # labels already carry the answer, and a model can score well
                # there by copying them.  The reverse chain, however, starts at
                # t = T-1, where the labels are nearly uniform and the ONLY
                # information is the music -- so that is the step that decides
                # the plan, and under uniform sampling it is 1% of the budget.
                #
                # Measured 2026-09-10 on the music-aligned K=8 labels, GroupKFold
                # over 226 recordings (3,240 held-out bars): a logistic
                # regression on the same music reads +0.0386 over the majority
                # floor and this planner reads -0.0012, i.e. the signal is in
                # the features and the training is discarding it.
                top = max(1, self.num_steps // 10)
                high = self.num_steps - 1 - torch.randint(
                    top, (batch,), device=clean_labels.device)
                take = torch.rand(batch, device=clean_labels.device) < self.high_noise_prob
                timesteps = torch.where(take, high, timesteps)

        if self.parameterization == "x0":
            # Standard D3PM: corrupt y_0 to y_t in one shot and regress the
            # clean labels.  The target never depends on t, so the gradient
            # points at the data distribution at every noise level.
            noisy = self.q_sample(clean_labels, timesteps)
            target = clean_labels
        else:
            # Construct the exact adjacent pair used in Eq. (3): y_{t-1} -> y_t.
            previous_t = torch.clamp(timesteps - 1, min=0)
            previous = self.q_sample(clean_labels, previous_t)
            previous = torch.where((timesteps == 0)[:, None], clean_labels, previous)
            noisy = self.q_step(previous, timesteps)
            target = previous

        # No explicit ``cond_drop_prob``: training is the one place the module's
        # own rate is the right one, and saying so here keeps the two call sites
        # from being read as the same call.
        logits = self.model(noisy, music_features, timesteps, padding_mask,
                            global_music=global_music)

        per_token = F.cross_entropy(logits.transpose(1, 2), target, reduction="none",
                                    weight=self._class_weights(logits.device))
        if padding_mask is not None:
            valid = ~padding_mask
            loss = (per_token * valid).sum() / valid.sum().clamp_min(1)
        else:
            loss = per_token.mean()
        return PlannerOutput(loss, logits, noisy, target, timesteps)

    def posterior_logits(
        self, noisy_labels: torch.Tensor, x0_logits: torch.Tensor, step: int
    ) -> torch.Tensor:
        """Log ``p(y_{t-1} | y_t)`` for the uniform kernel, marginalising over y_0.

        For a uniform (doubly stochastic) transition matrix,

            q(y_t | y_{t-1}=k)   = alpha_t * [y_t == k] + (1 - alpha_t) / K
            q(y_{t-1}=k | y_0)   = abar_{t-1} * [k == y_0] + (1 - abar_{t-1}) / K

        so the posterior is proportional to their product.  Averaging it under
        the predicted distribution over ``y_0`` gives the reverse step actually
        used at sampling time.
        """
        classes = self.num_classes
        alpha = self.alphas[step]
        alpha_bar_prev = self.alpha_bars[step - 1] if step > 0 else torch.ones_like(alpha)

        # Likelihood term: depends on y_t only, identical for every y_0.
        likelihood = torch.full(
            noisy_labels.shape + (classes,),
            float((1.0 - alpha) / classes),
            device=noisy_labels.device,
            dtype=x0_logits.dtype,
        )
        likelihood.scatter_add_(
            -1,
            noisy_labels.unsqueeze(-1),
            torch.full_like(noisy_labels.unsqueeze(-1), float(alpha), dtype=x0_logits.dtype),
        )

        # Prior term, marginalised: sum_{y0} p(y0) * q(y_{t-1}=k | y0) reduces to
        # a mixture of the predicted distribution and the uniform distribution.
        x0_probs = x0_logits.softmax(dim=-1)
        prior = alpha_bar_prev * x0_probs + (1.0 - alpha_bar_prev) / classes

        posterior = likelihood * prior
        return torch.log(posterior.clamp_min(1e-20))

    @torch.no_grad()
    def sample(
        self,
        music_features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        deterministic: bool = False,
        global_music: Optional[torch.Tensor] = None,
        guidance_weight: float = 1.0,
        transition_logit_bias: float = 0.0,
    ) -> torch.Tensor:
        """Run the reverse chain.

        ``transition_logit_bias`` is added to class 0's x0 logit at every step.
        It is a *calibration*, not a model improvement: it does not make any
        class decision more correct, it decides how much of the vocabulary is
        admitted.  It is offered because it has leverage where temperature has
        none -- temperature scales every logit, so under ``x0`` it is nearly
        cancelled by ``posterior_logits``' likelihood term, and measured over 65
        clips T=1.0 -> 0.2 moved the transition share only 0.5263 -> 0.5178,
        while a bias of -2.0 moved it 0.5392 -> 0.3553 against a ground truth of
        0.3318 and left 93.2% of the non-transition frames on the same class.
        A bias changes the ordering; a temperature changes only the sharpness.

        The cost, measured on the same clips: the plan fragments (6.12 -> 9.15
        atomic segments against a ground truth of 5.33), because the classes it
        promotes are not temporally coherent on their own.  Composing it with
        the bar grid removes that cost -- see ``infer_plan``'s ``plan_bar_grid``.

        **Fit it on val, apply it on test.**  Choosing it to match the test
        split's own transition rate is tuning on test.

        ``guidance_weight`` is classifier-free guidance on the x0 logits and
        needs a planner built with ``cond_drop_prob > 0``; 1.0 is plain
        conditional sampling and is what every checkpoint before 2026-08-23
        reproduces.

        Note the explicit ``cond_drop_prob=0.0`` below.  ``forward`` defaults to
        the module's *training* drop rate, so a guided-capable planner sampled
        without it would silently drop its own condition on a quarter of the
        sequences at inference -- the failure would look like a bad model, not a
        bad call.
        """
        if guidance_weight != 1.0 and getattr(self.model, "null_music", None) is None:
            raise ValueError(
                "classifier-free guidance needs a planner trained with "
                "cond_drop_prob > 0; this one has no null condition")
        labels = torch.randint(
            self.num_classes, music_features.shape[:2], device=music_features.device
        )
        for step in reversed(range(self.num_steps)):
            t = torch.full((labels.shape[0],), step, device=labels.device, dtype=torch.long)
            if guidance_weight == 1.0:
                logits = self.model(labels, music_features, t, padding_mask,
                                    global_music=global_music, cond_drop_prob=0.0)
            else:
                logits = self.model.guided_forward(
                    labels, music_features, t, guidance_weight, padding_mask,
                    global_music=global_music)
            if transition_logit_bias:
                logits = logits.clone()
                logits[..., 0] = logits[..., 0] + transition_logit_bias
            logits = logits / temperature

            if self.parameterization == "x0":
                # The network predicts y_0; the chain advances through the
                # posterior.  Sampling the network output directly would jump
                # straight to a clean guess and discard the schedule.
                if step == 0:
                    step_logits = logits
                else:
                    step_logits = self.posterior_logits(labels, logits, step)
            else:
                step_logits = logits

            if deterministic:
                labels = step_logits.argmax(dim=-1)
            else:
                labels = torch.distributions.Categorical(logits=step_logits).sample()
        if padding_mask is not None:
            labels = labels.masked_fill(padding_mask, 0)
        return labels
