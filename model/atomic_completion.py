"""Transition-aware continuous diffusion for atomic dance completion."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model import DanceDecoder
from model.utils import extract, make_beta_schedule


class MotionGeometry(nn.Module):
    """Differentiable decode of the 151-D vector into world joints, in metres.

    WHY THIS EXISTS.  The completion trains on an unweighted MSE over 151
    normalized dimensions, in which the three root channels carry 0.117% of the
    budget and a rotation error costs the same whether it swings a wrist or a
    thigh.  EDGE -- the denoiser this module wraps, by the paper's own
    statement -- does not train that way: its published objective adds joint
    positions through FK, their velocities, and a foot-contact consistency
    term.  This repository kept only the reconstruction and boundary terms, and
    the deliberate deviation is recorded at ``training_step``.  The measured
    bill for it, 2026-08-30: limb lag-0 synchrony 0.91-1.00 against the ground
    truth's 0.642 across NINE same-config checkpoints and every sampler knob;
    torso twist rate 29-34 deg/s against 49.3; sustained ground contact 0.03-
    0.05 of frames against 0.232.  Those are all *geometric* quantities, and
    the objective never sees geometry.

    The normalizer travels HERE, as buffers, for the same reason the music
    statistics do (see ``MusicNormalization``): a geometric loss computed
    against the wrong normalizer is not wrong loudly -- it is a plausible
    gradient toward a subtly wrong skeleton -- and a buffer cannot be dropped
    silently the way a path in ``args`` can.

    The FK mirrors ``vis.SMPLSkeleton.forward`` exactly but works on rotation
    MATRICES straight from the 6-D representation: matrix -> axis-angle ->
    quaternion would round-trip through ``acos`` whose gradient blows up at the
    identity rotation, which real dance visits constantly.
    """

    CONTACTS = 4
    ROOT = 3
    FEET = (7, 8, 10, 11)          # ankles and toes, the four contact channels' owners

    def __init__(self, normalizer):
        super().__init__()
        from vis import smpl_offsets, smpl_parents
        data_min = torch.as_tensor(normalizer["data_min"], dtype=torch.float32).reshape(-1)
        data_max = torch.as_tensor(normalizer["data_max"], dtype=torch.float32).reshape(-1)
        if data_min.numel() != 151:
            raise ValueError("normalizer must describe 151 dims, got {}".format(data_min.numel()))
        span = data_max - data_min
        self.register_buffer("data_min", data_min)
        self.register_buffer("safe_range", torch.where(span == 0, torch.ones_like(span), span))
        self.register_buffer("offsets", torch.tensor(smpl_offsets, dtype=torch.float32))
        self.parents = list(smpl_parents)

    def joints(self, motion):
        """[batch, frames, 151] normalized -> [batch, frames, 24, 3] world metres."""
        from dataset.quaternion import rotation_6d_to_matrix

        raw = (motion + 1.0) * self.safe_range / 2.0 + self.data_min
        root = raw[..., self.CONTACTS:self.CONTACTS + self.ROOT]
        six = raw[..., self.CONTACTS + self.ROOT:].reshape(*raw.shape[:-1], 24, 6)
        local = rotation_6d_to_matrix(six)
        world_rot = [local[..., 0, :, :]]
        world_pos = [root]
        for joint in range(1, len(self.parents)):
            parent = self.parents[joint]
            world_pos.append(
                torch.einsum("...ij,j->...i", world_rot[parent], self.offsets[joint])
                + world_pos[parent])
            world_rot.append(world_rot[parent] @ local[..., joint, :, :])
        return torch.stack(world_pos, dim=-2)

    def contacts(self, motion):
        """The four contact channels, unnormalized back to their 0..1 scale."""
        raw = (motion + 1.0) * self.safe_range / 2.0 + self.data_min
        return raw[..., :self.CONTACTS]


class AtomicCompletionDecoder(nn.Module):
    """EDGE denoiser augmented with retrieved motion draft M0 and noise mask w."""

    def __init__(
        self,
        motion_dim: int,
        seq_len: int,
        music_dim: int,
        latent_dim: int = 512,
        ff_size: int = 1024,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_classes: Optional[int] = None,
        label_dim: int = 64,
        music_stats=None,
        music_phase_features: bool = False,
        draft_guidance: bool = False,
    ) -> None:
        super().__init__()
        self.motion_dim = motion_dim
        # The plan reaches this model only through the retrieved draft, and
        # measured on 24 held-out wild clips it barely arrives: on the frames
        # that carry a prototype, the generated joint rotations sit 4-5% closer
        # to that prototype than to a shuffled ordering of the same prototype's
        # frames, and the root does not follow it at all.  Handing the model a
        # clean draft (``--draft-noise-ratio 0``) moves that to 5.1%, so the
        # training noise is not what is blocking it -- the 151-D soft hint is
        # simply a thin channel for "which atomic movement happens here".
        #
        # ``num_classes`` opens a second, discrete channel: the label itself,
        # embedded per frame.  Off by default because every checkpoint trained
        # before 2026-08-17 has no such embedding and must keep loading.
        self.label_embedding = (
            nn.Embedding(int(num_classes) + 1, label_dim) if num_classes else None
        )
        adapter_in = motion_dim * 2 + 1 + (label_dim if self.label_embedding is not None else 0)
        # Carried as a buffer so the checkpoint cannot be run without the
        # normalization it was trained with -- see MusicNormalization in
        # model/atomic_planner.py for why a path in ``args`` was rejected.
        from model.atomic_planner import MusicNormalization, MusicPhaseFeatures
        self.music_normalization = (MusicNormalization(music_stats)
                                    if music_stats is not None else None)
        # Derived rhythm channels, shared with the planner -- the module and
        # its measured WHY live at MusicPhaseFeatures in model/atomic_planner.py.
        # ``music_dim`` stays the 35-D release width; the widening happens
        # below, on the way into DanceDecoder's cond_projection, so no release
        # array is rebuilt.  Off by default because every completion checkpoint
        # trained before this flag existed has the 35-wide projection and no
        # marker buffer, and must keep loading.
        self.music_phase_features = (MusicPhaseFeatures()
                                     if music_phase_features else None)
        # THE DRAFT HAS NEVER BEEN GUIDABLE, and that is structural rather than
        # a matter of degree.  ``guided_forward`` below builds its difference
        # from two calls that differ only in ``cond_drop_prob``, and
        # ``cond_drop_prob`` reaches exactly one thing -- ``cond_projection`` in
        # model/model.py, i.e. the MUSIC.  The draft is fused into ``x`` by
        # ``input_adapter`` before the denoiser is entered, so it is present and
        # identical in both terms and cancels out of the difference exactly.
        # Classifier-free guidance at weight 2.0 therefore amplifies the music
        # by 2.0 and the draft by 1.0 -- which is to say, not at all.
        #
        # Measured consequence (2026-09-01, 8 clips, joint positions in metres,
        # root-relative, one window, everything else byte-identical): re-rolling
        # only the NOISE moves the output 0.4030 m; replacing the entire draft
        # with another clip's moves it 0.2282 m (0.57x); replacing the entire
        # music moves it 0.2393 m (0.59x).  Nothing-changed reads 0.0000, so the
        # probe is trustworthy.  The noise outweighs every conditioning channel
        # roughly two to one, which is why a ground-truth plan does not rescue
        # the dance: the model is not listening to the plan.
        #
        # ``null_draft`` is the learned "no draft" token that makes the draft
        # droppable, and therefore guidable.  A LEARNED null rather than zeros:
        # the draft lives in the normalizer's [-1, 1] space, where the zero
        # vector is the middle of the training range -- a perfectly meaningful
        # pose.  Dropping to zeros would teach the model that "no plan" means
        # "the average pose", which is a condition, not the absence of one.
        #
        # It is a PARAMETER, so it lands in the state_dict, so a checkpoint
        # trained with draft guidance refuses to load into a model built without
        # it and vice versa -- the same contract MusicNormalization and
        # MotionGeometry use, and for the same reason: this repository has three
        # recorded cases of an args-only flag silently becoming a no-op.
        self.null_draft = (
            nn.Parameter(torch.randn(1, 1, motion_dim) * motion_dim ** -0.5)
            if draft_guidance else None
        )
        self.input_adapter = nn.Sequential(
            nn.Linear(adapter_in, motion_dim),
            nn.SiLU(),
            nn.Linear(motion_dim, motion_dim),
        )
        self.denoiser = DanceDecoder(
            nfeats=motion_dim,
            seq_len=seq_len,
            latent_dim=latent_dim,
            ff_size=ff_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            cond_feature_dim=music_dim + (MusicPhaseFeatures.EXTRA_CHANNELS
                                          if music_phase_features else 0),
            activation=F.gelu,
        )

    def forward(
        self,
        noisy_motion: torch.Tensor,
        music_features: torch.Tensor,
        timesteps: torch.Tensor,
        draft_motion: torch.Tensor,
        noise_mask: torch.Tensor,
        cond_drop_prob: float = 0.0,
        labels: Optional[torch.Tensor] = None,
        draft_drop_prob: float = 0.0,
    ) -> torch.Tensor:
        if noise_mask.shape[-1] != 1:
            raise ValueError("noise_mask must have shape [batch, frames, 1]")
        if draft_drop_prob and self.null_draft is None:
            raise ValueError(
                "this completion model was built without draft guidance, so the "
                "draft cannot be dropped; rebuild it with draft_guidance=True "
                "(--draft-drop-prob) or ask for draft_drop_prob=0")
        if self.null_draft is not None and draft_drop_prob:
            # Per SAMPLE, not per frame: dropping frames would teach "the plan
            # is sometimes missing here", which is a different (and easier)
            # condition than "there is no plan at all", and only the second one
            # gives guidance a term to amplify.
            keep = torch.rand(draft_motion.shape[0], 1, 1,
                              device=draft_motion.device) >= draft_drop_prob
            draft_motion = torch.where(keep, draft_motion,
                                       self.null_draft.to(draft_motion.dtype))
            # The mask travels with the draft: it says which frames the draft
            # is trusted on, and a null draft is trusted nowhere.  Leaving it
            # behind would hand the "unconditional" branch a map of the plan's
            # own segment boundaries, which is most of what the draft carries.
            noise_mask = torch.where(keep, noise_mask, torch.zeros_like(noise_mask))
        # Order matters: phase is derived from the RAW beat one-hot (channel
        # 34), which z-scoring rescales out of its > 0.5 convention -- so
        # derive first, normalize the original 35 second, concatenate last.
        # The derived channels are unit-scale by construction and are never
        # themselves z-scored.
        derived = (self.music_phase_features(music_features)
                   if self.music_phase_features is not None else None)
        if self.music_normalization is not None:
            music_features = self.music_normalization(music_features)
        if derived is not None:
            music_features = torch.cat((music_features, derived), dim=-1)
        parts = [noisy_motion, draft_motion, noise_mask]
        # Both directions are errors rather than defaults.  A model built with
        # a label channel and then run without one would silently condition on
        # whatever the adapter makes of a missing block; a model built without
        # one that is handed labels would silently ignore the plan -- which is
        # the failure this channel exists to end.
        if self.label_embedding is not None:
            if labels is None:
                raise ValueError("this completion model was built with a label channel, "
                                 "so labels are required")
            if labels.shape != noisy_motion.shape[:2]:
                raise ValueError("labels must have shape [batch, frames]")
            parts.append(self.label_embedding(labels.long()))
        elif labels is not None:
            raise ValueError("this completion model has no label channel, so labels "
                             "cannot be conditioned on")
        fused = self.input_adapter(torch.cat(parts, dim=-1))
        return self.denoiser(fused, music_features, timesteps, cond_drop_prob=cond_drop_prob)

    def guided_forward(
        self,
        noisy_motion: torch.Tensor,
        music_features: torch.Tensor,
        timesteps: torch.Tensor,
        draft_motion: torch.Tensor,
        noise_mask: torch.Tensor,
        guidance_weight: float,
        labels: Optional[torch.Tensor] = None,
        draft_guidance_weight: Optional[float] = None,
    ) -> torch.Tensor:
        """Two-term guidance by default; three-term when the draft is guidable.

        The default path is byte-for-byte what it was, so every checkpoint
        trained before 2026-09-01 keeps sampling as the model it was.

        With ``draft_guidance_weight`` the composition is the standard
        multi-conditional one, and the ORDER of the two differences is the whole
        point: ``music`` is measured against nothing, and ``draft`` is measured
        against music-alone, so the draft term carries only what the plan adds
        on top of the music rather than double-counting whatever the two agree
        about.  Written out,

            e = e(-, -) + w_m * (e(m, -) - e(-, -)) + w_d * (e(m, d) - e(m, -))

        which at ``w_d = 1`` reduces exactly to the old two-term form with the
        draft always present -- i.e. the shipped behaviour is the w_d = 1 point
        of this family, which is precisely why the draft has never been
        amplified.  Costs three forwards per step instead of two.
        """
        if draft_guidance_weight is None:
            unconditional = self.forward(
                noisy_motion, music_features, timesteps, draft_motion, noise_mask,
                cond_drop_prob=1.0, labels=labels,
            )
            conditional = self.forward(
                noisy_motion, music_features, timesteps, draft_motion, noise_mask,
                cond_drop_prob=0.0, labels=labels,
            )
            return unconditional + guidance_weight * (conditional - unconditional)
        if self.null_draft is None:
            raise ValueError(
                "draft guidance needs a checkpoint trained with --draft-drop-prob; "
                "this one has no null draft to measure the draft against")
        neither = self.forward(
            noisy_motion, music_features, timesteps, draft_motion, noise_mask,
            cond_drop_prob=1.0, labels=labels, draft_drop_prob=1.0,
        )
        music_only = self.forward(
            noisy_motion, music_features, timesteps, draft_motion, noise_mask,
            cond_drop_prob=0.0, labels=labels, draft_drop_prob=1.0,
        )
        both = self.forward(
            noisy_motion, music_features, timesteps, draft_motion, noise_mask,
            cond_drop_prob=0.0, labels=labels, draft_drop_prob=0.0,
        )
        return (neither
                + guidance_weight * (music_only - neither)
                + float(draft_guidance_weight) * (both - music_only))


@dataclass
class CompletionLoss:
    total: torch.Tensor
    denoising: torch.Tensor
    transition: torch.Tensor
    prediction: torch.Tensor
    velocity: torch.Tensor = None


class AtomicCompletionDiffusion(nn.Module):
    """Clean-motion-predicting DDPM corresponding to Eq. (4) in the paper."""

    def __init__(
        self,
        model: AtomicCompletionDecoder,
        num_steps: int = 1000,
        schedule: str = "cosine",
        transition_weight: float = 1.0,
        velocity_weight: float = 0.0,
        velocity_skip_contact: bool = False,
        cond_drop_prob: float = 0.25,
        guidance_weight: float = 2.0,
        fk_weight: float = 0.0,
        fk_velocity_weight: float = 0.0,
        contact_weight: float = 0.0,
        energy_match_weight: float = 0.0,
        draft_drop_prob: float = 0.0,
        normalizer=None,
    ) -> None:
        super().__init__()
        self.model = model
        # The EDGE auxiliary objective, off by default so every earlier
        # checkpoint rebuilds as the model it was.  Asking for any of the three
        # without the normalizer is refused rather than defaulted: geometry in
        # normalized units is a different (and wrong) objective that would
        # train without complaint.
        self.fk_weight = float(fk_weight)
        self.fk_velocity_weight = float(fk_velocity_weight)
        self.contact_weight = float(contact_weight)
        # The one term here that is NOT mean-seeking.  Waves 2-4 (2026-08-31,
        # docs/DANCE_QUALITY_DEFECTS.md 12.5-12.6) measured every FK-space MSE
        # damping motion energy -- fk 1.0 took energy/GT from 0.79 to 0.56,
        # fk 0.25-0.5 with contact to 0.49-0.65 -- because the conditional
        # distribution over motion is multimodal in amplitude and an MSE pulls
        # to its mean.  This term penalises |mean joint speed(pred) - mean
        # joint speed(target)| per window, a scalar an under-driven prediction
        # makes WORSE by damping, so it pushes the opposite way.  The target's
        # own speed is the supervision, so it also carries "this song is
        # energetic" into training -- the e_corr column (0.21 against ground
        # truth) is the number it exists to move.
        self.energy_match_weight = float(energy_match_weight)
        wants_geometry = (self.fk_weight or self.fk_velocity_weight
                          or self.contact_weight or self.energy_match_weight)
        if wants_geometry and normalizer is None:
            raise ValueError(
                "fk/contact losses need the release normalizer to decode the "
                "151-D vector into metres; pass --normalizer-into-loss or drop "
                "the geometric weights")
        self.geometry = MotionGeometry(normalizer) if wants_geometry else None
        self.num_steps = num_steps
        self.transition_weight = transition_weight
        # EDGE -- the denoiser this class wraps and names in its own docstring --
        # trains on reconstruction *plus* a velocity term plus foot contact.
        # This repository kept only reconstruction and a boundary term, and the
        # cost of the missing velocity term was measured on 2026-08-23: fed the
        # ground truth noised to t=0, i.e. asked to reproduce an essentially
        # clean input, the completion returns a root trajectory whose path
        # length is **397%** of that input's (5.386 m against 1.356 m over a
        # 5 s window, median of 64 test windows).  The full reverse chain takes
        # it to 541%, so most of the damage is the model, not the sampler.
        #
        # Root position is 3 of 151 dimensions, so per-frame root error is ~2%
        # of an unweighted MSE and nothing in the objective notices that the
        # error is *uncorrelated between neighbouring frames* -- which is what
        # a viewer sees as a dancer vibrating in place.  A first-difference term
        # penalises exactly that correlation structure and nothing else.
        #
        # Default 0.0, so every checkpoint trained before this keeps rebuilding
        # and evaluating as the model it was.
        self.velocity_weight = float(velocity_weight)
        # See velocity_loss: the four contact channels take 86.03% of this
        # term's budget and are the one place a discontinuity is correct.
        self.velocity_skip_contact = bool(velocity_skip_contact)
        self.cond_drop_prob = cond_drop_prob
        # Conditioning dropout for the DRAFT, the mirror of cond_drop_prob for
        # the music.  0.0 keeps every earlier checkpoint's objective exactly.
        # Refused unless the decoder actually has a null draft to drop to, so a
        # run cannot ask for this and silently train the old objective -- the
        # failure mode this repository has recorded three times.
        self.draft_drop_prob = float(draft_drop_prob)
        if self.draft_drop_prob and getattr(model, "null_draft", None) is None:
            raise ValueError(
                "--draft-drop-prob needs a decoder built with draft_guidance=True; "
                "without a null draft the dropped samples would train on the "
                "zero vector, which in normalized units is a real pose")
        self.guidance_weight = guidance_weight

        betas = torch.as_tensor(make_beta_schedule(schedule, num_steps), dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_previous = torch.cat((torch.ones(1), alpha_bars[:-1]))
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", alpha_bars.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bars", (1.0 - alpha_bars).sqrt())
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alpha_bars_previous) / (1.0 - alpha_bars),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * alpha_bars_previous.sqrt() / (1.0 - alpha_bars),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alpha_bars_previous) * alphas.sqrt() / (1.0 - alpha_bars),
        )

    def q_sample(
        self, clean_motion: torch.Tensor, timesteps: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        noise = torch.randn_like(clean_motion) if noise is None else noise
        return (
            extract(self.sqrt_alpha_bars, timesteps, clean_motion.shape) * clean_motion
            + extract(self.sqrt_one_minus_alpha_bars, timesteps, clean_motion.shape) * noise
        )

    @staticmethod
    def perturb_draft(draft_motion: torch.Tensor, noise_mask: torch.Tensor) -> torch.Tensor:
        """Apply appropriately scaled noise only to retrieved atomic frames."""
        return draft_motion + torch.randn_like(draft_motion) * noise_mask

    def forward(self, *args, **kwargs) -> "CompletionLoss":
        """Alias for ``training_step``, so DistributedDataParallel can wrap this.

        See ``UniformD3PM.forward``: calling ``training_step`` through a DDP
        wrapper's ``.module`` skips gradient synchronisation silently, and a run
        that trains N unsynchronised copies looks exactly like one that works.
        """
        return self.training_step(*args, **kwargs)

    @staticmethod
    def velocity_loss(prediction: torch.Tensor, target: torch.Tensor,
                      skip_contact: bool = False) -> torch.Tensor:
        """MSE between the two sequences' first differences.

        The docstring this replaces said the term covers all 151 dimensions
        because "weighting three dimensions by hand would be a choice this
        repository has no measurement to justify".  There is now a measurement,
        and it says the unweighted term is itself a weighting -- a bad one.

        BUDGET CENSUS (training set, differences INSIDE a window, n ~ 990k
        frames).  Share of the squared first-difference this term is summing:

            4 foot-contact channels   86.03%
            23 joints (138 dims)      13.35%
            global orient (6 dims)     0.60%
            root translation (3 dims)  0.02%

        The contact channels are strictly binary -- 100% of their values sit at
        +-1, none in between -- and they flip on 11.61% of frame pairs, while
        every other dimension moves |delta| > 1 on 0.0003% of them.  So with
        ``--velocity-weight 4.0``, which is what the shipped checkpoint was
        trained with, **86% of the term's budget is spent making four binary
        flags flip smoothly, which is precisely where they must not** -- a foot
        lands or it does not -- while the channel that carries which way the
        body faces gets 0.60%.

        ``skip_contact`` drops those four dimensions.  What is left redivides as
        joints 95.57% / global orient 4.31% / root 0.12%.  It is off by default
        so every earlier checkpoint reproduces from its own arguments.

        WHAT MAY AND MAY NOT BE CLAIMED.  The census is reproducible and is a
        statement about the objective.  It is NOT a claim that removing the
        contact channels fixes the foot-slide or the missing torso twist; that
        needs a training run, judged on criteria fixed before it starts and
        paired with a "still has to be a dance" gate -- section 13.3 is what
        happens without one.
        """
        difference = (prediction[:, 1:] - prediction[:, :-1],
                      target[:, 1:] - target[:, :-1])
        if skip_contact:
            difference = tuple(d[..., MotionGeometry.CONTACTS:] for d in difference)
        return F.mse_loss(*difference)

    @staticmethod
    def transition_loss(prediction: torch.Tensor, boundaries: Optional[torch.Tensor]) -> torch.Tensor:
        if boundaries is None:
            return prediction.new_zeros(())
        if boundaries.shape != prediction.shape[:2]:
            raise ValueError("boundaries must have shape [batch, frames]")
        velocity = (prediction[:, 1:] - prediction[:, :-1]).abs().mean(dim=-1)
        selected = boundaries[:, 1:].to(dtype=velocity.dtype)
        return (velocity * selected).sum() / selected.sum().clamp_min(1.0)

    def training_step(
        self,
        clean_motion: torch.Tensor,
        music_features: torch.Tensor,
        draft_motion: torch.Tensor,
        noise_mask: torch.Tensor,
        boundaries: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> CompletionLoss:
        batch = clean_motion.shape[0]
        if timesteps is None:
            timesteps = torch.randint(self.num_steps, (batch,), device=clean_motion.device)
        noisy_motion = self.q_sample(clean_motion, timesteps)
        noisy_draft = self.perturb_draft(draft_motion, noise_mask)
        prediction = self.model(
            noisy_motion,
            music_features,
            timesteps,
            noisy_draft,
            noise_mask,
            cond_drop_prob=self.cond_drop_prob,
            labels=labels,
            draft_drop_prob=self.draft_drop_prob,
        )
        denoising = F.mse_loss(prediction, clean_motion)
        transition = self.transition_loss(prediction, boundaries)
        velocity = (self.velocity_loss(prediction, clean_motion,
                                       skip_contact=self.velocity_skip_contact)
                    if self.velocity_weight else prediction.new_zeros(()))
        geometric = prediction.new_zeros(())
        if self.geometry is not None:
            predicted_joints = self.geometry.joints(prediction)
            with torch.no_grad():
                target_joints = self.geometry.joints(clean_motion)
            if self.fk_weight:
                geometric = geometric + self.fk_weight * F.mse_loss(
                    predicted_joints, target_joints)
            if self.fk_velocity_weight:
                geometric = geometric + self.fk_velocity_weight * F.mse_loss(
                    predicted_joints[:, 1:] - predicted_joints[:, :-1],
                    target_joints[:, 1:] - target_joints[:, :-1])
            if self.energy_match_weight:
                relative_p = predicted_joints - predicted_joints[..., :1, :]
                relative_t = target_joints - target_joints[..., :1, :]
                speed_p = (relative_p[:, 1:] - relative_p[:, :-1]).norm(dim=-1).mean(dim=(1, 2))
                speed_t = (relative_t[:, 1:] - relative_t[:, :-1]).norm(dim=-1).mean(dim=(1, 2))
                geometric = geometric + self.energy_match_weight * (
                    (speed_p - speed_t).abs() / speed_t.clamp_min(1e-4)).mean()
            if self.contact_weight:
                # EDGE's consistency term: the model's OWN predicted contacts
                # gate its own predicted foot velocities, so "foot planted" and
                # "foot moving" cannot be asserted about the same frame.  The
                # gate is detached: the cheap gradient escape is to predict
                # no contact anywhere, which silences the term without planting
                # a single foot.
                feet = predicted_joints[:, :, list(self.geometry.FEET), :]
                foot_velocity = feet[:, 1:] - feet[:, :-1]
                gate = self.geometry.contacts(prediction)[:, 1:].detach().clamp(0.0, 1.0)
                geometric = geometric + self.contact_weight * (
                    foot_velocity.pow(2).sum(-1) * gate).mean()
        return CompletionLoss(
            denoising + self.transition_weight * transition
            + self.velocity_weight * velocity + geometric,
            denoising,
            transition,
            prediction,
            velocity,
        )

    @torch.no_grad()
    def sample(
        self,
        music_features: torch.Tensor,
        draft_motion: torch.Tensor,
        noise_mask: torch.Tensor,
        guidance_weight: Optional[float] = None,
        labels: Optional[torch.Tensor] = None,
        start_step: Optional[int] = None,
        reproject_every: Optional[int] = None,
        sample_steps: Optional[int] = None,
        draft_guidance_weight: Optional[float] = None,
        keep_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``start_step`` begins the reverse chain from a noised draft.

        The default -- pure noise at the last step -- is what every artifact
        before 2026-08-30 used and stays the default.  The reason the option
        exists is a measurement: the retrieval draft carries the limb structure
        of a real dancer (cross-correlation peaks at lag 0 on 0.000 of limb
        pairs after a 15-frame seam blend, ground truth 0.500), and the
        generated output does not (1.000 before the blend, 0.833 after).  So
        the reverse chain is *re-imposing* the synchrony the draft no longer
        has, and how much it re-imposes must be a function of how many steps it
        runs.  Starting part-way down turns that into something measurable, and
        possibly into a fix.

        It is not free: a draft-initialised chain is closer to retrieval and
        further from generation, so any use of it has to report how close the
        output sits to its own draft, or it has replaced a model with a
        smoother.

        ``reproject_every`` puts the iterate back on the draft's manifold every
        N steps -- ``q_sample(draft, t)`` in place of the running state.  It is
        the intervention that follows from the measured mechanism, and it has
        its own control.  **Measured 2026-08-30, 8 clips, arm span in metres
        against a ground truth of 0.630 and a draft of 0.663:**

        ================  ==========  =====================
        reprojection      arm span    distance to own draft
        ================  ==========  =====================
        none (shipped)         0.872                 0.2977
        every 50 steps         0.674                 0.2516
        every 25 steps         0.679                 0.2515
        every 10 steps         0.679                 0.2512
        ================  ==========  =====================

        The mechanism it acts on: the denoiser is unbiased **on** the data
        manifold -- fed real motion noised to any step it returns a span of
        0.64-0.75 against an input of 0.669 -- and biased **off** it, and its own
        iterates leave the manifold and never return.  Dumping x0 along the
        shipped chain shows the drift accumulating monotonically with no single
        culprit step: 0.750 at step 99, 0.804 at 50, 0.874 at 10, 0.923 at 0.

        The novelty column is why this is not simply retrieval with extra steps:
        the output stays 0.25 m from the draft it was pulled towards, against
        0.30 m without any reprojection -- a 15% cost for removing a 38% bias.
        One correction in the whole chain is enough; 10 and 50 read the same.
        """
        weight = self.guidance_weight if guidance_weight is None else guidance_weight
        conditioned_draft = self.perturb_draft(draft_motion, noise_mask)
        if start_step is None or start_step >= self.num_steps:
            first = self.num_steps - 1
            motion = torch.randn_like(draft_motion)
        else:
            first = max(0, int(start_step))
            timesteps = torch.full((draft_motion.shape[0],), first,
                                   dtype=torch.long, device=draft_motion.device)
            motion = self.q_sample(draft_motion, timesteps)
        chain = list(reversed(range(first + 1)))
        if sample_steps and 0 < int(sample_steps) < len(chain):
            # Evenly respaced reverse chain, both endpoints kept.  The model
            # predicts x0, so the posterior for a respaced pair (t, t_prev) is
            # the ordinary one with alpha = abar_t / abar_prev: no retrain, and
            # the trained schedule itself is untouched.
            #
            # WHY THIS SWITCH EXISTS.  Every other sampler switch measured on
            # 2026-08-30 -- seam blend, bar beats, start_step, reprojection --
            # turned out to move ONE quantity: how hard the output is pulled
            # toward the retrieval draft.  Twelve arms sit on one line,
            # ``lag0 = -0.409 * arm-span + 1.302``, r = -0.873, P = 0.00021,
            # residual sd 0.0161, and the ground truth sits 24 residual sd off
            # it.  No combination of those switches can therefore reach the
            # ground truth; they only choose a point on the line.
            #
            # This one is the falsification test for that reading.  The measured
            # mechanism behind the arm-span defect is drift that ACCUMULATES
            # over the reverse steps -- x0's arm span climbs monotonically 0.750
            # at step 99 to 0.923 at step 0, while the same denoiser fed real
            # motion noised to any step returns 0.64-0.75 against an input of
            # 0.669.  Cutting the step count cuts the accumulation WITHOUT
            # pulling anything toward the draft, so it is the one knob that has
            # a reason to leave the line.  If it lands on the line anyway, the
            # line is the whole story of what post-hoc sampling can buy here.
            index = torch.linspace(0, len(chain) - 1, int(sample_steps)).round().long().tolist()
            chain = [chain[i] for i in sorted(set(index))]
        for position, step in enumerate(chain):
            previous = chain[position + 1] if position + 1 < len(chain) else -1
            # ``0 < step``: the last correction has to leave the chain room to
            # denoise afterwards.  Including step 0 replaces the iterate with
            # ``q_sample(draft, 0)`` -- the draft itself, to within one step's
            # noise -- and one model step later the output IS the draft.  It
            # read as a spectacular result (arm span 0.665 against a ground
            # truth of 0.663) and was caught by a second arm: reprojecting
            # every 25 steps and every 50 steps produced outputs 0.0103 m
            # apart, which cannot happen if the intervening steps matter.
            if (reproject_every and 0 < step < first
                    and step % int(reproject_every) == 0):
                timesteps = torch.full((motion.shape[0],), step,
                                       dtype=torch.long, device=motion.device)
                motion = self.q_sample(draft_motion, timesteps)
            timesteps = torch.full(
                (motion.shape[0],), step, dtype=torch.long, device=motion.device
            )
            clean = self.model.guided_forward(
                motion,
                music_features,
                timesteps,
                conditioned_draft,
                noise_mask,
                weight,
                labels=labels,
                draft_guidance_weight=draft_guidance_weight,
            ).clamp(-1.0, 1.0)
            if previous == step - 1:
                mean = (
                    extract(self.posterior_mean_coef1, timesteps, motion.shape) * clean
                    + extract(self.posterior_mean_coef2, timesteps, motion.shape) * motion
                )
                variance = extract(self.posterior_variance, timesteps, motion.shape)
            else:
                bar_t = self.alpha_bars[step]
                bar_prev = (self.alpha_bars[previous] if previous >= 0
                            else self.alpha_bars.new_ones(()))
                alpha = bar_t / bar_prev
                beta = (1.0 - alpha).clamp_min(1e-20)
                denom = (1.0 - bar_t).clamp_min(1e-20)
                mean = ((beta * bar_prev.sqrt() / denom) * clean
                        + (alpha.sqrt() * (1.0 - bar_prev) / denom) * motion)
                variance = beta * (1.0 - bar_prev) / denom
            if previous >= 0:
                motion = mean + variance.sqrt() * torch.randn_like(motion)
            else:
                motion = mean
            if keep_mask is not None:
                # INPAINTING.  Overwrite the kept frames with the draft at this
                # noise level, EVERY step, leaving the rest of the sequence free.
                #
                # WHY THIS AND NOT ``reproject_every``.  That one resets the
                # WHOLE iterate, seams included, every N steps, and measured
                # 2026-09-05 on the T line even N=10 moved the output only 8.6%
                # closer to its own draft: the next model step maps it straight
                # back, because the prediction leans on the music and the
                # learned prior far more than on the conditioning draft.  Over
                # four levers -- guidance 1.0-4.5, --draft-timing-align
                # (retrained 600 epochs), --completion-start-step 100->30, and
                # reprojection -- nothing moved transmission above 17%.  Three
                # draft-side fixes that worked on the draft therefore vanished
                # in the final dance.  A hard constraint is the remaining
                # option: the kept frames stop being a suggestion.
                #
                # WHY IT IS NOT THE TRAP THIS FILE ALREADY RECORDS.  Reprojecting
                # at step 0 made the output BE the draft and read as a
                # spectacular result while the model did nothing.  Here the
                # seams are deliberately NOT kept, so the model still has to
                # produce every transition; tests assert the seam frames differ
                # from the draft while the kept frames match it.
                #
                # SOFT, NOT BOOLEAN.  The first version used torch.where on a
                # 0/1 mask, which switches from "this frame IS the draft" to
                # "this frame is generated" in one frame.  Measured on the T
                # line 2026-09-06, jerk profiled against distance from a plan
                # boundary: ground truth is flat at 0.21-0.27 everywhere, and
                # the boolean version spiked to 9.65 at -10 frames and 5.06/5.32
                # at +6/+8 -- the EDGES of the keep window -- with a worst-case
                # 16.85 against ground truth's 0.90.  Between -6 and +4 it was
                # clean, which is why the seam column (sampled at +-2) read
                # BETTER than ground truth while the operator watched the video
                # and reported "motion 之间有跳变,不流畅".
                #
                # So keep_mask is a WEIGHT in [0, 1], and the caller ramps it.
                # The ramp must itself be smooth: this file already records that
                # a linear cross-fade "removes the step in the value and leaves
                # one in the first derivative", and 2026-09-05 measured a
                # triangular ENVELOPE doing the same one level up.
                timesteps = torch.full((motion.shape[0],), max(previous, 0),
                                       dtype=torch.long, device=motion.device)
                anchored = (draft_motion if previous < 0
                            else self.q_sample(draft_motion, timesteps))
                weight = keep_mask.to(dtype=motion.dtype)
                motion = weight * anchored + (1.0 - weight) * motion
        return motion
