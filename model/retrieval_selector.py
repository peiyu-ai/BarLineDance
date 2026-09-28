"""Score a retrieval candidate against the segment it has to follow.

WHY THIS EXISTS.  The shipped rule is one line (``infer_atomic.py``,
``retrieval_rule == "duration"``)::

    chosen = min(candidates, key=lambda item: abs((item[2] - item[1]) - target_length))

It is deterministic, cached, and it looks at exactly one number: how close the
candidate's length is to the span the plan asked for.  It cannot see the
segment it will be pasted after, the segment that comes next, the music under
it, or which foot the dancer is standing on.  The paper does the same
(``WILD_ATOMIC_PIPELINE_PLAN.md`` M5: "时长匹配检索 + 缩放"), so this is not a
regression from it -- it is the part of M5 nobody has replaced yet.

WHAT THE MEASUREMENTS SAY THE SELECTION HAS TO FIX.
``docs/DANCE_QUALITY_DEFECTS.md`` section 15.3, masking +-4 frames around plan
boundaries: the retrieval draft's ``settle`` goes from -0.2356 (0/87 clips) to
**-0.0011 (43/87)**.  The library's content is already roughly on the beat;
essentially the whole phase inversion lives in the seams.  And section 14.2:
92.9% of the draft's over-threshold facing spins sit on a plan boundary, 11.84x
their share of frames, while AWAY from boundaries the draft turns 57.8 deg/s
against ground truth's 62.8 -- the content turns LESS than a real dancer and
every violent turn is manufactured at the paste.  So the thing to choose for is
**the join**, not the content.

THE RULE THIS REPLACES THAT ALREADY FAILED, AND THE THREE WAYS THIS DIFFERS.
``--retrieval-rule phase`` (section 15.8) ranked candidates by start-phase
distance.  Its plumbing worked -- median phase error 0.2381 -> 0.0000, share
within +-0.05 of the beat 4.9% -> 67.1% -- and the OUTPUT got worse: settle
-0.0769 -> -0.1111, e_corr 0.320 -> 0.220.  The diagnosis was that a hard
ranking narrows the candidate pool and squeezes out the variety the draft
exists to supply.  Therefore:

  1. what is scored is the JOIN (pose, velocity, contact continuity across the
     seam) and the fit to what comes next, not a property of the candidate on
     its own;
  2. the output is a TEMPERATURE-SAMPLED draw from the top ``k``, never an
     argmax -- ``select`` returns a distribution, and the caller samples it;
  3. the candidate pool handed to this scorer is the duration-slack band
     ``max(2, 0.15 * target_length)`` that ``tempo``, ``phase`` and
     ``--draft-recurrence-variety`` already draw from, so the pool is not
     narrowed relative to shipped behaviour by this file.

THE TRIVIAL SOLUTION, AND WHY THE CANONICALISATION IS LOAD-BEARING.
Training pairs are free: in the corpus, the true continuation of segment i is
segment i+1 of the same recording.  But the positive then continues its
predecessor EXACTLY -- same room, same facing, same root -- while every
negative was cut from a different upload and arrives with an arbitrary world
placement.  A scorer trained on raw frames learns "root and facing already
match", which no candidate satisfies at inference and which is therefore a
scorer that has learned nothing.

``align_to`` removes it: every candidate, positive included, is rotated about
world z so its first frame faces where the previous segment ended, and
translated so its first frame's ground position continues from there.  That is
byte-for-byte the transform ``build_draft`` can already apply
(``--draft-facing-continuity`` + ``--draft-root-continuity xy``), so after it
the positive is separable only by JOINT-ANGLE continuity, velocity continuity,
contact continuity and internal timing -- the real signal.
``tests/test_retrieval_selector.py`` holds this with a positive control: with
the seam block zeroed, held-out recall must fall back to the duration
baseline's.

THE SECOND LEAK, AND WHAT REPLACES IT.  ``target_length`` cannot be the
positive's own length, or the query hands the answer over: the positive alone
would sit at |len - target| = 0.  At inference the target comes from the plan's
bar grid (``snap_plan_to_bar_grid``), i.e. a whole number of the query's own
beats, so training quantises the target the same way.  The trainer refuses to
build a pair whose target is the positive's exact length more often than
chance; see ``tools/train_retrieval_selector.py``.

TIME STRETCH.  ``_values_at`` resamples linearly with ``align_corners=True``,
which leaves the FIRST and LAST frames exactly as they were -- so the seam pose
features are exact on the unstretched endpoints, and only the seam VELOCITY
scales, by ``1 / stretch``.  That is why the cache stores three frames at each
end of the raw segment and the stretch enters as a scalar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- layout ----
# The 151-D vector, same constants as infer_atomic.py and dataset/atomic.py:
# 4 foot-contact channels, 3 root positions, then 24 joints of 6-D rotation.
CONTACT_CHANNELS = 4
ROOT_POSITION_START = 4
ROOT_POSITION_DIMS = 3
ROT6D_START = ROOT_POSITION_START + ROOT_POSITION_DIMS  # 7, global orient first
MOTION_DIM = 151
ROT_DIMS = MOTION_DIM - ROT6D_START  # 144
# Joint rotations begin after the global orient block.  Every scalar summary
# below is computed over THESE dims only, never over 7:13, and that is not a
# detail: a global Rz left-multiplies the orient block, and rot6d is the first
# two columns of R, so a mean of |differences| over it is NOT invariant to
# where the recording happened to be facing.  Measured before the fix: rigidly
# spinning and translating a candidate moved its feature row by 2.4e-3, which
# is exactly the kind of "which upload was this cut from" cue that lets a
# trained scorer find the true continuation without learning anything.
JOINT_ROT_START = ROT6D_START + 6  # 13

# Same joint grouping as infer_atomic._LIMB_JOINTS; duplicated rather than
# imported because importing infer_atomic pulls in the whole inference stack,
# and a test asserts the two agree.
LIMB_JOINTS = {"left_arm": (16, 18, 20, 22), "right_arm": (17, 19, 21, 23),
               "left_leg": (1, 4, 7, 10), "right_leg": (2, 5, 8, 11)}
LIMB_NAMES = ("left_arm", "right_arm", "left_leg", "right_leg")
LIMB_DIMS = {name: tuple(ROT6D_START + 6 * joint + k
                         for joint in joints for k in range(6))
             for name, joints in LIMB_JOINTS.items()}

# Frames kept at each end of a segment.  Three, not two: a velocity estimated
# from two frames is one difference and carries the frame noise; three gives a
# central difference at the frame that actually touches the seam.
EDGE_FRAMES = 3
# Bins the within-segment speed profile is summarised into.  Four, because the
# shape being represented is "起势 - 落定" and four bins can hold a rise, a
# hold and a fall; it is not tuned.
PROFILE_BINS = 4

# Named spans of the feature row, so a control can remove a GROUP by name
# instead of by a hand-copied offset.  The reason this exists: the first honest
# control -- a model TRAINED with the seam zeroed -- still read held-out
# recall@1 0.9798, i.e. the rest of the row was naming the true successor on its
# own.  Hunting that needs to be cheap and exact, and an offset copied into a
# script is neither.
FEATURE_GROUPS = {
    "seam_pose":    (0, 144),
    "seam_velocity": (144, 288),
    "contact_step": (288, 300),
    "root_step":    (300, 303),
    "cand_scale":   (303, 306),   # log length, log stretch, articulation energy
    "cand_beat":    (306, 310),   # own phase sin/cos, period ratio, no-beat flag
    "cand_limb":    (310, 314),
    "cand_profile": (314, 318),
    "query":        (318, 329),
    "next_label":   (329, 353),
}
# Every group computed AGAINST THE PREVIOUS TAIL, i.e. the join.  The first
# version of this tuple held only the two rot6d blocks, which made the
# "seam-blind" control not blind: it still had contact continuity and the root
# height step across the same boundary, and read 0.9798.  A control is only a
# control if it removes the mechanism it is named after.
SEAM_GROUPS = ("seam_pose", "seam_velocity", "contact_step", "root_step")

FEATURE_DIM = (
    ROT_DIMS            # 0   seam pose delta
    + ROT_DIMS          # 1   seam velocity delta
    + 3 * CONTACT_CHANNELS   # 2 prev contacts, candidate contacts, difference
    + 3                 # 3   root height step, candidate ground travel (2)
    + 11                # 4   candidate summary (7 scalars + 4 limb activities)
    + PROFILE_BINS      # 5   candidate speed profile
    + 11                # 6   query summary
    + 6 * CONTACT_CHANNELS   # 7 next-label compatibility (contacts + activity)
)


# --------------------------------------------------------------- geometry ---
def _yaw_and_matrices(raw: torch.Tensor):
    """World yaw of the global-orient block, radians, per frame, plus R."""
    from dataset.rotation_ops import rotation_6d_to_matrix

    matrices = rotation_6d_to_matrix(raw[:, ROT6D_START:ROT6D_START + 6])
    return torch.atan2(matrices[:, 1, 0], matrices[:, 0, 0]), matrices


def rotate_about_z(raw: torch.Tensor, delta: float) -> torch.Tensor:
    """Turn a raw (unnormalized) segment about world z, about its first frame.

    Same operation as ``IndexedAtomicMotionLibrary._rotate_about_z``; z-up is
    measured there, not assumed (max |delta| 6.3e-07 against 1.77 m for y).
    """
    from dataset.rotation_ops import matrix_to_rotation_6d

    out = raw.clone()
    _, matrices = _yaw_and_matrices(out)
    cos, sin = math.cos(float(delta)), math.sin(float(delta))
    turn = torch.tensor([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]],
                        dtype=out.dtype)
    out[:, ROT6D_START:ROT6D_START + 6] = matrix_to_rotation_6d(turn @ matrices)
    root = slice(ROOT_POSITION_START, ROOT_POSITION_START + ROOT_POSITION_DIMS)
    anchor = out[0, root].clone()
    out[:, root] = (turn @ (out[:, root] - anchor).T).T + anchor
    return out


def align_to(raw_edges: torch.Tensor, previous_raw_tail: torch.Tensor) -> torch.Tensor:
    """Put a candidate's edge frames where ``build_draft`` would put them.

    ``raw_edges`` is [2 * EDGE_FRAMES, 151] raw: the head frames then the tail
    frames.  Rotation is applied to the whole block about the HEAD's first
    frame, which is what ``build_draft`` does to the whole segment, so the tail
    frames stay in the same body-relative place they were.

    This is the function that makes training honest -- see the module
    docstring.  It is deliberately the same transform inference can apply, and
    it is applied to positive and negative alike.
    """
    if previous_raw_tail is None:
        return raw_edges
    yaw_prev, _ = _yaw_and_matrices(previous_raw_tail[-1:])
    yaw_head, _ = _yaw_and_matrices(raw_edges[:1])
    # Wrapped to (-pi, pi]: aligning a facing must never take the long way
    # round, which would BE a spin.
    delta = torch.remainder(yaw_prev[0] - yaw_head[0] + math.pi,
                            2 * math.pi) - math.pi
    turned = rotate_about_z(raw_edges, float(delta))
    ground = slice(ROOT_POSITION_START, ROOT_POSITION_START + 2)
    turned[:, ground] = turned[:, ground] + (
        previous_raw_tail[-1, ground] - turned[0, ground])
    return turned


# ------------------------------------------------------------ descriptors ---
@dataclass(frozen=True)
class SegmentDescriptor:
    """Everything the scorer needs about one candidate, precomputed.

    ``edges`` is [2 * EDGE_FRAMES, 151] RAW (unnormalized): head frames first,
    then tail frames.  Raw, because the alignment is a rotation and the release
    normalizer is per-dimension min-max, so a rotation is only meaningful
    before it (the same reason ``_rotate_about_z`` unnormalizes first).
    """

    edges: torch.Tensor          # [2 * EDGE_FRAMES, 151] raw
    length: int
    energy: float
    profile: torch.Tensor        # [PROFILE_BINS] within-segment speed profile
    limb_activity: torch.Tensor  # [4] mean |delta rot6d| per limb
    travel: torch.Tensor         # [2] ground displacement over the segment
    beat_phase: float            # own start phase in its own beat, nan if none
    beat_period: float           # own beat period in frames, nan if none


def _speed(raw: torch.Tensor) -> torch.Tensor:
    """Mean |first difference| of the JOINT rotations, per frame gap.

    Not ``ROT6D_START:`` -- see ``JOINT_ROT_START``.  This makes the number a
    measure of how much the body articulates, exactly invariant to which way
    the dancer was facing when the clip was shot.  It is therefore NOT the same
    quantity as ``IndexedAtomicMotionLibrary._segment_energy``, which includes
    the orient block; the two are close but must not be quoted for each other.
    """
    if len(raw) < 2:
        return torch.zeros(1, dtype=raw.dtype)
    return (raw[1:, JOINT_ROT_START:] - raw[:-1, JOINT_ROT_START:]).abs().mean(dim=-1)


def describe_segment(raw: torch.Tensor, *, beat_phase: float = float("nan"),
                     beat_period: float = float("nan")) -> SegmentDescriptor:
    """Summarise one raw segment.  Called once per candidate and cached."""
    if raw.ndim != 2 or raw.shape[1] != MOTION_DIM:
        raise ValueError("segment must be [frames, {}], got {}".format(
            MOTION_DIM, tuple(raw.shape)))
    frames = len(raw)
    if frames < 2:
        raise ValueError("segment must have at least 2 frames, got {}".format(frames))
    index = torch.arange(EDGE_FRAMES)
    head = raw[torch.clamp(index, max=frames - 1)]
    tail = raw[torch.clamp(frames - EDGE_FRAMES + index, min=0)]
    speed = _speed(raw)
    # Bin the profile by position, then centre it: what is wanted is the SHAPE
    # (does it rise into the end or settle into it), not how fast the segment
    # is overall, which is already carried by ``energy``.
    edges = torch.linspace(0, len(speed), PROFILE_BINS + 1).round().long()
    bins = []
    for low, high in zip(edges[:-1].tolist(), edges[1:].tolist()):
        chunk = speed[low:max(high, low + 1)]
        bins.append(chunk.mean() if len(chunk) else speed.mean())
    profile = torch.stack(bins)
    scale = profile.mean().clamp_min(1e-8)
    profile = profile / scale - 1.0
    # Limb dims are joint rotations only (LIMB_JOINTS never names joint 0), so
    # these are already invariant to the recording's facing.
    activity = torch.stack([
        (raw[1:, list(LIMB_DIMS[name])] - raw[:-1, list(LIMB_DIMS[name])]
         ).abs().mean() if frames > 1 else torch.zeros((), dtype=raw.dtype)
        for name in LIMB_NAMES])
    # Ground travel in the candidate's OWN body frame at its first frame, not
    # in the world.  In the world it points wherever that upload was shot, and
    # a positive drawn from the query's own recording would share that
    # direction with the previous segment while every negative would not --
    # the leak this whole file's canonicalisation exists to close.
    ground = slice(ROOT_POSITION_START, ROOT_POSITION_START + 2)
    world_travel = raw[-1, ground] - raw[0, ground]
    yaw0, _ = _yaw_and_matrices(raw[:1])
    cos, sin = torch.cos(-yaw0[0]), torch.sin(-yaw0[0])
    travel = torch.stack((cos * world_travel[0] - sin * world_travel[1],
                          sin * world_travel[0] + cos * world_travel[1]))
    return SegmentDescriptor(
        edges=torch.cat((head, tail), dim=0),
        length=frames,
        energy=float(speed.mean()),
        profile=profile,
        limb_activity=activity,
        travel=travel,
        beat_phase=float(beat_phase),
        beat_period=float(beat_period),
    )


@dataclass(frozen=True)
class QueryContext:
    """The side of the join that does not depend on which candidate is picked."""

    previous_tail: Optional[torch.Tensor]   # [EDGE_FRAMES, 151] raw, or None
    target_length: int
    gap_frames: int                         # transition frames before this span
    beat_phase: float                       # phase of the span's first frame
    beat_period: float                      # frames per beat at the span
    onset_mean: float                       # window-z-scored onset, mean
    onset_profile: torch.Tensor             # [PROFILE_BINS]
    beats_in_span: float
    next_contacts: torch.Tensor             # [4] pool-mean head contacts
    next_activity: torch.Tensor             # [4] pool-mean head limb activity


def _finite(value: float, fallback: float = 0.0) -> float:
    return float(value) if value == value and abs(value) != float("inf") else fallback


@dataclass(frozen=True)
class PreparedQuery:
    """The parts of a feature row that do not depend on the candidate.

    Hoisted out because they are recomputed once per CANDIDATE otherwise, and
    a pool is a few hundred deep: measured, building a row cost 4.33 ms of
    which almost none was the geometry -- it was rebuilding these same eleven
    numbers and the previous frame's slices over and over.
    """

    previous_tail: Optional[torch.Tensor]
    previous_last: Optional[torch.Tensor]
    previous_step: Optional[torch.Tensor]
    target_length: int
    beat_period: float
    query_block: torch.Tensor
    next_contacts: torch.Tensor
    next_activity: torch.Tensor


def prepare_query(query: QueryContext) -> PreparedQuery:
    block = torch.tensor([
        math.log(max(query.target_length, 1)),
        math.log1p(max(query.gap_frames, 0)),
        math.sin(2 * math.pi * _finite(query.beat_phase)),
        math.cos(2 * math.pi * _finite(query.beat_phase)),
        _finite(query.beat_period, 0.0) / 30.0,
        _finite(query.onset_mean),
        *[_finite(v) for v in query.onset_profile.tolist()],
        _finite(query.beats_in_span),
    ], dtype=torch.float32)
    tail = query.previous_tail
    return PreparedQuery(
        previous_tail=tail,
        previous_last=None if tail is None else tail[-1],
        previous_step=None if tail is None else tail[-1] - tail[-2],
        target_length=int(query.target_length),
        beat_period=float(query.beat_period),
        query_block=block,
        next_contacts=query.next_contacts.to(torch.float32),
        next_activity=query.next_activity.to(torch.float32))


def candidate_features_many(query, descriptors) -> torch.Tensor:
    """[N, FEATURE_DIM] for a whole candidate pool, preparing the query once.

    This is what both the trainer and the inference path call; the
    single-candidate ``candidate_features`` stays as the reference
    implementation and a test asserts the two agree, because a fast path that
    silently disagrees with the one the docstrings describe is the defect shape
    this repository keeps recording.
    """
    prepared = query if isinstance(query, PreparedQuery) else prepare_query(query)
    if not descriptors:
        return torch.zeros(0, FEATURE_DIM)
    return torch.stack([candidate_features(prepared, d) for d in descriptors])


def candidate_features(query,
                       candidate: SegmentDescriptor) -> torch.Tensor:
    """One [FEATURE_DIM] row.  Deterministic, no learned parameters."""
    query = query if isinstance(query, PreparedQuery) else prepare_query(query)
    aligned = align_to(candidate.edges, query.previous_tail)
    head, tail = aligned[:EDGE_FRAMES], aligned[EDGE_FRAMES:]
    stretch = query.target_length / max(candidate.length, 1)

    if query.previous_last is not None:
        prev_last, prev_step = query.previous_last, query.previous_step
    else:
        prev_last = head[0]
        prev_step = head[1] - head[0]
    # The stretch changes the candidate's frame spacing but not its poses, so
    # the velocity it will actually show at the seam is its own step divided by
    # the stretch.  Endpoints are exact under align_corners=True interpolation.
    cand_step = (head[1] - head[0]) / max(stretch, 1e-6)

    seam_pose = head[0, ROT6D_START:] - prev_last[ROT6D_START:]
    seam_velocity = cand_step[ROT6D_START:] - prev_step[ROT6D_START:]

    contacts = torch.cat((
        prev_last[:CONTACT_CHANNELS],
        head[0, :CONTACT_CHANNELS],
        head[0, :CONTACT_CHANNELS] - prev_last[:CONTACT_CHANNELS]))

    height = torch.stack([
        head[0, ROOT_POSITION_START + 2] - prev_last[ROOT_POSITION_START + 2]])
    root_block = torch.cat((height, candidate.travel))

    summary = torch.tensor([
        math.log(max(candidate.length, 1)),
        math.log(max(stretch, 1e-6)),
        candidate.energy,
        math.sin(2 * math.pi * _finite(candidate.beat_phase)),
        math.cos(2 * math.pi * _finite(candidate.beat_phase)),
        _finite(candidate.beat_period, 0.0) / max(_finite(query.beat_period, 1.0), 1.0),
        # A candidate with no beat of its own must be distinguishable from one
        # whose phase happens to be 0.0, or the two collapse into one row.
        0.0 if candidate.beat_phase == candidate.beat_phase else 1.0,
    ], dtype=torch.float32)
    summary = torch.cat((summary, candidate.limb_activity.to(torch.float32)))

    query_block = query.query_block

    tail_activity = (tail[1:, :] - tail[:-1, :]).abs()
    tail_limb = torch.stack([
        tail_activity[:, list(LIMB_DIMS[name])].mean()
        for name in LIMB_NAMES])
    next_block = torch.cat((
        tail[-1, :CONTACT_CHANNELS],
        query.next_contacts,
        tail[-1, :CONTACT_CHANNELS] - query.next_contacts,
        tail_limb,
        query.next_activity,
        tail_limb - query.next_activity))

    row = torch.cat((seam_pose, seam_velocity, contacts, root_block,
                     summary, candidate.profile, query_block, next_block))
    if row.numel() != FEATURE_DIM:
        raise AssertionError("feature row is {} wide, FEATURE_DIM is {}".format(
            row.numel(), FEATURE_DIM))
    return torch.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)


def window_profile(track, start, end, bins: int = PROFILE_BINS) -> torch.Tensor:
    """Mean of ``track`` over ``bins`` equal slices of [start, end).

    Lives here rather than in the trainer so that training and inference cannot
    drift: every draft knob added in 2026-08 was a train/test mismatch by
    construction because the two sides had separate implementations
    (docs/DANCE_QUALITY_DEFECTS.md section 14.5 is what that costs -- the same
    change ranking two baselines oppositely).
    """
    out = torch.zeros(bins, dtype=torch.float32)
    if track is None or end <= start:
        return out
    span = track[start:min(int(end), len(track))]
    if len(span) == 0:
        return out
    edges = torch.linspace(0, len(span), bins + 1).round().long().tolist()
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        chunk = span[low:max(high, low + 1)]
        out[index] = float(torch.as_tensor(chunk).float().mean()) if len(chunk) else 0.0
    return out


def onset_z(music) -> Optional[torch.Tensor]:
    """Channel 0 z-scored over the whole clip, or ``None``.

    Whole clip, not whole corpus: what the scorer needs is "is the music loud
    HERE, for this song", which is invariant to the corpus scale and to any
    per-channel affine rescaling.  The constant-channel guard is relative for
    the reason ``MusicPhaseFeatures`` gives -- for a constant window float32
    rounding leaves std at ~2e-7 and a literal ``std > 0`` test amplifies that
    rounding to full scale.
    """
    if music is None:
        return None
    track = torch.as_tensor(music)[:, 0].float()
    spread = float(track.std())
    if spread <= 1e-6 * max(1.0, float(track.abs().mean())):
        return torch.zeros_like(track)
    return (track - track.mean()) / spread


@dataclass
class LoadedSelector:
    """A trained scorer plus the per-class pool statistics it was fitted with.

    The pool statistics are part of the model, not of the release: they are how
    "what comes next" reaches the features without naming a vocabulary, and a
    checkpoint scored against a different vocabulary's statistics would be
    reading a different feature.  They therefore travel in the checkpoint.
    """

    model: "RetrievalSelector"
    pool_contacts: dict
    pool_activity: dict
    default_contacts: torch.Tensor
    default_activity: torch.Tensor
    provenance: dict

    def stats_for(self, label: int):
        label = int(label)
        return (self.pool_contacts.get(label, self.default_contacts),
                self.pool_activity.get(label, self.default_activity))

    def select(self, features, **kwargs) -> int:
        return self.model.select(features, **kwargs)


def load_selector(path) -> LoadedSelector:
    """Read a checkpoint written by ``tools/train_retrieval_selector.py``."""
    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    if blob.get("stage") != "retrieval_selector":
        raise ValueError("{} is not a retrieval-selector checkpoint".format(path))
    model = RetrievalSelector(feature_dim=int(blob.get("feature_dim", FEATURE_DIM)),
                              hidden=int(blob.get("hidden", 256)),
                              layers=int(blob.get("layers", 3)),
                              dropout=float(blob.get("dropout", 0.1)),
                              disabled_groups=tuple(blob.get("disabled_groups", ())))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    stats = blob.get("label_pool_stats", {})
    default = blob.get("default_pool_stats", (torch.zeros(CONTACT_CHANNELS),
                                              torch.zeros(len(LIMB_NAMES))))
    return LoadedSelector(
        model=model,
        pool_contacts={int(k): v[0] for k, v in stats.items()},
        pool_activity={int(k): v[1] for k, v in stats.items()},
        default_contacts=default[0],
        default_activity=default[1],
        provenance={"path": str(path), "args": blob.get("args", {})})


# ------------------------------------------------------------------ model ---
class RetrievalSelector(nn.Module):
    """Score rows produced by ``candidate_features``.

    The feature statistics travel as BUFFERS, not as a path in ``args``: the
    failure this repository keeps paying for is a value that is recorded but
    never applied, and ``load_state_dict`` being strict makes a checkpoint
    physically unable to run without the normalisation it was fitted with.
    Same argument as ``MusicNormalization`` in ``model/atomic_planner.py``.
    """

    def __init__(self, feature_dim: int = FEATURE_DIM, hidden: int = 256,
                 layers: int = 3, dropout: float = 0.1,
                 seam_block: int = 2 * ROT_DIMS, disabled_groups=()):
        """``disabled_groups`` are zeroed on EVERY forward, training included.

        It travels in the checkpoint rather than living on a command line,
        because a model fitted with a group disabled and run with it enabled is
        reading a feature it never learned -- the same argument that puts
        ``MusicNormalization``'s statistics in a buffer.

        WHAT IS DISABLED BY DEFAULT AND WHY (measured 2026-09-01, 11,948
        recordings, 4,000 held-out pairs).  ``cand_beat`` -- the candidate's own
        beat phase and period -- is a SAME-SONG GIVEAWAY in training: the true
        successor is cut from the query's own recording, so its phase difference
        is exactly 0 and its period ratio exactly 1.0, values no other candidate
        hits by accident.  Its removal costs nothing this repository has
        measured a value for: ``--retrieval-rule tempo`` was closed at n=344
        with no measurable benefit (section 11.3) and ``--retrieval-rule phase``
        made the OUTPUT worse while winning its own criterion (section 15.8).
        """
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.seam_block = int(seam_block)
        for name in disabled_groups:
            if name not in FEATURE_GROUPS:
                raise ValueError("unknown feature group {!r}".format(name))
        self.disabled_groups = tuple(disabled_groups)
        self.register_buffer("feature_mean", torch.zeros(self.feature_dim))
        self.register_buffer("feature_std", torch.ones(self.feature_dim))
        blocks = []
        width = self.feature_dim
        for _ in range(max(layers - 1, 1)):
            blocks += [nn.Linear(width, hidden), nn.LayerNorm(hidden),
                       nn.GELU(), nn.Dropout(dropout)]
            width = hidden
        blocks.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*blocks)

    def set_feature_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Constant channels keep std 1 rather than an epsilon-inflated one --
        scaling up a channel's noise is inventing information (same rule as
        ``tools/fit_motion_normalizer.py``)."""
        mean = torch.as_tensor(mean, dtype=torch.float32).reshape(-1)
        std = torch.as_tensor(std, dtype=torch.float32).reshape(-1)
        if mean.shape[0] != self.feature_dim or std.shape[0] != self.feature_dim:
            raise ValueError("stats must be {} wide".format(self.feature_dim))
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(torch.where(std > 1e-6, std, torch.ones_like(std)))

    def forward(self, features: torch.Tensor, drop_seam: bool = False,
                drop_groups=()) -> torch.Tensor:
        """[..., FEATURE_DIM] -> [...]  scores, higher is a better join.

        ``drop_seam`` zeroes the seam pose and velocity blocks; ``drop_groups``
        zeroes any named spans of ``FEATURE_GROUPS``.  Zeroing at EVALUATION
        time only shows that a model which leaned on a block breaks without it.
        The control that decides anything is a model TRAINED with the block
        zeroed: if that one still wins, the remaining features are naming the
        answer and the mechanism story is wrong (CLAUDE.md 2.1 rule 3).
        """
        if features.shape[-1] != self.feature_dim:
            raise ValueError("features are {} wide, model is {}".format(
                features.shape[-1], self.feature_dim))
        names = list(drop_groups) + [name for name in self.disabled_groups
                                     if name not in drop_groups]
        if drop_seam:
            names += [name for name in SEAM_GROUPS if name not in names]
        if names:
            features = features.clone()
            for name in names:
                if name not in FEATURE_GROUPS:
                    raise ValueError("unknown feature group {!r}; have {}".format(
                        name, sorted(FEATURE_GROUPS)))
                low, high = FEATURE_GROUPS[name]
                features[..., low:high] = 0.0
        normalized = (features - self.feature_mean) / self.feature_std
        return self.net(normalized).squeeze(-1)

    @torch.no_grad()
    def select(self, features: torch.Tensor, *, top_k: int = 8,
               temperature: float = 1.0, generator=None,
               forbid: Sequence[int] = ()) -> int:
        """Sample an index from the top ``k`` scores.  Never an argmax.

        The argmax is what ``--retrieval-rule phase`` did, and it lost on the
        output while winning on its own criterion because it collapsed the
        pool (section 15.8).  ``generator`` is the CLIP's numpy generator, so a
        clip's picks depend on its name and the seed and on nothing else --
        the property ``_variety_rng`` was introduced to restore.

        EVAL MODE IS FORCED HERE, not left to the caller.  With dropout live
        the same candidate pool scores differently on every call, so a repeat
        of a class would draw from a different distribution for a reason that
        has nothing to do with the dance -- and ``_variety_rng`` would no longer
        be the only thing a pick depends on.  ``load_selector`` also calls
        ``eval()``, but a property this load-bearing may not rest on a caller
        remembering.  A test holds it.
        """
        was_training = self.training
        self.eval()
        try:
            scores = self.forward(features)
        finally:
            if was_training:
                self.train()
        if forbid:
            scores = scores.clone()
            keep = [i for i in forbid if 0 <= i < len(scores)]
            if len(keep) < len(scores):
                scores[keep] = -float("inf")
        finite = int(torch.isfinite(scores).sum())
        k = max(1, min(int(top_k), finite))
        top = torch.topk(scores, k)
        if temperature <= 0:
            return int(top.indices[0])
        weights = F.softmax(top.values / float(temperature), dim=0)
        if generator is None:
            pick = int(torch.multinomial(weights, 1).item())
        else:
            probabilities = weights.double().numpy()
            probabilities = probabilities / probabilities.sum()
            pick = int(generator.choice(len(probabilities), p=probabilities))
        return int(top.indices[pick])
