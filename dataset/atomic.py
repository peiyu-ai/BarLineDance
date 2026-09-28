"""Utilities for atomic movement plans and prototype retrieval."""

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

# Larger than any attainable |length - target|, so an excluded candidate can be
# pushed out of an argmin without being removed from the array it lives in.
_UNREACHABLE_DISTANCE = np.int64(1) << np.int64(60)


def _provenance_code(value: object, table: Dict[str, int]) -> int:
    """Intern a provenance string; anything that is not one stays unknown (-1)."""
    if not isinstance(value, str) or not value:
        return -1
    code = table.get(value)
    if code is None:
        code = len(table)
        table[value] = code
    return code


@dataclass(frozen=True)
class AtomicSegment:
    label: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class AtomicPrototype:
    """One retrievable segment and its explicit retrieval provenance.

    ``source_id`` remains only to deserialize/read historical libraries.  New
    materialized releases must use ``retrieval_group_id``: a performance can
    contain several recordings/camera views whose sample names and recording
    IDs differ but whose motions must all be excluded for a query.
    """

    motion: torch.Tensor
    retrieval_group_id: Optional[str] = None
    source_id: Optional[str] = None
    sample_name: Optional[str] = None
    start: Optional[int] = None
    end: Optional[int] = None


_SLICE_NAME = re.compile(r"^(?P<source>.+)_slice\d+$")


def source_id_from_name(name: str) -> str:
    """Collapse a fixed-window sample name to its source-video identity."""
    match = _SLICE_NAME.match(name)
    return match.group("source") if match else name


ROT6D_START = 7          # 4 contacts + 3 root, then 24 x 6 rot6d


def motion_accent_frames(motion: torch.Tensor, min_gap: int = 6,
                         columns: Optional[Sequence[int]] = None) -> np.ndarray:
    """Settle points of a 151-D span: local minima of the rot6d change rate.

    The paper's own notion of a motion beat -- "local minima of segment-wise
    joint velocities" (atomicDance 3.2) -- read directly off the rotation
    channels so it costs no forward kinematics and works identically on
    normalized and raw data (the normalizer is per-dimension affine, which
    preserves where a minimum sits).  ``min_gap`` merges minima closer than a
    fifth of a second: the trace is noisy and two minima two frames apart are
    one settle, not two.
    """
    # ``columns`` restricts the speed trace to one limb's own rot6d channels, so
    # each limb can be given its OWN settle points.  Absolute column indices into
    # the 151-D vector (that is what ``infer_atomic.LIMB_DIMS`` holds), not
    # offsets into the rot6d block, so the caller's group definition and this
    # one cannot drift.
    values = np.asarray(
        motion[..., ROT6D_START:] if columns is None else motion[..., list(columns)],
        dtype=np.float64)
    if len(values) < 5:
        return np.array([], dtype=np.int64)
    speed = np.abs(np.diff(values, axis=0)).mean(axis=1)
    minima = [i for i in range(1, len(speed) - 1)
              if speed[i] <= speed[i - 1] and speed[i] <= speed[i + 1]]
    kept: List[int] = []
    for frame in minima:
        if kept and frame - kept[-1] < min_gap:
            if speed[frame] < speed[kept[-1]]:
                kept[-1] = frame
            continue
        kept.append(frame)
    return np.array(kept, dtype=np.int64)


def dtw_time_map(source_trace: np.ndarray, target_trace: np.ndarray,
                 band: float = 0.35) -> Optional[np.ndarray]:
    """Monotone index map aligning ``source_trace``'s shape onto ``target_trace``.

    Plain banded DTW over z-scored traces with steps (1,1), (1,2), (2,1), i.e.
    local slope between 0.5 and 2.  Returns ``positions`` of length
    ``len(target_trace)`` giving, for each target frame, the (fractional)
    source frame to sample -- or None when no in-band path exists.

    Why DTW and not settle-anchor matching: anchors were tried first and moved
    the draft-to-target speed correlation from 0.036 only to 0.046, because the
    rot6d change trace has a noise minimum every few frames and nearest-anchor
    pairing degenerates to identity.  DTW over the whole trace reaches 0.456 on
    the same data -- that number is the ceiling this exists to buy.
    """
    n, m = len(source_trace), len(target_trace)
    if n < 6 or m < 6:
        return None

    def zscore(x):
        x = np.asarray(x, np.float64)
        return (x - x.mean()) / (x.std() + 1e-9)

    zs, zt = zscore(source_trace), zscore(target_trace)
    width = max(4, int(band * max(n, m)))
    cost = np.full((n, m), np.inf)
    cost[0, 0] = abs(zs[0] - zt[0])
    came = np.zeros((n, m), np.int8)          # 1:(1,1) 2:(1,2) 3:(2,1)
    for i in range(n):
        lo = max(0, int(i * m / n) - width)
        hi = min(m, int(i * m / n) + width)
        for j in range(lo, hi):
            here = cost[i, j]
            if not np.isfinite(here):
                continue
            local = abs(zs[min(i + 1, n - 1)] - zt[min(j + 1, m - 1)])
            for step, (di, dj) in enumerate(((1, 1), (1, 2), (2, 1)), start=1):
                a, b = i + di, j + dj
                if a < n and b < m:
                    value = here + abs(zs[a] - zt[b])
                    if value < cost[a, b]:
                        cost[a, b] = value
                        came[a, b] = step
    if not np.isfinite(cost[n - 1, m - 1]):
        return None
    path = [(n - 1, m - 1)]
    while path[-1] != (0, 0):
        i, j = path[-1]
        step = came[i, j]
        if step == 0:
            break
        di, dj = ((1, 1), (1, 2), (2, 1))[step - 1]
        path.append((i - di, j - dj))
    path.reverse()
    sources = np.array([p[0] for p in path], np.float64)
    targets = np.array([p[1] for p in path], np.float64)
    return np.interp(np.arange(m, dtype=np.float64), targets, sources)


def resample_by_map(motion: torch.Tensor, positions: np.ndarray) -> torch.Tensor:
    """Linear gather of motion rows at (fractional) source positions."""
    frames = motion.shape[0]
    index = torch.from_numpy(np.clip(positions, 0, frames - 1)).to(motion.device)
    low = index.floor().long().clamp(0, frames - 1)
    high = (low + 1).clamp(0, frames - 1)
    fraction = (index - low.to(index.dtype)).unsqueeze(-1).to(motion.dtype)
    return motion[low] * (1.0 - fraction) + motion[high] * fraction


def warp_to_anchors(motion: torch.Tensor, source_anchors: np.ndarray,
                    target_anchors: np.ndarray, max_stretch: float = 1.6,
                    columns: Optional[Sequence[int]] = None) -> torch.Tensor:
    """Monotone piecewise-linear time warp putting source anchors on targets.

    WHY.  A retrieved prototype reaches the draft through one global linear
    resample (``_resample``), so its internal accents land wherever the stretch
    happens to put them.  Measured 2026-08-31 on 50 clips, the completion's
    output speed profile correlates with its own draft's at **-0.110** -- the
    model transmits none of the draft's timing, and the training data explains
    why: the training draft is another performance of the same classes, so its
    timing is uncorrelated with the target's *by construction* and ignoring it
    is the optimum the model correctly found.  This warp is the first half of
    the repair (make the timing worth transmitting); which anchors to use is
    the caller's decision -- the target's own settle points in training, the
    music's beat frames at inference.

    Segments are stretched at most ``max_stretch``-fold either way; an anchor
    pairing that would exceed the cap is dropped rather than clamped, because a
    clamped anchor no longer lands where it claims to and the whole point is
    that it lands.  Pairing is nearest-neighbour, kept strictly monotonic.

    ``columns`` warps only those channels and leaves the rest on their original
    timeline.  That is what makes a PER-LIMB anchor possible, and the reason to
    want one is measured: warping every channel together lands the phase
    (settle -0.094 -> +0.127) but brakes the whole body on one frame, so the
    per-part settle spread collapses from ground truth's 0.578 to 0.222 and limb
    lockstep rises from 0.500 to 0.617.  rot6d is per-joint LOCAL rotation, so
    giving a limb its own timeline is a limb dancing to its own accent, not a
    broken skeleton.
    """
    frames = motion.shape[0]
    if frames < 4 or len(source_anchors) == 0 or len(target_anchors) == 0:
        return motion
    if columns is not None:
        columns = list(columns)
        warped = warp_to_anchors(motion[:, columns], source_anchors,
                                 target_anchors, max_stretch)
        if warped is motion[:, columns]:
            return motion
        out = motion.clone()
        out[:, columns] = warped
        return out
    pairs: List[Tuple[int, int]] = []
    for target in np.sort(np.asarray(target_anchors, dtype=np.int64)):
        if not (0 < target < frames - 1):
            continue
        source = int(source_anchors[np.abs(np.asarray(source_anchors) - target).argmin()])
        if not (0 < source < frames - 1):
            continue
        if pairs and (source <= pairs[-1][0] or target <= pairs[-1][1]):
            continue
        previous = pairs[-1] if pairs else (0, 0)
        ratio = (source - previous[0]) / max(target - previous[1], 1)
        if ratio > max_stretch or ratio < 1.0 / max_stretch:
            continue
        pairs.append((source, target))
    if not pairs:
        return motion
    source_knots = np.array([0] + [p[0] for p in pairs] + [frames - 1], dtype=np.float64)
    target_knots = np.array([0] + [p[1] for p in pairs] + [frames - 1], dtype=np.float64)
    tail = (source_knots[-1] - source_knots[-2]) / max(target_knots[-1] - target_knots[-2], 1)
    if tail > max_stretch or tail < 1.0 / max_stretch:
        return motion
    positions = np.interp(np.arange(frames, dtype=np.float64), target_knots, source_knots)
    index = torch.from_numpy(positions).to(motion.device)
    low = index.floor().long().clamp(0, frames - 1)
    high = (low + 1).clamp(0, frames - 1)
    fraction = (index - low.to(index.dtype)).unsqueeze(-1).to(motion.dtype)
    return motion[low] * (1.0 - fraction) + motion[high] * fraction


def labels_to_segments(labels: torch.Tensor,
                       split_at: Optional[Sequence[int]] = None,
                       min_fragment: Optional[int] = None) -> List[AtomicSegment]:
    """Runs of one label, optionally cut again at ``split_at``.

    ``split_at`` exists for the bar grid.  With ``--plan-bar-grid`` the plan
    holds one label per bar, so two adjacent bars that draw the same label merge
    into a single run and retrieval fills the whole thing with ONE prototype.
    Measured on the T corpus (20 test clips, 2026-09-03): plan runs reach a
    median 2.40 s and a p90 of 6.19 s against a 2.00 s vocabulary, so a quarter
    of runs stretch their prototype past 2x and the tail past 3x -- the dancer
    is then holding one 2 s movement across six seconds, which reads as
    dithering in place rather than as a movement.  The ground truth's own runs
    are 1.90 s median / 3.86 s p90, i.e. it does not do this.

    Passing the bar lines here splits such a run back into one retrieval unit
    per bar, so each bar gets its own prototype and the stretch returns to
    bar/vocabulary ~= 1.09x.  The labels are untouched: this changes how many
    prototypes fill a run, not what the plan says.

    ``min_fragment`` drops a bar line that lands too close to a label change.
    A bar line and a label boundary rarely coincide to the frame -- the plan's
    boundaries come from the vote and ``--plan-min-segment``, the bar lines from
    the beat channel -- so the split can leave a sliver.  On BAR ep160 it
    removes 2 of 174 retrieval units (2026-09-05).

    WHAT IT DOES NOT FIX, stated because this docstring first claimed it did.
    The worst stretch on that arm (1.400: a 7-frame slot filled with a 5-frame
    prototype) SURVIVES the fold, and re-measuring showed why: all 6 sub-beat
    units on that arm start at frame 0.  They are the partial bar at the head of
    the clip, bounded by a genuine label change where the bar-token plan was
    expanded to frames -- not by a bar line this function added.  A head
    fragment is the caller's problem (absorb it, or leave it to gap fill); the
    fold has no business editing a boundary the plan itself contains.

    THE THRESHOLD IS NOT INVENTED.  The T vocabulary is cut on a 4-beat grid
    and its shortest atomic movement is a whole bar (1.23 s); a quarter of a bar
    is one beat, and nothing in the corpus is that short.  So the caller passes
    one beat, and a split is kept only when BOTH sides of it survive it.

    WHAT IT MAY NOT DO.  It only declines to add a split of its own.  It never
    merges two different labels and never moves a label change, so the segment
    labels are identical with and without it -- a short run that the PLAN
    contains is the planner's output and is governed by ``--plan-min-segment``
    upstream, not here.  ``tests/test_bar_fragment_fold.py`` asserts both halves
    of that.
    """
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if labels.numel() == 0:
        return []
    changes = torch.nonzero(labels[1:] != labels[:-1], as_tuple=False).flatten() + 1
    bounds = {0, int(labels.numel()), *(int(c) for c in changes.tolist())}
    if split_at is not None:
        candidates = sorted(
            int(f) for f in split_at if 0 < int(f) < int(labels.numel()))
        if min_fragment is None or int(min_fragment) <= 0:
            bounds.update(candidates)
        else:
            gap = int(min_fragment)
            # Accept greedily against the bounds accepted SO FAR, so a run of
            # close-together lines cannot admit each other in turn and rebuild
            # the slivers this is here to prevent.
            for frame in candidates:
                lower = max(b for b in bounds if b <= frame)
                upper = min(b for b in bounds if b >= frame)
                if frame - lower >= gap and upper - frame >= gap:
                    bounds.add(frame)
    ordered = sorted(bounds)
    return [
        AtomicSegment(int(labels[start].item()), start, end)
        for start, end in zip(ordered[:-1], ordered[1:])
        if end > start
    ]


def majority_vote(labels: torch.Tensor, window_size: int = 5) -> torch.Tensor:
    """Centered sliding-window vote with deterministic center-label tie breaking."""
    if labels.ndim not in (1, 2):
        raise ValueError("labels must have shape [frames] or [batch, frames]")
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd number")
    batched = labels.unsqueeze(0) if labels.ndim == 1 else labels
    if batched.shape[1] == 0:
        return labels.clone()
    radius = window_size // 2
    if radius:
        left = batched[:, :1].expand(-1, radius)
        right = batched[:, -1:].expand(-1, radius)
        padded = torch.cat((left, batched, right), dim=1)
    else:
        padded = batched
    windows = padded.unfold(1, window_size, 1)
    num_classes = int(batched.max().item()) + 1 if batched.numel() else 1
    counts = F.one_hot(windows, num_classes=num_classes).sum(dim=-2)
    winners = counts.argmax(dim=-1)
    center = batched
    max_counts = counts.max(dim=-1).values
    center_counts = counts.gather(-1, center.unsqueeze(-1)).squeeze(-1)
    winners = torch.where(center_counts == max_counts, center, winners)
    return winners.squeeze(0) if labels.ndim == 1 else winners


TRANSITION = 0

TRANSITION_POLICIES = ("protect", "merge")

MERGE_ORDERS = ("shortest", "first")


def merge_short_segments(
    labels: torch.Tensor,
    min_length: int,
    compatibility: Optional[torch.Tensor] = None,
    transition_policy: str = "protect",
    merge_order: str = "shortest",
) -> torch.Tensor:
    """Merge short runs into a neighboring run until every run is long enough.

    This is the minimum-duration half of the paper's post-processing step,
    whose stated purpose is that "frame-level mispredictions in the planner may
    fragment continuous atomic movements": it exists to undo *over*-segmentation
    of atomic movements.

    ``transition_policy`` decides what that means for label 0, which the paper
    leaves open and which this function used to answer by accident:

    ``"protect"`` (default) treats transition as connective tissue rather than
    an undersized atomic movement.  A transition run is never itself merged
    away, and a short atomic run is handed to transition only when it has no
    atomic neighbour at all.  This is the rule
    ``tools/postprocess_atomic_plan.py`` has documented since it was written;
    it is now the only implementation, and that tool calls this one.

    ``"merge"`` is the behaviour every artifact before 2026-08-23 was produced
    with: neighbours were ranked by length with no regard for label, so a
    4-frame atomic fragment beside a 40-frame transition was absorbed *into*
    the transition -- deleting exactly the frames the completion stage is meant
    to condition on -- and a short transition between two atomic movements was
    deleted.  Measured over the 65 clean5b5 M6 clips
    (``runs/clean5b5_plan_rule_scan.json``), the two policies differ by 0.1
    points of transition share -- 0.5588 against 0.5600 at a fixed tie-break --
    because those two errors point in opposite directions and very nearly
    cancel.  What they do not leave alone is the segment structure: 5.54 atomic
    segments per clip against 5.63.  So this is not a lever on the transition
    share; it is the live path finally running the rule that was written down.

    ``merge_order`` is the second, independent decision, and it is separate
    because conflating the two costs a reproduction: ``"shortest"`` resolves the
    shortest offending run first, so a medium fragment cannot be cascaded into a
    shorter one that has not been resolved yet, while ``"first"`` walks left to
    right.  ``tools/postprocess_atomic_plan.py`` has always used ``"shortest"``;
    ``infer_atomic`` used ``"first"`` until 2026-08-23.  On 4,000 synthetic
    transition-heavy plans the two differ on 1.44% of frames and 18% of
    sequences -- small, but not nothing, and enough to move a FID over 260
    generated clips.

    So an artifact produced before 2026-08-23 by the inference path reproduces
    with ``transition_policy="merge", merge_order="first"``; one produced by the
    offline tool reproduces with the defaults.

    A compatibility matrix, when given, outranks length: its
    ``[source, target]`` value is compared first, then the neighbour's length,
    then the earlier neighbour.
    """
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if transition_policy not in TRANSITION_POLICIES:
        raise ValueError(
            "transition_policy must be one of {}, got {!r}".format(
                TRANSITION_POLICIES, transition_policy
            )
        )
    if merge_order not in MERGE_ORDERS:
        raise ValueError(
            "merge_order must be one of {}, got {!r}".format(
                MERGE_ORDERS, merge_order))
    if min_length <= 1:
        return labels.clone()
    protect = transition_policy == "protect"
    result = labels.clone()
    while True:
        segments = labels_to_segments(result)
        if len(segments) == 1:
            # A whole plan shorter than the minimum is not an over-segmentation
            # error, and there is no neighbour to hand it to.
            break
        offenders = [
            index
            for index, segment in enumerate(segments)
            if segment.length < min_length
            and not (protect and segment.label == TRANSITION)
        ]
        if not offenders:
            break
        short_index = (
            min(offenders, key=lambda index: segments[index].length)
            if merge_order == "shortest" else offenders[0]
        )
        segment = segments[short_index]
        candidates = []
        for index in (short_index - 1, short_index + 1):
            if not 0 <= index < len(segments):
                continue
            neighbor = segments[index]
            if protect and neighbor.label == TRANSITION:
                continue
            score = (
                float(compatibility[segment.label, neighbor.label])
                if compatibility is not None
                else 0.0
            )
            # Lexicographic: compatibility, then the longer neighbour, then the
            # earlier one -- a fragment at a boundary belongs to the movement it
            # interrupted.
            candidates.append((score, neighbor.length, -index, neighbor.label))
        if not candidates:
            # Only transitions adjacent: the fragment becomes transition rather
            # than surviving as an atomic movement too short to be one.
            result[segment.start : segment.end] = TRANSITION
            continue
        result[segment.start : segment.end] = max(candidates)[3]
    return result


def refine_plan(
    labels: torch.Tensor,
    vote_window: int = 5,
    min_length: int = 6,
    compatibility: Optional[torch.Tensor] = None,
    transition_policy: str = "protect",
    merge_order: str = "shortest",
) -> torch.Tensor:
    voted = majority_vote(labels, vote_window)
    if voted.ndim == 1:
        return merge_short_segments(voted, min_length, compatibility,
                                    transition_policy, merge_order)
    return torch.stack(
        [
            merge_short_segments(row, min_length, compatibility,
                                 transition_policy, merge_order)
            for row in voted
        ]
    )


def plan_boundaries(labels: torch.Tensor) -> torch.Tensor:
    """Mark the first frame of every new atomic/transition segment."""
    if labels.ndim not in (1, 2):
        raise ValueError("labels must have shape [frames] or [batch, frames]")
    batched = labels.unsqueeze(0) if labels.ndim == 1 else labels
    boundaries = torch.zeros_like(batched, dtype=torch.bool)
    boundaries[:, 1:] = batched[:, 1:] != batched[:, :-1]
    return boundaries.squeeze(0) if labels.ndim == 1 else boundaries


def blank_seam_evidence(mask: torch.Tensor, boundaries: torch.Tensor,
                        half_width: int) -> torch.Tensor:
    """Zero the draft's conditioning mask within ``half_width`` of every seam.

    WHY.  The draft is two prototypes butt-jointed, so the frames either side of
    a seam carry a velocity STEP that no dancer produced -- and with the mask at
    1.0 there, training tells the completion model that step IS the retrieval
    evidence to reproduce.  Measured on the T line 2026-09-05, 17 eval clips,
    filler frames excluded: jerk within +-2 frames of a unit boundary is 0.3515
    against ground truth's 0.2553 at the same frame indices, while the interior
    matches (0.2135 vs 0.2045).  The defect is entirely at the join.

    Zeroing the mask there asks the model to GENERATE the transition instead of
    copying it.  The label channel is deliberately NOT blanked (see
    ``--completion-label-channel``): the model still knows which class it is
    leaving and which it is entering, so this removes the false evidence without
    removing the instruction.

    ``half_width`` 0 returns the mask unchanged, which is every artifact before
    this existed.
    """
    if half_width <= 0:
        return mask
    if boundaries.shape != mask.shape[:boundaries.ndim]:
        raise ValueError("boundaries must match the mask's leading shape")
    batched = boundaries.unsqueeze(0) if boundaries.ndim == 1 else boundaries
    window = 2 * int(half_width) + 1
    near = torch.nn.functional.max_pool1d(
        batched.to(dtype=torch.float32).unsqueeze(1),
        kernel_size=window, stride=1, padding=int(half_width)).squeeze(1) > 0
    if boundaries.ndim == 1:
        near = near.squeeze(0)
    blanked = mask.clone()
    blanked[near] = 0.0
    return blanked


class AtomicMotionLibrary:
    """In-memory prototypes indexed by atomic label, as described in Sec. 3.3."""

    def __init__(self, motions: Mapping[int, Sequence[Union[torch.Tensor, AtomicPrototype]]]) -> None:
        self.motions: Dict[int, Tuple[AtomicPrototype, ...]] = {
            int(label): tuple(
                AtomicPrototype(
                    motion=segment.motion.detach().clone(),
                    retrieval_group_id=segment.retrieval_group_id,
                    source_id=segment.source_id,
                    sample_name=segment.sample_name,
                    start=segment.start,
                    end=segment.end,
                )
                if isinstance(segment, AtomicPrototype)
                else AtomicPrototype(motion=segment.detach().clone())
                for segment in segments
            )
            for label, segments in motions.items()
        }
        if 0 in self.motions:
            raise ValueError("label 0 is reserved for transitions")
        self._build_retrieval_index()

    def _build_retrieval_index(self) -> None:
        """Precompute the vectors ``retrieve`` scans, one per label.

        ``retrieve`` used to walk a label's whole candidate list twice in Python:
        once to build the filtered tuple, once inside ``min``'s key function.
        Measured on the wild 340-frame release (1,338,864 prototypes over 4,159
        labels, ``cProfile`` on real batches), those two walks were 5.4 M Python
        iterations per batch of 64 and 0.31 s of the 0.59 s that building one
        batch's completion draft cost -- against a 0.18 s model step on the same
        card.  The cost grows linearly with the corpus, so it gets worse as the
        wild release grows.

        The arrays below turn one query into a single NumPy pass.  What must not
        change is *which* prototype comes back: ``min`` returns the first
        candidate attaining the smallest ``|length - target|`` in insertion
        order, and ``np.argmin`` likewise returns the first minimum, so the two
        agree by construction rather than by luck.
        ``tests/test_atomic.py::test_vectorised_retrieval_matches_the_python_scan``
        fuzzes them against each other, including ties and exclusions, because
        "reads like it agrees" is not the standard this repository uses.

        Provenance is stored as integer codes.  ``-1`` means "not a non-empty
        string", i.e. unknown provenance, and it is *never* eligible once an
        exclusion is requested -- the same fail-closed rule the Python filter
        applied through its ``isinstance(..., str) and ...`` guard.
        """
        self._group_code_by_id: Dict[str, int] = {}
        self._source_code_by_id: Dict[str, int] = {}
        self._lengths: Dict[int, np.ndarray] = {}
        self._group_codes: Dict[int, np.ndarray] = {}
        self._source_codes: Dict[int, np.ndarray] = {}
        for label, prototypes in self.motions.items():
            count = len(prototypes)
            lengths = np.empty(count, dtype=np.int64)
            groups = np.empty(count, dtype=np.int64)
            sources = np.empty(count, dtype=np.int64)
            for position, prototype in enumerate(prototypes):
                lengths[position] = prototype.motion.shape[0]
                groups[position] = _provenance_code(prototype.retrieval_group_id, self._group_code_by_id)
                sources[position] = _provenance_code(prototype.source_id, self._source_code_by_id)
            self._lengths[label] = lengths
            self._group_codes[label] = groups
            self._source_codes[label] = sources

    @classmethod
    def from_sequences(
        cls,
        motions: Sequence[torch.Tensor],
        labels: Sequence[torch.Tensor],
        min_length: int = 1,
        names: Optional[Sequence[str]] = None,
        retrieval_group_ids: Optional[Sequence[Optional[str]]] = None,
    ) -> "AtomicMotionLibrary":
        if len(motions) != len(labels):
            raise ValueError("motions and labels must contain the same number of sequences")
        if names is not None and len(names) != len(motions):
            raise ValueError("names and motions must contain the same number of sequences")
        if retrieval_group_ids is not None and len(retrieval_group_ids) != len(motions):
            raise ValueError("retrieval_group_ids and motions must contain the same number of sequences")
        groups: Dict[int, List[AtomicPrototype]] = {}
        for index, (motion, plan) in enumerate(zip(motions, labels)):
            if motion.shape[0] != plan.shape[0]:
                raise ValueError("motion and label frame counts must match")
            sample_name = names[index] if names is not None else None
            source_id = (
                source_id_from_name(sample_name)
                if isinstance(sample_name, str) and sample_name
                else None
            )
            retrieval_group_id = (
                retrieval_group_ids[index] if retrieval_group_ids is not None else None
            )
            if retrieval_group_id is not None and (
                not isinstance(retrieval_group_id, str) or not retrieval_group_id.strip()
            ):
                raise ValueError("retrieval_group_ids must contain non-empty strings or None")
            for segment in labels_to_segments(plan):
                if segment.label and segment.length >= min_length:
                    groups.setdefault(segment.label, []).append(
                        AtomicPrototype(
                            motion=motion[segment.start : segment.end].detach().clone(),
                            retrieval_group_id=retrieval_group_id,
                            source_id=source_id,
                            sample_name=sample_name,
                            start=segment.start,
                            end=segment.end,
                        )
                    )
        return cls(groups)

    def state_dict(self):
        """Serialize motions together with their source-level provenance.

        Retrieval exclusions are only meaningful when every candidate carries
        provenance.  Earlier versions serialized bare tensors, which silently
        turned a source-safe library into one with unknown provenance after a
        save/load round-trip.  Keep the compact label-indexed layout, but make
        each record self-describing.
        """
        return {
            label: [
                {
                    "motion": prototype.motion.detach().clone(),
                    "retrieval_group_id": prototype.retrieval_group_id,
                    "source_id": prototype.source_id,
                    "sample_name": prototype.sample_name,
                    "start": prototype.start,
                    "end": prototype.end,
                }
                for prototype in prototypes
            ]
            for label, prototypes in self.motions.items()
        }

    @classmethod
    def from_state_dict(cls, state):
        """Restore a library, accepting legacy tensor-only states safely.

        Tensor-only states remain loadable for callers that do not request a
        source exclusion.  Their provenance is deliberately restored as
        unknown, so a later source-safe retrieval fails closed instead of
        reusing an unverifiable prototype.
        """
        if not isinstance(state, Mapping):
            raise TypeError("atomic motion library state must be a mapping")
        motions: Dict[int, List[AtomicPrototype]] = {}
        for raw_label, records in state.items():
            if not isinstance(records, Sequence):
                raise TypeError("atomic motion library records must be sequences")
            restored = []
            for record in records:
                if isinstance(record, AtomicPrototype):
                    restored.append(
                        AtomicPrototype(
                            motion=record.motion.detach().clone(),
                            retrieval_group_id=record.retrieval_group_id,
                            source_id=record.source_id,
                            sample_name=record.sample_name,
                            start=record.start,
                            end=record.end,
                        )
                    )
                elif torch.is_tensor(record):
                    # Compatibility with the old tensor-only state format.
                    restored.append(AtomicPrototype(motion=record.detach().clone()))
                elif isinstance(record, Mapping):
                    if "motion" not in record or not torch.is_tensor(record["motion"]):
                        raise TypeError("serialized atomic prototype requires a tensor motion")
                    restored.append(
                        AtomicPrototype(
                            motion=record["motion"].detach().clone(),
                            retrieval_group_id=record.get("retrieval_group_id"),
                            source_id=record.get("source_id"),
                            sample_name=record.get("sample_name"),
                            start=record.get("start"),
                            end=record.get("end"),
                        )
                    )
                else:
                    raise TypeError("unsupported serialized atomic prototype")
            motions[int(raw_label)] = restored
        return cls(motions)

    def retrieve(
        self,
        label: int,
        target_length: int,
        *,
        exclude_retrieval_group_ids: Iterable[str] = (),
        exclude_source_ids: Iterable[str] = (),
    ) -> torch.Tensor:
        """Return a duration-nearest prototype outside excluded source videos.

        Training windows from the same original video are highly overlapping,
        so excluding the whole source is stricter and safer than trying to
        infer an overlap from a potentially ambiguous window name.

        The eligibility rule and the duration-nearest choice are unchanged; only
        the scan is.  See ``_build_retrieval_index`` for why, and for why the two
        implementations must select the identical prototype rather than an
        equally good one.
        """
        if label == 0:
            raise ValueError("transition frames do not have atomic prototypes")
        label = int(label)
        candidates = self.motions.get(label, ())
        excluded_groups = (
            {exclude_retrieval_group_ids}
            if isinstance(exclude_retrieval_group_ids, str)
            else set(exclude_retrieval_group_ids)
        )
        excluded_sources = (
            {exclude_source_ids}
            if isinstance(exclude_source_ids, str)
            else set(exclude_source_ids)
        )
        if excluded_groups and excluded_sources:
            raise ValueError("use either retrieval-group or legacy source exclusions, not both")
        eligible = None
        if candidates:
            if excluded_groups:
                # Unknown provenance (code -1) cannot establish that a candidate
                # is outside the query group, so it is excluded too.  Do not
                # reconstruct the group from a window name: camera views
                # commonly have distinct names.
                eligible = self._eligibility(
                    self._group_codes[label], excluded_groups, self._group_code_by_id
                )
            elif excluded_sources:
                # Once an exclusion is requested, unknown provenance is rejected
                # rather than treated as safe.
                eligible = self._eligibility(
                    self._source_codes[label], excluded_sources, self._source_code_by_id
                )
        if not candidates or (eligible is not None and not eligible.any()):
            if excluded_groups:
                raise KeyError(
                    "no retrieval-group-safe prototype candidates for atomic label {}".format(label)
                )
            if excluded_sources:
                raise KeyError(
                    "no source-safe prototype candidates for atomic label {}".format(label)
                )
            raise KeyError("no prototype candidates for atomic label {}".format(label))
        distance = np.abs(self._lengths[label] - np.int64(target_length))
        if eligible is not None:
            distance = np.where(eligible, distance, _UNREACHABLE_DISTANCE)
        chosen = candidates[int(distance.argmin())]
        return self._resample(chosen.motion, target_length)

    @staticmethod
    def _eligibility(codes: np.ndarray, excluded: Iterable[str], table: Mapping[str, int]) -> np.ndarray:
        eligible = codes >= 0
        for value in excluded:
            code = table.get(value)
            if code is not None:
                eligible &= codes != code
        return eligible

    @staticmethod
    def _resample(motion: torch.Tensor, target_length: int) -> torch.Tensor:
        if motion.ndim != 2:
            raise ValueError("motion must have shape [frames, features]")
        if target_length < 1:
            raise ValueError("target_length must be positive")
        if motion.shape[0] == target_length:
            return motion.clone()
        values = motion.transpose(0, 1).unsqueeze(0)
        return F.interpolate(values, size=target_length, mode="linear", align_corners=True).squeeze(0).transpose(0, 1)

    def build_draft(
        self,
        labels: torch.Tensor,
        feature_dim: int,
        *,
        exclude_retrieval_group_ids: Iterable[str] = (),
        exclude_source_ids: Iterable[str] = (),
        allow_missing: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return coarse motion M0 and mask w; transition frames remain zero.

        With ``allow_missing=True``, a segment with no source-safe prototype is
        left as an unconditioned zero-mask region.  It is never replaced with a
        same-source prototype merely to fill the condition.
        """
        draft = torch.zeros(labels.shape[0], feature_dim, dtype=torch.float32, device=labels.device)
        mask = torch.zeros(labels.shape[0], 1, dtype=torch.float32, device=labels.device)
        self.fill_draft(
            labels,
            draft,
            mask,
            exclude_retrieval_group_ids=exclude_retrieval_group_ids,
            exclude_source_ids=exclude_source_ids,
            allow_missing=allow_missing,
        )
        return draft, mask

    def fill_draft(
        self,
        labels: torch.Tensor,
        draft: torch.Tensor,
        mask: torch.Tensor,
        *,
        exclude_retrieval_group_ids: Iterable[str] = (),
        exclude_source_ids: Iterable[str] = (),
        allow_missing: bool = False,
        align_to: Optional[torch.Tensor] = None,
    ) -> None:
        """Write one sequence's draft/mask into caller-owned views.

        ``align_to`` -- when given, the target window's own motion [frames,
        features]: each retrieved prototype is time-warped so its settle points
        land on the target segment's settle points (``warp_to_anchors``; the
        -0.110 measurement in its docstring is why).  Off by default so every
        earlier checkpoint keeps training against the drafts it was born with.

        Split out of ``build_draft`` so a caller holding a whole minibatch can
        allocate ``[batch, frames, features]`` once and pass row views in.
        ``build_draft``'s two per-sequence ``torch.zeros`` were 17% of the
        profiled cost of preparing a batch of 64 on the wild release: at 205 KB
        each they land above glibc's mmap threshold, so every call paid fresh
        page faults instead of reusing a buffer.  Nothing about *what* is written
        changes -- ``build_draft`` still allocates and delegates here.
        """
        frames = labels.shape[0]
        if draft.ndim != 2 or draft.shape[0] != frames:
            raise ValueError("draft view must have shape [frames, features]")
        if tuple(mask.shape) != (frames, 1):
            raise ValueError("mask view must have shape [frames, 1]")
        feature_dim = draft.shape[-1]
        excluded_groups = (
            (exclude_retrieval_group_ids,)
            if isinstance(exclude_retrieval_group_ids, str)
            else tuple(exclude_retrieval_group_ids)
        )
        excluded_sources = (
            (exclude_source_ids,)
            if isinstance(exclude_source_ids, str)
            else tuple(exclude_source_ids)
        )
        if excluded_groups and excluded_sources:
            raise ValueError("use either retrieval-group or legacy source exclusions, not both")
        for segment in labels_to_segments(labels):
            if segment.label == 0:
                continue
            try:
                motion = self.retrieve(
                    segment.label,
                    segment.length,
                    exclude_retrieval_group_ids=excluded_groups,
                    exclude_source_ids=excluded_sources,
                ).to(labels.device)
            except KeyError:
                if allow_missing:
                    continue
                raise
            if motion.shape[1] != feature_dim:
                raise ValueError("prototype feature dimension does not match requested draft")
            if align_to is not None and segment.length >= 12:
                trace = np.abs(np.diff(np.asarray(motion[:, ROT6D_START:], np.float64),
                                       axis=0)).mean(1)
                target_span = np.asarray(
                    align_to[segment.start:segment.end, ROT6D_START:], np.float64)
                target_trace = np.abs(np.diff(target_span, axis=0)).mean(1)
                positions = dtw_time_map(trace, target_trace)
                if positions is not None:
                    # trace has length L-1; the map addresses trace frames, and
                    # motion frame k sits between trace k-1 and k -- sampling
                    # motion at the same positions keeps the settle geometry.
                    motion = resample_by_map(
                        motion, np.append(positions, positions[-1] + 1.0))
            draft[segment.start : segment.end] = motion
            mask[segment.start : segment.end] = 1.0
