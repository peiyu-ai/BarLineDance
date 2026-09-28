"""One token per 4-beat BAR: the grid, the pooling, and the way back to frames.

WHY THIS MODULE EXISTS AT ALL.  M1 has cut the corpus on the music's own beat
grid since 2026-08-20 (``tools/segment_on_music_beats.py``, mode ``grid``,
``beats_per_segment 4``): ``data/wild3d/txy_t_labels/report.json`` records
``encoders.segmentation = "music-beat-grid"`` and the segmentation it was built
from validates with ``boundary_off_grid 0`` and ``span_not_k_beats 0`` over
2,015 segments.  Segment duration there is 2.00 s median with a MAXIMUM of
2.97 s, so nothing in the vocabulary is longer than one bar.  The planner,
however, is trained on the PER-FRAME label track: on ``train/labels.npy``
(5329 x 150) adjacent frame pairs differ on 9,874 of 794,021 positions, 1.24%,
so 98.76% of the tokens it is asked to emit are "same as the previous frame"
while a 150-frame window spans only about 2.5 bars.  Planning at bar resolution
is the experiment that asks whether that copy operation is what the capacity
went to.

THIS FILE IS THE SINGLE DEFINITION of the three operations that experiment
needs, so the training-release builder and inference cannot drift apart:

  ``bar_lines``      where the bars start, from the beat frames and a phase;
  ``pool_music``     a bar's frames of 35-D music reduced to one vector;
  ``expand_labels``  one label per bar written back across that bar's frames.

Two definitions of the pooling is precisely the train/test mismatch this
repository keeps paying for, so an inference path that pooled "the same way"
by copying the code would already be the defect.  Import from here instead.

THE HEAD AND TAIL ARE TOKENS, NOT DROPPED.  ``bar_bounds`` returns
``[0, cut_1, ..., cut_n, length]``: the frames before the first bar line and
after the last one are each a (short) token of their own.  That is not a new
rule -- it is exactly what ``infer_atomic.snap_plan_to_bar_grid`` has always
done with ``--plan-bar-grid``, and matching it is what makes the plan's bars
and ``--draft-bar-prototypes``' retrieval bars the SAME bars rather than two
grids that could drift.  Their pooled music is taken over fewer frames than a
full bar, so their statistics are not a full bar's; callers that care should
report ``partial_head_frames`` / ``partial_tail_frames`` rather than silently
treating them as full bars.
"""

from typing import Sequence

import numpy as np
import torch

# The beat one-hot's channel in the repository's 35-D music features.  The same
# number is spelled out in ``infer_atomic.BEAT_CHANNEL``,
# ``dataset/global_music.py`` and ``eval/utils/musicbeat.py``; it is repeated
# here only so this module can be imported without pulling in inference.
BEAT_CHANNEL = 34

# Channels a bar COUNTS rather than averages.  Channel 33 is the onset-peak
# one-hot: the number of onset peaks in a bar is a rhythmic density, and its
# mean divides that count by the bar's length, confounding "how many accents"
# with "how long the bar is".  ``tools/build_bar_planner_release.py`` has summed
# it since the bar release was first built and states that reasoning in its own
# docstring.
#
# THIS CONSTANT EXISTS BECAUSE THE TWO SIDES HAD DRIFTED.  Until 2026-09-08 the
# release builder summed channel 33 and this module -- the one inference pools
# with -- meaned it.  Measured on the first bar of the first val window (58
# frames): the release stores 15.0000 and inference produced 0.2586, a factor of
# 58, on the single channel that carries accent density; every other channel
# agreed to 1e-5.  So the shipped bar planner was trained on a channel whose
# inference-time value was ~1/58 of what it learned from.  That is exactly the
# defect this module's docstring warns about two paragraphs above, and it
# happened anyway because the builder implemented its own pooling instead of
# importing this one.
COUNT_CHANNELS = (33,)

# The onset envelope, and the number of sub-bar slices the rhythm pooling keeps.
# 8 slices of a 4-beat bar is one per eighth note, the finest division a 30 fps
# feature track resolves at this corpus's median 57-frame bar.
ONSET_CHANNEL = 0
SUBBAR_BINS = 8

# The pooling names a checkpoint may declare.  ``mean`` is the default because
# it is the smallest thing that can be pooled and it keeps the music width at
# the release's 35-D, so a bar-trained planner and a frame-trained completion
# still agree on ``music_dim``.  ``mean_std`` doubles the width and keeps a
# measure of WITHIN-bar variation, which plain averaging destroys -- the beat
# one-hot, for instance, means "4 beats happened in this span" after mean
# pooling and nothing about where.  NEITHER IS VALIDATED against a planning
# result; they are representations, and which one plans better is a question
# for the training experiment to measure, not for this module to assert.
# ``mean_rhythm`` appends the bar's rhythm SHAPE to the pooled vector:
# SUBBAR_BINS sub-bar means of the onset envelope and of the onset-peak count,
# 2 * SUBBAR_BINS extra channels.  It exists because the bar mean keeps only
# **3.5%** of channel 0's variance and 0.23% of channel 34's, against chroma's
# 67-74% -- pooling hands the planner harmony and timbre nearly intact and
# rhythm nearly not at all.  Measured 2026-09-08 on the T-line bar release,
# ridge predicting a bar's rotation energy on held-out RECORDINGS: bar mean
# -0.0507 val / +0.0411 test, this description +0.0597 / +0.1377, beyond all
# 200 permutation draws (null mean -0.003, p95 +0.011).  It does NOT make the
# atomic label predictable (0.089 / 0.126 against floors 0.130 / 0.221).
BAR_POOLINGS = ("mean", "mean_std", "mean_rhythm")
DEFAULT_BAR_POOLING = "mean"


def beat_frames(music, channel: int = BEAT_CHANNEL) -> np.ndarray:
    """The frames carrying a beat, read the way the whole repository reads them."""
    array = music.numpy() if hasattr(music, "numpy") else np.asarray(music)
    if array.ndim != 2 or array.shape[1] <= channel:
        raise ValueError(
            "the bar grid needs music channel {} (the beat one-hot), got shape {}"
            .format(channel, tuple(array.shape)))
    return np.flatnonzero(array[:, channel] > 0.5)


def bar_lines(beats: Sequence[int], beats_per_segment: int, phase: int):
    """Every ``beats_per_segment``-th beat from ``phase`` -- the bar starts.

    Same slice as ``infer_atomic.snap_plan_to_bar_grid`` and
    ``infer_atomic.bar_bounds_of``; both now come through here so there is one
    construction of the grid rather than three that happen to agree today.
    """
    if beats_per_segment < 1:
        raise ValueError("beats_per_segment must be >= 1")
    return [int(b) for b in np.asarray(beats)[int(phase)::int(beats_per_segment)]]


def bar_bounds(beats: Sequence[int], beats_per_segment: int, phase: int, length: int):
    """``[0, cut, ..., length]`` -- one span per token, covering every frame.

    Cuts at or outside ``(0, length)`` are dropped, so the result is strictly
    increasing and ``len(result) - 1`` is the token count.
    """
    length = int(length)
    if length <= 0:
        raise ValueError("a bar grid needs a positive length, got {}".format(length))
    cuts = [c for c in bar_lines(beats, beats_per_segment, phase) if 0 < c < length]
    return [0] + cuts + [length]


def subbar_rhythm(frames, bins: int = SUBBAR_BINS,
                  channels=(ONSET_CHANNEL, 33)):
    """A bar's rhythm shape: per-channel means over ``bins`` equal slices.

    THE SINGLE DEFINITION, imported by both the release builder and inference.
    Writing it twice is what produced the 58x channel-33 mismatch recorded in
    ``COUNT_CHANNELS`` above, three hours before this function existed.
    """
    values = torch.as_tensor(
        frames.numpy() if hasattr(frames, "numpy") else np.asarray(frames),
        dtype=torch.float64)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("a bar needs at least one frame of [frames, channels]")
    keep = [c for c in channels if c < values.shape[1]]
    edges = np.linspace(0, values.shape[0], int(bins) + 1).astype(int)
    slices = []
    for low, high in zip(edges, edges[1:]):
        if high > low:
            slices.append(values[low:high, keep].mean(dim=0))
        else:
            # A bar shorter than ``bins`` frames repeats its nearest frame
            # rather than emitting a zero, which would read as silence.
            slices.append(values[min(int(low), values.shape[0] - 1), keep])
    return torch.stack(slices).reshape(-1).to(torch.float32)


def pool_music(music, bounds: Sequence[int], pooling: str = DEFAULT_BAR_POOLING,
               count_channels=None):
    """One vector per bar span, ``[tokens, pooled_dim]``.

    ``mean`` gives ``music_dim`` channels, ``mean_std`` gives ``2 * music_dim``
    as ``concat(mean, std)`` with ``unbiased=False`` -- the same two moments and
    the same estimator ``infer_atomic.track_summary`` already uses for the
    whole-track summary, so a one-frame span reads std 0 instead of NaN.

    ``COUNT_CHANNELS`` are SUMMED rather than meaned; pass ``count_channels`` to
    override (``()`` reproduces the pre-2026-09-08 all-mean behaviour, which is
    what artifacts built before the mismatch above was found used at inference).
    """
    if pooling not in BAR_POOLINGS:
        raise ValueError("unknown bar pooling {!r}; expected one of {}".format(
            pooling, list(BAR_POOLINGS)))
    # float64 for the accumulation, float32 on the way out.  The release
    # builder has always pooled in float64 and cast once at the end, and
    # accumulating in float32 instead moves the stored bar music by up to
    # 2.3e-4 relative -- small, but it would mean the single shared definition
    # no longer reproduces the releases already built with it, which is the one
    # thing having a single definition is for.
    values = torch.as_tensor(
        music.numpy() if hasattr(music, "numpy") else np.asarray(music),
        dtype=torch.float64)
    if values.ndim != 2:
        raise ValueError("bar pooling needs [frames, music_dim]")
    bounds = [int(b) for b in bounds]
    if len(bounds) < 2 or any(b >= e for b, e in zip(bounds[:-1], bounds[1:])):
        raise ValueError("bar bounds must be strictly increasing, got {}".format(bounds))
    if bounds[-1] > len(values):
        raise ValueError("bar bounds run to frame {} but the music has {}".format(
            bounds[-1], len(values)))
    counts = tuple(c for c in (COUNT_CHANNELS if count_channels is None
                               else count_channels) if c < values.shape[1])
    rows = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        span = values[start:end]
        summary = span.mean(dim=0)
        for channel in counts:
            summary = summary.clone()
            summary[channel] = span[:, channel].sum()
        if pooling == "mean":
            rows.append(summary)
        elif pooling == "mean_rhythm":
            rows.append(torch.cat((summary, subbar_rhythm(span))))
        else:
            rows.append(torch.cat((summary, span.std(dim=0, unbiased=False))))
    return torch.stack(rows).to(torch.float32)


def pooled_dim(music_dim: int, pooling: str = DEFAULT_BAR_POOLING) -> int:
    """What ``pool_music`` will produce, without pooling anything.

    Used to state the mismatch when a checkpoint's ``music_dim`` and the
    declared pooling cannot both be true.
    """
    if pooling not in BAR_POOLINGS:
        raise ValueError("unknown bar pooling {!r}; expected one of {}".format(
            pooling, list(BAR_POOLINGS)))
    if pooling == "mean":
        return int(music_dim)
    if pooling == "mean_rhythm":
        return int(music_dim) + 2 * SUBBAR_BINS
    return int(music_dim) * 2


def expand_labels(bar_labels, bounds: Sequence[int]) -> torch.Tensor:
    """One label per bar -> one label per frame, exactly covering ``bounds``.

    The returned track has ``bounds[-1]`` frames.  An off-by-one here would
    mislabel every frame after it and nothing downstream could see it -- this
    repository has a recorded case of exactly that shape (a contrast read at
    ``change[f-1]`` instead of ``change[f]``, which systematically under-read
    its own arm, docs section 2.2) -- so the count is asserted rather than
    assumed.
    """
    labels = torch.as_tensor(bar_labels, dtype=torch.long).reshape(-1)
    bounds = [int(b) for b in bounds]
    if len(bounds) - 1 != len(labels):
        raise ValueError(
            "{} bar labels do not fill {} bar spans".format(len(labels), len(bounds) - 1))
    if any(b >= e for b, e in zip(bounds[:-1], bounds[1:])):
        raise ValueError("bar bounds must be strictly increasing, got {}".format(bounds))
    frames = torch.zeros(bounds[-1], dtype=torch.long)
    for label, start, end in zip(labels.tolist(), bounds[:-1], bounds[1:]):
        frames[start:end] = label
    return frames
