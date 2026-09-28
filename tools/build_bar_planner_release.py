#!/usr/bin/env python3
"""Re-tokenise the T-line planner release so that **one token is one bar**.

WHY THIS EXISTS
===============
``docs/DANCE_QUALITY_DEFECTS.md`` section 27 records the defect this release is
built to test.  The corpus was cut on a **4-beat music grid**
(``runs/txy_t_seg_beat4/segmentation.json``: ``mode grid / beats_per_segment 4 /
phase energy / edges drop``), so a "move" in the vocabulary is exactly one bar,
median 2.00 s, **max 2.97 s**.  Ground truth changes label from one bar to the
next **78.4%** of the time; the shipping planner reaches 58.3% and the best of
twelve checkpoints 65.6%, so this is not checkpoint selection.

Section 27.4's mechanism hypothesis: the planner is trained on the **per-frame**
label track (``release_v3/train/labels.npy``, shape ``(5329, 150)``), where
adjacent frames carry the same label on 98.76% of positions
(9,874 changes in 794,021 adjacent pairs).  A 150-frame window spans about
2.5 bars, so the model emits 150 tokens to make roughly 2 real decisions, and
the decision is worth 1.24% of the loss.  **It is a hypothesis.**  This tool
builds the data half of the judgement experiment: the same corpus, the same
splits, the same label space, with the token axis changed from frames to bars.

WHAT IT WRITES
==============
An indexed release ``AtomicSequenceDataset`` already reads -- ``motion.npy``,
``music.npy``, ``labels.npy``, ``names.json``, ``retrieval_groups.json`` under
``<root>/<split>/`` -- with ``K`` bars where the frame release had 150 frames:

    labels[n, k]      majority label among bar k's frames
    music[n, k, 35]   bar k's music, pooled from the frames it spans
    motion[n, k, 1]   **ZEROS.  A PLACEHOLDER.  The planner never reads motion**
                      (``train_atomic.py`` line 1060 calls
                      ``model(batch["labels"], batch["music"],
                      batch["padding_mask"], ...)``).  It is one channel of
                      zeros rather than a pooled 151-D vector because a pooled
                      motion vector is not motion -- averaging a rotation
                      representation over two seconds produces a pose nobody
                      performed -- and a future reader must not be able to
                      mistake it for one.  **Only the planner stage may train on
                      this release.**  Any completion run would silently be
                      conditioned on zeros.

Sidecars written beside the splits:

    bars.jsonl                one row per window: which recording, which bars,
                              and the frame span of each bar, so a bar-level
                              plan can be expanded back to frames.
    music_stats.pt            ``{"mean", "std"}`` over the **train** split's
                              bars, for ``train_atomic.py --music-stats``.
                              Pooling changes the music's distribution, so the
                              frame-level stats a previous checkpoint carries
                              are simply wrong here (see below).
    bar_release_build.json    provenance + every verification number.

**It is deliberately not called ``build.json``.**  ``train_atomic.py``'s
``validate_training_data_root`` demands a ``build.json`` declare 151-D motion,
35-D music and 150-frame timing at 30 fps; two of those are false here.  Writing
a ``build.json`` that satisfies the gate would be a gate that passes by lying --
the exact shape ``CLAUDE.md`` section 2 forbids.  With no ``build.json`` the
trainer takes its documented legacy path and stamps the run
``release_contract_validated: false``, ``headline_eligible: false``.  That is
the correct label for a diagnostic release and it should stay attached to it.

DECISIONS, EACH WITH THE MEASUREMENT THAT SETTLED IT
====================================================

1. WINDOW LENGTH ``K = 4`` BARS, STRIDE 1 BAR.
   270 of the release's recordings carry a bar grid; they hold 1,839 bars,
   median 7 bars per recording (min 3, max 14).  Measured windows and
   recordings kept, stride 1:

       K   train/val/test windows   train recordings kept   median span
       3        1037/149/113               226/226              5.9 s
       4         811/123/ 95               214/226              7.7 s
       5         597/ 97/ 77               189/226              9.5 s
       6         408/ 73/ 59               149/226             11.4 s
       8         139/ 31/ 27                80/226             14.9 s

   K = 4 is one 4-bar musical phrase (16 beats), gives **3 decisions per
   window** against the frame release's ~1.5, and still keeps 214 of 226 train
   recordings.  K = 6 doubles the decisions but throws away a third of the
   recordings, and the 12 dropped at K = 4 are the 3-bar ones, 24 of the
   corpus's 1,263 train bar transitions -- 1.9%.
   Stride 1 bar is the bar analogue of the frame release's stride 15 of 150:
   4x overlap instead of 10x.  Every adjacent bar pair in a kept recording
   appears in up to 3 windows at 3 different positions.  Note the honest
   ceiling this exposes: the train split contains **1,263 unique bar
   transitions**, and no stride can create more.  The frame release did not
   have more either -- it had the same 1,263, spread over 5,329 windows.

2. MUSIC POOLING.  Channels are ``[0] onset envelope, [1:21] 20 MFCC,
   [21:33] 12 chroma, [33] onset-peak one-hot, [34] beat one-hot``
   (``data/audio_extraction/baseline_features.py``).

   * **Channels 0-32: mean** over the bar's frames.  Measured variance
     retention (pooled variance / frame variance, over all 1,839 bars):
     onset 3.6%, MFCC 27-57%, chroma 66-74%.  All keep usable spread; the
     onset envelope loses the most because its frame-level variance *is* the
     transient, which no bar-level token can carry.
   * **Channel 33 (onset peak): SUM**, i.e. the number of detected onsets in
     the bar.  Measured: mean 12.27, std 3.84, range 0-26.  The mean instead
     retains 2.0% of frame variance and confounds "how many onsets" with "how
     long the bar is".
   * **Channel 34 (beat): MEAN, and the count would have been a disaster.**
     The grid puts exactly 4 beats in every bar -- verified here, the sum over
     each of all 1,839 bars is exactly 4.0, one unique value -- so a count is a
     **constant** and carries zero information.  The mean is ``4 / bar_frames``
     (correlation with ``1/bar_frames`` = 1.000000), which is the bar's tempo:
     mean 0.0669, std 0.0119, range 0.0449-0.1081, relative spread 17.8%.
     Its variance retention against the frame-level one-hot reads 0.23%, and
     that number is not a warning here: the frame variance of a one-hot is the
     on/off flicker between beats, which is precisely the thing a bar token
     must not pretend to carry.  What survives is the tempo, and the bar's
     duration in frames is recoverable exactly as ``4 / music[..., 34]``.
     (The repo's recorded failure -- the beat channel arriving at 0.05% of
     input variance, ``model/atomic_planner.py`` ``MusicNormalization`` -- is
     about *magnitude* against other channels, and is fixed by z-scoring, not
     by pooling.  Hence ``music_stats.pt``, refit on these pooled bars.)

3. SPLITS.  Taken per recording from ``windows.jsonl`` and asserted: every
   window of a recording declares the same split, and no recording appears in
   two splits of the output.  Nothing is re-split here.

WHAT THE VERIFICATION FOUND (all recomputed on every run, into the report)
==========================================================================
* **Round trip is exact in-grid.**  Expanding each bar's label back over its
  frames reproduces the per-frame track on **0 of 113,333 frames wrong**.  That
  is not luck: the labels were assigned per grid segment upstream, so no bar
  contains two labels (0 of 1,839 bars is non-constant).  The majority vote is
  therefore not discarding minority frames -- there are none.
* **What bar resolution genuinely cannot represent** is the 21,412 frames
  (15.9% of the 134,745 the release covers) that lie **outside** the grid --
  the head and tail the ``edges: drop`` policy discarded.  All 21,412 carry
  label 0.  This is why the frame-level and bar-level filler shares differ and
  must never be compared to each other: see the report's ``filler`` block.
* **The headline statistic survives**: bar-to-bar change on this release's
  ground-truth bar labels reads **78.46%** over 1,569 adjacent pairs in 270
  recordings, against the 78.4% recorded in section 27.2.

USAGE
=====
    python3 tools/build_bar_planner_release.py \
        --release-root /cache/atomicdance-assets/scratch/txy_t/release_v3 \
        --segmentation /cache/atomicdance-assets/runs/txy_t_seg_beat4/segmentation.json \
        --out /cache/atomicdance-assets/scratch/txy_t/release_bar_v1
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import sys

import numpy as np

# Run as ``python3 tools/build_bar_planner_release.py``, sys.path[0] is
# ``tools/`` and the repo root is absent, so ``from dataset.bar_tokens import``
# raises ModuleNotFoundError -- and this file's pooling now DELEGATES there, so
# that import is on the main path rather than in an optional branch.  Same trap
# tools/render_sample_strip.py records for its plan-strip import.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.bar_tokens import pool_music as _pool_music  # noqa: E402
from dataset.bar_tokens import subbar_rhythm as _subbar_rhythm  # noqa: E402

MUSIC_DIM = 35
BEAT_CHANNEL = 34
ONSET_PEAK_CHANNEL = 33
ONSET_CHANNEL = 0
# 8 slices of a 4-beat bar = one per eighth note, the finest division a 30 fps
# feature track resolves reliably at this corpus's median 57-frame bar.
SUBBAR_BINS = 8
FILLER_LABEL = 0
FPS = 30.0


# ---------------------------------------------------------------- identity ---

def recording_id_from_segmentation(name):
    """``"<upload>__clip000"`` -> ``"wild_v5:<upload>:clip000"``.

    The two artifacts genuinely disagree about how a recording is named:
    ``segmentation.json`` uses ``__``, the release uses ``wild_v5:`` with ``:``.
    Converted explicitly, and the caller asserts the match count, so a naming
    change upstream fails loudly instead of silently matching nothing.
    """
    if "__" not in name:
        raise ValueError("segmentation sequence id has no '__': {!r}".format(name))
    upload, clip = name.split("__", 1)
    if not upload or not clip:
        raise ValueError("segmentation sequence id is malformed: {!r}".format(name))
    return "wild_v5:{}:{}".format(upload, clip)


# ------------------------------------------------------------- pooling ------

def pool_bar_music(frames, beat_channel=BEAT_CHANNEL,
                   count_channels=(ONSET_PEAK_CHANNEL,)):
    """Pool one bar's ``(frames, 35)`` music into a single 35-D vector.

    Mean everywhere except ``count_channels``, which are summed.  See the module
    docstring for the measurement behind each: the beat channel is meaned on
    purpose (its sum is the constant 4 and carries nothing), the onset-peak
    channel is summed on purpose (its mean confounds count with bar length).
    ``beat_channel`` is named only to document that it is deliberately *not* in
    ``count_channels``.
    """
    frames = np.asarray(frames, dtype=np.float64)
    if frames.ndim != 2 or frames.shape[0] < 1:
        raise ValueError("a bar needs at least one frame of (frames, channels) music")
    if beat_channel is not None and beat_channel in tuple(count_channels):
        raise ValueError(
            "the beat channel must not be summed: the grid puts exactly k beats "
            "in every bar, so the sum is a constant")
    # DELEGATED, not reimplemented.  This function used to do its own mean-with-
    # summed-counts, and dataset/bar_tokens.pool_music -- the one INFERENCE
    # pools with -- meaned every channel, so the shipped bar planner trained on
    # a channel-33 value 58x larger than the one it was later fed.  Both sides
    # now come through the single definition; see COUNT_CHANNELS there.
    pooled = _pool_music(frames, [0, frames.shape[0]], "mean",
                         count_channels=tuple(count_channels))[0]
    return pooled.numpy().astype(np.float32)


def subbar_rhythm(frames, bins=SUBBAR_BINS,
                  channels=(ONSET_CHANNEL, ONSET_PEAK_CHANNEL)):
    """The bar's rhythm SHAPE: per-channel means over ``bins`` equal slices.

    WHY THIS EXISTS.  ``pool_bar_music`` takes one mean over the whole bar, and
    this release's own pooling table records what that costs: channel 0, the
    onset envelope, keeps **3.5% of its variance**, and channel 34, the beat
    one-hot, keeps 0.23%, while chroma keeps 67-74% and MFCC 27-57%.  The
    planner is handed harmony and timbre nearly intact and rhythm nearly not at
    all -- and rhythm is what the operator's complaint is about ("动作频率贴合
    音乐节奏的密度不及 gt").

    MEASURED, 2026-09-08, on this release's own source-disjoint split, ridge
    regression predicting a bar's rotation energy on held-out RECORDINGS:

        music description            val R^2    test R^2   dims
        bar mean (shipped)           -0.0507     0.0411      35
        bar mean + std               -0.0367     0.1142      70
        8 sub-bars, onset+peak       +0.0597    +0.1377      16
        8 sub-bars, all channels     -0.1178    -0.0135     280

    The 16-D rhythm shape is the only description positive on BOTH splits, and a
    200-draw permutation null (target shuffled within train) has mean -0.003 and
    p95 +0.011, so both readings sit beyond every draw.  All 280 sub-bar
    channels overfit 810 training rows, which is why this keeps only the two
    rhythmic ones rather than un-pooling everything.

    WHAT IT DOES NOT BUY, stated so nobody reads more into it: the same features
    predict the 21-class atomic label at 0.089 val / 0.126 test against
    majority-class floors of 0.130 / 0.221 -- better than the bar mean's 0.065 /
    0.063 and still under the floor.  This restores a rhythm signal that pooling
    destroyed; it does not make the atomic vocabulary music-predictable.

    DELEGATED to dataset.bar_tokens.subbar_rhythm, which inference also uses.
    Two implementations of a pooling is what produced the 58x channel-33
    mismatch this file's COUNT_CHANNELS note records.
    """
    return _subbar_rhythm(frames, bins=bins,
                          channels=channels).numpy().astype(np.float32)


def majority_label(frame_labels, num_classes):
    """The bar's label: the most frequent frame label, lowest index on a tie.

    A tie is broken toward the lower class index rather than toward 0/filler
    specifically -- ``np.argmax`` on the histogram.  On this corpus no bar is
    even mixed (0 of 1,839), so the rule is never exercised on real data; it
    exists so a future non-grid corpus fails predictably rather than randomly.
    """
    frame_labels = np.asarray(frame_labels)
    if frame_labels.size == 0:
        raise ValueError("a bar needs at least one frame label")
    if int(frame_labels.min()) < 0:
        raise ValueError(
            "frame labels carry the invalid sentinel {}; a bar covering frames "
            "no window labels must be dropped, not majority-voted"
            .format(int(frame_labels.min())))
    counts = np.bincount(frame_labels.astype(np.int64), minlength=int(num_classes))
    return int(np.argmax(counts))


def bar_shape_stats(sequences):
    """Shape of a bar-label corpus: how often it changes, and how it clumps.

    ``sequences`` is one list of bar labels PER RECORDING, and adjacent pairs
    never cross a recording boundary -- two recordings concatenated would invent
    a transition at the join.

    ``change_rate`` is the number this release exists to move: ground truth
    changes label from one bar to the next 78.4% of the time while the shipping
    frame-token planner reaches 58.3%.  ``runs_ge_3`` / ``runs_ge_5`` and
    ``longest_run`` are the other direction of the same question -- a planner
    that emits one label forever reads change_rate 0 and runs_ge_5 1.0.
    ``filler_bar_share`` is class 0's share, named rather than inferred.
    """
    bars = changes = adjacent_pairs = 0
    runs = []
    histogram = Counter()
    for labels in sequences:
        labels = [int(x) for x in labels]
        if not labels:
            continue
        bars += len(labels)
        histogram.update(labels)
        adjacent_pairs += len(labels) - 1
        run = 1
        for previous, current in zip(labels, labels[1:]):
            if previous != current:
                changes += 1
                runs.append(run)
                run = 1
            else:
                run += 1
        runs.append(run)
    return {
        "recordings": len(sequences),
        "bars": bars,
        "adjacent_pairs": adjacent_pairs,
        "changes": changes,
        "change_rate": (changes / adjacent_pairs) if adjacent_pairs else 0.0,
        "runs": len(runs),
        "runs_ge_3": (sum(1 for r in runs if r >= 3) / len(runs)) if runs else 0.0,
        "runs_ge_5": (sum(1 for r in runs if r >= 5) / len(runs)) if runs else 0.0,
        "longest_run": max(runs) if runs else 0,
        "distinct_classes": len(histogram),
        "label_histogram": {str(k): int(v) for k, v in sorted(histogram.items())},
        "filler_bar_share": (histogram.get(0, 0) / bars) if bars else 0.0,
        "top_class_share": (max(histogram.values()) / bars) if bars else 0.0,
    }


def window_starts(num_bars, window_bars, stride_bars):
    """Bar indices a window may start at.  Empty when the recording is short."""
    if window_bars < 2:
        raise ValueError(
            "a window of {} bar(s) carries no adjacent bar pair, so it encodes "
            "no transition".format(window_bars))
    if stride_bars < 1:
        raise ValueError("stride must be >= 1 bar, got {}".format(stride_bars))
    return list(range(0, num_bars - window_bars + 1, stride_bars))


def load_release_windows(release_root):
    """Group the frame release's ``windows.jsonl`` by recording.

    Returns ``(windows, split_of, group_of)``.  Split and retrieval group are
    read off the window rows and asserted CONSTANT per recording: the splits are
    per recording by construction, and a recording that straddled two of them
    would put the same song on both sides of the evaluation.
    """
    release_root = Path(release_root)
    windows, split_of, group_of = {}, {}, {}
    with open(str(release_root / "windows.jsonl"), "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sequence = row["sequence_id"]
            windows.setdefault(sequence, []).append(row)
            for field, store in (("split", split_of),
                                 ("retrieval_group_id", group_of)):
                value = row[field]
                if store.setdefault(sequence, value) != value:
                    raise ValueError(
                        "{} declares two values of {!r} ({} and {}); the frame "
                        "release is not per-recording consistent"
                        .format(sequence, field, store[sequence], value))
    for sequence in windows:
        windows[sequence].sort(key=lambda row: int(row["start_frame"]))
    return windows, split_of, group_of


def reconstruct_tracks(windows, arrays, num_channels=MUSIC_DIM):
    """Overlapping windows -> one per-frame label and music track per recording.

    The frame release is 150-frame windows at stride 15, so most frames are
    carried by up to ten windows.  Writing them back is only sound if the copies
    AGREE, so the overlap is compared rather than trusted: ``label_conflicts``
    counts frames where two windows disagree on the label and
    ``music_max_overlap_delta`` is the largest disagreement in the music.  Both
    read 0 on this corpus, which is what licenses the reconstruction; a nonzero
    either would mean the windows are not slices of one track.

    Frames no window covers keep the release's own invalid sentinel (-1) so that
    ``majority_label`` refuses them and ``covered_segments`` drops the bars that
    contain them, rather than either silently voting on a hole.
    """
    frames = max(int(row["end_frame_exclusive"]) for row in windows)
    label_track = np.full(frames, -1, dtype=np.int64)
    music_track = np.zeros((frames, num_channels), dtype=np.float64)
    written = np.zeros(frames, dtype=bool)
    compared = conflicts = 0
    max_delta = 0.0
    for row in windows:
        labels, valid_mask, music = arrays[row["split"]]
        index = int(row["array_index"])
        start = int(row["start_frame"])
        stop = int(row["end_frame_exclusive"])
        window_labels = np.asarray(labels[index], dtype=np.int64)
        window_music = np.asarray(music[index], dtype=np.float64)
        valid = np.asarray(valid_mask[index]).astype(bool)
        span = stop - start
        window_labels, window_music, valid = (window_labels[:span],
                                              window_music[:span], valid[:span])
        target = slice(start, stop)
        overlap = written[target]
        if overlap.any():
            compared += int(overlap.sum())
            conflicts += int(np.sum(
                label_track[target][overlap] != window_labels[overlap]))
            max_delta = max(max_delta, float(np.abs(
                music_track[target][overlap] - window_music[overlap]).max()))
        fresh = ~overlap
        label_track[target] = np.where(fresh, window_labels, label_track[target])
        music_track[target] = np.where(fresh[:, None], window_music,
                                       music_track[target])
        # An invalid frame label stays the sentinel even where a window covers
        # it: `required_label_valid_fraction` is 1.0 on this release, so this
        # branch never fires here and exists so a partially labelled release
        # fails at the bar rather than being voted on.
        label_track[target] = np.where(valid & fresh, window_labels,
                                       label_track[target])
        written[target] = True
    checks = {
        "frames": int(frames),
        "covered_frames": int(written.sum()),
        "label_frames_compared": int(compared),
        "label_conflicts": int(conflicts),
        "music_max_overlap_delta": float(max_delta),
    }
    return label_track, music_track, checks


def covered_segments(segments, covered_frames):
    """The bars the release's windows fully cover, which must be a PREFIX.

    Returns ``(kept, dropped)``.  A bar the windows stop inside is dropped
    rather than truncated -- a short bar is not a bar, and pooling one would put
    a partial bar's statistics beside whole ones.
    """
    kept = [index for index, segment in enumerate(segments)
            if int(segment["end"]) <= int(covered_frames)]
    if kept != list(range(len(kept))):
        raise ValueError(
            "the covered bars are not a prefix ({}); the release's windows do "
            "not tile this recording".format(kept[:8]))
    return [segments[i] for i in kept], len(segments) - len(kept)


def bar_tokenise(label_track, music_track, segments, num_classes,
                 beats_per_bar, count_channels=(ONSET_PEAK_CHANNEL,),
                 with_rhythm=False):
    """One recording's bars: labels, pooled music, frame spans, and checks.

    Returns ``(bar_labels, bar_music, spans, checks)``.  ``checks`` carries the
    round-trip and grid evidence per bar: how many frames the expanded bar label
    would get wrong, whether the bar starts on a beat, and how many beats it
    holds -- the last two are the assertion that this grid and this music are on
    the same frame index base, which nothing else in the pipeline states.
    """
    bar_labels = []
    bar_music = []
    spans = []
    roundtrip_wrong = 0
    roundtrip_frames = 0
    mixed_bars = 0
    off_beat_starts = 0
    beat_counts = Counter()
    for segment in segments:
        start = int(segment["start"])
        stop = int(segment["end"])
        if stop > label_track.shape[0] or start < 0 or stop <= start:
            raise ValueError("bar [{}, {}) is outside the reconstructed track "
                             "of {} frames".format(start, stop, label_track.shape[0]))
        frame_labels = label_track[start:stop]
        if int(frame_labels.min()) < 0:
            raise ValueError("bar [{}, {}) has frames no release window covers"
                             .format(start, stop))
        label = majority_label(frame_labels, num_classes)
        frames = music_track[start:stop]
        bar_labels.append(label)
        pooled = pool_bar_music(frames, count_channels=count_channels)
        if with_rhythm:
            # Appended, never substituted: the 35 pooled channels stay exactly
            # where every existing reader expects them, so a checkpoint trained
            # on 35-D and one trained on 51-D differ by a suffix and the
            # music_dim mismatch is caught by the trainer's own declaration
            # check rather than by silently reading chroma as onset.
            pooled = np.concatenate([pooled, subbar_rhythm(frames)])
        bar_music.append(pooled)
        spans.append((start, stop))
        roundtrip_wrong += int(np.sum(frame_labels != label))
        roundtrip_frames += int(stop - start)
        mixed_bars += int(np.any(frame_labels != frame_labels[0]))
        beats = np.flatnonzero(frames[:, BEAT_CHANNEL] > 0.5)
        beat_counts[int(beats.size)] += 1
        off_beat_starts += int(not (beats.size and beats[0] == 0))
    checks = {
        "roundtrip_frames": roundtrip_frames,
        "roundtrip_wrong": roundtrip_wrong,
        "mixed_bars": mixed_bars,
        "off_beat_starts": off_beat_starts,
        "beat_count_histogram": beat_counts,
        "bars_off_expected_beats": sum(
            n for beats, n in beat_counts.items() if beats != beats_per_bar),
    }
    return (np.asarray(bar_labels, dtype=np.int64),
            np.asarray(bar_music, dtype=np.float32), spans, checks)


# ----------------------------------------------------------------- build ----

def build(release_root, segmentation_path, out_root, window_bars=4,
          stride_bars=1, num_classes=21, beats_per_bar=4,
          expected_recordings=270, dry_run=False, with_rhythm=False):
    """Write the bar release and return the verification report."""
    token_music_dim = MUSIC_DIM + (SUBBAR_BINS * 2 if with_rhythm else 0)
    release_root = Path(release_root)
    out_root = Path(out_root)

    windows, split_of, group_of = load_release_windows(release_root)
    with open(str(segmentation_path), "r") as handle:
        segmentation = json.load(handle)
    grids = {}
    for record in segmentation["records"]:
        grids[recording_id_from_segmentation(record["sequence"])] = record

    matched = sorted(set(grids) & set(windows))
    report_ids = {
        "release_recordings": len(windows),
        "segmentation_records": len(grids),
        "matched_recordings": len(matched),
        "release_without_grid": sorted(set(windows) - set(grids)),
        "grid_without_release": len(set(grids) - set(windows)),
    }
    if expected_recordings is not None and len(matched) != expected_recordings:
        raise AssertionError(
            "expected {} recordings to match a bar grid, matched {} "
            "(release {}, segmentation {}) -- the id mapping "
            "'<upload>__clip' -> 'wild_v5:<upload>:clip' is the thing to check"
            .format(expected_recordings, len(matched), len(windows), len(grids)))
    if report_ids["release_without_grid"]:
        raise AssertionError(
            "{} release recordings have no bar grid; a bar release cannot "
            "represent them: {}".format(len(report_ids["release_without_grid"]),
                                        report_ids["release_without_grid"][:5]))

    arrays = {}
    for split in sorted({split_of[s] for s in matched}):
        base = release_root / split
        arrays[split] = (
            np.load(str(base / "labels.npy"), mmap_mode="r"),
            np.load(str(base / "label_valid_mask.npy"), mmap_mode="r"),
            np.load(str(base / "music.npy"), mmap_mode="r"),
        )

    per_recording = {}
    totals = Counter()
    beat_counts = Counter()
    music_max_overlap_delta = 0.0
    frame_labels_in_grid = Counter()
    out_of_grid_labels = Counter()
    # Frame-level moments of the in-grid music, accumulated in this same pass so
    # the pooling table below compares bars against exactly the frames they were
    # pooled from rather than against a second, separately reconstructed pass.
    frame_sum = np.zeros(MUSIC_DIM)
    frame_sumsq = np.zeros(MUSIC_DIM)
    frame_count = 0
    uncovered_tail = []
    for sequence in matched:
        label_track, music_track, track_checks = reconstruct_tracks(
            windows[sequence], arrays)
        segments, uncovered = covered_segments(
            grids[sequence]["segments"], label_track.shape[0])
        if uncovered:
            totals["bars_dropped_uncovered_tail"] += uncovered
            uncovered_tail.append({"sequence_id": sequence, "bars": uncovered})
        if not segments:
            raise AssertionError(
                "no bar of {} is fully covered by the release windows".format(sequence))
        bar_labels, bar_music, spans, checks = bar_tokenise(
            label_track, music_track, segments, num_classes, beats_per_bar,
            with_rhythm=with_rhythm)
        per_recording[sequence] = {
            "split": split_of[sequence],
            "retrieval_group_id": group_of[sequence],
            "bar_labels": bar_labels,
            "bar_music": bar_music,
            "spans": spans,
        }
        for key in ("label_frames_compared", "label_conflicts", "frames",
                    "covered_frames"):
            totals[key] += track_checks[key]
        music_max_overlap_delta = max(music_max_overlap_delta,
                                      track_checks["music_max_overlap_delta"])
        for key in ("roundtrip_frames", "roundtrip_wrong", "mixed_bars",
                    "off_beat_starts", "bars_off_expected_beats"):
            totals[key] += checks[key]
        beat_counts.update(checks["beat_count_histogram"])
        in_grid = np.zeros(label_track.shape[0], dtype=bool)
        for start, stop in spans:
            in_grid[start:stop] = True
            for value, count in zip(*np.unique(label_track[start:stop],
                                               return_counts=True)):
                frame_labels_in_grid[int(value)] += int(count)
            block = music_track[start:stop]
            frame_sum += block.sum(axis=0)
            frame_sumsq += (block ** 2).sum(axis=0)
            frame_count += block.shape[0]
        outside = label_track[(~in_grid) & (label_track >= 0)]
        for value, count in zip(*np.unique(outside, return_counts=True)):
            out_of_grid_labels[int(value)] += int(count)

    if totals["label_conflicts"]:
        raise AssertionError(
            "{} of {} overlapping label frames disagree between windows; the "
            "reconstruction is not well defined".format(
                totals["label_conflicts"], totals["label_frames_compared"]))
    if totals["bars_off_expected_beats"]:
        raise AssertionError(
            "{} bars do not hold exactly {} beats in the release's own beat "
            "channel; the grid and the music are not on the same frame index "
            "base".format(totals["bars_off_expected_beats"], beats_per_bar))

    # ---- shape statistics, at recording resolution (comparable to 27.2) ----
    by_split = defaultdict(list)
    for sequence in matched:
        by_split[per_recording[sequence]["split"]].append(
            per_recording[sequence]["bar_labels"].tolist())
    recording_stats = {
        "all": bar_shape_stats([per_recording[s]["bar_labels"].tolist()
                                for s in matched]),
    }
    for split, sequences in by_split.items():
        recording_stats[split] = bar_shape_stats(sequences)

    # ---- windows ----
    rows = defaultdict(list)
    bars_jsonl = []
    dropped = []
    for sequence in matched:
        entry = per_recording[sequence]
        starts = window_starts(len(entry["bar_labels"]), window_bars, stride_bars)
        if not starts:
            dropped.append((sequence, len(entry["bar_labels"])))
            continue
        split = entry["split"]
        for start in starts:
            stop = start + window_bars
            index = len(rows[split])
            rows[split].append({
                "labels": entry["bar_labels"][start:stop],
                "music": entry["bar_music"][start:stop],
                "name": "{}_bars{:04d}".format(sequence, start),
                "group": entry["retrieval_group_id"],
            })
            spans = entry["spans"][start:stop]
            bars_jsonl.append({
                "split": split,
                "array_index": index,
                "sequence_id": sequence,
                "recording_id": sequence,
                "retrieval_group_id": entry["retrieval_group_id"],
                "window_id": "{}/bars{:06d}".format(sequence, start),
                "bar_index_start": start,
                "bar_index_end_exclusive": stop,
                "window_bars": window_bars,
                "stride_bars": stride_bars,
                "beats_per_bar": beats_per_bar,
                "start_frame": int(spans[0][0]),
                "end_frame_exclusive": int(spans[-1][1]),
                "bar_frame_spans": [[int(a), int(b)] for a, b in spans],
                "bar_labels": [int(x) for x in entry["bar_labels"][start:stop]],
                "seconds": round((spans[-1][1] - spans[0][0]) / FPS, 4),
            })

    window_stats = {}
    for split, entries in rows.items():
        window_stats[split] = bar_shape_stats([e["labels"].tolist() for e in entries])
        window_stats[split]["windows"] = len(entries)
    spans_seconds = [row["seconds"] for row in bars_jsonl]
    window_span = {
        "median_s": float(np.median(spans_seconds)) if spans_seconds else None,
        "min_s": float(np.min(spans_seconds)) if spans_seconds else None,
        "max_s": float(np.max(spans_seconds)) if spans_seconds else None,
    }

    # ---- pooling effect on each channel's variance ----
    pooled = np.concatenate(
        [per_recording[s]["bar_music"] for s in matched], axis=0).astype(np.float64)
    frame_mean = frame_sum / frame_count
    frame_variance = np.maximum(frame_sumsq / frame_count - frame_mean ** 2, 0.0)
    pooled_variance = pooled.var(axis=0)
    channel_names = (["onset"] + ["mfcc{:02d}".format(i) for i in range(20)]
                     + ["chroma{:02d}".format(i) for i in range(12)]
                     + ["onset_peak", "beat"])
    pooling_table = []
    for channel in range(MUSIC_DIM):
        rule = "sum" if channel == ONSET_PEAK_CHANNEL else "mean"
        pooling_table.append({
            "channel": channel,
            "name": channel_names[channel],
            "rule": rule,
            "frame_std": float(np.sqrt(max(frame_variance[channel], 0.0))),
            "bar_std": float(np.sqrt(max(pooled_variance[channel], 0.0))),
            "variance_retained": (float(pooled_variance[channel] / frame_variance[channel])
                                  if frame_variance[channel] > 0 else None),
            # A summed channel's variance is not on the frame channel's scale,
            # so its ratio is a magnitude change, not a retention.  Flagged so
            # nobody reads onset_peak's 92.5 as "pooling added information".
            "variance_retained_comparable": rule == "mean",
            "bar_mean": float(pooled[:, channel].mean()),
            "bar_min": float(pooled[:, channel].min()),
            "bar_max": float(pooled[:, channel].max()),
            "constant_across_bars": bool(pooled_variance[channel] <= 0.0),
            # The gate below: a channel that varied frame to frame and does not
            # vary bar to bar is one the pooling *destroyed*.  A channel that
            # was already constant in the input is not this tool's doing, and
            # failing on it would only make the tool unusable on a small corpus.
            "destroyed_by_pooling": bool(frame_variance[channel] > 0.0
                                         and pooled_variance[channel] <= 0.0),
        })
    destroyed = [row["name"] for row in pooling_table if row["destroyed_by_pooling"]]
    if destroyed:
        raise AssertionError(
            "pooling turned {} from a varying frame channel into a constant "
            "across every bar; that channel is gone from the release".format(
                destroyed))

    in_grid_frames = sum(frame_labels_in_grid.values())
    out_grid_frames = sum(out_of_grid_labels.values())
    filler_block = {
        "bar_share": recording_stats["all"]["filler_bar_share"],
        "frame_share_in_grid": frame_labels_in_grid[FILLER_LABEL] / in_grid_frames,
        "frame_share_all_covered": (
            (frame_labels_in_grid[FILLER_LABEL] + out_of_grid_labels[FILLER_LABEL])
            / (in_grid_frames + out_grid_frames)),
        "in_grid_frames": in_grid_frames,
        "out_of_grid_frames": out_grid_frames,
        "out_of_grid_filler_frames": out_of_grid_labels[FILLER_LABEL],
        "reading": (
            "bar_share is the number a bar planner must be judged against. "
            "The frame-level share is larger because every frame the grid's "
            "'edges: drop' policy left outside a bar carries label 0 by "
            "construction, not because a dancer paused. Do not put the two in "
            "one column."),
    }

    report = {
        "tool": "tools/build_bar_planner_release.py",
        "inputs": {
            "release_root": str(release_root),
            "segmentation": str(segmentation_path),
        },
        "out_root": str(out_root),
        "token_axis": "bar",
        "window_bars": window_bars,
        "stride_bars": stride_bars,
        "beats_per_bar": beats_per_bar,
        "num_classes": num_classes,
        "music_dim": token_music_dim,
        "subbar_rhythm": with_rhythm,
        "subbar_rhythm_bins": SUBBAR_BINS if with_rhythm else 0,
        "motion_dim": 1,
        "motion_is_placeholder": True,
        "motion_placeholder_reading": (
            "motion.npy is zeros with one channel. The planner never reads "
            "motion; a pooled 151-D vector would be a pose nobody performed. "
            "Only the planner stage may train on this release."),
        "identity": report_ids,
        "reconstruction": {
            "label_frames_compared": totals["label_frames_compared"],
            "label_conflicts": totals["label_conflicts"],
            "music_max_overlap_delta": music_max_overlap_delta,
            "covered_frames": totals["covered_frames"],
        },
        "roundtrip": {
            "frames": totals["roundtrip_frames"],
            "wrong": totals["roundtrip_wrong"],
            "disagreement_rate": (totals["roundtrip_wrong"]
                                  / max(totals["roundtrip_frames"], 1)),
            "mixed_bars": totals["mixed_bars"],
            "reading": (
                "expanding each bar's label back over its frames, against the "
                "per-frame track the frame release carries. Exact in-grid "
                "because labels were assigned per grid segment upstream; the "
                "frames bar resolution cannot represent are the out-of-grid "
                "ones counted under 'filler'."),
        },
        "tail_bars_dropped_uncovered": {
            "bars": totals["bars_dropped_uncovered_tail"],
            "recordings": uncovered_tail,
            "reading": (
                "bars the grid cut but the frame release's last whole window "
                "stops inside. They are dropped rather than truncated."),
        },
        "grid_alignment": {
            "off_beat_bar_starts": totals["off_beat_starts"],
            "beat_count_histogram": {str(k): int(v) for k, v in sorted(beat_counts.items())},
        },
        "bar_shape_by_recording": recording_stats,
        "bar_shape_by_window": window_stats,
        "window_span_seconds": window_span,
        "windows_per_split": {k: len(v) for k, v in sorted(rows.items())},
        "recordings_per_split": {
            split: sum(1 for s in matched if per_recording[s]["split"] == split)
            for split in sorted({split_of[s] for s in matched})},
        "recordings_dropped_too_short": [
            {"sequence_id": s, "bars": n, "split": split_of[s]} for s, n in dropped],
        "music_pooling": pooling_table,
        "filler": filler_block,
    }

    if dry_run:
        return report

    out_root.mkdir(parents=True, exist_ok=True)
    written = {}
    for split, entries in sorted(rows.items()):
        base = out_root / split
        base.mkdir(parents=True, exist_ok=True)
        labels = np.stack([e["labels"] for e in entries]).astype(np.int64)
        music = np.stack([e["music"] for e in entries]).astype(np.float32)
        motion = np.zeros((len(entries), window_bars, 1), dtype=np.float32)
        np.save(str(base / "labels.npy"), labels)
        np.save(str(base / "music.npy"), music)
        np.save(str(base / "motion.npy"), motion)
        with open(str(base / "names.json"), "w") as handle:
            json.dump([e["name"] for e in entries], handle, indent=1)
        groups = [e["group"] for e in entries]
        if any(not isinstance(g, str) or not g.strip() for g in groups):
            raise AssertionError(
                "every window needs a non-empty retrieval group; "
                "AtomicSequenceDataset refuses the sidecar otherwise")
        with open(str(base / "retrieval_groups.json"), "w") as handle:
            json.dump(groups, handle, indent=1)
        written[split] = {"windows": len(entries),
                          "labels": list(labels.shape),
                          "music": list(music.shape),
                          "motion": list(motion.shape)}
    report["written"] = written

    with open(str(out_root / "bars.jsonl"), "w") as handle:
        for row in bars_jsonl:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    # Music statistics refit on the POOLED bars of the train split.  Pooling
    # changes the distribution -- the beat channel's mean goes from 0.066 as a
    # frame one-hot to 0.067 as a rate but its std from 0.246 to 0.012 -- so
    # carrying the frame-level stats across would z-score with a spread 20x too
    # large and put the channel back where MusicNormalization exists to rescue
    # it from.  Fit on train only, the repository's rule for every normalizer.
    train_entries = rows.get("train", [])
    if train_entries:
        train_music = np.concatenate(
            [e["music"] for e in train_entries], axis=0).astype(np.float64)
        mean = train_music.mean(axis=0)
        std = train_music.std(axis=0)
        # Constant channels keep std 1 rather than an epsilon-inflated one --
        # the rule in tools/fit_motion_normalizer.py: scaling up a channel's
        # noise is inventing information.
        std = np.where(std > 0, std, 1.0)
        try:
            import torch

            torch.save({"mean": torch.tensor(mean, dtype=torch.float32),
                        "std": torch.tensor(std, dtype=torch.float32)},
                       str(out_root / "music_stats.pt"))
            report["music_stats"] = {
                "path": str(out_root / "music_stats.pt"),
                "fit_split": "train",
                "bars": int(train_music.shape[0]),
                "mean": [float(x) for x in mean],
                "std": [float(x) for x in std],
            }
        except ImportError:  # pragma: no cover - torch is a hard dep of the repo
            report["music_stats"] = {"path": None,
                                     "reason": "torch unavailable at build time"}

    with open(str(out_root / "bar_release_build.json"), "w") as handle:
        json.dump(report, handle, indent=1, sort_keys=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--release-root",
        default="/cache/atomicdance-assets/scratch/txy_t/release_v3")
    parser.add_argument(
        "--segmentation",
        default="/cache/atomicdance-assets/runs/txy_t_seg_beat4/segmentation.json")
    parser.add_argument(
        "--out", default="/cache/atomicdance-assets/scratch/txy_t/release_bar_v1")
    parser.add_argument("--window-bars", type=int, default=4)
    parser.add_argument("--stride-bars", type=int, default=1)
    parser.add_argument("--beats-per-bar", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=21)
    parser.add_argument("--expected-recordings", type=int, default=270,
                        help="assert the id mapping matched this many recordings; "
                             "0 disables the assertion")
    parser.add_argument(
        "--subbar-rhythm", action="store_true",
        help="append the bar's rhythm SHAPE to its music token: "
             "{} sub-bar means of the onset envelope and the onset-peak count, "
             "{} extra dimensions, making the token 51-D. The bar mean this "
             "release ships keeps 3.5%% of the onset channel's variance and "
             "0.23%% of the beat channel's -- rhythm is the part pooling "
             "destroys. Measured on this release's own source-disjoint split, "
             "ridge predicting a bar's rotation energy on held-out recordings: "
             "bar mean -0.0507 val / +0.0411 test, this +0.0597 / +0.1377, "
             "beyond all 200 permutation draws (null mean -0.003, p95 +0.011). "
             "It does NOT make the atomic label predictable (0.089/0.126 "
             "against majority floors 0.130/0.221)"
             .format(SUBBAR_BINS, SUBBAR_BINS * 2))
    parser.add_argument("--dry-run", action="store_true",
                        help="compute and print every verification number, write nothing")
    args = parser.parse_args()

    out = Path(args.out)
    if not args.dry_run and str(out).startswith("/workspace"):
        raise SystemExit(
            "refusing to write a release under /workspace: that is the "
            "quota-limited NAS (CLAUDE.md 1.2). Write under /cache.")

    report = build(
        args.release_root, args.segmentation, args.out,
        window_bars=args.window_bars, stride_bars=args.stride_bars,
        num_classes=args.num_classes, beats_per_bar=args.beats_per_bar,
        expected_recordings=(args.expected_recordings or None),
        dry_run=args.dry_run, with_rhythm=args.subbar_rhythm)

    shape = report["bar_shape_by_recording"]["all"]
    print("matched recordings   : {}".format(report["identity"]["matched_recordings"]))
    print("bars                 : {}".format(shape["bars"]))
    print("bar-to-bar change    : {:.4f} ({}/{})".format(
        shape["change_rate"], shape["changes"], shape["adjacent_pairs"]))
    print("runs >=3 / >=5       : {:.4f} / {:.4f}".format(
        shape["runs_ge_3"], shape["runs_ge_5"]))
    print("filler bar share     : {:.4f}   (frame, in-grid {:.4f}; all covered {:.4f})".format(
        report["filler"]["bar_share"], report["filler"]["frame_share_in_grid"],
        report["filler"]["frame_share_all_covered"]))
    print("round trip wrong     : {}/{}".format(
        report["roundtrip"]["wrong"], report["roundtrip"]["frames"]))
    print("windows per split    : {}".format(report["windows_per_split"]))
    print("window span seconds  : {}".format(report["window_span_seconds"]))
    print("out                  : {}".format(report["out_root"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
