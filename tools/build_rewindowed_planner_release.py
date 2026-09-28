#!/usr/bin/env python3
"""Re-cut the T-line planner release into **longer frame windows**.

WHY THIS EXISTS
===============
``docs/DANCE_QUALITY_DEFECTS.md`` section 27 records the defect: ground truth
changes label from one bar to the next **78.4%** of the time; the shipping
planner reaches 58.3% and the best of twelve checkpoints 65.6%, so the planner's
label runs are too long, retrieval stretches one prototype across several bars,
and the dancer plays in slow motion.

Two causes are confounded in the obvious fix:

  (a) CONTEXT LENGTH.  The corpus is cut on a 4-beat grid with median bar
      2.00 s, so ``release_v3``'s 150-frame window spans ~2.5 bars -- about two
      real decisions per denoised sequence.  The paper's planner is
      "Full-music-awared" and denoises the WHOLE song at once (§3.2; see
      ``dataset/global_music.py``), which on AIST++ segments of ~0.9 s would be
      hundreds of decisions with phrase structure visible inside the sequence.
  (b) TOKEN RESOLUTION.  On the per-frame target adjacent frames differ on only
      9,874 / 794,021 = 1.24% of positions, so 98.76% of emitted tokens are
      "same as the previous frame".

``tools/build_bar_planner_release.py`` changes BOTH axes at once (one token =
one bar), so whatever it shows it cannot say which axis mattered.  **This tool
holds the token axis fixed at one token = one frame and varies only how many
frames one denoised sequence covers** -- a dose-response.  If the bar-change
metric improves monotonically with window length, the cause is (a), and the fix
is a training-window change rather than a change to what a token means.

WHAT IT WRITES
==============
A **per-file** release ``AtomicSequenceDataset`` reads
(``dataset/atomic_dataset.py``, the second layout):

    <root>/<split>/motion/<name>.npy    (T, 1)   PLACEHOLDER -- see below
    <root>/<split>/music/<name>.npy     (T, 35)  float32
    <root>/<split>/labels/<name>.npy    (T,)     int64

plus these sidecars beside the splits:

    windows.jsonl                  one row per emitted window: recording, split,
                                   start_frame, end_frame_exclusive, retrieval
                                   group, and the source release's provenance
                                   hashes carried forward.
    music_stats.pt                 {"mean", "std"} over this root's own **train**
                                   split, for ``train_atomic.py --music-stats``.
    rewindowed_release_build.json   provenance + every verification number.

**Why per-file and not the indexed layout.**  ``--window-frames full`` emits one
window per recording and recordings are 345-705 frames, so the indexed layout --
which requires ``motion.npy``/``music.npy``/``labels.npy`` to be one rectangular
array with equal lengths -- cannot hold it without inventing padding the loader
has no mask for.  ``--window-frames 300`` *could* be indexed.  It is written
per-file anyway so that **the only thing that differs between the arms of this
experiment is the window length**: an indexed 300 against a per-file whole-clip
would put the loader path and the window length on the same axis, which is the
confound this whole tool exists to remove.  ``--window-frames 150`` reproduces
``release_v3``'s own windowing in this layout and is the arm that prices the
layout change itself.

**What the per-file layout gives up, stated rather than discovered later.**
``AtomicSequenceDataset`` *refuses* ``--global-music`` on a per-file root
(``dataset/atomic_dataset.py``: "a per-file layout with no windows.jsonl, so a
whole-track music summary cannot be reconstructed"), and the refusal is raised
before anything looks for a file, so **the ``windows.jsonl`` this tool writes
does not lift it**.  It is written for provenance and for this tool's own
verification pass.  A global-music arm on a rewindowed root therefore needs
either the indexed layout or a loader change, and that is deliberate: the
whole-track summary exists precisely to buy song-level context *without*
lengthening the window (its docstring: turning 14,409 windows into 911 sequences
"is precisely the sample starvation that cost the 1,286-class vocabulary its
planner"), so switching it on inside a window-length dose-response would put
both treatments on one arm.

**No ``build.json``.**  ``train_atomic.py``'s ``validate_training_data_root``
requires ``build.json`` to list ``artifacts.splits.<split>.{motion,music,labels,
label_valid_mask,names,retrieval_groups}.npy/.json`` with matching hashes -- the
indexed layout's files, which a per-file root does not have.  Writing one would
be a gate that passes by lying (CLAUDE.md §2).  With no ``build.json`` the
trainer takes its documented legacy path and stamps the run
``release_contract_validated: false, headline_eligible: false``.  That is the
correct label for a diagnostic release and it should stay attached to it.  It
also means the trainer will not check split disjointness for you; this tool
asserts it instead and prints the per-split recording counts.

THE MOTION ARRAY IS A PLACEHOLDER AND THE PLANNER NEVER READS IT
================================================================
``train_atomic.py`` line 1060 calls
``model(batch["labels"], batch["music"], batch["padding_mask"], ...)`` -- the
planner step never touches ``batch["motion"]``.  ``--motion placeholder``
(the default) therefore writes **(T, 1) zeros**: valid for the loader
(``motion.ndim == 2`` and frame counts must agree) and 1/151 of the bytes.  It
is one channel of zeros rather than a plausible-looking 151-D array so that no
future reader can mistake it for motion.  **Only the planner stage may train on
a placeholder root; a completion run would silently be conditioned on zeros.**

Unlike the bar release, the real motion here is *exactly* recoverable -- the
window length changed but the frame grid did not -- so ``--motion real``
reconstructs the true normalized 151-D motion from the source release by the
same first-writer-wins pass and writes that instead.  It costs about 8.5x the
disk (151 channels instead of 1) and buys nothing for a planner arm, hence the
default.  The build report records which one was written under
``motion.kind``, so a consumer can tell without opening an array.

HOW THE DATA IS REWINDOWED
==========================
The source release's windows overlap ten deep (150 frames, stride 15), so the
continuous track has to be reassembled before it can be cut differently.  For
each recording, every source window's labels/music are laid into a per-frame
array at its ``start_frame``, honouring ``label_valid_mask``, **first writer
wins -- and every later writer is compared against what is already there rather
than skipped**, so two overlapping windows that disagree raise instead of the
disagreement vanishing into the overlap.  The recording's covered length is
``max(end_frame_exclusive)``; the source release stops at its last whole
150-frame window, so a few frames of the underlying clip are outside every
window and outside this release too, exactly as they were before.

Then the track is cut again at ``--window-frames`` / ``--stride-frames``.
Because every source length satisfies ``L = 150 + 15k``, a 300-frame window at
stride 15 ends exactly on ``L``; the tool asserts full coverage rather than
assuming it.

DECISIONS, EACH WITH THE MEASUREMENT THAT SETTLED IT
====================================================
1. STRIDE 15, UNCHANGED, AT EVERY WINDOW LENGTH.  Sample starvation is the known
   confound with window length in this project, so the stride is chosen to keep
   the sample count as high as the window length allows, and to keep the window
   *starts* a subset of the source release's starts.  Measured on
   ``release_v3`` (270 recordings, 345-705 frames, median 480):

       window   stride   train/val/test windows   recordings kept
         150      15         5329/727/497            226/26/18
         300      15         3069/467/317            226/26/18
         300      30         1601/240/163            226/26/18
        full       -          226/ 26/ 18            226/26/18

   Holding the overlap *ratio* instead (stride 30 for a 300-frame window) would
   have cost another 48% of the samples for nothing this experiment needs.
   **The whole-clip arm has 226 training sequences against 5,329** -- a 23x drop
   that is a real confound with window length, not a detail.  It is the same
   trade the paper's whole-song denoising makes, and the training stage has to
   handle it (more epochs at the same step count, and no claim that a whole-clip
   loss curve is comparable to a 150-frame one step for step).
2. NO RE-SPLITTING.  Split is taken per recording from the source
   ``windows.jsonl``, asserted unique per recording, and asserted identical in
   the output.  Nothing here decides a split.
3. MUSIC STATISTICS ARE REFIT PER ROOT.  ``MusicNormalization``'s mean/std are
   buffers inside the checkpoint and load strictly, so an arm must carry its
   own.  Rewindowing does not change the *frames*, only how often each frame is
   sampled by an overlapping window, so the statistics are expected to move very
   little -- the report gives the measured per-channel shift against the source
   release rather than asserting that it is small.

VERIFICATION, ALL RECOMPUTED ON EVERY RUN INTO THE REPORT
=========================================================
* IDENTITY.  Every recording's per-frame track is reconstructed *back* out of
  the new root and compared against the one reconstructed from the source
  release: ``music_max_abs_delta`` and ``label_disagreement_rate`` must be 0.
  Anything else means the rewindowing corrupted the data and everything
  downstream of it is void.
* SPLIT.  No recording changes split; per-split recording counts are reported.
* THE HEADLINE STATISTIC SURVIVES.  The ground-truth bar-to-bar change rate is
  recomputed from the NEW root's labels against the 4-beat grid in
  ``segmentation.json`` and checked against 78.4% within ``--bar-change-tol``.
  If it does not reproduce, the bar alignment or the reconstruction is wrong and
  that is the thing to chase, not the training.

USAGE
=====
    python3 tools/build_rewindowed_planner_release.py \
        --release-root /cache/atomicdance-assets/scratch/txy_t/release_v3 \
        --segmentation /cache/atomicdance-assets/runs/txy_t_seg_beat4/segmentation.json \
        --window-frames 300 --stride-frames 15 \
        --out /cache/atomicdance-assets/scratch/txy_t/release_ctx_w300
"""

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

MUSIC_DIM = 35
MOTION_DIM = 151
FILLER_LABEL = 0
FPS = 30.0
SPLITS = ("train", "val", "test")
GROUND_TRUTH_BAR_CHANGE = 0.784


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


def window_name(recording_id, start_frame, whole_recording):
    """Filename stem for one emitted window.

    The whole-clip arm names a file after the recording itself: there is exactly
    one window and a ``_w000000`` suffix would suggest otherwise.
    """
    if whole_recording:
        return recording_id
    return "{}_w{:06d}".format(recording_id, int(start_frame))


# ---------------------------------------------------------- source reading ---

def load_release_windows(release_root):
    """``windows.jsonl`` grouped by recording, with the split asserted unique."""
    release_root = Path(release_root)
    windows = defaultdict(list)
    split_of = {}
    group_of = {}
    with open(str(release_root / "windows.jsonl"), "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sequence = row["sequence_id"]
            if sequence in split_of and split_of[sequence] != row["split"]:
                raise ValueError(
                    "recording {} is in two splits ({} and {}); a rewindowed "
                    "release must not move a recording between splits".format(
                        sequence, split_of[sequence], row["split"]))
            split_of[sequence] = row["split"]
            group_of.setdefault(sequence, row.get("retrieval_group_id"))
            windows[sequence].append(row)
    return dict(windows), split_of, group_of


def open_indexed_arrays(release_root, want_motion=False):
    """Memmap the indexed release's per-split arrays.

    Returns ``{split: (labels, valid_mask, music, motion_or_None)}``.  Splits the
    release does not carry are simply absent; the caller only ever asks for the
    split a window declares.
    """
    release_root = Path(release_root)
    arrays = {}
    for split in SPLITS:
        directory = release_root / split
        if not (directory / "labels.npy").is_file():
            continue
        labels = np.load(str(directory / "labels.npy"), mmap_mode="r")
        music = np.load(str(directory / "music.npy"), mmap_mode="r")
        mask_path = directory / "label_valid_mask.npy"
        if mask_path.is_file():
            valid = np.load(str(mask_path), mmap_mode="r")
        else:
            valid = np.ones(labels.shape, dtype=bool)
        motion = None
        if want_motion:
            motion = np.load(str(directory / "motion.npy"), mmap_mode="r")
        arrays[split] = (labels, valid, music, motion)
    if not arrays:
        raise FileNotFoundError("no indexed split arrays under {}".format(release_root))
    return arrays


def reconstruct_tracks(windows, arrays, num_channels=MUSIC_DIM, want_motion=False,
                       motion_dim=MOTION_DIM):
    """Lay one recording's windows into continuous per-frame tracks.

    First writer wins, and every later writer is **compared** against what is
    already there rather than skipped, so two overlapping windows that disagree
    raise ``ValueError`` instead of the disagreement disappearing into the
    overlap.  Frames no window covers keep label -1 and are reported as
    uncovered; on a release whose windows tile from frame 0 there are none, and
    the caller asserts that.

    ``arrays`` maps split -> ``(labels, valid_mask, music, motion_or_None)``.
    """
    length = max(int(w["end_frame_exclusive"]) for w in windows)
    labels = np.full(length, -1, dtype=np.int64)
    music = np.zeros((length, num_channels), dtype=np.float64)
    motion = np.zeros((length, motion_dim), dtype=np.float64) if want_motion else None
    covered = np.zeros(length, dtype=bool)
    label_compared = label_conflicts = 0
    music_max_delta = 0.0
    for window in sorted(windows, key=lambda w: int(w["start_frame"])):
        split = window["split"]
        index = int(window["array_index"])
        start = int(window["start_frame"])
        stop = int(window["end_frame_exclusive"])
        source_labels, source_valid, source_music, source_motion = arrays[split]
        row_labels = np.asarray(source_labels[index], dtype=np.int64)
        row_valid = np.asarray(source_valid[index], dtype=bool)
        row_music = np.asarray(source_music[index], dtype=np.float64)
        if row_labels.shape[0] != stop - start:
            raise ValueError("window {} declares {} frames but carries {}".format(
                window.get("window_id"), stop - start, row_labels.shape[0]))
        existing = labels[start:stop]
        seen = (existing >= 0) & row_valid
        label_compared += int(seen.sum())
        conflicts = int(np.sum(existing[seen] != row_labels[seen]))
        if conflicts:
            raise ValueError(
                "overlapping source windows disagree on {} label frames in "
                "{}; the release is not internally consistent and nothing "
                "downstream of it can be trusted".format(
                    conflicts, window.get("sequence_id")))
        label_conflicts += conflicts
        seen_music = covered[start:stop]
        if seen_music.any():
            music_max_delta = max(music_max_delta, float(np.max(np.abs(
                music[start:stop][seen_music] - row_music[seen_music]))))
        fresh = (existing < 0) & row_valid
        existing[fresh] = row_labels[fresh]
        fresh_music = ~seen_music
        music[start:stop][fresh_music] = row_music[fresh_music]
        if want_motion:
            row_motion = np.asarray(source_motion[index], dtype=np.float64)
            motion[start:stop][fresh_music] = row_motion[fresh_music]
        covered[start:stop] = True
    diagnostics = {
        "frames": int(length),
        "covered_frames": int(covered.sum()),
        "label_frames_compared": label_compared,
        "label_conflicts": label_conflicts,
        "music_max_overlap_delta": music_max_delta,
        "uncovered_frames": int(length - covered.sum()),
    }
    return labels, music, motion, diagnostics


# ------------------------------------------------------------- rewindowing ---

def window_starts(num_frames, window_frames, stride_frames):
    """Frame indices a window may start at; empty when the recording is short.

    ``window_frames is None`` means one window over the whole recording.
    """
    if window_frames is None:
        return [0]
    if window_frames < 2:
        raise ValueError("a window shorter than 2 frames carries no transition")
    if stride_frames < 1:
        raise ValueError("stride_frames must be >= 1")
    if num_frames < window_frames:
        return []
    return list(range(0, num_frames - window_frames + 1, stride_frames))


def coverage_of(starts, window_frames, num_frames):
    """Frames of ``[0, num_frames)`` that at least one emitted window covers."""
    covered = np.zeros(num_frames, dtype=bool)
    for start in starts:
        stop = num_frames if window_frames is None else start + window_frames
        covered[start:stop] = True
    return covered


# ---------------------------------------------------------- the bar metric ---

def bar_shape_stats(bar_label_sequences):
    """The section 27.2 columns, computed the way section 27.2 computed them.

    ``change_rate`` counts adjacent bar pairs *within* a recording that carry
    different labels.  ``runs_ge_3`` / ``runs_ge_5`` are fractions of **runs**,
    not of bars -- that is the reading which reproduces the published ground
    truth 3.7% / 0.5%; the bar-weighted alternative does not.  ``filler_share``
    is the share of **bars** labelled 0 and is not the frame-level filler share.
    Deliberately identical in definition to
    ``tools/build_bar_planner_release.py``'s function of the same name so the two
    experiments' numbers can be put in one table.
    """
    sequences = [list(map(int, sequence)) for sequence in bar_label_sequences]
    changes = pairs = 0
    runs = []
    labels = []
    for sequence in sequences:
        labels.extend(sequence)
        for previous, current in zip(sequence, sequence[1:]):
            pairs += 1
            changes += int(previous != current)
        index = 0
        while index < len(sequence):
            end = index
            while end + 1 < len(sequence) and sequence[end + 1] == sequence[index]:
                end += 1
            runs.append(end - index + 1)
            index = end + 1
    runs = np.asarray(runs, dtype=np.int64) if runs else np.zeros(0, dtype=np.int64)
    histogram = Counter(labels)
    total = len(labels)
    return {
        "recordings": len(sequences),
        "bars": total,
        "adjacent_pairs": pairs,
        "changes": changes,
        "change_rate": (changes / pairs) if pairs else None,
        "runs": int(runs.size),
        "runs_ge_3": float((runs >= 3).mean()) if runs.size else None,
        "runs_ge_5": float((runs >= 5).mean()) if runs.size else None,
        "longest_run": int(runs.max()) if runs.size else None,
        "filler_share": (histogram[FILLER_LABEL] / total) if total else None,
        "distinct_classes": len(histogram),
        "top_class_share": (max(histogram.values()) / total) if total else None,
    }


def bar_labels_of(label_track, segments, num_classes):
    """Majority label per bar, over the bars the track actually covers.

    Bars that end past the track are dropped and counted, not clipped:
    ``track[a:b]`` on a short track silently returns fewer frames and every
    downstream check then passes on a bar that is not the bar the grid cut
    (CLAUDE.md §2.2 -- suspect the instrument).
    """
    bar_labels = []
    dropped = 0
    for segment in segments:
        start = int(segment["start"])
        stop = int(segment["end"])
        if stop > label_track.shape[0]:
            dropped += 1
            continue
        frames = label_track[start:stop]
        if frames.size == 0 or int(frames.min()) < 0:
            dropped += 1
            continue
        bar_labels.append(int(np.argmax(np.bincount(frames, minlength=num_classes))))
    return bar_labels, dropped


# ------------------------------------------------------------------ output ---

def write_window(out_root, split, name, labels, music, motion):
    """Write one window's three ``.npy`` files in the per-file layout."""
    for kind, array in (("labels", labels), ("music", music), ("motion", motion)):
        directory = Path(out_root) / split / kind
        directory.mkdir(parents=True, exist_ok=True)
        np.save(str(directory / "{}.npy".format(name)), array)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ------------------------------------------------------------------- build ---

def build(release_root, out_root, window_frames, stride_frames=15,
          segmentation_path=None, motion_kind="placeholder", num_classes=21,
          expected_recordings=None, bar_change_tolerance=0.01,
          bar_change_reference=GROUND_TRUTH_BAR_CHANGE, dry_run=False):
    """Write the rewindowed release and return the verification report.

    ``window_frames is None`` means one window per recording (the whole-clip
    arm).  Returns a dict; the caller writes it to
    ``rewindowed_release_build.json``.
    """
    release_root = Path(release_root)
    out_root = Path(out_root)
    want_motion = motion_kind == "real"
    if motion_kind not in ("placeholder", "real"):
        raise ValueError("motion_kind must be 'placeholder' or 'real'")

    source_windows, split_of, group_of = load_release_windows(release_root)
    arrays = open_indexed_arrays(release_root, want_motion=want_motion)
    if expected_recordings is not None and len(source_windows) != expected_recordings:
        raise AssertionError(
            "expected {} recordings in {}/windows.jsonl, found {}".format(
                expected_recordings, release_root, len(source_windows)))

    grids = {}
    if segmentation_path:
        with open(str(segmentation_path), "r") as handle:
            segmentation = json.load(handle)
        for record in segmentation["records"]:
            grids[recording_id_from_segmentation(record["sequence"])] = record

    report = {
        "tool": "tools/build_rewindowed_planner_release.py",
        "schema_version": "rewindowed-frame-window-release-v1",
        "source_release": str(release_root),
        "out_root": str(out_root),
        "layout": "per_file",
        "token": "one token = one frame (unchanged from the source release)",
        "window_frames": window_frames,
        "stride_frames": None if window_frames is None else stride_frames,
        "motion": {
            "kind": motion_kind,
            "dim": MOTION_DIM if want_motion else 1,
            "note": ("real normalized 151-D motion, reconstructed from the source "
                     "release" if want_motion else
                     "PLACEHOLDER zeros; the planner step never reads batch['motion'] "
                     "(train_atomic.py:1060). A completion run on this root would be "
                     "silently conditioned on zeros."),
        },
        "identity": {},
        "splits": {},
        "counts": {},
        "bar_metric": {},
        "recordings": {},
    }

    # ------------------------------------------------------------------ cut --
    per_split_names = defaultdict(list)
    window_rows = []
    source_tracks = {}
    recording_rows = {}
    dropped_short = []
    label_frames_compared = 0
    music_overlap_delta = 0.0

    for recording in sorted(source_windows):
        split = split_of[recording]
        labels, music, motion, diagnostics = reconstruct_tracks(
            source_windows[recording], arrays, want_motion=want_motion)
        if diagnostics["uncovered_frames"]:
            raise AssertionError(
                "{} has {} frames no source window covers; the source windows do "
                "not tile it and a rewindowed cut would invent labels".format(
                    recording, diagnostics["uncovered_frames"]))
        label_frames_compared += diagnostics["label_frames_compared"]
        music_overlap_delta = max(music_overlap_delta,
                                  diagnostics["music_max_overlap_delta"])
        source_tracks[recording] = (labels, music)
        length = int(labels.shape[0])
        starts = window_starts(length, window_frames, stride_frames)
        if not starts:
            dropped_short.append({"recording": recording, "frames": length})
            continue
        covered = coverage_of(starts, window_frames, length)
        if not bool(covered.all()):
            raise AssertionError(
                "{}: window {} stride {} leaves {} of {} frames in no window; "
                "the identity check would then compare a track that is missing "
                "data against one that is not".format(
                    recording, window_frames, stride_frames,
                    int((~covered).sum()), length))
        recording_rows[recording] = {"split": split, "frames": length,
                                     "windows": len(starts)}
        for start in starts:
            stop = length if window_frames is None else start + window_frames
            name = window_name(recording, start, window_frames is None)
            if not dry_run:
                if want_motion:
                    window_motion = motion[start:stop].astype(np.float32)
                else:
                    window_motion = np.zeros((stop - start, 1), dtype=np.float32)
                write_window(out_root, split, name,
                             labels[start:stop].astype(np.int64),
                             music[start:stop].astype(np.float32),
                             window_motion)
            per_split_names[split].append(name)
            window_rows.append({
                "name": name,
                "split": split,
                "sequence_id": recording,
                "recording_id": recording,
                "retrieval_group_id": group_of.get(recording),
                "start_frame": int(start),
                "end_frame_exclusive": int(stop),
                "length": int(stop - start),
                "source_release": str(release_root),
                "labels_path": "{}/labels/{}.npy".format(split, name),
                "music_path": "{}/music/{}.npy".format(split, name),
                "motion_path": "{}/motion/{}.npy".format(split, name),
                "motion_kind": motion_kind,
            })

    report["counts"] = {
        "source_recordings": len(source_windows),
        "recordings_kept": len(recording_rows),
        "recordings_dropped_short": dropped_short,
        "windows": {split: len(names) for split, names in sorted(per_split_names.items())},
        "total_windows": len(window_rows),
        "source_windows": {
            split: sum(1 for rows in source_windows.values() for row in rows
                       if row["split"] == split)
            for split in SPLITS},
        "frames_emitted": int(sum(row["length"] for row in window_rows)),
        "frames_unique": int(sum(row["frames"] for row in recording_rows.values())),
    }
    report["splits"] = {
        "recordings_per_split": dict(Counter(
            row["split"] for row in recording_rows.values())),
        "source_recordings_per_split": dict(Counter(split_of.values())),
    }
    report["identity"]["source_overlap"] = {
        "label_frames_compared": label_frames_compared,
        "label_conflicts": 0,
        "music_max_overlap_delta": music_overlap_delta,
        "note": "overlapping windows of the SOURCE release agreeing; a conflict raises",
    }

    if dry_run:
        return report, window_rows, source_tracks, grids

    # ---------------------------------------------------------- sidecars ----
    with open(str(out_root / "windows.jsonl"), "w") as handle:
        for row in window_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    # ------------------------------------------------------- verification ---
    report["identity"]["rewindow"] = verify_identity(
        out_root, window_rows, source_tracks)
    report["splits"]["changed"] = verify_splits(window_rows, split_of)
    if grids:
        report["bar_metric"] = verify_bar_metric(
            out_root, window_rows, grids, num_classes, bar_change_tolerance,
            reference=bar_change_reference)
    report["music_stats"] = write_music_stats(out_root, window_rows, release_root)
    return report, window_rows, source_tracks, grids


def read_root_tracks(out_root, window_rows):
    """Reconstruct per-recording tracks from the WRITTEN root, first writer wins.

    Deliberately reads the files back off disk rather than reusing the arrays in
    memory: an identity check that never opens the artifact proves the code that
    was about to write it, not the artifact.
    """
    out_root = Path(out_root)
    by_recording = defaultdict(list)
    for row in window_rows:
        by_recording[row["recording_id"]].append(row)
    tracks = {}
    for recording, rows in by_recording.items():
        length = max(int(row["end_frame_exclusive"]) for row in rows)
        labels = np.full(length, -1, dtype=np.int64)
        music = np.zeros((length, MUSIC_DIM), dtype=np.float64)
        covered = np.zeros(length, dtype=bool)
        for row in sorted(rows, key=lambda r: int(r["start_frame"])):
            start = int(row["start_frame"])
            stop = int(row["end_frame_exclusive"])
            window_labels = np.load(str(out_root / row["labels_path"]))
            window_music = np.load(str(out_root / row["music_path"]))
            fresh = ~covered[start:stop]
            labels[start:stop][fresh] = window_labels[fresh]
            music[start:stop][fresh] = window_music[fresh]
            covered[start:stop] = True
        tracks[recording] = (labels, music, covered)
    return tracks


def verify_identity(out_root, window_rows, source_tracks):
    """Same data, cut differently: max |music| delta and label disagreement.

    Compared only on the frames the new root covers -- a recording dropped for
    being shorter than the window contributes nothing, and its absence is a
    count in ``counts.recordings_dropped_short``, not a silent zero here.
    """
    tracks = read_root_tracks(out_root, window_rows)
    music_max = 0.0
    label_compared = label_wrong = 0
    frames_compared = 0
    per_recording_worst = None
    for recording, (labels, music, covered) in sorted(tracks.items()):
        source_labels, source_music = source_tracks[recording]
        span = labels.shape[0]
        if span > source_labels.shape[0]:
            raise AssertionError(
                "{}: rewindowed track is {} frames, source is {}".format(
                    recording, span, source_labels.shape[0]))
        select = covered
        frames_compared += int(select.sum())
        delta = float(np.max(np.abs(
            music[select] - source_music[:span][select]))) if select.any() else 0.0
        wrong = int(np.sum(labels[select] != source_labels[:span][select]))
        label_compared += int(select.sum())
        label_wrong += wrong
        if delta > music_max:
            music_max = delta
            per_recording_worst = recording
    return {
        "recordings_compared": len(tracks),
        "frames_compared": frames_compared,
        "music_max_abs_delta": music_max,
        "music_worst_recording": per_recording_worst,
        "label_frames_compared": label_compared,
        "label_disagreements": label_wrong,
        "label_disagreement_rate": (label_wrong / label_compared) if label_compared else None,
    }


def verify_splits(window_rows, source_split_of):
    """Assert no recording changed split, and none appears in two output splits."""
    seen = {}
    changed = []
    for row in window_rows:
        recording = row["recording_id"]
        if recording in seen and seen[recording] != row["split"]:
            raise AssertionError(
                "{} appears in two output splits ({}, {})".format(
                    recording, seen[recording], row["split"]))
        seen[recording] = row["split"]
        if source_split_of[recording] != row["split"]:
            changed.append({"recording": recording,
                            "was": source_split_of[recording],
                            "now": row["split"]})
    if changed:
        raise AssertionError(
            "{} recordings changed split, first {}".format(len(changed), changed[:3]))
    return changed


def verify_bar_metric(out_root, window_rows, grids, num_classes, tolerance,
                      reference=GROUND_TRUTH_BAR_CHANGE):
    """Recompute the ground-truth bar-to-bar change rate from the NEW root.

    ``reference`` is a parameter only so a synthetic test corpus -- whose bars
    are chosen by hand and have nothing to do with 78.4% -- can exercise both
    the pass and the fail branch.  Production callers leave it alone.
    """
    tracks = read_root_tracks(out_root, window_rows)
    matched = sorted(set(tracks) & set(grids))
    sequences = []
    dropped_bars = 0
    for recording in matched:
        labels = tracks[recording][0]
        bars, dropped = bar_labels_of(labels, grids[recording]["segments"], num_classes)
        dropped_bars += dropped
        if len(bars) >= 1:
            sequences.append(bars)
    stats = bar_shape_stats(sequences)
    stats["matched_recordings"] = len(matched)
    stats["recordings_without_grid"] = len(set(tracks) - set(grids))
    stats["bars_dropped_past_track_end"] = dropped_bars
    stats["reference_change_rate"] = reference
    stats["tolerance"] = tolerance
    rate = stats["change_rate"]
    stats["reproduces_reference"] = (
        rate is not None and abs(rate - reference) <= tolerance)
    if not stats["reproduces_reference"]:
        raise AssertionError(
            "ground-truth bar-to-bar change on the rewindowed root reads {} "
            "against the recorded {} (tolerance {}); the bar alignment or the "
            "reconstruction is wrong -- chase that before training anything"
            .format(rate, reference, tolerance))
    return stats


def write_music_stats(out_root, window_rows, release_root):
    """Per-channel mean/std over this root's own train split, plus the shift.

    Same payload shape as ``tools/fit_music_normalizer.py``
    (``{"mean", "std"}``) because ``train_atomic.py --music-stats`` loads it and
    ``MusicNormalization`` registers it as a buffer.  That tool cannot be used
    here: it reads ``<root>/train/music.npy``, which a per-file root has no such
    file for.

    The statistics are over the windows **as the loader will draw them**, i.e.
    an overlapping frame counts once per window that contains it -- which is the
    distribution the model actually sees, and the one the source release's stats
    were fit over too.
    """
    import torch

    out_root = Path(out_root)
    count = 0
    total = np.zeros(MUSIC_DIM, np.float64)
    total_squared = np.zeros(MUSIC_DIM, np.float64)
    for row in window_rows:
        if row["split"] != "train":
            continue
        music = np.asarray(np.load(str(out_root / row["music_path"])), np.float64)
        count += music.shape[0]
        total += music.sum(0)
        total_squared += (music * music).sum(0)
    if not count:
        raise AssertionError("no train windows; cannot fit music statistics")
    mean = total / count
    std = np.sqrt(np.maximum(total_squared / count - mean * mean, 0.0))
    payload = {"mean": torch.tensor(mean, dtype=torch.float32),
               "std": torch.tensor(std, dtype=torch.float32)}
    destination = out_root / "music_stats.pt"
    torch.save(payload, str(destination))

    shift = None
    source_music = Path(release_root) / "train" / "music.npy"
    if source_music.is_file():
        array = np.load(str(source_music), mmap_mode="r")
        source_count = 0
        source_total = np.zeros(MUSIC_DIM, np.float64)
        source_squared = np.zeros(MUSIC_DIM, np.float64)
        for start in range(0, len(array), 512):
            chunk = np.asarray(array[start:start + 512], np.float64).reshape(-1, MUSIC_DIM)
            source_count += len(chunk)
            source_total += chunk.sum(0)
            source_squared += (chunk * chunk).sum(0)
        source_mean = source_total / source_count
        source_std = np.sqrt(np.maximum(
            source_squared / source_count - source_mean * source_mean, 0.0))
        safe = np.where(source_std > 0, source_std, 1.0)
        shift = {
            "source_frames": int(source_count),
            "max_abs_mean_shift_in_source_std": float(
                np.max(np.abs(mean - source_mean) / safe)),
            "max_abs_std_ratio_minus_one": float(
                np.max(np.abs(std / safe - 1.0))),
            "worst_mean_channel": int(np.argmax(np.abs(mean - source_mean) / safe)),
            "worst_std_channel": int(np.argmax(np.abs(std / safe - 1.0))),
        }
    return {
        "path": str(destination),
        "sha256": sha256_file(destination),
        "train_frames_pooled": int(count),
        "constant_channels": [int(i) for i in np.flatnonzero(std <= 0)],
        "shift_vs_source_release": shift,
    }


# -------------------------------------------------------------------- main ---

def parse_window_frames(value):
    if str(value).lower() == "full":
        return None
    frames = int(value)
    if frames < 2:
        raise argparse.ArgumentTypeError("--window-frames must be >= 2 or 'full'")
    return frames


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--window-frames", required=True, type=parse_window_frames,
                        help="frames per emitted window, or 'full' for one window "
                             "per recording")
    parser.add_argument("--stride-frames", type=int, default=15,
                        help="ignored when --window-frames is 'full'")
    parser.add_argument("--segmentation", default="",
                        help="4-beat grid segmentation.json; without it the "
                             "bar-to-bar change verification is skipped and the "
                             "report says so")
    parser.add_argument("--motion", choices=("placeholder", "real"),
                        default="placeholder")
    parser.add_argument("--num-classes", type=int, default=21)
    parser.add_argument("--expect-recordings", type=int, default=None)
    parser.add_argument("--bar-change-tol", type=float, default=0.01)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out)
    if not args.dry_run:
        if out_root.exists() and any(out_root.iterdir()):
            raise SystemExit(
                "{} already exists and is not empty; publish to a new directory "
                "rather than mixing two windowings in one root".format(out_root))
        out_root.mkdir(parents=True, exist_ok=True)

    report, window_rows, _, grids = build(
        release_root=args.release_root,
        out_root=out_root,
        window_frames=args.window_frames,
        stride_frames=args.stride_frames,
        segmentation_path=args.segmentation or None,
        motion_kind=args.motion,
        num_classes=args.num_classes,
        expected_recordings=args.expect_recordings,
        bar_change_tolerance=args.bar_change_tol,
        dry_run=args.dry_run,
    )
    if not grids:
        report["bar_metric"] = {"skipped": "no --segmentation given"}
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.dry_run:
        print(text)
        return
    (out_root / "rewindowed_release_build.json").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
