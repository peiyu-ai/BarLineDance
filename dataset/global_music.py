"""Whole-track music summaries, for the planner conditioning the paper describes.

The paper's planner is "Full-music-awared": ``c_music = Enc(M)`` over the whole
track, and it denoises the whole song's label sequence at once (§3.2, Fig. 2).
Ours is trained on 150-frame windows and its per-frame projection gives it
exactly those five seconds, so nothing in the model can see a chorus arrive.
The measured cost of that is not hypothetical: on the 630-class pair, one
generated segment boundary in five sits exactly on the 150-frame grid -- 30.9x
the uniform expectation, z = +49.4, against 0.76x for ground truth on the same
statistic.

Closing the gap by training on whole sequences is the other route (W2) and it
is expensive twice over: attention is O(n^2), and on AIST it would turn 14,409
training windows into 911 sequences, which is precisely the sample starvation
that cost the 1,286-class vocabulary its planner.  This module takes the cheap
route instead: summarise the whole track into one vector per sequence and
broadcast it to every frame of every window.  The window stays 150, the sample
count does not move, and the model gains song-level context it currently has no
way to see.

**The summary is stitched, not averaged over windows.**  Windows overlap ten
deep, so averaging them weights the middle of a track roughly ten times the
first and last five seconds -- a summary of a different track.  The exact
timeline is reconstructed from ``windows.jsonl``, which records ``start_frame``
and ``end_frame_exclusive`` per window, so this needs no stride constant.  A
stride assumed here and changed in ``materialize_atomic_windows`` there is the
shape of defect this repo has hit four times.

**What this can and cannot prove.**  A whole-track vector is constant within a
song, so it can degenerate into a song id -- and a model that memorises "this
song -> this dance" would improve every same-song metric while learning nothing
about music.  Two things guard that.  The split is source-disjoint, so a gain on
unheard sequences is real; and ``shuffle_seed`` permutes the summaries *between*
sequences, so the same training run can be repeated with the vector carrying
another track's summary.  If the gain survives the shuffle, the vector is not
measuring music, and the shuffled run is the number that says so.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The same parser four other tools share.  "Which sequences belong to one song"
# has to mean the same thing here as it does in R-precision and gate v2, or the
# null is drawn over a different partition than the metrics are reported on.
from tools.eval_r_precision import music_key  # noqa: E402


class GlobalMusicError(RuntimeError):
    pass


def _window_records(root: Path, split: str) -> Tuple[List[dict], str]:
    """The split's window records, and which AXIS their spans are counted on.

    A frame release records ``windows.jsonl`` and its music rows are per-frame,
    so the track timeline is rebuilt in frames.  A BAR release records
    ``bars.jsonl`` and its music rows are one vector per bar, so the same
    reconstruction has to run on the bar axis -- ``bar_index_start`` /
    ``bar_index_end_exclusive`` -- or a 4-row window would be written into a
    230-frame span and the summary would be over mostly zeros.  Both files
    carry every field this needs; only the axis differs, and it is returned
    rather than guessed so the caller cannot mix them.
    """
    path = Path(root) / "windows.jsonl"
    if path.is_file():
        axis = ("start_frame", "end_frame_exclusive")
    else:
        path = Path(root) / "bars.jsonl"
        axis = ("bar_index_start", "bar_index_end_exclusive")
    if not path.is_file():
        raise GlobalMusicError(
            "{} has neither windows.jsonl nor bars.jsonl, so the whole-track "
            "timeline cannot be reconstructed; averaging the overlapping "
            "windows instead would summarise a differently-weighted "
            "track".format(root))
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("split")) == split:
            rows.append(row)
    if not rows:
        raise GlobalMusicError("{} records no window in split {!r}".format(path, split))
    missing = [k for k in axis if k not in rows[0]]
    if missing:
        raise GlobalMusicError(
            "{} records windows without {}, so their span on the {} axis is "
            "unknown".format(path, ", ".join(missing), axis[0]))
    return rows, axis


def derange_by_song(per_sequence: np.ndarray, sequences: List[str], seed: int) -> np.ndarray:
    """Reassign whole-track summaries **between songs**, keeping their granularity.

    The obvious null -- derange the summaries across sequences -- is the wrong
    one, and the training loss says so before any metric does.  The honest
    summary is largely *shared* within a song (AIST val: 228 sequences, 30
    distinct summaries), so a per-sequence derangement hands the sequences of
    one song several different vectors.  That is a **finer** conditioning signal
    than the thing it is nulling: it is a better sequence id.  Measured at epoch
    100 of the 630-class arm, the per-sequence-shuffled run reached mean loss
    0.169 against the honest run's 0.459 -- the null fit the training set nearly
    three times better.  A null that is easier to fit than the condition it
    nulls is not a null; it is a second, stronger condition.

    So the permutation is applied to *songs*.  Each song's sequences receive the
    summaries of another song's sequences, matched by rank within the group, so
    the number of distinct vectors a song's sequences see is preserved and only
    the mapping from song to summary is broken.  What survives that is not the
    music.

    A corpus whose names carry no song falls back to a per-sequence derangement,
    and the caller is told, because on such a corpus this control is the weaker
    one described above rather than the intended one.
    """
    rng = np.random.default_rng(seed)
    groups: Dict[str, List[int]] = {}
    for position, name in enumerate(sequences):
        groups.setdefault(music_key(name) or name, []).append(position)
    keys = sorted(groups)
    if len(keys) < 2:
        raise GlobalMusicError(
            "{} song group(s): a summary cannot be reassigned between songs when "
            "there is one".format(len(keys)))
    order = np.arange(len(keys))
    for _ in range(1000):
        rng.shuffle(order)
        if not np.any(order == np.arange(len(keys))):
            break
    else:
        raise GlobalMusicError("could not draw a derangement of the songs")

    shuffled = per_sequence.copy()
    for index, key in enumerate(keys):
        donor = groups[keys[order[index]]]
        for rank, position in enumerate(groups[key]):
            shuffled[position] = per_sequence[donor[rank % len(donor)]]
    return shuffled


def track_summaries(root: str, split: str, music: Optional[np.ndarray] = None,
                    shuffle_seed: Optional[int] = None
                    ) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    """Per-window whole-track summaries: ``[N_windows, 2 * music_dim]``.

    Returns the table indexed the same way the release's arrays are, so a
    dataset can attach row ``i`` to window ``i`` with no lookup, plus the
    sequence order and the window-to-sequence map for reporting.
    """
    root = Path(root)
    if music is None:
        music = np.load(str(root / split / "music.npy"), mmap_mode="r")
    rows, (span_start, span_end) = _window_records(root, split)
    if len(rows) != len(music):
        raise GlobalMusicError(
            "{} records {} windows for split {} but the array holds {}".format(
                root, len(rows), split, len(music)))

    by_sequence: Dict[str, List[dict]] = {}
    for row in rows:
        by_sequence.setdefault(str(row["recording_id"]), []).append(row)
    sequences = sorted(by_sequence)

    music_dim = int(music.shape[2])
    per_sequence = np.zeros((len(sequences), 2 * music_dim), dtype=np.float32)
    for position, name in enumerate(sequences):
        windows = by_sequence[name]
        length = max(int(row[span_end]) for row in windows)
        timeline = np.zeros((length, music_dim), dtype=np.float64)
        seen = np.zeros(length, dtype=bool)
        for row in windows:
            start = int(row[span_start])
            end = int(row[span_end])
            timeline[start:end] = np.asarray(music[int(row["array_index"])][: end - start])
            seen[start:end] = True
        # A gap would make the mean an average over an unstated subset of the
        # track.  Windows tile every sequence by construction, so a hole means
        # the record and the array disagree about what was materialised.
        if not seen.all():
            raise GlobalMusicError(
                "{}: {} of {} steps are covered by no window, so the track "
                "summary would be over an unstated subset".format(
                    name, int((~seen).sum()), length))
        per_sequence[position] = np.concatenate(
            (timeline.mean(axis=0), timeline.std(axis=0))).astype(np.float32)

    if shuffle_seed is not None:
        per_sequence = derange_by_song(per_sequence, sequences, shuffle_seed)

    index_of = {name: position for position, name in enumerate(sequences)}
    table = np.zeros((len(rows), 2 * music_dim), dtype=np.float32)
    for row in rows:
        table[int(row["array_index"])] = per_sequence[index_of[str(row["recording_id"])]]
    return table, sequences, index_of
