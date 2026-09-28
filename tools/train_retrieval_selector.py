#!/usr/bin/env python3
"""Train the thing that picks WHICH exemplar of a class to paste next.

The rule it replaces is one ``min`` over one number -- see the module docstring
of ``model/retrieval_selector.py`` for why that is the weak point and what the
measurements say the join, not the content, is where the defect lives.  This
file builds the supervision, trains the scorer, and -- more importantly --
runs the four checks that decide whether it is allowed to be believed.

------------------------------------------------------------- SUPERVISION

Free, and that is the point: in the corpus, the true continuation of segment i
IS segment i+1 of the same recording.  A pair is built for every ABUTTING
(``s_i.end == s_{i+1}.start``) pair of non-transition segments, because on the
bar-grid arm 90.9% of plan boundaries are prototype-to-prototype seams with no
transition between them, so that is the join the scorer will actually be asked
about.

Negatives are other members of the positive's own class, drawn from OTHER
retrieval groups and from inside the duration slack band
``max(2, 0.15 * target_length)`` -- the same band ``tempo``, ``phase`` and
``--draft-recurrence-variety`` already draw from, so nothing here narrows the
pool relative to shipped behaviour.

-------------------------------------------------------------- TWO LEAKS

Both were found by writing the check before the training, and both would have
produced a scorer with excellent numbers and no ability.

1. **The positive continues its predecessor exactly** -- same room, same
   facing, same root -- while negatives arrive from other uploads with
   arbitrary world placement.  Closed in ``model/retrieval_selector.align_to``
   plus body-frame travel and joint-only summaries; the invariance is asserted
   in ``tests/test_retrieval_selector.py`` (rigidly spinning and translating a
   candidate must not move its feature row).

2. **``target_length`` must not be the positive's own length**, or the query
   hands over the answer: only the positive would sit at |len - target| = 0.
   At inference the target comes from ``snap_plan_to_bar_grid``, i.e. a whole
   number of the query's own beats, so the target here is quantised the same
   way, off the clip's own beat channel.  The report records
   ``target_equals_positive_length`` -- if that is near 1.0 the quantisation
   is not doing its job and no number below may be quoted.

------------------------------------------------------------ THE CHECKS

* **Baseline.**  The duration rule scored on the same held-out queries.  A
  selector that does not beat it has bought nothing.
* **Power / positive control.**  The true successor is placed in the pool.  If
  the scorer cannot rank it first, the features are insufficient -- that is a
  statement about the features, NOT evidence that context is useless
  (CLAUDE.md 2.1 rule 3).
* **Ablation.**  Re-score with the seam block zeroed.  If recall does not
  fall, the seam is not what was learned and the mechanism story is wrong.
* **Chance.**  Random ranking over the same pools, so "better than the
  baseline" can be read against the width of the pool.

Held-out means val recordings; candidates still come from the TRAIN pool only,
which is what ``IndexedAtomicMotionLibrary`` reads.  The positive is added to
that pool for the check above and is the only val-sourced candidate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from model.retrieval_selector import (  # noqa: E402
    EDGE_FRAMES, FEATURE_DIM, PROFILE_BINS, RetrievalSelector, QueryContext,
    candidate_features_many, describe_segment,
)

BEAT_CHANNEL = 34   # data/audio_extraction/baseline_features.py layout
ONSET_CHANNEL = 0
MIN_SEGMENT_FRAMES = 6      # infer_atomic --plan-min-segment default
TRANSITION = 0


# ------------------------------------------------------------------ corpus --
def read_jsonl(path):
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_corpus(labels_dir, sequences_jsonl, exclude_path, limit=None):
    labels_dir = pathlib.Path(labels_dir)
    excluded = set()
    if exclude_path and os.path.exists(exclude_path):
        for row in read_jsonl(exclude_path):
            key = row.get("sequence") or row.get("sequence_id")
            if key:
                excluded.add(key)
    sequences = {row["sequence_id"]: row for row in read_jsonl(sequences_jsonl)}
    sequences_root = pathlib.Path(sequences_jsonl).resolve().parent
    records = []
    for row in read_jsonl(labels_dir / "labels.jsonl"):
        key = row["sequence_id"]
        if key in excluded or key not in sequences:
            continue
        source = sequences[key]
        records.append({
            "sequence_id": key,
            "split": row["split"],
            "group": row.get("retrieval_group_id") or row.get("recording_id"),
            "labels": labels_dir / row["labels_path"],
            # Resolved against the manifest that names them, not against the
            # process's cwd: sequences.jsonl stores "sequences/<hash>/..." and
            # this file had never been run, so the paths only worked if you
            # happened to start it from the performance tree.
            "motion": sequences_root / source["motion_path"],
            "music": (sequences_root / source["music_path"]
                      if source.get("music_path") else None),
            "frames": int(row["frame_count"]),
        })
        if limit and len(records) >= limit:
            break
    return records, len(excluded)


def beat_grid(music):
    if music is None or music.shape[1] <= BEAT_CHANNEL:
        return np.zeros(0, dtype=np.int64), float("nan")
    beats = np.flatnonzero(music[:, BEAT_CHANNEL] > 0.5).astype(np.int64)
    period = float(np.median(np.diff(beats))) if len(beats) > 1 else float("nan")
    return beats, period


def quantised_target(beats, period, start, length):
    """The span the bar grid would have asked for, in frames.

    Whole beats off the clip's own grid, which is what ``snap_plan_to_bar_grid``
    produces -- and the reason it is not simply ``length`` is leak 2 above.
    """
    if not np.isfinite(period) or period <= 0 or len(beats) < 2:
        return int(length)
    steps = max(1, int(round(length / period)))
    index = int(np.searchsorted(beats, start))
    if 0 <= index < len(beats) and index + steps < len(beats):
        span = int(beats[index + steps] - beats[index])
        if span >= MIN_SEGMENT_FRAMES:
            return span
    return max(MIN_SEGMENT_FRAMES, int(round(steps * period)))


def phase_at(beats, period, frame):
    if not np.isfinite(period) or period <= 0 or len(beats) < 2:
        return float("nan")
    index = int(np.searchsorted(beats, frame, side="right")) - 1
    if index < 0:
        return float("nan")
    return float(((frame - beats[index]) / period) % 1.0)


def onset_track(music):
    if music is None or music.shape[1] <= ONSET_CHANNEL:
        return None
    onset = music[:, ONSET_CHANNEL].astype(np.float64)
    spread = float(onset.std())
    # Relative guard, same rule as MusicPhaseFeatures: for a constant channel
    # float32 rounding leaves std at ~2e-7 and a literal `std > 0` guard would
    # amplify that rounding to full scale.
    if spread <= 1e-6 * max(1.0, float(np.abs(onset).mean())):
        return np.zeros_like(onset)
    return (onset - onset.mean()) / spread


def profile_of(track, start, end, bins=PROFILE_BINS):
    out = np.zeros(bins, dtype=np.float32)
    if track is None or end <= start:
        return out
    span = track[start:min(end, len(track))]
    if not len(span):
        return out
    edges = np.linspace(0, len(span), bins + 1).round().astype(int)
    for i, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        chunk = span[low:max(high, low + 1)]
        out[i] = float(chunk.mean()) if len(chunk) else 0.0
    return out


# -------------------------------------------------------------- describing --
def segments_of(labels):
    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    bounds = [0, *changes.tolist(), len(labels)]
    return [(int(labels[a]), int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])]


class Corpus:
    """Descriptors for every kept segment, plus the per-class candidate lists."""

    def __init__(self, records, normalizer_path, verbose=True):
        stats = torch.load(normalizer_path, map_location="cpu", weights_only=True)
        low = stats["data_min"].float()
        high = stats["data_max"].float()
        span = torch.where(high == low, torch.ones_like(high), high - low)
        # raw = normalized * scale + offset.  Because the features are computed
        # on RAW values, the selector is independent of WHICH release
        # normalizer a corpus carries -- unnormalising undoes whichever one was
        # applied.  That is why a selector trained here transfers to a release
        # that was re-normalised by tools/renormalize_release.py.
        self.scale, self.offset = span / 2.0, low + span / 2.0

        self.records = records
        self.descriptors = []        # SegmentDescriptor per kept segment
        self.meta = []               # (record_index, label, start, end)
        self.by_label = {}           # train-split candidates (what inference reads)
        self.by_label_split = {}     # per split, for an honest held-out pool
        self.record_segments = {}    # record_index -> [segment table index]
        self.record_music = {}       # record_index -> (beats, period, onset)
        started = time.time()
        for index, record in enumerate(records):
            self._ingest(index, record)
            if verbose and index % 1000 == 0:
                print("  described {}/{} recordings, {} segments, {:.0f}s".format(
                    index, len(records), len(self.meta), time.time() - started),
                    flush=True)
        self.label_pool_stats = self._pool_stats()

    def _ingest(self, index, record):
        labels = np.load(record["labels"])
        motion = np.load(record["motion"], mmap_mode="r")
        frames = min(len(labels), len(motion))
        if frames < MIN_SEGMENT_FRAMES + 2:
            return
        labels = labels[:frames]
        music = None
        if record["music"] and os.path.exists(record["music"]):
            music = np.load(record["music"], mmap_mode="r")
            music = np.asarray(music[:frames])
        beats, period = beat_grid(music)
        self.record_music[index] = (beats, period, onset_track(music))

        kept = []
        for label, start, end in segments_of(labels):
            if label == TRANSITION or end - start < MIN_SEGMENT_FRAMES:
                kept.append(None)
                continue
            raw = torch.from_numpy(np.array(motion[start:end], dtype=np.float32))
            raw = raw * self.scale + self.offset
            descriptor = describe_segment(
                raw,
                beat_phase=phase_at(beats, period, start),
                beat_period=period)
            table_index = len(self.descriptors)
            self.descriptors.append(descriptor)
            self.meta.append((index, label, start, end))
            kept.append(table_index)
            self.by_label_split.setdefault(record["split"], {}).setdefault(
                label, []).append(table_index)
            if record["split"] == "train":
                self.by_label.setdefault(label, []).append(table_index)
        self.record_segments[index] = (
            [(label, start, end) for label, start, end in segments_of(labels)], kept)

    def _pool_stats(self):
        """Per class, the mean head contacts and head limb activity of its pool.

        This is how "what comes next" enters the features without naming a
        vocabulary: the next segment's identity is unknown at draft time (it
        has not been retrieved yet), but its CLASS is, and a class's pool has a
        mean way of starting.  Contacts and limb activity are used because both
        are invariant to which way the recording faced.
        """
        stats = {}
        for label, members in self.by_label.items():
            contacts = torch.stack([self.descriptors[m].edges[0, :4] for m in members])
            activity = torch.stack([self.descriptors[m].limb_activity for m in members])
            stats[label] = (contacts.mean(dim=0), activity.mean(dim=0))
        if stats:
            self.default_stats = (
                torch.stack([v[0] for v in stats.values()]).mean(dim=0),
                torch.stack([v[1] for v in stats.values()]).mean(dim=0))
        else:
            self.default_stats = (torch.zeros(4), torch.zeros(4))
        return stats

    def next_stats(self, label):
        return self.label_pool_stats.get(int(label), self.default_stats)


# ------------------------------------------------------------------- pairs --
def build_pairs(corpus, split, rng, negatives, max_pairs=None, pool_split=None):
    """(query, positive, [negatives], positive length, target) tuples.

    ``pool_split`` decides WHERE the distractors come from, and for the
    held-out set that choice is the whole point.  Drawing them from the train
    pool leaves the positive as the only candidate cut from the query's own
    upload, so "which one is from this recording" wins the evaluation without
    the join being learned -- the first smoke run read 0.9933 that way, with
    zero pairs on which the shortcut was unavailable.  Drawing them from the
    query's OWN split puts other members of the class from the same recording
    in the pool, which is what makes the ``hard`` column exist at all.

    Note this makes the held-out set HARDER than inference, where no candidate
    shares the query's upload.  That is the direction a control should err in.
    """
    pairs = []
    groups = [record["group"] for record in corpus.records]
    for record_index, record in enumerate(corpus.records):
        if record["split"] != split:
            continue
        table = corpus.record_segments.get(record_index)
        if table is None:
            continue
        raw_segments, kept = table
        beats, period, onset = corpus.record_music[record_index]
        for position in range(len(raw_segments) - 1):
            here, following = kept[position], kept[position + 1]
            if here is None or following is None:
                continue
            label_next, start_next, end_next = raw_segments[position + 1]
            if raw_segments[position][2] != start_next:      # abutting only
                continue
            positive = corpus.descriptors[following]
            target = quantised_target(beats, period, start_next,
                                      end_next - start_next)
            slack = max(2.0, 0.15 * target)
            pool = corpus.by_label_split.get(pool_split or split, {}).get(
                label_next, ())
            band = [m for m in pool
                    if abs(corpus.descriptors[m].length - target) <= slack]
            # LEAK 2 IS WIDE OPEN ON A BEAT-GRID CORPUS, and closing it by
            # quantising the target does nothing here.  The T line's M1 cuts on
            # the music beat grid (mode: grid, beats_per_segment: 4), so every
            # segment is ALREADY a whole number of beats and ``quantised_target``
            # returns the positive's own length: measured, 99.7% of pairs.  The
            # positive is then the only candidate at |len - target| = 0 and the
            # query hands over the answer.
            #
            # The fix is not to hide the length but to make the negatives cost
            # the same on it -- which is also the task inference actually poses:
            # ``_duration_pick`` takes the minimum and 73.6% of slots reach that
            # minimum with a CROWD (median tie 6, 79.1% with an exact match), so
            # what the selector is asked for in deployment is a choice among
            # candidates already tied on duration.  Train it on exactly that.
            own_gap = abs(corpus.descriptors[following].length - target)
            tied = [m for m in band
                    if abs(corpus.descriptors[m].length - target) == own_gap]
            if len(tied) >= 2:
                band = tied
            here_phase = phase_at(beats, period, start_next)
            options = choose_negatives(corpus, band, groups, record["group"],
                                       target, here_phase, period, negatives, rng)
            if len(options) < 2:
                continue
            after = kept[position + 2] if position + 2 < len(kept) else None
            next_label = (raw_segments[position + 2][0]
                          if position + 2 < len(raw_segments) else TRANSITION)
            next_contacts, next_activity = corpus.next_stats(next_label)
            query = QueryContext(
                previous_tail=corpus.descriptors[here].edges[EDGE_FRAMES:],
                target_length=int(target),
                gap_frames=0,
                beat_phase=phase_at(beats, period, start_next),
                beat_period=float(period),
                onset_mean=float(np.mean(profile_of(onset, start_next,
                                                    start_next + target))),
                onset_profile=torch.from_numpy(
                    profile_of(onset, start_next, start_next + target)),
                beats_in_span=(target / period) if np.isfinite(period) and period > 0
                              else float("nan"),
                next_contacts=next_contacts,
                next_activity=next_activity)
            pairs.append((query, following, options, int(positive.length), int(target)))
            del after
            if max_pairs and len(pairs) >= max_pairs:
                return pairs
    return pairs


def choose_negatives(corpus, band, groups, own_group, target, own_phase,
                     own_period, count, rng):
    """Negatives chosen to remove every shortcut, not sampled uniformly.

    THE SMOKE RUN THAT FORCED THIS.  With uniform negatives drawn from other
    retrieval groups, held-out recall@1 read **0.8467** against a duration
    baseline of 0.2100 -- and the seam-zeroed ablation still read 0.4067, i.e.
    the scorer could find the true successor twice as often as chance with the
    join hidden from it.  There is no honest way to do that, so the features
    were carrying identity, and two channels were:

      * the positive is the ONLY candidate cut from the query's own upload, so
        anything upload-specific (this dancer, this reconstruction's biases,
        this clip's timebase) names it;
      * the positive's own beat phase and beat period are read off the SAME
        song as the query's, so its phase difference is exactly 0 and its
        period ratio exactly 1.0 -- values no other candidate hits by accident.

    Both are removed by making the negatives hard rather than by deleting
    features that are legitimate at inference (a candidate whose tempo matches
    IS a better candidate).  The mix, in priority order: same-recording members
    of the class, then the closest phase matches, then the closest lengths,
    then a uniform fill.  ``tests/test_retrieval_selector.py`` asserts the
    same-recording ones are present whenever the corpus has any.
    """
    if not band:
        return []
    same_recording = [m for m in band if groups[corpus.meta[m][0]] == own_group]
    others = [m for m in band if groups[corpus.meta[m][0]] != own_group]
    if not others:
        return []

    def phase_gap(index):
        phase = corpus.descriptors[index].beat_phase
        if not np.isfinite(phase) or not np.isfinite(own_phase):
            return 1.0
        return abs(((phase - own_phase + 0.5) % 1.0) - 0.5)

    def period_gap(index):
        period = corpus.descriptors[index].beat_period
        if not np.isfinite(period) or not np.isfinite(own_period) or own_period <= 0:
            return 1.0
        return abs(period - own_period) / own_period

    picked, seen = [], set()
    quota = max(count, 4)

    def take(source, how_many):
        for index in source:
            if len(picked) >= quota:
                return
            if index in seen:
                continue
            seen.add(index)
            picked.append(index)
            how_many -= 1
            if how_many <= 0:
                return

    take(sorted(same_recording, key=phase_gap), max(1, quota // 5))
    take(sorted(others, key=lambda i: (phase_gap(i), period_gap(i))), max(1, quota // 4))
    take(sorted(others, key=lambda i: abs(corpus.descriptors[i].length - target)),
         max(1, quota // 4))
    rest = [i for i in others if i not in seen]
    if rest:
        order = rng.permutation(len(rest))
        take([rest[i] for i in order], quota - len(picked))
    return picked


def materialize(corpus, pairs, groups, verbose=True):
    """Feature rows for every pair, once.

    The rows do not depend on the model, so recomputing them every epoch was
    pure waste -- and it is what made the first version 0.2 s per PAIR.  The
    positive is always column 0; ``valid`` masks the padding, ``lengths``
    carries the duration baseline's input, and ``same_group`` marks pools that
    contain another member of the class from the query's own recording (the
    hard subset -- see ``evaluate``).
    """
    width = max(1 + len(negatives) for _, _, negatives, _, _ in pairs)
    features = torch.zeros(len(pairs), width, FEATURE_DIM)
    valid = torch.zeros(len(pairs), width, dtype=torch.bool)
    lengths = torch.zeros(len(pairs), width)
    targets = torch.zeros(len(pairs))
    same_group = torch.zeros(len(pairs), dtype=torch.bool)
    started = time.time()
    for row, (query, positive, negatives, _, target) in enumerate(pairs):
        indices = [positive] + [n for n in negatives if n != positive]
        rows = candidate_features_many(
            query, [corpus.descriptors[i] for i in indices])
        features[row, :len(indices)] = rows
        valid[row, :len(indices)] = True
        lengths[row, :len(indices)] = torch.tensor(
            [float(corpus.descriptors[i].length) for i in indices])
        targets[row] = float(target)
        own = groups[corpus.meta[positive][0]]
        same_group[row] = any(groups[corpus.meta[i][0]] == own for i in indices[1:])
        if verbose and row and row % 5000 == 0:
            print("    {}/{} pairs materialized, {:.0f}s".format(
                row, len(pairs), time.time() - started), flush=True)
    return {"features": features, "valid": valid, "lengths": lengths,
            "targets": targets, "same_group": same_group}


def masked_scores(model, features, valid, drop_seam=False, drop_groups=()):
    flat = model(features.reshape(-1, FEATURE_DIM), drop_seam=drop_seam,
                 drop_groups=drop_groups)
    scores = flat.reshape(features.shape[:2])
    return scores.masked_fill(~valid, -float("inf"))


def recall_from(scores, valid, k, ties_win=False):
    """Positive is column 0.  Returns (recall@1, recall@k) as [P] floats.

    TIES COUNT AGAINST THE MODEL, and that is not a detail -- the first version
    of this function used ``scores > positive``, which counts ties FOR it, and
    the control that dropped every candidate-dependent group read **1.0000**:
    a model whose input is identical for every candidate in a pool scores them
    identically, ties everywhere, and swept the board.  The control caught the
    instrument, which is what controls are for.

    ``ties_win`` restores the old behaviour on purpose, for ONE use: the
    duration baseline genuinely ties a lot (many candidates share a length), so
    its true value lies between the two, and both bounds are reported rather
    than one being chosen.
    """
    positive = scores[:, :1]
    others, other_valid = scores[:, 1:], valid[:, 1:]
    beat = (others > positive) if ties_win else (others >= positive)
    ahead = (beat & other_valid).sum(dim=1)
    return (ahead == 0).float(), (ahead < k).float()


# -------------------------------------------------------------------- main --
def evaluate(model, data, k, drop_seam=False, device="cpu", drop_groups=()):
    """Held-out ranking, plus the same numbers on the HARD subset.

    Hard = pairs whose pool contains at least one other member of the class cut
    from the query's own recording.  On those "which candidate came from this
    upload" cannot name the positive, so the hard column is the one that may be
    quoted as evidence that the join is what was learned.

    Ties count against every arm; the duration baseline additionally gets its
    ties-won upper bound reported, because it ties often and the truth is
    between the two.  See ``recall_from`` for why this mattered.
    """
    model.eval()
    features = data["features"].to(device)
    valid = data["valid"].to(device)
    with torch.no_grad():
        scores = masked_scores(model, features, valid, drop_seam=drop_seam,
                               drop_groups=drop_groups)
    hits1, hitsk = recall_from(scores.cpu(), data["valid"], k)

    duration = -(data["lengths"] - data["targets"][:, None]).abs()
    duration = duration.masked_fill(~data["valid"], -float("inf"))
    base1, basek = recall_from(duration, data["valid"], k)
    base1_ties, basek_ties = recall_from(duration, data["valid"], k, ties_win=True)

    pool = data["valid"].sum(dim=1).float()
    hard = data["same_group"]
    return {
        "pairs": int(len(pool)),
        "recall@1": float(hits1.mean()),
        "recall@k": float(hitsk.mean()),
        "duration_recall@1": float(base1.mean()),
        "duration_recall@k": float(basek.mean()),
        # Upper bound: every tie resolved in the baseline's favour.
        "duration_recall@1_ties_won": float(base1_ties.mean()),
        "duration_recall@k_ties_won": float(basek_ties.mean()),
        "chance_recall@1": float((1.0 / pool).mean()),
        "chance_recall@k": float((torch.clamp(pool, max=k) / pool).mean()),
        "mean_pool": float(pool.mean()),
        "k": k,
        "hard_pairs": int(hard.sum()),
        "hard_recall@1": float(hits1[hard].mean()) if int(hard.sum()) else float("nan"),
        "hard_duration_recall@1": (float(base1[hard].mean()) if int(hard.sum())
                                   else float("nan")),
        "hard_chance_recall@1": (float((1.0 / pool[hard]).mean()) if int(hard.sum())
                                 else float("nan")),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--sequences", required=True)
    parser.add_argument("--normalizer", required=True)
    parser.add_argument("--exclude", default="runs/timebase_exclude_v1.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--negatives", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--eval-k", type=int, default=5)
    parser.add_argument("--max-train-pairs", type=int, default=60000)
    parser.add_argument("--max-eval-pairs", type=int, default=4000)
    parser.add_argument("--limit-recordings", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--disable-groups", default="cand_beat",
        help="feature groups zeroed in the SHIPPED model, training included; "
             "the list travels in the checkpoint.  Default cand_beat -- see "
             "RetrievalSelector.__init__ for the measurement that closed it.")
    parser.add_argument(
        "--control", action="append", default=[],
        metavar="GROUP[,GROUP...]",
        help="train a control model with these feature groups zeroed throughout "
             "and report its held-out recall; repeatable")
    options = parser.parse_args()

    torch.manual_seed(options.seed)
    rng = np.random.default_rng(options.seed)

    print("loading corpus", flush=True)
    records, excluded = load_corpus(options.labels_dir, options.sequences,
                                    options.exclude, options.limit_recordings)
    print("  {} recordings ({} excluded by the timebase census)".format(
        len(records), excluded), flush=True)
    corpus = Corpus(records, options.normalizer)
    print("  {} segments, {} classes in the train pool".format(
        len(corpus.descriptors), len(corpus.by_label)), flush=True)

    train_pairs = build_pairs(corpus, "train", rng, options.negatives,
                              options.max_train_pairs)
    val_pairs = build_pairs(corpus, "val", rng, options.negatives,
                            options.max_eval_pairs)
    print("  {} train pairs, {} held-out pairs".format(
        len(train_pairs), len(val_pairs)), flush=True)
    if not train_pairs or not val_pairs:
        raise SystemExit("no pairs; refusing to train on nothing")

    exact = float(np.mean([1.0 if length == target else 0.0
                           for _, _, _, length, target in train_pairs]))
    print("  target == positive's own length in {:.1%} of pairs "
          "(leak 2 check)".format(exact), flush=True)

    groups = [record["group"] for record in corpus.records]
    print("materialising features (once -- they do not depend on the model)",
          flush=True)
    train_data = materialize(corpus, train_pairs, groups)
    val_data = materialize(corpus, val_pairs, groups)
    print("  train {} x {}, held-out {} x {}".format(
        *train_data["features"].shape[:2], *val_data["features"].shape[:2]),
        flush=True)

    device = options.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("  device: {}".format(device), flush=True)

    disabled = tuple(name.strip() for name in options.disable_groups.split(",")
                     if name.strip())
    model = RetrievalSelector(hidden=options.hidden, layers=options.layers,
                              dropout=options.dropout, disabled_groups=disabled)
    if disabled:
        print("  feature groups disabled in the shipped model: {}".format(
            ", ".join(disabled)), flush=True)
    flat = train_data["features"][train_data["valid"]]
    model.set_feature_stats(flat.mean(dim=0), flat.std(dim=0))
    model = model.to(device)

    features = train_data["features"].to(device)
    valid = train_data["valid"].to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=options.learning_rate,
                                  weight_decay=1e-2)
    history = []
    for epoch in range(options.epochs):
        model.train()
        order = torch.from_numpy(rng.permutation(len(train_pairs))).to(device)
        total, seen = 0.0, 0
        started = time.time()
        for cursor in range(0, len(order), options.batch_size):
            chunk = order[cursor:cursor + options.batch_size]
            scores = masked_scores(model, features[chunk], valid[chunk])
            # The positive is column 0 by construction, so the target is 0 for
            # every row -- a softmax over the pool, i.e. InfoNCE.
            loss = F.cross_entropy(
                scores, torch.zeros(len(chunk), dtype=torch.long, device=device))
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total += float(loss) * len(chunk)
            seen += len(chunk)
        metrics = evaluate(model, val_data, options.eval_k, device=device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = total / max(seen, 1)
        metrics["seconds"] = time.time() - started
        history.append(metrics)
        print("epoch {} loss {:.4f}  held-out r@1 {:.4f} (duration {:.4f}, "
              "chance {:.4f})  r@{} {:.4f}  hard r@1 {:.4f} (duration {:.4f}, "
              "chance {:.4f}, n={})  {:.0f}s".format(
                  epoch, metrics["train_loss"], metrics["recall@1"],
                  metrics["duration_recall@1"], metrics["chance_recall@1"],
                  options.eval_k, metrics["recall@k"], metrics["hard_recall@1"],
                  metrics["hard_duration_recall@1"],
                  metrics["hard_chance_recall@1"], metrics["hard_pairs"],
                  metrics["seconds"]),
              flush=True)

    # THE CONTROLS THAT DECIDE WHETHER ANY OF THIS MEANS ANYTHING.
    # Zeroing a block at EVALUATION time only tells you that a model which
    # leaned on it breaks when you take it away.  It cannot tell you whether
    # the REST of the row is enough on its own -- and if it is, the row carries
    # identity and the headline is not about the join.  So each control is a
    # second model TRAINED with those groups zeroed throughout.
    #
    # Measured 2026-09-01 on 1,200 recordings: the seam-blind control still read
    # held-out recall@1 0.9798 against a duration baseline of 0.3895.  That is
    # what this sweep exists to localise.
    controls = {}
    for spec in options.control:
        names = tuple(name.strip() for name in spec.split(",") if name.strip())
        print("training control without {}".format(", ".join(names)), flush=True)
        blind_model = RetrievalSelector(hidden=options.hidden, layers=options.layers,
                                        dropout=options.dropout,
                                        disabled_groups=disabled)
        blind_model.set_feature_stats(flat.mean(dim=0), flat.std(dim=0))
        blind_model = blind_model.to(device)
        blind = torch.optim.AdamW(blind_model.parameters(),
                                  lr=options.learning_rate, weight_decay=1e-2)
        for epoch in range(options.epochs):
            blind_model.train()
            order = torch.from_numpy(rng.permutation(len(train_pairs))).to(device)
            for cursor in range(0, len(order), options.batch_size):
                chunk = order[cursor:cursor + options.batch_size]
                scores = masked_scores(blind_model, features[chunk], valid[chunk],
                                       drop_groups=names)
                loss = F.cross_entropy(
                    scores, torch.zeros(len(chunk), dtype=torch.long, device=device))
                blind.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(blind_model.parameters(), 1.0)
                blind.step()
        measured = evaluate(blind_model, val_data, options.eval_k,
                            drop_groups=names, device=device)
        controls[spec] = measured
        print("  CONTROL without [{}]: r@1 {:.4f}  (duration {:.4f}, chance "
              "{:.4f})".format(spec, measured["recall@1"],
                               measured["duration_recall@1"],
                               measured["chance_recall@1"]), flush=True)

    final = evaluate(model, val_data, options.eval_k, device=device)
    ablation = evaluate(model, val_data, options.eval_k, drop_seam=True,
                        device=device)
    print("ABLATION seam block zeroed: r@1 {:.4f} vs {:.4f} (hard {:.4f} vs "
          "{:.4f})".format(ablation["recall@1"], final["recall@1"],
                           ablation["hard_recall@1"], final["hard_recall@1"]),
          flush=True)
    model = model.to("cpu")

    output = pathlib.Path(options.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "stage": "retrieval_selector",
        "state_dict": model.state_dict(),
        "feature_dim": FEATURE_DIM,
        "hidden": options.hidden,
        "layers": options.layers,
        "dropout": options.dropout,
        "disabled_groups": list(disabled),
        "label_pool_stats": {int(k): (v[0], v[1])
                             for k, v in corpus.label_pool_stats.items()},
        "default_pool_stats": corpus.default_stats,
        "args": vars(options),
    }, output)

    report = {
        "final": final,
        "ablation_seam_zeroed_at_eval": ablation,
        "controls_trained_without_groups": controls,
        "history": history,
        "target_equals_positive_length": exact,
        "recordings": len(records),
        "excluded_by_timebase_census": excluded,
        "segments": len(corpus.descriptors),
        "train_pool_classes": len(corpus.by_label),
        "checkpoint": str(output),
    }
    pathlib.Path(options.report).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(options.report).write_text(json.dumps(report, indent=2))
    print("wrote {} and {}".format(output, options.report))


if __name__ == "__main__":
    main()
