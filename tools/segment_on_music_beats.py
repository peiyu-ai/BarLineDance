#!/usr/bin/env python3
"""Cut on the music's beat grid, in whole beats.

Why this arm exists.  Three motion-side rules have now been looked at on a
contact sheet and rejected by eye: Alg.1's own cuts, cuts snapped to the nearest
speed minimum, and cuts snapped to the most *recurring* settled pose.  The
operator's reading, 2026-08-20: "the pose the cut lands on is all over the place,
no different from random."  Each of those rules depends on the 3D reconstruction
or on S3D features, both of which are noisy on wild footage, and none of them has
a ground truth to be tuned against.

The beat grid does not depend on either.  It is already in every clip's 35-D
music feature -- channel 34, one-hot, frame-aligned at 30 fps by
``extract_wild_music_features`` -- so it is defined even where the reconstruction
failed, and it is stable: measured over all 2,103 clean5 clips, the inter-beat
interval has a coefficient of variation of 0.037 at the median, every clip reads
below 0.095, and **no clip has a single inter-beat gap wider than 1.5x its own
median** (0 of 2,103), i.e. the tracker never drops a beat mid-clip.  Median 32
beats per clip.

What that grid is, stated so it is not over-read: ``librosa.beat.beat_track`` with
``tightness=100`` fits a near-constant tempo, so these are the beats of a fitted
metronome rather than individually detected onsets.  The un-gridded onsets are
channel 33 and are *not* used here -- a segmentation wants a stable clock, and the
regularity above is the reason this arm is expected to be robust where the motion
rules were not.

**The downbeat is not detected.**  ``beat_track`` returns beats, not bars, so
which beat begins a bar is unknown.  ``--phase first`` starts at the first beat
and says so; ``--phase energy`` picks the offset whose beats carry the most onset
envelope (channel 0), which is a *guess* that downbeats are louder, labelled as a
guess and reported with the margin it won by so a near-tie is visible.

--------------------------------------------------------------------------
Edge policy -- ``--edges``, and why ``drop`` is the default (2026-08-20)
--------------------------------------------------------------------------

A beat grid does not start at frame 0 and does not end at the last frame.  On
clean5 the lead-in before the first beat is a median 0.30 s but reaches 5.20 s at
p99 and 11.73 s at the maximum; the trail-out after the last beat is a median
0.60 s, p99 5.27 s, maximum 11.07 s.  The first release of this tool cut
``[0] + beats[phase::k] + [frames]``, so the first and last segment were **not**
whole beats -- they were "whatever is left over", and they were 22.6% of the
segments (4,204 of 18,580) and 20.9% of the corpus duration.  Measured on that
release (``runs/wild_v4_seg_beat4``): interior segments ran 1.200-3.700 s, locked
to four beats, while the head ran 0.400-12.667 s and the tail 0.400-11.733 s.
The six longest "atomic movements" in the corpus, 10.6-12.7 s each, were all head
or tail segments.  That is the defect: the arm was chosen for "every cut is on a
beat and every segment is a whole number of beats", and one segment in five was
neither.

Worse, ``merge_short`` hid part of it.  With ``--min-length 12`` it removed 739
cuts on 697 of 2,102 clips (33.2%) -- 561 at the head, 178 at the tail, **0 in the
interior**.  The head merges are almost all one situation: when ``--phase energy``
picks offset 0 (878 clips, 41.8%), the first grid cut is ``beats[0]`` itself,
which sits a median 9 frames into the clip, below ``min_length``; the merge then
deletes that cut and the head becomes lead-in plus *eight* beats.  Nothing in the
old output said this had happened.  The interior escaped only because four beats
at the corpus's *fastest* tempo (200 BPM, a 9-frame beat) is still 36 frames,
three times ``min_length`` -- luck, not construction.

So ``--edges drop`` is now the default: the cuts are exactly
``beats[phase], beats[phase+k], ...`` and the incomplete head and tail are
discarded and *counted*, per clip and in the summary.  Every emitted segment is
then exactly ``k`` beats and every boundary is a beat frame, which is checkable,
and ``validate_grid_records`` below checks it.  ``--edges fold`` reproduces the
old behaviour and ``--edges keep`` skips the merge; both need
``--allow-partial-edges``, which is the flag's whole purpose -- it names the
defect instead of hiding it.

**What ``drop`` costs, stated up front:** 1.69 h of the 10.10 h corpus (16.7%),
0.73 h of head and 0.95 h of tail.  A median clip keeps 84.7% of its duration; 30
clips keep under half.  A median of 2 beats per clip is discarded (mean 2.41, max
6).

That 1.69 h splits in two, and only one half is forced.  The head loss is the
lead-in plus ``phase`` beats and the tail loss is ``(n_beats-1-phase) mod k`` beats
plus the trail-out, with ``phase`` fixed by the downbeat guess rather than free to
choose.  Measured on the published run: **0.94 h is lead-in (0.38 h) and trail-out
(0.56 h)**, which no beat-aligned rule can keep, and **0.73 h is whole beats** --
1 to 3 of them at each edge -- which an ``--edges partial`` policy could keep as
grid-aligned segments of fewer than ``k`` beats.  This tool does not offer that
policy: the operator's rule on 2026-08-20 was "cut the clip every 4 music beats",
and a 1-beat segment is a different object from a 4-beat one, so mixing them into
M1 without being asked would be answering a question that was not put.  The
number is written here so that trade is visible and can be taken later; it is not
a claim that ``drop`` is the only option, which an earlier draft of this docstring
wrongly asserted.

Captions are not an argument either way.  Captions are keyed by
``(recording_id, start, end)`` in ``wild_v4_acct/captions.jsonl``; of the 27,263
rows on clean5 clips, 27,152 (77.9%) match spans of ``runs/wild_v4_seg`` and only
32 (0.2%) match spans of ``runs/wild_v4_seg_beat4`` -- 0 match ``beat4h``.
Choosing D at all already invalidates essentially every caption, so hardening D
costs no captions that D had not already cost.

--------------------------------------------------------------------------
Which clips are refused, and which are only reported
--------------------------------------------------------------------------

*Refused.*  A clip that yields **zero** whole ``k``-beat segments.  The old test
was ``len(beats) < max(2, k)``, a beat *count*; it excluded exactly one clip
(``7518354096051883321__clip000``, 3 beats at frames 6/20/34 over 367 frames) and
let through ``7574019615476924133__clip000``, 4 beats over 398 frames, which then
produced an 11.03 s segment.  A count cannot express "there is no room for one
whole segment", so the test is now the thing itself: build the cut list, and if it
has fewer than two grid points the clip is excluded by name.  Excluded stems are
written to ``excluded_clips.txt`` beside the output and listed in the report, so a
downstream consumer reads a manifest instead of differencing two directories
(CLAUDE.md 1.1).

*Reported, not gated.*  ``grid_coverage`` -- the fraction of the clip between the
first and last beat (median 0.930, p10 0.800; 70 of 2,103 clips below 0.70
and 11 below 0.50, of which the 2 lowest are excluded anyway for holding no whole
segment, leaving 68 in the published run).
It is tempting to refuse low-coverage clips, and it was tried: if a low coverage
meant the tracker had given up while the music kept playing, the onset envelope
outside the beat span would be as loud as inside.  Measured, it is not -- the ratio
outside/inside is 0.829 at the median for clips with coverage >= 0.85 and *lower*,
0.757, for clips below 0.70 (the 11 clips below 0.50 read 0.46-0.80).  The
uncovered stretches are quieter, not equally loud, so there is no evidence that
low coverage is a tracker failure and this criterion has not earned a judgement.
It is written into every record so a consumer can filter on it, and
``--min-grid-coverage`` exists but defaults to 0.0 (off).

*Reported, not gated.*  Tempo.  ``k`` beats is a fixed number of beats and a
*variable* number of seconds: across clean5 the median inter-beat interval runs
9-28 frames (200-64 BPM, median 120), so a four-beat segment runs 1.200-3.833 s in the
published run, a 3.2x spread.  This is inherent to the arm and is the price of cutting on the beat;
it is reported in ``tempo`` so it cannot be mistaken for a uniform duration.  The
18 clips the tracker calls below 80 BPM were checked for a half-time octave error
with a ruler that has both controls: mean onset envelope at the *midpoints*
between beats, over the same at the beats.  Decimating a correct 110-130 BPM grid
by 2x (positive control, a known half-time grid) reads 0.769; the same clips
undecimated (negative control) read 0.538; the 18 slow clips read 0.220, below
even the negative control's p10 of 0.283.  They are slow tracks, not octave
errors.  The two controls overlap heavily, so this ruler is only strong enough to
say "not half-time", not to grade a grid.

*Reported, not gated.*  Tempo drift within a clip, measured as
``|median(first-half inter-beat) - median(second-half)| / median``: 0.000 at the
median, 0.077 at p99, 0.115 at the maximum.  This is why
``validate_grid_records`` checks the *grid index* span exactly and the *duration*
span only loosely: a segment can be exactly four grid points and still be 3.43 or
4.63 median-inter-beat-lengths long because the tempo moved under it.  On the
hardened run that is 8 segments of 15,115 at a tolerance of 0.5 beat -- a property
of the music, not of the cut, so it is counted and named, not refused.

Modes:

* ``--mode grid`` -- a cut every ``--beats-per-segment`` beats.  This is the
  operator's "cut to a whole beat" in its strongest form.  It ignores the dancer
  entirely, which is the point -- it is the arm that has no way to land
  mid-movement for a reason that depends on the reconstruction.
* ``--mode snap`` -- keep an existing arm's cuts and move each to the nearest
  beat.  This keeps Alg.1's judgement about *where the material changes* and
  fixes only the phase.  Both are built because they fail differently: the grid
  cannot follow a dancer who is off the beat, and the snap inherits whatever
  Alg.1 got wrong.  ``snap`` cannot satisfy the whole-beat invariant (its segments
  are whatever Alg.1's spacing was, rounded to beats), so it is validated for
  "every interior boundary is on a beat" only.

No claim is made here that either is better.  There is still no annotated
boundary set for this corpus, so the judge is the contact sheet.

Usage::

    segment_on_music_beats.py --bundle <raw bundle> --clips runs/clean5/clips.txt \\
        --mode grid --beats-per-segment 4 --phase energy --edges drop \\
        --output runs/wild_v4_seg_beat4h/segmentation.json
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

FPS = 30.0
BEAT_CHANNEL = 34
ENVELOPE_CHANNEL = 0

#: A segment is allowed to be this far from ``k`` beats *in duration* while still
#: being exactly ``k`` beats *in grid index*.  0.5 beat is set from the measured
#: within-clip tempo drift: the worst clip on clean5 moves 11.5% between its two
#: halves, and 4 x 0.115 = 0.46 beat.  Segments past it are counted and named.
DURATION_TOLERANCE_BEATS = 0.5

MAX_EXAMPLES = 20


class BeatSegmentError(RuntimeError):
    """A clip whose grid cannot support the request is reported, not silently cut."""


class GridInvariantError(BeatSegmentError):
    """``--edges drop`` promised whole beats and did not deliver them."""


def stem_of(row: dict) -> str:
    name = row.get("recording_id") or row.get("sequence_id") or ""
    parts = name.split(":")
    return "{}__{}".format(parts[1], parts[2]) if len(parts) == 3 else name


def choose_phase(beats: np.ndarray, envelope: np.ndarray, k: int,
                 exclude_anchor: bool = False) -> Dict[str, object]:
    """Offset 0..k-1 whose beats carry the most onset energy -- a downbeat guess.

    ``exclude_anchor`` drops ``beats[0]`` from offset 0's score.  It exists
    because the rule as shipped is partly measuring the beat tracker rather than
    the music: ``beats[0]`` sits at a +0.93 median z-score within its own clip's
    beat envelope (above the clip's beat mean in 75.8% of 2,101 clips), because
    ``librosa.beat.beat_track`` anchors the grid on a strong onset.  That one
    frame is in offset 0's mean and in no other offset's, and it moves the
    outcome: offset 0 wins 41.8% of clips as shipped and 30.6% with ``beats[0]``
    removed, i.e. z=17.8 against a uniform null falls to z=5.9.  Some real signal
    survives, so this is not "the rule is meaningless" -- it is "about 11 points
    of its margin is the instrument".  Neither variant is validated against a
    labelled downbeat, so the default is left on the rule the operator's arm was
    chosen with and the disagreement is reported in ``downbeat.anchor_artifact``.
    """
    if k <= 1:
        return {"phase": 0, "margin": None, "how": "k=1, no phase to choose"}
    scores = []
    for offset in range(k):
        picked = beats[offset::k]
        if exclude_anchor and offset == 0:
            picked = picked[1:]
        picked = picked[(picked >= 0) & (picked < len(envelope))]
        scores.append(float(envelope[picked].mean()) if len(picked) else -np.inf)
    order = np.argsort(scores)[::-1]
    best, second = float(scores[order[0]]), float(scores[order[1]])
    margin = (best - second) / abs(best) if best else 0.0
    return {"phase": int(order[0]), "margin": round(float(margin), 4),
            "how": "max mean onset envelope at the chosen beats{}; a guess that "
                   "downbeats are louder, not a downbeat detection".format(
                       ", with beats[0] excluded from offset 0 so the tracker's "
                       "anchor onset is not counted" if exclude_anchor else "")}


def merge_short(bounds: List[int], min_length: int) -> List[int]:
    """Alg.1 step 7, reused verbatim in spirit: fold a short span into the shorter neighbour.

    Only ``--edges fold`` and ``--mode snap`` reach this.  ``--edges drop`` must
    not: a merge deletes a grid point and the surviving segment is then 2k beats
    (or, at the head, lead-in plus 2k beats), which is exactly the invariant this
    arm was chosen for.  See the module docstring for the 739 cuts it removed on
    the first release.
    """
    bounds = sorted(set(int(b) for b in bounds))
    while len(bounds) > 2:
        spans = np.diff(bounds)
        index = int(np.argmin(spans))
        if spans[index] >= min_length:
            break
        if index == 0:
            bounds.pop(1)
        elif index == len(spans) - 1:
            bounds.pop(len(bounds) - 2)
        else:
            left, right = spans[index - 1], spans[index + 1]
            bounds.pop(index if left <= right else index + 1)
    return bounds


def grid_bounds(beats: np.ndarray, frames: int, k: int, phase: int,
                min_length: int, edges: str = "drop") -> List[int]:
    """Boundaries for one clip.

    ``edges='drop'`` returns ``beats[phase], beats[phase+k], ...`` and nothing
    else, so every boundary is a beat and every span is k grid points.  Fewer than
    two grid points means the clip cannot hold one whole segment; the caller
    excludes it by name.  ``'keep'`` and ``'fold'`` bracket that with frame 0 and
    ``frames``, which makes the first and last segment partial by construction --
    see the module docstring.
    """
    if edges not in ("drop", "keep", "fold"):
        raise ValueError("unknown edge policy {!r}".format(edges))
    grid = [int(b) for b in beats[phase::k] if 0 <= int(b) < frames]
    if edges == "drop":
        return grid
    picked = [b for b in grid if 0 < b < frames]
    bracketed = sorted(set([0] + picked + [frames]))
    return merge_short(bracketed, min_length) if edges == "fold" else bracketed


def snap_bounds(bounds: Sequence[int], beats: np.ndarray, frames: int,
                min_length: int, max_snap: Optional[int]) -> List[int]:
    out = [int(bounds[0])]
    for index in range(1, len(bounds) - 1):
        cut = int(bounds[index])
        remaining = len(bounds) - 1 - index
        low = max(out[-1] + min_length, cut - (max_snap if max_snap else frames))
        high = min(int(bounds[-1]) - remaining * min_length,
                   cut + (max_snap if max_snap else frames))
        if high < low:
            out.append(max(cut, out[-1] + min_length))
            continue
        window = beats[(beats >= low) & (beats <= high)]
        out.append(int(window[np.argmin(np.abs(window - cut))]) if len(window)
                   else min(max(cut, low), high))
    out.append(int(bounds[-1]))
    return merge_short(out, min_length)


def validate_grid_records(records: Sequence[dict], grids: Dict[str, np.ndarray],
                          k: int, interior_only: bool = False,
                          check_span: bool = True,
                          tolerance: float = DURATION_TOLERANCE_BEATS) -> Dict[str, object]:
    """The gate ``--edges drop`` is allowed to be trusted on.  It can fail.

    Three structural checks, each of which is a promise the arm was chosen for and
    each of which fails loudly on the pre-2026-08-20 output:

    * ``boundary_off_grid`` -- a boundary frame that is not a beat frame of that
      clip's own grid.  Alg.1 as shipped reads 6.8% on-beat against a 6.2% chance
      rate, so this check separates this arm from that one at all.
    * ``span_not_k_beats`` -- a segment that does not span exactly ``k``
      *consecutive* grid indices.  This is the one ``merge_short`` used to break
      silently, and the one the old head/tail segments broke by construction
      (4,204 of 18,580).
    * ``not_contiguous`` -- segments that do not tile their own boundary list.

    and one measurement that is a property of the music rather than of the cut, so
    it is counted and named rather than refused: ``duration_off_by_beats``, a
    segment whose length in units of its clip's median inter-beat interval is
    further than ``tolerance`` from ``k``.  See ``DURATION_TOLERANCE_BEATS``.

    ``interior_only=True`` exempts the first and last segment of each clip; it
    exists for ``--edges fold``/``keep``, where those two are partial on purpose,
    and it must never be set for ``--edges drop`` -- that would make the gate
    unable to fail on the very thing it was written for.  ``check_span=False``
    drops the k-beat check entirely and exists for ``--mode snap``, which inherits
    Alg.1's spacing and only ever promised "on a beat"; it must never be set for
    ``--mode grid``.
    """
    counts = collections.Counter()
    examples: Dict[str, List[str]] = collections.defaultdict(list)
    checked_segments = checked_boundaries = 0

    def note(kind: str, detail: str) -> None:
        counts[kind] += 1
        if len(examples[kind]) < MAX_EXAMPLES:
            examples[kind].append(detail)

    for record in records:
        stem = record["sequence"]
        beats = np.asarray(grids[stem], dtype=np.int64)
        index_of = {int(b): i for i, b in enumerate(beats)}
        period = float(np.median(np.diff(beats))) if len(beats) >= 2 else float("nan")
        bounds = [int(b) for b in record["boundaries"]]
        segments = record["segments"]

        if [ {"start": a, "end": b, "frames": b - a}
             for a, b in zip(bounds[:-1], bounds[1:]) ] != [
             {"start": int(s["start"]), "end": int(s["end"]),
              "frames": int(s["frames"])} for s in segments ]:
            note("not_contiguous", "{}: segments do not tile boundaries".format(stem))

        # Boundaries are checked once each, not once per adjoining segment: an
        # off-grid cut is one defect, and counting it twice would inflate the
        # number a reader compares against the 6.8%-on-beat figure for Alg.1.
        for position, boundary in enumerate(bounds):
            if interior_only and position in (0, len(bounds) - 1):
                continue
            checked_boundaries += 1
            if boundary not in index_of:
                note("boundary_off_grid", "{}: {} is not a beat".format(stem, boundary))

        last = len(segments) - 1
        for position, segment in enumerate(segments):
            if interior_only and position in (0, last):
                continue
            checked_segments += 1
            start, end = int(segment["start"]), int(segment["end"])
            i, j = index_of.get(start), index_of.get(end)
            if check_span and i is not None and j is not None and j - i != k:
                note("span_not_k_beats",
                     "{}: [{},{}] spans {} beats, wanted {}".format(stem, start, end, j - i, k))
            if period == period and period > 0:
                off = abs((end - start) / period - k)
                if off > tolerance:
                    note("duration_off_by_beats",
                         "{}: [{},{}] is {:.2f} beats long".format(
                             stem, start, end, (end - start) / period))

    structural = (("boundary_off_grid", "not_contiguous") if not check_span
                  else ("boundary_off_grid", "span_not_k_beats", "not_contiguous"))
    return {
        "checked": True,
        "segments_checked": checked_segments,
        "boundaries_checked": checked_boundaries,
        "interior_only": bool(interior_only),
        "span_checked": bool(check_span),
        "tolerance_beats": tolerance,
        "counts": {name: int(counts[name]) for name in
                   structural + ("duration_off_by_beats",)},
        "structural_violations": int(sum(counts[name] for name in structural)),
        "examples": {name: examples[name] for name in examples},
        "reading": "structural_violations > 0 means a boundary is off the grid or a "
                   "segment is not exactly k beats; duration_off_by_beats is the "
                   "tempo moving under a segment that is still exactly k beats",
    }


def _grid_report(beats: np.ndarray, frames: int) -> Dict[str, object]:
    intervals = np.diff(beats.astype(float))
    period = float(np.median(intervals)) if len(intervals) else float("nan")
    half = len(intervals) // 2
    drift = (abs(float(np.median(intervals[:half])) - float(np.median(intervals[half:])))
             / period if half >= 3 and period > 0 else 0.0)
    return {
        "beats": int(len(beats)),
        "grid_coverage": round(float(beats[-1] - beats[0]) / max(1, frames - 1), 4),
        "median_inter_beat_frames": round(period, 3),
        "bpm": round(60.0 * FPS / period, 2) if period > 0 else None,
        "tempo_drift": round(float(drift), 4),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=pathlib.Path, required=True)
    parser.add_argument("--clips", type=pathlib.Path, required=True)
    parser.add_argument("--mode", choices=("grid", "snap"), default="grid")
    parser.add_argument("--beats-per-segment", type=int, default=4)
    parser.add_argument("--phase", choices=("first", "energy", "energy_no_anchor"),
                        default="energy",
                        help="'energy' is the rule arm D was chosen with; "
                             "'energy_no_anchor' is the same rule with beats[0] "
                             "excluded from offset 0's score, because librosa "
                             "anchors the grid on a strong onset -- see choose_phase")
    parser.add_argument("--edges", choices=("drop", "keep", "fold"), default="drop",
                        help="grid mode: what to do with the lead-in before the first "
                             "grid point and the trail-out after the last.  'drop' "
                             "discards and counts them, which is the only policy under "
                             "which every segment is whole beats; 'fold' is the "
                             "pre-2026-08-20 behaviour and 'keep' is it without the "
                             "merge -- both need --allow-partial-edges")
    parser.add_argument("--allow-partial-edges", action="store_true",
                        help="record the whole-beat violations that 'fold'/'keep' "
                             "produce by construction instead of refusing to write")
    parser.add_argument("--min-grid-coverage", type=float, default=0.0,
                        help="exclude clips whose first-to-last-beat span covers less "
                             "than this fraction of the clip.  Default 0.0 (off): the "
                             "onset envelope outside the beat span is quieter, not "
                             "equally loud, so low coverage is not evidence of a "
                             "tracker failure -- see the module docstring")
    parser.add_argument("--arm", type=pathlib.Path, default=None,
                        help="snap mode: the segmentation whose cuts get moved")
    parser.add_argument("--max-snap", type=int, default=None)
    parser.add_argument("--min-length", type=int, default=18,
                        help="merge threshold in frames; used by --mode snap and by "
                             "--edges fold only.  --edges drop never merges")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    if args.mode == "snap" and args.arm is None:
        raise SystemExit("--mode snap needs --arm")
    if args.mode == "grid" and args.edges != "drop" and not args.allow_partial_edges:
        raise SystemExit(
            "--edges {} makes the first and last segment partial by construction "
            "(20.9% of the corpus on the 2026-08-20 release); pass "
            "--allow-partial-edges to record that instead of refusing".format(args.edges))
    if args.beats_per_segment < 1:
        raise SystemExit("--beats-per-segment must be >= 1")

    wanted = {line.strip() for line in args.clips.read_text(encoding="utf-8").splitlines()
              if line.strip()}
    arm = {}
    if args.arm:
        report = json.loads(args.arm.read_text(encoding="utf-8"))
        arm = {r["sequence"]: r["boundaries"] for r in report["records"]}

    records: List[dict] = []
    grids: Dict[str, np.ndarray] = {}
    phases: List[dict] = []
    excluded: Dict[str, List[str]] = collections.defaultdict(list)
    seen = set()
    dropped_head = dropped_tail = kept_frames = total_frames = 0
    anchor_disagreements = anchor_compared = 0

    with open(args.bundle / "sequences.jsonl", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    for row in rows:
        stem = stem_of(row)
        if stem not in wanted or stem in seen:
            continue
        seen.add(stem)
        music = row.get("music_path")
        path = args.bundle / music if music else None
        if path is None or not path.is_file():
            excluded["no_music"].append(stem)
            continue
        table = np.load(path)
        frames = int(table.shape[0])
        beats = np.flatnonzero(table[:, BEAT_CHANNEL] > 0.5)
        if len(beats) < 2:
            excluded["fewer_than_two_beats"].append(stem)
            continue
        grid = _grid_report(beats, frames)
        if grid["grid_coverage"] < args.min_grid_coverage:
            excluded["below_min_grid_coverage"].append(stem)
            continue

        if args.mode == "grid":
            if args.phase.startswith("energy"):
                choice = choose_phase(beats, table[:, ENVELOPE_CHANNEL],
                                      args.beats_per_segment,
                                      exclude_anchor=args.phase == "energy_no_anchor")
                other = choose_phase(beats, table[:, ENVELOPE_CHANNEL],
                                     args.beats_per_segment,
                                     exclude_anchor=args.phase == "energy")
            else:
                choice = other = {"phase": 0, "margin": None, "how": "first beat"}
            bounds = grid_bounds(beats, frames, args.beats_per_segment,
                                 choice["phase"], args.min_length, args.edges)
            if len(bounds) < 2:
                # The old test was ``len(beats) < max(2, k)`` -- a beat count, which
                # let through a clip with exactly k beats and no room for a segment.
                excluded["no_whole_segment"].append(stem)
                continue
            if args.phase.startswith("energy"):
                # counted over published clips only, so it can be read against
                # ``sequences`` without a second accounting
                anchor_disagreements += int(other["phase"] != choice["phase"])
                anchor_compared += 1
            phases.append(choice)
            grid["phase"] = choice["phase"]
            grid["phase_margin"] = choice["margin"]
        else:
            if stem not in arm:
                excluded["no_arm"].append(stem)
                continue
            bounds = snap_bounds(arm[stem], beats, frames, args.min_length,
                                 args.max_snap)
            if len(bounds) < 3:
                excluded["no_whole_segment"].append(stem)
                continue

        total_frames += frames
        dropped_head += bounds[0]
        dropped_tail += frames - bounds[-1]
        kept_frames += bounds[-1] - bounds[0]
        grids[stem] = beats
        record = {
            "sequence": stem, "encoder": "music-beat-grid",
            # motion_frames stays the whole clip: select_clean_accounts_corpus
            # compares it against the S3D feature length to catch a stale cut.
            "motion_frames": frames,
            "covered_frames": bounds[-1] - bounds[0],
            "dropped_head_frames": bounds[0],
            "dropped_tail_frames": frames - bounds[-1],
            "boundaries": bounds,
            "segments": [{"start": a, "end": b, "frames": b - a}
                         for a, b in zip(bounds[:-1], bounds[1:])],
        }
        record.update(grid)
        records.append(record)

    for stem in sorted(wanted - seen):
        excluded["not_in_bundle"].append(stem)

    if not records:
        raise BeatSegmentError("no clip produced a segmentation: {}".format(
            {k: len(v) for k, v in excluded.items()}))

    # The gate is enforced exactly where the promise was made: --mode grid
    # --edges drop.  snap never promised whole beats (it inherits Alg.1's spacing,
    # and 4.9% of its cuts land off-grid because the min-length window had no beat
    # in it), and fold/keep promised partial edges, so those two report the
    # violations into the artifact instead of refusing to write.
    enforcing = args.mode == "grid" and args.edges == "drop"
    validation = validate_grid_records(
        records, grids, args.beats_per_segment,
        interior_only=not enforcing, check_span=(args.mode == "grid"))
    validation["enforced"] = bool(enforcing)
    if not enforcing and args.mode == "grid":
        # Say how bad the edges actually are rather than only exempting them:
        # on the 2026-08-20 release this reads 4,204 of 18,580 segments.
        full = validate_grid_records(records, grids, args.beats_per_segment,
                                     interior_only=False, check_span=True)
        validation["edges_included"] = {
            "segments_checked": full["segments_checked"],
            "counts": full["counts"],
            "structural_violations": full["structural_violations"],
            "reading": "these are the head/tail segments --edges {} keeps; they are "
                       "not whole beats by construction".format(args.edges),
        }
    if enforcing and validation["structural_violations"]:
        raise GridInvariantError(
            "{} structural violations of the whole-beat invariant: {}; examples {}"
            .format(validation["structural_violations"], validation["counts"],
                    dict(list(validation["examples"].items())[:2])))

    durations = [s["frames"] / FPS for r in records for s in r["segments"]]
    margins = [p["margin"] for p in phases if p.get("margin") is not None]
    coverage = np.asarray([r["grid_coverage"] for r in records])
    bpm = np.asarray([r["bpm"] for r in records if r["bpm"]])
    drift = np.asarray([r["tempo_drift"] for r in records])
    result = {
        "features_dir": "music_35 channel {} (beat one-hot)".format(BEAT_CHANNEL),
        "config": {"mode": args.mode, "beats_per_segment": args.beats_per_segment,
                   "phase": args.phase, "edges": args.edges,
                   "allow_partial_edges": bool(args.allow_partial_edges),
                   "min_length_frames": args.min_length,
                   "min_grid_coverage": args.min_grid_coverage,
                   "max_snap": args.max_snap, "fps": FPS,
                   "source_arm": str(args.arm) if args.arm else None},
        "sequences": len(records), "total_segments": len(durations),
        "segments_per_sequence": len(durations) / len(records),
        "duration_seconds": {"mean": float(np.mean(durations)),
                             "median": float(np.median(durations)),
                             "min": float(np.min(durations)),
                             "max": float(np.max(durations))},
        "coverage": {
            "clips_requested": len(wanted),
            "clips_kept": len(records),
            "corpus_frames": total_frames,
            "kept_frames": kept_frames,
            "kept_fraction": round(kept_frames / total_frames, 4) if total_frames else None,
            "dropped_head_hours": round(dropped_head / FPS / 3600.0, 4),
            "dropped_tail_hours": round(dropped_tail / FPS / 3600.0, 4),
            "grid_coverage_median": round(float(np.median(coverage)), 4),
            "clips_below_grid_coverage_0.70": int((coverage < 0.70).sum()),
            "reading": "the dropped hours are the lead-in before the first beat and "
                       "the trail-out after the last; they are the price of every "
                       "segment being whole beats",
        },
        "tempo": {
            "bpm_median": round(float(np.median(bpm)), 2) if len(bpm) else None,
            "bpm_p10": round(float(np.percentile(bpm, 10)), 2) if len(bpm) else None,
            "bpm_p90": round(float(np.percentile(bpm, 90)), 2) if len(bpm) else None,
            "bpm_min": round(float(bpm.min()), 2) if len(bpm) else None,
            "bpm_max": round(float(bpm.max()), 2) if len(bpm) else None,
            "drift_median": round(float(np.median(drift)), 4),
            "drift_p99": round(float(np.percentile(drift, 99)), 4),
            "reading": "k beats is a fixed number of beats and a variable number of "
                       "seconds; the segment-duration spread below is the tempo "
                       "spread, not a property of the cut",
        },
        "excluded": {"counts": {k: len(v) for k, v in sorted(excluded.items())},
                     "clips": {k: sorted(v) for k, v in sorted(excluded.items())}},
        "validation": validation,
        "downbeat": {"detected": False,
                     "phase_rule": phases[0]["how"] if phases else "n/a",
                     "margin_median": round(float(np.median(margins)), 4) if margins else None,
                     "margin_below_2pct": (round(float(np.mean(np.asarray(margins) < 0.02)), 4)
                                           if margins else None),
                     "reading": "a small margin means the chosen offset barely beat "
                                "another; the bar phase is a guess either way",
                     "anchor_artifact": {
                         "clips_compared": anchor_compared,
                         "clips_whose_phase_changes_without_beats0":
                             anchor_disagreements,
                         "share": (round(anchor_disagreements / anchor_compared, 4)
                                   if anchor_compared else None),
                         "reading": "librosa anchors the grid on a strong onset, so "
                                    "beats[0] is loud (median z=+0.93 within its own "
                                    "clip) and sits in offset 0's score alone.  This "
                                    "is how many clips the energy rule would place "
                                    "differently with that one frame removed -- a "
                                    "disagreement between two versions of the same "
                                    "ruler, reported rather than resolved, because "
                                    "neither has a labelled downbeat to check against"},
                     },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result), encoding="utf-8")
    kept_list = args.output.parent / "clips_kept.txt"
    kept_list.write_text("\n".join(r["sequence"] for r in records) + "\n", encoding="utf-8")
    excluded_list = args.output.parent / "excluded_clips.txt"
    excluded_list.write_text(
        "".join("{}\t{}\n".format(stem, reason)
                for reason, stems in sorted(excluded.items()) for stem in sorted(stems)),
        encoding="utf-8")

    print("{} clips, {} segments, median {:.3f} s, {:.1f} seg/clip".format(
        len(records), len(durations), result["duration_seconds"]["median"],
        result["segments_per_sequence"]))
    print("  segment duration range {:.3f}-{:.3f} s ({}-{} BPM at {} beats)".format(
        result["duration_seconds"]["min"], result["duration_seconds"]["max"],
        result["tempo"]["bpm_max"], result["tempo"]["bpm_min"], args.beats_per_segment))
    print("  kept {:.1%} of the corpus; dropped {:.2f} h of lead-in + {:.2f} h of trail-out"
          .format(result["coverage"]["kept_fraction"],
                  result["coverage"]["dropped_head_hours"],
                  result["coverage"]["dropped_tail_hours"]))
    print("  excluded: {}".format(result["excluded"]["counts"] or "none"))
    print("  validation: {} segments checked, structural violations {}, "
          "duration off by >{} beat {}".format(
              validation["segments_checked"], validation["structural_violations"],
              validation["tolerance_beats"], validation["counts"]["duration_off_by_beats"]))
    if margins:
        print("  downbeat phase is a GUESS: median margin {:.1%}, near-ties (<2%) {:.1%}"
              .format(np.median(margins), float(np.mean(np.asarray(margins) < 0.02))))
    if anchor_compared:
        print("  and {:.1%} of clips ({}) would take a different phase if beats[0] -- "
              "the tracker's anchor onset -- were left out of the score"
              .format(anchor_disagreements / anchor_compared, anchor_disagreements))
    print("wrote", args.output)
    print("wrote", kept_list, "and", excluded_list)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
