"""Two-stage atomic planner/completion inference for AIST++ audio."""

import argparse
import bisect
import hashlib
import json
import math
import os
import pickle
import random
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/edge-numba-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/edge-matplotlib-cache")

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data.audio_extraction.baseline_features import FPS, SR, extract_audio
from dataset.bar_tokens import COUNT_CHANNELS as _BAR_COUNT_CHANNELS
from dataset.atomic import (
    MERGE_ORDERS,
    TRANSITION_POLICIES,
    labels_to_segments,
    motion_accent_frames,
    refine_plan,
    warp_to_anchors,
)
from train_atomic import (
    completion_model,
    planner_model,
    resolve_device,
    validate_training_data_root,
)


DEFAULT_PLANNER_CHECKPOINT = "runs/atomic_planner/planner.pt"
DEFAULT_COMPLETION_CHECKPOINT = "runs/atomic_completion/completion.pt"
SELF_DRIVEN_PLANNER = "SELF_DRIVEN_PLANNER"
ORACLE_GROUND_TRUTH_PLAN = "ORACLE_GROUND_TRUTH_PLAN"
# The AtomicDance 151-D vector: 4 foot-contact channels, then root position,
# then 24 joints of 6-D rotation.  ``decode_motion`` and ``build_draft`` both
# need to know where the root lives and must not be able to disagree about it,
# so the split is named once here rather than written out at each use.
CONTACT_CHANNELS = 4
# --draft-continuity-scorer decodes each library's continuation frames once and keeps them here (CLAUDE.md 1.2:
# large derived arrays live on /cache, not on the NAS)
CONTINUITY_CACHE_DIR = os.environ.get("ATOMICDANCE_CONTINUITY_CACHE",
                                      "/cache/atomicdance-assets/runs/continuation_scorer_cache")
ROOT_POSITION_START = CONTACT_CHANNELS
ROOT_POSITION_DIMS = 3


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_seed(seed, name):
    """A clip's noise draw, derived from its NAME rather than its position.

    WHY THIS IS NOT COSMETIC.  Until 2026-08-31 the draw was ``seed + index``
    (unbatched) or ``seed + batch[0][0]`` (batched), and the batched path's own
    comment already said what that meant -- "a clip's draw depends on how the
    work was divided, not only on its name".  The consequence was never drawn:
    a run over 90 clips and a run over 24 clips give the SAME clip different
    noise, so two arms scored on different clip lists are not paired even when
    the checkpoint, the plan and the music are identical.

    Measured cost of that, on one checkpoint and config with only the clip
    list's length changed: mean per-clip energy 0.5632 versus 0.6421 -- a 14%
    swing from re-indexing alone, which is larger than most of the arm-to-arm
    differences the 2026-08-30 waves were read off.  Several of those rows sit
    inside this band and cannot be re-audited without regenerating them.

    ``blake2b`` rather than ``hash()``: Python salts ``hash()`` per process, so
    an index-free seed built on it would be reproducible within a run and not
    across runs -- the failure this replaces, in a costume.
    """
    digest = hashlib.blake2b(str(name).encode("utf-8"), digest_size=8).digest()
    return (int(seed) ^ int.from_bytes(digest, "big")) % (2 ** 31 - 1)


def _variety_rng(seed, name):
    """The recurrence-variety draws for ONE clip, derived from its name.

    Same rule as ``sample_seed`` and for the same reason: a clip's result must
    depend on ``(seed, name)`` and on nothing about how the work was ordered or
    divided.  Kept a separate stream from the noise so that changing one does
    not silently re-roll the other.
    """
    return np.random.default_rng(sample_seed(seed, "variety|" + name))


def _code_revision():
    """Best-effort git identity of the code that produced a run."""
    import subprocess

    root = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                check=True, capture_output=True, text=True,
            ).stdout.strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    return {"commit": commit, "working_tree_dirty": dirty}


def _check_release_binding(binding, stage, checkpoint_args, data_root, key, allow_cross):
    """Refuse a checkpoint trained on a different release than the library was built from.

    WHY.  A library directory and the two checkpoints are chosen by three
    independent flags, and nothing compared them: the T line's T1 library
    (``scratch/txy_t/release_aligned_k8``) and its T2 successor (2026-09-22,
    +24 clips) carry the same ``label_space_id`` and the same normalizer bytes,
    so a T2 library run with T1 checkpoints -- or the reverse -- produced a
    manifest indistinguishable from a matched run.  The library's
    ``build.json`` already names the two releases it was derived from
    (``derived_from.release`` for the completion, ``.bar_release`` for the
    bar planner), and every checkpoint records the ``data_root`` it trained on,
    so the check needs no line name hard-coded anywhere.  A deliberate mix (an
    ablation that holds the models fixed and swaps the library) passes with
    ``--allow-cross-release-checkpoints``, and the manifest says so.
    """
    build_path = Path(data_root) / "build.json"
    derived = {}
    if build_path.is_file():
        try:
            derived = json.loads(build_path.read_text(encoding="utf-8")).get("derived_from") or {}
        except (OSError, ValueError):
            derived = {}
    expected = derived.get(key)
    trained_on = getattr(checkpoint_args, "data_root", None)
    if not expected or not trained_on:
        binding[stage] = {"status": "not_applicable", "library_derived_from": expected,
                          "checkpoint_trained_on": trained_on}
        return
    same = os.path.realpath(str(expected)) == os.path.realpath(str(trained_on))
    binding[stage] = {"status": "match" if same else "cross_release_allowed",
                      "library_derived_from": str(expected),
                      "checkpoint_trained_on": str(trained_on)}
    if not same and not allow_cross:
        raise SystemExit(
            "error: the {} checkpoint was trained on {} but the library {} was built "
            "from {}. Mixing releases silently is how a new library gets paired with "
            "old models (or the reverse); use the matching config in configs/t_line/, "
            "or pass --allow-cross-release-checkpoints for a deliberate ablation."
            .format(stage, trained_on, data_root, expected))


def _load_checkpoint(path, expected_stage, device):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("missing {} checkpoint: {}".format(expected_stage, path))
    try:
        checkpoint = torch.load(str(path), map_location="cpu", mmap=True)
    except TypeError:
        checkpoint = torch.load(str(path), map_location="cpu")
    if checkpoint.get("stage") != expected_stage:
        raise ValueError(
            "{} is a {} checkpoint, expected {}".format(
                path, checkpoint.get("stage"), expected_stage
            )
        )
    arguments = SimpleNamespace(**checkpoint["args"])
    model = planner_model(arguments) if expected_stage == "planner" else completion_model(arguments)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, arguments


class IndexedAtomicMotionLibrary:
    """Duration-aware atomic retrieval backed by memory-mapped training arrays."""

    RETRIEVAL_TIE_BREAKS = ("index", "salted")
    _LEVER_CACHE = None
    # "raw" compares the library bytes as stored (every artifact before
    # 2026-09-17); "placed" compares the candidate as build_draft will lay it
    # down.  See ``_placed_head``.
    JOIN_FRAMES = ("raw", "placed")
    # "inference" feeds the learned selector true raw units; "trained" feeds it
    # the units its checkpoint was fitted in.  See ``_selector_units``.
    SELECTOR_INPUT_UNITS = ("inference", "trained")
    # Music channel 1 is MFCC c0 (log energy): the only intensity feature that
    # predicts motion energy on the T-line train split (DEFECTS 86).
    LOUDNESS_CHANNEL = 1
    LOUDNESS_BLOCK = 60
    # Bit layout of tools/census_release_vertical.py.
    HOP_FLAG = 1
    SQUAT_FLAG = 2

    def __init__(self, data_root, retrieval_rule="duration", energy_floor_quantile=None,
                 selector=None, selector_top_k=8, selector_temperature=1.0,
                 tie_break="index", index_filler=False, join_top_k=None,
                 join_pose_weight=1.0, whole_units=False, floor_normalize=None,
                 phase_weight=0.0, rhythm_weight=0.0, feet_lead=False,
                 beat_span_tolerance=0.0, hold_by_music=False,
                 beat_fit=False, join_lever_weights=False,
                 feet_beat_lead=False, selector_join_band=0.0,
                 max_yaw_step=None, yaw_steps_path=None,
                 max_speed_spike=None, speed_spikes_path=None,
                 join_frame="raw", join_height_weight=0.0,
                 selector_input_units="inference",
                 hop_guard=False, music_energy=False, vertical_flags_path=None,
                 unit_floor_path=None, bar_units_path=None,
                 continue_source_path=None, continue_lookahead=False,
                 continue_any_label=False, continue_max_run=0, prefer_full=0.0,
                 prefer_full_mode="frames", rhythm_scorer=None, rhythm_keep=0.0,
                 rhythm_shift=0, continue_stop=False, continue_rhythm_min=0.0, continue_settle=0.0,
                 motif_return=0.0, motif_lags=(4, 2), motif_max=2,
                 continuity_scorer=None, continuity_keep=0.0, continuity_weight=0.0, motif_pick="first"):
        if retrieval_rule not in ("duration", "medoid", "random", "tempo", "phase",
                                  "learned"):
            raise ValueError("unknown retrieval rule {!r}".format(retrieval_rule))
        if join_frame not in self.JOIN_FRAMES:
            raise ValueError("unknown join frame {!r}; expected one of {}".format(
                join_frame, list(self.JOIN_FRAMES)))
        if join_height_weight and join_frame != "placed":
            # In the raw frame the head's z still carries its recording's floor
            # (0.26-0.47 m, measured 2026-09-17), so weighting it would rank
            # candidates by the floor of the upload they were cut from.
            raise ValueError("--draft-join-height-weight needs --draft-join-frame placed")
        if selector_input_units not in self.SELECTOR_INPUT_UNITS:
            raise ValueError("unknown selector input units {!r}; expected one of {}"
                             .format(selector_input_units,
                                     list(self.SELECTOR_INPUT_UNITS)))
        if (hop_guard or music_energy) and not vertical_flags_path:
            raise ValueError("--draft-hop-guard and --draft-music-energy need "
                             "--retrieval-vertical-flags")
        if retrieval_rule == "learned" and selector is None:
            # Fail closed rather than silently degrading to ``duration``: a run
            # that asked for the learned rule and got the shipped one would be
            # recorded in the manifest as ``learned`` and compared as if it
            # were.  That is the shape of defect this repository keeps paying
            # for -- a flag that is recorded but not applied.
            raise ValueError("--retrieval-rule learned needs --retrieval-selector")
        if tie_break not in self.RETRIEVAL_TIE_BREAKS:
            raise ValueError("unknown retrieval tie-break {!r}; expected one of {}"
                             .format(tie_break, list(self.RETRIEVAL_TIE_BREAKS)))
        self.retrieval_rule = retrieval_rule
        self.whole_units = bool(whole_units)
        if bar_units_path and whole_units:
            raise ValueError("--draft-bar-units and --draft-whole-units are two answers "
                             "to the same question; pass one")
        self.bar_units_path = str(bar_units_path) if bar_units_path else None
        self.continue_source_path = str(continue_source_path) if continue_source_path else None
        if continue_lookahead and not continue_source_path:
            raise ValueError("--draft-continue-lookahead needs --draft-continue-source")
        self.continue_lookahead = bool(continue_lookahead)
        self.continue_any_label = bool(continue_any_label)
        # --draft-continue-stop: a continued bar line still gets the after-seam coast (a brief stop on the
        # downbeat) -- the dancer pauses and carries on in the SAME direction, because the next bar is their
        # own continuation, so the stop cannot turn into the retract a cut produced
        self.continue_stop = bool(continue_stop)
        # --draft-continue-rhythm-min Q: continue only if the continued bar's learned alignment with THIS slot's
        # music ranks at or above the Q-quantile of the slot's own candidate pool; otherwise cut and let the
        # scorer choose.  Without it the scorer only ever judges run starts (~a quarter of the bars)
        self.continue_rhythm_min = float(continue_rhythm_min or 0.0)
        # --draft-continue-settle DEPTH: at a continued bar line, dip the PLAYBACK RATE of the dancer's own motion
        # (to 1-DEPTH, SETTLE_PEAK frames after the line, raised-cosine over SETTLE_WIDTH frames each side) and make
        # the time up over the rest of the bar.  A settle with no positional blend: the coast+fade stop left the
        # dancer behind their own continuation and a catch-up burst followed every downbeat
        self.continue_settle = float(continue_settle or 0.0)
        if self.continue_settle and self.continue_stop:
            raise ValueError("--draft-continue-settle replaces --draft-continue-stop; pass one")
        self.prefer_full = float(prefer_full or 0.0)
        if not 0.0 <= self.prefer_full < 1.0:
            raise ValueError("--draft-prefer-full is the fraction of candidates kept, in (0, 1)")
        if prefer_full_mode not in ("frames", "stops", "both", "frames_slow"):
            raise ValueError("--draft-prefer-full-mode is frames, stops, both or frames_slow")
        self.prefer_full_mode = prefer_full_mode
        # --draft-rhythm-scorer: the learned music-motion alignment score (model/rhythm_scorer.py)
        self.rhythm_keep = float(rhythm_keep or 0.0)
        self.rhythm_scorer_path = str(rhythm_scorer) if rhythm_scorer else None
        self.rhythm_shift = int(rhythm_shift or 0)
        if self.rhythm_scorer_path and not (self.rhythm_keep or self.rhythm_shift):
            raise ValueError("--draft-rhythm-scorer needs --draft-rhythm-keep and/or --draft-rhythm-shift")
        if (self.rhythm_keep or self.rhythm_shift) and not self.rhythm_scorer_path:
            raise ValueError("--draft-rhythm-keep/--draft-rhythm-shift need --draft-rhythm-scorer")
        self._rhythm = None
        # --draft-motif-return P: bring a move BACK on the phrase grid (DEFECTS §94).  Measured 2026-09-23: ground truth
        # returns a motif (a strict look-alike bar >= 2 bars away) in 19% of its bars, most at lags 2 and 4 (music bars
        # 2 and 4 apart are alike in 44/48 clips; GT motion follows, 34/48); every arm returns in 3-11%, because
        # --draft-recurrence-variety bars any recording already played in the clip, so a return can only be a fluke.
        # At a full bar k, with probability P, the unit played at bar k-lag (first lag that qualifies) is played again,
        # restretched to this slot.  Only the bar index on the beat grid is read -- music-only (CLAUDE.md 1.6).
        self.motif_return = float(motif_return or 0.0)
        if not 0.0 <= self.motif_return <= 1.0:
            raise ValueError("--draft-motif-return is a probability in [0, 1]")
        self.motif_lags = tuple(int(v) for v in (motif_lags.split(",") if isinstance(motif_lags, str) else motif_lags))
        if self.motif_return and (not self.motif_lags or min(self.motif_lags) < 2):
            raise ValueError("--draft-motif-lags must be >= 2 (lag 1 is the neighbouring bar: a stutter, not a return)")
        self.motif_max = int(motif_max)
        # --draft-motif-pick lively: bring back only a move at least as energetic as the median unit this clip has placed
        # so far.  The operator found C1's returns "有的时候不错"; on 818 the returned bar was the quiet opening groove
        # replacing A0's knee-lift turn (DEFECTS §94.3) -- a return should be a motif, not a transition.
        if motif_pick not in ("first", "lively"):
            raise ValueError("--draft-motif-pick is first or lively")
        self.motif_pick = motif_pick
        self._unit_energy_cache = {}
        # --draft-continuity-scorer: the learned "does this unit CONTINUE what was already placed" score
        # (model/continuation_scorer.py).  --draft-continuity-keep F keeps the best fraction of each slot's
        # candidates after the rhythm filter; --draft-continuity-weight W instead ranks jointly by
        # z(rhythm) + W * z(continuity) and keeps the rhythm keep fraction.  Reads the DRAFT only -- the frames this
        # clip has already placed, which are generated motion -- never the target clip's motion (CLAUDE.md 1.6).
        self.continuity_scorer_path = str(continuity_scorer) if continuity_scorer else None
        self.continuity_keep = float(continuity_keep or 0.0)
        self.continuity_weight = float(continuity_weight or 0.0)
        if not 0.0 <= self.continuity_keep < 1.0:
            raise ValueError("--draft-continuity-keep is the fraction of candidates kept, in (0, 1)")
        if self.continuity_weight < 0:
            raise ValueError("--draft-continuity-weight must be >= 0")
        if self.continuity_scorer_path and not (self.continuity_keep or self.continuity_weight):
            raise ValueError("--draft-continuity-scorer needs --draft-continuity-keep or --draft-continuity-weight")
        if (self.continuity_keep or self.continuity_weight) and not self.continuity_scorer_path:
            raise ValueError("--draft-continuity-keep/--draft-continuity-weight need --draft-continuity-scorer")
        if self.continuity_keep and self.continuity_weight:
            raise ValueError("--draft-continuity-keep filters after the rhythm scorer and --draft-continuity-weight "
                             "ranks jointly with it; pass one")
        if self.continuity_weight and not (self.rhythm_scorer_path and self.rhythm_keep):
            raise ValueError("--draft-continuity-weight ranks jointly with the rhythm scorer and keeps "
                             "--draft-rhythm-keep's fraction; it needs --draft-rhythm-scorer and --draft-rhythm-keep")
        self._continuity = None
        self._full_frames = None
        self._full_cache = {}
        self._foot_plants = None
        self._body_speed = None
        self._speed_cdf = None
        self._energy_tau = None
        self.continue_max_run = int(continue_max_run or 0)
        if (continue_any_label or continue_max_run) and not continue_source_path:
            raise ValueError("--draft-continue-any-label/--draft-continue-max-run need "
                             "--draft-continue-source")
        self.phase_weight = float(phase_weight)
        self.rhythm_weight = float(rhythm_weight)
        self.feet_lead = bool(feet_lead)
        self.beat_span_tolerance = float(beat_span_tolerance or 0.0)
        self.hold_by_music = bool(hold_by_music)
        self.beat_fit = bool(beat_fit)
        self.join_lever_weights = bool(join_lever_weights)
        self.feet_beat_lead = bool(feet_beat_lead)
        self.selector_join_band = float(selector_join_band)
        self.join_frame = join_frame
        self.join_height_weight = float(join_height_weight or 0.0)
        self.selector_input_units = selector_input_units
        self.hop_guard = bool(hop_guard)
        self.music_energy = bool(music_energy)
        self.undescribable_candidates = 0
        self.feet_lead_slots = 0
        self.feet_lead_skipped = 0
        self.feet_lead_applied = 0
        self.feet_lead_changed = 0
        self._rest_cache = {}
        self._beat_fit_cache = {}
        self._rhythm_cache = {}
        self.floor_normalize = floor_normalize
        # "index" is the default because it reproduces every artifact made
        # before this existed, byte for byte.  See ``_duration_pick``.
        self.tie_break = tie_break
        self.selector = selector
        self.selector_top_k = int(selector_top_k)
        self.selector_temperature = float(selector_temperature)
        # WHY THIS IS A KNOB NOW.  The constant's own note said "Not swept",
        # and on 2026-09-07 the sweep it was never given turned out to matter:
        # ranking by join cost and drawing from only the best few selects
        # against BIG movements, because a prototype that begins with the arms
        # overhead is far from a tail whose arms are down and so scores as a
        # poor continuation.  Measured on the 20 eval drafts, turning
        # seam-aware retrieval off entirely lifts hands-above-head 15.8% ->
        # 18.5% and wrist-above-shoulder p90 0.178 -> 0.193 against ground
        # truth's 23.5% / 0.250 and the source corpus's own 22.5% / 0.229 --
        # so the material exists and the ranking is what suppresses it.  That
        # is the operator's "不够舒展到位".  Widening the draw keeps the join
        # ranking while refusing to concentrate it on the smallest motion.
        self.join_top_k = self.JOIN_TOP_K if join_top_k is None else int(join_top_k)
        # HOW MUCH THE ABSOLUTE POSE GAP COUNTS against the velocity match.
        # The pose term is a raw rot6d distance from the previous frame, so a
        # prototype whose first frame has the arms overhead is penalised for
        # being a DIFFERENT, LARGER shape rather than for being discontinuous.
        # Since --completion-inpaint-seam-width regenerates the seam anyway,
        # the draft's job at a join is to get the DIRECTION OF TRAVEL right,
        # which is the velocity term.  0.0 makes the ranking velocity-only.
        self.join_pose_weight = float(join_pose_weight)
        self.index_filler = bool(index_filler)
        # Which candidates this clip has already played, reset per clip by
        # build_draft.  See the variety draw for why the first pick alone is
        # not enough to exclude.
        self._used_this_clip = set()
        # Which SOURCE RECORDINGS this clip has already played, same lifetime.
        #
        # WHY A SECOND SET.  ``_used_this_clip`` keys on the exact
        # ``(window, start, end)`` triple, so two picks from the same
        # performance count as different and pass the filter.  They do not look
        # different.  Measured 2026-09-07 on the clip the operator called out
        # (wild_v5:7618203431723357818:clip000): the bar at 8.23 s drew window
        # 3222 = ``wild_v5:7627866330033205883:clip000_slice13`` and the bar at
        # 13.10 s drew window 3223 = the SAME recording's ``_slice14`` -- two
        # adjacent windows of one performance, both class 5, played five
        # seconds apart.  The operator saw it as "9 秒附近和 14 秒附近的动作是
        # 一模一样的" while the triple-keyed reuse counter read 0.0%.
        #
        # The same trap had already been paid for once: the comment in the
        # plain variety draw below records occurrence 4 drawing (2972, 0, 67)
        # and occurrence 5 drawing (2973, 0, 52) -- again adjacent slices of one
        # recording (wild_v5:7622665736162993777:clip001).  Excluding the exact
        # triple only moved the replay one slice over.
        #
        # Across the 20 eval clips this was 16 of 200 units (8.0%) in 9 clips.
        # The key is ``candidate[3]``, the retrieval group id the index already
        # carries, so this costs no extra lookup.  It cannot starve a pool: the
        # thinnest class draws from 21 distinct uploads and the widest from 143,
        # against at most ~12 bars in a clip.
        self._used_groups_this_clip = set()
        self._descriptor_cache = {}
        # None == off, i.e. every artifact before 2026-09-01 reproduces exactly.
        self.energy_floor_quantile = energy_floor_quantile
        self.normalizer_path = str(Path(data_root) / "normalizer.pt")
        root = Path(data_root) / "train"
        self.motion = np.load(str(root / "motion.npy"), mmap_mode="r")
        self.labels = np.load(str(root / "labels.npy"), mmap_mode="r")
        # The training music, frame-aligned with ``motion`` -- so every
        # prototype in this library carries the beat grid of the recording it
        # was cut from.  Memory-mapped and only touched by the ``phase`` rule.
        music_path = root / "music.npy"
        self.music = (np.load(str(music_path), mmap_mode="r")
                      if music_path.is_file() else None)
        self._beat_cache = {}
        if self.motion.shape[:2] != self.labels.shape:
            raise ValueError("training motion and label arrays are not aligned")
        names_path = root / "names.json"
        if not names_path.is_file():
            raise FileNotFoundError(
                "source-safe prototype retrieval requires training names: {}".format(
                    names_path
                )
            )
        with open(str(names_path)) as handle:
            names = json.load(handle)
        if not isinstance(names, list) or len(names) != len(self.labels):
            raise ValueError("training names and label arrays are not aligned")
        self.names = tuple(names)
        self.retrieval_group_ids = self._read_retrieval_groups(root, self.names)
        self._query_groups = self._read_query_groups(Path(data_root))
        self.index = {}
        self._period_cache = {}
        self._energy_cache = {}
        # LABEL 0 IS INDEXED WHEN ASKED FOR.  The published behaviour excluded it
        # with a bare truthiness test and no comment, so the transition class
        # could never be RETRIEVED -- only bridged by --draft-gap-fill, which
        # draws a straight line (0.015 m/s) or holds a pose, across 30% of the
        # plan's frames, while ground truth on those same frames moves at 0.628
        # m/s.  Measured 2026-09-05 against ground truth AT THE SAME MOMENT IN
        # THE SAME SONG: the draft has one-second windows at literally zero
        # speed and 24.7% of its windows run below 0.6x, which is the operator's
        # "动作卡滞".  "Filler" names a span the vocabulary did not classify, NOT
        # a span where the dancer stopped -- ground truth's own filler frames
        # move as fast as its classified ones.
        #
        # Off by default because it changes which classes exist in the library,
        # and the source-safety and coverage gates are computed from that.
        # WHOLE UNITS ONLY, WHEN ASKED.  Each candidate is a run of one label
        # inside a 150-frame RELEASE WINDOW, and a window that falls in the
        # middle of a segment indexes the part it happens to contain as if it
        # were a unit.  Measured 2026-09-11 over the T library: 10,375 of
        # 15,203 candidates (68.2%) touch a window edge, and their lengths sit
        # low -- median 40 frames against 61 for whole runs, p10 nine frames,
        # which is a third of a beat.  That matters because every library unit
        # is supposed to be FOUR BEATS of its own song (runs/txy_t_seg_beat4:
        # mode grid, beats_per_segment 4), which is what makes a linear stretch
        # onto four beats of the target map beat k to beat k.  A fragment is
        # not four beats of anything, and the ``duration`` rule cannot tell the
        # two apart: three beats of a slow song has the same frame count as
        # four of a fast one.  Windows are strided 15, so a segment appears
        # whole in some window unless it is longer than 150 frames; dropping
        # the fragments leaves 4,828 candidates, median 236 per class and 81 at
        # the smallest, over 20 to 64 distinct recordings each.
        window_frames = int(np.asarray(self.labels).shape[1])
        dropped_fragments = 0
        # --draft-bar-units: one candidate per SOURCE BAR that lies wholly inside
        # a window, instead of per label run.  A label run clipped by the window
        # edge is a fragment whose cut lands mid-bar: measured 2026-09-22 on
        # fix7's 101 retrieved units, 31 were such fragments and their cut edges
        # sat a median 7 frames from the source's nearest bar line (45% >= 10
        # frames; a bar is ~55), against 0 frames / 99% within 2 for edges at a
        # run boundary -- i.e. the unit starts or stops MID-MOVE, the operator's
        # "做到一半就收".  --draft-whole-units drops those runs instead, and
        # because multi-bar runs then stay whole, four-beat candidates run short:
        # 17 of 101 slots found none of the right beat count and 16.8% of bars
        # came back a wrong whole number of beats (0.0% with fragments kept).
        #
        # SO THE BAR INDEX IS A SECOND INDEX, NOT A REPLACEMENT.  Replacing the
        # run index outright read 17.8% wrong beat counts too -- and every one of
        # those was a SLOT that is not four beats (a clip's partial first/last
        # bar, 0.3-2.5 beats, or an odd bar of the query's grid, 4.6-7.7), which
        # no whole bar can fill while a fragment can.  In fix8 the 83 regular
        # four-beat slots held 13 fragments and the 18 irregular ones held 18.
        # ``retrieve`` therefore takes whole bars for a slot of the bar's beat
        # count and the run index for anything else.
        bars_by_window = (self._bars_by_window(Path(data_root), window_frames)
                          if self.bar_units_path else None)
        self.bar_index = {} if bars_by_window is not None else None
        seen_bars = set()
        self.bar_units_indexed = 0
        self.bar_units_mixed = 0
        for sample_index, labels in enumerate(self.labels):
            if bars_by_window is not None:
                for start, end, key in bars_by_window.get(sample_index, ()):
                    if key in seen_bars:
                        continue          # the same bar, whole in an overlapping window
                    seen_bars.add(key)
                    run = np.asarray(labels[start:end])
                    label = int(run[0])
                    if (run != label).any():
                        # Not guessed: a bar the label track splits is counted
                        # (into the manifest) and left out.
                        self.bar_units_mixed += 1
                        continue
                    if not (label or self.index_filler):
                        continue
                    self.bar_index.setdefault(label, []).append(
                        (sample_index, start, end, self.retrieval_group_ids[sample_index]))
                    self.bar_units_indexed += 1
            plan = torch.from_numpy(np.array(labels, dtype=np.int64, copy=True))
            for segment in labels_to_segments(plan):
                if not (segment.label or self.index_filler):
                    continue
                if self.whole_units and (segment.start == 0
                                         or segment.end == window_frames):
                    dropped_fragments += 1
                    continue
                self.index.setdefault(segment.label, []).append(
                    (
                        sample_index,
                        segment.start,
                        segment.end,
                        self.retrieval_group_ids[sample_index],
                    )
                )
        self.dropped_fragments = dropped_fragments
        # --draft-continue-source: every label-pure source bar, keyed by where it
        # starts in its RECORDING, so a unit's own next bar can be found from the
        # unit alone.  Built from the same segmentation --draft-bar-units reads.
        self._source_bars = None
        self._window_origin = {}
        if self.continue_source_path:
            self._build_source_bars(Path(data_root), int(np.asarray(self.labels).shape[1]))
        if self.whole_units:
            # Fail closed.  An emptied class would be filled by whatever the
            # caller's fallback is and recorded in the manifest as though the
            # flag had simply worked.
            empty = [label for label, entries in self.index.items() if not entries]
            if empty or not self.index:
                raise ValueError(
                    "--draft-whole-units emptied {} label pool(s) ({}); the "
                    "release's windows must be too short to contain them whole"
                    .format(len(empty), empty[:8]))
        self._sample_floor = None
        if floor_normalize:
            # LEVEL THE CORPUS.  A monocular reconstruction has no absolute
            # height, and the T line's recordings disagree about where the
            # ground is: the floor -- the 5th percentile of the lowest foot, the
            # same rule ``anchor_floor`` and ``render_avatar_video.floor_of``
            # use -- runs from -1.074 m at the 5th percentile of recordings to
            # -0.883 at the 95th, a **0.192 m spread** (295 sequences,
            # data/wild3d/txy_t_normalized/recording_floors.json).
            #
            # WHY IT REACHES THE PICTURE.  Retrieval pastes a bar from recording
            # A into a clip built on recording B, and ``--draft-root-continuity``
            # with z makes the ROOT continuous at the seam, which is the wrong
            # invariant: when two prototypes hold their feet different distances
            # below the root, aligning the roots lifts the feet off the floor for
            # the rest of the bar.  Measured on 7412632116703350028:clip001,
            # median height of the lowest foot above the render floor: ground
            # truth 0.072 m, the shipped arm 0.070, ``--draft-join-pose-weight
            # 0.25`` 0.140, and with the seam-aware ranking off **0.439** -- while
            # every arm's own 5th percentile sits ON the floor, so
            # ``--floor-anchor`` is working and what floats is everything above it.
            #
            # Expressing every prototype relative to ITS OWN recording's floor
            # makes the vertical frame shared, so the feet can be left where the
            # dancer put them instead of being chased by the root.
            floors = json.load(open(str(floor_normalize)))["floors"]
            missing, values = [], []
            for name in self.names:
                sequence = name.rsplit("_slice", 1)[0]
                if sequence not in floors:
                    missing.append(sequence)
                values.append(floors.get(sequence, 0.0))
            if missing:
                # Fail closed: a sample silently left at 0.0 would be the one
                # prototype still carrying its own calibration, and it would be
                # recorded in the manifest as though the whole library had been
                # levelled.
                raise ValueError(
                    "--draft-floor-normalize has no floor for {} source "
                    "recording(s), first {}".format(len(set(missing)),
                                                    sorted(set(missing))[:3]))
            self._sample_floor = np.asarray(values, dtype=np.float32)

        # REFUSE PROTOTYPES THAT SNAP.  2026-09-14 the operator saw "人体旋转的
        # 跳变,视觉上看着不连续,缺帧" in a render.  One source was
        # ``face_camera``'s wrapped error; the other is the LIBRARY.  Censused
        # with the repository's own decode and the same ``_body_forward_yaw``
        # the judgement uses (tools/census_release_yaw_steps.py), 985 of 6553
        # windows -- 15.0% -- turn more than 30 degrees in ONE frame and 94
        # turn more than 90, worst 174.6.  At 30 fps that is 5238 deg/s, which
        # no body does; the ten held-out ground-truth clips never exceed 25.2
        # deg/frame.  It is a reconstruction artefact and retrieval pastes it
        # into the dance verbatim.
        #
        # PER FRAME, not per window: blacklisting a whole 150-frame window for
        # one bad frame would throw away 15% of the library, and a candidate is
        # a sub-range.  Only the range a candidate actually plays is asked
        # about.
        self._yaw_steps = None
        self.max_yaw_step = float(max_yaw_step) if max_yaw_step else None
        self.yaw_rejected = 0
        self.yaw_slots = 0
        self.yaw_exhausted = 0
        if self.max_yaw_step is not None:
            if not yaw_steps_path:
                raise ValueError("--retrieval-max-yaw-step needs --retrieval-yaw-steps")
            loaded = np.load(str(yaw_steps_path))
            self._yaw_steps = {name: loaded[name] for name in loaded.files}
            missing = [n for n in self.names if n not in self._yaw_steps]
            if missing:
                # Fail closed, the same reason --draft-floor-normalize does: a
                # window silently treated as clean would be the one prototype
                # still carrying the artefact, recorded as though the whole
                # library had been screened.
                raise ValueError(
                    "--retrieval-yaw-steps has no reading for {} window(s), "
                    "first {}".format(len(missing), missing[:3]))
        # REFUSE PROTOTYPES THAT JOLT.  The operator, 2026-09-16: "动作加速...
        # 是偶尔的出现".  Measured in the 3D motion, not in any render: sliced
        # into 150-frame windows, ground truth's worst frame-relative speed is
        # 3.12x and it NEVER reaches 4x (0 of 62 windows), while the library
        # reaches 13.4x with 5.8% of windows above 4x.  Our generated clips
        # inherit it -- 0.22% of frames above 2.5x against ground truth's 0.07%,
        # worst 9.8x against 2.8x -- and only 17% of those coincide with the yaw
        # snaps above, so it is a SEPARATE defect, not the same one seen twice.
        self._speed_spikes = None
        self.max_speed_spike = float(max_speed_spike) if max_speed_spike else None
        self.speed_rejected = 0
        self.speed_slots = 0
        self.speed_exhausted = 0
        if self.max_speed_spike is not None:
            if not speed_spikes_path:
                raise ValueError("--retrieval-max-speed-spike needs "
                                 "--retrieval-speed-spikes")
            loaded = np.load(str(speed_spikes_path))
            self._speed_spikes = {name: loaded[name] for name in loaded.files}
            missing = [n for n in self.names if n not in self._speed_spikes]
            if missing:
                raise ValueError(
                    "--retrieval-speed-spikes has no reading for {} window(s), "
                    "first {}".format(len(missing), missing[:3]))
        # PER-FRAME LOCAL FLOOR, for --draft-unit-floor (see _candidate_floor).
        self._local_floor = None
        self._local_floor_cache = {}
        if unit_floor_path:
            loaded = np.load(str(unit_floor_path))
            self._local_floor = {name: loaded[name] for name in loaded.files}
            missing = [n for n in self.names if n not in self._local_floor]
            if missing:
                raise ValueError(
                    "--draft-unit-floor has no reading for {} window(s), "
                    "first {}".format(len(missing), missing[:3]))
        # PER-FRAME HOP / DEEP-SQUAT FLAGS, from tools/census_release_vertical.py.
        # Needed by --draft-hop-guard and --draft-music-energy; fail closed for
        # the same reason as the two screens above.
        self._vertical_flags = None
        self.quiet_loudness = None
        if vertical_flags_path:
            loaded = np.load(str(vertical_flags_path))
            self._vertical_flags = {name: loaded[name] for name in loaded.files}
            missing = [n for n in self.names if n not in self._vertical_flags]
            if missing:
                raise ValueError(
                    "--retrieval-vertical-flags has no reading for {} window(s), "
                    "first {}".format(len(missing), missing[:3]))
        if self.music_energy:
            # WHERE "QUIET" STARTS, from the library's own music and nothing
            # else: the median over every training window of the mean loudness
            # (channel 1, MFCC c0) in 60-frame blocks, about one bar at 120 BPM.
            # One line for every query, so a soft song has mostly quiet bars
            # and a loud one mostly loud bars -- the song-level contrast the
            # operator asked for.  A per-clip median would give every song the
            # same half-and-half split and erase that contrast.
            if self.music is None or self.music.shape[-1] <= self.LOUDNESS_CHANNEL:
                raise ValueError("--draft-music-energy needs the library's "
                                 "train/music.npy with a loudness channel")
            block = self.LOUDNESS_BLOCK
            usable = (self.music.shape[1] // block) * block
            track = np.asarray(self.music[:, :usable, self.LOUDNESS_CHANNEL],
                               dtype=np.float64)
            self.quiet_loudness = float(np.median(
                track.reshape(len(track), -1, block).mean(axis=2)))
        self._retrieval_cache = {}
        # What each retrieval unit actually got, appended in plan order.  This
        # exists because the shipped artifact recorded the plan and the motion
        # but NOT which prototype filled each slot nor how far it was stretched,
        # so the 2026-09-05 defect below could only be found by replaying
        # retrieval offline against the same library.
        #
        # THE DEFECT.  ``_values_at`` resamples unconditionally.  A plan run
        # longer than anything its class can supply is therefore played in slow
        # motion rather than refused.  The ceiling is structural: the release
        # materialises 150-frame windows, so no segment can cross one and the
        # longest prototype in the whole T library is exactly 150 frames =
        # 5.00 s (196 of 12,145 candidates sit on that cap).  Measured on the
        # 18 sourced eval clips, 9 of 88 units (10.2%) ask for more than that;
        # the worst asks for 312 frames, is handed the 150-frame maximum, and
        # plays the last 10.4 s of a 14.3 s clip at 0.48x speed.  Nothing
        # reported it -- the pooled stretch median is 1.000 and the p90 is
        # 1.041, so a summary statistic hides it completely.
        #
        # A long run is NOT by itself illegitimate: ground truth's own label
        # runs exceed 5.0 s on 4.4% of runs (max 12.30 s).  What is illegitimate
        # is answering "I need 10.4 s" with "here is 5 s at half speed" and
        # saying nothing, so this records every unit and
        # ``retrieval_stretch_summary`` turns it into a gate that can fail.
        self.retrieval_log = []
        self._retrieval_record_cache = {}

    def retrieval_stretch_summary(self):
        """Turn ``retrieval_log`` into a gate reading, or ``None`` if unused.

        ``units_over_library_ceiling`` is the one that matters and it is a COUNT,
        not a rate: the pooled stretch median stays 1.000 while a single unit
        halves the tempo of two thirds of a clip, so a rate would read as clean.
        """
        if not self.retrieval_log:
            return None
        stretches = np.array([r["stretch"] for r in self.retrieval_log], float)
        capped = [r for r in self.retrieval_log if r["slot_exceeds_pool_max"]]
        return {
            "units": len(self.retrieval_log),
            "stretch_median": float(np.median(stretches)),
            "stretch_p90": float(np.percentile(stretches, 90)),
            "stretch_max": float(stretches.max()),
            "playback_min": float(1.0 / stretches.max()),
            "units_over_library_ceiling": len(capped),
            "worst": (max(self.retrieval_log, key=lambda r: r["stretch"])
                      if self.retrieval_log else None),
            # Every unit in plan order.  A summary cannot answer "which move was
            # slowed and where in the clip", which is the question a label lane
            # under the video needs, so the list travels with the artifact; it
            # is a handful of small dicts per clip.
            "units_detail": [dict(r) for r in self.retrieval_log],
        }

    @staticmethod
    def _read_retrieval_groups(root, names):
        """Load explicit indexed provenance, leaving legacy layouts unknown."""
        groups_path = root / "retrieval_groups.json"
        if not groups_path.is_file():
            return tuple(None for _ in names)
        try:
            with open(str(groups_path), encoding="utf-8") as handle:
                groups = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("cannot read retrieval groups {}: {}".format(groups_path, error)) from error
        if (
            not isinstance(groups, list)
            or len(groups) != len(names)
            or any(not isinstance(group, str) or not group.strip() for group in groups)
        ):
            raise ValueError("training retrieval groups and arrays are not aligned")
        return tuple(groups)

    @classmethod
    def _read_query_groups(cls, data_root):
        """Register only declared sample-name -> group mappings.

        The registry spans all splits so an evaluation sample can identify its
        group without deriving a recording ID from its string.  A legacy split
        without a sidecar contributes no mapping and therefore remains an
        unknown, fail-closed inference query.

        **Two key spaces, and until 2026-08-16 only one of them was registered.**
        ``names.json`` holds *window* names (``<recording>_slice7``), while an
        inference query is named for the *recording* (``wild_v4:123:clip000``)
        or, on AIST, for the song it was generated from.  So every lookup missed,
        every query resolved to ``None``, and ``_source_safe_draft`` fell through
        to its zero-mask branch -- the fail-closed path meant for a query whose
        provenance cannot be proved.

        The consequence is not a degraded draft, it is **no draft at all**: the
        completion model takes music, draft and mask, never labels, so with an
        empty draft the planner's output cannot reach the motion.  Measured:
        200 clips generated twice with deliberately different plans (planner
        window stride 150 vs 15) produced 200/200 *byte-identical* motion from
        200/200 different plans.  Every FID in this repo, on both corpora, was
        therefore produced by the completion stage conditioned on music alone.

        ``windows.jsonl`` carries ``recording_id`` and ``retrieval_group_id`` per
        window, so the recording-level mapping is *declared* rather than derived
        -- which is what the paragraph above requires.  It is read when present
        and both key spaces are registered.
        """
        groups_by_name = {}
        windows_path = data_root / "windows.jsonl"
        if windows_path.is_file():
            try:
                for line in windows_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    name = row.get("recording_id")
                    group = row.get("retrieval_group_id")
                    if not isinstance(name, str) or not name.strip() or group is None:
                        continue
                    previous = groups_by_name.get(name)
                    if previous is not None and previous != group:
                        raise ValueError(
                            "recording {!r} has conflicting retrieval groups".format(name))
                    groups_by_name[name] = group
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError("cannot read {}: {}".format(windows_path, error)) from error
        for split in ("train", "val", "test"):
            root = data_root / split
            names_path = root / "names.json"
            groups_path = root / "retrieval_groups.json"
            if not names_path.is_file() or not groups_path.is_file():
                continue
            try:
                with open(str(names_path), encoding="utf-8") as handle:
                    names = json.load(handle)
                groups = cls._read_retrieval_groups(root, names)
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError("cannot read query retrieval groups under {}: {}".format(root, error)) from error
            for name, group in zip(names, groups):
                if not isinstance(name, str) or not name.strip():
                    raise ValueError("retrieval group sidecar has invalid sample name under {}".format(root))
                previous = groups_by_name.get(name)
                if previous is not None and previous != group:
                    raise ValueError("sample {!r} has conflicting retrieval groups".format(name))
                groups_by_name[name] = group
        return groups_by_name

    def query_retrieval_group_id(self, sample_name):
        """Return an explicitly declared query group, never a name heuristic."""
        if not isinstance(sample_name, str) or not sample_name.strip():
            return None
        return self._query_groups.get(sample_name)

    @property
    def labels_available(self):
        return set(self.index)

    # How many candidates a content-aware rule may look at.  The medoid is
    # O(n^2) in this number and the answer converges long before a class's full
    # membership; the cap is reported in the artifact so a reader can tell a
    # medoid from a sample of one.
    MEDOID_CANDIDATE_CAP = 24

    def _medoid_index(self, candidates, target_length):
        """The candidate closest to the others -- the class's typical member.

        The shipped rule picks by duration alone and never looks at the motion,
        which the in-class oracle says costs 20%: a rule allowed to see the true
        answer sits that much closer.  This one is not allowed to see it, and
        collects about half of that gap -- measured on the wild release, 60 val
        recordings at candidate cap 32, median geodesic 0.4050 rad -> 0.3613,
        which is 51.0% of the oracle's headroom.

        Candidates beyond the cap are sampled at an even stride rather than
        truncated: the index is built in training-window order, so the first N
        are the first recordings and a medoid over them is a medoid over a few
        dancers.  The stride is deterministic, so two runs with the same inputs
        still return the same prototype.
        """
        from tools.score_generation_diagnostics import _rotations_from_151, geodesic_rad

        if len(candidates) == 1:
            return 0
        if len(candidates) > self.MEDOID_CANDIDATE_CAP:
            stride = len(candidates) // self.MEDOID_CANDIDATE_CAP
            picks = list(range(0, len(candidates), stride))[: self.MEDOID_CANDIDATE_CAP]
        else:
            picks = list(range(len(candidates)))
        rotations = []
        for index in picks:
            values = self._values_at(candidates[index], target_length)
            rotations.append(_rotations_from_151(values, self.normalizer_path))
        best, best_sum = picks[0], None
        for position, a in enumerate(rotations):
            total = sum(geodesic_rad(a, b) for other, b in enumerate(rotations)
                        if other != position)
            if best_sum is None or total < best_sum:
                best, best_sum = picks[position], total
        return best

    # rot6d starts after the contact and root-position channels; the FIRST
    # block is SMPL's global orientation, i.e. which way the whole body faces.
    GLOBAL_ORIENT_START = ROOT_POSITION_START + ROOT_POSITION_DIMS

    def _normalizer_affine(self):
        """``raw = normalized * scale + offset``, cached.

        The release normalizer is per-dimension min-max, so unnormalising is an
        affine map per dimension.  That is why ``root_continuity`` can add a
        constant directly in normalized space -- but a ROTATION mixes dimensions
        whose scales differ, so facing alignment cannot be done there and has to
        go through this.
        """
        if getattr(self, "_affine", None) is None:
            normalizer = torch.load(self.normalizer_path, map_location="cpu")
            low = normalizer["data_min"].float()
            high = normalizer["data_max"].float()
            span = torch.where(high == low, torch.ones_like(high), high - low)
            self._affine = (span / 2.0, low + span / 2.0)
        return self._affine

    def _facing_yaw(self, raw_values):
        """World yaw of the global orientation, radians, per frame.

        z is up, verified rather than assumed: applying ``Rz @ R`` to this block
        and ``Rz`` to the root translation rotates the DECODED joint positions
        about the world z axis to 6.3e-07 -- float noise -- while the y-axis
        alternative is off by 1.77 m.
        """
        from dataset.rotation_ops import rotation_6d_to_matrix

        start = self.GLOBAL_ORIENT_START
        matrices = rotation_6d_to_matrix(raw_values[:, start:start + 6])
        return torch.atan2(matrices[:, 1, 0], matrices[:, 0, 0]), matrices

    def _rotate_about_z_varying(self, values, deltas):
        """Turn a segment by a DIFFERENT angle on every frame.

        WHY IT EXISTS.  ``facing_anchor`` used to take its whole correction out
        of the alignment constant, which put all of it in the seam's single
        frame.  Measured on 7030793823240424742:clip000 at t=12.47 s: the
        accumulated drift was -206.4 degrees, the anchor's share
        0.6 x 206.4 = 123.8, and the rendered clip turned **110.4 degrees in one
        frame** against that clip's ground-truth maximum of 17.7.  On screen the
        dancer's back becomes her front with nothing in between -- the operator,
        2026-09-12: "渲染存在1次姿态的跳变。未有衔接motion 过度".  Spreading the
        same correction over the segment keeps the drift control and removes the
        jump; over a 47-frame bar, 123.8 degrees is 2.6 degrees a frame, inside
        ground truth's ordinary range (median 0.93, p99 13.24).

        THE ROOT TURNS WITH THE BODY, as in ``_rotate_about_z``, but it cannot
        be done by rotating positions about a fixed point when the angle varies:
        that would stretch the path.  Each frame's DISPLACEMENT is rotated by
        that frame's angle and the path re-integrated from the first frame, so
        the distance walked between any two frames is unchanged and only its
        direction turns.  Rotating the body without its travel is what creates
        foot skate (see ``fix_foot_skate``).
        """
        from dataset.rotation_ops import matrix_to_rotation_6d

        scale, offset = self._normalizer_affine()
        raw = values * scale + offset
        _, matrices = self._facing_yaw(raw)
        cos, sin = torch.cos(deltas), torch.sin(deltas)
        turn = torch.zeros(len(deltas), 3, 3, dtype=raw.dtype)
        turn[:, 0, 0] = cos
        turn[:, 0, 1] = -sin
        turn[:, 1, 0] = sin
        turn[:, 1, 1] = cos
        turn[:, 2, 2] = 1.0
        start = self.GLOBAL_ORIENT_START
        raw[:, start:start + 6] = matrix_to_rotation_6d(turn @ matrices)
        root = slice(ROOT_POSITION_START, ROOT_POSITION_START + ROOT_POSITION_DIMS)
        positions = raw[:, root].clone()
        steps = positions[1:] - positions[:-1]
        turned = (turn[1:] @ steps.unsqueeze(-1)).squeeze(-1)
        rebuilt = torch.empty_like(positions)
        rebuilt[0] = positions[0]
        if len(positions) > 1:
            rebuilt[1:] = positions[0] + torch.cumsum(turned, dim=0)
        raw[:, root] = rebuilt
        return (raw - offset) / scale

    def _rotate_about_z(self, values, delta):
        """Turn a whole retrieved segment by ``delta`` about the vertical axis.

        Rotates the global orientation and the root's ground track together --
        turning a dancer turns where they walk as well as where they look --
        and about the segment's OWN first frame, so the first frame's position
        is unchanged and ``root_continuity`` still composes with this untouched.
        """
        from dataset.rotation_ops import matrix_to_rotation_6d

        scale, offset = self._normalizer_affine()
        raw = values * scale + offset
        _, matrices = self._facing_yaw(raw)
        cos, sin = torch.cos(delta), torch.sin(delta)
        turn = torch.tensor([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]],
                            dtype=raw.dtype)
        start = self.GLOBAL_ORIENT_START
        raw[:, start:start + 6] = matrix_to_rotation_6d(turn @ matrices)
        root = slice(ROOT_POSITION_START, ROOT_POSITION_START + ROOT_POSITION_DIMS)
        anchor_point = raw[0, root].clone()
        raw[:, root] = (turn @ (raw[:, root] - anchor_point).T).T + anchor_point
        return (raw - offset) / scale

    def _descriptor(self, candidate):
        """A cached ``SegmentDescriptor`` for one candidate, in RAW units.

        Raw, not normalized: the selector's canonicalisation is a rotation, and
        the release normalizer is per-dimension min-max, so a rotation is only
        meaningful before it -- the same reason ``_rotate_about_z``
        unnormalizes first.  It also makes the selector independent of WHICH
        normalizer a release carries, so a selector fitted on the recording-level
        store transfers to a re-normalized release unchanged.

        The cache is per library instance and keyed by the candidate tuple.
        Overlapping training windows put near-duplicate spans in the index; they
        are described separately, exactly as every other rule already treats
        them.
        """
        from model.retrieval_selector import describe_segment

        cached = self._descriptor_cache.get(candidate)
        if cached is not None:
            return cached
        sample, start, end, _ = candidate
        values = torch.from_numpy(np.array(self.motion[sample, start:end], copy=True))
        if len(values) < 2:
            # A ONE-FRAME CANDIDATE CANNOT BE DESCRIBED, and until this path was
            # first run nothing had ever asked it to: ``describe_segment`` needs
            # a difference to read travel and contact changes from, and raised
            # ValueError, which took the whole arm down at the 15th clip of 20.
            # Returning None lets the caller fall back to the shipped rule for
            # that one slot rather than losing the run -- and the count is
            # reported, so a pool that is mostly unscoreable cannot pass as a
            # working selector.
            self._descriptor_cache[candidate] = None
            self.undescribable_candidates += 1
            return None
        phase, period = self._candidate_beat(candidate)
        descriptor = describe_segment(self._selector_units(values, candidate),
                                      beat_phase=phase, beat_period=period)
        self._descriptor_cache[candidate] = descriptor
        return descriptor

    def _selector_units(self, values, candidate):
        """One candidate's NORMALIZED release rows, in the units the selector reads.

        ``inference`` (every artifact before 2026-09-17) is true raw units.

        ``trained`` is what the shipped checkpoint was fitted on.
        ``tools/train_retrieval_selector.py`` read ``motion_151_raw.npy`` --
        already raw, its manifest says so -- and applied the release's
        unnormalize map to it a SECOND time.  Proven 2026-09-17: rebuilding the
        trainer's corpus with its own code and seed reproduces the checkpoint's
        ``feature_mean``/``feature_std`` to 0.0 only in those doubled units
        (0.319 off in true units).  The same 80 held-out pairs score recall@1
        1.000 as trained and 0.7125 when fed true units, and on fix5's real
        pools 10-12% of the seam joint features sat beyond 5 SD of anything
        the model saw.  Mapping the inputs into its units is the fix that needs
        no retraining; retraining in true units is the other.

        THE FLOOR COMES OFF FIRST in this mode.  Training compared a tail and a
        head from ONE recording, so the floor cancelled in the height step; at
        inference the tail is floor-subtracted (``_values_at``) and the raw head
        is not, which put the step a median +0.35 m (+8.4 SD) off.  Subtracting
        the candidate's own floor makes both sides floor-relative again.
        """
        scale, offset = self._normalizer_affine()
        raw = values * scale + offset
        if self.selector_input_units == "trained":
            raw = raw.clone()
            floor = self._candidate_floor(candidate)
            if floor is not None:
                raw[:, ROOT_POSITION_START + 2] -= floor
            raw = raw * scale + offset
        return raw

    def _selector_tail(self, previous_tail):
        """``previous_tail`` (raw, floor-subtracted) in the selector's units."""
        if previous_tail is None or self.selector_input_units != "trained":
            return previous_tail
        scale, offset = self._normalizer_affine()
        return previous_tail * scale + offset

    def _query_context(self, segments, position, segment, previous_tail,
                       previous_end, segment_phase, beat_period, onset):
        """The side of the join that does not depend on which candidate wins.

        ``next_label`` is the next NON-transition class in the plan.  Its
        identity is unknown here -- that prototype has not been retrieved yet --
        but its class is, and the selector carries each class's mean way of
        starting (contacts and limb activity, both invariant to which way the
        source recording faced).  That is the "look at what comes after" term.
        """
        from model.retrieval_selector import QueryContext, window_profile

        next_label = 0
        for later in segments[position + 1:]:
            if later.label != 0:
                next_label = int(later.label)
                break
        contacts, activity = self.selector.stats_for(next_label)
        profile = window_profile(onset, segment.start, segment.end)
        period = float(beat_period) if beat_period else float("nan")
        return QueryContext(
            previous_tail=self._selector_tail(previous_tail),
            target_length=int(segment.length),
            gap_frames=(0 if previous_end is None
                        else max(0, int(segment.start) - int(previous_end))),
            beat_phase=(float(segment_phase) if segment_phase is not None
                        else float("nan")),
            beat_period=period,
            onset_mean=float(profile.mean()),
            onset_profile=profile,
            beats_in_span=(segment.length / period
                           if period == period and period > 0 else float("nan")),
            next_contacts=contacts,
            next_activity=activity)

    def _duration_band(self, candidates, target_length):
        """The pool every content-aware rule here already draws from.

        ``max(2, 0.15 * target_length)`` is not a new number: ``tempo``,
        ``phase`` and ``--draft-recurrence-variety`` all use it, so a rule built
        on this band cannot be accused of narrowing the pool relative to
        shipped behaviour -- which is what sank ``--retrieval-rule phase``
        (docs/DANCE_QUALITY_DEFECTS.md section 15.8).
        """
        lengths = [abs((c[2] - c[1]) - target_length) for c in candidates]
        slack = max(2.0, 0.15 * target_length)
        near = [c for c, d in zip(candidates, lengths) if d <= slack]
        if not near:
            best = min(lengths)
            near = [c for c, d in zip(candidates, lengths) if d <= best]
        return near

    def _reject_yaw_snaps(self, candidates):
        """Drop candidates whose own played range turns faster than a body can.

        Counted, not just applied: three separate switches in this file were
        recorded in a manifest while never reaching the code, and each time the
        READING looked plausible.  ``yaw_slots`` is how often this was asked,
        ``yaw_rejected`` how many candidates it actually removed, and
        ``yaw_exhausted`` how often a label had nothing left -- which is not an
        error, because a slot must still be filled, but it is the number that
        says the threshold is too tight.
        """
        if self._yaw_steps is None or not candidates:
            return candidates
        self.yaw_slots += 1
        kept = []
        for candidate in candidates:
            sample, start, end, _ = candidate
            series = self._yaw_steps[self.names[sample]]
            played = series[int(start):max(int(start), int(end) - 1)]
            if len(played) == 0 or float(played.max()) <= self.max_yaw_step:
                kept.append(candidate)
        self.yaw_rejected += len(candidates) - len(kept)
        if not kept:
            self.yaw_exhausted += 1
            return candidates
        return tuple(kept)

    def _reject_speed_spikes(self, candidates):
        """Drop candidates whose own played range jolts harder than a dance does."""
        if self._speed_spikes is None or not candidates:
            return candidates
        self.speed_slots += 1
        kept = []
        for candidate in candidates:
            sample, start, end, _ = candidate
            series = self._speed_spikes[self.names[sample]]
            played = series[int(start):max(int(start), int(end) - 1)]
            if len(played) == 0 or float(played.max()) <= self.max_speed_spike:
                kept.append(candidate)
        self.speed_rejected += len(candidates) - len(kept)
        if not kept:
            self.speed_exhausted += 1
            return candidates
        return tuple(kept)

    def _candidate_floor(self, candidate):
        """Metres to take off this candidate's root z, or ``None`` to leave it.

        ``--draft-unit-floor`` (when given) wins over ``--draft-floor-normalize``:
        the median of the LOCAL floor over the frames this unit plays, instead of
        one floor for the whole upload.

        WHY.  The recording floor is the 5th percentile of the lowest foot over
        the entire recording, and a monocular reconstruction drifts in depth, so
        a bar cut from a stretch that drifted upward is pasted in floating.
        Measured 2026-09-17 on fix5's 20 eval drafts (194 units of >= 10
        frames): the unit's local-floor offset from its recording floor predicts
        how high that unit's feet sit relative to its neighbours at Pearson
        +0.973, slope 0.93; subtracting it takes the spread from median 3.2 cm,
        max 19.4 cm, 10 units over 10 cm, to 0.5 cm, 4.8 cm and none.  Those
        floating bars were most of what the hop detector read in the draft (only
        33 of 375 airborne frames came from a unit that really hops) and the
        source of the pelvis steps the root-seam smoothing then has to ramp.
        """
        sample, start, end = candidate[0], candidate[1], candidate[2]
        if self._local_floor is not None:
            key = (sample, start, end)
            cached = self._local_floor_cache.get(key)
            if cached is None:
                series = np.asarray(self._local_floor[self.names[sample]]
                                    [int(start):max(int(start) + 1, int(end))],
                                    dtype=np.float64)
                cached = float(np.nanmedian(series))
                self._local_floor_cache[key] = cached
            return cached
        if self._sample_floor is not None:
            return float(self._sample_floor[sample])
        return None

    def _build_source_bars(self, data_root, window_frames):
        """{(sequence_id, absolute bar start): [(array index, local start, local end, label)]}
        for every label-pure bar of ``--draft-continue-source``'s segmentation lying
        wholly inside a train window, plus each window's (sequence_id, start_frame)."""
        with open(self.continue_source_path) as handle:
            segmentation = json.load(handle)
        bounds = {record["sequence"]: [int(b) for b in record["boundaries"]]
                  for record in segmentation["records"]}
        self._source_bars = {}
        with open(str(Path(data_root) / "windows.jsonl")) as handle:
            for line in handle:
                window = json.loads(line)
                if window.get("split") != "train":
                    continue
                index = int(window["array_index"])
                first = int(window["start_frame"])
                self._window_origin[index] = (window["sequence_id"], first)
                _, recording, clip = str(window["sequence_id"]).split(":")
                edges = bounds.get("{}__{}".format(recording, clip))
                if edges is None:
                    continue
                labels = np.asarray(self.labels[index])
                for start, end in zip(edges[:-1], edges[1:]):
                    if first <= start and end <= first + window_frames:
                        run = labels[start - first:end - first]
                        if len(run) and (run == run[0]).all():
                            self._source_bars.setdefault((window["sequence_id"], start), []).append(
                                (index, start - first, end - first, int(run[0])))

    def _source_span(self, candidate):
        """(recording, first frame, end frame) of ``candidate`` in its recording's own frames, or None."""
        origin = getattr(self, "_window_origin", {}).get(int(candidate[0])) if candidate is not None else None
        if origin is None:
            return None
        return (origin[0], origin[1] + int(candidate[1]), origin[1] + int(candidate[2]))

    def _next_source_bar(self, candidate, label, target_length, slack_frames=2):
        """The bar its own dancer danced right after ``candidate``, if that bar has the
        planned ``label`` and fits ``target_length`` inside the duration band; else None.

        Continuing the source is the only way a bar line stops being a CUT between
        two unrelated dancers (DEFECTS 92): every cut makes the body travel from one
        dancer's arrival to the next one's start, a path no dancer performed, and
        that travel is where the output's extra wrist turn-backs sit (0..16 frames
        after the seam: 0.71 of seams against ground truth's 0.48 at the same frames).
        """
        if self._source_bars is None:
            return None
        origin = self._window_origin.get(int(candidate[0]))
        if origin is None:
            return None
        sequence, first = origin
        end = first + int(candidate[2])
        slack = max(2.0, 0.15 * float(target_length))
        for offset in sorted(range(-slack_frames, slack_frames + 1), key=abs):
            for index, start, stop, bar_label in self._source_bars.get((sequence, end + offset), ()):
                if abs((stop - start) - float(target_length)) > slack:
                    continue
                if bar_label != int(label) and not self.continue_any_label:
                    continue
                found = (index, start, stop, self.retrieval_group_ids[index])
                if self._passes_snap_filters(found):
                    return found
        return None

    def _passes_snap_filters(self, candidate):
        """The yaw-snap and speed-spike tests for ONE candidate, uncounted.  Not
        ``_reject_*``: those return the whole input when nothing passes (a slot must
        still be filled) and count every call into the manifest."""
        sample, start, end, _ = candidate
        for series_map, limit in ((self._yaw_steps, self.max_yaw_step),
                                  (self._speed_spikes, self.max_speed_spike)):
            if series_map is None:
                continue
            played = series_map[self.names[sample]][int(start):max(int(start), int(end) - 1)]
            if len(played) and float(played.max()) > limit:
                return False
        return True

    def _bars_by_window(self, data_root, window_frames):
        """{train array index: [(start, end, (sequence, bar start)), ...]} for every
        bar of ``--draft-bar-units``'s segmentation lying wholly inside that
        window, in window-local frames.  Windows are placed by the release's own
        ``windows.jsonl`` (``start_frame``), never re-derived from the slice
        number; a sequence the segmentation does not cover contributes nothing
        and is counted."""
        with open(self.bar_units_path) as handle:
            segmentation = json.load(handle)
        records = segmentation["records"]
        self.bar_beats = float(segmentation.get("config", {}).get("beats_per_segment", 4))
        bounds = {record["sequence"]: [int(b) for b in record["boundaries"]]
                  for record in records}
        out = {}
        self.bar_units_uncovered_windows = 0
        with open(str(Path(data_root) / "windows.jsonl")) as handle:
            for line in handle:
                window = json.loads(line)
                if window.get("split") != "train":
                    continue
                _, recording, clip = str(window["sequence_id"]).split(":")
                edges = bounds.get("{}__{}".format(recording, clip))
                if edges is None:
                    self.bar_units_uncovered_windows += 1
                    continue
                first = int(window["start_frame"])
                last = first + window_frames
                out[int(window["array_index"])] = [
                    (start - first, end - first, (window["sequence_id"], start))
                    for start, end in zip(edges[:-1], edges[1:])
                    if first <= start and end <= last]
        return out

    def _values_at(self, candidate, target_length):
        sample, start, end, _ = candidate
        values = torch.from_numpy(np.array(self.motion[sample, start:end], copy=True))
        floor = self._candidate_floor(candidate)
        if floor is not None:
            # In NORMALIZED units: the release normalizer is per-dimension
            # min-max, so subtracting c metres from the raw root z is
            # subtracting c / scale from the normalized channel.  Only the
            # height moves; x, y, the rotations and the contacts are untouched.
            scale, _offset = self._normalizer_affine()
            column = ROOT_POSITION_START + 2
            values[:, column] -= floor / float(scale[column])
        if len(values) != target_length:
            values = F.interpolate(
                values.T.unsqueeze(0), size=target_length, mode="linear",
                align_corners=True).squeeze(0).T
        return values

    def _values_with_lead(self, candidate, target_length, lead_frames):
        """--draft-seam-lead: ``candidate`` as ``_values_at`` plays it, PRECEDED by up to ``lead_frames`` frames of
        its own recording from before its start -- same playback rate, same floor -- so the approach into its first
        frame (its dancer's downbeat) is that dancer's own.  Returns (values, lead taken); (None, 0) when the window
        holds fewer than two frames before the unit."""
        sample, start, end, _ = candidate
        native, length = int(end) - int(start), int(target_length)
        if lead_frames <= 0 or native <= 1 or length <= 1:
            return None, 0
        # the spacing _values_at's align_corners resample uses, so the unit part is the same frames exactly
        step = (native - 1) / float(length - 1) if native != length else 1.0
        # A unit at the head of its 150-frame window (22% of A0's picks start in its first 8 frames) is the same
        # frames later in the window before it -- the release cuts each recording at stride 15 and the overlapping
        # frames are identical -- so read it from there rather than lose the lead.
        needed = int(np.ceil(lead_frames * step - 1e-9)) + 1
        while int(start) < needed:
            before = self._window_before(int(sample))
            if before is None or int(end) + before[1] > int(np.asarray(self.motion).shape[1]):
                break
            sample, start, end = before[0], int(start) + before[1], int(end) + before[1]
        lead = min(int(lead_frames), int(np.floor(int(start) / step + 1e-9)))
        if lead < 2:
            return None, 0
        low = int(np.floor(int(start) - lead * step + 1e-9))
        raw = torch.from_numpy(np.array(self.motion[sample, low:int(end)], copy=True))
        floor = self._candidate_floor(candidate)
        if floor is not None:
            scale, _offset = self._normalizer_affine()
            column = ROOT_POSITION_START + 2
            raw[:, column] -= floor / float(scale[column])
        position = torch.arange(-lead, length, dtype=torch.float64) * step + (int(start) - low)
        left = position.floor().long().clamp(0, len(raw) - 1)
        right = (left + 1).clamp(max=len(raw) - 1)
        weight = (position - left.double()).clamp(0.0, 1.0).to(raw.dtype).unsqueeze(1)
        values = raw[left] * (1 - weight) + raw[right] * weight
        return values, lead

    def _window_before(self, sample):
        """(train window, frames earlier) that starts just before ``sample`` in the same recording, from the
        release's windows.jsonl; None at a recording's first window or when the table is absent."""
        table = getattr(self, "_window_before_table", None)
        if table is None:
            table = {}
            path = Path(getattr(self, "normalizer_path", "") or ".").parent / "windows.jsonl"
            if path.is_file():
                by_sequence = {}
                for line in path.read_text().splitlines():
                    row = json.loads(line)
                    if row.get("split") == "train":
                        by_sequence.setdefault(row["sequence_id"], []).append(
                            (int(row["start_frame"]), int(row["array_index"])))
                for rows in by_sequence.values():
                    rows.sort()
                    for (first, earlier), (second, later) in zip(rows[:-1], rows[1:]):
                        table[later] = (earlier, second - first)
            self._window_before_table = table
        return table.get(int(sample))

    def _settle_period(self, candidate):
        """This prototype's own beat period, in frames -- cached per candidate.

        The measure is the paper's own motion beat: local minima of the
        segment-wise joint velocity (``tools/motion_beats``), read here on the
        normalised 151-D window's rotation dimensions rather than after forward
        kinematics, because it is wanted for every candidate of every class and
        the ordering is what matters, not the metre value.

        **Why any of this exists.**  Retrieval ranks by duration alone and never
        looks at the query's music (``retrieve`` below, and the paper's M5 is the
        same).  Measured 2026-08-29 over 400 held-out clips: the ground truth is
        locked to the music's *period* -- R = 0.1262 against a gap-shuffled null
        of 0.1132, 241/400 clips, P = 4.8e-05 -- and a draft built from the
        clip's **own true label track** has already lost that lock (P = 0.012, in
        the wrong direction).  So the lock is gone before the model runs, and it
        is gone for a mechanical reason: the ground-truth *spans* carry the
        query's tempo, while the *contents* pasted into them come from another
        recording at another tempo.
        """
        cached = self._period_cache.get(candidate)
        if cached is not None:
            return cached
        sample, start, end = candidate[0], candidate[1], candidate[2]
        window = np.asarray(self.motion[sample, start:end, ROOT_POSITION_START + 3:],
                            dtype=np.float32)
        period = float("nan")
        if len(window) >= 6:
            speed = np.abs(np.diff(window, axis=0)).mean(axis=1)
            if len(speed) >= 5:
                minima = [i for i in range(1, len(speed) - 1)
                          if speed[i] <= speed[i - 1] and speed[i] <= speed[i + 1]]
                if len(minima) >= 2:
                    period = float(np.median(np.diff(minima)))
        self._period_cache[candidate] = period
        return period

    BEAT_CHANNEL = 34
    BEAT_PAD = 45

    def _candidate_beat(self, candidate):
        """This prototype's OWN (start phase, beat period), cached.

        ``start phase`` is where the segment's first frame sits inside its own
        beat interval, in [0, 1); ``period`` is the median inter-beat interval
        of the recording around it.  Both come from the training music array,
        which is frame-aligned with the training motion -- the prototype is not
        an anonymous span of poses, it is a span of poses that happened at a
        known place in a real bar, and that is the "context" the duration rule
        throws away.

        Measured cost of throwing it away (361 plan segments, 40 clips): the
        duration rule's picks land at a median phase error of 0.250 against the
        query's grid, which is EXACTLY the uniform-random expectation, and 68.1%
        of them are at a tempo more than 10% from the query's.
        """
        cached = self._beat_cache.get(candidate)
        if cached is not None:
            return cached
        result = (float("nan"), float("nan"))
        if self.music is not None:
            sample, start, end, _ = candidate
            low = max(0, start - self.BEAT_PAD)
            high = min(self.music.shape[1], end + self.BEAT_PAD)
            column = np.asarray(self.music[sample, low:high, self.BEAT_CHANNEL])
            grid = np.flatnonzero(column > 0.5) + low
            if len(grid) >= 3:
                index = int(np.searchsorted(grid, start, side="right")) - 1
                period = float(np.median(np.diff(grid)))
                if 0 <= index < len(grid) - 1 and grid[index + 1] > grid[index]:
                    span = float(grid[index + 1] - grid[index])
                    result = (float(start - grid[index]) / span, period)
                else:
                    result = (float("nan"), period)
        self._beat_cache[candidate] = result
        return result

    def _segment_energy(self, candidate):
        """This prototype's own amplitude, cached per candidate.

        Same space and same reason as ``_settle_period`` above: the mean
        absolute first difference of the NORMALIZED rotation dimensions, not
        metres after forward kinematics.  It is wanted for every candidate of
        every class and only the ORDERING within a class is used, so the metre
        value would buy nothing and cost an FK per candidate.
        """
        cached = self._energy_cache.get(candidate)
        if cached is not None:
            return cached
        sample, start, end = candidate[0], candidate[1], candidate[2]
        window = np.asarray(self.motion[sample, start:end, ROOT_POSITION_START + 3:],
                            dtype=np.float32)
        value = (float(np.abs(np.diff(window, axis=0)).mean())
                 if len(window) >= 2 else 0.0)
        self._energy_cache[candidate] = value
        return value

    def _energy_floor(self, label, candidates, quantile):
        """Drop a class's least-energetic tail before the duration rule runs.

        WHY, and the measurement that forced it.  ``retrieve`` ranks by duration
        alone and never looks at the motion -- its own docstring says so.  On the
        90 shipped clips that is 1,277 prototype pastes, of which **31 (2.4%)
        are under 0.20 m/s**, i.e. a body that barely moves, covering 1,214 of
        45,178 output frames and touching 25 of 90 clips; 12 clips are over 10%
        frozen and the worst is 37.1%.  Named damage:
        ``wild_v5:7424419958844706087:clip000``, 34.3% of its prototype frames
        frozen, generated energy 0.200 against a ground truth of 0.699 -- the
        worst ratio of the 90.

        A WITHIN-CLASS QUANTILE, not an absolute floor, and that is the whole
        design.  27 of those 31 came from classes whose own pool is already
        low-energy -- M3 classes tagged "standing pose sustained", "arms crossed
        pose".  An absolute floor would empty those classes and force a fallback
        that silently changes which movement is danced; a quantile drops each
        class's own worst tail and still returns that class.  A pose class stays
        a pose class, it just stops returning its most frozen exemplar.

        WHAT THIS CANNOT DO.  It cannot make a clip's amplitude track its song.
        That was the original design and it was refused on measurement: over the
        90 evaluation clips, no combination of the 35 music channels predicts a
        clip's ground-truth energy -- 6-fold cross-validated R of 0.038 to 0.125
        against a permutation null whose 95th percentile is 0.225 (the in-sample
        R of 0.516 was 8 parameters on 90 samples).  There is no target to aim
        at, so this aims at the measured defect instead and claims nothing more.
        """
        if not quantile or len(candidates) < 4:
            return candidates
        energies = [self._segment_energy(c) for c in candidates]
        cut = float(np.quantile(energies, float(quantile)))
        kept = tuple(c for c, e in zip(candidates, energies) if e > cut)
        return kept or candidates

    def _duration_pick(self, candidates, target_length, excluded, beats=(),
                       beat_period=None, quiet=None, soft=None):
        """``min |duration difference|`` -- and what to do about its TIES.

        ``min`` returns the first item at the minimum, and the minimum here is
        reached by a crowd rather than by one candidate.  Measured over the 20
        held-out clips (91 retrieved segments, pool median 690 per class):
        **73.6% of segments have more than one candidate at the minimum**, the
        median tie holds **6** candidates and the largest holds 38, and 79.1% of
        segments have a candidate whose length matches EXACTLY.  Because the
        index order is fixed, all 79 distinct ``(label, target_length)`` keys
        returned the same prototype every time they occurred -- across clips as
        well as within one.  So a median of six equally-good exemplars was
        available and the same one was taken every time, in every song, which is
        the "every clip dances the same" the operator reports (across-clip
        novelty 0.373 against ground truth's 0.458, 17/20 clips).

        ``tie_break="salted"`` picks among the tied candidates using a hash of
        the QUERY's own retrieval group, so:

          * nothing about the rule's own criterion is given up -- every
            candidate drawn from is *exactly* as good on duration, and the
            resampling factor is therefore unchanged.  This is what separates
            it from the paper's ``Random Choice`` ablation, which draws from the
            whole class and loses on FID because unconstrained draws need
            heavier time-stretching;
          * two different recordings get different exemplars, which is the axis
            that is broken;
          * one recording is unchanged from occurrence to occurrence, so the
            within-clip behaviour -- which measures FINE already (novelty 0.60
            against ground truth's 0.62) -- is not disturbed by this switch;
          * it is order independent and cacheable: the salt is part of the
            existing cache key, so nothing depends on how many segments were
            retrieved before this one.
        """
        distances = [abs((item[2] - item[1]) - target_length) for item in candidates]
        best = min(distances)
        tied = [item for item, distance in zip(candidates, distances)
                if distance == best]
        tied = self._prefer_beat_span(tied, target_length, beat_period)
        tied = self._prefer_hop_guard(tied)
        tied = self._prefer_music_energy(tied, soft)
        tied = self._prefer_hold_by_music(tied, quiet)
        tied = self._prefer_feet_beat_lead(tied, target_length, beats)
        tied = self._prefer_beat_fit(tied, target_length, beats)
        tied = self._prefer_feet_lead(tied)
        if self.tie_break == "index" or len(tied) == 1:
            return tied[0]
        salt = "|".join(sorted(excluded))
        digest = hashlib.sha256(
            "{}|{}|{}".format(salt, int(target_length), len(tied)).encode()
        ).digest()
        return tied[int.from_bytes(digest[:8], "big") % len(tied)]

    # Joints of the two parts the corpus and the dancer disagree about, in the
    # SAME grouping tools/score_beat_phase_profile.py judges with, so the
    # property selected for and the property measured are one property.
    FEET_JOINTS = (7, 8, 10, 11)
    HIPS_KNEES_JOINTS = (1, 2, 4, 5)

    def _rest_depth(self, candidate, joints):
        """How deeply these joints come to rest inside this prototype.

        ``(mean - 5th percentile) / mean`` of the part's own rot6d speed: 1.0 is
        a part that fully stops somewhere in the segment, 0.0 one that never
        slows.  Read on the normalised window rather than after forward
        kinematics because it is wanted for every candidate of every class and
        only the ORDER between two parts of the same segment is used.
        """
        key = (candidate[0], candidate[1], candidate[2], joints)
        cached = self._rest_cache.get(key)
        if cached is not None:
            return cached
        columns = [ROOT_POSITION_START + ROOT_POSITION_DIMS + 6 * joint + k
                   for joint in joints for k in range(6)]
        window = np.asarray(self.motion[candidate[0], candidate[1]:candidate[2], columns],
                            dtype=np.float32)
        value = 0.0
        if len(window) >= 4:
            speed = np.abs(np.diff(window, axis=0)).mean(axis=1)
            mean = float(speed.mean())
            if mean > 1e-9:
                value = float((mean - np.percentile(speed, 5)) / mean)
        self._rest_cache[key] = value
        return value

    # The two parts whose ordering the judgement column and ground truth agree
    # on most strongly: ground truth settles the feet at 0.0943 and the
    # shoulders at 0.0207, while every generated arm inverts it.
    BEAT_FIT_LEAD = (7, 8, 10, 11)          # feet: ankles + toes
    BEAT_FIT_TRAIL = (13, 14, 16, 17)       # shoulders: collars + shoulders

    def _beat_fit_margin(self, candidate, target_length, beats):
        """How much MORE this candidate's feet decelerate on the slot's beats
        than its shoulders do, once it is stretched into the slot.

        SAME LANGUAGE AS THE JUDGEMENT COLUMN, and that is the whole point.  Two
        cheaper proxies were tried first and both failed validation, each in the
        way CLAUDE.md section 2.1 rule 1 warns about -- a criterion invented and
        then used to judge without being checked against the target property:

        * per-part "rest depth" on the NORMALISED rot6d channels.  Against the
          same property measured on world joints it reads Spearman rho +0.029
          and agrees on the sign in 30.2% of blocks.  It is not the same
          quantity, and a filter built on it selected something unrelated to
          what it claimed.
        * the same rest depth on world joints.  Correctly specified, but with no
          material: 93.2% of library windows already have feet resting deeper
          than hips and knees, against ground truth's own 93.4%.  A filter where
          93% of candidates qualify selects nothing.

        This one is measured where the judgement is made -- world joints, at the
        slot's own beat phases -- and it does have material: over sampled
        library excerpts the feet-minus-shoulders margin is neutral (positive on
        52.5%, sd 0.506), and taking the top quartile moves the mean from -0.037
        to +0.560.  With the measured tie sizes a top-quartile candidate is
        available in 82.2% of slots.

        Positive means the feet brake on the beat and the shoulders do not,
        which is the dancer's asymmetry.
        """
        key = (candidate[0], candidate[1], candidate[2], int(target_length),
               tuple(beats))
        cached = self._beat_fit_cache.get(key)
        if cached is not None:
            return cached
        values = self._values_at(candidate, target_length)
        joints = decode_motion(values, self.normalizer_path)["full_pose"]
        frames = np.asarray(joints, dtype=np.float64)
        inside = [b for b in beats if 0 <= b < len(frames) - 1]
        margin = 0.0
        if len(inside) >= 2:
            scores = []
            for group in (self.BEAT_FIT_LEAD, self.BEAT_FIT_TRAIL):
                speed = np.linalg.norm(
                    np.diff(frames[:, list(group), :], axis=0), axis=-1).mean(1)
                spread = speed.std()
                if spread < 1e-9:
                    scores.append(0.0)
                    continue
                z = (speed - speed.mean()) / spread
                scores.append(float(-np.mean([z[b] for b in inside])))
            margin = scores[0] - scores[1]
        self._beat_fit_cache[key] = margin
        return margin

    # Every part the judgement column reads, so "the feet lead" is measured
    # against the same body it is judged against.
    BEAT_PARTS = {"feet": (7, 8, 10, 11), "hips_knees": (1, 2, 4, 5),
                  "torso": (3, 6, 9, 12), "shoulders": (13, 14, 16, 17),
                  "elbows": (18, 19), "hands": (20, 21, 22, 23)}

    def _feet_lead_margin(self, candidate, target_length, beats):
        """How far the FEET's deceleration on this slot's beats stands above the
        rest of the body's.

        ONLY THE STABLE HALF OF THE TARGET IS USED.  The per-part profile this
        project has been aiming at -- feet > hips > torso > hands > shoulders --
        was measured on 20 eval clips.  Recomputed with the judge's own
        ``aligned_profile`` over 486 train and 481 test release windows it reads
        feet > elbows > hips > hands > shoulders > torso, with train and test
        agreeing at Spearman +0.94: the CORPUS is consistent and the ORDER below
        the feet is not, moving with the measurement (elbows 4th -> 2nd, torso
        3rd -> last).  Elbows is also the one part that fails the judge's own
        half-beat control.  So the only part of the target that survives is
        **the feet lead, and lead by a lot** (0.09-0.10 against <=0.05 for
        everything else), and that is all this contrast encodes.

        NOT the same as ``_beat_fit_margin``: that one is feet MINUS SHOULDERS,
        which selected for braking hard and bought jitter 0.0531 -> 0.0996.
        This is feet minus the mean of the WHOLE body, so a candidate that
        brakes everything equally scores zero and cannot win by braking harder.
        """
        key = (candidate[0], candidate[1], candidate[2], int(target_length),
               tuple(beats), "feetlead")
        cached = self._beat_fit_cache.get(key)
        if cached is not None:
            return cached
        values = self._values_at(candidate, target_length)
        frames = np.asarray(decode_motion(values, self.normalizer_path)["full_pose"],
                            dtype=np.float64)
        inside = [b for b in beats if 0 <= b < len(frames) - 1]
        margin = 0.0
        if len(inside) >= 2:
            scores = {}
            for name, group in self.BEAT_PARTS.items():
                speed = np.linalg.norm(
                    np.diff(frames[:, list(group), :], axis=0), axis=-1).mean(1)
                spread = speed.std()
                if spread < 1e-9:
                    scores[name] = 0.0
                    continue
                z = (speed - speed.mean()) / spread
                scores[name] = float(-np.mean([z[b] for b in inside]))
            margin = scores["feet"] - float(np.mean(list(scores.values())))
        self._beat_fit_cache[key] = margin
        return margin

    def _prefer_feet_beat_lead(self, tied, target_length, beats):
        """Keep the quarter of the tie whose feet lead the beat hardest.

        A quartile and a filter, not an argmax and not a re-ranking, for the
        reason ``--retrieval-rule phase`` recorded: ordering the whole pool by a
        new criterion landed the criterion perfectly and made the dance worse by
        squeezing out variety.  Survivors go back to the existing tie-break.

        MATERIAL, measured before this was built: over 400 decoded library
        excerpts the part-ordering against ground truth's profile is dead
        neutral (mean rho +0.002, sd 0.557) while the top quartile reaches
        +0.695, and with the measured tie sizes a top-quartile candidate is
        available in 82.2% of slots.  The pool is not short of the property;
        the duration rule simply never looks at it.
        """
        # TWO CANDIDATES IS ENOUGH TO CHOOSE, and half rather than a quarter.
        # The first version required four tied candidates and kept the top
        # quarter; measured, it changed the dance on only 9 of 20 clips and the
        # part-ordering judge read a median difference of exactly 0.000 with a
        # sign test over 5 moved clips -- the effect could not be established
        # because the filter mostly did not fire, not because it did nothing
        # (its paired mean was +0.119, the first positive of the session).
        # Halving still leaves the existing tie-break something to draw from,
        # which is what keeps this a filter rather than the argmax that sank
        # --retrieval-rule phase.
        self.feet_lead_slots += 1
        if not self.feet_beat_lead or len(tied) < 2 or not beats:
            self.feet_lead_skipped += 1
            return tied
        scored = [(self._feet_lead_margin(c, target_length, beats), i, c)
                  for i, c in enumerate(tied)]
        scored.sort(key=lambda row: (-row[0], row[1]))
        kept = [row[2] for row in scored[:max(1, len(scored) // 2)]]
        # COUNTED, because "the flag was on" and "the flag changed the pick" are
        # different claims.  With tie_break="index" the pick is kept[0], so the
        # filter only matters when the best-margin candidate is not already the
        # first by index -- and the operator, watching, asked why so many clips
        # came out identical to the baseline.
        self.feet_lead_applied += 1
        if kept[0] is not tied[0]:
            self.feet_lead_changed += 1
        return kept

    def _prefer_beat_fit(self, tied, target_length, beats):
        """Keep the top quartile of the tie by ``_beat_fit_margin``.

        A quartile and not the argmax, and a filter and not a re-ranking, for
        the reason ``--retrieval-rule phase`` recorded: ordering the whole pool
        by a new criterion landed that criterion perfectly (chosen-prototype
        phase error 0.2381 -> 0.0000) and made the output worse, because a hard
        ordering narrows the pool and squeezes out variety.  The survivors go
        back to the existing tie-break, so the variety machinery is untouched.
        """
        if not self.beat_fit or len(tied) < 4 or not beats:
            return tied
        scored = [(self._beat_fit_margin(c, target_length, beats), i, c)
                  for i, c in enumerate(tied)]
        keep = max(1, len(scored) // 4)
        scored.sort(key=lambda row: (-row[0], row[1]))
        return [row[2] for row in scored[:keep]]

    def _full_frame_features(self):
        """Per window, per frame: the higher wrist above its shoulder, the straighter elbow and the
        longer shoulder->wrist reach, all in arm lengths / degrees (scale-free), from the release's
        own decode -- computed once for the whole library (~5 s on CPU)."""
        if self._full_frames is None:
            motion = np.asarray(self.motion)
            count, frames = motion.shape[0], motion.shape[1]
            stops_mode = self.prefer_full_mode in ("stops", "both", "frames_slow")
            out = np.zeros((count, frames, 2, 4) if stops_mode else (count, frames, 3), dtype=np.float32)
            normalizer = self.normalizer_path
            for first in range(0, count, 64):
                block = torch.from_numpy(np.array(motion[first:first + 64], dtype=np.float32))
                joints = decode_motion(block.reshape(-1, block.shape[-1]), normalizer)["full_pose"]
                joints = np.asarray(joints, dtype=np.float64).reshape(len(block), frames, 24, 3)
                out[first:first + len(block)] = (_arm_stop_frames(joints) if stops_mode
                                                 else _arm_fullness_frames(joints))
            flat = out[..., :3].reshape(-1, 3)
            self._full_frames = (out, flat.mean(0), flat.std(0) + 1e-6)
            if stops_mode:
                speed = out[..., 3].reshape(-1)
                self._full_speed_stats = (float(speed.mean()), float(speed.std()) + 1e-6)
        return self._full_frames

    def _fullness(self, candidate):
        """How FULL a candidate's played range is: mean z of the 90th percentiles of arm raise,
        elbow straightness and reach.  Measured 2026-09-23 (DEFECTS 92): inside a slot's filtered
        pool this spreads 1.9 z from p10 to p90, and the unit the shipped chain picks sits at the
        41st percentile -- the join band and the selector both lean to the smaller moves."""
        key = tuple(int(v) for v in candidate[:3])
        cached = self._full_cache.get(key)
        if cached is None:
            frames, mean, std = self._full_frame_features()
            sample, start, end = key
            played = frames[sample, start:max(end, start + 1)]
            if self.prefer_full_mode == "frames_slow":
                # REACH HIGH WHILE MOVING SLOWLY: the 'frames' score, minus the z of the played
                # range's mean wrist speed.  'frames' lifted the peaks and bought them with speed
                # (energy 1.20/1.22x ground truth); this prefers units that get there and stay.
                peaks = played[:, :, :3].max(axis=1)
                speed_mean, speed_std = self._full_speed_stats
                cached = float(((np.percentile(peaks, 90, axis=0) - mean) / std).mean()) \
                    - (float(played[:, :, 3].mean()) - speed_mean) / speed_std
            elif self.prefer_full_mode in ("stops", "both"):
                # AT THE ARRIVALS: a stop is a local minimum of that wrist's body-frame speed with at
                # least 5 cm of travel over the three frames into it -- the frame a move lands.
                # Rewarding the 90th percentile over all frames also rewards sweeping THROUGH an
                # extended pose, and that version bought its fullness with speed (energy 1.20/1.22x
                # ground truth against a [0.85, 1.15] band).
                landed = []
                for arm in range(2):
                    speed = played[:, arm, 3]
                    for t in range(3, len(speed) - 1):
                        # strictly slower than the frame before: on a plateau of equal speeds
                        # every frame would otherwise count as an arrival
                        if speed[t] < speed[t - 1] and speed[t] <= speed[t + 1] \
                                and float(speed[t - 2:t + 1].sum()) >= 0.05:
                            landed.append(played[t, arm, :3])
                peaks = played[:, :, :3].max(axis=1)
                sample_rows = np.array(landed) if landed else peaks
                cached = float(((np.percentile(sample_rows, 50 if landed else 90, axis=0) - mean)
                                / std).mean())
                if self.prefer_full_mode == "both":
                    # Half landing extended, half reaching high: 'stops' alone kept elbows straight
                    # and dropped the overhead peaks (clip 818 raise90 0.498 -> 0.325), 'frames'
                    # alone raised them and cost energy.
                    cached = 0.5 * cached + 0.5 * float(
                        ((np.percentile(peaks, 90, axis=0) - mean) / std).mean())
            else:
                cached = float(((np.percentile(played, 90, axis=0) - mean) / std).mean())
            self._full_cache[key] = cached
        return cached

    def _prefer_full(self, tied):
        """--draft-prefer-full: keep the fullest FRACTION of the candidates (at least two), before the
        join band and the selector choose among them.  Narrowing like every filter here.  Counted."""
        if not self.prefer_full or len(tied) < 3:
            return tied
        self.prefer_full_slots = getattr(self, "prefer_full_slots", 0) + 1
        keep = max(2, int(math.ceil(len(tied) * self.prefer_full)))
        ranked = sorted(tied, key=self._fullness, reverse=True)[:keep]
        if len(ranked) < len(tied):
            self.prefer_full_applied = getattr(self, "prefer_full_applied", 0) + 1
        return ranked

    def _rhythm_model(self):
        """(scorer, device, library kinetics [N, T, KIN]) -- loaded and decoded once."""
        if self._rhythm is None:
            from model.rhythm_scorer import kinetic_frames, load_scorer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, blob = load_scorer(self.rhythm_scorer_path, device)
            plane = blob.get("kinetics") == "plane"
            motion = np.asarray(self.motion)
            kin = None
            for first in range(0, motion.shape[0], 64):
                block = torch.from_numpy(np.array(motion[first:first + 64], dtype=np.float32))
                joints = decode_motion(block.reshape(-1, block.shape[-1]), self.normalizer_path)["full_pose"]
                part = kinetic_frames(np.asarray(joints).reshape(len(block), motion.shape[1], 24, 3), plane=plane)
                if kin is None:
                    kin = np.zeros((motion.shape[0],) + part.shape[1:], dtype=np.float32)
                kin[first:first + len(block)] = part
            self._rhythm = (model, device, kin)
        return self._rhythm

    def _rhythm_scores(self, candidates, slot_music):
        """Learned alignment of each candidate, AS IT WILL PLAY (its native range stretched over the slot),
        with the query's own music under the slot.  Reads the query's MUSIC only (CLAUDE.md 1.6)."""
        from model.rhythm_scorer import CROP, normalise_kinetics, resample
        model, device, kin = self._rhythm_model()
        music = resample(torch.as_tensor(np.asarray(slot_music, dtype=np.float32))[None], CROP)
        crops = [resample(torch.from_numpy(np.array(kin[c[0], c[1]:max(c[2], c[1] + 2)]))[None], CROP)[0]
                 for c in candidates]
        crops = normalise_kinetics(torch.stack(crops))
        with torch.no_grad():
            out = model(music.expand(len(crops), -1, -1).to(device), crops.to(device))
        return out.float().cpu().numpy()

    def _rhythm_phase(self, candidate, slot_music):
        """--draft-rhythm-shift N: slide the chosen unit's source range by -N..+N frames (same length, inside
        its window) and keep the offset the learned scorer finds best aligned with THIS slot's music --
        timing chosen by alignment rather than forced by a stop at the bar line.  Returns (candidate, shift)."""
        if not self.rhythm_shift or slot_music is None or len(slot_music) < 4:
            return candidate, 0
        sample, start, end, group = candidate
        frames = int(np.asarray(self.labels).shape[1])
        shifts = [d for d in range(-self.rhythm_shift, self.rhythm_shift + 1)
                  if start + d >= 0 and end + d <= frames]
        if len(shifts) < 2:
            return candidate, 0
        options = [(sample, start + d, end + d, group) for d in shifts]
        scores = self._rhythm_scores(options, slot_music)
        best = int(np.argmax(scores))
        self.rhythm_shift_slots = getattr(self, "rhythm_shift_slots", 0) + 1
        if shifts[best] != 0:
            self.rhythm_shift_moved = getattr(self, "rhythm_shift_moved", 0) + 1
        return options[best], shifts[best]

    def _continuation_rhythm_ok(self, candidate, label, target_length, slot_music, excluded):
        """True if ``candidate`` (a continued bar) scores at or above the ``continue_rhythm_min`` quantile of the
        slot's duration-band pool (same label, exclusion and snap filters as retrieval), for this slot's music."""
        if not self.continue_rhythm_min or slot_music is None or not self.rhythm_scorer_path:
            return True
        pool = [c for c in self.index.get(int(label), ())
                if isinstance(c[3], str) and c[3] and c[3] not in excluded]
        pool = self._duration_band(pool, target_length) if pool else []
        pool = [c for c in pool if self._passes_snap_filters(c)]
        if len(pool) < 4:
            return True
        if len(pool) > 256:
            pool = [pool[i] for i in np.linspace(0, len(pool) - 1, 256).astype(int)]
        scores = self._rhythm_scores(list(pool) + [candidate], slot_music)
        rank = float((scores[:-1] < scores[-1]).mean())
        self.continue_rhythm_checked = getattr(self, "continue_rhythm_checked", 0) + 1
        ok = rank >= self.continue_rhythm_min
        if not ok:
            self.continue_rhythm_refused = getattr(self, "continue_rhythm_refused", 0) + 1
        return ok

    def _prefer_rhythm(self, tied, slot_music, continuity_context=None, target_length=None):
        """--draft-rhythm-keep: keep the best-aligned fraction (at least two) of the candidates, before the
        join band and the selector choose among them.  Narrowing like every filter here.  Counted.

        With --draft-continuity-weight W the ranking is z(rhythm) + W * z(continuity), both standardised within
        this slot's pool, so neither scorer's raw scale decides the mix; the kept fraction is still the rhythm
        keep.  A slot with no placed context (the clip's first) falls back to rhythm alone, counted."""
        if not self.rhythm_keep or slot_music is None or len(tied) < 3 or len(slot_music) < 4:
            return tied
        self.rhythm_slots = getattr(self, "rhythm_slots", 0) + 1
        scores = self._rhythm_scores(tied, slot_music)
        weight = float(getattr(self, "continuity_weight", 0.0) or 0.0)
        if weight:
            if continuity_context is None or target_length is None:
                self.continuity_no_context = getattr(self, "continuity_no_context", 0) + 1
            else:
                self.continuity_joint_slots = getattr(self, "continuity_joint_slots", 0) + 1
                cont = self._continuity_scores(tied, continuity_context, target_length)

                def z(v):
                    v = np.asarray(v, dtype=np.float64)
                    sd = float(v.std())
                    return (v - v.mean()) / sd if sd > 1e-12 else np.zeros_like(v)

                joint = z(scores) + weight * z(cont)
                keep = max(2, int(math.ceil(len(tied) * self.rhythm_keep)))
                order = np.argsort(-joint, kind="stable")[:keep]
                alone = set(np.argsort(-np.asarray(scores), kind="stable")[:keep].tolist())
                if set(order.tolist()) != alone:
                    self.continuity_joint_changed = getattr(self, "continuity_joint_changed", 0) + 1
                if keep < len(tied):
                    self.rhythm_applied = getattr(self, "rhythm_applied", 0) + 1
                return [tied[i] for i in sorted(order)]
        keep = max(2, int(math.ceil(len(tied) * self.rhythm_keep)))
        order = np.argsort(-scores, kind="stable")[:keep]
        if keep < len(tied):
            self.rhythm_applied = getattr(self, "rhythm_applied", 0) + 1
        return [tied[i] for i in sorted(order)]

    def _prefer_tempo(self, tied, target_length):
        """--draft-tempo-keep F (H series): keep the fraction F (at least two) of the candidates whose OWN length is
        closest to the slot's, i.e. that are played nearest their dancer's own speed.  The duration band allows +-15%,
        and the median stretch of the shipped picks is 0.92 (moves run ~8% fast).  Under phrase continuation the
        choice at a phrase start carries through the phrase: the continued bars are the same dancer at the same tempo.
        Library lengths and the slot only."""
        keep_frac = float(getattr(self, "tempo_keep", 0.0) or 0.0)
        if not keep_frac or len(tied) < 3:
            return tied
        self.tempo_slots = getattr(self, "tempo_slots", 0) + 1
        dev = np.array([abs(math.log(max(1, c[2] - c[1]) / float(target_length))) for c in tied])
        keep = max(2, int(math.ceil(len(tied) * keep_frac)))
        order = np.argsort(dev, kind="stable")[:keep]
        if keep < len(tied):
            self.tempo_applied = getattr(self, "tempo_applied", 0) + 1
        return [tied[i] for i in sorted(order)]

    def _library_kinematics(self):
        """Decode the whole library once: every window's foot landings (``_foot_plant_frames``) and its per-frame body
        speed (``_body_speed_frames``, m/s) -- what --draft-step-lock-keep and --draft-energy-follow read."""
        if self._foot_plants is None:
            motion = np.asarray(self.motion)
            count, frames = motion.shape[0], motion.shape[1]
            plants, speeds, points, speeds2d = [], [], [], []
            for first in range(0, count, 64):
                block = torch.from_numpy(np.array(motion[first:first + 64], dtype=np.float32))
                joints = decode_motion(block.reshape(-1, block.shape[-1]), self.normalizer_path)["full_pose"]
                joints = np.asarray(joints, dtype=np.float64).reshape(len(block), frames, 24, 3)
                plants.extend(_foot_plant_frames(w) for w in joints)
                for w in joints:
                    v = _body_speed_frames(w)
                    speeds.append(v)
                    points.append(_action_point_frames(v))
                    speeds2d.append(_image_speed_frames(w))
            self._foot_plants, self._body_speed, self._action_points = plants, speeds, points
            self._speed2d = speeds2d

    def _unit_speed(self, candidate, target_length):
        """Mean body speed (m/s) of a library unit AS PLAYED in a slot of ``target_length`` frames (the linear
        resample plays it (native-1)/(slot-1) times as fast)."""
        self._library_kinematics()
        sample, start, end = (int(v) for v in candidate[:3])
        native = max(end - start, 1)
        return float(self._body_speed[sample][start:max(end, start + 1)].mean()) \
            * max(native - 1, 1) / max(float(target_length) - 1.0, 1.0)

    def _unit_points(self, candidate):
        """Action points (moments a move lands, ``_action_point_frames``) inside a library unit -- a count per unit;
        units are one source bar, so this is points per bar."""
        self._library_kinematics()
        sample, start, end = (int(v) for v in candidate[:3])
        p = self._action_points[sample]
        return float(((p >= start) & (p < end)).sum())

    def _points_cdf(self):
        if getattr(self, "_points_cdf_cache", None) is None:
            self._library_kinematics()
            vals = []
            if self._source_bars:
                for entries in self._source_bars.values():
                    index, start, end, _ = entries[0]
                    p = self._action_points[index]
                    vals.append(float(((p >= start) & (p < end)).sum()))
            self._points_cdf_cache = np.sort(np.asarray(vals or [0.0], dtype=np.float64)) + \
                np.linspace(0, 1e-3, len(vals or [0.0]))          # break ties so a count maps to the middle of its run
        return self._points_cdf_cache

    def _energy_cdf(self):
        """Sorted body speeds of the library's source bars at their own tempo: the scale a bar's target percentile
        is read on (source bars from --draft-continue-source; 48-frame spans of every window without it)."""
        if self._speed_cdf is None:
            self._library_kinematics()
            vals = []
            if self._source_bars:
                for entries in self._source_bars.values():
                    index, start, end, _ = entries[0]
                    vals.append(float(self._body_speed[index][start:end].mean()))
            else:
                for v in self._body_speed:
                    vals.extend(float(v[k:k + 48].mean()) for k in range(0, len(v) - 47, 48))
            self._speed_cdf = np.sort(np.asarray(vals, dtype=np.float64))
        return self._speed_cdf

    def _prefer_energy_follow(self, tied, target_length, slot_start, label, lengths):
        """--draft-energy-follow SPEC (K series): keep the fraction ``keep`` of the candidates whose movement speed, as
        played in this slot, sits nearest this bar's TARGET percentile of the library's bar speeds -- and at a phrase
        start, the same for every source bar continuation would play after it, each against its own query bar.
        The targets are the song's own intensity plan (``_energy_targets``): quiet stretches ask for the slower end
        of the library, loud ones for the faster end.  Operator 2026-09-26: "beat 少的段落 motion 动作点密度小一些 ...
        beat 多的地方不要太安静 ... 随着音乐铺垫和高潮交替相应的动作快慢变化".  Measured before it: F11 and J7 whole
        songs follow loudness at Spearman -0.005 / +0.014, loudest third of bars 1.01-1.02x the quietest.
        Query music and library motion only."""
        spec = getattr(self, "energy_follow", None)
        tau = self._energy_tau
        if not spec or tau is None or slot_start is None or len(tied) < 3:
            return tied
        cdf = self._energy_cdf()

        metric = spec.get("metric", "speed")
        pcdf = self._points_cdf() if metric != "speed" else None

        def pct(speed):
            return float(np.searchsorted(cdf, speed)) / len(cdf)

        def ppct(count):
            lo, hi = np.searchsorted(pcdf, count - 0.5), np.searchsorted(pcdf, count + 0.5)
            return float(lo + hi) / 2.0 / len(pcdf)

        def unit_err(unit, length, frame):
            t = target(frame)
            e_speed = abs(pct(self._unit_speed(unit, length)) - t)
            if metric == "speed":
                return e_speed
            e_points = abs(ppct(self._unit_points(unit)) - t)
            return e_points if metric == "points" else 0.5 * (e_speed + e_points)

        def target(frame):
            return float(tau[int(min(max(frame, 0), len(tau) - 1))])
        self.energy_follow_slots = getattr(self, "energy_follow_slots", 0) + 1
        errors = []
        for candidate in tied:
            err = [unit_err(candidate, target_length, int(slot_start) + target_length // 2)]
            if lengths and self._source_bars is not None:
                current, position = candidate, int(slot_start) + int(target_length)
                for length_j in lengths:
                    current = self._next_source_bar(current, label, length_j)
                    if current is None:
                        break
                    err.append(unit_err(current, length_j, position + length_j // 2))
                    position += int(length_j)
            errors.append(float(np.mean(err)))
        keep = max(2, int(math.ceil(len(tied) * float(spec["keep"]))))
        order = np.argsort(np.asarray(errors), kind="stable")[:keep]
        self.energy_follow_err_all = getattr(self, "energy_follow_err_all", 0.0) + float(np.mean(errors))
        self.energy_follow_err_kept = getattr(self, "energy_follow_err_kept", 0.0) + float(np.mean([errors[i] for i in order]))
        if keep < len(tied):
            self.energy_follow_applied = getattr(self, "energy_follow_applied", 0) + 1
        return [tied[i] for i in sorted(order)]

    def _slot_hit(self, candidate, target_length, beats):
        """How much more this unit SETTLES on the slot's beats than half a beat later, as played in the slot and as the
        camera sees it (``_image_speed_frames``): mean over beats of the settle reading at the beat minus half a beat
        on (``_settle_reading``)."""
        self._library_kinematics()
        sample, start, end = (int(v) for v in candidate[:3])
        trace = self._speed2d[sample][start:max(end, start + 2)]
        played = np.interp(np.linspace(0.0, len(trace) - 1.0, int(target_length)), np.arange(len(trace)), trace)
        return _beat_settle_contrast(played, beats)

    def _prefer_hit(self, tied, target_length, beats, label, phrase_music, lengths):
        """``hit=F`` in --draft-energy-follow (K series): keep the fraction F of candidates whose body SETTLES on this
        song's beats more than off them, as played and as the camera sees it; at a phrase start, over the whole phrase
        continuation will play.  Measured 2026-09-27: the energy follow alone (K2, K5) loosened the beat on the
        rendered video (whole-song beat contrast J7 +0.036 -> K2 +0.026, K5 +0.019), and the loss is already in the
        image-plane projection of the 3D (J7 +0.043 -> K5 +0.024) though not in whole-body 3D speed.  Query beats
        and library motion only."""
        spec = getattr(self, "energy_follow", None)
        frac = float(spec.get("hit", 0.0)) if spec else 0.0
        if not frac or len(tied) < 3 or len(beats) < 2:
            return tied
        self.hit_slots = getattr(self, "hit_slots", 0) + 1
        scores = []
        for candidate in tied:
            vals = [self._slot_hit(candidate, target_length, beats)]
            if phrase_music and lengths and self._source_bars is not None:
                current = candidate
                for music_j, length_j in zip(phrase_music, lengths):
                    current = self._next_source_bar(current, label, length_j)
                    if current is None or music_j is None:
                        break
                    beats_j = [int(b) for b in np.flatnonzero(np.asarray(music_j)[:, BEAT_CHANNEL] > 0.5)]
                    if len(beats_j) >= 2:
                        vals.append(self._slot_hit(current, length_j, beats_j))
            scores.append(float(np.mean(vals)))
        keep = max(2, int(math.ceil(len(tied) * frac)))
        order = np.argsort(-np.asarray(scores), kind="stable")[:keep]
        self.hit_score_all = getattr(self, "hit_score_all", 0.0) + float(np.mean(scores))
        self.hit_score_kept = getattr(self, "hit_score_kept", 0.0) + float(np.mean([scores[i] for i in order]))
        return [tied[i] for i in sorted(order)]

    def _plants_of(self, candidate):
        """Foot landings of a library unit, as frame offsets from its own start (the whole library decoded once)."""
        self._library_kinematics()
        sample, start, end = (int(v) for v in candidate[:3])
        p = self._foot_plants[sample]
        return p[(p >= start) & (p < end)] - start

    def _prefer_step_lock(self, tied, target_length, beats, label, phrase_music, lengths):
        """--draft-step-lock-keep F (J series): keep the fraction F (at least two) of the candidates whose FEET land on
        this song's beats when played into the slot -- ``_step_lock_score`` of the unit's own foot landings, stretched
        exactly as the draft will stretch them, on the slot's beats; at a phrase start, plus the same for every source
        bar continuation would play after it (``_next_source_bar``) on the query bar it would land on.
        Measured 2026-09-26 (footwork_eval / pooled phase-lock R of foot landings on the clip's own beats): ground
        truth 0.152 (another song's grid 0.014), F11 0.096; the units F11 picks read 0.129 in their own songs and the
        library 0.083 -- the operator's "步伐卡节奏偏少".  Query music (beats) and library motion only."""
        keep_frac = float(getattr(self, "step_lock_keep", 0.0) or 0.0)
        if not keep_frac or len(tied) < 3 or len(beats) < 2:
            return tied
        self.step_lock_slots = getattr(self, "step_lock_slots", 0) + 1
        scores = []
        for candidate in tied:
            native = int(candidate[2]) - int(candidate[1])
            score = _step_lock_score(self._plants_of(candidate), native, target_length, beats)
            if phrase_music and lengths and self._source_bars is not None:
                current = candidate
                for music_j, length_j in zip(phrase_music, lengths):
                    current = self._next_source_bar(current, label, length_j)
                    if current is None or music_j is None:
                        break
                    beats_j = [int(b) for b in np.flatnonzero(np.asarray(music_j)[:, BEAT_CHANNEL] > 0.5)]
                    score += _step_lock_score(self._plants_of(current), int(current[2]) - int(current[1]),
                                              length_j, beats_j)
            scores.append(score)
        self.step_lock_seen = getattr(self, "step_lock_seen", 0) + len(tied)
        self.step_lock_positive = getattr(self, "step_lock_positive", 0) + sum(v > 0 for v in scores)
        keep = max(2, int(math.ceil(len(tied) * keep_frac)))
        order = np.argsort(-np.asarray(scores), kind="stable")[:keep]
        if keep < len(tied):
            self.step_lock_applied = getattr(self, "step_lock_applied", 0) + 1
        return [tied[i] for i in sorted(order)]

    def _prefer_phrase_chain(self, tied, label, lengths):
        """--draft-phrase-chain (H series): at a phrase START, keep the candidates whose source dancer can carry the
        REST of the phrase -- the longest chain of ``_next_source_bar`` (the chain the draft will actually take) over
        the phrase's remaining query bars, among the tied.  Measured on F11 (r40, test-20 + val-30, 2026-09-26): of
        195 bar lines where the dance did not continue its source, 49 were the source upload ENDING and 33 the next
        source bar falling outside the +-15% duration band -- cuts nobody chose, each a join between two unrelated
        dancers inside a phrase.  --draft-continue-lookahead looks one bar ahead only.  Narrowing: the longest chain
        available is kept (at least two candidates), so the pool never empties.  Library segmentation and the query's
        bar lengths only.  Counted: slots, candidates seen, candidates that already chain the whole phrase."""
        if not getattr(self, "phrase_chain", False) or not lengths or len(tied) < 2 or self._source_bars is None:
            return tied
        self.phrase_chain_slots = getattr(self, "phrase_chain_slots", 0) + 1
        reach = []
        for candidate in tied:
            n, current = 0, candidate
            for length_j in lengths:
                current = self._next_source_bar(current, label, length_j)
                if current is None:
                    break
                n += 1
            reach.append(n)
        self.phrase_chain_seen = getattr(self, "phrase_chain_seen", 0) + len(tied)
        self.phrase_chain_full = getattr(self, "phrase_chain_full", 0) + sum(r == len(lengths) for r in reach)
        order = sorted(range(len(tied)), key=lambda i: -reach[i])
        kept = [i for i in order if reach[i] == reach[order[0]]]
        if len(kept) < 2:
            kept = order[:2]
        if len(kept) < len(tied):
            self.phrase_chain_applied = getattr(self, "phrase_chain_applied", 0) + 1
        return [tied[i] for i in sorted(kept)]

    def _prefer_phrase_rhythm(self, tied, slot_music, phrase_music, label, lengths):
        """--draft-phrase-rhythm-keep F (H series): at a phrase START, keep the fraction F (at least two) of the candidates
        whose whole PHRASE fits this song's rhythm -- the candidate scored on this bar's music and each source bar
        continuation would play after it (``_next_source_bar``, the chain the draft will actually take) scored on the
        query bar it would land on, averaged.  Inside an F11 phrase the continued bars otherwise follow the SOURCE's
        music; this is the one place the query's music can choose them.  Music (query) and library motion only."""
        keep_frac = float(getattr(self, "phrase_rhythm_keep", 0.0) or 0.0)
        if (not keep_frac or not phrase_music or slot_music is None or len(tied) < 3
                or not getattr(self, "rhythm_scorer_path", None) or self._source_bars is None):
            return tied
        self.phrase_rhythm_slots = getattr(self, "phrase_rhythm_slots", 0) + 1
        total = np.asarray(self._rhythm_scores(tied, slot_music), dtype=np.float64)
        count = np.ones(len(tied))
        chains = list(tied)
        alive = list(range(len(tied)))
        for music_j, length_j in zip(phrase_music, lengths):
            nxt = []
            for i in alive:
                c = self._next_source_bar(chains[i], label, length_j)
                if c is not None:
                    chains[i] = c
                    nxt.append(i)
            if not nxt or music_j is None or len(music_j) < 4:
                break
            sc = np.asarray(self._rhythm_scores([chains[i] for i in nxt], music_j), dtype=np.float64)
            total[nxt] += sc
            count[nxt] += 1
            alive = nxt
        score = total / count
        keep = max(2, int(math.ceil(len(tied) * keep_frac)))
        order = np.argsort(-score, kind="stable")[:keep]
        if keep < len(tied):
            self.phrase_rhythm_applied = getattr(self, "phrase_rhythm_applied", 0) + 1
        return [tied[i] for i in sorted(order)]

    def _continuity_model(self):
        """(scorer, blob, device, library frames [N * W, C] on the device) -- loaded once.

        The library's continuation frames are decoded window by window (as the rhythm scorer's kinetics are)
        and cached on /cache per library + normalizer + feature version, so a round of arms decodes it once."""
        if self._continuity is None:
            from model.continuation_scorer import FEATURE_VERSION, continuation_frames, load_scorer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, blob = load_scorer(self.continuity_scorer_path, device)
            motion = self.motion
            cache = None
            source = getattr(motion, "filename", None)          # a memory-mapped release; None in tests
            if source:
                try:
                    stat = os.stat(source)
                    key = hashlib.sha1()
                    key.update("{}|{}|{}|{}|{}".format(FEATURE_VERSION, tuple(motion.shape), os.path.realpath(source),
                                                       stat.st_size, stat.st_mtime_ns).encode())
                    key.update(open(self.normalizer_path, "rb").read())
                    cache = Path(CONTINUITY_CACHE_DIR) / "library_{}.npy".format(key.hexdigest()[:20])
                except OSError:
                    cache = None
            frames = None
            if cache is not None and cache.is_file():
                frames = np.load(str(cache))
                if frames.shape[:2] != tuple(motion.shape[:2]):
                    frames = None
            if frames is None:
                for first in range(0, motion.shape[0], 64):
                    block = torch.from_numpy(np.array(motion[first:first + 64], dtype=np.float32))
                    joints = decode_motion(block.reshape(-1, block.shape[-1]), self.normalizer_path)["full_pose"]
                    part = continuation_frames(np.asarray(joints).reshape(len(block), motion.shape[1], 24, 3))
                    if frames is None:
                        frames = np.zeros((motion.shape[0],) + part.shape[1:], dtype=np.float32)
                    frames[first:first + len(block)] = part
                if cache is not None:
                    try:
                        cache.parent.mkdir(parents=True, exist_ok=True)
                        # per process: the arms of a round start together and may all miss the cache at once
                        partial = cache.with_suffix(".partial{}.npy".format(os.getpid()))
                        np.save(str(partial), frames)
                        os.replace(str(partial), str(cache))
                    except OSError:
                        pass
            window = frames.shape[1]
            flat = torch.from_numpy(frames.reshape(-1, frames.shape[-1])).to(device)
            self._continuity = (model, blob, device, flat, window)
        return self._continuity

    def _continuity_scores(self, candidates, context, target_length):
        """Learned continuation score of each candidate AS IT WILL PLAY (native range over the slot) after
        ``context`` = (continuation frames of the DRAFT placed so far, first valid frame).  Generated motion only."""
        from model.continuation_scorer import score_pool
        model, blob, device, flat, window = self._continuity_model()
        frames, valid_from = context
        starts = [int(c[0]) * window + int(c[1]) for c in candidates]
        natives = [int(c[2]) - int(c[1]) for c in candidates]
        return score_pool(model, frames, flat, starts, natives, int(target_length),
                          gap=int(blob.get("seam_mask", 6)), valid_from=int(valid_from), device=device,
                          scale_ref=float(blob.get("scale_ref", 1.0)))

    def _draft_context(self, draft, mask, start, length, gap=6):
        """(continuation frames, first valid index) of the draft's last CONTEXT_BARS * ``length`` placed frames before
        ``start``, or None when too little is placed: fewer than 6 frames beyond the scorer's seam mask ``gap`` (and
        never fewer than 12), since a context the mask hides entirely says nothing.  Unplaced frames (mask 0) end the
        context: they are not motion."""
        from model.continuation_scorer import CONTEXT_BARS, continuation_frames
        start = int(start)
        lo = max(0, start - CONTEXT_BARS * int(length) - 1)      # one frame more, so the first read rate is real
        placed = np.asarray(mask[lo:start, 0]) > 0.5
        unplaced = np.flatnonzero(~placed)
        first = lo + (int(unplaced[-1]) + 1 if len(unplaced) else 0)
        if start - first < max(12, int(gap) + 6):
            return None
        joints = decode_motion(draft[first:start].clone(), self.normalizer_path)["full_pose"]
        return continuation_frames(np.asarray(joints)), 0

    def _prefer_continuity(self, tied, continuity_context, target_length):
        """--draft-continuity-keep: keep the fraction (at least two) of the candidates the continuation scorer finds
        the most natural NEXT BAR after the motion already placed, in their ORIGINAL order, after the rhythm filter
        and before the join band.  No context (the clip's first slot) = no-op, counted."""
        keep = float(getattr(self, "continuity_keep", 0.0) or 0.0)
        if not keep or len(tied) < 3:
            return tied
        if continuity_context is None:
            self.continuity_no_context = getattr(self, "continuity_no_context", 0) + 1
            return tied
        self.continuity_slots = getattr(self, "continuity_slots", 0) + 1
        scores = self._continuity_scores(tied, continuity_context, target_length)
        count = max(2, int(math.ceil(len(tied) * keep)))
        order = np.argsort(-np.asarray(scores), kind="stable")[:count]
        if count < len(tied):
            self.continuity_applied = getattr(self, "continuity_applied", 0) + 1
        return [tied[i] for i in sorted(order)]

    def _prefer_continuable(self, tied, next_slot):
        """--draft-continue-lookahead: keep the candidates whose OWN next source bar
        can fill the next slot (its label, inside its duration band), so the bar
        line after this unit is the source dancer's own and not a cut.  Narrowing,
        like every filter here: if none qualify the pool is kept whole.  Counted."""
        if not self.continue_lookahead or next_slot is None or len(tied) < 2:
            return tied
        label, length = next_slot
        self.continue_lookahead_slots = getattr(self, "continue_lookahead_slots", 0) + 1
        kept = [c for c in tied if self._next_source_bar(c, label, length) is not None]
        if kept and len(kept) < len(tied):
            self.continue_lookahead_applied = getattr(self, "continue_lookahead_applied", 0) + 1
        return kept or tied

    def _prefer_feet_lead(self, tied):
        """Keep the tied candidates whose FEET come to rest deeper than their
        HIPS AND KNEES -- the asymmetry that is the dancer's and not ours.

        THE PREMISE IS MEASURED, in both directions.  Ground truth has it on
        20 of 20 eval clips.  The library has it on only 28.1% of windows
        (421 of 1500 sampled), so the material exists and the ranking is what
        misses it: ``min |duration difference|`` never looks, and the tie is
        then broken by a hash, which lands on a feet-leading prototype at the
        corpus base rate.  With the measured tie sizes -- 73.6% of slots tied,
        median 6 candidates -- filtering the tie lifts that to 1 - 0.72**6, or
        about 86%, and costs NOTHING on the duration criterion because every
        candidate in a tie is exactly as good on it.

        WHY IT DOES NOT REPLACE THE WARP.  Retiming cannot produce this shape:
        a monotone warp of one joint group settles every joint in the group at
        the same instants, so ``--draft-beat-anchor-per-limb legs`` put the feet
        on ground truth (0.0951 against 0.0943) and dragged hips+knees to 0.1731
        against 0.0649, inverting the very ordering it was aimed at.  The
        asymmetry has to come from the CONTENT.

        A FILTER AND NOT A RANKING, on purpose.  Ordering the whole pool by a
        new criterion is what ``--retrieval-rule phase`` did: the pipeline
        worked -- chosen-prototype phase error 0.2381 -> 0.0000 -- and the
        output got worse, because a hard ordering narrows the pool and squeezes
        out variety.  This only drops candidates that lack the property and
        hands the survivors to the existing tie-break, so the variety machinery
        is untouched.  If none qualify the whole tie is kept, so a class with no
        feet-leading exemplar degrades to today's behaviour instead of failing.
        """
        if not self.feet_lead or len(tied) < 2:
            return tied
        kept = [c for c in tied
                if self._rest_depth(c, self.FEET_JOINTS)
                > self._rest_depth(c, self.HIPS_KNEES_JOINTS)]
        return kept or tied

    HOLD_WINDOW = 5
    HOLD_RATIO = 0.4

    def _hold_share(self, candidate):
        """Share of this prototype's frames that are HOLDING a shape, cached.

        Pose displacement over ``HOLD_WINDOW`` frames below ``HOLD_RATIO`` of the
        segment's own median.  Read on the NORMALISED rot6d columns, no forward
        kinematics, because it is wanted for every candidate of every slot.  That
        is only allowed because it was shown to be the same property as the
        judged column: on 400 library windows decoded with the authoritative
        affine, Spearman rho against the joint-space hold is +0.794
        (P=7e-88, 81.5% sign agreement about the median).  Both halves are
        ratios to their own median, and the difference cancels the normaliser's
        offset, so this reading does not depend on the [-1, 1] affine at all.
        """
        key = ("hold", candidate[0], candidate[1], candidate[2])
        cached = self._rest_cache.get(key)
        if cached is not None:
            return cached
        start = ROOT_POSITION_START + ROOT_POSITION_DIMS
        window = np.asarray(self.motion[candidate[0], candidate[1]:candidate[2], start:],
                            dtype=np.float32)
        value = 0.0
        if len(window) > self.HOLD_WINDOW + 2:
            step = np.linalg.norm(window[self.HOLD_WINDOW:] - window[:-self.HOLD_WINDOW], axis=1)
            median = float(np.median(step))
            if median > 1e-9:
                value = float((step / median < self.HOLD_RATIO).mean())
        self._rest_cache[key] = value
        return value

    def _prefer_hold_by_music(self, tied, target_quiet):
        """Hold in the quiet bars, move in the busy ones -- as ground truth does.

        THE TARGET IS MEASURED, and it reads only the music.  Per bar, the share
        of frames holding a shape against that bar's onset-peak density (channel
        33), one correlation per clip, twenty eval clips: ground truth is
        NEGATIVE on 16 of 20, mean -0.225, two-sided sign test P=0.0118 -- the
        dancer holds when the music is sparse and moves when it is busy.  The
        shipped draft reads -0.082 (12/20, P=0.50) and the finished arm -0.094
        (11/20, P=0.82): nothing in the pipeline places a hold by the music.

        ROOM, same authoritative decode: 35.5% of library windows hold at least
        as much as ground truth's p75 bar, so there is material and the filter
        is not a no-op.

        WHAT THIS CANNOT PROVE.  Selecting on hold and then reading hold, or
        reading hold against the same density this filter keys on, moves those
        two columns BY CONSTRUCTION.  They confirm the plumbing, nothing more.
        The judgement is the operator's eye plus the columns this does not
        select on (jitter, variety, energy, skate, root path, flicker).

        A filter and not a ranking, with the whole tie kept when nothing
        qualifies, for the reason ``_prefer_feet_lead`` gives.
        ``target_quiet`` is ``None`` when the caller could not compute a bar
        density, and then the tie passes untouched and the slot is counted.
        """
        if not self.hold_by_music or len(tied) < 2:
            return tied
        if target_quiet is None:
            self.hold_music_blind = getattr(self, "hold_music_blind", 0) + 1
            return tied
        shares = [self._hold_share(c) for c in tied]
        middle = float(np.median(shares))
        if target_quiet:
            kept = [c for c, v in zip(tied, shares) if v >= middle and v > 0.0]
        else:
            kept = [c for c, v in zip(tied, shares) if v <= middle]
        self.hold_music_slots = getattr(self, "hold_music_slots", 0) + 1
        if target_quiet:
            self.hold_music_quiet = getattr(self, "hold_music_quiet", 0) + 1
        if kept and len(kept) < len(tied):
            self.hold_music_applied = getattr(self, "hold_music_applied", 0) + 1
        elif not kept:
            self.hold_music_empty = getattr(self, "hold_music_empty", 0) + 1
        return kept or tied

    def _prefer_beat_span(self, tied, target_length, target_beat_period):
        """Keep the tied candidates that are the same NUMBER OF BEATS as the slot.

        THE GAP THIS CLOSES, named by the operator 2026-09-16: "检索 label to
        motion 从不检测拍数是吗,对卡节奏旋律挑选 motion 没有贡献?"  It does not.
        ``_duration_band`` selects on ``abs(frames - target_length)`` alone, so a
        2.4-second prototype that is six beats of a 150 BPM song and one that is
        four beats of a 100 BPM song are equally eligible for a four-beat slot,
        and the winner is then stretched uniformly to fill it -- which is exactly
        how a movement's own preparation and landing stop having anything to do
        with THIS song (DEFECTS 1.6).

        MEASURED BEFORE THE FILTER WAS WRITTEN, on the twenty eval clips' 200
        retrieved bars, using only ``units_detail`` and the two music tracks:
        the slot is a median 4.00 query beats, the prototype a median 3.87 of
        its own; |mismatch| median 0.262 beats, p75 0.869, p90 1.211, max 3.64;
        37.5% of bars are off by more than half a beat and **40.0% import a
        prototype that is a different WHOLE NUMBER of beats than the slot**.
        ``_candidate_beat``'s own docstring had half of this already -- 68.1% of
        the duration rule's picks are at a tempo more than 10% from the query's.

        A FILTER AND NOT A RANKING, for the reason ``_prefer_feet_lead`` states:
        ``--retrieval-rule phase`` ordered the whole pool by a new criterion, its
        plumbing worked and the output got worse, because a hard ordering
        squeezes out variety.  This drops the candidates that are the wrong
        length in beats and hands the rest to the existing tie-break; if none
        qualify the whole tie is kept, so a class with no beat-matched exemplar
        degrades to today's behaviour instead of failing.

        Reads the query's beat grid and the candidate recording's own beat grid.
        Neither is the target clip's MOTION, so this is a legal inference-time
        input under DEFECTS 1.6.
        """
        if not self.beat_span_tolerance or len(tied) < 2:
            return tied
        if not target_beat_period or target_beat_period != target_beat_period:
            self.beat_span_blind = getattr(self, "beat_span_blind", 0) + 1
            return tied
        want = float(target_length) / float(target_beat_period)
        kept = []
        for candidate in tied:
            _phase, period = self._candidate_beat(candidate)
            if period != period or period <= 0:
                continue
            if abs((candidate[2] - candidate[1]) / period - want) <= self.beat_span_tolerance:
                kept.append(candidate)
        self.beat_span_slots = getattr(self, "beat_span_slots", 0) + 1
        if kept and len(kept) < len(tied):
            self.beat_span_applied = getattr(self, "beat_span_applied", 0) + 1
        elif not kept:
            self.beat_span_empty = getattr(self, "beat_span_empty", 0) + 1
        return kept or tied

    def _vertical_flags_of(self, candidate):
        """OR of the per-frame hop/squat bits over the candidate's played range."""
        sample, start, end, _ = candidate
        series = self._vertical_flags[self.names[sample]]
        played = np.asarray(series[int(start):max(int(start) + 1, int(end))])
        return int(np.bitwise_or.reduce(played)) if len(played) else 0

    def _prefer_hop_guard(self, tied):
        """Keep the candidates that do not leave the floor.

        MEASURED BEFORE WRITING, 2026-09-17, twenty eval clips, lowest foot more
        than 15 cm over its own 2 s floor (``tools/census_release_vertical.py``):
        fix5 is airborne on 3.45% of frames against ground truth's 0.57%, higher
        on 15 of 17 clips that differ -- and its DRAFT already reads 3.46%, so
        completion passes the hops through and the choice is made here.  The
        operator saw it on 7676475940934935409:clip001 (one-knee hop, f300).

        NOT MUSIC-KEYED, and that is a measurement, not an omission: on the
        train split, bars that contain a hop or deep squat are no quieter than
        the rest of their song (MFCC c0 z +0.004, 57/104 uploads, P=0.38), and
        the two clips the operator flagged are the two LOUDEST of the ten
        visual clips.  A "no hops in quiet music" rule would not reach them.

        A narrowing after the beat-span filter, so hitting the beat still comes
        first; the whole tie is kept when nothing qualifies.
        """
        if not self.hop_guard or len(tied) < 2:
            return tied
        kept = [c for c in tied if not self._vertical_flags_of(c) & self.HOP_FLAG]
        self.hop_guard_slots = getattr(self, "hop_guard_slots", 0) + 1
        if kept and len(kept) < len(tied):
            self.hop_guard_applied = getattr(self, "hop_guard_applied", 0) + 1
        elif not kept:
            self.hop_guard_empty = getattr(self, "hop_guard_empty", 0) + 1
        return kept or tied

    MUSIC_ENERGY_QUANTILES = {True: 0.5, False: 0.8}

    def _prefer_music_energy(self, tied, target_soft):
        """Cap the movement size by how loud this bar is.

        The operator, 2026-09-17: "音乐节奏和旋律的强度要匹配整体 motion
        energy ... 在满足卡点旋律节奏的情况下(卡点舞高优)".

        WHAT THE DATA ALLOWS.  Of the six music-intensity features available at
        inference, only loudness (MFCC c0, channel 1) predicts ground-truth
        motion energy on the T-line train split: within a song rho +0.16 (85/128
        uploads, P=0.0003, beats a bar shuffle and another song's music), across
        songs +0.34 (P<1e-4).  It is small -- the louder half of a song moves
        6.8% faster -- so this is a CAP, not a target.  Onset strength and
        onset-peak density predict nothing within a song on train.

        WHAT IT IS FOR.  fix5's draft moves 11% more than ground truth (0.687 vs
        0.618 m/s, 16/20 clips), about equally on quiet and loud songs.  The
        operator's criteria are small, well-placed movements, not size.

        THE RULE.  A bar is quiet when its mean loudness is below the library's
        median (``quiet_loudness``).  Quiet bars keep the candidates at or below
        the pool's median energy; loud bars only drop the top fifth.  Applied
        after the beat-span filter; the whole tie is kept when nothing qualifies.

        NO SQUAT BAN, and the first version had one.  With it, on the 20 eval
        drafts, deep-squat frames fell to 0.38% against ground truth's 3.95%:
        ground truth keeps a hop or deep squat in 19.9% of its quiet bars, and
        once --draft-unit-floor stops floating bars from reading as their
        neighbours' squats, the draft's real squats are already below ground
        truth (1.48%).  The operator asked for "not too many", not none; the
        energy cap is what limits them.
        """
        if not self.music_energy or len(tied) < 2:
            return tied
        if target_soft is None:
            self.music_energy_blind = getattr(self, "music_energy_blind", 0) + 1
            return tied
        energies = np.array([self._segment_energy(c) for c in tied])
        cut = float(np.quantile(energies, self.MUSIC_ENERGY_QUANTILES[bool(target_soft)]))
        kept = [c for c, e in zip(tied, energies) if e <= cut]
        if target_soft:
            self.music_energy_quiet = getattr(self, "music_energy_quiet", 0) + 1
        self.music_energy_slots = getattr(self, "music_energy_slots", 0) + 1
        if kept and len(kept) < len(tied):
            self.music_energy_applied = getattr(self, "music_energy_applied", 0) + 1
        elif not kept:
            self.music_energy_empty = getattr(self, "music_energy_empty", 0) + 1
        return kept or tied

    JOIN_VELOCITY_WEIGHT = 4.0
    # How many of the best-joining candidates the draw picks from.  1 is argmax
    # and reproduces the 2026-09-05 measurement above; the default is 4 because
    # that is the smallest pool that can hold a repeat of the same class without
    # re-using its first pick (the exclusion above removes one).  Not swept.
    JOIN_TOP_K = 4

    def _phase_cost(self, candidate, target_phase):
        """How far this candidate's OWN start phase is from the slot's, in beats.

        WHY A RANKING TERM AND NOT A WARP.  Two attempts to fix the timing by
        warping the content are now refuted: ``--draft-beat-anchor`` (settle
        points onto beats) and ``--draft-music-anchor`` (onto beat + the train
        split's measured 0.200 lag) both made the beat-phase gain WORSE, the
        second monotonically in stretch -- +0.0145 shipped, +0.0064 at 1.2,
        -0.0132 at 1.6, against ground truth's +0.0314.  The reason is in the
        offline measurement that came first: randomly substituted REAL bars keep
        most of the lock, so a prototype's internal timing is already musical,
        and warping it destroys real structure to impose an assumed one.

        So the timing has to be bought by CHOOSING, not by stretching.  A
        prototype is four beats of its own song; played from a slot that starts
        on a beat, its internal accents land on beats only if it too started on
        one.  Measured over 361 plan segments, the duration rule's picks sit at
        a median phase error of 0.250 -- exactly the uniform-random expectation,
        i.e. the current pipeline knows nothing about this.

        WHAT IT MUST NOT BECOME.  ``--retrieval-rule phase`` hard-sorted the
        band down to one answer and squeezed out the variety the recurrence fix
        had just bought.  This is a TERM in the seam-aware ranking, so the
        existing top-k draw is untouched -- the same shape the turn penalty
        used, and the reason that one was rejected does not apply here: it
        penalised turning, which is content, and this penalises misalignment,
        which is free.

        Reads the LIBRARY's music (the prototype's own recording) and the
        query's beat grid.  No target motion.
        """
        phase, _period = self._candidate_beat(candidate)
        if not np.isfinite(phase) or target_phase is None or not np.isfinite(target_phase):
            return 0.0
        difference = abs(float(phase) - float(target_phase)) % 1.0
        return min(difference, 1.0 - difference)

    RHYTHM_BINS = 8

    def _rhythm_pattern(self, candidate):
        """The candidate's own within-span ONSET pattern, in ``RHYTHM_BINS`` bins.

        Read from the LIBRARY's music -- channel 33, the onset-peak one-hot,
        frame-aligned with the library's motion -- so it describes the rhythm the
        prototype was danced to.  Normalised to sum 1 so a loud bar and a quiet
        bar with the same pattern compare equal, and the comparison is about
        WHERE the onsets fall, not how many there are.
        """
        hit = self._rhythm_cache.get(candidate[:3])
        if hit is not None:
            return hit
        pattern = None
        if self.music is not None:
            sample, start, end, _ = candidate
            column = np.asarray(self.music[sample, start:end, MUSIC_ACCENT_CHANNEL])
            if len(column) >= self.RHYTHM_BINS:
                edges = np.linspace(0, len(column), self.RHYTHM_BINS + 1).astype(int)
                counts = np.array([column[a:b].sum() for a, b in zip(edges[:-1], edges[1:])],
                                  dtype=np.float64)
                total = counts.sum()
                if total > 0:
                    pattern = counts / total
        self._rhythm_cache[candidate[:3]] = pattern
        return pattern

    def _rhythm_cost(self, candidate, target_pattern):
        """How unlike the query bar's rhythm this prototype's own bar was.

        WHY THIS AND NOT THE THREE THINGS ALREADY REFUTED.  The timing cannot be
        fixed by warping the content -- ``--draft-beat-anchor`` and
        ``--draft-music-anchor`` both made the beat-phase gain worse, the second
        monotonically in stretch, because a prototype's internal timing is
        already musical and a warp destroys real structure to impose an assumed
        one.  Nor is there a start-phase to fix: measured on the shipped arm,
        the chosen prototypes' median phase error is **0.000**, because
        ``--plan-bar-grid`` and the beat-4 segmentation make every prototype
        start on a beat by construction.  (The "median 0.250, exactly uniform
        random" figure in ``_candidate_beat`` predates that configuration.)

        So beat-LEVEL alignment is already right, and what is left is the
        pattern BETWEEN the beats: a prototype is four beats of ITS song, and
        the dancer's accents inside it answer THAT song's rhythm.  This prefers
        a prototype whose own bar had a rhythm like the one now playing.  Music
        on both sides -- the library's and the query's -- so nothing is warped
        and no motion is read.
        """
        if target_pattern is None:
            return 0.0
        pattern = self._rhythm_pattern(candidate)
        if pattern is None:
            return 0.0
        return float(np.abs(pattern - target_pattern).sum())

    @staticmethod
    def _lever_weights():
        """How far a unit rotation at each joint moves the BODY.

        WHY THE JOIN COST NEEDS THEM.  ``_join_cost`` is an equal-weight L2 over
        3 root dims and 24 rot6d blocks, so a pelvis degree and a wrist degree
        cost the same.  They are not the same: perturbing the pelvis moves every
        joint below it, perturbing a hand moves nothing, and the measured seam
        spike is overwhelmingly in the EXTREMITIES precisely because they are
        the ones a proximal discontinuity throws furthest -- at a bar line the
        draft's hands peak at 25.7x their own median speed change and its elbows
        at 19.1x, while its ROOT peaks at 5.7x (ground truth is a flat 2.6-3.4x
        everywhere, having no seams).

        MEASURED, not asserted: over 150 random butt-joined library pairs the
        equal-weight cost correlates with the actual world-space speed jump at
        Spearman rho +0.372, and this weighting at **+0.973**.  The ranking that
        exists to minimise the seam was only weakly related to it.

        Computed from the SMPL rest skeleton once per process: each joint is
        perturbed in rot6d and the mean displacement of all 24 joints is read.
        """
        if IndexedAtomicMotionLibrary._LEVER_CACHE is None:
            from dataset.quaternion import ax_from_6v
            from vis import SMPLSkeleton

            skeleton = SMPLSkeleton()
            identity = torch.zeros(1, 1, 24, 6)
            identity[..., 0] = 1.0
            identity[..., 4] = 1.0
            root = torch.zeros(1, 1, 3)

            def joints_of(values):
                return skeleton.forward(
                    ax_from_6v(values.reshape(-1, 24, 6)).unsqueeze(0), root)[0, 0]

            base = joints_of(identity)
            generator = torch.Generator().manual_seed(0)
            lever = []
            for joint in range(24):
                moved = []
                for _ in range(8):
                    probe = identity.clone()
                    probe[0, 0, joint] += torch.randn(6, generator=generator) * 0.05
                    moved.append(float((joints_of(probe) - base).norm(dim=-1).mean()))
                lever.append(float(np.mean(moved)))
            # ONLY THE ROTATIONS ARE RESCALED.  The three root columns are
            # already metres of travel, which is the unit this whole cost wants
            # to be in; the 144 rot6d columns are dimensionless, and the lever
            # arm is what converts a rotation into the metres of body movement
            # it causes.  Dividing the whole vector by its mean -- the first
            # version -- rescaled the root too, moving the root's importance
            # against the rotations by an arbitrary factor, which a unit test
            # caught before any arm was generated with it.
            rotation = np.repeat(np.asarray(lever), 6)
            rotation = rotation / rotation.mean()
            IndexedAtomicMotionLibrary._LEVER_CACHE = torch.as_tensor(
                np.concatenate([np.ones(ROOT_POSITION_DIMS), rotation]),
                dtype=torch.float32)
        return IndexedAtomicMotionLibrary._LEVER_CACHE

    def _join_cost(self, candidate, join_tail, target_length=None):
        """How badly this candidate's first frames continue what precedes it.

        THIS CRITERION IS MY OWN CONSTRUCTION (CLAUDE.md 2.1 rule 1), not the
        paper's and not fitted.  It exists because the operator asked for the
        draft to "choose the motion looking at the previous bar", and because
        the measured defect is at the join: seam jerk 0.3515 against ground
        truth's 0.2553 at the same frame indices, 17 eval clips, filler frames
        excluded (2026-09-05).

        POSITION AND VELOCITY, not position alone.  A cut that lands on the
        right pose but the wrong direction of travel still snaps, and the defect
        is measured in jerk -- a derivative.  ``JOIN_VELOCITY_WEIGHT`` is 4.0
        because one frame of velocity is 1/30 s of displacement, so a weight of
        about 4 puts a full frame of velocity mismatch on the same footing as a
        pose mismatch of the same size; it is a scale choice, NOT a fit, and
        nothing here has been swept over it.

        CONTACTS ARE EXCLUDED.  The four contact channels are binary and are
        SUPPOSED to flip when the dancer changes support -- 11.61% of adjacent
        training frames flip one -- so scoring them would penalise exactly the
        weight transfers the operator says ground truth is full of, and they
        would dominate a plain L2 the way they dominate the velocity loss (86%
        of its budget, 2026-09-01 census).

        Raw units on both sides: ``join_tail`` is already unnormalized by the
        caller, so the candidate is unnormalized here rather than the tail being
        renormalized -- one direction, so a release's own normalizer cannot
        change the ranking.
        """
        sample, start, end, _ = candidate
        if self.join_frame == "placed":
            head = self._placed_head(candidate, target_length, join_tail)
        else:
            head = torch.from_numpy(
                np.array(self.motion[sample, start:min(start + 2, end)], copy=True))
            if len(head) < 2:
                head = head.repeat(2, 1)[:2]
            scale, offset = self._normalizer_affine()
            head = head * scale + offset
        # CONTACT_CHANNELS is a COUNT (4), and the contacts are the first four
        # columns, so dropping them is a slice rather than a membership test.
        head = head[:, CONTACT_CHANNELS:]
        tail = join_tail[:, CONTACT_CHANNELS:]
        weights = (self._lever_weights().to(head.dtype)
                   if self.join_lever_weights else 1.0)
        pose_weights = weights
        if self.join_height_weight:
            # Root z is column 2 once the contacts are dropped.  Metres, in the
            # placed frame, so a 10 cm pelvis step costs join_height_weight/10.
            pose_weights = torch.ones(head.shape[1], dtype=head.dtype) * weights
            pose_weights[ROOT_POSITION_DIMS - 1] *= self.join_height_weight
        pose = torch.linalg.vector_norm((head[0] - tail[-1]) * pose_weights)
        if len(tail) >= 2:
            velocity = torch.linalg.vector_norm(
                ((head[1] - head[0]) - (tail[-1] - tail[-2])) * weights)
        else:
            velocity = torch.zeros((), dtype=pose.dtype)
        return float(self.join_pose_weight * pose
                     + self.JOIN_VELOCITY_WEIGHT * velocity)

    def _placed_head(self, candidate, target_length, join_tail):
        """The candidate's first two frames AS BUILD_DRAFT WILL PLACE THEM, raw units.

        WHY.  ``join_tail`` is taken from the draft after every placement step:
        floor subtracted (``_values_at``), resampled to its slot, turned by facing
        continuity, and its ground position chained onto the previous unit.  The
        raw head skipped all four, so ``_join_cost`` compared two frames that are
        not in the same space.  Measured 2026-09-17 on fix5's 20 eval drafts
        (179 joins, replay matched 200/200 units): median ground-position gap
        0.63 m, median heading gap 28.6 deg, median height gap 0.35 m -- almost
        exactly the source recording's floor.  About a fifth of the 0.35 join
        band was admitted or refused on where the library clip happened to stand
        in its own upload.  The operator, same day: "root 的位置没太大关系,
        后面 completion 的时候可以把 root 合到一起".

        What this does to the head, in build_draft's order:
          * ``_values_at`` -- the floor and the slot's time base, so the velocity
            term compares steps of the same length;
          * a turn about z that puts the head's first-frame heading on the
            tail's last, which is what facing continuity does before the anchor
            ramp (the ramp is zero and flat at frame 0);
          * the ground position moved onto the tail's, which is what root
            continuity does, so the ground-position difference is zero by
            construction and only the direction of travel still counts.
        Root z is NOT moved: continuity is xy under the shipped config, so a
        height gap here is a height step the draft will really contain.
        """
        sample, start, end, _ = candidate
        length = int(target_length) if target_length else int(end - start)
        values = self._values_at(candidate, max(length, 1))[:2]
        if len(values) < 2:
            values = values.repeat(2, 1)[:2]
        scale, offset = self._normalizer_affine()
        tail_yaw, _ = self._facing_yaw(join_tail[-1:])
        head_yaw, _ = self._facing_yaw(values[:1] * scale + offset)
        delta = torch.remainder(tail_yaw[0] - head_yaw[0] + math.pi,
                                2 * math.pi) - math.pi
        head = self._rotate_about_z(values, delta) * scale + offset
        ground = slice(ROOT_POSITION_START, ROOT_POSITION_START + 2)
        head[:, ground] += join_tail[-1, ground] - head[0, ground]
        self.join_placed_calls = getattr(self, "join_placed_calls", 0) + 1
        return head

    def retrieve(self, label, target_length, *, exclude_retrieval_group_ids=(),
                 target_period=None, occurrence=0, variety_rng=None,
                 target_phase=None, target_beat_period=None, context=None,
                 target_rhythm=None, target_beats=(),
                 slot_start=None, join_tail=None, target_quiet=None,
                 target_soft=None, next_slot=None, slot_music=None,
                 continuity_context=None, phrase_music=None, phrase_lengths=None):
        """``occurrence`` is which repeat of this label in the clip this is.

        Zero means the first, and the first is served exactly as before -- the
        rule's own pick, cached.  Later occurrences draw a different prototype
        from inside the duration rule's own tolerance, which is what makes a
        repeated label stop being the same bytes.

        WHY.  The default ``duration`` rule is a deterministic ``min`` and the
        result is cached on ``(label, target_length, ...)``, so every repeat of
        a label in a clip returned **the identical tensor**.  The bar grid makes
        that worse by construction: it forces every segment to the same number
        of beats, so ``target_length`` is near-constant across a clip and the
        cache key collides on every repeat.  Measured on the shipped arm, the
        same-label span pairs sit 0.128 m apart against 0.165 m for
        different-label pairs -- a 22% contrast where the label is supposed to
        be the thing that differs.  A clip of 8 segments and a handful of
        distinct classes is then two or three snippets replayed, which is the
        "monotonous, one repeated move" the operator reports.

        This is the paper's own answer, not an invention: "as choreographic
        theory indicates, structured movements should exhibit variation when
        they recur" (atomicDance 3.4).  The paper spends that variation as
        masked noise inside the completion; the retrieval side of it -- pick a
        different exemplar of the same movement -- is cheaper, and the ablation
        it must not become is the paper's own ``Random Choice`` row, which is
        worse on FID because unconstrained draws need heavier time-stretching.
        Hence the draw is confined to the duration rule's existing tolerance,
        ``max(2, 0.15 * target_length)``, the same band ``tempo`` already uses.

        ``variety_rng`` is the CLIP's generator (``_variety_rng(seed, name)``),
        not the run's.  It was the run's until 2026-08-31, and that made a
        clip's prototype picks depend on which clips were processed before it:
        reversing an 8-clip list moved the draft on 6 of them by up to 2.19 m
        while appending 12 clips moved nothing.  A clip must depend on its name
        and the seed, and on nothing else.
        """
        excluded = (
            frozenset((exclude_retrieval_group_ids,))
            if isinstance(exclude_retrieval_group_ids, str)
            else frozenset(exclude_retrieval_group_ids)
        )
        # A cached no-exclusion result must never be reused for a source-safe
        # query (and vice versa).
        key = (int(label), int(target_length), tuple(sorted(excluded)),
               self.retrieval_rule, self.tie_break, self.energy_floor_quantile,
               None if target_period is None else round(float(target_period), 2),
               # The phase rule's answer depends on WHERE in the bar this
               # segment starts, so two segments of the same class and length
               # must not share a cache entry.
               None if target_phase is None else round(float(target_phase), 3),
               None if target_beat_period is None else round(float(target_beat_period), 2),
               # WHERE THIS SLOT'S BEATS FALL.  --draft-beat-fit judges a
               # candidate against them, so two slots of the same class and
               # length whose beats sit differently must not share an entry.
               tuple(int(b) for b in target_beats)
               if (self.beat_fit or self.feet_beat_lead) else (),
               # A quiet and a loud slot of the same class and length get
               # different caps under --draft-music-energy.
               target_soft if self.music_energy else None)
        if self.retrieval_rule in ("random", "learned"):
            # Caching would hand every occurrence of a class the same draw,
            # which is the behaviour these rules exist to avoid.  ``learned``
            # also depends on the SEAM it is being asked to make, which is not
            # in the key at all -- caching it would return a pick made for a
            # different predecessor.
            key = None
        if occurrence:
            # Same reason, one level finer: the cache is what makes a repeat
            # identical, so a repeat must not read it or write it.
            key = None
        if key is not None and key in self._retrieval_cache:
            # A cache hit is still a unit the draft used, so it must appear in
            # the log too -- otherwise a clip that repeats one class reports
            # fewer units than it played, and the count gate under-reads.
            record = self._retrieval_record_cache.get(key)
            if record is not None:
                self.retrieval_log.append(dict(record))
            return self._retrieval_cache[key].clone()
        candidates = self.index.get(int(label), ())
        bar_index = getattr(self, "bar_index", None)
        if bar_index is not None and target_beat_period and target_beat_period == target_beat_period:
            # --draft-bar-units: a slot of the bar's own beat count takes WHOLE
            # source bars (a unit that starts and ends on its dancer's bar line);
            # a partial or odd slot keeps the run index, where a fragment of the
            # right length exists and a whole bar does not.  So does a slot no
            # bar of this class is long enough for within the duration band's
            # own slack -- the replacement version played those at 0.82x.
            slot_beats = float(target_length) / float(target_beat_period)
            slack = max(2.0, 0.15 * float(target_length))
            bars = tuple(b for b in bar_index.get(int(label), ())
                         if abs((b[2] - b[1]) - float(target_length)) <= slack)
            if abs(slot_beats - self.bar_beats) <= BAR_SLOT_TOLERANCE and bars:
                candidates = bars
                self.bar_units_slots = getattr(self, "bar_units_slots", 0) + 1
            else:
                self.bar_units_fallback = getattr(self, "bar_units_fallback", 0) + 1
        if excluded:
            candidates = tuple(
                candidate
                for candidate in candidates
                # Fail closed: unknown candidate provenance cannot prove it is
                # outside the query recording.
                if isinstance(candidate[3], str)
                and candidate[3]
                and candidate[3] not in excluded
            )
        candidates = self._energy_floor(label, candidates, self.energy_floor_quantile)
        candidates = self._reject_yaw_snaps(candidates)
        candidates = self._reject_speed_spikes(candidates)
        if not candidates:
            if excluded:
                raise KeyError(
                    "no retrieval-group-safe training prototype for atomic label {}".format(label)
                )
            raise KeyError("no training prototype for atomic label {}".format(label))
        if self.retrieval_rule == "random":
            # The rule the FID says to test.  ``medoid`` is 10.8% closer to the
            # true motion per frame and 5-18% *worse* on fid_k, and div_k falls
            # with it (ORACLE 10.579 -> 10.429 against ground truth's 10.742):
            # picking each class's most typical member hands every occurrence of
            # a class the same prototype, and the draft stops being varied.  If
            # variety is what the draft is for, the maximum-variety rule is the
            # one to measure, and it is also the cheapest thing in this method.
            #
            # Drawn from the run's own generator, so a run is still reproducible
            # from its seed -- but two runs that differ only in batch order are
            # not, which is why this is a diagnostic rule and not the default.
            chosen = candidates[int(torch.randint(len(candidates), (1,)).item())]
        elif self.retrieval_rule == "medoid":
            chosen = candidates[self._medoid_index(candidates, target_length)]
        elif self.retrieval_rule == "duration":
            chosen = self._duration_pick(candidates, target_length, excluded,
                                         target_beats,
                                         beat_period=target_beat_period,
                                         quiet=target_quiet,
                                         soft=target_soft)
        elif self.retrieval_rule == "tempo":
            # Duration still has to be respected -- a prototype resampled by more
            # than a little stops being the movement it was -- so this is a
            # tie-break inside the duration rule rather than a replacement for
            # it: among the candidates whose length is within a tolerance of the
            # span, prefer the one whose own settle period is closest to the
            # query's.  With no query period supplied it degrades to duration
            # exactly, and says so by returning the same prototype.
            if target_period is None or not np.isfinite(target_period):
                chosen = min(candidates,
                             key=lambda item: abs((item[2] - item[1]) - target_length))
            else:
                lengths = [abs((c[2] - c[1]) - target_length) for c in candidates]
                slack = max(2.0, 0.15 * target_length)
                near = [c for c, d in zip(candidates, lengths) if d <= slack]
                if not near:
                    best = min(lengths)
                    near = [c for c, d in zip(candidates, lengths) if d <= best]
                scored = []
                for candidate in near:
                    period = self._settle_period(candidate)
                    if np.isfinite(period):
                        scored.append((abs(period - float(target_period)), candidate))
                chosen = (min(scored)[1] if scored else
                          min(near, key=lambda item: abs((item[2] - item[1]) - target_length)))
        elif self.retrieval_rule == "learned":
            # Score the join, then SAMPLE.  See model/retrieval_selector.py for
            # why the output is a temperature draw over the top k rather than
            # an argmax, and for the two leaks the training had to close before
            # any of this could mean anything.
            from model.retrieval_selector import candidate_features_many

            near = self._duration_band(candidates, target_length)
            # THE VARIETY GUARDS DO NOT REACH THIS RULE, and until 2026-09-13
            # nothing had noticed, because this rule had never produced an arm.
            # The recurrence machinery below is gated on
            # ``self.retrieval_rule in ("duration", "tempo", "medoid", "phase")``
            # -- "learned" is not in that list -- so the selector scored every
            # slot independently and re-picked whatever scored highest.  The
            # operator saw the result before any column did: "我看的这个视频有
            # 高度的重复动作,重复了三次".  Measured on that clip
            # (wild_v5:7664973324456613370:clip000), 25% of its bars replay an
            # earlier bar of the same clip, against 0% for both the baseline and
            # ground truth.
            #
            # Excluded BEFORE scoring rather than penalised inside it: a score
            # that merely disfavours a repeat still returns it when the pool is
            # thin, and the defect is binary on screen -- the viewer either sees
            # the same movement again or does not.  The fallbacks keep the run
            # alive when the exclusion empties the pool, and are counted.
            # ``_used_this_clip`` is populated by build_draft only when the
            # recurrence fix is on, so an empty set means it is off and this
            # filter is a no-op -- no separate flag to keep in sync.
            if self._used_this_clip or self._used_groups_this_clip:
                unplayed = [c for c in near if c not in self._used_this_clip]
                unheard = [c for c in unplayed
                           if c[3] not in self._used_groups_this_clip]
                near = unheard or unplayed or near
            # THE JOIN-COST BAND, which this rule also never received.  The
            # seam-aware ranking is documented two branches below as selecting
            # AGAINST BIG MOVEMENTS ("a prototype that begins with the arms
            # overhead is far from a tail whose arms are down and so scores as a
            # poor continuation"), and "learned" is not in the rule list that
            # gets it -- the third consequence of that one list, after the
            # discarded selection and the missing variety guards.
            #
            # Measured 2026-09-13: the selector's picks move 25% faster than
            # ground truth (0.0306 against 0.0245 m/frame) while the baseline
            # sits at 0.0257, and tightening the draw to top-k 2 / T 0.3 does
            # NOT reduce it (0.0306, unchanged) -- so the size comes from the
            # selector's preference, not from sampling.  energy is 1.279 of
            # ground truth, outside the [0.85, 1.15] band, with foot skate 0.524
            # against 0.295.
            #
            # Narrowed, not overridden: the selector still ranks inside the band
            # (that is the whole point of the rule, and overriding it is the
            # defect fixed in the seam-aware branch), it simply may not reach
            # the candidates that join worst.
            # COUNTED at the point of use, not at the point of configuration:
            # "the flag was set" and "the band actually narrowed anything" are
            # different claims, and this repository has twice shipped a reading
            # for a flag that changed the manifest and not the dance.
            # THE SELECTION FILTERS, which this rule also never received -- the
            # FOURTH consequence of that one rule list, after the discarded
            # selection, the missing variety guards and the missing join band.
            # Measured 2026-09-16 on the twenty eval clips: a draft-only arm with
            # --draft-feet-lead recorded ``draft_feet_lead: true`` in its manifest
            # and came out BYTE-IDENTICAL to the baseline on 20 of 20 clips, and
            # every column agreed to four decimals.  The shipped configuration is
            # --retrieval-rule learned, so all three of --draft-feet-lead,
            # --draft-beat-fit and --draft-feet-beat-lead have been dead in it.
            #
            # tests/test_selection_survives_seam_aware.py passed throughout: it
            # asserts the filter names appear inside the seam-aware branch, which
            # they do, and never asked whether that branch RUNS for the rule the
            # shipped config uses.  That is CLAUDE.md section 2's gate that can
            # never fire, reading as "checked"; the replacement assertion is in
            # tests/test_selection_reaches_every_rule.py.
            #
            # Applied as a NARROWING, matching the seam-aware branch: the
            # selector still ranks inside the band -- overriding it is the defect
            # that branch already fixed -- the filters only say which candidates
            # are eligible, and if none qualify the band is kept whole.
            narrowed = self._prefer_beat_span(near, target_length, target_beat_period)
            narrowed = self._prefer_tempo(narrowed, target_length)
            narrowed = self._prefer_hop_guard(narrowed)
            narrowed = self._prefer_music_energy(narrowed, target_soft)
            narrowed = self._prefer_hold_by_music(narrowed, target_quiet)
            narrowed = self._prefer_feet_beat_lead(narrowed, target_length, target_beats)
            narrowed = self._prefer_beat_fit(narrowed, target_length, target_beats)
            narrowed = self._prefer_feet_lead(narrowed)
            narrowed = self._prefer_continuable(narrowed, next_slot)
            narrowed = self._prefer_phrase_chain(narrowed, label, phrase_lengths or [])
            narrowed = self._prefer_step_lock(narrowed, target_length, target_beats, label, phrase_music,
                                              phrase_lengths or [])
            narrowed = self._prefer_energy_follow(narrowed, target_length, slot_start, label, phrase_lengths or [])
            narrowed = self._prefer_hit(narrowed, target_length, target_beats, label, phrase_music, phrase_lengths or [])
            narrowed = self._prefer_full(narrowed)
            narrowed = self._prefer_rhythm(narrowed, slot_music, continuity_context, target_length)
            narrowed = self._prefer_phrase_rhythm(narrowed, slot_music, phrase_music, label, phrase_lengths or [])
            narrowed = self._prefer_continuity(narrowed, continuity_context, target_length)
            if narrowed and len(narrowed) < len(near):
                self.selection_filter_applied = getattr(
                    self, "selection_filter_applied", 0) + 1
            near = narrowed or near
            self.selection_filter_slots = getattr(
                self, "selection_filter_slots", 0) + 1
            self.selector_band_slots = getattr(self, "selector_band_slots", 0) + 1
            if self.selector_join_band and join_tail is not None and len(near) > 2:
                keep = max(2, int(round(len(near) * float(self.selector_join_band))))
                before = len(near)
                near = sorted(near, key=lambda c: self._join_cost(
                    c, join_tail, target_length))[:keep]
                if len(near) < before:
                    self.selector_band_applied = getattr(
                        self, "selector_band_applied", 0) + 1
            elif self.selector_join_band:
                self.selector_band_skipped = getattr(
                    self, "selector_band_skipped", 0) + 1
            if context is None or len(near) == 1:
                # ``context is None`` means a caller reached ``retrieve`` without
                # going through ``build_draft``; a band of one means there is
                # nothing to choose.  Both fall back to the shipped rule and are
                # counted, so a manifest can never imply the selector ran when
                # it did not.
                chosen = min(near, key=lambda item: abs((item[2] - item[1]) - target_length))
                self.selector_fallbacks = getattr(self, "selector_fallbacks", 0) + 1
            else:
                if context.previous_tail is None:
                    # The clip's FIRST conditioned segment has no join to score.
                    # It is still scored, not skipped: the rest of the row (the
                    # candidate's shape, and how it fits what comes next) is
                    # real, and ``candidate_features`` makes the seam block read
                    # zero rather than something invented.  Counted separately
                    # because "the selector ran" and "the selector had a
                    # predecessor to score against" are different claims.
                    self.selector_no_predecessor = getattr(
                        self, "selector_no_predecessor", 0) + 1
                if int(context.target_length) != int(target_length):
                    raise ValueError(
                        "selector context was built for a {}-frame span but "
                        "retrieval asked for {}".format(
                            context.target_length, target_length))
                # A candidate the descriptor cannot read is dropped rather
                # than scored: one-frame spans exist in the pool and returning
                # None for them is what keeps the run alive (see _descriptor).
                described = [(c, self._descriptor(c)) for c in near]
                described = [(c, d) for c, d in described if d is not None]
                if len(described) < 2:
                    # Nothing left to choose between -- fall back to the shipped
                    # rule for this slot instead of pretending the selector ran.
                    chosen = self._duration_pick(near, target_length, excluded,
                                                 target_beats,
                                                 beat_period=target_beat_period,
                                                 quiet=target_quiet,
                                                 soft=target_soft)
                    self.selector_fallbacks = getattr(
                        self, "selector_fallbacks", 0) + 1
                else:
                    rows = candidate_features_many(
                        context, [d for _c, d in described])
                    pick = self.selector.select(
                        rows, top_k=self.selector_top_k,
                        temperature=self.selector_temperature,
                        generator=variety_rng)
                    chosen = described[pick][0]
                    self.selector_calls = getattr(self, "selector_calls", 0) + 1
        elif self.retrieval_rule == "phase":
            # Pick the prototype that would land on THIS query's beats.
            #
            # Two numbers, both in units of the query's own beat, so they can be
            # added without a hand-set weight: how far the candidate's own start
            # phase is from the query segment's (wrapped, so 0.95 and 0.05 are
            # near), and how far its tempo is.  Duration stays a hard filter
            # rather than a term -- a prototype that has to be stretched 40% has
            # had its internal timing destroyed before any of this matters, and
            # that filter is the same slack band ``tempo`` and the recurrence
            # draw already use.
            lengths = [abs((c[2] - c[1]) - target_length) for c in candidates]
            slack = max(2.0, 0.15 * target_length)
            near = [c for c, d in zip(candidates, lengths) if d <= slack]
            if not near:
                best_length = min(lengths)
                near = [c for c, d in zip(candidates, lengths) if d <= best_length]
            chosen = min(near, key=lambda item: abs((item[2] - item[1]) - target_length))
            if target_phase is not None:
                scored = []
                for candidate in near:
                    phase, period = self._candidate_beat(candidate)
                    if not np.isfinite(phase):
                        continue
                    cost = abs(((phase - float(target_phase) + 0.5) % 1.0) - 0.5)
                    if target_beat_period and np.isfinite(period) and target_beat_period > 0:
                        cost += abs(period - float(target_beat_period)) / float(target_beat_period)
                    scored.append((cost, candidate))
                if scored:
                    best = min(scored)[0]
                    # Everything within a tenth of a beat of the best, so the
                    # recurrence draw below still has somewhere to go; without
                    # this the phase rule would hand every repeat of a label the
                    # same prototype, which is the defect --draft-recurrence
                    # -variety exists to prevent.
                    near = [c for cost, c in scored if cost <= best + 0.10] or [min(scored)[1]]
                    chosen = min(scored)[1]
        else:
            raise ValueError("unknown retrieval rule {!r}".format(self.retrieval_rule))
        # ``learned`` is absent on purpose: it draws from a distribution on every
        # call, so a repeat already differs, and re-drawing uniformly on top of
        # it would throw the join away on 60% of segments.
        # SEAM-AWARE PICK.  Both branches below choose from the SAME duration
        # band ``near``; they differ only in how they choose, so the length
        # guarantee P0 established is untouched either way.
        seam_aware = join_tail is not None and self.retrieval_rule in (
            "duration", "tempo", "medoid", "phase")
        if (occurrence or seam_aware) and self.retrieval_rule in (
                "duration", "tempo", "medoid", "phase"):
            if self.retrieval_rule != "phase":
                lengths = [abs((c[2] - c[1]) - target_length) for c in candidates]
                slack = max(2.0, 0.15 * target_length)
                near = [c for c, d in zip(candidates, lengths) if d <= slack] or [chosen]
                # THE SELECTION FILTERS APPLY HERE TOO, and until 2026-09-13
                # they did not.  This branch rebuilt the band from the FULL
                # candidate list, so whatever ``_duration_pick`` had filtered --
                # --draft-feet-lead, --draft-beat-fit, --draft-feet-beat-lead --
                # was discarded, and ``chosen`` survived only as the thing to
                # avoid.  Measured on wild_v5:7650126416710192357:clip000: the
                # feet-beat-lead filter changed its pick in 11 of 12 slots while
                # the DRAFT came out byte-identical with the flag on and off.
                # The operator saw it first, from the video: "跟已有的 baseline
                # 很多是一模一样的".
                #
                # Applied as a narrowing of the band, not as a replacement for
                # the draw: seam-aware still ranks by join cost and still SAMPLES
                # from the top few, which is what keeps --retrieval-rule phase's
                # failure (a hard ranking that squeezed out variety) from
                # repeating.  The filters only say which candidates are eligible.
                narrowed = self._prefer_beat_span(near, target_length,
                                                  target_beat_period)
                narrowed = self._prefer_hop_guard(narrowed)
                narrowed = self._prefer_music_energy(narrowed, target_soft)
                narrowed = self._prefer_hold_by_music(narrowed, target_quiet)
                narrowed = self._prefer_feet_beat_lead(narrowed, target_length,
                                                       target_beats)
                narrowed = self._prefer_beat_fit(narrowed, target_length,
                                                 target_beats)
                narrowed = self._prefer_feet_lead(narrowed)
                near = narrowed or near
            # For "phase", ``near`` is already the phase-equivalent set built
            # above: drawing from the duration band again would throw the
            # alignment away on every repeat, which is 60% of all segments.
            if len(near) > 1:
                if seam_aware:
                    # Rank the band by how well each candidate CONTINUES what is
                    # already on the floor, then DRAW FROM THE TOP FEW rather
                    # than taking the best.
                    #
                    # WHY NOT argmax.  That is what --retrieval-rule phase did,
                    # and it is why that rule failed: a hard sort narrows the
                    # pool to one answer per query and squeezes out the variety
                    # the recurrence fix had just bought back.  Measured here on
                    # 2026-09-05 with argmax: seam jerk 0.3515 -> 0.1756 (ground
                    # truth 0.2553, so it overshot into SMOOTHER than a real
                    # dancer) while adjacent-over-all-pairs pose distance fell
                    # 1.013 -> 0.957 against ground truth's 0.992 -- the
                    # diversity regression the top-k draw exists to prevent.
                    ranked = sorted(near, key=lambda c: (
                        self._join_cost(c, join_tail, target_length)
                        + self.phase_weight * self._phase_cost(c, target_phase)
                        + self.rhythm_weight * self._rhythm_cost(c, target_rhythm)))
                    # Prefer a recording this clip has NOT played yet, ahead of
                    # both the triple filter and the join ranking.  Applied on
                    # every unit, not only on repeat occurrences: the same
                    # performance showing up under two different class labels is
                    # exactly as visible as it is under one.
                    pool = ([c for c in ranked if c != chosen] if occurrence
                            else list(ranked))
                    unheard = [c for c in pool
                               if c[3] not in self._used_groups_this_clip]
                    if occurrence:
                        # Exclude the first occurrence's pick AND everything this
                        # clip has already played -- the same rule the plain
                        # variety draw uses.  Putting it only there was not
                        # enough: this branch is the one --draft-seam-aware-
                        # retrieval takes, and with it on, occurrence 4 of
                        # wild_v5:7608191311518369137:clip000 still drew
                        # occurrence 1's (2972, 0, 67) and occurrence 5 drew
                        # occurrence 2's, unchanged to the byte (2026-09-06).
                        fresh = [c for c in ranked
                                 if c != chosen and c not in self._used_this_clip]
                        ranked = unheard or fresh or pool or ranked
                    elif unheard:
                        ranked = unheard
                    top = ranked[:max(1, int(self.join_top_k))]
                    generator = (variety_rng if variety_rng is not None
                                 else np.random.default_rng(0))
                    chosen = top[int(generator.integers(len(top)))]
                else:
                    generator = variety_rng if variety_rng is not None else np.random.default_rng(0)
                    # Never the first occurrence's pick, AND never anything this
                    # clip has already played.
                    #
                    # Excluding only the first pick was not enough: measured
                    # 2026-09-06 on wild_v5:7608191311518369137:clip000, whose
                    # plan names just two classes over 20 s, occurrence 4 drew
                    # (2972, 0, 67) -- occurrence 1's pick -- and occurrence 5
                    # drew (2973, 0, 52), occurrence 2's.  Two of its ten units
                    # were byte-identical replays, 20% reuse, the highest of the
                    # twenty eval clips, and the operator picked it out by eye as
                    # 动作段落高度重复.  Falls back to the band when everything in
                    # it has been used: a repeat beats an empty slot.
                    # Same three-tier preference as the seam-aware branch:
                    # an unplayed RECORDING first, then merely an unplayed
                    # triple, then anything but the first pick.
                    unplayed_group = [c for c in near
                                      if c != chosen
                                      and c[3] not in self._used_groups_this_clip]
                    alternatives = unplayed_group or [
                        c for c in near
                        if c != chosen and c not in self._used_this_clip]
                    if not alternatives:
                        alternatives = [c for c in near if c != chosen] or near
                    chosen = alternatives[int(generator.integers(len(alternatives)))]
        chosen, phase_shift = self._rhythm_phase(chosen, slot_music)
        self._used_this_clip.add(chosen)
        self._used_groups_this_clip.add(chosen[3])
        native = int(chosen[2] - chosen[1])
        pool_max = max((int(c[2] - c[1]) for c in candidates), default=native)
        record = {
            "label": int(label),
            "slot_frames": int(target_length),
            "native_frames": native,
            # >1 means the prototype was slowed down to fill the slot.  The
            # playback multiplier the eye sees is 1/stretch.
            "stretch": float(target_length) / max(native, 1),
            "pool": len(candidates),
            "pool_max_frames": pool_max,
            # WHICH prototype, so a repeat is countable offline.  Without this
            # the artifact records how far a slot was stretched but not what
            # filled it, and "the same movement three times in a row" is
            # invisible to every reader -- the same instrument gap that let the
            # 2026-09-05 slow-motion defect ship.  (sample, start, end) is the
            # candidate's identity in the library index; the retrieval group is
            # dropped because it is provenance, not identity.
            "source": [int(chosen[0]), int(chosen[1]), int(chosen[2])],
            "occurrence": int(occurrence),
            # WHERE this unit starts in the clip.  Recorded rather than left to
            # the reader because the obvious reconstruction is WRONG: filler
            # spans are never retrieved and so never appear in this log, so
            # cumulating slot_frames drifts earlier by the total filler length
            # after the clip's first filler span.  A seam measurement built on
            # that reconstruction is measuring the wrong frames on exactly the
            # clips that have the most filler.
            "slot_start": None if slot_start is None else int(slot_start),
            # The distinguishing flag: this unit is not merely stretched, it is
            # UNFILLABLE -- the class has nothing long enough, so no tie-break,
            # energy floor or variety draw could have avoided the slow motion.
            "slot_exceeds_pool_max": bool(int(target_length) > pool_max),
            "rhythm_shift": int(phase_shift),
        }
        self.retrieval_log.append(record)
        values = self._values_at(chosen, target_length)
        if self._local_floor is not None:
            # Units LEVELLED, counted where the chosen unit is laid down; the
            # floor cache also fills from join-band and selector lookups.
            self.unit_floor_placed = getattr(self, "unit_floor_placed", 0) + 1
        if key is not None:
            self._retrieval_cache[key] = values
            self._retrieval_record_cache[key] = record
        return values.clone()

    CONTINUE_EASE_FRAMES = 8
    # where the settle dip sits: 3 frames after the bar line (~0.2 beat; the train split's beat-to-settle lag
    # is median 0.200 of a beat, docs 52) and how wide it is each side
    SETTLE_PEAK = 3
    SETTLE_WIDTH = 6

    def _exit_rate(self, previous):
        """Source frames per slot frame at the END of the unit just laid down, as
        ``_values_at``/``_values_continued`` played it (align_corners: (n-1)/(L-1))."""
        record = self.retrieval_log[-1] if self.retrieval_log else None
        if record is None:
            return 1.0
        if "exit_rate" in record:
            return float(record["exit_rate"])
        native, slot = int(record["native_frames"]), int(record["slot_frames"])
        return float(max(native - 1, 1)) / float(max(slot - 1, 1))

    def _values_continued(self, candidate, target_length, entry_rate):
        """The next source bar played so the SPEED does not step at the bar line.

        Each unit is linearly resampled into its own slot, so two consecutive bars of
        one dancer play at two different rates (stretch 0.87-1.15 per bar) and the
        continuation's bar line got a velocity step: measured 2026-09-23, seam jerk at
        continued seams 2.29/2.43x the clip median against ground truth's 1.54/1.60 at
        the same frames, while cut seams (which fade) read 1.40/1.50.  Here the bar
        starts one ``entry_rate`` step after the previous frame, eases from that rate to
        whatever rate still lands its LAST frame exactly on its own last source frame
        (so the next bar line stays exact), over ``CONTINUE_EASE_FRAMES``.  Floor as
        ``_values_at``; the result is logged as this unit's exit rate.
        """
        sample, start, end, _ = candidate
        source = torch.from_numpy(np.array(self.motion[sample, start:end], copy=True))
        native = len(source)
        length = int(target_length)
        if length < 2 or native < 2:
            return self._values_at(candidate, target_length)
        first = min(max(0.0, float(entry_rate) - 1.0), float(native - 1))
        span = float(native - 1) - first
        steps = length - 1
        ease = torch.ones(steps, dtype=torch.float64)
        k = min(self.CONTINUE_EASE_FRAMES, max(steps // 3, 1))
        ramp = 0.5 - 0.5 * torch.cos(math.pi * (torch.arange(k, dtype=torch.float64) + 0.5) / k)
        ease[:k] = ramp
        # sum_i [entry + (r - entry) * ease_i] = span  ->  r
        denominator = float(ease.sum())
        rate = (span - float(entry_rate) * (steps - denominator)) / max(denominator, 1e-9)
        if rate <= 0:
            return self._values_at(candidate, target_length)
        rates = float(entry_rate) + (rate - float(entry_rate)) * ease
        settle = float(getattr(self, "continue_settle", 0.0) or 0.0)
        if settle:
            index = torch.arange(steps, dtype=torch.float64)
            bump = 0.5 + 0.5 * torch.cos(math.pi * torch.clamp(
                (index - self.SETTLE_PEAK).abs() / self.SETTLE_WIDTH, max=1.0))
            rates = rates * (1.0 - settle * bump)
            rates = rates * (span / max(float(rates.sum()), 1e-9))   # still ends on its own last frame
        positions = torch.cat([torch.tensor([first], dtype=torch.float64),
                               first + torch.cumsum(rates, 0)]).clamp(0, native - 1)
        low = positions.floor().long().clamp(max=native - 2)
        weight = (positions - low.double()).unsqueeze(1).to(source.dtype)
        values = source[low] * (1 - weight) + source[low + 1] * weight
        floor = self._candidate_floor(candidate)
        if floor is not None:
            scale, _offset = self._normalizer_affine()
            column = ROOT_POSITION_START + 2
            values[:, column] -= floor / float(scale[column])
        self.retrieval_log[-1]["exit_rate"] = float(rates[-1])
        self.retrieval_log[-1]["entry_rate"] = float(entry_rate)
        return values

    def _continue_into(self, candidate, previous, target_length, occurrence, slot_start):
        """Lay down ``candidate`` -- the previous unit's own next source bar -- as this
        slot's unit.  Logged like a retrieval (``continued: True``) so the count gate
        and units_detail see it, and levelled on the PREVIOUS unit's floor so the
        dancer's own height carries across the bar line instead of re-flooring."""
        entry_rate = self._exit_rate(previous)   # read BEFORE this unit's record is appended
        self._used_this_clip.add(candidate)
        self._used_groups_this_clip.add(candidate[3])
        native = int(candidate[2] - candidate[1])
        self.retrieval_log.append({
            "label": int(np.asarray(self.labels[candidate[0]])[candidate[1]]),
            "slot_frames": int(target_length),
            "native_frames": native,
            "stretch": float(target_length) / max(native, 1),
            "pool": 1,
            "pool_max_frames": native,
            "source": [int(candidate[0]), int(candidate[1]), int(candidate[2])],
            "occurrence": int(occurrence),
            "slot_start": int(slot_start),
            "slot_exceeds_pool_max": False,
            "continued": True,
            # a continued bar carries its run's phase shift, so the dancer's own continuity survives
            "rhythm_shift": int(self.retrieval_log[-1].get("rhythm_shift", 0)) if self.retrieval_log else 0,
        })
        self.continue_applied = getattr(self, "continue_applied", 0) + 1
        values = self._values_continued(candidate, target_length, entry_rate)
        floor_new, floor_prev = self._candidate_floor(candidate), self._candidate_floor(previous)
        if floor_new is not None and floor_prev is not None:
            scale, _offset = self._normalizer_affine()
            column = ROOT_POSITION_START + 2
            values[:, column] += (floor_new - floor_prev) / float(scale[column])
        if self._local_floor is not None:
            self.unit_floor_placed = getattr(self, "unit_floor_placed", 0) + 1
        return values

    def _return_into(self, candidate, target_length, occurrence, slot_start, returned_from_bar, lag):
        """Lay down ``candidate`` -- the unit this clip already played ``lag`` bars ago -- again in this slot,
        restretched like any retrieval.  Logged with ``returned_from_bar`` so a return is countable offline."""
        native = int(candidate[2] - candidate[1])
        self.retrieval_log.append({
            "label": int(np.asarray(self.labels[candidate[0]])[candidate[1]]),
            "slot_frames": int(target_length),
            "native_frames": native,
            "stretch": float(target_length) / max(native, 1),
            "pool": 1,
            "pool_max_frames": native,
            "source": [int(candidate[0]), int(candidate[1]), int(candidate[2])],
            "occurrence": int(occurrence),
            "slot_start": int(slot_start),
            "slot_exceeds_pool_max": False,
            "returned_from_bar": int(returned_from_bar),
            "return_lag": int(lag),
            "rhythm_shift": 0,
        })
        self.motif_returned = getattr(self, "motif_returned", 0) + 1
        values = self._values_at(candidate, target_length)
        if self._local_floor is not None:
            self.unit_floor_placed = getattr(self, "unit_floor_placed", 0) + 1
        return values

    def _motif_candidate(self, bar, segment, placed_bars, returned_sources):
        """(candidate, source bar, lag) to play again at ``bar``, or None.  A source qualifies if it was retrieved or
        continued (never itself a return), has not been returned before, and its native length fits the slot inside
        the duration band every rule here uses."""
        slack = max(2.0, 0.15 * segment.length)
        for lag in self.motif_lags:
            placed = placed_bars.get(bar - lag)
            if placed is None:
                continue
            candidate, was_return = placed
            if was_return or candidate[:3] in returned_sources:
                continue
            if int(np.asarray(self.labels[candidate[0]])[candidate[1]]) == 0:
                continue                      # label 0 is filler (a transition), not a move worth bringing back
            if abs((candidate[2] - candidate[1]) - segment.length) > slack:
                continue
            if getattr(self, "motif_pick", "first") == "lively":
                placed_energy = [self._unit_energy(c) for c, _ in placed_bars.values()]
                if self._unit_energy(candidate) < float(np.median(placed_energy)):
                    self.motif_quiet_skipped = getattr(self, "motif_quiet_skipped", 0) + 1
                    continue
            return candidate, bar - lag, lag
        return None

    def _unit_energy(self, candidate):
        """Mean body-frame joint speed of a library unit (its own frames, decoded once, cached): how much it MOVES."""
        key = tuple(candidate[:3])
        if key not in self._unit_energy_cache:
            block = torch.from_numpy(np.array(self.motion[candidate[0], candidate[1]:candidate[2]], dtype=np.float32))
            j = np.asarray(decode_motion(block, self.normalizer_path)["full_pose"], dtype=np.float64)
            rel = j - j[:, :1]
            self._unit_energy_cache[key] = float(np.linalg.norm(np.diff(rel, axis=0), axis=-1).mean()) if len(j) > 1 else 0.0
        return self._unit_energy_cache[key]

    def build_draft(
        self,
        labels,
        feature_dim,
        *,
        exclude_retrieval_group_ids=(),
        allow_missing=False,
        root_continuity="off",
        gap_fill="zero",
        seam_blend=0,
        seam_stagger=False,
        seam_window="triangle",
        seam_transition="centred",
        seam_aware_retrieval=False,
        root_velocity_blend=0,
        root_seam_smooth=0,
        facing_anchor=0.0,
        target_period=None,
        recurrence_variety=False,
        variety_rng=None,
        facing_continuity=False,
        beat_grid=None,
        music=None,
        bar_bounds=None,
        beat_anchor=0.0,
        beat_anchor_per_limb=False,
        music_anchor=0.0,
        music_anchor_lag=0.20,
        rhythm_weight=0.0,
        music_anchor_shuffle=0,
        lower_body_delay=0.0,
        seam_lead=0.0,
        continue_phrase=0,
        motif_at_phrase=False,
        continue_no_replay=False,
        continue_phrase_novelty=0.0,
        phrase_rhythm_keep=0.0,
        tempo_keep=0.0,
        phrase_chain=False,
        step_lock_keep=0.0,
        energy_follow=None,
    ):
        """Build a prototype draft, optionally excluding query-source motion.

        ``allow_missing`` leaves a segment unconditioned when no verifiably
        external candidate exists.  It never falls back to same-source or
        unknown-provenance data.

        ``root_continuity`` decides what happens *between* two retrieved
        segments.  With ``"off"`` -- every artifact before 2026-08-17 -- each
        prototype is pasted at the absolute root position it had in its own
        recording, so the draft teleports the dancer at every segment boundary:
        the 151-D vector carries root position in dimensions 4:7 (see
        ``decode_motion``), and two prototypes cut from two different uploads
        share no origin.  Measured on 24 held-out clips, that boundary is where
        the roughness lives -- jerk within +-2 frames of a boundary is 3.2x the
        rest of the clip on 24/24 clips and both seeds, and 17% of frames carry
        30% of the frames rough enough to read as a visible pop.

        ``"xy"`` translates each prototype so its first frame's floor position
        continues from the previous conditioned segment's last frame; ``"xyz"``
        does the same for height.  The translation is applied in *normalized*
        space, which is exact rather than approximate: the release normalizer is
        per-dimension min-max, so adding a constant to a normalized dimension is
        adding a constant to the raw one.  Height is a separate mode because
        offsetting it lets vertical error accumulate across a clip and walk the
        dancer off the floor, while the horizontal teleport has no such
        counterweight -- which of the two is worth it is a measurement, not a
        preference, so both are expressible.

        ``gap_fill`` decides what the draft holds on the frames the plan calls
        transition.  ``"zero"`` -- every artifact before 2026-08-17 -- leaves
        them at literal zero, and zero in this normalized space is not "no
        opinion": it is the midpoint of the training min-max range, i.e. one
        particular pose.  Roughly half of all generated frames are transition,
        so the draft steps between a retrieved prototype and that pose about
        seventeen times per clip.  Measured: jerk within +-2 frames of a plan
        boundary is 3.21x the rest of the clip, and burying the draft in noise
        (``--draft-noise-ratio 1.0``, which destroys its structure without
        removing it) drops that to 1.68x -- so the step really is what the
        completion model is reacting to.  ``"hold"`` carries the last retrieved
        value across the gap and ``"interpolate"`` ramps between the segments on
        either side.  The mask stays zero either way, so the model is still told
        those frames are not a retrieved prototype; only the discontinuity goes.

        ``seam_blend`` is the half-width, in frames, of a cross-fade between two
        prototypes that **touch** -- and until 2026-08-29 there was none at all,
        which the two knobs above cannot cover.  ``root_continuity`` moves the
        root and leaves every joint angle stepping; ``gap_fill`` only ever
        touches frames the plan called transition.  A prototype-to-prototype
        seam got neither, and on the bar-grid arm **90.9% of plan boundaries are
        exactly that kind**, because the grid emits almost no transition.

        What it costs and what it buys, measured 2026-08-29 on the shipped
        checkpoints: the draft's own step across such a seam is **14.87x** the
        clip's interior speed, and the completion model neither follows it nor
        smooths it -- it brakes to **0.15x for exactly one frame** and
        over-accelerates on both sides, landing the minimum-step frame at offset
        -1 in 881 of 888 boundaries (99.2%).  That one-frame freeze, repeated
        once per bar and placed on the downbeat by the grid, is what a reviewer
        reads as a judder and what the accent criterion was scoring as an
        on-beat hit.

        **Default 0, so every earlier artifact reproduces from its own command
        line.**  A blend is a change to the retrieval evidence, not a free
        smoothing: it invents frames no single prototype contains, so it has to
        be asked for, and the width has to be small enough that it removes the
        step rather than the movement.  Nothing outside +-``seam_blend`` frames
        of a seam is touched, and the mask is left alone -- the model is still
        told those frames are retrieved.
        """
        if seam_blend < 0:
            raise ValueError("seam_blend must be >= 0, got {!r}".format(seam_blend))
        if gap_fill not in ("zero", "hold", "interpolate"):
            raise ValueError(
                "gap_fill must be zero, hold or interpolate, got {!r}".format(gap_fill)
            )
        if root_continuity not in ("off", "xy", "xyz"):
            raise ValueError(
                "root_continuity must be off, xy or xyz, got {!r}".format(root_continuity)
            )
        if root_seam_smooth and root_continuity == "off":
            raise ValueError(
                "--draft-root-seam-smooth needs --draft-root-continuity xy or xyz: it "
                "takes the chained root columns out of the seam blend, and unchained "
                "prototypes would then meet with a raw jump")

        continuity_dims = {
            "off": (),
            "xy": (ROOT_POSITION_START, ROOT_POSITION_START + 1),
            "xyz": (ROOT_POSITION_START, ROOT_POSITION_START + 1, ROOT_POSITION_START + 2),
        }[root_continuity]
        if continuity_dims and feature_dim <= max(continuity_dims):
            raise ValueError(
                "root continuity needs a motion vector carrying root position at {}, "
                "got feature_dim {}".format(list(continuity_dims), feature_dim)
            )
        excluded = (
            (exclude_retrieval_group_ids,)
            if isinstance(exclude_retrieval_group_ids, str)
            else tuple(exclude_retrieval_group_ids)
        )
        draft = torch.zeros(len(labels), feature_dim, dtype=torch.float32)
        mask = torch.zeros(len(labels), 1, dtype=torch.float32)
        # The last conditioned frame's root, carried across the unconditioned
        # transition gaps as well: a gap means the completion model is free
        # there, not that the dancer teleported over it.
        previous_root = None
        previous_root_velocity = None
        # The last conditioned frame's FACING, carried the same way.  Without
        # it two consecutive prototypes arrive with the facings of the two
        # different recordings they were cut from, and the draft demands up to
        # a half turn inside one frame.  Measured on the shipped draft (97
        # clips, threshold = ground truth's own pooled p99.5 turn rate of 549
        # deg/s): 92.9% of the draft's over-threshold frames sit on a plan
        # boundary, 11.84x their share of frames, while AWAY from boundaries the
        # draft turns 57.8 deg/s against ground truth's 62.8 -- so the library's
        # own content turns LESS than a real dancer and every violent turn is
        # manufactured at the paste.  The completion does not remove it, it
        # spreads it: boundary concentration falls to 0.69 while the number of
        # clips carrying at least one such frame rises from ground truth's
        # 21/97 to 55/97.
        previous_yaw = None
        opening_yaw = None
        # The last CONDITIONED frames as they were actually written into the
        # draft, in raw units, and where that segment ended.  Only the learned
        # rule reads them, and only because it scores the join: every other
        # rule here is a function of the candidate alone.
        previous_tail = None
        previous_end = None
        # --draft-continue-source: the unit just laid down, and where its slot ended,
        # so the next slot can take that dancer's own next bar instead of a cut.
        previous_unit = None
        previous_unit_end = None
        continued_starts = set()
        continued_run = 0
        # --draft-seam-lead: where each unit's seam actually is (its slot start minus the lead it took), where the
        # last unit was laid down, and that unit's per-frame facing, so a led unit can continue from the frame
        # before ITS seam rather than from the end of a unit it has just overwritten.
        seam_at = {}
        placed_span = None
        previous_yaw_track = None
        # --draft-continue-no-replay: every source span this clip has played, in its recording's own frames
        played_spans = []
        seen = {}
        self._used_this_clip = set()
        self._used_groups_this_clip = set()
        # --draft-motif-return: which unit each full bar played, what has been returned, and a draw per clip that
        # does not touch any other stream (seeded from the plan, so a rerun of the same plan returns the same bars)
        placed_bars = {}
        returned_sources = set()
        returns_here = 0
        motif_rng = (np.random.default_rng([int(np.asarray(labels).sum()) % (2 ** 31), int(np.asarray(labels).size)])
                     if self.motif_return else None)
        # ONE BEAT, derived from the bar lines themselves rather than passed
        # in, so the two can never disagree.  A bar line landing a few frames
        # from a label change would otherwise carve off a sliver and send
        # retrieval hunting the class for something shorter than anything the
        # corpus contains -- the T vocabulary's shortest movement is a whole
        # 4-beat bar (1.23 s).  On BAR ep160 this removes 2 of 174 units.  It
        # does NOT remove that arm's worst stretch (1.400): all 6 of its
        # sub-beat units start at frame 0 and are the partial bar at the clip
        # head, bounded by a real label change rather than by a line added
        # here.  See labels_to_segments for why the fold must not touch those.
        min_fragment = None
        if bar_bounds is not None and len(bar_bounds) > 2:
            widths = [int(b) - int(a) for a, b in zip(bar_bounds[:-1], bar_bounds[1:])]
            widths = [w for w in widths if w > 0]
            if widths:
                bar = sorted(widths)[len(widths) // 2]
                min_fragment = max(2, bar // 4)
        segments = list(labels_to_segments(labels, split_at=bar_bounds,
                                           min_fragment=min_fragment))
        onset = _selector_onset(music)
        # WHICH BARS ARE QUIET, from the music alone: onset-peak density
        # (channel 33 per frame), ranked against THIS clip's own segments, so a
        # loud song and a soft one each get a quiet half.  Computed only when
        # the filter is on, so nothing else changes.
        segment_quiet = [None] * len(segments)
        if self.hold_by_music:
            track = np.asarray(music.cpu() if hasattr(music, "cpu") else music)
            if track.ndim == 2 and track.shape[1] > 33:
                density = [float(track[seg.start:max(seg.end, seg.start + 1), 33].sum())
                           / max(1, seg.end - seg.start) for seg in segments]
                middle = float(np.median(density))
                segment_quiet = [d < middle for d in density]
        # WHICH BARS ARE SOFT, for --draft-music-energy: mean loudness (MFCC c0)
        # against the LIBRARY's median, not this clip's, so a soft song is
        # mostly soft bars.  Music only.
        segment_soft = [None] * len(segments)
        if self.music_energy and self.quiet_loudness is not None:
            track = np.asarray(music.cpu() if hasattr(music, "cpu") else music)
            if track.ndim == 2 and track.shape[1] > self.LOUDNESS_CHANNEL:
                segment_soft = [
                    bool(float(track[seg.start:max(seg.end, seg.start + 1),
                                     self.LOUDNESS_CHANNEL].mean())
                         < self.quiet_loudness)
                    for seg in segments]
        # --draft-continue-phrase N: which bars START a phrase, from the query's music alone (see _phrase_phase)
        phrase_phase = (_phrase_phase(music, bar_bounds, int(continue_phrase))
                        if continue_phrase and bar_bounds is not None and music is not None else None)
        # --draft-continue-phrase-novelty Q: phrase starts where the music changes instead of every N bars
        novelty_starts = (_phrase_starts(music, bar_bounds, float(continue_phrase_novelty))
                          if continue_phrase_novelty and bar_bounds is not None and music is not None else None)
        self.phrase_rhythm_keep = float(phrase_rhythm_keep or 0.0)
        self.tempo_keep = float(tempo_keep or 0.0)
        self.phrase_chain = bool(phrase_chain)
        self.step_lock_keep = float(step_lock_keep or 0.0)
        self.energy_follow = _parse_energy_follow(energy_follow)
        self._energy_tau = None
        if self.energy_follow and bar_bounds is not None and music is not None:
            self._energy_tau = _energy_targets(np.asarray(music.cpu() if hasattr(music, "cpu") else music),
                                               [int(b) for b in bar_bounds], self.energy_follow)
        if novelty_starts is not None:
            self.continue_phrase_novelty_starts = getattr(self, "continue_phrase_novelty_starts", 0) + len(novelty_starts)
        if motif_at_phrase and phrase_phase is None:
            raise ValueError("--draft-motif-at-phrase needs --draft-continue-phrase (and a bar grid): it returns "
                             "only at phrase starts")
        if phrase_phase is not None:
            self.continue_phrase_phases = getattr(self, "continue_phrase_phases", []) + [int(phrase_phase)]
        for position, segment in enumerate(segments):
            if segment.label == 0 and not self.index_filler:
                # Bridged by gap_fill instead.  See the index builder for why
                # that is a defect and what indexing filler costs.
                continue
            occurrence = seen.get(int(segment.label), 0)
            seen[int(segment.label)] = occurrence + 1
            # WHERE IN THE BAR this segment starts, which is the whole point
            # of the phase rule: the same atomic movement wants a different
            # exemplar depending on whether it begins on the downbeat or half a
            # beat late.  ``None`` for every other rule, so nothing else moves.
            segment_phase, query_beat_period = _grid_phase(beat_grid, segment.start)
            context = None
            if self.retrieval_rule == "learned":
                context = self._query_context(
                    segments, position, segment, previous_tail, previous_end,
                    segment_phase, query_beat_period, onset)
            next_slot = None
            if self.continue_lookahead and position + 1 < len(segments):
                following = segments[position + 1]
                if following.start == segment.end and (following.label or self.index_filler):
                    next_slot = (int(following.label), int(following.length))
            # WHICH BAR this segment fills, if it fills one whole bar of the grid (a return needs a whole bar)
            bar_index = None
            if bar_bounds is not None:
                bounds = [int(b) for b in bar_bounds]
                k = bisect.bisect_right(bounds, int(segment.start)) - 1
                if 0 <= k < len(bounds) - 1 and bounds[k] == segment.start and bounds[k + 1] == segment.end:
                    bar_index = k
            phrase_start = (phrase_phase is not None and bar_index is not None
                            and (bar_index - phrase_phase) % int(continue_phrase) == 0)
            if novelty_starts is not None:
                phrase_start = bar_index is not None and bar_index in novelty_starts
            returned = None
            # --draft-motif-at-phrase: a return only where a phrase STARTS, so that with --draft-continue-phrase the
            # returned bar's own continuation replays the whole earlier phrase (A A') instead of a single bar landing
            # in the middle of a phrase and cutting its continuation.
            if self.motif_return and bar_index is not None and bar_index >= min(self.motif_lags) \
                    and returns_here < self.motif_max and (not motif_at_phrase or phrase_start):
                self.motif_slots = getattr(self, "motif_slots", 0) + 1
                if motif_rng.random() < self.motif_return:
                    returned = self._motif_candidate(bar_index, segment, placed_bars, returned_sources)
                    if returned is None:
                        self.motif_unfilled = getattr(self, "motif_unfilled", 0) + 1
            continued_from = None
            # --draft-phrase-rhythm-keep: at a phrase start, the query bars the phrase will cover after this one
            phrase_music = phrase_lengths = None
            if phrase_start and (self.phrase_rhythm_keep or self.phrase_chain or self.step_lock_keep
                                 or self.energy_follow) \
                    and bar_bounds is not None \
                    and music is not None:
                track = np.asarray(music.cpu() if hasattr(music, "cpu") else music)
                bnds = [int(b) for b in bar_bounds]
                limit = int(self.continue_max_run) if self.continue_max_run else 8
                phrase_music, phrase_lengths, k = [], [], bar_index + 1
                while k < len(bnds) - 1 and len(phrase_music) < limit:
                    if novelty_starts is not None:
                        if k in novelty_starts:
                            break
                    elif phrase_phase is not None and (k - phrase_phase) % int(continue_phrase) == 0:
                        break
                    phrase_music.append(track[bnds[k]:bnds[k + 1]])
                    phrase_lengths.append(bnds[k + 1] - bnds[k])
                    k += 1
            if phrase_start and self._source_bars is not None and previous_unit is not None:
                self.continue_phrase_cuts = getattr(self, "continue_phrase_cuts", 0) + 1
            if returned is None and self._source_bars is not None and previous_unit is not None \
                    and previous_unit_end == segment.start and not phrase_start \
                    and not (self.continue_max_run and continued_run >= self.continue_max_run):
                self.continue_eligible = getattr(self, "continue_eligible", 0) + 1
                run_shift = int(self.retrieval_log[-1].get("rhythm_shift", 0)) if self.retrieval_log else 0
                base = (previous_unit[0], previous_unit[1] - run_shift, previous_unit[2] - run_shift,
                        previous_unit[3])
                continued_from = self._next_source_bar(base, segment.label, segment.length)
                if continued_from is not None and self.continue_rhythm_min and music is not None:
                    track = np.asarray(music.cpu() if hasattr(music, "cpu") else music)
                    if not self._continuation_rhythm_ok(continued_from, segment.label, segment.length,
                                                        track[segment.start:segment.end], excluded):
                        continued_from = None
                if continued_from is not None and run_shift:
                    frames = int(np.asarray(self.labels).shape[1])
                    moved = (continued_from[0], continued_from[1] + run_shift,
                             continued_from[2] + run_shift, continued_from[3])
                    continued_from = moved if 0 <= moved[1] and moved[2] <= frames else None
                if continued_from is not None and continue_no_replay:
                    # NEVER CONTINUE INTO FRAMES THIS CLIP HAS ALREADY PLAYED.  A motif return re-places an earlier
                    # bar on purpose; continuing from it then walks that dancer's NEXT bar, which this clip has also
                    # played -- so a one-bar callback became a verbatim multi-bar replay right after the original
                    # (A B A B).  The operator on 日不落 F5: "有重复段落".  Measured on 59 clips: F5 23 consecutive
                    # replayed bars against E11's 2, F6 (no returns) 0.  Refused here, the slot is retrieved afresh
                    # (which already excludes played units), so the return stays one move coming back.
                    span = self._source_span(continued_from)
                    if span is not None and any(
                            span[0] == other[0] and min(span[2], other[2]) - max(span[1], other[1])
                            > 0.5 * (span[2] - span[1]) for other in played_spans):
                        continued_from = None
                        self.continue_replay_refused = getattr(self, "continue_replay_refused", 0) + 1
            try:
                if returned is not None:
                    values = self._return_into(returned[0], segment.length, occurrence, segment.start,
                                               returned[1], returned[2])
                    returned_sources.add(tuple(returned[0][:3]))
                    returns_here += 1
                    continued_run = 0
                elif continued_from is not None:
                    values = self._continue_into(continued_from, previous_unit, segment.length,
                                                 occurrence, segment.start)
                    continued_starts.add(int(segment.start))
                    continued_run += 1
                else:
                    continued_run = 0
                    # --draft-continuity-*: what the draft has ALREADY placed before this slot (generated motion)
                    continuity_context = (
                        self._draft_context(draft, mask, segment.start, segment.length,
                                            gap=int(self._continuity_model()[1].get("seam_mask", 6)))
                        if getattr(self, "continuity_scorer_path", None) and segment.start > 0 else None)
                    values = self.retrieve(
                        segment.label,
                        segment.length,
                        exclude_retrieval_group_ids=excluded,
                        target_period=target_period,
                        occurrence=(occurrence if recurrence_variety else 0),
                        variety_rng=variety_rng,
                        target_phase=segment_phase,
                        target_rhythm=_bar_rhythm_pattern(
                            music, segment.start, segment.end,
                            IndexedAtomicMotionLibrary.RHYTHM_BINS)
                        if rhythm_weight else None,
                        target_beat_period=query_beat_period,
                        target_beats=_slot_beat_frames(music, segment.start, segment.end),
                        context=context,
                        slot_start=segment.start,
                        # Only when the caller asked for it, so every artifact made
                        # before 2026-09-05 reproduces from its own command line.
                        join_tail=(previous_tail if seam_aware_retrieval else None),
                        target_quiet=segment_quiet[position],
                        target_soft=segment_soft[position],
                        next_slot=next_slot,
                        slot_music=(None if music is None or not (self.rhythm_keep or self.rhythm_shift) else
                                    np.asarray(music.cpu() if hasattr(music, "cpu") else music)
                                    [segment.start:segment.end]),
                        continuity_context=continuity_context,
                        phrase_music=phrase_music,
                        phrase_lengths=phrase_lengths,
                    )
            except KeyError:
                if allow_missing:
                    continue
                raise
            if self.retrieval_log:
                last = self.retrieval_log[-1]["source"]
                previous_unit = (int(last[0]), int(last[1]), int(last[2]),
                                 self.retrieval_group_ids[int(last[0])])
                previous_unit_end = segment.end
                if bar_index is not None:
                    placed_bars[bar_index] = (previous_unit, returned is not None)
                span = self._source_span(previous_unit)
                if span is not None:
                    played_spans.append(span)
            if values.shape[1] != feature_dim:
                raise ValueError("prototype feature dimension does not match requested draft")
            # --draft-seam-lead BEATS: MOVE THE CUT OFF THE DOWNBEAT.  A unit is a whole bar of its own dancer, so
            # its first frame is that dancer AT their downbeat and the approach into it lies in the frames before,
            # in the same recording.  Played from the bar line, every downbeat of the draft is a seam, and the
            # centred fade + completion inpaint regenerate +-8 frames around it: the one pose the bar is built to
            # hit is always a compromise between two unrelated dancers (fix7 final: wrist speed 1.31x median at -3
            # frames where ground truth is 1.01x; DEFECTS 90).  Here the unit is read from LEAD beats earlier in
            # its own source and laid down LEAD beats before the bar line, so the downbeat -- the approach, the
            # arrival and what follows -- is one real dancer's, and the change of dancer happens on the "4-and",
            # mid-flight, where both sides are moving anyway.  '--seam-transition after' kept the arriving unit's
            # stop and then made the change in the 8 frames AFTER the bar line (a lurch right after every hit);
            # this is its mirror.  Continued bars keep their source's own bar line (nothing to cut).  Reads the
            # query's beat period only.
            lead = 0
            if (seam_lead and continued_from is None and placed_span is not None
                    and placed_span[1] == segment.start and query_beat_period
                    and query_beat_period == query_beat_period and self.retrieval_log):
                want = int(round(float(seam_lead) * float(query_beat_period)))
                # the unit being cut into keeps at least half of its slot, and so does this one
                want = min(want, (placed_span[1] - placed_span[0]) // 2, int(segment.length) // 2)
                source = self.retrieval_log[-1]["source"]
                candidate = (int(source[0]), int(source[1]), int(source[2]),
                             self.retrieval_group_ids[int(source[0])])
                extended, lead = self._values_with_lead(candidate, int(segment.length), want)
                if lead:
                    values = extended
                    self.seam_lead_applied = getattr(self, "seam_lead_applied", 0) + 1
                    self.retrieval_log[-1]["seam_lead"] = int(lead)
                elif want > 0:
                    self.seam_lead_short = getattr(self, "seam_lead_short", 0) + 1
            seam_at[int(segment.start)] = int(segment.start) - lead
            if lead and os.environ.get("ATOMICDANCE_SEAM_LEAD_WINDOW_ONLY"):
                # DIAGNOSTIC CONTROL (not a mode): move the seams -- fade, root smoothing, completion inpaint, skate
                # mask -- exactly as the lead does, but keep the draft CONTENT as it was (the arriving unit up to
                # the bar line, this one from it).  If the arrival readings improve as much as with the lead, what
                # bought them was moving the regenerated window off the downbeat, not the incoming dancer's own
                # approach.  Recorded in the manifest (draft_seam_lead_window_only).
                values = values[lead:]
                lead = 0
                self.seam_lead_window_only = True
            # WHICH ANCHORS.  ``--draft-beat-anchor`` targets the BEAT grid
            # and it made every reading worse, for a reason the absolute-m/s
            # profile names: our settle already sits just BEFORE the beat, so
            # pulling settle points onto beats pushes early points earlier.
            # ``--draft-music-anchor`` targets channel 33, the music's ONSET
            # PEAKS -- 17.2% of frames against the beat channel's 6.6%, i.e. the
            # song's own accent pattern rather than its metronome.  That is the
            # one thing in this pipeline that can make the timing depend on
            # WHICH SONG is playing: the planner picks a label (what to dance),
            # the grid puts the bar lines on beats (where the boundaries are),
            # and the uniform resample leaves the accents wherever the stretch
            # happened to drop them -- the same relative positions whatever the
            # music (docs 52).
            #
            # ONLY THE MUSIC IS READ.  Channel 33 comes from the query's audio,
            # which is the task's input; the target clip's MOTION is never
            # touched here.  CLAUDE.md 1.6 fixes that line.
            anchor_strength, anchor_grid = beat_anchor, beat_grid
            if music_anchor and beat_grid is not None:
                # THE TARGET IS THE BEAT PLUS A LAG, and both halves are
                # measured on the RELEASE'S TRAIN SPLIT, never on the clips
                # this is evaluated with.
                #
                # NOT THE ONSET PEAKS.  The first version of this aimed the
                # prototype's settle points at channel 33 and that is the wrong
                # target: speed at accent frames over speed elsewhere reads
                # **1.0333 for ground truth** -- a real dancer moves FASTER on
                # an accent, because an accent is a hit that the dancer strikes
                # into.  The shipped arm already read 0.9793 and anchoring to
                # the accent took it to 0.9578, further from ground truth.  And
                # there is no consistent lag to correct it with: over 8,747
                # accent-to-settle pairs in the train split the lag is nearly
                # uniform (p25 0.25, median 0.71, p75 1.00 of the interval).
                #
                # THE BEAT GRID DOES HAVE ONE.  Over 2,782 beat-to-settle pairs
                # in the same split the lag is **median 0.200 of a beat**, p25
                # 0.071, p75 0.385 -- an interquartile range less than half the
                # accent version's.  It also agrees with the independent
                # absolute-m/s reading on the eval clips (trough at phase 0.12).
                # That 0.20 is exactly what ``--draft-beat-anchor`` was missing:
                # it aimed at beat + 0 while ground truth settles at beat +
                # 0.20, so it pulled settle points that were ALREADY early
                # further forward, which is why every strength of it read worse.
                grid = np.asarray(beat_grid, dtype=np.int64)
                period = float(np.median(np.diff(grid))) if len(grid) > 2 else 0.0
                peaks = grid + int(round(music_anchor_lag * period))
                if music_anchor_shuffle:
                    # THE CONTROL THIS CHANGE MUST BRING.  A monotone warp also
                    # SMOOTHS, so an arm that improved could have bought the
                    # smoothing rather than the alignment.  Shuffling keeps the
                    # count and the range of the accent times and destroys only
                    # their correspondence with the music; if the reading does
                    # not fall, nothing was aligned.
                    rng = np.random.default_rng(int(music_anchor_shuffle))
                    span = int(grid.max()) + 1
                    peaks = np.sort(rng.choice(span, size=len(peaks), replace=False))
                anchor_strength, anchor_grid = music_anchor, peaks
            if anchor_strength and anchor_grid is not None:
                # PUT THE PROTOTYPE'S OWN SETTLE POINTS ON THE QUERY'S BEATS.
                #
                # WHY.  The operator, 2026-09-07: "动作卡点还是不如 gt,虽然动作
                # 节奏有,但是不够舒展到位".  The measurement that says the same
                # thing is ``settle`` (how much the motion slows INTO the beat):
                # ground truth +0.0569, the shipped arm -0.0769.  Ground truth
                # arrives at a shape and rests on the beat; the generated arm is
                # still accelerating through it.  That is what "有节奏但不到位"
                # is -- the accents exist but they do not LAND.
                #
                # WHERE IT COMES FROM.  ``_values_at`` fills a slot with ONE
                # global linear resample, so wherever the prototype's own
                # accents happened to sit, the stretch drags them off the beat.
                # This warp is monotone, length-preserving and stretch-capped,
                # so it moves the accents WITHOUT changing which frames the slot
                # owns or how long it plays.
                #
                # ANCHORS HAVE PROVENANCE, they are not invented here:
                # ``motion_accent_frames`` is the paper's own motion beat
                # ("local minima of segment-wise joint velocities", atomicDance
                # 3.2), and the target is the music's beat grid the bar planner
                # already cuts on.  ``warp_to_anchors``' own docstring names
                # this exact use -- "the music's beat frames at inference" --
                # and until now nothing outside the tests called it.
                #
                # BOTH ANCHOR SETS ARE SLOT-LOCAL.  The source anchors are read
                # off ``values`` AFTER the resample, so they are already in the
                # slot's frame numbering, and the targets are shifted by
                # ``segment.start``.  Reading the source anchors off the
                # unresampled prototype instead would mix two coordinate systems
                # and put every accent a stretch-factor away from where it
                # claims to be -- the off-by-one class of defect CLAUDE.md 2.2
                # records.
                grid = np.asarray(anchor_grid, dtype=np.int64)
                local = grid[(grid > segment.start - lead) & (grid < segment.end)]
                local = local - (int(segment.start) - lead)
                if len(local):
                    if beat_anchor_per_limb:
                        # EACH LIMB ON ITS OWN ACCENTS.  Warping every channel
                        # together lands the phase and brakes the whole body on
                        # one frame: measured on the 20 eval clips, per-part
                        # settle spread falls from ground truth's 0.578 to 0.222
                        # and limb lockstep rises from 0.500 to 0.617.  The seam
                        # stagger fixed the second at the seam only (spread
                        # 0.697, lag0 0.492) and reaches nothing between seams.
                        # rot6d is per-joint LOCAL rotation, so a limb on its own
                        # timeline is a limb dancing to its own accent, not a
                        # broken skeleton.  The remainder -- root, contacts,
                        # spine, neck, head -- keeps one whole-vector warp, the
                        # same split ``_blend_draft_seams`` uses for the stagger.
                        limb_columns = {c for dims in LIMB_DIMS.values() for c in dims}
                        rest = tuple(c for c in range(values.shape[1])
                                     if c not in limb_columns)
                        if beat_anchor_per_limb == "all":
                            groups = list(LIMB_DIMS.values()) + [rest]
                        elif beat_anchor_per_limb == "feet":
                            # WHY A GROUP NARROWER THAN "legs".  Measured
                            # 2026-09-12 on the aligned base, "legs" warps hips
                            # (1,2), knees (4,5), ankles (7,8) and toes (10,11)
                            # on ONE timeline, so the hips settle exactly when
                            # the feet do: per-part settle went feet 0.0339 ->
                            # 0.0951 (ground truth 0.0943, right on it) but
                            # hips_knees 0.0093 -> 0.1731 against ground truth's
                            # 0.0649, overshooting by 2.7x and inverting the
                            # feet > hips_knees ordering that IS the dancer's
                            # shape.  The part-ordering rho fell monotonically
                            # with dose, +0.20 -> +0.05 -> -0.10 -> -0.30, while
                            # whole-body settle and depth both improved -- the
                            # saturated column buying a judgement the
                            # discriminating one refuses.
                            #
                            # A dancer lands the ankle and keeps travelling
                            # through the hip.  This group is the ankles and
                            # toes only; the hips and knees stay on the
                            # whole-body warp.  Contacts and root travel with
                            # the feet for the reason the "legs" branch records
                            # below: they say where the feet are and where the
                            # body is, and splitting them from the feet invented
                            # a slide (skate 0.420 -> 0.588).  ``skate`` is
                            # therefore a required column for this option, not
                            # an optional one.
                            lower = tuple(range(0, ROOT_POSITION_START
                                                + ROOT_POSITION_DIMS))
                            feet_columns = tuple(
                                CONTACT_CHANNELS + ROOT_POSITION_DIMS + 6 * joint + k
                                for joint in _FOOT_JOINTS for k in range(6))
                            # ONE group, and the rest of the body is left
                            # UNWARPED -- the same shape the "legs" branch uses.
                            # A first version appended an "everything else"
                            # group, which warped the whole body onto the same
                            # beats and lifted every part at once: hips_knees
                            # 0.0093 -> 0.1984, torso 0.1085 -> 0.2001,
                            # shoulders 0.1201 -> 0.2144, part spread collapsing
                            # 0.608 -> 0.219.  Warping only the ankles cannot
                            # move the shoulders, and that impossible reading is
                            # what exposed it.
                            groups = [feet_columns + lower]
                        elif beat_anchor_per_limb == "legs":
                            # GROUND TRUTH DOES NOT LAND EVERYTHING EQUALLY.  Its
                            # per-part settle is feet 0.0943, hips+knees 0.0649,
                            # torso 0.0477, hands 0.0233 -- the feet land four
                            # times harder than the hands, and that UNEVENNESS is
                            # the 0.578 spread the whole-body anchor flattens to
                            # 0.073.  Anchoring every group, even on its own
                            # source accents, flattens it just as badly (0.094):
                            # the targets are shared, so every limb is pulled
                            # onto the same beats and the body ends up MORE
                            # synchronised, not less.  Anchoring only what ground
                            # truth lands hardest leaves the arms on their own
                            # timeline, which is where the spread comes from.
                            # THE ROOT AND THE CONTACTS TRAVEL WITH THE LEGS.
                            # Warping the legs alone put the feet on a different
                            # clock from the translation that carries them, and
                            # foot skate went 0.420 -> 0.588 against ground
                            # truth's 0.295 -- a slide invented by the warp, not
                            # by the dance.  Contacts (0:4) and root position
                            # (4:7) say where the feet are and where the body
                            # is; they belong to the same timeline as the legs.
                            lower = tuple(range(0, ROOT_POSITION_START
                                                + ROOT_POSITION_DIMS))
                            groups = [tuple(LIMB_DIMS["left_leg"])
                                      + tuple(LIMB_DIMS["right_leg"]) + lower]
                        else:
                            groups = [rest]
                        for columns in groups:
                            values = warp_to_anchors(
                                values,
                                motion_accent_frames(values, columns=columns),
                                local,
                                max_stretch=float(anchor_strength),
                                columns=columns,
                            )
                    else:
                        values = warp_to_anchors(
                            values,
                            motion_accent_frames(values),
                            local,
                            max_stretch=float(anchor_strength),
                        )
            if facing_continuity:
                yaw, _ = self._facing_yaw(
                    values * self._normalizer_affine()[0] + self._normalizer_affine()[1]
                )
                correction = None
                if previous_yaw is not None:
                    # Wrapped to (-pi, pi]: aligning a facing must never choose
                    # the long way round, which would BE a spin.
                    # --draft-seam-lead: a led unit continues from the frame before ITS seam; the last ``lead``
                    # frames of the unit before are about to be overwritten.
                    reference_yaw = (previous_yaw_track[-(lead + 1)]
                                     if lead and previous_yaw_track is not None and len(previous_yaw_track) > lead
                                     else previous_yaw)
                    delta = torch.remainder(reference_yaw - yaw[0] + math.pi,
                                            2 * math.pi) - math.pi
                    if facing_anchor and opening_yaw is not None:
                        # CONTINUITY ALONE ACCUMULATES.  Aligning each segment to
                        # the previous one keeps the facing smooth and lets every
                        # prototype's own internal turn add up, so the dancer
                        # turns away and never comes back.  Measured 2026-09-06
                        # on the 20 eval clips: ground truth is 0.0% of frames
                        # more than 90 deg from its opening facing on 14 of 20,
                        # while three generated clips spend 77-92% of their
                        # frames facing away, the worst for 14.77 s with a net
                        # turn of 593 deg -- and the operator's words were
                        # "背朝相机舞姿段落太长" and "不知道观众在哪".
                        #
                        # So pull a fraction of the accumulated deviation out at
                        # every join.  The deviation then decays geometrically
                        # instead of integrating, while each segment keeps its
                        # own turning; 0 reproduces the published behaviour.
                        drift = torch.remainder(
                            reference_yaw + delta - opening_yaw + math.pi,
                            2 * math.pi) - math.pi
                        correction = float(facing_anchor) * drift
                    values = self._rotate_about_z(values, delta)
                    yaw = yaw + delta
                    if correction is not None and len(values) > 1:
                        # RAMPED, NOT STEPPED.  Taking the correction out of the
                        # alignment constant puts all of it in the seam's one
                        # frame: measured on 7030793823240424742:clip000 at
                        # t=12.47 s, drift -206.4 deg, the anchor's share 123.8,
                        # and 110.4 degrees of yaw IN A SINGLE FRAME against that
                        # clip's ground-truth maximum of 17.7.  The alignment
                        # above now lands the first frame exactly on the previous
                        # segment's facing -- so the seam is continuous -- and the
                        # correction is spread across the segment by a raised
                        # cosine, which is zero AND flat at the first frame, so
                        # neither the facing nor its rate steps at the join.
                        # The segment still ends fully corrected, so the decay
                        # the anchor exists for is unchanged.
                        ramp = 0.5 - 0.5 * torch.cos(
                            math.pi * torch.arange(len(values), dtype=torch.float32)
                            / (len(values) - 1))
                        deltas = -correction * ramp.to(dtype=values.dtype)
                        values = self._rotate_about_z_varying(values, deltas)
                        yaw = yaw + deltas
                        if os.environ.get("ATOMICDANCE_FACING_PROBE"):
                            import sys as _s
                            print("PROBE seg start={} len={} delta={:+.1f} drift={:+.1f} "
                                  "correction={:+.1f} peak_rate={:+.2f}".format(
                                      int(segment.start), len(values),
                                      math.degrees(float(delta)),
                                      math.degrees(float(drift)),
                                      math.degrees(float(correction)),
                                      math.degrees(float(correction)) * math.pi
                                      / (2 * max(len(values) - 1, 1))),
                                  file=_s.stderr)
                else:
                    opening_yaw = yaw[0].clone()
                previous_yaw = yaw[-1]
                previous_yaw_track = yaw
            if continuity_dims:
                columns = list(continuity_dims)
                if lead and segment.start - lead >= 2:
                    seam = int(segment.start) - lead
                    previous_root = draft[seam - 1, columns].clone()
                    previous_root_velocity = draft[seam - 1, columns] - draft[seam - 2, columns]
                if previous_root is not None:
                    values = values.clone()
                    values[:, columns] = _continue_root(
                        values[:, columns], previous_root, previous_root_velocity,
                        velocity_blend=root_velocity_blend)
                previous_root = values[-1, columns].clone()
                previous_root_velocity = (values[-1, columns] - values[-2, columns]
                                          if len(values) > 1 else None)
            draft[segment.start - lead : segment.end] = values
            mask[segment.start - lead : segment.end] = 1.0
            placed_span = (int(segment.start) - lead, int(segment.end))
            if self.retrieval_rule == "learned" or seam_aware_retrieval:
                from model.retrieval_selector import EDGE_FRAMES

                scale, offset = self._normalizer_affine()
                tail = values[max(len(values) - EDGE_FRAMES, 0):]
                if len(tail) < EDGE_FRAMES:
                    tail = values[torch.clamp(
                        torch.arange(EDGE_FRAMES) + len(values) - EDGE_FRAMES,
                        min=0)]
                previous_tail = tail * scale + offset
                previous_end = segment.end
        if gap_fill != "zero":
            _fill_draft_gaps(draft, mask, gap_fill)
        if root_seam_smooth and continuity_dims:
            # AFTER the retrieval loop, on purpose.  Done during chaining, any
            # change to the root reranks candidates, because _join_cost's pose
            # term compares ABSOLUTE root position: measured 2026-09-16, the
            # in-chain version and --draft-root-velocity-blend 8 alone each
            # swapped the moves on the same 2 of 10 clips.  Here every
            # prototype is already chosen, so the dance content is untouched and
            # only the root's path through each seam changes.
            _smooth_root_seams(draft, mask, labels, continuity_dims,
                               int(root_seam_smooth),
                               starts=[seam_at.get(int(segment.start), int(segment.start))
                                       for segment in segments])
        if seam_blend and seam_transition == "after":
            # --seam-transition after: the fade starts AT the bar line and runs
            # forward, from the arriving unit's last frame into the next unit's
            # own trajectory.  The centred fade below pulls the arriving unit's
            # last frames toward the next unit's first -- measured 2026-09-22 on
            # the ten vis clips, that is where the draft's arrival stop (wrist
            # speed 0.86x median at -4 frames) goes, and the left arm's stagger
            # centres its fade 5 frames BEFORE the bar line.  Every UNIT seam,
            # not only label changes: an in-run bar seam is the same jump, and
            # with no fade there the completion (which follows the draft) jumped
            # too -- the first version did that and read seam jerk 7.96 against
            # fix7's 1.27.  Root height rides along, so it cannot step (fix5).
            # A continued seam is the source dancer's own bar line: nothing to fade.
            _blend_draft_seams(draft, mask, labels, seam_blend, window=seam_window,
                               seams=[segment.start for segment in segments
                                      if self.continue_stop or segment.start not in continued_starts],
                               side="after",
                               skip_columns=(continuity_dims if root_seam_smooth else ()))
        elif seam_blend and seam_lead:
            # The same centred fade at the same seams (the label changes), each moved to where its unit actually
            # starts; a continued bar is its source dancer's own bar line and gets none.
            _blend_draft_seams(draft, mask, labels, seam_blend,
                               stagger=seam_stagger, window=seam_window,
                               seams=[seam_at.get(int(segment.start), int(segment.start))
                                      for segment in labels_to_segments(labels)
                                      if int(segment.start) not in continued_starts],
                               skip_columns=(continuity_dims if root_seam_smooth else ()))
        elif seam_blend:
            if continued_starts:
                raise ValueError("--draft-continue-source needs --seam-transition after: "
                                 "the centred fade has no per-seam exclusion")
            _blend_draft_seams(draft, mask, labels, seam_blend,
                               stagger=seam_stagger, window=seam_window,
                               skip_columns=(continuity_dims if root_seam_smooth else ()))
        if root_seam_smooth and continuity_dims:
            # The root channels continuity does NOT chain (z under xy), at the
            # bar seams that are not label changes -- the only seams the blend
            # above never reaches.  Height only: the pose at those seams is left
            # exactly as the shipped path builds it.
            unchained = tuple(
                c for c in range(ROOT_POSITION_START,
                                 ROOT_POSITION_START + ROOT_POSITION_DIMS)
                if c not in continuity_dims)
            label_starts = {segment.start for segment in labels_to_segments(labels)}
            in_run = [seam_at.get(int(segment.start), int(segment.start)) for segment in segments
                      if segment.start not in label_starts and segment.start not in continued_starts]
            if seam_blend and seam_transition == "after":
                # The forward fade above already carried the height across every
                # unit seam; a second, centred ramp here would pull the arriving
                # unit's pelvis toward the next unit's before the bar line.
                in_run = []
            # Counted, so a manifest can tell this build from the one before it:
            # the flag's value did not change when this block was added.
            self.root_seam_height_seams = (
                getattr(self, "root_seam_height_seams", 0) + len(in_run))
            if unchained and in_run:
                _blend_draft_seams(draft, mask, labels,
                                   seam_blend or int(root_seam_smooth),
                                   window=seam_window, seams=in_run,
                                   only_columns=unchained)
        if lower_body_delay and beat_grid is not None:
            _delay_lower_body(draft, beat_grid, lower_body_delay)
        # Every unit seam as laid down (led ones moved, continued ones absent): what the completion inpaints and
        # the skate fix masks when --draft-seam-lead is on, instead of the label changes and the bar lines.
        self.draft_seams = sorted({seam_at.get(int(segment.start), int(segment.start)) for segment in segments
                                   if int(segment.start) > 0 and int(segment.start) not in continued_starts})
        return draft, mask


def _delay_lower_body(draft, beat_grid, fraction):
    """Run the lower body a fraction of a beat LATER than the rest.

    WHY, measured 2026-09-11 in ABSOLUTE metres per second -- no z-scoring, the
    thing that made ``settle`` misleading.  Foot speed binned by position inside
    the beat, 20 eval clips:

        ground truth   0.823  0.756  0.802  0.865  0.868  0.874  0.826  0.828
        shipped arm    0.764  0.825  0.855  0.862  0.848  0.871  0.847  0.748

    The MODULATION DEPTH already matches -- ground truth peak/trough 1.156,
    ours 1.164.  What differs is WHERE the trough sits: ground truth's feet are
    slowest at phase 0.12, just AFTER the beat; ours at 0.88, just BEFORE it.
    We brake about a quarter of a beat early.

    That is why ``--draft-beat-anchor`` made things worse at every strength: it
    pulls the settle points ONTO the beat, and ours are already early, so it
    moves them further from where ground truth puts them.  The repair is a
    shift, not a warp.

    Only the legs, feet, contacts and root translation move; the arms keep their
    own timeline, because the same measurement shows the arms' beat-phase
    modulation is 2.6% in ground truth against our 4.4% -- near nothing in both,
    so there is nothing there to repair.
    """
    grid = np.asarray(beat_grid, dtype=np.int64)
    if len(grid) < 3:
        return
    period = float(np.median(np.diff(grid)))
    shift = int(round(period * float(fraction)))
    if shift <= 0:
        return
    columns = sorted({c for name in ("left_leg", "right_leg")
                      for c in LIMB_DIMS[name]}
                     | set(range(0, ROOT_POSITION_START + ROOT_POSITION_DIMS)))
    block = draft[:, columns].clone()
    # Hold the first frame rather than wrapping: a wrap would splice the end of
    # the clip onto its beginning, which is a cut no dancer performed.
    draft[shift:, columns] = block[:-shift]
    draft[:shift, columns] = block[:1]


# Which 151-D dimensions belong to which limb, for the staggered blend below.
# rot6d starts after 4 contact + 3 root dims, six numbers per SMPL joint.
_LIMB_JOINTS = {"left_arm": (16, 18, 20, 22), "right_arm": (17, 19, 21, 23),
                "left_leg": (1, 4, 7, 10), "right_leg": (2, 5, 8, 11)}
# Ankles and toes only -- the same joints tools/score_beat_phase_profile.py
# calls "feet", so the group that is warped and the column that judges it are
# the same joints rather than two nearby definitions.
_FOOT_JOINTS = (7, 8, 10, 11)
LIMB_DIMS = {name: tuple(CONTACT_CHANNELS + ROOT_POSITION_DIMS + 6 * joint + k
                         for joint in joints for k in range(6))
             for name, joints in _LIMB_JOINTS.items()}
# Fractions of the half-width by which each limb's cross-fade is offset.  Not
# tuned: they are four distinct values spread over one half-width, which is the
# minimum needed to stop the four limbs changing on the same frame.  The sign
# pattern keeps left and right of the same pair apart, which is where a real
# dancer's lead-and-follow lives.
# --seam-transition after: frames before a bar line that the completion may
# still adjust (keep weight 0.25 at -1, 0.75 at -2).  Two, because the arrival
# the flag exists to protect sits at -3/-4 (the draft's wrist-speed minimum is at
# -4, ground truth's at -3), and a hard edge at -1 made the seam jerk 7.96
# against fix7's 1.27.
SEAM_AFTER_LEAD = 2
# --seam-transition after: frames over which the arriving unit keeps its last
# velocity past the bar line, eased to zero by a raised cosine (0.90, 0.65,
# 0.35, 0.10 of it) -- so it settles on the downbeat instead of freezing there.
SEAM_COAST = 4
# --draft-bar-units: a slot counts as a whole bar when its length in query beats
# is within this of the segmentation's beats per bar.  Half a beat: the regular
# slots on the ten vis clips read 3.58-4.11 beats, the irregular ones 0.3-2.5
# and 4.58-7.7.
BAR_SLOT_TOLERANCE = 0.5
LIMB_STAGGER = {"left_arm": -0.6, "right_arm": 0.6,
                "left_leg": 0.25, "right_leg": -0.25}


def _draft_held(draft, normalizer_path, threshold=0.012, run=6):
    """Frames where the draft's arms and hands hold still (speed of elbows+wrists, root-relative, below THRESHOLD
    m/frame for at least RUN frames) -- the same definition as the 'hold' column of the G-series evaluation."""
    joints = np.asarray(decode_motion(torch.as_tensor(draft).float().cpu(), normalizer_path)["full_pose"], dtype=np.float64)
    rel = joints - joints[:, :1]
    speed = np.r_[0.0, np.linalg.norm(np.diff(rel[:, [18, 19, 20, 21]], axis=0), axis=-1).mean(-1)]
    held = np.zeros(len(speed), dtype=bool); count = 0
    for i, still in enumerate(speed < threshold):
        count = count + 1 if still else 0
        if count >= run:
            held[i - run + 1:i + 1] = True
    return held


def _beat_free_profile(music, frames, keep, stride=1, bar_start=None, held=None, free_max=1.0, hold_frac=0.5):
    """--completion-beat-keep K: per-frame weight (1 = regenerate) that is 0 within K of every beat interval on each side
    of each beat of the QUERY's own grid (music channel 34) and rises as a raised cosine to 1 at the interval's middle.
    Music only.  Frames before the first beat / after the last are left alone (0)."""
    track = np.asarray(music.cpu() if hasattr(music, "cpu") else music)
    free = np.zeros(int(frames), dtype=np.float32)
    beats = np.flatnonzero(track[:, 34] > 0.5) if track.ndim == 2 and track.shape[1] > 34 else []
    if stride > 1 and len(beats):
        # anchor only every STRIDE-th beat, counted from the plan's first bar line (so 2 = beats 1 and 3 of the bar)
        first = int(np.argmin(np.abs(beats - int(bar_start)))) if bar_start is not None else 0
        beats = beats[(np.arange(len(beats)) - first) % int(stride) == 0]
    half = max(1e-3, 0.5 - float(keep))
    for a, b in zip(beats[:-1], beats[1:]):
        a, b = int(a), int(min(b, frames))
        if b - a < 4:
            continue
        if held is not None and np.asarray(held[a:b]).mean() > hold_frac:
            continue    # --completion-beat-keep-holds: the dancer is holding a shape here; leave it to the draft
        u = (np.arange(a, b) - a) / float(b - a)
        x = np.abs(u - 0.5) / half
        # free_max < 1: the middle is a BLEND of the draft and the re-drawn frames rather than a full re-draw, so the
        # completion's pull toward an average pose (DEFECTS 92.4) is only partly applied
        free[a:b] = float(free_max) * np.where(x < 1.0, 0.5 + 0.5 * np.cos(np.pi * x), 0.0)
    return free


def _bar_novelty(music, bar_bounds):
    """Per bar line k (1..n_bars-1): cosine distance between the mean timbre+harmony (MFCC 1-20, chroma; z-scored over
    the clip) of the two bars before it and the two after.  Music only.  {k: novelty}"""
    track = np.asarray(music.cpu() if hasattr(music, "cpu") else music, dtype=np.float64)
    bounds = [int(b) for b in bar_bounds]
    if track.ndim != 2 or track.shape[1] < 33 or len(bounds) < 3:
        return {}
    feats = track[:, 1:33]
    feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)
    bars = np.array([feats[a:max(b, a + 1)].mean(0) for a, b in zip(bounds[:-1], bounds[1:])])
    out = {}
    for k in range(1, len(bars)):
        left, right = bars[max(0, k - 2):k].mean(0), bars[k:k + 2].mean(0)
        out[k] = 1.0 - float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-9))
    return out


def _phrase_starts(music, bar_bounds, quantile=0.7, min_len=2, max_len=8):
    """--draft-continue-phrase-novelty Q: bar indices where a phrase STARTS because the MUSIC changes there -- bar lines
    whose novelty (_bar_novelty) is a local maximum and at or above the clip's Q-quantile, at least MIN_LEN bars
    apart; a phrase that runs MAX_LEN bars without a change is cut anyway.  Music only."""
    nov = _bar_novelty(music, bar_bounds)
    if not nov:
        return set()
    ks = sorted(nov)
    cut = float(np.quantile([nov[k] for k in ks], quantile))
    peaks = [k for k in ks if nov[k] >= cut and nov[k] >= nov.get(k - 1, -1) and nov[k] >= nov.get(k + 1, -1)]
    starts, last = [], 0
    for k in sorted(peaks, key=lambda k: -nov[k]):          # strongest changes first, keeping MIN_LEN spacing
        if all(abs(k - s) >= min_len for s in starts) and k >= min_len:
            starts.append(k)
    starts = sorted(starts)
    filled, last = [], 0
    for k in starts + [len(bar_bounds) - 1]:
        while k - last > max_len:
            last += max_len
            filled.append(last)
        if k < len(bar_bounds) - 1:
            filled.append(k)
        last = k
    return set(filled)


def _phrase_phase(music, bar_bounds, bars_per_phrase):
    """--draft-continue-phrase: which bar index (mod ``bars_per_phrase``) starts a phrase in THIS song.

    The phase whose bar lines carry the largest change in the music's timbre and harmony (MFCC 1-20 and chroma,
    z-scored over the clip; cosine distance between the two bars before a line and the two after it).  Ground
    truth repeats itself on the 2- and 4-bar grid (DEFECTS 94.1: bars 2/4 apart are more alike, 34/48), but the
    grid's PHASE -- which bar is a phrase's first -- is not in the plan; this reads it off the song.  My own
    heuristic, not a measured rule: the arms that use it are judged on video.  Music only.
    """
    track = np.asarray(music.cpu() if hasattr(music, "cpu") else music, dtype=np.float64)
    bounds = [int(b) for b in bar_bounds]
    if track.ndim != 2 or track.shape[1] < 33 or len(bounds) < 4 or bars_per_phrase < 2:
        return 0
    feats = track[:, 1:33]
    feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)
    bars = np.array([feats[a:max(b, a + 1)].mean(0) for a, b in zip(bounds[:-1], bounds[1:])])
    novelty = {}
    for k in range(1, len(bars)):
        left, right = bars[max(0, k - 2):k].mean(0), bars[k:k + 2].mean(0)
        novelty[k] = 1.0 - float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-9))
    scores = [np.mean([v for k, v in novelty.items() if k % bars_per_phrase == p] or [0.0])
              for p in range(bars_per_phrase)]
    return int(np.argmax(scores))


def _bar_rhythm_pattern(music, start, end, bins):
    """The QUERY bar's onset pattern, the same shape the library side reports.

    Channel 33 of the query's own audio -- the task's input -- binned over the
    slot and normalised to sum 1, so the comparison is about WHERE the onsets
    fall inside the bar rather than how many there are.  No motion is read on
    either side; see CLAUDE.md 1.6 for why that line matters here.
    """
    if music is None:
        return None
    column = np.asarray(music)[int(start):int(end), MUSIC_ACCENT_CHANNEL]
    if len(column) < bins:
        return None
    edges = np.linspace(0, len(column), bins + 1).astype(int)
    counts = np.array([column[a:b].sum() for a, b in zip(edges[:-1], edges[1:])],
                      dtype=np.float64)
    total = counts.sum()
    return counts / total if total > 0 else None


def _continue_root(root, previous_root, previous_velocity, velocity_blend=0,
                   lead=False):
    """Chain one retrieved segment's root onto the previous one's; returns a copy.

    ``root`` is the segment's root-position columns, ``[T, C]``, in normalised
    units -- the release normalizer is per-dimension affine, so a constant offset
    and a linear extrapolation are both exact there.

    POSITION.  The segment is translated so its first frame continues from the
    previous segment's last.  With ``lead`` off that first frame lands ON the
    previous last frame, which is what this function did inline until
    2026-09-16 -- and it leaves the step across every seam exactly ZERO, one
    frame of a stopped body.  ``lead`` puts it one step ahead, along the
    previous segment's own velocity, so the body keeps moving through the seam.

    VELOCITY.  Position continuity is not enough.  The offset above leaves the
    body's speed and direction changing in one frame: measured 2026-09-06 on the
    eval clips with --draft-root-continuity xy, the worst single-frame root step
    sits AT a join on 18 of 18 drafts, median distance 1 frame, at 0.13-0.37 m
    against ground truth's 0.022.  So ``velocity_blend`` eases the per-frame
    delta from what the previous segment was doing into what this one does,
    over that many frames.  A raised cosine, not a line: a linear blend removes
    the step in velocity and leaves one in acceleration, which is the mistake
    this file has now made twice at other levels.  It corrects frames 1.. of the
    segment, which is why it never reached the zero step at frame 0.
    """
    root = root.clone()
    anchor = previous_root
    if lead and previous_velocity is not None:
        anchor = previous_root + previous_velocity
    root += anchor - root[0]
    if velocity_blend and previous_velocity is not None and len(root) > 1:
        span = min(int(velocity_blend), len(root) - 1)
        correction = previous_velocity - (root[1] - root[0])
        for step_index in range(span):
            weight = 0.5 + 0.5 * math.cos(math.pi * (step_index + 1) / (span + 1))
            root[step_index + 1:] += correction * weight
    return root


def _smooth_root_seams(draft, mask, labels, columns, span, starts=None):
    """Carry the chained root through every bar seam at speed, in place.

    WHY.  The operator, 2026-09-16, on output/sample_20260916_fix2: "蒙皮上偶发的
    位置跳变 ... 视觉上偶尔就会有顿挫感".  tools/score_seam_root_speed.py, ten vis
    clips: root speed at a seam over speed 13-14 frames away read 0.24 against
    ground truth's 1.19 (lower on 10/10, P=0.002), with a catch-up surge at
    distances 5-6 -- the body stopped at every bar line and lurched out of it.
    Two things did it: continuity put each segment's first frame ON the previous
    segment's last (a zero step), and ``_blend_draft_seams`` dragged the chained
    root toward two frozen anchor frames.  The second is removed by keeping
    these columns out of that blend; this removes the first and eases the
    velocity from one prototype's into the next.

    For each seam where both sides are conditioned, every later conditioned
    frame is shifted so the seam frame lands one velocity step past the last
    frame before it, and the new segment's velocity is eased in over ``span``
    frames with ``_continue_root``'s raised cosine.  Shifting everything after
    the seam keeps the chain continuous; unconditioned frames are left as they
    are.  Only ``columns`` move.  Simulated with the speed doubling across a
    seam, the speed passes monotonically from one side's to the other's
    (min 1.00, max 2.00 of the slower), where velocity-extrapolated anchors
    inside the blend still dipped to 0.79.
    """
    # ``starts``: where retrieval units begin.  build_draft chains BAR units
    # (labels split at the bar grid), so a label run several bars long holds
    # seams that a label-change scan never sees -- measured 2026-09-16, 23 of 92
    # bar seams on the ten vis clips, 22 of which still carried the zero step
    # when this function found its seams from ``labels``.
    conditioned = mask[:, 0] > 0
    cols = list(columns)
    if starts is None:
        starts = [segment.start for segment in labels_to_segments(labels)]
    seams = [int(start) for start in sorted(set(int(x) for x in starts))
             if start >= 2 and bool(conditioned[start])
             and bool(conditioned[start - 1]) and bool(conditioned[start - 2])]
    for seam in seams:
        index = torch.nonzero(conditioned[seam:], as_tuple=False).squeeze(1) + seam
        if len(index) < 2:
            continue
        previous = draft[seam - 1, cols].clone()
        velocity = previous - draft[seam - 2, cols]
        rows = index[:, None]
        path = draft[rows, torch.tensor(cols)[None, :]]
        draft[rows, torch.tensor(cols)[None, :]] = _continue_root(
            path, previous, velocity, velocity_blend=span, lead=True)


def _coast(start, velocity, count):
    """``count`` frames that leave ``start`` at ``velocity`` per frame and ease it
    to zero over SEAM_COAST frames (raised cosine), then hold.  Contacts (the
    first CONTACT_CHANNELS) are flags and never move."""
    velocity = velocity.clone()
    velocity[:CONTACT_CHANNELS] = 0.0
    frames = []
    current = start.clone()
    for step in range(1, count + 1):
        if step <= SEAM_COAST:
            current = current + velocity * (0.5 + 0.5 * math.cos(math.pi * step / (SEAM_COAST + 1)))
        frames.append(current.clone())
    return torch.stack(frames) if frames else start.new_zeros((0,) + tuple(start.shape))


def _blend_draft_seams(draft, mask, labels, half_width, stagger=False,
                       window="triangle", skip_columns=(), seams=None,
                       only_columns=None, side="centred"):
    """Cross-fade the draft across every prototype-to-prototype seam, in place.

    Only seams where both sides are conditioned are touched: a boundary onto a
    transition gap is ``gap_fill``'s business and is left to it, so the two
    treatments cannot silently double up on the same frames.

    The ramp is a raised cosine rather than linear, because a linear cross-fade
    removes the step in the value and leaves one in the first derivative -- and
    the defect being removed is measured in jerk.

    ``window`` shapes the envelope that decides how much of the ramp replaces the
    prototype's own value.  ``"triangle"`` is the published behaviour and is
    ``1 - |index - centre| / half_width``: smooth ramp, but the envelope itself
    has a CORNER at the seam, and multiplying a corner into the signal puts a
    jerk spike exactly where the seam is.  The corner's slope is ``1/half_width``,
    so the narrower the blend the sharper the spike -- which is why the measured
    seam jerk is NON-MONOTONIC in the blend width (2026-09-05, 17 eval clips,
    filler frames excluded, boundaries read from the artifact's own
    ``slot_start``): ground truth 0.2553, no blending 0.3515, and then
    **half-width 4 at 0.5139, worse than not blending at all**, 6 at 0.3195,
    8 at 0.2272, 12 at 0.1805.  ``"cosine"`` replaces the corner with a raised
    cosine of the same argument, which is C1 at the seam and at the edges.
    The docstring above already records that the RAMP was made a raised cosine
    for exactly this reason; the envelope was left linear, one level up.

    ``stagger`` offsets each limb's fade so the four do not change on the same
    frame.  **This is the difference between removing the jolt and removing what
    the jolt destroyed.**  Measured 2026-08-30: a single library prototype has
    the limb structure of a real dancer -- cross-correlation peaks at lag 0 on
    **0.000** of limb pairs, PC1 0.558, against ground truth's 0.500 and 0.506.
    Butt-joint two of them and the draft reads **1.000** and 0.844: every seam is
    one instant at which all four limbs change together, and those perfectly
    synchronous events dominate the correlation.  A synchronous cross-fade
    softens the jolt without touching that -- at half-width 8 the draft is still
    1.000 -- because it moves all four limbs on the same frames it always did.
    """
    # ``seams``: an explicit list of seam frames instead of the label changes.
    # ``only_columns``: blend these channels and nothing else (no stagger).
    # Both exist for the root HEIGHT at bar seams inside a label run: build_draft
    # chains bar units, so a run several bars long holds seams this function never
    # saw, and --draft-root-continuity xy does not chain z -- the height stepped
    # raw there (median 4.7 cm, max 25.8 cm on the ten vis clips), and those
    # steps are the "位置跳变" the operator saw, because the strip's root-locked
    # view replaces every generated row's xy with ground truth's and shows only
    # height (DEFECTS 81).
    conditioned = mask[:, 0] > 0
    candidates = ([segment.start for segment in labels_to_segments(labels)]
                  if seams is None else sorted(set(int(x) for x in seams)))
    seams = []
    for start in candidates:
        if start <= 0 or start >= len(draft):
            continue
        if not conditioned[start] or not conditioned[start - 1]:
            continue
        seams.append(start)
    frames = len(draft)
    # ``skip_columns``: channels this blend must not touch.  Used for the root
    # translation that --draft-root-continuity already chains: the ramp below
    # interpolates between two FROZEN anchor frames, which for a travelling root
    # is a fixed point, so inside +-half_width it dragged the body to a halt at
    # every bar line and made it lurch to catch up afterwards (DEFECTS 81).
    skip = set(int(c) for c in skip_columns)
    if only_columns is not None:
        plans = [(tuple(int(c) for c in only_columns if int(c) not in skip), 0.0)]
    elif stagger:
        # The remainder -- root, contacts, spine, neck, head -- still needs a
        # fade, but it must NOT be a whole-vector one: applying the full blend
        # on top of the staggered limb blends re-synchronises exactly what the
        # stagger separated.  The first version did that and made lag-0 *worse*
        # at half-width 15 (0.917 against the synchronous 0.750).
        limb_columns = {column for dims in LIMB_DIMS.values() for column in dims}
        rest = tuple(c for c in range(draft.shape[1])
                     if c not in limb_columns and c not in skip)
        plans = [(LIMB_DIMS[name], LIMB_STAGGER[name]) for name in LIMB_DIMS]
        plans.append((rest, 0.0))
    elif skip:
        plans = [(tuple(c for c in range(draft.shape[1]) if c not in skip), 0)]
    else:
        plans = [(None, 0)]
    if side == "after":
        # ``--seam-transition after``: let the arriving unit coast to a stop past
        # the bar line and ease from there into the next unit's OWN trajectory
        # over [seam, seam + half_width), never past the next seam.  Nothing
        # before the bar line is touched, which is the point.  No stagger (the
        # CLI refuses it): each limb starts from where it arrived, so there is
        # no synchronous jump to spread out.
        if only_columns is not None:
            columns = plans[0][0]
        elif skip:
            columns = tuple(c for c in range(draft.shape[1]) if c not in skip)
        else:
            columns = None
        bounds = sorted(seams)
        for number, seam in enumerate(bounds):
            limit = bounds[number + 1] if number + 1 < len(bounds) else frames
            low, high = seam, min(frames, seam + int(half_width), limit)
            if high - low < 2:
                continue
            anchor = draft[seam - 1].clone()
            # COAST, don't freeze: the arriving frame is still moving, and holding
            # it is a velocity step at the bar line.  Keep its velocity and ease
            # it to zero over SEAM_COAST frames, so the body settles ON the
            # downbeat, then fade into the next unit.  Measured 2026-09-22 on the
            # ten vis clips, half-width 12, no completion (seam jerk / interior
            # median; fix7 1.27, ground truth 1.25): limbs following the arriving
            # SOURCE dancer's own frames past the cut with the root frozen 1.78,
            # root coasting 1.60, those frames capped at 4 1.54 -- and they
            # showed the start of that dancer's next move and dropped it
            # (7618203431723357818 seam 320, a lunge neither unit contains);
            # everything coasting 0.69.
            carry = _coast(anchor, (draft[seam - 1] - draft[seam - 2]) if seam >= 2
                           else torch.zeros_like(anchor), high - low)
            for index in range(low, high):
                weight = 0.5 - 0.5 * math.cos(math.pi * (index - low + 0.5) / (high - low))
                source = carry[index - low]
                if columns is None:
                    draft[index] = source * (1.0 - weight) + draft[index] * weight
                else:
                    for column in columns:
                        draft[index, column] = (source[column] * (1.0 - weight)
                                                + draft[index, column] * weight)
        return
    for seam in seams:
      for columns, offset in plans:
        centre = seam + offset * half_width
        low = max(0, int(round(centre - half_width)))
        high = min(frames, int(round(centre + half_width)))
        if high - low < 2:
            continue
        # Anchor on the two frames adjacent to the seam so the fade interpolates
        # between what each prototype actually held there.
        before = draft[seam - 1].clone()
        after = draft[seam].clone()
        for index in range(low, high):
            position = (index - low + 0.5) / max(high - low, 1)
            weight = 0.5 - 0.5 * math.cos(math.pi * position)
            target = before * (1.0 - weight) + after * weight
            # blend the prototype's own value towards that ramp, strongest at the
            # seam and zero at the window edge, so the interior is untouched
            reach = 1.0 - abs(index - (centre - 0.5)) / max(half_width, 1)
            reach = max(0.0, min(1.0, reach))
            if window == "cosine":
                # Same support, same peak, no corner: 0.5-0.5cos(pi*reach) is
                # flat-topped at reach=1 (the seam) and flat at reach=0 (the
                # edges), so the envelope no longer injects its own jerk.
                reach = 0.5 - 0.5 * math.cos(math.pi * reach)
            if columns is None:
                draft[index] = draft[index] * (1.0 - reach) + target * reach
            else:
                for column in columns:
                    draft[index, column] = (draft[index, column] * (1.0 - reach)
                                            + target[column] * reach)


def _fill_draft_gaps(draft, mask, mode):
    """Replace the zeros between retrieved segments, in place, leaving mask alone.

    The mask is deliberately untouched: it is the completion model's noise
    scale and its declaration of what was actually retrieved, and a filled gap
    is neither.  Only the step in the draft's *values* is removed.

    A draft with no retrieved segment at all stays zero -- there is nothing to
    hold or ramp between, and inventing a pose there would be conditioning the
    model on a value no prototype produced.
    """
    conditioned = torch.nonzero(mask[:, 0] > 0, as_tuple=False).flatten().tolist()
    if not conditioned:
        return
    first, last = conditioned[0], conditioned[-1]
    # Outside the retrieved span there is only one side to match, so both modes
    # do the same thing: extend the nearest retrieved frame.
    draft[:first] = draft[first]
    draft[last + 1 :] = draft[last]
    previous = first
    for index in conditioned[1:]:
        gap = index - previous
        if gap > 1:
            if mode == "hold":
                draft[previous + 1 : index] = draft[previous]
            else:
                steps = torch.arange(
                    1, gap, dtype=draft.dtype, device=draft.device
                ).unsqueeze(1) / float(gap)
                draft[previous + 1 : index] = (
                    draft[previous] * (1.0 - steps) + draft[index] * steps
                )
        previous = index


class GroundTruthPlanStore:
    """Read frame-aligned atomic labels for ORACLE-only diagnostics.

    WHY THIS WAS REWRITTEN (2026-09-01).  It used to read ``<sequence>_slice0``
    and nothing else, so ORACLE inference was capped at one 150-frame window and
    the caller raised "supports at most one 150-frame slice".  That made the one
    experiment that separates "the plan is wrong" from "the plan is fine and
    everything downstream of it loses the dance" impossible to run on a whole
    clip -- which is the question the operator asked on 2026-09-01.

    The obvious repair is wrong and was checked before it was written: the
    release's windows are NOT a tiling.  ``build.json``'s window_policy reads
    ``window_length 150, window_stride 15``, so a 17.5 s clip is 26 windows
    overlapping by 135 frames.  Concatenating slices end to end -- the natural
    thing to write -- fabricates a track 8x too long out of repeated frames, and
    it looks entirely plausible while doing it (a 195 s "clip").

    So the track is rebuilt from ``windows.jsonl``'s own ``start_frame`` /
    ``end_frame_exclusive`` for each window, with ``label_valid_mask`` deciding
    which frames a window is allowed to contribute.  Because the windows overlap,
    every interior frame is written many times over, and that redundancy is used
    as the check rather than discarded: a frame written twice with two different
    labels means the reconstruction is wrong, and ``conflicts`` counts them.
    Measured over the 90-clip evaluation set: 0 conflicting frames, 100% valid
    coverage, mean 0.928 segments/second against the scorecard's ground-truth
    calibration of 0.917.

    Falls back to the old ``_slice0`` behaviour only when a release has no
    ``windows.jsonl``, and records which mode it is in so a reader is never left
    inferring it.
    """

    def __init__(self, data_root, splits=("train", "test", "val")):
        root = Path(data_root)
        self.plans = {}
        self.windows = {}
        self._labels = {}
        self._masks = {}
        self.conflicts = 0
        manifest = root / "windows.jsonl"
        self.mode = "windows.jsonl" if manifest.is_file() else "slice0"
        for split in splits:
            split_root = root / split
            if not split_root.is_dir():
                continue
            with open(str(split_root / "names.json")) as handle:
                names = json.load(handle)
            labels = np.load(str(split_root / "labels.npy"), mmap_mode="r")
            if len(names) != len(labels):
                raise ValueError("unaligned names and labels under {}".format(split_root))
            self._labels[split] = labels
            mask_path = split_root / "label_valid_mask.npy"
            self._masks[split] = (np.load(str(mask_path), mmap_mode="r")
                                  if mask_path.is_file() else None)
            for index, name in enumerate(names):
                if name in self.plans:
                    raise ValueError("duplicate atomic sample name: {}".format(name))
                self.plans[name] = (labels, index)
        if self.mode == "windows.jsonl":
            for line in open(str(manifest)):
                row = json.loads(line)
                split = row.get("split")
                if split not in self._labels:
                    continue
                self.windows.setdefault(row["sequence_id"], []).append(
                    (int(row["start_frame"]), int(row["end_frame_exclusive"]),
                     int(row["array_index"]), split))
            for spans in self.windows.values():
                spans.sort()

    def has_sequence(self, sequence_name):
        if sequence_name in self.windows:
            return True
        return sequence_name + "_slice0" in self.plans

    def _full_track(self, sequence_name):
        spans = self.windows[sequence_name]
        total = max(end for _, end, _, _ in spans)
        track = np.full(total, -1, np.int64)
        for start, end, index, split in spans:
            labels = np.asarray(self._labels[split][index], np.int64)[: end - start]
            mask = self._masks[split]
            keep = (np.asarray(mask[index])[: end - start].astype(bool)
                    if mask is not None else np.ones(end - start, bool))
            window = track[start:end]
            written = (window >= 0) & keep
            self.conflicts += int((window[written] != labels[written]).sum())
            window[keep] = labels[keep]
        return track

    def get(self, sequence_name, length):
        if sequence_name in self.windows:
            track = self._full_track(sequence_name)
            if length > len(track):
                raise ValueError(
                    "ORACLE ground-truth labels for {} cover {} frames, {} requested".format(
                        sequence_name, len(track), length))
            head = track[:length]
            if (head < 0).any():
                raise ValueError(
                    "ORACLE ground-truth labels for {} leave {} of the first {} frames "
                    "uncovered; a hole would be filled with whatever -1 means downstream"
                    .format(sequence_name, int((head < 0).sum()), length))
            return torch.from_numpy(np.array(head, dtype=np.int64, copy=True))
        key = sequence_name + "_slice0"
        if key not in self.plans:
            raise KeyError("no ORACLE ground-truth atomic labels for {}".format(sequence_name))
        labels, index = self.plans[key]
        plan = torch.from_numpy(np.array(labels[index], dtype=np.int64, copy=True))
        if length > len(plan):
            raise ValueError(
                "ORACLE ground-truth label slice for {} has only {} frames".format(
                    sequence_name, len(plan)
                )
            )
        return plan[:length]


def _window_starts(length, window_size, stride):
    if length <= window_size:
        return [0]
    starts = list(range(0, length - window_size + 1, stride))
    final = length - window_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def _pad_frames(values, length):
    if len(values) >= length:
        return values[:length]
    padding = values.new_zeros((length - len(values),) + values.shape[1:])
    return torch.cat((values, padding), dim=0)


PLAN_VOTE_TIE_BREAKS = ("centre", "index")


def _fuse_windows(sampled, starts, lengths, total, window_size, mode,
                  tie_break="centre"):
    """One label per frame out of overlapping window plans.

    ``centre`` is the default because it keeps the sampling distribution the
    model was trained to produce: every frame is still one draw, taken from the
    window that has the most context around it.  ``vote`` is an ensemble over
    the draws covering a frame, which is a different estimator and is offered so
    the difference can be shown rather than assumed.

    ``tie_break`` decides what ``vote`` does when two classes hold the same
    number of votes, which here is common rather than exotic: at
    ``--plan-stride 15`` about seven to ten windows cover a frame and the median
    winning count is 5.9, so exact ties between the top two are frequent.

    ``"index"`` is what this function used to do unconditionally: it ended in
    ``counts.argmax(dim=1)``, and ``torch.argmax`` returns the *first* maximal
    index.  Transition is index 0, so **every tie transition was part of went to
    transition by construction** -- not because it won, but because it sorts
    first.  Measured on the 33 clean5b5 M6 clips with
    ``planner_v1A_s15/planner_step135648.pt`` at seed 20260816
    (``tools/probe_plan_vote_ties.py``): 7.28% of frames are ties, 62.8% of
    those ties have transition among the winners, and the rule alone accounts
    for **4.6 of the 22.0 points** by which the shipped plan's transition share
    exceeded the ground truth -- 9.0% of all planned frames.  Through the whole
    shipped path over all 65 M6 clips (``runs/clean5b5_plan_rule_scan.json``)
    it is 3.4 of 23.7 points: 0.5588 against 0.5253, ground truth 0.3216.  So it
    is a real term and a minority one -- the plurality itself is the larger half
    and only ``centre`` or ``taper`` touches that.

    ``"centre"`` (default since 2026-08-23) breaks a tie the way the ``centre``
    mode picks a frame: among the tied classes, take the one voted by the window
    holding this frame most centrally, and on an exact distance tie the earlier
    window.  Where the vote has an opinion nothing changes; where it has none,
    the fusion degrades to ``centre`` rather than to label 0.  It cannot fail to
    resolve, because a tied class has at least one vote and therefore at least
    one window behind it.

    ``dataset.atomic.majority_vote`` already treats a tie as something that must
    be decided rather than inherited (it keeps the incumbent centre label, and
    says so); this path never got the same treatment.  Artifacts produced before
    2026-08-23 reproduce with ``--plan-vote-tie-break index``, and every
    manifest written since records which rule ran.
    """
    if tie_break not in PLAN_VOTE_TIE_BREAKS:
        raise ValueError(
            "plan vote tie-break must be one of {}, got {!r}".format(
                PLAN_VOTE_TIE_BREAKS, tie_break))
    if mode in ("vote", "taper"):
        classes = int(sampled.max()) + 1
        counts = torch.zeros(total, classes,
                             dtype=torch.int32 if mode == "vote" else torch.float32)
        # rank[frame, class] = the most central window that voted that class,
        # as (distance to that window's centre, window order) packed into one
        # comparable number.  ``len(starts)`` is a strict upper bound on the
        # order, so the pair compares lexicographically and no two entries can
        # collide: a window votes exactly one class per frame.
        # Only materialised for the tie-break that needs it: this is another
        # total x classes array, and a long track times a 4,000-class vocabulary
        # is not free.
        rank = (torch.full((total, classes), float("inf"))
                if tie_break == "centre" else None)
        scale = float(max(len(starts), 1))
        for order, (row, start, length) in enumerate(zip(sampled, starts, lengths)):
            index = torch.arange(start, start + length)
            voted = row[:length].to(torch.long)
            centre = start + (length - 1) / 2.0
            distance = (index.to(torch.float32) - centre).abs()
            if mode == "vote":
                counts[index, voted] += 1
            else:
                # Triangular taper: a window's say in a frame falls off linearly
                # to (nearly) nothing at its own edge, which is the same shape
                # ``_blend_weights`` already applies to the motion it stitches.
                half = max((length - 1) / 2.0, 1.0)
                counts[index, voted] += (1.0 - distance / (half + 1.0))
            if tie_break == "centre":
                key = distance * scale + order
                rank[index, voted] = torch.minimum(rank[index, voted], key)
        if tie_break == "index":
            return counts.argmax(dim=1).to(sampled.dtype)
        winners = counts == counts.max(dim=1, keepdim=True).values
        contested = torch.where(winners, rank, torch.full_like(rank, float("inf")))
        return contested.argmin(dim=1).to(sampled.dtype)
    if mode != "centre":
        raise ValueError("unknown plan fusion {!r}".format(mode))
    labels = torch.zeros(total, dtype=sampled.dtype)
    # Distance from a frame to the centre of the window that supplied it; a
    # window only wins a frame it holds more centrally than every other.
    best = torch.full((total,), float("inf"))
    for row, start, length in zip(sampled, starts, lengths):
        index = torch.arange(start, start + length)
        centre = start + (length - 1) / 2.0
        distance = (index.to(torch.float32) - centre).abs()
        take = distance < best[index]
        labels[index[take]] = row[:length][take]
        best[index[take]] = distance[take]
    return labels


def track_summary(music):
    """``concat(mean, std)`` over a whole track's music -- the planner's c_music.

    Defined once in ``dataset/global_music.py`` for training, where it is
    stitched from ``windows.jsonl`` because overlapping windows would weight the
    middle of a track ten times its ends.  Inference has no such problem: the
    music array here *is* the track, in the same 35-D features the release
    stored, so the same two moments computed over it are the same vector.

    It is taken over the real frames only.  Padding a window to ``seq_len`` and
    summarising after would put a block of zeros into the mean, and the shorter
    the clip the further its summary would drift from the one training used.
    """
    values = torch.as_tensor(music, dtype=torch.float32)
    if values.ndim != 2:
        raise ValueError("track summary needs [frames, music_dim]")
    return torch.cat((values.mean(dim=0), values.std(dim=0, unbiased=False)))


def planner_wants_global_music(planner):
    """Read the whole-track head off the module, never off a flag.

    Same rule as ``completion_conditions_on_labels``: a planner built with the
    summary refuses to run without it, so asking the loaded module is the only
    way the caller and the checkpoint cannot disagree.
    """
    return getattr(getattr(planner, "model", None), "global_projection", None) is not None


BEAT_CHANNEL = 34
# Channel 33 is the onset-peak one-hot -- the song's ACCENTS, 17.2% of frames
# against the beat channel's 6.6%.  dataset/bar_tokens names it the same way
# and SUMS rather than averages it when pooling a bar, because the number of
# onsets in a bar is a rhythmic density.
MUSIC_ACCENT_CHANNEL = 33


def bar_grid_bounds(music, beats_per_segment=4, length=None):
    """``([0, cut, ..., length], phase)`` -- the bar spans, or ``(None, None)``.

    THE ONE PLACE the plan's bar grid is derived.  ``snap_plan_to_bar_grid``,
    ``bar_bounds_of`` (which is what ``--draft-bar-prototypes`` cuts retrieval
    on) and the bar-token planner all come through here, so the plan's bars and
    retrieval's bars are the same bars by construction rather than by three
    slices of ``beats[phase::k]`` that happen to agree.

    ``(None, None)`` -- and NOT a fallback grid -- when the clip carries fewer
    than ``beats_per_segment + 2`` beats, which is the threshold
    ``snap_plan_to_bar_grid`` has always used.  Callers decide what that means:
    the snap leaves the plan alone, the bar-token planner refuses to run,
    because a bar planner with no bars would otherwise plan one token for the
    whole clip and look like it worked.
    """
    from tools.segment_on_music_beats import choose_phase

    from dataset.bar_tokens import bar_bounds, beat_frames

    music = torch.as_tensor(music)
    beats = beat_frames(music, BEAT_CHANNEL)
    if len(beats) < beats_per_segment + 2:
        return None, None
    phase = int(choose_phase(beats, music[:, 0].numpy(), beats_per_segment)["phase"])
    return (bar_bounds(beats, beats_per_segment, phase,
                       len(music) if length is None else length),
            phase)


def snap_plan_to_bar_grid(labels, music, beats_per_segment=4):
    """Replace the plan's segmentation with the one M1 actually cut on.

    M1 has cut on the music's beat grid since 2026-08-20
    (``tools/segment_on_music_beats.py``): every ``beats_per_segment`` beats, at
    a per-clip phase chosen by onset energy.  That grid is channel 34 of the
    35-D music feature the planner is already conditioned on, so the boundaries
    the planner is trying to predict are a closed-form function of its own
    input.  Measured over the 65 clean5b5 M6 clips: **97.3%** of ground-truth
    boundaries sit on a single 4-beat phase of that grid against a 3.6%
    uniform-random control, while the planner's own boundaries reach 30.7%.

    Each bar then takes the majority *non-transition* class among the planner's
    frames inside it, and is transition only when the planner named no atomic
    movement anywhere in the bar.  A plain majority would not do: the planner is
    already over half transition, so majority pooling hands whole bars to
    transition -- the same "plurality favours the biggest class" defect one
    level up.  Measured over 596 bars, majority pooling leaves the share at
    0.5255 while this rule brings it to 0.3712, ground truth 0.3214.

    This is a rule this repository invented, and it is a calibration of how much
    vocabulary is admitted, not evidence that any class is more correct.

    Returns ``(labels, phase)`` unchanged with ``phase=None`` when the clip has
    too few beats to form a grid.
    """
    bounds, phase = bar_grid_bounds(music, beats_per_segment, length=len(labels))
    if bounds is None:
        return labels.clone(), None
    out = labels.clone()
    for start, end in zip(bounds[:-1], bounds[1:]):
        segment = labels[start:end]
        if not len(segment):
            continue
        named = segment[segment != 0]
        if len(named):
            values, counts = torch.unique(named, return_counts=True)
            out[start:end] = values[counts.argmax()]
        else:
            out[start:end] = 0
    return out, phase


def _slot_beat_frames(music, start, end):
    """This slot's beat frames, as offsets INSIDE the slot.

    Offsets and not absolute frames, because a candidate is judged after being
    resampled into the slot and its frame 0 is the slot's frame 0.
    """
    if music is None:
        return ()
    column = torch.as_tensor(music)[:, BEAT_CHANNEL]
    hits = torch.nonzero(column > 0.5).reshape(-1).tolist()
    return tuple(int(h) - int(start) for h in hits if start <= h < end)


def bar_music_descriptors(music, bounds):
    """One vector per bar, built so that "these two bars sound alike" is a dot
    product.  THE ONE PLACE this descriptor is defined -- the scoring tool
    imports it from here, so the measurement and the mechanism cannot drift.

    Timbre (channels 1:21) and harmony (21:33) are z-scored SEPARATELY over the
    clip before being concatenated, because the raw scales differ by two orders
    of magnitude (per-channel sd about 5-19 against 0.06-0.10): without it the
    cosine is the mel bands alone and chroma contributes nothing, so a chorus
    that returns with the same chords over a different mix would not be seen to
    return.  Channels 0, 33 and 34 are left out on purpose -- they are the
    tempo/onset/beat channels, which repeat every bar by construction and would
    make every pair of bars look alike.
    """
    music = torch.as_tensor(music).float()
    blocks = [
        torch.cat([music[start:end, 1:21].mean(0), music[start:end, 21:33].mean(0)])
        for start, end in zip(bounds[:-1], bounds[1:]) if end > start
    ]
    values = torch.stack(blocks)
    timbre, harmony = values[:, :20], values[:, 20:]

    def standardise(block):
        spread = block.std(0, keepdim=True)
        return (block - block.mean(0, keepdim=True)) / spread.clamp(min=1e-6)

    values = torch.cat([standardise(timbre), standardise(harmony)], 1)
    return values / values.norm(dim=1, keepdim=True).clamp(min=1e-6)


def music_repeat_plan(labels, music, beats_per_segment=4, strength=0.0):
    """Make the plan come back to a movement where the MUSIC comes back.

    THE MEASUREMENT THIS IMPLEMENTS (tools/score_music_repetition.py, the ten
    fixed T clips): bar pairs at least two bars apart whose music sounds alike
    carry the same ground-truth class far more than a circular-shift null --
    +0.2595 against +0.1174, dz +3.17, 10 clips of 10, median p 0.000.  Our own
    plans do not: the aligned planner reads dz +0.94 and the shipped 21-class
    plan dz -2.45.  A dancer brings the movement back when the chorus comes
    back; we redraw every bar.

    THIS IS NOT THE QUESTION ALREADY REFUTED.  That one was absolute -- "given
    this bar's music, name the class" -- and reached +0.005 over a train-fold
    majority floor, mean speed R^2 -0.31.  This one is relational: whatever the
    classes are, two bars that sound alike should carry the SAME one.  A
    choreography can be unpredictable bar by bar and still repeat its chorus,
    so the second question can have an answer where the first has none.

    NO GROUND TRUTH IS READ.  The only input is the query's own music, which is
    a legitimate inference input (the task is music -> dance).  No label is
    invented either: bars are grouped by music similarity and each group is
    handed the label THE PLANNER ITSELF already spent the most frames on inside
    that group.  The vocabulary stays the planner's; only the placement moves.

    ``strength`` in (0, 1] picks how many groups -- ``K = max(2, round(bars *
    (1 - strength)))`` -- so 0.0 leaves the plan untouched and larger values tie
    more bars together.  It is a knob and not a constant because the right
    amount is a measurement: ground truth holds 3.6 distinct classes per clip
    with its top class over 54% of frames, against 5.1 and 35% for the aligned
    planner, and those two columns are what say whether a given strength has
    gone past the dancer rather than toward them.

    Returns ``(labels, report)``; ``report`` is ``None`` when the clip carries
    too few bars to tie any of them, which is the same "no grid, no fallback
    grid" rule ``bar_grid_bounds`` states.
    """
    strength = float(strength)
    if strength <= 0.0:
        return labels.clone(), None
    if strength > 1.0:
        raise ValueError("plan_music_repeat must be in [0, 1], got {}".format(strength))
    bounds, phase = bar_grid_bounds(music, beats_per_segment, length=len(labels))
    if bounds is None or len(bounds) - 1 < 4:
        return labels.clone(), None
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    descriptors = bar_music_descriptors(music, bounds).numpy()
    bars = len(descriptors)
    groups_wanted = max(2, int(round(bars * (1.0 - strength))))
    if groups_wanted >= bars:
        return labels.clone(), None
    tree = linkage(pdist(descriptors, metric="cosine"), method="average")
    groups = fcluster(tree, groups_wanted, criterion="maxclust")
    out = labels.clone()
    moved = 0
    for group in sorted(set(int(g) for g in groups)):
        members = [i for i, g in enumerate(groups) if int(g) == group]
        counts = {}
        for bar in members:
            segment = labels[bounds[bar]:bounds[bar + 1]]
            values, occurrences = torch.unique(segment, return_counts=True)
            for value, occurrence in zip(values.tolist(), occurrences.tolist()):
                counts[int(value)] = counts.get(int(value), 0) + int(occurrence)
        named = {value: count for value, count in counts.items() if value != 0}
        pool = named or counts
        # Sorted rather than max(): ties between two labels with the same frame
        # count would otherwise follow dict insertion order, and the arm would
        # not be reproducible from its own manifest.
        winner = sorted(pool.items(), key=lambda item: (-item[1], item[0]))[0][0]
        for bar in members:
            start, end = bounds[bar], bounds[bar + 1]
            if int(labels[start]) != winner:
                moved += 1
            out[start:end] = winner
    return out, {
        "plan_music_repeat": strength,
        "bars": int(bars),
        "groups": int(len(set(int(g) for g in groups))),
        "bars_retied": int(moved),
        "bar_grid_phase": phase,
    }


PLANNER_TOKEN_RESOLUTIONS = ("frame", "bar")


def planner_token_resolution(planner_args):
    """What ONE TOKEN of this checkpoint is: a frame, or a 4-beat bar.

    Read off the checkpoint's own training arguments -- ``args`` travels inside
    every checkpoint (``train_atomic.save_checkpoint``) -- under the key
    ``planner_token_resolution``.  A checkpoint that does not carry the key
    reads ``"frame"``, which is every planner this repository has ever trained.

    The default is ``"frame"`` and not "whatever the caller asked for" because
    the two failures are not symmetric: a bar-trained checkpoint that forgot to
    declare itself is REFUSED in bar mode (loud, and the fix is one training
    argument), while the opposite default would run a frame-trained checkpoint
    on pooled music and produce a plan that looks like a plan.
    """
    value = getattr(planner_args, "planner_token_resolution", None) or "frame"
    value = str(value)
    if value not in PLANNER_TOKEN_RESOLUTIONS:
        raise SystemExit(
            "error: the planner checkpoint declares planner_token_resolution="
            "{!r}, which is not one of {}".format(
                value, list(PLANNER_TOKEN_RESOLUTIONS)))
    return value


def check_planner_bar_tokens(planner, planner_args, raw_music_dim, plan_bar_tokens,
                             plan_bar_beats):
    """Refuse a checkpoint whose tokens are not what the flags say they are.

    WHY THIS GATE HAS TO EXIST AS A GATE.  ``MusicNormalization`` deliberately
    carries its z-scoring statistics as buffers so that "a checkpoint physically
    cannot be run without the normalization it was trained with" -- but its
    buffers are ``(music_dim,)`` whether they were fit on FRAMES or on
    bar-POOLED music, so ``load_state_dict`` is strict about the shape and blind
    to which distribution produced the numbers.  Pooling changes that
    distribution: mean-pooling a 4-beat bar turns the beat one-hot on channel 34
    from a spike that is 1.0 on about 1 frame in 20 into a near-constant beat
    density, so frame statistics applied to pooled music mis-scale exactly the
    channel the bar grid is built on.  Nothing downstream would raise; the plan
    would just be wrong.  Hence an explicit declaration, checked here.

    Returns the bar pooling the checkpoint declares (``None`` in frame mode).
    """
    from dataset.bar_tokens import BAR_POOLINGS, DEFAULT_BAR_POOLING, pooled_dim

    declared = planner_token_resolution(planner_args)
    asked = "bar" if plan_bar_tokens else "frame"
    if declared != asked:
        raise SystemExit(
            "error: token-resolution mismatch. The planner checkpoint was "
            "trained on {} tokens (planner_token_resolution={!r}) and "
            "--plan-bar-tokens was {}, so it would plan over {} tokens.\n"
            "  The music normalization statistics inside the checkpoint were "
            "fit on {} music and are strict on shape only, so nothing else in "
            "the pipeline would notice.\n"
            "  Fix: {}".format(
                declared, declared, "given" if plan_bar_tokens else "not given",
                asked, declared,
                "drop --plan-bar-tokens to run this frame-trained checkpoint"
                if declared == "frame" else
                "pass --plan-bar-tokens to run this bar-trained checkpoint"))
    if not plan_bar_tokens:
        return None
    pooling = str(getattr(planner_args, "planner_bar_pooling", None)
                  or DEFAULT_BAR_POOLING)
    if pooling not in BAR_POOLINGS:
        raise SystemExit(
            "error: the planner checkpoint declares planner_bar_pooling={!r}, "
            "which is not one of {}".format(pooling, list(BAR_POOLINGS)))
    expected = pooled_dim(raw_music_dim, pooling)
    if int(planner_args.music_dim) != expected:
        raise SystemExit(
            "error: the planner checkpoint consumes {}-D music, but pooling {}-D "
            "release music with {!r} produces {}-D. The checkpoint's declared "
            "pooling and its own input width cannot both be true.".format(
                planner_args.music_dim, raw_music_dim, pooling, expected))
    beats = getattr(planner_args, "planner_bar_beats", None)
    if beats is not None and int(beats) != int(plan_bar_beats):
        raise SystemExit(
            "error: the planner checkpoint was trained on {}-beat bars and "
            "--plan-bar-beats is {}. The bar lines at inference would not be "
            "the bar lines it was trained on.".format(int(beats), int(plan_bar_beats)))
    if getattr(getattr(planner, "model", None), "music_phase_features", None) is not None:
        raise SystemExit(
            "error: this checkpoint derives MusicPhaseFeatures from channel 34 "
            "at run time, and channel 34 of POOLED music is a beat density, not "
            "the one-hot that module reads with `> 0.5`. Every one of its five "
            "rhythm channels would be computed from a signal that never crosses "
            "the threshold. A bar planner must not be trained with "
            "--planner-music-phase-features.")
    return pooling


def _bar_token_option_error(plan_stride, plan_fusion, vote_window,
                            min_segment_length):
    """The frame post-process in bar mode: what it means, and why it refuses.

    The shipped plan post-process is a 5-FRAME majority vote, a 6-FRAME minimum
    segment, and a fusion of windows overlapped at ``--plan-stride 15`` frames.
    All three are operations on a frame track.  A bar token is 37 to 89 frames
    (2.97 s maximum in the corpus, 1.23 s minimum), so "a 5-wide vote" over bar
    tokens is a five-BAR low-pass -- ten seconds of smoothing -- and "a minimum
    segment of 6" would delete every plan that is not six bars of one class.
    They are not the same operation under a different unit, so they are refused
    rather than reinterpreted.

    In bar mode there is therefore NO post-process: ``refine_plan`` is not
    called at all, the windows are the non-overlapping bar windows the model was
    trained on, and ``--plan-transition-policy`` / ``--plan-merge-order`` are
    inert because the merge they parameterise never runs (the stats record them
    as ``None`` for that reason).
    """
    offenders = []
    if plan_stride is not None:
        offenders.append("--plan-stride ({} frames)".format(plan_stride))
    if plan_fusion != "centre":
        offenders.append("--plan-fusion {}".format(plan_fusion))
    if int(vote_window) != 1:
        offenders.append("--plan-vote-window {}".format(vote_window))
    if int(min_segment_length) != 1:
        offenders.append("--plan-min-segment {}".format(min_segment_length))
    if not offenders:
        return None
    return (
        "--plan-bar-tokens is incompatible with the frame-level plan "
        "post-process: {}.\n"
        "  A bar token is a whole bar (1.23-2.97 s in this corpus), so a "
        "5-frame vote over bars is a five-BAR low-pass and a 6-frame minimum "
        "segment is a six-BAR one; overlapping windows at a frame stride "
        "cannot be fused across tokens of unequal frame length either.\n"
        "  Fix: pass --plan-vote-window 1 --plan-min-segment 1 and drop "
        "--plan-stride/--plan-fusion. In bar mode the model's own bar windows "
        "are the plan.".format(", ".join(offenders)))


def _rewrite_unavailable(labels, available_labels):
    """Plan labels with no retrievable prototype become transition.

    One definition for both resolutions.  Returns the labels and the BOOLEAN
    mask of what was rewritten, not a count: in bar mode the caller has to turn
    rewritten bars into rewritten frames, and it can only do that if it knows
    WHICH bars they were.
    """
    if available_labels is None:
        return labels, torch.zeros_like(labels, dtype=torch.bool)
    unavailable = torch.ones_like(labels, dtype=torch.bool)
    for label in available_labels:
        unavailable &= labels != label
    unavailable &= labels != 0
    labels = labels.clone()
    labels[unavailable] = 0
    return labels, unavailable


@torch.no_grad()
def _plan_over_bars(planner, music, window_size, device, deterministic, temperature,
                    available_labels, plan_bar_beats, bar_pooling,
                    planner_guidance_weight, planner_transition_logit_bias,
                    plan_bar_grid, plan_music_repeat, stats):
    """One label per BAR, expanded to the per-frame track the pipeline expects.

    THE GRID IS NOT A SECOND GRID.  The bar lines come from ``bar_grid_bounds``,
    the same function ``--plan-bar-grid`` and ``--draft-bar-prototypes`` use, so
    the bars the planner plans and the bars retrieval fills are the same bars by
    construction.  The phase is recorded in ``stats["bar_grid_phase"]`` exactly
    as the snap path already records it, and the spans themselves in
    ``stats["bar_bounds"]``, so a reader can reconstruct which bars were planned
    without re-deriving anything.

    HEAD AND TAIL.  ``bar_bounds`` starts at frame 0 and ends at ``len(music)``,
    so the frames before the first bar line and after the last one are each a
    token of their own and every frame of the clip carries a planned label.
    This is the same convention ``snap_plan_to_bar_grid`` has always used; the
    two partial spans are shorter than a bar and their pooled music is taken
    over fewer frames, which ``stats["bar_bounds"]`` makes visible.

    WINDOWS.  ``window_size`` here is the checkpoint's ``seq_len``, and for a
    bar-trained checkpoint that number counts BARS.  The windows are
    non-overlapping, which is what ``infer_plan`` does whenever no stride is
    given; a 15-second clip at 112 BPM is about 7 bars, so in practice one
    window holds the whole clip and no seam exists to fuse.
    """
    from dataset.bar_tokens import expand_labels, pool_music

    bounds, phase = bar_grid_bounds(music, plan_bar_beats, length=len(music))
    if bounds is None:
        raise ValueError(
            "--plan-bar-tokens cannot plan this clip: its music carries fewer "
            "than {} beats on channel {}, so there is no bar grid to plan over. "
            "Planning it as one long token would look like a plan.".format(
                plan_bar_beats + 2, BEAT_CHANNEL))
    tokens = pool_music(music, bounds, bar_pooling)
    starts = list(range(0, len(tokens), window_size))
    chunks, lengths = [], []
    for start in starts:
        chunk = tokens[start : start + window_size]
        lengths.append(len(chunk))
        chunks.append(_pad_frames(chunk, window_size))
    token_batch = torch.stack(chunks).to(device)
    padding_mask = torch.arange(window_size, device=device)[None] >= torch.tensor(
        lengths, device=device)[:, None]
    extra = {}
    if planner_wants_global_music(planner):
        # Over the POOLED rows, because a bar-trained release stores pooled rows
        # as its music array and ``dataset/global_music`` summarises whatever the
        # release stores.  Summarising the raw frames here would hand the model a
        # vector from a distribution it never saw.
        extra["global_music"] = track_summary(tokens).to(device)[None].expand(
            len(chunks), -1)
    sampled = planner.sample(
        token_batch,
        padding_mask=padding_mask,
        temperature=temperature,
        deterministic=deterministic,
        guidance_weight=planner_guidance_weight,
        transition_logit_bias=planner_transition_logit_bias,
        **extra,
    ).cpu()
    bar_labels = torch.cat([row[:length] for row, length in zip(sampled, lengths)])
    bar_labels, rewritten_mask = _rewrite_unavailable(bar_labels, available_labels)
    labels = expand_labels(bar_labels, bounds)
    if len(labels) != len(music):
        raise AssertionError(
            "the bar plan covers {} frames but the music has {}".format(
                len(labels), len(music)))
    if plan_bar_grid:
        # A no-op by construction -- the labels are already constant within each
        # bar of this very grid -- and it is run rather than skipped so that the
        # bar rule stays owned by the one function that defines it.  Asserted in
        # tests/test_plan_bar_tokens.py.
        labels, _ = snap_plan_to_bar_grid(labels, music, beats_per_segment=plan_bar_beats)
    labels, repeat_report = music_repeat_plan(
        labels, music, beats_per_segment=plan_bar_beats,
        strength=plan_music_repeat)
    if stats is not None:
        rewritten_frames = sum(
            end - start
            for was_rewritten, start, end in zip(
                rewritten_mask.tolist(), bounds[:-1], bounds[1:])
            if was_rewritten)
        stats.update({
            "frames": int(len(labels)),
            "windows": len(starts),
            "plan_bar_tokens": True,
            "bar_tokens": int(len(tokens)),
            "bar_pooling": bar_pooling,
            "bar_bounds": [int(b) for b in bounds],
            "bar_grid_phase": phase,
            "plan_bar_grid": plan_bar_grid,
            "plan_music_repeat": plan_music_repeat,
            "plan_music_repeat_report": repeat_report,
            "plan_bar_beats": plan_bar_beats,
            # In BARS -- the unit that was actually rewritten.
            "bars_rewritten_to_transition": int(rewritten_mask.sum()),
            # The frames those bars cover, so the number is comparable with the
            # frame-mode counter of the same name.  The two ``transition_frames``
            # keys are equal here and are both kept so a reader written for
            # frame-mode artifacts still finds what it looks for: no fusion and
            # no refine happen in bar mode, so there is only one count to report.
            "frames_rewritten_to_transition": int(rewritten_frames),
            "transition_frames_after_fusion": int((labels == 0).sum()),
            "transition_frames_after_refine": int((labels == 0).sum()),
            "atomic_segments_after_refine": _plan_atomic_segments(labels),
            # None, not their argparse defaults: no vote, no merge and no window
            # fusion runs in bar mode, so recording the values would say a
            # post-process happened.
            "plan_vote_tie_break": None,
            "plan_transition_policy": None,
            "plan_merge_order": None,
            "plan_vote_window": None,
            "plan_min_segment_length": None,
            "planner_transition_logit_bias": planner_transition_logit_bias,
        })
    return labels


@torch.no_grad()
def infer_plan(
    planner,
    music,
    window_size,
    device,
    deterministic=False,
    temperature=1.0,
    vote_window=5,
    min_segment_length=6,
    available_labels=None,
    plan_stride=None,
    plan_fusion="centre",
    plan_vote_tie_break="centre",
    plan_transition_policy="protect",
    plan_merge_order="shortest",
    planner_guidance_weight=1.0,
    planner_transition_logit_bias=0.0,
    plan_bar_grid=False,
    plan_music_repeat=0.0,
    plan_bar_beats=4,
    plan_bar_tokens=False,
    bar_pooling=None,
    stats=None,
):
    """Plan a whole track, window by window.

    ``plan_stride`` defaults to ``window_size``, which is what this did before:
    non-overlapping chunks, each planned with no knowledge of its neighbours.
    Measured on the 630-class pair's 40 generated sequences, that puts **84 of
    408** segment boundaries exactly on a chunk edge against 2.7 expected under
    a uniform null -- 30.9x, z = +49.4 -- while the ground-truth labels give
    0.76x (z = -2.2) on the same statistic.  So one generated boundary in five
    is the window grid rather than choreography, and "segments per window" is
    inflated by it.

    A stride below ``window_size`` overlaps the windows and fuses them:

    * ``centre`` takes each frame from the window whose centre is nearest, so a
      frame is never read off a window's edge and each frame still comes from
      exactly one sample of the model;
    * ``vote`` takes a majority across every window covering the frame, which is
      an ensemble and will smooth more -- it changes the per-frame distribution,
      where ``centre`` only changes which window supplies it.
    * ``taper`` is the same ensemble with each window's vote weighted by how
      centrally it holds the frame, falling off linearly to nothing at that
      window's edge.  It exists because ``vote`` and ``centre`` are the two ends
      of one axis and the repository had only ever measured the ends: a flat
      weight is ``vote``, a weight concentrated entirely on the nearest centre is
      ``centre``.  A flat weight is what makes plurality favour whichever class
      holds the largest share of the marginal -- transition -- even where no
      window is confident, and an edge vote is exactly the draw ``centre`` exists
      to avoid trusting.  This is opt-in and is **not** the default: it is a
      different estimator, not a repair of one, so it has to be priced.

    Neither widens the conditioning: the planner still sees ``window_size``
    frames of music at a time.  This treats the seam, not the context.

    ``stats``, when a dict is passed, is filled with what this function did to
    the plan on the way out.  The reason it exists is
    ``frames_rewritten_to_transition``: a planned label with no retrievable
    prototype is silently turned into transition below, which is a step the
    paper does not have -- its library covers all K prototypes by construction,
    ours can be emptied for a label by the fail-closed retrieval-group
    exclusion.  Until 2026-08-23 no artifact recorded how many frames that was,
    so a plan could be a third transition for a reason nothing in its own record
    could name.
    """
    if plan_bar_tokens:
        from dataset.bar_tokens import DEFAULT_BAR_POOLING

        problem = _bar_token_option_error(plan_stride, plan_fusion, vote_window,
                                          min_segment_length)
        if problem:
            raise ValueError(problem)
        return _plan_over_bars(
            planner, music, window_size, device, deterministic, temperature,
            available_labels, plan_bar_beats, bar_pooling or DEFAULT_BAR_POOLING,
            planner_guidance_weight, planner_transition_logit_bias, plan_bar_grid, plan_music_repeat,
            stats)
    stride = int(plan_stride or window_size)
    if not 1 <= stride <= window_size:
        raise ValueError("plan stride must be in [1, window_size]")
    starts = (
        list(range(0, len(music), window_size))
        if stride == window_size
        else _window_starts(len(music), window_size, stride)
    )
    chunks = []
    lengths = []
    for start in starts:
        chunk = music[start : start + window_size]
        lengths.append(len(chunk))
        chunks.append(_pad_frames(chunk, window_size))
    music_batch = torch.stack(chunks).to(device)
    padding_mask = torch.arange(window_size, device=device)[None] >= torch.tensor(
        lengths, device=device
    )[:, None]
    # One summary for the track, repeated per window: the point of the head is
    # that every window sees the same whole song, so a per-window summary would
    # be the thing it was built to replace.
    # The keyword is passed only when the checkpoint carries the head, so a
    # planner without one goes down the identical call it always did and every
    # artifact made before this path existed still reproduces.
    extra = {}
    if planner_wants_global_music(planner):
        extra["global_music"] = track_summary(music).to(device)[None].expand(len(chunks), -1)
    sampled = planner.sample(
        music_batch,
        padding_mask=padding_mask,
        temperature=temperature,
        deterministic=deterministic,
        guidance_weight=planner_guidance_weight,
        transition_logit_bias=planner_transition_logit_bias,
        **extra,
    ).cpu()
    if stride == window_size:
        labels = torch.cat([row[:length] for row, length in zip(sampled, lengths)])
    else:
        labels = _fuse_windows(sampled, starts, lengths, len(music), window_size,
                               plan_fusion, tie_break=plan_vote_tie_break)
    bar_phase = None
    if plan_bar_grid:
        labels, bar_phase = snap_plan_to_bar_grid(
            labels, music, beats_per_segment=plan_bar_beats)
    labels, repeat_report = music_repeat_plan(
        labels, music, beats_per_segment=plan_bar_beats,
        strength=plan_music_repeat)
    fused_transition = int((labels == 0).sum())
    labels, rewritten_mask = _rewrite_unavailable(labels, available_labels)
    rewritten = int(rewritten_mask.sum())
    refined = refine_plan(labels, vote_window, min_segment_length,
                          transition_policy=plan_transition_policy,
                          merge_order=plan_merge_order)
    if stats is not None:
        stats.update({
            "frames": int(len(labels)),
            "windows": len(starts),
            "transition_frames_after_fusion": fused_transition,
            "frames_rewritten_to_transition": rewritten,
            "transition_frames_after_refine": int((refined == 0).sum()),
            "atomic_segments_after_refine": _plan_atomic_segments(refined),
            "plan_vote_tie_break": (
                plan_vote_tie_break if plan_fusion in ("vote", "taper") else None),
            "plan_transition_policy": plan_transition_policy,
            "plan_merge_order": plan_merge_order,
            "plan_vote_window": vote_window,
            "plan_min_segment_length": min_segment_length,
            "planner_transition_logit_bias": planner_transition_logit_bias,
            "plan_bar_grid": plan_bar_grid,
            "plan_bar_beats": plan_bar_beats,
            "bar_grid_phase": bar_phase,
            "plan_bar_tokens": False,
            "plan_music_repeat": plan_music_repeat,
            "plan_music_repeat_report": repeat_report,
        })
    return refined


def _blend_weights(window_size, is_first, is_last, overlap, blend_width=None):
    """Overlap-add weights, with the averaged band decoupled from the overlap.

    WHY THE TWO ARE NOT THE SAME THING.  The window overlap exists so that each
    frame is denoised with real motion on both sides of it.  The cross-fade
    exists so the seam between two windows is not a step.  Until 2026-08-31
    this function conflated them: the ramp was the full ``overlap``, and since
    the shipped configuration is ``seq_len 150`` with ``--completion-stride
    75``, ``overlap == window_size / 2`` and the two ramps meet -- every one of
    an interior window's 150 frames carries a weight other than 1.0, so
    **79.75% of all output frames are a convex combination of two independently
    sampled diffusion outputs**.

    That combination is not a dance.  Two draws conditioned on the same music
    and the same draft share a conditional mean and have independent
    deviations, so averaging with weights w and 1-w keeps only
    ``sqrt(w^2 + (1-w)^2)`` of the deviation -- a minimum of 0.707 at w=0.5.
    It is a low-pass filter no training step ever saw, applied to four fifths
    of the output, and the operator's complaint is that the output is damped.

    ``blend_width`` narrows the averaged band without touching the context each
    window was denoised with: outside the band every frame is taken whole from
    whichever window's centre it is nearer to, and only ``blend_width`` frames
    around each junction are mixed.  ``None`` reproduces the old behaviour
    exactly, so every artifact produced before this date stays reproducible.

    NOT a fix for the model: this removes an artefact of how windows are
    reassembled, which is upstream of any judgement about what the model
    produced.  The magnitude is contested (one probe reads 12.0% of per-clip
    energy paired at p=0.0065, another 2.3% at p=0.31, and the first is
    confounded by the pre-2026-08-31 positional seeding), so it is a flag with
    a default of "unchanged" until measured on paired seeds.
    """
    weights = torch.ones(window_size, 1)
    if overlap <= 0:
        return weights
    band = overlap if blend_width is None else max(1, min(int(blend_width), overlap))
    # The junction sits at the middle of the overlap: a frame belongs to the
    # window whose centre it is nearer to, and the ramp is centred there.
    lead = (overlap - band) // 2
    if not is_first:
        weights[:lead] = 0.0
        weights[lead:lead + band] = torch.linspace(0.0, 1.0, band + 2)[1:-1, None]
    if not is_last:
        tail = overlap - band - lead
        if tail > 0:
            weights[-tail:] = 0.0
        stop = window_size - tail
        weights[stop - band:stop] = torch.linspace(1.0, 0.0, band + 2)[1:-1, None]
    return weights


@torch.no_grad()
def _pad_labels(window, window_size):
    """Right-pad a label window with 0, the transition class.

    Same convention as the model's own label channel uses for a short tail
    window: "nothing planned here" rather than whatever id happens to be zero.
    """
    if len(window) >= window_size:
        return window[:window_size]
    return torch.cat((window, window.new_zeros(window_size - len(window))))


def _hold_root_to_draft(keep_mask, noise_mask, motion_dim):
    """Pin the three root-translation channels to the draft, leave the rest.

    WHY THIS EXISTS, and it is a measurement, not a preference.  Six clips
    (7188505181892381984, 7287585049111711032, 7456795199654694202,
    7608191311518369137, 7610414564962183545, 7618203431723357818), one plan
    each, the SAME plan before and after -- ``--draft-only`` against the full
    run, so nothing but the completion differs.  The horizontal root, split by
    frequency because "travel less" and "stop shaking" are opposite
    instructions and one number cannot tell them apart:

                          0-0.5 Hz   0.5-2 Hz    >2 Hz   path      foot
                          (travel)   (stepping) (jitter) length    skate
        draft               0.3155     0.0762    0.0320   4.52 m   0.382
        after completion    0.2910     0.1299    0.0478  10.63 m   0.675
        ground truth        0.3981     0.0780    0.0355   5.00 m   0.374

    The draft's root IS the ground truth's -- it is retrieved from it -- and it
    reads at ground-truth level on every column.  The completion does not
    refine that root, it shakes it: 1.7x the stepping band, 1.5x the jitter
    band, 2.35x the path length walked over the same 20 s, 1.8x the foot skate.
    The travel band is the one column where all three agree, which is the whole
    point of splitting them: the defect is NOT that the body wanders.

    So the root channels are held to the draft wherever the draft is
    conditioned, and the pose channels keep whatever the seam ramp gave them --
    the ramp exists so the model can invent a transition between two
    prototypes, and a transition is a matter of limbs, not of where the pelvis
    is.  ``noise_mask`` supplies "is there a draft here at all"; an unconditioned
    frame has no root to hold and keeps its 0.
    """
    conditioned = (noise_mask[..., :1] > 0).to(dtype=torch.float32)
    if keep_mask is None:
        keep_mask = torch.zeros(noise_mask.shape[0], noise_mask.shape[1], motion_dim,
                                dtype=torch.float32, device=noise_mask.device)
    else:
        # ``.expand`` returns a view with a zero stride, so writing one channel
        # of it would write all of them.  Materialise before the assignment.
        keep_mask = keep_mask.contiguous().clone()
    root = slice(ROOT_POSITION_START, ROOT_POSITION_START + ROOT_POSITION_DIMS)
    keep_mask[..., root] = conditioned
    return keep_mask


def infer_completion(
    completion,
    music,
    draft,
    noise_mask,
    window_size,
    stride,
    device,
    guidance_weight=None,
    batch_size=1,
    labels=None,
    # Separate from ``labels`` on purpose: that one decides whether the MODEL
    # sees a label channel, this one only locates seams for the inpaint keep
    # mask.  Conflating them made the mask depend on --completion-label-channel,
    # which has nothing to do with where two prototypes meet.
    seam_labels=None,
    # The RETRIEVAL UNIT starts, which is what a seam actually is.  Label
    # changes are not enough: --draft-bar-prototypes gives every BAR its own
    # prototype, so on a clip with two label runs and seven bars there are six
    # prototype joins and only one label change.  Measured 2026-09-06 on
    # wild_v5:7456795199654694202:clip000, whose single plan boundary is frame
    # 118 while its six largest single-frame pose changes are at frames 62, 394,
    # 228, 283, 173 and 339 -- none of them near it.  Freeing only the label
    # changes therefore left almost every join untouched, which is what the
    # operator kept seeing.
    seam_frames=None,
    start_step=None,
    reproject_every=None,
    inpaint_seam_width=None,
    beat_free=None,
    # Which side of a seam the freed window sits on.  "centred" (published) frees
    # +-width around it; "after" frees only [seam, seam + width), so the unit
    # that is ARRIVING keeps its last frames.  See --seam-transition for why.
    inpaint_seam_side="centred",
    # Hold the root TRANSLATION channels to the draft.  Separate from the seam
    # width because it is answering a different measurement: the seam ramp is
    # about where two prototypes meet, this is about a defect that is spread
    # over every frame.  See _hold_root_to_draft for the numbers.
    keep_root=False,
    sample_steps=None,
    blend_width=None,
    draft_guidance_weight=None,
):
    if not 1 <= stride <= window_size:
        raise ValueError("completion stride must be in [1, window_size]")
    starts = _window_starts(len(music), window_size, stride)
    output = torch.zeros(len(music), draft.shape[1])
    weight_sum = torch.zeros(len(music), 1)
    overlap = window_size - stride
    if batch_size < 1:
        raise ValueError("inference batch size must be positive")
    batches = range(0, len(starts), batch_size)
    for batch_start in tqdm(
        batches,
        total=(len(starts) + batch_size - 1) // batch_size,
        desc="Completion",
        unit="batch",
    ):
        batch_starts = starts[batch_start : batch_start + batch_size]
        music_windows = []
        draft_windows = []
        mask_windows = []
        label_windows = []
        lengths = []
        for start in batch_starts:
            end = min(start + window_size, len(music))
            lengths.append(end - start)
            music_windows.append(_pad_frames(music[start:end], window_size))
            draft_windows.append(_pad_frames(draft[start:end], window_size))
            mask_windows.append(_pad_frames(noise_mask[start:end], window_size))
            if labels is not None:
                # Padded with 0 -- the transition label -- so a short tail window
                # is conditioned on "nothing planned here" rather than on
                # whatever atomic movement id happens to be zero-adjacent.
                window = labels[start:end]
                if len(window) < window_size:
                    window = torch.cat(
                        (window, window.new_zeros(window_size - len(window)))
                    )
                label_windows.append(window)
        # Passed only when there is a channel to pass it to: a completion model
        # without the embedding refuses the argument, and so does every stub.
        extra = (
            {"labels": torch.stack(label_windows).to(device)} if labels is not None else {}
        )
        mask_batch = torch.stack(mask_windows).to(device)
        keep_mask = None
        if inpaint_seam_width is not None and int(inpaint_seam_width) >= 0:
            # KEEP = conditioned AND not within inpaint_seam_width frames of a
            # seam.  Everything else is held to the draft, so the draft's own
            # richness (adjacent-unit pose distance 1.6811 against ground
            # truth's 1.3253 once filler is indexed) survives instead of being
            # regenerated away.
            #
            # A SEAM IS A PLAN BOUNDARY, NOT A MASK EDGE.  The first version of
            # this keyed on where the conditioning mask changes, and with
            # --index-filler every frame is conditioned, so the mask never
            # changes, nothing was freed, and the output came back BIT-IDENTICAL
            # TO THE DRAFT on all three widths -- the trap atomic_completion's
            # own docstring records, reached from a different direction.  The
            # unit test missed it because its fixture had a gap in the mask,
            # which is exactly the case that does not occur in production.
            # Plan boundaries are also the definition --draft-seam-mask-width
            # uses on the training side, so the two now agree.
            if seam_labels is None and not seam_frames:
                raise ValueError(
                    "--completion-inpaint-seam-width needs the plan's labels or "
                    "the retrieval unit starts to locate seams; both are absent, "
                    "and falling back to mask edges silently keeps the whole "
                    "sequence (measured 2026-09-05: output identical to the draft)")
            conditioned = mask_batch[..., :1] > 0
            edge = torch.zeros_like(conditioned)
            edge[:, 1:] |= conditioned[:, 1:] != conditioned[:, :-1]
            if seam_labels is not None:
                label_batch = torch.stack([
                    _pad_labels(seam_labels[st:min(st + window_size, len(seam_labels))],
                                window_size)
                    for st in batch_starts]).to(device)
                edge[:, 1:] |= (label_batch[:, 1:] != label_batch[:, :-1]).unsqueeze(-1)
            for row, st in enumerate(batch_starts):
                for f in (seam_frames or ()):
                    local = int(f) - st
                    if 0 < local < window_size:
                        edge[row, local] = True
            width = max(1, int(inpaint_seam_width))
            # A SOFT ramp, not a boolean window.  Distance to the nearest plan
            # boundary, in frames, via a max-pool over a widening kernel: cheap,
            # and exact for the widths this runs at.  weight = 0 at the boundary
            # and 1 once the distance passes ``width``, with a raised-cosine in
            # between so the ramp has no corner of its own to inject.
            e = edge.to(dtype=torch.float32).squeeze(-1).unsqueeze(1)
            distance = torch.full_like(e, float(width))
            for d in range(width, 0, -1):
                if inpaint_seam_side == "after":
                    # Distance to the nearest seam AT OR BEFORE this frame only:
                    # a frame before a seam is measured from the previous one, so
                    # the arriving unit is held to its draft right up to the bar
                    # line and only the frames after it are regenerated.
                    hit = torch.nn.functional.max_pool1d(
                        torch.nn.functional.pad(e, (d, 0)), kernel_size=d + 1,
                        stride=1) > 0
                else:
                    hit = torch.nn.functional.max_pool1d(
                        e, kernel_size=2 * d + 1, stride=1, padding=d) > 0
                distance = torch.where(hit, torch.full_like(distance, float(d)), distance)
            at_edge = torch.nn.functional.max_pool1d(e, kernel_size=1, stride=1) > 0
            distance = torch.where(at_edge, torch.zeros_like(distance), distance)
            ramp = 0.5 - 0.5 * torch.cos(math.pi * (distance / float(width)).clamp(0.0, 1.0))
            if beat_free is not None:
                # --completion-beat-keep: also free the MIDDLE of every beat interval (per-frame weight, 1 = free),
                # so the draft's poses ON the beats are kept and the model re-draws only the travel between them.
                profile = torch.as_tensor(np.asarray(beat_free, dtype=np.float32))
                rows = []
                for st in batch_starts:
                    piece = profile[st:st + window_size]
                    if len(piece) < window_size:
                        piece = torch.nn.functional.pad(piece, (0, window_size - len(piece)))
                    rows.append(piece)
                ramp = torch.minimum(ramp, 1.0 - torch.stack(rows).to(ramp.device, ramp.dtype).unsqueeze(1))
            if inpaint_seam_side == "after":
                # A HARD edge on the kept side is a jump: the model's first free
                # frame is its own, the last kept one is the draft's.  So the last
                # SEAM_AFTER_LEAD frames before the bar line are loosened (0.25,
                # 0.75 for a lead of 2) -- the arrival itself, 3+ frames out,
                # stays held.
                lead = torch.ones_like(ramp)
                for d in range(SEAM_AFTER_LEAD, 0, -1):
                    ahead = torch.nn.functional.max_pool1d(
                        torch.nn.functional.pad(e, (0, d)), kernel_size=d + 1,
                        stride=1) > 0
                    lead = torch.where(
                        ahead,
                        torch.full_like(lead, 0.5 - 0.5 * math.cos(
                            math.pi * d / (SEAM_AFTER_LEAD + 1))),
                        lead)
                ramp = torch.minimum(ramp, lead)
            keep_mask = (ramp.squeeze(1).unsqueeze(-1)
                         * conditioned.to(dtype=torch.float32)).expand(
                             -1, -1, draft_windows[0].shape[-1])
        if keep_root:
            keep_mask = _hold_root_to_draft(
                keep_mask, mask_batch, draft_windows[0].shape[-1])
        generated_batch = completion.sample(
            torch.stack(music_windows).to(device),
            torch.stack(draft_windows).to(device),
            mask_batch,
            guidance_weight=guidance_weight,
            start_step=start_step,
            reproject_every=reproject_every,
            sample_steps=sample_steps,
            draft_guidance_weight=draft_guidance_weight,
            keep_mask=keep_mask,
            **extra,
        ).cpu()
        for offset, (start, valid_length, generated) in enumerate(
            zip(batch_starts, lengths, generated_batch)
        ):
            end = start + valid_length
            index = batch_start + offset
            weights = _blend_weights(
                valid_length,
                is_first=index == 0,
                is_last=index == len(starts) - 1,
                overlap=min(overlap, valid_length),
                blend_width=blend_width,
            )
            output[start:end] += generated[:valid_length] * weights
            weight_sum[start:end] += weights
    return output / weight_sum.clamp_min(1e-6)


def unnormalize_motion(motion, normalizer_path):
    normalizer = torch.load(str(normalizer_path), map_location="cpu")
    data_min = normalizer["data_min"].float()
    data_max = normalizer["data_max"].float()
    data_range = data_max - data_min
    # This is deliberately the exact inverse of
    # tools/apply_motion_normalizer.py.  A tiny but non-zero training range
    # remains a range; only an exactly constant dimension receives the
    # conventional unit safe range.  Generated/held-out values may be outside
    # [-1, 1], so do not clamp them before inversion.
    safe_range = torch.where(
        data_max == data_min,
        torch.ones_like(data_range),
        data_range,
    )
    return (motion + 1.0) * safe_range / 2.0 + data_min


def decode_motion(motion, normalizer_path):
    # Keep indexed retrieval importable in lightweight preprocessing/test
    # environments; these optional visual/rotation dependencies are only
    # required when a generated tensor is decoded for output.
    from dataset.quaternion import ax_from_6v
    from vis import SMPLSkeleton

    motion = unnormalize_motion(motion, normalizer_path)
    if motion.shape[1] != 151:
        raise ValueError("expected 151-D normalized motion, got {}".format(motion.shape[1]))
    contacts, values = torch.split(motion, (CONTACT_CHANNELS, 151 - CONTACT_CHANNELS), dim=-1)
    root_positions = values[:, :ROOT_POSITION_DIMS]
    rotations = ax_from_6v(values[:, ROOT_POSITION_DIMS:].reshape(-1, 24, 6))
    full_pose = SMPLSkeleton().forward(
        rotations.unsqueeze(0), root_positions.unsqueeze(0)
    )[0]
    return {
        "smpl_poses": rotations.reshape(-1, 72).numpy(),
        "smpl_trans": root_positions.numpy(),
        "full_pose": full_pose.numpy(),
        "contacts": contacts.numpy(),
    }


FLOOR_PERCENTILE = 5
FOOT_JOINTS = (7, 8, 10, 11)


def _body_speed_frames(joints, fps=30.0):
    """Per-frame mean joint speed relative to the pelvis (m/s), Gaussian-smoothed sigma 1.5 -- the 'how fast is the
    dancer moving' reading runs/ext_20260923/energy_follow.py judges with."""
    from scipy.ndimage import gaussian_filter1d
    j = gaussian_filter1d(np.asarray(joints, dtype=np.float64), 1.5, axis=0, mode="nearest")
    rel = j - j[:, :1]
    return np.r_[0.0, np.linalg.norm(np.diff(rel, axis=0), axis=-1).mean(-1)] * fps


IMAGE_JOINTS = (16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8)     # shoulders, elbows, wrists, hips, knees, ankles


def _image_speed_frames(joints, fps=30.0):
    """Per-frame body speed AS THE CAMERA SEES IT: the 12 limb joints projected on the image plane (world x, z; the
    camera looks along +y), relative to the hip centre, in torso lengths per second -- what ViTPose reads off the
    rendered skin (hit2d.py).  Validated 2026-09-27: its beat contrast tracks the driving 2D skeleton's at Spearman
    +0.72 over 12 arm-songs (whole-body 3D speed did not)."""
    from scipy.ndimage import gaussian_filter1d
    j = gaussian_filter1d(np.asarray(joints, dtype=np.float64), 1.5, axis=0, mode="nearest")[:, list(IMAGE_JOINTS)][:, :, [0, 2]]
    hip = j[:, 6:8].mean(1, keepdims=True)
    torso = float(np.median(np.linalg.norm(j[:, 0:2].mean(1) - hip[:, 0], axis=-1))) + 1e-9
    rel = (j - hip) / torso
    return np.r_[0.0, np.linalg.norm(np.diff(rel, axis=0), axis=-1).mean(-1)] * fps


def _beat_settle_contrast(speed, beats):
    """Mean over beats of [1 - slowest speed in (-3, +7) frames / fastest in (-10, +11)] at the beat, minus the same
    half a beat later (choreo_eval.arrive's reading, 30 fps); 0 without two beats."""
    beats = sorted(int(b) for b in beats)
    if len(beats) < 2:
        return 0.0
    v = np.asarray(speed, dtype=np.float64)
    half = int(round(float(np.median(np.diff(beats))) / 2.0))

    def settle(b):
        lo, hi = max(0, b - 10), min(len(v), b + 11)
        if hi - lo < 6:
            return None
        top = v[lo:hi].max()
        return None if top <= 1e-9 else 1.0 - v[max(0, b - 3):min(len(v), b + 7)].min() / top
    on = [x for x in (settle(b) for b in beats) if x is not None]
    off = [x for x in (settle(b + half) for b in beats) if x is not None]
    return float(np.mean(on) - np.mean(off)) if on and off else 0.0


def _action_point_frames(speed, lookback=10, drop=0.6):
    """Frames where a movement LANDS: a local minimum of body speed that sits below ``drop`` x the preceding
    ``lookback``-frame peak, that peak above the window's median speed -- the 'action points' the operator counts
    (the same definition runs/ext_20260923/energy_follow.py reads)."""
    v = np.asarray(speed, dtype=np.float64)
    med = float(np.median(v)) if len(v) else 0.0
    out = []
    for t in range(lookback, len(v) - 1):
        top = v[t - lookback:t].max()
        if v[t] <= v[t - 1] and v[t] <= v[t + 1] and top > med and v[t] < drop * top:
            out.append(t)
    return np.asarray(out, dtype=np.int64)


def _parse_energy_follow(text):
    """'keep=0.5,smooth=2,lo=0.15,hi=0.85,feature=loudness' -> dict (None when empty)."""
    if not text:
        return None
    spec = {"keep": 0.5, "smooth": 2, "lo": 0.15, "hi": 0.85, "feature": "loudness", "metric": "speed", "hit": 0.0}
    for item in str(text).split(","):
        key, _, value = item.partition("=")
        key = key.strip()
        if key not in spec:
            raise ValueError("--draft-energy-follow: unknown key {!r}".format(key))
        spec[key] = value.strip() if key in ("feature", "metric") else float(value)
    if spec["feature"] not in ("loudness", "onset"):
        raise ValueError("--draft-energy-follow: feature must be loudness or onset")
    if spec["metric"] not in ("speed", "points", "both"):
        raise ValueError("--draft-energy-follow: metric must be speed, points or both")
    return spec


def _energy_targets(track, bounds, spec):
    """The song's INTENSITY PLAN, per frame: each bar's loudness (channel 1, MFCC c0 -- the one intensity feature that
    predicts real dancers' energy within a song on train) or onset-peak density (channel 33), averaged over +-smooth
    bars so the plan follows sections (build-up, chorus) rather than single bars, ranked within THIS song, and mapped
    linearly onto [lo, hi] of the library's speed percentiles.  Frames outside the bar grid take the nearest bar."""
    channel = 1 if spec.get("feature", "loudness") == "loudness" else 33
    bars = [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if int(b) > int(a)]
    if len(bars) < 2:
        return None
    x = np.array([float(np.asarray(track)[a:b, channel].mean()) for a, b in bars])
    k = int(spec.get("smooth", 0))
    if k > 0:
        x = np.array([x[max(0, i - k):i + k + 1].mean() for i in range(len(x))])
    rank = (np.argsort(np.argsort(x, kind="stable"), kind="stable") + 0.5) / len(x)
    per_bar = float(spec["lo"]) + (float(spec["hi"]) - float(spec["lo"])) * rank
    tau = np.empty(len(track), dtype=np.float64)
    tau[:bars[0][0]] = per_bar[0]
    for (a, b), value in zip(bars, per_bar):
        tau[a:b] = value
    tau[bars[-1][1]:] = per_bar[-1]
    return tau


STEP_LOCK_LAG = 0.20   # beats: where a dancer's foot lands after the beat (train split; see --draft-step-lock-keep)


def _foot_plant_frames(joints, fps=30.0, v_hi=0.6, v_lo=0.25, lookback=12):
    """Frames where a foot LANDS: its world speed drops below ``v_lo`` m/s having been above ``v_hi`` within the
    previous ``lookback`` frames (a foot that travelled and stopped).  Feet 10/11, Gaussian-smoothed (sigma 1.5)
    first so reconstruction jitter is not footwork.  The same definition runs/ext_20260923/footwork_eval.py judges
    with -- one definition, so the filter and the ruler cannot drift."""
    from scipy.ndimage import gaussian_filter1d
    j = gaussian_filter1d(np.asarray(joints, dtype=np.float64), 1.5, axis=0, mode="nearest")
    out = []
    for foot in (10, 11):
        v = np.r_[0.0, np.linalg.norm(np.diff(j[:, foot], axis=0), axis=-1)] * fps
        for t in range(1, len(v)):
            if v[t] < v_lo <= v[t - 1] and v[max(0, t - lookback):t].max() >= v_hi:
                out.append(t)
    return np.array(sorted(out), dtype=np.int64)


def _step_lock_score(plants, native, target_length, beats, lag=STEP_LOCK_LAG):
    """Sum over the unit's foot landings, played into the slot as ``_values_at`` plays them (align_corners), of
    cos(2 pi (phase - lag)), the phase read on the slot's own beats (offsets inside the slot, one period added at
    each end).  A landing ``lag`` of a beat after a beat scores +1, half a beat off scores -1; no landings score 0."""
    beats = sorted(int(b) for b in beats)
    if len(plants) == 0 or len(beats) < 2:
        return 0.0
    period = float(np.median(np.diff(beats)))
    grid = np.array([beats[0] - period] + beats + [beats[-1] + period], dtype=np.float64)
    t = (np.asarray(plants, dtype=np.float64)) * (float(target_length) - 1.0) / max(float(native) - 1.0, 1.0)
    k = np.clip(np.searchsorted(grid, t, side="right") - 1, 0, len(grid) - 2)
    phase = (t - grid[k]) / (grid[k + 1] - grid[k])
    return float(np.cos(2.0 * np.pi * (phase - lag)).sum())


def _arm_stop_frames(joints):
    """[..., T, 24, 3] -> [..., T, 2, 4]: per arm (left, right) its wrist height above the shoulder
    (arm lengths), elbow angle (degrees), shoulder->wrist reach (arm lengths), and the wrist's
    body-frame speed into that frame (metres; pelvis-relative, yaw from the hip axis)."""
    import numpy as np
    j = np.asarray(joints, dtype=np.float64)
    hips = j[..., 2, :2] - j[..., 1, :2]
    yaw = -np.arctan2(hips[..., 1], hips[..., 0])
    c, s_ = np.cos(yaw), np.sin(yaw)
    out = []
    for sh, el, wr in ((16, 18, 20), (17, 19, 21)):
        u = j[..., sh, :] - j[..., el, :]
        v = j[..., wr, :] - j[..., el, :]
        arm = np.linalg.norm(u, axis=-1) + np.linalg.norm(v, axis=-1)
        cos = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-9)
        off = j[..., wr, :] - j[..., 0, :]
        bx = c * off[..., 0] - s_ * off[..., 1]
        by = s_ * off[..., 0] + c * off[..., 1]
        pos = np.stack([bx, by, off[..., 2]], axis=-1)
        speed = np.zeros(pos.shape[:-1])
        speed[..., 1:] = np.linalg.norm(np.diff(pos, axis=-2), axis=-1)
        out.append(np.stack([(j[..., wr, 2] - j[..., sh, 2]) / arm,
                             np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))),
                             np.linalg.norm(j[..., wr, :] - j[..., sh, :], axis=-1) / arm,
                             speed], axis=-1))
    return np.stack(out, axis=-2).astype(np.float32)


def _arm_fullness_frames(joints):
    """[..., T, 24, 3] SMPL joints (z up) -> [..., T, 3]: the higher wrist above its shoulder (arm
    lengths), the straighter elbow (degrees), the longer shoulder->wrist reach (arm lengths)."""
    import numpy as np
    j = np.asarray(joints, dtype=np.float64)
    def length(a, b):
        return np.linalg.norm(j[..., a, :] - j[..., b, :], axis=-1)
    def angle(a, b, c):
        u = j[..., a, :] - j[..., b, :]
        v = j[..., c, :] - j[..., b, :]
        cos = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-9)
        return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    arm_l = length(16, 18) + length(18, 20)
    arm_r = length(17, 19) + length(19, 21)
    raise_ = np.maximum((j[..., 20, 2] - j[..., 16, 2]) / arm_l, (j[..., 21, 2] - j[..., 17, 2]) / arm_r)
    elbow = np.maximum(angle(16, 18, 20), angle(17, 19, 21))
    reach = np.maximum(length(20, 16) / arm_l, length(21, 17) / arm_r)
    return np.stack([raise_, elbow, reach], axis=-1).astype(np.float32)


# Forward = (0, -1) in world xy is straight at the render camera:
# ``tools/render_avatar_video.py --view front`` puts it there, and ground truth
# agrees -- over every eval frame its body-forward y is -0.969 median, 87.7% of
# frames within 60 degrees of the lens.  So this is a property of the renderer
# plus the corpus's own shooting convention, not a constant fitted to anything.
FACING_CAMERA_YAW = -math.pi / 2.0
# How fast ``face_camera``'s slow correction may turn the body, in degrees
# per frame at 30 fps.  Ground truth's own worst frame-to-frame yaw change
# over the 20 T-line eval clips is 25.2 deg, so this cannot be the fastest
# thing on screen; see face_camera for why a limit is needed at all.
# Overridable so the slew can be ABLATED without editing the file.  It has to
# be ablatable because "the fix is in the code" is not the same claim as "the
# fix binds on this corpus": measured 2026-09-16, an arm regenerated with the
# slew in place produced a facing_camera record bitwise identical to one from
# before it (slow_degrees_p95 24.631221647149594 both sides), which only the
# ablation can tell apart from a no-op.
FACING_SLEW_DEGREES_PER_FRAME = float(
    os.environ.get("FACING_SLEW_DEGREES_PER_FRAME", "3.0"))
FACING_SMOOTH_SECONDS = 3.0
HIP_JOINTS = (1, 2)          # left hip, right hip


def _body_forward_yaw(joints):
    """Yaw of the body's facing, from the hip axis.  ``joints`` is [T, J, 3]."""
    import numpy as np
    hips = joints[:, HIP_JOINTS[1], :2] - joints[:, HIP_JOINTS[0], :2]
    forward = np.stack([hips[:, 1], -hips[:, 0]], axis=1)
    return np.arctan2(forward[:, 1], forward[:, 0])


def face_camera(result, strength, window_seconds=FACING_SMOOTH_SECONDS, fps=30.0):
    """Turn the dance back toward the camera without flattening its turning.

    WHY.  The operator, 2026-09-08, on the retrained rhythm planner: "画面的确有
    长时间背对或者侧对的情况".  Measured share of frames NOT facing the lens
    (body-forward y > -0.5, so profile counts, which is half of what was named):
    ground truth 12.3%, that arm 29.9%, and its MEDIAN longest continuous
    off-camera span 1.5 s against ground truth's 0.3 s.  The share is the
    smaller half of the complaint; the continuous span is "长时间".

    WHERE IT COMES FROM, measured before anything was changed: the DRAFT is
    already 32.3% off camera and the completion brings it to 29.9%, i.e. the
    completion slightly HELPS and the defect is upstream of it.
    ``--draft-facing-anchor`` cannot fix it either, because it pulls drift back
    toward the clip's own OPENING yaw and 30% of draft openings are themselves
    off camera against ground truth's 10% -- turning it up locks those in.  So
    the correction is applied here, to the finished clip, against a target that
    is known rather than estimated.

    TWO PARTS, CHARGED DIFFERENTLY, and that split is the point:

    * the clip's mean facing is put on the camera by ONE rotation about the
      world origin.  That is a rigid transform of the whole clip -- pose, root
      path and contacts all carried together -- so it changes nothing about the
      dance and costs exactly nothing.  Measured: off camera 29.9% -> 9.2% with
      foot skate unmoved at 1.142.
    * the remaining SLOW deviation -- the yaw error low-passed over
      ``window_seconds`` -- is pulled back by ``strength``, about the PER-FRAME
      ROOT so the body turns in place.  Turns shorter than the window pass
      through untouched, which is what keeps the dance turning: total turning
      stays at 1054 deg against ground truth's 919 even at strength 0.35.
      This part is NOT free (it decouples facing from travel), and its cost is
      foot skate: 1.142 -> 1.147 at strength 0.15, -> 1.185 at 0.35.

    WHAT IT DELIBERATELY DOES NOT DO: drive off-camera time to zero.  Ground
    truth is 12.3%, a dancer who never turns is its own defect, and every
    setting here is chosen to land NEAR that number rather than under it.

    Returns a new result dict; the input is not mutated.
    """
    import numpy as np
    from dataset.rotation_ops import axis_angle_to_matrix, matrix_to_axis_angle

    joints = np.asarray(result["full_pose"], dtype=np.float64)
    trans = np.asarray(result["smpl_trans"], dtype=np.float64)
    poses = np.asarray(result["smpl_poses"], dtype=np.float64)

    def rotate_xy(points, angle, pivot=None):
        cos, sin = np.cos(angle), np.sin(angle)
        if np.ndim(angle):
            cos, sin = cos[:, None], sin[:, None]
        base = 0.0 if pivot is None else pivot
        x = points[..., 0] - (base[..., 0] if pivot is not None else 0.0)
        y = points[..., 1] - (base[..., 1] if pivot is not None else 0.0)
        out = points.copy()
        out[..., 0] = (base[..., 0] if pivot is not None else 0.0) + cos * x - sin * y
        out[..., 1] = (base[..., 1] if pivot is not None else 0.0) + sin * x + cos * y
        return out

    yaw = _body_forward_yaw(joints)
    mean_yaw = float(np.arctan2(np.sin(yaw).mean(), np.cos(yaw).mean()))
    rigid = float(FACING_CAMERA_YAW - mean_yaw)

    # Rigid: about the world origin, so smpl_trans travels with the body.
    joints = rotate_xy(joints, rigid)
    trans = rotate_xy(trans, rigid)

    slow = np.zeros(len(joints))
    if strength:
        unwrapped = np.unwrap(_body_forward_yaw(joints))
        width = max(3, int(round(float(window_seconds) * fps)) | 1)
        padded = np.pad(unwrapped, (width // 2, width // 2), mode="edge")
        smoothed = np.convolve(padded, np.ones(width) / width,
                               mode="valid")[:len(unwrapped)]
        error = smoothed - FACING_CAMERA_YAW
        # THE WRAP IS RIGHT; ITS DISCONTINUITY IS THE BUG.  ``smoothed``
        # low-passes the UNWRAPPED yaw, so it is continuous and routinely
        # leaves (-pi, pi].  Wrapping the error per sample is CORRECT -- facing
        # is periodic and a clip 350 degrees off camera should be turned the
        # short way -- but it makes the error JUMP BY 2*pi wherever the smoothed
        # yaw passes exactly away from the lens, and the correction jumped with
        # it by ``2*pi*strength``: 252 degrees at the shipped strength 0.7,
        # which an unwrapped reading folds to 108.  That is the
        # "人体旋转的跳变,视觉上看着不连续,缺帧" reported 2026-09-14; measured over
        # the 20 eval clips against a line ground truth never crosses (its worst
        # frame is 25.2 deg/frame), ground truth had 0 spikes above 30 and the
        # shipped arm had 12, worst 115 -- in ONE frame.
        error = np.arctan2(np.sin(error), np.cos(error))
        desired = -float(strength) * error

        # WHY NOT JUST UNWRAP THE ERROR.  Tried first, and it trades a visible
        # defect for a worse one: unwrapping makes the correction UNBOUNDED, so
        # a clip whose dancer turns several times accumulates it without limit.
        # Measured on 938161 -- slow_degrees_p95 reached 264.9 and the body
        # ended up 80.0% off camera, against ground truth's 12.3%, while the
        # spike it was meant to remove survived (3 spikes, worst 111).  The jump
        # is a genuine BIFURCATION, not an artefact: at exactly-away, "turn left
        # by 126 deg" and "turn right by 126 deg" are equally good and 252 deg
        # apart.
        #
        # So the wrapped target is kept -- bounded by ``strength * pi`` by
        # construction -- and the correction is SLEW-LIMITED, which is the only
        # thing the defect actually asks for: the body may be turned, it may not
        # be snapped.  The limit is derived, not chosen: ground truth's own
        # worst frame-to-frame yaw change over the 20 eval clips is 25.2 deg, so
        # a correction that moves at most 3 deg/frame can never be the fastest
        # thing on screen, and it crosses the 252-degree bifurcation over about
        # 2.8 s -- which reads as the dancer turning, because that is what it is.
        max_step = math.radians(FACING_SLEW_DEGREES_PER_FRAME) * (30.0 / float(fps))
        slow = np.empty_like(desired)
        slow[0] = desired[0]
        for index in range(1, len(desired)):
            slow[index] = slow[index - 1] + float(np.clip(
                desired[index] - slow[index - 1], -max_step, max_step))
        # About the per-frame root: the body turns, the root path does not move,
        # so smpl_trans is untouched here and full_pose[:, 0] stays bitwise it.
        joints = rotate_xy(joints, slow, pivot=joints[:, :1, :2])

    total = rigid + slow
    # The renderer draws smpl_poses/smpl_trans and uses full_pose only as a
    # GATE, so the global orient has to carry exactly the same rotation or the
    # mesh and the joints disagree and it refuses to render.
    orient = torch.as_tensor(poses[:, :3], dtype=torch.float64)
    spin = torch.zeros(len(total), 3, dtype=torch.float64)
    spin[:, 2] = torch.as_tensor(total, dtype=torch.float64)
    composed = axis_angle_to_matrix(spin) @ axis_angle_to_matrix(orient)
    poses = poses.copy()
    poses[:, :3] = matrix_to_axis_angle(composed).numpy()

    after = _body_forward_yaw(joints)
    off = float((np.sin(after) > -0.5).mean())
    result = dict(result)
    result["full_pose"] = joints.astype(np.asarray(result["full_pose"]).dtype)
    result["smpl_trans"] = trans.astype(np.asarray(result["smpl_trans"]).dtype)
    result["smpl_poses"] = poses.astype(np.asarray(result["smpl_poses"]).dtype)
    result["facing_camera"] = {
        "strength": float(strength), "window_seconds": float(window_seconds),
        "rigid_degrees": float(np.degrees(rigid)),
        "slow_degrees_p95": float(np.degrees(np.percentile(np.abs(slow), 95))),
        "off_camera_share": off,
    }
    return result


def release_floor_state(data_root):
    """``(levelled, reference_metres)`` for a release, from its own build.json.

    WHY THIS EXISTS.  Once the corpus is levelled at the source
    (``tools/level_release_floors.py``), every downstream repair that also moves
    the body vertically is a SECOND levelling, and the operator named that risk
    when asking for the source fix: "后处理相关的兜底同步避免2次拉平带来的副作用".
    Two of them can fire:

    * ``--draft-floor-normalize`` subtracts each prototype's own recording floor
      at retrieval.  On a levelled release that floor is already zero by
      construction and subtracting it again moves every prototype by the
      residual, so it is REFUSED rather than quietly doubled.
    * ``--floor-anchor`` shifts the finished clip so its 5th-percentile foot
      lands on a target.  That is still wanted -- the renderer stands every arm
      on the REFERENCE clip's floor, and ground-truth floors themselves run
      0.264 to 0.469 m -- but its target must now agree with what the corpus was
      levelled to, or the anchor is undoing the levelling one clip at a time.

    Returns ``(False, None)`` for a release that predates the levelling, so
    every earlier artifact reproduces unchanged.
    """
    build = Path(data_root) / "build.json"
    if not build.exists():
        return False, None
    try:
        payload = json.loads(build.read_text())
    except (OSError, ValueError):
        return False, None
    return bool(payload.get("floor_levelled")), payload.get("floor_reference_m")


def fix_foot_skate(result, strength=1.0, threshold=0.5, smooth=5, fps=30.0,
                   seam_frames=None):
    """Stop the planted foot from sliding, by moving the BODY instead.

    WHY.  The operator, 2026-09-11 on the rhythm-planner renders, and again
    2026-09-12: "脚步时不时有滑动 ... 脚滑也交给后处理来修".  Measured by
    ``tools/score_arm_table.foot_skate`` (mean horizontal foot speed over the
    frames where that foot is at its own ground level): ground truth 0.295 m/s,
    the shipped arm 0.420, and the arms that buy reach are worse still -- 0.471
    with whole units, 0.556 with the seam-aware ranking off.  Ground truth is
    the calibration, not zero: wild reconstructions skate a little and a dancer
    whose feet never move is its own defect.

    THE SIGNAL IS THE MODEL'S OWN CONTACT CHANNELS, not the metric's "at its own
    ground level" test.  Driving the repair from the criterion that scores it
    would make the score improve by construction (CLAUDE.md 2.1), and the four
    contact channels -- ankles and toes, joints 7, 8, 10, 11 -- are an
    independent statement about when a foot is down.

    WHAT IT CHANGES AND WHAT IT CANNOT.  Only a per-frame translation of the
    whole body: ``smpl_poses`` is not touched at all, so every joint angle, every
    limb trajectory relative to the root, and therefore the dance itself is
    bit-identical.  What moves is where the body is, which is the only thing
    that can be wrong when a planted foot slides.  ``full_pose`` and
    ``smpl_trans`` take the same shift so the renderer's millimetre joint gate
    still passes.

    WHY IT IS SMOOTHED.  The correction is the negative cumulative sum of the
    planted foot's own displacement, so it is exactly as smooth as that foot --
    except at a contact on/off transition, where the set of feet being averaged
    changes in one frame and steps the correction.  A raised-cosine low-pass
    over ``smooth`` frames removes that step; the guard that it has not merely
    traded skate for judder is the ``jitter`` column, ground truth 0.094.

    ``strength`` scales the correction, 0 reproduces the input exactly, and the
    clip's mean position is restored afterwards so this cannot walk the dancer
    out of the frame ([[render align_row_positions]] handles the rest).

    ``seam_frames`` IS THE REPAIR FOR A SLIDE THIS FUNCTION MANUFACTURES.
    ``_blend_draft_seams`` cross-fades the POSE channels across +-N frames at
    every bar seam while the contact channels stay on right through the ramp
    (measured 2026-09-16: 1.6-1.9 feet flagged down-on-both-frames inside the
    ramp, the same as the 1.6 outside it).  The rule above -- "a foot down on
    both frames that moved is sliding" -- then reads that pose morph as slide
    and charges every millimetre of it to the root.  Measured on the fixed ten
    by recovering the strength-0 path exactly from two arms that differ only in
    ``fix_skate`` (0.5 and 1.0, pose channels bit-identical, so the correction
    is linear and root0 = 2*trans_05 - trans_10): pelvis path 4.94 m at
    strength 0 against ground truth's 4.85 (+0.09, 5/10 -- FLAT) and 6.36 at
    strength 1.0 (+1.42 against strength 0, 10/10, P=0.002).  74.4% of the
    correction's own path falls inside +-8 frames of a seam, which is 28.4% of
    frames -- 2.6x over-represented -- and its per-frame profile peaks at
    16.4 mm/frame three frames from a seam against a 1.5 mm/frame background
    twelve frames away.

    Frames listed here are treated as NOT PLANTED, so no correction accrues
    across the ramp.  It does not disable the repair elsewhere, which is the
    point: the operator asked for foot skate to be fixed in post
    ("脚滑也交给后处理来修"), and ground truth skates 0.295 m/s rather than zero,
    so the answer is to stop charging a rendering artefact to the root, not to
    stop correcting real slide.  The cost is that genuine slide inside a seam
    window goes uncorrected; both columns have a ground-truth reading and both
    must be reported.
    """
    import numpy as np

    contacts = np.asarray(result.get("contacts"))
    if contacts is None or contacts.ndim != 2 or contacts.shape[1] != len(FOOT_JOINTS):
        raise ValueError(
            "--fix-foot-skate needs the {} contact channels this pipeline "
            "writes; got {}".format(len(FOOT_JOINTS), getattr(contacts, "shape", None)))
    joints = np.asarray(result["full_pose"], dtype=np.float64)
    trans = np.asarray(result["smpl_trans"], dtype=np.float64)
    frames = min(len(joints), len(contacts))
    planted = contacts[:frames] > float(threshold)
    masked = 0
    if seam_frames is not None and len(seam_frames):
        inside = np.zeros(frames, bool)
        keep = np.asarray(seam_frames, dtype=int)
        keep = keep[(keep >= 0) & (keep < frames)]
        inside[keep] = True
        masked = int(inside.sum())
        planted = planted & ~inside[:, None]

    step = np.zeros((frames, 2))
    for index, joint in enumerate(FOOT_JOINTS):
        travel = np.diff(joints[:frames, joint, :2], axis=0)
        # A foot counts only while it is down on BOTH frames of the step, or the
        # frame it lands on would be read as a slide of however far it swung.
        down = planted[1:frames, index] & planted[:frames - 1, index]
        weight = down.astype(float)[:, None]
        step[1:] += -travel * weight
        if index == 0:
            counts = np.zeros((frames, 1))
        counts[1:] += weight
    step[1:] /= np.maximum(counts[1:], 1.0)

    correction = np.cumsum(step, axis=0) * float(strength)
    width = max(1, int(smooth) | 1)
    if width > 1:
        kernel = np.hanning(width + 2)[1:-1]
        kernel = kernel / kernel.sum()
        padded = np.pad(correction, ((width // 2, width // 2), (0, 0)), mode="edge")
        correction = np.stack([np.convolve(padded[:, axis], kernel, mode="valid")[:frames]
                               for axis in (0, 1)], axis=1)
    correction -= correction.mean(axis=0)

    joints = joints.copy()
    trans = trans.copy()
    joints[:frames, :, :2] += correction[:, None, :]
    trans[:frames, :2] += correction

    result = dict(result)
    result["full_pose"] = joints.astype(np.asarray(result["full_pose"]).dtype)
    result["smpl_trans"] = trans.astype(np.asarray(result["smpl_trans"]).dtype)
    result["foot_skate_fix"] = {
        "strength": float(strength),
        "threshold": float(threshold),
        "smooth": int(width),
        "planted_frame_share": float(planted.any(axis=1).mean()),
        "correction_p95_m": float(np.percentile(np.linalg.norm(correction, axis=1), 95)),
        "correction_max_m": float(np.linalg.norm(correction, axis=1).max()),
        # Counted at the point of use: "the flag was set" and "frames were
        # actually masked" are different claims (DEFECTS 77).
        "seam_frames_masked": masked,
    }
    return result


def anchor_floor(result, height):
    """Stand the clip on a floor at ``height`` metres, by one rigid shift in z.

    WHY, and it is a measurement.  ``tools/render_avatar_video.py`` deliberately
    puts every arm on the REFERENCE's floor (its header: giving each arm its own
    floor would "delete the very defect this render was asked to show").  So a
    generated clip whose absolute standing height differs from ground truth's is
    drawn hovering above the checkerboard or sunk through it, and on 2026-09-07
    the operator named exactly that ("双脚凌空抬起").  Measured over the 20 eval
    clips, floor = 5th percentile of the per-frame lowest foot joint:

        ground truth   median 0.342 m   p5-p95 spread 0.047 m
        generated      median 0.310 m   p5-p95 spread 0.454 m   corr with GT -0.021

    Ground truth is tightly clustered and the generated height is essentially
    random -- ten times the spread and no correlation.  On
    ``7610414564962183545`` the generated body stands 0.196 m too high for the
    whole clip, which is visible as shoes floating over the floor tiles.

    TWO CAUSES, and this fixes the one that matters.  The clip is already wrong
    at frame 0 (first-second floor spread 0.290 m against ground truth's 0.047,
    correlating r=0.664 with the final error), AND it drifts, because
    ``build_draft``'s root continuity offsets each retrieved unit onto the
    previous unit's last frame (``values[:, columns] += previous_root -
    values[0, columns]``) with nothing pulling it back -- the identical
    "continuity is not an anchor" failure the yaw had before
    ``--draft-facing-anchor``.  But the drift WITHIN a clip is not the defect:
    measured the same day, the clip-internal range of the rolling lowest foot is
    0.231 m for ground truth against 0.213 for the generated arm, and ground
    truth drifts as much.  What is wrong is the constant offset, so the fix is a
    single rigid translation and NOT a per-frame correction -- a per-frame one
    would flatten vertical motion ground truth genuinely has.

    Because it is rigid, it cannot change the dance: every joint moves by the
    same vector, so all root-relative motion, every velocity and every contact
    is bit-identical.  Only where the body stands changes.

    ``height`` must come from the TRAINING corpus, never from the clip's own
    ground truth -- otherwise the anchor smuggles the answer into inference and
    could not run on a new song.
    """
    import numpy as np

    joints = np.asarray(result["full_pose"])
    lowest = joints[:, FOOT_JOINTS, 2].min(axis=1)
    # Percentile, not min: one frame of a foot punched through the floor would
    # otherwise define the plane, and wild reconstructions do that
    # (tools/build_dance_gallery.py:358).
    current = float(np.percentile(lowest, FLOOR_PERCENTILE))
    shift = float(height) - current
    result = dict(result)
    result["full_pose"] = joints + np.array([0.0, 0.0, shift], joints.dtype)
    result["smpl_trans"] = (np.asarray(result["smpl_trans"])
                            + np.array([0.0, 0.0, shift], np.asarray(result["smpl_trans"]).dtype))
    result["floor_anchor"] = {"target": float(height), "measured": current,
                              "shift": shift, "percentile": FLOOR_PERCENTILE}
    return result


def _audio_map(audio_dir):
    """Name -> audio source, which may be a WAV or an already-extracted 35-D array.

    Accepting ``.npy`` is not a convenience.  On AIST the released music features
    and a fresh extraction agree closely enough that either works; on the wild
    corpus there is no local WAV at all, and re-extracting from the ingest
    tree's audio would put the planner's *input* on a different extractor run
    than its *training data*.  ``tools/convert_aistpp_official.py`` measured what
    that costs on the one corpus where both exist: onset frames land identically
    and the beat channel correlates **0.36**.  Feeding one run at training and
    another at inference splits the corpus along exactly the axis the planner is
    supposed to read, and nothing would raise.

    So a corpus whose music identity *is* the stored array is read as the stored
    array.  A stem carrying both forms is refused rather than resolved by
    precedence: which one was used would then depend on a rule nobody stated.
    """
    mapping = {}
    for suffix in ("*.wav", "*.npy"):
        for path in sorted(Path(audio_dir).rglob(suffix)):
            if path.stem in mapping:
                raise ValueError("duplicate audio basename: {} and {}".format(
                    mapping[path.stem], path))
            mapping[path.stem] = path
    if not mapping:
        raise FileNotFoundError("no WAV or 35-D .npy files under {}".format(audio_dir))
    return mapping


def _aist_basename(name):
    fields = name.split("_")
    for index, field in enumerate(fields):
        if field.startswith("g") and len(field) == 3:
            return "_".join(fields[index:])
    return name


def _match_audio(name, audio):
    for candidate in (name, _aist_basename(name)):
        if candidate in audio:
            return audio[candidate]
    raise FileNotFoundError("no matching WAV for {}".format(name))


def _oracle_target_frames(path, raw_fps=60, output_fps=30):
    """Read target motion length for an explicitly ORACLE-only protocol."""
    with open(str(path), "rb") as handle:
        data = pickle.load(handle)
    if "full_pose" in data:
        return len(data["full_pose"])
    for key in ("smpl_poses", "q"):
        if key in data:
            return int(round(len(data[key]) * output_fps / float(raw_fps)))
    raise KeyError("cannot determine motion length from {}".format(path))


def _load_music(path, frames=None, feature_dim=None):
    """The 35-D music conditioning for one query, extracted or read back.

    ``.npy`` is read verbatim -- never re-derived, never resampled.  The whole
    point of that branch is that the array *is* the corpus's music identity, so
    touching it would reintroduce the second extractor run it exists to avoid.
    """
    path = Path(path)
    if path.suffix == ".npy":
        features = np.load(str(path))
        if features.ndim != 2:
            raise ValueError("{}: music features must be [frames, dim], got {}".format(
                path, features.shape))
        features = np.ascontiguousarray(features, dtype=np.float32)
    else:
        audio, _ = librosa.load(str(path), sr=SR)
        features = extract_audio(audio, path.stem, max_frames=None).astype(np.float32)
    if feature_dim is not None and features.shape[1] != feature_dim:
        raise ValueError("{}: music is {}-D, the checkpoint expects {}-D".format(
            path, features.shape[1], feature_dim))
    if frames is not None:
        features = features[:frames]
        if len(features) < frames:
            features = np.pad(features, ((0, frames - len(features)), (0, 0)))
    return torch.from_numpy(features)


DEFAULT_INGEST_ROOT = "data/wild_ingest_v1"
# Read off the distribution, not chosen -- see ``check_music_span``.
MUSIC_SPAN_TOLERANCE = 0.02


def _ingest_wav_for(name, ingest_root):
    """``wild_v4:<upload>:clipNNN`` -> that clip's own wav in the ingest tree.

    Returns None for a name this corpus does not use -- AIST sequence ids, for
    one -- rather than guessing a path, because a guessed path that does not
    exist and a name the check does not apply to must not land in the same
    bucket of the report.
    """
    parts = str(name).split(":")
    if len(parts) != 3:
        return None
    candidate = Path(ingest_root) / "{}__{}".format(parts[1], parts[2]) / "audio.wav"
    return candidate if candidate.is_file() else None


def _audio_seconds(path):
    """Duration from the header; no decode, and no re-derivation of anything."""
    try:
        import soundfile
        info = soundfile.info(str(path))
        return info.frames / float(info.samplerate)
    except Exception:
        return float(librosa.get_duration(path=str(path)))


def check_music_span(items, ingest_root=DEFAULT_INGEST_ROOT, policy="refuse", fps=30):
    """Does each query's music array cover the stretch of time its clip holds?

    The invariant every consumer downstream rides on is that ``music_35[i]`` is
    the music at motion frame ``i`` at ``fps``.  Nothing in this file could see
    it break, because the array is self-consistent whatever span it covers: the
    generated length IS the music length, so a music array cut from a different
    generation of the corpus produces a generation of exactly that length and
    every internal check agrees with itself.

    Measured 2026-08-24 on ``runs/wild_v4_acct_gt_eval``: 176 of 1,575 clips
    (11.2%) carry a music array whose span disagrees with their own clip's wav
    by more than 2% -- 86 of them by exactly 4/3, 52 by 5/6.  Those clips'
    ground-truth motion is measurably less locked to their own beat grid than
    the rest (per-clip alignment peak minus a phase-destroyed null: +0.0103
    against +0.0226, difference +0.0123 with a 95% bootstrap interval of
    [+0.0065, +0.0213]).  The rate is NOT the problem and this gate does not
    test it: the beat channel's implied tempo matches the wav's own tempo in
    both groups (0.9951 against a control's 1.0062, where a wrong clock would
    read 1/span = 0.754).  What differs is which stretch of audio the array is.

    The tolerance is relative and is read off the measured distribution rather
    than chosen: over those 1,575 clips ``|span-1|`` is 0.0022 at the median and
    0.0036 at p75 -- audio containers carry a little more than an exact frame
    count, so a one-frame absolute tolerance flags 46 of 65 M6 clips and the
    gate gets switched off within a day.  The defect family starts at 0.111
    (p90).  Between thresholds 0.02 and 0.05 the count moves 176 -> 174, so the
    threshold is sitting in a plateau and is not what decides the verdict; 0.02
    is the low end of that plateau.  A two-frame absolute floor rides along so
    a very short clip is not flagged for a rounding.

    Buckets, kept apart on purpose:
      ``ok``             span agrees to within one frame
      ``mismatch``       it does not -- the thing this gate exists to stop
      ``no_wav``         the clip has no local wav, so nothing was checked
      ``not_applicable`` the query's audio IS a wav, so the span is its own
    A run whose clips are mostly ``no_wav`` has not passed this gate, it has
    skipped it, and the report says so rather than reporting a clean pass.
    """
    if policy not in ("refuse", "warn", "off"):
        raise ValueError("music span policy must be refuse/warn/off, got {!r}".format(policy))
    report = {"policy": policy, "fps": fps,
              "tolerance_relative": MUSIC_SPAN_TOLERANCE,
              "tolerance_floor_frames": 2,
              "ok": 0, "mismatch": 0, "no_wav": 0, "not_applicable": 0,
              "mismatched": []}
    if policy == "off":
        report["checked"] = False
        return report
    report["checked"] = True
    for name, audio_path, _ in items:
        if Path(audio_path).suffix != ".npy":
            report["not_applicable"] += 1
            continue
        wav = _ingest_wav_for(name, ingest_root)
        if wav is None:
            report["no_wav"] += 1
            continue
        frames = int(np.load(str(audio_path), mmap_mode="r").shape[0])
        music_seconds = frames / float(fps)
        wav_seconds = _audio_seconds(wav)
        allowed = max(MUSIC_SPAN_TOLERANCE * wav_seconds, 2.0 / fps)
        if abs(music_seconds - wav_seconds) <= allowed:
            report["ok"] += 1
            continue
        report["mismatch"] += 1
        report["mismatched"].append({
            "name": name, "music_frames": frames,
            "music_seconds": round(music_seconds, 4),
            "wav_seconds": round(wav_seconds, 4),
            "ratio": round(music_seconds / wav_seconds, 4) if wav_seconds else None,
        })
    if report["mismatch"]:
        head = ", ".join("{} ({:.3f}x)".format(m["name"], m["ratio"] or float("nan"))
                         for m in report["mismatched"][:5])
        message = ("music span disagrees with the clip's own audio on {} of {} "
                   "query(ies): {}{}. The generated length is the music length, "
                   "so these would be scored against a different cut of their own "
                   "recording. Rebuild the eval set, or pass "
                   "--music-span-check warn to record it and continue.".format(
                       report["mismatch"],
                       report["mismatch"] + report["ok"] + report["no_wav"], head,
                       " ..." if len(report["mismatched"]) > 5 else ""))
        if policy == "refuse":
            raise ValueError(message)
        print("WARNING: " + message)
    if report["no_wav"]:
        print("music span gate: {} query(ies) had no local wav and were NOT "
              "checked".format(report["no_wav"]))
    return report


def _query_retrieval_group_id(library, name, derive_missing=False):
    """Resolve an inference query through the explicit sidecar registry.

    ``derive_missing`` falls back to the clip name's own recording prefix when
    the registry has no entry.  It is OFF by default because the fail-closed
    rule -- "an input without an explicit retrieval group cannot safely prove
    that a training prototype is external" -- is what keeps a query's own motion
    out of its draft.

    WHY IT IS NEVERTHELESS SAFE HERE, checked rather than assumed
    (2026-09-06, T line):

      * the registry's mapping is trivial: **0 of its 6,823 entries** map to
        anything other than ``name.rsplit(":", 1)[0]``, and **0 recordings**
        have clips in more than one group, so the derived id is the one the
        registry would have given;
      * and the exclusion it drives is a no-op for every eval clip anyway --
        the library holds **143** retrieval groups and **none of the 20 eval
        recordings is among them**, including the 18 that DO have registry
        entries.

    WHAT IT COSTS TO LEAVE OFF.  Two of the twenty eval clips
    (7610414564962183545, 7188505181892381984) have no registry entry, so
    ``_source_safe_draft`` returns an all-zero draft and they are generated from
    music alone -- which is what the operator saw on exactly those two as
    身体腾空旋转 and 动作段落高度重复.  Their artifacts say so:
    ``safe_draft_condition_fraction: 0.0``.
    """
    found = library.query_retrieval_group_id(name)
    if found or not derive_missing:
        return found
    prefix = str(name).rsplit(":", 1)[0]
    return prefix or None


def _grid_phase(beat_grid, frame):
    """Where ``frame`` sits inside its beat, and the grid's period, in frames.

    Returns ``(None, None)`` when there is no usable grid, which is what makes
    every rule other than ``phase`` reproduce byte-for-byte.
    """
    if beat_grid is None or len(beat_grid) < 3:
        return None, None
    grid = np.asarray(beat_grid)
    # ``side="right"``, so a frame landing exactly ON a beat reads phase 0.0 and
    # not 1.0.  The bar grid puts plan boundaries on beats deliberately, so that
    # is the most common case, not an edge one.  The wrapped distance below
    # happens to make 1.0 and 0.0 equivalent, which is exactly why this was
    # invisible until a unit test asked for the number itself.
    index = int(np.searchsorted(grid, frame, side="right")) - 1
    period = float(np.median(np.diff(grid)))
    if index < 0 or index >= len(grid) - 1 or grid[index + 1] <= grid[index]:
        return None, period
    span = float(grid[index + 1] - grid[index])
    return float(frame - grid[index]) / span, period


def _selector_onset(music):
    """Clip-wide z-scored onset envelope for the learned rule, or ``None``.

    Thin wrapper so the inference path and the trainer call the SAME
    implementation; see ``model.retrieval_selector.onset_z``.
    """
    from model.retrieval_selector import onset_z

    return onset_z(music)


def beat_grid_of(music, channel=34):
    """The query's beat frames, from the same channel the planner reads."""
    array = music.numpy() if hasattr(music, "numpy") else np.asarray(music)
    if array.ndim != 2 or array.shape[1] <= channel:
        return None
    grid = np.flatnonzero(array[:, channel] > 0.5)
    return grid if len(grid) >= 3 else None


def bar_bounds_of(music, beats_per_segment, phase, channel=34):
    """The frames where a bar starts, i.e. where the plan grid put its lines.

    Same construction as ``snap_plan_to_bar_grid``: every ``beats_per_segment``
    beats from ``phase``.  Returned so retrieval can be made to start a new
    prototype at each one, instead of letting two same-labelled bars merge into
    a single long segment that one prototype must be stretched across.
    """
    from dataset.bar_tokens import bar_lines

    grid = beat_grid_of(music, channel=channel)
    if grid is None or phase is None or beats_per_segment < 1:
        return None
    return bar_lines(grid, beats_per_segment, phase) or None


def quiet_bar_bounds(bounds, music, slide=0, channel=0):
    """Slide each retrieval cut to the QUIETEST frame near it.

    WHY THE CUT SHOULD MOVE OFF THE BEAT.  The plan's bar line is where the
    music is loudest: measured over 194 bars of the eval clips, the strongest
    onset inside a bar sits within 3 frames of the bar line **44.8%** of the
    time, and those frames are only about 12% of a bar -- 3.7x chance.  Our
    retrieval seam is on that line 100% of the time, and a seam is a velocity
    step: measured the same day, the top-10% speed changes of the final output
    are **21.4%** within 3 frames of a bar line (the raw draft 40.0%) against
    ground truth's 13.7%, which is chance because ground truth has no seams.

    So the pipeline puts its largest artifact exactly where the music is loudest
    and the viewer is most attending.  This moves the CUT, not the content:
    the plan still says what to dance over which bar, and the boundary slides by
    at most ``slide`` frames onto a local minimum of the onset envelope so the
    join lands in a gap in the music instead of on the hit.

    NOT a warp.  Warping content onto the beat is refuted four times over
    (``--draft-music-anchor`` at lag 0 and 0.2 s, ``--draft-beat-anchor`` whole
    body and per limb, ``--retrieval-rule phase``), and there is a constructive
    reason: one monotone time map settles every joint of a chain at the same
    instants.  Moving a boundary is a different operation -- each slot simply
    covers different frames, and retrieval fills it with whatever fits.

    ``slide`` of 0 returns the bounds unchanged, so earlier artifacts reproduce.
    The first and last bounds are never moved: they are the clip's ends.
    """
    if not slide or bounds is None or len(bounds) < 3:
        return bounds
    envelope = np.asarray(torch.as_tensor(music)[:, channel], dtype=np.float64)
    moved = [int(bounds[0])]
    for previous, cut, following in zip(bounds[:-2], bounds[1:-1], bounds[2:]):
        low = max(int(previous) + 1, int(cut) - int(slide))
        high = min(int(following) - 1, int(cut) + int(slide) + 1)
        if high <= low:
            moved.append(int(cut))
            continue
        window = envelope[low:high]
        # Ties go to the frame nearest the original cut, so a flat stretch of
        # silence does not drag every bar to the same edge of its window.
        order = np.lexsort((np.abs(np.arange(low, high) - int(cut)), window))
        moved.append(int(low + order[0]))
    moved.append(int(bounds[-1]))
    # Monotone by construction above, but asserted: a non-increasing cut list
    # would make labels_to_segments emit an empty or reversed span in silence.
    for a, b in zip(moved[:-1], moved[1:]):
        if b <= a:
            return bounds
    return moved


def music_settle_period(music, fps=30.0):
    """The query's own beat period in frames, from the music's beat channel.

    Channel 34 is the beat indicator this repository's 35-D features carry
    (``tools/run_m6_wild.sh`` and ``dataset/global_music`` both read it there).
    The median inter-beat interval is used rather than the mean because a
    missed beat doubles one gap and would drag a mean by half a period.
    Returns ``nan`` when the track carries too few beats to state one, and the
    tempo rule then degrades to the duration rule rather than inventing a
    number.
    """
    music = np.asarray(music)
    if music.ndim != 2 or music.shape[1] <= 34:
        return float("nan")
    beats = np.flatnonzero(music[:, 34] > 0.5)
    if len(beats) < 3:
        return float("nan")
    return float(np.median(np.diff(beats)))


def _release_label_space(data_root):
    """``label_space_id`` from a release's build.json, or ``None``."""
    try:
        with open(str(Path(data_root) / "build.json")) as handle:
            return json.load(handle).get("label_space_id")
    except (OSError, ValueError):
        return None


def _unit_floor_provenance(path):
    """The census sidecar that says which release a --draft-unit-floor file was built from."""
    sidecar = Path(str(path)).with_suffix(".json")
    try:
        with open(str(sidecar)) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _source_safe_draft(library, labels, feature_dim, query_retrieval_group_id,
                       unsourced_retrieval=False, root_continuity="off",
                       gap_fill="zero", seam_blend=0, seam_window="triangle",
                       seam_stagger=False, seam_transition="centred",
                       seam_aware_retrieval=False,
                       root_velocity_blend=0,
                       root_seam_smooth=0,
                       facing_anchor=0.0,
                       target_period=None,
                       no_plan_conditioning=False, recurrence_variety=False,
                       variety_rng=None, facing_continuity=False, beat_grid=None,
                       music=None, bar_bounds=None, beat_anchor=0.0,
                       beat_anchor_per_limb=False,
                       music_anchor=0.0, music_anchor_lag=0.20,
                       rhythm_weight=0.0,
                       music_anchor_shuffle=0,
                       lower_body_delay=0.0, seam_lead=0.0, continue_phrase=0, motif_at_phrase=False,
                       continue_no_replay=False, continue_phrase_novelty=0.0, phrase_rhythm_keep=0.0,
                       tempo_keep=0.0, phrase_chain=False, step_lock_keep=0.0, energy_follow=None):
    """Build a fail-closed draft for one inference query.

    ``no_plan_conditioning`` is the control the two-stage design has never been
    measured against: an all-zero draft and mask, so the completion model sees
    music alone and the plan reaches the motion through nothing.  Every
    comparison so far has been between two *ways* of planning; none has been
    against not planning.  It is diagnostic-only -- the artifact it produces is
    marked ``headline_eligible: false`` -- because a number produced without the
    vocabulary is not a number about this method.

    An input without an explicit retrieval group cannot safely prove that a
    training prototype is external.  Preserve the public library API for
    callers that intentionally do not ask for exclusion, but inference itself
    uses a zero-mask, unconditioned draft in that case.

    ``unsourced_retrieval`` is the caller's declaration that the query has no
    source recording *by construction*, so there is nothing to exclude.  That is
    the AIST M6 case and it is not a corner: the query there is named for a
    **song** (``mBR2``), a held-out song under a song-disjoint split, so no
    training window carries it and the exclusion would be vacuous.  Without this
    the fail-closed branch fires on every AIST query, and since the completion
    model takes ``(music, draft, mask)`` and never a label, the entire retrieval
    stage -- the paper's M5 -- silently contributes nothing.  Measured on the
    published artifacts: 33 of 40 samples behind the FID_k 17.43 headline had a
    zero draft, and the other 7 had plans that were 100% transition.

    It is a declaration and not an inference because only the caller knows which
    kind of name it passed.  It rides into the artifact so a reader can see which
    policy produced the motion.
    """
    if no_plan_conditioning:
        return (
            torch.zeros(
                len(labels), feature_dim, dtype=torch.float32, device=labels.device
            ),
            torch.zeros(len(labels), 1, dtype=torch.float32, device=labels.device),
        )
    if query_retrieval_group_id is None and unsourced_retrieval:
        return library.build_draft(
            labels, feature_dim, allow_missing=True,
            root_continuity=root_continuity, gap_fill=gap_fill,
            seam_blend=seam_blend,
            seam_window=seam_window,
            seam_stagger=seam_stagger,
            seam_transition=seam_transition,
            seam_aware_retrieval=seam_aware_retrieval,
        root_velocity_blend=root_velocity_blend,
        root_seam_smooth=root_seam_smooth,
        facing_anchor=facing_anchor,
            target_period=target_period,
            recurrence_variety=recurrence_variety,
            variety_rng=variety_rng,
            facing_continuity=facing_continuity,
            beat_grid=beat_grid,
            music=music,
            bar_bounds=bar_bounds,
            beat_anchor=beat_anchor,
            beat_anchor_per_limb=beat_anchor_per_limb,
            music_anchor=music_anchor,
            music_anchor_lag=music_anchor_lag,
            rhythm_weight=rhythm_weight,
            music_anchor_shuffle=music_anchor_shuffle,
            lower_body_delay=lower_body_delay,
            seam_lead=seam_lead,
            continue_phrase=continue_phrase,
            motif_at_phrase=motif_at_phrase,
            continue_no_replay=continue_no_replay,
            continue_phrase_novelty=continue_phrase_novelty,
            phrase_rhythm_keep=phrase_rhythm_keep,
            tempo_keep=tempo_keep,
            phrase_chain=phrase_chain,
            step_lock_keep=step_lock_keep,
            energy_follow=energy_follow,
        )
    if query_retrieval_group_id is None:
        return (
            torch.zeros(
                len(labels), feature_dim, dtype=torch.float32, device=labels.device
            ),
            torch.zeros(len(labels), 1, dtype=torch.float32, device=labels.device),
        )
    return library.build_draft(
        labels,
        feature_dim,
        exclude_retrieval_group_ids=(query_retrieval_group_id,),
        allow_missing=True,
        root_continuity=root_continuity,
        gap_fill=gap_fill,
        seam_blend=seam_blend,
        seam_window=seam_window,
        seam_stagger=seam_stagger,
        seam_transition=seam_transition,
        seam_aware_retrieval=seam_aware_retrieval,
        root_velocity_blend=root_velocity_blend,
        root_seam_smooth=root_seam_smooth,
        facing_anchor=facing_anchor,
        target_period=target_period,
        recurrence_variety=recurrence_variety,
        variety_rng=variety_rng,
        facing_continuity=facing_continuity,
        beat_grid=beat_grid,
        music=music,
        bar_bounds=bar_bounds,
        beat_anchor=beat_anchor,
        beat_anchor_per_limb=beat_anchor_per_limb,
        music_anchor=music_anchor,
        music_anchor_lag=music_anchor_lag,
        rhythm_weight=rhythm_weight,
        music_anchor_shuffle=music_anchor_shuffle,
        lower_body_delay=lower_body_delay,
        seam_lead=seam_lead,
        continue_phrase=continue_phrase,
        motif_at_phrase=motif_at_phrase,
        continue_no_replay=continue_no_replay,
        continue_phrase_novelty=continue_phrase_novelty,
        phrase_rhythm_keep=phrase_rhythm_keep,
        tempo_keep=tempo_keep,
        phrase_chain=phrase_chain,
        step_lock_keep=step_lock_keep,
        energy_follow=energy_follow,
    )


def _check_unfillable_slots(name, retrieval_stretch, allow_unfillable_slots):
    """Refuse a clip whose plan asked for more frames than its class contains.

    FAIL CLOSED.  A warning here would be the failure mode
    ``check_disk_headroom``'s docstring is about: a gate that cannot stop
    anything reads like "checked" while the defect ships.  The pooled stretch
    median is 1.000 while this is happening -- measured 2026-09-05, 8 of 18 eval
    clips tripped it and one played 10.4 of its 14.3 s at 0.48x speed -- so
    nothing downstream would have noticed.

    THIS IS A FUNCTION, not two copies, because two copies is exactly how the
    hole below happened: the gate and the per-clip log reset lived only on the
    single-clip branch, so with ``--inference-batch-size`` above 1 -- its
    DEFAULT is 4, and ``tools/run_m6_wild.sh`` does not set it -- no clip was
    ever checked on the path that actually ships, while every measurement of the
    defect was taken at batch size 1.
    """
    if not retrieval_stretch or not retrieval_stretch["units_over_library_ceiling"]:
        return
    worst = retrieval_stretch["worst"]
    message = (
        "{}: {} retrieval unit(s) ask for more frames than their "
        "class contains, so they are played in slow motion. Worst: "
        "label {} wants {} frames and the longest of its {} "
        "candidates is {} -> stretch {:.3f}, i.e. {:.2f}x speed.".format(
            name, retrieval_stretch["units_over_library_ceiling"],
            worst["label"], worst["slot_frames"], worst["pool"],
            worst["pool_max_frames"], worst["stretch"],
            1.0 / worst["stretch"]))
    if not allow_unfillable_slots:
        raise SystemExit(
            "error: " + message + "\n"
            "  A label RUN is not a retrieval unit: the T vocabulary is cut on "
            "a 4-beat grid, so nothing in the corpus is longer than 2.97 s and "
            "any longer slot is by construction several bars.\n"
            "  Fix: pass --draft-bar-prototypes (with --plan-bar-grid or "
            "--plan-bar-tokens) so each BAR gets its own prototype instead of "
            "one prototype being stretched across a merged label run. Measured "
            "on the 18 sourced eval clips that takes 9 unfillable units to 0 "
            "and the worst playback from 0.48x to 0.99x.\n"
            "  To reproduce a pre-2026-09-05 artifact byte for byte instead, "
            "pass --allow-unfillable-slots.")
    print("warning: " + message, file=sys.stderr)


def _safe_draft_condition_fraction(labels, mask, index_filler=False):
    """What share of the plan's atomic frames got a source-safe prototype.

    ``None`` -- not 1.0 -- when the plan names no atomic frame at all.  Zero of
    zero conditioned is not "everything was conditioned", and reporting 1.0
    there gives this check its best possible reading in exactly the case it
    exists to catch.  Measured cost of the old return value: the AIST arm at
    ``runs/m6_songsplit1286_retrieval`` has 21 of 40 clips whose plan is 100%
    transition, and all 40 recorded ``safe_draft_condition_fraction: 1.0``
    beside a ``query_retrieval_group_id`` that was also present -- both of
    2026-08-16's checks passed on a plan that was empty, and the fid_k those
    clips produced (23.473) was read as reproducing the paper's 24.02.

    ``plan_atomic_segments`` rides beside it in the manifest for the same
    reason: a fraction cannot distinguish "no segments" from "every segment
    conditioned", and only one of those is a generation the vocabulary took
    part in.

    ``index_filler`` widens the DENOMINATOR, and must.  With ``--index-filler``
    the transition class is retrieved like any other, so the mask covers filler
    frames too -- while this denominator counted only ``labels != 0``.  Measured
    2026-09-06 on the 20 eval clips, the "fraction" then read up to **7.792**,
    and a fraction above one is not a reading, it is a broken instrument.  The
    denominator is the set of frames a prototype was ALLOWED to cover under the
    configuration that produced the artifact.
    """
    retrievable = int(labels.numel() if index_filler else (labels != 0).sum().item())
    if not retrievable:
        return None
    return float(mask.sum().item() / retrievable)


def _plan_atomic_segments(labels):
    """How many non-transition segments the plan actually named."""
    from dataset.atomic import labels_to_segments

    return sum(1 for segment in labels_to_segments(labels) if segment.label)


def _plan_postprocess_totals(reports):
    """Pool the per-clip plan records into one line a reader can act on.

    ``None`` entries are ORACLE plans, which are used verbatim and so have no
    post-process to report; if every entry is ``None`` this returns ``None``
    rather than a row of zeros, because "nothing was rewritten" and "this arm
    does not rewrite" are different claims.
    """
    rows = [row for row in reports if row]
    if not rows:
        return None
    frames = sum(row["frames"] for row in rows)
    rewritten = sum(row["frames_rewritten_to_transition"] for row in rows)
    after_refine = sum(row["transition_frames_after_refine"] for row in rows)
    after_fusion = sum(row["transition_frames_after_fusion"] for row in rows)
    return {
        "clips": len(rows),
        "frames": frames,
        "transition_fraction_after_fusion": after_fusion / frames if frames else None,
        "transition_fraction_after_refine": after_refine / frames if frames else None,
        "frames_rewritten_to_transition": rewritten,
        "rewritten_fraction": rewritten / frames if frames else None,
        "atomic_segments_per_clip": (
            sum(row["atomic_segments_after_refine"] for row in rows) / len(rows)
        ),
        "clips_with_an_empty_plan": sum(
            1 for row in rows if not row["atomic_segments_after_refine"]),
    }


def _replay_plan(plan_dir, name, labels):
    """``--plan-from``: the plan an earlier run of ours drew for ``name``.

    Returns (labels, plan_report).  The report is replayed with the labels
    because ``bar_grid_phase`` in it places the bar grid the draft is built
    on; replaying the labels alone would put them on a different grid.
    Refuses a length mismatch rather than trimming: the plan is per frame of
    this clip's music, so a different length means a different clip or cut.
    """
    import pickle
    path = Path(plan_dir) / (name + ".pkl")
    with open(path, "rb") as handle:
        blob = pickle.load(handle)
    replay = torch.as_tensor(np.asarray(blob["atomic_labels"]), dtype=labels.dtype)
    if tuple(replay.shape) != tuple(labels.shape):
        raise ValueError("--plan-from {}: {} frames, this run planned {}".format(
            path, tuple(replay.shape), tuple(labels.shape)))
    report = dict(blob["prototype_retrieval"]["plan_postprocess"])
    return replay, report


def _generation_protocol(ground_truth_labels):
    if ground_truth_labels:
        return ORACLE_GROUND_TRUTH_PLAN, False
    return SELF_DRIVEN_PLANNER, True


def _generated_is_current(
    path,
    planner_checkpoint,
    completion_checkpoint,
    expected_frames,
    generation_protocol,
    headline_eligible,
    target_motion_selection,
    deterministic_planner,
    planner_temperature,
    query_retrieval_group_id,
    dataset_provenance,
):
    try:
        with open(str(path), "rb") as handle:
            data = pickle.load(handle)
    except (OSError, EOFError, pickle.UnpicklingError):
        return False
    return (
        data.get("planner_checkpoint") == str(planner_checkpoint)
        and data.get("completion_checkpoint") == str(completion_checkpoint)
        and data.get("plan_source") == generation_protocol
        and data.get("generation_protocol") == generation_protocol
        and data.get("headline_eligible") is headline_eligible
        and data.get("target_motion_selection") == target_motion_selection
        and data.get("deterministic_planner") == deterministic_planner
        and data.get("planner_temperature") == planner_temperature
        and data.get("prototype_retrieval", {}).get("policy")
        == "EXCLUDE_QUERY_RETRIEVAL_GROUP_FAIL_CLOSED"
        and data.get("prototype_retrieval", {}).get("query_retrieval_group_id") == query_retrieval_group_id
        and data.get("dataset_provenance") == dataset_provenance
        and "full_pose" in data
        and (expected_frames is None or len(data["full_pose"]) == expected_frames)
    )


def _filter_and_refine_labels(
    labels, available_labels, vote_window=5, min_segment_length=6,
    transition_policy="protect", merge_order="shortest", stats=None,
):
    """The short-clip counterpart of the tail of ``infer_plan``.

    ``generate`` takes a batched path when every pending clip is no longer than
    the planner window, so there is no windowing and therefore no fusion -- but
    the same availability rewrite and the same refine still run, and until
    2026-08-23 neither was counted.  ORACLE plans do not come through here: they
    are read from ``GroundTruthPlanStore`` and used verbatim.
    """
    labels = labels.clone()
    unavailable = torch.ones_like(labels, dtype=torch.bool)
    for label in available_labels:
        unavailable &= labels != label
    unavailable &= labels != 0
    rewritten = int(unavailable.sum())
    labels[unavailable] = 0
    refined = refine_plan(labels, vote_window, min_segment_length,
                          transition_policy=transition_policy,
                          merge_order=merge_order)
    if stats is not None:
        stats.update({
            "frames": int(len(labels)),
            "windows": None,
            "transition_frames_after_fusion": int((labels == 0).sum()) - rewritten,
            "frames_rewritten_to_transition": rewritten,
            "transition_frames_after_refine": int((refined == 0).sum()),
            "atomic_segments_after_refine": _plan_atomic_segments(refined),
            "plan_vote_tie_break": None,
            "plan_transition_policy": transition_policy,
            "plan_merge_order": merge_order,
            "plan_vote_window": vote_window,
            "plan_min_segment_length": min_segment_length,
        })
    return refined


def _write_generated_result(
    output_path,
    normalized,
    labels,
    audio_path,
    normalizer_path,
    planner_checkpoint,
    completion_checkpoint,
    plan_source=SELF_DRIVEN_PLANNER,
    deterministic_planner=False,
    planner_temperature=1.0,
    generation_protocol=None,
    headline_eligible=True,
    query_retrieval_group_id=None,
    safe_draft_condition_fraction=None,
    plan_atomic_segments=None,
    plan_postprocess=None,
    no_plan_conditioning=False,
    unsourced_retrieval=False,
    target_motion_selection="NONE",
    dataset_provenance=None,
    retrieval_stretch=None,
    floor_anchor=None,
    fix_skate=0.0,
    fix_skate_seam_blend=0,
    face_camera_strength=None,
    face_camera_window=FACING_SMOOTH_SECONDS,
):
    if generation_protocol is None:
        generation_protocol = (
            ORACLE_GROUND_TRUTH_PLAN
            if plan_source in {"ground_truth", "ground-truth", ORACLE_GROUND_TRUTH_PLAN}
            else SELF_DRIVEN_PLANNER
        )
    if generation_protocol == ORACLE_GROUND_TRUTH_PLAN:
        headline_eligible = False
    if no_plan_conditioning:
        headline_eligible = False
    result = decode_motion(normalized, normalizer_path)
    if fix_skate:
        # BEFORE the floor anchor and the camera rotation: this one moves the
        # body in the ground plane, those two are a z shift and a z rotation,
        # and putting the translation first means the rotation carries it like
        # any other part of the clip instead of being computed against a path
        # that is about to change.
        # The seam window comes from the plan's own bar grid, so a clip with no
        # bar grid (an ORACLE plan) simply gets the old behaviour rather than a
        # guess.  Half-width = the blend's own, because that is exactly the span
        # _blend_draft_seams ramps over.
        seam_frames = None
        half = int(fix_skate_seam_blend or 0)
        bounds = (plan_postprocess or {}).get("bar_bounds") or []
        laid = (plan_postprocess or {}).get("draft_seam_frames")
        if half > 0 and laid:
            # --draft-seam-lead: the seams as laid down, off the bar lines
            seam_frames = np.concatenate([np.arange(max(0, int(b) - half), int(b) + half + 1) for b in laid])
        elif half > 0 and len(bounds) > 2:
            seam_frames = np.concatenate([
                np.arange(max(0, int(b) - half), int(b) + half + 1)
                for b in bounds[1:-1]]) if len(bounds) > 2 else None
        result = fix_foot_skate(result, fix_skate, seam_frames=seam_frames)
    if floor_anchor is not None:
        # Applied HERE, in the one writer every path funnels through, rather
        # than at each of the three call sites -- the pattern that produced
        # eight "recorded but not wired" bugs on 2026-09-05/06 was a switch
        # applied at some call sites and not others.
        result = anchor_floor(result, floor_anchor)
    if face_camera_strength is not None:
        # Same reason as the floor anchor above: applied in the one writer every
        # path funnels through, not at the three call sites.  A z shift and a z
        # rotation commute, so the order here only fixes which report reads
        # first.
        result = face_camera(result, face_camera_strength, face_camera_window)
    result["atomic_labels"] = labels.numpy()
    result["audio_path"] = str(audio_path)
    result["planner_checkpoint"] = str(planner_checkpoint)
    result["completion_checkpoint"] = str(completion_checkpoint)
    # ``plan_source`` remains for existing readers, but is intentionally the
    # explicit protocol name rather than an ambiguous "ground_truth" token.
    result["plan_source"] = generation_protocol
    result["generation_protocol"] = generation_protocol
    result["headline_eligible"] = bool(headline_eligible)
    result["deterministic_planner"] = deterministic_planner
    result["planner_temperature"] = planner_temperature
    result["prototype_retrieval"] = {
        "policy": ("RETRIEVE_WITHOUT_EXCLUSION_QUERY_DECLARED_UNSOURCED"
                   if unsourced_retrieval and query_retrieval_group_id is None
                   else "EXCLUDE_QUERY_RETRIEVAL_GROUP_FAIL_CLOSED"),
        "query_retrieval_group_id": query_retrieval_group_id,
        "retrieval_group_exclusion_requested": query_retrieval_group_id is not None,
        "safe_draft_condition_fraction": safe_draft_condition_fraction,
        # A fraction cannot tell "no segments" from "every segment conditioned";
        # this can, and it is what says whether the vocabulary took part at all.
        "plan_atomic_segments": plan_atomic_segments,
        # What the plan post-process did to *this* clip: how much transition the
        # fusion produced, how much of it is the availability rewrite rather
        # than the planner, and which rules ran.  ``None`` for an ORACLE plan,
        # which is used verbatim.
        "plan_postprocess": plan_postprocess,
        # The control: the draft was zeroed on purpose, so this artifact says
        # what the completion stage does on music alone.
        "no_plan_conditioning": bool(no_plan_conditioning),
        # WHICH prototype filled each slot and how far it was stretched.  Absent
        # before 2026-09-05, which is why a clip playing its last 10.4 s at
        # 0.48x speed could not be diagnosed from its own artifact.  Read
        # ``units_over_library_ceiling`` first: it counts slots longer than
        # anything their class contains, and it is the only field here that a
        # pooled median does not hide.
        "retrieval_stretch": retrieval_stretch,
    }
    result["target_motion_selection"] = target_motion_selection
    result["dataset_provenance"] = dataset_provenance
    with open(str(output_path), "wb") as handle:
        pickle.dump(result, handle)


def infer_directory(
    audio_dir,
    output_dir,
    planner_checkpoint=DEFAULT_PLANNER_CHECKPOINT,
    completion_checkpoint=DEFAULT_COMPLETION_CHECKPOINT,
    data_root="data/atomic_aistpp",
    target_motion_dir=None,
    device="auto",
    seed=42,
    max_samples=None,
    max_frames=None,
    overwrite=False,
    deterministic_planner=False,
    temperature=1.0,
    completion_stride=75,
    completion_blend_width=None,
    completion_draft_guidance_weight=None,
    retrieval_energy_floor=None,
    draft_dump_dir=None,
    plan_stride=None,
    plan_fusion="centre",
    plan_vote_tie_break="centre",
    plan_transition_policy="protect",
    plan_merge_order="shortest",
    planner_guidance_weight=1.0,
    planner_transition_logit_bias=0.0,
    plan_bar_grid=False,
    plan_music_repeat=0.0,
    plan_bar_beats=4,
    plan_bar_tokens=False,
    plan_vote_window=5,
    plan_min_segment_length=6,
    unsourced_retrieval=False,
    guidance_weight=None,
    draft_noise_ratio=None,
    inference_batch_size=4,
    sequence_names=None,
    ground_truth_labels=False,
    plan_from=None,
    draft_root_continuity="off",
    draft_seam_blend=0,
    draft_seam_stagger=False,
    draft_seam_window="triangle",
    seam_transition="centred",
    draft_seam_aware_retrieval=False,
    draft_beat_anchor=0.0,
    draft_beat_anchor_per_limb=False,
    draft_music_anchor=0.0,
    draft_music_anchor_lag=0.20,
    draft_music_anchor_shuffle=0,
    draft_lower_body_delay=0.0,
    draft_seam_lead=0.0,
    draft_continue_phrase=0,
    draft_motif_at_phrase=False,
    draft_continue_no_replay=False,
    draft_continue_phrase_novelty=0.0,
    draft_phrase_rhythm_keep=0.0,
    draft_tempo_keep=0.0,
    draft_phrase_chain=False,
    draft_step_lock_keep=0.0,
    draft_energy_follow=None,
    completion_beat_keep=0.0,
    completion_beat_stride=1,
    completion_beat_keep_holds=False,
    completion_beat_free_max=1.0,
    completion_beat_hold_frac=0.5,
    draft_join_top_k=None,
    draft_join_pose_weight=1.0,
    draft_whole_units=False,
    draft_bar_units=None,
    draft_continue_source=None,
    draft_continue_lookahead=False,
    draft_continue_any_label=False,
    draft_continue_max_run=0,
    draft_prefer_full=0.0,
    draft_prefer_full_mode="frames",
    draft_rhythm_scorer=None,
    draft_rhythm_keep=0.0,
    draft_rhythm_shift=0,
    draft_continue_stop=False,
    draft_continue_rhythm_min=0.0,
    draft_continue_settle=0.0,
    draft_motif_return=0.0,
    draft_motif_lags="4,2",
    draft_motif_max=2,
    draft_motif_pick="first",
    draft_continuity_scorer=None,
    draft_continuity_keep=0.0,
    draft_continuity_weight=0.0,
    draft_floor_normalize=None,
    draft_phase_weight=0.0,
    draft_rhythm_weight=0.0,
    draft_feet_lead=False,
    draft_beat_span=0.0,
    draft_hold_by_music=False,
    draft_beat_fit=False,
    draft_join_lever_weights=False,
    draft_feet_beat_lead=False,
    draft_quiet_cut=0,
    draft_root_velocity_blend=0,
    draft_root_seam_smooth=0,
    draft_facing_anchor=0.0,
    floor_anchor=None,
    fix_skate=0.0,
    fix_skate_seam_blend=0,
    face_camera_strength=None,
    face_camera_window=FACING_SMOOTH_SECONDS,
    index_filler=False,
    derive_retrieval_group=False,
    draft_bar_prototypes=False,
    allow_unfillable_slots=False,
    completion_start_step=None,
    completion_reproject_every=None,
    completion_inpaint_seam_width=None,
    completion_keep_root=False,
    completion_sample_steps=None,
    draft_recurrence_variety=False,
    draft_facing_continuity=False,
    draft_gap_fill="zero",
    no_plan_conditioning=False,
    retrieval_rule="duration",
    draft_only=False,
    retrieval_selector=None,
    retrieval_tie_break="index",
    retrieval_selector_top_k=8,
    draft_selector_join_band=0.0,
    retrieval_max_yaw_step=None,
    retrieval_yaw_steps=None,
    retrieval_max_speed_spike=None,
    retrieval_speed_spikes=None,
    draft_join_frame="raw",
    draft_join_height_weight=0.0,
    retrieval_selector_input_units="inference",
    draft_hop_guard=False,
    draft_music_energy=False,
    retrieval_vertical_flags=None,
    draft_unit_floor=None,
    retrieval_selector_temperature=1.0,
    music_span_check="refuse",
    ingest_root=DEFAULT_INGEST_ROOT,
    allow_cross_release_checkpoints=False,
):
    seed_everything(seed)
    if inference_batch_size < 1:
        raise ValueError("inference batch size must be positive")
    levelled, reference = release_floor_state(data_root)
    if levelled and draft_floor_normalize:
        raise SystemExit(
            "error: {} is already levelled at the source (build.json says "
            "floor_levelled, reference {:.4f} m), so --draft-floor-normalize "
            "would subtract a floor that is already zero. Drop the flag."
            .format(data_root, float(reference or 0.0)))
    if levelled and floor_anchor is not None and reference is not None \
            and abs(float(floor_anchor) - float(reference)) > 0.05:
        raise SystemExit(
            "error: --floor-anchor {:.4f} disagrees with the levelled release's "
            "own reference {:.4f} m by more than 5 cm, which would undo the "
            "levelling one clip at a time. Use the reference, or drop the flag."
            .format(float(floor_anchor), float(reference)))
    if (draft_floor_normalize or draft_unit_floor) and "z" in str(draft_root_continuity):
        raise SystemExit(
            "error: --draft-floor-normalize / --draft-unit-floor put every "
            "prototype on a shared floor, and --draft-root-continuity {} then "
            "re-aligns the ROOT at each seam, which is the very thing that lifts "
            "the feet. Use --draft-root-continuity xy (or off) with it."
            .format(draft_root_continuity))
    if draft_unit_floor:
        # The local floors are metres of a SPECIFIC decode of a specific
        # release.  Window names are identical across the levelled and
        # unlevelled T releases, so the name check in the library cannot tell
        # them apart; the census sidecar can.
        provenance = _unit_floor_provenance(draft_unit_floor)
        if provenance is None:
            raise SystemExit(
                "error: --draft-unit-floor {} has no provenance sidecar ({}); "
                "rebuild it with tools/census_release_vertical.py --floor-out"
                .format(draft_unit_floor, Path(str(draft_unit_floor)).with_suffix(".json")))
        if bool(provenance.get("floor_levelled")) != bool(levelled):
            raise SystemExit(
                "error: --draft-unit-floor {} was built from a release with "
                "floor_levelled={}, but {} has floor_levelled={}; the floor would "
                "be taken off twice or not at all"
                .format(draft_unit_floor, provenance.get("floor_levelled"),
                        data_root, levelled))
    if draft_join_frame == "placed" and not (
            draft_facing_continuity and draft_root_continuity == "xy"
            and not draft_root_velocity_blend
            and not draft_beat_anchor and not draft_music_anchor):
        # _placed_head models exactly this placement: turn to the tail's
        # heading, move xy onto it, leave z.  Any other build_draft setting
        # would score a head the draft never writes (review 2026-09-17: up to
        # 1.59 m and 1.9 rot6d off with continuity off; a phantom 8.7 cm height
        # step, x30, under xyz).
        raise SystemExit(
            "error: --draft-join-frame placed models facing continuity plus "
            "--draft-root-continuity xy with no velocity blend and no beat or "
            "music anchor; this run sets facing_continuity={}, root_continuity={}, "
            "root_velocity_blend={}, beat_anchor={}, music_anchor={}"
            .format(draft_facing_continuity, draft_root_continuity,
                    draft_root_velocity_blend, draft_beat_anchor, draft_music_anchor))
    if draft_rhythm_weight and not draft_seam_aware_retrieval:
        raise SystemExit(
            "error: --draft-rhythm-weight adds a term to the seam-aware "
            "ranking, so it needs --draft-seam-aware-retrieval. Without it the "
            "ranking branch is never taken and the flag would be recorded but "
            "inert.")
    if draft_phase_weight and not draft_seam_aware_retrieval:
        raise SystemExit(
            "error: --draft-phase-weight adds a term to the seam-aware "
            "ranking, so it needs --draft-seam-aware-retrieval. Without it the "
            "ranking branch is never taken and the flag would be recorded but "
            "inert.")
    if draft_music_anchor and draft_beat_anchor:
        raise SystemExit(
            "error: --draft-music-anchor and --draft-beat-anchor both warp the "
            "prototype's settle points and would fight over the same frames. "
            "--draft-beat-anchor targets the beat grid and made every reading "
            "worse; --draft-music-anchor targets the song's onset peaks.")
    if draft_music_anchor_shuffle and not draft_music_anchor:
        raise SystemExit(
            "error: --draft-music-anchor-shuffle is the control for "
            "--draft-music-anchor and does nothing without it; it would be "
            "recorded in the manifest and inert.")
    if draft_beat_anchor_per_limb and not draft_beat_anchor:
        # Refused, not ignored: with the anchor off the per-limb branch is never
        # reached, so accepting the pair would write
        # ``draft_beat_anchor_per_limb: true`` into a manifest describing a run
        # in which no limb was ever warped.
        raise SystemExit(
            "error: --draft-beat-anchor-per-limb chooses HOW the beat anchor "
            "warps, so it needs --draft-beat-anchor > 0. With the anchor off "
            "the flag would be recorded but inert.")
    if seam_transition not in ("centred", "after"):
        raise SystemExit("error: --seam-transition is 'centred' or 'after', got %r"
                         % (seam_transition,))
    if draft_seam_lead and seam_transition == "after":
        raise SystemExit("error: --draft-seam-lead moves the cut BEFORE the bar line and --seam-transition after "
                         "makes it after; pick one")
    if seam_transition == "after" and draft_seam_stagger:
        # Same rule as the check below: "after" does no limb cross-fade, so a
        # stagger of that cross-fade would be recorded and never happen.
        raise SystemExit(
            "error: --draft-seam-stagger offsets the limb cross-fade, and "
            "--seam-transition after does not cross-fade the limbs; drop one.")
    if draft_seam_stagger and not draft_seam_blend:
        # Refused, not ignored.  ``_blend_draft_seams`` returns before it can
        # stagger anything when the half-width is 0, so accepting the pair would
        # write ``draft_seam_stagger: true`` into a manifest describing a run in
        # which the stagger never happened -- the "flag recorded but not wired"
        # defect this repository has already paid for twice.
        raise SystemExit(
            "error: --draft-seam-stagger offsets each limb's seam CROSS-FADE, so "
            "it needs --draft-seam-blend > 0. With no cross-fade there is nothing "
            "to stagger and the flag would be recorded but inert.")
    if plan_bar_tokens:
        # Checked here, before a 228 MB checkpoint is read, and by the same
        # function ``infer_plan`` enforces it with, so the CLI and the API
        # cannot disagree about what bar mode accepts.
        problem = _bar_token_option_error(plan_stride, plan_fusion,
                                          plan_vote_window, plan_min_segment_length)
        if problem:
            raise SystemExit("error: " + problem)
    # The same verified-release gate used by training protects the indexed
    # prototype library here.  A legacy root remains readable, but is marked
    # unverified and resolves unknown query provenance to a zero draft.
    dataset_provenance = validate_training_data_root(data_root)
    device = resolve_device(device)
    completion, completion_args = _load_checkpoint(
        completion_checkpoint, "completion", device
    )
    release_binding = {}
    _check_release_binding(release_binding, "completion", completion_args, data_root,
                           "release", allow_cross_release_checkpoints)
    bar_pooling = None
    if ground_truth_labels:
        planner = None
        planner_args = completion_args
        ground_truth_plans = GroundTruthPlanStore(data_root)
        if plan_bar_tokens:
            raise SystemExit(
                "error: --plan-bar-tokens is a way of running the PLANNER, and "
                "--ground-truth-labels / --plan-source ground-truth replaces the "
                "planner with the release's own per-frame label track. Drop one.")
    else:
        planner, planner_args = _load_checkpoint(planner_checkpoint, "planner", device)
        _check_release_binding(release_binding, "planner", planner_args, data_root,
                               "bar_release" if plan_bar_tokens else "release",
                               allow_cross_release_checkpoints)
        ground_truth_plans = None
        # The token-resolution gate runs before the two cross-stage checks
        # because in bar mode neither of them means what it says: the planner's
        # ``music_dim`` is the POOLED width (checked against the release width
        # inside the gate) and its ``seq_len`` counts BARS, not frames, so
        # comparing it to the completion's 150-frame window would refuse every
        # correctly built bar planner.
        bar_pooling = check_planner_bar_tokens(
            planner, planner_args, completion_args.music_dim, plan_bar_tokens,
            plan_bar_beats)
        if not plan_bar_tokens:
            if planner_args.music_dim != completion_args.music_dim:
                raise ValueError("planner and completion music dimensions differ")
            if planner_args.seq_len != completion_args.seq_len and not draft_only:
                # WHY draft_only IS THE ONLY EXEMPTION.  The two windows are
                # independent everywhere else in this function: ``infer_plan``
                # is handed ``planner_args.seq_len`` and ``infer_completion``
                # ``completion_args.seq_len``.  The equality is required
                # because the completion consumes the plan the planner wrote,
                # and a plan produced in 300-frame windows against a completion
                # trained on 150 has never been shown to be safe.  Under
                # ``--draft-only`` the completion is never called at all (the
                # retrieval draft is written as the output), so the checkpoint
                # is loaded for its ``motion_dim`` and normalizer and nothing
                # else -- which is what makes a context-length dose-response
                # measurable without first training a matching completion.
                raise ValueError(
                    "planner and completion window lengths differ "
                    "({} vs {}); only --draft-only may run them mismatched, "
                    "because there the completion never sees the plan"
                    .format(planner_args.seq_len, completion_args.seq_len))
    # What ``_load_music`` must find in the query's .npy: the RELEASE width.  In
    # bar mode the planner's own width is the pooled one and would refuse the
    # 35-D array that is about to be pooled into it.
    query_music_dim = (completion_args.music_dim if plan_bar_tokens
                       else planner_args.music_dim)
    # Whether the plan gets its own channel is a property of the checkpoint,
    # not a flag: a model built with the embedding refuses to run without labels
    # and one built without it refuses to be given them, so reading it off the
    # loaded module is the only way the two can never disagree.
    completion_conditions_on_labels = getattr(
        getattr(completion, "model", None), "label_embedding", None
    ) is not None
    selector = None
    if retrieval_selector:
        from model.retrieval_selector import load_selector

        selector = load_selector(retrieval_selector)
    library = IndexedAtomicMotionLibrary(
        data_root, retrieval_rule=retrieval_rule,
        energy_floor_quantile=retrieval_energy_floor,
        selector=selector, selector_top_k=retrieval_selector_top_k,
        selector_join_band=draft_selector_join_band,
        max_yaw_step=retrieval_max_yaw_step,
        yaw_steps_path=retrieval_yaw_steps,
        max_speed_spike=retrieval_max_speed_spike,
        speed_spikes_path=retrieval_speed_spikes,
        join_frame=draft_join_frame,
        join_height_weight=draft_join_height_weight,
        selector_input_units=retrieval_selector_input_units,
        hop_guard=draft_hop_guard,
        music_energy=draft_music_energy,
        vertical_flags_path=retrieval_vertical_flags,
        unit_floor_path=draft_unit_floor,
        selector_temperature=retrieval_selector_temperature,
        tie_break=retrieval_tie_break, index_filler=index_filler,
        join_top_k=draft_join_top_k,
        join_pose_weight=draft_join_pose_weight,
        whole_units=draft_whole_units,
        bar_units_path=draft_bar_units,
        continue_source_path=draft_continue_source,
        continue_lookahead=draft_continue_lookahead,
        continue_any_label=draft_continue_any_label,
        continue_max_run=draft_continue_max_run,
        prefer_full=draft_prefer_full,
        prefer_full_mode=draft_prefer_full_mode,
        rhythm_scorer=draft_rhythm_scorer,
        rhythm_keep=draft_rhythm_keep,
        rhythm_shift=draft_rhythm_shift,
        continue_stop=draft_continue_stop,
        continue_rhythm_min=draft_continue_rhythm_min,
        continue_settle=draft_continue_settle,
        motif_return=draft_motif_return,
        motif_lags=draft_motif_lags,
        motif_max=draft_motif_max,
        motif_pick=draft_motif_pick,
        continuity_scorer=draft_continuity_scorer,
        continuity_keep=draft_continuity_keep,
        continuity_weight=draft_continuity_weight,
        floor_normalize=draft_floor_normalize,
        phase_weight=draft_phase_weight,
        rhythm_weight=draft_rhythm_weight,
        feet_lead=draft_feet_lead,
        beat_span_tolerance=draft_beat_span,
        hold_by_music=draft_hold_by_music,
        beat_fit=draft_beat_fit,
        join_lever_weights=draft_join_lever_weights,
        feet_beat_lead=draft_feet_beat_lead)

    # A generator PER CLIP, derived from the clip's name -- see ``variety_rng``
    # below, built inside each loop by ``_variety_rng``.
    #
    # WHAT THIS REPLACED, and why the old line read as safe.  It was
    # ``np.random.default_rng(seed)``: one generator for the whole run, and its
    # comment said "so a run is reproducible from its seed".  That is true and
    # it is not the property anyone needs.  Because the draws are consumed in
    # clip order, a clip's prototype picks depend on every clip processed before
    # it, so a run is reproducible from its seed AND ITS CLIP ORDER -- which the
    # comment did not say.  This is the same defect ``sample_seed`` was written
    # for on the noise side, surviving in the retrieval side because the fix was
    # applied where it was noticed rather than to the class of bug.
    #
    # MEASURED, 8 clips, everything else byte-identical (2026-08-31):
    #   same list reversed        -> the DRAFT differs on 6 of 8 clips, max
    #                                |delta| 2.19 m, and the output on 6 of 8.
    #   same list + 12 appended   -> 0 of 8 differ.
    # Order, not length; retrieval, not noise.  Two arms scored on differently
    # ORDERED clip lists were therefore comparing different dances.
    normalizer_path = Path(data_root) / "normalizer.pt"
    if not normalizer_path.is_file():
        raise FileNotFoundError("missing motion normalizer: {}".format(normalizer_path))

    audio = _audio_map(audio_dir)
    if target_motion_dir:
        target_map = {
            path.stem: path for path in sorted(Path(target_motion_dir).glob("*.pkl"))
        }
        if sequence_names is None:
            targets = list(target_map.values())
        else:
            missing = [name for name in sequence_names if name not in target_map]
            if missing:
                raise FileNotFoundError(
                    "sequence list motions are missing: {}".format(missing[:5])
                )
            targets = [target_map[name] for name in sequence_names]
        # A target-motion directory is often supplied by evaluation tooling to
        # select sequence names.  SELF_DRIVEN_PLANNER must not open or derive
        # timing from those target motions; audio determines its output length.
        target_motion_selection = (
            "ORACLE_TARGET_FRAME_LENGTH_READ"
            if ground_truth_labels
            else "NAME_SELECTION_ONLY_NO_CONTENT_READ"
        )
        items = [
            (
                path.stem,
                _match_audio(path.stem, audio),
                _oracle_target_frames(path) if ground_truth_labels else None,
            )
            for path in targets
        ]
    else:
        target_motion_selection = "NONE"
        names = sorted(audio) if sequence_names is None else sequence_names
        missing = [name for name in names if name not in audio]
        if missing:
            raise FileNotFoundError(
                "sequence list audio files are missing: {}".format(missing[:5])
            )
        items = [(name, audio[name], None) for name in names]
    if max_samples is not None:
        items = items[:max_samples]
    if ground_truth_plans is not None:
        original_count = len(items)
        items = [item for item in items if ground_truth_plans.has_sequence(item[0])]
        if len(items) != original_count:
            print(
                "Skipping {} sequences without ORACLE ground-truth atomic labels".format(
                    original_count - len(items)
                )
            )
    if not items:
        raise FileNotFoundError("no inference inputs found")

    # Before a single frame is generated: the generated length IS the music
    # length, so a query whose music comes from a different cut of its own
    # recording produces a full, self-consistent, wrongly-timed result that no
    # later check can distinguish from a right one.
    music_span_report = check_music_span(items, ingest_root, music_span_check)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if draft_dump_dir is not None:
        Path(draft_dump_dir).mkdir(parents=True, exist_ok=True)
    generated_paths = []
    plan_reports = []
    per_sample_seed = {}
    ratio = (
        completion_args.draft_noise_ratio
        if draft_noise_ratio is None
        else draft_noise_ratio
    )
    generation_protocol, headline_eligible = _generation_protocol(ground_truth_labels)
    if no_plan_conditioning:
        # The run-level flag as well as the per-clip one.  A manifest that says
        # "headline_eligible: true" over clips that each say false is the record
        # a reader checks first, and it would be the wrong one.
        headline_eligible = False
    pending = []
    for index, (name, audio_path, target_frames) in enumerate(items):
        output_path = output_dir / (name + ".pkl")
        generated_paths.append(output_path)
        frames = target_frames
        if max_frames is not None:
            frames = max_frames if frames is None else min(frames, max_frames)
        query_retrieval_group_id = _query_retrieval_group_id(
            library, name, derive_missing=derive_retrieval_group)
        if (
            output_path.is_file()
            and not overwrite
            and _generated_is_current(
                output_path,
                planner_checkpoint,
                completion_checkpoint,
                frames,
                generation_protocol,
                headline_eligible,
                target_motion_selection,
                deterministic_planner,
                temperature,
                query_retrieval_group_id,
                dataset_provenance,
            )
        ):
            print("Using generated motion: {}".format(output_path))
            continue
        pending.append(
            (
                index,
                name,
                audio_path,
                output_path,
                frames,
                query_retrieval_group_id,
            )
        )

    short_batch_mode = pending and not plan_bar_tokens and all(
        frames is not None and frames <= planner_args.seq_len
        for _, _, _, _, frames, _ in pending
    ) and planner_args.seq_len == completion_args.seq_len
    # The equality conjunct is a no-op for every run whose two windows match --
    # which is every run that existed before a planner could be trained on a
    # window longer than the completion's.  It is here because this batched
    # path pads the draft to ``completion_args.seq_len`` and always calls the
    # completion: a 300-frame planner window would silently truncate the plan
    # to 150 frames.  With the conjunct such a run takes the per-clip path
    # below, which honours the window lengths separately.
    # ``not plan_bar_tokens`` is explicit rather than incidental: in bar mode
    # ``planner_args.seq_len`` counts bars, so "frames <= seq_len" would compare
    # a frame count to a bar count and be false by accident rather than by rule.
    if short_batch_mode and plan_from is not None:
        # The batched path plans without infer_plan's per-clip loop, which is
        # the only place the replay below is applied.
        raise ValueError("--plan-from is only supported on the per-clip path "
                         "(bar tokens or clips longer than the planner window)")
    if short_batch_mode and plan_music_repeat:
        # This path plans without infer_plan, so it applies neither the bar
        # snap nor this pass.  Silently ignoring the flag is exactly the shape
        # of the batched-path bug that produced two byte-identical arms with
        # different names, so refuse and say which path would have run.
        raise ValueError(
            "--plan-music-repeat needs the per-clip plan path, but every clip "
            "fits one planner window so the batched path was selected.  Pass "
            "--inference-batch-size 1 is NOT enough -- the batched path is "
            "chosen by clip length, not batch size; use a planner window "
            "shorter than the clips, or drop the flag.")
    if short_batch_mode:
        batches = range(0, len(pending), inference_batch_size)
        for batch_start in tqdm(
            batches,
            total=(len(pending) + inference_batch_size - 1) // inference_batch_size,
            desc=(
                "ORACLE ground-truth-plan completion inference"
                if ground_truth_plans is not None
                else "Two-stage inference"
            ),
            unit="batch",
        ):
            batch = pending[batch_start : batch_start + inference_batch_size]
            music_rows = [_load_music(row[2], row[4], query_music_dim)
                          for row in batch]
            lengths = [len(music) for music in music_rows]
            if min(lengths) < 3:
                raise ValueError("audio is too short for inference")
            music_batch = torch.stack(
                [_pad_frames(music, planner_args.seq_len) for music in music_rows]
            ).to(device)
            padding_mask = torch.arange(planner_args.seq_len, device=device)[None] >= torch.tensor(
                lengths, device=device
            )[:, None]
            # One seed per *batch*.  Derived from the batch's member NAMES
            # rather than the first item's position, so re-ordering the work
            # no longer re-rolls it; but with batch_size > 1 a clip still
            # shares its draw with whoever it was batched beside, so two runs
            # over different clip LISTS are still not paired.  ``seed_pairing``
            # in the manifest says which of the two a run is, because the cost
            # of guessing has been measured: 14% of per-clip energy, larger
            # than most of the arm differences the 2026-08-30 waves were read
            # off.  Use ``--inference-batch-size 1`` for anything that will be
            # compared against another run.
            batch_seed = sample_seed(seed, "|".join(row[1] for row in batch))
            seed_everything(batch_seed)
            for row in batch:
                per_sample_seed[row[1]] = batch_seed
            if ground_truth_plans is not None:
                plans = [
                    ground_truth_plans.get(row[1], length)
                    for row, length in zip(batch, lengths)
                ]
                # Used verbatim: an ORACLE plan is the measurement, so neither
                # the availability rewrite nor the refine touches it.
                plan_stats = [None] * len(plans)
            else:
                extra = {}
                if planner_wants_global_music(planner):
                    # From music_rows, not music_batch: the batch is padded to
                    # seq_len and summarising the padding would hand short clips
                    # a different vector than training computed.
                    extra["global_music"] = torch.stack(
                        [track_summary(row) for row in music_rows]).to(device)
                sampled = planner.sample(
                    music_batch,
                    padding_mask=padding_mask,
                    temperature=temperature,
                    deterministic=deterministic_planner,
                    guidance_weight=planner_guidance_weight,
                    **extra,
                ).cpu()
                plan_stats = [{} for _ in lengths]
                plans = [
                    _filter_and_refine_labels(
                        labels[:length], library.labels_available,
                        plan_vote_window, plan_min_segment_length,
                        transition_policy=plan_transition_policy,
                        merge_order=plan_merge_order, stats=stats,
                    )
                    for (labels, length), stats in zip(zip(sampled, lengths), plan_stats)
                ]
            query_retrieval_group_ids = [row[5] for row in batch]
            # Each query's own beat period, so ``--retrieval-rule tempo`` can
            # prefer a prototype whose internal rhythm matches this track's.
            query_periods = [music_settle_period(row) for row in music_rows]
            # An explicit loop rather than a comprehension: the retrieval log has
            # to be cleared and read BETWEEN rows, or the gate would see the
            # batch's running total and blame the wrong clip.  Until 2026-09-05
            # this branch did neither -- see _check_unfillable_slots.
            conditions = []
            batch_stretches = []
            for labels, query_retrieval_group_id, query_period, row_name, row_music, row_stats \
                    in zip(plans, query_retrieval_group_ids, query_periods,
                           [row[1] for row in batch], music_rows, plan_stats):
                library.retrieval_log = []
                conditions.append(_source_safe_draft(
                    library,
                    labels,
                    completion_args.motion_dim,
                    query_retrieval_group_id,
                    unsourced_retrieval,
                    root_continuity=draft_root_continuity,
                    gap_fill=draft_gap_fill,
                    seam_blend=draft_seam_blend,
                    seam_window=draft_seam_window,
                    seam_stagger=draft_seam_stagger,
                    seam_transition=seam_transition,
                    seam_aware_retrieval=draft_seam_aware_retrieval,
                root_velocity_blend=draft_root_velocity_blend,
                root_seam_smooth=draft_root_seam_smooth,
                facing_anchor=draft_facing_anchor,
                    target_period=query_period,
                    no_plan_conditioning=no_plan_conditioning,
                    recurrence_variety=draft_recurrence_variety,
                    facing_continuity=draft_facing_continuity,
                    beat_grid=beat_grid_of(row_music),
                    music=row_music,
                    # THE BATCHED PATH MUST CUT ON BARS TOO.  It did not until
                    # 2026-09-05, and because --inference-batch-size defaults to
                    # 4 while run_m6_wild.sh does not set it, that made
                    # --draft-bar-prototypes INERT on the shipping path while
                    # every measurement of it was taken at batch size 1.  Each
                    # row carries its own music, so the grid is per row.
                    bar_bounds=(quiet_bar_bounds(
                                    bar_bounds_of(row_music, plan_bar_beats,
                                                  (row_stats or {}).get("bar_grid_phase")),
                                    row_music, draft_quiet_cut)
                                if (draft_bar_prototypes
                                    and (plan_bar_grid or plan_bar_tokens)) else None),
                    variety_rng=_variety_rng(seed, row_name),
                    beat_anchor=draft_beat_anchor,
                    beat_anchor_per_limb=draft_beat_anchor_per_limb,
                    music_anchor=draft_music_anchor,
                    music_anchor_lag=draft_music_anchor_lag,
                    rhythm_weight=draft_rhythm_weight,
                    music_anchor_shuffle=draft_music_anchor_shuffle,
                    lower_body_delay=draft_lower_body_delay,
                    seam_lead=draft_seam_lead,
                    continue_phrase=draft_continue_phrase,
                    motif_at_phrase=draft_motif_at_phrase,
                    continue_no_replay=draft_continue_no_replay,
                    continue_phrase_novelty=draft_continue_phrase_novelty,
                    phrase_rhythm_keep=draft_phrase_rhythm_keep,
                    tempo_keep=draft_tempo_keep,
                    phrase_chain=draft_phrase_chain,
                    step_lock_keep=draft_step_lock_keep,
                    energy_follow=draft_energy_follow,
                ))
                row_stretch = library.retrieval_stretch_summary()
                _check_unfillable_slots(row_name, row_stretch, allow_unfillable_slots)
                batch_stretches.append(row_stretch)
            draft_batch = torch.stack(
                [_pad_frames(draft, completion_args.seq_len) for draft, _ in conditions]
            ).to(device)
            mask_batch = torch.stack(
                [_pad_frames(mask * ratio, completion_args.seq_len) for _, mask in conditions]
            ).to(device)
            label_batch = None
            if completion_conditions_on_labels:
                label_batch = torch.stack([
                    torch.cat((plan, plan.new_zeros(completion_args.seq_len - len(plan))))
                    if len(plan) < completion_args.seq_len
                    else plan[: completion_args.seq_len]
                    for plan in plans
                ]).to(device)
            normalized_batch = completion.sample(
                music_batch,
                draft_batch,
                mask_batch,
                guidance_weight=guidance_weight,
                start_step=completion_start_step,
                reproject_every=completion_reproject_every,
                # ``sample`` has no ``inpaint_seam_width`` parameter and never
                # had one; this call passed it as a keyword until 2026-09-06,
                # which is a TypeError the moment anything reaches here with the
                # flag set.  Nothing did, because short_batch_mode also requires
                # every clip to be shorter than the completion window and the
                # evaluation clips are 606 frames, so the flag was never inert
                # here -- it was unreachable.  Build the keep mask the way
                # infer_completion does instead, so the two paths agree.
                keep_mask=(_hold_root_to_draft(None, mask_batch,
                                               draft_batch.shape[-1])
                           if completion_keep_root else None),
                sample_steps=completion_sample_steps,
                **({"labels": label_batch} if label_batch is not None else {}),
            ).cpu()
            for (row, length, labels, normalized, query_retrieval_group_id,
                 (_, mask), plan_report) in zip(
                batch,
                lengths,
                plans,
                normalized_batch,
                query_retrieval_group_ids,
                conditions,
                plan_stats,
            ):
                _, _, audio_path, output_path, _, _ = row
                _write_generated_result(
                    output_path,
                    normalized[:length],
                    labels,
                    audio_path,
                    normalizer_path,
                    planner_checkpoint,
                    completion_checkpoint,
                    generation_protocol,
                    deterministic_planner,
                    temperature,
                    generation_protocol=generation_protocol,
                    headline_eligible=headline_eligible,
                    query_retrieval_group_id=query_retrieval_group_id,
                    safe_draft_condition_fraction=_safe_draft_condition_fraction(
                        labels, mask, index_filler=index_filler
                    ),
                    plan_atomic_segments=_plan_atomic_segments(labels),
                plan_postprocess=(plan_report or None),
                    no_plan_conditioning=no_plan_conditioning,
                    unsourced_retrieval=unsourced_retrieval,
                    target_motion_selection=target_motion_selection,
                    dataset_provenance=dataset_provenance,
                    floor_anchor=floor_anchor,
                    fix_skate=fix_skate,
                    fix_skate_seam_blend=fix_skate_seam_blend,
                    face_camera_strength=face_camera_strength,
                    face_camera_window=face_camera_window,
                )
                plan_reports.append(plan_report or None)
    else:
        if ground_truth_plans is not None:
            raise ValueError(
                "ORACLE ground-truth-plan inference currently supports at most one 150-frame slice"
            )
        for index, name, audio_path, output_path, frames, query_retrieval_group_id in pending:
            music = _load_music(audio_path, frames, query_music_dim)
            if len(music) < 3:
                raise ValueError("audio is too short for inference: {}".format(audio_path))
            clip_seed = sample_seed(seed, name)
            seed_everything(clip_seed)
            per_sample_seed[name] = clip_seed
            plan_report = {}
            labels = infer_plan(
                planner,
                music,
                planner_args.seq_len,
                device,
                deterministic=deterministic_planner,
                temperature=temperature,
                vote_window=plan_vote_window,
                min_segment_length=plan_min_segment_length,
                available_labels=library.labels_available,
                plan_stride=plan_stride,
                plan_fusion=plan_fusion,
                plan_vote_tie_break=plan_vote_tie_break,
                plan_transition_policy=plan_transition_policy,
                plan_merge_order=plan_merge_order,
                planner_guidance_weight=planner_guidance_weight,
                planner_transition_logit_bias=planner_transition_logit_bias,
                plan_bar_grid=plan_bar_grid,
                plan_music_repeat=plan_music_repeat,
                plan_bar_beats=plan_bar_beats,
                plan_bar_tokens=plan_bar_tokens,
                bar_pooling=bar_pooling,
                stats=plan_report,
            )
            if plan_from is not None:
                # Planned as usual first, so the per-clip RNG is consumed exactly
                # as in the run being replayed; then that run's plan replaces it.
                labels, plan_report = _replay_plan(plan_from, name, labels)
            # Per clip, so the log describes THIS clip's units and the count
            # gate is a per-clip count rather than a running total.
            library.retrieval_log = []
            draft, noise_mask = _source_safe_draft(
                library,
                labels,
                completion_args.motion_dim,
                query_retrieval_group_id,
                unsourced_retrieval,
                root_continuity=draft_root_continuity,
                gap_fill=draft_gap_fill,
                seam_blend=draft_seam_blend,
                seam_window=draft_seam_window,
                seam_stagger=draft_seam_stagger,
                seam_transition=seam_transition,
                seam_aware_retrieval=draft_seam_aware_retrieval,
                root_velocity_blend=draft_root_velocity_blend,
                root_seam_smooth=draft_root_seam_smooth,
                facing_anchor=draft_facing_anchor,
                target_period=music_settle_period(music),
                no_plan_conditioning=no_plan_conditioning,
                recurrence_variety=draft_recurrence_variety,
                facing_continuity=draft_facing_continuity,
                beat_grid=beat_grid_of(music),
                music=music,
                # One prototype per BAR rather than per label-run.  Off by
                # default so every earlier artifact reproduces from its own
                # command line; see labels_to_segments' docstring for the
                # measurement that motivates it.
                bar_bounds=(quiet_bar_bounds(
                                bar_bounds_of(music, plan_bar_beats,
                                              (plan_report or {}).get("bar_grid_phase")),
                                music, draft_quiet_cut)
                            if (draft_bar_prototypes
                                and (plan_bar_grid or plan_bar_tokens)) else None),
                variety_rng=_variety_rng(seed, name),
                beat_anchor=draft_beat_anchor,
                beat_anchor_per_limb=draft_beat_anchor_per_limb,
                music_anchor=draft_music_anchor,
                music_anchor_lag=draft_music_anchor_lag,
                rhythm_weight=draft_rhythm_weight,
                music_anchor_shuffle=draft_music_anchor_shuffle,
                lower_body_delay=draft_lower_body_delay,
                seam_lead=draft_seam_lead,
                continue_phrase=draft_continue_phrase,
                motif_at_phrase=draft_motif_at_phrase,
                continue_no_replay=draft_continue_no_replay,
                continue_phrase_novelty=draft_continue_phrase_novelty,
                phrase_rhythm_keep=draft_phrase_rhythm_keep,
                tempo_keep=draft_tempo_keep,
                phrase_chain=draft_phrase_chain,
                step_lock_keep=draft_step_lock_keep,
                energy_follow=draft_energy_follow,
            )
            if draft_seam_lead:
                # the skate fix masks the seams as laid down, not the bar lines (see _write_generated_result)
                plan_report = dict(plan_report or {}, draft_seam_frames=list(library.draft_seams))
            retrieval_stretch = library.retrieval_stretch_summary()
            _check_unfillable_slots(name, retrieval_stretch,
                                    allow_unfillable_slots)
            if draft_dump_dir is not None:
                # The draft is the plan turned into motion by retrieval, BEFORE
                # the completion model sees it.  Dumping it is the only way to
                # separate "the plan and the library do not carry beat-hit
                # density" from "the completion destroys it", which is the
                # question the operator asked on 2026-09-01.  Written through
                # the SAME writer as the generated result, from the SAME tensor
                # object that is handed to infer_completion on the next line, so
                # the two are read by the same scorers in the same space and a
                # divergence cannot hide in a second code path.
                _write_generated_result(
                    Path(draft_dump_dir) / output_path.name,
                    draft,
                    labels,
                    audio_path,
                    normalizer_path,
                    planner_checkpoint,
                    completion_checkpoint,
                    generation_protocol,
                    deterministic_planner,
                    temperature,
                    generation_protocol="retrieval-draft (no completion)",
                    headline_eligible=False,
                    query_retrieval_group_id=query_retrieval_group_id,
                    safe_draft_condition_fraction=_safe_draft_condition_fraction(
                        labels, noise_mask, index_filler=index_filler
                    ),
                    plan_atomic_segments=_plan_atomic_segments(labels),
                    plan_postprocess=(plan_report or None),
                    no_plan_conditioning=no_plan_conditioning,
                    unsourced_retrieval=unsourced_retrieval,
                    target_motion_selection=target_motion_selection,
                    dataset_provenance=dataset_provenance,
                    retrieval_stretch=retrieval_stretch,
                    floor_anchor=floor_anchor,
                    fix_skate=fix_skate,
                    fix_skate_seam_blend=fix_skate_seam_blend,
                    face_camera_strength=face_camera_strength,
                    face_camera_window=face_camera_window,
                )
            if draft_only:
                # DIAGNOSTIC.  The retrieval draft written as if it were the
                # generated motion, so every scorer reads it in the same space
                # as a real arm.  It exists to split one question in two: a
                # change to WHICH prototype is pasted can fail either because it
                # did not change the draft or because the completion model does
                # not pass the change through -- and section 14.3 already
                # measured the second happening (fixing 73% of the draft's
                # facing spins moved the output not at all).  Without this the
                # two are indistinguishable from the output alone.
                normalized = draft
            else:
                normalized = infer_completion(
                    completion,
                    music,
                    draft,
                    noise_mask * ratio,
                    completion_args.seq_len,
                    completion_stride,
                    device,
                    guidance_weight,
                    inference_batch_size,
                    labels=(labels if completion_conditions_on_labels else None),
                    # --draft-seam-lead: the seams where the units were actually cut; the label changes and slot
                    # starts would put the inpaint back on the downbeats the lead moved them off.
                    seam_labels=(None if draft_seam_lead else labels),
                    seam_frames=(list(library.draft_seams) if draft_seam_lead else
                                 [int(r["slot_start"]) for r in library.retrieval_log
                                  if r.get("slot_start")]),
                    start_step=completion_start_step,
                    reproject_every=completion_reproject_every,
                    inpaint_seam_width=completion_inpaint_seam_width,
                    beat_free=(_beat_free_profile(
                        music, len(draft), completion_beat_keep, completion_beat_stride,
                        bar_start=((plan_report or {}).get("bar_bounds") or [0, None])[1],
                        held=(_draft_held(draft, normalizer_path) if completion_beat_keep_holds else None),
                        free_max=completion_beat_free_max, hold_frac=completion_beat_hold_frac)
                               if completion_beat_keep else None),
                    inpaint_seam_side=seam_transition,
                    keep_root=completion_keep_root,
                    sample_steps=completion_sample_steps,
                    blend_width=completion_blend_width,
                    draft_guidance_weight=completion_draft_guidance_weight,
                )
            _write_generated_result(
                output_path,
                normalized,
                labels,
                audio_path,
                normalizer_path,
                planner_checkpoint,
                completion_checkpoint,
                generation_protocol,
                deterministic_planner,
                temperature,
                generation_protocol=generation_protocol,
                headline_eligible=headline_eligible,
                query_retrieval_group_id=query_retrieval_group_id,
                safe_draft_condition_fraction=_safe_draft_condition_fraction(
                    labels, noise_mask, index_filler=index_filler
                ),
                plan_atomic_segments=_plan_atomic_segments(labels),
                plan_postprocess=(plan_report or None),
                no_plan_conditioning=no_plan_conditioning,
                unsourced_retrieval=unsourced_retrieval,
                target_motion_selection=target_motion_selection,
                dataset_provenance=dataset_provenance,
                retrieval_stretch=retrieval_stretch,
                floor_anchor=floor_anchor,
                fix_skate=fix_skate,
                fix_skate_seam_blend=fix_skate_seam_blend,
                    face_camera_strength=face_camera_strength,
                    face_camera_window=face_camera_window,
            )
            plan_reports.append(plan_report or None)

    manifest = {
        "samples": len(generated_paths),
        "names": [path.stem for path in generated_paths],
        "output_dir": str(output_dir),
        "planner_checkpoint": str(planner_checkpoint),
        "completion_checkpoint": str(completion_checkpoint),
        "device": str(device),
        "plan_source": generation_protocol,
        "generation_protocol": generation_protocol,
        "headline_eligible": headline_eligible,
        "prototype_retrieval_policy": "EXCLUDE_QUERY_RETRIEVAL_GROUP_FAIL_CLOSED",
        # Which queries' music covers the same stretch of time their own clip
        # does.  In the manifest rather than only on stdout, because a run made
        # with the gate relaxed must be identifiable from its own record.
        "music_span_check": music_span_report,
        "target_motion_selection": target_motion_selection,
        "deterministic_planner": deterministic_planner,
        "planner_temperature": temperature,
        # Everything a rerun needs to land on the same samples.  Two runs that
        # agree on these are bit-identical -- verified across repeat runs and
        # across GPUs -- so a run that omits them is not reproducible from its
        # own record even though the pipeline itself is deterministic.
        "sampling": {
            "seed": seed,
            "temperature": temperature,
            "deterministic_planner": deterministic_planner,
            "completion_stride": completion_stride,
            # None == the pre-2026-08-31 behaviour, where the ramp is the whole
            # overlap and 79.75% of output frames are an average of two draws.
            "completion_blend_width": completion_blend_width,
            # None == the two-term guidance every checkpoint before 2026-09-01
            # used, in which the draft cancels out of the guidance difference.
            "completion_draft_guidance_weight": completion_draft_guidance_weight,
            "plan_stride": plan_stride or planner_args.seq_len,
            "plan_fusion": plan_fusion if plan_stride else "none",
            # Which rule resolved a tied vote.  Before 2026-08-23 there was no
            # choice: ``counts.argmax`` handed every tie to index 0, transition.
            # A manifest without this key was produced by ``index``.
            "plan_vote_tie_break": (
                plan_vote_tie_break
                if (plan_stride and plan_fusion in ("vote", "taper")) else None
            ),
            "plan_transition_policy": plan_transition_policy,
            "plan_merge_order": plan_merge_order,
            "planner_guidance_weight": planner_guidance_weight,
            "planner_transition_logit_bias": planner_transition_logit_bias,
            "plan_bar_grid": plan_bar_grid,
            # Recorded for the same reason plan_bar_beats is: an arm is named
            # by this value and nothing else in the artifact would say so.
            "plan_music_repeat": plan_music_repeat,
            # Recorded because it was NOT, and the omission cost a run: the
            # shipped arm used 2 while the CLI default is 4, so a rerun
            # reconstructed from this manifest reproduced section 15.4's
            # CONTROL (0.493 seg/s, 8.5 segments per clip) instead of the arm
            # the operator watched (0.842, 14.1).  Nothing in the artifact
            # said which one it was; the value had to be inferred from a
            # sentence in the defect log.
            "plan_bar_beats": plan_bar_beats,
            # Whether the planner's tokens were bars.  With it true the vote and
            # merge below did not run at all and the plan_postprocess block
            # records them as null; ``planner_bar_pooling`` is read off the
            # checkpoint, never off a flag, so a manifest cannot claim a pooling
            # the weights were not trained with.
            "plan_bar_tokens": plan_bar_tokens,
            "planner_bar_pooling": bar_pooling,
            # WHICH CHANNELS THE BAR SUMS RATHER THAN MEANS.  Recorded because
            # the pooling changed on 2026-09-08 without any flag changing: the
            # release builder had always SUMMED channel 33 (the onset-peak
            # count) while inference MEANED it, a factor of 58 on the one
            # channel carrying accent density, and the two now agree.  Without
            # this key a pre-fix and a post-fix artifact are indistinguishable
            # -- same commit, same flags, same recorded pooling name -- while
            # the planner's actual input differs by 58x on a rhythm channel.
            "planner_bar_count_channels": (
                list(_BAR_COUNT_CHANNELS) if plan_bar_tokens else None),
            "plan_vote_window": plan_vote_window,
            "plan_min_segment_length": plan_min_segment_length,
            "inference_batch_size": inference_batch_size,
            "guidance_weight": guidance_weight,
            "draft_noise_ratio": draft_noise_ratio,
            "max_frames": max_frames,
            "max_samples": max_samples,
            "ground_truth_labels": ground_truth_labels,
            "plan_from": None if plan_from is None else str(plan_from),
            "draft_root_continuity": draft_root_continuity,
            "draft_gap_fill": draft_gap_fill,
            "draft_seam_blend": draft_seam_blend,
            "draft_seam_window": draft_seam_window,
            "seam_transition": seam_transition,
            "draft_seam_stagger": draft_seam_stagger,
            "draft_seam_aware_retrieval": draft_seam_aware_retrieval,
            "draft_beat_anchor": draft_beat_anchor,
            "draft_beat_anchor_per_limb": draft_beat_anchor_per_limb,
            "draft_music_anchor": draft_music_anchor,
            "draft_music_anchor_lag": draft_music_anchor_lag,
            "draft_music_anchor_shuffle": draft_music_anchor_shuffle,
            "draft_lower_body_delay": draft_lower_body_delay,
            "draft_seam_lead": float(draft_seam_lead or 0.0),
            "draft_seam_lead_applied": getattr(library, "seam_lead_applied", 0),
            "draft_seam_lead_short": getattr(library, "seam_lead_short", 0),
            "draft_seam_lead_window_only": bool(getattr(library, "seam_lead_window_only", False)),
            "draft_continue_phrase": int(draft_continue_phrase or 0),
            "draft_motif_at_phrase": bool(draft_motif_at_phrase),
            "draft_continue_no_replay": bool(draft_continue_no_replay),
            "draft_continue_phrase_novelty": float(draft_continue_phrase_novelty or 0.0),
            "draft_continue_phrase_novelty_starts": getattr(library, "continue_phrase_novelty_starts", 0),
            "draft_phrase_rhythm_keep": float(draft_phrase_rhythm_keep or 0.0),
            "draft_phrase_rhythm_slots": getattr(library, "phrase_rhythm_slots", 0),
            "draft_phrase_rhythm_applied": getattr(library, "phrase_rhythm_applied", 0),
            "draft_tempo_keep": float(draft_tempo_keep or 0.0),
            "draft_phrase_chain": bool(draft_phrase_chain),
            "draft_phrase_chain_slots": getattr(library, "phrase_chain_slots", 0),
            "draft_phrase_chain_applied": getattr(library, "phrase_chain_applied", 0),
            "draft_phrase_chain_seen": getattr(library, "phrase_chain_seen", 0),
            "draft_phrase_chain_full": getattr(library, "phrase_chain_full", 0),
            "draft_step_lock_keep": float(draft_step_lock_keep or 0.0),
            "draft_step_lock_slots": getattr(library, "step_lock_slots", 0),
            "draft_step_lock_applied": getattr(library, "step_lock_applied", 0),
            "draft_step_lock_seen": getattr(library, "step_lock_seen", 0),
            "draft_step_lock_positive": getattr(library, "step_lock_positive", 0),
            "draft_energy_follow": draft_energy_follow or None,
            "draft_energy_follow_slots": getattr(library, "energy_follow_slots", 0),
            "draft_energy_follow_applied": getattr(library, "energy_follow_applied", 0),
            "draft_energy_follow_err_all": round(getattr(library, "energy_follow_err_all", 0.0), 3),
            "draft_energy_follow_err_kept": round(getattr(library, "energy_follow_err_kept", 0.0), 3),
            "draft_hit_slots": getattr(library, "hit_slots", 0),
            "draft_hit_score_all": round(getattr(library, "hit_score_all", 0.0), 3),
            "draft_hit_score_kept": round(getattr(library, "hit_score_kept", 0.0), 3),
            "draft_tempo_applied": getattr(library, "tempo_applied", 0),
            "completion_beat_keep": float(completion_beat_keep or 0.0),
            "completion_beat_stride": int(completion_beat_stride or 1),
            "completion_beat_keep_holds": bool(completion_beat_keep_holds),
            "completion_beat_free_max": float(completion_beat_free_max),
            "completion_beat_hold_frac": float(completion_beat_hold_frac),
            "draft_continue_replay_refused": getattr(library, "continue_replay_refused", 0),
            "draft_continue_phrase_cuts": getattr(library, "continue_phrase_cuts", 0),
            "draft_continue_phrase_phases": list(getattr(library, "continue_phrase_phases", [])),
            "draft_join_top_k": draft_join_top_k,
            "draft_join_pose_weight": draft_join_pose_weight,
            "draft_whole_units": draft_whole_units,
            "draft_bar_units": draft_bar_units,
            "draft_continue_source": draft_continue_source,
            "draft_continue_lookahead": bool(draft_continue_lookahead),
            "draft_continue_any_label": bool(draft_continue_any_label),
            "draft_continue_max_run": int(draft_continue_max_run or 0),
            "draft_prefer_full": float(draft_prefer_full or 0.0),
            "draft_prefer_full_mode": draft_prefer_full_mode,
            "draft_rhythm_scorer": draft_rhythm_scorer,
            "draft_rhythm_keep": float(draft_rhythm_keep or 0.0),
            "draft_rhythm_slots": getattr(library, "rhythm_slots", 0),
            "draft_rhythm_applied": getattr(library, "rhythm_applied", 0),
            "draft_rhythm_shift": int(draft_rhythm_shift or 0),
            "draft_continue_stop": bool(draft_continue_stop),
            "draft_continue_rhythm_min": float(draft_continue_rhythm_min or 0.0),
            "draft_continue_settle": float(draft_continue_settle or 0.0),
            "draft_motif_return": float(draft_motif_return or 0.0),
            "draft_motif_lags": str(draft_motif_lags),
            "draft_motif_max": int(draft_motif_max),
            "draft_motif_pick": str(draft_motif_pick),
            "draft_motif_quiet_skipped": getattr(library, "motif_quiet_skipped", 0),
            "draft_motif_slots": getattr(library, "motif_slots", 0),
            "draft_motif_returned": getattr(library, "motif_returned", 0),
            "draft_motif_unfilled": getattr(library, "motif_unfilled", 0),
            "draft_continuity_scorer": draft_continuity_scorer,
            "draft_continuity_keep": float(draft_continuity_keep or 0.0),
            "draft_continuity_weight": float(draft_continuity_weight or 0.0),
            "draft_continuity_seam_mask": (int(library._continuity[1].get("seam_mask", 6))
                                           if getattr(library, "_continuity", None) is not None else None),
            "draft_continuity_slots": getattr(library, "continuity_slots", 0),
            "draft_continuity_applied": getattr(library, "continuity_applied", 0),
            "draft_continuity_no_context": getattr(library, "continuity_no_context", 0),
            "draft_continuity_joint_slots": getattr(library, "continuity_joint_slots", 0),
            "draft_continuity_joint_changed": getattr(library, "continuity_joint_changed", 0),
            "draft_continue_rhythm_checked": getattr(library, "continue_rhythm_checked", 0),
            "draft_continue_rhythm_refused": getattr(library, "continue_rhythm_refused", 0),
            "draft_rhythm_shift_slots": getattr(library, "rhythm_shift_slots", 0),
            "draft_rhythm_shift_moved": getattr(library, "rhythm_shift_moved", 0),
            "draft_prefer_full_slots": getattr(library, "prefer_full_slots", 0),
            "draft_prefer_full_applied": getattr(library, "prefer_full_applied", 0),
            "draft_continue_eligible": getattr(library, "continue_eligible", 0),
            "draft_continue_applied": getattr(library, "continue_applied", 0),
            "draft_continue_lookahead_slots": getattr(library, "continue_lookahead_slots", 0),
            "draft_continue_lookahead_applied": getattr(library, "continue_lookahead_applied", 0),
            "draft_bar_units_indexed": getattr(library, "bar_units_indexed", None),
            "draft_bar_units_mixed": getattr(library, "bar_units_mixed", None),
            "draft_bar_units_slots": getattr(library, "bar_units_slots", 0),
            "draft_bar_units_fallback": getattr(library, "bar_units_fallback", 0),
            "draft_phase_weight": draft_phase_weight,
            "draft_rhythm_weight": draft_rhythm_weight,
            "draft_feet_lead": draft_feet_lead,
            "draft_beat_span": draft_beat_span,
            "draft_hold_by_music": draft_hold_by_music,
            "draft_hold_music_slots": getattr(library, "hold_music_slots", 0),
            "draft_hold_music_quiet": getattr(library, "hold_music_quiet", 0),
            "draft_hold_music_applied": getattr(library, "hold_music_applied", 0),
            "draft_hold_music_empty": getattr(library, "hold_music_empty", 0),
            "draft_hold_music_blind": getattr(library, "hold_music_blind", 0),
            "draft_beat_span_slots": getattr(library, "beat_span_slots", 0),
            "draft_beat_span_applied": getattr(library, "beat_span_applied", 0),
            "draft_beat_span_empty": getattr(library, "beat_span_empty", 0),
            "draft_beat_span_blind": getattr(library, "beat_span_blind", 0),
            "draft_beat_fit": draft_beat_fit,
            "draft_join_lever_weights": draft_join_lever_weights,
            "draft_feet_beat_lead": draft_feet_beat_lead,
            "feet_beat_lead_slots": getattr(library, "feet_lead_slots", 0),
            "feet_beat_lead_skipped": getattr(library, "feet_lead_skipped", 0),
            "feet_beat_lead_applied": getattr(library, "feet_lead_applied", 0),
            "feet_beat_lead_changed": getattr(library, "feet_lead_changed", 0),
            "draft_quiet_cut": draft_quiet_cut,
            "retrieval_selector_join_band": draft_selector_join_band,
            "retrieval_max_yaw_step": retrieval_max_yaw_step,
            "retrieval_yaw_steps": (str(retrieval_yaw_steps)
                                    if retrieval_yaw_steps else None),
            "retrieval_yaw_slots": getattr(library, "yaw_slots", 0),
            "retrieval_yaw_rejected": getattr(library, "yaw_rejected", 0),
            "retrieval_yaw_exhausted": getattr(library, "yaw_exhausted", 0),
            "retrieval_max_speed_spike": retrieval_max_speed_spike,
            "retrieval_speed_spikes": (str(retrieval_speed_spikes)
                                       if retrieval_speed_spikes else None),
            "retrieval_speed_slots": getattr(library, "speed_slots", 0),
            "retrieval_speed_rejected": getattr(library, "speed_rejected", 0),
            "retrieval_speed_exhausted": getattr(library, "speed_exhausted", 0),
            # 2026-09-17 selection package (DEFECTS 86).  The counters are read
            # at the point of use, so a switch that never reached the code reads
            # zero here however the flag was spelled.
            "draft_join_frame": draft_join_frame,
            "draft_join_placed_calls": getattr(library, "join_placed_calls", 0),
            "draft_join_height_weight": draft_join_height_weight,
            "retrieval_selector_input_units": retrieval_selector_input_units,
            "draft_hop_guard": draft_hop_guard,
            "draft_hop_guard_slots": getattr(library, "hop_guard_slots", 0),
            "draft_hop_guard_applied": getattr(library, "hop_guard_applied", 0),
            "draft_hop_guard_empty": getattr(library, "hop_guard_empty", 0),
            "draft_music_energy": draft_music_energy,
            "draft_music_energy_quiet_loudness": getattr(library, "quiet_loudness", None),
            "draft_music_energy_slots": getattr(library, "music_energy_slots", 0),
            "draft_music_energy_quiet": getattr(library, "music_energy_quiet", 0),
            "draft_music_energy_applied": getattr(library, "music_energy_applied", 0),
            "draft_music_energy_empty": getattr(library, "music_energy_empty", 0),
            "draft_music_energy_blind": getattr(library, "music_energy_blind", 0),
            "retrieval_vertical_flags": (str(retrieval_vertical_flags)
                                         if retrieval_vertical_flags else None),
            "draft_unit_floor": (str(draft_unit_floor) if draft_unit_floor else None),
            "draft_unit_floor_units": getattr(library, "unit_floor_placed", 0),
            "draft_unit_floor_lookups": len(getattr(library, "_local_floor_cache", {})),
            # PRE-EXISTING MISMATCH, recorded rather than refused because every
            # learned-rule arm on the k8 release carries it (DEFECTS 86): the
            # selector's next-label pool statistics are keyed by the label space
            # it was trained on, and nothing maps them to the release's.
            "retrieval_selector_labels_dir": (
                str(selector.provenance.get("args", {}).get("labels_dir"))
                if selector is not None else None),
            "release_label_space_id": _release_label_space(data_root),
            # "the flag was set" and "the band actually narrowed" are different
            # claims, and until 2026-09-16 the selection filters were dead under
            # the shipped rule while the manifest said otherwise (DEFECTS 77).
            # These two are what distinguishes them from the outside.
            "draft_root_seam_height_seams": getattr(library, "root_seam_height_seams", 0),
            "draft_selection_filter_slots": getattr(library, "selection_filter_slots", 0),
            "draft_selection_filter_applied": getattr(library, "selection_filter_applied", 0),
            "retrieval_selector_band_slots": getattr(library, "selector_band_slots", 0),
            "retrieval_selector_band_applied": getattr(library, "selector_band_applied", 0),
            "retrieval_selector_band_skipped": getattr(library, "selector_band_skipped", 0),
            "draft_floor_normalize": (str(draft_floor_normalize)
                                     if draft_floor_normalize else None),
            "release_floor_levelled": levelled,
            "release_floor_reference_m": reference,
            "draft_root_velocity_blend": draft_root_velocity_blend,
            "draft_root_seam_smooth": draft_root_seam_smooth,
            "draft_bar_prototypes": draft_bar_prototypes,
            "completion_start_step": completion_start_step,
            "completion_reproject_every": completion_reproject_every,
            "completion_inpaint_seam_width": completion_inpaint_seam_width,
            "completion_keep_root": completion_keep_root,
            # Recorded from 2026-09-06.  It was accepted and wired from the
            # start but never written down, so runs t_v3 and t_barfull cannot
            # say from their own manifests whether label 0 was retrievable --
            # which decides what a third of their frames contain.
            "index_filler": index_filler,
            "completion_sample_steps": completion_sample_steps,
            "draft_recurrence_variety": draft_recurrence_variety,
            # WHICH IDENTITY the variety draw refuses to replay.  Recorded
            # because the 2026-09-07 fix changed the MEANING of
            # draft_recurrence_variety=True without changing the flag: it
            # used to key on the (window, start, end) triple, which let two
            # adjacent slices of ONE recording pass as different material.
            # Without this key an artifact from before that fix and one
            # from after are indistinguishable in the manifest, which is
            # the defect this repository has already paid for once (two
            # arms differing only by a key nobody wrote down).
            "draft_variety_key": "retrieval_group",
            "draft_facing_continuity": draft_facing_continuity,
            # Recorded from 2026-09-07, for the same reason ``index_filler``
            # above was: both were wired and shipped without ever being written
            # down.  The cost this time was that ``t_v4/final`` and
            # ``t_bfa/a0p6`` -- the kept-root arm and the facing-anchor arm --
            # differed in exactly ONE recorded key (``completion_keep_root``),
            # so nothing in either artifact could say whether the anchor ran.
            # The directory name said ``a0p6``; a directory name is a label I
            # chose, not a measurement.  ``tests/test_manifest_records_every_
            # sampling_option.py`` now enumerates ``infer_directory``'s
            # parameters with ast and fails if any is absent from this dict, so
            # the next flag cannot be added without being recorded.
            "draft_facing_anchor": draft_facing_anchor,
            # The absolute height the body is stood at, in metres, or None for
            # "wherever generation put it". See anchor_floor for why that is
            # not a neutral default.
            "floor_anchor": floor_anchor,
            "fix_skate": fix_skate,
            "fix_skate_seam_blend": fix_skate_seam_blend,
            "face_camera_strength": face_camera_strength,
            "face_camera_window": face_camera_window,
            "derive_retrieval_group": derive_retrieval_group,
            "draft_dump_dir": str(draft_dump_dir) if draft_dump_dir else None,
            # Both are gate overrides, and a gate that was overridden must say
            # so in the artifact or the run reads as one that passed it.
            # ``allow_unfillable_slots`` turns the refusal from section 1.5's
            # slow-motion defect into a warning; ``unsourced_retrieval`` lets a
            # prototype with no recorded source into the draft, which is what
            # ``prototype_retrieval_policy`` above claims cannot happen.
            "allow_unfillable_slots": allow_unfillable_slots,
            "unsourced_retrieval": unsourced_retrieval,
            "completion_conditions_on_labels": completion_conditions_on_labels,
            "no_plan_conditioning": no_plan_conditioning,
            "retrieval_rule": retrieval_rule,
            "retrieval_tie_break": retrieval_tie_break,
            "retrieval_energy_floor": retrieval_energy_floor,
            "draft_only": draft_only,
            "retrieval_selector": retrieval_selector,
            "retrieval_selector_top_k": retrieval_selector_top_k,
            "retrieval_selector_temperature": retrieval_selector_temperature,
            # How often the learned rule actually ran, and how often there was
            # no predecessor to score a join against.  Printed so a reader can
            # tell a run that used the selector from one that fell back to the
            # shipped rule on every segment -- a flag that is recorded but not
            # applied is the defect shape this repository keeps paying for.
            "retrieval_selector_calls": getattr(library, "selector_calls", 0),
            "retrieval_selector_fallbacks": getattr(library, "selector_fallbacks", 0),
            "retrieval_selector_no_predecessor": getattr(
                library, "selector_no_predecessor", 0),
            # Also read off the module rather than a flag, and recorded because
            # the whole-track summary changes what the planner saw without
            # changing any argument on the command line.
            "planner_conditions_on_global_music": (
                planner is not None and planner_wants_global_music(planner)
            ),
            # Per-sample seeds used to be offsets from the base seed by
            # position in this list, which made the draw a property of how the
            # driver sharded the work rather than of the clip.  That was found
            # twice and fixed neither time: on 2026-08-23 the same arm, same
            # code, same base seed and same 65 clips run over 2 shards and over
            # 6 gave fid_k 9.997 and 9.515; on 2026-08-31 the same checkpoint
            # and config on 24 clips read mean energy 0.5632 against 0.6421
            # from a clip-list length change alone -- 14%, wider than most of
            # the arm-to-arm differences the 2026-08-30 waves were read off.
            # ``sample_seed`` now derives the draw from the clip's NAME.
            #
            # ``seed_pairing`` says whether two runs of this artifact can be
            # compared per clip at all.  "per-clip" (batch size 1) means the
            # same clip gets the same draw in any run.  "batch" means the draw
            # is shared with whoever the clip was batched beside, so a run over
            # a different clip LIST is still unpaired -- comparable in aggregate,
            # not per clip.  Stated rather than left to be inferred, because
            # inferring it is what failed twice.
            "seed_pairing": "per-clip" if inference_batch_size == 1 else "batch",
            "sequence_order": [path.stem for path in generated_paths],
            "per_sample_seed": per_sample_seed,
        },
        # What the plan post-process actually did across this run.  It answers
        # one question no earlier artifact could: how much of the transition in
        # these plans is the planner, and how much is a label the retrieval
        # library could not serve being rewritten to 0.
        "plan_postprocess_totals": _plan_postprocess_totals(plan_reports),
        "code_revision": _code_revision(),
        "dataset_provenance": dataset_provenance,
        # Which release each checkpoint was trained on against the library's own
        # derived_from; "match" is the only state a shipped arm should carry.
        "release_binding": release_binding,
        "ingest_root": str(ingest_root),
        "audio_dir": str(audio_dir),
    }
    with open(str(output_dir / "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


class _ArgvFileParser(argparse.ArgumentParser):
    """``@file`` argv with one token per line; blank lines and ``#`` comments skipped.

    This is how the T line's current generation config is named
    (``configs/t_line/current.argv``): the shipped flag set lives in one tracked
    file instead of a scratch argv, and anything written after ``@file`` on the
    command line overrides it -- argparse expands the file in place, so later
    tokens win.  Nested ``@`` lines are expanded too, which is what lets
    ``current.argv`` be a one-line pointer at the version it selects.
    """

    def convert_arg_line_to_args(self, arg_line):
        line = arg_line.strip()
        if not line or line.startswith("#"):
            return []
        return [line]


def _generation_config(argv):
    """The literal argv and every ``@`` file it pulled in, with their sha256."""
    files, seen = [], set()

    def walk(token):
        path = Path(token[1:])
        if str(path) in seen or not path.is_file():
            return
        seen.add(str(path))
        data = path.read_bytes()
        files.append({"path": str(path), "sha256": hashlib.sha256(data).hexdigest()})
        for line in data.decode("utf-8").splitlines():
            line = line.strip()
            if line.startswith("@"):
                walk(line)

    for token in argv:
        if token.startswith("@"):
            walk(token)
    return {"argv": list(argv), "files": files}


def parse_args():
    parser = _ArgvFileParser(description="Run two-stage atomic dance inference",
                             fromfile_prefix_chars="@")
    parser.add_argument("--allow-cross-release-checkpoints", action="store_true",
                        help="accept checkpoints trained on a different release than "
                             "the one --data-root was derived from (an ablation that "
                             "swaps the library under fixed models); recorded in the "
                             "manifest's release_binding")
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--music-span-check", choices=("refuse", "warn", "off"),
                        default="refuse",
                        help="compare each query's music length against its own "
                             "clip's wav before generating. The generated length "
                             "IS the music length, so a music array from another "
                             "cut of the same recording yields a self-consistent "
                             "result nothing downstream can tell from a correct "
                             "one. Measured 11.2%% of runs/wild_v4_acct_gt_eval "
                             "on 2026-08-24. `warn` records it in the manifest "
                             "and continues -- which is what an artifact made "
                             "before this gate needs in order to reproduce")
    parser.add_argument("--ingest-root", default=DEFAULT_INGEST_ROOT,
                        help="where the per-clip wavs live for that comparison")
    parser.add_argument("--output-dir", default="eval/generated_motions")
    parser.add_argument("--target-motion-dir")
    parser.add_argument("--planner-checkpoint", default=DEFAULT_PLANNER_CHECKPOINT)
    parser.add_argument("--completion-checkpoint", default=DEFAULT_COMPLETION_CHECKPOINT)
    parser.add_argument("--data-root", default="data/atomic_aistpp")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--overwrite", action="store_true")
    sampler = parser.add_mutually_exclusive_group()
    sampler.add_argument(
        "--deterministic-planner",
        action="store_true",
        help="use argmax reverse steps for debugging instead of D3PM sampling",
    )
    sampler.add_argument(
        "--stochastic-planner",
        dest="deterministic_planner",
        action="store_false",
        help="sample every categorical reverse step (default)",
    )
    parser.set_defaults(deterministic_planner=False)
    parser.add_argument("--temperature", type=float, default=1.0)
    # Defaults to the window, which is the non-overlapping behaviour every
    # artifact so far was generated with.  Lowering it overlaps the planner's
    # windows; see docs/MUSIC_CONDITIONING_ALIGNMENT.md for the measurement that
    # says why (84 of 408 generated segment boundaries sit on a chunk edge,
    # against 2.7 expected, while ground truth gives 0.76x).
    # Default "off" rather than the better value, for the same reason
    # --plan-stride defaults to the window: every artifact produced before
    # 2026-08-17 must still reproduce from its own command line.
    parser.add_argument(
        "--draft-root-continuity",
        choices=("off", "xy", "xyz"),
        default="off",
        help="translate each retrieved prototype so its root continues from the "
             "previous segment instead of being pasted at its own recording's "
             "absolute position; xy leaves height alone",
    )
    parser.add_argument(
        "--plan-bar-beats", type=int, default=4, metavar="BEATS",
        help="how many beats the bar grid puts in one segment. The default 4 is "
             "what M1 cut on and what every artifact before 2026-08-30 used, and "
             "at 120 BPM it makes every segment exactly 2.00 s -- measured, and "
             "against a ground truth whose own atomic segments run 0.98 s. That "
             "is why the gridded arm emits 0.461 atomic segments per second "
             "against the ground truth's 0.793: the grid, not the planner, sets "
             "the rate. 2 is the value that lands between the gridded arm and "
             "the ungridded one (0.5 s per segment, 28.4%% filler)",
    )
    parser.add_argument(
        "--draft-facing-continuity", action="store_true",
        help="turn each retrieved prototype about the vertical axis so its first "
             "frame faces the way the previous segment ended. Off by default so "
             "every earlier artifact reproduces from its own command line. "
             "Measured defect it exists for: 92.9%% of the draft's violent turns "
             "(> ground truth's own pooled p99.5 of 549 deg/s) sit on a plan "
             "boundary, 11.84x their share of frames, while away from boundaries "
             "the draft turns LESS than ground truth (57.8 vs 62.8 deg/s) -- the "
             "library does not contain these turns, the paste manufactures them.",
    )
    parser.add_argument(
        "--draft-recurrence-variety", action="store_true",
        help="give each REPEAT of an atomic label a different prototype of the same "
             "class, drawn inside the duration rule's own tolerance. Without it the "
             "duration rule is a deterministic min over a cached key, so every repeat "
             "of a label in a clip is the identical tensor -- and the bar grid makes "
             "every segment the same length, so the key collides on every repeat. The "
             "paper asks for exactly this ('structured movements should exhibit "
             "variation when they recur'); confining the draw to the duration "
             "tolerance is what keeps it from becoming the paper's own Random Choice "
             "ablation, which is worse on FID because unconstrained draws need heavier "
             "time-stretching",
    )
    parser.add_argument(
        "--completion-sample-steps", type=int, default=None, metavar="N",
        help="respace the reverse chain to N evenly spaced steps instead of all "
             "of them. The falsification test for the single-axis reading of "
             "the other sampler switches: the arm-span drift accumulates over "
             "the steps, so fewer steps means less drift WITHOUT pulling the "
             "output toward the retrieval draft, which is the one thing every "
             "other switch does",
    )
    parser.add_argument(
        "--completion-reproject-every", type=int, default=None, metavar="N",
        help="put the reverse chain's iterate back on the draft's manifold every "
             "N steps. Follows from the measured mechanism -- the denoiser is "
             "unbiased on the manifold and biased off it, and its own iterates "
             "leave and never return. Arm span 0.872 -> 0.674 m against a ground "
             "truth of 0.630, at a 15%% cost in distance from the draft; one "
             "correction in a 100-step chain is enough",
    )
    parser.add_argument(
        "--completion-keep-root", action="store_true",
        help="hold the root translation to the retrieval draft instead of "
             "letting the completion regenerate it. The draft's root is "
             "retrieved ground truth and measures at ground-truth level "
             "(stepping band 0.0762 against 0.0780, path 4.52 m against "
             "5.00 m, foot skate 0.382 against 0.374); the completion's, on "
             "the same six plans, reads 0.1299 / 10.63 m / 0.675. See "
             "_hold_root_to_draft.")
    parser.add_argument(
        "--completion-inpaint-seam-width", type=int, default=None, metavar="FRAMES",
        help="hold the completion to the draft everywhere EXCEPT within this "
             "many frames of a seam, at every denoising step. The draft is a "
             "suggestion the model can and does ignore: measured 2026-09-05 on "
             "the T line, three draft-side fixes that worked on the draft "
             "(seam-aware retrieval, facing continuity, --index-filler) reached "
             "the final dance at 5-17%%, and no lever moved that -- guidance "
             "1.0-4.5, a 600-epoch retrain with --draft-timing-align, "
             "--completion-start-step 100->30 (4.6%% closer to the draft) and "
             "--completion-reproject-every 10 (8.6%%). This makes the kept "
             "frames a constraint instead. The seams are deliberately NOT kept, "
             "so the model still generates every transition")
    parser.add_argument(
        "--completion-start-step", type=int, default=None, metavar="STEP",
        help="begin the reverse diffusion chain from the noised draft at this "
             "step instead of from pure noise at the last one. Diagnostic for "
             "where in the chain the limb synchrony is re-imposed: the draft "
             "carries a real dancer's limb structure (lag-0 share 0.667 after a "
             "15-frame seam blend, ground truth 0.500) and the output does not "
             "(0.833). Report how close the output sits to its own draft "
             "whenever this is used -- a short chain is closer to retrieval "
             "than to generation",
    )
    parser.add_argument(
        "--allow-unfillable-slots", action="store_true",
        help="proceed when a plan slot is longer than every prototype in its "
             "class, instead of refusing. Such a slot is filled by stretching "
             "the longest available prototype, i.e. played in slow motion: "
             "measured 2026-09-05 on the T line, 8 of 18 eval clips hit this "
             "and the worst played 10.4 of its 14.3 s at 0.48x speed while "
             "every summary statistic read clean (pooled stretch median 1.000). "
             "Use it only to reproduce a pre-2026-09-05 artifact byte for byte; "
             "the fix is --draft-bar-prototypes.")
    parser.add_argument(
        "--draft-bar-prototypes", action="store_true",
        help="with --plan-bar-grid or --plan-bar-tokens, retrieve one prototype per BAR instead of "
             "one per label run. Two adjacent bars that draw the same label "
             "otherwise merge into a single retrieval unit, and one prototype "
             "is stretched across all of it: measured on the T corpus, plan "
             "runs reach 2.40 s median and 6.19 s at p90 against a 2.00 s "
             "vocabulary, while the ground truth's own runs are 1.90 s / 3.86 "
             "s. Off by default so earlier artifacts reproduce")
    parser.add_argument(
        "--draft-seam-blend", type=int, default=0, metavar="FRAMES",
        help="cross-fade half-width across a prototype-to-prototype seam; 0 is "
             "the published behaviour, which butt-joints them. The two flags "
             "beside this one cannot reach that seam: root continuity only "
             "moves the root and gap fill only touches transition frames, while "
             "on the bar-grid arm 90.9%% of plan boundaries are "
             "prototype-to-prototype",
    )
    parser.add_argument(
        "--draft-seam-window", choices=("triangle", "cosine"), default="triangle",
        help="envelope shape for --draft-seam-blend. 'triangle' is the published "
             "behaviour: 1 - |i-centre|/half_width, whose CORNER at the seam "
             "injects its own jerk, more sharply the narrower the blend -- which "
             "is why seam jerk is non-monotonic in the width (2026-09-05, 17 eval "
             "clips, filler excluded: ground truth 0.2553, no blend 0.3515, "
             "half-width 4 0.5139 i.e. WORSE than no blend, 6 0.3195, 8 0.2272). "
             "'cosine' is the same support with a C1 envelope. Default stays "
             "'triangle' so earlier artifacts reproduce")
    parser.add_argument(
        "--seam-transition", choices=("centred", "after"), default="centred",
        help="where the change from one retrieved unit to the next happens. "
             "'centred' (published) cross-fades the draft and frees the "
             "completion over +-width around every seam. 'after' keeps the "
             "ARRIVING unit intact up to the bar line and makes the transition "
             "in the frames after it: no limb cross-fade in the draft, and "
             "--completion-inpaint-seam-width frees [seam, seam+width). WHY "
             "(2026-09-22, ten vis clips, fix7): a bar seam is a downbeat, where "
             "ground truth decelerates into its pose (wrist speed 1.01x its median "
             "at -3 frames); the draft has the arriving unit's stop there (0.86x "
             "at -4) but the centred window regenerates it, so the final moves "
             "FASTEST there (1.31x at -3) and shows a compromise between the two "
             "units' poses -- the operator's 'pose 做到一半就收了'. Refuses "
             "--draft-seam-stagger (nothing to stagger)")
    parser.add_argument(
        "--draft-seam-stagger", action="store_true",
        help="offset each limb's seam cross-fade so the four limbs do NOT change "
             "on the same frame. Implemented since the blend landed and reachable "
             "from nothing until now, which is why it has never been scored. What "
             "it is for: a single library prototype already has a real dancer's "
             "limb structure -- cross-correlation peaks at lag 0 on 0.000 of limb "
             "pairs against ground truth's 0.500 -- and butt-joining two of them "
             "reads 1.000, because every seam is one instant at which all four "
             "limbs change together. A synchronous cross-fade softens the jolt "
             "and leaves that at 1.000; only the stagger separates them. It is "
             "the lever for the operator's '用不同肢体部位换着卡点' and for the "
             "cost --draft-beat-anchor charges for fixing the phase: measured "
             "2026-09-08 on the 20 T-series clips, per-part settle spread "
             "(sd/mean across feet, hips+knees, torso, shoulders, hands) is 0.578 "
             "for ground truth, 0.222 at anchor 1.4 and 0.051 at anchor 2.0 -- "
             "the anchor lands the phase by braking every part on the same frame, "
             "which is the texture it was meant to restore. REQUIRES "
             "--draft-seam-blend > 0; with no cross-fade there is nothing to "
             "stagger and the flag is silently inert, so it is refused instead")
    parser.add_argument(
        "--draft-seam-aware-retrieval", action="store_true",
        help="choose each prototype by how well it CONTINUES the previous unit, "
             "not only by duration. Within the same duration band the plain "
             "rule already uses (max(2, 0.15*len), so the length guarantee is "
             "unchanged), rank candidates by pose and velocity continuity with "
             "the last frames already written, and take the best. Off by "
             "default so earlier artifacts reproduce; the criterion is this "
             "repository's own construction, not the paper's, and its "
             "positive control is in tests/test_seam_aware_retrieval.py")
    parser.add_argument(
        "--draft-beat-anchor", type=float, default=0.0, metavar="MAX_STRETCH",
        help="put each prototype's OWN settle points on the query's beats, "
             "instead of letting one global linear resample drag them wherever "
             "the stretch happens to land. The operator, 2026-09-07: "
             "'\u52a8\u4f5c\u5361\u70b9\u8fd8\u662f\u4e0d\u5982 gt ... "
             "\u4e0d\u591f\u8212\u5c55\u5230\u4f4d'; the reading that "
             "agrees is settle (deceleration INTO the beat) at ground truth "
             "+0.0569 against the shipped arm's -0.0769, i.e. ground truth "
             "arrives and rests on the beat while the generated arm is still "
             "accelerating through it. The warp is monotone and "
             "length-preserving, so it moves accents without changing which "
             "frames a slot owns; the value is the per-segment stretch cap and "
             "a pairing that would exceed it is DROPPED rather than clamped "
             "(a clamped anchor no longer lands where it claims). Source "
             "anchors are the paper's own motion beat (atomicDance 3.2, local "
             "minima of joint velocity) and targets are the bar planner's beat "
             "grid. 0 = off, so earlier artifacts reproduce. Positive control "
             "in tests/test_draft_beat_anchor.py")
    parser.add_argument(
        "--completion-beat-free-max", type=float, default=1.0, metavar="W", dest="completion_beat_free_max",
        help="with --completion-beat-keep: the re-draw weight at the middle of a beat interval (1 = fully re-drawn, "
             "0.5 = half draft / half re-drawn)")
    parser.add_argument(
        "--completion-beat-hold-frac", type=float, default=0.5, metavar="F", dest="completion_beat_hold_frac",
        help="with --completion-beat-keep-holds: an interval counts as held when more than this fraction of it is")
    parser.add_argument(
        "--completion-beat-keep-holds", action="store_true", dest="completion_beat_keep_holds",
        help="with --completion-beat-keep: leave an anchor interval entirely to the draft when the draft's arms hold "
             "still over most of it, so the completion does not put motion where the dancer held a shape")
    parser.add_argument(
        "--completion-beat-stride", type=int, default=1, metavar="N", dest="completion_beat_stride",
        help="with --completion-beat-keep: anchor only every N-th beat counted from the plan's first bar line "
             "(2 = beats 1 and 3), the intervals between anchors re-drawn as one")
    parser.add_argument(
        "--completion-beat-keep", type=float, default=0.0, metavar="K", dest="completion_beat_keep",
        help="with --completion-inpaint-seam-width: keep the draft within K of each beat interval around EVERY beat of "
             "the query's grid and let the completion re-draw the middle of each interval (raised cosine), so every "
             "beat lands on a real dancer's pose and the travel between is generated to this song.  0 = off")
    parser.add_argument(
        "--draft-tempo-keep", type=float, default=0.0, metavar="F", dest="draft_tempo_keep",
        help="keep the fraction F of candidates whose own length is closest to the slot's (played nearest their "
             "dancer's own speed; the duration band allows +-15%%).  0 = off")
    parser.add_argument(
        "--draft-energy-follow", default=None, metavar="SPEC", dest="draft_energy_follow",
        help="K series: make the dance calmer where the song is quiet and busier where it is loud.  SPEC is "
             "'keep=F,smooth=B,lo=L,hi=H,feature=loudness|onset,metric=speed|points|both,hit=K' (defaults 0.5, 2, 0.15, "
             "0.85, loudness, speed, 0; hit=K then keeps the K of candidates that settle most on the beat as the camera "
             "sees it): each bar's "
             "loudness, averaged over +-B bars, ranked within the song, maps to a target percentile in [L, H] of the "
             "library's bar speeds; keep the fraction F of candidates (with their phrase continuation) nearest it.  "
             "Whole songs of F11/J7 follow loudness at Spearman ~0.  Query music and library motion only.  Off by default")
    parser.add_argument(
        "--draft-step-lock-keep", type=float, default=0.0, metavar="F", dest="draft_step_lock_keep",
        help="keep the fraction F of candidates whose FOOT LANDINGS fall 0.2 beat after this song's beats when played "
             "into the slot (at a phrase start: over the whole phrase continuation will play).  Ground truth's feet "
             "lock to its own song's beats (pooled phase-lock 0.152, another song's grid 0.014); F11's 0.096 -- the "
             "operator's '步伐卡节奏偏少'.  Query beats and library motion only.  0 = off")
    parser.add_argument(
        "--draft-phrase-chain", action="store_true", dest="draft_phrase_chain",
        help="with --draft-continue-source/--draft-continue-phrase: at a phrase start keep the candidates whose source "
             "dancer can be continued through the REST of the phrase (the longest chain of next source bars that fit "
             "the query's remaining bars).  On F11, 82 of 195 in-dance cuts were the source upload ending or its next "
             "bar missing the duration band -- mid-phrase joins between unrelated dancers nobody chose.  Library "
             "segmentation and the query's bar grid only")
    parser.add_argument(
        "--draft-phrase-rhythm-keep", type=float, default=0.0, metavar="F", dest="draft_phrase_rhythm_keep",
        help="with --draft-continue-source/--draft-continue-phrase and --draft-rhythm-scorer: at a phrase start keep the "
             "fraction F of candidates whose whole phrase (the bar and the source bars continuation will play after "
             "it) scores best against THIS song's bars, so the continued bars follow the query's rhythm too.  0 = off")
    parser.add_argument(
        "--draft-continue-phrase-novelty", type=float, default=0.0, metavar="Q", dest="draft_continue_phrase_novelty",
        help="with --draft-continue-source: start a phrase (retrieve afresh) at bar lines where the MUSIC changes -- "
             "local maxima of timbre/harmony novelty at or above the clip's Q-quantile, phrases 2-8 bars -- instead "
             "of every --draft-continue-phrase bars.  0 = off")
    parser.add_argument(
        "--draft-continue-no-replay", action="store_true", dest="draft_continue_no_replay",
        help="with --draft-continue-source: never continue into source frames this clip has already played, so a "
             "motif return stays one move coming back instead of replaying the passage that followed it (the "
             "verbatim A B A B the operator saw in 日不落 F5)")
    parser.add_argument(
        "--draft-motif-at-phrase", action="store_true", dest="draft_motif_at_phrase",
        help="with --draft-motif-return and --draft-continue-phrase: return only at phrase starts, so the returned "
             "bar's continuation replays the whole earlier phrase (A A') instead of one bar cutting a phrase")
    parser.add_argument(
        "--draft-continue-phrase", type=int, default=0, metavar="BARS", dest="draft_continue_phrase",
        help="with --draft-continue-source: continue the source dancer only INSIDE a phrase of this many bars "
             "and retrieve afresh at every phrase start, the phrase phase read off the song (the bar line with "
             "the largest timbre/harmony change).  So a phrase is one dancer's real choreography and the dance "
             "changes where the music does.  0 = off (continuation cuts only where the next source bar does "
             "not fit)")
    parser.add_argument(
        "--draft-seam-lead", type=float, default=0.0, metavar="BEATS", dest="draft_seam_lead",
        help="cut from one retrieved unit to the next this many of the query's beats BEFORE the bar line, "
             "reading the incoming unit's lead-in from its own recording, so every downbeat -- the approach, "
             "the arrival and what follows -- is one real dancer's and the change of dancer happens mid-flight "
             "on the '4-and'.  The draft fade, root smoothing, completion inpaint and skate mask all move to "
             "the new seams; continued bars (--draft-continue-source) keep their source's bar line and get no "
             "seam.  The mirror of --seam-transition after, which kept the arrival and moved the change to "
             "just after the downbeat.  Reads the query's beat period only.  0 = off")
    parser.add_argument(
        "--draft-lower-body-delay", type=float, default=0.0, metavar="BEATS",
        help="run the legs, feet, contacts and root a fraction of a beat LATER "
             "than the arms. Measured 2026-09-11 in absolute m/s (no z-scoring, "
             "which is what made settle misleading): foot speed by beat phase "
             "is slowest at 0.12 for ground truth, just after the beat, and at "
             "0.88 for ours, just before it, while the modulation DEPTH already "
             "matches (1.156 against 1.164). We brake a quarter beat early, so "
             "the repair is a shift and not a warp -- which is why every "
             "--draft-beat-anchor strength made it worse, pulling already-early "
             "settle points further onto the beat. 0.25 is the measured offset")
    parser.add_argument(
        "--draft-floor-normalize", metavar="JSON", default=None,
        help="express every retrieved prototype relative to ITS OWN recording's "
             "floor, from a JSON of per-recording floors (the 5th percentile of "
             "the lowest foot, the rule --floor-anchor and the renderer already "
             "use). A monocular reconstruction has no absolute height and this "
             "corpus disagrees about the ground: over the 295 T sequences the "
             "floor runs -1.074 m at the 5th percentile of recordings to -0.883 "
             "at the 95th, a 0.192 m spread. Pasting a bar from one recording "
             "into a clip built on another and making the ROOT continuous at the "
             "seam then lifts the FEET, because the two prototypes hold their "
             "feet different distances below the root. Measured on "
             "7412632116703350028:clip001, median height of the lowest foot "
             "above the render floor: ground truth 0.072 m, shipped 0.070, "
             "--draft-join-pose-weight 0.25 0.140, seam-aware ranking off 0.439 "
             "-- with every arm's own 5th percentile ON the floor, so the anchor "
             "is working and what floats is everything above it. Refuses to run "
             "with a root continuity that includes z, which would undo it")
    parser.add_argument(
        "--draft-continue-source", default=None, metavar="SEGMENTATION_JSON",
        dest="draft_continue_source",
        help="at a bar line, if the unit just laid down has a NEXT SOURCE BAR (same "
             "recording, the segmentation's next bar, label-pure) with the planned "
             "label and a length inside the duration band, play that bar instead of "
             "retrieving -- the bar line becomes the source dancer's own and not a "
             "cut, and it is left out of the seam fade. Measured 2026-09-23 "
             "(DEFECTS 92): every cut makes the body travel from one dancer's "
             "arrival to another's start, and the output's extra wrist turn-backs "
             "sit in that travel -- a turn-back within 16 frames after the seam at "
             "0.71 of seams against ground truth's 0.48 at the same frames, while the "
             "library itself turns back at ground truth's rate. Needs --seam-transition "
             "after. Off by default so earlier artifacts reproduce")
    parser.add_argument(
        "--draft-continue-any-label", action="store_true", dest="draft_continue_any_label",
        help="with --draft-continue-source: continue into the source's next bar whatever "
             "its label (duration band still applies).  The planner's label is the only "
             "thing that stops a continuation otherwise, and the planner sits below the "
             "mode floor on val (worklog 2026-09-08, 2026-09-22)")
    parser.add_argument(
        "--draft-continue-max-run", type=int, default=0, metavar="BARS",
        dest="draft_continue_max_run",
        help="with --draft-continue-source: after this many continued bars in a row, "
             "retrieve afresh, so one source dancer cannot take over the clip.  0 = no cap")
    parser.add_argument(
        "--draft-prefer-full", type=float, default=0.0, metavar="FRACTION",
        dest="draft_prefer_full",
        help="among the candidates already fitting the slot, keep the FULLEST fraction (arm raise, "
             "elbow straightness and reach at their 90th percentile over the played range, world "
             "joints) before the join band and the selector choose.  The shipped chain picks units "
             "at the 41st percentile of fullness inside their pool (DEFECTS 92).  0 = off")
    parser.add_argument(
        "--draft-rhythm-scorer", default=None, metavar="CKPT", dest="draft_rhythm_scorer",
        help="learned music-motion alignment scorer (tools/train_rhythm_scorer.py).  With "
             "--draft-rhythm-keep, each slot's candidates are scored against the QUERY's music under "
             "the slot, as they will play, and only the best-aligned fraction goes on to the join band "
             "and the selector.  Reads the query's music only")
    parser.add_argument(
        "--draft-rhythm-keep", type=float, default=0.0, metavar="FRACTION", dest="draft_rhythm_keep",
        help="fraction of the candidates the rhythm scorer keeps (0 = off)")
    parser.add_argument(
        "--draft-continue-stop", action="store_true", dest="draft_continue_stop",
        help="with --draft-continue-source and --seam-transition after: continued bar lines keep the "
             "after-seam coast, i.e. a brief stop on the downbeat, then the SAME dancer carries on")
    parser.add_argument(
        "--draft-continue-settle", type=float, default=0.0, metavar="DEPTH", dest="draft_continue_settle",
        help="continued bar lines settle by TIME-WARPING the dancer's own motion: playback rate dips to 1-DEPTH "
             "3 frames after the line and the time is made up over the bar (ends exactly on the bar).  "
             "Replaces --draft-continue-stop's coast+fade, which left a catch-up burst after every downbeat")
    parser.add_argument(
        "--draft-motif-return", type=float, default=0.0, metavar="P", dest="draft_motif_return",
        help="at each whole bar k, with probability P, play again the unit this clip played at bar k-lag (first "
             "qualifying lag of --draft-motif-lags), restretched to the slot: a move that comes back on the phrase "
             "grid, which ground truth does in 19%% of its bars and the variety draw makes impossible (DEFECTS §94). "
             "Reads only the bar index of the beat grid.  0 = off")
    parser.add_argument(
        "--draft-motif-lags", default="4,2", metavar="LAGS", dest="draft_motif_lags",
        help="comma list of bar lags a motif may come back from, tried in order (each >= 2)")
    parser.add_argument(
        "--draft-motif-max", type=int, default=2, metavar="N", dest="draft_motif_max",
        help="at most N returns per clip")
    parser.add_argument(
        "--draft-motif-pick", choices=("first", "lively"), default="first", dest="draft_motif_pick",
        help="which earlier unit may come back: 'first' qualifying lag, or 'lively' = only a unit at least as energetic "
             "as the median unit placed so far (a motif, not a transition)")
    parser.add_argument(
        "--draft-continuity-scorer", default=None, metavar="CKPT", dest="draft_continuity_scorer",
        help="learned continuation scorer (tools/train_continuation_scorer.py): scores each retrieval candidate, "
             "played over its slot, as the NEXT BAR after the motion the draft has already placed (its last two "
             "slots, generated motion only -- never the target clip's).  Needs --draft-continuity-keep or "
             "--draft-continuity-weight")
    parser.add_argument(
        "--draft-continuity-keep", type=float, default=0.0, metavar="FRACTION", dest="draft_continuity_keep",
        help="keep the best-continuing fraction (at least two) of each slot's candidates, in their original order, "
             "after --draft-rhythm-keep and before the join band.  0 = off")
    parser.add_argument(
        "--draft-continuity-weight", type=float, default=0.0, metavar="W", dest="draft_continuity_weight",
        help="instead of a second filter, rank jointly by z(rhythm) + W * z(continuity) inside --draft-rhythm-keep "
             "(same kept fraction).  Needs --draft-rhythm-scorer and --draft-rhythm-keep.  0 = off")
    parser.add_argument(
        "--draft-continue-rhythm-min", type=float, default=0.0, metavar="Q", dest="draft_continue_rhythm_min",
        help="with --draft-rhythm-scorer and continuation: continue into the source's next bar only if its "
             "learned alignment with the slot's music is at or above the Q-quantile of the slot's candidate "
             "pool; otherwise cut and let the scorer choose.  0 = always continue when possible")
    parser.add_argument(
        "--draft-rhythm-shift", type=int, default=0, metavar="FRAMES", dest="draft_rhythm_shift",
        help="with --draft-rhythm-scorer: slide each retrieved unit's source range by up to this many "
             "frames either way and keep the offset the scorer finds best aligned with the slot's music; "
             "continued bars inherit their run's offset.  0 = off")
    parser.add_argument(
        "--draft-prefer-full-mode", choices=["frames", "stops", "both", "frames_slow"], default="frames",
        dest="draft_prefer_full_mode",
        help="what --draft-prefer-full ranks: 'frames' = 90th percentile of arm raise / elbow "
             "straightness / reach over the played range; 'stops' = their median AT THE ARRIVALS "
             "(local minima of wrist speed after >= 5 cm of travel), i.e. land extended rather than "
             "sweep through it")
    parser.add_argument(
        "--draft-continue-lookahead", action="store_true",
        dest="draft_continue_lookahead",
        help="with --draft-continue-source: when choosing a unit, narrow the pool to "
             "candidates whose own next source bar can fill the NEXT slot, so the "
             "continuation has something to continue (a narrowing like the other "
             "selection filters: an empty result keeps the pool whole). Counted")
    parser.add_argument(
        "--draft-bar-units", default=None, metavar="SEGMENTATION_JSON",
        help="add a second library index by SOURCE BAR (the segmentation's "
             "boundaries, e.g. runs/txy_t_seg_beat4/segmentation.json): each bar "
             "lying wholly inside a release window is one candidate. A slot of the "
             "bar's beat count (+-0.5) takes whole bars; a partial or odd slot keeps "
             "the label-run index. A run clipped by the window edge starts or stops "
             "mid-bar (31 of fix7's 101 units, cut a median 7 frames off the bar "
             "line), i.e. mid-move; 13 of them sat in ordinary four-beat slots")
    parser.add_argument(
        "--draft-whole-units", action="store_true",
        help="retrieve only prototypes that are a COMPLETE label run, dropping "
             "the ones a 150-frame release window cut short. Every library unit "
             "is meant to be four beats of its own song, which is what lets a "
             "linear stretch onto four beats of the target map beat to beat; a "
             "window that lands mid-segment indexes the fragment it contains as "
             "if it were a unit, and %%(prog)s's duration rule cannot tell three "
             "beats of a slow song from four of a fast one because they have the "
             "same frame count. Measured over the T library: 10,375 of 15,203 "
             "candidates (68.2%%) touch a window edge, median length 40 frames "
             "against 61 for whole runs, p10 nine frames. Dropping them leaves "
             "4,828 candidates, median 236 per class, minimum 81, over 20 to 64 "
             "recordings each. Refuses rather than falls back if a class empties")
    parser.add_argument(
        "--draft-quiet-cut", type=int, default=0, metavar="FRAMES",
        help="slide each retrieval cut by up to this many frames onto the "
             "QUIETEST frame near it, so the seam stops landing on the beat.  "
             "The plan still says what to dance over which bar; only the "
             "boundary moves.  Why: the strongest onset inside a bar sits "
             "within 3 frames of the bar line 44.8%% of the time (3.7x chance) "
             "and our seam is on that line 100%% of the time, so the pipeline "
             "puts its largest artifact where the music is loudest -- the "
             "output's top-10%% speed changes are 21.4%% at a bar line against "
             "ground truth's 13.7%%, which is chance.  This MOVES a boundary "
             "and does not warp content: warping onto the beat is refuted four "
             "times and one monotone time map settles a whole joint chain at "
             "once. 0 is the published behaviour")
    parser.add_argument(
        "--draft-feet-beat-lead", action="store_true",
        help="among the candidates already TIED on duration, keep the quarter "
             "whose FEET decelerate on this slot's beats furthest above the "
             "whole body's average.  Only the stable half of the part target is "
             "used: recomputed with the judge's own aligned_profile over 486 "
             "train and 481 test windows, the corpus agrees at Spearman +0.94 "
             "that the feet lead by 0.09-0.10 against <=0.05 for every other "
             "part, while the ORDER below the feet moves with the measurement. "
             "Unlike --draft-beat-fit (feet minus shoulders, which selected for "
             "braking hard and took jitter 0.0531 -> 0.0996), a candidate that "
             "brakes everything equally scores zero here")
    parser.add_argument(
        "--draft-join-lever-weights", action="store_true",
        help="weight the seam join cost by each joint's LEVER ARM instead of "
             "treating a pelvis degree and a wrist degree as equal.  The join "
             "ranking exists to minimise the seam, and measured over 150 random "
             "butt-joined library pairs the equal-weight cost correlates with "
             "the actual world-space speed jump at Spearman rho +0.372 while "
             "this weighting reaches +0.973 -- it was ranking nearly blind.  "
             "The spike is proximal in cause and distal in appearance: at a bar "
             "line the draft's hands peak at 25.7x their own median and its "
             "elbows at 19.1x, while its root peaks at 5.7x")
    parser.add_argument(
        "--draft-beat-fit", action="store_true",
        help="among the candidates already TIED on duration, keep the quarter "
             "whose FEET decelerate on this slot's beats more than their "
             "SHOULDERS do -- the dancer's asymmetry (ground truth settles feet "
             "0.0943 and shoulders 0.0207; every generated arm inverts it).  "
             "Judged in the same terms as the judgement column: world joints, "
             "at the slot's own beat phases, on the candidate AFTER it is "
             "stretched into the slot.  Two cheaper proxies were tried and both "
             "failed validation -- rot6d rest depth is unrelated to the "
             "property (rho +0.029), and world-joint rest depth is correctly "
             "specified but has no material (93.2%% of the library already "
             "qualifies, ground truth 93.4%%).  This one has both: the library "
             "is neutral on it (positive in 52.5%% of excerpts, sd 0.506) and "
             "the top quartile moves the mean margin -0.037 -> +0.560")
    parser.add_argument(
        "--draft-hold-by-music", action="store_true",
        dest="draft_hold_by_music",
        help="in bars whose music is sparse, keep the tied candidates that HOLD a "
             "shape; in busy bars, the ones that move. Ground truth does this: "
             "per-bar hold share against onset-peak density is negative on 16 "
             "of 20 eval clips, mean -0.225, P=0.0118, while the shipped arm "
             "reads -0.094 (P=0.82). Reads only the music. NOTE the hold and "
             "hold-vs-music columns move BY CONSTRUCTION under this flag; judge "
             "it on the video and the columns it does not select on.")
    parser.add_argument(
        "--draft-beat-span", type=float, default=0.0, metavar="BEATS",
        dest="draft_beat_span",
        help="among the candidates already tied on duration, keep only those "
             "that are the same NUMBER OF BEATS as the slot, within this "
             "tolerance. Retrieval selects on frame count alone, so a 2.4 s "
             "prototype that is six beats of a 150 BPM song and one that is "
             "four beats of a 100 BPM song are equally eligible for a four-beat "
             "slot. Measured on the 200 retrieved bars of the twenty eval "
             "clips: |mismatch| median 0.262 beats, p75 0.869, p90 1.211, and "
             "40.0%% of bars import a prototype that is a different WHOLE "
             "NUMBER of beats than its slot. Reads the query's beat grid and "
             "the candidate recording's own; neither is the target's motion. "
             "0 is off.")

    parser.add_argument(
        "--draft-feet-lead", action="store_true",
        help="among the candidates already TIED on duration, keep only those "
             "whose feet come to rest deeper than their hips and knees.  "
             "Ground truth has that asymmetry on 20 of 20 eval clips; the "
             "library has it on 28.1%% of windows, so the material exists and "
             "the duration rule -- which never looks -- lands on it at the "
             "corpus base rate.  With 73.6%% of slots tied and a median tie of "
             "6, the filter lifts that to about 86%% and costs nothing on "
             "duration, every tied candidate being exactly as good on it.  A "
             "filter and not a re-ranking: --retrieval-rule phase re-ranked the "
             "whole pool, landed its criterion perfectly and made the output "
             "worse by squeezing out variety")
    parser.add_argument(
        "--draft-rhythm-weight", type=float, default=0.0, metavar="W",
        help="prefer prototypes whose OWN bar had a rhythm like the one now "
             "playing, inside the same seam-aware candidate band. Both sides "
             "are channel 33, the onset-peak one-hot, binned over the bar and "
             "normalised, so the comparison is WHERE the onsets fall. "
             "EVERYTHING ELSE IS ALREADY RULED OUT: warping the content is "
             "refuted twice (--draft-beat-anchor onto beats, "
             "--draft-music-anchor onto beat + the train split's 0.200 lag, the "
             "latter monotonically worse in stretch) because a prototype's "
             "internal timing is already musical; and there is no start-phase "
             "left to fix -- the shipped arm's chosen prototypes sit at a "
             "median phase error of 0.000, since --plan-bar-grid and the beat-4 "
             "segmentation make every prototype begin on a beat. Beat-LEVEL "
             "alignment is therefore already correct and what remains is the "
             "pattern BETWEEN beats, which this is the only lever for that does "
             "not touch the content. Music on both sides, no motion read. Needs "
             "--draft-seam-aware-retrieval, the branch that ranks")
    parser.add_argument(
        "--draft-phase-weight", type=float, default=0.0, metavar="W",
        help="prefer prototypes whose OWN start phase matches the slot's, "
             "inside the same seam-aware candidate band. A prototype is four "
             "beats of its own song; played from a slot that starts on a beat "
             "its internal accents land on beats only if it too started on one, "
             "and measured over 361 plan segments the duration rule's picks sit "
             "at a median phase error of 0.250 -- exactly the uniform-random "
             "expectation. BUYS THE TIMING BY CHOOSING, NOT BY STRETCHING: two "
             "warps are refuted, --draft-beat-anchor (settle onto beats) and "
             "--draft-music-anchor (onto beat + the train split's 0.200 lag), "
             "the latter monotonically in stretch -- beat-phase gain +0.0145 "
             "shipped, +0.0064 at 1.2, -0.0132 at 1.6 against ground truth's "
             "+0.0314 -- because a prototype's internal timing is already "
             "musical and warping it destroys real structure. Added as a TERM "
             "to the ranking so the top-k draw survives; --retrieval-rule phase "
             "failed by hard-sorting the band to one answer. Needs "
             "--draft-seam-aware-retrieval, the branch that ranks")
    parser.add_argument(
        "--draft-music-anchor", type=float, default=0.0, metavar="MAX_STRETCH",
        help="warp each retrieved prototype so its OWN settle points land on "
             "the song's ONSET PEAKS inside that bar, capped at MAX_STRETCH per "
             "segment. This is the one thing in the pipeline that can make the "
             "timing depend on which song is playing: the planner picks a label "
             "(what to dance), --plan-bar-grid puts the bar lines on beats "
             "(where the boundaries are), and the uniform resample leaves the "
             "accents wherever the stretch dropped them -- the same relative "
             "positions whatever the music. Measured: ground truth's motion is "
             "locked to its own song's grid (+0.034 against other songs' grids, "
             "14/19, P=0.03) and no arm of ours is; the loss is in the DRAFT, "
             "not the completion; and posture already matches ground truth "
             "(crouch 0.908/0.900, wrist-above-shoulder 42.6%%/45.7%%), so what "
             "is missing is WHEN and not WHAT. Targets channel 33, the onset "
             "peaks (17.2%% of frames), NOT channel 34's beats -- "
             "--draft-beat-anchor targeted beats and made every reading worse, "
             "because our settle already sits just before the beat and pulling "
             "it onto beats pushes early points earlier. Reads only the query's "
             "AUDIO; the target's motion is never touched")
    parser.add_argument(
        "--draft-music-anchor-lag", type=float, default=0.20, metavar="BEATS",
        help="how far after the beat the settle should land. Measured on the "
             "RELEASE'S TRAIN SPLIT over 2,782 beat-to-settle pairs: median "
             "0.200 of a beat, p25 0.071, p75 0.385. This is the number "
             "--draft-beat-anchor was missing -- it aimed at beat + 0 while a "
             "real dancer settles at beat + 0.20, so it pulled already-early "
             "settle points further forward and every strength of it read worse")
    parser.add_argument(
        "--draft-music-anchor-shuffle", type=int, default=0, metavar="SEED",
        help="CONTROL for --draft-music-anchor: keep the number and range of "
             "the accent times and destroy their correspondence with the music "
             "by drawing them at random. A monotone warp also SMOOTHS, so an "
             "arm that improves could have bought the smoothing rather than the "
             "alignment; if this reads the same, nothing was aligned. Never a "
             "shipping setting")
    parser.add_argument(
        "--draft-beat-anchor-per-limb", choices=("all", "legs", "feet", "core"),
        default=None,
        help="give EACH LIMB its own settle points and its own warp, instead of "
             "one warp for the whole body. Requires --draft-beat-anchor. Why: "
             "the whole-body anchor lands the phase and pays for it by braking "
             "everything on one frame -- measured on the 20 eval clips, settle "
             "-0.0942 -> +0.1268 (ground truth +0.0594) but the per-part settle "
             "spread collapses 0.578 -> 0.222 and limb lockstep rises 0.500 -> "
             "0.617 against ground truth's 0.417, which is the operator's "
             "'\u7528\u4e0d\u540c\u80a2\u4f53\u90e8\u4f4d\u6362\u7740\u5361\u70b9' "
             "going the wrong way. --draft-seam-stagger fixes that at the SEAM "
             "only (spread 0.697, lag0 0.492) and reaches nothing between seams. "
             "rot6d is per-joint local rotation, so a limb on its own timeline "
             "is a limb dancing to its own accent, not a broken skeleton. "
             "MEASURED 2026-09-11 and it does NOT work as 'all': every group is "
             "warped onto the SAME targets (the beats), so giving each its own "
             "source accents still pulls them all onto one instant -- settle "
             "+0.4348 and spread 0.094, no better than the whole-body warp's "
             "0.073. 'legs' anchors only the two legs, because ground truth's "
             "own per-part settle is feet 0.0943 against hands 0.0233 and that "
             "unevenness IS the spread; 'core' anchors only root/spine/contacts")
    parser.add_argument(
        "--draft-join-top-k", type=int, default=None, metavar="K",
        help="how many of the best-joining candidates --draft-seam-aware-"
             "retrieval draws from (default 4). The constant's own note said "
             "'Not swept', and the sweep it never got turned out to matter: "
             "ranking by join cost and drawing from only the best few selects "
             "against BIG movements, because a prototype starting with the "
             "arms overhead is far from a tail whose arms are down. Measured "
             "on the 20 eval drafts, disabling seam-aware retrieval entirely "
             "lifts hands-above-head 15.8%% -> 18.5%% and wrist-above-shoulder "
             "p90 0.178 -> 0.193, against ground truth 23.5%% / 0.250 and the "
             "SOURCE CORPUS's own 22.5%% / 0.229 -- the material is there and "
             "the ranking is what suppresses it. This is the operator's "
             "'不够舒展到位'. Widening keeps the join ranking without "
             "concentrating the draw on the smallest motion; the cost to watch "
             "is seam jerk, which is why seam-aware exists at all")
    parser.add_argument(
        "--draft-join-pose-weight", type=float, default=1.0, metavar="W",
        help="how much --draft-seam-aware-retrieval's ranking counts the "
             "ABSOLUTE pose gap at a join, against the velocity match "
             "(default 1.0; 0.0 = velocity only). The pose term is a raw "
             "rot6d distance, so a candidate starting with the arms overhead "
             "is scored as a bad continuation for being a bigger shape, not "
             "for being discontinuous -- a plausible source of the measured "
             "reach deficit (wrist-above-shoulder p90 0.178 against ground "
             "truth 0.250 and the source corpus's own 0.229). Since "
             "--completion-inpaint-seam-width regenerates the seam, the "
             "draft's job there is the direction of travel. The cost to "
             "watch is seam judder (tools/measure_seam_judder.py)")
    parser.add_argument(
        "--draft-root-seam-smooth", type=int, default=0, metavar="FRAMES",
        dest="draft_root_seam_smooth",
        help="keep the body moving through every bar seam. Two things braked it: "
             "root continuity put each segment's first frame ON the previous "
             "segment's last (a zero step), and the seam blend pulled the chained "
             "root toward two frozen anchor frames. Measured 2026-09-16 on the "
             "ten vis clips: root speed at a seam over speed 13-14 frames away "
             "was 0.24 in fix2 against ground truth's 1.19, lower on 10/10 clips, "
             "P=0.002. After every prototype is chosen, this carries the root one "
             "velocity step across each seam, eases the next segment's velocity in "
             "over FRAMES, and keeps the chained root columns out of the blend. "
             "Done after retrieval so the chosen moves are untouched. Needs "
             "--draft-root-continuity xy|xyz. 0 is off.")
    parser.add_argument(
        "--draft-root-velocity-blend", type=int, default=0, metavar="FRAMES",
        help="ease the root's per-frame travel from the previous segment's into "
             "the new one's over this many frames. --draft-root-continuity "
             "aligns the root POSITION at a join and leaves the body's speed "
             "and direction changing in one frame: measured 2026-09-06 with "
             "xy on, the worst single-frame root step sits AT a join on 18 of "
             "18 drafts (median distance 1 frame) at 0.13-0.37 m against ground "
             "truth's 0.022. 0 is the published behaviour")
    parser.add_argument(
        "--face-camera", type=float, default=None, metavar="STRENGTH",
        dest="face_camera_strength",
        help="turn the finished dance back toward the render camera. The "
             "operator, 2026-09-08: '画面的确有长时间背对或者侧对的情况'. Share of "
             "frames NOT facing the lens (profile counts, which is half of what "
             "was named): ground truth 12.3%%, the retrained rhythm arm 29.9%%, "
             "and its MEDIAN longest continuous off-camera span 1.5 s against "
             "ground truth's 0.3 s. Measured before choosing where to fix it: "
             "the DRAFT is already 32.3%% off camera and the completion brings "
             "it to 29.9%%, so the completion is not the source, and "
             "--draft-facing-anchor cannot fix it either because it pulls "
             "toward the clip's own OPENING yaw while 30%% of draft openings "
             "are themselves off camera (ground truth 10%%). Two parts: the "
             "clip's mean facing is put on the camera by one rotation about the "
             "world ORIGIN, which is rigid and costs nothing (29.9%% -> 9.2%% "
             "with foot skate unmoved at 1.142); then STRENGTH times the "
             "low-passed residual is removed about the PER-FRAME ROOT so the "
             "body turns in place. Turns shorter than --face-camera-window pass "
             "through, so total turning stays 1054 deg against ground truth's "
             "919. STRENGTH is not free -- skate 1.142 -> 1.147 at 0.15, 1.185 "
             "at 0.35 -- and it is NOT meant to reach zero: ground truth is "
             "12.3%% and a dancer who never turns is its own defect. Omit to "
             "leave the facing alone, which is what every earlier artifact did")
    parser.add_argument(
        "--face-camera-window", type=float, default=FACING_SMOOTH_SECONDS,
        metavar="SECONDS",
        help="how slow a facing deviation has to be before --face-camera pulls "
             "it back. Turns shorter than this are left alone, which is the "
             "whole reason the dance keeps turning; default {}"
             .format(FACING_SMOOTH_SECONDS))
    parser.add_argument(
        "--fix-foot-skate", type=float, default=0.0, metavar="STRENGTH",
        dest="fix_skate",
        help="stop a planted foot sliding by translating the BODY, at this "
             "strength. The operator, 2026-09-12: '脚滑也交给后处理来修'. "
             "Measured skate (mean horizontal foot speed while the foot is at "
             "its own ground level): ground truth 0.295 m/s, shipped 0.420, "
             "and the arms that buy reach are worse -- 0.471 with "
             "--draft-whole-units, 0.556 with the seam-aware ranking off. "
             "Ground truth is the target, not zero: a dancer whose feet never "
             "move is its own defect. The planted signal is the model's own "
             "four CONTACT channels, deliberately NOT the scoring tool's 'at "
             "its own ground level' test, so the score stays an independent "
             "check. Only a per-frame translation is applied -- smpl_poses is "
             "untouched, so every joint angle and the dance itself are "
             "bit-identical and only WHERE the body is changes. 0 reproduces "
             "the published behaviour exactly")
    parser.add_argument(
        "--fix-skate-seam-mask", type=int, default=0, metavar="FRAMES",
        dest="fix_skate_seam_blend",
        help="treat the feet as NOT planted within this many frames of a bar "
             "seam, so --fix-foot-skate stops charging the seam blend's pose "
             "morph to the root. Measured 2026-09-16 on the fixed ten by "
             "recovering the strength-0 pelvis path exactly from two arms that "
             "differ only in strength: 4.94 m at strength 0 against ground "
             "truth's 4.85 (5/10, flat) and 6.36 at strength 1.0 (+1.42, "
             "10/10, P=0.002), with 74.4%% of the correction's own path inside "
             "+-8 frames of a seam -- 28.4%% of frames, 2.6x over-represented. "
             "Set it to --draft-seam-blend, which is the span the ramp covers. "
             "0 keeps the old behaviour so existing arms reproduce.")
    parser.add_argument(
        "--floor-anchor", type=float, default=None, metavar="METRES",
        help="stand every generated clip on a floor at this height, by one "
             "rigid shift in z applied after decoding. The value must come "
             "from the TRAINING corpus, not from the clip's own ground truth. "
             "Off by default, which is not neutral: measured 2026-09-07 over "
             "the 20 eval clips, ground truth's floor (5th percentile of the "
             "per-frame lowest foot) sits at 0.342 m with a p5-p95 spread of "
             "0.047 m, while the generated floor spreads 0.454 m and "
             "correlates with ground truth's at r=-0.021 -- the standing "
             "height is essentially random. render_avatar_video.py puts every "
             "arm on the REFERENCE's floor on purpose, so that shows up as a "
             "body hovering over the tiles (0.196 m on 7610414564962183545) or "
             "sunk through them (-0.243 m on 7287585049111711032). The shift "
             "is rigid, so every root-relative motion, velocity and contact is "
             "bit-identical; only where the body stands changes. See "
             "anchor_floor for why a per-frame correction would be wrong.")
    parser.add_argument(
        "--draft-facing-anchor", type=float, default=0.0, metavar="FRACTION",
        help="pull this fraction of the accumulated facing deviation back "
             "towards the clip's opening direction at every join, so the dancer "
             "keeps performing to where it started. --draft-facing-continuity "
             "keeps the facing SMOOTH across a join and lets each prototype's "
             "own turn accumulate: measured 2026-09-06, ground truth spends "
             "0.0%% of frames more than 90 deg from its opening facing on 14 of "
             "20 eval clips, while three generated clips spend 77-92%% facing "
             "away, the worst for 14.77 s with a net turn of 593 deg. 0 is the "
             "published behaviour; needs --draft-facing-continuity")
    parser.add_argument(
        "--index-filler", action="store_true",
        help="let the transition class (label 0) be RETRIEVED like any other, "
             "instead of being bridged by --draft-gap-fill. Filler is 30%% of "
             "the plan's frames; the straight line gap-fill draws there reads "
             "0.015 m/s against ground truth's 0.628 m/s on the SAME frames, "
             "and measured against ground truth at the same moment in the same "
             "song the draft has one-second windows at literally zero speed "
             "with 24.7%% below 0.6x. 'Filler' names a span the vocabulary did "
             "not classify, not a span where the dancer stopped. Off by "
             "default: it changes which classes the library holds, and the "
             "source-safety and coverage gates are computed from that")
    parser.add_argument(
        "--derive-retrieval-group", action="store_true",
        help="when the sidecar registry has no entry for a query, derive its "
             "retrieval group from the clip name's own recording prefix instead "
             "of refusing to retrieve anything. Two of the twenty T eval clips "
             "have no entry and are therefore generated from MUSIC ALONE "
             "(safe_draft_condition_fraction 0.0). Checked before offering "
             "this: 0 of the registry's 6,823 entries map to anything but that "
             "prefix, no recording spans two groups, and none of the 20 eval "
             "recordings is among the library's 143 groups, so the exclusion is "
             "a no-op either way. Off by default -- the fail-closed rule is what "
             "keeps a query's own motion out of its draft")
    parser.add_argument(
        "--draft-gap-fill",
        choices=("zero", "hold", "interpolate"),
        default="zero",
        help="what the draft holds on transition frames; zero is the published "
             "behaviour and is a pose, not an absence",
    )
    parser.add_argument("--plan-stride", type=int, default=None)
    parser.add_argument("--plan-fusion", choices=("centre", "vote", "taper"),
                        default="centre")
    parser.add_argument(
        "--plan-vote-tie-break", choices=PLAN_VOTE_TIE_BREAKS, default="centre",
        help="how --plan-fusion vote resolves a tied count; 'index' is the "
             "pre-2026-08-23 behaviour, which handed every tie to transition "
             "because label 0 sorts first (see _fuse_windows)")
    parser.add_argument(
        "--plan-transition-policy", choices=TRANSITION_POLICIES, default="protect",
        help="whether the minimum-duration merge may delete a transition or "
             "absorb an atomic movement into one; reproducing a pre-2026-08-23 "
             "artifact needs 'merge' AND --plan-merge-order first AND "
             "--plan-vote-tie-break index -- 'merge' alone is not it")
    parser.add_argument(
        "--plan-merge-order", choices=MERGE_ORDERS, default="shortest",
        help="which offending segment the minimum-duration merge resolves "
             "first; 'first' is the pre-2026-08-23 inference behaviour, "
             "'shortest' is what tools/postprocess_atomic_plan.py has always "
             "done (they differ on ~1.4%% of frames)")
    parser.add_argument(
        "--planner-guidance-weight", type=float, default=1.0,
        help="classifier-free guidance on the planner's x0 logits; needs a "
             "planner trained with --planner-cond-drop-prob > 0 and is refused "
             "otherwise.  1.0 is plain conditional sampling, which is what "
             "every artifact before 2026-08-23 used")
    parser.add_argument(
        "--planner-transition-logit-bias", type=float, default=0.0,
        help="added to the transition class's x0 logit at every reverse step.  "
             "A calibration of how much vocabulary the plan admits, not a "
             "quality change; fit it on val, never on test.  Unlike temperature "
             "it has leverage: T 1.0->0.2 moved the share 0.5263->0.5178 while "
             "a bias of -2.0 moved it 0.5392->0.3553")
    parser.add_argument(
        "--plan-bar-grid", action="store_true",
        help="replace the plan's segmentation with M1's own music bar grid "
             "(channel 34, every 4 beats at a per-clip phase), one class per "
             "bar.  97.3%% of ground-truth boundaries are on that grid against "
             "30.7%% of the planner's own")
    parser.add_argument(
        "--plan-music-repeat", type=float, default=0.0, metavar="STRENGTH",
        help="0..1; tie bars whose MUSIC sounds alike to one class, so the "
             "plan comes back to a movement where the song comes back.  Bars "
             "are grouped by their own music (timbre+chroma cosine, K = bars * "
             "(1 - STRENGTH)) and each group takes the class the PLANNER "
             "already spent the most frames on, so no label is invented and no "
             "ground truth is read.  Ground truth does this (+0.2595 against a "
             "+0.1174 circular-shift null, dz +3.17, 10/10 clips); our plans do "
             "not (aligned dz +0.94, shipped 21-class dz -2.45).  0 disables")
    parser.add_argument(
        "--plan-bar-tokens", action="store_true",
        help="plan ONE LABEL PER BAR instead of per frame: the query's music is "
             "pooled over the same 4-beat bars --plan-bar-grid snaps to and the "
             "planner runs over those tokens, whose labels are then expanded "
             "back across their frames.  Needs a planner trained on bar tokens "
             "(planner_token_resolution=bar in its checkpoint args) and refuses "
             "a frame-trained one.  Incompatible with --plan-stride, "
             "--plan-fusion and any --plan-vote-window/--plan-min-segment other "
             "than 1: those are frame operations and a bar is 37-89 frames")
    parser.add_argument(
        "--plan-vote-window", type=int, default=5,
        help="sliding-window width of the plan's majority vote, in frames "
             "(5 = 0.17 s at 30 fps)")
    parser.add_argument(
        "--plan-min-segment", type=int, default=6, dest="plan_min_segment_length",
        help="atomic plan segments shorter than this are merged away "
             "(6 = 0.2 s at 30 fps)")
    parser.add_argument("--completion-stride", type=int, default=75)
    parser.add_argument(
        "--retrieval-energy-floor",
        type=float,
        default=None,
        help="drop each atomic class's least-energetic tail before the duration rule "
             "picks, as a within-class quantile (0.25 = drop the bottom quarter). "
             "Aimed at a measured defect: 31 of 1,277 prototype pastes on the shipped "
             "arm are under 0.20 m/s. Not an absolute floor -- 27 of those 31 come "
             "from classes that are low-energy by nature, and an absolute floor would "
             "empty them.",
    )
    parser.add_argument(
        "--draft-guidance-weight",
        type=float,
        default=None,
        help="amplify the PLAN the way --guidance-weight amplifies the music. Needs a "
             "checkpoint trained with --draft-drop-prob; 1.0 reproduces the old "
             "behaviour exactly. Costs a third forward per diffusion step.",
    )
    parser.add_argument(
        "--dump-draft-dir",
        default=None,
        help="also write the retrieval draft (the plan turned into motion, before the "
             "completion model touches it) into this directory, in the same format as the "
             "generated result so the same scorers read both",
    )
    parser.add_argument(
        "--completion-blend-width",
        type=int,
        default=None,
        help="frames averaged at each window junction (default: the whole overlap, "
             "which at stride 75 / seq_len 150 makes 79.75%% of output frames a convex "
             "combination of two independently sampled diffusion draws)",
    )
    parser.add_argument("--guidance-weight", type=float)
    parser.add_argument("--draft-noise-ratio", type=float)
    parser.add_argument("--inference-batch-size", type=int, default=4)
    parser.add_argument("--sequence-list")
    parser.add_argument(
        "--plan-source",
        choices=("planner", "ground-truth"),
        default="planner",
        help="planner is self-driven; ground-truth is ORACLE-only and not headline-eligible",
    )
    parser.add_argument("--ground-truth-labels", action="store_true")
    parser.add_argument(
        "--plan-from", default=None, metavar="DIR", dest="plan_from",
        help="DIAGNOSTIC: replay the plan (atomic_labels and the plan report, "
             "incl. bar_grid_phase) of an EARLIER RUN of ours in DIR instead of "
             "the planner's draw.  --stochastic-planner draws a different plan "
             "under GPU load (DEFECTS 90.8: 20%% of labels, 60%% of units), so "
             "arms run in parallel are not comparable without it.  Reads only "
             "our own generated plan, never a target motion; the planner still "
             "runs first so the per-clip RNG is consumed exactly as before.")
    parser.add_argument(
        "--retrieval-tie-break", default="index", choices=["index", "salted"],
        help="what the duration rule does when several candidates match the "
             "target length equally well, which is the usual case rather than "
             "the exception: 73.6%% of retrieved segments have a tie and the "
             "median tie holds six candidates.  'index' takes the first, which "
             "is what every earlier artifact did and is therefore the default, "
             "and which hands the same exemplar to every clip that asks for "
             "that (label, length).  'salted' picks among the TIED candidates "
             "only -- all exactly as good on duration, so the resampling "
             "factor is unchanged -- using a hash of the query's own retrieval "
             "group, so two recordings get different exemplars while one "
             "recording stays consistent with itself")
    parser.add_argument(
        "--retrieval-rule", default="duration",
        choices=["duration", "medoid", "random", "tempo", "phase", "learned"],
        help="how a prototype is chosen among a class's candidates.  'duration' "
             "picks the nearest length and never looks at the motion -- the "
             "shipped rule, and the default so every earlier artifact still "
             "reproduces.  'medoid' picks the class's most typical member, "
             "which collects 51%% of the in-class oracle's headroom on the wild "
             "release without needing anything the query does not have.  "
             "'learned' scores every candidate in the duration band against the "
             "segment it has to FOLLOW and samples from the top k; it needs "
             "--retrieval-selector and refuses without one.")
    parser.add_argument(
        "--draft-only", action="store_true",
        help="DIAGNOSTIC: write the retrieval draft as the result and never run "
             "the completion model.  Splits 'the selection did not change the "
             "draft' from 'the completion did not pass the change through' -- "
             "section 14.3 measured the second happening, so the two cannot be "
             "told apart from the output alone.  Artifacts are still marked with "
             "their generation protocol; read them as drafts, not as dances.")
    parser.add_argument(
        "--retrieval-selector", default=None,
        help="checkpoint from tools/train_retrieval_selector.py.  Required by "
             "--retrieval-rule learned, which fails closed without it: a run "
             "that asked for the learned rule, silently got the shipped one and "
             "recorded 'learned' in its manifest would be compared as if it "
             "were two dances when it is one.")
    parser.add_argument(
        "--retrieval-max-yaw-step", type=float, default=None, metavar="DEGREES",
        help="refuse retrieval candidates whose own played range turns the body "
             "faster than this in one frame at 30 fps. The library carries the "
             "artefact: censused with the repo's own decode, 985 of 6553 T-line "
             "windows (15.0%%) exceed 30 deg/frame and 94 exceed 90, worst "
             "174.6 -- 5238 deg/s, which no body does -- while the ten held-out "
             "ground-truth clips never exceed 25.2. Needs --retrieval-yaw-steps")
    parser.add_argument(
        "--retrieval-max-speed-spike", type=float, default=None, metavar="RATIO",
        help="refuse candidates whose played range contains a frame moving more "
             "than this many times the LOCAL median speed. Ground truth sliced "
             "into 150-frame windows never reaches 4.0 (0 of 62, worst 3.12) "
             "while the library reaches 13.4 with 5.8%% of windows above 4. "
             "Needs --retrieval-speed-spikes")
    parser.add_argument(
        "--draft-join-frame", choices=IndexedAtomicMotionLibrary.JOIN_FRAMES,
        default="raw", dest="draft_join_frame",
        help="where the join cost looks at a candidate. 'raw' (default, every "
             "artifact before 2026-09-17) compares the library bytes as stored "
             "against a tail that has already been floor-levelled, turned and "
             "chained, so ground position (median gap 0.63 m), heading (28.6 "
             "deg) and floor (0.35 m) decide about a fifth of the join band. "
             "'placed' compares the candidate as build_draft will lay it down")
    parser.add_argument(
        "--draft-join-height-weight", type=float, default=0.0, metavar="W",
        dest="draft_join_height_weight",
        help="multiply the pelvis-height gap (metres) inside the join cost's pose "
             "term by W. fix5's pick is no closer in height than the pool median "
             "(41/90 slots), while the band already holds a candidate within "
             "0.42 cm (median); x30 brings the pick to 3.1 cm, lower on 10/10 "
             "clips. Needs --draft-join-frame placed")
    parser.add_argument(
        "--retrieval-selector-input-units", default="inference",
        choices=IndexedAtomicMotionLibrary.SELECTOR_INPUT_UNITS,
        dest="retrieval_selector_input_units",
        help="'trained' feeds the learned selector the units its checkpoint was "
             "fitted in (the trainer unnormalized already-raw motion a second "
             "time), with the candidate's floor removed so the height step is "
             "floor-relative on both sides")
    parser.add_argument(
        "--draft-hop-guard", action="store_true", dest="draft_hop_guard",
        help="after the beat-span filter, drop candidates whose played range "
             "leaves the floor (lowest foot >15 cm over its 2 s floor). fix5 is "
             "airborne on 3.45%% of frames, ground truth 0.57%%, 15/17 clips, and "
             "the draft already carries it. Needs --retrieval-vertical-flags")
    parser.add_argument(
        "--draft-music-energy", action="store_true", dest="draft_music_energy",
        help="after the beat-span filter, cap movement size by bar loudness "
             "(MFCC c0 against the library median): quiet bars keep candidates "
             "at or below the pool's median energy, loud bars drop the top "
             "fifth. Needs --retrieval-vertical-flags")
    parser.add_argument(
        "--draft-unit-floor", default=None, metavar="NPZ", dest="draft_unit_floor",
        help="level each retrieved unit by the median LOCAL floor over the frames "
             "it plays (tools/census_release_vertical.py --floor-out) instead of "
             "its recording's one floor. On fix5's drafts the local offset "
             "predicts a unit's float at r=+0.973 and removing it takes units "
             "more than 10 cm off their neighbours from 10 to 0")
    parser.add_argument(
        "--retrieval-vertical-flags", default=None, metavar="NPZ",
        dest="retrieval_vertical_flags",
        help="per-window per-frame hop/deep-squat bits from "
             "tools/census_release_vertical.py")
    parser.add_argument(
        "--retrieval-speed-spikes", default=None, metavar="NPZ",
        help="per-window per-frame relative speed from "
             "tools/census_release_yaw_steps.py --steps-out (the _speed.npz)")
    parser.add_argument(
        "--retrieval-yaw-steps", default=None, metavar="NPZ",
        help="per-window per-frame yaw steps from "
             "tools/census_release_yaw_steps.py --steps-out")
    parser.add_argument(
        "--retrieval-selector-join-band", type=float, default=0.0, metavar="FRACTION",
        help="before the selector scores, keep only this fraction of the "
             "duration band -- the fraction that JOINS BEST.  The seam-aware "
             "ranking is documented as selecting against big movements, and "
             "--retrieval-rule learned never received it: measured 2026-09-13 "
             "the selector's picks move 25%% faster than ground truth (0.0306 "
             "against 0.0245 m/frame, baseline 0.0257), energy 1.279 against a "
             "[0.85,1.15] band, foot skate 0.524 against 0.295 -- and "
             "tightening the draw to top-k 2 / T 0.3 leaves the speed unchanged "
             "at 0.0306, so it is the selector's preference and not the "
             "sampling.  The selector still ranks INSIDE the band; it simply "
             "cannot reach the worst-joining candidates. 0 keeps the published "
             "behaviour")
    parser.add_argument(
        "--retrieval-selector-top-k", type=int, default=8,
        help="how many of the scored candidates the draw may come from.  The "
             "output is a temperature-weighted SAMPLE over these, never an "
             "argmax: --retrieval-rule phase won its own criterion and lost on "
             "the output because a hard ranking collapsed the candidate pool "
             "(docs/DANCE_QUALITY_DEFECTS.md section 15.8).")
    parser.add_argument(
        "--retrieval-selector-temperature", type=float, default=1.0,
        help="softmax temperature over the top k.  0 makes it an argmax, which "
             "is the failure mode above and is therefore not the default.")
    parser.add_argument(
        "--no-plan-conditioning", action="store_true",
        help="DIAGNOSTIC: zero the draft and mask, so the completion model sees "
             "music alone.  This is the control the two-stage design has never "
             "been measured against -- every comparison so far has been between "
             "two ways of planning, none against not planning.  Artifacts are "
             "marked headline_eligible: false.")
    parser.add_argument("--unsourced-retrieval", action="store_true",
                        help="declare that the query has no source recording in this "
                             "corpus, so prototype retrieval has nothing to exclude.  "
                             "True for AIST M6, where a query is named for a held-out "
                             "*song* under a song-disjoint split; without it the "
                             "fail-closed branch fires and the retrieval stage -- the "
                             "paper's M5 -- contributes nothing at all.  A declaration, "
                             "not an inference: only the caller knows which kind of "
                             "name it passed, and the choice rides into the artifact")
    return parser.parse_args()


def main(options):
    sequence_names = None
    if options.sequence_list:
        sequence_names = [
            line.strip()
            for line in Path(options.sequence_list).read_text().splitlines()
            if line.strip()
        ]
    manifest = infer_directory(
        audio_dir=options.audio_dir,
        music_span_check=options.music_span_check,
        ingest_root=options.ingest_root,
        output_dir=options.output_dir,
        planner_checkpoint=options.planner_checkpoint,
        completion_checkpoint=options.completion_checkpoint,
        data_root=options.data_root,
        target_motion_dir=options.target_motion_dir,
        device=options.device,
        seed=options.seed,
        max_samples=options.max_samples,
        max_frames=options.max_frames,
        overwrite=options.overwrite,
        deterministic_planner=options.deterministic_planner,
        temperature=options.temperature,
        completion_stride=options.completion_stride,
        completion_blend_width=options.completion_blend_width,
        completion_draft_guidance_weight=options.draft_guidance_weight,
        retrieval_energy_floor=options.retrieval_energy_floor,
        draft_dump_dir=options.dump_draft_dir,
        plan_stride=options.plan_stride,
        plan_fusion=options.plan_fusion,
        plan_vote_tie_break=options.plan_vote_tie_break,
        plan_transition_policy=options.plan_transition_policy,
        plan_merge_order=options.plan_merge_order,
        planner_guidance_weight=options.planner_guidance_weight,
        planner_transition_logit_bias=options.planner_transition_logit_bias,
        plan_bar_grid=options.plan_bar_grid,
        plan_music_repeat=options.plan_music_repeat,
        plan_bar_beats=options.plan_bar_beats,
        plan_bar_tokens=options.plan_bar_tokens,
        plan_vote_window=options.plan_vote_window,
        plan_min_segment_length=options.plan_min_segment_length,
        unsourced_retrieval=options.unsourced_retrieval,
        guidance_weight=options.guidance_weight,
        draft_noise_ratio=options.draft_noise_ratio,
        inference_batch_size=options.inference_batch_size,
        sequence_names=sequence_names,
        ground_truth_labels=(
            options.ground_truth_labels or options.plan_source == "ground-truth"
        ),
        plan_from=options.plan_from,
        draft_root_continuity=options.draft_root_continuity,
        draft_gap_fill=options.draft_gap_fill,
        draft_seam_blend=options.draft_seam_blend,
        draft_seam_window=options.draft_seam_window,
        seam_transition=options.seam_transition,
        draft_seam_stagger=options.draft_seam_stagger,
        draft_seam_aware_retrieval=options.draft_seam_aware_retrieval,
        draft_beat_anchor=options.draft_beat_anchor,
        draft_beat_anchor_per_limb=options.draft_beat_anchor_per_limb,
        draft_music_anchor=options.draft_music_anchor,
        draft_music_anchor_lag=options.draft_music_anchor_lag,
        draft_music_anchor_shuffle=options.draft_music_anchor_shuffle,
        draft_lower_body_delay=options.draft_lower_body_delay,
        draft_seam_lead=options.draft_seam_lead,
        draft_continue_phrase=options.draft_continue_phrase,
        draft_motif_at_phrase=options.draft_motif_at_phrase,
        draft_continue_no_replay=options.draft_continue_no_replay,
        draft_continue_phrase_novelty=options.draft_continue_phrase_novelty,
        draft_phrase_rhythm_keep=options.draft_phrase_rhythm_keep,
        draft_tempo_keep=options.draft_tempo_keep,
        draft_phrase_chain=options.draft_phrase_chain,
        draft_step_lock_keep=options.draft_step_lock_keep,
        draft_energy_follow=options.draft_energy_follow,
        completion_beat_keep=options.completion_beat_keep,
        completion_beat_stride=options.completion_beat_stride,
        completion_beat_keep_holds=options.completion_beat_keep_holds,
        completion_beat_free_max=options.completion_beat_free_max,
        completion_beat_hold_frac=options.completion_beat_hold_frac,
        draft_join_top_k=options.draft_join_top_k,
        draft_join_pose_weight=options.draft_join_pose_weight,
        draft_whole_units=options.draft_whole_units,
        draft_bar_units=options.draft_bar_units,
        draft_continue_source=options.draft_continue_source,
        draft_continue_lookahead=options.draft_continue_lookahead,
        draft_continue_any_label=options.draft_continue_any_label,
        draft_continue_max_run=options.draft_continue_max_run,
        draft_prefer_full=options.draft_prefer_full,
        draft_prefer_full_mode=options.draft_prefer_full_mode,
        draft_rhythm_scorer=options.draft_rhythm_scorer,
        draft_rhythm_keep=options.draft_rhythm_keep,
        draft_rhythm_shift=options.draft_rhythm_shift,
        draft_continue_stop=options.draft_continue_stop,
        draft_continue_rhythm_min=options.draft_continue_rhythm_min,
        draft_continue_settle=options.draft_continue_settle,
        draft_motif_return=options.draft_motif_return,
        draft_motif_lags=options.draft_motif_lags,
        draft_motif_max=options.draft_motif_max,
        draft_motif_pick=options.draft_motif_pick,
        draft_continuity_scorer=options.draft_continuity_scorer,
        draft_continuity_keep=options.draft_continuity_keep,
        draft_continuity_weight=options.draft_continuity_weight,
        draft_phase_weight=options.draft_phase_weight,
        draft_rhythm_weight=options.draft_rhythm_weight,
        draft_feet_lead=options.draft_feet_lead,
        draft_beat_span=options.draft_beat_span,
        draft_hold_by_music=options.draft_hold_by_music,
        draft_beat_fit=options.draft_beat_fit,
        draft_join_lever_weights=options.draft_join_lever_weights,
        draft_feet_beat_lead=options.draft_feet_beat_lead,
        draft_quiet_cut=options.draft_quiet_cut,
        draft_selector_join_band=options.retrieval_selector_join_band,
        retrieval_max_yaw_step=options.retrieval_max_yaw_step,
        retrieval_yaw_steps=options.retrieval_yaw_steps,
        retrieval_max_speed_spike=options.retrieval_max_speed_spike,
        retrieval_speed_spikes=options.retrieval_speed_spikes,
        draft_join_frame=options.draft_join_frame,
        draft_join_height_weight=options.draft_join_height_weight,
        retrieval_selector_input_units=options.retrieval_selector_input_units,
        draft_hop_guard=options.draft_hop_guard,
        draft_music_energy=options.draft_music_energy,
        retrieval_vertical_flags=options.retrieval_vertical_flags,
        draft_unit_floor=options.draft_unit_floor,
        draft_floor_normalize=options.draft_floor_normalize,
        draft_root_velocity_blend=options.draft_root_velocity_blend,
        draft_root_seam_smooth=options.draft_root_seam_smooth,
        draft_facing_anchor=options.draft_facing_anchor,
        floor_anchor=options.floor_anchor,
        fix_skate=options.fix_skate,
        fix_skate_seam_blend=options.fix_skate_seam_blend,
        face_camera_strength=options.face_camera_strength,
        face_camera_window=options.face_camera_window,
        index_filler=options.index_filler,
        derive_retrieval_group=options.derive_retrieval_group,
        draft_bar_prototypes=options.draft_bar_prototypes,
        allow_unfillable_slots=options.allow_unfillable_slots,
        completion_start_step=options.completion_start_step,
        completion_reproject_every=options.completion_reproject_every,
        completion_inpaint_seam_width=options.completion_inpaint_seam_width,
        completion_keep_root=options.completion_keep_root,
        completion_sample_steps=options.completion_sample_steps,
        draft_recurrence_variety=options.draft_recurrence_variety,
        draft_facing_continuity=options.draft_facing_continuity,
        no_plan_conditioning=options.no_plan_conditioning,
        retrieval_rule=options.retrieval_rule,
        retrieval_tie_break=options.retrieval_tie_break,
        draft_only=options.draft_only,
        retrieval_selector=options.retrieval_selector,
        retrieval_selector_top_k=options.retrieval_selector_top_k,
        retrieval_selector_temperature=options.retrieval_selector_temperature,
        allow_cross_release_checkpoints=getattr(options, "allow_cross_release_checkpoints", False),
    )
    config = getattr(options, "generation_config", None)
    if config is not None:
        # Written beside, not into, the per-clip outputs: the pkl bytes stay
        # comparable across runs that name the same flags through different files.
        manifest["generation_config"] = config
        with open(str(Path(options.output_dir) / "manifest.json"), "w") as handle:
            json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    _options = parse_args()
    _options.generation_config = _generation_config(sys.argv[1:])
    main(_options)
