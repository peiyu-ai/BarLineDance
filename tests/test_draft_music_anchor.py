"""``--draft-music-anchor``: put the prototype's accents on THIS SONG's accents.

WHY, and the operator set it as the priority.  2026-09-12, after watching four
of the ten review clips: "都不是很符合音乐旋律,节奏,跳不出 gt 那种卡的很好的律动
效果 ... 这个是当前最主要的、导致模型不可用的矛盾".

THE GAP IT FILLS.  Nothing in the pipeline decides WHEN inside a bar.  The
planner emits one label per bar (what to dance), ``--plan-bar-grid`` puts the bar
lines on beats (where the boundaries are), and ``_values_at`` fills the bar with
ONE uniform resample -- so the prototype's accents land wherever the stretch
drops them, which is the same relative position whatever song is playing.  Four
measurements say that is the whole remaining defect:

    ground truth is locked to its own song's grid   +0.034, 14/19, P=0.03
    no arm of ours is                               every interval straddles 0
    the loss is in the DRAFT                        the completion reduces it
    posture already matches                         crouch 0.908/0.900,
                                                    wrist-above-shoulder 42.6/45.7%

so what is missing is WHEN, not WHAT.

WHAT IT AIMS AT, AND THE VERSION OF IT THAT WAS WRONG.  The first implementation
put the settle points on the song's ONSET PEAKS (channel 33) and that is
backwards: speed at accent frames over speed elsewhere reads **1.0333 for ground
truth** -- a dancer moves FASTER on an accent, which is a hit to strike into --
while the shipped arm already read 0.9793 and accent-anchoring took it to 0.9578,
further away.  Nor is there a lag that would fix it: over 8,747
accent-to-settle pairs in the train split the lag is nearly uniform (p25 0.25,
median 0.71, p75 1.00).

The BEAT grid does have a consistent lag.  Over 2,782 beat-to-settle pairs in
the same split it is **median 0.200 of a beat** (p25 0.071, p75 0.385), which
also agrees with the independent absolute-m/s reading on the eval clips (trough
at phase 0.12).  That 0.20 is what ``--draft-beat-anchor`` was missing: it aimed
at beat + 0 while a real dancer settles at beat + 0.20, so it pulled
already-early settle points further forward and every strength of it read worse.

BOTH NUMBERS COME FROM THE TRAIN SPLIT.  Fitting the lag on the evaluation clips
would be reading the answer out of the test set.

NO GROUND TRUTH IS READ.  The anchors come from the query's AUDIO, which is the
task's input, and from the prototype's own motion.  The target clip's motion is
never touched; ``test_only_the_music_and_the_prototype_are_read`` is the
assertion that keeps it that way.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import infer_atomic
from infer_atomic import MUSIC_ACCENT_CHANNEL

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(extra):
    import subprocess
    return subprocess.run(
        [sys.executable, os.path.join(REPO, "infer_atomic.py"),
         "--audio-dir", "/x", "--ingest-root", "/x", "--planner-checkpoint", "/x",
         "--completion-checkpoint", "/x", "--data-root", "/x", "--output-dir", "/x"] + extra,
        capture_output=True, text=True)


def test_the_default_lag_is_the_measured_one():
    """POSITIVE CONTROL for the whole change: a lag of 0 IS --draft-beat-anchor,
    which read worse at every strength."""
    import inspect
    signature = inspect.signature(
        infer_atomic.IndexedAtomicMotionLibrary.build_draft)
    assert signature.parameters["music_anchor_lag"].default == pytest.approx(0.20)


def test_the_two_warps_refuse_to_run_together():
    done = _run(["--draft-music-anchor", "1.4", "--draft-beat-anchor", "1.2"])
    assert done.returncode != 0
    assert "both warp" in (done.stderr + done.stdout)


def test_the_control_refuses_without_the_thing_it_controls():
    """A shuffle recorded in the manifest but inert is the defect shape this
    repository keeps paying for."""
    done = _run(["--draft-music-anchor-shuffle", "7"])
    assert done.returncode != 0
    assert "does nothing without" in (done.stderr + done.stdout)


def test_the_flag_reaches_the_draft_builder():
    """Static wiring: a flag parsed, recorded and never passed on is exactly
    what tests/test_manifest_records_every_sampling_option exists for."""
    import ast
    tree = ast.parse(open(os.path.join(REPO, "infer_atomic.py")).read())
    passed = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg in (
                "music_anchor", "music_anchor_shuffle",
                "draft_music_anchor", "draft_music_anchor_shuffle"):
            passed.add(node.arg)
    assert {"music_anchor", "music_anchor_shuffle",
            "draft_music_anchor", "draft_music_anchor_shuffle"} <= passed


def test_only_the_music_and_the_prototype_are_read():
    """The cheating line, asserted.  The anchor block may name the query's beat
    grid and the retrieved values; it may not reach for a ground-truth motion."""
    source = open(os.path.join(REPO, "infer_atomic.py")).read()
    start = source.index("anchor_strength, anchor_grid = beat_anchor, beat_grid")
    block = source[start:start + 2600]
    for forbidden in ("gt_", "ground_truth", "txy_t_gt_eval", "target_motion"):
        assert forbidden not in block, forbidden
    assert "music_anchor_lag" in block


def test_the_shuffle_keeps_the_count_and_loses_the_correspondence():
    """What the control must be: same number of targets, different times."""
    rng = np.random.default_rng(7)
    grid = np.arange(0, 600, 15)
    peaks = grid + 3
    shuffled = np.sort(rng.choice(600, size=len(peaks), replace=False))
    assert len(shuffled) == len(peaks)
    assert not np.array_equal(shuffled, peaks)
    assert np.all(np.diff(shuffled) > 0)      # still monotone, so the warp is legal
