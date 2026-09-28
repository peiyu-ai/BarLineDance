import numpy as np
import torch

from tools.render_plan_grid_sheet import bar_bounds, snap_to_bars


def test_a_bar_takes_its_majority_non_transition_class():
    """The rule has no free parameter, and it is deliberately not a majority.

    A plain majority would hand a bar to transition whenever transition holds
    more than half its frames -- and the planner is already 52.6% transition,
    so that is the same "plurality favours the biggest class" defect one level
    up.  Measured 2026-08-23 over 596 bars: majority pooling leaves the share
    at 0.5255, this rule brings it to 0.3712 against a ground truth of 0.3214.
    """
    # Bar 0: mostly transition but the planner did name class 7 twice.
    # Bar 1: the planner named nothing at all.
    plan = torch.tensor([0, 0, 0, 7, 0, 7] + [0] * 6)
    out = snap_to_bars(plan, [0, 6, 12])
    assert out[:6].tolist() == [7] * 6, "a named class must survive a transition majority"
    assert out[6:].tolist() == [0] * 6, "a bar with no class named stays transition"


def test_the_bar_wins_by_count_when_two_classes_are_named():
    plan = torch.tensor([3, 3, 3, 9, 0, 0])
    assert snap_to_bars(plan, [0, 6]).tolist() == [3] * 6


def test_bars_are_cut_on_the_music_and_refused_when_there_is_no_grid():
    music = np.zeros((200, 35), dtype=np.float32)
    music[:, 0] = 1.0                      # flat onset envelope
    music[np.arange(0, 200, 16), 34] = 1.0  # a beat every 16 frames
    bounds, phase, beats = bar_bounds(music, 200)
    assert bounds[0] == 0 and bounds[-1] == 200
    assert len(beats) == 13
    # every interior cut is a beat, four beats apart
    interior = bounds[1:-1]
    assert all(c % 16 == 0 for c in interior)
    assert all(b - a == 64 for a, b in zip(interior[:-1], interior[1:]))

    silent = np.zeros((200, 35), dtype=np.float32)
    assert bar_bounds(silent, 200) == (None, None, None)


def test_the_clips_own_wav_is_found_in_the_ingest_tree(tmp_path, monkeypatch):
    """A silent inference video cannot be read for beat alignment at all.

    The first version of this page claimed the corpus had no playable audio.
    It does -- the directory the *models* read holds 35-D feature arrays, and
    that was mistaken for the corpus having no sound.  The wav is in the ingest
    tree under a different naming convention, which is the whole of the bug.
    """
    import tools.render_plan_grid_sheet as sheet

    ingest = tmp_path / "wild_ingest_v1"
    (ingest / "7326579334368578868__clip000").mkdir(parents=True)
    wav = ingest / "7326579334368578868__clip000" / "audio.wav"
    wav.write_bytes(b"RIFF")
    monkeypatch.setattr(sheet, "INGEST", ingest)

    assert sheet.source_audio("wild_v4:7326579334368578868:clip000") == wav
    # absent, and malformed, both resolve to None rather than to a guess
    assert sheet.source_audio("wild_v4:0000000000000000000:clip000") is None
    assert sheet.source_audio("not-a-clip-id") is None
