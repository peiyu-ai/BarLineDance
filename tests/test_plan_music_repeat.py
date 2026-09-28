"""``--plan-music-repeat``: tie bars that sound alike to one class.

Every test here names which half of the mechanism it pins.  The two that would
have caught the failures this repository has already paid for are
``test_strength_zero_is_a_no_op`` (a flag that changes nothing while the
manifest says it was set) and ``test_reads_no_motion_and_no_labels_but_its_own``
(a timing fix that works by reading the target's ground truth).
"""
import numpy as np
import pytest
import torch

import infer_atomic


BEAT = infer_atomic.BEAT_CHANNEL


def music_with_beats(bars, frames_per_bar=12, beats_per_bar=4, dim=35, seed=0):
    """A clip whose beat channel marks a regular grid of ``bars`` bars.

    Channel 0 carries an onset peak at every bar line ON PURPOSE.  ``choose_phase``
    picks the 4-beat phase with the most onset energy on it, so a flat channel 0
    let it pick phase 3 and the grid came back as a leading 9-frame stub plus
    eight bars -- nine bars for eight painted sections, and the positive control
    failed for a reason that had nothing to do with the mechanism under test.
    With the peaks the chosen phase is 0 and the grid is the one painted.
    """
    rng = np.random.default_rng(seed)
    frames = bars * frames_per_bar
    music = rng.normal(size=(frames, dim)).astype(np.float32) * 0.01
    step = frames_per_bar // beats_per_bar
    assert step * beats_per_bar == frames_per_bar
    music[:, BEAT] = 0.0
    music[::step, BEAT] = 1.0
    music[:, 0] = 0.0
    music[::frames_per_bar, 0] = 1.0
    return torch.from_numpy(music)


def grid(music, beats_per_bar=4):
    """The bars the pass will actually use, so a test paints the same bars."""
    bounds, _ = infer_atomic.bar_grid_bounds(music, beats_per_bar, length=len(music))
    assert bounds is not None
    return list(bounds)


def paint_sections(music, sections, beats_per_bar=4):
    """Give every bar of a section one timbre+chroma, so the descriptor cosine
    within a section is 1 and between sections is not.  Painted on the bounds
    ``bar_grid_bounds`` returns rather than on a stride of its own."""
    bounds = grid(music, beats_per_bar)
    assert len(bounds) - 1 == len(sections), (len(bounds) - 1, len(sections))
    rng = np.random.default_rng(7)
    signatures = {s: rng.normal(size=32) * 3.0 for s in sorted(set(sections))}
    for (lo, hi), section in zip(zip(bounds[:-1], bounds[1:]), sections):
        music[lo:hi, 1:33] = torch.tensor(signatures[section], dtype=torch.float32)
    return music


def labels_per_bar(music, values, beats_per_bar=4):
    bounds = grid(music, beats_per_bar)
    assert len(bounds) - 1 == len(values), (len(bounds) - 1, len(values))
    labels = torch.zeros(len(music), dtype=torch.long)
    for (lo, hi), value in zip(zip(bounds[:-1], bounds[1:]), values):
        labels[lo:hi] = int(value)
    return labels


def bar_values(labels, music, beats_per_bar=4):
    bounds = grid(music, beats_per_bar)
    return [int(labels[lo]) for lo in bounds[:-1]]


def test_bars_that_sound_alike_end_on_one_class():
    """THE POSITIVE CONTROL.  Sections A B A B with the planner naming a
    different class in every bar: the pass must hand each section one class."""
    sections = list("ABABABAB")
    music = paint_sections(music_with_beats(len(sections)), sections)
    labels = labels_per_bar(music, [1, 2, 3, 4, 5, 6, 7, 8])
    out, report = infer_atomic.music_repeat_plan(labels, music, 4, strength=0.75)
    values = bar_values(out, music)
    a = {v for v, s in zip(values, sections) if s == "A"}
    b = {v for v, s in zip(values, sections) if s == "B"}
    assert len(a) == 1 and len(b) == 1, values
    assert a != b, "two sections that sound different collapsed onto one class"
    assert report["groups"] == 2


def test_strength_zero_is_a_no_op():
    """A flag that is off must be off: identical tensor, and no report, so an
    arm cannot be named after a strength that did nothing."""
    sections = list("ABABABAB")
    music = paint_sections(music_with_beats(len(sections)), sections)
    labels = labels_per_bar(music, [1, 2, 3, 4, 5, 6, 7, 8])
    out, report = infer_atomic.music_repeat_plan(labels, music, 4, strength=0.0)
    assert torch.equal(out, labels)
    assert report is None


def test_reads_no_motion_and_no_labels_but_its_own():
    """The forbidden failure mode: reading the target clip's ground-truth
    MOTION.  The pass takes exactly two arguments, the planner's own labels and
    the query's own music, so changing the ground truth cannot change it.  This
    test asserts the signature, not the intent."""
    import inspect
    parameters = list(inspect.signature(infer_atomic.music_repeat_plan).parameters)
    assert parameters == ["labels", "music", "beats_per_segment", "strength"]


def test_invents_no_class():
    """Every class in the output was already in the planner's own plan."""
    sections = list("ABCABCAB")
    music = paint_sections(music_with_beats(len(sections)), sections)
    labels = labels_per_bar(music, [4, 9, 2, 4, 9, 2, 4, 9])
    out, _ = infer_atomic.music_repeat_plan(labels, music, 4, strength=0.7)
    assert set(out.tolist()) <= set(labels.tolist())


def test_the_group_takes_the_class_it_spent_most_frames_on():
    """Not the first bar's class and not the rarest: the frame-weighted winner.
    Section A holds class 5 in two bars and class 6 in one, so A must read 5."""
    sections = list("AAABBBBB")
    music = paint_sections(music_with_beats(len(sections)), sections)
    labels = labels_per_bar(music, [5, 5, 6, 7, 7, 7, 7, 7])
    out, _ = infer_atomic.music_repeat_plan(labels, music, 4, strength=0.75)
    assert bar_values(out, music)[:3] == [5, 5, 5]


def test_transition_does_not_win_a_group_that_names_anything():
    """Class 0 is transition.  A group that is mostly transition but names one
    atomic movement must take the movement, the same rule
    ``snap_plan_to_bar_grid`` states: plurality would hand whole bars to the
    biggest class, and transition is already the biggest."""
    sections = list("AAAABBBB")
    music = paint_sections(music_with_beats(len(sections)), sections)
    labels = labels_per_bar(music, [0, 0, 0, 3, 9, 9, 9, 9])
    out, _ = infer_atomic.music_repeat_plan(labels, music, 4, strength=0.75)
    assert bar_values(out, music)[:4] == [3, 3, 3, 3]


def test_ties_are_broken_reproducibly():
    """Two classes with the same frame count must not resolve by dict order, or
    the arm is not reproducible from its own manifest."""
    sections = list("AAAABBBB")
    music = paint_sections(music_with_beats(len(sections)), sections)
    labels = labels_per_bar(music, [2, 2, 8, 8, 5, 5, 5, 5])
    first, _ = infer_atomic.music_repeat_plan(labels, music, 4, strength=0.75)
    # Section A splits its frames evenly between 2 and 8.  The lower class id
    # has to win, every time, or an arm is not reproducible from its manifest.
    assert bar_values(first, music)[:4] == [2, 2, 2, 2]
    again, _ = infer_atomic.music_repeat_plan(labels, music, 4, strength=0.75)
    assert torch.equal(first, again)


def test_a_clip_with_no_bar_grid_is_left_alone():
    """Same rule ``bar_grid_bounds`` states: no grid, and NOT a fallback grid."""
    music = torch.zeros(40, 35)
    labels = torch.arange(40) % 5
    out, report = infer_atomic.music_repeat_plan(labels, music, 4, strength=0.8)
    assert torch.equal(out, labels)
    assert report is None


def test_too_few_bars_to_tie_is_left_alone():
    sections = list("AB")
    music = paint_sections(music_with_beats(len(sections)), sections)
    labels = labels_per_bar(music, [1, 2])
    out, report = infer_atomic.music_repeat_plan(labels, music, 4, strength=0.8)
    assert torch.equal(out, labels)
    assert report is None


def test_strength_is_monotone_in_how_much_it_ties():
    """More strength must never mean more distinct classes; that is what makes
    it a dose and lets a sweep be read as one."""
    sections = list("ABCDABCDABCDABCD")
    music = paint_sections(music_with_beats(len(sections)), sections)
    labels = labels_per_bar(music, list(range(1, 17)))
    counts = []
    for strength in (0.2, 0.4, 0.6, 0.8):
        out, _ = infer_atomic.music_repeat_plan(labels, music, 4, strength=strength)
        counts.append(len(set(out.tolist())))
    assert counts == sorted(counts, reverse=True), counts


def test_out_of_range_strength_is_refused():
    music = paint_sections(music_with_beats(8), list("ABABABAB"))
    with pytest.raises(ValueError):
        infer_atomic.music_repeat_plan(labels_per_bar(music, list(range(8))), music, 4, 1.5)


def test_descriptor_ignores_the_beat_and_onset_channels():
    """Channels 0/33/34 repeat every bar by construction; if the descriptor saw
    them every pair of bars would look alike and the pass would collapse any
    clip onto one class."""
    bars = 8
    music = paint_sections(music_with_beats(bars), list("ABABABAB"))
    bounds, _ = infer_atomic.bar_grid_bounds(music, 4, length=len(music))
    before = infer_atomic.bar_music_descriptors(music, bounds)
    loud = music.clone()
    loud[:, 0] *= 50.0
    loud[:, 33] = 1.0
    after = infer_atomic.bar_music_descriptors(loud, bounds)
    assert torch.allclose(before, after, atol=1e-6)
