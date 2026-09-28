"""Accent density: the one column this round found that separates and holds up.

Three things are pinned, each because it already went wrong:
* the column must be BUYABLE-BY-NOISE aware -- adding 1 cm of jitter raises it
  2.276 -> 2.45, so jitter is gated beside it (CLAUDE.md 13.3);
* it must fall when the dance is dulled, or it is not reading movement;
* the on-beat half of the same file is REFUTED and must never be what a
  judgement is paired on -- the first version paired on exactly that.
"""
import numpy as np
import pytest
from scipy.ndimage import uniform_filter1d

from tools.render_plan_strip import accents


def dancer(frames=600, joints=24, seed=0, hits=25):
    """A body with real, sharp accents: a smooth carrier plus periodic hits.
    ``hits`` is the spacing in frames -- smaller means accenting more often."""
    rng = np.random.default_rng(seed)
    t = np.arange(frames)[:, None, None]
    motion = 0.3 * np.sin(t / 9.0 + rng.normal(size=(1, joints, 3)))
    for hit in range(20, frames - 20, hits):
        motion[hit:hit + 3] += 0.12
    return motion


def density(joints, fps=30.0):
    return len(accents(joints)) / (len(joints) / fps)


def test_more_hits_read_as_more_accents():
    """The property the column is actually used for: two bodies of the same kind,
    one hitting more often, must separate."""
    assert density(dancer(hits=25)) > density(dancer(hits=60))


def test_the_threshold_is_relative_so_uniform_dulling_need_not_show():
    """NOT a defect, a limit, and it is pinned so nobody re-derives it as one.
    ``accents`` thresholds at the 90th percentile of the clip's own speed
    changes, so scaling a whole dance down scales the threshold with it.  On
    real ground truth the smoothing control DOES bite (2.276 -> 1.404) because
    smoothing removes isolated hits and changes the distribution's SHAPE; a
    uniform scale change is a different thing and this test says so."""
    original = dancer()
    assert density(0.4 * original) == pytest.approx(density(original), abs=0.2)


def test_noise_raises_it_so_it_can_never_be_read_alone():
    """The control that says it must be gated with jitter.  If this ever stops
    holding, the gate can be relaxed -- until then it cannot."""
    rng = np.random.default_rng(1)
    original = dancer()
    noisy = original + rng.normal(0, 0.01, original.shape)
    assert density(noisy) > density(original)


def test_a_still_body_has_no_accents():
    """A percentile threshold with no floor promotes numerical noise on a static
    skeleton to the 90th percentile and calls every frame an accent."""
    assert density(np.zeros((300, 24, 3))) == 0.0


def test_the_judgement_is_paired_on_density_and_against_ground_truth():
    """The first version paired on the REFUTED on-beat rate, and against the
    first arm instead of the dancer."""
    import pathlib
    source = pathlib.Path("tools/score_accent_density.py").read_text()
    block = source.split("paired against GROUND TRUTH", 1)[1]
    assert 'rows[name][c]["accents_per_s"]' in block
    assert 'rows[name][c]["rate"]' not in block
    assert 'base = "ground truth"' in source


def test_the_refuted_on_beat_column_still_prints_its_refutation():
    """Deleting it would let someone rediscover it and repeat the mistake."""
    import pathlib
    source = pathlib.Path("tools/score_accent_density.py").read_text()
    assert "REFUTED" in source and "chance" in source
    assert "Do not judge with on-beat/s" in source
