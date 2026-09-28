"""The pre-split audit's two criteria, shown failing and passing.

A gate that cannot fail reads like a check and is worse than no check -- this
repo has the `df`-based quota gate on record for exactly that.  So both of the
audit's verdicts are exercised in both directions here, on corpora built so the
right answer is known before the tool runs.
"""

import numpy as np

from tools.audit_genre_presplit import (
    PAPER_SAMPLES_PER_SUBPROTOTYPE,
    shares,
    summarise,
)


def _corpus(rows):
    """``[(prototype, genre, count)]`` -> the two aligned columns the audit reads."""
    prototypes, genres = [], []
    for prototype, genre, count in rows:
        prototypes += [prototype] * count
        genres += [genre] * count
    return np.asarray(prototypes), np.asarray(genres)


class TestShapeIsScaleFree:
    def test_doubling_every_cell_leaves_the_histogram_alone(self):
        # The comparison against Fig. 4c is in units of each side's own mean,
        # so that a corpus with half the segments per cell is not scored for
        # its density twice -- once as a rate and again as a shape.
        sizes = np.array([4.0, 9.0, 30.0, 31.0, 32.0, 90.0])
        first = shares(sizes, float(sizes.mean()))
        second = shares(sizes * 2, float((sizes * 2).mean()))
        assert first == second

    def test_a_cell_at_the_paper_mean_lands_in_the_papers_mode_bucket(self):
        sizes = np.array([PAPER_SAMPLES_PER_SUBPROTOTYPE] * 5)
        assert shares(sizes, float(sizes.mean()))["20-35"] == 1.0


class TestCountCriterion:
    def test_seven_genres_per_prototype_agrees_with_the_papers_7_3(self):
        rows = [(p, "g%d" % g, 30) for p in range(1, 11) for g in range(7)]
        report = summarise(*_corpus(rows), repeats=20)
        assert report["cells"]["per_prototype"] == 7.0
        assert report["hypothesis"]["count_agrees_within_10pct"] is True

    def test_genre_pure_prototypes_disagree_with_it(self):
        # One genre per prototype is the opposite regime: the pre-split does
        # nothing and every sub-prototype has to come from the clustering.  The
        # criterion has to say so rather than pass on a technicality.
        rows = [(p, "g%d" % p, 200) for p in range(1, 11)]
        report = summarise(*_corpus(rows), repeats=20)
        assert report["cells"]["per_prototype"] == 1.0
        assert report["hypothesis"]["count_agrees_within_10pct"] is False


class TestShapeCriterion:
    def test_cells_shaped_like_fig_4c_pass(self):
        # Roughly the published mix: a mode at 20-35, a low tail near 15% and a
        # high tail near 12%.
        rows = ([(p, "gA", 27) for p in range(1, 60)]
                + [(p, "gB", 10) for p in range(1, 11)]
                + [(p, "gC", 65) for p in range(1, 9)])
        report = summarise(*_corpus(rows), repeats=20)
        assert report["hypothesis"]["shape_agrees_within_50pct"] is True

    def test_a_long_tail_of_tiny_cells_fails(self):
        # What the AIST++ pre-split actually produces: a median of 6 against a
        # mean of 18.6.  Fig. 4c has 15% of its classes below 20 samples and
        # this has most of them, so the identification must be refused.
        rows = ([(p, "gA", 200) for p in range(1, 11)]
                + [(p, "g%d" % g, 2) for p in range(1, 11) for g in range(2, 10)])
        report = summarise(*_corpus(rows), repeats=20)
        assert report["hypothesis"]["low_tail_ratio"] > 2.0
        assert report["hypothesis"]["shape_agrees_within_50pct"] is False


class TestNull:
    def test_genre_blind_data_reaches_every_cell(self):
        # With enough segments per prototype and no association, a prototype
        # touches all ten genres -- which is why "7 of 10" is only evidence of
        # structure once the null is on the page.
        rng = np.random.default_rng(7)
        prototypes = rng.integers(1, 11, size=4000)
        genres = np.asarray(["g%d" % g for g in rng.integers(0, 10, size=4000)])
        report = summarise(prototypes, genres, repeats=20)
        assert report["null_genre_permutation"]["cells_per_prototype"] > 9.5
        assert report["null_genre_permutation"]["p_value_fewer_cells"] > 0.05

    def test_concentration_is_significant_against_it(self):
        rows = [(p, "g%d" % ((p + g) % 10), 40) for p in range(1, 11) for g in range(3)]
        report = summarise(*_corpus(rows), repeats=50)
        assert report["cells"]["per_prototype"] == 3.0
        assert report["null_genre_permutation"]["p_value_fewer_cells"] == 0.0

    def test_the_null_preserves_the_marginals_it_claims_to(self):
        # If permutation changed the genre marginals the null would answer a
        # different question -- "what if the corpus had other genres" rather
        # than "what if genre did not predict the prototype".
        prototypes, genres = _corpus([(1, "gA", 30), (1, "gB", 10), (2, "gA", 5)])
        report = summarise(prototypes, genres, repeats=10)
        assert report["corpus"]["genres"] == ["gA", "gB"]
        assert report["corpus"]["segments"] == 45
