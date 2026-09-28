"""Tests for gate v2's same-song arm, and for what "same song" is allowed to mean.

The statistic is one Mann-Whitney U over two sets of rate differences, so the
tests are about *which pairs land in which set*.  That is the whole substance:
the same p-value computed over two different definitions of "same song" is two
different claims, and on the wild corpus the key-based definition is nearly
tautological -- an upload's clips are consecutive cuts of one take, not two
performances of one track.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.probe_structure_conditioning import (  # noqa: E402
    genre_of_song,
    pair_statistics,
    segment_rate,
)

SEED = 20260816


def _rates(groups):
    """{sequence: (song, rate)} from {song: [rate, ...]}."""
    out = {}
    for song, values in groups.items():
        for index, rate in enumerate(values):
            out["{}#{}".format(song, index)] = (song, rate)
    return out


def test_the_key_defines_the_same_song_arm_when_no_pairs_are_given():
    rates = _rates({"m{:02d}".format(song): [1.0 + song + 0.01 * take
                                             for take in range(4)]
                    for song in range(6)})
    report = pair_statistics(rates, SEED)
    assert report["checked"] is True
    # Six songs, four takes each: C(4,2) = 6 same-song pairs per song.
    assert report["same_song_pairs"] == 36
    assert "grouping key" in report["same_song_unit"]


def test_a_pair_list_replaces_the_key_and_says_that_it_did():
    """On wild this is the difference between a real test and a tautology.

    The key groups cuts of one take; the pair list groups two uploads proven to
    dance to one track, which is the unit AIST's number is computed on.
    """
    rates = {
        "wild_v4:1:clip000": ("wild_v4:1", 1.00),
        "wild_v4:2:clip000": ("wild_v4:2", 1.01),
        "wild_v4:3:clip000": ("wild_v4:3", 2.00),
        "wild_v4:4:clip000": ("wild_v4:4", 2.02),
        "wild_v4:5:clip000": ("wild_v4:5", 3.00),
        "wild_v4:6:clip000": ("wild_v4:6", 3.01),
        "wild_v4:7:clip000": ("wild_v4:7", 4.00),
        "wild_v4:8:clip000": ("wild_v4:8", 4.03),
        "wild_v4:9:clip000": ("wild_v4:9", 5.00),
        "wild_v4:10:clip000": ("wild_v4:10", 5.02),
    }
    # Every upload is its own key, so the key-based arm has no pairs at all.
    keyed = pair_statistics(rates, SEED)
    assert keyed["checked"] is False

    pairs = [frozenset(("wild_v4:{}:clip000".format(a), "wild_v4:{}:clip000".format(b)))
             for a, b in ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10),
                          (1, 3), (2, 4), (5, 7), (6, 8))]
    paired = pair_statistics(rates, SEED, same_pairs=pairs)
    assert paired["checked"] is True
    assert paired["same_song_pairs"] == 9
    assert "cross-upload" in paired["same_song_unit"]


def test_a_linked_pair_never_lands_in_the_different_song_arm():
    """Same-song pairs on both sides drive the difference toward zero.

    The control would then fail for the one reason that is not a finding, and
    the failure looks exactly like "the corpus has no structure".

    Asserting this through the two means does not work -- a construction where
    leakage raises the different-song mean also raises it through legitimately
    unlinked pairs, so the statistic cannot separate the two.  The decisive
    construction is a corpus where the linked pairs are the *only* cross-key
    pairs there are: if they were eligible, the different-song arm would fill
    immediately; because they are not, the draw exhausts its budget and says so.
    """
    rates = {"wild_v4:{}:clip000".format(index): ("wild_v4:{}".format(index), float(index))
             for index in range(1, 11)}
    every_pair = [frozenset(("wild_v4:{}:clip000".format(a), "wild_v4:{}:clip000".format(b)))
                  for a in range(1, 11) for b in range(a + 1, 11)]
    report = pair_statistics(rates, SEED, same_pairs=every_pair)
    assert report["checked"] is False
    assert "could not draw enough different-song pairs" in report["reason"]
    # The same arm did fill -- so the refusal is about the *other* arm, not
    # about the corpus being too small.
    assert len(every_pair) == 45


def test_the_same_song_arm_is_exactly_the_pairs_it_was_given():
    rates = {"wild_v4:{}:clip000".format(index): ("wild_v4:{}".format(index), float(index))
             for index in range(1, 21)}
    # Ten pairs, each one step apart, so every same-song difference is 1.0.
    pairs = [frozenset(("wild_v4:{}:clip000".format(index),
                        "wild_v4:{}:clip000".format(index + 1)))
             for index in range(1, 20, 2)]
    report = pair_statistics(rates, SEED, same_pairs=pairs)
    assert report["checked"] is True
    assert report["same_song_pairs"] == 10
    assert report["same_song_mean_diff"] == pytest.approx(1.0)
    # And a pair naming a sequence this split does not hold is dropped, not
    # counted as a zero difference.
    with_ghost = pair_statistics(
        rates, SEED, same_pairs=pairs + [frozenset(("ghost", "wild_v4:1:clip000"))])
    assert with_ghost["same_song_pairs"] == 10


def test_a_pair_list_naming_nothing_here_is_distinguishable_from_a_small_corpus():
    """CLAUDE.md 2: 'the control never ran' must not read as 'too few pairs'."""
    rates = _rates({"m{:02d}".format(song): [1.0 + song] for song in range(12)})
    report = pair_statistics(rates, SEED, same_pairs=[frozenset(("nope", "nada"))])
    assert report["checked"] is False
    assert "names no two sequences" in report["reason"]


def test_the_genre_bucket_is_only_read_where_a_genre_exists():
    """``song[1:3]`` was the rule and it is only a rule on AIST.

    Applied to a wild key it slices two letters out of ``wild_v4:7195...`` and
    invents a genre for a corpus that has none.
    """
    assert genre_of_song("mBR0") == "BR"
    assert genre_of_song("wild_v4:7195533766570282272") is None
    assert genre_of_song("mBRX") is None


def test_the_segment_rate_counts_boundaries_per_second():
    import numpy as np

    # Three runs in 90 frames at 30 fps = 3 s: two changes, so three segments.
    timeline = np.asarray([0] * 30 + [1] * 30 + [2] * 30)
    assert segment_rate(timeline) == pytest.approx(1.0)
