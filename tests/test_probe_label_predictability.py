"""The mispaired-music control, shown pairing and refusing to pair.

The criterion built on it -- ``--min-real-minus-shuffled`` -- demands that
handing a window the wrong music COSTS accuracy.  For that to be a check rather
than a decoration, the wrong music has to actually be a different song.  It was
not: with ``shuffle=False`` on the eval loader, ``music.roll(1)`` handed each
window another recording of the SAME backing track 99.5% of the time on this
corpus, so the criterion could not fail whatever the model did.
"""

import numpy as np

from tools.probe_label_predictability import donor_permutation


def _names(songs):
    return ["gBR_sBM_cAll_d04_{}_ch{:02d}__slice{}".format(song, i, i)
            for i, song in enumerate(songs)]


class TestDonorIsADifferentSong:
    def test_every_window_is_given_another_songs_music(self):
        names = _names(["mBR0", "mBR0", "mBR1", "mBR1", "mBR2", "mBR2"])
        donor, stats = donor_permutation(names)
        assert stats["donor_is_a_different_song"] == 1.0
        assert stats["songs_in_split"] == 3
        for index, target in enumerate(donor):
            assert names[index].split("_")[4] != names[target].split("_")[4]

    def test_the_old_roll_would_have_paired_same_song_neighbours(self):
        # The window order this control has to survive: adjacent entries are
        # different recordings of one track, which is what made roll(1) vacuous.
        names = _names(["mBR0", "mBR0", "mBR0", "mBR1", "mBR1", "mBR1"])
        rolled_same_song = sum(
            1 for i in range(len(names))
            if names[i].split("_")[4] == names[i - 1].split("_")[4]
        )
        assert rolled_same_song >= 4                      # the defect, reproduced
        _, stats = donor_permutation(names)
        assert stats["donor_is_a_different_song"] == 1.0  # and removed

    def test_a_single_song_cannot_be_mispaired_and_says_so(self):
        # One track in the split: there is no different song to donate, and the
        # statistic must report that rather than quietly returning identity.
        names = _names(["mBR0", "mBR0", "mBR0"])
        donor, stats = donor_permutation(names)
        assert stats["songs_in_split"] == 1
        assert stats["donor_is_a_different_song"] == 0.0
        assert list(donor) == [0, 1, 2]

    def test_uneven_song_sizes_still_cover_every_window(self):
        names = _names(["mBR0", "mBR0", "mBR0", "mBR1"])
        donor, stats = donor_permutation(names)
        assert len(donor) == len(names)
        assert stats["donor_is_a_different_song"] == 1.0

    def test_names_without_a_song_id_are_counted_not_hidden(self):
        names = ["wild_clip_000__slice0", "wild_clip_001__slice0"]
        _, stats = donor_permutation(names)
        assert stats["names_without_a_song_id"] == 2

    def test_a_corpus_with_no_song_ids_still_gets_mispaired(self):
        # Grouping every window under one None key left the identity permutation
        # -- each window handed back its OWN music -- while the report claimed
        # 100% mispairing, because a None donor for a None window counted as
        # "different".  The shuffled pass was then byte-identical to the real
        # one and the gate fired "reading the label prior" on every non-AIST
        # release, whatever the model did.
        names = ["wild/clip{:03d}__slice0".format(i) for i in range(4)]
        donor, stats = donor_permutation(names)
        assert list(donor) != list(range(4))               # it really mispairs
        assert stats["donor_is_a_different_group"] == 1.0
        assert stats["donor_is_a_different_song"] == 0.0   # and does not pretend otherwise
        assert stats["grouping_unit"].startswith("recording")

    def test_slices_of_one_recording_are_one_group_in_the_fallback(self):
        names = ["wild/clipA__slice{}".format(i) for i in range(3)] + ["wild/clipB__slice0"]
        _, stats = donor_permutation(names)
        assert stats["groups_in_split"] == 2
