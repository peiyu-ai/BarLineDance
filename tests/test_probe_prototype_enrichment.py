"""The instrument M2's numbers have to pass through before they judge anything.

M2 reports acceptance rate and dispersion, and neither can fail usefully:
K-means minimises within-cluster distance, so "members of a prototype are close
to each other" is true of noise.  This probe measures something outside the
embedding instead, and the tests here pin the two properties that make it an
instrument rather than a number.

The second test is the one worth having.  Segments of one recording resemble
each other for reasons that have nothing to do with the attribute, so a null
that shuffles the attribute over *segments* is beaten by any clustering that
merely keeps a recording together -- it would report a large lift for a
partition that found nothing.  Permuting across entities removes exactly that
and leaves the alignment between partition and attribute as the only thing to
explain.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.probe_prototype_enrichment import (  # noqa: E402
    attribute_of, entity_of, probe, purity, restrict)


def test_purity_is_the_modal_share_inside_each_prototype():
    prototypes = [1, 1, 1, 2, 2]
    values = ["a", "a", "b", "c", "c"]
    assert purity(prototypes, values) == 4 / 5


def test_a_partition_that_only_tracks_the_entity_does_not_beat_the_null():
    """Four uploads, two accounts, and a clustering that groups by upload.

    Each prototype is one upload, so its account purity is 1.0 -- and it should
    be 1.0 under the null too, because permuting the account across uploads
    leaves every prototype holding one upload.  A segment-level null would call
    this a perfect result.
    """
    prototypes, entities = [], []
    for index, upload in enumerate(["u1", "u2", "u3", "u4"]):
        prototypes.extend([index] * 5)
        entities.extend([upload] * 5)
    attributes = {"u1": "A", "u2": "A", "u3": "B", "u4": "B"}
    recordings = ["{}:clip000".format(entity) for entity in entities]

    report = probe(recordings, prototypes, attributes, entities,
                   rounds=200, seed=1)

    assert report["observed_purity"] == 1.0
    assert report["lift"] == 1.0
    assert report["p_value"] > 0.5      # indistinguishable from the null


def test_a_partition_aligned_with_the_attribute_across_entities_does_beat_it():
    """Two prototypes, each holding several uploads that share an account."""
    prototypes, entities = [], []
    for upload in ["u1", "u2", "u3", "u4"]:
        prototypes.extend([0] * 3)
        entities.extend([upload] * 3)
    for upload in ["u5", "u6", "u7", "u8"]:
        prototypes.extend([1] * 3)
        entities.extend([upload] * 3)
    attributes = {u: ("A" if u in {"u1", "u2", "u3", "u4"} else "B")
                  for u in ["u1", "u2", "u3", "u4", "u5", "u6", "u7", "u8"]}
    recordings = ["{}:clip000".format(entity) for entity in entities]

    report = probe(recordings, prototypes, attributes, entities,
                   rounds=500, seed=1)

    assert report["observed_purity"] == 1.0
    assert report["lift"] > 1.2
    assert report["p_value"] < 0.05


def test_the_entity_is_the_upload_in_the_wild_and_the_recording_on_aist():
    assert entity_of("wild_v4:7052021260485676328:clip002", "account") == \
        "wild_v4:7052021260485676328"
    assert entity_of("aistpp/gBR_sBM_cAll_d04_mBR0_ch01", "aist-genre") == \
        "aistpp/gBR_sBM_cAll_d04_mBR0_ch01"


def test_the_aist_genre_comes_out_of_the_recording_name():
    assert attribute_of("aistpp/gBR_sBM_cAll_d04_mBR0_ch01", "aist-genre", None) == "gBR"
    assert attribute_of("aistpp/gWA_sFM_c01_d25_mWA4_ch05", "aist-genre", None) == "gWA"
    assert attribute_of("something-else", "aist-genre", None) is None


def test_an_account_that_cannot_be_resolved_is_reported_not_guessed():
    assert attribute_of("wild_v4:999:clip000", "account", {"111": "someone"}) is None


def test_the_aist_dancer_is_the_identity_control():
    """Same name, a different field: identity with movement held roughly fixed.

    AIST++ has one choreography danced by several dancers and one dancer across
    genres, so a clustering that grouped *movement* should sit near chance on
    this attribute while sitting far above it on genre.  Two readings from one
    statistic are what let the wild corpus's account -- which is a person and a
    style at once -- be placed between them instead of guessed at.
    """
    assert attribute_of("aistpp/gBR_sBM_cAll_d04_mBR0_ch01", "aist-dancer", None) == "d04"
    assert attribute_of("aistpp/gWA_sFM_c01_d25_mWA4_ch05", "aist-dancer", None) == "d25"
    assert attribute_of("aistpp/no_dancer_here", "aist-dancer", None) is None
    assert entity_of("aistpp/gBR_sBM_cAll_d04_mBR0_ch01", "aist-dancer") == \
        "aistpp/gBR_sBM_cAll_d04_mBR0_ch01"


def test_restricting_to_a_clip_list_crosses_the_two_id_shapes():
    """A label tree names ``wild_v4:<upload>:clip000``; a corpus list names
    ``<upload>__clip000``.  Reading one generation's labels on another
    generation's corpus is the only way to compare two clusterings without also
    changing who is in the corpus, and it is worth nothing if the join silently
    matches nothing."""
    recordings = ["wild_v4:111:clip000", "wild_v4:111:clip001", "wild_v4:222:clip000"]
    prototypes = [7, 8, 9]

    kept_recordings, kept_prototypes = restrict(
        recordings, prototypes, ["111__clip000", "222__clip000"])

    assert kept_recordings == ["wild_v4:111:clip000", "wild_v4:222:clip000"]
    assert kept_prototypes == [7, 9]


def test_a_stem_shape_that_matches_nothing_keeps_nothing():
    """The failure the CLI turns into a message naming both shapes.

    Without it an id-shape drift reads as a small corpus rather than as a
    broken join, and a lift computed over the survivors would look ordinary."""
    kept_recordings, kept_prototypes = restrict(
        ["wild_v4:111:clip000"], [7], ["wild_v4:111:clip000"])

    assert kept_recordings == [] and kept_prototypes == []
