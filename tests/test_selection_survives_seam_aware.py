"""The selection filters must survive --draft-seam-aware-retrieval.

THE DEFECT THIS PINS, found by the operator from the video: "跟已有的 baseline
很多是一模一样的,这个很奇怪".  ``--draft-seam-aware-retrieval`` is on in the
shipped configuration and it rebuilt its candidate band from the FULL candidate
list after ``_duration_pick`` had already filtered:

    near = [c for c, d in zip(candidates, lengths) if d <= slack] or [chosen]

so every selection filter -- --draft-feet-lead, --draft-beat-fit,
--draft-feet-beat-lead -- was discarded and ``chosen`` survived only as the
thing to avoid.  Measured on wild_v5:7650126416710192357:clip000: the
feet-beat-lead filter changed its pick in 11 of 12 slots while the draft came
out BYTE-IDENTICAL with the flag on and off.
"""
import pathlib
import re

SOURCE = pathlib.Path("infer_atomic.py").read_text()


def seam_aware_branch():
    body = SOURCE.split("if (occurrence or seam_aware) and self.retrieval_rule in (", 1)[1]
    return body.split("def ", 1)[0]


def test_the_band_is_narrowed_by_every_selection_filter():
    branch = seam_aware_branch()
    for name in ("_prefer_feet_beat_lead", "_prefer_beat_fit", "_prefer_feet_lead"):
        assert name in branch, (
            "{} does not narrow the seam-aware band, so turning it on changes "
            "nothing in the shipped configuration".format(name))


def test_the_narrowing_happens_before_the_ranking():
    branch = seam_aware_branch()
    narrow = branch.index("_prefer_feet_beat_lead")
    ranked = branch.index("ranked = sorted(")
    assert narrow < ranked, (
        "filtering after the ranking would let the ranking pick a candidate the "
        "filter had excluded")


def test_an_empty_narrowing_falls_back_to_the_band():
    """A filter that excludes everything must not leave the draw with nothing."""
    assert "near = narrowed or near" in seam_aware_branch()


def test_the_draw_is_still_a_sample_over_the_top_k():
    """The fix must not turn seam-aware into an argmax: that is exactly how
    --retrieval-rule phase failed -- it landed its criterion perfectly and lost
    on the output because a hard ranking squeezed out the variety."""
    branch = seam_aware_branch()
    assert "top = ranked[:max(1, int(self.join_top_k))]" in branch
    assert "generator.integers(len(top))" in branch


def test_the_defect_is_recorded_where_the_code_is():
    """A future reader who deletes the narrowing must meet the reason."""
    branch = seam_aware_branch()
    assert "byte-identical" in branch.lower() or "BYTE-IDENTICAL" in branch
