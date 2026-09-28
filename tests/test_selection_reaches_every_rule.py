"""Every retrieval rule must reach the selection filters -- names are not enough.

THE DEFECT THIS PINS.  ``tests/test_selection_survives_seam_aware.py`` asserts
that ``_prefer_feet_lead`` and friends appear inside the seam-aware branch.
They do.  It never asked whether that branch RUNS for the rule the shipped
configuration uses, and it does not: the branch is gated on

    self.retrieval_rule in ("duration", "tempo", "medoid", "phase")

while the shipped arm is ``--retrieval-rule learned``.  Measured 2026-09-16, a
draft-only arm over the twenty eval clips with ``--draft-feet-lead`` recorded
``draft_feet_lead: true`` in its manifest and came out BYTE-IDENTICAL to the
baseline on 20 of 20 clips.  The older test stayed green the whole time.

That is exactly the shape CLAUDE.md section 2 names: a gate that can never fire
reads as "checked", and is worse than no gate.  So this file asserts the
property the other one only implied -- that for EVERY rule ``--retrieval-rule``
accepts, some code path applies the filters -- and it does so by checking the
rule lists, because a name appearing in a file proves nothing about reachability.
"""
import pathlib
import re

SOURCE = pathlib.Path("infer_atomic.py").read_text()
FILTERS = ("_prefer_feet_beat_lead", "_prefer_beat_fit", "_prefer_feet_lead")


def cli_rules():
    """The rules --retrieval-rule accepts, read from argparse rather than listed."""
    match = re.search(r'choices=\[([^\]]*)\],\s*\n?\s*(?:help=)?[^\n]*retrieval',
                      SOURCE)
    if match is None:
        match = re.search(r'"--retrieval-rule"[\s\S]{0,400}?choices=\[([^\]]*)\]',
                          SOURCE)
    assert match is not None, "could not find --retrieval-rule's choices"
    return [r.strip().strip('"\'') for r in match.group(1).split(",") if r.strip()]


def branch_after(anchor, stop='\n        elif ', limit=9000):
    """The source of the branch that begins at ``anchor``."""
    assert anchor in SOURCE, anchor
    body = SOURCE.split(anchor, 1)[1][:limit]
    return body.split(stop, 1)[0]


def test_the_cli_offers_the_rules_this_file_reasons_about():
    rules = cli_rules()
    assert "learned" in rules and "duration" in rules, rules


def test_the_learned_rule_applies_the_selection_filters():
    branch = branch_after('elif self.retrieval_rule == "learned":')
    for name in FILTERS:
        assert name in branch, (
            "--retrieval-rule learned is the shipped rule and does not reach "
            "{}; --draft-feet-lead and friends are dead in the shipped "
            "configuration".format(name))


def test_the_learned_rule_narrows_rather_than_overrides():
    """The filters may not replace the selector's own ranking."""
    branch = branch_after('elif self.retrieval_rule == "learned":')
    assert "near = narrowed or near" in branch, (
        "the filters must narrow the band and fall back to it when nothing "
        "qualifies, as the seam-aware branch does")


def test_every_cli_rule_reaches_the_filters():
    """No rule may be silently exempt.

    ``duration`` reaches them through ``_duration_pick``; ``learned`` through
    its own branch; the rest through the seam-aware branch's rule list.  A rule
    in none of those is one where setting the flag changes the manifest and not
    the dance.
    """
    seam_list = re.search(
        r'if \(occurrence or seam_aware\) and self\.retrieval_rule in \(\s*([^)]*)\)',
        SOURCE)
    assert seam_list is not None
    covered = {r.strip().strip('"\'')
               for r in seam_list.group(1).replace("\n", " ").split(",") if r.strip()}
    covered.add("duration")      # _duration_pick applies them directly
    covered.add("learned")       # its own branch, asserted above
    missing = [r for r in cli_rules() if r not in covered and r != "random"]
    assert not missing, (
        "these rules reach no code path that applies the selection filters, so "
        "--draft-feet-lead / --draft-beat-fit / --draft-feet-beat-lead are "
        "no-ops under them: {}".format(missing))


def test_the_filters_are_counted_where_they_are_used():
    """A counter, so a manifest can never imply a filter narrowed anything it did not."""
    branch = branch_after('elif self.retrieval_rule == "learned":')
    assert "selection_filter_slots" in branch and "selection_filter_applied" in branch, (
        "the repository has twice shipped a reading for a flag that changed the "
        "manifest and not the dance; count the slots and the applications")


def test_the_counters_reach_the_manifest():
    """A counter nobody can read from the outside cannot settle the argument.

    ``lev_feetlead``'s first manifest carried ``draft_feet_lead: true`` and no
    way to tell whether the filter narrowed anything -- which is the exact
    ambiguity that let three dead flags look alive for three days.
    """
    for key in ("draft_selection_filter_slots", "draft_selection_filter_applied"):
        assert '"{}"'.format(key) in SOURCE, (
            "{} is counted in the retrieval loop but never written to the "
            "manifest, so no arm's record can distinguish 'the flag was set' "
            "from 'the band actually narrowed'".format(key))
