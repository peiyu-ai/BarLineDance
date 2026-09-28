"""What "the corpus" is, once a listing stopped being allowed to answer it.

The number this file protects is 17,015 = 17,790 - 775.  The published
``runs/wild_v4_inventory.jsonl`` carries the un-subtracted 17,790 because stage
C derived its clip set by enumerating the OSS prefix, and these credentials
cannot delete, so every fps re-cut orphan comes back from every listing forever.

The second test is the one worth having.  The corpus is smaller than the prefix
for two unrelated reasons -- orphans (excluded here) and the worklist's own
mid-clip-dancer-switch exclusions (kept here, they have no 3D and drop out at
reconcile) -- and those two sets overlap.  The first version of this tool met
the resulting 1,059-vs-1,075 gap and explained it with a guess that sounded
right and was wrong.  So the reconciliation is arithmetic that can leave a
residue, not a sentence.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.build_ingest_corpus_manifest import build  # noqa: E402


def test_orphans_are_subtracted_from_the_prefix():
    present = {"a__clip000", "b__clip001", "gone__clip000"}
    report = build(present, {"gone__clip000"}, present)

    assert report["clips"] == ["a__clip000", "b__clip001"]
    assert report["counts"]["corpus"] == 2
    assert report["counts"]["orphans_excluded"] == 1


def test_the_two_reasons_the_corpus_is_smaller_are_reported_apart():
    """An orphan that is *also* outside the worklist is counted once each way.

    ``not_in_frozen_worklist`` is over the corpus (orphans already gone) while
    the worklist's ``excluded_dancer_switch`` is over the prefix, so only their
    sum can be checked against it.
    """
    present = {"kept__clip000",           # in the worklist
               "switch__clip000",         # excluded for a dancer switch
               "switch__clip001",         # excluded, and an orphan too
               }
    orphans = {"switch__clip001"}
    worklist = {"kept__clip000"}

    report = build(present, orphans, worklist)

    assert report["counts"]["not_in_frozen_worklist"] == 1        # switch__clip000
    assert report["counts"]["orphans_outside_the_worklist"] == 1  # switch__clip001
    # 1 + 1 is what the worklist would have recorded as excluded_dancer_switch.
    assert (report["counts"]["not_in_frozen_worklist"]
            + report["counts"]["orphans_outside_the_worklist"]) == 2


def test_an_orphan_the_prefix_no_longer_serves_is_counted_but_not_subtracted_twice():
    present = {"a__clip000"}
    report = build(present, {"a__clip000", "never__clip000"}, present)
    assert report["counts"]["orphans_named"] == 2
    assert report["counts"]["orphans_excluded"] == 1
    assert report["counts"]["orphans_named_but_not_under_the_prefix"] == 1
    assert report["clips"] == []
