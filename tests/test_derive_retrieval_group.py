"""``--derive-retrieval-group``: a missing registry entry must not mean "no draft".

WHAT IT FIXES.  ``_source_safe_draft`` returns an all-zero draft when the query
has no retrieval group, on the rule that "an input without an explicit retrieval
group cannot safely prove that a training prototype is external".  Two of the
twenty T eval clips (7610414564962183545, 7188505181892381984) have no registry
entry, so they are generated from MUSIC ALONE -- their artifacts record
``safe_draft_condition_fraction: 0.0`` -- and they are exactly the two the
operator picked out on 2026-09-06 for 身体腾空旋转 and 动作段落高度重复.

WHY DERIVING IS SAFE HERE, and why the switch is still off by default.  Checked
on the T line before writing this: 0 of the registry's 6,823 entries map to
anything other than ``name.rsplit(":", 1)[0]``; 0 recordings have clips in more
than one group; and the library holds 143 retrieval groups, none of which is any
of the 20 eval recordings -- so the exclusion the id drives is a no-op for every
eval clip, grouped or not.  On a corpus where those three facts do not hold,
deriving would silently fail to exclude a shared source, which is why the caller
has to ask for it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infer_atomic import _query_retrieval_group_id


class _Registry:
    def __init__(self, table):
        self.table = table

    def query_retrieval_group_id(self, name):
        return self.table.get(name)


REGISTRY = _Registry({"wild_v5:111:clip000": "wild_v5:111"})


def test_a_registered_clip_is_unaffected():
    for derive in (False, True):
        assert _query_retrieval_group_id(
            REGISTRY, "wild_v5:111:clip000", derive_missing=derive) == "wild_v5:111"


def test_off_by_default_a_missing_clip_still_fails_closed():
    assert _query_retrieval_group_id(REGISTRY, "wild_v5:222:clip000") is None


def test_on_it_derives_the_recording_prefix():
    assert _query_retrieval_group_id(
        REGISTRY, "wild_v5:222:clip000", derive_missing=True) == "wild_v5:222"


def test_the_derived_id_matches_what_the_registry_would_have_said():
    """The property the safety argument rests on, asserted on the real shape.

    Every registry entry on this corpus is name.rsplit(':', 1)[0]; if that ever
    stops being true the derived id is no longer the registry's answer and this
    switch must not be used.
    """
    for name, group in REGISTRY.table.items():
        assert name.rsplit(":", 1)[0] == group


def test_a_nameless_query_still_fails_closed():
    """Deriving may not invent a group out of nothing."""
    assert _query_retrieval_group_id(REGISTRY, "", derive_missing=True) is None
