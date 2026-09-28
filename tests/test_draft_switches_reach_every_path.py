"""Every draft switch must reach EVERY build_draft call, not just the first.

WHY THIS IS GENERIC AND NOT ONE MORE PER-FLAG TEST.  ``_source_safe_draft`` has
two ``library.build_draft`` calls -- an early return and the normal one -- and a
switch added to only one of them is accepted, recorded in the manifest and
silently inert on whichever path the run takes.  This repository has paid for
that shape at least twice: the batched path skipped the draft switches while
``--inference-batch-size`` defaulted to 4, and on 2026-09-12
``--draft-rhythm-weight`` and ``--draft-music-anchor-lag`` were wired into the
early-return call only.  The rhythm arms then came out BYTE-IDENTICAL to the
phase arms and to each other across weights 0.5 and 2.0, which is what exposed
it -- the flag was doing nothing at all while the scorecard reported a number
for it.

A per-flag test cannot catch the next one.  This reads the source and asserts
the property.
"""
import ast
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError("{} not found".format(name))


def _build_draft_calls(node):
    calls = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Attribute) and func.attr == "build_draft":
                calls.append(child)
    return calls


def test_both_build_draft_calls_forward_the_same_switches():
    tree = ast.parse(open(os.path.join(REPO, "infer_atomic.py")).read())
    draft = _function(tree, "_source_safe_draft")
    calls = _build_draft_calls(draft)
    assert len(calls) >= 2, "expected the early return AND the normal path"
    # ``exclude_retrieval_group_ids`` is the ONE legitimate difference: the
    # early return is the ``--unsourced-retrieval`` path, which by definition
    # excludes nothing, and ``allow_missing`` goes with it.  Everything else is
    # a draft-shaping switch and must appear on both, or it is inert on one.
    BY_DESIGN = {"exclude_retrieval_group_ids", "allow_missing"}
    keyword_sets = [{k.arg for k in call.keywords if k.arg} - BY_DESIGN
                    for call in calls]
    first = keyword_sets[0]
    for index, other in enumerate(keyword_sets[1:], start=1):
        missing = first - other
        extra = other - first
        assert not missing, "call {} is missing {}".format(index, sorted(missing))
        assert not extra, "call {} has extra {}".format(index, sorted(extra))


def test_every_switch_source_safe_draft_accepts_is_forwarded():
    """A parameter the function takes and never passes on is inert by
    construction -- the flag would be recorded and do nothing."""
    tree = ast.parse(open(os.path.join(REPO, "infer_atomic.py")).read())
    draft = _function(tree, "_source_safe_draft")
    accepted = {a.arg for a in draft.args.args + draft.args.kwonlyargs}
    # Arguments that belong to the source-safety decision itself, not to the
    # draft it builds; they are consumed here on purpose.
    consumed = {"library", "labels", "feature_dim", "query_retrieval_group_id",
                "no_plan_conditioning", "unsourced_retrieval", "allow_missing"}
    forwarded = set()
    for call in _build_draft_calls(draft):
        forwarded |= {k.arg for k in call.keywords if k.arg}
        for keyword in call.keywords:
            if isinstance(keyword.value, ast.Name):
                forwarded.add(keyword.value.id)
    inert = accepted - forwarded - consumed
    assert not inert, "accepted but never forwarded: {}".format(sorted(inert))
