"""A candidate the descriptor cannot read must not take the whole arm down.

FOUND BY RUNNING A PATH THAT HAD NEVER BEEN RUN.  ``--retrieval-rule learned``
existed in the repository with a trainer, a model and its own tests, and had
never produced an arm.  The first attempt died at the 15th clip of 20 with
``ValueError: segment must have at least 2 frames, got 1`` -- the retrieval pool
contains one-frame spans, ``describe_segment`` needs a difference to read travel
and contact changes from, and nothing had ever asked it to describe one.

The fix is a fallback and a COUNT, not a silent skip: a pool that is mostly
unreadable would otherwise pass as a working selector.
"""
import pathlib
import re


SOURCE = pathlib.Path("infer_atomic.py").read_text()


def descriptor_body():
    body = SOURCE.split("def _descriptor(self, candidate):", 1)[1]
    return body.split("def _query_context", 1)[0]


def selector_branch():
    body = SOURCE.split('if self.retrieval_rule == "learned"', 1)[-1]
    return body.split('elif self.retrieval_rule == "phase"', 1)[0]


def test_a_short_span_returns_none_instead_of_raising():
    body = descriptor_body()
    assert "if len(values) < 2:" in body
    assert "return None" in body


def test_the_unreadable_ones_are_counted():
    """Silent skipping would let a pool that cannot be scored look like a
    selector that is choosing."""
    assert "self.undescribable_candidates += 1" in descriptor_body()
    assert "self.undescribable_candidates = 0" in SOURCE


def test_the_scoring_site_drops_them_before_scoring():
    branch = selector_branch()
    assert "if d is not None" in branch


def test_too_few_readable_candidates_falls_back_to_the_shipped_rule():
    """Falling back is honest; scoring one candidate and calling it a choice is
    not."""
    branch = selector_branch()
    assert "if len(described) < 2:" in branch
    assert "self._duration_pick(" in branch
    assert "selector_fallbacks" in branch


def test_the_fallback_and_the_real_call_are_counted_separately():
    branch = selector_branch()
    assert "selector_calls" in branch and "selector_fallbacks" in branch
    calls = branch.index("self.selector_calls")
    fallbacks = branch.index("self.selector_fallbacks")
    assert fallbacks < calls, (
        "the fallback must be counted in its own branch, not after the real "
        "call, or one number would cover both")
