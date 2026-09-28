"""The panel-label font gate, in both directions.

WHAT THIS EXISTS FOR.  The gate used to be an allowlist of CJK font names with
a comment saying the winner's cmap "was CHECKED".  It had been -- for CJK.  On
this host the winner is ``Droid Sans Fallback``, which contains no Latin at all,
not even lowercase "g", and it was placed FIRST in the family list.  The moment
the operator asked for English labels, every title in every rendered panel came
out as boxes and the gate reported success, because it never asked about Latin.

A criterion that can only fail in one direction is the repository's recurring
defect (CLAUDE.md 2.1).  These tests pin both directions.
"""

import pathlib
import sys

import matplotlib
import pytest

matplotlib.use("Agg")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt

import tools.render_dance_video as R


def test_a_latin_capable_font_comes_first():
    """Ordering is the fix: matplotlib falls back per glyph down this list.

    Asserted on ``_font_stack()`` rather than on ``plt.rcParams``: rcParams is
    process-global mutable state and any other test module that sets a font
    changes it.  The first version of this test read rcParams, passed alone, and
    failed in the full suite -- i.e. it was testing whichever test ran last, not
    this module's decision.
    """
    stack = R._font_stack()
    assert stack, "no font stack configured"
    assert R._covered(stack[0], "abcdefg") == set("abcdefg")


def test_the_stack_covers_every_latin_label_character():
    stack = R._font_stack()
    covered = set().union(*[R._covered(name, R._LATIN_SAMPLE) for name in stack])
    missing = set(R._LATIN_SAMPLE) - covered
    assert not missing, "labels would render as boxes: {}".format(sorted(missing))


def test_import_actually_installs_the_stack():
    """The module must APPLY its decision, not merely be able to compute it."""
    import importlib

    importlib.reload(R)
    assert list(plt.rcParams["font.sans-serif"]) == list(R._font_stack())


def test_the_font_that_broke_it_really_lacks_latin():
    """Guards the diagnosis, not just the fix.

    If a future host ships a Droid Sans Fallback that DOES carry Latin, this
    test failing is the signal that the comment above needs rewriting -- rather
    than the explanation quietly becoming folklore.
    """
    installed = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    if "Droid Sans Fallback" not in installed:
        pytest.skip("host does not have the font this was diagnosed on")
    assert R._covered("Droid Sans Fallback", "abcdefg") == set()


def test_the_gate_fails_when_no_font_covers_latin(monkeypatch):
    """The gate must be able to fail, and fail for the RIGHT reason."""
    monkeypatch.setattr(R, "_covered", lambda family, sample: set())
    with pytest.raises(SystemExit) as failure:
        R._font_stack()
    assert "Latin" in str(failure.value)


def test_the_gate_fails_when_no_font_is_installed_at_all(monkeypatch):
    class Empty:
        ttflist = []

    monkeypatch.setattr(matplotlib.font_manager, "fontManager", Empty())
    with pytest.raises(SystemExit) as failure:
        R._font_stack()
    assert "no usable font" in str(failure.value)


def test_a_cjk_font_is_still_reachable_for_chinese_labels():
    """English is the default now, but Chinese must not silently regress."""
    stack = R._font_stack()
    covered = set().union(*[R._covered(name, R._CJK_SAMPLE) for name in stack])
    installed = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    if not ({"Noto Sans CJK JP", "Noto Sans CJK SC", "Noto Serif CJK JP",
             "Droid Sans Fallback"} & installed):
        pytest.skip("host has no CJK font")
    assert covered == set(R._CJK_SAMPLE)


def test_covered_reports_nothing_for_an_unknown_family():
    assert R._covered("No Such Font At All 12345", "abc") == set()
