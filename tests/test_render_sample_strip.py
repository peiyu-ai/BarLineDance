"""Panel naming for the sample-strip renderer.

The length bound lives in ``test_render_sample_strip_bounds.py``; this file
covers the OTHER defect found on 2026-09-03, which was silent rather than loud.

Staging filenames were built from ``title.split(" ")[0][:6]``.  With arms named
"epoch12 fix" and "epoch16 fix" both panels became ``..._stick_epoch1.mp4``, so
the second render overwrote the first and the finished strip showed one arm
twice while labelling them differently -- a comparison that cannot be wrong
because both sides are the same motion.  It also made the file the operator
opened, ``_stick_epoch1.mp4``, unreadable as to which arm it held.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.render_sample_strip import slug


def panel_name(stem, index, title):
    """The name the renderer builds, kept in one place for the tests."""
    return "{}_stick_{}_{}.mp4".format(stem, index, slug(title))


def test_a_slug_keeps_the_whole_title_readable():
    assert slug("epoch12 fix · pose") == "epoch12-fix-pose"
    assert slug("ground truth · pose") == "ground-truth-pose"


def test_two_arms_sharing_a_six_character_prefix_do_not_collide():
    names = {panel_name("clip", i, t) for i, t in
             enumerate(["epoch12 fix · pose", "epoch16 fix · pose"])}
    assert len(names) == 2
    # The rule this replaced really did collide -- that is why the index is there.
    assert len({"epoch12 fix".split(" ")[0][:6],
                "epoch16 fix".split(" ")[0][:6]}) == 1


def test_identically_titled_panels_still_get_distinct_files():
    # The index, not the title, is what guarantees uniqueness.
    names = {panel_name("clip", i, "same name") for i in range(3)}
    assert len(names) == 3


def test_a_slug_cannot_escape_the_staging_directory():
    assert "/" not in slug("../../etc/passwd")
    assert ".." not in slug("../../etc/passwd").strip("-")


def test_an_empty_title_still_yields_a_usable_name():
    assert slug("") == "panel"
    assert slug("···") == "panel"


def test_slugs_do_not_run_unboundedly_long():
    assert len(slug("x" * 500)) <= 40
