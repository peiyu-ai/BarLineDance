"""The ``feet`` anchor group: ankles and toes, and NOT the hips and knees.

Why the group exists is measured, not stylistic.  On the aligned base the
``legs`` group put hips, knees, ankles and toes on one timeline, so the hips
settled exactly when the feet did: per-part settle moved feet 0.0339 -> 0.0951
(ground truth 0.0943) but hips_knees 0.0093 -> 0.1731 (ground truth 0.0649),
inverting the feet > hips_knees ordering that is the dancer's shape.  These
tests pin the joint membership, because the defect would be silent: a group with
the wrong joints still runs, still writes a manifest, and still prints a row.
"""
import ast
import pathlib

import infer_atomic
from tools.score_beat_phase_profile import PARTS


def anchor_columns(joints):
    return {infer_atomic.CONTACT_CHANNELS + infer_atomic.ROOT_POSITION_DIMS
            + 6 * joint + k for joint in joints for k in range(6)}


def test_the_warped_joints_are_the_judged_joints():
    """THE ONE THAT MATTERS.  The group that is warped and the column that
    judges it must be the same joints, or the experiment moves one set of
    joints and reads another."""
    assert tuple(infer_atomic._FOOT_JOINTS) == tuple(PARTS["feet"])


def test_hips_and_knees_are_not_in_the_feet_group():
    hips_knees = anchor_columns(PARTS["hips_knees"])
    feet = anchor_columns(infer_atomic._FOOT_JOINTS)
    assert not (feet & hips_knees)


def test_the_feet_group_is_strictly_inside_the_legs_group():
    """It is a narrowing of ``legs``, so anything it warps ``legs`` warped too;
    if that stopped holding, the two options would no longer be comparable as a
    dose of the same idea."""
    legs = set(infer_atomic.LIMB_DIMS["left_leg"]) | set(infer_atomic.LIMB_DIMS["right_leg"])
    feet = anchor_columns(infer_atomic._FOOT_JOINTS)
    assert feet < legs


def test_contacts_and_root_travel_with_the_feet():
    """Splitting the contacts and the root from the feet invented a slide once
    already: foot skate 0.420 -> 0.588 against ground truth's 0.295.  The source
    is read rather than the behaviour simulated, because reaching this branch
    needs a library, a planner and a beat grid."""
    source = pathlib.Path(infer_atomic.__file__).read_text()
    branch = source.split('elif beat_anchor_per_limb == "feet":', 1)[1]
    branch = branch.split('elif beat_anchor_per_limb == "legs":', 1)[0]
    assert "lower = tuple(range(0, ROOT_POSITION_START" in branch
    assert "feet_columns + lower" in branch


def test_feet_is_an_accepted_choice_and_still_needs_an_anchor():
    source = pathlib.Path(infer_atomic.__file__).read_text()
    tree = ast.parse(source)
    choices = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg != "choices":
                continue
            args = [a for a in node.args if isinstance(a, ast.Constant)]
            if any(a.value == "--draft-beat-anchor-per-limb" for a in args):
                choices = [e.value for e in keyword.value.elts]
    assert choices is not None, "the flag lost its choices list"
    assert "feet" in choices and "legs" in choices
    # the guard that refuses per-limb without an anchor must still be there, or
    # a manifest would record per-limb for a run that did no warping at all
    assert "if draft_beat_anchor_per_limb and not draft_beat_anchor:" in source


def test_the_feet_branch_warps_one_group_and_leaves_the_body_alone():
    """The bug this catches was live for one sweep: the branch appended an
    "everything else" group, so the whole body was warped onto the same beats
    and every part rose at once -- hips_knees 0.0093 -> 0.1984, shoulders
    0.1201 -> 0.2144, part spread 0.608 -> 0.219.  Warping only the ankles
    cannot move the shoulders; that impossible reading is what exposed it.

    The shape is pinned against the ``legs`` branch rather than spelled out,
    so the two options cannot drift into different semantics.
    """
    source = pathlib.Path(infer_atomic.__file__).read_text()
    feet = source.split('elif beat_anchor_per_limb == "feet":', 1)[1]
    feet = feet.split('elif beat_anchor_per_limb == "legs":', 1)[0]
    legs = source.split('elif beat_anchor_per_limb == "legs":', 1)[1]
    legs = legs.split("else:", 1)[0]
    for name, branch in (("feet", feet), ("legs", legs)):
        assigns = [line for line in branch.splitlines()
                   if line.strip().startswith("groups")]
        appends = [line for line in assigns if ".append(" in line]
        assert not appends, (
            "{} appends a second group; the rest of the body must stay "
            "unwarped".format(name))
