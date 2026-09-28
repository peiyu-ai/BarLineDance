"""The shipping driver's draft switches: --draft-bar-prototypes and
--draft-recurrence-variety.

WHY A TEST FOR A SHELL VARIABLE.  This flag is not a preference, it is a
correctness fix, and the way it fails is silent-looking: without it two adjacent
bars that draw the same label merge into ONE retrieval unit and a single
prototype is stretched over all of it.  Measured on the 18 T eval clips
(2026-09-05): 9 unfillable units across 8 of 18 clips, the worst asking 312
frames, handed 150, played at 0.481x -- while the pooled stretch median read
1.000 and no KPI moved.  A driver that quietly stopped emitting the flag would
reproduce exactly that, so the emission is asserted rather than assumed.

THE NEGATIVE HALF MATTERS TOO.  DRAFT_BAR_PROTOTYPES=0 must drop the flag, or
there is no way to reproduce a pre-2026-09-05 artifact from its own command
line, which is the rule PLAN_BIAS and PLAN_STRIDE already follow.

AND IT MUST NOT FIRE UNGRIDDED.  infer_atomic only honours the flag alongside
--plan-bar-grid or --plan-bar-tokens (it needs bar lines to cut on), so a driver
that emitted it with PLAN_BAR_GRID=0 would be writing a command line whose flag
does nothing -- the kind of thing that reads as "the fix is on" while it is not.
"""
import os
import subprocess

DRIVER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "tools", "run_m6_wild.sh")


def _flag(**env):
    """Evaluate the driver's flag block alone, without running inference.

    The block is extracted by line range rather than sourced, because sourcing
    the whole driver would run its argument parsing and its disk checks.
    """
    with open(DRIVER) as handle:
        lines = handle.readlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("DRAFT_BAR_PROTOTYPES="))
    end = next(i for i, l in enumerate(lines[start:], start)
               if "BAR_PROTO_FLAG=\"--draft-bar-prototypes\"" in l)
    block = "".join(lines[start:end + 1]) + '\necho "$BAR_PROTO_FLAG"\n'
    environment = dict(os.environ)
    environment.update({k: str(v) for k, v in env.items()})
    out = subprocess.run(["bash", "-c", block], env=environment,
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def test_the_flag_is_on_by_default():
    assert _flag(PLAN_BAR_GRID="1") == "--draft-bar-prototypes"


def test_it_can_be_turned_off_to_reproduce_an_older_artifact():
    assert _flag(PLAN_BAR_GRID="1", DRAFT_BAR_PROTOTYPES="0") == ""


def test_it_is_not_emitted_without_a_bar_grid_to_cut_on():
    """infer_atomic ignores it there, so emitting it would be a false comfort."""
    assert _flag(PLAN_BAR_GRID="0") == ""


def test_the_driver_actually_passes_the_variable_to_inference():
    """The block above could be right while the command line never uses it."""
    with open(DRIVER) as handle:
        text = handle.read()
    invocation = text.split("python3 infer_atomic.py", 1)[1].split("\n\n", 1)[0]
    assert "$BAR_PROTO_FLAG" in invocation
    # and the run banner must name it, so a log says which behaviour produced it
    assert "barproto=$DRAFT_BAR_PROTOTYPES" in text


# ---------------------------------------------------------------------------
# --draft-recurrence-variety
#
# WHY IT IS ASSERTED.  With it off, retrieve()'s cache hands every repeat of a
# label inside a clip the identical tensor, because build_draft passes
# `occurrence if recurrence_variety else 0`.  Measured on the 18 T eval clips
# (2026-09-05): 27.1% of the shipping draft's retrieval units replayed an
# earlier one and 12.6% of ADJACENT pairs were the same motion twice in a row,
# against 0 of 110 for ground truth.  --draft-bar-prototypes makes it worse by
# construction (constant target_length -> colliding cache key), so the two
# switches have to travel together.


def _variety(**env):
    with open(DRIVER) as handle:
        lines = handle.readlines()
    start = next(i for i, l in enumerate(lines)
                 if l.startswith("DRAFT_RECURRENCE_VARIETY="))
    end = next(i for i, l in enumerate(lines[start:], start)
               if 'VARIETY_FLAG="--draft-recurrence-variety"' in l)
    block = "".join(lines[start:end + 1]) + '\necho "$VARIETY_FLAG"\n'
    environment = dict(os.environ)
    environment.update({k: str(v) for k, v in env.items()})
    return subprocess.run(["bash", "-c", block], env=environment,
                          capture_output=True, text=True, check=True).stdout.strip()


def test_variety_is_on_by_default():
    assert _variety() == "--draft-recurrence-variety"


def test_variety_can_be_turned_off_to_reproduce_an_older_artifact():
    assert _variety(DRAFT_RECURRENCE_VARIETY="0") == ""


def test_variety_does_not_depend_on_the_bar_grid():
    """Unlike the bar-prototype split, this one is meaningful ungridded too.

    The cache collision needs only a label to repeat inside a clip, which
    happens with or without a bar grid -- so gating it on PLAN_BAR_GRID would
    leave the ungridded path silently reusing prototypes.
    """
    assert _variety(PLAN_BAR_GRID="0") == "--draft-recurrence-variety"


def test_the_driver_passes_the_variety_flag_and_names_it_in_the_banner():
    with open(DRIVER) as handle:
        text = handle.read()
    invocation = text.split("python3 infer_atomic.py", 1)[1].split("\n\n", 1)[0]
    assert "$VARIETY_FLAG" in invocation
    assert "variety=$DRAFT_RECURRENCE_VARIETY" in text
