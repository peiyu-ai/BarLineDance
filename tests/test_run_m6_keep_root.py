"""The shipping driver must actually pass --completion-keep-root.

This test exists because of a defect this repository has already paid for
twice: --draft-bar-prototypes was recorded in the manifest and honoured at
batch size 1 while run_m6_wild.sh's default batch size took a path that ignored
it, so every measurement of the switch was taken somewhere the shipped run
never went.  A driver-level assertion is the cheapest place to catch the
inverse -- a flag that exists in infer_atomic.py and never reaches it.
"""
import pathlib
import re

import pytest

DRIVER = pathlib.Path(__file__).resolve().parents[1] / "tools/run_m6_wild.sh"
INFER = pathlib.Path(__file__).resolve().parents[1] / "infer_atomic.py"


@pytest.fixture(scope="module")
def driver():
    return DRIVER.read_text()


def test_the_flag_defaults_on(driver):
    assert 'COMPLETION_KEEP_ROOT="${COMPLETION_KEEP_ROOT:-1}"' in driver


def test_the_flag_reaches_the_inference_call(driver):
    call = driver.split("python3 infer_atomic.py")[1].split("done")[0]
    assert "$KEEP_ROOT_FLAG" in call, \
        "run_m6_wild.sh builds KEEP_ROOT_FLAG but never passes it"


def test_the_banner_says_which_way_it_ran(driver):
    """A run whose log does not say how it was configured cannot be compared
    against another run, which is the only way any of these are judged."""
    assert "keeproot=$COMPLETION_KEEP_ROOT" in driver


def test_setting_it_to_zero_produces_an_empty_flag(driver):
    assert re.search(r'\[ "\$COMPLETION_KEEP_ROOT" = "1" \] && KEEP_ROOT_FLAG=', driver)
    assert 'KEEP_ROOT_FLAG=""' in driver


def test_infer_atomic_accepts_the_name_the_driver_passes(driver):
    """The two halves are edited in different files; assert they still agree."""
    name = re.search(r'KEEP_ROOT_FLAG="(--[a-z-]+)"', driver).group(1)
    assert '"{}"'.format(name) in INFER.read_text()
