"""Never level the same corpus twice.

WHY.  The operator, 2026-09-12, asking for the fix at the source: "从源头解决z
值不对齐的问题,避免悬空的语料漏到后面给completion 和 后处理,后处理相关的兜底
同步避免2次拉平带来的副作用".

The corpus disagrees about where the ground is -- each recording's floor (5th
percentile of its lowest foot, in metres) has median 0.341, p5/p95 0.306/0.396,
full range 0.264 to 0.469, with 21 of 295 recordings more than 5 cm from the
median.  ``tools/level_release_floors.py`` removes that at the source.  After it
has, two downstream repairs would fire a second time:

* ``--draft-floor-normalize`` subtracts each prototype's recording floor at
  retrieval, which on a levelled release is already zero.  REFUSED.
* ``--floor-anchor`` shifts the finished clip's 5th-percentile foot onto a
  target.  Still wanted -- the renderer stands every arm on the REFERENCE clip's
  floor and those vary by 20 cm -- but its target has to agree with what the
  corpus was levelled to, or it undoes the levelling one clip at a time.

POSITIVE CONTROL is ``test_a_levelled_release_is_recognised``: every refusal
below is vacuous if the state is always read as "not levelled".
"""
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infer_atomic import release_floor_state

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _release(tmpdir, payload=None):
    if payload is not None:
        with open(os.path.join(tmpdir, "build.json"), "w") as handle:
            json.dump(payload, handle)
    return tmpdir


def test_a_levelled_release_is_recognised():
    """POSITIVE CONTROL for every refusal in this file."""
    with tempfile.TemporaryDirectory() as d:
        levelled, reference = release_floor_state(
            _release(d, {"floor_levelled": True, "floor_reference_m": 0.341}))
    assert levelled is True
    assert reference == 0.341


def test_a_release_without_a_build_file_predates_this():
    """Every artifact made before the levelling must reproduce unchanged."""
    with tempfile.TemporaryDirectory() as d:
        assert release_floor_state(_release(d)) == (False, None)


def test_an_unlevelled_release_is_not_claimed():
    with tempfile.TemporaryDirectory() as d:
        assert release_floor_state(_release(d, {"materializer_version": "v1"})) \
            == (False, None)


def test_a_corrupt_build_file_reads_as_not_levelled():
    """Fail SAFE here rather than closed: an unreadable build.json must not
    make a normal run refuse, but it must never assert levelling either."""
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "build.json"), "w") as handle:
            handle.write("{not json")
        assert release_floor_state(d) == (False, None)


def _run(extra, data_root):
    return subprocess.run(
        [sys.executable, os.path.join(REPO, "infer_atomic.py"),
         "--audio-dir", "/x", "--ingest-root", "/x", "--planner-checkpoint", "/x",
         "--completion-checkpoint", "/x", "--output-dir", "/x",
         "--data-root", data_root] + extra,
        capture_output=True, text=True)


def test_library_levelling_is_refused_on_a_levelled_release():
    with tempfile.TemporaryDirectory() as d:
        _release(d, {"floor_levelled": True, "floor_reference_m": 0.341})
        done = _run(["--draft-floor-normalize", "/x.json",
                     "--draft-root-continuity", "xy"], d)
    assert done.returncode != 0
    assert "already levelled" in (done.stderr + done.stdout)


def test_a_disagreeing_floor_anchor_is_refused():
    with tempfile.TemporaryDirectory() as d:
        _release(d, {"floor_levelled": True, "floor_reference_m": 0.341})
        done = _run(["--floor-anchor", "0.60"], d)
    assert done.returncode != 0
    assert "disagrees" in (done.stderr + done.stdout)


def test_an_agreeing_floor_anchor_is_allowed():
    """The anchor is still needed: the renderer stands every arm on the
    reference CLIP's floor, and those run 0.264 to 0.469 m."""
    with tempfile.TemporaryDirectory() as d:
        _release(d, {"floor_levelled": True, "floor_reference_m": 0.341})
        done = _run(["--floor-anchor", "0.3396"], d)
    assert "disagrees" not in (done.stderr + done.stdout)
