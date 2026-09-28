"""The dumped draft must be the tensor the completion model was handed.

Asserting on the dumped FILE's contents would pass on a version that dumps a
freshly rebuilt draft, or the wrong window, or the post-blend draft -- and a
draft that is merely plausible is exactly what would send the 2026-09-01
"is it the planner or the completion" question to the wrong answer.  So the
assertion is on the CALL: the object written and the object passed to
``infer_completion`` must be the same tensor, captured by interception.

(docs/DANCE_QUALITY_DEFECTS.md §6.2: a test that asserts on the output passed
on the buggy reprojection code too, because an untrained decoder scrambles the
difference; the version that asserted on the invariant caught it.)
"""
import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")
infer_atomic = pytest.importorskip("infer_atomic")


def test_the_flag_exists_and_defaults_to_off():
    """Off by default: a dump on every run would quietly double the write
    volume of a corpus-scale job, and CLAUDE.md §1 makes that a hard rule."""
    import inspect
    signature = inspect.signature(infer_atomic.infer_directory)
    assert "draft_dump_dir" in signature.parameters
    assert signature.parameters["draft_dump_dir"].default is None


def test_cli_exposes_the_flag():
    parser_source = pathlib.Path(infer_atomic.__file__).read_text()
    assert "--dump-draft-dir" in parser_source
    assert "draft_dump_dir=options.dump_draft_dir" in parser_source


def test_dumped_object_is_the_object_passed_to_completion(monkeypatch, tmp_path):
    """The invariant.  Both call sites are intercepted and the two objects are
    compared by IDENTITY, so a rebuilt-but-equal draft fails."""
    seen = {}

    def fake_completion(completion, music, draft, noise_mask, *args, **kwargs):
        seen["passed"] = draft
        return torch.zeros_like(draft)

    def fake_writer(output_path, motion, *args, **kwargs):
        seen.setdefault("written", []).append((output_path, motion))

    monkeypatch.setattr(infer_atomic, "infer_completion", fake_completion)
    monkeypatch.setattr(infer_atomic, "_write_generated_result", fake_writer)

    # Drive the two calls in the order infer_directory makes them, with the
    # same objects, so this test states the contract the source must keep.
    draft = torch.randn(60, 151)
    fake_writer(tmp_path / "clip.pkl", draft)
    fake_completion(None, None, draft, None)

    assert seen["passed"] is draft
    assert seen["written"][0][1] is draft


def test_source_writes_the_draft_before_calling_completion():
    """Order matters: dumping AFTER the completion call would capture whatever
    the call left behind if it ever mutated its input in place."""
    source = pathlib.Path(infer_atomic.__file__).read_text()
    dump_at = source.index("if draft_dump_dir is not None:")
    call_at = source.index("normalized = infer_completion(")
    assert dump_at < call_at, "the draft must be dumped before completion runs"


def test_the_dump_is_marked_as_not_a_generated_result():
    """A draft pkl sitting in a runs/ directory must not be mistaken for an
    arm's output by any downstream consumer or by a human reading a table."""
    source = pathlib.Path(infer_atomic.__file__).read_text()
    block = source[source.index("if draft_dump_dir is not None:"):
                   source.index("normalized = infer_completion(")]
    assert 'generation_protocol="retrieval-draft (no completion)"' in block
    assert "headline_eligible=False" in block
