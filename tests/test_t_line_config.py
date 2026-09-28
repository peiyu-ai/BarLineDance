"""The T line's current generation config must be T2, whole, and unable to mix versions.

Why this file exists (2026-09-22).  The shipped T1 argv lived only in another
session's /tmp scratchpad; nothing named "the current config", and every entry
point took the library and the two checkpoints as three independent flags.
After the T2 upgrade (+24 clips in the library, retrained planner/completion)
the operator's instruction was that future generation must use the new library
and the new models.  These tests pin that: ``configs/t_line/current.argv``
resolves to T2 in every generation-relevant flag, T1 stays reachable only by
name, the two differ in nothing but the swapped artifacts, and infer_atomic
refuses a checkpoint trained on another release than the library's.
"""
import argparse
import json
import pathlib
import sys
import tempfile

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import infer_atomic  # noqa: E402

CONFIGS = REPO / "configs" / "t_line"
T1_TOKENS = ("scratch/txy_t/", "runs/planner_aligned_k8/", "completion_txy_t_seam4_s20260905",
             "txy_t_normalized/", "txy_t_gt_eval", "wild_ingest_v1")
SWAPPED = {"data_root", "planner_checkpoint", "completion_checkpoint", "draft_floor_normalize",
           "retrieval_yaw_steps", "retrieval_speed_spikes", "draft_unit_floor", "floor_anchor",
           "ingest_root"}
PER_INVOCATION = ["--audio-dir", "/tmp/q", "--sequence-list", "/tmp/l", "--output-dir", "/tmp/o"]


def parse(config, monkeypatch):
    monkeypatch.chdir(REPO)
    monkeypatch.setattr(sys, "argv", ["infer_atomic.py", "@" + str(config.relative_to(REPO))]
                        + PER_INVOCATION)
    return infer_atomic.parse_args()


def test_current_resolves_to_t2_in_every_generation_flag(monkeypatch):
    current = vars(parse(CONFIGS / "current.argv", monkeypatch))
    t2 = vars(parse(CONFIGS / "t2.argv", monkeypatch))
    assert current == t2
    assert "scratch/txy_t2/release_aligned_k8_20260922" in current["data_root"]
    assert "planner_aligned_k8_t2_" in current["planner_checkpoint"]
    assert "completion_txy_t2_" in current["completion_checkpoint"]
    assert "wild_ingest_txy_t2_audiofix" in current["ingest_root"]
    for key, value in current.items():
        if isinstance(value, str):
            assert not any(tok in value for tok in T1_TOKENS), (key, value)


def test_t1_and_t2_differ_only_in_the_swapped_artifacts(monkeypatch):
    t1 = vars(parse(CONFIGS / "t1_fix7.argv", monkeypatch))
    t2 = vars(parse(CONFIGS / "t2.argv", monkeypatch))
    differing = {k for k in set(t1) | set(t2) if t1.get(k) != t2.get(k)}
    assert differing == SWAPPED, differing ^ SWAPPED


def test_t1_fallback_is_the_shipped_fix7_flag_set(monkeypatch):
    preserved = REPO / "runs/txy_t2_20260922/preserved_from_516f17be/argv_fix7.txt"
    if not preserved.is_file():
        pytest.skip("preserved fix7 argv not on this machine")
    t1 = vars(parse(CONFIGS / "t1_fix7.argv", monkeypatch))
    monkeypatch.setattr(sys, "argv", ["infer_atomic.py"]
                        + [t for t in preserved.read_text().split("\n") if t] + PER_INVOCATION)
    shipped = vars(infer_atomic.parse_args())
    # Per-invocation flags are removed from the fallback on purpose (the shipped
    # argv named fix7's own output dirs); every other flag must be identical --
    # including ingest_root, which the fallback writes out as the shipped default.
    per_invocation = {"dump_draft_dir", "output_dir", "sequence_list", "audio_dir"}
    differing = {k for k in set(t1) | set(shipped) if t1.get(k) != shipped.get(k)} - per_invocation
    assert differing == set(), differing


def test_later_flags_override_the_config(monkeypatch):
    monkeypatch.chdir(REPO)
    monkeypatch.setattr(sys, "argv", ["infer_atomic.py", "@configs/t_line/current.argv"]
                        + PER_INVOCATION + ["--seed", "7"])
    assert infer_atomic.parse_args().seed == 7


def test_generation_config_records_every_file_with_its_sha(monkeypatch):
    monkeypatch.chdir(REPO)
    config = infer_atomic._generation_config(["@configs/t_line/current.argv", "--seed", "7"])
    paths = [f["path"] for f in config["files"]]
    assert paths == ["configs/t_line/current.argv", "configs/t_line/t2.argv"]
    assert all(len(f["sha256"]) == 64 for f in config["files"])


def _library(tmp, release, bar_release):
    root = pathlib.Path(tmp) / "lib"
    root.mkdir()
    (root / "build.json").write_text(json.dumps({"derived_from": {
        "release": release, "bar_release": bar_release}}))
    return root


def test_a_checkpoint_from_another_release_is_refused_unless_allowed():
    with tempfile.TemporaryDirectory() as tmp:
        lib = _library(tmp, "/r/t2/release_v3", "/r/t2/bar")
        ok = {}
        infer_atomic._check_release_binding(ok, "planner", argparse.Namespace(data_root="/r/t2/bar"),
                                            lib, "bar_release", False)
        assert ok["planner"]["status"] == "match"
        with pytest.raises(SystemExit, match="trained on /r/t1/bar"):
            infer_atomic._check_release_binding({}, "planner", argparse.Namespace(data_root="/r/t1/bar"),
                                                lib, "bar_release", False)
        allowed = {}
        infer_atomic._check_release_binding(allowed, "completion",
                                            argparse.Namespace(data_root="/r/t1/release_v3"),
                                            lib, "release", True)
        assert allowed["completion"]["status"] == "cross_release_allowed"


def test_a_library_without_derived_from_is_recorded_not_refused():
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        (root / "build.json").write_text(json.dumps({}))
        binding = {}
        infer_atomic._check_release_binding(binding, "completion", argparse.Namespace(data_root="/x"),
                                            root, "release", False)
        assert binding["completion"]["status"] == "not_applicable"


def test_the_t2_artifacts_bind_to_the_t2_library(monkeypatch):
    """On this machine only: the configured checkpoints were trained on what the library names."""
    t2 = parse(CONFIGS / "t2.argv", monkeypatch)
    build = pathlib.Path(t2.data_root) / "build.json"
    if not build.is_file():
        pytest.skip("T2 library not on this machine")
    import torch
    derived = json.loads(build.read_text())["derived_from"]
    for path, key in ((t2.planner_checkpoint, "bar_release"), (t2.completion_checkpoint, "release")):
        if not pathlib.Path(path).is_file():
            pytest.skip("{} not trained yet".format(path))
        args = torch.load(path, map_location="cpu", weights_only=False)["args"]
        trained_on = args.data_root if not isinstance(args, dict) else args["data_root"]
        assert pathlib.Path(trained_on).resolve() == pathlib.Path(derived[key]).resolve(), (path, key)


def test_t2_fix8_variant_differs_from_t2_only_by_the_fix8_deltas(monkeypatch):
    """fix8 (DEFECTS §90.5) = fix7's flags minus stagger/inpaint, blend 12, after, draft-only."""
    t2 = vars(parse(CONFIGS / "t2.argv", monkeypatch))
    fix8 = vars(parse(CONFIGS / "t2_fix8.argv", monkeypatch))
    differing = {k for k in set(t2) | set(fix8) if t2.get(k) != fix8.get(k)}
    assert differing == {"draft_seam_stagger", "completion_inpaint_seam_width", "draft_seam_blend",
                         "seam_transition", "draft_only"}, differing
    assert fix8["draft_seam_blend"] == 12 and fix8["seam_transition"] == "after" and fix8["draft_only"]
    assert fix8["data_root"] == t2["data_root"] and fix8["planner_checkpoint"] == t2["planner_checkpoint"]


def test_t2_fix8_turns_differs_from_t2_fix8_only_by_the_facing_pull(monkeypatch):
    """DEFECTS §91: the slow face_camera pull aborts sustained turns; strength 0 keeps only the
    rigid mean-facing rotation.  Nothing else may ride along, or the §91 comparison is void."""
    fix8 = vars(parse(CONFIGS / "t2_fix8.argv", monkeypatch))
    turns = vars(parse(CONFIGS / "t2_fix8_turns.argv", monkeypatch))
    differing = {k for k in set(fix8) | set(turns) if fix8.get(k) != turns.get(k)}
    assert differing == {"face_camera_strength"}, differing
    # 0, not None: None skips face_camera entirely (no rigid rotation either).
    assert turns["face_camera_strength"] == 0.0 and fix8["face_camera_strength"] == 0.7


def test_t2_fix8_cont_and_full_add_only_their_own_flags(monkeypatch):
    """DEFECTS §92: continuation is t2_fix8_turns + the four continuation flags; the "打满" recipe is
    continuation + prefer-full (stops) + hold-by-music.  Nothing else may ride along."""
    turns = vars(parse(CONFIGS / "t2_fix8_turns.argv", monkeypatch))
    cont = vars(parse(CONFIGS / "t2_fix8_cont.argv", monkeypatch))
    full = vars(parse(CONFIGS / "t2_fix8_full.argv", monkeypatch))
    assert {k for k in set(turns) | set(cont) if turns.get(k) != cont.get(k)} == {
        "draft_continue_source", "draft_continue_lookahead", "draft_continue_any_label",
        "draft_continue_max_run"}
    assert cont["draft_continue_any_label"] and cont["draft_continue_max_run"] == 4
    assert {k for k in set(cont) | set(full) if cont.get(k) != full.get(k)} == {
        "draft_prefer_full", "draft_prefer_full_mode", "draft_hold_by_music"}
    assert full["draft_prefer_full"] == 0.25 and full["draft_prefer_full_mode"] == "stops"


def test_t2_fix8_beat_is_cont_plus_the_downbeat_stop_and_the_rhythm_scorer(monkeypatch):
    """DEFECTS §93: BEAT = t2_fix8_cont + the downbeat stop on continued bar lines + the learned rhythm
    scorer keeping the best-aligned quarter.  The phase shift (the "hit" arm) must NOT ride along: it is the
    one that buys beat gain with more post-seam reversal, and the two were compared as separate arms."""
    cont = vars(parse(CONFIGS / "t2_fix8_cont.argv", monkeypatch))
    beat = vars(parse(CONFIGS / "t2_fix8_beat.argv", monkeypatch))
    assert {k for k in set(cont) | set(beat) if cont.get(k) != beat.get(k)} == {
        "draft_continue_stop", "draft_rhythm_scorer", "draft_rhythm_keep"}
    assert beat["draft_continue_stop"] and beat["draft_rhythm_keep"] == 0.25
    assert beat["draft_rhythm_shift"] == 0
    assert beat["draft_rhythm_scorer"].endswith("rhythm_scorer_v5_rhythm3_w128.pt")
