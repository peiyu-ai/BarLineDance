"""Tests for the wild ground-truth export that feeds M6's FID.

The export's job is to hand ``extract_aist_features`` a motion directory and an
audio directory, and its correctness is almost entirely about what it refuses.
A wrong-but-plausible export produces features, produces an FID, and the number
is wrong in a way nothing downstream can see -- so every guard here is asserted
to fire rather than assumed.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import pickle
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.export_wild_eval_motion import ExportError, export  # noqa: E402

MOTION_DIM = 151
MUSIC_DIM = 35


def _write_array(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), array)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(tmp_path, count=2, frames=40, split="test", break_frames=False):
    bundle = tmp_path / "bundle"
    rows = []
    rng = np.random.default_rng(7)
    for index in range(count):
        name = "wild_v4:{}:clip000".format(700 + index)
        stem = "sequences/seq{}".format(index)
        # rot6d that is not degenerate: forward kinematics on all-zeros still
        # works, but a real rotation exercises the same path the corpus takes.
        motion = rng.normal(size=(frames, MOTION_DIM)).astype(np.float32)
        music_frames = frames + (3 if break_frames and index == 0 else 0)
        music = rng.normal(size=(music_frames, MUSIC_DIM)).astype(np.float32)
        motion_sha = _write_array(bundle / stem / "motion_151_raw.npy", motion)
        music_sha = _write_array(bundle / stem / "music_35.npy", music)
        rows.append({
            "recording_id": name, "split": split,
            "motion_path": stem + "/motion_151_raw.npy", "motion_sha256": motion_sha,
            "music_path": stem + "/music_35.npy", "music_sha256": music_sha,
        })
    (bundle / "sequences.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return bundle, rows


def test_the_export_writes_full_pose_which_is_the_key_infer_atomic_writes(tmp_path):
    """One branch of ``load_keypoints``, not two.

    ``infer_atomic.decode_motion`` writes ``full_pose`` from
    ``SMPLSkeleton().forward``; the ground truth goes through the same call, so
    both sides of the FID take the identical z-up -> y-up path.  Exporting the
    ground truth through any other joint convention would make FID measure the
    convention.
    """
    bundle, rows = _bundle(tmp_path, count=2, frames=40)
    manifest = export(bundle, "test", tmp_path / "out")
    assert manifest["sequences"] == 2
    for row in rows:
        name = row["recording_id"]
        with open(str(tmp_path / "out" / "motion" / (name + ".pkl")), "rb") as handle:
            payload = pickle.load(handle)
        assert set(payload) == {"full_pose"}
        assert payload["full_pose"].shape == (40, 24, 3)
        assert np.isfinite(payload["full_pose"]).all()


def test_music_is_copied_verbatim_never_re_derived(tmp_path):
    """The stored array *is* the corpus's music identity.

    Re-extracting or resampling it would put BAS's beats on a second extractor
    run -- 0.36 correlation on the one corpus where both exist -- which splits
    the corpus along the axis the planner reads.
    """
    bundle, rows = _bundle(tmp_path, count=1, frames=30)
    export(bundle, "test", tmp_path / "out")
    source = np.load(str(bundle / rows[0]["music_path"]))
    exported = np.load(str(tmp_path / "out" / "audio" / (rows[0]["recording_id"] + ".npy")))
    assert exported.shape == source.shape
    np.testing.assert_array_equal(exported, source)


def test_a_split_with_no_rows_is_refused(tmp_path):
    bundle, _ = _bundle(tmp_path, split="train")
    with pytest.raises(ExportError, match="names no sequence"):
        export(bundle, "test", tmp_path / "out")


def test_a_zero_byte_array_is_refused_rather_than_loaded(tmp_path):
    """This repo's canonical silent failure: the file exists and is empty.

    A quota-exhausted write creates the file before it fails, so "the product is
    there" is not evidence (CLAUDE.md 1).
    """
    bundle, rows = _bundle(tmp_path, count=1)
    (bundle / rows[0]["motion_path"]).write_bytes(b"")
    with pytest.raises(ExportError, match="0 bytes"):
        export(bundle, "test", tmp_path / "out")


def test_an_array_that_is_not_the_bytes_the_manifest_names_is_refused(tmp_path):
    bundle, rows = _bundle(tmp_path, count=1, frames=20)
    np.save(str(bundle / rows[0]["motion_path"]),
            np.zeros((20, MOTION_DIM), dtype=np.float32))
    with pytest.raises(ExportError, match="not the bytes the manifest names"):
        export(bundle, "test", tmp_path / "out")


def test_motion_and_music_disagreeing_on_length_is_refused(tmp_path):
    """They are frame-aligned by construction -- the audio was extracted against
    the converted 3D's own frame_ids -- so a disagreement means two runs."""
    bundle, _ = _bundle(tmp_path, count=2, frames=25, break_frames=True)
    with pytest.raises(ExportError, match="frame-aligned by construction"):
        export(bundle, "test", tmp_path / "out")


def test_the_manifest_records_that_wild_figures_are_not_aist_figures(tmp_path):
    bundle, _ = _bundle(tmp_path, count=1)
    manifest = export(bundle, "test", tmp_path / "out")
    assert "AIST" in manifest["not_comparable_to"]
    assert manifest["frames"]["total"] > 0


# --- the training-bundle cross-check -----------------------------------------
#
# Every other check in the exporter compares the eval bundle against its OWN
# manifest, and a bundle built from a stale generation of the corpus passes all
# of them: its hashes verify and its motion and music agree on frame count.
# What it cannot pass is a comparison against the bundle the models trained on.

def _bundle_with(tmp, rows):
    path = pathlib.Path(tmp)
    path.mkdir(parents=True, exist_ok=True)
    (path / "sequences.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def _row(name, motion="aaa", music="bbb", frames=100, split="test"):
    return {"recording_id": name, "motion_sha256": motion, "music_sha256": music,
            "frame_count": frames, "split": split}


def test_the_cross_check_refuses_bytes_the_training_bundle_does_not_name():
    import pytest
    from tools.export_wild_eval_motion import ExportError, cross_check_training_bundle
    with tempfile.TemporaryDirectory() as raw:
        training = _bundle_with(pathlib.Path(raw) / "train", [_row("wild_v4:1:clip000")])
        rows = [_row("wild_v4:1:clip000", motion="STALE")]
        with pytest.raises(ExportError, match="do not hold the bytes"):
            cross_check_training_bundle(rows, training, "refuse")


def test_a_difference_at_equal_frame_count_is_counted_on_its_own():
    # 113 of the 292 differing test sequences measured 2026-08-24 differ at the
    # same frame count.  A length comparison reports those as clean, which is
    # the R0 lesson (3,929 stale by frame count against 3,991 by hash).
    from tools.export_wild_eval_motion import cross_check_training_bundle
    with tempfile.TemporaryDirectory() as raw:
        training = _bundle_with(pathlib.Path(raw) / "train",
                                [_row("wild_v4:1:clip000", frames=409),
                                 _row("wild_v4:2:clip000", frames=409)])
        rows = [_row("wild_v4:1:clip000", motion="STALE", frames=409),   # invisible to length
                _row("wild_v4:2:clip000", motion="STALE", frames=545)]   # visible to length
        report = cross_check_training_bundle(rows, training, "warn")
        assert report["motion_differs"] == 2
        assert report["motion_differs_at_equal_length"] == 1


def test_a_sequence_the_training_bundle_never_had_is_its_own_bucket():
    from tools.export_wild_eval_motion import cross_check_training_bundle
    with tempfile.TemporaryDirectory() as raw:
        training = _bundle_with(pathlib.Path(raw) / "train", [_row("wild_v4:1:clip000")])
        report = cross_check_training_bundle(
            [_row("wild_v4:9:clip000")], training, "warn")
        assert report["absent_in_training"] == 1
        assert report["motion_differs"] == 0 and report["match"] == 0


def test_no_training_bundle_reads_as_unchecked_rather_than_as_agreement():
    from tools.export_wild_eval_motion import cross_check_training_bundle
    report = cross_check_training_bundle([_row("wild_v4:1:clip000")], None, "refuse")
    assert report["checked"] is False
    assert report["match"] == 0


def test_matching_bytes_pass_and_are_counted():
    from tools.export_wild_eval_motion import cross_check_training_bundle
    with tempfile.TemporaryDirectory() as raw:
        training = _bundle_with(pathlib.Path(raw) / "train", [_row("wild_v4:1:clip000")])
        report = cross_check_training_bundle(
            [_row("wild_v4:1:clip000")], training, "refuse")
        assert report["match"] == 1 and not report["differing"]


def test_music_only_disagreement_is_labelled_as_music():
    from tools.export_wild_eval_motion import cross_check_training_bundle
    with tempfile.TemporaryDirectory() as raw:
        training = _bundle_with(pathlib.Path(raw) / "train", [_row("wild_v4:1:clip000")])
        report = cross_check_training_bundle(
            [_row("wild_v4:1:clip000", music="OTHER")], training, "warn")
        assert report["music_differs"] == 1 and report["motion_differs"] == 0
        assert report["differing"][0]["reason"] == "music"
