import json
import pathlib
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from infer_atomic import (
    GroundTruthPlanStore,
    IndexedAtomicMotionLibrary,
    ORACLE_GROUND_TRUTH_PLAN,
    SELF_DRIVEN_PLANNER,
    _generation_protocol,
    _query_retrieval_group_id,
    _source_safe_draft,
    decode_motion,
    infer_directory,
    infer_completion,
    infer_plan,
    unnormalize_motion,
)


class PlannerStub:
    # The signature tracks ``UniformD3PM.sample``: a stub that quietly accepts
    # anything would let a caller pass an argument the real planner refuses.
    def sample(self, music, padding_mask=None, temperature=1.0, deterministic=False,
               guidance_weight=1.0, transition_logit_bias=0.0):
        return torch.ones(music.shape[:2], dtype=torch.long, device=music.device)


class CompletionStub:
    """Records the sampler options it was handed rather than ignoring them.

    A stub with a fixed signature turns "a new option is not forwarded" into
    "the stub cannot be called at all", which is a TypeError in a test that was
    checking something else -- and it hid exactly that for ``start_step``.
    ``**options`` keeps the stub callable; ``seen`` is what the forwarding test
    reads.
    """

    def __init__(self):
        self.seen = []

    def sample(self, music, draft, noise_mask, guidance_weight=None, **options):
        self.seen.append(options)
        return draft


class AtomicInferenceTests(unittest.TestCase):
    def test_ground_truth_plan_store(self):
        with tempfile.TemporaryDirectory() as directory:
            for split, name in (("train", "song_slice0"), ("test", "other_slice0")):
                root = os.path.join(directory, split)
                os.makedirs(root)
                np.save(
                    os.path.join(root, "labels.npy"),
                    np.array([[1, 1, 0, 2]], dtype=np.uint8),
                )
                with open(os.path.join(root, "names.json"), "w") as handle:
                    import json

                    json.dump([name], handle)
            store = GroundTruthPlanStore(directory)
            self.assertTrue(store.has_sequence("song"))
            self.assertEqual(store.get("song", 3).tolist(), [1, 1, 0])

    def test_indexed_library_and_windowed_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            train = os.path.join(directory, "train")
            os.makedirs(train)
            motion = np.arange(2 * 6 * 3, dtype=np.float32).reshape(2, 6, 3)
            labels = np.array([[1, 1, 1, 0, 2, 2], [2, 2, 2, 2, 1, 1]])
            np.save(os.path.join(train, "motion.npy"), motion)
            np.save(os.path.join(train, "labels.npy"), labels)
            with open(os.path.join(train, "names.json"), "w") as handle:
                json.dump(["source_a_slice0", "source_b_slice0"], handle)

            library = IndexedAtomicMotionLibrary(directory)
            draft, mask = library.build_draft(torch.tensor([1, 1, 0, 2, 2]), 3)
            self.assertEqual(draft.shape, (5, 3))
            self.assertEqual(mask[:, 0].tolist(), [1.0, 1.0, 0.0, 1.0, 1.0])

            music = torch.zeros(8, 2)
            plan = infer_plan(
                PlannerStub(), music, 4, torch.device("cpu"), available_labels={1, 2}
            )
            self.assertEqual(plan.tolist(), [1] * 8)
            expected = torch.arange(24, dtype=torch.float32).reshape(8, 3)
            generated = infer_completion(
                CompletionStub(),
                music,
                expected,
                torch.ones(8, 1),
                4,
                2,
                torch.device("cpu"),
            )
            self.assertTrue(torch.allclose(generated, expected))

    def test_indexed_library_excludes_query_group_and_keys_cache_by_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            train = os.path.join(directory, "train")
            os.makedirs(train)
            np.save(
                os.path.join(train, "motion.npy"),
                np.array(
                    [
                        [[1.0, 1.0], [1.0, 1.0]],
                        [[2.0, 2.0], [2.0, 2.0]],
                    ],
                    dtype=np.float32,
                ),
            )
            np.save(
                os.path.join(train, "labels.npy"),
                np.array([[1, 1], [1, 1]], dtype=np.uint8),
            )
            with open(os.path.join(train, "names.json"), "w") as handle:
                json.dump(["source_a_slice0", "source_b_slice0"], handle)
            with open(os.path.join(train, "retrieval_groups.json"), "w") as handle:
                json.dump(["performance/a", "performance/b"], handle)

            library = IndexedAtomicMotionLibrary(directory)
            # Populate the legacy/no-exclusion cache entry first.
            self.assertTrue(torch.allclose(library.retrieve(1, 2), torch.ones(2, 2)))
            source_safe = library.retrieve(
                1, 2, exclude_retrieval_group_ids={"performance/a"}
            )
            self.assertTrue(torch.allclose(source_safe, torch.full((2, 2), 2.0)))
            draft, mask = _source_safe_draft(
                library,
                torch.tensor([1, 1]),
                2,
                _query_retrieval_group_id(library, "source_a"),
            )
            self.assertTrue(torch.equal(draft, torch.zeros(2, 2)))
            self.assertTrue(torch.equal(mask, torch.zeros(2, 1)))
            draft, mask = _source_safe_draft(
                library, torch.tensor([1, 1]), 2, "performance/a"
            )
            self.assertTrue(torch.allclose(draft, torch.full((2, 2), 2.0)))
            self.assertTrue(torch.equal(mask, torch.ones(2, 1)))

    def test_indexed_library_unknown_provenance_and_unknown_query_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            train = os.path.join(directory, "train")
            os.makedirs(train)
            np.save(
                os.path.join(train, "motion.npy"),
                np.full((1, 2, 2), 3.0, dtype=np.float32),
            )
            np.save(
                os.path.join(train, "labels.npy"),
                np.array([[1, 1]], dtype=np.uint8),
            )
            with open(os.path.join(train, "names.json"), "w") as handle:
                json.dump([None], handle)

            library = IndexedAtomicMotionLibrary(directory)
            self.assertTrue(torch.allclose(library.retrieve(1, 2), torch.full((2, 2), 3.0)))
            with self.assertRaises(KeyError):
                library.retrieve(1, 2, exclude_retrieval_group_ids={"query_group"})
            draft, mask = _source_safe_draft(
                library, torch.tensor([1, 1]), 2, None
            )
            self.assertTrue(torch.equal(draft, torch.zeros(2, 2)))
            self.assertTrue(torch.equal(mask, torch.zeros(2, 1)))

    def test_explicit_group_blocks_cross_camera_prototype_and_query_uses_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            train = os.path.join(directory, "train")
            os.makedirs(train)
            np.save(
                os.path.join(train, "motion.npy"),
                np.array(
                    [
                        [[1.0, 1.0], [1.0, 1.0]],
                        [[2.0, 2.0], [2.0, 2.0]],
                        [[3.0, 3.0], [3.0, 3.0]],
                    ],
                    dtype=np.float32,
                ),
            )
            np.save(
                os.path.join(train, "labels.npy"),
                np.ones((3, 2), dtype=np.uint8),
            )
            with open(os.path.join(train, "names.json"), "w") as handle:
                json.dump(
                    ["dance42_front_slice0", "dance42_side_slice0", "dance43_front_slice0"],
                    handle,
                )
            with open(os.path.join(train, "retrieval_groups.json"), "w") as handle:
                json.dump(["performance/dance42", "performance/dance42", "performance/dance43"], handle)

            library = IndexedAtomicMotionLibrary(directory)
            # Merely sharing a dance-name prefix is not how the group is
            # resolved; only this explicit sidecar mapping makes it known.
            self.assertEqual(
                _query_retrieval_group_id(library, "dance42_front_slice0"),
                "performance/dance42",
            )
            self.assertIsNone(_query_retrieval_group_id(library, "dance42_front"))
            values = library.retrieve(
                1, 2, exclude_retrieval_group_ids={"performance/dance42"}
            )
            self.assertTrue(torch.allclose(values, torch.full((2, 2), 3.0)))

    def test_generation_protocol_marks_oracle_as_not_headline_eligible(self):
        self.assertEqual(_generation_protocol(False), (SELF_DRIVEN_PLANNER, True))
        self.assertEqual(
            _generation_protocol(True), (ORACLE_GROUND_TRUTH_PLAN, False)
        )

    def test_unnormalize_matches_exact_frozen_normalizer_contract(self):
        """Tiny real ranges stay invertible, and model outputs are unclipped."""
        with tempfile.TemporaryDirectory() as directory:
            normalizer_path = os.path.join(directory, "normalizer.pt")
            tiny_range = torch.finfo(torch.float32).eps
            data_min = torch.tensor([0.0, 10.0, 3.0], dtype=torch.float32)
            data_max = torch.tensor([tiny_range, 20.0, 3.0], dtype=torch.float32)
            torch.save({"data_min": data_min, "data_max": data_max}, normalizer_path)

            # The first dimension has a non-zero range far below the old
            # tolerance threshold.  It must round-trip as a real range rather
            # than be treated as a constant dimension.
            raw = torch.tensor(
                [[0.0, 10.0, 3.0], [tiny_range, 20.0, 3.0]], dtype=torch.float32
            )
            safe_range = torch.where(
                data_max == data_min,
                torch.ones_like(data_min),
                data_max - data_min,
            )
            normalized = 2.0 * (raw - data_min) / safe_range - 1.0
            recovered = unnormalize_motion(normalized, normalizer_path)
            torch.testing.assert_close(recovered, raw, rtol=0.0, atol=tiny_range)
            self.assertGreater(float(recovered[1, 0]), 0.0)

            # Held-out/model values may fall beyond the fitted range.  The
            # inverse is intentionally affine, not a clipped decoder.
            outside = torch.tensor([[3.0, -3.0, 2.0]], dtype=torch.float32)
            decoded_outside = unnormalize_motion(outside, normalizer_path)
            expected = torch.tensor(
                [[2.0 * tiny_range, 0.0, 4.5]], dtype=torch.float32
            )
            torch.testing.assert_close(decoded_outside, expected, rtol=0.0, atol=tiny_range)

    def test_self_driven_target_motion_is_name_selection_only(self):
        with tempfile.TemporaryDirectory() as directory:
            train = os.path.join(directory, "dataset", "train")
            os.makedirs(train)
            np.save(
                os.path.join(train, "motion.npy"),
                np.ones((1, 4, 2), dtype=np.float32),
            )
            np.save(
                os.path.join(train, "labels.npy"),
                np.ones((1, 4), dtype=np.uint8),
            )
            with open(os.path.join(train, "names.json"), "w") as handle:
                json.dump(["query_slice0"], handle)
            open(os.path.join(directory, "dataset", "normalizer.pt"), "wb").close()

            audio_dir = os.path.join(directory, "audio")
            target_dir = os.path.join(directory, "targets")
            os.makedirs(audio_dir)
            os.makedirs(target_dir)
            open(os.path.join(audio_dir, "query.wav"), "wb").close()
            # This is intentionally not a pickle.  SELF_DRIVEN_PLANNER must
            # select its stem without opening target-motion contents.
            with open(os.path.join(target_dir, "query.pkl"), "wb") as handle:
                handle.write(b"not a pickle")

            args = SimpleNamespace(
                music_dim=2,
                seq_len=4,
                draft_noise_ratio=0.25,
                motion_dim=2,
            )

            def load_checkpoint(_, expected_stage, __):
                return (
                    CompletionStub() if expected_stage == "completion" else PlannerStub(),
                    args,
                )

            written = []

            def record_result(*_, **kwargs):
                written.append(kwargs)

            with (
                mock.patch("infer_atomic._load_checkpoint", side_effect=load_checkpoint),
                mock.patch(
                    "infer_atomic._load_music",
                    return_value=torch.zeros(4, 2),
                ),
                mock.patch("infer_atomic._write_generated_result", side_effect=record_result),
            ):
                manifest = infer_directory(
                    audio_dir=audio_dir,
                    output_dir=os.path.join(directory, "output"),
                    data_root=os.path.join(directory, "dataset"),
                    target_motion_dir=target_dir,
                    device="cpu",
                    completion_stride=4,
                )

            self.assertEqual(manifest["generation_protocol"], SELF_DRIVEN_PLANNER)
            self.assertTrue(manifest["headline_eligible"])
            self.assertEqual(
                manifest["target_motion_selection"],
                "NAME_SELECTION_ONLY_NO_CONTENT_READ",
            )
            # This fixture intentionally has no build.json or group sidecar.
            # Inference must retain the legacy compatibility path, but record
            # it as unverified rather than treating the layout as a checked
            # materialized release.
            self.assertFalse(manifest["dataset_provenance"]["release_contract_validated"])
            self.assertEqual(
                manifest["dataset_provenance"]["release_contract"],
                "absent_legacy_layout",
            )
            self.assertEqual(len(written), 1)
            self.assertEqual(
                written[0]["target_motion_selection"],
                "NAME_SELECTION_ONLY_NO_CONTENT_READ",
            )
            self.assertEqual(
                written[0]["dataset_provenance"], manifest["dataset_provenance"]
            )
            # The pipeline is deterministic given these, so a manifest that
            # omits them cannot reproduce its own run.
            sampling = manifest["sampling"]
            self.assertEqual(sampling["seed"], 42)
            self.assertEqual(sampling["completion_stride"], 4)
            self.assertEqual(sampling["sequence_order"], manifest["names"])
            for key in (
                "temperature",
                "deterministic_planner",
                "inference_batch_size",
                "guidance_weight",
                "draft_noise_ratio",
                "max_frames",
                "max_samples",
                "ground_truth_labels",
            ):
                self.assertIn(key, sampling)

    def test_decode_motion_produces_full_pose(self):
        # No longer gated on pytorch3d: dataset.rotation_ops provides the
        # rotation conversions, pinned against scipy in test_rotation_ops.py.
        with tempfile.TemporaryDirectory() as directory:
            identity_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            raw = torch.cat((torch.zeros(7), identity_6d.repeat(24)))
            normalizer = os.path.join(directory, "normalizer.pt")
            torch.save({"data_min": raw, "data_max": raw}, normalizer)
            decoded = decode_motion(torch.zeros(5, 151), normalizer)
            self.assertEqual(decoded["smpl_poses"].shape, (5, 72))
            self.assertEqual(decoded["smpl_trans"].shape, (5, 3))
            self.assertEqual(decoded["full_pose"].shape, (5, 24, 3))
            self.assertTrue(np.isfinite(decoded["full_pose"]).all())


if __name__ == "__main__":
    unittest.main()


def test_centre_fusion_takes_each_frame_from_its_most_central_window():
    """The seam, not the context.

    Non-overlapping chunks put 84 of the 630-class pair's 408 generated segment
    boundaries exactly on a chunk edge, against 2.7 expected under a uniform
    null and 0.76x on ground truth.  Overlapping the windows and taking each
    frame from the window that holds it most centrally means no frame is ever
    read off a window's edge, while every frame is still one draw from the
    model rather than an ensemble of several.
    """
    import torch

    from infer_atomic import _fuse_windows

    # Two windows of 4 over a track of 6, stride 2.  Frame 2 and 3 sit at the
    # edge of one window and near the centre of the other.
    sampled = torch.tensor([[1, 1, 1, 1], [2, 2, 2, 2]])
    fused = _fuse_windows(sampled, [0, 2], [4, 4], 6, 4, "centre")
    # Frames 0,1 are only in window 0; 4,5 only in window 1.  Frame 2 is at
    # distance 0.5 from window 0's centre and 1.5 from window 1's, so window 0
    # keeps it; frame 3 is 1.5 from window 0 and 0.5 from window 1.
    assert fused.tolist() == [1, 1, 1, 2, 2, 2]


def test_vote_fusion_is_a_majority_over_the_windows_covering_a_frame():
    import torch

    from infer_atomic import _fuse_windows

    sampled = torch.tensor([[1, 1, 1, 1], [1, 1, 2, 2], [2, 2, 2, 2]])
    fused = _fuse_windows(sampled, [0, 1, 2], [4, 4, 4], 6, 4, "vote")
    # Frame 3 is covered by all three: labels 1, 2, 2 -> 2.
    assert int(fused[3]) == 2


def test_a_stride_wider_than_the_window_is_refused():
    """A stride above the window would leave frames no window covers, and the
    fusion would silently return label 0 -- transition -- for them."""
    import pytest
    import torch

    from infer_atomic import infer_plan

    with pytest.raises(ValueError):
        infer_plan(None, torch.zeros(300, 35), 150, "cpu", plan_stride=200)


def test_precomputed_music_features_are_read_not_re_extracted(tmp_path):
    """The wild corpus's music identity is the stored 35-D array, not a WAV.

    Re-extracting is the silent failure this branch exists to prevent:
    ``tools/convert_aistpp_official.py`` measured a beat-channel correlation of
    0.36 between two runs of the same extractor over the same audio, so
    conditioning inference on a fresh extraction while the planner trained on
    the released arrays splits the corpus along the axis the planner reads --
    and nothing raises.  So the array is loaded verbatim: same values, same
    length, no resampling.
    """
    import numpy as np
    import pytest

    from infer_atomic import _audio_map, _load_music

    features = np.arange(300 * 35, dtype=np.float32).reshape(300, 35)
    path = tmp_path / "wild_v4:7195533766570282272:clip000.npy"
    np.save(path, features)

    loaded = _load_music(path, feature_dim=35)
    assert loaded.shape == (300, 35)
    np.testing.assert_array_equal(loaded.numpy(), features)

    # Truncation and padding still apply, because a query is generated at the
    # length the caller asked for.
    assert _load_music(path, 120, 35).shape == (120, 35)
    padded = _load_music(path, 400, 35)
    assert padded.shape == (400, 35)
    assert float(padded[350].abs().sum()) == 0.0

    # A width the checkpoint cannot consume fails here, with the path in the
    # message, rather than inside the first matmul.
    with pytest.raises(ValueError, match="expects 39-D"):
        _load_music(path, None, 39)

    mapping = _audio_map(tmp_path)
    assert mapping["wild_v4:7195533766570282272:clip000"] == path


def test_a_stem_carrying_both_a_wav_and_an_array_is_refused(tmp_path):
    """Which one won would otherwise depend on a precedence rule nobody stated,
    and the two are not interchangeable -- that is the whole point."""
    import numpy as np
    import pytest

    from infer_atomic import _audio_map

    np.save(tmp_path / "mBR0.npy", np.zeros((10, 35), dtype=np.float32))
    (tmp_path / "mBR0.wav").write_bytes(b"")
    with pytest.raises(ValueError, match="duplicate audio basename"):
        _audio_map(tmp_path)


def test_an_empty_audio_directory_names_both_accepted_forms(tmp_path):
    import pytest

    from infer_atomic import _audio_map

    with pytest.raises(FileNotFoundError, match="WAV or 35-D .npy"):
        _audio_map(tmp_path)


def test_an_unsourced_query_retrieves_instead_of_falling_through_to_an_empty_draft():
    """The AIST case, and the reason the paper's M5 never ran there.

    A query named for a held-out *song* has no source recording, so
    ``query_retrieval_group_id`` is None and the fail-closed branch fires.  The
    completion model takes ``(music, draft, mask)`` and never a label, so an
    empty draft means the plan cannot reach the motion at all -- 33 of the 40
    samples behind the published FID_k 17.43 were generated that way.  Declaring
    the query unsourced retrieves without exclusion, which is sound precisely
    because a song-disjoint split puts no training window on that song.
    """
    import torch

    from infer_atomic import _source_safe_draft

    class Library:
        def __init__(self):
            self.calls = []

        def build_draft(self, labels, feature_dim, **kwargs):
            self.calls.append(kwargs)
            return (torch.ones(len(labels), feature_dim), torch.ones(len(labels), 1))

    labels = torch.tensor([1, 1, 2, 2])

    fail_closed = Library()
    draft, mask = _source_safe_draft(fail_closed, labels, 3, None)
    assert fail_closed.calls == []
    assert float(mask.sum()) == 0.0 and float(draft.abs().sum()) == 0.0

    declared = Library()
    draft, mask = _source_safe_draft(declared, labels, 3, None, True)
    assert len(declared.calls) == 1
    # Nothing to exclude, so no exclusion is requested -- not an empty one.
    assert "exclude_retrieval_group_ids" not in declared.calls[0]
    assert float(mask.sum()) == 4.0

    # A query that *does* have a group still excludes it, declaration or not.
    grouped = Library()
    _source_safe_draft(grouped, labels, 3, "upload:7", True)
    assert grouped.calls[0]["exclude_retrieval_group_ids"] == ("upload:7",)


def test_the_unsourced_declaration_is_off_by_default():
    """Off by default: every artifact produced before 2026-08-16 keeps its policy."""
    import infer_atomic

    source = pathlib.Path(infer_atomic.__file__).read_text(encoding="utf-8")
    assert '"--unsourced-retrieval", action="store_true"' in source
    import inspect

    signature = inspect.signature(infer_atomic.infer_directory)
    assert signature.parameters["unsourced_retrieval"].default is False


def _continuity_library(tmpdir):
    """A release whose two prototypes sit at deliberately far-apart root origins.

    Label 1 lives near root x=100, label 2 near x=-100.  Pasted at their own
    absolute positions -- which is what ``root_continuity="off"`` does -- a plan
    that uses both makes the dancer jump 200 units at the boundary, which is the
    artifact this option exists to remove.
    """
    train = os.path.join(tmpdir, "train")
    os.makedirs(train)
    feature_dim = 8                      # 4 contacts + 3 root + 1 stand-in rotation
    motion = np.zeros((2, 4, feature_dim), dtype=np.float32)
    # windows carry [contacts x4, root xyz, one rotation channel]
    motion[0, :, 4:7] = np.array([[100.0, 10.0, 1.0], [101.0, 10.0, 1.5],
                                  [102.0, 10.0, 2.0], [103.0, 10.0, 2.5]])
    motion[1, :, 4:7] = np.array([[-100.0, -10.0, 5.0], [-101.0, -10.0, 5.5],
                                  [-102.0, -10.0, 6.0], [-103.0, -10.0, 6.5]])
    motion[0, :, 7] = 0.25               # a non-root channel that must not move
    motion[1, :, 7] = 0.75
    labels = np.array([[1, 1, 1, 1], [2, 2, 2, 2]])
    np.save(os.path.join(train, "motion.npy"), motion)
    np.save(os.path.join(train, "labels.npy"), labels)
    with open(os.path.join(train, "names.json"), "w") as handle:
        json.dump(["source_a_slice0", "source_b_slice0"], handle)
    return IndexedAtomicMotionLibrary(tmpdir), feature_dim


# Plan: label 1, an unconditioned transition, then label 2.  The gap matters --
# a transition means the completion model is free there, not that the dancer
# teleported over it, so continuity has to be carried across it.
_CONTINUITY_PLAN = torch.tensor([1, 1, 1, 1, 0, 0, 2, 2, 2, 2])


def test_root_continuity_off_is_the_published_behaviour():
    """Off must paste absolute positions -- byte-identical to every prior artifact."""
    with tempfile.TemporaryDirectory() as directory:
        library, dim = _continuity_library(directory)
        default, _ = library.build_draft(_CONTINUITY_PLAN, dim)
        explicit, _ = library.build_draft(_CONTINUITY_PLAN, dim, root_continuity="off")
        assert torch.equal(default, explicit)
        # the 200-unit jump is still there, which is the point of the comparison
        assert float(default[6, 4] - default[3, 4]) == -203.0


def test_root_continuity_xy_joins_the_segments_and_leaves_height_alone():
    with tempfile.TemporaryDirectory() as directory:
        library, dim = _continuity_library(directory)
        off, _ = library.build_draft(_CONTINUITY_PLAN, dim)
        xy, _ = library.build_draft(_CONTINUITY_PLAN, dim, root_continuity="xy")

        # The second segment now starts where the first one ended, across the gap.
        assert float(xy[6, 4]) == float(xy[3, 4])
        assert float(xy[6, 5]) == float(xy[3, 5])
        # Its internal motion is preserved: a translation, not a rewrite.
        assert torch.allclose(xy[6:, 4] - xy[6, 4], off[6:, 4] - off[6, 4])
        # Height is deliberately untouched by "xy", so its jump survives.
        assert float(xy[6, 6]) == float(off[6, 6])
        # Nothing outside the root columns moves.
        assert torch.equal(xy[:, :4], off[:, :4])
        assert torch.equal(xy[:, 7], off[:, 7])
        # The first segment is never moved -- there is nothing to continue from.
        assert torch.equal(xy[:4], off[:4])


def test_root_continuity_xyz_also_joins_height():
    with tempfile.TemporaryDirectory() as directory:
        library, dim = _continuity_library(directory)
        off, _ = library.build_draft(_CONTINUITY_PLAN, dim)
        xyz, _ = library.build_draft(_CONTINUITY_PLAN, dim, root_continuity="xyz")
        assert float(xyz[6, 6]) == float(xyz[3, 6])
        assert torch.allclose(xyz[6:, 6] - xyz[6, 6], off[6:, 6] - off[6, 6])
        assert torch.equal(xyz[:, 7], off[:, 7])


def test_root_continuity_rejects_an_unknown_mode_and_a_too_narrow_vector():
    with tempfile.TemporaryDirectory() as directory:
        library, dim = _continuity_library(directory)
        try:
            library.build_draft(_CONTINUITY_PLAN, dim, root_continuity="root")
        except ValueError as error:
            assert "off, xy or xyz" in str(error)
        else:
            raise AssertionError("an unknown continuity mode must be refused")
        # A feature vector with no room for root position must fail loudly rather
        # than translate whatever happens to sit at columns 4:7.
        try:
            library.build_draft(_CONTINUITY_PLAN, 5, root_continuity="xy")
        except ValueError as error:
            assert "root position" in str(error) or "feature dimension" in str(error)
        else:
            raise AssertionError("a vector too narrow for root position must be refused")


def test_root_continuity_is_off_by_default_and_rides_into_the_manifest():
    import inspect
    import infer_atomic

    signature = inspect.signature(infer_atomic.infer_directory)
    assert signature.parameters["draft_root_continuity"].default == "off"
    source = pathlib.Path(infer_atomic.__file__).read_text(encoding="utf-8")
    assert '"draft_root_continuity": draft_root_continuity' in source
    assert '"--draft-root-continuity"' in source


def test_gap_fill_zero_is_the_published_behaviour():
    """Zero is a pose, not an absence -- and it must stay the default."""
    with tempfile.TemporaryDirectory() as directory:
        library, dim = _continuity_library(directory)
        default, mask = library.build_draft(_CONTINUITY_PLAN, dim)
        explicit, _ = library.build_draft(_CONTINUITY_PLAN, dim, gap_fill="zero")
        assert torch.equal(default, explicit)
        assert float(default[4].abs().sum()) == 0.0        # the transition frames
        assert mask[4, 0] == 0.0


def test_gap_fill_hold_and_interpolate_remove_the_step_but_not_the_mask():
    with tempfile.TemporaryDirectory() as directory:
        library, dim = _continuity_library(directory)
        zero, mask_zero = library.build_draft(_CONTINUITY_PLAN, dim)
        held, mask_hold = library.build_draft(_CONTINUITY_PLAN, dim, gap_fill="hold")
        ramp, mask_ramp = library.build_draft(_CONTINUITY_PLAN, dim, gap_fill="interpolate")

        # The mask is the model's noise scale and its record of what was really
        # retrieved; a filled gap is neither, so it must not move.
        assert torch.equal(mask_zero, mask_hold)
        assert torch.equal(mask_zero, mask_ramp)
        # The retrieved frames themselves are untouched.
        assert torch.equal(held[:4], zero[:4])
        assert torch.equal(held[6:], zero[6:])
        # hold carries the last retrieved frame across the gap
        assert torch.equal(held[4], zero[3])
        assert torch.equal(held[5], zero[3])
        # interpolate ramps between the two sides: gap of 3 steps, so 1/3 and 2/3
        expected = zero[3] * (2.0 / 3.0) + zero[6] * (1.0 / 3.0)
        assert torch.allclose(ramp[4], expected)
        # and it really is monotone between them on the root column
        assert (zero[3, 4] - ramp[4, 4]).sign() == (ramp[4, 4] - ramp[5, 4]).sign()


def test_gap_fill_extends_past_the_ends_and_leaves_an_empty_plan_alone():
    with tempfile.TemporaryDirectory() as directory:
        library, dim = _continuity_library(directory)
        plan = torch.tensor([0, 0, 1, 1, 0, 0])
        held, _ = library.build_draft(plan, dim, gap_fill="hold")
        assert torch.equal(held[0], held[2])               # leading frames
        assert torch.equal(held[5], held[3])               # trailing frames
        # A plan with nothing retrieved has nothing to hold; inventing a pose
        # there would condition the model on a value no prototype produced.
        empty, empty_mask = library.build_draft(torch.zeros(5, dtype=torch.long), dim,
                                                gap_fill="interpolate")
        assert float(empty.abs().sum()) == 0.0
        assert float(empty_mask.sum()) == 0.0


def test_gap_fill_rejects_an_unknown_mode_and_is_off_by_default():
    import inspect
    import infer_atomic

    with tempfile.TemporaryDirectory() as directory:
        library, dim = _continuity_library(directory)
        try:
            library.build_draft(_CONTINUITY_PLAN, dim, gap_fill="linear")
        except ValueError as error:
            assert "zero, hold or interpolate" in str(error)
        else:
            raise AssertionError("an unknown gap-fill mode must be refused")
    assert inspect.signature(infer_atomic.infer_directory).parameters["draft_gap_fill"].default == "zero"
    source = pathlib.Path(infer_atomic.__file__).read_text(encoding="utf-8")
    assert '"draft_gap_fill": draft_gap_fill' in source


class GlobalMusicPlannerTest(unittest.TestCase):
    """A whole-track planner must be fed its summary, and it must be the right one.

    ``AtomicPlannerTransformer`` refuses to run without the summary when it was
    built with one -- so before this path existed, a global-music checkpoint
    could be trained and evaluated but not used to generate anything, and the
    failure was an exception at sampling time rather than a missing feature
    anyone had written down.
    """

    def test_summary_is_the_two_moments_of_the_real_frames(self):
        from infer_atomic import track_summary

        music = torch.randn(97, 35)
        summary = track_summary(music)
        self.assertEqual(tuple(summary.shape), (70,))
        self.assertTrue(torch.allclose(summary[:35], music.mean(dim=0), atol=1e-6))
        self.assertTrue(
            torch.allclose(summary[35:], music.std(dim=0, unbiased=False), atol=1e-6))

    def test_padding_would_change_the_summary(self):
        """Why the summary is taken before padding, stated as a test."""
        from infer_atomic import track_summary, _pad_frames

        music = torch.randn(60, 35) + 3.0
        honest = track_summary(music)
        padded = track_summary(_pad_frames(music, 150))
        self.assertGreater(float((honest - padded).abs().max()), 0.5)

    def test_a_plain_planner_is_not_handed_a_summary(self):
        from infer_atomic import planner_wants_global_music
        from model.atomic_planner import AtomicPlannerTransformer, UniformD3PM

        model = AtomicPlannerTransformer(num_atomic_classes=4, music_dim=35,
                                         latent_dim=16, num_layers=1, num_heads=2,
                                         ff_size=32)
        self.assertFalse(planner_wants_global_music(UniformD3PM(model, num_steps=2)))

    def test_a_global_music_planner_is_detected_and_plans(self):
        from infer_atomic import infer_plan, planner_wants_global_music
        from model.atomic_planner import AtomicPlannerTransformer, UniformD3PM

        model = AtomicPlannerTransformer(num_atomic_classes=4, music_dim=35,
                                         latent_dim=16, num_layers=1, num_heads=2,
                                         ff_size=32, global_music=True)
        planner = UniformD3PM(model, num_steps=2)
        self.assertTrue(planner_wants_global_music(planner))
        # Without the plumbing this raises; the assertion is that it does not.
        labels = infer_plan(planner, torch.randn(200, 35), 150, "cpu",
                            plan_stride=50, plan_fusion="vote")
        self.assertEqual(len(labels), 200)


class EmptyPlanReportingTest(unittest.TestCase):
    """An empty plan must not report the best possible conditioning.

    ``safe_draft_condition_fraction`` returned 1.0 when the plan named no atomic
    frame -- zero of zero -- and that is the reading it gave on the AIST arm
    whose fid_k (23.473) was quoted as reproducing the paper: 21 of its 40 clips
    have a 100%-transition plan, and all 40 recorded 1.0.  Both of the 2026-08-16
    checks passed on generations the vocabulary took no part in.
    """

    def test_no_atomic_frame_reports_null_not_one(self):
        from infer_atomic import _safe_draft_condition_fraction

        labels = torch.zeros(64, dtype=torch.long)
        mask = torch.zeros(64, 1)
        self.assertIsNone(_safe_draft_condition_fraction(labels, mask))

    def test_a_conditioned_plan_still_reports_its_fraction(self):
        from infer_atomic import _safe_draft_condition_fraction

        labels = torch.zeros(10, dtype=torch.long)
        labels[2:8] = 5
        mask = torch.zeros(10, 1)
        mask[2:5] = 1.0
        self.assertAlmostEqual(_safe_draft_condition_fraction(labels, mask), 0.5)

    def test_segment_count_separates_empty_from_fully_conditioned(self):
        from infer_atomic import _plan_atomic_segments, _safe_draft_condition_fraction

        empty = torch.zeros(30, dtype=torch.long)
        full = torch.zeros(30, dtype=torch.long)
        full[5:10] = 3
        full[15:20] = 4
        full_mask = torch.zeros(30, 1)
        full_mask[5:10] = 1.0
        full_mask[15:20] = 1.0
        # The fraction alone would read 1.0 for one and None for the other only
        # because of the fix above; the count says which is which outright.
        self.assertEqual(_plan_atomic_segments(empty), 0)
        self.assertEqual(_plan_atomic_segments(full), 2)
        self.assertEqual(_safe_draft_condition_fraction(full, full_mask), 1.0)


class NoPlanConditioningControlTest(unittest.TestCase):
    """The control the two-stage design has never been measured against.

    Every comparison in this repo has been between two ways of planning --
    window strides, fusion modes, vocabularies, condition paths.  None has been
    against *not* planning, and without that the question "does the plan
    contribute anything" has no denominator.
    """

    def test_the_draft_and_mask_are_zero_and_no_retrieval_happens(self):
        from infer_atomic import _source_safe_draft

        class RefusingLibrary:
            def build_draft(self, *args, **kwargs):
                raise AssertionError("retrieval must not run under the control")

        labels = torch.zeros(24, dtype=torch.long)
        labels[4:12] = 7
        draft, mask = _source_safe_draft(
            RefusingLibrary(), labels, 151, "group-a", no_plan_conditioning=True)
        self.assertEqual(tuple(draft.shape), (24, 151))
        self.assertEqual(float(draft.abs().sum()), 0.0)
        self.assertEqual(float(mask.sum()), 0.0)

    def test_it_overrides_even_a_declared_unsourced_query(self):
        from infer_atomic import _source_safe_draft

        class RefusingLibrary:
            def build_draft(self, *args, **kwargs):
                raise AssertionError("retrieval must not run under the control")

        labels = torch.ones(8, dtype=torch.long)
        draft, mask = _source_safe_draft(
            RefusingLibrary(), labels, 151, None, unsourced_retrieval=True,
            no_plan_conditioning=True)
        self.assertEqual(float(mask.sum()), 0.0)

    def test_a_control_artifact_is_not_headline_eligible(self):
        """A number produced without the vocabulary is not a number about this method."""
        import inspect

        from infer_atomic import _write_generated_result

        source = inspect.getsource(_write_generated_result)
        self.assertIn("if no_plan_conditioning:", source)
        self.assertIn("headline_eligible = False", source)


def test_a_tied_vote_no_longer_goes_to_transition_just_because_zero_sorts_first():
    """``counts.argmax`` handed every tie to label 0, and label 0 is transition.

    Measured on the 33 clean5b5 M6 clips (``tools/probe_plan_vote_ties.py``,
    ``planner_v1A_s15/planner_step135648.pt``, seed 20260816): 7.28% of frames
    are ties, 62.8% of those have transition among the winners, and the rule
    alone accounted for 4.6 of the 22.0 points by which the shipped plan's
    transition share exceeded the ground truth.  Nothing chose that; it is the
    index order of the label space.
    """
    import torch

    from infer_atomic import _fuse_windows

    # Two windows of 4 over a track of 6, stride 2, disagreeing everywhere.
    # Frames 2 and 3 are covered by both, so each is a 1-1 tie between 5 and 0.
    sampled = torch.tensor([[5, 5, 5, 5], [0, 0, 0, 0]])
    args = (sampled, [0, 2], [4, 4], 6, 4, "vote")

    # Frame 2 is 0.5 from window 0's centre and 1.5 from window 1's, so the
    # window that voted 5 holds it more centrally; frame 3 is the mirror image.
    assert _fuse_windows(*args, tie_break="centre").tolist() == [5, 5, 5, 0, 0, 0]
    # The published behaviour: both contested frames go to transition.
    assert _fuse_windows(*args, tie_break="index").tolist() == [5, 5, 0, 0, 0, 0]


def test_the_tie_break_changes_nothing_where_the_vote_has_a_winner():
    """The fix must be confined to ties, or it is a different estimator."""
    import torch

    from infer_atomic import _fuse_windows, _window_starts

    generator = torch.Generator().manual_seed(20260823)
    window, stride, total = 12, 3, 40
    # ``_window_starts`` is what ``infer_plan`` uses: it appends the final
    # start, so every frame is covered.  Building the list by hand leaves the
    # tail uncovered, and an uncovered frame has no votes at all -- which both
    # rules resolve to label 0, and which is the case
    # ``test_a_stride_wider_than_the_window_is_refused`` exists to keep out.
    starts = _window_starts(total, window, stride)
    sampled = torch.randint(0, 4, (len(starts), window), generator=generator)
    lengths = [window] * len(starts)

    classes = int(sampled.max()) + 1
    counts = torch.zeros(total, classes, dtype=torch.int32)
    for row, start in zip(sampled, starts):
        counts[torch.arange(start, start + window), row.long()] += 1
    tied = (counts == counts.max(dim=1, keepdim=True).values).sum(dim=1) > 1

    args = (sampled, starts, lengths, total, window, "vote")
    centre = _fuse_windows(*args, tie_break="centre")
    index = _fuse_windows(*args, tie_break="index")
    assert tied.any(), "the fixture must actually produce ties"
    assert torch.equal(centre[~tied], index[~tied])
    # And a tie is never resolved to a class nobody voted for.
    assert (counts[torch.arange(total), centre.long()] > 0).all()


def test_an_unknown_tie_break_is_refused_rather_than_defaulted():
    import pytest
    import torch

    from infer_atomic import _fuse_windows

    with pytest.raises(ValueError):
        _fuse_windows(torch.zeros(1, 4, dtype=torch.long), [0], [4], 4, 4,
                      "vote", tie_break="lowest")


def test_labels_the_library_cannot_serve_are_counted_not_only_rewritten():
    """A plan frame with no retrievable prototype becomes transition.

    That step is not in the paper -- its library covers all K prototypes by
    construction -- and until 2026-08-23 no artifact recorded how many frames it
    was, so a plan could be a third transition for a reason nothing in its own
    record could name.
    """
    import torch

    from infer_atomic import infer_plan

    class ThreeClassPlanner:
        def sample(self, music, padding_mask=None, temperature=1.0,
                   deterministic=False, guidance_weight=1.0,
                   transition_logit_bias=0.0):
            row = torch.tensor([1, 1, 1, 1, 7, 7, 7, 7])
            return row.repeat(music.shape[0], 1)[:, : music.shape[1]]

    report = {}
    plan = infer_plan(
        ThreeClassPlanner(), torch.zeros(8, 2), 8, torch.device("cpu"),
        available_labels={1}, min_segment_length=2, vote_window=1, stats=report,
    )
    # Label 7 has no prototype, so those four frames are transition now.
    assert plan.tolist() == [1, 1, 1, 1, 0, 0, 0, 0]
    assert report["frames_rewritten_to_transition"] == 4
    assert report["transition_frames_after_fusion"] == 0
    assert report["transition_frames_after_refine"] == 4
    assert report["atomic_segments_after_refine"] == 1
    assert report["plan_transition_policy"] == "protect"


def test_taper_fusion_discounts_a_vote_cast_from_a_window_edge():
    """``vote`` and ``centre`` are the ends of one axis; taper is the middle.

    A flat plurality is what lets the class holding the largest share of the
    marginal -- transition -- win where no window is confident, and it counts a
    draw taken from a window's edge, which is the draw ``centre`` exists to
    avoid trusting.  Weighting by centrality is the same triangular shape
    ``_blend_weights`` already applies to the motion being stitched.

    The margin here is 2.5x rather than an epsilon on purpose: a linear taper is
    additive in distance, so votes whose distances sum alike land within float
    noise of each other, and a fixture that turns on that noise would be pinning
    the rounding, not the rule.  On the real geometry (window 150, stride 15)
    exact weight ties are 0.39% of frames, so the rule usually has an opinion.
    """
    import torch

    from infer_atomic import _fuse_windows

    # Frame 9 is held at distance 0.5 by two windows that vote 1, and at 3.5-4.5
    # by three that vote 2.  A flat count is 3-2 for label 2.
    starts, lengths, total, window = [0, 1, 4, 5, 9], [10] * 5, 19, 10
    sampled = torch.full((5, 10), 7, dtype=torch.long)
    for row, (start, label) in enumerate(zip(starts, [2, 2, 1, 1, 2])):
        sampled[row, 9 - start] = label
    assert int(_fuse_windows(sampled, starts, lengths, total, window, "vote")[9]) == 2
    # Weighted: 0.727 for label 2 against 1.818 for label 1.
    assert int(_fuse_windows(sampled, starts, lengths, total, window, "taper")[9]) == 1


def test_taper_is_opt_in_and_never_what_an_unspecified_fusion_does():
    import inspect

    from infer_atomic import infer_plan

    assert inspect.signature(infer_plan).parameters["plan_fusion"].default == "centre"


def test_the_manifest_records_the_seed_each_clip_actually_got(tmp_path):
    """Inferring a per-sample seed from a rule is what failed on 2026-08-23.

    ``seed_everything(seed + index)`` uses the clip's position in *this
    process's* pending list, so the driver's shard count is part of the
    reproduction key.  Two runs of the same arm with the same base seeds over 2
    shards and over 6 gave fid_k 9.997 and 9.515.  Nothing in either manifest
    said so; ``sequence_order`` alone does not, because it is per shard and the
    rule that turns it into a seed was only ever a comment.
    """
    import inspect

    import infer_atomic

    source = inspect.getsource(infer_atomic.infer_directory)
    assert '"per_sample_seed": per_sample_seed' in source
    # Both dispatch paths must fill it, or a short-clip corpus records nothing.
    assert source.count("per_sample_seed[") == 2


def test_the_bar_grid_uses_the_music_channel_m1_actually_cut_on():
    """M1's boundaries are a closed-form function of the planner's own input.

    Channel 34 is the beat one-hot; M1 cuts every 4 beats at a per-clip phase.
    Measured over the 65 clean5b5 M6 clips: 97.3% of ground-truth boundaries
    are on a single 4-beat phase of that grid (uniform-random control 3.6%),
    against 30.7% of the planner's own.
    """
    import numpy as np
    import torch

    from infer_atomic import BEAT_CHANNEL, snap_plan_to_bar_grid

    music = np.zeros((160, 35), dtype=np.float32)
    music[:, 0] = 1.0
    beats = np.arange(0, 160, 16)
    music[beats, BEAT_CHANNEL] = 1.0                     # a beat every 16 frames
    # ``choose_phase`` guesses the downbeat from onset energy, so a flat
    # envelope leaves the phase arbitrary -- a first version of this fixture was
    # flat and the rule chose phase 3, which is the tool behaving correctly on
    # degenerate input.  Put the energy on every fourth beat so phase 0 wins for
    # the reason the rule exists.
    music[beats[0::4], 0] = 10.0
    # Bar 0 (frames 0-63) is mostly transition but names class 4 twice;
    # bar 1 (64-127) names nothing at all.
    plan = torch.zeros(160, dtype=torch.long)
    plan[10] = 4
    plan[40] = 4
    plan[130] = 9
    snapped, phase = snap_plan_to_bar_grid(plan, music)
    assert phase == 0
    assert set(snapped[:64].tolist()) == {4}, "a named class survives a transition majority"
    assert set(snapped[64:128].tolist()) == {0}, "a bar naming nothing stays transition"
    assert set(snapped[128:].tolist()) == {9}


def test_the_bar_grid_leaves_a_clip_with_no_usable_grid_alone():
    import numpy as np
    import torch

    from infer_atomic import snap_plan_to_bar_grid

    plan = torch.arange(40) % 3
    silent = np.zeros((40, 35), dtype=np.float32)
    same, phase = snap_plan_to_bar_grid(plan, silent)
    assert phase is None
    assert torch.equal(same, plan)


def test_the_bar_grid_refuses_music_without_the_beat_channel():
    import numpy as np
    import pytest
    import torch

    from infer_atomic import snap_plan_to_bar_grid

    with pytest.raises(ValueError):
        snap_plan_to_bar_grid(torch.zeros(20, dtype=torch.long),
                              np.zeros((20, 8), dtype=np.float32))


def _span_fixture(tmp, name, music_frames, wav_seconds, fps=30, sr=16000):
    """One query: a 35-D music array and the ingest wav it will be compared to."""
    import soundfile
    _, upload, clip = name.split(":")
    ingest = tmp / "ingest" / "{}__{}".format(upload, clip)
    ingest.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(ingest / "audio.wav"),
                    np.zeros(int(round(wav_seconds * sr)), dtype=np.float32), sr)
    audio_dir = tmp / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    npy = audio_dir / (name + ".npy")
    np.save(str(npy), np.zeros((music_frames, 35), dtype=np.float32))
    return (name, npy, None), tmp / "ingest"


def test_the_music_span_gate_refuses_a_clip_cut_from_another_generation():
    # The 4/3 shape this gate was written for: 545 frames of music against a
    # 13.63 s clip.  Refusing is the point -- the generated length IS the music
    # length, so this would produce a complete, self-consistent, wrongly-timed
    # result.
    import pytest
    from infer_atomic import check_music_span
    with tempfile.TemporaryDirectory() as raw:
        tmp = pathlib.Path(raw)
        item, ingest = _span_fixture(tmp, "wild_v4:100:clip000", 545, 13.633)
        with pytest.raises(ValueError, match="music span disagrees"):
            check_music_span([item], ingest, "refuse")
        report = check_music_span([item], ingest, "warn")
        assert report["mismatch"] == 1 and report["ok"] == 0
        assert 1.33 < report["mismatched"][0]["ratio"] < 1.34


def test_the_music_span_gate_tolerates_a_container_carrying_a_partial_frame():
    # The reason the tolerance is 2% and not one frame: an audio container holds
    # a little more than an exact frame count, and a one-frame tolerance flagged
    # 46 of the 65 M6 clips -- a gate that fires on everything gets switched off.
    from infer_atomic import check_music_span
    with tempfile.TemporaryDirectory() as raw:
        tmp = pathlib.Path(raw)
        item, ingest = _span_fixture(tmp, "wild_v4:100:clip000", 589, 19.6905)
        report = check_music_span([item], ingest, "refuse")
        assert report["ok"] == 1 and report["mismatch"] == 0


def test_a_query_with_no_local_wav_is_reported_as_unchecked_not_as_a_pass():
    # "Nothing was checked" and "everything passed" must not be the same row.
    from infer_atomic import check_music_span
    with tempfile.TemporaryDirectory() as raw:
        tmp = pathlib.Path(raw)
        item, ingest = _span_fixture(tmp, "wild_v4:100:clip000", 545, 13.633)
        (ingest / "100__clip000" / "audio.wav").unlink()
        report = check_music_span([item], ingest, "refuse")
        assert report == dict(report, ok=0, mismatch=0, no_wav=1)


def test_a_wav_query_is_not_applicable_rather_than_silently_passing():
    # On AIST the query's audio IS the wav, so the span is its own by
    # construction and the gate has nothing to say about it.
    from infer_atomic import check_music_span
    with tempfile.TemporaryDirectory() as raw:
        tmp = pathlib.Path(raw)
        item, ingest = _span_fixture(tmp, "wild_v4:100:clip000", 545, 13.633)
        wav_item = (item[0], ingest / "100__clip000" / "audio.wav", None)
        report = check_music_span([wav_item], ingest, "refuse")
        assert report["not_applicable"] == 1 and report["ok"] == 0


def test_the_music_span_gate_can_be_turned_off_and_says_so():
    from infer_atomic import check_music_span
    report = check_music_span([], "nowhere", "off")
    assert report["checked"] is False


def test_the_query_period_reaches_the_retrieval_rule(tmp_path):
    """``--retrieval-rule tempo`` was recorded in the manifest and did nothing.

    ``_source_safe_draft`` has three returns and only one of them carried
    ``target_period``; generation took the other, so the rule was active, the
    period was computed correctly, and the argument was dropped at the last
    step.  The output was byte-identical to the duration rule on all 20 clips
    checked, and nothing anywhere said so -- the manifest said ``tempo``.

    This pins the wiring rather than the rule: every path that builds a draft
    must forward the period, or the flag is a label on an experiment that did
    not happen.
    """
    import inspect

    from infer_atomic import _source_safe_draft

    source = inspect.getsource(_source_safe_draft)
    builds = source.count("build_draft(")
    forwards = source.count("target_period=target_period")
    assert builds >= 2, "if there is only one path this test proves nothing"
    assert forwards == builds, (
        "{} of {} build_draft call(s) in _source_safe_draft forward "
        "target_period".format(forwards, builds))


def test_the_music_period_is_read_from_the_beat_channel():
    """Known answers, both directions."""
    import numpy as np

    from infer_atomic import music_settle_period

    music = np.zeros((300, 35), dtype=np.float32)
    music[::15, 34] = 1.0                       # a beat every 15 frames
    assert music_settle_period(music) == 15.0

    quiet = np.zeros((300, 35), dtype=np.float32)
    assert np.isnan(music_settle_period(quiet))  # too few beats to state one
    assert np.isnan(music_settle_period(np.zeros((300, 8), dtype=np.float32)))


def test_every_generation_path_forwards_the_sampler_options():
    """Three flags in a row reached one path and not the other.

    ``target_period`` (2026-08-29), ``plan_bar_beats`` and ``completion_start_step``
    (both 2026-08-30) were each recorded in the manifest, each computed
    correctly, and each dropped before the code that would have used it.  The
    symptom is the same every time and it is invisible: the run produces output
    byte-identical to the default -- or, for the start step, identical across
    three different values -- while every artifact says the option was on.

    ``infer_directory`` has two generation paths, a batched one for clips that
    fit the planner window and a single-clip one for the rest.  Anything the CLI
    can set has to reach both.
    """
    import inspect

    from infer_atomic import infer_directory

    source = inspect.getsource(infer_directory)
    for option in ("draft_seam_blend", "draft_root_continuity", "draft_gap_fill",
                   "completion_start_step", "plan_bar_beats", "draft_recurrence_variety",
                   "completion_reproject_every", "completion_sample_steps"):
        assert source.count(option) >= 3, (
            "{} appears {} time(s) in infer_directory: it must be a parameter and "
            "reach both the batched and the single-clip path".format(
                option, source.count(option)))
