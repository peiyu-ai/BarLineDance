import json
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.preprocess_wild_3d import (
    Y_UP_TO_EDGE_Z_UP,
    build_wild_staging_manifests,
    build_wham_queue,
    command_convert_wham,
    command_validate,
    convert_wham_result,
    inventory_wild_cache,
    load_wham_global_run_provenance,
    record_wham_global_run,
    reconcile_wild_hmr_sequences,
    save_converted_wham,
    split_round_robin,
    wham_preflight,
)
from tools.audit_atomic_dataset import audit_dataset


class Wild3DTests(unittest.TestCase):
    def test_wham_global_conversion_matches_edge_coordinate_transform(self):
        frames = 4
        pose = np.zeros((frames, 72), dtype=np.float32)
        trans = np.tile(np.array([1.0, 2.0, 3.0], dtype=np.float32), (frames, 1))
        converted = convert_wham_result(
            {"pose_world": pose, "trans_world": trans, "frame_ids": np.arange(frames)}
        )
        expected = Y_UP_TO_EDGE_Z_UP.dot(trans[0])
        self.assertEqual(converted["motion_151"].shape, (frames, 151))
        self.assertTrue(np.allclose(converted["root_translation_z_up"][0], expected))
        self.assertTrue(np.allclose(converted["motion_151"][:, :4], 1.0))
        self.assertEqual(converted["rotation6d_z_up"].shape, (frames, 24, 6))

    def test_save_and_inventory_are_self_describing(self):
        with tempfile.TemporaryDirectory() as directory:
            pose = np.zeros((3, 72), dtype=np.float32)
            trans = np.zeros((3, 3), dtype=np.float32)
            converted = convert_wham_result({"pose_world": pose, "trans_world": trans})
            output = os.path.join(directory, "converted")
            metadata = save_converted_wham(
                converted,
                output_dir=__import__("pathlib").Path(output),
                track_id="0",
                input_path=__import__("pathlib").Path(directory) / "wham_output.pkl",
                fps=30.0,
            )
            self.assertEqual(metadata["frames"], 3)
            self.assertTrue(os.path.isfile(os.path.join(output, "atomic_motion_151.npy")))
            self.assertTrue(os.path.isfile(os.path.join(output, "quality.json")))

            cache = __import__("pathlib").Path(directory) / "cache" / "clip_a"
            cache.mkdir(parents=True)
            np.save(cache / "keypoints.npy", np.full((180, 18, 2), 0.5, dtype=np.float32))
            np.save(cache / "scores.npy", np.ones((180, 18), dtype=np.float32))
            with (cache / "meta.json").open("w") as handle:
                json.dump({"fps": 30, "source": "/tmp/clip_a.mp4"}, handle)
            records = inventory_wild_cache(
                cache.parent,
                recursive=False,
                min_frames=180,
                min_visible_fraction=0.60,
                min_score=0.30,
                max_frozen_fraction=1.0,
            )
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["status"], "ready_for_wham")

    def test_atomic_dataset_audit_rejects_cross_split_name_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = __import__("pathlib").Path(directory)
            for split in ("train", "test"):
                split_root = root / split
                split_root.mkdir()
                np.save(split_root / "motion.npy", np.zeros((1, 3, 151), dtype=np.float32))
                np.save(split_root / "music.npy", np.zeros((1, 3, 35), dtype=np.float32))
                np.save(split_root / "labels.npy", np.zeros((1, 3), dtype=np.uint8))
                with (split_root / "names.json").open("w") as handle:
                    json.dump(["same_source_slice0"], handle)
            (root / "normalizer.pt").write_bytes(b"placeholder")
            report = audit_dataset(root)
            self.assertFalse(report["valid"])
            self.assertEqual(report["cross_split_name_overlap"], 1)

    def test_atomic_dataset_audit_rejects_source_and_window_label_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = __import__("pathlib").Path(directory)
            for split, names, labels, retrieval_groups in (
                (
                    "train",
                    ["dance_a_slice0", "dance_a_slice1"],
                    np.asarray([[1, 1, 1, 1], [2, 2, 2, 2]], dtype=np.uint8),
                    ["performance/dance_a", "performance/dance_a"],
                ),
                (
                    "test",
                    ["dance_b_slice0"],
                    np.zeros((1, 4), dtype=np.uint8),
                    ["performance/dance_b"],
                ),
            ):
                split_root = root / split
                split_root.mkdir()
                np.save(split_root / "motion.npy", np.zeros((len(names), 4, 151), dtype=np.float32))
                np.save(split_root / "music.npy", np.zeros((len(names), 4, 35), dtype=np.float32))
                np.save(split_root / "labels.npy", labels)
                with (split_root / "names.json").open("w") as handle:
                    json.dump(names, handle)
                with (split_root / "retrieval_groups.json").open("w") as handle:
                    json.dump(retrieval_groups, handle)
            (root / "normalizer.pt").write_bytes(b"placeholder")
            evaluation = root / "evaluation.txt"
            evaluation.write_text("dance_a\n", encoding="utf-8")
            report = audit_dataset(
                root,
                eval_source_list=evaluation,
                window_stride=1,
                min_overlap_label_agreement=1.0,
            )
            self.assertFalse(report["valid"])
            self.assertEqual(report["train_evaluation_source_overlap"], 1)
            self.assertLess(report["adjacent_window_label_audit"]["train"]["label_agreement"], 1.0)
            self.assertEqual(
                report["source_safe_retrieval_audit"]["source_safe_atomic_frame_fraction"],
                0.0,
            )

    def test_wham_queue_is_rooted_and_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = __import__("pathlib").Path(directory)
            wham = root / "WHAM"
            wham.mkdir()
            (wham / "demo.py").write_text("# placeholder\n", encoding="utf-8")
            video = root / "dance_clip.mp4"
            video.write_bytes(b"placeholder")
            commands, skipped = build_wham_queue(
                [{"status": "ready_for_wham", "clip_id": "clip_a", "source_video": str(video)}],
                wham_root=wham,
                result_root=root / "results",
                video_root=None,
                python_executable="python",
            )
            self.assertEqual(skipped, [])
            self.assertEqual(len(commands), 1)
            self.assertIn("cd {}".format(wham), commands[0])
            self.assertIn("wham_output.pkl", commands[0])
            self.assertIn("[skip WHAM] clip_a", commands[0])
            self.assertIn("record-wham-provenance", commands[0])
            self.assertIn("unprovenanced result", commands[0])

    def test_wham_queue_shards_are_balanced_and_disjoint(self):
        groups = split_round_robin(["a", "b", "c", "d", "e"], 3)
        self.assertEqual(groups, [["a", "d"], ["b", "e"], ["c"]])
        self.assertEqual(sorted(item for group in groups for item in group), ["a", "b", "c", "d", "e"])
        with self.assertRaises(ValueError):
            split_round_robin(["a"], 0)

    def test_wham_shard_launcher_preflights_before_starting_children(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wham = root / "WHAM"
            wham.mkdir()
            demo_was_run = root / "demo_was_run"
            (wham / "demo.py").write_text(
                "from pathlib import Path\nPath({!r}).write_text('ran')\n".format(str(demo_was_run)),
                encoding="utf-8",
            )
            records = []
            for index in range(2):
                video = root / "dance_{}.mp4".format(index)
                video.write_bytes(b"placeholder")
                records.append({"status": "ready_for_wham", "clip_id": "clip_{}".format(index), "source_video": str(video)})
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            from tools.preprocess_wild_3d import command_queue_wham_shards

            queue_dir = root / "queues"
            command_queue_wham_shards(
                __import__("argparse").Namespace(
                    manifest=str(manifest),
                    wham_root=str(wham),
                    result_root=str(root / "results"),
                    output_dir=str(queue_dir),
                    shards=2,
                    video_root=None,
                    python=sys.executable,
                    max_tasks=None,
                )
            )
            launcher = queue_dir / "launch_wham_shards.sh"
            launcher_text = launcher.read_text(encoding="utf-8")
            self.assertIn('"${WHAM_PYTHON}" "${ATOMIC_WILD3D_ADAPTER}" preflight-wham', launcher_text)
            completed = subprocess.run(
                ["bash", str(launcher)],
                cwd=str(root),
                env={**os.environ, "GPU_IDS": "0,1"},
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 2, completed.stderr + completed.stdout)
            self.assertFalse(demo_was_run.exists())
            self.assertFalse((root / "results").exists())

    def test_wild_staging_manifest_groups_crops_at_recording_level(self):
        records = [
            {
                "status": "ready_for_wham",
                "clip_id": "123__clip000",
                "fps": 30.0,
                "cache_dir": "/cache/123__clip000",
                "source_video": "/video/123__clip000.mp4",
                "pose2d_path": "/cache/123__clip000/keypoints.npy",
                "scores_path": "/cache/123__clip000/scores.npy",
                "pose2d_metrics": {"frames": 480},
                "reasons": [],
            },
            {
                "status": "ready_for_wham",
                "clip_id": "123__clip001",
                "fps": 30.0,
                "cache_dir": "/cache/123__clip001",
                "source_video": "/video/123__clip001.mp4",
                "pose2d_path": "/cache/123__clip001/keypoints.npy",
                "scores_path": "/cache/123__clip001/scores.npy",
                "pose2d_metrics": {"frames": 480},
                "reasons": [],
            },
            {
                "status": "ready_for_wham",
                "clip_id": "456__clip007",
                "fps": 25.0,
                "cache_dir": "/cache/456__clip007",
                "source_video": "/video/456__clip007.mp4",
                "pose2d_path": "/cache/456__clip007/keypoints.npy",
                "scores_path": "/cache/456__clip007/scores.npy",
                "pose2d_metrics": {"frames": 375},
                "reasons": [],
            },
            {
                "status": "quarantine",
                "clip_id": "bad__clip000",
                "fps": 30.0,
                "pose2d_metrics": {"frames": 10},
                "reasons": ["too short"],
            },
        ]
        sources, sequences, summary = build_wild_staging_manifests(
            records,
            corpus="tiktok",
            split_seed=7,
            train_fraction=0.8,
            val_fraction=0.1,
        )
        self.assertEqual(len(sources), 2)
        self.assertEqual(len(sequences), 3)
        self.assertEqual(summary["excluded_by_inventory_status"], {"quarantine": 1})
        by_recording = {item["recording_id"]: item for item in sources}
        first = by_recording["tiktok:123"]
        self.assertEqual(len(first["sequence_ids"]), 2)
        self.assertEqual(first["split_status"], "provisional_pending_duplicate_content_qc")
        child = [item for item in sequences if item["recording_id"] == "tiktok:123"]
        self.assertEqual({item["split"] for item in child}, {first["split"]})
        self.assertEqual(
            {item["sequence_id"] for item in child},
            {"tiktok:123:clip000", "tiktok:123:clip001"},
        )
        self.assertTrue(all(item["assets"]["motion_151_raw"] is None for item in sequences))
        repeat_sources, _, _ = build_wild_staging_manifests(
            records,
            corpus="tiktok",
            split_seed=7,
            train_fraction=0.8,
            val_fraction=0.1,
        )
        self.assertEqual(
            [(item["recording_id"], item["split"]) for item in sources],
            [(item["recording_id"], item["split"]) for item in repeat_sources],
        )

    def test_reconcile_wild_hmr_keeps_pending_and_registers_only_valid_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            converted_root = root / "converted"
            pose = np.zeros((3, 72), dtype=np.float32)
            trans = np.zeros((3, 3), dtype=np.float32)
            converted = convert_wham_result(
                {"pose_world": pose, "trans_world": trans, "frame_ids": np.asarray([2, 3, 4])}
            )
            slam = np.zeros((5, 7), dtype=np.float32)
            slam[:, 6] = 1.0
            slam_path = root / "slam_results.pth"
            with slam_path.open("wb") as handle:
                pickle.dump(slam, handle)
            cache = root / "cache"
            cache.mkdir()
            np.save(cache / "keypoints.npy", np.zeros((5, 18, 2), dtype=np.float32))
            provenance = {
                "schema_version": "wham-global-run-v1",
                "global_requested": True,
                "estimate_local_only": False,
                "fresh_cache_verified": True,
            }
            save_converted_wham(
                converted,
                converted_root / "clip_a",
                track_id="7",
                input_path=root / "wham_output.pkl",
                fps=30.0,
                source_cache=cache,
                slam_path=slam_path,
                run_provenance=provenance,
            )
            staging = [
                {
                    "sequence_id": "tiktok:recording:clip_a",
                    "legacy_clip_id": "clip_a",
                    "timeline": {"source_start_frame": 0, "source_end_frame_exclusive": 5},
                    "assets": {"source_cache": str(cache)},
                    "representation": {"camera_in_model_input": False},
                    "qc": {"inventory_status": "ready_for_wham", "reason_codes": []},
                },
                {
                    "sequence_id": "tiktok:recording:clip_b",
                    "legacy_clip_id": "clip_b",
                    "timeline": {},
                    "assets": {},
                    "representation": {},
                    "qc": {"inventory_status": "ready_for_wham", "reason_codes": []},
                },
            ]
            records, summary = reconcile_wild_hmr_sequences(staging, converted_root=converted_root)
            by_id = {item["sequence_id"]: item for item in records}
            candidate = by_id["tiktok:recording:clip_a"]
            self.assertEqual(candidate["qc"]["hmr_status"], "candidate")
            self.assertFalse(candidate["qc"]["accepted_for_training"])
            self.assertEqual(candidate["person_track_id"], "7")
            self.assertEqual(candidate["timeline"]["source_start_frame"], 2)
            self.assertEqual(candidate["timeline"]["source_end_frame_exclusive"], 5)
            self.assertTrue(candidate["timeline"]["motion_frames_are_contiguous"])
            self.assertTrue(candidate["assets"]["motion_151_raw"].endswith("atomic_motion_151.npy"))
            self.assertEqual(by_id["tiktok:recording:clip_b"]["qc"]["hmr_status"], "pending")
            self.assertEqual(summary["status_counts"], {"pending": 1, "quarantine": 0, "candidate": 1})

    def test_wham_queue_requires_the_world_slam_runtime_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = __import__("pathlib").Path(directory)
            wham = root / "WHAM"
            wham.mkdir()
            (wham / "demo.py").write_text("# placeholder\n", encoding="utf-8")
            manifest = root / "manifest.jsonl"
            video = root / "dance_clip.mp4"
            video.write_bytes(b"placeholder")
            manifest.write_text(
                json.dumps({"status": "ready_for_wham", "clip_id": "clip_a", "source_video": str(video)})
                + "\n",
                encoding="utf-8",
            )
            output = root / "queue.sh"
            from tools.preprocess_wild_3d import command_queue_wham

            command_queue_wham(
                __import__("argparse").Namespace(
                    manifest=str(manifest),
                    wham_root=str(wham),
                    result_root=str(root / "results"),
                    output=str(output),
                    video_root=None,
                    python="python",
                    max_tasks=None,
                )
            )
            script = output.read_text(encoding="utf-8")
            self.assertIn('WHAM_PYTHON="${WHAM_PYTHON:-$DEFAULT_WHAM_PYTHON}"', script)
            self.assertIn('"${WHAM_PYTHON}" "${ATOMIC_WILD3D_ADAPTER}" preflight-wham', script)
            self.assertIn('--python "${WHAM_PYTHON}"', script)
            self.assertIn(
                '"${WHAM_PYTHON}" '
                + str(Path(__file__).resolve().parents[1] / "tools" / "preprocess_wild_3d.py")
                + " record-wham-provenance",
                script,
            )
            self.assertIn('"${WHAM_PYTHON}" ' + str(wham / "demo.py"), script)
            self.assertNotIn("(cd {} && python -c".format(wham), script)

    def test_wham_queue_missing_assets_blocks_before_marker_or_demo(self):
        """The generated shell gate, not a later task, owns the asset refusal."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wham = root / "WHAM"
            wham.mkdir()
            demo_was_run = root / "demo_was_run"
            (wham / "demo.py").write_text(
                "from pathlib import Path\nPath({!r}).write_text('ran')\n".format(str(demo_was_run)),
                encoding="utf-8",
            )
            video = root / "dance_clip.mp4"
            video.write_bytes(b"placeholder")
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps({"status": "ready_for_wham", "clip_id": "clip_a", "source_video": str(video)})
                + "\n",
                encoding="utf-8",
            )
            output = root / "queue.sh"
            from tools.preprocess_wild_3d import command_queue_wham

            command_queue_wham(
                __import__("argparse").Namespace(
                    manifest=str(manifest),
                    wham_root=str(wham),
                    result_root=str(root / "results"),
                    output=str(output),
                    video_root=None,
                    python=sys.executable,
                    max_tasks=None,
                )
            )
            completed = subprocess.run(
                ["bash", str(output)],
                cwd=str(root),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 2, completed.stderr + completed.stdout)
            result_dir = root / "results" / "clip_a" / video.stem
            self.assertFalse((result_dir / "wild3d_wham_global_run.json").exists())
            self.assertFalse(demo_was_run.exists())

    def test_wham_queue_override_uses_one_python_for_preflight_marker_and_demo(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wham = root / "WHAM"
            wham.mkdir()
            # Satisfy the official preflight with minimal importable stand-ins;
            # demo itself only records that the launch reached it.
            for relative in (
                "third-party/DPVO",
                "third-party/ViTPose",
                "lib",
                "lib/models",
                "lib/models/preproc",
                "dpvo",
            ):
                (wham / relative).mkdir(parents=True, exist_ok=True)
            for relative in (
                "lib/__init__.py",
                "lib/models/__init__.py",
                "lib/models/preproc/__init__.py",
                "dpvo/__init__.py",
            ):
                (wham / relative).write_text("", encoding="utf-8")
            for relative, symbol in (
                ("lib/models/preproc/slam.py", "SLAMModel"),
                ("lib/models/preproc/detector.py", "DetectionModel"),
                ("lib/models/preproc/extractor.py", "FeatureExtractor"),
                ("dpvo/dpvo.py", "DPVO"),
            ):
                (wham / relative).write_text("class {}:\n    pass\n".format(symbol), encoding="utf-8")
            for relative in (
                "dataset/body_models/smpl/SMPL_NEUTRAL.pkl",
                "dataset/body_models/J_regressor_wham.npy",
                "dataset/body_models/J_regressor_h36m.npy",
                "dataset/body_models/J_regressor_feet.npy",
                "dataset/body_models/smpl_mean_params.npz",
                "checkpoints/wham_vit_bedlam_w_3dpw.pth.tar",
                "checkpoints/hmr2a.ckpt",
                "checkpoints/dpvo.pth",
                "checkpoints/yolov8x.pt",
                "checkpoints/vitpose-h-multi-coco.pth",
            ):
                asset = wham / relative
                asset.parent.mkdir(parents=True, exist_ok=True)
                asset.write_bytes(b"fixture")
            demo_was_run = wham / "demo_was_run"
            (wham / "demo.py").write_text(
                "from pathlib import Path\nPath(__file__).with_name('demo_was_run').write_text('ran')\n",
                encoding="utf-8",
            )
            video = root / "dance_clip.mp4"
            video.write_bytes(b"placeholder")
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps({"status": "ready_for_wham", "clip_id": "clip_a", "source_video": str(video)})
                + "\n",
                encoding="utf-8",
            )
            invocation_log = root / "python_invocations.txt"
            wrapper = root / "wham_python_wrapper.sh"
            wrapper.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' \"$*\" >> \"$WHAM_PYTHON_LOG\"\n"
                "exec {} \"$@\"\n".format(__import__("shlex").quote(sys.executable)),
                encoding="utf-8",
            )
            os.chmod(wrapper, 0o755)
            output = root / "queue.sh"
            from tools.preprocess_wild_3d import command_queue_wham

            command_queue_wham(
                __import__("argparse").Namespace(
                    manifest=str(manifest),
                    wham_root=str(wham),
                    result_root=str(root / "results"),
                    output=str(output),
                    video_root=None,
                    python="configured-python-that-is-not-used-after-override",
                    max_tasks=None,
                )
            )
            script = output.read_text(encoding="utf-8")
            self.assertIn("DEFAULT_WHAM_PYTHON=configured-python-that-is-not-used-after-override", script)
            environment = dict(os.environ)
            environment.update({"WHAM_PYTHON": str(wrapper), "WHAM_PYTHON_LOG": str(invocation_log)})
            completed = subprocess.run(
                ["bash", str(output)],
                cwd=str(root),
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            self.assertTrue(demo_was_run.is_file())
            result_dir = root / "results" / "clip_a" / video.stem
            self.assertTrue((result_dir / "wild3d_wham_global_run.json").is_file())
            invocations = invocation_log.read_text(encoding="utf-8")
            self.assertIn("preprocess_wild_3d.py preflight-wham", invocations)
            self.assertIn("preprocess_wild_3d.py record-wham-provenance", invocations)
            self.assertIn("demo.py --video", invocations)
            # The preflight's internal world-runtime import is also launched
            # via the override, rather than accidentally via PATH's python.
            self.assertIn("-c from lib.models.preproc.slam import SLAMModel", invocations)

    def test_wham_preflight_does_not_claim_missing_assets_are_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            root = __import__("pathlib").Path(directory)
            (root / "demo.py").write_text("# placeholder\n", encoding="utf-8")
            report = wham_preflight(root, "python")
            self.assertFalse(report["ready"])
            self.assertIn("dataset/body_models/smpl/SMPL_NEUTRAL.pkl", report["missing_body_model_assets"])

    def test_slam_is_selected_by_track_frame_ids_and_not_reinterpreted_as_body_world(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pose = np.zeros((3, 72), dtype=np.float32)
            trans = np.zeros((3, 3), dtype=np.float32)
            converted = convert_wham_result(
                {"pose_world": pose, "trans_world": trans, "frame_ids": np.asarray([1, 3, 5])}
            )
            # qxyzw identity, with distinct translations so alignment is observable.
            slam = np.zeros((6, 7), dtype=np.float32)
            slam[:, 0] = np.arange(6)
            slam[:, 6] = 1.0
            slam_path = root / "slam_results.pth"
            with slam_path.open("wb") as handle:
                pickle.dump(slam, handle)
            cache = root / "cache"
            cache.mkdir()
            np.save(cache / "keypoints.npy", np.zeros((6, 18, 2), dtype=np.float32))
            provenance = {
                "schema_version": "wham-global-run-v1",
                "global_requested": True,
                "estimate_local_only": False,
                "fresh_cache_verified": True,
            }
            output = root / "converted"
            save_converted_wham(
                converted,
                output,
                track_id="0",
                input_path=root / "wham_output.pkl",
                fps=30.0,
                source_cache=cache,
                slam_path=slam_path,
                run_provenance=provenance,
            )
            with np.load(output / "camera.npz", allow_pickle=False) as camera:
                self.assertEqual(
                    set(camera.files),
                    {
                        "dpvo_c2w_unregistered_full_video",
                        "dpvo_c2w_unregistered_track",
                        "camera_frame_ids",
                        "full_video_frame_count",
                    },
                )
                self.assertTrue(np.array_equal(camera["camera_frame_ids"], [1, 3, 5]))
                self.assertTrue(
                    np.array_equal(camera["dpvo_c2w_unregistered_track"][:, 0], [1, 3, 5])
                )
            self.assertEqual(command_validate(__import__("argparse").Namespace(output_dir=str(output))), 2)
            # Sparse track IDs are retained for audit, but cannot supply frame-to-frame contact supervision.

    def test_contiguous_camera_aware_output_validates_and_local_only_sentinel_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pose = np.zeros((3, 72), dtype=np.float32)
            trans = np.zeros((3, 3), dtype=np.float32)
            converted = convert_wham_result(
                {"pose_world": pose, "trans_world": trans, "frame_ids": np.asarray([1, 2, 3])}
            )
            slam = np.zeros((4, 7), dtype=np.float32)
            slam[:, 0] = np.arange(4)
            slam[:, 6] = 1.0
            slam_path = root / "slam_results.pth"
            with slam_path.open("wb") as handle:
                pickle.dump(slam, handle)
            cache = root / "cache"
            cache.mkdir()
            np.save(cache / "keypoints.npy", np.zeros((4, 18, 2), dtype=np.float32))
            provenance = {
                "schema_version": "wham-global-run-v1",
                "global_requested": True,
                "estimate_local_only": False,
                "fresh_cache_verified": True,
            }
            output = root / "converted"
            save_converted_wham(
                converted,
                output,
                track_id="0",
                input_path=root / "wham_output.pkl",
                fps=30.0,
                source_cache=cache,
                slam_path=slam_path,
                run_provenance=provenance,
            )
            self.assertEqual(command_validate(__import__("argparse").Namespace(output_dir=str(output))), 0)

            fallback = np.zeros((4, 7), dtype=np.float32)
            fallback[:, 3] = 1.0
            fallback_path = root / "fallback_slam_results.pth"
            with fallback_path.open("wb") as handle:
                pickle.dump(fallback, handle)
            with self.assertRaisesRegex(ValueError, "local-only/SLAM-fallback"):
                save_converted_wham(
                    converted,
                    root / "fallback",
                    track_id="0",
                    input_path=root / "wham_output.pkl",
                    fps=30.0,
                    slam_path=fallback_path,
                    run_provenance=provenance,
                )

    def test_global_provenance_refuses_stale_cache_and_binds_expected_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wham = root / "WHAM"
            wham.mkdir()
            (wham / "demo.py").write_text("# placeholder\n", encoding="utf-8")
            video = root / "clip.mp4"
            video.write_bytes(b"placeholder")
            result_dir = root / "results" / "clip"
            marker = record_wham_global_run(result_dir, video=video, wham_root=wham)
            marker_path = result_dir / "wild3d_wham_global_run.json"
            self.assertTrue(marker_path.is_file())
            self.assertEqual(
                load_wham_global_run_provenance(
                    marker_path,
                    wham_output=result_dir / "wham_output.pkl",
                    source_video=video,
                ),
                marker,
            )
            (result_dir / "slam_results.pth").write_bytes(b"stale")
            with self.assertRaisesRegex(RuntimeError, "potentially local-only"):
                record_wham_global_run(result_dir, video=video, wham_root=wham)

    def test_convert_command_requires_and_consumes_fresh_global_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wham = root / "WHAM"
            wham.mkdir()
            (wham / "demo.py").write_text("# placeholder\n", encoding="utf-8")
            video = root / "clip.mp4"
            video.write_bytes(b"placeholder")
            result_dir = root / "raw" / "clip"
            record_wham_global_run(result_dir, video=video, wham_root=wham)
            wham_output = result_dir / "wham_output.pkl"
            payload = {
                0: {
                    "pose_world": np.zeros((3, 72), dtype=np.float32),
                    "trans_world": np.zeros((3, 3), dtype=np.float32),
                    "frame_ids": np.arange(3),
                }
            }
            with wham_output.open("wb") as handle:
                pickle.dump(payload, handle)
            slam = np.zeros((3, 7), dtype=np.float32)
            slam[:, 6] = 1.0
            slam_path = result_dir / "slam_results.pth"
            with slam_path.open("wb") as handle:
                pickle.dump(slam, handle)
            cache = root / "cache"
            cache.mkdir()
            np.save(cache / "keypoints.npy", np.zeros((3, 18, 2), dtype=np.float32))
            converted = root / "converted"
            self.assertEqual(
                command_convert_wham(
                    __import__("argparse").Namespace(
                        wham_output=str(wham_output),
                        output_dir=str(converted),
                        person_id=None,
                        slam=str(slam_path),
                        run_provenance=str(result_dir / "wild3d_wham_global_run.json"),
                        source_cache=str(cache),
                        source_video=str(video),
                        fps=30.0,
                        contact_velocity_threshold=0.01,
                        overwrite=False,
                    )
                ),
                0,
            )
            self.assertEqual(command_validate(__import__("argparse").Namespace(output_dir=str(converted))), 0)


if __name__ == "__main__":
    unittest.main()


def test_frozen_fraction_ignores_joints_the_detector_never_saw():
    """"Frozen" must mean a stuck tracker, not an invisible joint.

    An unseen joint has no position.  Whatever stands in for it -- NaN zeroed,
    or the frame centre the old pipeline substituted -- is the same value in
    every frame, so an unmasked difference reads it as a joint that did not
    move.  Measured on the rebuilt corpus that made ``frozen`` a second copy of
    ``not visible``: all 49 quarantined clips tripped both, nine of them on
    ``frozen`` alone, and none had a stuck tracker.
    """
    import numpy as np

    from tools.preprocess_wild_3d import _cache_metrics

    frames, joints = 60, 18
    keypoints = np.full((frames, joints, 2), np.nan, dtype=np.float64)
    scores = np.zeros((frames, joints), dtype=np.float64)
    # Half the joints are visible and moving; the other half were never seen.
    moving = np.arange(joints) < joints // 2
    for t in range(frames):
        keypoints[t, moving] = np.array([0.1 + 0.01 * t, 0.2 + 0.01 * t])
    scores[:, moving] = 0.9

    metrics = _cache_metrics(keypoints, scores, min_score=0.3)
    assert metrics["visible_joint_fraction"] == 0.5
    # Every measurable joint moved, so nothing is frozen.  Counting the unseen
    # half would report 0.5.
    assert metrics["frozen_joint_pair_fraction"] == 0.0


def test_frozen_fraction_still_catches_a_genuinely_stuck_tracker():
    import numpy as np

    from tools.preprocess_wild_3d import _cache_metrics

    frames, joints = 60, 18
    keypoints = np.tile(np.array([0.4, 0.5]), (frames, joints, 1))
    scores = np.full((frames, joints), 0.9)
    metrics = _cache_metrics(keypoints, scores, min_score=0.3)
    assert metrics["visible_joint_fraction"] == 1.0
    assert metrics["frozen_joint_pair_fraction"] == 1.0
