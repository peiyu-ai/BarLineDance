import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.convert_aistpp_official import (
    MOTION_DIM,
    SupplementError,
    build_supplement,
    encode_motion_151,
    retrieval_group_of,
    stable_val_group,
    verify_overlaps,
)


def _official(frames60=8, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "smpl_poses": rng.normal(scale=0.3, size=(frames60, 72)).astype(np.float64),
        "smpl_trans": rng.normal(scale=50.0, size=(frames60, 3)).astype(np.float64),
        "smpl_scaling": np.asarray([100.0]),
    }


class EncodeMotionTests(unittest.TestCase):
    def test_downsamples_from_index_zero_and_lays_out_151_dimensions(self):
        payload = _official(frames60=10)
        motion = encode_motion_151(
            payload["smpl_poses"], payload["smpl_trans"], payload["smpl_scaling"]
        )
        self.assertEqual(motion.shape, (5, MOTION_DIM))
        # Translation is the rotated, rescaled even-indexed frames: (x, y, z) -> (x, -z, y).
        expected = payload["smpl_trans"][::2] / payload["smpl_scaling"]
        np.testing.assert_allclose(motion[:, 4], expected[:, 0], rtol=0, atol=1e-6)
        np.testing.assert_allclose(motion[:, 5], -expected[:, 2], rtol=0, atol=1e-6)
        np.testing.assert_allclose(motion[:, 6], expected[:, 1], rtol=0, atol=1e-6)

    def test_contacts_are_binary_and_final_frame_is_never_a_contact(self):
        payload = _official(frames60=12, seed=3)
        motion = encode_motion_151(
            payload["smpl_poses"], payload["smpl_trans"], payload["smpl_scaling"]
        )
        contacts = motion[:, :4]
        self.assertTrue(set(np.unique(contacts).tolist()) <= {0.0, 1.0})
        # Foot speed is a forward difference, so the last frame has no speed and
        # therefore reads as contact; that is upstream's behaviour, pinned here.
        np.testing.assert_array_equal(contacts[-1], np.ones(4, dtype=np.float32))

    def test_rejects_input_that_is_not_smpl_shaped(self):
        with self.assertRaises(SupplementError):
            encode_motion_151(np.zeros((4, 71)), np.zeros((4, 3)), np.asarray([1.0]))


class SplitPolicyTests(unittest.TestCase):
    def test_retrieval_group_drops_only_the_channel_suffix(self):
        self.assertEqual(
            retrieval_group_of("gBR_sBM_cAll_d04_mBR0_ch01"), "aistpp/gBR_sBM_cAll_d04_mBR0"
        )

    def test_group_partition_is_stable_and_independent_of_call_order(self):
        groups = ["aistpp/g{}".format(i) for i in range(200)]
        first = [stable_val_group(g) for g in groups]
        second = [stable_val_group(g) for g in reversed(groups)][::-1]
        self.assertEqual(first, second)
        self.assertTrue(0 < sum(first) < len(first))


class _Fixture:
    """A miniature release plus an official archive that agrees with it."""

    def __init__(self, root: Path, *, corrupt: int = 0):
        self.root = root
        self.motions = root / "official"
        self.motions.mkdir()
        self.raw = root / "raw"
        (self.raw / "sequences").mkdir(parents=True)

        shared = ["gBR_sBM_cAll_d04_mBR0_ch01", "gBR_sBM_cAll_d04_mBR0_ch02"]
        new = ["gBR_sBM_cAll_d04_mBR0_ch03", "gBR_sBM_cAll_d07_mBR0_ch01"]
        records = []
        for index, name in enumerate(shared + new):
            payload = _official(frames60=12, seed=index)
            with (self.motions / "{}.pkl".format(name)).open("wb") as handle:
                pickle.dump(payload, handle)
            if name not in shared:
                continue
            motion = encode_motion_151(
                payload["smpl_poses"], payload["smpl_trans"], payload["smpl_scaling"]
            )
            if index < corrupt:
                motion = motion + 0.5
            store = self.raw / "sequences" / name
            store.mkdir()
            np.save(store / "motion_151_raw.npy", motion)
            np.save(store / "music_35.npy", np.full((len(motion), 35), index, dtype=np.float32))
            records.append(
                {
                    "sequence_id": "aistpp/{}/sequence0".format(name),
                    "recording_id": "aistpp/{}".format(name),
                    "retrieval_group_id": retrieval_group_of(name),
                    "split": "train",
                    "frame_count": int(len(motion)),
                    "motion_path": "sequences/{}/motion_151_raw.npy".format(name),
                    "music_path": "sequences/{}/music_35.npy".format(name),
                    "music_sha256": "0" * 64,
                }
            )
        with (self.raw / "sequences.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")


class SupplementBuildTests(unittest.TestCase):
    def test_publishes_only_official_only_names_with_donated_music(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory))
            out = Path(directory) / "supplement"
            report = build_supplement(
                motions_dir=fixture.motions, raw_root=fixture.raw, output_dir=out
            )
            self.assertEqual(report["counts"]["published"], 2)
            self.assertEqual(report["counts"]["already_in_release"], 2)
            rows = [json.loads(line) for line in (out / "sequences.jsonl").open()]
            names = sorted(row["recording_id"] for row in rows)
            self.assertEqual(
                names, ["aistpp/gBR_sBM_cAll_d04_mBR0_ch03", "aistpp/gBR_sBM_cAll_d07_mBR0_ch01"]
            )
            for row in rows:
                music = np.load(out / row["music_path"])
                motion = np.load(out / row["motion_path"])
                self.assertEqual(len(music), len(motion))
                self.assertEqual(music.shape[1], 35)
                self.assertEqual(motion.shape[1], MOTION_DIM)
                # Music is a donor copy, so it carries a donor's constant value.
                self.assertTrue(np.all(np.isin(music, [0.0, 1.0])))

    def test_sequence_in_a_frozen_group_inherits_that_split(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory))
            out = Path(directory) / "supplement"
            build_supplement(motions_dir=fixture.motions, raw_root=fixture.raw, output_dir=out)
            rows = {
                json.loads(line)["recording_id"]: json.loads(line)
                for line in (out / "sequences.jsonl").open()
            }
            inherited = rows["aistpp/gBR_sBM_cAll_d04_mBR0_ch03"]
            self.assertEqual(inherited["split"], "train")
            self.assertEqual(inherited["split_note"], "inherited_from_frozen_performance_group")
            fresh = rows["aistpp/gBR_sBM_cAll_d07_mBR0_ch01"]
            self.assertIn(fresh["split"], {"train", "val"})
            self.assertEqual(
                fresh["split_note"], "new_performance_group_stable_hash_partition"
            )

    def test_a_disagreeing_transform_aborts_before_anything_is_published(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory), corrupt=2)
            out = Path(directory) / "supplement"
            with self.assertRaises(SupplementError):
                build_supplement(motions_dir=fixture.motions, raw_root=fixture.raw, output_dir=out)
            self.assertFalse(out.exists())
            self.assertFalse(out.with_name(out.name + ".staging").exists())

    def test_verification_separates_round_trip_noise_from_a_real_outlier(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory), corrupt=1)
            existing = {
                json.loads(line)["sequence_id"].split("/")[1]: json.loads(line)
                for line in (fixture.raw / "sequences.jsonl").open()
            }
            report = verify_overlaps(
                fixture.motions, fixture.raw, existing, tolerance=1e-5, min_fraction=0.5
            )
            self.assertTrue(report["transform_confirmed"])
            self.assertEqual(report["within_tolerance"], 1)
            self.assertEqual(len(report["outliers"]), 1)
            self.assertEqual(report["outliers"][0]["sequence"], "gBR_sBM_cAll_d04_mBR0_ch01")

    def test_refuses_to_overwrite_an_existing_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory))
            out = Path(directory) / "supplement"
            out.mkdir()
            with self.assertRaises(SupplementError):
                build_supplement(motions_dir=fixture.motions, raw_root=fixture.raw, output_dir=out)


if __name__ == "__main__":
    unittest.main()
