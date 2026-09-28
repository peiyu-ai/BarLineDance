"""The verdict logic must fail where 2026-08-18 failed by hand.

Each test here is one of the four incomplete-criteria failures that produced an
overturned conclusion that day, reduced to the smallest input that reproduces
it.  A change to judge_arm that loses any of these has broken the gate.
"""

import json
import pathlib
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools import judge_arm


def _write_features(root, family, names, matrix):
    directory = pathlib.Path(root) / judge_arm.FAMILY_DIR[family]
    directory.mkdir(parents=True, exist_ok=True)
    for name, row in zip(names, matrix):
        np.save(directory / "{}.npy".format(name), row.astype(np.float32))


def _stats(delta, p, baseline=10.0):
    return {"observed_delta": delta, "p_one_sided": p, "null_sd": 0.1,
            "permutations": 10, "arm_fid": baseline - delta, "baseline_fid": baseline}


class FidAgreementTests(unittest.TestCase):
    def test_fast_fid_matches_eval_metrics(self):
        """The eigenvalue path is only usable while it reproduces the published number."""
        rng = np.random.default_rng(7)
        prediction = rng.normal(size=(60, 12))
        truth = rng.normal(loc=0.4, scale=1.3, size=(80, 12))
        reference = judge_arm.check_against_reference(prediction, truth)
        self.assertAlmostEqual(judge_arm.fid(prediction, truth), reference, places=9)

    def test_fid_is_invariant_to_row_order(self):
        """Recorded here because it is why per-clip proxies cannot predict fid.

        Permuting the generated rows destroys every generated-to-ground-truth
        pairing and leaves fid unchanged, so a rule selected on per-clip
        closeness has no monotone claim on this number.
        """
        rng = np.random.default_rng(11)
        prediction = rng.normal(size=(50, 8))
        truth = rng.normal(size=(50, 8))
        shuffled = prediction[rng.permutation(len(prediction))]
        self.assertAlmostEqual(judge_arm.fid(prediction, truth),
                               judge_arm.fid(shuffled, truth), places=10)


class ClipKeyTests(unittest.TestCase):
    def test_seed_suffix_is_stripped_but_other_underscores_survive(self):
        self.assertEqual(judge_arm.clip_of("wild_v4:700_clip001_s20260816"),
                         "wild_v4:700_clip001")
        self.assertEqual(judge_arm.clip_of("mBR2_seq_alpha"), "mBR2_seq_alpha")

    def test_alignment_refuses_arms_that_do_not_overlap(self):
        with self.assertRaises(judge_arm.JudgeError):
            judge_arm.align(["a", "b"], np.zeros((2, 3)), ["c", "d"], np.zeros((2, 3)))

    def test_alignment_pairs_by_name_not_by_position(self):
        a = np.array([[1.0], [2.0]])
        b = np.array([[20.0], [10.0]])
        shared, left, right = judge_arm.align(["x", "y"], a, ["y", "x"], b)
        self.assertEqual(shared, ["x", "y"])
        self.assertEqual(right.ravel().tolist(), [10.0, 20.0])


class PermutationTests(unittest.TestCase):
    def test_no_effect_gives_a_large_p(self):
        rng = np.random.default_rng(3)
        truth = rng.normal(size=(60, 6))
        arm = rng.normal(size=(60, 6))
        baseline = arm + rng.normal(scale=1e-3, size=arm.shape)
        stats = judge_arm.paired_permutation(arm, baseline, truth, permutations=200, seed=1)
        self.assertGreater(stats["p_one_sided"], 0.05)

    def test_a_real_shift_gives_a_small_p(self):
        """The difference has to be one fid can see: correlation structure.

        A first version of this test made the baseline `normal * 3 + 5` and the
        permutation correctly refused it -- standardisation removes a
        per-dimension scale and shift, so both arms were the same distribution
        and the observed delta (0.096) sat well inside the null sd (0.336).
        That is the invariance, not a weak test.
        """
        rng = np.random.default_rng(4)
        truth = rng.normal(size=(80, 6))
        arm = truth + rng.normal(scale=0.05, size=truth.shape)
        factor = rng.normal(size=(80, 1))
        baseline = np.repeat(factor, 6, axis=1) + rng.normal(scale=0.05, size=(80, 6))
        stats = judge_arm.paired_permutation(arm, baseline, truth, permutations=200, seed=1)
        self.assertGreater(stats["observed_delta"], 0.0)
        self.assertLess(stats["p_one_sided"], 0.05)

    def test_a_per_dimension_affine_change_is_invisible_to_the_gate(self):
        """Pinned because it is why roughness and amplitude need their own axes."""
        rng = np.random.default_rng(19)
        truth = rng.normal(size=(60, 7))
        arm = rng.normal(size=(60, 7))
        scaled = arm * rng.uniform(0.1, 10.0, size=7) + rng.normal(scale=100.0, size=7)
        self.assertAlmostEqual(judge_arm.fid(arm, truth),
                               judge_arm.fid(scaled, truth), places=8)


class VerdictTests(unittest.TestCase):
    """Order of criteria: band, then family agreement, then p."""

    BAND = {"kinetic": 0.027, "manual": 0.030}
    BASE = {"kinetic": 3.55, "manual": 3.90}

    def judge(self, kinetic, manual):
        return judge_arm.verdict_for_set({"kinetic": kinetic, "manual": manual},
                                         self.BAND, self.BASE, 0.05)

    def test_families_disagreeing_outside_their_bands_blocks_a_verdict(self):
        """The `random` retrieval arm: fid_k improved, fid_m did not."""
        out = self.judge(_stats(+0.22, 0.01, 3.55), _stats(-0.49, 0.01, 3.90))
        self.assertEqual(out["verdict"], "SPLIT_FAMILIES")

    def test_disagreeing_signs_inside_the_bands_are_not_a_split(self):
        """`gapfill interpolate`: +0.45% and -0.89%, i.e. two re-draws of nothing.

        Reported as SPLIT_FAMILIES by the first version of this function.  A
        family whose delta is inside its own band carries no sign to disagree
        with, so the band has to be tested first.
        """
        out = self.judge(_stats(+0.0160, 0.01, 3.55), _stats(-0.0347, 0.01, 3.90))
        self.assertEqual(out["verdict"], "UNDETERMINED")
        self.assertIn("inside their seed-noise bands", out["reasons"][0])

    def test_a_large_move_is_not_swallowed_by_a_small_opposite_drift(self):
        """`no-plan` at seed 20260817: kinetic -41%, manual +4.8% inside its band."""
        out = self.judge(_stats(-1.4208, 0.001, 3.45), _stats(+0.1000, 0.4, 3.90))
        self.assertEqual(out["verdict"], "REGRESSED")
        self.assertTrue(any("drifted the other way" in r for r in out["reasons"]))

    def test_a_large_delta_with_a_weak_p_is_undetermined(self):
        out = self.judge(_stats(+0.40, 0.206, 3.55), _stats(+0.30, 0.3, 3.90))
        self.assertEqual(out["verdict"], "UNDETERMINED")
        self.assertIn("permutation", out["reasons"][0])

    def test_a_clear_improvement_passes(self):
        out = self.judge(_stats(+0.60, 0.001, 3.55), _stats(+0.40, 0.01, 3.90))
        self.assertEqual(out["verdict"], "IMPROVED")

    def test_manual_decides_alone_when_kinetic_did_not_move(self):
        """A pose-geometry change with no velocity change is still a result."""
        out = self.judge(_stats(+0.0100, 0.5, 3.55), _stats(+0.60, 0.002, 3.90))
        self.assertEqual(out["verdict"], "IMPROVED")
        self.assertIn("manual decides", out["reasons"][0])


class CombineTests(unittest.TestCase):
    def test_sets_disagreeing_is_undetermined(self):
        """SELF medoid: +5.18% on the leak-free set, -0.49% on the full one."""
        sets = {"clean": {"verdict": "REGRESSED", "reasons": ["clean says worse"]},
                "all": {"verdict": "IMPROVED", "reasons": ["all says better"]}}
        out = judge_arm.combine(sets, None)
        self.assertEqual(out["verdict"], "UNDETERMINED")
        self.assertTrue(any("sets disagree" in r for r in out["reasons"]))

    def test_regression_with_roughness_moving_the_other_way_is_flagged(self):
        """dn005: every visible axis toward ground truth, fid_k +64%."""
        sets = {"clean": {"verdict": "REGRESSED", "reasons": ["fid regressed"]}}
        movement = {"toward_ground_truth": ["jerk", "ankle_speed", "across_joint_cv"],
                    "away_from_ground_truth": ["reach_span"],
                    "unchanged": [], "net": 2}
        out = judge_arm.combine(sets, movement)
        self.assertEqual(out["verdict"], "AXES_DISAGREE")

    def test_a_plain_regression_stays_a_regression(self):
        sets = {"clean": {"verdict": "REGRESSED", "reasons": ["fid regressed"]}}
        movement = {"toward_ground_truth": [], "away_from_ground_truth": ["jerk"],
                    "unchanged": [], "net": -1}
        self.assertEqual(judge_arm.combine(sets, movement)["verdict"], "REGRESSED")


class RoughnessTests(unittest.TestCase):
    def _report(self, jerk, cv, reach):
        return {"summary": {"jerk": {"median": jerk},
                            "speed": {"median_of_all": 1.0, "ankle": 1.0,
                                      "across_joint_cv": cv, "wrist_over_pelvis": 2.0},
                            "amplitude": {"floor_path_m_per_s": 1.0,
                                          "reach_span_shoulders": reach,
                                          "turn_rate_deg_s": 45.0}},
                "ground_truth": {"jerk": {"median": 360.0},
                                 "speed": {"median_of_all": 1.0, "ankle": 1.0,
                                           "across_joint_cv": 0.584, "wrist_over_pelvis": 3.44},
                                 "amplitude": {"floor_path_m_per_s": 1.0,
                                               "reach_span_shoulders": 0.351,
                                               "turn_rate_deg_s": 45.0}}}

    def test_ratios_are_against_each_run_own_ground_truth(self):
        ratios = judge_arm.roughness_ratios(self._report(1954.0, 0.294, 0.376))
        self.assertAlmostEqual(ratios["jerk"], 1954.0 / 360.0, places=6)
        self.assertAlmostEqual(ratios["across_joint_cv"], 0.294 / 0.584, places=6)

    def test_movement_counts_axes_by_distance_to_one(self):
        baseline = judge_arm.roughness_ratios(self._report(1954.0, 0.294, 0.376))
        arm = judge_arm.roughness_ratios(self._report(1102.0, 0.381, 0.551))
        movement = judge_arm.roughness_movement(arm, baseline)
        self.assertIn("jerk", movement["toward_ground_truth"])
        self.assertIn("across_joint_cv", movement["toward_ground_truth"])
        self.assertIn("reach_span", movement["away_from_ground_truth"])

    def test_a_missing_axis_is_skipped_not_counted_as_unchanged(self):
        report = self._report(1954.0, 0.294, 0.376)
        del report["summary"]["jerk"]
        ratios = judge_arm.roughness_ratios(report)
        self.assertIsNone(ratios["jerk"])
        movement = judge_arm.roughness_movement(ratios, ratios)
        self.assertNotIn("jerk", movement["unchanged"])


class EndToEndTests(unittest.TestCase):
    def test_run_produces_a_verdict_from_feature_directories(self):
        rng = np.random.default_rng(21)
        names = ["clip{:03d}_s20260816".format(i) for i in range(60)]
        truth = rng.normal(size=(60, 5))
        # The baseline differs in correlation structure, not in per-dimension
        # scale: a scale-and-shift baseline is invisible to fid (see
        # test_a_per_dimension_affine_change_is_invisible_to_the_gate) and the
        # gate would correctly return UNDETERMINED on it.
        factor = rng.normal(size=(60, 1))
        baseline = np.repeat(factor, 5, axis=1) + rng.normal(scale=0.05, size=(60, 5))
        arm = truth + rng.normal(scale=0.05, size=truth.shape)
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            for family in judge_arm.FAMILIES:
                _write_features(root / "arm", family, names, arm)
                _write_features(root / "base", family, names, baseline)
                _write_features(root / "gt", family, names, truth)
            options = judge_arm.build_parser().parse_args([
                "--arm", str(root / "arm"), "--baseline", str(root / "base"),
                "--ground-truth", str(root / "gt"), "--permutations", "300",
                "--no-verify-fid", "--name", "smoke",
            ])
            report = judge_arm.run(options)
        self.assertEqual(report["verdict"], "IMPROVED")
        self.assertEqual(report["sets"]["all"]["clips"], 60)
        self.assertIn("fid_k", report["sets"]["all"]["fid"])
        self.assertIn("fid_m", report["sets"]["all"]["fid"])

    def test_missing_feature_family_refuses_rather_than_scoring_one(self):
        rng = np.random.default_rng(5)
        names = ["clip{:03d}".format(i) for i in range(6)]
        block = rng.normal(size=(6, 4))
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            for part in ("arm", "base", "gt"):
                _write_features(root / part, "kinetic", names, block)
            options = judge_arm.build_parser().parse_args([
                "--arm", str(root / "arm"), "--baseline", str(root / "base"),
                "--ground-truth", str(root / "gt"), "--permutations", "10",
                "--no-verify-fid",
            ])
            with self.assertRaises(judge_arm.JudgeError):
                judge_arm.run(options)


if __name__ == "__main__":
    unittest.main()
