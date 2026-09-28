"""The ruler has to read a good segmentation differently from a random one.

Every number this probe produced about M1 -- boundary contrast 1.014 against a
random control's 0.993 -- is worth nothing unless the two statistics can tell a
segmentation placed on the motion's turns from one placed anywhere.  So the
tests build a motion whose turns are known by construction and require the
separation, and they require the refusals, because a probe that scores an empty
intersection reports a clean 1.0 while measuring nothing.
"""

import json
import pathlib
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools import probe_segmentation_boundaries as probe


def _motion(blocks, frames_per_block=40, joints=24, seed=0):
    """A 151-D motion that holds still, then jumps to a new pose, repeatedly.

    The turns are exactly at the block boundaries, so a segmentation that cuts
    there is right by construction and one that cuts elsewhere is not.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(blocks):
        pose = rng.normal(size=(joints, 6)) * 0.5
        wobble = rng.normal(size=(frames_per_block, joints, 6)) * 0.002
        rows.append(pose[None] + wobble)
    rot = np.concatenate(rows, axis=0).reshape(blocks * frames_per_block, joints * 6)
    motion = np.zeros((len(rot), 151), np.float32)
    motion[:, 7:] = rot
    return motion, [i * frames_per_block for i in range(blocks + 1)]


def _bundle(root, clips):
    """A raw-bundle-shaped directory: sequences.jsonl + per-sequence arrays."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "sequences").mkdir(exist_ok=True)
    lines = []
    for name, motion in clips.items():
        sha = name.replace(":", "_")
        (root / "sequences" / sha).mkdir(exist_ok=True)
        np.save(root / "sequences" / sha / "motion_151_raw.npy", motion)
        lines.append(json.dumps({
            "recording_id": name,
            "assets": {"motion_151_raw": "sequences/{}/motion_151_raw.npy".format(sha)},
        }))
    (root / "sequences.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _report(path, boundaries_by_sequence):
    records = [{"sequence": name, "boundaries": b}
               for name, b in boundaries_by_sequence.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"records": records}), encoding="utf-8")


class CanonicalKey(unittest.TestCase):
    def test_the_three_spellings_of_one_clip_meet(self):
        self.assertEqual(probe.canonical_key("wild_v4:7010:clip001"), "7010:clip001")
        self.assertEqual(probe.canonical_key("7010__clip001"), "7010:clip001")

    def test_an_unrelated_name_is_left_alone(self):
        self.assertEqual(probe.canonical_key("gBR_sBM_c01_d04_mBR0_ch01"),
                         "gBR_sBM_c01_d04_mBR0_ch01")


class TheRulerSeparates(unittest.TestCase):
    """Without this the M1 reading is unreadable: 1.014 could be a blind ruler."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.clips, self.truth = {}, {}
        for i in range(6):
            motion, edges = _motion(blocks=8, seed=i)
            name = "wild_v4:{}:clip000".format(7000 + i)
            self.clips[name] = motion
            self.truth[name] = edges
        _bundle(self.tmp / "bundle", self.clips)

    def test_cuts_on_the_turns_beat_cuts_of_the_same_lengths_elsewhere(self):
        _report(self.tmp / "true.json", self.truth)
        # Same number of segments, same lengths, shifted by half a block.
        shifted = {k: [max(0, b - 20) for b in v[:-1]] + [v[-1]]
                   for k, v in self.truth.items()}
        _report(self.tmp / "shifted.json", shifted)
        out = probe.probe(
            self.tmp / "bundle",
            {"true": probe.load_arm(self.tmp / "true.json"),
             "shifted": probe.load_arm(self.tmp / "shifted.json")},
            limit=0, seed=1, min_length=18)
        true_c = out["arms"]["true"]["boundary_contrast"]
        shifted_c = out["arms"]["shifted"]["boundary_contrast"]
        self.assertGreater(true_c, 3.0, "a cut on the turn must read well above 1")
        self.assertLess(shifted_c, true_c / 2,
                        "a cut off the turn must read far lower; ruler is blind otherwise")

    def test_within_segment_variance_prefers_the_true_cuts(self):
        _report(self.tmp / "true.json", self.truth)
        out = probe.probe(self.tmp / "bundle",
                          {"true": probe.load_arm(self.tmp / "true.json")},
                          limit=0, seed=1, min_length=18)
        self.assertLess(out["arms"]["true"]["within_vs_random"], 0.9)

    def test_the_positive_control_lands_on_the_turn_not_one_frame_before(self):
        """The control cut at the peak index while the ruler scores the step
        *into* a span, so it used to place its cuts one frame early and report
        itself at chance.  A control that under-reports itself makes every arm
        look closer to the ceiling than it is."""
        _report(self.tmp / "true.json", self.truth)
        out = probe.probe(self.tmp / "bundle",
                          {"true": probe.load_arm(self.tmp / "true.json")},
                          limit=0, seed=1, min_length=18)
        self.assertGreater(out["arms"]["_peaks"]["boundary_contrast"], 3.0)

    def test_the_positive_control_is_computed_and_beats_random(self):
        _report(self.tmp / "true.json", self.truth)
        out = probe.probe(self.tmp / "bundle",
                          {"true": probe.load_arm(self.tmp / "true.json")},
                          limit=0, seed=1, min_length=18)
        self.assertIn("_peaks", out["arms"])
        self.assertGreater(out["arms"]["_peaks"]["boundary_contrast"],
                           out["arms"]["_random"]["boundary_contrast"])


class TheVisualRulerIsTheMirrorOfTheMotionOne(unittest.TestCase):
    """Whether the visual half of a fused cut earns its place is a C1 question.

    Deferring it to whatever clusters downstream would mix "the cuts moved"
    with "the modality mix changed" in one reading, and then a regression
    cannot be attributed to either.  So the same two statistics run in S3D
    feature space, and the test is that they separate there too.
    """

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        (self.tmp / "feat").mkdir()
        self.truth = {}
        for i in range(6):
            rng = np.random.default_rng(100 + i)
            blocks, per = 8, 40
            rows = [np.tile(rng.normal(size=(1, 64)), (per, 1)) for _ in range(blocks)]
            feats = np.concatenate(rows) + rng.normal(size=(blocks * per, 64)) * 0.01
            name = "wild_v4:{}:clip000".format(8000 + i)
            # Real feature files are named <upload>__clipNNN.npz, with no
            # corpus prefix -- canonical_key exists to make that meet the
            # bundle's wild_v4:<upload>:clipNNN.
            np.savez(self.tmp / "feat" / (name.split(":", 1)[1].replace(":", "__") + ".npz"),
                     features=feats.astype(np.float16))
            self.truth[name] = [b * per for b in range(blocks + 1)]

    def test_cuts_on_the_visual_turns_beat_cuts_elsewhere(self):
        _report(self.tmp / "true.json", self.truth)
        shifted = {k: [max(0, b - 20) for b in v[:-1]] + [v[-1]]
                   for k, v in self.truth.items()}
        _report(self.tmp / "shifted.json", shifted)
        out = probe.probe(None,
                          {"true": probe.load_arm(self.tmp / "true.json"),
                           "shifted": probe.load_arm(self.tmp / "shifted.json")},
                          limit=0, seed=1, min_length=18,
                          signal="visual", features_dir=self.tmp / "feat")
        self.assertEqual(out["signal"], "visual")
        self.assertGreater(out["arms"]["true"]["boundary_contrast"], 3.0)
        self.assertLess(out["arms"]["shifted"]["boundary_contrast"],
                        out["arms"]["true"]["boundary_contrast"] / 2)


class ItRefusesRatherThanScoringNothing(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        motion, edges = _motion(blocks=6)
        _bundle(self.tmp / "bundle", {"wild_v4:7000:clip000": motion})
        self.edges = edges

    def test_names_that_do_not_meet_the_bundle_are_refused(self):
        _report(self.tmp / "other.json", {"some_other_corpus_clip": self.edges})
        with self.assertRaises(SystemExit) as caught:
            probe.probe(self.tmp / "bundle",
                        {"other": probe.load_arm(self.tmp / "other.json")},
                        limit=0, seed=1, min_length=18)
        self.assertIn("meet the bundle", str(caught.exception))

    def test_a_missing_bundle_manifest_is_refused(self):
        with self.assertRaises(SystemExit):
            probe.load_bundle_index(self.tmp / "no_such_bundle")


class TheSamplerIsNotAPrefix(unittest.TestCase):
    def test_limit_takes_an_even_stride(self):
        tmp = pathlib.Path(tempfile.mkdtemp())
        clips, arm = {}, {}
        for i in range(20):
            motion, edges = _motion(blocks=5, seed=i)
            name = "wild_v4:{}:clip000".format(7000 + i)
            clips[name] = motion
            arm[name] = edges
        _bundle(tmp / "bundle", clips)
        _report(tmp / "arm.json", arm)
        out = probe.probe(tmp / "bundle", {"a": probe.load_arm(tmp / "arm.json")},
                          limit=5, seed=1, min_length=18)
        self.assertEqual(out["recordings"], 5)
        # 5 of 20 at an even stride is 4 segments-worth of clips apart; a prefix
        # would have scored the first five, which here means the first five seeds.
        self.assertEqual(out["arms"]["a"]["segments"] % 5, 0)


class TheClipListRestricts(unittest.TestCase):
    """--clips has to cut the corpus down BEFORE --limit samples it.

    The reason it exists: ``wild_v4_seg_r0visual`` covers all 13,783 wild_v4
    clips while the beat arms cover only the 2,103 of ``runs/clean5/clips.txt``.
    Scored without a clip list the two arms are read on different footage, and
    the difference in footage reads as a difference between the arms.
    """

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.clips, self.arm = {}, {}
        for i in range(10):
            motion, edges = _motion(blocks=5, seed=i)
            name = "wild_v4:{}:clip000".format(7000 + i)
            self.clips[name] = motion
            self.arm[name] = edges
        _bundle(self.tmp / "bundle", self.clips)
        _report(self.tmp / "arm.json", self.arm)

    def test_only_the_listed_clips_are_scored(self):
        listed = ["7001__clip000", "7003__clip000", "7007__clip000"]
        (self.tmp / "clips.txt").write_text("\n".join(listed) + "\n", encoding="utf-8")
        out = probe.probe(self.tmp / "bundle",
                          {"a": probe.load_arm(self.tmp / "arm.json")},
                          limit=0, seed=1, min_length=18,
                          clips=self.tmp / "clips.txt")
        self.assertEqual(out["recordings"], 3)

    def test_the_restriction_happens_before_the_limit_samples(self):
        listed = ["700{}__clip000".format(i) for i in range(4)]
        (self.tmp / "clips.txt").write_text("\n".join(listed) + "\n", encoding="utf-8")
        out = probe.probe(self.tmp / "bundle",
                          {"a": probe.load_arm(self.tmp / "arm.json")},
                          limit=2, seed=1, min_length=18,
                          clips=self.tmp / "clips.txt")
        # 2 of the 4 listed, not 2 of the 10 in the bundle.
        self.assertEqual(out["recordings"], 2)

    def test_a_clip_list_that_meets_nothing_is_refused_not_scored_empty(self):
        (self.tmp / "clips.txt").write_text("nothing__clip000\n", encoding="utf-8")
        with self.assertRaises(SystemExit):
            probe.probe(self.tmp / "bundle",
                        {"a": probe.load_arm(self.tmp / "arm.json")},
                        limit=0, seed=1, min_length=18,
                        clips=self.tmp / "clips.txt")


if __name__ == "__main__":
    unittest.main()
