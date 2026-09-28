"""The split-leakage audit has to fail on a crossed split and pass on a clean one.

Both directions are tested, and separately for the two readings the tool makes,
because they fail independently: the declared reading cannot see a pair the
fingerprint list omits, and that omission is exactly what the measured reading
exists to catch.  A version of this check that only tested "it reports leakage
when the pair list says so" would pass while being unable to discover anything.
"""

from __future__ import annotations

import json
import pathlib
import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools import audit_split_leakage as audit


def _motion(frames, seed, scale=1.0):
    """A 151-D array whose joints move smoothly, distinct per seed."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(1, 151)) * 0.1
    phase = np.linspace(0, 4 * np.pi, frames)[:, None]
    wave = np.sin(phase + rng.normal(size=(1, 151))) * 0.05 * scale
    return (base + wave).astype(np.float32)


def _bundle(tmp_path, rows):
    """A minimal performance bundle: sequences.jsonl plus the arrays it names."""
    root = tmp_path / "bundle"
    (root / "sequences").mkdir(parents=True, exist_ok=True)
    with (root / "sequences.jsonl").open("w", encoding="utf-8") as handle:
        for name, split, motion in rows:
            relative = "sequences/{}.npy".format(name.replace(":", "_"))
            np.save(str(root / relative), motion)
            handle.write(json.dumps({
                "recording_id": name, "split": split, "motion_path": relative,
                "frame_count": len(motion), "retrieval_group_id": name.rsplit(":", 1)[0],
            }) + "\n")
    return root


def _pairs(tmp_path, pairs):
    path = tmp_path / "pairs.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for left, right in pairs:
            handle.write(json.dumps({"left": left, "right": right,
                                     "score": 0.99, "lag_frames": 0}) + "\n")
    return path


def test_pose_distance_refuses_a_reading_built_from_too_few_frames():
    left = np.zeros((300, 24, 3))
    right = np.zeros((300, 24, 3))
    assert audit.pose_distance(left, right)[0] == pytest.approx(0.0)
    # A lag that leaves less than the floor of overlap is not a small number, it
    # is not a number: returning 0.0 here would read as a perfect match.
    assert audit.pose_distance(left, right, lag=280) == (None, 0)


def test_root_translation_cannot_score():
    """Two clips of one routine filmed from different places must read as close."""
    rng = np.random.default_rng(0)
    pose = rng.normal(size=(200, 24, 3))
    shifted = pose + np.array([5.0, -3.0, 0.0])
    assert audit.pose_distance(pose, shifted)[0] == pytest.approx(0.0, abs=1e-9)


def test_best_over_lag_finds_an_offset_copy_and_the_sign_is_pinned():
    """Both directions, because a lag sign that flips still returns a plausible number."""
    rng = np.random.default_rng(1)
    pose = rng.normal(size=(400, 24, 3))
    # left starts 40 frames later than right: left[t] == right[t + 40], and the
    # convention is right index = t - lag, so the lag is negative.
    value, _, lag = audit.best_over_lag(pose[40:], pose)
    assert value == pytest.approx(0.0, abs=1e-9)
    assert lag == -40
    # the mirror image has to come back with the opposite sign, not the same one
    value, _, lag = audit.best_over_lag(pose, pose[40:])
    assert value == pytest.approx(0.0, abs=1e-9)
    assert lag == 40


def test_components_keep_singletons_and_merge_chains():
    adjacency = {"a": {"b"}, "b": {"a", "c"}, "c": {"b"}}
    groups = audit.components(adjacency, {"a", "b", "c", "d"})
    assert sorted(len(g) for g in groups) == [1, 3]


def test_declared_reading_fails_on_a_crossed_split_and_passes_on_a_clean_one(tmp_path):
    records = {
        "w:1:c0": {"recording_id": "w:1:c0", "split": "train"},
        "w:2:c0": {"recording_id": "w:2:c0", "split": "test"},
        "w:3:c0": {"recording_id": "w:3:c0", "split": "train"},
    }
    adjacency = {"w:1:c0": {"w:2:c0"}, "w:2:c0": {"w:1:c0"}}
    crossed = audit.declared_leakage(records, adjacency)
    assert crossed["straddling_groups"] == 1
    assert crossed["recordings_that_must_move"] == 1

    records["w:2:c0"]["split"] = "train"
    clean = audit.declared_leakage(records, adjacency)
    assert clean["straddling_groups"] == 0
    assert clean["recordings_that_must_move"] == 0


def _corpus(tmp_path, held_out=4, train=30, seed0=100):
    """A synthetic corpus big enough that a per-query ratio has a denominator."""
    rows = [("w:t{}:c0".format(i), "train", _motion(300, seed=seed0 + i))
            for i in range(train)]
    rows += [("w:h{}:c0".format(i), "test", _motion(300, seed=seed0 + 500 + i))
             for i in range(held_out)]
    return rows


def test_measured_reading_finds_a_pair_the_fingerprint_list_omits(tmp_path):
    """The recall check: a leak that is real and undeclared has to be surfaced."""
    rows = _corpus(tmp_path)
    twin = {name: motion for name, _, motion in rows}["w:t0:c0"]
    # one held-out clip is a near-copy of a training clip, and the pair list is
    # deliberately empty -- the declared reading cannot see it, the measured one must
    rows = [r for r in rows if r[0] != "w:h0:c0"]
    rows.append(("w:h0:c0", "test", twin + np.float32(0.0005)))
    root = _bundle(tmp_path, rows)
    pairs = _pairs(tmp_path, [])
    output = tmp_path / "report.json"
    code = audit.main(["--bundle", str(root), "--pairs", str(pairs),
                       "--output", str(output), "--test-sample", "4",
                       "--train-sample", "30", "--calibration-sample", "20",
                       "--threshold", "0.5", "--policy", "refuse"])
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["declared"]["straddling_groups"] == 0        # the list says clean
    flagged = report["measured"]["undeclared_examples"]
    assert report["measured"]["flagged"] >= 1                  # the poses say otherwise
    assert any(r["recording_id"] == "w:h0:c0" and r["nearest_train"] == "w:t0:c0"
               for r in flagged)
    assert code == 1, "refuse must exit non-zero when leakage is measured"


def test_a_corpus_with_no_copy_is_not_flagged(tmp_path):
    """The ratio must not fire just because some neighbour is nearest -- one always is."""
    root = _bundle(tmp_path, _corpus(tmp_path))
    pairs = _pairs(tmp_path, [])
    output = tmp_path / "report.json"
    code = audit.main(["--bundle", str(root), "--pairs", str(pairs),
                       "--output", str(output), "--test-sample", "4",
                       "--train-sample", "30", "--calibration-sample", "20",
                       "--threshold", "0.5", "--policy", "refuse"])
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["measured"]["held_out_checked"] == 4
    assert report["measured"]["flagged"] == 0
    assert code == 0


def test_a_hub_is_not_mistaken_for_a_copy(tmp_path):
    """A near-static reconstruction is close to everyone and must not flag everyone.

    Measured on the corpus before this correction existed: one clip whose joint
    speed was a sixth of the corpus median was the nearest neighbour of 53 of
    200 held-out recordings.
    """
    rows = _corpus(tmp_path, held_out=8, train=30)
    # replace one training clip with a near-motionless one: it sits near the mean
    # pose, so it is the closest thing to every query without copying any of them
    rows = [r for r in rows if r[0] != "w:t0:c0"]
    rows.append(("w:t0:c0", "train", _motion(300, seed=1, scale=0.02)))
    root = _bundle(tmp_path, rows)
    pairs = _pairs(tmp_path, [])
    output = tmp_path / "report.json"
    audit.main(["--bundle", str(root), "--pairs", str(pairs), "--output", str(output),
                "--test-sample", "8", "--train-sample", "30", "--calibration-sample", "20",
                "--threshold", "0.5"])
    measured = json.loads(output.read_text(encoding="utf-8"))["measured"]
    hub_rows = [r for r in measured["rows"] if r["nearest_train"] == "w:t0:c0"]
    assert hub_rows, "the hub should still be somebody's nearest neighbour"
    assert all(r["candidate_is_near_static"] for r in hub_rows)
    assert measured["flagged"] == 0, "hubness must not be reported as leakage"
    assert measured["near_static_in_pool"] >= 1


def test_the_time_reversed_null_is_reported_and_can_veto_the_forward_reading(tmp_path):
    """A flag the reversed search also produces is an artifact, and must not convict.

    Reversing a candidate keeps its poses, its joint speeds and its length and
    destroys the order -- the only thing a choreography leak is made of.
    """
    rows = _corpus(tmp_path, held_out=8, train=30)
    twin = {name: motion for name, _, motion in rows}["w:t0:c0"]
    rows = [r for r in rows if r[0] != "w:h0:c0"]
    rows.append(("w:h0:c0", "test", twin + np.float32(0.0005)))
    root = _bundle(tmp_path, rows)
    pairs = _pairs(tmp_path, [])
    output = tmp_path / "report.json"
    code = audit.main(["--bundle", str(root), "--pairs", str(pairs), "--output", str(output),
                       "--test-sample", "8", "--train-sample", "30",
                       "--calibration-sample", "20", "--threshold", "0.5",
                       "--policy", "refuse"])
    measured = json.loads(output.read_text(encoding="utf-8"))["measured"]
    # a real copy survives reversal of the candidate, so the excess is positive
    assert measured["flagged"] >= 1
    assert measured["excess_over_time_reversed_null"] >= 1
    assert measured["reading_is_at_its_noise_floor"] is False
    assert code == 1


def test_a_reading_matched_by_its_own_null_does_not_convict(tmp_path):
    """No copy anywhere: forward and reversed must agree, and refuse must not fire."""
    root = _bundle(tmp_path, _corpus(tmp_path, held_out=8, train=30))
    pairs = _pairs(tmp_path, [])
    output = tmp_path / "report.json"
    code = audit.main(["--bundle", str(root), "--pairs", str(pairs), "--output", str(output),
                       "--test-sample", "8", "--train-sample", "30",
                       "--calibration-sample", "20", "--threshold", "0.5",
                       "--policy", "refuse"])
    measured = json.loads(output.read_text(encoding="utf-8"))["measured"]
    assert measured["excess_over_time_reversed_null"] <= 0
    assert code == 0


def test_skipped_recordings_are_reported_rather_than_counted_clean(tmp_path):
    """"We found none" and "we could not look" must not print the same."""
    rows = _corpus(tmp_path, held_out=1, train=30)
    rows = [r for r in rows if r[0] != "w:h0:c0"]
    rows.append(("w:h0:c0", "test", _motion(20, seed=2)))     # too short to compare
    root = _bundle(tmp_path, rows)
    pairs = _pairs(tmp_path, [])
    output = tmp_path / "report.json"
    audit.main(["--bundle", str(root), "--pairs", str(pairs), "--output", str(output),
                "--test-sample", "1", "--train-sample", "30",
                "--calibration-sample", "20", "--threshold", "0.5"])
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["measured"]["skipped_too_few_comparisons"] == 1
    assert report["measured"]["held_out_checked"] == 0


def test_threshold_comes_from_the_negative_control_not_a_constant(tmp_path):
    root = _bundle(tmp_path, _corpus(tmp_path, held_out=12, train=30))
    pairs = _pairs(tmp_path, [])
    output = tmp_path / "report.json"
    audit.main(["--bundle", str(root), "--pairs", str(pairs), "--output", str(output),
                "--test-sample", "12", "--train-sample", "30", "--calibration-sample", "20"])
    report = json.loads(output.read_text(encoding="utf-8"))
    measured = report["measured"]
    assert measured["threshold"] == pytest.approx(
        measured["negative_control"]["ratio_p05"], abs=1e-4)


def test_a_criterion_that_cannot_discriminate_says_so(tmp_path):
    """With no positive control there is nothing to justify the count, and it must show."""
    root = _bundle(tmp_path, _corpus(tmp_path, held_out=12, train=30))
    pairs = _pairs(tmp_path, [])
    output = tmp_path / "report.json"
    audit.main(["--bundle", str(root), "--pairs", str(pairs), "--output", str(output),
                "--test-sample", "12", "--train-sample", "30", "--calibration-sample", "20"])
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["measured"]["controls_separate"] is False
    assert report["measured"]["positive_control"]["queries"] == 0


def _generated(tmp_path, name, motion, plan=None):
    directory = tmp_path / "generated"
    directory.mkdir(exist_ok=True)
    blob = {"full_pose": motion}
    if plan is not None:
        blob["atomic_labels"] = plan
    with (directory / "{}_s20260816.pkl".format(name)).open("wb") as handle:
        pickle.dump(blob, handle)
    return directory


def test_model_reading_separates_reproducing_its_own_clip_from_reproducing_the_twin(tmp_path):
    """The acceptance test: a generation that copies the training twin must show it."""
    own = _motion(300, seed=21)
    twin = own + np.float32(0.002)
    rows = [("w:1:c0", "test", own), ("w:2:c0", "train", twin),
            ("w:3:c0", "train", _motion(300, seed=31)),
            ("w:4:c0", "train", _motion(300, seed=41))]
    root = _bundle(tmp_path, rows)
    bundle_root, records = audit.read_bundle(root)
    adjacency, _, _ = audit.read_pairs(_pairs(tmp_path, [("w:1:c0", "w:2:c0")]))
    cache = {}
    joints_twin = audit.joints_for(bundle_root, records["w:2:c0"], cache)
    generated = {"w:1:c0": (joints_twin, None)}       # the generation IS the twin
    result = audit.model_leakage(bundle_root, records, adjacency, generated, {}, cache)
    row = result["rows"][0]
    assert row["has_train_twin"] is True
    assert row["distance_to_train_twin"] < row["distance_to_own"]
    assert result["with_train_twin"]["closer_to_train_twin_than_own"] == 1


def test_plan_agreement_reads_the_twin_and_the_clip_apart(tmp_path):
    own_track = np.array([1] * 100 + [2] * 100)
    twin_track = np.array([5] * 200)
    plan = np.array([5] * 200)                       # the planner replayed the twin
    assert audit.label_agreement(plan, twin_track) == pytest.approx(1.0)
    assert audit.label_agreement(plan, own_track) == pytest.approx(0.0)
    assert audit.label_agreement(plan, np.full(200, -1)) is None


# --------------------------------------------------------------------------- #
# The twin cohort.  The acceptance test reads a group that a *correct* re-split
# empties, so the group has to be carried in from the corpus being disproved
# rather than derived from the corpus being accepted.
# --------------------------------------------------------------------------- #


def _leaky_and_fixed(tmp_path):
    """The same four recordings under two splits, with two generation prefixes.

    Under ``v4`` the twin of the held-out clip is in train (that is the leak).
    Under ``v5`` the fingerprint pair sits wholly inside test, which is what a
    song-disjoint split does -- and is exactly the state in which the old
    derivation reported nothing.
    """
    own, twin = _motion(300, seed=21), _motion(300, seed=21) + np.float32(0.002)
    other = [_motion(300, seed=31), _motion(300, seed=41)]
    leaky = _bundle(tmp_path / "v4", [
        ("wild_v4:1:c0", "test", own), ("wild_v4:2:c0", "train", twin),
        ("wild_v4:3:c0", "train", other[0]), ("wild_v4:4:c0", "train", other[1])])
    fixed = _bundle(tmp_path / "v5", [
        ("wild_v5:1:c0", "test", own), ("wild_v5:2:c0", "test", twin),
        ("wild_v5:3:c0", "train", other[0]), ("wild_v5:4:c0", "train", other[1])])
    return leaky, fixed


def test_the_derived_group_is_empty_on_a_song_disjoint_split(tmp_path):
    """Reproduce the defect the cohort exists for, so the fix has a baseline."""
    _leaky, fixed = _leaky_and_fixed(tmp_path)
    bundle_root, records = audit.read_bundle(fixed)
    adjacency, _, _ = audit.read_pairs(
        _pairs(tmp_path, [("wild_v5:1:c0", "wild_v5:2:c0")]))
    cache = {}
    generated = {"wild_v5:1:c0": (
        audit.joints_for(bundle_root, records["wild_v5:2:c0"], cache), None)}
    result = audit.model_leakage(bundle_root, records, adjacency, generated, {}, cache)
    # The generation *is* the twin, and the reading says nothing about it.
    assert result["with_train_twin"]["clips"] == 0
    assert result["twin_source"] == "current split"


def test_a_carried_cohort_reads_the_group_the_re_split_emptied(tmp_path):
    """Same generation, same corpus -- now the copy is visible, across prefixes."""
    leaky, fixed = _leaky_and_fixed(tmp_path)
    (tmp_path / "p4").mkdir(); (tmp_path / "p5").mkdir()
    pairs_v4 = _pairs(tmp_path / "p4", [("wild_v4:1:c0", "wild_v4:2:c0")])
    cohort_path = tmp_path / "cohort.json"
    audit.main(["--bundle", str(leaky), "--pairs", str(pairs_v4),
                "--emit-twin-cohort", str(cohort_path),
                "--output", str(tmp_path / "v4.json"),
                "--test-sample", "1", "--train-sample", "3", "--calibration-sample", "4"])
    emitted = json.loads(cohort_path.read_text(encoding="utf-8"))
    assert emitted["cohort"] == {"wild_v4:1:c0": ["wild_v4:2:c0"]}

    bundle_root, records = audit.read_bundle(fixed)
    adjacency, _, _ = audit.read_pairs(
        _pairs(tmp_path / "p5", [("wild_v5:1:c0", "wild_v5:2:c0")]))
    cohort, stats = audit.read_twin_cohort(cohort_path, records)
    assert cohort == {"wild_v5:1:c0": ["wild_v5:2:c0"]}   # prefix stripped, ids remapped
    assert stats["clips_resolved"] == 1
    cache = {}
    generated = {"wild_v5:1:c0": (
        audit.joints_for(bundle_root, records["wild_v5:2:c0"], cache), None)}
    result = audit.model_leakage(bundle_root, records, adjacency, generated, {}, cache,
                                 cohort=cohort)
    row = result["rows"][0]
    assert row["has_train_twin"] is True          # it used to
    assert row["twin_in_train_now"] is False      # and no longer does
    assert result["clips_with_a_twin_in_train_now"] == 0
    assert result["twin_source"] == "cohort"
    assert result["with_train_twin"]["closer_to_train_twin_than_own"] == 1


def test_a_cohort_that_matches_nothing_is_refused_not_reported_as_clean(tmp_path):
    """An unresolvable cohort reads exactly like a passing one; it must not."""
    _leaky, fixed = _leaky_and_fixed(tmp_path)
    _bundle_root, records = audit.read_bundle(fixed)
    stray = tmp_path / "stray.json"
    stray.write_text(json.dumps({"cohort": {"wild_v4:999:c0": ["wild_v4:998:c0"]}}),
                     encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        audit.read_twin_cohort(stray, records)
    assert "resolves onto no recording" in str(excinfo.value)


def test_a_cohort_reports_what_the_recut_dropped_rather_than_shrinking_quietly(tmp_path):
    """Half the cohort survives; the missing half is counted, not merged away."""
    _leaky, fixed = _leaky_and_fixed(tmp_path)
    _bundle_root, records = audit.read_bundle(fixed)
    path = tmp_path / "partial.json"
    path.write_text(json.dumps({"cohort": {"wild_v4:1:c0": ["wild_v4:2:c0"],
                                           "wild_v4:7:c0": ["wild_v4:2:c0"],
                                           "wild_v4:3:c0": ["wild_v4:9:c0"]}}),
                    encoding="utf-8")
    cohort, stats = audit.read_twin_cohort(path, records)
    assert set(cohort) == {"wild_v5:1:c0"}
    assert stats["entries_read"] == 3
    assert stats["clips_not_in_this_bundle"] == 2      # 7:c0 absent, 3:c0 lost its only twin
    assert stats["twin_recordings_not_in_this_bundle"] == 1


def test_emitting_from_the_corpus_that_fixed_the_split_is_refused(tmp_path):
    """Emit is only meaningful on the leaky corpus; on the fixed one it is empty.

    An empty cohort file would resolve onto nothing later and read like a pass,
    so the refusal happens where the mistake is made rather than one stage on.
    """
    leaky, fixed = _leaky_and_fixed(tmp_path)
    (tmp_path / "p4").mkdir(); (tmp_path / "p5").mkdir()
    ok = audit.main(["--bundle", str(leaky),
                     "--pairs", str(_pairs(tmp_path / "p4", [("wild_v4:1:c0", "wild_v4:2:c0")])),
                     "--emit-twin-cohort", str(tmp_path / "a.json"),
                     "--output", str(tmp_path / "a-report.json")])
    bad = audit.main(["--bundle", str(fixed),
                      "--pairs", str(_pairs(tmp_path / "p5", [("wild_v5:1:c0", "wild_v5:2:c0")])),
                      "--emit-twin-cohort", str(tmp_path / "b.json"),
                      "--output", str(tmp_path / "b-report.json")])
    assert ok == 0 and bad == 1


# --------------------------------------------------------------------------- #
# "plan vs any training recording".  The twin group is thin after a correct
# re-split (21 clips on wild_v5), so the acceptance criterion also has to be
# readable on every generated clip -- with a null, because a max over N is an
# extreme-value statistic and this file already records what comparing one to a
# median cost (64 of 80 recordings convicted).
# --------------------------------------------------------------------------- #


def _corpus_with_tracks(tmp_path, plan, memorised):
    """One held-out clip plus four training recordings, with label tracks.

    ``memorised`` is the training track the plan is built to replay (or None).
    """
    rows = [("wild_v5:1:c0", "test", _motion(300, seed=21))]
    rows += [("wild_v5:{}:c0".format(i), "train", _motion(300, seed=30 + i))
             for i in range(2, 6)]
    root = _bundle(tmp_path, rows)
    bundle_root, records = audit.read_bundle(root)
    tracks = {"wild_v5:2:c0": memorised if memorised is not None
              else np.array([7] * 120 + [8] * 80)}
    tracks.update({"wild_v5:{}:c0".format(i):
                   np.array([i] * 100 + [i + 10] * 100) for i in range(3, 6)})
    tracks["wild_v5:1:c0"] = np.array([1] * 200)
    cache = {}
    generated = {"wild_v5:1:c0": (
        audit.joints_for(bundle_root, records["wild_v5:1:c0"], cache), plan)}
    return bundle_root, records, tracks, cache, generated


def test_a_plan_that_replays_a_training_track_beats_its_own_reversed_null(tmp_path):
    memorised = np.array([4] * 60 + [9] * 140)       # asymmetric, so reversing bites
    _b, records, tracks, cache, generated = _corpus_with_tracks(
        tmp_path, plan=memorised.copy(), memorised=memorised)
    result = audit.model_leakage(_b, records, {}, generated, tracks, cache)
    row = result["rows"][0]
    assert row["plan_vs_best_train"] == pytest.approx(1.0)
    assert row["plan_vs_best_train_reversed"] < 1.0
    summary = result["all_clips"]
    assert summary["plan_excess_over_reversed_median"] > 0
    assert summary["clips_whose_plan_beats_its_own_reversed_null"] == 1


def test_a_plan_the_reversed_null_matches_is_not_counted_as_evidence(tmp_path):
    """A palindromic track agrees forwards and backwards -- that is the floor.

    Without the null this reads exactly like the memorising case above: a
    perfect 1.0 against a training recording.
    """
    palindrome = np.array([3] * 200)
    _b, records, tracks, cache, generated = _corpus_with_tracks(
        tmp_path, plan=palindrome.copy(), memorised=palindrome)
    result = audit.model_leakage(_b, records, {}, generated, tracks, cache)
    row = result["rows"][0]
    assert row["plan_vs_best_train"] == pytest.approx(1.0)          # looks damning
    assert row["plan_vs_best_train_reversed"] == pytest.approx(1.0)  # and is not
    assert result["all_clips"]["plan_excess_over_reversed_median"] == pytest.approx(0.0)
    assert result["all_clips"]["clips_whose_plan_beats_its_own_reversed_null"] == 0


def test_the_broad_reading_scores_every_clip_not_only_the_twins(tmp_path):
    """The point of it: no cohort, no fingerprint pair, still a number."""
    _b, records, tracks, cache, generated = _corpus_with_tracks(
        tmp_path, plan=np.array([2] * 200), memorised=None)
    result = audit.model_leakage(_b, records, {}, generated, tracks, cache)
    assert result["with_train_twin"]["clips"] == 0          # nothing to read there
    assert result["all_clips"]["clips"] == 1                # and a reading anyway
    assert result["plan_vs_train_pool"] == 4
    assert result["rows"][0]["train_recordings_compared"] == 4


def test_the_training_sample_is_drawn_once_for_every_clip(tmp_path):
    """A per-clip redraw would let the sample explain a difference between clips."""
    import random
    rows = [("wild_v5:{}:c0".format(i), "test", _motion(300, seed=20 + i))
            for i in range(1, 3)]
    rows += [("wild_v5:{}:c0".format(i), "train", _motion(300, seed=40 + i))
             for i in range(3, 12)]
    root = _bundle(tmp_path, rows)
    bundle_root, records = audit.read_bundle(root)
    tracks = {name: np.array([i] * 100 + [i + 20] * 100)
              for i, name in enumerate(records)}
    cache = {}
    generated = {name: (audit.joints_for(bundle_root, records[name], cache),
                        np.array([5] * 200))
                 for name in ("wild_v5:1:c0", "wild_v5:2:c0")}
    result = audit.model_leakage(bundle_root, records, {}, generated, tracks, cache,
                                 plan_train_sample=4, rng=random.Random(7))
    assert result["plan_vs_train_pool"] == 4
    compared = {r["train_recordings_compared"] for r in result["rows"]}
    assert compared == {4}          # same pool for both clips, neither in its own


def test_the_audit_reads_every_pair_list_it_is_handed(tmp_path):
    """It has to consume at least what the split consumed, or it cannot fail on it.

    The split built from v5's list alone was clean against that list and had
    281 test<->train edges of v4's crossing it.  An audit given only the newer
    list would have reported the same clean bill of health.
    """
    own, twin = _motion(300, seed=21), _motion(300, seed=21) + np.float32(0.002)
    root = _bundle(tmp_path / "b", [
        ("wild_v5:1:c0", "test", own), ("wild_v5:2:c0", "train", twin),
        ("wild_v5:3:c0", "train", _motion(300, seed=31))])
    _bundle_root, records = audit.read_bundle(root)
    (tmp_path / "p").mkdir()
    new_list = _pairs(tmp_path / "p", [])
    old_list = tmp_path / "p" / "old.jsonl"
    old_list.write_text(json.dumps(
        {"left": "wild_v4:1:c0", "right": "wild_v4:2:c0", "lag_frames": 0}) + "\n",
        encoding="utf-8")

    alone, _lags, rows = audit.merge_pair_files([new_list], records)
    assert alone == {} and rows[0]["edges_on_this_bundle"] == 0

    both, _lags, rows = audit.merge_pair_files([new_list, old_list], records)
    # Carried across the generation prefix, so the audit sees the crossing pair.
    assert both["wild_v5:1:c0"] == {"wild_v5:2:c0"}
    assert [r["edges_on_this_bundle"] for r in rows] == [0, 1]
    assert [r["edges_this_file_added"] for r in rows] == [0, 1]
    declared = audit.declared_leakage(records, both)
    assert declared["straddling_groups"] == 1


def test_a_repeated_pair_list_lands_and_adds_nothing(tmp_path):
    """"Landed nothing" and "added nothing" are different, and only one is a bug."""
    root = _bundle(tmp_path / "b", [
        ("wild_v5:1:c0", "test", _motion(300, seed=21)),
        ("wild_v5:2:c0", "train", _motion(300, seed=31))])
    _bundle_root, records = audit.read_bundle(root)
    (tmp_path / "p").mkdir()
    first = _pairs(tmp_path / "p", [("wild_v5:1:c0", "wild_v5:2:c0")])
    second = tmp_path / "p" / "again.jsonl"
    second.write_text(first.read_text(encoding="utf-8"), encoding="utf-8")
    _adj, _lags, rows = audit.merge_pair_files([first, second], records)
    assert [r["edges_on_this_bundle"] for r in rows] == [1, 1]
    assert [r["edges_this_file_added"] for r in rows] == [1, 0]


def test_the_lag_matched_null_is_reported_beside_the_original(tmp_path):
    """The forward search minimises over candidates AND lags; the null did not.

    ``best_over_lag`` scans 121 lags and the forward reading runs it on each of
    LAG_RESCORE_TOP_K candidates; the time-reversed null took the lag-0 argmin
    and stopped.  A null that is a strictly weaker search than its own statistic
    returns systematically larger ratios, so ``excess_over_time_reversed_null``
    is biased positive and ``reading_is_at_its_noise_floor`` is biased against
    firing -- on a corpus with no leak at all.  The block comment the null sits
    under names the artifact it exists to cancel as "minimising over 1,500
    candidates x 121 lags", so this is a departure from the tool's own stated
    design, not a preference.
    """
    root = _bundle(tmp_path, _corpus(tmp_path, held_out=8, train=30))
    pairs = _pairs(tmp_path, [])
    output = tmp_path / "report.json"
    audit.main(["--bundle", str(root), "--pairs", str(pairs), "--output", str(output),
                "--test-sample", "8", "--train-sample", "30",
                "--calibration-sample", "20", "--threshold", "0.5", "--policy", "warn"])
    measured = json.loads(output.read_text(encoding="utf-8"))["measured"]
    for key in ("time_reversed_null_flagged", "excess_over_time_reversed_null",
                "reading_is_at_its_noise_floor", "lag_matched_null_flagged",
                "excess_over_lag_matched_null",
                "reading_is_at_its_lag_matched_noise_floor",
                "the_two_nulls_disagree"):
        assert key in measured, key
    # The matched null searches at least as hard, so it can never flag fewer.
    assert measured["lag_matched_null_flagged"] >= measured["time_reversed_null_flagged"]
    assert (measured["excess_over_lag_matched_null"]
            <= measured["excess_over_time_reversed_null"])


def test_a_verdict_that_depends_on_which_null_is_used_says_so(tmp_path):
    """Two criteria pointing opposite ways is a stop, not a choice.

    Adopting the matched null alone would not be an improvement: on this file's
    planted-copy control it cancels a leak that is really there, trading a
    false-positive bias for a false-negative one.  So the report carries both
    and flags the disagreement rather than resolving it silently.
    """
    rows = _corpus(tmp_path, held_out=8, train=30)
    twin = {name: motion for name, _, motion in rows}["w:t0:c0"]
    rows = [r for r in rows if r[0] != "w:h0:c0"]
    rows.append(("w:h0:c0", "test", twin + np.float32(0.0005)))
    root = _bundle(tmp_path, rows)
    pairs = _pairs(tmp_path, [])
    output = tmp_path / "report.json"
    audit.main(["--bundle", str(root), "--pairs", str(pairs), "--output", str(output),
                "--test-sample", "8", "--train-sample", "30",
                "--calibration-sample", "20", "--threshold", "0.5", "--policy", "warn"])
    measured = json.loads(output.read_text(encoding="utf-8"))["measured"]
    # The planted copy survives candidate reversal, so the original null leaves
    # an excess; the matched null cancels it.  That is exactly the disagreement
    # the flag exists to make loud.
    assert measured["excess_over_time_reversed_null"] >= 1
    assert measured["reading_is_at_its_noise_floor"] is False
    assert measured["the_two_nulls_disagree"] is (
        measured["reading_is_at_its_lag_matched_noise_floor"] is True)


def test_generated_motion_from_another_generation_lands_instead_of_reading_clean(tmp_path):
    """The defect this pins: an unlandable ``--generated`` reads exactly like a clean bill.

    ``read_generated`` keys on the pickle's file name, which is the recording id
    of the corpus that produced it, and ``model_leakage`` used to join that key
    against the bundle exactly.  Against a bundle of another generation that
    join is empty, and an empty join is invisible: every count in the summary is
    0, which is the same shape as "no clip reproduced its twin".  The module
    docstring already states the rule for the cohort -- match on the stripped id
    -- and this holds ``--generated`` to it.
    """
    own = _motion(300, seed=21)
    twin = own + np.float32(0.002)
    rows = [("wild_v5:1:c0", "test", own), ("wild_v5:2:c0", "train", twin),
            ("wild_v5:3:c0", "train", _motion(300, seed=31)),
            ("wild_v5:4:c0", "train", _motion(300, seed=41))]
    root = _bundle(tmp_path, rows)
    bundle_root, records = audit.read_bundle(root)
    adjacency, _, _ = audit.read_pairs(
        _pairs(tmp_path, [("wild_v5:1:c0", "wild_v5:2:c0")]))
    cache = {}
    joints_twin = audit.joints_for(bundle_root, records["wild_v5:2:c0"], cache)

    # Generated under the OLD generation's ids -- the corpus being disproved.
    from_v4 = {"wild_v4:1:c0": (joints_twin, None)}

    before = audit.model_leakage(bundle_root, records, adjacency, from_v4, {}, cache)
    assert before["in_bundle"] == 0                        # nothing measured...
    assert before["all_clips"]["clips"] == 0
    assert before["with_train_twin"]["closer_to_train_twin_than_own"] == 0  # ...reads clean

    landed, stats = audit.resolve_generated(from_v4, records)
    assert stats == {"pickles_read": 1, "clips_resolved": 1,
                     "pickles_not_in_this_bundle": 0, "pickles_dropped_to_a_taken_id": 0}
    after = audit.model_leakage(bundle_root, records, adjacency, landed, {}, cache)
    assert after["in_bundle"] == 1
    assert after["with_train_twin"]["closer_to_train_twin_than_own"] == 1


def test_generated_from_this_bundles_own_generation_resolves_unchanged(tmp_path):
    """Positive control: the same-generation pairing is exactly what it always was."""
    rows = [("wild_v5:1:c0", "test", _motion(300, seed=21)),
            ("wild_v5:2:c0", "train", _motion(300, seed=31))]
    root = _bundle(tmp_path, rows)
    _bundle_root, records = audit.read_bundle(root)
    generated = {"wild_v5:1:c0": ("motion", None), "wild_v5:2:c0": ("other", None)}
    landed, stats = audit.resolve_generated(generated, records)
    assert landed == generated
    assert stats["clips_resolved"] == 2
    assert stats["pickles_not_in_this_bundle"] == 0


def test_generated_that_resolves_onto_nothing_is_refused_not_reported_as_clean(tmp_path):
    """A directory from an unrelated corpus is a hard error, like the cohort's."""
    rows = [("wild_v5:1:c0", "test", _motion(300, seed=21)),
            ("wild_v5:2:c0", "train", _motion(300, seed=31))]
    root = _bundle(tmp_path, rows)
    _bundle_root, records = audit.read_bundle(root)
    with pytest.raises(SystemExit) as excinfo:
        audit.resolve_generated({"wild_v4:999:c9": ("motion", None)}, records)
    message = str(excinfo.value)
    assert "none of them resolves" in message
    assert "wild_v4:999:c9" in message                     # names what failed to land


def test_generated_pickles_that_land_nowhere_are_counted_not_merged_away(tmp_path):
    """"The directory shrank" and "the directory did not match" must read apart."""
    rows = [("wild_v5:1:c0", "test", _motion(300, seed=21)),
            ("wild_v5:2:c0", "train", _motion(300, seed=31))]
    root = _bundle(tmp_path, rows)
    _bundle_root, records = audit.read_bundle(root)
    landed, stats = audit.resolve_generated(
        {"wild_v5:1:c0": ("a", None), "wild_v5:404:c0": ("b", None)}, records)
    assert set(landed) == {"wild_v5:1:c0"}
    assert stats["clips_resolved"] == 1
    assert stats["pickles_not_in_this_bundle"] == 1
