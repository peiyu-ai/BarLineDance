"""The clean-account corpus selector, exercised where each rule can fail.

The gate this file exists for is the last one: a corpus whose segmentation was
built before its features were rebuilt reports hours and segment counts that
describe a cut nobody holds any more.  On 2026-08-20 the first version of the
tool did exactly that and understated the real corpus by 20,141 frames with
nothing saying so, so the check is tested in both directions rather than only
for "it can raise".
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.select_clean_accounts_corpus import SelectionError, select  # noqa: E402


def _corpus(tmp_path, *, accounts, feature_frames=None, orphans=()):
    """A population audit, a group-key map, a segmentation and S3D features.

    ``accounts`` maps account -> (solo_share, [clip frame counts]).  The
    features are written with whatever length ``feature_frames`` gives for a
    stem, defaulting to the segmentation's, so a test can make exactly one clip
    disagree.
    """
    features = tmp_path / "s3d"
    features.mkdir()
    by_account, group_keys, records = {}, {}, []
    upload = 7000000000000000000
    for account, (share, frames) in accounts.items():
        by_account[account] = {"clips": len(frames), "solo_share": share,
                               "subject_area_median": {"n": len(frames), "p50": 0.3}}
        for count in frames:
            upload += 1
            group_keys[str(upload)] = account
            stem = "{}__clip000".format(upload)
            bounds = list(range(0, count + 1, max(1, count // 4)))
            if bounds[-1] != count:
                bounds.append(count)
            records.append({
                "sequence": stem, "motion_frames": count,
                "boundaries": bounds,
                "segments": [{"start": a, "end": b, "frames": b - a}
                             for a, b in zip(bounds[:-1], bounds[1:])],
            })
            rows = (feature_frames or {}).get(stem, count)
            np.savez(features / (stem + ".npz"),
                     features=np.zeros((rows, 4), dtype=np.float32),
                     meta=json.dumps({"motion_frames": rows}))

    population = tmp_path / "population.json"
    population.write_text(json.dumps({
        "generated_by": "test", "solo_definition": "test",
        "by_account": by_account}), encoding="utf-8")
    keys = tmp_path / "group_keys.json"
    keys.write_text(json.dumps(group_keys), encoding="utf-8")
    segmentation = tmp_path / "segmentation.json"
    segmentation.write_text(json.dumps({"records": records}), encoding="utf-8")
    orphan_file = tmp_path / "orphans.txt"
    orphan_file.write_text("\n".join(orphans), encoding="utf-8")
    return population, keys, segmentation, orphan_file, features


def _stems(segmentation):
    report = json.loads(segmentation.read_text(encoding="utf-8"))
    return [record["sequence"] for record in report["records"]]


def test_threshold_selects_accounts_and_not_a_typed_list(tmp_path):
    args = _corpus(tmp_path, accounts={
        "high": (0.90, [300, 300]), "mid": (0.60, [300]), "low": (0.20, [300])})
    population, keys, segmentation, orphans, features = args
    result = select(population, keys, segmentation, orphans, 0.55,
                    features_dir=features)
    assert result["selection"]["accounts"] == ["high", "mid"]
    assert result["totals"]["clips"] == 3
    # the threshold, not the names, is what the corpus can be rebuilt from
    assert result["selection"]["min_solo_share"] == 0.55


def test_no_account_clears_the_threshold_is_refused(tmp_path):
    args = _corpus(tmp_path, accounts={"low": (0.20, [300])})
    population, keys, segmentation, orphans, features = args
    with pytest.raises(SelectionError, match="no account has solo_share"):
        select(population, keys, segmentation, orphans, 0.55, features_dir=features)


def test_orphans_are_excluded_by_name_and_counted(tmp_path):
    args = _corpus(tmp_path, accounts={"high": (0.90, [300, 300])})
    population, keys, segmentation, orphans, features = args
    victim = _stems(segmentation)[0]
    orphans.write_text(victim + "\n", encoding="utf-8")
    result = select(population, keys, segmentation, orphans, 0.55,
                    features_dir=features)
    assert result["totals"]["clips"] == 1
    assert result["excluded"]["counts"]["orphan"] == 1
    assert victim in result["excluded"]["examples"]["orphan"]


def test_an_upload_with_no_recorded_account_is_an_error_not_a_bucket(tmp_path):
    args = _corpus(tmp_path, accounts={"high": (0.90, [300])})
    population, keys, segmentation, orphans, features = args
    keys.write_text(json.dumps({}), encoding="utf-8")
    with pytest.raises(SelectionError, match="no recorded account"):
        select(population, keys, segmentation, orphans, 0.55, features_dir=features)


def test_segmentation_older_than_its_features_is_refused(tmp_path):
    """Negative control: one clip whose features were rebuilt after the cut."""
    args = _corpus(tmp_path, accounts={"high": (0.90, [300, 400])})
    population, keys, segmentation, orphans, features = args
    stale = _stems(segmentation)[0]
    # rewrite that one feature file shorter than the segmentation believes
    np.savez(features / (stale + ".npz"),
             features=np.zeros((211, 4), dtype=np.float32),
             meta=json.dumps({"motion_frames": 211}))
    with pytest.raises(SelectionError, match="segmentation older than their S3D"):
        select(population, keys, segmentation, orphans, 0.55, features_dir=features)


def test_matching_features_pass_and_the_check_is_recorded(tmp_path):
    """Positive control: the same corpus with features the cut describes."""
    args = _corpus(tmp_path, accounts={"high": (0.90, [300, 400])})
    population, keys, segmentation, orphans, features = args
    result = select(population, keys, segmentation, orphans, 0.55,
                    features_dir=features)
    check = result["segmentation_vs_features"]
    assert check["checked"] is True
    assert check["clips_whose_segmentation_predates_their_features"] == 0
    assert result["totals"]["frames"] == 700


def test_the_defect_can_be_recorded_instead_of_refused(tmp_path):
    """``--allow-stale-segmentation`` names the defect rather than hiding it."""
    args = _corpus(tmp_path, accounts={"high": (0.90, [300, 400])})
    population, keys, segmentation, orphans, features = args
    stale = _stems(segmentation)[0]
    np.savez(features / (stale + ".npz"),
             features=np.zeros((211, 4), dtype=np.float32),
             meta=json.dumps({"motion_frames": 211}))
    result = select(population, keys, segmentation, orphans, 0.55,
                    features_dir=features, allow_stale_segmentation=True)
    check = result["segmentation_vs_features"]
    assert check["clips_whose_segmentation_predates_their_features"] == 1
    assert check["by_account"] == {"high": 1}
    assert stale in check["examples"]


def test_without_a_features_dir_the_check_says_so_rather_than_passing(tmp_path):
    """A skipped check must not read as a satisfied one."""
    args = _corpus(tmp_path, accounts={"high": (0.90, [300])})
    population, keys, segmentation, orphans, _ = args
    result = select(population, keys, segmentation, orphans, 0.55, features_dir=None)
    assert result["segmentation_vs_features"] == {"checked": False}


def test_a_cut_that_covers_the_whole_clip_reports_all_of_it(tmp_path):
    """Positive control for the segmented-hours split.

    Alg.1's cuts run frame 0 to the last frame, so footage and segmented material
    are the same number and the ratio has to read 1.0 -- otherwise the number
    below would be flagging a defect that is not there.
    """
    args = _corpus(tmp_path, accounts={"high": (0.90, [300, 400])})
    population, keys, segmentation, orphans, features = args
    result = select(population, keys, segmentation, orphans, 0.55,
                    features_dir=features)
    assert result["totals"]["frames"] == 700
    assert result["totals"]["segmented_frames"] == 700
    assert result["totals"]["segmented_fraction"] == 1.0


def test_a_beat_grid_cut_reports_the_footage_it_leaves_outside_every_segment(tmp_path):
    """Negative control: the lead-in and trail-out a beat grid cannot cover.

    ``segment_on_music_beats.py --edges drop`` starts at the first beat and stops
    at the last, so 1.69 h of the 10.10 h corpus sits inside no segment.  Reporting
    only ``hours`` would let the corpus claim material that no consumer of the
    segments receives, so both numbers are carried.
    """
    args = _corpus(tmp_path, accounts={"high": (0.90, [300, 400])})
    population, keys, segmentation, orphans, features = args
    report = json.loads(segmentation.read_text(encoding="utf-8"))
    for record in report["records"]:
        # keep the interior cuts only, as --edges drop does
        bounds = record["boundaries"][1:-1]
        record["boundaries"] = bounds
        record["covered_frames"] = bounds[-1] - bounds[0]
        record["segments"] = [{"start": a, "end": b, "frames": b - a}
                              for a, b in zip(bounds[:-1], bounds[1:])]
    segmentation.write_text(json.dumps(report), encoding="utf-8")
    result = select(population, keys, segmentation, orphans, 0.55,
                    features_dir=features)
    assert result["totals"]["frames"] == 700            # footage is unchanged
    assert result["totals"]["segmented_frames"] < 700   # what a consumer gets
    assert result["totals"]["segmented_fraction"] < 1.0
    assert result["by_account"]["high"]["segmented_hours"] < \
        result["by_account"]["high"]["hours"]


def _freshness(tmp_path, stems, name="freshness.json"):
    path = tmp_path / name
    path.write_text(json.dumps({
        "generated_by": "tools/audit_clip_freshness.py",
        "stale": {"3d": list(stems), "s3d": list(stems)},
    }), encoding="utf-8")
    return path


def test_clips_whose_derivatives_are_still_stale_are_excluded(tmp_path):
    """A live name whose 3D is from an older cut is not an orphan.

    On 2026-08-20 stage B could not re-extract 102 of the re-cut clips --
    visual odometry diverged -- so their published 3D stayed the pre-re-cut
    generation.  Nothing downstream can tell those from clean clips: the
    objects exist, join, and are the right length.  They are excluded here, and
    counted under their own reason, because folding them into ``orphan`` would
    make ``corpus.json`` report a cause it never measured.
    """
    args = _corpus(tmp_path, accounts={"high": (0.90, [300, 300, 300])})
    population, keys, segmentation, orphans, features = args
    stems = _stems(segmentation)
    stale = _freshness(tmp_path, stems[:1])

    result = select(population, keys, segmentation, orphans, 0.55,
                    features_dir=features, stale_path=stale)

    assert result["totals"]["clips"] == 2
    assert result["excluded"]["counts"]["stale_derivatives"] == 1
    assert stems[0] not in result["clips"]
    assert result["inputs"]["stale"] == str(stale)


def test_a_name_that_is_both_orphan_and_stale_is_counted_once(tmp_path):
    """Under the stronger statement: an orphan may never be read again."""
    args = _corpus(tmp_path, accounts={"high": (0.90, [300, 300])})
    population, keys, segmentation, _orphans, features = args
    stems = _stems(segmentation)
    orphan_file = tmp_path / "orphans2.txt"
    orphan_file.write_text(stems[0], encoding="utf-8")
    stale = _freshness(tmp_path, [stems[0]])

    result = select(population, keys, segmentation, orphan_file, 0.55,
                    features_dir=features, stale_path=stale)

    counts = result["excluded"]["counts"]
    assert counts["orphan"] == 1
    assert "stale_derivatives" not in counts
    assert result["totals"]["clips"] == 1


def test_without_a_stale_manifest_nothing_extra_is_dropped(tmp_path):
    args = _corpus(tmp_path, accounts={"high": (0.90, [300, 300])})
    population, keys, segmentation, orphans, features = args
    result = select(population, keys, segmentation, orphans, 0.55,
                    features_dir=features)
    assert result["totals"]["clips"] == 2
    assert result["inputs"]["stale"] is None
