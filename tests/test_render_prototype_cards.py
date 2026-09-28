"""Tests for the sub-prototype card renderer.

The page exists to let a human judge whether a category holds together, so the
tests protect the things that would make that judgement wrong without looking
wrong: a caption attached to the wrong segment, a selection that quietly shows
only the easy cases, and padding that invents a pose.
"""

import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.render_prototype_cards import (  # noqa: E402
    BODY_JOINTS,
    CardError,
    build_parser,
    caption_text,
    keyframe_poses,
    load_caption_index,
    project,
    select_subprototypes,
)


def _folded_pose():
    """A body bent forward: extent lives on the depth axis, not on x or z."""
    pose = np.zeros((BODY_JOINTS, 3))
    pose[:, 1] = np.linspace(0.0, 1.8, BODY_JOINTS)   # forward, the depth axis
    pose[:, 2] = np.linspace(0.0, 0.2, BODY_JOINTS)   # barely any height
    return pose


def _members(label_sizes):
    """Build a grouped-members mapping with a given size and upload spread."""
    grouped = {}
    for label, (size, uploads) in label_sizes.items():
        entries = []
        for index in range(size):
            upload = "upload{}".format(index % uploads)
            entries.append({"recording_id": "{}:clip0".format(upload), "upload": upload,
                            "start": index * 40, "end": index * 40 + 30})
        grouped[label] = entries
    return grouped


def test_captions_are_keyed_by_the_exact_segment(tmp_path):
    path = tmp_path / "captions.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in [
        {"recording_id": "a", "start": 0, "end": 30, "fields": {"arms": "raised"}},
        {"recording_id": "a", "start": 30, "end": 60, "fields": {"arms": "low"}},
    ]) + "\n", encoding="utf-8")
    index = load_caption_index(path)
    # Two segments of one recording must not collapse onto each other, or the
    # page would show text that belongs to a different piece of motion.
    assert index[("a", 0, 30)]["fields"]["arms"] == "raised"
    assert index[("a", 30, 60)]["fields"]["arms"] == "low"


def test_caption_index_keeps_the_first_of_a_duplicated_segment(tmp_path):
    path = tmp_path / "captions.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in [
        {"recording_id": "a", "start": 0, "end": 30, "caption": "first"},
        {"recording_id": "a", "start": 0, "end": 30, "caption": "second"},
    ]) + "\n", encoding="utf-8")
    assert load_caption_index(path)[("a", 0, 30)]["caption"] == "first"


def test_no_captions_path_is_not_an_error():
    assert load_caption_index(None) == {}


def test_caption_text_prefers_the_fields_the_grouping_consumed():
    row = {"fields": {"arms": "raised", "level": "high"},
           "caption": "a person raises both arms", "summary": "arms up"}
    assert caption_text(row) == "arms: raised, level: high"


def test_caption_text_falls_back_without_inventing():
    assert caption_text({"caption": "a person spins"}) == "a person spins"
    assert caption_text({"summary": "spin"}) == "spin"
    assert caption_text(None) == "(no caption)"
    assert caption_text({}) == "(no caption)"
    assert caption_text({"fields": {}}) == "(no caption)"


def test_keyframe_poses_pad_by_repeating_a_real_beat():
    # A short, nearly still segment yields fewer beats than requested; padding
    # must repeat the last real pose rather than insert a collapsed skeleton at
    # the origin, which would read as a move the dancer never made.
    joints = np.zeros((6, 24, 3))
    joints[:, :, 2] = np.linspace(0.0, 0.05, 6)[:, None]
    joints[:, 16, 0], joints[:, 17, 0] = 0.2, -0.2
    poses = keyframe_poses(joints, 4)
    assert len(poses) == 4
    assert all(pose.shape == (BODY_JOINTS, 3) for pose in poses)
    assert np.allclose(poses[-1], poses[-2])
    assert np.isfinite(np.stack(poses)).all()


def test_keyframe_poses_are_canonical_and_root_centred():
    rng = np.random.default_rng(0)
    joints = rng.normal(size=(40, 24, 3)) * 0.1
    joints[:, 16, 0], joints[:, 17, 0] = 0.2, -0.2
    poses = keyframe_poses(joints, 3)
    # canonical_pose subtracts the root, so joint 0 sits at the origin in every
    # drawn frame; without this two members would differ by where they stood.
    for pose in poses:
        assert np.allclose(pose[0], 0.0, atol=1e-9)


def test_spread_selection_covers_the_size_range_not_just_the_head():
    grouped = _members({label: (400 - label * 6, 3) for label in range(1, 60)})
    everything = [len(entries) for entries in grouped.values()]
    picked, _ = select_subprototypes(grouped, count=5, members=4,
                                     strategy="spread", min_uploads=2)
    sizes = [len(entries) for _, entries, _ in picked]
    assert len(picked) == 5
    # The point of "spread": both ends of the size distribution are represented,
    # where "largest" would show only the head.
    assert min(sizes) == min(everything)
    assert max(sizes) == max(everything)


def test_largest_selection_takes_only_the_head():
    grouped = _members({label: (400 - label * 4, 3) for label in range(1, 60)})
    picked, _ = select_subprototypes(grouped, count=5, members=4,
                                     strategy="largest", min_uploads=2)
    sizes = [len(entries) for _, entries, _ in picked]
    assert sizes == sorted(sizes, reverse=True)
    assert min(sizes) > 300


def test_selection_drops_single_upload_categories():
    grouped = _members({1: (30, 1), 2: (30, 5)})
    picked, _ = select_subprototypes(grouped, count=10, members=4,
                                     strategy="spread", min_uploads=2)
    # A category drawn from one clip was memorised, not discovered, so it cannot
    # be used as evidence that the vocabulary works.
    assert [label for label, _, _ in picked] == [2]


def test_selection_never_repeats_a_subprototype():
    grouped = _members({label: (50, 4) for label in range(1, 8)})
    picked, _ = select_subprototypes(grouped, count=7, members=3,
                                     strategy="spread", min_uploads=2)
    labels = [label for label, _, _ in picked]
    assert len(labels) == len(set(labels))


def test_unknown_selection_strategy_is_refused():
    grouped = _members({1: (30, 5)})
    with pytest.raises(CardError, match="unknown selection"):
        select_subprototypes(grouped, count=1, members=2,
                             strategy="handpicked", min_uploads=2)


def test_a_forward_fold_is_invisible_head_on_and_visible_off_axis():
    pose = _folded_pose()
    head_on = project(pose, 0.0)
    turned = project(pose, 30.0)
    # Straight on, a fold into the depth axis collapses to a point and reads as
    # broken data; that misreading is the whole reason the default is not 0.
    assert head_on[:, 0].max() - head_on[:, 0].min() < 1e-9
    assert turned[:, 0].max() - turned[:, 0].min() > 0.8


def test_projection_keeps_height_untouched():
    rng = np.random.default_rng(9)
    pose = rng.normal(size=(BODY_JOINTS, 3))
    for azimuth in (0.0, 30.0, 90.0):
        assert np.allclose(project(pose, azimuth)[:, 1], pose[:, 2])


def test_projection_preserves_a_purely_sideways_pose_at_small_angles():
    pose = np.zeros((BODY_JOINTS, 3))
    pose[:, 0] = np.linspace(-1.0, 1.0, BODY_JOINTS)
    turned = project(pose, 30.0)
    # Foreshortening is the price of showing depth; at 30 degrees a sideways
    # pose keeps cos(30) = 87% of its width, which stays legible.
    assert turned[:, 0].max() - turned[:, 0].min() == pytest.approx(2.0 * np.cos(np.pi / 6),
                                                                   abs=1e-9)


def test_parser_defaults_to_the_representative_selection():
    args = build_parser().parse_args(
        ["--labels", "l", "--bundle", "b", "--output", "o"])
    assert args.select == "spread"
    assert args.frames == 4
    assert args.min_uploads == 2
    assert args.view_azimuth == 30.0
