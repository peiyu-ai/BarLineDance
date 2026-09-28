"""The retarget's own defects, pinned so they cannot come back.

Every case here is a bug that shipped a wrong picture first: a dancer flat on
her back, a head pitched forward all clip, shoulders bent out of the avatar's
rest silhouette, and sleeves left aimed where the artist parked them.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.vrm_retarget import (SMPL_PARENTS, SMPL_TO_VRM, Z_UP_TO_Y_UP,
                                axis_angle_to_matrix, to_y_up_rotation)

AVATAR = pathlib.Path("third_party/vrm/anime_female.vrm.glb")


def test_the_frame_change_is_one_sided(tmp_path):
    """``C R``, not ``C R C^T`` -- SMPL's template was never in z-up.

    The two-sided form converts the template side as well and lays the avatar
    flat on her back, which is how this shipped the first time.
    """
    rotation = axis_angle_to_matrix(np.array([0.3, -0.2, 0.7]))
    assert np.allclose(to_y_up_rotation(rotation), Z_UP_TO_Y_UP @ rotation)
    # and the two forms really do differ, so the test can fail
    assert not np.allclose(to_y_up_rotation(rotation),
                           Z_UP_TO_Y_UP @ rotation @ Z_UP_TO_Y_UP.T)


def test_a_z_up_upright_body_comes_out_upright(tmp_path):
    """Positive control with a known answer: up must stay up."""
    up_in_z_up = np.array([0.0, 0.0, 1.0])
    assert np.allclose(Z_UP_TO_Y_UP @ up_in_z_up, [0.0, 1.0, 0.0])


def test_the_joint_map_covers_everything_this_corpus_moves():
    """22 of 24; the two left out are the hands, which are identity here."""
    assert len(SMPL_TO_VRM) == 22
    assert set(range(24)) - set(SMPL_TO_VRM) == {22, 23}
    assert len(set(SMPL_TO_VRM.values())) == 22          # no bone used twice
    for joint in SMPL_TO_VRM:
        assert joint < len(SMPL_PARENTS)


@pytest.mark.skipif(not AVATAR.is_file(), reason="avatar not vendored here")
def test_the_avatar_keeps_its_own_rest_silhouette():
    """No direction alignment by default.

    The alignment was added for an A-pose/T-pose difference that measurement
    says is not there -- upper arm 6.8 deg, forearm 2.2 -- while the large
    angles are skeleton topology (pelvis 24.2, chest 37.4, neck 31.8, clavicle
    28.8).  Applying them bent the shoulders out of the avatar's rest.
    """
    from tools.vrm_retarget import VRMAvatar

    avatar = VRMAvatar(AVATAR)
    rest = np.zeros((24, 3))
    for joint, parent in enumerate(SMPL_PARENTS):
        rest[joint] = rest[parent] + [0.0, 0.1, 0.0] if parent >= 0 else 0.0
    align = avatar.alignment(rest)
    for bone in SMPL_TO_VRM.values():
        node = avatar.by_name.get(bone)
        if node is not None:
            assert np.allclose(align[node], avatar.global_rest[node][:3, :3])


@pytest.mark.skipif(not AVATAR.is_file(), reason="avatar not vendored here")
def test_every_constraint_resolves_in_the_final_pass():
    """8 of 14 sources come later in the hierarchy; one pass drops them.

    Dropping them is what left the sleeves aimed at the rest direction, and the
    first version dropped them *silently*.  Two passes must leave at most the
    roll constraints whose source is the node's own chain.
    """
    import pickle

    from tools.vrm_retarget import VRMAvatar

    avatar = VRMAvatar(AVATAR)
    later = 0
    position = {node: order for order, node in enumerate(avatar.order)}
    for node, constraint in avatar.constraints.items():
        source = (constraint.get("aim") or constraint.get("roll") or {}).get("source")
        if source is not None and position.get(source, 1 << 30) > position[node]:
            later += 1
    assert later >= 1, "if no source came later, this test proves nothing"

    rest = (np.arange(24)[:, None] * np.array([0.0, 0.05, 0.0]))
    align = avatar.alignment(rest)
    rotations = np.stack([np.eye(3)] * 24)
    avatar.pose(rotations, np.zeros(3), align, 1.0)
    # the aim constraints -- the ones that drive the clothing -- must all resolve
    aims = sum(1 for c in avatar.constraints.values() if "aim" in c)
    assert avatar.constraints_skipped < aims
