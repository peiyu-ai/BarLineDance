"""The 2026-09-17 selection package: the join cost in the placed frame, the
height term, the selector's input units, the hop guard and the loudness cap.

WHAT WAS MEASURED BEFORE ANY OF IT WAS WRITTEN (fix5, twenty eval drafts, a
replay that reproduced 200/200 units):

  * ``_join_cost`` compared a raw library head with a tail that had already
    been floor-levelled, turned and chained -- median ground-position gap
    0.63 m, heading gap 28.6 deg, height gap 0.35 m (the source floor) -- so
    about a fifth of the 0.35 join band was decided by where the library clip
    stood in its own upload.  The operator: root position does not matter,
    completion merges it.
  * The shipped pick is no closer in pelvis height than the pool median
    (41/90 slots) although the band holds a candidate within 0.42 cm.
  * The selector checkpoint was fitted on doubly-unnormalized features.
  * The draft is airborne on 3.46% of frames against ground truth's 0.57%.

Each switch is off by default, and each guard below fails on the source as it
was before the package (the peer session's condition: a test that only finds a
name has already been dead once in this repository).
"""
import math
import pathlib
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.rotation_ops import matrix_to_rotation_6d            # noqa: E402
from infer_atomic import (CONTACT_CHANNELS, ROOT_POSITION_START,  # noqa: E402
                          IndexedAtomicMotionLibrary as Library)
from tools.census_release_vertical import (HOP_FLAG, SQUAT_FLAG,   # noqa: E402
                                           vertical_flags)

SOURCE = (ROOT / "infer_atomic.py").read_text()
ORIENT = Library.GLOBAL_ORIENT_START


def rz(theta):
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def clip(frames, yaw, start_xy, step_xy, z, seed=0, body=None):
    """A raw 151-D segment: facing ``yaw``, walking ``step_xy`` a frame."""
    generator = torch.Generator().manual_seed(seed)
    values = torch.zeros(frames, 151)
    values[:, :CONTACT_CHANNELS] = 1.0
    heading = rz(yaw)
    for t in range(frames):
        values[t, ROOT_POSITION_START:ROOT_POSITION_START + 2] = (
            torch.tensor(start_xy) + t * torch.tensor(step_xy))
        values[t, ROOT_POSITION_START + 2] = z
        tilt = rz(0.02 * t)  # a little turning inside the clip
        values[t, ORIENT:ORIENT + 6] = matrix_to_rotation_6d(heading @ tilt)
    rest = torch.zeros(23, 6)
    rest[:, 0] = 1.0
    rest[:, 4] = 1.0
    if body is None:
        body = 0.05 * torch.randn(23, 6, generator=generator)
    values[:, ORIENT + 6:] = (rest + body).reshape(-1).repeat(frames, 1)
    return values


class Lib:
    """The real methods under test, no data files.  Identity affine unless set."""

    JOIN_FRAMES = Library.JOIN_FRAMES
    GLOBAL_ORIENT_START = Library.GLOBAL_ORIENT_START
    JOIN_VELOCITY_WEIGHT = Library.JOIN_VELOCITY_WEIGHT
    HOP_FLAG = Library.HOP_FLAG
    SQUAT_FLAG = Library.SQUAT_FLAG
    MUSIC_ENERGY_QUANTILES = Library.MUSIC_ENERGY_QUANTILES

    def __init__(self, motion, floors=None, affine=None, join_frame="raw",
                 join_height_weight=0.0, units="inference"):
        self.motion = np.asarray(motion, dtype=np.float32)
        self._sample_floor = (None if floors is None
                              else np.asarray(floors, dtype=np.float32))
        width = self.motion.shape[-1]
        self._affine = affine or (torch.ones(width), torch.zeros(width))
        self.join_frame = join_frame
        self.join_height_weight = join_height_weight
        self.join_lever_weights = False
        self.join_pose_weight = 1.0
        self.selector_input_units = units
        self._energy_cache = {}
        self._local_floor = None
        self._local_floor_cache = {}
        self.names = tuple("w{}".format(i) for i in range(len(self.motion)))

    _join_cost = Library._join_cost
    _placed_head = Library._placed_head
    _values_at = Library._values_at
    _candidate_floor = Library._candidate_floor
    _normalizer_affine = Library._normalizer_affine
    _facing_yaw = Library._facing_yaw
    _rotate_about_z = Library._rotate_about_z
    _selector_units = Library._selector_units
    _selector_tail = Library._selector_tail
    _segment_energy = Library._segment_energy
    _vertical_flags_of = Library._vertical_flags_of
    _prefer_hop_guard = Library._prefer_hop_guard
    _prefer_music_energy = Library._prefer_music_energy


def tail_and_pool():
    """A chained tail far from the library origin, and one candidate moved about."""
    tail = clip(3, yaw=0.4, start_xy=(2.5, -1.5), step_xy=(0.01, 0.0), z=0.95, seed=1)
    base = clip(20, yaw=-2.0, start_xy=(0.1, 0.3), step_xy=(0.0, 0.01), z=0.95, seed=2)
    return tail, base


def moved(values, dxy, dyaw):
    """The same dance cut from somewhere else in its upload, facing elsewhere."""
    out = values.clone()
    turn = rz(dyaw)
    first = out[0, ROOT_POSITION_START:ROOT_POSITION_START + 2].clone()
    for t in range(len(out)):
        rel = out[t, ROOT_POSITION_START:ROOT_POSITION_START + 2] - first
        out[t, ROOT_POSITION_START:ROOT_POSITION_START + 2] = (
            first + torch.tensor(dxy) + (turn[:2, :2] @ rel))
        from dataset.rotation_ops import rotation_6d_to_matrix
        matrix = rotation_6d_to_matrix(out[t:t + 1, ORIENT:ORIENT + 6])[0]
        out[t, ORIENT:ORIENT + 6] = matrix_to_rotation_6d(turn @ matrix)
    return out


# --------------------------------------------------------------- placed frame
def test_placed_cost_ignores_where_and_which_way_the_clip_was_recorded():
    """The guard the peer session asked for: shift xy, turn the yaw -- the
    placed cost must not move, the raw cost must."""
    tail, base = tail_and_pool()
    other = moved(base, dxy=(3.0, -2.0), dyaw=1.3)
    lib = Lib(np.stack([base.numpy(), other.numpy()]), join_frame="placed")
    a = lib._join_cost((0, 0, 20, "g0"), tail, 20)
    b = lib._join_cost((1, 0, 20, "g1"), tail, 20)
    assert a == pytest.approx(b, rel=1e-4, abs=1e-4)
    raw = Lib(np.stack([base.numpy(), other.numpy()]), join_frame="raw")
    ra = raw._join_cost((0, 0, 20, "g0"), tail, 20)
    rb = raw._join_cost((1, 0, 20, "g1"), tail, 20)
    assert abs(ra - rb) > 0.5, (ra, rb)


def test_placed_cost_ranks_a_better_join_first_whatever_the_placement():
    """Ranking, not just invariance: a candidate whose body matches the tail
    must beat one that does not, even when the worse one happens to stand on
    the tail's spot and face its way in the library."""
    tail, _ = tail_and_pool()
    body = tail[0, ORIENT + 6:].reshape(23, 6) - torch.tensor([1.0, 0, 0, 0, 1.0, 0])
    good = clip(20, yaw=-2.5, start_xy=(-4.0, 4.0), step_xy=(0.0, 0.01), z=0.95,
                body=body)
    bad = clip(20, yaw=0.4, start_xy=(2.52, -1.5), step_xy=(0.01, 0.0), z=0.95,
               seed=7, body=body + 0.4)
    lib = Lib(np.stack([good.numpy(), bad.numpy()]), join_frame="placed")
    pool = [(0, 0, 20, "a"), (1, 0, 20, "b")]
    assert sorted(pool, key=lambda c: lib._join_cost(c, tail, 20))[0] == pool[0]


def test_placed_head_takes_the_floor_off_like_the_draft_does():
    tail, base = tail_and_pool()
    lifted = base.clone()
    lifted[:, ROOT_POSITION_START + 2] += 0.34
    lib = Lib(np.stack([lifted.numpy()]), floors=[0.34], join_frame="placed")
    head = lib._placed_head((0, 0, 20, "g"), 20, tail)
    assert float(head[0, ROOT_POSITION_START + 2]) == pytest.approx(0.95, abs=1e-5)
    assert torch.allclose(head[0, ROOT_POSITION_START:ROOT_POSITION_START + 2],
                          tail[-1, ROOT_POSITION_START:ROOT_POSITION_START + 2], atol=1e-5)
    yaw_head, _ = lib._facing_yaw(head[:1])
    yaw_tail, _ = lib._facing_yaw(tail[-1:])
    assert float(yaw_head[0]) == pytest.approx(float(yaw_tail[0]), abs=1e-4)


def test_raw_frame_is_the_default_and_still_reads_the_stored_bytes():
    assert 'join_frame="raw"' in SOURCE
    assert 'default="raw", dest="draft_join_frame"' in SOURCE
    tail, base = tail_and_pool()
    lib = Lib(np.stack([base.numpy()]))
    lib.join_frame = "raw"
    # the placed path must not run
    lib._placed_head = None
    lib._join_cost((0, 0, 20, "g"), tail, 20)


# --------------------------------------------------------------- height term
def test_height_weight_prefers_the_candidate_at_the_tails_height():
    tail, base = tail_and_pool()
    level = base.clone()
    step = base.clone()
    step[:, ROOT_POSITION_START + 2] -= 0.25
    step[:, ORIENT + 6:] = tail[0, ORIENT + 6:]  # a better body match...
    motion = np.stack([level.numpy(), step.numpy()])
    pool = [(0, 0, 20, "a"), (1, 0, 20, "b")]
    plain = Lib(motion, join_frame="placed")
    assert sorted(pool, key=lambda c: plain._join_cost(c, tail, 20))[0] == pool[1]
    weighted = Lib(motion, join_frame="placed", join_height_weight=30.0)
    # ...loses once a 25 cm pelvis step is priced
    assert sorted(pool, key=lambda c: weighted._join_cost(c, tail, 20))[0] == pool[0]


def test_height_weight_refuses_the_raw_frame():
    with pytest.raises(ValueError, match="needs --draft-join-frame placed"):
        Library("/nonexistent", join_height_weight=30.0)


# --------------------------------------------------------------- selector units
def test_trained_units_repeat_the_trainers_map_after_taking_the_floor_off():
    scale = torch.linspace(0.5, 2.0, 151)
    offset = torch.linspace(-1.0, 1.0, 151)
    values = torch.randn(6, 151)
    lib = Lib(np.zeros((2, 6, 151)), floors=[0.0, 0.3], affine=(scale, offset),
              units="trained")
    raw = values * scale + offset
    raw[:, ROOT_POSITION_START + 2] -= 0.3
    assert torch.allclose(lib._selector_units(values, (1, 0, 6, 'g')), raw * scale + offset, atol=1e-5)
    tail = torch.randn(3, 151)
    assert torch.allclose(lib._selector_tail(tail), tail * scale + offset)
    plain = Lib(np.zeros((2, 6, 151)), floors=[0.0, 0.3], affine=(scale, offset))
    assert torch.allclose(plain._selector_units(values, (1, 0, 6, 'g')), values * scale + offset)
    assert plain._selector_tail(tail) is tail


def test_both_selector_inputs_go_through_the_units_map():
    descriptor = SOURCE.split("    def _descriptor(self, candidate):", 1)[1].split("\n    def ", 1)[0]
    assert "self._selector_units(values, candidate)" in descriptor
    context = SOURCE.split("    def _query_context(", 1)[1].split("\n    def ", 1)[0]
    assert "previous_tail=self._selector_tail(previous_tail)" in context


# --------------------------------------------------------------- hop guard / loudness
class Flags(Lib):
    def __init__(self, flags, energies, hop_guard=False, music_energy=False):
        frames = 10
        motion = np.zeros((len(flags), frames, 151), dtype=np.float32)
        for index, energy in enumerate(energies):
            motion[index, :, ROOT_POSITION_START + 3:] = (
                np.arange(frames)[:, None] * energy)
        super().__init__(motion)
        self._vertical_flags = {
            name: np.full(frames, bits, dtype=np.uint8)
            for name, bits in zip(self.names, flags)}
        self.hop_guard = hop_guard
        self.music_energy = music_energy


def pool(n):
    return [(i, 0, 10, "g{}".format(i)) for i in range(n)]


def test_hop_guard_drops_airborne_candidates_and_counts_it():
    lib = Flags([0, HOP_FLAG, SQUAT_FLAG, HOP_FLAG | SQUAT_FLAG], [1, 1, 1, 1],
                hop_guard=True)
    assert lib._prefer_hop_guard(pool(4)) == [pool(4)[0], pool(4)[2]]
    assert lib.hop_guard_applied == 1


def test_hop_guard_keeps_the_tie_when_everything_hops_and_is_off_by_default():
    lib = Flags([HOP_FLAG, HOP_FLAG], [1, 1], hop_guard=True)
    assert lib._prefer_hop_guard(pool(2)) == pool(2)
    assert lib.hop_guard_empty == 1
    off = Flags([0, HOP_FLAG], [1, 1])
    assert off._prefer_hop_guard(pool(2)) == pool(2)


def test_music_energy_caps_quiet_bars_harder_than_loud_ones():
    energies = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    flags = [0] * 10
    flags[1] = SQUAT_FLAG
    lib = Flags(flags, energies, music_energy=True)
    quiet = lib._prefer_music_energy(pool(10), True)
    assert [c[0] for c in quiet] == [0, 1, 2, 3, 4]       # <= median; squats allowed
    loud = lib._prefer_music_energy(pool(10), False)
    assert [c[0] for c in loud] == [0, 1, 2, 3, 4, 5, 6, 7]  # top fifth dropped
    assert lib.music_energy_quiet == 1 and lib.music_energy_slots == 2
    assert lib._prefer_music_energy(pool(10), None) == pool(10)
    assert lib.music_energy_blind == 1
    off = Flags(flags, energies)
    assert off._prefer_music_energy(pool(10), True) == pool(10)


def test_both_filters_run_after_the_beat_span_in_every_branch():
    """Hitting the beat comes first (the operator's order), and every rule the
    CLI offers must reach both filters -- DEFECTS 77 was a filter that existed
    in a branch the shipped rule never entered."""
    duration = SOURCE.split("    def _duration_pick(", 1)[1].split("\n    def ", 1)[0]
    learned = SOURCE.split('elif self.retrieval_rule == "learned":', 1)[1].split(
        '\n        elif self.retrieval_rule == "phase":', 1)[0]
    seam = SOURCE.split("narrowed = self._prefer_beat_span(near, target_length,\n", 1)[1][:600]
    for body in (duration, learned):
        span = body.index("self._prefer_beat_span(")
        assert span < body.index("self._prefer_hop_guard(") < body.index("self._prefer_hold_by_music(")
        assert span < body.index("self._prefer_music_energy(")
    assert "self._prefer_hop_guard(narrowed)" in seam
    assert "self._prefer_music_energy(narrowed, target_soft)" in seam
    assert SOURCE.count("soft=target_soft)") == 2


def test_build_draft_marks_soft_bars_against_the_library_median():
    body = SOURCE.split("    def build_draft(", 1)[1].split("\n    def ", 1)[0]
    assert "< self.quiet_loudness" in body
    assert "target_soft=segment_soft[position]" in body


def test_filters_need_the_flag_file():
    with pytest.raises(ValueError, match="need --retrieval-vertical-flags"):
        Library("/nonexistent", hop_guard=True)
    with pytest.raises(ValueError, match="need --retrieval-vertical-flags"):
        Library("/nonexistent", music_energy=True)


# --------------------------------------------------------------- detectors
def standing(frames=150):
    joints = np.zeros((frames, 24, 3))
    joints[:, 0, 2] = 0.95
    joints[:, [7, 8], 2] = 0.08
    joints[:, [10, 11], 2] = 0.02
    return joints


def test_a_hop_is_flagged_and_standing_is_not():
    joints = standing()
    assert vertical_flags(joints).max() == 0
    joints[70:80, :, 2] += 0.3
    flags = vertical_flags(joints)
    assert (flags[70:80] & HOP_FLAG).all()
    assert not (flags[:60] & HOP_FLAG).any()


def test_a_deep_squat_is_flagged_and_slow_drift_is_not():
    joints = standing()
    joints[:, :, 2] += np.linspace(0.0, 0.3, 150)[:, None]   # depth drift
    assert vertical_flags(joints).max() == 0
    joints[60:75, 0, 2] -= 0.3
    flags = vertical_flags(joints)
    assert (flags[60:75] & SQUAT_FLAG).all()


# --------------------------------------------------------------- wiring
@pytest.mark.parametrize("flag,dest", [
    ("--draft-join-frame", "draft_join_frame"),
    ("--draft-join-height-weight", "draft_join_height_weight"),
    ("--retrieval-selector-input-units", "retrieval_selector_input_units"),
    ("--draft-hop-guard", "draft_hop_guard"),
    ("--draft-music-energy", "draft_music_energy"),
    ("--retrieval-vertical-flags", "retrieval_vertical_flags"),
    ("--draft-unit-floor", "draft_unit_floor"),
])
def test_every_switch_reaches_the_library_and_the_manifest(flag, dest):
    assert '"{}"'.format(flag) in SOURCE
    assert "{0}=options.{0}".format(dest) in SOURCE
    assert '"{}": '.format(dest) in SOURCE
    library_argument = {
        "draft_join_frame": "join_frame=draft_join_frame",
        "draft_join_height_weight": "join_height_weight=draft_join_height_weight",
        "retrieval_selector_input_units": "selector_input_units=retrieval_selector_input_units",
        "draft_hop_guard": "hop_guard=draft_hop_guard",
        "draft_music_energy": "music_energy=draft_music_energy",
        "retrieval_vertical_flags": "vertical_flags_path=retrieval_vertical_flags",
        "draft_unit_floor": "unit_floor_path=draft_unit_floor",
    }[dest]
    assert library_argument in SOURCE


def test_the_counters_that_prove_the_switches_ran_are_recorded():
    for key in ("draft_join_placed_calls", "draft_hop_guard_applied",
                "draft_music_energy_applied", "draft_music_energy_quiet_loudness"):
        assert '"{}"'.format(key) in SOURCE


# --------------------------------------------------------------- unit floor
def test_unit_floor_levels_each_unit_by_its_own_stretch_of_the_recording():
    """Two cuts of ONE upload, the second from a stretch whose depth drifted
    up 0.18 m: the recording floor treats them alike and pastes the second one
    floating; the local floor puts both on the ground."""
    motion = np.zeros((1, 100, 151), dtype=np.float32)
    motion[0, :, ROOT_POSITION_START + 2] = 1.30
    motion[0, 60:, ROOT_POSITION_START + 2] = 1.48
    local = np.full(100, 0.34, dtype=np.float32)
    local[60:] = 0.52
    lib = Lib(motion, floors=[0.34])
    early, late = (0, 0, 40, "g"), (0, 60, 100, "g")
    column = ROOT_POSITION_START + 2
    assert float(lib._values_at(late, 40)[0, column]) == pytest.approx(1.14, abs=1e-5)
    lib._local_floor = {"w0": local}
    assert float(lib._values_at(early, 40)[0, column]) == pytest.approx(0.96, abs=1e-5)
    assert float(lib._values_at(late, 40)[0, column]) == pytest.approx(0.96, abs=1e-5)


def test_unit_floor_reaches_the_selector_and_the_placed_head_too():
    """One floor per candidate everywhere it is read, or the selector's height
    step and the join cost's would disagree with the draft they score."""
    for name in ("def _values_at", "def _selector_units"):
        body = SOURCE.split("    " + name, 1)[1].split("\n    def ", 1)[0]
        assert "self._candidate_floor(candidate)" in body, name
    assert SOURCE.count("self._sample_floor[") == 1


def test_unit_floor_refuses_a_library_it_cannot_read():
    body = SOURCE.split("    def __init__(self, data_root", 1)[1].split("\n    def ", 1)[0]
    assert "--draft-unit-floor has no reading for" in body


# --------------------------------------------------------------- guards (review 2026-09-17)
def test_placed_frame_is_refused_outside_the_placement_it_models():
    """_placed_head models facing continuity + xy chaining with no velocity
    blend and no anchor warp; any other build_draft setting would score a head
    the draft never writes (1.59 m / 1.9 rot6d off with continuity off)."""
    body = SOURCE.split("def infer_directory(", 1)[1].split("\ndef ", 1)[0]
    guard = body.split('if draft_join_frame == "placed" and not (', 1)[1][:400]
    for needed in ("draft_facing_continuity", 'draft_root_continuity == "xy"',
                   "not draft_root_velocity_blend", "not draft_beat_anchor",
                   "not draft_music_anchor"):
        assert needed in guard, needed


def test_unit_floor_shares_the_floor_guards_and_checks_its_provenance():
    body = SOURCE.split("def infer_directory(", 1)[1].split("\ndef ", 1)[0]
    assert '(draft_floor_normalize or draft_unit_floor) and "z" in str(draft_root_continuity)' in body
    assert "has no provenance sidecar" in body
    assert 'bool(provenance.get("floor_levelled")) != bool(levelled)' in body
    census = (ROOT / "tools/census_release_vertical.py").read_text()
    assert 'floor_out.with_suffix(".json")' in census and '"floor_levelled"' in census


def test_unit_floor_provenance_is_read_from_the_sidecar(tmp_path):
    from infer_atomic import _unit_floor_provenance
    npz = tmp_path / "floor.npz"
    assert _unit_floor_provenance(npz) is None
    (tmp_path / "floor.json").write_text('{"floor_levelled": false}')
    assert _unit_floor_provenance(npz) == {"floor_levelled": False}


def test_levelled_units_are_counted_where_they_are_laid_down():
    body = SOURCE.split("    def retrieve(", 1)[1].split("\n    def ", 1)[0]
    placed = body.split("values = self._values_at(chosen, target_length)", 1)[1][:300]
    assert "unit_floor_placed" in placed
    for key in ('"draft_unit_floor_units": getattr(library, "unit_floor_placed", 0)',
                '"draft_unit_floor_lookups"', '"retrieval_selector_labels_dir"',
                '"release_label_space_id"'):
        assert key in SOURCE, key
