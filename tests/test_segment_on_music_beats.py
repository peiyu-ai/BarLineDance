"""The music-beat segmenter, exercised where each promise can be broken.

The promise this arm was chosen for on 2026-08-20 is narrow and checkable: every
boundary sits on a beat and every segment is exactly ``k`` beats.  The first
release kept that promise for 14,376 of its 18,580 segments and broke it for the
other 4,204 -- the head and the tail, which were "whatever is left over" -- while
``merge_short`` quietly deleted 739 grid points on 33.2% of clips.  Nothing in the
artifact said so.

So the checks here come in pairs.  Every test that shows the validator refusing
something has a sibling that shows it accepting the same thing built correctly,
because a gate that only ever raises is as uninformative as one that never does.
"""

from __future__ import annotations

import copy
import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.segment_on_music_beats import (  # noqa: E402
    GridInvariantError,
    grid_bounds,
    main,
    validate_grid_records,
)

BEAT_CHANNEL = 34
ENVELOPE_CHANNEL = 0


def _clip(bundle: pathlib.Path, upload: str, beats, frames, *, loud=()):
    """One synthetic clip: a 35-D music table with a beat one-hot on channel 34.

    ``loud`` is the set of beat *indices* given extra onset envelope, which is how
    ``--phase energy`` is steered in a test: it picks the offset whose beats carry
    the most channel-0 energy.
    """
    table = np.zeros((frames, 35), dtype=np.float32)
    table[:, ENVELOPE_CHANNEL] = 0.1
    for index, frame in enumerate(beats):
        table[frame, BEAT_CHANNEL] = 1.0
        table[frame, ENVELOPE_CHANNEL] = 1.0 if index in loud else 0.3
    name = "{}__clip000".format(upload)
    relative = "sequences/{}/music_35.npy".format(name)
    path = bundle / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, table)
    return {"recording_id": "wild_v4:{}:clip000".format(upload),
            "music_path": relative, "frame_count": frames}, name


def _bundle(tmp_path, clips):
    """``clips`` is a list of ``(upload, beats, frames, loud)`` tuples."""
    bundle = tmp_path / "bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    rows, stems = [], []
    for upload, beats, frames, loud in clips:
        row, stem = _clip(bundle, upload, beats, frames, loud=loud)
        rows.append(row)
        stems.append(stem)
    (bundle / "sequences.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    listing = tmp_path / "clips.txt"
    listing.write_text("\n".join(stems) + "\n", encoding="utf-8")
    return bundle, listing, stems


def _run(tmp_path, clips, *extra):
    bundle, listing, stems = _bundle(tmp_path, clips)
    output = tmp_path / "run" / "segmentation.json"
    main(["--bundle", str(bundle), "--clips", str(listing),
          "--output", str(output), *extra])
    return json.loads(output.read_text(encoding="utf-8")), output, stems


# a 20 s clip at 120 BPM (15-frame beats) with a 1.0 s lead-in and 1.2 s trail-out
STEADY = ("7000000000000000001", list(range(30, 565, 15)), 600, ())


# --------------------------------------------------------------------------
# 1. --edges drop: the promise, and what it costs
# --------------------------------------------------------------------------

def test_drop_keeps_only_whole_beats_and_says_what_it_threw_away(tmp_path):
    """Positive control: a well-formed grid passes the gate and the loss is counted."""
    report, _, _ = _run(tmp_path, [STEADY], "--beats-per-segment", "4",
                        "--phase", "first")
    record = report["records"][0]
    beats = set(STEADY[1])
    assert all(b in beats for b in record["boundaries"])
    assert {s["frames"] for s in record["segments"]} == {60}
    # the lead-in and trail-out are gone, and both are reported rather than folded
    assert record["dropped_head_frames"] == 30
    assert record["dropped_tail_frames"] == 600 - 510   # last grid point is beat 32
    assert record["motion_frames"] == 600          # still the whole clip, for the
    assert record["covered_frames"] == 480         # stale-cut check downstream
    assert report["validation"]["enforced"] is True
    assert report["validation"]["structural_violations"] == 0
    assert report["coverage"]["kept_frames"] == 480


def test_drop_never_merges_even_when_a_segment_is_under_min_length(tmp_path):
    """Negative control for merge_short: the old path would have eaten these cuts.

    A 4-beat span at 600 BPM is 12 frames, below the default ``--min-length`` of
    18.  Under ``--edges drop`` the grid points survive; under ``--edges fold``
    the same clip loses them, which is the 739-cut behaviour of the first release.
    """
    fast = ("7000000000000000002", list(range(0, 300, 3)), 300, ())
    dropped, _, _ = _run(tmp_path / "a", [fast], "--phase", "first")
    assert {s["frames"] for s in dropped["records"][0]["segments"]} == {12}
    assert dropped["validation"]["structural_violations"] == 0

    folded, _, _ = _run(tmp_path / "b", [fast], "--phase", "first",
                        "--edges", "fold", "--allow-partial-edges")
    assert min(s["frames"] for s in folded["records"][0]["segments"]) >= 18
    assert len(folded["records"][0]["boundaries"]) < len(dropped["records"][0]["boundaries"])


def test_fold_is_refused_unless_its_defect_is_named(tmp_path):
    bundle, listing, _ = _bundle(tmp_path, [STEADY])
    with pytest.raises(SystemExit, match="partial by construction"):
        main(["--bundle", str(bundle), "--clips", str(listing),
              "--edges", "fold", "--output", str(tmp_path / "x.json")])


def test_fold_counts_the_partial_segments_it_makes(tmp_path):
    """Negative control on the pre-2026-08-20 output shape.

    ``--edges fold`` is allowed, but the artifact has to carry the number of
    segments that are not whole beats -- the old release carried no such field and
    the 4,204 partial segments were invisible.
    """
    report, _, _ = _run(tmp_path, [STEADY], "--phase", "first",
                        "--edges", "fold", "--allow-partial-edges")
    assert report["validation"]["enforced"] is False
    edges = report["validation"]["edges_included"]
    assert edges["structural_violations"] >= 2
    assert edges["counts"]["boundary_off_grid"] >= 2   # frame 0 and frame `frames`


# --------------------------------------------------------------------------
# 2. The validator itself: it has to fail on each break, and pass when clean
# --------------------------------------------------------------------------

def _good_record():
    beats = np.asarray(STEADY[1], dtype=np.int64)
    bounds = grid_bounds(beats, STEADY[2], 4, 0, 18, "drop")
    return {"sequence": "s", "boundaries": bounds,
            "segments": [{"start": a, "end": b, "frames": b - a}
                         for a, b in zip(bounds[:-1], bounds[1:])]}, {"s": beats}


def test_validator_passes_an_untampered_grid_cut(tmp_path):
    """Positive control: the ruler has to read clean on the thing it approves."""
    record, grids = _good_record()
    result = validate_grid_records([record], grids, 4)
    assert result["structural_violations"] == 0
    assert result["counts"]["duration_off_by_beats"] == 0
    assert result["segments_checked"] == len(record["segments"])


def test_validator_catches_a_boundary_nudged_off_the_grid():
    """Negative control: one frame off a beat is the failure Alg.1 shows at 93%."""
    record, grids = _good_record()
    broken = copy.deepcopy(record)
    broken["boundaries"][2] += 1
    broken["segments"] = [{"start": a, "end": b, "frames": b - a}
                          for a, b in zip(broken["boundaries"][:-1],
                                          broken["boundaries"][1:])]
    result = validate_grid_records([broken], grids, 4)
    # one cut off the grid is ONE defect, not two: it is the end of one segment and
    # the start of the next, and double-counting it would inflate the number a
    # reader compares against the 6.8%-on-beat figure for Alg.1 as shipped.
    assert result["counts"]["boundary_off_grid"] == 1
    assert result["structural_violations"] >= 1
    assert any("is not a beat" in e for e in result["examples"]["boundary_off_grid"])
    assert result["boundaries_checked"] == len(broken["boundaries"])


def test_validator_catches_a_merged_segment():
    """Negative control for the exact mutation merge_short performs.

    Deleting one grid point leaves a segment that is still bounded by beats -- so
    an "is every cut on a beat" check passes it -- but spans 8 beats instead of 4.
    This is the check the first release did not have.
    """
    record, grids = _good_record()
    broken = copy.deepcopy(record)
    del broken["boundaries"][2]
    broken["segments"] = [{"start": a, "end": b, "frames": b - a}
                          for a, b in zip(broken["boundaries"][:-1],
                                          broken["boundaries"][1:])]
    result = validate_grid_records([broken], grids, 4)
    assert result["counts"]["boundary_off_grid"] == 0
    assert result["counts"]["span_not_k_beats"] == 1
    assert "spans 8 beats, wanted 4" in result["examples"]["span_not_k_beats"][0]


def test_validator_catches_segments_that_do_not_tile_their_boundaries():
    record, grids = _good_record()
    broken = copy.deepcopy(record)
    broken["segments"] = broken["segments"][:-1]
    result = validate_grid_records([broken], grids, 4)
    assert result["counts"]["not_contiguous"] == 1


def test_a_broken_cut_stops_the_run_rather_than_being_written(tmp_path, monkeypatch):
    """The gate is wired to the artifact, not only available as a function."""
    import tools.segment_on_music_beats as tool

    def sabotage(beats, frames, k, phase, min_length, edges="drop"):
        bounds = grid_bounds(beats, frames, k, phase, min_length, edges)
        bounds[2] += 1                      # one frame off the grid
        return bounds

    monkeypatch.setattr(tool, "grid_bounds", sabotage)
    bundle, listing, _ = _bundle(tmp_path, [STEADY])
    with pytest.raises(GridInvariantError, match="structural violations"):
        tool.main(["--bundle", str(bundle), "--clips", str(listing),
                   "--phase", "first", "--output", str(tmp_path / "x.json")])
    assert not (tmp_path / "x.json").exists()


# --------------------------------------------------------------------------
# 3. Clips that cannot hold a whole segment are excluded BY NAME
# --------------------------------------------------------------------------

def test_a_clip_with_exactly_k_beats_yields_nothing_and_is_named(tmp_path):
    """The case the old beat-count test let through.

    ``len(beats) < max(2, k)`` passes a clip with exactly 4 beats, which then got
    one cut, no closing grid point, and -- once frame 0 and ``frames`` were added --
    an 11.03 s "segment" (``7574019615476924133__clip000`` in the real corpus).
    """
    thin = ("7000000000000000003", [40, 55, 70, 85], 400, ())
    report, output, stems = _run(tmp_path, [STEADY, thin], "--phase", "first")
    assert report["excluded"]["counts"]["no_whole_segment"] == 1
    assert report["excluded"]["clips"]["no_whole_segment"] == [stems[1]]
    assert [r["sequence"] for r in report["records"]] == [stems[0]]
    written = (output.parent / "excluded_clips.txt").read_text(encoding="utf-8")
    assert "{}\tno_whole_segment".format(stems[1]) in written
    assert (output.parent / "clips_kept.txt").read_text(
        encoding="utf-8").split() == [stems[0]]


def test_a_clip_with_fewer_than_two_beats_is_excluded_not_dropped_silently(tmp_path):
    lonely = ("7000000000000000004", [12], 400, ())
    report, _, stems = _run(tmp_path, [STEADY, lonely], "--phase", "first")
    assert report["excluded"]["clips"]["fewer_than_two_beats"] == [stems[1]]


def test_a_clip_the_bundle_does_not_carry_is_reported(tmp_path):
    """A clip listed in the corpus but absent from the bundle used to vanish."""
    bundle, listing, stems = _bundle(tmp_path, [STEADY])
    listing.write_text("\n".join(stems + ["7999999999999999999__clip000"]) + "\n",
                       encoding="utf-8")
    output = tmp_path / "run" / "segmentation.json"
    main(["--bundle", str(bundle), "--clips", str(listing), "--phase", "first",
          "--output", str(output)])
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["excluded"]["clips"]["not_in_bundle"] == ["7999999999999999999__clip000"]
    assert report["coverage"]["clips_requested"] == 2
    assert report["coverage"]["clips_kept"] == 1


# --------------------------------------------------------------------------
# 4. Things that are properties of the music, reported and not refused
# --------------------------------------------------------------------------

def test_tempo_drift_is_counted_and_does_not_stop_the_run(tmp_path):
    """A segment can be exactly 4 grid points and not 4 median-beats long.

    Beats at 10 frames for the first half and 20 for the second: every span is
    still exactly 4 grid points, so the structural checks pass, but measured
    against the clip's median inter-beat interval the spans read far from 4 and
    that has to be visible rather than either hidden or fatal.
    """
    beats = list(range(20, 220, 10)) + list(range(240, 640, 20))
    drifting = ("7000000000000000005", beats, 700, ())
    report, _, _ = _run(tmp_path, [drifting], "--phase", "first")
    assert report["validation"]["structural_violations"] == 0
    assert report["validation"]["counts"]["duration_off_by_beats"] > 0
    assert report["tempo"]["drift_median"] > 0.3


def test_a_steady_grid_reports_no_duration_deviation(tmp_path):
    """Positive control for the same measurement."""
    report, _, _ = _run(tmp_path, [STEADY], "--phase", "first")
    assert report["validation"]["counts"]["duration_off_by_beats"] == 0
    assert report["tempo"]["drift_median"] == 0.0
    assert report["tempo"]["bpm_median"] == pytest.approx(120.0)


def test_grid_coverage_is_recorded_and_only_excludes_when_asked(tmp_path):
    """Off by default is a deliberate choice, so both settings are pinned.

    The uncovered stretch of a low-coverage clip is *quieter* than the covered
    one on real data (0.757 vs 0.829 outside/inside onset envelope), so there is
    no evidence it means a tracker failure and the default is 0.0.
    """
    partial = ("7000000000000000006", list(range(10, 200, 15)), 800, ())
    kept, _, stems = _run(tmp_path / "a", [partial], "--phase", "first")
    assert kept["records"][0]["grid_coverage"] < 0.3
    assert kept["excluded"]["counts"] == {}

    with pytest.raises(Exception) as excinfo:
        _run(tmp_path / "b", [partial], "--phase", "first",
             "--min-grid-coverage", "0.5")
    assert "no clip produced a segmentation" in str(excinfo.value)


def test_phase_energy_moves_the_first_cut_off_the_first_beat(tmp_path):
    """The downbeat guess is a real choice, so it has to be able to change one.

    Beats 2, 6, 10, ... carry the loud envelope, so ``--phase energy`` must pick
    offset 2 and start the cut list at beat index 2, where ``--phase first``
    starts at index 0.
    """
    loud = ("7000000000000000007", list(range(30, 630, 15)), 700, set(range(2, 40, 4)))
    energy, _, _ = _run(tmp_path / "a", [loud], "--phase", "energy")
    first, _, _ = _run(tmp_path / "b", [loud], "--phase", "first")
    assert energy["records"][0]["phase"] == 2
    assert energy["records"][0]["boundaries"][0] == loud[1][2]
    assert first["records"][0]["boundaries"][0] == loud[1][0]
    assert energy["downbeat"]["detected"] is False


def test_the_anchor_artifact_in_the_phase_rule_is_counted(tmp_path):
    """Negative control: a clip whose phase is decided by ``beats[0]`` alone.

    ``librosa.beat.beat_track`` anchors on a strong onset, so ``beats[0]`` is loud
    (median z=+0.93 within its own clip on clean5) and it is in offset 0's score
    and no other offset's.  Here only ``beats[0]`` is loud, so the shipped rule
    picks offset 0 and the beats[0]-excluded rule picks something else, and the
    artifact has to carry that the two disagree instead of publishing one of them
    as if it were settled.
    """
    anchored = ("7000000000000000008", list(range(30, 630, 15)), 700, {0})
    report, _, _ = _run(tmp_path, [anchored], "--phase", "energy")
    assert report["records"][0]["phase"] == 0
    artifact = report["downbeat"]["anchor_artifact"]
    assert artifact["clips_compared"] == 1
    assert artifact["clips_whose_phase_changes_without_beats0"] == 1


def test_a_clip_with_a_real_loud_offset_survives_the_anchor_correction(tmp_path):
    """Positive control: an offset that is loud on many beats, not on one.

    Beats 2, 6, 10, ... are loud, so both versions of the rule pick offset 2 and
    the disagreement count reads 0 -- the correction must not fire on a phase that
    was decided by the music rather than by the tracker's first frame.
    """
    real = ("7000000000000000009", list(range(30, 630, 15)), 700, set(range(2, 40, 4)))
    report, _, _ = _run(tmp_path, [real], "--phase", "energy")
    assert report["records"][0]["phase"] == 2
    assert report["downbeat"]["anchor_artifact"][
        "clips_whose_phase_changes_without_beats0"] == 0
