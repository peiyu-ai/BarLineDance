"""The review-render clip list must not be able to serve a defective clip.

The controls here are the two directions the filter can fail in, and the
second one is the one that actually happened: a filter that exists but is not
applied to the list a human is shown reads exactly like a filter that works.
"""
import json
import pathlib
import subprocess
import sys

import pytest

TOOL = pathlib.Path(__file__).resolve().parents[1] / "tools" / "build_vis_clip_list.py"


def build_release(root, split, sequences):
    (root / split).mkdir(parents=True)
    names = ["{}_slice{}".format(sequence, index)
             for sequence in sequences for index in range(2)]
    (root / split / "names.json").write_text(json.dumps(names))
    return root


def write_manifest(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def run(*args):
    return subprocess.run([sys.executable, str(TOOL), *args],
                          capture_output=True, text=True)


@pytest.fixture()
def corpus(tmp_path):
    sequences = ["wild_v5:{}:clip000".format(1000 + i) for i in range(8)]
    release = build_release(tmp_path / "release", "val", sequences)
    manifest = tmp_path / "exclude.jsonl"
    write_manifest(manifest, [
        {"sequence": "wild_v5:1003:clip000", "clip": "1003__clip000",
         "scale": 0.5, "source_fps": 60.0},
    ])
    return release, manifest, sequences


def test_excluded_clip_never_reaches_the_list(corpus, tmp_path):
    release, manifest, _ = corpus
    output = tmp_path / "clips.txt"
    result = run("--release", str(release), "--split", "val", "--count", "7",
                 "--exclude", str(manifest), "--population", str(tmp_path / "missing.json"),
                 "--output", str(output))
    assert result.returncode == 0, result.stderr
    chosen = output.read_text().split()
    assert len(chosen) == 7
    assert "wild_v5:1003:clip000" not in chosen


def test_unfiltered_corpus_would_have_served_it(corpus, tmp_path):
    """The positive control: with an EMPTY manifest the defective clip is
    served.  Without this the test above passes on a tool that simply never
    picks that clip, which is the same reading a working filter gives."""
    release, _, _ = corpus
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    output = tmp_path / "clips_all.txt"
    result = run("--release", str(release), "--split", "val", "--count", "8",
                 "--exclude", str(empty), "--population", str(tmp_path / "missing.json"),
                 "--output", str(output))
    assert result.returncode == 0, result.stderr
    assert "wild_v5:1003:clip000" in output.read_text().split()


def test_a_defective_pin_is_refused_not_silently_replaced(corpus, tmp_path):
    """A pin is a reviewer naming a clip.  Substituting another one quietly is
    worse than failing: the reviewer believes they are looking at what they
    asked for."""
    release, manifest, _ = corpus
    output = tmp_path / "clips.txt"
    result = run("--release", str(release), "--split", "val", "--count", "5",
                 "--exclude", str(manifest), "--population", str(tmp_path / "missing.json"),
                 "--pin", "1003__clip000", "--output", str(output))
    assert result.returncode != 0
    assert "time-base defective" in result.stderr
    assert "60.0" in result.stderr
    assert not output.exists()


def test_a_pin_outside_the_split_is_refused(corpus, tmp_path):
    release, manifest, _ = corpus
    output = tmp_path / "clips.txt"
    result = run("--release", str(release), "--split", "val", "--count", "5",
                 "--exclude", str(manifest), "--population", str(tmp_path / "missing.json"),
                 "--pin", "9999__clip000", "--output", str(output))
    assert result.returncode != 0
    assert "not in" in result.stderr


def test_pins_come_first_and_the_fill_is_deterministic(corpus, tmp_path):
    release, manifest, _ = corpus
    first, second = tmp_path / "a.txt", tmp_path / "b.txt"
    for output in (first, second):
        result = run("--release", str(release), "--split", "val", "--count", "5",
                     "--exclude", str(manifest), "--population", str(tmp_path / "missing.json"),
                     "--pin", "1005__clip000", "--pin", "1001__clip000",
                     "--output", str(output))
        assert result.returncode == 0, result.stderr
    chosen = first.read_text().split()
    assert chosen[:2] == ["wild_v5:1005:clip000", "wild_v5:1001:clip000"]
    assert first.read_text() == second.read_text()


def test_cleanliness_ordering_is_an_ordering_not_a_gate(corpus, tmp_path):
    """A clip with a rival dancer must still be selectable -- ranking must not
    quietly become a second filter."""
    release, manifest, sequences = corpus
    population = tmp_path / "population.json"
    population.write_text(json.dumps({"rows": [
        {"clip": "1007__clip000", "subject_area_median": 0.4,
         "rival_ratio": 0.0, "crop_travel_p95": 0.01},
        {"clip": "1000__clip000", "subject_area_median": 0.1,
         "rival_ratio": 0.9, "crop_travel_p95": 0.9},
    ]}))
    output = tmp_path / "clips.txt"
    result = run("--release", str(release), "--split", "val", "--count", "7",
                 "--exclude", str(manifest), "--population", str(population),
                 "--rank-by-cleanliness", "--output", str(output))
    assert result.returncode == 0, result.stderr
    chosen = output.read_text().split()
    assert chosen[0] == "wild_v5:1007:clip000"
    assert "wild_v5:1000:clip000" in chosen
