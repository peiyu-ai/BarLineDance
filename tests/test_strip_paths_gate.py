"""The sample strip must refuse to render a strip that is missing a panel.

WHY.  Its three source paths default to the v5 line.  Pointed at a T-line run
they simply do not resolve, and every read of them is a soft `if path.is_file()`
-- so the strip still renders, WITHOUT the music envelope and WITHOUT the
ground-truth lane, and prints "0.00 on beat" for every arm as though that had
been measured.  A gate that reads like "checked" while checking nothing is the
failure this repository names first in CLAUDE.md section 2.  The same default
had already cost a round once (a strip that dropped five ground truths, exit 0).
"""
import os
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
TOOL = REPO / "tools/render_sample_strip.py"


def run(env_overrides, clips="runs/vis_clips_t10.txt"):
    env = dict(os.environ)
    env.update(env_overrides)
    return subprocess.run(
        [sys.executable, str(TOOL), "--clips", clips,
         "--arm", "x=runs/t_beat/alignbase", "--out", "/tmp/strip_gate_probe"],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=300)


def test_a_missing_source_path_is_refused_and_names_itself():
    result = run({"STRIP_AUDIO": "runs/does_not_exist",
                  "STRIP_GT_MOTION": "runs/txy_t_gt_eval/motion",
                  "STRIP_LABELS": "data/wild3d/txy_t_labels"})
    assert result.returncode != 0
    assert "STRIP_AUDIO" in result.stderr + result.stdout


def test_the_refusal_says_what_would_have_been_lost():
    """A refusal that does not say WHICH panel disappears sends the reader back
    into the code, which is what CLAUDE.md section 4 forbids."""
    result = run({"STRIP_AUDIO": "runs/nope", "STRIP_GT_MOTION": "runs/nope",
                  "STRIP_LABELS": "runs/nope"})
    text = result.stderr + result.stdout
    assert "music" in text and "ground-truth" in text and "on beat" in text


def test_the_refusal_gives_the_t_line_paths():
    """The fix has to be in the error, not in someone's memory."""
    result = run({"STRIP_AUDIO": "runs/nope", "STRIP_GT_MOTION": "runs/nope",
                  "STRIP_LABELS": "runs/nope"})
    text = result.stderr + result.stdout
    assert "runs/txy_t_gt_eval/audio" in text
    assert "data/wild3d/txy_t_labels" in text


def test_every_path_is_checked_not_just_the_first():
    """Checking only the first would let the other two fail silently."""
    for missing in ("STRIP_AUDIO", "STRIP_GT_MOTION", "STRIP_LABELS"):
        env = {"STRIP_AUDIO": "runs/txy_t_gt_eval/audio",
               "STRIP_GT_MOTION": "runs/txy_t_gt_eval/motion",
               "STRIP_LABELS": "data/wild3d/txy_t_labels"}
        env[missing] = "runs/definitely_not_here"
        result = run(env)
        assert result.returncode != 0, missing
        assert missing in result.stderr + result.stdout, missing


def test_the_staging_directory_is_per_process():
    """The tool is run several at a time to render ten clips in parallel; a
    shared staging tree makes the first worker to finish delete files the others
    are still writing."""
    assert 'staging = out / ".staging-{}".format(os.getpid())' in TOOL.read_text()
