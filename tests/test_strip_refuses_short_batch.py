"""The strip must refuse a batch that is not the fixed ten.

The branch this pins used to print "skip <clip> (missing asset)" and carry on to
exit 0.  On 2026-09-13 four of the ten clips were dropped that way -- their
``ingest_v1_converted`` directories are not on this machine -- and the batch
would have been handed over as though it were the fixed set.  CLAUDE.md 1.5
rule 6 fixes those ten precisely so a round cannot be compared against a
different set: 换片子等于换尺子, and a silent skip is how the set changes.
"""
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
TOOL = REPO / "tools/render_sample_strip.py"


def run(clips_file, extra=()):
    env = dict(os.environ)
    env.update({"STRIP_AUDIO": "runs/txy_t_gt_eval/audio",
                "STRIP_GT_MOTION": "runs/txy_t_gt_eval/motion",
                "STRIP_LABELS": "data/wild3d/txy_t_labels"})
    return subprocess.run(
        [sys.executable, str(TOOL), "--clips", str(clips_file),
         "--arm", "x=runs/t_beat/alignbase", "--out", "/tmp/strip_short_probe",
         *extra],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=600)


def bogus(tmp_path):
    path = tmp_path / "clips.txt"
    path.write_text("wild_v5:9999999999999999999:clip000\n")
    return path


def test_a_clip_that_cannot_render_fails_the_run(tmp_path):
    assert run(bogus(tmp_path)).returncode != 0


def test_the_failure_names_the_clip_and_every_missing_file(tmp_path):
    """"missing asset" sent the reader back into the code to find out WHICH."""
    text = "".join(run(bogus(tmp_path))[1:3] if False else
                   [run(bogus(tmp_path)).stdout, run(bogus(tmp_path)).stderr])
    assert "9999999999999999999__clip000" in text
    assert "audio.wav" in text and "STRIP_CONVERTED" in text


def test_the_failure_says_how_to_fix_it(tmp_path):
    result = run(bogus(tmp_path))
    text = result.stdout + result.stderr
    assert "stage_ground_truth_for_render" in text


def test_a_deliberate_subset_can_opt_out(tmp_path):
    """A short batch is sometimes what you meant; it just has to be said out
    loud rather than happening by default."""
    result = run(bogus(tmp_path), extra=("--allow-missing-clips",))
    assert result.returncode == 0
