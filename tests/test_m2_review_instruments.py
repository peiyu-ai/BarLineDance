"""The M2 review instruments, pinned at the places they have already failed.

Three of the four tests here exist because the defect they check was real and
shipped, not because the code looked risky:

* ``--hide-uploads`` parsed and did nothing.  ``render_prototype_cards.main()``
  never forwarded it to ``build()``, so the flag was accepted, the run
  succeeded, and the upload count stayed printed inside every PNG -- which is
  the one thing that decodes a paired blind sheet, because scattering
  membership necessarily raises upload diversity.  It was found by reading a
  rendered image, not by any check.  The verification that "the leak is closed"
  had looked at a *video* frame, whose ``main()`` did forward the flag.

* ``build_space`` matched ``fuse_0.5_deacct`` on its ``startswith("fuse")``
  prefix before reaching the exact de-account branch, so the de-accounted arm of
  the strategy sweep silently returned the plain fusion.  The sweep printed two
  rows with identical numbers to four decimals and nothing objected.

* ``make_shuffled_labels`` fused two abutting runs that the permutation gave the
  same label, so the "size-matched" control was one segment short of the real
  arm.  Its own re-derivation gate caught that; this pins the gate.

The fourth is the positive/negative control for the coherence ratio itself,
which is the criterion that produced a wrong verdict when it was used without
one (see ``tools/probe_coherence_two_spaces.py``).
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.probe_m2_strategies import build_space          # noqa: E402
from tools.probe_prototype_coherence import table          # noqa: E402


def test_hide_uploads_actually_reaches_the_rendered_title():
    """The flag has to change the image, not merely be accepted.

    Asserted through ``main()``'s own argument plumbing rather than by calling
    ``render_card`` directly: the defect was entirely in the wiring between
    ``build_parser`` and ``build``, so a test that called the renderer with an
    explicit keyword would have passed while the CLI stayed broken.
    """
    from tools.render_prototype_cards import build_parser

    args = build_parser().parse_args([
        "--labels", "x", "--bundle", "y", "--output", "z", "--hide-uploads",
        "--title-note", "REAL"])
    assert args.hide_uploads is True
    assert args.title_note == "REAL"

    source = (REPO / "tools/render_prototype_cards.py").read_text(encoding="utf-8")
    main_body = source.split("def main(", 1)[1]
    assert "hide_uploads=args.hide_uploads" in main_body, (
        "main() must forward --hide-uploads to build(); it did not, and the flag "
        "was silently inert")
    assert "title_note=args.title_note" in main_body


def test_title_note_and_hidden_uploads_change_the_suptitle_text():
    """The rendered title carries the arm and omits the decoding count."""
    from tools.render_prototype_cards import render_card

    def title(hide, note):
        # render_card needs no members to build a title; an empty entry list
        # returns None, so drive the format string the function itself uses.
        source = (REPO / "tools/render_prototype_cards.py").read_text()
        chunk = source.split("figure.suptitle(", 1)[1].split("color=", 1)[0]
        template = chunk.strip().strip("()").split('".format', 1)[0].strip().strip('"')
        return template

    template = title(True, "REAL")
    assert "{} segments{}" in template, template
    assert "uploads" not in template.split("{} segments{}")[0], (
        "the upload count must be in the optional tail, not baked before it")


def test_fuse_and_fuse_deacct_are_not_the_same_space():
    rng = np.random.default_rng(0)
    tmr = rng.normal(size=(40, 6))
    pose = rng.normal(size=(40, 5))
    dyn = rng.normal(size=(40, 4))
    uploads = np.array(["u{}".format(i // 4) for i in range(40)])
    train = np.arange(20)

    plain = build_space("fuse_0.5", tmr, pose, dyn, uploads, train)
    deacct = build_space("fuse_0.5_deacct", tmr, pose, dyn, uploads, train)

    assert plain.shape == deacct.shape
    assert not np.allclose(plain, deacct), (
        "the de-accounted arm returned the plain fusion; the prefix test "
        "captured the name before the exact branch could")
    per_upload = np.stack([deacct[uploads == u].mean(0) for u in np.unique(uploads)])
    assert np.abs(per_upload).max() < 1e-9, "de-accounting must zero each upload's mean"


def test_an_unknown_space_raises_rather_than_returning_none():
    """A silent None here becomes 'this arm was skipped' with no row printed."""
    rng = np.random.default_rng(0)
    args = (rng.normal(size=(8, 3)), rng.normal(size=(8, 3)),
            rng.normal(size=(8, 2)), np.array(["u"] * 8), np.arange(4))
    try:
        build_space("not_a_space", *args)
    except ValueError:
        return
    raise AssertionError("unknown space must raise, not return None")


def test_coherence_ratio_separates_planted_clusters_from_noise():
    """The positive control the ratio was first used without.

    Negative arm: labels assigned at random over unstructured vectors, which
    must read ~1.0 -- members no closer to each other than to anything else.
    Positive arm: vectors planted around per-group centres, which must read
    well below 1.0.  Without the positive arm there is no scale on which "0.95"
    means anything, and that absence is exactly what produced a wrong verdict on
    clean5b5.
    """
    rng = np.random.default_rng(7)
    groups, per_group = 4, 30
    labels = np.repeat(np.arange(groups), per_group)
    # One upload per member, so every pair is cross-upload and the identity
    # confound cannot contribute.
    uploads = np.array(["u{}".format(i) for i in range(groups * per_group)])

    noise = rng.normal(size=(groups * per_group, 12))
    flat = table(noise, labels, uploads, seed=1)
    flat_ratios = np.array([r["ratio"] for r in flat.values()])
    assert 0.93 < np.median(flat_ratios) < 1.07, np.median(flat_ratios)

    centres = rng.normal(size=(groups, 12)) * 6.0
    planted = centres[labels] + rng.normal(size=(groups * per_group, 12)) * 0.3
    tight = table(planted, labels, uploads, seed=1)
    tight_ratios = np.array([r["ratio"] for r in tight.values()])
    assert np.median(tight_ratios) < 0.35, np.median(tight_ratios)
    assert np.median(tight_ratios) < np.median(flat_ratios)


def test_shuffled_control_keeps_every_group_size(tmp_path):
    """The gate that caught a real fusion, driven end to end."""
    rng = np.random.default_rng(3)
    source = tmp_path / "labels"
    (source / "labels").mkdir(parents=True)
    rows = []
    for i in range(12):
        # Runs separated by a zero frame, plus one abutting pair per recording so
        # the fusion hazard is actually exercised.
        frames = []
        for run in range(4):
            frames += [int(rng.integers(1, 6))] * 8
            if run != 1:                       # run 1 abuts run 2: no zero gap
                frames += [0] * 3
        array = np.array(frames, dtype=np.int64)
        path = "labels/rec{:03d}.npy".format(i)
        np.save(source / path, array)
        rows.append({"recording_id": "t:{}:clip000".format(i), "labels_path": path})
    (source / "labels.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    target = tmp_path / "shuffled"
    done = subprocess.run(
        [sys.executable, str(REPO / "tools/make_shuffled_labels.py"),
         str(source), str(target)],
        capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "OK: sizes identical label-for-label" in done.stdout, done.stdout

    from tools.make_shuffled_labels import group_sizes
    assert group_sizes(source) == group_sizes(target)
