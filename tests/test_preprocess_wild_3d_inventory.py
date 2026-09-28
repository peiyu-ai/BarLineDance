"""A directory glob is not an enumeration of this corpus.

Nothing deletes a clip directory, and these credentials cannot delete an OSS
object (``ossutil rm`` answers 403 AccessDenied because of bucket acl), so an
upload that stops producing a clip leaves it on disk looking exactly like a
current one -- same name shape, same meta.json, same everything except that no
run produces it.  Measured 2026-08-25 on the live ingest tree: 17,985
directories carry a meta.json while the manifests produce 17,225, so a glob
hands back **760 orphans**.

Re-admitting them is invisible downstream.  An orphan's 3D, S3D and music were
all built from its own old bytes, so every hash among them agrees and
``audit_clip_freshness`` calls it fresh -- freshness is not membership.  The
exclusion has to happen here, at the only point that enumerates.
"""

import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools import preprocess_wild_3d


def _cache(root, names):
    root = pathlib.Path(root)
    for name in names:
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "meta.json").write_text(
            json.dumps({"fps": 30.0, "source": "/x/{}.mp4".format(name)}),
            encoding="utf-8")
    return root


def _inventory(root, **kwargs):
    return preprocess_wild_3d.inventory_wild_cache(
        root, recursive=False, min_frames=1, min_visible_fraction=0.0,
        min_score=0.0, max_frozen_fraction=1.0, **kwargs)


class InventoryExclusion(unittest.TestCase):
    def test_named_clips_are_left_out_and_the_rest_are_kept(self):
        with tempfile.TemporaryDirectory() as raw:
            root = _cache(raw, ["u__clip000", "u__clip001", "u__clip002"])
            kept = _inventory(root, exclude=["u__clip001"])
            self.assertEqual(sorted(r["clip_id"] for r in kept),
                             ["u__clip000", "u__clip002"])

    def test_no_exclusion_list_keeps_the_old_behaviour(self):
        # The flag is opt-in, so every existing caller has to keep working --
        # and the summary records that no list was given, because "no list" and
        # "empty list" mean opposite things about whether orphans are present.
        with tempfile.TemporaryDirectory() as raw:
            root = _cache(raw, ["u__clip000", "u__clip001"])
            self.assertEqual(len(_inventory(root)), 2)

    def test_blank_lines_do_not_exclude_a_nameless_clip(self):
        with tempfile.TemporaryDirectory() as raw:
            root = _cache(raw, ["u__clip000"])
            self.assertEqual(len(_inventory(root, exclude=["", "  ", "\n"])), 1)

    def test_a_missing_exclusion_file_is_refused_rather_than_treated_as_empty(self):
        import argparse
        with tempfile.TemporaryDirectory() as raw:
            root = _cache(raw, ["u__clip000"])
            args = argparse.Namespace(
                cache_root=str(root), output=str(pathlib.Path(raw) / "out.jsonl"),
                recursive=False, min_frames=1, min_visible_fraction=0.0,
                min_score=0.0, max_frozen_fraction=1.0,
                exclude=str(pathlib.Path(raw) / "nope.txt"))
            with self.assertRaises(FileNotFoundError) as caught:
                preprocess_wild_3d.command_inventory(args)
            self.assertIn("760", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
