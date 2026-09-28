"""The audio rebuild must find the right span and must not touch the ingest tree.

Both failure modes here were real on 2026-09-22: pairing a recut row's
``clips[i]`` with ``spans[i]`` put 19 clip000s at 12-23 s into their uploads,
and the ingest tree is the /cache read copy of a shared OSS tree whose
``clip.mp4`` bytes every 3D freshness hash names.  The decode itself is gated
at run time (``check_decoder``), not here, because it needs a real upload.
"""

import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.rebuild_clip_audio import build_overlay, overlay_entries, resolve_span  # noqa: E402


class ResolveSpanTests(unittest.TestCase):
    def test_meta_span_wins_over_manifest(self):
        meta = {"source": "u.mp4", "source_frame_span": [475, 950]}
        rows = [{"source": "other.mp4", "spans": [[0, 475, "x"]],
                 "clips": [{"clip": "1__clip000", "status": "ok"}]}]
        self.assertEqual(resolve_span("1__clip001", meta, rows), ("u.mp4", 475, 950))

    def test_recut_marker_entries_do_not_shift_the_pairing(self):
        """A recut row lists 'recut' and 'ok' for each clip; pairing by raw index
        would give clip001 the *first* span and clip002 the second."""
        row = {"source": "u.mp4", "ingested_at": 2.0,
               "spans": [[0, 525, "a"], [525, 1051, "b"], [1051, 1577, "c"]],
               "clips": [{"clip": "1__clip000", "status": "recut"},
                         {"clip": "1__clip000", "status": "ok"},
                         {"clip": "1__clip001", "status": "recut"},
                         {"clip": "1__clip001", "status": "ok"},
                         {"clip": "1__clip002", "status": "recut"},
                         {"clip": "1__clip002", "status": "ok"}]}
        self.assertEqual(resolve_span("1__clip000", {}, [row]), ("u.mp4", 0, 525))
        self.assertEqual(resolve_span("1__clip001", {}, [row]), ("u.mp4", 525, 1051))
        self.assertEqual(resolve_span("1__clip002", {}, [row]), ("u.mp4", 1051, 1577))

    def test_latest_row_wins(self):
        old = {"source": "old.mp4", "ingested_at": 1.0, "spans": [[0, 400, "a"]],
               "clips": [{"clip": "1__clip000", "status": "ok"}]}
        new = {"source": "new.mp4", "ingested_at": 2.0, "spans": [[0, 480, "a"]],
               "clips": [{"clip": "1__clip000", "status": "ok"}]}
        self.assertEqual(resolve_span("1__clip000", {}, [new, old]), ("new.mp4", 0, 480))

    def test_unknown_clip_is_an_error_not_a_default(self):
        with self.assertRaises(KeyError):
            resolve_span("9__clip000", {}, [])


class OverlayTests(unittest.TestCase):
    def test_overlay_links_everything_but_audio_and_leaves_source_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = pathlib.Path(tmp) / "ingest" / "1__clip000"
            (src / "preprocess").mkdir(parents=True)
            for name, body in [("clip.mp4", b"v"), ("audio.wav", b"old"),
                               ("meta.json", json.dumps({}).encode()),
                               ("keypoints.npy", b"k")]:
                (src / name).write_bytes(body)
            (src / "preprocess" / "bbx.pt").write_bytes(b"b")
            self.assertEqual(overlay_entries(src),
                             ["clip.mp4", "keypoints.npy", "meta.json", "preprocess"])
            dst = pathlib.Path(tmp) / "overlay" / "1__clip000"
            build_overlay(src, dst)
            build_overlay(src, dst)  # idempotent: our own links are accepted
            self.assertFalse((dst / "audio.wav").exists())
            self.assertTrue((dst / "clip.mp4").is_symlink())
            self.assertEqual((dst / "preprocess" / "bbx.pt").read_bytes(), b"b")
            (dst / "audio.wav").write_bytes(b"new")
            self.assertEqual((src / "audio.wav").read_bytes(), b"old")

    def test_a_foreign_file_in_the_overlay_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = pathlib.Path(tmp) / "ingest" / "1__clip000"
            src.mkdir(parents=True)
            (src / "clip.mp4").write_bytes(b"v")
            dst = pathlib.Path(tmp) / "overlay" / "1__clip000"
            dst.mkdir(parents=True)
            (dst / "clip.mp4").write_bytes(b"someone else's")
            with self.assertRaises(FileExistsError):
                build_overlay(src, dst)


if __name__ == "__main__":
    unittest.main()
