import io
import json
import os
import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from tools import asset_io


class PathMappingTests(unittest.TestCase):
    def test_repo_relative_paths_land_under_this_repo_prefix(self):
        self.assertEqual(
            asset_io.remote_path("data/wild3d/wild_performance_v1/sequences.jsonl"),
            "example-bucket/example/prefix/"
            "AtomicDance/data/wild3d/wild_performance_v1/sequences.jsonl")

    def test_a_leading_slash_does_not_escape_the_prefix(self):
        self.assertEqual(asset_io.remote_path("/data/x.npy"),
                         asset_io.remote_path("data/x.npy"))

    def test_bucket_and_prefix_are_overridable_for_another_corpus(self):
        with mock.patch.dict(os.environ, {"ATOMICDANCE_OSS_BUCKET": "other",
                                          "ATOMICDANCE_OSS_PREFIX": "some/where"}):
            self.assertEqual(asset_io.remote_path("a/b.npy"), "other/some/where/a/b.npy")


class ResolutionOrderTests(unittest.TestCase):
    """A file in the checkout wins, so existing call sites keep working."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        patcher = mock.patch.object(asset_io, "REPO_ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def test_a_local_file_is_read_without_touching_oss(self):
        target = self.root / "data" / "here.json"
        target.parent.mkdir(parents=True)
        target.write_text('{"a": 1}', encoding="utf-8")
        with mock.patch.object(asset_io, "filesystem") as filesystem:
            self.assertEqual(asset_io.read_json("data/here.json"), {"a": 1})
        filesystem.assert_not_called()

    def test_a_missing_file_falls_through_to_oss(self):
        payload = b'{"a": 2}'
        with mock.patch.object(asset_io, "_fetch_via_ossutil", return_value=payload) as fetch:
            self.assertEqual(asset_io.read_json("data/absent.json"), {"a": 2})
        fetch.assert_called_once_with("data/absent.json")

    def test_a_remote_read_does_not_go_through_fsspec_first(self):
        """ossutil is tried first, and fsspec is never reached when it works.

        ``ossfs.open`` resolves the object's size by listing its parent
        directory, and a deep LIST on this bridge bucket is the operation that
        fails.  Trying it first and catching the exception worked only while the
        LIST *failed*; on 2026-08-15 it stopped answering instead, and with no
        timeout on that path the fallback was unreachable -- one 28 MiB
        ``clip.mp4`` held a captioning worker for over twenty minutes with the
        log's last line still saying it was staging a batch."""
        with mock.patch.object(asset_io, "_fetch_via_ossutil", return_value=b"x") as fetch:
            with mock.patch.object(asset_io, "filesystem") as filesystem:
                self.assertEqual(asset_io.read_bytes("data/absent.bin"), b"x")
        fetch.assert_called_once()
        filesystem.assert_not_called()

    def test_fsspec_still_covers_an_ossutil_failure(self):
        """The reversal must not remove a route, only reorder it."""
        fake = mock.MagicMock()
        fake.open.return_value = io.BytesIO(b'{"a": 3}')
        with mock.patch.object(asset_io, "_fetch_via_ossutil",
                               side_effect=asset_io.AssetIOError("nope")):
            with mock.patch.object(asset_io, "filesystem", return_value=fake):
                self.assertEqual(asset_io.read_json("data/absent.json"), {"a": 3})
        fake.open.assert_called_once()
        self.assertIn("AtomicDance/data/absent.json", fake.open.call_args[0][0])

    def test_a_directory_is_not_mistaken_for_a_local_file(self):
        # local_path must test is_file, not exists: a directory of the same
        # name would otherwise be opened and fail with IsADirectoryError
        # instead of falling through to the object store.
        (self.root / "data" / "bundle").mkdir(parents=True)
        self.assertIsNone(asset_io.local_path("data/bundle"))


class CredentialTests(unittest.TestCase):
    def test_the_sts_token_is_preferred_and_read_fresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            token = pathlib.Path(tmp) / "token"
            token.write_text(json.dumps({
                "access_key_id": "STS.one", "access_key_secret": "s1",
                "security_token": "t1", "expired_at": "2026-08-12T00:00:00Z"}),
                encoding="utf-8")
            with mock.patch.object(asset_io, "STS_TOKEN", token):
                first = asset_io._credentials()
                self.assertEqual(first["key"], "STS.one")
                self.assertEqual(first["token"], "t1")
                # Rotation: the platform rewrites the file in place, and a
                # 36-hour token outlives most training runs, so the value must
                # not be cached from the first read.
                token.write_text(json.dumps({
                    "access_key_id": "STS.two", "access_key_secret": "s2",
                    "security_token": "t2", "expired_at": "2026-08-13T00:00:00Z"}),
                    encoding="utf-8")
                second = asset_io._credentials()
        self.assertEqual(second["key"], "STS.two")
        self.assertNotEqual(first["stamp"], second["stamp"])

    def test_the_filesystem_is_rebuilt_when_the_credential_rotates(self):
        calls = []

        class FakeOSSFS:
            def __init__(self, **kwargs):
                calls.append(kwargs["key"])

        module = mock.MagicMock()
        module.OSSFileSystem = FakeOSSFS
        stamps = iter([
            {"key": "a", "secret": "s", "token": "t", "stamp": "exp-1"},
            {"key": "a", "secret": "s", "token": "t", "stamp": "exp-1"},
            {"key": "b", "secret": "s", "token": "t", "stamp": "exp-2"},
        ])
        with mock.patch.dict("sys.modules", {"ossfs": module}), \
             mock.patch.object(asset_io, "_credentials", lambda: next(stamps)), \
             mock.patch.object(asset_io, "_filesystem", None), \
             mock.patch.object(asset_io, "_credential_stamp", None):
            asset_io.filesystem()
            asset_io.filesystem()      # same stamp -> reuse
            asset_io.filesystem()      # rotated  -> rebuild
        self.assertEqual(calls, ["a", "b"])

    def test_no_credentials_at_all_is_an_error_not_an_anonymous_client(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(asset_io, "STS_TOKEN", pathlib.Path(tmp) / "absent"), \
                 mock.patch.dict(os.environ, {"OSSUTIL_CONFIG": str(pathlib.Path(tmp) / "none")}):
                with self.assertRaises(asset_io.AssetIOError):
                    asset_io._credentials()

    def test_no_long_lived_key_is_embedded_in_the_module(self):
        import inspect

        self.assertNotIn("LTAI", inspect.getsource(asset_io))


class NpyTests(unittest.TestCase):
    def test_remote_arrays_are_buffered_whole_before_numpy_seeks_them(self):
        # np.load seeks; handing it a streamed object turns every seek into a
        # fresh ranged request.  One round trip for a 250 KB array beats a
        # dozen, so the bytes are read once into memory first.
        import numpy as np

        array = np.arange(12, dtype=np.float32).reshape(3, 4)
        buffer = io.BytesIO()
        np.save(buffer, array)
        with mock.patch.object(asset_io, "local_path", return_value=None), \
             mock.patch.object(asset_io, "read_bytes", return_value=buffer.getvalue()) as read:
            np.testing.assert_array_equal(asset_io.load_npy("data/x.npy"), array)
        read.assert_called_once_with("data/x.npy")


class WriteDestinationTests(unittest.TestCase):
    """Writes go to OSS unconditionally -- unlike reads, which prefer local."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        self.scratch = self.root / "scratch"
        self.scratch.mkdir()
        patcher = mock.patch.object(asset_io, "REPO_ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.dict(os.environ,
                                  {"ATOMICDANCE_SCRATCH": str(self.scratch)})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

        # Intercept at ossutil, reading the staged file while it still exists:
        # that file *is* the payload, and asserting on it is what proves the
        # upload carries finished bytes rather than a handle to a growing one.
        self.calls = []
        def record(argv, attempts=3):
            argv = list(argv)
            if argv[0] == "cp":
                self.calls.append({"url": argv[2],
                                   "payload": pathlib.Path(argv[1]).read_bytes()})
            else:
                self.calls.append({"argv": argv})
        patcher = mock.patch.object(asset_io, "_ossutil", side_effect=record)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_an_existing_local_copy_does_not_capture_the_write(self):
        # The read path prefers local; if the write path *went there instead*, a
        # stage's output would land wherever the previous run's leftovers
        # happened to be, and two runs would disagree about where the artifact
        # lives.  So the upload is unconditional and the destination is always
        # the store -- that half is unchanged and is what the first three
        # assertions pin.
        #
        # Until 2026-08-20 this test also asserted the local copy stayed at its
        # old contents, which is a stronger claim than the rationale above and
        # turned out to be the defect rather than the guarantee: ``runs/`` is a
        # parked tree on this pod, so stage C merged a corrected 17,015-row
        # inventory to the store and ``staging`` -- reading local-first --
        # rebuilt itself from the 17,790-row copy beside it and printed the old
        # counts as a success.  An existing copy now follows the write.
        stale = self.root / "runs" / "seg.json"
        stale.parent.mkdir(parents=True)
        stale.write_text("{}", encoding="utf-8")
        asset_io.write_json("runs/seg.json", {"segments": 3})
        self.assertEqual(len(self.calls), 1)
        self.assertIn("AtomicDance/runs/seg.json", self.calls[0]["url"])
        self.assertEqual(json.loads(self.calls[0]["payload"]), {"segments": 3})
        self.assertEqual(json.loads(stale.read_text(encoding="utf-8")),
                         {"segments": 3})

    def test_no_local_copy_is_created_where_none_existed(self):
        # The other half of §1.2: the checkout holds code, not corpora.  A hook
        # that created copies would put bytes on this pod nobody asked for.
        asset_io.write_json("runs/fresh.json", {"a": 1})
        self.assertEqual(len(self.calls), 1)
        self.assertFalse((self.root / "runs" / "fresh.json").exists())

    def test_a_write_uploads_one_finished_file(self):
        # The 2026-08-12 quota failure produced 22 artifacts that existed and
        # were empty, because write_text creates the file before it writes.
        # Staging the whole payload and then copying it has no such
        # intermediate at the destination: the object is complete or absent.
        asset_io.write_jsonl("runs/x.jsonl", [{"a": 1}, {"b": 2}])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0]["payload"], b'{"a": 1}\n{"b": 2}\n')

    def test_the_staging_file_is_removed_even_when_the_upload_fails(self):
        with mock.patch.object(asset_io, "_ossutil",
                               side_effect=asset_io.AssetIOError("502")):
            with self.assertRaises(asset_io.AssetIOError):
                asset_io.write_json("runs/seg.json", {"a": 1})
        self.assertEqual(list(self.scratch.iterdir()), [])

    def test_arrays_round_trip_through_the_file_they_are_staged_in(self):
        import numpy as np

        array = np.arange(6, dtype=np.float32).reshape(2, 3)
        asset_io.save_npy("runs/a.npy", array)
        np.testing.assert_array_equal(
            np.load(io.BytesIO(self.calls[0]["payload"])), array)


class ListingTests(unittest.TestCase):
    LISTING = """\
2026-08-12 15:54:23 +0000 UTC       369344      Standard   EF67   oss://b/p/a__clip000/motion.npy
2026-08-12 15:54:24 +0000 UTC          969      Standard   AA01   oss://b/p/a__clip000/quality.json
2026-08-12 15:54:25 +0000 UTC            0      Standard   BB02   oss://b/p/b__clip001/

Object Number is: 3

4.960102(s) elapsed
"""

    def test_only_object_lines_are_parsed(self):
        # The trailer lines do not share the object layout, and parsing one of
        # them as an object is how a skip-set gains a member that never existed.
        found = asset_io.parse_ossutil_listing(self.LISTING, "p/")
        self.assertEqual(found, {"a__clip000/motion.npy": 369344,
                                 "a__clip000/quality.json": 969})

    def test_a_size_is_read_by_position_relative_to_UTC_not_column(self):
        found = asset_io.parse_ossutil_listing(self.LISTING, "p/")
        self.assertEqual(found["a__clip000/quality.json"], 969)


class ScratchTests(unittest.TestCase):
    def test_the_scratch_root_refuses_to_sit_on_the_quota_it_protects(self):
        # A scratch directory on /workspace would refill the exact mount this
        # module exists to keep empty, and would do it invisibly -- nobody
        # audits a temp dir.
        with mock.patch.dict(os.environ,
                             {"ATOMICDANCE_SCRATCH": "/workspace/user/tmp"}):
            with self.assertRaises(asset_io.AssetIOError) as caught:
                asset_io.scratch_root()
        self.assertIn("NAS", str(caught.exception))

    def test_a_fetched_clip_is_deleted_even_when_the_consumer_raises(self):
        # The caller most likely to leak is the one that crashed, and a retry
        # loop over a crashing clip is how a bounded scratch becomes unbounded.
        with tempfile.TemporaryDirectory() as scratch:
            with mock.patch.dict(os.environ, {"ATOMICDANCE_SCRATCH": scratch}), \
                 mock.patch.object(asset_io, "local_path", return_value=None), \
                 mock.patch.object(asset_io, "read_bytes", return_value=b"mp4"):
                seen = {}
                with self.assertRaises(ValueError):
                    with asset_io.scratch_file("data/c/clip.mp4") as path:
                        seen["path"] = path
                        self.assertEqual(path.read_bytes(), b"mp4")
                        raise ValueError("decoder died")
            self.assertFalse(seen["path"].exists())
            self.assertEqual(list(pathlib.Path(scratch).iterdir()), [])

    def test_an_asset_already_on_disk_is_used_in_place_and_not_deleted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "data").mkdir()
            local = root / "data" / "clip.mp4"
            local.write_bytes(b"mp4")
            with mock.patch.object(asset_io, "REPO_ROOT", root):
                with asset_io.scratch_file("data/clip.mp4") as path:
                    self.assertEqual(path, local)
            self.assertTrue(local.exists())

    def test_require_free_names_the_mount_before_anything_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp)
            with mock.patch.object(asset_io, "free_bytes", return_value=2**20):
                with self.assertRaises(asset_io.AssetIOError) as caught:
                    asset_io.require_free(path, 2**30)
        self.assertIn(str(path), str(caught.exception))


class PublishTests(unittest.TestCase):
    def test_publishing_a_tool_output_dir_names_what_it_uploaded(self):
        # A stage that reports success over an empty upload is a failure this
        # repo has met more than once, so the caller gets the list to record
        # rather than a boolean to trust.
        with tempfile.TemporaryDirectory() as tmp:
            produced = pathlib.Path(tmp)
            (produced / "sub").mkdir()
            (produced / "a.json").write_bytes(b"{}")
            (produced / "sub" / "b.npy").write_bytes(b"\x93NUMPY")
            (produced / "junk.pyc").write_bytes(b"x")
            with mock.patch.object(asset_io, "_ossutil") as ossutil:
                names = asset_io.publish_dir(produced, "data/out", skip=("*.pyc",))
        self.assertEqual(names, ["data/out/a.json", "data/out/sub/b.npy"])
        # One sync for the directory, not a process per file.
        ossutil.assert_called_once()
        self.assertEqual(ossutil.call_args[0][0][0], "sync")

    def test_an_empty_directory_uploads_nothing_rather_than_syncing_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(asset_io, "_ossutil") as ossutil:
                self.assertEqual(asset_io.publish_dir(pathlib.Path(tmp), "data/out"), [])
        ossutil.assert_not_called()


if __name__ == "__main__":
    unittest.main()


class OssutilDeadlineTests(unittest.TestCase):
    """A transfer that stops answering has to become an error, not a wait.

    Every caller sits in a loop whose own timeout is hours, so an untimed
    ``subprocess.run`` turns one wedged object into an afternoon of a silent
    card.  The retry is what makes a *slow* transfer survivable; the raise is
    what makes a *hung* one visible.
    """

    def test_a_hung_transfer_is_retried_and_then_raised(self):
        calls = []

        def never_answers(argv, **kwargs):
            calls.append(kwargs.get("timeout"))
            raise asset_io.subprocess.TimeoutExpired(argv, kwargs.get("timeout"))

        with mock.patch.object(asset_io.subprocess, "run", side_effect=never_answers):
            with mock.patch.object(asset_io.time, "sleep"):
                with self.assertRaises(asset_io.AssetIOError) as caught:
                    asset_io._ossutil(["cp", "oss://x", "/tmp/y"], attempts=3, timeout=5)
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls, [5, 5, 5])
        self.assertIn("no answer in 5s", str(caught.exception))

    def test_a_slow_transfer_that_finishes_is_not_an_error(self):
        answers = [asset_io.subprocess.TimeoutExpired(["ossutil"], 5),
                   mock.Mock(returncode=0, stdout="", stderr="")]

        def flaky(argv, **kwargs):
            result = answers.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        with mock.patch.object(asset_io.subprocess, "run", side_effect=flaky):
            with mock.patch.object(asset_io.time, "sleep"):
                asset_io._ossutil(["cp", "oss://x", "/tmp/y"], attempts=3, timeout=5)
        self.assertEqual(answers, [])


class OssutilIdentity(unittest.TestCase):
    """Every ossutil subprocess names the config it authenticates with.

    Left to itself ossutil reads ``~/.ossutilconfig``.  In this container that
    file holds a third identity -- a stale STS key -- which the bridge bucket
    refuses with 403 InvalidAccessKeyId, while both credentials this module can
    resolve are accepted.  So an unflagged ``ossutil ls`` failed as an identity
    no caller chose, which is what broke ``audit_clip_freshness --records
    store`` on 2026-08-25: the listing died at the first prefix and the audit
    could not run against the object store at all.

    The property is not "it works here" -- it is that the flag is on the
    command line, which is the only part a later reader can check.
    """

    def _argv_of(self, call):
        with mock.patch.object(asset_io, "ossutil_flags",
                               return_value=["-c", "/chosen/config"]), \
             mock.patch("subprocess.run") as runner:
            runner.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")
            call()
            self.assertTrue(runner.call_args_list, "nothing was run")
            return runner.call_args_list[0].args[0]

    def test_list_prefix_names_its_config(self):
        argv = self._argv_of(lambda: asset_io.list_prefix("data/x"))
        self.assertIn("-c", argv)
        self.assertEqual(argv[argv.index("-c") + 1], "/chosen/config")

    def test_exists_names_its_config_on_the_ossutil_fallback(self):
        with mock.patch.object(asset_io, "local_path", return_value=None), \
             mock.patch.object(asset_io, "filesystem",
                               side_effect=RuntimeError("bridge bucket 502")):
            argv = self._argv_of(lambda: asset_io.exists("data/x"))
        self.assertIn("-c", argv)

    def test_the_flag_is_the_config_oss_assets_resolves_not_a_second_answer(self):
        # One answer to "which key signed this transfer".  The two modules
        # resolving it apart is unrecoverable from the artifact afterwards.
        from tools import oss_assets
        with mock.patch.object(oss_assets, "ossutil_config",
                               return_value=pathlib.Path("/tmp/does-not-exist")):
            self.assertEqual(asset_io.ossutil_flags(), [])
