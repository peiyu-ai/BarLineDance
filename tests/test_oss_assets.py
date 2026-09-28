import argparse
import os
import pathlib
import tempfile
import unittest
from unittest import mock

from tools import oss_assets


class PrefixTests(unittest.TestCase):
    def test_splits_bucket_from_key(self):
        bucket, key = oss_assets.parse_prefix(
            "oss://example-bucket/example/prefix/")
        self.assertEqual(bucket, "example-bucket")
        self.assertEqual(key, "example/prefix/")

    def test_adds_the_trailing_slash_a_prefix_needs(self):
        _, key = oss_assets.parse_prefix("oss://bucket/some/prefix")
        self.assertEqual(key, "some/prefix/")

    def test_refuses_a_non_oss_url(self):
        with self.assertRaises(oss_assets.AssetError):
            oss_assets.parse_prefix("s3://bucket/prefix/")


class KeyMappingTests(unittest.TestCase):
    ROOT = "example/prefix/"

    def test_repo_paths_land_under_their_own_component(self):
        # The prefix already holds data/tiktok/ and models/ from other work.
        # Without the AtomicDance/ component this repo's data/ would interleave
        # with a tree nobody could later separate it from.
        self.assertEqual(
            oss_assets.remote_key("data/wild3d/wild_performance_v1", self.ROOT),
            self.ROOT + "AtomicDance/data/wild3d/wild_performance_v1")

    def test_upstream_models_keep_the_key_the_store_already_uses(self):
        # Mirroring these would spend 89 GB making a second copy that can drift
        # from the one setup_qwenvl_env.sh pulls.
        self.assertEqual(
            oss_assets.remote_key("third_party/QwenVL/Qwen2.5-VL-7B-Instruct", self.ROOT),
            self.ROOT + "models/Qwen2.5-VL-7B-Instruct")
        self.assertEqual(
            oss_assets.remote_key(
                "third_party/QwenVL/Qwen2.5-VL-7B-Instruct/main/config.json", self.ROOT),
            self.ROOT + "models/Qwen2.5-VL-7B-Instruct/main/config.json")

    def test_a_sibling_of_an_upstream_model_is_still_mirrored(self):
        # Prefix matching must respect path components: QwenVL itself is not
        # upstream just because two of its children are.
        self.assertEqual(
            oss_assets.remote_key("third_party/QwenVL/README.md", self.ROOT),
            self.ROOT + "AtomicDance/third_party/QwenVL/README.md")
        self.assertIsNone(oss_assets.upstream_for("third_party/QwenVL"))

    def test_every_upstream_entry_names_the_script_that_restores_it(self):
        for root, (key, restore) in oss_assets.UPSTREAM.items():
            self.assertTrue(key, root)
            self.assertIn("setup", restore)


class ExclusionTests(unittest.TestCase):
    def test_matches_on_path_components_not_substrings(self):
        self.assertTrue(oss_assets.excluded(pathlib.PurePath("a/__pycache__/b.pyc")))
        self.assertTrue(oss_assets.excluded(pathlib.PurePath("third_party/GVHMR/.git/HEAD")))
        self.assertTrue(oss_assets.excluded(pathlib.PurePath("third_party/QwenVL/.venv-qwen3vl/bin/python")))
        # A directory merely *named* like one is not one.
        self.assertFalse(oss_assets.excluded(pathlib.PurePath("data/venv_notes/readme.md")))
        self.assertFalse(oss_assets.excluded(pathlib.PurePath("data/wild3d/a.npy")))

    def test_ossutil_patterns_carry_no_directory_information(self):
        # ossutil rejects any --exclude containing a path separator, so the
        # directory rules cannot be delegated to it; push_targets does that.
        for pattern in oss_assets.OSSUTIL_EXCLUDE:
            self.assertNotIn("/", pattern)


class TreeWalkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        patcher = mock.patch.object(oss_assets, "REPO_ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def write(self, relative, content=b"x"):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def test_walk_skips_excluded_files_and_symlinks(self):
        self.write("data/real.npy", b"1234")
        self.write("data/__pycache__/cached.pyc", b"junk")
        (self.root / "data" / "linked.mp4").symlink_to("/elsewhere/linked.mp4")
        found = oss_assets.walk_local(self.root / "data")
        self.assertEqual(found, {"data/real.npy": 4})

    def test_a_clean_tree_is_pushed_whole(self):
        self.write("data/wild3d/a.npy")
        self.write("data/wild3d/nested/b.npy")
        target = self.root / "data"
        base = pathlib.PurePath("data")
        self.assertEqual(oss_assets.excluded_dirs(target, base), [])
        self.assertEqual(oss_assets.push_targets(target, [], base), [target])

    def test_a_tree_holding_a_git_dir_is_split_around_it(self):
        self.write("third_party/WHAM/code.py")
        self.write("third_party/WHAM/.git/HEAD")
        self.write("third_party/WHAM/models/w.pt")
        target = self.root / "third_party" / "WHAM"
        base = pathlib.PurePath("third_party/WHAM")
        forbidden = oss_assets.excluded_dirs(target, base)
        self.assertEqual(forbidden, [target / ".git"])
        pieces = oss_assets.push_targets(target, forbidden, base)
        self.assertEqual(sorted(pieces), sorted([target / "code.py", target / "models"]))
        self.assertNotIn(target / ".git", pieces)

    def test_excluded_dirs_does_not_descend_into_what_it_finds(self):
        # A .git holding another .git must be reported once, not walked.
        self.write("third_party/tram/.git/modules/x/.git/HEAD")
        found = oss_assets.excluded_dirs(self.root / "third_party" / "tram",
                                         pathlib.PurePath("third_party/tram"))
        self.assertEqual(found, [self.root / "third_party/tram/.git"])

    def test_split_still_covers_every_file_the_census_expects(self):
        # The invariant that makes verify meaningful: the union of the pushed
        # pieces is exactly the set of files walk_local will later demand.
        self.write("third_party/WHAM/code.py")
        self.write("third_party/WHAM/.git/HEAD")
        self.write("third_party/WHAM/third-party/DPVO/.git/config")
        self.write("third_party/WHAM/third-party/DPVO/vo.py")
        self.write("third_party/WHAM/__pycache__/code.pyc")
        target = self.root / "third_party" / "WHAM"
        base = pathlib.PurePath("third_party/WHAM")
        pieces = oss_assets.push_targets(target, oss_assets.excluded_dirs(target, base),
                                         base)
        covered = {}
        for piece in pieces:
            covered.update(oss_assets.walk_local(piece))
        self.assertEqual(set(covered), set(oss_assets.walk_local(target)))
        self.assertEqual(set(covered),
                         {"third_party/WHAM/code.py",
                          "third_party/WHAM/third-party/DPVO/vo.py"})

    def test_a_parked_tree_can_be_walked_by_the_name_it_is_not_stored_under(self):
        # runs/ was parked in the cache on 2026-08-12 and every push of it
        # failed: the code resolved the symlink and then asked the cache path
        # where it sat inside the repo, which raises.  The consequence was
        # silent in the worst way -- 29 GB of experiment output existed only on
        # the cache mount while `status` cheerfully reported the tree.
        cache = self.root / "_cache" / "runs"
        (cache / "sub").mkdir(parents=True)
        (cache / "report.json").write_bytes(b"{}")
        (cache / "sub" / "vocab.npy").write_bytes(b"1234")
        (cache / "__pycache__").mkdir()
        (cache / "__pycache__" / "x.pyc").write_bytes(b"junk")
        (self.root / "runs").symlink_to(cache)

        base = pathlib.PurePath("runs")
        target = (self.root / "runs").resolve()
        forbidden = oss_assets.excluded_dirs(target, base)
        self.assertEqual(forbidden, [cache / "__pycache__"])
        pieces = oss_assets.push_targets(target, forbidden, base)
        covered = {}
        for piece in pieces:
            name = base if piece == target else base / piece.relative_to(target)
            covered.update(oss_assets.walk_local(piece, str(name)))
        # Keys are the repo-relative names, not the cache location they live at.
        self.assertEqual(set(covered), {"runs/report.json", "runs/sub/vocab.npy"})


class CacheTests(unittest.TestCase):
    """A parked tree lives off the NAS but keeps its repo-relative identity."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name) / "repo"
        self.cache = pathlib.Path(self.tmp.name) / "cache"
        self.root.mkdir()
        patcher = mock.patch.object(oss_assets, "REPO_ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def test_cache_root_defaults_off_the_nas(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(str(oss_assets.cache_root()).startswith("/workspace"))
        with mock.patch.dict(os.environ, {"ATOMICDANCE_ASSET_CACHE": "/somewhere"}):
            self.assertEqual(oss_assets.cache_root(), pathlib.Path("/somewhere"))

    def test_a_cached_tree_still_reports_repo_relative_names(self):
        # verify compares these keys against OSS keys built from the repo path,
        # so a cached tree that reported /cache/... names would look 100%
        # missing and evict would refuse to free anything.
        parked = self.cache / "data" / "wild3d"
        (parked / "sub").mkdir(parents=True)
        (parked / "sub" / "motion.npy").write_bytes(b"1234")
        found = oss_assets.walk_local(parked, "data/wild3d")
        self.assertEqual(found, {"data/wild3d/sub/motion.npy": 4})

    def test_pull_to_cache_refuses_to_shadow_a_real_directory(self):
        # Symlinking over a populated directory would strand its bytes on the
        # NAS, invisible to every later census.
        real = self.root / "data" / "wild3d"
        real.mkdir(parents=True)
        (real / "keep.npy").write_bytes(b"x")
        args = argparse.Namespace(
            paths=["data/wild3d"], prefix=oss_assets.DEFAULT_PREFIX,
            jobs=1, parallel=1, cache=True)
        with mock.patch.object(oss_assets, "run_ossutil", return_value=0) as ossutil:
            with self.assertRaises(oss_assets.AssetError):
                oss_assets.cmd_pull(args)
        ossutil.assert_not_called()

    def test_pull_to_cache_links_the_repo_path_at_the_cached_tree(self):
        args = argparse.Namespace(
            paths=["data/wild3d"], prefix=oss_assets.DEFAULT_PREFIX,
            jobs=1, parallel=1, cache=True)
        with mock.patch.dict(os.environ, {"ATOMICDANCE_ASSET_CACHE": str(self.cache)}):
            with mock.patch.object(oss_assets, "run_ossutil", return_value=0):
                self.assertEqual(oss_assets.cmd_pull(args), 0)
        link = self.root / "data" / "wild3d"
        self.assertTrue(link.is_symlink())
        self.assertEqual(link.resolve(), (self.cache / "data" / "wild3d").resolve())


class EvictTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        patcher = mock.patch.object(oss_assets, "REPO_ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def _run(self, report):
        args = argparse.Namespace(paths=list(report), prefix=oss_assets.DEFAULT_PREFIX,
                                  yes=True)
        with mock.patch.object(oss_assets, "census", return_value=report):
            return oss_assets.cmd_evict(args)

    @staticmethod
    def _clear(bytes_=10, files=1):
        return {"upstream": False, "linked": False, "materialised_at": "",
                "local_files": files, "local_bytes": bytes_,
                "remote_objects": files, "remote_bytes": bytes_,
                "missing": [], "mismatched": []}

    def test_a_single_file_asset_is_unlinked_not_rmtree_d(self):
        # data/atomic_aistpp.zip is a real target; rmtree raises on a file.
        target = self.root / "data" / "atomic_aistpp.zip"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"zip")
        self.assertEqual(self._run({"data/atomic_aistpp.zip": self._clear()}), 0)
        self.assertFalse(target.exists())

    def test_one_undeletable_tree_does_not_strand_the_rest_of_the_queue(self):
        good = self.root / "data" / "good"
        good.mkdir(parents=True)
        (good / "f").write_bytes(b"x")
        report = {"data/absent": self._clear(), "data/good": self._clear()}
        self.assertEqual(self._run(report), 1)      # reported as a failure...
        self.assertFalse(good.exists())             # ...but the queue finished

    def test_a_tree_still_missing_from_oss_is_refused(self):
        target = self.root / "data" / "hot"
        target.mkdir(parents=True)
        (target / "f").write_bytes(b"x")
        entry = self._clear()
        entry["missing"] = ["data/hot/f"]
        self.assertEqual(self._run({"data/hot": entry}), 1)
        self.assertTrue(target.exists())


class CredentialTests(unittest.TestCase):
    def test_credentials_are_never_read_from_the_repo(self):
        import inspect

        source = inspect.getsource(oss_assets)
        self.assertNotIn("LTAI", source)          # no access key id
        self.assertIn("OSS_ACCESS_KEY_ID", source)
        self.assertIn(".ossutilconfig", source)

    def test_missing_credentials_fail_loudly_rather_than_anonymously(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {k: v for k, v in os.environ.items()
                   if k not in ("OSS_ACCESS_KEY_ID", "OSS_ACCESS_KEY_SECRET")}
            env["OSSUTIL_CONFIG"] = str(pathlib.Path(tmp) / "absent")
            with mock.patch.dict(os.environ, env, clear=True):
                with self.assertRaises(oss_assets.AssetError):
                    oss_assets.make_bucket("oss://bucket/prefix/")


if __name__ == "__main__":
    unittest.main()
