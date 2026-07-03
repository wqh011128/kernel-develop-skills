from __future__ import annotations

import io
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import pallas_xprof_batch  # noqa: E402
import pallas_xprof_registry_runner  # noqa: E402
import xprof_pallas_tools  # noqa: E402


class XprofHelpersTest(unittest.TestCase):
    def test_default_libtpu_flags_are_current_and_consistent(self) -> None:
        expected = "--xla_xprof_register_llo_debug_info=true"
        self.assertEqual(pallas_xprof_batch.LIBTPU_FLAGS, expected)
        self.assertEqual(
            pallas_xprof_registry_runner.REQUIRED_LIBTPU_FLAGS, (expected,)
        )

    def test_failure_classification(self) -> None:
        result = pallas_xprof_batch._classify_failure("No TPU backend")
        self.assertEqual(result["failure_class"], "environment_no_tpu_backend")

    def test_safe_extract_rejects_parent_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "bad.tgz"
            with tarfile.open(archive, "w:gz") as handle:
                info = tarfile.TarInfo("../escape.txt")
                data = b"escape"
                info.size = len(data)
                handle.addfile(info, io.BytesIO(data))
            with self.assertRaisesRegex(RuntimeError, "unsafe tar"):
                pallas_xprof_batch._safe_extract(archive, root / "out")

    def test_safe_extract_rejects_links(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "link.tgz"
            with tarfile.open(archive, "w:gz") as handle:
                info = tarfile.TarInfo("link")
                info.type = tarfile.SYMTYPE
                info.linkname = "../escape.txt"
                handle.addfile(info)
            with self.assertRaisesRegex(RuntimeError, "unsafe tar link"):
                pallas_xprof_batch._safe_extract(archive, root / "out")

    def test_profile_run_names_are_relative(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "plugins" / "profile" / "run-a"
            run.mkdir(parents=True)
            (run / "trace.xplane.pb").write_bytes(b"")
            self.assertEqual(xprof_pallas_tools._profile_run_names(root), ["plugins/profile/run-a"])


if __name__ == "__main__":
    unittest.main()
