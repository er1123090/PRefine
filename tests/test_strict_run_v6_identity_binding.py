from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from strict_run.canonical import StrictRunError, sha256_bytes  # noqa: E402
from strict_run.checkpoint import (  # noqa: E402
    CP0_TRUSTED_PROVIDER_SHA256,
    _descriptor_bound_absolute_sha256,
    _validate_cp0_runtime_executable,
)
from strict_run.g0_bootstrap import (  # noqa: E402
    _directory_mount_evidence,
    _trusted_provider_bytes,
)


class StrictRunV6IdentityBindingTests(unittest.TestCase):
    @staticmethod
    def repository_mount_id() -> int:
        return _directory_mount_evidence(str(ROOT), "repository")[1]

    def test_runtime_accepts_only_descriptor_bound_controller_interpreter(
        self,
    ) -> None:
        path = os.path.realpath(sys.executable)
        digest, size = _descriptor_bound_absolute_sha256(
            path, "test runtime"
        )
        self.assertGreater(size, 0)
        self.assertEqual(
            _validate_cp0_runtime_executable(path, digest, digest),
            (digest, size),
        )

    def test_forged_runtime_path_is_rejected_before_read(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="exp7-runtime-forgery-"
        ) as base:
            forged = Path(base) / "python"
            forged.write_bytes(Path(os.path.realpath(sys.executable)).read_bytes())
            digest = sha256_bytes(forged.read_bytes())
            with self.assertRaises(StrictRunError) as raised:
                _validate_cp0_runtime_executable(str(forged), digest, digest)
        self.assertEqual(raised.exception.code, "CP0_CHILD_EVIDENCE_INVALID")

    def test_forged_runtime_hash_is_rejected(self) -> None:
        path = os.path.realpath(sys.executable)
        actual, _ = _descriptor_bound_absolute_sha256(path, "test runtime")
        with self.assertRaises(StrictRunError) as raised:
            _validate_cp0_runtime_executable(path, "0" * 64, actual)
        self.assertEqual(raised.exception.code, "CP0_CHILD_EVIDENCE_INVALID")
        with self.assertRaises(StrictRunError):
            _validate_cp0_runtime_executable(path, actual, "f" * 64)

    def test_arbitrary_provider_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="exp7-provider-forgery-"
        ) as base:
            forged = Path(base) / "provider.py"
            forged.write_bytes(
                b"def apply_readonly_envelope(_): return {}\n"
            )
            with self.assertRaises(StrictRunError) as raised:
                _trusted_provider_bytes(
                    str(forged),
                    expected_mount_id=self.repository_mount_id(),
                )
        self.assertEqual(raised.exception.code, "G0_TRUSTED_PROVIDER_INVALID")

    def test_repository_provider_matches_pin(self) -> None:
        provider = ROOT / "scripts" / "g0" / "provider.py"
        payload = _trusted_provider_bytes(
            str(provider), expected_mount_id=self.repository_mount_id()
        )
        self.assertEqual(
            sha256_bytes(payload), CP0_TRUSTED_PROVIDER_SHA256
        )


if __name__ == "__main__":
    unittest.main()
