from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import uuid


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facade.external import (  # noqa: E402
    RUNTIME_SCHEMA,
    RUNTIME_SEAL_SCHEMA,
    RUN_CONFIG_SCHEMA,
    validate_external_contract,
)
from facade.sandbox import (  # noqa: E402
    DEFAULT_SANDBOX_POLICY,
    RUNTIME_THREAT_MODEL_SHA256,
    sandbox_policy_sha256,
)
from facade.planner import profile_selection  # noqa: E402
from facade.registry import FacadeError, load_bundle, load_json  # noqa: E402
from facade.runner import run_external  # noqa: E402
from facade.selection import canonical_bytes, sha256_bytes, sha256_file  # noqa: E402
from facade.snapshots import publish_snapshot_overlay, validate_snapshot_publication  # noqa: E402


CONTROL_FILES = (
    "variants/registry.json",
    "variants/profiles.json",
    "variants/compatibility.json",
    "configs/adapters.json",
    "configs/snapshot-plan.json",
    "lineage/code.jsonl",
    "lineage/config.jsonl",
)
PROFILE_ID = "exp5_legacy_multiturn_eval5"
PUBLICATION_ID = "synthetic-publication"
RUNTIME_ID = "synthetic-runtime"
RUN_CONFIG_ID = "synthetic-run"
RUNTIME_PYTHON = Path("/usr/bin/python3.10")


SCRIPT = b"""\
import argparse
import json
import os
from pathlib import Path
import socket

parser = argparse.ArgumentParser()
parser.add_argument('--out_csv', required=True)
parser.add_argument('--json_path', required=True)
args = parser.parse_args()
payload = json.loads(Path(args.json_path).read_text(encoding='utf-8'))
denials = []
for operation in (
    lambda: open(args.json_path, 'w', encoding='utf-8'),
    lambda: os.chmod(args.json_path, 0o600),
    lambda: os.rename(args.json_path, args.json_path + '.moved'),
    lambda: os.link(args.json_path, args.json_path + '.linked'),
    lambda: os.utime(args.json_path, None),
    lambda: os.setxattr(args.json_path, 'user.experiments7', b'blocked'),
    lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM),
    lambda: os.fork(),
):
    try:
        operation()
    except PermissionError:
        denials.append(True)
if len(denials) != 8:
    raise SystemExit(73)
Path(args.out_csv).write_text('value\\n' + str(payload['value']) + '\\n', encoding='utf-8')
"""


def write_readonly(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(0o444)


class ExternalRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="experiments7-runtime-")
        self.root = Path(self.temporary.name)
        for relative in CONTROL_FILES:
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)
        (self.root / "runs").mkdir()
        self.bundle = load_bundle(self.root)
        plan = load_json(self.root / "configs/snapshot-plan.json")
        entrypoint = self.bundle.adapters_by_variant["eval5_legacy_mt"]["entrypoint_destination"]
        for row in plan["records"]:
            payload = SCRIPT if row["planned_destination"] == entrypoint else (
                f"synthetic snapshot for {row['lineage_id']}\n".encode()
            )
            write_readonly(self.root / row["planned_destination"], payload)
        self.publication = publish_snapshot_overlay(self.bundle, PUBLICATION_ID)
        self._write_runtime_closure()
        self.input_path = self.root / "inputs/example.json"
        write_readonly(self.input_path, b'{"value":7}\n')
        self._write_run_config(RUN_CONFIG_ID)

    def tearDown(self) -> None:
        for directory in (
            self.root / "snapshots/publications" / PUBLICATION_ID,
            self.root / "runtime/closures" / RUNTIME_ID,
        ):
            if directory.exists():
                directory.chmod(0o755)
        for current, directories, files in os.walk(self.root, topdown=False):
            for name in files:
                try:
                    (Path(current) / name).chmod(0o600)
                except FileNotFoundError:
                    pass
            for name in directories:
                try:
                    (Path(current) / name).chmod(0o700)
                except FileNotFoundError:
                    pass
        self.temporary.cleanup()

    def _write_runtime_closure(self) -> None:
        python = RUNTIME_PYTHON.resolve(strict=True)
        digest, size = sha256_file(python)
        artifacts = [{
            "artifact_id": "python",
            "path": str(python),
            "sha256": digest,
            "bytes": size,
        }]
        manifest = {
            "schema": RUNTIME_SCHEMA,
            "runtime_id": RUNTIME_ID,
            "python_artifact_id": "python",
            "artifact_count": len(artifacts),
            "artifacts_sha256": sha256_bytes(canonical_bytes(artifacts)),
            "artifacts": artifacts,
            "environment_allowlist": [],
            "environment_defaults": {"PYTHONHASHSEED": "0"},
            "timeout_seconds": 30,
            "declared_runtime_dependencies": ["python"],
            "sandbox_policy": DEFAULT_SANDBOX_POLICY,
            "sandbox_policy_sha256": sandbox_policy_sha256(),
            "runtime_threat_model_sha256": RUNTIME_THREAT_MODEL_SHA256,
        }
        payload = canonical_bytes(manifest)
        self.runtime_manifest_sha256 = sha256_bytes(payload)
        seal = {
            "schema": RUNTIME_SEAL_SCHEMA,
            "runtime_id": RUNTIME_ID,
            "manifest_sha256": sha256_bytes(payload),
            "manifest_bytes": len(payload),
        }
        directory = self.root / "runtime/closures" / RUNTIME_ID
        directory.mkdir(parents=True)
        write_readonly(directory / "manifest.json", payload)
        write_readonly(directory / "seal.json", canonical_bytes(seal))
        directory.chmod(0o555)

    def _write_run_config(
        self,
        run_config_id: str,
        *,
        input_sha256: str | None = None,
        output_dir: str = "runs/success",
        runtime_manifest_sha256: str | None = None,
    ) -> None:
        digest, size = sha256_file(self.input_path)
        python = str(RUNTIME_PYTHON.resolve(strict=True))
        entrypoint = self.bundle.adapters_by_variant["eval5_legacy_mt"]["entrypoint_destination"]
        argv = [
            python,
            str(self.root / entrypoint),
            "--out_csv",
            str(self.root / output_dir / "metrics.csv"),
            "--json_path",
            str(self.input_path),
        ]
        expected_output = b"value\n7\n"
        selection = profile_selection(
            self.bundle,
            PROFILE_ID,
            execution="external_contract_validation",
            run_id=None,
            output_root=self.root / output_dir,
        )
        config = {
            "schema": RUN_CONFIG_SCHEMA,
            "run_config_id": run_config_id,
            "profile_id": PROFILE_ID,
            "publication_id": PUBLICATION_ID,
            "runtime_id": RUNTIME_ID,
            "environment_sources": {},
            "control_hashes": selection["control_hashes"],
            "publication_manifest_sha256": self.publication.manifest_sha256,
            "runtime_manifest_sha256": (
                runtime_manifest_sha256 or self.runtime_manifest_sha256
            ),
            "sandbox_policy_sha256": sandbox_policy_sha256(),
            "runtime_threat_model_sha256": RUNTIME_THREAT_MODEL_SHA256,
            "output_dir": output_dir,
            "dispatch_path": "production_external_contract",
            "transport": {
                "mode": "not_applicable",
                "boundary": "no_external_exchange",
            },
            "steps": [{
                "stage": "evaluation",
                "variant_id": "eval5_legacy_mt",
                "template": "singleturn",
                "argv_sha256": sha256_bytes(canonical_bytes(argv)),
                "expected_outputs": [{
                    "path": "metrics.csv",
                    "sha256": sha256_bytes(expected_output),
                    "bytes": len(expected_output),
                }],
                "bindings": {
                    "json_path": {
                        "kind": "input",
                        "path": "inputs/example.json",
                        "sha256": input_sha256 or digest,
                        "bytes": size,
                    },
                    "out_csv": {"kind": "output", "path": "metrics.csv"},
                },
            }],
        }
        write_readonly(
            self.root / "configs/external-runs" / f"{run_config_id}.json",
            canonical_bytes(config),
        )

    def test_publication_is_exhaustive_and_pending_controls_stay_pending(self) -> None:
        verified = validate_snapshot_publication(self.bundle, PUBLICATION_ID)
        self.assertEqual(len(verified.records_by_lineage), 129)
        plan = load_json(self.root / "configs/snapshot-plan.json")
        self.assertTrue(all(row["final_sha256"] is None for row in plan["records"]))
        self.assertTrue(all(
            row["snapshot_state"] == "pending_protected_snapshot" for row in plan["records"]
        ))

    def test_snapshot_mode_or_hash_drift_blocks_before_execution(self) -> None:
        record = next(iter(self.publication.records_by_lineage.values()))
        path = self.root / record["path"]
        path.chmod(0o644)
        try:
            with self.assertRaises(FacadeError) as caught:
                validate_snapshot_publication(self.bundle, PUBLICATION_ID)
            self.assertEqual(caught.exception.code, "SNAPSHOT_NOT_IMMUTABLE")
        finally:
            path.chmod(0o444)

    def test_run_config_runtime_hash_drift_blocks_before_entrypoint(self) -> None:
        bad_id = f"bad-{uuid.uuid4().hex}"
        self._write_run_config(
            bad_id,
            output_dir="runs/not-created",
            runtime_manifest_sha256="0" * 64,
        )
        selection = profile_selection(
            self.bundle,
            PROFILE_ID,
            execution="external_contract_validation",
            run_id=None,
            output_root=self.root / "runs/not-created",
        )
        with self.assertRaises(FacadeError) as caught:
            validate_external_contract(
                self.bundle,
                selection,
                self.root / "runs/not-created",
                publication_id=PUBLICATION_ID,
                runtime_id=RUNTIME_ID,
                run_config_id=bad_id,
                require_environment=False,
            )
        self.assertEqual(caught.exception.code, "RUN_CONFIG_RUNTIME_HASH_MISMATCH")

    def test_external_dispatch_blocks_unavailable_runtime_proof_before_subprocess(self) -> None:
        with self.assertRaises(FacadeError) as caught:
            run_external(
                self.bundle,
                PROFILE_ID,
                "runs/success",
                allow_external=True,
                publication_id=PUBLICATION_ID,
                runtime_id=RUNTIME_ID,
                run_config_id=RUN_CONFIG_ID,
            )
        self.assertEqual(caught.exception.code, "RUNTIME_PROOF_ENVIRONMENT_UNAVAILABLE")
        self.assertEqual(caught.exception.detail["environment_capability"], "mechanism-only")
        self.assertIn(
            "NO_PRIVILEGE_SEPARATED_SUPERVISOR",
            caught.exception.detail["reason_codes"],
        )
        self.assertTrue((self.root / "runs/success/selection.json").is_file())
        self.assertFalse((self.root / "runs/success/external-result.json").exists())
        self.assertTrue((self.root / "runs/success/blocked.json").exists())
        self.assertEqual((self.root / "runs/success").stat().st_mode & 0o222, 0)

    def test_partial_contract_ids_fail_closed_without_subprocess(self) -> None:
        with self.assertRaises(FacadeError) as caught:
            run_external(
                self.bundle,
                PROFILE_ID,
                "runs/partial",
                allow_external=True,
                publication_id=PUBLICATION_ID,
            )
        self.assertEqual(caught.exception.code, "EXTERNAL_CONTRACT_INCOMPLETE")
        self.assertTrue((self.root / "runs/partial/blocked.json").is_file())
        self.assertFalse((self.root / "runs/partial/external-result.json").exists())


if __name__ == "__main__":
    unittest.main()
