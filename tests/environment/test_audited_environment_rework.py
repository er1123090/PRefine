from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from experiment_env import audited


AUDIT = ROOT / "configs" / "environment" / "audit-overlay-782df2d84077f93b69aed4c79b334643f7ef5cb33470fe5f38fc412434f90b6b.json"
SOURCE_PRE = ROOT / "paper_outputs" / "strict-runs" / "exp7-strict-v6-20260718T091006Z-e9ff93d0c9fe0d6f15fb03c1214dd6e2" / "manifests" / "source-pre.jsonl"
V1 = ROOT / "environments" / "exp45-audited-overlay-v1-782df2d84077f93b"
V2 = ROOT / "environments" / "exp45-audited-overlay-v2-782df2d84077f93b"
V2_SPEC = ROOT / "configs" / "environment" / "exp45-audited-overlay-v2.snapshot.json"
V1_SEAL_SHA256 = "21f8eeca341d9aa312e88619cd2d634a571bef0cd8c4b8e26cd961ec5b430edf"


def sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def rewrite_json(path: Path, value: object) -> None:
    os.chmod(path.parent, 0o755)
    os.chmod(path, 0o644)
    path.write_bytes(audited.canonical_json_bytes(value) + b"\n")
    os.chmod(path, 0o444)
    os.chmod(path.parent, 0o555)


def thaw_tree(root: Path) -> None:
    for current, directories, files in os.walk(root, topdown=False, followlinks=False):
        base = Path(current)
        for name in files:
            path = base / name
            if not path.is_symlink():
                os.chmod(path, 0o600)
        for name in directories:
            path = base / name
            if not path.is_symlink():
                os.chmod(path, 0o700)
        os.chmod(base, 0o700)


@contextmanager
def cloned_v2():
    isolated = Path(tempfile.mkdtemp(prefix=".audited-v2-hostile-", dir="/tmp"))
    try:
        spec = json.loads(V2_SPEC.read_bytes())
        required_paths = (
            V2_SPEC.relative_to(ROOT).as_posix(),
            spec["audit_path"],
            spec["cp0_path"],
            spec["profiles_path"],
            spec["registry_path"],
            spec["source_pre_path"],
        )
        for relative in required_paths:
            source = ROOT / relative
            destination = isolated / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        clone = isolated / spec["destination"]
        shutil.copytree(V2, clone, symlinks=True)
        yield isolated, clone
    finally:
        thaw_tree(isolated)
        shutil.rmtree(isolated)


def thaw_directories(root: Path) -> None:
    for current, directories, _ in os.walk(root, topdown=False, followlinks=False):
        base = Path(current)
        for name in directories:
            path = base / name
            if not path.is_symlink():
                os.chmod(path, 0o700)
        os.chmod(base, 0o700)


@contextmanager
def isolated_tampered_base_binding():
    isolated = Path(tempfile.mkdtemp(prefix=".audited-base-binding-", dir=ROOT))
    try:
        spec = json.loads(V2_SPEC.read_bytes())
        required_paths = (
            spec["audit_path"],
            spec["cp0_path"],
            spec["profiles_path"],
            spec["registry_path"],
            spec["source_pre_path"],
        )
        for relative in required_paths:
            source = ROOT / relative
            destination = isolated / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

        base = isolated / spec["base_snapshot"]
        base.parent.mkdir(parents=True, exist_ok=True)
        # A hardlink fixture lets copytree's copystat mutate the sealed source's
        # ctime. Use real copies so this adversarial test stays fully isolated.
        shutil.copytree(
            ROOT / spec["base_snapshot"],
            base,
            copy_function=shutil.copy2,
            symlinks=True,
        )
        overlay = isolated / spec["destination"]
        shutil.copytree(V2, overlay, symlinks=True)

        isolated_spec = isolated / V2_SPEC.relative_to(ROOT)
        shutil.copy2(V2_SPEC, isolated_spec)
        spec["expected_base_seal_sha256"] = "0" * 64
        rewrite_json(isolated_spec, spec)
        seal_path = overlay / ".experiment-env-overlay" / "seal.json"
        seal = json.loads(seal_path.read_bytes())
        seal["spec_sha256"] = sha256(isolated_spec)
        rewrite_json(seal_path, seal)
        yield isolated, overlay
    finally:
        thaw_directories(isolated)
        shutil.rmtree(isolated)


class AuditedEnvironmentReworkTests(unittest.TestCase):
    def generated_registry(self) -> dict:
        audit = json.loads(AUDIT.read_bytes())
        registry, _ = audited.build_registry_documents(ROOT, audit, SOURCE_PRE)
        return registry

    def test_eval4_expansions_are_turn_specific_and_never_dangle(self) -> None:
        registry = self.generated_registry()
        variants = {value["variant_id"]: value for value in registry["variants"]}
        expected_single = {
            "aggregation": "aggregation4_single_legacy",
            "denominator": "denominator4_single_legacy",
            "error_policy": "error4_single_legacy",
            "metric": "metric4_single_legacy",
            "normalization": "normalization4_single_raw",
            "parser": "parser4_inline",
            "reporting": "report4_single_legacy",
        }
        expected_multi = {
            "aggregation": "aggregation4_multiturn_legacy",
            "denominator": "denominator4_multiturn_legacy",
            "error_policy": "error4_multiturn_legacy",
            "metric": "metric4_multiturn_legacy",
            "normalization": "normalization4_multiturn_dateparser",
            "parser": "parser4_inline",
            "reporting": "report4_multiturn_legacy",
        }
        self.assertEqual(variants["eval4_single_legacy"]["expands_to"], expected_single)
        self.assertEqual(variants["eval4_multiturn_legacy"]["expands_to"], expected_multi)
        labels = set(variants)
        self.assertFalse(
            [
                (value["variant_id"], target)
                for value in variants.values()
                for target in value.get("expands_to", {}).values()
                if target not in labels
            ]
        )

    def test_registry_validator_rejects_dangling_expansion(self) -> None:
        registry = self.generated_registry()
        tampered = copy.deepcopy(registry)
        target = next(value for value in tampered["variants"] if value["variant_id"] == "eval5_canonical")
        target["expands_to"]["metric"] = "missing_metric"
        with self.assertRaisesRegex(audited.AuditedEnvironmentError, "dangling expands_to"):
            audited.validate_registry_document(tampered)

    def test_eval5_and_self_refine_entrypoints_are_usable_and_distinct(self) -> None:
        registry = self.generated_registry()
        variants = {value["variant_id"]: value for value in registry["variants"]}
        self.assertEqual(
            variants["eval5_canonical"]["entrypoints"],
            [
                "origins/experiments5/evaluation/eval_singleturn.py",
                "origins/experiments5/evaluation/eval_multiturn.py",
            ],
        )
        self.assertEqual(
            variants["eval5_canonical"]["supporting_paths"],
            ["origins/experiments5/src/evaluation/metrics.py"],
        )
        api = variants["self_refine5_api"]["entrypoints"]
        vllm = variants["self_refine5_vllm"]["entrypoints"]
        self.assertNotEqual(api, vllm)
        self.assertTrue(all("run_api_" in value for value in api))
        self.assertTrue(all("run_vllm_" in value for value in vllm))

    def test_forbidden_label_regex_catches_all_eval6_infer6_positions(self) -> None:
        forbidden = ["eval6_release", "infer6_api", "x_eval6", "x_infer6", "x_eval6_tail", "x_infer6_tail"]
        allowed = ["dataset_dev6", "preference_config_dev6", "evaluate6_release", "inference6_api"]
        self.assertTrue(all(audited.FORBIDDEN_LABEL_RE.search(value) for value in forbidden))
        self.assertFalse(any(audited.FORBIDDEN_LABEL_RE.search(value) for value in allowed))

    def test_v2_spec_pins_exact_trust_inputs_and_v1_is_unchanged(self) -> None:
        self.assertEqual(sha256(V1 / ".experiment-env-overlay" / "seal.json"), V1_SEAL_SHA256)
        spec = json.loads(V2_SPEC.read_bytes())
        self.assertEqual(set(spec["expected_sha256"]), {"audit", "cp0", "profiles", "registry", "source_pre"})
        self.assertTrue(spec["destination"].endswith("exp45-audited-overlay-v2-782df2d84077f93b"))

    def test_publisher_rejects_missing_expected_hash_key_before_hash_checks(self) -> None:
        spec = json.loads(V2_SPEC.read_bytes())
        spec["expected_sha256"].pop("registry")
        spec["expected_sha256"]["audit"] = "0" * 64
        spec["destination"] = f"environments/.must-not-publish-{uuid.uuid4().hex}"
        with tempfile.NamedTemporaryFile("w", suffix=".json", dir="/tmp", delete=False) as handle:
            json.dump(spec, handle)
            bad_spec = Path(handle.name)
        try:
            with self.assertRaisesRegex(audited.AuditedEnvironmentError, "expected_sha256 keys"):
                audited.publish_audited_overlay(ROOT, bad_spec)
            self.assertFalse(ROOT.joinpath(spec["destination"]).exists())
        finally:
            bad_spec.unlink()

    def test_list_validates_overlay_before_reading_registry(self) -> None:
        with mock.patch.object(audited, "validate_audited_overlay", side_effect=audited.ValidationError("blocked")) as validate:
            with self.assertRaisesRegex(audited.ValidationError, "blocked"):
                audited.audited_variants(ROOT)
        validate.assert_called_once()

    def test_v2_dry_run_covers_eval4_eval5_and_runtime_distinction(self) -> None:
        single4 = audited.build_audited_plan(ROOT, "eval4_single_legacy", 0, "DRY4S")
        multi4 = audited.build_audited_plan(ROOT, "eval4_multiturn_legacy", 0, "DRY4M")
        single5 = audited.build_audited_plan(ROOT, "eval5_canonical", 0, "DRY5S")
        multi5 = audited.build_audited_plan(ROOT, "eval5_canonical", 1, "DRY5M")
        api = audited.build_audited_plan(ROOT, "self_refine5_api", 0, "DRYAPI")
        vllm = audited.build_audited_plan(ROOT, "self_refine5_vllm", 0, "DRYVLLM")
        self.assertIn("singleturn", " ".join(single4["argv"]))
        self.assertIn("multiturn", " ".join(multi4["argv"]))
        self.assertIn("eval_singleturn.py", " ".join(single5["argv"]))
        self.assertIn("eval_multiturn.py", " ".join(multi5["argv"]))
        self.assertIn("run_api_singleturn.sh", " ".join(api["argv"]))
        self.assertIn("run_vllm_singleturn.sh", " ".join(vllm["argv"]))
        self.assertNotEqual(api["argv"], vllm["argv"])

    def test_negative_entrypoint_index_is_rejected(self) -> None:
        with self.assertRaisesRegex(audited.AuditedEnvironmentError, "non-negative"):
            audited.build_audited_plan(ROOT, "eval5_canonical", -1, "DRYNEGATIVE")

    def test_build_cli_defaults_to_the_checked_in_audit(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-B", str(ROOT / "scripts" / "build_audited_environment_registry.py")],
            cwd=ROOT,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
            text=True,
            capture_output=True,
            check=True,
        )
        result = json.loads(completed.stdout)
        self.assertEqual(Path(result["audit_path"]), AUDIT)
        source = (ROOT / "scripts" / "build_audited_environment_registry.py").read_text(encoding="utf-8")
        self.assertNotIn("/tmp/experiments7_registry_keep_merge_add.json", source)

    def test_passthrough_arguments_are_dry_run_only(self) -> None:
        plan = audited.build_audited_plan(
            ROOT, "eval5_canonical", 0, "DRYPASSTHROUGH", ["--sentinel-argument"]
        )
        self.assertIn("--sentinel-argument", plan["argv"])
        run_id = "must-not-run-with-passthrough"
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(ROOT / "scripts" / "experiment_variants.py"),
                "run",
                "--run-id",
                run_id,
                "--",
                "--sentinel-argument",
            ],
            cwd=ROOT,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertFalse((ROOT / "runs" / run_id).exists())

    def test_every_registry_entrypoint_resolves_to_sealed_regular_payload(self) -> None:
        registry = json.loads((V2 / ".experiment-env-overlay" / "registry.json").read_bytes())
        manifest = {
            json.loads(line)["path"]
            for line in (V2 / ".experiment-env-overlay" / "manifest.jsonl").read_bytes().splitlines()
        }
        entrypoints = [path for value in registry["variants"] for path in value.get("entrypoints", [])]
        self.assertTrue(entrypoints)
        for relative in entrypoints:
            with self.subTest(relative=relative):
                self.assertIn(relative, manifest)
                path = V2.joinpath(*Path(relative).parts)
                self.assertTrue(path.is_file())
                self.assertFalse(path.is_symlink())

    def test_validator_rejects_tampered_registry_profile_and_spec_bindings(self) -> None:
        mutations = ("registry", "profiles", "spec")
        for mutation in mutations:
            with self.subTest(mutation=mutation), cloned_v2() as (isolated, clone):
                control = clone / ".experiment-env-overlay"
                seal_path = control / "seal.json"
                seal = json.loads(seal_path.read_bytes())
                if mutation == "registry":
                    path = control / "registry.json"
                    value = json.loads(path.read_bytes())
                    value["tampered"] = True
                    rewrite_json(path, value)
                    seal["registry_sha256"] = sha256(path)
                elif mutation == "profiles":
                    path = control / "profiles.json"
                    value = json.loads(path.read_bytes())
                    value["registry_sha256"] = "0" * 64
                    rewrite_json(path, value)
                    seal["profiles_sha256"] = sha256(path)
                else:
                    seal["spec_sha256"] = "0" * 64
                rewrite_json(seal_path, seal)
                with self.assertRaises(audited.ValidationError):
                    audited.validate_audited_overlay(isolated, clone)

    def test_base_seal_binding_tamper_blocks_validate_list_and_dry_run(self) -> None:
        with isolated_tampered_base_binding() as (isolated, overlay):
            operations = (
                ("validate", lambda: audited.validate_audited_overlay(isolated, overlay)),
                ("list", lambda: audited.audited_variants(isolated)),
                (
                    "dry_run",
                    lambda: audited.build_audited_plan(
                        isolated, "eval5_canonical", 0, "TAMPERED-BASE-BINDING"
                    ),
                ),
            )
            for operation, invoke in operations:
                with self.subTest(operation=operation):
                    with self.assertRaisesRegex(
                        audited.ValidationError, "base seal differs from canonical spec"
                    ):
                        invoke()

    def test_validator_rejects_extra_or_writable_control_members(self) -> None:
        mutations = ("extra_file", "extra_dir", "extra_symlink", "writable")
        for mutation in mutations:
            with self.subTest(mutation=mutation), cloned_v2() as (isolated, clone):
                control = clone / ".experiment-env-overlay"
                os.chmod(control, 0o755)
                if mutation == "extra_file":
                    (control / "extra.txt").write_text("x", encoding="utf-8")
                    os.chmod(control / "extra.txt", 0o444)
                elif mutation == "extra_dir":
                    (control / "extra").mkdir(mode=0o555)
                elif mutation == "extra_symlink":
                    os.symlink("seal.json", control / "extra-link")
                else:
                    os.chmod(control / "seal.json", 0o644)
                os.chmod(control, 0o555)
                with self.assertRaises(audited.ValidationError):
                    audited.validate_audited_overlay(isolated, clone)


if __name__ == "__main__":
    unittest.main()
