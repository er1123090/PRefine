from __future__ import annotations

import copy
import tempfile
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import run_gpu
from ecpr import integrity
from ecpr.io import (
    load_json,
    sha256_bytes,
    sha256_file,
    write_json,
    write_jsonl,
)
from ecpr.prepare import build_public_tasks


MODEL = {
    "id": "Fixture/Model",
    "snapshot": "fixture-snapshot",
    "tensor_parallel_size": 4,
}
CANDIDATE_POLICY = {
    "minimum_support": 2,
    "minimum_confidence": 0.67,
    "maximum_conflict_ratio": 0.34,
    "maximum_stored_hypotheses": 24,
    "maximum_routed_hypotheses": 6,
    "overlay_lexical_token_cap": 384,
    "memory_lexical_token_cap": 1536,
    "recency_tiebreak_only": True,
    "raw_api_history_in_candidate_prompt": False,
}
INFERENCE_BUDGET = {
    "temperature": 0.0,
    "seed": 2026071600,
    "max_tokens": 1024,
    "calls_per_case": 1,
    "retries": 0,
}


def fixture_contract(
    latent_scope_amendment_sha256: str = "0" * 64,
) -> dict:
    return {
        "schema_version": 1,
        "kind": "expected_final_runtime_model_pipeline_contract",
        "runner_mode": "explicit_--final_or_--preflight-only",
        "candidate_id": "ecpr_v1",
        "final_attempts_allowed": 1,
        "model": {
            **MODEL,
            "snapshot_path": "/model/fixture-snapshot",
            "served_model_name": MODEL["id"],
        },
        "runtime": {
            "vllm_executable": "/runtime/vllm",
            "host": "127.0.0.1",
            "port": 8129,
            "gpu_indices": [0, 1, 2, 3],
            "gpu_preflight_command": [
                "timeout",
                "5",
                "nvidia-smi",
                "--query-gpu=index",
                "--format=csv,noheader",
            ],
            "gpu_preflight_subprocess_timeout_seconds": 8,
            "server_start_timeout_seconds": 600,
        },
        "action_budget": {**INFERENCE_BUDGET, "n": 1},
        "candidate_policy": copy.deepcopy(CANDIDATE_POLICY),
        "routing_gate": copy.deepcopy(integrity.ROUTING_GATE_CONTRACT),
        "candidate_memory_scope": copy.deepcopy(integrity.CANDIDATE_MEMORY_SCOPE_CONTRACT),
        "latent_scope_amendment_sha256": latent_scope_amendment_sha256,
        "pipeline": {
            "stages": list(integrity.EXPECTED_PIPELINE_STAGES),
            "forbidden_stages": ["prepare"],
            "base_url": "http://127.0.0.1:8129",
            "outputs": {
                "journal": "artifacts/provider_calls.final.jsonl",
                "baseline_latent": "artifacts/memory.prefine.jsonl",
                "baseline_latent_manifest": "artifacts/memory.prefine.jsonl.manifest.json",
                "memory": "artifacts/memory.ecpr.jsonl",
                "memory_manifest": "artifacts/memory.ecpr.jsonl.manifest.json",
                "baseline": "artifacts/predictions.baseline.jsonl",
                "baseline_manifest": "artifacts/predictions.baseline.jsonl.manifest.json",
                "candidate": "artifacts/predictions.candidate.jsonl",
                "candidate_manifest": "artifacts/predictions.candidate.jsonl.manifest.json",
                "summary": "reports/summary.json",
            },
        },
    }


@contextmanager
def sealed_fixture():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for relative in (
            "ecpr",
            "configs",
            "artifacts",
            "evaluator_vault",
            "manifests",
            "reports",
            "external",
        ):
            (root / relative).mkdir(parents=True, exist_ok=True)

        source_root = Path(__file__).resolve().parents[1]
        internal_input_configs = {
            "single_query": "query_singleturn.json",
            "single_schema": "schema_single.json",
            "multi_query": "query_multiturn-domain.json",
            "multi_schema": "schema_multi.json",
            "preference_slots": "preference_slots.json",
        }
        inputs = {}
        for name in sorted(integrity.EXPECTED_EXTERNAL_INPUTS):
            path = root / "external" / f"{name}.json"
            config_name = internal_input_configs.get(name)
            if config_name is None:
                path.write_text('{"fixture":true}\n', encoding="utf-8")
            else:
                path.write_bytes(
                    (source_root / "configs" / config_name).read_bytes()
                )
            inputs[name] = {
                "path": str(path),
                "sha256": sha256_file(path),
            }

        preregistration = {
            "schema_version": 1,
            "status": "locked_before_target_metrics",
            "candidate": "ecpr_v1",
            "inputs": inputs,
            "model": copy.deepcopy(MODEL),
            "candidate_parameters": copy.deepcopy(CANDIDATE_POLICY),
            "inference_budget": copy.deepcopy(INFERENCE_BUDGET),
        }
        write_json(root / "preregistration.json", preregistration)
        preregistration_sha256 = sha256_file(root / "preregistration.json")

        amendment = {
            "schema_version": 1,
            "status": "immutable_before_any_target_metric_or_model_call",
            "original_preregistration_sha256": preregistration_sha256,
            "confirmatory_candidate": "ecpr_v1",
            "confirmatory_candidate_count": 1,
            "development_candidates_evaluated": 0,
            "development_stopping_rule_invoked": False,
            "final_attempts_allowed": 1,
        }
        write_json(root / integrity.AMENDMENT, amendment)
        registration = {
            "schema_version": 1,
            "kind": "confirmatory_candidate_registration",
            "status": "immutable_before_any_target_model_or_metric_call",
            "candidate_id": "ecpr_v1",
            "candidate_count": 1,
            "confirmatory_ablation_flags": [],
            "original_preregistration_sha256": preregistration_sha256,
            "protocol_amendment_sha256": sha256_file(
                root / integrity.AMENDMENT
            ),
        }
        write_json(root / integrity.CANDIDATE_REGISTRATION, registration)
        (root / "PREREGISTRATION.md").write_text(
            "# fixture preregistration\n", encoding="utf-8"
        )

        ontology = load_json(source_root / integrity.PUBLIC_DOMAIN_ONTOLOGY)
        for config_name in integrity.EXPECTED_CONFIG_FILES:
            source = source_root / "configs" / config_name
            if config_name in {
                "query_singleturn.json",
                "query_multiturn-domain.json",
                "schema_single.json",
                "schema_multi.json",
                "preference_slots.json",
                "latent_trait_ontology.json",
                "latent_trait_ontology.vlt2.json",
                "public_domain_ontology.json",
            }:
                (root / "configs" / config_name).write_bytes(source.read_bytes())
            else:
                write_json(root / "configs" / config_name, {})

        routing_amendment = {
            "schema_version": 1,
            "kind": "public_query_relevance_gate_integrity_amendment",
            "status": "immutable_before_any_target_model_or_metric_call",
            "candidate_id": "ecpr_v1",
            "original_preregistration_sha256": preregistration_sha256,
            "protocol_amendment_sha256": sha256_file(root / integrity.AMENDMENT),
            "candidate_registration_sha256": sha256_file(
                root / integrity.CANDIDATE_REGISTRATION
            ),
            "public_domain_ontology_sha256": sha256_file(
                root / integrity.PUBLIC_DOMAIN_ONTOLOGY
            ),
            "routing_gate": copy.deepcopy(integrity.ROUTING_GATE_CONTRACT),
            "prior_target_model_calls": 0,
            "prior_target_metrics_computed": 0,
        }
        write_json(
            root / integrity.ROUTING_INTEGRITY_AMENDMENT,
            routing_amendment,
        )

        latent_scope_amendment = {
            "schema_version": 1,
            "kind": "candidate_typed_memory_scope_integrity_amendment",
            "status": "immutable_before_any_target_model_or_metric_call",
            "candidate_id": "ecpr_v1",
            "parent_routing_integrity_amendment_sha256": sha256_file(
                root / integrity.ROUTING_INTEGRITY_AMENDMENT
            ),
            "candidate_memory_scope": copy.deepcopy(
                integrity.CANDIDATE_MEMORY_SCOPE_CONTRACT
            ),
            "trigger": (
                "static_cross_domain_latent_leakage_audit_before_any_"
                "target_model_or_metric_call"
            ),
            "prior_target_model_calls": 0,
            "prior_target_metrics_computed": 0,
        }
        write_json(
            root / integrity.LATENT_SCOPE_AMENDMENT,
            latent_scope_amendment,
        )
        latent_scope_amendment_sha256 = sha256_file(
            root / integrity.LATENT_SCOPE_AMENDMENT
        )

        schema_hashes = {
            "single": sha256_file(root / "configs/schema_single.json"),
            "multi": sha256_file(root / "configs/schema_multi.json"),
        }
        tasks = build_public_tasks(
            [
                {
                    "case_key": "ignored",
                    "example_id": "u1",
                    "mode": "singleturn",
                    "query": "User: fixture",
                    "schema_key": "single",
                }
            ],
            schema_hashes,
        )
        write_jsonl(root / "artifacts/tasks.jsonl", tasks)
        write_jsonl(
            root / "artifacts/history.sanitized.jsonl",
            [{"example_id": "u1", "sessions": []}],
        )
        write_json(
            root / integrity.PQR_ROUTING_COVERAGE,
            integrity.build_routing_coverage(root),
        )
        write_json(
            root / "evaluator_vault/sealed_manifest.json",
            {
                "schema_version": 1,
                "preregistration_sha256": preregistration_sha256,
                "task_sha256": sha256_file(root / "artifacts/tasks.jsonl"),
                "history_sha256": sha256_file(
                    root / "artifacts/history.sanitized.jsonl"
                ),
                "gold_sha256": "0" * 64,
                "case_count": 1,
                "history_count": 1,
                "unassigned_target_variant_count": 0,
            },
        )
        write_json(
            root / "manifests/environment.json",
            {
                "schema_version": 1,
                "model_snapshot": "/model/fixture-snapshot",
                "vllm_executable": "/runtime/vllm",
                "gpu_contract": {
                    "indices": [0, 1, 2, 3],
                    "tensor_parallel_size": 4,
                },
            },
        )
        write_json(root / "manifests/singleturn.bundle.json", {})
        write_json(root / "manifests/multiturn.bundle.json", {})
        write_json(
            root / integrity.HISTORICAL_EXPECTED_RUNTIME_CONTRACT,
            fixture_contract(latent_scope_amendment_sha256),
        )

        for name in integrity.EXPECTED_ECPR_PYTHON:
            (root / "ecpr" / name).write_text(
                "# sealed fixture source\n", encoding="utf-8"
            )
        for name in integrity.EXPECTED_ROOT_PYTHON:
            (root / name).write_text(
                "# sealed fixture source\n", encoding="utf-8"
            )

        historical_runtime_sha256 = sha256_file(
            root / integrity.HISTORICAL_EXPECTED_RUNTIME_CONTRACT
        )
        v1_latent_trait_ontology = load_json(
            root / integrity.LATENT_TRAIT_ONTOLOGY_V1
        )
        v1_latent_trait_ontology_sha256 = sha256_file(
            root / integrity.LATENT_TRAIT_ONTOLOGY_V1
        )
        safe_transfer = {
            "schema_version": 1,
            "kind": "verified_latent_transfer_safe_restoration_amendment",
            "status": "immutable_before_any_target_model_or_metric_call",
            "candidate_id": "ecpr_v1",
            "candidate_revision": integrity.VLT1_CANDIDATE_REVISION,
            "parent_latent_scope_amendment_sha256": latent_scope_amendment_sha256,
            "parent_expected_runtime_contract_sha256": historical_runtime_sha256,
            "latent_trait_ontology_sha256": v1_latent_trait_ontology_sha256,
            "provenance_contract_sha256": integrity._digest_json(
                v1_latent_trait_ontology["provenance"]
            ),
            "safe_transfer_contract": copy.deepcopy(integrity.VLT1_CONTRACT),
            "candidate_memory_scope": copy.deepcopy(
                integrity.VLT1_CANDIDATE_MEMORY_SCOPE_CONTRACT
            ),
            "supersession": copy.deepcopy(
                integrity.VLT1_SUPERSESSION_CONTRACT
            ),
            "implementation_sha256": copy.deepcopy(
                integrity.VLT1_IMPLEMENTATION_HASHES
            ),
            "justification": (
                "safe_restoration_of_originally_preregistered_latent_channel_"
                "before_any_target_model_call_or_target_metric"
            ),
            "prior_target_model_calls": 0,
            "prior_target_metrics_computed": 0,
        }
        write_json(root / integrity.SAFE_TRANSFER_AMENDMENT, safe_transfer)
        safe_transfer_sha256 = sha256_file(
            root / integrity.SAFE_TRANSFER_AMENDMENT
        )

        v1_runtime = load_json(
            root / integrity.HISTORICAL_EXPECTED_RUNTIME_CONTRACT
        )
        v1_runtime.update(
            {
                "candidate_revision": integrity.VLT1_CANDIDATE_REVISION,
                "parent_expected_runtime_contract_sha256": historical_runtime_sha256,
                "safe_transfer_amendment_sha256": safe_transfer_sha256,
                "latent_trait_ontology_sha256": v1_latent_trait_ontology_sha256,
                "safe_transfer_contract": copy.deepcopy(integrity.VLT1_CONTRACT),
                "candidate_memory_scope": copy.deepcopy(
                    integrity.VLT1_CANDIDATE_MEMORY_SCOPE_CONTRACT
                ),
            }
        )
        write_json(root / integrity.EXPECTED_RUNTIME_CONTRACT_V1, v1_runtime)
        v1_runtime_sha256 = sha256_file(
            root / integrity.EXPECTED_RUNTIME_CONTRACT_V1
        )

        v2_latent_trait_ontology = load_json(
            root / integrity.LATENT_TRAIT_ONTOLOGY_V2
        )
        v2_latent_trait_ontology_sha256 = sha256_file(
            root / integrity.LATENT_TRAIT_ONTOLOGY_V2
        )
        vlt_audit = {
            "schema_version": 2,
            "kind": "audited_verified_latent_transfer_v2_amendment",
            "status": "immutable_before_any_target_model_or_metric_call",
            "candidate_id": "ecpr_v1",
            "candidate_revision": integrity.CANDIDATE_REVISION,
            "parent_safe_transfer_amendment_sha256": safe_transfer_sha256,
            "parent_expected_runtime_contract_vlt1_sha256": v1_runtime_sha256,
            "parent_latent_trait_ontology_vlt1_sha256": (
                v1_latent_trait_ontology_sha256
            ),
            "latent_trait_ontology_vlt2_sha256": (
                v2_latent_trait_ontology_sha256
            ),
            "provenance_contract_sha256": integrity._digest_json(
                v2_latent_trait_ontology["provenance"]
            ),
            "safe_transfer_contract": copy.deepcopy(integrity.VLT_CONTRACT),
            "candidate_memory_scope": copy.deepcopy(
                integrity.VLT_CANDIDATE_MEMORY_SCOPE_CONTRACT
            ),
            "supersession": copy.deepcopy(integrity.VLT_SUPERSESSION_CONTRACT),
            "implementation_sha256": integrity._vlt_implementation_hashes(root),
            "audit_guarantees": copy.deepcopy(integrity.VLT2_AUDIT_GUARANTEES),
            "prior_target_model_calls": 0,
            "prior_target_metrics_computed": 0,
        }
        write_json(root / integrity.VLT_AUDIT_AMENDMENT, vlt_audit)
        vlt_audit_sha256 = sha256_file(root / integrity.VLT_AUDIT_AMENDMENT)

        active_runtime = copy.deepcopy(v1_runtime)
        active_runtime.update(
            {
                "candidate_revision": integrity.CANDIDATE_REVISION,
                "parent_expected_runtime_contract_sha256": v1_runtime_sha256,
                "vlt_audit_amendment_sha256": vlt_audit_sha256,
                "latent_trait_ontology_sha256": v2_latent_trait_ontology_sha256,
                "latent_trait_ontology_path": str(
                    integrity.LATENT_TRAIT_ONTOLOGY_V2
                ),
                "provider_timeout_seconds": 120.0,
                "routing_gate": copy.deepcopy(
                    integrity.VLT2_ROUTING_GATE_CONTRACT
                ),
                "safe_transfer_contract": copy.deepcopy(integrity.VLT_CONTRACT),
                "candidate_memory_scope": copy.deepcopy(
                    integrity.VLT_CANDIDATE_MEMORY_SCOPE_CONTRACT
                ),
            }
        )
        write_json(root / integrity.EXPECTED_RUNTIME_CONTRACT_V2, active_runtime)

        with mock.patch.multiple(
            integrity,
            ORIGINAL_PREREGISTRATION_SHA256=preregistration_sha256,
            ROUTING_INTEGRITY_AMENDMENT_SHA256=sha256_file(
                root / integrity.ROUTING_INTEGRITY_AMENDMENT
            ),
            LATENT_SCOPE_AMENDMENT_SHA256=latent_scope_amendment_sha256,
            HISTORICAL_EXPECTED_RUNTIME_CONTRACT_SHA256=historical_runtime_sha256,
            VLT1_EXPECTED_RUNTIME_CONTRACT_SHA256=v1_runtime_sha256,
            VLT1_SAFE_TRANSFER_AMENDMENT_SHA256=safe_transfer_sha256,
            VLT1_LATENT_TRAIT_ONTOLOGY_FILE_SHA256=(
                v1_latent_trait_ontology_sha256
            ),
            VLT2_LATENT_TRAIT_ONTOLOGY_FILE_SHA256=(
                v2_latent_trait_ontology_sha256
            ),
        ):
            integrity.seal_implementation(root)
            yield root


class IntegritySealTests(unittest.TestCase):
    def test_every_sealed_file_and_external_input_mutation_fails(self):
        with sealed_fixture() as root:
            manifest = load_json(root / integrity.IMPLEMENTATION_MANIFEST)
            for relative in manifest["files"]:
                with self.subTest(relative=relative):
                    path = root / relative
                    original = path.read_bytes()
                    path.write_bytes(original + b"\n")
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        integrity.validate_implementation_manifest(root)
                    path.write_bytes(original)

            for name, record in manifest["frozen_external_inputs"].items():
                with self.subTest(external=name):
                    path = Path(record["declared_path"])
                    original = path.read_bytes()
                    path.write_bytes(original + b"\n")
                    integrity.validate_implementation_manifest(root)
                    path.write_bytes(original)

            _manifest, digest = integrity.validate_implementation_manifest(root)
            self.assertEqual(
                digest, sha256_file(root / integrity.IMPLEMENTATION_MANIFEST)
            )
            unexpected = root / "ecpr/unexpected.py"
            unexpected.write_text("# unexpected\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unexpected"):
                integrity.validate_implementation_manifest(root)
            unexpected.unlink()

    def test_ontology_mutation_invalidates_routing_chain_and_seal(self):
        with sealed_fixture() as root:
            path = root / integrity.PUBLIC_DOMAIN_ONTOLOGY
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "ontology binding"):
                integrity.validate_registration_chain(root)
            with self.assertRaises(ValueError):
                integrity.validate_implementation_manifest(root)

    def test_latent_scope_runtime_scope_and_coverage_tampering_fail(self):
        with sealed_fixture() as root:
            manifest = load_json(root / integrity.IMPLEMENTATION_MANIFEST)
        invalid_values = [
            {},
            {"GetHotels": []},
            {"GetHotels": ["seat", "Seat"]},
            {"GetHotels": ["missing"]},
            {" GetHotels": ["seat"]},
            {"GetHotels": [" seat"]},
        ]
        with sealed_fixture() as root:
            for value in invalid_values:
                with self.subTest(value=value):
                    write_json(root / "configs/preference_slots.json", value)
                    with self.assertRaises(ValueError):
                        integrity.validate_preference_slots(root)

    def test_attempt_and_terminal_are_immutable_and_attempt_bound(self):
        with sealed_fixture() as root:
            attempt = integrity.acquire_final_attempt(
                root,
                attempt_id_factory=lambda: uuid.UUID(
                    "11111111-1111-4111-8111-111111111111"
                ),
                clock=lambda: 10,
            )
            validated, _manifest, _digest = integrity.validate_final_attempt(
                root
            )
            self.assertEqual(validated, attempt)
            terminal = integrity.commit_attempt_terminal(
                root,
                attempt["attempt_id"],
                "pipeline_failed",
                "fixture",
                clock=lambda: 20,
            )
            self.assertEqual(
                integrity.validate_attempt_terminal(
                    root, attempt["attempt_id"]
                ),
                terminal,
            )
            with self.assertRaises(FileExistsError):
                integrity.acquire_final_attempt(root)
            with self.assertRaises(FileExistsError):
                integrity.commit_attempt_terminal(
                    root,
                    attempt["attempt_id"],
                    "pipeline_failed",
                    "replacement",
                )
            self.assertFalse((root / "evaluator_vault/gold.jsonl").exists())


class FinalRunnerProtocolTests(unittest.TestCase):
    ATTEMPT_ID = "22222222-2222-4222-8222-222222222222"

    def minimal_manifest(self):
        return {"expected_contract": fixture_contract()}

    def fake_acquire(self, root, binding, runner_nonce):
        attempt = {
            "schema_version": 2,
            "kind": "sealed_final_attempt",
            "attempt_id": self.ATTEMPT_ID,
            "acquired_unix_ns": 1,
            "runner_nonce_sha256": sha256_bytes(bytes(runner_nonce)),
            "binding": binding,
        }
        integrity.commit_json_once(root / integrity.ATTEMPT_LOCK, attempt)
        return attempt

    def test_final_command_inventory_never_rebuilds_frozen_inputs(self):
        commands = run_gpu.final_pipeline_commands(
            Path("/fixture"), fixture_contract()
        )
        flattened = [
            argument for arguments, _check in commands for argument in arguments
        ]
        self.assertNotIn("prepare", flattened)
        self.assertEqual(
            [arguments[0] for arguments, _check in commands],
            ["build-memory", "infer", "infer", "evaluate"],
        )

    def test_preflight_failure_does_not_consume_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            acquired = []
            result = run_gpu.main(
                ["--final"],
                root=root,
                seal_validate_fn=lambda _root: (
                    self.minimal_manifest(),
                    "seal",
                ),
                preflight_fn=lambda _contract: (False, "blocked"),
                runtime_check_fn=lambda _root, _contract: (True, "ok"),
                binding_fn=lambda _root: {"seal": "seal"},
                acquire_fn=lambda _root, binding: acquired.append(binding),
                attempt_validate_fn=lambda _root: (_ for _ in ()).throw(
                    AssertionError("attempt validation must not run")
                ),
                executor_fn=lambda _root, _attempt, _nonce: (_ for _ in ()).throw(
                    AssertionError("executor must not run")
                ),
            )
            self.assertEqual(result, 2)
            self.assertEqual(acquired, [])
            self.assertFalse((root / integrity.ATTEMPT_LOCK).exists())

    def test_post_lock_failures_retain_attempt_and_second_final_rejects(self):
        failure_modes = ("validation", "executor")
        for failure_mode in failure_modes:
            with self.subTest(failure_mode=failure_mode):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    manifest = self.minimal_manifest()
                    seal_calls = []

                    def seal_validate(_root):
                        seal_calls.append(True)
                        return manifest, "seal"

                    def validate_attempt(_root):
                        if failure_mode == "validation":
                            raise RuntimeError("post-lock validation failure")
                        return load_json(root / integrity.ATTEMPT_LOCK), manifest, "seal"

                    def execute(_root, _attempt, _nonce):
                        if failure_mode == "executor":
                            raise RuntimeError("delegated pipeline failure")
                        return 0

                    kwargs = {
                        "root": root,
                        "seal_validate_fn": seal_validate,
                        "preflight_fn": lambda _contract: (True, "ok"),
                        "runtime_check_fn": lambda _root, _contract: (
                            True,
                            "ok",
                        ),
                        "binding_fn": lambda _root: {
                            "implementation_manifest_sha256": "seal"
                        },
                        "acquire_fn": self.fake_acquire,
                        "attempt_validate_fn": validate_attempt,
                        "executor_fn": execute,
                    }
                    with mock.patch.object(
                        run_gpu, "validate_stage_chain", return_value={}
                    ), mock.patch.object(run_gpu, "commit_stage_once") as commit:
                        first = run_gpu.main(["--final"], **kwargs)
                    self.assertEqual(first, 1)
                    self.assertTrue((root / integrity.ATTEMPT_LOCK).is_file())
                    self.assertEqual(commit.call_count, 1)
                    failed_payload = commit.call_args.args[3]
                    self.assertEqual(failed_payload["execution_status"], "FAILED")
                    self.assertNotIn(
                        "delegated pipeline failure", failed_payload["detail"]
                    )
                    self.assertEqual(len(seal_calls), 2)

                    second = run_gpu.main(["--final"], **kwargs)
                    self.assertEqual(second, 2)
                    self.assertTrue((root / integrity.ATTEMPT_LOCK).is_file())

    def test_preflight_only_reverifies_without_consuming_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []

            def seal_validate(_root):
                calls.append("seal")
                return self.minimal_manifest(), "seal"

            result = run_gpu.main(
                ["--preflight-only"],
                root=root,
                seal_validate_fn=seal_validate,
                preflight_fn=lambda _contract: (True, "ok"),
                runtime_check_fn=lambda _root, _contract: (True, "ok"),
                acquire_fn=lambda _root, _binding: (_ for _ in ()).throw(
                    AssertionError("preflight-only must not acquire")
                ),
            )
            self.assertEqual(result, 0)
            self.assertEqual(calls, ["seal", "seal"])
            self.assertFalse((root / integrity.ATTEMPT_LOCK).exists())


if __name__ == "__main__":
    unittest.main()
