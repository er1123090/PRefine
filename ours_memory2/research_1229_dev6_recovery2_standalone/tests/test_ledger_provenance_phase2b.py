from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ecpr.integrity import acquire_final_attempt
from ecpr.io import canonical_json, iter_jsonl, load_json, sha256_text, write_jsonl
from ecpr.latent_firewall import CANDIDATE_REVISION, unattested_latent
from ecpr.ledger import (
    ECPR_MEMORY,
    PREFINE_MEMORY,
    PROVIDER_JOURNAL,
    ZERO_SHA256,
    ProviderJournal,
    action_seed,
    enforce_final_build_memory_arguments,
    enforce_final_inference_arguments,
    final_paths,
    journal_record_sha256,
    publish_memory_manifests,
    publish_prediction_manifest,
    validate_arm_memory_binding,
    validate_journal_records,
    validate_run_dag,
)
from ecpr.request_contract import CallSpec
from tests.test_integrity_attempt_phase2a import fixture_contract, sealed_fixture


USAGE = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}


def _intent(journal: ProviderJournal, phase: str, call_key: str, **overrides):
    request = {
        "endpoint": "http://127.0.0.1:8129/v1/chat/completions",
        "model": "Fixture/Model",
        "messages": [{"role": "user", "content": "fixture"}],
        "seed": 7,
        "temperature": 0.0,
        "max_tokens": 32,
        "timeout_seconds": 120.0,
        "json_object": False,
        "schema_sha256": "1" * 64,
    }
    request.update(overrides)
    return journal.begin_call(
        phase=phase,
        call_key=call_key,
        call_spec=CallSpec.from_messages(**request),
    )


class ProviderJournalTests(unittest.TestCase):
    ATTEMPT_ID = "33333333-3333-4333-8333-333333333333"

    def test_tamper_missing_orphan_and_duplicate_terminal_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            journal = ProviderJournal(root / "journal.jsonl", self.ATTEMPT_ID)
            intent = _intent(journal, "memory_generation", "u1")
            journal.finish_call(intent, status="ok", usage=USAGE, response="{}")
            records = journal.validate()

            tampered = copy.deepcopy(records)
            tampered[1]["usage"]["total_tokens"] += 1
            with self.assertRaisesRegex(ValueError, "digest"):
                validate_journal_records(tampered, self.ATTEMPT_ID)

            missing = ProviderJournal(root / "missing.jsonl", self.ATTEMPT_ID)
            _intent(missing, "memory_generation", "u1")
            with self.assertRaisesRegex(ValueError, "missing its terminal"):
                missing.validate()

            orphan = copy.deepcopy(records[1])
            orphan["sequence"] = 1
            orphan["previous_record_sha256"] = ZERO_SHA256
            orphan["record_sha256"] = journal_record_sha256(orphan)
            with self.assertRaisesRegex(ValueError, "orphan"):
                validate_journal_records([orphan], self.ATTEMPT_ID)

            duplicate = copy.deepcopy(records[1])
            duplicate["sequence"] = 3
            duplicate["previous_record_sha256"] = records[1]["record_sha256"]
            duplicate["record_sha256"] = journal_record_sha256(duplicate)
            with self.assertRaisesRegex(ValueError, "duplicate provider terminal"):
                validate_journal_records([*records, duplicate], self.ATTEMPT_ID)


class FinalArgumentGuardTests(unittest.TestCase):
    def test_arbitrary_latent_ablation_and_output_are_rejected(self):
        contract = fixture_contract()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "latent-path"):
                enforce_final_build_memory_arguments(
                    root,
                    contract,
                    preregistration=root / "preregistration.json",
                    history=root / "artifacts/history.sanitized.jsonl",
                    preference_slots=root / "configs/preference_slots.json",
                    output=root / ECPR_MEMORY,
                    base_url=contract["pipeline"]["base_url"],
                    latent_path=root / "arbitrary.jsonl",
                )
            with self.assertRaisesRegex(ValueError, "ablations"):
                enforce_final_inference_arguments(
                    root,
                    contract,
                    arm="candidate",
                    preregistration=root / "preregistration.json",
                    output=root / final_paths(root)["candidate"],
                    base_url=contract["pipeline"]["base_url"],
                    ablations={"no_typed"},
                )
            with self.assertRaisesRegex(ValueError, "alternate output"):
                enforce_final_inference_arguments(
                    root,
                    contract,
                    arm="baseline",
                    preregistration=root / "preregistration.json",
                    output=root / "alternate.jsonl",
                    base_url=contract["pipeline"]["base_url"],
                    ablations=set(),
                )

    def test_arm_memory_binding_is_directional(self):
        digest = "1" * 64
        validate_arm_memory_binding(
            "baseline",
            baseline_latent_manifest_sha256=digest,
            memory_manifest_sha256=None,
        )
        validate_arm_memory_binding(
            "candidate",
            baseline_latent_manifest_sha256=None,
            memory_manifest_sha256=digest,
        )
        with self.assertRaisesRegex(ValueError, "baseline cannot consume"):
            validate_arm_memory_binding(
                "baseline",
                baseline_latent_manifest_sha256=digest,
                memory_manifest_sha256=digest,
            )
        with self.assertRaisesRegex(ValueError, "candidate memory"):
            validate_arm_memory_binding(
                "candidate",
                baseline_latent_manifest_sha256=None,
                memory_manifest_sha256=None,
            )


class ManifestDagTests(unittest.TestCase):
    def _journal_call(
        self,
        journal: ProviderJournal,
        *,
        phase: str,
        call_key: str,
        seed: int,
        schema_sha256: str,
        response: str,
    ) -> None:
        intent = _intent(
            journal,
            phase,
            call_key,
            seed=seed,
            temperature=0.0,
            max_tokens=1024,
            schema_sha256=schema_sha256,
        )
        journal.finish_call(intent, status="ok", usage=USAGE, response=response)

    def _prediction_row(self, task: dict, arm: str, seed: int, schema_sha256: str) -> dict:
        return {
            "case_key": task["case_key"],
            "example_id": task["example_id"],
            "mode": task["mode"],
            "arm": arm,
            "llm_output": "[]",
            "status": "ok",
            "model_snapshot": "fixture-snapshot",
            "seed": seed,
            "temperature": 0.0,
            "max_tokens": 1024,
            "calls": 1,
            "prompt_hash": sha256_text("fixture"),
            "schema_hash": schema_sha256,
            "memory_hash": sha256_text("fixture-memory"),
            "usage": USAGE,
        }

    @mock.patch(
        "ecpr.ledger.replay_memory_generation",
        return_value=({"kind": "fixture_semantic_replay"}, {}),
    )
    @mock.patch("ecpr.ledger.build_action_case")
    def test_runner_manifests_override_row_metadata_and_form_forward_dag(
        self, action_case, _replay
    ):
        action_case.side_effect = lambda **kwargs: SimpleNamespace(
            prompt="fixture",
            memory_block="fixture-memory",
            audit_artifact={"arm": kwargs["arm"]},
        )
        with sealed_fixture() as root:
            attempt = acquire_final_attempt(root)
            attempt_id = attempt["attempt_id"]
            journal = ProviderJournal(root / PROVIDER_JOURNAL, attempt_id)
            journal.initialize_once()

            write_jsonl(
                root / ECPR_MEMORY,
                [
                    {
                        "example_id": "u1",
                        "latent_abstraction": {"implicit_pref": "aisle"},
                        "latent_attestation": unattested_latent(
                            {"implicit_pref": "aisle"},
                            source="external_latent",
                            reason_code="EXTERNAL_UNATTESTED",
                        ).as_dict(),
                        "typed_hypotheses": [],
                        "method": "ecpr_v1",
                        "candidate_revision": CANDIDATE_REVISION,
                    }
                ],
            )
            write_jsonl(
                root / PREFINE_MEMORY,
                [
                    {
                        "example_id": "u1",
                        "latent_abstraction": {"implicit_pref": "aisle"},
                        "method": "prefine_v1",
                    }
                ],
            )
            checkpoint = journal.checkpoint()
            self._journal_call(
                journal,
                phase="memory_generation",
                call_key="u1:memory",
                seed=1,
                schema_sha256="1" * 64,
                response="{}",
            )
            memory_slice = journal.slice_from(checkpoint, "memory_generation")
            publish_memory_manifests(root, attempt_id, memory_slice)

            task = next(iter(iter_jsonl(root / "artifacts/tasks.jsonl")))
            schema = load_json(root / "configs/schema_single.json")
            schema_sha256 = sha256_text(canonical_json(schema))
            budget = fixture_contract()["action_budget"]
            seed = action_seed(budget["seed"], task["case_key"])

            baseline_path = final_paths(root)["baseline"]
            write_jsonl(baseline_path, [self._prediction_row(task, "baseline", seed, schema_sha256)])
            checkpoint = journal.checkpoint()
            self._journal_call(
                journal,
                phase="action_baseline",
                call_key=task["case_key"],
                seed=seed,
                schema_sha256=schema_sha256,
                response="[]",
            )
            baseline_slice = journal.slice_from(checkpoint, "action_baseline")
            publish_prediction_manifest(
                root, attempt_id, "baseline", baseline_path, baseline_slice
            )

            candidate_path = final_paths(root)["candidate"]
            write_jsonl(candidate_path, [self._prediction_row(task, "candidate", seed, schema_sha256)])
            checkpoint = journal.checkpoint()
            self._journal_call(
                journal,
                phase="action_candidate",
                call_key=task["case_key"],
                seed=seed,
                schema_sha256=schema_sha256,
                response="[]",
            )
            candidate_slice = journal.slice_from(checkpoint, "action_candidate")
            publish_prediction_manifest(
                root, attempt_id, "candidate", candidate_path, candidate_slice
            )

            dag = validate_run_dag(
                root, attempt_id, baseline_path, candidate_path
            )
            self.assertTrue(dag["equal_action_budget"])


if __name__ == "__main__":
    unittest.main()
