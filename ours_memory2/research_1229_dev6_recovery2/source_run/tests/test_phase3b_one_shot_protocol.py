from __future__ import annotations

import copy
import inspect
import json
import os
import stat
import subprocess
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import critic
import run_gpu
from ecpr import final_protocol
from ecpr.final_protocol import (
    EVALUATION_EVIDENCE,
    classify_protocol_state,
    commit_stage_once,
    open_regular_bytes_once,
    strict_success,
    validate_stage_chain,
    verify_runner_nonce,
)
from ecpr.independent_audit import audit_paired_rows, assert_registered_agreement
from ecpr.io import (
    canonical_json,
    load_json,
    sha256_bytes,
    sha256_file,
    write_json,
    write_json_once,
    write_jsonl,
)
from ecpr.ledger import ProviderJournal, validate_run_dag
from ecpr.request_contract import CallSpec


NONCE = b"r" * 32
DIGEST = "1" * 64
ATTEMPT_ID = "77777777-7777-4777-8777-777777777777"


def process_identity(pid: int = 12) -> dict:
    return {
        "pid": pid,
        "start_ticks": 34,
        "executable": "/python",
        "executable_sha256": "2" * 64,
        "argv_sha256": "3" * 64,
    }


def compute_baseline() -> dict:
    return {
        "ownership_contract": run_gpu.GPU_OWNERSHIP_CONTRACT,
        "capture_scope": "all_nvidia_compute_apps",
        "compute_apps": [],
        "expected_gpus": [
            {
                "gpu_uuid": f"GPU-{index}",
                "pci_bus_id": f"00000000:{index + 1:02x}:00.0",
            }
            for index in range(4)
        ],
    }


def topology_payload(process_group_id: int = 22) -> dict:
    processes = [process_identity(22 + index) for index in range(4)]
    mappings = [
        {
            "host_pid": 9000 + index,
            "gpu_uuid": f"GPU-{index}",
            "pci_bus_id": f"00000000:{index + 1:02x}:00.0",
            "used_gpu_memory_mib": 40000 + index,
        }
        for index in range(4)
    ]
    baseline = compute_baseline()
    return {
        "ownership_contract": run_gpu.GPU_OWNERSHIP_CONTRACT,
        "pid_namespace_contract": run_gpu.PID_NAMESPACE_CONTRACT,
        "process_group_id": process_group_id,
        "descendant_processes": processes,
        "compute_app_mappings": mappings,
        "mapped_gpu_uuids": [f"GPU-{index}" for index in range(4)],
        "prelaunch_compute_apps_sha256": sha256_bytes(
            canonical_json(baseline).encode()
        ),
    }


def runtime_payload() -> dict:
    python_argv = ["python", "run_gpu.py", "--final"]
    vllm_argv = ["vllm", "serve", "/model"]
    gvr = {
        "state_machine_source_sha256": "4" * 64,
        "max_refinements_per_session": 10,
        "sequence": ["draft", "verify", "refine_if_invalid"],
        "generation": {"prompt_sha256": "5" * 64},
        "verification": {"prompt_sha256": "6" * 64},
        "transitions": {"valid": "stop"},
        "held_out_action_feedback": "forbidden_no_outgoing_provider_or_memory_edges",
    }
    values = {"PATH": "/bin"}
    return {
        "python": {
            "executable": "/python",
            "sha256": "7" * 64,
            "version": "3.10",
            "argv": python_argv,
            "argv_sha256": sha256_bytes(canonical_json(python_argv).encode()),
        },
        "vllm": {
            "executable": "/vllm",
            "sha256": "8" * 64,
            "version": "0.13.0",
            "argv": vllm_argv,
            "argv_sha256": sha256_bytes(canonical_json(vllm_argv).encode()),
            "process": process_identity(22),
            "prelaunch_compute_apps": compute_baseline(),
            "topology_at_readiness": topology_payload(),
        },
        "model": {"snapshot_path": "/model", "tree_sha256": "9" * 64},
        "gpus": [
            {
                "index": index,
                "uuid": f"GPU-{index}",
                "pci_bus_id": f"00000000:{index + 1:02x}:00.0",
            }
            for index in range(4)
        ],
        "server": {
            "base_url": "http://127.0.0.1:8129",
            "models_response": {"data": [{"id": "model"}]},
            "models_response_raw_sha256": "a" * 64,
        },
        "environment": {
            "policy": "fixed_allowlist_no_credentials",
            "values": values,
            "sha256": sha256_bytes(canonical_json(values).encode()),
        },
        "runner": process_identity(11),
        "planned_commands": [
            {
                "name": name,
                "argv_sha256": character * 64,
                "timeout_seconds": 60,
            }
            for name, character in zip(
                ("memory", "baseline", "candidate", "evaluate", "critic"),
                ("b", "c", "d", "e", "f"),
            )
        ],
        "gvr_call_graph": gvr,
    }


def pre_gold_payload(runtime_record: dict) -> dict:
    authoritative = {
        "model": {"id": "fixture-model", "snapshot": "fixture-snapshot"},
        "action_budget": {
            "seed": 1,
            "temperature": 0.0,
            "max_tokens": 10,
            "calls_per_case": 1,
            "retries": 0,
            "n": 1,
        },
        "provider_timeout_seconds": 2.0,
        "call_count_per_arm": 1,
    }
    dag = {
        "schema_version": 1,
        "attempt_id": ATTEMPT_ID,
        "provider_journal_sha256": "f" * 64,
        "memory_manifest_sha256": "1" * 64,
        "baseline_prediction_manifest_sha256": "2" * 64,
        "candidate_prediction_manifest_sha256": "3" * 64,
        "equal_action_budget": True,
        "task_count": 1,
        "arms": ["baseline", "candidate"],
        "memory_call_count": 1,
        "baseline_action_call_count": 1,
        "candidate_action_call_count": 1,
        "total_call_count": 3,
        "expected_total_call_count": 3,
        "journal_record_count": 6,
        "expected_journal_record_count": 6,
        "final_journal_sequence": 6,
        "final_journal_record_sha256": "e" * 64,
        "phases": ["memory_generation", "action_baseline", "action_candidate"],
        "no_extra_provider_calls_or_phases": True,
        "authoritative_action_contract": authoritative,
        "authoritative_action_contract_sha256": sha256_bytes(
            canonical_json(authoritative).encode()
        ),
    }
    post_first = topology_payload()
    post_all = topology_payload()
    return {
        "run_dag": dag,
        "run_dag_sha256": sha256_bytes(canonical_json(dag).encode()),
        "official_outputs": {
            name: {
                "path": path,
                "sha256": {
                    "journal": "f",
                    "memory_manifest": "1",
                    "baseline_manifest": "2",
                    "candidate_manifest": "3",
                }.get(name, "d")
                * 64,
            }
            for name, path in final_protocol.PRE_GOLD_OUTPUT_PATHS.items()
        },
        "server_stopped": {
            "process": process_identity(22),
            "process_group_id": 22,
            "recorded_processes": topology_payload()["descendant_processes"],
            "return_code": -15,
            "all_recorded_processes_dead": True,
            "process_group_empty": True,
        },
        "runtime_revalidated_before_gold": True,
        "implementation_manifest_sha256": DIGEST,
        "runtime_record_sha256": runtime_record["record_sha256"],
        "provider_journal_seal": {
            "path": "artifacts/provider_calls.final.jsonl",
            "sha256": "f" * 64,
            "device": 1,
            "inode": 2,
            "size": 100,
            "record_count": 6,
            "final_sequence": 6,
            "final_record_sha256": "e" * 64,
            "mode": "0400",
        },
        "post_first_provider_topology": post_first,
        "post_first_provider_topology_sha256": sha256_bytes(
            canonical_json(post_first).encode()
        ),
        "post_provider_topology": post_all,
        "post_provider_topology_sha256": sha256_bytes(
            canonical_json(post_all).encode()
        ),
    }


def gold_open_payload(pre_gold_record: dict) -> dict:
    return {
        "declared_gold_sha256": "2" * 64,
        "pre_gold_record_sha256": pre_gold_record["record_sha256"],
        "capability_transport": "anonymous_pipe_fd",
        "capability_secret_persisted": False,
        "capability_digest_persisted": False,
    }


def result_payload(performance: str) -> dict:
    reported_pass = performance == "PASS"
    return {
        "execution_status": "COMPLETE",
        "integrity_status": "PASS",
        "performance_status": performance,
        "summary": {
            "path": "reports/summary.json",
            "sha256": "3" * 64,
            "reported_pass": reported_pass,
        },
        "evaluation_evidence": {
            "path": str(EVALUATION_EVIDENCE),
            "sha256": "4" * 64,
        },
        "detail": "fixture",
    }


def critic_payload(
    root: Path,
    result_record: dict,
    performance: str,
    audit: str = "PASS",
    *,
    axes: dict | None = None,
) -> dict:
    passed = audit == "PASS"
    if axes is None:
        axes = {
            "schema_version": 1,
            "integrity": {"status": "PASS"},
            "execution": {"status": "COMPLETE"},
            "performance": {"status": performance},
        }
    return {
        "kind": "strict_final_critic_receipt",
        "audit_status": audit,
        "integrity_status": "PASS" if passed else "FAIL",
        "execution_status": "COMPLETE",
        "performance_status": performance if passed else "NOT_RUN",
        "result_record_sha256": result_record["record_sha256"],
        "implementation_manifest_sha256": DIGEST,
        "critic_source_sha256": sha256_file(root / "critic.py"),
        "summary_sha256": "3" * 64,
        "evaluation_evidence_sha256": "4" * 64,
        "audit_axes_sha256": sha256_bytes(canonical_json(axes).encode()),
        "detail": (
            "strict_critic_audit_passed"
            if passed
            else "strict_critic_audit_failed_no_retry"
        ),
    }


@contextmanager
def protocol_root():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "artifacts").mkdir()
        (root / "reports").mkdir()
        critic_path = root / "critic.py"
        critic_path.write_text("# fixture critic\n", encoding="utf-8")
        implementation = {
            "expected_contract": {},
            "files": {
                "critic.py": {
                    "sha256": sha256_file(critic_path),
                    "size": critic_path.stat().st_size,
                }
            },
        }
        attempt = {
            "schema_version": 2,
            "kind": "sealed_final_attempt",
            "attempt_id": ATTEMPT_ID,
            "acquired_unix_ns": 1,
            "runner_nonce_sha256": sha256_bytes(NONCE),
            "binding": {
                "implementation_manifest_sha256": DIGEST,
                "declared_gold_sha256": "2" * 64,
            },
        }
        write_json(root / "artifacts/final_test.attempt.json", attempt)
        with mock.patch.object(
            final_protocol,
            "validate_final_attempt",
            return_value=(attempt, implementation, DIGEST),
        ):
            yield root, attempt


class Phase3BStateTests(unittest.TestCase):
    def test_nonce_wrong_missing_and_exact_pipe_length_reject(self):
        attempt = {"runner_nonce_sha256": sha256_bytes(NONCE)}
        verify_runner_nonce(attempt, NONCE)
        with self.assertRaisesRegex(ValueError, "wrong runner nonce"):
            verify_runner_nonce(attempt, b"x" * 32)
        with self.assertRaisesRegex(ValueError, "exactly 32"):
            verify_runner_nonce(attempt, b"")
        read_fd, write_fd = os.pipe()
        os.write(write_fd, b"short")
        os.close(write_fd)
        with self.assertRaisesRegex(ValueError, "exactly 32"):
            final_protocol.read_secret_fd(read_fd, "fixture")

    def test_full_chain_commit_once_readonly_and_performance_fail_complete(self):
        with protocol_root() as (root, _attempt):
            runtime = commit_stage_once(root, ATTEMPT_ID, "runtime", runtime_payload(), NONCE)
            pre_gold = commit_stage_once(root, ATTEMPT_ID, "pre_gold", pre_gold_payload(runtime), NONCE)
            commit_stage_once(root, ATTEMPT_ID, "gold_open", gold_open_payload(pre_gold), NONCE)
            result = commit_stage_once(
                root, ATTEMPT_ID, "result", result_payload("FAIL"), NONCE
            )
            pending = classify_protocol_state(root)
            self.assertEqual(pending["execution"], "CONSUMED_INCOMPLETE")
            self.assertEqual(pending["last_stage"], "result")
            commit_stage_once(
                root,
                ATTEMPT_ID,
                "critic",
                critic_payload(root, result, "FAIL"),
                NONCE,
            )
            self.assertEqual(list(validate_stage_chain(root)), list(final_protocol.STAGE_ORDER))
            state = classify_protocol_state(root)
            self.assertEqual(state["execution"], "COMPLETE")
            self.assertEqual(state["performance"], "FAIL")
            self.assertFalse(strict_success(state))
            axes = {
                "integrity": {"status": "PASS"},
                "execution": {"status": "COMPLETE"},
                "performance": {"status": "FAIL"},
            }
            self.assertEqual(critic.critic_exit_code(axes), 1)
            mode = stat.S_IMODE(final_protocol.stage_path(root, "runtime").stat().st_mode)
            self.assertEqual(mode, 0o400)
            with self.assertRaises(FileExistsError):
                commit_stage_once(root, ATTEMPT_ID, "runtime", runtime_payload(), NONCE)

    def test_hard_crash_is_consumed_incomplete_and_failed_terminal_is_valid(self):
        with protocol_root() as (root, _attempt):
            commit_stage_once(root, ATTEMPT_ID, "runtime", runtime_payload(), NONCE)
            self.assertEqual(classify_protocol_state(root)["execution"], "CONSUMED_INCOMPLETE")
        with protocol_root() as (root, _attempt):
            failed = {
                "execution_status": "FAILED",
                "integrity_status": "FAIL",
                "performance_status": "NOT_RUN",
                "summary": None,
                "evaluation_evidence": None,
                "detail": "pre-gold failure",
            }
            commit_stage_once(root, ATTEMPT_ID, "result", failed, NONCE)
            state = classify_protocol_state(root)
            self.assertEqual(state["execution"], "FAILED")
            self.assertEqual(state["integrity"], "FAIL")
            self.assertNotIn("gold_open", validate_stage_chain(root))

    def test_failed_critic_receipt_forces_integrity_failure(self):
        with protocol_root() as (root, _attempt):
            runtime = commit_stage_once(
                root, ATTEMPT_ID, "runtime", runtime_payload(), NONCE
            )
            pre_gold = commit_stage_once(
                root,
                ATTEMPT_ID,
                "pre_gold",
                pre_gold_payload(runtime),
                NONCE,
            )
            commit_stage_once(
                root,
                ATTEMPT_ID,
                "gold_open",
                gold_open_payload(pre_gold),
                NONCE,
            )
            result = commit_stage_once(
                root,
                ATTEMPT_ID,
                "result",
                result_payload("PASS"),
                NONCE,
            )
            commit_stage_once(
                root,
                ATTEMPT_ID,
                "critic",
                critic_payload(root, result, "PASS", audit="FAIL"),
                NONCE,
            )
            state = classify_protocol_state(root)
            self.assertEqual(state["integrity"], "FAIL")
            self.assertEqual(state["execution"], "COMPLETE")
            self.assertEqual(state["performance"], "NOT_RUN")
            self.assertFalse(strict_success(state))

    def test_critic_source_tamper_is_rejected_against_manifest(self):
        with protocol_root() as (root, _attempt):
            runtime = commit_stage_once(root, ATTEMPT_ID, "runtime", runtime_payload(), NONCE)
            pre_gold = commit_stage_once(root, ATTEMPT_ID, "pre_gold", pre_gold_payload(runtime), NONCE)
            commit_stage_once(root, ATTEMPT_ID, "gold_open", gold_open_payload(pre_gold), NONCE)
            result = commit_stage_once(root, ATTEMPT_ID, "result", result_payload("PASS"), NONCE)
            commit_stage_once(
                root,
                ATTEMPT_ID,
                "critic",
                critic_payload(root, result, "PASS"),
                NONCE,
            )
            (root / "critic.py").write_text("# tampered critic\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source binding"):
                validate_stage_chain(root)

    def test_critic_audit_axis_tamper_is_rejected_after_rehash(self):
        axes = {
            "schema_version": 1,
            "integrity": {"status": "PASS"},
            "execution": {"status": "COMPLETE"},
            "performance": {"status": "PASS"},
        }
        with protocol_root() as (root, attempt):
            runtime = commit_stage_once(root, ATTEMPT_ID, "runtime", runtime_payload(), NONCE)
            pre_gold = commit_stage_once(root, ATTEMPT_ID, "pre_gold", pre_gold_payload(runtime), NONCE)
            commit_stage_once(root, ATTEMPT_ID, "gold_open", gold_open_payload(pre_gold), NONCE)
            result = commit_stage_once(root, ATTEMPT_ID, "result", result_payload("PASS"), NONCE)
            commit_stage_once(
                root,
                ATTEMPT_ID,
                "critic",
                critic_payload(root, result, "PASS", axes=axes),
                NONCE,
            )
            critic_path = root / "critic.py"
            implementation = {
                "files": {
                    "critic.py": {
                        "sha256": sha256_file(critic_path),
                        "size": critic_path.stat().st_size,
                    }
                }
            }
            with mock.patch.object(
                critic,
                "validate_final_attempt",
                return_value=(attempt, implementation, DIGEST),
            ):
                critic.validate_committed_critic_receipt(root, axes)
                receipt_path = final_protocol.stage_path(root, "critic")
                receipt = load_json(receipt_path)
                receipt["payload"]["audit_axes_sha256"] = "a" * 64
                receipt["record_sha256"] = final_protocol.stage_record_sha256(receipt)
                write_json(receipt_path, receipt)
                with self.assertRaisesRegex(critic.AuditFailure, "audit-axis digest"):
                    critic.validate_committed_critic_receipt(root, axes)

    def test_orphan_artifact_without_attempt_is_integrity_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "reports").mkdir()
            (root / "reports/summary.json").write_text("{}\n", encoding="utf-8")
            state = classify_protocol_state(root)
            self.assertEqual(state["integrity"], "FAIL")
            self.assertEqual(state["execution"], "NOT_RUN")

    def test_pre_lock_broken_symlink_orphan_rejects_without_nonce_or_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "artifacts").mkdir()
            orphan = root / "artifacts/predictions.baseline.jsonl"
            orphan.symlink_to(root / "missing-predictions.jsonl")
            acquire = mock.Mock(side_effect=AssertionError("attempt must not be acquired"))
            nonce = mock.Mock(side_effect=AssertionError("nonce must not be generated"))
            result = run_gpu.main(
                ["--final"],
                root=root,
                seal_validate_fn=mock.Mock(
                    side_effect=AssertionError("static seal must follow orphan check")
                ),
                acquire_fn=acquire,
                nonce_factory=nonce,
            )
            self.assertEqual(result, 2)
            self.assertFalse((root / "artifacts/final_test.attempt.json").exists())
            self.assertEqual(
                load_json(root / "reports/GPU_BLOCKER.json")["reason"],
                "orphan_final_artifact_before_attempt",
            )
            acquire.assert_not_called()
            nonce.assert_not_called()

    def test_failed_detail_is_fixed_and_does_not_persist_exception_text(self):
        payload = run_gpu._failed_result_payload(
            RuntimeError("secret=gold-capability;token=credential"),
            "pre_gold",
        )
        self.assertEqual(
            payload["detail"],
            "stage=pre_gold;error_class=RuntimeError;"
            "code=FINAL_PIPELINE_ABORTED_NO_RETRY",
        )
        self.assertNotIn("secret", payload["detail"])
        self.assertNotIn("credential", payload["detail"])
        self.assertEqual(payload["integrity_status"], "FAIL")

    def test_stage_symlink_is_rejected_without_following(self):
        with protocol_root() as (root, _attempt):
            target = root / "target.json"
            target.write_text("{}\n", encoding="utf-8")
            final_protocol.stage_path(root, "runtime").symlink_to(target)
            with self.assertRaisesRegex(ValueError, "non-symlink"):
                validate_stage_chain(root)

    def test_attempt_symlink_is_rejected_without_following(self):
        with protocol_root() as (root, _attempt):
            attempt_path = root / "artifacts/final_test.attempt.json"
            target = root / "attempt-target.json"
            attempt_path.rename(target)
            attempt_path.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "non-symlink"):
                final_protocol.validate_final_attempt_v2(root)


class Phase3BIOTests(unittest.TestCase):
    def test_same_fd_bytes_survive_path_swap_and_symlink_rejects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "gold.jsonl"
            path.write_bytes(b'{"value":"original"}\n')
            replacement = root / "replacement"
            replacement.write_bytes(b'{"value":"replacement"}\n')
            real_fstat = os.fstat
            swapped = False

            def swap_after_open(fd):
                nonlocal swapped
                identity = real_fstat(fd)
                if not swapped:
                    swapped = True
                    path.unlink()
                    replacement.rename(path)
                return identity

            with mock.patch("ecpr.final_protocol.os.fstat", side_effect=swap_after_open):
                payload, _identity = open_regular_bytes_once(path)
            self.assertIn(b"original", payload)
            self.assertIn(b"replacement", path.read_bytes())
            link = root / "link.jsonl"
            link.symlink_to(path)
            with self.assertRaisesRegex(ValueError, "non-symlink"):
                open_regular_bytes_once(link)

    def test_commit_once_is_readonly_nonoverwriting_symlink_safe_and_concurrent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "receipt.json"

            def publish(value):
                try:
                    write_json_once(path, {"value": value})
                    return "ok"
                except FileExistsError:
                    return "exists"

            with ThreadPoolExecutor(max_workers=8) as pool:
                outcomes = list(pool.map(publish, range(8)))
            self.assertEqual(outcomes.count("ok"), 1)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o400)
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_json_once(path, {"replacement": True})
            self.assertEqual(path.read_bytes(), original)
            symlink = root / "symlink.json"
            symlink.symlink_to(path)
            with self.assertRaisesRegex(ValueError, "symlink"):
                write_json_once(symlink, {})

    def test_provider_journal_same_fd_read_symlink_reject_and_0400_seal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "artifacts/provider_calls.final.jsonl"
            journal = ProviderJournal(path, ATTEMPT_ID)
            journal.initialize_once()
            call = CallSpec.from_messages(
                endpoint="http://127.0.0.1/v1/chat/completions",
                model="fixture-model",
                messages=[{"role": "user", "content": "fixture"}],
                seed=7,
                temperature=0.0,
                max_tokens=8,
                json_object=False,
                schema_sha256="a" * 64,
                timeout_seconds=2.0,
            )
            intent = journal.begin_call(
                phase="memory_generation",
                call_key="session-0",
                call_spec=call,
            )
            journal.finish_call(
                intent,
                status="ok",
                usage={
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
                response="{}",
            )

            replacement = root / "replacement.jsonl"
            replacement.write_text("not-json\n", encoding="utf-8")
            original = root / "original.jsonl"
            real_fstat = os.fstat
            swapped = False

            def swap_after_open(fd):
                nonlocal swapped
                identity = real_fstat(fd)
                if not swapped:
                    swapped = True
                    path.rename(original)
                    replacement.rename(path)
                return identity

            with mock.patch("ecpr.ledger.os.fstat", side_effect=swap_after_open):
                records = journal.validate()
            self.assertEqual(len(records), 2)
            self.assertEqual(path.read_text(encoding="utf-8"), "not-json\n")

            path.unlink()
            original.rename(path)
            link = root / "artifacts/journal-link.jsonl"
            link.symlink_to(path)
            with self.assertRaisesRegex(ValueError, "missing or unsafe"):
                ProviderJournal(link, ATTEMPT_ID).validate()

            receipt = run_gpu.seal_provider_journal(root, ATTEMPT_ID)
            self.assertEqual(receipt["record_count"], 2)
            self.assertEqual(receipt["final_sequence"], 2)
            self.assertEqual(receipt["sha256"], sha256_file(path))
            self.assertEqual(receipt["size"], path.stat().st_size)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o400)


class Phase3BMechanismTests(unittest.TestCase):
    def test_registered_and_independent_paths_agree_and_check_identity(self):
        gold = [{
            "case_key": "a",
            "example_id": "u1",
            "mode": "singleturn",
            "difficulty": "easy",
            "reference_ground_truth": ['Book(seat="quiet")'],
        }]
        base = self._prediction("baseline", "u1")
        candidate = self._prediction("candidate", "u1")
        prereg = self._prereg()
        provenance = {"equal_action_budget": True}
        independent = audit_paired_rows(
            gold_rows=gold,
            baseline_rows=[base],
            candidate_rows=[candidate],
            preference_slots={"Book": ["seat"]},
            preregistration=prereg,
            provenance=provenance,
        )
        from ecpr.evaluate import compute_paired_summary

        registered = compute_paired_summary(
            gold_rows=gold,
            baseline_rows=[base],
            candidate_rows=[candidate],
            preference_slots={"Book": ["seat"]},
            preregistration=prereg,
            provenance=provenance,
        )
        assert_registered_agreement(registered, independent)
        wrong = dict(candidate, example_id="wrong")
        with self.assertRaisesRegex(ValueError, "identity"):
            audit_paired_rows(
                gold_rows=gold,
                baseline_rows=[base],
                candidate_rows=[wrong],
                preference_slots={"Book": ["seat"]},
                preregistration=prereg,
                provenance=provenance,
            )
        wrong_mode = dict(candidate, mode="multiturn")
        with self.assertRaisesRegex(ValueError, "identity"):
            audit_paired_rows(
                gold_rows=gold,
                baseline_rows=[base],
                candidate_rows=[wrong_mode],
                preference_slots={"Book": ["seat"]},
                preregistration=prereg,
                provenance=provenance,
            )
        with self.assertRaisesRegex(ValueError, "identity"):
            compute_paired_summary(
                gold_rows=gold,
                baseline_rows=[base],
                candidate_rows=[wrong_mode],
                preference_slots={"Book": ["seat"]},
                preregistration=prereg,
                provenance=provenance,
            )

    def test_process_identity_binds_executable_bytes_and_argv(self):
        identity = final_protocol.process_start_identity(os.getpid())
        self.assertEqual(identity["executable_sha256"], sha256_file(identity["executable"]))
        self.assertEqual(
            identity["argv_sha256"],
            sha256_bytes((Path("/proc") / str(os.getpid()) / "cmdline").read_bytes()),
        )

    def test_gpu_receipt_filters_exact_visible_set_and_rejects_bad_mapping(self):
        rows = "\n".join(
            f"{index}, GPU-{index}, 00000000:{index + 1:02x}:00.0"
            for index in range(6)
        )
        with mock.patch.object(run_gpu, "_capture_text", return_value=rows):
            identities = run_gpu._gpu_identities(
                {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}
            )
        self.assertEqual([item["index"] for item in identities], [0, 1, 2, 3])
        self.assertEqual(len({item["uuid"] for item in identities}), 4)
        self.assertTrue(all(item["pci_bus_id"] for item in identities))

        with protocol_root() as (root, _attempt):
            payload = runtime_payload()
            payload["vllm"]["topology_at_readiness"]["compute_app_mappings"][0][
                "host_pid"
            ] = payload["vllm"]["topology_at_readiness"]["compute_app_mappings"][1][
                "host_pid"
            ]
            with self.assertRaisesRegex(ValueError, "four-way unique"):
                commit_stage_once(root, ATTEMPT_ID, "runtime", payload, NONCE)

    def test_compute_app_parser_keeps_host_namespace_pids_opaque(self):
        output = "\n".join(
            f"{3705600 + index}, GPU-{index}, "
            f"00000000:{index + 1:02x}:00.0, {45000 + index}"
            for index in range(4)
        )
        with mock.patch.object(run_gpu, "_capture_text", return_value=output):
            rows = run_gpu._compute_app_rows({"PATH": "/bin"})
        self.assertEqual([row["host_pid"] for row in rows], list(range(3705600, 3705604)))
        self.assertTrue(all(row["host_pid"] > 1_000_000 for row in rows))

        gpus = runtime_payload()["gpus"]
        validated = run_gpu.validate_exclusive_compute_apps(rows, gpus)
        self.assertEqual(validated, rows)

        process = mock.Mock(pid=22)
        with mock.patch.object(os, "getpgid", return_value=22), mock.patch.object(
            run_gpu,
            "descendant_process_identities",
            return_value=[process_identity(22), process_identity(23)],
        ), mock.patch.object(run_gpu, "_compute_app_rows", return_value=rows):
            topology = run_gpu.capture_server_topology(
                process, {"PATH": "/bin"}, gpus, compute_baseline()
            )
        self.assertEqual(
            {mapping["host_pid"] for mapping in topology["compute_app_mappings"]},
            set(range(3705600, 3705604)),
        )
        self.assertTrue(
            {identity["pid"] for identity in topology["descendant_processes"]}.isdisjoint(
                {mapping["host_pid"] for mapping in topology["compute_app_mappings"]}
            )
        )

    def test_exclusive_compute_app_gate_rejects_busy_and_hostile_snapshots(self):
        gpus = runtime_payload()["gpus"]
        valid_rows = topology_payload()["compute_app_mappings"]
        with mock.patch.object(run_gpu, "_compute_app_rows", return_value=[]):
            self.assertEqual(
                run_gpu.capture_empty_compute_baseline({"PATH": "/bin"}, gpus),
                compute_baseline(),
            )
        with mock.patch.object(run_gpu, "_compute_app_rows", return_value=valid_rows):
            with self.assertRaisesRegex(ValueError, "already have active"):
                run_gpu.capture_empty_compute_baseline({"PATH": "/bin"}, gpus)

        hostile = {
            "missing": valid_rows[:-1],
            "duplicate_pid": [
                {**row, "host_pid": valid_rows[0]["host_pid"]}
                if index == 1
                else row
                for index, row in enumerate(valid_rows)
            ],
            "wrong_uuid": [
                {**row, "gpu_uuid": "GPU-unknown"} if index == 0 else row
                for index, row in enumerate(valid_rows)
            ],
            "wrong_bus": [
                {**row, "pci_bus_id": "00000000:ff:00.0"}
                if index == 0
                else row
                for index, row in enumerate(valid_rows)
            ],
            "zero_memory": [
                {**row, "used_gpu_memory_mib": 0} if index == 0 else row
                for index, row in enumerate(valid_rows)
            ],
        }
        for name, rows in hostile.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                run_gpu.validate_exclusive_compute_apps(rows, gpus)

        with mock.patch.object(
            run_gpu,
            "_capture_text",
            return_value="not-a-pid, GPU-0, 00000000:01:00.0, 1",
        ):
            with self.assertRaisesRegex(ValueError, "malformed"):
                run_gpu._compute_app_rows({"PATH": "/bin"})

    def test_attempt_locked_gpu_race_check_precedes_server_launch(self):
        source = inspect.getsource(run_gpu.execute_final_pipeline)
        self.assertLess(
            source.index("capture_empty_compute_baseline"),
            source.index("subprocess.Popen"),
        )

    def test_stop_receipt_proves_leader_descendant_and_group_are_dead(self):
        process = subprocess.Popen(
            ["/bin/sh", "-c", "sleep 60 & wait"],
            start_new_session=True,
        )
        try:
            leader = final_protocol.process_start_identity(process.pid)
            deadline = time.monotonic() + 2.0
            recorded = [leader]
            while time.monotonic() < deadline:
                recorded = run_gpu.descendant_process_identities(process.pid)
                if len(recorded) >= 2:
                    break
                time.sleep(0.01)
            self.assertGreaterEqual(len(recorded), 2)
            receipt = run_gpu.stop_exact_server(
                process,
                leader,
                recorded,
                process.pid,
            )
            self.assertTrue(receipt["all_recorded_processes_dead"])
            self.assertTrue(receipt["process_group_empty"])
            self.assertEqual(run_gpu.process_group_pids(process.pid), [])
            self.assertTrue(
                all(
                    not final_protocol.process_identity_is_live(identity)
                    for identity in recorded
                )
            )
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)

    def test_run_dag_materializes_model_seed_budget_and_exact_cardinality(self):
        memory = {
            "generation_call_ledger": {
                "first_sequence": 1,
                "last_sequence": 2,
                "previous_record_sha256": "0" * 64,
                "terminal_record_sha256": "1" * 64,
                "call_count": 1,
            }
        }
        contract = {
            "model": {"id": "fixture-model", "snapshot": "snapshot"},
            "action_budget": {
                "seed": 41,
                "temperature": 0.0,
                "max_tokens": 128,
                "calls_per_case": 1,
                "retries": 0,
                "n": 1,
            },
            "provider_timeout_seconds": 12.0,
            "call_count": 2,
        }
        baseline = {
            "tasks": {"row_count": 2},
            "action_call_ledger": {
                "first_sequence": 3,
                "last_sequence": 6,
                "previous_record_sha256": "1" * 64,
                "terminal_record_sha256": "2" * 64,
                "call_count": 2,
            },
            "authoritative_action_contract": copy.deepcopy(contract),
        }
        candidate = copy.deepcopy(baseline)
        candidate["action_call_ledger"] = {
            "first_sequence": 7,
            "last_sequence": 10,
            "previous_record_sha256": "2" * 64,
            "terminal_record_sha256": "3" * 64,
            "call_count": 2,
        }
        records = [
            {"sequence": index, "record_sha256": f"{index:064x}"}
            for index in range(1, 11)
        ]

        def manifest(_root, _attempt_id, arm, _path):
            return baseline if arm == "baseline" else candidate

        with mock.patch(
            "ecpr.ledger.validate_memory_manifests", return_value=({}, memory)
        ), mock.patch(
            "ecpr.ledger.validate_prediction_manifest", side_effect=manifest
        ), mock.patch(
            "ecpr.ledger.ProviderJournal.validate", return_value=records
        ), mock.patch(
            "ecpr.ledger.sha256_file", return_value="f" * 64
        ):
            dag = validate_run_dag(
                Path("/fixture"), ATTEMPT_ID, Path("b"), Path("c")
            )
            self.assertEqual(dag["task_count"], 2)
            self.assertEqual(dag["total_call_count"], 5)
            self.assertEqual(dag["expected_total_call_count"], 5)
            self.assertEqual(dag["journal_record_count"], 10)
            self.assertEqual(dag["expected_journal_record_count"], 10)
            self.assertEqual(
                dag["authoritative_action_contract"]["action_budget"]["seed"],
                41,
            )
            self.assertEqual(
                dag["authoritative_action_contract"]["model"]["id"],
                "fixture-model",
            )

            candidate["authoritative_action_contract"]["action_budget"]["seed"] = 42
            with self.assertRaisesRegex(ValueError, "contracts differ"):
                validate_run_dag(
                    Path("/fixture"), ATTEMPT_ID, Path("b"), Path("c")
                )

        candidate["authoritative_action_contract"] = copy.deepcopy(contract)
        candidate["action_call_ledger"]["last_sequence"] = 12
        twelve_records = [
            {"sequence": index, "record_sha256": f"{index:064x}"}
            for index in range(1, 13)
        ]
        with mock.patch(
            "ecpr.ledger.validate_memory_manifests", return_value=({}, memory)
        ), mock.patch(
            "ecpr.ledger.validate_prediction_manifest", side_effect=manifest
        ), mock.patch(
            "ecpr.ledger.ProviderJournal.validate", return_value=twelve_records
        ), mock.patch(
            "ecpr.ledger.sha256_file", return_value="f" * 64
        ):
            with self.assertRaisesRegex(ValueError, "record count"):
                validate_run_dag(
                    Path("/fixture"), ATTEMPT_ID, Path("b"), Path("c")
                )

    def test_evaluator_is_v2_only_and_opens_synthetic_gold_once_after_commit(self):
        from ecpr.evaluate import evaluate_paired

        with mock.patch(
            "ecpr.evaluate.validate_final_attempt_v2",
            side_effect=ValueError("Phase3B-B requires v2"),
        ):
            with self.assertRaisesRegex(ValueError, "requires v2"):
                evaluate_paired(
                    root=Path("/fixture"),
                    baseline_path=Path("b"),
                    candidate_path=Path("c"),
                    output=Path("o"),
                    preregistration=Path("p"),
                    final_attempt_id=ATTEMPT_ID,
                    runner_nonce=NONCE,
                    gold_open_fd=99,
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in ("configs", "evaluator_vault", "artifacts", "reports"):
                (root / relative).mkdir()
            write_json(root / "configs/preference_slots.json", {"Book": ["seat"]})
            write_json(root / "preregistration.json", self._prereg())
            (root / "artifacts/tasks.jsonl").write_text("fixture\n", encoding="utf-8")
            gold = [{
                "case_key": "a",
                "example_id": "u1",
                "mode": "singleturn",
                "difficulty": "easy",
                "reference_ground_truth": ['Book(seat="quiet")'],
            }]
            write_jsonl(root / "evaluator_vault/gold.jsonl", gold)
            baseline = self._prediction("baseline", "u1")
            candidate = self._prediction("candidate", "u1")
            write_jsonl(root / "artifacts/predictions.baseline.jsonl", [baseline])
            write_jsonl(root / "artifacts/predictions.candidate.jsonl", [candidate])
            write_json(
                root / "evaluator_vault/sealed_manifest.json",
                {
                    "preregistration_sha256": sha256_file(root / "preregistration.json"),
                    "task_sha256": sha256_file(root / "artifacts/tasks.jsonl"),
                    "gold_sha256": sha256_file(root / "evaluator_vault/gold.jsonl"),
                },
            )
            attempt = {
                "attempt_id": ATTEMPT_ID,
                "runner_nonce_sha256": sha256_bytes(NONCE),
            }
            implementation = {
                "preference_slots_sha256": sha256_file(
                    root / "configs/preference_slots.json"
                )
            }
            stages = {
                "runtime": {"payload": {}},
                "pre_gold": {
                    "record_sha256": "b" * 64,
                    "payload": {
                        "official_outputs": {
                            "baseline": {
                                "path": "artifacts/predictions.baseline.jsonl",
                                "sha256": sha256_file(
                                    root / "artifacts/predictions.baseline.jsonl"
                                ),
                            },
                            "candidate": {
                                "path": "artifacts/predictions.candidate.jsonl",
                                "sha256": sha256_file(
                                    root / "artifacts/predictions.candidate.jsonl"
                                ),
                            },
                        }
                    },
                },
            }
            events: list[str] = []
            real_open = final_protocol.open_regular_bytes_once

            def consume(_fd, _payload):
                events.append("consume")

            def commit(*_args, **_kwargs):
                events.append("commit")

            def open_once(path):
                events.append("open")
                return real_open(path)

            with mock.patch(
                "ecpr.evaluate.validate_final_attempt_v2",
                return_value=(attempt, implementation, "seal"),
            ), mock.patch(
                "ecpr.evaluate.enforce_final_evaluation_arguments"
            ), mock.patch(
                "ecpr.evaluate.validate_run_dag",
                return_value={"equal_action_budget": True},
            ), mock.patch(
                "ecpr.evaluate.validate_preference_slots",
                return_value={"Book": ["seat"]},
            ), mock.patch(
                "ecpr.evaluate.validate_stage_chain", return_value=stages
            ), mock.patch(
                "ecpr.evaluate.consume_gold_open_capability", side_effect=consume
            ), mock.patch(
                "ecpr.evaluate.commit_stage_once", side_effect=commit
            ), mock.patch(
                "ecpr.evaluate.open_regular_bytes_once", side_effect=open_once
            ) as opened:
                summary = evaluate_paired(
                    root=root,
                    baseline_path=root / "artifacts/predictions.baseline.jsonl",
                    candidate_path=root / "artifacts/predictions.candidate.jsonl",
                    output=root / "reports/summary.json",
                    preregistration=root / "preregistration.json",
                    final_attempt_id=ATTEMPT_ID,
                    runner_nonce=NONCE,
                    gold_open_fd=99,
                )
            self.assertFalse(summary["pass"])
            self.assertEqual(
                events,
                ["open", "open", "consume", "commit", "open"],
            )
            self.assertEqual(
                opened.call_args_list,
                [
                    mock.call(root / "artifacts/predictions.baseline.jsonl"),
                    mock.call(root / "artifacts/predictions.candidate.jsonl"),
                    mock.call(root / "evaluator_vault/gold.jsonl"),
                ],
            )
            self.assertEqual(stat.S_IMODE((root / "reports/summary.json").stat().st_mode), 0o400)

            replay_stages = {
                **stages,
                "gold_open": {"payload": {}},
            }
            replay_open = mock.Mock(
                side_effect=AssertionError("replay must fail before any file open")
            )
            with mock.patch(
                "ecpr.evaluate.validate_final_attempt_v2",
                return_value=(attempt, implementation, "seal"),
            ), mock.patch(
                "ecpr.evaluate.enforce_final_evaluation_arguments"
            ), mock.patch(
                "ecpr.evaluate.validate_run_dag",
                return_value={"equal_action_budget": True},
            ), mock.patch(
                "ecpr.evaluate.validate_preference_slots",
                return_value={"Book": ["seat"]},
            ), mock.patch(
                "ecpr.evaluate.validate_stage_chain",
                return_value=replay_stages,
            ), mock.patch(
                "ecpr.evaluate.open_regular_bytes_once",
                replay_open,
            ):
                with self.assertRaisesRegex(ValueError, "sealed pre_gold checkpoint"):
                    evaluate_paired(
                        root=root,
                        baseline_path=root / "artifacts/predictions.baseline.jsonl",
                        candidate_path=root / "artifacts/predictions.candidate.jsonl",
                        output=root / "reports/summary.json",
                        preregistration=root / "preregistration.json",
                        final_attempt_id=ATTEMPT_ID,
                        runner_nonce=NONCE,
                        gold_open_fd=99,
                    )
            replay_open.assert_not_called()

    def test_runner_order_and_exact_official_arms(self):
        contract = {
            "pipeline": {
                "base_url": "http://127.0.0.1:8129",
                "outputs": {
                    "baseline": "artifacts/b.jsonl",
                    "candidate": "artifacts/c.jsonl",
                    "summary": "reports/s.json",
                },
            }
        }
        commands = run_gpu.final_pipeline_commands(Path("/root"), contract, ATTEMPT_ID)
        self.assertEqual([command[0][0] for command in commands], ["build-memory", "infer", "infer", "evaluate"])
        self.assertEqual([commands[1][0][2], commands[2][0][2]], ["baseline", "candidate"])
        planned = run_gpu.planned_command_receipts(
            Path("/root"), contract, ATTEMPT_ID
        )
        self.assertEqual(
            [receipt["name"] for receipt in planned],
            ["memory", "baseline", "candidate", "evaluate", "critic"],
        )
        self.assertTrue(
            all(receipt["timeout_seconds"] > 0 for receipt in planned)
        )
        source = inspect.getsource(run_gpu.execute_final_pipeline)
        self.assertLess(source.index("stop_exact_server"), source.index('"pre_gold"'))
        self.assertLess(source.index('"pre_gold"'), source.index("gold_secret"))

    @staticmethod
    def _prediction(arm: str, example_id: str) -> dict:
        return {
            "case_key": "a",
            "example_id": example_id,
            "mode": "singleturn",
            "arm": arm,
            "llm_output": 'Book(seat="quiet")',
            "status": "ok",
            "model_snapshot": "snap",
            "seed": 1,
            "temperature": 0.0,
            "max_tokens": 10,
            "calls": 1,
            "prompt_hash": "p",
            "schema_hash": "s",
            "memory_hash": "m",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    @staticmethod
    def _prereg() -> dict:
        return {
            "statistics": {
                "bootstrap_draws": 3,
                "bootstrap_seed": 1,
                "randomization_draws": 3,
                "randomization_seed": 2,
            },
            "guardrails": {
                "minimum_each_task_delta_f1": -0.005,
                "maximum_preference_f1_drop": 0.005,
                "maximum_nonpreference_f1_drop": 0.005,
                "maximum_parse_failure_rate_increase": 0.005,
                "required_coverage": 1.0,
            },
            "pass": {"minimum_delta_bmf1": 0.01, "maximum_p_value_exclusive": 0.05},
        }


if __name__ == "__main__":
    unittest.main()
