"""Four-GPU, one-shot runner for the sealed fresh holdout."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from ecpr.contracts import PREDICTION_FIELDS, TASK_FIELDS
from ecpr.io import (
    canonical_json,
    iter_jsonl,
    load_json,
    sha256_file,
    sha256_text,
    unique_by,
    write_json_once,
)
from seal import validate


ROOT = Path(__file__).resolve().parent
VLLM = Path("/home/minseo/miniconda3/bin/vllm")
HOST = "127.0.0.1"
PORT = 8131
GPU_INDICES = (0, 1, 2, 3)


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in GPU_INDICES)
    environment["PYTHONHASHSEED"] = "0"
    return environment


def _gpu_preflight() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    if result.returncode != 0:
        raise RuntimeError("nvidia-smi preflight failed")
    identities: dict[int, dict[str, str]] = {}
    for line in result.stdout.splitlines():
        pieces = [piece.strip() for piece in line.split(",")]
        if len(pieces) != 4:
            raise ValueError("malformed nvidia-smi identity row")
        identities[int(pieces[0])] = {
            "index": pieces[0],
            "uuid": pieces[1],
            "memory_total_mib": pieces[2],
            "memory_used_mib": pieces[3],
        }
    if any(index not in identities for index in GPU_INDICES):
        raise RuntimeError("one or more required GPUs are unavailable")
    return {"gpus": [identities[index] for index in GPU_INDICES]}


def _runtime_contract(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    implementation = validate(root)
    preregistration = load_json(root / "preregistration.json")
    model = preregistration["model"]
    if tuple(model.get("gpu_indices", ())) != GPU_INDICES:
        raise ValueError("preregistration GPU set differs from fixed four-GPU contract")
    if int(model.get("tensor_parallel_size", 0)) != len(GPU_INDICES):
        raise ValueError("model tensor parallelism must equal four")
    model_path = Path(str(model["snapshot_path"]))
    if not VLLM.is_file() or not model_path.is_dir():
        raise RuntimeError("local vLLM executable or model snapshot is unavailable")
    return implementation, preregistration


def _vllm_argv(preregistration: dict[str, Any]) -> list[str]:
    model = preregistration["model"]
    return [
        str(VLLM),
        "serve",
        str(model["snapshot_path"]),
        "--served-model-name",
        str(model["served_model_name"]),
        "--tensor-parallel-size",
        str(model["tensor_parallel_size"]),
        "--host",
        HOST,
        "--port",
        str(PORT),
        "--dtype",
        "auto",
    ]


def _wait_server(process: subprocess.Popen[str], timeout_seconds: int = 600) -> None:
    deadline = time.monotonic() + timeout_seconds
    endpoint = f"http://{HOST}:{PORT}/v1/models"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("vLLM exited before readiness")
        try:
            with urllib.request.urlopen(endpoint, timeout=2) as response:
                value = json.loads(response.read().decode("utf-8"))
            if isinstance(value, dict) and isinstance(value.get("data"), list):
                return
        except (urllib.error.URLError, TimeoutError, UnicodeDecodeError, json.JSONDecodeError):
            time.sleep(2)
    raise RuntimeError("vLLM readiness timeout")


def _stop_server(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=30)


def _start_server(root: Path, preregistration: dict[str, Any], label: str) -> tuple[subprocess.Popen[str], Path]:
    log_path = root / "logs" / f"vllm.{label}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("x", encoding="utf-8")
    try:
        process = subprocess.Popen(
            _vllm_argv(preregistration),
            cwd=root,
            env=_environment(),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    finally:
        log.close()
    try:
        _wait_server(process)
    except BaseException:
        _stop_server(process)
        raise
    return process, log_path


def _run_command(root: Path, arguments: list[str], timeout_seconds: int) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, *arguments],
        cwd=root,
        env=_environment(),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    receipt = {
        "argv": [sys.executable, *arguments],
        "argv_sha256": sha256_text(canonical_json([sys.executable, *arguments])),
        "returncode": result.returncode,
        "stdout_sha256": sha256_text(result.stdout),
        "stderr_sha256": sha256_text(result.stderr),
    }
    if result.returncode != 0:
        raise RuntimeError("model/evaluator child command failed")
    return receipt


def _assert_final_outputs_absent(root: Path) -> None:
    paths = (
        root / "artifacts/final_attempt.json",
        root / "artifacts/memory.prefine_anchor.jsonl",
        root / "artifacts/memory.manifest.json",
        root / "artifacts/predictions.baseline.jsonl",
        root / "artifacts/predictions.candidate.jsonl",
        root / "artifacts/pre_gold_receipt.json",
        root / "reports/summary.json",
        root / "reports/evaluation_evidence.json",
        root / "reports/final_result.json",
        root / "reports/critic.json",
    )
    present = [str(path.relative_to(root)) for path in paths if path.exists() or path.is_symlink()]
    if present:
        raise FileExistsError("final-run output already exists: " + ", ".join(present))


def _pre_gold_receipt(root: Path, implementation: dict[str, Any]) -> dict[str, Any]:
    tasks = unique_by(iter_jsonl(root / "artifacts/tasks.jsonl"), "case_key")
    outputs: dict[str, dict[str, Any]] = {}
    paired = True
    rows_by_arm: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in ("baseline", "candidate"):
        path = root / "artifacts" / f"predictions.{arm}.jsonl"
        rows = unique_by(iter_jsonl(path), "case_key")
        if set(rows) != set(tasks):
            raise ValueError(f"{arm} action coverage differs from frozen task universe")
        for case_key, row in rows.items():
            if set(row) != PREDICTION_FIELDS or row.get("arm") != arm:
                raise ValueError(f"{arm} prediction field contract violation")
            task = tasks[case_key]
            if row.get("example_id") != task.get("example_id") or row.get("mode") != task.get("mode"):
                raise ValueError(f"{arm} prediction identity differs from public task")
            if row.get("calls") != 1:
                raise ValueError(f"{arm} action budget differs from one call per case")
        rows_by_arm[arm] = rows
        outputs[arm] = {"path": str(path.relative_to(root)), "sha256": sha256_file(path), "case_count": len(rows)}
    for key in tasks:
        baseline, candidate = rows_by_arm["baseline"][key], rows_by_arm["candidate"][key]
        for field in ("example_id", "mode", "model_snapshot", "seed", "temperature", "max_tokens", "calls", "schema_hash"):
            if baseline[field] != candidate[field]:
                paired = False
    if not paired:
        raise ValueError("paired arms differ outside their memory block")
    receipt = {
        "schema_version": 1,
        "kind": "fresh_holdout_pre_gold_receipt_v1",
        "implementation_manifest_sha256": implementation["implementation_manifest_sha256"],
        "task_sha256": sha256_file(root / "artifacts/tasks.jsonl"),
        "memory_sha256": sha256_file(root / "artifacts/memory.prefine_anchor.jsonl"),
        "outputs": outputs,
        "equal_action_budget": True,
        "gold_opened": False,
    }
    write_json_once(root / "artifacts/pre_gold_receipt.json", receipt)
    return receipt


def _write_failure(root: Path, implementation: dict[str, Any] | None, exc: BaseException) -> None:
    path = root / "reports/final_result.json"
    if path.exists() or path.is_symlink():
        return
    payload = {
        "execution_status": "FAILED",
        "integrity_status": "FAIL",
        "performance_status": "NOT_RUN",
        "implementation_manifest_sha256": (
            implementation.get("implementation_manifest_sha256") if implementation else None
        ),
        "detail": f"final_pipeline_failed:{type(exc).__name__}",
    }
    write_json_once(path, payload)


def preflight(root: Path) -> dict[str, Any]:
    root = root.resolve()
    implementation, preregistration = _runtime_contract(root)
    gpu = _gpu_preflight()
    label = f"preflight-{uuid.uuid4()}"
    process, log_path = _start_server(root, preregistration, label)
    try:
        return {
            "implementation_manifest_sha256": implementation["implementation_manifest_sha256"],
            "gpu_indices": list(GPU_INDICES),
            "gpu_count": len(gpu["gpus"]),
            "vllm_log": str(log_path.relative_to(root)),
        }
    finally:
        _stop_server(process)


def final(root: Path) -> dict[str, Any]:
    root = root.resolve()
    implementation: dict[str, Any] | None = None
    process: subprocess.Popen[str] | None = None
    attempt_committed = False
    try:
        _assert_final_outputs_absent(root)
        implementation, preregistration = _runtime_contract(root)
        _gpu_preflight()
        attempt = {
            "schema_version": 1,
            "kind": "fresh_holdout_single_final_attempt_v1",
            "attempt_id": str(uuid.uuid4()),
            "implementation_manifest_sha256": implementation["implementation_manifest_sha256"],
            "gpu_indices": list(GPU_INDICES),
            "acquired_unix_ns": time.time_ns(),
        }
        write_json_once(root / "artifacts/final_attempt.json", attempt)
        attempt_committed = True
        # Recheck after acquiring the one-shot lock and before starting any model.
        implementation = _runtime_contract(root)[0]
        process, _log_path = _start_server(root, preregistration, attempt["attempt_id"])
        base_url = f"http://{HOST}:{PORT}"
        commands = [
            _run_command(root, ["holdout_runner.py", "build-memory", "--base-url", base_url], 43200),
            _run_command(root, ["holdout_runner.py", "infer", "--arm", "baseline", "--base-url", base_url], 43200),
            _run_command(root, ["holdout_runner.py", "infer", "--arm", "candidate", "--base-url", base_url], 43200),
        ]
        _stop_server(process)
        process = None
        receipt = _pre_gold_receipt(root, implementation)
        commands.append(_run_command(root, ["holdout_evaluator.py"], 3600))
        summary = load_json(root / "reports/summary.json")
        result = {
            "execution_status": "COMPLETE",
            "integrity_status": "PASS",
            "performance_status": "PASS" if summary.get("pass") is True else "FAIL",
            "implementation_manifest_sha256": implementation["implementation_manifest_sha256"],
            "pre_gold_receipt_sha256": sha256_file(root / "artifacts/pre_gold_receipt.json"),
            "summary_sha256": sha256_file(root / "reports/summary.json"),
            "evaluation_evidence_sha256": sha256_file(root / "reports/evaluation_evidence.json"),
            "command_receipts": commands,
            "detail": "paired_one_call_per_case_then_single_open_dual_evaluator",
        }
        write_json_once(root / "reports/final_result.json", result)
        commands.append(_run_command(root, ["critic.py"], 3600))
        return {
            "execution_status": result["execution_status"],
            "integrity_status": result["integrity_status"],
            "performance_status": result["performance_status"],
            "case_count": receipt["outputs"]["baseline"]["case_count"],
            "summary_sha256": result["summary_sha256"],
        }
    except BaseException as exc:
        if attempt_committed:
            _write_failure(root, implementation, exc)
        raise
    finally:
        if process is not None:
            _stop_server(process)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="four-GPU one-shot fresh-holdout runner")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--final", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    try:
        outcome = preflight(args.root) if args.preflight_only else final(args.root)
        print(json.dumps(outcome, ensure_ascii=False, sort_keys=True))
        return 0
    except BaseException as exc:
        print(json.dumps({"status": "FAILED", "error_class": type(exc).__name__}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
