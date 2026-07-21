#!/usr/bin/env python3
"""One-shot TP=4 final runner with anonymous-pipe capabilities."""

from __future__ import annotations

import argparse
import json
import os
import signal
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable

from ecpr.final_protocol import (
    EVALUATION_EVIDENCE,
    acquire_final_attempt_v2,
    assert_no_stage_artifacts,
    classify_protocol_state,
    command_sha256,
    commit_stage_once,
    environment_receipt,
    hash_tree,
    open_regular_bytes_once,
    parse_jsonl_bytes,
    parse_json_object_bytes,
    process_identity_is_live,
    process_start_identity,
    sanitized_child_environment,
    strict_success,
    validate_final_attempt_v2,
    validate_stage_chain,
)
from ecpr.integrity import (
    ATTEMPT_LOCK,
    build_attempt_binding,
    validate_implementation_manifest,
)
from ecpr.io import canonical_json, load_json, sha256_bytes, sha256_file
from ecpr.ledger import (
    PROVIDER_JOURNAL,
    ProviderJournal,
    assert_final_artifacts_absent,
    final_paths,
    validate_journal_records,
    validate_run_dag,
)


ROOT = Path(__file__).resolve().parent
ATTEMPT_RELATIVE_PATH = ATTEMPT_LOCK
FD_MARKER = "<ANONYMOUS_FD>"
FINAL_COMMAND_TIMEOUT_SECONDS = {
    "memory": 172800,
    "baseline": 43200,
    "candidate": 43200,
    "evaluate": 1800,
    "critic": 1800,
}


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_replace_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    data = (canonical_json(value) + "\n").encode("utf-8")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_blocker(root: Path, reason: str, evidence: str) -> None:
    _atomic_replace_json(
        root / "reports/GPU_BLOCKER.json",
        {
            "status": "blocked_no_metrics",
            "reason": reason,
            "evidence": evidence[-2000:],
        },
    )


def gpu_preflight(contract: dict[str, Any]) -> tuple[bool, str]:
    runtime = contract["runtime"]
    try:
        result = subprocess.run(
            list(runtime["gpu_preflight_command"]),
            capture_output=True,
            text=True,
            timeout=int(runtime["gpu_preflight_subprocess_timeout_seconds"]),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    evidence = (result.stdout + result.stderr).strip()
    if result.returncode != 0:
        return False, evidence or f"nvidia-smi exited {result.returncode}"
    indices = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    required = {str(index) for index in runtime["gpu_indices"]}
    if not required.issubset(indices):
        return False, f"required GPU indices unavailable; visible={sorted(indices)}"
    return True, evidence


def runtime_precheck(_root: Path, contract: dict[str, Any]) -> tuple[bool, str]:
    model_path = Path(contract["model"]["snapshot_path"])
    vllm_path = Path(contract["runtime"]["vllm_executable"])
    if not model_path.is_dir() or not vllm_path.is_file():
        return False, f"model={model_path.is_dir()} vllm={vllm_path.is_file()}"
    return True, "local model snapshot and vLLM executable present"


def wait_server(
    process: subprocess.Popen[str],
    base_url: str,
    timeout_seconds: int = 600,
) -> bytes | None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return None
        try:
            with urllib.request.urlopen(
                base_url.rstrip("/") + "/v1/models", timeout=2
            ) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2)
    return None


def _vllm_argv(contract: dict[str, Any]) -> list[str]:
    runtime = contract["runtime"]
    model = contract["model"]
    return [
        str(runtime["vllm_executable"]),
        "serve",
        str(model["snapshot_path"]),
        "--served-model-name",
        str(model["served_model_name"]),
        "--tensor-parallel-size",
        str(model["tensor_parallel_size"]),
        "--host",
        str(runtime["host"]),
        "--port",
        str(runtime["port"]),
        "--dtype",
        "auto",
    ]


def final_pipeline_commands(
    root: Path, contract: dict[str, Any], attempt_id: str | None = None
) -> list[tuple[tuple[str, ...], bool]]:
    base_url = str(contract["pipeline"]["base_url"])
    outputs = contract["pipeline"]["outputs"]
    final = ("--final-attempt-id", attempt_id) if attempt_id else ()
    return [
        (("build-memory", "--base-url", base_url, *final), True),
        (
            (
                "infer",
                "--arm",
                "baseline",
                "--base-url",
                base_url,
                "--output",
                str(root / outputs["baseline"]),
                *final,
            ),
            True,
        ),
        (
            (
                "infer",
                "--arm",
                "candidate",
                "--base-url",
                base_url,
                "--output",
                str(root / outputs["candidate"]),
                *final,
            ),
            True,
        ),
        (
            (
                "evaluate",
                "--baseline",
                str(root / outputs["baseline"]),
                "--candidate",
                str(root / outputs["candidate"]),
                "--output",
                str(root / outputs["summary"]),
                *final,
            ),
            False,
        ),
    ]


def _normalized_cli_argv(arguments: tuple[str, ...], *, gold: bool) -> list[str]:
    argv = [
        sys.executable,
        "-m",
        "ecpr",
        *arguments,
        "--runner-nonce-fd",
        FD_MARKER,
    ]
    if gold:
        argv.extend(("--gold-open-fd", FD_MARKER))
    return argv


def planned_command_receipts(
    root: Path, contract: dict[str, Any], attempt_id: str
) -> list[dict[str, Any]]:
    names = ("memory", "baseline", "candidate", "evaluate")
    commands = final_pipeline_commands(root, contract, attempt_id)
    receipts = [
        {
            "name": name,
            "argv_sha256": command_sha256(
                _normalized_cli_argv(arguments, gold=name == "evaluate")
            ),
            "timeout_seconds": FINAL_COMMAND_TIMEOUT_SECONDS[name],
        }
        for name, (arguments, _check) in zip(names, commands)
    ]
    critic_argv = _normalized_critic_argv(root, attempt_id)
    receipts.append(
        {
            "name": "critic",
            "argv_sha256": command_sha256(critic_argv),
            "timeout_seconds": FINAL_COMMAND_TIMEOUT_SECONDS["critic"],
        }
    )
    return receipts


def gvr_call_graph_receipt(root: Path) -> dict[str, Any]:
    """Bind the sole live/replay PReFine transition graph and prompt templates."""
    from ecpr.prefine import (
        LATENT_GENERATION_SCHEMA_SHA256,
        LATENT_VERIFICATION_SCHEMA_SHA256,
    )
    from ecpr.prompts import LATENT_INITIAL, LATENT_REFINE, LATENT_SYSTEM, LATENT_VERIFY

    digest = lambda value: sha256_bytes(value.encode("utf-8"))
    return {
        "state_machine_source_sha256": sha256_file(root / "ecpr/prefine.py"),
        "max_refinements_per_session": 10,
        "sequence": ["draft", "verify", "refine_if_invalid"],
        "generation": {
            "temperature": 0.4,
            "max_tokens": 2048,
            "json_object": True,
            "provider_retries": 0,
            "seed_formula": "base_seed+session_index*100+attempt_index*2",
            "schema_sha256": LATENT_GENERATION_SCHEMA_SHA256,
            "system_prompt_template_sha256": digest(LATENT_SYSTEM),
            "initial_prompt_template_sha256": digest(LATENT_INITIAL),
            "refine_prompt_template_sha256": digest(LATENT_REFINE),
        },
        "verification": {
            "temperature": 0.0,
            "max_tokens": 512,
            "json_object": True,
            "provider_retries": 0,
            "seed_formula": "base_seed+session_index*100+attempt_index*2+1",
            "schema_sha256": LATENT_VERIFICATION_SCHEMA_SHA256,
            "system_prompt_sha256": digest(
                "You are a Preference Verification Module. Output JSON only."
            ),
            "prompt_template_sha256": digest(LATENT_VERIFY),
        },
        "transitions": {
            "draft_to_verifier": "exact_parsed_draft_only",
            "invalid_verifier_to_refine": "exact_feedback_and_previous_draft",
            "valid_verifier_stop": "valid_is_exactly_true",
            "retry_exhaustion_stop": "ten_attempts_then_invalid_attestation",
            "accepted_memory_to_next_session": "previous_memory_only",
            "accepted_memory_to_action": "sealed_memory_manifest_only",
            "semantic_replay": "same_run_prefine_state_machine_and_exact_journal_slice",
        },
        "held_out_action_feedback": "forbidden_no_outgoing_provider_or_memory_edges",
    }


def _secret_pipe(secret: bytes | bytearray) -> int:
    if len(secret) != 32:
        raise ValueError("anonymous-pipe capability must be 32 bytes")
    read_fd, write_fd = os.pipe2(getattr(os, "O_CLOEXEC", 0))
    try:
        if os.write(write_fd, secret) != len(secret):
            raise OSError("short anonymous-pipe capability write")
    except BaseException:
        os.close(read_fd)
        raise
    finally:
        os.close(write_fd)
    return read_fd


def run_cli_command(
    root: Path,
    arguments: tuple[str, ...],
    environment: dict[str, str],
    runner_nonce: bytes | bytearray,
    *,
    expected_sha256: str,
    timeout_seconds: int,
    gold_secret: bytes | bytearray | None = None,
) -> int:
    normalized = _normalized_cli_argv(arguments, gold=gold_secret is not None)
    if command_sha256(normalized) != expected_sha256:
        raise ValueError("actual final command differs from the runtime plan")
    runner_fd = _secret_pipe(runner_nonce)
    pass_fds = [runner_fd]
    argv = [sys.executable, "-m", "ecpr", *arguments, "--runner-nonce-fd", str(runner_fd)]
    gold_fd = None
    if gold_secret is not None:
        gold_fd = _secret_pipe(gold_secret)
        pass_fds.append(gold_fd)
        argv.extend(("--gold-open-fd", str(gold_fd)))
    try:
        return subprocess.run(
            argv,
            cwd=root,
            env=environment,
            pass_fds=tuple(pass_fds),
            check=False,
            timeout=timeout_seconds,
        ).returncode
    finally:
        for descriptor in pass_fds:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _normalized_critic_argv(root: Path, attempt_id: str) -> list[str]:
    return [
        sys.executable,
        str((root / "critic.py").resolve()),
        "--strict",
        "--commit-final-receipt",
        "--final-attempt-id",
        attempt_id,
        "--runner-nonce-fd",
        FD_MARKER,
    ]


def run_critic_command(
    root: Path,
    attempt_id: str,
    environment: dict[str, str],
    runner_nonce: bytes | bytearray,
    *,
    expected_sha256: str,
    timeout_seconds: int,
) -> int:
    normalized = _normalized_critic_argv(root, attempt_id)
    if command_sha256(normalized) != expected_sha256:
        raise ValueError("actual critic command differs from the runtime plan")
    runner_fd = _secret_pipe(runner_nonce)
    argv = [
        value if value != FD_MARKER else str(runner_fd)
        for value in normalized
    ]
    try:
        return subprocess.run(
            argv,
            cwd=root,
            env=environment,
            pass_fds=(runner_fd,),
            check=False,
            timeout=timeout_seconds,
        ).returncode
    finally:
        try:
            os.close(runner_fd)
        except OSError:
            pass


def _capture_text(argv: list[str], environment: dict[str, str]) -> str:
    result = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"runtime identity command failed: {argv[0]}")
    return (result.stdout + result.stderr).strip()


def _gpu_identities(environment: dict[str, str]) -> list[dict[str, Any]]:
    output = _capture_text(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id",
            "--format=csv,noheader,nounits",
        ],
        environment,
    )
    identities: dict[int, dict[str, Any]] = {}
    for line in output.splitlines():
        pieces = [piece.strip() for piece in line.split(",")]
        if len(pieces) != 3:
            raise ValueError("malformed nvidia-smi identity row")
        index = int(pieces[0])
        if index in identities:
            raise ValueError("duplicate nvidia-smi GPU identity row")
        identities[index] = {
            "index": index,
            "uuid": pieces[1],
            "pci_bus_id": pieces[2],
        }
    try:
        visible = [
            int(value)
            for value in environment["CUDA_VISIBLE_DEVICES"].split(",")
        ]
    except (KeyError, ValueError) as exc:
        raise ValueError("CUDA_VISIBLE_DEVICES must contain physical GPU indices") from exc
    if len(visible) != 4 or len(set(visible)) != 4 or any(
        index not in identities for index in visible
    ):
        raise ValueError("sealed four-GPU identity set is unavailable")
    return [identities[index] for index in visible]


def _proc_parent_and_group(pid: int) -> tuple[int, int]:
    text = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
    close = text.rfind(") ")
    if close < 0:
        raise ValueError("malformed process stat")
    tail = text[close + 2 :].split()
    if len(tail) < 3:
        raise ValueError("truncated process stat")
    return int(tail[1]), int(tail[2])


def descendant_process_identities(root_pid: int) -> list[dict[str, Any]]:
    children: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            parent, _group = _proc_parent_and_group(pid)
        except (OSError, ValueError):
            continue
        children.setdefault(parent, []).append(pid)
    pending = [root_pid]
    seen: set[int] = set()
    identities: list[dict[str, Any]] = []
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        pending.extend(children.get(pid, ()))
        try:
            identities.append(process_start_identity(pid))
        except (OSError, ValueError):
            if pid == root_pid:
                raise
    return sorted(identities, key=lambda value: (value["pid"], value["start_ticks"]))


def process_group_pids(process_group: int) -> list[int]:
    """Enumerate a process group using /proc stat only, including zombies."""
    members: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            _parent, group = _proc_parent_and_group(pid)
        except (OSError, ValueError):
            continue
        if group == process_group:
            members.append(pid)
    return sorted(members)


def merge_process_identities(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = {
        (value["pid"], value["start_ticks"]): value
        for group in groups
        for value in group
    }
    return [merged[key] for key in sorted(merged)]


def capture_server_topology(
    process: subprocess.Popen[str],
    environment: dict[str, str],
    gpus: list[dict[str, Any]],
) -> dict[str, Any]:
    process_group = os.getpgid(process.pid)
    descendants = descendant_process_identities(process.pid)
    descendant_pids = {identity["pid"] for identity in descendants}
    output = _capture_text(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
        environment,
    )
    mappings: list[dict[str, Any]] = []
    for line in output.splitlines():
        pieces = [piece.strip() for piece in line.split(",")]
        if len(pieces) != 2 or not pieces[0].isdigit():
            continue
        pid = int(pieces[0])
        if pid in descendant_pids:
            mappings.append({"pid": pid, "gpu_uuid": pieces[1]})
    mappings.sort(key=lambda value: (value["pid"], value["gpu_uuid"]))
    expected_uuids = {gpu["uuid"] for gpu in gpus}
    mapped_uuids = {mapping["gpu_uuid"] for mapping in mappings}
    if len(expected_uuids) != 4 or mapped_uuids != expected_uuids:
        raise ValueError("vLLM descendants do not map to all four sealed GPU UUIDs")
    return {
        "process_group_id": process_group,
        "descendant_processes": descendants,
        "compute_app_mappings": mappings,
        "mapped_gpu_uuids": sorted(mapped_uuids),
    }


def build_runtime_payload(
    root: Path,
    contract: dict[str, Any],
    process: subprocess.Popen[str],
    models_response: bytes,
    environment: dict[str, str],
    planned_commands: list[dict[str, Any]],
) -> dict[str, Any]:
    vllm_argv = _vllm_argv(contract)
    model_path = Path(contract["model"]["snapshot_path"])
    runtime = contract["runtime"]
    try:
        parsed_models = json.loads(models_response.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("/v1/models returned invalid JSON") from exc
    gpus = _gpu_identities(environment)
    payload = {
        "python": {
            "executable": str(Path(sys.executable).resolve()),
            "sha256": sha256_file(sys.executable),
            "version": sys.version,
            "argv": list(sys.argv),
            "argv_sha256": command_sha256(list(sys.argv)),
        },
        "vllm": {
            "executable": str(Path(runtime["vllm_executable"]).resolve()),
            "sha256": sha256_file(runtime["vllm_executable"]),
            "version": _capture_text([str(runtime["vllm_executable"]), "--version"], environment),
            "argv": vllm_argv,
            "argv_sha256": command_sha256(vllm_argv),
            "process": process_start_identity(process.pid),
            "topology_at_readiness": capture_server_topology(
                process, environment, gpus
            ),
        },
        "model": {
            "snapshot_path": str(model_path.resolve()),
            "tree_sha256": hash_tree(model_path),
        },
        "gpus": gpus,
        "server": {
            "base_url": str(contract["pipeline"]["base_url"]),
            "models_response": parsed_models,
            "models_response_raw_sha256": sha256_bytes(models_response),
        },
        "environment": environment_receipt(environment),
        "runner": process_start_identity(os.getpid()),
        "planned_commands": planned_commands,
        "gvr_call_graph": gvr_call_graph_receipt(root),
    }
    validate_runtime_payload(
        payload,
        root,
        contract,
        process,
        environment,
        expected_planned_commands=planned_commands,
        rehash_tree=False,
    )
    return payload


def validate_runtime_payload(
    payload: dict[str, Any],
    root: Path,
    contract: dict[str, Any],
    process: subprocess.Popen[str],
    environment: dict[str, str],
    *,
    expected_planned_commands: list[dict[str, Any]],
    rehash_tree: bool,
) -> None:
    if set(payload) != {
        "python",
        "vllm",
        "model",
        "gpus",
        "server",
        "environment",
        "runner",
        "planned_commands",
        "gvr_call_graph",
    }:
        raise ValueError("runtime receipt field contract violation")
    runtime = contract["runtime"]
    model = contract["model"]
    if payload["python"]["executable"] != str(Path(sys.executable).resolve()):
        raise ValueError("wrong Python executable")
    if payload["python"]["sha256"] != sha256_file(sys.executable):
        raise ValueError("Python executable changed")
    if payload["vllm"]["executable"] != str(Path(runtime["vllm_executable"]).resolve()):
        raise ValueError("wrong vLLM executable")
    if payload["vllm"]["sha256"] != sha256_file(runtime["vllm_executable"]):
        raise ValueError("vLLM executable changed")
    if payload["vllm"]["argv"] != _vllm_argv(contract):
        raise ValueError("wrong vLLM argv")
    if payload["vllm"]["process"] != process_start_identity(process.pid):
        raise ValueError("vLLM PID/start identity changed")
    expected_indices = [int(index) for index in runtime["gpu_indices"]]
    current_gpus = _gpu_identities(environment)
    if payload["gpus"] != current_gpus:
        raise ValueError("GPU UUID/PCI identity changed")
    if [value["index"] for value in payload["gpus"]] != expected_indices:
        raise ValueError("wrong GPU index set or ordering")
    if len({value["uuid"] for value in payload["gpus"]}) != 4:
        raise ValueError("GPU UUIDs are not four-way unique")
    topology = payload["vllm"]["topology_at_readiness"]
    if topology != capture_server_topology(process, environment, payload["gpus"]):
        raise ValueError("vLLM descendant/GPU topology changed before runtime validation")
    if payload["server"]["base_url"] != contract["pipeline"]["base_url"]:
        raise ValueError("wrong runtime base URL")
    if payload["environment"] != environment_receipt(environment):
        raise ValueError("runtime child environment changed")
    if payload["runner"] != process_start_identity(os.getpid()):
        raise ValueError("runner PID/start identity changed")
    if payload["model"]["snapshot_path"] != str(Path(model["snapshot_path"]).resolve()):
        raise ValueError("wrong model snapshot path")
    if rehash_tree and payload["model"]["tree_sha256"] != hash_tree(Path(model["snapshot_path"])):
        raise ValueError("model snapshot tree changed during the final run")
    if payload["gvr_call_graph"] != gvr_call_graph_receipt(root):
        raise ValueError("GVR call graph changed during the final run")
    if payload["planned_commands"] != expected_planned_commands:
        raise ValueError("planned final commands or timeouts changed")


def stop_exact_server(
    process: subprocess.Popen[str],
    identity: dict[str, Any],
    recorded_processes: list[dict[str, Any]],
    process_group: int,
) -> dict[str, Any]:
    if process.poll() is None:
        try:
            current_identity = process_start_identity(process.pid)
        except (OSError, ValueError):
            current_identity = None
        if current_identity is not None and current_identity != identity:
            raise RuntimeError("refusing to stop a PID with the wrong start identity")
        process.poll()
    if process.poll() is None or process_group_pids(process_group):
        try:
            os.killpg(process_group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 20.0
    kill_sent = False
    while time.monotonic() < deadline:
        process.poll()
        alive = [
            value for value in recorded_processes if process_identity_is_live(value)
        ]
        remaining_pids = process_group_pids(process_group)
        if process.poll() is not None and not alive and not remaining_pids:
            break
        if not kill_sent and time.monotonic() >= deadline - 10.0:
            try:
                os.killpg(process_group, signal.SIGKILL)
            except ProcessLookupError:
                pass
            kill_sent = True
        time.sleep(0.1)
    if process.poll() is None:
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("sealed vLLM leader did not exit") from exc
    alive = [value for value in recorded_processes if process_identity_is_live(value)]
    remaining_pids = process_group_pids(process_group)
    if alive or remaining_pids:
        raise RuntimeError("sealed vLLM descendant/process group remains live")
    return {
        "process": identity,
        "process_group_id": process_group,
        "recorded_processes": recorded_processes,
        "return_code": int(process.returncode),
        "all_recorded_processes_dead": True,
        "process_group_empty": True,
    }


def _terminate_unreceipted_server(
    process: subprocess.Popen[str], process_group: int
) -> None:
    """Best-effort cleanup when identity capture fails before any runtime receipt."""
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
    if process_group_pids(process_group):
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 5.0
        while process_group_pids(process_group) and time.monotonic() < deadline:
            time.sleep(0.1)
        if process_group_pids(process_group):
            raise RuntimeError("unreceipted vLLM process group did not exit")


def _official_output_receipts(root: Path) -> dict[str, dict[str, Any]]:
    receipts: dict[str, dict[str, Any]] = {}
    for name, path in final_paths(root).items():
        if name == "summary":
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"pre_gold output is missing: {name}")
        receipts[name] = {
            "path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
        }
    return receipts


def seal_provider_journal(root: Path, attempt_id: str) -> dict[str, Any]:
    """Validate, fsync, and make the complete provider ledger read-only."""
    path = root / PROVIDER_JOURNAL
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        identity = os.fstat(descriptor)
        if not stat.S_ISREG(identity.st_mode):
            raise ValueError("provider journal must be a regular file")
        os.fsync(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        records = validate_journal_records(
            parse_jsonl_bytes(payload, "provider journal"),
            attempt_id,
        )
        if not records:
            raise ValueError("provider journal cannot be sealed empty")
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        sealed_identity = os.fstat(descriptor)
        if stat.S_IMODE(sealed_identity.st_mode) != 0o400:
            raise ValueError("provider journal did not become read-only")
        path_identity = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(path_identity.st_mode)
            or (path_identity.st_dev, path_identity.st_ino)
            != (sealed_identity.st_dev, sealed_identity.st_ino)
        ):
            raise ValueError("provider journal path changed during same-FD sealing")
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    return {
        "path": str(path.relative_to(root)),
        "sha256": sha256_bytes(payload),
        "device": int(sealed_identity.st_dev),
        "inode": int(sealed_identity.st_ino),
        "size": int(sealed_identity.st_size),
        "record_count": len(records),
        "final_sequence": int(records[-1]["sequence"]),
        "final_record_sha256": str(records[-1]["record_sha256"]),
        "mode": "0400",
    }


def _failed_result_payload(exc: BaseException, last_stage: str) -> dict[str, Any]:
    return {
        "execution_status": "FAILED",
        "integrity_status": "FAIL",
        "performance_status": "NOT_RUN",
        "summary": None,
        "evaluation_evidence": None,
        "detail": (
            f"stage={last_stage};error_class={type(exc).__name__};"
            "code=FINAL_PIPELINE_ABORTED_NO_RETRY"
        ),
    }


def execute_final_pipeline(
    root: Path,
    expected_attempt: dict[str, Any],
    runner_nonce: bytes | bytearray,
) -> int:
    attempt, manifest, _manifest_digest = validate_final_attempt_v2(root)
    if attempt != expected_attempt:
        raise RuntimeError("delegated pipeline attempt binding mismatch")
    contract = manifest["expected_contract"]
    runtime = contract["runtime"]
    base_url = str(contract["pipeline"]["base_url"])
    assert_final_artifacts_absent(root)
    assert_no_stage_artifacts(root)
    environment = sanitized_child_environment(root, runtime["gpu_indices"])
    commands = final_pipeline_commands(root, contract, attempt["attempt_id"])
    planned = planned_command_receipts(root, contract, attempt["attempt_id"])
    log_path = root / "logs" / f"vllm.{attempt['attempt_id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process: subprocess.Popen[str] | None = None
    stopped_receipt: dict[str, Any] | None = None
    recorded_processes: list[dict[str, Any]] = []
    process_group: int | None = None
    post_first_provider_topology: dict[str, Any] | None = None
    post_provider_topology: dict[str, Any] | None = None
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            _vllm_argv(contract),
            cwd=root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        server_identity: dict[str, Any] | None = None
        process_group = process.pid
        try:
            server_identity = process_start_identity(process.pid)
            recorded_processes = [server_identity]
            if os.getpgid(process.pid) != process_group:
                raise RuntimeError("vLLM did not become its own process-group leader")
            models_response = wait_server(
                process,
                base_url,
                int(runtime["server_start_timeout_seconds"]),
            )
            if models_response is None:
                raise RuntimeError("vLLM failed readiness before runtime receipt")
            runtime_payload = build_runtime_payload(
                root,
                contract,
                process,
                models_response,
                environment,
                planned,
            )
            commit_stage_once(
                root,
                attempt["attempt_id"],
                "runtime",
                runtime_payload,
                runner_nonce,
            )
            readiness_topology = runtime_payload["vllm"]["topology_at_readiness"]
            recorded_processes = merge_process_identities(
                recorded_processes,
                readiness_topology["descendant_processes"],
            )
            ProviderJournal(root / PROVIDER_JOURNAL, attempt["attempt_id"]).initialize_once()
            for index, ((arguments, check), receipt) in enumerate(zip(commands[:3], planned[:3])):
                return_code = run_cli_command(
                    root,
                    arguments,
                    environment,
                    runner_nonce,
                    expected_sha256=receipt["argv_sha256"],
                    timeout_seconds=receipt["timeout_seconds"],
                )
                if check and return_code != 0:
                    raise RuntimeError(f"final model command {index} failed with {return_code}")
                if index == 0:
                    post_first_provider_topology = capture_server_topology(
                        process, environment, runtime_payload["gpus"]
                    )
                    recorded_processes = merge_process_identities(
                        recorded_processes,
                        post_first_provider_topology["descendant_processes"],
                    )
            post_provider_topology = capture_server_topology(
                process, environment, runtime_payload["gpus"]
            )
            recorded_processes = merge_process_identities(
                recorded_processes,
                post_provider_topology["descendant_processes"],
            )
            dag = validate_run_dag(
                root,
                attempt["attempt_id"],
                root / contract["pipeline"]["outputs"]["baseline"],
                root / contract["pipeline"]["outputs"]["candidate"],
            )
            validate_runtime_payload(
                runtime_payload,
                root,
                contract,
                process,
                environment,
                expected_planned_commands=planned,
                rehash_tree=True,
            )
            journal_seal = seal_provider_journal(root, attempt["attempt_id"])
        finally:
            if process is not None and server_identity is not None:
                try:
                    recorded_processes = merge_process_identities(
                        recorded_processes,
                        descendant_process_identities(process.pid),
                    )
                except (OSError, ValueError):
                    pass
                stopped_receipt = stop_exact_server(
                    process,
                    server_identity,
                    recorded_processes,
                    process_group if process_group is not None else process.pid,
                )
            elif process is not None:
                _terminate_unreceipted_server(process, process_group)

    if stopped_receipt is None:
        raise AssertionError("missing exact server-stop receipt")
    if post_first_provider_topology is None or post_provider_topology is None:
        raise AssertionError("missing post-provider vLLM/GPU topology receipts")
    pre_gold_payload = {
        "run_dag": dag,
        "run_dag_sha256": sha256_bytes(canonical_json(dag).encode("utf-8")),
        "official_outputs": _official_output_receipts(root),
        "server_stopped": stopped_receipt,
        "runtime_revalidated_before_gold": True,
        "implementation_manifest_sha256": attempt["binding"][
            "implementation_manifest_sha256"
        ],
        "runtime_record_sha256": validate_stage_chain(root)["runtime"][
            "record_sha256"
        ],
        "provider_journal_seal": journal_seal,
        "post_first_provider_topology": post_first_provider_topology,
        "post_first_provider_topology_sha256": sha256_bytes(
            canonical_json(post_first_provider_topology).encode("utf-8")
        ),
        "post_provider_topology": post_provider_topology,
        "post_provider_topology_sha256": sha256_bytes(
            canonical_json(post_provider_topology).encode("utf-8")
        ),
    }
    commit_stage_once(
        root,
        attempt["attempt_id"],
        "pre_gold",
        pre_gold_payload,
        runner_nonce,
    )

    gold_secret = bytearray(os.urandom(32))
    try:
        evaluation_return = run_cli_command(
            root,
            commands[3][0],
            environment,
        runner_nonce,
        expected_sha256=planned[3]["argv_sha256"],
        timeout_seconds=planned[3]["timeout_seconds"],
        gold_secret=gold_secret,
        )
    finally:
        for index in range(len(gold_secret)):
            gold_secret[index] = 0

    summary_path = final_paths(root)["summary"]
    evidence_path = root / EVALUATION_EVIDENCE
    try:
        summary_bytes, _summary_identity = open_regular_bytes_once(summary_path)
        evidence_bytes, _evidence_identity = open_regular_bytes_once(evidence_path)
    except ValueError as exc:
        raise RuntimeError(
            f"evaluator failed before sealed summary/evidence commit: {evaluation_return}"
        ) from exc
    summary_sha256 = sha256_bytes(summary_bytes)
    evidence_sha256 = sha256_bytes(evidence_bytes)
    summary = parse_json_object_bytes(summary_bytes, "sealed evaluator summary")
    evidence = parse_json_object_bytes(
        evidence_bytes, "sealed evaluator evidence"
    )
    if evidence.get("registered_summary_sha256") != summary_sha256:
        raise ValueError("dual-audit evidence does not bind the summary")
    if evidence.get("agreement") is not True:
        raise ValueError("registered and independent evaluator did not agree")
    reported_pass = summary.get("pass")
    if not isinstance(reported_pass, bool) or evaluation_return != 0:
        raise ValueError("sealed evaluator did not complete cleanly")
    result_payload = {
        "execution_status": "COMPLETE",
        "integrity_status": "PASS",
        "performance_status": "PASS" if reported_pass else "FAIL",
        "summary": {
            "path": str(summary_path.relative_to(root)),
            "sha256": summary_sha256,
            "reported_pass": reported_pass,
        },
        "evaluation_evidence": {
            "path": str(evidence_path.relative_to(root)),
            "sha256": evidence_sha256,
        },
        "detail": "same_process_dual_metric_implementations_agree_shared_parser",
    }
    commit_stage_once(
        root,
        attempt["attempt_id"],
        "result",
        result_payload,
        runner_nonce,
    )
    critic_return = run_critic_command(
        root,
        attempt["attempt_id"],
        environment,
        runner_nonce,
        expected_sha256=planned[4]["argv_sha256"],
        timeout_seconds=planned[4]["timeout_seconds"],
    )
    stages = validate_stage_chain(root)
    critic_payload = stages.get("critic", {}).get("payload", {})
    if critic_payload.get("audit_status") != "PASS":
        raise RuntimeError("strict critic did not commit a PASS receipt")
    expected_critic_return = 0 if reported_pass else 1
    if critic_return != expected_critic_return:
        raise RuntimeError("strict critic exit differs from the sealed performance axis")
    axes = classify_protocol_state(root)
    return 0 if strict_success(axes) else 1


def main(
    argv: list[str] | None = None,
    *,
    root: Path = ROOT,
    preflight_fn: Callable[[dict[str, Any]], tuple[bool, str]] = gpu_preflight,
    runtime_check_fn: Callable[[Path, dict[str, Any]], tuple[bool, str]] = runtime_precheck,
    seal_validate_fn: Callable[[Path], tuple[dict[str, Any], str]] = validate_implementation_manifest,
    binding_fn: Callable[[Path], dict[str, Any]] = build_attempt_binding,
    acquire_fn: Callable[..., dict[str, Any]] = acquire_final_attempt_v2,
    attempt_validate_fn: Callable[[Path], tuple[dict[str, Any], dict[str, Any], str]] = validate_final_attempt_v2,
    executor_fn: Callable[..., int] = execute_final_pipeline,
    nonce_factory: Callable[[int], bytes] = os.urandom,
) -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--final", action="store_true", help="irrevocably consume the sole sealed final attempt after successful preflight")
    mode.add_argument("--preflight-only", action="store_true", help="bounded GPU/NVML check; never consumes the final attempt")
    args = parser.parse_args(argv)
    root = root.resolve()
    attempt_path = root / ATTEMPT_RELATIVE_PATH

    if args.final and (attempt_path.exists() or attempt_path.is_symlink()):
        write_blocker(root, "final_test_attempt_already_committed", str(attempt_path))
        return 2
    if args.final:
        try:
            assert_final_artifacts_absent(root)
            assert_no_stage_artifacts(root)
        except BaseException as exc:
            write_blocker(
                root,
                "orphan_final_artifact_before_attempt",
                f"{type(exc).__name__}: {exc}",
            )
            return 2
    try:
        first_manifest, first_digest = seal_validate_fn(root)
    except BaseException as exc:
        write_blocker(root, "immutable_static_seal_failed", f"{type(exc).__name__}: {exc}")
        return 2
    contract = first_manifest["expected_contract"]
    okay, evidence = preflight_fn(contract)
    if not okay:
        write_blocker(root, "gpu_preflight_failed", evidence)
        return 2
    runtime_okay, runtime_evidence = runtime_check_fn(root, contract)
    if not runtime_okay:
        write_blocker(root, "local_model_or_vllm_missing", runtime_evidence)
        return 2
    try:
        second_manifest, second_digest = seal_validate_fn(root)
        if second_digest != first_digest or second_manifest != first_manifest:
            raise ValueError("immutable static seal changed during preflight")
    except BaseException as exc:
        write_blocker(root, "post_preflight_static_seal_failed", f"{type(exc).__name__}: {exc}")
        return 2
    if args.preflight_only:
        return 0

    runner_nonce = bytearray(nonce_factory(32))
    if len(runner_nonce) != 32:
        write_blocker(root, "runner_nonce_generation_failed", "nonce factory did not return 32 bytes")
        return 2
    attempt: dict[str, Any] | None = None
    try:
        binding = binding_fn(root)
        attempt = acquire_fn(root, binding, runner_nonce)
        validated_attempt, _manifest, _digest = attempt_validate_fn(root)
        if validated_attempt != attempt:
            raise ValueError("committed attempt changed before launch")
        return executor_fn(root, attempt, runner_nonce)
    except FileExistsError:
        write_blocker(root, "final_test_attempt_race_lost", str(attempt_path))
        return 2
    except BaseException as exc:
        if attempt is not None:
            try:
                stages = validate_stage_chain(root)
                if "result" not in stages:
                    last_stage = next(reversed(stages), "attempt")
                    commit_stage_once(
                        root,
                        attempt["attempt_id"],
                        "result",
                        _failed_result_payload(exc, last_stage),
                        runner_nonce,
                    )
            except BaseException:
                pass
        return 1
    finally:
        for index in range(len(runner_nonce)):
            runner_nonce[index] = 0


if __name__ == "__main__":
    raise SystemExit(main())
