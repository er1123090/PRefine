"""One-shot, aggregate-only paired evaluator for ours_memory3.

The runner reads only public history/tasks/configuration until the baseline and
candidate prediction files and a pre-gold pairing receipt exist.  The vault is
then opened once, inside ``_evaluate_after_receipt``.  No case-level reference,
prediction, prompt, or model response is printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ours_memory3.contracts import OverlayPolicy, PublicInputError
from ours_memory3.overlay import build_overlay
from ours_memory3.parsing import extract_calls, safe_scalar
from ours_memory3.prefine import baseline_memory_block, build_prefine_latent
from ours_memory3.prompts import action_template_sha256, build_action_prompt


VLLM = Path("/home/minseo/miniconda3/bin/vllm")
GPU_INDICES = (0, 1, 2, 3)
HOST = "127.0.0.1"
PORT = 8137
TASK_FIELDS = frozenset({"case_key", "example_id", "mode", "query", "schema_key"})
PREDICTION_FIELDS = frozenset(
    {
        "case_key", "example_id", "mode", "arm", "llm_output", "status", "model", "seed",
        "temperature", "max_tokens", "calls", "prompt_hash", "prompt_template_hash",
        "schema_hash", "memory_hash", "usage",
    }
)


class EvaluatorError(RuntimeError):
    pass


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluatorError(f"invalid JSONL in {path.name} at line {number}") from exc
            if not isinstance(row, dict):
                raise EvaluatorError(f"non-object JSONL row in {path.name}")
            rows.append(row)
    if not rows:
        raise EvaluatorError(f"empty JSONL: {path.name}")
    return rows


def _write_json_once(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"output already exists: {path}")
    path.write_text(_canonical_json(value) + "\n", encoding="utf-8")


def _write_jsonl_once(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"output already exists: {path}")
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(row) + "\n")


def _unique(rows: Iterable[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(field)
        if not isinstance(value, str) or not value or value in result:
            raise EvaluatorError(f"invalid or duplicate {field}")
        result[value] = row
    return result


def _runtime(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, tuple[str, ...]]]:
    manifest = _load_json(root / "sealed_manifest.json")
    if not isinstance(manifest, dict) or manifest.get("kind") != "ours_memory3_1229_dev6_sealed_paired_evaluator_v1":
        raise EvaluatorError("sealed evaluator manifest is invalid")
    public = manifest.get("public")
    if not isinstance(public, dict):
        raise EvaluatorError("sealed public input manifest is invalid")
    expected = {
        "tasks": root / "public/tasks.jsonl",
        "history": root / "public/history.sanitized.jsonl",
        "preference_slots": root / "public/preference_slots.json",
        "schema_single": root / "public/schema_single.json",
        "schema_multi": root / "public/schema_multi.json",
        "domain_ontology": root / "public/public_domain_ontology.json",
    }
    for key, path in expected.items():
        digest = public.get(key)
        if not isinstance(digest, str) or not path.is_file() or _sha256_file(path) != digest:
            raise EvaluatorError(f"sealed public input mismatch: {key}")
    if not (root / "vault/gold.jsonl").is_file():
        raise EvaluatorError("sealed vault is missing")
    tasks = _read_jsonl(expected["tasks"])
    if any(frozenset(row) != TASK_FIELDS for row in tasks):
        raise EvaluatorError("public task field contract violation")
    task_by_key = _unique(tasks, "case_key")
    if any(row.get("mode") not in {"singleturn", "multiturn"} for row in tasks):
        raise EvaluatorError("public task mode contract violation")
    histories = _unique(_read_jsonl(expected["history"]), "example_id")
    if any(row.get("example_id") not in histories for row in tasks):
        raise EvaluatorError("public task references missing history")
    preference_slots = _load_json(expected["preference_slots"])
    if not isinstance(preference_slots, dict):
        raise EvaluatorError("preference slot configuration is invalid")
    normalized_slots = {
        str(domain): tuple(str(slot) for slot in slots)
        for domain, slots in preference_slots.items()
        if isinstance(domain, str) and isinstance(slots, list)
    }
    if len(normalized_slots) != len(preference_slots) or not normalized_slots:
        raise EvaluatorError("preference slot configuration is malformed")
    schemas = {"single": _load_json(expected["schema_single"]), "multi": _load_json(expected["schema_multi"])}
    if any(not isinstance(row.get("schema_key"), str) or row["schema_key"] not in schemas for row in tasks):
        raise EvaluatorError("public task schema key is invalid")
    return manifest, tasks, histories, schemas, _load_json(expected["domain_ontology"]), normalized_slots


def _domain_aliases(ontology: object) -> dict[str, tuple[str, ...]]:
    if not isinstance(ontology, Mapping) or not isinstance(ontology.get("domains"), list):
        return {}
    result: dict[str, tuple[str, ...]] = {}
    for entry in ontology["domains"]:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("schema_domain"), str):
            continue
        aliases = entry.get("aliases")
        texts = tuple(
            item["text"] for item in aliases
            if isinstance(item, Mapping) and isinstance(item.get("text"), str)
        ) if isinstance(aliases, list) else ()
        if texts:
            result[entry["schema_domain"]] = texts
    return result


class _Client:
    def __init__(self, base_url: str, model: str) -> None:
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise EvaluatorError("only an explicit loopback model endpoint is allowed")
        self.endpoint = base_url.rstrip("/") + "/v1/chat/completions"
        self.model = model

    def complete(self, messages: list[dict[str, str]], seed: int, max_tokens: int, temperature: float, json_object: bool) -> tuple[str, dict[str, int]]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "seed": int(seed),
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "n": 1,
            "stream": False,
        }
        if json_object:
            payload["response_format"] = {"type": "json_object"}
        request = urllib.request.Request(
            self.endpoint,
            data=_canonical_json(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content:
                raise EvaluatorError("model response content is empty")
            raw_usage = body.get("usage", {})
            usage = {name: int(raw_usage.get(name, 0) or 0) for name in ("prompt_tokens", "completion_tokens", "total_tokens")}
            return content, usage
        except (urllib.error.URLError, TimeoutError, UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise EvaluatorError("loopback provider request failed") from exc


def _seed(base_seed: int, identifier: str) -> int:
    return int(base_seed) + int(hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:8], 16)


def _assert_absent(root: Path) -> None:
    paths = (
        root / "artifacts/final_attempt.json", root / "artifacts/memories.jsonl",
        root / "artifacts/predictions.baseline.jsonl", root / "artifacts/predictions.candidate.jsonl",
        root / "artifacts/pre_gold_receipt.json", root / "reports/summary.json",
        root / "reports/evaluation_evidence.json", root / "reports/final_result.json",
    )
    existing = [path.name for path in paths if path.exists() or path.is_symlink()]
    if existing:
        raise FileExistsError("sealed final outputs already exist")


def _build_memories(root: Path, client: _Client, manifest: Mapping[str, Any], histories: Mapping[str, dict[str, Any]]) -> dict[str, str]:
    output_path = root / "artifacts/memories.jsonl"
    policy = manifest.get("prefine", {})
    maximum_attempts = int(policy.get("maximum_attempts", 10)) if isinstance(policy, Mapping) else 10
    inference = manifest["inference"]
    base_seed = int(inference["base_seed"])
    workers = int(inference["request_workers"])

    def build(example_id: str, history: dict[str, Any]) -> dict[str, Any]:
        latent = build_prefine_latent(
            history=history,
            complete=lambda messages, seed, max_tokens, temperature, json_object: client.complete(messages, seed, max_tokens, temperature, json_object)[0],
            seed=_seed(base_seed, "memory:" + example_id),
            maximum_attempts=maximum_attempts,
        )
        memory = baseline_memory_block(latent=latent, history=history)
        return {"example_id": example_id, "memory": memory, "memory_hash": _sha256_text(memory)}

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(build, example_id, history): example_id for example_id, history in histories.items()}
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: row["example_id"])
    _write_jsonl_once(output_path, results)
    return {row["example_id"]: row["memory"] for row in results}


def _infer_arm(
    root: Path,
    *,
    arm: str,
    client: _Client,
    manifest: Mapping[str, Any],
    tasks: list[dict[str, Any]],
    histories: Mapping[str, dict[str, Any]],
    schemas: Mapping[str, Any],
    slots: Mapping[str, tuple[str, ...]],
    aliases: Mapping[str, tuple[str, ...]],
    memories: Mapping[str, str],
) -> None:
    if arm not in {"baseline", "candidate"}:
        raise EvaluatorError("unknown paired arm")
    inference = manifest["inference"]
    policy = OverlayPolicy(**manifest["overlay_policy"])
    workers = int(inference["request_workers"])
    base_seed = int(inference["base_seed"])
    temperature = float(inference["temperature"])
    max_tokens = int(inference["max_tokens"])
    model = str(manifest["model"]["served_model_name"])

    def infer(task: dict[str, Any]) -> dict[str, Any]:
        example_id = task["example_id"]
        baseline_memory = memories[example_id]
        schema = schemas[task["schema_key"]]
        candidate = build_overlay(
            baseline_memory=baseline_memory,
            history=histories[example_id],
            query=task["query"],
            mode=task["mode"],
            preference_slots=slots,
            schema=schema,
            domain_aliases=aliases,
            policy=policy,
        )
        memory = baseline_memory if arm == "baseline" else candidate.memory
        prompt = build_action_prompt(mode=task["mode"], schema=schema, memory=memory, query=task["query"])
        marker = "__OURS_MEMORY3_MEMORY_BOUNDARY__"
        skeleton = build_action_prompt(mode=task["mode"], schema=schema, memory=marker, query=task["query"])
        if prompt.replace(memory, marker, 1) != skeleton:
            raise EvaluatorError("action prompt differs outside memory boundary")
        output, usage = client.complete([{"role": "user", "content": prompt}], _seed(base_seed, "action:" + task["case_key"]), max_tokens, temperature, False)
        return {
            "case_key": task["case_key"], "example_id": example_id, "mode": task["mode"], "arm": arm,
            "llm_output": output, "status": "ok", "model": model,
            "seed": _seed(base_seed, "action:" + task["case_key"]), "temperature": temperature,
            "max_tokens": max_tokens, "calls": 1, "prompt_hash": _sha256_text(prompt),
            "prompt_template_hash": action_template_sha256(task["mode"]),
            "schema_hash": _sha256_text(_canonical_json(schema)), "memory_hash": _sha256_text(memory), "usage": usage,
        }

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(infer, task) for task in tasks]
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: row["case_key"])
    _write_jsonl_once(root / f"artifacts/predictions.{arm}.jsonl", rows)


def _pre_gold_receipt(root: Path, manifest: Mapping[str, Any], tasks: list[dict[str, Any]]) -> dict[str, Any]:
    task_by_key = _unique(tasks, "case_key")
    outputs: dict[str, dict[str, Any]] = {}
    paired_rows: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in ("baseline", "candidate"):
        path = root / f"artifacts/predictions.{arm}.jsonl"
        rows = _unique(_read_jsonl(path), "case_key")
        if set(rows) != set(task_by_key):
            raise EvaluatorError("prediction coverage differs from public task universe")
        for case_key, row in rows.items():
            if frozenset(row) != PREDICTION_FIELDS or row.get("arm") != arm or row.get("calls") != 1:
                raise EvaluatorError("prediction field or one-call contract violation")
            task = task_by_key[case_key]
            if row.get("example_id") != task["example_id"] or row.get("mode") != task["mode"]:
                raise EvaluatorError("prediction identity differs from public task")
        paired_rows[arm] = rows
        outputs[arm] = {"sha256": _sha256_file(path), "case_count": len(rows)}
    for case_key in task_by_key:
        baseline = paired_rows["baseline"][case_key]
        candidate = paired_rows["candidate"][case_key]
        for field in ("case_key", "example_id", "mode", "model", "seed", "temperature", "max_tokens", "calls", "prompt_template_hash", "schema_hash"):
            if baseline[field] != candidate[field]:
                raise EvaluatorError("paired arms differ outside the memory block")
    receipt = {
        "schema_version": 1, "kind": "ours_memory3_pre_gold_paired_receipt_v1",
        "task_sha256": _sha256_file(root / "public/tasks.jsonl"), "outputs": outputs,
        "equal_action_budget": True, "gold_opened": False,
    }
    _write_json_once(root / "artifacts/pre_gold_receipt.json", receipt)
    return receipt


_THINK_RE = re.compile(r"<think>.*?</think>", re.I | re.S)


def _call_identity(value: object) -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
    if isinstance(value, str):
        value = _THINK_RE.sub("", value)
    calls = extract_calls(value)
    rendered: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    for call in calls:
        name = call.get("name")
        arguments = call.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, Mapping):
            continue
        items: list[tuple[str, str]] = []
        for slot, raw in arguments.items():
            if isinstance(slot, str):
                scalar = safe_scalar(raw, 10000)
                items.append((slot.casefold(), scalar if scalar is not None else _canonical_json(raw)))
        rendered.append((name.casefold(), tuple(sorted(items))))
    return tuple(sorted(rendered))


def _evaluate_rows(gold: list[dict[str, Any]], baseline: list[dict[str, Any]], candidate: list[dict[str, Any]], minimum_delta: float) -> dict[str, Any]:
    gold_by = _unique(gold, "case_key")
    base_by = _unique(baseline, "case_key")
    cand_by = _unique(candidate, "case_key")
    if set(gold_by) != set(base_by) or set(gold_by) != set(cand_by):
        raise EvaluatorError("gold and prediction coverage mismatch")
    metrics: dict[str, dict[str, float | int]] = {}
    for mode in ("singleturn", "multiturn"):
        rows = [row for row in gold_by.values() if row.get("mode") == mode]
        if not rows:
            raise EvaluatorError("a required mode has no gold rows")
        base_correct = sum(_call_identity(base_by[row["case_key"]]["llm_output"]) == _call_identity(row.get("reference_ground_truth")) for row in rows)
        cand_correct = sum(_call_identity(cand_by[row["case_key"]]["llm_output"]) == _call_identity(row.get("reference_ground_truth")) for row in rows)
        base_rate = base_correct / len(rows)
        cand_rate = cand_correct / len(rows)
        metrics[mode] = {
            "case_count": len(rows), "baseline_correct": base_correct, "candidate_correct": cand_correct,
            "baseline_exact_match": round(base_rate, 8), "candidate_exact_match": round(cand_rate, 8),
            "delta_percentage_points": round((cand_rate - base_rate) * 100.0, 6),
        }
    passed = all(float(metrics[mode]["delta_percentage_points"]) >= minimum_delta for mode in metrics)
    return {"pass": passed, "minimum_each_mode_delta_percentage_points": minimum_delta, "metrics": metrics}


def _evaluate_after_receipt(root: Path, receipt: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    if receipt.get("gold_opened") is not False or receipt.get("equal_action_budget") is not True:
        raise EvaluatorError("invalid pre-gold receipt")
    # The sole read of private gold bytes occurs here, after both prediction files and receipt exist.
    gold_bytes = (root / "vault/gold.jsonl").read_bytes()
    gold_sha = hashlib.sha256(gold_bytes).hexdigest()
    if gold_sha != manifest.get("gold_sha256"):
        raise EvaluatorError("sealed gold hash mismatch")
    gold = [json.loads(line) for line in gold_bytes.decode("utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) and {"case_key", "example_id", "mode", "difficulty", "reference_ground_truth"}.issubset(row) for row in gold):
        raise EvaluatorError("sealed gold field contract violation")
    summary = _evaluate_rows(
        gold,
        _read_jsonl(root / "artifacts/predictions.baseline.jsonl"),
        _read_jsonl(root / "artifacts/predictions.candidate.jsonl"),
        float(manifest["pass"]["minimum_each_mode_delta_percentage_points"]),
    )
    _write_json_once(root / "reports/summary.json", summary)
    evidence = {
        "schema_version": 1, "kind": "ours_memory3_single_gold_open_evidence_v1",
        "gold_sha256": gold_sha, "pre_gold_receipt_sha256": _sha256_file(root / "artifacts/pre_gold_receipt.json"),
        "baseline_predictions_sha256": _sha256_file(root / "artifacts/predictions.baseline.jsonl"),
        "candidate_predictions_sha256": _sha256_file(root / "artifacts/predictions.candidate.jsonl"),
        "gold_opened_once": True, "aggregate_only": True,
    }
    _write_json_once(root / "reports/evaluation_evidence.json", evidence)
    return summary


def _gpu_preflight(manifest: Mapping[str, Any]) -> None:
    model = manifest.get("model")
    if not isinstance(model, Mapping) or tuple(model.get("gpu_indices", ())) != GPU_INDICES or int(model.get("tensor_parallel_size", 0)) != len(GPU_INDICES):
        raise EvaluatorError("four-GPU model contract mismatch")
    if not VLLM.is_file() or not Path(str(model.get("snapshot_path", ""))).is_dir():
        raise EvaluatorError("local vLLM executable or model snapshot is unavailable")
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], capture_output=True, text=True, timeout=10, check=False)
    seen = {int(line.split(",", 1)[0].strip()) for line in result.stdout.splitlines() if "," in line and line.split(",", 1)[0].strip().isdigit()}
    if result.returncode != 0 or not set(GPU_INDICES).issubset(seen):
        raise EvaluatorError("four-GPU nvidia-smi preflight failed")


def _start_server(root: Path, manifest: Mapping[str, Any]) -> tuple[subprocess.Popen[str], str]:
    model = manifest["model"]
    log_path = root / "logs/vllm.final.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("x", encoding="utf-8")
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in GPU_INDICES)
    environment["PYTHONHASHSEED"] = "0"
    command = [str(VLLM), "serve", str(model["snapshot_path"]), "--served-model-name", str(model["served_model_name"]), "--tensor-parallel-size", str(model["tensor_parallel_size"]), "--host", HOST, "--port", str(PORT), "--dtype", "auto"]
    try:
        process = subprocess.Popen(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    finally:
        log.close()
    endpoint = f"http://{HOST}:{PORT}"
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise EvaluatorError("vLLM exited before readiness")
        try:
            with urllib.request.urlopen(endpoint + "/v1/models", timeout=2) as response:
                if isinstance(json.loads(response.read().decode("utf-8")), dict):
                    return process, endpoint
        except (urllib.error.URLError, TimeoutError, UnicodeDecodeError, json.JSONDecodeError):
            time.sleep(2)
    raise EvaluatorError("vLLM readiness timeout")


def _stop_server(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=60)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def preflight(root: Path) -> dict[str, object]:
    manifest, tasks, _histories, _schemas, _ontology, _slots = _runtime(root.resolve())
    _gpu_preflight(manifest)
    return {"status": "PREFLIGHT_OK", "public_case_count": len(tasks), "gpu_indices": list(GPU_INDICES), "gold_opened": False}


def final(root: Path, base_url: str | None = None) -> dict[str, object]:
    root = root.resolve()
    manifest, tasks, histories, schemas, ontology, slots = _runtime(root)
    _assert_absent(root)
    process: subprocess.Popen[str] | None = None
    if base_url is None:
        _gpu_preflight(manifest)
    _write_json_once(root / "artifacts/final_attempt.json", {"schema_version": 1, "kind": "ours_memory3_one_shot_final_attempt_v1", "attempt_id": str(uuid.uuid4()), "acquired_unix_ns": time.time_ns()})
    try:
        endpoint = base_url
        if endpoint is None:
            process, endpoint = _start_server(root, manifest)
        client = _Client(endpoint, str(manifest["model"]["served_model_name"]))
        memories = _build_memories(root, client, manifest, histories)
        aliases = _domain_aliases(ontology)
        _infer_arm(root, arm="baseline", client=client, manifest=manifest, tasks=tasks, histories=histories, schemas=schemas, slots=slots, aliases=aliases, memories=memories)
        _infer_arm(root, arm="candidate", client=client, manifest=manifest, tasks=tasks, histories=histories, schemas=schemas, slots=slots, aliases=aliases, memories=memories)
        _stop_server(process)
        process = None
        receipt = _pre_gold_receipt(root, manifest, tasks)
        summary = _evaluate_after_receipt(root, receipt, manifest)
        result = {"status": "COMPLETE", "integrity": "PASS", "performance": "PASS" if summary["pass"] else "FAIL", "summary_sha256": _sha256_file(root / "reports/summary.json"), "evaluation_evidence_sha256": _sha256_file(root / "reports/evaluation_evidence.json")}
        _write_json_once(root / "reports/final_result.json", result)
        return {"status": result["status"], "integrity": result["integrity"], "performance": result["performance"], "metrics": summary["metrics"]}
    except BaseException:
        raise
    finally:
        _stop_server(process)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="sealed paired ours_memory3 evaluator")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--final", action="store_true")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base-url", help="existing explicit loopback endpoint; otherwise start local four-GPU vLLM")
    args = parser.parse_args(argv)
    try:
        result = preflight(args.root) if args.preflight_only else final(args.root, args.base_url)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except BaseException as exc:
        print(json.dumps({"status": "FAILED", "error_class": type(exc).__name__}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
