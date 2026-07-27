"""Run Experiment8 A-MEM retrieval and final prediction through OpenAI Batch."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.amem.common import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    atomic_write_json,
    atomic_write_jsonl,
    format_retrieved_notes,
    index_unique_rows,
    load_memory_artifact,
    make_embedder,
    now_iso,
    public_note,
    read_json,
    read_jsonl,
    retrieve_notes_from_embedding,
    sha256_file,
)
from scripts.run_vanilla_batch_sample import (  # noqa: E402
    build_population,
    evaluation_report,
    openai_result_payload,
)
from src.exp4_prompts import (  # noqa: E402
    IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE,
    IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE_MULTITURN,
)
from src.exp4_runtime.inference import common as runtime_common  # noqa: E402


DEFAULT_MEMORY_PATH = (
    ROOT
    / "outputs"
    / "amem"
    / "MPT_v2_0725_gpt-5-mini_batch"
    / "memory.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "amem"
    / "MPT_v2_0725_gpt-5-mini_batch"
    / "inference"
)


def paths_for(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = Path(args.output_dir).resolve()
    return {
        "output_dir": output_dir,
        "manifest": output_dir / "manifest.jsonl",
        "requests": output_dir / "requests.jsonl",
        "summary": output_dir / "prepare_summary.json",
        "state": output_dir / "batch_state.json",
        "raw": output_dir / "raw_results.jsonl",
        "errors": output_dir / "raw_errors.jsonl",
        "predictions": output_dir / "predictions.json",
        "log": output_dir / "inference.jsonl",
        "evaluation": output_dir / "evaluation.json",
    }


def require_api_key(args: argparse.Namespace) -> str:
    value = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not value:
        raise RuntimeError("Pass --api_key or set OPENAI_API_KEY.")
    return value


def client_for(args: argparse.Namespace) -> OpenAI:
    return OpenAI(api_key=require_api_key(args), timeout=120.0)


def stratified_sample(
    population: Sequence[dict[str, Any]],
    *,
    per_condition: int,
    seed: int,
) -> list[dict[str, Any]]:
    if per_condition <= 0:
        raise ValueError("--sample_per_condition must be positive")
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for row in population:
        by_condition.setdefault(str(row["condition"]), []).append(row)
    selected: list[dict[str, Any]] = []
    generator = random.Random(seed)
    for condition, rows in sorted(by_condition.items()):
        if per_condition > len(rows):
            raise ValueError(
                f"sample_per_condition={per_condition} exceeds "
                f"{condition} population={len(rows)}"
            )
        indexes = sorted(generator.sample(range(len(rows)), per_condition))
        selected.extend(copy.deepcopy(rows[index]) for index in indexes)
    selected.sort(key=lambda row: int(row["population_index"]))
    return selected


def inference_request(
    row: Mapping[str, Any],
    *,
    model: str,
    reasoning_effort: str,
    max_completion_tokens: int = 2048,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": row["prompt"]}],
        "max_completion_tokens": max_completion_tokens,
    }
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    return {
        "custom_id": row["sample_id"],
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


def prepare(
    args: argparse.Namespace, *, embedder: Any | None = None
) -> dict[str, Any]:
    paths = paths_for(args)
    immutable = (
        paths["state"],
        paths["raw"],
        paths["predictions"],
        paths["evaluation"],
    )
    if any(path.exists() for path in immutable):
        raise FileExistsError(
            f"Recorded A-MEM inference state exists under {paths['output_dir']}; "
            "use a new output directory."
        )
    prepared_paths = (paths["manifest"], paths["requests"], paths["summary"])
    if any(path.exists() for path in prepared_paths) and not args.force:
        raise FileExistsError(
            f"Prepared A-MEM inference files exist under {paths['output_dir']}; "
            "pass --force only before a batch has been submitted."
        )
    memory_path = Path(args.memory_path).resolve()
    memory = load_memory_artifact(memory_path)
    embedder = embedder or make_embedder(args.embedding_model)
    population = build_population(
        query=args.query,
        context_type=args.context_type,
        input_path=args.input_path,
        exclude_easy_conflict=args.exclude_easy_conflict,
    )
    if args.sample_per_condition is not None:
        selected = stratified_sample(
            population,
            per_condition=args.sample_per_condition,
            seed=args.seed,
        )
        sampling = "stratified_random_without_replacement"
    elif args.max_queries is not None:
        if args.max_queries <= 0:
            raise ValueError("--max_queries must be positive")
        selected = copy.deepcopy(population[: args.max_queries])
        sampling = "population_prefix"
    else:
        selected = copy.deepcopy(population)
        sampling = "full_population"

    tools_by_schema = {
        schema: runtime_common.load_tools_from_file(
            str(ROOT / "config" / f"schema_{schema}.json")
        )
        for schema in {str(row["schema"]) for row in selected}
    }
    manifest: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    query_embeddings = embedder.encode(
        [str(row["utterance"]) for row in selected]
    )
    if len(query_embeddings) != len(selected):
        raise RuntimeError("Embedding result count does not match query count")
    for row, query_embedding in zip(selected, query_embeddings):
        example_id = str(row["example_id"])
        notes = (memory.get(example_id) or {}).get("notes") or []
        retrieved = retrieve_notes_from_embedding(
            notes,
            query_embedding,
            top_k=args.memory_top_k,
            linked_neighbor_limit=args.linked_neighbor_limit,
        )
        template = (
            IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE
            if row["turn"] == "single"
            else IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE_MULTITURN
        )
        prompt = runtime_common.build_memory_prompt(
            example=row["original_ex"],
            retrieved_memories_text=format_retrieved_notes(retrieved),
            current_user_utterance=row["utterance"],
            template=template,
            context_type=args.context_type,
            tools_schema=tools_by_schema[str(row["schema"])],
        )
        prepared = {
            **row,
            "sample_id": f"amem-{int(row['population_index']):05d}",
            "prompt": prompt,
            "context_type": args.context_type,
            "retrieved_memories": [public_note(note) for note in retrieved],
            "retrieved_memory_count": len(retrieved),
            "memory_top_k": args.memory_top_k,
            "linked_neighbor_limit": args.linked_neighbor_limit,
        }
        manifest.append(prepared)
        requests.append(
            inference_request(
                prepared,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                max_completion_tokens=args.max_completion_tokens,
            )
        )

    atomic_write_jsonl(paths["manifest"], manifest)
    atomic_write_jsonl(paths["requests"], requests)
    summary = {
        "created_at": now_iso(),
        "input_path": str(Path(args.input_path).resolve()),
        "input_sha256": sha256_file(Path(args.input_path).resolve()),
        "memory_path": str(memory_path),
        "memory_sha256": sha256_file(memory_path),
        "query": args.query,
        "context_type": args.context_type,
        "exclude_easy_conflict": args.exclude_easy_conflict,
        "population_count": len(population),
        "request_count": len(requests),
        "sampling": sampling,
        "sample_per_condition": args.sample_per_condition,
        "seed": args.seed,
        "condition_counts": dict(
            sorted(Counter(str(row["condition"]) for row in manifest).items())
        ),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "max_completion_tokens": args.max_completion_tokens,
        "embedding_model": args.embedding_model,
        "memory_top_k": args.memory_top_k,
        "linked_neighbor_limit": args.linked_neighbor_limit,
        "endpoint": "/v1/chat/completions",
        "manifest_sha256": sha256_file(paths["manifest"]),
        "requests_sha256": sha256_file(paths["requests"]),
        "request_bytes": paths["requests"].stat().st_size,
    }
    atomic_write_json(paths["summary"], summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def submit(args: argparse.Namespace) -> dict[str, Any]:
    paths = paths_for(args)
    if paths["state"].exists():
        state = read_json(paths["state"])
        raise RuntimeError(
            f"A-MEM inference batch already recorded as {state.get('batch_id')}; "
            "refusing duplicate charges."
        )
    if not paths["requests"].exists():
        raise FileNotFoundError(
            f"Prepare requests before submission: {paths['requests']}"
        )
    summary = read_json(paths["summary"])
    if not isinstance(summary, dict):
        raise ValueError("A-MEM inference prepare summary must be a JSON object")
    expected = {
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "max_completion_tokens": args.max_completion_tokens,
        "embedding_model": args.embedding_model,
        "memory_top_k": args.memory_top_k,
        "linked_neighbor_limit": args.linked_neighbor_limit,
        "query": args.query,
        "context_type": args.context_type,
        "exclude_easy_conflict": args.exclude_easy_conflict,
        "sample_per_condition": args.sample_per_condition,
        "seed": args.seed,
    }
    mismatches = {
        key: {"recorded": summary.get(key), "requested": value}
        for key, value in expected.items()
        if summary.get(key) != value
    }
    requests_sha256 = sha256_file(paths["requests"])
    if summary.get("requests_sha256") != requests_sha256:
        mismatches["requests_sha256"] = {
            "recorded": summary.get("requests_sha256"),
            "requested": requests_sha256,
        }
    if mismatches:
        raise RuntimeError(
            "A-MEM inference prepare/config mismatch; use the recorded flags "
            f"or a new output directory: {json.dumps(mismatches, sort_keys=True)}"
        )
    client = client_for(args)
    state = {
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "phase": "submission_pending",
        "input_file_id": None,
        "batch_id": None,
        "status": "submission_pending",
        "output_file_id": None,
        "error_file_id": None,
        "requests_sha256": requests_sha256,
        "model": summary["model"],
        "reasoning_effort": summary["reasoning_effort"],
        "max_completion_tokens": summary["max_completion_tokens"],
        "input_path": summary["input_path"],
        "input_sha256": summary["input_sha256"],
        "memory_path": summary["memory_path"],
        "memory_sha256": summary["memory_sha256"],
        "manifest_sha256": summary["manifest_sha256"],
    }
    atomic_write_json(paths["state"], state)
    with paths["requests"].open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="batch")
    state["input_file_id"] = uploaded.id
    state["updated_at"] = now_iso()
    atomic_write_json(paths["state"], state)
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "experiment": "experiment8",
            "method": "amem",
            "stage": "inference",
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "requests_sha256": requests_sha256,
        },
    )
    state.update(
        {
            "updated_at": now_iso(),
            "phase": "submitted",
            "batch_id": batch.id,
            "status": batch.status,
            "output_file_id": batch.output_file_id,
            "error_file_id": batch.error_file_id,
        }
    )
    atomic_write_json(paths["state"], state)
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return state


def refresh_status(args: argparse.Namespace) -> dict[str, Any]:
    paths = paths_for(args)
    state = read_json(paths["state"])
    batch_id = state.get("batch_id")
    if not batch_id:
        raise RuntimeError(
            "A-MEM inference submission is pending without a batch_id. "
            "Reconcile it in the OpenAI dashboard before any resubmission."
        )
    batch = client_for(args).batches.retrieve(batch_id)
    state.update(
        {
            "updated_at": now_iso(),
            "status": batch.status,
            "output_file_id": batch.output_file_id,
            "error_file_id": batch.error_file_id,
            "request_counts": (
                batch.request_counts.model_dump()
                if batch.request_counts is not None
                else None
            ),
        }
    )
    atomic_write_json(paths["state"], state)
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return state


def download_results(args: argparse.Namespace) -> Path:
    paths = paths_for(args)
    if args.raw_results:
        source = Path(args.raw_results).resolve()
        if source != paths["raw"]:
            paths["raw"].parent.mkdir(parents=True, exist_ok=True)
            paths["raw"].write_bytes(source.read_bytes())
        return paths["raw"]
    state = refresh_status(args)
    if state.get("status") != "completed":
        raise RuntimeError(f"Batch is not completed: {state.get('status')}")
    output_file_id = state.get("output_file_id")
    if not output_file_id:
        raise RuntimeError("Completed batch has no output_file_id")
    client = client_for(args)
    paths["raw"].write_bytes(client.files.content(output_file_id).content)
    if state.get("error_file_id"):
        paths["errors"].write_bytes(
            client.files.content(state["error_file_id"]).content
        )
    return paths["raw"]


def build_prediction(
    manifest: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    model: str,
    batch_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    error = payload.get("error")
    record = {
        "example_id": manifest["example_id"],
        "example_id_sub": manifest["example_id_sub"],
        "method": "amem",
        "model_name": model,
        "context_type": manifest.get("context_type", "memory_api"),
        "test_utterance": manifest["utterance"],
        "reference_ground_truth": manifest["reference_ground_truth"],
        "retrieved_memories": manifest["retrieved_memories"],
        "retrieved_memory_count": manifest["retrieved_memory_count"],
        "memory_top_k": manifest["memory_top_k"],
        "linked_neighbor_limit": manifest["linked_neighbor_limit"],
        "status": "ERROR" if error else "OK",
        "error": error,
        "model_input": manifest["prompt"],
        "llm_output": error or payload.get("content", ""),
        "reasoning_content": "" if error else payload.get("reasoning", ""),
        "token_counts": {} if error else payload.get("usage", {}),
        "population_index": manifest["population_index"],
        "turn": manifest["turn"],
        "query": manifest["query"],
        "schema": manifest["schema"],
        "pref_type": manifest["pref_type"],
        "condition": manifest["condition"],
        "batch_provider": "openai",
        "batch_id": batch_id,
    }
    prediction = copy.deepcopy(manifest["original_ex"])
    prediction.update(record)
    return prediction, record


def collect(args: argparse.Namespace) -> dict[str, Any]:
    paths = paths_for(args)
    raw_path = download_results(args)
    manifest_rows = read_jsonl(paths["manifest"])
    raw_rows = read_jsonl(raw_path)
    raw_by_id = index_unique_rows(
        raw_rows,
        id_key="custom_id",
        label="A-MEM inference Batch results",
    )
    expected_ids = {str(row["sample_id"]) for row in manifest_rows}
    unknown = set(raw_by_id) - expected_ids
    if unknown:
        raise RuntimeError(
            f"Results contain unknown custom IDs: {sorted(unknown)[:5]}"
        )
    state = (
        read_json(paths["state"])
        if paths["state"].exists()
        else {"batch_id": "local-test"}
    )
    summary = read_json(paths["summary"])
    if not isinstance(summary, dict):
        raise ValueError("A-MEM inference prepare summary must be a JSON object")
    recorded_model = str(state.get("model") or summary["model"])
    recorded_reasoning_effort = str(
        state.get("reasoning_effort") or summary["reasoning_effort"]
    )
    predictions: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []
    for manifest in manifest_rows:
        raw = raw_by_id.get(str(manifest["sample_id"]))
        payload = (
            openai_result_payload(raw)
            if raw is not None
            else {
                "error": "BATCH_ERROR: missing result for custom_id",
                "content": "",
                "usage": {},
            }
        )
        prediction, log = build_prediction(
            manifest,
            payload,
            model=recorded_model,
            batch_id=str(state.get("batch_id") or "unknown"),
        )
        predictions.append(prediction)
        logs.append(log)
    predictions.sort(key=lambda row: int(row["population_index"]))
    logs.sort(key=lambda row: int(row["population_index"]))
    atomic_write_json(paths["predictions"], predictions)
    atomic_write_jsonl(paths["log"], logs)
    report = evaluation_report(predictions)
    report.update(
        {
            "method": "amem",
            "provider": "openai_batch",
            "model": recorded_model,
            "reasoning_effort": recorded_reasoning_effort,
            "batch_id": state.get("batch_id"),
            "prediction_count": len(predictions),
            "raw_result_count": len(raw_rows),
            "manifest_sha256": sha256_file(paths["manifest"]),
            "raw_results_sha256": sha256_file(raw_path),
            "generated_at": now_iso(),
        }
    )
    atomic_write_json(paths["evaluation"], report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "submit", "status", "collect")
    )
    parser.add_argument("--memory_path", default=str(DEFAULT_MEMORY_PATH))
    parser.add_argument(
        "--input_path", default=str(ROOT / "data/MPT_v2_0725.json")
    )
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--reasoning_effort", default=DEFAULT_REASONING_EFFORT
    )
    parser.add_argument("--max_completion_tokens", type=int, default=2048)
    parser.add_argument(
        "--embedding_model", default=DEFAULT_EMBEDDING_MODEL
    )
    parser.add_argument("--query", choices=["hint", "nohint"], default="hint")
    parser.add_argument("--context_type", default="memory_api")
    parser.add_argument("--memory_top_k", type=int, default=10)
    parser.add_argument("--linked_neighbor_limit", type=int, default=10)
    parser.add_argument("--sample_per_condition", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument(
        "--exclude_easy_conflict",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--raw_results", default=None)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "submit":
        submit(args)
    elif args.command == "status":
        refresh_status(args)
    else:
        collect(args)


if __name__ == "__main__":
    main()
