#!/usr/bin/env python3
"""Resume MPT_v2_0725 vanilla inference with direct provider Batch APIs.

The OpenAI and Anthropic checkpoints have different reusable sample sets.  This
script therefore prepares, submits, and collects a provider-specific complement
for each checkpoint instead of assuming one shared remaining manifest.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_vanilla_batch_sample import (  # noqa: E402
    DEFAULT_ANTHROPIC_MAX_TOKENS,
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OPENAI_REASONING_EFFORT,
    DEFAULT_SEED,
    anthropic_request,
    anthropic_result_payload,
    build_population,
    build_prediction,
    download_anthropic_results,
    download_openai_results,
    evaluation_report,
    fetch_anthropic_status,
    fetch_openai_status,
    now_iso,
    openai_request,
    openai_result_payload,
    read_json,
    read_jsonl,
    sha256_file,
    submit_anthropic,
    submit_openai,
    write_json,
)


DEFAULT_DATASET = ROOT / "data" / "MPT_v2_0725.json"
DEFAULT_TARGET_ROOT = (
    ROOT / "outputs" / "mpt_v2_0725_hint_no_easy_conflict"
)
DEFAULT_OUTPUT_DIR = DEFAULT_TARGET_ROOT / "vanilla_api_batch_resume"
PROVIDERS = ("openai", "anthropic")
CONDITIONS = (
    "single_easy",
    "single_medium",
    "single_hard",
    "multi_easy",
    "multi_medium",
    "multi_hard",
)
PROVIDER_NAMES = {
    "openai": "openai_gpt-5_minimal",
    "anthropic": "anthropic_claude-haiku-4-5_non_reasoning",
}


def provider_name(provider: str) -> str:
    try:
        return PROVIDER_NAMES[provider]
    except KeyError as exc:
        raise ValueError(f"Unsupported provider: {provider}") from exc


def provider_dir(base: Path, provider: str) -> Path:
    return base / provider_name(provider)


def shard_dir(base: Path, provider: str, condition: str) -> Path:
    return provider_dir(base, provider) / "shards" / condition


def checkpoint_path(target_root: Path, provider: str) -> Path:
    return target_root / "vanilla_llm" / provider_name(provider) / "inference.jsonl"


def resolve_providers(provider: str) -> Sequence[str]:
    return PROVIDERS if provider == "both" else (provider,)


def atomic_write_jsonl(
    path: Path, rows: Iterable[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def read_checkpoint(path: Path) -> Dict[str, Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    for record in read_jsonl(path):
        sample_id = str(record.get("sample_id", ""))
        if not sample_id:
            raise ValueError(f"Checkpoint record is missing sample_id: {path}")
        latest[sample_id] = record
    return latest


def record_is_complete(record: Mapping[str, Any]) -> bool:
    output = str(record.get("llm_output", "")).lstrip()
    if record.get("status") != "OK":
        return False
    if record.get("finish_reason") == "length":
        return False
    if not bool(record.get("thinking")):
        return True
    return not output.startswith("<think>")


def select_pending(
    population: Sequence[Dict[str, Any]],
    checkpoint: Mapping[str, Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    population_by_id = {str(row["sample_id"]): row for row in population}
    unknown = sorted(set(checkpoint) - set(population_by_id))
    if unknown:
        raise ValueError(f"Checkpoint contains unknown sample IDs: {unknown[:5]}")
    for sample_id, record in checkpoint.items():
        expected = population_by_id[sample_id]
        if int(record.get("population_index", -1)) != int(
            expected["population_index"]
        ):
            raise ValueError(f"Population index drift for {sample_id}")
    completed_ids = {
        sample_id
        for sample_id, record in checkpoint.items()
        if record_is_complete(record)
    }
    return [
        copy.deepcopy(row)
        for row in population
        if str(row["sample_id"]) not in completed_ids
    ]


def minimal_manifest_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "sample_id": row["sample_id"],
        "population_index": row["population_index"],
        "condition": row["condition"],
        "turn": row["turn"],
        "pref_type": row["pref_type"],
        "schema": row["schema"],
        "example_id_sub": row["example_id_sub"],
    }


def provider_settings(args: argparse.Namespace, provider: str) -> Dict[str, Any]:
    if provider == "openai":
        return {
            "model": args.openai_model,
            "reasoning_effort": args.openai_reasoning_effort,
            "endpoint": "/v1/chat/completions",
        }
    return {
        "model": args.anthropic_model,
        "thinking": "disabled",
        "max_tokens": args.anthropic_max_tokens,
        "endpoint": "/v1/messages/batches",
    }


def request_for_row(
    args: argparse.Namespace, provider: str, row: Mapping[str, Any]
) -> Dict[str, Any]:
    if provider == "openai":
        return openai_request(
            row, args.openai_model, args.openai_reasoning_effort
        )
    return anthropic_request(
        row, args.anthropic_model, args.anthropic_max_tokens
    )


def prepare_command(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    target_root = Path(args.target_root).resolve()
    dataset = Path(args.dataset).resolve()
    summary_path = output_dir / "resume_summary.json"
    recorded_batches = sorted(output_dir.glob("**/batch_state.json"))
    if recorded_batches:
        raise RuntimeError(
            "Refusing to prepare over recorded provider batches: "
            f"{recorded_batches[0]}"
        )
    if summary_path.exists() and not args.force:
        raise FileExistsError(
            f"Resume run already prepared under {output_dir}; use --force to replace"
        )

    population = build_population(
        query=args.query,
        context_type=args.context_type,
        input_path=dataset,
        exclude_easy_conflict=args.exclude_easy_conflict,
    )
    if len({str(row["sample_id"]) for row in population}) != len(population):
        raise RuntimeError("Target population contains duplicate sample IDs")

    summary: Dict[str, Any] = {
        "created_at": now_iso(),
        "dataset": str(dataset),
        "dataset_sha256": sha256_file(dataset),
        "target_root": str(target_root),
        "query": args.query,
        "context_type": args.context_type,
        "exclude_easy_conflict": args.exclude_easy_conflict,
        "sample_seed": DEFAULT_SEED,
        "population_count": len(population),
        "providers": {},
    }
    for provider in PROVIDERS:
        source_checkpoint = checkpoint_path(target_root, provider)
        checkpoint = read_checkpoint(source_checkpoint)
        pending = select_pending(population, checkpoint)
        manifest_path = provider_dir(output_dir, provider) / "pending_manifest.jsonl"
        atomic_write_jsonl(
            manifest_path, (minimal_manifest_row(row) for row in pending)
        )

        counts = Counter(str(row["condition"]) for row in pending)
        settings = provider_settings(args, provider)
        settings.update(
            {
                "checkpoint": str(source_checkpoint),
                "checkpoint_sha256": sha256_file(source_checkpoint),
                "checkpoint_latest_rows": len(checkpoint),
                "completed_rows": len(population) - len(pending),
                "pending_rows": len(pending),
                "pending_manifest_sha256": sha256_file(manifest_path),
                "condition_counts": dict(sorted(counts.items())),
                "shards": {},
            }
        )
        for condition in CONDITIONS:
            rows = [row for row in pending if row["condition"] == condition]
            requests_path = (
                shard_dir(output_dir, provider, condition) / "requests.jsonl"
            )
            atomic_write_jsonl(
                requests_path,
                (request_for_row(args, provider, row) for row in rows),
            )
            settings["shards"][condition] = {
                "count": len(rows),
                "bytes": requests_path.stat().st_size,
                "sha256": sha256_file(requests_path),
            }
        summary["providers"][provider] = settings
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def validate_prepared(
    output_dir: Path,
    providers: Sequence[str] = PROVIDERS,
) -> Dict[str, Any]:
    summary = read_json(output_dir / "resume_summary.json")
    dataset = Path(summary["dataset"])
    if sha256_file(dataset) != summary["dataset_sha256"]:
        raise RuntimeError("Dataset checksum mismatch")
    population = build_population(
        query=summary["query"],
        context_type=summary["context_type"],
        input_path=dataset,
        exclude_easy_conflict=bool(summary["exclude_easy_conflict"]),
    )
    if len(population) != int(summary["population_count"]):
        raise RuntimeError("Population count drift")

    population_ids = {str(row["sample_id"]) for row in population}
    for provider in providers:
        provider_summary = summary["providers"][provider]
        source_checkpoint = Path(provider_summary["checkpoint"])
        if sha256_file(source_checkpoint) != provider_summary["checkpoint_sha256"]:
            raise RuntimeError(f"{provider} checkpoint changed after preparation")
        checkpoint = read_checkpoint(source_checkpoint)
        pending = select_pending(population, checkpoint)
        manifest_path = provider_dir(output_dir, provider) / "pending_manifest.jsonl"
        manifest = read_jsonl(manifest_path)
        if sha256_file(manifest_path) != provider_summary[
            "pending_manifest_sha256"
        ]:
            raise RuntimeError(f"{provider} pending manifest checksum mismatch")
        pending_ids = [str(row["sample_id"]) for row in pending]
        manifest_ids = [str(row["sample_id"]) for row in manifest]
        if manifest_ids != pending_ids:
            raise RuntimeError(f"{provider} pending manifest is not the exact complement")
        if len(set(manifest_ids)) != len(manifest_ids):
            raise RuntimeError(f"{provider} pending manifest has duplicate IDs")
        if not set(manifest_ids).issubset(population_ids):
            raise RuntimeError(f"{provider} pending manifest has unknown IDs")

        for condition in CONDITIONS:
            requests_path = (
                shard_dir(output_dir, provider, condition) / "requests.jsonl"
            )
            expected = provider_summary["shards"][condition]
            requests = read_jsonl(requests_path)
            if sha256_file(requests_path) != expected["sha256"]:
                raise RuntimeError(
                    f"{provider}/{condition} request checksum mismatch"
                )
            if len(requests) != int(expected["count"]):
                raise RuntimeError(
                    f"{provider}/{condition} request count mismatch"
                )
            request_ids = [str(row["custom_id"]) for row in requests]
            expected_ids = [
                str(row["sample_id"])
                for row in pending
                if row["condition"] == condition
            ]
            if request_ids != expected_ids:
                raise RuntimeError(
                    f"{provider}/{condition} request IDs do not match manifest"
                )
    return summary


def submit_command(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    providers = resolve_providers(args.provider)
    summary = validate_prepared(output_dir, providers)
    for provider in providers:
        for condition in CONDITIONS:
            shard = summary["providers"][provider]["shards"][condition]
            if int(shard["count"]) == 0:
                continue
            directory = shard_dir(output_dir, provider, condition)
            state_path = directory / "batch_state.json"
            if state_path.exists() and not args.force_resubmit:
                state = read_json(state_path)
                print(
                    f"SKIP provider={provider} condition={condition} "
                    f"batch_id={state.get('batch_id')}"
                )
                continue
            if provider == "openai":
                submit_openai(directory, args.force_resubmit)
            else:
                submit_anthropic(directory, args.force_resubmit)


def shard_status(
    output_dir: Path,
    summary: Mapping[str, Any],
    provider: str,
    condition: str,
) -> Dict[str, Any]:
    count = int(summary["providers"][provider]["shards"][condition]["count"])
    if count == 0:
        return {
            "provider": provider,
            "condition": condition,
            "status": "empty",
            "total": 0,
        }
    directory = shard_dir(output_dir, provider, condition)
    if not (directory / "batch_state.json").exists():
        return {
            "provider": provider,
            "condition": condition,
            "status": "not_submitted",
            "total": count,
        }
    if provider == "openai":
        body = fetch_openai_status(directory)
        counts = body.get("request_counts") or {}
        return {
            "provider": provider,
            "condition": condition,
            "status": body.get("status"),
            "total": counts.get("total", count),
            "completed": counts.get("completed", 0),
            "failed": counts.get("failed", 0),
        }
    body = fetch_anthropic_status(directory)
    counts = body.get("request_counts") or {}
    return {
        "provider": provider,
        "condition": condition,
        "status": body.get("processing_status"),
        "processing": counts.get("processing", 0),
        "succeeded": counts.get("succeeded", 0),
        "errored": counts.get("errored", 0),
        "canceled": counts.get("canceled", 0),
        "expired": counts.get("expired", 0),
    }


def status_command(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    providers = resolve_providers(args.provider)
    summary = validate_prepared(output_dir, providers)
    for provider in providers:
        for condition in CONDITIONS:
            print(
                json.dumps(
                    shard_status(output_dir, summary, provider, condition),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )


def reconstruct_pending(
    output_dir: Path,
    summary: Mapping[str, Any],
    provider: str,
) -> list[Dict[str, Any]]:
    manifest = read_jsonl(
        provider_dir(output_dir, provider) / "pending_manifest.jsonl"
    )
    manifest_by_id = {str(row["sample_id"]): row for row in manifest}
    population = build_population(
        query=summary["query"],
        context_type=summary["context_type"],
        input_path=summary["dataset"],
        exclude_easy_conflict=bool(summary["exclude_easy_conflict"]),
    )
    pending = [
        row for row in population if str(row["sample_id"]) in manifest_by_id
    ]
    if len(pending) != len(manifest):
        raise RuntimeError(f"Could not reconstruct every {provider} pending row")
    return pending


def collect_raw_shards(
    output_dir: Path,
    summary: Mapping[str, Any],
    provider: str,
) -> Dict[str, Dict[str, Any]]:
    raw_by_id: Dict[str, Dict[str, Any]] = {}
    for condition in CONDITIONS:
        if int(summary["providers"][provider]["shards"][condition]["count"]) == 0:
            continue
        directory = shard_dir(output_dir, provider, condition)
        raw_path = directory / "raw_results.jsonl"
        if not raw_path.exists():
            raw_path = (
                download_openai_results(directory)
                if provider == "openai"
                else download_anthropic_results(directory)
            )
        for raw in read_jsonl(raw_path):
            custom_id = str(raw["custom_id"])
            if custom_id in raw_by_id:
                raise RuntimeError(f"Duplicate raw result custom_id: {custom_id}")
            raw_by_id[custom_id] = raw
    return raw_by_id


def collect_provider(
    output_dir: Path,
    summary: Mapping[str, Any],
    provider: str,
) -> Dict[str, Any]:
    pending = reconstruct_pending(output_dir, summary, provider)
    raw_by_id = collect_raw_shards(output_dir, summary, provider)
    expected_ids = {str(row["sample_id"]) for row in pending}
    if set(raw_by_id) != expected_ids:
        missing = sorted(expected_ids - set(raw_by_id))
        extra = sorted(set(raw_by_id) - expected_ids)
        raise RuntimeError(
            f"{provider} raw coverage mismatch: missing={missing[:5]} extra={extra[:5]}"
        )

    provider_summary = summary["providers"][provider]
    model = str(provider_summary["model"])
    new_records: Dict[str, Dict[str, Any]] = {}
    for row in pending:
        raw = raw_by_id[str(row["sample_id"])]
        payload = (
            openai_result_payload(raw)
            if provider == "openai"
            else anthropic_result_payload(raw)
        )
        state = read_json(
            shard_dir(output_dir, provider, str(row["condition"]))
            / "batch_state.json"
        )
        _, record = build_prediction(
            row,
            payload,
            provider,
            model,
            str(state["batch_id"]),
            int(summary["sample_seed"]),
        )
        new_records[str(row["sample_id"])] = record

    destination = Path(provider_summary["checkpoint"])
    merged = read_checkpoint(destination)
    merged.update(new_records)
    population = build_population(
        query=summary["query"],
        context_type=summary["context_type"],
        input_path=summary["dataset"],
        exclude_easy_conflict=bool(summary["exclude_easy_conflict"]),
    )
    population_ids = {str(row["sample_id"]) for row in population}
    if set(merged) != population_ids:
        missing = sorted(population_ids - set(merged))
        extra = sorted(set(merged) - population_ids)
        raise RuntimeError(
            f"{provider} merged coverage mismatch: missing={missing[:5]} extra={extra[:5]}"
        )
    ordered_records = [
        merged[str(row["sample_id"])]
        for row in population
    ]
    if any(not record_is_complete(row) for row in ordered_records):
        incomplete = [
            str(row["sample_id"])
            for row in ordered_records
            if not record_is_complete(row)
        ]
        raise RuntimeError(
            f"{provider} has incomplete results; checkpoint not replaced: "
            f"{incomplete[:5]}"
        )

    predictions: list[Dict[str, Any]] = []
    for row in population:
        prediction = copy.deepcopy(row["original_ex"])
        prediction.update(merged[str(row["sample_id"])])
        predictions.append(prediction)
    predictions.sort(key=lambda row: int(row["population_index"]))
    ordered_records.sort(key=lambda row: int(row["population_index"]))

    atomic_write_jsonl(destination, ordered_records)
    write_json(destination.parent / "predictions.json", predictions)
    report = evaluation_report(predictions)
    report.update(
        {
            "provider": provider,
            "model": model,
            "prediction_count": len(predictions),
            "new_batch_rows": len(new_records),
            "inference_sha256": sha256_file(destination),
            "generated_at": now_iso(),
        }
    )
    write_json(destination.parent / "evaluation.json", report)
    return report


def collect_command(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    providers = resolve_providers(args.provider)
    summary = validate_prepared(output_dir, providers)
    for provider in providers:
        report = collect_provider(output_dir, summary, provider)
        print(json.dumps(report, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    prepare_parser.add_argument("--target-root", default=str(DEFAULT_TARGET_ROOT))
    prepare_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    prepare_parser.add_argument("--query", choices=["hint", "nohint"], default="hint")
    prepare_parser.add_argument("--context-type", default="diag-apilist")
    prepare_parser.add_argument(
        "--exclude-easy-conflict",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    prepare_parser.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL)
    prepare_parser.add_argument(
        "--openai-reasoning-effort",
        choices=["minimal", "low", "medium", "high"],
        default=DEFAULT_OPENAI_REASONING_EFFORT,
    )
    prepare_parser.add_argument(
        "--anthropic-model", default=DEFAULT_ANTHROPIC_MODEL
    )
    prepare_parser.add_argument(
        "--anthropic-max-tokens",
        type=int,
        default=DEFAULT_ANTHROPIC_MAX_TOKENS,
    )
    prepare_parser.add_argument("--force", action="store_true")
    prepare_parser.set_defaults(func=prepare_command)

    for command, func in (
        ("submit", submit_command),
        ("status", status_command),
        ("collect", collect_command),
    ):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
        command_parser.add_argument(
            "--provider",
            choices=["openai", "anthropic", "both"],
            default="both",
        )
        if command == "submit":
            command_parser.add_argument("--force-resubmit", action="store_true")
        command_parser.set_defaults(func=func)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
