"""Run the unsampled Experiment8 vanilla requests and merge the full population.

This companion to ``run_vanilla_batch_sample.py`` excludes the completed
1,000-request manifest, shards the remaining requests by turn/difficulty, and
creates one final inference file per model after all provider batches finish.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


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
    now_iso,
    openai_request,
    openai_result_payload,
    read_json,
    read_jsonl,
    sha256_file,
    submit_anthropic,
    submit_openai,
    write_json,
    write_jsonl,
)


SAMPLE_DIR = (
    ROOT / "outputs" / "vanilla_llm" / "batch_sample_1000_seed_20260723"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "outputs" / "vanilla_llm" / "batch_remaining_5508_seed_20260723"
)
DEFAULT_FULL_DIR = ROOT / "outputs" / "vanilla_llm" / "full_6508"
PROVIDERS = ("openai", "anthropic")
CONDITIONS = (
    "single_easy",
    "single_medium",
    "single_hard",
    "multi_easy",
    "multi_medium",
    "multi_hard",
)


def provider_name(provider: str) -> str:
    if provider == "openai":
        return "openai_gpt-5_minimal"
    if provider == "anthropic":
        return "anthropic_claude-haiku-4-5_non_reasoning"
    raise ValueError(f"Unsupported provider: {provider}")


def provider_dir(base: Path, provider: str) -> Path:
    return base / provider_name(provider)


def shard_dir(base: Path, provider: str, condition: str) -> Path:
    return provider_dir(base, provider) / "shards" / condition


def resolve_providers(provider: str) -> Sequence[str]:
    return PROVIDERS if provider == "both" else (provider,)


def select_remaining(
    population: Sequence[Dict[str, Any]],
    sampled_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    sampled_ids = [str(row["sample_id"]) for row in sampled_rows]
    if len(sampled_ids) != len(set(sampled_ids)):
        raise ValueError("Completed sample manifest contains duplicate sample_id values")
    population_by_id = {str(row["sample_id"]): row for row in population}
    unknown = sorted(set(sampled_ids) - set(population_by_id))
    if unknown:
        raise ValueError(f"Completed sample contains unknown IDs: {unknown[:5]}")
    for sampled in sampled_rows:
        current = population_by_id[str(sampled["sample_id"])]
        if int(sampled["population_index"]) != int(current["population_index"]):
            raise ValueError(
                f"Population index drift for {sampled['sample_id']}: "
                f"{sampled['population_index']} != {current['population_index']}"
            )
    sampled_set = set(sampled_ids)
    return [
        copy.deepcopy(row)
        for row in population
        if str(row["sample_id"]) not in sampled_set
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


def condition_rows(
    rows: Sequence[Dict[str, Any]], condition: str
) -> List[Dict[str, Any]]:
    return [row for row in rows if row["condition"] == condition]


def prepare_command(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    manifest_path = output_dir / "remaining_manifest.jsonl"
    summary_path = output_dir / "remaining_summary.json"
    if (manifest_path.exists() or summary_path.exists()) and not args.force:
        raise FileExistsError(
            f"Remaining run already prepared under {output_dir}; use --force to replace"
        )

    sample_manifest = read_jsonl(Path(args.sample_dir) / "sample_manifest.jsonl")
    population = build_population(query=args.query, context_type=args.context_type)
    remaining = select_remaining(population, sample_manifest)
    if len(sample_manifest) + len(remaining) != len(population):
        raise RuntimeError("Sample and remaining rows do not cover the full population")

    write_jsonl(manifest_path, (minimal_manifest_row(row) for row in remaining))
    for condition in CONDITIONS:
        rows = condition_rows(remaining, condition)
        write_jsonl(
            shard_dir(output_dir, "openai", condition) / "requests.jsonl",
            (
                openai_request(
                    row, args.openai_model, args.openai_reasoning_effort
                )
                for row in rows
            ),
        )
        write_jsonl(
            shard_dir(output_dir, "anthropic", condition) / "requests.jsonl",
            (
                anthropic_request(
                    row, args.anthropic_model, args.anthropic_max_tokens
                )
                for row in rows
            ),
        )

    sample_ids = {row["sample_id"] for row in sample_manifest}
    remaining_ids = {row["sample_id"] for row in remaining}
    population_ids = {row["sample_id"] for row in population}
    if sample_ids & remaining_ids:
        raise RuntimeError("Sample and remaining manifests overlap")
    if sample_ids | remaining_ids != population_ids:
        raise RuntimeError("Sample and remaining manifests do not cover population")

    counts = Counter(row["condition"] for row in remaining)
    summary: Dict[str, Any] = {
        "created_at": now_iso(),
        "query": args.query,
        "context_type": args.context_type,
        "seed": DEFAULT_SEED,
        "population_count": len(population),
        "completed_sample_count": len(sample_manifest),
        "remaining_count": len(remaining),
        "remaining_condition_counts": dict(sorted(counts.items())),
        "sample_manifest_sha256": sha256_file(
            Path(args.sample_dir) / "sample_manifest.jsonl"
        ),
        "remaining_manifest_sha256": sha256_file(manifest_path),
        "providers": {
            "openai": {
                "model": args.openai_model,
                "reasoning_effort": args.openai_reasoning_effort,
                "shards": {},
            },
            "anthropic": {
                "model": args.anthropic_model,
                "thinking": "disabled",
                "max_tokens": args.anthropic_max_tokens,
                "shards": {},
            },
        },
    }
    for provider in PROVIDERS:
        for condition in CONDITIONS:
            requests_path = (
                shard_dir(output_dir, provider, condition) / "requests.jsonl"
            )
            summary["providers"][provider]["shards"][condition] = {
                "count": counts[condition],
                "bytes": requests_path.stat().st_size,
                "sha256": sha256_file(requests_path),
            }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def validate_prepared(output_dir: Path) -> Dict[str, Any]:
    summary = read_json(output_dir / "remaining_summary.json")
    manifest_path = output_dir / "remaining_manifest.jsonl"
    if sha256_file(manifest_path) != summary["remaining_manifest_sha256"]:
        raise RuntimeError("Remaining manifest checksum mismatch")
    manifest = read_jsonl(manifest_path)
    if len(manifest) != int(summary["remaining_count"]):
        raise RuntimeError("Remaining manifest row count mismatch")
    if len({row["sample_id"] for row in manifest}) != len(manifest):
        raise RuntimeError("Remaining manifest contains duplicate sample IDs")
    for provider in PROVIDERS:
        for condition in CONDITIONS:
            requests_path = (
                shard_dir(output_dir, provider, condition) / "requests.jsonl"
            )
            expected = summary["providers"][provider]["shards"][condition]
            if sha256_file(requests_path) != expected["sha256"]:
                raise RuntimeError(
                    f"{provider}/{condition} request checksum mismatch"
                )
            if len(read_jsonl(requests_path)) != int(expected["count"]):
                raise RuntimeError(f"{provider}/{condition} request count mismatch")
    return summary


def submit_command(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    validate_prepared(output_dir)
    for provider in resolve_providers(args.provider):
        for condition in CONDITIONS:
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


def shard_status(output_dir: Path, provider: str, condition: str) -> Dict[str, Any]:
    directory = shard_dir(output_dir, provider, condition)
    state_path = directory / "batch_state.json"
    if not state_path.exists():
        return {"provider": provider, "condition": condition, "status": "not_submitted"}
    if provider == "openai":
        from scripts.run_vanilla_batch_sample import fetch_openai_status

        body = fetch_openai_status(directory)
        counts = body.get("request_counts") or {}
        return {
            "provider": provider,
            "condition": condition,
            "status": body.get("status"),
            "total": counts.get("total", 0),
            "completed": counts.get("completed", 0),
            "failed": counts.get("failed", 0),
        }
    from scripts.run_vanilla_batch_sample import fetch_anthropic_status

    body = fetch_anthropic_status(directory)
    counts = body.get("request_counts") or {}
    return {
        "provider": provider,
        "condition": condition,
        "status": body.get("processing_status"),
        "processing": counts.get("processing", 0),
        "succeeded": counts.get("succeeded", 0),
        "errored": counts.get("errored", 0),
    }


def status_command(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    for provider in resolve_providers(args.provider):
        for condition in CONDITIONS:
            print(
                json.dumps(
                    shard_status(output_dir, provider, condition),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )


def reconstruct_remaining(output_dir: Path) -> List[Dict[str, Any]]:
    summary = validate_prepared(output_dir)
    manifest = read_jsonl(output_dir / "remaining_manifest.jsonl")
    manifest_by_id = {row["sample_id"]: row for row in manifest}
    population = build_population(
        query=summary["query"], context_type=summary["context_type"]
    )
    remaining = [
        row for row in population if row["sample_id"] in manifest_by_id
    ]
    if len(remaining) != len(manifest):
        raise RuntimeError("Could not reconstruct every remaining population row")
    for row in remaining:
        recorded = manifest_by_id[row["sample_id"]]
        if int(recorded["population_index"]) != int(row["population_index"]):
            raise RuntimeError(f"Population drift for {row['sample_id']}")
    return remaining


def collect_raw_shards(output_dir: Path, provider: str) -> Dict[str, Dict[str, Any]]:
    raw_by_id: Dict[str, Dict[str, Any]] = {}
    for condition in CONDITIONS:
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


def collect_provider(output_dir: Path, provider: str) -> Dict[str, Any]:
    summary = read_json(output_dir / "remaining_summary.json")
    remaining = reconstruct_remaining(output_dir)
    raw_by_id = collect_raw_shards(output_dir, provider)
    expected_ids = {row["sample_id"] for row in remaining}
    if set(raw_by_id) != expected_ids:
        missing = sorted(expected_ids - set(raw_by_id))
        extra = sorted(set(raw_by_id) - expected_ids)
        raise RuntimeError(
            f"{provider} raw coverage mismatch: missing={missing[:5]} extra={extra[:5]}"
        )

    model = str(summary["providers"][provider]["model"])
    predictions: List[Dict[str, Any]] = []
    logs: List[Dict[str, Any]] = []
    for manifest_row in remaining:
        raw = raw_by_id[manifest_row["sample_id"]]
        payload = (
            openai_result_payload(raw)
            if provider == "openai"
            else anthropic_result_payload(raw)
        )
        condition = manifest_row["condition"]
        state = read_json(
            shard_dir(output_dir, provider, condition) / "batch_state.json"
        )
        prediction, log = build_prediction(
            manifest_row,
            payload,
            provider,
            model,
            state["batch_id"],
            int(summary["seed"]),
        )
        predictions.append(prediction)
        logs.append(log)
    predictions.sort(key=lambda row: row["population_index"])
    logs.sort(key=lambda row: row["population_index"])

    directory = provider_dir(output_dir, provider)
    write_json(directory / "predictions.json", predictions)
    write_jsonl(directory / "inference.jsonl", logs)
    report = evaluation_report(predictions)
    report.update(
        {
            "provider": provider,
            "model": model,
            "prediction_count": len(predictions),
            "raw_result_count": len(raw_by_id),
            "remaining_manifest_sha256": summary["remaining_manifest_sha256"],
            "generated_at": now_iso(),
        }
    )
    write_json(directory / "evaluation.json", report)
    return report


def collect_command(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    for provider in resolve_providers(args.provider):
        report = collect_provider(output_dir, provider)
        print(json.dumps(report, ensure_ascii=False, indent=2))


def read_model_rows(
    sample_dir: Path, remaining_dir: Path, provider: str
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    name = provider_name(provider)
    sample_predictions = read_json(sample_dir / name / "predictions.json")
    remaining_predictions = read_json(remaining_dir / name / "predictions.json")
    sample_logs = read_jsonl(sample_dir / name / "inference.jsonl")
    remaining_logs = read_jsonl(remaining_dir / name / "inference.jsonl")
    return (
        list(sample_predictions) + list(remaining_predictions),
        sample_logs + remaining_logs,
    )


def validate_full_rows(
    rows: Sequence[Mapping[str, Any]], expected_count: int
) -> None:
    if len(rows) != expected_count:
        raise RuntimeError(f"Expected {expected_count} rows, found {len(rows)}")
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(set(sample_ids)) != expected_count:
        raise RuntimeError("Full rows contain duplicate sample_id values")
    indices = sorted(int(row["population_index"]) for row in rows)
    if indices != list(range(expected_count)):
        raise RuntimeError("Full rows do not cover every population index exactly once")


def merge_command(args: argparse.Namespace) -> None:
    sample_dir = Path(args.sample_dir).resolve()
    remaining_dir = Path(args.output_dir).resolve()
    full_dir = Path(args.full_dir).resolve()
    remaining_summary = read_json(remaining_dir / "remaining_summary.json")
    expected_count = int(remaining_summary["population_count"])
    full_summary: Dict[str, Any] = {
        "created_at": now_iso(),
        "population_count": expected_count,
        "sample_count": int(remaining_summary["completed_sample_count"]),
        "remaining_count": int(remaining_summary["remaining_count"]),
        "sample_manifest_sha256": remaining_summary["sample_manifest_sha256"],
        "remaining_manifest_sha256": remaining_summary[
            "remaining_manifest_sha256"
        ],
        "providers": {},
    }
    for provider in PROVIDERS:
        predictions, logs = read_model_rows(sample_dir, remaining_dir, provider)
        validate_full_rows(predictions, expected_count)
        validate_full_rows(logs, expected_count)
        predictions.sort(key=lambda row: int(row["population_index"]))
        logs.sort(key=lambda row: int(row["population_index"]))
        destination = provider_dir(full_dir, provider)
        write_json(destination / "predictions.json", predictions)
        write_jsonl(destination / "inference.jsonl", logs)
        report = evaluation_report(predictions)
        report.update(
            {
                "provider": provider,
                "model": predictions[0]["model_name"],
                "prediction_count": len(predictions),
                "generated_at": now_iso(),
            }
        )
        write_json(destination / "evaluation.json", report)
        full_summary["providers"][provider] = {
            "model": predictions[0]["model_name"],
            "inference_path": str(destination / "inference.jsonl"),
            "inference_sha256": sha256_file(destination / "inference.jsonl"),
            "predictions_path": str(destination / "predictions.json"),
            "predictions_sha256": sha256_file(destination / "predictions.json"),
            "evaluation_path": str(destination / "evaluation.json"),
            "api_errors": report["overall"]["api_errors"],
            "parsing_failures": report["overall"]["parsing_failures"],
        }
    write_json(full_dir / "full_summary.json", full_summary)
    print(json.dumps(full_summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--sample-dir", default=str(SAMPLE_DIR))
    prepare_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    prepare_parser.add_argument("--query", choices=["hint", "nohint"], default="hint")
    prepare_parser.add_argument("--context-type", default="diag-apilist")
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

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--sample-dir", default=str(SAMPLE_DIR))
    merge_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    merge_parser.add_argument("--full-dir", default=str(DEFAULT_FULL_DIR))
    merge_parser.set_defaults(func=merge_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
