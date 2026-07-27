#!/usr/bin/env python3
"""Seed the MPT_v2_0725 experiment with strictly reusable mix600 artifacts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
OURS_DIR = ROOT / "methods" / "ours_memory"
for path in (ROOT, SCRIPT_DIR, OURS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import inference_multiturn as ours_multi  # type: ignore  # noqa: E402
import inference_singleturn as ours_single  # type: ignore  # noqa: E402
import run_local_full_inference as local_runtime  # noqa: E402
import run_vanilla_batch_sample as population_runtime  # noqa: E402
from src.exp4_prompts import (  # noqa: E402
    IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE,
    IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE_MULTITURN,
)
from src.exp4_runtime import common  # noqa: E402


DEFAULT_ARCHIVE_OUTPUTS = (
    ROOT / "archive" / "mix600_20260725_233232" / "outputs"
)
DEFAULT_OLD_INPUT = ROOT / "data" / "MPT_v2_mix600.json"
DEFAULT_NEW_INPUT = ROOT / "data" / "MPT_v2_0725.json"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "mpt_v2_0725_hint_no_easy_conflict"

MEMORY_MODELS = ("qwen3_8b", "gemma4_12b_it", "gpt_oss_20b")
REUSABLE_EXAMPLE_FIELDS = (
    "sessions",
    "api_calls",
    "api_calls_drop",
    "api_calls_pref",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_digest(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def normalized_ground_truth(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        return tuple(sorted(str(item) for item in value))
    return (str(value),)


def is_conflict_example(example: Mapping[str, Any]) -> bool:
    return population_runtime.is_conflict_example(example)


def reusable_example_ids(
    old_dataset: Sequence[Mapping[str, Any]],
    new_dataset: Sequence[Mapping[str, Any]],
) -> set[str]:
    old_by_id = {
        str(example.get("example_id")): example for example in old_dataset
    }
    reusable: set[str] = set()
    for example in new_dataset:
        example_id = str(example.get("example_id"))
        old_example = old_by_id.get(example_id)
        if old_example is None or is_conflict_example(example):
            continue
        if all(
            old_example.get(field) == example.get(field)
            for field in REUSABLE_EXAMPLE_FIELDS
        ):
            reusable.add(example_id)
    return reusable


def read_unique_jsonl(path: Path, key_name: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            key = str(value.get(key_name, ""))
            if not key:
                raise ValueError(f"{path}:{line_number} missing {key_name}")
            if key in records:
                raise ValueError(f"{path}:{line_number} duplicate {key_name}={key}")
            records[key] = value
    return records


def read_latest_checkpoint(
    path: Path,
    reusable_ids: set[str],
) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if str(value.get("example_id", "")) not in reusable_ids:
                continue
            sample_id = str(value.get("sample_id", ""))
            if not sample_id:
                raise ValueError(f"{path}:{line_number} missing sample_id")
            latest[sample_id] = value
    return latest


def match_key(
    *,
    example_id: str,
    turn: str,
    pref_type: str,
    utterance: str,
    ground_truth: Any,
    prompt: str,
) -> tuple[Any, ...]:
    return (
        example_id,
        turn,
        pref_type,
        utterance,
        normalized_ground_truth(ground_truth),
        prompt_digest(prompt),
    )


def target_key(row: Mapping[str, Any], prompt: str) -> tuple[Any, ...]:
    return match_key(
        example_id=str(row["example_id"]),
        turn=str(row["turn"]),
        pref_type=str(row["pref_type"]),
        utterance=str(row["utterance"]),
        ground_truth=row["reference_ground_truth"],
        prompt=prompt,
    )


def source_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return match_key(
        example_id=str(record["example_id"]),
        turn=str(record["turn"]),
        pref_type=str(record["pref_type"]),
        utterance=str(record["test_utterance"]),
        ground_truth=record["reference_ground_truth"],
        prompt=str(record["model_input"]),
    )


def remap_record(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    target_prompt: str,
    source_path: Path,
    memory_path: Path | None = None,
) -> dict[str, Any]:
    record = copy.deepcopy(dict(source))
    source_identity = {
        "checkpoint": str(source_path),
        "sample_id": source.get("sample_id"),
        "population_index": source.get("population_index"),
        "example_id_sub": source.get("example_id_sub"),
    }
    for field in (
        "sample_id",
        "population_index",
        "example_id",
        "example_id_sub",
        "turn",
        "query",
        "schema",
        "pref_type",
        "condition",
    ):
        record[field] = target[field]
    record["test_utterance"] = target["utterance"]
    record["reference_ground_truth"] = target["reference_ground_truth"]
    record["model_input"] = target_prompt
    if memory_path is not None:
        record["memory_path"] = str(memory_path)
    record["reuse_provenance"] = {
        "reused_at": now_iso(),
        "source": source_identity,
        "validation": {
            "example_fields_equal": list(REUSABLE_EXAMPLE_FIELDS),
            "query_equal": True,
            "ground_truth_equal_as_set": True,
            "prompt_equal": True,
            "source_record_complete": True,
        },
    }
    return record


def reusable_checkpoint_rows(
    source_path: Path,
    target_rows: Sequence[Mapping[str, Any]],
    reusable_ids: set[str],
    prompt_for_row: Callable[[Mapping[str, Any]], str],
    memory_path: Path | None = None,
) -> list[dict[str, Any]]:
    queues: dict[
        tuple[Any, ...],
        deque[tuple[Mapping[str, Any], str]],
    ] = defaultdict(deque)
    for row in target_rows:
        if str(row["example_id"]) not in reusable_ids:
            continue
        prompt = prompt_for_row(row)
        queues[target_key(row, prompt)].append((row, prompt))

    source_records = read_latest_checkpoint(source_path, reusable_ids)
    reused: list[dict[str, Any]] = []
    for source in sorted(
        source_records.values(),
        key=lambda record: int(record.get("population_index", -1)),
    ):
        if not local_runtime.record_is_complete(source):
            continue
        queue = queues.get(source_key(source))
        if not queue:
            continue
        target, target_prompt = queue.popleft()
        if str(source["model_input"]) != target_prompt:
            continue
        reused.append(
            remap_record(
                source,
                target,
                target_prompt,
                source_path,
                memory_path=memory_path,
            )
        )
    reused.sort(key=lambda record: int(record["population_index"]))
    return reused


def ours_prompt_builder(
    memory: Mapping[str, Mapping[str, Any]],
) -> Callable[[Mapping[str, Any]], str]:
    schema_cache: dict[str, list[dict[str, Any]]] = {}

    def build(row: Mapping[str, Any]) -> str:
        schema_name = str(row["schema"])
        if schema_name not in schema_cache:
            schema_cache[schema_name] = common.load_tools_from_file(
                str(ROOT / "config" / f"schema_{schema_name}.json")
            )
        example_id = str(row["example_id"])
        if row["turn"] == "single":
            return ours_single.build_memory_input_prompt(
                example=row["original_ex"],
                user_memory=memory[example_id],
                current_user_utterance=row["utterance"],
                template=IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE,
                context_type="memory_api",
                tools_schema=schema_cache[schema_name],
            )
        return ours_multi.build_memory_input_prompt(
            example=row["original_ex"],
            user_memory=memory[example_id],
            current_user_utterance=row["utterance"],
            template=IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE_MULTITURN,
            context_type="memory_api",
            tools_schema=schema_cache[schema_name],
        )

    return build


def filter_example_logs(
    source_path: Path,
    destination_path: Path,
    reusable_ids: set[str],
) -> int:
    selected: list[dict[str, Any]] = []
    with source_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if str(value.get("example_id", "")) in reusable_ids:
                selected.append(value)
    write_jsonl(destination_path, selected)
    return len(selected)


def population_manifest_row(
    row: Mapping[str, Any],
    reusable_ids: set[str],
) -> dict[str, Any]:
    return {
        "sample_id": row["sample_id"],
        "population_index": row["population_index"],
        "example_id": row["example_id"],
        "example_id_sub": row["example_id_sub"],
        "turn": row["turn"],
        "query": row["query"],
        "schema": row["schema"],
        "pref_type": row["pref_type"],
        "condition": row["condition"],
        "utterance": row["utterance"],
        "reference_ground_truth": row["reference_ground_truth"],
        "prompt_sha256": prompt_digest(str(row["prompt"])),
        "reusable_mix600_example": str(row["example_id"]) in reusable_ids,
        "conflict": is_conflict_example(row["original_ex"]),
    }


def bootstrap(args: argparse.Namespace) -> dict[str, Any]:
    archive_outputs = Path(args.archive_outputs).resolve()
    old_input = Path(args.old_input).resolve()
    new_input = Path(args.new_input).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_root}")

    stage_root = output_root.with_name(
        f".{output_root.name}.staging-{os.getpid()}"
    )
    if stage_root.exists():
        raise FileExistsError(f"Staging path already exists: {stage_root}")
    stage_root.mkdir(parents=True)

    old_dataset = read_json(old_input)
    new_dataset = read_json(new_input)
    reusable_ids = reusable_example_ids(old_dataset, new_dataset)
    if len(reusable_ids) != args.expected_reusable_examples:
        raise RuntimeError(
            "Reusable example mismatch: "
            f"expected={args.expected_reusable_examples}, actual={len(reusable_ids)}"
        )

    target_rows = population_runtime.build_population(
        query="hint",
        context_type="diag-apilist",
        input_path=new_input,
        exclude_easy_conflict=True,
    )
    if len(target_rows) != args.expected_count:
        raise RuntimeError(
            f"Population mismatch: expected={args.expected_count}, "
            f"actual={len(target_rows)}"
        )
    if len({str(row["sample_id"]) for row in target_rows}) != len(target_rows):
        raise RuntimeError("Target population has duplicate sample IDs")

    write_jsonl(
        stage_root / "population" / "population.jsonl",
        (population_manifest_row(row, reusable_ids) for row in target_rows),
    )

    memory_summary: dict[str, Any] = {}
    memory_maps: dict[str, dict[str, dict[str, Any]]] = {}
    archived_memory_root = (
        archive_outputs
        / "ours_memory"
        / "cross_model_mix600_high_reasoning_20260721"
    )
    for model in MEMORY_MODELS:
        source_path = archived_memory_root / model / "memory.jsonl"
        source_memory = read_unique_jsonl(source_path, "example_id")
        missing = sorted(reusable_ids - set(source_memory))
        if missing:
            raise RuntimeError(f"{model} memory missing reusable IDs: {missing[:5]}")
        selected = {
            example_id: source_memory[example_id]
            for example_id in sorted(reusable_ids)
        }
        destination = stage_root / "memory" / model / "memory.jsonl"
        write_jsonl(destination, selected.values())
        memory_maps[model] = selected
        entry: dict[str, Any] = {
            "source": str(source_path),
            "destination": str(
                output_root / "memory" / model / "memory.jsonl"
            ),
            "reused_examples": len(selected),
            "remaining_examples": len(new_dataset) - len(selected),
            "sha256": sha256_file(destination),
        }
        for log_name in ("refinement_logs.jsonl", "verifier_logs.jsonl"):
            source_log = archived_memory_root / model / log_name
            if source_log.exists():
                destination_log = stage_root / "memory" / model / log_name
                entry[log_name] = {
                    "source": str(source_log),
                    "reused_rows": filter_example_logs(
                        source_log,
                        destination_log,
                        reusable_ids,
                    ),
                }
        memory_summary[model] = entry

    vanilla_summary: dict[str, Any] = {}
    archived_vanilla_root = archive_outputs / "vanilla_llm" / "full_6508"
    for source_path in sorted(archived_vanilla_root.glob("*/inference.jsonl")):
        model_slug = source_path.parent.name
        reused = reusable_checkpoint_rows(
            source_path,
            target_rows,
            reusable_ids,
            prompt_for_row=lambda row: str(row["prompt"]),
        )
        destination = (
            stage_root / "vanilla_llm" / model_slug / "inference.jsonl"
        )
        write_jsonl(destination, reused)
        vanilla_summary[model_slug] = {
            "source": str(source_path),
            "destination": str(
                output_root / "vanilla_llm" / model_slug / "inference.jsonl"
            ),
            "reused_rows": len(reused),
            "pending_rows": len(target_rows) - len(reused),
            "sha256": sha256_file(destination),
        }

    ours_summary: dict[str, Any] = {}
    archived_ours_root = archive_outputs / "ours_memory" / "full_6508"
    for source_path in sorted(archived_ours_root.glob("*/inference.jsonl")):
        combination = source_path.parent.name
        memory_model = combination.split("_memory__to__", 1)[0]
        if memory_model not in memory_maps:
            raise RuntimeError(
                f"Cannot resolve memory model for combination: {combination}"
            )
        final_memory_path = (
            output_root / "memory" / memory_model / "memory.jsonl"
        )
        reused = reusable_checkpoint_rows(
            source_path,
            target_rows,
            reusable_ids,
            prompt_for_row=ours_prompt_builder(memory_maps[memory_model]),
            memory_path=final_memory_path,
        )
        destination = (
            stage_root / "ours_memory" / combination / "inference.jsonl"
        )
        write_jsonl(destination, reused)
        ours_summary[combination] = {
            "source": str(source_path),
            "destination": str(
                output_root / "ours_memory" / combination / "inference.jsonl"
            ),
            "memory_path": str(final_memory_path),
            "reused_rows": len(reused),
            "pending_rows": len(target_rows) - len(reused),
            "sha256": sha256_file(destination),
        }

    summary = {
        "created_at": now_iso(),
        "dataset": {
            "old": str(old_input),
            "old_sha256": sha256_file(old_input),
            "new": str(new_input),
            "new_sha256": sha256_file(new_input),
        },
        "archive_outputs": str(archive_outputs),
        "target": {
            "query": "hint",
            "exclude_easy_conflict": True,
            "population_count": len(target_rows),
            "reusable_examples": len(reusable_ids),
            "reusable_population_rows": sum(
                str(row["example_id"]) in reusable_ids for row in target_rows
            ),
            "conflict_population_rows": sum(
                is_conflict_example(row["original_ex"]) for row in target_rows
            ),
        },
        "reuse_policy": {
            "memory": (
                "same example_id and identical sessions/api_calls/"
                "api_calls_drop/api_calls_pref"
            ),
            "inference": (
                "reusable memory identity plus identical turn/difficulty/query/"
                "ground-truth set/full prompt and a complete source record"
            ),
        },
        "memory": memory_summary,
        "vanilla_llm": vanilla_summary,
        "ours_memory": ours_summary,
    }
    write_json(stage_root / "reuse_summary.json", summary)
    os.replace(stage_root, output_root)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-outputs",
        default=str(DEFAULT_ARCHIVE_OUTPUTS),
    )
    parser.add_argument("--old-input", default=str(DEFAULT_OLD_INPUT))
    parser.add_argument("--new-input", default=str(DEFAULT_NEW_INPUT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--expected-count", type=int, default=4695)
    parser.add_argument("--expected-reusable-examples", type=int, default=299)
    return parser


if __name__ == "__main__":
    result = bootstrap(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
