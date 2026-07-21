#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence


RELEASE_ROOT = Path(__file__).resolve().parents[3]
ROOT = RELEASE_ROOT / "src" / "extended_schema" / "mem0"

DEFAULT_SINGLE_JSON = ROOT / "output/singleturn/api/memory_api/hard/gemini-3-flash-preview_high/implicit_zs/0330_fixed_400_rerun.json"
DEFAULT_MULTI_JSON = ROOT / "output/multiturn/api/memory_api/hard/gemini-3-flash-preview_high/implicit_zs/0330_fixed_400_rerun.json"

SINGLE_SOURCE = RELEASE_ROOT / "src" / "baselines" / "mem0" / "step2_evaluate_singleturn.py"
MULTI_SOURCE = RELEASE_ROOT / "src" / "baselines" / "mem0" / "step2_evaluate_multiturn.py"

SINGLE_SCHEMA_PATH = RELEASE_ROOT / "schema_easy_extended.json"
MULTI_SCHEMA_PATH = RELEASE_ROOT / "schema_all_extended_complete.json"

MODEL_NAME = "gemini-3-flash-preview"
REASONING_EFFORT = "high"
CONTEXT_TYPE = "memory_api"
REQUEST_TIMEOUT_SECONDS = 45
MAX_ATTEMPTS = 4
RETRY_BACKOFF_SECONDS = 5


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def contains_503(row: Dict[str, Any]) -> bool:
    return "503" in json.dumps(row, ensure_ascii=False)


def extract_503_ids(rows: Sequence[Dict[str, Any]]) -> List[str]:
    ids: List[str] = []
    for row in rows:
        if contains_503(row):
            example_id_sub = str(row.get("example_id_sub", "")).strip()
            if example_id_sub:
                ids.append(example_id_sub)
    return ids


def build_backup_path(target_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return target_path.with_name(f"{target_path.name}.before_503_rerun_{timestamp}.bak")


def load_python_module(module_path: Path, module_name: str) -> Any:
    module_dir = str(module_path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def should_retry(llm_result: Dict[str, Any]) -> bool:
    error_text = str(llm_result.get("error") or "").strip()
    return bool(error_text) and "503" in error_text


async def call_with_retries(
    module: Any,
    prompt: str,
    tools_schema: List[Dict[str, Any]],
) -> Dict[str, Any]:
    last_result: Dict[str, Any] | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            llm_result = await asyncio.wait_for(
                module.call_llm_api_async(
                    prompt,
                    MODEL_NAME,
                    None,
                    tools_schema,
                    REASONING_EFFORT,
                ),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            llm_result = {
                "content": "",
                "reasoning": "",
                "token_counts": {},
                "error": f"RERUN_TIMEOUT_{REQUEST_TIMEOUT_SECONDS}s_attempt_{attempt}",
            }
        except Exception as exc:  # pragma: no cover - defensive path for flaky API/client failures
            llm_result = {
                "content": "",
                "reasoning": "",
                "token_counts": {},
                "error": f"{type(exc).__name__}: {exc}",
            }

        last_result = llm_result
        if not should_retry(llm_result) or attempt == MAX_ATTEMPTS:
            return llm_result

        await asyncio.sleep(RETRY_BACKOFF_SECONDS * attempt)

    return last_result or {
        "content": "",
        "reasoning": "",
        "token_counts": {},
        "error": "UNKNOWN_RERUN_ERROR",
    }


async def rerun_singleturn_row(
    module: Any,
    row: Dict[str, Any],
    tools_schema: List[Dict[str, Any]],
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    async with semaphore:
        updated = json.loads(json.dumps(row, ensure_ascii=False))
        prompt = module.build_mem0_input_prompt(
            example=updated,
            retrieved_memories=updated.get("retrieved_memories") or [],
            current_user_utterance=str(updated.get("test_utterance") or ""),
            template=module.IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE,
            context_type=CONTEXT_TYPE,
            tools_schema=tools_schema,
        )
        llm_result = await call_with_retries(module, prompt, tools_schema)
        if llm_result.get("error"):
            updated["llm_output"] = llm_result["error"]
            updated["reasoning_content"] = ""
            updated["token_counts"] = {}
        else:
            updated["llm_output"] = llm_result["content"]
            updated["reasoning_content"] = llm_result["reasoning"]
            updated["token_counts"] = llm_result["token_counts"]
        return updated


async def rerun_multiturn_row(
    module: Any,
    row: Dict[str, Any],
    tools_schema: List[Dict[str, Any]],
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    async with semaphore:
        updated = json.loads(json.dumps(row, ensure_ascii=False))
        prompt = module.build_mem0_multiturn_prompt(
            example=updated,
            retrieved_memories=updated.get("retrieved_memories") or [],
            current_user_utterance=str(updated.get("test_utterance") or ""),
            template=module.IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE,
            context_type=CONTEXT_TYPE,
            tools_schema=tools_schema,
        )
        llm_result = await call_with_retries(module, prompt, tools_schema)
        if llm_result.get("error"):
            updated["llm_output"] = llm_result["error"]
            updated["reasoning_content"] = ""
            updated["token_counts"] = {}
        else:
            updated["llm_output"] = llm_result["content"]
            updated["reasoning_content"] = llm_result["reasoning"]
            updated["token_counts"] = llm_result["token_counts"]
        return updated


async def rerun_rows(
    rows: Sequence[Dict[str, Any]],
    rerun_one,
    module: Any,
    tools_schema: List[Dict[str, Any]],
    concurrency: int,
    label: str,
) -> List[Dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [asyncio.create_task(rerun_one(module, row, tools_schema, semaphore)) for row in rows]
    results: List[Dict[str, Any]] = []
    total = len(tasks)
    for idx, task in enumerate(asyncio.as_completed(tasks), start=1):
        results.append(await task)
        print(f"[PROGRESS] {label} completed={idx}/{total}")
    return results


def merge_rows(
    original_rows: Sequence[Dict[str, Any]],
    rerun_rows: Sequence[Dict[str, Any]],
    expected_ids: set[str],
) -> List[Dict[str, Any]]:
    rerun_map = {
        str(row.get("example_id_sub", "")).strip(): row
        for row in rerun_rows
        if str(row.get("example_id_sub", "")).strip()
    }
    missing = sorted(expected_ids - set(rerun_map))
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"Missing rerun rows for {len(missing)} ids. First few: {preview}")

    merged: List[Dict[str, Any]] = []
    for row in original_rows:
        example_id_sub = str(row.get("example_id_sub", "")).strip()
        merged.append(rerun_map.get(example_id_sub, row))
    return merged


def rerun_singleturn(target_json: Path, work_dir: Path, module: Any, concurrency: int) -> None:
    original_rows = load_json(target_json)
    target_ids = set(extract_503_ids(original_rows))
    if not target_ids:
        print(f"[SKIP] No 503 rows found in {target_json}")
        return

    rows_to_rerun = [row for row in original_rows if str(row.get("example_id_sub", "")).strip() in target_ids]
    tools_schema = module.load_tools_from_file(str(SINGLE_SCHEMA_PATH))

    print(f"[INFO] singleturn rerun_count={len(rows_to_rerun)}")
    rerun_results = asyncio.run(
        rerun_rows(rows_to_rerun, rerun_singleturn_row, module, tools_schema, concurrency, "singleturn")
    )

    rerun_json = work_dir / "singleturn_503_rerun.json"
    write_json(rerun_json, rerun_results)

    merged_rows = merge_rows(original_rows, rerun_results, target_ids)
    backup_path = build_backup_path(target_json)
    shutil.copy2(target_json, backup_path)
    write_json(target_json, merged_rows)

    remaining = len(extract_503_ids(merged_rows))
    print(f"[DONE] singleturn backup={backup_path}")
    print(f"[DONE] singleturn merged={target_json} remaining_503={remaining}")


def rerun_multiturn(target_json: Path, work_dir: Path, module: Any, concurrency: int) -> None:
    original_rows = load_json(target_json)
    target_ids = set(extract_503_ids(original_rows))
    if not target_ids:
        print(f"[SKIP] No 503 rows found in {target_json}")
        return

    rows_to_rerun = [row for row in original_rows if str(row.get("example_id_sub", "")).strip() in target_ids]
    tools_schema = module.load_tools_from_file(str(MULTI_SCHEMA_PATH))

    print(f"[INFO] multiturn rerun_count={len(rows_to_rerun)}")
    rerun_results = asyncio.run(
        rerun_rows(rows_to_rerun, rerun_multiturn_row, module, tools_schema, concurrency, "multiturn")
    )

    rerun_json = work_dir / "multiturn_503_rerun.json"
    write_json(rerun_json, rerun_results)

    merged_rows = merge_rows(original_rows, rerun_results, target_ids)
    backup_path = build_backup_path(target_json)
    shutil.copy2(target_json, backup_path)
    write_json(target_json, merged_rows)

    remaining = len(extract_503_ids(merged_rows))
    print(f"[DONE] multiturn backup={backup_path}")
    print(f"[DONE] multiturn merged={target_json} remaining_503={remaining}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun only 503 rows for mem0 gemini outputs and merge results.")
    parser.add_argument("--single_json", type=Path, default=DEFAULT_SINGLE_JSON)
    parser.add_argument("--multi_json", type=Path, default=DEFAULT_MULTI_JSON)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--work_dir", type=Path, default=ROOT / "output" / "_503_rerun_work")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.environ.get("GOOGLE_API_KEY"):
        raise EnvironmentError("GOOGLE_API_KEY is required")

    single_module = load_python_module(SINGLE_SOURCE, "mem0_single_source_rerun_503")
    multi_module = load_python_module(MULTI_SOURCE, "mem0_multi_source_rerun_503")

    work_dir = args.work_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir.mkdir(parents=True, exist_ok=True)

    rerun_singleturn(args.single_json.resolve(), work_dir, single_module, args.concurrency)
    rerun_multiturn(args.multi_json.resolve(), work_dir, multi_module, args.concurrency)


if __name__ == "__main__":
    main()
