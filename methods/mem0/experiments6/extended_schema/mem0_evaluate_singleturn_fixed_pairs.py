from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
RELEASE_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = str(RELEASE_ROOT / "src" / "baselines" / "mem0")
for candidate in (ROOT_DIR, Path(SOURCE_DIR)):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from common_fixed_pairs import (
    build_example_lookup,
    get_singleturn_ground_truth,
    get_singleturn_utterance,
    get_sub_idx,
    load_fixed_pairs_items,
    load_python_module,
)


SOURCE_MODULE_PATH = str(Path(SOURCE_DIR) / "step2_evaluate_singleturn.py")
source = load_python_module(
    SOURCE_MODULE_PATH,
    "extended_schema_mem0_single_source",
    extra_sys_path=SOURCE_DIR,
)


EXTENDED_INPUT_PATH = str(RELEASE_ROOT / "data" / "1229_dev_6.json")
EXTENDED_QUERY_PATH = str(RELEASE_ROOT / "query_new_single.json")
EXTENDED_PREF_LIST_PATH = str(RELEASE_ROOT / "pref_list_extended.json")
EXTENDED_PREF_GROUP_PATH = str(RELEASE_ROOT / "pref_group_extended.json")
EXTENDED_SCHEMA_PATH = str(RELEASE_ROOT / "schema_easy_extended.json")


def limit_prepared_items(
    prepared_items: List[Dict[str, Any]],
    max_queries: Optional[int],
) -> List[Dict[str, Any]]:
    if max_queries is None or max_queries <= 0:
        return prepared_items
    total_items = len(prepared_items)
    if total_items <= max_queries:
        return prepared_items
    if max_queries == 1:
        selected_indices = [0]
    else:
        selected_indices = [((total_items - 1) * idx) // (max_queries - 1) for idx in range(max_queries)]
    return [prepared_items[idx] for idx in selected_indices]


def build_fixed_prepared_items(input_path: str, fixed_pairs_path: str) -> List[Dict[str, Any]]:
    df = source.load_chains_dataset(input_path)
    fixed_items = load_fixed_pairs_items(fixed_pairs_path)
    example_lookup = build_example_lookup([row.to_dict() for _, row in df.iterrows()])

    prepared_items: List[Dict[str, Any]] = []
    missing_example_ids: List[str] = []

    for item in fixed_items:
        example_id = str(item.get("example_id", "")).strip()
        original_ex = example_lookup.get(example_id)
        if original_ex is None:
            missing_example_ids.append(example_id)
            continue

        prepared_items.append(
            {
                "original_ex": original_ex,
                "utterance": get_singleturn_utterance(item),
                "ground_truth": get_singleturn_ground_truth(item),
                "sub_idx": get_sub_idx(item),
            }
        )

    if missing_example_ids:
        preview = ", ".join(missing_example_ids[:10])
        raise ValueError(
            f"{len(missing_example_ids)} fixed pairs did not match the input dataset. "
            f"First missing example_id values: {preview}"
        )

    return prepared_items


async def process_with_llm_async(
    input_path: str,
    output_path: str,
    log_path: str,
    query_map_path: str,
    pref_list_path: str,
    pref_group_path: str,
    tools_schema_path: str,
    prompt_template: str,
    context_type: str,
    pref_type: str,
    model_name: str,
    fixed_pairs_path: Optional[str],
    concurrency: int = 10,
    reasoning_effort: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    max_queries: Optional[int] = None,
) -> None:
    if not fixed_pairs_path and not base_url and max_queries is None:
        await source.process_with_llm_async(
            input_path=input_path,
            output_path=output_path,
            log_path=log_path,
            query_map_path=query_map_path,
            pref_list_path=pref_list_path,
            pref_group_path=pref_group_path,
            tools_schema_path=tools_schema_path,
            prompt_template=prompt_template,
            context_type=context_type,
            pref_type=pref_type,
            model_name=model_name,
            concurrency=concurrency,
            reasoning_effort=reasoning_effort,
        )
        return

    df = source.load_chains_dataset(input_path)
    tools_schema = source.load_tools_from_file(tools_schema_path)

    if fixed_pairs_path:
        prepared_items = build_fixed_prepared_items(input_path, fixed_pairs_path)
        if max_queries is not None:
            print("[Info] fixed_pairs_path is active; ignoring max_queries sampling.")
    else:
        query_map = source.load_query_map(query_map_path)
        prepared_items = []
        skipped_count = 0
        for _, row in df.iterrows():
            original_ex = row.to_dict()
            if pref_type == "easy":
                if not original_ex.get("api_calls"):
                    skipped_count += 1
                    continue
            elif pref_type in {"medium", "hard"}:
                if not original_ex.get("api_calls_pref"):
                    skipped_count += 1
                    continue

            pairs_list = source.assign_user_utterances(
                pref_list_path,
                original_ex,
                query_map,
                pref_type,
                pref_group_path,
            )
            if not pairs_list:
                skipped_count += 1
                continue

            for sub_idx, (utterance, ground_truth) in enumerate(pairs_list):
                prepared_items.append(
                    {
                        "original_ex": original_ex,
                        "utterance": utterance,
                        "ground_truth": ground_truth,
                        "sub_idx": sub_idx,
                    }
                )

        prepared_items = limit_prepared_items(prepared_items, max_queries)
        print(f"Total tasks after optional sampling: {len(prepared_items)}. Skipped dialogues: {skipped_count}")

    openai_client = None
    if "gemini" not in model_name.lower():
        client_args: Dict[str, str] = {}
        if base_url:
            print(f"[Info] Using Custom/vLLM Base URL: {base_url}")
            client_args["base_url"] = base_url
            client_args["api_key"] = api_key if api_key and api_key != "ENV" else "EMPTY"
        elif os.environ.get("OPENAI_API_KEY"):
            client_args["api_key"] = os.environ.get("OPENAI_API_KEY")

        if "api_key" in client_args:
            openai_client = source.AsyncOpenAI(**client_args)

    memory_client = source.MemoryClient(api_key=os.environ.get("MEM0_API_KEY"))

    print(
        f"Starting ASYNC fixed-pairs process... (Model: {model_name}) "
        f"| Reasoning: {reasoning_effort} | Prepared Items: {len(prepared_items)}"
    )

    tasks = []
    semaphore = asyncio.Semaphore(concurrency)
    file_lock = asyncio.Lock()
    pbar = tqdm.tqdm(total=len(prepared_items), desc="Processing Async")

    for item in prepared_items:
        tasks.append(
            asyncio.create_task(
                source.process_single_item(
                    original_ex=item["original_ex"],
                    memory_client=memory_client,
                    utterance=item["utterance"],
                    ground_truth=item["ground_truth"],
                    sub_idx=item["sub_idx"],
                    model_name=model_name,
                    prompt_template=prompt_template,
                    context_type=context_type,
                    openai_client=openai_client,
                    tools_schema=tools_schema,
                    log_path=log_path,
                    semaphore=semaphore,
                    file_lock=file_lock,
                    pbar=pbar,
                    reasoning_effort=reasoning_effort,
                )
            )
        )

    processed_data = await asyncio.gather(*tasks)
    pbar.close()

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        source.json.dump(processed_data, f, indent=4, ensure_ascii=False)
    print(f"Saved -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default=EXTENDED_INPUT_PATH)
    parser.add_argument("--output_path", type=str, default="output_mem0_eval.json")
    parser.add_argument("--log_path", type=str, default="process_mem0.log")
    parser.add_argument("--query_path", type=str, default=EXTENDED_QUERY_PATH)
    parser.add_argument("--pref_list_path", type=str, default=EXTENDED_PREF_LIST_PATH)
    parser.add_argument("--pref_group_path", type=str, default=EXTENDED_PREF_GROUP_PATH)
    parser.add_argument("--tools_schema_path", type=str, default=EXTENDED_SCHEMA_PATH)
    parser.add_argument("--fixed_pairs_path", type=str, default=None)
    parser.add_argument("--pref_type", type=str, choices=["medium", "easy", "hard"], required=True)
    parser.add_argument(
        "--context_type",
        type=str,
        choices=["memory_only", "memory_diag", "memory_api"],
        required=True,
    )
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        choices=["minimal", "low", "medium", "high"],
        default=None,
    )
    parser.add_argument("--base_url", type=str, default="")
    parser.add_argument("--api_key", type=str, default="ENV")
    parser.add_argument("--max_queries", type=int, default=None)
    args = parser.parse_args()

    if not os.environ.get("MEM0_API_KEY"):
        raise SystemExit("[Error] MEM0_API_KEY environment variable is not set.")

    asyncio.run(
        process_with_llm_async(
            input_path=args.input_path,
            output_path=args.output_path,
            log_path=args.log_path,
            query_map_path=args.query_path,
            pref_list_path=args.pref_list_path,
            pref_group_path=args.pref_group_path,
            tools_schema_path=args.tools_schema_path,
            prompt_template=source.IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE,
            context_type=args.context_type,
            pref_type=args.pref_type,
            model_name=args.model_name,
            fixed_pairs_path=args.fixed_pairs_path,
            concurrency=args.concurrency,
            reasoning_effort=args.reasoning_effort,
            base_url=args.base_url or None,
            api_key=args.api_key,
            max_queries=args.max_queries,
        )
    )


if __name__ == "__main__":
    main()
