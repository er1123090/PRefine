"""Small runner around the inference primitives preserved from experiment4."""

from __future__ import annotations

import asyncio
import copy
import json
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

RUNTIME_DIR = Path(__file__).resolve().parent
if str(RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_DIR))

import common  # type: ignore  # noqa: E402


RetrievalResult = Tuple[List[Dict[str, Any]], str]
Retriever = Callable[[Dict[str, Any], str], Awaitable[RetrievalResult]]


async def empty_retriever(
    example: Dict[str, Any], utterance: str
) -> RetrievalResult:
    del example, utterance
    return [], ""


async def run_inference(
    *,
    prepared_items: Sequence[Dict[str, Any]],
    output_path: str,
    log_path: str,
    prompt_template: str,
    context_type: str,
    tools_schema_path: str,
    model_name: str,
    concurrency: int,
    reasoning_effort: Optional[str],
    method_name: str,
    request_timeout_seconds: Optional[float] = None,
    client_max_retries: Optional[int] = None,
    retriever: Retriever = empty_retriever,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> None:
    tools_schema = common.load_tools_from_file(tools_schema_path)
    openai_client = None
    if not (common.is_gemini_model(model_name) and not base_url):
        openai_client = common.create_async_openai_client(
            api_key=api_key,
            base_url=base_url,
            allow_empty=bool(base_url),
            timeout=request_timeout_seconds,
            max_retries=client_max_retries,
        )

    semaphore = asyncio.Semaphore(concurrency)
    log_lock = asyncio.Lock()
    results: List[Optional[Dict[str, Any]]] = [None] * len(prepared_items)
    common.ensure_parent_dir(log_path)
    Path(log_path).write_text("", encoding="utf-8")

    async def process(index: int, item: Dict[str, Any]) -> None:
        async with semaphore:
            example = item["original_ex"]
            utterance = item["utterance"]
            ground_truth = item["ground_truth"]
            sub_idx = item["sub_idx"]
            example_id = str(example.get("example_id", "unknown_user"))
            retrieved, retrieved_text = await retriever(example, utterance)
            prompt = common.build_memory_prompt(
                example=example,
                retrieved_memories_text=retrieved_text,
                current_user_utterance=utterance,
                template=prompt_template,
                context_type=context_type,
                tools_schema=tools_schema,
            )
            response = await common.call_chat_async(
                prompt=prompt,
                model_name=model_name,
                openai_client=openai_client,
                api_key=api_key,
                reasoning_effort=reasoning_effort,
            )
            error = response.get("error")
            output = error or response.get("content", "")
            reasoning = "" if error else response.get("reasoning", "")
            token_counts = {} if error else response.get("token_counts", {})

            record = {
                "example_id": example_id,
                "example_id_sub": f"{example_id}_{sub_idx}",
                "method": method_name,
                "model_name": model_name,
                "context_type": context_type,
                "test_utterance": utterance,
                "reference_ground_truth": ground_truth,
                "retrieved_memories": retrieved,
                "status": "ERROR" if error else "OK",
                "error": error,
                "model_input": prompt,
                "llm_output": output,
                "reasoning_content": reasoning,
                "token_counts": token_counts,
            }
            async with log_lock:
                with Path(log_path).open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            result = copy.deepcopy(example)
            result.update(record)
            results[index] = result

    await asyncio.gather(
        *(process(index, item) for index, item in enumerate(prepared_items))
    )
    common.write_json(output_path, [row for row in results if row is not None])


def prepare_items(
    *,
    turn: str,
    input_path: str,
    query_path: str,
    pref_list_path: str,
    pref_group_path: str,
    pref_type: str,
    max_queries: Optional[int],
) -> List[Dict[str, Any]]:
    if turn == "single":
        return common.prepare_singleturn_items(
            input_path=input_path,
            query_path=query_path,
            pref_list_path=pref_list_path,
            pref_group_path=pref_group_path,
            pref_type=pref_type,
            max_queries=max_queries,
        )
    return common.prepare_multiturn_items(
        input_path=input_path,
        multiturn_path=query_path,
        pref_list_path=pref_list_path,
        pref_group_path=pref_group_path,
        pref_type=pref_type,
        max_queries=max_queries,
    )
