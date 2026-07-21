from __future__ import annotations

import argparse
import asyncio
import contextvars
import csv
import getpass
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
import tiktoken
from openai import AsyncOpenAI, OpenAI


LANGMEM_DIR = Path("/data/minseo/experiments6/langmem")
if str(LANGMEM_DIR) not in sys.path:
    sys.path.insert(0, str(LANGMEM_DIR))

from _compat import import_langmem_runtime  # noqa: E402
from common import (  # noqa: E402
    DEFAULT_ENCODING,
    SemanticMemory,
    build_session_messages,
    count_serialized_memory_tokens,
    get_encoding,
    list_namespace_items,
    load_chains_dataset,
    make_snapshot_item,
    materialize_namespace,
    normalize_memory_value,
    prepare_texts_for_embedding_transport,
    render_memory_line,
)
from step1_build_memory import (  # noqa: E402
    COMPACT_STRUCTURED_OUTPUT_JSON_SCHEMA,
    STRUCTURED_OUTPUT_JSON_SCHEMA,
    apply_structured_memories,
    build_compact_structured_output_prompt,
    build_structured_output_prompt,
    normalize_semantic_memory_payload,
    salvage_partial_structured_items,
    structured_output_memory_limit,
)

_, _, _, _langchain_core_embeddings_mod = import_langmem_runtime()
LangChainEmbeddings = _langchain_core_embeddings_mod.Embeddings


DEFAULT_INPUT = "/data/minseo/experiments6/data/1229_dev_6.json"
DEFAULT_OUTPUT_DIR = "/data/minseo/experiments6/rebuttal_construction_cost_gemma"
DEFAULT_MODEL = "google/gemma-3-12b-it"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 1536


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure local Mem0/LangMem memory-construction token cost with "
            "Gemma served by vLLM and OpenAI embeddings."
        )
    )
    parser.add_argument("--input_path", default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL)
    parser.add_argument("--embedding_model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--methods",
        default="langmem,mem0",
        help="Comma-separated subset of: langmem,mem0",
    )
    parser.add_argument("--start_example", type=int, default=0)
    parser.add_argument("--end_example", type=int, default=None)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--max_sessions_per_example", type=int, default=None)
    parser.add_argument("--langmem_concurrency", type=int, default=4)
    parser.add_argument("--request_timeout", type=float, default=180.0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove previous generated measurement state under output_dir before running.",
    )
    parser.add_argument(
        "--embedding_key_from_stdin",
        action="store_true",
        help="Read the OpenAI embedding API key from stdin/getpass instead of env.",
    )
    return parser.parse_args()


def read_embedding_key(args: argparse.Namespace) -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if key and not args.embedding_key_from_stdin:
        return key
    if args.embedding_key_from_stdin:
        if sys.stdin.isatty():
            key = getpass.getpass("OpenAI embedding API key: ")
        else:
            key = sys.stdin.readline().strip()
    if not key:
        raise RuntimeError(
            "OpenAI embedding API key is required for text-embedding-3-small."
        )
    return key


def text_tokens(text: Any, encoding) -> int:
    if text is None:
        return 0
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return 0
    return len(encoding.encode(text))


def messages_tokens(messages: Sequence[Dict[str, Any]], encoding) -> int:
    return sum(text_tokens(message.get("content", ""), encoding) for message in messages)


def usage_value(usage: Any, name: str) -> Optional[int]:
    if usage is None:
        return None
    if isinstance(usage, dict):
        value = usage.get(name)
    else:
        value = getattr(usage, name, None)
    return int(value) if value is not None else None


@dataclass
class SessionCost:
    method: str
    example_index: int
    example_id: str
    session_index: int
    dialogue_id: str
    status: str = "OK"
    error: str = ""
    llm_call_count: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_total_tokens: int = 0
    embedding_call_count: int = 0
    embedding_input_tokens: int = 0
    embedding_total_tokens: int = 0
    raw_session_input_tokens: int = 0
    extracted_memory_count: int = 0
    stored_memory_count_after_session: int = 0
    stored_memory_content_tokens_after_session: int = 0
    stored_memory_serialized_tokens_after_session: int = 0
    elapsed_seconds: float = 0.0


@dataclass
class ActiveContext:
    method: str = ""
    example_index: int = -1
    example_id: str = ""
    session_index: int = -1


@dataclass
class TokenTracker:
    encoding: Any
    session_costs: Dict[tuple[str, int, int], SessionCost] = field(default_factory=dict)
    last_active_context: ActiveContext = field(default_factory=ActiveContext)
    allow_empty_context_fallback: bool = False
    active_var: contextvars.ContextVar[ActiveContext] = field(
        default_factory=lambda: contextvars.ContextVar(
            "memory_construction_active_context",
            default=ActiveContext(),
        )
    )

    def set_active(self, method: str, example_index: int, example_id: str, session_index: int) -> None:
        active = ActiveContext(method, example_index, example_id, session_index)
        self.last_active_context = active
        self.active_var.set(active)

    def current_cost(self) -> SessionCost:
        active = self.active_var.get()
        if not active.method and self.allow_empty_context_fallback:
            active = self.last_active_context
        key = (active.method, active.example_index, active.session_index)
        if key not in self.session_costs:
            self.session_costs[key] = SessionCost(
                method=active.method,
                example_index=active.example_index,
                example_id=active.example_id,
                session_index=active.session_index,
                dialogue_id="",
            )
        return self.session_costs[key]

    def add_llm_usage(
        self,
        messages: Sequence[Dict[str, Any]],
        content: str,
        usage: Any,
    ) -> None:
        cost = self.current_cost()
        prompt_tokens = usage_value(usage, "prompt_tokens")
        completion_tokens = usage_value(usage, "completion_tokens")
        total_tokens = usage_value(usage, "total_tokens")
        if prompt_tokens is None:
            prompt_tokens = messages_tokens(messages, self.encoding)
        if completion_tokens is None:
            completion_tokens = text_tokens(content, self.encoding)
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens
        cost.llm_call_count += 1
        cost.llm_input_tokens += prompt_tokens
        cost.llm_output_tokens += completion_tokens
        cost.llm_total_tokens += total_tokens

    def add_embedding_usage(
        self,
        texts: Sequence[str],
        usage: Any = None,
    ) -> None:
        cost = self.current_cost()
        prompt_tokens = usage_value(usage, "prompt_tokens")
        total_tokens = usage_value(usage, "total_tokens")
        if prompt_tokens is None:
            prompt_tokens = sum(text_tokens(text, self.encoding) for text in texts)
        if total_tokens is None:
            total_tokens = prompt_tokens
        cost.embedding_call_count += 1
        cost.embedding_input_tokens += prompt_tokens
        cost.embedding_total_tokens += total_tokens


class CountingOpenAIEmbeddings(LangChainEmbeddings):
    def __init__(
        self,
        embedding_model: str,
        api_key: str,
        tracker: TokenTracker,
        timeout: float,
    ) -> None:
        self.embedding_model = embedding_model
        self.tracker = tracker
        self.sync_client = OpenAI(api_key=api_key, timeout=timeout)
        self.async_client = AsyncOpenAI(api_key=api_key, timeout=timeout)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        prepared = prepare_texts_for_embedding_transport(texts)
        response = self.sync_client.embeddings.create(
            model=self.embedding_model,
            input=prepared,
        )
        self.tracker.add_embedding_usage(prepared, response.usage)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        prepared = prepare_texts_for_embedding_transport(texts)
        response = await self.async_client.embeddings.create(
            model=self.embedding_model,
            input=prepared,
        )
        self.tracker.add_embedding_usage(prepared, response.usage)
        return [item.embedding for item in response.data]

    async def aembed_query(self, text: str) -> List[float]:
        return (await self.aembed_documents([text]))[0]


def make_langmem_store(embedding_model: str, embedding_key: str, tracker: TokenTracker, timeout: float):
    _, langgraph_store_memory_mod, _, _ = import_langmem_runtime()
    embeddings = CountingOpenAIEmbeddings(
        embedding_model=embedding_model,
        api_key=embedding_key,
        tracker=tracker,
        timeout=timeout,
    )
    return langgraph_store_memory_mod.InMemoryStore(
        index={"dims": EMBEDDING_DIMS, "embed": embeddings}
    )


def extract_content(response: Any) -> str:
    return response.choices[0].message.content or ""


async def extract_langmem_structured_memories(
    client: AsyncOpenAI,
    model: str,
    messages: List[Dict[str, str]],
    tracker: TokenTracker,
) -> List[Dict[str, Any]]:
    prompt = build_structured_output_prompt(
        messages,
        max_memories=structured_output_memory_limit(model),
    )
    request_messages = [{"role": "user", "content": prompt}]
    parsed: Optional[Dict[str, Any]] = None
    raw_attempts: List[str] = []
    for max_tokens in (768, 1536, 2048):
        response = await client.chat.completions.create(
            model=model,
            messages=request_messages,
            temperature=0.0,
            max_tokens=max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": STRUCTURED_OUTPUT_JSON_SCHEMA,
            },
        )
        raw_content = extract_content(response)
        raw_attempts.append(raw_content)
        tracker.add_llm_usage(request_messages, raw_content, response.usage)
        try:
            parsed = json.loads(raw_content)
            break
        except json.JSONDecodeError:
            continue
    if parsed is None:
        salvaged_items: List[Dict[str, Any]] = []
        for raw_content in reversed(raw_attempts):
            candidate_items = salvage_partial_structured_items(raw_content)
            if len(candidate_items) > len(salvaged_items):
                salvaged_items = candidate_items
        if salvaged_items:
            parsed = {"items": salvaged_items}
        else:
            return await extract_langmem_compact_memories(
                client,
                model,
                messages,
                tracker,
            )
    memories: List[Dict[str, Any]] = []
    for item in parsed.get("items", []):
        normalized = normalize_semantic_memory_payload(item)
        if normalized.get("content"):
            memories.append(normalized)
    return memories


async def extract_langmem_compact_memories(
    client: AsyncOpenAI,
    model: str,
    messages: List[Dict[str, str]],
    tracker: TokenTracker,
) -> List[Dict[str, Any]]:
    prompt = build_compact_structured_output_prompt(messages)
    request_messages = [{"role": "user", "content": prompt}]
    for max_tokens in (192, 256, 384):
        response = await client.chat.completions.create(
            model=model,
            messages=request_messages,
            temperature=0.0,
            max_tokens=max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": COMPACT_STRUCTURED_OUTPUT_JSON_SCHEMA,
            },
        )
        raw_content = extract_content(response)
        tracker.add_llm_usage(request_messages, raw_content, response.usage)
        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError:
            continue

        memories: List[Dict[str, Any]] = []
        for item in parsed.get("items", []):
            normalized = normalize_semantic_memory_payload(item)
            if normalized.get("content"):
                memories.append(normalized)
        if memories:
            return memories
    return []


def langmem_memory_token_counts(store: Any, namespace: tuple[str, ...], encoding) -> tuple[int, int, int]:
    items = [make_snapshot_item(item) for item in list_namespace_items(store, namespace)]
    content_tokens = sum(
        text_tokens(render_memory_line(item.get("value")), encoding)
        for item in items
    )
    serialized_tokens = count_serialized_memory_tokens(items, encoding)
    return len(items), content_tokens, serialized_tokens


async def run_langmem(
    rows: List[Dict[str, Any]],
    args: argparse.Namespace,
    embedding_key: str,
    tracker: TokenTracker,
    detail_rows: List[SessionCost],
) -> None:
    client = AsyncOpenAI(
        api_key="EMPTY",
        base_url=args.base_url,
        timeout=args.request_timeout,
    )
    store = make_langmem_store(
        args.embedding_model,
        embedding_key,
        tracker,
        args.request_timeout,
    )
    semaphore = asyncio.Semaphore(args.langmem_concurrency)
    store_lock = asyncio.Lock()
    encoding = tracker.encoding

    async def process_example(example_index: int, example: Dict[str, Any]) -> None:
        async with semaphore:
            example_id = str(example.get("example_id", f"example_{example_index}"))
            namespace = materialize_namespace(example_id)
            sessions = example.get("sessions", [])
            if args.max_sessions_per_example is not None:
                sessions = sessions[: args.max_sessions_per_example]
            for session_index, session in enumerate(sessions, start=1):
                messages = build_session_messages(session)
                dialogue_id = str(session.get("dialogue_id", ""))
                cost = SessionCost(
                    method="langmem",
                    example_index=example_index,
                    example_id=example_id,
                    session_index=session_index,
                    dialogue_id=dialogue_id,
                    raw_session_input_tokens=messages_tokens(messages, encoding),
                )
                tracker.session_costs[("langmem", example_index, session_index)] = cost
                if not messages:
                    cost.status = "SKIPPED_EMPTY_SESSION"
                    detail_rows.append(cost)
                    continue
                started = time.time()
                tracker.set_active("langmem", example_index, example_id, session_index)
                try:
                    extracted_memories = await extract_langmem_structured_memories(
                        client,
                        args.model,
                        messages,
                        tracker,
                    )
                    async with store_lock:
                        tracker.set_active("langmem", example_index, example_id, session_index)
                        await asyncio.to_thread(
                            apply_structured_memories,
                            store,
                            namespace,
                            extracted_memories,
                            None,
                            example_id,
                            session_index,
                        )
                        (
                            memory_count,
                            content_tokens,
                            serialized_tokens,
                        ) = langmem_memory_token_counts(store, namespace, encoding)
                    cost.extracted_memory_count = len(extracted_memories)
                    cost.stored_memory_count_after_session = memory_count
                    cost.stored_memory_content_tokens_after_session = content_tokens
                    cost.stored_memory_serialized_tokens_after_session = serialized_tokens
                except Exception as exc:
                    cost.status = "ERROR"
                    cost.error = str(exc)
                finally:
                    cost.elapsed_seconds = time.time() - started
                    detail_rows.append(cost)

    tasks = [process_example(i, row) for i, row in enumerate(rows, start=args.start_example)]
    await asyncio.gather(*tasks)


def build_mem0_messages(session: Dict[str, Any]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    for turn in session.get("dialogue", []):
        role = str(turn.get("role", "")).lower()
        content = turn.get("message") or turn.get("content") or ""
        if role and content:
            messages.append({"role": role, "content": content})
    api_calls = session.get("api_call", [])
    if api_calls:
        messages.append(
            {
                "role": "assistant",
                "content": f"[System Summary] API Calls executed in this session: {str(api_calls)}",
            }
        )
    return messages


def normalize_mem0_results(response: Any) -> List[Dict[str, Any]]:
    if isinstance(response, dict) and isinstance(response.get("results"), list):
        return response["results"]
    if isinstance(response, list):
        return response
    return []


def mem0_memory_token_counts(memory: Any, user_id: str, encoding) -> tuple[int, int, int]:
    memories = normalize_mem0_results(memory.get_all(user_id=user_id, limit=1000))
    content_tokens = sum(text_tokens(item.get("memory", ""), encoding) for item in memories)
    serialized_tokens = sum(
        text_tokens(json.dumps(item, ensure_ascii=False, sort_keys=True), encoding)
        for item in memories
    )
    return len(memories), content_tokens, serialized_tokens


def make_mem0_memory(args: argparse.Namespace, embedding_key: str, tracker: TokenTracker):
    tracker.allow_empty_context_fallback = True
    output_dir = Path(args.output_dir)
    mem0_state_dir = output_dir / "mem0_state"
    mem0_state_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MEM0_DIR"] = str(mem0_state_dir / "mem0_home")

    from mem0 import Memory

    def response_callback(_llm: Any, response: Any, params: Dict[str, Any]) -> None:
        messages = params.get("messages", [])
        content = extract_content(response)
        tracker.add_llm_usage(messages, content, response.usage)

    config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "mem0_rebuttal_gemma",
                "path": str(mem0_state_dir / "qdrant"),
                "embedding_model_dims": EMBEDDING_DIMS,
            },
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": args.model,
                "api_key": "EMPTY",
                "openai_base_url": args.base_url,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 1,
                "max_tokens": 2000,
                "response_callback": response_callback,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": args.embedding_model,
                "api_key": embedding_key,
                "embedding_dims": EMBEDDING_DIMS,
            },
        },
        "history_db_path": str(mem0_state_dir / "history.db"),
    }
    memory = Memory.from_config(config)
    original_embed = memory.embedding_model.embed

    def counted_embed(text: str, memory_action: Optional[str] = None):
        normalized = (text or "").replace("\n", " ")
        tracker.add_embedding_usage([normalized], usage=None)
        return original_embed(text, memory_action=memory_action)

    memory.embedding_model.embed = counted_embed
    return memory


def run_mem0(
    rows: List[Dict[str, Any]],
    args: argparse.Namespace,
    embedding_key: str,
    tracker: TokenTracker,
    detail_rows: List[SessionCost],
) -> None:
    memory = make_mem0_memory(args, embedding_key, tracker)
    encoding = tracker.encoding
    for example_index, example in enumerate(rows, start=args.start_example):
        example_id = str(example.get("example_id", f"example_{example_index}"))
        sessions = example.get("sessions", [])
        if args.max_sessions_per_example is not None:
            sessions = sessions[: args.max_sessions_per_example]
        for session_index, session in enumerate(sessions, start=1):
            messages = build_mem0_messages(session)
            dialogue_id = str(session.get("dialogue_id", ""))
            cost = SessionCost(
                method="mem0",
                example_index=example_index,
                example_id=example_id,
                session_index=session_index,
                dialogue_id=dialogue_id,
                raw_session_input_tokens=messages_tokens(messages, encoding),
            )
            tracker.session_costs[("mem0", example_index, session_index)] = cost
            if not messages:
                cost.status = "SKIPPED_EMPTY_SESSION"
                detail_rows.append(cost)
                continue
            started = time.time()
            tracker.set_active("mem0", example_index, example_id, session_index)
            try:
                result = memory.add(messages, user_id=example_id, infer=True)
                extracted = normalize_mem0_results(result)
                (
                    memory_count,
                    content_tokens,
                    serialized_tokens,
                ) = mem0_memory_token_counts(memory, example_id, encoding)
                cost.extracted_memory_count = len(extracted)
                cost.stored_memory_count_after_session = memory_count
                cost.stored_memory_content_tokens_after_session = content_tokens
                cost.stored_memory_serialized_tokens_after_session = serialized_tokens
            except Exception as exc:
                cost.status = "ERROR"
                cost.error = str(exc)
            finally:
                cost.elapsed_seconds = time.time() - started
                detail_rows.append(cost)


def selected_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    df = load_chains_dataset(args.input_path)
    rows = df.to_dict("records")[args.start_example : args.end_example]
    if args.max_examples is not None:
        rows = rows[: args.max_examples]
    return rows


def write_detail_csv(path: Path, rows: List[SessionCost]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(SessionCost.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r.method, r.example_index, r.session_index)):
            writer.writerow({name: getattr(row, name) for name in fieldnames})


def summarize_dialogues(detail_rows: List[SessionCost]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, int, str], List[SessionCost]] = {}
    for row in detail_rows:
        grouped.setdefault((row.method, row.example_index, row.example_id), []).append(row)

    dialogue_rows: List[Dict[str, Any]] = []
    for (method, example_index, example_id), sessions in grouped.items():
        sessions = sorted(sessions, key=lambda r: r.session_index)
        ok_sessions = [row for row in sessions if row.status == "OK"]
        final = sessions[-1]
        dialogue_rows.append(
            {
                "method": method,
                "example_index": example_index,
                "example_id": example_id,
                "sessions": len(sessions),
                "ok_sessions": len(ok_sessions),
                "failed_sessions": sum(1 for row in sessions if row.status == "ERROR"),
                "llm_call_count": sum(row.llm_call_count for row in sessions),
                "llm_input_tokens": sum(row.llm_input_tokens for row in sessions),
                "llm_output_tokens": sum(row.llm_output_tokens for row in sessions),
                "llm_total_tokens": sum(row.llm_total_tokens for row in sessions),
                "embedding_call_count": sum(row.embedding_call_count for row in sessions),
                "embedding_input_tokens": sum(row.embedding_input_tokens for row in sessions),
                "embedding_total_tokens": sum(row.embedding_total_tokens for row in sessions),
                "raw_session_input_tokens": sum(row.raw_session_input_tokens for row in sessions),
                "extracted_memory_count": sum(row.extracted_memory_count for row in sessions),
                "final_stored_memory_count": final.stored_memory_count_after_session,
                "final_stored_memory_content_tokens": final.stored_memory_content_tokens_after_session,
                "final_stored_memory_serialized_tokens": final.stored_memory_serialized_tokens_after_session,
                "elapsed_seconds": sum(row.elapsed_seconds for row in sessions),
                "status": "OK" if all(row.status in {"OK", "SKIPPED_EMPTY_SESSION"} for row in sessions) else "ERROR",
            }
        )
    return sorted(dialogue_rows, key=lambda r: (r["method"], r["example_index"]))


def write_dialogue_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_method(values: List[Dict[str, Any]]) -> Dict[str, Any]:
    def stats(key: str) -> Dict[str, Optional[float]]:
        nums = [float(row[key]) for row in values if row.get(key) is not None]
        if not nums:
            return {"mean": None, "median": None, "min": None, "max": None}
        return {
            "mean": mean(nums),
            "median": median(nums),
            "min": min(nums),
            "max": max(nums),
        }

    keys = [
        "llm_input_tokens",
        "llm_output_tokens",
        "llm_total_tokens",
        "embedding_input_tokens",
        "embedding_total_tokens",
        "raw_session_input_tokens",
        "final_stored_memory_content_tokens",
        "final_stored_memory_serialized_tokens",
        "llm_call_count",
        "embedding_call_count",
        "elapsed_seconds",
    ]
    summary = {
        "dialogues": len(values),
        "dialogues_ok": sum(1 for row in values if row["status"] == "OK"),
        "dialogues_failed": sum(1 for row in values if row["status"] == "ERROR"),
    }
    for key in keys:
        summary[key] = stats(key)
    return summary


def write_summary(path: Path, dialogue_rows: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for row in dialogue_rows:
        by_method.setdefault(row["method"], []).append(row)
    summary = {
        "input_path": args.input_path,
        "model": args.model,
        "base_url": args.base_url,
        "embedding_model": args.embedding_model,
        "start_example": args.start_example,
        "end_example": args.end_example,
        "max_examples": args.max_examples,
        "max_sessions_per_example": args.max_sessions_per_example,
        "methods": {method: summarize_method(rows) for method, rows in sorted(by_method.items())},
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def write_markdown(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# Gemma Local Memory Construction Cost",
        "",
        f"- LLM: `{summary['model']}` via `{summary['base_url']}`",
        f"- Embedding: `{summary['embedding_model']}`",
        "",
        "| method | dialogues | LLM input/dialogue | LLM output/dialogue | embedding input/dialogue | final memory content tokens | final memory serialized tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, stats in summary["methods"].items():
        lines.append(
            "| {method} | {dialogues} | {llm_in:.1f} | {llm_out:.1f} | {emb_in:.1f} | {mem_content:.1f} | {mem_serialized:.1f} |".format(
                method=method,
                dialogues=stats["dialogues"],
                llm_in=stats["llm_input_tokens"]["mean"] or 0.0,
                llm_out=stats["llm_output_tokens"]["mean"] or 0.0,
                emb_in=stats["embedding_input_tokens"]["mean"] or 0.0,
                mem_content=stats["final_stored_memory_content_tokens"]["mean"] or 0.0,
                mem_serialized=stats["final_stored_memory_serialized_tokens"]["mean"] or 0.0,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    methods = {method.strip() for method in args.methods.split(",") if method.strip()}
    unknown = methods - {"langmem", "mem0"}
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        for generated in (
            output_dir / "mem0_state",
            output_dir / "session_detail.csv",
            output_dir / "dialogue_summary.csv",
            output_dir / "summary.json",
            output_dir / "summary.md",
        ):
            if generated.is_dir():
                shutil.rmtree(generated)
            elif generated.exists():
                generated.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_key = read_embedding_key(args)
    rows = selected_rows(args)
    encoding = get_encoding(DEFAULT_ENCODING)
    tracker = TokenTracker(encoding=encoding)
    detail_rows: List[SessionCost] = []

    started = time.time()
    if "langmem" in methods:
        asyncio.run(run_langmem(rows, args, embedding_key, tracker, detail_rows))
    if "mem0" in methods:
        run_mem0(rows, args, embedding_key, tracker, detail_rows)

    detail_csv = output_dir / "session_detail.csv"
    dialogue_csv = output_dir / "dialogue_summary.csv"
    summary_json = output_dir / "summary.json"
    summary_md = output_dir / "summary.md"

    write_detail_csv(detail_csv, detail_rows)
    dialogue_rows = summarize_dialogues(detail_rows)
    write_dialogue_csv(dialogue_csv, dialogue_rows)
    summary = write_summary(summary_json, dialogue_rows, args)
    summary["duration_seconds"] = time.time() - started
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary_md, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
