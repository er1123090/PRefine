from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from typing import Any, Dict, List

from common import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INPUT_PATH,
    SemanticMemory,
    build_session_messages,
    count_serialized_memory_tokens,
    create_async_openai_client,
    create_langmem_runtime,
    default_run_log_path,
    export_snapshot_record,
    get_encoding,
    last_user_utterance,
    list_namespace_items,
    make_runtime_config,
    materialize_namespace,
    make_snapshot_item,
    normalize_memory_value,
    now_iso,
    sanitize_text_for_transport,
    session_input_token_count,
    setup_logger,
    write_json,
    write_jsonl,
)
from common import load_chains_dataset


STRUCTURED_OUTPUT_CATEGORY_ENUM = [
    "explicit_preference",
    "implicit_preference",
    "behavior_pattern",
    "api_outcome",
    "profile_fact",
]


STRUCTURED_OUTPUT_JSON_SCHEMA = {
    "name": "semantic_memories",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "category": {
                            "type": "string",
                            "enum": STRUCTURED_OUTPUT_CATEGORY_ENUM,
                        },
                        "domain": {"type": ["string", "null"]},
                        "slot": {"type": ["string", "null"]},
                        "value": {"type": ["string", "null"]},
                        "context": {"type": ["string", "null"]},
                    },
                    "required": [
                        "content",
                        "category",
                        "domain",
                        "slot",
                        "value",
                        "context",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    },
}


COMPACT_STRUCTURED_OUTPUT_JSON_SCHEMA = {
    "name": "compact_semantic_memories",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "category": {
                            "type": "string",
                            "enum": STRUCTURED_OUTPUT_CATEGORY_ENUM,
                        },
                    },
                    "required": ["content", "category"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a LangMem semantic-memory snapshot from the experiments5 dataset."
    )
    parser.add_argument("--input_path", type=str, default=DEFAULT_INPUT_PATH)
    parser.add_argument(
        "--output_path",
        type=str,
        default="/data/minseo/experiments5/methods/langmem/memory_snapshots/langmem_1229_dev_6.jsonl",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        default="/data/minseo/experiments5/methods/langmem/memory_snapshots/langmem_1229_dev_6.manifest.json",
    )
    parser.add_argument("--memory_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--embedding_model", type=str, default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--embedding_base_url", type=str, default=None)
    parser.add_argument("--embedding_api_key", type=str, default=None)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument(
        "--memory_backend",
        type=str,
        choices=["auto", "langmem", "structured_output"],
        default="auto",
        help="Memory extraction backend. 'auto' uses structured output for models that do not tool-call reliably.",
    )
    parser.add_argument(
        "--langmem_memory_schema",
        type=str,
        choices=["semantic", "string"],
        default="semantic",
        help="Only used with the langmem backend. 'semantic' stores the custom schema, 'string' stores plain string memories.",
    )
    parser.add_argument(
        "--langmem_prompt_style",
        type=str,
        choices=["custom", "default"],
        default="custom",
        help="Only used with the langmem backend. 'default' uses LangMem's built-in instructions.",
    )
    parser.add_argument("--start_example", type=int, default=0)
    parser.add_argument("--end_example", type=int, default=None)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--run_log_path", type=str, default=None)
    parser.add_argument(
        "--disable_session_exports",
        action="store_true",
        help="Do not store per-session summary metadata in the snapshot.",
    )
    return parser.parse_args()


def resolve_memory_backend(requested_backend: str, memory_model: str) -> str:
    if requested_backend != "auto":
        return requested_backend
    structured_output_models = (
        "DeepSeek-R1-0528-Qwen3",
        "Qwen/Qwen3",
        "google/gemma-3",
        "DeepSeek-R1-Distill-Llama",
    )
    if any(name in memory_model for name in structured_output_models):
        return "structured_output"
    return "langmem"


def render_messages_for_structured_output(messages: List[Dict[str, str]]) -> str:
    rendered: List[str] = []
    for message in messages:
        role = str(message.get("role", "user")).capitalize()
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        rendered.append(f"{role}: {content}")
    return "\n".join(rendered)


def structured_output_memory_limit(memory_model: str) -> int:
    return 6


def build_structured_output_prompt(
    messages: List[Dict[str, str]],
    max_memories: int,
) -> str:
    return (
        "Extract durable user memories from the conversation.\n"
        "Focus only on information that remains useful across future interactions.\n"
        "Prefer explicit preferences, repeated/implicit preferences, behavior patterns, profile facts, and API outcomes.\n"
        "Do not include transient one-off details unless they indicate a stable preference.\n"
        "Set domain, slot, and value only when they are clear; otherwise use null.\n"
        "Keep memories concise and standalone.\n"
        "Prefer fewer, higher-confidence memories over long lists.\n"
        f"Return at most {max_memories} memories.\n\n"
        "Conversation:\n"
        f"{render_messages_for_structured_output(messages)}"
    )


def build_compact_structured_output_prompt(messages: List[Dict[str, str]]) -> str:
    return (
        "Extract only the highest-confidence durable user memories from the conversation.\n"
        "Return at most 3 memories.\n"
        "Each memory must contain only content and category.\n"
        "Keep each memory concise and standalone.\n"
        "Prefer fewer items over speculative items.\n\n"
        "Conversation:\n"
        f"{render_messages_for_structured_output(messages)}"
    )


def normalize_semantic_memory_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(payload)
    payload.setdefault("domain", None)
    payload.setdefault("slot", None)
    payload.setdefault("value", None)
    payload.setdefault("context", None)
    memory = SemanticMemory(**payload)
    normalized = (
        memory.model_dump()
        if hasattr(memory, "model_dump")
        else memory.dict()
    )
    for key, value in list(normalized.items()):
        if isinstance(value, str):
            cleaned = sanitize_text_for_transport(value).strip()
            if cleaned.lower() == "null":
                normalized[key] = None
            else:
                normalized[key] = cleaned
        if normalized[key] == "":
            normalized[key] = None
    normalized["content"] = normalized["content"] or ""
    return normalized


def preview_structured_memory_payload(payload: Dict[str, Any], limit: int = 280) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    compact = " ".join(sanitize_text_for_transport(serialized).split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."


def structured_memory_key(payload: Dict[str, Any]) -> str:
    stable_payload = {
        "content": payload.get("content"),
        "category": payload.get("category"),
        "domain": payload.get("domain"),
        "slot": payload.get("slot"),
        "value": payload.get("value"),
    }
    digest = hashlib.sha1(
        json.dumps(stable_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"mem_{digest[:20]}"


def salvage_partial_structured_items(raw_content: str) -> List[Dict[str, Any]]:
    items_index = raw_content.find('"items"')
    if items_index == -1:
        return []
    array_start = raw_content.find("[", items_index)
    if array_start == -1:
        return []

    candidates: List[Dict[str, Any]] = []
    depth = 0
    object_start: int | None = None
    in_string = False
    escaping = False

    for index in range(array_start + 1, len(raw_content)):
        char = raw_content[index]

        if in_string:
            if escaping:
                escaping = False
            elif char == "\\":
                escaping = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                object_start = index
            depth += 1
            continue
        if char == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and object_start is not None:
                candidate = raw_content[object_start : index + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    object_start = None
                    continue
                if isinstance(parsed, dict):
                    candidates.append(parsed)
                object_start = None
            continue
        if char == "]" and depth == 0:
            break

    return candidates


async def extract_compact_memories_with_structured_output(
    client: Any,
    memory_model: str,
    messages: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    prompt = build_compact_structured_output_prompt(messages)
    for max_tokens in (192, 256, 384):
        response = await client.chat.completions.create(
            model=memory_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": COMPACT_STRUCTURED_OUTPUT_JSON_SCHEMA,
            },
        )
        raw_content = response.choices[0].message.content or "{}"
        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError:
            continue

        memories: List[Dict[str, Any]] = []
        for item in parsed.get("items", []):
            normalized = normalize_semantic_memory_payload(item)
            if not normalized.get("content"):
                continue
            memories.append(normalized)
        if memories:
            return memories
    return []


def apply_structured_memories(
    store: Any,
    namespace: tuple[str, ...],
    memories: List[Dict[str, Any]],
    logger: Any | None = None,
    example_id: str | None = None,
    session_index: int | None = None,
) -> None:
    active_items = [
        (item.key, normalize_memory_value(item.value))
        for item in list_namespace_items(store, namespace)
    ]
    for payload in memories:
        if payload.get("domain") and payload.get("slot"):
            survivors = []
            for key, existing_payload in active_items:
                same_slot = (
                    existing_payload.get("domain") == payload.get("domain")
                    and existing_payload.get("slot") == payload.get("slot")
                )
                changed_value = (
                    existing_payload.get("value") != payload.get("value")
                    or existing_payload.get("content") != payload.get("content")
                )
                if same_slot and changed_value:
                    try:
                        store.delete(namespace, key)
                    except Exception:
                        pass
                    continue
                survivors.append((key, existing_payload))
            active_items = survivors

        key = structured_memory_key(payload)
        try:
            store.put(namespace, key, payload)
        except Exception as exc:
            if logger is not None:
                logger.warning(
                    "Skipping structured memory after store.put failure. example_id=%s session_index=%s key=%s error=%s payload_preview=%s",
                    example_id,
                    session_index,
                    key,
                    exc,
                    preview_structured_memory_payload(payload),
                )
                continue
            raise
        active_items = [(existing_key, existing_payload) for existing_key, existing_payload in active_items if existing_key != key]
        active_items.append((key, payload))


async def extract_memories_with_structured_output(
    client: Any,
    memory_model: str,
    messages: List[Dict[str, str]],
    logger: Any | None = None,
    example_id: str | None = None,
    session_index: int | None = None,
) -> List[Dict[str, Any]]:
    max_memories = structured_output_memory_limit(memory_model)
    prompt = build_structured_output_prompt(messages, max_memories=max_memories)
    parsed: Dict[str, Any] | None = None
    last_error: Exception | None = None
    last_raw_content = ""
    raw_attempts: List[str] = []
    max_token_schedule = (768, 1536, 2048)
    for max_tokens in max_token_schedule:
        response = await client.chat.completions.create(
            model=memory_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": STRUCTURED_OUTPUT_JSON_SCHEMA,
            },
        )
        raw_content = response.choices[0].message.content or "{}"
        last_raw_content = raw_content
        raw_attempts.append(raw_content)
        try:
            parsed = json.loads(raw_content)
            break
        except json.JSONDecodeError as exc:
            last_error = exc
    if parsed is None:
        salvaged_items: List[Dict[str, Any]] = []
        for raw_content in reversed(raw_attempts):
            candidate_items = salvage_partial_structured_items(raw_content)
            if len(candidate_items) > len(salvaged_items):
                salvaged_items = candidate_items
                last_raw_content = raw_content
        if salvaged_items:
            if logger is not None:
                logger.info(
                    "Structured output parsing failed; recovered %d partial items. model=%s example_id=%s session_index=%s error=%s",
                    len(salvaged_items),
                    memory_model,
                    example_id,
                    session_index,
                    last_error,
                )
            parsed = {"items": salvaged_items}
        else:
            compact_memories = await extract_compact_memories_with_structured_output(
                client,
                memory_model,
                messages,
            )
            if compact_memories:
                if logger is not None:
                    compact_preview = " ".join(
                        sanitize_text_for_transport(last_raw_content).split()
                    )
                    logger.info(
                        "Structured output parsing failed; compact fallback recovered %d items. model=%s example_id=%s session_index=%s error=%s raw_preview=%s",
                        len(compact_memories),
                        memory_model,
                        example_id,
                        session_index,
                        last_error,
                        compact_preview[:280],
                    )
                return compact_memories
        if logger is not None:
            if parsed is None:
                compact_preview = " ".join(last_raw_content.split())
                logger.warning(
                    "Structured output parsing failed; skipping session memory extraction. model=%s example_id=%s session_index=%s error=%s raw_preview=%s",
                    memory_model,
                    example_id,
                    session_index,
                    last_error,
                    compact_preview[:500],
                )
        if parsed is None:
            return []
    memories: List[Dict[str, Any]] = []
    for item in parsed.get("items", []):
        normalized = normalize_semantic_memory_payload(item)
        if not normalized.get("content"):
            continue
        memories.append(normalized)
    return memories


async def main() -> None:
    args = parse_args()
    run_log_path = args.run_log_path or default_run_log_path(args.output_path)
    logger = setup_logger("langmem_build", run_log_path)
    memory_backend = resolve_memory_backend(args.memory_backend, args.memory_model)
    df = load_chains_dataset(args.input_path)
    rows = df.to_dict("records")[args.start_example : args.end_example]
    if args.max_examples is not None:
        rows = rows[: args.max_examples]
    logger.info(
        "Starting LangMem build: dataset_rows=%d selected_examples=%d input=%s output=%s manifest=%s model=%s embedding_model=%s concurrency=%d memory_base_url=%s embedding_base_url=%s memory_backend=%s langmem_memory_schema=%s langmem_prompt_style=%s",
        len(df),
        len(rows),
        args.input_path,
        args.output_path,
        args.manifest_path,
        args.memory_model,
        args.embedding_model,
        args.concurrency,
        args.base_url,
        args.embedding_base_url,
        memory_backend,
        args.langmem_memory_schema,
        args.langmem_prompt_style,
    )
    encoding = get_encoding()
    semaphore = asyncio.Semaphore(args.concurrency)
    runtime_memory_schema = (
        args.langmem_memory_schema if memory_backend == "langmem" else "semantic"
    )
    runtime_prompt_style = (
        args.langmem_prompt_style if memory_backend == "langmem" else "custom"
    )
    runtime = create_langmem_runtime(
        memory_model=args.memory_model,
        embedding_model=args.embedding_model,
        base_url=args.base_url,
        api_key=args.api_key,
        embedding_base_url=args.embedding_base_url,
        embedding_api_key=args.embedding_api_key,
        enable_deletes=True,
        memory_schema=runtime_memory_schema,
        prompt_style=runtime_prompt_style,
    )
    logger.info(
        "LangMem runtime ready: embedding_dimensions=%d session_exports_enabled=%s",
        runtime.embedding_dimensions,
        not args.disable_session_exports,
    )
    if memory_backend != "langmem" and (
        args.langmem_memory_schema != "semantic"
        or args.langmem_prompt_style != "custom"
    ):
        logger.info(
            "Ignoring langmem-specific settings because memory_backend=%s",
            memory_backend,
        )
    snapshot_rows: List[Dict[str, Any]] = []
    file_lock = asyncio.Lock()
    progress_lock = asyncio.Lock()
    store_lock = asyncio.Lock()
    completed_examples = 0
    progress_interval = max(1, min(10, len(rows))) if rows else 1
    structured_output_client = None
    if memory_backend == "structured_output":
        structured_output_client = create_async_openai_client(
            api_key=args.api_key,
            base_url=args.base_url,
            allow_empty=bool(args.base_url),
        )

    async def process_example(example: Dict[str, Any]) -> None:
        async with semaphore:
            example_id = str(example.get("example_id", "unknown_user"))
            namespace = materialize_namespace(example_id)
            config = make_runtime_config(example_id)
            session_exports: List[Dict[str, Any]] = []

            for session_index, session in enumerate(example.get("sessions", []), start=1):
                messages = build_session_messages(session)
                if not messages:
                    session_exports.append(
                        {
                            "session_index": session_index,
                            "dialogue_id": session.get("dialogue_id"),
                            "status": "SKIPPED_EMPTY_SESSION",
                            "last_user_utterance": last_user_utterance(session),
                            "session_input_tokens": 0,
                            "memory_count_after_session": None,
                            "stored_memory_tokens_after_session": None,
                        }
                    )
                    continue

                if memory_backend == "structured_output":
                    extracted_memories = await extract_memories_with_structured_output(
                        structured_output_client,
                        args.memory_model,
                        messages,
                        logger=logger,
                        example_id=example_id,
                        session_index=session_index,
                    )
                    async with store_lock:
                        await asyncio.to_thread(
                            apply_structured_memories,
                            runtime.store,
                            namespace,
                            extracted_memories,
                            logger,
                            example_id,
                            session_index,
                        )
                else:
                    await runtime.manager.ainvoke({"messages": messages}, config=config)

                if args.disable_session_exports:
                    continue

                items = await asyncio.to_thread(list_namespace_items, runtime.store, namespace)
                session_exports.append(
                    {
                        "session_index": session_index,
                        "dialogue_id": session.get("dialogue_id"),
                        "status": "OK",
                        "last_user_utterance": last_user_utterance(session),
                        "session_input_tokens": session_input_token_count(session, encoding),
                        "memory_count_after_session": len(items),
                        "stored_memory_tokens_after_session": count_serialized_memory_tokens(
                            [make_snapshot_item(item) for item in items],
                            encoding,
                        ),
                    }
                )

            items = await asyncio.to_thread(list_namespace_items, runtime.store, namespace)
            record = export_snapshot_record(
                example_id=example_id,
                namespace=namespace,
                memory_items=items,
                session_exports=session_exports,
            )
            async with file_lock:
                snapshot_rows.append(record)

    async def wrapped_process(example: Dict[str, Any]) -> None:
        nonlocal completed_examples
        example_id = str(example.get("example_id", "unknown_user"))
        try:
            await process_example(example)
        except Exception:
            logger.exception("Build failed for example_id=%s", example_id)
            raise
        async with progress_lock:
            completed_examples += 1
            if completed_examples == len(rows) or completed_examples % progress_interval == 0:
                logger.info(
                    "Build progress: %d/%d examples processed",
                    completed_examples,
                    len(rows),
                )

    await asyncio.gather(*(wrapped_process(example) for example in rows))

    snapshot_rows.sort(key=lambda row: row.get("example_id", ""))
    write_jsonl(args.output_path, snapshot_rows)
    manifest = {
        "created_at": now_iso(),
        "input_path": args.input_path,
        "output_path": args.output_path,
        "memory_model": args.memory_model,
        "embedding_model": args.embedding_model,
        "embedding_dimensions": runtime.embedding_dimensions,
        "memory_base_url": args.base_url,
        "embedding_base_url": args.embedding_base_url,
        "run_log_path": run_log_path,
        "namespace_template": ["langmem", "{user_id}", "semantic"],
        "examples_processed": len(snapshot_rows),
        "start_example": args.start_example,
        "end_example": args.end_example,
        "max_examples": args.max_examples,
        "session_exports_enabled": not args.disable_session_exports,
        "memory_backend": memory_backend,
        "langmem_memory_schema": args.langmem_memory_schema,
        "langmem_prompt_style": args.langmem_prompt_style,
    }
    write_json(args.manifest_path, manifest)
    logger.info(
        "Build finished: examples_processed=%d snapshot=%s manifest=%s",
        len(snapshot_rows),
        args.output_path,
        args.manifest_path,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
