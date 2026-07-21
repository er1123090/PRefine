#!/usr/bin/env python3
"""Prepare, submit, and collect ours_memory_v1 MEMORY3 Gemini Batch API jobs."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from google import genai
from google.genai import types as genai_types

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from methods.our_memory.v1.inference_api_singleturn import build_memory_input_prompt  # noqa: E402
from src.data_utils import (  # noqa: E402
    assign_user_utterances_multiturn,
    assign_user_utterances_singleturn,
    filter_prepared_items_by_example_id_sub,
    limit_prepared_items,
    load_chains_dataset,
    load_example_id_sub_filter,
    load_memory_file,
    load_multiturn_data,
    load_query_map,
    load_tools_from_file,
    normalize_context_type,
)
from src.llm_client import _extract_text_function_call, parse_deepseek_reasoning  # noqa: E402
from src.prompts import (  # noqa: E402
    IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE,
    IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE_MULTITURN,
)


CONTEXT = "memory_api"
PROMPT = "implicit_zs"
DEFAULT_RUN_ID = "memory3_v1_gemini3_flash_high_batch_2seed_20260526"
DEFAULT_MEMORY_ROOT = Path("/data/minseo/experiments4/ours_memory/inference/1231_MEMORY3")
DEFAULT_MEMORY_MODELS = [
    "google_gemma-3-12b-it",
    "gpt-4o-mini",
    "deepseek-ai_DeepSeek-R1-0528-Qwen3-8B",
    "deepseek-ai_DeepSeek-R1-Distill-Llama-8B",
]
DEFAULT_REPEATS = ["seed_001", "seed_002"]
DEFAULT_TURNS = ["singleturn", "multiturn"]
DEFAULT_DIFFICULTIES = ["easy", "medium", "hard"]
DEFAULT_MODEL_SPECS = ["gemini-3-flash[high]"]
GEMINI_ALIASES = {
    "gemini-3-flash": "gemini-3-flash-preview",
    "gemini-3-pro": "gemini-3-pro-preview",
}


def model_safe(name: str) -> str:
    return (
        name.replace("/", "_")
        .replace("[", "__")
        .replace("]", "")
        .replace(":", "_")
    )


def enum_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(getattr(value, "value", value))


def read_split(value: Optional[str], default: List[str]) -> List[str]:
    return value.split() if value else list(default)


def resolve_gemini_model(name: str) -> str:
    normalized = name.strip()
    return GEMINI_ALIASES.get(normalized, normalized)


def parse_model_spec(spec: str) -> Dict[str, Optional[str]]:
    """Parse strings like gemini-3-flash[high] into request model + metadata."""
    if spec.endswith("]") and "[" in spec:
        model, effort = spec[:-1].split("[", 1)
        return {
            "display": spec,
            "model": resolve_gemini_model(model),
            "reasoning_effort": effort,
        }
    return {"display": spec, "model": resolve_gemini_model(spec), "reasoning_effort": None}


def json_line(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"


def output_file(run_root: Path, repeat: str, turn: str, model_display: str, pref: str, memory_safe: str) -> Path:
    return (
        run_root
        / "repeats"
        / repeat
        / "inference"
        / turn
        / model_safe(model_display)
        / CONTEXT
        / pref
        / memory_safe
        / PROMPT
        / "result.json"
    )


def log_file(log_root: Path, repeat: str, turn: str, model_display: str, pref: str, memory_safe: str) -> Path:
    return (
        log_root
        / "repeats"
        / repeat
        / "inference"
        / turn
        / model_safe(model_display)
        / CONTEXT
        / pref
        / memory_safe
        / PROMPT
        / "result.jsonl"
    )


def memory_file(source_root: Path, memory_safe: str, filename: str) -> Path:
    return source_root / memory_safe / filename


def prepared_items(
    *,
    turn: str,
    pref_type: str,
    data_path: Path,
    memory_path: Path,
    query_single: Path,
    query_multi: Path,
    pref_list: Path,
    pref_group: Path,
    max_queries: Optional[int],
    example_id_sub_filter_path: Optional[Path],
) -> List[Dict[str, Any]]:
    df = load_chains_dataset(str(data_path))
    memory_storage = load_memory_file(str(memory_path))
    filter_ids = load_example_id_sub_filter(str(example_id_sub_filter_path)) if example_id_sub_filter_path else None

    if turn == "singleturn":
        query_map = load_query_map(str(query_single))
        assign_fn = lambda ex: assign_user_utterances_singleturn(  # noqa: E731
            str(pref_list), ex, query_map, pref_type, str(pref_group)
        )
    else:
        multiturn_data = load_multiturn_data(str(query_multi))
        assign_fn = lambda ex: assign_user_utterances_multiturn(  # noqa: E731
            str(pref_list), ex, multiturn_data, pref_type, str(pref_group)
        )

    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        original_ex = row.to_dict()
        example_id = str(original_ex.get("example_id", ""))
        if pref_type == "easy" and not original_ex.get("api_calls"):
            continue
        if pref_type in ("medium", "hard") and not original_ex.get("api_calls_pref"):
            continue
        pairs = assign_fn(original_ex)
        if not pairs:
            continue
        for sub_idx, (utterance, ground_truth) in enumerate(pairs):
            rows.append(
                {
                    "original_ex": original_ex,
                    "user_memory": memory_storage.get(example_id, {}),
                    "utterance": utterance,
                    "ground_truth": ground_truth,
                    "sub_idx": sub_idx,
                }
            )

    if filter_ids:
        rows = filter_prepared_items_by_example_id_sub(rows, filter_ids)
    else:
        rows = limit_prepared_items(rows, max_queries)
    return rows


class ChunkWriter:
    def __init__(self, batch_root: Path, max_file_bytes: int):
        self.batch_root = batch_root
        self.max_file_bytes = max_file_bytes
        self.states: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.chunks: List[Dict[str, Any]] = []

    def _open_state(self, group: Tuple[str, str]) -> Dict[str, Any]:
        state = self.states.get(group)
        if state is not None:
            return state
        model_display, repeat = group
        return self._new_state(model_display, repeat, 1)

    def _new_state(self, model_display: str, repeat: str, chunk_index: int) -> Dict[str, Any]:
        group_dir = self.batch_root / model_safe(model_display) / repeat
        group_dir.mkdir(parents=True, exist_ok=True)
        stem = f"batch_{model_safe(model_display)}_{repeat}_part{chunk_index:03d}"
        request_path = group_dir / f"{stem}.jsonl"
        mapping_path = group_dir / f"{stem}.mapping.jsonl"
        state = {
            "model_display": model_display,
            "repeat": repeat,
            "chunk_index": chunk_index,
            "request_path": request_path,
            "mapping_path": mapping_path,
            "request_file": request_path.open("w", encoding="utf-8"),
            "mapping_file": mapping_path.open("w", encoding="utf-8"),
            "bytes": 0,
            "requests": 0,
        }
        self.states[(model_display, repeat)] = state
        return state

    def write(self, group: Tuple[str, str], request: Dict[str, Any], mapping: Dict[str, Any]) -> None:
        state = self._open_state(group)
        req_line = json_line(request)
        map_line = json_line(mapping)
        line_bytes = len(req_line.encode("utf-8"))
        if state["requests"] > 0 and state["bytes"] + line_bytes > self.max_file_bytes:
            self._close_state(group)
            state = self._new_state(group[0], group[1], state["chunk_index"] + 1)
        state["request_file"].write(req_line)
        state["mapping_file"].write(map_line)
        state["bytes"] += line_bytes
        state["requests"] += 1

    def _close_state(self, group: Tuple[str, str]) -> None:
        state = self.states.pop(group)
        state["request_file"].close()
        state["mapping_file"].close()
        self.chunks.append(
            {
                "model_display": state["model_display"],
                "repeat": state["repeat"],
                "chunk_index": state["chunk_index"],
                "request_path": str(state["request_path"]),
                "mapping_path": str(state["mapping_path"]),
                "bytes": state["bytes"],
                "requests": state["requests"],
                "input_file_name": None,
                "batch_name": None,
                "status": "prepared",
                "output_file_name": None,
                "error": None,
            }
        )

    def close(self) -> List[Dict[str, Any]]:
        for group in list(self.states):
            self._close_state(group)
        self.chunks.sort(key=lambda row: (row["model_display"], row["repeat"], row["chunk_index"]))
        return self.chunks


def request_body(model_info: Dict[str, Optional[str]], prompt: str) -> Dict[str, Any]:
    generation_config: Dict[str, Any] = {"temperature": 0.0}
    if model_info.get("reasoning_effort"):
        generation_config["thinking_config"] = {
            "thinking_level": str(model_info["reasoning_effort"]).lower(),
        }
    return {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generation_config": generation_config,
    }


def prepare(args: argparse.Namespace) -> None:
    run_root = ROOT / "outputs" / "our_memory" / args.parent_run_id
    log_root = ROOT / "logs" / "our_memory" / args.parent_run_id
    batch_root = run_root / "batch_api"
    source_root = Path(args.source_memory_root)
    query_single = ROOT / "config" / "query_singleturn.json"
    query_multi = ROOT / "config" / "query_multiturn-domain.json"
    pref_list = ROOT / "config" / "pref_list.json"
    pref_group = ROOT / "config" / "pref_group.json"
    schema_single = ROOT / "config" / "schema_easy.json"
    schema_multi = ROOT / "config" / "schema_all.json"
    data_path = Path(args.data_path)
    model_infos = [parse_model_spec(spec) for spec in read_split(args.only_models, DEFAULT_MODEL_SPECS)]
    memory_sources = read_split(args.only_memory_models, DEFAULT_MEMORY_MODELS)
    repeats = read_split(args.only_repeats, DEFAULT_REPEATS)
    turns = read_split(args.only_turns, DEFAULT_TURNS)
    difficulties = read_split(args.only_prefs, DEFAULT_DIFFICULTIES)
    max_queries = args.max_queries
    if args.smoke_only and max_queries is None:
        max_queries = 2

    batch_root.mkdir(parents=True, exist_ok=True)
    writer = ChunkWriter(batch_root=batch_root, max_file_bytes=args.max_file_bytes)
    sequence = 0

    for repeat in repeats:
        for memory_safe in memory_sources:
            mem_path = memory_file(source_root, memory_safe, args.v1_memory_filename)
            if not mem_path.exists():
                raise SystemExit(f"missing memory file: {mem_path}")
            for difficulty in difficulties:
                for turn in turns:
                    tools_schema = load_tools_from_file(str(schema_single if turn == "singleturn" else schema_multi))
                    prompt_template = (
                        IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE
                        if turn == "singleturn"
                        else IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE_MULTITURN
                    )
                    items = prepared_items(
                        turn=turn,
                        pref_type=difficulty,
                        data_path=data_path,
                        memory_path=mem_path,
                        query_single=query_single,
                        query_multi=query_multi,
                        pref_list=pref_list,
                        pref_group=pref_group,
                        max_queries=max_queries,
                        example_id_sub_filter_path=Path(args.example_id_sub_filter_path)
                        if args.example_id_sub_filter_path
                        else None,
                    )
                    for model_info in model_infos:
                        model_display = str(model_info["display"])
                        out_path = output_file(run_root, repeat, turn, model_display, difficulty, memory_safe)
                        l_path = log_file(log_root, repeat, turn, model_display, difficulty, memory_safe)
                        for item in items:
                            current_ex = copy.deepcopy(item["original_ex"])
                            current_ex["user_utterance"] = item["utterance"]
                            current_ex["reference_ground_truth"] = item["ground_truth"]
                            current_ex["example_id_sub"] = (
                                f"{current_ex.get('example_id', 'unknown')}_{item['sub_idx']}"
                            )
                            current_ex["model_name"] = model_display
                            prompt = build_memory_input_prompt(
                                example=current_ex,
                                user_memory=item["user_memory"],
                                current_user_utterance=item["utterance"],
                                template=prompt_template,
                                context_type=normalize_context_type(CONTEXT),
                                tools_schema=tools_schema,
                            )
                            custom_id = (
                                f"{repeat}-{model_safe(model_display)}-{turn}-{difficulty}-"
                                f"{memory_safe}-{sequence}"
                            )
                            request = {
                                "key": custom_id,
                                "request": request_body(model_info, prompt),
                            }
                            mapping = {
                                "custom_id": custom_id,
                                "repeat_id": repeat,
                                "turn": turn,
                                "difficulty": difficulty,
                                "memory_safe_name": memory_safe,
                                "model_display": model_display,
                                "request_model": model_info["model"],
                                "reasoning_effort": model_info["reasoning_effort"],
                                "context_type": CONTEXT,
                                "pref_type": difficulty,
                                "prompt_type": PROMPT,
                                "output_path": str(out_path),
                                "log_path": str(l_path),
                                "sequence": sequence,
                                "current_ex": current_ex,
                                "injected_utterance": item["utterance"],
                                "reference_ground_truth": item["ground_truth"],
                            }
                            writer.write((model_display, repeat), request, mapping)
                            sequence += 1

    chunks = writer.close()
    for chunk in chunks:
        model_info = next(info for info in model_infos if info["display"] == chunk["model_display"])
        chunk["request_model"] = model_info["model"]
        chunk["reasoning_effort"] = model_info["reasoning_effort"]
    manifest = {
        "created_at": datetime.now().isoformat(),
        "parent_run_id": args.parent_run_id,
        "memory_method": "ours_memory_v1",
        "source_memory_root": str(source_root),
        "v1_memory_filename": args.v1_memory_filename,
        "api": "gemini_batch",
        "endpoint": "batchGenerateContent",
        "context": CONTEXT,
        "prompt_type": PROMPT,
        "repeats": repeats,
        "turns": turns,
        "difficulties": difficulties,
        "memory_sources": memory_sources,
        "inference_models": [info["display"] for info in model_infos],
        "request_models": [info["model"] for info in model_infos],
        "seed_policy": "deterministic_batch_rerun",
        "request_count": sequence,
        "chunks": chunks,
    }
    (run_root / "metadata.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (batch_root / "batch_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"prepared_requests": sequence, "chunks": len(chunks), "batch_root": str(batch_root)}, indent=2))


def make_client() -> genai.Client:
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_CHAT_API_KEY")
    if not key:
        raise SystemExit("GOOGLE_API_KEY or GEMINI_API_KEY is required")
    return genai.Client(api_key=key)


def submit(args: argparse.Namespace) -> None:
    run_root = ROOT / "outputs" / "our_memory" / args.parent_run_id
    manifest_path = run_root / "batch_api" / "batch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    client = make_client()
    changed = False
    for chunk in manifest["chunks"]:
        if chunk.get("batch_name"):
            continue
        request_path = Path(chunk["request_path"])
        display_name = f"{args.parent_run_id}_{model_safe(chunk['model_display'])}_{chunk['repeat']}_{chunk['chunk_index']:03d}"
        uploaded = client.files.upload(
            file=str(request_path),
            config=genai_types.UploadFileConfig(display_name=display_name, mime_type="jsonl"),
        )
        batch = client.batches.create(
            model=chunk["request_model"],
            src=uploaded.name,
            config=genai_types.CreateBatchJobConfig(display_name=display_name),
        )
        chunk["input_file_name"] = uploaded.name
        chunk["input_file_uri"] = uploaded.uri
        chunk["batch_name"] = batch.name
        chunk["status"] = enum_value(batch.state)
        chunk["output_file_name"] = getattr(getattr(batch, "dest", None), "file_name", None)
        chunk["error"] = str(batch.error) if getattr(batch, "error", None) else None
        changed = True
        print(
            json.dumps(
                {
                    "submitted": request_path.name,
                    "input_file": uploaded.name,
                    "batch_name": batch.name,
                    "status": chunk["status"],
                },
                ensure_ascii=False,
            )
        )
    if changed:
        manifest["submitted_at"] = datetime.now().isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (run_root / "metadata.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def status(args: argparse.Namespace) -> None:
    run_root = ROOT / "outputs" / "our_memory" / args.parent_run_id
    manifest_path = run_root / "batch_api" / "batch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    client = make_client()
    rows = []
    for chunk in manifest["chunks"]:
        batch_name = chunk.get("batch_name")
        if not batch_name:
            rows.append({"request_path": chunk["request_path"], "status": "not_submitted"})
            continue
        batch = client.batches.get(name=batch_name)
        chunk["status"] = enum_value(batch.state)
        chunk["output_file_name"] = getattr(getattr(batch, "dest", None), "file_name", None)
        chunk["error"] = batch.error.model_dump() if getattr(batch.error, "model_dump", None) else str(batch.error or "")
        stats = getattr(batch, "completion_stats", None)
        rows.append(
            {
                "batch_name": batch_name,
                "model": chunk["model_display"],
                "request_model": chunk["request_model"],
                "repeat": chunk["repeat"],
                "chunk": chunk["chunk_index"],
                "status": chunk["status"],
                "completion_stats": stats.model_dump() if hasattr(stats, "model_dump") else str(stats),
                "output_file_name": chunk["output_file_name"],
                "error": chunk["error"],
            }
        )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (run_root / "metadata.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2, ensure_ascii=False))


def load_request_prompts(chunks: Iterable[Dict[str, Any]]) -> Dict[str, str]:
    prompts: Dict[str, str] = {}
    for chunk in chunks:
        with Path(chunk["request_path"]).open(encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                custom_id = obj["key"]
                parts = obj.get("request", {}).get("contents", [{}])[0].get("parts", [])
                prompts[custom_id] = "\n".join(str(part.get("text", "")) for part in parts)
    return prompts


def load_ordered_mappings(chunks: Iterable[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]]]:
    mappings: Dict[str, Dict[str, Any]] = {}
    ordered: Dict[str, List[str]] = {}
    for chunk in chunks:
        ids: List[str] = []
        with Path(chunk["mapping_path"]).open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                mappings[row["custom_id"]] = row
                ids.append(row["custom_id"])
        ordered[str(chunk["request_path"])] = ids
    return mappings, ordered


def get_any(obj: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in obj:
            return obj[key]
    return default


def parse_generate_content_body(body: Dict[str, Any], tools_schema: Optional[List[Dict[str, Any]]]) -> Tuple[str, str, Dict[str, Any]]:
    candidates = body.get("candidates") or []
    content = candidates[0].get("content", {}) if candidates else {}
    parts = content.get("parts") or []
    thought_parts: List[str] = []
    answer_parts: List[str] = []
    for part in parts:
        text = part.get("text") or ""
        if not text:
            continue
        if part.get("thought"):
            thought_parts.append(text)
        else:
            answer_parts.append(text)

    raw_content = "\n".join(answer_parts).strip()
    parsed = parse_deepseek_reasoning(raw_content)
    final_output = parsed["llm_output"]
    reasoning_content = "\n".join(thought_parts).strip() or parsed["reasoning_content"]
    extracted = _extract_text_function_call(final_output or raw_content, tools_schema)
    llm_output = extracted or final_output

    usage = body.get("usageMetadata") or body.get("usage_metadata") or {}
    token_counts = {
        "total_tokens": get_any(usage, "totalTokenCount", "total_token_count", default=0),
        "reasoning_tokens": get_any(usage, "thoughtsTokenCount", "thoughts_token_count", default=0),
        "input_tokens": get_any(usage, "promptTokenCount", "prompt_token_count", default=0),
        "output_tokens": get_any(usage, "candidatesTokenCount", "candidates_token_count", default=0),
    }
    return llm_output, reasoning_content, token_counts


def output_record_and_key(result: Dict[str, Any], fallback_key: str) -> Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]:
    key = (
        result.get("key")
        or result.get("custom_id")
        or result.get("metadata", {}).get("key")
        or fallback_key
    )
    if "response" in result:
        return key, result.get("response") or {}, result.get("error")
    if "error" in result and "candidates" not in result:
        return key, {}, result.get("error")
    return key, result, None


def collect(args: argparse.Namespace) -> None:
    run_root = ROOT / "outputs" / "our_memory" / args.parent_run_id
    manifest_path = run_root / "batch_api" / "batch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    client = make_client()
    prompts = load_request_prompts(manifest["chunks"])
    mappings, ordered_ids = load_ordered_mappings(manifest["chunks"])
    tools_cache = {
        "singleturn": load_tools_from_file(str(ROOT / "config" / "schema_easy.json")),
        "multiturn": load_tools_from_file(str(ROOT / "config" / "schema_all.json")),
    }

    grouped_outputs: Dict[str, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    grouped_logs: Dict[str, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    collected = 0
    skipped = 0
    for chunk in manifest["chunks"]:
        batch_name = chunk.get("batch_name")
        if batch_name:
            batch = client.batches.get(name=batch_name)
            chunk["status"] = enum_value(batch.state)
            chunk["output_file_name"] = getattr(getattr(batch, "dest", None), "file_name", None)
            chunk["error"] = batch.error.model_dump() if getattr(batch.error, "model_dump", None) else str(batch.error or "")
        if chunk.get("status") != "JOB_STATE_SUCCEEDED" or not chunk.get("output_file_name"):
            skipped += 1
            continue

        content = client.files.download(file=chunk["output_file_name"]).decode("utf-8")
        output_dump = Path(chunk["request_path"]).with_suffix(".output.jsonl")
        output_dump.write_text(content, encoding="utf-8")
        fallback_ids = ordered_ids[str(chunk["request_path"])]
        for idx, line in enumerate(content.splitlines()):
            if not line.strip():
                continue
            result = json.loads(line)
            fallback_key = fallback_ids[idx] if idx < len(fallback_ids) else ""
            custom_id, body, err = output_record_and_key(result, fallback_key)
            meta = mappings[custom_id]
            if err:
                llm_output = f"BATCH_ERROR: {json.dumps(err, ensure_ascii=False)}"
                reasoning_content = ""
                token_counts: Dict[str, Any] = {}
            else:
                tools_schema = tools_cache.get(meta["turn"])
                llm_output, reasoning_content, token_counts = parse_generate_content_body(body, tools_schema)
            current_ex = meta["current_ex"]
            current_ex["llm_output"] = llm_output
            current_ex["reasoning_content"] = reasoning_content
            current_ex["reasoning_token_count"] = token_counts.get("reasoning_tokens", 0)
            grouped_outputs[meta["output_path"]].append((meta["sequence"], current_ex))
            grouped_logs[meta["log_path"]].append(
                (
                    meta["sequence"],
                    {
                        "timestamp": datetime.now().isoformat(),
                        "example_id": current_ex.get("example_id", ""),
                        "example_id_sub": current_ex.get("example_id_sub", ""),
                        "model_name": meta["model_display"],
                        "request_model": meta["request_model"],
                        "context_type": meta["context_type"],
                        "pref_type": meta["pref_type"],
                        "injected_utterance": meta["injected_utterance"],
                        "reasoning_effort": meta["reasoning_effort"],
                        "reference_ground_truth": meta["reference_ground_truth"],
                        "model_input": prompts.get(custom_id, ""),
                        "model_output": llm_output,
                        "reasoning_content": reasoning_content,
                        "token_counts": token_counts,
                    },
                )
            )
            collected += 1

    for path_str, rows in grouped_outputs.items():
        path = Path(path_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [row for _, row in sorted(rows, key=lambda pair: pair[0])]
        path.write_text(json.dumps(payload, indent=4, ensure_ascii=False), encoding="utf-8")
    for path_str, rows in grouped_logs.items():
        path = Path(path_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for _, row in sorted(rows, key=lambda pair: pair[0]):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest["collected_at"] = datetime.now().isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (run_root / "metadata.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"collected_responses": collected, "result_files": len(grouped_outputs), "skipped_chunks": skipped}, indent=2))


def write_summary_script(args: argparse.Namespace) -> None:
    run_root = ROOT / "outputs" / "our_memory" / args.parent_run_id
    manifest_path = run_root / "batch_api" / "batch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    counts = defaultdict(int)
    for chunk in manifest["chunks"]:
        counts[chunk["status"]] += 1
    print(json.dumps({"parent_run_id": args.parent_run_id, "chunk_status_counts": counts}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "submit", "prepare-submit", "status", "collect", "summary"])
    parser.add_argument("--parent-run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--data-path", default=str(ROOT / "data" / "1229_dev_6.json"))
    parser.add_argument("--source-memory-root", default=str(DEFAULT_MEMORY_ROOT))
    parser.add_argument("--v1-memory-filename", default="_memory1.jsonl")
    parser.add_argument("--only-models", default=None, help="Space-separated model specs, e.g. 'gemini-3-flash[high]'.")
    parser.add_argument("--only-memory-models", default=None)
    parser.add_argument("--only-repeats", default=None)
    parser.add_argument("--only-turns", default=None)
    parser.add_argument("--only-prefs", default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--example-id-sub-filter-path", default=None)
    parser.add_argument("--max-file-bytes", type=int, default=1_800_000_000)
    args = parser.parse_args()

    if args.action in {"prepare", "prepare-submit"}:
        prepare(args)
    if args.action in {"submit", "prepare-submit"}:
        submit(args)
    elif args.action == "status":
        status(args)
    elif args.action == "collect":
        collect(args)
    elif args.action == "summary":
        write_summary_script(args)


if __name__ == "__main__":
    main()
