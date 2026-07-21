#!/usr/bin/env python3
"""Prepare, submit, and collect ours_memory_v1 MEMORY3 OpenAI Batch API jobs."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI

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
from src.llm_client import parse_deepseek_reasoning  # noqa: E402
from src.prompts import (  # noqa: E402
    IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE,
    IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE_MULTITURN,
)


CONTEXT = "memory_api"
PROMPT = "implicit_zs"
DEFAULT_RUN_ID = "memory3_v1_openai_batch_2seed_20260524"
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
DEFAULT_MODEL_SPECS = ["gpt-4o-mini", "gpt-5-mini[high]"]


def model_safe(name: str) -> str:
    return (
        name.replace("/", "_")
        .replace("[", "__")
        .replace("]", "")
        .replace(":", "_")
    )


def read_split(value: Optional[str], default: List[str]) -> List[str]:
    return value.split() if value else list(default)


def parse_model_spec(spec: str) -> Dict[str, Optional[str]]:
    """Parse strings like gpt-5-mini[high] into request model + display metadata."""
    if spec.endswith("]") and "[" in spec:
        model, effort = spec[:-1].split("[", 1)
        return {"display": spec, "model": model, "reasoning_effort": effort}
    return {"display": spec, "model": spec, "reasoning_effort": None}


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
                "input_file_id": None,
                "batch_id": None,
                "status": "prepared",
                "output_file_id": None,
                "error_file_id": None,
            }
        )

    def close(self) -> List[Dict[str, Any]]:
        for group in list(self.states):
            self._close_state(group)
        self.chunks.sort(key=lambda row: (row["model_display"], row["repeat"], row["chunk_index"]))
        return self.chunks


def request_body(model_info: Dict[str, Optional[str]], prompt: str, tools_schema: List[Dict[str, Any]]) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "model": model_info["model"],
        "messages": [{"role": "user", "content": prompt}],
    }
    if tools_schema:
        body["tools"] = tools_schema
        body["tool_choice"] = "auto"
    model_name = str(model_info["model"]).lower()
    if model_info.get("reasoning_effort") and "gpt-5" in model_name:
        body["reasoning_effort"] = str(model_info["reasoning_effort"]).lower()
    elif not model_info.get("reasoning_effort"):
        body["temperature"] = 0.0
    return body


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
    totals = defaultdict(int)
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
                                "custom_id": custom_id,
                                "method": "POST",
                                "url": "/v1/chat/completions",
                                "body": request_body(model_info, prompt, tools_schema),
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
                            totals[(model_display, repeat)] += 1
                            sequence += 1

    chunks = writer.close()
    manifest = {
        "created_at": datetime.now().isoformat(),
        "parent_run_id": args.parent_run_id,
        "memory_method": "ours_memory_v1",
        "source_memory_root": str(source_root),
        "v1_memory_filename": args.v1_memory_filename,
        "api": "openai_batch",
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
        "context": CONTEXT,
        "prompt_type": PROMPT,
        "repeats": repeats,
        "turns": turns,
        "difficulties": difficulties,
        "memory_sources": memory_sources,
        "inference_models": [info["display"] for info in model_infos],
        "seed_policy": "deterministic_batch_rerun",
        "request_count": sequence,
        "chunks": chunks,
    }
    (run_root / "metadata.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (batch_root / "batch_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"prepared_requests": sequence, "chunks": len(chunks), "batch_root": str(batch_root)}, indent=2))


def submit(args: argparse.Namespace) -> None:
    run_root = ROOT / "outputs" / "our_memory" / args.parent_run_id
    manifest_path = run_root / "batch_api" / "batch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    client = OpenAI()
    changed = False
    for chunk in manifest["chunks"]:
        if chunk.get("batch_id"):
            continue
        request_path = Path(chunk["request_path"])
        with request_path.open("rb") as f:
            uploaded = client.files.create(file=f, purpose="batch")
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "parent_run_id": args.parent_run_id,
                "model": chunk["model_display"],
                "repeat": chunk["repeat"],
                "chunk_index": str(chunk["chunk_index"]),
            },
        )
        chunk["input_file_id"] = uploaded.id
        chunk["batch_id"] = batch.id
        chunk["status"] = batch.status
        chunk["output_file_id"] = getattr(batch, "output_file_id", None)
        chunk["error_file_id"] = getattr(batch, "error_file_id", None)
        changed = True
        print(json.dumps({"submitted": request_path.name, "file_id": uploaded.id, "batch_id": batch.id, "status": batch.status}))
    if changed:
        manifest["submitted_at"] = datetime.now().isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (run_root / "metadata.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def status(args: argparse.Namespace) -> None:
    run_root = ROOT / "outputs" / "our_memory" / args.parent_run_id
    manifest_path = run_root / "batch_api" / "batch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    client = OpenAI()
    rows = []
    for chunk in manifest["chunks"]:
        batch_id = chunk.get("batch_id")
        if not batch_id:
            rows.append({"request_path": chunk["request_path"], "status": "not_submitted"})
            continue
        batch = client.batches.retrieve(batch_id)
        chunk["status"] = batch.status
        chunk["output_file_id"] = getattr(batch, "output_file_id", None)
        chunk["error_file_id"] = getattr(batch, "error_file_id", None)
        counts = getattr(batch, "request_counts", None)
        rows.append(
            {
                "batch_id": batch_id,
                "model": chunk["model_display"],
                "repeat": chunk["repeat"],
                "chunk": chunk["chunk_index"],
                "status": batch.status,
                "request_counts": counts.model_dump() if hasattr(counts, "model_dump") else str(counts),
                "output_file_id": chunk["output_file_id"],
                "error_file_id": chunk["error_file_id"],
            }
        )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (run_root / "metadata.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2, ensure_ascii=False))


def load_request_prompts(chunks: Iterable[Dict[str, Any]]) -> Dict[str, str]:
    prompts = {}
    for chunk in chunks:
        with Path(chunk["request_path"]).open(encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                prompts[obj["custom_id"]] = obj["body"]["messages"][-1]["content"]
    return prompts


def parse_chat_completion_body(body: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    choices = body.get("choices") or []
    message = choices[0].get("message", {}) if choices else {}
    raw_content = message.get("content") or ""
    reasoning_content = message.get("reasoning_content") or ""
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        fn = tool_calls[0].get("function", {})
        name = fn.get("name", "")
        try:
            args = json.loads(fn.get("arguments") or "{}")
            args_str = ", ".join(f'{k}="{v}"' for k, v in args.items())
            output = f"{name}({args_str})"
        except Exception:
            output = f"ERROR_JSON_PARSE: {fn.get('arguments', '')}"
    else:
        parsed = parse_deepseek_reasoning(raw_content)
        output = parsed["llm_output"]
        reasoning_content = reasoning_content or parsed["reasoning_content"]

    usage = body.get("usage") or {}
    details = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
    token_counts = {
        "total_tokens": usage.get("total_tokens", 0),
        "reasoning_tokens": details.get("reasoning_tokens", 0) if isinstance(details, dict) else 0,
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
    }
    return output, reasoning_content, token_counts


def collect(args: argparse.Namespace) -> None:
    run_root = ROOT / "outputs" / "our_memory" / args.parent_run_id
    manifest_path = run_root / "batch_api" / "batch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    client = OpenAI()
    prompts = load_request_prompts(manifest["chunks"])
    mappings: Dict[str, Dict[str, Any]] = {}
    for chunk in manifest["chunks"]:
        with Path(chunk["mapping_path"]).open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                mappings[row["custom_id"]] = row

    grouped_outputs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    grouped_logs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    collected = 0
    for chunk in manifest["chunks"]:
        batch_id = chunk.get("batch_id")
        if batch_id:
            batch = client.batches.retrieve(batch_id)
            chunk["status"] = batch.status
            chunk["output_file_id"] = getattr(batch, "output_file_id", None)
            chunk["error_file_id"] = getattr(batch, "error_file_id", None)
        if chunk.get("status") != "completed" or not chunk.get("output_file_id"):
            continue
        content = client.files.content(chunk["output_file_id"]).read().decode("utf-8")
        output_dump = Path(chunk["request_path"]).with_suffix(".output.jsonl")
        output_dump.write_text(content, encoding="utf-8")
        for line in content.splitlines():
            if not line.strip():
                continue
            result = json.loads(line)
            custom_id = result["custom_id"]
            meta = mappings[custom_id]
            response = result.get("response") or {}
            if response.get("status_code") == 200:
                llm_output, reasoning_content, token_counts = parse_chat_completion_body(response.get("body") or {})
            else:
                err = result.get("error") or response
                llm_output, reasoning_content, token_counts = f"BATCH_ERROR: {json.dumps(err, ensure_ascii=False)}", "", {}
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
    print(json.dumps({"collected_responses": collected, "result_files": len(grouped_outputs)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "submit", "prepare-submit", "status", "collect"])
    parser.add_argument("--parent-run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--data-path", default=str(ROOT / "data" / "1229_dev_6.json"))
    parser.add_argument("--source-memory-root", default=str(DEFAULT_MEMORY_ROOT))
    parser.add_argument("--v1-memory-filename", default="_memory1.jsonl")
    parser.add_argument("--only-models", default=None, help="Space-separated model specs, e.g. 'gpt-4o-mini gpt-5-mini[high]'.")
    parser.add_argument("--only-memory-models", default=None)
    parser.add_argument("--only-repeats", default=None)
    parser.add_argument("--only-turns", default=None)
    parser.add_argument("--only-prefs", default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--example-id-sub-filter-path", default=None)
    parser.add_argument("--max-file-bytes", type=int, default=180_000_000)
    args = parser.parse_args()

    if args.action in {"prepare", "prepare-submit"}:
        prepare(args)
    if args.action in {"submit", "prepare-submit"}:
        submit(args)
    elif args.action == "status":
        status(args)
    elif args.action == "collect":
        collect(args)


if __name__ == "__main__":
    main()
