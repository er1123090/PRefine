from __future__ import annotations

import asyncio
import copy
import itertools
import json
import logging
import os
import pickle
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tiktoken
from openai import AsyncOpenAI


REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("SUPPORT_JSON_SCHEMA", "true")

from emem import EMem
from emem.utils.config_utils import BaseConfig
from emem.utils.conversation_data_utils import (
    Conversation,
    EventSummary,
    LoCoMoSample,
    Observation,
    Session,
    Turn,
)


DEFAULT_INPUT_PATH = "/data/minseo/experiments4/data/1229_dev_6.json"
DEFAULT_SINGLETURN_QUERY_PATH = "/data/minseo/experiments4/query_singleturn.json"
DEFAULT_MULTITURN_QUERY_PATH = "/data/minseo/experiments4/query_multiturn-domain.json"
DEFAULT_PREF_LIST_PATH = "/data/minseo/experiments4/pref_list.json"
DEFAULT_PREF_GROUP_PATH = "/data/minseo/experiments4/pref_group.json"
DEFAULT_SCHEMA_PATH = "/data/minseo/experiments4/schema_all.json"
DEFAULT_INDEX_ROOT = "/data/minseo/experiments4/e-mem/indexes/emem_1229_dev_6"
DEFAULT_MANIFEST_PATH = "/data/minseo/experiments4/e-mem/output/emem_1229_dev_6.manifest.jsonl"
DEFAULT_EMEM_LLM_MODEL = "gpt-4o-mini"
DEFAULT_EMEM_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_ENCODING = "cl100k_base"
DEFAULT_SUPPORT_JSON_SCHEMA = "true"
SYNTHETIC_BASE_DATETIME = datetime(2024, 1, 1, 9, 0)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def default_run_log_path(path: str) -> str:
    base, _ = os.path.splitext(path)
    return f"{base}.run.log"


def setup_logger(
    name: str,
    log_path: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_path:
        ensure_parent_dir(log_path)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_json_safe(data: Any) -> Any:
    return json.loads(json.dumps(data, ensure_ascii=False, default=str))


def load_chains_dataset(fpath: str) -> pd.DataFrame:
    try:
        return pd.read_json(fpath, lines=True)
    except ValueError:
        data = read_json(fpath)
        if isinstance(data, dict) and "dataset" in data:
            data = data["dataset"]
        return pd.DataFrame(data)


def load_query_map(fpath: str) -> Dict[str, str]:
    if not os.path.exists(fpath):
        return {}
    return read_json(fpath)


def load_multiturn_data(fpath: str) -> Dict[str, Any]:
    if not os.path.exists(fpath):
        return {}
    return read_json(fpath)


def load_tools_from_file(file_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(file_path):
        return []
    try:
        return read_json(file_path)
    except Exception:
        return []


def get_encoding(name: str = DEFAULT_ENCODING):
    return tiktoken.get_encoding(name)


def count_text_tokens(strings: Iterable[str], encoding) -> int:
    total = 0
    for text in strings:
        if not text:
            continue
        total += len(encoding.encode(text))
    return total


def count_string_tokens(text: str, encoding) -> int:
    if not text:
        return 0
    return len(encoding.encode(text))


def resolve_api_key(api_key: Optional[str], allow_empty: bool = False) -> str:
    resolved = api_key or os.environ.get("OPENAI_API_KEY")
    if resolved:
        return resolved
    if allow_empty:
        return "EMPTY"
    raise RuntimeError(
        "OPENAI_API_KEY is required. Pass --api_key or set OPENAI_API_KEY."
    )


def create_async_openai_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    allow_empty: bool = False,
) -> AsyncOpenAI:
    resolved_api_key = resolve_api_key(api_key, allow_empty=allow_empty)
    kwargs: Dict[str, Any] = {"api_key": resolved_api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return AsyncOpenAI(**kwargs)


def parse_deepseek_reasoning(raw_content: str) -> Dict[str, str]:
    if not raw_content:
        return {"llm_output": "", "reasoning_content": ""}
    end_tag = "</think>"
    end_idx = raw_content.rfind(end_tag)
    if end_idx == -1:
        return {"llm_output": raw_content.strip(), "reasoning_content": ""}
    reasoning_part = raw_content[:end_idx].replace("<think>", "").strip()
    clean_content = raw_content[end_idx + len(end_tag) :].strip()
    return {"llm_output": clean_content, "reasoning_content": reasoning_part}


async def call_openai_chat_async(
    prompt: str,
    model_name: str,
    client: AsyncOpenAI,
    reasoning_effort: Optional[str] = None,
) -> Dict[str, Any]:
    result = {
        "content": "",
        "reasoning": "",
        "token_counts": {},
        "error": None,
    }
    try:
        kwargs: Dict[str, Any] = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
        }
        is_reasoning_model = any(
            token in model_name.lower() for token in ["o1", "o3", "gpt-5"]
        )
        if is_reasoning_model and reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort.lower()

        response = await client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        raw_content = message.content or ""
        reasoning_text = getattr(message, "reasoning_content", "")
        if reasoning_text:
            result["content"] = raw_content
            result["reasoning"] = reasoning_text
        else:
            parsed = parse_deepseek_reasoning(raw_content)
            result["content"] = parsed["llm_output"]
            result["reasoning"] = parsed["reasoning_content"]

        if response.usage:
            details = getattr(response.usage, "completion_tokens_details", None)
            if not details:
                details = getattr(response.usage, "output_tokens_details", None)
            reasoning_tokens = getattr(details, "reasoning_tokens", 0) if details else 0
            result["token_counts"] = {
                "total_tokens": response.usage.total_tokens,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "reasoning_tokens": reasoning_tokens,
            }
        return result
    except Exception as exc:
        result["error"] = f"API_ERROR: {exc}"
        return result


def flatten_api_calls(example: Dict[str, Any]) -> List[str]:
    if isinstance(example.get("api_calls"), list):
        return [str(item) for item in example["api_calls"] if item]
    sessions = example.get("sessions", [])
    collected: List[str] = []
    for session in sessions:
        api_calls = session.get("api_call", [])
        if isinstance(api_calls, str):
            api_calls = [api_calls]
        if isinstance(api_calls, list):
            collected.extend(str(call) for call in api_calls if call)
    return collected


def format_dialogue_history(example: Dict[str, Any]) -> str:
    sessions = example.get("sessions", [])
    if sessions:
        rendered_sessions = []
        for idx, session in enumerate(sessions, start=1):
            lines = [f"[Session {idx}]"]
            for turn in session.get("dialogue", []):
                role = turn.get("role", "").capitalize()
                content = turn.get("message") or turn.get("content") or ""
                if role and content:
                    lines.append(f"{role}: {content}")
            rendered_sessions.append("\n".join(lines))
        return "\n\n".join(rendered_sessions) if rendered_sessions else "None"
    return "None"


def format_api_history(example: Dict[str, Any]) -> str:
    api_calls = flatten_api_calls(example)
    if not api_calls:
        return "None"
    return "\n".join(api_calls)


def format_context_block(example: Dict[str, Any], context_type: str) -> str:
    if context_type == "memory_only":
        return "None"
    if context_type == "memory_api":
        return f"[Past API History]:\n{format_api_history(example)}"
    return format_dialogue_history(example)


def parse_api_call_to_dict(api_str: str) -> Tuple[str, Dict[str, str]]:
    if "(" not in api_str:
        return api_str.strip(), {}
    domain = api_str.split("(", 1)[0].strip()
    try:
        args_content = api_str.split("(", 1)[1].rsplit(")", 1)[0]
    except IndexError:
        return domain, {}
    pattern = r'(\w+)=["\']([^"\']+)["\']'
    matches = re.findall(pattern, args_content)
    return domain, {k: v for k, v in matches}


def generate_single_api_string(domain: str, args_dict: Dict[str, str]) -> str:
    parts = [f'{key}="{args_dict[key]}"' for key in sorted(args_dict.keys())]
    return f"{domain}({', '.join(parts)})"


def generate_func_strings(domain: str, slot_values_map: Dict[str, List[str]]) -> List[str]:
    if not slot_values_map:
        return []
    sorted_keys = sorted(slot_values_map.keys())
    values_lists = [slot_values_map[key] for key in sorted_keys]
    combinations = list(itertools.product(*values_lists))
    results: List[str] = []
    for combo in combinations:
        args_dict = {key: value for key, value in zip(sorted_keys, combo)}
        results.append(generate_single_api_string(domain, args_dict))
    return results


def merge_and_generate_api_strings(
    domain: str,
    base_args: Dict[str, str],
    pref_slot_map: Dict[str, List[str]],
) -> List[str]:
    if not pref_slot_map:
        return [generate_single_api_string(domain, base_args)]
    sorted_pref_keys = sorted(pref_slot_map.keys())
    pref_values_lists = [pref_slot_map[key] for key in sorted_pref_keys]
    combinations = list(itertools.product(*pref_values_lists))
    results: List[str] = []
    for combo in combinations:
        current_args = base_args.copy()
        for key, value in zip(sorted_pref_keys, combo):
            current_args[key] = value
        results.append(generate_single_api_string(domain, current_args))
    return results


def format_multiturn_dialogue(query_list: List[Dict[str, str]]) -> str:
    lines = []
    for turn in query_list:
        role = turn.get("role", "User")
        message = turn.get("message", "")
        lines.append(f"{role}: {message}")
    return "\n".join(lines)


def _to_str(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def assign_singleturn_utterances(
    pref_list_path: str,
    example: Dict[str, Any],
    query_map: Dict[str, str],
    pref_type: str,
    pref_group_path: Optional[str] = None,
) -> List[Tuple[str, List[str]]]:
    results: List[Tuple[str, List[str]]] = []

    if pref_type == "easy":
        if not os.path.exists(pref_list_path):
            return []
        pref_list = read_json(pref_list_path)
        api_calls = example.get("api_calls", []) or flatten_api_calls(example)
        if isinstance(api_calls, list):
            for call_str in api_calls:
                domain, args_dict = parse_api_call_to_dict(call_str)
                if domain not in query_map or domain not in pref_list:
                    continue
                current_slot_map = {
                    slot: [_to_str(value)]
                    for slot, value in args_dict.items()
                    if slot in pref_list.get(domain, [])
                }
                if current_slot_map:
                    results.append(
                        (query_map[domain], generate_func_strings(domain, current_slot_map))
                    )
        return results

    prefs = example.get("api_calls_pref", [])
    if not isinstance(prefs, list) or not prefs:
        return []
    if not pref_group_path or not os.path.exists(pref_group_path):
        return []
    pref_group_data = read_json(pref_group_path)

    if pref_type == "medium":
        for pref in prefs:
            group_name = pref.get("value_group")
            if group_name not in pref_group_data:
                continue
            group_rules = pref_group_data[group_name].get("rules", [])
            domain_data_map: Dict[str, Dict[str, set[str]]] = {}
            for evidence in pref.get("evidence", []):
                domain = evidence.get("domain")
                slot = evidence.get("slot")
                if not domain or not slot or domain not in query_map:
                    continue
                candidate_values = [
                    _to_str(rule.get("value"))
                    for rule in group_rules
                    if rule.get("domain") == domain and rule.get("slot") == slot
                ]
                if not candidate_values:
                    continue
                domain_data_map.setdefault(domain, {}).setdefault(slot, set()).update(
                    candidate_values
                )
            for domain, slot_map in domain_data_map.items():
                final_slot_map = {key: sorted(values) for key, values in slot_map.items()}
                results.append(
                    (query_map[domain], generate_func_strings(domain, final_slot_map))
                )
        return results

    if pref_type == "hard":
        for pref in prefs:
            group_name = pref.get("value_group")
            if not group_name or group_name not in pref_group_data:
                continue
            used_domains = {
                evidence.get("domain")
                for evidence in pref.get("evidence", [])
                if evidence.get("domain")
            }
            rules = pref_group_data[group_name].get("rules", [])
            candidate_domains = {
                rule.get("domain")
                for rule in rules
                if rule.get("domain")
                and rule.get("domain") in query_map
                and rule.get("domain") not in used_domains
            }
            for domain in candidate_domains:
                slot_values_map: Dict[str, List[str]] = {}
                for rule in rules:
                    if rule.get("domain") != domain:
                        continue
                    slot = rule.get("slot")
                    value = rule.get("value")
                    if slot and value is not None:
                        slot_values_map.setdefault(slot, [])
                        value_str = _to_str(value)
                        if value_str not in slot_values_map[slot]:
                            slot_values_map[slot].append(value_str)
                if slot_values_map:
                    results.append(
                        (query_map[domain], generate_func_strings(domain, slot_values_map))
                    )
        return results

    return results


def assign_multiturn_utterances(
    pref_list_path: str,
    example: Dict[str, Any],
    multiturn_data: Dict[str, Any],
    pref_type: str,
    pref_group_path: Optional[str] = None,
) -> List[Tuple[str, List[str]]]:
    results: List[Tuple[str, List[str]]] = []

    def process_extraction(domain: str, extracted_pref_map: Dict[str, List[str]]):
        if domain not in multiturn_data:
            return None
        template_data = multiturn_data[domain][0]
        dialogue_text = format_multiturn_dialogue(template_data.get("query", []))
        base_api_str = template_data.get("api_call", [""])[0]
        _, base_args = parse_api_call_to_dict(base_api_str)
        ground_truth = merge_and_generate_api_strings(domain, base_args, extracted_pref_map)
        return (dialogue_text, ground_truth)

    if pref_type == "easy":
        if not os.path.exists(pref_list_path):
            return []
        pref_list = read_json(pref_list_path)
        api_calls = example.get("api_calls", []) or flatten_api_calls(example)
        if isinstance(api_calls, list):
            for call_str in api_calls:
                domain, args_dict = parse_api_call_to_dict(call_str)
                if domain not in multiturn_data or domain not in pref_list:
                    continue
                current_pref_map = {
                    slot: [_to_str(value)]
                    for slot, value in args_dict.items()
                    if slot in pref_list.get(domain, [])
                }
                if current_pref_map:
                    result = process_extraction(domain, current_pref_map)
                    if result:
                        results.append(result)
        return results

    prefs = example.get("api_calls_pref", [])
    if not isinstance(prefs, list) or not prefs:
        return []
    if not pref_group_path or not os.path.exists(pref_group_path):
        return []
    pref_group_data = read_json(pref_group_path)

    if pref_type == "medium":
        for pref in prefs:
            group_name = pref.get("value_group")
            if group_name not in pref_group_data:
                continue
            group_rules = pref_group_data[group_name].get("rules", [])
            domain_data_map: Dict[str, Dict[str, set[str]]] = {}
            for evidence in pref.get("evidence", []):
                domain = evidence.get("domain")
                slot = evidence.get("slot")
                if not domain or not slot or domain not in multiturn_data:
                    continue
                candidate_values = [
                    _to_str(rule.get("value"))
                    for rule in group_rules
                    if rule.get("domain") == domain and rule.get("slot") == slot
                ]
                if not candidate_values:
                    continue
                domain_data_map.setdefault(domain, {}).setdefault(slot, set()).update(
                    candidate_values
                )
            for domain, slot_map in domain_data_map.items():
                final_slot_map = {key: sorted(values) for key, values in slot_map.items()}
                result = process_extraction(domain, final_slot_map)
                if result:
                    results.append(result)
        return results

    if pref_type == "hard":
        for pref in prefs:
            group_name = pref.get("value_group")
            if not group_name or group_name not in pref_group_data:
                continue
            used_domains = {
                evidence.get("domain")
                for evidence in pref.get("evidence", [])
                if evidence.get("domain")
            }
            rules = pref_group_data[group_name].get("rules", [])
            candidate_domains = {
                rule.get("domain")
                for rule in rules
                if rule.get("domain")
                and rule.get("domain") in multiturn_data
                and rule.get("domain") not in used_domains
            }
            for domain in candidate_domains:
                slot_values_map: Dict[str, List[str]] = {}
                for rule in rules:
                    if rule.get("domain") != domain:
                        continue
                    slot = rule.get("slot")
                    value = rule.get("value")
                    if slot and value is not None:
                        slot_values_map.setdefault(slot, [])
                        value_str = _to_str(value)
                        if value_str not in slot_values_map[slot]:
                            slot_values_map[slot].append(value_str)
                if slot_values_map:
                    result = process_extraction(domain, slot_values_map)
                    if result:
                        results.append(result)
        return results

    return results


def prepare_singleturn_items(
    input_path: str,
    query_path: str,
    pref_list_path: str,
    pref_group_path: str,
    pref_type: str,
) -> List[Dict[str, Any]]:
    df = load_chains_dataset(input_path)
    query_map = load_query_map(query_path)
    prepared: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        example = row.to_dict()
        if pref_type == "easy" and not (
            example.get("api_calls") or flatten_api_calls(example)
        ):
            continue
        if pref_type in {"medium", "hard"} and not example.get("api_calls_pref"):
            continue
        pairs = assign_singleturn_utterances(
            pref_list_path=pref_list_path,
            example=example,
            query_map=query_map,
            pref_type=pref_type,
            pref_group_path=pref_group_path,
        )
        for sub_idx, (utterance, ground_truth) in enumerate(pairs):
            prepared.append(
                {
                    "original_ex": example,
                    "utterance": utterance,
                    "ground_truth": ground_truth,
                    "sub_idx": sub_idx,
                }
            )
    return prepared


def prepare_multiturn_items(
    input_path: str,
    multiturn_path: str,
    pref_list_path: str,
    pref_group_path: str,
    pref_type: str,
) -> List[Dict[str, Any]]:
    df = load_chains_dataset(input_path)
    multiturn_data = load_multiturn_data(multiturn_path)
    prepared: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        example = row.to_dict()
        if pref_type == "easy" and not (
            example.get("api_calls") or flatten_api_calls(example)
        ):
            continue
        if pref_type in {"medium", "hard"} and not example.get("api_calls_pref"):
            continue
        pairs = assign_multiturn_utterances(
            pref_list_path=pref_list_path,
            example=example,
            multiturn_data=multiturn_data,
            pref_type=pref_type,
            pref_group_path=pref_group_path,
        )
        for sub_idx, (utterance, ground_truth) in enumerate(pairs):
            prepared.append(
                {
                    "original_ex": example,
                    "utterance": utterance,
                    "ground_truth": ground_truth,
                    "sub_idx": sub_idx,
                }
            )
    return prepared


def build_memory_prompt(
    example: Dict[str, Any],
    retrieved_memories_text: str,
    current_user_utterance: str,
    template: str,
    context_type: str,
    tools_schema: Optional[List[Dict[str, Any]]] = None,
) -> str:
    schema_str = (
        json.dumps(tools_schema, indent=2, ensure_ascii=False)
        if tools_schema
        else "No specific schema provided."
    )
    return template.format(
        preference_schema=schema_str,
        retrieved_memories=retrieved_memories_text,
        dialogue_history=format_context_block(example, context_type),
        user_utterance=current_user_utterance.strip(),
    )


def sanitize_model_label(model_name: str) -> str:
    return model_name.replace("/", "_")


def set_emem_env(
    *,
    llm_base_url: Optional[str] = None,
    embedding_base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    embedding_api_key: Optional[str] = None,
    support_json_schema: str = DEFAULT_SUPPORT_JSON_SCHEMA,
) -> None:
    resolved_key = api_key or embedding_api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved_key and (llm_base_url or embedding_base_url):
        resolved_key = "sk-"
    if resolved_key:
        os.environ["OPENAI_API_KEY"] = resolved_key
    os.environ["SUPPORT_JSON_SCHEMA"] = support_json_schema.lower()


def create_emem_config(
    *,
    workspace_dir: str,
    llm_model: str,
    llm_base_url: Optional[str],
    embedding_model: str,
    embedding_base_url: Optional[str],
    api_key: Optional[str] = None,
    embedding_api_key: Optional[str] = None,
    force_rebuild: bool = False,
    skip_retrieval_ppr: bool = True,
    memory_top_k: int = 5,
    support_json_schema: str = DEFAULT_SUPPORT_JSON_SCHEMA,
) -> BaseConfig:
    set_emem_env(
        llm_base_url=llm_base_url,
        embedding_base_url=embedding_base_url,
        api_key=api_key,
        embedding_api_key=embedding_api_key,
        support_json_schema=support_json_schema,
    )
    return BaseConfig(
        llm_name=llm_model,
        llm_base_url=llm_base_url,
        api_key=api_key,
        embedding_model_name=embedding_model,
        embedding_base_url=embedding_base_url,
        embedding_api_key=embedding_api_key,
        force_openie_from_scratch=force_rebuild,
        force_index_from_scratch=force_rebuild,
        skip_retrieval_ppr=skip_retrieval_ppr,
        skip_edu_context_gen=True,
        query_to_session_retrieval=False,
        date_format_type="longmemeval",
        save_dir=workspace_dir,
        linking_top_k=memory_top_k,
        retrieval_top_k=memory_top_k,
        qa_top_k=memory_top_k,
    )


def manifest_config_dict(
    *,
    llm_model: str,
    llm_base_url: Optional[str],
    embedding_model: str,
    embedding_base_url: Optional[str],
    support_json_schema: str,
    skip_retrieval_ppr: bool,
) -> Dict[str, Any]:
    return {
        "llm_model": llm_model,
        "llm_base_url": llm_base_url,
        "embedding_model": embedding_model,
        "embedding_base_url": embedding_base_url,
        "support_json_schema": support_json_schema.lower(),
        "skip_retrieval_ppr": skip_retrieval_ppr,
        "date_format_type": "longmemeval",
    }


def synthetic_session_datetime(example_position: int, session_index: int) -> str:
    dt = SYNTHETIC_BASE_DATETIME + timedelta(
        days=(example_position * 3) + (session_index - 1),
        hours=session_index - 1,
    )
    return dt.strftime("%Y/%m/%d (%a) %H:%M")


def build_session_turns(session: Dict[str, Any], session_index: int) -> List[Turn]:
    turns: List[Turn] = []
    turn_index = 1
    for turn in session.get("dialogue", []):
        role = str(turn.get("role", "")).lower()
        speaker = "User" if role == "user" else "Assistant"
        content = turn.get("message") or turn.get("content") or ""
        if not content:
            continue
        turns.append(
            Turn(
                speaker=speaker,
                dia_id=f"D{session_index}:{turn_index}",
                text=str(content),
            )
        )
        turn_index += 1

    api_calls = session.get("api_call", [])
    if isinstance(api_calls, str):
        api_calls = [api_calls]
    if isinstance(api_calls, list) and api_calls:
        turns.append(
            Turn(
                speaker="Assistant",
                dia_id=f"D{session_index}:{turn_index}",
                text=f"[System Summary] API Calls executed in this session: {str(api_calls)}",
            )
        )
    return turns


def build_pseudo_conversation(example: Dict[str, Any], example_position: int) -> LoCoMoSample:
    sessions: Dict[int, Session] = {}
    for session_index, session in enumerate(example.get("sessions", []), start=1):
        turns = build_session_turns(session, session_index)
        sessions[session_index] = Session(
            session_id=session_index,
            date_time=synthetic_session_datetime(example_position, session_index),
            turns=turns,
        )

    return LoCoMoSample(
        sample_id=str(example.get("example_id", example_position)),
        qa=[],
        conversation=Conversation(
            speaker_a="User",
            speaker_b="Assistant",
            sessions=sessions,
        ),
        event_summary=EventSummary(events={}),
        observation=Observation(observations={}),
        session_summary={},
    )


def count_workspace_edus(workspace_dir: str, llm_model: str) -> int:
    stats_path = os.path.join(
        workspace_dir,
        f"openie_results_ner_{sanitize_model_label(llm_model)}.pkl",
    )
    if not os.path.exists(stats_path):
        return 0
    with open(stats_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "total_edus" in data:
        return int(data["total_edus"])
    sessions = data.get("sessions", []) if isinstance(data, dict) else []
    return sum(len(session.get("edus", [])) for session in sessions)


def load_manifest_records(manifest_path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not os.path.exists(manifest_path):
        return records
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def materialize_manifest_map(manifest_path: str) -> Dict[str, Dict[str, Any]]:
    manifest_map: Dict[str, Dict[str, Any]] = {}
    for record in load_manifest_records(manifest_path):
        example_id = str(record.get("example_id", ""))
        if example_id:
            manifest_map[example_id] = record
    return manifest_map


def compact_event_pairs(role_argument_pairs: Optional[List[Dict[str, Any]]]) -> Optional[str]:
    if not role_argument_pairs:
        return None
    pairs = []
    for pair in role_argument_pairs:
        role = str(pair.get("role", "")).strip()
        argument = str(pair.get("argument", "")).strip()
        if role and argument:
            pairs.append(f"{role}={argument}")
    if not pairs:
        return None
    return "; ".join(pairs)


def render_edu_memory_line(edu: Any) -> str:
    if hasattr(edu, "to_context_string"):
        base_text = edu.to_context_string(date_format_type="longmemeval")
    else:
        base_text = getattr(edu, "edu_text", str(edu))
    facts = compact_event_pairs(getattr(edu, "event_role_argument_pairs", None))
    if facts:
        return f"{base_text} | facts: {facts}"
    return base_text


def serialize_retrieved_edu(edu: Any, score: Optional[float]) -> Dict[str, Any]:
    metadata = getattr(edu, "metadata", {}) or {}
    date_value = getattr(edu, "date", None)
    if isinstance(date_value, datetime):
        date_value = date_value.isoformat()
    record = {
        "edu_id": getattr(edu, "edu_id", None),
        "edu_text": getattr(edu, "edu_text", None),
        "display_text": render_edu_memory_line(edu),
        "score": float(score) if score is not None else None,
        "date": date_value,
        "source_speakers": list(getattr(edu, "source_speakers", []) or []),
        "source_turn_ids": list(getattr(edu, "source_turn_ids", []) or []),
        "event_type": getattr(edu, "event_type", None),
        "event_role_argument_pairs": getattr(edu, "event_role_argument_pairs", None),
        "edu_type": metadata.get("edu_type"),
    }
    if metadata.get("edu_type") == "assistant_chunk":
        record["chunk_content"] = metadata.get("chunk_content")
    return make_json_safe(record)


def format_retrieved_edus(
    serialized_records: Sequence[Dict[str, Any]],
) -> str:
    if not serialized_records:
        return "No relevant memories found."
    return "\n".join(f"- {record['display_text']}" for record in serialized_records)


def _extract_top_k_indices(query_trace: Dict[str, Any]) -> List[int]:
    post_rerank = (
        query_trace.get("query_to_edu", {})
        .get("post_rerank", {})
    )
    raw_indices = post_rerank.get("top_k_edu_indices") or []
    return [int(index) for index in raw_indices]


def _compute_dense_scores(emem_runtime: EMem, query: str, edu_indices: List[int]) -> List[Optional[float]]:
    if not edu_indices:
        return []
    query_embedding = emem_runtime.query_to_embedding.get("edu", {}).get(query)
    if query_embedding is None:
        return [None] * len(edu_indices)
    query_scores = np.dot(emem_runtime.edu_embeddings, query_embedding.T)
    query_scores = np.squeeze(query_scores) if getattr(query_scores, "ndim", 1) == 2 else query_scores
    results: List[Optional[float]] = []
    for edu_index in edu_indices:
        if 0 <= edu_index < len(query_scores):
            results.append(float(query_scores[edu_index]))
        else:
            results.append(None)
    return results


def retrieve_emem_for_queries(
    manifest_record: Dict[str, Any],
    queries: Sequence[str],
    memory_top_k: int,
    api_key: Optional[str] = None,
    embedding_api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    config = manifest_record.get("config", {})
    workspace_dir = str(manifest_record["workspace_dir"])
    emem_config = create_emem_config(
        workspace_dir=workspace_dir,
        llm_model=config.get("llm_model", DEFAULT_EMEM_LLM_MODEL),
        llm_base_url=config.get("llm_base_url"),
        embedding_model=config.get("embedding_model", DEFAULT_EMEM_EMBEDDING_MODEL),
        embedding_base_url=config.get("embedding_base_url"),
        api_key=api_key,
        embedding_api_key=embedding_api_key,
        force_rebuild=False,
        skip_retrieval_ppr=bool(config.get("skip_retrieval_ppr", True)),
        memory_top_k=memory_top_k,
        support_json_schema=config.get("support_json_schema", DEFAULT_SUPPORT_JSON_SCHEMA),
    )
    runtime = EMem(global_config=emem_config)
    retrieval_output = runtime.retrieve_structuring_conversation_enhanced_v2(
        queries=list(queries),
        num_to_retrieve=memory_top_k,
        no_query_trace_saving=True,
        use_batched_reranking=len(queries) > 1,
    )
    if isinstance(retrieval_output, tuple):
        query_solutions, query_traces = retrieval_output
    else:
        query_solutions = retrieval_output
        query_traces = [{} for _ in query_solutions]

    edu_key_to_idx = {key: idx for idx, key in enumerate(runtime.edu_node_keys)}
    results: List[Dict[str, Any]] = []

    for query, query_solution, query_trace in zip(queries, query_solutions, query_traces):
        retrieved_edus = list(query_solution.edus or [])
        edu_indices = _extract_top_k_indices(query_trace)
        if not edu_indices:
            edu_indices = [
                edu_key_to_idx.get(getattr(edu, "edu_id", ""), -1)
                for edu in retrieved_edus
            ]
            edu_indices = [index for index in edu_indices if index >= 0]
        score_list = _compute_dense_scores(runtime, query, edu_indices)

        serialized_records: List[Dict[str, Any]] = []
        for idx, edu in enumerate(retrieved_edus):
            score = score_list[idx] if idx < len(score_list) else None
            serialized_records.append(serialize_retrieved_edu(edu, score))

        results.append(
            {
                "workspace_dir": workspace_dir,
                "retrieved_edus": serialized_records,
                "retrieved_text": format_retrieved_edus(serialized_records),
                "retrieved_scores": [record.get("score") for record in serialized_records],
                "query_trace": make_json_safe(query_trace),
            }
        )

    return results


async def run_inference_async(
    prepared_items: Sequence[Dict[str, Any]],
    manifest_path: str,
    output_path: str,
    log_path: str,
    run_log_path: Optional[str],
    prompt_template: str,
    context_type: str,
    tools_schema_path: str,
    model_name: str,
    memory_top_k: int,
    concurrency: int,
    reasoning_effort: Optional[str],
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    retrieval_api_key: Optional[str] = None,
    retrieval_embedding_api_key: Optional[str] = None,
) -> None:
    logger = setup_logger("emem_inference", run_log_path)
    logger.info(
        "Starting inference run: items=%d manifest=%s output=%s structured_log=%s model=%s context=%s top_k=%d concurrency=%d",
        len(prepared_items),
        manifest_path,
        output_path,
        log_path,
        model_name,
        context_type,
        memory_top_k,
        concurrency,
    )
    tools_schema = load_tools_from_file(tools_schema_path)
    manifest_map = materialize_manifest_map(manifest_path)
    encoding = get_encoding()
    openai_client = create_async_openai_client(
        api_key=api_key,
        base_url=base_url,
        allow_empty=bool(base_url),
    )
    semaphore = asyncio.Semaphore(concurrency)
    file_lock = asyncio.Lock()
    progress_lock = asyncio.Lock()
    results: List[Optional[Dict[str, Any]]] = [None] * len(prepared_items)
    ensure_parent_dir(log_path)
    with open(log_path, "w", encoding="utf-8"):
        pass

    grouped_items: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for index, item in enumerate(prepared_items):
        prepared_item = dict(item)
        prepared_item["__order__"] = index
        example_id = str(prepared_item["original_ex"].get("example_id", "unknown_user"))
        grouped_items[example_id].append(prepared_item)

    completed = 0
    succeeded = 0
    failed = 0
    progress_interval = max(1, min(25, len(prepared_items))) if prepared_items else 1

    async def process_group(example_id: str, items: List[Dict[str, Any]]) -> None:
        nonlocal completed, succeeded, failed
        async with semaphore:
            manifest_record = manifest_map.get(example_id)
            retrieval_results: List[Dict[str, Any]]

            if not manifest_record or manifest_record.get("build_status") != "OK":
                retrieval_error = "Missing successful manifest record for example."
                retrieval_results = [
                    {
                        "workspace_dir": manifest_record.get("workspace_dir") if manifest_record else None,
                        "retrieved_edus": [],
                        "retrieved_text": "No relevant memories found.",
                        "retrieved_scores": [],
                        "query_trace": {"error": retrieval_error},
                        "retrieval_error": retrieval_error,
                    }
                    for _ in items
                ]
            else:
                try:
                    retrieval_results = await asyncio.to_thread(
                        retrieve_emem_for_queries,
                        manifest_record,
                        [item["utterance"] for item in items],
                        memory_top_k,
                        retrieval_api_key,
                        retrieval_embedding_api_key,
                    )
                except Exception as exc:
                    retrieval_error = f"RETRIEVAL_ERROR: {exc}"
                    logger.exception("Retrieval failed for example_id=%s", example_id)
                    retrieval_results = [
                        {
                            "workspace_dir": manifest_record.get("workspace_dir"),
                            "retrieved_edus": [],
                            "retrieved_text": "No relevant memories found.",
                            "retrieved_scores": [],
                            "query_trace": {"error": retrieval_error},
                            "retrieval_error": retrieval_error,
                        }
                        for _ in items
                    ]

            for item, retrieval in zip(items, retrieval_results):
                example = item["original_ex"]
                utterance = item["utterance"]
                ground_truth = item["ground_truth"]
                sub_idx = item["sub_idx"]
                order = item["__order__"]
                example_id_sub = f"{example_id}_{sub_idx}"
                retrieved_edus = retrieval.get("retrieved_edus", [])
                retrieved_text = retrieval.get("retrieved_text", "No relevant memories found.")
                retrieved_tokens = count_string_tokens(retrieved_text, encoding)

                prompt = build_memory_prompt(
                    example=example,
                    retrieved_memories_text=retrieved_text,
                    current_user_utterance=utterance,
                    template=prompt_template,
                    context_type=context_type,
                    tools_schema=tools_schema,
                )

                retrieval_error = retrieval.get("retrieval_error")
                if retrieval_error:
                    llm_result = {
                        "content": retrieval_error,
                        "reasoning": "",
                        "token_counts": {},
                        "error": retrieval_error,
                    }
                else:
                    llm_result = await call_openai_chat_async(
                        prompt=prompt,
                        model_name=model_name,
                        client=openai_client,
                        reasoning_effort=reasoning_effort,
                    )

                llm_error = llm_result.get("error")
                clean_output = llm_result.get("content", "")
                reasoning_content = llm_result.get("reasoning", "")
                token_counts = llm_result.get("token_counts", {})

                log_record = {
                    "timestamp": now_iso(),
                    "example_id": example_id,
                    "example_id_sub": example_id_sub,
                    "model_name": model_name,
                    "context_type": context_type,
                    "memory_top_k": memory_top_k,
                    "workspace_dir": retrieval.get("workspace_dir"),
                    "injected_utterance": utterance,
                    "reference_ground_truth": ground_truth,
                    "retrieved_memory_count": len(retrieved_edus),
                    "retrieved_memory_tokens": retrieved_tokens,
                    "retrieved_edus": retrieved_edus,
                    "retrieved_scores": retrieval.get("retrieved_scores", []),
                    "retrieval_trace": retrieval.get("query_trace", {}),
                    "status": "ERROR" if llm_error else "OK",
                    "error": llm_error,
                    "model_input": prompt,
                    "model_output": clean_output,
                    "reasoning_content": reasoning_content,
                    "token_counts": token_counts,
                }
                async with file_lock:
                    append_jsonl(log_path, log_record)

                result_ex = copy.deepcopy(example)
                result_ex["example_id_sub"] = example_id_sub
                result_ex["test_utterance"] = utterance
                result_ex["reference_ground_truth"] = ground_truth
                result_ex["workspace_dir"] = retrieval.get("workspace_dir")
                result_ex["retrieved_memories"] = retrieved_edus
                result_ex["retrieved_scores"] = retrieval.get("retrieved_scores", [])
                result_ex["retrieved_memory_count"] = len(retrieved_edus)
                result_ex["retrieved_memory_tokens"] = retrieved_tokens
                result_ex["memory_top_k"] = memory_top_k
                result_ex["status"] = "ERROR" if llm_error else "OK"
                result_ex["error"] = llm_error
                result_ex["llm_output"] = clean_output
                result_ex["reasoning_content"] = reasoning_content
                result_ex["reasoning_token_count"] = token_counts.get("reasoning_tokens", 0)
                result_ex["token_counts"] = token_counts
                result_ex["retrieval_trace"] = retrieval.get("query_trace", {})
                results[order] = result_ex

                async with progress_lock:
                    completed += 1
                    if llm_error:
                        failed += 1
                        logger.error(
                            "Inference item failed: %s status=ERROR retrieved=%d tokens=%d error=%s",
                            example_id_sub,
                            len(retrieved_edus),
                            retrieved_tokens,
                            llm_error,
                        )
                    else:
                        succeeded += 1
                    if (
                        completed == len(prepared_items)
                        or completed % progress_interval == 0
                        or llm_error
                    ):
                        logger.info(
                            "Inference progress: %d/%d complete (ok=%d, error=%d)",
                            completed,
                            len(prepared_items),
                            succeeded,
                            failed,
                        )

    await asyncio.gather(
        *(process_group(example_id, items) for example_id, items in grouped_items.items())
    )

    ordered_results = [result for result in results if result is not None]
    write_json(output_path, ordered_results)
    logger.info(
        "Inference finished: written_results=%d output=%s structured_log=%s",
        len(ordered_results),
        output_path,
        log_path,
    )
    await openai_client.close()
