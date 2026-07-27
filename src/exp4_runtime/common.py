from __future__ import annotations

import asyncio
import copy
import itertools
import json
import hashlib
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import pandas as pd
except ImportError:
    import tabular as pd
import tiktoken
from openai import AsyncOpenAI, BadRequestError, OpenAI
from pydantic import BaseModel, Field
from tqdm.auto import tqdm
from typing_extensions import Literal

try:
    from .majority_preference import select_query_preferences, select_query_rules
except ImportError:
    from majority_preference import select_query_preferences, select_query_rules

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

from _compat import import_langmem_runtime


DEFAULT_INPUT_PATH = "/data/minseo/experiment8/data/MPT_v2_mix600.json"
DEFAULT_SINGLETURN_QUERY_PATH = "/data/minseo/experiment8/config/query_singleturn_hint.json"
DEFAULT_MULTITURN_QUERY_PATH = "/data/minseo/experiment8/config/query_multiturn_hint.json"
DEFAULT_PREF_LIST_PATH = "/data/minseo/experiment8/config/pref_list.json"
DEFAULT_PREF_GROUP_PATH = "/data/minseo/experiment8/config/pref_group.json"
DEFAULT_SCHEMA_PATH = "/data/minseo/experiment8/config/schema_all.json"
DEFAULT_ENCODING = "cl100k_base"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_NAMESPACE_TEMPLATE = ("langmem", "{user_id}", "semantic")
JSON_PARSE_ERROR_HINT = "could not parse the json body"


MEMORY_INSTRUCTIONS = """Extract durable user preferences and behavioral patterns that can improve future service API selection.

Focus on memories that remain useful across future interactions:
- explicit preferences directly stated by the user
- implicit preferences inferred from repeated selections
- cross-domain behavioral patterns like budget sensitivity, group size, or premium preference
- stable profile facts about the user
- API outcomes that reveal a confirmed choice or preference

Memory requirements:
- `content` must be a concise, standalone sentence that is still understandable when retrieved later.
- Fill `domain`, `slot`, and `value` whenever the memory maps cleanly to a service schema slot.
- Use `category` to distinguish explicit preference, implicit preference, behavior pattern, API outcome, and profile fact.
- Use `context` only when it helps disambiguate the memory.
- Avoid storing transient one-off facts unless they indicate a reusable preference or profile fact.
- Remove or replace stale memories when the latest interaction clearly contradicts them.
"""


EMBEDDING_DIMS_BY_MODEL = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class SemanticMemory(BaseModel):
    content: str = Field(..., description="Concise standalone semantic memory.")
    category: Literal[
        "explicit_preference",
        "implicit_preference",
        "behavior_pattern",
        "api_outcome",
        "profile_fact",
    ] = Field(..., description="Type of semantic memory.")
    domain: Optional[str] = Field(default=None, description="Related API domain if known.")
    slot: Optional[str] = Field(default=None, description="Related API slot if known.")
    value: Optional[str] = Field(default=None, description="Preferred value if known.")
    context: Optional[str] = Field(default=None, description="Short disambiguating context.")


@dataclass
class LangMemRuntime:
    store: Any
    manager: Any
    embedding_dimensions: int


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
    raw_data = read_json(fpath)

    if isinstance(raw_data, dict):
        return raw_data

    if isinstance(raw_data, list):
        query_map = {}
        for item in raw_data:
            if not isinstance(item, dict):
                continue

            domain = None
            targets = item.get("target") or []
            if isinstance(targets, list):
                for target in targets:
                    if isinstance(target, dict) and target.get("domain"):
                        domain = str(target["domain"])
                        break

            if not domain:
                api_calls = item.get("api_call") or []
                if isinstance(api_calls, list):
                    for api_call in api_calls:
                        if isinstance(api_call, str) and "(" in api_call:
                            domain = api_call.split("(", 1)[0].strip()
                            break

            if not domain:
                continue

            utterance = None
            query_turns = item.get("query") or []
            if isinstance(query_turns, list):
                for turn in query_turns:
                    if not isinstance(turn, dict):
                        continue
                    role = str(turn.get("role", "")).lower()
                    message = turn.get("message") or turn.get("content")
                    if role == "user" and message:
                        utterance = str(message).strip()
                        break

            if utterance:
                query_map[domain] = utterance

        return query_map

    return {}


def load_multiturn_data(fpath: str) -> Dict[str, Any]:
    if not os.path.exists(fpath):
        return {}
    raw_data = read_json(fpath)

    if isinstance(raw_data, dict):
        return raw_data

    if isinstance(raw_data, list):
        grouped_data: Dict[str, List[Dict[str, Any]]] = {}
        for item in raw_data:
            if not isinstance(item, dict):
                continue

            domain = None
            targets = item.get("target") or []
            if isinstance(targets, list):
                for target in targets:
                    if isinstance(target, dict) and target.get("domain"):
                        domain = str(target["domain"])
                        break

            if not domain:
                api_calls = item.get("api_call") or []
                if isinstance(api_calls, list):
                    for api_call in api_calls:
                        if isinstance(api_call, str) and "(" in api_call:
                            domain = api_call.split("(", 1)[0].strip()
                            break

            if not domain:
                continue

            grouped_data.setdefault(domain, []).append(item)

        return grouped_data

    return {}


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


def sanitize_text_for_transport(text: Any) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\x00", " ")
    return text.encode("utf-8", errors="replace").decode("utf-8")


def sanitize_texts_for_transport(texts: Sequence[Any]) -> List[str]:
    return [sanitize_text_for_transport(text) for text in texts]


def prepare_text_for_embedding_transport(text: Any) -> str:
    cleaned = sanitize_text_for_transport(text)
    cleaned = "".join(
        char if char.isprintable() or char in "\n\t " else " "
        for char in cleaned
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or " "


def prepare_texts_for_embedding_transport(texts: Sequence[Any]) -> List[str]:
    return [prepare_text_for_embedding_transport(text) for text in texts]


def is_json_transport_parse_error(exc: Exception) -> bool:
    return isinstance(exc, BadRequestError) and JSON_PARSE_ERROR_HINT in str(exc).lower()


def is_gemini_model(model_name: str) -> bool:
    return model_name.lower().startswith("gemini")


def resolve_gemini_model_name(model_name: str) -> str:
    normalized = model_name.strip()
    aliases = {
        "gemini-3-flash": "gemini-3-flash-preview",
        "gemini-3-pro": "gemini-3-pro-preview",
    }
    return aliases.get(normalized, normalized)


def format_text_preview(text: Any, limit: int = 120) -> str:
    cleaned = sanitize_text_for_transport(text).replace("\n", "\\n")
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit]}..."


def resolve_api_key(api_key: Optional[str], allow_empty: bool = False) -> str:
    resolved = (
        api_key
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
    )
    if resolved:
        return resolved
    if allow_empty:
        return "EMPTY"
    raise RuntimeError(
        "OPENAI_API_KEY is required. Pass --api_key or set OPENAI_API_KEY."
    )


def resolve_gemini_api_key(api_key: Optional[str]) -> Optional[str]:
    return (
        api_key
        or os.environ.get("GEMINI_CHAT_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )


def create_async_openai_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    allow_empty: bool = False,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
) -> AsyncOpenAI:
    resolved_api_key = resolve_api_key(api_key, allow_empty=allow_empty)
    kwargs: Dict[str, Any] = {"api_key": resolved_api_key}
    if base_url:
        kwargs["base_url"] = base_url
    if timeout is not None:
        kwargs["timeout"] = timeout
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    return AsyncOpenAI(**kwargs)


def create_openai_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    allow_empty: bool = False,
) -> OpenAI:
    resolved_api_key = resolve_api_key(api_key, allow_empty=allow_empty)
    kwargs: Dict[str, Any] = {"api_key": resolved_api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def use_local_embedding_backend(
    embedding_model: str,
    base_url: Optional[str] = None,
) -> bool:
    return not base_url and embedding_model not in EMBEDDING_DIMS_BY_MODEL


def _local_embedding_device() -> str:
    return os.environ.get("LANGMEM_LOCAL_EMBEDDING_DEVICE", "cpu")


def _local_embedding_max_length() -> int:
    raw = os.environ.get("LANGMEM_LOCAL_EMBEDDING_MAX_LENGTH", "512")
    try:
        return max(8, int(raw))
    except ValueError:
        return 512


def _prepare_local_embedding_texts(
    embedding_model: str,
    texts: List[str],
    mode: str,
) -> List[str]:
    lowered = embedding_model.lower()
    if "e5" not in lowered:
        return texts
    prefix = "query: " if mode == "query" else "passage: "
    return [prefix + text for text in texts]


def _build_local_transformers_components(embedding_model: str):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    model = AutoModel.from_pretrained(embedding_model)
    model.eval()
    device = _local_embedding_device()
    if device and device != "cpu":
        model.to(device)
    return tokenizer, model


def build_local_transformers_embeddings(embedding_model: str):
    import torch
    import torch.nn.functional as F

    _, _, _, langchain_core_embeddings_mod = import_langmem_runtime()
    Embeddings = langchain_core_embeddings_mod.Embeddings
    tokenizer, model = _build_local_transformers_components(embedding_model)
    device = next(model.parameters()).device
    max_length = _local_embedding_max_length()

    class LocalTransformersEmbeddings(Embeddings):
        def __init__(self) -> None:
            self.device = device
            self.max_length = max_length

        def _encode(self, texts: List[str], mode: str) -> List[List[float]]:
            if not texts:
                return []
            prepared_texts = _prepare_local_embedding_texts(embedding_model, texts, mode)
            encoded = tokenizer(
                prepared_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(self.device)
                for key, value in encoded.items()
            }
            with torch.no_grad():
                outputs = model(**encoded)
                hidden = outputs.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).expand_as(hidden).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                normalized = F.normalize(pooled, p=2, dim=1)
            return normalized.cpu().tolist()

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return self._encode(texts, mode="document")

        def embed_query(self, text: str) -> List[float]:
            return self._encode([text], mode="query")[0]

        async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
            return await asyncio.to_thread(self._encode, texts, "document")

        async def aembed_query(self, text: str) -> List[float]:
            return (await self.aembed_documents([text]))[0]

    return LocalTransformersEmbeddings()


def infer_embedding_dimensions(
    embedding_model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> int:
    if embedding_model in EMBEDDING_DIMS_BY_MODEL:
        return EMBEDDING_DIMS_BY_MODEL[embedding_model]
    if use_local_embedding_backend(embedding_model, base_url=base_url):
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(embedding_model)
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            raise RuntimeError(
                f"Could not infer local embedding dimensions for {embedding_model}"
            )
        return int(hidden_size)
    client = create_openai_client(
        api_key=api_key,
        base_url=base_url,
        allow_empty=bool(base_url),
    )
    try:
        response = client.embeddings.create(model=embedding_model, input=["dimension probe"])
    except Exception as exc:
        if is_json_transport_parse_error(exc):
            endpoint = base_url or "<provider default>"
            raise RuntimeError(
                f"Embedding dimension probe failed with a JSON transport parse error "
                f"model={embedding_model} base_url={endpoint}: {exc}"
            ) from exc
        raise
    if not response.data:
        raise RuntimeError(f"Could not infer embedding dimensions for {embedding_model}")
    return len(response.data[0].embedding)


def build_openai_compatible_embeddings(
    embedding_model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
):
    _, _, _, langchain_core_embeddings_mod = import_langmem_runtime()
    Embeddings = langchain_core_embeddings_mod.Embeddings

    class OpenAICompatibleEmbeddings(Embeddings):
        def __init__(self) -> None:
            self.sync_client = create_openai_client(
                api_key=api_key,
                base_url=base_url,
                allow_empty=bool(base_url),
            )
            self.async_client = create_async_openai_client(
                api_key=api_key,
                base_url=base_url,
                allow_empty=bool(base_url),
            )

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            if not texts:
                return []
            prepared_texts = prepare_texts_for_embedding_transport(texts)
            try:
                response = self.sync_client.embeddings.create(
                    model=embedding_model,
                    input=prepared_texts,
                )
                return [item.embedding for item in response.data]
            except Exception as exc:
                if not is_json_transport_parse_error(exc):
                    raise
                if len(prepared_texts) <= 1:
                    preview = format_text_preview(prepared_texts[0] if prepared_texts else "")
                    endpoint = base_url or "<provider default>"
                    raise RuntimeError(
                        f"Embedding request failed with a JSON transport parse error "
                        f"model={embedding_model} base_url={endpoint} text_preview={preview!r}: {exc}"
                    ) from exc
                embeddings: List[List[float]] = []
                for index, text in enumerate(prepared_texts):
                    try:
                        response = self.sync_client.embeddings.create(
                            model=embedding_model,
                            input=[text],
                        )
                    except Exception as inner_exc:
                        preview = format_text_preview(text)
                        endpoint = base_url or "<provider default>"
                        raise RuntimeError(
                            f"Embedding request failed after batch fallback at index={index} "
                            f"model={embedding_model} base_url={endpoint} "
                            f"text_preview={preview!r}: {inner_exc}"
                        ) from inner_exc
                    embeddings.append(response.data[0].embedding)
                return embeddings

        def embed_query(self, text: str) -> List[float]:
            return self.embed_documents([text])[0]

        async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
            if not texts:
                return []
            prepared_texts = prepare_texts_for_embedding_transport(texts)
            try:
                response = await self.async_client.embeddings.create(
                    model=embedding_model,
                    input=prepared_texts,
                )
                return [item.embedding for item in response.data]
            except Exception as exc:
                if not is_json_transport_parse_error(exc):
                    raise
                if len(prepared_texts) <= 1:
                    preview = format_text_preview(prepared_texts[0] if prepared_texts else "")
                    endpoint = base_url or "<provider default>"
                    raise RuntimeError(
                        f"Async embedding request failed with a JSON transport parse error "
                        f"model={embedding_model} base_url={endpoint} text_preview={preview!r}: {exc}"
                    ) from exc
                embeddings: List[List[float]] = []
                for index, text in enumerate(prepared_texts):
                    try:
                        response = await self.async_client.embeddings.create(
                            model=embedding_model,
                            input=[text],
                        )
                    except Exception as inner_exc:
                        preview = format_text_preview(text)
                        endpoint = base_url or "<provider default>"
                        raise RuntimeError(
                            f"Async embedding request failed after batch fallback at index={index} "
                            f"model={embedding_model} base_url={endpoint} "
                            f"text_preview={preview!r}: {inner_exc}"
                        ) from inner_exc
                    embeddings.append(response.data[0].embedding)
                return embeddings

        async def aembed_query(self, text: str) -> List[float]:
            return (await self.aembed_documents([text]))[0]

    return OpenAICompatibleEmbeddings()


def build_embedding_backend(
    embedding_model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
):
    if use_local_embedding_backend(embedding_model, base_url=base_url):
        return build_local_transformers_embeddings(embedding_model)
    return build_openai_compatible_embeddings(
        embedding_model=embedding_model,
        api_key=api_key,
        base_url=base_url,
    )


def create_langmem_runtime(
    memory_model: str,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    embedding_base_url: Optional[str] = None,
    embedding_api_key: Optional[str] = None,
    memory_base_url: Optional[str] = None,
    memory_api_key: Optional[str] = None,
    namespace_template: Tuple[str, ...] = DEFAULT_NAMESPACE_TEMPLATE,
    query_limit: int = 5,
    enable_deletes: bool = True,
    memory_schema: str = "semantic",
    prompt_style: str = "custom",
) -> LangMemRuntime:
    langmem_mod, langgraph_store_memory_mod, langchain_openai_mod, _ = import_langmem_runtime()
    create_memory_store_manager = langmem_mod.create_memory_store_manager
    InMemoryStore = langgraph_store_memory_mod.InMemoryStore
    ChatOpenAI = langchain_openai_mod.ChatOpenAI
    resolved_memory_base_url = memory_base_url if memory_base_url is not None else base_url
    resolved_memory_api_key = memory_api_key if memory_api_key is not None else api_key
    if embedding_base_url is not None:
        resolved_embedding_base_url = embedding_base_url
    elif use_local_embedding_backend(embedding_model):
        resolved_embedding_base_url = None
    else:
        resolved_embedding_base_url = base_url
    if embedding_api_key is not None:
        resolved_embedding_api_key = embedding_api_key
    elif resolved_embedding_base_url:
        resolved_embedding_api_key = api_key
    else:
        resolved_embedding_api_key = None

    embedding_dimensions = infer_embedding_dimensions(
        embedding_model=embedding_model,
        api_key=resolved_embedding_api_key,
        base_url=resolved_embedding_base_url,
    )
    embeddings = build_embedding_backend(
        embedding_model=embedding_model,
        api_key=resolved_embedding_api_key,
        base_url=resolved_embedding_base_url,
    )
    store = InMemoryStore(
        index={
            "dims": embedding_dimensions,
            "embed": embeddings,
        }
    )
    chat_model = ChatOpenAI(
        model=memory_model,
        api_key=resolve_api_key(
            resolved_memory_api_key,
            allow_empty=bool(resolved_memory_base_url),
        ),
        base_url=resolved_memory_base_url,
        temperature=0.0,
    )
    manager_kwargs: Dict[str, Any] = {
        "namespace": namespace_template,
        "store": store,
        "enable_inserts": True,
        "enable_deletes": enable_deletes,
        "query_limit": query_limit,
    }
    if memory_schema == "semantic":
        manager_kwargs["schemas"] = [SemanticMemory]
    elif memory_schema != "string":
        raise ValueError(f"Unsupported memory schema: {memory_schema}")

    if prompt_style == "custom":
        manager_kwargs["instructions"] = MEMORY_INSTRUCTIONS
    elif prompt_style != "default":
        raise ValueError(f"Unsupported prompt style: {prompt_style}")

    manager = create_memory_store_manager(
        chat_model,
        **manager_kwargs,
    )
    return LangMemRuntime(
        store=store,
        manager=manager,
        embedding_dimensions=embedding_dimensions,
    )


def make_runtime_config(user_id: str) -> Dict[str, Any]:
    return {"configurable": {"user_id": str(user_id)}}


def materialize_namespace(user_id: str) -> Tuple[str, ...]:
    return ("langmem", str(user_id), "semantic")


def build_session_messages(session: Dict[str, Any]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    for turn in session.get("dialogue", []):
        role = str(turn.get("role", "")).lower()
        content = turn.get("message") or turn.get("content") or ""
        if not role or not content:
            continue
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
    turns = example.get("turns", [])
    if not turns:
        return "None"
    return "\n".join(
        f"{turn.get('speaker', 'User')}: {turn.get('utterance', '')}" for turn in turns
    )


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

    prefs = select_query_preferences(example)
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
            group_rules = select_query_rules(
                example,
                pref,
                pref_group_data[group_name].get("rules", []),
            )
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
            rules = select_query_rules(
                example,
                pref,
                pref_group_data[group_name].get("rules", []),
            )
            candidate_domains = {
                rule.get("domain")
                for rule in rules
                if rule.get("domain")
                and rule.get("domain") in query_map
                and rule.get("domain") not in used_domains
            }
            for domain in sorted(candidate_domains):
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

    prefs = select_query_preferences(example)
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
            group_rules = select_query_rules(
                example,
                pref,
                pref_group_data[group_name].get("rules", []),
            )
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
            rules = select_query_rules(
                example,
                pref,
                pref_group_data[group_name].get("rules", []),
            )
            candidate_domains = {
                rule.get("domain")
                for rule in rules
                if rule.get("domain")
                and rule.get("domain") in multiturn_data
                and rule.get("domain") not in used_domains
            }
            for domain in sorted(candidate_domains):
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


def normalize_memory_value(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        if isinstance(value.get("content"), dict):
            payload = value["content"]
            if isinstance(payload, dict):
                return payload
        return value
    return {"content": str(value)}


def render_memory_line(value: Any) -> str:
    payload = normalize_memory_value(value)
    ordered_keys = ["content", "category", "domain", "slot", "value", "context"]
    parts = []
    for key in ordered_keys:
        raw_value = payload.get(key)
        if raw_value in (None, "", [], {}):
            continue
        parts.append(f"{key}={raw_value}")
    if parts:
        return " | ".join(parts)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def make_snapshot_item(item: Any) -> Dict[str, Any]:
    return {
        "namespace": list(getattr(item, "namespace", ()) or ()),
        "key": getattr(item, "key", ""),
        "value": getattr(item, "value", {}),
        "created_at": getattr(item, "created_at", None).isoformat()
        if getattr(item, "created_at", None)
        else None,
        "updated_at": getattr(item, "updated_at", None).isoformat()
        if getattr(item, "updated_at", None)
        else None,
        "score": getattr(item, "score", None),
    }


def serialize_memory_objects(memory_items: Sequence[Dict[str, Any]]) -> List[str]:
    serialized = []
    for item in memory_items:
        serialized.append(
            json_dumps(
                {
                    "key": item.get("key"),
                    "value": item.get("value"),
                }
            )
        )
    return serialized


def count_serialized_memory_tokens(
    memory_items: Sequence[Dict[str, Any]],
    encoding,
) -> int:
    return count_text_tokens(serialize_memory_objects(memory_items), encoding)


def format_retrieved_memories(
    items: Sequence[Any],
) -> Tuple[List[Dict[str, Any]], str]:
    raw_records: List[Dict[str, Any]] = []
    rendered: List[str] = []
    for item in items:
        raw = make_snapshot_item(item)
        raw["display_text"] = render_memory_line(raw.get("value"))
        raw_records.append(raw)
        rendered.append(f"- {raw['display_text']}")
    if not rendered:
        return raw_records, "No relevant memories found."
    return raw_records, "\n".join(rendered)


def list_namespace_items(
    store: Any,
    namespace: Tuple[str, ...],
    limit: int = 1000,
) -> List[Any]:
    all_items: List[Any] = []
    offset = 0
    while True:
        batch = store.search(namespace, limit=limit, offset=offset)
        if not batch:
            break
        all_items.extend(batch)
        if len(batch) < limit:
            break
        offset += limit
    return sorted(all_items, key=lambda item: getattr(item, "key", ""))


def search_namespace_items(
    store: Any,
    namespace: Tuple[str, ...],
    query: str,
    limit: int,
) -> List[Any]:
    if not query:
        return []
    return store.search(namespace, query=query, limit=limit)


def export_snapshot_record(
    example_id: str,
    namespace: Tuple[str, ...],
    memory_items: Sequence[Any],
    session_exports: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    return {
        "example_id": str(example_id),
        "namespace": list(namespace),
        "memory_items": [make_snapshot_item(item) for item in memory_items],
        "session_exports": session_exports or [],
    }


def load_snapshot_records(snapshot_path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(snapshot_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def restore_store_from_snapshot(
    snapshot_path: str,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    embedding_base_url: Optional[str] = None,
    embedding_api_key: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Any:
    runtime_base_url = embedding_base_url or base_url
    runtime_api_key = embedding_api_key or os.environ.get("OPENAI_API_KEY")
    if runtime_base_url and not runtime_api_key:
        runtime_api_key = api_key
    if not runtime_base_url and not runtime_api_key:
        # Snapshot restore only needs the embedding-backed store, but the
        # LangMem runtime also instantiates a chat model. Use a dummy local
        # endpoint when no chat credentials are otherwise required.
        runtime_base_url = "http://127.0.0.1:9/v1"
        runtime_api_key = "EMPTY"

    runtime = create_langmem_runtime(
        memory_model="gpt-4o-mini",
        embedding_model=embedding_model,
        base_url=runtime_base_url,
        api_key=runtime_api_key,
        embedding_base_url=embedding_base_url,
        embedding_api_key=embedding_api_key,
        enable_deletes=True,
    )
    store = runtime.store
    records = load_snapshot_records(snapshot_path)
    total_memory_items = sum(len(record.get("memory_items", [])) for record in records)
    restored_memory_count = 0
    progress_bar = tqdm(
        total=total_memory_items,
        desc=f"Restore {Path(snapshot_path).parent.name}",
        unit="mem",
        dynamic_ncols=True,
        leave=False,
    )
    try:
        for record in records:
            namespace = tuple(record.get("namespace") or ())
            for item in record.get("memory_items", []):
                key = str(item.get("key", ""))
                value = item.get("value", {})
                if namespace and key:
                    store.put(namespace, key, value)
                    restored_memory_count += 1
                progress_bar.update(1)
    finally:
        progress_bar.close()

    if logger:
        logger.info(
            "Restored snapshot from %s with %d records and %d memory items",
            snapshot_path,
            len(records),
            restored_memory_count,
        )
    return store


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
        sanitized_prompt = sanitize_text_for_transport(prompt)
        kwargs: Dict[str, Any] = {
            "model": model_name,
            "messages": [{"role": "user", "content": sanitized_prompt}],
        }
        is_reasoning_model = any(
            token in model_name.lower() for token in ["o1", "o3", "gpt-5"]
        )
        if is_reasoning_model and reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort.lower()
        fallback_kwargs = dict(kwargs)
        fallback_kwargs.pop("reasoning_effort", None)

        try:
            response = await client.chat.completions.create(**kwargs)
        except Exception as exc:
            should_retry_without_reasoning = (
                "reasoning_effort" in kwargs
                and (
                    is_json_transport_parse_error(exc)
                    or isinstance(exc, BadRequestError)
                )
            )
            if not should_retry_without_reasoning:
                raise
            response = await client.chat.completions.create(**fallback_kwargs)

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


async def call_gemini_chat_async(
    prompt: str,
    model_name: str,
    api_key: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> Dict[str, Any]:
    result = {
        "content": "",
        "reasoning": "",
        "token_counts": {},
        "error": None,
    }
    resolved_api_key = resolve_gemini_api_key(api_key)
    if not resolved_api_key:
        result["error"] = "API_KEY_MISSING_GOOGLE"
        return result
    if genai is None or genai_types is None:
        result["error"] = "google-genai library not installed"
        return result

    try:
        resolved_model_name = resolve_gemini_model_name(model_name)
        client = genai.Client(api_key=resolved_api_key)
        config_params: Dict[str, Any] = {"temperature": 0.0}
        if reasoning_effort:
            config_params["thinking_config"] = genai_types.ThinkingConfig(
                include_thoughts=True,
                thinking_level=reasoning_effort.lower(),
            )
        response = await client.aio.models.generate_content(
            model=resolved_model_name,
            contents=sanitize_text_for_transport(prompt),
            config=genai_types.GenerateContentConfig(**config_params),
        )

        thought_parts: List[str] = []
        answer_parts: List[str] = []
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            content = getattr(candidates[0], "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                text = getattr(part, "text", None)
                if not text:
                    continue
                if getattr(part, "thought", False):
                    thought_parts.append(text)
                else:
                    answer_parts.append(text)

        if not answer_parts:
            raw_text = getattr(response, "text", "") or ""
            if raw_text:
                answer_parts = [raw_text]

        result["content"] = "\n".join(answer_parts).strip()
        result["reasoning"] = "\n".join(thought_parts).strip()

        usage_metadata = getattr(response, "usage_metadata", None)
        if usage_metadata:
            result["token_counts"] = {
                "total_tokens": getattr(usage_metadata, "total_token_count", 0),
                "input_tokens": getattr(usage_metadata, "prompt_token_count", 0),
                "output_tokens": getattr(usage_metadata, "candidates_token_count", 0),
                "reasoning_tokens": getattr(usage_metadata, "thoughts_token_count", 0),
            }
        return result
    except Exception as exc:
        result["error"] = f"API_ERROR: {exc}"
        return result


async def call_chat_async(
    prompt: str,
    model_name: str,
    openai_client: Optional[AsyncOpenAI] = None,
    api_key: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> Dict[str, Any]:
    if is_gemini_model(model_name) and openai_client is None:
        return await call_gemini_chat_async(
            prompt=prompt,
            model_name=model_name,
            api_key=api_key,
            reasoning_effort=reasoning_effort,
        )
    if openai_client is None:
        return {
            "content": "",
            "reasoning": "",
            "token_counts": {},
            "error": "API_KEY_MISSING_OPENAI",
        }
    return await call_openai_chat_async(
        prompt=prompt,
        model_name=model_name,
        client=openai_client,
        reasoning_effort=reasoning_effort,
    )


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


def write_json(path: str, data: Any) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def last_user_utterance(session: Dict[str, Any]) -> str:
    dialogue = session.get("dialogue", [])
    for turn in reversed(dialogue):
        role = str(turn.get("role", "")).lower()
        message = turn.get("message") or turn.get("content") or ""
        if role == "user" and message:
            return message
    if dialogue:
        last_turn = dialogue[-1]
        return last_turn.get("message") or last_turn.get("content") or ""
    return ""


def session_input_token_count(session: Dict[str, Any], encoding) -> int:
    return count_text_tokens(
        (message.get("content", "") for message in build_session_messages(session)),
        encoding,
    )


def limit_prepared_items(
    prepared_items: Sequence[Dict[str, Any]],
    max_queries: Optional[int],
) -> List[Dict[str, Any]]:
    prepared_list = list(prepared_items)
    if max_queries is None:
        return prepared_list
    if max_queries <= 0:
        raise ValueError(f"max_queries must be positive, got {max_queries}")

    total_items = len(prepared_list)
    if total_items <= max_queries:
        return prepared_list

    if max_queries == 1:
        selected_indices = [0]
    else:
        selected_indices = [
            ((total_items - 1) * idx) // (max_queries - 1)
            for idx in range(max_queries)
        ]

    limited_items = [prepared_list[idx] for idx in selected_indices]
    print(
        f"[Info] Limiting prepared items from {total_items} to {len(limited_items)} "
        f"with deterministic evenly spaced sampling (max_queries={max_queries})."
    )
    return limited_items


def prepare_singleturn_items(
    input_path: str,
    query_path: str,
    pref_list_path: str,
    pref_group_path: str,
    pref_type: str,
    max_queries: Optional[int] = None,
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
    return limit_prepared_items(prepared, max_queries)


def prepare_multiturn_items(
    input_path: str,
    multiturn_path: str,
    pref_list_path: str,
    pref_group_path: str,
    pref_type: str,
    max_queries: Optional[int] = None,
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
    return limit_prepared_items(prepared, max_queries)


async def run_inference_async(
    prepared_items: Sequence[Dict[str, Any]],
    snapshot_path: str,
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
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    embedding_base_url: Optional[str] = None,
    embedding_api_key: Optional[str] = None,
    store: Optional[Any] = None,
) -> None:
    logger = setup_logger("langmem_inference", run_log_path)
    logger.info(
        "Starting inference run: items=%d snapshot=%s output=%s structured_log=%s model=%s context=%s top_k=%d concurrency=%d",
        len(prepared_items),
        snapshot_path,
        output_path,
        log_path,
        model_name,
        context_type,
        memory_top_k,
        concurrency,
    )
    tools_schema = load_tools_from_file(tools_schema_path)
    if store is None:
        store = restore_store_from_snapshot(
            snapshot_path=snapshot_path,
            embedding_model=embedding_model,
            base_url=base_url,
            api_key=api_key,
            embedding_base_url=embedding_base_url,
            embedding_api_key=embedding_api_key,
            logger=logger,
        )
    else:
        logger.info("Reusing preloaded snapshot store for %s", snapshot_path)
    encoding = get_encoding()
    openai_client: Optional[AsyncOpenAI] = None
    if not (is_gemini_model(model_name) and not base_url):
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
    completed = 0
    succeeded = 0
    failed = 0
    progress_interval = max(1, min(25, len(prepared_items))) if prepared_items else 1
    inference_progress_bar = tqdm(
        total=len(prepared_items),
        desc=f"Infer {Path(snapshot_path).parent.name}",
        unit="item",
        dynamic_ncols=True,
        leave=False,
    )

    if not prepared_items:
        logger.warning("No prepared inference items found. Writing an empty result file.")

    async def process_item(index: int, item: Dict[str, Any]) -> None:
        nonlocal completed, succeeded, failed
        async with semaphore:
            example = item["original_ex"]
            utterance = item["utterance"]
            ground_truth = item["ground_truth"]
            sub_idx = item["sub_idx"]
            example_id = str(example.get("example_id", "unknown_user"))
            try:
                namespace = materialize_namespace(example_id)
                retrieved_items = await asyncio.to_thread(
                    search_namespace_items,
                    store,
                    namespace,
                    utterance,
                    memory_top_k,
                )
                retrieved_memories, retrieved_memories_text = format_retrieved_memories(
                    retrieved_items
                )
                retrieved_memory_tokens = count_string_tokens(retrieved_memories_text, encoding)
                prompt = build_memory_prompt(
                    example=example,
                    retrieved_memories_text=retrieved_memories_text,
                    current_user_utterance=utterance,
                    template=prompt_template,
                    context_type=context_type,
                    tools_schema=tools_schema,
                )
                llm_result = await call_chat_async(
                    prompt=prompt,
                    model_name=model_name,
                    openai_client=openai_client,
                    api_key=api_key,
                    reasoning_effort=reasoning_effort,
                )
                llm_error = llm_result.get("error")
                if llm_error:
                    clean_output = llm_error
                    reasoning_content = ""
                    token_counts: Dict[str, Any] = {}
                else:
                    clean_output = llm_result["content"]
                    reasoning_content = llm_result["reasoning"]
                    token_counts = llm_result["token_counts"]

                log_record = {
                    "timestamp": now_iso(),
                    "example_id": example_id,
                    "example_id_sub": f"{example_id}_{sub_idx}",
                    "model_name": model_name,
                    "context_type": context_type,
                    "injected_utterance": utterance,
                    "memory_top_k": memory_top_k,
                    "retrieved_memory_count": len(retrieved_memories),
                    "retrieved_memory_tokens": retrieved_memory_tokens,
                    "retrieved_memories": retrieved_memories,
                    "reference_ground_truth": ground_truth,
                    "status": "ERROR" if llm_error else "OK",
                    "error": llm_error,
                    "model_input": prompt,
                    "model_output": clean_output,
                    "reasoning_content": reasoning_content,
                    "token_counts": token_counts,
                }
                async with file_lock:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_record, ensure_ascii=False) + "\n")

                result_ex = copy.deepcopy(example)
                result_ex["example_id_sub"] = f"{example_id}_{sub_idx}"
                result_ex["test_utterance"] = utterance
                result_ex["reference_ground_truth"] = ground_truth
                result_ex["retrieved_memories"] = retrieved_memories
                result_ex["retrieved_memory_count"] = len(retrieved_memories)
                result_ex["retrieved_memory_tokens"] = retrieved_memory_tokens
                result_ex["memory_top_k"] = memory_top_k
                result_ex["status"] = "ERROR" if llm_error else "OK"
                result_ex["error"] = llm_error
                result_ex["llm_output"] = clean_output
                result_ex["reasoning_content"] = reasoning_content
                result_ex["reasoning_token_count"] = token_counts.get("reasoning_tokens", 0)
                result_ex["token_counts"] = token_counts
                results[index] = result_ex

                async with progress_lock:
                    completed += 1
                    inference_progress_bar.update(1)
                    if llm_error:
                        failed += 1
                        logger.error(
                            "Inference item failed: %s status=ERROR retrieved=%d tokens=%d error=%s",
                            result_ex["example_id_sub"],
                            len(retrieved_memories),
                            retrieved_memory_tokens,
                            llm_error,
                        )
                    else:
                        succeeded += 1
                    inference_progress_bar.set_postfix(
                        ok=succeeded,
                        err=failed,
                        refresh=False,
                    )
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
            except Exception:
                logger.exception(
                    "Inference crashed before completion: example_id=%s sub_idx=%s",
                    example_id,
                    sub_idx,
                )
                raise

    tasks = [asyncio.create_task(process_item(i, item)) for i, item in enumerate(prepared_items)]
    try:
        if tasks:
            await asyncio.gather(*tasks)
    finally:
        inference_progress_bar.close()

    final_results = [item for item in results if item is not None]
    write_json(output_path, final_results)
    logger.info(
        "Inference finished: total=%d ok=%d error=%d output=%s structured_log=%s",
        len(prepared_items),
        succeeded,
        failed,
        output_path,
        log_path,
    )
