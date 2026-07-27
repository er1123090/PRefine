"""Safe local token measurement for stored and retrieved memory text."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Dict, Iterable, Optional, Tuple

try:
    import tiktoken
except ImportError:
    tiktoken = None


@lru_cache(maxsize=None)
def get_encoding(name: str = "cl100k_base") -> Tuple[Any, Optional[str]]:
    if tiktoken is None:
        return None, "tiktoken is not installed"
    try:
        return tiktoken.get_encoding(name), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def count_text_tokens(
    text: Any, encoding_name: str = "cl100k_base"
) -> Optional[int]:
    encoding, _ = get_encoding(encoding_name)
    if encoding is None:
        return None
    if text is None:
        return 0
    if not isinstance(text, str):
        text = str(text)
    return len(encoding.encode(text))


def count_texts_tokens(
    texts: Iterable[Any], encoding_name: str = "cl100k_base"
) -> Optional[int]:
    encoding, _ = get_encoding(encoding_name)
    if encoding is None:
        return None
    total = 0
    for text in texts:
        if text is not None:
            total += len(encoding.encode(str(text)))
    return total


def count_json_tokens(
    value: Any, encoding_name: str = "cl100k_base"
) -> Optional[int]:
    return count_text_tokens(
        json.dumps(value, ensure_ascii=False, sort_keys=True),
        encoding_name,
    )


def memory_construction_lower_bound(
    input_tokens: Optional[int],
    before_memory_tokens: Optional[int],
    after_memory_tokens: Optional[int],
) -> Dict[str, Optional[int]]:
    """Return the E8 Mem0 construction lower bound for one session."""
    if (
        input_tokens is None
        or before_memory_tokens is None
        or after_memory_tokens is None
    ):
        return {
            "stored_memory_delta_tokens": None,
            "construction_input_tokens_lower_bound": input_tokens,
            "construction_output_tokens_lower_bound": None,
            "construction_total_tokens_lower_bound": None,
        }

    delta = after_memory_tokens - before_memory_tokens
    output_lower_bound = max(delta, 0)
    return {
        "stored_memory_delta_tokens": delta,
        "construction_input_tokens_lower_bound": input_tokens,
        "construction_output_tokens_lower_bound": output_lower_bound,
        "construction_total_tokens_lower_bound": input_tokens + output_lower_bound,
    }


def encoding_metadata(name: str = "cl100k_base") -> dict:
    encoding, error = get_encoding(name)
    return {
        "token_encoding": name,
        "local_tokenizer_available": encoding is not None,
        "local_tokenizer_error": error,
    }
