"""Safe local token measurement for stored and retrieved memory text."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Iterable, Optional, Tuple

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


def encoding_metadata(name: str = "cl100k_base") -> dict:
    encoding, error = get_encoding(name)
    return {
        "token_encoding": name,
        "local_tokenizer_available": encoding is not None,
        "local_tokenizer_error": error,
    }
