from __future__ import annotations

import importlib
import sys
import typing
from contextlib import contextmanager
from pathlib import Path

from typing_extensions import NotRequired, Required


THIS_DIR = Path(__file__).resolve().parent
REPO_PARENT = THIS_DIR.parent.resolve()


def ensure_langmem_typing_compat() -> None:
    """Patch Python 3.10 typing so newer langmem/langgraph imports can succeed."""
    if not hasattr(typing, "NotRequired"):
        typing.NotRequired = NotRequired  # type: ignore[attr-defined]
    if not hasattr(typing, "Required"):
        typing.Required = Required  # type: ignore[attr-defined]


@contextmanager
def _without_repo_parent_on_path():
    original = list(sys.path)
    filtered: list[str] = []
    for entry in sys.path:
        try:
            resolved = Path(entry or ".").resolve()
        except Exception:
            filtered.append(entry)
            continue
        if resolved == REPO_PARENT:
            continue
        filtered.append(entry)
    sys.path[:] = filtered
    try:
        yield
    finally:
        sys.path[:] = original


def import_langmem_runtime():
    """Import external langmem/langgraph packages without local folder shadowing."""
    ensure_langmem_typing_compat()
    with _without_repo_parent_on_path():
        langmem_mod = importlib.import_module("langmem")
        langgraph_store_memory_mod = importlib.import_module("langgraph.store.memory")
        langchain_openai_mod = importlib.import_module("langchain_openai")
        langchain_core_embeddings_mod = importlib.import_module(
            "langchain_core.embeddings"
        )
    return (
        langmem_mod,
        langgraph_store_memory_mod,
        langchain_openai_mod,
        langchain_core_embeddings_mod,
    )
