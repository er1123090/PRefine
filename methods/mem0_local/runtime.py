"""Runtime adapter around an unmodified official Mem0 checkout."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from src.construction_usage import record_response_usage


HERE = Path(__file__).resolve().parent
DEFAULT_UPSTREAM_PATH = HERE / "upstream" / "mem0"
UPSTREAM_LOCK_PATH = HERE / "UPSTREAM.json"


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def load_upstream_lock() -> dict[str, Any]:
    return json.loads(UPSTREAM_LOCK_PATH.read_text(encoding="utf-8"))


def upstream_metadata(repo_path: Path = DEFAULT_UPSTREAM_PATH) -> dict[str, Any]:
    repo_path = _resolved(repo_path)
    lock = load_upstream_lock()
    metadata: dict[str, Any] = {
        **lock,
        "repo_path": str(repo_path),
        "source_import_path": str(repo_path / "mem0"),
        "checkout_present": (repo_path / "mem0" / "memory" / "main.py").is_file(),
    }
    if (repo_path / ".git").exists():
        try:
            metadata["checkout_commit"] = subprocess.run(
                ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            metadata["checkout_clean"] = not bool(
                subprocess.run(
                    ["git", "-C", str(repo_path), "status", "--porcelain"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            )
        except (OSError, subprocess.CalledProcessError):
            metadata["checkout_commit"] = None
            metadata["checkout_clean"] = False
    else:
        metadata["checkout_commit"] = None
        metadata["checkout_clean"] = False
    metadata["matches_lock"] = (
        metadata["checkout_commit"] == lock["commit"]
        and metadata["checkout_clean"]
    )
    return metadata


def load_official_memory(repo_path: Path = DEFAULT_UPSTREAM_PATH):
    """Import Memory from the pinned checkout, never silently from site-packages."""
    repo_path = _resolved(repo_path)
    expected_module = repo_path / "mem0" / "__init__.py"
    if not expected_module.is_file():
        raise RuntimeError(
            f"Official Mem0 checkout is missing at {repo_path}. Run "
            "`python methods/mem0_local/bootstrap_upstream.py` first."
        )

    loaded = sys.modules.get("mem0")
    if loaded is not None:
        loaded_path = _resolved(Path(getattr(loaded, "__file__", "")))
        if not loaded_path.is_relative_to(repo_path):
            raise RuntimeError(
                "mem0 was imported from site-packages before the official checkout: "
                f"{loaded_path}"
            )

    repo_string = str(repo_path)
    if repo_string not in sys.path:
        sys.path.insert(0, repo_string)
    os.environ.setdefault("MEM0_TELEMETRY", "False")

    module = importlib.import_module("mem0")
    module_path = _resolved(Path(module.__file__))
    if not module_path.is_relative_to(repo_path):
        raise RuntimeError(
            "Failed to load the official Mem0 checkout; imported "
            f"{module_path} instead of a module under {repo_path}"
        )
    return module.Memory


def build_mem0_config(
    *,
    model: str,
    base_url: str,
    vector_store_path: Path,
    history_db_path: Path,
    collection_name: str,
    max_tokens: int,
    embedding_provider: str,
    embedding_model: str,
    embedding_dims: int,
    embedding_api_key: str | None = None,
    embedding_base_url: str | None = None,
) -> dict[str, Any]:
    if embedding_provider not in {"fastembed", "huggingface", "openai"}:
        raise ValueError(
            "embedding_provider must be fastembed, huggingface, or openai"
        )
    if embedding_dims < 1:
        raise ValueError("embedding_dims must be positive")

    embedder_config: dict[str, Any] = {
        "model": embedding_model,
        "embedding_dims": embedding_dims,
    }
    if embedding_provider == "huggingface":
        embedder_config["model_kwargs"] = {"device": "cpu"}
    elif embedding_provider == "openai":
        if embedding_api_key:
            embedder_config["api_key"] = embedding_api_key
        if embedding_base_url:
            embedder_config["openai_base_url"] = embedding_base_url

    return {
        "version": "v1.1",
        "llm": {
            "provider": "vllm",
            "config": {
                "model": model,
                "vllm_base_url": base_url.rstrip("/"),
                "api_key": "EMPTY",
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "top_p": 1.0,
            },
        },
        "embedder": {
            "provider": embedding_provider,
            "config": embedder_config,
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": collection_name,
                "path": str(_resolved(vector_store_path)),
                "embedding_model_dims": embedding_dims,
                "on_disk": True,
            },
        },
        "history_db_path": str(_resolved(history_db_path)),
    }


class _TrackedCompletions:
    def __init__(
        self,
        delegate: Any,
        *,
        model: str,
        reasoning_effort: str | None,
        max_completion_tokens: int | None,
        disable_response_format: bool,
    ):
        self._delegate = delegate
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._max_completion_tokens = max_completion_tokens
        self._disable_response_format = disable_response_format

    def create(self, *args: Any, **kwargs: Any) -> Any:
        expects_json = False
        if self._disable_response_format:
            expects_json = kwargs.pop("response_format", None) is not None
        if self._reasoning_effort:
            kwargs.setdefault("reasoning_effort", self._reasoning_effort)
        if self._max_completion_tokens:
            kwargs.setdefault(
                "max_completion_tokens",
                self._max_completion_tokens,
            )
        response = self._delegate.create(*args, **kwargs)
        record_response_usage(
            response,
            component="mem0_local_llm",
            provider="vllm",
            model=self._model,
        )
        if expects_json:
            normalize_json_chat_response(response)
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


def normalize_json_chat_response(response: Any) -> None:
    """Canonicalize a local JSON response or fail so Mem0's retry can run."""

    choices = getattr(response, "choices", None) or []
    if not choices:
        raise ValueError("Local Mem0 response has no choices")
    choice = choices[0]
    message = getattr(choice, "message", None)
    content = str(getattr(message, "content", "") or "").strip()
    decoder = json.JSONDecoder(strict=False)
    parsed: Any = None
    try:
        parsed = json.loads(content, strict=False)
    except json.JSONDecodeError:
        for index, character in enumerate(content):
            if character != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(content[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                parsed = candidate
                break
    if not isinstance(parsed, dict):
        finish_reason = getattr(choice, "finish_reason", None)
        raise ValueError(
            "Local Mem0 response was not valid JSON; "
            f"finish_reason={finish_reason!r}; content_preview={content[:300]!r}"
        )
    message.content = json.dumps(parsed, ensure_ascii=False)


class _TrackedChat:
    def __init__(
        self,
        delegate: Any,
        *,
        model: str,
        reasoning_effort: str | None,
        max_completion_tokens: int | None,
        disable_response_format: bool,
    ):
        self._delegate = delegate
        self.completions = _TrackedCompletions(
            delegate.completions,
            model=model,
            reasoning_effort=reasoning_effort,
            max_completion_tokens=max_completion_tokens,
            disable_response_format=disable_response_format,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class TrackedOpenAIClient:
    """Delegate to Mem0's vLLM client while recording raw provider usage."""

    def __init__(
        self,
        delegate: Any,
        *,
        model: str,
        reasoning_effort: str | None,
        max_completion_tokens: int | None,
        disable_response_format: bool = False,
    ):
        self._delegate = delegate
        self.chat = _TrackedChat(
            delegate.chat,
            model=model,
            reasoning_effort=reasoning_effort,
            max_completion_tokens=max_completion_tokens,
            disable_response_format=disable_response_format,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class LockedVectorStore:
    """Serialize access to qdrant-client's non-thread-safe local backend."""

    def __init__(self, delegate: Any, lock: threading.RLock):
        self._delegate = delegate
        self._lock = lock

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._delegate, name)
        if not callable(value):
            return value

        def locked_call(*args: Any, **kwargs: Any) -> Any:
            with self._lock:
                return value(*args, **kwargs)

        return locked_call


def lock_local_vector_stores(memory: Any) -> bool:
    """Lock only local Qdrant calls; LLM generation remains concurrent."""

    vector_store = memory.vector_store
    if not bool(getattr(vector_store, "is_local", False)):
        return False

    lock = threading.RLock()
    memory.vector_store = LockedVectorStore(vector_store, lock)
    entity_store = memory.entity_store
    # The lazy entity store shares the embedded client's handle and reports
    # is_local=False because it was constructed from an existing client.
    memory._entity_store = LockedVectorStore(entity_store, lock)
    return True


def create_memory(
    config: dict[str, Any],
    *,
    repo_path: Path = DEFAULT_UPSTREAM_PATH,
    reasoning_effort: str | None = "low",
    disable_response_format: bool = False,
):
    Memory = load_official_memory(repo_path)
    memory = Memory.from_config(config)
    model = str(config["llm"]["config"]["model"])
    max_completion_tokens = int(config["llm"]["config"]["max_tokens"])

    # Official VllmConfig currently inherits the reasoning fields but does not
    # expose them in its constructor. Setting them on the official config object
    # keeps the upstream generation path and makes _get_supported_params select
    # reasoning-model-safe parameters.
    memory.llm.config.is_reasoning_model = True
    memory.llm.config.reasoning_effort = reasoning_effort
    memory.llm.client = TrackedOpenAIClient(
        memory.llm.client,
        model=model,
        reasoning_effort=reasoning_effort,
        max_completion_tokens=max_completion_tokens,
        disable_response_format=disable_response_format,
    )
    memory.local_vector_store_locking = lock_local_vector_stores(memory)
    return memory


def check_vllm_endpoint(
    base_url: str,
    *,
    expected_model: str,
    timeout_seconds: float = 10.0,
) -> list[str]:
    request = Request(
        f"{base_url.rstrip('/')}/models",
        headers={"Authorization": "Bearer EMPTY"},
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    models = [
        str(item.get("id"))
        for item in payload.get("data", [])
        if isinstance(item, dict) and item.get("id")
    ]
    if expected_model not in models:
        raise RuntimeError(
            f"vLLM model mismatch: expected={expected_model} available={models}"
        )
    return models
