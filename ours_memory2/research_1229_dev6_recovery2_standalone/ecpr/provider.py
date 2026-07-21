"""Single-call, loopback-only OpenAI-compatible provider."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from .ledger import ProviderJournal
from .request_contract import CallSpec


class ProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class Completion:
    content: str
    usage: dict[str, int]
    call_id: str | None = None
    request_sha256: str | None = None


class LoopbackChatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: float = 120.0,
        *,
        journal: ProviderJournal | None = None,
    ):
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("provider URL must use http or https")
        if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("only a loopback inference server is permitted")
        self.endpoint = base_url.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.journal = journal

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        seed: int,
        max_tokens: int,
        temperature: float,
        json_object: bool = False,
        phase: str | None = None,
        call_key: str | None = None,
        schema_sha256: str | None = None,
    ) -> Completion:
        if schema_sha256 is None:
            raise ValueError("provider call is missing schema digest")
        call_spec = CallSpec.from_messages(
            endpoint=self.endpoint,
            model=self.model,
            messages=messages,
            seed=seed,
            temperature=temperature,
            max_tokens=max_tokens,
            json_object=json_object,
            schema_sha256=schema_sha256,
            timeout_seconds=self.timeout_seconds,
        )
        intent: dict[str, Any] | None = None
        if self.journal is not None:
            if phase is None or call_key is None:
                raise ValueError("journaled provider call is missing audit context")
            intent = self.journal.begin_call(
                phase=phase,
                call_key=call_key,
                call_spec=call_spec,
            )
        request = urllib.request.Request(
            self.endpoint,
            data=call_spec.wire_bytes(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=call_spec.timeout_seconds
            ) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"] or ""
            raw_usage = body.get("usage", {})
            usage = {
                key: int(raw_usage.get(key, 0) or 0)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            }
        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            KeyError,
            IndexError,
            TypeError,
        ) as exc:
            error_text = f"{type(exc).__name__}:{exc}"
            if intent is not None:
                self.journal.finish_call(
                    intent,
                    status="error",
                    usage={
                        key: 0
                        for key in (
                            "prompt_tokens",
                            "completion_tokens",
                            "total_tokens",
                        )
                    },
                    error=error_text,
                )
            raise ProviderError(error_text) from exc
        if intent is not None:
            self.journal.finish_call(
                intent,
                status="ok",
                usage=usage,
                response=str(content),
            )
        return Completion(
            content=str(content),
            usage=usage,
            call_id=str(intent["call_id"]) if intent is not None else None,
            request_sha256=call_spec.audit_request_sha256(),
        )
