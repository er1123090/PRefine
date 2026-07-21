"""Canonical provider request contract shared by wire transport and audit replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .io import canonical_json, sha256_bytes, sha256_text


_HEX = frozenset("0123456789abcdef")


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class CallSpec:
    """One immutable request source for HTTP bytes and provider-journal audit."""

    endpoint: str
    model: str
    messages: tuple[tuple[str, str], ...]
    seed: int
    temperature: float
    max_tokens: int
    json_object: bool
    schema_sha256: str
    timeout_seconds: float
    n: int = 1
    stream: bool = False

    @classmethod
    def from_messages(
        cls,
        *,
        endpoint: str,
        model: str,
        messages: Sequence[dict[str, str]],
        seed: int,
        temperature: float,
        max_tokens: int,
        json_object: bool,
        schema_sha256: str,
        timeout_seconds: float,
    ) -> "CallSpec":
        normalized: list[tuple[str, str]] = []
        for message in messages:
            if (
                not isinstance(message, dict)
                or set(message) != {"role", "content"}
                or not isinstance(message["role"], str)
                or not isinstance(message["content"], str)
            ):
                raise ValueError("provider messages must contain only string role/content")
            normalized.append((message["role"], message["content"]))
        value = cls(
            endpoint=str(endpoint),
            model=str(model),
            messages=tuple(normalized),
            seed=seed,
            temperature=float(temperature),
            max_tokens=max_tokens,
            json_object=json_object,
            schema_sha256=schema_sha256,
            timeout_seconds=float(timeout_seconds),
        )
        value.validate()
        return value

    def validate(self) -> None:
        if not self.endpoint or not self.model:
            raise ValueError("provider endpoint and model must be nonempty")
        if not self.messages:
            raise ValueError("provider messages must be nonempty")
        if type(self.seed) is not int:
            raise ValueError("provider seed must be an integer")
        if isinstance(self.temperature, bool) or not isinstance(
            self.temperature, (int, float)
        ):
            raise ValueError("provider temperature must be numeric")
        if type(self.max_tokens) is not int or self.max_tokens <= 0:
            raise ValueError("provider max_tokens must be positive")
        if type(self.json_object) is not bool:
            raise ValueError("provider json_object must be boolean")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or self.timeout_seconds <= 0
        ):
            raise ValueError("provider timeout must be positive")
        if self.n != 1 or self.stream is not False:
            raise ValueError("provider calls must be single and nonstreaming")
        _sha256(self.schema_sha256, "provider schema digest")

    def message_list(self) -> list[dict[str, str]]:
        return [
            {"role": role, "content": content}
            for role, content in self.messages
        ]

    def wire_payload(self) -> dict[str, Any]:
        self.validate()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self.message_list(),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "n": self.n,
            "stream": self.stream,
        }
        if self.json_object:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def wire_bytes(self) -> bytes:
        return canonical_json(self.wire_payload()).encode("utf-8")

    def audit_request(self) -> dict[str, Any]:
        self.validate()
        messages = self.message_list()
        wire = self.wire_bytes()
        return {
            "endpoint": self.endpoint,
            "model": self.model,
            "seed": self.seed,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "n": self.n,
            "stream": self.stream,
            "json_object": self.json_object,
            "schema_sha256": self.schema_sha256,
            "messages_sha256": sha256_text(canonical_json(messages)),
            "timeout_seconds": self.timeout_seconds,
            "wire_payload_sha256": sha256_bytes(wire),
        }

    def audit_request_sha256(self) -> str:
        return sha256_text(canonical_json(self.audit_request()))

