"""Exhaustive path-class to writer-role/task policy."""
from __future__ import annotations

from dataclasses import dataclass

from .canonical import (
    canonical_json,
    fail,
    require_exact_keys,
    sha256_bytes,
    validate_relative_path,
)


POLICY_SCHEMA = "experiments7-strict-writer-policy/v6"


@dataclass(frozen=True)
class WriterIdentity:
    role: str
    task_id: str
    writer_id: str

    def __post_init__(self) -> None:
        for field, value in (
            ("role", self.role), ("task_id", self.task_id), ("writer_id", self.writer_id)
        ):
            if not value or "\x00" in value or len(value) > 256:
                fail("INVALID_WRITER", f"{field} is not a bounded nonempty value")

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "task_id": self.task_id, "writer_id": self.writer_id}

    @classmethod
    def from_dict(cls, value: object) -> "WriterIdentity":
        row = require_exact_keys(value, {"role", "task_id", "writer_id"}, "writer")
        if not all(isinstance(row[key], str) for key in row):
            fail("INVALID_WRITER", "writer fields must be strings")
        return cls(str(row["role"]), str(row["task_id"]), str(row["writer_id"]))


@dataclass(frozen=True)
class WriterRule:
    rule_id: str
    match: str
    path: str
    role: str
    task_prefix: str
    writer_id: str

    def matches_path(self, relative_path: str) -> bool:
        if self.match == "exact":
            return relative_path == self.path
        return relative_path == self.path or relative_path.startswith(f"{self.path}/")

    def authorizes(self, relative_path: str, writer: WriterIdentity) -> bool:
        return (
            self.matches_path(relative_path)
            and writer.role == self.role
            and writer.task_id.startswith(self.task_prefix)
            and writer.writer_id == self.writer_id
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "rule_id": self.rule_id,
            "match": self.match,
            "path": self.path,
            "role": self.role,
            "task_prefix": self.task_prefix,
            "writer_id": self.writer_id,
        }


class WriterPolicy:
    def __init__(self, rules: tuple[WriterRule, ...]) -> None:
        if not rules:
            fail("WRITER_POLICY_INVALID", "writer policy must contain rules")
        if len({rule.rule_id for rule in rules}) != len(rules):
            fail("WRITER_POLICY_INVALID", "writer rule IDs must be unique")
        for rule in rules:
            validate_relative_path(rule.path)
            if rule.match not in {"exact", "prefix"}:
                fail("WRITER_POLICY_INVALID", "writer rule match must be exact or prefix")
            if (
                not rule.rule_id
                or not rule.role
                or not rule.task_prefix
                or not rule.writer_id
            ):
                fail("WRITER_POLICY_INVALID", "writer rule fields must be nonempty")
        self.rules = rules

    def matching_rules(self, relative_path: str) -> tuple[WriterRule, ...]:
        relative_path = validate_relative_path(relative_path)
        return tuple(rule for rule in self.rules if rule.matches_path(relative_path))

    def authorize(self, relative_path: str, writer: WriterIdentity) -> WriterRule:
        matches = self.matching_rules(relative_path)
        if len(matches) != 1:
            fail(
                "WRITER_POLICY_COVERAGE",
                f"path must match exactly one writer rule, found {len(matches)}",
            )
        rule = matches[0]
        if not rule.authorizes(relative_path, writer):
            fail("WRITER_MISMATCH", "writer role/task is not authorized for path")
        return rule

    def to_dict(self) -> dict[str, object]:
        rows = [rule.to_dict() for rule in self.rules]
        return {
            "schema": POLICY_SCHEMA,
            "rule_count": len(rows),
            "rules_sha256": sha256_bytes(canonical_json(rows)),
            "rules": rows,
        }

    @classmethod
    def from_dict(cls, value: object) -> "WriterPolicy":
        row = require_exact_keys(
            value, {"schema", "rule_count", "rules_sha256", "rules"}, POLICY_SCHEMA
        )
        if row["schema"] != POLICY_SCHEMA or not isinstance(row["rules"], list):
            fail("WRITER_POLICY_INVALID", "writer policy schema or rules are invalid")
        rules: list[WriterRule] = []
        for raw in row["rules"]:
            item = require_exact_keys(
                raw,
                {"rule_id", "match", "path", "role", "task_prefix", "writer_id"},
                "writer-rule",
            )
            if not all(isinstance(item[key], str) for key in item):
                fail("WRITER_POLICY_INVALID", "writer rule fields must be strings")
            rules.append(
                WriterRule(
                    str(item["rule_id"]), str(item["match"]), str(item["path"]),
                    str(item["role"]), str(item["task_prefix"]), str(item["writer_id"]),
                )
            )
        if row["rule_count"] != len(rules):
            fail("WRITER_POLICY_INVALID", "writer rule count differs")
        if row["rules_sha256"] != sha256_bytes(canonical_json([item.to_dict() for item in rules])):
            fail("WRITER_POLICY_INVALID", "writer rule digest differs")
        return cls(tuple(rules))


def default_writer_policy() -> WriterPolicy:
    specs = (
        (
            "reservation", "exact", "reservation.json", "run_reservation",
            "reservation:", "run_reservation-writer",
        ),
        (
            "reservation-acceptance", "exact", "reservation-acceptance.json",
            "reservation_acceptance", "acceptance:", "reservation_acceptance-writer",
        ),
        (
            "owner-binding", "exact", "owner-binding.json", "g0", "g0:", "g0-writer",
        ),
        ("frozen", "prefix", "frozen", "g0", "g0:", "g0-writer"),
        ("manifests", "prefix", "manifests", "g0", "g0:", "g0-writer"),
        (
            "registry", "prefix", "registry", "registry", "registry:",
            "registry-writer",
        ),
        (
            "inventory", "prefix", "inventory", "inventory", "inventory:",
            "inventory-writer",
        ),
        (
            "adapters", "prefix", "adapters", "adapter_snapshot", "snapshot:",
            "adapter_snapshot-writer",
        ),
        (
            "snapshots", "prefix", "snapshots", "adapter_snapshot", "snapshot:",
            "adapter_snapshot-writer",
        ),
        (
            "runtime", "prefix", "runtime", "adapter_snapshot", "snapshot:",
            "adapter_snapshot-writer",
        ),
        (
            "run-configs", "prefix", "run-configs", "adapter_snapshot", "snapshot:",
            "adapter_snapshot-writer",
        ),
        (
            "golden", "prefix", "golden", "adapter_snapshot", "snapshot:",
            "adapter_snapshot-writer",
        ),
        (
            "provenance", "prefix", "provenance", "provenance", "provenance:",
            "provenance-writer",
        ),
        (
            "admission", "prefix", "admission", "admission", "admission:",
            "admission-writer",
        ),
        ("raw", "prefix", "raw", "copy", "copy:", "copy-writer"),
        ("copies", "exact", "copies.jsonl", "copy", "copy:", "copy-writer"),
        (
            "verification-root", "exact", "verification", "verification",
            "verification:verifier:", "verification-verifier-writer",
        ),
        (
            "verification-verifier", "exact", "verification/verifier.json",
            "verification", "verification:verifier:", "verification-verifier-writer",
        ),
        (
            "verification-code-reviewer", "exact", "verification/code-reviewer.json",
            "verification", "verification:code-reviewer:",
            "verification-code-reviewer-writer",
        ),
        (
            "verification-adversarial-qa", "exact",
            "verification/adversarial-qa.json", "verification",
            "verification:adversarial-qa:", "verification-adversarial-qa-writer",
        ),
        (
            "checkpoints", "prefix", "checkpoints", "checkpoint_controller",
            "checkpoint:", "checkpoint_controller-writer",
        ),
        (
            "terminal", "prefix", "terminal", "checkpoint_controller", "checkpoint:",
            "checkpoint_controller-writer",
        ),
    )
    return WriterPolicy(tuple(WriterRule(*spec) for spec in specs))
