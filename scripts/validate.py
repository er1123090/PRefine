#!/usr/bin/env python3
"""Read-only, API-free cutover validator for the canonical experiments7 surface."""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exp7.datasets.build_instances import build_all_scenarios, group_multiturn_templates
from exp7.datasets.pipeline import (
    _bundle_sha256,
    _input_manifest,
    _load_json,
    _resolve,
    _verified_input,
)
from exp7.datasets.validation import build_validation_report
from exp7.provenance.cutover_receipt import (
    cutover_receipt_report,
    post_cutover_local_state_report,
)
from experiments7_layout.catalog import validate_layout, write_catalogs


SCENARIO_COUNTS = {
    "singleturn.easy": 554,
    "singleturn.medium": 293,
    "singleturn.hard": 472,
    "multiturn.easy": 554,
    "multiturn.medium": 293,
    "multiturn.hard": 472,
}
SCHEMA_WARNING_COUNTS = {
    "singleturn.easy": 39,
    "singleturn.medium": 12,
    "singleturn.hard": 27,
    "multiturn.easy": 63,
    "multiturn.medium": 129,
    "multiturn.hard": 193,
}
SCHEMA_WARNING_SLOTS = {
    "singleturn": ["number_of_seats"],
    "multiturn": ["city", "departure_date", "destination", "pickup_time"],
}
IGNORED_GROUPS = {"eco": 130, "prefers_star": 468}
CONDITIONS = tuple(SCENARIO_COUNTS)
ACTIVE_PYTHON_SCRIPTS = (
    "scripts/prepare.py",
    "scripts/run.py",
    "scripts/run_suite.py",
)
CANONICAL_SHELL = "scripts/run_vanilla_llm_mix600_all_difficulties.sh"
PHASE_MODE = "phase-readiness"
FINAL_MODE = "final-completion"
VALIDATION_MODES = (PHASE_MODE, FINAL_MODE)
REPORT_SCHEMA = "experiments7-cutover-validation/v5"
KNOWN_GAPS = [
    {
        "id": "historical_paper_cleanup_planned_only",
        "detail": (
            "Historical top-level trees remain under a planning-only archive map, and raw "
            "externalization remains forbidden until verified backup/restore evidence exists."
        ),
    },
    {
        "id": "cutover_receipt_missing_or_invalid",
        "detail": (
            "Final completion requires a durable receipt that cryptographically binds the "
            "canonical archive map, complete source/destination inventories, verified restore "
            "evidence, provider-authenticated off-host versions, and independent approval."
        ),
    },
    {
        "id": "docs_navigation_gap",
        "detail": "The established docs/README navigation does not yet expose the new quickstart contracts.",
    },
]


class CutoverValidationError(RuntimeError):
    """Raised when a required cutover invariant is false."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CutoverValidationError(message)


def _regular_file(path: Path, *, label: str) -> None:
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a real file: {path}")


def _archive_cleanup_pending(repo_root: Path) -> bool:
    archive_map = repo_root / "archive/archive-map.json"
    if not archive_map.is_file() or archive_map.is_symlink():
        return True
    try:
        document = json.loads(archive_map.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return True
    if not isinstance(document, dict) or document.get("planned_only") is not False:
        return True
    records = document.get("records")
    if not isinstance(records, list):
        return True
    by_path = {
        row.get("path"): row
        for row in records
        if isinstance(row, dict) and isinstance(row.get("path"), str)
    }
    for relative in ("data/sources", "methods", "experiments"):
        if (repo_root / relative).exists():
            return True
    raw = repo_root / "paper_outputs/raw"
    raw_record = by_path.get("paper_outputs/raw")
    if raw.exists() and (
        not isinstance(raw_record, dict)
        or raw_record.get("externalization_allowed") is not True
    ):
        return True
    return False


def _docs_navigation_missing(repo_root: Path) -> bool:
    root_readme = repo_root / "README.md"
    docs_readme = repo_root / "docs/README.md"
    quickstart = repo_root / "docs/quickstart.md"
    for path in (root_readme, docs_readme, quickstart):
        if not path.is_file() or path.is_symlink():
            return True
    try:
        root_text = root_readme.read_text(encoding="utf-8")
        docs_text = docs_readme.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return True
    return (
        "docs/quickstart.md" not in root_text
        and "quickstart.md" not in docs_text
    )


def completion_blockers(
    repo_root: Path,
    *,
    receipt_report: Mapping[str, Any] | None = None,
    local_state_report: Mapping[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Return only completion gaps supported by current repository evidence."""

    templates = {gap["id"]: gap for gap in KNOWN_GAPS}
    blockers = []
    local_state = local_state_report or post_cutover_local_state_report(repo_root)
    if _archive_cleanup_pending(repo_root) or local_state.get("verified") is not True:
        blockers.append(dict(templates["historical_paper_cleanup_planned_only"]))
    receipt = receipt_report or cutover_receipt_report(repo_root)
    if receipt.get("verified") is not True:
        blockers.append(dict(templates["cutover_receipt_missing_or_invalid"]))
    if _docs_navigation_missing(repo_root):
        blockers.append(dict(templates["docs_navigation_gap"]))
    return blockers


def _completion_semantics(
    *,
    mode: str,
    phase_ready: bool,
    final_complete: bool,
) -> dict[str, Any]:
    """Describe this project-cutover check without claiming a provider run."""

    return {
        "actual_experiment_completion_claimed": False,
        "completion_domain": (
            "project_structure"
            if mode == PHASE_MODE
            else "project_physical_cutover"
        ),
        "experiment_completion": "not_evaluated",
        "project_cutover_complete": final_complete,
        "structural_readiness": "ready" if phase_ready else "failed",
    }


def _verified_dataset_inputs(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset_path = repo_root / "configs/datasets/mix600-v1.json"
    _regular_file(dataset_path, label="dataset config")
    dataset = _load_json(dataset_path, label="dataset config")
    config_paths = {
        "dataset": dataset_path,
        "queries": _resolve(repo_root, dataset["queries_config"]),
        "preferences": _resolve(repo_root, dataset["preferences_config"]),
        "schemas": _resolve(repo_root, dataset["schemas_config"]),
    }
    for label, path in config_paths.items():
        _regular_file(path, label=f"{label} config")
    configs = {
        label: _load_json(path, label=f"{label} config")
        for label, path in config_paths.items()
    }
    _require(
        {value.get("dataset_id") for value in configs.values()} == {"mix600-v1"},
        "canonical configs do not agree on dataset_id=mix600-v1",
    )
    source_spec = dict(dataset["source"])
    source_spec["path"] = source_spec.pop("default_path")
    input_specs = {
        "mix600": source_spec,
        "query_singleturn": configs["queries"]["inputs"]["singleturn"],
        "query_multiturn": configs["queries"]["inputs"]["multiturn"],
        "pref_list": configs["preferences"]["inputs"]["pref_list"],
        "pref_group": configs["preferences"]["inputs"]["pref_group"],
        "schema_singleturn": configs["schemas"]["inputs"]["singleturn"],
        "schema_multiturn": configs["schemas"]["inputs"]["multiturn"],
    }
    paths = {
        label: _verified_input(repo_root, spec, label=label)
        for label, spec in input_specs.items()
    }

    frozen = _load_json(
        repo_root / "tests/fixtures/dataset/mix600_legacy_contract.json",
        label="frozen mix600 contract",
    )["sources"]
    for label, path in paths.items():
        actual = _input_manifest(repo_root, path)
        expected = frozen[label]
        _require(actual["bytes"] == expected["bytes"], f"{label} frozen byte count drift")
        _require(actual["sha256"] == expected["sha256"], f"{label} frozen SHA-256 drift")

    return {
        "config_bundle_sha256": _bundle_sha256(repo_root, tuple(config_paths.values())),
        "configs": {
            label: _input_manifest(repo_root, path)
            for label, path in sorted(config_paths.items())
        },
        "inputs": {
            label: _input_manifest(repo_root, path)
            for label, path in sorted(paths.items())
        },
        "status": "pass",
    }, {"configs": configs, "paths": paths}


def validate_warning_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    """Check the frozen warning boundary without implying full dialogue/GT consistency."""

    schema = report.get("schema_warnings", {})
    preferences = report.get("preference_groups", {})
    overlap = report.get("multiturn_base_preference_conflicts", {})
    _require(schema.get("total") == 463, "schema warning total must be 463")
    _require(
        schema.get("scenarios") == SCHEMA_WARNING_COUNTS,
        "schema warning scenario split drift",
    )
    _require(
        schema.get("missing_slots") == SCHEMA_WARNING_SLOTS,
        "schema warning slot set drift",
    )
    _require(
        preferences.get("source_examples", {}).get("unsupported_only") == 477,
        "unsupported-only source count must be 477",
    )
    _require(
        preferences.get("ignored_groups") == IGNORED_GROUPS,
        "ignored preference annotation counts drift",
    )
    _require(overlap.get("template_count") == 25, "multiturn template count must be 25")
    _require(
        overlap.get("conflict_count") == 0,
        "detected multiturn template base/preference slot-name overlap must be zero",
    )
    return {
        "ignored_groups": dict(IGNORED_GROUPS),
        "multiturn_template_slot_name_overlap": {
            "conflicts": 0,
            "templates": 25,
        },
        "schema_missing_slot_instances": {
            "scenarios": dict(SCHEMA_WARNING_COUNTS),
            "slots": dict(SCHEMA_WARNING_SLOTS),
            "total": 463,
        },
        "status": "pass_with_known_warnings",
        "unsupported_only_source_examples": 477,
    }


def check_dataset_contract(repo_root: Path) -> dict[str, Any]:
    hashes, loaded = _verified_dataset_inputs(repo_root)
    configs = loaded["configs"]
    paths = loaded["paths"]
    values = {label: _load_json(path, label=label) for label, path in paths.items()}
    rows = values["mix600"]
    _require(isinstance(rows, list), "mix600 must be a JSON list")
    scenarios = build_all_scenarios(
        rows,
        dataset_id="mix600-v1",
        turns=configs["dataset"]["turns"],
        difficulties=configs["dataset"]["difficulties"],
        query_singleturn=values["query_singleturn"],
        query_multiturn_raw=values["query_multiturn"],
        pref_list=values["pref_list"],
        pref_groups=values["pref_group"],
    )
    counts = {name: len(records) for name, records in scenarios.items()}
    _require(counts == SCENARIO_COUNTS, f"scenario count drift: {counts!r}")
    _require(sum(counts.values()) == 2638, "total instance count must be 2638")
    report = build_validation_report(
        rows=rows,
        scenarios=scenarios,
        pref_groups=values["pref_group"],
        query_multiturn=values["query_multiturn"],
        schemas={
            "singleturn": values["schema_singleturn"],
            "multiturn": values["schema_multiturn"],
        },
        expected=configs["dataset"]["expected"],
    )
    warnings = validate_warning_contract(report)
    _require(
        len(group_multiturn_templates(values["query_multiturn"])) > 0,
        "multiturn templates are not addressable by domain",
    )
    return {
        "counts": counts,
        "hashes": hashes,
        "parity": report["contract_parity"],
        "status": "pass_with_known_warnings",
        "total_instances": 2638,
        "warnings": warnings,
    }


_FORBIDDEN_IMPORT_ROOTS = {
    "archive", "environments", "experiments4", "experiments5",
    "experiments6", "methods", "paper_outputs",
}
_FORBIDDEN_PATH_PATTERN = re.compile(
    r"(?:^|[/\\])(?:experiments[456]|data[/\\]sources|methods|environments|archive|paper_outputs)(?:[/\\]|$)"
)
_HISTORICAL_LITERAL_ALLOWLIST = {
    PurePosixPath("src/exp7/provenance/archive_map.py"): frozenset({
        "archive/archive-map.json",
        "archive/archive-map.md",
        "data/sources",
        "environments",
        "methods",
        "paper_outputs/admission",
        "paper_outputs/admission/precopy/seal.json",
        "paper_outputs/final",
        "paper_outputs/final/mapping.jsonl",
        "paper_outputs/final/raw-manifest.jsonl",
        "paper_outputs/final/seal.json",
        "paper_outputs/final/unresolved.jsonl",
        "paper_outputs/inventory",
        "paper_outputs/inventory/task-evidence.json",
        "paper_outputs/provenance",
        "paper_outputs/provenance/README.md",
        "paper_outputs/provenance/edges.jsonl",
        "paper_outputs/provenance/nodes.jsonl",
        "paper_outputs/raw",
        "paper_outputs/raw/README.md",
        "paper_outputs/raw/schema.json",
        "paper_outputs/strict-runs",
        "paper_outputs/strict-runs/exp7-strict-v6-20260718T032509Z-11942ff2120f70e5fd0d18a08d70569d/checkpoints/cp0.json",
    }),
    PurePosixPath("src/exp7/provenance/cutover_receipt.py"): frozenset({
        "archive/archive-map.json",
        "archive/cutover-destination-inventory.jsonl",
        "archive/cutover-receipt.json",
        "archive/cutover-restore-report.json",
        "archive/cutover-source-inventory.jsonl",
        "archive/provider-offhost-proof.json",
        "data/sources",
        "methods",
        "paper_outputs/raw",
    }),
}


def _string_literals(tree: ast.AST) -> Iterable[tuple[int, str]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield getattr(node, "lineno", 0), node.value


def _historical_module(module: str) -> bool:
    return (
        module.split(".", 1)[0] in _FORBIDDEN_IMPORT_ROOTS
        or module == "data.sources"
        or module.startswith("data.sources.")
    )


def check_python_boundaries(
    repo_root: Path,
    python_paths: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Reject active-code links, historical imports, and historical path literals."""

    repo_root = repo_root.resolve(strict=True)
    paths = list(python_paths or sorted((repo_root / "src/exp7").rglob("*.py")))
    if python_paths is None:
        paths.extend(repo_root / relative for relative in ACTIVE_PYTHON_SCRIPTS)
    errors: list[str] = []
    literal_exceptions: list[str] = []
    for path in paths:
        if path.is_symlink() or not path.is_file():
            errors.append(f"not a real Python file: {path}")
            continue
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            compile(source, str(path), "exec")
        except (OSError, SyntaxError, UnicodeError) as exc:
            errors.append(f"cannot compile {path}: {exc}")
            continue
        try:
            relative_path = PurePosixPath(path.resolve(strict=True).relative_to(repo_root).as_posix())
        except (OSError, ValueError):
            relative_path = None
        allowed_historical_literals = _HISTORICAL_LITERAL_ALLOWLIST.get(
            relative_path,
            frozenset(),
        )
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            for module in modules:
                if _historical_module(module):
                    errors.append(f"forbidden historical import {module!r} in {path}:{node.lineno}")
            if isinstance(node, ast.Call) and node.args:
                function = node.func
                dynamic_import = (
                    isinstance(function, ast.Name) and function.id == "__import__"
                ) or (
                    isinstance(function, ast.Attribute) and function.attr == "import_module"
                )
                argument = node.args[0]
                if (
                    dynamic_import
                    and isinstance(argument, ast.Constant)
                    and isinstance(argument.value, str)
                    and _historical_module(argument.value)
                ):
                    errors.append(
                        f"forbidden dynamic historical import {argument.value!r} "
                        f"in {path}:{node.lineno}"
                    )
        for line, literal in _string_literals(tree):
            normalized = literal.replace("\\", "/")
            if _FORBIDDEN_PATH_PATTERN.search(normalized):
                if normalized in allowed_historical_literals:
                    literal_exceptions.append(f"{relative_path}:{line}")
                    continue
                errors.append(f"forbidden historical path literal {literal!r} in {path}:{line}")
    _require(not errors, "; ".join(errors))
    return {
        "files": len(paths),
        "historical_literal_exceptions": sorted(set(literal_exceptions)),
        "status": "pass",
    }


def check_config_boundaries(repo_root: Path) -> dict[str, Any]:
    roots = ("datasets", "queries", "preferences", "schemas", "experiments", "models")
    paths = sorted(
        path
        for root in roots
        for path in (repo_root / "configs" / root).rglob("*")
        if path.is_file() or path.is_symlink()
    )
    allowed = repo_root / "configs/datasets/mix600-v1.json"
    external_references = []
    for path in paths:
        _regular_file(path, label="canonical config input")
        text = path.read_text(encoding="utf-8")
        matches = _FORBIDDEN_PATH_PATTERN.findall(text.replace("\\", "/"))
        if not matches:
            continue
        _require(path == allowed, f"historical path outside allowed dataset config: {path}")
        document = json.loads(text)
        source = document.get("source", {})
        _require(
            source.get("default_path") == "../experiments4/data/mix600.json"
            and isinstance(source.get("expected_bytes"), int)
            and re.fullmatch(r"[0-9a-f]{64}", str(source.get("expected_sha256", ""))) is not None,
            "external mix600 reference is not byte-size/SHA-256 bound",
        )
        external_references.append(path.relative_to(repo_root).as_posix())
    _require(
        external_references == ["configs/datasets/mix600-v1.json"],
        "the sole historical reference must be the hash-bound external mix600 source",
    )
    return {
        "allowed_external_reference": external_references[0],
        "files": len(paths),
        "status": "pass",
    }


def check_launcher(repo_root: Path) -> dict[str, Any]:
    shell = repo_root / CANONICAL_SHELL
    _regular_file(shell, label="canonical launcher")
    _require(
        _FORBIDDEN_PATH_PATTERN.search(shell.read_text(encoding="utf-8")) is None,
        "canonical launcher contains a historical path literal",
    )
    syntax = subprocess.run(
        ["bash", "-n", str(shell)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    _require(syntax.returncode == 0, f"bash -n failed: {syntax.stderr.strip()}")
    planned = subprocess.run(
        [
            "bash",
            str(shell),
            "--dry-run",
            "--prepared-root",
            "/tmp/exp7-cutover-validator-prepared",
            "--output-root",
            "/tmp/exp7-cutover-validator-run",
            "--model",
            "cutover-validator",
        ],
        cwd=repo_root,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    _require(planned.returncode == 0, f"launcher dry-run failed: {planned.stderr.strip()}")
    lines = [line for line in planned.stdout.splitlines() if line.strip()]
    labels = [line.split("]", 1)[0].lstrip("[") for line in lines]
    _require(labels == list(CONDITIONS), f"launcher order drift: {labels!r}")
    _require(len(lines) == 6, "launcher must emit exactly six dry-run commands")
    _require(sum("schema_easy.json" in line for line in lines[:3]) == 3, "singleturn schema drift")
    _require(sum("schema_all.json" in line for line in lines[3:]) == 3, "multiturn schema drift")
    _require(all("scripts/run.py" in line for line in lines), "launcher does not use canonical runner")
    return {"bash_syntax": "pass", "conditions": labels, "status": "pass"}


_MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")


def check_markdown_links(repo_root: Path) -> dict[str, Any]:
    paths = [repo_root / "README.md", *sorted((repo_root / "docs").rglob("*.md"))]
    for relative in (
        "configs/datasets/README.md",
        "configs/queries/README.md",
        "configs/preferences/README.md",
        "configs/schemas/README.md",
        "configs/experiments/README.md",
        "configs/models/README.md",
        "scripts/README.md",
        "artifacts/README.md",
        "archive/README.md",
        "tests/integration/README.md",
        "tests/methods/README.md",
        "tests/evaluation/README.md",
    ):
        candidate = repo_root / relative
        if candidate.is_file():
            paths.append(candidate)
    missing = []
    checked = 0
    for path in dict.fromkeys(paths):
        _regular_file(path, label="Markdown document")
        for raw in _MARKDOWN_LINK.findall(path.read_text(encoding="utf-8")):
            target = raw.strip().split(maxsplit=1)[0].strip("<>")
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            target = unquote(target.split("#", 1)[0])
            if not target:
                continue
            checked += 1
            candidate = Path(target) if Path(target).is_absolute() else path.parent / target
            if not candidate.exists():
                missing.append(f"{path.relative_to(repo_root)} -> {target}")
    _require(not missing, "missing local Markdown links: " + ", ".join(missing))
    return {"documents": len(dict.fromkeys(paths)), "links": checked, "status": "pass"}


def check_layout(repo_root: Path) -> dict[str, Any]:
    write_catalogs(repo_root, check=True)
    report = validate_layout(repo_root)
    _require(report.get("status") == "ok", "official layout validation did not pass")
    return {"build_indexes_check": "pass", "status": "pass", "validate": report}


def check_api_free_targeted_tests(repo_root: Path) -> dict[str, Any]:
    modules = (
        "tests.dataset.test_pipeline_path_safety",
        "tests.integration.test_run_suite_artifacts",
        "tests.integration.test_experiment_config_entrypoint",
        "tests.evaluation.test_manifest_evaluator",
        "tests.methods.test_adapter_contract",
        "tests.methods.test_method_registry",
        "tests.methods.test_vanilla_llm_runner",
        "tests.methods.test_rag_adapter",
        "tests.methods.test_mem0_adapter",
        "tests.methods.test_langmem_adapter",
        "tests.methods.test_preference_memory_adapter",
    )
    command = [sys.executable, "-B", "-m", "unittest", "-v", *modules]
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        command,
        cwd=repo_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    _require(result.returncode == 0, "API-free targeted tests failed: " + output[-2000:])
    match = re.search(r"Ran (\d+) tests?", output)
    _require(match is not None, "targeted unittest count was not reported")
    return {"modules": list(modules), "status": "pass", "tests": int(match.group(1))}


def run_checks(
    repo_root: Path,
    checks: Sequence[tuple[str, Callable[[Path], dict[str, Any]]]] | None = None,
    *,
    mode: str = PHASE_MODE,
    trust_bundle: Path | None = None,
    expected_trust_sha256: str | None = None,
) -> dict[str, Any]:
    _require(mode in VALIDATION_MODES, f"unsupported validation mode: {mode!r}")
    if checks is None:
        checks = (
            ("dataset", check_dataset_contract),
            ("active_code", check_python_boundaries),
            ("configs", check_config_boundaries),
            ("launcher", check_launcher),
            ("markdown_links", check_markdown_links),
            ("layout", check_layout),
            ("api_free_targeted_tests", check_api_free_targeted_tests),
        )
    results = {}
    failures = []
    for name, checker in checks:
        try:
            results[name] = checker(repo_root)
        except Exception as exc:
            failures.append({"check": name, "error": str(exc), "type": type(exc).__name__})
    local_state = post_cutover_local_state_report(repo_root)
    receipt = cutover_receipt_report(
        repo_root,
        trust_bundle=trust_bundle,
        expected_trust_sha256=expected_trust_sha256,
    )
    blockers = completion_blockers(
        repo_root,
        receipt_report=receipt,
        local_state_report=local_state,
    )
    phase_ready = not failures
    receipt_verified = receipt.get("verified") is True
    local_state_verified = local_state.get("verified") is True
    final_complete = (
        mode == FINAL_MODE
        and phase_ready
        and not blockers
        and receipt_verified
        and local_state_verified
    )
    if failures:
        status = "validation_failed"
    elif mode == PHASE_MODE:
        status = "phase_ready"
    elif blockers or not receipt_verified or not local_state_verified:
        status = "final_blocked"
    else:
        status = "final_complete"
    report = {
        "checks": results,
        "completion_blockers": blockers,
        "cutover_receipt": receipt,
        "failures": failures,
        "final_complete": final_complete,
        "known_gaps": blockers,
        "mode": mode,
        "post_cutover_local_state": local_state,
        "phase_ready": phase_ready,
        "schema": REPORT_SCHEMA,
        "status": status,
    }
    report.update(
        _completion_semantics(
            mode=mode,
            phase_ready=phase_ready,
            final_complete=final_complete,
        )
    )
    return report


def report_exit_code(report: Mapping[str, Any]) -> int:
    return 0 if report.get("status") in {"phase_ready", "final_complete"} else 2


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    value.add_argument(
        "--mode",
        choices=VALIDATION_MODES,
        default=PHASE_MODE,
        help=(
            "phase-readiness allows declared completion blockers; final-completion "
            "fails closed until every blocker is resolved"
        ),
    )
    value.add_argument(
        "--cutover-trust-bundle",
        type=Path,
        help="external trust bundle outside the repository (final completion only)",
    )
    value.add_argument(
        "--cutover-trust-sha256",
        dest="expected_trust_sha256",
        help="expected SHA-256 of the external cutover trust bundle",
    )
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        repo_root = args.root.resolve(strict=True)
        report = run_checks(
            repo_root,
            mode=args.mode,
            trust_bundle=args.cutover_trust_bundle,
            expected_trust_sha256=args.expected_trust_sha256,
        )
    except Exception as exc:
        report = {
            "checks": {},
            "completion_blockers": [],
            "cutover_receipt": {
                "error": "not evaluated because validator startup failed",
                "externalization_authorized": False,
                "independent_approval_verified": False,
                "no_mutations": True,
                "physical_cutover_complete": False,
                "provider_authenticity_verified": False,
                "schema": "experiments7-cutover-receipt-validation/v1",
                "status": "blocked",
                "verified": False,
            },
            "failures": [{"check": "startup", "error": str(exc), "type": type(exc).__name__}],
            "final_complete": False,
            "known_gaps": [],
            "mode": getattr(args, "mode", PHASE_MODE),
            "post_cutover_local_state": {
                "error": "not evaluated because validator startup failed",
                "no_mutations": True,
                "schema": "experiments7-post-cutover-local-state-validation/v1",
                "status": "blocked",
                "verified": False,
            },
            "phase_ready": False,
            "schema": REPORT_SCHEMA,
            "status": "validation_failed",
        }
        report.update(
            _completion_semantics(
                mode=getattr(args, "mode", PHASE_MODE),
                phase_ready=False,
                final_complete=False,
            )
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return report_exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
