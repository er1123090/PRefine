"""Deterministic G2 metadata construction from the sealed G1 registry/lineage."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .selection import pretty_bytes, sha256_bytes


STAGES = (
    "dataset",
    "session",
    "preprocessing",
    "prompt",
    "tool_schema",
    "inference",
    "runtime",
    "parser",
    "normalization",
    "error_policy",
    "filtering",
    "metric",
    "denominator",
    "aggregation",
    "reporting",
    "evaluation",
)

REFERENCE_ONLY = frozenset({"eval4_mt_parse_0103a", "eval4_mt_parse_0103b"})
SUBPROCESS_STAGES = frozenset(
    {"preprocessing", "inference", "evaluation", "reporting", "filtering"}
)
SECRET_ENV_KEYS = ("OPENAI_API_KEY", "GOOGLE_API_KEY", "VLLM_API_KEY", "EMBEDDING_API_KEY")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _origin_label(variant: dict[str, Any]) -> str:
    if variant["variant_id"] in REFERENCE_ONLY:
        return "cross-origin reference"
    roots = sorted({row["origin_root"] for row in variant["origin_evidence"]})
    return roots[0] if len(roots) == 1 else "cross-origin"


def _execution_kind(variant: dict[str, Any]) -> str:
    if variant["variant_id"] in REFERENCE_ONLY:
        return "reference_only"
    if variant.get("expands_to"):
        return "compatibility_selector"
    if variant["stage"] == "dataset":
        return "data"
    if variant["stage"] == "prompt":
        return "prompt"
    if variant["stage"] in SUBPROCESS_STAGES:
        return "subprocess"
    return "library"


def _expected_outputs(stage: str, execution_kind: str) -> list[str]:
    if execution_kind == "reference_only":
        return ["{output_dir}/reference-comparison.json"]
    return {
        "preprocessing": [
            "{output_dir}/memory.json",
            "{output_dir}/verifier.json",
            "{output_dir}/refinement.json",
        ],
        "inference": ["{output_dir}/predictions.jsonl", "{output_dir}/run.log"],
        "evaluation": ["{output_dir}/metrics.csv"],
        "reporting": ["{output_dir}/report.csv"],
        "filtering": ["{output_dir}/analysis/summary.json"],
    }.get(stage, [])


def _argv_templates(variant_id: str, stage: str) -> dict[str, list[str]]:
    python = ["{python}", "{entrypoint}"]
    if variant_id.startswith("latentpref4_") or variant_id.startswith("latentpref6_"):
        common = python + [
            "--input", "{input}", "--output", "{output}",
            "--verifier_output", "{verifier_output}",
            "--refinement_output", "{refinement_output}",
        ]
        if variant_id.endswith("_api"):
            common += ["--provider", "{provider}"]
        return {"default": common + [
            "--model", "{model}", "--api_base", "{api_base}",
            "--api_key", "{env:OPENAI_API_KEY}", "--concurrency", "{concurrency}",
        ]}
    if variant_id.startswith("preprocess5_v"):
        common = python + [
            "--input", "{input}", "--output", "{output}",
            "--verifier_output", "{verifier_output}",
            "--refinement_output", "{refinement_output}",
            "--provider", "{provider}", "--model", "{model}",
            "--api_base", "{api_base}", "--api_key", "{env:OPENAI_API_KEY}",
            "--concurrency", "{concurrency}", "--max_retries", "{max_retries}",
            "--memory_mode", "{memory_mode}",
        ]
        if variant_id == "preprocess5_v1":
            common += ["--true_blind", "{true_blind}"]
        if variant_id == "preprocess5_v3":
            common += [
                "--pref_list_path", "{pref_list_path}",
                "--v3_min_support", "{v3_min_support}",
            ]
        return {"default": common}
    if variant_id in {"emem4_preprocess", "emem5_preprocess"}:
        return {"build": python + [
            "--input_path", "{input_path}", "--index_root", "{index_root}",
            "--manifest_path", "{manifest_path}", "--llm_model", "{llm_model}",
            "--llm_base_url", "{llm_base_url}", "--embedding_model", "{embedding_model}",
            "--embedding_base_url", "{embedding_base_url}",
            "--api_key", "{env:OPENAI_API_KEY}",
            "--embedding_api_key", "{env:EMBEDDING_API_KEY}",
            "--support_json_schema", "{support_json_schema}",
            "--run_log_path", "{run_log_path}",
        ]}
    if variant_id in {"emem4_infer", "emem5_infer"}:
        base = python + [
            "--manifest_path", "{manifest_path}", "--input_path", "{input_path}",
            "--output_path", "{output_path}", "--log_path", "{log_path}",
            "--run_log_path", "{run_log_path}", "--pref_list_path", "{pref_list_path}",
            "--pref_group_path", "{pref_group_path}",
            "--tools_schema_path", "{tools_schema_path}", "--pref_type", "{pref_type}",
            "--context_type", "{context_type}", "--model_name", "{model_name}",
            "--reasoning_effort", "{reasoning_effort}", "--concurrency", "{concurrency}",
            "--memory_top_k", "{memory_top_k}", "--base_url", "{base_url}",
            "--api_key", "{env:OPENAI_API_KEY}",
            "--retrieval_api_key", "{env:OPENAI_API_KEY}",
            "--retrieval_embedding_api_key", "{env:EMBEDDING_API_KEY}",
        ]
        return {
            "singleturn": base + ["--query_path", "{query_path}"],
            "multiturn": base + ["--multiturn_path", "{multiturn_path}"],
        }
    if stage == "inference":
        base = python + [
            "--input_path", "{input_path}", "--memory_path", "{memory_path}",
            "--output_path", "{output_path}", "--log_path", "{log_path}",
            "--pref_list_path", "{pref_list_path}", "--pref_group_path", "{pref_group_path}",
            "--tools_schema_path", "{tools_schema_path}", "--pref_type", "{pref_type}",
            "--context_type", "{context_type}", "--model_name", "{model_name}",
            "--concurrency", "{concurrency}", "--max_queries", "{max_queries}",
        ]
        if variant_id.startswith(("infer4_", "infer6_")):
            base += ["--reasoning_effort", "{reasoning_effort}"]
        if "vllm" in variant_id:
            base += (["--vllm_url", "{vllm_url}"] if variant_id.startswith("infer5_")
                     else ["--base_url", "{base_url}", "--api_key", "{env:VLLM_API_KEY}"])
        return {
            "singleturn": base + ["--query_path", "{query_path}"],
            "multiturn": base + ["--multiturn_path", "{multiturn_path}"],
        }
    if stage in {"evaluation", "aggregation"}:
        base = python + ["--out_csv", "{out_csv}"]
        return {
            "singleturn": base + ["--json_path", "{json_path}"],
            "multiturn": base + ["--root_dir", "{root_dir}", "--pref_list_path", "{pref_list_path}"],
        }
    if stage == "reporting" and "session_memory" in variant_id:
        return {"default": python + [
            "--run_manifest", "{run_manifest}", "--out_csv", "{out_csv}",
            "--out_summary_csv", "{out_summary_csv}", "--logs_dir", "{logs_dir}",
        ]}
    if stage == "filtering":
        return {"default": python + [
            "--run_dir", "{run_dir}", "--task", "{task}",
            "--analysis_dir", "{analysis_dir}",
        ]}
    return {}


def build_adapters(
    registry: dict[str, Any],
    lineage: list[dict[str, Any]],
    *,
    registry_hash: str,
    code_lineage_hash: str,
    config_lineage_hash: str,
) -> dict[str, Any]:
    lineage_by_id = {row["lineage_id"]: row for row in lineage}
    records: list[dict[str, Any]] = []
    for variant in sorted(registry["variants"], key=lambda item: item["variant_id"]):
        variant_id = variant["variant_id"]
        kind = _execution_kind(variant)
        roots = sorted({origin["origin_root"] for origin in variant["origin_evidence"]})
        origin_label = _origin_label(variant)
        source_rows = [lineage_by_id[lineage_id] for lineage_id in variant["lineage_ids"]]
        destinations = list(variant["destinations"])
        entrypoints = [path for path in destinations if path.endswith(".py")]
        prerequisites: list[str] = []
        if kind in {"subprocess", "prompt", "library"}:
            prerequisites.append("published_origin_snapshot")
        if kind == "data":
            prerequisites.append("published_exact_data_snapshot")
        if kind == "subprocess":
            prerequisites.append("declared_runtime_dependencies")
        if variant["stage"] in {"preprocessing", "inference"}:
            prerequisites.append("credentials_or_local_service")
        record = {
            "adapter_id": f"adapter.{variant_id}.v1",
            "variant_id": variant_id,
            "label": f"{origin_label} / {variant['stage']} / {variant_id}",
            "stage": variant["stage"],
            "neutral_family": variant["neutral_family"],
            "kind": variant["kind"],
            "execution_kind": kind,
            "origin_root": roots[0] if len(roots) == 1 else "cross-origin",
            "origin_label": origin_label,
            "destinations": destinations,
            "source_lineage_ids": list(variant["lineage_ids"]),
            "source_manifest_record_ids": [row["source_manifest_record_id"] for row in source_rows],
            "source_sha256": [row["origin_sha256"] for row in source_rows],
            "source_bindings": [
                {
                    "lineage_id": row["lineage_id"],
                    "source_manifest_record_id": row["source_manifest_record_id"],
                    "origin_root": row["origin_root"],
                    "origin_relative_path": row["origin_relative_path"],
                    "origin_sha256": row["origin_sha256"],
                    "planned_destination": row["planned_destination"],
                }
                for row in source_rows
            ],
            "entrypoint_destination": entrypoints[0] if entrypoints and kind == "subprocess" else None,
            "alternate_entrypoints": entrypoints[1:] if kind == "subprocess" else [],
            "argv_template": _argv_templates(variant_id, variant["stage"]),
            "required_inputs": [],
            "required_configs": [],
            "expected_outputs": _expected_outputs(variant["stage"], kind),
            "environment_keys": list(SECRET_ENV_KEYS) if variant["stage"] in {"preprocessing", "inference"} else [],
            "external_prerequisites": prerequisites,
            "affected_outputs": list(variant["affected_outputs"]),
            "equivalence_state": variant["equivalence_state"],
            "reference_only_reason": (
                "historical parser output snapshot has no executable parser source"
                if kind == "reference_only" else None
            ),
            "snapshot_state": "reference_unpublished" if kind == "reference_only" else "unpublished",
            "final_snapshot_bindings": [
                {"path": path, "sha256": None, "state": "unpublished"}
                for path in destinations
            ],
        }
        records.append(record)
    return {
        "schema": "experiments7-g2-adapters/v1",
        "registry_sha256": registry_hash,
        "code_lineage_sha256": code_lineage_hash,
        "config_lineage_sha256": config_lineage_hash,
        "selection_policy": {
            "implicit_default": False,
            "byte_identity_is_not_equivalence": True,
            "unpublished_snapshots_are_not_executable": True,
        },
        "adapters": records,
    }


def _profile(
    profile_id: str,
    label: str,
    kind: str,
    stages: dict[str, str],
    reason: str,
) -> dict[str, Any]:
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise ValueError(f"unknown stages in {profile_id}: {sorted(unknown)}")
    not_applicable = [
        {"stage": stage, "reason": reason}
        for stage in STAGES
        if stage not in stages
    ]
    return {
        "profile_id": profile_id,
        "label": label,
        "kind": kind,
        "stages": {stage: stages[stage] for stage in STAGES if stage in stages},
        "not_applicable": not_applicable,
    }


def _origin_full(origin: int, *, preprocessing: str, inference: str, runtime: str) -> dict[str, str]:
    if origin == 4:
        return {
            "dataset": "dataset4_dev6", "session": "session4_inline",
            "preprocessing": preprocessing, "prompt": "prompt4_inference",
            "tool_schema": "toolschema4_inline", "inference": inference,
            "runtime": runtime, "parser": "parser4_inline",
            "normalization": "normalization4_legacy", "error_policy": "error4_compat",
            "filtering": "filter4_session", "metric": "metric4_legacy",
            "denominator": "denominator4_legacy", "aggregation": "aggregation4_legacy",
            "reporting": "report4_legacy", "evaluation": "eval4_legacy",
        }
    if origin == 5:
        return {
            "dataset": "dataset5_dev6", "session": "session5_data_utils",
            "preprocessing": preprocessing, "prompt": "prompt5_bundle",
            "tool_schema": "toolschema5_serialized", "inference": inference,
            "runtime": runtime, "parser": "parser5_canonical",
            "normalization": "normalization5_canonical", "error_policy": "error5_native",
            "filtering": "filter5_data_utils", "metric": "metric5_canonical",
            "denominator": "denominator5_canonical", "aggregation": "aggregation5_canonical",
            "reporting": "report5_canonical", "evaluation": "eval5_canonical",
        }
    return {
        "dataset": "dataset6_dev6", "session": "session6_inline",
        "preprocessing": preprocessing, "prompt": "prompt6_inference",
        "tool_schema": "toolschema6_inline", "inference": inference,
        "runtime": runtime, "parser": "parser6_inline",
        "normalization": "normalization6_release", "error_policy": "error6_release",
        "filtering": "filter6_session", "metric": "metric6_release",
        "denominator": "denominator6_release", "aggregation": "aggregation6_release",
        "reporting": "report6_release", "evaluation": "eval6_release",
    }


def _ensure_selector_expansion(stages: dict[str, str], variants: dict[str, dict[str, Any]]) -> dict[str, str]:
    result = dict(stages)
    evaluation = result.get("evaluation")
    if evaluation:
        expansion = variants[evaluation].get("expands_to") or {}
        for stage, variant_id in expansion.items():
            current = result.get(stage)
            if current is not None and current != variant_id:
                raise ValueError(f"selector conflict: {evaluation}:{stage}:{current}:{variant_id}")
            result[stage] = variant_id
    return result


def build_profiles(registry: dict[str, Any], *, registry_hash: str, adapters_hash: str) -> dict[str, Any]:
    variants = {item["variant_id"]: item for item in registry["variants"]}
    records: list[dict[str, Any]] = []

    definitions = [
        ("exp4_api_eval4", "experiments4 / API / eval4_legacy", 4, "latentpref4_api", "infer4_api", "runtime4_api"),
        ("exp4_vllm_eval4", "experiments4 / vLLM / eval4_legacy", 4, "latentpref4_vllm", "infer4_vllm", "runtime4_vllm"),
        ("exp4_compat_eval4", "experiments4 / compatibility / infer4_compat", 4, "latentpref4_api", "infer4_compat", "runtime4_api"),
        ("exp5_native_eval5", "experiments5 / native / eval5_canonical", 5, "preprocess5_v1", "infer5_native", "runtime5_native"),
        ("exp5_v1_api_eval5", "experiments5 / v1 API / eval5_canonical", 5, "preprocess5_v1", "infer5_v1_api", "runtime5_native"),
        ("exp5_v1_vllm_eval5", "experiments5 / v1 vLLM / eval5_canonical", 5, "preprocess5_v1", "infer5_v1_vllm", "runtime5_native"),
        ("exp5_v2_api_eval5", "experiments5 / v2 API / eval5_canonical", 5, "preprocess5_v2", "infer5_v2_api", "runtime5_native"),
        ("exp5_v2_vllm_eval5", "experiments5 / v2 vLLM / eval5_canonical", 5, "preprocess5_v2", "infer5_v2_vllm", "runtime5_native"),
        ("exp5_v3_api_eval5", "experiments5 / v3 API / eval5_canonical", 5, "preprocess5_v3", "infer5_v3_api", "runtime5_native"),
        ("exp5_v3_vllm_eval5", "experiments5 / v3 vLLM / eval5_canonical", 5, "preprocess5_v3", "infer5_v3_vllm", "runtime5_native"),
        ("exp6_api_eval6", "experiments6 / API / eval6_release", 6, "latentpref6_api", "infer6_api", "runtime6_api"),
        ("exp6_vllm_eval6", "experiments6 / vLLM / eval6_release", 6, "latentpref6_vllm", "infer6_vllm", "runtime6_vllm"),
        ("exp6_compat_eval6", "experiments6 / compatibility / infer6_release", 6, "latentpref6_api", "infer6_release", "runtime6_api"),
    ]
    for profile_id, label, origin, preprocessing, inference, runtime in definitions:
        records.append(_profile(
            profile_id, label, "composed",
            _origin_full(origin, preprocessing=preprocessing, inference=inference, runtime=runtime),
            "not applicable to this complete origin profile",
        ))

    emem4 = _origin_full(4, preprocessing="emem4_preprocess", inference="emem4_infer", runtime="runtime4_api")
    emem4["prompt"] = "emem4_prompt"
    emem5 = _origin_full(5, preprocessing="emem5_preprocess", inference="emem5_infer", runtime="runtime5_native")
    emem5["prompt"] = "emem5_prompt"
    records.extend([
        _profile("emem4_eval4", "experiments4 / E-Mem / eval4_legacy", "composed", emem4, "not applicable"),
        _profile("emem5_eval5", "experiments5 / E-Mem / eval5_canonical", "composed", emem5, "not applicable"),
    ])

    session4 = {
        "dataset": "dataset4_dev6",
        "filtering": "filter4_session",
        "reporting": "report4_session_memory",
    }
    session6 = {
        "dataset": "dataset6_dev6",
        "filtering": "filter6_session",
        "reporting": "report6_session_memory",
    }
    records.extend([
        _profile("session4_report_eval4", "experiments4 / session reporting / eval4 label", "stage_isolated", session4, "session-report-only profile; inference and aggregate evaluation are not implied"),
        _profile("session6_report_eval6", "experiments6 / session reporting / eval6 label", "stage_isolated", session6, "session-report-only profile; inference and aggregate evaluation are not implied"),
        _profile(
            "exp5_legacy_multiturn_eval5",
            "experiments5 / evaluation / eval5_legacy_mt",
            "stage_isolated",
            {"dataset": "dataset5_dev6", "evaluation": "eval5_legacy_mt"},
            "monolithic historical evaluator; no semantic substage expansion asserted",
        ),
        _profile(
            "parser_reference_0103a_eval4",
            "cross-origin reference / parser / eval4_mt_parse_0103a",
            "reference",
            {"parser": "eval4_mt_parse_0103a"},
            "historical parser snapshot is reference-only",
        ),
        _profile(
            "parser_reference_0103b_eval4",
            "cross-origin reference / parser / eval4_mt_parse_0103b",
            "reference",
            {"parser": "eval4_mt_parse_0103b"},
            "historical parser snapshot is reference-only",
        ),
    ])

    covered = {variant_id for record in records for variant_id in record["stages"].values()}
    for variant_id in sorted(variants):
        if variant_id in covered:
            continue
        variant = variants[variant_id]
        isolated = _ensure_selector_expansion({variant["stage"]: variant_id}, variants)
        records.append(_profile(
            f"coverage_{variant_id}",
            f"{_origin_label(variant)} / {variant['stage']} / {variant_id}",
            "stage_isolated",
            isolated,
            "stage-isolated coverage; no unstated semantic composition",
        ))

    records.sort(key=lambda item: item["profile_id"])
    return {
        "schema": "experiments7-g2-profiles/v1",
        "profiles_schema_version": 1,
        "registry_sha256": registry_hash,
        "adapters_sha256": adapters_hash,
        "implicit_default": False,
        "stages": list(STAGES),
        "profiles": records,
    }


def build_compatibility(
    registry: dict[str, Any],
    profiles: dict[str, Any],
    adapters: dict[str, Any],
    *,
    registry_hash: str,
    profiles_hash: str,
    adapters_hash: str,
    code_lineage_hash: str,
    config_lineage_hash: str,
) -> dict[str, Any]:
    variants = {item["variant_id"]: item for item in registry["variants"]}
    adapter_by_id = {item["variant_id"]: item for item in adapters["adapters"]}
    allowed: list[dict[str, Any]] = []
    for profile in profiles["profiles"]:
        expanded = _ensure_selector_expansion(profile["stages"], variants)
        origins = sorted({adapter_by_id[item]["origin_label"] for item in expanded.values()})
        prerequisites = sorted({
            prerequisite
            for item in expanded.values()
            for prerequisite in adapter_by_id[item]["external_prerequisites"]
        })
        allowed.append({
            "profile_id": profile["profile_id"],
            "expanded_stages": {stage: expanded[stage] for stage in STAGES if stage in expanded},
            "not_applicable": [item["stage"] for item in profile["not_applicable"]],
            "origin_labels": origins,
            "external_prerequisites": prerequisites,
        })
    return {
        "schema": "experiments7-g2-compatibility/v1",
        "compatibility_schema_version": 1,
        "registry_schema_version": registry["registry_schema_version"],
        "profiles_schema_version": profiles["profiles_schema_version"],
        "hash_bindings": {
            "registry_sha256": registry_hash,
            "profiles_sha256": profiles_hash,
            "adapters_sha256": adapters_hash,
            "code_lineage_sha256": code_lineage_hash,
            "config_lineage_sha256": config_lineage_hash,
        },
        "allowed_profiles": allowed,
        "forbidden_cross_origin_pairs": [
            ["experiments4", "experiments5"],
            ["experiments4", "experiments6"],
            ["experiments5", "experiments6"],
            ["cross-origin reference", "experiments4"],
            ["cross-origin reference", "experiments5"],
            ["cross-origin reference", "experiments6"],
        ],
        "reference_only_restrictions": {
            variant_id: "compare-reference only; live execution forbidden"
            for variant_id in sorted(REFERENCE_ONLY)
        },
        "reason_codes": {
            "implicit_default": "PROFILE_ID_REQUIRED",
            "cross_origin": "FORBIDDEN_CROSS_ORIGIN_PAIR",
            "parser_reference_execution": "REFERENCE_ONLY_EXECUTION_FORBIDDEN",
            "unpublished_snapshot": "EXTERNAL_PREREQUISITES_UNAVAILABLE",
            "stale_hash": "STALE_CONTROL_HASH",
        },
    }


def build_snapshot_plan(
    lineage: Iterable[dict[str, Any]],
    *,
    registry_hash: str,
    code_lineage_hash: str,
    config_lineage_hash: str,
) -> dict[str, Any]:
    records = []
    for row in sorted(lineage, key=lambda item: item["lineage_id"]):
        records.append({
            **row,
            "snapshot_state": "pending_protected_snapshot",
            "final_sha256": None,
            "reason_code": "PROTECTED_SNAPSHOT_REQUIRES_SEALED_G0_EXECUTOR",
        })
    return {
        "schema": "experiments7-g2-snapshot-plan/v1",
        "hash_bindings": {
            "registry_sha256": registry_hash,
            "code_lineage_sha256": code_lineage_hash,
            "config_lineage_sha256": config_lineage_hash,
        },
        "no_globs": True,
        "protected_roots_writable": False,
        "records": records,
    }


def construct_metadata(root: Path) -> dict[str, bytes]:
    registry_path = root / "variants/registry.json"
    code_path = root / "lineage/code.jsonl"
    config_path = root / "lineage/config.jsonl"
    registry = _load_json(registry_path)
    code = _load_jsonl(code_path)
    config = _load_jsonl(config_path)
    registry_hash = _sha(registry_path)
    code_hash = _sha(code_path)
    config_hash = _sha(config_path)
    adapters = build_adapters(
        registry, code + config,
        registry_hash=registry_hash,
        code_lineage_hash=code_hash,
        config_lineage_hash=config_hash,
    )
    adapters_payload = pretty_bytes(adapters)
    adapters_hash = sha256_bytes(adapters_payload)
    profiles = build_profiles(registry, registry_hash=registry_hash, adapters_hash=adapters_hash)
    profiles_payload = pretty_bytes(profiles)
    profiles_hash = sha256_bytes(profiles_payload)
    compatibility = build_compatibility(
        registry, profiles, adapters,
        registry_hash=registry_hash,
        profiles_hash=profiles_hash,
        adapters_hash=adapters_hash,
        code_lineage_hash=code_hash,
        config_lineage_hash=config_hash,
    )
    snapshot_plan = build_snapshot_plan(
        code + config,
        registry_hash=registry_hash,
        code_lineage_hash=code_hash,
        config_lineage_hash=config_hash,
    )
    return {
        "configs/adapters.json": adapters_payload,
        "variants/profiles.json": profiles_payload,
        "variants/compatibility.json": pretty_bytes(compatibility),
        "configs/snapshot-plan.json": pretty_bytes(snapshot_plan),
    }
