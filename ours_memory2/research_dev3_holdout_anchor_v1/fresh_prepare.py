"""Evaluator-owned fresh-holdout builder.

This command is the only code path that reads ``api_calls_pref`` for this
experiment.  It writes a sanitized history and public tasks for inference, and
writes reference calls only to ``evaluator_vault/gold.jsonl``.  Its stdout is
deliberately aggregate-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ecpr.firewall import sanitize_example, validate_sanitized_history
from ecpr.io import (
    canonical_json,
    load_json,
    sha256_file,
    sha256_text,
    verify_sha256,
    write_json_once,
    write_jsonl_once,
)
from ecpr.prepare import (
    _multi_materializer,
    _multi_queries,
    _pairs,
    _single_materializer,
    _single_queries,
    build_task_gold_rows,
)


ROOT = Path(__file__).resolve().parent


def _rows(path: Path) -> list[dict[str, Any]]:
    value = load_json(path)
    if isinstance(value, dict):
        value = value.get("dataset")
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError("dataset source must be a JSON array of objects")
    return value


def _input(preregistration: dict[str, Any], name: str) -> dict[str, str]:
    value = preregistration.get("inputs", {}).get(name)
    if not isinstance(value, dict):
        raise ValueError(f"missing preregistered input: {name}")
    path = value.get("path")
    digest = value.get("sha256")
    if not isinstance(path, str) or not Path(path).is_absolute() or not isinstance(digest, str):
        raise ValueError(f"invalid preregistered input: {name}")
    verify_sha256(path, digest)
    return {"path": path, "sha256": digest}


def _assert_absent(root: Path) -> None:
    paths = (
        root / "artifacts/history.sanitized.jsonl",
        root / "artifacts/tasks.jsonl",
        root / "evaluator_vault/gold.jsonl",
        root / "evaluator_vault/sealed_manifest.json",
    )
    present = [str(path.relative_to(root)) for path in paths if path.exists() or path.is_symlink()]
    if present:
        raise FileExistsError("fresh holdout artifacts already exist: " + ", ".join(present))


def prepare(root: Path, preregistration_path: Path) -> dict[str, Any]:
    root = root.resolve()
    preregistration = load_json(preregistration_path)
    if preregistration.get("status") != "locked_before_target_metrics":
        raise ValueError("preregistration must be locked before evaluator preparation")
    _assert_absent(root)
    source_spec = _input(preregistration, "fresh_source_history")
    exclusion_spec = _input(preregistration, "excluded_history")
    local_inputs = {
        "single_query": "query_singleturn.json",
        "single_schema": "schema_single.json",
        "multi_query": "query_multiturn-domain.json",
        "multi_schema": "schema_multi.json",
        "preference_slots": "preference_slots.json",
        "preference_groups": "preference_groups.json",
    }
    for name, local_name in local_inputs.items():
        specification = _input(preregistration, name)
        local_path = root / "configs" / local_name
        if sha256_file(local_path) != specification["sha256"]:
            raise ValueError(f"local frozen config differs from preregistration: {name}")

    excluded_rows = _rows(Path(exclusion_spec["path"]))
    excluded_ids = {str(row.get("example_id", "")).strip() for row in excluded_rows}
    if "" in excluded_ids or len(excluded_ids) != len(excluded_rows):
        raise ValueError("exclusion history has invalid or duplicate example IDs")
    source_rows = _rows(Path(source_spec["path"]))
    source_ids = [str(row.get("example_id", "")).strip() for row in source_rows]
    if "" in source_ids or len(set(source_ids)) != len(source_ids):
        raise ValueError("fresh source has invalid or duplicate example IDs")
    fresh_rows = [row for row in source_rows if str(row["example_id"]).strip() not in excluded_ids]
    expected_fresh = int(preregistration["fresh_holdout"]["expected_example_count"])
    if len(fresh_rows) != expected_fresh:
        raise ValueError(
            f"fresh ID filter count mismatch: expected {expected_fresh}, got {len(fresh_rows)}"
        )
    if any(str(row["example_id"]).strip() in excluded_ids for row in fresh_rows):
        raise AssertionError("fresh ID filter overlap")

    configs = root / "configs"
    single_query = load_json(configs / "query_singleturn.json")
    multi_query = load_json(configs / "query_multiturn-domain.json")
    preference_slots = load_json(configs / "preference_slots.json")
    preference_groups = load_json(configs / "preference_groups.json")
    if not isinstance(preference_slots, dict) or not isinstance(preference_groups, dict):
        raise ValueError("frozen preference configs have invalid shapes")
    schema_hashes = {
        "single": sha256_file(configs / "schema_single.json"),
        "multi": sha256_file(configs / "schema_multi.json"),
    }
    materializers = {
        "singleturn": _single_materializer(_single_queries(single_query)),
        "multiturn": _multi_materializer(_multi_queries(multi_query)),
    }
    frozen_seed_rows: list[dict[str, Any]] = []
    for mode, materialize in materializers.items():
        schema_key = "single" if mode == "singleturn" else "multi"
        for difficulty in ("easy", "medium", "hard"):
            for example in fresh_rows:
                for _domain, query, _reference in _pairs(
                    example,
                    difficulty,
                    preference_slots,
                    preference_groups,
                    materialize,
                ):
                    # ``case_key`` is intentionally an ignored seed field here;
                    # build_task_gold_rows deterministically replaces it using
                    # only public identity/multiplicity.
                    frozen_seed_rows.append(
                        {
                            "case_key": "evaluator_seed",
                            "example_id": str(example["example_id"]),
                            "mode": mode,
                            "query": query,
                            "schema_key": schema_key,
                        }
                    )
    tasks, gold, unassigned = build_task_gold_rows(
        frozen_task_rows=frozen_seed_rows,
        raw_examples=fresh_rows,
        preference_slots=preference_slots,
        preference_groups=preference_groups,
        materializers=materializers,
        schema_hashes=schema_hashes,
    )
    if not tasks or len(tasks) != len(gold) or unassigned != 0:
        raise ValueError("fresh public task/gold construction did not close")
    if any(not row["reference_ground_truth"] for row in gold):
        raise ValueError("fresh evaluator produced an empty reference row")

    sanitized = [sanitize_example(row) for row in fresh_rows]
    artifacts = root / "artifacts"
    vault = root / "evaluator_vault"
    write_jsonl_once(artifacts / "history.sanitized.jsonl", sanitized)
    write_jsonl_once(artifacts / "tasks.jsonl", tasks)
    write_jsonl_once(vault / "gold.jsonl", gold)
    history_count = validate_sanitized_history(artifacts / "history.sanitized.jsonl")
    if history_count != len(fresh_rows):
        raise AssertionError("sanitized history count mismatch")
    manifest = {
        "schema_version": 1,
        "kind": "fresh_holdout_evaluator_seal_v1",
        "preregistration_sha256": sha256_file(preregistration_path),
        "source_history_sha256": source_spec["sha256"],
        "excluded_history_sha256": exclusion_spec["sha256"],
        "selection_rule": "source_example_id_not_in_excluded_history_example_id_set",
        "fresh_example_count": len(fresh_rows),
        "fresh_example_id_set_sha256": sha256_text(
            canonical_json(sorted(str(row["example_id"]) for row in fresh_rows))
        ),
        "history_sha256": sha256_file(artifacts / "history.sanitized.jsonl"),
        "task_sha256": sha256_file(artifacts / "tasks.jsonl"),
        "gold_sha256": sha256_file(vault / "gold.jsonl"),
        "schema_sha256": schema_hashes,
        "case_count": len(tasks),
        "gold_opened_by_preparation": False,
        "stdout_policy": "aggregate_hashes_and_counts_only",
    }
    write_json_once(vault / "sealed_manifest.json", manifest)
    return {
        "fresh_example_count": len(fresh_rows),
        "case_count": len(tasks),
        "task_sha256": manifest["task_sha256"],
        "history_sha256": manifest["history_sha256"],
        "gold_sha256": manifest["gold_sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="prepare fresh evaluator-owned holdout")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--preregistration", type=Path, default=ROOT / "preregistration.json")
    args = parser.parse_args(argv)
    result = prepare(args.root, args.preregistration)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
