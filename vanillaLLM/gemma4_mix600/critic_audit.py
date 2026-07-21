from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from core import ContractError, DIFFICULTIES, MODES, SEED, canonical_json, load_json, read_jsonl, sha256_file, write_json_atomic


EXPECTED = {"easy": 554, "medium": 293, "hard": 472}


def _resolve_run(path: Path) -> Path:
    if (path / "run_manifest.json").is_file():
        return path
    candidates = sorted(item for item in path.iterdir() if item.is_dir() and (item / "run_manifest.json").is_file())
    if not candidates:
        raise ContractError(f"no run manifests under {path}")
    return candidates[-1]


def _check(condition: bool, message: str, evidence: list[str]) -> None:
    if not condition:
        raise ContractError(message)
    evidence.append(message)


def audit(run_root: Path) -> dict[str, Any]:
    evidence: list[str] = []
    manifest = load_json(run_root / "run_manifest.json")
    for name, expected in manifest["input_sha256"].items():
        _check(sha256_file(manifest["input_paths"][name]) == expected, f"input hash stable: {name}", evidence)
    for relative, expected in manifest["code_sha256"].items():
        _check(sha256_file(Path("/data/minseo/experiments4") / relative) == expected, f"code hash stable: {relative}", evidence)
    for name, expected in manifest["artifact_sha256"].items():
        _check(sha256_file(run_root / "prepared" / name) == expected, f"prepared hash stable: {name}", evidence)

    targets = read_jsonl(run_root / "prepared/targets.jsonl")
    cases = read_jsonl(run_root / "prepared/cases.jsonl")
    _check(len(targets) == 1319, "exactly 1319 mode-neutral targets", evidence)
    _check(len(cases) == 2638, "exactly 2638 paired cases", evidence)
    _check(len({item["target_id"] for item in targets}) == 1319, "target IDs unique", evidence)
    _check(len({item["case_id"] for item in cases}) == 2638, "case IDs unique", evidence)
    by_pair: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for case in cases:
        by_pair[case["pair_id"]].append(case)
    _check(len(by_pair) == 1319, "exactly 1319 pair IDs", evidence)
    for pair_id, pair in by_pair.items():
        _check(len(pair) == 2 and {item["mode"] for item in pair} == set(MODES), f"pair complete: {pair_id}", evidence)
        _check(pair[0]["target_domain"] == pair[1]["target_domain"], f"pair domain identical: {pair_id}", evidence)
        _check(canonical_json(pair[0]["reference_ground_truth_preference"]) == canonical_json(pair[1]["reference_ground_truth_preference"]), f"pair preference GT identical: {pair_id}", evidence)
    counts = Counter((case["mode"], case["difficulty"]) for case in cases)
    _check(all(counts[(mode, difficulty)] == expected for mode in MODES for difficulty, expected in EXPECTED.items()), "cell matrix is 554/293/472 per mode", evidence)

    runtime = load_json(run_root / "runtime/runtime_manifest.json")
    _check(runtime["gpus"] == [0, 1, 2, 3], "runtime uses GPUs 0,1,2,3", evidence)
    _check(runtime["tensor_parallel_size"] == 4, "runtime tensor parallel size is 4", evidence)
    _check(runtime["tool_call_parser"] == "gemma4" and runtime["reasoning_parser"] == "gemma4", "Gemma4 tool/reasoning parsers enabled", evidence)
    _check(runtime["enable_auto_tool_choice"] is True and runtime["thinking"] is False, "auto tool choice enabled and thinking disabled", evidence)
    _check(runtime["temperature"] == 0.0 and runtime["seed"] == SEED, "decoding seed and temperature frozen", evidence)
    census = load_json(run_root / "runtime/token_census_summary.json")
    _check(census["fits_without_truncation"] is True, "all prompts fit without truncation", evidence)

    canary_ids = set(load_json(run_root / "runtime/canary_manifest.json")["case_ids"])
    seq = read_jsonl(run_root / "predictions/canary-sequential/predictions.jsonl")
    conc = read_jsonl(run_root / "predictions/canary-concurrent/predictions.jsonl")
    _check({row["case_id"] for row in seq} == canary_ids == {row["case_id"] for row in conc}, "sequential/concurrent canary reconciliation", evidence)
    _check(not any(row["status"] == "infra_failure" for row in seq + conc), "canaries have zero infrastructure failures", evidence)
    seq_map = {row["case_id"]: (row["status"], canonical_json(row["normalized_calls"])) for row in seq}
    conc_map = {row["case_id"]: (row["status"], canonical_json(row["normalized_calls"])) for row in conc}
    _check(seq_map == conc_map, "sequential/concurrent normalized canaries are identical", evidence)

    predictions = read_jsonl(run_root / "predictions/full/predictions.jsonl")
    _check(len(predictions) == 2638, "full run has 2638 terminal records", evidence)
    _check(len({row["case_id"] for row in predictions}) == 2638, "full prediction IDs unique", evidence)
    _check({row["case_id"] for row in predictions} == {case["case_id"] for case in cases}, "full prediction reconciliation has zero missing/foreign", evidence)
    _check(not any(row["status"] == "infra_failure" for row in predictions), "full run has zero unresolved infrastructure failures", evidence)

    metrics = load_json(run_root / "metrics/metrics.json")
    _check(metrics["cases"] == 2638 and metrics["pairs"] == 1319, "metrics cover all cases and pairs", evidence)
    for label in ("pooled", *DIFFICULTIES):
        item = metrics["paired_bootstrap"][label]
        _check(item["seed"] == SEED and item["draws"] == 10000 and item["ci_indices"] == [249, 9749], f"bootstrap contract: {label}", evidence)
    _check(metrics["status_counts"].get("infra_failure", 0) == 0, "headline is not suppressed by infrastructure failure", evidence)

    dirty_before = load_json(run_root / "dirty_before.json")
    dirty_after = load_json(run_root / "dirty_after.json")
    _check(dirty_before["git_head"] == dirty_after["git_head"], "git HEAD preserved", evidence)
    _check(dirty_before["tracked_dirty_hashes"] == dirty_after["tracked_dirty_hashes"], "pre-existing tracked dirty bytes preserved", evidence)
    return {
        "verdict": "PASS",
        "run_root": str(run_root),
        "evidence_count": len(evidence),
        "evidence": evidence,
        "artifact_sha256": {
            "predictions": sha256_file(run_root / "predictions/full/predictions.jsonl"),
            "metrics": sha256_file(run_root / "metrics/metrics.json"),
            "report": sha256_file(run_root / "metrics/report.md"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latest-root", required=True)
    args = parser.parse_args()
    run_root = _resolve_run(Path(args.latest_root).resolve())
    result = audit(run_root)
    write_json_atomic(run_root / "critic_audit.json", result)
    print(canonical_json(result))


if __name__ == "__main__":
    main()
