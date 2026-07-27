"""Offline integrity checks for the experiment8 MPT_v2 workspace."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "data/MPT_v2_mix600.json",
    "config/schema_easy.json",
    "config/schema_all.json",
    "config/pref_list.json",
    "config/pref_group.json",
    "config/query_singleturn_hint.json",
    "config/query_singleturn_nohint.json",
    "config/query_multiturn_hint.json",
    "config/query_multiturn_nohint.json",
    "config/ablation_matrix.json",
    "methods/vanilla_llm/inference.py",
    "methods/ours_memory/build_memory.py",
    "methods/ours_memory/inference_singleturn.py",
    "methods/ours_memory/inference_multiturn.py",
    "methods/rag/build_index.py",
    "methods/rag/inference.py",
    "methods/mem0/build_memory.py",
    "methods/mem0/inference_singleturn.py",
    "methods/mem0/inference_multiturn.py",
    "methods/mem0_local/UPSTREAM.json",
    "methods/mem0_local/bootstrap_upstream.py",
    "methods/mem0_local/build_memory.py",
    "methods/mem0_local/runtime.py",
    "methods/langmem/build_memory.py",
    "methods/langmem/inference.py",
    "methods/amem/build_memory_batch.py",
    "methods/amem/inference.py",
    "methods/amem/inference_batch.py",
    "src/evaluation/metrics.py",
    "evaluation/eval_singleturn.py",
    "evaluation/eval_multiturn.py",
    "ablations/gvr/build_memory.py",
    "ablations/gvr/run_mpt_v2.sh",
    "ablations/token_count/analyze.py",
    "ablations/api_arguments/analyze.py",
    "scripts/run_inference.py",
    "src/construction_usage.py",
    "src/exp4_runtime/tabular.py",
    "src/token_measurement.py",
]

EXPECTED_SHA256 = {
    "data/MPT_v2_mix600.json": (
        "679fd024219953b49e565c9bd401e9bd0af33297c1a1d683634d8cc2e4abd01b"
    ),
    "src/evaluation/metrics.py": (
        "ae2db7707a3ba577ace94a7436b57ee5f17c0e87be1ee6376e2f50fef645e604"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    missing = [relative for relative in REQUIRED if not (ROOT / relative).is_file()]
    if missing:
        raise SystemExit(f"Missing required files: {missing}")

    dataset = json.loads((ROOT / "data/MPT_v2_mix600.json").read_text())
    if not isinstance(dataset, list) or len(dataset) != 600:
        raise SystemExit("MPT_v2_mix600.json must contain exactly 600 examples")

    actual_hashes = {
        relative: sha256(ROOT / relative) for relative in EXPECTED_SHA256
    }
    mismatches = {
        relative: {"expected": EXPECTED_SHA256[relative], "actual": actual}
        for relative, actual in actual_hashes.items()
        if actual != EXPECTED_SHA256[relative]
    }
    if mismatches:
        raise SystemExit(f"Source integrity mismatch: {mismatches}")

    for path in ROOT.rglob("*.py"):
        if "__pycache__" not in path.parts:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    print(
        json.dumps(
            {
                "status": "ok",
                "dataset_examples": len(dataset),
                "required_files": len(REQUIRED),
                "python_files_parsed": len(list(ROOT.rglob("*.py"))),
                "source_hashes": actual_hashes,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
