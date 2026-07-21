#!/usr/bin/env python3
"""Archive Distill-Qwen memory-source artifacts before 0528-Qwen3 replacement.

This intentionally classifies by path position, not substring. Distill-Qwen
inference-model directories remain active; only Distill-Qwen memory-source
directories and stale top-level result files are archived.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


BASE = Path("/data/minseo")
EXP5 = BASE / "experiments5"
DISTILL_SAFE = "deepseek-ai_DeepSeek-R1-Distill-Qwen-7B"
QWEN3_SAFE = "deepseek-ai_DeepSeek-R1-0528-Qwen3-8B"

OUTPUT_RUNS = [
    EXP5 / "outputs/our_memory/1229_dev6_memory_variants_20260517",
    EXP5 / "outputs/our_memory/1229_dev6_session_incremental_gen_only_20260524",
]

RESULT_DIRS = [
    EXP5 / "results/our_memory/1229_dev6_distill_qwen_memory_4inf_eval_20260524",
    EXP5
    / "results/our_memory/1229_dev6_session_incremental_gen_only_20260524"
    / "comparison_vs_refine_ours_common_distill_qwen",
]


@dataclass
class MoveRow:
    path_class: str
    original_path: str
    archive_path: str
    replacement_path: str
    exists: bool
    bytes: int
    replacement_status: str


def byte_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def unique_existing(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        try:
            resolved = path.resolve()
        except FileNotFoundError:
            resolved = path
        if resolved in seen or not path.exists():
            continue
        seen.add(resolved)
        out.append(path)
    # Move deeper paths first to avoid child paths being moved after parents.
    return sorted(out, key=lambda p: len(p.parts), reverse=True)


def replace_memory_source_component(path: Path) -> Path:
    """Replace only the final memory-source component, not inference model names."""

    if path.name != DISTILL_SAFE:
        raise ValueError(f"expected final component {DISTILL_SAFE}, got: {path}")
    return path.with_name(QWEN3_SAFE)


def output_candidates(run_root: Path, stamp: str) -> list[MoveRow]:
    candidates: list[tuple[str, Path]] = []
    candidates.extend(
        ("memory_dir", p)
        for p in run_root.glob(f"memories/*/{DISTILL_SAFE}")
        if p.is_dir()
    )
    candidates.extend(
        ("inference_memory_source_dir", p)
        for p in run_root.glob(f"inference/*/*/*/memory_api/*/{DISTILL_SAFE}")
        if p.is_dir()
    )
    candidates.extend(
        ("repeat_inference_memory_source_dir", p)
        for p in run_root.glob(f"repeats/*/inference/*/*/memory_api/*/{DISTILL_SAFE}")
        if p.is_dir()
    )

    rows: list[MoveRow] = []
    for path_class, path in [(c, p) for c, p in candidates if p.exists()]:
        rel = path.relative_to(run_root)
        archive = run_root / "_deprecated" / f"distill_qwen_memory_source_mismatch_{stamp}" / rel
        replacement = replace_memory_source_component(path)
        rows.append(
            MoveRow(
                path_class=path_class,
                original_path=str(path),
                archive_path=str(archive),
                replacement_path=str(replacement),
                exists=True,
                bytes=byte_size(path),
                replacement_status="pending",
            )
        )
    return rows


def result_candidates(result_dir: Path, stamp: str) -> list[MoveRow]:
    rows: list[MoveRow] = []
    if not result_dir.exists():
        return rows
    for path in sorted(result_dir.iterdir()):
        if path.name == "_deprecated":
            continue
        if not path.is_file():
            continue
        archive = result_dir / "_deprecated" / f"distill_qwen_memory_source_mismatch_{stamp}" / path.name
        rows.append(
            MoveRow(
                path_class="stable_result_file",
                original_path=str(path),
                archive_path=str(archive),
                replacement_path=str(path),
                exists=True,
                bytes=byte_size(path),
                replacement_status="pending_recreate_at_original_path",
            )
        )
    return rows


def collect_rows(stamp: str) -> list[MoveRow]:
    rows: list[MoveRow] = []
    for run_root in OUTPUT_RUNS:
        rows.extend(output_candidates(run_root, stamp))
    for result_dir in RESULT_DIRS:
        rows.extend(result_candidates(result_dir, stamp))

    by_original: dict[str, MoveRow] = {}
    for row in rows:
        by_original[row.original_path] = row
    return list(by_original.values())


def write_csv(path: Path, rows: list[MoveRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "path_class",
        "original_path",
        "archive_path",
        "replacement_path",
        "exists",
        "bytes",
        "replacement_status",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_jsonl(path: Path, rows: list[MoveRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def move_rows(rows: list[MoveRow]) -> None:
    for row in rows:
        src = Path(row.original_path)
        dst = Path(row.archive_path)
        if not src.exists():
            continue
        if dst.exists():
            raise FileExistsError(f"archive destination already exists: {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))


def write_archive_readmes(rows: list[MoveRow], stamp: str) -> None:
    archive_roots = sorted({Path(row.archive_path).parents[0] for row in rows})
    for archive_root in archive_roots:
        readme = archive_root / "README.md"
        readme.write_text(
            "\n".join(
                [
                    "# Distill-Qwen memory-source mismatch archive",
                    "",
                    f"Archived at: {stamp}",
                    "",
                    "Reason: Distill-Qwen-7B appeared as the memory source, but the",
                    "experiments4-aligned DeepSeek-Qwen memory source is",
                    "`deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`.",
                    "",
                    "Distill-Qwen-7B inference-model artifacts are intentionally kept active.",
                    "",
                ]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stamp", default=datetime.now().strftime("%Y%m%dT%H%M%S"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = collect_rows(args.stamp)
    state_root = EXP5 / "results/our_memory/_alignment_work" / f"0528_qwen3_{args.stamp}"
    write_csv(state_root / "replacement_ledger.csv", rows)
    write_jsonl(state_root / "archive_manifest.jsonl", rows)

    if args.dry_run:
        print(f"DRY_RUN rows={len(rows)}")
        print(f"ledger={state_root / 'replacement_ledger.csv'}")
        return

    move_rows(rows)
    write_archive_readmes(rows, args.stamp)
    print(f"ARCHIVED rows={len(rows)}")
    print(f"ledger={state_root / 'replacement_ledger.csv'}")


if __name__ == "__main__":
    main()
