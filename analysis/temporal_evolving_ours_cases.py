#!/usr/bin/env python3
"""Find temporal/evolving-memory cases where ours_memory outperforms ablations.

The slice labels are built only from raw session metadata and ground truth, not
from any method output, so this can be used as an ablation diagnostic artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import metrics  # noqa: E402


metrics._normalize_date_time_value = lru_cache(maxsize=200_000)(metrics._normalize_date_time_value)
metrics.parse_call_to_slotvals = lru_cache(maxsize=200_000)(metrics.parse_call_to_slotvals)


DEFAULT_PER_FILE = (
    ROOT
    / "results/our_memory/1229_dev6_session_incremental_gen_only_20260524"
    / "comparison_vs_refine_ours_common_distill_qwen/per_file_common_grid.csv"
)
DEFAULT_PREF_LIST = ROOT / "config/pref_list.json"
DEFAULT_OUT = (
    ROOT
    / "results/our_memory/1229_dev6_session_incremental_gen_only_20260524"
    / "temporal_evolving_ours_case_analysis"
)

METHOD_ORDER = [
    "gen_only_session_incremental",
    "refine1",
    "refine2",
    "refine3",
    "ours_memory",
]

CORRECTION_RE = re.compile(
    r"\b("
    r"actually|instead|rather|change|changed|switch|update|updated|"
    r"not anymore|no longer|prefer now|now prefer|new preference|"
    r"correction|correct that|make it|should be"
    r")\b",
    re.IGNORECASE,
)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_examples(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return data


def stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


def make_task_key(ex: Dict[str, Any]) -> str:
    """Key exact tasks, not only example ids.

    A few result files reuse an example_id_sub across different generated task
    variants, so win/loss examples must align on the actual input and GT.
    """
    payload = [
        ex.get("example_id_sub") or ex.get("example_id"),
        ex.get("reference_ground_truth"),
        ex.get("user_utterance"),
    ]
    return hashlib.sha1(stable_json(payload).encode("utf-8")).hexdigest()[:16]


def slot_values_from_calls(calls_field: Any) -> Dict[Tuple[str, str], set[str]]:
    out: Dict[Tuple[str, str], set[str]] = defaultdict(set)
    for call in metrics.extract_calls(calls_field):
        for domain, slot, value in metrics.parse_call_to_slotvals(call):
            out[(domain, slot)].add(value)
    return out


def collect_session_slot_history(sessions: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    history: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for idx, sess in enumerate(sessions or []):
        dialogue_id = str(sess.get("dialogue_id", f"session_{idx}"))
        for call in metrics.extract_calls(sess.get("api_call")):
            for domain, slot, value in metrics.parse_call_to_slotvals(call):
                key = (domain, slot)
                if key not in history:
                    history[key] = {"values": set(), "events": []}
                history[key]["values"].add(value)
                history[key]["events"].append(
                    {"session_index": idx, "dialogue_id": dialogue_id, "value": value}
                )
    return history


def collect_user_text(sessions: Iterable[Dict[str, Any]]) -> str:
    chunks: List[str] = []
    for sess in sessions or []:
        pref = sess.get("preference_utterance")
        if isinstance(pref, list):
            chunks.extend(str(x) for x in pref)
        elif pref:
            chunks.append(str(pref))
        for turn in sess.get("dialogue") or []:
            if str(turn.get("role", "")).lower() == "user":
                chunks.append(str(turn.get("message", "")))
    return "\n".join(chunks)


def pref_evidence_summary(api_calls_pref: Any) -> Dict[str, Any]:
    dialogue_ids: set[str] = set()
    pref_slots: set[str] = set()
    max_count = 0
    evidence_items = 0

    if not isinstance(api_calls_pref, list):
        api_calls_pref = []

    for group in api_calls_pref or []:
        if not isinstance(group, dict):
            continue
        for ev in group.get("evidence") or []:
            if not isinstance(ev, dict):
                continue
            evidence_items += 1
            domain = str(ev.get("domain", ""))
            slot = str(ev.get("slot", ""))
            if domain and slot:
                pref_slots.add(f"{domain}.{slot}")
            for item in ev.get("values") or []:
                if not isinstance(item, dict):
                    continue
                meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
                try:
                    max_count = max(max_count, int(meta.get("count", 0)))
                except Exception:
                    pass
                for did in meta.get("dialogue_ids") or []:
                    dialogue_ids.add(str(did))

    return {
        "evidence_items": evidence_items,
        "dialogue_ids": sorted(dialogue_ids),
        "n_dialogue_ids": len(dialogue_ids),
        "pref_slots": sorted(pref_slots),
        "max_count": max_count,
    }


def classify_example(ex: Dict[str, Any], pref_map: Dict[str, set[str]]) -> Dict[str, Any]:
    sessions = ex.get("sessions") or []
    history = collect_session_slot_history(sessions)
    gt_map = metrics.build_gt_allowed_map(ex.get("reference_ground_truth"))

    any_conflict = []
    target_conflict = []
    any_pref_conflict = []
    target_pref_conflict = []
    for (domain, slot), info in history.items():
        values = sorted(info["values"])
        if len(values) <= 1:
            continue
        is_pref = metrics.is_pref_slot(domain, slot, pref_map)
        label = f"{domain}.{slot}"
        payload = {
            "slot": label,
            "is_pref_slot": is_pref,
            "values": values,
            "events": info["events"],
            "gt_values": sorted(gt_map.get((domain, slot), set())),
        }
        any_conflict.append(payload)
        if is_pref:
            any_pref_conflict.append(payload)
        if (domain, slot) in gt_map:
            target_conflict.append(payload)
            if is_pref:
                target_pref_conflict.append(payload)

    text = collect_user_text(sessions)
    correction_hits = sorted(set(m.group(1).lower() for m in CORRECTION_RE.finditer(text)))
    pref_summary = pref_evidence_summary(ex.get("api_calls_pref"))
    evolving_pref = pref_summary["n_dialogue_ids"] >= 2 or pref_summary["max_count"] >= 2

    return {
        "example_id": ex.get("example_id", ""),
        "example_id_sub": ex.get("example_id_sub", ""),
        "n_sessions": len(sessions),
        "target_slot_conflict": bool(target_conflict),
        "any_slot_conflict": bool(any_conflict),
        "target_pref_slot_conflict": bool(target_pref_conflict),
        "any_pref_slot_conflict": bool(any_pref_conflict),
        "correction_language": bool(correction_hits),
        "evolving_pref": evolving_pref,
        "temporal_or_evolving": bool(target_conflict or evolving_pref or correction_hits),
        "pref_conflict_or_evolving": bool(target_pref_conflict or evolving_pref),
        "target_conflict_slots": target_conflict,
        "any_conflict_slots": any_conflict,
        "target_pref_conflict_slots": target_pref_conflict,
        "any_pref_conflict_slots": any_pref_conflict,
        "correction_hits": correction_hits,
        "pref_evidence": pref_summary,
    }


def score_example(ex: Dict[str, Any], pref_map: Dict[str, set[str]]) -> Dict[str, Any]:
    gt = metrics.build_gt_allowed_map(ex.get("reference_ground_truth"))
    pred = metrics.build_pred_map(ex.get("llm_output"))
    tp, fp, fn = metrics.counts_slot_and_value_or(gt, pred)
    prf = metrics.prf_from_counts(tp, fp, fn)

    gt_pref = metrics.filter_map_by_pref(gt, pref_map, want_pref=True)
    pred_pref = metrics.filter_map_by_pref(pred, pref_map, want_pref=True)
    pref_tp, pref_fp, pref_fn = metrics.counts_slot_and_value_or(gt_pref, pred_pref)
    pref_prf = metrics.prf_from_counts(pref_tp, pref_fp, pref_fn)
    pref_em = 1 if pref_fp == 0 and pref_fn == 0 else 0

    gt_nonpref = metrics.filter_map_by_pref(gt, pref_map, want_pref=False)
    pred_nonpref = metrics.filter_map_by_pref(pred, pref_map, want_pref=False)
    non_tp, non_fp, non_fn = metrics.counts_slot_and_value_or(gt_nonpref, pred_nonpref)
    non_prf = metrics.prf_from_counts(non_tp, non_fp, non_fn)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": prf.precision,
        "recall": prf.recall,
        "f1": prf.f1,
        "pref_tp": pref_tp,
        "pref_fp": pref_fp,
        "pref_fn": pref_fn,
        "pref_f1": pref_prf.f1,
        "pref_em": pref_em,
        "nonpref_tp": non_tp,
        "nonpref_fp": non_fp,
        "nonpref_fn": non_fn,
        "nonpref_f1": non_prf.f1,
    }


def safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def summarize_scores(rows: List[Dict[str, Any]], slice_name: str) -> Dict[str, Any]:
    tp = sum(int(r["tp"]) for r in rows)
    fp = sum(int(r["fp"]) for r in rows)
    fn = sum(int(r["fn"]) for r in rows)
    prf = metrics.prf_from_counts(tp, fp, fn)

    pref_tp = sum(int(r["pref_tp"]) for r in rows)
    pref_fp = sum(int(r["pref_fp"]) for r in rows)
    pref_fn = sum(int(r["pref_fn"]) for r in rows)
    pref_prf = metrics.prf_from_counts(pref_tp, pref_fp, pref_fn)

    non_tp = sum(int(r["nonpref_tp"]) for r in rows)
    non_fp = sum(int(r["nonpref_fp"]) for r in rows)
    non_fn = sum(int(r["nonpref_fn"]) for r in rows)
    non_prf = metrics.prf_from_counts(non_tp, non_fp, non_fn)

    return {
        "slice": slice_name,
        "n_examples": len(rows),
        "overall_tp": tp,
        "overall_fp": fp,
        "overall_fn": fn,
        "precision": prf.precision,
        "recall": prf.recall,
        "f1": prf.f1,
        "pref_em": (sum(int(r["pref_em"]) for r in rows) / len(rows)) if rows else 0.0,
        "pref_f1": pref_prf.f1,
        "nonpref_f1": non_prf.f1,
    }


def compact_output(text: Any, limit: int = 700) -> str:
    if text is None:
        return ""
    s = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:limit] + ("..." if len(s) > limit else "")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-file", type=Path, default=DEFAULT_PER_FILE)
    parser.add_argument("--pref-list", type=Path, default=DEFAULT_PREF_LIST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--turn", choices=["singleturn", "multiturn", "all"], default="multiturn")
    args = parser.parse_args()

    pref_map = metrics.load_pref_list(str(args.pref_list))
    per_file = [r for r in read_csv(args.per_file) if r.get("method") in METHOD_ORDER]
    if args.turn != "all":
        per_file = [r for r in per_file if r.get("turn") == args.turn]

    score_rows: List[Dict[str, Any]] = []
    manifest_by_example: Dict[str, Dict[str, Any]] = {}
    grouped_examples: Dict[Tuple[str, str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    missing_paths: List[Dict[str, Any]] = []

    for row in per_file:
        path = Path(row["path"])
        if not path.exists():
            missing_paths.append({**row, "missing_path": str(path)})
            continue
        examples = load_examples(path)
        group_key = (row["turn"], row["difficulty"], row["inference_model"])
        method = row["method"]
        for ex in examples:
            ex_id = str(ex.get("example_id_sub") or ex.get("example_id"))
            task_key = make_task_key(ex)
            manifest_key = f"{row['turn']}|{row['difficulty']}|{task_key}"
            if manifest_key not in manifest_by_example:
                manifest_by_example[manifest_key] = {
                    "turn": row["turn"],
                    "difficulty": row["difficulty"],
                    "task_key": task_key,
                    **classify_example(ex, pref_map),
                }
            score = score_example(ex, pref_map)
            record = {
                "method": method,
                "turn": row["turn"],
                "difficulty": row["difficulty"],
                "inference_model": row["inference_model"],
                "example_id": ex.get("example_id", ""),
                "example_id_sub": ex_id,
                "task_key": task_key,
                **manifest_by_example[manifest_key],
                **score,
            }
            score_rows.append(record)
            grouped_examples[group_key].setdefault(task_key, {})[method] = {
                "example": ex,
                "score": score,
                "class": manifest_by_example[manifest_key],
            }

    slice_defs = {
        "all": lambda r: True,
        "target_slot_conflict": lambda r: bool(r["target_slot_conflict"]),
        "target_pref_slot_conflict": lambda r: bool(r["target_pref_slot_conflict"]),
        "any_pref_slot_conflict": lambda r: bool(r["any_pref_slot_conflict"]),
        "evolving_pref": lambda r: bool(r["evolving_pref"]),
        "pref_conflict_or_evolving": lambda r: bool(r["pref_conflict_or_evolving"]),
        "correction_language": lambda r: bool(r["correction_language"]),
        "temporal_or_evolving": lambda r: bool(r["temporal_or_evolving"]),
    }

    summary_rows: List[Dict[str, Any]] = []
    for (method, turn, difficulty, inference_model), rows in sorted(
        defaultdict(list, {
            key: val
            for key, val in (
                ((r["method"], r["turn"], r["difficulty"], r["inference_model"]), []) for r in []
            )
        }).items()
    ):
        pass

    by_group: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in score_rows:
        by_group[(r["method"], r["turn"], r["difficulty"], r["inference_model"])].append(r)

    for group_key, rows in sorted(by_group.items()):
        method, turn, difficulty, inference_model = group_key
        for slice_name, predicate in slice_defs.items():
            subset = [r for r in rows if predicate(r)]
            out = summarize_scores(subset, slice_name)
            out.update(
                {
                    "method": method,
                    "turn": turn,
                    "difficulty": difficulty,
                    "inference_model": inference_model,
                }
            )
            summary_rows.append(out)

    overall_rows: List[Dict[str, Any]] = []
    by_method_slice: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in score_rows:
        for slice_name, predicate in slice_defs.items():
            if predicate(r):
                by_method_slice[(r["method"], slice_name)].append(r)

    for (method, slice_name), rows in sorted(by_method_slice.items()):
        out = summarize_scores(rows, slice_name)
        out.update({"method": method, "turn": args.turn, "difficulty": "all", "inference_model": "all"})
        overall_rows.append(out)

    win_cases: List[Dict[str, Any]] = []
    win_over_gen_cases: List[Dict[str, Any]] = []
    for (turn, difficulty, inference_model), examples_by_method in sorted(grouped_examples.items()):
        for task_key, method_payloads in examples_by_method.items():
            if "ours_memory" not in method_payloads:
                continue
            cls = method_payloads["ours_memory"]["class"]
            if not cls["temporal_or_evolving"]:
                continue
            ours_score = method_payloads["ours_memory"]["score"]
            baseline_scores = {
                m: method_payloads[m]["score"]
                for m in METHOD_ORDER
                if m != "ours_memory" and m in method_payloads
            }
            if not baseline_scores:
                continue
            gen_score = baseline_scores.get("gen_only_session_incremental")
            max_baseline = max(safe_float(v["f1"]) for v in baseline_scores.values())
            best_baseline = max(baseline_scores.items(), key=lambda item: safe_float(item[1]["f1"]))[0]
            row_common = {
                "turn": turn,
                "difficulty": difficulty,
                "inference_model": inference_model,
                "example_id": method_payloads["ours_memory"]["example"].get("example_id", ""),
                "example_id_sub": cls["example_id_sub"],
                "task_key": task_key,
                "slice_target_slot_conflict": cls["target_slot_conflict"],
                "slice_target_pref_slot_conflict": cls["target_pref_slot_conflict"],
                "slice_any_pref_slot_conflict": cls["any_pref_slot_conflict"],
                "slice_evolving_pref": cls["evolving_pref"],
                "slice_pref_conflict_or_evolving": cls["pref_conflict_or_evolving"],
                "slice_correction_language": cls["correction_language"],
                "ours_f1": ours_score["f1"],
                "gen_only_f1": gen_score["f1"] if gen_score else "",
                "max_baseline_f1": max_baseline,
                "best_baseline": best_baseline,
                "delta_vs_gen": (ours_score["f1"] - gen_score["f1"]) if gen_score else "",
                "delta_vs_best_baseline": ours_score["f1"] - max_baseline,
            }
            if gen_score and ours_score["f1"] > gen_score["f1"]:
                win_over_gen_cases.append(row_common)
            has_all_baselines = all(
                m in method_payloads for m in METHOD_ORDER if m != "ours_memory"
            )
            if has_all_baselines and ours_score["f1"] > max_baseline:
                ex = method_payloads["ours_memory"]["example"]
                case = {
                    **row_common,
                    "reference_ground_truth": ex.get("reference_ground_truth"),
                    "user_utterance": compact_output(ex.get("user_utterance"), 1200),
                    "target_conflict_slots": cls["target_conflict_slots"],
                    "target_pref_conflict_slots": cls["target_pref_conflict_slots"],
                    "any_pref_conflict_slots": cls["any_pref_conflict_slots"],
                    "pref_evidence": cls["pref_evidence"],
                    "outputs": {
                        m: compact_output(method_payloads[m]["example"].get("llm_output"), 1200)
                        for m in METHOD_ORDER
                        if m in method_payloads
                    },
                    "scores": {
                        m: method_payloads[m]["score"]
                        for m in METHOD_ORDER
                        if m in method_payloads
                    },
                }
                win_cases.append(case)

    win_cases.sort(key=lambda r: (safe_float(r["delta_vs_best_baseline"]), safe_float(r["delta_vs_gen"])), reverse=True)
    win_over_gen_cases.sort(key=lambda r: safe_float(r["delta_vs_gen"]), reverse=True)

    manifest_rows = []
    for manifest_key, cls in sorted(manifest_by_example.items()):
        manifest_rows.append(
            {
                "manifest_key": manifest_key,
                "turn": cls["turn"],
                "difficulty": cls["difficulty"],
                "task_key": cls["task_key"],
                "example_id_sub": cls["example_id_sub"],
                "example_id": cls["example_id"],
                "n_sessions": cls["n_sessions"],
                "target_slot_conflict": cls["target_slot_conflict"],
                "any_slot_conflict": cls["any_slot_conflict"],
                "target_pref_slot_conflict": cls["target_pref_slot_conflict"],
                "any_pref_slot_conflict": cls["any_pref_slot_conflict"],
                "evolving_pref": cls["evolving_pref"],
                "pref_conflict_or_evolving": cls["pref_conflict_or_evolving"],
                "correction_language": cls["correction_language"],
                "temporal_or_evolving": cls["temporal_or_evolving"],
                "target_conflict_slots": json.dumps(cls["target_conflict_slots"], ensure_ascii=False),
                "target_pref_conflict_slots": json.dumps(cls["target_pref_conflict_slots"], ensure_ascii=False),
                "any_pref_conflict_slots": json.dumps(cls["any_pref_conflict_slots"], ensure_ascii=False),
                "pref_slots": ";".join(cls["pref_evidence"]["pref_slots"]),
                "pref_dialogue_ids": ";".join(cls["pref_evidence"]["dialogue_ids"]),
                "correction_hits": ";".join(cls["correction_hits"]),
            }
        )

    summary_fields = [
        "method",
        "turn",
        "difficulty",
        "inference_model",
        "slice",
        "n_examples",
        "precision",
        "recall",
        "f1",
        "pref_em",
        "pref_f1",
        "nonpref_f1",
        "overall_tp",
        "overall_fp",
        "overall_fn",
    ]
    win_fields = [
        "turn",
        "difficulty",
        "inference_model",
        "example_id",
        "example_id_sub",
        "task_key",
        "slice_target_slot_conflict",
        "slice_target_pref_slot_conflict",
        "slice_any_pref_slot_conflict",
        "slice_evolving_pref",
        "slice_pref_conflict_or_evolving",
        "slice_correction_language",
        "ours_f1",
        "gen_only_f1",
        "max_baseline_f1",
        "best_baseline",
        "delta_vs_gen",
        "delta_vs_best_baseline",
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "slice_manifest.csv", manifest_rows, list(manifest_rows[0].keys()) if manifest_rows else [])
    write_csv(args.out_dir / "summary_by_file_slice.csv", summary_rows, summary_fields)
    write_csv(args.out_dir / "summary_by_method_slice.csv", overall_rows, summary_fields)
    write_csv(args.out_dir / "win_cases_ours_over_all_baselines.csv", win_cases, win_fields)
    write_csv(args.out_dir / "win_cases_ours_over_gen_only.csv", win_over_gen_cases, win_fields)
    write_json(args.out_dir / "win_cases_ours_over_all_baselines.full.json", win_cases)
    write_json(args.out_dir / "win_cases_ours_over_all_baselines.top20.json", win_cases[:20])
    write_json(
        args.out_dir / "win_cases_target_slot_conflict.top20.json",
        [c for c in win_cases if c["slice_target_slot_conflict"]][:20],
    )
    write_json(
        args.out_dir / "win_cases_target_pref_slot_conflict.top20.json",
        [c for c in win_cases if c["slice_target_pref_slot_conflict"]][:20],
    )
    write_json(
        args.out_dir / "win_cases_pref_conflict_or_evolving.top20.json",
        [c for c in win_cases if c["slice_pref_conflict_or_evolving"]][:20],
    )
    write_json(args.out_dir / "missing_paths.json", missing_paths)

    report = {
        "input_per_file": str(args.per_file),
        "turn": args.turn,
        "n_score_rows": len(score_rows),
        "n_unique_examples": len(manifest_by_example),
        "n_missing_paths": len(missing_paths),
        "n_temporal_or_evolving_unique_examples": sum(
            1 for cls in manifest_by_example.values() if cls["temporal_or_evolving"]
        ),
        "n_target_slot_conflict_unique_examples": sum(
            1 for cls in manifest_by_example.values() if cls["target_slot_conflict"]
        ),
        "n_target_pref_slot_conflict_unique_examples": sum(
            1 for cls in manifest_by_example.values() if cls["target_pref_slot_conflict"]
        ),
        "n_any_pref_slot_conflict_unique_examples": sum(
            1 for cls in manifest_by_example.values() if cls["any_pref_slot_conflict"]
        ),
        "n_evolving_pref_unique_examples": sum(
            1 for cls in manifest_by_example.values() if cls["evolving_pref"]
        ),
        "n_pref_conflict_or_evolving_unique_examples": sum(
            1 for cls in manifest_by_example.values() if cls["pref_conflict_or_evolving"]
        ),
        "n_ours_over_all_baselines_cases": len(win_cases),
        "n_ours_over_gen_only_cases": len(win_over_gen_cases),
        "outputs": {
            "slice_manifest": str(args.out_dir / "slice_manifest.csv"),
            "summary_by_file_slice": str(args.out_dir / "summary_by_file_slice.csv"),
            "summary_by_method_slice": str(args.out_dir / "summary_by_method_slice.csv"),
            "win_cases_ours_over_all_baselines": str(args.out_dir / "win_cases_ours_over_all_baselines.csv"),
            "win_cases_ours_over_gen_only": str(args.out_dir / "win_cases_ours_over_gen_only.csv"),
            "full_json": str(args.out_dir / "win_cases_ours_over_all_baselines.full.json"),
            "top20_json": str(args.out_dir / "win_cases_ours_over_all_baselines.top20.json"),
            "target_slot_conflict_top20_json": str(args.out_dir / "win_cases_target_slot_conflict.top20.json"),
            "target_pref_slot_conflict_top20_json": str(args.out_dir / "win_cases_target_pref_slot_conflict.top20.json"),
            "pref_conflict_or_evolving_top20_json": str(args.out_dir / "win_cases_pref_conflict_or_evolving.top20.json"),
        },
    }
    write_json(args.out_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
