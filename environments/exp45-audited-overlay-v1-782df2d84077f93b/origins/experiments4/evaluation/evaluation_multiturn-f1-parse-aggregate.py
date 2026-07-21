#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
import re
import dateparser  # 날짜/시간 파싱 라이브러리
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple, Union, Optional
from collections import defaultdict

# =========================================================
# Robust Parsing (Func(...), {Func}(...), JSON, code fences)
# =========================================================

_CALL_RE = re.compile(r"([A-Za-z_]\w*)\s*\((.*?)\)")
_BRACED_FUNC_RE = re.compile(r"\{([A-Za-z_]\w*)\}\s*\(")  # {GetHotels}
_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)

def _strip_code_fences(s: str) -> str:
    return _CODE_FENCE_RE.sub("", s.strip())

def _normalize_braced_func(s: str) -> str:
    return _BRACED_FUNC_RE.sub(r"\1(", s)

def _strip_think_tags(s: str) -> str:
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"</think>", "", s, flags=re.IGNORECASE)
    return s

def _remove_code_fences_keep_content(s: str) -> str:
    s = re.sub(r"```(?:json)?", "", s, flags=re.IGNORECASE)
    s = s.replace("```", "")
    return s

def _build_call_string(func: str, args: Any) -> str:
    if not isinstance(args, dict) or not args:
        return f"{func}()"
    parts = []
    for k, v in args.items():
        if v is None:
            continue
        parts.append(f'{k}="{str(v)}"')
    return f"{func}({', '.join(parts)})"

def _json_to_calls(obj: Any) -> List[str]:
    """
    JSON 객체를 함수 호출 문자열 리스트로 변환
    [UPDATED] 'domain' 키가 평면(Flat) 구조로 있거나 대소문자가 섞여 있어도 처리
    """
    calls: List[str] = []

    if isinstance(obj, list):
        for item in obj:
            calls.extend(_json_to_calls(item))
        return calls

    if isinstance(obj, dict):
        # 1. "calls" 리스트가 내부에 있는 경우
        if "calls" in obj and isinstance(obj["calls"], list):
            for c in obj["calls"]:
                calls.extend(_json_to_calls(c))
            return calls

        # --- 키 탐색 (대소문자 무시) ---
        keys_lower = {k.lower(): k for k in obj.keys()}
        
        # 함수 이름을 나타내는 키 후보
        func_key_candidates = ["name", "tool", "function", "domain"]
        found_func_key = None
        for cand in func_key_candidates:
            if cand in keys_lower:
                found_func_key = keys_lower[cand]
                break
        
        # 파라미터를 담고 있는 키 후보
        arg_key_candidates = ["arguments", "args", "parameters"]
        found_arg_key = None
        for cand in arg_key_candidates:
            if cand in keys_lower:
                found_arg_key = keys_lower[cand]
                break

        # STRATEGY A: 명시적 중첩 구조 (domain + parameters)
        # 예: {"domain": "GetHotels", "parameters": {...}}
        if found_func_key and found_arg_key:
            name = obj[found_func_key]
            args = obj[found_arg_key]
            if isinstance(name, str):
                calls.append(_build_call_string(name, args if isinstance(args, dict) else {}))
            return calls

        # STRATEGY B: 평면 구조 (domain + 나머지 키들)
        # 예: {"domain": "GetRestaurants", "price_range": "moderate"}
        # 예: {"Domain": "GetHotels"} (인자 없음)
        if found_func_key and not found_arg_key:
            name = obj[found_func_key]
            if isinstance(name, str):
                # domain 키와 메타데이터 키를 제외한 나머지를 모두 인자로 간주
                flat_args = {}
                skip_keys = {
                    "reasoning", "raw_content", "model_name", "evaluation_result", "reasoning_tokens",
                    found_func_key, # "domain" 키 자체 제외
                    "domain", "name", "tool", "function" # 혹시 모를 중복 키 제외
                }
                for k, v in obj.items():
                    if k in skip_keys or k.lower() in skip_keys:
                        continue
                    flat_args[k] = v
                calls.append(_build_call_string(name, flat_args))
            return calls

        # STRATEGY C: Key-as-Function ({"GetHotels": {"location": "Seoul"}})
        for k, v in obj.items():
            if not isinstance(k, str):
                continue
            
            # 메타데이터나 함수 식별자 키는 건너뜀 (대소문자 무시)
            k_low = k.lower()
            if k_low in set(func_key_candidates + arg_key_candidates + 
                            ["reasoning", "raw_content", "model_name", "evaluation_result", "reasoning_tokens"]):
                continue

            if isinstance(v, dict):
                calls.append(_build_call_string(k, v))
            elif v is None:
                calls.append(f"{k}()")
            elif isinstance(v, list):
                # 리스트 내부가 dict면 각각 호출, 아니면 값으로 처리
                is_list_of_dicts = all(isinstance(i, dict) for i in v)
                if is_list_of_dicts and v: 
                    for item in v:
                        calls.append(_build_call_string(k, item))
                else:
                    calls.append(_build_call_string(k, {"value": v}))
            else:
                calls.append(_build_call_string(k, {"value": v}))
        return calls

    return calls

def _try_parse_json_to_calls(s: str) -> List[str]:
    try:
        obj = json.loads(s)
    except Exception:
        return []
    return _json_to_calls(obj)

def extract_calls(x: Union[str, List[str], None]) -> List[str]:
    if x is None:
        return []

    if isinstance(x, list):
        out: List[str] = []
        for item in x:
            if isinstance(item, str) and item.strip():
                out.extend(extract_calls(item))
        return out

    if not isinstance(x, str):
        return []

    s = x.strip()
    if not s:
        return []

    s = _strip_think_tags(s)
    s = _remove_code_fences_keep_content(s)
    s = _strip_code_fences(s)
    s = _normalize_braced_func(s)

    json_calls = _try_parse_json_to_calls(s)
    if json_calls:
        return json_calls

    calls = [m.group(0) for m in _CALL_RE.finditer(s)]
    if not calls and _CALL_RE.fullmatch(s):
        calls = [s]
    return calls

# =========================================================
# Slot/Value parsing inside Func(...)
# [UPDATED] Added Date/Time Normalization Logic
# =========================================================

def _process_value(val_str: str) -> str:
    """
    따옴표 제거 및 Boolean(true/false) 정규화 처리
    """
    s = val_str.strip()
    
    # 1. 따옴표가 있다면 -> 내용만 추출
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        return s[1:-1]
    
    # 2. 따옴표가 없다면 -> Boolean 정규화 (true -> True, false -> False)
    if s.lower() == "true":
        return "True"
    if s.lower() == "false":
        return "False"
        
    return s

def _normalize_date_time_value(slot: str, value: str) -> str:
    """
    슬롯 이름에 따라 날짜/시간 값을 정규화합니다.
    1. Time: 24시간제 (HH:MM)로 변환
    2. Date: 연도를 제외한 월-일 (MM-DD)로 변환
    """
    slot_lower = slot.lower()
    
    # 파싱 설정을 통해 미래 우선, 비엄격 모드 적용
    settings = {
        'PREFER_DATES_FROM': 'future', 
        'STRICT_PARSING': False
    }

    try:
        # 1. 시간 (Time) 처리
        if "time" in slot_lower:
            # 값에 숫자가 없으면(예: "evening", "now") 파싱하지 않고 원본 반환 (오작동 방지)
            if not any(char.isdigit() for char in value):
                return value
                
            dt = dateparser.parse(value, settings=settings)
            if dt:
                return dt.strftime("%H:%M") # 예: 18:15

        # 2. 날짜 (Date) 처리 (date, day 등이 포함된 슬롯)
        elif "date" in slot_lower or "day" in slot_lower:
            dt = dateparser.parse(value, settings=settings)
            if dt:
                # [요청사항 반영] 연도는 제외하고 월-일만 반환
                return dt.strftime("%m-%d") # 예: 03-05
                
    except Exception:
        # 파싱 실패 시 조용히 원본 값 반환
        return value

    return value

def _split_args(arg_str: str) -> List[str]:
    parts, buf = [], []
    in_quote: Optional[str] = None
    escape = False
    for ch in arg_str:
        if escape:
            buf.append(ch); escape = False; continue
        if ch == "\\":
            buf.append(ch); escape = True; continue
        if in_quote:
            buf.append(ch)
            if ch == in_quote:
                in_quote = None
            continue
        if ch in ("'", '"'):
            in_quote = ch
            buf.append(ch)
            continue
        if ch == ",":
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts

def parse_call_to_slotvals(call: str) -> List[Tuple[str, str, str]]:
    call = call.strip()
    m = _CALL_RE.fullmatch(call)
    if not m:
        return []
    domain = m.group(1).strip()
    args_str = m.group(2).strip()
    if not args_str:
        return []
    out: List[Tuple[str, str, str]] = []
    for part in _split_args(args_str):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        slot = k.strip()
        
        # 1. 기존 처리 (따옴표 제거, Bool 처리)
        val = _process_value(v.strip())
        
        # 2. [NEW] 날짜/시간 정규화 적용
        val = _normalize_date_time_value(slot, val)
        
        out.append((domain, slot, val))
    return out

# =========================================================
# Data classes
# =========================================================

@dataclass
class PRF:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int

@dataclass
class EMR:
    em: float
    em_count: int
    n: int

# =========================================================
# Core metrics: F1 over (domain,slot) AND; value OR within slot
# =========================================================

def prf_from_counts(tp: int, fp: int, fn: int) -> PRF:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return PRF(p, r, f1, tp, fp, fn)

def build_gt_allowed_map(gt_field: Any) -> Dict[Tuple[str, str], Set[str]]:
    allowed = defaultdict(set)
    for call in extract_calls(gt_field):
        for d, s, v in parse_call_to_slotvals(call):
            allowed[(d, s)].add(v)
    return dict(allowed)

def build_pred_map(pred_field: Any) -> Dict[Tuple[str, str], Set[str]]:
    pred = defaultdict(set)
    for call in extract_calls(pred_field):
        for d, s, v in parse_call_to_slotvals(call):
            pred[(d, s)].add(v)
    return dict(pred)

def counts_slot_and_value_or(
    gt_allowed: Dict[Tuple[str, str], Set[str]],
    pred_vals: Dict[Tuple[str, str], Set[str]],
) -> Tuple[int, int, int]:
    tp = fp = fn = 0

    # TP/FN over GT slots
    for key, allowed_vals in gt_allowed.items():
        pv = pred_vals.get(key, set())
        if pv and (pv & allowed_vals):
            tp += 1
        else:
            fn += 1

    # FP over predicted slots
    for key, pv in pred_vals.items():
        if key not in gt_allowed:
            fp += 1
        else:
            allowed_vals = gt_allowed[key]
            if not (pv & allowed_vals):
                fp += 1

    return tp, fp, fn

def micro_f1_slot_and_value_or(
    examples: List[Dict[str, Any]],
    gt_key: str = "reference_ground_truth",
    pred_key: str = "llm_output",
) -> PRF:
    TP = FP = FN = 0
    for ex in examples:
        gt_allowed = build_gt_allowed_map(ex.get(gt_key))
        pred_vals = build_pred_map(ex.get(pred_key))
        tp, fp, fn = counts_slot_and_value_or(gt_allowed, pred_vals)
        TP += tp; FP += fp; FN += fn
    return prf_from_counts(TP, FP, FN)

# =========================================================
# Pref/non-pref filtering helpers
# =========================================================

def load_pref_list(path: str) -> Dict[str, Set[str]]:
    """
    pref_list.json format:
    {
      "GetHotels": ["average_star", ...],
      ...
    }
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        out: Dict[str, Set[str]] = {}
        if isinstance(obj, dict):
            for d, slots in obj.items():
                if isinstance(d, str) and isinstance(slots, list):
                    out[d] = set(str(s) for s in slots)
        return out
    except Exception:
        # 파일이 없거나 에러가 나면 빈 딕셔너리 반환하여 중단 방지
        return {}

def is_pref_slot(domain: str, slot: str, pref_map: Dict[str, Set[str]]) -> bool:
    return (domain in pref_map) and (slot in pref_map[domain])

def filter_map_by_pref(
    slotval_map: Dict[Tuple[str, str], Set[str]],
    pref_map: Dict[str, Set[str]],
    want_pref: bool
) -> Dict[Tuple[str, str], Set[str]]:
    """
    want_pref=True  -> keep only pref slots
    want_pref=False -> keep only non-pref slots
    """
    out: Dict[Tuple[str, str], Set[str]] = {}
    for (d, s), vals in slotval_map.items():
        flag = is_pref_slot(d, s, pref_map)
        if (want_pref and flag) or ((not want_pref) and (not flag)):
            out[(d, s)] = set(vals)
    return out

# =========================================================
# Exact Match for pref slots ONLY
# (Strict: pref slots must match, and NO extra pref slots predicted)
# Value rule: OR allowed if GT has multiple values for same (d,s)
# =========================================================

def pref_em_one(
    ex: Dict[str, Any],
    pref_map: Dict[str, Set[str]],
    gt_key: str = "reference_ground_truth",
    pred_key: str = "llm_output",
) -> int:
    gt_allowed_all = build_gt_allowed_map(ex.get(gt_key))
    pred_vals_all = build_pred_map(ex.get(pred_key))

    gt_allowed_pref = filter_map_by_pref(gt_allowed_all, pref_map, want_pref=True)
    pred_vals_pref  = filter_map_by_pref(pred_vals_all, pref_map, want_pref=True)

    tp, fp, fn = counts_slot_and_value_or(gt_allowed_pref, pred_vals_pref)
    # EM=1 iff pref slice has no FP/FN
    return 1 if (fp == 0 and fn == 0) else 0

def pref_em_rate(
    examples: List[Dict[str, Any]],
    pref_map: Dict[str, Set[str]],
    gt_key: str = "reference_ground_truth",
    pred_key: str = "llm_output",
) -> EMR:
    n = len(examples)
    if n == 0:
        return EMR(0.0, 0, 0)
    em_count = 0
    for ex in examples:
        em_count += pref_em_one(ex, pref_map, gt_key=gt_key, pred_key=pred_key)
    return EMR(em_count / n, em_count, n)

# =========================================================
# OR-case logging
# =========================================================

def format_or_case_block(
    ex: Dict[str, Any],
    gt_allowed: Dict[Tuple[str, str], Set[str]],
    pred_vals: Dict[Tuple[str, str], Set[str]],
    tp: int, fp: int, fn: int
) -> str:
    or_slots = {k: v for k, v in gt_allowed.items() if len(v) >= 2}
    if not or_slots:
        return ""

    ex_id = ex.get("example_id_sub") or ex.get("example_id") or "NA"
    utt = ex.get("user_utterance", "")
    gt_raw = ex.get("reference_ground_truth", None)
    pred_raw = ex.get("llm_output", None)
    m = prf_from_counts(tp, fp, fn)

    lines = []
    lines.append("=" * 80)
    lines.append(f"EXAMPLE: {ex_id}")
    if utt:
        lines.append(f"USER_UTTERANCE: {utt}")
    lines.append(f"GT_RAW: {gt_raw}")
    lines.append(f"PRED_RAW: {pred_raw}")
    lines.append("")
    lines.append("OR SLOTS (same domain+slot, multiple acceptable values):")
    for (d, s), allowed_vals in sorted(or_slots.items(), key=lambda x: (x[0][0], x[0][1])):
        pv = pred_vals.get((d, s), set())
        hit = bool(pv & allowed_vals)
        lines.append(f"- ({d}, {s}) allowed={sorted(list(allowed_vals))}")
        lines.append(f"  pred={sorted(list(pv)) if pv else []}  -> {'HIT' if hit else 'MISS'}")

    lines.append("")
    lines.append(f"COUNTS: TP={tp} FP={fp} FN={fn}")
    lines.append(f"PRF: P={m.precision:.4f} R={m.recall:.4f} F1={m.f1:.4f}")
    lines.append("")
    return "\n".join(lines)

def write_or_case_log(
    examples: List[Dict[str, Any]],
    out_path: str,
    gt_key: str = "reference_ground_truth",
    pred_key: str = "llm_output",
) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        wrote_any = False
        for ex in examples:
            gt_allowed = build_gt_allowed_map(ex.get(gt_key))
            pred_vals = build_pred_map(ex.get(pred_key))
            tp, fp, fn = counts_slot_and_value_or(gt_allowed, pred_vals)
            block = format_or_case_block(ex, gt_allowed, pred_vals, tp, fp, fn)
            if block:
                f.write(block)
                f.write("\n")
                wrote_any = True

        if not wrote_any:
            f.write("No OR cases found (no (domain,slot) with >=2 acceptable values).\n")

# =========================================================
# Parsing-failure logging
# =========================================================

def is_parsing_failed_pred(ex: Dict[str, Any], pred_key: str = "llm_output") -> Tuple[bool, str]:
    pred_raw = ex.get(pred_key)

    calls = extract_calls(pred_raw)
    if not calls:
        return True, "no_calls_extracted"

    pred_map = build_pred_map(pred_raw)
    # calls가 있는데 map이 비어있다면, 0-arg 함수(예: "GetCars()")일 수 있음.
    # 이는 구조적 실패가 아닐 수 있으므로, calls만 추출되면 OK로 간주.
    # 하지만 더 엄격하게 보려면 여기를 수정 가능. 현재는 OK.
    
    return False, "ok"

def write_parsing_failures(
    examples: List[Dict[str, Any]],
    out_path: str,
    pred_key: str = "llm_output",
) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        n_fail = 0
        for ex in examples:
            failed, reason = is_parsing_failed_pred(ex, pred_key=pred_key)
            if failed:
                ex_id = ex.get("example_id_sub") or ex.get("example_id") or "NA"
                
                # Raw output 안전하게 문자열 변환
                raw_val = ex.get(pred_key, "")
                safe_raw = json.dumps(str(raw_val), ensure_ascii=False)

                f.write(f"{ex_id}\t{reason}\t{safe_raw}\n")
                n_fail += 1
        f.write(f"\n# total_failures={n_fail} / total_examples={len(examples)}\n")

# =========================================================
# JSON loading helpers
# =========================================================

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_examples(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ["data", "examples", "items"]:
            if key in data and isinstance(data[key], list):
                return data[key]
    raise ValueError("Unsupported JSON structure: expected list or dict with data/examples/items list.")

# =========================================================
# Evaluate one file -> 3 metrics
#   1) overall F1 (all slots)
#   2) pref-slot EM (pref_list filter)
#   3) non-pref F1 (complement filter)
# =========================================================

def micro_f1_with_filter(
    examples: List[Dict[str, Any]],
    pref_map: Dict[str, Set[str]],
    want_pref: bool,
    gt_key: str = "reference_ground_truth",
    pred_key: str = "llm_output",
) -> PRF:
    TP = FP = FN = 0
    for ex in examples:
        gt_allowed_all = build_gt_allowed_map(ex.get(gt_key))
        pred_vals_all = build_pred_map(ex.get(pred_key))
        gt_allowed = filter_map_by_pref(gt_allowed_all, pref_map, want_pref=want_pref)
        pred_vals  = filter_map_by_pref(pred_vals_all, pref_map, want_pref=want_pref)
        tp, fp, fn = counts_slot_and_value_or(gt_allowed, pred_vals)
        TP += tp; FP += fp; FN += fn
    return prf_from_counts(TP, FP, FN)

def eval_one_file(
    json_path: str,
    pref_map: Dict[str, Set[str]],
    or_log_path: Optional[str] = None,
    parsing_fail_path: Optional[str] = None,
) -> Tuple[PRF, EMR, PRF]:
    data = load_json(json_path)
    examples = extract_examples(data)

    overall_prf = micro_f1_slot_and_value_or(examples)                  # (1) all slots
    pref_emr    = pref_em_rate(examples, pref_map)                      # (2) pref EM
    nonpref_prf = micro_f1_with_filter(examples, pref_map, want_pref=False)  # (3) non-pref slots

    if or_log_path:
        os.makedirs(os.path.dirname(or_log_path), exist_ok=True)
        write_or_case_log(examples, or_log_path)

    if parsing_fail_path:
        os.makedirs(os.path.dirname(parsing_fail_path), exist_ok=True)
        write_parsing_failures(examples, parsing_fail_path)

    return overall_prf, pref_emr, nonpref_prf

# =========================================================
# Evaluate all json under root_dir and save CSV
# Folder schema:
#   root_dir/context/pref_type/query_turns/model_name/prompt_type/file.json
# =========================================================

def iter_json_files(root_dir: str) -> List[str]:
    out = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith(".json"):
                out.append(os.path.join(dirpath, fn))
    out.sort()
    return out

def ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def parse_hparams_from_relpath(rel_path: str) -> Dict[str, str]:
    parts = rel_path.split(os.sep)
    if len(parts) >= 6:
        context, pref_type, query_turns, model_name, prompt_type, file_name = parts[-6:]
    else:
        padded = [""] * (6 - len(parts)) + parts
        context, pref_type, query_turns, model_name, prompt_type, file_name = padded[-6:]
    return {
        "context": context,
        "pref_type": pref_type,
        "query_turns": query_turns,
        "model_name": model_name,
        "prompt_type": prompt_type,
        "file_name": file_name,
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--pref_list_path", type=str, default="/data/minseo/experiments4/pref_list.json",
                        help="Path to pref_list.json")

    parser.add_argument("--root_dir", type=str, default=None,
                        help="Recursively evaluate all .json files under this directory.")
    parser.add_argument("--json_path", type=str, default=None,
                        help="Evaluate a single json file (ignored if --root_dir is set).")

    parser.add_argument("--out_csv", type=str, required=True,
                        help="Output CSV path (per-file metrics).")

    parser.add_argument("--save_or_logs", action="store_true",
                        help="Save OR-case logs per json file.")
    parser.add_argument("--save_parsing_failures", action="store_true",
                        help="Save parsing-failure example_id_sub list per json file.")

    parser.add_argument("--logs_dir", type=str, default=None,
                        help="Directory to place logs. Default: alongside out_csv in <out_csv_dir>/logs")

    args = parser.parse_args()

    pref_map = load_pref_list(args.pref_list_path)

    if args.root_dir:
        targets = iter_json_files(args.root_dir)
        if not targets:
            raise FileNotFoundError(f"No .json files found under root_dir={args.root_dir}")
    elif args.json_path:
        targets = [args.json_path]
    else:
        raise ValueError("Provide either --root_dir or --json_path")

    out_csv = args.out_csv
    ensure_parent_dir(out_csv)

    logs_dir = args.logs_dir or os.path.join(os.path.dirname(out_csv) or ".", "logs")
    os.makedirs(logs_dir, exist_ok=True)

    rows = []
    for jp in targets:
        if args.root_dir:
            rel = os.path.relpath(jp, args.root_dir)
            h = parse_hparams_from_relpath(rel)
            group_key = os.path.join(h["context"], h["pref_type"], h["query_turns"], h["model_name"], h["prompt_type"])
        else:
            rel = os.path.basename(jp)
            h = {"context": "", "pref_type": "", "query_turns": "", "model_name": "", "prompt_type": "", "file_name": os.path.basename(jp)}
            group_key = ""

        stem = rel.replace(os.sep, "__")

        or_log_path = None
        parsing_fail_path = None
        if args.save_or_logs:
            or_log_path = os.path.join(logs_dir, f"{stem}.or_cases.txt")
        if args.save_parsing_failures:
            parsing_fail_path = os.path.join(logs_dir, f"{stem}.parsing_failures.txt")

        try:
            overall_prf, pref_emr, nonpref_prf = eval_one_file(
                jp, pref_map,
                or_log_path=or_log_path,
                parsing_fail_path=parsing_fail_path
            )
            status = "ok"
            err = ""
        except Exception as e:
            overall_prf = PRF(0.0, 0.0, 0.0, 0, 0, 0)
            pref_emr    = EMR(0.0, 0, 0)
            nonpref_prf = PRF(0.0, 0.0, 0.0, 0, 0, 0)
            status = "error"
            err = repr(e)

        rows.append({
            "group_key": group_key,
            "context": h["context"],
            "pref_type": h["pref_type"],
            "query_turns": h["query_turns"],
            "model_name": h["model_name"],
            "prompt_type": h["prompt_type"],
            "file_name": h["file_name"],
            "json_path": jp,
            "rel_path": rel,
            "status": status,
            "error": err,

            # (1) Overall F1
            "overall_tp": overall_prf.tp,
            "overall_fp": overall_prf.fp,
            "overall_fn": overall_prf.fn,
            "overall_precision": overall_prf.precision,
            "overall_recall": overall_prf.recall,
            "overall_f1": overall_prf.f1,

            # (2) Pref-slot EM
            "pref_em": pref_emr.em,
            "pref_em_count": pref_emr.em_count,
            "n_examples": pref_emr.n,

            # (3) Non-pref F1
            "nonpref_tp": nonpref_prf.tp,
            "nonpref_fp": nonpref_prf.fp,
            "nonpref_fn": nonpref_prf.fn,
            "nonpref_precision": nonpref_prf.precision,
            "nonpref_recall": nonpref_prf.recall,
            "nonpref_f1": nonpref_prf.f1,

            "or_log_path": or_log_path or "",
            "parsing_fail_path": parsing_fail_path or "",
        })

    fieldnames = [
        "group_key",
        "context", "pref_type", "query_turns", "model_name", "prompt_type", "file_name",
        "json_path", "rel_path", "status", "error",

        # overall F1
        "overall_tp", "overall_fp", "overall_fn",
        "overall_precision", "overall_recall", "overall_f1",

        # pref EM
        "pref_em", "pref_em_count", "n_examples",

        # non-pref F1
        "nonpref_tp", "nonpref_fp", "nonpref_fn",
        "nonpref_precision", "nonpref_recall", "nonpref_f1",

        "or_log_path", "parsing_fail_path",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    ok_rows = [r for r in rows if r["status"] == "ok"]
    if ok_rows:
        # aggregate overall F1 (sum counts)
        TP = sum(int(r["overall_tp"]) for r in ok_rows)
        FP = sum(int(r["overall_fp"]) for r in ok_rows)
        FN = sum(int(r["overall_fn"]) for r in ok_rows)
        agg_overall = prf_from_counts(TP, FP, FN)

        # aggregate non-pref F1 (sum counts)
        NTP = sum(int(r["nonpref_tp"]) for r in ok_rows)
        NFP = sum(int(r["nonpref_fp"]) for r in ok_rows)
        NFN = sum(int(r["nonpref_fn"]) for r in ok_rows)
        agg_nonpref = prf_from_counts(NTP, NFP, NFN)

        # aggregate pref EM
        EM_count = sum(int(r["pref_em_count"]) for r in ok_rows)
        N_total = sum(int(r["n_examples"]) for r in ok_rows)
        agg_pref_em = (EM_count / N_total) if N_total > 0 else 0.0

        print(f"[DONE] files={len(rows)} ok={len(ok_rows)} error={len(rows)-len(ok_rows)}")
        print(f"[AGG overall F1] TP={agg_overall.tp} FP={agg_overall.fp} FN={agg_overall.fn}  "
              f"P={agg_overall.precision:.4f} R={agg_overall.recall:.4f} F1={agg_overall.f1:.4f}")
        print(f"[AGG pref EM] em={agg_pref_em:.4f} (em_count={EM_count} / n={N_total})")
        print(f"[AGG non-pref F1] TP={agg_nonpref.tp} FP={agg_nonpref.fp} FN={agg_nonpref.fn}  "
              f"P={agg_nonpref.precision:.4f} R={agg_nonpref.recall:.4f} F1={agg_nonpref.f1:.4f}")
        print(f"[CSV] {out_csv}")
        print(f"[LOGS_DIR] {logs_dir}")
    else:
        print(f"[DONE] files={len(rows)} ok=0 error={len(rows)}")
        print(f"[CSV] {out_csv}")
        print(f"[LOGS_DIR] {logs_dir}")

if __name__ == "__main__":
    main()