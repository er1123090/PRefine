import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple, Union, Optional
from collections import defaultdict
import argparse

# -----------------------------
# Parsing Logic (Updated)
# -----------------------------
_CALL_RE = re.compile(r"(?:\{?)([A-Za-z_]\w*)(?:\}?)\s*\((.*?)\)")

def _strip_quotes(s: str) -> str:
    """단순 따옴표 제거 (JSON 파싱 등에서 사용)"""
    s = str(s).strip()
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        return s[1:-1]
    return s

def _process_regex_value(val_str: str) -> str:
    """
    Regex 파싱 전용 값 처리 함수
    1. 따옴표가 감싸져 있다면 -> 내용만 추출 (String Literal)
    2. 따옴표가 없다면 -> Boolean(true/false) 정규화 또는 그대로 반환
    """
    s = val_str.strip()
    
    # 1. Quoted String: 따옴표 제거 후 내용 반환 (Boolean 변환 안 함)
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        return s[1:-1]
    
    # 2. Unquoted Literal: Boolean 정규화 (true -> True, false -> False)
    # 이는 Python의 str(True) == "True" 동작과 일치시키기 위함입니다.
    if s.lower() == "true":
        return "True"
    if s.lower() == "false":
        return "False"
        
    return s

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

def parse_regex_call(call: str) -> List[Tuple[str, str, str]]:
    """정규표현식 기반 파싱 (개선됨: Unquoted Boolean 지원)"""
    call = call.strip()
    m = _CALL_RE.fullmatch(call)
    if not m:
        return []
    domain = m.group(1).strip()
    args_str = m.group(2).strip()
    if not args_str:
        return []
    out = []
    for part in _split_args(args_str):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        slot = k.strip()
        # 변경: _strip_quotes 대신 _process_regex_value 사용
        val = _process_regex_value(v)
        out.append((domain, slot, val))
    return out

def extract_from_json(text: str) -> List[Tuple[str, str, str]]:
    """다양한 JSON 포맷 파싱 지원"""
    # 1. 마크다운 코드 블록 제거 및 클리닝
    cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    
    # 2. JSON 파싱 시도
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find('{')
        end = cleaned.rfind('}')
        if start != -1 and end != -1:
            try:
                data = json.loads(cleaned[start:end+1])
            except json.JSONDecodeError:
                return []
        else:
            return []

    results = []
    
    if isinstance(data, dict):
        items = [data]
    elif isinstance(data, list):
        items = data
    else:
        return []

    for item in items:
        if not isinstance(item, dict):
            continue
        
        matched_strategy = False

        # Strategy 1: OpenAI Function Call Format
        if "function" in item and isinstance(item["function"], dict):
            f = item["function"]
            if "name" in f and "parameters" in f and isinstance(f["parameters"], dict):
                domain = f["name"]
                params = f["parameters"]
                for slot, val in params.items():
                    val_str = str(val) if not isinstance(val, (list, dict)) else json.dumps(val)
                    results.append((str(domain), str(slot), _strip_quotes(val_str)))
                matched_strategy = True
        
        # Strategy 2: Flat Format
        if not matched_strategy:
            if "name" in item and "parameters" in item and isinstance(item["parameters"], dict):
                domain = item["name"]
                params = item["parameters"]
                for slot, val in params.items():
                    val_str = str(val) if not isinstance(val, (list, dict)) else json.dumps(val)
                    results.append((str(domain), str(slot), _strip_quotes(val_str)))
                matched_strategy = True

        # Strategy 3: Key-as-Function Format
        if not matched_strategy:
            for key, val in item.items():
                if isinstance(val, dict):
                    domain = key
                    params = val
                    for slot, v in params.items():
                        val_str = str(v) if not isinstance(v, (list, dict)) else json.dumps(v)
                        results.append((str(domain), str(slot), _strip_quotes(val_str)))
    
    return results

def extract_all_slot_values(x: Union[str, List[str], None]) -> List[Tuple[str, str, str]]:
    if x is None:
        return []
    
    if isinstance(x, list):
        all_res = []
        for item in x:
            if isinstance(item, str):
                res = extract_from_json(item)
                if not res:
                    calls = [m.group(0) for m in _CALL_RE.finditer(item)]
                    if not calls and _CALL_RE.fullmatch(item.strip()):
                        calls = [item.strip()]
                    for c in calls:
                        res.extend(parse_regex_call(c))
                all_res.extend(res)
        return all_res

    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        
        # 1. Try JSON
        json_res = extract_from_json(s)
        if json_res:
            return json_res
        
        # 2. Try Regex
        calls = [m.group(0) for m in _CALL_RE.finditer(s)]
        if not calls and _CALL_RE.fullmatch(s):
            calls = [s]
        
        regex_res = []
        for c in calls:
            regex_res.extend(parse_regex_call(c))
        
        return regex_res
        
    return []


# -----------------------------
# Metric: slot-set AND, value OR
# -----------------------------
@dataclass
class PRF:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int

def prf_from_counts(tp: int, fp: int, fn: int) -> PRF:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2*p*r/(p+r)) if (p+r) > 0 else 0.0
    return PRF(p, r, f1, tp, fp, fn)

def build_gt_allowed_map(gt_field: Any) -> Dict[Tuple[str, str], Set[str]]:
    allowed = defaultdict(set)
    slotvals = extract_all_slot_values(gt_field)
    for d, s, v in slotvals:
        allowed[(d, s)].add(v)
    return dict(allowed)

def build_pred_map(pred_field: Any) -> Dict[Tuple[str, str], Set[str]]:
    pred = defaultdict(set)
    slotvals = extract_all_slot_values(pred_field)
    for d, s, v in slotvals:
        pred[(d, s)].add(v)
    return dict(pred)

def counts_slot_and_value_or(
    gt_allowed: Dict[Tuple[str, str], Set[str]],
    pred_vals: Dict[Tuple[str, str], Set[str]],
) -> Tuple[int, int, int]:
    tp = fp = fn = 0

    for key, allowed_vals in gt_allowed.items():
        pv = pred_vals.get(key, set())
        if pv and (pv & allowed_vals):
            tp += 1
        else:
            fn += 1

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


# -----------------------------
# Logging (OR-case & Parse Failure)
# -----------------------------
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
    lines.append("-" * 80)
    lines.append(f"[OR-CASE] EXAMPLE: {ex_id}")
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
    return "\n".join(lines)

def format_parse_failure_block(ex: Dict[str, Any]) -> str:
    ex_id = ex.get("example_id_sub") or ex.get("example_id") or "NA"
    utt = ex.get("user_utterance", "")
    gt_raw = ex.get("reference_ground_truth", None)
    pred_raw = ex.get("llm_output", None)

    lines = []
    lines.append("!" * 80)
    lines.append(f"[PARSE FAILURE] EXAMPLE: {ex_id}")
    if utt:
        lines.append(f"USER_UTTERANCE: {utt}")
    lines.append(f"GT_RAW:   {gt_raw}")
    lines.append(f"PRED_RAW: {pred_raw}")
    lines.append("")
    lines.append("REASON: LLM output is not empty, but valid function calls could not be extracted.")
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
            raw_pred = ex.get(pred_key)
            pred_vals = build_pred_map(raw_pred)
            
            tp, fp, fn = counts_slot_and_value_or(gt_allowed, pred_vals)
            
            or_block = format_or_case_block(ex, gt_allowed, pred_vals, tp, fp, fn)
            if or_block:
                f.write(or_block)
                f.write("\n\n")
                wrote_any = True
            
            # Check for Parse Failures
            has_raw_content = False
            if isinstance(raw_pred, str) and raw_pred.strip():
                has_raw_content = True
            elif isinstance(raw_pred, list) and len(raw_pred) > 0:
                has_raw_content = True
            
            if has_raw_content and not pred_vals:
                fail_block = format_parse_failure_block(ex)
                f.write(fail_block)
                f.write("\n\n")
                wrote_any = True

        if not wrote_any:
            f.write("No OR cases or Parsing Failures found.\n")

# -----------------------------
# Main
# -----------------------------
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

def main(json_path: str, or_log_path: Optional[str] = None):
    data = load_json(json_path)
    examples = extract_examples(data)

    overall = micro_f1_slot_and_value_or(examples)
    print("=== Micro-F1 (Enhanced JSON & Unquoted Regex Parser) ===")
    print(f"TP={overall.tp} FP={overall.fp} FN={overall.fn}")
    print(f"P={overall.precision:.4f} R={overall.recall:.4f} F1={overall.f1:.4f}")

    if or_log_path:
        write_or_case_log(examples, or_log_path)
        print(f"[Saved Log] {or_log_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--or_log_path", type=str, default=None,
                        help="If set, writes OR-case and Parse Failure logs.")
    args = parser.parse_args()
    main(args.json_path, or_log_path=args.or_log_path)