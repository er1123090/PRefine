import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple, Union, Optional
from collections import defaultdict
import argparse

# =========================================================
# Robust Parsing Logic (Updated from Reference Code)
# =========================================================

_CALL_RE = re.compile(r"([A-Za-z_]\w*)\s*\((.*?)\)")
_BRACED_FUNC_RE = re.compile(r"\{([A-Za-z_]\w*)\}\s*\(")  # {GetHotels}(

def _strip_think_tags(s: str) -> str:
    """Removes <think>...</think> blocks."""
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"</think>", "", s, flags=re.IGNORECASE)
    return s.strip()

def _strip_quotes(s: str) -> str:
    """단순 따옴표 제거"""
    s = str(s).strip()
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        return s[1:-1]
    return s

def _process_regex_value(val_str: str) -> str:
    """
    Regex 파싱 전용 값 처리 함수
    1. 따옴표가 있다면 -> 내용만 추출
    2. 따옴표가 없다면 -> Boolean(true/false) 정규화
    """
    s = val_str.strip()
    
    # 1. Quoted String: 따옴표 제거 후 내용 반환
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        return s[1:-1]
    
    # 2. Unquoted Literal: Boolean 정규화
    if s.lower() == "true":
        return "True"
    if s.lower() == "false":
        return "False"
        
    return s

def _split_args(arg_str: str) -> List[str]:
    """Splits arguments properly handling quotes and escapes."""
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
    """정규표현식 기반 파싱 (Unquoted Boolean 지원)"""
    call = call.strip()
    # Handle {Func}(...) case
    call = _BRACED_FUNC_RE.sub(r"\1(", call)
    
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
        val = _process_regex_value(v)
        out.append((domain, slot, val))
    return out

def _extract_from_json_object(item: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """Applies 3 strategies to extract function calls from a single JSON dict."""
    results = []
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
        if ("name" in item or "tool" in item) and ("parameters" in item or "arguments" in item or "args" in item):
            domain = item.get("name") or item.get("tool")
            params = item.get("parameters") or item.get("arguments") or item.get("args")
            if domain and isinstance(params, dict):
                for slot, val in params.items():
                    val_str = str(val) if not isinstance(val, (list, dict)) else json.dumps(val)
                    results.append((str(domain), str(slot), _strip_quotes(val_str)))
                matched_strategy = True

    # Strategy 3: Key-as-Function Format
    if not matched_strategy:
        for key, val in item.items():
            if key in {"reasoning", "raw_content", "model_name", "evaluation_result", "reasoning_tokens"}:
                continue
            if isinstance(val, dict):
                domain = key
                params = val
                for slot, v in params.items():
                    val_str = str(v) if not isinstance(v, (list, dict)) else json.dumps(v)
                    results.append((str(domain), str(slot), _strip_quotes(val_str)))
    
    return results

def extract_all_slot_values(x: Union[str, List[str], None]) -> List[Tuple[str, str, str]]:
    """
    Main Parsing Entry Point (Integrated)
    """
    if x is None:
        return []
    
    if isinstance(x, list):
        all_res = []
        for item in x:
            if isinstance(item, str):
                res = extract_all_slot_values(item)
                all_res.extend(res)
            elif isinstance(item, dict):
                res = _extract_from_json_object(item)
                all_res.extend(res)
        return all_res

    if isinstance(x, str):
        # 1. Clean <think> tags first
        s = _strip_think_tags(x).strip()
        if not s:
            return []
        
        # 2. Try JSON Extraction
        cleaned_json = re.sub(r"```(?:json)?", "", s, flags=re.IGNORECASE).replace("```", "").strip()
        json_success = False
        json_results = []
        
        try:
            data = json.loads(cleaned_json)
            json_success = True
        except json.JSONDecodeError:
            start = cleaned_json.find('{')
            end = cleaned_json.rfind('}')
            if start != -1 and end != -1:
                try:
                    data = json.loads(cleaned_json[start:end+1])
                    json_success = True
                except json.JSONDecodeError:
                    pass
        
        if json_success:
            if isinstance(data, dict):
                json_results.extend(_extract_from_json_object(data))
            elif isinstance(data, list):
                for sub in data:
                    if isinstance(sub, dict):
                        json_results.extend(_extract_from_json_object(sub))
            
            if json_results:
                return json_results

        # 3. Try Regex Extraction
        calls = [m.group(0) for m in _CALL_RE.finditer(s)]
        braced_calls = [m.group(0) for m in _BRACED_FUNC_RE.finditer(s)]
        
        combined_calls = list(set(calls + braced_calls))
        if not combined_calls:
            if _CALL_RE.fullmatch(s) or _BRACED_FUNC_RE.match(s):
                combined_calls = [s]
        
        regex_res = []
        for c in combined_calls:
            regex_res.extend(parse_regex_call(c))
        
        return regex_res
        
    return []

# -----------------------------
# Metric Utils
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

def micro_f1_slot_and_value_or(examples: List[Dict[str, Any]]) -> PRF:
    TP = FP = FN = 0
    for ex in examples:
        gt_allowed = build_gt_allowed_map(ex.get("reference_ground_truth"))
        pred_vals = build_pred_map(ex.get("llm_output"))
        tp, fp, fn = counts_slot_and_value_or(gt_allowed, pred_vals)
        TP += tp; FP += fp; FN += fn
    return prf_from_counts(TP, FP, FN)

# -----------------------------
# New: Pref Group Logic
# -----------------------------

PREF_GROUP_DEFINITIONS = {
  "low_cost": {
    "group_preference": "budget_conscious",
    "rules": [
      { "domain": "GetRestaurants", "slot": "price_range", "value": "cheap" },
      { "domain": "GetRentalCars", "slot": "car_type", "value": "Compact" },
      { "domain": "GetHotels", "slot": "average_star", "value": 1 },
      { "domain": "GetHotels", "slot": "average_star", "value": 2 },
      { "domain": "GetRideSharing", "slot": "shared_ride", "value": True },
      { "domain": "GetTravel", "slot": "free_entry", "value": True },
      { "domain": "GetFlights", "slot": "flight_class", "value": "Economy" }
    ]
  },
  "high_cost": {
    "group_preference": "budget_conscious",
    "rules": [
      { "domain": "GetRestaurants", "slot": "price_range", "value": "pricey" },
      { "domain": "GetRentalCars", "slot": "car_type", "value": "Full-size" },
      { "domain": "GetHotels", "slot": "average_star", "value": 4 },
      { "domain": "GetHotels", "slot": "average_star", "value": 5 }
    ]
  },
  "solo_usage": {
    "group_preference": "solo_travel",
    "rules": [
      { "domain": "GetBuses", "slot": "group_size", "value": 1 },
      { "domain": "GetFlights", "slot": "passengers", "value": 1 },
      { "domain": "GetRideSharing", "slot": "number_of_seats", "value": 1 },
      { "domain": "GetEvents", "slot": "number_of_tickets", "value": 1 },
      { "domain": "GetRestaurants", "slot": "number_of_seats", "value": 1 }
    ]
  }
}

def normalize_val(v: Any) -> str:
    """JSON raw 값을 파서의 문자열 포맷으로 정규화 (Boolean, Int 처리)"""
    if isinstance(v, bool):
        return "True" if v else "False" # Reference Parser aligns Boolean to "True"/"False"
    if isinstance(v, (int, float)):
        return str(v)
    return str(v)

class GroupMatcher:
    def __init__(self, definitions: Dict[str, Any]):
        # (domain, slot, normalized_value) -> list of group_names
        self.rule_map = defaultdict(list)
        self.all_group_names = sorted(list(definitions.keys()))
        
        for g_name, content in definitions.items():
            for rule in content.get("rules", []):
                d = rule["domain"]
                s = rule["slot"]
                v = rule["value"]
                # 파싱된 결과와 매칭하기 위해 값을 문자열로 정규화
                norm_v = normalize_val(v)
                self.rule_map[(d, s, norm_v)].append(g_name)

    def get_groups(self, domain: str, slot: str, value: str) -> List[str]:
        return self.rule_map.get((domain, slot, value), [])

def evaluate_by_groups(examples: List[Dict[str, Any]]) -> Dict[str, PRF]:
    matcher = GroupMatcher(PREF_GROUP_DEFINITIONS)
    
    # group_name -> {tp, fp, fn}
    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    
    for ex in examples:
        gt_allowed = build_gt_allowed_map(ex.get("reference_ground_truth"))
        pred_vals = build_pred_map(ex.get("llm_output"))
        
        # 1. Analyze GT items (Check for TP and FN)
        for (d, s), allowed_set in gt_allowed.items():
            pv = pred_vals.get((d, s), set())
            
            # Match Logic: 
            # OR-case 지원: allowed_set 중 하나라도 매칭되면 HIT
            hit_val = None
            intersection = pv & allowed_set
            if intersection:
                hit_val = list(intersection)[0] # 매칭된 값 중 하나 선택
            
            if hit_val:
                # HIT (TP): 매칭된 값이 속한 그룹에 TP 부여
                groups = matcher.get_groups(d, s, hit_val)
                for g in groups:
                    stats[g]["tp"] += 1
            else:
                # MISS (FN): GT에 존재했으나 예측하지 못한 값들이 속한 그룹에 FN 부여
                missed_groups = set()
                for val in allowed_set:
                    gs = matcher.get_groups(d, s, val)
                    missed_groups.update(gs)
                
                for g in missed_groups:
                    stats[g]["fn"] += 1

        # 2. Analyze Pred items (Check for FP)
        for (d, s), p_set in pred_vals.items():
            allowed_set = gt_allowed.get((d, s), set())
            
            for p_val in p_set:
                if p_val in allowed_set:
                    continue # 정답인 경우는 위에서 TP로 처리됨
                
                # 정답이 아닌 예측값 (FP)
                # 이 잘못된 값이 어떤 그룹 규칙에 해당한다면 해당 그룹의 FP 증가
                groups = matcher.get_groups(d, s, p_val)
                for g in groups:
                    stats[g]["fp"] += 1

    # Convert counts to PRF objects
    results = {}
    for g_name in matcher.all_group_names:
        s = stats[g_name]
        results[g_name] = prf_from_counts(s["tp"], s["fp"], s["fn"])
    
    return results

# -----------------------------
# Main Execution
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
    raise ValueError("Unsupported JSON structure")

def main(json_path: str):
    data = load_json(json_path)
    examples = extract_examples(data)

    # 1. Overall Metric
    overall = micro_f1_slot_and_value_or(examples)
    print("="*40)
    print(" [Overall Performance] ")
    print(f" TP: {overall.tp} | FP: {overall.fp} | FN: {overall.fn}")
    print(f" Precision: {overall.precision:.4f}")
    print(f" Recall:    {overall.recall:.4f}")
    print(f" F1-Score:  {overall.f1:.4f}")
    print("="*40)
    print("")

    # 2. Group-wise Metric
    group_results = evaluate_by_groups(examples)
    print(" [Performance by Preference Group] ")
    print(f" {'Group Name':<15} | {'F1':<6} | {'Prec':<6} | {'Rec':<6} | {'TP':<3} {'FP':<3} {'FN':<3}")
    print("-" * 60)
    
    for g_name, prf in group_results.items():
        print(f" {g_name:<15} | {prf.f1:.4f} | {prf.precision:.4f} | {prf.recall:.4f} | {prf.tp:<3} {prf.fp:<3} {prf.fn:<3}")
    print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True, help="Path to evaluation .json file")
    args = parser.parse_args()
    
    main(args.json_path)