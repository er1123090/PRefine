"""
Core evaluation metrics shared by single-turn and multi-turn evaluation.

Metrics:
    1. Micro-F1 over (domain, slot) pairs with OR-allowed values
    2. Pref-slot Exact Match Rate
    3. Non-pref-slot Micro-F1

Parsing supports:
    - Function-call syntax: Domain(slot="val", ...)
    - JSON variants: flat, nested, key-as-function
    - Braced function syntax: {Domain}(...)
    - Markdown-bold function names: **Domain**(...)
    - Quoted/braced model variants: {"Domain"(...)}
    - A single missing closing parenthesis in an otherwise clear wrapped call
    - Markdown code fences
    - <think> tag stripping
    - Date/time normalisation for date/time slots
"""

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    import dateparser
    _DATEPARSER_AVAILABLE = True
except ImportError:
    _DATEPARSER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_CALL_RE = re.compile(r"([A-Za-z_]\w*)\s*\((.*?)\)")
_BRACED_FUNC_RE = re.compile(r"\{([A-Za-z_]\w*)\}\s*\(")
_BOLD_FUNC_RE = re.compile(r"\*{2}\s*([A-Za-z_]\w*)\s*\*{2}\s*\(")
_QUOTED_BRACED_FUNC_RE = re.compile(
    r"""\{\s*(?P<quote>["'])(?P<func>[A-Za-z_]\w*)(?P=quote)"""
    r"""\s*(?:\}\s*|:\s*)?\("""
)
_COMMA_WRAPPED_FUNC_RE = re.compile(
    r"""^\s*\{\s*(?P<quote>["'])(?P<func>[A-Za-z_]\w*)(?P=quote)"""
    r"""\s*,\s*(?P<args>.+?)\s*\}\s*$""",
    re.DOTALL,
)
_QUOTED_ARG_NAME_RE = re.compile(
    r"""(?P<quote>["'])(?P<slot>[A-Za-z_]\w*)(?P=quote)\s*="""
)
_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

def _strip_code_fences(s: str) -> str:
    return _CODE_FENCE_RE.sub("", s.strip())


def _normalize_braced_func(s: str) -> str:
    return _BRACED_FUNC_RE.sub(r"\1(", s)


def _normalize_malformed_func(s: str) -> str:
    comma_wrapped = _COMMA_WRAPPED_FUNC_RE.fullmatch(s)
    if comma_wrapped and "=" in comma_wrapped.group("args"):
        s = (
            f"{comma_wrapped.group('func')}"
            f"({comma_wrapped.group('args')})"
        )

    s = _BOLD_FUNC_RE.sub(r"\1(", s)
    s = _QUOTED_BRACED_FUNC_RE.sub(
        lambda match: f"{match.group('func')}(",
        s,
    )
    return _QUOTED_ARG_NAME_RE.sub(
        lambda match: f"{match.group('slot')}=",
        s,
    )


def _parenthesis_balance(s: str) -> int:
    balance = 0
    in_quote: Optional[str] = None
    escaped = False
    for char in s:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if in_quote:
            if char == in_quote:
                in_quote = None
            continue
        if char in {'"', "'"}:
            in_quote = char
        elif char == "(":
            balance += 1
        elif char == ")":
            balance -= 1
    return balance


def _repair_single_missing_closing_paren(s: str) -> str:
    stripped = s.strip()
    if not re.match(r"^\{?\s*[A-Za-z_]\w*\s*\(", stripped):
        return s
    if not stripped.endswith("}") or _parenthesis_balance(stripped) != 1:
        return s
    closing_brace = s.rfind("}")
    return f"{s[:closing_brace]}){s[closing_brace:]}"


def _strip_think_tags(s: str) -> str:
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL | re.IGNORECASE)
    return re.sub(r"</think>", "", s, flags=re.IGNORECASE)


def _remove_code_fences_keep_content(s: str) -> str:
    s = re.sub(r"```(?:json)?", "", s, flags=re.IGNORECASE)
    return s.replace("```", "")


# ---------------------------------------------------------------------------
# JSON -> function-call strings
# ---------------------------------------------------------------------------

def _build_call_string(func: str, args: Any) -> str:
    if not isinstance(args, dict) or not args:
        return f"{func}()"
    parts = [f'{k}="{str(v)}"' for k, v in args.items() if v is not None]
    return f"{func}({', '.join(parts)})"


def _json_to_calls(obj: Any) -> List[str]:
    calls: List[str] = []
    if isinstance(obj, list):
        for item in obj:
            calls.extend(_json_to_calls(item))
        return calls

    if not isinstance(obj, dict):
        return calls

    # "calls" wrapper
    if "calls" in obj and isinstance(obj["calls"], list):
        for c in obj["calls"]:
            calls.extend(_json_to_calls(c))
        return calls

    keys_lower = {k.lower(): k for k in obj}
    func_candidates = ["name", "tool", "function", "domain"]
    arg_candidates = ["arguments", "args", "parameters"]

    found_func_key = next((keys_lower[c] for c in func_candidates if c in keys_lower), None)
    found_arg_key = next((keys_lower[c] for c in arg_candidates if c in keys_lower), None)

    if found_func_key and found_arg_key:
        name = obj[found_func_key]
        args = obj[found_arg_key]
        if isinstance(name, str):
            calls.append(_build_call_string(name, args if isinstance(args, dict) else {}))
        return calls

    if found_func_key and not found_arg_key:
        name = obj[found_func_key]
        if isinstance(name, str):
            skip_keys = {
                "reasoning", "raw_content", "model_name", "evaluation_result",
                "reasoning_tokens", "domain", "name", "tool", "function", found_func_key,
            }
            flat_args = {k: v for k, v in obj.items() if k not in skip_keys and k.lower() not in skip_keys}
            calls.append(_build_call_string(name, flat_args))
        return calls

    # Strategy C: key-as-function {"GetHotels": {...}}
    skip_set = set(func_candidates + arg_candidates + [
        "reasoning", "raw_content", "model_name", "evaluation_result", "reasoning_tokens",
    ])
    for k, v in obj.items():
        if not isinstance(k, str) or k.lower() in skip_set:
            continue
        if isinstance(v, dict):
            calls.append(_build_call_string(k, v))
        elif v is None:
            calls.append(f"{k}()")
        elif isinstance(v, list):
            if all(isinstance(i, dict) for i in v) and v:
                for item in v:
                    calls.append(_build_call_string(k, item))
            else:
                calls.append(_build_call_string(k, {"value": v}))
        else:
            calls.append(_build_call_string(k, {"value": v}))
    return calls


def _try_parse_json_to_calls(s: str) -> List[str]:
    try:
        return _json_to_calls(json.loads(s))
    except Exception:
        return []


@lru_cache(maxsize=32768)
def _extract_calls_from_string(x: str) -> Tuple[str, ...]:
    s = x.strip()
    if not s:
        return ()

    s = _strip_think_tags(s)
    s = _remove_code_fences_keep_content(s)
    s = _strip_code_fences(s)
    s = _normalize_braced_func(s)
    s = _normalize_malformed_func(s)
    s = _repair_single_missing_closing_paren(s)

    json_calls = _try_parse_json_to_calls(s)
    if json_calls:
        return tuple(json_calls)

    calls = [m.group(0) for m in _CALL_RE.finditer(s)]
    if not calls and _CALL_RE.fullmatch(s):
        calls = [s]
    return tuple(calls)


def extract_calls(x: Union[str, List[str], None]) -> List[str]:
    """Extract all function-call strings from a model output field."""
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
    return list(_extract_calls_from_string(x))


# ---------------------------------------------------------------------------
# Slot/value parsing inside Func(...)
# ---------------------------------------------------------------------------

def _process_value(val_str: str) -> str:
    s = val_str.strip()
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        return s[1:-1]
    if s.lower() == "true":
        return "True"
    if s.lower() == "false":
        return "False"
    return s


@lru_cache(maxsize=32768)
def _normalize_date_time_value(slot: str, value: str) -> str:
    """Normalise date/time slot values for fair comparison."""
    if not _DATEPARSER_AVAILABLE:
        return value
    slot_lower = slot.lower()
    settings = {"PREFER_DATES_FROM": "future", "STRICT_PARSING": False}
    try:
        if "time" in slot_lower:
            if not any(c.isdigit() for c in value):
                return value
            dt = dateparser.parse(value, settings=settings)
            if dt:
                return dt.strftime("%H:%M")
        elif "date" in slot_lower or "day" in slot_lower:
            dt = dateparser.parse(value, settings=settings)
            if dt:
                return dt.strftime("%m-%d")
    except Exception:
        pass
    return value


def _split_args(arg_str: str) -> List[str]:
    parts, buf = [], []
    in_quote: Optional[str] = None
    escape = False
    for ch in arg_str:
        if escape:
            buf.append(ch)
            escape = False
            continue
        if ch == "\\":
            buf.append(ch)
            escape = True
            continue
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


@lru_cache(maxsize=32768)
def _parse_call_to_slotvals_cached(
    call: str,
) -> Tuple[Tuple[str, str, str], ...]:
    call = call.strip()
    m = _CALL_RE.fullmatch(call)
    if not m:
        return ()
    domain = m.group(1).strip()
    args_str = m.group(2).strip()
    if not args_str:
        return ()
    out: List[Tuple[str, str, str]] = []
    for part in _split_args(args_str):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        slot = k.strip()
        val = _process_value(v.strip())
        val = _normalize_date_time_value(slot, val)
        out.append((domain, slot, val))
    return tuple(out)


def parse_call_to_slotvals(call: str) -> List[Tuple[str, str, str]]:
    """Parse 'Domain(slot="val", ...)' into [(domain, slot, value), ...]."""
    return list(_parse_call_to_slotvals_cached(call))


# ---------------------------------------------------------------------------
# Map builders
# ---------------------------------------------------------------------------

def build_gt_allowed_map(gt_field: Any) -> Dict[Tuple[str, str], Set[str]]:
    """Build {(domain, slot): {allowed_values}} from a ground-truth field."""
    allowed: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for call in extract_calls(gt_field):
        for d, s, v in parse_call_to_slotvals(call):
            allowed[(d, s)].add(v)
    return dict(allowed)


def build_pred_map(pred_field: Any) -> Dict[Tuple[str, str], Set[str]]:
    """Build {(domain, slot): {predicted_values}} from a model output field."""
    pred: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for call in extract_calls(pred_field):
        for d, s, v in parse_call_to_slotvals(call):
            pred[(d, s)].add(v)
    return dict(pred)


# ---------------------------------------------------------------------------
# Core counting
# ---------------------------------------------------------------------------

def counts_slot_and_value_or(
    gt_allowed: Dict[Tuple[str, str], Set[str]],
    pred_vals: Dict[Tuple[str, str], Set[str]],
) -> Tuple[int, int, int]:
    """
    Slot-level TP/FP/FN with OR-allowed values:
        - A predicted slot hits if ANY of its values intersects the allowed set.
    """
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
        elif not (pv & gt_allowed[key]):
            fp += 1
    return tp, fp, fn


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

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


def prf_from_counts(tp: int, fp: int, fn: int) -> PRF:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return PRF(p, r, f1, tp, fp, fn)


# ---------------------------------------------------------------------------
# Pref-list helpers
# ---------------------------------------------------------------------------

def load_pref_list(path: str) -> Dict[str, Set[str]]:
    """Load pref_list.json -> {domain: {pref_slot, ...}}."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return {d: set(str(s) for s in slots) for d, slots in obj.items() if isinstance(slots, list)}
    except Exception:
        return {}


def is_pref_slot(domain: str, slot: str, pref_map: Dict[str, Set[str]]) -> bool:
    return domain in pref_map and slot in pref_map[domain]


def filter_map_by_pref(
    slotval_map: Dict[Tuple[str, str], Set[str]],
    pref_map: Dict[str, Set[str]],
    want_pref: bool,
) -> Dict[Tuple[str, str], Set[str]]:
    return {
        (d, s): vals
        for (d, s), vals in slotval_map.items()
        if is_pref_slot(d, s, pref_map) == want_pref
    }


# ---------------------------------------------------------------------------
# Micro-F1 variants
# ---------------------------------------------------------------------------

def micro_f1_slot_and_value_or(
    examples: List[Dict[str, Any]],
    gt_key: str = "reference_ground_truth",
    pred_key: str = "llm_output",
) -> PRF:
    TP = FP = FN = 0
    for ex in examples:
        gt = build_gt_allowed_map(ex.get(gt_key))
        pred = build_pred_map(ex.get(pred_key))
        tp, fp, fn = counts_slot_and_value_or(gt, pred)
        TP += tp
        FP += fp
        FN += fn
    return prf_from_counts(TP, FP, FN)


def micro_f1_with_filter(
    examples: List[Dict[str, Any]],
    pref_map: Dict[str, Set[str]],
    want_pref: bool,
    gt_key: str = "reference_ground_truth",
    pred_key: str = "llm_output",
) -> PRF:
    TP = FP = FN = 0
    for ex in examples:
        gt_all = build_gt_allowed_map(ex.get(gt_key))
        pred_all = build_pred_map(ex.get(pred_key))
        gt = filter_map_by_pref(gt_all, pref_map, want_pref)
        pred = filter_map_by_pref(pred_all, pref_map, want_pref)
        tp, fp, fn = counts_slot_and_value_or(gt, pred)
        TP += tp
        FP += fp
        FN += fn
    return prf_from_counts(TP, FP, FN)


# ---------------------------------------------------------------------------
# Pref-slot Exact Match
# ---------------------------------------------------------------------------

def pref_em_one(
    ex: Dict[str, Any],
    pref_map: Dict[str, Set[str]],
    gt_key: str = "reference_ground_truth",
    pred_key: str = "llm_output",
) -> int:
    gt_pref = filter_map_by_pref(build_gt_allowed_map(ex.get(gt_key)), pref_map, want_pref=True)
    pred_pref = filter_map_by_pref(build_pred_map(ex.get(pred_key)), pref_map, want_pref=True)
    _, fp, fn = counts_slot_and_value_or(gt_pref, pred_pref)
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
    em_count = sum(pref_em_one(ex, pref_map, gt_key, pred_key) for ex in examples)
    return EMR(em_count / n, em_count, n)


# ---------------------------------------------------------------------------
# Parsing failure detection
# ---------------------------------------------------------------------------

def is_parsing_failed(ex: Dict[str, Any], pred_key: str = "llm_output") -> Tuple[bool, str]:
    calls = extract_calls(ex.get(pred_key))
    if not calls:
        return True, "no_calls_extracted"
    return False, "ok"


# ---------------------------------------------------------------------------
# File-level evaluation
# ---------------------------------------------------------------------------

def load_json_examples(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "examples", "items"):
            if key in data and isinstance(data[key], list):
                return data[key]
    raise ValueError(f"Unsupported JSON structure in {path}")


def eval_one_file(
    json_path: str,
    pref_map: Dict[str, Set[str]],
) -> Tuple[PRF, EMR, PRF]:
    """
    Evaluate a single result file.

    Returns:
        (overall_prf, pref_emr, nonpref_prf)
    """
    examples = load_json_examples(json_path)
    overall_prf = micro_f1_slot_and_value_or(examples)
    pref_emr = pref_em_rate(examples, pref_map)
    nonpref_prf = micro_f1_with_filter(examples, pref_map, want_pref=False)
    return overall_prf, pref_emr, nonpref_prf
