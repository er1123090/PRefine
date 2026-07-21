# Split conflicting api_call slots into api_calls_drop and normalize average_rating -> average_star.
import json
import re
from collections import defaultdict

API_PATTERN = re.compile(r"^(\w+)\((.*)\)$")


def _split_args(args_str):
    parts, buf = [], []
    in_quotes, esc = False, False

    for ch in args_str:
        if esc:
            buf.append(ch)
            esc = False
            continue

        if ch == "\\":
            buf.append(ch)
            esc = True
            continue

        if ch == '"':
            in_quotes = not in_quotes
            buf.append(ch)
            continue

        if ch == "," and not in_quotes:
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


def parse_api_args(api_str):
    m = API_PATTERN.match(api_str.strip())
    if not m:
        return None, []

    name = m.group(1)
    args = m.group(2).strip()
    if args == "":
        return name, []

    pairs = []
    for part in _split_args(args):
        if "=" in part:
            k, v = part.split("=", 1)
            pairs.append((k.strip(), v.strip()))

    return name, pairs


def find_conflicts(api_calls):
    values_by_key = defaultdict(set)

    for api_str in api_calls:
        name, pairs = parse_api_args(api_str)
        if name is None:
            continue
        for slot, value in pairs:
            values_by_key[(name, slot)].add(value)

    return {key for key, values in values_by_key.items() if len(values) >= 2}


def split_api_calls(api_calls):
    conflicts = find_conflicts(api_calls)
    kept, dropped = [], []

    for api_str in api_calls:
        name, pairs = parse_api_args(api_str)
        if name is None:
            kept.append(api_str)
            continue

        kept_pairs = []
        dropped_pairs = []
        for slot, value in pairs:
            if (name, slot) in conflicts:
                dropped_pairs.append((slot, value))
            else:
                kept_pairs.append((slot, value))

        if kept_pairs:
            args = ", ".join(f"{k}={v}" for k, v in kept_pairs)
            kept.append(f"{name}({args})")
        else:
            kept.append(f"{name}()")

        if dropped_pairs:
            args = ", ".join(f"{k}={v}" for k, v in dropped_pairs)
            dropped.append(f"{name}({args})")

    return kept, dropped, len(conflicts)


def process_groups(groups):
    total_drop = 0
    total_conflicts = 0

    for group in groups:
        api_calls = group.get("api_calls", [])
        kept, dropped, conflict_count = split_api_calls(api_calls)
        group["api_calls"] = kept
        group["api_calls_drop"] = dropped
        if "api_calls" in group:
            reordered = {}
            for key, value in group.items():
                if key == "api_calls":
                    reordered[key] = value
                    reordered["api_calls_drop"] = group["api_calls_drop"]
                elif key == "api_calls_drop":
                    continue
                else:
                    reordered[key] = value
            group.clear()
            group.update(reordered)
        total_drop += len(dropped)
        total_conflicts += conflict_count

    return total_drop, total_conflicts


if __name__ == "__main__":
    input_path = "./result/dev_3.json"
    output_path = "./result/dev_4.json"

    with open(input_path, "r", encoding="utf-8") as f:
        groups = json.load(f)

    dropped, conflicts = process_groups(groups)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(groups, f, indent=2, ensure_ascii=False)

    print(
        f"saved={output_path} | groups={len(groups)} | "
        f"api_calls_drop={dropped} | conflict_keys={conflicts}"
    )
