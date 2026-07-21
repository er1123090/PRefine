import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


CALL_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="FALSE JSON file to analyze.")
    parser.add_argument(
        "--inputs-dir",
        help="Directory containing FALSE JSON files to analyze.",
    )
    parser.add_argument("--out-dir", required=True, help="Directory for output files.")
    return parser.parse_args()


def split_args(arg_str):
    args = []
    buf = []
    in_quote = False
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
        if ch == '"':
            in_quote = not in_quote
            buf.append(ch)
            continue
        if ch == "," and not in_quote:
            part = "".join(buf).strip()
            if part:
                args.append(part)
            buf = []
            continue
        buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        args.append(tail)
    return args


def parse_call(text):
    if not isinstance(text, str):
        return None, {}
    text = text.strip()
    m = CALL_RE.search(text)
    if not m:
        return None, {}
    name = m.group(1)
    args_str = m.group(2).strip()
    if not args_str:
        return name, {}
    args = {}
    for part in split_args(args_str):
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == '"' and val[-1] == '"':
            val = val[1:-1]
        args[key] = val
    return name, args


def classify_diff(pred_name, pred_args, gold_name, gold_args):
    missing = sorted(k for k in gold_args if k not in pred_args)
    extra = sorted(k for k in pred_args if k not in gold_args)
    mismatched = sorted(
        k for k in gold_args if k in pred_args and pred_args[k] != gold_args[k]
    )

    if pred_name != gold_name:
        return "wrong_tool", missing, extra, mismatched

    issue_flags = sum(bool(x) for x in (missing, extra, mismatched))
    if issue_flags > 1:
        return "mixed", missing, extra, mismatched
    if mismatched:
        return "wrong_slot_value", missing, extra, mismatched
    if missing:
        return "missing_slot", missing, extra, mismatched
    if extra:
        return "extra_slot", missing, extra, mismatched
    if pred_name and gold_name and pred_name == gold_name:
        return "formatting_only", missing, extra, mismatched
    return "unknown", missing, extra, mismatched


def summarize_lengths(lengths):
    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)
    if n == 0:
        return {"count": 0, "avg": 0, "median": 0, "min": 0, "max": 0}
    mid = n // 2
    if n % 2 == 0:
        median = (lengths_sorted[mid - 1] + lengths_sorted[mid]) / 2
    else:
        median = lengths_sorted[mid]
    return {
        "count": n,
        "avg": sum(lengths_sorted) / n,
        "median": median,
        "min": lengths_sorted[0],
        "max": lengths_sorted[-1],
    }

def sort_dict(d):
    return {k: d[k] for k in sorted(d)}


def sort_counter(counter):
    return {k: counter[k] for k in sorted(counter)}


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        input_paths = [Path(args.input)]
    elif args.inputs_dir:
        input_paths = sorted(Path(args.inputs_dir).glob("*_FALSE.json"))
    else:
        raise SystemExit("Provide --input or --inputs-dir.")

    for in_path in input_paths:
        data = json.loads(in_path.read_text(encoding="utf-8"))
        rows = []
        type_counts = Counter()
        slot_mismatch_counts = Counter()
        slot_missing_counts = Counter()
        slot_extra_counts = Counter()
        tool_confusions = Counter()
        reasoning_by_type = defaultdict(list)
        missing_slot_by_tool = defaultdict(Counter)
        extra_slot_by_tool = defaultdict(Counter)
        mismatched_slot_by_tool = defaultdict(Counter)
        errors_by_tool = Counter()

        for item in data:
            pred_call = item.get("llm_output")
            gold_call = item.get("reference_ground_truth")
            pred_name, pred_args = parse_call(pred_call)
            gold_name, gold_args = parse_call(gold_call)

            error_type, missing_slots, extra_slots, mismatched_slots = classify_diff(
                pred_name, pred_args, gold_name, gold_args
            )
            type_counts[error_type] += 1
            if gold_name:
                errors_by_tool[gold_name] += 1

            if pred_name != gold_name:
                tool_confusions[(pred_name, gold_name)] += 1
            for k in missing_slots:
                slot_missing_counts[k] += 1
                if gold_name:
                    missing_slot_by_tool[gold_name][k] += 1
            for k in extra_slots:
                slot_extra_counts[k] += 1
                if gold_name:
                    extra_slot_by_tool[gold_name][k] += 1
            for k in mismatched_slots:
                slot_mismatch_counts[k] += 1
                if gold_name:
                    mismatched_slot_by_tool[gold_name][k] += 1

            reasoning_tokens = item.get("reasoning_tokens")
            if isinstance(reasoning_tokens, (int, float)):
                reasoning_len = int(reasoning_tokens)
            else:
                reasoning_len = len(str(reasoning_tokens)) if reasoning_tokens is not None else 0

            reasoning_by_type[error_type].append(reasoning_len)

            rows.append(
                {
                    "example_id": item.get("example_id"),
                    "user_utterance": item.get("user_utterance"),
                    "predicted_call": pred_call,
                    "gold_call": gold_call,
                    "error_type": error_type,
                    "missing_slots": missing_slots,
                    "extra_slots": extra_slots,
                    "mismatched_slots": mismatched_slots,
                    "pred_args": pred_args,
                    "gold_args": gold_args,
                    "reasoning_tokens": reasoning_tokens,
                    "reasoning_len": reasoning_len,
                }
            )

        out_jsonl = out_dir / (in_path.stem + "_diff.jsonl")
        with out_jsonl.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")

        reasoning_stats = {
            err_type: summarize_lengths(lengths)
            for err_type, lengths in reasoning_by_type.items()
        }

        summary = {
            "total_records": len(rows),
            "error_type_counts": sort_counter(type_counts),
            "missing_slot_counts": sort_counter(slot_missing_counts),
            "missing_slot_by_tool": {
                tool: sort_counter(counts)
                for tool, counts in sorted(missing_slot_by_tool.items())
            },
            "extra_slot_by_tool": {
                tool: sort_counter(counts)
                for tool, counts in sorted(extra_slot_by_tool.items())
            },
            "mismatched_slot_by_tool": {
                tool: sort_counter(counts)
                for tool, counts in sorted(mismatched_slot_by_tool.items())
            },
            "error_counts_by_tool": sort_counter(errors_by_tool),
            "extra_slot_counts": sort_counter(slot_extra_counts),
            "mismatched_slot_counts": sort_counter(slot_mismatch_counts),
            "tool_confusions": sort_dict(
                {f"{k[0]} -> {k[1]}": v for k, v in tool_confusions.items()}
            ),
            "reasoning_length_stats": {
                err_type: reasoning_stats[err_type]
                for err_type in sorted(reasoning_stats)
            },
        }
        out_summary = out_dir / (in_path.stem + "_summary.json")
        out_summary.write_text(
            json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8"
        )

        # Sample top/bottom reasoning lengths per error type.
        rows_by_type = defaultdict(list)
        for row in rows:
            rows_by_type[row["error_type"]].append(row)

        sample_rows = []
        for err_type, items in rows_by_type.items():
            items_sorted = sorted(items, key=lambda r: r.get("reasoning_len", 0))
            for row in items_sorted[:5]:
                sample_rows.append(
                    {
                        "error_type": err_type,
                        "sample_rank": "bottom",
                        "example_id": row.get("example_id"),
                        "reasoning_tokens": row.get("reasoning_tokens"),
                        "reasoning_len": row.get("reasoning_len"),
                        "user_utterance": row.get("user_utterance"),
                        "predicted_call": row.get("predicted_call"),
                        "gold_call": row.get("gold_call"),
                    }
                )
            for row in items_sorted[-5:]:
                sample_rows.append(
                    {
                        "error_type": err_type,
                        "sample_rank": "top",
                        "example_id": row.get("example_id"),
                        "reasoning_tokens": row.get("reasoning_tokens"),
                        "reasoning_len": row.get("reasoning_len"),
                        "user_utterance": row.get("user_utterance"),
                        "predicted_call": row.get("predicted_call"),
                        "gold_call": row.get("gold_call"),
                    }
                )

        out_samples = out_dir / (in_path.stem + "_reasoning_samples.jsonl")
        with out_samples.open("w", encoding="utf-8") as f:
            for row in sample_rows:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
