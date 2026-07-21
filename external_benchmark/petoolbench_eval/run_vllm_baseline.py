"""Run a vLLM baseline on PEToolBench test splits."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


HISTORY_KEYS = {
    "p": "instruction_preferred",
    "r": "instruction_ratings",
    "c": "instruction_chronological",
}


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def first_json_object(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return {}


def normalize_rows(path: Path, history_type: str, limit: int | None) -> List[Dict[str, Any]]:
    rows = read_json(path)
    selected = rows if limit is None else rows[:limit]
    prompt_key = HISTORY_KEYS[history_type]
    normalized = []
    for idx, row in enumerate(selected):
        normalized.append(
            {
                "example_id": f"petoolbench_{history_type}_{idx:06d}",
                "source_index": idx,
                "history_type": history_type,
                "system_prompt": row[prompt_key],
                "query": row["query"],
                "api_call_ground_truth": row["api_call_ground_truth"],
            }
        )
    return normalized


def build_prompt(tokenizer: Any, record: Dict[str, Any], max_input_tokens: int) -> str:
    messages = [
        {"role": "system", "content": record["system_prompt"]},
        {"role": "user", "content": record["query"]},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_input_tokens,
        add_special_tokens=False,
    )
    return tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)


def evaluate(predictions: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(predictions)
    tool_correct = 0
    parameter_correct = 0
    parse_failures = 0
    for row in rows:
        gt = row.get("api_call_ground_truth") or {}
        pred = row.get("parsed_response") or {}
        if not pred:
            parse_failures += 1
        tool_correct += int(gt.get("tool_name") == pred.get("tool_name"))
        parameter_correct += int(gt.get("parameters") == pred.get("parameters"))
    n = len(rows)
    total_elapsed = sum(float(row.get("elapsed_seconds", 0.0) or 0.0) for row in rows)
    return {
        "n": n,
        "tool_correct": tool_correct,
        "parameter_correct": parameter_correct,
        "tool_accuracy": tool_correct / n if n else 0.0,
        "parameter_accuracy": parameter_correct / n if n else 0.0,
        "parse_failures": parse_failures,
        "total_generation_seconds": total_elapsed,
        "seconds_per_example": total_elapsed / n if n else 0.0,
    }


def run_split(llm: LLM, tokenizer: Any, args: argparse.Namespace, history_type: str) -> Dict[str, Any]:
    input_path = args.petoolbench_dir / "dataset_test" / f"user_entries_test_{history_type}.json"
    records = normalize_rows(input_path, history_type, args.limit)
    output_dir = args.output_dir / args.run_name
    predictions_path = output_dir / f"predictions_{history_type}.json"
    summary_path = output_dir / f"summary_{history_type}.json"

    if predictions_path.exists() and args.resume:
        predictions = read_json(predictions_path)
    else:
        predictions = []

    done_ids = {row["example_id"] for row in predictions}
    remaining = [record for record in records if record["example_id"] not in done_ids]
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        stop=args.stop or None,
    )

    for start in range(0, len(remaining), args.batch_size):
        batch = remaining[start : start + args.batch_size]
        prompts = [build_prompt(tokenizer, record, args.max_input_tokens) for record in batch]
        started = time.time()
        outputs = llm.generate(prompts, sampling)
        elapsed = time.time() - started
        per_example = elapsed / max(len(batch), 1)

        for record, output in zip(batch, outputs):
            response = output.outputs[0].text.strip() if output.outputs else ""
            predictions.append(
                {
                    "example_id": record["example_id"],
                    "source_index": record["source_index"],
                    "history_type": record["history_type"],
                    "query": record["query"],
                    "api_call_ground_truth": record["api_call_ground_truth"],
                    "response": response,
                    "parsed_response": first_json_object(response),
                    "elapsed_seconds": per_example,
                }
            )

        write_json(predictions_path, predictions)
        write_json(summary_path, evaluate(predictions))
        print(
            f"{history_type} progress={len(predictions)}/{len(records)} "
            f"batch_elapsed={elapsed:.2f}s"
        )

    summary = evaluate(predictions)
    write_json(predictions_path, predictions)
    write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a vLLM baseline on PEToolBench.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--petoolbench-dir",
        type=Path,
        default=Path("experiments5/external_benchmark/PEToolBench"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments5/external_benchmark/petoolbench_eval/results"),
    )
    parser.add_argument("--run-name", default="qwen25_7b_instruct_vllm")
    parser.add_argument("--history-types", nargs="+", choices=["p", "r", "c"], default=["p", "r", "c"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-input-tokens", type=int, default=4000)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--stop", nargs="*", default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_input_tokens + args.max_new_tokens,
        trust_remote_code=True,
    )

    summaries = {}
    for history_type in args.history_types:
        summaries[history_type] = run_split(llm, tokenizer, args, history_type)
        print(history_type, json.dumps(summaries[history_type], ensure_ascii=False))

    aggregate = {
        "model": args.model,
        "backend": "vllm",
        "run_name": args.run_name,
        "limit": args.limit,
        "history_types": args.history_types,
        "summaries": summaries,
    }
    write_json(args.output_dir / args.run_name / "summary_all.json", aggregate)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

