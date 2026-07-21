"""Run a Hugging Face causal LM baseline on PEToolBench test splits."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def build_prompt(tokenizer: Any, system_prompt: str, query: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return f"{system_prompt}\nInstruction: {query}\nAssistant:"


def load_model(args: argparse.Namespace):
    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype_map[args.dtype],
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model.to("cpu")
    elif args.device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype_map[args.dtype],
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model.to("cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype_map[args.dtype],
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    model.eval()
    return tokenizer, model


def model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def generate_one(tokenizer: Any, model: Any, record: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    prompt = build_prompt(tokenizer, record["system_prompt"], record["query"])
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_input_tokens,
    )
    input_len = int(encoded["input_ids"].shape[-1])
    device = model_device(model)
    encoded = {key: value.to(device) for key, value in encoded.items()}

    started = time.time()
    with torch.inference_mode():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - started
    generated_ids = output_ids[0][input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    parsed = first_json_object(response)
    return {
        "example_id": record["example_id"],
        "source_index": record["source_index"],
        "history_type": record["history_type"],
        "query": record["query"],
        "api_call_ground_truth": record["api_call_ground_truth"],
        "response": response,
        "parsed_response": parsed,
        "input_tokens": input_len,
        "output_tokens": int(generated_ids.shape[-1]),
        "elapsed_seconds": elapsed,
    }


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


def run_split(tokenizer: Any, model: Any, args: argparse.Namespace, history_type: str) -> Dict[str, Any]:
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
    for index, record in enumerate(records, start=1):
        if record["example_id"] in done_ids:
            continue
        prediction = generate_one(tokenizer, model, record, args)
        predictions.append(prediction)
        if index % args.save_every == 0 or index == len(records):
            write_json(predictions_path, predictions)
            write_json(summary_path, evaluate(predictions))
            print(
                f"{history_type} progress={len(predictions)}/{len(records)} "
                f"last_elapsed={prediction['elapsed_seconds']:.2f}s"
            )

    summary = evaluate(predictions)
    write_json(predictions_path, predictions)
    write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a HF model baseline on PEToolBench.")
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
    parser.add_argument("--run-name", default="qwen25_7b_instruct")
    parser.add_argument("--history-types", nargs="+", choices=["p", "r", "c"], default=["p", "r", "c"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=4000)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    tokenizer, model = load_model(args)
    summaries = {}
    for history_type in args.history_types:
        summaries[history_type] = run_split(tokenizer, model, args, history_type)
        print(history_type, json.dumps(summaries[history_type], ensure_ascii=False))

    aggregate = {
        "model": args.model,
        "run_name": args.run_name,
        "limit": args.limit,
        "history_types": args.history_types,
        "summaries": summaries,
    }
    write_json(args.output_dir / args.run_name / "summary_all.json", aggregate)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

