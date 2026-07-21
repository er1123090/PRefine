"""Run PEToolBench with the original experiments5 ours_memory flow.

This runner adapts the experiments5 ours_memory memory construction to
PEToolBench while preserving PEToolBench's native output contract:

    {"tool_name": "...", "parameters": {...}}

Memory construction follows experiments5/methods/our_memory/step1_extract.py:
incremental latent preference generation plus verifier-feedback refinement.
Final inference uses the same memory surfaces as experiments5 ours_memory
(`final_implicit_preference` and `final_accumulated_api_calls`) in a
PEToolBench-specific JSON tool-call prompt.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from run_petool_memory_gvr import (
    combine_summaries,
    first_json_object,
    normalize_rows,
    write_json,
)


EXPERIMENTS5_ROOT = Path(__file__).resolve().parents[2]
if str(EXPERIMENTS5_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS5_ROOT))
EXPERIMENTS4_ROOT = Path(__file__).resolve().parents[3] / "experiments4"

from src.prompts import (  # noqa: E402
    LATENT_PREF_INITIAL_PROMPT,
    LATENT_PREF_REFINEMENT_PROMPT,
    LATENT_PREF_SYSTEM_PROMPT,
    LATENT_PREF_VERIFIER_PROMPT,
)


def load_experiment4_prompts() -> Dict[str, str]:
    prompt_path = EXPERIMENTS4_ROOT / "ours_memory" / "prompt_update3.py"
    spec = importlib.util.spec_from_file_location("exp4_prompt_update3", prompt_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load experiment4 prompts from {prompt_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {
        "system": module.LATENT_PREF_SYSTEM_PROMPT,
        "initial": module.LATENT_PREF_INITIAL_PROMPT,
        "refinement": module.LATENT_PREF_REFINEMENT_PROMPT,
        "verifier": module.LATENT_PREF_VERIFIER_PROMPT,
    }


EXPERIMENT5_PROMPTS = {
    "system": LATENT_PREF_SYSTEM_PROMPT,
    "initial": LATENT_PREF_INITIAL_PROMPT,
    "refinement": LATENT_PREF_REFINEMENT_PROMPT,
    "verifier": LATENT_PREF_VERIFIER_PROMPT,
}
EXPERIMENT4_PROMPTS: Optional[Dict[str, str]] = None
ACTIVE_PROMPTS = EXPERIMENT5_PROMPTS


def configure_method_style(args: argparse.Namespace) -> None:
    global ACTIVE_PROMPTS, EXPERIMENT4_PROMPTS
    if args.method_style == "experiment4_style":
        if EXPERIMENT4_PROMPTS is None:
            EXPERIMENT4_PROMPTS = load_experiment4_prompts()
        ACTIVE_PROMPTS = EXPERIMENT4_PROMPTS
        if args.max_retries == 3:
            args.max_retries = 10
        if args.memory_max_new_tokens == 512:
            args.memory_max_new_tokens = 4096
        if args.verifier_max_new_tokens == 256:
            args.verifier_max_new_tokens = 2048
    else:
        ACTIVE_PROMPTS = EXPERIMENT5_PROMPTS


@dataclass
class MemoryState:
    implicit_pref: str = "{}"
    accumulated_dialogue: str = ""
    accumulated_api_calls: List[str] = field(default_factory=list)
    session_count: int = 0
    evolution_log: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PendingSession:
    record: Dict[str, Any]
    state: MemoryState
    session_index: int
    full_dialogue: str
    full_api_calls: List[str]
    full_api_str: str
    session_attempts: List[Dict[str, Any]] = field(default_factory=list)
    candidate_pref: str = "None"
    feedback: str = ""
    finalized: bool = False


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def parse_json_object(text: str) -> Dict[str, Any]:
    parsed = first_json_object(text)
    if isinstance(parsed, dict):
        return parsed
    if not text:
        return {}
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def tool_call_to_text(tool_call: Dict[str, Any]) -> str:
    return json.dumps(
        {
            "tool_name": tool_call.get("tool_name", ""),
            "parameters": tool_call.get("parameters", {}),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def build_sessions(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    sessions = []
    for turn in record.get("history", []) or []:
        instruction = str(turn.get("instruction", "")).strip()
        rating = turn.get("rating")
        if rating is not None:
            instruction = (
                f"{instruction}\nObserved user satisfaction rating for this tool call: {rating}."
            )
        sessions.append(
            {
                "dialogue": [{"role": "User", "message": instruction}],
                "api_call": [tool_call_to_text(turn.get("tool_call", {}) or {})],
            }
        )
    return sessions


def format_dialogue(dialogue_list: List[Dict[str, Any]]) -> str:
    return "\n".join(
        f"{turn.get('role', 'User')}: {turn.get('message', '')}"
        for turn in dialogue_list
    )


def build_generation_messages(pending: PendingSession, previous_draft: Optional[str]) -> Tuple[str, str]:
    system_prompt = ACTIVE_PROMPTS["system"].format(
        prev_implicit=pending.state.implicit_pref or "None",
        full_dialogue=pending.full_dialogue,
        full_api_calls=pending.full_api_str,
    )
    if previous_draft and pending.feedback:
        user_prompt = ACTIVE_PROMPTS["refinement"].format(
            previous_draft=previous_draft,
            feedback=pending.feedback,
        )
    else:
        user_prompt = ACTIVE_PROMPTS["initial"]
    return system_prompt, user_prompt


def build_verifier_messages(pending: PendingSession, candidate_pref: str) -> Tuple[str, str]:
    prompt = ACTIVE_PROMPTS["verifier"].format(
        full_dialogue=pending.full_dialogue,
        full_api_calls=pending.full_api_str,
        candidate_pref=candidate_pref,
    )
    return "You are a Preference Verification Module. Output JSON only.", prompt


def build_experiment4_final_prompt(record: Dict[str, Any], memory: Dict[str, Any]) -> str:
    memory_text = memory.get("final_implicit_preference") or "None"
    api_history = memory.get("final_accumulated_api_calls") or []
    retrieved_memories = (
        "[Implicit Preferences]:\n"
        f"{memory_text}\n\n"
        "[Past API History]:\n"
        f"{chr(10).join(api_history) if api_history else 'None'}"
    )
    schema = json.dumps(record.get("candidate_tools", []), ensure_ascii=False, indent=2)
    return (
        "You are a Personalized Preference Reasoning Agent.\n"
        "Your task is to choose the most appropriate PEToolBench tool call for the "
        "current user utterance by integrating (1) Retrieved Long-term Memories and "
        "(2) Accumulated API Call History, strictly following the candidate tool schema.\n\n"
        "[Task Definition]\n"
        "The user may not explicitly state all information in the current turn. Deduce "
        "missing information by analyzing inferred preferences in the retrieved memories "
        "and accumulated API call history.\n"
        "- Repetitiveness: If a user frequently chose a specific value in the past, assume this is their preference.\n"
        "- Cross-domain Consistency: Apply stable behavioral patterns or constraints when direct evidence is missing.\n\n"
        "[Reasoning Steps]\n"
        "1. Schema Filtering: choose ONLY one tool from the candidate tool schema.\n"
        "2. Relevant Memories: map abstract constraints to the most appropriate candidate tool and parameters.\n"
        "3. Formulate Output: do not hallucinate tools or parameters beyond the candidate schema.\n\n"
        "Candidate Tool Schema:\n"
        f"{schema}\n\n"
        "Relevant Memories (User Preferences & Constraints):\n"
        f"{retrieved_memories}\n\n"
        "Current User Utterance:\n"
        f"{record.get('query', '')}\n\n"
        "Return exactly one JSON object and no extra text:\n"
        '{"tool_name": "<one candidate tool_name>", "parameters": {...}}\n'
    )


def build_final_prompt(record: Dict[str, Any], memory: Dict[str, Any], args: argparse.Namespace) -> str:
    if args.method_style == "experiment4_style":
        return build_experiment4_final_prompt(record, memory)

    memory_text = memory.get("final_implicit_preference") or "None"
    api_history = memory.get("final_accumulated_api_calls") or []
    history_note = {
        "p": "The interaction history contains positive preference examples.",
        "r": "The interaction history contains ratings; rating 1 is preferred and rating 0 should be avoided.",
        "c": "The interaction history is chronological; later behavior is stronger evidence.",
    }.get(record.get("history_type"), "Use the interaction history as preference evidence.")
    return (
        "You are solving PEToolBench, a personalized tool selection benchmark.\n"
        "Choose exactly one candidate tool for the current query by using the user's "
        "long-term latent preference memory and accumulated API-call history.\n\n"
        f"{history_note}\n\n"
        "Long-term preference memory:\n"
        f"{memory_text}\n\n"
        "Accumulated API-call history:\n"
        f"{json.dumps(api_history[-20:], ensure_ascii=False, indent=2)}\n\n"
        "Candidate tools:\n"
        f"{json.dumps(record.get('candidate_tools', []), ensure_ascii=False, indent=2)}\n\n"
        "Current user query:\n"
        f"{record.get('query', '')}\n\n"
        "Return exactly one JSON object and no extra text:\n"
        '{"tool_name": "<one candidate tool_name>", "parameters": {...}}\n'
    )


def evaluate_predictions(predictions: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(predictions)
    tool_correct = 0
    parameter_correct = 0
    parse_failures = 0
    for row in rows:
        gt = row.get("api_call_ground_truth", {})
        pred = row.get("parsed_response", {})
        if not isinstance(pred, dict):
            pred = {}
        if not pred:
            parse_failures += 1
        tool_correct += int(gt.get("tool_name") == pred.get("tool_name"))
        parameter_correct += int(gt.get("parameters") == pred.get("parameters"))
    n = len(rows)
    return {
        "n": n,
        "tool_correct": tool_correct,
        "parameter_correct": parameter_correct,
        "tool_accuracy": tool_correct / n if n else 0.0,
        "parameter_accuracy": parameter_correct / n if n else 0.0,
        "parse_failures": parse_failures,
        "refined_count": 0,
    }


class VllmBatchRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM

        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        self.llm = LLM(
            model=args.model,
            dtype=args.dtype,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_input_tokens
            + max(args.memory_max_new_tokens, args.inference_max_new_tokens),
            trust_remote_code=True,
        )

    def generate_chat(
        self,
        messages_list: List[List[Dict[str, str]]],
        max_new_tokens: int,
        temperature: float,
        label: str,
    ) -> List[Dict[str, Any]]:
        from vllm import SamplingParams

        sampling = SamplingParams(temperature=temperature, max_tokens=max_new_tokens)
        outputs: List[Dict[str, Any]] = []
        total = len(messages_list)
        for start in range(0, total, self.args.batch_size):
            batch_messages = messages_list[start : start + self.args.batch_size]
            prompts = []
            for messages in batch_messages:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                encoded = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.args.max_input_tokens,
                    add_special_tokens=False,
                )
                prompts.append(self.tokenizer.decode(encoded["input_ids"], skip_special_tokens=False))
            started = time.time()
            generations = self.llm.generate(prompts, sampling)
            elapsed = time.time() - started
            for generation in generations:
                response = generation.outputs[0].text.strip() if generation.outputs else ""
                outputs.append({"response": response, "elapsed_seconds": elapsed / max(len(prompts), 1)})
            print(f"{label} generated {min(start + len(prompts), total)}/{total}")
        return outputs


def finalize_pending(pending: PendingSession, candidate_pref: str) -> None:
    final_pref = candidate_pref if candidate_pref not in {"", "{}"} else "None"
    pending.state.implicit_pref = final_pref
    pending.state.accumulated_dialogue = pending.full_dialogue
    pending.state.accumulated_api_calls = pending.full_api_calls
    pending.state.session_count = pending.session_index + 1
    try:
        final_pref_obj = json.loads(final_pref) if final_pref not in {"None", "{}"} else {}
    except json.JSONDecodeError:
        final_pref_obj = {}
    pending.state.evolution_log.append(
        {
            "session_index": pending.session_index + 1,
            "memory_mode": "verified_refine",
            "refinement_process": pending.session_attempts,
            "final_preference_at_session": final_pref_obj,
        }
    )
    pending.finalized = True


def generate_memory_records(
    records: List[Dict[str, Any]],
    args: argparse.Namespace,
    runner: VllmBatchRunner,
) -> List[Dict[str, Any]]:
    sessions_by_id = {record["example_id"]: build_sessions(record) for record in records}
    states = {record["example_id"]: MemoryState() for record in records}
    max_sessions = max((len(sessions) for sessions in sessions_by_id.values()), default=0)

    for session_idx in range(max_sessions):
        pendings: List[PendingSession] = []
        for record in records:
            sessions = sessions_by_id[record["example_id"]]
            if session_idx >= len(sessions):
                continue
            state = states[record["example_id"]]
            session = sessions[session_idx]
            dialogue_text = format_dialogue(session.get("dialogue", []))
            current_session_apis = [
                f"[Session {session_idx + 1}] {api}" for api in session.get("api_call", [])
            ]
            full_api_calls = state.accumulated_api_calls + current_session_apis
            pending = PendingSession(
                record=record,
                state=state,
                session_index=session_idx,
                full_dialogue=(
                    state.accumulated_dialogue
                    + f"\n=== Session {session_idx + 1} ===\n"
                    + dialogue_text
                ),
                full_api_calls=full_api_calls,
                full_api_str="\n".join(full_api_calls) if full_api_calls else "No API calls recorded.",
                candidate_pref=state.implicit_pref if state.implicit_pref != "{}" else "None",
            )
            pendings.append(pending)

        unresolved = pendings
        for attempt in range(args.max_retries):
            if not unresolved:
                break

            gen_messages = []
            previous_drafts = []
            for pending in unresolved:
                previous_draft = pending.candidate_pref if attempt > 0 else None
                previous_drafts.append(previous_draft)
                system_prompt, user_prompt = build_generation_messages(pending, previous_draft)
                gen_messages.append(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                )

            gen_outputs = runner.generate_chat(
                gen_messages,
                args.memory_max_new_tokens,
                args.memory_temperature,
                label=f"memory session={session_idx + 1} attempt={attempt + 1}",
            )

            parsed_rows = []
            next_unresolved = []
            for pending, previous_draft, output in zip(unresolved, previous_drafts, gen_outputs):
                candidate_pref_json = parse_json_object(output["response"])
                if not candidate_pref_json:
                    pending.session_attempts.append(
                        {
                            "step": attempt + 1,
                            "mode": "verified_refine",
                            "stage": "initial_generation" if attempt == 0 else "verifier_refinement",
                            "draft_preference": {},
                            "parse_failed": True,
                            "is_valid": None,
                            "verifier_feedback": "Failed to parse generator output",
                        }
                    )
                    if attempt == args.max_retries - 1:
                        finalize_pending(pending, pending.candidate_pref)
                    else:
                        next_unresolved.append(pending)
                    continue

                candidate_pref = json.dumps(candidate_pref_json, ensure_ascii=False, indent=2)
                pending.candidate_pref = candidate_pref
                parsed_rows.append((pending, candidate_pref_json, candidate_pref))

            if parsed_rows:
                verifier_messages = []
                for pending, _, candidate_pref in parsed_rows:
                    system_prompt, user_prompt = build_verifier_messages(pending, candidate_pref)
                    verifier_messages.append(
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ]
                    )
                verifier_outputs = runner.generate_chat(
                    verifier_messages,
                    args.verifier_max_new_tokens,
                    args.verifier_temperature,
                    label=f"verify session={session_idx + 1} attempt={attempt + 1}",
                )

                for (pending, candidate_pref_json, candidate_pref), output in zip(
                    parsed_rows, verifier_outputs
                ):
                    verifier_json = parse_json_object(output["response"])
                    is_valid = bool(verifier_json.get("valid", False)) if verifier_json else False
                    feedback = str(verifier_json.get("feedback", "")) if verifier_json else "Failed to parse verifier output"
                    pending.session_attempts.append(
                        {
                            "step": attempt + 1,
                            "mode": "verified_refine",
                            "stage": "initial_generation" if attempt == 0 else "verifier_refinement",
                            "draft_preference": candidate_pref_json,
                            "parse_failed": False,
                            "is_valid": is_valid,
                            "verifier_feedback": feedback,
                            "verifier_output": verifier_json,
                        }
                    )
                    if is_valid or attempt == args.max_retries - 1:
                        finalize_pending(pending, candidate_pref)
                    else:
                        pending.feedback = feedback
                        next_unresolved.append(pending)

            unresolved = [pending for pending in next_unresolved if not pending.finalized]

        for pending in unresolved:
            if not pending.finalized:
                finalize_pending(pending, pending.candidate_pref)

    memory_records = []
    for record in records:
        state = states[record["example_id"]]
        memory_records.append(
            {
                "example_id": record["example_id"],
                "source_index": record["source_index"],
                "history_type": record["history_type"],
                "memory_mode": (
                    "experiments4_style_verified_refine"
                    if args.method_style == "experiment4_style"
                    else "experiments5_original_verified_refine"
                ),
                "final_implicit_preference": state.implicit_pref,
                "final_accumulated_implicit_preferences": [],
                "final_accumulated_api_calls": state.accumulated_api_calls,
                "total_sessions_processed": state.session_count,
                "preference_evolution_history": state.evolution_log,
                "petoolbench_metadata": {
                    "history_length": len(record.get("history", []) or []),
                    "candidate_tool_count": len(record.get("candidate_tools", []) or []),
                    "ground_truth": record.get("api_call_ground_truth"),
                },
            }
        )
    return memory_records


def generate_predictions(
    records: List[Dict[str, Any]],
    memory_records: List[Dict[str, Any]],
    args: argparse.Namespace,
    runner: VllmBatchRunner,
) -> List[Dict[str, Any]]:
    memory_by_id = {row["example_id"]: row for row in memory_records}
    messages = [
        [
            {
                "role": "user",
                "content": build_final_prompt(
                    record,
                    memory_by_id[record["example_id"]],
                    args,
                ),
            }
        ]
        for record in records
    ]
    outputs = runner.generate_chat(
        messages,
        args.inference_max_new_tokens,
        args.inference_temperature,
        label="original-ours inference",
    )
    predictions = []
    for record, output in zip(records, outputs):
        memory = memory_by_id[record["example_id"]]
        parsed = parse_json_object(output["response"])
        predictions.append(
            {
                "example_id": record["example_id"],
                "source_index": record["source_index"],
                "history_type": record["history_type"],
                "query": record["query"],
                "api_call_ground_truth": record["api_call_ground_truth"],
                "memory_mode": memory.get("memory_mode"),
                "final_implicit_preference": memory.get("final_implicit_preference"),
                "response": output["response"],
                "parsed_response": parsed,
                "elapsed_seconds": output["elapsed_seconds"],
            }
        )
    return predictions


def load_existing_memory(memory_dir: Path, history_type: str) -> Optional[List[Dict[str, Any]]]:
    path = memory_dir / f"memory_{history_type}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    return rows if isinstance(rows, list) else None


def run(args: argparse.Namespace) -> None:
    configure_method_style(args)
    output_dir = args.output_dir / args.run_name
    memory_dir = output_dir / "memory"
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_dir.mkdir(parents=True, exist_ok=True)

    runner = VllmBatchRunner(args)
    summaries: Dict[str, Dict[str, Any]] = {}
    for history_type in args.history_types:
        records = normalize_rows(args.dataset_dir, history_type, args.limit)
        if args.start_index or args.end_index is not None:
            records = records[args.start_index : args.end_index]
        memory_records = None if args.force_memory else load_existing_memory(memory_dir, history_type)
        if memory_records is None:
            print(f"generating original experiments5 memory {history_type} n={len(records)}")
            memory_records = generate_memory_records(records, args, runner)
            write_json(memory_dir / f"memory_{history_type}.json", memory_records)
        else:
            print(f"using cached memory {history_type} n={len(memory_records)}")

        print(f"generating original experiments5 final inference {history_type} n={len(records)}")
        predictions = generate_predictions(records, memory_records, args, runner)
        summary = evaluate_predictions(predictions)
        summaries[history_type] = summary
        write_json(output_dir / f"predictions_{history_type}.json", predictions)
        write_json(output_dir / f"summary_{history_type}.json", summary)
        print(history_type, json.dumps(summary, ensure_ascii=False))

    aggregate = {
        "model": args.model,
        "method": (
            "petool_experiment4_style_ours_memory_verified_refine"
            if args.method_style == "experiment4_style"
            else "petool_experiments5_original_ours_memory_verified_refine"
        ),
        "run_name": args.run_name,
        "limit": args.limit,
        "history_types": args.history_types,
        "start_index": args.start_index,
        "end_index": args.end_index,
        "memory_mode": "verified_refine",
        "method_style": args.method_style,
        "memory_prompt_source": (
            "experiments4/ours_memory/Preference_Memory_step1_LATENTPREF_VLLM.py + prompt_update3.py"
            if args.method_style == "experiment4_style"
            else "experiments5/src/prompts.py LATENT_PREF_*"
        ),
        "inference_prompt_source": (
            "PEToolBench JSON-tool-call adaptation of experiments4 Preference_Memory_step2_ACTION_* memory_api prompt"
            if args.method_style == "experiment4_style"
            else "PEToolBench JSON-tool-call adaptation of experiments5 memory_api"
        ),
        "max_retries": args.max_retries,
        "memory_max_new_tokens": args.memory_max_new_tokens,
        "verifier_max_new_tokens": args.verifier_max_new_tokens,
        "summaries": summaries,
        "overall": combine_summaries(summaries),
    }
    write_json(output_dir / "summary_all.json", aggregate)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run original experiments5 ours_memory on PEToolBench.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset_test"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-name", default="petool_original_ours_memory_qwen25_full")
    parser.add_argument(
        "--method-style",
        choices=["experiment5_original", "experiment4_style"],
        default="experiment5_original",
    )
    parser.add_argument("--history-types", nargs="+", choices=["p", "r", "c"], default=["p", "r", "c"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--force-memory", action="store_true")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-input-tokens", type=int, default=5000)
    parser.add_argument("--memory-max-new-tokens", type=int, default=512)
    parser.add_argument("--verifier-max-new-tokens", type=int, default=256)
    parser.add_argument("--inference-max-new-tokens", type=int, default=160)
    parser.add_argument("--memory-temperature", type=float, default=0.4)
    parser.add_argument("--verifier-temperature", type=float, default=0.0)
    parser.add_argument("--inference-temperature", type=float, default=0.0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
