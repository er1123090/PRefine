"""
Our Memory — Step 1: Latent Preference Extraction.

Processes each user's session history through a generate → verify → refine loop
and writes per-user preference records to a JSONL file.

Supports OpenAI (including vLLM-hosted models) and Google Gemini providers.
"""

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from openai import AsyncOpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.prompts import (
    LATENT_PREF_SYSTEM_PROMPT,
    LATENT_PREF_INITIAL_PROMPT,
    LATENT_PREF_VERIFIER_PROMPT,
    LATENT_PREF_REFINEMENT_PROMPT,
)


LATENT_PREF_BLIND_REFINEMENT_PROMPT = """
### Task: Refine Preference Without Verifier Feedback

Review the previous preference abstraction against the accumulated dialogue and API-call evidence already provided in the system context.

You must:
- Preserve only claims that are directly supported by the logs.
- Increase abstraction if details are over-specified.
- Keep the preference actionable for future API argument selection.
- Do NOT add unsupported details.
- Do NOT rely on external verifier feedback; no verifier feedback is available.

### Input
Draft Preference:
"{previous_draft}"

### Output Format (JSON)
{{
  "reasoning": "How the draft was tightened against the evidence",
  "implicit_pref": "The revised preference constraint"
}}
"""

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Memory state
# ---------------------------------------------------------------------------

@dataclass
class MemoryState:
    implicit_pref: str = "{}"
    accumulated_dialogue: str = ""
    accumulated_api_calls: List[str] = field(default_factory=list)
    session_count: int = 0
    evolution_log: List[Dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Preference aggregator
# ---------------------------------------------------------------------------

class PreferenceAggregator:
    def __init__(self, model: str, provider: str, api_key: str,
                 api_base: Optional[str] = None, max_retries: int = 3,
                 memory_mode: str = "verified_refine"):
        self.model = model
        self.provider = provider.lower()
        self.api_key = api_key
        self.api_base = api_base
        self.max_retries = max_retries
        self.memory_mode = memory_mode

        valid_memory_modes = {
            "verified_refine",
            "generation_only",
            "blind_refine_1",
            "blind_refine_2",
            "blind_refine_3",
        }
        if self.memory_mode not in valid_memory_modes:
            raise ValueError(
                f"Unsupported memory_mode={self.memory_mode}. "
                f"Choose one of {sorted(valid_memory_modes)}."
            )

        if self.provider == "openai":
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.api_base)
        elif self.provider == "google":
            if not GOOGLE_AVAILABLE:
                raise ImportError("google-generativeai is required for Google provider.")
            genai.configure(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    async def _call_llm(self, system_prompt: str, user_prompt: str,
                        temperature: float = 0.0) -> str:
        if self.provider == "openai":
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=2048,
                    response_format={"type": "json_object"},
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"[OpenAI Error] {e}")
                return "{}"

        elif self.provider == "google":
            try:
                model = genai.GenerativeModel(
                    model_name=self.model,
                    system_instruction=system_prompt,
                )
                generation_config = genai.types.GenerationConfig(
                    candidate_count=1,
                    temperature=temperature,
                    max_output_tokens=2048,
                    response_mime_type="application/json",
                )
                response = await model.generate_content_async(
                    user_prompt, generation_config=generation_config
                )
                return response.text
            except Exception as e:
                print(f"[Google Error] {e}")
                return "{}"

        return "{}"

    async def update_memory(self, current_state: MemoryState,
                            session_dialogue: str,
                            session_api_calls: List[str]) -> MemoryState:
        session_idx = current_state.session_count + 1
        full_dialogue = (
            current_state.accumulated_dialogue
            + f"\n=== Session {session_idx} ===\n"
            + session_dialogue
        )
        current_session_apis = [f"[Session {session_idx}] {api}" for api in session_api_calls]
        full_api_list = current_state.accumulated_api_calls + current_session_apis
        full_api_str = "\n".join(full_api_list) if full_api_list else "No API calls recorded."

        candidate_pref = current_state.implicit_pref
        if candidate_pref == "{}":
            candidate_pref = "None"

        feedback = ""
        updated_log = list(current_state.evolution_log)
        session_attempts_log = []
        candidate_pref_str = (
            candidate_pref if isinstance(candidate_pref, str)
            else json.dumps(candidate_pref)
        )

        if self.memory_mode != "verified_refine":
            blind_refine_steps = {
                "generation_only": 0,
                "blind_refine_1": 1,
                "blind_refine_2": 2,
                "blind_refine_3": 3,
            }[self.memory_mode]

            for i in range(blind_refine_steps + 1):
                candidate_pref_json = await self._generate_preference(
                    prev_implicit=current_state.implicit_pref,
                    full_dialogue=full_dialogue,
                    full_api_calls=full_api_str,
                    previous_draft=candidate_pref if i > 0 else None,
                    force_blind_refine=i > 0,
                )

                if not candidate_pref_json:
                    session_attempts_log.append({
                        "step": i + 1,
                        "mode": self.memory_mode,
                        "stage": "blind_refinement" if i > 0 else "initial_generation",
                        "draft_preference": {},
                        "parse_failed": True,
                        "is_valid": None,
                        "verifier_feedback": "",
                    })
                    continue

                candidate_pref_str = json.dumps(candidate_pref_json, ensure_ascii=False, indent=2)
                candidate_pref = candidate_pref_str
                session_attempts_log.append({
                    "step": i + 1,
                    "mode": self.memory_mode,
                    "stage": "blind_refinement" if i > 0 else "initial_generation",
                    "draft_preference": candidate_pref_json,
                    "parse_failed": False,
                    "is_valid": None,
                    "verifier_feedback": "",
                })

            updated_log.append({
                "session_index": session_idx,
                "memory_mode": self.memory_mode,
                "refinement_process": session_attempts_log,
                "final_preference_at_session": (
                    json.loads(candidate_pref)
                    if isinstance(candidate_pref, str) and candidate_pref not in ["None", "{}"]
                    else {}
                ),
            })

            return MemoryState(
                implicit_pref=candidate_pref,
                accumulated_dialogue=full_dialogue,
                accumulated_api_calls=full_api_list,
                session_count=session_idx,
                evolution_log=updated_log,
            )

        for i in range(self.max_retries):
            candidate_pref_json = await self._generate_preference(
                prev_implicit=current_state.implicit_pref,
                full_dialogue=full_dialogue,
                full_api_calls=full_api_str,
                feedback=feedback,
                previous_draft=candidate_pref if i > 0 else None,
            )

            if not candidate_pref_json:
                continue

            candidate_pref_str = json.dumps(candidate_pref_json, ensure_ascii=False, indent=2)

            is_valid, new_feedback, verifier_input, verifier_output_json = await self._verify_preference(
                full_dialogue=full_dialogue,
                full_api_calls=full_api_str,
                candidate_pref=candidate_pref_str,
            )

            session_attempts_log.append({
                "step": i + 1,
                "mode": self.memory_mode,
                "stage": "verifier_refinement" if i > 0 else "initial_generation",
                "draft_preference": candidate_pref_json,
                "parse_failed": False,
                "is_valid": is_valid,
                "verifier_feedback": new_feedback,
                "verifier_input": verifier_input,
                "verifier_output": verifier_output_json,
            })

            if is_valid:
                candidate_pref = candidate_pref_str
                break
            else:
                feedback = new_feedback
                candidate_pref = candidate_pref_str

        updated_log.append({
            "session_index": session_idx,
            "memory_mode": self.memory_mode,
            "refinement_process": session_attempts_log,
            "final_preference_at_session": (
                json.loads(candidate_pref)
                if isinstance(candidate_pref, str) and candidate_pref not in ["None", "{}"]
                else {}
            ),
        })

        return MemoryState(
            implicit_pref=candidate_pref,
            accumulated_dialogue=full_dialogue,
            accumulated_api_calls=full_api_list,
            session_count=session_idx,
            evolution_log=updated_log,
        )

    async def _generate_preference(self, prev_implicit: str, full_dialogue: str,
                                   full_api_calls: str, feedback: str = "",
                                   previous_draft: Optional[str] = None,
                                   force_blind_refine: bool = False) -> dict:
        context_prompt = LATENT_PREF_SYSTEM_PROMPT.format(
            prev_implicit=prev_implicit or "None",
            full_dialogue=full_dialogue,
            full_api_calls=full_api_calls,
        )
        if previous_draft and force_blind_refine:
            task_prompt = LATENT_PREF_BLIND_REFINEMENT_PROMPT.format(
                previous_draft=previous_draft,
            )
        elif previous_draft and feedback:
            task_prompt = LATENT_PREF_REFINEMENT_PROMPT.format(
                previous_draft=previous_draft,
                feedback=feedback,
            )
        else:
            task_prompt = LATENT_PREF_INITIAL_PROMPT

        content = await self._call_llm(context_prompt, task_prompt, temperature=0.4)
        return self._parse_json(content)

    async def _verify_preference(self, full_dialogue: str, full_api_calls: str,
                                 candidate_pref: str) -> Tuple[bool, str, str, Dict]:
        prompt = LATENT_PREF_VERIFIER_PROMPT.format(
            full_dialogue=full_dialogue,
            full_api_calls=full_api_calls,
            candidate_pref=candidate_pref,
        )
        content = await self._call_llm(
            "You are a Preference Verification Module. Output JSON only.",
            prompt,
            temperature=0.0,
        )
        res_json = self._parse_json(content)
        if not res_json:
            return False, "Failed to parse verifier output", prompt, {}
        return res_json.get("valid", False), res_json.get("feedback", ""), prompt, res_json

    def _parse_json(self, content: str) -> dict:
        if not content:
            return {}
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        if "```json" in content:
            content = re.sub(r"```json\s*", "", content)
            content = re.sub(r"\s*```", "", content)
        elif "```" in content:
            content = content.replace("```", "")
        start_idx = content.find("{")
        end_idx = content.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = content[start_idx: end_idx + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                try:
                    cleaned = re.sub(r",\s*([\]}])", r"\1", json_str)
                    return json.loads(cleaned)
                except Exception:
                    return {}
        return {}


# ---------------------------------------------------------------------------
# Processing logic
# ---------------------------------------------------------------------------

def _format_dialogue(dialogue_list: List[Dict]) -> str:
    return "\n".join(
        f"{t.get('role', 'User')}: {t.get('message', '')}"
        for t in dialogue_list
    )


def _load_dataset(filepath: str) -> List[Dict]:
    if not os.path.exists(filepath):
        print(f"[Error] File not found: {filepath}")
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


async def process_single_user(
    example: Dict,
    aggregator: PreferenceAggregator,
    file_lock: asyncio.Lock,
    output_file: str,
    verifier_file: str,
    refinement_file: str,
    semaphore: asyncio.Semaphore,
    pbar: tqdm,
) -> None:
    async with semaphore:
        example_id = example.get("example_id", "unknown")
        sessions = example.get("sessions", [])
        current_state = MemoryState()

        for session in sessions:
            dialogue_text = _format_dialogue(session.get("dialogue", []))
            api_calls = session.get("api_call", [])
            current_state = await aggregator.update_memory(current_state, dialogue_text, api_calls)

        result_record = {
            "example_id": example_id,
            "memory_mode": aggregator.memory_mode,
            "final_implicit_preference": current_state.implicit_pref,
            "final_accumulated_api_calls": current_state.accumulated_api_calls,
            "total_sessions_processed": current_state.session_count,
            "preference_evolution_history": current_state.evolution_log,
        }

        refinement_logs = []
        verifier_logs = []
        for log_entry in current_state.evolution_log:
            session_idx = log_entry.get("session_index")
            for step_info in log_entry.get("refinement_process", []):
                base_meta = {"example_id": example_id, "session_index": session_idx,
                             "step": step_info.get("step")}
                r_log = {**base_meta,
                         "memory_mode": step_info.get("mode"),
                         "stage": step_info.get("stage"),
                         "draft_preference": step_info.get("draft_preference"),
                         "parse_failed": step_info.get("parse_failed")}
                refinement_logs.append(json.dumps(r_log, ensure_ascii=False))
                if "verifier_input" in step_info or "verifier_output" in step_info:
                    v_log = {**base_meta, "is_valid": step_info.get("is_valid"),
                             "verifier_input": step_info.get("verifier_input"),
                             "verifier_output": step_info.get("verifier_output")}
                    verifier_logs.append(json.dumps(v_log, ensure_ascii=False))

        async with file_lock:
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result_record, ensure_ascii=False) + "\n")
            if refinement_logs:
                with open(refinement_file, "a", encoding="utf-8") as f:
                    for line in refinement_logs:
                        f.write(line + "\n")
            if verifier_logs:
                with open(verifier_file, "a", encoding="utf-8") as f:
                    for line in verifier_logs:
                        f.write(line + "\n")

        pbar.update(1)


async def run_pipeline(args: argparse.Namespace) -> None:
    api_key = args.api_key
    if args.provider == "openai":
        api_key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    elif args.provider == "google":
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("[Error] Google API Key is missing.")
            return

    dataset = _load_dataset(args.input)
    if not dataset:
        return

    try:
        aggregator = PreferenceAggregator(
            model=args.model,
            provider=args.provider,
            api_key=api_key,
            api_base=args.api_base,
            max_retries=args.max_retries,
            memory_mode=args.memory_mode,
        )
    except Exception as e:
        print(f"[Error] {e}")
        return

    semaphore = asyncio.Semaphore(args.concurrency)
    file_lock = asyncio.Lock()

    for fpath in [args.output, args.verifier_output, args.refinement_output]:
        os.makedirs(os.path.dirname(fpath) or ".", exist_ok=True)
        open(fpath, "w").close()

    pbar = tqdm(total=len(dataset), desc="Extracting preferences")
    tasks = [
        asyncio.create_task(
            process_single_user(
                example, aggregator, file_lock,
                args.output, args.verifier_output, args.refinement_output,
                semaphore, pbar,
            )
        )
        for example in dataset
    ]
    await asyncio.gather(*tasks)
    pbar.close()

    if args.provider == "openai":
        await aggregator.client.close()

    print(f"\nResults -> {args.output}")
    print(f"Verifier log -> {args.verifier_output}")
    print(f"Refinement log -> {args.refinement_output}")


if __name__ == "__main__":
    _ROOT = os.path.join(os.path.dirname(__file__), "..", "..")

    parser = argparse.ArgumentParser(description="Step 1: Latent preference extraction.")
    parser.add_argument("--input", default=os.path.join(_ROOT, "data", "dev.json"))
    parser.add_argument("--output", default="outputs/our_memory/preferences.jsonl")
    parser.add_argument("--verifier_output", default="outputs/our_memory/verifier_logs.jsonl")
    parser.add_argument("--refinement_output",
                        default="outputs/our_memory/refinement_logs.jsonl")
    parser.add_argument("--provider", default="openai", choices=["openai", "google"])
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--api_base", default=None)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--concurrency", type=int, default=30)
    parser.add_argument("--max_retries", type=int, default=3,
                        help="Max verifier-refine loop iterations per session (0 = no update).")
    parser.add_argument("--memory_mode",
                        choices=[
                            "verified_refine",
                            "generation_only",
                            "blind_refine_1",
                            "blind_refine_2",
                            "blind_refine_3",
                        ],
                        default="verified_refine",
                        help=(
                            "Memory generation mode. verified_refine preserves the existing "
                            "generate -> verifier -> feedback-refine loop; generation_only runs "
                            "one generation with no verifier/refinement; blind_refine_1, "
                            "blind_refine_2, and blind_refine_3 run unconditional refinements "
                            "without verifier feedback."
                        ))
    args = parser.parse_args()

    asyncio.run(run_pipeline(args))
