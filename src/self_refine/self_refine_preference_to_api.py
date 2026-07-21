import argparse
import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from openai import AsyncOpenAI

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False


RELEASE_ROOT = Path(__file__).resolve().parents[2]


PREF_SYSTEM_PROMPT = (
    "You are a preference inference agent for tool-calling systems. "
    "Infer latent user preferences that can constrain future API arguments. "
    "Output JSON only."
)

PREF_INITIAL_PROMPT = """
### Task
Infer a concise, stable preference from the interaction history.

Rules:
- Do not list every slot value or restate the entire dialogue.
- Provide a preference that can guide future API argument selection.
- If evidence is insufficient, say "insufficient evidence".

### Evidence
Dialogue:
{dialogue}

API Calls:
{api_calls}

### Output JSON
{{
  "reasoning": "short reasoning grounded in the evidence",
  "implicit_pref": "single-sentence constraint"
}}
"""

PREF_CRITIQUE_PROMPT = """
### Task
Critique the preference for over-specificity, hallucination, or non-actionability.

### Evidence
Dialogue:
{dialogue}

API Calls:
{api_calls}

### Candidate Preference
{candidate_pref}

### Output JSON
{{
  "issues": ["issue1", "issue2"],
  "revised_pref": "a corrected preference sentence or 'insufficient evidence'"
}}
"""

PREF_REVISION_PROMPT = """
### Task
Revise the preference based on critique. Remove unsupported claims and keep it actionable.

### Original Preference
{candidate_pref}

### Critique
{critique}

### Output JSON
{{
  "reasoning": "what changed and why",
  "implicit_pref": "final preference sentence"
}}
"""

API_CALL_PROMPT = """
You are a tool-calling agent. Produce exactly ONE API call.
Use ONLY the schema fields for the selected function.
Do not invent slots or values. If a value is unknown, omit the slot.

Schema:
{schema}

Inferred Preference:
{implicit_pref}

User Utterance:
{user_utterance}

Dialogue Context:
{dialogue}

Output Format:
FunctionName(slot="value", ...)
"""


@dataclass
class LLMConfig:
    provider: str
    model: str
    api_key: str
    api_base: Optional[str]


class SelfRefineClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.provider = config.provider.lower()
        self.model = config.model
        self.api_key = config.api_key
        self.api_base = config.api_base

        if self.provider == "openai":
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.api_base)
        elif self.provider == "google":
            if not GOOGLE_AVAILABLE:
                raise ImportError("google-generativeai library is required for Google provider.")
            genai.configure(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    async def call_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> Dict[str, Any]:
        if self.provider == "openai":
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    max_tokens=2048,
                    response_format={"type": "json_object"}
                )
                return parse_json(response.choices[0].message.content)
            except Exception:
                return {}

        if self.provider == "google":
            try:
                model = genai.GenerativeModel(
                    model_name=self.model,
                    system_instruction=system_prompt
                )
                generation_config = genai.types.GenerationConfig(
                    candidate_count=1,
                    temperature=temperature,
                    max_output_tokens=2048,
                    response_mime_type="application/json"
                )
                response = await model.generate_content_async(
                    user_prompt,
                    generation_config=generation_config
                )
                return parse_json(response.text)
            except Exception:
                return {}

        return {}

    async def call_text(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        if self.provider == "openai":
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    max_tokens=512
                )
                return response.choices[0].message.content.strip()
            except Exception:
                return ""

        if self.provider == "google":
            try:
                model = genai.GenerativeModel(
                    model_name=self.model,
                    system_instruction=system_prompt
                )
                generation_config = genai.types.GenerationConfig(
                    candidate_count=1,
                    temperature=temperature,
                    max_output_tokens=512
                )
                response = await model.generate_content_async(
                    user_prompt,
                    generation_config=generation_config
                )
                return (response.text or "").strip()
            except Exception:
                return ""

        return ""

    async def close(self) -> None:
        if self.provider == "openai":
            await self.client.close()


def parse_json(content: str) -> Dict[str, Any]:
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
        json_str = content[start_idx : end_idx + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                cleaned = re.sub(r",\s*([\]}])", r"\1", json_str)
                return json.loads(cleaned)
            except Exception:
                return {}
    return {}


def format_dialogue(session: Dict[str, Any]) -> str:
    turns = session.get("dialogue", [])
    lines = []
    for turn in turns:
        role = turn.get("role", "User")
        msg = turn.get("message", "")
        lines.append(f"{role}: {msg}")
    return "\n".join(lines)


def get_target_utterance(session: Dict[str, Any]) -> str:
    pref_utt = session.get("preference_utterance", [])
    if isinstance(pref_utt, list) and pref_utt:
        return str(pref_utt[0])

    turns = session.get("dialogue", [])
    for turn in reversed(turns):
        if str(turn.get("role", "")).lower() == "user":
            return turn.get("message", "")
    return ""


def extract_api_calls(session: Dict[str, Any]) -> str:
    api_calls = session.get("api_call", [])
    if not api_calls:
        return "None"
    if isinstance(api_calls, list):
        return "\n".join(api_calls)
    return str(api_calls)


def load_dataset(filepath: str) -> List[Dict[str, Any]]:
    if not os.path.exists(filepath):
        print(f"[Error] File not found: {filepath}")
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data if isinstance(data, list) else [data]


def load_schema(filepath: str) -> List[Dict[str, Any]]:
    if not filepath or not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def compact_schema(tools_schema: List[Dict[str, Any]]) -> str:
    if not tools_schema:
        return "No schema provided."
    compact = []
    for tool in tools_schema:
        fn = tool.get("function", {})
        name = fn.get("name")
        params = fn.get("parameters", {}).get("properties", {})
        compact.append({"name": name, "parameters": params})
    return json.dumps(compact, ensure_ascii=False, indent=2)


async def self_refine_preference(
    client: SelfRefineClient,
    dialogue: str,
    api_calls: str,
    rounds: int
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    logs = []

    initial_prompt = PREF_INITIAL_PROMPT.format(dialogue=dialogue, api_calls=api_calls)
    pref = await client.call_json(PREF_SYSTEM_PROMPT, initial_prompt, temperature=0.4)
    if not pref:
        pref = {"reasoning": "", "implicit_pref": "insufficient evidence"}

    logs.append({"stage": "initial", "output": pref})

    current_pref = pref
    for idx in range(rounds):
        critique_prompt = PREF_CRITIQUE_PROMPT.format(
            dialogue=dialogue,
            api_calls=api_calls,
            candidate_pref=json.dumps(current_pref, ensure_ascii=False)
        )
        critique = await client.call_json(PREF_SYSTEM_PROMPT, critique_prompt, temperature=0.0)
        logs.append({"stage": f"critique_{idx+1}", "output": critique})

        revision_prompt = PREF_REVISION_PROMPT.format(
            candidate_pref=json.dumps(current_pref, ensure_ascii=False),
            critique=json.dumps(critique, ensure_ascii=False)
        )
        revised = await client.call_json(PREF_SYSTEM_PROMPT, revision_prompt, temperature=0.2)
        if revised:
            current_pref = revised
        logs.append({"stage": f"revision_{idx+1}", "output": current_pref})

    return current_pref, logs


def normalize_api_call(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*\(.*\)", text, flags=re.S)
    if match:
        return match.group(0).strip()
    return text.strip()


async def process_example(
    example: Dict[str, Any],
    client: SelfRefineClient,
    schema_str: str,
    rounds: int
) -> List[Dict[str, Any]]:
    outputs = []
    example_id = example.get("example_id", "unknown")
    sessions = example.get("sessions", [])

    for idx, session in enumerate(sessions, start=1):
        dialogue = format_dialogue(session)
        api_calls = extract_api_calls(session)
        user_utterance = get_target_utterance(session)

        pref, pref_logs = await self_refine_preference(client, dialogue, api_calls, rounds)
        implicit_pref = pref.get("implicit_pref", "insufficient evidence")

        api_prompt = API_CALL_PROMPT.format(
            schema=schema_str,
            implicit_pref=implicit_pref,
            user_utterance=user_utterance,
            dialogue=dialogue
        )
        api_call_text = await client.call_text("You are a tool-calling agent.", api_prompt, temperature=0.0)
        api_call = normalize_api_call(api_call_text)

        outputs.append({
            "example_id": example_id,
            "session_index": idx,
            "user_utterance": user_utterance,
            "implicit_pref": implicit_pref,
            "api_call": api_call,
            "self_refine_logs": pref_logs
        })

    return outputs


async def run(args: argparse.Namespace) -> None:
    api_key = args.api_key
    if args.provider == "openai":
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY") or "EMPTY"
    elif args.provider == "google":
        if not api_key:
            api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("[Error] Google API Key is missing. Please set GOOGLE_API_KEY or pass --api_key.")
            return

    dataset = load_dataset(args.input)
    if not dataset:
        return

    tools_schema = load_schema(args.schema)
    schema_str = compact_schema(tools_schema)

    config = LLMConfig(
        provider=args.provider,
        model=args.model,
        api_key=api_key,
        api_base=args.api_base
    )
    client = SelfRefineClient(config)

    semaphore = asyncio.Semaphore(args.concurrency)
    file_lock = asyncio.Lock()

    with open(args.output, "w", encoding="utf-8") as f:
        pass

    pbar = tqdm(total=len(dataset), desc="Processing", unit="example")

    async def worker(example: Dict[str, Any]) -> None:
        async with semaphore:
            result_rows = await process_example(example, client, schema_str, args.refine_rounds)
            async with file_lock:
                with open(args.output, "a", encoding="utf-8") as f:
                    for row in result_rows:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
            pbar.update(1)

    tasks = [asyncio.create_task(worker(ex)) for ex in dataset]
    await asyncio.gather(*tasks)
    pbar.close()

    await client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Self-refine preference inference and API call generation."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(RELEASE_ROOT / "data" / "1229_dev_6.json"),
        help="Path to input JSON file"
    )
    parser.add_argument(
        "--schema",
        type=str,
        default=str(RELEASE_ROOT / "configs" / "schema_all.json"),
        help="Path to tools schema JSON"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="self_refine_results.jsonl",
        help="Output JSONL file"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "google"],
        help="Model provider"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model name"
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default=None,
        help="API Base URL (OpenAI/vLLM only)"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key (optional if env var is set)"
    )
    parser.add_argument(
        "--refine_rounds",
        type=int,
        default=1,
        help="Number of self-refine iterations"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent examples"
    )

    args = parser.parse_args()
    asyncio.run(run(args))
