"""
Multi-call (THINK -> VERIFY -> (optional) REFINE) agent for generating GetX(...) API calls.

- Call 1 (THINK): generate draft API call(s)
- Call 2 (VERIFY): judge draft (ACCEPT/REJECT) + reason(s)
- Call 3 (REFINE): fix draft based on verification feedback

Requires:
  pip install openai
  export OPENAI_API_KEY="..."

OpenAI Python SDK uses the Responses API (client.responses.create).  [oai_citation:0‡OpenAI Platform](https://platform.openai.com/docs/api-reference/responses/create?utm_source=chatgpt.com)
"""

from __future__ import annotations

import json
import os
import re
import logging
import argparse
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI  # official SDK  [oai_citation:1‡OpenAI Platform](https://platform.openai.com/docs/libraries?utm_source=chatgpt.com)


# -----------------------------
# 0) Prompts (loaded from file)
# -----------------------------
PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "remem.prompt")

def _load_prompts(path: str = PROMPTS_PATH) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    schema = data["schema"].strip()
    def render(template: str) -> str:
        return template.replace("{schema}", schema).strip()
    return {
        "schema": schema,
        "think_system": render(data["think_system"]),
        "verify_system": render(data["verify_system"]),
        "refine_system": render(data["refine_system"]),
    }

PROMPTS = _load_prompts()
THINK_SYSTEM = PROMPTS["think_system"]
VERIFY_SYSTEM = PROMPTS["verify_system"]
REFINE_SYSTEM = PROMPTS["refine_system"]

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False

PROGRESS_LOGGER = logging.getLogger("progress")
PROGRESS_LOGGER.setLevel(logging.INFO)
PROGRESS_LOGGER.propagate = False

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
LOG_PATH = os.path.join(
    LOG_DIR,
    f"runs_{datetime.now():%y%m%d_%H%M%S}.log",
)
os.makedirs(LOG_DIR, exist_ok=True)
_formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

_file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
_file_handler.setLevel(logging.INFO)
_file_handler.setFormatter(_formatter)

_progress_handler = logging.StreamHandler()
_progress_handler.setLevel(logging.INFO)
_progress_handler.setFormatter(_formatter)

if not LOGGER.handlers:
    LOGGER.addHandler(_file_handler)

if not PROGRESS_LOGGER.handlers:
    PROGRESS_LOGGER.addHandler(_progress_handler)


# -----------------------------
# 2) Light parsing / validation helpers (cheap sanity checks)
# -----------------------------
API_CALL_RE = re.compile(r"^Get([A-Za-z]+)\((.*)\)$")
API_CALL_KV_RE = re.compile(r'(\w+)=["\']([^"\']+)["\']')

def _loads_json_maybe(text: str) -> Any:
    text = text.strip()
    # common failure mode: model wraps JSON in ```json ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)

def _ensure_list_of_str(x: Any) -> List[str]:
    if not isinstance(x, list) or any(not isinstance(s, str) for s in x):
        raise ValueError("Expected a list[str].")
    return x

def _basic_api_call_shape_ok(call: str) -> bool:
    m = API_CALL_RE.match(call.strip())
    if not m:
        return False
    # Allow empty args too: GetX()
    return True

def _extract_domains_from_calls(calls: List[str]) -> List[str]:
    domains: List[str] = []
    seen = set()
    for call in calls:
        m = API_CALL_RE.match(call.strip())
        if not m:
            continue
        domain = m.group(1)
        if domain not in seen:
            seen.add(domain)
            domains.append(domain)
    return domains

def _group_gold_by_domain(calls: List[str]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for call in calls:
        m = API_CALL_RE.match(call.strip())
        if not m:
            continue
        domain = m.group(1)
        grouped.setdefault(domain, []).append(call)
    return grouped

def _strip_get_prefix(domain: str) -> str:
    return domain[3:] if domain.startswith("Get") else domain

def _parse_api_call(call_str: str) -> Tuple[Optional[str], List[Tuple[str, str]]]:
    if "(" in call_str:
        domain = call_str.split("(", 1)[0].strip()
        try:
            args_content = call_str.split("(", 1)[1].rsplit(")", 1)[0]
        except IndexError:
            return domain, []
    else:
        domain = call_str.strip()
        args_content = ""
    matches = API_CALL_KV_RE.findall(args_content)
    return domain, matches

def _load_json_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)

def _build_pref_cases(
    example: Dict[str, Any],
    query_map: Dict[str, str],
    pref_type: str,
    pref_list: Optional[Dict[str, List[str]]] = None,
    pref_group: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []

    if pref_type == "easy":
        api_calls = example.get("api_calls", [])
        if not isinstance(api_calls, list):
            return results
        if not pref_list:
            return results
        for call_str in api_calls:
            domain, matches = _parse_api_call(call_str)
            if not domain:
                continue
            if domain not in pref_list:
                continue
            query_domain = _strip_get_prefix(domain)
            if query_domain not in query_map:
                continue
            target_slots = pref_list.get(domain, [])
            if not any(slot in target_slots for slot, _ in matches):
                continue
            if not matches:
                continue
            slots_str = [f'{slot}="{value}"' for slot, value in matches]
            ground_truth = f"{domain}({', '.join(slots_str)})"
            results.append(
                {
                    "domain": query_domain,
                    "query": query_map[query_domain],
                    "ground_truth": ground_truth,
                }
            )
        return results

    if pref_type == "medium":
        api_calls = example.get("api_calls", [])
        easy_domains = set()
        if isinstance(api_calls, list):
            for call_str in api_calls:
                domain, _ = _parse_api_call(call_str)
                if domain:
                    easy_domains.add(domain)

        prefs = example.get("api_calls_pref", [])
        if not isinstance(prefs, list) or not prefs:
            return results
        if not pref_group:
            return results

        for pref in prefs:
            if pref.get("value_group") not in pref_group:
                continue
            evidence_list = pref.get("evidence", [])
            if not isinstance(evidence_list, list):
                continue
            for evidence in evidence_list:
                domain = evidence.get("domain")
                if not domain or domain in easy_domains:
                    continue
                query_domain = _strip_get_prefix(domain)
                if query_domain not in query_map:
                    continue
                slot = evidence.get("slot")
                value = evidence.get("value")
                if slot is None or value is None:
                    continue
                val_str = _format_value(value)
                ground_truth = f'{domain}({slot}="{val_str}")'
                results.append(
                    {
                        "domain": query_domain,
                        "query": query_map[query_domain],
                        "ground_truth": ground_truth,
                    }
                )
        return results

    if pref_type == "hard":
        prefs = example.get("api_calls_pref", [])
        if not isinstance(prefs, list) or not prefs:
            return results
        if not pref_group:
            return results

        for pref in prefs:
            current_group = pref.get("value_group")
            if not current_group or current_group not in pref_group:
                continue
            used_domains = set()
            for evidence in pref.get("evidence", []):
                d = evidence.get("domain")
                if d:
                    used_domains.add(d)
            rules = pref_group[current_group].get("rules", [])
            for rule in rules:
                domain = rule.get("domain")
                if not domain or domain in used_domains:
                    continue
                query_domain = _strip_get_prefix(domain)
                if query_domain not in query_map:
                    continue
                slot = rule.get("slot")
                if slot is None:
                    continue
                value = rule.get("value")
                val_str = _format_value(value)
                ground_truth = f'{domain}({slot}="{val_str}")'
                results.append(
                    {
                        "domain": query_domain,
                        "query": query_map[query_domain],
                        "ground_truth": ground_truth,
                    }
                )
        return results

    return results

def _build_payload(
    query: str,
    sessions: Optional[Any] = None,
    derived_preferences: Optional[List[str]] = None,
    **extra: Any,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "query": query,
    }
    if sessions is not None:
        payload["sessions"] = _strip_service_fields(sessions)
    if derived_preferences is not None:
        payload["derived_preferences"] = derived_preferences
    payload.update(extra)
    return payload

def _strip_service_fields(sessions: Any) -> Any:
    if not isinstance(sessions, list):
        return sessions
    cleaned_sessions = []
    for session in sessions:
        if not isinstance(session, dict):
            cleaned_sessions.append(session)
            continue
        if "dialogue" not in session or not isinstance(session["dialogue"], list):
            cleaned_sessions.append(session)
            continue
        cleaned_dialogue = []
        for turn in session["dialogue"]:
            if isinstance(turn, dict) and "service" in turn:
                t = dict(turn)
                t.pop("service", None)
                cleaned_dialogue.append(t)
            else:
                cleaned_dialogue.append(turn)
        s = dict(session)
        s["dialogue"] = cleaned_dialogue
        cleaned_sessions.append(s)
    return cleaned_sessions


# -----------------------------
# 3) OpenAI call wrapper (Responses API)
# -----------------------------
@dataclass
class LLMConfig:
    model: str = "gpt-5-mini"  # change as you like
    temperature: float = 0.0

class MultiCallAgent:
    def __init__(self, cfg: LLMConfig, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.cfg = cfg

    def _call(self, system: str, user_payload: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Single model call.
        Uses Responses API: client.responses.create(...).  [oai_citation:2‡OpenAI Platform](https://platform.openai.com/docs/api-reference/responses/create?utm_source=chatgpt.com)
        """
        request = dict(
            model=self.cfg.model,
            input=[
                {"role": "developer", "content": system},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
        )
        if self.cfg.temperature is not None and not self.cfg.model.startswith("gpt-5"):
            request["temperature"] = self.cfg.temperature
        resp = self.client.responses.create(**request)
        usage = {}
        if hasattr(resp, "usage") and resp.usage is not None:
            try:
                usage = {
                    "input_tokens": getattr(resp.usage, "input_tokens", None),
                    "output_tokens": getattr(resp.usage, "output_tokens", None),
                    "total_tokens": getattr(resp.usage, "total_tokens", None),
                }
            except Exception:
                usage = {}
        return resp.output_text, usage

    # ---- Step 1: THINK ----
    def think(
        self,
        query: str,
        sessions: Optional[Any] = None,
    ) -> Tuple[List[str], List[str], str]:
        payload = _build_payload(
            query,
            sessions=sessions,
            instruction="Generate draft API call(s) as JSON.",
        )
        out, usage = self._call(THINK_SYSTEM, payload)
        obj = _loads_json_maybe(out)
        calls = _ensure_list_of_str(obj["api_calls"])
        derived = obj.get("derived_preferences", [])
        if derived:
            derived = _ensure_list_of_str(derived)
        # cheap sanity
        bad = [c for c in calls if not _basic_api_call_shape_ok(c)]
        if bad:
            raise ValueError(f"THINK produced invalid call shape: {bad}\nRaw:\n{out}")
        return calls, derived, out, usage

    # ---- Step 2: VERIFY ----
    def verify(
        self,
        query: str,
        draft_calls: List[str],
        sessions: Optional[Any] = None,
        derived_preferences: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, Any], str]:
        payload = _build_payload(
            query,
            sessions=sessions,
            derived_preferences=derived_preferences,
            draft_api_calls=draft_calls,
        )
        out, usage = self._call(VERIFY_SYSTEM, payload)
        obj = _loads_json_maybe(out)
        if obj.get("verdict") not in ("ACCEPT", "REJECT"):
            raise ValueError(f"VERIFY returned bad verdict. Raw:\n{out}")
        if not isinstance(obj.get("issues", []), list):
            raise ValueError(f"VERIFY returned bad issues. Raw:\n{out}")
        return obj, out, usage

    # ---- Step 3: REFINE (conditional) ----
    def refine(
        self,
        query: str,
        draft_calls: List[str],
        issues: List[Dict[str, Any]],
        sessions: Optional[Any] = None,
        derived_preferences: Optional[List[str]] = None,
    ) -> Tuple[List[str], str]:
        payload = _build_payload(
            query,
            sessions=sessions,
            derived_preferences=derived_preferences,
            draft_api_calls=draft_calls,
            verifier_issues=issues,
        )
        out, usage = self._call(REFINE_SYSTEM, payload)
        obj = _loads_json_maybe(out)
        calls = _ensure_list_of_str(obj["api_calls"])
        bad = [c for c in calls if not _basic_api_call_shape_ok(c)]
        if bad:
            raise ValueError(f"REFINE produced invalid call shape: {bad}\nRaw:\n{out}")
        return calls, out, usage

    # ---- Full pipeline ----
    def run(
        self,
        query: str,
        sessions: Optional[Any] = None,
        progress_tag: Optional[str] = None,
    ) -> Tuple[List[str], Dict[str, Any], List[str], List[Dict[str, Any]], Dict[str, int]]:
        LOGGER.info("run: query_len=%s sessions=%s", len(query), 0 if sessions is None else len(sessions))
        if progress_tag:
            PROGRESS_LOGGER.info("%s THINK", progress_tag)
        draft, derived, think_raw, think_usage = self.think(
            query,
            sessions=sessions,
        )
        token_totals = {
            "input_tokens": think_usage.get("input_tokens") or 0,
            "output_tokens": think_usage.get("output_tokens") or 0,
            "total_tokens": think_usage.get("total_tokens") or 0,
        }
        steps: List[Dict[str, Any]] = [
            {
                "step": "think",
                "iteration": 1,
                "api_calls": draft,
                "derived_preferences": derived,
                "raw_output": think_raw,
                "usage": think_usage,
            }
        ]
        if progress_tag:
            PROGRESS_LOGGER.info("%s VERIFY", progress_tag)
        ver, verify_raw, verify_usage = self.verify(
            query,
            draft,
            sessions=sessions,
            derived_preferences=derived,
        )
        token_totals["input_tokens"] += verify_usage.get("input_tokens") or 0
        token_totals["output_tokens"] += verify_usage.get("output_tokens") or 0
        token_totals["total_tokens"] += verify_usage.get("total_tokens") or 0
        steps.append(
            {
                "step": "verify",
                "iteration": 1,
                "verdict": ver.get("verdict"),
                "issues": ver.get("issues", []),
                "raw_output": verify_raw,
                "usage": verify_usage,
            }
        )

        if ver["verdict"] == "ACCEPT":
            LOGGER.info("run: verification ACCEPT")
            return draft, ver, derived, steps, token_totals

        LOGGER.info("run: verification REJECT -> refine")
        if progress_tag:
            PROGRESS_LOGGER.info("%s REFINE", progress_tag)
        final_calls, refine_raw, refine_usage = self.refine(
            query,
            draft,
            ver.get("issues", []),
            sessions=sessions,
            derived_preferences=derived,
        )
        token_totals["input_tokens"] += refine_usage.get("input_tokens") or 0
        token_totals["output_tokens"] += refine_usage.get("output_tokens") or 0
        token_totals["total_tokens"] += refine_usage.get("total_tokens") or 0
        steps.append(
            {
                "step": "refine",
                "iteration": 1,
                "api_calls": final_calls,
                "raw_output": refine_raw,
                "usage": refine_usage,
            }
        )
        # Optional: one more verify pass (turn this on if you want strictness)
        # ver2 = self.verify(query, preference_list, final_calls)
        # return final_calls, ver2

        return final_calls, ver, derived, steps, token_totals


# -----------------------------
# 4) Example usage
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "datasets", "dev_4.json"),
    )
    parser.add_argument(
        "--query_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "domain_queries.json"),
    )
    parser.add_argument(
        "--pref_type",
        type=str,
        choices=["easy", "medium", "hard"],
        default=None,
        help="Enable pref-based case selection (easy/medium/hard).",
    )
    parser.add_argument(
        "--pref_list_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "evaluate", "pref_list.json"),
    )
    parser.add_argument(
        "--pref_group_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "evaluate", "pref_group.json"),
    )
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    agent = MultiCallAgent(LLMConfig(model=args.model, temperature=args.temperature))
    run_start = time.time()

    LOGGER.info("loading dataset: %s", args.dataset_path)
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    LOGGER.info("loading domain queries: %s", args.query_path)
    with open(args.query_path, "r", encoding="utf-8") as f:
        domain_queries: Dict[str, str] = json.load(f)

    pref_list = None
    pref_group = None
    if args.pref_type:
        if args.pref_type in ("easy", "medium"):
            pref_list = _load_json_file(args.pref_list_path)
        if args.pref_type in ("medium", "hard"):
            pref_group = _load_json_file(args.pref_group_path)

    results = []
    sample_count = len(data)
    total_cases_generated = 0
    skipped_no_cases = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_all_tokens = 0
    PROGRESS_LOGGER.info("processing %s/%s instances", sample_count, len(data))
    for idx, item in enumerate(data, start=1):
        example_id = item.get("example_id")
        PROGRESS_LOGGER.info("item %s/%s example_id=%s", idx, sample_count, example_id)
        sessions = item.get("sessions", [])
        gold_by_domain = _group_gold_by_domain(item.get("api_calls", []))
        pct = int((idx / sample_count) * 100)
        progress_tag = f"{idx}/{sample_count} ({pct}%) {example_id}"

        if args.pref_type:
            cases = _build_pref_cases(
                item,
                domain_queries,
                pref_type=args.pref_type,
                pref_list=pref_list,
                pref_group=pref_group,
            )
            if not cases:
                LOGGER.info("item %s example_id=%s has no %s cases; skipping", idx, example_id, args.pref_type)
                skipped_no_cases += 1
                continue
            for case in cases:
                total_cases_generated += 1
                domain = case["domain"]
                query = case["query"]
                LOGGER.info("domain=%s query_len=%s pref_type=%s", domain, len(query), args.pref_type)
                api_calls, verification, derived_preferences, steps, token_totals = agent.run(
                    query,
                    sessions=sessions,
                    progress_tag=progress_tag,
                )
                total_input_tokens += token_totals.get("input_tokens", 0)
                total_output_tokens += token_totals.get("output_tokens", 0)
                total_all_tokens += token_totals.get("total_tokens", 0)
                results.append(
                    {
                        "example_id": example_id,
                        "domain": domain,
                        "query": query,
                        "prediction": api_calls,
                        "gold": [case["ground_truth"]],
                        "verification": verification,
                        "derived_preferences": derived_preferences,
                        "final": {
                            "prediction": api_calls,
                            "verification": verification,
                        },
                        "steps": steps,
                        "usage": token_totals,
                    }
                )
        else:
            domains = _extract_domains_from_calls(item.get("api_calls", []))
            if not domains:
                LOGGER.info("item %s example_id=%s has no domains; skipping", idx, example_id)
            for domain in domains:
                total_cases_generated += 1
                query = domain_queries.get(domain, "")
                LOGGER.info("domain=%s query_len=%s", domain, len(query))
                api_calls, verification, derived_preferences, steps, token_totals = agent.run(
                    query,
                    sessions=sessions,
                    progress_tag=progress_tag,
                )
                total_input_tokens += token_totals.get("input_tokens", 0)
                total_output_tokens += token_totals.get("output_tokens", 0)
                total_all_tokens += token_totals.get("total_tokens", 0)
                results.append(
                    {
                        "example_id": example_id,
                        "domain": domain,
                        "query": query,
                        "prediction": api_calls,
                        "gold": gold_by_domain.get(domain, []),
                        "verification": verification,
                        "derived_preferences": derived_preferences,
                        "final": {
                            "prediction": api_calls,
                            "verification": verification,
                        },
                        "steps": steps,
                        "usage": token_totals,
                    }
                )

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    difficulty_tag = f"_{args.pref_type}" if args.pref_type else ""
    output_path = os.path.join(
        results_dir,
        f"output_{datetime.now():%y%m%d_%H%M%S}{difficulty_tag}.json",
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        f.write("\n")
    LOGGER.info("saved results: %s", output_path)
    run_end = time.time()
    total_seconds = run_end - run_start
    avg_seconds = (total_seconds / total_cases_generated) if total_cases_generated else 0.0
    LOGGER.info("summary: total_cases_generated=%s skipped_no_cases=%s", total_cases_generated, skipped_no_cases)
    LOGGER.info("timing: total_seconds=%.2f avg_seconds_per_case=%.4f", total_seconds, avg_seconds)
    LOGGER.info(
        "usage: input_tokens=%s output_tokens=%s total_tokens=%s",
        total_input_tokens,
        total_output_tokens,
        total_all_tokens,
    )

    print("API CALLS:")
    for c in api_calls:
        print(" -", c)

    print("\nVERIFICATION:")
    print(json.dumps(verification, indent=2, ensure_ascii=False))
