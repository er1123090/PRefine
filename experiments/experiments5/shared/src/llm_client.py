"""
Unified async LLM client supporting OpenAI, Gemini (Google), and Anthropic (Claude).

Usage example:
    from src.llm_client import call_llm_api_async, parse_deepseek_reasoning
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = None


# ---------------------------------------------------------------------------
# Reasoning output helpers
# ---------------------------------------------------------------------------

FALSE_ENV_VALUES = {"0", "false", "no", "off"}


def _env_enabled(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in FALSE_ENV_VALUES


def _is_deepseek_strict_text_mode(model_name: str, tools_schema: Optional[List[Dict]]) -> bool:
    if not tools_schema:
        return False
    if "deepseek" not in model_name.lower():
        return False
    return _env_enabled("LLM_DEEPSEEK_STRICT_TEXT", default=True)


def _tool_names(tools_schema: Optional[List[Dict]]) -> List[str]:
    names: List[str] = []
    for tool in tools_schema or []:
        fn = tool.get("function") if isinstance(tool, dict) else None
        name = fn.get("name") if isinstance(fn, dict) else None
        if isinstance(name, str) and name:
            names.append(name)
    return names


def _tool_slot_names(tools_schema: Optional[List[Dict]]) -> List[str]:
    slots = set()
    for names in _tool_slot_map(tools_schema).values():
        slots.update(names)
    return sorted(slots)


def _tool_slot_map(tools_schema: Optional[List[Dict]]) -> Dict[str, List[str]]:
    slot_map: Dict[str, List[str]] = {}
    for tool in tools_schema or []:
        fn = tool.get("function") if isinstance(tool, dict) else None
        fn_name = fn.get("name") if isinstance(fn, dict) else None
        params = fn.get("parameters") if isinstance(fn, dict) else None
        props = params.get("properties") if isinstance(params, dict) else None
        if isinstance(fn_name, str) and isinstance(props, dict):
            slot_map[fn_name] = sorted(str(name) for name in props)
    return slot_map


def _service_call_regex(tools_schema: Optional[List[Dict]]) -> str:
    max_args_raw = os.environ.get("LLM_DEEPSEEK_GUIDED_MAX_ARGS", "6")
    try:
        max_args = max(1, int(max_args_raw))
    except ValueError:
        max_args = 6

    slot_map = _tool_slot_map(tools_schema)
    patterns: List[str] = []
    for fn_name in _tool_names(tools_schema):
        slots = slot_map.get(fn_name, [])
        escaped_fn = re.escape(fn_name)
        if not slots:
            patterns.append(rf"{escaped_fn}\(\)")
            continue
        slot_alt = "|".join(re.escape(name) for name in slots)
        arg = rf"(?:{slot_alt})=\"[^\"]*\""
        patterns.append(rf"{escaped_fn}\((?:{arg}(?:, {arg}){{0,{max_args - 1}}})?\)")
    if not patterns:
        return r"Get[A-Za-z0-9_]+\(\)"
    return rf"(?:{'|'.join(patterns)})"


def _strict_text_prompt(prompt: str, tools_schema: Optional[List[Dict]]) -> str:
    names = ", ".join(_tool_names(tools_schema))
    return (
        "Return exactly one Service API call and nothing else.\n"
        "The first non-whitespace characters of your answer must be `Get`.\n"
        "Valid output format: GetDomain(slot_name=\"value\", ...)\n"
        "Do not explain. Do not show reasoning. Do not use markdown. Do not output JSON.\n"
        "Use only schema-valid function names and slot names.\n"
        f"Valid function names: {names}\n\n"
        "Task prompt:\n"
        f"{prompt}\n\n"
        "Final answer, one line only:"
    )


def _extract_text_function_call(text: str, tools_schema: Optional[List[Dict]] = None) -> str:
    if not text:
        return ""
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    clean = clean.replace(r"\"", '"')
    match = re.search(r"\b(Get[A-Za-z0-9_]*)\s*\(([^()]*)\)", clean, flags=re.DOTALL)
    if not match:
        return ""
    func_name = match.group(1)
    arg_text = match.group(2)
    slot_map = _tool_slot_map(tools_schema)
    valid_slots = set(slot_map.get(func_name, []))
    pairs = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\"([^\"]*)\"", arg_text)
    if not pairs:
        return f"{func_name}()"
    seen = set()
    args = []
    for slot, value in pairs:
        if valid_slots and slot not in valid_slots:
            continue
        if slot in seen:
            continue
        seen.add(slot)
        args.append(f'{slot}="{value}"')
    return f"{func_name}({', '.join(args)})"


def parse_deepseek_reasoning(raw_content: str) -> Dict[str, str]:
    """
    Split <think>...</think> reasoning from the final answer.

    Useful for vLLM-hosted DeepSeek or other models that embed reasoning
    inline with <think> tags.
    """
    if not raw_content:
        return {"llm_output": "", "reasoning_content": ""}

    end_tag = "</think>"
    end_idx = raw_content.rfind(end_tag)

    if end_idx != -1:
        reasoning_content = raw_content[:end_idx].replace("<think>", "").strip()
        clean_content = raw_content[end_idx + len(end_tag):].strip()
    else:
        reasoning_content = ""
        clean_content = raw_content.strip()

    return {"llm_output": clean_content, "reasoning_content": reasoning_content}


# ---------------------------------------------------------------------------
# Unified async call
# ---------------------------------------------------------------------------

async def call_llm_api_async(
    prompt: str,
    model_name: str,
    openai_client: Optional[AsyncOpenAI] = None,
    anthropic_client: Optional[Any] = None,
    tools_schema: Optional[List[Dict]] = None,
    reasoning_effort: Optional[str] = None,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Call an LLM asynchronously and return a normalised result dict.

    Supported providers (detected from model_name):
        - "gemini"  -> Google Gemini via google-genai SDK
        - "claude"  -> Anthropic Claude via anthropic SDK
        - anything else -> OpenAI (or OpenAI-compatible endpoint)

    Returns:
        {
            "output": str,             # Final model answer
            "reasoning_content": str,  # Chain-of-thought / thinking (if any)
            "token_counts": dict,      # Token usage metadata
            "error": str | None        # Error message if any
        }
    """
    result: Dict[str, Any] = {
        "output": "",
        "reasoning_content": "",
        "token_counts": {},
        "error": None,
    }

    try:
        # ---------------------------------------------------------------
        # Gemini
        # ---------------------------------------------------------------
        if "gemini" in model_name.lower():
            google_api_key = os.environ.get("GOOGLE_API_KEY")
            if not google_api_key:
                result["error"] = "API_KEY_MISSING_GOOGLE"
                return result
            if genai is None:
                result["error"] = "google-genai library not installed"
                return result

            client = genai.Client(api_key=google_api_key)
            config_params: Dict[str, Any] = {"temperature": 0.0}

            if reasoning_effort:
                config_params["thinking_config"] = genai_types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_level=reasoning_effort.lower(),
                )

            conf = genai_types.GenerateContentConfig(**config_params)
            response = await client.aio.models.generate_content(
                model=model_name, contents=prompt, config=conf
            )

            thought_parts: List[str] = []
            answer_parts: List[str] = []
            if response.candidates and response.candidates[0].content:
                for part in response.candidates[0].content.parts:
                    if not part.text:
                        continue
                    if hasattr(part, "thought") and part.thought:
                        thought_parts.append(part.text)
                    else:
                        answer_parts.append(part.text)

            result["output"] = "\n".join(answer_parts).strip()
            result["reasoning_content"] = "\n".join(thought_parts).strip()

            if response.usage_metadata:
                result["token_counts"] = {
                    "total_tokens": response.usage_metadata.total_token_count,
                    "input_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                    "output_tokens": response.usage_metadata.candidates_token_count,
                    "reasoning_tokens": getattr(response.usage_metadata, "thoughts_token_count", 0),
                }
            return result

        # ---------------------------------------------------------------
        # Anthropic (Claude)
        # ---------------------------------------------------------------
        elif "claude" in model_name.lower():
            if not anthropic_client:
                result["error"] = "API_KEY_MISSING_ANTHROPIC"
                return result

            max_tokens_limit = 16000
            kwargs: Dict[str, Any] = {
                "model": model_name,
                "max_tokens": max_tokens_limit,
                "messages": [{"role": "user", "content": prompt}],
            }

            if reasoning_effort:
                effort_level = reasoning_effort.lower()
                if "opus-4-5" in model_name:
                    kwargs["output_config"] = {"effort": effort_level}
                    kwargs.setdefault("betas", []).append("effort-2025-11-24")
                else:
                    budget_map = {
                        "minimal": 2048, "low": 4096, "medium": 8192,
                        "high": 16000, "max": 24000,
                    }
                    budget = min(budget_map.get(effort_level, 4096), max_tokens_limit - 1000)
                    kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}

            if "output_config" in kwargs and hasattr(anthropic_client, "beta"):
                response = await anthropic_client.beta.messages.create(**kwargs)
            else:
                response = await anthropic_client.messages.create(**kwargs)

            text_parts: List[str] = []
            thought_parts = []
            if hasattr(response, "content"):
                for block in response.content:
                    if block.type == "text":
                        text_parts.append(block.text)
                    elif block.type == "thinking":
                        thought_parts.append(block.thinking)

            result["output"] = "\n".join(text_parts).strip()
            result["reasoning_content"] = "\n".join(thought_parts).strip()

            if hasattr(response, "usage"):
                result["token_counts"] = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                }
                if hasattr(response.usage, "thinking_tokens"):
                    result["token_counts"]["reasoning_tokens"] = response.usage.thinking_tokens

            return result

        # ---------------------------------------------------------------
        # OpenAI (or OpenAI-compatible vLLM)
        # ---------------------------------------------------------------
        else:
            if not openai_client:
                result["error"] = "API_KEY_MISSING_OPENAI"
                return result

            deepseek_strict_text = _is_deepseek_strict_text_mode(model_name, tools_schema)
            messages = [{"role": "user", "content": prompt}]
            if deepseek_strict_text:
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a deterministic API-call formatter. "
                            "Your entire response must be one Service API call."
                        ),
                    },
                    {"role": "user", "content": _strict_text_prompt(prompt, tools_schema)},
                ]

            kwargs = {
                "model": model_name,
                "messages": messages,
            }
            max_tokens_env = os.environ.get("LLM_MAX_TOKENS")
            if max_tokens_env:
                try:
                    kwargs["max_tokens"] = int(max_tokens_env)
                except ValueError:
                    pass
            elif deepseek_strict_text:
                try:
                    kwargs["max_tokens"] = int(os.environ.get("LLM_DEEPSEEK_STRICT_MAX_TOKENS", "256"))
                except ValueError:
                    kwargs["max_tokens"] = 256

            if tools_schema and not deepseek_strict_text:
                kwargs["tools"] = tools_schema
                kwargs["tool_choice"] = "auto"
            elif deepseek_strict_text and _env_enabled("LLM_DEEPSEEK_GUIDED_REGEX", default=True):
                kwargs["extra_body"] = {
                    "structured_outputs": {"regex": _service_call_regex(tools_schema)}
                }

            is_reasoning_model = any(k in model_name.lower() for k in ["o1", "o3", "gpt-5"])
            if is_reasoning_model and reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort.lower()

            if temperature is not None:
                kwargs["temperature"] = temperature
            elif deepseek_strict_text:
                kwargs["temperature"] = 0.0

            response = await openai_client.chat.completions.create(**kwargs)
            message = response.choices[0].message
            raw_content = message.content or ""

            # Extract reasoning: try SDK field first, then <think> tags
            reasoning_text = getattr(message, "reasoning_content", "")
            if not reasoning_text and hasattr(response, "reasoning"):
                r_obj = response.reasoning
                if hasattr(r_obj, "summary"):
                    reasoning_text = r_obj.summary or ""
                elif isinstance(r_obj, dict):
                    reasoning_text = r_obj.get("summary", "")

            if not reasoning_text:
                parsed = parse_deepseek_reasoning(raw_content)
                final_output = parsed["llm_output"]
                reasoning_text = parsed["reasoning_content"]
            else:
                final_output = raw_content

            result["reasoning_content"] = str(reasoning_text) if reasoning_text else ""

            # Tool call handling
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                func_name = tool_call.function.name
                try:
                    func_args = json.loads(tool_call.function.arguments)
                    args_str_list = [f'{k}="{v}"' for k, v in func_args.items()]
                    result["output"] = f"{func_name}({', '.join(args_str_list)})"
                except Exception:
                    result["output"] = f"ERROR_JSON_PARSE: {tool_call.function.arguments}"
            else:
                # Try to extract a function-call pattern from text
                extracted_call = _extract_text_function_call(final_output or raw_content, tools_schema)
                if extracted_call:
                    result["output"] = extracted_call
                else:
                    result["output"] = final_output

            # Token counts
            total_tokens = 0
            reasoning_tokens = 0
            if hasattr(response, "usage") and response.usage:
                total_tokens = response.usage.total_tokens
                details = getattr(response.usage, "completion_tokens_details", None)
                if not details:
                    details = getattr(response.usage, "output_tokens_details", None)
                if details:
                    reasoning_tokens = getattr(details, "reasoning_tokens", 0)

            result["token_counts"] = {
                "total_tokens": total_tokens,
                "reasoning_tokens": reasoning_tokens,
                "input_tokens": getattr(response.usage, "prompt_tokens", 0),
                "output_tokens": getattr(response.usage, "completion_tokens", 0),
            }

            return result

    except Exception as e:
        print(f"[LLM Error] ({model_name}): {e}")
        result["error"] = f"API_ERROR: {str(e)}"
        return result


# ---------------------------------------------------------------------------
# Client factory helpers
# ---------------------------------------------------------------------------

def make_openai_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> Optional[AsyncOpenAI]:
    """Create an AsyncOpenAI client if an API key is available."""
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    kwargs: Dict[str, Any] = {"api_key": key}
    if base_url:
        kwargs["base_url"] = base_url
    return AsyncOpenAI(**kwargs)


def make_anthropic_client(api_key: Optional[str] = None) -> Optional[Any]:
    """Create an AsyncAnthropic client if the library is available and a key is set."""
    if AsyncAnthropic is None:
        return None
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    return AsyncAnthropic(api_key=key)
