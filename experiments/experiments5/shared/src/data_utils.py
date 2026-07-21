"""
Shared data loading and preprocessing utilities used across all methods.
"""

import json
import os
import re
import itertools
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_tools_from_file(file_path: str) -> List[Dict]:
    """Load a JSON schema file containing tool definitions."""
    if not os.path.exists(file_path):
        print(f"[Warning] Tools schema file not found: {file_path}")
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tools = json.load(f)
        print(f"[Info] Loaded {len(tools)} tools from {file_path}")
        return tools
    except Exception as e:
        print(f"[Error] Failed to load tools schema: {e}")
        return []


def load_chains_dataset(fpath: str) -> pd.DataFrame:
    """Load dataset from a JSONL or JSON file into a DataFrame."""
    try:
        df = pd.read_json(fpath, lines=True)
        return df
    except ValueError:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "dataset" in data:
            return pd.DataFrame(data["dataset"])
        return pd.DataFrame(data)


def load_query_map(fpath: str) -> Dict[str, str]:
    """Load a mapping from domain name to single-turn user utterance."""
    if not os.path.exists(fpath):
        return {}
    with open(fpath, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if isinstance(raw_data, dict):
        return raw_data

    if isinstance(raw_data, list):
        query_map: Dict[str, str] = {}
        for item in raw_data:
            if not isinstance(item, dict):
                continue

            domain = None
            targets = item.get("target") or []
            if isinstance(targets, list):
                for target in targets:
                    if isinstance(target, dict) and target.get("domain"):
                        domain = str(target["domain"])
                        break

            if not domain:
                api_calls = item.get("api_call") or []
                if isinstance(api_calls, list):
                    for api_call in api_calls:
                        if isinstance(api_call, str) and "(" in api_call:
                            domain = api_call.split("(", 1)[0].strip()
                            break

            if not domain:
                continue

            utterance = None
            query_turns = item.get("query") or []
            if isinstance(query_turns, list):
                for turn in query_turns:
                    if not isinstance(turn, dict):
                        continue
                    role = str(turn.get("role", "")).lower()
                    message = turn.get("message") or turn.get("content")
                    if role == "user" and message:
                        utterance = str(message).strip()
                        break

            if utterance:
                query_map[domain] = utterance

        return query_map

    return {}


def load_multiturn_data(fpath: str) -> Dict[str, Any]:
    """
    Load multi-turn query templates.

    Expected structure:
        { "DomainName": [ { "query": [...turns...], "api_call": [...] } ] }
    """
    if not os.path.exists(fpath):
        print(f"[Error] Multi-turn query file not found: {fpath}")
        return {}
    with open(fpath, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if isinstance(raw_data, dict):
        return raw_data

    if isinstance(raw_data, list):
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for item in raw_data:
            if not isinstance(item, dict):
                continue

            domain = None
            targets = item.get("target") or []
            if isinstance(targets, list):
                for target in targets:
                    if isinstance(target, dict) and target.get("domain"):
                        domain = str(target["domain"])
                        break

            if not domain:
                api_calls = item.get("api_call") or []
                if isinstance(api_calls, list):
                    for api_call in api_calls:
                        if isinstance(api_call, str) and "(" in api_call:
                            domain = api_call.split("(", 1)[0].strip()
                            break

            if not domain:
                continue

            grouped.setdefault(domain, []).append(item)

        return grouped

    return {}


def load_memory_file(fpath: str) -> Dict[str, Any]:
    """
    Load a memory JSONL file keyed by example_id.

    Each line is a JSON record with at least an "example_id" field.
    """
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Memory file not found: {fpath}")

    memory_storage = {}
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                rec_id = str(record.get("example_id"))
                memory_storage[rec_id] = record
            except json.JSONDecodeError:
                continue
    print(f"[Info] Loaded {len(memory_storage)} memory records from {fpath}")
    return memory_storage


def limit_prepared_items(
    prepared_items: List[Dict[str, Any]],
    max_queries: Optional[int],
) -> List[Dict[str, Any]]:
    """Deterministically subsample prepared items while preserving endpoint coverage."""
    if max_queries is None:
        return prepared_items
    if max_queries <= 0:
        raise ValueError(f"max_queries must be positive, got {max_queries}")

    total_items = len(prepared_items)
    if total_items <= max_queries:
        return prepared_items

    if max_queries == 1:
        selected_indices = [0]
    else:
        selected_indices = [
            ((total_items - 1) * idx) // (max_queries - 1)
            for idx in range(max_queries)
        ]

    limited_items = [prepared_items[idx] for idx in selected_indices]
    print(
        f"[Info] Limiting prepared items from {total_items} to {len(limited_items)} "
        f"with deterministic evenly spaced sampling (max_queries={max_queries})."
    )
    return limited_items


def load_example_id_sub_filter(fpath: Optional[str]) -> Optional[List[str]]:
    """Load a legacy example_id_sub allowlist from TXT or JSON."""
    if not fpath:
        return None
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"example_id_sub filter file not found: {fpath}")

    if fpath.endswith(".json"):
        with open(fpath, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if isinstance(raw_data, list):
            filter_ids = []
            for item in raw_data:
                if isinstance(item, str):
                    filter_ids.append(item.strip())
                elif isinstance(item, dict):
                    value = item.get("example_id_sub") or item.get("id")
                    if value:
                        filter_ids.append(str(value).strip())
            return [item for item in filter_ids if item]

        if isinstance(raw_data, dict):
            candidate_values = (
                raw_data.get("example_id_sub")
                or raw_data.get("ids")
                or raw_data.get("items")
            )
            if isinstance(candidate_values, list):
                return [str(item).strip() for item in candidate_values if str(item).strip()]

        raise ValueError(
            f"Unsupported example_id_sub filter JSON format: {type(raw_data).__name__}"
        )

    with open(fpath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def filter_prepared_items_by_example_id_sub(
    prepared_items: List[Dict[str, Any]],
    filter_ids: List[str],
) -> List[Dict[str, Any]]:
    """Filter prepared items by example_id_sub while preserving requested order."""
    item_map: Dict[str, Dict[str, Any]] = {}
    duplicate_ids = set()

    for item in prepared_items:
        example_id_sub = f"{item['original_ex'].get('example_id')}_{item['sub_idx']}"
        if example_id_sub in item_map:
            duplicate_ids.add(example_id_sub)
        item_map[example_id_sub] = item

    if duplicate_ids:
        duplicate_preview = ", ".join(sorted(duplicate_ids)[:10])
        raise ValueError(
            f"Duplicate prepared example_id_sub values detected: {duplicate_preview}"
        )

    filtered_items = []
    missing_ids = []
    for example_id_sub in filter_ids:
        item = item_map.get(example_id_sub)
        if item is None:
            missing_ids.append(example_id_sub)
            continue
        filtered_items.append(item)

    print(
        f"[Info] example_id_sub filter selected {len(filtered_items)} / {len(filter_ids)} "
        "requested items."
    )
    if missing_ids:
        preview = ", ".join(missing_ids[:10])
        print(
            f"[Warning] {len(missing_ids)} requested example_id_sub values were not prepared. "
            f"First missing: {preview}"
        )

    return filtered_items


def normalize_context_type(
    context_type: str,
    alias_map: Optional[Dict[str, str]] = None,
) -> str:
    """Map legacy context aliases to canonical values."""
    alias_map = alias_map or {
        "memory_diag": "memory_dialogue",
    }
    return alias_map.get(context_type, context_type)


def append_run_log(run_log_path: Optional[str], message: str) -> None:
    """Append a human-readable timestamped line to a run log file."""
    if not run_log_path:
        return
    dirpath = os.path.dirname(run_log_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")
    with open(run_log_path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


# ---------------------------------------------------------------------------
# Ground-truth generation helpers
# ---------------------------------------------------------------------------

def _to_str(val: Any) -> str:
    """Convert a value to string, normalising Python booleans."""
    if isinstance(val, bool):
        return "True" if val else "False"
    return str(val)


def generate_func_strings(domain: str, slot_values_map: Dict[str, List[str]]) -> List[str]:
    """
    Generate all possible function-call strings from a slot->values mapping.

    Example:
        domain = "GetHotels"
        slot_values_map = {"star": ["4", "5"], "rooms": ["1"]}
        => ["GetHotels(rooms=\"1\", star=\"4\")", "GetHotels(rooms=\"1\", star=\"5\")"]
    """
    if not slot_values_map:
        return []

    sorted_keys = sorted(slot_values_map.keys())
    values_lists = [slot_values_map[k] for k in sorted_keys]
    combinations = list(itertools.product(*values_lists))

    results = []
    for combo in combinations:
        args_parts = [f'{k}="{v}"' for k, v in zip(sorted_keys, combo)]
        results.append(f"{domain}({', '.join(args_parts)})")
    return results


# ---------------------------------------------------------------------------
# Single-turn utterance / ground-truth assignment
# ---------------------------------------------------------------------------

def assign_user_utterances_singleturn(
    pref_list_path: str,
    example: Dict[str, Any],
    query_map: Dict[str, str],
    pref_type: str,
    pref_group_path: Optional[str] = None,
) -> List[Tuple[str, List[str]]]:
    """
    Derive (user_utterance, [ground_truth_api_call, ...]) pairs for
    single-turn evaluation.

    Args:
        pref_list_path: Path to pref_list.json.
        example: One dataset record.
        query_map: Mapping from domain name to user utterance string.
        pref_type: "easy" | "medium" | "hard".
        pref_group_path: Path to pref_group.json (required for medium/hard).

    Returns:
        List of (utterance, ground_truth_list) tuples.
    """
    results: List[Tuple[str, List[str]]] = []

    if pref_type == "easy":
        if not os.path.exists(pref_list_path):
            return []
        with open(pref_list_path, "r", encoding="utf-8") as f:
            pref_list = json.load(f)

        api_calls = example.get("api_calls", [])
        if not isinstance(api_calls, list):
            return []

        for call_str in api_calls:
            if "(" in call_str:
                domain = call_str.split("(")[0].strip()
                try:
                    args_content = call_str.split("(", 1)[1].rsplit(")", 1)[0]
                except IndexError:
                    continue
            else:
                domain = call_str.strip()
                args_content = ""

            if domain not in query_map or domain not in pref_list:
                continue

            matches = re.findall(r'(\w+)=["\']([^"\']+)["\']', args_content)
            target_pref_slots = pref_list.get(domain, [])

            slot_map: Dict[str, List[str]] = {}
            for slot, value in matches:
                if slot in target_pref_slots:
                    slot_map[slot] = [_to_str(value)]

            if slot_map:
                results.append((query_map[domain], generate_func_strings(domain, slot_map)))

        return results

    elif pref_type == "medium":
        prefs = example.get("api_calls_pref", [])
        if not isinstance(prefs, list) or not prefs:
            return []
        if not pref_group_path or not os.path.exists(pref_group_path):
            return []

        with open(pref_group_path, "r", encoding="utf-8") as f:
            pref_group_data = json.load(f)

        for pref in prefs:
            group_name = pref.get("value_group")
            if group_name not in pref_group_data:
                continue

            group_rules = pref_group_data[group_name].get("rules", [])
            domain_data_map: Dict[str, Dict[str, set]] = {}

            for evidence in pref.get("evidence", []):
                e_domain = evidence.get("domain")
                e_slot = evidence.get("slot")
                if not e_domain or not e_slot or e_domain not in query_map:
                    continue

                candidate_values = [
                    _to_str(rule.get("value"))
                    for rule in group_rules
                    if rule.get("domain") == e_domain and rule.get("slot") == e_slot
                ]

                if candidate_values:
                    domain_data_map.setdefault(e_domain, {}).setdefault(e_slot, set())
                    domain_data_map[e_domain][e_slot].update(candidate_values)

            for domain, slot_map in domain_data_map.items():
                final_slot_map = {k: list(v) for k, v in slot_map.items()}
                results.append((query_map[domain], generate_func_strings(domain, final_slot_map)))

        return results

    elif pref_type == "hard":
        if not pref_group_path or not os.path.exists(pref_group_path):
            return []
        with open(pref_group_path, "r", encoding="utf-8") as f:
            pref_group_data = json.load(f)

        prefs = example.get("api_calls_pref", [])
        if not isinstance(prefs, list) or not prefs:
            return []

        for pref in prefs:
            current_group_name = pref.get("value_group")
            if not current_group_name or current_group_name not in pref_group_data:
                continue

            used_domains = {e.get("domain") for e in pref.get("evidence", []) if e.get("domain")}
            rules = pref_group_data[current_group_name].get("rules", [])

            candidate_domains = {
                rule.get("domain")
                for rule in rules
                if rule.get("domain") and rule.get("domain") in query_map
                and rule.get("domain") not in used_domains
            }

            for cand_domain in candidate_domains:
                slot_values_map: Dict[str, List[str]] = {}
                for rule in rules:
                    if rule.get("domain") == cand_domain:
                        s, v = rule.get("slot"), rule.get("value")
                        if s and v is not None:
                            val_str = _to_str(v)
                            if val_str not in slot_values_map.get(s, []):
                                slot_values_map.setdefault(s, []).append(val_str)

                if slot_values_map:
                    results.append((query_map[cand_domain], generate_func_strings(cand_domain, slot_values_map)))

        return results

    return results


# ---------------------------------------------------------------------------
# Multi-turn utterance / ground-truth assignment
# ---------------------------------------------------------------------------

def parse_api_call_to_dict(api_str: str) -> Tuple[str, Dict[str, str]]:
    """Parse 'Domain(slot="val", ...)' into (domain, {slot: val}) dict."""
    if "(" not in api_str:
        return api_str.strip(), {}
    domain = api_str.split("(")[0].strip()
    try:
        args_content = api_str.split("(", 1)[1].rsplit(")", 1)[0]
    except IndexError:
        return domain, {}
    matches = re.findall(r'(\w+)=["\']([^"\']+)["\']', args_content)
    return domain, {k: v for k, v in matches}


def generate_single_api_string(domain: str, args_dict: Dict[str, str]) -> str:
    sorted_keys = sorted(args_dict.keys())
    parts = [f'{k}="{args_dict[k]}"' for k in sorted_keys]
    return f"{domain}({', '.join(parts)})"


def merge_and_generate_api_strings(
    domain: str,
    base_args: Dict[str, str],
    pref_slot_map: Dict[str, List[str]],
) -> List[str]:
    """
    Combine fixed base arguments with preference-derived slot values
    to produce all candidate API call strings.
    """
    if not pref_slot_map:
        return [generate_single_api_string(domain, base_args)]

    sorted_pref_keys = sorted(pref_slot_map.keys())
    pref_values_lists = [pref_slot_map[k] for k in sorted_pref_keys]

    results = []
    for combo in itertools.product(*pref_values_lists):
        current_args = base_args.copy()
        for key, val in zip(sorted_pref_keys, combo):
            current_args[key] = val
        results.append(generate_single_api_string(domain, current_args))
    return results


def format_multiturn_dialogue(query_list: List[Dict[str, str]]) -> str:
    """Convert a list of turn dicts into a single dialogue string."""
    return "\n".join(
        f"{turn.get('role', 'User')}: {turn.get('message', '')}"
        for turn in query_list
    )


def assign_user_utterances_multiturn(
    pref_list_path: str,
    example: Dict[str, Any],
    multiturn_data: Dict[str, Any],
    pref_type: str,
    pref_group_path: Optional[str] = None,
) -> List[Tuple[str, List[str]]]:
    """
    Derive (dialogue_text, [ground_truth_api_call, ...]) pairs for
    multi-turn evaluation.

    Args:
        pref_list_path: Path to pref_list.json.
        example: One dataset record.
        multiturn_data: Multi-turn template data keyed by domain.
        pref_type: "easy" | "medium" | "hard".
        pref_group_path: Path to pref_group.json (required for medium/hard).

    Returns:
        List of (dialogue_text, ground_truth_list) tuples.
    """
    results: List[Tuple[str, List[str]]] = []

    def process_extraction(domain: str, pref_map: Dict[str, List[str]]) -> Optional[Tuple[str, List[str]]]:
        if domain not in multiturn_data:
            return None
        template_data = multiturn_data[domain][0]
        dialogue_text = format_multiturn_dialogue(template_data.get("query", []))
        base_api_str = template_data.get("api_call", [""])[0]
        _, base_args = parse_api_call_to_dict(base_api_str)
        gt_strings = merge_and_generate_api_strings(domain, base_args, pref_map)
        return (dialogue_text, gt_strings)

    if pref_type == "easy":
        if not os.path.exists(pref_list_path):
            return []
        with open(pref_list_path, "r", encoding="utf-8") as f:
            pref_list = json.load(f)

        for call_str in example.get("api_calls", []):
            domain, current_args = parse_api_call_to_dict(call_str)
            if domain not in multiturn_data or domain not in pref_list:
                continue

            target_pref_slots = pref_list.get(domain, [])
            pref_map = {
                slot: [_to_str(value)]
                for slot, value in current_args.items()
                if slot in target_pref_slots
            }

            if pref_map:
                res = process_extraction(domain, pref_map)
                if res:
                    results.append(res)

    elif pref_type == "medium":
        prefs = example.get("api_calls_pref", [])
        if not isinstance(prefs, list) or not prefs:
            return []
        if not pref_group_path or not os.path.exists(pref_group_path):
            return []

        with open(pref_group_path, "r", encoding="utf-8") as f:
            pref_group_data = json.load(f)

        for pref in prefs:
            group_name = pref.get("value_group")
            if group_name not in pref_group_data:
                continue

            group_rules = pref_group_data[group_name].get("rules", [])
            domain_data_map: Dict[str, Dict[str, set]] = {}

            for evidence in pref.get("evidence", []):
                e_domain = evidence.get("domain")
                e_slot = evidence.get("slot")
                if not e_domain or not e_slot or e_domain not in multiturn_data:
                    continue

                candidate_values = [
                    _to_str(rule.get("value"))
                    for rule in group_rules
                    if rule.get("domain") == e_domain and rule.get("slot") == e_slot
                ]

                if candidate_values:
                    domain_data_map.setdefault(e_domain, {}).setdefault(e_slot, set())
                    domain_data_map[e_domain][e_slot].update(candidate_values)

            for domain, slot_map in domain_data_map.items():
                res = process_extraction(domain, {k: list(v) for k, v in slot_map.items()})
                if res:
                    results.append(res)

    elif pref_type == "hard":
        if not pref_group_path or not os.path.exists(pref_group_path):
            return []
        with open(pref_group_path, "r", encoding="utf-8") as f:
            pref_group_data = json.load(f)

        prefs = example.get("api_calls_pref", [])
        if not isinstance(prefs, list) or not prefs:
            return []

        for pref in prefs:
            current_group_name = pref.get("value_group")
            if not current_group_name or current_group_name not in pref_group_data:
                continue

            used_domains = {e.get("domain") for e in pref.get("evidence", []) if e.get("domain")}
            rules = pref_group_data[current_group_name].get("rules", [])

            candidate_domains = {
                rule.get("domain")
                for rule in rules
                if rule.get("domain") and rule.get("domain") in multiturn_data
                and rule.get("domain") not in used_domains
            }

            for cand_domain in candidate_domains:
                slot_values_map: Dict[str, List[str]] = {}
                for rule in rules:
                    if rule.get("domain") == cand_domain:
                        s, v = rule.get("slot"), rule.get("value")
                        if s and v is not None:
                            val_str = _to_str(v)
                            if val_str not in slot_values_map.get(s, []):
                                slot_values_map.setdefault(s, []).append(val_str)

                if slot_values_map:
                    res = process_extraction(cand_domain, slot_values_map)
                    if res:
                        results.append(res)

    return results


# ---------------------------------------------------------------------------
# History string helpers
# ---------------------------------------------------------------------------

def get_api_calls_string(example: Dict[str, Any]) -> str:
    """Format all session API calls into a single string."""
    collected_apis = []
    for idx, session in enumerate(example.get("sessions", []), start=1):
        api_calls = session.get("api_call", [])
        if isinstance(api_calls, str) and api_calls:
            api_calls = [api_calls]
        if isinstance(api_calls, list):
            for call in api_calls:
                collected_apis.append(f"[Session {idx}] {call}")
    return "\n".join(collected_apis)


def get_dialogue_history_string(example: Dict[str, Any]) -> str:
    """Format all session dialogues into a single string."""
    sessions_str = []
    for idx, session in enumerate(example.get("sessions", []), start=1):
        lines = [f"[Session {idx}]"]
        for turn in session.get("dialogue", []):
            role = turn.get("role", "").capitalize()
            content = turn.get("message") or turn.get("content") or ""
            if role and content:
                lines.append(f"{role}: {content}")
        sessions_str.append("\n".join(lines))
    return "\n\n".join(sessions_str)


def build_input_prompt(
    example: Dict[str, Any],
    current_user_utterance: str,
    template: str,
    context_type: str,
    tools_schema: Optional[List[Dict]] = None,
    retrieved_context: Optional[str] = None,
) -> str:
    """
    Construct the final inference prompt.

    If ``retrieved_context`` is provided (e.g. RAG mode), it replaces the
    static history. Otherwise the history is built from ``context_type``.
    """
    if retrieved_context is not None:
        final_context = f"\n--- Relevant Past Context (Retrieved) ---\n{retrieved_context}\n"
    else:
        api_str = get_api_calls_string(example)
        history_str = get_dialogue_history_string(example)
        if context_type == "diag-apilist":
            final_context = (
                f"\n--- Dialogue History ---\n{history_str}\n\n"
                f"--- Past API Calls ---\n{api_str}\n"
            )
        elif context_type == "apilist-only":
            final_context = api_str
        elif context_type == "diag-only":
            final_context = history_str
        else:
            final_context = ""

    schema_str = (
        json.dumps(tools_schema, indent=2, ensure_ascii=False)
        if tools_schema
        else "No specific schema provided."
    )

    try:
        return template.format(
            dialogue_history=final_context,
            user_utterance=current_user_utterance.strip(),
            preference_schema=schema_str,
        )
    except KeyError:
        return template.format(
            dialogue_history=final_context,
            user_utterance=current_user_utterance.strip(),
        )
