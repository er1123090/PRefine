# utils_mem0.py
import os
import json
import pandas as pd
import re
from typing import List, Dict, Any, Tuple

# --- Prompt Templates ---
EXPLICIT_ZS_PROMPT_TEMPLATE = "Relevant Memories:\n{retrieved_memories}\n\nHistory:\n{dialogue_history}\nUser: {user_utterance}"
EXPLICIT_FS_PROMPT_TEMPLATE = EXPLICIT_ZS_PROMPT_TEMPLATE
IMPLICIT_ZS_PROMPT_TEMPLATE = EXPLICIT_ZS_PROMPT_TEMPLATE
IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE = """You are an API selection assistant.
Relevant Memories (User Preferences & Constraints):
{retrieved_memories}

Dialogue History:
{dialogue_history}

User Utterance:
{user_utterance}

Output ONLY one API call.
"""

# --- Load Helpers ---
def load_chains_dataset(fpath: str) -> pd.DataFrame:
    try:
        df = pd.read_json(fpath, lines=True)
        return df
    except ValueError:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)

def load_query_map(fpath: str) -> Dict[str, str]:
    if not os.path.exists(fpath):
        return {}
    with open(fpath, "r", encoding="utf-8") as f:
        return json.load(f)

# --- Data Parsing Helpers (Updated for easy/medium/hard) ---
def assign_user_utterances(
    pref_list_path: str, 
    example: Dict[str, Any], 
    query_map: Dict[str, str], 
    pref_type: str, 
    pref_group_path: str = None
) -> List[Tuple[str, str]]:
    
    results = []

    # -------------------------------------------------------
    # [CASE 1] easy (Rule-based from api_calls)
    # -------------------------------------------------------
    if pref_type == "easy":
        if not os.path.exists(pref_list_path):
            return []  
        with open(pref_list_path, "r", encoding="utf-8") as f:
            pref_list = json.load(f)

        api_calls = example.get("api_calls", [])
        if isinstance(api_calls, list):
            for call_str in api_calls:
                # 1. Parse Domain
                if "(" in call_str:
                    domain = call_str.split("(")[0].strip()
                    try:
                        args_content = call_str.split("(", 1)[1].rsplit(")", 1)[0]
                    except IndexError:
                        continue 
                else:
                    domain = call_str.strip()
                    args_content = ""

                if domain not in query_map: continue
                if domain not in pref_list: continue

                # 2. Parse Slots
                pattern = r'(\w+)=["\']([^"\']+)["\']'
                matches = re.findall(pattern, args_content)
                target_pref_slots = pref_list.get(domain, [])
                
                # 3. Check Intersection
                has_target_slot = False
                for slot, _ in matches:
                    if slot in target_pref_slots:
                        has_target_slot = True
                        break
                
                if not has_target_slot: continue

                # 4. Construct GT
                filtered_slots = []
                for slot, value in matches:
                    filtered_slots.append(f'{slot}="{value}"')

                if filtered_slots:
                    new_ground_truth = f"{domain}({', '.join(filtered_slots)})"
                    results.append((query_map[domain], new_ground_truth))
        return results

    # -------------------------------------------------------
    # [CASE 2] medium (Explicit Evidence Usage)
    # -------------------------------------------------------
    elif pref_type == "medium":
        prefs = example.get("api_calls_pref", [])
        if not isinstance(prefs, list) or not prefs: return []

        for pref in prefs:
            evidence_list = pref.get("evidence", [])
            if not isinstance(evidence_list, list): continue
            
            for evidence in evidence_list:
                domain = evidence.get("domain")
                if domain and domain in query_map:
                    if 'api_call' in evidence:
                         ground_truth_str = evidence['api_call']
                    else:
                        slots_str_list = [f'{evidence["slot"]}="{evidence["value"]}"']
                        ground_truth_str = f"{domain}({', '.join(slots_str_list)})"
                    
                    results.append((query_map[domain], ground_truth_str))
        return results

    # -------------------------------------------------------
    # [CASE 3] hard (Unseen Domain within Group)
    # -------------------------------------------------------
    elif pref_type == "hard":
        if not pref_group_path or not os.path.exists(pref_group_path):
            print(f"Warning: pref_group_path not found: {pref_group_path}")
            return []
        
        with open(pref_group_path, "r", encoding="utf-8") as f:
            pref_group_data = json.load(f)

        prefs = example.get("api_calls_pref", [])
        if not isinstance(prefs, list) or not prefs: return []

        for pref in prefs:
            current_group_name = pref.get("value_group")
            if not current_group_name or current_group_name not in pref_group_data:
                continue

            # Collect Used Domains
            used_domains = set()
            for evidence in pref.get("evidence", []):
                d = evidence.get("domain")
                if d: used_domains.add(d)

            # Find Unseen Domain & Construct GT
            group_rules = pref_group_data[current_group_name].get("rules", [])
            for rule in group_rules:
                candidate_domain = rule.get("domain")
                
                # Condition: Domain in QueryMap AND NOT in UsedDomains
                if candidate_domain and (candidate_domain in query_map) and (candidate_domain not in used_domains):
                    target_slot = rule.get("slot")
                    target_value = rule.get("value")
                    
                    if isinstance(target_value, bool):
                        val_str = "True" if target_value else "False"
                    else:
                        val_str = str(target_value)
                    
                    ground_truth_str = f'{candidate_domain}({target_slot}="{val_str}")'
                    results.append((query_map[candidate_domain], ground_truth_str))
        return results

    return results

# --- String & Mem0 Format Helpers ---
def get_api_calls_string(example: Dict[str, Any]) -> str:
    all_api_data = example.get("api_calls_all", [])
    collected_apis = []
    for item in all_api_data:
        calls = item.get("api_call", [])
        if isinstance(calls, list):
            collected_apis.extend(calls)
        elif isinstance(calls, str) and calls:
            collected_apis.append(calls)
    return "\n".join(collected_apis)

def get_dialogue_history_string(example: Dict[str, Any]) -> str:
    history_data = example.get("all_standing_instructions", [])
    sessions_str = []
    for idx, instruction_data in enumerate(history_data, start=1):
        turns = instruction_data.get("generated_dialogue", [])
        lines = [f"[Session {idx}]"]
        for turn in turns:
            role = turn.get("role", "").capitalize()
            content = turn.get("message") or turn.get("content") or ""
            if role and content:
                lines.append(f"{role}: {content}")
        sessions_str.append("\n".join(lines))
    return "\n\n".join(sessions_str)

def prepare_messages_for_mem0(example: Dict[str, Any]) -> List[Dict[str, str]]:
    messages = []
    history_data = example.get("all_standing_instructions", [])
    for instruction_data in history_data:
        turns = instruction_data.get("generated_dialogue", [])
        for turn in turns:
            role = turn.get("role", "").lower()
            content = turn.get("message") or turn.get("content") or ""
            if role and content:
                if role not in ["user", "assistant", "system"]:
                    role = "user" if role == "user" else "assistant"
                messages.append({"role": role, "content": content})
    return messages