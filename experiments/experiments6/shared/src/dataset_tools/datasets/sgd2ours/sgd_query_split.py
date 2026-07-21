import csv
import json
import re


INPUTS = {
    "train": "result/train_2_pref_group_dialogues.json",
    "test": "result/test_2_pref_group_dialogues.json",
}
OUTPUT_TEMPLATE = "result/{split}_2_pref_group_queries.json"

# common_pref.csv에서 low_cost/solo_usage slot 매칭용
PREF_CSV_PATH = "common_pref.csv"
TARGET_PREF_GROUPS = {"low_cost", "solo_usage"}

# 필요하면 여기만 바꾸면 됨
REST_FIELD = "rest_dialogue"

API_PATTERN = re.compile(r"(\w+)\((.*)\)$")


def load_pref_slots(csv_path, pref_groups):
    pref_slots = set()
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("pref_group") in pref_groups:
                pref_slots.add((row.get("domain"), row.get("slot")))
    return pref_slots


def parse_service_call(call_str):
    m = API_PATTERN.match(call_str.strip())
    if not m:
        return None, {}
    domain = m.group(1)
    args = m.group(2).strip()
    if not args:
        return domain, {}

    slots = {}
    for part in args.split(","):
        if "=" not in part:
            continue
        slot, value = part.split("=", 1)
        slots[slot.strip()] = value.strip().strip('"')
    return domain, slots


def is_pref_service_turn(turn, pref_slots):
    evidence = []
    for call in turn.get("service", []):
        domain, slots = parse_service_call(call)
        if not domain:
            continue
        for slot in slots.keys():
            if (domain, slot) in pref_slots:
                evidence.append({
                    "domain": domain,
                    "slot": slot,
                    "value": slots.get(slot),
                })
    return evidence


def split_dialogue(dialogue, pref_slots):
    turns = dialogue.get("dialogue", [])
    first_pref_idx = None

    first_pref_evidence = None
    for i, turn in enumerate(turns):
        evidence = is_pref_service_turn(turn, pref_slots)
        if evidence:
            first_pref_idx = i
            first_pref_evidence = evidence
            break

    if first_pref_idx is None:
        return None, None, None

    last_user_idx = None
    for j in range(first_pref_idx - 1, -1, -1):
        if turns[j].get("role") == "User":
            last_user_idx = j
            break

    if last_user_idx is None:
        return None, None, None

    turns_with_tag = []
    for i, t in enumerate(turns):
        t_copy = dict(t)
        if i == first_pref_idx:
            t_copy["query_candidate"] = True
            t_copy["query_candidate_evidence"] = first_pref_evidence
        turns_with_tag.append(t_copy)

    query = []
    for t in turns_with_tag[:last_user_idx + 1]:
        q_turn = {
            "role": t.get("role"),
            "message": t.get("message", "")
        }
        if "service" in t:
            q_turn["service"] = t.get("service")
        if t.get("query_candidate") is True:
            q_turn["query_candidate"] = True
        query.append(q_turn)
    rest = turns_with_tag[last_user_idx + 1:]

    return query, rest, first_pref_idx


def process_split(split, input_path, output_path, pref_slots):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = []
    no_updated = 0

    for dialogue in data:
        query, rest, updated_idx = split_dialogue(dialogue, pref_slots)
        if query is None:
            no_updated += 1
            continue

        out.append({
            "dialogue_id": dialogue.get("dialogue_id"),
            "api_call": dialogue.get("api_call", []),
            "query": query,
            REST_FIELD: rest,
            "query_candidate_turn_index": updated_idx,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"[{split}] total={len(data)}")
    print(f"[{split}] saved={len(out)}")
    print(f"[{split}] no_updated_slot={no_updated}")
    print(f"[{split}] output={output_path}")


def main():
    pref_slots = load_pref_slots(PREF_CSV_PATH, TARGET_PREF_GROUPS)

    for split, input_path in INPUTS.items():
        output_path = OUTPUT_TEMPLATE.format(split=split)
        process_split(split, input_path, output_path, pref_slots)


if __name__ == "__main__":
    main()
