import json
import csv

INPUT_JSON = "result/dev_6.json"
OUTPUT_CSV = "result/dev_6_domain_slot_values.csv"

# rows = []

# with open(INPUT_JSON, "r", encoding="utf-8") as f:
#     data = json.load(f)

# for example in data:
#     example_id = example.get("example_id")

#     for inst in example.get("api_calls_all", []):
#         instance_id = inst.get("instance_id")
#         dialogue_id = inst.get("dialogue_id")

#         api_calls = inst.get("api_call", [])
#         api_calls_joined = " | ".join(api_calls)

#         rows.append({
#             "example_id": example_id,
#             "instance_id": instance_id,
#             "dialogue_id": dialogue_id,
#             "api_calls": api_calls_joined
#         })

# # CSV 저장
# with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
#     writer = csv.DictWriter(
#         f,
#         fieldnames=["example_id", "instance_id", "dialogue_id", "api_calls"]
#     )
#     writer.writeheader()
#     writer.writerows(rows)

# print(f"Saved {len(rows)} rows to {OUTPUT_CSV}")

####################################################################################3
# with open(INPUT_JSON, "r", encoding="utf-8") as f:
#     data = json.load(f)

# for example in data:
#     example_id = example.get("example_id")

#     for inst in example.get("api_calls_all", []):
#         instance_id = inst.get("instance_id")
#         dialogue_id = inst.get("dialogue_id")

#         for api_call in inst.get("api_call", []):
#             rows.append({
#                 "example_id": example_id,
#                 "instance_id": instance_id,
#                 "dialogue_id": dialogue_id,
#                 "api_call": api_call
#             })

# # CSV 저장
# with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
#     writer = csv.DictWriter(
#         f,
#         fieldnames=["example_id", "instance_id", "dialogue_id", "api_call"]
#     )
#     writer.writeheader()
#     writer.writerows(rows)

# print(f"Saved {len(rows)} api calls to {OUTPUT_CSV}")

#########################################################################3
# import json
import re
from collections import defaultdict

# =========================
# domain -> slot -> set(values)
# =========================
slot_values = defaultdict(lambda: defaultdict(set))

# =========================
# 정규식
# =========================
API_PATTERN = re.compile(r"^(\w+)\((.*)\)$")
SLOT_PATTERN = re.compile(r'(\w+)\s*=\s*"([^"]*)"')

# =========================
# JSON 로드
# =========================
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

# =========================
# API call 파싱 (sessions[*].api_call 기준)
# =========================
for example in data:
    for session in example.get("sessions", []):
        for api_call in session.get("api_call", []):
            match = API_PATTERN.match(api_call)
            if not match:
                continue

            domain = match.group(1)
            slot_str = match.group(2)

            for slot, value in SLOT_PATTERN.findall(slot_str):
                slot_values[domain][slot].add(value)

# =========================
# CSV 저장
# =========================
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["domain", "slot", "value"])

    for domain in sorted(slot_values):
        for slot in sorted(slot_values[domain]):
            for value in sorted(slot_values[domain][slot]):
                writer.writerow([domain, slot, value])

print(f"Saved domain-slot-value table to {OUTPUT_CSV}")
