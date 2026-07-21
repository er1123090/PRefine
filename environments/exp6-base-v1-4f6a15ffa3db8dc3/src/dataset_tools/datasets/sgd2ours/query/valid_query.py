import json, csv
from pathlib import Path

csv_path = Path("memory-dev/datasets/sgd2ours/query/dev_5_domain_slot_values.csv")
qs = Path("memory-dev/datasets/sgd2ours/query/query_slot.json")
qg = Path("memory-dev/datasets/sgd2ours/query/query_group.json")

allowed = {}
with csv_path.open(newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        allowed.setdefault((row["domain"], row["slot"]), set()).add(row["value"])

def extract_values(data):
    used = []
    for item in data:
        qid = item.get("query_id")
        for t in item.get("target", []):
            domain = t.get("domain")
            slot = t.get("slot")
            vals = t.get("value")
            if vals is None:
                continue
            if isinstance(vals, list):
                for v in vals:
                    used.append((qid, domain, slot, v))
            else:
                used.append((qid, domain, slot, vals))
    return used

def find_missing(data):
    missing = []
    for qid, domain, slot, v in extract_values(data):
        key = (domain, slot)
        if key not in allowed:
            missing.append((qid, domain, slot, v, "domain-slot-not-in-csv"))
        elif str(v) not in allowed[key]:
            missing.append((qid, domain, slot, v, "value-not-in-csv"))
    return missing

qs_missing = find_missing(json.loads(qs.read_text()))
qg_missing = find_missing(json.loads(qg.read_text()))

print("query_slot missing", qs_missing)
print("query_group missing", qg_missing)
