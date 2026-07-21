#!/usr/bin/env python3
# Group api_calls_pref evidence by (domain, slot) with value counts and dialogue_id sources.

import argparse
import json
from collections import defaultdict


def group_evidence(evidence):
    # Skip if already converted (heuristic: items have "values")
    if (
        isinstance(evidence, list)
        and evidence
        and isinstance(evidence[0], dict)
        and "values" in evidence[0]
    ):
        return evidence

    slot_map = defaultdict(lambda: defaultdict(set))
    for e in evidence:
        domain = e.get("domain")
        slot = e.get("slot")
        value = e.get("value")
        if domain is None or slot is None or value is None:
            continue
        dialogue_id = e.get("dialogue_id")
        if dialogue_id is None:
            continue
        slot_map[(domain, slot)][value].add(dialogue_id)

    new_evidence = []
    for (domain, slot), value_map in slot_map.items():
        values_list = []
        for value, sources in value_map.items():
            values_list.append(
                {
                    "value": value,
                    "meta": {
                        "count": len(sources),
                        "dialogue_ids": sorted(sources),
                    },
                }
            )
        values_list.sort(key=lambda x: x["value"])
        new_evidence.append({"domain": domain, "slot": slot, "values": values_list})
    new_evidence.sort(key=lambda x: (x["domain"], x["slot"]))

    return new_evidence


def convert(in_path, out_path):
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    updated = 0
    for group in data:
        prefs = group.get("api_calls_pref")
        if not prefs:
            continue
        for pref in prefs:
            evidence = pref.get("evidence")
            if not evidence:
                continue
            pref["evidence"] = group_evidence(evidence)
            updated += 1

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return updated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="./result/dev_4.json",
        help="Input dev_4.json path",
    )
    parser.add_argument(
        "--output",
        default="./result/dev_5.json",
        help="Output dev_5.json path",
    )
    args = parser.parse_args()

    updated = convert(args.input, args.output)
    print(f"saved={args.output} | updated_prefs={updated}")


if __name__ == "__main__":
    main()
