#!/usr/bin/env python3
# Export per-group api_calls and api_calls_pref value_groups into a CSV.

import argparse
import csv
import json


def normalize_api_calls(api_calls):
    return " | ".join(api_calls) if api_calls else ""


def normalize_value_groups(api_calls_pref):
    if not api_calls_pref:
        return ""
    vgs = []
    for pref in api_calls_pref:
        vg = pref.get("value_group")
        if vg:
            vgs.append(vg)
    if not vgs:
        return ""
    return " | ".join(sorted(set(vgs)))


def normalize_value_group_counts(api_calls_pref):
    if not api_calls_pref:
        return ""
    parts = []
    for pref in api_calls_pref:
        vg = pref.get("value_group")
        count = pref.get("count")
        if vg and count is not None:
            parts.append(f"{vg}:{count}")
    return " | ".join(parts)


def export_csv(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        groups = json.load(f)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["example_id", "api_calls", "value_groups", "value_group_counts"])

        for group in groups:
            example_id = group.get("example_id", "")
            api_calls = normalize_api_calls(group.get("api_calls", []))
            value_groups = normalize_value_groups(group.get("api_calls_pref", []))
            value_group_counts = normalize_value_group_counts(
                group.get("api_calls_pref", [])
            )
            writer.writerow(
                [example_id, api_calls, value_groups, value_group_counts]
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="./result/dev_6.json",
        help="Input dev_6.json path",
    )
    parser.add_argument(
        "--output",
        default="./result/dev_6_instance_api_calls_pref.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    export_csv(args.input, args.output)
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
