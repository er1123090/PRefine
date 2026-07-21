#!/usr/bin/env python3
# Drop groups with empty api_calls and api_calls_pref.

import argparse
import json


def is_empty_group(group):
    api_calls = group.get("api_calls") or []
    api_calls_pref = group.get("api_calls_pref") or []
    return not api_calls and not api_calls_pref


def filter_groups(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        groups = json.load(f)

    kept = [g for g in groups if not is_empty_group(g)]
    dropped = len(groups) - len(kept)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(kept, f, indent=2, ensure_ascii=False)

    return len(groups), len(kept), dropped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="./result/dev_5.json",
        help="Input dev_5.json path",
    )
    parser.add_argument(
        "--output",
        default="./result/dev_6.json",
        help="Output dev_6.json path",
    )
    args = parser.parse_args()

    total, kept, dropped = filter_groups(args.input, args.output)
    print(f"saved={args.output} | total={total} | kept={kept} | dropped={dropped}")


if __name__ == "__main__":
    main()
