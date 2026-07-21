import argparse
import json
import re
from pathlib import Path

# Keep only the requested top-level fields.
ordered_keys = [
    "example_id",
    "user_utterance",
    "api_calls_pref",
    "api_calls",
    "reference_ground_truth",
    "evaluation_result",
    "llm_output",
    "reasoning_tokens",
]

def trim_api_calls_pref(item):
    pref = item.get("api_calls_pref")
    if isinstance(pref, list):
        item["api_calls_pref"] = [
            {
                "value_group": p.get("value_group"),
                "group_preference": p.get("group_preference"),
                "count": p.get("count"),
            }
            for p in pref
            if isinstance(p, dict)
        ]
    return item

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, help="Directory with JSON files.")
    parser.add_argument("--out-dir", required=True, help="Directory for output files.")
    parser.add_argument(
        "--suffix",
        default="_slim",
        help="Suffix for slim output files (default: _slim).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(results_dir.glob("*.json")):
        text = path.read_text(encoding="utf-8")

        # JSON 키-값에서만 NaN -> null
        text = re.sub(r'(:\s*)NaN\b', r'\1null', text)

        data = json.loads(text)
        filtered = [{k: item.get(k) for k in ordered_keys} for item in data]
        filtered = [trim_api_calls_pref(item) for item in filtered]

        out_path = out_dir / (path.stem + args.suffix + ".json")
        out_path.write_text(
            json.dumps(filtered, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

        true_items = [item for item in filtered if item.get("evaluation_result") == "TRUE"]
        false_items = [item for item in filtered if item.get("evaluation_result") == "FALSE"]

        out_true = out_dir / (path.stem + args.suffix + "_TRUE.json")
        out_false = out_dir / (path.stem + args.suffix + "_FALSE.json")
        out_true.write_text(
            json.dumps(true_items, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        out_false.write_text(
            json.dumps(false_items, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
