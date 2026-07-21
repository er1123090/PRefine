import argparse
import importlib.util
import json
import os


def load_eval_module():
    here = os.path.dirname(os.path.abspath(__file__))
    eval_path = os.path.join(here, "evaluation_aggregate.py")
    spec = importlib.util.spec_from_file_location("evaluation_aggregate", eval_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def to_str(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join([str(v) for v in value])
    return str(value)


def build_eval_input(src_path, tmp_path):
    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    converted = []
    for item in data:
        if not isinstance(item, dict):
            continue
        converted.append(
            {
                "reference_ground_truth": to_str(item.get("gold")),
                "llm_output": to_str(item.get("prediction")),
            }
        )
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)


def load_pref_file(pref_file):
    if not pref_file:
        return None
    if not os.path.exists(pref_file):
        raise FileNotFoundError(f"Preference file not found: {pref_file}")
    with open(pref_file, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Path to output_*.json file with prediction/gold fields",
    )
    parser.add_argument(
        "--pref_file",
        default="",
        help="Optional preference file JSON for pref-slot metrics",
    )
    parser.add_argument(
        "--tmp",
        default="",
        help="Optional temp json path; defaults to <input>.__eval_input.json",
    )
    parser.add_argument(
        "--keep_tmp",
        action="store_true",
        help="Keep the generated temp file",
    )
    args = parser.parse_args()

    src_path = args.input
    tmp_path = args.tmp or (src_path + ".__eval_input.json")
    build_eval_input(src_path, tmp_path)

    pref_map = load_pref_file(args.pref_file)
    eval_mod = load_eval_module()
    metrics = eval_mod.evaluate_single_file(tmp_path, preference_map=pref_map)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if not args.keep_tmp:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
