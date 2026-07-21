import json
import os


def export_value_groups_tsv(
    input_path=None,
    output_path=None,
):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if input_path is None:
        input_path = os.path.join(base_dir, "result", "dev_3.json")
    if output_path is None:
        output_path = os.path.join(
            base_dir, "result", "dev_3_api_calls_pref_value_groups.tsv"
        )
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("example_id\tvalue_group\tcount\n")
        for g in data:
            example_id = g.get("example_id", "")
            prefs = g.get("api_calls_pref", [])
            for p in prefs:
                vg = p.get("value_group", "")
                count = p.get("count", "")
                f.write(f"{example_id}\t{vg}\t{count}\n")


if __name__ == "__main__":
    export_value_groups_tsv()
