# Group dialogs into examples using value_group prefs; filter invalid api_calls and build api_calls_pref.
import json
import random
import re
import csv
from collections import defaultdict, Counter

MIN_PREF_COUNT = 2
MAX_PREF_COUNT = 6
EXCLUDED_VALUE_GROUPS = {"group_usage", "mid_cost"}
TARGET_VALUE_GROUPS = {"low_cost", "solo_usage", "high_cost"}
PREFERRED_VALUE_GROUP = "high_cost"

# --------------------
# 1) 빈 argument API call 판별 함수
# --------------------
def is_empty_api_call(call_str):
    s = call_str.strip()
    return s.endswith("()") and "(" in s and s.index("(") == len(s) - 2

# --------------------
# 2) dialogue 필터링
# --------------------
def filter_dialogues(data):
    filtered, excluded = [], []
    total = len(data)

    for i, dialog in enumerate(data):
        if i % 500 == 0:
            print(f"[filter] processing {i}/{total} ...")

        api_calls = dialog.get("api_call", [])
        has_empty = any(is_empty_api_call(c) for c in api_calls)

        (excluded if has_empty else filtered).append(dialog)

    print("[filter] DONE")
    return filtered, excluded

# --------------------
# 3) API 파싱 (따옴표 안 쉼표 안전 분리)
# --------------------
API_PATTERN = re.compile(r"^(\w+)\((.*)\)$")

def _split_args(args_str: str):
    parts, buf = [], []
    in_quotes, esc = False, False

    for ch in args_str:
        if esc:
            buf.append(ch)
            esc = False
            continue

        if ch == "\\":
            buf.append(ch)
            esc = True
            continue

        if ch == '"':
            in_quotes = not in_quotes
            buf.append(ch)
            continue

        if ch == "," and not in_quotes:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
        else:
            buf.append(ch)

    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)

    return parts

def parse_api(api_str):
    m = API_PATTERN.match(api_str.strip())
    if not m:
        return None, {}

    name = m.group(1)
    args = m.group(2).strip()
    if args == "":
        return name, {}

    slot_dict = {}
    for part in _split_args(args):
        if "=" in part:
            k, v = part.split("=", 1)
            slot_dict[k.strip()] = v.strip()

    return name, slot_dict

# --------------------
# 3.5) example_id 인덱스 계산
# --------------------
def get_max_group_idx(groups, split_name):
    prefix = f"{split_name}_"
    max_idx = 0
    for g in groups:
        example_id = g.get("example_id", "")
        if not example_id.startswith(prefix):
            continue
        tail = example_id[len(prefix):]
        if not tail.isdigit():
            continue
        idx = int(tail)
        if idx > max_idx:
            max_idx = idx
    return max_idx

# --------------------
# 4) value_group 로드
# --------------------
def load_value_group_map(path):
    """
    return:
      vg_map[(domain, slot, value)] = value_group
      vg_to_group_pref[value_group] = group_preference
    """
    vg_map = {}
    vg_to_group_pref = {}

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            domain = row["domain"].strip()
            slot = row["slot"].strip()
            value = row["value"].strip()
            value_group = row["pref_group"].strip()
            group_pref = row["group"].strip()

            vg_map[(domain, slot, value)] = value_group
            vg_to_group_pref[value_group] = group_pref

    return vg_map, vg_to_group_pref

# --------------------
# 5) api_calls 생성: (API, slot, value) 완전 동일 2회 이상만 전부 담기
# --------------------
# def merge_partial_api(api_calls_all):
#     triple_counter = Counter()

#     for api_str in api_calls_all:
#         name, slots = parse_api(api_str)
#         if name is None:
#             continue
#         for slot, value in slots.items():
#             triple_counter[(name, slot, value.strip())] += 1

#     by_name = defaultdict(list)
#     for (name, slot, value), cnt in triple_counter.items():
#         if cnt >= 2:
#             by_name[name].append((slot, value))

#     merged = []
#     for name, pairs in by_name.items():
#         pairs_sorted = sorted(set(pairs), key=lambda x: (x[0], x[1]))
#         args = ", ".join(f"{k}={v}" for k, v in pairs_sorted)
#         merged.append(f"{name}({args})")

#     return merged
def merge_partial_api(api_calls_all):
    triple_counter = Counter()

    for api_str in api_calls_all:
        name, slots = parse_api(api_str)
        if name is None:
            continue
        for slot, value in slots.items():
            triple_counter[(name, slot, value.strip())] += 1

    by_name = defaultdict(list)
    for (name, slot, value), cnt in triple_counter.items():
        if cnt >= 2:
            by_name[name].append((slot, value))

    merged = []
    for name, pairs in by_name.items():
        pairs_sorted = sorted(set(pairs), key=lambda x: (x[0], x[1]))
        args = ", ".join(f"{k}={v}" for k, v in pairs_sorted)
        merged.append(f"{name}({args})")

    return merged

# --------------------
# 6) dialogue 내 value_group 추출
# --------------------
def extract_dialogue_vgs(dialog, vg_map, vg_to_group_pref):
    """
    return:
      {
        vg: set((domain, slot, value))
      }
    """
    vg_triplets = defaultdict(set)

    for api_str in dialog.get("api_call", []):
        domain, slots = parse_api(api_str)
        if domain is None:
            continue

        for slot, value in slots.items():
            raw = value.strip()
            dequoted = raw.strip('"')

            key = None
            if (domain, slot, raw) in vg_map:
                key = (domain, slot, raw)
            elif (domain, slot, dequoted) in vg_map:
                key = (domain, slot, dequoted)

            if key is None:
                continue

            vg = vg_map[key]
            vg_triplets[vg].add(key)

    return vg_triplets

# --------------------
# 7) dialogue 분할: value_group 기준
# --------------------
def split_dialogues_by_vg(dialogues, vg_map, vg_to_group_pref):
    vg_buckets = defaultdict(list)
    fallback = []

    for dialog in dialogues:
        vg_triplets = extract_dialogue_vgs(dialog, vg_map, vg_to_group_pref)

        # Prefer high_cost when present and has at least 2 triplets.
        if (
            PREFERRED_VALUE_GROUP in vg_triplets
            and len(vg_triplets[PREFERRED_VALUE_GROUP]) >= 2
        ):
            vg_triplets = {
                PREFERRED_VALUE_GROUP: vg_triplets[PREFERRED_VALUE_GROUP]
            }

        # group_pref 충돌 검사
        gp_seen = {}
        conflict = False

        for vg, triplets in vg_triplets.items():
            gp = vg_to_group_pref[vg]
            if gp in gp_seen:
                conflict = True
                break
            gp_seen[gp] = vg

        if conflict or not vg_triplets:
            fallback.append(dialog)
            continue

        # value 다양성 검사
        valid_vgs = []
        for vg, triplets in vg_triplets.items():
            min_triplets = 1 if vg == PREFERRED_VALUE_GROUP else 2
            if len(set(triplets)) >= min_triplets:
                valid_vgs.append(vg)
        valid_vgs = [vg for vg in valid_vgs if vg in TARGET_VALUE_GROUPS]

        if not valid_vgs:
            fallback.append(dialog)
            continue

        # 하나만 허용 (여기선 1개만 있다고 가정)
        if PREFERRED_VALUE_GROUP in valid_vgs:
            vg_buckets[PREFERRED_VALUE_GROUP].append(dialog)
        else:
            vg_buckets[valid_vgs[0]].append(dialog)

    return vg_buckets, fallback

# --------------------
# 8) 그룹 생성 보조 함수 (버킷 제거 버전)
# --------------------
def make_groups_from_bucket(dialogs, split_name, start_idx, bucket_vg=None):
    groups = []
    rejected = []
    idx = start_idx
    random.shuffle(dialogs)

    if bucket_vg == PREFERRED_VALUE_GROUP:
        triplet_buckets = defaultdict(list)
        for dialog in dialogs:
            vg_triplets = extract_dialogue_vgs(dialog, vg_map, vg_to_group_pref)
            triplets = list(vg_triplets.get(PREFERRED_VALUE_GROUP, []))
            if not triplets:
                rejected.append(dialog)
                continue
            triplet_key = sorted(triplets)[0]
            triplet_buckets[triplet_key].append(dialog)

        # Rebuild dialogs list with high_cost triplet buckets first.
        dialogs = []
        for _, bucket in triplet_buckets.items():
            random.shuffle(bucket)
            dialogs.extend(bucket)

    i = 0
    while i + 4 < len(dialogs):
        min_size = 2 if bucket_vg == PREFERRED_VALUE_GROUP else 5
        size = min(random.randint(min_size, 10), len(dialogs) - i)
        chunk = dialogs[i:i+size]
        i += size
        idx += 1

        # ✅ 여기서 common_calls 생성
        flat_api_calls = []
        for dialog in chunk:
            flat_api_calls.extend(dialog.get("api_call", []))

        common_calls = merge_partial_api(flat_api_calls)

        api_calls_pref = build_api_calls_pref(
            chunk, vg_map, vg_to_group_pref
        )
        if any(p.get("count", 0) > MAX_PREF_COUNT for p in api_calls_pref):
            rejected.extend(chunk)
            continue
        if len(api_calls_pref)>=2:
            print(
                f"[GROUP {split_name}_{idx:04d}] "
                f"api_calls_pref_len={len(api_calls_pref)}"
            )

        if len(api_calls_pref) > 2:
            rejected.extend(chunk)
            continue

        groups.append({
            "example_id": f"{split_name}_{idx:04d}",
            "sessions": chunk,
            "api_calls": common_calls,
            "api_calls_pref": api_calls_pref,
            # "value_group": vg
        })

    return groups, rejected, idx

# # --------------------
# # 6) value_group 라벨링
# # --------------------
# def detect_value_group_for_chunk(group_dialogues, vg_map):
#     """
#     반환:
#       final_value_group: str
#       conflict_info: dict
#         {
#           vg: {
#             "Domain.slot": { value: count, ... }
#           }
#         }
#     """
#     vg_slot_value_counter = defaultdict(lambda: defaultdict(Counter))
#     vg_triplets = defaultdict(set)

#     for dialog in group_dialogues:
#         seen_group_prefs = set()

#         for api_str in dialog.get("api_call", []):
#             domain, slots = parse_api(api_str)
#             if domain is None:
#                 continue

#             for slot, value in slots.items():
#                 raw = value.strip()
#                 dequoted = raw.strip('"')

#                 key = None
#                 if (domain, slot, raw) in vg_map:
#                     key = (domain, slot, raw)
#                 elif (domain, slot, dequoted) in vg_map:
#                     key = (domain, slot, dequoted)

#                 if key is None:
#                     continue

#                 domain_, slot_, value_ = key
#                 vg = vg_map[key]

#                 group_pref = vg_to_group_pref[vg]

#                 if group_pref in seen_group_prefs:
#                     continue    # ❌ 같은 dialogue에서 같은 group_preference 두 번 금지

#                 seen_group_prefs.add(group_pref)    

#                 vg_triplets[vg].add(key)
#                 vg_slot_value_counter[vg][(domain_, slot_)][value_] += 1

#     valid_vgs = {}
#     conflict_info = {}

#     for vg, triplets in vg_triplets.items():
#         # 조건 1: triplet 종류 2개 이상
#         if len(triplets) < 2:
#             continue

#         vg_conflicts = {}

#         for (domain, slot), value_counter in vg_slot_value_counter[vg].items():
#             frequent_values = {
#                 v: c for v, c in value_counter.items() if c >= 2
#             }
#             if len(frequent_values) >= 2:
#                 vg_conflicts[f"{domain}.{slot}"] = frequent_values

#         if vg_conflicts:
#             conflict_info[vg] = vg_conflicts
#             continue  # ❌ conflict 있는 vg는 label 후보에서 제외

#         valid_vgs[vg] = len(triplets)

#     # ---- label 결정 ----
#     if not valid_vgs:
#         return "NO_VALUE_GROUP", conflict_info

#     max_score = max(valid_vgs.values())
#     winners = [vg for vg, sc in valid_vgs.items() if sc == max_score]

#     if len(winners) != 1:
#         return "NO_VALUE_GROUP", conflict_info

#     winner_vg = winners[0]
#     winner_gp = vg_to_group_pref[winner_vg]

#     # 🔑 같은 group_preference 축에서
#     # 서로 다른 (domain, slot) 개수 계산
#     domain_slot_set = set()
#     for (domain, slot, _value) in vg_triplets[winner_vg]:
#         domain_slot_set.add((domain, slot))

#     if len(domain_slot_set) < 2:
#         # ❌ evidence 부족 → group_preference 승격 금지
#         return "NO_VALUE_GROUP", conflict_info

#     return winner_vg, conflict_info

def build_api_calls_pref(group_dialogues, vg_map, vg_to_group_pref):
    vg_max_triplets = defaultdict(set)
    for (domain, slot, value), vg in vg_map.items():
        vg_max_triplets[vg].add((domain, slot, value))
    vg_max_triplet_counts = {
        vg: len(triplets) for vg, triplets in vg_max_triplets.items()
    }

    vg_counter = defaultdict(list)

    for inst_id, dialog in enumerate(group_dialogues):
        dialogue_id = dialog.get("dialogue_id")

        for api_str in dialog.get("api_call", []):
            domain, slots = parse_api(api_str)
            if domain is None:
                continue

            for slot, value in slots.items():
                raw = value.strip()
                dequoted = raw.strip('"')

                key = None
                if (domain, slot, raw) in vg_map:
                    key = (domain, slot, raw)
                elif (domain, slot, dequoted) in vg_map:
                    key = (domain, slot, dequoted)

                if key is None:
                    continue

                vg = vg_map[key]

                vg_counter[vg].append({
                    "dialogue_id": dialogue_id,
                    "instance_id": inst_id,
                    "api_call": api_str,
                    "domain": domain,
                    "slot": slot,
                    "value": dequoted
                })

    results = []
    for vg, evidences in vg_counter.items():
        # unique_dialogues = {
        #     e["dialogue_id"] if e.get("dialogue_id") is not None else e["instance_id"]
        #     for e in evidences
        # }
        # count = len(unique_dialogues)
        count = len(evidences)

        # ❌ 규칙 1: 최소 2회 이상
        if count < MIN_PREF_COUNT:
            continue

        # ❌ 규칙 2: 동일 triple 반복만 있는 경우 제외
        unique_triples = {
            (e["domain"], e["slot"], e["value"])
            for e in evidences
        }
        if len(unique_triples) == 1 and vg != PREFERRED_VALUE_GROUP:
            continue
        max_triplets = vg_max_triplet_counts.get(vg)
        if max_triplets is not None and max_triplets > 2:
            # Avoid fully covering all possible triplets for a value_group.
            if len(unique_triples) >= max_triplets:
                continue

        if vg in EXCLUDED_VALUE_GROUPS:
            continue

        results.append({
            "value_group": vg,
            "group_preference": vg_to_group_pref.get(vg),
            "count": count,
            "evidence": evidences
        })

    return results

# --------------------
# 7) 그룹 생성 (bucket 제거: 전체에서 랜덤 그룹 만든 뒤 vg post-hoc 판정)
# --------------------
# def make_groups(data, split_name, vg_map):
#     groups = []
#     group_count = 0

#     dialogs = list(data)
#     random.shuffle(dialogs)

#     idx = 0
#     total = len(dialogs)

#     while idx < total:
#         remaining = total - idx
#         if remaining < 5:
#             break

#         group_size = min(random.randint(5, 10), remaining)
#         group_dialogues = dialogs[idx : idx + group_size]
#         idx += group_size

#         group_count += 1

#         api_calls_all = []
#         flat_api_calls = []

#         for i, dialog in enumerate(group_dialogues):
#             calls = dialog.get("api_call", [])
#             api_calls_all.append({
#                 "instance_id": i,
#                 "dialogue_id": dialog.get("dialogue_id"),
#                 "api_call": calls
#             })
#             flat_api_calls.extend(calls)

#         common_calls = merge_partial_api(flat_api_calls)
#         example_id = f"{split_name}_{group_count:04d}"

#         # ✅ group 만든 뒤에만 value_group 판정 (강한 bucket 가정 제거)
#         # final_value_group, conflict_info = detect_value_group_for_chunk(
#         #     group_dialogues, vg_map
#         # )

#         # metadata = {"value_group": final_value_group}
#         # if conflict_info:
#         #     metadata["value_group_conflict"] = conflict_info

#         groups.append({
#             "example_id": example_id,
#             "sessions": group_dialogues,
#             "api_calls": common_calls,
#             # "api_calls_all": api_calls_all,
#             "metadata": metadata,
#         })

#     return groups
def make_groups(
    data,
    split_name,
    vg_map,
    vg_to_group_pref,
    start_idx=0,
    do_stabilize=True   # 👈 추가
):
    regroup_pool = []
    groups = []
    group_idx = start_idx

    vg_buckets, fallback = split_dialogues_by_vg(
        data, vg_map, vg_to_group_pref
    )

    # 1️⃣ value_group 기반 그룹 먼저
    bucket_order = list(vg_buckets.keys())
    if PREFERRED_VALUE_GROUP in bucket_order:
        bucket_order.remove(PREFERRED_VALUE_GROUP)
        bucket_order.insert(0, PREFERRED_VALUE_GROUP)

    for vg in bucket_order:
        dialogs = vg_buckets[vg]
        vg_groups, rejected, group_idx = make_groups_from_bucket(
            dialogs, split_name, group_idx, bucket_vg=vg
        )
        # for g in vg_groups:
        #     g["value_group"] = vg
        groups.extend(vg_groups)
        regroup_pool.extend(rejected)

    # 2️⃣ 나머지 랜덤 그룹
    random.shuffle(fallback)
    i = 0
    while i + 4 < len(fallback):
        size = min(random.randint(5, 10), len(fallback) - i)
        chunk = fallback[i:i+size]
        i += size
        group_idx += 1

        # ✅ fallback도 동일하게 common_calls 생성
        flat_api_calls = []
        for dialog in chunk:
            flat_api_calls.extend(dialog.get("api_call", []))

        common_calls = merge_partial_api(flat_api_calls)
        api_calls_pref = build_api_calls_pref(
            chunk, vg_map, vg_to_group_pref
        )
        if any(p.get("count", 0) > MAX_PREF_COUNT for p in api_calls_pref):
            regroup_pool.extend(chunk)
            continue

        # fallback 그룹은 value_group 2개 초과면 버림
        if len(api_calls_pref) > 2:
            regroup_pool.extend(chunk)
            continue

        group = {
            "example_id": f"{split_name}_{group_idx:04d}",
            "sessions": chunk,
            "api_calls": common_calls,
            # "value_group": None
        }
        if api_calls_pref:
            group["api_calls_pref"] = api_calls_pref

        groups.append(group)

    if do_stabilize:
        groups, regroup_pool = stabilize_groups(
            groups,
            regroup_pool,
            split_name,
            vg_map,
            vg_to_group_pref
        )

    return groups, regroup_pool

def stabilize_groups(
    groups,
    regroup_pool,
    split_name,
    vg_map,
    vg_to_group_pref,
    max_iter=5,
    iter_idx=0
):
    print(
        f"[STABILIZE {iter_idx}] "
        f"groups={len(groups)} | pool={len(regroup_pool)}"
    )

    if iter_idx >= max_iter:
        return groups, regroup_pool

    max_group_idx = get_max_group_idx(groups, split_name)
    stable_groups = []
    newly_regrouped = []

    for g in groups:
        gp_to_vg = defaultdict(set)

        for x in g.get("api_calls_pref", []):
            gp = x["group_preference"]
            vg = x["value_group"]
            gp_to_vg[gp].add(vg)

        # ❌ 충돌 조건:
        # 하나의 group_preference 아래
        # value_group이 2개 이상이면 해체
        conflict = any(len(vgs) > 1 for vgs in gp_to_vg.values())

        if conflict:
            newly_regrouped.extend(g["sessions"])
        else:
            stable_groups.append(g)

    # 안정화 완료
    if not newly_regrouped:
        return stable_groups, regroup_pool

    # 해체된 dialogue 누적
    regroup_pool.extend(newly_regrouped)

    # 재그룹 시도
    new_groups, remaining_pool = make_groups(
        regroup_pool,
        split_name=split_name,
        vg_map=vg_map,
        vg_to_group_pref=vg_to_group_pref,
        start_idx=max_group_idx,
        do_stabilize=False   # 🔑 재귀 무한 방지
    )

    return stabilize_groups(
        stable_groups + new_groups,
        remaining_pool,
        split_name,
        vg_map,
        vg_to_group_pref,
        max_iter,
        iter_idx + 1
    )

# --------------------
# 9) 전체 파이프라인 (split별)
# --------------------
# if __name__ == "__main__":
#     save_base = "./result"

#     VALUE_GROUP_CSV = "common_pref.csv"
#     vg_map, vg_to_group_pref = load_value_group_map(VALUE_GROUP_CSV)

#     splits = ["dev", "test"]

#     for split in splits:
#         input_path = f"{save_base}/{split}_2.json"
#         excluded_path = f"{save_base}/{split}_3_excluded.json"
#         grouped_path = f"{save_base}/{split}_3.json"

#         with open(input_path, "r", encoding="utf-8") as f:
#             data = json.load(f)

#         filtered, excluded = filter_dialogues(data)

#         with open(excluded_path, "w", encoding="utf-8") as f:
#             json.dump(excluded, f, indent=2, ensure_ascii=False)

#         groups = make_groups(filtered, split_name=split, vg_map=vg_map)

#         with open(grouped_path, "w", encoding="utf-8") as f:
#             json.dump(groups, f, indent=2, ensure_ascii=False)

#         print(
#             f"[{split}] "
#             f"input={len(data)} | "
#             f"filtered={len(filtered)} | "
#             f"excluded={len(excluded)} | "
#             f"groups={len(groups)}"
#         )

if __name__ == "__main__":
    save_base = "./result"

    VALUE_GROUP_CSV = "common_pref.csv"
    vg_map, vg_to_group_pref = load_value_group_map(VALUE_GROUP_CSV)

    splits = ["dev"]

    for split in splits:
        input_path = f"{save_base}/{split}_2.json"
        api_invalid_path = f"{save_base}/{split}_3_api_invalid.json"
        grouped_path = f"{save_base}/{split}_3.json"
        regroup_path = f"{save_base}/{split}_3_regroup_pool.json"

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        filtered, api_invalid = filter_dialogues(data)

        with open(api_invalid_path, "w", encoding="utf-8") as f:
            json.dump(api_invalid, f, indent=2, ensure_ascii=False)

        groups, regroup_pool = make_groups(
            filtered,
            split_name=split,
            vg_map=vg_map,
            vg_to_group_pref=vg_to_group_pref
        )

        with open(grouped_path, "w", encoding="utf-8") as f:
            json.dump(groups, f, indent=2, ensure_ascii=False)

        with open(regroup_path, "w", encoding="utf-8") as f:
            json.dump(regroup_pool, f, indent=2, ensure_ascii=False)

        print(
            f"[{split}] "
            f"input={len(data)} | "
            f"api_invalid={len(api_invalid)} | "
            f"grouped={len(groups)} | "
            f"regroup_pool={len(regroup_pool)}"
        )
