# Convert raw SGD dialogs to initial result/*_1.json with api_call, prefs, and updated_slot tracking.
# # GetBanks ['recipient_account_type']
# # GetBuses ['departure_date', 'departure_time', 'destination', 'fare_type', 'group_size', 'origin']
# GetDentists ['city', 'dentist_name', 'offers_cosmetic_services', 'phone_number']
# GetDoctors ['appointment_date', 'appointment_time', 'city', 'doctor_name', 'type']
# # GetEvents ['category', 'city', 'date', 'event_name', 'event_type', 'number_of_tickets', 'time', 'venue_address']
# # GetFlights ['airlines', 'departure_date', 'destination', 'origin', 'outbound_departure_time', 'passengers', 'refundable', 'return_date', 'seating_class']
# # GetHomes ['area', 'number_of_baths', 'number_of_beds', 'pets_allowed']
# # GetHotels ['average_rating', 'check_in_date', 'check_out_date', 'has_wifi', 'hotel_name', 'location', 'number_of_days', 'number_of_rooms', 'pets_welcome']
# GetHouseStays ['has_laundry_service', 'rating', 'where_to']
# # GetMedia ['directed_by', 'genre', 'title']
# # GetMovies ['genre', 'location', 'movie_name', 'number_of_tickets', 'show_date', 'show_time', 'show_type', 'street_address', 'theater_name']
# # GetMusic ['artist', 'genre']
# # GetRentalCars ['car_type', 'dropoff_date', 'pickup_city', 'pickup_date', 'pickup_location', 'pickup_time']
# # GetRestaurants ['city', 'cuisine', 'date', 'has_live_music', 'party_size', 'price_range', 'restaurant_name', 'serves_alcohol', 'street_address', 'time']
# GetRideSharing
# GetSalons ['appointment_date', 'appointment_time', 'average_rating', 'city', 'is_unisex', 'street_address', 'stylist_name']
# # GetTravel ['category', 'free_entry', 'good_for_kids', 'location', 'phone_number']
# # GetWeather ['city', 'date', 'humidity', 'wind']

import json
import csv
import os
from glob import glob
import re

def format_state(state):
    if not state:
        return None
    intent = state.get("active_intent", "")
    parts = []

    if state.get("requested_slots"):
        req = ",".join(state["requested_slots"])
        parts.append(f'requested_slots="{req}"')

    for slot, val in state.get("slot_values", {}).items():
        v = val[0] if isinstance(val, list) else val
        parts.append(f'{slot}="{v}"')

    return f'{intent}({", ".join(parts)})'

def extract_service_slot(frame, speaker):
    service = frame.get("service")
    if not service:
        return None

    if speaker == "SYSTEM":
        call = frame.get("service_call")
        if call:
            params = call.get("parameters", {})
            inner = ", ".join([f'{k}=\"{v}\"' for k, v in sorted(params.items())])
            return f"{service}({inner})"
    return None

domains = [
    'Banks_1', 'Buses_1', 'Buses_2', 'Calendar_1', 'Events_1', 'Events_2',
    'Flights_1', 'Flights_2', 'Homes_1', 'Hotels_1', 'Hotels_2', 'Hotels_3',
    'Media_1', 'Movies_1', 'Music_1', 'Music_2', 'RentalCars_1',
    'RentalCars_2', 'Restaurants_1', 'RideSharing_1', 'RideSharing_2',
    'Services_1', 'Services_2', 'Services_3', 'Travel_1', 'Weather_1'
]

domain_to_api = {}

# 1) 기본 규칙: Domain_X → GetDomain
for domain in domains:
    base = domain.split("_")[0]  # Hotels_2 > Hotels
    domain_to_api[domain] = "Get" + base   # > GetHotels

# 2) 특수 규칙 override
domain_to_api["Hotels_2"]   = "GetHouseStays"
domain_to_api["Services_1"] = "GetSalons"
domain_to_api["Services_2"] = "GetDentists"
domain_to_api["Services_3"] = "GetDoctors"

PREF_KEYWORDS = ["prefer", "always", "love", "go-to", "a fan"]

COMMON_PREF_PATH = "./common_pref.csv"

def load_common_pref(path):
    pref = set()
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pref.add((row["domain"].strip(), row["slot"].strip()))
    return pref

COMMON_PREF_SLOTS = load_common_pref(COMMON_PREF_PATH)

API_PATTERN = re.compile(r"(\w+)\((.*)\)$")

def parse_service_calls(service_calls):
    """
    ["GetHotels(location=\"Seattle\", number_of_rooms=\"1\")", ...]
    -> list of (api_name, {slot:value})
    """
    out = []
    for call in service_calls:
        m = API_PATTERN.match(call.strip())
        if not m:
            continue
        api_name = m.group(1)
        args = m.group(2).strip()

        slots = {}
        if args:
            for part in args.split(","):
                if "=" not in part:
                    continue
                k, v = part.split("=", 1)
                slot_name = k.strip()
                if api_name == "GetHotels" and slot_name == "average_rating":
                    slot_name = "average_star"
                slots[slot_name] = v.strip().strip('"')
        out.append((api_name, slots))
    return out


def normalize_average_rating_call(call_str, api_name):
    if api_name == "GetHotels" and "average_rating" in call_str:
        return call_str.replace("average_rating", "average_star")
    return call_str

def contains_preference(text):
    text_low = text.lower()
    return any(k in text_low for k in PREF_KEYWORDS)

AFFIRMATIVE_KEYWORDS = [
    "yes",
    "yeah",
    "yep",
    "sure",
    "okay",
    "ok",

    "sounds good",
    "sounds great",
    "sounds perfect",

    "correct",
    "exactly",

    "that works",
    "works for me",
    "work for me",

    "you got it",

    "thats it",        
    "thats right",     
    "that is fine",
    "thats fine",
    "that is right",

    "I could do that",
    
    "thanks",
    "super",
    "fine",
    "good",
    "perfect",
    "awesome",
    "excellent",
    "great",
    "cool"
]


def normalize_text(text):
    return re.sub(r"[^\w\s]", "", text.lower()).strip()

AFFIRMATIVE_KEYWORDS_NORM = [normalize_text(k) for k in AFFIRMATIVE_KEYWORDS]

def is_simple_affirmative(text):
    t = normalize_text(text)

    # ✅ 핵심: "that's it ..." 뒤에 질문이 붙어도 무조건 동의로 간주
    if t.startswith("thats it"):
        return True

    return any(k in t for k in AFFIRMATIVE_KEYWORDS_NORM)

# def parse_service_call(call_str):
#     # GetRestaurants(a="1", b="2") → {"a": "1", "b": "2"}
#     inside = call_str[call_str.find("(")+1 : call_str.rfind(")")]
#     if not inside.strip():
#         return {}
#     slots = {}
#     for part in inside.split(","):
#         k, v = part.split("=", 1)
#         slots[k.strip()] = v.strip().strip('"')
#     return slots

def get_api_name(call_str):
    return call_str.split("(", 1)[0]

def convert_dialogue(dialogue):
    out = {
        "dialogue_id": dialogue.get("dialogue_id", ""),
        "preference_utterance": [],
        # # "": "",
        # "instruction_type": "",
        "explicit_constraints": {},
        "api_call": [],
        "dialogue": []
    }

    api_call_dict = {}
    last_user_utterance = None

    # ✅ 1) 현재까지의 slot state 유지: (api_name, slot) -> value
    prev_state = {}

    for turn in dialogue["turns"]:
        role = "User" if turn["speaker"] == "USER" else "Assistant"
        msg = turn.get("utterance", "")

        entry = {"role": role, "message": msg}
        turn_api_calls = []  # turn-level service_call list

        # -------------------------
        # ① USER preference 문장 수집 + last_user_utterance 갱신
        # -------------------------
        if role == "User":
            last_user_utterance = msg
            if contains_preference(msg):
                out["preference_utterance"].append(msg)

        # -------------------------
        # ② SYSTEM API call 추출 → Assistant turn의 service로 넣기
        # -------------------------
        for frame in turn.get("frames", []):
            slot_candidate = extract_service_slot(frame, turn["speaker"])
            if not slot_candidate:
                continue

            if role == "Assistant":
                service_name = frame["service"]

                # 1) 매핑 우선
                if service_name in domain_to_api:
                    api_prefix = domain_to_api[service_name]
                else:
                    base = service_name.split("_")[0]
                    api_prefix = "Get" + base

                # prefix 교체
                paren_index = slot_candidate.find("(")
                slot_candidate = api_prefix + slot_candidate[paren_index:]
                slot_candidate = normalize_average_rating_call(
                    slot_candidate, api_prefix
                )

                api_call_dict[service_name] = slot_candidate
                turn_api_calls.append(slot_candidate)

        # -------------------------
        # ③ service가 있으면: updated_slot_value 계산 (네 정의 1~4)
        # -------------------------
        if turn_api_calls:
            entry["service"] = turn_api_calls

            # 2) 이번 turn의 service slot/value 파싱
            parsed = parse_service_calls(turn_api_calls)

            updated_slot_value = []

            # 여러 service call이 있을 수 있으니 전부 처리
            for api_name, curr_slots in parsed:
                for slot, curr_val in curr_slots.items():
                    key = (api_name, slot)

                    # (2) 비교: 이전에 존재하던 slot만 update 후보
                    if key in prev_state:
                        prev_val = prev_state[key]

                        # (3) 값이 달라진 slot만 + common_pref에 있는지 확인
                        if prev_val != curr_val and (api_name, slot) in COMMON_PREF_SLOTS:
                            # (옵션) 단순 동의면 update로 치지 않기
                            if last_user_utterance and is_simple_affirmative(last_user_utterance):
                                continue

                            updated_slot_value.append({
                                "domain": api_name,
                                "slot": slot,
                                "value": curr_val
                            })

                    # ✅ state 갱신은 항상 해야 함 (1번 정의 유지)
                    prev_state[key] = curr_val

            # (4) 하나라도 있으면 True
            if updated_slot_value:
                entry["updated_slot"] = True
                entry["updated_slot_value"] = updated_slot_value

        out["dialogue"].append(entry)

    out["api_call"] = list(api_call_dict.values())
    return out


# --------------------------------
# 🔥 dataset split 파일 위치
# --------------------------------
base = "/workspace/dstc8-schema-guided-dialogue"

SPLITS = ["dev"]

split_dirs = {
    split: os.path.join(base, split)
    for split in SPLITS
}

# split별 결과 저장
final_output = { "dev": [], "test": [], "train": [] }


# --------------------------------
# 🔥 변환 + split 별 accumulate
# --------------------------------
found_any = False

for split, folder in split_dirs.items():
    json_files = glob(os.path.join(folder, "*.json"))
    if not json_files:
        continue
    found_any = True

    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)

        # dialogues 구조 감지
        if isinstance(data, dict) and "dialogues" in data:
            dialogues = data["dialogues"]

        elif isinstance(data, list):
            dialogues = [d for d in data if isinstance(d, dict) and "turns" in d]

        elif isinstance(data, dict) and "turns" in data:
            dialogues = [data]

        else:
            continue

        for dlg in dialogues:
            final_output[split].append(convert_dialogue(dlg))

if not found_any:
    raise FileNotFoundError(
        f"No input JSON files found under {base}. "
        "Check the dataset path and split folder."
    )


# --------------------------------
# save 
# --------------------------------
save_base = "./result"

for split in SPLITS:
    out_path = os.path.join(save_base, f"{split}_1.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_output[split], f, indent=2, ensure_ascii=False)
    print(f"saved {split}")
