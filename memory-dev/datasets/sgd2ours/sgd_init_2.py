# Normalize service call slots via mapping and write result/*_2.json plus deprecated_api_call.
import csv
import json
import re
from collections import defaultdict

FORCE_ALLOWED_SLOTS = {
    "GetBanks": {"recipient_account_type"},
    "GetBuses": {"departure_date", "departure_time", "destination", "fare_type", "group_size", "origin"},
    "GetDentists": {"city", "dentist_name", "offers_cosmetic_services", "phone_number"},
    "GetDoctors": {"appointment_date", "appointment_time", "city", "doctor_name", "type"},
    "GetEvents": {"category", "city", "date", "event_name", "event_type", "number_of_tickets", "time", "venue_address"},
    "GetFlights": {"airlines", "departure_date", "destination", "origin", "outbound_departure_time", "passengers",
                   "refundable", "return_date", "seating_class", "flight_class"},
    "GetHomes": {"area", "number_of_baths", "number_of_beds", "pets_allowed"},
    "GetHotels": {"average_star", "check_in_date", "check_out_date", "has_wifi", "hotel_name", "location",
                  "number_of_days", "number_of_rooms", "pets_welcome"},
    "GetHouseStays": {"has_laundry_service", "rating", "where_to"},
    "GetMedia": {"directed_by", "genre", "title"},
    "GetMovies": {"genre", "location", "movie_name", "number_of_tickets", "show_date", "show_time", "show_type",
                  "street_address", "theater_name"},
    "GetMusic": {"artist", "genre"},
    "GetRentalCars": {"car_type", "dropoff_date", "pickup_city", "pickup_date", "pickup_location", "pickup_time"},
    "GetRestaurants": {"city", "cuisine", "date", "has_live_music", "party_size", "price_range", "restaurant_name", "category",
                       "serves_alcohol", "street_address", "time", "number_of_seats"},
    "GetSalons": {"appointment_date", "appointment_time", "average_star", "city", "is_unisex", "street_address", "stylist_name"},
    "GetTravel": {"category", "free_entry", "good_for_kids", "location", "phone_number"},
    "GetWeather": {"city", "date", "humidity", "wind"},
}

# -----------------------------
# mapping table 로드
# -----------------------------
tsv_path = "./common-domain-slots.tsv"
mapping = {}
domain_slots = {}

with open(tsv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        domain = row["domain"].strip()
        s1 = row["slot_1"].strip()
        s2 = row["slot_2"].strip()
        val = row["value"].strip().lower()

        mapping.setdefault(domain, {})
        domain_slots.setdefault(domain, set())
        domain_slots[domain].update([s1, s2])

        if val == "y":
            mapping[domain][s1] = s2
            mapping[domain][s2] = s2

def extract_domain(api_str):
    m = re.match(r"(Get\w+)\(", api_str)
    return m.group(1) if m else None

def replace_slots(api_str, canonical_map, domain_slot_set, force_allowed):
    if not api_str or str(api_str).strip() == "":
        return None, [], []

    match = re.match(r"(Get\w+)\(", api_str)
    if not match:
        return None, [api_str], []

    api_name = match.group(1)

    pairs = re.findall(r'(\w+)\s*=\s*("[^"]*"|\d+|\w+)', api_str)
    if not pairs:
        return None, [api_str], []

    new_parts = []
    unmapped = []
    updated_slot_values = []

    for slot, value in pairs:
        value_clean = value.strip('"')
        if api_name in {"GetHotels", "GetSalons"} and slot == "average_rating":
            slot = "average_star"

        if slot in canonical_map:
            canon = canonical_map[slot]
            if api_name in {"GetHotels", "GetSalons"} and canon == "average_rating":
                canon = "average_star"
            new_parts.append(f"{canon}={value}")
            updated_slot_values.append({"domain": api_name, "slot": canon, "value": value_clean})

        elif slot in domain_slot_set:
            new_parts.append(f"{slot}={value}")
            updated_slot_values.append({"domain": api_name, "slot": slot, "value": value_clean})

        elif slot in force_allowed:
            new_parts.append(f"{slot}={value}")
            updated_slot_values.append({"domain": api_name, "slot": slot, "value": value_clean})

        else:
            # 🔥 de는 "dialogue-level api_call_de"로 모을 것
            unmapped.append(f"{api_name}({slot}={value})")

    if not new_parts:
        return None, unmapped, []

    return (f"{api_name}(" + ", ".join(new_parts) + ")", unmapped, updated_slot_values)

# -----------------------------
# split별 JSON 처리 (TURN-LEVEL service만 수정, api_call/api_call_de는 dialogue-level에 저장)
# -----------------------------
SPLITS = ["dev"]

for split in SPLITS:
    save_base = "./result"
    input_json = f"{save_base}/{split}_1.json"

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        # ✅ dialogue-level containers
        item_api_calls = {}       # domain -> last api
        item_api_call_de = []

        for turn in item.get("dialogue", []):
            if "deprecated_api_call" in turn:
                del turn["deprecated_api_call"]

            if turn.get("role") != "Assistant":
                continue
            if "service" not in turn:
                continue

            new_services = []

            for call in turn.get("service", []):
                domain = extract_domain(call)
                if not domain:
                    continue

                canonical_map = mapping.get(domain.replace("Get", ""), {})
                slot_set = domain_slots.get(domain.replace("Get", ""), set())
                force_allowed = FORCE_ALLOWED_SLOTS.get(domain, set())

                new_api, unmapped, _ = replace_slots(
                    call, canonical_map, slot_set, force_allowed
                )

                if new_api:
                    new_services.append(new_api)

                    # ✅ 핵심: 같은 도메인은 마지막 것만 남김
                    api_name = extract_domain(new_api)
                    if api_name:
                        item_api_calls[api_name] = new_api

                item_api_call_de.extend(unmapped)

            if new_services:
                turn["service"] = new_services

        # ✅ dialogue-level only
        item["api_call"] = list(item_api_calls.values())
        item["deprecated_api_call"] = item_api_call_de

    out_path = f"{save_base}/{split}_2.json"
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"{split} 처리 완료 → {out_path}")
