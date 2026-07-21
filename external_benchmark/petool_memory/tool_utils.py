"""Tool-name parsing and lightweight PEToolBench scoring helpers."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Set, Tuple


TOOL_PART_RE = re.compile(r"<([^>]+)>")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "based",
    "be",
    "by",
    "call",
    "current",
    "for",
    "from",
    "get",
    "i",
    "in",
    "including",
    "is",
    "it",
    "list",
    "me",
    "of",
    "on",
    "please",
    "retrieve",
    "show",
    "the",
    "to",
    "user",
    "want",
    "with",
}


def parse_tool_name(tool_name: str) -> Dict[str, str]:
    parts = TOOL_PART_RE.findall(tool_name or "")
    category = parts[0] if len(parts) >= 1 else ""
    provider = parts[1] if len(parts) >= 2 else ""
    operation = parts[2] if len(parts) >= 3 else (parts[-1] if parts else tool_name)
    namespace = f"<{category}>.<{provider}>" if category and provider else tool_name
    return {
        "tool_name": tool_name,
        "category": category,
        "provider": provider,
        "operation": operation,
        "namespace": namespace,
    }


def tokenize(text: str) -> List[str]:
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text or "")
    expanded = expanded.replace("/", " ").replace("-", " ").replace("_", " ")
    raw_tokens = re.findall(r"[A-Za-z0-9]+", expanded.lower())
    tokens = [token for token in raw_tokens if token and token not in STOPWORDS]
    token_set = set(tokens)
    if {"log", "out"} <= token_set or "logout" in token_set:
        tokens.extend(["logout", "auth", "session"])
    if {"log", "in"} <= token_set or "login" in token_set:
        tokens.extend(["login", "auth", "session"])
    if "signout" in token_set or "sign" in token_set and "out" in token_set:
        tokens.extend(["logout", "auth", "session"])
    if "signin" in token_set or "sign" in token_set and "in" in token_set:
        tokens.extend(["login", "auth", "session"])
    return tokens


def token_set(text: str) -> Set[str]:
    return set(tokenize(text))


def operation_family(operation: str, instruction: str = "") -> str:
    tokens = set(tokenize(f"{operation} {instruction}"))
    if tokens & {"login", "logout", "auth", "session", "signin", "signout"}:
        return "auth/session"
    if "pet" in tokens or "petstore" in tokens:
        return "petstore/pet"
    if "order" in tokens:
        return "commerce/order"
    if "inventory" in tokens:
        return "commerce/inventory"
    if "search" in tokens:
        return "search"
    if "brewery" in tokens:
        return "data/brewery"
    if tokens & {"match", "sport", "sports", "hockey"}:
        return "sports"
    if tokens & {"map", "distance", "geo", "geocode", "h3"}:
        return "mapping/location"
    if tokens & {"movie", "netflix", "show", "title"}:
        return "media/catalog"
    useful = [token for token, _ in Counter(tokenize(operation)).most_common(2)]
    return "/".join(useful) if useful else "other"


def history_signal(history_type: str, rating: Any, index: int, total: int) -> Tuple[str, float]:
    if history_type == "r":
        if rating == 1:
            return "positive_rating", 1.0
        if rating == 0:
            return "negative_rating", -1.0
        return "missing_rating", 0.0
    if history_type == "c":
        recency = index / max(total - 1, 1)
        return "chronological_positive", round(0.25 + 0.75 * recency, 4)
    return "observed_preferred", 1.0


def shorten(text: str, limit: int = 240) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def namespace_score_by_memory(memory: Dict[str, Any]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for item in memory.get("provider_preferences", []) or []:
        namespace = item.get("namespace")
        if not namespace:
            continue
        confidence = float(item.get("confidence", 0.0) or 0.0)
        polarity = item.get("polarity")
        if polarity in {"prefer", "recency_prefer"}:
            scores[namespace] = max(scores.get(namespace, 0.0), 4.0 * confidence)
        elif polarity == "weak_prefer":
            scores[namespace] = max(scores.get(namespace, 0.0), 2.0 * confidence)
        elif polarity == "avoid":
            scores[namespace] = min(scores.get(namespace, 0.0), -5.0 * max(confidence, 0.5))
    return scores


def operation_preferences_by_family(memory: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        item.get("operation_family"): item
        for item in memory.get("operation_preferences", []) or []
        if item.get("operation_family")
    }


def default_parameters_for_tool(tool: Dict[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for item in tool.get("required_parameters", []) or []:
        if isinstance(item, dict) and item.get("name"):
            params[item["name"]] = item.get("default", "")
    return params


def semantic_score(query: str, tool: Dict[str, Any]) -> float:
    parsed = parse_tool_name(str(tool.get("tool_name", "")))
    query_tokens = token_set(query)
    tool_text = " ".join(
        [
            parsed["operation"],
            str(tool.get("tool_description", "")),
            " ".join(str(param.get("description", "")) for param in tool.get("required_parameters", []) if isinstance(param, dict)),
        ]
    )
    tool_tokens = token_set(tool_text)
    overlap = len(query_tokens & tool_tokens)
    score = float(overlap)

    query_lower = (query or "").lower()
    operation_lower = parsed["operation"].lower()
    description_lower = str(tool.get("tool_description", "")).lower()
    if ("log out" in query_lower or "logout" in query_lower) and (
        "logout" in operation_lower or "logs out" in description_lower
    ):
        score += 10.0
    if ("log in" in query_lower or "login" in query_lower) and (
        "login" in operation_lower or "log in" in description_lower
    ):
        score += 10.0
    if not tool.get("required_parameters"):
        score += 0.1
    return score


def rank_candidate_tools(record: Dict[str, Any], memory: Dict[str, Any]) -> List[Tuple[float, int, Dict[str, Any], Dict[str, Any]]]:
    namespace_scores = namespace_score_by_memory(memory)
    op_prefs = operation_preferences_by_family(memory)
    ranked = []
    for index, tool in enumerate(record.get("candidate_tools", []) or []):
        parsed = parse_tool_name(str(tool.get("tool_name", "")))
        family = operation_family(parsed["operation"], str(tool.get("tool_description", "")))
        score = semantic_score(record.get("query", ""), tool)
        score += namespace_scores.get(parsed["namespace"], 0.0)

        family_pref = op_prefs.get(family, {})
        preferred = family_pref.get("preferred_namespaces", []) or []
        lower_priority = family_pref.get("lower_priority_namespaces", []) or []
        avoid = family_pref.get("avoid_namespaces", []) or []
        if parsed["namespace"] in preferred:
            score += 10.0
        if parsed["namespace"] in lower_priority:
            score -= 2.0
        if parsed["namespace"] in avoid:
            score -= 10.0

        ranked.append((score, -index, tool, parsed))
    return sorted(ranked, key=lambda item: (item[0], item[1]), reverse=True)


def aggregate_by_namespace(evidence: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    aggregate: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "score": 0.0,
            "support_count": 0,
            "negative_count": 0,
            "latest_history_index": -1,
            "category": "",
            "provider": "",
            "operation_families": set(),
            "evidence_indices": [],
        }
    )
    for item in evidence:
        namespace = item["namespace"]
        bucket = aggregate[namespace]
        bucket["category"] = item["category"]
        bucket["provider"] = item["provider"]
        bucket["score"] += item["weight"]
        if item["weight"] > 0:
            bucket["support_count"] += 1
        elif item["weight"] < 0:
            bucket["negative_count"] += 1
        bucket["latest_history_index"] = max(bucket["latest_history_index"], item["history_index"])
        bucket["operation_families"].add(item["operation_family"])
        bucket["evidence_indices"].append(item["history_index"])
    return aggregate
