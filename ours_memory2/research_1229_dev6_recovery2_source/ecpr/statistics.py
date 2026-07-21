"""Paired example-cluster bootstrap and one-sided cluster randomization."""

from __future__ import annotations

import random
from typing import Any


Count = tuple[int, int, int]


def _add(left: Count, right: Count) -> Count:
    return left[0] + right[0], left[1] + right[1], left[2] + right[2]


def f1(counts: Count) -> float:
    tp, fp, fn = counts
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def bmf1(counts: dict[str, Count]) -> float:
    return 0.5 * f1(counts.get("singleturn", (0, 0, 0))) + 0.5 * f1(
        counts.get("multiturn", (0, 0, 0))
    )


def _aggregate(
    clusters: dict[str, dict[str, dict[str, Count]]], cluster_ids: list[str], arm: str
) -> dict[str, Count]:
    totals = {"singleturn": (0, 0, 0), "multiturn": (0, 0, 0)}
    for cluster_id in cluster_ids:
        for mode in totals:
            totals[mode] = _add(totals[mode], clusters[cluster_id][arm].get(mode, (0, 0, 0)))
    return totals


def paired_cluster_statistics(
    clusters: dict[str, dict[str, dict[str, Count]]],
    *,
    bootstrap_draws: int,
    bootstrap_seed: int,
    randomization_draws: int,
    randomization_seed: int,
) -> dict[str, Any]:
    ids = sorted(clusters)
    if not ids:
        raise ValueError("no paired clusters")
    for cluster_id in ids:
        if set(clusters[cluster_id]) != {"baseline", "candidate"}:
            raise ValueError(f"cluster {cluster_id} is not paired")
    observed = bmf1(_aggregate(clusters, ids, "candidate")) - bmf1(
        _aggregate(clusters, ids, "baseline")
    )

    bootstrap_rng = random.Random(bootstrap_seed)
    bootstrap_deltas: list[float] = []
    for _ in range(bootstrap_draws):
        sample = [ids[bootstrap_rng.randrange(len(ids))] for _ in ids]
        bootstrap_deltas.append(
            bmf1(_aggregate(clusters, sample, "candidate"))
            - bmf1(_aggregate(clusters, sample, "baseline"))
        )
    bootstrap_deltas.sort()
    lower_index = max(0, int(0.025 * bootstrap_draws))
    upper_index = min(bootstrap_draws - 1, int(0.975 * bootstrap_draws))

    randomization_rng = random.Random(randomization_seed)
    at_least_observed = 0
    for _ in range(randomization_draws):
        permuted: dict[str, dict[str, Count]] = {
            "baseline": {"singleturn": (0, 0, 0), "multiturn": (0, 0, 0)},
            "candidate": {"singleturn": (0, 0, 0), "multiturn": (0, 0, 0)},
        }
        for cluster_id in ids:
            swap = bool(randomization_rng.getrandbits(1))
            for mode in ("singleturn", "multiturn"):
                base = clusters[cluster_id]["baseline"].get(mode, (0, 0, 0))
                cand = clusters[cluster_id]["candidate"].get(mode, (0, 0, 0))
                if swap:
                    base, cand = cand, base
                permuted["baseline"][mode] = _add(permuted["baseline"][mode], base)
                permuted["candidate"][mode] = _add(permuted["candidate"][mode], cand)
        delta = bmf1(permuted["candidate"]) - bmf1(permuted["baseline"])
        if delta >= observed - 1e-15:
            at_least_observed += 1
    return {
        "delta_bmf1": observed,
        "bootstrap_ci_95": [bootstrap_deltas[lower_index], bootstrap_deltas[upper_index]],
        "bootstrap_draws": bootstrap_draws,
        "bootstrap_seed": bootstrap_seed,
        "randomization_p_one_sided": (at_least_observed + 1) / (randomization_draws + 1),
        "randomization_draws": randomization_draws,
        "randomization_seed": randomization_seed,
        "cluster_count": len(ids),
    }

