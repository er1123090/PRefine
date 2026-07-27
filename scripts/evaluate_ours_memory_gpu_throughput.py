#!/usr/bin/env python3
"""Evaluate live gpt-oss-20b construction throughput across four replicas."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path


OUTPUT_DIR = Path(
    "/data/minseo/experiment8/outputs/ours_memory/"
    "cross_model_mix600_high_reasoning_20260721/gpt_oss_20b"
)
LOG_DIR = OUTPUT_DIR / "logs"
RUN_TAG = os.environ.get(
    "RUN_TAG",
    "20260724_gpt_oss_20b_high_replica_dp4_c256_b8192_async_prompt_json_resume",
)
MIN_AGGREGATE_GENERATION_TPS = float(
    os.environ.get("MIN_AGGREGATE_GENERATION_TPS", "9000")
)
EXPECTED_IN_FLIGHT_PER_REPLICA = int(
    os.environ.get("EXPECTED_IN_FLIGHT_PER_REPLICA", "64")
)
THROUGHPUT_PATTERN = re.compile(r"Avg generation throughput: ([0-9.]+) tokens/s")


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.load(response)


def backend_is_healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=5) as response:
            return response.status == 200
    except OSError:
        return False


def recent_generation_tps(gpu_index: int, sample_count: int = 3) -> float:
    log_path = LOG_DIR / f"{RUN_TAG}.gpu{gpu_index}.server.log"
    values = [
        float(match.group(1))
        for match in THROUGHPUT_PATTERN.finditer(
            log_path.read_text(encoding="utf-8", errors="replace")
        )
    ]
    if not values:
        return 0.0
    recent = values[-sample_count:]
    return sum(recent) / len(recent)


def sample_gpu_utilization(sample_count: int = 5) -> dict[int, float]:
    samples: dict[int, list[int]] = {}
    for sample_index in range(sample_count):
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        for line in result.stdout.splitlines():
            gpu, utilization = (part.strip() for part in line.split(",", 1))
            samples.setdefault(int(gpu), []).append(int(utilization))
        if sample_index + 1 < sample_count:
            time.sleep(1)
    return {
        gpu: round(sum(values) / len(values), 1)
        for gpu, values in samples.items()
    }


def main() -> int:
    stats = get_json("http://127.0.0.1:8002/stats")
    backends = stats["backends"]
    generation_tps = {
        gpu: round(recent_generation_tps(gpu), 1) for gpu in range(4)
    }
    aggregate_generation_tps = round(sum(generation_tps.values()), 1)
    gpu_utilization = sample_gpu_utilization()

    failures = sum(backend["failures"] for backend in backends)
    healthy = all(backend_is_healthy(backend["url"]) for backend in backends)
    saturated = all(
        backend["in_flight"] >= EXPECTED_IN_FLIGHT_PER_REPLICA
        for backend in backends
    )
    passed = (
        healthy
        and failures == 0
        and saturated
        and aggregate_generation_tps >= MIN_AGGREGATE_GENERATION_TPS
    )

    print(
        json.dumps(
            {
                "status": "pass" if passed else "fail",
                "aggregate_generation_tps": aggregate_generation_tps,
                "minimum_generation_tps": MIN_AGGREGATE_GENERATION_TPS,
                "generation_tps_by_gpu": generation_tps,
                "average_gpu_utilization_percent": gpu_utilization,
                "healthy": healthy,
                "backend_failures": failures,
                "in_flight_by_backend": [
                    backend["in_flight"] for backend in backends
                ],
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
