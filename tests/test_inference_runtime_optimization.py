from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
RUN_INFERENCE_PATH = ROOT / "scripts/run_inference.py"
RUNTIME_COMMON_PATH = ROOT / "src/exp4_runtime/common.py"
LOCAL_FULL_INFERENCE_PATH = ROOT / "scripts/run_local_full_inference.py"


def load_run_inference_module():
    spec = importlib.util.spec_from_file_location(
        "experiment8_run_inference",
        RUN_INFERENCE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_local_full_inference_module():
    spec = importlib.util.spec_from_file_location(
        "experiment8_local_full_inference",
        LOCAL_FULL_INFERENCE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


class InferenceRuntimeOptimizationTest(unittest.TestCase):
    def test_local_population_forwards_dataset_and_conflict_filter(self) -> None:
        module = load_local_full_inference_module()
        args = SimpleNamespace(
            query="hint",
            input_path="/tmp/MPT_v2_0725.json",
            exclude_easy_conflict=True,
            expected_count=0,
            method="vanilla_llm",
        )
        with patch.object(
            module.population_runtime,
            "build_population",
            return_value=[],
        ) as build_population:
            rows = module.prepare_population(args)

        self.assertEqual(rows, [])
        build_population.assert_called_once_with(
            query="hint",
            context_type="diag-apilist",
            input_path="/tmp/MPT_v2_0725.json",
            exclude_easy_conflict=True,
        )

    def test_local_full_inference_forwards_optional_reasoning_effort(self) -> None:
        module = load_local_full_inference_module()
        row = {"prompt": "Return exactly: GetWeather(location=\"Seoul\")"}
        base_args = {
            "model": "gpt-oss-20b",
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
            "top_k": -1,
            "min_p": 0.0,
            "thinking": False,
        }

        without_effort = module.build_chat_completion_request(
            row,
            args=SimpleNamespace(**base_args, reasoning_effort=None),
            max_tokens=256,
        )
        with_effort = module.build_chat_completion_request(
            row,
            args=SimpleNamespace(**base_args, reasoning_effort="low"),
            max_tokens=256,
        )

        self.assertNotIn("reasoning_effort", without_effort)
        self.assertEqual(with_effort["reasoning_effort"], "low")
        self.assertFalse(
            with_effort["extra_body"]["chat_template_kwargs"]["enable_thinking"]
        )

    def test_async_client_receives_timeout_and_retry_policy(self) -> None:
        runtime_dir = str(RUNTIME_COMMON_PATH.parent)
        if runtime_dir not in sys.path:
            sys.path.insert(0, runtime_dir)
        spec = importlib.util.spec_from_file_location(
            "experiment8_runtime_common",
            RUNTIME_COMMON_PATH,
        )
        assert spec is not None and spec.loader is not None
        common = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = common
        spec.loader.exec_module(common)
        captured: dict[str, object] = {}

        class FakeAsyncOpenAI:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

        common.AsyncOpenAI = FakeAsyncOpenAI
        common.create_async_openai_client(
            api_key="EMPTY",
            base_url="http://127.0.0.1:8002/v1",
            allow_empty=True,
            timeout=7200.0,
            max_retries=0,
        )

        self.assertEqual(
            captured,
            {
                "api_key": "EMPTY",
                "base_url": "http://127.0.0.1:8002/v1",
                "timeout": 7200.0,
                "max_retries": 0,
            },
        )

    def test_optimized_methods_forward_runtime_settings(self) -> None:
        module = load_run_inference_module()
        cases = [
            ("vanilla_llm", "single", []),
            ("ours_memory", "single", ["--memory_path", "/tmp/memory.jsonl"]),
            ("ours_memory", "multi", ["--memory_path", "/tmp/memory.jsonl"]),
        ]

        for method, turn, extra in cases:
            with self.subTest(method=method, turn=turn):
                args = module.parser().parse_args(
                    [
                        "--method",
                        method,
                        "--turn",
                        turn,
                        "--pref_type",
                        "easy",
                        "--model",
                        "gpt-oss-20b",
                        "--concurrency",
                        "256",
                        "--base_url",
                        "http://127.0.0.1:8002/v1",
                        "--api_key",
                        "EMPTY",
                        "--request_timeout_seconds",
                        "7200",
                        "--client_max_retries",
                        "0",
                        *extra,
                    ]
                )

                command = module.build_command(args)

                self.assertEqual(option_value(command, "--concurrency"), "256")
                self.assertEqual(
                    option_value(command, "--base_url"),
                    "http://127.0.0.1:8002/v1",
                )
                self.assertEqual(option_value(command, "--api_key"), "EMPTY")
                self.assertEqual(
                    option_value(command, "--request_timeout_seconds"),
                    "7200.0",
                )
                self.assertEqual(option_value(command, "--client_max_retries"), "0")

    def test_other_methods_keep_the_original_client_defaults(self) -> None:
        module = load_run_inference_module()
        args = module.parser().parse_args(
            [
                "--method",
                "rag",
                "--turn",
                "single",
                "--pref_type",
                "easy",
                "--model",
                "gpt-oss-20b",
            ]
        )

        command = module.build_command(args)

        self.assertEqual(option_value(command, "--concurrency"), "20")
        self.assertNotIn("--request_timeout_seconds", command)
        self.assertNotIn("--client_max_retries", command)

    def test_dp4_launcher_contains_the_validated_server_settings(self) -> None:
        launcher = (
            ROOT / "scripts/run_gpt_oss20b_high_dp4_inference.sh"
        ).read_text(encoding="utf-8")

        self.assertIn('CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"', launcher)
        self.assertIn(
            'MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"',
            launcher,
        )
        self.assertIn('MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"', launcher)
        self.assertIn('CONCURRENCY="${CONCURRENCY:-256}"', launcher)
        self.assertIn('ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-1}"', launcher)
        self.assertIn("--async-scheduling", launcher)


if __name__ == "__main__":
    unittest.main()
