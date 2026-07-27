from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
VANILLA_INFERENCE_PATH = ROOT / "methods/vanilla_llm/inference.py"
RUN_INFERENCE_PATH = ROOT / "scripts/run_inference.py"
FULL_INFERENCE_PATH = ROOT / "scripts/run_local_full_inference.py"
PROVIDER_CONFIG_PATH = ROOT / "src/provider_config.py"
GVR_BUILD_PATH = ROOT / "ablations/gvr/build_memory.py"
RAG_BUILD_PATH = ROOT / "methods/rag/build_index.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VanillaOpenRouterTest(unittest.TestCase):
    def test_full_runner_uses_unified_openrouter_reasoning_controls(self) -> None:
        module = load_module("full_inference_openrouter", FULL_INFERENCE_PATH)
        args = SimpleNamespace(
            api_base="https://openrouter.ai/api/v1",
            model="qwen/qwen3-8b",
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            min_p=0.0,
            seed=42,
            thinking=False,
            reasoning_effort="none",
        )
        request = module.build_chat_completion_request(
            {"prompt": "test"},
            args=args,
            max_tokens=256,
        )

        self.assertNotIn("reasoning_effort", request)
        self.assertEqual(
            request["extra_body"]["reasoning"],
            {"effort": "none", "exclude": True},
        )
        self.assertNotIn("chat_template_kwargs", request["extra_body"])

    def test_full_runner_loads_openrouter_key_from_environment(self) -> None:
        module = load_module("full_inference_openrouter_parser", FULL_INFERENCE_PATH)
        argv = [
            "run_local_full_inference.py",
            "--method",
            "vanilla_llm",
            "--model",
            "qwen/qwen3-8b",
            "--api-base",
            "https://openrouter.ai/api/v1",
            "--output-dir",
            "/tmp/output",
            "--max-tokens",
            "256",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.dict(
                os.environ,
                {"OPENROUTER_API_KEY": "openrouter-key"},
                clear=False,
            ),
        ):
            args = module.build_parser()

        self.assertEqual(args.api_key, "openrouter-key")

    def test_openrouter_defaults_to_official_endpoint_and_environment_key(self) -> None:
        module = load_module("vanilla_openrouter_inference", VANILLA_INFERENCE_PATH)

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-key"}, clear=False):
            base_url, api_key = module.resolve_endpoint("openrouter", None, None)

        self.assertEqual(base_url, "https://openrouter.ai/api/v1")
        self.assertEqual(api_key, "or-key")

    def test_explicit_openrouter_endpoint_and_key_take_precedence(self) -> None:
        module = load_module("vanilla_openrouter_override", VANILLA_INFERENCE_PATH)

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "env-key"}, clear=False):
            base_url, api_key = module.resolve_endpoint(
                "openrouter",
                "https://openrouter.example/v1",
                "explicit-key",
            )

        self.assertEqual(base_url, "https://openrouter.example/v1")
        self.assertEqual(api_key, "explicit-key")

    def test_openrouter_requires_a_key(self) -> None:
        module = load_module("vanilla_openrouter_missing_key", VANILLA_INFERENCE_PATH)

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "OPENROUTER_API_KEY"):
                module.resolve_endpoint("openrouter", None, None)

    def test_common_cli_forwards_provider_and_conflict_input(self) -> None:
        module = load_module("run_inference_openrouter", RUN_INFERENCE_PATH)
        input_path = str(ROOT / "data/MPT_v2_conflict_mixed_noise.json")
        args = module.parser().parse_args(
            [
                "--method",
                "vanilla_llm",
                "--provider",
                "openrouter",
                "--turn",
                "single",
                "--input_path",
                input_path,
                "--pref_type",
                "medium",
                "--model",
                "openai/gpt-4o",
            ]
        )

        command = module.build_command(args)

        self.assertEqual(command[command.index("--provider") + 1], "openrouter")
        self.assertEqual(command[command.index("--input_path") + 1], input_path)
        self.assertNotIn("--api_key", command)

    def test_common_cli_supports_openrouter_for_all_methods(self) -> None:
        module = load_module("run_inference_openrouter_scope", RUN_INFERENCE_PATH)
        cases = {
            "rag": [],
            "mem0": [],
            "langmem": ["--memory_path", "/tmp/langmem.jsonl"],
            "ours_memory": ["--memory_path", "/tmp/ours.jsonl"],
        }

        for method, extra in cases.items():
            with self.subTest(method=method):
                args = module.parser().parse_args(
                    [
                        "--method",
                        method,
                        "--provider",
                        "openrouter",
                        "--turn",
                        "single",
                        "--pref_type",
                        "medium",
                        "--model",
                        "openai/gpt-4o",
                        *extra,
                    ]
                )
                command = module.build_command(args)

                self.assertEqual(
                    command[command.index("--provider") + 1],
                    "openrouter",
                )
                self.assertNotIn("--api_key", command)
                if method in {"rag", "langmem"}:
                    self.assertEqual(
                        command[command.index("--embedding_model") + 1],
                        "text-embedding-3-small",
                    )

    def test_openrouter_embedding_defaults_use_provider_qualified_model(self) -> None:
        module = load_module("provider_config_openrouter", PROVIDER_CONFIG_PATH)

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-key"}, clear=False):
            model, base_url, api_key = module.resolve_embedding_endpoint(
                "openrouter",
                "text-embedding-3-small",
                None,
                None,
            )

        self.assertEqual(model, "openai/text-embedding-3-small")
        self.assertEqual(base_url, "https://openrouter.ai/api/v1")
        self.assertEqual(api_key, "or-key")

    def test_gvr_ablation_accepts_openrouter_as_openai_compatible(self) -> None:
        module = load_module("gvr_openrouter", GVR_BUILD_PATH)
        captured = {}

        class FakeAsyncOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with patch.object(module, "AsyncOpenAI", FakeAsyncOpenAI):
            aggregator = module.PreferenceAggregator(
                model="openai/gpt-4o",
                provider="openrouter",
                api_key="or-key",
                api_base="https://openrouter.ai/api/v1",
            )

        self.assertEqual(aggregator.provider, "openrouter")
        self.assertEqual(
            captured,
            {
                "api_key": "or-key",
                "base_url": "https://openrouter.ai/api/v1",
            },
        )

    def test_rag_build_uses_openrouter_embedding_endpoint_and_model(self) -> None:
        module = load_module("rag_build_openrouter", RAG_BUILD_PATH)
        captured = {}

        class EmptyFrame:
            def __len__(self):
                return 0

            def iterrows(self):
                return iter(())

        class FakeEmbeddingFunction:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        class FakeCollection:
            pass

        class FakePersistentClient:
            def __init__(self, path):
                captured["db_path"] = path

            def get_or_create_collection(self, **kwargs):
                captured["collection_name"] = kwargs["name"]
                return FakeCollection()

        embedding_functions = SimpleNamespace(
            OpenAIEmbeddingFunction=FakeEmbeddingFunction
        )
        chromadb_module = ModuleType("chromadb")
        chromadb_module.PersistentClient = FakePersistentClient
        chromadb_utils_module = ModuleType("chromadb.utils")
        chromadb_utils_module.embedding_functions = embedding_functions
        chromadb_module.utils = chromadb_utils_module

        with (
            patch.object(module, "load_chains_dataset", return_value=EmptyFrame()),
            patch.dict(
                sys.modules,
                {
                    "chromadb": chromadb_module,
                    "chromadb.utils": chromadb_utils_module,
                },
            ),
        ):
            module.run_ingestion(
                input_path="unused.json",
                db_path="/tmp/rag-openrouter",
                collection_name="test",
                provider="openrouter",
                embedding_api_key="or-key",
                embedding_model="text-embedding-3-small",
            )

        self.assertEqual(captured["api_key"], "or-key")
        self.assertEqual(captured["api_base"], "https://openrouter.ai/api/v1")
        self.assertEqual(captured["model_name"], "openai/text-embedding-3-small")


if __name__ == "__main__":
    unittest.main()
