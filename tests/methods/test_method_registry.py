from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


from exp7.methods import (
    LangMemAdapter,
    Mem0Adapter,
    MethodRegistry,
    MethodRegistryError,
    PreferenceMemoryAdapter,
    RAGAdapter,
    VanillaLLMAdapter,
    builtin_method_registry,
)


EXPECTED_BUILTINS = (
    "langmem",
    "mem0",
    "preference_memory",
    "rag",
    "vanilla_llm",
)


class MethodRegistryTests(unittest.TestCase):
    def test_builtin_registry_has_exact_lazy_builtin_set(self) -> None:
        registry = builtin_method_registry()
        self.assertEqual(registry.method_ids, EXPECTED_BUILTINS)
        self.assertIsInstance(registry.get("vanilla_llm"), VanillaLLMAdapter)
        self.assertIsInstance(registry.get("rag"), RAGAdapter)
        self.assertIsInstance(registry.get("langmem"), LangMemAdapter)
        self.assertIsInstance(
            registry.get("preference_memory"),
            PreferenceMemoryAdapter,
        )

    def test_factories_construct_fresh_configured_adapters(self) -> None:
        registry = builtin_method_registry()
        first = registry.create("rag", state_relative="method_state/rag-a")
        second = registry.create("rag", state_relative="method_state/rag-b")
        self.assertIsInstance(first, RAGAdapter)
        self.assertIsInstance(second, RAGAdapter)
        self.assertIsNot(first, second)

        with self.assertRaisesRegex(MethodRegistryError, "cannot construct.*mem0"):
            registry.get("mem0")
        mem0 = registry.create("mem0", namespace="registry-test")
        self.assertIsInstance(mem0, Mem0Adapter)
        self.assertEqual(mem0.namespace, "registry-test")

    def test_unknown_duplicate_and_invalid_factories_fail_closed(self) -> None:
        registry = builtin_method_registry()
        with self.assertRaisesRegex(MethodRegistryError, "unknown method adapter"):
            registry.create("unknown")
        with self.assertRaisesRegex(MethodRegistryError, "duplicate method adapter"):
            registry.register_factory("rag", RAGAdapter)
        with self.assertRaisesRegex(MethodRegistryError, "duplicate method adapter"):
            registry.register(registry.get("vanilla_llm"))

        custom = MethodRegistry()
        custom.register_factory("rag", lambda: VanillaLLMAdapter())
        with self.assertRaisesRegex(MethodRegistryError, "identity mismatch"):
            custom.get("rag")
        with self.assertRaisesRegex(MethodRegistryError, "must be callable"):
            custom.register_factory("broken", None)  # type: ignore[arg-type]

    def test_registered_instances_reject_factory_options(self) -> None:
        adapter = VanillaLLMAdapter()
        registry = MethodRegistry((adapter,))
        self.assertIs(registry.get("vanilla_llm"), adapter)
        with self.assertRaisesRegex(MethodRegistryError, "does not accept factory options"):
            registry.create("vanilla_llm", state_relative="unexpected")

    def test_importing_builtins_needs_no_optional_sdk_or_network(self) -> None:
        script = r'''
import builtins
import socket

real_import = builtins.__import__
blocked = {"openai", "mem0", "langmem", "langchain", "langchain_core", "chromadb"}

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] in blocked:
        raise AssertionError(f"optional SDK imported: {name}")
    return real_import(name, globals, locals, fromlist, level)

def denied_network(*args, **kwargs):
    raise AssertionError("network access during import")

builtins.__import__ = guarded_import
socket.create_connection = denied_network

from exp7.methods import builtin_method_registry

registry = builtin_method_registry()
print(",".join(registry.method_ids))
'''
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(ROOT / "src")
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), ",".join(EXPECTED_BUILTINS))


if __name__ == "__main__":
    unittest.main()
