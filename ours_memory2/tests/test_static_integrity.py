from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import sys
import unittest


ROOT = Path(__file__).parents[1]
PACKAGE = ROOT / "ours_memory2"


class StaticIntegrityTests(unittest.TestCase):
    def test_all_python_parses_and_compiles(self) -> None:
        sources = sorted((*PACKAGE.rglob("*.py"), *(ROOT / "tests").rglob("*.py")))
        self.assertTrue(sources)
        for path in sources:
            source = path.read_text(encoding="utf-8")
            ast.parse(source, filename=str(path))
            compile(source, str(path), "exec")

    def test_runtime_imports_and_literals_are_isolated(self) -> None:
        for path in sorted(PACKAGE.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            parents = {
                child: parent
                for parent in ast.walk(tree)
                for child in ast.iter_child_nodes(parent)
            }
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self.assertNotIn("ours_memory", {alias.name.split(".")[0] for alias in node.names})
                elif isinstance(node, ast.ImportFrom) and node.module:
                    self.assertNotEqual(node.module.split(".")[0], "ours_memory")
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if node.value == "ours_memory":
                        self.assertEqual(path, PACKAGE / "jsonl.py")
                        owner = node
                        while owner in parents and not isinstance(
                            owner, (ast.FunctionDef, ast.AsyncFunctionDef)
                        ):
                            owner = parents[owner]
                        self.assertIsInstance(owner, ast.FunctionDef)
                        self.assertEqual(owner.name, "validate_output_root")
                    self.assertNotIn("/ours_memory/", node.value)
                    if "/" in node.value or "\\" in node.value:
                        components = {
                            component
                            for component in re.split(r"[/\\]+", node.value)
                            if component
                        }
                        self.assertNotIn("ours_memory", components, (path, node.value))
                elif isinstance(node, ast.Call):
                    function_name = _call_name(node.func)
                    if function_name in {
                        "__import__",
                        "importlib.import_module",
                        "import_module",
                    } and node.args and isinstance(node.args[0], ast.Constant):
                        self.assertNotEqual(
                            str(node.args[0].value).split(".")[0], "ours_memory"
                        )

    def test_runtime_imports_are_stdlib_or_package_local(self) -> None:
        allowed = set(sys.stdlib_module_names) | {"ours_memory2"}
        for path in sorted(PACKAGE.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    roots = {alias.name.split(".")[0] for alias in node.names}
                    self.assertTrue(roots <= allowed, (path, roots - allowed))
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    root = node.module.split(".")[0]
                    self.assertIn(root, allowed, (path, root))

    def test_manifest_has_no_dependencies_and_console_is_exact(self) -> None:
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('dependencies = []', text)
        self.assertIn('ours-memory2 = "ours_memory2.cli:main"', text)
        for forbidden in ("openai", "torch", "vllm"):
            self.assertNotIn(forbidden, text.lower())
        self.assertNotIn("ScriptedProvider", "".join(
            path.read_text(encoding="utf-8") for path in PACKAGE.rglob("*.py")
        ))

    def test_target_has_no_symlinks_shells_or_scope_expansion(self) -> None:
        for path in ROOT.rglob("*"):
            self.assertFalse(path.is_symlink(), path)
        self.assertFalse(list(ROOT.rglob("*.sh")))
        runtime_names = {path.name for path in PACKAGE.glob("*.py")}
        forbidden = {"evaluation.py", "server.py", "launcher.py", "legacy.py"}
        self.assertTrue(runtime_names.isdisjoint(forbidden))

    def test_normalization_ledger_is_valid_and_complete(self) -> None:
        ledger = json.loads((ROOT / "intentional_normalizations.json").read_text(encoding="utf-8"))
        names = {item["name"] for item in ledger["normalizations"]}
        self.assertEqual(
            names,
            {
                "scalar_ids_to_strings",
                "strict_line_local_utf8_jsonl",
                "strict_verifier_shape",
                "provider_errors_are_explicit",
                "source_context_alias",
            },
        )

    def test_normalizations_are_behavioral_not_name_only(self) -> None:
        from ours_memory2.contracts import InputContractError, ProviderError, normalize_scalar_id
        from ours_memory2.providers import OpenAICompatibleProvider
        from ours_memory2.step1 import parse_verifier_decision
        from ours_memory2.step2_common import normalize_context

        self.assertEqual(normalize_scalar_id(7, path="id"), "7")
        with self.assertRaises(ValueError):
            parse_verifier_decision('{"valid":"yes"}')
        with self.assertRaises(InputContractError):
            normalize_context("api-only")
        with self.assertRaises(ProviderError):
            OpenAICompatibleProvider(endpoint="http://127.0.0.1:bad/x", model="m")


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


if __name__ == "__main__":
    unittest.main()
