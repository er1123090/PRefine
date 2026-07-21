from __future__ import annotations

import contextlib
import io
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import venv

from ours_memory2 import cli
from tests.test_cli_execution import _LoopbackServer, _success


ROOT = Path(__file__).parents[1]


class CliIsolationTests(unittest.TestCase):
    HELP_COMMANDS = (
        ("--help",),
        ("step1", "build", "--help"),
        ("step2", "single", "--help"),
        ("step2", "multi", "--help"),
    )

    def test_all_module_help_paths_work_from_tmp(self) -> None:
        env = _clean_env()
        env["PYTHONPATH"] = str(ROOT)
        env["PYTHONNOUSERSITE"] = "1"
        for command in self.HELP_COMMANDS:
            with self.subTest(command=command):
                result = subprocess.run(
                    [sys.executable, "-m", "ours_memory2", *command],
                    cwd="/tmp",
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)
                self.assertNotIn("Traceback", result.stderr)

    def test_rejected_output_root_precedes_other_side_effects_for_every_command(self) -> None:
        commands = (
            ("step1", "build", "--input", "/missing", "--endpoint", "http://127.0.0.1/x", "--model", "m"),
            ("step2", "single", "--manifest", "/missing", "--difficulty", "all", "--context", "memory_api", "--endpoint", "http://127.0.0.1/x", "--model", "m"),
            ("step2", "multi", "--manifest", "/missing", "--difficulty", "all", "--context", "memory_api", "--endpoint", "http://127.0.0.1/x", "--model", "m"),
        )
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            reserved = base / "ours_memory"
            reserved.mkdir()
            sentinel = reserved / "sentinel"
            sentinel.write_text("unchanged", encoding="utf-8")
            alias = base / "alias"
            alias.symlink_to(reserved, target_is_directory=True)
            live = ROOT.parent / "ours_memory"
            rejected_values = (
                live,
                live / "nonexisting-foundation-child",
                reserved,
                reserved / "future",
                alias,
                alias / "future",
            )
            for command in commands:
                for rejected in rejected_values:
                    with self.subTest(command=command, rejected=rejected), mock.patch(
                        "builtins.open", side_effect=AssertionError("open reached")
                    ), mock.patch(
                        "pathlib.Path.open", side_effect=AssertionError("Path.open reached")
                    ), mock.patch(
                        "pathlib.Path.mkdir", side_effect=AssertionError("mkdir reached")
                    ), mock.patch(
                        "os.open", side_effect=AssertionError("os.open reached")
                    ):
                        stderr = io.StringIO()
                        with contextlib.redirect_stderr(stderr):
                            code = cli.main((*command, "--output-root", str(rejected)))
                        self.assertEqual(code, 2)
                        self.assertIn("output_root", stderr.getvalue())
                        self.assertFalse((reserved / "future").exists())
                        self.assertTrue(sentinel.is_file())
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "unchanged")

    def test_import_every_runtime_module_from_tmp(self) -> None:
        legacy_root = ROOT.parent / "ours_memory"
        code = f"""
import importlib, importlib.abc, pathlib, pkgutil, sys
legacy = pathlib.Path({str(legacy_root)!r}).resolve(strict=False)
class BlockLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'ours_memory':
            raise RuntimeError('legacy import blocked')
        return None
def audit(event, args):
    if event == 'open' and args and isinstance(args[0], (str, bytes)):
        candidate = pathlib.Path(args[0]).resolve(strict=False)
        if candidate == legacy or legacy in candidate.parents:
            raise RuntimeError('legacy file access blocked')
sys.meta_path.insert(0, BlockLegacy())
sys.addaudithook(audit)
import ours_memory2
[importlib.import_module(m.name) for m in pkgutil.walk_packages(ours_memory2.__path__, ours_memory2.__name__+'.')]
print(ours_memory2.__file__)
"""
        env = _clean_env()
        env["PYTHONPATH"] = str(ROOT)
        env["PYTHONNOUSERSITE"] = "1"
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd="/tmp",
            env=env,
            text=True,
            capture_output=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(ROOT / "ours_memory2"), result.stdout)

    def test_offline_installed_console_provenance_inventory_and_help(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source_copy = Path(temp) / "source"
            shutil.copytree(
                ROOT,
                source_copy,
                ignore=shutil.ignore_patterns("__pycache__", "build", "*.egg-info"),
            )
            environment = Path(temp) / "venv"
            # Reuse only host build tooling (setuptools/wheel); provenance below
            # still requires the product package itself to live in this venv.
            venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment)
            python = environment / "bin" / "python"
            console = environment / "bin" / "ours-memory2"
            env = _clean_env()
            env.pop("PYTHONPATH", None)
            env.pop("PYTHONHOME", None)
            install = subprocess.run(
                [
                    str(python),
                    "-m",
                    "pip",
                    "install",
                    "--no-index",
                    "--no-deps",
                    "--no-build-isolation",
                    str(source_copy),
                ],
                cwd="/tmp",
                env=env,
                text=True,
                capture_output=True,
                timeout=60,
            )
            self.assertEqual(install.returncode, 0, install.stderr)
            provenance = subprocess.run(
                [
                    str(python),
                    "-c",
                    (
                        "import importlib.metadata as m,ours_memory2,pathlib,sysconfig;"
                        "module=pathlib.Path(ours_memory2.__file__).resolve();"
                        "pure=pathlib.Path(sysconfig.get_paths()['purelib']).resolve();"
                        "assert module.is_relative_to(pure),(module,pure);"
                        f"assert not module.is_relative_to(pathlib.Path({str(ROOT)!r}).resolve()),module;"
                        "files=[str(x) for x in m.distribution('ours-memory2').files];"
                        "assert not any(x.startswith('tests/') or '.omx' in x or 'probes/' in x or 'support.py' in x or 'reference_probe' in x for x in files),files;"
                        "print(module);print(len(files))"
                    ),
                ],
                cwd="/tmp",
                env=env,
                text=True,
                capture_output=True,
                timeout=15,
            )
            self.assertEqual(provenance.returncode, 0, provenance.stderr)
            self.assertIn("site-packages", provenance.stdout)
            for command in self.HELP_COMMANDS:
                with self.subTest(installed_command=command):
                    for executable in (
                        [str(console), *command],
                        [str(python), "-m", "ours_memory2", *command],
                    ):
                        result = subprocess.run(
                            executable,
                            cwd="/tmp",
                            env=env,
                            text=True,
                            capture_output=True,
                            timeout=10,
                        )
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertIn("usage:", result.stdout)

            input_path = Path(temp) / "installed-input.jsonl"
            input_path.write_text(
                '{"example_id":"installed","sessions":[{"dialogue":[{"role":"user","message":"hello"}],"api_call":[]}]}\n',
                encoding="utf-8",
            )
            output_root = Path(temp) / "installed-output"
            server = _LoopbackServer(
                [
                    _success('{"implicit_pref":"installed-pref"}', "installed-generate"),
                    _success('{"valid":true,"feedback":"ok"}', "installed-verify"),
                    _success('{"implicit_pref":"module-pref"}', "module-generate"),
                    _success('{"valid":true,"feedback":"ok"}', "module-verify"),
                ]
            )
            with server as endpoint:
                execution = subprocess.run(
                    [
                        str(console), "step1", "build", "--input", str(input_path),
                        "--output-root", str(output_root), "--endpoint", endpoint,
                        "--model", "installed-model", "--timeout", "2",
                    ],
                    cwd="/tmp",
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=15,
                )
                module_output = Path(temp) / "installed-module-output"
                module_execution = subprocess.run(
                    [
                        str(python), "-m", "ours_memory2", "step1", "build",
                        "--input", str(input_path), "--output-root", str(module_output),
                        "--endpoint", endpoint, "--model", "installed-model",
                        "--timeout", "2",
                    ],
                    cwd="/tmp",
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=15,
                )
            self.assertEqual(execution.returncode, 0, execution.stderr)
            self.assertEqual(module_execution.returncode, 0, module_execution.stderr)
            self.assertEqual(len(server.requests), 4)
            self.assertEqual(
                [path.name for path in sorted(output_root.iterdir())],
                ["drafts.jsonl", "memories.jsonl", "verifiers.jsonl"],
            )
            self.assertEqual(
                [path.name for path in sorted(module_output.iterdir())],
                ["drafts.jsonl", "memories.jsonl", "verifiers.jsonl"],
            )


def _clean_env() -> dict[str, str]:
    env = os.environ.copy()
    blocked = {
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "VLLM_HOST",
        "PYTHONHOME",
    }
    for key in tuple(env):
        if key.upper() in blocked or key.upper() in {
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "NO_PROXY",
        }:
            env.pop(key, None)
    return env


if __name__ == "__main__":
    unittest.main()
