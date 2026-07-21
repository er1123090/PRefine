from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from ours_memory2.contracts import InputContractError, OutputFilename, normalize_scalar_id
from ours_memory2.jsonl import (
    append_jsonl_streams,
    encode_jsonl,
    read_examples_jsonl,
    validate_output_root,
)


class IdNormalizationTests(unittest.TestCase):
    def test_scalar_normalization(self) -> None:
        self.assertEqual(normalize_scalar_id(7, path="id"), "7")
        self.assertEqual(normalize_scalar_id(" 7 ", path="id"), "7")
        self.assertEqual(normalize_scalar_id(True, path="id"), "true")
        self.assertEqual(normalize_scalar_id(2.5, path="id"), "2.5")

    def test_invalid_identifiers_have_stable_path(self) -> None:
        for value in (None, "", "   ", [], {}, float("inf")):
            with self.subTest(value=value), self.assertRaises(InputContractError) as caught:
                normalize_scalar_id(value, path="example.example_id")
            self.assertEqual(caught.exception.path, "example.example_id")


class StrictJsonlTests(unittest.TestCase):
    FIXTURE = Path(__file__).parent / "fixtures" / "valid_two_examples.jsonl"

    def test_valid_rows_preserve_order_unicode_and_metadata(self) -> None:
        examples = read_examples_jsonl(self.FIXTURE)
        self.assertEqual([item.example_id for item in examples], ["7", "8"])
        self.assertEqual(examples[0].sessions[0].dialogue[0].message, "서울에서 조용한 곳을 찾아줘")
        self.assertEqual(examples[0].sessions[0].api_call, ())
        self.assertEqual(examples[1].metadata["metadata"]["order"], 2)

    def test_duplicate_after_normalization_reports_later_line(self) -> None:
        rows = [self._valid(7), self._valid("7")]
        with self._write(b"\n".join(rows) + b"\n") as path:
            with self.assertRaises(InputContractError) as caught:
                read_examples_jsonl(path)
        self.assertEqual(caught.exception.path, "input.line[2].example_id")

    def test_negative_matrix(self) -> None:
        cases = {
            "array": (b"[]\n", "input.line[1]"),
            "wrapper": (b'{"examples":[]}\n', "input.line[1].example_id"),
            "pretty": (b'{\n  "example_id": "x"\n}\n', "input.line[1]"),
            "utf8": (b"\xff\n", "input.line[1]"),
            "json": (b"{bad}\n", "input.line[1]"),
            "missing_id": (self._valid(None), "input.line[1].example_id"),
            "empty_id": (self._valid("  "), "input.line[1].example_id"),
            "sessions_missing": (b'{"example_id":"x"}\n', "input.line[1].sessions"),
            "sessions_type": (b'{"example_id":"x","sessions":{}}\n', "input.line[1].sessions"),
            "sessions": (b'{"example_id":"x","sessions":[]}\n', "input.line[1].sessions"),
            "session_object": (b'{"example_id":"x","sessions":[1]}\n', "input.line[1].sessions[0]"),
            "dialogue_missing": (b'{"example_id":"x","sessions":[{"api_call":[]}]}\n', "input.line[1].sessions[0].dialogue"),
            "dialogue_type": (b'{"example_id":"x","sessions":[{"dialogue":{},"api_call":[]}]}\n', "input.line[1].sessions[0].dialogue"),
            "dialogue": (b'{"example_id":"x","sessions":[{"dialogue":[],"api_call":[]}]}\n', "input.line[1].sessions[0].dialogue"),
            "turn": (b'{"example_id":"x","sessions":[{"dialogue":[1],"api_call":[]}]}\n', "input.line[1].sessions[0].dialogue[0]"),
            "role": (b'{"example_id":"x","sessions":[{"dialogue":[{"role":"","message":"m"}],"api_call":[]}]}\n', "input.line[1].sessions[0].dialogue[0].role"),
            "role_type": (b'{"example_id":"x","sessions":[{"dialogue":[{"role":1,"message":"m"}],"api_call":[]}]}\n', "input.line[1].sessions[0].dialogue[0].role"),
            "message": (b'{"example_id":"x","sessions":[{"dialogue":[{"role":"u","message":""}],"api_call":[]}]}\n', "input.line[1].sessions[0].dialogue[0].message"),
            "message_type": (b'{"example_id":"x","sessions":[{"dialogue":[{"role":"u","message":1}],"api_call":[]}]}\n', "input.line[1].sessions[0].dialogue[0].message"),
            "api_missing": (b'{"example_id":"x","sessions":[{"dialogue":[{"role":"u","message":"m"}]}]}\n', "input.line[1].sessions[0].api_call"),
            "api_type": (b'{"example_id":"x","sessions":[{"dialogue":[{"role":"u","message":"m"}],"api_call":{}}]}\n', "input.line[1].sessions[0].api_call"),
            "api_item": (b'{"example_id":"x","sessions":[{"dialogue":[{"role":"u","message":"m"}],"api_call":[1]}]}\n', "input.line[1].sessions[0].api_call[0]"),
        }
        for name, (payload, expected_path) in cases.items():
            with self.subTest(name=name), self._write(payload) as path:
                with self.assertRaises(InputContractError) as caught:
                    read_examples_jsonl(path)
                self.assertEqual(caught.exception.path, expected_path)

    def test_encode_jsonl_is_utf8_compact_and_newline_terminated(self) -> None:
        self.assertEqual(encode_jsonl([{"text": "한글", "n": 1}]), b'{"text":"\xed\x95\x9c\xea\xb8\x80","n":1}\n')

    @staticmethod
    def _valid(identifier: object) -> bytes:
        return json.dumps(
            {
                "example_id": identifier,
                "sessions": [
                    {"dialogue": [{"role": "user", "message": "m"}], "api_call": []}
                ],
            }
        ).encode("utf-8")

    class _WriteContext:
        def __init__(self, payload: bytes) -> None:
            self.payload = payload
            self.temp = tempfile.TemporaryDirectory()

        def __enter__(self) -> Path:
            path = Path(self.temp.name) / "input.jsonl"
            path.write_bytes(self.payload)
            return path

        def __exit__(self, *args: object) -> None:
            self.temp.cleanup()

    def _write(self, payload: bytes) -> _WriteContext:
        return self._WriteContext(payload)


class OutputRootTests(unittest.TestCase):
    def test_exact_component_and_resolved_alias_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            reserved = base / "ours_memory"
            reserved.mkdir()
            sentinel = reserved / "sentinel"
            sentinel.write_text("unchanged", encoding="utf-8")
            alias = base / "alias"
            alias.symlink_to(reserved, target_is_directory=True)
            rejected = (reserved, reserved / "new", alias, alias / "new")
            before = sentinel.stat()
            for value in rejected:
                with self.subTest(value=value), self.assertRaises(InputContractError) as caught:
                    validate_output_root(value)
                self.assertEqual(caught.exception.path, "output_root")
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "unchanged")
            self.assertEqual(sentinel.stat(), before)
            self.assertFalse((reserved / "new").exists())

    def test_similar_sibling_is_allowed_without_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "ours_memory2" / "future"
            expected = target.expanduser().resolve(strict=False)
            self.assertEqual(validate_output_root(target), expected)
            self.assertFalse(target.exists())


class TransactionalOutputTests(unittest.TestCase):
    FILENAMES = (
        OutputFilename.MEMORIES,
        OutputFilename.DRAFTS,
        OutputFilename.VERIFIERS,
    )

    def test_child_symlink_hardlink_and_untyped_filename_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root = base / "output"
            root.mkdir()
            outside = base / "outside.jsonl"
            outside.write_bytes(b"outside\n")
            (root / OutputFilename.MEMORIES.value).symlink_to(outside)
            with self.assertRaises(InputContractError) as caught:
                append_jsonl_streams(
                    root, {OutputFilename.MEMORIES: ({"row": 1},)}
                )
            self.assertEqual(
                caught.exception.path, "output.memories.jsonl"
            )
            self.assertEqual(outside.read_bytes(), b"outside\n")

            missing = base / "missing"
            with self.assertRaises(InputContractError) as caught:
                append_jsonl_streams(missing, {"../escape.jsonl": ({"row": 1},)})  # type: ignore[dict-item]
            self.assertEqual(caught.exception.path, "output.filename")
            self.assertFalse(missing.exists())

        for kind in ("symlink", "hardlink"):
            for blocked_index, filename in enumerate(self.FILENAMES):
                with self.subTest(kind=kind, blocked_index=blocked_index), tempfile.TemporaryDirectory() as temp:
                    base = Path(temp)
                    root = base / "output"
                    root.mkdir()
                    outside = base / f"outside-{kind}.jsonl"
                    outside.write_bytes(b"outside-stays-exact\n")
                    before: dict[OutputFilename, bytes] = {}
                    for index, current in enumerate(self.FILENAMES):
                        target = root / current.value
                        if index == blocked_index:
                            if kind == "symlink":
                                target.symlink_to(outside)
                            else:
                                os.link(outside, target)
                        else:
                            content = f'{{"existing":"{current.value}"}}\n'.encode()
                            target.write_bytes(content)
                            before[current] = content
                    with self.assertRaises(InputContractError) as caught:
                        append_jsonl_streams(root, self._streams())
                    self.assertEqual(
                        caught.exception.path, f"output.{filename.value}"
                    )
                    self.assertEqual(outside.read_bytes(), b"outside-stays-exact\n")
                    for current, content in before.items():
                        self.assertEqual((root / current.value).read_bytes(), content)
                    if kind == "symlink":
                        self.assertTrue((root / filename.value).is_symlink())
                    else:
                        self.assertEqual((root / filename.value).stat().st_nlink, 2)

        for blocked_index, filename in enumerate(self.FILENAMES):
            with self.subTest(
                kind="fifo", blocked_index=blocked_index
            ), tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / "output"
                root.mkdir()
                before: dict[OutputFilename, bytes] = {}
                for index, current in enumerate(self.FILENAMES):
                    target = root / current.value
                    if index == blocked_index:
                        os.mkfifo(target)
                    else:
                        content = f'{{"existing":"{current.value}"}}\n'.encode()
                        target.write_bytes(content)
                        before[current] = content
                with self.assertRaises(InputContractError) as caught:
                    append_jsonl_streams(root, self._streams())
                self.assertEqual(caught.exception.path, f"output.{filename.value}")
                for current, content in before.items():
                    self.assertEqual((root / current.value).read_bytes(), content)
                self.assertTrue((root / filename.value).is_fifo())

    def test_every_child_open_fstat_write_and_fsync_failure_rolls_back(self) -> None:
        for layout in ("absent", "mixed"):
            for operation in ("open", "fstat", "write", "fsync"):
                for failure_index in range(1, len(self.FILENAMES) + 1):
                    with self.subTest(
                        layout=layout,
                        operation=operation,
                        failure_index=failure_index,
                    ), tempfile.TemporaryDirectory() as temp:
                        root = Path(temp) / "output"
                        before = self._prepare_layout(root, layout)
                        self._inject_failure(
                            operation, failure_index, root, self._streams()
                        )
                        self._assert_rolled_back(root, layout, before)
                        append_jsonl_streams(root, self._streams())
                        self._assert_exact_retry(root, before)

    def test_partial_write_at_every_child_rolls_back_and_retry_is_exact(self) -> None:
        import ours_memory2.jsonl as jsonl_module

        for layout in ("absent", "mixed"):
            for target_index in range(1, len(self.FILENAMES) + 1):
                with self.subTest(
                    layout=layout, target_index=target_index
                ), tempfile.TemporaryDirectory() as temp:
                    root = Path(temp) / "output"
                    before = self._prepare_layout(root, layout)
                    real_write = jsonl_module.os.write
                    descriptor_order: list[int] = []
                    partially_written: set[int] = set()

                    def partial_then_fail(descriptor, payload):
                        if descriptor not in descriptor_order:
                            descriptor_order.append(descriptor)
                        position = descriptor_order.index(descriptor) + 1
                        if position != target_index:
                            return real_write(descriptor, payload)
                        if descriptor not in partially_written:
                            partially_written.add(descriptor)
                            return real_write(descriptor, payload[:1])
                        raise OSError("injected failure after partial write")

                    with mock.patch.object(
                        jsonl_module.os, "write", side_effect=partial_then_fail
                    ), self.assertRaises(OSError):
                        append_jsonl_streams(root, self._streams())
                    self._assert_rolled_back(root, layout, before)
                    append_jsonl_streams(root, self._streams())
                    self._assert_exact_retry(root, before)

    def test_directory_open_failure_removes_only_transaction_created_root(self) -> None:
        import ours_memory2.jsonl as jsonl_module

        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            parent = base / "transaction-created-parent"
            root = parent / "output"
            with mock.patch.object(
                jsonl_module.os,
                "open",
                side_effect=OSError("injected directory open failure"),
            ), self.assertRaises(OSError):
                append_jsonl_streams(root, self._streams())
            self.assertTrue(parent.is_dir())
            self.assertFalse(root.exists())

            existing = base / "pre-existing-output"
            existing.mkdir()
            with mock.patch.object(
                jsonl_module.os,
                "open",
                side_effect=OSError("injected directory open failure"),
            ), self.assertRaises(OSError):
                append_jsonl_streams(existing, self._streams())
            self.assertTrue(existing.is_dir())

    def _inject_failure(self, operation, failure_index, root, streams) -> None:
        import ours_memory2.jsonl as jsonl_module

        target = (
            jsonl_module._open_direct_child
            if operation == "open"
            else jsonl_module._write_all
            if operation == "write"
            else getattr(jsonl_module.os, operation)
        )
        calls = 0

        def failing(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == failure_index:
                raise OSError(f"injected {operation} failure")
            return target(*args, **kwargs)

        owner = jsonl_module if operation in {"open", "write"} else jsonl_module.os
        attribute = (
            "_open_direct_child"
            if operation == "open"
            else "_write_all"
            if operation == "write"
            else operation
        )
        with mock.patch.object(owner, attribute, side_effect=failing):
            with self.assertRaises(OSError):
                append_jsonl_streams(root, streams)

    def _prepare_layout(
        self, root: Path, layout: str
    ) -> dict[OutputFilename, bytes | None]:
        before = {filename: None for filename in self.FILENAMES}
        if layout == "mixed":
            root.mkdir()
            for index in (0, 2):
                filename = self.FILENAMES[index]
                content = f'{{"existing":"{filename.value}"}}\n'.encode()
                (root / filename.value).write_bytes(content)
                before[filename] = content
        return before

    def _assert_rolled_back(
        self,
        root: Path,
        layout: str,
        before: dict[OutputFilename, bytes | None],
    ) -> None:
        if layout == "absent":
            self.assertFalse(root.exists())
            return
        self.assertTrue(root.is_dir())
        for filename, content in before.items():
            path = root / filename.value
            if content is None:
                self.assertFalse(path.exists())
            else:
                self.assertEqual(path.read_bytes(), content)

    def _assert_exact_retry(
        self, root: Path, before: dict[OutputFilename, bytes | None]
    ) -> None:
        for filename, content in before.items():
            rows = [
                json.loads(line)
                for line in (root / filename.value).read_text().splitlines()
            ]
            expected = []
            if content is not None:
                expected.append({"existing": filename.value})
            expected.append({"retry": filename.value})
            self.assertEqual(rows, expected)

    def _streams(self):
        return {
            filename: ({"retry": filename.value},) for filename in self.FILENAMES
        }


if __name__ == "__main__":
    unittest.main()
