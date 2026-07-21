from __future__ import annotations

import io
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "strict_g0_protected",
    ROOT / "scripts" / "strict_run" / "g0_protected.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load strict G0 protected child")
G0 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(G0)


RUN_ID = "exp7-strict-v6-20260718T010000Z-" + "c" * 32


class _BinaryStdin:
    def __init__(self, raw: bytes) -> None:
        self.buffer = io.BytesIO(raw)


class _UnreadableBinaryStdin:
    def __init__(self) -> None:
        self.buffer = self

    def read(self, *_args: object) -> bytes:
        raise AssertionError("stdin binding was consumed")


class StrictG0ProtectedBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="exp7-g0-producer-binding-"
        )
        self.base = Path(self.temporary.name)
        self.run_root = self.base / "strict-runs" / RUN_ID
        self.config = {
            "sealed_run_id": RUN_ID,
            "run_root": str(self.run_root),
            "paper_path": "/protected/paper.pdf",
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def producer_binding(self) -> dict[str, object]:
        context = {"namespace": "synthetic"}
        acceptance = {
            "schema": "experiments7-publication-transcript/v6",
            "sealed_run_id": RUN_ID,
            "run_root": str(self.run_root),
            "relative_path": "reservation-acceptance.json",
            "context": context,
            "emitted_monotonic_ns": 10,
        }
        acceptance_sha256 = G0.sha256_bytes(G0.canonical_json(acceptance))
        envelope_payload = {
            "schema": "experiments7-cp0-envelope-evidence/v6",
            "sealed_run_id": RUN_ID,
            "run_root": str(self.run_root),
            "acceptance_transcript_sha256": acceptance_sha256,
        }
        envelope_transcript = {
            "schema": "experiments7-publication-transcript/v6",
            "sealed_run_id": RUN_ID,
            "run_root": str(self.run_root),
            "relative_path": "frozen/envelope_evidence/artifact.json",
            "context": context,
            "emitted_monotonic_ns": 20,
        }
        envelope_evidence = {
            "schema": "experiments7-stage-artifact/v6",
            "relative_path": "frozen/envelope_evidence/artifact.json",
            "artifact_type": "regular",
            "sha256": G0.sha256_bytes(G0.canonical_json(envelope_payload)),
            "bytes": len(G0.canonical_json(envelope_payload)),
            "publication_transcript_sha256": G0.sha256_bytes(
                G0.canonical_json(envelope_transcript)
            ),
        }
        return {
            "schema": "experiments7-cp0-producer-binding/v6",
            "acceptance_transcript": acceptance,
            "envelope_artifact_evidence": envelope_evidence,
            "envelope_transcript": envelope_transcript,
            "envelope_payload": envelope_payload,
        }

    def test_provider_path_swap_after_hash_executes_only_held_bytes(self) -> None:
        provider_path = self.run_root / "frozen" / "provider" / "provider.py"
        provider_path.parent.mkdir(parents=True)
        original = b'VALUE = "held-original"\n'
        provider_path.write_bytes(original)
        provider_fd = os.open(provider_path, os.O_RDONLY | os.O_CLOEXEC)
        config = {
            "run_root": str(self.run_root),
            "provider_path": str(provider_path),
            "provider_sha256": G0.sha256_bytes(original),
            "g0_executable_sha256": G0._sha256_path(
                os.path.abspath(G0.__file__)
            ),
        }
        displaced = provider_path.with_name("provider-original.py")

        def swap_path() -> None:
            provider_path.rename(displaced)
            provider_path.write_bytes(b'VALUE = "path-replacement"\n')

        try:
            module = G0._load_provider(
                config,
                provider_fd,
                race_hook=swap_path,
            )
        finally:
            os.close(provider_fd)
        self.assertEqual(module.VALUE, "held-original")
        self.assertEqual(
            provider_path.read_bytes(), b'VALUE = "path-replacement"\n'
        )

    def test_valid_binding_records_strictly_later_operation_start(self) -> None:
        binding = self.producer_binding()
        with patch.object(G0.time, "monotonic_ns", return_value=30):
            attestation = G0._begin_protected_read(
                binding, self.config, "source-pre"
            )
        self.assertEqual(
            attestation["schema"],
            "experiments7-cp0-protected-read-attestation/v6",
        )
        self.assertEqual(attestation["operation"], "source-pre")
        self.assertEqual(attestation["started_monotonic_ns"], 30)

    def test_invalid_gate_fails_before_any_protected_open(self) -> None:
        cases: list[tuple[str, dict[str, object], int]] = []
        early = self.producer_binding()
        cases.append(("early", early, 20))

        wrong_hash = self.producer_binding()
        wrong_hash["envelope_artifact_evidence"] = dict(
            wrong_hash["envelope_artifact_evidence"]
        )
        wrong_hash["envelope_artifact_evidence"][
            "publication_transcript_sha256"
        ] = "0" * 64
        cases.append(("hash", wrong_hash, 30))

        wrong_context = self.producer_binding()
        wrong_context["envelope_transcript"] = dict(
            wrong_context["envelope_transcript"]
        )
        wrong_context["envelope_transcript"]["context"] = {
            "namespace": "other"
        }
        evidence = dict(wrong_context["envelope_artifact_evidence"])
        evidence["publication_transcript_sha256"] = G0.sha256_bytes(
            G0.canonical_json(wrong_context["envelope_transcript"])
        )
        wrong_context["envelope_artifact_evidence"] = evidence
        cases.append(("context", wrong_context, 30))

        for label, binding, started in cases:
            with self.subTest(label=label), patch.object(
                G0.time, "monotonic_ns", return_value=started
            ), patch.object(
                G0.os,
                "open",
                side_effect=AssertionError("protected open occurred before gate"),
            ):
                with self.assertRaises(RuntimeError):
                    G0._begin_protected_read(
                        binding, self.config, "source-pre"
                    )

    def test_main_source_and_paper_modes_use_canonical_stdin(self) -> None:
        result_keys = {
            "schema",
            "mode",
            "sealed_run_id",
            "run_root",
            "provider_evidence",
            "paper_write_probe",
            "producer_attestation",
        }
        attestation_keys = {
            "schema",
            "sealed_run_id",
            "run_root",
            "operation",
            "started_monotonic_ns",
            "acceptance_transcript_sha256",
            "envelope_artifact_evidence_sha256",
            "envelope_transcript_sha256",
            "envelope_context",
        }
        for mode in ("source-pre", "paper-pre"):
            provider = Mock()
            events: list[str] = []
            emitted: list[dict[str, object]] = []

            def apply_provider(_paths: tuple[object, ...]) -> dict[str, bool]:
                events.append("provider")
                return {"readonly": True}

            def started() -> int:
                events.append("attestation")
                return 30

            def denial(_path: str) -> dict[str, object]:
                events.append("protected-open")
                return {"denied": True}

            provider.apply_readonly_envelope.side_effect = apply_provider
            binding_raw = G0.canonical_json(self.producer_binding())
            argv = [
                "g0_protected.py",
                "--config",
                "frozen-config.json",
                "--provider-fd",
                "99",
                "--mode",
                mode,
                "--producer-binding-stdin",
            ]
            with (
                self.subTest(mode=mode),
                patch.object(G0.sys, "argv", argv),
                patch.object(G0.sys, "stdin", _BinaryStdin(binding_raw)),
                patch.object(G0, "_require_frozen_invocation"),
                patch.object(G0, "_load_config", return_value=self.config),
                patch.object(
                    G0,
                    "_load_canonical_json",
                    side_effect=AssertionError("binding path was opened"),
                ),
                patch.object(G0, "_load_provider", return_value=provider),
                patch.object(G0, "_denial_probe", side_effect=denial),
                patch.object(
                    G0,
                    "_build_source_records",
                    return_value=[{"record": "synthetic"}],
                ) as source_builder,
                patch.object(
                    G0,
                    "_build_paper_record",
                    return_value={"paper": "synthetic"},
                ) as paper_builder,
                patch.object(G0, "_emit", side_effect=emitted.append),
                patch.object(G0.time, "monotonic_ns", side_effect=started),
            ):
                self.assertEqual(G0.main(), 0)

            self.assertEqual(events, ["provider", "attestation", "protected-open"])
            self.assertEqual(len(emitted), 1)
            payload = emitted[0]
            expected_keys = result_keys | {
                "records" if mode == "source-pre" else "paper"
            }
            self.assertEqual(set(payload), expected_keys)
            attestation = payload["producer_attestation"]
            self.assertIsInstance(attestation, dict)
            self.assertEqual(set(attestation), attestation_keys)
            self.assertEqual(attestation["operation"], mode)
            if mode == "source-pre":
                source_builder.assert_called_once_with(self.config)
                paper_builder.assert_not_called()
            else:
                source_builder.assert_not_called()
                paper_builder.assert_called_once_with(self.config, provider)

    def test_main_rejects_invalid_stdin_before_provider_or_protected_open(
        self,
    ) -> None:
        wrong_schema = self.producer_binding()
        wrong_schema["schema"] = "wrong"
        canonical = G0.canonical_json(self.producer_binding())
        cases = (
            ("missing", b""),
            ("oversized", b"x" * (G0.PRODUCER_BINDING_MAX_BYTES + 1)),
            ("noncanonical", b'{"schema": "x"}\n'),
            ("trailing", canonical + b"\n"),
            ("semantic", G0.canonical_json(wrong_schema)),
        )
        for mode in ("source-pre", "paper-pre"):
            for label, raw in cases:
                argv = [
                    "g0_protected.py",
                    "--config",
                    "frozen-config.json",
                    "--provider-fd",
                    "99",
                    "--mode",
                    mode,
                    "--producer-binding-stdin",
                ]
                with (
                    self.subTest(mode=mode, label=label),
                    patch.object(G0.sys, "argv", argv),
                    patch.object(G0.sys, "stdin", _BinaryStdin(raw)),
                    patch.object(G0, "_require_frozen_invocation"),
                    patch.object(
                        G0, "_load_config", return_value=self.config
                    ),
                    patch.object(
                        G0,
                        "_load_provider",
                        side_effect=AssertionError(
                            "provider loaded before binding rejection"
                        ),
                    ) as load_provider,
                    patch.object(
                        G0.os,
                        "open",
                        side_effect=AssertionError(
                            "protected open occurred before binding rejection"
                        ),
                    ),
                ):
                    with self.assertRaises(RuntimeError):
                        G0.main()
                load_provider.assert_not_called()

    def test_main_source_and_paper_require_stdin_flag_without_reading(
        self,
    ) -> None:
        for mode in ("source-pre", "paper-pre"):
            argv = [
                "g0_protected.py",
                "--config",
                "frozen-config.json",
                "--provider-fd",
                "99",
                "--mode",
                mode,
            ]
            with (
                self.subTest(mode=mode),
                patch.object(G0.sys, "argv", argv),
                patch.object(G0.sys, "stdin", _UnreadableBinaryStdin()),
                patch.object(G0, "_require_frozen_invocation"),
                patch.object(G0, "_load_config", return_value=self.config),
                patch.object(
                    G0,
                    "_load_provider",
                    side_effect=AssertionError("provider loaded without flag"),
                ) as load_provider,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "requires producer binding stdin"
                ):
                    G0.main()
            load_provider.assert_not_called()

    def test_main_rejects_legacy_path_binding_flag(self) -> None:
        stderr = io.StringIO()
        argv = [
            "g0_protected.py",
            "--config",
            "frozen-config.json",
            "--provider-fd",
            "99",
            "--mode",
            "source-pre",
            "--producer-binding",
            "/protected/binding.json",
        ]
        with (
            patch.object(G0.sys, "argv", argv),
            patch.object(G0.sys, "stdin", _UnreadableBinaryStdin()),
            patch.object(G0.sys, "stderr", stderr),
            patch.object(
                G0.os,
                "open",
                side_effect=AssertionError("legacy binding path was opened"),
            ),
        ):
            with self.assertRaises(SystemExit) as raised:
                G0.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("unrecognized arguments: --producer-binding", stderr.getvalue())

    def test_verify_envelope_rejects_stdin_flag_without_consuming(self) -> None:
        argv = [
            "g0_protected.py",
            "--config",
            "frozen-config.json",
            "--provider-fd",
            "99",
            "--mode",
            "verify-envelope",
            "--producer-binding-stdin",
        ]
        with (
            patch.object(G0.sys, "argv", argv),
            patch.object(G0.sys, "stdin", _UnreadableBinaryStdin()),
            patch.object(G0, "_require_frozen_invocation"),
            patch.object(G0, "_load_config", return_value=self.config),
            patch.object(
                G0,
                "_load_provider",
                side_effect=AssertionError("provider loaded for rejected flag"),
            ) as load_provider,
            patch.object(
                G0.os,
                "open",
                side_effect=AssertionError("protected open occurred"),
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "verify-envelope must not accept producer binding stdin",
            ):
                G0.main()
        load_provider.assert_not_called()


if __name__ == "__main__":
    unittest.main()
