from __future__ import annotations

import ctypes
import errno
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PROVIDER_PATH = ROOT / "scripts" / "g0" / "provider.py"
SPEC = importlib.util.spec_from_file_location("g0_provider_seccomp_test", PROVIDER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to load provider module: {PROVIDER_PATH}")
PROVIDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROVIDER)


Instruction = tuple[int, int, int, int]


class _PrctlCapture:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int, int, int]] = []
        self.instructions: list[Instruction] = []

    def prctl(
        self,
        option: int,
        argument: int,
        program_pointer: object,
        fourth: int,
        fifth: int,
    ) -> int:
        program = ctypes.cast(
            program_pointer,
            ctypes.POINTER(PROVIDER.SockFprog),
        ).contents
        self.instructions = [
            (
                int(program.filter[index].code),
                int(program.filter[index].jt),
                int(program.filter[index].jf),
                int(program.filter[index].k),
            )
            for index in range(int(program.len))
        ]
        self.calls.append((option, argument, fourth, fifth))
        return 0


def _evaluate_filter(
    instructions: list[Instruction],
    *,
    arch: int,
    syscall_number: int,
) -> int:
    accumulator = 0
    program_counter = 0
    while program_counter < len(instructions):
        code, jump_true, jump_false, value = instructions[program_counter]
        if code == PROVIDER.BPF_LD_W_ABS:
            if value == PROVIDER.SECCOMP_DATA_ARCH_OFFSET:
                accumulator = arch
            elif value == PROVIDER.SECCOMP_DATA_NR_OFFSET:
                accumulator = syscall_number
            else:
                raise AssertionError(f"unexpected seccomp_data offset: {value}")
            program_counter += 1
            continue
        if code == PROVIDER.BPF_JMP_JEQ_K:
            program_counter += 1 + (
                jump_true if accumulator == value else jump_false
            )
            continue
        if code == PROVIDER.BPF_RET_K:
            return value
        raise AssertionError(f"unexpected BPF opcode: {code:#x}")
    raise AssertionError("BPF program fell off the end without a return action")


class G0ProviderSeccompArchitectureTests(unittest.TestCase):
    def _capture_program(self) -> _PrctlCapture:
        capture = _PrctlCapture()
        with patch.object(PROVIDER.platform, "machine", return_value="x86_64"), patch.object(
            PROVIDER,
            "LIBC",
            capture,
        ):
            PROVIDER._apply_seccomp_metadata_deny()
        self.assertEqual(
            capture.calls,
            [(PROVIDER.PR_SET_SECCOMP, PROVIDER.SECCOMP_MODE_FILTER, 0, 0)],
        )
        return capture

    def test_filter_checks_audit_arch_before_loading_syscall_number(self) -> None:
        instructions = self._capture_program().instructions
        deny_action = PROVIDER.SECCOMP_RET_ERRNO | errno.EPERM

        self.assertEqual(
            instructions[:4],
            [
                (
                    PROVIDER.BPF_LD_W_ABS,
                    0,
                    0,
                    PROVIDER.SECCOMP_DATA_ARCH_OFFSET,
                ),
                (
                    PROVIDER.BPF_JMP_JEQ_K,
                    1,
                    0,
                    PROVIDER.AUDIT_ARCH_X86_64,
                ),
                (PROVIDER.BPF_RET_K, 0, 0, PROVIDER.SECCOMP_RET_KILL_PROCESS),
                (
                    PROVIDER.BPF_LD_W_ABS,
                    0,
                    0,
                    PROVIDER.SECCOMP_DATA_NR_OFFSET,
                ),
            ],
        )
        self.assertEqual(
            len(instructions),
            5 + 2 * len(PROVIDER.DENIED_METADATA_SYSCALLS),
        )
        for index, syscall_number in enumerate(PROVIDER.DENIED_METADATA_SYSCALLS):
            instruction_index = 4 + 2 * index
            self.assertEqual(
                instructions[instruction_index],
                (PROVIDER.BPF_JMP_JEQ_K, 0, 1, syscall_number),
            )
            self.assertEqual(
                instructions[instruction_index + 1],
                (PROVIDER.BPF_RET_K, 0, 0, deny_action),
            )
        self.assertEqual(
            instructions[-1],
            (PROVIDER.BPF_RET_K, 0, 0, PROVIDER.SECCOMP_RET_ALLOW),
        )

    def test_arch_mismatch_cannot_reach_syscall_allow_path(self) -> None:
        instructions = self._capture_program().instructions
        denied_number = PROVIDER.DENIED_METADATA_SYSCALLS[0]
        allowed_number = max(PROVIDER.DENIED_METADATA_SYSCALLS) + 1

        for syscall_number in (denied_number, allowed_number):
            with self.subTest(syscall_number=syscall_number):
                self.assertEqual(
                    _evaluate_filter(
                        instructions,
                        arch=PROVIDER.AUDIT_ARCH_X86_64 ^ 1,
                        syscall_number=syscall_number,
                    ),
                    PROVIDER.SECCOMP_RET_KILL_PROCESS,
                )
        self.assertEqual(
            _evaluate_filter(
                instructions,
                arch=PROVIDER.AUDIT_ARCH_X86_64,
                syscall_number=denied_number,
            ),
            PROVIDER.SECCOMP_RET_ERRNO | errno.EPERM,
        )
        self.assertEqual(
            _evaluate_filter(
                instructions,
                arch=PROVIDER.AUDIT_ARCH_X86_64,
                syscall_number=allowed_number,
            ),
            PROVIDER.SECCOMP_RET_ALLOW,
        )

    def test_unsupported_machine_fails_before_prctl(self) -> None:
        capture = _PrctlCapture()
        with patch.object(PROVIDER.platform, "machine", return_value="aarch64"), patch.object(
            PROVIDER,
            "LIBC",
            capture,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "unsupported seccomp architecture: aarch64",
            ):
                PROVIDER._apply_seccomp_metadata_deny()
        self.assertEqual(capture.calls, [])
        self.assertEqual(capture.instructions, [])


if __name__ == "__main__":
    unittest.main()
