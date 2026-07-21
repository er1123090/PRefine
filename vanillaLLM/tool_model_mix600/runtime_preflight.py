from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

import torch
import transformers
import vllm

from core import MODEL_ID, MODEL_REVISION, SEED, sha256_file, write_json_atomic


def _capture(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--chat-template")
    parser.add_argument("--prompt-renderer", default="chat-template")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--tool-call-parser", default="gemma4")
    parser.add_argument("--reasoning-parser")
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"))
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, required=True)
    parser.add_argument("--server-command-file", required=True)
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    visible = [0, 1, 2, 3]
    gpu_rows = _capture([
        "nvidia-smi", "--query-gpu=index,name,uuid,memory.total,driver_version", "--format=csv,noheader,nounits"
    ]).splitlines()
    if args.chat_template:
        prompt_renderer = {
            "kind": args.prompt_renderer,
            "path": str(Path(args.chat_template).resolve()),
            "sha256": sha256_file(args.chat_template),
        }
    else:
        prompt_renderer = {"kind": args.prompt_renderer}
    manifest = {
        "backend": "vllm-openai-server",
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "model_path": str(Path(args.model_path).resolve()),
        "python": platform.python_version(),
        "vllm": vllm.__version__,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "gpus": visible,
        "gpu_inventory": gpu_rows,
        "tensor_parallel_size": 4,
        "tool_call_parser": args.tool_call_parser,
        "reasoning_parser": args.reasoning_parser,
        "enable_auto_tool_choice": True,
        "prompt_renderer": prompt_renderer,
        "thinking": args.thinking,
        "reasoning_effort": args.reasoning_effort,
        "temperature": 0.0,
        "seed": SEED,
        "max_tokens": 256,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "client_concurrency": args.concurrency,
        "limit_mm_per_prompt": {"image": 0, "audio": 0},
        "server_command": Path(args.server_command_file).read_text(encoding="utf-8").strip(),
    }
    write_json_atomic(run_root / "runtime/runtime_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
