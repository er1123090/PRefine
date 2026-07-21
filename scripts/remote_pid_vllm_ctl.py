#!/usr/bin/env python3
"""Manage one remote vLLM server by PID file over SSH.

This is intentionally narrower than the workspace-level remote_vllm_ctl.py:
stop only touches the PID recorded in --pid-file, so it is safe on shared
machines that may have unrelated vLLM servers.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from typing import Any

import paramiko


def q(value: Any) -> str:
    return shlex.quote(str(value))


def read_secrets() -> dict[str, str]:
    raw = sys.stdin.readline()
    if not raw.strip():
        return {}
    data = json.loads(raw)
    return {str(k): str(v) for k, v in data.items() if v is not None}


def connect(args: argparse.Namespace, password: str | None) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=args.host,
        port=args.ssh_port,
        username=args.user,
        password=password or None,
        look_for_keys=True,
        allow_agent=True,
        timeout=args.timeout,
        banner_timeout=args.timeout,
        auth_timeout=args.timeout,
    )
    return client


def run(client: paramiko.SSHClient, cmd: str, timeout: int | None) -> tuple[int, str, str]:
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    stdin.close()
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    return code, out, err


def checked(client: paramiko.SSHClient, cmd: str, label: str, timeout: int | None) -> str:
    code, out, err = run(client, cmd, timeout=timeout)
    if code != 0:
        raise SystemExit(f"{label} failed exit={code}\nSTDOUT:\n{out}\nSTDERR:\n{err}")
    return out


def parser_for_model(model: str, requested: str) -> str:
    if requested != "auto":
        return requested
    if "Llama-3" in model:
        return "llama3_json"
    return "hermes"


def extra_env_items(raw_items: list[str]) -> dict[str, str]:
    items: dict[str, str] = {}
    for raw in raw_items:
        if "=" not in raw:
            raise SystemExit(f"--env must be KEY=VALUE, got: {raw}")
        key, value = raw.split("=", 1)
        if not key:
            raise SystemExit(f"--env key cannot be empty, got: {raw}")
        items[key] = value
    return items


def stop_shell(pid_file: str) -> str:
    return f"""
set -u
pid="$(cat {q(pid_file)} 2>/dev/null || true)"
if [ -z "$pid" ]; then
  rm -f {q(pid_file)}
  echo "NO_PID"
  exit 0
fi
cmd="$(ps -p "$pid" -o args= 2>/dev/null || true)"
if [ -z "$cmd" ]; then
  rm -f {q(pid_file)}
  echo "STALE_PID $pid"
  exit 0
fi
case "$cmd" in
  *vllm*serve*) ;;
  *)
    echo "PID_NOT_OURS $pid $cmd"
    rm -f {q(pid_file)}
    exit 0
    ;;
esac
collect_tree() {{
  local root="$1"
  echo "$root"
  local child
  for child in $(pgrep -P "$root" 2>/dev/null || true); do
    collect_tree "$child"
  done
}}
pids="$(collect_tree "$pid" | tac | tr '\\n' ' ')"
echo "STOPPING $pids"
kill -TERM $pids 2>/dev/null || true
for _ in $(seq 1 30); do
  kill -0 "$pid" 2>/dev/null || break
  sleep 1
done
if kill -0 "$pid" 2>/dev/null; then
  kill -KILL $pids 2>/dev/null || true
fi
rm -f {q(pid_file)}
"""


def start_shell(args: argparse.Namespace, hf_token: str | None) -> str:
    parser = parser_for_model(args.model, args.tool_parser)
    env_items = {
        "CUDA_VISIBLE_DEVICES": args.gpu,
        "HF_HOME": args.hf_home,
        "VLLM_LOGGING_LEVEL": "INFO",
    }
    if hf_token:
        env_items["HF_TOKEN"] = hf_token
        env_items["HUGGING_FACE_HUB_TOKEN"] = hf_token
    if args.cpath:
        env_items["CPATH"] = args.cpath
        env_items["C_INCLUDE_PATH"] = args.cpath
        env_items["CPLUS_INCLUDE_PATH"] = args.cpath
    env_items.update(extra_env_items(args.env))
    env_prefix = " ".join(f"{q(k)}={q(v)}" for k, v in env_items.items() if v)
    serve_parts = [
        q(args.vllm_bin),
        "serve",
        q(args.model),
        "--host",
        "0.0.0.0",
        "--port",
        str(args.remote_port),
        "--tensor-parallel-size",
        str(args.tp_size),
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        q(parser),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--trust-remote-code",
        "--disable-custom-all-reduce",
    ]
    if args.max_num_seqs:
        serve_parts.extend(["--max-num-seqs", str(args.max_num_seqs)])
    if args.extra_args:
        serve_parts.append(args.extra_args)
    serve_cmd = " ".join(serve_parts)
    return f"""
set -euo pipefail
mkdir -p {q(args.log_dir)}
bash -lc {q(stop_shell(args.pid_file))}
python3 - <<'PY'
import socket
import time
port = {args.remote_port}
deadline = time.time() + {args.port_wait_seconds}
last = None
while True:
    s = socket.socket()
    try:
        s.bind(("0.0.0.0", port))
        break
    except OSError as exc:
        last = exc
        if time.time() >= deadline:
            raise SystemExit(f"PORT_BUSY {{port}}: {{last}}")
        time.sleep(2)
    finally:
        s.close()
PY
nohup env {env_prefix} {serve_cmd} > {q(args.log_file)} 2>&1 < /dev/null &
echo $! > {q(args.pid_file)}
sleep 3
pid="$(cat {q(args.pid_file)})"
if ! kill -0 "$pid" 2>/dev/null; then
  tail -n 80 {q(args.log_file)} 2>/dev/null || true
  exit 1
fi
echo "STARTED $pid model={args.model} gpu={args.gpu} port={args.remote_port}"
"""


def status_shell(args: argparse.Namespace) -> str:
    return f"""
set -u
echo "PID_FILE {args.pid_file}"
pid="$(cat {q(args.pid_file)} 2>/dev/null || true)"
if [ -n "$pid" ]; then
  echo "PID $pid"
  ps -p "$pid" -o pid,ppid,stat,etime,args 2>/dev/null || true
else
  echo "PID none"
fi
echo "VLLM_PROCS"
pgrep -af 'vllm serve|python.*vllm|run_.*vllm' \
  | sed -E 's/(HF_TOKEN|HUGGING_FACE_HUB_TOKEN)=[^ ]+/\1=<redacted>/g' || true
echo "GPU"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
echo "COMPUTE_APPS"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true
echo "PORTS"
(ss -ltnp 2>/dev/null || netstat -ltnp 2>/dev/null || true) | grep -E ':{args.remote_port}\\b|:800[0-9]\\b' || true
echo "LOG_TAIL"
tail -n 80 {q(args.log_file)} 2>/dev/null || true
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "stop", "status"])
    parser.add_argument("--host", required=True)
    parser.add_argument("--ssh-port", type=int, default=14233)
    parser.add_argument("--user", default="minseo")
    parser.add_argument("--model", default="")
    parser.add_argument("--gpu", default="2,3")
    parser.add_argument("--remote-port", type=int, default=8004)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--max-num-seqs", type=int, default=0)
    parser.add_argument("--port-wait-seconds", type=int, default=120)
    parser.add_argument("--vllm-bin", default="/data/minseo/.venvs/vllm/bin/vllm")
    parser.add_argument("--hf-home", default="/data/minseo/.cache/huggingface")
    parser.add_argument("--cpath", default="")
    parser.add_argument("--log-dir", default="/data/minseo/.logs")
    parser.add_argument("--log-file", default="/data/minseo/.logs/remote_pid_vllm.log")
    parser.add_argument("--pid-file", default="/data/minseo/.logs/remote_pid_vllm.pid")
    parser.add_argument("--tool-parser", default="auto")
    parser.add_argument("--extra-args", default="")
    parser.add_argument("--env", action="append", default=[], help="Extra environment variable as KEY=VALUE")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    secrets = read_secrets()
    client = connect(args, secrets.get("password"))
    try:
        if args.action == "status":
            print(checked(client, status_shell(args), "status", args.timeout))
        elif args.action == "stop":
            print(checked(client, stop_shell(args.pid_file), "stop", args.timeout))
            print(checked(client, status_shell(args), "status", args.timeout))
        else:
            if not args.model:
                raise SystemExit("--model is required for start")
            print(checked(client, start_shell(args, secrets.get("hf_token")), "start", args.timeout))
            print(checked(client, status_shell(args), "status", args.timeout))
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
