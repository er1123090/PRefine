#!/usr/bin/env python3
"""Forward a local TCP port to a remote vLLM port over SSH using Paramiko."""

from __future__ import annotations

import os
import select
import socket
import socketserver
import sys
import threading
import time
from dataclasses import dataclass

import paramiko


@dataclass(frozen=True)
class TunnelConfig:
    ssh_host: str
    ssh_port: int
    ssh_user: str
    ssh_password: str
    local_host: str
    local_port: int
    remote_host: str
    remote_port: int


class ForwardServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


class ForwardHandler(socketserver.BaseRequestHandler):
    ssh_transport: paramiko.Transport
    remote_host: str
    remote_port: int

    def handle(self) -> None:
        try:
            channel = self.ssh_transport.open_channel(
                "direct-tcpip",
                (self.remote_host, self.remote_port),
                self.request.getpeername(),
            )
        except Exception as exc:
            print(f"[tunnel] open_channel failed: {exc}", file=sys.stderr, flush=True)
            return

        if channel is None:
            print("[tunnel] open_channel returned None", file=sys.stderr, flush=True)
            return

        try:
            while True:
                readable, _, _ = select.select([self.request, channel], [], [], 10)
                if self.request in readable:
                    data = self.request.recv(65536)
                    if not data:
                        break
                    channel.sendall(data)
                if channel in readable:
                    data = channel.recv(65536)
                    if not data:
                        break
                    self.request.sendall(data)
        finally:
            channel.close()
            self.request.close()


def load_config() -> TunnelConfig:
    password = os.environ.get("VLLM_SSH_PASSWORD", "")
    if not password:
        raise SystemExit("VLLM_SSH_PASSWORD is required")
    return TunnelConfig(
        ssh_host=os.environ.get("REMOTE_HOST", "166.104.112.134"),
        ssh_port=int(os.environ.get("SSH_PORT", "14233")),
        ssh_user=os.environ.get("SSH_USER", "minseo"),
        ssh_password=password,
        local_host=os.environ.get("LOCAL_HOST", "127.0.0.1"),
        local_port=int(os.environ.get("LOCAL_PORT", "18004")),
        remote_host=os.environ.get("REMOTE_VLLM_HOST", "127.0.0.1"),
        remote_port=int(os.environ.get("REMOTE_VLLM_PORT", "8004")),
    )


def connect_transport(config: TunnelConfig) -> paramiko.Transport:
    sock = socket.create_connection((config.ssh_host, config.ssh_port), timeout=20)
    transport = paramiko.Transport(sock)
    transport.start_client(timeout=20)
    transport.auth_password(config.ssh_user, config.ssh_password)
    if not transport.is_authenticated():
        raise RuntimeError("SSH authentication failed")
    transport.set_keepalive(30)
    return transport


def main() -> int:
    config = load_config()
    transport = connect_transport(config)

    ForwardHandler.ssh_transport = transport
    ForwardHandler.remote_host = config.remote_host
    ForwardHandler.remote_port = config.remote_port

    server = ForwardServer((config.local_host, config.local_port), ForwardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(
        f"[tunnel] {config.local_host}:{config.local_port} -> "
        f"{config.ssh_host}:{config.remote_host}:{config.remote_port}",
        flush=True,
    )

    try:
        while transport.is_active():
            time.sleep(5)
    finally:
        server.shutdown()
        server.server_close()
        transport.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
