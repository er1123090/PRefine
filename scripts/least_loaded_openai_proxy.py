#!/usr/bin/env python3
"""Small least-loaded HTTP proxy for independent local vLLM replicas."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web


@dataclass
class BackendState:
    url: str
    forwarded: int = 0
    in_flight: int = 0
    completed: int = 0
    failures: int = 0


class LeastLoadedProxy:
    def __init__(self, backend_urls: list[str]) -> None:
        self.backends = [BackendState(url.rstrip("/")) for url in backend_urls]
        self.next_tie_breaker = 0
        self.session: ClientSession | None = None

    async def start(self, _: web.Application) -> None:
        self.session = ClientSession(
            connector=TCPConnector(limit=0),
            timeout=ClientTimeout(total=7200),
        )

    async def stop(self, _: web.Application) -> None:
        if self.session is not None:
            await self.session.close()

    def choose_backend(self) -> BackendState:
        minimum = min(backend.in_flight for backend in self.backends)
        for offset in range(len(self.backends)):
            index = (self.next_tie_breaker + offset) % len(self.backends)
            backend = self.backends[index]
            if backend.in_flight == minimum:
                self.next_tie_breaker = (index + 1) % len(self.backends)
                return backend
        raise RuntimeError("No backend available")

    async def health(self, _: web.Request) -> web.Response:
        return web.Response(text="ok")

    async def stats(self, _: web.Request) -> web.Response:
        return web.json_response(
            {
                "backends": [
                    {
                        "url": backend.url,
                        "forwarded": backend.forwarded,
                        "in_flight": backend.in_flight,
                        "completed": backend.completed,
                        "failures": backend.failures,
                    }
                    for backend in self.backends
                ]
            }
        )

    async def forward(self, request: web.Request) -> web.Response:
        if self.session is None:
            raise web.HTTPServiceUnavailable(text="Proxy is not ready")

        backend = self.choose_backend()
        backend.forwarded += 1
        backend.in_flight += 1
        target = f"{backend.url}{request.rel_url}"
        excluded_headers = {
            "connection",
            "content-length",
            "host",
            "transfer-encoding",
        }
        request_headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in excluded_headers
        }

        try:
            async with self.session.request(
                request.method,
                target,
                headers=request_headers,
                data=await request.read(),
            ) as response:
                body = await response.read()
                response_headers = {
                    key: value
                    for key, value in response.headers.items()
                    if key.lower() not in excluded_headers
                }
                backend.completed += 1
                return web.Response(
                    body=body,
                    status=response.status,
                    headers=response_headers,
                )
        except (asyncio.TimeoutError, OSError) as exc:
            backend.failures += 1
            raise web.HTTPBadGateway(text=f"Backend request failed: {exc}") from exc
        finally:
            backend.in_flight -= 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--backend", action="append", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    proxy = LeastLoadedProxy(args.backend)
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.on_startup.append(proxy.start)
    app.on_cleanup.append(proxy.stop)
    app.router.add_get("/health", proxy.health)
    app.router.add_get("/stats", proxy.stats)
    app.router.add_route("*", "/{path:.*}", proxy.forward)
    web.run_app(app, host=args.host, port=args.port, access_log=None)


if __name__ == "__main__":
    main()
