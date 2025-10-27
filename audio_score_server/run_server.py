#!/usr/bin/env python3
"""Helper script to launch the audio score server over TCP or a Unix socket."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

import uvicorn


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 9000


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the audio score server.")
    transport = parser.add_mutually_exclusive_group()
    transport.add_argument(
        "--unix-socket",
        help="Path to a Unix domain socket to bind. Overrides host/port when provided.",
    )
    transport.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"TCP host/interface to bind (default: {DEFAULT_HOST}).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"TCP port to bind (default: {DEFAULT_PORT}). Ignored when --unix-socket is set.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (useful for local development).",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Uvicorn log level (default: info).",
    )
    parser.add_argument(
        "--model-keys",
        help="Comma-separated list of model keys to load. Passed via AUDIO_SCORE_MODEL_KEYS.",
    )
    parser.add_argument(
        "--disable-cache-flush",
        action="store_true",
        help="Set AUDIO_SCORE_DISABLE_CACHE_FLUSH=1 to keep existing cache directory.",
    )
    return parser.parse_args(argv)


def configure_environment(args: argparse.Namespace) -> None:
    if args.model_keys:
        os.environ["AUDIO_SCORE_MODEL_KEYS"] = args.model_keys
    if args.disable_cache_flush:
        os.environ["AUDIO_SCORE_DISABLE_CACHE_FLUSH"] = "1"


def ensure_socket_path(path: Path) -> Path:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    configure_environment(args)

    uds: str | None = None
    host = args.host or DEFAULT_HOST
    port = args.port or DEFAULT_PORT

    if args.unix_socket:
        uds_path = ensure_socket_path(Path(args.unix_socket))
        uds = str(uds_path)
        # host/port ignored by uvicorn when uds is supplied.

    config = uvicorn.Config(
        "main:app",
        host=host,
        port=port,
        uds=uds,
        reload=args.reload,
        log_level=args.log_level,
    )
    server = uvicorn.Server(config)
    return server.run()


if __name__ == "__main__":
    raise SystemExit(main())
