#!/usr/bin/env python3
"""Launch multiple audio score workers behind an Nginx reverse proxy."""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import List, Sequence


ROOT = Path(__file__).resolve().parent
RUN_SERVER = ROOT / "run_server.py"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start audio score workers behind an Nginx gateway.")
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of worker processes to launch (default: 2).",
    )
    parser.add_argument(
        "--socket-dir",
        default=str(ROOT / "run"),
        help="Directory for Unix domain sockets (default: ./run).",
    )
    parser.add_argument(
        "--listen-host",
        default="0.0.0.0",
        help="Host/interface for Nginx to listen on (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--listen-port",
        type=int,
        default=8080,
        help="Port for Nginx to expose (default: 8080).",
    )
    parser.add_argument(
        "--nginx",
        default="nginx",
        help="Path to the nginx executable (default: nginx found in PATH).",
    )
    parser.add_argument(
        "--model-keys",
        help="Comma-separated model keys to load inside each worker (passed via run_server.py).",
    )
    parser.add_argument(
        "--disable-cache-flush",
        action="store_true",
        help="Set AUDIO_SCORE_DISABLE_CACHE_FLUSH=1 so workers reuse existing cache.",
    )
    parser.add_argument(
        "--worker-log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Log level for worker uvicorn instances (default: info).",
    )
    parser.add_argument(
        "--worker-reload",
        action="store_true",
        help="Enable uvicorn reload mode for workers (development only).",
    )
    parser.add_argument(
        "--client-max-body",
        default="64m",
        help="client_max_body_size for Nginx (default: 64m).",
    )
    parser.add_argument(
        "--log-dir",
        default=str(ROOT / "logs"),
        help="Directory to store Nginx logs (default: ./logs).",
    )
    return parser.parse_args(argv)


def ensure_dependencies(nginx_binary: str) -> None:
    if not RUN_SERVER.exists():
        raise FileNotFoundError(f"run_server.py not found at {RUN_SERVER}")
    if shutil.which(nginx_binary) is None:
        raise RuntimeError(f"Nginx binary '{nginx_binary}' not found. Install nginx or specify --nginx path.")


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_socket_paths(socket_dir: Path, workers: int) -> List[Path]:
    ensure_directory(socket_dir)
    sockets = []
    for idx in range(workers):
        sock = socket_dir / f"audio_score_{idx}.sock"
        if sock.exists():
            sock.unlink()
        sockets.append(sock)
    return sockets


def start_workers(
    sockets: Sequence[Path],
    args: argparse.Namespace,
) -> List[subprocess.Popen]:
    worker_procs: List[subprocess.Popen] = []
    env = os.environ.copy()
    if args.disable_cache_flush:
        env["AUDIO_SCORE_DISABLE_CACHE_FLUSH"] = "1"
    if args.model_keys:
        env["AUDIO_SCORE_MODEL_KEYS"] = args.model_keys

    for idx, socket_path in enumerate(sockets):
        cmd = [
            sys.executable,
            str(RUN_SERVER),
            "--unix-socket",
            str(socket_path),
            "--log-level",
            args.worker_log_level,
        ]
        if args.worker_reload:
            cmd.append("--reload")
        proc = subprocess.Popen(cmd, env=env)
        worker_procs.append(proc)
        print(f"[INFO] Started worker #{idx} (PID={proc.pid}) listening on {socket_path}")
    return worker_procs


def generate_nginx_config(
    sockets: Sequence[Path],
    listen_host: str,
    listen_port: int,
    client_max_body: str,
    log_dir: Path,
    client_body_temp_dir: Path,
) -> str:
    ensure_directory(log_dir)
    ensure_directory(client_body_temp_dir)
    upstream_servers = "\n".join(f"        server unix:{sock};" for sock in sockets)
    access_log = (log_dir / "access.log").as_posix()
    error_log = (log_dir / "error.log").as_posix()
    pid_path = (log_dir / "nginx.pid").as_posix()
    client_body_temp_path = client_body_temp_dir.as_posix()
    config = textwrap.dedent(
        f"""
        worker_processes auto;
        daemon off;
        error_log {error_log} info;
        pid {pid_path};

        events {{
            worker_connections 1024;
        }}

        http {{
            log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                            '$status $body_bytes_sent "$http_referer" '
                            '"$http_user_agent" "$http_x_forwarded_for"';
            access_log {access_log} main;

            upstream audio_score_backend {{
                least_conn;
{upstream_servers}
            }}

            client_body_temp_path {client_body_temp_path};

            server {{
                listen {listen_host}:{listen_port};
                client_max_body_size {client_max_body};

                location / {{
                    proxy_pass http://audio_score_backend;
                    proxy_http_version 1.1;
                    proxy_set_header Host $host;
                    proxy_set_header X-Real-IP $remote_addr;
                    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
                    proxy_set_header X-Forwarded-Proto $scheme;
                }}
            }}
        }}
        """
    ).strip()
    return config


def write_nginx_config(config: str, directory: Path) -> Path:
    ensure_directory(directory)
    conf_path = directory / "nginx.conf"
    conf_path.write_text(config, encoding="utf-8")
    return conf_path


def start_nginx(nginx_binary: str, config_path: Path, error_log: Path) -> subprocess.Popen:
    prefix = config_path.parent
    cmd = [
        nginx_binary,
        "-p",
        str(prefix),
        "-c",
        config_path.name,
        "-g",
        f"error_log {error_log.as_posix()} info;",
    ]
    proc = subprocess.Popen(cmd)
    print(f"[INFO] Started Nginx (PID={proc.pid}) with config {config_path}")
    return proc


def terminate_process(proc: subprocess.Popen, name: str, timeout: float = 10.0) -> None:
    if proc.poll() is not None:
        return
    print(f"[INFO] Stopping {name} (PID={proc.pid})")
    try:
        proc.terminate()
        start = time.time()
        while proc.poll() is None and time.time() - start < timeout:
            time.sleep(0.2)
    except Exception:
        pass
    finally:
        if proc.poll() is None:
            proc.kill()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    ensure_dependencies(args.nginx)

    socket_dir = Path(args.socket_dir).expanduser()
    sockets = build_socket_paths(socket_dir, args.workers)
    worker_procs = start_workers(sockets, args)

    tempdir = Path(tempfile.mkdtemp(prefix="audio_score_nginx_"))
    ensure_directory(tempdir / "logs")
    log_dir_path = Path(args.log_dir).expanduser()
    ensure_directory(log_dir_path)
    log_dir_resolved = log_dir_path.resolve()
    client_body_temp_dir = log_dir_resolved / "client_body_temp"
    nginx_conf = generate_nginx_config(
        sockets,
        listen_host=args.listen_host,
        listen_port=args.listen_port,
        client_max_body=args.client_max_body,
        log_dir=log_dir_resolved,
        client_body_temp_dir=client_body_temp_dir,
    )
    config_path = write_nginx_config(nginx_conf, tempdir)
    error_log_path = log_dir_resolved / "error.log"

    try:
        nginx_proc = start_nginx(args.nginx, config_path, error_log_path)
    except Exception:
        for proc in worker_procs:
            terminate_process(proc, "worker")
        raise

    print(f"[INFO] Gateway listening on http://{args.listen_host}:{args.listen_port}")
    print("[INFO] Press Ctrl+C to stop.")

    try:
        while True:
            if nginx_proc.poll() is not None:
                raise RuntimeError("Nginx terminated unexpectedly.")
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[INFO] Received interrupt, shutting down...")
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}", file=sys.stderr)
    finally:
        terminate_process(nginx_proc, "nginx")
        for proc in worker_procs:
            terminate_process(proc, "worker")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
