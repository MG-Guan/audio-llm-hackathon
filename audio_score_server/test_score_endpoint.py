#!/usr/bin/env python3
"""Send a quick request to the audio score server for smoke testing."""
from __future__ import annotations

import argparse
import io
import random
import statistics
import sys
import time
import typing as t
import wave
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote as urlquote

import numpy as np
import requests

try:
    import requests_unixsocket
except ImportError:  # pragma: no cover - optional dependency
    requests_unixsocket = None


DEFAULT_MODEL = "whisper-medium_wavlm-large"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache"


def _generate_sine_wave(duration_s: float = 2.0, frequency_hz: float = 440.0, sample_rate: int = 16000) -> bytes:
    """Create a simple sine wave and return WAV bytes."""
    t_axis = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    waveform = 0.2 * np.sin(2 * np.pi * frequency_hz * t_axis)
    pcm = (waveform * np.iinfo(np.int16).max).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def _load_audio(path: str) -> bytes:
    with open(path, "rb") as handle:
        payload = handle.read()
    if not payload:
        raise ValueError(f"Audio file {path!r} is empty.")
    return payload


def _post_score(
    base_url: str,
    audio_bytes: bytes,
    uid: str,
    cid: str,
    model: str,
    unix_socket: str | None = None,
    timeout: float = 120.0,
) -> requests.Response:
    files = {
        "audio": ("sample.wav", audio_bytes, "audio/wav"),
    }
    data = {
        "uid": uid,
        "cid": cid,
        "model": model,
    }
    if unix_socket:
        if requests_unixsocket is None:
            raise RuntimeError(
                "requests-unixsocket is required for Unix socket testing. Install it via 'pip install requests-unixsocket'."
            )
        session = requests_unixsocket.Session()
        encoded_socket = urlquote(unix_socket, safe="")
        url = f"http+unix://{encoded_socket}/score"
        response = session.post(url, data=data, files=files, timeout=timeout)
    else:
        response = requests.post(f"{base_url.rstrip('/')}/score", data=data, files=files, timeout=timeout)
    return response


def _gather_cached_references(model_key: str, cache_root: Path) -> list[tuple[str, str]]:
    model_dir = cache_root / model_key
    if not model_dir.exists():
        raise FileNotFoundError(f"Cache directory not found for model '{model_key}': {model_dir}")

    references: list[tuple[str, str]] = []
    for video_dir in model_dir.iterdir():
        if not video_dir.is_dir():
            continue
        for clip_dir in video_dir.iterdir():
            if clip_dir.is_dir():
                references.append((video_dir.name, clip_dir.name))
    return references


def parse_args(argv: t.Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for /score endpoint.")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:9000",
        help="Base URL for the audio score server (default: http://127.0.0.1:9000).",
    )
    parser.add_argument(
        "--unix-socket",
        help="Path to a Unix domain socket exposed by the server. Overrides --url when provided.",
    )
    parser.add_argument(
        "--uid",
        help="Video identifier to send in the request. If omitted, one is sampled from the local cache.",
    )
    parser.add_argument(
        "--cid",
        help="Clip identifier to send in the request. If omitted, one is sampled from the local cache.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model key to score with (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--audio",
        help="Path to a WAV/MP3/etc. to upload. If omitted, a synthetic sine wave is generated.",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=1,
        help="Total number of requests to send (default: 1). Values >1 enable load testing.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Maximum number of concurrent requests (default: 1).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Request timeout in seconds (default: 120).",
    )
    return parser.parse_args(argv)


def _execute_request(
    url: str,
    unix_socket: str | None,
    audio_bytes: bytes,
    uid: str,
    cid: str,
    model: str,
    timeout: float,
) -> dict[str, t.Any]:
    start_time = time.perf_counter()
    try:
        response = _post_score(
            url,
            audio_bytes,
            uid,
            cid,
            model,
            unix_socket=unix_socket,
            timeout=timeout,
        )
        latency = time.perf_counter() - start_time
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "latency": time.perf_counter() - start_time,
            "exception": str(exc),
        }

    try:
        payload = response.json()
        json_error = None
    except ValueError as exc:  # noqa: BLE001
        payload = None
        json_error = str(exc)

    ok = response.status_code == 200 and json_error is None
    result: dict[str, t.Any] = {
        "ok": ok,
        "latency": latency,
        "status": response.status_code,
        "payload": payload,
    }
    if json_error is not None:
        result["json_error"] = json_error
    if not ok:
        result["body"] = response.text[:1000]
    return result


def _run_single(
    args: argparse.Namespace,
    audio_bytes: bytes,
    uid: str,
    cid: str,
) -> int:
    result = _execute_request(
        args.url,
        args.unix_socket,
        audio_bytes,
        uid,
        cid,
        args.model,
        timeout=args.timeout,
    )

    if result.get("exception"):
        print(f"[ERROR] Request failed: {result['exception']}", file=sys.stderr)
        return 2

    print(f"[INFO] Latency: {result['latency'] * 1000:.2f} ms")
    status = result.get("status")
    print(f"[INFO] Status: {status}")

    if not result["ok"]:
        if result.get("json_error"):
            print(f"[ERROR] Failed to decode JSON: {result['json_error']}", file=sys.stderr)
        if result.get("body"):
            print(result["body"])
        return 3

    payload = result.get("payload") or {}
    print("[INFO] Response JSON:")
    for key, value in payload.items():
        print(f"  {key}: {value}")
    return 0


def _run_load_test(
    args: argparse.Namespace,
    audio_bytes: bytes,
    uid: str,
    cid: str,
    total_requests: int,
    concurrency: int,
    references: list[tuple[str, str]] | None,
) -> int:
    print(
        f"[LOAD] Starting load test: {total_requests} requests, concurrency={concurrency}, model={args.model}",
    )

    latencies: list[float] = []
    failures: list[dict[str, t.Any]] = []
    status_counter: Counter[str] = Counter()

    start = time.perf_counter()

    def _task(_: int) -> dict[str, t.Any]:
        if references:
            ref_uid, ref_cid = random.choice(references)
        else:
            ref_uid, ref_cid = uid, cid
        return _execute_request(
            args.url,
            args.unix_socket,
            audio_bytes,
            ref_uid,
            ref_cid,
            args.model,
            timeout=args.timeout,
        )

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(_task, idx) for idx in range(total_requests)]
        for future in as_completed(futures):
            result = future.result()
            latencies.append(result.get("latency", 0.0))
            if result.get("ok"):
                status_counter["200"] += 1
            else:
                key = "exception" if result.get("exception") else str(result.get("status", "error"))
                status_counter[key] += 1
                failures.append(result)

    duration = time.perf_counter() - start
    success = status_counter.get("200", 0)
    errors = total_requests - success

    success_rate = (success / total_requests * 100.0) if total_requests else 0.0
    print(f"[LOAD] Completed in {duration:.2f}s ({success} success / {errors} failure, success_rate={success_rate:.2f}%).")
    if latencies:
        avg_ms = statistics.mean(latencies) * 1000
        p50_ms = statistics.median(latencies) * 1000
        p95_ms = statistics.quantiles(latencies, n=20)[18] * 1000 if len(latencies) >= 20 else max(latencies) * 1000
        min_ms = min(latencies) * 1000
        max_ms = max(latencies) * 1000
        throughput = total_requests / duration if duration > 0 else float("inf")
        print(f"[LOAD] Latency (ms): avg={avg_ms:.2f} p50={p50_ms:.2f} p95={p95_ms:.2f} min={min_ms:.2f} max={max_ms:.2f}")
        print(f"[LOAD] Throughput: {throughput:.2f} req/s")

    if errors:
        print("[LOAD] Failure breakdown:")
        for key, count in status_counter.items():
            if key == "200":
                continue
            print(f"  {key}: {count}")
        sample = failures[:5]
        for idx, failure in enumerate(sample, start=1):
            if failure.get("exception"):
                detail = failure["exception"]
            else:
                detail = failure.get("body") or failure.get("json_error", "Unknown error")
            print(f"    Sample #{idx}: {detail}")

    return 0 if success == total_requests else 4
def main(argv: t.Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    if args.audio:
        audio_bytes = _load_audio(args.audio)
    else:
        audio_bytes = _generate_sine_wave()

    chosen_uid = args.uid
    chosen_cid = args.cid
    available_refs: list[tuple[str, str]] | None = None
    selected_reference: tuple[str, str] | None = None
    if not chosen_uid or not chosen_cid:
        try:
            available_refs = _gather_cached_references(args.model, DEFAULT_CACHE_DIR)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] Unable to enumerate references from cache: {exc}", file=sys.stderr)
            return 4
        if args.uid:
            available_refs = [(uid, cid) for uid, cid in available_refs if uid == args.uid]
        if args.cid:
            available_refs = [(uid, cid) for uid, cid in available_refs if cid == args.cid]
        if not available_refs:
            print("[ERROR] No cached references match the specified filters.", file=sys.stderr)
            return 5
        selected_reference = random.choice(available_refs)
        if not chosen_uid:
            chosen_uid = selected_reference[0]
        if not chosen_cid:
            chosen_cid = selected_reference[1]

    total_requests = max(1, args.num_requests)
    concurrency = max(1, min(args.concurrency, total_requests))

    if total_requests == 1 and concurrency == 1:
        if selected_reference:
            print(f"[INFO] Using cached reference uid={chosen_uid}, cid={chosen_cid}")
        return _run_single(
            args,
            audio_bytes,
            chosen_uid,
            chosen_cid,
        )

    # Load test path
    if available_refs is None:
        # Either both uid/cid provided, or we haven't loaded references yet.
        try:
            available_refs = _gather_cached_references(args.model, DEFAULT_CACHE_DIR)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] Unable to enumerate references from cache: {exc}", file=sys.stderr)
            return 4
    if args.uid:
        available_refs = [(uid, clip) for uid, clip in available_refs if uid == args.uid]
    if args.cid:
        available_refs = [(uid, clip) for uid, clip in available_refs if clip == args.cid]
    if not available_refs:
        print("[ERROR] No cached references available for load testing after applying filters.", file=sys.stderr)
        return 5

    unique_refs = len(available_refs)
    if unique_refs > 1:
        print(f"[LOAD] Sampling uid/cid from {unique_refs} cached combinations.")
    else:
        print("[LOAD] Only one cached reference available; all requests will reuse the same uid/cid.")

    return _run_load_test(
        args,
        audio_bytes,
        chosen_uid,
        chosen_cid,
        total_requests,
        concurrency,
        available_refs,
    )


if __name__ == "__main__":
    raise SystemExit(main())
