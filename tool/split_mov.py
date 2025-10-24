#!/usr/bin/env python3
"""
Utility to split a .mov file into multiple clips using timestamp cut points.

Example:
    python split_mov.py ~/video/source.mov --timestamps 1.2 2.5 7.0
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a .mov file into several clips using timestamp cut points."
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the source .mov file.",
    )
    parser.add_argument(
        "--timestamps",
        "-t",
        type=float,
        nargs="+",
        required=True,
        help="Cut points in seconds (e.g. --timestamps 1.2 2.5 7.0).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Directory to write clips to (defaults to the input file directory).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Optional prefix for generated clip filenames.",
    )
    parser.add_argument(
        "--ffmpeg",
        type=str,
        default="ffmpeg",
        help="Path to the ffmpeg executable (defaults to 'ffmpeg' on PATH).",
    )
    parser.add_argument(
        "--ffprobe",
        type=str,
        default="ffprobe",
        help="Path to the ffprobe executable (defaults to 'ffprobe' on PATH).",
    )
    return parser.parse_args()


def run_command(cmd: Sequence[str]) -> subprocess.CompletedProcess:
    """Run a command and raise a descriptive error if it fails."""
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        message = (
            f"Command failed: {' '.join(cmd)}\n"
            f"stdout:\n{exc.stdout}\n"
            f"stderr:\n{exc.stderr}"
        )
        raise RuntimeError(message) from exc
    return result


def probe_duration(ffprobe_bin: str, input_file: Path) -> float:
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(input_file),
    ]
    result = run_command(cmd)
    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"Unable to parse duration from ffprobe output: {result.stdout}") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(f"Invalid media duration reported by ffprobe: {duration}")
    return duration


def build_segments(timestamps: Sequence[float], duration: float) -> List[Tuple[float, float]]:
    segments: List[Tuple[float, float]] = []
    previous = 0.0

    for raw_ts in timestamps:
        ts = float(raw_ts)
        if ts <= 0.0:
            print(f"Skipping timestamp {ts} (must be > 0).", file=sys.stderr)
            continue
        if ts >= duration:
            print(f"Skipping timestamp {ts} (exceeds duration {duration:.3f}).", file=sys.stderr)
            continue
        if ts <= previous or math.isclose(ts, previous, rel_tol=0.0, abs_tol=1e-6):
            print(f"Skipping timestamp {ts} (not strictly greater than previous cut {previous}).", file=sys.stderr)
            continue
        segments.append((previous, ts))
        previous = ts

    if duration > previous:
        segments.append((previous, duration))

    return segments


def format_time(seconds: float) -> str:
    return f"{seconds:.6f}".rstrip("0").rstrip(".")


def split_mov(
    ffmpeg_bin: str,
    input_file: Path,
    output_dir: Path,
    prefix: str,
    segments: Sequence[Tuple[float, float]],
) -> None:
    for index, (start, end) in enumerate(segments, start=1):
        clip_duration = max(0.0, end - start)
        if clip_duration <= 0.0:
            continue

        output_name = f"{prefix}_part{index:02d}.mov"
        output_path = output_dir / output_name
        cmd = [
            ffmpeg_bin,
            "-y",
            "-ss",
            format_time(start),
            "-i",
            str(input_file),
            "-t",
            format_time(clip_duration),
            "-c",
            "copy",
            str(output_path),
        ]
        print(f"Creating clip {index}: {format_time(start)}s -> {format_time(end)}s ({output_path})")
        run_command(cmd)


def main() -> None:
    args = parse_args()
    input_file = args.input_file.resolve()

    if not input_file.exists():
        print(f"Input file not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    output_dir = (args.output_dir or input_file.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    duration = probe_duration(args.ffprobe, input_file)
    segments = build_segments(args.timestamps, duration)

    if not segments:
        print("No valid segments to create. Check your timestamps.", file=sys.stderr)
        sys.exit(1)

    prefix = args.prefix or input_file.stem
    split_mov(args.ffmpeg, input_file, output_dir, prefix, segments)
    print("Done.")


if __name__ == "__main__":
    main()
