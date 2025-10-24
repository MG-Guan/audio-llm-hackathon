#!/usr/bin/env python3
"""
Extract WAV audio tracks from .mov files that share a given prefix.

Example:
    python extract_wav_batch.py ./videos --prefix session_ --output ./audio
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract WAV audio from matching .mov files within a directory."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing .mov files.",
    )
    parser.add_argument(
        "--prefix",
        required=True,
        help="Filename prefix to match (case-sensitive, applied before the extension).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        required=True,
        help="Directory to write extracted .wav files.",
    )
    parser.add_argument(
        "--ffmpeg",
        type=str,
        default="ffmpeg",
        help="Path to ffmpeg executable (defaults to 'ffmpeg' on PATH).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output files.",
    )
    return parser.parse_args()


def find_mov_files(input_dir: Path, prefix: str) -> Iterable[Path]:
    for candidate in sorted(input_dir.iterdir()):
        if not candidate.is_file():
            continue
        if not candidate.name.startswith(prefix):
            continue
        if candidate.suffix.lower() != ".mov":
            continue
        yield candidate.resolve()


def run_ffmpeg(ffmpeg_bin: str, command: Sequence[str]) -> None:
    cmd = [ffmpeg_bin, *command]
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        message = (
            f"ffmpeg command failed: {' '.join(cmd)}\n"
            f"stdout:\n{exc.stdout}\n"
            f"stderr:\n{exc.stderr}"
        )
        raise RuntimeError(message) from exc


def extract_audio(
    ffmpeg_bin: str,
    sources: Iterable[Path],
    output_dir: Path,
    overwrite: bool,
) -> None:
    overwrite_flag = "-y" if overwrite else "-n"

    for source in sources:
        output_name = f"{source.stem}.wav"
        destination = output_dir / output_name

        command = [
            overwrite_flag,
            "-i",
            str(source),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "48000",
            "-ac",
            "2",
            str(destination),
        ]

        print(f"Extracting audio from {source} -> {destination}")
        run_ffmpeg(ffmpeg_bin, command)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory not found or not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    sources = list(find_mov_files(input_dir, args.prefix))
    if not sources:
        print("No matching .mov files found.", file=sys.stderr)
        sys.exit(1)

    try:
        extract_audio(args.ffmpeg, sources, output_dir, args.overwrite)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    print("Done.")


if __name__ == "__main__":
    main()
