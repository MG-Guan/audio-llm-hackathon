#!/usr/bin/env python3
"""
Small helper around ffmpeg that re-muxes a .mp4 file into a .mov container.

Example:
    python convert_mp4_to_mov.py ~/video/input.mp4 --output ~/video/output.mov
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an .mp4 file to .mov using ffmpeg (streams copied by default)."
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the source .mp4 file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Destination .mov file. Defaults to the input stem with .mov extension.",
    )
    parser.add_argument(
        "--ffmpeg",
        type=str,
        default="ffmpeg",
        help="Path to the ffmpeg executable (defaults to 'ffmpeg' on PATH).",
    )
    parser.add_argument(
        "--video-codec",
        type=str,
        default="copy",
        help="Video codec for ffmpeg (default: copy to avoid re-encoding).",
    )
    parser.add_argument(
        "--audio-codec",
        type=str,
        default="copy",
        help="Audio codec for ffmpeg (default: copy to avoid re-encoding).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def run_ffmpeg(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed with exit code {exc.returncode}") from exc


def main() -> None:
    args = parse_args()
    input_path = args.input_file.expanduser().resolve()

    if not input_path.exists():
        print(f"Input file does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)
    if input_path.suffix.lower() != ".mp4":
        print(f"Expected an .mp4 file, got: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output
    if output_path is None:
        output_path = input_path.with_suffix(".mov")
    output_path = output_path.expanduser().resolve()

    if output_path.exists() and not args.overwrite:
        print(f"Output file already exists: {output_path}. Use --overwrite to replace it.", file=sys.stderr)
        sys.exit(1)

    cmd = [
        args.ffmpeg,
        "-y" if args.overwrite else "-n",
        "-i",
        str(input_path),
        "-c:v",
        args.video_codec,
        "-c:a",
        args.audio_codec,
        str(output_path),
    ]
    run_ffmpeg(cmd)
    print(f"Created {output_path}")


if __name__ == "__main__":
    main()
