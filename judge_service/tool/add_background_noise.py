#!/usr/bin/env python3
"""
Add background noise to a WAV file by mixing in a separate noise clip.

Example:
    python add_background_noise.py input.wav noise.wav output.wav --snr 15
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf
import torch
from torchaudio.functional import resample as ta_resample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mix a WAV file with background noise.")
    parser.add_argument("input_wav", type=Path, help="Path to the clean source WAV file.")
    parser.add_argument("noise_wav", type=Path, help="Path to the noise WAV file to mix in.")
    parser.add_argument("output_wav", type=Path, help="Destination path for the noisy WAV file.")
    parser.add_argument(
        "--snr",
        type=float,
        default=20.0,
        help="Desired signal-to-noise ratio in dB (larger values keep the speech cleaner).",
    )
    return parser.parse_args()


def load_audio(path: Path) -> Tuple[np.ndarray, int]:
    data, sr = sf.read(path)
    if data.ndim == 1:
        data = data[:, None]
    return data.astype(np.float32), sr


def match_channels(audio: np.ndarray, noise: np.ndarray) -> np.ndarray:
    target_channels = audio.shape[1]
    noise_channels = noise.shape[1]

    if noise_channels == target_channels:
        return noise
    if noise_channels == 1:
        return np.repeat(noise, target_channels, axis=1)
    if target_channels == 1:
        return noise.mean(axis=1, keepdims=True)
    raise ValueError(
        f"Cannot automatically match channel counts (audio={target_channels}, noise={noise_channels})."
    )


def extend_noise(noise: np.ndarray, target_length: int) -> np.ndarray:
    if noise.shape[0] >= target_length:
        return noise[:target_length]
    reps = int(np.ceil(target_length / noise.shape[0]))
    tiled = np.tile(noise, (reps, 1))
    return tiled[:target_length]


def scale_noise_for_snr(audio: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    signal_power = np.mean(audio**2)
    if signal_power <= 0:
        raise ValueError("Input audio appears to be silent; cannot compute SNR.")
    noise_power = np.mean(noise**2)
    if noise_power <= 0:
        raise ValueError("Noise clip appears to be silent; cannot mix.")

    desired_noise_power = signal_power / (10 ** (snr_db / 10.0))
    scale = np.sqrt(desired_noise_power / (noise_power + 1e-12))
    return noise * scale


def write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    audio = np.clip(audio, -1.0, 1.0)
    if audio.shape[1] == 1:
        audio = audio[:, 0]
    sf.write(path, audio, sample_rate)


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    tensor = torch.from_numpy(audio.T).float()  # [channels, time]
    resampled = ta_resample(tensor, orig_sr, target_sr)
    return resampled.T.numpy()


def main() -> None:
    args = parse_args()

    input_audio, input_sr = load_audio(args.input_wav)
    noise_audio, noise_sr = load_audio(args.noise_wav)
    noise_audio = resample_audio(noise_audio, noise_sr, input_sr)

    try:
        noise_audio = match_channels(input_audio, noise_audio)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    noise_audio = extend_noise(noise_audio, input_audio.shape[0])
    noise_audio = scale_noise_for_snr(input_audio, noise_audio, args.snr)

    mixed = input_audio + noise_audio
    args.output_wav.parent.mkdir(parents=True, exist_ok=True)
    write_audio(args.output_wav, mixed, input_sr)

    print(f"Wrote noisy audio to {args.output_wav}")


if __name__ == "__main__":
    main()
