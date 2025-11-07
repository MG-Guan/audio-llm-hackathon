import argparse
from pathlib import Path
from typing import Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoConfig, AutoProcessor

from boson_multimodal.model.higgs_audio import HiggsAudioModel


def load_waveform(audio_path: Path, target_sr: int) -> Tuple[torch.Tensor, int]:
    """Load audio as mono float32 at the target sample rate."""
    wav, sr = librosa.load(str(audio_path), sr=target_sr, mono=True)
    return torch.from_numpy(wav), sr


def encode_audio(
    encoder: torch.nn.Module,
    processor: AutoProcessor,
    audio_path: Path,
    device: torch.device,
    check_seq_length: bool = False,
) -> torch.Tensor:
    """Run Whisper encoder on an audio file and return last_hidden_state [seq, hidden]."""
    target_sr = processor.feature_extractor.sampling_rate
    waveform, sr = load_waveform(audio_path, target_sr)
    inputs = processor(
        waveform.numpy(),
        sampling_rate=target_sr,
        return_tensors="pt",
        padding="max_length",
    )
    input_features = inputs.input_features.to(device=device, dtype=encoder.conv1.weight.dtype)
    attention_mask = getattr(inputs, "attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    outputs = encoder(
        input_features,
        attention_mask=attention_mask,
        check_seq_length=check_seq_length,
        return_dict=True,
    )
    hidden = outputs.last_hidden_state.squeeze(0).detach().cpu()
    return hidden


def plot_histogram(
    features_a: np.ndarray,
    features_b: np.ndarray,
    labels: Tuple[str, str],
    output_path: Path,
):
    plt.figure(figsize=(9, 5))
    plt.hist(features_a, bins=120, alpha=0.6, label=labels[0], color="#1f77b4")
    plt.hist(features_b, bins=120, alpha=0.6, label=labels[1], color="#d62728")
    plt.title("Higgs-Audio Tower Hidden Distribution")
    plt.xlabel("Hidden Value")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare encoder outputs from the Higgs-Audio Whisper tower for two WAV files, "
            "plot histograms, and compute similarity metrics."
        )
    )
    parser.add_argument(
        "--audio-paths",
        type=Path,
        nargs=2,
        required=True,
        metavar=("AUDIO_A", "AUDIO_B"),
        help="Two audio files to compare.",
    )
    parser.add_argument(
        "--model-id",
        default="bosonai/higgs-audio-v2-understanding-3B",
        help="Checkpoint that includes the audio tower (skip_audio_tower must be False).",
    )
    parser.add_argument(
        "--processor-id",
        default="openai/whisper-large-v3-turbo",
        help="Whisper processor repo id.",
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=Path("audio_tower_hidden_hist.png"),
        help="Path of the histogram figure to save.",
    )
    parser.add_argument(
        "--force-device",
        default=None,
        choices=["cpu", "cuda"],
        help="Optionally force CPU or CUDA. Defaults to CUDA when available.",
    )
    args = parser.parse_args()

    audio_a, audio_b = args.audio_paths
    for audio_path in (audio_a, audio_b):
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

    device = torch.device(args.force_device if args.force_device else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_grad_enabled(False)

    print(f"Loading config from {args.model_id} ...")
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    if getattr(config, "skip_audio_tower", False):
        raise RuntimeError(
            f"Checkpoint {args.model_id} was saved with skip_audio_tower=True; "
            "choose a checkpoint that retains the Whisper encoder."
        )

    print(f"Loading model weights from {args.model_id} ...")
    model = HiggsAudioModel.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map={"": "cpu"},
        trust_remote_code=True,
    )
    encoder = model.audio_tower
    if encoder is None:
        raise RuntimeError("Model did not expose audio_tower module.")

    encoder = encoder.to(device)
    encoder.eval()

    print(f"Loading Whisper processor from {args.processor_id} ...")
    processor = AutoProcessor.from_pretrained(args.processor_id, trust_remote_code=True)

    print(f"Encoding {audio_a} ...")
    hidden_a = encode_audio(encoder, processor, audio_a, device, check_seq_length=False)
    print(f"Encoding {audio_b} ...")
    hidden_b = encode_audio(encoder, processor, audio_b, device, check_seq_length=False)

    flat_a = hidden_a.flatten().float()
    flat_b = hidden_b.flatten().float()

    cosine_sim = torch.nn.functional.cosine_similarity(flat_a.unsqueeze(0), flat_b.unsqueeze(0)).item()
    l2_distance = torch.norm(flat_a - flat_b, p=2).item()
    mae = torch.mean(torch.abs(flat_a - flat_b)).item()

    print(
        f"Cosine similarity: {cosine_sim:.6f}\n"
        f"L2 distance: {l2_distance:.6f}\n"
        f"Mean absolute error: {mae:.6f}"
    )

    print(f"Saving histogram to {args.output_figure} ...")
    plot_histogram(
        flat_a.numpy(),
        flat_b.numpy(),
        (audio_a.name, audio_b.name),
        args.output_figure,
    )

    model.to("cpu")
    del encoder
    del model
    torch.cuda.empty_cache()
    print("Done.")


if __name__ == "__main__":
    main()
