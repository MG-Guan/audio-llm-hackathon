import argparse
from pathlib import Path
from typing import Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import WhisperFeatureExtractor, WhisperModel


def load_waveform(audio_path: Path, target_sr: int) -> Tuple[torch.Tensor, int]:
    wav, sr = librosa.load(str(audio_path), sr=target_sr, mono=True)
    return torch.from_numpy(wav), sr


def extract_hidden_states(
    model: WhisperModel,
    feature_extractor: WhisperFeatureExtractor,
    audio_path: Path,
    device: torch.device,
) -> torch.Tensor:
    waveform, _ = load_waveform(audio_path, feature_extractor.sampling_rate)
    inputs = feature_extractor(
        waveform.numpy(),
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="pt",
        padding="max_length",
    )
    input_features = inputs.input_features.to(device=device, dtype=model.dtype)
    outputs = model.encoder(input_features)
    hidden = outputs.last_hidden_state.squeeze(0).float().cpu()
    return hidden


def save_hidden_heatmap(matrix: np.ndarray, title: str, output_path: Path):
    plt.figure(figsize=(10, 4))
    plt.imshow(matrix, aspect="auto", origin="lower", cmap="magma")
    plt.colorbar(label="Activation value")
    plt.title(title)
    plt.xlabel("Hidden dimension")
    plt.ylabel("Time step")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compare two WAV files using OpenAI Whisper encoder embeddings."
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
        default="openai/whisper-small",
        help="Whisper checkpoint to use.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("whisper_hidden"),
        help="Prefix for heatmap image outputs (appends _A.png / _B.png).",
    )
    parser.add_argument(
        "--force-device",
        default=None,
        choices=["cpu", "cuda"],
        help="Optionally force CPU or CUDA. Defaults to CUDA if available.",
    )
    args = parser.parse_args()

    audio_a, audio_b = args.audio_paths
    for path in (audio_a, audio_b):
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

    device = torch.device(args.force_device if args.force_device else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    torch.set_grad_enabled(False)

    print(f"Loading Whisper model {args.model_id} ...")
    model = WhisperModel.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
    ).to(device)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_id)

    print(f"Encoding {audio_a} ...")
    hidden_a = extract_hidden_states(model, feature_extractor, audio_a, device)
    print(f"Encoding {audio_b} ...")
    hidden_b = extract_hidden_states(model, feature_extractor, audio_b, device)

    hidden_a_np = hidden_a.numpy()
    hidden_b_np = hidden_b.numpy()

    flat_a_t = torch.from_numpy(hidden_a_np.reshape(-1))
    flat_b_t = torch.from_numpy(hidden_b_np.reshape(-1))

    cosine_sim = torch.nn.functional.cosine_similarity(flat_a_t.unsqueeze(0), flat_b_t.unsqueeze(0)).item()
    l2_distance = torch.norm(flat_a_t - flat_b_t, p=2).item()
    mae = torch.mean(torch.abs(flat_a_t - flat_b_t)).item()

    print(
        f"Cosine similarity: {cosine_sim:.6f}\n"
        f"L2 distance: {l2_distance:.6f}\n"
        f"Mean absolute error: {mae:.6f}"
    )

    output_a = args.output_prefix.with_name(f"{args.output_prefix.stem}_A.png")
    output_b = args.output_prefix.with_name(f"{args.output_prefix.stem}_B.png")
    print(f"Saving encoder heatmap for {audio_a.name} to {output_a} ...")
    save_hidden_heatmap(
        hidden_a_np,
        f"Whisper Encoder Activations - {audio_a.name}",
        output_a,
    )
    print(f"Saving encoder heatmap for {audio_b.name} to {output_b} ...")
    save_hidden_heatmap(
        hidden_b_np,
        f"Whisper Encoder Activations - {audio_b.name}",
        output_b,
    )

    model.to("cpu")
    del model
    torch.cuda.empty_cache()
    print("Done.")


if __name__ == "__main__":
    main()
