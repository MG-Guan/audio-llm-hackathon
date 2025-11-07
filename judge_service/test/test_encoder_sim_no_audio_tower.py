import argparse
from pathlib import Path
from typing import Tuple

import librosa
import matplotlib.pyplot as plt
import torch
from transformers import AutoConfig, AutoProcessor

from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer

from boson_multimodal.model.higgs_audio import HiggsAudioModel


def load_waveform(audio_path: Path, target_sr: int) -> Tuple[torch.Tensor, int]:
    """Load audio as mono float32 at the target sample rate."""
    wav, sr = librosa.load(str(audio_path), sr=target_sr, mono=True)
    return torch.from_numpy(wav), sr


def main():
    parser = argparse.ArgumentParser(description="Run Higgs-Audio encoder and plot hidden-state distribution.")
    parser.add_argument(
        "--audio-path",
        type=Path,
        required=True,
        help="Path to a local WAV/FLAC audio file.",
    )
    parser.add_argument(
        "--model-id",
        default="bosonai/higgs-audio-v2-generation-3B-base",
        help="Hugging Face repo id or local path with model weights.",
    )
    parser.add_argument(
        "--processor-id",
        default="openai/whisper-large-v3-turbo",
        help="Whisper processor used to build input features.",
    )
    parser.add_argument(
        "--audio-tokenizer-id",
        default="bosonai/higgs-audio-v2-tokenizer",
        help="Fallback audio tokenizer repo id if the checkpoint skips the Whisper encoder.",
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=Path("encoder_hidden_hist.png"),
        help="Output path for the histogram figure.",
    )
    parser.add_argument(
        "--force-device",
        default=None,
        choices=["cpu", "cuda"],
        help="Optionally force running the encoder on CPU or CUDA. Defaults to CUDA when available.",
    )
    args = parser.parse_args()

    if not args.audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {args.audio_path}")

    device = torch.device(args.force_device if args.force_device else ("cuda" if torch.cuda.is_available() else "cpu"))

    torch.set_grad_enabled(False)

    print(f"Loading config from {args.model_id} ...")
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)

    flattened = None
    model = None

    if getattr(config, "skip_audio_tower", False):
        print(
            "Checkpoint was saved with `skip_audio_tower=true`; the Whisper encoder is absent. "
            "Falling back to the RVQ audio tokenizer encoder instead."
        )
        audio_tokenizer = load_higgs_audio_tokenizer(args.audio_tokenizer_id, device=str(device))
        target_sr = audio_tokenizer.sampling_rate

        print(f"Reading audio file {args.audio_path} ...")
        waveform, sr = load_waveform(args.audio_path, target_sr)
        print("Running audio tokenizer encoder (RVQ) ...")
        codes = audio_tokenizer.encode(waveform.numpy(), sr=target_sr)
        codes = codes.cpu()
        flattened = codes.flatten().float().numpy()

        print(
            f"Audio tokenizer code stats -- shape: {codes.shape}, mean: {flattened.mean():.4f}, "
            f"std: {flattened.std():.4f}, min: {flattened.min():.4f}, max: {flattened.max():.4f}"
        )

        audio_tokenizer.to("cpu")
        del audio_tokenizer
    else:
        print(f"Loading model weights from {args.model_id} ...")
        model = HiggsAudioModel.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map={"": "cpu"},
            trust_remote_code=True,
        )

        encoder = getattr(model, "audio_tower", None)
        if encoder is None:
            raise RuntimeError("Model config advertised an audio tower, but it was not instantiated.")

        print("Extracting Whisper encoder module ...")
        encoder = encoder.to(device)
        encoder.eval()

        print(f"Loading Whisper processor from {args.processor_id} ...")
        processor = AutoProcessor.from_pretrained(args.processor_id, trust_remote_code=True)
        target_sr = processor.feature_extractor.sampling_rate

        print(f"Reading audio file {args.audio_path} ...")
        waveform, sr = load_waveform(args.audio_path, target_sr)
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

        print("Running encoder forward pass ...")
        outputs = encoder(
            input_features,
            attention_mask=attention_mask,
            check_seq_length=False,
            return_dict=True,
        )

        hidden = outputs.last_hidden_state.squeeze(0).float().cpu()
        flattened = hidden.flatten().numpy()

        print(
            f"Encoder hidden stats -- shape: {hidden.shape}, mean: {flattened.mean():.4f}, "
            f"std: {flattened.std():.4f}, min: {flattened.min():.4f}, max: {flattened.max():.4f}"
        )

        encoder.to("cpu")
        del encoder

    print(f"Saving histogram to {args.output_figure} ...")
    plt.figure(figsize=(8, 4.5))
    plt.hist(flattened, bins=100, color="#1f77b4", alpha=0.85)
    plt.title("Higgs-Audio Encoder Hidden Distribution")
    plt.xlabel("Hidden Value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(args.output_figure)
    plt.close()

    if model is not None:
        model.to("cpu")
        del model
    torch.cuda.empty_cache()
    print("Done.")


if __name__ == "__main__":
    main()
