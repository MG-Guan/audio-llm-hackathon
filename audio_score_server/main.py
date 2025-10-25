#!/usr/bin/env python3
"""Audio scoring service that compares user recordings against reference clips."""
from __future__ import annotations

import asyncio
import io
import json
import logging
import shutil
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydub import AudioSegment
from transformers import AutoFeatureExtractor, AutoModel, AutoProcessor, WhisperForConditionalGeneration, WhisperProcessor
from torchaudio.functional import resample as ta_resample

try:
    from boson_multimodal.model.higgs_audio import HiggsAudioModel
except ImportError:  # pragma: no cover - optional dependency during development
    HiggsAudioModel = None


LOGGER = logging.getLogger("audio_score_server")


@dataclass
class ClipReference:
    model_key: str
    video_id: str
    clip_id: str
    transcript: str
    transcript_path: Path
    hidden_path: Path


@dataclass
class ModelSpec:
    key: str
    type: str
    model_id: str
    processor_id: str | None = None
    transcriber_id: str | None = None
    sampling_rate: int | None = None
    style_model_id: str | None = None


class BaseModelRunner:
    """Common interface for model-specific inference logic."""

    def __init__(self, spec: ModelSpec, device: torch.device):
        self.spec = spec
        self.device = device
        self.key = spec.key

    @staticmethod
    def _resample_if_needed(waveform: np.ndarray, sampling_rate: int, target_sr: int | None) -> Tuple[np.ndarray, int]:
        if target_sr is None or sampling_rate == target_sr:
            return waveform, sampling_rate
        tensor = torch.from_numpy(waveform).float()
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        resampled = ta_resample(tensor, sampling_rate, target_sr)
        return resampled.squeeze(0).cpu().numpy().astype(np.float32), target_sr

    def transcribe_and_embed(self, waveform: np.ndarray, sampling_rate: int) -> Tuple[str, torch.Tensor]:
        raise NotImplementedError

    @staticmethod
    def _pool_hidden_state(hidden_state: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        """Average only valid frames (per attention mask) and L2-normalize."""
        if hidden_state.dim() != 2:
            raise ValueError("hidden_state must be 2D [seq, hidden_dim].")

        pooled_source = hidden_state
        if attention_mask is not None:
            mask = attention_mask
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            mask = mask.to(hidden_state.device, dtype=torch.float32)
            seq_len = hidden_state.size(0)
            if mask.size(-1) != seq_len:
                mask = F.interpolate(mask.unsqueeze(1), size=seq_len, mode="nearest").squeeze(1)
            mask = mask.squeeze(0) > 0.5
            if mask.any():
                pooled_source = hidden_state[mask]

        pooled = pooled_source.mean(dim=0)
        return F.normalize(pooled, p=2, dim=0)


class WhisperModelRunner(BaseModelRunner):
    """Run inference using a Whisper checkpoint."""

    def __init__(self, spec: ModelSpec, device: torch.device):
        super().__init__(spec, device)
        self.model_id = spec.model_id
        self.processor = WhisperProcessor.from_pretrained(self.model_id)
        model_kwargs: Dict[str, Any] = {}
        if device.type == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_id, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()
        self.feature_rate = self.processor.feature_extractor.sampling_rate
        self.model_dtype = self.model.model.encoder.conv1.weight.dtype

    def transcribe_and_embed(self, waveform: np.ndarray, sampling_rate: int) -> Tuple[str, torch.Tensor]:
        audio, audio_sr = self._resample_if_needed(waveform, sampling_rate, self.feature_rate)

        features = self.processor(
            audio,
            sampling_rate=audio_sr,
            return_tensors="pt",
        )
        attention_mask = getattr(features, "attention_mask", None)
        input_features = features.input_features.to(self.device, dtype=self.model_dtype)

        with torch.no_grad():
            encoder_outputs = self.model.model.encoder(input_features=input_features)
            pooled_hidden = self._pool_hidden_state(
                encoder_outputs.last_hidden_state.squeeze(0),
                attention_mask,
            )
            hidden_state = pooled_hidden.float().cpu()
            generated_ids = self.model.generate(input_features=input_features)

        transcript = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return transcript, hidden_state


class WhisperStyleModelRunner(BaseModelRunner):
    """Whisper for ASR paired with a Hugging Face speaker/prosody embedding model."""

    DEFAULT_STYLE_MODEL = "microsoft/wavlm-base-plus-sv"

    def __init__(self, spec: ModelSpec, device: torch.device):
        super().__init__(spec, device)
        self.model_id = spec.model_id
        self.processor = WhisperProcessor.from_pretrained(self.model_id)
        model_kwargs: Dict[str, Any] = {}
        if device.type == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_id, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()
        self.feature_rate = self.processor.feature_extractor.sampling_rate
        self.model_dtype = self.model.model.encoder.conv1.weight.dtype

        requested_style_model = spec.style_model_id or self.DEFAULT_STYLE_MODEL
        self.style_sample_rate = spec.sampling_rate or self.feature_rate
        try:
            self.style_processor = self._load_hf_processor(requested_style_model)
            sr = getattr(self.style_processor, "sampling_rate", None)
            if sr is None:
                sr = getattr(getattr(self.style_processor, "feature_extractor", None), "sampling_rate", None)
            if sr is not None:
                self.style_sample_rate = int(sr)
            self.style_model = AutoModel.from_pretrained(requested_style_model)
            self.style_model.to(self.device)
            self.style_model.eval()
            self.style_dtype = next(self.style_model.parameters()).dtype
            self.style_model_id = requested_style_model
        except Exception as exc:  # pragma: no cover - requires HF download
            raise RuntimeError(
                f"Failed to load Hugging Face style model '{requested_style_model}': {exc}"
            ) from exc

    def transcribe_and_embed(self, waveform: np.ndarray, sampling_rate: int) -> Tuple[str, torch.Tensor]:
        audio, audio_sr = self._resample_if_needed(waveform, sampling_rate, self.feature_rate)

        features = self.processor(
            audio,
            sampling_rate=audio_sr,
            return_tensors="pt",
        )
        input_features = features.input_features.to(self.device, dtype=self.model_dtype)

        with torch.no_grad():
            generated_ids = self.model.generate(input_features=input_features)

        transcript = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        style_audio, _ = self._resample_if_needed(waveform, sampling_rate, self.style_sample_rate)
        hidden_state = self._encode_style_with_hf(style_audio)
        return transcript, hidden_state

    @staticmethod
    def _load_hf_processor(model_id: str):
        try:
            return AutoProcessor.from_pretrained(model_id)
        except Exception:
            return AutoFeatureExtractor.from_pretrained(model_id)

    def _encode_style_with_hf(self, audio: np.ndarray) -> torch.Tensor:
        if self.style_processor is None or self.style_model is None:
            raise RuntimeError("Hugging Face style model not initialized.")
        inputs = self.style_processor(
            audio,
            sampling_rate=self.style_sample_rate,
            return_tensors="pt",
        )
        input_values = inputs.get("input_values")
        if input_values is None:
            raise RuntimeError("Style processor did not return 'input_values'.")
        input_values = input_values.to(self.device, dtype=self.style_dtype)
        with torch.no_grad():
            outputs = self.style_model(input_values)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            style_vec = outputs.pooler_output
        else:
            style_vec = outputs.last_hidden_state.mean(dim=1)
        return F.normalize(style_vec.squeeze(0), p=2, dim=-1).float().cpu()


class HiggsAudioRunner(BaseModelRunner):
    """Run inference using the Higgs-Audio audio tower with a configurable transcript model."""

    def __init__(self, spec: ModelSpec, device: torch.device):
        if HiggsAudioModel is None:
            raise RuntimeError("boson_multimodal package is required for Higgs-Audio models.")
        if spec.processor_id is None:
            raise ValueError("processor_id must be provided for Higgs-Audio models.")
        if spec.transcriber_id is None:
            raise ValueError("transcriber_id must be provided for Higgs-Audio models.")

        super().__init__(spec, device)

        processor_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        self.processor = AutoProcessor.from_pretrained(spec.processor_id, **processor_kwargs)
        self.processor_rate = self.processor.feature_extractor.sampling_rate

        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": {"": "cpu"},
        }
        if device.type == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float32
        base_model = HiggsAudioModel.from_pretrained(spec.model_id, **model_kwargs)

        audio_tower = getattr(base_model, "audio_tower", None)
        if audio_tower is None:
            raise RuntimeError("Configured Higgs-Audio checkpoint does not contain an audio tower.")

        self.audio_tower = audio_tower.to(self.device)
        self.audio_tower.eval()
        self.audio_dtype = self.audio_tower.conv1.weight.dtype

        transcriber_kwargs: Dict[str, Any] = {}
        if device.type == "cuda":
            transcriber_kwargs["torch_dtype"] = torch.float16
        self.transcriber_processor = WhisperProcessor.from_pretrained(spec.transcriber_id)
        self.transcriber_rate = self.transcriber_processor.feature_extractor.sampling_rate
        self.transcriber_model = WhisperForConditionalGeneration.from_pretrained(
            spec.transcriber_id,
            **transcriber_kwargs,
        )
        self.transcriber_model.to(self.device)
        self.transcriber_model.eval()
        self.transcriber_dtype = self.transcriber_model.model.encoder.conv1.weight.dtype

    def transcribe_and_embed(self, waveform: np.ndarray, sampling_rate: int) -> Tuple[str, torch.Tensor]:
        transcript_audio, transcript_sr = self._resample_if_needed(waveform, sampling_rate, self.transcriber_rate)
        transcript_features = self.transcriber_processor(
            transcript_audio,
            sampling_rate=transcript_sr,
            return_tensors="pt",
        )
        transcript_inputs = transcript_features.input_features.to(self.device, dtype=self.transcriber_dtype)

        with torch.no_grad():
            generated_ids = self.transcriber_model.generate(input_features=transcript_inputs)
        transcript = self.transcriber_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        embed_audio, embed_sr = self._resample_if_needed(waveform, sampling_rate, self.processor_rate)
        embed_features = self.processor(
            embed_audio,
            sampling_rate=embed_sr,
            return_tensors="pt",
            padding="max_length",
        )
        input_features = embed_features.input_features.to(self.device, dtype=self.audio_dtype)
        attention_mask = getattr(embed_features, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.audio_tower(
                input_features,
                attention_mask=attention_mask,
                check_seq_length=False,
                return_dict=True,
            )
        pooled_hidden = self._pool_hidden_state(
            outputs.last_hidden_state.squeeze(0),
            attention_mask,
        )
        hidden_state = pooled_hidden.float().cpu()
        return transcript, hidden_state


class AudioScoreService:
    """Encapsulates model loading, caching, and scoring logic."""

    def __init__(self) -> None:
        self.app_root = Path(__file__).resolve().parent
        self.config_path = self.app_root / "config.json"
        self.config: Dict[str, Any] = {}

        self.video_clip_root: Path | None = None
        self.cache_root: Path | None = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_sr: int = 16000

        self.model_specs: Dict[str, ModelSpec] = {}
        self.runners: Dict[str, BaseModelRunner] = {}
        self.references: Dict[str, Dict[Tuple[str, str], ClipReference]] = {}
        self.default_model_key: str | None = None

        self._startup_lock = asyncio.Lock()
        self._ready_event = asyncio.Event()

    async def startup(self) -> None:
        """Load configuration, initialize the model, and prepare cached references."""
        if self._ready_event.is_set():
            return

        async with self._startup_lock:
            if self._ready_event.is_set():
                return

            await asyncio.to_thread(self._initialize_sync)
            self._ready_event.set()

    async def ensure_ready(self) -> None:
        """Wait until startup initialization completes."""
        await self._ready_event.wait()

    def _initialize_sync(self) -> None:
        self.config = self._load_config()
        self.target_sr = int(self.config.get("target_sampling_rate", 16000))

        self.video_clip_root = (self.app_root / self.config["video_clip_dir"]).resolve()

        cache_root_path = (self.app_root / self.config["cache_dir"]).resolve()
        self._flush_cache_root(cache_root_path)
        cache_root_path.mkdir(parents=True, exist_ok=True)
        self.cache_root = cache_root_path

        if not self.video_clip_root.exists():
            raise RuntimeError(f"Video clip directory not found: {self.video_clip_root}")

        self.model_specs = self._parse_model_specs(self.config)
        if not self.model_specs:
            raise RuntimeError("No models specified in the configuration.")
        self.default_model_key = None

        LOGGER.info("Loading models on device %s", self.device)
        for key, spec in self.model_specs.items():
            try:
                runner = self._create_runner(spec)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Failed to load model '%s' (%s): %s", key, spec.model_id, exc)
                continue
            self.runners[key] = runner
            self.references[key] = {}
            if self.default_model_key is None:
                self.default_model_key = key
            LOGGER.info("Loaded model '%s' (%s)", key, spec.model_id)

        if not self.runners:
            raise RuntimeError("No models could be initialized successfully.")

        clip_paths = list(self._iter_clip_files())
        LOGGER.info("Preparing reference cache from %s", self.video_clip_root)
        for clip_path in clip_paths:
            video_id, clip_id = self._parse_clip_ids(clip_path)
            for key, runner in self.runners.items():
                try:
                    reference = self._prepare_clip_reference(key, runner, video_id, clip_id, clip_path)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception(
                        "Failed to prepare reference for video=%s clip=%s using model '%s': %s",
                        video_id,
                        clip_id,
                        key,
                        exc,
                    )
                    continue
                self.references[key][(video_id, clip_id)] = reference
                LOGGER.debug("Cached reference for %s / %s using model %s", video_id, clip_id, key)

        LOGGER.info("Reference preparation complete. Total clips processed: %d", len(clip_paths))

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise RuntimeError(f"Missing configuration file: {self.config_path}")
        with self.config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)

        required_keys = {"video_clip_dir", "cache_dir"}
        missing = required_keys - config.keys()
        if missing:
            raise RuntimeError(f"Configuration missing keys: {', '.join(sorted(missing))}")
        return config

    def _flush_cache_root(self, cache_path: Path) -> None:
        """Remove cache directory before rebuilding cached references."""
        if cache_path.exists():
            try:
                cache_path.relative_to(self.app_root)
            except ValueError as exc:
                raise RuntimeError(
                    f"Refusing to delete cache directory outside application root: {cache_path}"
                ) from exc
            LOGGER.info("Clearing cache directory: %s", cache_path)
            shutil.rmtree(cache_path)

    def _parse_model_specs(self, config: Dict[str, Any]) -> Dict[str, ModelSpec]:
        model_entries = config.get("models")
        if not model_entries:
            legacy_model = config.get("model_name")
            if not legacy_model:
                return {}
            model_entries = [
                {
                    "key": "default",
                    "type": "whisper",
                    "model_id": legacy_model,
                }
            ]

        specs: Dict[str, ModelSpec] = {}
        for entry in model_entries:
            if "model_id" not in entry:
                raise RuntimeError("Model entry must include 'model_id'.")
            key = (entry.get("key") or entry.get("name") or entry["model_id"]).strip()
            spec = ModelSpec(
                key=key,
                type=entry.get("type", "whisper"),
                model_id=entry["model_id"],
                processor_id=entry.get("processor_id"),
                transcriber_id=entry.get("transcriber_id"),
                sampling_rate=entry.get("sampling_rate"),
                style_model_id=entry.get("style_model_id"),
            )
            specs[spec.key] = spec
        return specs

    def _create_runner(self, spec: ModelSpec) -> BaseModelRunner:
        if spec.type.startswith("whisper-style"):
            return WhisperStyleModelRunner(spec, self.device)
        if spec.type.startswith("whisper"):
            return WhisperModelRunner(spec, self.device)
        if spec.type.startswith("higgs-audio"):
            return HiggsAudioRunner(spec, self.device)
        raise RuntimeError(f"Unsupported model type '{spec.type}' for model '{spec.key}'.")

    def _iter_clip_files(self) -> Iterable[Path]:
        if self.video_clip_root is None:
            raise RuntimeError("Video clip root not initialized.")
        return sorted(self.video_clip_root.glob("*.mov"))

    def _parse_clip_ids(self, clip_path: Path) -> Tuple[str, str]:
        stem = clip_path.stem
        if "_" not in stem:
            raise RuntimeError(f"Clip filename '{clip_path.name}' does not contain an underscore.")
        video_id, clip_id = stem.rsplit("_", 1)
        return video_id, clip_id

    def _prepare_clip_reference(
        self,
        model_key: str,
        runner: BaseModelRunner,
        video_id: str,
        clip_id: str,
        clip_path: Path,
    ) -> ClipReference:
        if self.cache_root is None:
            raise RuntimeError("Cache root not initialized.")

        clip_cache_dir = self.cache_root / model_key / video_id / clip_id
        clip_cache_dir.mkdir(parents=True, exist_ok=True)

        transcript_path = clip_cache_dir / "transcript.txt"
        hidden_path = clip_cache_dir / "hidden.pt"

        if transcript_path.exists() and hidden_path.exists():
            transcript = transcript_path.read_text(encoding="utf-8").strip()
            if transcript:
                return ClipReference(model_key, video_id, clip_id, transcript, transcript_path, hidden_path)

        waveform, sampling_rate = self._load_audio_file(clip_path)
        transcript, hidden = runner.transcribe_and_embed(waveform, sampling_rate)

        transcript_path.write_text(transcript, encoding="utf-8")
        torch.save(hidden, hidden_path)

        return ClipReference(model_key, video_id, clip_id, transcript, transcript_path, hidden_path)

    def _load_audio_file(self, path: Path) -> Tuple[np.ndarray, int]:
        with path.open("rb") as stream:
            data = stream.read()
        return self._decode_audio_bytes(data)

    def _decode_audio_bytes(self, data: bytes) -> Tuple[np.ndarray, int]:
        if not data:
            raise RuntimeError("Audio data is empty.")

        audio = AudioSegment.from_file(io.BytesIO(data))
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

        if audio.channels > 1:
            samples = samples.reshape((-1, audio.channels)).mean(axis=1)

        max_int_value = float(2 ** (8 * audio.sample_width - 1))
        if max_int_value <= 0:
            raise RuntimeError("Invalid audio sample width.")
        samples /= max_int_value

        sampling_rate = audio.frame_rate
        if sampling_rate != self.target_sr:
            tensor = torch.from_numpy(samples).float().unsqueeze(0)
            resampled = ta_resample(tensor, sampling_rate, self.target_sr)
            samples = resampled.squeeze(0).cpu().numpy()
            sampling_rate = self.target_sr

        return samples.astype(np.float32), sampling_rate

    async def process_uploaded_audio(self, file: UploadFile) -> Tuple[np.ndarray, int]:
        payload = await file.read()
        if not payload:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")
        try:
            waveform, sampling_rate = self._decode_audio_bytes(payload)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Failed to decode audio: {exc}") from exc
        finally:
            await file.close()
        return waveform, sampling_rate

    def score(
        self,
        model_key: str,
        video_id: str,
        clip_id: str,
        query_transcript: str,
        query_hidden: torch.Tensor,
    ) -> Dict[str, float]:
        reference_map = self.references.get(model_key)
        if reference_map is None:
            raise HTTPException(status_code=404, detail=f"Unknown model '{model_key}'.")

        reference = reference_map.get((video_id, clip_id))
        if reference is None:
            raise HTTPException(status_code=404, detail="Reference clip not found.")

        try:
            reference_hidden = torch.load(reference.hidden_path, map_location="cpu")
        except FileNotFoundError as exc:
            raise HTTPException(status_code=500, detail="Reference hidden state missing.") from exc

        audio_sim = self._audio_similarity(reference_hidden, query_hidden)
        text_sim = self._text_similarity(reference.transcript, query_transcript)
        sim_score = float((audio_sim + text_sim) / 2.0)

        return {
            "sim_score": sim_score,
            "audio_sim_score": audio_sim,
            "text_sim_score": text_sim,
        }

    def _audio_similarity(self, reference_hidden: torch.Tensor, query_hidden: torch.Tensor) -> float:
        ref_vec = self._prepare_audio_vector(reference_hidden)
        qry_vec = self._prepare_audio_vector(query_hidden)

        if ref_vec.shape[0] != qry_vec.shape[0]:
            ref_vec, qry_vec = self._match_vector_dimensions(ref_vec, qry_vec)

        ref_norm = ref_vec / (ref_vec.norm(p=2) + 1e-8)
        qry_norm = qry_vec / (qry_vec.norm(p=2) + 1e-8)

        cosine = torch.dot(ref_norm, qry_norm).item()
        score = (cosine + 1.0) / 2.0
        return float(max(0.0, min(1.0, score)))

    def _text_similarity(self, reference_transcript: str, query_transcript: str) -> float:
        reference_clean = reference_transcript.lower().strip()
        query_clean = query_transcript.lower().strip()
        matcher = SequenceMatcher(None, reference_clean, query_clean)
        return float(matcher.ratio())

    @staticmethod
    def _prepare_audio_vector(hidden: torch.Tensor) -> torch.Tensor:
        if hidden.dim() == 1:
            return hidden.float()
        if hidden.dim() == 2:
            return hidden.mean(dim=0).float()
        raise ValueError("hidden tensor must be either 1D or 2D.")

    @staticmethod
    def _match_vector_dimensions(ref_vec: torch.Tensor, qry_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        target_dim = max(ref_vec.shape[0], qry_vec.shape[0])

        def _resize(vec: torch.Tensor, target: int) -> torch.Tensor:
            if vec.shape[0] == target:
                return vec
            source = vec.unsqueeze(0).unsqueeze(0)  # shape [1, 1, dim]
            mode = "linear" if vec.shape[0] > 1 else "nearest"
            resized = F.interpolate(source, size=target, mode=mode, align_corners=False)
            return resized.squeeze(0).squeeze(0)

        return _resize(ref_vec, target_dim), _resize(qry_vec, target_dim)


service = AudioScoreService()
app = FastAPI()


@app.on_event("startup")
async def handle_startup() -> None:
    await service.startup()


@app.post("/score")
async def score_endpoint(
    uid: str = Form(..., description="Unique video identifier."),
    cid: str = Form(..., description="Clip identifier within the video."),
    audio: UploadFile = File(..., description="User recorded audio file to score."),
    model: str | None = Form(
        None,
        description="Optional model key to use for scoring. Defaults to the first configured model.",
    ),
) -> JSONResponse:
    await service.ensure_ready()

    video_id = uid.strip()
    clip_id = cid.strip()
    if not video_id or not clip_id:
        raise HTTPException(status_code=400, detail="uid and cid must be non-empty.")

    model_key = model.strip() if model else service.default_model_key
    if not model_key:
        raise HTTPException(status_code=500, detail="No models are available for scoring.")

    runner = service.runners.get(model_key)
    if runner is None:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model_key}'.")

    waveform, sampling_rate = await service.process_uploaded_audio(audio)
    query_transcript, query_hidden = await asyncio.to_thread(
        runner.transcribe_and_embed,
        waveform,
        sampling_rate,
    )

    result = service.score(model_key, video_id, clip_id, query_transcript, query_hidden)
    payload = {"model": model_key, **result}
    return JSONResponse(payload)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=9000,
        reload=False,
    )
