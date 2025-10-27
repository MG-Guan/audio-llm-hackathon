#!/usr/bin/env python3
"""Audio scoring service that compares user recordings against reference clips."""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import re
import shutil
import wave
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydub import AudioSegment
from transformers import AutoFeatureExtractor, AutoModel, AutoProcessor, WhisperForConditionalGeneration, WhisperProcessor
from torchaudio.functional import resample as ta_resample

try:
    import openai
except ImportError:  # pragma: no cover - optional dependency during development
    openai = None


LOGGER = logging.getLogger("audio_score_server")


@dataclass
class ClipReference:
    model_key: str
    video_id: str
    clip_id: str
    transcript: str
    transcript_path: Path
    hidden_path: Path
    audio_path: Path


@dataclass
class ModelSpec:
    key: str
    type: str
    model_id: str
    processor_id: str | None = None
    transcriber_id: str | None = None
    sampling_rate: int | None = None
    style_model_id: str | None = None
    api_base: str | None = None
    api_key_env: str | None = None
    system_prompt: str | None = None
    user_prompt: str | None = None
    temperature: float | None = None
    max_completion_tokens: int | None = None


class BaseModelRunner:
    """Common interface for model-specific inference logic."""

    supports_batching: bool = False

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

    def transcribe_and_embed_batch(
        self,
        waveforms: Sequence[np.ndarray],
        sampling_rates: Sequence[int],
    ) -> List[Tuple[str, torch.Tensor]]:
        """Fallback batching logic executes requests sequentially."""
        if len(waveforms) != len(sampling_rates):
            raise ValueError("waveforms and sampling_rates must have the same length.")
        return [
            self.transcribe_and_embed(waveform, sampling_rate)
            for waveform, sampling_rate in zip(waveforms, sampling_rates)
        ]

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


class WhisperStyleModelRunner(BaseModelRunner):
    """Whisper for ASR paired with a Hugging Face speaker/prosody embedding model."""

    DEFAULT_STYLE_MODEL = "microsoft/wavlm-base-plus-sv"

    supports_batching = True

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
        self.feature_padding_value = float(getattr(self.processor.feature_extractor, "padding_value", 0.0))
        self.expected_mel_length = self._determine_expected_mel_length()

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
            padding="longest",
            return_attention_mask=True,
        )
        input_features = features.input_features.to(self.device, dtype=self.model_dtype)
        input_features = self._pad_mel_features(input_features)

        with torch.no_grad():
            generated_ids = self.model.generate(input_features=input_features)

        transcript = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        style_audio, _ = self._resample_if_needed(waveform, sampling_rate, self.style_sample_rate)
        hidden_state = self._encode_style_with_hf(style_audio)
        return transcript, hidden_state

    def transcribe_and_embed_batch(
        self,
        waveforms: Sequence[np.ndarray],
        sampling_rates: Sequence[int],
    ) -> List[Tuple[str, torch.Tensor]]:
        if len(waveforms) != len(sampling_rates):
            raise ValueError("waveforms and sampling_rates must have the same length.")

        asr_audio: List[np.ndarray] = []
        style_audio_inputs: List[np.ndarray] = []

        for waveform, sampling_rate in zip(waveforms, sampling_rates):
            whisper_audio, _ = self._resample_if_needed(waveform, sampling_rate, self.feature_rate)
            style_audio, _ = self._resample_if_needed(waveform, sampling_rate, self.style_sample_rate)
            asr_audio.append(whisper_audio)
            style_audio_inputs.append(style_audio)

        features = self.processor(
            asr_audio,
            sampling_rate=self.feature_rate,
            return_tensors="pt",
            padding="longest",
            return_attention_mask=True,
        )
        input_features = features.input_features.to(self.device, dtype=self.model_dtype)
        input_features = self._pad_mel_features(input_features)

        with torch.no_grad():
            generated_ids = self.model.generate(input_features=input_features)

        transcripts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        style_embeddings = self._encode_style_batch(style_audio_inputs)

        results: List[Tuple[str, torch.Tensor]] = []
        for transcript, hidden_state in zip(transcripts, style_embeddings):
            results.append((transcript.strip(), hidden_state))

        return results

    @staticmethod
    def _load_hf_processor(model_id: str):
        try:
            return AutoProcessor.from_pretrained(model_id)
        except Exception:
            return AutoFeatureExtractor.from_pretrained(model_id)

    def _encode_style_with_hf(self, audio: np.ndarray) -> torch.Tensor:
        embeddings = self._encode_style_batch([audio])
        if not embeddings:
            raise RuntimeError("Failed to compute style embedding.")
        return embeddings[0]

    def _encode_style_batch(self, audios: Sequence[np.ndarray]) -> List[torch.Tensor]:
        if self.style_processor is None or self.style_model is None:
            raise RuntimeError("Hugging Face style model not initialized.")
        if not audios:
            return []

        inputs = self.style_processor(
            list(audios),
            sampling_rate=self.style_sample_rate,
            return_tensors="pt",
            padding=True,
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

        normalized = F.normalize(style_vec, p=2, dim=-1).float().cpu()
        return [normalized[idx] for idx in range(normalized.size(0))]

    def _determine_expected_mel_length(self) -> int:
        extractor = self.processor.feature_extractor
        for attr in ("nb_max_frames", "max_length", "n_frames"):
            value = getattr(extractor, attr, None)
            if value:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue
        value = getattr(self.model.config, "max_source_positions", None)
        if value:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
        return 3000

    def _pad_mel_features(self, features: torch.Tensor) -> torch.Tensor:
        target_len = max(self.expected_mel_length, 3000)
        current_len = features.size(-1)
        if current_len == target_len:
            return features
        if current_len > target_len:
            return features[..., :target_len]
        pad_amount = target_len - current_len
        return F.pad(features, (0, pad_amount), value=self.feature_padding_value)


class HiggsAudioUnderstandingRunner(BaseModelRunner):
    """Runner that proxies similarity scoring to the Higgs Audio Understanding API."""

    DEFAULT_MODEL = "higgs-audio-understanding-Hackathon"
    DEFAULT_BASE_URL = "https://hackathon.boson.ai/v1"
    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful assistant. I will provide you with an audio file, "
        "Answer my questions about the audio."
    )
    DEFAULT_USER_PROMPT = (
        "How many different speakers are in the audio? "
    )
    TRANSFER_FORMAT = "wav"

    def __init__(self, spec: ModelSpec, device: torch.device):
        super().__init__(spec, device)
        if openai is None:
            raise RuntimeError(
                "The 'openai' package is required for the Higgs Audio Understanding runner."
            )
        self.model_id = spec.model_id or self.DEFAULT_MODEL
        self.api_base = spec.api_base or self.DEFAULT_BASE_URL
        self.api_key_env = spec.api_key_env or "BOSON_API_KEY"
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Environment variable '{self.api_key_env}' must be set for model '{self.model_id}'."
            )
        self.client = openai.Client(api_key=api_key, base_url=self.api_base)
        self.temperature = spec.temperature if spec.temperature is not None else 1.0
        self.max_tokens = spec.max_completion_tokens or 256
        self.system_prompt = spec.system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.user_prompt = spec.user_prompt or self.DEFAULT_USER_PROMPT

    def prepare_reference(
        self,
        video_id: str,
        clip_id: str,
        clip_path: Path,
        transcript_path: Path,
        hidden_path: Path,
    ) -> ClipReference:
        """Store minimal metadata for remote similarity comparisons."""
        return ClipReference(
            self.key,
            video_id,
            clip_id,
            "",
            transcript_path,
            hidden_path,
            clip_path,
        )

    async def score_similarity(
        self,
        service: "AudioScoreService",
        model_key: str,
        video_id: str,
        clip_id: str,
        query_waveform: np.ndarray,
        query_sampling_rate: int,
    ) -> Dict[str, float]:
        reference = service.get_reference(model_key, video_id, clip_id)
        try:
            ref_waveform, ref_sr = service._load_audio_file(reference.audio_path)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to load reference audio: {exc}") from exc

        ref_b64 = service.waveform_to_base64(ref_waveform, ref_sr)
        qry_b64 = service.waveform_to_base64(query_waveform, query_sampling_rate)
        concatenated_audio = ref_b64 + qry_b64

        def _call_remote() -> Dict[str, float]:
            return self._invoke_remote_similarity(concatenated_audio)

        try:
            return await asyncio.to_thread(_call_remote)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"Higgs audio understanding request failed: {exc}") from exc

    def _invoke_remote_similarity(self, concatenated_audio_b64: str) -> Dict[str, float]:
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": concatenated_audio_b64,
                                    "format": self.TRANSFER_FORMAT,
                                },
                            },
                        ],
                    },
                    {"role": "user", "content": self.user_prompt},
                ],
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"Failed to reach Higgs audio understanding: {exc}") from exc

        return self._parse_similarity_response(response)

    def _parse_similarity_response(self, response: Any) -> Dict[str, float]:
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, KeyError) as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail="Malformed response from Higgs audio understanding.") from exc

        text = self._coerce_content_to_text(content).strip()
        json_payload = self._extract_json_block(text)
        try:
            payload = json.loads(json_payload)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Unable to parse similarity response: {text}",
            ) from exc

        def _extract_score(key: str) -> float | None:
            if key not in payload:
                return None
            try:
                value = float(payload[key])
            except (TypeError, ValueError) as exc:  # noqa: BLE001
                raise HTTPException(
                    status_code=502,
                    detail=f"Invalid value returned for '{key}': {payload[key]!r}",
                ) from exc
            return max(0.0, min(1.0, value))

        sim_score = _extract_score("sim_score")
        audio_sim = _extract_score("audio_sim_score")
        text_sim = _extract_score("text_sim_score")

        if audio_sim is None and text_sim is None and sim_score is None:
            raise HTTPException(
                status_code=502,
                detail="No similarity scores returned by Higgs audio understanding endpoint.",
            )

        if sim_score is None and audio_sim is not None and text_sim is not None:
            sim_score = float((audio_sim + text_sim) / 2.0)

        sim_score = sim_score if sim_score is not None else audio_sim or text_sim or 0.0
        audio_sim = audio_sim if audio_sim is not None else sim_score
        text_sim = text_sim if text_sim is not None else sim_score

        return {
            "sim_score": sim_score,
            "audio_sim_score": audio_sim,
            "text_sim_score": text_sim,
        }

    @staticmethod
    def _coerce_content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                text = getattr(item, "text", None)
                if text is None and isinstance(item, dict):
                    text = item.get("text")
                if text is None:
                    text = str(item)
                parts.append(text)
            return "".join(parts)
        return str(content)

    @staticmethod
    def _extract_json_block(text: str) -> str:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group(0)
        return text


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
        self.batchers: Dict[str, DynamicBatcher] = {}
        self.batch_max_size = int(os.getenv("AUDIO_SCORE_MAX_BATCH_SIZE", "8"))
        self.batch_max_wait_ms = float(os.getenv("AUDIO_SCORE_MAX_BATCH_WAIT_MS", "300"))
        if self.batch_max_size < 1:
            self.batch_max_size = 1
        if self.batch_max_wait_ms < 0:
            self.batch_max_wait_ms = 0.0
        self.disable_cache_flush = self._env_flag("AUDIO_SCORE_DISABLE_CACHE_FLUSH")
        self.requested_model_keys = self._load_requested_model_keys()

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
            await self._initialize_batchers()
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

        model_specs = self._parse_model_specs(self.config)
        if self.requested_model_keys:
            missing = sorted(set(self.requested_model_keys) - set(model_specs))
            if missing:
                LOGGER.warning("Requested model keys not found in config: %s", ", ".join(missing))
            model_specs = {key: spec for key, spec in model_specs.items() if key in self.requested_model_keys}
        self.model_specs = model_specs
        if not self.model_specs:
            if self.requested_model_keys:
                raise RuntimeError(
                    "No models available after applying AUDIO_SCORE_MODEL_KEY(S) filter."
                )
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

    @staticmethod
    def _env_flag(name: str) -> bool:
        value = os.getenv(name)
        if value is None:
            return False
        return value.strip().lower() in {"1", "true", "yes", "on"}

    def _load_requested_model_keys(self) -> set[str] | None:
        candidates = []
        multi = os.getenv("AUDIO_SCORE_MODEL_KEYS")
        single = os.getenv("AUDIO_SCORE_MODEL_KEY")
        if multi:
            candidates.append(multi)
        if single:
            candidates.append(single)

        keys: set[str] = set()
        for raw in candidates:
            for segment in raw.split(","):
                token = segment.strip()
                if token:
                    keys.add(token)
        return keys or None

    async def _initialize_batchers(self) -> None:
        self.batchers.clear()
        if self.batch_max_size <= 1:
            return

        enabled_models: list[str] = []
        for key, runner in self.runners.items():
            if not runner.supports_batching:
                continue
            batcher = DynamicBatcher(
                runner=runner,
                max_batch_size=self.batch_max_size,
                max_wait_ms=self.batch_max_wait_ms,
            )
            await batcher.start()
            self.batchers[key] = batcher
            enabled_models.append(key)

        if enabled_models:
            LOGGER.info(
                "Dynamic batching enabled for models: %s (max_batch_size=%d, max_wait_ms=%.2f)",
                ", ".join(sorted(enabled_models)),
                self.batch_max_size,
                self.batch_max_wait_ms,
            )

    async def shutdown(self) -> None:
        if not self.batchers:
            return
        tasks = [batcher.stop() for batcher in self.batchers.values()]
        await asyncio.gather(*tasks, return_exceptions=True)
        self.batchers.clear()

    def _flush_cache_root(self, cache_path: Path) -> None:
        """Remove cache directory before rebuilding cached references."""
        if self.disable_cache_flush:
            LOGGER.info("Skipping cache clear for %s (AUDIO_SCORE_DISABLE_CACHE_FLUSH=1).", cache_path)
            return
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
                api_base=entry.get("api_base"),
                api_key_env=entry.get("api_key_env"),
                system_prompt=entry.get("system_prompt"),
                user_prompt=entry.get("user_prompt"),
                temperature=entry.get("temperature"),
                max_completion_tokens=entry.get("max_completion_tokens"),
            )
            specs[spec.key] = spec
        return specs

    def _create_runner(self, spec: ModelSpec) -> BaseModelRunner:
        if spec.type.startswith("whisper-style"):
            return WhisperStyleModelRunner(spec, self.device)
        if spec.type.startswith("higgs-endpoint"):
            return HiggsAudioUnderstandingRunner(spec, self.device)
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

        if isinstance(runner, HiggsAudioUnderstandingRunner):
            return runner.prepare_reference(video_id, clip_id, clip_path, transcript_path, hidden_path)

        if transcript_path.exists() and hidden_path.exists():
            transcript = transcript_path.read_text(encoding="utf-8").strip()
            if transcript:
                return ClipReference(
                    model_key,
                    video_id,
                    clip_id,
                    transcript,
                    transcript_path,
                    hidden_path,
                    clip_path,
                )

        waveform, sampling_rate = self._load_audio_file(clip_path)
        transcript, hidden = runner.transcribe_and_embed(waveform, sampling_rate)

        transcript_path.write_text(transcript, encoding="utf-8")
        torch.save(hidden, hidden_path)

        return ClipReference(
            model_key,
            video_id,
            clip_id,
            transcript,
            transcript_path,
            hidden_path,
            clip_path,
        )

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

    @staticmethod
    def _waveform_to_wav_bytes(waveform: np.ndarray, sampling_rate: int) -> bytes:
        """Serialize a mono waveform into 16-bit PCM WAV bytes."""
        clipped = np.clip(waveform, -1.0, 1.0)
        pcm = (clipped * np.iinfo(np.int16).max).astype(np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(int(sampling_rate))
            handle.writeframes(pcm.tobytes())
        return buffer.getvalue()

    @staticmethod
    def waveform_to_base64(waveform: np.ndarray, sampling_rate: int) -> str:
        wav_bytes = AudioScoreService._waveform_to_wav_bytes(waveform, sampling_rate)
        return base64.b64encode(wav_bytes).decode("utf-8")

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

    def get_reference(self, model_key: str, video_id: str, clip_id: str) -> ClipReference:
        reference_map = self.references.get(model_key)
        if reference_map is None:
            raise HTTPException(status_code=404, detail=f"Unknown model '{model_key}'.")

        reference = reference_map.get((video_id, clip_id))
        if reference is None:
            raise HTTPException(status_code=404, detail="Reference clip not found.")
        return reference

    def score(
        self,
        model_key: str,
        video_id: str,
        clip_id: str,
        query_transcript: str,
        query_hidden: torch.Tensor,
    ) -> Dict[str, float]:
        reference = self.get_reference(model_key, video_id, clip_id)

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


@dataclass
class BatchRequest:
    waveform: np.ndarray
    sampling_rate: int
    future: asyncio.Future


class DynamicBatcher:
    """Simple dynamic batcher that groups requests before invoking the runner."""

    def __init__(
        self,
        runner: BaseModelRunner,
        max_batch_size: int,
        max_wait_ms: float,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1.")

        self.runner = runner
        self.max_batch_size = max_batch_size
        self.max_wait_seconds = max(0.0, max_wait_ms) / 1000.0

        self._queue: asyncio.Queue[BatchRequest | None] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._closed = False

    async def start(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._closed = True
        if self._worker_task is not None and not self._worker_task.done():
            await self._queue.put(None)
            await self._worker_task
        # Drain remaining queued items and notify callers.
        try:
            while True:
                request = self._queue.get_nowait()
                if request is None:
                    continue
                if not request.future.done():
                    request.future.set_exception(RuntimeError("Batcher stopped."))
        except asyncio.QueueEmpty:
            pass

    async def enqueue(self, waveform: np.ndarray, sampling_rate: int) -> Tuple[str, torch.Tensor]:
        if self._closed:
            raise RuntimeError("Cannot enqueue request on a stopped batcher.")

        await self.start()
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        request = BatchRequest(waveform=waveform, sampling_rate=sampling_rate, future=future)
        await self._queue.put(request)
        return await future

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            request = await self._queue.get()
            if request is None:
                break

            batch = [request]
            deadline = loop.time() + self.max_wait_seconds

            while len(batch) < self.max_batch_size:
                timeout = deadline - loop.time()
                if timeout <= 0:
                    break
                try:
                    next_request = await asyncio.wait_for(self._queue.get(), timeout)
                except asyncio.TimeoutError:
                    break

                if next_request is None:
                    await self._queue.put(None)
                    break
                batch.append(next_request)

            await self._process_batch(batch)

    async def _process_batch(self, batch: List[BatchRequest]) -> None:
        waveforms = [item.waveform for item in batch]
        sampling_rates = [item.sampling_rate for item in batch]

        try:
            results = await asyncio.to_thread(
                self.runner.transcribe_and_embed_batch,
                waveforms,
                sampling_rates,
            )
        except Exception as exc:  # noqa: BLE001
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(exc)
            return

        if len(results) != len(batch):
            error = RuntimeError("Batch result count does not match request count.")
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(error)
            return

        for item, result in zip(batch, results):
            if not item.future.done():
                item.future.set_result(result)


service = AudioScoreService()
app = FastAPI()


@app.on_event("startup")
async def handle_startup() -> None:
    await service.startup()


@app.on_event("shutdown")
async def handle_shutdown() -> None:
    await service.shutdown()


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
    batcher = service.batchers.get(model_key)
    if isinstance(runner, HiggsAudioUnderstandingRunner):
        result = await runner.score_similarity(
            service=service,
            model_key=model_key,
            video_id=video_id,
            clip_id=clip_id,
            query_waveform=waveform,
            query_sampling_rate=sampling_rate,
        )
    else:
        if batcher is not None:
            query_transcript, query_hidden = await batcher.enqueue(waveform, sampling_rate)
        else:
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

    socket_path = os.getenv("AUDIO_SCORE_SERVER_SOCKET")
    if socket_path:
        uds_path = Path(socket_path).resolve()
        uds_path.parent.mkdir(parents=True, exist_ok=True)
        if uds_path.exists():
            uds_path.unlink()
        uvicorn.run(
            "main:app",
            uds=str(uds_path),
            reload=False,
        )
    else:
        host = os.getenv("AUDIO_SCORE_SERVER_HOST", "0.0.0.0")
        port = int(os.getenv("AUDIO_SCORE_SERVER_PORT", "9000"))
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=False,
        )
