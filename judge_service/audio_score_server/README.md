# Audio Score Server

This FastAPI service evaluates recorded clips against pre-segmented reference videos. On startup it loads one or more ASR/encoder models (e.g. Whisper or Higgs-Audio), precomputes transcripts and hidden representations for every `.mov` clip under `video_clip/`, and then exposes a `/score` endpoint that returns audio/text similarity scores for user uploads.

## Setup

1. Install the project dependencies (includes FastAPI, Transformers, Torch, Torchaudio, etc.):
   ```bash
   pip install -r ../../requirements.txt
   ```
2. Ensure `ffmpeg` is available so `pydub` can extract audio from `.mov` files.

## Configuration

Edit `config.json` to describe the models you want to load and where reference clips live:

```json
{
  "models": [
    {
      "key": "whisper-small",
      "type": "whisper",
      "model_id": "openai/whisper-small"
    },
    {
      "key": "higgs-audio-v2",
      "type": "higgs_audio",
      "model_id": "bosonai/higgs-audio-v2-understanding-3B",
      "processor_id": "openai/whisper-large-v3-turbo",
      "transcriber_id": "openai/whisper-small"
    }
  ],
  "video_clip_dir": "../video_clip",
  "cache_dir": "./cache",
  "target_sampling_rate": 16000
}
```

- `models`: list of scoring backends. Whisper models only require `model_id`. Higgs-Audio models also need `processor_id` (for encoder features) and `transcriber_id` (for generating transcripts).
- `video_clip_dir`: folder containing `.mov` reference clips (relative to this directory).
- `cache_dir`: where transcripts and encoder tensors are persisted. Cached artifacts are organized per model so subsequent restarts skip recomputation.
- `target_sampling_rate`: audio is resampled to this value before feature extraction.

If any model fails to load, the server logs the error and continues with the remaining models. Startup only aborts when none succeed.

## Run the server

Single worker (binds to TCP by default; respects `AUDIO_SCORE_SERVER_*` env vars):

```bash
python run_server.py --host 0.0.0.0 --port 9000
```

For production you can launch multiple workers behind a local Nginx gateway:

```bash
python run_server_with_gate.py --workers 2 --listen-port 9000
```

`run_server_with_gate.py` starts the requested number of Unix-socket workers and generates a temp Nginx config that writes logs to `logs/` and stores request bodies under `logs/client_body_temp/` (avoids `/var/lib/nginx` permission errors). Stop the script with `Ctrl+C` to shut down Nginx and all workers.

Startup phases for each worker:
1. Load every configured model onto the available device (CUDA if present, else CPU).
2. Walk `video_clip_dir` and produce transcripts + encoder tensors for each `.mov`, caching the results under `cache/<model>/<video>/<clip>/`.
3. Begin serving requests either on the requested TCP port or Unix domain socket.

If all workers share a single GPU they still serialize on that device; add more GPUs or lower model cost to increase throughput.

### Environment variables

Key knobs exposed by the service:

- `AUDIO_SCORE_MODEL_KEYS` / `AUDIO_SCORE_MODEL_KEY`: limit which models load from `config.json` (comma separated).
- `AUDIO_SCORE_MAX_BATCH_SIZE` (default `8`): maximum requests combined in the dynamic batcher.
- `AUDIO_SCORE_MAX_BATCH_WAIT_MS` (default `10`): longest wait to fill a batch before dispatch.
- `AUDIO_SCORE_DISABLE_CACHE_FLUSH=1`: preserve the existing cache directory on startup (useful when multiple workers share it).
- `AUDIO_SCORE_SERVER_SOCKET`: when set, workers bind to the specified Unix domain socket; `run_server_with_gate.py` handles this automatically.
- `AUDIO_SCORE_SERVER_HOST` / `AUDIO_SCORE_SERVER_PORT`: override TCP binding when not using sockets.

## Scoring API

`POST /score` with `multipart/form-data` fields:

- `uid`: video id (matches the reference filename prefix)
- `cid`: clip id (matches the suffix)
- `audio`: the user-uploaded audio sample to compare (any format `pydub` can decode; WAV recommended)
- `model` *(optional)*: model key from `config.json`. When omitted the server uses the first successfully loaded model.

Example:

```bash
curl -X POST http://localhost:9000/score \
  -F uid=PeppaPig-s06e02 \
  -F cid=part01 \
  -F model=whisper-small \
  -F audio=@../audio_clip/PeppaPig-s06e02_part01.wav
```

Response schema:

```json
{
  "model": "whisper-small",
  "sim_score": 0.83,
  "audio_sim_score": 0.78,
  "text_sim_score": 0.88
}
```

- `audio_sim_score`: cosine similarity between cached encoder features and the upload.
- `text_sim_score`: normalized string similarity between reference transcript and the upload transcript.
- `sim_score`: simple average of audio and text scores.

If the specified `uid`/`cid` combination is unknown, the server returns `404`.

## Cache Maintenance

Cached files are regenerated automatically when missing. Delete a subdirectory under `cache/` if you want to force a refresh for a particular model/video/clip.
