# Higgs Audio Hackathon Judge Service

This workspace contains the judge service and tooling used in the Boson AI 2025 Higgs Audio Hackathon. It bundles utilities for preparing media clips, recording user submissions, generating automatic transcripts, and serving similarity scores against curated references.

## Repository Layout

- `audio_score_server/` – FastAPI service that loads one or more ASR/encoder models, precomputes embeddings for reference clips, and exposes a `/score` endpoint. See the local README for configuration details.
- `audio_recorder_server/` – FastAPI app with a browser-based recorder UI plus a `/api/recordings` ingestion endpoint. Captured WAV files are written to `audio_recorder_server/recordings/`.
- `tool/` – Command-line helpers for preparing data (e.g. splitting `.mov` files into segments, extracting WAV stems).
- `audio_clip/` – Canonical reference audio for each video segment (mirrors the contents of `audio_recorder_server/recordings/` when testing end-to-end).
- `video_clip/` – Trimmed `.mov` clips created from the raw footage; these are what the score server transcribes/embeds during warm-up.
- `video_raw/` – Original long-form recordings kept for archival and for regenerating `video_clip/` segments when the source material changes.
- `test/` – Smoke-test scripts for exercising the scoring pipeline end-to-end.
- `requirements.txt` – Python dependencies shared by both services and the helper tools.

## End-to-End Flow

1. **Clip preparation**  
   Use tools such as `split_mov.py` to divide long videos into aligned `.mov` clips and extract `.wav` counterparts if needed.

2. **Reference caching**  
   `audio_score_server` scans `video_clip/` at startup, runs ASR and encoder forward passes, and saves transcripts plus hidden tensors to `audio_score_server/cache/<model>/<video>/<clip>/`.

3. **Recording**  
   Hosts run `audio_recorder_server` so participants can submit recordings via the web UI or the `/api/recordings` endpoint. Files land in `audio_recorder_server/recordings/`.

4. **Scoring**  
   Clients upload recordings to `audio_score_server`’s `/score` endpoint. The service looks up the cached reference for the specified `uid` (video id) and `cid` (clip id), computes text/audio similarity, and returns the scores.

## Quick Start

```bash
pip install -r requirements.txt
# ensure ffmpeg is installed for audio extraction
```

### Start the services

```bash
# 1. Scoring backend (loads models and prepares caches)
python audio_score_server/main.py  # default port 9000

# 2. Recorder UI (optional helper for collecting submissions)
python audio_recorder_server/main.py  # default port 8000
```

Wait for the score server logs to report “Reference preparation complete” before issuing requests.

## APIs

### `/score` (audio_score_server)

`POST multipart/form-data`

- `uid`: video id (e.g. `PeppaPig-s06e02`)
- `cid`: clip id (e.g. `part02`)
- `audio`: uploaded WAV (or any format supported by `pydub`)
- `model` *(optional)*: model key defined in `audio_score_server/config.json`

Response:

```json
{
  "model": "whisper-small",
  "sim_score": 0.80,
  "audio_sim_score": 0.74,
  "text_sim_score": 0.86
}
```

### `/api/recordings` (audio_recorder_server)

`POST multipart/form-data`

- `user_id`, `video_id`, `clip_id`, `timestamp`: identifiers used to construct the output filename.
- `file`: WAV payload.

Response:

```json
{
  "message": "Recording saved.",
  "filename": "U123_PeppaPig-s06e02_part02_20250101T120000.wav",
  "path": "judge_service/audio_recorder_server/recordings/U123_PeppaPig-s06e02_part02_20250101T120000.wav"
}
```

## Testing

- `test/test_score_server.bash` – Example curl script to probe the scoring endpoint.
- `audio_score_server/cache/` – Inspect cached transcripts or delete per-model folders to force regeneration.

## Useful Tips

- The score server tolerates individual model failures during startup; ensure at least one model loads successfully to keep the service running.
- When adding new clips, drop the files into `video_clip/` and restart the score server so caches refresh.
- The recorder and score services are intentionally decoupled: you can deploy them independently or integrate recorded files into any custom evaluation pipeline.
