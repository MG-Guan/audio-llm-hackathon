# Audio Recorder Server

This FastAPI app provides a lightweight browser UI for recording audio clips and persisting them to disk. The service focuses purely on capture and storage; scoring is handled separately by the `audio_score_server`.

## Setup

```bash
pip install fastapi uvicorn
```

## Run the server

```bash
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000
```

Visit [http://localhost:8000](http://localhost:8000) to access the recorder page. Saved WAV files appear under `recordings/` in this directory and can be downloaded directly via `/recordings/<filename>`. Use the optional **Filename Prefix** field to prepend a custom label before the timestamp (e.g. `session1_20240101_123000.wav`).

## REST API

For automated ingestion, `POST` to `/api/recordings` with `multipart/form-data` containing:

- `user_id`, `video_id`, `clip_id`, `timestamp` (text fields)
- `file` (WAV audio upload)

Example:

```bash
curl -X POST http://localhost:8000/api/recordings \
  -F user_id=U123 \
  -F video_id=V456 \
  -F clip_id=C789 \
  -F timestamp=20240101T010203 \
  -F file=@sample.wav
```

Response:

```json
{
  "message": "Recording saved.",
  "filename": "U123_V456_C789_20240101T010203.wav",
  "path": "hackathon/audio_recorder_server/recordings/U123_V456_C789_20240101T010203.wav"
}
```

> ℹ️ Downstream services (e.g. the audio score server) can watch the `recordings/` directory and trigger scoring pipelines as needed.
