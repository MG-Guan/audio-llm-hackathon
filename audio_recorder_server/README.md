# Audio Recorder Server

This hackathon utility exposes a minimal FastAPI server with a single-page UI that lets you record microphone input in the browser and save the captured audio as timestamped WAV files on disk.

## Setup

```bash
pip install fastapi uvicorn requests
```

## Usage

```bash
uvicorn main:app --reload --port 8000
```

Then visit [http://localhost:8000](http://localhost:8000) and use the **Start Recording** button. The resulting WAV files will appear under `recordings/` inside this directory. You can also access them from the browser at `/recordings/<filename>`.

Use the optional *Filename Prefix* field in the UI to prepend a custom label before each timestamp (e.g. `session1_20240101_123000.wav`).

## REST API

For automated ingestion you can `POST` to `/api/recordings` with `multipart/form-data` containing:

- `user_id`, `video_id`, `clip_id`, `timestamp` (text fields)
- `file` (WAV audio upload)

Example using `curl`:

```bash
curl -X POST http://localhost:8000/api/recordings \
  -F user_id=U123 \
  -F video_id=V456 \
  -F clip_id=C789 \
  -F timestamp=20240101T010203 \
  -F file=@sample.wav
```

The server saves the audio as `recordings/U123_V456_C789_20240101T010203.wav`, calls the local scoring service, and responds with:

```json
{
  "message": "Recording saved and scored.",
  "filename": "U123_V456_C789_20240101T010203.wav",
  "path": "hackathon/audio_recorder_server/recordings/U123_V456_C789_20240101T010203.wav",
  "score": 0.87
}
```

Set the `SCORE_ENDPOINT` environment variable (default `http://127.0.0.1:9000/score`) so the server knows where to send scoring requests. The endpoint is expected to accept `POST` requests with `{"file_path": "<absolute path>"}` payload and return JSON containing a numeric `score`.
