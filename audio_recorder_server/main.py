import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import requests

ALLOWED_WAV_CONTENT_TYPES = {"audio/wav", "audio/x-wav"}

APP_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = APP_ROOT / "static"
RECORDINGS_ROOT = APP_ROOT / "recordings"

# Endpoint of the local scoring service; configurable via environment variable.
SCORE_ENDPOINT = os.getenv("SCORE_ENDPOINT", "http://127.0.0.1:9000/score")

# Ensure required folders exist at startup.
STATIC_ROOT.mkdir(parents=True, exist_ok=True)
RECORDINGS_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI()


def sanitize_fragment(value: str | None) -> str:
    """Normalize user-provided tokens so they are safe for filenames."""
    if value is None:
        return ""
    normalized = value.replace(" ", "_")
    sanitized = "".join(char for char in normalized if char.isalnum() or char in {"_", "-"})
    return sanitized


async def persist_upload(file: UploadFile, output_path: Path) -> int:
    """Write the uploaded file to disk in chunks to avoid high memory usage."""
    await file.seek(0)
    total_bytes = 0

    with output_path.open("wb") as destination:
        while True:
            chunk = await file.read(1_048_576)  # 1 MiB chunks
            if not chunk:
                break
            total_bytes += len(chunk)
            destination.write(chunk)

    await file.close()

    if total_bytes == 0:
        output_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    return total_bytes


def request_score(file_path: Path) -> float:
    """Synchronously call the local scoring endpoint and return the numeric score."""
    if not SCORE_ENDPOINT:
        raise RuntimeError("SCORE_ENDPOINT environment variable is not configured.")

    response = requests.post(
        SCORE_ENDPOINT,
        json={"file_path": str(file_path)},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()

    if "score" not in payload:
        raise ValueError("Scoring response missing 'score' field.")

    return float(payload["score"])


async def fetch_score(file_path: Path) -> float:
    """Wrapper that runs the blocking scoring request in a thread."""
    try:
        score = await asyncio.to_thread(request_score, file_path)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Scoring service failed: {exc}") from exc
    return score


@app.middleware("http")
async def add_caching_headers(request: Request, call_next):
    """Disable caching so updated frontend loads reliably during development."""
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/", response_class=FileResponse)
async def index() -> FileResponse:
    """Serve the single-page recorder UI."""
    index_html = STATIC_ROOT / "index.html"
    if not index_html.exists():
        raise HTTPException(status_code=500, detail="Recorder UI missing.")
    return FileResponse(path=index_html)


@app.post("/upload")
async def upload_audio(
    file: Annotated[UploadFile, File(...)],
    prefix: Annotated[str | None, Form()] = None,
) -> JSONResponse:
    """Accept a WAV file upload and persist it to disk with a timestamped filename."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file is missing a filename.")

    if file.content_type not in ALLOWED_WAV_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Expected WAV audio but received '{file.content_type}'.",
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_name = Path(file.filename).stem
    sanitized_name = sanitize_fragment(prefix) if prefix else sanitize_fragment(source_name)
    if not sanitized_name:
        sanitized_name = "recording"
    output_name = f"{sanitized_name}_{timestamp}.wav"
    output_path = RECORDINGS_ROOT / output_name

    # Persist the uploaded file to disk.
    await persist_upload(file, output_path)

    return JSONResponse(
        {
            "message": "Recording saved.",
            "filename": output_name,
            "path": str(output_path),
        }
    )


@app.post("/api/recordings")
async def upload_audio_with_ids(
    user_id: Annotated[str, Form(...)],
    video_id: Annotated[str, Form(...)],
    clip_id: Annotated[str, Form(...)],
    timestamp: Annotated[str, Form(...)],
    file: Annotated[UploadFile, File(...)],
) -> JSONResponse:
    """Upload a WAV file, store it with composite identifiers, and return a score."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file is missing a filename.")

    if file.content_type not in ALLOWED_WAV_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Expected WAV audio but received '{file.content_type}'.",
        )

    fragments = {}
    for key, raw_value in {
        "user_id": user_id,
        "video_id": video_id,
        "clip_id": clip_id,
        "timestamp": timestamp,
    }.items():
        sanitized = sanitize_fragment(raw_value)
        if not sanitized:
            raise HTTPException(status_code=400, detail=f"{key} cannot be empty.")
        fragments[key] = sanitized

    output_name = "{u}_{v}_{c}_{t}.wav".format(
        u=fragments["user_id"],
        v=fragments["video_id"],
        c=fragments["clip_id"],
        t=fragments["timestamp"],
    )
    output_path = RECORDINGS_ROOT / output_name

    await persist_upload(file, output_path)

    score = await fetch_score(output_path)

    return JSONResponse(
        {
            "message": "Recording saved and scored.",
            "filename": output_name,
            "path": str(output_path),
            "score": score,
        }
    )


# Expose recordings for convenience during development (e.g., manual playback).
app.mount(
    "/recordings",
    StaticFiles(directory=RECORDINGS_ROOT),
    name="recordings",
)
