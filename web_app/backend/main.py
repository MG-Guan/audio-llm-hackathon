from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Form
from fastapi.responses import FileResponse as FastAPIFileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc
from sqlalchemy.sql import func, and_
import uuid
import os
import aiohttp
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")  # Default to production for safety
RECAPTCHA_SITE_KEY = os.getenv("RECAPTCHA_SITE_KEY")
RECAPTCHA_SECRET_KEY = os.getenv("RECAPTCHA_SECRET_KEY")

from database import get_db, engine, Base
from models import (
    FileResponse, File as FileModel,
    Submission, SubmissionCreate, SubmissionResponse, ScoreRequest
)
import file_service
from services import external_score_service
from helpers.db_utils import reset_database

ALLOWED_SORT_FIELDS = {"score", "timestamp", "username", "model"}

# Create the FastAPI app
app = FastAPI(title="Dubbing Backend API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create tables
Base.metadata.create_all(bind=engine)

# Configure static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

async def verify_recaptcha(token: str) -> bool:
    """
    Verify Google reCAPTCHA v2 token.
    Returns True if verification is successful, False otherwise.
    """
    if not RECAPTCHA_SECRET_KEY:
        raise HTTPException(
            status_code=500,
            detail="reCAPTCHA secret key is not configured"
        )
    
    if not token:
        return False
    
    verify_url = "https://www.google.com/recaptcha/api/siteverify"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                verify_url,
                data={
                    "secret": RECAPTCHA_SECRET_KEY,
                    "response": token
                }
            ) as response:
                result = await response.json()
                return result.get("success", False)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to verify reCAPTCHA: {str(e)}"
        )

@app.post("/upload", response_model=FileResponse)
async def upload_file(
    file: UploadFile = File(...),
    token: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    Upload a file with reCAPTCHA verification.
    The request must include a valid reCAPTCHA token.
    """
    # Verify reCAPTCHA token
    is_valid = await verify_recaptcha(token)
    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail="reCAPTCHA verification failed"
        )
    
    return await file_service.save_upload_file(file, db)

@app.get("/files/{file_id}")
async def download_file(file_id: str, db: Session = Depends(get_db)):
    file_path, original_filename = file_service.get_file_path(file_id, db)
    return FastAPIFileResponse(
        path=file_path,
        filename=original_filename,
        media_type=external_score_service.get_audio_content_type(original_filename)
    )

@app.get("/files", response_model=List[FileResponse])
async def list_files(db: Session = Depends(get_db)):
    return file_service.list_all_files(db)

@app.post("/proxy/score")
async def score_audio_file(
    request: ScoreRequest,
    db: Session = Depends(get_db)
):
    """
    Score an uploaded audio file using the external scoring API.
    
    The audio file must have been previously uploaded using the /upload endpoint.
    The file_id from the upload response should be provided in the request.
    
    Request body format:
    {
        "program_id": "PeppaPig-s06e02",
        "clip_id": "part01",
        "file_id": "a6d1f7e99dcf17798a16b9078e1c140fcbf4f861",
        "model": "whisper" (optional)
    }
    """
    return await external_score_service.score_audio(
        request.program_id,
        request.clip_id,
        request.file_id,
        db,
        request.model
    )

@app.post("/submissions", response_model=SubmissionResponse)
async def create_submission(submission: SubmissionCreate, db: Session = Depends(get_db)):
    """
    Create a new submission by scoring an audio file and storing the result.
    """
    # Generate a unique ID for the submission
    submission_id = str(uuid.uuid4())
    
    # Get the score from external service
    score_result = await external_score_service.score_audio(
        submission.program_id,
        submission.clip_id,
        submission.file_id,
        db,
        submission.model
    )
    
    # Create submission record
    db_submission = Submission(
        id=submission_id,
        program_id=submission.program_id,
        clip_id=submission.clip_id,
        username=submission.username,
        file_id=submission.file_id,
        model=score_result["model"],
        score=score_result["sim_score"],
        audio_similarity_score=score_result.get("audio_sim_score"),
        text_similarity_score=score_result.get("text_sim_score")
    )
    
    db.add(db_submission)
    db.commit()
    db.refresh(db_submission)
    return db_submission

@app.get("/submissions", response_model=List[SubmissionResponse])
async def list_submissions(
    program_id: Optional[str] = None,
    clip_id: Optional[str] = None,
    model: Optional[str] = None,
    sort_by: str = "score",
    sort_direction: str = "desc",
    limit: int = 10,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """
    List submissions with optional filtering, sorting and pagination.
    """
    if sort_by not in ALLOWED_SORT_FIELDS:
        raise HTTPException(400, "Invalid sort field")
    if sort_direction not in {"asc", "desc"}:
        raise HTTPException(400, "Invalid sort direction")
    
    query = db.query(Submission)
    
    # Filtering conditions
    if program_id:
        query = query.filter(Submission.program_id == program_id)
    if clip_id:
        query = query.filter(Submission.clip_id == clip_id)
    if model:
        query = query.filter(Submission.model == model)

    # Apply sorting
    if sort_direction == "desc":
        query = query.order_by(desc(getattr(Submission, sort_by)))
    else:
        query = query.order_by(asc(getattr(Submission, sort_by)))
    
    return query.offset(offset).limit(limit).all()

@app.get("/leaderboard", response_model=List[SubmissionResponse])
async def get_leaderboard(
    program_id: str,
    clip_id: str,
    model: Optional[str] = None,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """
    List leaderboard entries with program_id and clip_id filtering. One username can only have one entry per program and clip.
    """
    query = db.query(Submission)
    
    # Filtering conditions
    query = query.filter(Submission.program_id == program_id)
    query = query.filter(Submission.clip_id == clip_id)
    if model:
        query = query.filter(Submission.model == model)

    # First, get the highest score for each user using a subquery
    max_scores = db.query(
        Submission.username,
        func.max(Submission.score).label('max_score')
    ).filter(
        Submission.program_id == program_id,
        Submission.clip_id == clip_id
    )
    if model:
        max_scores = max_scores.filter(Submission.model == model)
    max_scores = max_scores.group_by(Submission.username).subquery()

    # Then join back to get the full submission details for these max scores
    query = db.query(Submission).join(
        max_scores,
        and_(
            Submission.username == max_scores.c.username,
            Submission.score == max_scores.c.max_score
        )
    )

    # Apply sorting on the final results
    query = query.order_by(desc(Submission.score))

    return query.limit(limit).all()

@app.get("/submissions/{submission_id}", response_model=SubmissionResponse)
async def get_submission(submission_id: str, db: Session = Depends(get_db)):
    """
    Get a specific submission by ID, including its rank within the program.
    """
    submission = db.query(Submission).filter(Submission.id == submission_id).first()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    
    # Get all submissions for the program ordered by score to calculate rank
    program_submissions = db.query(Submission)\
        .filter(Submission.program_id == submission.program_id)\
        .filter(Submission.clip_id == submission.clip_id)\
        .filter(Submission.model == submission.model)\
        .order_by(desc(Submission.score))\
        .all()
    
    # Calculate rank (1-based index) and total submissions
    rank = next(i for i, s in enumerate(program_submissions, 1) if s.id == submission_id)
    total_submissions = len(program_submissions)
    
    # Create response with rank information
    response_dict = {
        **submission.__dict__,
        "rank": rank,
        "total_submissions": total_submissions
    }
    return response_dict

@app.post("/debug/reset-db")
async def debug_reset_database():
    """
    Debug endpoint to reset the database.
    WARNING: This will delete all data in the database.
    This endpoint is only available in development environment.
    """
    if ENVIRONMENT.lower() != "development":
        raise HTTPException(
            status_code=404,
            detail="Debug endpoints are not available in production"
        )
    return reset_database()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)