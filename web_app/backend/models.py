from sqlalchemy import Column, String, Integer, Float, JSON, DateTime
from sqlalchemy.sql import func
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from database import Base

class File(Base):
    __tablename__ = "files"

    file_id = Column(String, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

class SubmissionScore(BaseModel):
    model: str
    score: float
    audioSimilarityScore: Optional[float] = None
    textSimilarityScore: Optional[float] = None

class Submission(Base):
    __tablename__ = "submissions"

    id = Column(String, primary_key=True, index=True)
    program_id = Column(String, nullable=False, index=True)
    clip_id = Column(String, nullable=False)
    username = Column(String, nullable=False)
    file_id = Column(String, nullable=False)
    model = Column(String, nullable=False)
    score = Column(Float, nullable=False)
    audio_similarity_score = Column(Float, nullable=True)
    text_similarity_score = Column(Float, nullable=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

# Pydantic models for response
class FileResponse(BaseModel):
    file_id: str
    filename: str
    timestamp: datetime

    class Config:
        from_attributes = True

class SubmissionResponse(BaseModel):
    id: str
    program_id: str
    clip_id: str
    username: str
    file_id: str
    model: str
    score: float
    audio_similarity_score: Optional[float] = None
    text_similarity_score: Optional[float] = None
    timestamp: datetime
    rank: Optional[int] = None
    total_submissions: Optional[int] = None

    class Config:
        from_attributes = True

class ScoreRequest(BaseModel):
    program_id: str
    clip_id: str
    file_id: str
    model: Optional[str] = None

class SubmissionCreate(BaseModel):
    program_id: str
    clip_id: str
    username: str
    file_id: str
    model: Optional[str] = None