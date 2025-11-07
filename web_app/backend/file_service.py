import os
import hashlib
import shutil
from datetime import datetime
import random
from sqlalchemy.orm import Session
from fastapi import UploadFile, HTTPException
from models import File
import uuid
import re

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

def calculate_sha1(file_path: str) -> str:
    """Calculate SHA1 hash of a file."""
    sha1 = hashlib.sha1()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha1.update(chunk)
    return sha1.hexdigest()

def generate_file_id(file_hash: str, timestamp: datetime) -> str:
    """Generate unique file_id using SHA1 hash, timestamp with milliseconds, and random component."""
    # Format timestamp with milliseconds
    timestamp_ms = int(timestamp.timestamp() * 1000)
    # Generate random number seeded by timestamp milliseconds
    random.seed(timestamp_ms)
    random_component = random.randint(0, 999999)
    # Combine all components
    combined = f"{file_hash}_{timestamp_ms}_{random_component}"
    return hashlib.sha1(combined.encode()).hexdigest()

async def save_upload_file(file: UploadFile, db: Session) -> File:
    """Save uploaded file and create database record."""
    temp_path = None  # Initialize outside try block
    final_path = None  # Track final path for cleanup
    
    try:
        # Stream file with size limit
        file_size = 0
        temp_path = os.path.join(UPLOAD_DIR, f"temp_{uuid.uuid4()}")
        
        with open(temp_path, "wb") as buffer:
            while chunk := await file.read(8192):
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE:
                    os.remove(temp_path)
                    raise HTTPException(413, "File too large")
                buffer.write(chunk)
        
        # Calculate file hash and generate unique file_id
        file_hash = calculate_sha1(temp_path)
        upload_time = datetime.utcnow()
        file_id = generate_file_id(file_hash, upload_time)
        
        # Move file to its final location using the file_id
        final_path = os.path.join(UPLOAD_DIR, file_id)
        os.rename(temp_path, final_path)
        temp_path = None  # Mark as moved

        # Create new file record
        db_file = File(
            file_id=file_id,
            filename=file.filename,
            timestamp=upload_time
        )
        db.add(db_file)
        db.commit()
        db.refresh(db_file)
        
        return db_file
    
    except Exception as e:
        # Rollback database changes
        db.rollback()
        
        # Clean up temporary file if it exists
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

def get_file_path(file_id: str, db: Session) -> tuple[str, str]:
    """Get file path and original filename for download."""
    # Validate file_id format (should be sha1 string)
    if not re.match(r'^[a-f0-9]{1,40}$', file_id):
        raise HTTPException(400, "Invalid file ID")
        
    db_file = db.query(File).filter(File.file_id == file_id).first()
    if not db_file:
        raise HTTPException(status_code=404, detail="File not found")
        
    file_path = os.path.join(UPLOAD_DIR, file_id)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on disk")
        
    return file_path, db_file.filename

def list_all_files(db: Session) -> list[File]:
    """Get all files from database."""
    return db.query(File).all()