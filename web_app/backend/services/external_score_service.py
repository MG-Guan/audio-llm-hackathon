import aiohttp
import os
import mimetypes
from fastapi import HTTPException
from sqlalchemy.orm import Session
import file_service
import logging
from typing import Optional

SCORING_API_URL = "https://example.com/score"

# Register common audio MIME types
mimetypes.add_type('audio/webm', '.webm')
mimetypes.add_type('audio/mp4', '.mp4')
mimetypes.add_type('audio/ogg', '.ogg')
mimetypes.add_type('audio/wav', '.wav')
mimetypes.add_type('audio/x-wav', '.wav')  # Alternative WAV type
mimetypes.add_type('audio/mpeg', '.mp3')
mimetypes.add_type('audio/aac', '.aac')

def get_audio_content_type(filename: str) -> str:
    """
    Infer content type from filename extension.
    Falls back to audio/wav if type cannot be determined.
    """
    content_type, _ = mimetypes.guess_type(filename)
    
    # If no type found or not an audio type, check common extensions manually
    if not content_type or not content_type.startswith('audio/'):
        ext = os.path.splitext(filename)[1].lower()
        content_types = {
            '.webm': 'audio/webm',
            '.mp4': 'audio/mp4',
            '.m4a': 'audio/mp4',
            '.ogg': 'audio/ogg',
            '.oga': 'audio/ogg',
            '.wav': 'audio/wav',
            '.mp3': 'audio/mpeg',
            '.aac': 'audio/aac'
        }
        content_type = content_types.get(ext, 'audio/wav')  # Default to wav if unknown
    
    return content_type

async def score_audio(uid: str, cid: str, file_id: str, db: Session) -> dict:
    return await score_audio(uid, cid, file_id, db, None)

async def score_audio(uid: str, cid: str, file_id: str, db: Session, model: Optional[str] = None) -> dict:
    """
    Send audio file to external scoring API and get results.
    """
    try:
        # Get the file path from file_id
        file_path, original_filename = file_service.get_file_path(file_id, db)
        logging.info(f"File path: {file_path}, original filename: {original_filename}")
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Audio file not found")

        # Determine content type from original filename
        content_type = get_audio_content_type(original_filename)

        # Prepare the form data
        form_data = aiohttp.FormData()
        form_data.add_field('uid', uid)
        form_data.add_field('cid', cid)
        if model:
            form_data.add_field('model', model)
        
        # Add the audio file with inferred content type
        file_data = open(file_path, 'rb')
        try:
            form_data.add_field('audio', 
                            file_data,
                            filename=original_filename,
                            content_type=content_type)

            logging.info(f"Sending request to {SCORING_API_URL} with form data: {form_data}")
            # Make the API call
            async with aiohttp.ClientSession() as session:
                async with session.post(SCORING_API_URL, data=form_data) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise HTTPException(
                            status_code=response.status,
                            detail=f"Scoring API error: {error_text}"
                        )
                    return await response.json()
        finally:
            file_data.close()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to score audio: {str(e)}"
        )