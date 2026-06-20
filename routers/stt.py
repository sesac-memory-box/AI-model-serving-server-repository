from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from schemas import STTResponse
from openai_client import get_openai_client, get_openai_stt_model, raise_openai_http_error

router = APIRouter()
SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".webm", ".mp4", ".mpeg", ".mpga"}


@router.post("/", response_model=STTResponse)
async def speech_to_text(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="audio file must not be empty.",
        )

    suffix = Path(audio.filename or "").suffix.lower()
    if suffix not in SUPPORTED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="unsupported audio file type. Supported: mp3, wav, m4a, webm, mp4, mpeg, mpga.",
        )

    client = get_openai_client()
    try:
        response = await client.audio.transcriptions.create(
            model=get_openai_stt_model(),
            file=(audio.filename, audio_bytes, audio.content_type),
        )
    except Exception as error:
        raise_openai_http_error(error)

    return STTResponse(transcript=response.text)
