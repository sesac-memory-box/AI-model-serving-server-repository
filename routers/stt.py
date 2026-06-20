from fastapi import APIRouter, UploadFile, File
from schemas import STTResponse
from openai_client import client

router = APIRouter()


@router.post("/", response_model=STTResponse)
async def speech_to_text(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    response = await client.audio.transcriptions.create(
        model="whisper-1",
        file=(audio.filename, audio_bytes, audio.content_type),
    )
    return STTResponse(transcript=response.text)
