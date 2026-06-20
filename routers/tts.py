import io
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from schemas import TTSRequest
from openai_client import get_openai_client

router = APIRouter()


@router.post("/")
async def text_to_speech(request: TTSRequest):
    client = get_openai_client()
    response = await client.audio.speech.create(
        model="tts-1",
        voice=request.voice,
        input=request.text,
    )
    return StreamingResponse(
        io.BytesIO(response.content),
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline; filename=speech.mp3"},
    )
