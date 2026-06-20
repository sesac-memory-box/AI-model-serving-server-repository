import io
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from schemas import TTSRequest
from openai_client import get_openai_client, get_openai_tts_model, raise_openai_http_error

router = APIRouter()


@router.post("/")
async def text_to_speech(request: TTSRequest):
    client = get_openai_client()
    try:
        response = await client.audio.speech.create(
            model=get_openai_tts_model(),
            voice=request.voice,
            input=request.text,
        )
    except Exception as error:
        raise_openai_http_error(error)

    return StreamingResponse(
        io.BytesIO(response.content),
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline; filename=speech.mp3"},
    )
