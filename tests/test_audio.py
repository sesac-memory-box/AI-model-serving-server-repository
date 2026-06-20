import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from routers import stt as stt_router
from routers import tts as tts_router
from schemas import TTSRequest


def test_stt_empty_file_returns_400():
    class FakeUploadFile:
        filename = "empty.wav"
        content_type = "audio/wav"

        async def read(self):
            return b""

    upload = FakeUploadFile()

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(stt_router.speech_to_text(upload))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "audio file must not be empty."


def test_tts_response_mock(monkeypatch):
    calls = {}

    class FakeSpeech:
        async def create(self, **kwargs):
            calls.update(kwargs)
            return SimpleNamespace(content=b"fake mp3 bytes")

    class FakeClient:
        audio = SimpleNamespace(speech=FakeSpeech())

    monkeypatch.setattr(tts_router, "get_openai_client", lambda: FakeClient())

    async def run_request():
        return await tts_router.text_to_speech(
            TTSRequest(text="안녕하세요", voice="alloy")
        )

    response = asyncio.run(run_request())

    assert calls["model"] == "gpt-4o-mini-tts"
    assert calls["voice"] == "alloy"
    assert calls["input"] == "안녕하세요"
    assert response.media_type == "audio/mpeg"
    assert response.headers["content-disposition"] == "inline; filename=speech.mp3"
