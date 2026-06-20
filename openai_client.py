import os

from fastapi import HTTPException, status
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    AsyncOpenAI,
    OpenAIError,
    RateLimitError,
)

_client: AsyncOpenAI | None = None


def get_openai_client() -> AsyncOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OPENAI_API_KEY is not configured. Set OPENAI_API_KEY in the environment.",
        )

    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=api_key)
    return _client


def get_openai_model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def get_openai_stt_model() -> str:
    return os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")


def get_openai_tts_model() -> str:
    return os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")


def raise_openai_http_error(error: Exception) -> None:
    if isinstance(error, AuthenticationError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="OpenAI authentication failed. Check OPENAI_API_KEY.",
        ) from error
    if isinstance(error, RateLimitError):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="OpenAI rate limit exceeded. Please try again later.",
        ) from error
    if isinstance(error, APITimeoutError):
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="OpenAI request timed out. Please try again later.",
        ) from error
    if isinstance(error, APIConnectionError):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not connect to OpenAI. Please try again later.",
        ) from error
    if isinstance(error, (APIError, OpenAIError)):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="OpenAI API request failed. Please try again later.",
        ) from error
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Unexpected OpenAI request failure.",
    ) from error
