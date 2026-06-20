import os

from fastapi import HTTPException, status
from openai import AsyncOpenAI

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
