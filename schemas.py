from typing import Literal, Optional

from pydantic import BaseModel, field_validator


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: Optional[list[Message]] = None
    summary: Optional[str] = None
    query: str


class ChatResponse(BaseModel):
    answer: str
    summary: Optional[str] = None


class SummaryRequest(BaseModel):
    messages: list[Message]


class SummaryResponse(BaseModel):
    places: list[str]
    people: list[str]
    next_topics: list[str]


class STTResponse(BaseModel):
    transcript: str


class TTSRequest(BaseModel):
    text: str
    voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = "alloy"

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value
