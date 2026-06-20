from pydantic import BaseModel
from typing import Optional


class Message(BaseModel):
    role: str
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
    voice: str = "alloy" 
