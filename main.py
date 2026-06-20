from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from routers import chat, stt, tts

app = FastAPI(title="RAG Chatbot API")

app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(stt.router, prefix="/stt", tags=["stt"])
app.include_router(tts.router, prefix="/tts", tags=["tts"])


@app.get("/health")
async def health():
    return {"status": "ok"}
