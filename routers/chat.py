import json
import logging
import os

from fastapi import APIRouter
from schemas import ChatRequest, ChatResponse, SummaryRequest, SummaryResponse
from openai_client import get_openai_client, get_openai_model, raise_openai_http_error
from services.retrieval import build_context, retrieve

router = APIRouter()
logger = logging.getLogger(__name__)

NO_CONTEXT_ANSWER = "현재 저장된 자료에서 관련 내용을 찾지 못했습니다. 다른 키워드로 질문해 주세요."
SUMMARY_FALLBACK = SummaryResponse(
    places=[],
    people=[],
    next_topics=["최근 기억", "가족 이야기", "추억 회상"],
)
SYSTEM_PROMPT = """너는 공모전 MVP용 기억 회상 챗봇이다.
반드시 검색된 자료를 우선 사용한다.
자료에 없는 내용을 단정하지 않는다.
노년층이 이해하기 쉽게 짧고 자연스러운 한국어로 답한다.
따뜻한 말투를 사용한다.
의학적 진단이나 치료 조언은 하지 않는다."""


def _summary_threshold() -> int:
    return int(os.getenv("SUMMARY_THRESHOLD", "10"))


async def _generate_summary(messages: list, previous_summary: str | None = None) -> str:
    client = get_openai_client()
    conversation_text = "\n".join(f"{m.role}: {m.content}" for m in messages)
    summary_input = conversation_text
    if previous_summary:
        summary_input = f"[이전 요약]\n{previous_summary}\n\n[새 대화]\n{conversation_text}"

    try:
        res = await client.chat.completions.create(
            model=get_openai_model(),
            messages=[
                {
                    "role": "system",
                    "content": "이전 요약과 새 대화를 합쳐 핵심 내용과 맥락만 포함한 한국어 요약으로 갱신하세요.",
                },
                {"role": "user", "content": summary_input},
            ],
        )
    except Exception as error:
        raise_openai_http_error(error)
    return res.choices[0].message.content or ""


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        documents = await retrieve(request.query)
    except Exception as error:
        logger.warning(
            "Qdrant unavailable. Falling back to empty retrieval. error=%s: %s",
            type(error).__name__,
            error,
        )
        documents = []

    if not documents:
        return ChatResponse(answer=NO_CONTEXT_ANSWER, summary=None)

    try:
        context = build_context(documents)
    except Exception as error:
        logger.warning(
            "Failed to build RAG context. Falling back to no context. error=%s: %s",
            type(error).__name__,
            error,
        )
        context = ""

    if not context.strip():
        return ChatResponse(answer=NO_CONTEXT_ANSWER, summary=None)

    client = get_openai_client()

    messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\n[검색된 자료]\n{context}"}]

    if request.summary:
        messages.append({"role": "system", "content": f"[이전 대화 요약]\n{request.summary}"})
    if request.messages:
        messages += [{"role": m.role, "content": m.content} for m in request.messages]

    messages.append({"role": "user", "content": request.query})

    try:
        response = await client.chat.completions.create(
            model=get_openai_model(),
            messages=messages,
        )
    except Exception as error:
        raise_openai_http_error(error)
    answer = response.choices[0].message.content or ""

    new_summary = None
    if request.messages and len(request.messages) >= _summary_threshold():
        new_summary = await _generate_summary(request.messages, request.summary)

    return ChatResponse(answer=answer, summary=new_summary)


@router.post("/summary", response_model=SummaryResponse)
async def session_summary(request: SummaryRequest):
    client = get_openai_client()
    conversation_text = "\n".join(
        f"{m.role}: {m.content}" for m in request.messages
    )

    try:
        res = await client.chat.completions.create(
            model=get_openai_model(),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "다음 대화를 분석하여 JSON 형식으로만 반환하세요.\n"
                        "언급된 장소 목록, 사람 이름 목록, 치매 노인과 꺼낼 수 있는 다음 주제 3가지를 추출하세요.\n"
                        '형식: {"places": [...], "people": [...], "next_topics": [...]}'
                    ),
                },
                {"role": "user", "content": conversation_text},
            ],
            response_format={"type": "json_object"},
        )
        result = json.loads(res.choices[0].message.content or "")
    except json.JSONDecodeError:
        return SUMMARY_FALLBACK
    except Exception as error:
        raise_openai_http_error(error)

    return SummaryResponse(
        places=result.get("places", []),
        people=result.get("people", []),
        next_topics=result.get("next_topics") or SUMMARY_FALLBACK.next_topics,
    )
