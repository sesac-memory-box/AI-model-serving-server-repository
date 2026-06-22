import json
import logging
import os

from fastapi import APIRouter
from schemas import ChatRequest, ChatResponse, Message, SummaryRequest, SummaryResponse
from openai_client import get_openai_client, get_openai_model, raise_openai_http_error
from services.retrieval import build_context, extract_content, retrieve

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
사용자가 이전 대화에서 직접 언급한 내용은 근거로 사용할 수 있다.
검색 자료에 없더라도 대화 히스토리에 있는 장소, 인물, 사건은 사용자가 제공한 정보로 보고 답변에 활용한다.
사용자가 "그때", "거기", "그곳", "같이 갔던 곳"처럼 이전 대화를 가리키는 질문을 하면, 먼저 대화 히스토리에서 장소/인물/사건 후보를 찾는다.
대화 히스토리에 장소가 있으면 "말씀하신 내용으로 보면 ..." 형태로 답한다.
확실하지 않으면 단정하지 말고 "남대문 시장을 말씀하신 것 같아요"처럼 부드럽게 답한다.
노년층이 이해하기 쉽게 짧고 자연스러운 한국어로 답한다.
따뜻한 말투를 사용한다.
의학적 진단이나 치료 조언은 하지 않는다."""
RECENT_RETRIEVAL_MESSAGE_LIMIT = 6
ASSISTANT_RETRIEVAL_MESSAGE_LIMIT = 2


def _summary_threshold() -> int:
    return int(os.getenv("SUMMARY_THRESHOLD", "10"))


def build_retrieval_query(
    query: str,
    messages: list[Message] | None = None,
    summary: str | None = None,
) -> str:
    parts = []

    if summary and summary.strip():
        parts.append(summary.strip())

    recent_messages = (messages or [])[-RECENT_RETRIEVAL_MESSAGE_LIMIT:]
    assistant_messages = [
        message
        for message in recent_messages
        if message.role == "assistant" and message.content.strip()
    ]
    included_assistants = set(
        id(message) for message in assistant_messages[-ASSISTANT_RETRIEVAL_MESSAGE_LIMIT:]
    )

    for message in recent_messages:
        content = message.content.strip()
        if not content:
            continue
        if message.role == "user" or id(message) in included_assistants:
            parts.append(content)

    if query.strip():
        parts.append(query.strip())

    return "\n".join(parts)


def has_conversation_context(
    messages: list[Message] | None = None,
    summary: str | None = None,
) -> bool:
    if summary and summary.strip():
        return True
    return any(message.content.strip() for message in messages or [])


def build_conversation_history_section(
    messages: list[Message] | None = None,
    summary: str | None = None,
) -> str:
    parts = []
    if summary and summary.strip():
        parts.append(f"[이전 대화 요약]\n{summary.strip()}")

    conversation_lines = [
        f"{message.role}: {message.content.strip()}"
        for message in messages or []
        if message.content.strip()
    ]
    if conversation_lines:
        parts.append("[최근 대화]\n" + "\n".join(conversation_lines))

    return "\n\n".join(parts)


def build_chat_messages(request: ChatRequest, context: str) -> list[dict[str, str]]:
    conversation_history = build_conversation_history_section(
        request.messages,
        request.summary,
    )
    system_sections = [
        SYSTEM_PROMPT,
        f"[검색된 자료]\n{context.strip() or '검색된 자료 없음'}",
        f"[대화 히스토리]\n{conversation_history or '대화 히스토리 없음'}",
    ]
    messages = [{"role": "system", "content": "\n\n".join(system_sections)}]

    if request.messages:
        messages += [{"role": m.role, "content": m.content} for m in request.messages]

    messages.append({"role": "user", "content": request.query})
    return messages


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
    retrieval_query = build_retrieval_query(
        request.query,
        messages=request.messages,
        summary=request.summary,
    )
    logger.info(
        "Retrieval query length=%s preview=%s",
        len(retrieval_query),
        retrieval_query[:200],
    )

    try:
        documents = await retrieve(retrieval_query)
    except Exception as error:
        logger.warning(
            "Qdrant unavailable. Falling back to empty retrieval. error=%s: %s",
            type(error).__name__,
            error,
        )
        logger.warning("RAG fallback reason: qdrant_unavailable")
        return ChatResponse(answer=NO_CONTEXT_ANSWER, summary=None)

    conversation_context_exists = has_conversation_context(
        request.messages,
        request.summary,
    )

    if not documents and not conversation_context_exists:
        logger.warning("RAG fallback reason: no_retrieval_documents")
        return ChatResponse(answer=NO_CONTEXT_ANSWER, summary=None)

    context = ""
    if documents:
        try:
            context = build_context(documents)
        except Exception as error:
            logger.warning(
                "Failed to build RAG context. Falling back to no context. error=%s: %s",
                type(error).__name__,
                error,
            )
            context = ""

    context_document_count = sum(
        1 for document in documents if extract_content(getattr(document, "payload", {}))
    )
    logger.info("Context documents: %s", context_document_count)

    if not context.strip() and not conversation_context_exists:
        logger.warning("RAG fallback reason: no_context_documents")
        return ChatResponse(answer=NO_CONTEXT_ANSWER, summary=None)

    client = get_openai_client()
    messages = build_chat_messages(request, context)

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
