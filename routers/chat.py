import json
from fastapi import APIRouter
from schemas import ChatRequest, ChatResponse, SummaryRequest, SummaryResponse
from openai_client import client
from services.retrieval import retrieve

router = APIRouter()

SUMMARY_THRESHOLD = 10


async def _generate_summary(messages: list) -> str:
    conversation_text = "\n".join(f"{m.role}: {m.content}" for m in messages)
    res = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "다음 대화를 핵심 내용과 맥락만 포함한 한국어로 간결하게 요약하세요.",
            },
            {"role": "user", "content": conversation_text},
        ],
    )
    return res.choices[0].message.content


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    context = await retrieve(request.query)

    system_content = "당신은 치매 노인과 대화하는 AI 어시스턴트이다."
    if context:
        system_content += f"\n\n[참고 문서]\n{context}"

    messages = [{"role": "system", "content": system_content}]

    if request.summary:
        messages.append({"role": "system", "content": f"[이전 대화 요약]\n{request.summary}"})
    if request.messages:
        messages += [{"role": m.role, "content": m.content} for m in request.messages]

    messages.append({"role": "user", "content": request.query})

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )
    answer = response.choices[0].message.content

    new_summary = None
    if request.messages and len(request.messages) >= SUMMARY_THRESHOLD:
        new_summary = await _generate_summary(request.messages)

    return ChatResponse(answer=answer, summary=new_summary)


@router.post("/summary", response_model=SummaryResponse)
async def session_summary(request: SummaryRequest):
    conversation_text = "\n".join(
        f"{m.role}: {m.content}" for m in request.messages
    )

    rag_context = await retrieve(conversation_text)

    res = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "다음 대화를 분석하여 JSON 형식으로만 반환하세요.\n"
                    "언급된 장소 목록, 사람 이름 목록, 치매 노인과 꺼낼 수 있는 다음 주제 3가지를 추출하세요.\n"
                    "아래 참고 문서를 주제 추천에 활용하세요.\n\n"
                    f"[참고 문서]\n{rag_context}\n\n"
                    '형식: {"places": [...], "people": [...], "next_topics": [...]}'
                ),
            },
            {"role": "user", "content": conversation_text},
        ],
        response_format={"type": "json_object"},
    )
    result = json.loads(res.choices[0].message.content)

    return SummaryResponse(
        places=result.get("places", []),
        people=result.get("people", []),
        next_topics=result.get("next_topics", []),
    )
