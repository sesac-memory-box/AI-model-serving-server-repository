import asyncio
from types import SimpleNamespace

from routers import chat as chat_router
from schemas import ChatRequest, Message, SummaryRequest


def test_build_retrieval_query_includes_user_history():
    retrieval_query = chat_router.build_retrieval_query(
        "그때 같이 갔던 곳이 어디였죠?",
        messages=[
            Message(role="user", content="예전에 자주 가던 시장이 있었는데"),
            Message(role="assistant", content="어떤 시장이었나요? 기억나시는 게 있으세요?"),
            Message(role="user", content="남대문 시장이었나... 거기서 국밥을 먹었던 것 같아요"),
            Message(role="assistant", content="남대문 시장 국밥, 참 맛있었겠네요!"),
        ],
        summary=None,
    )

    assert "남대문 시장" in retrieval_query
    assert retrieval_query.endswith("그때 같이 갔던 곳이 어디였죠?")


def test_build_retrieval_query_includes_summary_first():
    retrieval_query = chat_router.build_retrieval_query(
        "그때 같이 갔던 곳이 어디였죠?",
        messages=[Message(role="user", content="남대문 시장에서 국밥을 먹었어요")],
        summary="사용자는 남대문 시장 추억을 이야기했다.",
    )

    assert retrieval_query.startswith("사용자는 남대문 시장 추억을 이야기했다.")
    assert "남대문 시장에서 국밥을 먹었어요" in retrieval_query


def test_build_retrieval_query_uses_only_query_when_history_is_empty():
    retrieval_query = chat_router.build_retrieval_query(
        "그때 같이 갔던 곳이 어디였죠?",
        messages=[],
        summary=None,
    )

    assert retrieval_query == "그때 같이 갔던 곳이 어디였죠?"


def test_chat_uses_history_for_retrieval_query(monkeypatch):
    captured = {}

    async def fake_retrieve(query):
        captured["query"] = query
        return []

    class FakeCompletions:
        async def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="남대문 시장을 말씀하신 것 같아요.")
                    )
                ]
            )

    class FakeClient:
        chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(chat_router, "retrieve", fake_retrieve)
    monkeypatch.setattr(chat_router, "get_openai_client", lambda: FakeClient())

    response = asyncio.run(
        chat_router.chat(
            ChatRequest(
                query="그때 같이 갔던 곳이 어디였죠?",
                messages=[
                    Message(role="user", content="남대문 시장에서 국밥을 먹었어요"),
                ],
                summary=None,
            )
        )
    )

    assert response.answer == "남대문 시장을 말씀하신 것 같아요."
    assert "남대문 시장" in captured["query"]


def test_build_chat_messages_includes_conversation_history_section():
    request = ChatRequest(
        query="그때 같이 갔던 곳이 어디였죠?",
        messages=[Message(role="user", content="남대문 시장에서 국밥을 먹었어요")],
        summary=None,
    )

    messages = chat_router.build_chat_messages(request, "검색 context")

    assert "[대화 히스토리]" in messages[0]["content"]
    assert "남대문 시장에서 국밥을 먹었어요" in messages[0]["content"]


def test_system_prompt_allows_user_history_as_evidence():
    assert (
        "사용자가 이전 대화에서 직접 언급한 내용은 근거로 사용할 수 있다"
        in chat_router.SYSTEM_PROMPT
    )


def test_chat_generates_answer_from_history_when_retrieval_is_empty(monkeypatch):
    captured = {}

    async def fake_retrieve(query):
        return []

    class FakeCompletions:
        async def create(self, **kwargs):
            captured["messages"] = kwargs["messages"]
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=(
                                "말씀하신 내용으로 보면 남대문 시장을 떠올리신 것 같아요."
                            )
                        )
                    )
                ]
            )

    class FakeClient:
        chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(chat_router, "retrieve", fake_retrieve)
    monkeypatch.setattr(chat_router, "get_openai_client", lambda: FakeClient())

    response = asyncio.run(
        chat_router.chat(
            ChatRequest(
                query="그때 같이 갔던 곳이 어디였죠?",
                messages=[
                    Message(role="user", content="예전에 자주 가던 시장이 있었는데"),
                    Message(role="assistant", content="어떤 시장이었나요?"),
                    Message(
                        role="user",
                        content="남대문 시장이었나... 거기서 국밥을 먹었던 것 같아요",
                    ),
                ],
                summary=None,
            )
        )
    )

    assert response.answer != chat_router.NO_CONTEXT_ANSWER
    assert "남대문" in response.answer
    assert "[대화 히스토리]" in captured["messages"][0]["content"]


def test_chat_does_not_fallback_when_retrieval_has_content(monkeypatch):
    async def fake_retrieve(query):
        return [
            SimpleNamespace(
                payload={
                    "title": "남대문 시장",
                    "content": "남대문 시장에서 국밥을 먹던 기억",
                },
                score=0.66,
            )
        ]

    class FakeCompletions:
        async def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="남대문 시장에 같이 가셨던 것 같아요.")
                    )
                ]
            )

    class FakeClient:
        chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(chat_router, "retrieve", fake_retrieve)
    monkeypatch.setattr(chat_router, "get_openai_client", lambda: FakeClient())

    response = asyncio.run(
        chat_router.chat(
            ChatRequest(
                query="그때 같이 갔던 곳이 어디였죠?",
                messages=[
                    Message(role="user", content="남대문 시장에서 국밥을 먹었어요"),
                ],
                summary=None,
            )
        )
    )

    assert response.answer == "남대문 시장에 같이 가셨던 것 같아요."
    assert response.answer != chat_router.NO_CONTEXT_ANSWER


def test_chat_returns_fallback_without_openai_when_search_result_is_empty(monkeypatch):
    async def fake_retrieve(query):
        return []

    def fail_openai_call():
        raise AssertionError("OpenAI must not be called when retrieval has no results")

    monkeypatch.setattr(chat_router, "retrieve", fake_retrieve)
    monkeypatch.setattr(chat_router, "get_openai_client", fail_openai_call)

    response = asyncio.run(
        chat_router.chat(
            ChatRequest(query="절대검색안될키워드12345", messages=[], summary=None)
        )
    )

    assert response.answer == chat_router.NO_CONTEXT_ANSWER
    assert response.summary is None


def test_chat_returns_fallback_without_openai_when_retrieval_fails(monkeypatch):
    async def fake_retrieve(query):
        raise ConnectionError("All connection attempts failed")

    def fail_openai_call():
        raise AssertionError("OpenAI must not be called when retrieval fails")

    monkeypatch.setattr(chat_router, "retrieve", fake_retrieve)
    monkeypatch.setattr(chat_router, "get_openai_client", fail_openai_call)

    response = asyncio.run(
        chat_router.chat(
            ChatRequest(query="서울역", messages=[], summary=None)
        )
    )

    assert response.answer == chat_router.NO_CONTEXT_ANSWER
    assert response.summary is None


def test_chat_returns_fallback_without_openai_when_context_is_empty(monkeypatch):
    async def fake_retrieve(query):
        return [object()]

    def fake_build_context(documents):
        return ""

    def fail_openai_call():
        raise AssertionError("OpenAI must not be called when context is empty")

    monkeypatch.setattr(chat_router, "retrieve", fake_retrieve)
    monkeypatch.setattr(chat_router, "build_context", fake_build_context)
    monkeypatch.setattr(chat_router, "get_openai_client", fail_openai_call)

    response = asyncio.run(
        chat_router.chat(
            ChatRequest(query="본문 없는 검색 결과", messages=[], summary=None)
        )
    )

    assert response.answer == chat_router.NO_CONTEXT_ANSWER
    assert response.summary is None


def test_chat_summary_returns_fallback_when_openai_json_is_invalid(monkeypatch):
    class FakeCompletions:
        async def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="not json")
                    )
                ]
            )

    class FakeClient:
        chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(chat_router, "get_openai_client", lambda: FakeClient())

    response = asyncio.run(
        chat_router.session_summary(
            SummaryRequest(messages=[Message(role="user", content="한강 공원에 갔어요")])
        )
    )

    assert response.places == []
    assert response.people == []
    assert response.next_topics == ["최근 기억", "가족 이야기", "추억 회상"]
