import asyncio
from types import SimpleNamespace

from routers import chat as chat_router
from schemas import ChatRequest, Message, SummaryRequest


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
