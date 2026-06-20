from schemas import ChatRequest, Message, SummaryResponse


def test_chat_request_schema_accepts_optional_history():
    request = ChatRequest(query="1970년대 서울역에 대해 이야기해줘")

    assert request.query == "1970년대 서울역에 대해 이야기해줘"
    assert request.messages is None
    assert request.summary is None


def test_chat_request_schema_accepts_messages_and_summary():
    request = ChatRequest(
        query="다음 이야기를 이어줘",
        messages=[Message(role="user", content="안녕하세요")],
        summary="이전 대화 요약",
    )

    assert request.messages is not None
    assert request.messages[0].role == "user"
    assert request.summary == "이전 대화 요약"


def test_summary_response_schema():
    response = SummaryResponse(
        places=["한강 공원"],
        people=["민준"],
        next_topics=["한강 공원에서의 추억", "민준이와의 최근 일상", "자전거 타기"],
    )

    assert response.places == ["한강 공원"]
    assert response.people == ["민준"]
    assert len(response.next_topics) == 3
