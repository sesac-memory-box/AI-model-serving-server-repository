from types import SimpleNamespace

from services.retrieval import RetrievedDocument, build_context, extract_content, points_to_documents


def test_payload_without_content_is_excluded_from_context():
    context = build_context(
        [
            RetrievedDocument(
                payload={
                    "title": "제목만 있는 자료",
                    "source": "테스트",
                }
            )
        ]
    )

    assert context == ""


def test_payload_content_key_priority():
    payload = {
        "body": "body 값",
        "chunk": "chunk 값",
        "text": "text 값",
        "content_text": "content_text 값",
        "content": "content 값",
    }

    assert extract_content(payload) == "content 값"


def test_context_formatting_uses_content_key_priority():
    context = build_context(
        [
            RetrievedDocument(
                payload={
                    "title": "서울역",
                    "source": "아카이브",
                    "url": "https://example.com/seoul-station",
                    "year": "1975",
                    "era": "1970년대",
                    "category": "교통",
                    "content_text": "서울역은 많은 사람이 오가던 교통 중심지였습니다.",
                }
            ),
            RetrievedDocument(
                payload={
                    "title": "본문 없는 자료",
                    "source": "아카이브",
                }
            ),
            RetrievedDocument(
                payload={
                    "title": "추가 자료",
                    "chunk": "명절에는 귀성객으로 붐볐습니다.",
                }
            ),
        ]
    )

    assert "[자료 1]" in context
    assert "제목: 서울역" in context
    assert "출처: 아카이브" in context
    assert "URL: https://example.com/seoul-station" in context
    assert "연도: 1975" in context
    assert "시대: 1970년대" in context
    assert "분류: 교통" in context
    assert "내용: 서울역은 많은 사람이 오가던 교통 중심지였습니다." in context
    assert "[자료 2]" in context
    assert "내용: 명절에는 귀성객으로 붐볐습니다." in context
    assert "본문 없는 자료" not in context


def test_points_to_documents_filters_results_below_score_threshold():
    documents = points_to_documents(
        [
            SimpleNamespace(payload={"content": "낮은 점수 자료"}, score=0.2),
            SimpleNamespace(payload={"content": "높은 점수 자료"}, score=0.8),
        ],
        score_threshold=0.5,
    )

    assert len(documents) == 1
    assert documents[0].payload["content"] == "높은 점수 자료"


def test_points_to_documents_keeps_score_066_without_score_threshold():
    documents = points_to_documents(
        [
            SimpleNamespace(payload={"content": "남대문 시장 자료"}, score=0.66),
        ]
    )

    assert len(documents) == 1
    assert documents[0].payload["content"] == "남대문 시장 자료"
