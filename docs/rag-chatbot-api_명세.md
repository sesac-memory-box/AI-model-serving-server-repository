# RAG Chatbot API 설계 문서

## 엔드포인트 목록

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/chat/` | 대화 응답 생성 |
| POST | `/chat/summary` | 대화 세션 요약 |
| POST | `/stt/` | 음성 → 텍스트 변환 |
| POST | `/tts/` | 텍스트 → 음성 변환 |

---

## Chat

### `POST /chat/`

- 사용자 쿼리에 대해 RAG 기반 AI 응답을 반환
- 이전 대화는 `messages` 배열 또는 압축된 `summary` 문자열 중 하나로 전달
- [대화 0~임계치 개수 이전] Streamlit → messages 전송
- [대화 임계치 개수 도달] FastAPI → summary 생성해서 응답에 포함
- [대화 임계치 개수 도달 이후] Streamlit이 messages 초기화, summary 보관 (messages로 streamlit 서버에 보여지는 것과 FastAPI에 요청 시 보낼 summary 이후 내용을 별도로 유지해야 함)
- → 새 대화는 summary + 새 messages(0~N)로 전송
- [messages가 다시 임계치 도달] 또 summary 갱신 (이전 summary + 새 messages로 재요약)


**Request Body**

```json
{
  "query": "오늘 날씨 어때요?",
  "messages": [
    { "role": "user", "content": "안녕하세요" },
    { "role": "assistant", "content": "안녕하세요! 오늘 어떻게 지내셨나요?" }
  ],
  "summary": null
}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `query` | string | O | 현재 사용자 입력 |
| `messages` | Message[] | X | 이전 대화 메시지 목록 (`summary`와 택일 or 둘 다 전송) |
| `summary` | string | X | 압축된 이전 대화 요약 (`messages`와 택일 or 둘 다 전송) |

> `messages`와 `summary` 중 하나만 전달하거나 둘 다 전달함, 둘 다 없으면 첫 번째 발화로 처리됨.

**Message 객체**

| 필드 | 타입 | 설명 |
|------|------|------|
| `role` | `"user"` \| `"assistant"` | 발화 주체 |
| `content` | string | 발화 내용 |

**Response Body**

```json
{
  "answer": "오늘은 맑고 따뜻한 날씨네요. 산책 나가시면 좋을 것 같아요.",
  "summary": null
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `answer` | string | AI 응답 텍스트 |
| `summary` | string \| null | 대화 요약본. `messages` 수가 임계치 이상일 때만 반환, 그 외에는 `null` |

---

### `POST /chat/summary`

- 대화 세션이 끝났을 때 전체 메시지를 분석하여 장소, 인물, 다음 대화 추천 주제를 추출

**Request Body**

```json
{
  "messages": [
    { "role": "user", "content": "어제 아들이랑 한강 공원에 갔어요" },
    { "role": "assistant", "content": "한강 공원이요! 날씨는 좋았나요?" },
    { "role": "user", "content": "응, 민준이가 자전거도 빌려줬어" }
  ]
}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `messages` | Message[] | O | 분석할 전체 대화 메시지 목록 |

**Response Body**

```json
{
  "places": ["한강 공원"],
  "people": ["민준"],
  "next_topics": [
    "한강 공원에서의 추억",
    "민준이와의 최근 일상",
    "자전거 타기"
  ]
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `places` | string[] | 대화에서 언급된 장소 목록 |
| `people` | string[] | 대화에서 언급된 인물 이름 목록 |
| `next_topics` | string[] | 다음 대화에서 꺼낼 수 있는 주제 3가지 |

---

## STT (Speech-to-Text)

### `POST /stt/`

- 음성 파일을 받아 텍스트로 변환

**Request**

`multipart/form-data` 형식으로 전송

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `audio` | File | O | 음성 파일 (mp3, wav, m4a, webm 등) |

**Response Body**

```json
{
  "transcript": "오늘 날씨가 참 좋네요"
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `transcript` | string | 변환된 텍스트 |

---

## TTS (Text-to-Speech)

### `POST /tts/`

- 텍스트를 음성으로 변환하여 MP3 스트림으로 반환

**Request Body**

```json
{
  "text": "안녕하세요! 오늘 어떻게 지내셨나요?",
  "voice": "alloy"
}
```

| 필드 | 타입 | 필수 | 기본값 | 설명 |
|------|------|------|--------|------|
| `text` | string | O | - | 음성으로 변환할 텍스트 |
| `voice` | string | X | `"alloy"` | 음성 종류: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer` |

**Response**

`audio/mpeg` 스트림으로 반환

| 헤더 | 값 |
|------|----|
| `Content-Type` | `audio/mpeg` |
| `Content-Disposition` | `inline; filename=speech.mp3` |

---
