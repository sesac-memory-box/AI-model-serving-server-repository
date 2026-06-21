# AI Model Serving Server

FastAPI 기반 공모전 MVP용 AI 서빙 서버입니다. Data Pipeline Repository가 Qdrant Cloud에 적재한 `memory_box_contents` collection을 검색하고 OpenAI API로 RAG 채팅, 대화 요약, STT, TTS를 제공합니다.

## 환경변수 설정

```bash
cp .env.example .env
```

`.env`에 실제 값을 입력합니다. `.env`와 실제 API key는 커밋하지 않습니다.

```env
OPENAI_API_KEY=...
QDRANT_URL=Qdrant Cloud Endpoint URL
QDRANT_API_KEY=Qdrant Cloud API Key
QDRANT_COLLECTION=memory_box_contents
FASTEMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

## 환경변수

| 이름 | 설명 |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI API key |
| `OPENAI_MODEL` | 채팅/요약 모델, 기본 `gpt-4o-mini` |
| `OPENAI_STT_MODEL` | STT 모델, 기본 `gpt-4o-mini-transcribe` |
| `OPENAI_TTS_MODEL` | TTS 모델, 기본 `gpt-4o-mini-tts` |
| `QDRANT_URL` | Qdrant Cloud Endpoint URL |
| `QDRANT_API_KEY` | Qdrant Cloud API key |
| `QDRANT_COLLECTION` | Qdrant collection, `memory_box_contents` |
| `FASTEMBED_MODEL` | Query embedding 모델 |
| `RAG_TOP_K` | 검색 결과 개수 |
| `SUMMARY_THRESHOLD` | 채팅 요약 생성 기준 message 개수 |

Qdrant Cloud에 데이터가 적재되어 있지 않거나, Qdrant 연결 정보가 비어 있거나 잘못된 경우 `/chat/`은 500을 반환하지 않고 검색 결과 없음 fallback을 반환합니다. 이때 OpenAI는 호출하지 않습니다.

```json
{
  "answer": "현재 저장된 자료에서 관련 내용을 찾지 못했습니다. 다른 키워드로 질문해 주세요.",
  "summary": null
}
```

## 로컬 실행

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Docker build

```bash
docker build -t ai-serving-server:local .
```

`.dockerignore`에서 `.env`, `.env.*`, 가상환경, 캐시, 음성 산출물, Qdrant 로컬 저장소, Git metadata를 제외합니다. 실제 `OPENAI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`는 이미지에 넣지 말고 컨테이너 실행 시 `--env-file .env`로 주입합니다.

## Docker run

```bash
docker run --rm -p 8000:8000 --env-file .env ai-serving-server:local
```

## Docker Hub 이미지 실행

```bash
docker run -d \
  --name memorybox-ai-serving-server \
  -p 8000:8000 \
  --env-file .env \
  leo1504/memorybox-ai-serving-server:latest
```

## Docker Compose 실행

```bash
docker compose up --build
```

`docker-compose.yml`은 API service만 실행합니다. Qdrant는 로컬 Docker Compose service로 띄우지 않고 `.env`의 Qdrant Cloud 설정을 사용합니다.

Docker Hub에 올릴 때는 예를 들어 아래처럼 태그를 붙인 뒤, 사용자가 직접 로그인과 push를 수행합니다.

```bash
docker tag ai-serving-server:local leo1504/memorybox-ai-serving-server:latest
docker push leo1504/memorybox-ai-serving-server:latest
```

## API 확인

```bash
curl http://localhost:8000/health
```

```bash
curl -X POST http://localhost:8000/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "query":"서울역",
    "messages":[],
    "summary":null
  }'
```

검색 결과가 없거나 context가 비어 있으면 OpenAI를 호출하지 않고 fallback을 반환합니다.

```json
{
  "answer": "현재 저장된 자료에서 관련 내용을 찾지 못했습니다. 다른 키워드로 질문해 주세요.",
  "summary": null
}
```

## 주의사항

- Qdrant Cloud에 데이터가 적재되어 있어야 RAG 검색이 동작합니다.
- 데이터 적재는 Data Pipeline Repository에서 수행합니다.
- Qdrant Cloud가 비어 있으면 `/chat/`은 fallback 응답을 반환합니다.
- `QDRANT_API_KEY`는 공개 채널에 공유하지 않습니다.
- 로컬 Qdrant Docker service와 `qdrant_storage/`는 이 서버 배포 흐름에 포함하지 않습니다.

## Chat Summary 테스트

```bash
curl -X POST http://localhost:8000/chat/summary \
  -H "Content-Type: application/json" \
  -d '{
    "messages":[
      {"role":"user","content":"어제 아들이랑 한강 공원에 갔어요"},
      {"role":"assistant","content":"한강 공원이요. 날씨는 좋았나요?"}
    ]
  }'
```

## STT 테스트

```bash
curl -X POST http://localhost:8000/stt/ \
  -F "audio=@sample.wav"
```

지원 확장자: `mp3`, `wav`, `m4a`, `webm`, `mp4`, `mpeg`, `mpga`

## TTS 테스트

```bash
curl -X POST http://localhost:8000/tts/ \
  -H "Content-Type: application/json" \
  -d '{
    "text":"안녕하세요. 오늘은 어떤 추억을 이야기해볼까요?",
    "voice":"alloy"
  }' \
  --output speech.mp3
```

허용 voice: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`

## 검증

```bash
python -m compileall main.py openai_client.py schemas.py routers services
python -m pytest
```
