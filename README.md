# AI Model Serving Server

FastAPI 기반 공모전 MVP용 AI 서빙 서버입니다. Data Pipeline에서 구축한 Qdrant vector store를 검색하고 OpenAI API로 RAG 채팅, 대화 요약, STT, TTS를 제공합니다.

## 환경 설정

```bash
cp .env.example .env
```

`.env`에 실제 값을 채웁니다. `.env`와 실제 API key는 커밋하지 않습니다.

## 환경변수

| 이름 | 설명 |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI API key |
| `OPENAI_MODEL` | 채팅/요약 모델, 기본 `gpt-4o-mini` |
| `OPENAI_STT_MODEL` | STT 모델, 기본 `gpt-4o-mini-transcribe` |
| `OPENAI_TTS_MODEL` | TTS 모델, 기본 `gpt-4o-mini-tts` |
| `QDRANT_URL` | Qdrant URL, 로컬 기본 `http://localhost:6333` |
| `QDRANT_API_KEY` | Qdrant API key, 로컬에서는 비워도 됨 |
| `QDRANT_COLLECTION` | Qdrant collection, `memory_box_contents` |
| `FASTEMBED_MODEL` | Query embedding 모델 |
| `RAG_TOP_K` | 검색 결과 개수 |
| `RAG_SCORE_THRESHOLD` | 검색 결과 최소 점수 |
| `SUMMARY_THRESHOLD` | 채팅 요약 생성 기준 message 개수 |

Data Pipeline과 반드시 맞춰야 하는 값:

```env
QDRANT_COLLECTION=memory_box_contents
FASTEMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 실행

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Docker 빌드

```bash
docker build -t ai-serving-server:local .
```

`.dockerignore`에서 `.env`, `.env.*`, 가상환경, 캐시, 음성 산출물, Git metadata를 제외합니다. 실제 `OPENAI_API_KEY`는 이미지에 넣지 말고 컨테이너 실행 시 환경변수 또는 `--env-file .env`로 주입합니다.

## Docker 실행

로컬 Qdrant에 접근하지 않는 `/health`만 확인할 때:

```bash
docker run --rm -p 8000:8000 \
  --env-file .env \
  --name ai-serving-server-local \
  ai-serving-server:local
```

호스트에서 실행 중인 Qdrant(`http://localhost:6333`)를 컨테이너에서 사용해야 할 때:

```bash
docker run --rm -p 8000:8000 \
  --add-host=host.docker.internal:host-gateway \
  -e QDRANT_URL=http://host.docker.internal:6333 \
  --env-file .env \
  --name ai-serving-server-local \
  ai-serving-server:local
```

Linux/WSL에서 `host.docker.internal` 접근이 되지 않으면 host network로 실행합니다.

```bash
docker run --rm --network host \
  --env-file .env \
  --name ai-serving-server-local \
  ai-serving-server:local
```

Qdrant를 함께 띄우는 로컬 개발용 실행:

```bash
docker compose up --build
```

Docker Hub에 올릴 때는 예를 들어 아래처럼 태그를 붙인 뒤, 사용자가 직접 로그인과 push를 수행합니다.

```bash
docker tag ai-serving-server:local <dockerhub-username>/ai-serving-server:latest
docker push <dockerhub-username>/ai-serving-server:latest
```

## Health 테스트

```bash
curl http://localhost:8000/health
```

예상 응답:

```json
{"status":"ok"}
```

## Chat 테스트

```bash
curl -X POST http://localhost:8000/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "query":"1970년대 서울역",
    "messages":[],
    "summary":null
  }'
```

검색 결과가 없거나 context가 비어 있으면 OpenAI를 호출하지 않고 아래 fallback을 반환합니다.

```json
{
  "answer": "현재 저장된 자료에서 관련 내용을 찾지 못했습니다. 다른 키워드로 질문해 주세요.",
  "summary": null
}
```

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
