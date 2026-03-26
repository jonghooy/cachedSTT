# xVisor S2S — 한국어 음성-to-음성 콜봇 시스템

## 1. 개요

고객의 음성을 실시간으로 인식하고, LLM이 감정에 맞는 답변을 생성하며, 자연스러운 음성으로 응답하는 **엔드-투-엔드 Speech-to-Speech 콜봇** 시스템.

두 개의 독립 상품으로 구성:
- **Brain** — S2S 엔진 (STT → Turn Detection → LLM → TTS)
- **Knowledge** — 지식 관리 서비스 (RAG, FAQ, 프롬프트, 문서 관리)

---

## 2. 시스템 아키텍처

```
┌─────────────────────────────────────────────────┐
│                Brain (S2S 엔진)                   │
│                port 3000                         │
│                                                 │
│  ┌───────────┐  ┌──────────────┐  ┌───────────┐ │
│  │    STT    │→ │ Turn Detect  │→ │  Emotion   │ │
│  │  NeMo     │  │ Kiwi 형태소  │  │  Prosody   │ │
│  │ FastConf  │  │ + User Rules │  │  + LLM     │ │
│  └───────────┘  └──────────────┘  └───────────┘ │
│        │              │                │         │
│        ▼              ▼                ▼         │
│  ┌─────────────────────────────────────────────┐ │
│  │           S2S Pipeline                       │ │
│  │  RAG 검색 → LLM 스트리밍 → TTS 절 단위 합성  │ │
│  │  (Knowledge API)  (Qwen3.5 9B)  (StyleTTS2) │ │
│  └─────────────────────────────────────────────┘ │
│        │                                         │
│  ┌───────────┐  ┌──────────────┐                │
│  │  Barge-in │  │ Knowledge    │                │
│  │  끼어들기  │  │ Client       │←── webhook ──┐ │
│  └───────────┘  └──────────────┘              │ │
│                                               │ │
│  WebSocket (클라이언트 ↔ 서버)                  │ │
└─────────────────────────────────────────────────┘
         │                                      │
         │ HTTP API                             │
         ▼                                      │
┌─────────────────────────────────────────────────┐
│            Knowledge (지식 서비스)                 │
│            port 8100 (API) + 5173 (UI)           │
│                                                 │
│  ┌──────────────────────────────────────┐       │
│  │          RAG Pipeline                 │       │
│  │  Query Transform → Multi-Query        │       │
│  │  → Hybrid Search (BGE-M3 + BM25)     │       │
│  │  → RRF Fusion → Reranker             │       │
│  │  → Parent Context Expansion           │       │
│  └──────────────────────────────────────┘       │
│                                                 │
│  ┌────────┐ ┌─────┐ ┌──────────┐ ┌──────────┐ │
│  │ 문서    │ │ FAQ │ │ 프롬프트  │ │ 동의어    │ │
│  │ 업로드  │ │ CRUD│ │ 버전관리  │ │ 관리     │ │
│  └────────┘ └─────┘ └──────────┘ └──────────┘ │
│                                                 │
│  ChromaDB (벡터) + SQLite (메타) + Vue 3 (UI)   │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│            vLLM (LLM 서빙)                       │
│            port 8000                             │
│  Qwen3.5-9B BF16 | prefix caching               │
└─────────────────────────────────────────────────┘
```

---

## 3. 핵심 모듈 상세

### 3.1 STT (음성 인식)

| 항목 | 값 |
|------|-----|
| 모델 | Nemotron Speech ASR 0.6B (FastConformer RNN-T) |
| 파라미터 | 618M |
| 인코더 | 24 layers, d_model=1024 |
| 학습 데이터 | 763.8시간 한국어 |
| WER | 16.3% |
| 스트리밍 | Cache-aware, chunk 150ms |
| RTF | 0.0098 (102x 실시간) |
| GPU 메모리 | 2.4GB |

### 3.2 Turn Detection (발화 종결 감지)

| 항목 | 값 |
|------|-----|
| 방식 | Kiwi 형태소 분석 + 사용자 정의 규칙 |
| 종결어미 감지 | EF, IC, EC+요 패턴 |
| 동적 임계값 | EOU score → silence threshold (0.15s~2.5s) |
| 사용자 규칙 | REST API + Web UI로 관리 (현재 12개 규칙) |
| 종결어미 시 지연 | **150ms** (silence 0.15초) |

### 3.3 감정 인식 (Emotion Recognition)

| 항목 | 값 |
|------|-----|
| 방식 | 프로소디 메타데이터 + LLM 판단 (추가 비용 0) |
| 감정 분류 | 5단계: anger, anxiety, neutral, satisfaction, joy |
| 프로소디 특징 | energy level, energy trend, speech rate |
| 음성 특징 | RMS 에너지 히스토리 → 발화 구간 통계 |
| 텍스트 특징 | LLM이 [EMOTION:xxx] 태그로 판단 |
| 감정별 응대 | 시스템 프롬프트에 감정별 지침 포함 |

### 3.4 LLM (응답 생성)

| 항목 | 값 |
|------|-----|
| 모델 | Qwen3.5-9B |
| 서빙 | vLLM 0.18.0 (별도 프로세스, port 8000) |
| 정밀도 | BF16 |
| max_model_len | 2048 |
| Thinking 모드 | 비활성화 (enable_thinking: false) |
| Prefix caching | 활성화 (시스템 프롬프트 캐싱) |
| GPU 메모리 | ~24GB (0.75 utilization) |
| TTFT | ~50-90ms |
| max_tokens | 80 (1-2문장 제한) |

### 3.5 TTS (음성 합성)

| 항목 | 값 |
|------|-----|
| 모델 | StyleTTS2 (한국어 KB 60h fine-tune) |
| 출력 | 24kHz mono PCM |
| Diffusion steps | 5 |
| 합성 속도 | **~40ms/문장** |
| GPU 메모리 | ~0.6GB |
| 스트리밍 | 절(clause) 단위 flush (쉼표, 마침표에서 끊기) |

### 3.6 Barge-in (끼어들기)

| 항목 | 값 |
|------|-----|
| 감지 방식 | STT partial 텍스트 기반 (2글자 이상 인식 시) |
| 동작 | TTS 즉시 중단 + LLM 생성 취소 + STT 계속 |
| 구현 | asyncio.Event cancel + 클라이언트 AudioSource.stop() |
| 쿨다운 | 2초 (연속 트리거 방지) |
| 소음 구분 | RMS 에너지가 아닌 실제 텍스트 인식으로 판단 |

### 3.7 Knowledge Service (지식 관리)

| 항목 | 값 |
|------|-----|
| 백엔드 | FastAPI (port 8100) |
| 프론트엔드 | Vue 3 + Vite (port 5173) |
| 벡터DB | ChromaDB |
| 메타데이터DB | SQLite |
| 임베딩 모델 | BGE-M3 (1024-dim, **CPU 실행** — GPU 경합 방지) |
| 문서 파싱 | PDF (pypdf), Word (python-docx) |
| 청킹 | Semantic parent-child (구조 인식, ~300자/child) |
| 검색 | Hybrid (Dense BGE-M3 + Sparse BM25 Kiwi + RRF) |
| 동의어 | DB 관리 + UI 편집 (양방향 확장) |
| Brain 연동 | 사전 로딩 + webhook 캐시 갱신 |
| 테스트 | 63 unit tests |

---

## 4. 성능 벤치마크 (RTX 5090 32GB)

### 4.1 E2E 지연 시간

| 구간 | 시간 | 설명 |
|------|------|------|
| Turn Detection (종결어미) | ~2-150ms | 종결어미 0.15s, 비종결 1.5s |
| RAG 검색 (Knowledge) | ~20ms | CPU 임베딩, Hybrid search |
| LLM TTFT | ~50-90ms | Qwen3.5-9B, prefix cached |
| LLM 첫 문장 생성 | ~200-300ms | 토큰 스트리밍 |
| [EMOTION] 태그 파싱 | ~50ms | 25자 버퍼링 |
| TTS 합성 | ~40ms | StyleTTS2 |
| 클라이언트 전송 | ~20ms | WebSocket + base64 |
| **총 First Audio** | **~420-560ms** | 종결어미 발화 기준 |

### 4.2 체감 지연 (Turn Detection 포함)

| 시나리오 | 체감 지연 | 평가 |
|----------|----------|------|
| 종결어미 ("~했습니다") | ~560ms | 매우 빠름 |
| 종결어미 + 규칙 ("~거든요") | ~600ms | 빠름 |
| 비종결어미 ("~는데") | ~2000ms | Turn Detection 대기 |
| 실제 VoIP 전화 (예상) | ~800-1000ms | 자연스러운 수준 |

### 4.3 GPU 메모리 사용

| 컴포넌트 | 메모리 |
|----------|--------|
| vLLM (Qwen3.5-9B BF16) | ~24GB |
| STT (NeMo FastConformer) | ~2.4GB |
| TTS (StyleTTS2) | ~0.6GB |
| **합계** | **~27GB / 32GB** |
| Knowledge (BGE-M3) | CPU (GPU 미사용) |

---

## 5. 데이터 흐름

### 5.1 음성 입력 → 응답 음성 출력

```
1. 클라이언트 마이크 → PCM int16 (150ms 청크) → WebSocket
2. 서버: float32 변환 → audio_buffer 누적 → mel 변환
3. conformer_stream_step() → RNN-T 디코딩 → partial 텍스트
4. Turn Detection: 형태소 분석 → EOU score → silence threshold
5. Endpoint 발생 → Final 텍스트 확정
6. 프로소디 계산: energy, trend, speech_rate
7. Knowledge RAG 검색: query → hybrid search → top-3 문서 청크
8. LLM 프롬프트 조립:
   시스템 프롬프트 (Knowledge)
   + [참고 문서] RAG 결과
   + [audio: energy=high, rate=fast, trend=rising]
   + 고객: {발화 텍스트}
9. LLM 스트리밍 생성:
   [EMOTION:anxiety] → 감정 태그 파싱 (TTS에서 제외)
   "걱정 마세요," → TTS 합성 → 클라이언트 재생
   "바로 도와드리겠습니다." → TTS 합성 → 이어서 재생
10. 클라이언트: base64 디코드 → AudioContext → 스피커 재생
```

### 5.2 Barge-in (끼어들기)

```
1. TTS 재생 중 사용자가 말하기 시작
2. STT partial 텍스트 2글자 이상 인식
3. 클라이언트: TTS 즉시 중단 + {"type":"barge_in"} 전송
4. 서버: cancel_event.set() → LLM 생성 중단
5. STT 계속 인식 → 새 Final → 새 S2S 응답
```

### 5.3 Knowledge 연동

```
Brain 시작 시:
  GET /api/brain/config → 시스템 프롬프트 + FAQ 캐시

매 발화 시:
  POST /api/brain/search → RAG 검색 결과 (top-3)
  → LLM 프롬프트에 [참고 문서]로 포함

Knowledge 변경 시:
  POST brain:3000/api/knowledge/refresh → Brain 캐시 갱신
  (또는 Brain UI RESTART 버튼)
```

---

## 6. 프로젝트 구조

### 6.1 Brain (`/home/jonghooy/work/cachedSTT/`)

```
realtime_demo/
├── server.py              # FastAPI + WebSocket 서버 (메인)
│                           # STT 스트리밍, Turn Detection, S2S 파이프라인 실행
│                           # Barge-in 처리, Knowledge webhook
├── s2s_pipeline.py        # S2S 파이프라인 (LLM + TTS 오케스트레이션)
│                           # RAG 검색 → 감정 태그 파싱 → 절 단위 TTS 스트리밍
├── turn_detector.py       # Turn Detection (Kiwi 형태소 + 사용자 규칙)
│                           # EOU scoring, dynamic silence threshold
├── knowledge_client.py    # Knowledge Service HTTP 클라이언트
│                           # config 사전 로딩 + RAG 검색 + FAQ 캐시
├── static/index.html      # 웹 UI (마이크, STT, S2S 응답, 감정 배지, Turn 패널)
├── rules/turn_rules.json  # 사용자 정의 Turn Detection 규칙 (JSON)
└── tests/
    └── test_turn_detector.py  # 14 unit tests
```

### 6.2 Knowledge (`/home/jonghooy/work/knowledge-service/`)

```
backend/
├── main.py                # FastAPI app + 싱글톤 초기화 (embedder, searcher, reranker)
├── config.py              # 경로, 모델명 설정
├── shared.py              # 싱글톤 인스턴스 (embedder, searcher, reranker, chroma)
├── api/
│   ├── documents.py       # 문서 업로드/파싱/청킹/임베딩 (백그라운드)
│   ├── faq.py             # FAQ CRUD
│   ├── prompts.py         # 시스템 프롬프트 버전관리 + 활성화
│   ├── synonyms.py        # 동의어 CRUD (검색 확장용)
│   ├── search.py          # RAG 검색 (full pipeline + fallback)
│   └── brain.py           # Brain 연동 API (config, search)
├── ingestion/
│   ├── parser.py          # PDF/Word → 텍스트 + 섹션 구조
│   ├── chunker.py         # Semantic parent-child 청킹
│   ├── contextual.py      # Contextual retrieval prefix
│   └── embedder.py        # BGE-M3 dense (CPU) + Kiwi BM25 sparse
├── retrieval/
│   ├── hybrid_search.py   # Dense + Sparse + RRF fusion
│   ├── reranker.py        # BGE cross-encoder reranking
│   └── query_transform.py # 구어체→검색어 변환 + multi-query
├── db/
│   ├── sqlite.py          # SQLite CRUD (documents, faq, prompts, synonyms)
│   └── chroma.py          # ChromaDB wrapper
└── tests/                 # 63 unit tests

frontend/                  # Vue 3 + Vite (port 5173)
├── src/views/
│   ├── DocumentsView.vue  # 문서 업로드/목록/청크 미리보기
│   ├── FaqView.vue        # FAQ 편집
│   ├── PromptsView.vue    # 시스템 프롬프트 편집기 + 버전관리
│   ├── SynonymsView.vue   # 동의어 관리
│   └── SearchView.vue     # RAG 검색 테스트
└── src/api/client.js      # Backend API 클라이언트
```

---

## 7. 기술 스택

| 컴포넌트 | 기술 | 버전 |
|----------|------|------|
| STT 모델 | NeMo FastConformer RNN-T | NeMo 2.6.1 |
| LLM | Qwen3.5-9B | BF16 |
| LLM 서빙 | vLLM | 0.18.0 |
| TTS | StyleTTS2 (5Hyeons vocos) | 한국어 KB 60h |
| 형태소 분석 | Kiwi (kiwipiepy) | |
| 임베딩 | BGE-M3 (BAAI/bge-m3) | 1024-dim, CPU |
| 벡터DB | ChromaDB | 1.5.5 |
| Backend | FastAPI + uvicorn | |
| Frontend | Vue 3 + Vite | |
| GPU | NVIDIA RTX 5090 | 32GB |
| CUDA | 13.0 | Driver 580 |
| Python | 3.10 (nemo-asr), 3.11 (vllm) | |
| Conda | nemo-asr, vllm_serving | 2 환경 |

---

## 8. 실행 방법

### 8.1 전체 시스템 시작 (3개 서비스)

```bash
# 1. vLLM (LLM 서빙) — 가장 먼저 시작 (GPU 메모리 확보)
conda activate vllm_serving
CUDA_VISIBLE_DEVICES=0 vllm serve /mnt/usb/models/Qwen3.5-9B \
    --host 0.0.0.0 --port 8000 \
    --dtype bfloat16 --max-model-len 2048 \
    --gpu-memory-utilization 0.75 \
    --trust-remote-code --enforce-eager \
    --enable-prefix-caching --max-num-seqs 4 &

# 2. Knowledge Service (지식 관리)
cd /home/jonghooy/work/knowledge-service
conda activate nemo-asr
PYTHONPATH=backend python -m uvicorn backend.main:app --host 0.0.0.0 --port 8100 &

# 3. Knowledge Frontend (관리 UI)
cd /home/jonghooy/work/knowledge-service/frontend
npx vite --host 0.0.0.0 --port 5173 &

# 4. Brain S2S (STT + TTS + 파이프라인)
cd /home/jonghooy/work/cachedSTT
conda activate nemo-asr
CUDA_VISIBLE_DEVICES=0 python realtime_demo/server.py --s2s --port 3000
```

### 8.2 접속 URL

| 서비스 | URL | 용도 |
|--------|-----|------|
| Brain (S2S 데모) | http://192.168.0.62:3000 | 음성 대화 테스트 |
| Knowledge 관리 UI | http://192.168.0.62:5173 | 문서/FAQ/프롬프트 관리 |
| Knowledge API | http://192.168.0.62:8100 | REST API |
| vLLM API | http://192.168.0.62:8000 | LLM OpenAI-compatible API |

### 8.3 전체 시스템 종료

```bash
pkill -f "realtime_demo/server.py"   # Brain
pkill -f "uvicorn backend"           # Knowledge
pkill -f "vllm serve"               # vLLM
pkill -f "vite"                     # Frontend
```

### 8.4 Brain만 재시작 (코드 변경 후)

```bash
pkill -f "realtime_demo/server.py"
cd /home/jonghooy/work/cachedSTT
conda activate nemo-asr
CUDA_VISIBLE_DEVICES=0 python realtime_demo/server.py --s2s --port 3000
```

---

## 9. Brain ↔ Knowledge API 계약

### 시작 시 설정 로딩

```
GET /api/brain/config?brain_id=default
→ {
    "system_prompt": "당신은 친절한 상담원...",
    "faq": [{"q": "카드 분실", "a": "정지 처리..."}],
    "metadata": {"updated_at": "..."}
  }
```

### 실시간 RAG 검색

```
POST /api/brain/search
  {"query": "카드 분실 신고", "top_k": 3}
→ {"results": [
    {"text": "...", "source": "매뉴얼.docx", "section": "1. 카드 분실", "score": 0.92, "parent_text": "..."},
    ...
  ]}
```

### Knowledge 변경 시 캐시 갱신

```
POST brain:3000/api/knowledge/refresh
→ {"status": "refreshed"}
```

---

## 10. 주요 해결 이슈

### 10.1 GPU 경합 문제
- **문제**: Knowledge BGE-M3 임베딩(GPU)이 Brain TTS와 경합 → TTFT 5.6초
- **해결**: BGE-M3를 **CPU로 전환** → TTFT 90ms 복원

### 10.2 Turn Detection 지연
- **문제**: 무음 구간에서 STT 추론 중단 → blank_count 미증가 → 0.8-1.1초 대기
- **해결**: 무음 구간에서도 추론 계속 + 종결어미 시 silence 0.15초에 즉시 endpoint

### 10.3 Qwen3.5 Thinking 모드
- **문제**: 기본 thinking 모드 활성화 → 영어 reasoning 출력
- **해결**: `chat_template_kwargs: {enable_thinking: false}`

### 10.4 감정 태그 대기 지연
- **문제**: [EMOTION:xxx] 완성까지 40자 대기 → 200ms 낭비
- **해결**: 25자로 축소 (최대 태그 길이 24자)

### 10.5 Barge-in 오탐
- **문제**: RMS 에너지 기반 → 탁자 소리에도 끼어들기 발생
- **해결**: **STT partial 텍스트 기반**으로 전환 (실제 음성 인식만 반응)

### 10.6 StyleTTS2 모델 누락
- **문제**: styletts2_korean 디렉토리 및 모델 파일 없음
- **해결**: 5Hyeons/StyleTTS2 HuggingFace에서 다운로드 + 의존성 설치

### 10.7 NeMo ↔ transformers 버전 충돌
- **문제**: nemo-asr의 transformers 4.53이 Qwen3.5 미지원
- **해결**: vLLM을 **별도 프로세스**(vllm_serving env)로 분리, HTTP API 통신

---

## 11. 향후 계획

### 11.1 속도 최적화
- [ ] LoRA Fine-tune: 콜센터 도메인 특화 → 프롬프트 축소 → TTFT 개선
- [ ] 단일 패스 프롬프트: Intent + 감정 + 답변을 1회 LLM 호출로 통합
- [ ] Speculative decoding: draft 모델로 토큰 생성 가속

### 11.2 Knowledge 고도화
- [ ] OCR 파이프라인 (스캔 PDF, 이미지)
- [ ] HWP/Excel/PPT 지원
- [ ] Contextual Retrieval 강화 (LLM 요약 기반)
- [ ] RAG A/B 실험 프레임워크

### 11.3 상품화
- [ ] Docker Compose 패키징 (단일 실행)
- [x] 대화 히스토리 관리 (멀티턴) — Brain 세션별, 최근 6턴, 글자 수 기반 truncation
- [ ] 통화 기록 분석 서비스 (3번째 상품)
- [ ] Multi-Brain 지원 (하나의 Knowledge → 여러 Brain)

### 11.4 모델 진화 (xVisor Phase 2/3)
- [ ] Speech Projector: STT 인코더 임베딩 → LLM 직접 주입 (감정/억양 보존)
- [ ] Full Audio-to-Audio: 텍스트 병목 완전 제거

---

## 12. Key Paths

| 항목 | 경로 |
|------|------|
| Brain 프로젝트 | `/home/jonghooy/work/cachedSTT/` |
| Knowledge 프로젝트 | `/home/jonghooy/work/knowledge-service/` |
| Brain 서버 | `realtime_demo/server.py` |
| S2S 파이프라인 | `realtime_demo/s2s_pipeline.py` |
| Turn Detector | `realtime_demo/turn_detector.py` |
| Knowledge Client | `realtime_demo/knowledge_client.py` |
| Brain UI | `realtime_demo/static/index.html` |
| Turn Rules | `realtime_demo/rules/turn_rules.json` |
| STT 모델 | `/home/jonghooy/work/timbel-asr-pilot/pretrained_models/` |
| TTS 모델 | `/home/jonghooy/work/zhisper/styletts2_korean/Models/KB_60h/` |
| LLM 모델 (BF16) | `/mnt/usb/models/Qwen3.5-9B/` |
| LLM 모델 (AWQ) | `/mnt/usb/models/Qwen3.5-9B-AWQ/` |
| Knowledge DB | `knowledge-service/storage/knowledge.db` |
| ChromaDB | `knowledge-service/storage/chroma/` |
| 문서 저장소 | `knowledge-service/storage/documents/` |
