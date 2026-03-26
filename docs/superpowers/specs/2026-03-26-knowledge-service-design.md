# Knowledge Service Design Spec

## Overview

Brain(S2S 엔진)과 독립적으로 패키징되는 지식 관리 서비스. 콜센터 운영자가 상담 매뉴얼, FAQ, 시스템 프롬프트를 관리하고, Brain이 API로 지식을 조회하여 LLM 응답에 활용.

## 상품 구조

| 상품 | 역할 | 독립 배포 |
|------|------|-----------|
| **Brain** | S2S 엔진 (STT→LLM→TTS), 대화 맥락 관리, 감정 인식 | Yes (기본 프롬프트로 동작) |
| **Knowledge** | 지식 저장/검색, 문서 파싱, RAG, 관리 UI | Yes (검색 API + 관리 UI) |

## 요구사항

- **사용자**: 콜센터 운영자/관리자
- **지식 형태**: 문서 업로드(PDF/Word) + FAQ 쌍 + 시스템 프롬프트
- **벡터DB**: ChromaDB (경량, 파이썬 내장)
- **기술 스택**: FastAPI + Vue + ChromaDB
- **Brain 연동**: 사전 로딩 + 로컬 캐시 (실시간 지연 0ms)
- **대화 히스토리**: Brain이 소유 (Knowledge는 지식만 담당)
- **통화 기록**: Brain에서 로깅만 (장기 보관/분석은 향후 결정)

## 아키텍처

```
┌─────────────────────────────┐     ┌─────────────────────────────┐
│      Brain (S2S 엔진)        │     │   Knowledge (지식 서비스)     │
│                             │     │                             │
│  STT → LLM → TTS           │     │  관리 UI (Vue)              │
│  Turn Detection             │     │    - 문서 업로드/파싱/청킹    │
│  Emotion Recognition        │     │    - FAQ 편집               │
│  Barge-in                   │     │    - 프롬프트 편집기         │
│  SessionManager (대화 맥락)  │     │    - 검색 테스트/RAG 실험    │
│                             │     │                             │
│  ┌───────────────────────┐  │     │  Backend (FastAPI)          │
│  │  Knowledge Client     │  │     │    - Ingestion Pipeline     │
│  │  - 시작 시 config 로딩 │◄─┼─────┤    - Hybrid Search (RAG)    │
│  │  - 로컬 캐시          │  │     │    - Cross-Encoder Rerank   │
│  │  - webhook 수신       │◄─┼─────┤    - Brain API 엔드포인트    │
│  └───────────────────────┘  │     │    - Webhook 발행           │
│                             │     │                             │
│  포트: 3000                 │     │  Storage                    │
│  GPU: STT + TTS             │     │    - ChromaDB (벡터)        │
└─────────────────────────────┘     │    - SQLite (메타데이터)     │
                                    │    - 파일 스토리지 (원본)    │
                                    │                             │
                                    │  포트: 8100                 │
                                    │  GPU: 임베딩 (~1GB)         │
                                    └─────────────────────────────┘
```

## RAG 파이프라인

### Ingestion (문서 → 벡터화)

```
PDF/Word 업로드
  ↓
① 문서 파싱 (pypdf, python-docx)
  ↓
② Semantic Chunking (구조 인식)
   - 제목/소제목 기반 섹션 분리 (고정 크기 아님)
   - Parent-Child: 큰 청크(parent)로 컨텍스트 보존,
     작은 청크(child)로 정밀 검색
   - 매뉴얼 구조 계층 유지: "1장 > 1.1절 > 절차1"
  ↓
③ Contextual Retrieval (Anthropic 방식)
   - 각 청크 앞에 문서 전체 맥락 요약 자동 삽입
   - "이 청크는 [카드 분실 매뉴얼]의 [3. 긴급정지 절차]에 해당"
  ↓
④ Hybrid Embedding
   - Dense: BGE-M3 (한국어 포함 다국어, 1024-dim)
   - Sparse: BM25 인덱스 (한국어 형태소 분석 기반)
   - 둘 다 ChromaDB에 저장
```

### Retrieval (검색)

```
사용자 발화
  ↓
① Query Transformation
   - 구어체 → 검색 쿼리 변환
   - Multi-Query: 원본 + 변환 쿼리 2-3개로 검색 범위 확장
  ↓
② Hybrid Search
   - Dense (BGE-M3 cosine) + Sparse (BM25 형태소)
   - RRF (Reciprocal Rank Fusion)로 결과 병합
   - Top-20 후보 추출
  ↓
③ Cross-Encoder Reranking
   - BGE-Reranker로 쿼리-청크 쌍 정밀 점수
   - Top-3으로 축소
  ↓
④ Parent Context 확장
   - child 매칭 시 parent 청크 반환 (넓은 맥락)
  ↓
⑤ Brain에 전달
```

## Brain ↔ Knowledge API 계약

### Brain 시작 시 (사전 로딩)

```
GET /api/brain/config?brain_id={id}
→ {
    "system_prompt": "당신은 친절한 상담원...",
    "faq": [{"q": "카드 분실", "a": "즉시 정지..."},...],
    "metadata": {"updated_at": "2026-03-26T..."}
  }
```

### 실시간 RAG 검색 (선택적, 캐시 미스 시)

```
POST /api/brain/search
  {"query": "카드 분실 신고", "top_k": 3}
→ {"results": [{"text": "...", "source": "매뉴얼.pdf p.12", "score": 0.92},...]}
```

### Knowledge 변경 시 (webhook)

```
POST http://brain:3000/api/knowledge/refresh
  {"updated": ["prompt", "faq"], "timestamp": "..."}
```

## Brain 측 LLM 프롬프트 조립

```
시스템 프롬프트 (from Knowledge, 캐시)
+ 관련 지식 Top-3 (from Knowledge, 캐시)
+ 대화 히스토리 (Brain SessionManager)
+ [audio: energy=high, rate=fast, trend=rising]
+ 고객: {현재 발화}
```

## Knowledge 관리 UI 주요 화면

1. **문서 관리**: 업로드, 파싱 상태, 청크 미리보기, 삭제
2. **FAQ 관리**: 질문-답변 쌍 CRUD, 카테고리 분류
3. **프롬프트 편집기**: 시스템 프롬프트 작성/버전 관리
4. **검색 테스트**: 쿼리 입력 → RAG 결과 확인, 점수 표시
5. **향후: RAG 실험**: 파이프라인 A/B 비교, 청킹 전략 비교

## 프로젝트 구조

```
/home/jonghooy/work/knowledge-service/
  ├── backend/
  │   ├── main.py              # FastAPI app
  │   ├── api/
  │   │   ├── documents.py     # 문서 업로드/파싱/CRUD
  │   │   ├── faq.py           # FAQ CRUD
  │   │   ├── prompts.py       # 프롬프트 관리
  │   │   ├── search.py        # RAG 검색 API
  │   │   └── brain.py         # Brain 연동 API
  │   ├── ingestion/
  │   │   ├── parser.py        # PDF/Word 파싱
  │   │   ├── chunker.py       # Semantic chunking + parent-child
  │   │   ├── contextual.py    # Contextual retrieval (맥락 삽입)
  │   │   └── embedder.py      # BGE-M3 + BM25 하이브리드
  │   ├── retrieval/
  │   │   ├── hybrid_search.py # Dense + Sparse + RRF
  │   │   ├── reranker.py      # Cross-encoder reranking
  │   │   └── query_transform.py # 쿼리 변환/확장
  │   ├── db/
  │   │   ├── chroma.py        # ChromaDB 래퍼
  │   │   └── sqlite.py        # 메타데이터 DB
  │   └── config.py
  ├── frontend/
  │   ├── src/
  │   │   ├── views/           # 문서/FAQ/프롬프트/검색 페이지
  │   │   ├── components/      # 공통 컴포넌트
  │   │   └── api/             # Backend API 클라이언트
  │   └── package.json
  ├── storage/
  │   ├── documents/           # 원본 파일
  │   ├── chroma/              # ChromaDB 데이터
  │   └── knowledge.db         # SQLite
  └── requirements.txt
```

## 기술 스택

| 컴포넌트 | 기술 | 비고 |
|----------|------|------|
| Backend | FastAPI | REST API + WebSocket |
| Frontend | Vue 3 + Vite | 관리 UI |
| 벡터DB | ChromaDB | 경량, 내장 |
| 메타데이터DB | SQLite | 문서/FAQ 메타 |
| 임베딩 | BGE-M3 | 한국어 다국어 1024-dim |
| Reranker | BGE-Reranker | Cross-encoder |
| 문서 파싱 | pypdf + python-docx | PDF/Word |
| 한국어 형태소 | Kiwi | BM25 토크나이저 |
| 청킹 | LangChain TextSplitters | Semantic + recursive |

## 향후 확장

- OCR 파이프라인 (스캔 PDF, 이미지 문서)
- HWP/Excel/PPT 지원
- RAG 전략 A/B 실험 프레임워크
- 통화 기록 분석 서비스 (3번째 상품)
- Multi-Brain 지원 (하나의 Knowledge → 여러 Brain)
