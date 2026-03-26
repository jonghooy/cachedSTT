# Scenario Builder + Dialogue Engine Design Spec

**Date:** 2026-03-27
**Status:** Draft
**Author:** AI-assisted

## Overview

기업용 콜봇을 위한 시나리오 빌더 UI + 대화 엔진 설계. AI(Claude)가 시나리오를 생성/검증하고, 사람은 비주얼 에디터에서 검토/수정만 한다. Brain(cachedSTT)에 Dialogue Engine을 추가하여 시나리오 기반 구조화된 대화와 LLM 자유 응답을 하이브리드로 처리한다.

## Design Decisions

| 결정 사항 | 선택 | 이유 |
|-----------|------|------|
| 대화 범위 | 하이브리드 | 시나리오 우선, 나머지 LLM. 컴플라이언스 필수 업무는 시나리오가 제어, 나머지는 LLM 자유 응답 |
| 사용자 모델 | AI 생성 + 사람 검토 | 프론티어 AI가 시나리오 제작/검증, 사람은 최종 승인만 |
| 입력 방식 | 복합 | 자연어 지시 + 문서 업로드 + 상담 로그 분석 조합 |
| 액션 타입 | 복합 | API 호출 + 상담원 전환 + 종결 처리 체이닝 |
| UI 위치 | Knowledge Service 통합 | 기존 "봇 설정 포털" 패턴 확장. 별도 서비스 추가 불필요 |
| 시나리오 모델 | Graph + LLM 어댑터 | 노드 그래프 기반 + 모호한 분기/추출은 LLM 위임 |
| Intent 감지 | 임베딩 + LLM 판정 | BGE-M3 유사도 빠른 필터 + Qwen 최종 판정. AI가 트리거 예시 자동 생성 |
| 상태 저장 | Brain 인메모리 | WebSocket 끊기면 통화 끊김. 영속화 불필요. 다중 인스턴스 시 Redis 전환 |

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────┐
│              Knowledge Service (Vue 3)               │
│                 Port 5173 / 8100                     │
│                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ 문서/RAG │  │ FAQ 관리 │  │ 시나리오 빌더     │  │
│  │ 관리     │  │          │  │ ├─ 비주얼 에디터  │  │
│  └──────────┘  └──────────┘  │ ├─ AI 생성       │  │
│  ┌──────────┐  ┌──────────┐  │ └─ 테스트/검증   │  │
│  │ 프롬프트 │  │ 동의어   │  └──────────────────┘  │
│  │ 관리     │  │ 관리     │                         │
│  └──────────┘  └──────────┘                         │
│                                                     │
│  Storage: SQLite (scenarios) + ChromaDB (triggers)  │
└──────────────────┬──────────────────────────────────┘
                   │ REST API
┌──────────────────▼──────────────────────────────────┐
│                  Brain (cachedSTT)                   │
│                    Port 3000                         │
│                                                     │
│  STT → Turn Detection → Dialogue Engine → LLM → TTS│
│                          │                          │
│                  ┌───────┴───────┐                  │
│                  │ Intent Matcher│                  │
│                  │ Graph Walker  │                  │
│                  │ Slot Manager  │                  │
│                  │ Action Runner │                  │
│                  └───────────────┘                  │
│                                                     │
│  StreamingSession += scenario_state, dialogue_mode  │
└─────────────────────────────────────────────────────┘
```

### Roles

- **Knowledge Service (정의):** 시나리오 CRUD, AI 생성/검증, 트리거 임베딩, 비주얼 에디터, 실행 로그 수집
- **Brain (실행):** 시나리오 캐시, Intent 매칭, 그래프 순회, 슬롯 관리, 액션 실행

### Runtime Flow

```
고객 발화 → STT 텍스트

  dialogue_mode == "freeform"?
  ├─ YES → Intent Matcher
  │         ├─ 매칭 → scenario 모드 진입 → 노드 실행
  │         └─ 미매칭 → 기존 S2S 파이프라인 (LLM 자유 응답)
  │
  └─ NO (scenario 모드)
     → Graph Walker: 현재 노드 실행
        ├─ speak → TTS 출력, 자동 다음 노드
        ├─ slot_collect → 슬롯 추출 (규칙/LLM), 실패 시 재질문
        ├─ condition → 조건 평가 (규칙/LLM) → 분기
        ├─ api_call → HTTP 호출 → 결과 저장
        ├─ rag_search → Knowledge 검색 → 컨텍스트 주입
        ├─ llm_response → Qwen에게 위임
        ├─ confirm → 슬롯 값 확인 (yes/no)
        ├─ transfer → 상담원 전환
        └─ end → 종결, freeform 복귀
```

## Data Model

### Scenario Top-Level Schema

```json
{
  "id": "card-lost-report",
  "name": "카드 분실 신고",
  "description": "고객 카드 분실 시 본인확인 후 카드 정지 처리",
  "version": 3,
  "status": "active",
  "priority": 10,
  "created_by": "ai",
  "source": {
    "type": "natural_language",
    "input": "카드 분실 신고 시나리오 만들어줘..."
  },
  "triggers": {
    "examples": [
      "카드를 잃어버렸어요",
      "카드 분실 신고 하려고요",
      "카드가 없어졌는데요",
      "신용카드를 도난당했어요"
    ],
    "description": "고객이 카드 분실/도난을 신고하려는 의도"
  },
  "schema_version": 1,
  "slots": {
    "card_number": { "type": "string", "required": true, "extract": "regex", "pattern": "\\d{4}[- ]?\\d{4}[- ]?\\d{4}[- ]?\\d{4}", "prompt": "카드번호 16자리를 말씀해주세요" },
    "loss_type": { "type": "enum", "required": true, "extract": "llm", "values": ["분실", "도난"], "prompt": "분실이신가요, 도난이신가요?" },
    "reissue": { "type": "boolean", "required": false, "extract": "llm", "prompt": "카드 재발급을 원하시나요?" }
  },
  "nodes": [],
  "edges": [],
  "metadata": {
    "created_at": "2026-03-27T10:00:00Z",
    "updated_at": "2026-03-27T12:30:00Z",
    "test_coverage": 0.92,       // float 0-1. 배포 시 시뮬레이션으로 산출. (테스트된 노드 / 전체 노드). 0.9 이상이면 배포 허용.
    "execution_count": 0,
    "avg_completion_rate": null
  }
}
```

### Node Types (9)

**1. Speak** — 봇 발화 (TTS 출력)
```json
{ "id": "n1", "type": "speak", "text": "카드 분실 신고 도와드리겠습니다.", "emotion": "empathy" }
```

**2. Slot Collect** — 슬롯 수집 (사용자 응답 대기)
```json
{ "id": "n2", "type": "slot_collect", "target_slot": "card_number", "max_retries": 3, "retry_prompt": "다시 한번 말씀해주시겠어요?", "fail_action": "transfer" }
```

슬롯의 `extract` 방식: `"llm"` (LLM 추출), `"regex"` (정규식 매칭, `pattern` 필드 필요). 카드번호/전화번호 등 정형 데이터는 regex가 빠르고 정확.

**3. Condition** — 조건 분기 (규칙 or LLM)
```json
{ "id": "n3", "type": "condition", "mode": "rule", "rule": { "field": "slots.loss_type", "op": "eq", "value": "도난" } }
```

Rule 모드는 안전한 표현식 평가기를 사용한다 (eval 금지). 지원 연산자: `eq`, `neq`, `contains`, `gt`, `lt`, `exists`, `in`. 복합 조건은 `{"and": [...]}`, `{"or": [...]}` 로 표현.

**4. API Call** — 외부 API 호출
```json
{ "id": "n5", "type": "api_call", "method": "POST", "url": "https://internal-api/card/block", "body": { "card_no": "{{slots.card_number}}" }, "result_var": "block_result", "timeout_ms": 5000, "on_error": "n_error" }
```

**5. Transfer** — 상담원 전환
```json
{ "id": "n6", "type": "transfer", "reason": "card_theft_report", "message": "전문 상담원에게 연결해드리겠습니다.", "transfer_data": { "slots": "all" } }
```

**6. End** — 시나리오 종결
```json
{ "id": "n7", "type": "end", "message": "카드 정지 처리 완료되었습니다.", "disposition": "resolved", "tags": ["card", "lost"] }
```

**7. LLM Response** — LLM 자유 응답 위임
```json
{ "id": "n8", "type": "llm_response", "instruction": "카드 재발급 절차를 안내해주세요", "context_slots": ["card_number", "loss_type"], "max_tokens": 120 }
```

**8. Confirm** — 슬롯 값 확인
```json
{ "id": "n9", "type": "confirm", "template": "카드번호 {{slots.card_number}}, {{slots.loss_type}} 맞으시죠?" }
```

**9. RAG Search** — Knowledge Service 검색
```json
{ "id": "n10", "type": "rag_search", "query_template": "{{slots.loss_type}} 카드 처리 절차", "top_k": 3, "result_var": "rag_context", "inject_to_next_llm": true }
```

### Edge Schema

모든 라우팅은 edges 배열로 통합한다. 노드 내부에 분기 참조를 두지 않는다.

```json
{ "from": "n1", "to": "n2" }
{ "from": "n3", "to": "n4_police", "label": "true" }
{ "from": "n3", "to": "n4_normal", "label": "false" }
{ "from": "n9", "to": "n5", "label": "yes" }
{ "from": "n9", "to": "n2", "label": "no" }
```

- 기본 edge: `label` 없음 (선형 흐름)
- Condition edge: `label`이 분기 결과값 ("true"/"false" 또는 커스텀)
- Confirm edge: `label`이 "yes" 또는 "no"
- 비주얼 에디터에서 edge = 화살표로 일관되게 렌더링

### DB Schema (Knowledge Service)

- **scenarios** (SQLite): `id, name, description, schema_version, graph_json, version, status, priority, created_by, source_json, triggers_json, slots_json, metadata_json, created_at, updated_at`

**Schema Versioning:** `schema_version`은 시나리오 JSON 구조 버전 (현재 1). 엔진 업그레이드 시 schema_version이 달라지면 모든 active 시나리오를 재검증 후 재배포 필요. `version`은 시나리오 콘텐츠 수정 버전.
- **trigger_examples** (ChromaDB): `id: "{scenario_id}_{idx}", embedding: BGE-M3(text), metadata: {scenario_id, text}`
- **scenario_logs** (SQLite): `id, scenario_id, session_id, completed, nodes_visited, slots_filled, duration_sec, exit_reason, created_at`

## AI Scenario Generation Pipeline

### 3 Generation Paths

1. **Natural Language** — 사용자가 자연어로 지시 → Claude API가 시나리오 JSON 생성
2. **Document** — 상담 매뉴얼 PDF 업로드 → Knowledge 파서로 추출 → Claude가 시나리오로 변환
3. **Call Log** — 상담 로그 업로드 → Claude가 반복 패턴 분석 → 시나리오 초안

### Generation → Deployment Workflow

1. **생성**: Claude API 호출 → 시나리오 JSON 초안
2. **구조 검증 (자동)**: JSON Schema validation, 노드 연결 무결성, 슬롯 참조 유효성, 도달 불가 노드 탐지, End 노드 존재
3. **대화 시뮬레이션 (자동)**: Claude가 고객 역할로 정상/재시도/이탈/에러 경로 테스트. 커버리지 점수 산출
4. **트리거 검증 (자동)**: 기존 시나리오와 임베딩 충돌 체크, 테스트 발화 매칭 정확도
5. **사람 검토**: 비주얼 에디터에서 결과 확인, 수정/승인
6. **배포**: DB 저장 + ChromaDB 임베딩 + Brain 캐시 갱신 webhook

Step 2~4 실패 시 Claude에게 오류 전달 → 재생성 (최대 3회).

## Visual Editor UI

### 3-Panel Layout

- **좌측 패널**: 시나리오 목록 (상태별 필터) + AI 채팅 패널 (생성/수정 자연어 요청)
- **중앙 캔버스**: Vue Flow (vueflow.dev) 그래프 에디터. 노드 드래그앤드롭, 줌/패닝/미니맵. 노드 색상 = 타입별 구분
- **우측 패널**: 선택된 노드 속성 편집, 슬롯 현황, 검증 결과/경고

### Node Color Scheme

- 파랑: Speak
- 초록: Slot Collect
- 주황: Confirm
- 보라: Condition
- 회색: End
- 남색: API Call
- 청록: RAG Search
- 하늘: LLM Response
- 빨강: Transfer

### AI Chat Integration

좌측 AI 채팅 패널에서 자연어로 시나리오 수정 요청 → Claude API → 그래프 자동 업데이트. 예: "도난일 때 경찰 신고 안내도 추가해줘" → speak 노드 자동 삽입.

## Brain Dialogue Engine

### Module Structure

```
realtime_demo/
├── server.py              # 기존 (변경 최소화)
├── s2s_pipeline.py        # 기존
├── turn_detector.py       # 기존
│
├── dialogue/              # 신규
│   ├── __init__.py
│   ├── engine.py          # DialogueEngine 메인 클래스
│   ├── intent_matcher.py  # 임베딩 + LLM 인텐트 매칭
│   ├── graph_walker.py    # 노드 그래프 순회 실행기
│   ├── slot_manager.py    # 슬롯 추출 + 관리
│   ├── action_runner.py   # API 호출, 전환, 종결 실행
│   └── scenario_cache.py  # Knowledge에서 시나리오 로드/캐시
```

### DialogueEngine Interface

```python
class DialogueEngine:
    async def process_utterance(self, session, text, prosody) -> DialogueResult:
        # prosody는 session.scenario_state["prosody"]에 저장하여
        # llm_response 노드에서 감정 컨텍스트로 활용
        if session.dialogue_mode == "freeform":
            match = await self.intent_matcher.match(text)
            if match and match.confidence > 0.7:
                session.enter_scenario(match.scenario)
                session.scenario_state["prosody"] = prosody
                return await self._execute_current_node(session, text)
            else:
                return DialogueResult(mode="freeform", should_use_s2s=True)
        elif session.dialogue_mode == "scenario":
            session.scenario_state["prosody"] = prosody
            return await self._execute_current_node(session, text)
```

### server.py Integration

```python
async def handle_endpoint(session, final_text, prosody):
    result = await dialogue_engine.process_utterance(session, final_text, prosody)
    if result.should_use_s2s:
        await run_s2s_pipeline(session, final_text, prosody)
    elif result.response_text:
        await synthesize_and_send(session, result.response_text)
    if result.action:
        await action_runner.execute(result.action, session)
```

### StreamingSession Extension

```python
class StreamingSession:
    # 기존 필드 유지
    # 신규 필드
    dialogue_mode: str = "freeform"
    scenario_state: dict = {
        "scenario_id": None,
        "scenario_version": None,   # 시작 시 버전 고정 (캐시 갱신 영향 안 받음)
        "current_node": None,
        "slots": {},
        "variables": {},
        "history": [],
        "retry_count": 0,
        "prosody": None,            # 현재 턴의 prosody 컨텍스트
        "stack": []                  # 시나리오 중첩 스택 (깊이 2 제한)
    }
```

**시나리오 스택:** 시나리오 A 진행 중 새 intent 감지 시, 현재 `scenario_state`를 `stack`에 push하고 새 시나리오 진입. 새 시나리오 완료 후 `stack`에서 pop하여 이전 시나리오 복귀. `len(stack) >= 2`이면 중첩 거부 → freeform 폴백.

**버전 고정:** `enter_scenario()` 시 시나리오 JSON을 세션에 복사. 캐시 갱신이 진행 중인 세션에 영향을 주지 않는다.

### Auto-Advance

사용자 입력이 불필요한 노드(speak, condition, api_call, rag_search)는 자동으로 다음 노드를 실행. slot_collect, confirm만 사용자 응답을 대기.

## Knowledge Service API Extension

### Scenario CRUD

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/scenarios | 목록 조회 (필터: status, created_by) |
| POST | /api/scenarios | 수동 생성 |
| GET | /api/scenarios/{id} | 상세 조회 |
| PUT | /api/scenarios/{id} | 수정 |
| DELETE | /api/scenarios/{id} | 삭제 (soft → archived) |
| POST | /api/scenarios/{id}/deploy | draft → active + Brain 캐시 갱신 |
| POST | /api/scenarios/{id}/archive | active → archived |

### AI Generation

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/scenarios/generate | AI 시나리오 생성 (natural_language / document / call_log) |
| POST | /api/scenarios/{id}/refine | AI 수정 요청 |
| POST | /api/scenarios/{id}/validate | 구조 + 시뮬레이션 + 트리거 검증 |
| POST | /api/scenarios/{id}/simulate | 대화 시뮬레이션만 실행 |

### Knowledge Service → Brain 전용 API (Knowledge에서 호스팅, Brain이 호출)

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/brain/scenarios | active 시나리오 전체 (그래프 JSON + 트리거 벡터, version 타임스탬프 포함) |
| POST | /api/brain/log-execution | 시나리오 실행 로그 수집 |

### Brain 내부 API (Brain에서 호스팅, Knowledge가 호출)

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/knowledge/refresh | 시나리오 캐시 갱신 트리거 (기존 엔드포인트 확장) |

### Brain 내부 처리 (HTTP API 아님)

- **Intent 매칭**: Brain이 로컬 캐시 벡터로 cosine similarity 계산 (~1ms). HTTP 호출 불필요.

### Brain ↔ Knowledge Communication

1. **Brain 시작 시**: Brain → GET /api/brain/scenarios (Knowledge) → 시나리오 + 트리거 벡터 캐시
2. **시나리오 배포 시**: Knowledge → POST /api/knowledge/refresh (Brain) → Brain이 GET /api/brain/scenarios 재호출
3. **인텐트 매칭**: Brain 로컬 벡터 cosine similarity → top-3 → Qwen LLM 판정
4. **실행 로그**: Brain → POST /api/brain/log-execution (Knowledge, 비동기 fire-and-forget)

### API 인증 및 Rate Limiting

- Knowledge Service의 시나리오 API는 API key 기반 인증 적용
- `/api/scenarios/generate` (Claude API 호출): rate limit 5회/분 (비용 제어)
- `/api/scenarios/{id}/validate` (시뮬레이션 포함): rate limit 10회/분
- Brain ↔ Knowledge 내부 통신: 내부 네트워크 신뢰 (별도 인증 없음)

## Error Handling

### Node-Level Errors

| 상황 | 처리 |
|------|------|
| Slot 추출 실패 | retry_count 증가 → max_retries 초과 시 fail_action (transfer/end/skip) |
| API 호출 실패 | on_error 노드로 이동 → 미정의 시 기본 에러 메시지 + 재시도 1회 → 상담원 전환 |
| LLM 호출 실패 | condition → default branch, slot → 규칙 폴백, llm_response → 정적 메시지 |
| 시나리오 중 이탈 | 현재 시나리오 스택 push → 새 시나리오 진입 → 완료 후 pop (깊이 2 제한) |
| 캐시 미스 | freeform 유지, 기존 S2S 정상 동작 |
| 무한 루프 | 같은 노드 3회 방문 시 강제 종결 + 로그 기록 |

### Graceful Degradation

시나리오 시스템의 어떤 부분이 실패해도 기존 S2S 파이프라인은 정상 동작. Knowledge 다운, Intent 매칭 실패, 그래프 에러 → 모두 freeform 폴백. 시나리오 엔진은 "있으면 더 좋은" 레이어.

## Testing Strategy

### Layer 1: Unit Tests (pytest)

- `test_graph_walker.py`: speak auto-advance, slot collect/retry/fail, condition rule/llm, confirm yes/no, end exit, infinite loop detection
- `test_intent_matcher.py`: embedding match, no match → freeform, confidence threshold
- `test_slot_manager.py`: LLM extract card number/enum, failure returns None
- `test_action_runner.py`: API call success/timeout, transfer action
- `test_scenario_api.py`: CRUD lifecycle, deploy status, brain scenarios active only
- `test_scenario_generator.py`: generate from NL, validate structure, trigger conflict

### Layer 2: Integration Tests

- freeform → scenario transition
- full scenario walkthrough (진입 → slot × 2 → confirm → api → end)
- scenario exit → freeform 복귀
- scenario stack switch (A 진행 중 → B 진입 → B 완료 → A 복귀)
- Knowledge → Brain cache sync

### Layer 3: AI Simulation (E2E)

Claude가 고객 역할, Brain이 상담원 역할. 시나리오별 자동 생성 테스트:
- 정상 경로, 슬롯 재시도, 최대 재시도 초과, 조건 분기, 시나리오 이탈, API 에러

커버리지 = (테스트된 노드 / 전체 노드) × 100. 목표 90% 이상 → deploy 허용.
