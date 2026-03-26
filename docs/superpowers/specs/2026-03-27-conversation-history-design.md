# Conversation History Design Spec

## Overview

Brain S2S 파이프라인에 세션별 대화 히스토리를 추가하여 멀티턴 대화를 지원한다.
현재는 매 발화가 독립적인 single-turn으로 처리되어 LLM이 이전 맥락을 알 수 없다.

## Requirements

- 히스토리 우선: RAG/FAQ보다 대화 맥락이 우선
- WebSocket 연결 단위 리셋 + UI 리셋 버튼
- vLLM `max_model_len` 1024 -> 2048 상향

## Architecture

### Data Structure

`StreamingSession`에 `conversation_history: list[dict]` 추가.

```python
conversation_history = [
    {"role": "user", "content": "어제 물건을 주문했거든요"},
    {"role": "assistant", "content": "네, 주문 건으로 문의 주셨군요."},
]
```

- 노이즈 필터 통과 후 Final 확정 시: `{"role": "user", "content": text}` append
- LLM 응답 완료 시: `_run_s2s()`가 `session.conversation_history`에 직접 append
- Barge-in으로 S2S 취소 시: assistant 턴은 히스토리에 추가하지 않음
- WebSocket 종료 또는 UI 리셋: `conversation_history = []`

### Token Budget (max_model_len=2048)

| 구성 요소 | 토큰 (대략) |
|-----------|------------|
| 시스템 프롬프트 | ~200 |
| RAG + FAQ | ~200-400 (히스토리 길이에 따라 동적 축소) |
| 대화 히스토리 | ~800 (최대) |
| 현재 발화 | ~50 |
| 응답 (max_tokens) | 80 |
| 여유 | ~500 |

### RAG Budget Adaptation

히스토리가 길어지면 RAG top_k를 동적으로 줄인다:

| 히스토리 글자 수 | RAG top_k | 설명 |
|-----------------|-----------|------|
| 0-600자 | 3 | 풀 RAG |
| 601-900자 | 2 | RAG 축소 |
| 901자 이상 | 1 | 최소 RAG |

### Truncation Strategy

- 최근 N턴 유지 (1턴 = user + assistant 쌍)
- 기본 최대 6턴 (12 메시지)
- 글자 수 기반 추정: 한국어 1토큰 ~ 1.5자 -> 800토큰 ~ 1200자 한도
- 1200자 초과 시 오래된 턴부터 제거

### Messages Assembly Order

```python
messages = [
    {"role": "system", "content": system_prompt},
    # --- 히스토리 (최근 N턴, 원본 텍스트만) ---
    {"role": "user", "content": "어제 물건을 주문했거든요"},
    {"role": "assistant", "content": "네, 주문 건으로 문의 주셨군요."},
    # --- 현재 발화 (RAG + audio context 포함) ---
    {"role": "user", "content": "[audio: ...]\n[참고 문서]\n...\n고객: 내가 언제 주문했다고 말했죠"},
]
```

히스토리 메시지는 원본 텍스트만 저장 (audio context, RAG 제외).
현재 발화만 RAG/audio context 포함.

### Reset Triggers

- WebSocket 연결 종료: StreamingSession 소멸과 함께 자동 소멸
- 기존 RESTART 버튼: WebSocket 재연결 → StreamingSession 재생성 → 히스토리 소멸
- 새 "대화 리셋" 버튼: 히스토리만 초기화 (녹음/연결 유지)

## Edge Cases

### 노이즈 필터와 히스토리

노이즈로 판정된 발화는 히스토리에 추가하지 않는다.
코드 흐름: Final → 노이즈 필터 → 통과한 경우에만 user append → S2S 시작.

### Barge-in (끼어들기)

S2S 생성 중 barge-in 발생 시:
- 부분 생성된 assistant 응답은 히스토리에 **추가하지 않음** (불완전한 답변이 맥락을 오염)
- 이전 user 발화는 이미 히스토리에 있음
- 새 user 발화가 Final되면 정상적으로 추가

결과적으로 히스토리에 연속 user 메시지가 생길 수 있지만, chat 모델은 이를 처리할 수 있다.

### 빠른 연속 발화

사용자가 빠르게 두 발화를 하면 (Final #1 → Final #2, S2S #1 완료 전):
- Final #1: user append → S2S #1 시작
- Final #2: S2S #1 cancel (barge-in) → user append → S2S #2 시작
- 히스토리: [user #1, user #2] (assistant #1 없음)
- 이는 허용 가능. LLM은 연속 user 메시지를 자연스럽게 처리한다.

### vLLM 컨텍스트 길이 초과

글자 수 추정은 정확하지 않으므로, vLLM이 context length 에러를 반환할 수 있다.
이 경우 히스토리를 절반으로 줄여서 재시도한다 (최대 1회).

## Changes

### 1. `server.py`

- `StreamingSession.__init__()`: `self.conversation_history = []` 추가
- `StreamingSession.reset_for_new_utterance()`: 히스토리는 리셋하지 않음 (발화 간 유지)
- `_run_s2s()`: `session` 파라미터 추가. `llm_done` 이벤트 수신 시 `session.conversation_history.append({"role": "assistant", "content": full_text})` 직접 수행. barge-in cancel 시 append 하지 않음.
- WebSocket 핸들러: 노이즈 필터 통과 후 S2S 시작 직전에 `session.conversation_history.append({"role": "user", "content": text})`
- WebSocket 핸들러: `{"type": "reset_history"}` 메시지 수신 시 `session.conversation_history = []`
- S2S 호출 시 `session.conversation_history`를 `_run_s2s()`에 전달

### Append 순서 (동시성)

1. Final 확정 + 노이즈 필터 통과
2. `session.conversation_history.append({"role": "user", ...})` — WebSocket 핸들러에서
3. `_run_s2s()` 시작 (asyncio.Task) — history 스냅샷 전달
4. LLM 응답 완료 시 `session.conversation_history.append({"role": "assistant", ...})` — `_run_s2s` 내에서
5. 다음 Final 시 2번으로 돌아감

asyncio 단일 스레드 이벤트 루프이므로 append 순서가 자연스럽게 보장된다.

### 2. `s2s_pipeline.py`

- `LLMEngine.generate_stream()`: 기존 `user_text` + `system_prompt` 파라미터 대신 `messages: list[dict]`를 직접 받도록 시그니처 변경
- `S2SPipeline.process()`: `history: list[dict] = None` 파라미터 추가
- Messages 조립 로직을 `process()`에서 수행:
  1. `_truncate_history(history, max_chars=1200)` → 최근 턴 우선 유지
  2. 히스토리 길이에 따라 RAG `top_k` 동적 결정 (3/2/1)
  3. `messages = [system] + truncated_history + [current_user_with_rag_audio]`
  4. `LLMEngine.generate_stream(messages)`에 전달

### 3. `static/index.html`

- 기존 RESTART 버튼은 그대로 유지 (WebSocket 재연결 = 히스토리 자동 소멸)
- 새 "대화 리셋" 버튼 추가 (녹음 유지, 히스토리만 초기화)
- 클릭 시 `ws.send(JSON.stringify({"type": "reset_history"}))` 전송
- 서버 응답 `{"type": "history_reset"}` 수신 시 UI 대화 영역 표시 초기화

### 4. vLLM 실행 설정

- `--max-model-len 1024` -> `--max-model-len 2048`
- xVisor_S2S.md 실행 가이드 업데이트

## Non-Goals

- 대화 로그 영구 저장 (향후 "통화 기록 분석 서비스"에서 처리)
- 세션 간 히스토리 공유
- 토큰 수 정밀 계산 (tokenizer 호출 없이 글자 수 추정)
- WebSocket 재연결 시 히스토리 복원 (production에서 필요 시 session token으로 확장)
