# Conversation History Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Brain S2S 파이프라인에 세션별 대화 히스토리를 추가하여 멀티턴 대화 지원

**Architecture:** `StreamingSession`에 `conversation_history` 리스트를 추가. 노이즈 필터 통과 후 user 턴 append, `_run_s2s()`에서 LLM 완료 시 assistant 턴 append. `S2SPipeline.process()`가 히스토리를 받아 messages 배열을 조립하고, `LLMEngine.generate_stream()`에 전달. vLLM context 초과 시 히스토리 절반으로 재시도.

**Tech Stack:** Python 3.10, FastAPI, asyncio, vLLM OpenAI API

**Spec:** `docs/superpowers/specs/2026-03-27-conversation-history-design.md`

---

## File Structure

| 파일 | 변경 | 역할 |
|------|------|------|
| `realtime_demo/s2s_pipeline.py` | Modify | LLM messages 조립, truncation, RAG top_k 동적 조절, context 초과 재시도 |
| `realtime_demo/server.py` | Modify | Session에 히스토리 추가, append 로직, reset 핸들링 |
| `realtime_demo/static/index.html` | Modify | 대화 리셋 버튼 추가 |
| `xVisor_S2S.md` | Modify | vLLM max-model-len 2048 업데이트 |

---

### Task 1: s2s_pipeline.py — LLM messages 조립 + truncation + 재시도

히스토리 truncation 함수와 messages 조립 로직을 먼저 구현한다. 서버 측 변경 없이 단독 테스트 가능.

**Files:**
- Modify: `realtime_demo/s2s_pipeline.py:63-130` (LLMEngine), `realtime_demo/s2s_pipeline.py:229-295` (S2SPipeline.process)

- [ ] **Step 1: LLMEngine.generate_stream() 시그니처 변경 + context 에러 감지**

`user_text` + `system_prompt` 대신 `messages: list[dict]`를 직접 받도록 변경.
vLLM이 400 에러 (context length 초과)를 반환하면 `httpx.HTTPStatusError`를 raise하여 caller가 재시도할 수 있게 함.

`realtime_demo/s2s_pipeline.py`에서 `generate_stream` 메서드를 수정:

```python
# 기존:
async def generate_stream(
    self,
    user_text: str,
    system_prompt: str = SYSTEM_PROMPT,
    max_tokens: int = LLM_MAX_TOKENS,
    temperature: float = LLM_TEMPERATURE,
) -> AsyncGenerator[str, None]:
    ...
    payload = {
        "model": self.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        ...
    }

# 변경:
async def generate_stream(
    self,
    messages: list[dict],
    max_tokens: int = LLM_MAX_TOKENS,
    temperature: float = LLM_TEMPERATURE,
) -> AsyncGenerator[str, None]:
    """vLLM API 스트리밍 생성 — 토큰 단위로 yield."""
    if not self._loaded:
        self.load()

    payload = {
        "model": self.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": LLM_TOP_P,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    async with self._client.stream(
        "POST", "/chat/completions", json=payload
    ) as response:
        # vLLM context length 초과 시 400 에러
        if response.status_code == 400:
            body = await response.aread()
            raise ValueError(f"vLLM rejected request: {body.decode()}")
        # ... 나머지 스트리밍 로직 동일 (기존 aiter_lines 부분)
```

- [ ] **Step 2: truncation 함수 추가**

`S2SPipeline` 클래스 바로 위 (모듈 레벨)에 추가:

```python
MAX_HISTORY_CHARS = 1200
MAX_HISTORY_TURNS = 6

def _truncate_history(history: list[dict], max_chars: int = MAX_HISTORY_CHARS, max_turns: int = MAX_HISTORY_TURNS) -> list[dict]:
    """히스토리를 최근 N턴, max_chars 이내로 잘라냄.

    1턴 = user + assistant 쌍 (연속 user만 있을 수도 있음).
    오래된 메시지부터 제거. 단일 메시지가 max_chars 초과하면 그대로 반환.
    """
    if not history:
        return []

    # 최대 턴 수 제한 (user 메시지 기준으로 카운트)
    user_indices = [i for i, m in enumerate(history) if m["role"] == "user"]
    if len(user_indices) > max_turns:
        start = user_indices[-max_turns]
        history = history[start:]

    # 글자 수 제한 — 최소 1개 메시지는 유지
    total_chars = sum(len(m["content"]) for m in history)
    while total_chars > max_chars and len(history) > 1:
        removed = history[0]
        total_chars -= len(removed["content"])
        history = history[1:]
        # user로 시작하도록 보정 (잘린 assistant만 남는 경우 방지)
        if len(history) > 1 and history[0]["role"] == "assistant":
            total_chars -= len(history[0]["content"])
            history = history[1:]

    return history


def _get_rag_top_k(history: list[dict]) -> int:
    """히스토리 길이에 따라 RAG top_k를 동적 결정."""
    if not history:
        return 3
    total_chars = sum(len(m["content"]) for m in history)
    if total_chars > 900:
        return 1
    if total_chars > 600:
        return 2
    return 3
```

- [ ] **Step 3: S2SPipeline.process()에 history 파라미터 추가 + messages 조립 + 재시도**

`process()` 메서드를 수정:

```python
async def process(
    self,
    user_text: str,
    system_prompt: str = SYSTEM_PROMPT,
    cancel_event: asyncio.Event = None,
    audio_context: dict = None,
    history: list[dict] = None,        # ← 추가
) -> AsyncGenerator[dict, None]:
```

기존 RAG 검색 부분에서 `top_k`를 동적으로 — **truncated** 히스토리 기준:

```python
# 기존:
rag_results = await self.knowledge_client.search(user_text, top_k=3)

# 변경:
truncated = _truncate_history(history or [])
rag_top_k = _get_rag_top_k(truncated)
rag_results = await self.knowledge_client.search(user_text, top_k=rag_top_k)
```

기존 `user_msg` 조립 이후, LLM 호출 직전에 messages 배열 조립 + context 초과 재시도:

```python
# 기존:
async for token in self.llm.generate_stream(user_msg, system_prompt):

# 변경: messages 조립
messages = [{"role": "system", "content": system_prompt}]
messages.extend(truncated)
messages.append({"role": "user", "content": user_msg})

logger.info(f"[S2S] History: {len(truncated)} msgs, "
            f"{sum(len(m['content']) for m in truncated)} chars, "
            f"RAG top_k={rag_top_k}")

# context 초과 시 히스토리 절반으로 재시도 (최대 1회)
try:
    token_stream = self.llm.generate_stream(messages)
    # 첫 토큰 시도로 에러 조기 감지
    first_token = None
    async for token in token_stream:
        first_token = token
        break
except ValueError as e:
    if "rejected" in str(e) and truncated:
        logger.warning(f"[S2S] Context overflow, retrying with halved history")
        half = _truncate_history(truncated, max_chars=MAX_HISTORY_CHARS // 2)
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(half)
        messages.append({"role": "user", "content": user_msg})
        token_stream = self.llm.generate_stream(messages)
        async for token in token_stream:
            first_token = token
            break
    else:
        raise

# first_token 처리 후 나머지 토큰 스트리밍 계속
if first_token is not None:
    # first_token 처리 (기존 토큰 처리 로직과 동일)
    ...
async for token in token_stream:
    # 기존 토큰 처리 로직
    ...
```

**참고**: 실제 구현 시 기존 `async for token in ...` 루프를 재사용하는 방식으로 리팩터링. 위 코드는 재시도 로직의 개념을 보여줌. 구현 시 `_generate_with_retry()` 헬퍼로 추출하는 것을 권장:

```python
async def _generate_with_retry(self, messages, truncated, system_prompt, user_msg):
    """LLM 스트리밍 생성. context 초과 시 히스토리 절반으로 1회 재시도."""
    try:
        async for token in self.llm.generate_stream(messages):
            yield token
    except ValueError as e:
        if "rejected" in str(e) and truncated:
            logger.warning("[S2S] Context overflow, retrying with halved history")
            half = _truncate_history(truncated, max_chars=MAX_HISTORY_CHARS // 2)
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(half)
            messages.append({"role": "user", "content": user_msg})
            async for token in self.llm.generate_stream(messages):
                yield token
        else:
            raise
```

그 다음 기존 `async for token in self.llm.generate_stream(...)` 을:

```python
async for token in self._generate_with_retry(messages, truncated, system_prompt, user_msg):
```

로 교체.

- [ ] **Step 4: 수동 테스트 — vLLM 없이 messages 조립 확인**

Python REPL로 truncation + messages 조립이 올바른지 확인:

```bash
cd /home/jonghooy/work/cachedSTT && python3 -c "
import sys; sys.path.insert(0, '.')
from realtime_demo.s2s_pipeline import _truncate_history, _get_rag_top_k

# 빈 히스토리
assert _truncate_history([]) == []
assert _get_rag_top_k([]) == 3

# 짧은 히스토리 (2턴)
h = [
    {'role': 'user', 'content': '어제 물건을 주문했거든요'},
    {'role': 'assistant', 'content': '네, 주문 건으로 문의 주셨군요.'},
    {'role': 'user', 'content': '그런데 아직 도착하지 않았어요'},
    {'role': 'assistant', 'content': '배송 상태를 확인해 드리겠습니다.'},
]
result = _truncate_history(h)
assert len(result) == 4
assert _get_rag_top_k(h) == 3

# 긴 히스토리 (글자 수 초과)
long_h = []
for i in range(10):
    long_h.append({'role': 'user', 'content': f'질문 {i} ' + '가' * 100})
    long_h.append({'role': 'assistant', 'content': f'답변 {i} ' + '나' * 100})
result = _truncate_history(long_h)
total = sum(len(m['content']) for m in result)
assert total <= 1200
assert result[0]['role'] == 'user'
assert _get_rag_top_k(result) == 1  # truncated 기준

# 연속 user (barge-in 케이스)
h2 = [
    {'role': 'user', 'content': '첫 번째'},
    {'role': 'user', 'content': '두 번째'},
    {'role': 'assistant', 'content': '답변'},
]
result = _truncate_history(h2)
assert len(result) == 3

# 단일 긴 메시지 (max_chars 초과해도 1개는 유지)
h3 = [{'role': 'user', 'content': '가' * 2000}]
result = _truncate_history(h3)
assert len(result) == 1

# RAG top_k 경계값
h_600 = [{'role': 'user', 'content': '가' * 300}, {'role': 'assistant', 'content': '나' * 300}]
assert _get_rag_top_k(h_600) == 3  # 정확히 600
h_601 = [{'role': 'user', 'content': '가' * 301}, {'role': 'assistant', 'content': '나' * 300}]
assert _get_rag_top_k(h_601) == 2  # 601
h_901 = [{'role': 'user', 'content': '가' * 451}, {'role': 'assistant', 'content': '나' * 450}]
assert _get_rag_top_k(h_901) == 1  # 901

print('All truncation tests passed!')
"
```

Expected: `All truncation tests passed!`

- [ ] **Step 5: Commit**

```bash
git add realtime_demo/s2s_pipeline.py
git commit -m "feat: add conversation history support to S2S pipeline

- LLMEngine.generate_stream() accepts messages list
- Add _truncate_history() with max 6 turns / 1200 chars
- Add _get_rag_top_k() for dynamic RAG budget (truncated history 기준)
- S2SPipeline.process() accepts history parameter
- Context overflow retry: halve history and retry once"
```

---

### Task 2: server.py — 히스토리 append + reset 핸들링

서버 측에서 히스토리를 관리하고, `_run_s2s()`가 assistant 응답을 히스토리에 추가.

**Files:**
- Modify: `realtime_demo/server.py:84-152` (StreamingSession), `realtime_demo/server.py:466-544` (_run_s2s), `realtime_demo/server.py:547-780` (websocket_endpoint)

- [ ] **Step 1: StreamingSession에 conversation_history 추가**

`StreamingSession.__init__()`에 추가 (line ~107 부근):

```python
# 대화 히스토리 (멀티턴)
self.conversation_history = []
```

`reset_for_new_utterance()` 메서드에는 히스토리 리셋을 **추가하지 않음** (발화 간 유지).

- [ ] **Step 2: _run_s2s()에 session 파라미터 추가 + assistant append**

```python
# 기존:
async def _run_s2s(websocket: WebSocket, user_text: str, utterance_id: int,
                   cancel_event: asyncio.Event = None, audio_context: dict = None):

# 변경:
async def _run_s2s(websocket: WebSocket, user_text: str, utterance_id: int,
                   cancel_event: asyncio.Event = None, audio_context: dict = None,
                   session: StreamingSession = None, history: list[dict] = None):
```

`s2s_pipeline.process()` 호출에 history 전달:

```python
# 기존:
async for event in s2s_pipeline.process(user_text, cancel_event=cancel_event,
                                         audio_context=audio_context):

# 변경:
async for event in s2s_pipeline.process(user_text, cancel_event=cancel_event,
                                         audio_context=audio_context,
                                         history=history):
```

`llm_done` 이벤트 처리 부분에서 assistant 히스토리 append:

```python
elif etype == "llm_done":
    await websocket.send_json({
        "type": "s2s_llm_done",
        "full_text": event["full_text"],
        "utterance_id": utterance_id,
    })
    # 대화 히스토리에 assistant 응답 추가
    # barge-in으로 취소된 경우 추가하지 않음
    if session is not None and not (cancel_event and cancel_event.is_set()):
        session.conversation_history.append({
            "role": "assistant",
            "content": event["full_text"],
        })
```

- [ ] **Step 3: WebSocket 핸들러 — user append + history 전달 + reset 처리**

**reset_history 메시지 처리** (기존 barge_in 처리 바로 뒤, line ~574 부근):

```python
if msg.get("type") == "barge_in":
    ...
    continue
# 대화 히스토리 리셋
if msg.get("type") == "reset_history":
    session.conversation_history = []
    await websocket.send_json({"type": "history_reset"})
    logger.info("[History] Reset by user")
    continue
```

**노이즈 필터 통과 후, S2S 시작 전 — snapshot-then-append 패턴** (line ~762 부근):

```python
# 기존 (노이즈 필터 continue 이후):
# S2S 파이프라인: Final 텍스트 → LLM → TTS → 음성 응답
if S2S_ENABLED and s2s_pipeline and s2s_pipeline.is_loaded():

# 변경: snapshot 먼저 → user append → S2S에 snapshot 전달
# 히스토리 스냅샷 (현재 user 발화는 process()에서 current로 들어가므로 제외)
history_snapshot = list(session.conversation_history)

# 대화 히스토리에 user 발화 추가
session.conversation_history.append({
    "role": "user",
    "content": session.last_text,
})

# S2S 파이프라인: Final 텍스트 → LLM → TTS → 음성 응답
if S2S_ENABLED and s2s_pipeline and s2s_pipeline.is_loaded():
```

**S2S 호출에 session + history 전달**:

```python
# 기존:
s2s_task = asyncio.create_task(
    _run_s2s(websocket, final_text_for_s2s,
             session.utterance_count, s2s_cancel_event,
             audio_context=prosody)
)

# 변경:
s2s_task = asyncio.create_task(
    _run_s2s(websocket, final_text_for_s2s,
             session.utterance_count, s2s_cancel_event,
             audio_context=prosody,
             session=session, history=history_snapshot)
)
```

- [ ] **Step 4: 서버 시작하여 수동 검증**

```bash
cd /home/jonghooy/work/cachedSTT
conda activate nemo-asr
# vLLM이 실행 중인지 확인
curl -s http://localhost:8000/v1/models | python3 -m json.tool
# 서버 시작 (테스트)
CUDA_VISIBLE_DEVICES=0 python realtime_demo/server.py --s2s --port 3000
```

브라우저에서 http://localhost:3000 접속, 두 번 연속 발화 후 두 번째 응답이 첫 번째 맥락을 참조하는지 확인.

- [ ] **Step 5: Commit**

```bash
git add realtime_demo/server.py
git commit -m "feat: wire conversation history in server

- StreamingSession.conversation_history for multi-turn
- snapshot-then-append pattern for safe history passing
- Assistant turn appended on llm_done (skipped on barge-in)
- reset_history WebSocket message handler"
```

---

### Task 3: static/index.html — 대화 리셋 버튼

**Files:**
- Modify: `realtime_demo/static/index.html`

- [ ] **Step 1: 리셋 버튼 CSS 추가**

기존 `#btn-restart:hover` 스타일 (line 78) 바로 뒤에 추가:

```css
#btn-reset-history {
    padding: 0.5rem 1rem;
    border-radius: 6px;
    border: 1px solid #334155;
    background: #1e293b;
    color: #94a3b8;
    font-size: 0.85rem;
    cursor: pointer;
    transition: all 0.2s;
}
#btn-reset-history:hover { border-color: #38bdf8; color: #38bdf8; }
```

- [ ] **Step 2: 리셋 버튼 HTML 추가**

기존 RESTART 버튼 (line 418) 바로 뒤에 추가:

```html
<button id="btn-restart" onclick="restartSession()">RESTART</button>
<button id="btn-reset-history" onclick="resetHistory()">대화 리셋</button>
```

- [ ] **Step 3: resetHistory() JavaScript 함수 추가**

`restartSession()` 함수 바로 뒤에 추가:

```javascript
function resetHistory() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({type: 'reset_history'}));
    }
}
```

- [ ] **Step 4: history_reset 응답 처리**

WebSocket `onmessage` 핸들러의 기존 `switch` 문에 `case` 추가:

```javascript
case 'history_reset':
    const histStatus = document.getElementById('status');
    histStatus.textContent = '대화 히스토리 초기화됨';
    setTimeout(() => {
        if (isRecording) histStatus.textContent = '녹음 중...';
        else histStatus.textContent = '마이크 버튼을 눌러 시작';
    }, 2000);
    break;
```

- [ ] **Step 5: Commit**

```bash
git add realtime_demo/static/index.html
git commit -m "feat: add conversation reset button to UI

- New '대화 리셋' button next to RESTART
- Sends reset_history message via WebSocket
- switch/case handler shows confirmation for 2 seconds"
```

---

### Task 4: vLLM 설정 + 문서 업데이트

**Files:**
- Modify: `xVisor_S2S.md`

- [ ] **Step 1: xVisor_S2S.md에서 max-model-len 업데이트**

섹션 8.1 vLLM 실행 명령어에서:

```bash
# 기존:
--max-model-len 1024

# 변경:
--max-model-len 2048
```

섹션 3.4 LLM 테이블에서:

```markdown
# 기존:
| max_model_len | 1024 |

# 변경:
| max_model_len | 2048 |
```

- [ ] **Step 2: 섹션 11.3에서 대화 히스토리 TODO 완료 표시**

```markdown
# 기존:
- [ ] 대화 히스토리 관리 (멀티턴)

# 변경:
- [x] 대화 히스토리 관리 (멀티턴) — Brain 세션별, 최근 6턴, 글자 수 기반 truncation
```

- [ ] **Step 3: Commit**

```bash
git add xVisor_S2S.md
git commit -m "docs: update vLLM max-model-len to 2048, mark history as done"
```

---

### Task 5: 통합 테스트

**Files:** 없음 (실행 확인만)

- [ ] **Step 1: vLLM 재시작 (max-model-len 2048)**

```bash
pkill -f "vllm serve"
conda activate vllm_serving
CUDA_VISIBLE_DEVICES=0 vllm serve /mnt/usb/models/Qwen3.5-9B \
    --host 0.0.0.0 --port 8000 \
    --dtype bfloat16 --max-model-len 2048 \
    --gpu-memory-utilization 0.75 \
    --trust-remote-code --enforce-eager \
    --enable-prefix-caching --max-num-seqs 4 &
```

- [ ] **Step 2: Brain 서버 시작**

```bash
pkill -f "realtime_demo/server.py"
cd /home/jonghooy/work/cachedSTT
conda activate nemo-asr
CUDA_VISIBLE_DEVICES=0 python realtime_demo/server.py --s2s --port 3000
```

- [ ] **Step 3: 멀티턴 대화 검증**

브라우저에서 http://localhost:3000 접속:

1. "어제 물건을 주문했거든요" -> 응답 확인
2. "그런데 아직 도착하지 않았어요" -> 응답이 주문 맥락을 참조하는지 확인
3. "내가 언제 주문했다고 말했죠" -> "어제"를 기억하는지 확인
4. "대화 리셋" 버튼 클릭 -> 상태 메시지 확인
5. "내가 뭐라고 했죠?" -> 히스토리 없이 응답하는지 확인

서버 로그에서 `[S2S] History: N msgs, M chars, RAG top_k=K` 출력 확인.

- [ ] **Step 4: Edge case 검증**

1. Barge-in: TTS 재생 중 끼어들기 -> 히스토리에 불완전 assistant 없는지 확인
2. 빠른 연속 발화: 짧은 간격으로 2회 발화 -> 서버 에러 없는지 확인
3. 긴 대화: 7-8턴 이상 대화 -> truncation 동작 확인 (서버 로그에서 히스토리 크기 확인)

- [ ] **Step 5: Final commit (있으면)**

통합 테스트 중 발견된 이슈 수정 후 커밋.
