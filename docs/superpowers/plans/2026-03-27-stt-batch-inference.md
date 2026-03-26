# STT Model-Level Batch Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `_process_batch()`를 모델 레벨 배칭으로 교체 — 여러 세션의 캐시를 stack하여 `conformer_stream_step()` 1회 호출로 처리

**Architecture:** 세션별 mel 청크와 캐시를 batch dimension으로 concat/stack하여 단일 GPU 호출. step=0은 개별 처리, step>0만 배치. mel 길이 차이는 0-padding + length masking. 라운드 방식으로 다중 청크 소화.

**Tech Stack:** PyTorch, NeMo ASR, FastAPI/asyncio

**Spec:** `docs/superpowers/specs/2026-03-27-stt-batch-inference-design.md`

---

## File Structure

| 파일 | 변경 | 역할 |
|------|------|------|
| `realtime_demo/server.py` | Modify | `_process_batch()` 교체 + `_batched_stream_step()` 추가 |

단일 파일 수정. `_process_batch()` 함수를 배치 추론으로 교체하고, 배치 추론 로직을 `_batched_stream_step()` 헬퍼로 분리.

---

### Task 1: `_batched_stream_step()` — 배치 추론 핵심 함수

여러 세션의 mel + 캐시를 받아 단일 `conformer_stream_step()` 호출로 처리하고 결과를 세션별로 분배.

**Files:**
- Modify: `realtime_demo/server.py` — `_process_batch()` 바로 위에 새 함수 추가

- [ ] **Step 1: `_batched_stream_step()` 함수 작성**

`_process_batch()` 바로 위에 추가:

```python
def _batched_stream_step(sessions_and_chunks):
    """여러 세션의 (session, mel_chunk, chunk_length, step_idx)를 배치로 추론.

    모든 항목이 step > 0이어야 함 (동일 chunk 파라미터).
    mel 길이가 다르면 0-padding.

    Args:
        sessions_and_chunks: list of (session, mel_chunk, chunk_length, step_idx)

    Returns:
        list of (text, session) — 각 세션의 디코딩 결과
    """
    if not sessions_and_chunks:
        return []

    n = len(sessions_and_chunks)
    drop = streaming_cfg.drop_extra_pre_encoded  # step > 0 공통

    # 1. Mel padding + stack
    mel_chunks = [item[1] for item in sessions_and_chunks]  # 각각 (1, n_mels, frames)
    max_frames = max(m.shape[2] for m in mel_chunks)

    padded_mels = []
    lengths = []
    for m in mel_chunks:
        frames = m.shape[2]
        if frames < max_frames:
            m = torch.nn.functional.pad(m, (0, max_frames - frames))
        padded_mels.append(m)
        lengths.append(frames)

    batched_mel = torch.cat(padded_mels, dim=0)  # (N, n_mels, max_frames)
    batched_len = torch.tensor(lengths, device="cuda:0")  # (N,)

    # 2. 캐시 stack (batch dim = dim 1)
    cache_channel = torch.cat(
        [s.cache_last_channel for s, _, _, _ in sessions_and_chunks], dim=1
    )  # (layers, N, C, D)
    cache_time = torch.cat(
        [s.cache_last_time for s, _, _, _ in sessions_and_chunks], dim=1
    )  # (layers, N, D, T)
    cache_len = torch.cat(
        [s.cache_last_channel_len for s, _, _, _ in sessions_and_chunks], dim=0
    )  # (N,)

    # 3. Hypotheses concat
    hypotheses = []
    for s, _, _, _ in sessions_and_chunks:
        if s.previous_hypotheses and len(s.previous_hypotheses) > 0:
            hypotheses.append(s.previous_hypotheses[0])
        else:
            hypotheses.append(None)

    # 4. 배치 추론
    with torch.no_grad():
        result = model.conformer_stream_step(
            processed_signal=batched_mel,
            processed_signal_length=batched_len,
            cache_last_channel=cache_channel,
            cache_last_time=cache_time,
            cache_last_channel_len=cache_len,
            keep_all_outputs=False,
            previous_hypotheses=hypotheses,
            drop_extra_pre_encoded=drop,
            return_transcription=True,
        )

    (greedy_preds, transcriptions,
     out_cache_channel, out_cache_time, out_cache_len,
     best_hyp) = result[:6]

    # 5. 결과 split → 각 세션에 복원
    texts = []
    for i, (session, _, _, _) in enumerate(sessions_and_chunks):
        # 캐시 복원 (batch dim slice)
        session.cache_last_channel = out_cache_channel[:, i:i+1, :, :]
        session.cache_last_time = out_cache_time[:, i:i+1, :, :]
        session.cache_last_channel_len = out_cache_len[i:i+1]

        # Hypothesis 복원 + 블랭크 감지
        if best_hyp and i < len(best_hyp):
            hyp = best_hyp[i]
            session.previous_hypotheses = [hyp]

            current_tokens = len(hyp.y_sequence)
            if current_tokens == session.last_token_count:
                session.blank_step_count += 1
            else:
                session.blank_step_count = 0
                session.last_token_count = current_tokens

            token_ids = hyp.y_sequence
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.cpu().tolist()
            text = model.tokenizer.ids_to_text(token_ids)
            texts.append(text)
        else:
            texts.append("")

    return texts
```

- [ ] **Step 2: 검증 — 단일 세션으로 호출해도 동작 확인**

서버 시작 후 브라우저에서 1채널 접속, 발화 → partial/final 정상 수신 확인.
(이 시점에서는 아직 `_process_batch`가 기존 코드이므로, 함수만 추가된 상태)

- [ ] **Step 3: Commit**

```bash
git add realtime_demo/server.py
git commit -m "feat: add _batched_stream_step() for multi-session GPU inference

- Stack mel with 0-padding, caches on batch dim
- Single conformer_stream_step() call for N sessions
- Split results back to per-session state

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `_process_batch()` 교체 — 배치 추론 사용

기존 순차 처리를 배치 추론으로 교체.

**Files:**
- Modify: `realtime_demo/server.py` — `_process_batch()` 전면 교체

- [ ] **Step 1: `_process_batch()` 교체**

기존:
```python
def _process_batch(sessions: list):
    results = []
    for session in sessions:
        chunks = session.get_available_chunks()
        if not chunks:
            results.append(None)
            continue
        text = ""
        for mel_chunk, chunk_length, step_idx in chunks:
            text = session.run_streaming_step(mel_chunk, chunk_length, step_idx)
        results.append((text, session.blank_step_count))
    return results
```

변경:
```python
def _process_batch(sessions: list):
    """GPU 스레드: 여러 세션의 가용 청크를 배치 추론으로 처리.

    step=0 세션은 개별 처리, step>0 세션은 모델 레벨 배치.
    라운드 방식: 세션당 1청크씩 배치 → 남은 청크 반복.

    Returns:
        list of (text, blank_step_count) or None per session
    """
    # 각 세션의 가용 청크 수집
    session_chunks = []  # [(session, chunks_list), ...]
    for session in sessions:
        chunks = session.get_available_chunks()
        session_chunks.append((session, chunks))

    # 세션별 최종 텍스트 추적
    session_texts = {id(s): "" for s in sessions}

    # 라운드 방식: 매 라운드마다 세션당 1청크씩 처리
    while True:
        step0_items = []    # 개별 처리 (step=0)
        stepN_items = []    # 배치 처리 (step>0)
        active = False

        for session, chunks in session_chunks:
            if not chunks:
                continue
            active = True
            mel_chunk, chunk_length, step_idx = chunks.pop(0)
            if step_idx == 0:
                step0_items.append((session, mel_chunk, chunk_length, step_idx))
            else:
                stepN_items.append((session, mel_chunk, chunk_length, step_idx))

        if not active:
            break

        # step=0: 개별 batch=1 처리
        for session, mel_chunk, chunk_length, step_idx in step0_items:
            text = session.run_streaming_step(mel_chunk, chunk_length, step_idx)
            session_texts[id(session)] = text

        # step>0: 모델 레벨 배치
        if stepN_items:
            texts = _batched_stream_step(stepN_items)
            for (session, _, _, _), text in zip(stepN_items, texts):
                session_texts[id(session)] = text

    # 결과 조립
    results = []
    for session in sessions:
        text = session_texts[id(session)]
        if text:
            results.append((text, session.blank_step_count))
        else:
            results.append(None)

    return results
```

- [ ] **Step 2: 1채널 테스트**

서버 재시작 후 브라우저 1채널 접속:
1. 발화 → partial 텍스트 표시 확인
2. 종결어미 → final + S2S 응답 확인
3. 서버 로그에서 에러 없는지 확인

- [ ] **Step 3: 2채널 동시 테스트**

브라우저 탭 2개로 동시 접속:
1. 양쪽에서 발화
2. 서버 로그에서 `[Batch] Processing 2 sessions` 확인
3. 양쪽 모두 partial/final 정상 수신 확인

- [ ] **Step 4: 배치 크기 로깅 추가**

`_process_batch()`에서 배치 크기를 로그:

기존 STTBatchProcessor.run()의 로그를 더 상세하게:
```python
if len(sessions) > 1:
    logger.info(f"[Batch] Processing {len(sessions)} sessions")
```

이 부분은 이미 있으므로, `_process_batch` 내부에 배치 추론 로그 추가:

```python
# step>0: 모델 레벨 배치
if stepN_items:
    if len(stepN_items) > 1:
        logger.info(f"[Batch] Batched inference: {len(stepN_items)} sessions")
    texts = _batched_stream_step(stepN_items)
```

- [ ] **Step 5: Commit**

```bash
git add realtime_demo/server.py
git commit -m "feat: replace sequential STT with model-level batch inference

- step=0: individual batch=1 processing
- step>0: batched conformer_stream_step(batch=N)
- Round-based multi-chunk handling
- 30 channels target on single RTX 5090

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: 멀티채널 스트레스 테스트

**Files:** 없음 (테스트 스크립트 실행만)

- [ ] **Step 1: 3채널 동시 테스트 (더미 오디오)**

기존 테스트 스크립트 재실행:

```bash
cd /home/jonghooy/work/cachedSTT && conda activate nemo-asr && python3 -c "
import asyncio, websockets, numpy as np, json, time

async def test_channel(name, duration_sec=3):
    uri = 'ws://localhost:3000/ws'
    results = []
    try:
        async with websockets.connect(uri) as ws:
            for _ in range(5):
                silence = np.zeros(4000, dtype=np.int16)
                await ws.send(silence.tobytes())
                try: await asyncio.wait_for(ws.recv(), timeout=0.3)
                except asyncio.TimeoutError: pass
            t = np.linspace(0, duration_sec, int(16000 * duration_sec))
            audio = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
            for i in range(0, len(audio), 4000):
                chunk = audio[i:i+4000]
                await ws.send(chunk.tobytes())
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.3)
                    data = json.loads(msg)
                    results.append(data['type'])
                except asyncio.TimeoutError: pass
            for _ in range(10):
                silence = np.zeros(4000, dtype=np.int16)
                await ws.send(silence.tobytes())
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.5)
                    data = json.loads(msg)
                    results.append(data['type'])
                except asyncio.TimeoutError: pass
    except Exception as e:
        results.append(f'ERROR:{e}')
    return name, results

async def main():
    print('=== 배치 추론 멀티채널 테스트 ===')
    t0 = time.time()
    tasks = [test_channel(f'CH-{i}', 2) for i in range(3)]
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - t0
    for name, msgs in results:
        err = [m for m in msgs if 'ERROR' in str(m)]
        print(f'  {name}: {len(msgs)} msgs, errors={len(err)}')
    print(f'  총 소요: {elapsed:.1f}s')

asyncio.run(main())
"
```

서버 로그에서 `[Batch] Batched inference: N sessions` 확인.

- [ ] **Step 2: 에러 없으면 완료**

서버 로그에서:
1. `[Batch] Batched inference: 2 sessions` 또는 `3 sessions` 출력
2. WebSocket 에러 없음
3. 크래시 없음
