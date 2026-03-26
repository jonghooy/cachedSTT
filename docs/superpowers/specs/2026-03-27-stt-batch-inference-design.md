# STT Model-Level Batch Inference Design Spec

## Overview

현재 STT 추론은 세션별 batch=1로 순차 처리된다. `conformer_stream_step()`이 batch > 1을 지원하므로, 여러 세션의 캐시와 mel을 stack하여 단일 GPU 호출로 처리한다. 30채널 목표.

## Requirements

- 여러 세션의 mel 청크 + 캐시를 batch로 묶어 `conformer_stream_step()` 1회 호출
- step=0 세션은 개별 batch=1 처리 (드묾, chunk_size가 다름)
- mel 길이가 다른 세션은 0-padding + length masking
- 기존 STTBatchProcessor 큐/수집 구조는 유지

## Architecture

### 현재 (순차 처리)

```
_process_batch(sessions=[A, B, C]):
  model.forward(A, batch=1)  # GPU 호출 1
  model.forward(B, batch=1)  # GPU 호출 2
  model.forward(C, batch=1)  # GPU 호출 3
```

### 변경 후 (모델 레벨 배칭)

```
_process_batch(sessions=[A, B, C]):
  step0_sessions = [세션 where step==0]  → 개별 batch=1 처리
  stepN_sessions = [A, B, C]             → 배치 처리:
    1. 각 세션에서 mel 청크 1개 추출
    2. mel 0-padding → stack (3, n_mels, max_frames)
    3. 캐시 concat: (layers, 1, ...) × 3 → (layers, 3, ...)
    4. hypotheses concat: [hyp_A, hyp_B, hyp_C]
    5. conformer_stream_step(batch=3)  # GPU 호출 1회
    6. 결과 split → 각 세션 캐시/hypothesis 복원
    7. 남은 청크 있으면 반복
```

### 캐시 Stack/Split

```python
# Stack (추론 전)
cache_channel = torch.cat([s.cache_last_channel for s in sessions], dim=1)
# (layers, 1, C, D) × N → (layers, N, C, D)

cache_time = torch.cat([s.cache_last_time for s in sessions], dim=1)
# (layers, 1, D, T) × N → (layers, N, D, T)

cache_len = torch.cat([s.cache_last_channel_len for s in sessions], dim=0)
# (1,) × N → (N,)

hypotheses = [s.previous_hypotheses[0] if s.previous_hypotheses else None for s in sessions]

# Split (추론 후)
for i, session in enumerate(sessions):
    session.cache_last_channel = result_cache_channel[:, i:i+1, :, :]
    session.cache_last_time = result_cache_time[:, i:i+1, :, :]
    session.cache_last_channel_len = result_cache_len[i:i+1]
    session.previous_hypotheses = [result_best_hyp[i]]
```

### Mel Padding

```python
# 각 세션의 mel 청크 수집
mel_chunks = [session_mel for session in sessions]  # 각각 (1, n_mels, frames_i)
max_frames = max(m.shape[2] for m in mel_chunks)

# 0-padding + stack
padded = []
lengths = []
for m in mel_chunks:
    pad_size = max_frames - m.shape[2]
    if pad_size > 0:
        m = torch.nn.functional.pad(m, (0, pad_size))
    padded.append(m)
    lengths.append(mel_chunks_original_length)

batched_mel = torch.cat(padded, dim=0)  # (N, n_mels, max_frames)
batched_len = torch.tensor([original lengths], device=...)  # (N,)
```

### 세션별 다중 청크 처리

한 세션이 여러 청크를 가질 수 있다 (오디오 밀림). 라운드 방식:

```
라운드 1: 모든 세션에서 청크 1개씩 → batch=N 추론
라운드 2: 남은 청크가 있는 세션만 → batch=M 추론 (M <= N)
...대부분 1-2 라운드로 완료
```

### 텍스트 추출

배치 추론 후 세션별 개별 처리:

```python
for i, session in enumerate(sessions):
    hyp = result_best_hyp[i]
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
```

## Changes

### `server.py`

- `_process_batch()` 전면 교체: 순차 처리 → 배치 추론
- 새 헬퍼 함수들:
  - `_batch_infer(sessions, chunks_per_session)`: 캐시 stack → 추론 → split
  - mel padding, 캐시 stack/split 로직
- `StreamingSession`: 변경 없음 (캐시가 이미 batch=1 텐서)
- `STTBatchProcessor`: 변경 없음

### step=0 처리

step=0 세션은 chunk_size, pre_encode_cache_size, drop_extra_pre_encoded가 다르므로 기존 `run_streaming_step()` 메서드로 개별 처리. step=0은 세션 시작 시 1회만 발생하므로 성능 영향 없음.

## Performance Expectation

| 항목 | 현재 (순차) | 변경 후 (배치) |
|------|-----------|---------------|
| 8세션 GPU 호출 | 8회 | 1회 |
| 30세션 GPU 호출 | 30회 | ~4회 (batch=8) |
| 350ms 간격 내 처리 | ~8채널 한계 | ~30채널 가능 |

batch=8일 때 RTF=0.004 → 1 step ~50ms. 30채널 = 4 batch × 50ms = 200ms < 350ms 간격.

## Non-Goals

- 최대 batch size 자동 조절 (고정 또는 가용 세션 수 사용)
- 인코더 자체의 vectorized 디코딩 (NeMo RNN-T 디코더가 내부적으로 순차 처리)
- GPU 메모리 기반 동적 batch 제한
