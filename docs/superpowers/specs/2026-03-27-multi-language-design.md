# Multi-Language Support Design Spec (한/영)

## Overview

Brain S2S 파이프라인에 한국어/영어 자동 감지를 추가. 기본은 한국어 전용, UI 토글로 자동 감지 모드 활성화. 영어로 말하면 영어로 응답.

## Requirements

- 기본 모드: 한국어 전용 (현재 동작 유지)
- 자동 감지 모드: 첫 발화 시 LangID로 언어 감지 → 해당 언어로 STT/LLM/TTS 전체 전환
- UI에서 수동 언어 전환 가능
- 영어 STT: Nemotron 0.6B English (이미 있음)
- 영어 TTS: StyleTTS2 LibriTTS (다운로드 완료)
- LangID: NeMo AmberNet (107언어, 0.22GB, ~50ms)

## Architecture

### Models (서버 시작 시 전부 로드)

| 모델 | 언어 | GPU 메모리 | 경로 |
|------|------|-----------|------|
| STT 한국어 | ko | 2.4GB | `runpod_trained_Stage3-best-cer0.1968.nemo` |
| STT 영어 | en | 2.4GB | `nemotron-speech-streaming-en-0.6b.nemo` |
| LangID | 107 | 0.2GB | `langid_ambernet` (NeMo pretrained) |
| TTS 한국어 | ko | 0.6GB | `KB_60h/epoch_2nd_00034.pth` |
| TTS 영어 | en | 0.6GB | `LibriTTS_EN/Vocos/LibriTTS/epoch_2nd_00029.pth` |
| vLLM AWQ | multi | 8GB | `Qwen3.5-9B-AWQ` |
| **합계** | | **~14.2GB / 32GB** | |

### Data Flow

```
UI 토글: [한국어 전용] / [자동 감지]
  ↓ WebSocket: {"type": "set_language_mode", "mode": "auto"|"ko"}

StreamingSession:
  language_mode = "ko" | "auto"
  detected_language = "ko"  (기본)
  lang_detected = False

첫 발화 (is_speaking=True, 오디오 ~1초 누적):
  if language_mode == "auto" and not lang_detected:
    LangID(오디오) → "ko" or "en"
    detected_language = result
    lang_detected = True
    → WebSocket: {"type": "language_detected", "language": "en"}

STT 라우팅:
  detected_language == "ko" → model_ko.conformer_stream_step()
  detected_language == "en" → model_en.conformer_stream_step()

S2S 파이프라인:
  LLM: 언어별 시스템 프롬프트
  TTS: detected_language == "ko" → tts_ko, "en" → tts_en
```

### Language Detection

- 모델: NeMo `langid_ambernet` (AmberNet, 107언어)
- 시점: 첫 발화 감지 시 (`is_speaking=True` 후 오디오 ~1초 누적)
- 방법: `lang_id_model.get_label(audio_file)` 또는 텐서 직접 전달
- 결과: "ko" or "en" (그 외 언어는 "ko"로 fallback)
- 1회만 실행 (세션 내 첫 발화에서)

### Batch Inference 변경

STT 배치 시 같은 언어 모델끼리 그룹핑:

```python
ko_sessions = [(s, chunk) for s, chunk in items if s.detected_language == "ko"]
en_sessions = [(s, chunk) for s, chunk in items if s.detected_language == "en"]

if ko_sessions:
    _batched_stream_step(ko_sessions, model=model_ko)
if en_sessions:
    _batched_stream_step(en_sessions, model=model_en)
```

### StreamingSession 캐시 분리

한국어/영어 STT 모델이 다르므로, 세션 생성 시 detected_language에 맞는 모델로 캐시 초기화 필요.

- 초기: 한국어 모델로 캐시 생성 (기본)
- LangID 결과 "en" → 영어 모델로 캐시 재초기화 + 오디오 버퍼 유지

### 수동 언어 전환

```
WebSocket: {"type": "set_language", "language": "en"}
→ session.detected_language = "en"
→ 캐시 재초기화 (영어 모델)
→ 오디오 버퍼 리셋
→ WebSocket: {"type": "language_changed", "language": "en"}
```

## Changes

### 1. `server.py`

- 글로벌: `model_ko`, `model_en`, `lang_id_model` 추가
- startup: 영어 STT + LangID 모델 로드
- `StreamingSession`: `language_mode`, `detected_language`, `lang_detected` 필드
- `_init_caches()`: `detected_language`에 따라 적절한 모델로 캐시 생성
- `run_streaming_step()`: `detected_language`에 따라 적절한 모델로 추론
- `_batched_stream_step()`: model 파라미터 추가, 언어별 그룹핑
- `_process_batch()`: 언어별로 세션 분리 후 각각 배치
- WebSocket 핸들러: `set_language_mode`, `set_language` 메시지 처리
- WebSocket 핸들러: 첫 발화 시 LangID 실행

### 2. `s2s_pipeline.py`

- `S2SPipeline`: `tts_en` 인스턴스 추가
- `process()`: `language` 파라미터 추가
- 언어별 시스템 프롬프트 선택
- 언어별 TTS 선택 (`tts_ko` or `tts_en`)
- 영어 시스템 프롬프트 상수 추가

### 3. `static/index.html`

- 언어 모드 토글: [한국어 전용] / [자동 감지 (한/영)]
- 수동 언어 선택 드롭다운 (자동 감지 모드일 때)
- 감지된 언어 배지 표시
- WebSocket 메시지: `set_language_mode`, `set_language`
- `language_detected` 응답 처리

## Non-Goals

- 일본어/중국어 등 추가 언어 (향후 확장)
- 대화 중 언어 자동 전환 (첫 발화에서만 감지)
- 통역 모드 (한→영 번역) — 향후 별도 기능
