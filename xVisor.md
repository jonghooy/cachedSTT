# xVisor: 모듈형 S2S (Speech-to-Speech) 콜봇 아키텍처

## 개요

기존 보유 자산(STT 인코더 + LLM + TTS Vocoder)을 조립하여 음성-to-음성 콜센터 자동 응대 시스템 구축.

| 항목 | 값 |
|------|-----|
| 목표 | 콜센터 자동 응대 (콜봇) |
| 지연시간 목표 | 1초 이하 (고객 발화 종료 → 첫 음성 출력) |
| LLM | Qwen3.5 9B |
| STT | FastConformer RNN-T 0.6B (한국어, WER 16.3%) |
| TTS | StyleTTS2 (한국어 콜봇 200epoch 학습 완료, 24kHz) |
| PoC 인프라 | RTX 5090 32GB (단일 GPU) |
| 프로덕션 인프라 | 별도 계획 |
| 지식 연동 (RAG) | 나중에 (S2S 파이프라인 먼저) |
| 음성 페르소나 | StyleTTS2 기존 콜봇 목소리 |

---

## 점진적 진화 전략: A → B → C

### Phase 1 (PoC): 텍스트 파이프라인 — 2주 ⭐ 현재 단계

```
고객 음성 → [STT 전체] → 텍스트 → [Qwen3.5 9B] → 텍스트 → [StyleTTS2] → 응답 음성
         (cache-aware       (스트리밍 생성)      (문장 단위 합성)
          streaming)
```

**핵심 전략**: 새로운 모델 학습 없이 기존 검증된 모듈을 연결하여 동작하는 S2S 데모 완성.

**예상 지연시간 (첫 음성 출력까지):**

| 단계 | 시간 | 설명 |
|------|------|------|
| Turn Detection + Final | ~200ms | 종결어미 감지 + endpoint |
| Qwen3.5 TTFT | ~300ms | 첫 토큰 생성 (vLLM/transformers) |
| 첫 문장 생성 완료 | ~200ms | 스트리밍 토큰 → 문장 단위 버퍼링 |
| StyleTTS2 합성 | ~300ms | 첫 문장 → 음성 변환 |
| **총 첫 음성 출력** | **~800ms** | ✅ 1초 이하 목표 달성 |

**장점:**
- 2주 안에 확실히 완성 가능
- 각 모듈 독립 검증됨 — 디버깅 용이
- 중간 텍스트 확인 가능 (품질 모니터링)
- 파이프라인 구조가 Phase 2/3 업그레이드에 호환

**학습 데이터: 불필요** (기존 모델 + 시스템 프롬프트)

**구현 상태: ✅ 완료 (2026-03-26)**

실행 구조 (마이크로서비스):
- **vLLM 서버** (port 8000, `vllm_serving` env): Qwen3.5-9B BF16, GPU 23.8GB
- **STT+S2S 서버** (port 3000, `nemo-asr` env): NeMo STT + S2S Pipeline, GPU 3.3GB
- **Edge TTS** (Microsoft 클라우드): GPU 불필요, 한국어 남/여 음성

실행 방법:
```bash
# 1. vLLM 서버 시작
conda activate vllm_serving && \
CUDA_VISIBLE_DEVICES=0 vllm serve /mnt/usb/models/Qwen3.5-9B \
    --port 8000 --dtype bfloat16 --max-model-len 2048 \
    --gpu-memory-utilization 0.75 --trust-remote-code --enforce-eager &

# 2. STT + S2S 서버 시작
conda activate nemo-asr && python realtime_demo/server.py --s2s --port 3000

# 브라우저에서 http://localhost:3000 접속
```

주요 이슈 해결:
- Qwen3.5 (`model_type: qwen3_5`)는 transformers 4.57 미지원 → vLLM 0.18.0 별도 서빙
- Qwen3.5 thinking 모드 → `chat_template_kwargs: {enable_thinking: false}` 로 비활성화
- StyleTTS2 모델 파일 미존재 → Edge TTS (클라우드)로 대체 (프로덕션에서 StyleTTS2 복원 예정)
- GPU 메모리 관리: vLLM 0.75 utilization (24.5GB) + STT (3.3GB) = 27.8GB / 32.6GB

---

### Phase 2: Speech Projector — +3~4주

```
고객 음성 → [STT 인코더만] → 1024-dim → [Projector] → [Qwen3.5] → 텍스트 → [StyleTTS2] → 응답 음성
```

**목표**: 인코더 임베딩을 LLM에 직접 주입하여 감정/억양 정보 보존.

**핵심 작업:**
- Speech Projector (Linear 1024 → Qwen hidden_dim) 학습
- 기존 STT 학습 데이터 763.8시간 재활용

**학습 데이터 (Projector 정렬용):**
```jsonl
{"audio": "call_001.wav", "transcript": "카드 분실 신고하려고요", "duration": 2.1}
{"audio": "call_002.wav", "transcript": "네 본인 확인 부탁드립니다", "duration": 1.8}
```
- 형태: (음성 → 인코더 1024-dim 프레임) ↔ (텍스트 → LLM 토큰 임베딩) 정렬
- 기존 763.8시간 STT 데이터 활용 가능
- Projector만 학습 (인코더/LLM은 freeze)

---

### Phase 3: Full Audio-to-Audio — +2~3개월

```
고객 음성 → [STT 인코더] → [Projector] → [Qwen3.5] → 오디오 토큰 → [Vocoder] → 응답 음성
```

**목표**: 텍스트 병목 완전 제거, 최저 지연시간 달성.

**핵심 작업:**
- Encodec/DAC 오디오 코덱 통합
- Qwen3.5 fine-tune (텍스트 대신 오디오 토큰 출력)
- StyleTTS2 Vocoder를 오디오 토큰 디코더로 전환

**학습 데이터 (대화 쌍):**
```jsonl
{"input_audio": "customer.wav", "output_audio": "agent.wav", "input_text": "카드 분실했어요", "output_text": "네 바로 정지 처리하겠습니다", "semantic_tokens": [123, 456, ...]}
```
- 콜센터 대화 쌍 (고객 ↔ 상담원) 대량 필요
- 오디오 코덱 토큰화 전처리 파이프라인 필요

---

## 보유 자산 상세

### STT: FastConformer RNN-T (cachedSTT)

| 항목 | 값 |
|------|-----|
| 인코더 | FastConformer 24L, d_model=1024 |
| 인코더 출력 | (batch, time, 1024), 80ms/frame |
| 디코더 | RNN-T (2-layer LSTM 640-dim) |
| 파라미터 | 618M |
| 한국어 WER | 16.3% |
| 스트리밍 | Cache-aware, multi-lookahead [70,0/1/6/13] |
| RTF | 0.0098 (102x 실시간) |
| GPU 메모리 | 2.37GB |
| 학습 데이터 | 763.8시간 한국어 |

### TTS: StyleTTS2 (zhisper)

| 항목 | 값 |
|------|-----|
| 위치 | /home/jonghooy/work/zhisper/tts/engine/styletts2_engine.py |
| 학습 | 한국어 콜봇 데이터 200 epoch |
| 출력 | 24kHz |
| 보코더 | HiFi-GAN variant |
| 합성기 | Diffusion sampler (ADPM2, Karras schedule) |
| 텍스트 처리 | G2P (g2pK) + 숫자/통화/날짜 정규화 |

### LLM: Qwen3.5 9B

| 항목 | 값 |
|------|-----|
| 모델 | Qwen3.5 9B |
| 추론 | vLLM 또는 transformers 스트리밍 |
| GPU 메모리 (예상) | ~18GB (BF16) |
| 총 GPU 사용 (예상) | STT 2.4GB + LLM 18GB + TTS ~3GB = ~23.4GB / 32GB |

---

## Key Paths

| 항목 | 경로 |
|------|------|
| 이 문서 | /home/jonghooy/work/cachedSTT/xVisor.md |
| STT 서버 | /home/jonghooy/work/cachedSTT/realtime_demo/server.py |
| STT 프론트엔드 | /home/jonghooy/work/cachedSTT/realtime_demo/static/index.html |
| Turn Detector | /home/jonghooy/work/cachedSTT/realtime_demo/turn_detector.py |
| StyleTTS2 엔진 | /home/jonghooy/work/zhisper/tts/engine/styletts2_engine.py |
| STT 학습 모델 | /home/jonghooy/work/timbel-asr-pilot/experiments/Stage3-Full-Finetune_final.nemo |
| 모델 아키텍처 문서 | /home/jonghooy/work/cachedSTT/MODEL_ARCHITECTURE.md |
