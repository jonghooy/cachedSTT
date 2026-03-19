# CachedSTT - Korean ASR Training Documentation

NeMo Cache-Aware Streaming 한국어 ASR 모델 훈련 프로젝트.

## 현재 접근: Nemotron 0.6B Transfer Learning

**상세 가이드**: `cachedSTT_new.md` 참조

### 아키텍처

| 항목 | 값 |
|------|-----|
| Base Model | Nemotron Speech ASR 0.6B (영어) |
| Architecture | FastConformer RNN-T (24 layers, d_model=1024) |
| Parameters | 618M |
| Tokenizer | SentencePiece Unigram, vocab=4096 (한국어, byte_fallback) |
| Training Data | 763.8시간 (315,341 샘플) |
| GPU | RTX 5090 32GB (single) |
| Streaming | Cache-Aware Multi-lookahead `[[70,13],[70,6],[70,1],[70,0]]` |
| Framework | NeMo 2.6.1, PyTorch 2.6, Lightning 2.5 |
| Conda env | `nemo-asr` |

### 4단계 학습 전략

```
Stage 1: Decoder Warmup (3 epochs)
  - Encoder 전체 freeze, Decoder+Joint만 학습
  - lr=5.0, warmup=5000, fastemit=0.01
  - batch=8, accumulate=8, effective=64
  - 결과: val_wer 0.518

Stage 2: Upper Encoder (5 epochs)
  - Layer 0-11 freeze, 12-23 + Decoder 학습
  - lr=2.0, warmup=10000, fastemit=0.005
  - 결과: val_wer 0.194

Stage 3: Full Fine-tune (5 epochs)
  - 전체 unfreeze, lr=0.5, warmup=5000
  - 결과: val_wer 0.163

Stage 4: Punctuation Fine-tune (3 epochs) ← 진행 중
  - Stage 3 체크포인트에서 시작, 전체 unfreeze
  - lr=0.3, warmup=3000, fastemit=0.005
  - 구두점(. ?) 포함 데이터로 학습
  - 목적: 모델이 문장 끝에 구두점 출력 → 서버에서 EOS 감지
```

---

## 현재 진행 상황 (2026-02-09)

### Stage 1: 완료

- **최종 val_wer: 0.518** (목표 0.8 달성)
- 3 epochs, 모델 저장: `experiments/Stage1-Decoder-Warmup_final.nemo`

| Epoch | val_wer |
|-------|---------|
| 0 중간 | 0.836 |
| 0 완료 | 0.651 |
| 1 완료 | 0.574 |
| 2 완료 | **0.518** |

### Stage 2: 완료

- **최종 val_wer: 0.194** (목표 0.3 달성)
- 5 epochs, 모델 저장: `experiments/Stage2-Upper-Encoder_final.nemo`

| Epoch | 중간 (50%) | 완료 (100%) |
|-------|-----------|------------|
| 0 | 0.335 | 0.292 |
| 1 | 0.272 | 0.246 |
| 2 | 0.232 | 0.221 |
| 3 | 0.209 | 0.203 |
| 4 | 0.199 | **0.194** |

### Stage 3: 완료

- **최종 val_wer: 0.163** (목표 0.15 근접)
- 5 epochs, 모델 저장: `experiments/Stage3-Full-Finetune_final.nemo`

| Epoch | 중간 (50%) | 완료 (100%) |
|-------|-----------|------------|
| 0 | 0.182 | 0.183 |
| 1 | 0.181 | - |
| 2 | 0.172 | 0.169 |
| 3 | 0.169 | 0.166 |
| 4 | 0.164 | **0.163** |

### Stage 4: 구두점 학습 — 진행 중 (2026-02-09)

- Stage 3 체크포인트에서 시작, 구두점(. ?) 포함 데이터로 Fine-tune
- **목적**: 모델이 문장 끝에 . 또는 ?를 출력하게 학습 → 서버에서 `"." in text`로 EOS 감지
- 학습 데이터: `train_manifest_punct.json` (319,354 샘플, 60.4% 구두점 포함)
- 검증 데이터: `val_manifest_punct.json` (17,715 샘플)
- 설정: lr=0.3, warmup=3000, max_epochs=3, 전체 unfreeze

**구두점 데이터 준비 과정**:
- 기존 train_manifest에 이미 35.3% 구두점 포함 (원본 유지)
- 나머지 64.7% 중 25.0%에 규칙 기반 구두점 추가 (한국어 종결어미 패턴)
- 스크립트: `scripts/add_punctuation_to_manifest.py`

**배경: 규칙 기반 EOS → 학습 기반으로 전환**:
- 서버에 규칙 기반 EOS 감지 (한국어 종결어미 regex) 시도 → 오류 증가
- 대안: 모델 자체가 구두점을 출력하게 학습 (토크나이저에 `.` ID=266, `?` ID=450 이미 존재)
- Stage 4 완료 후 서버에서 `"." in text`로 간단하게 EOS 감지 가능

### 전체 val_wer 추이 요약

```
Stage 1 (Decoder):     0.836 → 0.518  (3 epochs)
Stage 2 (Upper Enc):   0.335 → 0.194  (5 epochs)
Stage 3 (Full):        0.182 → 0.163  (5 epochs)
Stage 4 (Punctuation): 진행 중...      (3 epochs)
```

**접근법 검증 결과: 성공** — 영어 Nemotron 0.6B에서 한국어 WER 16.3%까지 도달 (764시간 데이터)

### 디코딩 속도 벤치마크 (RTX 5090, BF16)

| 항목 | 값 |
|------|-----|
| Median RTF | **0.0098** (실시간 대비 **102배 빠름**) |
| 모델 로드 시간 | 6.3초 |
| GPU 메모리 (추론) | 2.36GB / 32GB |
| 동시 처리 가능 스트림 | **~100개** (단일 GPU) |

| Batch Size | RTF | 처리 속도 |
|-----------|-----|----------|
| bs=1 | 0.009 | 109x 실시간 |
| bs=4 | 0.005 | 210x 실시간 |
| bs=8 | 0.004 | 252x 실시간 |

벤치마크 스크립트: `scratchpad/measure_rtf.py`

### 저장된 체크포인트

```
experiments/
├── Stage1-Decoder-Warmup_final.nemo          # Stage 1 최종 (val_wer=0.518)
├── Stage1-Decoder-Warmup/checkpoints/        # Stage 1 중간 체크포인트
├── Stage2-Upper-Encoder_final.nemo           # Stage 2 최종 (val_wer=0.194)
├── Stage2-Upper-Encoder/checkpoints/         # Stage 2 중간 체크포인트
├── Stage3-Full-Finetune_final.nemo           # Stage 3 최종 (val_wer=0.163)
├── Stage3-Full-Finetune/checkpoints/         # Stage 3 중간 체크포인트
├── Stage4-Punctuation_final.nemo             # Stage 4 (학습 중...)
└── Stage4-Punctuation/checkpoints/           # Stage 4 중간 체크포인트
```

### 이전 중단 이력

- **Stage 1 version_0**: CUDA graphs 에러 → `model.change_decoding_strategy()` 사용하여 해결
- **Stage 1 version_1**: Epoch 1, 43%에서 원인불명 중단 (OOM 추정) → `ValidationMemoryCleanup` 콜백 추가
- 이후 Stage 1~3 안정적으로 완료

---

## Quick Start 커맨드

### Stage 1 시작 (처음부터)
```bash
source /home/jonghooy/miniconda3/etc/profile.d/conda.sh && conda activate nemo-asr && \
CUDA_VISIBLE_DEVICES=0 nohup python /home/jonghooy/work/timbel-asr-pilot/scripts/korean_streaming_asr_train.py \
    --stage 1 \
    > /home/jonghooy/work/timbel-asr-pilot/logs/train_transfer_stage1.log 2>&1 &
echo "Training started with PID: $!"
```

> **참고**: `resume_if_exists: true`이므로 체크포인트가 있으면 자동으로 이어서 학습합니다.
> 처음부터 다시 하려면 `experiments/Stage1-Decoder-Warmup/checkpoints/` 삭제 필요.

### Stage 2 시작 (Stage 1 완료 후)
```bash
source /home/jonghooy/miniconda3/etc/profile.d/conda.sh && conda activate nemo-asr && \
CUDA_VISIBLE_DEVICES=0 nohup python /home/jonghooy/work/timbel-asr-pilot/scripts/korean_streaming_asr_train.py \
    --stage 2 --resume_from /home/jonghooy/work/timbel-asr-pilot/experiments/Stage1-Decoder-Warmup_final.nemo \
    --max_epochs 5 \
    > /home/jonghooy/work/timbel-asr-pilot/logs/train_transfer_stage2.log 2>&1 &
```

### Stage 3 시작 (Stage 2 완료 후)
```bash
source /home/jonghooy/miniconda3/etc/profile.d/conda.sh && conda activate nemo-asr && \
CUDA_VISIBLE_DEVICES=0 nohup python /home/jonghooy/work/timbel-asr-pilot/scripts/korean_streaming_asr_train.py \
    --stage 3 --resume_from /home/jonghooy/work/timbel-asr-pilot/experiments/Stage2-Upper-Encoder_final.nemo \
    --max_epochs 5 \
    > /home/jonghooy/work/timbel-asr-pilot/logs/train_transfer_stage3.log 2>&1 &
```

### Stage 4 시작 (Stage 3 완료 후, 구두점 학습)
```bash
source /home/jonghooy/miniconda3/etc/profile.d/conda.sh && conda activate nemo-asr && \
CUDA_VISIBLE_DEVICES=0 nohup python /home/jonghooy/work/timbel-asr-pilot/scripts/korean_streaming_asr_train.py \
    --stage 4 \
    --resume_from /home/jonghooy/work/timbel-asr-pilot/experiments/Stage3-Full-Finetune_final.nemo \
    --train_manifest /mnt/usb_2tb/timbel-data/manifests/train_manifest_punct.json \
    --val_manifest /mnt/usb_2tb/timbel-data/manifests/val_manifest_punct.json \
    --max_epochs 3 \
    > /home/jonghooy/work/timbel-asr-pilot/logs/train_transfer_stage4.log 2>&1 &
```

### 모니터링
```bash
# 실시간 로그
tail -f /home/jonghooy/work/timbel-asr-pilot/logs/train_transfer_stage1.log

# val_wer 추출
grep -E "val_wer" /home/jonghooy/work/timbel-asr-pilot/logs/train_transfer_stage1.log

# TensorBoard 메트릭 확인 (conda activate nemo-asr 필요)
python3 -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
# 최신 version 폴더 자동 탐색
versions = sorted(glob.glob('/home/jonghooy/work/timbel-asr-pilot/experiments/Stage1-Decoder-Warmup/version_*/'))
ea = EventAccumulator(versions[-1])
ea.Reload()
for tag in sorted(ea.Tags().get('scalars', [])):
    events = ea.Scalars(tag)
    if events:
        print(f'{tag}: {events[-1].value:.4f} (step {events[-1].step})')
"

# GPU 상태
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader

# 프로세스 확인
ps aux | grep korean_streaming_asr_train | grep -v grep
```

### 학습 중단
```bash
pkill -f "korean_streaming_asr_train.py"
```

---

## 실시간 Cache-Aware Streaming 서버 (2026-02-09)

### 아키텍처 전환: Offline → Streaming

기존 서버는 오디오가 들어올 때마다 **전체 버퍼를 `model.transcribe()`로 재전사**하는 방식 (O(buffer) per step).
NeMo의 `conformer_stream_step()` API를 사용하여 **160ms 청크 단위 증분 처리** (O(1) per step)로 전환.

| 항목 | 기존 (Offline) | 현재 (Streaming) |
|------|---------------|-----------------|
| 추론 방식 | `model.transcribe()` 전체 재전사 | `conformer_stream_step()` 증분 |
| 복잡도 | O(buffer) per step | O(1) per step |
| 지연 | 버퍼 길어지면 증가 | ~170ms 고정 |
| 인코더 캐시 | 없음 | `cache_last_channel`, `cache_last_time` |
| 디코더 상태 | 매번 초기화 | `previous_hypotheses` 연속 |

### 핵심 구현: `realtime_demo/server.py`

**StreamingSession 클래스** — WebSocket 연결별 상태 관리:
- 오디오 버퍼 + mel 프레임 추적 (`audio_buffer`, `mel_buffer_idx`, `step`)
- 인코더 캐시 3종: `cache_last_channel (24,1,70,1024)`, `cache_last_time (24,1,1024,8)`, `cache_last_channel_len (1,)`
- RNN-T 디코더 연속성: `previous_hypotheses` 스텝 간 전달
- 블랭크 토큰 카운터: `blank_step_count` (토큰 수 변화 없으면 증가)

**스트리밍 파이프라인**:
```
WebSocket PCM int16 (250ms)
  → float32 변환 → audio_buffer 누적
  → 전체 mel 변환 (streaming preprocessor: dither=0, pad_to=0)
  → mel_buffer_idx부터 chunk_size 단위로 청크 추출
  → conformer_stream_step() × N개 청크
  → 텍스트 추출 → partial/final WebSocket 전송
```

**스트리밍 설정** (`att_context_size=[70, 1]`):
```
chunk_size:            [9, 16]     # 첫 스텝 9프레임, 이후 16프레임
shift_size:            [9, 16]     # mel 슬라이딩 간격
pre_encode_cache_size: [0, 9]      # 이전 프레임 캐시
drop_extra_pre_encoded: 2          # 중복 제거
valid_out_len:          2          # 유효 출력 길이
```

**끝점 감지 (현재)**:
- 무음 0.8초 + 블랭크 3 스텝 → Final
- 무음 1.5초 단독 → Final
- 버퍼 30초 → 강제 Final
- 배경 노이즈 자동 보정 (첫 4청크 = ~1초)

### 해결한 이슈

**1. `loop_labels` 방향 (학습 vs 추론)**
- **학습**: `loop_labels: false` + `use_cuda_graph_decoder: false` (RTX 5090 CUDA graphs 비호환)
- **추론 (스트리밍)**: `loop_labels: true` + `use_cuda_graph_decoder: false`
- `loop_labels=false`에서 `previous_hypotheses` 전달 시 `NotImplementedError` 발생
- 스트리밍 서버는 반드시 `loop_labels=true` 사용

**2. step_idx 추적 버그**
- `get_available_chunks()`에서 `self.step`을 증가시킨 후 `run_streaming_step()`에서 사용 → 잘못된 `drop_extra_pre_encoded` 값
- 수정: `step_idx = self.step` 저장 후 증가, `run_streaming_step(mel_chunk, chunk_length, step_idx)`로 전달

**3. 스트리밍 전처리기 분리**
- 모델 내장 preprocessor는 `dither > 0`, `pad_to > 0` → 스트리밍에 부적합
- 별도 preprocessor 생성: `dither=0.0`, `pad_to=0` (길이 패딩 없음)
- `model._cfg.preprocessor`에서 deep copy 후 수정

**4. GPU 스레드 직렬화**
- `ThreadPoolExecutor(max_workers=1)` → GPU 추론 직렬화
- WebSocket 이벤트 루프에서 `loop.run_in_executor()` 비동기 호출

### 실행 방법

```bash
# 서버 시작
source /home/jonghooy/miniconda3/etc/profile.d/conda.sh && conda activate nemo-asr && \
cd /home/jonghooy/work/cachedSTT/realtime_demo && \
CUDA_VISIBLE_DEVICES=0 nohup python server.py > server.log 2>&1 &

# 브라우저에서 http://localhost:3000 접속

# 서버 종료
pkill -f "realtime_demo/server.py"
```

### 성능

| 항목 | 값 |
|------|-----|
| 모델 로드 | 6.2초 |
| GPU 메모리 | 2.37GB / 32GB |
| 스텝 지연 | ~170ms (160ms 청크 + 추론) |
| 워밍업 | 더미 오디오 1회 스트리밍 스텝 |

### TODO (Stage 4 완료 후)

- [ ] `MODEL_PATH`를 Stage 4 체크포인트로 교체
- [ ] 구두점 기반 EOS 감지 추가: `"." in text` 또는 `"?" in text` → 적응형 silence 임계값 하향
- [ ] 적응형 끝점 계획 (`cheeky-finding-allen.md`) 반영 검토

---

## Key Paths

| 항목 | 경로 |
|------|------|
| 스트리밍 서버 | `/home/jonghooy/work/cachedSTT/realtime_demo/server.py` |
| 프론트엔드 | `/home/jonghooy/work/cachedSTT/realtime_demo/static/index.html` |
| 훈련 프로젝트 | `/home/jonghooy/work/timbel-asr-pilot/` |
| 학습 스크립트 | `scripts/korean_streaming_asr_train.py` |
| Config 파일 | `configs/korean_streaming_rnnt_transfer.yaml` |
| Pretrained 모델 | `pretrained_models/nemotron-speech-streaming-en-0.6b.nemo` (2.36GB) |
| 한국어 토크나이저 | `tokenizer/tokenizer_unigram_4096/tokenizer_spe_unigram_v4096/` |
| 학습 데이터 | `/mnt/usb_2tb/timbel-data/manifests/train_manifest.json` |
| 검증 데이터 | `/mnt/usb_2tb/timbel-data/manifests/val_manifest.json` |
| 구두점 학습 데이터 | `/mnt/usb_2tb/timbel-data/manifests/train_manifest_punct.json` |
| 구두점 검증 데이터 | `/mnt/usb_2tb/timbel-data/manifests/val_manifest_punct.json` |
| 구두점 추가 스크립트 | `scripts/add_punctuation_to_manifest.py` |
| 실험 결과 | `experiments/Stage{1,2,3,4}-*/` |
| 훈련 로그 | `logs/train_transfer_stage{1,2,3,4}.log` |
| TensorBoard | `experiments/Stage*/version_*/` |
| 모델 아키텍처 문서 | `/home/jonghooy/work/cachedSTT/MODEL_ARCHITECTURE.md` |

---

## 구현 시 해결한 이슈들

### 1. 토크나이저 vocab 크기 (문서 2048 → 실제 4096)
- `cachedSTT_new.md`는 vocab=2048 권장하지만, `byte_fallback=True` 사용 시 한국어 문자 1903개 + byte 256개 = 최소 2166 필요
- vocab=2048로는 SentencePiece 학습 실패 → **vocab=4096으로 변경**
- RNN-T 권장 범위(1024-4096) 내이므로 문제 없음

### 2. NeMo 2.6.1 import 호환성
- `cachedSTT_new.md`의 예제는 `import pytorch_lightning as pl` 사용
- NeMo 2.6.1은 `lightning.pytorch`를 사용 → `import lightning.pytorch as pl`로 변경
- 다른 모듈(`pytorch_lightning`)의 LightningModule과 호환 안 됨

### 3. SpecAugment `_target_` 키 필터링
- Hydra config의 `_target_` 키가 SpectrogramAugmentation 생성자에 전달되면 에러
- `{k: v for k, v in config.items() if k != '_target_'}` 로 필터링

### 4. RTX 5090 CUDA graphs 비호환
- Validation 시 CUDA graph compilation 에러 (`_full_graph_compile` 에서 값 언패킹 실패)
- `model.cfg.decoding = ...` 만으로는 실제 decoding 모듈이 갱신 안 됨
- **`model.change_decoding_strategy(decoding_cfg)`** 사용하여 해결
- Config에 `loop_labels: false`, `use_cuda_graph_decoder: false` 필수

### 5. Validation 메모리 문제
- 학습 중 Epoch 1, 43%에서 원인불명 중단 (에러 없이 프로세스 kill)
- `ValidationMemoryCleanup` 콜백 추가: validation 전후 `gc.collect()` + `torch.cuda.empty_cache()`
- `pin_memory: false`는 이미 train/val 모두 적용됨

### 6. FastEmit lambda cfg 전파 주의
- `model.cfg.loss.warprnnt_numba_kwargs.fastemit_lambda` 수정이 런타임에 전파되는지 확인 필요
- 이미 초기화된 loss 모듈에는 적용 안 될 수 있음 (현재 0.005 기본값으로 동작 가능)
- Stage 1에서 0.01 대신 0.005가 적용되더라도 학습은 정상 진행됨 (확인 완료)

---

## Known Issues (RTX 5090)

- CUDA graphs 비호환: `decoding.greedy`에 `loop_labels: false`, `use_cuda_graph_decoder: false` 필수
- RNNT loss fp16 미지원 (numba): `bf16-mixed` precision 사용
- `pin_memory: true` + 높은 num_workers → validation 시 OOM 위험 → 둘 다 false/낮게
- Nemotron 0.6B (618M) + batch_size 8 → GPU ~23-28GB / 32GB
- `torch.set_float32_matmul_precision('medium')` 경고 무시 가능

---

## 이전 접근 (참고용)

이전 방식 (처음부터 학습, 17L d_model=512, Hybrid CTC+RNNT):
- Config: `configs/fastconformer_hybrid_streaming_ko_pilot.yaml`
- 문제: 학습률 설정 오류 (lr=0.0005 with NoamAnnealing → peak lr이 10000배 너무 낮음)
- Blank collapse 원인이 learning rate였음 (lr=5.0으로 수정 후 해결)
- 현재는 Nemotron 0.6B transfer learning 접근으로 전환
