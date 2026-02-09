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

### 3단계 학습 전략 (검증용 축소 에포크)

현재는 소량 데이터(764시간)로 접근법 검증 목적이므로 에포크를 축소하여 진행.

```
Stage 1: Decoder Warmup (3 epochs)
  - Encoder 전체 freeze, Decoder+Joint만 학습
  - lr=5.0, warmup=5000, fastemit=0.01
  - batch=8, accumulate=8, effective=64
  - 목표: WER < 0.8 → 결과: 0.518 (달성)

Stage 2: Upper Encoder (5 epochs)
  - Layer 0-11 freeze, 12-23 + Decoder 학습
  - lr=2.0, warmup=10000, fastemit=0.005
  - 목표: WER < 0.3 → 결과: 0.194 (달성)

Stage 3: Full Fine-tune (5 epochs)
  - 전체 unfreeze, lr=0.5, warmup=5000
  - 목표: WER < 0.15 → 결과: 0.163 (근접)
```

---

## 현재 진행 상황 (2026-02-08)

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

### 전체 val_wer 추이 요약

```
Stage 1 (Decoder):     0.836 → 0.518  (3 epochs)
Stage 2 (Upper Enc):   0.335 → 0.194  (5 epochs)
Stage 3 (Full):        0.182 → 0.163  (5 epochs)
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
├── Stage3-Full-Finetune_final.nemo           # Stage 3 최종 (val_wer=0.169)
└── Stage3-Full-Finetune/checkpoints/         # Stage 3 중간 체크포인트
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

## Key Paths

| 항목 | 경로 |
|------|------|
| 훈련 프로젝트 | `/home/jonghooy/work/timbel-asr-pilot/` |
| 학습 스크립트 | `scripts/korean_streaming_asr_train.py` |
| Config 파일 | `configs/korean_streaming_rnnt_transfer.yaml` |
| Pretrained 모델 | `pretrained_models/nemotron-speech-streaming-en-0.6b.nemo` (2.36GB) |
| 한국어 토크나이저 | `tokenizer/tokenizer_unigram_4096/tokenizer_spe_unigram_v4096/` |
| 학습 데이터 | `/mnt/usb_2tb/timbel-data/manifests/train_manifest.json` |
| 검증 데이터 | `/mnt/usb_2tb/timbel-data/manifests/val_manifest.json` |
| 실험 결과 | `experiments/Stage{1,2,3}-*/` |
| 훈련 로그 | `logs/train_transfer_stage{1,2,3}.log` |
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
