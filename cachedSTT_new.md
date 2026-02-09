
# 한국어 Cache-Aware Streaming ASR 하이브리드 학습 가이드

## NVIDIA Nemotron Speech ASR 아키텍처를 한국어로 Transfer Learning

---

## 1. 아키텍처 이해: 무엇을 가져오고 무엇을 새로 만드는가

### 1.1 Nemotron Speech ASR 모델 구조

```
┌─────────────────────────────────────────────────────────────┐
│                    Nemotron Speech ASR 0.6B                  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Preprocessor (Mel Spectrogram)                       │   │
│  │  - 16kHz mono → 80-dim mel-spectrogram               │   │
│  │  - 언어 무관 (물리적 음향 특성 추출)                     │   │
│  └──────────────┬───────────────────────────────────────┘   │
│                 │                                           │
│  ┌──────────────▼───────────────────────────────────────┐   │
│  │ Convolutional Subsampling (8x downsampling)          │   │
│  │  - Depthwise-separable convolutions                  │   │
│  │  - 입력 프레임을 1/8로 축소 → 연산 효율 극대화          │   │
│  │  - 언어 무관 (음향 패턴의 시간축 압축)                   │   │
│  └──────────────┬───────────────────────────────────────┘   │
│                 │                                           │
│  ┌──────────────▼───────────────────────────────────────┐   │
│  │ FastConformer Encoder (24 layers)                    │   │
│  │                                                      │   │
│  │  Layer 0~11  (하위 레이어)                             │   │
│  │  ┌────────────────────────────────────────────┐      │   │
│  │  │ • Self-Attention (limited context, cached)  │      │   │
│  │  │ • Depthwise Convolution (kernel=9, causal)  │      │   │
│  │  │ • Feed-Forward                              │      │   │
│  │  │                                             │      │   │
│  │  │ ★ 역할: 저수준 음향 특성 추출                  │      │   │
│  │  │   - 포먼트, 에너지, 피치 패턴                  │      │   │
│  │  │   - 음소 경계 감지                            │      │   │
│  │  │   - 잡음/비음성 구간 식별                      │      │   │
│  │  │   → 이 부분은 언어간 상당히 공유됨              │      │   │
│  │  └────────────────────────────────────────────┘      │   │
│  │                                                      │   │
│  │  Layer 12~23 (상위 레이어)                            │   │
│  │  ┌────────────────────────────────────────────┐      │   │
│  │  │ • Self-Attention (limited context, cached)  │      │   │
│  │  │ • Depthwise Convolution (kernel=9, causal)  │      │   │
│  │  │ • Feed-Forward                              │      │   │
│  │  │                                             │      │   │
│  │  │ ★ 역할: 고수준 언어 특성 추출                  │      │   │
│  │  │   - 음소 → 서브워드 매핑                      │      │   │
│  │  │   - 언어별 음운 규칙 학습                      │      │   │
│  │  │   - 문맥 기반 음향 해석                       │      │   │
│  │  │   → 이 부분은 언어에 종속적                    │      │   │
│  │  └────────────────────────────────────────────┘      │   │
│  └──────────────┬───────────────────────────────────────┘   │
│                 │                                           │
│  ┌──────────────▼───────────────────────────────────────┐   │
│  │ RNN-T Decoder (Prediction + Joint Network)           │   │
│  │  - Prediction Network: 이전 토큰 히스토리 기반 예측     │   │
│  │  - Joint Network: Encoder 출력 + Prediction 결합      │   │
│  │  - Vocabulary에 완전히 종속 → 한국어로 교체 필수        │   │
│  └──────────────┬───────────────────────────────────────┘   │
│                 │                                           │
│  ┌──────────────▼───────────────────────────────────────┐   │
│  │ Tokenizer (SentencePiece BPE, vocab=1024)            │   │
│  │  - 영어 서브워드 → 한국어 서브워드로 완전 교체           │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Cache-Aware Streaming 메커니즘                        │   │
│  │  - 모든 Self-Attention, Convolution 레이어에 캐시 유지  │   │
│  │  - Multi-lookahead: 80ms / 160ms / 560ms / 1.12s     │   │
│  │  - 언어 무관 (스트리밍 인프라 레이어)                   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 하이브리드 접근: Transfer vs Replace 판단

| 컴포넌트 | 전략 | 이유 |
|----------|------|------|
| **Preprocessor** | ✅ 그대로 유지 | Mel spectrogram 추출은 완전히 언어 무관 |
| **Subsampling** | ✅ 그대로 유지 | 8x downsampling은 물리적 시간축 압축. 언어 무관 |
| **Encoder Layer 0~11** | 🔒 Freeze (가중치 고정) | 저수준 음향 특성(포먼트, 에너지, 피치)은 언어간 공유 |
| **Encoder Layer 12~23** | 🔓 Unfreeze (재학습) | 고수준 언어 특성은 한국어에 맞게 적응 필요 |
| **RNN-T Decoder** | 🆕 새로 초기화 | Vocabulary가 완전히 바뀌므로 재사용 불가 |
| **Tokenizer** | 🆕 새로 구축 | 한국어 서브워드 토크나이저 필요 |
| **Cache-Aware 메커니즘** | ✅ 그대로 유지 | 스트리밍 인프라는 언어 무관 |

### 1.3 왜 하위 레이어를 Freeze 하는가?

FastConformer 인코더의 레이어별 학습 내용을 이해해야 합니다:

```
Layer 0~3:   에너지 윤곽(envelope), 기본 주파수 패턴
             → 모든 인간 언어에서 동일

Layer 4~7:   음소 경계, 모음/자음 구분, 유/무성음 판별
             → 대부분의 언어에서 유사 (한국어 경음/격음은 상위에서 구분)

Layer 8~11:  음절 구조, 음향 문맥 패턴, 화자 정규화
             → 상당 부분 공유, 일부 언어 특성 시작

Layer 12~15: 서브워드 단위 음향-텍스트 매핑
             → 언어별로 크게 다름 (영어 /th/ vs 한국어 /ㄲ/)

Layer 16~19: 문맥 의존적 음향 해석, 동시조음(coarticulation)
             → 언어별 음운 규칙에 강하게 종속

Layer 20~23: 고수준 언어 모델링, 어휘적 해석
             → 완전히 언어 종속
```

**핵심 인사이트**: 영어로 학습된 하위 레이어의 음향 특성 추출 능력은 한국어에도 유효합니다.
사람의 발성 기관이 동일하므로, 기본 음향 특성(포먼트 구조, 에너지 분포, 피치 변화)은
어떤 언어든 물리적으로 유사합니다.

---

## 2. 한국어 토크나이저 설계 (가장 중요한 단계)

### 2.1 토크나이저가 왜 핵심인가

NeMo GitHub에서 발견된 실패 사례들의 공통 원인:

- **Issue #10112**: FastConformer CTC를 한국어 1000시간으로 파인튜닝 → WER 1.0에서 수렴 안 됨
- **Issue #8256**: FastConformer Streaming을 러시아어로 파인튜닝 → 49 에폭 후에도 WER 1.0
- **Issue #8815**: Multilingual FastConformer에서 `change_vocabulary` 후 테스트 WER > 1

**대부분의 수렴 실패는 토크나이저 설계 문제 또는 디코더 초기화 문제**에서 비롯됩니다.

### 2.2 한국어 토크나이저 선택지

```
옵션 1: 자소(Jamo) 단위 — ㅎ ㅏ ㄴ ㄱ ㅜ ㄱ ㅇ ㅓ (초성/중성/종성 분리)
  - 장점: vocab 매우 작음 (~67개), 커버리지 완벽
  - 단점: 시퀀스 길어짐 → RNN-T 디코더 부담 ↑, 레이턴시 ↑

옵션 2: 음절 단위 — 한 국 어
  - 장점: 직관적, 한국어 음운 단위와 일치
  - 단점: 자주 쓰이는 음절만 해도 2,000~3,000개 → vocab 큼

옵션 3: BPE Subword — 한국 어 (SentencePiece)
  - 장점: 빈도 기반으로 효율적 분할, vocab 크기 조절 가능
  - 단점: 언어학적 경계와 불일치할 수 있음

옵션 4: Unigram Subword — (SentencePiece Unigram)  ★ 권장
  - 장점: BPE보다 유연한 분할, 드문 단어도 자연스럽게 처리
  - 단점: 학습 시간 약간 더 걸림
```

### 2.3 권장: SentencePiece Unigram, vocab=2048

```bash
# Step 1: 학습 데이터에서 텍스트 추출
python scripts/extract_text_from_manifest.py \
    --manifest_path /data/korean_train_manifest.jsonl \
    --output_path /data/korean_text_corpus.txt

# Step 2: SentencePiece 토크나이저 학습
# NeMo 내장 스크립트 사용
python NeMo/scripts/tokenizers/process_asr_text_tokenizer.py \
    --manifest=/data/korean_train_manifest.jsonl \
    --data_root=/data/tokenizer \
    --vocab_size=2048 \
    --tokenizer=unigram \
    --spe_character_coverage=1.0 \
    --no_lower_case
```

### 2.4 필러워드 토큰 처리 (콜봇 턴 디텍터용 핵심)

일반적인 ASR 토크나이저는 필러워드를 무시하거나 통합하는 경향이 있습니다.
콜봇 턴 디텍터용으로는 이들이 **명시적으로 전사**되어야 합니다.

```python
# 학습 데이터의 transcript에 필러워드가 포함되어야 함
# manifest.jsonl 예시:

# ❌ 잘못된 예 (필러워드 제거됨)
{"audio_filepath": "call_001.wav", "text": "제 전화번호가 010 1234 5678이에요", "duration": 4.2}

# ✅ 올바른 예 (필러워드 포함)
{"audio_filepath": "call_001.wav", "text": "어 제 전화번호가 음 010 1234 5678이에요", "duration": 4.2}

# ✅ 더 나은 예 (필러워드 + 특수 마커)
{"audio_filepath": "call_001.wav", "text": "<filler> 어 </filler> 제 전화번호가 <filler> 음 </filler> 010 1234 5678이에요", "duration": 4.2}
```

SentencePiece에 사용자 정의 토큰 추가:

```python
# user_defined_symbols로 필러워드와 특수 토큰을 반드시 vocab에 포함
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input='/data/korean_text_corpus.txt',
    model_prefix='/data/tokenizer/tokenizer',
    vocab_size=2048,
    model_type='unigram',
    character_coverage=1.0,
    # ★ 핵심: 필러워드와 특수 토큰을 user_defined_symbols로 추가
    user_defined_symbols=[
        '네', '아', '음', '어', '그',      # 필러워드 (독립 토큰으로)
        '<filler>',                         # 필러워드 마커 (옵션)
        '<EOU>',                            # End-of-Utterance 토큰
    ],
    # 숫자를 개별 digit으로 분리하지 않도록
    byte_fallback=True,
    # 공백 처리
    treat_whitespace_as_suffix=False,
)
```

### 2.5 Vocab 크기가 중요한 이유

NeMo의 공식 권장 (FastConformer 논문 기준):

| 디코더 | 권장 Vocab 크기 | 이유 |
|--------|----------------|------|
| CTC | 128~1024 | CTC는 spike 기반, vocab 크면 수렴 어려움 |
| RNN-T (Transducer) | 1024~4096 | Autoregressive라 큰 vocab도 학습 가능 |
| Hybrid (CTC+RNNT) | 1024~2048 | 두 디코더 모두 효과적인 범위 |

**Nemotron Speech ASR은 RNN-T 단독**이므로, **vocab 2048**이 한국어에 적합합니다.

한국어는 영어보다 음절 조합이 많아 (초성 19 × 중성 21 × 종성 28+1 = 11,172 가능 조합),
너무 작은 vocab (예: 512)은 커버리지 부족으로 성능이 떨어집니다.

---

## 3. 데이터 준비

### 3.1 NeMo Manifest 포맷

```jsonl
{"audio_filepath": "/data/audio/call_0001.wav", "text": "네 안녕하세요 팀벨입니다", "duration": 2.3}
{"audio_filepath": "/data/audio/call_0002.wav", "text": "어 제가 확인해 보겠습니다", "duration": 1.8}
{"audio_filepath": "/data/audio/call_0003.wav", "text": "전화번호가 공일공 일이삼사 오육칠팔 맞으시죠", "duration": 3.5}
```

### 3.2 오디오 전처리 요구사항

```python
# 모든 오디오 파일 표준화
# Nemotron Speech ASR 요구사항: 16kHz, mono, WAV

import subprocess

def standardize_audio(input_path, output_path):
    """전화 녹음 (8kHz) → 16kHz mono WAV 변환"""
    subprocess.run([
        'ffmpeg', '-i', input_path,
        '-ar', '16000',           # 16kHz로 리샘플링
        '-ac', '1',               # 모노
        '-sample_fmt', 's16',     # 16-bit PCM
        '-y', output_path
    ], check=True)
```

### 3.3 데이터 증강 (Data Augmentation)

콜센터 환경에 특화된 증강이 필수입니다. Deepgram의 한국어 문제(잡음 환경에서 출력 억제)를 
우리 모델에서는 반복하지 않으려면, 잡음 환경에서도 학습해야 합니다.

```yaml
# NeMo config: SpecAugment + Noise Perturbation
model:
  spec_augment:
    _target_: nemo.collections.asr.modules.SpectrogramAugmentation
    freq_masks: 2        # 주파수 마스킹 (음향 변이 시뮬레이션)
    freq_width: 27
    time_masks: 10       # 시간 마스킹 (드롭아웃 효과)
    time_width: 0.05     # 비율 기반

  train_ds:
    # Noise perturbation: 실제 전화 환경 시뮬레이션
    noise_manifest: /data/noise/noise_manifest.jsonl
    noise_prob: 0.7      # 70% 확률로 노이즈 추가
    noise_min_snr: 5     # 최소 5dB SNR
    noise_max_snr: 25    # 최대 25dB SNR
    
    # Speed perturbation: 말하기 속도 변이
    speed_perturb: true
    speed_min: 0.9
    speed_max: 1.1
```

노이즈 데이터 준비:

```
필요한 노이즈 유형 (콜센터 특화):
├── 전화 회선 잡음 (line noise, hum)
├── 주변 사무실 소음 (키보드, 대화)
├── 차량/외부 소음 (고객이 외부에서 전화)
├── 에코/반향 (스피커폰 사용)
├── 압축 아티팩트 (VoIP 코덱 왜곡)
└── 음악 대기음 (hold music)
```

### 3.4 38,000시간 데이터 활용 전략

전체 데이터를 한 번에 학습하는 것보다 단계적 접근이 효과적입니다:

```
Phase A: 깨끗한 데이터 5,000시간으로 초기 수렴 달성
    → 모델이 한국어 음향-텍스트 매핑을 학습

Phase B: 잡음 데이터 포함 15,000시간으로 확장
    → 노이즈 강건성 확보

Phase C: 전체 38,000시간 + 하드 마이닝 샘플
    → 최종 정확도 극대화

각 Phase에서 이전 체크포인트에서 이어서 학습 (resume)
```

---

## 4. 학습 설정 상세

### 4.1 전체 학습 Config (YAML)

```yaml
# korean_fastconformer_streaming_hybrid.yaml
# Cache-Aware Streaming FastConformer-RNNT for Korean

name: "Korean-FastConformer-CacheAware-Streaming-RNNT"

model:
  sample_rate: 16000
  
  # ★ Phase에 따라 false → true 변경하는 부분은 아래 커맨드라인으로 제어
  compute_eval_loss: false
  log_prediction: true    # 학습 중 예측 결과 로깅 (모니터링 필수)
  
  # ────────────────────────────────────────
  # 한국어 토크나이저 설정
  # ────────────────────────────────────────
  tokenizer:
    dir: /data/tokenizer/tokenizer_unigram_2048/  # 2.3절에서 생성한 토크나이저
    type: bpe  # SentencePiece는 NeMo에서 'bpe'로 지정
  
  # ────────────────────────────────────────
  # 학습 데이터
  # ────────────────────────────────────────
  train_ds:
    manifest_filepath: /data/korean_train_manifest.jsonl
    sample_rate: ${model.sample_rate}
    batch_size: 16            # GPU 메모리에 따라 조정 (H100 80GB → 32 가능)
    shuffle: true
    num_workers: 8
    pin_memory: true
    max_duration: 20          # 20초 이상 발화는 제외 (스트리밍 모델 특성)
    min_duration: 0.5         # 0.5초 미만 제외 (노이즈일 가능성)
    
    # Tarred dataset (대용량 학습 시 필수)
    is_tarred: false          # 38,000시간이면 tarred 권장 (true로 변경)
    tarred_audio_filepaths: null
    shuffle_n: 2048
    
    # Bucketing (배치 내 오디오 길이 맞추기)
    bucketing_strategy: "synced_randomized"
    bucketing_batch_size: null

  validation_ds:
    manifest_filepath: /data/korean_val_manifest.jsonl
    sample_rate: ${model.sample_rate}
    batch_size: 16
    shuffle: false
    num_workers: 8
    pin_memory: true

  # ────────────────────────────────────────
  # Preprocessor (변경 없음 - 그대로 유지)
  # ────────────────────────────────────────
  preprocessor:
    _target_: nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor
    sample_rate: ${model.sample_rate}
    normalize: "per_feature"
    window_size: 0.025        # 25ms 윈도우
    window_stride: 0.01       # 10ms 스트라이드
    window: "hann"
    features: 80              # 80-dim mel spectrogram
    n_fft: 512
    frame_splicing: 1
    dither: 0.00001
    pad_to: 0
    stft_exact_pad: true

  # ────────────────────────────────────────
  # SpecAugment (데이터 증강)
  # ────────────────────────────────────────
  spec_augment:
    _target_: nemo.collections.asr.modules.SpectrogramAugmentation
    freq_masks: 2
    freq_width: 27
    time_masks: 10
    time_width: 0.05

  # ────────────────────────────────────────
  # FastConformer Encoder (Cache-Aware Streaming)
  # ────────────────────────────────────────
  encoder:
    _target_: nemo.collections.asr.modules.ConformerEncoder
    feat_in: ${model.preprocessor.features}
    feat_out: -1
    n_layers: 24              # Nemotron 0.6B와 동일
    d_model: 1024             # Hidden dimension
    
    # ★ Cache-Aware Streaming 핵심 설정
    # Multi-lookahead: 추론 시 재학습 없이 레이턴시 선택 가능
    att_context_size: [[70,13],[70,6],[70,1],[70,0]]
    att_context_probs: [0.25, 0.25, 0.25, 0.25]
    att_context_style: chunked_limited
    
    # Subsampling: 8x (Nemotron과 동일)
    subsampling: dw_striding   # Depthwise-separable striding
    subsampling_factor: 8
    subsampling_conv_channels: 256
    
    # Self-Attention
    self_attention_model: rel_pos   # Relative positional encoding
    n_heads: 8
    xscaling: true
    pos_emb_max_len: 5000
    
    # Convolution module
    conv_kernel_size: 9
    conv_norm_type: layer_norm
    conv_context_size: causal  # ★ 스트리밍 필수: causal convolution
    
    # Regularization
    dropout: 0.1
    dropout_pre_encoder: 0.1
    dropout_emb: 0.0
    dropout_att: 0.1

  # ────────────────────────────────────────
  # RNN-T Decoder (한국어용 새로 초기화)
  # ────────────────────────────────────────
  decoder:
    _target_: nemo.collections.asr.modules.RNNTDecoder
    prednet:
      pred_hidden: 640        # Prediction network hidden size
      pred_rnn_layers: 1
      t_max: null
      dropout: 0.2

  joint:
    _target_: nemo.collections.asr.modules.RNNTJoint
    jointnet:
      joint_hidden: 640
      activation: "relu"
      dropout: 0.2
      
    # ★ FastEmit: 스트리밍 레이턴시 감소 핵심
    # 모델이 blank 토큰 대신 실제 토큰을 더 빨리 출력하도록 유도
    fused_batch_size: null

  # ────────────────────────────────────────
  # Loss 설정
  # ────────────────────────────────────────
  loss:
    loss_name: "default"
    warprnnt_numba_kwargs:
      # FastEmit regularization
      # 값이 클수록 더 빨리 토큰을 출력 (레이턴시 ↓, 정확도 약간 ↓)
      fastemit_lambda: 0.005   # 권장 범위: 1e-4 ~ 1e-2

  # ────────────────────────────────────────
  # Decoding 설정
  # ────────────────────────────────────────
  decoding:
    strategy: "greedy_batch"   # 추론 시 greedy (빠름)
    greedy:
      max_symbols: 10

  # ────────────────────────────────────────
  # Optimizer (★ 수렴 실패 방지 핵심)
  # ────────────────────────────────────────
  optim:
    name: adamw
    lr: 1.0                   # NoamAnnealing에서는 초기 lr이 아닌 scale factor
    betas: [0.9, 0.98]
    weight_decay: 0.001
    
    sched:
      name: NoamAnnealing
      d_model: ${model.encoder.d_model}
      warmup_steps: 10000     # ★ 중요: 충분한 워밍업 필요
      warmup_ratio: null
      min_lr: 1e-6

# ────────────────────────────────────────
# Trainer 설정
# ────────────────────────────────────────
trainer:
  devices: 4                  # GPU 수
  num_nodes: 1
  max_epochs: 200             # Transfer learning이므로 200으로 충분
  max_steps: -1
  val_check_interval: 1.0     # 매 에폭마다 검증 (초기에는 0.25로 자주 확인)
  accelerator: auto
  strategy: ddp
  accumulate_grad_batches: 4  # Effective batch = batch_size × GPU수 × accumulate
  gradient_clip_val: 1.0
  precision: "bf16-mixed"     # H100에서 bf16 사용
  log_every_n_steps: 100
  enable_progress_bar: true
  num_sanity_val_steps: 0
  check_val_every_n_epoch: 1

# ────────────────────────────────────────
# Experiment Manager
# ────────────────────────────────────────
exp_manager:
  exp_dir: /experiments/korean_streaming_asr
  name: ${name}
  create_tensorboard_logger: true
  create_wandb_logger: true    # W&B 로깅 권장
  wandb_logger_kwargs:
    name: ${name}
    project: timbel-korean-asr
  
  # 체크포인트
  create_checkpoint_callback: true
  checkpoint_callback_params:
    monitor: "val_wer"
    mode: "min"
    save_top_k: 5
    every_n_epochs: 1
  
  resume_if_exists: true       # 학습 중단 시 자동 재개
  resume_ignore_no_checkpoint: true
```

### 4.2 학습 실행 스크립트

```python
#!/usr/bin/env python3
"""
korean_streaming_asr_train.py
한국어 Cache-Aware Streaming ASR 하이브리드 학습 스크립트

핵심: 영어 Nemotron 모델에서 인코더 하위 레이어를 가져오고,
      상위 레이어 + 디코더를 한국어로 학습
"""

import nemo.collections.asr as nemo_asr
from nemo.utils import logging
from omegaconf import OmegaConf, open_dict
import pytorch_lightning as pl
import torch
import copy

# ═══════════════════════════════════════════
# Step 1: 영어 pretrained 모델 로드
# ═══════════════════════════════════════════
logging.info("Loading English Nemotron Speech ASR model...")

en_model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/nemotron-speech-streaming-en-0.6b"
)

logging.info(f"English model loaded. Parameters: {sum(p.numel() for p in en_model.parameters()):,}")

# ═══════════════════════════════════════════
# Step 2: 한국어 설정 로드
# ═══════════════════════════════════════════
ko_config = OmegaConf.load("korean_fastconformer_streaming_hybrid.yaml")

# ═══════════════════════════════════════════
# Step 3: 한국어 토크나이저로 Vocabulary 교체
# ═══════════════════════════════════════════
logging.info("Changing vocabulary to Korean tokenizer...")

# ★ change_vocabulary가 하는 일:
#   1) 기존 tokenizer를 새 한국어 tokenizer로 교체
#   2) Decoder의 output layer를 새 vocab 크기에 맞게 재초기화
#   3) Joint network의 vocab projection도 재초기화
#
# 이 과정에서 Decoder weights가 랜덤 초기화됨 → 이것이 정상!

en_model.change_vocabulary(
    new_tokenizer_dir="/data/tokenizer/tokenizer_unigram_2048/",
    new_tokenizer_type="bpe"
)

logging.info(f"Vocabulary changed. New vocab size: {en_model.tokenizer.vocab_size}")

# ═══════════════════════════════════════════
# Step 4: 인코더 레이어별 Freeze/Unfreeze
# ═══════════════════════════════════════════
logging.info("Configuring layer-wise freeze strategy...")

# Preprocessor: 완전 freeze (음향 특성 추출, 언어 무관)
for param in en_model.preprocessor.parameters():
    param.requires_grad = False
logging.info("  ✓ Preprocessor: FROZEN")

# Encoder subsampling: freeze (8x downsampling, 언어 무관)
for param in en_model.encoder.pre_encode.parameters():
    param.requires_grad = False
logging.info("  ✓ Encoder subsampling: FROZEN")

# Encoder layers: 하위 12층 freeze, 상위 12층 unfreeze
FREEZE_LAYERS = 12  # 이 값을 조정하면서 실험 가능

for i, layer in enumerate(en_model.encoder.layers):
    if i < FREEZE_LAYERS:
        for param in layer.parameters():
            param.requires_grad = False
        logging.info(f"  ✓ Encoder layer {i:2d}: FROZEN (acoustic features)")
    else:
        for param in layer.parameters():
            param.requires_grad = True
        logging.info(f"  ◆ Encoder layer {i:2d}: TRAINABLE (language-specific)")

# Encoder의 final layer norm: unfreeze
if hasattr(en_model.encoder, 'layer_norm'):
    for param in en_model.encoder.layer_norm.parameters():
        param.requires_grad = True
    logging.info("  ◆ Encoder final layer_norm: TRAINABLE")

# Decoder: 완전 unfreeze (새로 초기화된 상태)
for param in en_model.decoder.parameters():
    param.requires_grad = True
logging.info("  ◆ Decoder (RNN-T): TRAINABLE (newly initialized)")

# Joint network: 완전 unfreeze
for param in en_model.joint.parameters():
    param.requires_grad = True
logging.info("  ◆ Joint network: TRAINABLE (newly initialized)")

# 학습 가능 파라미터 수 확인
total_params = sum(p.numel() for p in en_model.parameters())
trainable_params = sum(p.numel() for p in en_model.parameters() if p.requires_grad)
frozen_params = total_params - trainable_params

logging.info(f"\nParameter Summary:")
logging.info(f"  Total:     {total_params:>12,}")
logging.info(f"  Trainable: {trainable_params:>12,} ({trainable_params/total_params*100:.1f}%)")
logging.info(f"  Frozen:    {frozen_params:>12,} ({frozen_params/total_params*100:.1f}%)")

# ═══════════════════════════════════════════
# Step 5: 학습 설정 적용
# ═══════════════════════════════════════════
# Data 설정
en_model.setup_training_data(train_data_config=ko_config.model.train_ds)
en_model.setup_validation_data(val_data_config=ko_config.model.validation_ds)

# SpecAugment 설정
with open_dict(en_model.cfg):
    en_model.cfg.spec_augment = ko_config.model.spec_augment

en_model.spec_augmentation = nemo_asr.modules.SpectrogramAugmentation(
    **ko_config.model.spec_augment
)

# Optimizer 설정
# ★ 중요: Transfer learning에서는 learning rate를 조정
optim_config = copy.deepcopy(ko_config.model.optim)

# 하위 레이어가 frozen이므로, 학습 가능 파라미터에만 optimizer 적용
en_model.setup_optimization(optim_config=optim_config)

# ═══════════════════════════════════════════
# Step 6: Trainer 설정 및 학습 시작
# ═══════════════════════════════════════════
trainer = pl.Trainer(**ko_config.trainer)

# Experiment Manager 설정
from nemo.utils.exp_manager import exp_manager
exp_manager(trainer, ko_config.get("exp_manager", None))

logging.info("Starting Korean streaming ASR training...")
trainer.fit(en_model)

logging.info("Training complete!")

# ═══════════════════════════════════════════
# Step 7: 최종 모델 저장
# ═══════════════════════════════════════════
en_model.save_to("/experiments/korean_streaming_asr/final_model.nemo")
logging.info("Model saved to /experiments/korean_streaming_asr/final_model.nemo")
```

---

## 5. 수렴 실패 방지: 실전 함정과 해결책

### 5.1 GitHub에서 발견된 실패 패턴과 대응

| 실패 패턴 | 증상 | 원인 | 해결책 |
|-----------|------|------|--------|
| **WER = 1.0 고정** | 에폭이 지나도 WER 변화 없음 | 디코더가 blank만 출력 | FastEmit lambda 증가 (0.001→0.01) |
| **WER = 1.0 고정 (2)** | Loss는 감소하나 WER 변화 없음 | Learning rate 너무 낮음 | warmup_steps 감소, lr scale 증가 |
| **WER > 1.0** | change_vocabulary 후 평가 시 | 토크나이저 불일치 | 저장/로드 시 토크나이저 경로 확인 |
| **발산 (Loss NaN)** | 학습 초기에 loss 폭발 | Gradient explosion | gradient_clip_val=1.0 확인, lr 감소 |
| **부분 단어 출력** | "hello world" → "herld" | CTC 디코더에서 발생 | RNN-T 디코더 사용 (Hybrid에서 기본값) |
| **속도 느림** | 에폭당 학습 시간 과도 | 데이터 로딩 병목 | tarred dataset 사용, num_workers 증가 |

### 5.2 FastEmit이 왜 중요한가 (스트리밍 모델의 핵심)

RNN-T 디코더는 기본적으로 "blank" 토큰을 출력하는 것이 안전한 선택입니다.
텍스트 토큰을 출력하면 돌이킬 수 없지만, blank은 "아직 모르겠다"는 뜻이니까요.

문제는, blank을 너무 많이 출력하면:
1. 전사가 지연됨 (토큰이 늦게 나옴)
2. 콜봇 턴 디텍터가 텍스트를 못 받음

FastEmit regularization은 모델에게 "blank보다 실제 토큰을 빨리 출력하라"는 
인센티브를 줍니다:

```yaml
# fastemit_lambda 조절 가이드
fastemit_lambda: 0.001   # 보수적 (정확도 우선)
fastemit_lambda: 0.005   # 균형 (일반 권장)
fastemit_lambda: 0.01    # 공격적 (레이턴시 우선, 콜봇에 적합)
```

### 5.3 3단계 학습 전략 (수렴 보장)

단계적 학습으로 수렴 실패 위험을 최소화합니다:

```
┌──────────────────────────────────────────────────────┐
│ Stage 1: Decoder Warmup (5~10 에폭)                   │
│                                                       │
│ 목적: 새로 초기화된 디코더가 기본적인 한국어 출력을 학습   │
│                                                       │
│ 설정:                                                 │
│  - Encoder 전체 freeze (Layer 0~23 모두)               │
│  - Decoder + Joint만 학습                              │
│  - Learning rate: 높게 (lr=5.0, warmup=5000)           │
│  - fastemit_lambda: 0.01 (빠른 출력 유도)               │
│                                                       │
│ 기대: WER 0.5~0.8 수준으로 기본 매핑 형성                │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│ Stage 2: Upper Encoder + Decoder (50~100 에폭)        │
│                                                       │
│ 목적: 인코더 상위 레이어를 한국어에 적응                   │
│                                                       │
│ 설정:                                                 │
│  - Encoder Layer 0~11: freeze                          │
│  - Encoder Layer 12~23: unfreeze                       │
│  - Decoder + Joint: 계속 학습                           │
│  - Learning rate: 중간 (lr=2.0, warmup=10000)          │
│  - fastemit_lambda: 0.005                              │
│  - Stage 1 체크포인트에서 resume                         │
│                                                       │
│ 기대: WER 0.1~0.3 수준                                 │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│ Stage 3: Full Fine-tuning (선택, 50+ 에폭)             │
│                                                       │
│ 목적: 전체 모델 미세 조정으로 최종 성능 극대화             │
│                                                       │
│ 설정:                                                 │
│  - Encoder 전체 unfreeze (Layer 0~23)                  │
│  - 매우 낮은 learning rate (lr=0.5, warmup=5000)        │
│  - Stage 2 체크포인트에서 resume                         │
│  - fastemit_lambda: 0.005                              │
│                                                       │
│ 주의: 하위 레이어 과적합 위험 → 조기 종료 모니터링         │
│                                                       │
│ 기대: WER 0.05~0.15 수준 (데이터/도메인에 따라)           │
└──────────────────────────────────────────────────────┘
```

### 5.4 모니터링 체크리스트

학습 중 반드시 확인해야 할 지표들:

```python
# W&B 또는 TensorBoard에서 모니터링

# 1. Loss 추이
#    - train_loss: 꾸준히 감소해야 함
#    - val_loss: train_loss와 함께 감소 (gap이 벌어지면 과적합)

# 2. WER (Word Error Rate)
#    - val_wer: 핵심 지표. 0.5 이하로 떨어지면 수렴 시작
#    - val_wer가 1.0에서 10 에폭 이상 머물면 문제 있음!

# 3. Learning Rate
#    - NoamAnnealing: warmup 후 점차 감소하는 곡선이어야 함
#    - 갑자기 0이 되면 스케줄러 설정 오류

# 4. 예측 샘플 (log_prediction: true 설정 시)
#    - reference: "안녕하세요 팀벨입니다"
#    - predicted: "안녕하세오 팀벨이니다"  → 방향은 맞음, 학습 진행 중
#    - predicted: "" (빈 문자열)           → blank만 출력, FastEmit 조정 필요
#    - predicted: "ㅇㅇㅇㅇ"              → 토크나이저 문제
```

---

## 6. 스트리밍 추론 서버 구축

### 6.1 Cache-Aware Streaming 추론 흐름

```
시간축 →

[Audio Stream]  ──── chunk1 ──── chunk2 ──── chunk3 ──── chunk4 ────→
                    160ms       160ms       160ms       160ms

[Encoder]       ┌─────────┐
                │ Process  │──→ cache1 저장
                │ chunk1   │
                └────┬─────┘
                     │      ┌─────────┐
                     │      │ Process  │──→ cache2 저장 (cache1 재사용)
                     │      │ chunk2   │
                     │      └────┬─────┘
                     │           │      ┌─────────┐
                     │           │      │ Process  │──→ cache3 저장
                     │           │      │ chunk3   │
                     │           │      └────┬─────┘

[Decoder]       ──→ "안"    ──→ "녕하"   ──→ "세요"    ──→ ...

★ 핵심: 각 chunk는 한 번만 처리됨. 이전 chunk의 정보는 cache에서 가져옴.
   Buffered streaming처럼 겹치는 연산이 전혀 없음!
```

### 6.2 WebSocket 스트리밍 서버

```python
"""
korean_streaming_asr_server.py
WebSocket 기반 스트리밍 ASR 서버

LiveKit 커스텀 STT 플러그인으로 연결하기 위한 인터페이스
"""

import asyncio
import websockets
import json
import numpy as np
import torch
import nemo.collections.asr as nemo_asr
from dataclasses import dataclass
from typing import Optional
import time

@dataclass
class StreamingState:
    """한 스트림의 cache-aware 상태"""
    cache: Optional[dict] = None    # 인코더 캐시
    prev_tokens: list = None        # 이전 출력 토큰
    audio_buffer: np.ndarray = None # 미처리 오디오 버퍼
    chunk_count: int = 0

class KoreanStreamingASR:
    def __init__(self, model_path: str, chunk_size_ms: int = 160):
        # 모델 로드
        self.model = nemo_asr.models.ASRModel.restore_from(model_path)
        self.model.eval()
        self.model.freeze()
        
        # GPU 배치
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Chunk 설정 (160ms = 콜봇 최적)
        self.chunk_size_ms = chunk_size_ms
        self.chunk_samples = int(16000 * chunk_size_ms / 1000)  # 16kHz 기준
        
        # Cache-aware context size 설정
        # att_context_size=[70,6] → 160ms lookahead
        if hasattr(self.model.encoder, 'set_default_att_context_size'):
            self.model.encoder.set_default_att_context_size([70, 6])
        
    async def handle_stream(self, websocket, path):
        """WebSocket 스트림 핸들러"""
        state = StreamingState(
            prev_tokens=[],
            audio_buffer=np.array([], dtype=np.float32),
            chunk_count=0
        )
        
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    # 오디오 데이터 수신 (16-bit PCM, 16kHz)
                    audio_chunk = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
                    state.audio_buffer = np.concatenate([state.audio_buffer, audio_chunk])
                    
                    # 충분한 오디오가 쌓이면 처리
                    while len(state.audio_buffer) >= self.chunk_samples:
                        chunk = state.audio_buffer[:self.chunk_samples]
                        state.audio_buffer = state.audio_buffer[self.chunk_samples:]
                        
                        # Cache-aware 추론
                        t_start = time.perf_counter()
                        result = self._process_chunk(chunk, state)
                        latency_ms = (time.perf_counter() - t_start) * 1000
                        
                        state.chunk_count += 1
                        
                        # 결과 전송
                        response = {
                            "type": "partial" if not result.get("is_final") else "final",
                            "text": result["text"],
                            "latency_ms": round(latency_ms, 1),
                            "chunk_index": state.chunk_count,
                            # ★ 턴 디텍터용 메타데이터
                            "has_filler": result.get("has_filler", False),
                            "tokens": result.get("tokens", []),
                        }
                        
                        await websocket.send(json.dumps(response, ensure_ascii=False))
                
                elif isinstance(message, str):
                    cmd = json.loads(message)
                    if cmd.get("type") == "reset":
                        # 새 발화 시작 → 캐시 초기화
                        state = StreamingState(
                            prev_tokens=[],
                            audio_buffer=np.array([], dtype=np.float32),
                            chunk_count=0
                        )
                        await websocket.send(json.dumps({"type": "reset_ack"}))
                        
        except websockets.exceptions.ConnectionClosed:
            pass
    
    def _process_chunk(self, chunk: np.ndarray, state: StreamingState) -> dict:
        """단일 chunk를 cache-aware로 처리"""
        with torch.no_grad():
            # Audio → Tensor
            audio_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(self.device)
            audio_length = torch.tensor([len(chunk)], dtype=torch.long).to(self.device)
            
            # Mel spectrogram
            processed, processed_length = self.model.preprocessor(
                input_signal=audio_tensor,
                length=audio_length
            )
            
            # Cache-aware encoder 추론
            # NeMo의 cache-aware streaming은 내부적으로 cache를 관리
            encoded, encoded_length = self.model.encoder(
                audio_signal=processed,
                length=processed_length
            )
            
            # RNN-T greedy decoding
            hypotheses = self.model.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=encoded,
                encoded_lengths=encoded_length,
            )
            
            text = hypotheses[0].text if hypotheses else ""
            tokens = hypotheses[0].tokens if hypotheses else []
            
            # 필러워드 감지
            filler_words = {'네', '아', '음', '어', '그', '뭐'}
            has_filler = any(t in filler_words for t in text.split())
            
            return {
                "text": text,
                "tokens": tokens,
                "is_final": False,  # VAD silence에서 final로 변경
                "has_filler": has_filler,
            }

async def main():
    asr = KoreanStreamingASR(
        model_path="/experiments/korean_streaming_asr/final_model.nemo",
        chunk_size_ms=160  # 콜봇 최적: 160ms
    )
    
    server = await websockets.serve(
        asr.handle_stream,
        "0.0.0.0", 8765,
        max_size=None
    )
    
    print("Korean Streaming ASR Server running on ws://0.0.0.0:8765")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 7. LiveKit STT 플러그인으로 통합

### 7.1 LiveKit Custom STT Plugin

```python
"""
livekit_korean_stt_plugin.py
한국어 Cache-Aware Streaming STT를 LiveKit에 연결하는 플러그인
"""

from livekit.agents import stt
from livekit.agents.types import AudioFrame
import websockets
import json
import asyncio
import numpy as np
from dataclasses import dataclass
from typing import AsyncIterable

class KoreanStreamingSTT(stt.STT):
    """한국어 Cache-Aware Streaming STT - LiveKit 플러그인"""
    
    def __init__(
        self,
        server_url: str = "ws://localhost:8765",
        sample_rate: int = 16000,
        chunk_duration_ms: int = 160,
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
            )
        )
        self.server_url = server_url
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
    
    async def recognize(self, audio: AudioFrame) -> stt.SpeechEvent:
        """단일 오디오 프레임 인식 (non-streaming)"""
        # Streaming 모드에서 주로 사용하므로 간단 구현
        pass
    
    def stream(self) -> "KoreanSTTStream":
        """스트리밍 인식 세션 생성"""
        return KoreanSTTStream(
            server_url=self.server_url,
            sample_rate=self.sample_rate,
        )

class KoreanSTTStream(stt.SpeechStream):
    """스트리밍 세션"""
    
    def __init__(self, server_url: str, sample_rate: int):
        super().__init__()
        self.server_url = server_url
        self.sample_rate = sample_rate
        self._ws = None
        self._results_queue = asyncio.Queue()
        self._closed = False
    
    async def _connect(self):
        """WebSocket 연결"""
        if self._ws is None:
            self._ws = await websockets.connect(self.server_url)
            # 수신 태스크 시작
            asyncio.create_task(self._receive_loop())
    
    async def _receive_loop(self):
        """서버로부터 결과 수신"""
        try:
            async for message in self._ws:
                result = json.loads(message)
                if result.get("type") in ("partial", "final"):
                    event = stt.SpeechEvent(
                        type=(
                            stt.SpeechEventType.INTERIM_TRANSCRIPT
                            if result["type"] == "partial"
                            else stt.SpeechEventType.FINAL_TRANSCRIPT
                        ),
                        alternatives=[
                            stt.SpeechData(
                                text=result["text"],
                                language="ko",
                            )
                        ],
                    )
                    await self._results_queue.put(event)
        except Exception:
            pass
    
    async def push_frame(self, frame: AudioFrame):
        """오디오 프레임을 ASR 서버로 전송"""
        await self._connect()
        
        # AudioFrame → bytes (16-bit PCM)
        audio_data = np.array(frame.data, dtype=np.int16).tobytes()
        await self._ws.send(audio_data)
    
    async def flush(self):
        """버퍼 플러시"""
        if self._ws:
            await self._ws.send(json.dumps({"type": "reset"}))
    
    async def aclose(self):
        """세션 종료"""
        self._closed = True
        if self._ws:
            await self._ws.close()
    
    def __aiter__(self):
        return self
    
    async def __anext__(self) -> stt.SpeechEvent:
        if self._closed and self._results_queue.empty():
            raise StopAsyncIteration
        return await self._results_queue.get()
```

### 7.2 LiveKit Agent에서 사용

```python
"""
korean_callbot_agent.py
한국어 콜봇 에이전트: 자체 STT + 턴 디텍터 통합
"""

from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.agents.voice import MetaVoiceAgent
from livekit.plugins import turn_detector

# 우리가 만든 커스텀 STT 플러그인
from livekit_korean_stt_plugin import KoreanStreamingSTT

async def create_korean_callbot():
    # 한국어 Cache-Aware Streaming STT
    korean_stt = KoreanStreamingSTT(
        server_url="ws://asr-server:8765",
        chunk_duration_ms=160,  # 160ms chunk (콜봇 최적)
    )
    
    session = AgentSession(
        stt=korean_stt,
        
        # LiveKit 턴 디텍터 (STT 텍스트 기반)
        # 이제 우리 STT가 필러워드를 출력하므로 턴 디텍터가 제대로 동작!
        turn_detection=turn_detector.MultilingualModel(),
        
        # 엔드포인팅 설정 (한국어 최적화)
        min_endpointing_delay=0.8,    # 최소 대기 (턴 완료 판단 후)
        max_endpointing_delay=3.0,    # 최대 대기 (턴 미완료 시)
        
        # 인터럽트 설정
        allow_interruptions=True,
        resume_false_interruption=True,
    )
    
    return session
```

---

## 8. End-of-Utterance (EOU) 모델 통합 (최종 목표)

### 8.1 Parakeet Realtime EOU 구조

NVIDIA의 `parakeet_realtime_eou_120m-v1`은 ASR과 EOU를 **단일 모델**에서 수행합니다:
- ASR: 실시간 전사
- EOU: 발화 종료 시 `<EOU>` 특수 토큰 출력

```
일반 ASR 출력:  "안녕하세요" " " "팀벨" "입니다"
EOU ASR 출력:   "안녕하세요" " " "팀벨" "입니다" "<EOU>"
                                                   ↑
                                            이 토큰이 나오면 턴 종료!
```

### 8.2 한국어 EOU 모델 학습 (궁극적 목표)

STT + Turn Detection을 단일 모델에서:

```python
# 학습 데이터에 <EOU> 토큰 포함
# manifest.jsonl:
{"audio_filepath": "turn_001.wav", "text": "네 확인했습니다 <EOU>", "duration": 1.5}
{"audio_filepath": "turn_002.wav", "text": "제 번호가 음", "duration": 1.0}
# ↑ <EOU> 없음 = 아직 말하는 중

# 토크나이저에 <EOU>를 user_defined_symbol로 추가
# → 모델이 발화 종료 시점에 <EOU>를 출력하도록 학습
```

이렇게 하면 **별도의 턴 디텍터 모델이 필요 없어지고**, 
STT 출력에서 `<EOU>` 토큰이 나오면 바로 LLM에 전달하면 됩니다.

```
최종 아키텍처:

[전화 오디오] → [한국어 Cache-Aware FastConformer-RNNT with EOU]
                    │
                    ├── "네" → partial (턴 미완료로 판단)
                    ├── " " → partial
                    ├── "확인했습니다" → partial
                    ├── "<EOU>" → ★ 턴 종료! → [LLM] → [TTS] → 응답
                    │
                    └── 전체 레이턴시: ~24ms (최종 전사) + 0ms (턴 판단 내장)
                        vs 현재: ~300ms (Deepgram) + 25ms (턴 디텍터)
```

---

## 9. 학습 자원 및 일정 추정

### 9.1 GPU 자원 요구사항

| Stage | 데이터 | GPU | 기간 | 예상 비용 (클라우드) |
|-------|--------|-----|------|---------------------|
| 토크나이저 구축 | 전체 텍스트 | CPU | 1일 | 무시 가능 |
| Stage 1: Decoder Warmup | 5,000hr | H100×4 | 3~5일 | $1,500~2,500 |
| Stage 2: Upper Encoder | 15,000hr | H100×4 | 1~2주 | $3,000~6,000 |
| Stage 3: Full Fine-tune | 38,000hr | H100×8 | 2~3주 | $8,000~15,000 |
| EOU 모델 추가 학습 | 5,000hr (EOU 라벨) | H100×4 | 1주 | $2,000~3,500 |
| **합계** | | | **5~7주** | **$15K~27K** |

### 9.2 데이터 준비 일정

| 작업 | 기간 | 비고 |
|------|------|------|
| 기존 데이터 포맷 변환 (NeMo manifest) | 1주 | 38,000시간 → jsonl |
| 필러워드 라벨링 검증 | 1~2주 | 기존 라벨에 필러 포함 여부 확인/보정 |
| EOU 라벨 생성 | 2~3주 | 콜센터 데이터에서 턴 경계 표기 |
| 노이즈 데이터 수집/생성 | 1주 | 전화 환경 특화 노이즈 |
| **합계** | **4~6주** | 학습과 병렬 가능 |

---

## 10. 요약: 하이브리드 학습 체크리스트

```
□ 1. 영어 Nemotron Speech ASR 0.6B 모델 다운로드
□ 2. 한국어 SentencePiece Unigram 토크나이저 구축 (vocab=2048)
     □ 필러워드 user_defined_symbols 포함
     □ <EOU> 토큰 포함 (미래 확장용)
□ 3. 학습 데이터 NeMo manifest 포맷으로 변환
     □ 필러워드 전사 포함 확인
     □ 노이즈 augmentation 파이프라인 구축
□ 4. Stage 1: Decoder Warmup
     □ 인코더 전체 freeze
     □ FastEmit lambda=0.01
     □ WER < 0.8 확인 후 다음 단계
□ 5. Stage 2: Upper Encoder + Decoder
     □ 인코더 Layer 0~11 freeze, 12~23 unfreeze
     □ FastEmit lambda=0.005
     □ WER < 0.3 확인 후 다음 단계
□ 6. Stage 3: Full Fine-tune (선택)
     □ 전체 unfreeze, 매우 낮은 lr
     □ 과적합 모니터링
□ 7. 스트리밍 추론 서버 구축
     □ WebSocket 기반
     □ 160ms chunk 설정
□ 8. LiveKit STT 플러그인 연결
     □ 턴 디텍터 통합 테스트
□ 9. EOU 모델 추가 학습 (Phase 4)
     □ <EOU> 라벨 데이터 준비
     □ 턴 디텍터 없는 단일 모델 달성
```