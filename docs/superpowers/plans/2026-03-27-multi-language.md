# Multi-Language Support (한/영) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Brain S2S에 한국어/영어 자동 감지 추가 — 영어로 말하면 영어로 응답

**Architecture:** 서버 시작 시 한국어/영어 STT + LangID + 영어 TTS를 모두 로드. 자동 감지 모드에서 첫 발화 시 LangID로 언어 판별 → 해당 언어 STT/TTS로 라우팅. LLM은 Qwen3.5 다국어 지원으로 프롬프트만 전환.

**Tech Stack:** NeMo ASR, NeMo LangID (AmberNet), StyleTTS2, vLLM, FastAPI

**Spec:** `docs/superpowers/specs/2026-03-27-multi-language-design.md`

---

## File Structure

| 파일 | 변경 | 역할 |
|------|------|------|
| `realtime_demo/server.py` | Modify | 영어 STT + LangID 로드, 세션 언어 관리, 배치 라우팅 |
| `realtime_demo/s2s_pipeline.py` | Modify | 영어 TTS, 언어별 프롬프트/TTS 선택 |
| `realtime_demo/static/index.html` | Modify | 언어 모드 토글 + 감지 결과 표시 |

---

### Task 1: s2s_pipeline.py — 영어 TTS + 언어별 프롬프트

서버 변경 없이 독립적으로 테스트 가능. 영어 TTS 인스턴스를 추가하고 `process()`에 language 파라미터 추가.

**Files:**
- Modify: `realtime_demo/s2s_pipeline.py`

- [ ] **Step 1: 영어 TTS 경로 상수 + 영어 시스템 프롬프트 추가**

파일 상단 설정 영역에 추가:

```python
# 기존 TTS 설정 바로 뒤에:
TTS_REF_AUDIO_EN = "/home/jonghooy/work/zhisper/styletts2_korean/Models/LibriTTS_EN/reference_audio/reference_audio/4077-13754-0000.wav"
STYLETTS2_EN_CONFIG = "/home/jonghooy/work/zhisper/styletts2_korean/Models/LibriTTS_EN/Vocos/LibriTTS/config_libritts_vocos.yml"
STYLETTS2_EN_MODEL = "/home/jonghooy/work/zhisper/styletts2_korean/Models/LibriTTS_EN/Vocos/LibriTTS/epoch_2nd_00029.pth"
```

영어 시스템 프롬프트 (기존 SYSTEM_PROMPT 바로 뒤에):

```python
SYSTEM_PROMPT_EN = """You are a friendly and professional call center agent.

Absolute rules:
1. Start with [EMOTION:anger|anxiety|neutral|satisfaction|joy] tag on the first line.
2. Keep your response to 1-2 sentences maximum. Never exceed 2 sentences.
3. Use natural conversational English since this will be read by TTS.
4. Do not use markdown, asterisks, brackets or special formatting.

Emotion guidelines:
- anger: Apologize + immediate resolution
- anxiety: Reassure + guide
- neutral: Answer kindly
- satisfaction/joy: Thank + respond warmly"""
```

- [ ] **Step 2: S2SPipeline에 영어 TTS 인스턴스 추가**

`__init__`과 `load()`를 수정:

```python
def __init__(self, device: str = "cuda:0", knowledge_client=None):
    self.llm = LLMEngine()
    self.tts = TTSEngine(device=device)           # 한국어 TTS
    self.tts_en = TTSEngine(device=device, engine_type="styletts2",
                            config_path=STYLETTS2_EN_CONFIG,
                            model_path=STYLETTS2_EN_MODEL,
                            ref_audio_path=TTS_REF_AUDIO_EN)  # 영어 TTS
    self.knowledge_client = knowledge_client
    self._loaded = False

def load(self):
    if self._loaded:
        return
    self.tts.load()
    self.tts_en.load()
    self.llm.load()
    self._loaded = True
    logger.info("S2S Pipeline ready (LLM: vLLM API, TTS: KO+EN StyleTTS2)")
```

이를 위해 **TTSEngine에 config_path, model_path, ref_audio_path 파라미터 추가** 필요:

```python
class TTSEngine:
    def __init__(self, device: str = "cuda:0", engine_type: str = TTS_ENGINE_TYPE,
                 config_path: str = None, model_path: str = None, ref_audio_path: str = None):
        self.device = device
        self.engine_type = engine_type
        self._config_path = config_path
        self._model_path = model_path
        self._ref_audio_path = ref_audio_path
        self.engine = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        # ... edge tts 부분 동일 ...
        elif self.engine_type == "styletts2":
            import os
            if ZHISPER_TTS_PATH not in sys.path:
                sys.path.insert(0, ZHISPER_TTS_PATH)
            from tts.engine.styletts2_engine import StyleTTS2Engine
            config = self._config_path or os.path.join(STYLETTS2_PATH, "Models/KB_60h/config_ko_finetune_KB_60h.yml")
            model = self._model_path or os.path.join(STYLETTS2_PATH, "Models/KB_60h/epoch_2nd_00034.pth")
            ref = self._ref_audio_path or TTS_REF_AUDIO
            self.engine = StyleTTS2Engine(
                device=self.device,
                config_path=config,
                model_path=model,
                ref_audio_path=ref,
            )
            self.engine.load()
        self._loaded = True
```

- [ ] **Step 3: process()에 language 파라미터 추가**

```python
async def process(
    self,
    user_text: str,
    system_prompt: str = SYSTEM_PROMPT,
    cancel_event: asyncio.Event = None,
    audio_context: dict = None,
    history: list = None,
    language: str = "ko",        # ← 추가
) -> AsyncGenerator[dict, None]:
```

process() 내에서 언어에 따라 프롬프트와 TTS 선택:

시스템 프롬프트 선택 (Knowledge prompt 로딩 직전):
```python
# 언어별 기본 시스템 프롬프트
if language == "en":
    system_prompt = SYSTEM_PROMPT_EN
```

TTS 선택 (TTS 합성 호출 부분):
```python
# 기존:
pcm_bytes, sr = await loop.run_in_executor(
    None, self.tts.synthesize_to_pcm16, chunk_text
)

# 변경:
active_tts = self.tts_en if language == "en" else self.tts
pcm_bytes, sr = await loop.run_in_executor(
    None, active_tts.synthesize_to_pcm16, chunk_text
)
```

이 패턴을 TTS 호출이 있는 3곳에 모두 적용:
1. 절/문장 끝 감지 시 TTS (flush_delimiters)
2. 남은 버퍼 처리 시 TTS (remaining)

- [ ] **Step 4: Commit**

```bash
git add realtime_demo/s2s_pipeline.py
git commit -m "feat: add English TTS + language-aware S2S pipeline

- TTSEngine accepts custom config/model/ref paths
- S2SPipeline loads Korean + English TTS
- process() accepts language param for prompt/TTS routing
- English system prompt added

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: server.py — 영어 STT + LangID + 언어 라우팅

**Files:**
- Modify: `realtime_demo/server.py`

- [ ] **Step 1: 글로벌 상태 + 설정 추가**

설정 영역에 추가:
```python
MODEL_PATH_EN = "/home/jonghooy/work/timbel-asr-pilot/pretrained_models/nemotron-speech-streaming-en-0.6b.nemo"
```

글로벌 상태에 추가:
```python
model_en = None          # 영어 STT 모델
preprocessor_en = None   # 영어 STT 전처리기
streaming_cfg_en = None  # 영어 스트리밍 설정
lang_id_model = None     # 언어 감지 모델
```

기존 `model`을 `model_ko`로 의미를 명확히 하되, **코드 전체를 변경하면 리스크가 크므로** 기존 `model` 변수는 유지하고 `model_ko = model`로 별칭 추가.

- [ ] **Step 2: startup에서 영어 STT + LangID 로드**

기존 모델 로드 후 (warmup 전에):

```python
    # 영어 STT 모델 로드
    global model_en, preprocessor_en, streaming_cfg_en, lang_id_model

    logger.info(f"Loading English STT from {MODEL_PATH_EN}...")
    model_en = nemo_asr.models.ASRModel.restore_from(MODEL_PATH_EN, map_location="cuda:0")
    model_en.eval()
    model_en = model_en.to("cuda:0")

    # 영어 모델도 동일 디코딩 설정
    en_decoding_cfg = model_en.cfg.decoding
    with open_dict(en_decoding_cfg):
        en_decoding_cfg.greedy.loop_labels = True
        en_decoding_cfg.greedy.use_cuda_graph_decoder = False
    model_en.change_decoding_strategy(en_decoding_cfg)

    model_en.encoder.set_default_att_context_size(ATT_CONTEXT_SIZE)
    streaming_cfg_en = model_en.encoder.streaming_cfg

    # 영어 전처리기
    cfg_pre_en = copy.deepcopy(model_en._cfg.preprocessor)
    OmegaConf.set_struct(cfg_pre_en, False)
    cfg_pre_en.dither = 0.0
    cfg_pre_en.pad_to = 0
    preprocessor_en = model_en.from_config_dict(cfg_pre_en).to("cuda:0").eval()

    logger.info("English STT loaded")

    # LangID 모델 로드
    logger.info("Loading LangID model...")
    lang_id_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("langid_ambernet")
    lang_id_model = lang_id_model.to("cuda:0")
    lang_id_model.eval()
    logger.info("LangID model loaded (AmberNet, 107 languages)")
```

- [ ] **Step 3: StreamingSession에 언어 필드 추가**

```python
# 언어 설정
self.language_mode = "ko"       # "ko" (한국어 전용) | "auto" (자동 감지)
self.detected_language = "ko"   # 감지된 언어 ("ko" | "en")
self.lang_detected = False      # 언어 감지 완료 여부
```

`_init_caches()`를 언어별로:
```python
def _init_caches(self):
    """인코더 캐시 초기화 — detected_language에 따른 모델 사용"""
    stt_model = model_en if self.detected_language == "en" else model
    cache = stt_model.encoder.get_initial_cache_state(
        batch_size=1, dtype=torch.float32, device="cuda:0"
    )
    self.cache_last_channel = cache[0]
    self.cache_last_time = cache[1]
    self.cache_last_channel_len = cache[2]
```

- [ ] **Step 4: get_available_chunks()에서 언어별 전처리기 사용**

`get_available_chunks()` 내에서 mel 변환 시:
```python
# 기존:
mel, mel_len = preprocessor(input_signal=audio_tensor, length=audio_len)

# 변경:
active_preprocessor = preprocessor_en if self.detected_language == "en" else preprocessor
mel, mel_len = active_preprocessor(input_signal=audio_tensor, length=audio_len)
```

`run_streaming_step()`에서도:
```python
# 기존:
result = model.conformer_stream_step(...)

# 변경:
stt_model = model_en if self.detected_language == "en" else model
result = stt_model.conformer_stream_step(...)
```

텍스트 추출:
```python
# 기존:
text = model.tokenizer.ids_to_text(token_ids)

# 변경:
text = stt_model.tokenizer.ids_to_text(token_ids)
```

- [ ] **Step 5: _batched_stream_step()에서 언어별 그룹핑**

`_process_batch()`에서 step>0 세션을 언어별로 분리:

```python
# 기존:
if stepN_items:
    if len(stepN_items) > 1:
        logger.info(f"[Batch] Batched inference: {len(stepN_items)} sessions")
    texts = _batched_stream_step(stepN_items)

# 변경:
if stepN_items:
    # 언어별 그룹핑
    ko_items = [(s, m, l, idx) for s, m, l, idx in stepN_items if s.detected_language == "ko"]
    en_items = [(s, m, l, idx) for s, m, l, idx in stepN_items if s.detected_language == "en"]

    all_texts = {}
    if ko_items:
        if len(ko_items) > 1:
            logger.info(f"[Batch] KO inference: {len(ko_items)} sessions")
        texts = _batched_stream_step(ko_items, stt_model=model)
        for (s, _, _, _), text in zip(ko_items, texts):
            all_texts[id(s)] = text
    if en_items:
        if len(en_items) > 1:
            logger.info(f"[Batch] EN inference: {len(en_items)} sessions")
        texts = _batched_stream_step(en_items, stt_model=model_en)
        for (s, _, _, _), text in zip(en_items, texts):
            all_texts[id(s)] = text

    for (session, _, _, _) in stepN_items:
        session_texts[id(session)] = all_texts.get(id(session), "")
```

`_batched_stream_step()`에 `stt_model` 파라미터 추가:
```python
def _batched_stream_step(sessions_and_chunks, stt_model=None):
    if stt_model is None:
        stt_model = model
    # ... 내부에서 model → stt_model로 교체 (conformer_stream_step, tokenizer)
```

- [ ] **Step 6: WebSocket 핸들러 — 언어 감지 + 메시지 처리**

텍스트 메시지 처리부에 추가:
```python
# set_language_mode: UI 토글
if msg.get("type") == "set_language_mode":
    session.language_mode = msg.get("mode", "ko")
    if session.language_mode == "ko":
        session.detected_language = "ko"
        session.lang_detected = True
    else:
        session.lang_detected = False
    logger.info(f"[Lang] Mode set to: {session.language_mode}")
    continue

# set_language: 수동 언어 전환
if msg.get("type") == "set_language":
    new_lang = msg.get("language", "ko")
    if new_lang in ("ko", "en"):
        session.detected_language = new_lang
        session.lang_detected = True
        session._init_caches()
        session.reset_for_new_utterance()
        await websocket.send_json({"type": "language_changed", "language": new_lang})
        logger.info(f"[Lang] Manual switch to: {new_lang}")
    continue
```

첫 발화 감지 시 LangID 실행 (노이즈 보정 완료 후, `is_speaking=True` 처리 부근):
```python
# 언어 감지: 자동 모드에서 첫 발화 시 1회
if (session.language_mode == "auto" and not session.lang_detected
        and session.is_speaking and len(session.audio_buffer) >= SAMPLE_RATE):
    # LangID 실행
    import tempfile, soundfile as sf
    detect_audio = session.audio_buffer[-SAMPLE_RATE:]  # 최근 1초
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as f:
        sf.write(f.name, detect_audio, SAMPLE_RATE)
        detected = lang_id_model.get_label(f.name)
    lang = detected if detected in ("ko", "en") else "ko"
    session.detected_language = lang
    session.lang_detected = True

    if lang != "ko":
        # 영어 감지 → 캐시 재초기화 (영어 모델로)
        session._init_caches()
        session.audio_buffer = np.array([], dtype=np.float32)
        session.mel_buffer_idx = 0
        session.step = 0

    await websocket.send_json({"type": "language_detected", "language": lang})
    logger.info(f"[Lang] Detected: {lang}")
```

- [ ] **Step 7: _run_s2s()에 language 전달**

```python
# 기존:
async for event in s2s_pipeline.process(user_text, cancel_event=cancel_event,
                                         audio_context=audio_context, history=history):

# 변경:
async for event in s2s_pipeline.process(user_text, cancel_event=cancel_event,
                                         audio_context=audio_context, history=history,
                                         language=session.detected_language if session else "ko"):
```

`_run_s2s()` 시그니처에서 session은 이미 전달되므로, session.detected_language를 직접 사용.

- [ ] **Step 8: Commit**

```bash
git add realtime_demo/server.py
git commit -m "feat: add English STT + LangID + language routing

- Load English Nemotron 0.6B + AmberNet LangID at startup
- StreamingSession: language_mode, detected_language fields
- Auto-detect language on first speech (~1s audio)
- Route STT/TTS by detected language
- Batch inference groups by language
- set_language_mode, set_language WebSocket handlers

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: static/index.html — 언어 모드 UI

**Files:**
- Modify: `realtime_demo/static/index.html`

- [ ] **Step 1: 언어 토글 CSS**

기존 `#btn-reset-history:hover` 스타일 뒤에 추가:

```css
#lang-toggle {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.3rem 0.8rem;
    border-radius: 6px;
    border: 1px solid #334155;
    background: #1e293b;
    color: #94a3b8;
    font-size: 0.8rem;
    cursor: pointer;
}
#lang-toggle:hover { border-color: #38bdf8; }
#lang-badge {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    background: #1e3a5f;
    color: #60a5fa;
}
#lang-badge.ko { background: #1e3a5f; color: #60a5fa; }
#lang-badge.en { background: #1e3f1e; color: #4ade80; }
```

- [ ] **Step 2: 언어 토글 HTML**

기존 RESTART + 대화 리셋 버튼 줄 뒤에 추가:

```html
<div id="lang-toggle" onclick="toggleLanguageMode()">
    <span>언어:</span>
    <span id="lang-badge" class="ko">한국어</span>
    <span id="lang-mode-label">전용</span>
</div>
```

- [ ] **Step 3: JavaScript 함수**

```javascript
let languageMode = 'ko';  // 'ko' or 'auto'
let detectedLanguage = 'ko';

function toggleLanguageMode() {
    languageMode = languageMode === 'ko' ? 'auto' : 'ko';
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({type: 'set_language_mode', mode: languageMode}));
    }
    document.getElementById('lang-mode-label').textContent =
        languageMode === 'ko' ? '전용' : '자동';
    if (languageMode === 'ko') {
        updateLangBadge('ko');
    }
}

function updateLangBadge(lang) {
    detectedLanguage = lang;
    const badge = document.getElementById('lang-badge');
    badge.className = 'lang-badge ' + lang;
    badge.textContent = lang === 'ko' ? '한국어' : 'English';
}
```

- [ ] **Step 4: WebSocket 메시지 핸들러에 case 추가**

```javascript
case 'language_detected':
    updateLangBadge(msg.language);
    break;

case 'language_changed':
    updateLangBadge(msg.language);
    break;
```

- [ ] **Step 5: Commit**

```bash
git add realtime_demo/static/index.html
git commit -m "feat: add language mode toggle UI (Korean/English)

- Toggle: 한국어 전용 ↔ 자동 감지
- Language badge shows detected language
- Sends set_language_mode via WebSocket

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: 통합 테스트

- [ ] **Step 1: 서버 재시작 + 모델 로드 확인**

```bash
pkill -f "realtime_demo/server.py"
cd /home/jonghooy/work/cachedSTT
conda activate nemo-asr
CUDA_VISIBLE_DEVICES=0 python realtime_demo/server.py --s2s --port 3000
```

로그에서 확인:
- `English STT loaded`
- `LangID model loaded`
- `S2S Pipeline ready (LLM: vLLM API, TTS: KO+EN StyleTTS2)`

- [ ] **Step 2: 한국어 전용 모드 테스트 (기존 동작 유지)**

브라우저에서 http://localhost:3000, 한국어로 발화 → 한국어 응답 확인.

- [ ] **Step 3: 자동 감지 모드 테스트**

1. 언어 토글 클릭 → "자동" 모드
2. 영어로 발화 → 언어 배지 "English"로 변경 확인
3. LLM 영어 응답 + 영어 TTS 음성 확인

- [ ] **Step 4: GPU 메모리 확인**

```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
```

Expected: ~14-15GB / 32GB
