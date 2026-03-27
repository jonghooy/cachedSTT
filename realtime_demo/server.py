#!/usr/bin/env python3
"""
Real-time Cache-Aware Streaming ASR Server + S2S Pipeline
NeMo conformer_stream_step() 기반 증분 추론

- 160ms 청크 단위 증분 처리 (O(1) per step)
- 디스크 I/O 없음
- 블랭크 토큰 기반 끝점 감지
- ~170ms 지연
- S2S 모드: STT Final → Qwen3.5 9B → StyleTTS2 → 음성 응답

Usage:
    conda activate nemo-asr
    python server.py              # STT only
    python server.py --s2s        # STT + S2S (LLM + TTS)
    # 브라우저에서 http://localhost:3000 접속
"""

import argparse
import asyncio
import base64
import copy
import json
import logging
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from omegaconf import OmegaConf, open_dict

import nemo.collections.asr as nemo_asr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from realtime_demo.turn_detector import TurnDetector
from realtime_demo.knowledge_client import KnowledgeClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── CLI 인자 파싱 ──
parser = argparse.ArgumentParser()
parser.add_argument("--s2s", action="store_true", help="Enable S2S mode (LLM + TTS)")
parser.add_argument("--port", type=int, default=3000)
cli_args, _ = parser.parse_known_args()

# ── 설정 ──
MODEL_PATH = "/home/jonghooy/work/timbel-asr-pilot/pretrained_models/runpod_trained_Stage3-best-cer0.1968.nemo"
MODEL_PATH_EN = "/home/jonghooy/work/timbel-asr-pilot/pretrained_models/nemotron-speech-streaming-en-0.6b.nemo"
SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.008       # RMS 에너지 임계값 (보정 전 기본값)
SILENCE_DURATION = 1.2          # 무음 N초 이상이면 Final
MAX_BUFFER_SECONDS = 30         # 최대 버퍼 (강제 Final)
ATT_CONTEXT_SIZE = [70, 13]     # 멀티 룩어헤드: ~350ms 청크 (멀티채널 서빙 최적화)
S2S_ENABLED = cli_args.s2s

# ── FastAPI App ──
app = FastAPI(title="Real-time Korean ASR + S2S Demo")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── 글로벌 상태 ──
model = None
preprocessor = None  # 스트리밍용 전처리기 (dither=0, pad_to=0)
streaming_cfg = None
model_en = None          # 영어 STT 모델
preprocessor_en = None   # 영어 STT 전처리기
streaming_cfg_en = None  # 영어 스트리밍 설정
lang_id_model = None     # 언어 감지 모델
turn_detector = None  # Turn Detection (형태소 기반)
s2s_pipeline = None   # S2S Pipeline (LLM + TTS)
knowledge_client = None  # Knowledge Service 클라이언트
dialogue_engine = None   # 시나리오 기반 대화 엔진
stt_batch = None      # 배치 STT 처리기
gpu_executor = ThreadPoolExecutor(max_workers=1)


def _get_cfg_val(cfg_val, idx):
    """streaming_cfg 값이 list일 때 인덱스로 접근, 스칼라면 그대로 반환."""
    if isinstance(cfg_val, (list, tuple)):
        return cfg_val[idx]
    return cfg_val


class StreamingSession:
    """WebSocket 연결별 스트리밍 상태"""

    def __init__(self):
        # 오디오 상태
        self.audio_buffer = np.array([], dtype=np.float32)
        self.mel_buffer_idx = 0   # 다음에 처리할 mel 프레임 시작 위치
        self.step = 0             # 스트리밍 스텝 카운터

        # 인코더 캐시
        self.cache_last_channel = None
        self.cache_last_time = None
        self.cache_last_channel_len = None

        # RNN-T 디코더 상태
        self.previous_hypotheses = None
        self.last_token_count = 0   # 블랭크 감지용

        # 끝점 감지
        self.blank_step_count = 0
        self.silence_frames = 0
        self.is_speaking = False
        self.utterance_count = 0
        self.last_text = ""

        # 감정 인식용 에너지 히스토리
        self.energy_history = []

        # 대화 히스토리 (멀티턴)
        self.conversation_history = []

        # 시나리오 대화 엔진
        self.dialogue_mode: str = "freeform"
        self.scenario_state: dict = {
            "scenario_id": None,
            "scenario_version": None,
            "current_node": None,
            "slots": {},
            "variables": {},
            "history": [],
            "retry_count": 0,
            "prosody": None,
            "stack": [],
            "awaiting_confirm": False,
            "_scenario_snapshot": None,
        }
        self._skip_dialogue_engine: bool = False

        # 노이즈 보정
        self.noise_samples = []
        self.noise_threshold = SILENCE_THRESHOLD
        self.noise_calibrated = False

        # 언어 설정
        self.language_mode = "ko"       # "ko" (한국어 전용) | "auto" (자동 감지)
        self.detected_language = "ko"   # "ko" | "en"
        self.lang_detected = False      # 감지 완료 여부

        self._init_caches()

    def _init_caches(self):
        """인코더 캐시 초기화 — detected_language에 따른 모델 사용"""
        stt_model = model_en if self.detected_language == "en" else model
        cache = stt_model.encoder.get_initial_cache_state(
            batch_size=1, dtype=torch.float32, device="cuda:0"
        )
        self.cache_last_channel = cache[0]
        self.cache_last_time = cache[1]
        self.cache_last_channel_len = cache[2]

    def get_dialogue_session(self) -> dict:
        """Return dict compatible with DialogueEngine.process_utterance."""
        return {
            "dialogue_mode": self.dialogue_mode,
            "scenario_state": self.scenario_state,
        }

    def sync_dialogue_session(self, session_dict: dict):
        """Write back dialogue state after processing."""
        self.dialogue_mode = session_dict["dialogue_mode"]
        self.scenario_state = session_dict["scenario_state"]

    def reset_for_new_utterance(self, keep_tail_seconds=0.0):
        """발화 완료 시 상태 리셋. endpoint 이후 도착한 오디오는 보존.

        Args:
            keep_tail_seconds: 버퍼 끝에서 보존할 오디오 길이 (초).
                               무음 구간 중 이미 도착한 후속 발화를 살리기 위함.
        """
        # endpoint 이후 도착한 오디오 보존
        tail_samples = int(keep_tail_seconds * SAMPLE_RATE)
        if tail_samples > 0 and len(self.audio_buffer) > tail_samples:
            self.audio_buffer = self.audio_buffer[-tail_samples:]
        else:
            self.audio_buffer = np.array([], dtype=np.float32)

        self.mel_buffer_idx = 0
        self.step = 0
        self._init_caches()
        self.previous_hypotheses = None
        self.last_token_count = 0
        self.blank_step_count = 0
        self.silence_frames = 0
        self.is_speaking = False
        self.last_text = ""
        self.energy_history = []

    def get_available_chunks(self):
        """누적 오디오에서 처리 가능한 mel 청크들을 추출.

        Returns:
            list of (mel_chunk, chunk_length, step_idx) tuples
        """
        if len(self.audio_buffer) < 160:  # 최소 10ms
            return []

        # 전체 오디오 → mel 변환
        with torch.no_grad():
            audio_tensor = torch.from_numpy(self.audio_buffer).unsqueeze(0).to("cuda:0")
            audio_len = torch.tensor([len(self.audio_buffer)], device="cuda:0")
            active_preprocessor = preprocessor_en if self.detected_language == "en" else preprocessor
            mel, mel_len = active_preprocessor(input_signal=audio_tensor, length=audio_len)
            # mel: (1, n_mels, n_frames)

        total_mel_frames = mel.shape[2]
        chunks = []
        active_streaming_cfg = streaming_cfg_en if self.detected_language == "en" else streaming_cfg

        while True:
            if self.step == 0:
                chunk_size = _get_cfg_val(active_streaming_cfg.chunk_size, 0)
                pre_encode_size = _get_cfg_val(active_streaming_cfg.pre_encode_cache_size, 0)
                shift_size = _get_cfg_val(active_streaming_cfg.shift_size, 0)
            else:
                chunk_size = _get_cfg_val(active_streaming_cfg.chunk_size, 1)
                pre_encode_size = _get_cfg_val(active_streaming_cfg.pre_encode_cache_size, 1)
                shift_size = _get_cfg_val(active_streaming_cfg.shift_size, 1)

            # 사용 가능한 mel 프레임이 충분한지 확인
            if self.mel_buffer_idx + chunk_size > total_mel_frames:
                break

            # pre-encode cache 포함 청크 추출
            if self.step == 0:
                # 첫 스텝: pre-encode cache 없음, 청크만
                mel_start = self.mel_buffer_idx
                mel_end = mel_start + chunk_size
                mel_chunk = mel[:, :, mel_start:mel_end]
            else:
                # 이후 스텝: 이전 pre_encode_size 프레임을 cache로 포함
                cache_start = max(0, self.mel_buffer_idx - pre_encode_size)
                mel_end = self.mel_buffer_idx + chunk_size
                mel_chunk = mel[:, :, cache_start:mel_end]

            chunk_length = torch.tensor([mel_chunk.shape[2]], device="cuda:0")
            step_idx = self.step
            chunks.append((mel_chunk, chunk_length, step_idx))

            self.mel_buffer_idx += shift_size
            self.step += 1

        return chunks

    def run_streaming_step(self, mel_chunk, chunk_length, step_idx):
        """conformer_stream_step 1회 호출.

        Args:
            mel_chunk: 전처리된 mel 스펙트로그램 텐서
            chunk_length: mel 프레임 수
            step_idx: 이 청크의 스트리밍 스텝 번호

        Returns:
            텍스트 (str) - 현재까지의 전체 디코딩 결과
        """
        stt_model = model_en if self.detected_language == "en" else model
        active_streaming_cfg = streaming_cfg_en if self.detected_language == "en" else streaming_cfg
        drop = active_streaming_cfg.drop_extra_pre_encoded if step_idx > 0 else 0

        with torch.no_grad():
            result = stt_model.conformer_stream_step(
                processed_signal=mel_chunk,
                processed_signal_length=chunk_length,
                cache_last_channel=self.cache_last_channel,
                cache_last_time=self.cache_last_time,
                cache_last_channel_len=self.cache_last_channel_len,
                keep_all_outputs=False,
                previous_hypotheses=self.previous_hypotheses,
                drop_extra_pre_encoded=drop,
                return_transcription=True,
            )

        (greedy_preds, transcriptions, self.cache_last_channel,
         self.cache_last_time, self.cache_last_channel_len,
         best_hyp) = result[:6]

        # 디코더 상태 업데이트
        self.previous_hypotheses = best_hyp

        # 블랭크 감지: 토큰 수 변화 확인
        if best_hyp and len(best_hyp) > 0:
            current_tokens = len(best_hyp[0].y_sequence)
            if current_tokens == self.last_token_count:
                self.blank_step_count += 1
            else:
                self.blank_step_count = 0
                self.last_token_count = current_tokens

            # 텍스트 추출
            token_ids = best_hyp[0].y_sequence
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.cpu().tolist()
            text = stt_model.tokenizer.ids_to_text(token_ids)
            return text

        return ""


def _batched_stream_step(sessions_and_chunks, stt_model=None):
    """여러 세션의 (session, mel_chunk, chunk_length, step_idx)를 배치로 추론.

    모든 항목이 step > 0이어야 함 (동일 chunk 파라미터).
    mel 길이가 다르면 0-padding.

    Args:
        sessions_and_chunks: list of (session, mel_chunk, chunk_length, step_idx)
        stt_model: 사용할 STT 모델 (None이면 한국어 모델 사용)

    Returns:
        list of text strings — 각 세션의 디코딩 결과
    """
    if not sessions_and_chunks:
        return []

    if stt_model is None:
        stt_model = model

    drop = stt_model.encoder.streaming_cfg.drop_extra_pre_encoded  # step > 0 공통

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
        result = stt_model.conformer_stream_step(
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
            text = stt_model.tokenizer.ids_to_text(token_ids)
            texts.append(text)
        else:
            texts.append("")

    return texts


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

        # step>0: 모델 레벨 배치 — 언어별 그룹핑
        if stepN_items:
            ko_items = [(s, m, l, idx) for s, m, l, idx in stepN_items if s.detected_language == "ko"]
            en_items = [(s, m, l, idx) for s, m, l, idx in stepN_items if s.detected_language == "en"]

            if ko_items:
                if len(ko_items) > 1:
                    logger.info(f"[Batch] KO inference: {len(ko_items)} sessions")
                texts = _batched_stream_step(ko_items, stt_model=model)
                for (session, _, _, _), text in zip(ko_items, texts):
                    session_texts[id(session)] = text
            if en_items:
                if len(en_items) > 1:
                    logger.info(f"[Batch] EN inference: {len(en_items)} sessions")
                texts = _batched_stream_step(en_items, stt_model=model_en)
                for (session, _, _, _), text in zip(en_items, texts):
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


class STTBatchProcessor:
    """배치 STT 처리기. 여러 세션의 청크를 모아 한 번에 GPU 추론.

    WebSocket 핸들러가 submit()으로 세션을 제출하면,
    collect_window_ms 동안 추가 요청을 모은 뒤 GPU 스레드에서 일괄 처리.
    """

    def __init__(self, collect_window_ms=30):
        self._collect_window = collect_window_ms / 1000
        self._queue = asyncio.Queue()
        self._running = False

    async def submit(self, session: StreamingSession):
        """세션을 배치 처리 큐에 제출. 결과를 기다려 반환.

        Returns:
            (text, blank_step_count) or None
        """
        future = asyncio.get_event_loop().create_future()
        await self._queue.put((session, future))
        return await future

    async def run(self):
        """배치 처리 루프. startup에서 asyncio.create_task()로 시작."""
        self._running = True
        loop = asyncio.get_event_loop()

        while self._running:
            # 첫 요청 대기 (블로킹)
            first = await self._queue.get()
            batch = [first]

            # 수집 윈도우 동안 추가 요청 수집
            deadline = loop.time() + self._collect_window
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(
                        self._queue.get(), timeout=remaining
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            # GPU 스레드에서 일괄 처리
            sessions = [s for s, _ in batch]
            futures = [f for _, f in batch]

            if len(sessions) > 1:
                logger.info(f"[Batch] Processing {len(sessions)} sessions")

            try:
                results = await loop.run_in_executor(
                    gpu_executor, _process_batch, sessions
                )
                for future, result in zip(futures, results):
                    if not future.cancelled():
                        future.set_result(result)
            except Exception as e:
                logger.error(f"[Batch] Error: {e}", exc_info=True)
                for future in futures:
                    if not future.cancelled():
                        future.set_result(None)

    def stop(self):
        self._running = False


@app.on_event("startup")
async def startup():
    global model, preprocessor, streaming_cfg

    logger.info(f"Loading model from {MODEL_PATH}...")
    t0 = time.time()

    # 1. 모델 로드
    model = nemo_asr.models.ASRModel.restore_from(MODEL_PATH, map_location="cuda:0")
    model.eval()
    model = model.to("cuda:0")

    # 2. RTX 5090 호환 디코딩 설정
    decoding_cfg = model.cfg.decoding
    with open_dict(decoding_cfg):
        decoding_cfg.greedy.loop_labels = True
        decoding_cfg.greedy.use_cuda_graph_decoder = False
    model.change_decoding_strategy(decoding_cfg)

    # 3. 스트리밍 파라미터 설정
    model.encoder.set_default_att_context_size(ATT_CONTEXT_SIZE)
    streaming_cfg = model.encoder.streaming_cfg
    logger.info(f"Streaming config: chunk_size={streaming_cfg.chunk_size}, "
                f"shift_size={streaming_cfg.shift_size}, "
                f"pre_encode_cache_size={streaming_cfg.pre_encode_cache_size}, "
                f"drop_extra_pre_encoded={streaming_cfg.drop_extra_pre_encoded}, "
                f"valid_out_len={streaming_cfg.valid_out_len}")

    # 4. 추론용 전처리기 생성 (dither=0, pad_to=0)
    cfg_pre = copy.deepcopy(model._cfg.preprocessor)
    OmegaConf.set_struct(cfg_pre, False)
    cfg_pre.dither = 0.0
    cfg_pre.pad_to = 0
    preprocessor = model.from_config_dict(cfg_pre)
    preprocessor = preprocessor.to("cuda:0")
    preprocessor.eval()

    logger.info(f"Model loaded in {time.time() - t0:.1f}s")

    # 5. 워밍업: 더미 오디오로 1회 스트리밍 스텝
    warmup_session = StreamingSession()
    warmup_audio = np.random.randn(SAMPLE_RATE).astype(np.float32) * 0.01
    warmup_session.audio_buffer = warmup_audio
    _process_batch([warmup_session])
    del warmup_session
    torch.cuda.empty_cache()

    logger.info("Streaming warmup complete. Ready!")
    logger.info(f"GPU memory: {torch.cuda.memory_allocated() / (1024**3):.2f}GB")

    # English STT 모델 로드
    global model_en, preprocessor_en, streaming_cfg_en, lang_id_model

    logger.info(f"Loading English STT from {MODEL_PATH_EN}...")
    model_en = nemo_asr.models.ASRModel.restore_from(MODEL_PATH_EN, map_location="cuda:0")
    model_en.eval()
    model_en = model_en.to("cuda:0")

    en_decoding_cfg = model_en.cfg.decoding
    with open_dict(en_decoding_cfg):
        en_decoding_cfg.greedy.loop_labels = True
        en_decoding_cfg.greedy.use_cuda_graph_decoder = False
    model_en.change_decoding_strategy(en_decoding_cfg)

    model_en.encoder.set_default_att_context_size(ATT_CONTEXT_SIZE)
    streaming_cfg_en = model_en.encoder.streaming_cfg

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
    logger.info(f"LangID loaded ({torch.cuda.memory_allocated()/(1024**3):.2f}GB GPU)")

    # 6. 배치 STT 처리기 시작
    global stt_batch
    stt_batch = STTBatchProcessor(collect_window_ms=30)
    asyncio.create_task(stt_batch.run())
    logger.info("STT Batch Processor started (30ms collect window)")

    # 6.5. 힌트 단어 로드 + boosting tree 적용
    saved_hints = _load_hint_words()
    if saved_hints:
        _apply_boosting(model, saved_hints)
        if model_en:
            _apply_boosting(model_en, saved_hints)
        logger.info(f"Hint words loaded: {len(saved_hints)} words")

    # 7. Turn Detector 초기화
    global turn_detector
    log_dir = Path(__file__).parent / "logs" / "turn_decisions"
    rules_path = Path(__file__).parent / "rules" / "turn_rules.json"
    turn_detector = TurnDetector(log_dir=log_dir, rules_path=rules_path)
    logger.info("Turn Detector initialized (Phase 1: rule-based)")

    # 7. S2S Pipeline 초기화 (--s2s 플래그 사용 시)
    if S2S_ENABLED:
        global s2s_pipeline
        from realtime_demo.s2s_pipeline import S2SPipeline
        s2s_pipeline = S2SPipeline(device="cuda:0")
        s2s_pipeline.load()
        logger.info("S2S Pipeline ready (Qwen3.5-9B + StyleTTS2)")

    # 8. Knowledge Service 연동 (optional)
    if S2S_ENABLED:
        global knowledge_client
        knowledge_client = KnowledgeClient()
        loaded = await knowledge_client.load_config()
        if loaded:
            s2s_pipeline.knowledge_client = knowledge_client
            logger.info("Knowledge Service connected")
        else:
            logger.info("Knowledge Service not available, using default prompts")

    # 9. Dialogue Engine (시나리오 기반 대화)
    global dialogue_engine
    from realtime_demo.dialogue.engine import DialogueEngine
    from realtime_demo.dialogue.scenario_cache import ScenarioCache
    from realtime_demo.dialogue.slot_manager import SlotManager
    from realtime_demo.dialogue.intent_matcher import IntentMatcher
    from realtime_demo.dialogue.action_runner import ActionRunner

    scenario_cache = ScenarioCache()
    try:
        await scenario_cache.refresh()
    except Exception:
        logging.warning("Could not load scenarios from Knowledge Service — running in freeform-only mode")

    slot_manager_d = SlotManager(llm_engine=s2s_pipeline.llm if s2s_pipeline else None)
    # embed_fn: calls Knowledge Service to embed user text (with LRU cache)
    _embed_client = knowledge_client  # may be None
    _embed_cache = {}
    _EMBED_CACHE_MAX = 200

    async def _embed_fn(text: str):
        if text in _embed_cache:
            return _embed_cache[text]
        if _embed_client is None:
            return np.zeros(1024, dtype=np.float32)
        vec = await _embed_client.embed(text)
        if vec:
            result = np.array(vec, dtype=np.float32)
            if len(_embed_cache) >= _EMBED_CACHE_MAX:
                # Remove oldest entry
                _embed_cache.pop(next(iter(_embed_cache)))
            _embed_cache[text] = result
            return result
        return np.zeros(1024, dtype=np.float32)

    intent_matcher = IntentMatcher(embed_fn=_embed_fn)
    intent_matcher.set_scenarios(scenario_cache.scenarios)
    # Load pre-computed trigger embeddings from Knowledge Service
    trigger_data = scenario_cache.get_trigger_data_with_embeddings()
    if trigger_data:
        intent_matcher.load_trigger_cache(trigger_data)
        logging.info(f"Loaded {sum(len(t['triggers']) for t in trigger_data)} trigger embeddings")
    action_runner = ActionRunner(knowledge_client=knowledge_client)

    dialogue_engine = DialogueEngine(
        scenario_cache=scenario_cache,
        intent_matcher=intent_matcher,
        slot_manager=slot_manager_d,
        llm_engine=s2s_pipeline.llm if s2s_pipeline else None,
        action_runner=action_runner,
    )
    logging.info(f"DialogueEngine initialized with {len(scenario_cache.scenarios)} scenarios")

    # 10. vLLM 워밍업 (콜드스타트 방지: CUDA 커널 컴파일 + prefix cache)
    if S2S_ENABLED and s2s_pipeline and s2s_pipeline.llm.is_loaded():
        try:
            logger.info("vLLM warmup: sending dummy request...")
            warmup_system = s2s_pipeline.knowledge_client.get_system_prompt() if (
                s2s_pipeline.knowledge_client and s2s_pipeline.knowledge_client.is_loaded()
            ) else "당신은 친절한 상담원입니다."
            warmup_messages = [
                {"role": "system", "content": warmup_system},
                {"role": "user", "content": "고객: 안녕하세요"},
            ]
            t_warmup = time.time()
            async for _ in s2s_pipeline.llm.generate_stream(warmup_messages, max_tokens=5):
                pass
            warmup_ms = (time.time() - t_warmup) * 1000
            logger.info(f"vLLM warmup done in {warmup_ms:.0f}ms (prefix cached)")
        except Exception as e:
            logger.warning(f"vLLM warmup failed: {e}")


@app.get("/")
async def root():
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(
        content=html_path.read_text(encoding="utf-8"),
        status_code=200,
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


# ── Turn Rules REST API ──

@app.get("/api/rules")
async def get_rules():
    return {"rules": turn_detector.get_rules()}


@app.post("/api/rules")
async def add_rule(body: dict):
    result = turn_detector.add_rule(
        suffix=body.get("suffix", ""),
        ending=body.get("ending", ""),
        note=body.get("note", ""),
    )
    return result


@app.delete("/api/rules/{suffix}")
async def delete_rule(suffix: str):
    return turn_detector.delete_rule(suffix)


@app.post("/api/rules/test")
async def test_rule(body: dict):
    """텍스트에 대해 분류 결과를 미리보기."""
    text = body.get("text", "")
    ending = turn_detector.classify_ending(text)
    eou = turn_detector.compute_eou(text, 0.0, 0, 0.01)
    return {"text": text, "ending": ending, "eou": eou}


# ── Hint Words (Word Boosting) API ──

HINT_WORDS_PATH = Path(__file__).parent / "rules" / "hint_words.json"
BOOSTING_ALPHA = 0.5  # boosting tree weight

# 영문자 → 한국어 발음 매핑
_ALPHA_TO_KO = {
    'A': '에이', 'B': '비', 'C': '씨', 'D': '디', 'E': '이', 'F': '에프',
    'G': '지', 'H': '에이치', 'I': '아이', 'J': '제이', 'K': '케이', 'L': '엘',
    'M': '엠', 'N': '엔', 'O': '오', 'P': '피', 'Q': '큐', 'R': '알',
    'S': '에스', 'T': '티', 'U': '유', 'V': '브이', 'W': '더블유', 'X': '엑스',
    'Y': '와이', 'Z': '제트',
}

def _expand_hint_variants(word: str) -> list:
    """힌트 단어의 변환 버전 생성 (영문 → 한국어 발음).

    예: "ECS은행" → ["ECS은행", "이씨에스은행"]
    """
    import re
    variants = [word]

    # 영문자가 포함되어 있으면 한국어 발음 버전 추가
    if re.search(r'[A-Za-z]', word):
        ko_version = ""
        for ch in word:
            upper = ch.upper()
            if upper in _ALPHA_TO_KO:
                ko_version += _ALPHA_TO_KO[upper]
            else:
                ko_version += ch
        if ko_version != word:
            variants.append(ko_version)

    return variants


def _expand_all_hints(words: list) -> list:
    """모든 힌트 단어에 대해 변환 버전 생성."""
    expanded = []
    for w in words:
        for v in _expand_hint_variants(w):
            if v not in expanded:
                expanded.append(v)
    return expanded

def _load_hint_words() -> list:
    """힌트 단어 파일 로드."""
    if HINT_WORDS_PATH.exists():
        try:
            with open(HINT_WORDS_PATH, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if not content:
                return []
            data = json.loads(content)
            return data.get("words", [])
        except (json.JSONDecodeError, KeyError):
            return []
    return []

def _save_hint_words(words: list):
    """힌트 단어 파일 저장."""
    HINT_WORDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HINT_WORDS_PATH, "w", encoding="utf-8") as f:
        json.dump({"words": words}, f, ensure_ascii=False, indent=2)

def _apply_boosting(stt_model, words: list, alpha: float = BOOSTING_ALPHA):
    """STT 모델에 boosting tree 적용 (또는 해제). 영문 힌트는 한국어 발음도 자동 추가."""
    try:
        # 영문 → 한국어 발음 변환 포함
        expanded = _expand_all_hints(words) if words else []
        if expanded and len(expanded) != len(words):
            logger.info(f"[Hints] Expanded: {words} → {expanded}")
        decoding_cfg = stt_model.cfg.decoding
        with open_dict(decoding_cfg):
            if expanded:
                decoding_cfg.greedy.boosting_tree.key_phrases_list = expanded
                decoding_cfg.greedy.boosting_tree_alpha = alpha
                decoding_cfg.greedy.boosting_tree.context_score = 1.0
                decoding_cfg.greedy.boosting_tree.depth_scaling = 2.0
            else:
                decoding_cfg.greedy.boosting_tree.key_phrases_list = None
                decoding_cfg.greedy.boosting_tree_alpha = 0.0
        stt_model.change_decoding_strategy(decoding_cfg)
        logger.info(f"[Hints] Applied {len(expanded)} phrases (from {len(words)} words), alpha={alpha if expanded else 0.0}")
    except Exception as e:
        logger.error(f"[Hints] Failed to apply boosting: {e}")


@app.get("/api/hints")
async def get_hints():
    """현재 힌트 단어 목록."""
    words = _load_hint_words()
    return {"words": words, "alpha": BOOSTING_ALPHA}


@app.post("/api/hints")
async def set_hints(body: dict):
    """힌트 단어 등록 (전체 교체)."""
    words = body.get("words", [])
    words = [w.strip() for w in words if w.strip()]
    _save_hint_words(words)
    # 한국어 + 영어 STT 모두 적용
    _apply_boosting(model, words)
    if model_en:
        _apply_boosting(model_en, words)
    return {"ok": True, "words": words, "count": len(words)}


@app.post("/api/hints/add")
async def add_hint(body: dict):
    """힌트 단어 추가."""
    word = body.get("word", "").strip()
    if not word:
        return {"error": "word가 비어있습니다"}
    words = _load_hint_words()
    if word not in words:
        words.append(word)
        _save_hint_words(words)
        _apply_boosting(model, words)
        if model_en:
            _apply_boosting(model_en, words)
    return {"ok": True, "words": words}


@app.delete("/api/hints/{word}")
async def delete_hint(word: str):
    """힌트 단어 삭제."""
    words = _load_hint_words()
    words = [w for w in words if w != word]
    _save_hint_words(words)
    _apply_boosting(model, words)
    if model_en:
        _apply_boosting(model_en, words)
    return {"ok": True, "words": words}


import re as _re

def _compute_prosody(session: StreamingSession, text: str, buffer_seconds: float) -> dict:
    """발화의 프로소디 요약 계산 (감정 인식용)."""
    energy_hist = session.energy_history
    if not energy_hist:
        return {"energy": "normal", "energy_trend": "flat", "speech_rate": "normal"}

    avg_energy = sum(energy_hist) / len(energy_hist)

    # 에너지 트렌드: 후반부 vs 전반부
    mid = len(energy_hist) // 2
    if mid > 0:
        first_half = sum(energy_hist[:mid]) / mid
        second_half = sum(energy_hist[mid:]) / len(energy_hist[mid:])
        ratio = second_half / first_half if first_half > 0.001 else 1.0
        if ratio > 1.3:
            trend = "rising"
        elif ratio < 0.7:
            trend = "falling"
        else:
            trend = "flat"
    else:
        trend = "flat"

    # 에너지 레벨
    noise_th = session.noise_threshold
    if avg_energy > noise_th * 8:
        energy_level = "high"
    elif avg_energy > noise_th * 3:
        energy_level = "normal"
    else:
        energy_level = "low"

    # 말 속도: 한글 음절 수 / 발화 시간
    syllables = len(_re.findall(r'[\uac00-\ud7a3]', text))
    if buffer_seconds > 0.3:
        rate = syllables / buffer_seconds
        if rate > 6.0:
            rate_level = "fast"
        elif rate > 3.5:
            rate_level = "normal"
        else:
            rate_level = "slow"
    else:
        rate_level = "normal"

    return {
        "energy": energy_level,
        "energy_trend": trend,
        "speech_rate": rate_level,
    }


@app.get("/api/s2s/status")
async def s2s_status():
    """S2S 파이프라인 상태 확인."""
    return {
        "enabled": S2S_ENABLED,
        "loaded": s2s_pipeline.is_loaded() if s2s_pipeline else False,
        "llm_loaded": s2s_pipeline.llm.is_loaded() if s2s_pipeline else False,
        "tts_loaded": s2s_pipeline.tts.is_loaded() if s2s_pipeline else False,
    }


@app.get("/api/scenarios/status")
async def scenario_status():
    """Return loaded scenarios status for Brain UI."""
    if not dialogue_engine or not dialogue_engine.scenario_cache:
        return {"loaded": False, "scenarios": [], "main_scenario": None}

    scenarios = []
    main_name = None
    for s in dialogue_engine.scenario_cache.scenarios.values():
        info = {
            "id": s.id,
            "name": s.name,
            "status": s.status,
            "version": s.version,
            "is_main": getattr(s, '_is_main', False),
            "node_count": len(s.nodes),
            "trigger_count": len(s.triggers.get("examples", [])),
            "slots": list(s.slots.keys()),
        }
        scenarios.append(info)
        if info["is_main"]:
            main_name = s.name

    trigger_cache_count = len(dialogue_engine.intent_matcher._trigger_cache) if dialogue_engine.intent_matcher else 0

    return {
        "loaded": True,
        "scenario_count": len(scenarios),
        "trigger_cache_count": trigger_cache_count,
        "main_scenario": main_name,
        "scenarios": scenarios,
    }


@app.post("/api/knowledge/refresh")
async def knowledge_refresh(body: dict = {}):
    """Knowledge Service 변경 시 캐시 갱신."""
    if knowledge_client:
        success = await knowledge_client.load_config()
        if success and s2s_pipeline:
            s2s_pipeline.knowledge_client = knowledge_client
        # Refresh scenarios
        if dialogue_engine and dialogue_engine.scenario_cache:
            await dialogue_engine.scenario_cache.refresh()
            dialogue_engine.intent_matcher.set_scenarios(
                dialogue_engine.scenario_cache.scenarios
            )
            trigger_data = dialogue_engine.scenario_cache.get_trigger_data_with_embeddings()
            if trigger_data:
                dialogue_engine.intent_matcher.load_trigger_cache(trigger_data)
            return {"status": "refreshed", "scenarios": len(dialogue_engine.scenario_cache.scenarios)}
        return {"status": "refreshed" if success else "failed"}
    return {"status": "no_client"}


async def _run_s2s(websocket: WebSocket, user_text: str, utterance_id: int,
                   cancel_event: asyncio.Event = None, audio_context: dict = None,
                   session: StreamingSession = None, history: list = None):
    """S2S 파이프라인 비동기 실행: LLM 스트리밍 → 문장 단위 TTS → WebSocket 전송."""
    try:
        ctx_str = f"energy={audio_context['energy']},rate={audio_context['speech_rate']},trend={audio_context['energy_trend']}" if audio_context else "none"
        logger.info(f"[S2S #{utterance_id}] Starting: {user_text} [audio: {ctx_str}]")

        async for event in s2s_pipeline.process(user_text, cancel_event=cancel_event,
                                                 audio_context=audio_context,
                                                 history=history,
                                                 language=session.detected_language if session else "ko"):
            etype = event["type"]

            if etype == "llm_start":
                await websocket.send_json({
                    "type": "s2s_llm_start",
                    "utterance_id": utterance_id,
                })

            elif etype == "s2s_emotion":
                await websocket.send_json({
                    "type": "s2s_emotion",
                    "emotion": event["emotion"],
                    "utterance_id": utterance_id,
                })

            elif etype == "llm_token":
                await websocket.send_json({
                    "type": "s2s_llm_token",
                    "token": event["token"],
                    "full_text": event["full_text"],
                    "utterance_id": utterance_id,
                })

            elif etype == "tts_audio":
                # TTS 오디오를 base64로 인코딩하여 JSON으로 전송
                audio_b64 = base64.b64encode(event["audio"]).decode("ascii")
                await websocket.send_json({
                    "type": "s2s_tts_audio",
                    "audio_b64": audio_b64,
                    "sample_rate": event["sample_rate"],
                    "sentence": event["sentence"],
                    "sentence_idx": event["sentence_idx"],
                    "utterance_id": utterance_id,
                })

            elif etype == "llm_done":
                await websocket.send_json({
                    "type": "s2s_llm_done",
                    "full_text": event["full_text"],
                    "utterance_id": utterance_id,
                })
                # 대화 히스토리에 assistant 응답 추가 (barge-in 취소 시 제외)
                if session is not None and not (cancel_event and cancel_event.is_set()):
                    session.conversation_history.append({
                        "role": "assistant",
                        "content": event["full_text"],
                    })

            elif etype == "s2s_done":
                latency = event["latency"]
                emotion = event.get("emotion", "neutral")
                await websocket.send_json({
                    "type": "s2s_done",
                    "latency": latency,
                    "emotion": emotion,
                    "utterance_id": utterance_id,
                })
                logger.info(
                    f"[S2S #{utterance_id}] Done: "
                    f"emotion={emotion}, "
                    f"TTFT={latency['ttft_ms']}ms, "
                    f"first_audio={latency['first_audio_ms']}ms, "
                    f"total={latency['total_ms']}ms, "
                    f"sentences={latency['sentences']}"
                )

    except Exception as e:
        logger.error(f"[S2S #{utterance_id}] Error: {e}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "s2s_error",
                "error": str(e),
                "utterance_id": utterance_id,
            })
        except Exception:
            pass


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected")

    session = StreamingSession()
    loop = asyncio.get_event_loop()

    # Default: freeform (일반 S2S) mode.
    # Scenario mode is entered via UI toggle (set_mode message).
    session._skip_dialogue_engine = True  # Start in freeform by default
    logger.info("Session started in freeform mode (use mode toggle for scenario)")

    # S2S barge-in 상태
    s2s_cancel_event: asyncio.Event = None   # 현재 S2S 취소 이벤트
    s2s_task: asyncio.Task = None            # 현재 S2S 비동기 태스크

    try:
        while True:
            message = await websocket.receive()

            # 텍스트 메시지 (barge-in 등 제어)
            if "text" in message:
                try:
                    msg = __import__("json").loads(message["text"])
                except Exception:
                    continue
                if msg.get("type") == "barge_in":
                    if s2s_cancel_event is not None:
                        s2s_cancel_event.set()
                        logger.info("[Barge-in] S2S cancelled by user")
                    await websocket.send_json({"type": "s2s_cancelled"})
                    continue
                # 대화 모드 전환 (시나리오 ↔ 일반 S2S)
                if msg.get("type") == "set_mode":
                    new_mode = msg.get("mode", "scenario")
                    # Reset scenario state completely
                    session.dialogue_mode = "freeform"
                    session.scenario_state = {
                        "scenario_id": None, "scenario_version": None, "current_node": None,
                        "slots": {}, "variables": {}, "history": [], "retry_count": 0,
                        "prosody": None, "stack": [], "awaiting_confirm": False,
                        "_scenario_snapshot": None,
                    }
                    if new_mode == "freeform":
                        session._skip_dialogue_engine = True
                        logger.info("[Mode] Switched to: freeform (S2S only)")
                    else:
                        session._skip_dialogue_engine = False
                        # Re-enter main scenario
                        if dialogue_engine:
                            ds = session.get_dialogue_session()
                            main_result = await dialogue_engine.auto_enter_main(ds)
                            session.sync_dialogue_session(ds)
                            if main_result and main_result.response_text:
                                await websocket.send_json({
                                    "type": "scenario_response",
                                    "text": main_result.response_text,
                                    "mode": "scenario",
                                    "action": main_result.action,
                                    "utterance_id": 0,
                                })
                                # TTS for greeting
                                if s2s_pipeline and s2s_pipeline.tts and s2s_pipeline.tts.is_loaded():
                                    try:
                                        import base64
                                        loop = asyncio.get_event_loop()
                                        pcm_bytes, sr = await loop.run_in_executor(
                                            gpu_executor,
                                            s2s_pipeline.tts.synthesize_to_pcm16,
                                            main_result.response_text,
                                        )
                                        if pcm_bytes:
                                            await websocket.send_json({
                                                "type": "tts_audio",
                                                "audio": base64.b64encode(pcm_bytes).decode(),
                                                "sample_rate": sr,
                                                "sentence": main_result.response_text,
                                                "sentence_idx": 0,
                                            })
                                    except Exception:
                                        pass
                        logger.info("[Mode] Switched to: scenario (main scenario entered)")
                    await websocket.send_json({"type": "mode_changed", "mode": new_mode})
                    continue
                # 대화 히스토리 리셋
                if msg.get("type") == "reset_history":
                    session.conversation_history = []
                    await websocket.send_json({"type": "history_reset"})
                    logger.info("[History] Reset by user")
                    continue

                # 언어 모드 설정
                if msg.get("type") == "set_language_mode":
                    session.language_mode = msg.get("mode", "ko")
                    if session.language_mode == "ko":
                        session.detected_language = "ko"
                        session.lang_detected = True
                        session._init_caches()
                        session.reset_for_new_utterance()
                    else:
                        session.lang_detected = False
                    logger.info(f"[Lang] Mode: {session.language_mode}")
                    continue

                # 수동 언어 전환
                if msg.get("type") == "set_language":
                    new_lang = msg.get("language", "ko")
                    if new_lang in ("ko", "en"):
                        session.language_mode = new_lang  # 고정 모드로 전환
                        session.detected_language = new_lang
                        session.lang_detected = True
                        session._init_caches()
                        session.reset_for_new_utterance()
                        await websocket.send_json({"type": "language_changed", "language": new_lang})
                        logger.info(f"[Lang] Manual: {new_lang} (mode={new_lang})")
                    continue

            # 바이너리 메시지 (오디오)
            if "bytes" not in message:
                continue
            data = message["bytes"]

            # PCM int16 → float32
            chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            session.audio_buffer = np.concatenate([session.audio_buffer, chunk])

            # 에너지 계산
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            session.energy_history.append(rms)

            # 배경 노이즈 자동 보정 (처음 4청크 = ~1초)
            if not session.noise_calibrated:
                session.noise_samples.append(rms)
                if len(session.noise_samples) >= 4:
                    avg_noise = sum(session.noise_samples) / len(session.noise_samples)
                    session.noise_threshold = min(max(avg_noise * 3.0, 0.005), 0.05)
                    session.noise_calibrated = True
                    logger.info(f"Noise calibrated: avg={avg_noise:.4f}, threshold={session.noise_threshold:.4f}")
                    session.audio_buffer = np.array([], dtype=np.float32)
                continue

            # 오디오 레벨 전송
            await websocket.send_json({
                "type": "level",
                "value": min(rms * 10, 1.0)
            })

            # 음성/무음 판단
            if rms > session.noise_threshold:
                session.is_speaking = True
                session.silence_frames = 0
            else:
                session.silence_frames += len(chunk)

            silence_seconds = session.silence_frames / SAMPLE_RATE
            buffer_seconds = len(session.audio_buffer) / SAMPLE_RATE

            # 언어 감지: 자동 모드에서 음성이 충분히 쌓였을 때 1회
            if (session.language_mode == "auto" and not session.lang_detected
                    and session.is_speaking and len(session.audio_buffer) >= SAMPLE_RATE * 3):
                # 최근 에너지에서 실제 음성 구간이 충분한지 확인
                energy_chunks = session.energy_history[-12:]  # 최근 ~3초
                voiced_count = sum(1 for e in energy_chunks if e > session.noise_threshold)
                if voiced_count >= 4:  # 음성이 4청크(~1초) 이상
                    detect_audio = session.audio_buffer[-int(SAMPLE_RATE * 2):]
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                        sf.write(f.name, detect_audio, SAMPLE_RATE)
                        detected = lang_id_model.get_label(f.name)
                    import os; os.unlink(f.name)
                    lang = detected if detected in ("ko", "en") else "ko"
                    session.detected_language = lang
                    session.lang_detected = True
                    logger.info(f"[Lang] Detected: {lang} (voiced={voiced_count}/12, {len(detect_audio)/SAMPLE_RATE:.1f}s)")

                    if lang != "ko":
                        session._init_caches()
                        session.mel_buffer_idx = 0
                        session.step = 0

                    await websocket.send_json({"type": "language_detected", "language": lang})

            # 스트리밍 추론: GPU 스레드에서 가용 청크 처리
            # 무음 구간에서도 추론 계속 (blank_count 업데이트를 위해)
            # 단, 텍스트가 있을 때만 (발화 중이거나 endpoint 대기 중)
            should_infer = (session.is_speaking or session.last_text) and buffer_seconds >= 0.16
            if should_infer:
                result = await stt_batch.submit(session)

                if result is not None:
                    text, blank_count = result

                    if text and text != session.last_text:
                        # Turn Detection 정보 포함
                        ending = turn_detector.classify_ending(text)
                        partial_eou = turn_detector.compute_eou(
                            text, silence_seconds, blank_count, rms)
                        partial_th = turn_detector.get_silence_threshold(partial_eou)
                        await websocket.send_json({
                            "type": "partial",
                            "text": text,
                            "turn": {
                                "ending": ending,
                                "eou": round(partial_eou, 2),
                                "threshold": partial_th,
                            }
                        })
                        session.last_text = text

            # 버퍼 무한 성장 방지: is_speaking=False인데 버퍼가 쌓이면 리셋
            if not session.is_speaking and buffer_seconds >= 2.0 and not session.last_text:
                session.audio_buffer = np.array([], dtype=np.float32)
                session.mel_buffer_idx = 0
                session.silence_frames = 0

            # 끝점 판단: Turn Detection 기반 동적 임계값
            blank_count = session.blank_step_count
            eou_score = 0.0

            endpoint_reason = ""
            if (session.is_speaking or session.last_text) and buffer_seconds >= 0.3 and session.last_text:
                ending_type = turn_detector.classify_ending(session.last_text)
                eou_score = turn_detector.compute_eou(
                    text=session.last_text,
                    silence_sec=silence_seconds,
                    blank_count=blank_count,
                    energy=rms,
                )

                # 종결어미: blank 또는 짧은 무음이면 즉시 endpoint
                if ending_type in ("final",):
                    if blank_count >= 1 or silence_seconds >= 0.15:
                        endpoint_reason = f"final_ending({ending_type},eou={eou_score:.2f})"
                # 그 외: 동적 silence 임계값 기반
                elif silence_seconds > 0:
                    dynamic_threshold = turn_detector.get_silence_threshold(eou_score)
                    if silence_seconds >= dynamic_threshold and (blank_count >= 2 or silence_seconds >= dynamic_threshold + 0.3):
                        endpoint_reason = f"turn(eou={eou_score:.2f},th={dynamic_threshold})"
                    elif silence_seconds >= 2.0:
                        endpoint_reason = "timeout"         # 절대 안전망

                if not endpoint_reason and buffer_seconds >= MAX_BUFFER_SECONDS:
                    endpoint_reason = "buffer_max"      # 버퍼 초과

            if endpoint_reason and session.last_text:
                pre_flush_step = session.step
                pre_flush_blanks = blank_count
                flush_changed = False

                # Flush: 남은 mel 프레임 처리를 위해 무음 패딩 추가 후 1회 더 추론
                pre_flush_text = session.last_text
                pad = np.zeros(int(SAMPLE_RATE * 0.32), dtype=np.float32)
                session.audio_buffer = np.concatenate([session.audio_buffer, pad])
                flush_result = await stt_batch.submit(session)
                if flush_result:
                    flushed_text, _ = flush_result
                    if flushed_text:
                        session.last_text = flushed_text
                flush_changed = session.last_text != pre_flush_text
                if flush_changed:
                    logger.info(
                        f"  [Flush] step {pre_flush_step}→{session.step} "
                        f"| \"{pre_flush_text}\" → \"{session.last_text}\""
                    )

                session.utterance_count += 1
                final_ending = turn_detector.classify_ending(session.last_text)
                await websocket.send_json({
                    "type": "final",
                    "text": session.last_text,
                    "utterance_id": session.utterance_count,
                    "duration": round(buffer_seconds, 1),
                    "turn": {
                        "ending": final_ending,
                        "eou": round(eou_score, 2),
                        "reason": endpoint_reason,
                    }
                })
                logger.info(
                    f"[Final #{session.utterance_count}] ({buffer_seconds:.1f}s) "
                    f"step={pre_flush_step}→{session.step} blanks={pre_flush_blanks} "
                    f"silence={silence_seconds:.1f}s reason={endpoint_reason} "
                    f"flush={'Y' if flush_changed else 'N'} "
                    f"| {session.last_text}"
                )

                # Turn Detection 로깅 (Phase 2 학습 데이터)
                turn_detector.log_decision(
                    text=session.last_text,
                    eou_score=eou_score,
                    silence_sec=silence_seconds,
                    blank_count=blank_count,
                    energy=rms,
                    endpoint=True,
                    endpoint_reason=endpoint_reason,
                )

                # ── 노이즈 필터: 배경 소음/원거리 대화 걸러내기 ──
                # 1) 텍스트 너무 짧음 (한글 4자 미만 = 의미 없는 조각)
                # 2) 발화 대비 텍스트 밀도 너무 낮음 (긴 버퍼에 짧은 텍스트 = 노이즈)
                # 3) 평균 에너지가 너무 낮음 (원거리 대화)
                # 한글 또는 영문 글자 수로 밀도 계산 (다국어 지원)
                text_chars = len(_re.findall(r'[\uac00-\ud7a3a-zA-Z]', session.last_text))
                text_density = text_chars / max(buffer_seconds, 0.1)
                # 에너지: 전체 버퍼가 아닌 최근 5초 구간만 사용 (TTS 재생 중 무음 희석 방지)
                recent_energy = session.energy_history[-20:] if len(session.energy_history) > 20 else session.energy_history  # ~5초 (250ms 청크 × 20)
                avg_energy = sum(recent_energy) / max(len(recent_energy), 1)
                noise_threshold = session.noise_threshold

                # Noise filter disabled — all recognized text is processed
                # (Previously filtered low-density text, but caused false positives after TTS playback)

                # ── Dialogue Engine: try scenario matching first ──
                if dialogue_engine and not getattr(session, '_skip_dialogue_engine', False):
                    ds = session.get_dialogue_session()
                    _prosody = _compute_prosody(session, session.last_text, buffer_seconds)
                    logger.info(f"[DialogueEngine] Processing: mode={ds['dialogue_mode']}, text=\"{session.last_text}\", node={ds['scenario_state'].get('current_node')}")
                    d_result = await dialogue_engine.process_utterance(
                        ds, session.last_text, _prosody
                    )
                    session.sync_dialogue_session(ds)
                    logger.info(f"[DialogueEngine] Result: s2s={d_result.should_use_s2s}, mode={d_result.mode}, text={d_result.response_text[:50] if d_result.response_text else 'None'}, action={d_result.action}, await={d_result.awaiting_input}")

                    if not d_result.should_use_s2s:
                        # Add to conversation history for S2S context continuity
                        session.conversation_history.append({
                            "role": "user",
                            "content": session.last_text,
                        })
                        if d_result.response_text:
                            session.conversation_history.append({
                                "role": "assistant",
                                "content": d_result.response_text,
                            })

                        if d_result.response_text:
                            await websocket.send_json({
                                "type": "scenario_response",
                                "text": d_result.response_text,
                                "mode": d_result.mode,
                                "action": d_result.action,
                                "utterance_id": session.utterance_count,
                            })
                            # TTS synthesis for scenario response (sentence-level with barge-in)
                            if s2s_pipeline and s2s_pipeline.tts and s2s_pipeline.tts.is_loaded():
                                try:
                                    import base64
                                    loop = asyncio.get_event_loop()
                                    pcm_bytes, sr = await loop.run_in_executor(
                                        gpu_executor,
                                        s2s_pipeline.tts.synthesize_to_pcm16,
                                        d_result.response_text,
                                    )
                                    if pcm_bytes:
                                        await websocket.send_json({
                                            "type": "tts_audio",
                                            "audio": base64.b64encode(pcm_bytes).decode(),
                                            "sample_rate": sr,
                                            "sentence": d_result.response_text,
                                            "sentence_idx": 0,
                                        })
                                        logger.info(f"[Scenario TTS] Sent {len(pcm_bytes)} bytes")
                                except Exception as e:
                                    logger.warning(f"Scenario TTS failed: {e}")
                        session.reset_for_new_utterance(keep_tail_seconds=0.5)
                        continue
                # ── End Dialogue Engine ──

                # S2S 파이프라인: Final 텍스트 → LLM → TTS → 음성 응답
                if S2S_ENABLED and s2s_pipeline and s2s_pipeline.is_loaded():
                    # 히스토리 스냅샷 (현재 user 발화는 process()에서 current로 들어가므로 제외)
                    history_snapshot = list(session.conversation_history)

                    # 대화 히스토리에 user 발화 추가
                    session.conversation_history.append({
                        "role": "user",
                        "content": session.last_text,
                    })

                    # 이전 S2S 실행 중이면 취소
                    if s2s_cancel_event is not None:
                        s2s_cancel_event.set()
                    s2s_cancel_event = asyncio.Event()
                    final_text_for_s2s = session.last_text
                    # 프로소디 계산 (감정 인식용)
                    prosody = _compute_prosody(session, final_text_for_s2s, buffer_seconds)
                    s2s_task = asyncio.create_task(
                        _run_s2s(websocket, final_text_for_s2s,
                                 session.utterance_count, s2s_cancel_event,
                                 audio_context=prosody,
                                 session=session, history=history_snapshot)
                    )

                # 무음 구간 중 이미 도착한 후속 오디오 보존
                # silence_seconds 만큼은 이미 무음이므로, 그 뒤에 올 수 있는 발화를 살림
                keep_seconds = min(silence_seconds, 0.5)
                session.reset_for_new_utterance(keep_tail_seconds=keep_seconds)

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)


if __name__ == "__main__":
    if S2S_ENABLED:
        logger.info("S2S mode enabled (--s2s)")
    uvicorn.run(app, host="0.0.0.0", port=cli_args.port)
