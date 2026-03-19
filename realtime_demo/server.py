#!/usr/bin/env python3
"""
Real-time Cache-Aware Streaming ASR Server
NeMo conformer_stream_step() 기반 증분 추론

- 160ms 청크 단위 증분 처리 (O(1) per step)
- 디스크 I/O 없음
- 블랭크 토큰 기반 끝점 감지
- ~170ms 지연

Usage:
    conda activate nemo-asr
    python server.py
    # 브라우저에서 http://localhost:3000 접속
"""

import asyncio
import copy
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from omegaconf import OmegaConf, open_dict

import nemo.collections.asr as nemo_asr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── 설정 ──
MODEL_PATH = "/home/jonghooy/work/timbel-asr-pilot/pretrained_models/runpod_trained_Stage3-best-cer0.1968.nemo"
SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.008       # RMS 에너지 임계값 (보정 전 기본값)
SILENCE_DURATION = 1.2          # 무음 N초 이상이면 Final
MAX_BUFFER_SECONDS = 30         # 최대 버퍼 (강제 Final)
ATT_CONTEXT_SIZE = [70, 1]      # 멀티 룩어헤드: 160ms per step

# ── FastAPI App ──
app = FastAPI(title="Real-time Korean ASR Demo (Streaming)")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── 글로벌 상태 ──
model = None
preprocessor = None  # 스트리밍용 전처리기 (dither=0, pad_to=0)
streaming_cfg = None
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

        # 노이즈 보정
        self.noise_samples = []
        self.noise_threshold = SILENCE_THRESHOLD
        self.noise_calibrated = False

        self._init_caches()

    def _init_caches(self):
        """인코더 캐시 초기화"""
        cache = model.encoder.get_initial_cache_state(
            batch_size=1, dtype=torch.float32, device="cuda:0"
        )
        self.cache_last_channel = cache[0]
        self.cache_last_time = cache[1]
        self.cache_last_channel_len = cache[2]

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
            mel, mel_len = preprocessor(input_signal=audio_tensor, length=audio_len)
            # mel: (1, n_mels, n_frames)

        total_mel_frames = mel.shape[2]
        chunks = []

        while True:
            if self.step == 0:
                chunk_size = _get_cfg_val(streaming_cfg.chunk_size, 0)
                pre_encode_size = _get_cfg_val(streaming_cfg.pre_encode_cache_size, 0)
                shift_size = _get_cfg_val(streaming_cfg.shift_size, 0)
            else:
                chunk_size = _get_cfg_val(streaming_cfg.chunk_size, 1)
                pre_encode_size = _get_cfg_val(streaming_cfg.pre_encode_cache_size, 1)
                shift_size = _get_cfg_val(streaming_cfg.shift_size, 1)

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
        drop = streaming_cfg.drop_extra_pre_encoded if step_idx > 0 else 0

        with torch.no_grad():
            result = model.conformer_stream_step(
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
            text = model.tokenizer.ids_to_text(token_ids)
            return text

        return ""


def process_session_chunks(session: StreamingSession):
    """GPU 스레드에서 실행: 가용 청크를 모두 처리하고 텍스트 반환.

    Returns:
        (text, blank_step_count) or None if no chunks
    """
    chunks = session.get_available_chunks()
    if not chunks:
        return None

    text = ""
    for mel_chunk, chunk_length, step_idx in chunks:
        text = session.run_streaming_step(mel_chunk, chunk_length, step_idx)

    return (text, session.blank_step_count)


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
    process_session_chunks(warmup_session)
    del warmup_session
    torch.cuda.empty_cache()

    logger.info("Streaming warmup complete. Ready!")
    logger.info(f"GPU memory: {torch.cuda.memory_allocated() / (1024**3):.2f}GB")


@app.get("/")
async def root():
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"), status_code=200)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected")

    session = StreamingSession()
    loop = asyncio.get_event_loop()

    try:
        while True:
            data = await websocket.receive_bytes()

            # PCM int16 → float32
            chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            session.audio_buffer = np.concatenate([session.audio_buffer, chunk])

            # 에너지 계산
            rms = float(np.sqrt(np.mean(chunk ** 2)))

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

            # 스트리밍 추론: GPU 스레드에서 가용 청크 처리
            if session.is_speaking and buffer_seconds >= 0.16:
                result = await loop.run_in_executor(
                    gpu_executor, process_session_chunks, session
                )

                if result is not None:
                    text, blank_count = result

                    if text and text != session.last_text:
                        await websocket.send_json({
                            "type": "partial",
                            "text": text
                        })
                        session.last_text = text

            # 버퍼 무한 성장 방지: is_speaking=False인데 버퍼가 쌓이면 리셋
            if not session.is_speaking and buffer_seconds >= 2.0 and not session.last_text:
                session.audio_buffer = np.array([], dtype=np.float32)
                session.mel_buffer_idx = 0
                session.silence_frames = 0

            # 끝점 판단: 구두점 + 무음 + 블랭크
            has_punct = session.last_text.rstrip().endswith(('.', '?'))
            blank_count = session.blank_step_count

            # 끝점 조건 판정 + reason 태깅
            endpoint_reason = ""
            if session.is_speaking and buffer_seconds >= 0.5:
                if has_punct and silence_seconds >= 0.5 and blank_count >= 3:
                    endpoint_reason = "punct"           # 구두점 + 무음
                elif silence_seconds >= SILENCE_DURATION and blank_count >= 5:
                    endpoint_reason = "silence+blank"   # 충분한 무음 + 모델 확인
                elif silence_seconds >= 2.0 and blank_count >= 2:
                    endpoint_reason = "long_silence"    # 긴 무음
                elif silence_seconds >= 3.0:
                    endpoint_reason = "timeout"         # 절대 안전망
                elif buffer_seconds >= MAX_BUFFER_SECONDS:
                    endpoint_reason = "buffer_max"      # 버퍼 초과

            if endpoint_reason and session.last_text:
                pre_flush_step = session.step
                pre_flush_blanks = blank_count
                flush_changed = False

                # Flush: 남은 mel 프레임 처리를 위해 무음 패딩 추가 후 1회 더 추론
                pre_flush_text = session.last_text
                pad = np.zeros(int(SAMPLE_RATE * 0.32), dtype=np.float32)
                session.audio_buffer = np.concatenate([session.audio_buffer, pad])
                flush_result = await loop.run_in_executor(
                    gpu_executor, process_session_chunks, session
                )
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
                await websocket.send_json({
                    "type": "final",
                    "text": session.last_text,
                    "utterance_id": session.utterance_count,
                    "duration": round(buffer_seconds, 1)
                })
                logger.info(
                    f"[Final #{session.utterance_count}] ({buffer_seconds:.1f}s) "
                    f"step={pre_flush_step}→{session.step} blanks={pre_flush_blanks} "
                    f"silence={silence_seconds:.1f}s reason={endpoint_reason} "
                    f"flush={'Y' if flush_changed else 'N'} "
                    f"| {session.last_text}"
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
    uvicorn.run(app, host="0.0.0.0", port=3000)
