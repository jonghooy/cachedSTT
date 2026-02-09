#!/usr/bin/env python3
"""
Real-time ASR Demo Server
마이크 입력을 받아 실시간 한국어 음성인식 결과를 웹으로 출력

Usage:
    conda activate nemo-asr
    python server.py
    # 브라우저에서 http://localhost:8000 접속
"""

import asyncio
import logging
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import nemo.collections.asr as nemo_asr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── 설정 ──
MODEL_PATH = "/home/jonghooy/work/timbel-asr-pilot/experiments/Stage3-Full-Finetune_final.nemo"
SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.008       # RMS 에너지 임계값
SILENCE_DURATION = 0.8          # 무음 N초 이상이면 Final
MAX_BUFFER_SECONDS = 30         # 최대 버퍼 (강제 Final)
TRANSCRIBE_INTERVAL = 0.15      # Partial 갱신 주기 (초)

# ── FastAPI App ──
app = FastAPI(title="Real-time Korean ASR Demo")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

model = None


def patch_transcribe_dataloader(model_instance):
    """NeMo 2.6.1 lhotse 비호환 패치 - _setup_transcribe_dataloader 완전 교체"""
    from omegaconf import DictConfig

    def patched_setup(config):
        if 'manifest_filepath' in config:
            manifest_filepath = config['manifest_filepath']
            batch_size = config['batch_size']
        else:
            manifest_filepath = os.path.join(config['temp_dir'], 'manifest.json')
            batch_size = min(config['batch_size'], len(config['paths2audio_files']))

        dl_config = {
            'use_lhotse': False,
            'manifest_filepath': manifest_filepath,
            'sample_rate': model_instance.preprocessor._sample_rate,
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': config.get('num_workers', min(batch_size, os.cpu_count() - 1)),
            'pin_memory': False,
            'channel_selector': config.get('channel_selector', None),
        }

        temporary_datalayer = model_instance._setup_dataloader_from_config(
            config=DictConfig(dl_config)
        )
        return temporary_datalayer

    model_instance._setup_transcribe_dataloader = patched_setup


@app.on_event("startup")
async def startup():
    global model
    logger.info(f"Loading model from {MODEL_PATH}...")
    t0 = time.time()
    model = nemo_asr.models.ASRModel.restore_from(MODEL_PATH, map_location="cuda:0")
    model.eval()
    model = model.to("cuda:0")
    patch_transcribe_dataloader(model)
    logger.info(f"Model loaded in {time.time() - t0:.1f}s")

    # Warmup
    warmup_audio = np.zeros(16000, dtype=np.float32)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sf.write(f.name, warmup_audio, SAMPLE_RATE)
        model.transcribe([f.name], batch_size=1)
        os.unlink(f.name)
    logger.info("Model warmup complete. Ready!")
    logger.info(f"GPU memory: {torch.cuda.memory_allocated() / (1024**3):.2f}GB")


@app.get("/")
async def root():
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"), status_code=200)


def transcribe_audio(audio: np.ndarray) -> str:
    """오디오 버퍼 → 텍스트 변환"""
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, audio, SAMPLE_RATE)
            result = model.transcribe([f.name], batch_size=1)
            os.unlink(f.name)
            if not result:
                return ""
            # Hypothesis 객체 또는 문자열 처리
            item = result[0]
            if isinstance(item, str):
                return item
            elif hasattr(item, 'text'):
                return item.text
            else:
                return str(item)
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return ""


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected")

    audio_buffer = np.array([], dtype=np.float32)
    silence_frames = 0
    last_text = ""
    last_transcribe_time = 0.0
    is_speaking = False
    utterance_count = 0

    # 배경 노이즈 레벨 자동 측정 (처음 1초)
    noise_samples = []
    noise_threshold = SILENCE_THRESHOLD
    noise_calibrated = False
    chunk_count = 0

    loop = asyncio.get_event_loop()

    try:
        while True:
            data = await websocket.receive_bytes()

            # PCM int16 → float32
            chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            audio_buffer = np.concatenate([audio_buffer, chunk])
            chunk_count += 1

            # 에너지 계산
            rms = float(np.sqrt(np.mean(chunk ** 2)))

            # 배경 노이즈 자동 보정 (처음 4청크 = ~1초)
            if not noise_calibrated:
                noise_samples.append(rms)
                if len(noise_samples) >= 4:
                    avg_noise = sum(noise_samples) / len(noise_samples)
                    noise_threshold = max(avg_noise * 3.0, 0.005)
                    noise_calibrated = True
                    logger.info(f"Noise calibrated: avg_noise={avg_noise:.4f}, threshold={noise_threshold:.4f}")
                    # 보정 중 쌓인 버퍼 리셋
                    audio_buffer = np.array([], dtype=np.float32)
                    continue

            # 오디오 레벨 전송
            await websocket.send_json({
                "type": "level",
                "value": min(rms * 10, 1.0)
            })

            # 말하는 중 / 무음 판단
            if rms > noise_threshold:
                is_speaking = True
                silence_frames = 0
            else:
                silence_frames += len(chunk)

            silence_seconds = silence_frames / SAMPLE_RATE
            buffer_seconds = len(audio_buffer) / SAMPLE_RATE
            current_time = time.time()

            # Partial: 주기적 전사
            should_transcribe = (
                buffer_seconds >= 0.3
                and is_speaking
                and (current_time - last_transcribe_time) >= TRANSCRIBE_INTERVAL
            )

            # Final: 무음 감지 또는 버퍼 초과
            force_final = (
                is_speaking
                and buffer_seconds >= 1.0
                and (silence_seconds >= SILENCE_DURATION or buffer_seconds >= MAX_BUFFER_SECONDS)
            )

            if should_transcribe or force_final:
                text = await loop.run_in_executor(
                    None, transcribe_audio, audio_buffer.copy()
                )
                last_transcribe_time = current_time

                if text and text != last_text:
                    await websocket.send_json({
                        "type": "partial",
                        "text": text
                    })
                    last_text = text

                if force_final and last_text:
                    utterance_count += 1
                    await websocket.send_json({
                        "type": "final",
                        "text": last_text,
                        "utterance_id": utterance_count,
                        "duration": round(buffer_seconds, 1)
                    })
                    logger.info(f"[Final #{utterance_count}] ({buffer_seconds:.1f}s) rms={rms:.4f} thr={noise_threshold:.4f} silence={silence_seconds:.1f}s | {last_text}")
                    # 리셋
                    audio_buffer = np.array([], dtype=np.float32)
                    silence_frames = 0
                    last_text = ""
                    is_speaking = False

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
