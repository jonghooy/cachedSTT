#!/usr/bin/env python3
"""
S2S (Speech-to-Speech) Pipeline — Phase 1: Text Pipeline
STT Final 텍스트 → Qwen3.5 9B (vLLM API) → StyleTTS2 (TTS) → 응답 음성

Architecture:
    고객 음성 → [STT] → 텍스트 → [LLM 스트리밍 via HTTP] → 응답 텍스트 → [TTS] → 음성 PCM

LLM은 별도 vLLM 서버 프로세스로 실행 (nemo-asr 환경과 분리):
    vllm serve /mnt/usb/models/Qwen3.5-9B --port 8000
"""

import asyncio
import json
import logging
import sys
import time
import threading
from pathlib import Path
from typing import AsyncGenerator

import numpy as np
import torch
import httpx

logger = logging.getLogger(__name__)

# ── TTS 엔진 타입 ──
TTS_ENGINE_TYPE = "styletts2"  # "edge" (클라우드) 또는 "styletts2" (로컬)
ZHISPER_TTS_PATH = "/home/jonghooy/work/zhisper"
STYLETTS2_PATH = "/home/jonghooy/work/zhisper/styletts2_korean"

# ── vLLM API 설정 ──
VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_MODEL = "/mnt/usb/models/Qwen3.5-9B"
LLM_MAX_TOKENS = 256
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.9

# ── TTS 설정 ──
TTS_SAMPLE_RATE = 24000
TTS_REF_AUDIO = "/home/jonghooy/work/zhisper/tts/voices/styletts2/00003.wav"

# ── 시스템 프롬프트 ──
SYSTEM_PROMPT = """당신은 친절하고 전문적인 한국어 콜센터 상담원입니다.
고객의 발화 내용과 음성 특징(에너지, 말 속도, 톤 변화)을 종합하여 감정을 파악하고, 감정에 맞게 응대하세요.

규칙:
1. 반드시 첫 줄에 [EMOTION:anger|anxiety|neutral|satisfaction|joy] 태그를 출력하세요.
2. 태그 다음 줄부터 답변을 작성하세요.
3. 답변은 1-2문장으로 짧게 유지하세요. TTS로 읽히므로 자연스러운 구어체를 사용하세요.
4. 마크다운이나 특수 기호를 사용하지 마세요.

감정별 응대 지침:
- anger: 진심으로 사과하고, 즉각적인 해결 의지를 표현하세요.
- anxiety: 안심시키고, 차근차근 안내하세요.
- neutral: 친절하고 명확하게 답변하세요.
- satisfaction: 감사를 표현하고 추가 도움을 제안하세요.
- joy: 함께 기뻐하며 밝게 응대하세요."""

import re as _re_mod
EMOTION_TAG_PATTERN = _re_mod.compile(r'\[EMOTION:(anger|anxiety|neutral|satisfaction|joy)\]')


class LLMEngine:
    """Qwen3.5 9B via vLLM OpenAI-compatible API (HTTP 스트리밍)."""

    def __init__(self, base_url: str = VLLM_BASE_URL):
        self.base_url = base_url
        self.model = VLLM_MODEL
        self._client = None
        self._loaded = False

    def load(self):
        """HTTP 클라이언트 초기화 + vLLM 서버 연결 확인."""
        if self._loaded:
            return

        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=60.0)
        self._loaded = True
        logger.info(f"LLM Engine ready (vLLM API: {self.base_url})")

    async def check_health(self) -> bool:
        """vLLM 서버가 응답하는지 확인."""
        try:
            resp = await self._client.get("/models")
            return resp.status_code == 200
        except Exception:
            return False

    async def generate_stream(
        self,
        user_text: str,
        system_prompt: str = SYSTEM_PROMPT,
        max_tokens: int = LLM_MAX_TOKENS,
        temperature: float = LLM_TEMPERATURE,
    ) -> AsyncGenerator[str, None]:
        """vLLM API 스트리밍 생성 — 토큰 단위로 yield."""
        if not self._loaded:
            self.load()

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": LLM_TOP_P,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        async with self._client.stream(
            "POST", "/chat/completions", json=payload
        ) as response:
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    def is_loaded(self) -> bool:
        return self._loaded

    def unload(self):
        self._loaded = False


class TTSEngine:
    """TTS 엔진 래퍼 — Edge TTS (클라우드) 또는 StyleTTS2 (로컬) 지원."""

    def __init__(self, device: str = "cuda:0", engine_type: str = TTS_ENGINE_TYPE):
        self.device = device
        self.engine_type = engine_type
        self.engine = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return

        if self.engine_type == "edge":
            logger.info("Loading Edge TTS (cloud)...")
            if ZHISPER_TTS_PATH not in sys.path:
                sys.path.insert(0, ZHISPER_TTS_PATH)
            from tts.engine.edge_engine import EdgeTTSEngine
            self.engine = EdgeTTSEngine()
            self.engine.load()
        elif self.engine_type == "styletts2":
            logger.info("Loading StyleTTS2...")
            import os
            if ZHISPER_TTS_PATH not in sys.path:
                sys.path.insert(0, ZHISPER_TTS_PATH)
            from tts.engine.styletts2_engine import StyleTTS2Engine
            self.engine = StyleTTS2Engine(
                device=self.device,
                config_path=os.path.join(STYLETTS2_PATH, "Models/KB_60h/config_ko_finetune_KB_60h.yml"),
                model_path=os.path.join(STYLETTS2_PATH, "Models/KB_60h/epoch_2nd_00034.pth"),
                ref_audio_path=TTS_REF_AUDIO,
            )
            self.engine.load()

        self._loaded = True
        logger.info(f"TTS Engine ({self.engine_type}) ready")

    def synthesize_to_pcm16(self, text: str) -> tuple:
        """텍스트 → (PCM int16 bytes, sample_rate)."""
        if not self._loaded:
            self.load()

        result = self.engine.synthesize(text=text, language="ko", speed=1.0)

        if self.engine_type == "edge":
            # Edge TTS: 이미 int16 numpy array 반환
            pcm = result.audio
            if pcm.dtype != np.int16:
                pcm = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
            return pcm.tobytes(), result.sample_rate
        else:
            # StyleTTS2: float32 [-1,1] 반환
            pcm = (result.audio * 32767).clip(-32768, 32767).astype(np.int16)
            return pcm.tobytes(), result.sample_rate

    def is_loaded(self) -> bool:
        return self._loaded

    def unload(self):
        if self.engine:
            self.engine.unload()
            self.engine = None
        self._loaded = False


class S2SPipeline:
    """STT Final → LLM (vLLM API) → TTS 파이프라인.

    Usage:
        pipeline = S2SPipeline()
        pipeline.load()

        async for event in pipeline.process("카드 분실 신고하려고요"):
            ...
    """

    def __init__(self, device: str = "cuda:0", knowledge_client=None):
        self.llm = LLMEngine()
        self.tts = TTSEngine(device=device)
        self.knowledge_client = knowledge_client
        self._loaded = False

    def load(self):
        """TTS 모델 로드 + LLM 클라이언트 초기화."""
        if self._loaded:
            return
        self.tts.load()
        self.llm.load()
        self._loaded = True
        logger.info("S2S Pipeline ready (LLM: vLLM API, TTS: StyleTTS2)")

    async def process(
        self,
        user_text: str,
        system_prompt: str = SYSTEM_PROMPT,
        cancel_event: asyncio.Event = None,
        audio_context: dict = None,
    ) -> AsyncGenerator[dict, None]:
        """STT Final 텍스트를 받아 LLM 응답 + TTS 음성을 스트리밍 생성.

        Args:
            cancel_event: set되면 즉시 생성 중단 (barge-in).
            audio_context: 프로소디 정보 {"energy", "speech_rate", "energy_trend"}

        Yields:
            {"type": "llm_start"}
            {"type": "llm_token", "token": str, "full_text": str}
            {"type": "tts_start", "sentence": str}
            {"type": "tts_audio", "audio": bytes, "sample_rate": int, ...}
            {"type": "llm_done", "full_text": str}
            {"type": "s2s_done", "latency": {...}, "emotion": str}
        """
        t_start = time.time()

        def _cancelled():
            return cancel_event is not None and cancel_event.is_set()

        # Use Knowledge service prompt if available, else fall back to default
        if self.knowledge_client and self.knowledge_client.is_loaded():
            kb_prompt = self.knowledge_client.get_system_prompt()
            if kb_prompt:
                system_prompt = kb_prompt
            # Append FAQ context
            faq_ctx = self.knowledge_client.get_faq_context(user_text)
            if faq_ctx:
                # Prepend FAQ to user message
                if audio_context:
                    user_msg = (
                        f"[audio: energy={audio_context['energy']}, "
                        f"speech_rate={audio_context['speech_rate']}, "
                        f"tone_trend={audio_context['energy_trend']}]\n"
                        f"{faq_ctx}\n"
                        f"고객: {user_text}"
                    )
                else:
                    user_msg = f"{faq_ctx}\n고객: {user_text}"
            else:
                # No FAQ match — build user_msg normally below
                user_msg = None
        else:
            user_msg = None

        # vLLM 서버 상태 확인
        if not await self.llm.check_health():
            yield {"type": "s2s_error", "error": "vLLM 서버에 연결할 수 없습니다 (port 8000)"}
            return

        # 프로소디 메타데이터를 사용자 메시지에 포함 (knowledge client가 설정하지 않은 경우)
        if user_msg is None:
            if audio_context:
                user_msg = (
                    f"[audio: energy={audio_context['energy']}, "
                    f"speech_rate={audio_context['speech_rate']}, "
                    f"tone_trend={audio_context['energy_trend']}]\n"
                    f"고객: {user_text}"
                )
            else:
                user_msg = f"고객: {user_text}"

        yield {"type": "llm_start"}

        full_text = ""
        chunk_buffer = ""
        detected_emotion = "neutral"
        emotion_parsed = False
        # 절(clause) 단위 구분자: 쉼표, 마침표, 접속어 뒤 등
        # 짧은 단위로 끊어서 TTS에 먼저 보냄
        flush_delimiters = {".", "?", "!", "。", ",", "，", ";"}
        t_first_token = None
        t_first_audio = None
        chunks_synthesized = 0
        MIN_CHUNK_CHARS = 6   # 최소 글자 수 (너무 짧으면 TTS 품질 저하)

        loop = asyncio.get_event_loop()

        async for token in self.llm.generate_stream(user_msg, system_prompt):
            if _cancelled():
                logger.info("[S2S] Cancelled during LLM generation (barge-in)")
                break

            if t_first_token is None:
                t_first_token = time.time()

            full_text += token

            # [EMOTION:xxx] 태그 파싱 (첫 줄에서)
            if not emotion_parsed:
                m = EMOTION_TAG_PATTERN.search(full_text)
                if m:
                    detected_emotion = m.group(1)
                    emotion_parsed = True
                    # 태그를 텍스트에서 제거 (TTS로 읽지 않도록)
                    full_text = full_text[m.end():].lstrip('\n')
                    chunk_buffer = full_text
                    logger.info(f"[S2S] Detected emotion: {detected_emotion}")
                    yield {"type": "s2s_emotion", "emotion": detected_emotion}
                    continue
                # 아직 태그가 완성되지 않았으면 TTS에 보내지 않음
                if len(full_text) < 40:
                    continue

            chunk_buffer += token

            yield {"type": "llm_token", "token": token, "full_text": full_text}

            # 절/문장 끝 감지 → 즉시 TTS 합성
            stripped = chunk_buffer.rstrip()
            if (len(stripped) >= MIN_CHUNK_CHARS and
                    any(stripped.endswith(d) for d in flush_delimiters)):
                chunk_text = stripped
                if chunk_text and not _cancelled():
                    yield {"type": "tts_start", "sentence": chunk_text}

                    pcm_bytes, sr = await loop.run_in_executor(
                        None, self.tts.synthesize_to_pcm16, chunk_text
                    )

                    if _cancelled():
                        break

                    if t_first_audio is None:
                        t_first_audio = time.time()

                    chunks_synthesized += 1
                    yield {
                        "type": "tts_audio",
                        "audio": pcm_bytes,
                        "sample_rate": sr,
                        "sentence": chunk_text,
                        "sentence_idx": chunks_synthesized,
                    }

                chunk_buffer = ""

        # 남은 버퍼 처리 (취소되지 않은 경우만)
        remaining = chunk_buffer.strip()
        if remaining and not _cancelled():
            yield {"type": "tts_start", "sentence": remaining}

            pcm_bytes, sr = await loop.run_in_executor(
                None, self.tts.synthesize_to_pcm16, remaining
            )

            if not _cancelled():
                if t_first_audio is None:
                    t_first_audio = time.time()

                chunks_synthesized += 1
                yield {
                    "type": "tts_audio",
                    "audio": pcm_bytes,
                    "sample_rate": sr,
                    "sentence": remaining,
                    "sentence_idx": chunks_synthesized,
                }

        t_end = time.time()
        was_cancelled = _cancelled()
        latency = {
            "total_ms": round((t_end - t_start) * 1000),
            "ttft_ms": round((t_first_token - t_start) * 1000) if t_first_token else None,
            "first_audio_ms": round((t_first_audio - t_start) * 1000) if t_first_audio else None,
            "sentences": chunks_synthesized,
            "cancelled": was_cancelled,
        }

        yield {"type": "llm_done", "full_text": full_text}
        yield {"type": "s2s_done", "latency": latency, "emotion": detected_emotion}

    def is_loaded(self) -> bool:
        return self._loaded

    def unload(self):
        self.llm.unload()
        self.tts.unload()
        self._loaded = False
