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
VLLM_MODEL = "/mnt/usb/models/Qwen3.5-9B-AWQ"
LLM_MAX_TOKENS = 80
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.9

# ── TTS 설정 ──
TTS_SAMPLE_RATE = 24000
TTS_REF_AUDIO = "/home/jonghooy/work/zhisper/tts/voices/styletts2/00003.wav"
TTS_REF_AUDIO_EN = "/home/jonghooy/work/zhisper/styletts2_korean/Models/LibriTTS_EN/reference_audio/reference_audio/4077-13754-0000.wav"
STYLETTS2_EN_CONFIG = "/home/jonghooy/work/zhisper/styletts2_korean/Models/LibriTTS_EN/Vocos/LibriTTS/config_libritts_vocos.yml"
STYLETTS2_EN_MODEL = "/home/jonghooy/work/zhisper/styletts2_korean/Models/LibriTTS_EN/Vocos/LibriTTS/epoch_2nd_00029.pth"

# ── 시스템 프롬프트 ──
SYSTEM_PROMPT = """당신은 친절하고 전문적인 한국어 콜센터 상담원입니다.

절대 규칙 (반드시 지켜야 함):
1. 첫 줄에 [EMOTION:anger|anxiety|neutral|satisfaction|joy] 태그를 출력하세요.
2. 답변은 반드시 1문장, 최대 2문장으로 제한하세요. 절대 3문장 이상 금지.
3. TTS로 읽히므로 자연스러운 구어체를 사용하세요.
4. 마크다운, 별표, 괄호 등 특수 기호를 사용하지 마세요.

감정별 응대:
- anger: 사과 + 즉각 해결
- anxiety: 안심 + 안내
- neutral: 친절하게 답변
- satisfaction/joy: 감사 + 밝게 응대"""

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
        messages: list,
        max_tokens: int = LLM_MAX_TOKENS,
        temperature: float = LLM_TEMPERATURE,
    ) -> AsyncGenerator[str, None]:
        """vLLM API 스트리밍 생성 — 토큰 단위로 yield."""
        if not self._loaded:
            self.load()

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": LLM_TOP_P,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        async with self._client.stream(
            "POST", "/chat/completions", json=payload
        ) as response:
            # vLLM context length 초과 시 400 에러
            if response.status_code == 400:
                body = await response.aread()
                raise ValueError(f"vLLM rejected request: {body.decode()}")
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
            config = self._config_path or os.path.join(STYLETTS2_PATH, "Models/KB_60h/config_ko_finetune_KB_60h.yml")
            model_file = self._model_path or os.path.join(STYLETTS2_PATH, "Models/KB_60h/epoch_2nd_00034.pth")
            ref = self._ref_audio_path or TTS_REF_AUDIO
            self.engine = StyleTTS2Engine(
                device=self.device,
                config_path=config,
                model_path=model_file,
                ref_audio_path=ref,
            )
            self.engine.load()

        self._loaded = True
        logger.info(f"TTS Engine ({self.engine_type}) ready")

    def synthesize_to_pcm16(self, text: str, language: str = "ko") -> tuple:
        """텍스트 → (PCM int16 bytes, sample_rate)."""
        if not self._loaded:
            self.load()

        result = self.engine.synthesize(text=text, language=language, speed=1.0)

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


MAX_HISTORY_CHARS = 1200
MAX_HISTORY_TURNS = 6


def _truncate_history(history: list, max_chars: int = MAX_HISTORY_CHARS, max_turns: int = MAX_HISTORY_TURNS) -> list:
    """히스토리를 최근 N턴, max_chars 이내로 잘라냄.

    1턴 = user + assistant 쌍 (연속 user만 있을 수도 있음).
    오래된 메시지부터 제거. 단일 메시지가 max_chars 초과하면 그대로 반환.
    """
    if not history:
        return []

    # 최대 턴 수 제한 (user 메시지 기준으로 카운트)
    user_indices = [i for i, m in enumerate(history) if m["role"] == "user"]
    if len(user_indices) > max_turns:
        start = user_indices[-max_turns]
        history = history[start:]

    # 글자 수 제한 — 최소 1개 메시지는 유지
    total_chars = sum(len(m["content"]) for m in history)
    while total_chars > max_chars and len(history) > 1:
        removed = history[0]
        total_chars -= len(removed["content"])
        history = history[1:]
        # user로 시작하도록 보정 (잘린 assistant만 남는 경우 방지)
        if len(history) > 1 and history[0]["role"] == "assistant":
            total_chars -= len(history[0]["content"])
            history = history[1:]

    # 최종 보정: assistant로 시작하면 제거 (orphaned assistant 방지)
    if history and history[0]["role"] == "assistant":
        history = history[1:]

    return history if history else []


def _get_rag_top_k(history: list) -> int:
    """히스토리 길이에 따라 RAG top_k를 동적 결정."""
    if not history:
        return 3
    total_chars = sum(len(m["content"]) for m in history)
    if total_chars > 900:
        return 1
    if total_chars > 600:
        return 2
    return 3


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
        self.tts = TTSEngine(device=device)  # Korean TTS (default)
        self.tts_en = TTSEngine(device=device, engine_type="edge")  # English TTS (Edge TTS cloud)
        self.knowledge_client = knowledge_client
        self._loaded = False

    def load(self):
        """TTS 모델 로드 + LLM 클라이언트 초기화."""
        if self._loaded:
            return
        self.tts.load()
        self.tts_en.load()
        self.llm.load()
        self._loaded = True
        logger.info("S2S Pipeline ready (LLM: vLLM API, TTS: KO+EN StyleTTS2)")

    async def _generate_with_retry(self, messages, truncated, system_prompt, user_msg):
        """LLM 스트리밍 생성. context 초과 시 히스토리 절반으로 1회 재시도."""
        try:
            async for token in self.llm.generate_stream(messages):
                yield token
        except ValueError as e:
            err_msg = str(e).lower()
            if ("context" in err_msg or "length" in err_msg) and truncated:
                logger.warning("[S2S] Context overflow, retrying with halved history")
                half = _truncate_history(truncated, max_chars=MAX_HISTORY_CHARS // 2)
                retry_messages = [{"role": "system", "content": system_prompt}]
                retry_messages.extend(half)
                retry_messages.append({"role": "user", "content": user_msg})
                async for token in self.llm.generate_stream(retry_messages):
                    yield token
            else:
                raise

    async def process(
        self,
        user_text: str,
        system_prompt: str = SYSTEM_PROMPT,
        cancel_event: asyncio.Event = None,
        audio_context: dict = None,
        history: list = None,
        language: str = "ko",
    ) -> AsyncGenerator[dict, None]:
        """STT Final 텍스트를 받아 LLM 응답 + TTS 음성을 스트리밍 생성.

        Args:
            cancel_event: set되면 즉시 생성 중단 (barge-in).
            audio_context: 프로소디 정보 {"energy", "speech_rate", "energy_trend"}
            language: "ko" (Korean) or "en" (English) — selects system prompt and TTS.

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

        # 언어별 시스템 프롬프트
        if language == "en":
            system_prompt = SYSTEM_PROMPT_EN
        # else: use the default SYSTEM_PROMPT (Korean), which may be overridden by Knowledge below

        # Select active TTS engine based on language
        active_tts = self.tts_en if language == "en" else self.tts
        logger.info(f"[S2S] Language={language}, TTS engine={active_tts.engine_type}")

        # Truncate history for context management
        truncated = _truncate_history(history or [])
        rag_top_k = _get_rag_top_k(truncated)

        # Use Knowledge service prompt + RAG search if available (Korean only)
        knowledge_ctx = ""
        if language != "en" and self.knowledge_client and self.knowledge_client.is_loaded():
            kb_prompt = self.knowledge_client.get_system_prompt()
            if kb_prompt:
                system_prompt = kb_prompt

            # 1. 실시간 RAG 검색 (문서에서 관련 정보 가져오기)
            try:
                rag_results = await self.knowledge_client.search(user_text, top_k=rag_top_k)
                if rag_results:
                    rag_lines = ["[참고 문서]"]
                    for r in rag_results:
                        text = r.get("text", "").replace("[문서:", "").split("]", 1)[-1].strip()
                        section = r.get("section", "")
                        if section:
                            rag_lines.append(f"[{section}] {text[:300]}")
                        else:
                            rag_lines.append(text[:300])
                    knowledge_ctx = "\n".join(rag_lines)
            except Exception as e:
                logger.warning(f"[S2S] RAG search failed: {e}")

            # 2. FAQ context (캐시된 FAQ에서 매칭)
            faq_ctx = self.knowledge_client.get_faq_context(user_text)
            if faq_ctx:
                knowledge_ctx = f"{knowledge_ctx}\n{faq_ctx}" if knowledge_ctx else faq_ctx

        # Build user message with knowledge context + audio context
        parts = []
        if audio_context:
            parts.append(
                f"[audio: energy={audio_context['energy']}, "
                f"speech_rate={audio_context['speech_rate']}, "
                f"tone_trend={audio_context['energy_trend']}]"
            )
        if knowledge_ctx:
            parts.append(knowledge_ctx)
        parts.append(f"고객: {user_text}")
        user_msg = "\n".join(parts)

        # Assemble messages list with system prompt + truncated history + current user message
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(truncated)
        messages.append({"role": "user", "content": user_msg})

        logger.info(f"[S2S] History: {len(truncated)} msgs, "
                    f"{sum(len(m['content']) for m in truncated)} chars, "
                    f"RAG top_k={rag_top_k}")

        # vLLM 서버 상태 확인
        if not await self.llm.check_health():
            yield {"type": "s2s_error", "error": "vLLM 서버에 연결할 수 없습니다 (port 8000)"}
            return

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

        async for token in self._generate_with_retry(messages, truncated, system_prompt, user_msg):
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
                # [EMOTION:satisfaction] = 24자가 최대
                if len(full_text) < 25:
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
                        None, active_tts.synthesize_to_pcm16, chunk_text, language
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
                None, active_tts.synthesize_to_pcm16, remaining, language
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
        self.tts_en.unload()
        self._loaded = False
