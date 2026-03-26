"""Slot extraction — regex for structured data, LLM for natural language."""
from __future__ import annotations

import logging
import re
from typing import Any

from realtime_demo.dialogue.models import SlotDef

logger = logging.getLogger(__name__)


class SlotManager:
    def __init__(self, llm_engine=None):
        self.llm_engine = llm_engine

    def extract_regex(self, slot: SlotDef, text: str) -> str | None:
        """Extract slot value via regex pattern match."""
        if not slot.pattern:
            return None
        match = re.search(slot.pattern, text)
        return match.group(0) if match else None

    async def extract_llm(self, slot: SlotDef, text: str) -> str | None:
        """Extract slot value via LLM."""
        if not self.llm_engine:
            logger.warning("LLM engine not available for slot extraction")
            return None

        if slot.type == "enum" and slot.values:
            prompt = f"사용자 발화: \"{text}\"\n선택지: {slot.values}\n위 발화에서 선택지 중 하나를 골라 해당 값만 출력하세요. 해당하는 값이 없으면 'NONE'을 출력하세요."
        elif slot.type == "boolean":
            prompt = f"사용자 발화: \"{text}\"\n위 발화가 긍정이면 'true', 부정이면 'false', 판단 불가면 'NONE'을 출력하세요."
        else:
            prompt = f"사용자 발화: \"{text}\"\n{slot.prompt}\n위 발화에서 해당 정보를 추출하세요. 없으면 'NONE'을 출력하세요."

        messages = [
            {"role": "system", "content": "You are a slot extraction assistant. Output ONLY the extracted value, nothing else."},
            {"role": "user", "content": prompt},
        ]

        try:
            result_text = ""
            stream = await self.llm_engine.generate_stream(messages, max_tokens=20, temperature=0.0)
            async for token in stream:
                result_text += token
            result_text = result_text.strip()

            if result_text.upper() == "NONE" or not result_text:
                return None

            # Validate enum values
            if slot.type == "enum" and slot.values:
                if result_text not in slot.values:
                    return None

            # Validate boolean
            if slot.type == "boolean":
                if result_text.lower() in ("true", "예", "네"):
                    return True
                elif result_text.lower() in ("false", "아니요", "아니오"):
                    return False
                return None

            return result_text
        except Exception:
            logger.exception("LLM slot extraction failed")
            return None

    async def extract(self, slot: SlotDef, text: str) -> Any:
        """Extract slot value using the configured method."""
        if slot.extract == "regex":
            return self.extract_regex(slot, text)
        elif slot.extract == "llm":
            return await self.extract_llm(slot, text)
        return None
