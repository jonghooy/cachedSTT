"""Test slot manager."""
import pytest
from unittest.mock import AsyncMock

from realtime_demo.dialogue.models import SlotDef
from realtime_demo.dialogue.slot_manager import SlotManager


class TestSlotManagerRegex:
    def test_extract_card_number(self):
        slot = SlotDef(name="card_number", type="string", required=True,
                       extract="regex", prompt="", pattern=r"\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}")
        sm = SlotManager(llm_engine=None)
        result = sm.extract_regex(slot, "카드번호는 1234-5678-9012-3456이에요")
        assert result == "1234-5678-9012-3456"

    def test_extract_card_number_spaces(self):
        slot = SlotDef(name="card_number", type="string", required=True,
                       extract="regex", prompt="", pattern=r"\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}")
        sm = SlotManager(llm_engine=None)
        result = sm.extract_regex(slot, "1234 5678 9012 3456입니다")
        assert result == "1234 5678 9012 3456"

    def test_no_match_returns_none(self):
        slot = SlotDef(name="card_number", type="string", required=True,
                       extract="regex", prompt="", pattern=r"\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}")
        sm = SlotManager(llm_engine=None)
        result = sm.extract_regex(slot, "잠깐만요")
        assert result is None


class TestSlotManagerLLM:
    @pytest.mark.asyncio
    async def test_extract_enum_via_llm(self):
        mock_llm = AsyncMock()
        mock_llm.generate_stream = AsyncMock(return_value=_async_iter(["분실"]))
        sm = SlotManager(llm_engine=mock_llm)
        slot = SlotDef(name="loss_type", type="enum", required=True,
                       extract="llm", prompt="", values=["분실", "도난"])
        result = await sm.extract_llm(slot, "카드를 잃어버렸어요")
        assert result == "분실"

    @pytest.mark.asyncio
    async def test_extract_enum_invalid_returns_none(self):
        mock_llm = AsyncMock()
        mock_llm.generate_stream = AsyncMock(return_value=_async_iter(["모르겠어요"]))
        sm = SlotManager(llm_engine=mock_llm)
        slot = SlotDef(name="loss_type", type="enum", required=True,
                       extract="llm", prompt="", values=["분실", "도난"])
        result = await sm.extract_llm(slot, "잘 모르겠어요")
        assert result is None


async def _async_iter(items):
    for item in items:
        yield item
