"""Test graph walker."""
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from realtime_demo.dialogue.models import Scenario, DialogueResult
from realtime_demo.dialogue.graph_walker import GraphWalker

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_scenario() -> Scenario:
    return Scenario.from_json_file(FIXTURE_DIR / "card_lost_scenario.json")


def _make_state(scenario_id="card-lost", current_node="n1"):
    return {
        "scenario_id": scenario_id,
        "scenario_version": 1,
        "current_node": current_node,
        "slots": {},
        "variables": {},
        "history": [],
        "retry_count": 0,
        "prosody": None,
        "stack": [],
        "awaiting_confirm": False,
    }


class TestSpeakNode:
    @pytest.mark.asyncio
    async def test_speak_returns_text_and_advances(self):
        scenario = _load_scenario()
        state = _make_state(current_node="n1")
        walker = GraphWalker(scenario, slot_manager=MagicMock(), llm_engine=None)
        result = await walker.step(state, "")
        assert "카드 분실 신고" in result.response_text
        assert state["current_node"] == "n2"
        assert result.awaiting_input is True


class TestSlotCollectNode:
    @pytest.mark.asyncio
    async def test_slot_collect_success(self):
        scenario = _load_scenario()
        state = _make_state(current_node="n2")
        mock_sm = MagicMock()
        mock_sm.extract = AsyncMock(return_value="1234-5678-9012-3456")
        walker = GraphWalker(scenario, slot_manager=mock_sm, llm_engine=None)
        result = await walker.step(state, "카드번호는 1234-5678-9012-3456이에요")
        assert state["slots"]["card_number"] == "1234-5678-9012-3456"
        assert state["retry_count"] == 0

    @pytest.mark.asyncio
    async def test_slot_collect_retry(self):
        scenario = _load_scenario()
        state = _make_state(current_node="n2")
        mock_sm = MagicMock()
        mock_sm.extract = AsyncMock(return_value=None)
        walker = GraphWalker(scenario, slot_manager=mock_sm, llm_engine=None)
        result = await walker.step(state, "잠깐만요")
        assert state["retry_count"] == 1
        assert "다시" in result.response_text
        assert state["current_node"] == "n2"

    @pytest.mark.asyncio
    async def test_slot_collect_max_retry_transfer(self):
        scenario = _load_scenario()
        state = _make_state(current_node="n2")
        state["retry_count"] = 2
        mock_sm = MagicMock()
        mock_sm.extract = AsyncMock(return_value=None)
        walker = GraphWalker(scenario, slot_manager=mock_sm, llm_engine=None)
        result = await walker.step(state, "모르겠어요")
        assert result.action is not None
        assert result.action["type"] == "transfer"


class TestConditionNode:
    @pytest.mark.asyncio
    async def test_condition_true_branch(self):
        scenario = _load_scenario()
        state = _make_state(current_node="n5")
        state["slots"]["loss_type"] = "도난"
        walker = GraphWalker(scenario, slot_manager=MagicMock(), llm_engine=None)
        result = await walker.step(state, "")
        assert "경찰 신고" in result.response_text

    @pytest.mark.asyncio
    async def test_condition_false_branch(self):
        scenario = _load_scenario()
        state = _make_state(current_node="n5")
        state["slots"]["loss_type"] = "분실"
        walker = GraphWalker(scenario, slot_manager=MagicMock(), llm_engine=None)
        result = await walker.step(state, "")
        assert result.response_text is None or "경찰" not in (result.response_text or "")


class TestConfirmNode:
    @pytest.mark.asyncio
    async def test_confirm_asks_question(self):
        scenario = _load_scenario()
        state = _make_state(current_node="n4")
        state["slots"] = {"card_number": "1234-5678", "loss_type": "분실"}
        walker = GraphWalker(scenario, slot_manager=MagicMock(), llm_engine=None)
        result = await walker.step(state, "")
        assert "1234-5678" in result.response_text
        assert "분실" in result.response_text
        assert result.awaiting_input is True


class TestEndNode:
    @pytest.mark.asyncio
    async def test_end_returns_action(self):
        scenario = _load_scenario()
        state = _make_state(current_node="n8")
        walker = GraphWalker(scenario, slot_manager=MagicMock(), llm_engine=None)
        result = await walker.step(state, "")
        assert result.action is not None
        assert result.action["type"] == "end"
        assert result.action["disposition"] == "resolved"
        assert "완료" in result.response_text


class TestInfiniteLoopDetection:
    @pytest.mark.asyncio
    async def test_loop_detection(self):
        scenario = _load_scenario()
        state = _make_state(current_node="n1")
        state["history"] = ["n1", "n1", "n1"]
        walker = GraphWalker(scenario, slot_manager=MagicMock(), llm_engine=None)
        result = await walker.step(state, "")
        assert result.action is not None
        assert result.action["type"] == "end"
        assert "loop" in result.action.get("reason", "").lower() or result.action.get("disposition") == "error"
