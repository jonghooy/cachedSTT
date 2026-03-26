"""Test dialogue engine."""
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from realtime_demo.dialogue.engine import DialogueEngine
from realtime_demo.dialogue.models import Scenario, DialogueResult, IntentMatch
from realtime_demo.dialogue.scenario_cache import ScenarioCache

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _make_session_state():
    return {
        "dialogue_mode": "freeform",
        "scenario_state": {
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
        },
    }


def _load_cache():
    with open(FIXTURE_DIR / "card_lost_scenario.json") as f:
        data = json.load(f)
    cache = ScenarioCache()
    cache.load_from_dicts([data])
    return cache


class TestFreeformMode:
    @pytest.mark.asyncio
    async def test_no_match_returns_freeform(self):
        cache = _load_cache()
        mock_matcher = AsyncMock()
        mock_matcher.match = AsyncMock(return_value=None)
        engine = DialogueEngine(
            scenario_cache=cache,
            intent_matcher=mock_matcher,
            slot_manager=MagicMock(),
            llm_engine=None,
        )
        session = _make_session_state()
        result = await engine.process_utterance(session, "오늘 날씨 좋네요", None)
        assert result.should_use_s2s is True
        assert session["dialogue_mode"] == "freeform"

    @pytest.mark.asyncio
    async def test_match_enters_scenario(self):
        cache = _load_cache()
        scenario = cache.get_scenario("card-lost")
        mock_matcher = AsyncMock()
        mock_matcher.match = AsyncMock(return_value=IntentMatch(
            scenario_id="card-lost", scenario=scenario, confidence=0.9, matched_example="카드 분실"
        ))
        engine = DialogueEngine(
            scenario_cache=cache,
            intent_matcher=mock_matcher,
            slot_manager=MagicMock(),
            llm_engine=None,
        )
        session = _make_session_state()
        result = await engine.process_utterance(session, "카드 잃어버렸어요", None)
        assert session["dialogue_mode"] == "scenario"
        assert session["scenario_state"]["scenario_id"] == "card-lost"
        assert "카드 분실 신고" in result.response_text


class TestScenarioMode:
    @pytest.mark.asyncio
    async def test_slot_collect_flow(self):
        cache = _load_cache()
        scenario = cache.get_scenario("card-lost")
        mock_sm = MagicMock()
        mock_sm.extract = AsyncMock(return_value="1234-5678-9012-3456")
        engine = DialogueEngine(
            scenario_cache=cache,
            intent_matcher=AsyncMock(),
            slot_manager=mock_sm,
            llm_engine=None,
        )
        session = _make_session_state()
        session["dialogue_mode"] = "scenario"
        session["scenario_state"]["scenario_id"] = "card-lost"
        session["scenario_state"]["scenario_version"] = 1
        session["scenario_state"]["current_node"] = "n2"
        session["scenario_state"]["_scenario_snapshot"] = scenario

        result = await engine.process_utterance(session, "1234-5678-9012-3456", None)
        assert session["scenario_state"]["slots"].get("card_number") == "1234-5678-9012-3456"


class TestEndReturnsToFreeform:
    @pytest.mark.asyncio
    async def test_end_node_exits(self):
        cache = _load_cache()
        scenario = cache.get_scenario("card-lost")
        engine = DialogueEngine(
            scenario_cache=cache,
            intent_matcher=AsyncMock(),
            slot_manager=MagicMock(),
            llm_engine=None,
        )
        session = _make_session_state()
        session["dialogue_mode"] = "scenario"
        session["scenario_state"]["scenario_id"] = "card-lost"
        session["scenario_state"]["current_node"] = "n8"
        session["scenario_state"]["_scenario_snapshot"] = scenario

        result = await engine.process_utterance(session, "", None)
        assert result.action is not None
        assert result.action["type"] == "end"
        assert session["dialogue_mode"] == "freeform"
