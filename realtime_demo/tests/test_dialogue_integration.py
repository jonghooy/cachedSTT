"""Integration test: full scenario walkthrough without external services."""
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from realtime_demo.dialogue.engine import DialogueEngine
from realtime_demo.dialogue.scenario_cache import ScenarioCache
from realtime_demo.dialogue.slot_manager import SlotManager
from realtime_demo.dialogue.intent_matcher import IntentMatcher
from realtime_demo.dialogue.action_runner import ActionRunner
from realtime_demo.dialogue.models import Scenario, IntentMatch

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _setup():
    """Create engine with fixture scenario and mocked intent matcher."""
    with open(FIXTURE_DIR / "card_lost_scenario.json") as f:
        data = json.load(f)
    cache = ScenarioCache()
    cache.load_from_dicts([data])

    scenario = cache.get_scenario("card-lost")

    matcher = AsyncMock()
    matcher.match = AsyncMock(return_value=IntentMatch(
        scenario_id="card-lost", scenario=scenario, confidence=0.95
    ))
    matcher.set_scenarios = MagicMock()

    mock_llm = AsyncMock()
    sm = SlotManager(llm_engine=mock_llm)

    engine = DialogueEngine(
        scenario_cache=cache,
        intent_matcher=matcher,
        slot_manager=sm,
        llm_engine=mock_llm,
        action_runner=ActionRunner(),
    )

    session = {
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
    return engine, session, mock_llm


class TestFullScenarioWalkthrough:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        """Complete scenario: entry -> card_number -> loss_type -> confirm -> condition -> end."""
        engine, session, mock_llm = _setup()

        # Turn 1: enters scenario, gets greeting + asks card number
        r1 = await engine.process_utterance(session, "카드 잃어버렸어요", None)
        assert session["dialogue_mode"] == "scenario"
        assert r1.response_text is not None
        assert "카드 분실 신고" in r1.response_text
        assert r1.awaiting_input is True

        # Turn 2: Provide card number (regex extraction)
        r2 = await engine.process_utterance(session, "1234 5678 9012 3456입니다", None)
        assert session["scenario_state"]["slots"].get("card_number") is not None

        # Turn 3: Provide loss type (LLM extraction); engine auto-advances to confirm node
        async def mock_stream(*args, **kwargs):
            yield "분실"
        mock_llm.generate_stream = AsyncMock(return_value=mock_stream())

        r3 = await engine.process_utterance(session, "분실이에요", None)
        assert session["scenario_state"]["slots"].get("loss_type") == "분실"
        # After slot collection, engine auto-advances to confirm node
        assert r3.awaiting_input is True
        assert "맞으시죠" in r3.response_text

        # Turn 4: Yes -> condition -> api_call action
        r4 = await engine.process_utterance(session, "네 맞아요", None)
        assert r4.action is not None

    @pytest.mark.asyncio
    async def test_no_match_stays_freeform(self):
        engine, session, _ = _setup()
        engine.intent_matcher.match = AsyncMock(return_value=None)
        result = await engine.process_utterance(session, "오늘 날씨가 좋네요", None)
        assert result.should_use_s2s is True
        assert session["dialogue_mode"] == "freeform"
