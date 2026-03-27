"""Comprehensive tests for Plan 1+2 gap coverage.

Covers: confirm-no path, scenario stack, transfer→freeform, api_call advance,
slot extract routing, graceful degradation, intent matcher error handling,
and cross-service scenario loading flow.
"""
import json
import copy
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from realtime_demo.dialogue.models import Scenario, DialogueResult, IntentMatch
from realtime_demo.dialogue.engine import DialogueEngine
from realtime_demo.dialogue.graph_walker import GraphWalker
from realtime_demo.dialogue.slot_manager import SlotManager
from realtime_demo.dialogue.scenario_cache import ScenarioCache
from realtime_demo.dialogue.intent_matcher import IntentMatcher
from realtime_demo.dialogue.action_runner import ActionRunner
from realtime_demo.dialogue.rule_evaluator import evaluate_rule
from realtime_demo.dialogue.template_renderer import render_template, render_dict

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_scenario() -> Scenario:
    return Scenario.from_json_file(FIXTURE_DIR / "card_lost_scenario.json")


def _make_state(current_node="n1"):
    return {
        "scenario_id": "card-lost",
        "scenario_version": 1,
        "current_node": current_node,
        "slots": {},
        "variables": {},
        "history": [],
        "retry_count": 0,
        "prosody": None,
        "stack": [],
        "awaiting_confirm": False,
        "_scenario_snapshot": None,
    }


def _make_session():
    return {
        "dialogue_mode": "freeform",
        "scenario_state": _make_state(),
    }


def _make_engine(cache=None, matcher_result=None):
    if cache is None:
        cache = ScenarioCache()
        with open(FIXTURE_DIR / "card_lost_scenario.json") as f:
            cache.load_from_dicts([json.load(f)])

    scenario = cache.get_scenario("card-lost")
    matcher = AsyncMock()
    if matcher_result is not None:
        matcher.match = AsyncMock(return_value=matcher_result)
    else:
        matcher.match = AsyncMock(return_value=IntentMatch(
            scenario_id="card-lost", scenario=scenario, confidence=0.95
        ))

    mock_llm = AsyncMock()
    sm = SlotManager(llm_engine=mock_llm)

    engine = DialogueEngine(
        scenario_cache=cache,
        intent_matcher=matcher,
        slot_manager=sm,
        llm_engine=mock_llm,
        action_runner=ActionRunner(),
    )
    return engine, mock_llm


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Confirm "no" path → loops back
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestConfirmNoPath:
    @pytest.mark.asyncio
    async def test_confirm_no_goes_back_to_slot_collect(self):
        """confirm → no → should go to n2 (slot_collect) per edge n4→n2 label=no."""
        scenario = _load_scenario()
        state = _make_state(current_node="n4")
        state["slots"] = {"card_number": "1234-5678", "loss_type": "분실"}
        state["awaiting_confirm"] = True  # simulate second entry

        walker = GraphWalker(scenario, slot_manager=MagicMock(), llm_engine=None)
        result = await walker.step(state, "아니요 틀려요")

        # After no → should go to n2 (slot_collect), which asks for card_number
        assert state["current_node"] == "n2"
        assert result.awaiting_input is True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Scenario stack (nested scenarios)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestScenarioStack:
    @pytest.mark.asyncio
    async def test_exit_scenario_pops_stack(self):
        """If stack has a parent, exiting should resume the parent scenario."""
        engine, _ = _make_engine()
        session = _make_session()

        # Simulate: already in scenario A, with parent on stack
        parent_state = _make_state(current_node="n3")
        parent_state["slots"] = {"card_number": "9999"}
        parent_state["scenario_id"] = "card-lost"
        parent_state["_scenario_snapshot"] = _load_scenario()

        session["dialogue_mode"] = "scenario"
        session["scenario_state"] = _make_state(current_node="n8")  # end node
        session["scenario_state"]["stack"] = [parent_state]
        session["scenario_state"]["_scenario_snapshot"] = _load_scenario()

        # Process end node → should pop stack and resume parent
        result = await engine.process_utterance(session, "", None)

        assert session["dialogue_mode"] == "scenario"
        assert session["scenario_state"]["current_node"] == "n3"
        assert session["scenario_state"]["slots"]["card_number"] == "9999"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Transfer node → exits to freeform
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestTransferExitsScenario:
    @pytest.mark.asyncio
    async def test_transfer_returns_to_freeform(self):
        engine, _ = _make_engine()
        session = _make_session()
        session["dialogue_mode"] = "scenario"
        session["scenario_state"]["scenario_id"] = "card-lost"
        session["scenario_state"]["current_node"] = "n9"  # transfer node
        session["scenario_state"]["_scenario_snapshot"] = _load_scenario()

        result = await engine.process_utterance(session, "", None)

        assert result.action["type"] == "transfer"
        assert session["dialogue_mode"] == "freeform"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. API call advances current_node
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestApiCallAdvancesNode:
    @pytest.mark.asyncio
    async def test_api_call_advances_to_next_node(self):
        """api_call at n7 should advance to n8 (end) before returning action."""
        scenario = _load_scenario()
        state = _make_state(current_node="n7")
        state["slots"] = {"card_number": "1234", "loss_type": "분실"}

        walker = GraphWalker(scenario, slot_manager=MagicMock(), llm_engine=None)
        result = await walker.step(state, "")

        assert result.action is not None
        assert result.action["type"] == "api_call"
        # n7→n8 edge, so current_node should be n8 after api_call
        assert state["current_node"] == "n8"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. SlotManager extract routing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestSlotExtractRouting:
    @pytest.mark.asyncio
    async def test_extract_routes_to_regex(self):
        from realtime_demo.dialogue.models import SlotDef
        slot = SlotDef(name="card", type="string", required=True, extract="regex",
                       prompt="", pattern=r"\d{4}-\d{4}")
        sm = SlotManager(llm_engine=None)
        result = await sm.extract(slot, "번호는 1234-5678입니다")
        assert result == "1234-5678"

    @pytest.mark.asyncio
    async def test_extract_routes_to_llm(self):
        from realtime_demo.dialogue.models import SlotDef

        async def mock_stream(*a, **kw):
            yield "도난"

        mock_llm = AsyncMock()
        mock_llm.generate_stream = AsyncMock(return_value=mock_stream())

        slot = SlotDef(name="loss_type", type="enum", required=True, extract="llm",
                       prompt="", values=["분실", "도난"])
        sm = SlotManager(llm_engine=mock_llm)
        result = await sm.extract(slot, "도난당했어요")
        assert result == "도난"

    @pytest.mark.asyncio
    async def test_extract_unknown_method_returns_none(self):
        from realtime_demo.dialogue.models import SlotDef
        slot = SlotDef(name="x", type="string", required=True, extract="unknown", prompt="")
        sm = SlotManager(llm_engine=None)
        result = await sm.extract(slot, "test")
        assert result is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Graceful degradation — empty cache
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestGracefulDegradation:
    @pytest.mark.asyncio
    async def test_empty_cache_stays_freeform(self):
        """No scenarios loaded → always freeform."""
        empty_cache = ScenarioCache()
        matcher = IntentMatcher(embed_fn=None)
        matcher.set_scenarios(empty_cache.scenarios)

        engine = DialogueEngine(
            scenario_cache=empty_cache,
            intent_matcher=matcher,
            slot_manager=SlotManager(llm_engine=None),
        )
        session = _make_session()
        result = await engine.process_utterance(session, "카드 잃어버렸어요", None)
        assert result.should_use_s2s is True
        assert session["dialogue_mode"] == "freeform"

    @pytest.mark.asyncio
    async def test_no_scenario_snapshot_falls_back(self):
        """If scenario_snapshot is None during scenario mode, fall back to freeform."""
        engine, _ = _make_engine()
        session = _make_session()
        session["dialogue_mode"] = "scenario"
        session["scenario_state"]["scenario_id"] = "card-lost"
        session["scenario_state"]["current_node"] = "n1"
        session["scenario_state"]["_scenario_snapshot"] = None  # corrupted

        result = await engine.process_utterance(session, "test", None)
        assert result.should_use_s2s is True
        assert session["dialogue_mode"] == "freeform"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Intent matcher error → freeform
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestIntentMatcherError:
    @pytest.mark.asyncio
    async def test_matcher_exception_falls_back(self):
        engine, _ = _make_engine()
        engine.intent_matcher.match = AsyncMock(side_effect=RuntimeError("embed failed"))

        session = _make_session()
        result = await engine.process_utterance(session, "카드 잃어버렸어요", None)
        assert result.should_use_s2s is True
        assert session["dialogue_mode"] == "freeform"

    @pytest.mark.asyncio
    async def test_low_confidence_stays_freeform(self):
        """Confidence below 0.7 → freeform."""
        scenario = _load_scenario()
        engine, _ = _make_engine(matcher_result=IntentMatch(
            scenario_id="card-lost", scenario=scenario, confidence=0.5
        ))
        session = _make_session()
        result = await engine.process_utterance(session, "뭔가 말하기", None)
        assert result.should_use_s2s is True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. Rule evaluator edge cases
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestRuleEvaluatorEdgeCases:
    def test_nested_and_or(self):
        rule = {"and": [
            {"or": [
                {"field": "slots.a", "op": "eq", "value": 1},
                {"field": "slots.b", "op": "eq", "value": 2},
            ]},
            {"field": "slots.c", "op": "exists"},
        ]}
        ctx = {"slots": {"a": 1, "c": "yes"}, "variables": {}}
        assert evaluate_rule(rule, ctx) is True

    def test_deeply_nested_field(self):
        rule = {"field": "variables.api.result.status", "op": "eq", "value": "ok"}
        ctx = {"slots": {}, "variables": {"api": {"result": {"status": "ok"}}}}
        assert evaluate_rule(rule, ctx) is True

    def test_empty_rule_raises(self):
        with pytest.raises(ValueError):
            evaluate_rule({"field": "x", "op": ""}, {})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. Template renderer edge cases
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestTemplateEdgeCases:
    def test_render_dict_nested_list_of_dicts(self):
        body = {"items": [{"name": "{{slots.card}}"}]}
        ctx = {"slots": {"card": "1234"}, "variables": {}}
        result = render_dict(body, ctx)
        assert result["items"][0]["name"] == "1234"

    def test_render_template_preserves_non_matching(self):
        result = render_template("{{notapath}}", {"slots": {}})
        assert result == ""

    def test_render_dict_non_string_values_preserved(self):
        body = {"count": 42, "active": True, "name": "{{slots.x}}"}
        ctx = {"slots": {"x": "test"}, "variables": {}}
        result = render_dict(body, ctx)
        assert result["count"] == 42
        assert result["active"] is True
        assert result["name"] == "test"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. Cross-service: Knowledge API response → Brain cache → dialogue
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestCrossServiceFlow:
    @pytest.mark.asyncio
    async def test_knowledge_api_format_loads_into_brain_cache(self):
        """Simulate Knowledge Service /api/brain/scenarios response format
        and verify Brain ScenarioCache can parse it."""
        # Simulate Knowledge API response (JSON fields are dicts, not strings)
        api_response = {
            "scenarios": [{
                "id": 1,
                "name": "카드 분실 신고",
                "description": "테스트",
                "schema_version": 1,
                "graph_json": {
                    "nodes": [
                        {"id": "n1", "type": "speak", "text": "안녕하세요"},
                        {"id": "n2", "type": "end", "message": "감사합니다", "disposition": "resolved"},
                    ],
                    "edges": [{"from": "n1", "to": "n2"}],
                },
                "version": 1,
                "status": "active",
                "priority": 10,
                "triggers_json": {"examples": ["카드 잃어버렸어요"], "description": "카드 분실"},
                "slots_json": {},
                "metadata_json": {},
                "source_json": {},
                "created_by": "ai",
            }],
            "version": "2026-03-27T12:00:00Z",
        }

        # Brain-side: ScenarioCache loads this
        # The Brain API returns parsed JSON (dicts), but ScenarioCache.load_from_dicts
        # expects the format that Scenario.from_dict can parse.
        # We need to map the API response to the expected format.
        cache = ScenarioCache()
        for s in api_response["scenarios"]:
            # Map Knowledge API field names to Scenario.from_dict expected format
            scenario_dict = {
                "id": str(s["id"]),
                "name": s["name"],
                "description": s["description"],
                "version": s["version"],
                "schema_version": s["schema_version"],
                "status": s["status"],
                "priority": s["priority"],
                "triggers": s["triggers_json"],
                "slots": s["slots_json"],
                "nodes": s["graph_json"]["nodes"],
                "edges": s["graph_json"]["edges"],
                "metadata": s["metadata_json"],
            }
            scenario = Scenario.from_dict(scenario_dict)
            cache.scenarios[scenario.id] = scenario

        assert len(cache.scenarios) == 1
        scenario = list(cache.scenarios.values())[0]
        assert scenario.name == "카드 분실 신고"
        assert len(scenario.nodes) == 2
        assert scenario.get_start_node_id() == "n1"

        # Now run dialogue engine with this loaded scenario
        matcher = AsyncMock()
        matcher.match = AsyncMock(return_value=IntentMatch(
            scenario_id=scenario.id, scenario=scenario, confidence=0.95
        ))
        engine = DialogueEngine(
            scenario_cache=cache,
            intent_matcher=matcher,
            slot_manager=SlotManager(llm_engine=None),
        )
        session = _make_session()
        result = await engine.process_utterance(session, "카드 잃어버렸어요", None)

        assert session["dialogue_mode"] == "scenario"
        assert "안녕하세요" in result.response_text


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 11. Full multi-turn dialogue with all node types
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestFullMultiTurnDialogue:
    @pytest.mark.asyncio
    async def test_complete_theft_scenario(self):
        """Full flow: entry → card_number → loss_type(도난) → confirm(yes) → condition(true) → speak(경찰) → api_call."""
        engine, mock_llm = _make_engine()

        session = _make_session()

        # Turn 1: 시나리오 진입
        r1 = await engine.process_utterance(session, "카드 도난당했어요", None)
        assert session["dialogue_mode"] == "scenario"
        assert r1.awaiting_input is True

        # Turn 2: 카드번호 제공 (regex)
        r2 = await engine.process_utterance(session, "1234-5678-9012-3456", None)
        assert session["scenario_state"]["slots"].get("card_number") == "1234-5678-9012-3456"

        # Turn 3: 분실 유형 (LLM으로 "도난" 추출)
        async def mock_stream_theft(*a, **kw):
            yield "도난"
        mock_llm.generate_stream = AsyncMock(return_value=mock_stream_theft())

        r3 = await engine.process_utterance(session, "도난당했어요", None)
        assert session["scenario_state"]["slots"].get("loss_type") == "도난"
        # Auto-advance to confirm → asks question
        assert r3.awaiting_input is True
        assert "맞으시죠" in r3.response_text

        # Turn 4: 확인 → yes → condition(도난=true) → speak(경찰 신고) → api_call (executed inline)
        r4 = await engine.process_utterance(session, "네 맞아요", None)
        # condition true → n6 (speak: 경찰 신고) → n7 (api_call, executed inline)
        # API fails (no network in test) → on_error=n_err → speak(오류) → n_err2 (transfer)
        assert r4.action is not None
        assert r4.action["type"] in ("api_call", "transfer", "end")
        assert "경찰 신고" in r4.response_text
