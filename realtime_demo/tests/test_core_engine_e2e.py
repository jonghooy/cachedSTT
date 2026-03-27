"""Core Engine End-to-End Test Suite

전체 핵심 엔진 동작 검증. 5개 테스트 계층:
1. 시나리오 로딩 (Knowledge → Brain 캐시)
2. Intent 매칭 (임베딩 유사도 + 신뢰도)
3. 전체 대화 흐름 (분실/도난 시나리오 각각)
4. 에러 처리 (LLM 다운, API 실패, on_error 라우팅)
5. 엔진 라이프사이클 (freeform ↔ scenario, 스택, 로그)
"""
import json
import copy
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from realtime_demo.dialogue.models import Scenario, DialogueResult, IntentMatch
from realtime_demo.dialogue.engine import DialogueEngine
from realtime_demo.dialogue.graph_walker import GraphWalker
from realtime_demo.dialogue.slot_manager import SlotManager
from realtime_demo.dialogue.scenario_cache import ScenarioCache
from realtime_demo.dialogue.intent_matcher import IntentMatcher, cosine_similarity
from realtime_demo.dialogue.action_runner import ActionRunner
from realtime_demo.dialogue.rule_evaluator import evaluate_rule
from realtime_demo.dialogue.template_renderer import render_template, render_dict

FIXTURE_DIR = Path(__file__).parent / "fixtures"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Shared helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _load_fixture():
    with open(FIXTURE_DIR / "card_lost_scenario.json") as f:
        return json.load(f)


def _make_session():
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


def _make_llm_mock():
    """Create a mock LLM engine that can be configured per-call."""
    mock = AsyncMock()
    return mock


def _make_llm_stream(text: str):
    """Create an async generator that yields the given text."""
    async def stream(*args, **kwargs):
        yield text
    return stream


_BASE_EMBEDDING = np.random.randn(64).astype(np.float32)


def _build_engine(scenario_data=None, embed_fn=None, llm_mock=None, api_mock=None):
    """Build a complete DialogueEngine with all components."""
    if scenario_data is None:
        scenario_data = _load_fixture()

    cache = ScenarioCache()
    cache.load_from_dicts([scenario_data])
    scenario = cache.get_scenario("card-lost")

    # Intent matcher with real cosine similarity
    if embed_fn is None:
        async def default_embed(text):
            card_keywords = ["카드", "분실", "잃어", "도난", "없어"]
            if any(kw in text for kw in card_keywords):
                return _BASE_EMBEDDING + np.random.randn(64).astype(np.float32) * 0.05
            return np.random.randn(64).astype(np.float32) * 10

        embed_fn = default_embed

    matcher = IntentMatcher(embed_fn=embed_fn, similarity_threshold=0.5)
    matcher.set_scenarios(cache.scenarios)
    # Load trigger embeddings (simulate what Knowledge Service provides)
    trigger_embeddings = [_BASE_EMBEDDING + np.random.randn(64).astype(np.float32) * 0.05
                          for _ in scenario_data["triggers"]["examples"]] if scenario else []
    if scenario:
        matcher.load_trigger_cache([{
            "scenario_id": scenario.id,
            "triggers": scenario_data["triggers"]["examples"],
            "embeddings": trigger_embeddings,
        }])

    if llm_mock is None:
        llm_mock = _make_llm_mock()

    sm = SlotManager(llm_engine=llm_mock)

    mock_knowledge = AsyncMock()
    mock_knowledge.search = AsyncMock(return_value=[{"text": "카드 분실 시 정지 처리", "score": 0.9}])
    mock_knowledge._client = AsyncMock()
    mock_knowledge._client.post = AsyncMock()

    action_runner = ActionRunner(knowledge_client=mock_knowledge)

    engine = DialogueEngine(
        scenario_cache=cache,
        intent_matcher=matcher,
        slot_manager=sm,
        llm_engine=llm_mock,
        action_runner=action_runner,
    )
    return engine, llm_mock, mock_knowledge


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Layer 1: 시나리오 로딩 검증
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestScenarioLoading:
    """Knowledge Service → Brain 캐시 로딩 검증"""

    def test_fixture_loads_all_nodes(self):
        data = _load_fixture()
        scenario = Scenario.from_dict(data)
        assert len(scenario.nodes) == 11
        assert len(scenario.edges) == 10
        assert len(scenario.slots) == 2

    def test_cache_filters_active_only(self):
        data = _load_fixture()
        data2 = copy.deepcopy(data)
        data2["id"] = "inactive"
        data2["status"] = "draft"
        cache = ScenarioCache()
        cache.load_from_dicts([data, data2])
        assert len(cache.scenarios) == 1
        assert "card-lost" in cache.scenarios

    def test_trigger_embeddings_loaded_into_matcher(self):
        engine, _, _ = _build_engine()
        assert len(engine.intent_matcher._trigger_cache) == 4  # 4 trigger examples

    def test_scenario_graph_connectivity(self):
        """Verify all edges point to valid nodes."""
        data = _load_fixture()
        scenario = Scenario.from_dict(data)
        node_ids = set(scenario.nodes.keys())
        for edge in scenario.edges:
            assert edge.from_id in node_ids, f"Edge from {edge.from_id} not in nodes"
            assert edge.to_id in node_ids, f"Edge to {edge.to_id} not in nodes"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Layer 2: Intent 매칭 검증
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestIntentMatching:
    """임베딩 유사도 기반 시나리오 매칭 검증"""

    @pytest.mark.asyncio
    async def test_card_related_text_matches(self):
        engine, _, _ = _build_engine()
        session = _make_session()
        result = await engine.process_utterance(session, "카드를 잃어버렸어요", None)
        assert session["dialogue_mode"] == "scenario"
        assert result.should_use_s2s is False

    @pytest.mark.asyncio
    async def test_unrelated_text_stays_freeform(self):
        engine, _, _ = _build_engine()
        session = _make_session()
        result = await engine.process_utterance(session, "오늘 날씨가 어때요", None)
        assert session["dialogue_mode"] == "freeform"
        assert result.should_use_s2s is True

    @pytest.mark.asyncio
    async def test_low_confidence_stays_freeform(self):
        """When similarity threshold is set very high, no match occurs."""
        # Use embed_fn that returns dissimilar vectors
        async def low_sim_embed(text):
            return np.random.randn(64).astype(np.float32) * 50  # very different from triggers

        engine, _, _ = _build_engine(embed_fn=low_sim_embed)
        session = _make_session()
        result = await engine.process_utterance(session, "카드 분실", None)
        assert result.should_use_s2s is True

    @pytest.mark.asyncio
    async def test_embed_fn_failure_graceful(self):
        async def broken_embed(text):
            raise RuntimeError("Embedding service down")

        engine, _, _ = _build_engine(embed_fn=broken_embed)
        session = _make_session()
        result = await engine.process_utterance(session, "카드 분실", None)
        assert result.should_use_s2s is True  # graceful fallback


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Layer 3: 전체 대화 흐름 — 분실 시나리오
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestFullDialogueLoss:
    """카드 분실(loss_type=분실) 전체 대화 흐름 검증"""

    @pytest.mark.asyncio
    async def test_loss_scenario_full_flow(self):
        engine, llm, knowledge = _build_engine()

        session = _make_session()

        # ── Turn 1: 시나리오 진입 ──
        r1 = await engine.process_utterance(session, "카드를 잃어버렸어요", None)
        assert session["dialogue_mode"] == "scenario"
        assert "카드 분실 신고" in r1.response_text
        assert r1.awaiting_input is True  # slot_collect 대기

        # ── Turn 2: 카드번호 입력 (regex 추출) ──
        r2 = await engine.process_utterance(session, "1234-5678-9012-3456 입니다", None)
        assert session["scenario_state"]["slots"]["card_number"] == "1234-5678-9012-3456"
        assert r2.awaiting_input is True  # loss_type slot 대기

        # ── Turn 3: 분실 유형 (LLM 추출) ──
        llm.generate_stream = AsyncMock(return_value=_make_llm_stream("분실")())
        r3 = await engine.process_utterance(session, "분실이에요", None)
        assert session["scenario_state"]["slots"]["loss_type"] == "분실"
        # Auto-advance: slot_collect → confirm → 질문 출력
        assert r3.awaiting_input is True
        assert "맞으시죠" in r3.response_text

        # ── Turn 4: 확인 → yes ──
        # Mock API 성공
        with patch.object(engine.action_runner, 'execute_api_call', new_callable=AsyncMock) as mock_api:
            mock_api.return_value = {"success": True, "card_blocked": True}

            r4 = await engine.process_utterance(session, "네 맞아요", None)
            # confirm(yes) → condition(분실=false) → api_call(성공) → end
            assert "완료" in r4.response_text
            assert r4.action is not None
            assert r4.action["type"] == "end"
            assert r4.action["disposition"] == "resolved"

        # ── 시나리오 종료 → freeform 복귀 ──
        assert session["dialogue_mode"] == "freeform"

        # ── 실행 로그 전송 확인 ──
        knowledge._client.post.assert_called()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Layer 3b: 전체 대화 흐름 — 도난 시나리오
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestFullDialogueTheft:
    """카드 도난(loss_type=도난) 전체 대화 흐름 검증 — 경찰 신고 안내 경로"""

    @pytest.mark.asyncio
    async def test_theft_scenario_includes_police_notice(self):
        engine, llm, _ = _build_engine()
        session = _make_session()

        # Turn 1: 시나리오 진입
        await engine.process_utterance(session, "카드 도난당했어요", None)

        # Turn 2: 카드번호
        await engine.process_utterance(session, "9876-5432-1098-7654", None)

        # Turn 3: 도난 유형 (LLM)
        llm.generate_stream = AsyncMock(return_value=_make_llm_stream("도난")())
        r3 = await engine.process_utterance(session, "도난이에요", None)
        assert session["scenario_state"]["slots"]["loss_type"] == "도난"

        # Turn 4: 확인 yes → condition(도난=true) → speak(경찰 신고) → api_call
        with patch.object(engine.action_runner, 'execute_api_call', new_callable=AsyncMock) as mock_api:
            mock_api.return_value = {"success": True}
            r4 = await engine.process_utterance(session, "네", None)
            assert "경찰 신고" in r4.response_text


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Layer 3c: 확인 거부 → 재수집 루프
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestConfirmRejectLoop:
    """confirm 'no' → slot_collect 재수집 루프 검증"""

    @pytest.mark.asyncio
    async def test_confirm_no_restarts_slot_collection(self):
        engine, llm, _ = _build_engine()
        session = _make_session()

        # 시나리오 진입 + 카드번호 + 분실유형
        await engine.process_utterance(session, "카드 분실이요", None)
        await engine.process_utterance(session, "1111-2222-3333-4444", None)
        llm.generate_stream = AsyncMock(return_value=_make_llm_stream("분실")())
        await engine.process_utterance(session, "분실", None)

        # confirm 거부
        r = await engine.process_utterance(session, "아니요 틀려요", None)
        # no → n2 (card_number slot_collect) 재수집
        assert session["scenario_state"]["current_node"] == "n2"
        assert r.awaiting_input is True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Layer 3d: 슬롯 추출 실패 → 재시도 → 전환
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestSlotRetryAndFail:
    """슬롯 추출 실패 시 재시도 + max_retries 초과 시 상담원 전환"""

    @pytest.mark.asyncio
    async def test_slot_retry_prompt(self):
        engine, _, _ = _build_engine()
        session = _make_session()

        # 시나리오 진입
        await engine.process_utterance(session, "카드 분실", None)

        # 잘못된 입력 → 재시도 프롬프트
        r = await engine.process_utterance(session, "모르겠어요", None)
        assert "다시" in r.response_text
        assert session["scenario_state"]["retry_count"] == 1
        assert r.awaiting_input is True

    @pytest.mark.asyncio
    async def test_max_retries_triggers_transfer(self):
        engine, _, _ = _build_engine()
        session = _make_session()

        # 시나리오 진입
        await engine.process_utterance(session, "카드 분실", None)

        # max_retries=2: 2번 실패 후 3번째 실패에서 fail_action 트리거
        await engine.process_utterance(session, "뭐라고요", None)  # retry 1
        await engine.process_utterance(session, "잘 모르겠어요", None)  # retry 2
        r = await engine.process_utterance(session, "어떻게 하죠", None)  # retry 3 → fail

        assert r.action is not None
        # fail_action이 "transfer"이거나 무한루프 감지로 "end" 중 하나
        assert r.action["type"] in ("transfer", "end")
        assert session["dialogue_mode"] == "freeform"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Layer 4: 에러 처리
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestErrorHandling:
    """LLM/API 장애 시 graceful degradation 검증"""

    @pytest.mark.asyncio
    async def test_llm_slot_extraction_failure_retries(self):
        """LLM 에러 시 슬롯 추출 실패 → 재시도 프롬프트"""
        engine, llm, _ = _build_engine()
        session = _make_session()

        await engine.process_utterance(session, "카드 분실", None)
        await engine.process_utterance(session, "1234-5678-9012-3456", None)

        # LLM 에러
        llm.generate_stream = AsyncMock(side_effect=RuntimeError("LLM down"))
        r = await engine.process_utterance(session, "분실이에요", None)
        # LLM 실패 → extract returns None → retry
        assert r.awaiting_input is True
        assert session["scenario_state"]["retry_count"] >= 1

    @pytest.mark.asyncio
    async def test_api_call_failure_routes_to_on_error(self):
        """API 호출 실패 시 on_error 노드로 라우팅"""
        engine, llm, _ = _build_engine()
        session = _make_session()

        # 빠르게 api_call 노드까지 진행
        await engine.process_utterance(session, "카드 분실", None)
        await engine.process_utterance(session, "1234-5678-9012-3456", None)
        llm.generate_stream = AsyncMock(return_value=_make_llm_stream("분실")())
        await engine.process_utterance(session, "분실", None)

        # API 실패 mock
        with patch.object(engine.action_runner, 'execute_api_call', new_callable=AsyncMock) as mock_api:
            mock_api.return_value = None  # API failed

            r = await engine.process_utterance(session, "네 맞아요", None)
            # on_error → n_err (speak: 시스템 오류) → n_err2 (transfer)
            assert r.action is not None
            assert r.action["type"] == "transfer"
            assert "오류" in r.response_text or "연결" in r.response_text

    @pytest.mark.asyncio
    async def test_api_call_success_continues_to_end(self):
        """API 호출 성공 시 end 노드까지 진행"""
        engine, llm, _ = _build_engine()
        session = _make_session()

        await engine.process_utterance(session, "카드 분실", None)
        await engine.process_utterance(session, "1234-5678-9012-3456", None)
        llm.generate_stream = AsyncMock(return_value=_make_llm_stream("분실")())
        await engine.process_utterance(session, "분실", None)

        with patch.object(engine.action_runner, 'execute_api_call', new_callable=AsyncMock) as mock_api:
            mock_api.return_value = {"success": True}

            r = await engine.process_utterance(session, "네 맞아요", None)
            assert r.action["type"] == "end"
            assert "완료" in r.response_text

    @pytest.mark.asyncio
    async def test_infinite_loop_protection(self):
        """그래프 순환 시 무한 루프 감지 → 강제 종결"""
        engine, _, _ = _build_engine()
        session = _make_session()
        session["dialogue_mode"] = "scenario"

        scenario = Scenario.from_dict(_load_fixture())
        session["scenario_state"].update({
            "scenario_id": "card-lost",
            "current_node": "n1",
            "history": ["n1", "n1", "n1"],  # 이미 3회 방문
            "_scenario_snapshot": scenario,
        })

        r = await engine.process_utterance(session, "", None)
        assert r.action is not None
        assert r.action.get("disposition") == "error"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Layer 5: 엔진 라이프사이클
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestEngineLifecycle:
    """freeform↔scenario 전환, 스택, 로그, 세션 격리"""

    @pytest.mark.asyncio
    async def test_scenario_end_returns_to_freeform(self):
        engine, _, _ = _build_engine()
        session = _make_session()
        session["dialogue_mode"] = "scenario"
        scenario = Scenario.from_dict(_load_fixture())
        session["scenario_state"].update({
            "scenario_id": "card-lost",
            "current_node": "n8",  # end node
            "_scenario_snapshot": scenario,
        })

        r = await engine.process_utterance(session, "", None)
        assert r.action["type"] == "end"
        assert session["dialogue_mode"] == "freeform"
        assert session["scenario_state"]["scenario_id"] is None

    @pytest.mark.asyncio
    async def test_transfer_returns_to_freeform(self):
        engine, _, _ = _build_engine()
        session = _make_session()
        session["dialogue_mode"] = "scenario"
        scenario = Scenario.from_dict(_load_fixture())
        session["scenario_state"].update({
            "scenario_id": "card-lost",
            "current_node": "n9",  # transfer node
            "_scenario_snapshot": scenario,
        })

        r = await engine.process_utterance(session, "", None)
        assert r.action["type"] == "transfer"
        assert session["dialogue_mode"] == "freeform"

    @pytest.mark.asyncio
    async def test_scenario_stack_resume(self):
        """중첩 시나리오 종료 후 부모 시나리오 복귀"""
        engine, _, _ = _build_engine()
        scenario = Scenario.from_dict(_load_fixture())

        parent_state = {
            "scenario_id": "card-lost",
            "scenario_version": 1,
            "current_node": "n3",
            "slots": {"card_number": "9999-8888-7777-6666"},
            "variables": {},
            "history": ["n1", "n2"],
            "retry_count": 0,
            "prosody": None,
            "stack": [],
            "awaiting_confirm": False,
            "_scenario_snapshot": scenario,
        }

        session = _make_session()
        session["dialogue_mode"] = "scenario"
        session["scenario_state"] = {
            "scenario_id": "card-lost",
            "scenario_version": 1,
            "current_node": "n8",  # end node
            "slots": {},
            "variables": {},
            "history": [],
            "retry_count": 0,
            "prosody": None,
            "stack": [parent_state],
            "awaiting_confirm": False,
            "_scenario_snapshot": scenario,
        }

        r = await engine.process_utterance(session, "", None)
        # End → pop stack → resume parent at n3
        assert session["dialogue_mode"] == "scenario"
        assert session["scenario_state"]["current_node"] == "n3"
        assert session["scenario_state"]["slots"]["card_number"] == "9999-8888-7777-6666"

    @pytest.mark.asyncio
    async def test_execution_log_sent_on_end(self):
        """시나리오 end 시 Knowledge에 실행 로그 전송 확인"""
        engine, _, knowledge = _build_engine()
        session = _make_session()
        session["dialogue_mode"] = "scenario"
        scenario = Scenario.from_dict(_load_fixture())
        session["scenario_state"].update({
            "scenario_id": "card-lost",
            "current_node": "n8",
            "slots": {"card_number": "1234"},
            "history": ["n1", "n2", "n8"],
            "_scenario_snapshot": scenario,
        })

        await engine.process_utterance(session, "", None)
        # Verify log was sent
        knowledge._client.post.assert_called_once()
        call_args = knowledge._client.post.call_args
        assert "/log-execution" in str(call_args)

    @pytest.mark.asyncio
    async def test_execution_log_failure_non_blocking(self):
        """로그 전송 실패해도 대화는 정상 진행"""
        engine, _, knowledge = _build_engine()
        knowledge._client.post = AsyncMock(side_effect=RuntimeError("Network error"))

        session = _make_session()
        session["dialogue_mode"] = "scenario"
        scenario = Scenario.from_dict(_load_fixture())
        session["scenario_state"].update({
            "scenario_id": "card-lost",
            "current_node": "n8",
            "_scenario_snapshot": scenario,
        })

        # Should not raise, should still return result
        r = await engine.process_utterance(session, "", None)
        assert r.action["type"] == "end"
        assert session["dialogue_mode"] == "freeform"

    @pytest.mark.asyncio
    async def test_multiple_sessions_independent(self):
        """두 세션이 독립적으로 시나리오 실행"""
        engine, llm, _ = _build_engine()

        session_a = _make_session()
        session_b = _make_session()

        # 세션 A: 시나리오 진입
        await engine.process_utterance(session_a, "카드 분실", None)
        assert session_a["dialogue_mode"] == "scenario"

        # 세션 B: freeform 유지
        r_b = await engine.process_utterance(session_b, "오늘 날씨가 좋네요", None)
        assert session_b["dialogue_mode"] == "freeform"
        assert r_b.should_use_s2s is True

        # 세션 A 상태가 B에 영향 안 줌
        assert session_a["dialogue_mode"] == "scenario"
        assert session_b["dialogue_mode"] == "freeform"

    @pytest.mark.asyncio
    async def test_prosody_stored_in_state(self):
        """prosody 정보가 scenario_state에 저장됨"""
        engine, _, _ = _build_engine()
        session = _make_session()
        prosody = {"energy": "high", "speech_rate": "fast", "energy_trend": "rising"}

        await engine.process_utterance(session, "카드 분실", prosody)
        assert session["scenario_state"]["prosody"] == prosody

    @pytest.mark.asyncio
    async def test_version_pinning_on_enter(self):
        """시나리오 진입 시 snapshot이 deepcopy됨"""
        engine, _, _ = _build_engine()
        session = _make_session()

        await engine.process_utterance(session, "카드 분실", None)
        snapshot = session["scenario_state"]["_scenario_snapshot"]
        original = engine.scenario_cache.get_scenario("card-lost")

        # snapshot과 original은 다른 객체여야 함 (deepcopy)
        assert snapshot is not original
        assert snapshot.id == original.id

    @pytest.mark.asyncio
    async def test_corrupted_snapshot_graceful_fallback(self):
        """scenario_snapshot 없으면 freeform 폴백"""
        engine, _, _ = _build_engine()
        session = _make_session()
        session["dialogue_mode"] = "scenario"
        session["scenario_state"]["_scenario_snapshot"] = None

        r = await engine.process_utterance(session, "test", None)
        assert r.should_use_s2s is True
        assert session["dialogue_mode"] == "freeform"
