"""전체 시스템 통합 테스트.

Brain ↔ Knowledge Service 간 데이터 흐름을 검증합니다.
외부 서비스 없이 모든 것을 mock하여 실행합니다.

테스트 계층:
1. Knowledge → Brain 시나리오 로딩 (API 응답 포맷 → ScenarioCache → IntentMatcher)
2. 전체 라이프사이클 (생성 → 검증 → 배포 → Brain 로딩 → 대화 → 종료 → 로그)
3. 다중 시나리오 동시 운영 (매칭 정확도, 시나리오 전환)
4. AI 생성 시나리오 실행 (Claude가 만든 JSON → Brain에서 실행)
5. 장애 복원력 (Knowledge 다운, 잘못된 시나리오, 캐시 갱신 중 대화)
6. 보안 검증 (규칙 평가기 접근 제한, 템플릿 경로 검증, URL 검증)
"""
import json
import copy
import asyncio
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from realtime_demo.dialogue.models import Scenario, DialogueResult, IntentMatch
from realtime_demo.dialogue.engine import DialogueEngine
from realtime_demo.dialogue.graph_walker import GraphWalker
from realtime_demo.dialogue.slot_manager import SlotManager
from realtime_demo.dialogue.scenario_cache import ScenarioCache
from realtime_demo.dialogue.intent_matcher import IntentMatcher
from realtime_demo.dialogue.action_runner import ActionRunner
from realtime_demo.dialogue.rule_evaluator import evaluate_rule
from realtime_demo.dialogue.template_renderer import render_template

FIXTURE_DIR = Path(__file__).parent / "fixtures"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _knowledge_api_response():
    """Knowledge Service GET /api/brain/scenarios 응답 형식 시뮬레이션."""
    with open(FIXTURE_DIR / "card_lost_scenario.json") as f:
        fixture = json.load(f)

    return {
        "scenarios": [{
            "id": 1,
            "name": fixture["name"],
            "description": fixture["description"],
            "schema_version": 1,
            "graph_json": json.dumps({"nodes": fixture["nodes"], "edges": fixture["edges"]}),
            "version": 1,
            "status": "active",
            "priority": 10,
            "triggers_json": json.dumps(fixture["triggers"]),
            "slots_json": json.dumps(fixture["slots"]),
            "metadata_json": json.dumps(fixture.get("metadata", {})),
            "source_json": json.dumps({"type": "ai"}),
            "created_by": "ai",
            "trigger_embeddings": [],  # Brain에서 embed_fn으로 처리
        }],
        "version": "2026-03-27T12:00:00Z",
    }


def _second_scenario_api():
    """두 번째 시나리오 (잔액 조회)."""
    return {
        "id": 2,
        "name": "잔액 조회",
        "description": "계좌 잔액 확인",
        "schema_version": 1,
        "graph_json": json.dumps({
            "nodes": [
                {"id": "b1", "type": "speak", "text": "잔액 조회 도와드리겠습니다. 계좌번호를 말씀해주세요."},
                {"id": "b2", "type": "slot_collect", "target_slot": "account", "max_retries": 2,
                 "retry_prompt": "계좌번호를 다시 말씀해주세요.", "fail_action": "transfer"},
                {"id": "b3", "type": "speak", "text": "잔액은 100만원입니다."},
                {"id": "b4", "type": "end", "message": "감사합니다.", "disposition": "resolved", "tags": ["balance"]},
            ],
            "edges": [
                {"from": "b1", "to": "b2"},
                {"from": "b2", "to": "b3"},
                {"from": "b3", "to": "b4"},
            ],
        }),
        "version": 1,
        "status": "active",
        "priority": 5,
        "triggers_json": json.dumps({
            "examples": ["잔액 알려줘", "잔고 조회", "통장에 얼마 있어"],
            "description": "잔액 조회 인텐트",
        }),
        "slots_json": json.dumps({
            "account": {"type": "string", "required": True, "extract": "regex",
                        "pattern": r"\d{3}-\d{6}-\d{2}", "prompt": "계좌번호를 말씀해주세요."},
        }),
        "metadata_json": "{}",
        "source_json": "{}",
        "created_by": "ai",
        "trigger_embeddings": [],
    }


_BASE_VEC = np.random.RandomState(42).randn(64).astype(np.float32)


def _build_multi_scenario_engine():
    """두 개 시나리오를 로딩한 엔진 구축."""
    api_resp = _knowledge_api_response()
    api_resp["scenarios"].append(_second_scenario_api())

    cache = ScenarioCache()
    cache.load_from_dicts(api_resp["scenarios"])

    # 임베딩 함수: 카드 관련 → 카드 시나리오 벡터, 잔액 관련 → 잔액 벡터, 기타 → 랜덤
    card_vec = _BASE_VEC.copy()
    balance_vec = -_BASE_VEC  # 반대 방향

    async def embed_fn(text):
        card_kw = ["카드", "분실", "잃어", "도난"]
        balance_kw = ["잔액", "잔고", "통장", "얼마"]
        if any(k in text for k in card_kw):
            return card_vec + np.random.randn(64).astype(np.float32) * 0.05
        if any(k in text for k in balance_kw):
            return balance_vec + np.random.randn(64).astype(np.float32) * 0.05
        return np.random.randn(64).astype(np.float32) * 10

    matcher = IntentMatcher(embed_fn=embed_fn, similarity_threshold=0.5)
    matcher.set_scenarios(cache.scenarios)

    # 트리거 임베딩 로딩
    trigger_data = []
    for s in cache.scenarios.values():
        examples = s.triggers.get("examples", [])
        if "카드" in s.name:
            embs = [card_vec + np.random.randn(64).astype(np.float32) * 0.05 for _ in examples]
        else:
            embs = [balance_vec + np.random.randn(64).astype(np.float32) * 0.05 for _ in examples]
        trigger_data.append({"scenario_id": s.id, "triggers": examples, "embeddings": embs})
    matcher.load_trigger_cache(trigger_data)

    mock_llm = AsyncMock()
    sm = SlotManager(llm_engine=mock_llm)

    mock_knowledge = AsyncMock()
    mock_knowledge.search = AsyncMock(return_value=[])
    mock_knowledge._client = AsyncMock()
    mock_knowledge._client.post = AsyncMock()

    engine = DialogueEngine(
        scenario_cache=cache,
        intent_matcher=matcher,
        slot_manager=sm,
        llm_engine=mock_llm,
        action_runner=ActionRunner(knowledge_client=mock_knowledge),
    )
    return engine, mock_llm, mock_knowledge


def _make_session():
    return {
        "dialogue_mode": "freeform",
        "scenario_state": {
            "scenario_id": None, "scenario_version": None, "current_node": None,
            "slots": {}, "variables": {}, "history": [], "retry_count": 0,
            "prosody": None, "stack": [], "awaiting_confirm": False,
            "_scenario_snapshot": None,
        },
    }


def _llm_stream(text):
    async def stream(*a, **kw):
        yield text
    return stream


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Layer 1: Knowledge → Brain 시나리오 로딩
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestKnowledgeToBrainLoading:
    """Knowledge API 응답 → ScenarioCache → IntentMatcher 전체 파이프라인."""

    def test_knowledge_json_string_fields_parsed(self):
        """graph_json/triggers_json이 문자열이어도 정상 파싱."""
        api = _knowledge_api_response()
        cache = ScenarioCache()
        cache.load_from_dicts(api["scenarios"])
        assert len(cache.scenarios) == 1
        s = list(cache.scenarios.values())[0]
        assert len(s.nodes) == 11
        assert len(s.triggers.get("examples", [])) == 4

    def test_knowledge_dict_fields_parsed(self):
        """graph_json/triggers_json이 이미 dict여도 정상 파싱."""
        api = _knowledge_api_response()
        sc = api["scenarios"][0]
        sc["graph_json"] = json.loads(sc["graph_json"])
        sc["triggers_json"] = json.loads(sc["triggers_json"])
        sc["slots_json"] = json.loads(sc["slots_json"])
        cache = ScenarioCache()
        cache.load_from_dicts(api["scenarios"])
        assert len(cache.scenarios) == 1

    def test_inactive_scenario_filtered(self):
        api = _knowledge_api_response()
        api["scenarios"][0]["status"] = "draft"
        cache = ScenarioCache()
        cache.load_from_dicts(api["scenarios"])
        assert len(cache.scenarios) == 0

    def test_malformed_scenario_skipped(self):
        """잘못된 시나리오 데이터 → 건너뛰고 나머지 로딩."""
        api = _knowledge_api_response()
        api["scenarios"].append({"id": 999, "bad": "data"})  # name 없음
        cache = ScenarioCache()
        cache.load_from_dicts(api["scenarios"])
        assert len(cache.scenarios) == 1  # 정상 시나리오만 로딩

    def test_trigger_embeddings_loaded(self):
        engine, _, _ = _build_multi_scenario_engine()
        assert len(engine.intent_matcher._trigger_cache) == 7  # 4 (카드) + 3 (잔액)

    def test_scenario_id_is_string(self):
        api = _knowledge_api_response()
        cache = ScenarioCache()
        cache.load_from_dicts(api["scenarios"])
        for sid in cache.scenarios:
            assert isinstance(sid, str), f"scenario_id should be str, got {type(sid)}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Layer 2: 전체 라이프사이클
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestFullLifecycle:
    """생성 → 배포 → Brain 로딩 → 대화 실행 → 종료 → 로그 전송."""

    @pytest.mark.asyncio
    async def test_complete_lifecycle(self):
        engine, llm, knowledge = _build_multi_scenario_engine()
        session = _make_session()

        # 1. 시나리오 진입 (카드 분실)
        r1 = await engine.process_utterance(session, "카드를 잃어버렸어요", None)
        assert session["dialogue_mode"] == "scenario"
        assert "카드 분실 신고" in r1.response_text

        # 2. 슬롯 수집 — 카드번호 (regex)
        r2 = await engine.process_utterance(session, "1234-5678-9012-3456", None)
        assert session["scenario_state"]["slots"]["card_number"] == "1234-5678-9012-3456"

        # 3. 슬롯 수집 — 분실유형 (LLM)
        llm.generate_stream = AsyncMock(return_value=_llm_stream("분실")())
        r3 = await engine.process_utterance(session, "분실이에요", None)
        assert session["scenario_state"]["slots"]["loss_type"] == "분실"

        # 4. 확인 → yes
        with patch.object(engine.action_runner, 'execute_api_call', new_callable=AsyncMock) as mock_api:
            mock_api.return_value = {"success": True}
            r_confirm = await engine.process_utterance(session, "네 맞아요", None)

        # 5. 시나리오 종료 → freeform 복귀
        assert session["dialogue_mode"] == "freeform"

        # 6. 실행 로그 전송 확인
        knowledge._client.post.assert_called()

    @pytest.mark.asyncio
    async def test_freeform_after_scenario_end(self):
        """시나리오 종료 후 다음 발화는 freeform으로 처리."""
        engine, llm, _ = _build_multi_scenario_engine()
        session = _make_session()

        # 시나리오 완료 시뮬레이션
        session["dialogue_mode"] = "scenario"
        scenario = list(engine.scenario_cache.scenarios.values())[0]
        session["scenario_state"].update({
            "scenario_id": scenario.id,
            "current_node": list(scenario.nodes.keys())[-4],  # end node
            "_scenario_snapshot": scenario,
        })
        # Find end node
        for nid, node in scenario.nodes.items():
            if node.type == "end":
                session["scenario_state"]["current_node"] = nid
                break
        await engine.process_utterance(session, "", None)
        assert session["dialogue_mode"] == "freeform"

        # 다음 발화 → freeform
        r = await engine.process_utterance(session, "감사합니다", None)
        assert r.should_use_s2s is True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Layer 3: 다중 시나리오 매칭 정확도
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestMultiScenarioMatching:
    """여러 시나리오 동시 운영 시 정확한 매칭 검증."""

    @pytest.mark.asyncio
    async def test_card_text_matches_card_scenario(self):
        engine, _, _ = _build_multi_scenario_engine()
        session = _make_session()
        r = await engine.process_utterance(session, "카드 분실 신고해주세요", None)
        assert session["dialogue_mode"] == "scenario"
        assert "카드 분실" in r.response_text

    @pytest.mark.asyncio
    async def test_balance_text_matches_balance_scenario(self):
        engine, _, _ = _build_multi_scenario_engine()
        session = _make_session()
        r = await engine.process_utterance(session, "잔액이 얼마인지 알려주세요", None)
        assert session["dialogue_mode"] == "scenario"
        assert "잔액 조회" in r.response_text

    @pytest.mark.asyncio
    async def test_unrelated_text_stays_freeform(self):
        engine, _, _ = _build_multi_scenario_engine()
        session = _make_session()
        r = await engine.process_utterance(session, "오늘 점심 뭐 먹지", None)
        assert session["dialogue_mode"] == "freeform"
        assert r.should_use_s2s is True

    @pytest.mark.asyncio
    async def test_two_sessions_different_scenarios(self):
        """두 세션이 각각 다른 시나리오에 진입."""
        engine, _, _ = _build_multi_scenario_engine()
        s1 = _make_session()
        s2 = _make_session()

        await engine.process_utterance(s1, "카드 도난당했어요", None)
        await engine.process_utterance(s2, "잔고 확인해주세요", None)

        assert s1["dialogue_mode"] == "scenario"
        assert s2["dialogue_mode"] == "scenario"
        assert s1["scenario_state"]["scenario_id"] != s2["scenario_state"]["scenario_id"]

    @pytest.mark.asyncio
    async def test_sequential_different_scenarios(self):
        """한 세션이 시나리오 A 완료 후 시나리오 B 진입."""
        engine, _, _ = _build_multi_scenario_engine()
        session = _make_session()

        # 카드 시나리오 진입 → end 노드로 바로 이동
        await engine.process_utterance(session, "카드 분실", None)
        assert session["dialogue_mode"] == "scenario"
        card_id = session["scenario_state"]["scenario_id"]

        # 강제로 end 노드 실행
        scenario = session["scenario_state"]["_scenario_snapshot"]
        for nid, node in scenario.nodes.items():
            if node.type == "end":
                session["scenario_state"]["current_node"] = nid
                break
        await engine.process_utterance(session, "", None)
        assert session["dialogue_mode"] == "freeform"

        # 잔액 시나리오 진입
        await engine.process_utterance(session, "잔액 조회해주세요", None)
        assert session["dialogue_mode"] == "scenario"
        assert session["scenario_state"]["scenario_id"] != card_id


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Layer 4: AI 생성 시나리오 실행
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestAiGeneratedScenarioExecution:
    """Claude가 생성한 시나리오 JSON이 Brain에서 실행 가능한지 검증."""

    @pytest.mark.asyncio
    async def test_ai_generated_scenario_runs(self):
        """AI가 만든 전형적인 시나리오 구조가 정상 실행."""
        ai_scenario = {
            "id": "ai-gen-1",
            "name": "비밀번호 변경",
            "description": "AI가 생성한 비밀번호 변경 시나리오",
            "version": 1,
            "schema_version": 1,
            "status": "active",
            "priority": 5,
            "triggers": {
                "examples": ["비밀번호 바꾸고 싶어요"],
                "description": "비밀번호 변경 인텐트",
            },
            "slots": {
                "current_pw": {"type": "string", "required": True, "extract": "llm", "prompt": "현재 비밀번호를 입력해주세요."},
                "new_pw": {"type": "string", "required": True, "extract": "llm", "prompt": "새 비밀번호를 입력해주세요."},
            },
            "nodes": [
                {"id": "s1", "type": "speak", "text": "비밀번호 변경 도와드리겠습니다."},
                {"id": "s2", "type": "slot_collect", "target_slot": "current_pw", "max_retries": 2, "retry_prompt": "다시 입력해주세요.", "fail_action": "transfer"},
                {"id": "s3", "type": "slot_collect", "target_slot": "new_pw", "max_retries": 2, "retry_prompt": "다시 입력해주세요.", "fail_action": "transfer"},
                {"id": "s4", "type": "confirm", "template": "비밀번호를 변경하시겠습니까?"},
                {"id": "s5", "type": "end", "message": "비밀번호가 변경되었습니다.", "disposition": "resolved"},
                {"id": "s6", "type": "transfer", "reason": "pw_fail", "message": "상담원 연결합니다.", "transfer_data": {"slots": "all"}},
            ],
            "edges": [
                {"from": "s1", "to": "s2"},
                {"from": "s2", "to": "s3"},
                {"from": "s3", "to": "s4"},
                {"from": "s4", "to": "s5", "label": "yes"},
                {"from": "s4", "to": "s2", "label": "no"},
            ],
            "metadata": {},
        }

        cache = ScenarioCache()
        cache.load_from_dicts([ai_scenario])
        scenario = cache.get_scenario("ai-gen-1")
        assert scenario is not None
        assert len(scenario.nodes) == 6

        # 직접 실행
        mock_llm = AsyncMock()
        sm = SlotManager(llm_engine=mock_llm)
        matcher = AsyncMock()
        matcher.match = AsyncMock(return_value=IntentMatch(
            scenario_id="ai-gen-1", scenario=scenario, confidence=0.95))

        engine = DialogueEngine(
            scenario_cache=cache, intent_matcher=matcher,
            slot_manager=sm, llm_engine=mock_llm,
            action_runner=ActionRunner(),
        )
        session = _make_session()

        # Turn 1: 진입
        r1 = await engine.process_utterance(session, "비밀번호 변경", None)
        assert "비밀번호 변경" in r1.response_text
        assert r1.awaiting_input is True

        # Turn 2: 현재 비밀번호
        mock_llm.generate_stream = AsyncMock(return_value=_llm_stream("old123")())
        r2 = await engine.process_utterance(session, "old123", None)
        assert session["scenario_state"]["slots"]["current_pw"] == "old123"

        # Turn 3: 새 비밀번호
        mock_llm.generate_stream = AsyncMock(return_value=_llm_stream("new456")())
        r3 = await engine.process_utterance(session, "new456", None)
        assert session["scenario_state"]["slots"]["new_pw"] == "new456"

        # Turn 4: 확인 yes → end
        r4 = await engine.process_utterance(session, "네", None)
        assert r4.action["type"] == "end"
        assert session["dialogue_mode"] == "freeform"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Layer 5: 장애 복원력
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestResilience:
    """시스템 장애 시 graceful degradation 검증."""

    @pytest.mark.asyncio
    async def test_knowledge_down_stays_freeform(self):
        """Knowledge 서비스 다운 → 0개 시나리오 → freeform."""
        cache = ScenarioCache()
        # refresh 실패 시뮬레이션 (Knowledge 없음)
        assert cache.get_active_scenarios() == []

        matcher = IntentMatcher(embed_fn=None)
        engine = DialogueEngine(
            scenario_cache=cache, intent_matcher=matcher,
            slot_manager=SlotManager(), action_runner=ActionRunner(),
        )
        session = _make_session()
        r = await engine.process_utterance(session, "카드 분실", None)
        assert r.should_use_s2s is True

    @pytest.mark.asyncio
    async def test_embed_fn_crash_stays_freeform(self):
        """임베딩 함수 에러 → freeform."""
        async def broken_embed(t):
            raise RuntimeError("GPU OOM")

        engine, _, _ = _build_multi_scenario_engine()
        engine.intent_matcher.embed_fn = broken_embed
        session = _make_session()
        r = await engine.process_utterance(session, "카드 분실", None)
        assert r.should_use_s2s is True

    @pytest.mark.asyncio
    async def test_llm_crash_during_slot_retries(self):
        """LLM 다운 중 슬롯 추출 실패 → 재시도 프롬프트."""
        engine, llm, _ = _build_multi_scenario_engine()
        session = _make_session()

        await engine.process_utterance(session, "카드 분실", None)
        await engine.process_utterance(session, "1234-5678-9012-3456", None)

        llm.generate_stream = AsyncMock(side_effect=RuntimeError("LLM down"))
        r = await engine.process_utterance(session, "분실이에요", None)
        assert r.awaiting_input is True  # 재시도

    @pytest.mark.asyncio
    async def test_api_failure_routes_to_error_node(self):
        """API 호출 실패 → on_error 노드."""
        engine, llm, _ = _build_multi_scenario_engine()
        session = _make_session()

        await engine.process_utterance(session, "카드 분실", None)
        await engine.process_utterance(session, "1234-5678-9012-3456", None)
        llm.generate_stream = AsyncMock(return_value=_llm_stream("분실")())
        await engine.process_utterance(session, "분실", None)

        with patch.object(engine.action_runner, 'execute_api_call', new_callable=AsyncMock) as mock_api:
            mock_api.return_value = None  # API 실패
            r = await engine.process_utterance(session, "네 맞아요", None)
            assert r.action["type"] == "transfer"  # on_error → transfer

    @pytest.mark.asyncio
    async def test_corrupted_scenario_snapshot(self):
        """시나리오 스냅샷 손상 → freeform 복귀."""
        engine, _, _ = _build_multi_scenario_engine()
        session = _make_session()
        session["dialogue_mode"] = "scenario"
        session["scenario_state"]["_scenario_snapshot"] = None
        r = await engine.process_utterance(session, "test", None)
        assert r.should_use_s2s is True
        assert session["dialogue_mode"] == "freeform"

    @pytest.mark.asyncio
    async def test_scenario_cache_refresh_doesnt_affect_active_session(self):
        """캐시 갱신이 진행 중인 세션에 영향 없음 (deepcopy 격리)."""
        engine, _, _ = _build_multi_scenario_engine()
        session = _make_session()

        # 시나리오 진입
        await engine.process_utterance(session, "카드 분실", None)
        snapshot_before = session["scenario_state"]["_scenario_snapshot"]

        # 캐시 갱신 (새 데이터로)
        engine.scenario_cache.load_from_dicts([])  # 빈 캐시로 갱신
        assert len(engine.scenario_cache.scenarios) == 0

        # 진행 중인 세션의 스냅샷은 영향 없음
        assert snapshot_before is session["scenario_state"]["_scenario_snapshot"]
        assert len(snapshot_before.nodes) > 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Layer 6: 보안 검증
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestSecurity:
    """규칙 평가기, 템플릿 렌더러, 액션 러너의 보안 검증."""

    def test_rule_evaluator_blocks_private_fields(self):
        """slots/variables 외 필드 접근 차단."""
        rule = {"field": "_scenario_snapshot.id", "op": "eq", "value": "card-lost"}
        ctx = {"slots": {}, "variables": {}, "_scenario_snapshot": {"id": "card-lost"}}
        assert evaluate_rule(rule, ctx) is False

    def test_rule_evaluator_allows_slots(self):
        rule = {"field": "slots.card_number", "op": "eq", "value": "1234"}
        ctx = {"slots": {"card_number": "1234"}, "variables": {}}
        assert evaluate_rule(rule, ctx) is True

    def test_rule_evaluator_allows_variables(self):
        rule = {"field": "variables.result", "op": "exists"}
        ctx = {"slots": {}, "variables": {"result": True}}
        assert evaluate_rule(rule, ctx) is True

    def test_template_blocks_special_chars_in_path(self):
        """경로에 특수문자 포함 시 regex가 매치 안 함 → 그대로 출력."""
        # 하이픈/괄호 등은 \w에 매치 안 되어 템플릿으로 인식 안 됨
        result = render_template("{{slots.card-number}}", {"slots": {"card-number": "hack"}})
        assert result == "{{slots.card-number}}"  # regex 미매치 → 원본 유지
        # 알파벳+숫자+언더스코어만 유효한 경로로 처리됨 (안전)

    def test_template_normal_underscore_allowed(self):
        result = render_template("{{slots.card_number}}", {"slots": {"card_number": "1234"}})
        assert result == "1234"

    @pytest.mark.asyncio
    async def test_action_runner_rejects_invalid_url(self):
        runner = ActionRunner()
        state = {"variables": {}}
        action = {"method": "GET", "url": "file:///etc/passwd", "result_var": "x"}
        result = await runner.execute_api_call(action, state)
        assert result is None

    @pytest.mark.asyncio
    async def test_action_runner_rejects_invalid_method(self):
        runner = ActionRunner()
        state = {"variables": {}}
        action = {"method": "HACK", "url": "http://example.com", "result_var": "x"}
        result = await runner.execute_api_call(action, state)
        assert result is None

    def test_rule_evaluator_rejects_dangerous_operator(self):
        rule = {"field": "slots.x", "op": "eval", "value": "os.system('rm -rf /')"}
        with pytest.raises(ValueError):
            evaluate_rule(rule, {"slots": {"x": "1"}, "variables": {}})
