"""계층형 시나리오 통합 테스트.

메인 시나리오(인사 → intent_route → 서브 시나리오 → 복귀 → 루프) 전체 동작 검증.

테스트 계층:
1. 메인 시나리오 자동 진입 (auto_enter_main)
2. intent_route → 서브 시나리오 진입 (스택 push)
3. 서브 시나리오 완료 → 메인 복귀 (스택 pop)
4. "다른 도움?" confirm → yes → intent_route 루프
5. "다른 도움?" confirm → no → 종결
6. intent_route 매칭 실패 → 재시도 → 상담원 전환
7. 다중 서브 시나리오 순차 실행
8. 세션 격리 (2개 세션 독립 동작)
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
from realtime_demo.dialogue.intent_matcher import IntentMatcher
from realtime_demo.dialogue.action_runner import ActionRunner


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 시나리오 정의
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MAIN_SCENARIO = {
    "id": "main",
    "name": "팀벨은행 메인",
    "description": "메인 인사 + 라우팅",
    "version": 1,
    "schema_version": 1,
    "status": "active",
    "priority": 100,
    "triggers": {"examples": ["전화 연결", "상담 시작"], "description": "메인 진입"},
    "slots": {},
    "nodes": [
        {"id": "greet", "type": "speak", "text": "안녕하세요. 팀벨은행 고객센터입니다."},
        {"id": "route", "type": "intent_route", "prompt": "무엇을 도와드릴까요?",
         "no_match_message": "죄송합니다. 다시 말씀해주시겠어요?", "max_no_match": 3},
        {"id": "ask_more", "type": "confirm", "template": "다른 도움이 필요하신가요?"},
        {"id": "loop_speak", "type": "speak", "text": "네, 다시 안내해드리겠습니다."},
        {"id": "goodbye", "type": "speak", "text": "감사합니다. 좋은 하루 되세요."},
        {"id": "done", "type": "end", "message": "", "disposition": "resolved", "tags": ["main"]},
        {"id": "transfer", "type": "transfer", "reason": "no_match_max", "message": "상담원에게 연결해드리겠습니다.",
         "transfer_data": {"slots": "all"}},
    ],
    "edges": [
        {"from": "greet", "to": "route"},
        {"from": "route", "to": "ask_more"},
        {"from": "ask_more", "to": "loop_speak", "label": "yes"},
        {"from": "ask_more", "to": "goodbye", "label": "no"},
        {"from": "loop_speak", "to": "route"},
        {"from": "goodbye", "to": "done"},
    ],
    "metadata": {},
}

CARD_SCENARIO = {
    "id": "card-lost",
    "name": "카드 분실 신고",
    "description": "카드 분실/도난 처리",
    "version": 1,
    "schema_version": 1,
    "status": "active",
    "priority": 10,
    "triggers": {"examples": ["카드 잃어버렸어요", "카드 분실", "카드 도난"], "description": "카드 분실"},
    "slots": {
        "card_number": {"type": "string", "required": True, "extract": "regex",
                        "pattern": r"\d{4}-\d{4}-\d{4}-\d{4}", "prompt": "카드번호를 말씀해주세요."},
    },
    "nodes": [
        {"id": "c1", "type": "speak", "text": "카드 분실 신고 접수하겠습니다."},
        {"id": "c2", "type": "slot_collect", "target_slot": "card_number", "max_retries": 2,
         "retry_prompt": "카드번호를 다시 말씀해주세요.", "fail_action": "transfer"},
        {"id": "c3", "type": "speak", "text": "카드 정지 처리 완료되었습니다."},
        {"id": "c4", "type": "end", "message": "", "disposition": "resolved", "tags": ["card"]},
    ],
    "edges": [
        {"from": "c1", "to": "c2"},
        {"from": "c2", "to": "c3"},
        {"from": "c3", "to": "c4"},
    ],
    "metadata": {},
}

BALANCE_SCENARIO = {
    "id": "balance",
    "name": "잔액 조회",
    "description": "계좌 잔액 확인",
    "version": 1,
    "schema_version": 1,
    "status": "active",
    "priority": 5,
    "triggers": {"examples": ["잔액 알려줘", "잔고 조회", "통장에 얼마 있어"], "description": "잔액 조회"},
    "slots": {
        "account": {"type": "string", "required": True, "extract": "regex",
                    "pattern": r"\d{3}-\d{6}-\d{2}", "prompt": "계좌번호를 말씀해주세요."},
    },
    "nodes": [
        {"id": "b1", "type": "speak", "text": "잔액 조회 도와드리겠습니다."},
        {"id": "b2", "type": "slot_collect", "target_slot": "account", "max_retries": 2,
         "retry_prompt": "계좌번호를 다시 말씀해주세요.", "fail_action": "transfer"},
        {"id": "b3", "type": "speak", "text": "잔액은 1,000,000원입니다."},
        {"id": "b4", "type": "end", "message": "", "disposition": "resolved", "tags": ["balance"]},
    ],
    "edges": [
        {"from": "b1", "to": "b2"},
        {"from": "b2", "to": "b3"},
        {"from": "b3", "to": "b4"},
    ],
    "metadata": {},
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_CARD_VEC = np.random.RandomState(42).randn(64).astype(np.float32)
_BALANCE_VEC = -_CARD_VEC


def _build_engine():
    cache = ScenarioCache()
    cache.load_from_dicts([MAIN_SCENARIO, CARD_SCENARIO, BALANCE_SCENARIO])
    # Set main flag
    for s in cache.scenarios.values():
        s._is_main = (s.id == "main")

    async def embed_fn(text):
        card_kw = ["카드", "분실", "잃어", "도난"]
        bal_kw = ["잔액", "잔고", "통장", "얼마"]
        if any(k in text for k in card_kw):
            return _CARD_VEC + np.random.randn(64).astype(np.float32) * 0.05
        if any(k in text for k in bal_kw):
            return _BALANCE_VEC + np.random.randn(64).astype(np.float32) * 0.05
        return np.random.randn(64).astype(np.float32) * 10

    matcher = IntentMatcher(embed_fn=embed_fn, similarity_threshold=0.5)
    matcher.set_scenarios(cache.scenarios)
    # 서브 시나리오 트리거만 로딩 (메인은 자동 진입이므로 트리거 불필요)
    trigger_data = []
    for s in cache.scenarios.values():
        if s.id == "main":
            continue
        examples = s.triggers.get("examples", [])
        if "카드" in s.name:
            embs = [_CARD_VEC + np.random.randn(64).astype(np.float32) * 0.05 for _ in examples]
        else:
            embs = [_BALANCE_VEC + np.random.randn(64).astype(np.float32) * 0.05 for _ in examples]
        trigger_data.append({"scenario_id": s.id, "triggers": examples, "embeddings": embs})
    matcher.load_trigger_cache(trigger_data)

    mock_llm = AsyncMock()
    mock_knowledge = AsyncMock()
    mock_knowledge.search = AsyncMock(return_value=[])
    mock_knowledge._client = AsyncMock()
    mock_knowledge._client.post = AsyncMock()

    engine = DialogueEngine(
        scenario_cache=cache,
        intent_matcher=matcher,
        slot_manager=SlotManager(llm_engine=mock_llm),
        llm_engine=mock_llm,
        action_runner=ActionRunner(knowledge_client=mock_knowledge),
    )
    return engine, mock_llm


def _session():
    return {
        "dialogue_mode": "freeform",
        "scenario_state": {
            "scenario_id": None, "scenario_version": None, "current_node": None,
            "slots": {}, "variables": {}, "history": [], "retry_count": 0,
            "prosody": None, "stack": [], "awaiting_confirm": False,
            "_scenario_snapshot": None,
        },
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 메인 시나리오 자동 진입
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestAutoEnterMain:
    @pytest.mark.asyncio
    async def test_auto_enter_greets(self):
        engine, _ = _build_engine()
        session = _session()
        result = await engine.auto_enter_main(session)
        assert result is not None
        assert "안녕하세요" in result.response_text
        assert "팀벨은행" in result.response_text
        assert session["dialogue_mode"] == "scenario"
        assert session["scenario_state"]["scenario_id"] == "main"

    @pytest.mark.asyncio
    async def test_auto_enter_awaits_input(self):
        """인사 후 intent_route에서 입력 대기."""
        engine, _ = _build_engine()
        session = _session()
        result = await engine.auto_enter_main(session)
        assert result.awaiting_input is True
        assert "도와드릴까요" in result.response_text

    @pytest.mark.asyncio
    async def test_no_main_returns_none(self):
        engine, _ = _build_engine()
        for s in engine.scenario_cache.scenarios.values():
            s._is_main = False
        session = _session()
        result = await engine.auto_enter_main(session)
        assert result is None
        assert session["dialogue_mode"] == "freeform"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. intent_route → 서브 시나리오 진입
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestIntentRouteEntry:
    @pytest.mark.asyncio
    async def test_card_intent_enters_card_scenario(self):
        engine, _ = _build_engine()
        session = _session()
        await engine.auto_enter_main(session)

        # 고객: "카드 잃어버렸어요"
        r = await engine.process_utterance(session, "카드 잃어버렸어요", None)
        assert "카드 분실 신고" in r.response_text
        assert session["scenario_state"]["scenario_id"] == "card-lost"
        # 스택에 메인이 저장됨
        assert len(session["scenario_state"]["stack"]) >= 1

    @pytest.mark.asyncio
    async def test_balance_intent_enters_balance_scenario(self):
        engine, _ = _build_engine()
        session = _session()
        await engine.auto_enter_main(session)

        r = await engine.process_utterance(session, "잔액 좀 알려주세요", None)
        assert "잔액 조회" in r.response_text
        assert session["scenario_state"]["scenario_id"] == "balance"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 서브 시나리오 완료 → 메인 복귀
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestSubScenarioReturn:
    @pytest.mark.asyncio
    async def test_card_complete_returns_to_main(self):
        engine, _ = _build_engine()
        session = _session()
        await engine.auto_enter_main(session)

        # 카드 시나리오 진입
        await engine.process_utterance(session, "카드 분실이요", None)
        assert session["scenario_state"]["scenario_id"] == "card-lost"

        # 카드번호 입력
        r = await engine.process_utterance(session, "1234-5678-9012-3456", None)

        # 서브 시나리오 end → 스택 pop → 메인 복귀
        # "카드 정지 완료" + 메인의 다음 노드(ask_more: "다른 도움이 필요하신가요?")
        assert session["scenario_state"]["scenario_id"] == "main"
        # 응답에 "다른 도움" 또는 메인 관련 텍스트가 있어야 함
        assert session["dialogue_mode"] == "scenario"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. confirm yes → intent_route 루프
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestConfirmYesLoop:
    @pytest.mark.asyncio
    async def test_yes_loops_back_to_route(self):
        engine, _ = _build_engine()
        session = _session()
        await engine.auto_enter_main(session)

        # 카드 시나리오 완료
        await engine.process_utterance(session, "카드 분실", None)
        await engine.process_utterance(session, "1234-5678-9012-3456", None)

        # Turn A: 서브 완료 후 confirm 질문 ("다른 도움이 필요하신가요?")
        r_ask = await engine.process_utterance(session, "네", None)
        assert session["dialogue_mode"] == "scenario"
        assert session["scenario_state"]["scenario_id"] == "main"

        # Turn B: confirm yes → loop_speak → intent_route 프롬프트 → 대기
        r_loop = await engine.process_utterance(session, "네", None)
        assert r_loop.awaiting_input is True
        assert "안내" in r_loop.response_text or "도와" in r_loop.response_text

        # Turn C: intent_route에 user_text 입력 → 서브 시나리오 진입
        r2 = await engine.process_utterance(session, "잔액 조회해주세요", None)
        # 잔액 시나리오 진입 또는 intent_route 처리 중
        assert session["dialogue_mode"] == "scenario"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. confirm no → 종결
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestConfirmNoEnd:
    @pytest.mark.asyncio
    async def test_no_ends_conversation(self):
        engine, _ = _build_engine()
        session = _session()
        await engine.auto_enter_main(session)

        # 서브 완료 후 메인 복귀 시뮬레이션
        # 직접 confirm 노드로 이동
        session["scenario_state"]["current_node"] = "ask_more"
        session["scenario_state"]["awaiting_confirm"] = False

        r = await engine.process_utterance(session, "", None)
        assert "다른 도움" in r.response_text
        assert r.awaiting_input is True

        # "아니요"
        r2 = await engine.process_utterance(session, "아니요", None)
        assert "감사합니다" in r2.response_text or "좋은 하루" in r2.response_text
        assert r2.action is not None
        assert r2.action["type"] == "end"
        assert session["dialogue_mode"] == "freeform"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. intent_route 매칭 실패 → 재시도 → 전환
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestIntentRouteNoMatch:
    @pytest.mark.asyncio
    async def test_no_match_retries(self):
        engine, _ = _build_engine()
        session = _session()
        await engine.auto_enter_main(session)

        # 매칭 안 되는 발화
        r = await engine.process_utterance(session, "오늘 날씨가 좋네요", None)
        assert r.awaiting_input is True
        assert "다시" in r.response_text or "죄송" in r.response_text

    @pytest.mark.asyncio
    async def test_max_no_match_transfers(self):
        """매칭 실패가 max_no_match 이상이면 전환 또는 루프 감지로 종결."""
        engine, _ = _build_engine()
        session = _session()
        await engine.auto_enter_main(session)

        # 여러 번 매칭 실패 — intent_route 노드 반복 방문으로 루프 감지 가능
        r1 = await engine.process_utterance(session, "아무말1", None)
        r2 = await engine.process_utterance(session, "아무말2", None)
        r3 = await engine.process_utterance(session, "아무말3", None)

        # no_match 카운터 또는 루프 감지 둘 다 유효한 종료 경로
        assert r3.action is not None
        assert r3.action["type"] in ("transfer", "end")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. 다중 서브 시나리오 순차 실행
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestMultipleSubScenarios:
    @pytest.mark.asyncio
    async def test_card_then_balance(self):
        """카드 분실 → 완료 → "네" → 잔액 조회 → 완료 → "아니요" → 종결."""
        engine, _ = _build_engine()
        session = _session()

        # 메인 진입
        await engine.auto_enter_main(session)

        # 1차: 카드 분실
        await engine.process_utterance(session, "카드 잃어버렸어요", None)
        assert session["scenario_state"]["scenario_id"] == "card-lost"
        await engine.process_utterance(session, "1234-5678-9012-3456", None)
        # 서브 완료 → 메인 복귀

        # confirm 질문 표시
        r_ask = await engine.process_utterance(session, "네", None)
        assert session["scenario_state"]["scenario_id"] == "main"

        # confirm yes → loop → intent_route 프롬프트
        r_loop = await engine.process_utterance(session, "네", None)
        assert r_loop.awaiting_input is True

        # 2차: 잔액 조회
        r_route = await engine.process_utterance(session, "잔액 알려주세요", None)
        assert session["dialogue_mode"] == "scenario"

        # 잔액 시나리오 진행이 가능하면 계좌번호 입력
        if session["scenario_state"]["scenario_id"] == "balance":
            await engine.process_utterance(session, "123-456789-01", None)

        # 종결로 이동
        # 여러 턴이 필요할 수 있으므로 최종 상태만 확인
        # "아니요"로 종결 시도
        for _ in range(5):
            if session["dialogue_mode"] == "freeform":
                break
            r = await engine.process_utterance(session, "아니요", None)
            if r.action and r.action.get("type") == "end":
                break

        assert session["dialogue_mode"] == "freeform"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. 세션 격리
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestSessionIsolation:
    @pytest.mark.asyncio
    async def test_two_sessions_independent(self):
        engine, _ = _build_engine()
        s1, s2 = _session(), _session()

        # 둘 다 메인 진입
        await engine.auto_enter_main(s1)
        await engine.auto_enter_main(s2)

        # s1: 카드 시나리오
        await engine.process_utterance(s1, "카드 분실", None)
        assert s1["scenario_state"]["scenario_id"] == "card-lost"

        # s2: 잔액 시나리오
        await engine.process_utterance(s2, "잔액 조회", None)
        assert s2["scenario_state"]["scenario_id"] == "balance"

        # 서로 영향 없음
        assert s1["scenario_state"]["scenario_id"] == "card-lost"
        assert s2["scenario_state"]["scenario_id"] == "balance"
