"""Test intent matcher."""
import numpy as np
import pytest
from unittest.mock import AsyncMock

from realtime_demo.dialogue.models import Scenario, IntentMatch
from realtime_demo.dialogue.intent_matcher import IntentMatcher


def _make_scenarios():
    """Create two scenarios with pre-computed trigger embeddings."""
    base = np.random.randn(16).astype(np.float32)
    s1_triggers = ["카드를 잃어버렸어요", "카드 분실 신고", "카드가 없어졌어요"]
    s1_embeddings = [base + np.random.randn(16).astype(np.float32) * 0.1 for _ in s1_triggers]
    s2_triggers = ["한도 변경 요청", "카드 한도 올려주세요"]
    s2_embeddings = [np.random.randn(16).astype(np.float32) for _ in s2_triggers]
    return [
        {"scenario_id": "card-lost", "triggers": s1_triggers, "embeddings": s1_embeddings},
        {"scenario_id": "limit-change", "triggers": s2_triggers, "embeddings": s2_embeddings},
    ]


class TestCosineSimilarity:
    def test_identical_vectors(self):
        from realtime_demo.dialogue.intent_matcher import cosine_similarity
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=0.01)

    def test_orthogonal_vectors(self):
        from realtime_demo.dialogue.intent_matcher import cosine_similarity
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        assert cosine_similarity(v1, v2) == pytest.approx(0.0, abs=0.01)


class TestIntentMatcher:
    def test_match_top_k(self):
        matcher = IntentMatcher(embed_fn=None)
        scenarios = _make_scenarios()
        matcher.load_trigger_cache(scenarios)
        assert len(matcher._trigger_cache) == 5

    @pytest.mark.asyncio
    async def test_no_match_below_threshold(self):
        async def mock_embed(text):
            return np.random.randn(16).astype(np.float32) * 100
        matcher = IntentMatcher(embed_fn=mock_embed)
        matcher.load_trigger_cache(_make_scenarios())
        result = await matcher.match("오늘 날씨가 좋네요")
        assert result is None or result.confidence < 0.5
