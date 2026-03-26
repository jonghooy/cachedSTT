"""Intent matcher — cosine similarity on trigger embeddings + optional LLM judgment."""
from __future__ import annotations

import logging
from typing import Callable, Awaitable

import numpy as np

from realtime_demo.dialogue.models import IntentMatch, Scenario

logger = logging.getLogger(__name__)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class IntentMatcher:
    def __init__(self, embed_fn=None, llm_engine=None, similarity_threshold: float = 0.5, confidence_threshold: float = 0.7):
        self.embed_fn = embed_fn
        self.llm_engine = llm_engine
        self.similarity_threshold = similarity_threshold
        self.confidence_threshold = confidence_threshold
        self._trigger_cache: list[dict] = []
        self._scenarios: dict[str, Scenario] = {}

    def load_trigger_cache(self, trigger_data: list[dict]):
        self._trigger_cache = []
        for entry in trigger_data:
            sid = entry["scenario_id"]
            for text, emb in zip(entry["triggers"], entry["embeddings"]):
                self._trigger_cache.append({"scenario_id": sid, "text": text, "embedding": np.asarray(emb, dtype=np.float32)})

    def set_scenarios(self, scenarios: dict[str, Scenario]):
        self._scenarios = scenarios

    async def match(self, text: str, top_k: int = 3) -> IntentMatch | None:
        if not self._trigger_cache or not self.embed_fn:
            return None
        try:
            user_embedding = await self.embed_fn(text)
        except Exception:
            logger.exception("Embedding failed")
            return None
        scores = []
        for entry in self._trigger_cache:
            sim = cosine_similarity(user_embedding, entry["embedding"])
            scores.append((sim, entry))
        scores.sort(key=lambda x: x[0], reverse=True)
        top = scores[:top_k]
        if not top or top[0][0] < self.similarity_threshold:
            return None
        best_sim, best_entry = top[0]
        scenario_id = best_entry["scenario_id"]
        scenario = self._scenarios.get(scenario_id)
        if scenario is None:
            return None
        return IntentMatch(scenario_id=scenario_id, scenario=scenario, confidence=best_sim, matched_example=best_entry["text"])
