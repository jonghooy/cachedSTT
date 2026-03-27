"""Knowledge Service client for Brain.
Loads system prompt + FAQ from Knowledge API at startup.
Caches locally for zero-latency LLM prompt assembly.
"""

import asyncio
import logging
import httpx
from typing import Optional

logger = logging.getLogger(__name__)

KNOWLEDGE_BASE_URL = "http://localhost:8100/api/brain"


class KnowledgeClient:
    """Connects Brain to Knowledge Service."""

    def __init__(self, base_url: str = KNOWLEDGE_BASE_URL, brain_id: str = "default"):
        self.base_url = base_url
        self.brain_id = brain_id
        self._system_prompt: str = ""
        self._faq: list[dict] = []
        self._loaded = False
        self._client = httpx.AsyncClient(base_url=base_url, timeout=10.0)

    async def load_config(self) -> bool:
        """Load system prompt + FAQ from Knowledge Service. Returns True on success."""
        try:
            resp = await self._client.get("/config", params={"brain_id": self.brain_id})
            if resp.status_code == 200:
                data = resp.json()
                self._system_prompt = data.get("system_prompt", "")
                self._faq = data.get("faq", [])
                self._loaded = True
                logger.info(
                    f"[Knowledge] Loaded: prompt={len(self._system_prompt)} chars, "
                    f"faq={len(self._faq)} items"
                )
                return True
        except Exception as e:
            logger.warning(f"[Knowledge] Failed to load config: {e}")
        return False

    async def search(self, query: str, top_k: int = 3) -> list[dict]:
        """RAG search against Knowledge Service."""
        try:
            resp = await self._client.post("/search", json={"query": query, "top_k": top_k})
            if resp.status_code == 200:
                return resp.json().get("results", [])
        except Exception as e:
            logger.warning(f"[Knowledge] Search failed: {e}")
        return []

    def get_system_prompt(self) -> str:
        """Return cached system prompt. Empty string if not loaded."""
        return self._system_prompt

    def get_faq_context(self, query: str, max_items: int = 3) -> str:
        """Simple keyword matching against cached FAQ.
        Returns formatted context string for LLM prompt.
        """
        if not self._faq:
            return ""

        # Simple relevance: count overlapping characters
        query_lower = query.lower()
        scored = []
        for faq in self._faq:
            q = faq.get("q", "").lower()
            # Count common chars as simple relevance
            common = sum(1 for c in query_lower if c in q)
            if common > 0:
                scored.append((common, faq))

        scored.sort(reverse=True, key=lambda x: x[0])
        top = scored[:max_items]

        if not top:
            return ""

        lines = ["[참고 FAQ]"]
        for _, faq in top:
            lines.append(f"Q: {faq['q']}")
            lines.append(f"A: {faq['a']}")
            lines.append("")
        return "\n".join(lines)

    async def embed(self, text: str) -> list[float]:
        """Get BGE-M3 embedding vector from Knowledge Service."""
        try:
            resp = await self._client.post("/embed", json={"text": text})
            if resp.status_code == 200:
                data = resp.json()
                return data.get("embedding", [])
        except Exception as e:
            logger.warning(f"[Knowledge] Embed failed: {e}")
        return []

    def is_loaded(self) -> bool:
        return self._loaded
