"""Action runner — executes API calls, transfers, and end actions."""
from __future__ import annotations

import inspect
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class ActionRunner:
    def __init__(self, knowledge_client=None):
        self.knowledge_client = knowledge_client

    async def execute(self, action: dict, state: dict) -> dict:
        """Execute an action and return result metadata."""
        action_type = action.get("type")

        if action_type == "api_call":
            result = await self.execute_api_call(action, state)
            return {"api_result": result, "success": result is not None}

        elif action_type == "rag_search":
            result = await self.execute_rag_search(action, state)
            return {"rag_result": result}

        elif action_type == "transfer":
            return {"transfer": True, "reason": action.get("reason", ""), "slots": action.get("slots", {})}

        elif action_type == "end":
            return {"completed": True, "disposition": action.get("disposition", "resolved")}

        elif action_type == "llm_response":
            return {"llm_response": True}

        return {}

    async def execute_api_call(self, action: dict, state: dict) -> Any | None:
        """Execute an HTTP API call. Returns JSON response or None on error."""
        method = action.get("method", "GET")
        url = action.get("url", "")
        body = action.get("body")
        headers = action.get("headers", {})
        timeout_ms = action.get("timeout_ms", 5000)
        result_var = action.get("result_var")

        try:
            async with httpx.AsyncClient(timeout=timeout_ms / 1000) as client:
                resp = await client.request(
                    method=method,
                    url=url,
                    json=body,
                    headers=headers,
                )
                json_result = resp.json()
                if inspect.iscoroutine(json_result):
                    result = await json_result
                else:
                    result = json_result
                if result_var:
                    state.setdefault("variables", {})[result_var] = result
                return result
        except Exception:
            logger.exception(f"API call failed: {method} {url}")
            return None

    async def execute_rag_search(self, action: dict, state: dict) -> list | None:
        """Execute RAG search via Knowledge Service."""
        if not self.knowledge_client:
            logger.warning("No knowledge client for RAG search")
            return None

        query = action.get("query", "")
        top_k = action.get("top_k", 3)
        result_var = action.get("result_var")

        try:
            results = await self.knowledge_client.search(query, top_k)
            if result_var:
                state.setdefault("variables", {})[result_var] = results
            return results
        except Exception:
            logger.exception("RAG search failed")
            return None
