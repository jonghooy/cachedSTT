"""Scenario cache — loads and caches scenarios from Knowledge Service."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

from realtime_demo.dialogue.models import Scenario

logger = logging.getLogger(__name__)


class ScenarioCache:
    def __init__(self, knowledge_base_url: str = "http://localhost:8100/api/brain"):
        self.knowledge_base_url = knowledge_base_url
        self.scenarios: dict[str, Scenario] = {}
        self._version: str | None = None
        self._trigger_data: list[dict] = []

    def load_from_dicts(self, scenario_dicts: list[dict]):
        """Load scenarios from a list of raw dicts (API response or fixtures)."""
        self.scenarios.clear()
        for data in scenario_dicts:
            try:
                # Map Knowledge Service field names to Scenario.from_dict expected format
                mapped = self._map_knowledge_fields(data)
                scenario = Scenario.from_dict(mapped)
                if scenario.status == "active":
                    self.scenarios[scenario.id] = scenario
            except Exception:
                logger.exception(f"Failed to parse scenario: {data.get('id', 'unknown')}")

    @staticmethod
    def _map_knowledge_fields(data: dict) -> dict:
        """Map Knowledge Service API fields to Scenario.from_dict format.

        Handles both formats:
        - Direct format (fixtures): has 'triggers', 'slots', 'nodes', 'edges' at top level
        - Knowledge Service format: has 'graph_json', 'triggers_json', 'slots_json', 'metadata_json'
        """
        def _parse(val):
            if isinstance(val, str):
                try:
                    return json.loads(val)
                except (ValueError, TypeError):
                    return {}
            return val if val else {}

        graph = _parse(data.get("graph_json", {}))
        triggers = _parse(data.get("triggers_json", data.get("triggers", {})))
        slots = _parse(data.get("slots_json", data.get("slots", {})))
        metadata = _parse(data.get("metadata_json", data.get("metadata", {})))

        # For nodes/edges: prefer graph_json contents, fall back to top-level keys
        nodes = graph.get("nodes", []) or data.get("nodes", [])
        edges = graph.get("edges", []) or data.get("edges", [])

        return {
            "id": str(data.get("id", "")),
            "name": data.get("name", ""),
            "description": data.get("description", ""),
            "version": data.get("version", 1),
            "schema_version": data.get("schema_version", 1),
            "status": data.get("status", "draft"),
            "priority": data.get("priority", 0),
            "triggers": triggers,
            "slots": slots,
            "nodes": nodes,
            "edges": edges,
            "metadata": metadata,
        }

    async def refresh(self, retries: int = 3) -> bool:
        for attempt in range(retries):
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(f"{self.knowledge_base_url}/scenarios")
                    if resp.status_code != 200:
                        logger.error(f"Failed to fetch scenarios: {resp.status_code}")
                        if attempt < retries - 1:
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                            continue
                        return False
                    data = resp.json()
                    scenario_list = data.get("scenarios", [])
                    self._version = data.get("version")
                    self.load_from_dicts(scenario_list)
                    # Extract trigger embeddings for IntentMatcher
                    self._trigger_data = []
                    for s_data in scenario_list:
                        sid = str(s_data.get("id", ""))
                        triggers = s_data.get("triggers_json", s_data.get("triggers", {}))
                        if isinstance(triggers, str):
                            triggers = json.loads(triggers)
                        examples = triggers.get("examples", []) if isinstance(triggers, dict) else []
                        embeddings = s_data.get("trigger_embeddings", [])
                        if examples and embeddings and len(examples) == len(embeddings):
                            self._trigger_data.append({
                                "scenario_id": sid,
                                "triggers": examples,
                                "embeddings": embeddings,
                            })
                    logger.info(f"Loaded {len(self.scenarios)} active scenarios (version: {self._version})")
                    return True
            except Exception:
                logger.exception(f"Failed to refresh scenario cache (attempt {attempt + 1}/{retries})")
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
        return False

    def get_scenario(self, scenario_id: str) -> Scenario | None:
        return self.scenarios.get(scenario_id)

    def get_active_scenarios(self) -> list[Scenario]:
        return list(self.scenarios.values())

    def get_trigger_data_with_embeddings(self) -> list[dict]:
        """Return trigger data with pre-computed embeddings for IntentMatcher."""
        return self._trigger_data

    def get_trigger_data(self) -> list[dict]:
        result = []
        for s in self.scenarios.values():
            examples = s.triggers.get("examples", [])
            result.append({"scenario_id": s.id, "triggers": examples, "description": s.triggers.get("description", "")})
        return result
