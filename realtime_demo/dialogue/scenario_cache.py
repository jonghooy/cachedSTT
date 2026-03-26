"""Scenario cache — loads and caches scenarios from Knowledge Service."""
from __future__ import annotations

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

    def load_from_dicts(self, scenario_dicts: list[dict]):
        self.scenarios.clear()
        for data in scenario_dicts:
            try:
                scenario = Scenario.from_dict(data)
                if scenario.status == "active":
                    self.scenarios[scenario.id] = scenario
            except Exception:
                logger.exception(f"Failed to parse scenario: {data.get('id', 'unknown')}")

    async def refresh(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.knowledge_base_url}/scenarios")
                if resp.status_code != 200:
                    logger.error(f"Failed to fetch scenarios: {resp.status_code}")
                    return False
                data = resp.json()
                scenario_list = data.get("scenarios", [])
                self._version = data.get("version")
                self.load_from_dicts(scenario_list)
                logger.info(f"Loaded {len(self.scenarios)} active scenarios (version: {self._version})")
                return True
        except Exception:
            logger.exception("Failed to refresh scenario cache")
            return False

    def get_scenario(self, scenario_id: str) -> Scenario | None:
        return self.scenarios.get(scenario_id)

    def get_active_scenarios(self) -> list[Scenario]:
        return list(self.scenarios.values())

    def get_trigger_data(self) -> list[dict]:
        result = []
        for s in self.scenarios.values():
            examples = s.triggers.get("examples", [])
            result.append({"scenario_id": s.id, "triggers": examples, "description": s.triggers.get("description", "")})
        return result
