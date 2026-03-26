"""Test scenario cache."""
import json
import pytest
from pathlib import Path

from realtime_demo.dialogue.scenario_cache import ScenarioCache
from realtime_demo.dialogue.models import Scenario

FIXTURE_DIR = Path(__file__).parent / "fixtures"


class TestScenarioCache:
    def test_load_from_fixture(self):
        with open(FIXTURE_DIR / "card_lost_scenario.json") as f:
            data = json.load(f)
        cache = ScenarioCache(knowledge_base_url="http://localhost:8100/api/brain")
        cache.load_from_dicts([data])
        assert "card-lost" in cache.scenarios
        assert isinstance(cache.scenarios["card-lost"], Scenario)

    def test_get_active_scenarios(self):
        with open(FIXTURE_DIR / "card_lost_scenario.json") as f:
            data = json.load(f)
        cache = ScenarioCache(knowledge_base_url="http://localhost:8100/api/brain")
        cache.load_from_dicts([data])
        active = cache.get_active_scenarios()
        assert len(active) == 1
        assert active[0].id == "card-lost"

    def test_empty_cache(self):
        cache = ScenarioCache(knowledge_base_url="http://localhost:8100/api/brain")
        assert cache.get_active_scenarios() == []
        assert cache.get_scenario("nonexistent") is None
