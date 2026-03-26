"""Test data models."""
import json
from pathlib import Path

from realtime_demo.dialogue.models import Scenario, Node, Edge, SlotDef


FIXTURE_DIR = Path(__file__).parent / "fixtures"


class TestScenarioFromDict:
    def _load_fixture(self) -> Scenario:
        return Scenario.from_json_file(FIXTURE_DIR / "card_lost_scenario.json")

    def test_basic_fields(self):
        s = self._load_fixture()
        assert s.id == "card-lost"
        assert s.name == "카드 분실 신고"
        assert s.version == 1
        assert s.status == "active"

    def test_slots_parsed(self):
        s = self._load_fixture()
        assert "card_number" in s.slots
        assert s.slots["card_number"].extract == "regex"
        assert s.slots["card_number"].pattern is not None
        assert "loss_type" in s.slots
        assert s.slots["loss_type"].extract == "llm"
        assert s.slots["loss_type"].values == ["분실", "도난"]

    def test_nodes_parsed(self):
        s = self._load_fixture()
        assert len(s.nodes) == 11
        assert s.nodes["n1"].type == "speak"
        assert s.nodes["n2"].type == "slot_collect"
        assert s.nodes["n5"].type == "condition"

    def test_edges_parsed(self):
        s = self._load_fixture()
        assert len(s.edges) == 10
        labeled = [e for e in s.edges if e.label is not None]
        assert len(labeled) == 4  # yes, no, true, false

    def test_get_next_node_linear(self):
        s = self._load_fixture()
        assert s.get_next_node_id("n1") == "n2"
        assert s.get_next_node_id("n2") == "n3"

    def test_get_next_node_branched(self):
        s = self._load_fixture()
        assert s.get_next_node_id("n5", "true") == "n6"
        assert s.get_next_node_id("n5", "false") == "n7"

    def test_get_start_node(self):
        s = self._load_fixture()
        assert s.get_start_node_id() == "n1"
