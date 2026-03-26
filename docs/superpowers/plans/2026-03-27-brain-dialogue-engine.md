# Brain Dialogue Engine Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Dialogue Engine to Brain (cachedSTT) that executes scenario graphs, matching intents via embeddings + LLM, walking nodes, extracting slots, and running actions — with graceful fallback to existing S2S pipeline.

**Architecture:** Hybrid graph-based dialogue engine inserted between Turn Detection and S2S pipeline. When a user utterance matches a scenario trigger, the engine enters scenario mode and walks the graph node-by-node. Non-matching utterances fall through to the existing freeform LLM pipeline unchanged.

**Tech Stack:** Python 3.10+, pytest, pytest-asyncio, numpy (cosine similarity), httpx (async HTTP), existing LLMEngine from s2s_pipeline.py

**Prerequisites:** `pip install pytest-asyncio` (required for async tests)

**Spec:** `docs/superpowers/specs/2026-03-27-scenario-builder-design.md`

**Scope:** Brain-side only (Plan 1 of 3). Knowledge Service backend/frontend are separate plans. This plan uses fixture JSON scenarios for testing.

---

## File Structure

```
realtime_demo/
├── dialogue/
│   ├── __init__.py              # Exports DialogueEngine, DialogueResult
│   ├── models.py                # Dataclasses: Scenario, Node, Edge, SlotDef, DialogueResult, IntentMatch
│   ├── rule_evaluator.py        # Safe expression evaluator (no eval)
│   ├── template_renderer.py     # {{slots.x}} → value substitution
│   ├── slot_manager.py          # Slot extraction: regex + LLM
│   ├── graph_walker.py          # Node graph traversal engine
│   ├── action_runner.py         # API calls, transfer, end actions
│   ├── intent_matcher.py        # Cosine similarity + LLM intent matching
│   ├── scenario_cache.py        # Load/cache scenarios from Knowledge Service
│   └── engine.py                # DialogueEngine orchestrator
├── tests/
│   ├── test_turn_detector.py    # Existing (unchanged)
│   ├── test_rule_evaluator.py   # New
│   ├── test_template_renderer.py # New
│   ├── test_slot_manager.py     # New
│   ├── test_graph_walker.py     # New
│   ├── test_action_runner.py    # New
│   ├── test_intent_matcher.py   # New
│   ├── test_dialogue_engine.py  # New
│   └── fixtures/
│       └── card_lost_scenario.json  # Test scenario fixture
```

**Existing files modified:**
- `realtime_demo/server.py` — Add `dialogue_mode` + `scenario_state` to StreamingSession, integrate DialogueEngine at endpoint dispatch

---

### Task 1: Data Models + Test Fixture

**Files:**
- Create: `realtime_demo/dialogue/__init__.py`
- Create: `realtime_demo/dialogue/models.py`
- Create: `realtime_demo/tests/fixtures/card_lost_scenario.json`

- [ ] **Step 1: Create test fixture JSON**

Create a realistic "카드 분실 신고" scenario fixture that all subsequent tests will use.

```json
{
  "id": "card-lost",
  "name": "카드 분실 신고",
  "description": "카드 분실/도난 시 카드 정지 처리",
  "version": 1,
  "schema_version": 1,
  "status": "active",
  "priority": 10,
  "created_by": "ai",
  "triggers": {
    "examples": [
      "카드를 잃어버렸어요",
      "카드 분실 신고 하려고요",
      "카드가 없어졌는데요",
      "신용카드를 도난당했어요"
    ],
    "description": "고객이 카드 분실/도난을 신고하려는 의도"
  },
  "slots": {
    "card_number": {
      "type": "string",
      "required": true,
      "extract": "regex",
      "pattern": "\\d{4}[- ]?\\d{4}[- ]?\\d{4}[- ]?\\d{4}",
      "prompt": "카드번호 16자리를 말씀해주세요."
    },
    "loss_type": {
      "type": "enum",
      "required": true,
      "extract": "llm",
      "values": ["분실", "도난"],
      "prompt": "분실이신가요, 도난이신가요?"
    }
  },
  "nodes": [
    {"id": "n1", "type": "speak", "text": "카드 분실 신고 도와드리겠습니다. 카드번호 16자리를 말씀해주세요."},
    {"id": "n2", "type": "slot_collect", "target_slot": "card_number", "max_retries": 2, "retry_prompt": "카드번호를 다시 말씀해주세요.", "fail_action": "transfer"},
    {"id": "n3", "type": "slot_collect", "target_slot": "loss_type", "max_retries": 2, "retry_prompt": "분실인지 도난인지 말씀해주세요.", "fail_action": "transfer"},
    {"id": "n4", "type": "confirm", "template": "카드번호 {{slots.card_number}}, {{slots.loss_type}} 맞으시죠?"},
    {"id": "n5", "type": "condition", "mode": "rule", "rule": {"field": "slots.loss_type", "op": "eq", "value": "도난"}},
    {"id": "n6", "type": "speak", "text": "경찰 신고도 함께 진행해주시기 바랍니다."},
    {"id": "n7", "type": "api_call", "method": "POST", "url": "https://internal-api/card/block", "body": {"card_no": "{{slots.card_number}}", "type": "{{slots.loss_type}}"}, "result_var": "block_result", "timeout_ms": 5000, "on_error": "n_err"},
    {"id": "n8", "type": "end", "message": "카드 정지 처리 완료되었습니다. 감사합니다.", "disposition": "resolved", "tags": ["card", "lost"]},
    {"id": "n9", "type": "transfer", "reason": "slot_fail", "message": "상담원에게 연결해드리겠습니다.", "transfer_data": {"slots": "all"}},
    {"id": "n_err", "type": "speak", "text": "시스템 오류가 발생했습니다. 상담원에게 연결해드리겠습니다."},
    {"id": "n_err2", "type": "transfer", "reason": "api_error", "message": "상담원에게 연결해드리겠습니다.", "transfer_data": {"slots": "all"}}
  ],
  "edges": [
    {"from": "n1", "to": "n2"},
    {"from": "n2", "to": "n3"},
    {"from": "n3", "to": "n4"},
    {"from": "n4", "to": "n5", "label": "yes"},
    {"from": "n4", "to": "n2", "label": "no"},
    {"from": "n5", "to": "n6", "label": "true"},
    {"from": "n5", "to": "n7", "label": "false"},
    {"from": "n6", "to": "n7"},
    {"from": "n7", "to": "n8"},
    {"from": "n_err", "to": "n_err2"}
  ],
  "metadata": {
    "created_at": "2026-03-27T10:00:00Z",
    "updated_at": "2026-03-27T10:00:00Z",
    "test_coverage": 0.0,
    "execution_count": 0,
    "avg_completion_rate": null
  }
}
```

Write to: `realtime_demo/tests/fixtures/card_lost_scenario.json`

- [ ] **Step 2: Write models.py**

```python
"""Scenario dialogue data models."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SlotDef:
    """Slot definition from scenario JSON."""
    name: str
    type: str           # "string" | "enum" | "boolean"
    required: bool
    extract: str         # "llm" | "regex"
    prompt: str
    pattern: str | None = None    # regex pattern (extract=="regex")
    values: list[str] | None = None  # enum values (type=="enum")


@dataclass
class Node:
    """A single node in the scenario graph."""
    id: str
    type: str   # speak|slot_collect|condition|api_call|transfer|end|llm_response|confirm|rag_search
    data: dict  # All other fields specific to node type


@dataclass
class Edge:
    """Directed edge between nodes."""
    from_id: str
    to_id: str
    label: str | None = None   # "true"/"false", "yes"/"no", or None


@dataclass
class Scenario:
    """A complete scenario definition."""
    id: str
    name: str
    description: str
    version: int
    schema_version: int
    status: str
    priority: int
    triggers: dict           # {"examples": [...], "description": str}
    slots: dict[str, SlotDef]
    nodes: dict[str, Node]   # keyed by node id
    edges: list[Edge]
    metadata: dict = field(default_factory=dict)

    def get_node(self, node_id: str) -> Node | None:
        return self.nodes.get(node_id)

    def get_outgoing_edges(self, node_id: str, label: str | None = None) -> list[Edge]:
        """Get edges from node_id, optionally filtered by label."""
        edges = [e for e in self.edges if e.from_id == node_id]
        if label is not None:
            edges = [e for e in edges if e.label == label]
        return edges

    def get_next_node_id(self, node_id: str, label: str | None = None) -> str | None:
        """Get the next node id via edge. For linear flow, label=None."""
        edges = self.get_outgoing_edges(node_id, label)
        if not edges:
            # Fallback: try unlabeled edge if labeled edge not found
            if label is not None:
                edges = self.get_outgoing_edges(node_id, None)
        return edges[0].to_id if edges else None

    def get_start_node_id(self) -> str | None:
        """First node in the nodes list."""
        if self.nodes:
            return next(iter(self.nodes))
        return None

    @classmethod
    def from_dict(cls, data: dict) -> Scenario:
        """Parse scenario from JSON dict."""
        slots = {}
        for name, sdef in data.get("slots", {}).items():
            slots[name] = SlotDef(
                name=name,
                type=sdef["type"],
                required=sdef["required"],
                extract=sdef["extract"],
                prompt=sdef["prompt"],
                pattern=sdef.get("pattern"),
                values=sdef.get("values"),
            )

        nodes = {}
        for ndata in data.get("nodes", []):
            node_id = ndata["id"]
            node_type = ndata["type"]
            extra = {k: v for k, v in ndata.items() if k not in ("id", "type")}
            nodes[node_id] = Node(id=node_id, type=node_type, data=extra)

        edges = []
        for edata in data.get("edges", []):
            edges.append(Edge(
                from_id=edata["from"],
                to_id=edata["to"],
                label=edata.get("label"),
            ))

        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            version=data.get("version", 1),
            schema_version=data.get("schema_version", 1),
            status=data.get("status", "draft"),
            priority=data.get("priority", 0),
            triggers=data.get("triggers", {}),
            slots=slots,
            nodes=nodes,
            edges=edges,
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> Scenario:
        with open(path) as f:
            return cls.from_dict(json.load(f))


@dataclass
class DialogueResult:
    """Result of processing one utterance through the dialogue engine."""
    mode: str = "freeform"              # "freeform" | "scenario"
    response_text: str | None = None    # Text for TTS output
    should_use_s2s: bool = False        # True → use existing S2S pipeline
    action: dict | None = None          # {"type": "transfer"|"end"|"api_call", ...}
    awaiting_input: bool = False        # True → waiting for user response (slot/confirm)


@dataclass
class IntentMatch:
    """Result of intent matching."""
    scenario_id: str
    scenario: Scenario
    confidence: float
    matched_example: str | None = None
```

Write to: `realtime_demo/dialogue/models.py`

- [ ] **Step 3: Write `__init__.py`**

```python
"""Dialogue engine for scenario-based conversation management."""
from realtime_demo.dialogue.models import DialogueResult, Scenario, IntentMatch
```

Write to: `realtime_demo/dialogue/__init__.py`

- [ ] **Step 4: Write test for Scenario.from_dict**

```python
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
```

Write to: `realtime_demo/tests/test_models.py`

- [ ] **Step 5: Run tests**

Run: `cd /home/jonghooy/work/cachedSTT && python -m pytest realtime_demo/tests/test_models.py -v`
Expected: All 7 tests PASS

- [ ] **Step 6: Commit**

```bash
git add realtime_demo/dialogue/ realtime_demo/tests/test_models.py realtime_demo/tests/fixtures/
git commit -m "feat(dialogue): add data models and test fixture for scenario engine"
```

---

### Task 2: Safe Rule Evaluator

**Files:**
- Create: `realtime_demo/dialogue/rule_evaluator.py`
- Create: `realtime_demo/tests/test_rule_evaluator.py`

The condition node uses `{"field": "slots.loss_type", "op": "eq", "value": "도난"}` — no `eval()` allowed.

- [ ] **Step 1: Write failing tests**

```python
"""Test safe rule evaluator."""
import pytest
from realtime_demo.dialogue.rule_evaluator import evaluate_rule


class TestEvaluateRule:
    def _ctx(self, **kwargs):
        """Build a context dict with slots and variables."""
        return {"slots": kwargs.get("slots", {}), "variables": kwargs.get("variables", {})}

    def test_eq(self):
        rule = {"field": "slots.loss_type", "op": "eq", "value": "도난"}
        assert evaluate_rule(rule, self._ctx(slots={"loss_type": "도난"})) is True

    def test_eq_false(self):
        rule = {"field": "slots.loss_type", "op": "eq", "value": "도난"}
        assert evaluate_rule(rule, self._ctx(slots={"loss_type": "분실"})) is False

    def test_neq(self):
        rule = {"field": "slots.loss_type", "op": "neq", "value": "도난"}
        assert evaluate_rule(rule, self._ctx(slots={"loss_type": "분실"})) is True

    def test_contains(self):
        rule = {"field": "slots.card_number", "op": "contains", "value": "1234"}
        assert evaluate_rule(rule, self._ctx(slots={"card_number": "1234-5678-9012-3456"})) is True

    def test_exists_true(self):
        rule = {"field": "slots.card_number", "op": "exists"}
        assert evaluate_rule(rule, self._ctx(slots={"card_number": "1234"})) is True

    def test_exists_false(self):
        rule = {"field": "slots.card_number", "op": "exists"}
        assert evaluate_rule(rule, self._ctx(slots={})) is False

    def test_in_list(self):
        rule = {"field": "slots.loss_type", "op": "in", "value": ["분실", "도난"]}
        assert evaluate_rule(rule, self._ctx(slots={"loss_type": "분실"})) is True

    def test_gt(self):
        rule = {"field": "variables.retry_count", "op": "gt", "value": 2}
        assert evaluate_rule(rule, self._ctx(variables={"retry_count": 3})) is True

    def test_lt(self):
        rule = {"field": "variables.retry_count", "op": "lt", "value": 2}
        assert evaluate_rule(rule, self._ctx(variables={"retry_count": 1})) is True

    def test_and_compound(self):
        rule = {"and": [
            {"field": "slots.loss_type", "op": "eq", "value": "도난"},
            {"field": "slots.card_number", "op": "exists"},
        ]}
        ctx = self._ctx(slots={"loss_type": "도난", "card_number": "1234"})
        assert evaluate_rule(rule, ctx) is True

    def test_or_compound(self):
        rule = {"or": [
            {"field": "slots.loss_type", "op": "eq", "value": "도난"},
            {"field": "slots.loss_type", "op": "eq", "value": "분실"},
        ]}
        assert evaluate_rule(rule, self._ctx(slots={"loss_type": "분실"})) is True

    def test_missing_field_returns_false(self):
        rule = {"field": "slots.nonexistent", "op": "eq", "value": "x"}
        assert evaluate_rule(rule, self._ctx()) is False

    def test_invalid_op_raises(self):
        rule = {"field": "slots.x", "op": "EVAL", "value": "__import__('os')"}
        with pytest.raises(ValueError, match="Unsupported operator"):
            evaluate_rule(rule, self._ctx())
```

Write to: `realtime_demo/tests/test_rule_evaluator.py`

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/jonghooy/work/cachedSTT && python -m pytest realtime_demo/tests/test_rule_evaluator.py -v`
Expected: FAIL (ImportError: cannot import name 'evaluate_rule')

- [ ] **Step 3: Implement rule_evaluator.py**

```python
"""Safe rule evaluator — no eval(), no exec()."""
from __future__ import annotations

from typing import Any

_OPERATORS = {"eq", "neq", "contains", "gt", "lt", "exists", "in"}


def _resolve_field(field_path: str, context: dict) -> Any:
    """Resolve dotted field path like 'slots.card_number' from context dict."""
    parts = field_path.split(".")
    current = context
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return _MISSING
    return current


class _MissingSentinel:
    """Sentinel for missing field values."""
    pass


_MISSING = _MissingSentinel()


def evaluate_rule(rule: dict, context: dict) -> bool:
    """Evaluate a rule expression against context.

    Supports:
        Simple: {"field": "slots.x", "op": "eq", "value": "y"}
        Compound: {"and": [rule1, rule2, ...]} or {"or": [rule1, ...]}

    Operators: eq, neq, contains, gt, lt, exists, in

    Args:
        rule: Rule dict from scenario condition node
        context: {"slots": {...}, "variables": {...}}

    Returns:
        bool: evaluation result

    Raises:
        ValueError: if operator is not in the allowed set
    """
    # Compound rules
    if "and" in rule:
        return all(evaluate_rule(r, context) for r in rule["and"])
    if "or" in rule:
        return any(evaluate_rule(r, context) for r in rule["or"])

    # Simple rule
    op = rule.get("op", "")
    if op not in _OPERATORS:
        raise ValueError(f"Unsupported operator: {op!r}")

    field_path = rule.get("field", "")
    value = rule.get("value")
    resolved = _resolve_field(field_path, context)

    if op == "exists":
        return not isinstance(resolved, _MissingSentinel)

    if isinstance(resolved, _MissingSentinel):
        return False

    if op == "eq":
        return resolved == value
    if op == "neq":
        return resolved != value
    if op == "contains":
        return value in str(resolved)
    if op == "gt":
        return resolved > value
    if op == "lt":
        return resolved < value
    if op == "in":
        return resolved in value

    return False
```

Write to: `realtime_demo/dialogue/rule_evaluator.py`

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/jonghooy/work/cachedSTT && python -m pytest realtime_demo/tests/test_rule_evaluator.py -v`
Expected: All 13 tests PASS

- [ ] **Step 5: Commit**

```bash
git add realtime_demo/dialogue/rule_evaluator.py realtime_demo/tests/test_rule_evaluator.py
git commit -m "feat(dialogue): add safe rule evaluator for condition nodes"
```

---

### Task 3: Template Renderer

**Files:**
- Create: `realtime_demo/dialogue/template_renderer.py`
- Create: `realtime_demo/tests/test_template_renderer.py`

Handles `{{slots.card_number}}` and `{{variables.block_result}}` substitution in speak text, API body, etc.

- [ ] **Step 1: Write failing tests**

```python
"""Test template renderer."""
from realtime_demo.dialogue.template_renderer import render_template


class TestRenderTemplate:
    def _ctx(self, **kwargs):
        return {"slots": kwargs.get("slots", {}), "variables": kwargs.get("variables", {})}

    def test_simple_slot(self):
        result = render_template("카드번호 {{slots.card_number}}", self._ctx(slots={"card_number": "1234-5678"}))
        assert result == "카드번호 1234-5678"

    def test_multiple_slots(self):
        ctx = self._ctx(slots={"card_number": "1234", "loss_type": "분실"})
        result = render_template("{{slots.card_number}}, {{slots.loss_type}}", ctx)
        assert result == "1234, 분실"

    def test_variable(self):
        result = render_template("결과: {{variables.block_result}}", self._ctx(variables={"block_result": "OK"}))
        assert result == "결과: OK"

    def test_missing_value_empty_string(self):
        result = render_template("{{slots.missing}}", self._ctx())
        assert result == ""

    def test_no_template(self):
        result = render_template("안녕하세요", self._ctx())
        assert result == "안녕하세요"

    def test_render_dict(self):
        """Render templates inside a dict (for API body)."""
        from realtime_demo.dialogue.template_renderer import render_dict
        body = {"card_no": "{{slots.card_number}}", "type": "{{slots.loss_type}}"}
        ctx = self._ctx(slots={"card_number": "1234", "loss_type": "분실"})
        result = render_dict(body, ctx)
        assert result == {"card_no": "1234", "type": "분실"}
```

Write to: `realtime_demo/tests/test_template_renderer.py`

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/jonghooy/work/cachedSTT && python -m pytest realtime_demo/tests/test_template_renderer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement template_renderer.py**

```python
"""Template renderer for {{slots.x}} and {{variables.x}} substitution."""
from __future__ import annotations

import re
from typing import Any

_TEMPLATE_RE = re.compile(r"\{\{(\w+(?:\.\w+)*)\}\}")


def _resolve(path: str, context: dict) -> str:
    """Resolve dotted path from context, return empty string if missing."""
    parts = path.split(".")
    current = context
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return ""
    return str(current)


def render_template(template: str, context: dict) -> str:
    """Replace all {{path}} placeholders in template string."""
    return _TEMPLATE_RE.sub(lambda m: _resolve(m.group(1), context), template)


def render_dict(d: dict, context: dict) -> dict:
    """Recursively render templates in a dict's string values."""
    result = {}
    for k, v in d.items():
        if isinstance(v, str):
            result[k] = render_template(v, context)
        elif isinstance(v, dict):
            result[k] = render_dict(v, context)
        elif isinstance(v, list):
            result[k] = [
                render_dict(i, context) if isinstance(i, dict)
                else (render_template(i, context) if isinstance(i, str) else i)
                for i in v
            ]
        else:
            result[k] = v
    return result
```

Write to: `realtime_demo/dialogue/template_renderer.py`

- [ ] **Step 4: Run tests**

Run: `cd /home/jonghooy/work/cachedSTT && python -m pytest realtime_demo/tests/test_template_renderer.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add realtime_demo/dialogue/template_renderer.py realtime_demo/tests/test_template_renderer.py
git commit -m "feat(dialogue): add template renderer for slot/variable substitution"
```

---

### Task 4: Slot Manager

**Files:**
- Create: `realtime_demo/dialogue/slot_manager.py`
- Create: `realtime_demo/tests/test_slot_manager.py`

Extracts slot values from user text via regex or LLM.

- [ ] **Step 1: Write failing tests**

```python
"""Test slot manager."""
import pytest
from unittest.mock import AsyncMock

from realtime_demo.dialogue.models import SlotDef
from realtime_demo.dialogue.slot_manager import SlotManager


class TestSlotManagerRegex:
    def test_extract_card_number(self):
        slot = SlotDef(name="card_number", type="string", required=True,
                       extract="regex", prompt="", pattern=r"\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}")
        sm = SlotManager(llm_engine=None)
        result = sm.extract_regex(slot, "카드번호는 1234-5678-9012-3456이에요")
        assert result == "1234-5678-9012-3456"

    def test_extract_card_number_spaces(self):
        slot = SlotDef(name="card_number", type="string", required=True,
                       extract="regex", prompt="", pattern=r"\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}")
        sm = SlotManager(llm_engine=None)
        result = sm.extract_regex(slot, "1234 5678 9012 3456입니다")
        assert result == "1234 5678 9012 3456"

    def test_no_match_returns_none(self):
        slot = SlotDef(name="card_number", type="string", required=True,
                       extract="regex", prompt="", pattern=r"\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}")
        sm = SlotManager(llm_engine=None)
        result = sm.extract_regex(slot, "잠깐만요")
        assert result is None


class TestSlotManagerLLM:
    @pytest.mark.asyncio
    async def test_extract_enum_via_llm(self):
        mock_llm = AsyncMock()
        mock_llm.generate_stream = AsyncMock(return_value=_async_iter(["분실"]))
        sm = SlotManager(llm_engine=mock_llm)
        slot = SlotDef(name="loss_type", type="enum", required=True,
                       extract="llm", prompt="", values=["분실", "도난"])
        result = await sm.extract_llm(slot, "카드를 잃어버렸어요")
        assert result == "분실"

    @pytest.mark.asyncio
    async def test_extract_enum_invalid_returns_none(self):
        mock_llm = AsyncMock()
        mock_llm.generate_stream = AsyncMock(return_value=_async_iter(["모르겠어요"]))
        sm = SlotManager(llm_engine=mock_llm)
        slot = SlotDef(name="loss_type", type="enum", required=True,
                       extract="llm", prompt="", values=["분실", "도난"])
        result = await sm.extract_llm(slot, "잘 모르겠어요")
        assert result is None


async def _async_iter(items):
    for item in items:
        yield item
```

Write to: `realtime_demo/tests/test_slot_manager.py`

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/jonghooy/work/cachedSTT && python -m pytest realtime_demo/tests/test_slot_manager.py -v`
Expected: FAIL

- [ ] **Step 3: Implement slot_manager.py**

```python
"""Slot extraction — regex for structured data, LLM for natural language."""
from __future__ import annotations

import logging
import re
from typing import Any

from realtime_demo.dialogue.models import SlotDef

logger = logging.getLogger(__name__)


class SlotManager:
    def __init__(self, llm_engine=None):
        self.llm_engine = llm_engine

    def extract_regex(self, slot: SlotDef, text: str) -> str | None:
        """Extract slot value via regex pattern match."""
        if not slot.pattern:
            return None
        match = re.search(slot.pattern, text)
        return match.group(0) if match else None

    async def extract_llm(self, slot: SlotDef, text: str) -> str | None:
        """Extract slot value via LLM."""
        if not self.llm_engine:
            logger.warning("LLM engine not available for slot extraction")
            return None

        if slot.type == "enum" and slot.values:
            prompt = f"사용자 발화: \"{text}\"\n선택지: {slot.values}\n위 발화에서 선택지 중 하나를 골라 해당 값만 출력하세요. 해당하는 값이 없으면 'NONE'을 출력하세요."
        elif slot.type == "boolean":
            prompt = f"사용자 발화: \"{text}\"\n위 발화가 긍정이면 'true', 부정이면 'false', 판단 불가면 'NONE'을 출력하세요."
        else:
            prompt = f"사용자 발화: \"{text}\"\n{slot.prompt}\n위 발화에서 해당 정보를 추출하세요. 없으면 'NONE'을 출력하세요."

        messages = [
            {"role": "system", "content": "You are a slot extraction assistant. Output ONLY the extracted value, nothing else."},
            {"role": "user", "content": prompt},
        ]

        try:
            result_text = ""
            async for token in self.llm_engine.generate_stream(messages, max_tokens=20, temperature=0.0):
                result_text += token
            result_text = result_text.strip()

            if result_text.upper() == "NONE" or not result_text:
                return None

            # Validate enum values
            if slot.type == "enum" and slot.values:
                if result_text not in slot.values:
                    return None

            # Validate boolean
            if slot.type == "boolean":
                if result_text.lower() in ("true", "예", "네"):
                    return True
                elif result_text.lower() in ("false", "아니요", "아니오"):
                    return False
                return None

            return result_text
        except Exception:
            logger.exception("LLM slot extraction failed")
            return None

    async def extract(self, slot: SlotDef, text: str) -> Any:
        """Extract slot value using the configured method."""
        if slot.extract == "regex":
            return self.extract_regex(slot, text)
        elif slot.extract == "llm":
            return await self.extract_llm(slot, text)
        return None
```

Write to: `realtime_demo/dialogue/slot_manager.py`

- [ ] **Step 4: Run tests**

Run: `cd /home/jonghooy/work/cachedSTT && python -m pytest realtime_demo/tests/test_slot_manager.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add realtime_demo/dialogue/slot_manager.py realtime_demo/tests/test_slot_manager.py
git commit -m "feat(dialogue): add slot manager with regex and LLM extraction"
```

---

### Task 5: Graph Walker

**Files:**
- Create: `realtime_demo/dialogue/graph_walker.py`
- Create: `realtime_demo/tests/test_graph_walker.py`

Core node traversal engine. Auto-advances through non-input nodes (speak, condition, api_call), stops at input nodes (slot_collect, confirm).

- [ ] **Step 1: Write failing tests**

```python
"""Test graph walker."""
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from realtime_demo.dialogue.models import Scenario, DialogueResult
from realtime_demo.dialogue.graph_walker import GraphWalker

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_scenario() -> Scenario:
    return Scenario.from_json_file(FIXTURE_DIR / "card_lost_scenario.json")


def _make_state(scenario_id="card-lost", current_node="n1"):
    return {
        "scenario_id": scenario_id,
        "scenario_version": 1,
        "current_node": current_node,
        "slots": {},
        "variables": {},
        "history": [],
        "retry_count": 0,
        "prosody": None,
        "stack": [],
        "awaiting_confirm": False,
    }


class TestSpeakNode:
    @pytest.mark.asyncio
    async def test_speak_returns_text_and_advances(self):
        scenario = _load_scenario()
        state = _make_state(current_node="n1")
        walker = GraphWalker(scenario, slot_manager=MagicMock(), llm_engine=None)
        result = await walker.step(state, "")
        # n1 is speak → should output text and advance to n2 (slot_collect)
        assert "카드 분실 신고" in result.response_text
        assert state["current_node"] == "n2"
        assert result.awaiting_input is True  # n2 is slot_collect


class TestSlotCollectNode:
    @pytest.mark.asyncio
    async def test_slot_collect_success(self):
        scenario = _load_scenario()
        state = _make_state(current_node="n2")
        mock_sm = MagicMock()
        mock_sm.extract = AsyncMock(return_value="1234-5678-9012-3456")
        walker = GraphWalker(scenario, slot_manager=mock_sm, llm_engine=None)
        result = await walker.step(state, "카드번호는 1234-5678-9012-3456이에요")
        assert state["slots"]["card_number"] == "1234-5678-9012-3456"
        assert state["retry_count"] == 0
        # Should advance past n2

    @pytest.mark.asyncio
    async def test_slot_collect_retry(self):
        scenario = _load_scenario()
        state = _make_state(current_node="n2")
        mock_sm = MagicMock()
        mock_sm.extract = AsyncMock(return_value=None)
        walker = GraphWalker(scenario, slot_manager=mock_sm, llm_engine=None)
        result = await walker.step(state, "잠깐만요")
        assert state["retry_count"] == 1
        assert "다시" in result.response_text
        assert state["current_node"] == "n2"  # stays at same node

    @pytest.mark.asyncio
    async def test_slot_collect_max_retry_transfer(self):
        scenario = _load_scenario()
        state = _make_state(current_node="n2")
        state["retry_count"] = 2  # max_retries=2, so next fail triggers fail_action
        mock_sm = MagicMock()
        mock_sm.extract = AsyncMock(return_value=None)
        walker = GraphWalker(scenario, slot_manager=mock_sm, llm_engine=None)
        result = await walker.step(state, "모르겠어요")
        assert result.action is not None
        assert result.action["type"] == "transfer"


class TestConditionNode:
    @pytest.mark.asyncio
    async def test_condition_true_branch(self):
        scenario = _load_scenario()
        state = _make_state(current_node="n5")
        state["slots"]["loss_type"] = "도난"
        walker = GraphWalker(scenario, slot_manager=MagicMock(), llm_engine=None)
        result = await walker.step(state, "")
        # n5 condition true → n6 (speak: 경찰 신고) → n7 (api_call)
        assert "경찰 신고" in result.response_text

    @pytest.mark.asyncio
    async def test_condition_false_branch(self):
        scenario = _load_scenario()
        state = _make_state(current_node="n5")
        state["slots"]["loss_type"] = "분실"
        walker = GraphWalker(scenario, slot_manager=MagicMock(), llm_engine=None)
        result = await walker.step(state, "")
        # n5 condition false → n7 (api_call) — no "경찰" in text
        assert result.response_text is None or "경찰" not in (result.response_text or "")


class TestConfirmNode:
    @pytest.mark.asyncio
    async def test_confirm_asks_question(self):
        scenario = _load_scenario()
        state = _make_state(current_node="n4")
        state["slots"] = {"card_number": "1234-5678", "loss_type": "분실"}
        walker = GraphWalker(scenario, slot_manager=MagicMock(), llm_engine=None)
        result = await walker.step(state, "")
        assert "1234-5678" in result.response_text
        assert "분실" in result.response_text
        assert result.awaiting_input is True


class TestEndNode:
    @pytest.mark.asyncio
    async def test_end_returns_action(self):
        scenario = _load_scenario()
        state = _make_state(current_node="n8")
        walker = GraphWalker(scenario, slot_manager=MagicMock(), llm_engine=None)
        result = await walker.step(state, "")
        assert result.action is not None
        assert result.action["type"] == "end"
        assert result.action["disposition"] == "resolved"
        assert "완료" in result.response_text


class TestInfiniteLoopDetection:
    @pytest.mark.asyncio
    async def test_loop_detection(self):
        scenario = _load_scenario()
        state = _make_state(current_node="n1")
        state["history"] = ["n1", "n1", "n1"]  # already visited 3 times
        walker = GraphWalker(scenario, slot_manager=MagicMock(), llm_engine=None)
        result = await walker.step(state, "")
        assert result.action is not None
        assert result.action["type"] == "end"
        assert "loop" in result.action.get("reason", "").lower() or result.action.get("disposition") == "error"
```

Write to: `realtime_demo/tests/test_graph_walker.py`

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/jonghooy/work/cachedSTT && python -m pytest realtime_demo/tests/test_graph_walker.py -v`
Expected: FAIL

- [ ] **Step 3: Implement graph_walker.py**

```python
"""Graph walker — traverses scenario node graph step by step."""
from __future__ import annotations

import logging
from typing import Any

from realtime_demo.dialogue.models import Scenario, Node, DialogueResult
from realtime_demo.dialogue.rule_evaluator import evaluate_rule
from realtime_demo.dialogue.template_renderer import render_template, render_dict

logger = logging.getLogger(__name__)

MAX_VISITS_PER_NODE = 3
MAX_AUTO_ADVANCE = 20  # prevent infinite auto-advance chains


class GraphWalker:
    def __init__(self, scenario: Scenario, slot_manager, llm_engine=None):
        self.scenario = scenario
        self.slot_manager = slot_manager
        self.llm_engine = llm_engine

    async def step(self, state: dict, user_text: str) -> DialogueResult:
        """Execute one step from the current node.

        For auto-advance nodes (speak, condition, api_call, rag_search),
        continues to the next node automatically. Stops at input nodes
        (slot_collect, confirm) or terminal nodes (end, transfer).
        """
        collected_text_parts: list[str] = []
        auto_steps = 0

        while auto_steps < MAX_AUTO_ADVANCE:
            auto_steps += 1
            node_id = state["current_node"]
            node = self.scenario.get_node(node_id)

            if node is None:
                return self._error_result(state, f"Node {node_id} not found")

            # Infinite loop detection
            visit_count = state["history"].count(node_id)
            if visit_count >= MAX_VISITS_PER_NODE:
                logger.warning(f"Infinite loop detected at node {node_id}")
                return DialogueResult(
                    mode="scenario",
                    response_text="시스템 오류가 발생했습니다.",
                    action={"type": "end", "disposition": "error", "reason": "infinite_loop"},
                )

            state["history"].append(node_id)

            result = await self._execute_node(node, state, user_text)

            if result.response_text:
                collected_text_parts.append(result.response_text)

            # Terminal conditions: stop auto-advance
            if result.action is not None:
                result.response_text = " ".join(collected_text_parts) if collected_text_parts else result.response_text
                return result
            if result.awaiting_input:
                result.response_text = " ".join(collected_text_parts) if collected_text_parts else result.response_text
                return result

            # Auto-advance: user_text only applies to first input node
            user_text = ""

        return self._error_result(state, "Max auto-advance steps exceeded")

    async def _execute_node(self, node: Node, state: dict, user_text: str) -> DialogueResult:
        """Execute a single node and update state."""
        handler = getattr(self, f"_handle_{node.type}", None)
        if handler is None:
            logger.error(f"Unknown node type: {node.type}")
            return self._error_result(state, f"Unknown node type: {node.type}")
        return await handler(node, state, user_text)

    async def _handle_speak(self, node: Node, state: dict, user_text: str) -> DialogueResult:
        ctx = {"slots": state["slots"], "variables": state["variables"]}
        text = render_template(node.data.get("text", ""), ctx)
        next_id = self.scenario.get_next_node_id(node.id)
        if next_id:
            state["current_node"] = next_id
        return DialogueResult(mode="scenario", response_text=text)

    async def _handle_slot_collect(self, node: Node, state: dict, user_text: str) -> DialogueResult:
        target_slot = node.data.get("target_slot")
        slot_def = self.scenario.slots.get(target_slot)
        if not slot_def:
            return self._error_result(state, f"Slot {target_slot} not defined")

        # If no user_text, ask the question (first entry into this node)
        if not user_text:
            return DialogueResult(mode="scenario", response_text=slot_def.prompt, awaiting_input=True)

        # Try extraction
        value = await self.slot_manager.extract(slot_def, user_text)
        if value is not None:
            state["slots"][target_slot] = value
            state["retry_count"] = 0
            next_id = self.scenario.get_next_node_id(node.id)
            if next_id:
                state["current_node"] = next_id
            return DialogueResult(mode="scenario")
        else:
            state["retry_count"] += 1
            max_retries = node.data.get("max_retries", 3)
            if state["retry_count"] > max_retries:
                return self._handle_fail_action(node, state)
            retry_prompt = node.data.get("retry_prompt", slot_def.prompt)
            return DialogueResult(mode="scenario", response_text=retry_prompt, awaiting_input=True)

    async def _handle_condition(self, node: Node, state: dict, user_text: str) -> DialogueResult:
        mode = node.data.get("mode", "rule")
        ctx = {"slots": state["slots"], "variables": state["variables"]}

        if mode == "rule":
            rule = node.data.get("rule", {})
            result = evaluate_rule(rule, ctx)
            label = "true" if result else "false"
        elif mode == "llm":
            # LLM-based condition evaluation
            label = await self._llm_condition(node, state, user_text)
        else:
            label = "false"

        next_id = self.scenario.get_next_node_id(node.id, label)
        if next_id:
            state["current_node"] = next_id
        else:
            # Fallback: try default (unlabeled) edge
            next_id = self.scenario.get_next_node_id(node.id)
            if next_id:
                state["current_node"] = next_id
        return DialogueResult(mode="scenario")

    async def _handle_confirm(self, node: Node, state: dict, user_text: str) -> DialogueResult:
        ctx = {"slots": state["slots"], "variables": state["variables"]}

        if not state.get("awaiting_confirm"):
            # First entry: ask confirmation question
            text = render_template(node.data.get("template", ""), ctx)
            state["awaiting_confirm"] = True
            return DialogueResult(mode="scenario", response_text=text, awaiting_input=True)
        else:
            # Second entry: evaluate yes/no
            state["awaiting_confirm"] = False
            confirmed = await self._judge_yes_no(user_text)
            label = "yes" if confirmed else "no"
            next_id = self.scenario.get_next_node_id(node.id, label)
            if next_id:
                state["current_node"] = next_id
            return DialogueResult(mode="scenario")

    async def _handle_api_call(self, node: Node, state: dict, user_text: str) -> DialogueResult:
        # API call is handled by action_runner, not in graph_walker
        # Advance current_node BEFORE returning action (auto-advance resumes after action execution)
        ctx = {"slots": state["slots"], "variables": state["variables"]}
        next_id = self.scenario.get_next_node_id(node.id)
        if next_id:
            state["current_node"] = next_id
        action = {
            "type": "api_call",
            "method": node.data.get("method", "GET"),
            "url": render_template(node.data.get("url", ""), ctx),
            "body": render_dict(node.data.get("body", {}), ctx) if node.data.get("body") else None,
            "headers": render_dict(node.data.get("headers", {}), ctx),
            "timeout_ms": node.data.get("timeout_ms", 5000),
            "result_var": node.data.get("result_var"),
            "on_error": node.data.get("on_error"),
        }
        return DialogueResult(mode="scenario", action=action)

    async def _handle_rag_search(self, node: Node, state: dict, user_text: str) -> DialogueResult:
        """RAG search is auto-advance: execute inline, store result, continue."""
        ctx = {"slots": state["slots"], "variables": state["variables"]}
        query = render_template(node.data.get("query_template", ""), ctx)
        result_var = node.data.get("result_var")
        # Store search params in variables for later execution by engine
        if result_var:
            state["variables"][f"_pending_rag_{result_var}"] = {
                "query": query,
                "top_k": node.data.get("top_k", 3),
                "result_var": result_var,
                "inject_to_next_llm": node.data.get("inject_to_next_llm", False),
            }
        next_id = self.scenario.get_next_node_id(node.id)
        if next_id:
            state["current_node"] = next_id
        return DialogueResult(mode="scenario")  # No action → auto-advance continues

    async def _handle_llm_response(self, node: Node, state: dict, user_text: str) -> DialogueResult:
        """LLM response: generates text via LLM, returns as response_text (NOT as action)."""
        if not self.llm_engine:
            next_id = self.scenario.get_next_node_id(node.id)
            if next_id:
                state["current_node"] = next_id
            return DialogueResult(mode="scenario", response_text="죄송합니다. 잠시 후 다시 시도해주세요.")

        instruction = node.data.get("instruction", "")
        context_slots = {k: state["slots"].get(k) for k in node.data.get("context_slots", [])}
        rag_context = state["variables"].get("rag_context", "")
        prosody = state.get("prosody")

        prompt = f"지시: {instruction}\n"
        if context_slots:
            prompt += f"슬롯: {context_slots}\n"
        if rag_context:
            prompt += f"참고 문서: {rag_context}\n"

        messages = [
            {"role": "system", "content": "당신은 친절한 한국어 상담원입니다. 1-2문장으로 답변하세요."},
            {"role": "user", "content": prompt},
        ]

        try:
            result_text = ""
            max_tokens = node.data.get("max_tokens", 120)
            async for token in self.llm_engine.generate_stream(messages, max_tokens=max_tokens):
                result_text += token
            response = result_text.strip()
        except Exception:
            logger.exception("LLM response node failed")
            response = "죄송합니다. 잠시 후 다시 시도해주세요."

        next_id = self.scenario.get_next_node_id(node.id)
        if next_id:
            state["current_node"] = next_id
        return DialogueResult(mode="scenario", response_text=response)  # auto-advance continues

    async def _handle_transfer(self, node: Node, state: dict, user_text: str) -> DialogueResult:
        text = node.data.get("message", "상담원에게 연결해드리겠습니다.")
        transfer_data = node.data.get("transfer_data", {})
        slots_to_send = state["slots"] if transfer_data.get("slots") == "all" else {}
        return DialogueResult(
            mode="scenario",
            response_text=text,
            action={
                "type": "transfer",
                "reason": node.data.get("reason", ""),
                "slots": slots_to_send,
            },
        )

    async def _handle_end(self, node: Node, state: dict, user_text: str) -> DialogueResult:
        text = node.data.get("message", "")
        return DialogueResult(
            mode="scenario",
            response_text=text,
            action={
                "type": "end",
                "disposition": node.data.get("disposition", "resolved"),
                "tags": node.data.get("tags", []),
            },
        )

    def _handle_fail_action(self, node: Node, state: dict) -> DialogueResult:
        fail_action = node.data.get("fail_action", "end")
        if fail_action == "transfer":
            return DialogueResult(
                mode="scenario",
                response_text="상담원에게 연결해드리겠습니다.",
                action={"type": "transfer", "reason": "slot_fail", "slots": state["slots"]},
            )
        elif fail_action == "skip":
            next_id = self.scenario.get_next_node_id(node.id)
            if next_id:
                state["current_node"] = next_id
            return DialogueResult(mode="scenario")
        else:  # "end"
            return DialogueResult(
                mode="scenario",
                response_text="죄송합니다. 다시 전화 부탁드립니다.",
                action={"type": "end", "disposition": "failed"},
            )

    async def _judge_yes_no(self, text: str) -> bool:
        """Simple yes/no judgment. LLM-based if available, keyword fallback."""
        positive = {"네", "예", "맞아요", "맞습니다", "그래요", "맞아", "응", "넵"}
        negative = {"아니요", "아니오", "아닙니다", "아니에요", "틀려요", "아니"}
        text_lower = text.strip()
        if any(w in text_lower for w in positive):
            return True
        if any(w in text_lower for w in negative):
            return False
        # Default to True for ambiguous
        return True

    async def _llm_condition(self, node: Node, state: dict, user_text: str) -> str:
        """Evaluate condition via LLM. Returns branch label string."""
        # Placeholder — returns "false" if no LLM
        return "false"

    def _error_result(self, state: dict, msg: str) -> DialogueResult:
        logger.error(msg)
        return DialogueResult(
            mode="scenario",
            response_text="시스템 오류가 발생했습니다.",
            action={"type": "end", "disposition": "error", "reason": msg},
        )
```

Write to: `realtime_demo/dialogue/graph_walker.py`

- [ ] **Step 4: Run tests**

Run: `cd /home/jonghooy/work/cachedSTT && python -m pytest realtime_demo/tests/test_graph_walker.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add realtime_demo/dialogue/graph_walker.py realtime_demo/tests/test_graph_walker.py
git commit -m "feat(dialogue): add graph walker with all 9 node types"
```

---

### Task 6: Action Runner

**Files:**
- Create: `realtime_demo/dialogue/action_runner.py`
- Create: `realtime_demo/tests/test_action_runner.py`

Executes actions returned by GraphWalker: API calls, transfers, end.

- [ ] **Step 1: Write failing tests**

```python
"""Test action runner."""
import pytest
from unittest.mock import AsyncMock, patch

from realtime_demo.dialogue.action_runner import ActionRunner


class TestApiCall:
    @pytest.mark.asyncio
    async def test_api_call_success(self):
        runner = ActionRunner()
        state = {"slots": {}, "variables": {}}
        action = {
            "type": "api_call",
            "method": "POST",
            "url": "https://example.com/api/block",
            "body": {"card": "1234"},
            "timeout_ms": 5000,
            "result_var": "block_result",
            "next_node": "n8",
            "on_error": "n_err",
        }
        with patch("httpx.AsyncClient.request", new_callable=AsyncMock) as mock_req:
            mock_resp = AsyncMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"success": True}
            mock_req.return_value = mock_resp
            result = await runner.execute_api_call(action, state)
        assert result["success"] is True
        assert state["variables"]["block_result"] == {"success": True}

    @pytest.mark.asyncio
    async def test_api_call_failure_returns_none(self):
        runner = ActionRunner()
        state = {"slots": {}, "variables": {}}
        action = {
            "type": "api_call",
            "method": "POST",
            "url": "https://example.com/api/block",
            "body": {},
            "timeout_ms": 100,
            "result_var": "result",
            "next_node": "n8",
            "on_error": "n_err",
        }
        with patch("httpx.AsyncClient.request", new_callable=AsyncMock, side_effect=Exception("timeout")):
            result = await runner.execute_api_call(action, state)
        assert result is None


class TestExecute:
    @pytest.mark.asyncio
    async def test_end_action(self):
        runner = ActionRunner()
        action = {"type": "end", "disposition": "resolved"}
        result = await runner.execute(action, state={})
        assert result["completed"] is True

    @pytest.mark.asyncio
    async def test_transfer_action(self):
        runner = ActionRunner()
        action = {"type": "transfer", "reason": "slot_fail", "slots": {"card": "1234"}}
        result = await runner.execute(action, state={})
        assert result["transfer"] is True
        assert result["reason"] == "slot_fail"
```

Write to: `realtime_demo/tests/test_action_runner.py`

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/jonghooy/work/cachedSTT && python -m pytest realtime_demo/tests/test_action_runner.py -v`
Expected: FAIL

- [ ] **Step 3: Implement action_runner.py**

```python
"""Action runner — executes API calls, transfers, and end actions."""
from __future__ import annotations

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
            # LLM response is handled by the engine, not action runner
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
                result = resp.json()
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
```

Write to: `realtime_demo/dialogue/action_runner.py`

- [ ] **Step 4: Run tests**

Run: `cd /home/jonghooy/work/cachedSTT && python -m pytest realtime_demo/tests/test_action_runner.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add realtime_demo/dialogue/action_runner.py realtime_demo/tests/test_action_runner.py
git commit -m "feat(dialogue): add action runner for API calls, transfers, and end"
```

---

### Task 7: Intent Matcher

**Files:**
- Create: `realtime_demo/dialogue/intent_matcher.py`
- Create: `realtime_demo/tests/test_intent_matcher.py`

Cosine similarity on cached trigger embeddings → top-3 → LLM final judgment.

- [ ] **Step 1: Write failing tests**

```python
"""Test intent matcher."""
import numpy as np
import pytest
from unittest.mock import AsyncMock

from realtime_demo.dialogue.models import Scenario, IntentMatch
from realtime_demo.dialogue.intent_matcher import IntentMatcher


def _make_scenarios():
    """Create two scenarios with pre-computed trigger embeddings."""
    # Scenario 1: card lost
    s1_triggers = ["카드를 잃어버렸어요", "카드 분실 신고", "카드가 없어졌어요"]
    s1_embeddings = [np.random.randn(16).astype(np.float32) for _ in s1_triggers]
    # Make them similar to each other
    base = np.random.randn(16).astype(np.float32)
    s1_embeddings = [base + np.random.randn(16).astype(np.float32) * 0.1 for _ in s1_triggers]

    # Scenario 2: limit change
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
        matcher = IntentMatcher(embed_fn=None, llm_engine=None)
        scenarios = _make_scenarios()
        matcher.load_trigger_cache(scenarios)
        assert len(matcher._trigger_cache) == 5  # 3 + 2

    @pytest.mark.asyncio
    async def test_no_match_below_threshold(self):
        async def mock_embed(text):
            return np.random.randn(16).astype(np.float32) * 100  # very different

        matcher = IntentMatcher(embed_fn=mock_embed, llm_engine=None)
        matcher.load_trigger_cache(_make_scenarios())
        result = await matcher.match("오늘 날씨가 좋네요")
        assert result is None or result.confidence < 0.5
```

Write to: `realtime_demo/tests/test_intent_matcher.py`

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/jonghooy/work/cachedSTT && python -m pytest realtime_demo/tests/test_intent_matcher.py -v`
Expected: FAIL

- [ ] **Step 3: Implement intent_matcher.py**

```python
"""Intent matcher — cosine similarity on trigger embeddings + optional LLM judgment."""
from __future__ import annotations

import logging
from typing import Callable, Awaitable

import numpy as np

from realtime_demo.dialogue.models import IntentMatch, Scenario

logger = logging.getLogger(__name__)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class IntentMatcher:
    """Matches user utterances to scenarios via embedding similarity."""

    def __init__(
        self,
        embed_fn: Callable[[str], Awaitable[np.ndarray]] | None = None,
        llm_engine=None,
        similarity_threshold: float = 0.5,
        confidence_threshold: float = 0.7,
    ):
        self.embed_fn = embed_fn
        self.llm_engine = llm_engine
        self.similarity_threshold = similarity_threshold
        self.confidence_threshold = confidence_threshold
        self._trigger_cache: list[dict] = []  # [{scenario_id, text, embedding}, ...]
        self._scenarios: dict[str, Scenario] = {}

    def load_trigger_cache(self, trigger_data: list[dict]):
        """Load pre-computed trigger embeddings.

        Args:
            trigger_data: [{"scenario_id": str, "triggers": [str], "embeddings": [np.array]}, ...]
        """
        self._trigger_cache = []
        for entry in trigger_data:
            sid = entry["scenario_id"]
            for text, emb in zip(entry["triggers"], entry["embeddings"]):
                self._trigger_cache.append({
                    "scenario_id": sid,
                    "text": text,
                    "embedding": np.asarray(emb, dtype=np.float32),
                })

    def set_scenarios(self, scenarios: dict[str, Scenario]):
        """Set scenario lookup dict."""
        self._scenarios = scenarios

    async def match(self, text: str, top_k: int = 3) -> IntentMatch | None:
        """Match user text against cached triggers.

        Returns:
            IntentMatch if confidence > threshold, else None
        """
        if not self._trigger_cache or not self.embed_fn:
            return None

        # Embed user text
        try:
            user_embedding = await self.embed_fn(text)
        except Exception:
            logger.exception("Embedding failed")
            return None

        # Compute similarities
        scores = []
        for entry in self._trigger_cache:
            sim = cosine_similarity(user_embedding, entry["embedding"])
            scores.append((sim, entry))

        # Sort by similarity descending
        scores.sort(key=lambda x: x[0], reverse=True)
        top = scores[:top_k]

        if not top or top[0][0] < self.similarity_threshold:
            return None

        # Best match
        best_sim, best_entry = top[0]
        scenario_id = best_entry["scenario_id"]
        scenario = self._scenarios.get(scenario_id)

        if scenario is None:
            return None

        return IntentMatch(
            scenario_id=scenario_id,
            scenario=scenario,
            confidence=best_sim,
            matched_example=best_entry["text"],
        )
```

Write to: `realtime_demo/dialogue/intent_matcher.py`

- [ ] **Step 4: Run tests**

Run: `cd /home/jonghooy/work/cachedSTT && python -m pytest realtime_demo/tests/test_intent_matcher.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add realtime_demo/dialogue/intent_matcher.py realtime_demo/tests/test_intent_matcher.py
git commit -m "feat(dialogue): add intent matcher with cosine similarity"
```

---

### Task 8: Scenario Cache

**Files:**
- Create: `realtime_demo/dialogue/scenario_cache.py`
- Create: `realtime_demo/tests/test_scenario_cache.py`

Loads scenarios from Knowledge Service and caches them in memory.

- [ ] **Step 1: Write failing tests**

```python
"""Test scenario cache."""
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from realtime_demo.dialogue.scenario_cache import ScenarioCache
from realtime_demo.dialogue.models import Scenario

FIXTURE_DIR = Path(__file__).parent / "fixtures"


class TestScenarioCache:
    @pytest.mark.asyncio
    async def test_load_from_fixture(self):
        """Test loading scenarios from a list of dicts (simulating API response)."""
        with open(FIXTURE_DIR / "card_lost_scenario.json") as f:
            data = json.load(f)

        cache = ScenarioCache(knowledge_base_url="http://localhost:8100/api/brain")
        cache.load_from_dicts([data])
        assert "card-lost" in cache.scenarios
        assert isinstance(cache.scenarios["card-lost"], Scenario)

    @pytest.mark.asyncio
    async def test_get_active_scenarios(self):
        with open(FIXTURE_DIR / "card_lost_scenario.json") as f:
            data = json.load(f)
        cache = ScenarioCache(knowledge_base_url="http://localhost:8100/api/brain")
        cache.load_from_dicts([data])
        active = cache.get_active_scenarios()
        assert len(active) == 1
        assert active[0].id == "card-lost"

    @pytest.mark.asyncio
    async def test_empty_cache(self):
        cache = ScenarioCache(knowledge_base_url="http://localhost:8100/api/brain")
        assert cache.get_active_scenarios() == []
        assert cache.get_scenario("nonexistent") is None
```

Write to: `realtime_demo/tests/test_scenario_cache.py`

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/jonghooy/work/cachedSTT && python -m pytest realtime_demo/tests/test_scenario_cache.py -v`
Expected: FAIL

- [ ] **Step 3: Implement scenario_cache.py**

```python
"""Scenario cache — loads and caches scenarios from Knowledge Service."""
from __future__ import annotations

import logging
from typing import Any

import httpx

from realtime_demo.dialogue.models import Scenario

logger = logging.getLogger(__name__)


class ScenarioCache:
    """Loads active scenarios from Knowledge Service and caches in memory."""

    def __init__(self, knowledge_base_url: str = "http://localhost:8100/api/brain"):
        self.knowledge_base_url = knowledge_base_url
        self.scenarios: dict[str, Scenario] = {}
        self._version: str | None = None

    def load_from_dicts(self, scenario_dicts: list[dict]):
        """Load scenarios from a list of raw dicts (API response or fixtures)."""
        self.scenarios.clear()
        for data in scenario_dicts:
            try:
                scenario = Scenario.from_dict(data)
                if scenario.status == "active":
                    self.scenarios[scenario.id] = scenario
            except Exception:
                logger.exception(f"Failed to parse scenario: {data.get('id', 'unknown')}")

    async def refresh(self) -> bool:
        """Fetch active scenarios from Knowledge Service.

        Calls GET /api/brain/scenarios and updates local cache.

        Returns:
            bool: True on success
        """
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
        """Get trigger data for intent matcher (without embeddings — those come from Knowledge)."""
        result = []
        for s in self.scenarios.values():
            examples = s.triggers.get("examples", [])
            result.append({
                "scenario_id": s.id,
                "triggers": examples,
                "description": s.triggers.get("description", ""),
            })
        return result
```

Write to: `realtime_demo/dialogue/scenario_cache.py`

- [ ] **Step 4: Run tests**

Run: `cd /home/jonghooy/work/cachedSTT && python -m pytest realtime_demo/tests/test_scenario_cache.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add realtime_demo/dialogue/scenario_cache.py realtime_demo/tests/test_scenario_cache.py
git commit -m "feat(dialogue): add scenario cache for Knowledge Service integration"
```

---

### Task 9: Dialogue Engine

**Files:**
- Create: `realtime_demo/dialogue/engine.py`
- Create: `realtime_demo/tests/test_dialogue_engine.py`

Main orchestrator that ties everything together.

- [ ] **Step 1: Write failing tests**

```python
"""Test dialogue engine."""
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from realtime_demo.dialogue.engine import DialogueEngine
from realtime_demo.dialogue.models import Scenario, DialogueResult, IntentMatch
from realtime_demo.dialogue.scenario_cache import ScenarioCache

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _make_session_state():
    return {
        "dialogue_mode": "freeform",
        "scenario_state": {
            "scenario_id": None,
            "scenario_version": None,
            "current_node": None,
            "slots": {},
            "variables": {},
            "history": [],
            "retry_count": 0,
            "prosody": None,
            "stack": [],
        },
    }


def _load_cache():
    with open(FIXTURE_DIR / "card_lost_scenario.json") as f:
        data = json.load(f)
    cache = ScenarioCache()
    cache.load_from_dicts([data])
    return cache


class TestFreeformMode:
    @pytest.mark.asyncio
    async def test_no_match_returns_freeform(self):
        cache = _load_cache()
        mock_matcher = AsyncMock()
        mock_matcher.match = AsyncMock(return_value=None)
        engine = DialogueEngine(
            scenario_cache=cache,
            intent_matcher=mock_matcher,
            slot_manager=MagicMock(),
            llm_engine=None,
        )
        session = _make_session_state()
        result = await engine.process_utterance(session, "오늘 날씨 좋네요", None)
        assert result.should_use_s2s is True
        assert session["dialogue_mode"] == "freeform"

    @pytest.mark.asyncio
    async def test_match_enters_scenario(self):
        cache = _load_cache()
        scenario = cache.get_scenario("card-lost")
        mock_matcher = AsyncMock()
        mock_matcher.match = AsyncMock(return_value=IntentMatch(
            scenario_id="card-lost", scenario=scenario, confidence=0.9, matched_example="카드 분실"
        ))
        engine = DialogueEngine(
            scenario_cache=cache,
            intent_matcher=mock_matcher,
            slot_manager=MagicMock(),
            llm_engine=None,
        )
        session = _make_session_state()
        result = await engine.process_utterance(session, "카드 잃어버렸어요", None)
        assert session["dialogue_mode"] == "scenario"
        assert session["scenario_state"]["scenario_id"] == "card-lost"
        assert "카드 분실 신고" in result.response_text


class TestScenarioMode:
    @pytest.mark.asyncio
    async def test_slot_collect_flow(self):
        cache = _load_cache()
        scenario = cache.get_scenario("card-lost")
        mock_sm = MagicMock()
        mock_sm.extract = AsyncMock(return_value="1234-5678-9012-3456")
        engine = DialogueEngine(
            scenario_cache=cache,
            intent_matcher=AsyncMock(),
            slot_manager=mock_sm,
            llm_engine=None,
        )
        session = _make_session_state()
        session["dialogue_mode"] = "scenario"
        session["scenario_state"]["scenario_id"] = "card-lost"
        session["scenario_state"]["scenario_version"] = 1
        session["scenario_state"]["current_node"] = "n2"
        session["scenario_state"]["_scenario_snapshot"] = scenario

        result = await engine.process_utterance(session, "1234-5678-9012-3456", None)
        assert session["scenario_state"]["slots"].get("card_number") == "1234-5678-9012-3456"


class TestEndReturnsToFreeform:
    @pytest.mark.asyncio
    async def test_end_node_exits(self):
        cache = _load_cache()
        scenario = cache.get_scenario("card-lost")
        engine = DialogueEngine(
            scenario_cache=cache,
            intent_matcher=AsyncMock(),
            slot_manager=MagicMock(),
            llm_engine=None,
        )
        session = _make_session_state()
        session["dialogue_mode"] = "scenario"
        session["scenario_state"]["scenario_id"] = "card-lost"
        session["scenario_state"]["current_node"] = "n8"
        session["scenario_state"]["_scenario_snapshot"] = scenario

        result = await engine.process_utterance(session, "", None)
        assert result.action is not None
        assert result.action["type"] == "end"
        assert session["dialogue_mode"] == "freeform"
```

Write to: `realtime_demo/tests/test_dialogue_engine.py`

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/jonghooy/work/cachedSTT && python -m pytest realtime_demo/tests/test_dialogue_engine.py -v`
Expected: FAIL

- [ ] **Step 3: Implement engine.py**

```python
"""Dialogue Engine — orchestrates scenario matching, graph walking, and freeform fallback."""
from __future__ import annotations

import copy
import logging
from typing import Any

from realtime_demo.dialogue.models import DialogueResult, Scenario
from realtime_demo.dialogue.graph_walker import GraphWalker
from realtime_demo.dialogue.scenario_cache import ScenarioCache

logger = logging.getLogger(__name__)


class DialogueEngine:
    """Main dialogue engine. Inserted between Turn Detection and S2S pipeline."""

    def __init__(
        self,
        scenario_cache: ScenarioCache,
        intent_matcher,
        slot_manager,
        llm_engine=None,
        action_runner=None,
    ):
        self.scenario_cache = scenario_cache
        self.intent_matcher = intent_matcher
        self.slot_manager = slot_manager
        self.llm_engine = llm_engine
        self.action_runner = action_runner

    async def process_utterance(
        self,
        session: dict,
        text: str,
        prosody: dict | None,
    ) -> DialogueResult:
        """Main entry point. Called from server.py after endpoint detection.

        Args:
            session: dict with "dialogue_mode" and "scenario_state" keys
            text: final STT text
            prosody: audio context (energy, speech_rate, energy_trend)

        Returns:
            DialogueResult
        """
        mode = session.get("dialogue_mode", "freeform")

        if mode == "freeform":
            return await self._handle_freeform(session, text, prosody)
        elif mode == "scenario":
            return await self._handle_scenario(session, text, prosody)
        else:
            return DialogueResult(should_use_s2s=True)

    async def _handle_freeform(
        self, session: dict, text: str, prosody: dict | None
    ) -> DialogueResult:
        """Try intent matching. If matched, enter scenario. Otherwise, freeform."""
        try:
            match = await self.intent_matcher.match(text)
        except Exception:
            logger.exception("Intent matching failed")
            match = None

        if match and match.confidence >= 0.7:
            self._enter_scenario(session, match.scenario)
            session["scenario_state"]["prosody"] = prosody
            return await self._execute_scenario_step(session, text)
        else:
            return DialogueResult(mode="freeform", should_use_s2s=True)

    async def _handle_scenario(
        self, session: dict, text: str, prosody: dict | None
    ) -> DialogueResult:
        """Continue executing current scenario."""
        session["scenario_state"]["prosody"] = prosody
        result = await self._execute_scenario_step(session, text)

        # Check if scenario ended
        if result.action and result.action.get("type") == "end":
            self._exit_scenario(session)
        elif result.action and result.action.get("type") == "transfer":
            self._exit_scenario(session)

        return result

    async def _execute_scenario_step(self, session: dict, text: str) -> DialogueResult:
        """Execute one step of the current scenario graph."""
        state = session["scenario_state"]
        scenario = state.get("_scenario_snapshot")

        if scenario is None:
            logger.error("No scenario snapshot in session")
            self._exit_scenario(session)
            return DialogueResult(should_use_s2s=True)

        walker = GraphWalker(scenario, self.slot_manager, self.llm_engine)
        return await walker.step(state, text)

    def _enter_scenario(self, session: dict, scenario: Scenario):
        """Enter a scenario: set mode, copy scenario snapshot, initialize state."""
        session["dialogue_mode"] = "scenario"
        state = session["scenario_state"]
        state["scenario_id"] = scenario.id
        state["scenario_version"] = scenario.version
        state["current_node"] = scenario.get_start_node_id()
        state["slots"] = {}
        state["variables"] = {}
        state["history"] = []
        state["retry_count"] = 0
        state["stack"] = []
        # Deep copy scenario to isolate from cache refreshes
        state["_scenario_snapshot"] = copy.deepcopy(scenario)

    def _exit_scenario(self, session: dict):
        """Exit scenario: return to freeform mode."""
        state = session["scenario_state"]

        # Check if there's a stacked scenario to resume
        if state.get("stack"):
            parent = state["stack"].pop()
            session["scenario_state"] = parent
            session["dialogue_mode"] = "scenario"
            return

        session["dialogue_mode"] = "freeform"
        state["scenario_id"] = None
        state["scenario_version"] = None
        state["current_node"] = None
        state["slots"] = {}
        state["variables"] = {}
        state["history"] = []
        state["retry_count"] = 0
        state["_scenario_snapshot"] = None
```

Write to: `realtime_demo/dialogue/engine.py`

- [ ] **Step 4: Update `__init__.py`**

```python
"""Dialogue engine for scenario-based conversation management."""
from realtime_demo.dialogue.models import DialogueResult, Scenario, IntentMatch
from realtime_demo.dialogue.engine import DialogueEngine
```

Update: `realtime_demo/dialogue/__init__.py`

- [ ] **Step 5: Run tests**

Run: `cd /home/jonghooy/work/cachedSTT && python -m pytest realtime_demo/tests/test_dialogue_engine.py -v`
Expected: All 4 tests PASS

- [ ] **Step 6: Run all tests**

Run: `cd /home/jonghooy/work/cachedSTT && python -m pytest realtime_demo/tests/ -v --ignore=realtime_demo/tests/test_turn_detector.py`
Expected: All tests across all test files PASS (skip turn_detector tests which need Kiwi)

- [ ] **Step 7: Commit**

```bash
git add realtime_demo/dialogue/engine.py realtime_demo/dialogue/__init__.py realtime_demo/tests/test_dialogue_engine.py
git commit -m "feat(dialogue): add DialogueEngine orchestrator with freeform/scenario mode switching"
```

---

### Task 10: server.py Integration

**Files:**
- Modify: `realtime_demo/server.py`

Minimal changes: extend StreamingSession, integrate DialogueEngine at the endpoint dispatch point.

- [ ] **Step 1: Add dialogue fields to StreamingSession.__init__**

In `realtime_demo/server.py`, add to `StreamingSession.__init__()` after the existing fields (after `self.conversation_history`):

```python
        # 시나리오 대화 엔진
        self.dialogue_mode: str = "freeform"
        self.scenario_state: dict = {
            "scenario_id": None,
            "scenario_version": None,
            "current_node": None,
            "slots": {},
            "variables": {},
            "history": [],
            "retry_count": 0,
            "prosody": None,
            "stack": [],
            "_scenario_snapshot": None,
        }
```

- [ ] **Step 2: Add a helper to build session dict for DialogueEngine**

Add method to StreamingSession:

```python
    def get_dialogue_session(self) -> dict:
        """Return dict compatible with DialogueEngine.process_utterance."""
        return {
            "dialogue_mode": self.dialogue_mode,
            "scenario_state": self.scenario_state,
        }

    def sync_dialogue_session(self, session_dict: dict):
        """Write back dialogue state after processing."""
        self.dialogue_mode = session_dict["dialogue_mode"]
        self.scenario_state = session_dict["scenario_state"]
```

- [ ] **Step 3: Initialize DialogueEngine in startup**

In the `startup` event (around line 280), after KnowledgeClient init, add:

```python
    # Dialogue Engine (시나리오 기반 대화)
    global dialogue_engine
    from realtime_demo.dialogue.engine import DialogueEngine
    from realtime_demo.dialogue.scenario_cache import ScenarioCache
    from realtime_demo.dialogue.slot_manager import SlotManager
    from realtime_demo.dialogue.intent_matcher import IntentMatcher
    from realtime_demo.dialogue.action_runner import ActionRunner

    scenario_cache = ScenarioCache()
    # Try loading scenarios (non-blocking — empty cache is OK)
    try:
        await scenario_cache.refresh()
    except Exception:
        logging.warning("Could not load scenarios from Knowledge Service — running in freeform-only mode")

    slot_manager = SlotManager(llm_engine=s2s_pipeline.llm if s2s_pipeline else None)
    intent_matcher = IntentMatcher(embed_fn=None)  # embed_fn set when Knowledge provides embeddings
    intent_matcher.set_scenarios(scenario_cache.scenarios)
    action_runner = ActionRunner(knowledge_client=knowledge_client)

    dialogue_engine = DialogueEngine(
        scenario_cache=scenario_cache,
        intent_matcher=intent_matcher,
        slot_manager=slot_manager,
        llm_engine=s2s_pipeline.llm if s2s_pipeline else None,
        action_runner=action_runner,
    )
    logging.info(f"DialogueEngine initialized with {len(scenario_cache.scenarios)} scenarios")
```

Add `dialogue_engine = None` to the globals section (around line 70).

- [ ] **Step 4: Integrate at endpoint dispatch**

In the endpoint dispatch section (around line 779), BEFORE the S2S dispatch, add:

```python
        # ── Dialogue Engine: try scenario matching first ──
        # Insert BEFORE the `if S2S_ENABLED` block, AFTER noise filter
        if dialogue_engine:
            ds = session.get_dialogue_session()
            _prosody = _compute_prosody(session, session.last_text, buffer_seconds)
            d_result = await dialogue_engine.process_utterance(
                ds, session.last_text, _prosody
            )
            session.sync_dialogue_session(ds)

            if not d_result.should_use_s2s:
                # Scenario handled it — send response via TTS
                if d_result.response_text:
                    await websocket.send_json({
                        "type": "scenario_response",
                        "text": d_result.response_text,
                        "mode": d_result.mode,
                        "action": d_result.action,
                        "utterance_id": session.utterance_count,
                    })
                    # TODO: TTS synthesis for scenario responses (Plan 2)
                # Skip S2S pipeline
                session.reset_for_new_utterance(keep_tail_seconds=0.5)
                continue
        # ── End Dialogue Engine ──
```

- [ ] **Step 5: Extend /api/knowledge/refresh to reload scenarios**

Update the existing endpoint:

```python
@app.post("/api/knowledge/refresh")
async def knowledge_refresh(body: dict = {}):
    """Knowledge Service 변경 시 캐시 갱신."""
    if knowledge_client:
        success = await knowledge_client.load_config()
        if success and s2s_pipeline:
            s2s_pipeline.knowledge_client = knowledge_client
    # Refresh scenarios
    if dialogue_engine and dialogue_engine.scenario_cache:
        await dialogue_engine.scenario_cache.refresh()
        dialogue_engine.intent_matcher.set_scenarios(
            dialogue_engine.scenario_cache.scenarios
        )
        return {"status": "refreshed", "scenarios": len(dialogue_engine.scenario_cache.scenarios)}
    return {"status": "refreshed"}
```

- [ ] **Step 6: Manual test**

Start the server and verify it doesn't crash:

```bash
source /home/jonghooy/miniconda3/etc/profile.d/conda.sh && conda activate nemo-asr && \
cd /home/jonghooy/work/cachedSTT/realtime_demo && \
timeout 10 python server.py --port 3001 2>&1 | head -20 || true
```

Expected: Server starts without import errors. "DialogueEngine initialized with 0 scenarios" in output (Knowledge Service not running is OK).

- [ ] **Step 7: Commit**

```bash
git add realtime_demo/server.py
git commit -m "feat(dialogue): integrate DialogueEngine into server.py with graceful fallback"
```

---

### Task 11: Integration Test

**Files:**
- Create: `realtime_demo/tests/test_dialogue_integration.py`

End-to-end test: freeform → scenario entry → slot collection → condition → end → freeform.

- [ ] **Step 1: Write integration test**

```python
"""Integration test: full scenario walkthrough without external services."""
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from realtime_demo.dialogue.engine import DialogueEngine
from realtime_demo.dialogue.scenario_cache import ScenarioCache
from realtime_demo.dialogue.slot_manager import SlotManager
from realtime_demo.dialogue.intent_matcher import IntentMatcher
from realtime_demo.dialogue.action_runner import ActionRunner
from realtime_demo.dialogue.models import Scenario, IntentMatch

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _setup():
    """Create engine with fixture scenario and mocked intent matcher."""
    with open(FIXTURE_DIR / "card_lost_scenario.json") as f:
        data = json.load(f)
    cache = ScenarioCache()
    cache.load_from_dicts([data])

    scenario = cache.get_scenario("card-lost")

    # Mock intent matcher that always matches card-lost
    matcher = AsyncMock()
    matcher.match = AsyncMock(return_value=IntentMatch(
        scenario_id="card-lost", scenario=scenario, confidence=0.95
    ))
    matcher.set_scenarios = MagicMock()

    # Real slot manager with mocked LLM
    mock_llm = AsyncMock()
    sm = SlotManager(llm_engine=mock_llm)

    engine = DialogueEngine(
        scenario_cache=cache,
        intent_matcher=matcher,
        slot_manager=sm,
        llm_engine=mock_llm,
        action_runner=ActionRunner(),
    )

    session = {
        "dialogue_mode": "freeform",
        "scenario_state": {
            "scenario_id": None,
            "scenario_version": None,
            "current_node": None,
            "slots": {},
            "variables": {},
            "history": [],
            "retry_count": 0,
            "prosody": None,
            "stack": [],
            "_scenario_snapshot": None,
        },
    }
    return engine, session, mock_llm


class TestFullScenarioWalkthrough:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        """Complete scenario: entry → card_number → loss_type → confirm → condition → end."""
        engine, session, mock_llm = _setup()

        # Turn 1: "카드 잃어버렸어요" → enters scenario, gets greeting + asks card number
        r1 = await engine.process_utterance(session, "카드 잃어버렸어요", None)
        assert session["dialogue_mode"] == "scenario"
        assert r1.response_text is not None
        assert "카드 분실 신고" in r1.response_text
        assert r1.awaiting_input is True

        # Turn 2: Provide card number (regex extraction)
        r2 = await engine.process_utterance(session, "1234 5678 9012 3456입니다", None)
        assert session["scenario_state"]["slots"].get("card_number") is not None
        # Now asking for loss_type

        # Turn 3: Provide loss type (LLM extraction)
        async def mock_stream(*args, **kwargs):
            yield "분실"
        mock_llm.generate_stream = AsyncMock(side_effect=lambda *a, **kw: mock_stream())

        r3 = await engine.process_utterance(session, "분실이에요", None)
        assert session["scenario_state"]["slots"].get("loss_type") == "분실"

        # Turn 4: Confirm → yes
        r4 = await engine.process_utterance(session, "", None)  # confirm asks question
        assert r4.awaiting_input is True
        assert "맞으시죠" in r4.response_text

        # Turn 5: Yes → condition → api_call action
        r5 = await engine.process_utterance(session, "네 맞아요", None)
        # After confirm(yes) → condition(분실=false) → api_call action
        assert r5.action is not None

    @pytest.mark.asyncio
    async def test_no_match_stays_freeform(self):
        engine, session, _ = _setup()
        engine.intent_matcher.match = AsyncMock(return_value=None)
        result = await engine.process_utterance(session, "오늘 날씨가 좋네요", None)
        assert result.should_use_s2s is True
        assert session["dialogue_mode"] == "freeform"
```

Write to: `realtime_demo/tests/test_dialogue_integration.py`

- [ ] **Step 2: Run integration test**

Run: `cd /home/jonghooy/work/cachedSTT && python -m pytest realtime_demo/tests/test_dialogue_integration.py -v`
Expected: All tests PASS

- [ ] **Step 3: Run full test suite**

Run: `cd /home/jonghooy/work/cachedSTT && python -m pytest realtime_demo/tests/ -v --ignore=realtime_demo/tests/test_turn_detector.py`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add realtime_demo/tests/test_dialogue_integration.py
git commit -m "test(dialogue): add full scenario walkthrough integration test"
```

---

## Summary

| Task | Component | Files | Tests |
|------|-----------|-------|-------|
| 1 | Data models + fixture | models.py, __init__.py, fixture.json | test_models.py (7) |
| 2 | Rule evaluator | rule_evaluator.py | test_rule_evaluator.py (13) |
| 3 | Template renderer | template_renderer.py | test_template_renderer.py (6) |
| 4 | Slot manager | slot_manager.py | test_slot_manager.py (5) |
| 5 | Graph walker | graph_walker.py | test_graph_walker.py (9) |
| 6 | Action runner | action_runner.py | test_action_runner.py (4) |
| 7 | Intent matcher | intent_matcher.py | test_intent_matcher.py (4) |
| 8 | Scenario cache | scenario_cache.py | test_scenario_cache.py (3) |
| 9 | Dialogue engine | engine.py | test_dialogue_engine.py (4) |
| 10 | server.py integration | server.py (modify) | manual test |
| 11 | Integration test | — | test_dialogue_integration.py (2) |
| **Total** | **11 tasks** | **11 new + 1 modified** | **~57 tests** |

## Next Plans

- **Plan 2:** Knowledge Service Backend — scenarios DB table, CRUD API, Brain API endpoints, AI generation pipeline
- **Plan 3:** Visual Editor Frontend — Vue Flow graph editor, AI chat panel, scenario list
