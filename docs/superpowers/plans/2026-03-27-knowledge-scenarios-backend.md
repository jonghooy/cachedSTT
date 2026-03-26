# Knowledge Service Scenarios Backend Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add scenario CRUD, validation, and Brain-facing API to Knowledge Service so Brain can load and execute scenarios, and scenarios can be managed via REST API.

**Architecture:** Extends existing Knowledge Service (FastAPI + SQLite + ChromaDB) with a `scenarios` table, a `scenario_logs` table, a `trigger_examples` ChromaDB collection, CRUD endpoints, structure validation, and Brain-facing endpoints. Follows existing patterns exactly (aiosqlite async CRUD, Pydantic models, APIRouter, monkeypatch test fixtures).

**Tech Stack:** Python 3.10+, FastAPI, aiosqlite, chromadb, pytest, pytest-asyncio, httpx (TestClient)

**Spec:** `docs/superpowers/specs/2026-03-27-scenario-builder-design.md`

**Scope:** Knowledge Service backend only (Plan 2 of 3). AI generation pipeline (Claude API) and Vue Flow frontend are Plan 3+. This plan delivers working CRUD + Brain API with structure validation.

**Working directory:** `/home/jonghooy/work/knowledge-service`

**Prerequisites:** Knowledge Service codebase at `/home/jonghooy/work/knowledge-service`. Existing pytest.ini has `asyncio_mode = auto`.

---

## File Structure

```
backend/
├── db/
│   ├── sqlite.py              # MODIFY: add scenarios + scenario_logs tables + CRUD functions
│   └── chroma.py              # MODIFY: add trigger collection helpers
├── api/
│   ├── scenarios.py           # NEW: /api/scenarios CRUD + deploy + archive
│   └── brain.py               # MODIFY: add GET /scenarios, POST /log-execution
├── scenarios/
│   ├── __init__.py            # NEW
│   └── validator.py           # NEW: scenario structure validation
├── main.py                    # MODIFY: register scenarios router
└── tests/
    ├── test_scenarios_db.py   # NEW: async DB CRUD tests
    └── test_scenarios_api.py  # NEW: API integration tests
```

---

### Task 1: Scenarios DB Schema + CRUD

**Files:**
- Modify: `backend/db/sqlite.py`
- Create: `backend/tests/test_scenarios_db.py`

- [ ] **Step 1: Write failing tests**

```python
"""Test scenarios DB CRUD."""
import json
import pytest
import pytest_asyncio

from db.sqlite import (
    init_db,
    add_scenario, list_scenarios, get_scenario, update_scenario, delete_scenario,
    update_scenario_status, add_scenario_log, list_scenario_logs,
)


@pytest_asyncio.fixture
async def db_path(tmp_path):
    path = str(tmp_path / "test.db")
    await init_db(path)
    return path


_SAMPLE_SCENARIO = {
    "name": "카드 분실 신고",
    "description": "카드 분실/도난 시 정지 처리",
    "graph_json": json.dumps({"nodes": [], "edges": []}),
    "status": "draft",
    "priority": 10,
    "created_by": "ai",
    "source_json": json.dumps({"type": "natural_language", "input": "test"}),
    "triggers_json": json.dumps({"examples": ["카드 잃어버렸어요"], "description": "카드 분실"}),
    "slots_json": json.dumps({}),
    "metadata_json": json.dumps({}),
}


@pytest.mark.asyncio
async def test_add_and_get_scenario(db_path):
    sid = await add_scenario(db_path, **_SAMPLE_SCENARIO)
    assert isinstance(sid, int)
    s = await get_scenario(db_path, sid)
    assert s is not None
    assert s["name"] == "카드 분실 신고"
    assert s["status"] == "draft"
    assert s["version"] == 1
    assert s["schema_version"] == 1


@pytest.mark.asyncio
async def test_list_scenarios(db_path):
    await add_scenario(db_path, **_SAMPLE_SCENARIO)
    await add_scenario(db_path, **{**_SAMPLE_SCENARIO, "name": "한도 변경"})
    items = await list_scenarios(db_path)
    assert len(items) == 2


@pytest.mark.asyncio
async def test_list_scenarios_filter_status(db_path):
    sid = await add_scenario(db_path, **_SAMPLE_SCENARIO)
    await update_scenario_status(db_path, sid, "active")
    drafts = await list_scenarios(db_path, status="draft")
    actives = await list_scenarios(db_path, status="active")
    assert len(drafts) == 0
    assert len(actives) == 1


@pytest.mark.asyncio
async def test_update_scenario(db_path):
    sid = await add_scenario(db_path, **_SAMPLE_SCENARIO)
    await update_scenario(db_path, sid, name="Updated", graph_json=json.dumps({"nodes": [1]}))
    s = await get_scenario(db_path, sid)
    assert s["name"] == "Updated"
    assert s["version"] == 2


@pytest.mark.asyncio
async def test_delete_scenario(db_path):
    sid = await add_scenario(db_path, **_SAMPLE_SCENARIO)
    await delete_scenario(db_path, sid)
    s = await get_scenario(db_path, sid)
    assert s is None


@pytest.mark.asyncio
async def test_update_status_deploy(db_path):
    sid = await add_scenario(db_path, **_SAMPLE_SCENARIO)
    await update_scenario_status(db_path, sid, "active")
    s = await get_scenario(db_path, sid)
    assert s["status"] == "active"


@pytest.mark.asyncio
async def test_scenario_logs(db_path):
    sid = await add_scenario(db_path, **_SAMPLE_SCENARIO)
    await add_scenario_log(db_path, scenario_id=sid, session_id="ws_123",
                           completed=True, nodes_visited="n1,n2,n3",
                           slots_filled=json.dumps({"card": "1234"}),
                           duration_sec=45.0, exit_reason="end_node")
    logs = await list_scenario_logs(db_path, scenario_id=sid)
    assert len(logs) == 1
    assert logs[0]["completed"] == 1
    assert logs[0]["exit_reason"] == "end_node"
```

Write to: `backend/tests/test_scenarios_db.py`

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/jonghooy/work/knowledge-service && python -m pytest backend/tests/test_scenarios_db.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Add scenarios + scenario_logs tables to init_db**

In `backend/db/sqlite.py`, add to the `init_db` function (after existing CREATE TABLE statements):

```python
        await db.execute("""
            CREATE TABLE IF NOT EXISTS scenarios (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                schema_version INTEGER DEFAULT 1,
                graph_json TEXT DEFAULT '{}',
                version INTEGER DEFAULT 1,
                status TEXT DEFAULT 'draft',
                priority INTEGER DEFAULT 0,
                created_by TEXT DEFAULT 'human',
                source_json TEXT DEFAULT '{}',
                triggers_json TEXT DEFAULT '{}',
                slots_json TEXT DEFAULT '{}',
                metadata_json TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS scenario_logs (
                id INTEGER PRIMARY KEY,
                scenario_id INTEGER NOT NULL,
                session_id TEXT NOT NULL,
                completed INTEGER DEFAULT 0,
                nodes_visited TEXT DEFAULT '',
                slots_filled TEXT DEFAULT '{}',
                duration_sec REAL DEFAULT 0,
                exit_reason TEXT DEFAULT '',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (scenario_id) REFERENCES scenarios(id)
            )
        """)
```

- [ ] **Step 4: Add scenario CRUD functions**

Append to `backend/db/sqlite.py`:

```python
# ── Scenarios ──

async def add_scenario(db_path: str, name: str, description: str = "",
                       graph_json: str = "{}", status: str = "draft",
                       priority: int = 0, created_by: str = "human",
                       source_json: str = "{}", triggers_json: str = "{}",
                       slots_json: str = "{}", metadata_json: str = "{}") -> int:
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            """INSERT INTO scenarios (name, description, graph_json, status, priority,
               created_by, source_json, triggers_json, slots_json, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (name, description, graph_json, status, priority,
             created_by, source_json, triggers_json, slots_json, metadata_json),
        )
        await db.commit()
        return cursor.lastrowid


async def list_scenarios(db_path: str, status: str | None = None) -> list[dict[str, Any]]:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        if status:
            cursor = await db.execute("SELECT * FROM scenarios WHERE status = ? ORDER BY priority DESC, updated_at DESC", (status,))
        else:
            cursor = await db.execute("SELECT * FROM scenarios ORDER BY priority DESC, updated_at DESC")
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def get_scenario(db_path: str, scenario_id: int) -> dict[str, Any] | None:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM scenarios WHERE id = ?", (scenario_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None


async def update_scenario(db_path: str, scenario_id: int, **kwargs) -> None:
    allowed = {"name", "description", "graph_json", "priority", "source_json",
               "triggers_json", "slots_json", "metadata_json"}
    fields = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
    if not fields:
        return
    fields["updated_at"] = "CURRENT_TIMESTAMP"
    set_clause = ", ".join(f"{k} = ?" if k != "updated_at" else f"{k} = CURRENT_TIMESTAMP" for k in fields)
    values = [v for k, v in fields.items() if k != "updated_at"]
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            f"UPDATE scenarios SET {set_clause}, version = version + 1 WHERE id = ?",
            (*values, scenario_id),
        )
        await db.commit()


async def update_scenario_status(db_path: str, scenario_id: int, status: str) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE scenarios SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (status, scenario_id),
        )
        await db.commit()


async def delete_scenario(db_path: str, scenario_id: int) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute("DELETE FROM scenarios WHERE id = ?", (scenario_id,))
        await db.execute("DELETE FROM scenario_logs WHERE scenario_id = ?", (scenario_id,))
        await db.commit()


# ── Scenario Logs ──

async def add_scenario_log(db_path: str, scenario_id: int, session_id: str,
                           completed: bool = False, nodes_visited: str = "",
                           slots_filled: str = "{}", duration_sec: float = 0,
                           exit_reason: str = "") -> int:
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            """INSERT INTO scenario_logs (scenario_id, session_id, completed,
               nodes_visited, slots_filled, duration_sec, exit_reason)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (scenario_id, session_id, int(completed), nodes_visited,
             slots_filled, duration_sec, exit_reason),
        )
        await db.commit()
        return cursor.lastrowid


async def list_scenario_logs(db_path: str, scenario_id: int | None = None,
                             limit: int = 100) -> list[dict[str, Any]]:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        if scenario_id:
            cursor = await db.execute(
                "SELECT * FROM scenario_logs WHERE scenario_id = ? ORDER BY created_at DESC LIMIT ?",
                (scenario_id, limit),
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM scenario_logs ORDER BY created_at DESC LIMIT ?", (limit,),
            )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
```

- [ ] **Step 5: Run tests**

Run: `cd /home/jonghooy/work/knowledge-service && python -m pytest backend/tests/test_scenarios_db.py -v`
Expected: All 8 tests PASS

- [ ] **Step 6: Commit**

```bash
cd /home/jonghooy/work/knowledge-service && git add backend/db/sqlite.py backend/tests/test_scenarios_db.py
git commit -m "feat(scenarios): add scenarios + scenario_logs DB tables and CRUD"
```

---

### Task 2: Scenario Structure Validator

**Files:**
- Create: `backend/scenarios/__init__.py`
- Create: `backend/scenarios/validator.py`
- Create: `backend/tests/test_scenario_validator.py`

- [ ] **Step 1: Write failing tests**

```python
"""Test scenario structure validator."""
import json
import pytest
from scenarios.validator import validate_scenario


def _valid_scenario():
    return {
        "nodes": [
            {"id": "n1", "type": "speak", "text": "Hello"},
            {"id": "n2", "type": "end", "message": "Bye", "disposition": "resolved"},
        ],
        "edges": [{"from": "n1", "to": "n2"}],
        "slots": {},
        "triggers": {"examples": ["hi"], "description": "greeting"},
    }


class TestValidateScenario:
    def test_valid_scenario(self):
        result = validate_scenario(_valid_scenario())
        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_missing_nodes(self):
        s = _valid_scenario()
        s["nodes"] = []
        result = validate_scenario(s)
        assert result["valid"] is False
        assert any("node" in e.lower() for e in result["errors"])

    def test_missing_end_node(self):
        s = _valid_scenario()
        s["nodes"] = [{"id": "n1", "type": "speak", "text": "Hello"}]
        s["edges"] = []
        result = validate_scenario(s)
        assert result["valid"] is False
        assert any("end" in e.lower() for e in result["errors"])

    def test_dangling_edge_target(self):
        s = _valid_scenario()
        s["edges"].append({"from": "n1", "to": "n_nonexistent"})
        result = validate_scenario(s)
        assert result["valid"] is False
        assert any("n_nonexistent" in e for e in result["errors"])

    def test_orphan_node(self):
        s = _valid_scenario()
        s["nodes"].append({"id": "n_orphan", "type": "speak", "text": "Orphan"})
        result = validate_scenario(s)
        assert result["valid"] is False
        assert any("n_orphan" in e for e in result["errors"])

    def test_slot_collect_references_valid_slot(self):
        s = _valid_scenario()
        s["nodes"].insert(1, {"id": "n_slot", "type": "slot_collect", "target_slot": "card_number"})
        s["edges"] = [{"from": "n1", "to": "n_slot"}, {"from": "n_slot", "to": "n2"}]
        s["slots"] = {"card_number": {"type": "string", "required": True, "extract": "regex", "prompt": "card?"}}
        result = validate_scenario(s)
        assert result["valid"] is True

    def test_slot_collect_references_missing_slot(self):
        s = _valid_scenario()
        s["nodes"].insert(1, {"id": "n_slot", "type": "slot_collect", "target_slot": "missing_slot"})
        s["edges"] = [{"from": "n1", "to": "n_slot"}, {"from": "n_slot", "to": "n2"}]
        result = validate_scenario(s)
        assert result["valid"] is False
        assert any("missing_slot" in e for e in result["errors"])

    def test_duplicate_node_ids(self):
        s = _valid_scenario()
        s["nodes"].append({"id": "n1", "type": "speak", "text": "Duplicate"})
        result = validate_scenario(s)
        assert result["valid"] is False
        assert any("duplicate" in e.lower() for e in result["errors"])

    def test_missing_triggers(self):
        s = _valid_scenario()
        s["triggers"] = {"examples": [], "description": ""}
        result = validate_scenario(s)
        assert result["valid"] is False
        assert any("trigger" in e.lower() for e in result["errors"])
```

Write to: `backend/tests/test_scenario_validator.py`

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/jonghooy/work/knowledge-service && python -m pytest backend/tests/test_scenario_validator.py -v`
Expected: FAIL

- [ ] **Step 3: Implement validator**

Create `backend/scenarios/__init__.py` (empty):
```python
```

Create `backend/scenarios/validator.py`:
```python
"""Scenario structure validation — checks graph integrity before deployment."""
from __future__ import annotations


_VALID_NODE_TYPES = {
    "speak", "slot_collect", "condition", "api_call", "transfer",
    "end", "llm_response", "confirm", "rag_search",
}


def validate_scenario(scenario: dict) -> dict:
    """Validate scenario structure.

    Args:
        scenario: dict with keys: nodes, edges, slots, triggers

    Returns:
        {"valid": bool, "errors": list[str]}
    """
    errors: list[str] = []
    nodes = scenario.get("nodes", [])
    edges = scenario.get("edges", [])
    slots = scenario.get("slots", {})
    triggers = scenario.get("triggers", {})

    # 1. Must have at least one node
    if not nodes:
        errors.append("Scenario has no nodes")
        return {"valid": False, "errors": errors}

    # 2. Check duplicate node IDs
    node_ids = [n["id"] for n in nodes]
    seen = set()
    for nid in node_ids:
        if nid in seen:
            errors.append(f"Duplicate node ID: {nid}")
        seen.add(nid)
    node_id_set = set(node_ids)

    # 3. Must have at least one end or transfer node
    terminal_types = {"end", "transfer"}
    has_terminal = any(n.get("type") in terminal_types for n in nodes)
    if not has_terminal:
        errors.append("Scenario has no end or transfer node")

    # 4. Validate node types
    for n in nodes:
        ntype = n.get("type", "")
        if ntype not in _VALID_NODE_TYPES:
            errors.append(f"Invalid node type '{ntype}' on node {n['id']}")

    # 5. Validate edge references
    for e in edges:
        if e.get("from") not in node_id_set:
            errors.append(f"Edge references non-existent source node: {e.get('from')}")
        if e.get("to") not in node_id_set:
            errors.append(f"Edge references non-existent target node: {e.get('to')}")

    # 6. Check for orphan nodes (not connected by any edge, except first node)
    connected = set()
    for e in edges:
        connected.add(e.get("from"))
        connected.add(e.get("to"))
    first_node_id = node_ids[0] if node_ids else None
    for nid in node_id_set:
        if nid not in connected and nid != first_node_id and len(nodes) > 1:
            errors.append(f"Orphan node not connected by any edge: {nid}")

    # 7. Validate slot_collect references
    for n in nodes:
        if n.get("type") == "slot_collect":
            target = n.get("target_slot")
            if target and target not in slots:
                errors.append(f"slot_collect node {n['id']} references undefined slot: {target}")

    # 8. Validate triggers
    examples = triggers.get("examples", [])
    if not examples:
        errors.append("Scenario has no trigger examples")

    return {"valid": len(errors) == 0, "errors": errors}
```

- [ ] **Step 4: Run tests**

Run: `cd /home/jonghooy/work/knowledge-service && python -m pytest backend/tests/test_scenario_validator.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /home/jonghooy/work/knowledge-service && git add backend/scenarios/ backend/tests/test_scenario_validator.py
git commit -m "feat(scenarios): add scenario structure validator"
```

---

### Task 3: Scenarios CRUD API Router

**Files:**
- Create: `backend/api/scenarios.py`
- Create: `backend/tests/test_scenarios_api.py`

- [ ] **Step 1: Write failing tests**

```python
"""Test scenarios API endpoints."""
import json
import pytest
from starlette.testclient import TestClient

from db.sqlite import init_db
import api.scenarios as scenarios_module


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    db_file = str(tmp_path / "test.db")
    monkeypatch.setattr("config.SQLITE_PATH", db_file)
    monkeypatch.setattr(scenarios_module, "SQLITE_PATH", db_file)
    return db_file


@pytest.fixture()
def client(tmp_db):
    from main import app
    with TestClient(app) as c:
        yield c


def _create_scenario(client, name="카드 분실 신고"):
    return client.post("/api/scenarios/", json={
        "name": name,
        "description": "테스트 시나리오",
        "graph_json": {"nodes": [
            {"id": "n1", "type": "speak", "text": "Hello"},
            {"id": "n2", "type": "end", "message": "Bye", "disposition": "resolved"},
        ], "edges": [{"from": "n1", "to": "n2"}]},
        "triggers": {"examples": ["카드 잃어버렸어요"], "description": "카드 분실"},
        "slots": {},
    })


class TestScenariosCRUD:
    def test_create_scenario(self, client):
        resp = _create_scenario(client)
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "카드 분실 신고"
        assert data["status"] == "draft"

    def test_list_scenarios(self, client):
        _create_scenario(client, "A")
        _create_scenario(client, "B")
        resp = client.get("/api/scenarios/")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_list_filter_status(self, client):
        resp = _create_scenario(client)
        sid = resp.json()["id"]
        client.post(f"/api/scenarios/{sid}/deploy")
        resp = client.get("/api/scenarios/?status=active")
        assert len(resp.json()) == 1

    def test_get_scenario(self, client):
        resp = _create_scenario(client)
        sid = resp.json()["id"]
        resp = client.get(f"/api/scenarios/{sid}")
        assert resp.status_code == 200
        assert resp.json()["id"] == sid

    def test_get_scenario_not_found(self, client):
        resp = client.get("/api/scenarios/999")
        assert resp.status_code == 404

    def test_update_scenario(self, client):
        resp = _create_scenario(client)
        sid = resp.json()["id"]
        resp = client.put(f"/api/scenarios/{sid}", json={"name": "Updated"})
        assert resp.status_code == 200
        assert resp.json()["name"] == "Updated"
        assert resp.json()["version"] == 2

    def test_delete_scenario(self, client):
        resp = _create_scenario(client)
        sid = resp.json()["id"]
        resp = client.delete(f"/api/scenarios/{sid}")
        assert resp.status_code == 204
        resp = client.get(f"/api/scenarios/{sid}")
        assert resp.status_code == 404

    def test_deploy_scenario(self, client):
        resp = _create_scenario(client)
        sid = resp.json()["id"]
        resp = client.post(f"/api/scenarios/{sid}/deploy")
        assert resp.status_code == 200
        assert resp.json()["status"] == "active"

    def test_archive_scenario(self, client):
        resp = _create_scenario(client)
        sid = resp.json()["id"]
        client.post(f"/api/scenarios/{sid}/deploy")
        resp = client.post(f"/api/scenarios/{sid}/archive")
        assert resp.status_code == 200
        assert resp.json()["status"] == "archived"

    def test_validate_scenario(self, client):
        resp = _create_scenario(client)
        sid = resp.json()["id"]
        resp = client.post(f"/api/scenarios/{sid}/validate")
        assert resp.status_code == 200
        assert resp.json()["structure"]["valid"] is True
```

Write to: `backend/tests/test_scenarios_api.py`

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/jonghooy/work/knowledge-service && python -m pytest backend/tests/test_scenarios_api.py -v`
Expected: FAIL

- [ ] **Step 3: Implement scenarios.py router**

```python
"""Scenarios CRUD API router."""
from __future__ import annotations

import json
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import SQLITE_PATH
from db.sqlite import (
    add_scenario, list_scenarios, get_scenario, update_scenario,
    delete_scenario, update_scenario_status,
)
from scenarios.validator import validate_scenario

router = APIRouter()


class ScenarioCreate(BaseModel):
    name: str
    description: str = ""
    graph_json: dict = {}
    triggers: dict = {"examples": [], "description": ""}
    slots: dict = {}
    priority: int = 0
    created_by: str = "human"
    source: dict = {}


class ScenarioUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    graph_json: dict | None = None
    triggers: dict | None = None
    slots: dict | None = None
    priority: int | None = None
    source: dict | None = None


def _serialize(scenario: dict) -> dict:
    """Parse JSON string fields back to dicts for API response."""
    result = dict(scenario)
    for field in ("graph_json", "source_json", "triggers_json", "slots_json", "metadata_json"):
        if field in result and isinstance(result[field], str):
            try:
                result[field] = json.loads(result[field])
            except (json.JSONDecodeError, TypeError):
                pass
    return result


@router.get("/")
async def list_all(status: Optional[str] = None) -> list[dict]:
    items = await list_scenarios(SQLITE_PATH, status=status)
    return [_serialize(s) for s in items]


@router.post("/", status_code=201)
async def create(body: ScenarioCreate) -> dict:
    sid = await add_scenario(
        SQLITE_PATH,
        name=body.name,
        description=body.description,
        graph_json=json.dumps(body.graph_json),
        status="draft",
        priority=body.priority,
        created_by=body.created_by,
        source_json=json.dumps(body.source),
        triggers_json=json.dumps(body.triggers),
        slots_json=json.dumps(body.slots),
    )
    s = await get_scenario(SQLITE_PATH, sid)
    return _serialize(s)


@router.get("/{scenario_id}")
async def get_one(scenario_id: int) -> dict:
    s = await get_scenario(SQLITE_PATH, scenario_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Scenario not found")
    return _serialize(s)


@router.put("/{scenario_id}")
async def update_one(scenario_id: int, body: ScenarioUpdate) -> dict:
    existing = await get_scenario(SQLITE_PATH, scenario_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Scenario not found")
    kwargs = {}
    if body.name is not None:
        kwargs["name"] = body.name
    if body.description is not None:
        kwargs["description"] = body.description
    if body.graph_json is not None:
        kwargs["graph_json"] = json.dumps(body.graph_json)
    if body.triggers is not None:
        kwargs["triggers_json"] = json.dumps(body.triggers)
    if body.slots is not None:
        kwargs["slots_json"] = json.dumps(body.slots)
    if body.priority is not None:
        kwargs["priority"] = body.priority
    if body.source is not None:
        kwargs["source_json"] = json.dumps(body.source)
    if kwargs:
        await update_scenario(SQLITE_PATH, scenario_id, **kwargs)
    s = await get_scenario(SQLITE_PATH, scenario_id)
    return _serialize(s)


@router.delete("/{scenario_id}", status_code=204)
async def delete_one(scenario_id: int) -> None:
    existing = await get_scenario(SQLITE_PATH, scenario_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Scenario not found")
    await delete_scenario(SQLITE_PATH, scenario_id)


@router.post("/{scenario_id}/deploy")
async def deploy(scenario_id: int) -> dict:
    existing = await get_scenario(SQLITE_PATH, scenario_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Scenario not found")
    await update_scenario_status(SQLITE_PATH, scenario_id, "active")
    s = await get_scenario(SQLITE_PATH, scenario_id)
    return _serialize(s)


@router.post("/{scenario_id}/archive")
async def archive(scenario_id: int) -> dict:
    existing = await get_scenario(SQLITE_PATH, scenario_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Scenario not found")
    await update_scenario_status(SQLITE_PATH, scenario_id, "archived")
    s = await get_scenario(SQLITE_PATH, scenario_id)
    return _serialize(s)


@router.post("/{scenario_id}/validate")
async def validate(scenario_id: int) -> dict:
    existing = await get_scenario(SQLITE_PATH, scenario_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Scenario not found")
    s = _serialize(existing)
    graph = s.get("graph_json", {})
    structure_result = validate_scenario({
        "nodes": graph.get("nodes", []),
        "edges": graph.get("edges", []),
        "slots": s.get("slots_json", {}),
        "triggers": s.get("triggers_json", {}),
    })
    return {"structure": structure_result}
```

Write to: `backend/api/scenarios.py`

- [ ] **Step 4: Register router in main.py**

In `backend/main.py`, add after existing router imports:
```python
from api.scenarios import router as scenarios_router
```

And add to router registration:
```python
app.include_router(scenarios_router, prefix="/api/scenarios", tags=["scenarios"])
```

- [ ] **Step 5: Run tests**

Run: `cd /home/jonghooy/work/knowledge-service && python -m pytest backend/tests/test_scenarios_api.py -v`
Expected: All 10 tests PASS

- [ ] **Step 6: Commit**

```bash
cd /home/jonghooy/work/knowledge-service && git add backend/api/scenarios.py backend/main.py backend/tests/test_scenarios_api.py
git commit -m "feat(scenarios): add scenarios CRUD API with validation"
```

---

### Task 4: Brain API Extensions

**Files:**
- Modify: `backend/api/brain.py`
- Create: `backend/tests/test_brain_scenarios.py`

- [ ] **Step 1: Write failing tests**

```python
"""Test Brain API scenario endpoints."""
import json
import pytest
from starlette.testclient import TestClient

from db.sqlite import init_db, add_scenario, update_scenario_status
import api.brain as brain_module
import api.scenarios as scenarios_module


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    db_file = str(tmp_path / "test.db")
    monkeypatch.setattr("config.SQLITE_PATH", db_file)
    monkeypatch.setattr(brain_module, "SQLITE_PATH", db_file)
    monkeypatch.setattr(scenarios_module, "SQLITE_PATH", db_file)
    return db_file


@pytest.fixture()
def client(tmp_db):
    from main import app
    with TestClient(app) as c:
        yield c


async def _seed_active_scenario(db_path):
    sid = await add_scenario(
        db_path, name="카드 분실",
        graph_json=json.dumps({"nodes": [{"id": "n1", "type": "end", "message": "bye", "disposition": "resolved"}], "edges": []}),
        triggers_json=json.dumps({"examples": ["카드 잃어버렸어요"], "description": "카드 분실"}),
        slots_json=json.dumps({}),
    )
    await update_scenario_status(db_path, sid, "active")
    return sid


class TestBrainScenarios:
    def test_get_scenarios_empty(self, client):
        resp = client.get("/api/brain/scenarios")
        assert resp.status_code == 200
        data = resp.json()
        assert data["scenarios"] == []
        assert "version" in data

    def test_get_scenarios_returns_active_only(self, client, tmp_db):
        import asyncio
        db_path = tmp_db
        asyncio.get_event_loop().run_until_complete(_seed_active_scenario(db_path))
        resp = client.get("/api/brain/scenarios")
        data = resp.json()
        assert len(data["scenarios"]) == 1
        assert data["scenarios"][0]["name"] == "카드 분실"
        assert data["scenarios"][0]["status"] == "active"

    def test_log_execution(self, client, tmp_db):
        import asyncio
        db_path = tmp_db
        sid = asyncio.get_event_loop().run_until_complete(_seed_active_scenario(db_path))
        resp = client.post("/api/brain/log-execution", json={
            "scenario_id": sid,
            "session_id": "ws_abc123",
            "completed": True,
            "nodes_visited": ["n1"],
            "slots_filled": {"card": "1234"},
            "duration_sec": 45.0,
            "exit_reason": "end_node",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "logged"
```

Write to: `backend/tests/test_brain_scenarios.py`

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/jonghooy/work/knowledge-service && python -m pytest backend/tests/test_brain_scenarios.py -v`
Expected: FAIL

- [ ] **Step 3: Add endpoints to brain.py**

Add to `backend/api/brain.py` (after existing endpoints):

```python
from db.sqlite import list_scenarios, add_scenario_log
from datetime import datetime, timezone

SQLITE_PATH = __import__("config").SQLITE_PATH


class ExecutionLogRequest(BaseModel):
    scenario_id: int
    session_id: str
    completed: bool = False
    nodes_visited: list[str] = []
    slots_filled: dict = {}
    duration_sec: float = 0
    exit_reason: str = ""


@router.get("/scenarios")
async def brain_scenarios() -> dict:
    """Return all active scenarios for Brain to cache."""
    items = await list_scenarios(SQLITE_PATH, status="active")
    # Parse JSON fields
    scenarios = []
    for s in items:
        parsed = dict(s)
        for field in ("graph_json", "triggers_json", "slots_json", "metadata_json", "source_json"):
            if field in parsed and isinstance(parsed[field], str):
                try:
                    parsed[field] = __import__("json").loads(parsed[field])
                except Exception:
                    pass
        scenarios.append(parsed)
    return {
        "scenarios": scenarios,
        "version": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/log-execution")
async def brain_log_execution(body: ExecutionLogRequest) -> dict:
    """Record scenario execution log from Brain."""
    import json as _json
    await add_scenario_log(
        SQLITE_PATH,
        scenario_id=body.scenario_id,
        session_id=body.session_id,
        completed=body.completed,
        nodes_visited=",".join(body.nodes_visited),
        slots_filled=_json.dumps(body.slots_filled),
        duration_sec=body.duration_sec,
        exit_reason=body.exit_reason,
    )
    return {"status": "logged"}
```

Also ensure `SQLITE_PATH` import exists at module level. Check if brain.py already imports it from config — if so, reuse. If not, add:
```python
from config import SQLITE_PATH
```

- [ ] **Step 4: Run tests**

Run: `cd /home/jonghooy/work/knowledge-service && python -m pytest backend/tests/test_brain_scenarios.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Run all tests**

Run: `cd /home/jonghooy/work/knowledge-service && python -m pytest backend/tests/ -v`
Expected: All tests PASS (existing + new)

- [ ] **Step 6: Commit**

```bash
cd /home/jonghooy/work/knowledge-service && git add backend/api/brain.py backend/tests/test_brain_scenarios.py
git commit -m "feat(scenarios): add Brain API endpoints for scenario loading and execution logging"
```

---

## Summary

| Task | Component | Files | Tests |
|------|-----------|-------|-------|
| 1 | DB schema + CRUD | sqlite.py (modify) | 8 |
| 2 | Structure validator | scenarios/validator.py | 9 |
| 3 | CRUD API router | api/scenarios.py, main.py (modify) | 10 |
| 4 | Brain API extensions | api/brain.py (modify) | 3 |
| **Total** | **4 tasks** | **4 new + 3 modified** | **~30 tests** |

## Next Plans

- **Plan 3:** AI Scenario Generation Pipeline (Claude API integration, /generate, /refine, /simulate)
- **Plan 4:** Vue Flow Visual Editor Frontend
