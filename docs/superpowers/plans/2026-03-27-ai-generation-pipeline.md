# AI Scenario Generation Pipeline — Plan 3

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development or superpowers:executing-plans.

**Goal:** Add Claude API integration to Knowledge Service for AI-powered scenario generation, refinement, and simulation testing.

**Architecture:** New `backend/scenarios/generator.py` module with Claude API client. Three new endpoints on the existing scenarios router. Claude generates scenario JSON matching the established schema, validator checks structure, simulation tests paths.

**Tech Stack:** anthropic Python SDK, existing FastAPI scenarios router, existing validator

**Working directory:** `/home/jonghooy/work/knowledge-service`

---

## File Structure

```
backend/
├── scenarios/
│   ├── generator.py       # NEW: Claude API client + scenario generation
│   └── validator.py       # EXISTING (no changes)
├── api/
│   └── scenarios.py       # MODIFY: add /generate, /refine, /simulate endpoints
└── tests/
    └── test_scenario_generator.py  # NEW
```

---

### Task 1: Scenario Generator Module

**Files:**
- Create: `backend/scenarios/generator.py`
- Create: `backend/tests/test_scenario_generator.py`

Generator calls Claude API with scenario schema in system prompt. Returns parsed scenario JSON. Includes validation + retry loop (max 3).

### Task 2: API Endpoints — /generate, /refine, /simulate

**Files:**
- Modify: `backend/api/scenarios.py`

Add three endpoints:
- `POST /api/scenarios/generate` — mode: natural_language | document | call_log
- `POST /api/scenarios/{id}/refine` — instruction text → Claude modifies scenario
- `POST /api/scenarios/{id}/simulate` — Claude plays customer role, tests paths

---

## Summary

| Task | Component | Files |
|------|-----------|-------|
| 1 | Generator module (Claude API) | scenarios/generator.py + test |
| 2 | API endpoints (generate/refine/simulate) | api/scenarios.py (modify) |
