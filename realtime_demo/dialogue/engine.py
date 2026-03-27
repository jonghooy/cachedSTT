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
    def __init__(self, scenario_cache: ScenarioCache, intent_matcher, slot_manager, llm_engine=None, action_runner=None):
        self.scenario_cache = scenario_cache
        self.intent_matcher = intent_matcher
        self.slot_manager = slot_manager
        self.llm_engine = llm_engine
        self.action_runner = action_runner

    async def process_utterance(self, session: dict, text: str, prosody: dict | None) -> DialogueResult:
        mode = session.get("dialogue_mode", "freeform")
        if mode == "freeform":
            return await self._handle_freeform(session, text, prosody)
        elif mode == "scenario":
            return await self._handle_scenario(session, text, prosody)
        else:
            return DialogueResult(should_use_s2s=True)

    async def _handle_freeform(self, session: dict, text: str, prosody: dict | None) -> DialogueResult:
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

    async def _handle_scenario(self, session: dict, text: str, prosody: dict | None) -> DialogueResult:
        session["scenario_state"]["prosody"] = prosody
        result = await self._execute_scenario_step(session, text)
        if result.action and result.action.get("type") in ("end", "transfer"):
            # Send execution log to Knowledge Service (fire-and-forget)
            await self._log_execution(session, result.action)
            self._exit_scenario(session)
        return result

    async def _execute_scenario_step(self, session: dict, text: str) -> DialogueResult:
        state = session["scenario_state"]
        scenario = state.get("_scenario_snapshot")
        if scenario is None:
            logger.error("No scenario snapshot in session")
            self._exit_scenario(session)
            return DialogueResult(should_use_s2s=True)
        walker = GraphWalker(scenario, self.slot_manager, self.llm_engine,
                             knowledge_client=self.action_runner.knowledge_client if self.action_runner else None)
        result = await walker.step(state, text)

        # Execute api_call action inline and continue walking
        if result.action and result.action.get("type") == "api_call" and self.action_runner:
            api_result = await self.action_runner.execute_api_call(result.action, state)
            if api_result is None and result.action.get("on_error"):
                # API failed → route to on_error node
                state["current_node"] = result.action["on_error"]
                # Continue walking from error node
                error_result = await walker.step(state, "")
                if error_result.response_text:
                    result.response_text = (result.response_text or "") + " " + error_result.response_text
                    result.response_text = result.response_text.strip()
                result.action = error_result.action
            else:
                # API succeeded → continue walking from current node (already advanced by api_call handler)
                continue_result = await walker.step(state, "")
                if continue_result.response_text:
                    result.response_text = (result.response_text or "") + " " + continue_result.response_text
                    result.response_text = result.response_text.strip()
                if continue_result.action:
                    result.action = continue_result.action
                if continue_result.awaiting_input:
                    result.awaiting_input = True

        return result

    async def _log_execution(self, session: dict, action: dict):
        """Send scenario execution log to Knowledge Service (non-blocking)."""
        if not self.action_runner or not self.action_runner.knowledge_client:
            return
        state = session["scenario_state"]
        try:
            client = self.action_runner.knowledge_client._client
            await client.post("/log-execution", json={
                "scenario_id": state.get("scenario_id", 0),
                "session_id": f"ws_{id(session)}",
                "completed": action.get("type") == "end",
                "nodes_visited": state.get("history", []),
                "slots_filled": state.get("slots", {}),
                "duration_sec": 0,  # TODO: track actual duration
                "exit_reason": action.get("type", "unknown"),
            })
        except Exception:
            logger.debug("Failed to send execution log (non-critical)")

    def _enter_scenario(self, session: dict, scenario: Scenario):
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
        state["awaiting_confirm"] = False
        state["_scenario_snapshot"] = copy.deepcopy(scenario)

    def _exit_scenario(self, session: dict):
        state = session["scenario_state"]
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
        state["awaiting_confirm"] = False
        state["_scenario_snapshot"] = None
