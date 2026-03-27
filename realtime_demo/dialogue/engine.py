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
            success = self._enter_scenario(session, match.scenario)
            if not success:
                return DialogueResult(mode="freeform", should_use_s2s=True)
            session["scenario_state"]["prosody"] = prosody
            return await self._execute_scenario_step(session, text)
        else:
            return DialogueResult(mode="freeform", should_use_s2s=True)

    async def _handle_scenario(self, session: dict, text: str, prosody: dict | None) -> DialogueResult:
        session["scenario_state"]["prosody"] = prosody
        result = await self._execute_scenario_step(session, text)

        # Check for mid-scenario intent switch
        # If slot extraction failed (awaiting_input + retry), check if user wants something else
        if result.awaiting_input and session["scenario_state"]["retry_count"] > 0 and text:
            try:
                new_match = await self.intent_matcher.match(text)
                if (new_match and new_match.confidence >= 0.85 and
                    new_match.scenario_id != session["scenario_state"]["scenario_id"]):
                    logger.info(f"Mid-scenario intent switch detected: {new_match.scenario_id} (conf={new_match.confidence:.2f})")
                    # Exit current scenario and enter new one
                    self._exit_scenario(session)
                    success = self._enter_scenario(session, new_match.scenario)
                    if success:
                        session["scenario_state"]["prosody"] = prosody
                        return await self._execute_scenario_step(session, text)
            except Exception:
                pass  # Intent matching failure is non-critical

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

        # Handle intent_route action — match intent and enter sub-scenario
        if result.action and result.action.get("type") == "intent_route":
            user_text = result.action.get("user_text", "")
            try:
                match = await self.intent_matcher.match(user_text)
            except Exception:
                match = None

            if match and match.confidence >= 0.7 and match.scenario_id != state.get("scenario_id"):
                # Push current scenario to stack and enter sub-scenario
                import copy as _copy
                if len(state.get("stack", [])) < 3:  # depth limit
                    parent_state = _copy.deepcopy(state)
                    # Set next node for when we return
                    parent_state["current_node"] = result.action.get("next_node") or state["current_node"]
                    state["stack"] = [parent_state]

                    # Enter sub-scenario
                    sub_scenario = match.scenario
                    state["scenario_id"] = sub_scenario.id
                    state["scenario_version"] = sub_scenario.version
                    state["current_node"] = sub_scenario.get_start_node_id()
                    state["slots"] = {}
                    state["variables"] = {}
                    state["history"] = []
                    state["retry_count"] = 0
                    state["awaiting_confirm"] = False
                    state["_scenario_snapshot"] = _copy.deepcopy(sub_scenario)

                    # Execute first step of sub-scenario
                    walker = GraphWalker(sub_scenario, self.slot_manager, self.llm_engine,
                                        knowledge_client=self.action_runner.knowledge_client if self.action_runner else None)
                    sub_result = await walker.step(state, user_text)
                    if sub_result.response_text:
                        result.response_text = (result.response_text or "") + " " + sub_result.response_text
                        result.response_text = result.response_text.strip()
                    result.action = sub_result.action
                    result.awaiting_input = sub_result.awaiting_input
                else:
                    # Stack full — use no_match message
                    no_match_msg = result.action.get("no_match_message", "")
                    result.action = None
                    result.response_text = no_match_msg
                    result.awaiting_input = True
            else:
                # No match — check no_match counter
                no_match_key = "_intent_route_no_match"
                state["variables"][no_match_key] = state["variables"].get(no_match_key, 0) + 1
                max_no = result.action.get("max_no_match", 3)

                if state["variables"][no_match_key] >= max_no:
                    # Too many no-matches — transfer to agent
                    result.action = {"type": "transfer", "reason": "intent_route_no_match"}
                    result.response_text = "정확한 도움을 드리기 위해 상담원에게 연결해드리겠습니다."
                else:
                    no_match_msg = result.action.get("no_match_message", "다시 말씀해주세요.")
                    result.action = None
                    result.response_text = no_match_msg
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

    def _enter_scenario(self, session: dict, scenario: Scenario) -> bool:
        start = scenario.get_start_node_id()
        if start is None:
            logger.error(f"Scenario {scenario.id} has no nodes, cannot enter")
            return False

        state = session["scenario_state"]
        # Enforce stack depth limit (max 3 nested scenarios: main → sub → sub-switch)
        if len(state.get("stack", [])) >= 3:
            logger.warning("Scenario nesting limit (3) exceeded, rejecting new scenario")
            return False

        session["dialogue_mode"] = "scenario"
        state["scenario_id"] = scenario.id
        state["scenario_version"] = scenario.version
        state["current_node"] = start
        state["slots"] = {}
        state["variables"] = {}
        state["history"] = []
        state["retry_count"] = 0
        state["stack"] = state.get("stack", [])
        state["awaiting_confirm"] = False
        state["_scenario_snapshot"] = copy.deepcopy(scenario)
        return True

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

    async def auto_enter_main(self, session: dict) -> DialogueResult | None:
        """Auto-enter main scenario on new call. Called by server.py on WebSocket connect."""
        # Find main scenario
        main = None
        for s in self.scenario_cache.scenarios.values():
            # Check is_main flag (set via Knowledge Service)
            if getattr(s, '_is_main', False):
                main = s
                break

        if main is None:
            return None

        success = self._enter_scenario(session, main)
        if not success:
            return None

        return await self._execute_scenario_step(session, "")
