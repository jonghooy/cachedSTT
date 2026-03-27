"""Graph walker — traverses scenario node graph step by step."""
from __future__ import annotations

import logging
from typing import Any

from realtime_demo.dialogue.models import Scenario, Node, DialogueResult
from realtime_demo.dialogue.rule_evaluator import evaluate_rule
from realtime_demo.dialogue.template_renderer import render_template, render_dict

logger = logging.getLogger(__name__)

MAX_VISITS_PER_NODE = 3
MAX_AUTO_ADVANCE = 20


class GraphWalker:
    def __init__(self, scenario: Scenario, slot_manager, llm_engine=None, knowledge_client=None):
        self.scenario = scenario
        self.slot_manager = slot_manager
        self.llm_engine = llm_engine
        self._knowledge_client = knowledge_client

    async def step(self, state: dict, user_text: str) -> DialogueResult:
        collected_text_parts: list[str] = []
        auto_steps = 0

        while auto_steps < MAX_AUTO_ADVANCE:
            auto_steps += 1
            node_id = state["current_node"]
            node = self.scenario.get_node(node_id)

            if node is None:
                return self._error_result(state, f"Node {node_id} not found")

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

            if result.action is not None:
                result.response_text = " ".join(collected_text_parts) if collected_text_parts else result.response_text
                return result
            if result.awaiting_input:
                result.response_text = " ".join(collected_text_parts) if collected_text_parts else result.response_text
                return result

            user_text = ""

        return self._error_result(state, "Max auto-advance steps exceeded")

    async def _execute_node(self, node: Node, state: dict, user_text: str) -> DialogueResult:
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
        else:
            return self._error_result(state, f"Speak node {node.id} has no outgoing edge")
        return DialogueResult(mode="scenario", response_text=text)

    async def _handle_slot_collect(self, node: Node, state: dict, user_text: str) -> DialogueResult:
        target_slot = node.data.get("target_slot")
        slot_def = self.scenario.slots.get(target_slot)
        if not slot_def:
            return self._error_result(state, f"Slot {target_slot} not defined")

        if not user_text:
            return DialogueResult(mode="scenario", response_text=slot_def.prompt, awaiting_input=True)

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
            label = await self._llm_condition(node, state, user_text)
        else:
            label = "false"

        next_id = self.scenario.get_next_node_id(node.id, label)
        if next_id:
            state["current_node"] = next_id
        else:
            next_id = self.scenario.get_next_node_id(node.id)
            if next_id:
                state["current_node"] = next_id
            else:
                return self._error_result(state, f"Condition node {node.id} has no outgoing edge for label={label}")
        return DialogueResult(mode="scenario")

    async def _handle_confirm(self, node: Node, state: dict, user_text: str) -> DialogueResult:
        ctx = {"slots": state["slots"], "variables": state["variables"]}

        if not state.get("awaiting_confirm"):
            text = render_template(node.data.get("template", ""), ctx)
            state["awaiting_confirm"] = True
            return DialogueResult(mode="scenario", response_text=text, awaiting_input=True)
        else:
            state["awaiting_confirm"] = False
            confirmed = await self._judge_yes_no(user_text)
            label = "yes" if confirmed else "no"
            next_id = self.scenario.get_next_node_id(node.id, label)
            if next_id:
                state["current_node"] = next_id
            return DialogueResult(mode="scenario")

    async def _handle_api_call(self, node: Node, state: dict, user_text: str) -> DialogueResult:
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
        """RAG search: execute via action_runner's knowledge_client, store result, auto-advance."""
        ctx = {"slots": state["slots"], "variables": state["variables"]}
        query = render_template(node.data.get("query_template", ""), ctx)
        result_var = node.data.get("result_var")
        top_k = node.data.get("top_k", 3)

        # Try to execute RAG search if knowledge_client available via slot_manager or engine
        # For now, store the query for the engine to execute
        if result_var:
            state["variables"][result_var] = []  # default empty
            # Attempt search if we have a reference to knowledge_client
            if hasattr(self, '_knowledge_client') and self._knowledge_client:
                try:
                    results = await self._knowledge_client.search(query, top_k)
                    state["variables"][result_var] = results
                except Exception:
                    logger.warning(f"RAG search failed for query: {query}")

        next_id = self.scenario.get_next_node_id(node.id)
        if next_id:
            state["current_node"] = next_id
        return DialogueResult(mode="scenario")

    async def _handle_llm_response(self, node: Node, state: dict, user_text: str) -> DialogueResult:
        if not self.llm_engine:
            next_id = self.scenario.get_next_node_id(node.id)
            if next_id:
                state["current_node"] = next_id
            return DialogueResult(mode="scenario", response_text="죄송합니다. 잠시 후 다시 시도해주세요.")

        instruction = node.data.get("instruction", "")
        context_slots = {k: state["slots"].get(k) for k in node.data.get("context_slots", [])}
        # Collect any RAG results from variables (stored by rag_search nodes)
        rag_context = ""
        for key, val in state["variables"].items():
            if isinstance(val, list) and val:  # RAG results are lists
                rag_context += "\n".join(str(r.get("text", r) if isinstance(r, dict) else r) for r in val) + "\n"

        prompt = f"지시: {instruction}\n"
        if context_slots:
            prompt += f"슬롯: {context_slots}\n"
        if rag_context:
            prompt += f"참고 문서: {rag_context}\n"

        prosody = state.get("prosody")
        if prosody:
            prompt += f"\n고객 음성: 에너지={prosody.get('energy', 'normal')}, 속도={prosody.get('speech_rate', 'normal')}, 추세={prosody.get('energy_trend', 'stable')}"

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
        return DialogueResult(mode="scenario", response_text=response)

    async def _handle_transfer(self, node: Node, state: dict, user_text: str) -> DialogueResult:
        text = node.data.get("message", "상담원에게 연결해드리겠습니다.")
        transfer_data = node.data.get("transfer_data", {})
        slots_to_send = state["slots"] if transfer_data.get("slots") == "all" else {}
        return DialogueResult(
            mode="scenario",
            response_text=text,
            action={"type": "transfer", "reason": node.data.get("reason", ""), "slots": slots_to_send},
        )

    async def _handle_end(self, node: Node, state: dict, user_text: str) -> DialogueResult:
        text = node.data.get("message", "")
        return DialogueResult(
            mode="scenario",
            response_text=text,
            action={"type": "end", "disposition": node.data.get("disposition", "resolved"), "tags": node.data.get("tags", [])},
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
        else:
            return DialogueResult(
                mode="scenario",
                response_text="죄송합니다. 다시 전화 부탁드립니다.",
                action={"type": "end", "disposition": "failed"},
            )

    async def _judge_yes_no(self, text: str) -> bool:
        positive = {"네", "예", "맞아요", "맞습니다", "그래요", "맞아", "응", "넵"}
        negative = {"아니요", "아니오", "아닙니다", "아니에요", "틀려요", "아니"}
        text_lower = text.strip()
        if any(w in text_lower for w in positive):
            return True
        if any(w in text_lower for w in negative):
            return False
        return True

    async def _llm_condition(self, node: Node, state: dict, user_text: str) -> str:
        """Evaluate condition via LLM. Returns branch label string."""
        if not self.llm_engine:
            return node.data.get("default_branch", "false")

        instruction = node.data.get("instruction", "")
        branches = []
        for e in self.scenario.get_outgoing_edges(node.id):
            if e.label:
                branches.append(e.label)

        ctx = {"slots": state["slots"], "variables": state["variables"]}
        prompt = f"상황: {instruction}\n슬롯: {ctx['slots']}\n사용자 발화: {user_text}\n선택지: {branches}\n위 상황에서 적절한 선택지를 하나만 출력하세요."

        messages = [
            {"role": "system", "content": "You are a decision assistant. Output ONLY the chosen branch label, nothing else."},
            {"role": "user", "content": prompt},
        ]

        try:
            result_text = ""
            async for token in self.llm_engine.generate_stream(messages, max_tokens=10, temperature=0.0):
                result_text += token
            result_text = result_text.strip()
            if result_text in branches:
                return result_text
            return node.data.get("default_branch", branches[0] if branches else "false")
        except Exception:
            logger.exception("LLM condition evaluation failed")
            return node.data.get("default_branch", "false")

    def _error_result(self, state: dict, msg: str) -> DialogueResult:
        logger.error(msg)
        return DialogueResult(
            mode="scenario",
            response_text="시스템 오류가 발생했습니다.",
            action={"type": "end", "disposition": "error", "reason": msg},
        )
