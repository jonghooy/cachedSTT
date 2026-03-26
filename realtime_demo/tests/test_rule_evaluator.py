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
