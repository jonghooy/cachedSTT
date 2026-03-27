"""Safe rule evaluator — no eval(), no exec()."""
from __future__ import annotations

from typing import Any

_OPERATORS = {"eq", "neq", "contains", "gt", "lt", "exists", "in"}


class _MissingSentinel:
    """Sentinel for missing field values."""
    pass


_MISSING = _MissingSentinel()


_ALLOWED_ROOTS = {"slots", "variables"}


def _resolve_field(field_path: str, context: dict) -> Any:
    """Resolve dotted field path like 'slots.card_number' from context dict."""
    parts = field_path.split(".")
    if parts and parts[0] not in _ALLOWED_ROOTS:
        return _MISSING
    current = context
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return _MISSING
    return current


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
