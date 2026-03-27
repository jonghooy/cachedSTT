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
    if current is None:
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
