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
