"""Backward-compat shim. Graph vocabulary now lives in ``src/graph/vocab.py``.

This file re-exports the public API so existing callers using
``from data_generation.graph_vocab import NodeKind`` keep working without edits.
New code should import from ``graph.vocab`` directly.

Will be removed once all consumers are migrated.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from graph.vocab import (  # noqa: E402, F401
    ALLOWED_ENDPOINTS,
    EDGE_ID_PREFIX,
    EdgeKind,
    GraphVocabError,
    NODE_ID_PREFIX,
    NodeKind,
    make_edge,
    make_node,
    validate_graph,
    validate_network,
)

__all__ = [
    "ALLOWED_ENDPOINTS",
    "EDGE_ID_PREFIX",
    "EdgeKind",
    "GraphVocabError",
    "NODE_ID_PREFIX",
    "NodeKind",
    "make_edge",
    "make_node",
    "validate_graph",
    "validate_network",
]
