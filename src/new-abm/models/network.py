"""Backward-compat shim. The graph dataclasses now live in ``src/graph/network.py``.

This file re-exports them so existing callers using ``from models.network import RoadNode``
keep working without edits. New code should import from ``graph.network`` directly.

Will be removed once all consumers are migrated.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from graph.network import RoadEdge, RoadNetwork, RoadNode  # noqa: E402, F401

__all__ = ["RoadEdge", "RoadNetwork", "RoadNode"]
