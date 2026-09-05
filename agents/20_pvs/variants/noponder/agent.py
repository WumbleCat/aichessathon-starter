"""Harness variant of 20_pvs with pondering switched off, for real-clock A/B games.

    python -m harness.play --white agents/20_pvs --black agents/20_pvs/variants/noponder

The real agent is loaded under another module name (a plain ``import agent`` here would
import this file again) and its PONDER flag is cleared before any move is played.
"""

from __future__ import annotations

import importlib.util
import os

_REAL = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "agent.py"))
_spec = importlib.util.spec_from_file_location("pvs20_real_agent", _REAL)
assert _spec is not None and _spec.loader is not None
_real = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_real)
_real.PONDER = False


def get_move(fen: str, time_left_ms: int) -> str:
    return str(_real.get_move(fen, time_left_ms))
