"""The DeepChess agent with the 2026-09-05 search/evaluation additions switched off.

This exists so the additions can be A/B tested against the exact same engine, model and
build rather than against a remembered number: the only difference between this and the
shipped agent is the three environment flags set below.

It is not a copy of the engine. It sets the flags, then loads the real ``agent.py`` from the
parent directory under a different module name (importing it as ``agent`` would import this
file again) and re-exports its entry point. ``harness/package.py`` only packages root-level
``*.py`` and ``weights/``, so nothing here reaches a submission.
"""

from __future__ import annotations

import importlib.util
import os
import sys

os.environ["DEEPCHESS_ASPIRATION"] = "0"
os.environ["DEEPCHESS_EXTEND_UNSTABLE"] = "0"
os.environ["DEEPCHESS_ADAPTIVE_BLEND"] = "0"

REAL_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REAL_DIR not in sys.path:
    sys.path.insert(0, REAL_DIR)

_spec = importlib.util.spec_from_file_location(
    "deepchess_real_agent", os.path.join(REAL_DIR, "agent.py")
)
assert _spec is not None and _spec.loader is not None
_real = importlib.util.module_from_spec(_spec)
sys.modules["deepchess_real_agent"] = _real
_spec.loader.exec_module(_real)

get_move = _real.get_move
