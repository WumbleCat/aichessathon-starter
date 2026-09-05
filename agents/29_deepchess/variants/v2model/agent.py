"""The DeepChess agent pinned to the v2 network, to gate v3 against it.

Same code, same search, same flags: only the weights differ. See ``variants/base`` for why
the real ``agent.py`` is loaded under another module name.
"""

from __future__ import annotations

import importlib.util
import os
import sys

os.environ["DEEPCHESS_MODEL"] = "deepchess_v2.npz"

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
