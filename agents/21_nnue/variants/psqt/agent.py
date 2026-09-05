"""A/B variant: the same engine with the PSQT fallback evaluation instead of the network.

Loads the real ``agents/21_nnue/agent.py`` under another module name (so it does not import
itself) and swaps in a Searcher without the network.  Used only by local arena runs:

    python -m harness.arena --agent agents/21_nnue --opponent agents/21_nnue/variants/psqt
"""

from __future__ import annotations

import importlib.util
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_spec = importlib.util.spec_from_file_location("nnue21_agent", os.path.join(ROOT, "agent.py"))
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
sys.modules["nnue21_agent"] = _mod
_spec.loader.exec_module(_mod)

_mod.SEARCHER = _mod.csearch.Searcher(None, use_nnue=False)
get_move = _mod.get_move
reset_game = _mod.reset_game
