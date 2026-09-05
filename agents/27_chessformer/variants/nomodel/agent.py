"""Arena variant: the 27_chessformer engine with the network switched off (CF_USE_MODEL=0).

Loads the real agent.py two directories up under a different module name so that
`harness.arena --agent agents/27_chessformer --opponent agents/27_chessformer/variants/nomodel`
is a wall-clock A/B of the same engine with and without the Chessformer priors.
"""

import importlib.util
import os
import sys

os.environ["CF_USE_MODEL"] = "0"

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_spec = importlib.util.spec_from_file_location("cf_agent_nomodel", os.path.join(_ROOT, "agent.py"))
assert _spec is not None and _spec.loader is not None
_real = importlib.util.module_from_spec(_spec)
sys.modules["cf_agent_nomodel"] = _real
_spec.loader.exec_module(_real)


def get_move(fen: str, time_left_ms: int) -> str:
    return _real.get_move(fen, time_left_ms)
