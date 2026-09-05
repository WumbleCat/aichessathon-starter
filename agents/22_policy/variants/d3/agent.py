"""Policy prior at the root and at every node with remaining depth >= 3."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _variant import load_main

_main = load_main(PN_USE_POLICY="1", PN_POLICY_ROOT="1", PN_POLICY_MIN_DEPTH="3", PN_POLICY_LMR="1")
get_move = _main.get_move
