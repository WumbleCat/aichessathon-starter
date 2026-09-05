"""Control: the same engine with the policy network switched off."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _variant import load_main

_main = load_main(PN_USE_POLICY="0")
get_move = _main.get_move
