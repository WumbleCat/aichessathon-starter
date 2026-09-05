"""Helper for A/B variant directories: load the main agent under a configuration.

Each ``variants/<name>/agent.py`` sets environment variables and calls ``load_main()``; the
harness then plays that directory like any other agent, so two configurations of the same code
can be matched against each other with ``harness.arena``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from types import ModuleType

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_main(**env: str) -> ModuleType:
    for key, value in env.items():
        os.environ[key] = value
    if MAIN_DIR not in sys.path:
        sys.path.insert(0, MAIN_DIR)
    spec = importlib.util.spec_from_file_location(
        "pn_main_agent", os.path.join(MAIN_DIR, "agent.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["pn_main_agent"] = module
    spec.loader.exec_module(module)
    return module
