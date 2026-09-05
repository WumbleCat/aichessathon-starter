"""One place to configure numba for the engine modules.

The platform starts a fresh process per game, so an on-disk numba cache never helps there; it
only matters for local test/arena runs.  It is opt-in (``NNUE21_NUMBA_CACHE=1``) because a
cached reload of the recursive search functions has produced access violations locally.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from numba import njit

CACHE = os.environ.get("NNUE21_NUMBA_CACHE", "0") == "1"


def jit(fn: Callable[..., Any]) -> Any:
    return njit(cache=CACHE)(fn)


def jit_inline(fn: Callable[..., Any]) -> Any:
    """For tiny helpers called with literal arguments: inlined at the numba IR level, so no
    separate specialisation is compiled per distinct literal."""
    return njit(cache=CACHE, inline="always")(fn)
