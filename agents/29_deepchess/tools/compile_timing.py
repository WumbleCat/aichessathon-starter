"""Measure the import/compile cost of the compiled engine, per function (CPU seconds).

    DEEPCHESS_COMPILE_TIMING=1 .venv/Scripts/python.exe agents/29_deepchess/tools/compile_timing.py

Set NUMBA_OPT to compare optimisation levels.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("DEEPCHESS_COMPILE_TIMING", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

t = time.process_time()
w = time.perf_counter()
import numba  # noqa: E402,F401

print(f"numba import {time.process_time() - t:.1f}s cpu (opt={os.environ.get('NUMBA_OPT', '3')})",
      flush=True)
t = time.process_time()
import dc_engine  # noqa: E402,F401

print(f"dc_engine total {time.process_time() - t:.1f}s cpu", flush=True)
t = time.process_time()
import dc_search  # noqa: E402,F401

print(f"dc_search total {time.process_time() - t:.1f}s cpu", flush=True)
print(f"all: {time.perf_counter() - w:.1f}s wall", flush=True)
print("done", flush=True)
