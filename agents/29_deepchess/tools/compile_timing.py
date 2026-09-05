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

# where the time goes inside numba's pipeline for the big functions
for name in ("gen_moves", "make_move", "quiesce", "search"):
    fn = getattr(dc_search if hasattr(dc_search, name) else dc_engine, name)
    for cres in fn.overloads.values():
        for pipeline, passes in cres.metadata.get("pipeline_times", {}).items():
            items = sorted(passes.items(), key=lambda kv: -kv[1].run)
            print(f"{name} [{pipeline}]: " + ", ".join(f"{p}={t.run:.1f}" for p, t in items[:6]),
                  flush=True)
print("done", flush=True)
