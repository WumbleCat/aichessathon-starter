# 27_chessformer — results log

Machine: AMD Ryzen 7 5800H, Windows 11, shared with ~a dozen other build sessions at 100 % CPU
load throughout (all timings below are inflated by that load; the arena clock is wall time).
Interpreter: project `.venv` (torch 2.13 CPU, numpy 2.5, python-chess 1.11, numba 0.67).
No pytest on the box: tests run with `tests/run_tests.py`.

## Phase 1 — search-only agent (no network)

Tests: 24/24 (rules + clocks 50 ms .. 120 s) pass.

Arena, fast control 10 s + 0.1 s, alternating colours:

| Opponent | Games | Result | Score | Terminations |
|---|---|---|---|---|
| baselines/greedy | 6 | +6 =0 -0 | 100 % | 6 checkmate |
| baselines/minimax | 6 | +6 =0 -0 | 100 % | 6 flag (minimax has no time management; it flagged) |

Search statistics (start position, 0.3 s budget, loaded machine): depth 3, 4-14k nodes/s.
Depth-4 search of a middlegame position: 7.5k nodes in 1.27 s (5.9k nps), ~50 % quiescence.

## Model latency (random weights, batch 1, one thread, loaded machine)

| Config | dim/layers/heads/smol | Params | File | torch | numpy |
|---|---|---|---|---|---|
| tiny | 64/2/4/64 | 1.28 M | 5.2 MB | 9.9 ms | 4.2 ms |
| small | 96/4/4/128 | 2.98 M | 11.9 MB | 24.5 ms | 10.5 ms |
| medium | 128/4/8/128 | 5.34 M | 21.4 MB | 35.1 ms | 14.7 ms |
| large | 128/6/8/256 | 11.25 M | 45.0 MB | 61.6 ms | 41.3 ms |

The numpy forward pass wins at batch 1, so that is what ships (torch is used only to read the
checkpoint at import). `onnx` is not installed in the venv, so ONNX export could not be compared.
Note the first measurement without pinning BLAS threads was 3-30x slower on this oversubscribed
machine: `OMP_NUM_THREADS=1` is set in `agent.py` before numpy is imported.
