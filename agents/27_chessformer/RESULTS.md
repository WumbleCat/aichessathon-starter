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

## Session 2 (2026-09-05) — training and integration

Box state during this session: 16 logical cores at 100 %, ~60 claude and ~60 python processes,
1.4-3 GB RAM free. Two findings that decide how everything below is run:

- **torch threads.** A training step (0.69 M params, batch 256) takes 1.0 s with one torch
  thread and 11 s with four (OpenMP spin-waiting on an oversubscribed machine). Everything
  trains with `--threads 1` (now the default).
- **Priority.** Detached workers started at BelowNormal priority got ~0 CPU for 10 minutes and,
  even after being raised to Normal, kept an 8 MB working set and page-faulted continuously.
  They were restarted at Normal priority.

Wall-clock arena scores are therefore not comparable between runs. Model-vs-no-model
comparisons use `training/match.py`: both engines in one process, a fixed node budget per
move, same random openings with colours swapped, and every network call charged its measured
node-equivalent (engine 14.9k nodes/s of CPU time, network 3.9 ms at batch 1 -> 58 nodes).

### Data

Teacher: `cf_search.Searcher` at depth 3 (budget cap 2 s), epsilon 0.1 random moves, root and
exact PV nodes of depth >= 2 harvested. ~4.1k positions per shard; ~2-3 positions/s per worker
when the worker actually gets a core.

### Smoke model (`smoke_tiny`, 64/2/4, smol_hidden 32, 0.69 M params, 2.8 MB npz)

Trained 3 epochs on the first 24.5k positions: val top-1 21.3 %, top-3 39.0 %, value MSE 0.091.

Paired node-budget match, 2000 nodes/move, policy at depth >= 3, cost 58 nodes/call
(run cut short by a machine restart of the workers): **smoke model +4 =0 -0** vs the same
engine without the network.

Harness check with the model loaded through `agent.py` (npz, no torch at import): import 14 s
on the loaded box, 2/2 wins vs baselines/greedy by checkmate, no illegal moves or flags.
