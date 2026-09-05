# 27_chessformer — implementation plan

## Harness facts (from harness/, agent.py, AGENTS.md)

- Platform imports `agent.py` from the zip root and calls `get_move(fen, time_left_ms) -> str` (UCI).
- One process per game; module state survives between moves, not between games.
- Only the FEN is passed: no move history, so the agent keeps its own repetition memory.
- Clock is wall time; the watchdog grace is 500 ms; a flag is a loss. 120 s + 0.5 s/move (arena
  default is a fast 10 s + 0.1 s).
- Init budget 90 s: load weights and warm everything at import time.
- 1 core, 2 GB, no GPU, no network, torch/onnxruntime/numba/numpy/python-chess preinstalled.
- 50 MB uncompressed cap; no native binaries; only self-trained weights.

## Environment on this dev box

- No Stockfish, no internet, CPU-only torch, machine shared with many other sessions (100 % load).
- Therefore the **teacher is our own alpha-beta engine** (`cf_search.py`), which is legal: the
  network is trained from random init on positions our own engine labelled.

## Architecture (Chessformer-inspired, own implementation)

- 64 square tokens, board normalised so the side to move is always "white" moving up.
- Token = linear(piece one-hot 13, ep-square flag, castling rights x4, in-check flag) + learned
  square embedding.
- Encoder layer = pre-LN multi-head self-attention with:
  - **Geometric Attention Bias**: learned per-head tables indexed by (dx, dy), same-rank,
    same-file, same-diagonal, same-anti-diagonal, knight relation, king-adjacency, identity.
  - **Dynamic (smolgen-style) bias**: per-token compression -> flatten -> MLP -> per-head 64x64
    additive attention logits, generated from the whole board (shared final projection).
  - MLP (ratio 2).
- Policy head: attention-style. logits[src,dst] = q(src) . k(dst) / sqrt(d); promotions add a
  per-destination offset for N/B/R/Q. Flat move index space = 4096 + 96 = 4192, unique per move.
- Value head: mean-pool -> MLP -> WDL logits.

## Deployment (ranked in README): policy for move ordering inside alpha-beta

- `cf_search.py`: iterative deepening PVS, TT, quiescence (delta pruning), MVV-LVA, killers,
  history, null-move, LMR, check extension, repetition memory, time checks every <=256 nodes,
  depth-1 unclocked so a legal fallback always exists.
- `cf_eval.py`: tapered material + PST evaluation (fast leaf evaluation).
- The network is called at the root and at interior nodes with depth >= N (measured), ordering
  moves by policy and driving LMR reductions by policy rank. Value head optionally blended at
  root children. Everything measured with the arena before it is kept.

## Phases

1. Minimal legal agent (search + PST eval, no net). Tests: rules + clocks.
2. Model (`cf_model.py`, `cf_encode.py`), size/latency benchmark at batch 1, ONNX export.
3. Data: self-play with the engine, harvesting root and PV-node (exact TT) positions with
   best move + score. `training/gen_data.py`.
4. Train (`training/train.py`), export (`training/export.py`), tests for encoding/flip/promos.
5. Integrate policy in search, benchmark with `harness.arena` paired colours vs baselines and
   vs the no-net version of this same engine. Record in RESULTS.md.
6. Fix weaknesses, re-benchmark.

## Status checkpoint (2026-09-05, session 2)

Session 1 finished phases 1-2 and started phase 3 (its Pool-based data run died with the
session; the six `shard_d3_s2000..2005` shards, 24.5k positions, survived). Session 2:

- Data generation now runs as independent detached processes (`training/launch_gen.ps1`), one
  shard per 4000 positions, so it survives the launching session. Seeds 2006-2045, depth 3.
- `agent.py` loads `models/chessformer.npz` (numpy only) or `models/chessformer.pt`;
  `train.py` writes both. (Session 1 pointed at a non-existent `.onnx`.)
- `tests/run_tests.py` understands real pytest parametrize marks (pytest 9 is now in the venv).
- Next: train on the growing data, integrate, paired arena vs the no-model engine
  (`CF_USE_MODEL=0`), record in RESULTS.md, then package and size-check.
