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

### tiny_v1 (64/2/4, smol 32; 8 epochs on 40.8k positions, no augmentation)

Val top-1 31 %, top-3 51 %, but it overfits from epoch 4 (train policy loss 0.77 vs val 4.69
at epoch 8; the shipped checkpoint is the last one).

Paired node-budget matches vs the no-model engine, 3000 nodes/move (average depth ~3.2),
network charged 58-62 nodes/call, 10 pairs each:

| Setting | Result for the model | Score | Elo (95 % CI) | calls/move |
|---|---|---|---|---|
| policy at depth >= 3 | +10 =1 -9 | 52.5 % | +17 [-142, +185] | 6.6 |
| policy at depth >= 4 | +8 =2 -10 | 45.0 % | -35 [-208, +121] | 1.6 |

No measurable difference at this budget.

**Fixed-depth ordering benchmark** (`training/bench_ordering.py`, 40 positions from a shard,
depth 5, nodes relative to the engine without the network; 60 nodes charged per call):

| Ordering scheme | where consulted | raw nodes | charged | calls/pos | same best move |
|---|---|---|---|---|---|
| prior-first (session 1) | depth >= 3 | 0.99x | 2.30x* | 82 | 34/40 |
| prior-first | depth >= 4 | 1.00x | 1.42x* | 26 | 32/40 |
| prior-first | depth >= 5 (root) | 0.98x | 1.03x* | 2.8 | 38/40 |
| quiet moves by prior, captures/killers classic | depth >= 3 | 0.99x | 1.46x | 83 | 33/40 |

\* charged at 171 nodes/call, the CPU-time ratio measured in that run (the box was loaded).

Conclusion for tiny_v1: its priors are not sharper than hash move + MVV-LVA + killers +
history, so they save no nodes, and every call is pure cost. The overfit and the depth-3
teacher are the likely causes; tiny_v2 (augmentation, label smoothing, best-epoch checkpoint,
57k positions) reaches val policy loss 2.57 / top-1 35 % at epoch 5 and is benchmarked next.

### tiny_v2 (64/2/4, smol 32; 6 epochs on 57.3k positions, mirror augmentation, label smoothing 0.05, wd 0.05)

Best epoch (5) by validation policy loss: **val policy loss 2.57, top-1 34.5 %, top-3 59.2 %,
value MSE 0.073** (tiny_v1: 3.06 / 31 % / 51 % at its best epoch). Weights 2.8 MB (npz).

Fixed-depth ordering benchmark, quiet moves ordered by prior (60 nodes charged per call):

| depth | where consulted | raw nodes | charged | calls/pos | same best move |
|---|---|---|---|---|---|
| 5 | root only | 1.01x | 1.02x | 3.0 | 34/40 |
| 5 | depth >= 3 (everywhere) | 1.00x | 1.46x | 81 | 32/40 |
| 6 | within 2 plies of the root (16 pos) | **0.90x** | 1.14x | 91 | 11/16 |

At depth 5 the prior still saves nothing; at depth 6 it saves 10 % of the nodes, but ~90
calls per position cost more than that. The trend says the prior pays off only when the
subtrees below the consulted nodes are large, i.e. deeper searches with the network confined
to the top plies. Benchmarked next: depth 6 and 7 with the network within one ply of the root.

### Depth 6-7, network within one ply of the root (tiny_v2, min depth 4)

| depth | positions | raw nodes | charged | calls/pos | same best move |
|---|---|---|---|---|---|
| 6 | 16 | 1.08x | 1.16x | 32 | 11/16 |
| 7 | 8 | 1.00x | 1.03x | 27 | 5/8 |

So the 0.90x at depth 6 above was tree-shape noise; across depths 5-7 and every consultation
policy the prior leaves the node count unchanged (0.99-1.08x) and only adds its own cost.

### Value head vs static eval (held-out shard `s2039`, 1500 positions, target tanh(depth-3 score / 500))

| predictor | MSE | corr |
|---|---|---|
| static eval `cf_eval.evaluate` | 0.034 | 0.966 |
| network value head (tiny_v2) | 0.047 | 0.952 |
| least-squares blend 0.79 static + 0.21 net | 0.032 | |

The network is distilled from searches that use the static eval as leaf evaluator, so it
cannot beat that eval at predicting them; the blend gains 7 % MSE, not enough to pay for a
forward pass at leaves. Top-1 agreement with the teacher on the held-out shard: 35.7 %.

### Shipped configuration and wall-clock checks (tiny_v2 weights, network at the root only)

`agent.py` defaults: `CF_POLICY_REL_DEPTH=0`, `CF_POLICY_MIN_DEPTH=3` (2-3 network calls per
move). 41/41 unit tests pass. Harness arena at the fast control (10 s + 0.1 s), five arenas
running concurrently on the loaded box:

| Opponent | Games | Result | Terminations |
|---|---|---|---|
| baselines/greedy | 4 | +4 =0 -0 | 4 checkmate |
| baselines/minimax | 2 | +2 =0 -0 | 2 flag (minimax) |
| baselines/numba | 2 | +2 =0 -0 | 2 flag (numba) |
| variants/nomodel (same engine, network off) | 4 | +2 =1 -1 | 2 checkmate, 1 insufficient material, 1 flag (the variant) |

The flags are the opponents' (including our own no-model variant once): on this machine a
starved process can overrun the 500 ms grace at a 10 s clock. Nothing flagged on our side.

Contest control (120 s + 0.5 s), harness arena: **+2 =0 -0 vs baselines/minimax, both by
checkmate**, no flag, while five other arena processes and a training run shared the box.

Move-time profile (`training/clock_profile.py`, self-play, 60 plies at 120 s + 0.5 s, loaded
box): import 1.0 s; **median 2.4 s, p95 4.2 s, p99 4.4 s, max 4.4 s** per move; both sides
had 59 s left after 30 moves each. The budget is deliberately conservative (about 1/28 of the
remaining time in the middlegame) because the clock is wall time and a flag loses the game.
