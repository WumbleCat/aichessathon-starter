# 27_chessformer

A Chessformer-inspired transformer, implemented and trained from scratch by this project, used as
a policy prior inside an alpha-beta search. Entry point: `agent.py` (`get_move(fen, time_left_ms)`).

## Files

| File | Role |
|---|---|
| `agent.py` | submission entry point, time management, repetition memory, model loading |
| `cf_search.py` | iterative deepening PVS with TT, quiescence, killers/history, null move, LMR; policy hook |
| `cf_eval.py` | tapered material + piece-square evaluation (formula-generated tables) |
| `cf_encode.py` | board -> 64 square tokens (side-to-move perspective), move <-> policy index, geometry tables |
| `cf_model.py` | the torch Chessformer (training-time definition) |
| `cf_infer.py` | pure-numpy forward pass used at move time (no torch on the clock) |
| `models/chessformer.npz` | trained weights as numpy arrays plus the config (the `.pt` twin with provenance in `meta` is not shipped) |
| `training/gen_data.py` | self-play data generation labelled by `cf_search` (the teacher) |
| `training/train.py` | training script |
| `training/bench_latency.py` | parameter count / file size / batch-1 latency per config |
| `tests/` | rules, clock, model tests (`python tests/run_tests.py`; the venv has no pytest) |

## The architecture

Following Monroe et al. (Chessformer, ICLR 2026; "Mastering Chess with a Transformer Model",
2024) in spirit, scaled to one CPU core:

- **64 square tokens.** Each token is a linear projection of 19 features (13 piece classes,
  en-passant target, 4 castling rights broadcast to every square, in-check) plus a learned
  square embedding. The board is mirrored when Black is to move so the mover always moves "up".
- **Geometric Attention Bias.** Every head in every layer adds learned biases indexed by the
  (dx, dy) offset between source and target square, the chess relation (same rank / file /
  diagonal / anti-diagonal / knight jump / self) and the Chebyshev distance.
- **Dynamic bias.** A board-dependent attention-logit map (the "smolgen" idea): tokens are
  compressed, concatenated, passed through a small MLP and a shared projection that emits a
  per-head 64x64 bias for this position.
- **Attention policy head.** `logit(from, to) = q(from) . k(to)`; promotions add a per-destination
  offset for N/B/R/Q. 4096 + 96 = 4192 logits, one per possible move, legal-masked at inference.
- **Value head.** Mean pooled tokens -> MLP -> tanh scalar.

## How the model is used at move time

The search runs on the handcrafted evaluation. The network is a numpy forward pass at batch 1
(about 4 ms of CPU for the shipped 0.69 M-parameter model, val top-1 39 % / top-3 64 % against the depth-3 teacher) that returns a prior over the legal
moves; the search orders quiet moves by that prior (hash move, captures by MVV-LVA and killers
keep their classic order first) and steers late-move reductions by it (low-prior quiet moves are
reduced more, high-prior ones less).

Where it is consulted is controlled by two switches, because the benchmarks in `RESULTS.md`
showed that in this alpha-beta search the prior does **not** save nodes: to depth 5-7 the tree
is the same size with or without it, so every call is a cost. The shipped default therefore
consults the network only at the root (`CF_POLICY_REL_DEPTH=0`, two to three calls per move,
about 0.1 % of the move time), where it cannot hurt; `CF_POLICY_REL_DEPTH=n` consults it within
`n` plies of the root, `CF_POLICY_MIN_DEPTH=d` only at nodes with at least `d` plies left,
and `CF_USE_MODEL=0` disables it. `training/bench_ordering.py` and `training/match.py` are the
tools that measured this; use them before changing the defaults.

## Files added for measurement

| File | Role |
|---|---|
| `training/match.py` | paired self-play A/B at a fixed node budget; network calls charged in nodes |
| `training/bench_ordering.py` | nodes to a fixed depth with and without the network, per consultation policy |
| `training/launch_gen.ps1` | detached data-generation workers that survive the launching shell |
| `variants/nomodel/agent.py` | the same engine with the network off, for `harness.arena` A/B games |

## Training provenance

- Teacher: `cf_search.Searcher` (this repository), fixed depth, see shard `params`.
- Positions: engine self-play from random openings with epsilon-random moves; root positions and
  harvested exact PV nodes. No external games, no external engine, no downloaded weights.
- Seeds and parameters are stored in every shard and in the checkpoint's `meta`.
- Reproduce: `python training/gen_data.py ...` then `python training/train.py ...` (see the
  docstrings). Use the project interpreter `.venv/Scripts/python.exe`.

## Results

See `RESULTS.md`.
