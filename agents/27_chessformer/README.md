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
| `models/chessformer.pt` | trained weights: `{"config", "state_dict", "meta"}` (provenance inside `meta`) |
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

The search runs on the handcrafted evaluation; the network is called at the root and at
interior nodes with remaining depth >= `CF_POLICY_MIN_DEPTH` (default 3), where its priors
order the moves and steer late-move reductions (low-prior quiet moves are reduced more,
high-prior ones less). Inference is a numpy forward pass at batch 1.

Environment switches (for experiments): `CF_USE_MODEL=0` disables the network,
`CF_POLICY_MIN_DEPTH=n` changes where it is consulted.

## Training provenance

- Teacher: `cf_search.Searcher` (this repository), fixed depth, see shard `params`.
- Positions: engine self-play from random openings with epsilon-random moves; root positions and
  harvested exact PV nodes. No external games, no external engine, no downloaded weights.
- Seeds and parameters are stored in every shard and in the checkpoint's `meta`.
- Reproduce: `python training/gen_data.py ...` then `python training/train.py ...` (see the
  docstrings). Use the project interpreter `.venv/Scripts/python.exe`.

## Results

See `RESULTS.md`.
