# DeepChess pairwise engine: implementation plan

Session 29. Directory: `agents/29_deepchess`. Everything ships from this directory;
`teacher/` (Stockfish binary, used only offline for labelling) and `data/` are git-ignored
and never packaged.

## How the harness calls us

- `harness/runner.py` does `import agent` from the agent directory and calls
  `agent.get_move(fen, time_left_ms)` once per request line on stdin. stdout is redirected
  to stderr, so `print` is safe.
- The clock is wall time measured by the referee; the watchdog allows 500 ms grace. Flagging,
  an illegal move, or a crash loses the game. 300 plies goes to material adjudication.
- Only a FEN arrives: no move history. Repetition detection has to be kept by the agent.
- `harness/arena.py` alternates colours every game (even game index = agent is White) and
  defaults to 10 s + 0.1 s per game. `harness/play.py` plays one game at 120 s + 0.5 s.

## Architecture (from the README, option B "recommended adaptation")

Shared position encoder that maps a board feature vector to a latent and then to a scalar
value. Training uses the DeepChess pairwise preference loss
`BCEWithLogits(V(A) - V(B), y)` on pairs of positions, with a value-regression auxiliary
target from teacher scores. At play time the scalar `V` is the leaf evaluation of an
alpha-beta search.

Feature vector: 768 binary piece-square features (12 piece kinds x 64 squares) from the
side-to-move perspective (board mirrored when Black moves, colours swapped) plus 5 extra
bits (4 castling rights in our/their order, en-passant available). Network:

```
773 -> 256 (clipped ReLU) -> 32 (clipped ReLU) -> 32 (clipped ReLU) -> 1
```

Because the input is binary and sparse (<= 32 pieces), the first dense layer is computed as
a sum of active rows, and the whole forward pass runs in a numba kernel that reads the
python-chess bitboards directly (no piece_map). Weights ship as a `.npz` in `models/`,
loaded at import, ~0.8 MB.

## Search

Negamax alpha-beta with iterative deepening, transposition table keyed on
`board._transposition_key()`, MVV-LVA captures first, killer and history move ordering,
quiescence search with delta pruning, null-move pruning, late move reductions, check
extension. Time budget from `time_left_ms`, deadline checked every <= 128 nodes, depth 1
finished unclocked so a legal fallback move always exists. Root repetition avoidance from a
module-level history of positions seen this game.

## Phases

1. Minimal agent: handcrafted material + PST eval, alpha-beta, fallback move. Legal in all
   the mandatory positions and at all the mandatory clocks.
2. Training pipeline: `training/gen_data.py` (random-walk / self-play positions labelled by
   Stockfish offline), `training/train.py` (pairwise + value loss in torch, exports npz),
   numba inference in `agent.py`. Debug checks: A==B gives 0.5, swap inverts, Spearman vs
   teacher.
3. `tests/test_agent.py`: move legality tests, clock tests, repeated calls, model sanity.
4. Runtime optimisation: nps, eval latency, TT.
5. Arena benchmarks vs baselines (paired colours), recorded in `RESULTS.md`.
6. Weakness hunting and re-benchmark; compare handcrafted vs DeepChess scalar vs blend.
