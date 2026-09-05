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

Negamax alpha-beta with iterative deepening, transposition table, MVV-LVA captures first,
killer and history move ordering, quiescence search with delta pruning, null-move pruning,
late move reductions, reverse futility and futility pruning, check extension, mate distance
pruning. Two implementations of the same search exist and share the time management in
`agent.py`:

- **python** (`Searcher` in `agent.py`): python-chess board, numba leaf evaluation. About
  7k nodes/s. Selected with `DEEPCHESS_ENGINE=python`; also the automatic fallback if the
  compiled modules fail to import.
- **numba** (`dc_engine.py` + `dc_search.py`, default): mailbox board, own move generator
  (perft-verified), make/unmake with incremental first-layer accumulators for both
  perspectives (NNUE style), the whole tree search compiled. Root move loop stays in
  Python (`NumbaSearcher.search_root`), one compiled call per root move. The clock is a
  `threading.Timer` writing into a `stop` array (search runs with `nogil=True`) plus a node
  cap derived from the measured nodes/s as a backstop. Every function has an explicit
  signature, so all compilation happens at import; `tools/validate_numba.py` cross-checks
  perft, the incremental evaluation and hashes against the python path, the mates, and
  nodes/s.

Compile time was the hard part (`tools/compile_timing.py`, numbers are CPU seconds on the
loaded dev machine, roughly half that on an idle core): the first version of `search`
took 85 s because numba's type inference is superlinear in the number of variables and
call-site arguments of one big recursive function (39 separate array arguments at 5
recursive call sites), and literal `True`/`False` arguments at recursive call sites made
numba compile the function three times over. The current layout, all arrays in one
`ctx` tuple (`CTX_FIELDS`), two recursive call sites, helpers for move scoring and
selection, no literal arguments, compiles `search` in 19 s and everything in ~60 s.
`agent.py` compiles the engine in a daemon thread and joins it for what is left of a 70 s
budget measured from the start of the import; if the compile is not done the python
engine plays (with a quarter of the time budget while the thread shares the GIL) and the
compiled engine takes over at the first `get_move` after it is ready.

numba caches compiled kernels on disk when the agent directory is writable
(`_NUMBA_CACHE` in `agent.py`): the local harness starts a fresh process per game and,
under machine load, a cold compile can exceed the 90 s init budget (games lost "by init").
The platform filesystem is read-only, so there the cache is off and the compile runs on the
idle core (a few seconds of CPU).

Only a FEN arrives per move, so repetition memory is kept in the agent: the compiled engine
stores the Zobrist hashes of the positions it was asked to move in (`game_hashes`) and the
search path in one `hist` array (`is_repetition` in `dc_search.py`).

Evaluation modes (`DEEPCHESS_EVAL`): `net` (default), `hand`, `blend` with
`DEEPCHESS_BLEND` as the network weight. `tools/selfplay.py` compares modes and engines
with paired openings at fixed nodes or fixed CPU time per move, which is what the shared,
overloaded dev machine allows (wall-clock results there measure load, not strength).

## Phases and status (2026-09-05)

1. Done. Minimal agent: handcrafted material + PST eval, alpha-beta, fallback move. Legal in
   all the mandatory positions and at all the mandatory clocks.
2. Done. Training pipeline: `training/gen_data.py` (random-walk / self-play positions
   labelled by Stockfish offline, 1M positions at depth 8, `data_gen.log`),
   `training/train.py` (pairwise + value loss in torch, exports npz). Model v1
   (`models/deepchess_v1.*`, `train_v1.log`): 207k parameters, 20 epochs on 697k
   non-check non-capture positions, validation pairwise accuracy 0.93, Spearman 0.954
   against the teacher, MAE 97 cp. `models/deepchess.npz` is v1.
3. Done. `tests/test_agent.py`: move legality, special moves, clocks, repeated calls,
   model sanity (run the file directly; the venv has no pytest). Two tests depend on the
   search reaching depth 2-3 and can fail on the overloaded machine.
4. In progress. Compiled engine (`dc_engine.py` perft-verified, `dc_search.py` written,
   driver in `agent.py`), being validated with `tools/validate_numba.py`. Open questions:
   import/compile time on an idle core, nodes/s, equal-CPU-time match vs the python engine.
5. Partly. Harness games so far only checked the protocol (wins vs greedy as White, "init"
   losses as Black from cold numba compiles under load, since fixed by the disk cache).
   Real arena runs need a quieter machine; results go in `RESULTS.md`.
6. Open. Handcrafted vs DeepChess scalar vs blend (`tools/selfplay.py`). Known weakness:
   the network saturates in very lopsided positions (all moves in a K+2R vs K position
   score +835), so conversions rely on the mate search; a blend with the handcrafted PST
   is the candidate fix and is what the self-play comparison is for.
