# Giraffe-style learned evaluation: implementation plan

Agent directory: `agents/30_giraffe`. Entry point `agent.py` exposes
`get_move(fen, time_left_ms) -> str`.

## What the harness does

- `harness/runner.py` puts the agent directory first on `sys.path`, does `import agent`
  (90 s init budget, weights load here), then loops: JSON `{fen, time_left_ms}` in,
  `{move}` out. The process lives for one game, so module state survives between moves.
- `harness/referee.py` charges wall time, adds the increment after the move, and loses the
  game on flag (with a 500 ms watchdog grace), illegal move, or crash. Only the FEN is sent,
  never the move list, so repetition detection needs our own memory.
- `harness/arena.py` alternates colours from the standard start (`--games`, `--base-ms`,
  `--increment-ms`); `harness/play.py` plays one game, optionally from `--fen`.

## Architecture (Giraffe, adapted for one Python core)

1. **Feature representation** (`giraffe_eval.py`, numba): board normalised so the side to
   move is always "us" (vertical flip + colour swap for black). Three feature groups, as in
   the paper: global (castling rights, material counts, en passant), piece-centric (32 fixed
   piece slots with presence, normalised file/rank, lowest-valued attacker and defender,
   mobility) and square-centric (lowest-valued attacker of each square by each side).
   Total 335 floats, computed straight from bitboards.
2. **Network**: Giraffe's three-group first hidden layer (global 15->16, pieces 192->128,
   squares 128->64), merged 208->64->32->1, ReLU inside, `tanh` output scaled to
   centipawns. About 48k weights, evaluated in numba in ~20 us. Weights ship as a flat
   float32 `.npz` in `models/`, trained from random initialisation by this project.
3. **Search** (`giraffe_search.py`, python-chess): iterative deepening alpha-beta with
   PVS, transposition table, MVV-LVA / killer / history ordering, check extension, null
   move, late move reductions, and a capture quiescence with delta pruning. The evaluator
   is a plug-in so the same search runs with the handcrafted PST evaluation (control) or
   the network (experiment). Mate, stalemate and repetition scores are exact and outside
   the network.
4. **Training** (`training/`):
   - `gen_positions.py`: self-play positions with noise, labelled by a quiescence-resolved
     handcrafted evaluation (Giraffe's material bootstrap stage).
   - `bootstrap.py`: supervised pretraining of the torch model, export to `models/`.
   - `tdleaf.py`: TD-Leaf(lambda) self-play refinement using the engine's own search and
     a replay buffer; checkpoints are gated by an arena against the previous one.
5. **Tests** (`tests/`): legality across move types, clock tests down to 50 ms, feature
   symmetry (flipping the board must negate the value), search sanity (mate in 1/2).

## Time management

Budget per move ~ `time_left / 30 + 0.6 * increment`, capped, with the clock checked every
128 nodes. Depth 1 runs unclocked so a legal move always exists before the deadline is armed;
a fallback move is chosen before any search starts.

## Phases

1. Minimal legal agent (HCE + shallow search) and smoke tests.
2. Numba feature extractor + network forward + handcrafted control evaluator.
3. Full search, tests.
4. Data generation, supervised bootstrap, TD-Leaf refinement.
5. Arena benchmarks: net vs HCE with identical search, both vs baselines; RESULTS.md.
