# 21_nnue — implementation plan

Self-trained NNUE evaluator inside an own alpha-beta/PVS search, written for the
AI Chessathon contract `get_move(fen, time_left_ms) -> uci`.

## How the harness drives us (from `harness/`)

- `runner.py` does `import agent` once per game (90 s budget, `rules.INIT_BUDGET_S`), then
  reads JSON lines `{"fen", "time_left_ms"}` from stdin and writes `{"move": uci}`.
- `referee.py` charges wall time for the whole call; over the clock (+500 ms watchdog
  grace) is a flag loss. Illegal UCI is a loss. 300 plies -> material adjudication.
- Only a FEN is sent: no move history. Repetition memory must be kept in the module.
- Arena: `python -m harness.arena --agent agents/21_nnue --opponent baselines/<x> --games N`
  at 10 s + 0.1 s by default; colours alternate every game.

## Architecture

```
FEN -> python-chess (legal fallback move, first)            [agent.py]
    -> own bitboard position in numpy arrays                 [cboard.py, numba]
    -> iterative deepening PVS, TT, killers, history, NMP,
       LMR, qsearch w/ delta pruning, check extension        [csearch.py, numba]
    -> leaf eval = NNUE (768 -> H)x2 -> 1, int16 accumulator
       updated incrementally in make/unmake                  [nnue.py, numba]
    -> best move validated against python-chess legal moves  [agent.py]
```

Feature set: per perspective (side to move / opponent), one input per
(piece type 6 x colour-relative-to-perspective 2 x square 64) = 768 sparse inputs,
squares mirrored vertically for the black perspective. Hidden H = 256 (tuned in
Phase E), clipped ReLU, one linear output layer over the concatenated
[acc(stm), acc(nstm)]. This is the classic small NNUE shape. King-bucketed
inputs are an optional later experiment if data volume allows.

Quantisation: feature weights/biases int16 (scale 64 = 1.0), output weights int16,
output scaled to centipawns. Stored as `.safetensors` written/read by our own tiny
reader (no torch import at runtime).

## Training pipeline (`training/`)

1. `datagen.py`: self-play from random openings (random 4-12 plies then engine play
   with noise) using Stockfish (offline teacher, `training/teacher/`, never shipped),
   plus random-perturbed positions; sample positions, skip in-check positions; label
   with Stockfish at fixed depth; store FEN + cp + WDL-ish result.
2. `train.py`: PyTorch CPU, sparse inputs, sigmoid(cp/scale) target, MSE loss,
   Adam, checkpoints to `models/`.
3. `export.py`: quantise and write `weights/nnue.safetensors`; write provenance
   (`models/PROVENANCE.md`).

## Phases

1. Minimal legal agent (python-chess material alpha-beta) — done first so the
   contract is always satisfied.
2. Numba board + movegen (perft-tested against python-chess), search, PSQT eval.
3. NNUE inference + training pipeline; incremental == full recompute tests.
4. Optimise: node rate, compile time inside 90 s, time management.
5. Benchmark with the harness (paired colours) vs baselines and the team's
   `my-agents/*` search bots.
6. Find weaknesses (flagging, repetition draws, weak endgames), fix, re-measure.

## Files

- `agent.py` entry point; `cboard.py`, `csearch.py`, `nnue.py` numba modules;
  `weights/` shipped model; `training/`, `models/`, `tests/`; `RESULTS.md` numbers.
- `training/teacher/` holds the Stockfish binary used only for labelling; it is
  excluded from the submission (`harness/package.py` only ships root `*.py` and
  `weights/`).

## Status log (checkpoint for a fresh session; newest first)

### 2026-09-05 (session 3)

- Found on disk: engine (`cboard.py`, `csearch.py`, `nnue.py`, `agent.py`), tests, training
  scripts, and 865k Stockfish-labelled positions in `training/data/positions_sf6k_*.txt`
  (datagen ran with `--nodes 6000 --tag sf6k --workers 8`, stopped before its 1M target).
  No model had been trained and `weights/` was empty. Code on disk == branch
  `feature/agent-21-nnue` apart from CRLF line endings.
- `tests.test_perft` and `tests.test_nnue` pass (perft 4 kiwipete ~4.9 Mnps under heavy load).
- `tests.test_agent`: three tests used wrong positions (the "hanging queen" was not attackable,
  the "stalemate" FEN had no legal moves, and in `k7/8/1K6/8/8/8/8/1Q6 w` the own king on b6
  blocks Qb7, so the engine's Qh1+ Kb8 Qh8# is right). All three rewritten.
- Intermittent "Windows fatal exception: access violation" in the compiled search on the
  second process that loaded numba's on-disk cache (also seen by agent 20). Fix: `jitconf.py`
  makes `cache=` opt-in (`NNUE21_NUMBA_CACHE=1`), default off, matching the platform.
- Measured a fresh compile at ~118 s CPU on the loaded box (wall 15+ min). Too close to the
  90 s init budget, so `agent.py` now compiles in a background thread, waits up to 70 s at
  import, and answers with a pure python-chess alpha-beta (`_python_search`) until the engine
  is ready. Tests call `agent.wait_ready()`.
- Training: `train.py --hidden 256 --epochs 12 --threads 4 --out models/nnue_h256.pt`
  (log `models/train_h256.log`). Epoch 8: val MSE 0.0051, MAE 135 cp, sign accuracy 94.9%.
  The epoch-4 checkpoint is exported to `weights/nnue.safetensors` (396 KB) so integration
  runs can start; re-export from the final checkpoint when training ends.
- Quantised net sanity (stm centipawns): start +49, +queen +1008 / -918, KQ v K +678, symmetric
  under colour flip. ruff (dir `ruff.toml` ignores the `P` naming rules) and strict mypy clean.
- `variants/psqt/agent.py`: same engine with the PSQT eval, for A/B arena runs.
- Training finished: epoch 12 val MSE 0.00494, MAE 134 cp, sign accuracy 95.0%. Exported to
  `weights/nnue.safetensors` (|W1| max 91, worst-case accumulator 1931).
- Probe of the final net (stm cp, start = +31): removing a black pawn gives a7 0, b7 +91,
  c7 +61, d7 +54, e7 +48, f7 +151, g7 +123, h7 +9. Rook pawns are worth ~nothing and f/g
  pawns carry king-safety weight; material is compressed (queen ~1000, rook ~560, KQvK 600).
  Likely under-training / data volume (865k positions, 12 epochs). Second run started:
  `train.py --epochs 30 --threads 1 --lambda-result 0.0 --seed 2 --out models/nnue_h256_long.pt`
  (log `models/train_h256_long.log`); compare val loss and the pawn probe before swapping.
- `tools/selfplay_ab.py --games 40 --nodes 20000` (NNUE vs PSQT, epoch-4 weights loaded at
  start) running; log `results/ab_run.log`, summary `results/ab_nnue_vs_psqt.txt`.
- All 25 tests pass under the threaded agent.py (engine ready after 125 s CPU / 18 min wall on
  the loaded box; 120 s clock level used 4.2 s).
- A/B `tools/selfplay_ab.py --games 40 --nodes 20000` with the epoch-4 net vs PSQT:
  +28 =3 -9, 73.8%, about +180 Elo at equal nodes. But nodes/s: NNUE 46.8k vs PSQT 724k.
  Micro-benchmark: `evaluate` 8.7 us, `update` 1.85 us, `refresh` 39 us, make+unmake 2.4 us
  per move. `evaluate` was the hot spot (3-index access + branches per element); rewritten
  over 1-D row views with min/max, `copy_acc` as a row copy. Re-measure before trusting.
- `harness.play` vs greedy (120 s + 0.5 s): draw by threefold repetition, played entirely by
  the python fallback because the compile outlasted the game here. Fallback now scores root
  moves that repeat a seen position as 0 (`_position_key`, `_HISTORY_FENS`).
- nnue.py row-view rewrite is exact (`tests.test_nnue` pass) and committed. Second A/B (epoch-12
  net): +28 =4 -8, +191 Elo at 20k nodes, node rate still 48.7k vs 732k by wall clock. Profiled
  in CPU time (`prof_search.py`): NNUE 545 knps vs PSQT 826 knps, so the network costs 1.5x per
  node and the 15x was a wall-clock artefact of the swapping machine. selfplay_ab now reports
  CPU-time nps.
- 30-epoch run finished: val MAE 113.8 cp, sign 96.0%; exported to
  `models/nnue_h256_long.safetensors` (not yet shipped). Pawn probe is much saner (see RESULTS.md).
- Running: `selfplay_ab --weights models/nnue_h256_long.safetensors --weights-b weights/nnue.safetensors`
  (results/ab_run3.log), `selfplay_ab --movetime 0.1` shipped net vs PSQT (results/ab_run4.log),
  `tools/vs_bot.py --bot ../../my-agents/10_principal_variation_search --movetime 0.5`
  (results/vs_bot_run.log).
- Results: long net vs shipped net +17 =5 -18 (48.8%), so the epoch-12 net stays shipped.
  Shipped net vs PSQT at equal time (0.1 s/move): +23 =6 -11, +108 Elo. Engine vs
  my-agents/10_principal_variation_search at 0.5 s/move: +9 =0 -1. Game-replay profile: NNUE
  539-567 knps vs PSQT 708 knps in CPU time (1.3x per node); the A/B tool's CPU figures are
  unreliable for 0.1 s moves (15.6 ms process_time ticks).
- train.py now holds out whole games for validation (the old position split leaked adjacent
  plies, which is the likely reason the 30-epoch net's better MAE bought no strength).
- Final full test suite relaunched on the current code (scratch agent_tests5.txt).
- Next (for a fresh session): more training games (datagen with more workers when the box is
  free; 865k positions from ~12k games is the limiting factor), retrain with the game split and
  lambda 0.1, A/B vs weights/nnue.safetensors with tools/selfplay_ab.py --weights-b before
  shipping; consider H=384 or king-bucketed inputs once data exceeds a few million positions.
  Merge of feature/agent-21-nnue into main is pending (left to the user, see memory)., re-export final weights, A/B NNUE vs
  PSQT and vs baselines with `harness.arena`, record numbers in `RESULTS.md`, commit on
  `feature/agent-21-nnue` (throwaway index, see memory) and note "merge pending".
