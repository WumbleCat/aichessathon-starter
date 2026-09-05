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
