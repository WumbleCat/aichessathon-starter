# 19 Handcrafted Evaluation — implementation plan

Spec: `my-agents-readmes/19_HANDCRAFTED_EVALUATION.md`. The deliverable is a fast, tapered,
handcrafted evaluation that a Python alpha-beta search can call tens of thousands of times per
move, wrapped in a competent search so that its quality turns into Elo.

## How the harness calls us

- `harness/runner.py` puts our directory first on `sys.path`, does `import agent`, and then for
  every JSON request line calls `agent.get_move(fen, time_left_ms)`; the reply is one UCI string.
- Only a FEN arrives, never the move list, so repetition/fifty-move memory has to live in module
  state (the process lives for one game).
- The referee charges wall time; overshooting `time_left_ms` + 500 ms grace loses on flag.
- Init budget 90 s: warm every numba function at import time.
- Fast arena is 10 s + 0.1 s per move, so deadline checks must be fine-grained (every ~256 nodes).

## Files

| File | Role |
| --- | --- |
| `agent.py` | `get_move`; time management; repetition memory; legal fallback move first. |
| `hce_eval.py` | Pure-Python reference evaluation working on python-chess bitboards. Source of truth for the feature set and the unit tests. |
| `hce_fast.py` | numba port of the same evaluation (uint64 bitboards in, int out). Verified against `hce_eval.py` on thousands of positions. Falls back to the Python version if numba is unavailable. |
| `hce_search.py` | Iterative deepening, PVS alpha-beta, TT, MVV-LVA/killer/history ordering, quiescence with delta pruning, null move, check extension. |
| `tests/` | unittest suites: evaluation properties, mandatory chess tests, clock tests, equivalence numba vs python. |
| `bench.py` | nodes/s, depth, eval calls, eval time, arena driver with paired openings. |

## Evaluation design (perspective: White minus Black, search flips by side to move)

Two accumulators MG/EG, phase from non-pawn material (knight 1, bishop 1, rook 2, queen 4, total 24),
`score = (mg * phase + eg * (24 - phase)) // 24`.

1. Material (separate MG/EG values).
2. Piece-square tables, MG and EG, mirrored for Black.
3. Mobility: safe squares per piece type (not occupied by own pieces, not attacked by enemy pawns),
   piece-type-specific weights.
4. Pawn structure: doubled, isolated, backward; cached per pawn-bitboard pair (pawn hash).
5. Passed pawns: rank-scaled bonus (MG/EG), extra for protected/connected, king-distance in EG.
6. King safety: pawn shield in front of the king, open/semi-open files near the king, attack
   units from enemy pieces hitting the king zone; scaled out by phase.
7. Bishop pair, rook on open/semi-open file, rook on seventh, knight outposts.
8. Endgame king activity: EG king PST + mop-up (drive the lone king to the edge) when one side
   has no pawns and a decisive material edge.
9. Tapered interpolation. Tempo bonus for the side to move.

The evaluator never returns mate scores; terminal detection is the search's job.

## Phases

1. Minimal legal `get_move` (done first, kept as fallback path).
2. Full evaluation in Python + search.
3. Tests (spec tests, mandatory chess tests, clock tests).
4. numba evaluation, incremental material/PST tracking if profiling says so.
5. Benchmarks with the harness (paired colours, same openings).
6. Weakness hunting from lost games; iterate.
