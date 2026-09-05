# Rule-Based Chess Engine Implementation Roadmap

This folder contains implementation-ready README files for building a classical, rule-based chess engine without reinforcement learning or neural networks.

## Recommended Implementation Order

1. `01_negamax.md`
2. `02_alpha_beta_pruning.md`
3. `03_handcrafted_evaluation.md`
4. `04_quiescence_search.md`
5. `05_iterative_deepening.md`
6. `06_move_ordering.md`
7. `07_transposition_table_zobrist.md`
8. `08_killer_heuristic.md`
9. `09_history_heuristic.md`
10. `10_principal_variation_search.md`
11. `11_null_move_pruning.md`
12. `12_late_move_reductions.md`
13. `13_static_exchange_evaluation.md`
14. `14_futility_pruning_and_razoring.md`
15. `15_selective_extensions.md`

## General Rules for the Coding Agent

- Preserve all existing engine interfaces unless a change is necessary.
- Keep search logic separate from board representation and evaluation logic.
- Never generate illegal moves.
- Always restore the board exactly after simulating a move.
- Treat checkmate, stalemate, repetition, the fifty-move rule, and insufficient material correctly if the board library exposes them.
- Prefer deterministic behavior unless randomness is explicitly requested.
- Add unit tests or regression tests for each feature.
- Benchmark every search optimization by tracking:
  - nodes searched
  - depth reached
  - elapsed time
  - chosen move
  - legality of result
- New optimizations must not change the correctness of forced tactical results unless they are intentionally selective pruning methods.
- Add configuration flags for advanced pruning features so they can be enabled or disabled independently.
- If the engine uses `python-chess`, prefer its legal move generation, push/pop operations, FEN support, and game-over detection rather than reimplementing chess rules.

## Recommended Engine Architecture

```text
agent.py
  |
  +-- search.py
  |     +-- iterative deepening
  |     +-- negamax
  |     +-- alpha-beta / PVS
  |     +-- quiescence
  |     +-- pruning / reductions
  |
  +-- evaluation.py
  |     +-- material
  |     +-- piece-square tables
  |     +-- mobility
  |     +-- pawn structure
  |     +-- king safety
  |
  +-- ordering.py
  |     +-- TT move
  |     +-- captures / SEE
  |     +-- killer moves
  |     +-- history heuristic
  |
  +-- transposition.py
  |     +-- Zobrist hash
  |     +-- TT entries
  |
  +-- tests/
```

## Search Score Convention

Use a score from the perspective of the player to move.

Recommended centipawn scale:

```text
Pawn   = 100
Knight = 320
Bishop = 330
Rook   = 500
Queen  = 900
Mate   = 100000 or larger
```

A positive score means the side to move is better. A negative score means the side to move is worse.

## Important

Implement one stage at a time and run the arena after every stage. If a new optimization makes the bot weaker or introduces illegal moves, disable that feature and debug before continuing.
