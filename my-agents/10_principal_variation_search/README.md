# 10 — Principal Variation Search

Implements `my-agents-readmes/10_principal_variation_search.md`.

## Files

- `agent.py` — the bot. Exposes `get_move(fen, time_left_ms) -> str` as the harness requires.
  Self-contained: evaluation, ordering, transposition table, quiescence, iterative deepening
  and the PVS search all live here so the file can be copied to the repo root and zipped as-is.
- `test_pvs.py` — the spec's test list plus a node-count benchmark. Run from the repo root:

  ```
  uv run python my-agents/10_principal_variation_search/test_pvs.py
  ```

## What PVS does here

In `Searcher.negamax` and `Searcher.search_root` the first move of every node is searched with
the full `(alpha, beta)` window. Every later move is first searched with the null window
`(alpha, alpha + 1)`. Scores are integer centipawns, so that window is one point wide and can only
answer "worse than alpha" or "better than alpha". Only when the answer is "better" and the score
also sits below beta is the move re-searched with the full window to get its exact value.

Counters on the `Searcher` (`null_window_searches`, `researches`, `nodes`) make the behaviour
observable in tests and in the per-depth log the bot prints to stderr.

## Prerequisites the spec assumes, also implemented

- fail-soft alpha-beta negamax with mate-distance pruning
- material + piece-square evaluation (separate king table for endgames)
- quiescence search over captures and promotions, with checkmate detection
- iterative deepening on a wall-clock budget; depth 1 always completes
- move ordering: TT move, promotions, MVV-LVA captures, two killers per ply, history
- transposition table keyed by python-chess's transposition key (bitboards, side to move,
  castling and en-passant state; ~25x cheaper than a polyglot Zobrist hash), mate scores
  normalised by ply, principal variation reconstructed from TT best moves
- two-fold repetition, fifty-move and insufficient-material draws inside the search, plus
  positions already seen in the game (module state survives between moves)

## Switches

`Config` in `agent.py` has one flag per feature. `Config(use_pvs=False)` gives plain alpha-beta
with everything else unchanged, which is what the tests compare against.

## Play it

```
uv run python -m harness.arena --agent my-agents/10_principal_variation_search --opponent baselines/greedy --games 2
uv run python -m harness.play --white my-agents/10_principal_variation_search --black baselines/minimax
```
