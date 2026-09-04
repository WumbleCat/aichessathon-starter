# 01 — Negamax

Stage 01 of the rule-based engine roadmap in `my-agents-readmes/01_negamax.md`: one recursive
search function, no pruning, no ordering, no quiescence, no transposition table.

## What is here

| File | Purpose |
|---|---|
| `agent.py` | The bot. `get_move(fen, time_left_ms)` as the harness requires. |
| `test_agent.py` | Unit tests for the spec's cases and a few edge cases. |

## How it works

- `negamax(board, depth, ply)` scores the side to move. Terminal states are scored before
  static evaluation at every node: mate is `-MATE_SCORE + ply`, stalemate and rule draws are 0.
- `search_root(board, depth)` searches every legal move and returns `(move, score)`, or
  `(None, terminal score)` when no legal move exists. Ties keep generation order, so the same
  input gives the same output.
- `evaluate(board)` is material plus a small attack-count mobility term, from the side to
  move's view. Stage 03 replaces it.
- Depth is chosen per move from the clock: the deepest depth whose estimated node count
  (root branching factor to that power, at the node cost measured at import) fits one
  thirtieth of the remaining time, capped at 3. A hard deadline in the root loop stops the
  search early rather than flagging.
- Positions already reached this game are remembered, and a root move that returns to one
  is scored as a draw, since the search board has no history to see repetition otherwise.

## Run

```
uv run python -m unittest discover -s my-agents/01_negamax -v
uv run python -m harness.arena --agent my-agents/01_negamax --opponent baselines/greedy --games 10
uv run python -m harness.play --white my-agents/01_negamax --black baselines/minimax
```

## Results at 10 s + 0.1 s

| Opponent | Games | Score |
|---|---|---|
| baselines/random | 2 (5 s base) | +2 =0 -0 |
| baselines/greedy | 10 | +10 =0 -0 |
| baselines/minimax | 6 | +5 =0 -1 |

## Known limitations

- Roughly 13k nodes/s in pure Python, so depth 3 is affordable only in narrow positions and
  the middlegame is mostly searched at depth 2. Alpha-beta (stage 02) is what buys depth.
- No quiescence: leaf scores mid-exchange are wrong, which stage 04 addresses.
- Repetition is only seen at the root, from the game history the bot keeps itself.
