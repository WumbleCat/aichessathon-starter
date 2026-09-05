# 05 — Iterative Deepening

## Goal

Search progressively deeper:

```text
depth 1
depth 2
depth 3
...
```

rather than searching only the final depth.

## Root Loop

Conceptually:

```python
best_move = None

for depth in range(1, max_depth + 1):
    score, move = search_root(board, depth)

    if move is not None:
        best_move = move

return best_move
```

## Why

Iterative deepening provides:

- a valid move if time expires
- better move ordering from previous iterations
- principal variation information
- better transposition table usage

## Time Management

If the harness gives a time budget:

1. Record search start time.
2. Check time periodically.
3. Abort the current iteration safely before exceeding the limit.
4. Return the best move from the last fully completed iteration.

Do not return a partially searched root result unless explicitly designed to.

## Previous Best Move

Search the previous iteration's best move first in the next iteration.

Later, integrate with transposition-table move ordering.

## Principal Variation

Store the best line found at each iteration when practical.

Useful debugging output:

```text
depth=5
score=+37
nodes=128441
time=0.82s
pv=e2e4 e7e5 g1f3 ...
```

## Mate Handling

If a forced mate is proven, optionally stop deepening.

## Tests

- Depth 1 result exists.
- Depth increases sequentially.
- Search can stop after any completed iteration.
- Best move remains legal.
- Search statistics report depth reached.
- With sufficient time, result matches fixed-depth search.

## Done Criteria

- Last completed depth always provides the returned move.
- Works with fixed-depth mode and timed mode.
- Previous iteration improves move ordering.
