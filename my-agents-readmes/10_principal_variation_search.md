# 10 — Principal Variation Search (PVS)

## Goal

Optimize Alpha-Beta under the assumption that the first well-ordered move is probably best.

Search the first move with the full window, then later moves with a null/narrow window.

## Core Logic

For each node:

1. Search first move normally.
2. For subsequent moves, search with a narrow window.
3. If the narrow-window search indicates the move may improve alpha, re-search it with the full window.

Conceptually:

```python
for i, move in enumerate(moves):
    board.push(move)

    if i == 0:
        score = -search(board, depth - 1, -beta, -alpha)
    else:
        score = -search(board, depth - 1, -alpha - 1, -alpha)

        if alpha < score < beta:
            score = -search(board, depth - 1, -beta, -alpha)

    board.pop()
```

For non-integer score systems, use an appropriately tiny window.

## Preconditions

Do not implement PVS before:

- Alpha-Beta is correct
- move ordering is reasonably good

Poor ordering can cause excessive re-searches.

## Principal Variation

Track the best move at each node and optionally reconstruct the PV.

## TT Integration

The TT move should be searched first when legal.

## Tests

- PVS score equals Alpha-Beta score at same depth.
- PVS best move equals Alpha-Beta best move in deterministic tests.
- re-search happens when a later move exceeds alpha
- no re-search when null-window search fails low
- board restored correctly

## Done Criteria

- Correctness matches Alpha-Beta.
- Search is equal or faster on well-ordered positions.
- Feature can be disabled for comparison.
