# 02 — Alpha-Beta Pruning

## Goal

Add alpha-beta pruning to Negamax so the engine returns the same minimax result while searching fewer nodes.

## Core Idea

Maintain a search window:

```text
alpha = best score already guaranteed by current player
beta  = score the opponent can already force us below
```

If:

```text
alpha >= beta
```

stop searching the remaining moves at that node.

## Negamax Form

Implement:

```python
def negamax(board, depth, alpha, beta):
    if depth == 0 or board.is_game_over():
        return evaluate(board)

    best = -INF

    for move in board.legal_moves:
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha)
        board.pop()

        best = max(best, score)
        alpha = max(alpha, score)

        if alpha >= beta:
            break

    return best
```

The window inversion is essential:

```python
-beta, -alpha
```

Do not pass `alpha, beta` unchanged to the child.

## Root Search

At the root:

```text
alpha = -INF
beta  = +INF
```

Track the best move separately.

## Instrumentation

Add counters:

```text
nodes_searched
beta_cutoffs
```

Reset counters before every root search.

## Verification

Compare plain Negamax and Alpha-Beta at identical depth.

They should:

- return the same best move in deterministic positions
- return the same score
- Alpha-Beta should search fewer or equal nodes

## Edge Cases

Do not prune before:

- legal move generation
- terminal-state handling
- restoring the board

Always `pop()` before breaking after a cutoff.

## Tests

- Alpha-Beta score equals plain Negamax score at depths 1–4.
- Find a tactical move at the same depth.
- Count nodes and verify pruning occurs.
- Verify board state is unchanged after cutoff.
- Verify checkmate scores propagate correctly.

## Done Criteria

- Same search result as Negamax.
- Significantly fewer nodes on representative positions.
- No illegal moves.
- No board corruption.
