# 01 — Negamax

## Goal

Replace separate maximizing/minimizing minimax logic with a single recursive search function based on the zero-sum identity:

```text
score(position for me) = -score(position for opponent)
```

Negamax should become the core recursive search function used by later features.

## Requirements

Implement a function conceptually equivalent to:

```python
def negamax(board, depth):
    if depth == 0 or board.is_game_over():
        return evaluate_terminal_or_static(board)

    best_score = -INF

    for move in board.legal_moves:
        board.push(move)
        score = -negamax(board, depth - 1)
        board.pop()

        best_score = max(best_score, score)

    return best_score
```

At the root, return both the score and the best move.

## Terminal Positions

Handle terminal states before static evaluation.

Recommended scoring:

```text
checkmate against side to move = -MATE_SCORE
stalemate                     = 0
draw                           = 0
```

If possible, adjust mate scores by ply so the engine prefers faster mates:

```text
winning mate score = MATE_SCORE - ply
losing mate score  = -MATE_SCORE + ply
```

## Board Restoration

Every `push(move)` must have a matching `pop()`.

The board after search must be identical to the board before search.

## Root Search

The root function should:

1. Generate legal moves.
2. Search each move using negamax.
3. Track the highest score.
4. Return the corresponding move.
5. Return `None` if no legal move exists.

Example:

```python
best_move = None
best_score = -INF

for move in board.legal_moves:
    board.push(move)
    score = -negamax(board, depth - 1)
    board.pop()

    if score > best_score:
        best_score = score
        best_move = move
```

## Evaluation Convention

The evaluation function must return a score relative to the side to move, or the search must convert a White-centric score into side-to-move perspective consistently.

Do not mix conventions.

## Tests

Test at least:

- starting position returns a legal move
- one-move checkmate is found
- a hanging queen is captured
- stalemate returns zero
- checkmate returns a large negative score for the side to move
- board FEN is unchanged after the search

## Done Criteria

Negamax is complete when:

- it always returns legal moves
- terminal states are correctly scored
- search works for configurable depths
- the board is never corrupted
- results are deterministic for identical inputs
