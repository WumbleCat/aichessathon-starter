# 11 — Null-Move Pruning

## Goal

Prune positions that are so strong that even giving the opponent an extra move still produces a beta cutoff.

This is a selective pruning technique.

## Core Idea

When safe:

1. Temporarily pass the turn without moving a piece.
2. Search the opponent at reduced depth.
3. If the result is still >= beta, prune the current node.

Conceptually:

```python
if can_try_null_move(board, depth):
    board.push(chess.Move.null())

    score = -search(
        board,
        depth - 1 - R,
        -beta,
        -beta + 1
    )

    board.pop()

    if score >= beta:
        return beta
```

Typical reduction:

```text
R = 2 or 3
```

Start conservatively.

## When NOT to Use

Disable null-move pruning when:

- side to move is in check
- depth is too small
- position is a likely zugzwang
- very little non-pawn material remains
- null moves are not safely supported by the board implementation

## Zugzwang Risk

Null-move pruning can be wrong in positions where being forced to move is a disadvantage.

This commonly occurs in:

- king-and-pawn endings
- sparse endgames

Use material-based safeguards.

## Verification Search

Advanced versions may perform verification search before accepting certain null-move cutoffs.

Do not add this until the basic method is stable.

## Tests

- no null move while in check
- no null pruning in king-and-pawn-only endgames
- board restores correctly after null move
- tactical middlegame node count decreases
- known zugzwang test positions are not incorrectly pruned

## Done Criteria

- Feature can be toggled.
- Significant node reduction in middlegames.
- Conservative safeguards prevent obvious zugzwang errors.
