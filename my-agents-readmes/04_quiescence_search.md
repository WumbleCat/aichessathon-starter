# 04 — Quiescence Search

## Goal

Avoid the horizon effect by extending tactical positions after the normal depth reaches zero.

Instead of evaluating immediately at depth 0, call quiescence search.

## Basic Flow

```python
def quiescence(board, alpha, beta):
    stand_pat = evaluate(board)

    if stand_pat >= beta:
        return beta

    if stand_pat > alpha:
        alpha = stand_pat

    for move in tactical_moves(board):
        board.push(move)
        score = -quiescence(board, -beta, -alpha)
        board.pop()

        if score >= beta:
            return beta

        if score > alpha:
            alpha = score

    return alpha
```

## Tactical Moves

Initially search:

- captures
- promotions

Optionally later add:

- checking moves
- tactical evasions

Do not blindly include every check if it causes search explosions.

## Critical Rule: In Check

If the side to move is in check, `stand_pat` is invalid.

When in check:

- generate all legal evasions
- do not allow the engine to simply evaluate without escaping check

## Move Ordering

Order captures before searching them.

Recommended future integration:

- SEE
- MVV-LVA

## Delta Pruning

Do not implement initially.

First make quiescence correct.

## Integration

Replace:

```python
if depth == 0:
    return evaluate(board)
```

with:

```python
if depth == 0:
    return quiescence(board, alpha, beta)
```

## Tests

Use positions where:

- queen captures defended rook
- apparent winning capture loses material one move later
- forcing recapture sequence is evaluated correctly
- side in check must search evasions
- quiescence terminates
- board state remains unchanged

Track:

```text
qnodes
```

separately from normal search nodes.

## Done Criteria

- Tactical leaf positions are searched until reasonably quiet.
- No infinite tactical recursion.
- In-check nodes are handled legally.
- Basic tactical blunders decrease.
