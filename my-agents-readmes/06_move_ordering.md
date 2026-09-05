# 06 — Move Ordering

## Goal

Search promising moves first to increase Alpha-Beta cutoffs.

Move ordering should not change the theoretical result of a full-width search; it should reduce work.

## Recommended Priority

Use approximately:

```text
1. transposition-table move
2. winning captures
3. promotions
4. killer moves
5. quiet moves ordered by history score
6. losing captures
```

At the initial stage, before TT/SEE/history exist:

```text
1. captures
2. promotions
3. checks
4. quiet moves
```

## MVV-LVA

For captures, score:

```text
Most Valuable Victim
Least Valuable Attacker
```

Example ordering:

```text
pawn x queen
knight x rook
queen x pawn
```

## Move Scoring

Create a function:

```python
def move_order_score(board, move):
    ...
```

Then:

```python
moves = sorted(
    board.legal_moves,
    key=lambda move: move_order_score(board, move),
    reverse=True
)
```

Avoid expensive sorting if performance becomes an issue; scoring plus partial selection can be optimized later.

## Checks

Checking moves may be useful but should not automatically outrank strong captures in every position.

Use a moderate bonus.

## Promotions

Queen promotions should usually receive a very high score.

Underpromotions must still remain legal candidates.

## Instrumentation

Measure:

```text
beta_cutoffs
first_move_cutoffs
nodes
```

A good ordering system should increase first-move cutoffs.

## Tests

- TT move ranks first when supplied.
- queen capture ranks above pawn capture.
- promotion receives high priority.
- every legal move remains present exactly once.
- no illegal moves appear.
- same final score as unordered Alpha-Beta at fixed depth.

## Done Criteria

- Search node count decreases.
- No moves are dropped.
- Ordering logic is modular enough to add TT, SEE, killer, and history scores later.
