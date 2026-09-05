# 12 — Late Move Reductions (LMR)

## Goal

Search later, low-priority moves at reduced depth because good move ordering makes them less likely to be best.

If a reduced search unexpectedly looks promising, re-search at full depth.

## Basic Conditions

Consider reducing a move when:

- move index is sufficiently late
- depth is sufficiently large
- move is quiet
- side is not in check
- move is not a major tactical move
- move is not the current principal variation move

## Example

Instead of:

```text
all 25 moves at depth 8
```

search approximately:

```text
early promising moves -> depth 8
late quiet moves      -> depth 6 or 7
```

## Algorithm

Conceptually:

```python
for move_index, move in enumerate(moves):
    reduction = 0

    if should_reduce(move, move_index, depth):
        reduction = compute_reduction(move_index, depth)

    board.push(move)

    score = -search(
        board,
        depth - 1 - reduction,
        ...
    )

    if reduction > 0 and score > alpha:
        score = -search(
            board,
            depth - 1,
            ...
        )

    board.pop()
```

## Conservative First Version

Start with:

```text
depth >= 3 or 4
move_index >= 4
quiet moves only
reduction = 1 ply
```

Tune later.

## Do Not Reduce

Initially avoid reducing:

- captures
- promotions
- checks
- TT move
- killer moves
- moves while in check
- moves near mate scores

## Tests

- early moves are full depth
- late quiet moves receive reduction
- promising reduced move is re-searched
- tactical captures are not reduced
- best move remains stable on tactical suites

## Done Criteria

- Reduced node count.
- Tactical strength does not collapse.
- Re-search protects unexpectedly strong moves.
- Reduction logic is isolated and configurable.
