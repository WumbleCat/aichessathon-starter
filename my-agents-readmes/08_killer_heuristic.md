# 08 — Killer Move Heuristic

## Goal

Improve move ordering for quiet moves that caused beta cutoffs at the same search depth or ply.

The idea:

> A quiet move that refuted one line may refute another line at the same ply.

## Data Structure

Maintain typically two killer moves per ply:

```python
killers[ply][0]
killers[ply][1]
```

## When to Store

If a move:

- causes a beta cutoff
- is not a capture
- is not normally a promotion

store it as a killer for that ply.

Example:

```python
if score >= beta:
    if not board.is_capture(move):
        store_killer(move, ply)
    break
```

## Replacement

If new move differs from killer 0:

```text
killer 1 = killer 0
killer 0 = new move
```

## Move Ordering

Suggested priority:

```text
TT move
good captures
promotions
killer 0
killer 1
history quiet moves
bad captures
```

Do not remove the move from the legal move list.

Only change its score/order.

## Scope

Killers are indexed by ply, not board position.

Reset between root searches if desired, or preserve cautiously.

## Tests

- quiet beta-cutoff move is stored
- capture cutoff does not become killer
- same move is not duplicated in both killer slots
- killer receives ordering bonus
- final Alpha-Beta result is unchanged

## Done Criteria

- Killer moves measurably improve move ordering or reduce nodes.
- No legality changes.
- Feature can be enabled/disabled independently.
