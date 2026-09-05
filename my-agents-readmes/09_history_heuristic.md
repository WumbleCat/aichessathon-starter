# 09 — History Heuristic

## Goal

Rank quiet moves based on how often they have historically caused useful Alpha-Beta cutoffs.

Unlike killer moves, history scores are not restricted to a single ply.

## Data Structure

Simple version:

```text
history[color][from_square][to_square]
```

Alternative:

```text
history[piece][to_square]
```

Use one consistent design.

## Updating

When a quiet move causes a beta cutoff, increase its history score.

Typical depth-weighted bonus:

```python
history[...] += depth * depth
```

Optionally penalize previously searched quiet moves that failed to cause the cutoff.

## Move Ordering

For quiet moves:

```text
higher history score -> search earlier
```

Use after TT move and killer moves.

## Aging

History scores can grow indefinitely.

Periodically decay:

```python
history //= 2
```

or clamp within a safe range.

## Do Not Use For

Initially avoid applying history scoring to:

- captures
- promotions

These should have tactical ordering mechanisms.

## Tests

- quiet cutoff increases history value
- higher-history quiet move sorts first
- captures remain governed by capture ordering
- history table can be reset
- same search result as Alpha-Beta without history

## Done Criteria

- Quiet move ordering becomes informed by previous search outcomes.
- Scores do not overflow or grow without bound.
- Node count improves on representative positions.
