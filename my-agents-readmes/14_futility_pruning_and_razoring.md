# 14 — Futility Pruning and Razoring

## Goal

Avoid expensive searches near the horizon when the static evaluation indicates a move or node is extremely unlikely to affect Alpha-Beta bounds.

These are selective pruning techniques and must be added conservatively.

# Part A — Futility Pruning

## Idea

Near leaf nodes, if:

```text
static_eval + optimistic_margin <= alpha
```

then quiet moves are unlikely to raise alpha.

Skip selected quiet moves.

## Recommended Conditions

Only use when:

- depth is small, such as 1–3
- side is not in check
- move is quiet
- move is not a promotion
- move is not a checking move
- score is not near mate values

Example:

```python
if (
    depth <= 2
    and not in_check
    and quiet_move
    and static_eval + margin[depth] <= alpha
):
    continue
```

Use conservative margins initially.

# Part B — Reverse Futility Pruning

## Idea

If:

```text
static_eval - margin >= beta
```

the position may already be so strong that deeper search is unnecessary.

Return a beta cutoff under safe conditions.

Use only at shallow depth initially.

# Part C — Razoring

## Idea

At low depth, if static evaluation is far below alpha, skip or reduce the full normal search and go directly toward quiescence.

Conceptually:

```python
if depth <= razor_depth:
    if static_eval + razor_margin < alpha:
        return quiescence(...)
```

## Safety Rules

Do not apply near:

- check
- mate scores
- promotions
- highly tactical positions
- PV nodes initially

## Tests

- feature off gives baseline result
- quiet hopeless branches prune
- checks are not futility-pruned
- promotions are not pruned
- tactical test suite does not regress badly
- node count decreases

## Done Criteria

- Configurable margins.
- All pruning can be toggled.
- Tactical correctness remains acceptable.
- Node reduction is measurable.
