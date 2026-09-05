# 15 — Selective Extensions

## Goal

Search critical positions deeper than the nominal depth when tactical or strategically forcing circumstances justify additional calculation.

Extensions do the opposite of pruning.

## Possible Extensions

Implement carefully and incrementally:

### Check Extension

If a move gives check or the side to move is in check, extend selected lines by one ply.

Do not extend every checking sequence indefinitely.

### Recapture Extension

Extend when a move immediately recaptures on the same square as the previous capture.

### Passed-Pawn Extension

Extend advanced passed pawns near promotion.

### Singular Extension

Advanced feature.

If the transposition-table best move appears much stronger than all alternatives, extend that move.

Do not implement singular extensions until TT, PVS, and stable search are working.

## Extension Budget

Prevent runaway search.

Track total extensions along a line.

Example:

```text
maximum extension budget = 2 plies
```

or apply fractional/limited rules.

## Suggested Initial Version

Start only with a conservative check extension:

```python
extension = 0

if move_gives_check and depth >= some_threshold:
    extension = 1

new_depth = depth - 1 + extension
```

Then evaluate node growth before adding more.

## Risks

Too many extensions can cause:

- exponential node explosion
- inconsistent time management
- repeated checking loops
- reduced effective search elsewhere

## Tests

- forcing check line receives additional depth
- quiet line does not
- extension budget prevents unlimited growth
- mate combinations become easier to find
- search still terminates under time control

## Done Criteria

- Extensions are selective and bounded.
- Node growth remains manageable.
- Forcing tactical positions improve.
- Each extension type can be toggled independently.
