# 13 — Static Exchange Evaluation (SEE)

## Goal

Estimate the material result of a capture sequence on one square without running a full normal search.

SEE is useful for:

- capture ordering
- identifying obviously losing captures
- pruning poor tactical moves in quiescence/search

## Example

Position contains:

```text
White queen captures defended pawn
Black pawn captures queen
```

SEE should detect that the initial capture loses material.

## High-Level Algorithm

For a target square:

1. Identify the value of the captured piece.
2. Identify all attackers of the target square.
3. Simulate exchanges using the least valuable attacker first.
4. Alternate sides.
5. Calculate the best material outcome assuming each side may stop exchanging when beneficial.

Typical attacker order:

```text
pawn
knight
bishop
rook
queen
king
```

## Output

Return a centipawn estimate.

Examples:

```text
+500 -> favorable exchange
0    -> roughly neutral
-700 -> losing capture
```

## Complications

SEE must account for:

- x-ray attacks after pieces are removed
- pinned pieces
- promotions
- en passant
- king legality where relevant

Start with a correct simpler implementation before aggressive pruning use.

## Integration

Capture move ordering:

```text
SEE >= 0 -> good capture
SEE < 0  -> bad capture
```

Suggested ordering:

```text
TT move
good captures by SEE
promotions
killers
history quiets
bad captures
```

## Tests

Create small tactical positions:

- pawn takes queen -> strongly positive
- queen takes defended pawn -> strongly negative
- equal pawn exchange -> near zero
- rook exchange sequence
- discovered/x-ray attacker case

## Done Criteria

- SEE provides sensible exchange estimates.
- Does not modify the real board state.
- Improves capture ordering.
- Do not use SEE pruning aggressively until validated.
