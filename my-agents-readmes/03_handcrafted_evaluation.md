# 03 — Handcrafted Evaluation Function

## Goal

Create a deterministic, rule-based position evaluation function measured in centipawns.

No machine learning or trained weights.

## Recommended Components

Start with:

```text
evaluation =
    material
  + piece-square tables
  + bishop pair
  + pawn structure
  + mobility
  + king safety
  + rook activity
  + passed pawns
```

Implement incrementally.

## Material

Recommended values:

```python
PAWN   = 100
KNIGHT = 320
BISHOP = 330
ROOK   = 500
QUEEN  = 900
```

Do not assign a normal material value to the king.

## Piece-Square Tables

Create a 64-square table for each piece.

Use different king tables for:

- middlegame
- endgame

Mirror tables for Black rather than maintaining duplicated values.

## Bishop Pair

Example:

```text
+30 to +50 centipawns
```

for possessing both bishops.

Keep configurable.

## Pawn Structure

Possible rules:

- doubled pawn penalty
- isolated pawn penalty
- backward pawn penalty
- passed pawn bonus
- connected passed pawn bonus

Scale passed pawn bonuses by rank.

## Mobility

Reward pieces for having useful legal or pseudo-legal moves.

Be careful not to count mobility in a way that makes evaluation too slow.

## King Safety

Possible signals:

- pawn shield around king
- open files near king
- enemy attackers near king
- king exposed before endgame

## Rook Activity

Possible bonuses:

- rook on open file
- rook on semi-open file
- rook on seventh rank
- connected rooks

## Game Phase

Blend middlegame and endgame evaluation.

Example:

```text
phase = weighted amount of non-pawn material remaining

score =
    middlegame_score * phase
    +
    endgame_score * (1 - phase)
```

## Perspective

Prefer calculating:

```text
white_score - black_score
```

then convert it to side-to-move perspective when returning to Negamax.

## Performance

Evaluation is called extremely often.

Avoid:

- copying boards
- expensive global calculations
- repeated conversion to strings/FEN
- unnecessary object allocation

## Tests

Create positions where:

- extra queen gives roughly +900
- extra pawn gives roughly +100
- centralized knight scores above corner knight
- passed pawn receives bonus
- doubled pawns receive penalty
- bishop pair receives bonus
- mirrored equivalent positions have opposite scores

## Done Criteria

- Fully deterministic.
- No neural network or learned weights.
- Score is symmetric between White and Black.
- Material dominates small positional bonuses.
- Evaluation is fast enough to run at leaf nodes.
