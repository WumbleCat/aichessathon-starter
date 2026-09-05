# 07 — Transposition Table + Zobrist Hashing

## Goal

Cache previously searched positions so repeated positions do not need to be fully searched again.

## Zobrist Hash

Maintain a 64-bit position key based on random numbers assigned to:

- piece type
- piece color
- square
- side to move
- castling rights
- en-passant state

If your chess library already exposes a robust transposition key, prefer using it.

## Important

A position key must distinguish positions relevant to legal move state.

Do not hash only piece locations.

## TT Entry

Recommended structure:

```python
class TTEntry:
    key
    depth
    score
    flag
    best_move
```

Flags:

```text
EXACT
LOWERBOUND
UPPERBOUND
```

## Probe Logic

Before searching a node:

1. Probe the table.
2. Confirm stored key matches.
3. If stored depth >= requested depth:
   - EXACT: return score
   - LOWERBOUND: raise alpha
   - UPPERBOUND: lower beta
4. If alpha >= beta, return the cached bound.
5. Use cached best move for move ordering even if score cannot be reused.

## Store Logic

Remember the original alpha.

After search:

```text
best_score <= original_alpha -> UPPERBOUND
best_score >= beta           -> LOWERBOUND
otherwise                    -> EXACT
```

Store:

- key
- depth
- score
- best move
- flag

## Mate Score Normalization

If mate scores include ply distance, normalize when storing/probing so the same position reached at a different ply has consistent mate meaning.

## Replacement Strategy

Simple starting approach:

- replace if slot empty
- replace if new entry has equal or greater depth

Later:

- age/generation
- depth-preferred replacement

## Memory

Use a fixed-size table rather than an unlimited Python dictionary if memory control matters.

For first implementation, a dictionary is acceptable.

## Tests

- identical position returns same hash
- move + undo restores previous hash
- different side to move changes hash
- castling-right change changes hash
- en-passant state changes hash
- TT reduces nodes on repeated searches
- cached result matches uncached result

## Done Criteria

- Position keys are correct.
- TT provides valid bound reuse.
- TT move integrates with move ordering.
- Search remains deterministic and legal.
